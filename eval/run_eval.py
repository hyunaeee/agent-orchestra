"""Evaluation harness for the agent team.

Offline (no key): a rule-based judge checks rubric coverage in the final draft.
With ANTHROPIC_API_KEY: an LLM-judge additionally scores usefulness 1-5.

Reported per task: pass/fail, revision rounds, latency, approx tokens.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestra import Orchestra  # noqa: E402

EVALSET = Path(__file__).with_name("evalset.jsonl")


def rule_judge(draft: str, criteria: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in criteria if c not in draft]
    return (not missing, missing)


def llm_judge(llm, task: str, draft: str) -> str:
    return llm.complete(
        "judge",
        "1(무용)~5(바로 사용 가능)로 결과물의 실용성을 평가하고 근거를 한 줄로 써라.",
        f"과제: {task}\n\n[결과물]\n{draft}",
    )


def main() -> None:
    rows = [json.loads(l) for l in EVALSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    mem_path = Path(__file__).with_name("eval_memory.json")
    if mem_path.exists():
        mem_path.unlink()  # fresh procedural memory per eval run

    results = []
    for row in rows:
        team = Orchestra(memory_path=mem_path)  # shared lessons accumulate across tasks
        t0 = time.perf_counter()
        out = team.run(row["task"])
        dt = time.perf_counter() - t0
        ok, missing = rule_judge(out["draft"], row["criteria"])
        tokens = (len(out["draft"]) + len(out["notes"]) + len(out["plan"])) // 4
        results.append({
            "id": row["id"], "pass": ok, "missing": missing,
            "rounds": out["rounds"], "sec": round(dt, 3), "approx_tokens": tokens,
        })
        judge_note = ""
        if os.environ.get("ANTHROPIC_API_KEY"):
            judge_note = " · judge: " + llm_judge(team.llm, row["task"], out["draft"]).splitlines()[0]
        print(f"[{'PASS' if ok else 'FAIL'}] {row['id']:<18} rounds={out['rounds']} "
              f"{dt:.2f}s ~{tokens}tok{judge_note}"
              + (f"  missing={missing}" if missing else ""))

    n = len(results)
    passed = sum(r["pass"] for r in results)
    print("-" * 56)
    print(f"pass rate: {passed}/{n} ({passed / n:.0%}) · "
          f"avg latency: {sum(r['sec'] for r in results) / n:.2f}s · "
          f"avg rounds: {sum(r['rounds'] for r in results) / n:.1f}")
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved → {out_path.name}")


if __name__ == "__main__":
    main()
