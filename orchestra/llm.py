"""LLM backend — Anthropic Claude when ANTHROPIC_API_KEY is set, otherwise a
deterministic offline mock so the whole graph (and the eval harness) runs
without any key. The mock is intentionally simple: it exists to exercise the
orchestration/memory/eval plumbing, not to imitate model quality."""

from __future__ import annotations

import os
import re


class MockLLM:
    """Deterministic role-conditioned completions for offline runs."""

    name = "mock (offline)"

    def complete(self, role: str, system: str, prompt: str) -> str:
        if role == "supervisor":
            return (
                "1) researcher가 과제의 핵심 항목을 조사한다\n"
                "2) writer가 조사 노트를 바탕으로 결과물을 작성한다\n"
                "3) critic이 루브릭 충족 여부를 검수하고 미흡하면 재작성을 요청한다"
            )
        if role == "researcher":
            items = _keywords(prompt)
            lines = [f"- {kw}: 과제 요구사항 — 결과물에 반드시 다뤄야 함" for kw in items]
            return "\n".join(lines) if lines else "- 과제 일반 요건 정리"
        if role == "writer":
            notes = re.findall(r"^- ([^:]+):", prompt, flags=re.M)
            body = "\n".join(f"### {n.strip()}\n{n.strip()}에 대한 구체적 계획과 근거를 정리했다." for n in notes)
            lesson = ""
            m = re.search(r"\[절차 메모리\]\n(.+?)(?:\n\n|\Z)", prompt, flags=re.S)
            if m:
                lesson = "\n\n> 이전 실행에서 배운 것 반영: " + m.group(1).strip().splitlines()[0]
            return f"## 결과 보고\n{body}{lesson}"
        if role == "critic":
            required = _keywords(prompt)
            draft_m = re.search(r"\[초안\]\n(.+)", prompt, flags=re.S)
            draft = draft_m.group(1) if draft_m else ""
            missing = [kw for kw in required if kw not in draft]
            if missing:
                return "REVISE\n누락 항목: " + ", ".join(missing)
            return "APPROVE\n루브릭 전 항목 충족. 교훈: 요구 항목을 섹션 헤더로 명시하면 누락이 줄어든다."
        return "OK"


def _keywords(prompt: str) -> list[str]:
    m = re.search(r"포함(?:할 항목)?\s*[:：]\s*(.+)", prompt)
    if not m:
        return []
    return [w.strip() for w in re.split(r"[,·]", m.group(1).splitlines()[0]) if w.strip()]


class AnthropicLLM:
    name = "claude"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, role: str, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=os.environ.get("ORCHESTRA_MODEL", "claude-sonnet-5"),
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


def get_llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception:
            pass
    return MockLLM()
