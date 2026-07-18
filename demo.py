"""Run one task through the agent team (parallel fan-out) and print every hop."""

from orchestra import Orchestra

TASK = "사내 문의 응대 에이전트 도입 제안서 작성 — 포함: 도입 배경, 비용 추정, 보안 검토, 단계별 일정"

if __name__ == "__main__":
    team = Orchestra()
    print(f"backend: {team.llm.name} · langsmith tracing: {'ON' if team.tracing else 'off'}\n")
    result = team.run_parallel(TASK)  # researchers fan out concurrently

    print("=== PLAN (supervisor) ===\n" + result["plan"] + "\n")
    print(f"=== NOTES ({len(result['notes'])} researchers, parallel superstep) ===")
    for n in result["notes"]:
        print(n)
    print("\n=== FINAL DRAFT (writer) ===\n" + result["draft"] + "\n")
    print("=== REVIEW (critic) ===\n" + result["review"] + "\n")
    print(f"rounds: {result['rounds']} · verdict: {result['verdict']}")
    print(f"procedural memory size: {len(team.procedural)} lesson(s)")
    print("\n=== A2A MESSAGE LOG ===")
    for e in team.shared.dump():
        print(f"  {e['sender']:>24} → {e['recipient']:<8} [{e['intent']}] {e['content'][:52].replace(chr(10), ' ')}")
