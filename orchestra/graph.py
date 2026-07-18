"""LangGraph orchestration with parallel researcher fan-out,
a conditional revise loop, and both memory layers wired in.

                       ┌─▶ researcher_requirements ─┐
    supervisor ──fan───┼─▶ researcher_risks ────────┼──▶ writer ──▶ critic ──▶ END
                out    └─▶ researcher_structure ────┘     ▲           │
                        (same superstep — concurrent      └─ REVISE ──┘
                         under `arun`/ainvoke)            (max ORCHESTRA_MAX_ROUNDS)
"""

from __future__ import annotations

import asyncio
import operator
import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from .llm import get_llm
from .memory import ProceduralMemory, SharedMemory
from .tracing import traceable_safe, tracing_enabled

MAX_ROUNDS = int(os.environ.get("ORCHESTRA_MAX_ROUNDS", "2"))

LENSES = {
    "requirements": "과제가 명시적으로 요구하는 항목",
    "risks": "누락되기 쉬운 리스크·엣지케이스",
    "structure": "결과물이 갖춰야 할 구조와 형식",
}


class OrchestraState(TypedDict, total=False):
    task: str
    plan: str
    notes: Annotated[list[str], operator.add]  # fan-out reducer: researchers append
    draft: str
    review: str
    verdict: str
    rounds: int


class Orchestra:
    def __init__(self, memory_path: str = "procedural_memory.json") -> None:
        self.llm = get_llm()
        self.shared = SharedMemory()
        self.procedural = ProceduralMemory(memory_path)
        self._complete = traceable_safe("llm.complete")(self.llm.complete)
        self.app = self._build()

    # ---- agents -----------------------------------------------------------
    def _supervisor(self, state: OrchestraState) -> OrchestraState:
        plan = self._complete(
            "supervisor",
            "당신은 에이전트 팀의 오케스트레이터다. 과제를 단계로 분해하라.",
            f"과제: {state['task']}",
        )
        self.shared.send("supervisor", "*", "plan", plan)
        return {"plan": plan, "rounds": 0}

    def _make_researcher(self, lens: str):
        def researcher(state: OrchestraState) -> OrchestraState:
            notes = self._complete(
                f"researcher_{lens}",
                f"당신은 조사 에이전트다. 관점: {LENSES[lens]}. 그 관점에서만 노트를 작성하라.",
                f"과제: {state['task']}\n\n[공유 컨텍스트]\n{self.shared.context_for(lens)}",
            )
            self.shared.send(f"researcher_{lens}", "writer", "notes", notes)
            return {"notes": [notes]}

        researcher.__name__ = f"researcher_{lens}"
        return researcher

    def _writer(self, state: OrchestraState) -> OrchestraState:
        merged = "\n".join(state["notes"])
        lessons = self.procedural.relevant(state["task"])
        lesson_block = ("\n[절차 메모리]\n" + "\n".join(lessons) + "\n") if lessons else ""
        feedback = f"\n[비평 피드백]\n{state['review']}\n" if state.get("review") else ""
        draft = self._complete(
            "writer",
            "당신은 작성 담당 에이전트다. 세 리서처의 노트를 모두 반영해 결과물을 작성하라.",
            f"과제: {state['task']}\n{merged}\n{lesson_block}{feedback}",
        )
        self.shared.send("writer", "critic", "draft", draft)
        return {"draft": draft}

    def _critic(self, state: OrchestraState) -> OrchestraState:
        review = self._complete(
            "critic",
            "당신은 검수 에이전트다. 초안이 과제의 요구 항목을 모두 다루면 첫 줄에 APPROVE, "
            "아니면 첫 줄에 REVISE를 쓰고 누락을 지적하라. 승인 시 다음 실행을 위한 교훈을 남겨라.",
            f"과제: {state['task']}\n\n[초안]\n{state['draft']}",
        )
        verdict = "approve" if review.upper().startswith("APPROVE") else "revise"
        rounds = state.get("rounds", 0) + 1
        self.shared.send("critic", "writer", "review", review)
        if verdict == "approve" and "교훈:" in review:
            lesson = review.split("교훈:", 1)[1].strip()
            self.procedural.add(state["task"], lesson)
            self.shared.send("critic", "*", "lesson", lesson)
        return {"review": review, "verdict": verdict, "rounds": rounds}

    # ---- graph ------------------------------------------------------------
    def _route(self, state: OrchestraState) -> str:
        if state["verdict"] == "approve" or state["rounds"] >= MAX_ROUNDS:
            return "done"
        return "revise"

    def _build(self):
        g = StateGraph(OrchestraState)
        g.add_node("supervisor", self._supervisor)
        for lens in LENSES:
            g.add_node(f"researcher_{lens}", self._make_researcher(lens))
        g.add_node("writer", self._writer)
        g.add_node("critic", self._critic)

        g.set_entry_point("supervisor")
        for lens in LENSES:  # fan-out: three researchers share one superstep
            g.add_edge("supervisor", f"researcher_{lens}")
            g.add_edge(f"researcher_{lens}", "writer")  # implicit join barrier
        g.add_edge("writer", "critic")
        g.add_conditional_edges("critic", self._route, {"revise": "writer", "done": END})
        return g.compile()

    def run(self, task: str) -> OrchestraState:
        return self.app.invoke({"task": task})

    async def arun(self, task: str) -> OrchestraState:
        """Async run — researcher fan-out executes concurrently."""
        return await self.app.ainvoke({"task": task})

    def run_parallel(self, task: str) -> OrchestraState:
        return asyncio.run(self.arun(task))

    @property
    def tracing(self) -> bool:
        return tracing_enabled()
