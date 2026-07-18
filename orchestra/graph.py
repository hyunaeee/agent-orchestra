"""LangGraph orchestration: supervisor → researcher → writer → critic,
with a conditional revise loop (critic → writer) and both memory layers wired in.

    supervisor ──▶ researcher ──▶ writer ──▶ critic ──▶ END
                                    ▲           │
                                    └─ REVISE ──┘   (max ORCHESTRA_MAX_ROUNDS)
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

from .llm import get_llm
from .memory import ProceduralMemory, SharedMemory

MAX_ROUNDS = int(os.environ.get("ORCHESTRA_MAX_ROUNDS", "2"))


class OrchestraState(TypedDict, total=False):
    task: str
    plan: str
    notes: str
    draft: str
    review: str
    verdict: str
    rounds: int


class Orchestra:
    def __init__(self, memory_path: str = "procedural_memory.json") -> None:
        self.llm = get_llm()
        self.shared = SharedMemory()
        self.procedural = ProceduralMemory(memory_path)
        self.app = self._build()

    # ---- agents -----------------------------------------------------------
    def _supervisor(self, state: OrchestraState) -> OrchestraState:
        plan = self.llm.complete(
            "supervisor",
            "당신은 에이전트 팀의 오케스트레이터다. 과제를 단계로 분해하라.",
            f"과제: {state['task']}",
        )
        self.shared.send("supervisor", "*", "plan", plan)
        return {"plan": plan, "rounds": 0}

    def _researcher(self, state: OrchestraState) -> OrchestraState:
        notes = self.llm.complete(
            "researcher",
            "당신은 조사 담당 에이전트다. 과제에서 다뤄야 할 항목을 노트로 정리하라.",
            f"과제: {state['task']}\n\n[공유 컨텍스트]\n{self.shared.context_for('researcher')}",
        )
        self.shared.send("researcher", "writer", "notes", notes)
        return {"notes": notes}

    def _writer(self, state: OrchestraState) -> OrchestraState:
        lessons = self.procedural.relevant(state["task"])
        lesson_block = ("\n[절차 메모리]\n" + "\n".join(lessons) + "\n") if lessons else ""
        feedback = f"\n[비평 피드백]\n{state['review']}\n" if state.get("review") else ""
        draft = self.llm.complete(
            "writer",
            "당신은 작성 담당 에이전트다. 조사 노트의 모든 항목을 반영해 결과물을 작성하라.",
            f"과제: {state['task']}\n{state['notes']}\n{lesson_block}{feedback}",
        )
        self.shared.send("writer", "critic", "draft", draft)
        return {"draft": draft}

    def _critic(self, state: OrchestraState) -> OrchestraState:
        review = self.llm.complete(
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
        g.add_node("researcher", self._researcher)
        g.add_node("writer", self._writer)
        g.add_node("critic", self._critic)
        g.set_entry_point("supervisor")
        g.add_edge("supervisor", "researcher")
        g.add_edge("researcher", "writer")
        g.add_edge("writer", "critic")
        g.add_conditional_edges("critic", self._route, {"revise": "writer", "done": END})
        return g.compile()

    def run(self, task: str) -> OrchestraState:
        return self.app.invoke({"task": task})
