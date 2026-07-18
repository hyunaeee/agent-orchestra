"""Two memory layers for the agent team.

- SharedMemory   : run-scoped scratchpad every agent reads/writes (공유 메모리).
                   Also keeps the A2A message log and a rolling context window.
- ProceduralMemory: JSON-persisted "lessons" that survive across runs
                   (절차적 메모리). The critic writes a lesson when it approves;
                   the writer gets relevant lessons injected on later runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Envelope:
    """A2A-style message envelope passed between agents."""

    sender: str
    recipient: str
    intent: str  # plan | notes | draft | review | lesson
    content: str
    ts: float = field(default_factory=time.time)


class SharedMemory:
    def __init__(self, window: int = 12) -> None:
        self._log: list[Envelope] = []
        self._window = window
        self.scratch: dict[str, str] = {}

    def send(self, sender: str, recipient: str, intent: str, content: str) -> Envelope:
        env = Envelope(sender, recipient, intent, content)
        self._log.append(env)
        return env

    def context_for(self, agent: str) -> str:
        """Rolling window of messages addressed to (or broadcast past) an agent."""
        recent = self._log[-self._window :]
        return "\n".join(
            f"[{e.sender}→{e.recipient}|{e.intent}] {e.content[:400]}" for e in recent
        )

    def dump(self) -> list[dict]:
        return [asdict(e) for e in self._log]


class ProceduralMemory:
    def __init__(self, path: str | Path = "procedural_memory.json") -> None:
        self._path = Path(path)
        self._lessons: list[dict] = []
        if self._path.exists():
            try:
                self._lessons = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._lessons = []

    def add(self, task: str, lesson: str) -> None:
        self._lessons.append({"task": task, "lesson": lesson, "ts": time.time()})
        self._path.write_text(
            json.dumps(self._lessons, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def relevant(self, task: str, k: int = 3) -> list[str]:
        """Naive keyword-overlap retrieval — deliberately dependency-free."""
        words = set(task.split())
        scored = sorted(
            self._lessons,
            key=lambda l: len(words & set(l["task"].split())),
            reverse=True,
        )
        return [l["lesson"] for l in scored[:k]]

    def __len__(self) -> int:
        return len(self._lessons)
