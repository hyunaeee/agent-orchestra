"""Optional LangSmith tracing.

Two layers:
1. LangGraph itself is built on langchain-core runnables, so setting
   LANGSMITH_TRACING=true + LANGSMITH_API_KEY traces every graph superstep
   automatically — no code needed.
2. `traceable_safe` adds explicit spans around our custom LLM calls (which are
   plain functions, invisible to layer 1). It degrades to a no-op when
   langsmith is not installed or no key is set, so offline runs stay clean.
"""

from __future__ import annotations

import os


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


def traceable_safe(name: str, run_type: str = "llm"):
    if tracing_enabled():
        try:
            from langsmith import traceable

            return traceable(name=name, run_type=run_type)
        except ImportError:
            pass

    def _noop(fn):
        return fn

    return _noop
