from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """One generated-program event in the hierarchical LMP pipeline."""

    stage: str
    query: str
    code: str


@dataclass
class ExecutionTrace:
    """Generated code plus the concrete primitive calls it produced."""

    instruction: str
    events: list[TraceEvent] = field(default_factory=list)
    primitive_calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def add_event(self, stage: str, query: str, code: str) -> None:
        self.events.append(TraceEvent(stage=stage, query=query, code=code))

    def add_primitive(self, name: str, *args: Any) -> None:
        self.primitive_calls.append((name, args))
