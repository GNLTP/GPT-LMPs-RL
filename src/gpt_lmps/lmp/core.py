from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gpt_lmps.llm import LLMBackend
from gpt_lmps.lmp.safety import SafeExecutor, strip_markdown_fences
from gpt_lmps.types import ExecutionTrace


@dataclass
class LanguageModelProgram:
    """One code-producing stage with optional execution history."""

    name: str
    system_prompt: str
    backend: LLMBackend
    executor: SafeExecutor
    symbols: dict[str, Any]
    return_name: str | None = None
    maintain_session: bool = False
    history: list[str] = field(default_factory=list)

    def clear_history(self) -> None:
        self.history.clear()

    def __call__(
        self,
        query: str,
        *,
        context: str,
        trace: ExecutionTrace,
        extra_symbols: dict[str, Any] | None = None,
    ) -> Any:
        use_context = context
        if self.maintain_session and self.history:
            use_context = f"{context}\nExecution history:\n" + "\n".join(self.history)
        source = self.backend.generate(
            name=self.name,
            system=self.system_prompt,
            query=query,
            context=use_context,
        )
        source = strip_markdown_fences(source)
        trace.add_event(self.name, query, source)

        symbols = dict(self.symbols)
        if extra_symbols:
            symbols.update(extra_symbols)
        locals_dict = self.executor.execute(source, symbols)
        if self.maintain_session:
            self.history.append(source)
        if self.return_name is None:
            return None
        if self.return_name not in locals_dict:
            raise RuntimeError(f"LMP {self.name!r} did not assign {self.return_name!r}")
        return locals_dict[self.return_name]
