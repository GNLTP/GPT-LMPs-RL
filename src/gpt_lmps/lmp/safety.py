from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any


class UnsafeCodeError(ValueError):
    """Raised when generated code is outside the public robot API sandbox."""


def strip_markdown_fences(source: str) -> str:
    """Normalize a common LLM failure mode without silently editing program logic."""

    text = source.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


class _PolicyValidator(ast.NodeVisitor):
    _ALLOWED_NODES = {
        ast.Module,
        ast.Expr,
        ast.Call,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Assign,
        ast.AnnAssign,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.keyword,
        ast.If,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.In,
        ast.NotIn,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.For,
        ast.Subscript,
        ast.Slice,
        ast.JoinedStr,
        ast.FormattedValue,
    }

    def __init__(self, allowed_calls: set[str]) -> None:
        self.allowed_calls = allowed_calls

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self._ALLOWED_NODES:
            raise UnsafeCodeError(
                f"Generated code contains forbidden syntax: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise UnsafeCodeError("Dunder names are forbidden")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_calls:
            name = ast.unparse(node.func)
            raise UnsafeCodeError(f"Call is not in the robot API allowlist: {name}")
        self.generic_visit(node)


class SafeExecutor:
    """AST-validated execution for small generated robot policy programs."""

    def execute(self, source: str, symbols: Mapping[str, Any]) -> dict[str, Any]:
        source = strip_markdown_fences(source)
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise UnsafeCodeError(f"Generated code is not valid Python: {exc.msg}") from exc

        allowed_calls = {name for name, value in symbols.items() if callable(value)}
        _PolicyValidator(allowed_calls).visit(tree)

        globals_dict = {"__builtins__": {}}
        globals_dict.update(symbols)
        locals_dict: dict[str, Any] = {}
        exec(compile(tree, filename="<generated-policy>", mode="exec"), globals_dict, locals_dict)
        return locals_dict
