"""A lightweight sandbox for executing dynamically generated Python code."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict


@dataclass
class ExecutionResult:
    """Result of executing user-provided Python code."""

    output: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def _build_safe_globals(extra_globals: Dict[str, Any] | None = None) -> Dict[str, Any]:
    allowed_builtins = MappingProxyType(
        {
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "print": print,
                "enumerate": enumerate,
                "list": list,
                "dict": dict,
                "set": set,
                "float": float,
                "int": int,
                "str": str,
                "zip": zip,
            }
        }
    )

    sandbox_globals: Dict[str, Any] = {}
    sandbox_globals.update(allowed_builtins)
    if extra_globals:
        sandbox_globals.update(extra_globals)
    return sandbox_globals


def execute_python(code: str, context: Dict[str, Any] | None = None) -> ExecutionResult:
    """Execute ``code`` with optional context variables.

    Parameters
    ----------
    code:
        Arbitrary Python code to execute. Multi-line strings are supported.
    context:
        Additional variables that should be available to the executed code.

    Returns
    -------
    ExecutionResult
        Object describing the execution outcome and textual output.
    """

    sandbox_globals = _build_safe_globals(context)
    sandbox_locals: Dict[str, Any] = {}

    try:
        exec(compile(code, "<code_executor>", "exec"), sandbox_globals, sandbox_locals)
    except Exception as exc:  # pragma: no cover - reformat message only
        return ExecutionResult(output="", error=str(exc))

    captured_output = sandbox_locals.get("__output__", "")
    if not isinstance(captured_output, str):
        captured_output = str(captured_output)

    return ExecutionResult(output=captured_output or "Execution completed successfully.")
