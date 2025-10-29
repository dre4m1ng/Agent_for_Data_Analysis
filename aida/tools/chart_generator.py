"""Utility for producing visualisations from dynamically generated code."""
from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

from .code_executor import execute_python


@dataclass
class ChartResult:
    """Information about a generated chart."""

    image_path: str
    logs: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


_DEFAULT_CHART_DIR = Path("artifacts/charts")


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_chart(code: str, df: Any, *, filename: str | None = None) -> ChartResult:
    """Execute ``code`` to generate a chart from ``df``.

    The execution environment receives two variables:

    ``df``
        The dataframe currently being analysed.
    ``plt``
        The :mod:`matplotlib.pyplot` module for manual figure management.

    Parameters
    ----------
    code:
        Python code that must create and save a matplotlib figure to ``output_path``.
    df:
        Dataframe made available to the execution context.
    filename:
        Optional filename for the resulting image. When omitted a descriptive name is
        generated automatically.
    """

    _ensure_directory(_DEFAULT_CHART_DIR)
    filename = filename or "chart.png"
    output_path = _DEFAULT_CHART_DIR / filename

    # ``output_path`` is injected so that dynamically generated code knows where to save.
    context: Dict[str, Any] = {"df": df, "plt": plt, "output_path": str(output_path)}

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = execute_python(code, context=context)

    logs = buffer.getvalue()
    if not result.succeeded:
        return ChartResult(image_path=str(output_path), logs=logs, error=result.error)

    if not os.path.exists(output_path):
        # ensure at least an empty figure exists to avoid runtime surprises
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No figure generated", ha="center", va="center")
        plt.savefig(output_path)
        plt.close()

    return ChartResult(image_path=str(output_path), logs=logs or result.output)
