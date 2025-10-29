"""Utilities for generating diagnostic metadata about uploaded datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ColumnDiagnostics:
    """Metadata about a single column in a dataframe."""

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_ratio: float
    distinct_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataDiagnosticsReport:
    """A structured diagnostics report for a dataframe."""

    shape: List[int]
    columns: List[ColumnDiagnostics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "columns": [column.to_dict() for column in self.columns],
        }


def _describe_column(series: pd.Series) -> ColumnDiagnostics:
    total = len(series)
    non_null = series.count()
    null_count = total - non_null
    null_ratio = float(null_count) / float(total) if total else 0.0
    distinct_count = series.nunique(dropna=True)

    return ColumnDiagnostics(
        name=str(series.name),
        dtype=str(series.dtype),
        non_null_count=int(non_null),
        null_count=int(null_count),
        null_ratio=float(round(null_ratio, 5)),
        distinct_count=int(distinct_count),
    )


def analyze_dataframe(df: pd.DataFrame) -> DataDiagnosticsReport:
    """Generate diagnostics metadata for ``df``.

    Parameters
    ----------
    df:
        A :class:`pandas.DataFrame` to analyse.

    Returns
    -------
    DataDiagnosticsReport
        A structured representation of the dataframe's metadata.
    """

    columns = [_describe_column(df[column]) for column in df.columns]
    return DataDiagnosticsReport(shape=[df.shape[0], df.shape[1]], columns=columns)
