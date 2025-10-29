"""Core reasoning engine orchestrating the A.I.D.A. workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..tools import data_diagnostics
from ..tools.chart_generator import ChartResult, generate_chart
from ..tools.data_diagnostics import DataDiagnosticsReport
from ..tools.web_searcher import WebSearcher
from .models import AgentModel


@dataclass
class AgentEvent:
    """Streaming event emitted by the agent during execution."""

    stage: str
    message: str


@dataclass
class Insight:
    """Structured representation of an analytical insight."""

    title: str
    detail: str


@dataclass
class AgentRunArtifacts:
    diagnostics: DataDiagnosticsReport
    cleaned_dataframe: pd.DataFrame
    charts: List[ChartResult]
    insights: List[Insight]
    final_report: str
    search_results: List[Dict[str, str]] = field(default_factory=list)


EventCallback = Callable[[AgentEvent], None]


class AgentCore:
    """Deterministic implementation of the multi-stage analysis pipeline."""

    def __init__(self, model: AgentModel) -> None:
        self.model = model
        self.searcher = WebSearcher()

    def _emit(self, callback: Optional[EventCallback], stage: str, message: str) -> None:
        if callback:
            callback(AgentEvent(stage=stage, message=message))

    def analyse(self, df: pd.DataFrame, callback: Optional[EventCallback] = None) -> AgentRunArtifacts:
        self._emit(callback, "initialise", f"선택된 엔진: {self.model.name} ({self.model.mode})")

        diagnostics = data_diagnostics.analyze_dataframe(df)
        self._emit(callback, "diagnostics", json.dumps(diagnostics.to_dict(), indent=2, ensure_ascii=False))

        target_column = self._infer_target_column(df)
        problem_statement = self._build_problem_statement(df, target_column)
        self._emit(callback, "problem", problem_statement)

        search_results = self._perform_search(df, target_column, callback)

        cleaned_df, cleaning_logs = self._clean_dataframe(df, callback)

        charts = self._run_eda(cleaned_df, target_column, callback)

        insights = self._generate_insights(cleaned_df, target_column, charts, callback)

        final_report = self._compose_final_report(
            diagnostics=diagnostics,
            problem_statement=problem_statement,
            cleaning_logs=cleaning_logs,
            charts=charts,
            insights=insights,
            search_results=search_results,
        )

        self._emit(callback, "final", "최종 보고서 생성 완료")
        return AgentRunArtifacts(
            diagnostics=diagnostics,
            cleaned_dataframe=cleaned_df,
            charts=charts,
            insights=insights,
            final_report=final_report,
            search_results=search_results,
        )

    # ---------------------------------------------------------------------
    # 1단계: 문제 정의
    def _infer_target_column(self, df: pd.DataFrame) -> str:
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if not str(col).lower().endswith("id")]
        if not numeric_columns:
            return df.columns[0]

        heuristics = ["price", "cost", "revenue", "amount", "score", "target"]
        for column in numeric_columns:
            lower = column.lower()
            if any(token in lower for token in heuristics):
                return column
        return numeric_columns[0]

    def _build_problem_statement(self, df: pd.DataFrame, target_column: str) -> str:
        dtype = df[target_column].dtype
        if np.issubdtype(dtype, np.number):
            problem_type = "회귀"
            objective = f"`{target_column}` 값을 예측"
        else:
            problem_type = "분류"
            objective = f"`{target_column}` 범주를 분류"
        return (
            f"데이터의 핵심 문제를 {problem_type} 문제로 정의합니다. 주요 목표는 {objective}하는 것입니다. "
            "초기 분석 계획으로는 데이터 품질 진단, 결측치 처리, 핵심 변수 탐색 및 상관관계 분석을 포함합니다."
        )

    def _perform_search(
        self, df: pd.DataFrame, target_column: str, callback: Optional[EventCallback]
    ) -> List[Dict[str, str]]:
        keywords = [target_column]
        for column in df.columns:
            if column != target_column and len(keywords) < 3:
                keywords.append(column)
        query = " ".join(keywords[:3])
        self._emit(callback, "search", f"외부 검색 실행: {query}")

        results = self.searcher.search(query)
        formatted = [
            {"title": result.title, "link": result.link, "snippet": result.snippet}
            for result in results
        ]
        if formatted:
            self._emit(callback, "search", json.dumps(formatted, ensure_ascii=False, indent=2))
        return formatted

    # ---------------------------------------------------------------------
    # 2단계: 정제 및 전처리
    def _clean_dataframe(self, df: pd.DataFrame, callback: Optional[EventCallback]):
        cleaned_df = df.copy()
        logs: List[str] = []

        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_df.select_dtypes(exclude=[np.number]).columns

        for column in numeric_columns:
            if cleaned_df[column].isna().any():
                mean_value = cleaned_df[column].mean()
                cleaned_df[column].fillna(mean_value, inplace=True)
                logs.append(f"{column}: 결측치 평균값({mean_value:.3f})으로 대체")

        for column in categorical_columns:
            if cleaned_df[column].isna().any():
                mode_value = cleaned_df[column].mode().iloc[0]
                cleaned_df[column].fillna(mode_value, inplace=True)
                logs.append(f"{column}: 결측치 최빈값({mode_value})으로 대체")

        if not logs:
            logs.append("결측치가 발견되지 않았습니다.")

        for entry in logs:
            self._emit(callback, "cleaning", entry)
        return cleaned_df, logs

    # ---------------------------------------------------------------------
    # 3단계: EDA 및 시각화
    def _run_eda(
        self, df: pd.DataFrame, target_column: str, callback: Optional[EventCallback]
    ) -> List[ChartResult]:
        charts: List[ChartResult] = []

        hist_code = (
            "plt.figure(figsize=(8, 4))\n"
            f"df['{target_column}'].dropna().hist(bins=30, color='#2563eb')\n"
            f"plt.title('Distribution of {target_column}')\n"
            "plt.xlabel('Value')\n"
            "plt.ylabel('Frequency')\n"
            "plt.tight_layout()\n"
            "plt.savefig(output_path)\n"
            "plt.close()\n"
        )
        self._emit(callback, "eda", f"{target_column} 분포 히스토그램 생성")
        charts.append(generate_chart(hist_code, df, filename=f"{target_column}_distribution.png"))

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) >= 2:
            corr_code = (
                "import seaborn as sns\n"
                "plt.figure(figsize=(6, 5))\n"
                "corr = df.corr(numeric_only=True)\n"
                "sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2f', square=True)\n"
                "plt.title('Numeric Feature Correlation')\n"
                "plt.tight_layout()\n"
                "plt.savefig(output_path)\n"
                "plt.close()\n"
            )
            self._emit(callback, "eda", "상관관계 히트맵 생성")
            charts.append(generate_chart(corr_code, df, filename="correlation_heatmap.png"))

        return charts

    # ---------------------------------------------------------------------
    # 4단계: 인사이트 도출
    def _generate_insights(
        self,
        df: pd.DataFrame,
        target_column: str,
        charts: List[ChartResult],
        callback: Optional[EventCallback],
    ) -> List[Insight]:
        insights: List[Insight] = []

        if target_column in df.columns and np.issubdtype(df[target_column].dtype, np.number):
            target_mean = df[target_column].mean()
            target_std = df[target_column].std()
            insights.append(
                Insight(
                    title=f"{target_column} 평균/변동성",
                    detail=f"평균 {target_mean:.2f}, 표준편차 {target_std:.2f}로 분산 수준을 파악할 수 있습니다.",
                )
            )

            corr = df.corr(numeric_only=True)[target_column].drop(target_column).sort_values(ascending=False)
            if not corr.empty:
                top_feature, top_value = corr.index[0], corr.iloc[0]
                insights.append(
                    Insight(
                        title=f"{target_column}와의 최고 상관 변수",
                        detail=f"`{top_feature}` 변수가 {top_value:.2f}의 상관계수로 가장 큰 선형 관계를 보입니다.",
                    )
                )

            if target_std > 0:
                coef_var = target_std / target_mean if target_mean else np.nan
                insights.append(
                    Insight(
                        title="변동계수 해석",
                        detail=(
                            "변동계수(CV)가 {:.2f}로 측정되어 데이터의 상대적 변동성이 {}수준임을 의미합니다.".format(
                                coef_var, "높은" if coef_var > 0.5 else "낮은"
                            )
                        ),
                    )
                )
        else:
            value_counts = df[target_column].value_counts().head(3)
            formatted = ", ".join(f"{idx}: {count}" for idx, count in value_counts.items())
            insights.append(
                Insight(
                    title=f"{target_column} 빈도 상위 3개",
                    detail=f"상위 범주 분포 - {formatted}",
                )
            )

        for insight in insights:
            self._emit(callback, "insight", f"{insight.title}: {insight.detail}")
        return insights

    # ---------------------------------------------------------------------
    # 5단계: 보고서
    def _compose_final_report(
        self,
        diagnostics: DataDiagnosticsReport,
        problem_statement: str,
        cleaning_logs: Iterable[str],
        charts: List[ChartResult],
        insights: List[Insight],
        search_results: List[Dict[str, str]],
    ) -> str:
        lines: List[str] = ["# A.I.D.A. 분석 보고서", ""]

        lines.extend(["## 1. 문제 정의", "", problem_statement, ""])

        lines.append("## 2. 데이터 진단 요약")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

        lines.append("## 3. 외부 검색 결과")
        lines.append("")
        if search_results:
            for result in search_results:
                title = result.get("title") or "제목 없음"
                link = result.get("link") or ""
                snippet = result.get("snippet") or ""
                lines.append(f"- **{title}**: {snippet} ({link})")
        else:
            lines.append("- 검색 결과가 없습니다.")
        lines.append("")

        lines.append("## 4. 전처리/정제 로그")
        lines.append("")
        for log in cleaning_logs:
            lines.append(f"- {log}")
        lines.append("")

        lines.append("## 5. 핵심 EDA 시각화")
        lines.append("")
        for chart in charts:
            lines.append(f"- {chart.image_path}")
        lines.append("")

        lines.append("## 6. 주요 인사이트")
        lines.append("")
        for insight in insights:
            lines.append(f"- **{insight.title}**: {insight.detail}")
        lines.append("")

        lines.append("## 7. 다음 단계 제안")
        lines.append("")
        lines.append(
            "- 추가적인 모델링 실험을 위해 훈련/검증 데이터 분할 및 하이퍼파라미터 최적화를 진행하세요."
        )
        lines.append("- 비즈니스 맥락과 연결된 지표를 정의하고 결과를 검증하세요.")

        return "\n".join(lines)
