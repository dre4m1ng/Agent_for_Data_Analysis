"""Streamlit front-end for the A.I.D.A. autonomous data analysis agent."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from aida.agent import AgentCore, AgentEvent, AgentRunArtifacts, available_models, initialize_llm

st.set_page_config(page_title="A.I.D.A. Agent", layout="wide")

st.title("🧠 A.I.D.A. (AI-driven Data Analyst) Agent")
st.markdown(
    """
    일반 CSV 데이터를 업로드하고 분석 엔진을 선택하면 A.I.D.A.가 자율적으로 문제를 정의하고
    데이터를 정제한 뒤, 탐색적 분석과 인사이트 도출까지 수행합니다.
    """
)


@st.cache_data(show_spinner=False)
def _load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def _group_models() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {"LLM": [], "SLM": []}
    for name, model in available_models().items():
        grouped.setdefault(model.mode, []).append(name)
    return grouped


if "logs" not in st.session_state:
    st.session_state.logs = []


uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    try:
        df = _load_csv(uploaded_file)
        st.success(f"데이터 로드 완료: {df.shape[0]}행 {df.shape[1]}열")
        st.dataframe(df.head())
    except Exception as exc:  # pragma: no cover - UI guard
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {exc}")
        df = None
else:
    df = None

st.sidebar.header("분석 엔진 선택")
model_groups = _group_models()
model_lookup = available_models()

model_options: List[str] = []
for mode in ("LLM", "SLM"):
    for name in sorted(model_groups.get(mode, [])):
        model_options.append(f"{mode} - {name}")

default_model = next(iter(model_lookup.keys())) if model_lookup else ""

if model_options:
    selected_label = st.sidebar.selectbox(
        "모델을 선택하세요",
        options=model_options,
        index=0,
        help="고성능 LLM과 경량 SLM 옵션을 자유롭게 전환할 수 있습니다.",
    )
    selected_model_name = selected_label.split(" - ", 1)[1]
else:
    st.sidebar.warning("등록된 모델이 없습니다. models.py를 확인하세요.")
    selected_model_name = default_model

if selected_model_name and selected_model_name in model_lookup:
    selected_model = model_lookup[selected_model_name]
    st.sidebar.markdown(f"**선택된 모델:** {selected_model.name} ({selected_model.mode})")
    description = selected_model.metadata.get("description")
    if description:
        st.sidebar.caption(description)

run_button = st.button("A.I.D.A. 분석 시작", disabled=df is None or not selected_model_name)

log_placeholder = st.empty()
report_placeholder = st.empty()
charts_placeholder = st.container()


def _render_logs(logs: List[str]) -> None:
    formatted = "\n".join(logs[-200:])  # keep the latest 200 entries for readability
    log_placeholder.markdown(f"```\n{formatted}\n```")


def _on_event(event: AgentEvent) -> None:
    st.session_state.logs.append(f"[{event.stage}] {event.message}")
    _render_logs(st.session_state.logs)


if run_button and df is not None:
    st.session_state.logs = []
    _render_logs(st.session_state.logs)

    with st.spinner("Agent가 분석을 수행 중입니다..."):
        model_config = initialize_llm(selected_model_name)
        agent = AgentCore(model_config)
        artifacts: AgentRunArtifacts = agent.analyse(df, callback=_on_event)

    st.success("분석이 완료되었습니다. 아래 결과를 확인하세요.")

    report_placeholder.markdown(artifacts.final_report)

    with charts_placeholder:
        st.subheader("EDA 시각화")
        for chart in artifacts.charts:
            if chart.error:
                st.warning(f"차트 생성 오류: {chart.error}")
                continue
            st.image(chart.image_path, caption=chart.image_path)

    st.subheader("주요 인사이트")
    for insight in artifacts.insights:
        st.markdown(f"- **{insight.title}**: {insight.detail}")

else:
    st.info("CSV 파일을 업로드하고 분석 엔진을 선택한 뒤 'A.I.D.A. 분석 시작' 버튼을 클릭하세요.")
