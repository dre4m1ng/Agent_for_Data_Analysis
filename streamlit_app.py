"""Streamlit front-end for the A.I.D.A. autonomous data analysis agent."""
from __future__ import annotations

codex/implement-a.i.d.a.-agent-features-in-github-m21dss
from typing import Dict, List, Tuple

from typing import Dict, List
main

import pandas as pd
import streamlit as st

codex/implement-a.i.d.a.-agent-features-in-github-m21dss
from aida.agent import (
    AGENT_ROLES,
    ROLE_ANALYST,
    ROLE_CLEANER,
    ROLE_PLANNER,
    ROLE_REPORTER,
    AgentCore,
    AgentEvent,
    AgentRunArtifacts,
    StageLog,
    available_models,
    initialize_llm,
)

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


  codex/implement-a.i.d.a.-agent-features-in-github-m21dss
def _catalogue_models() -> List[Tuple[str, str, str]]:
    """Return a list of ``(label, name, mode)`` tuples for select boxes."""

    models = available_models()
    catalogue: List[Tuple[str, str, str]] = []
    for mode in ("LLM", "SLM"):
        names = sorted(name for name, model in models.items() if model.mode == mode)
        for name in names:
            catalogue.append((f"{mode} - {name}", name, mode))
    # append any remaining models with unexpected modes to ensure visibility
    for name, model in models.items():
        if all(entry[1] != name for entry in catalogue):
            catalogue.append((f"{model.mode} - {name}", name, model.mode))
    return catalogue


ROLE_DEFAULT_MODE = {
    ROLE_PLANNER: "LLM",
    ROLE_CLEANER: "SLM",
    ROLE_ANALYST: "LLM",
    ROLE_REPORTER: "LLM",
}

def _group_models() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {"LLM": [], "SLM": []}
    for name, model in available_models().items():
        grouped.setdefault(model.mode, []).append(name)
    return grouped
 main


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

    codex/implement-a.i.d.a.-agent-features-in-github-m21dss
st.sidebar.header("역할별 분석 엔진")
catalogue = _catalogue_models()
model_lookup = available_models()

role_model_names: Dict[str, str] = {}

if not catalogue:
    st.sidebar.warning("등록된 모델이 없습니다. models.py를 확인하세요.")
else:
    st.sidebar.write(
        "각 단계별로 사용할 LLM/SLM을 선택하세요. 역할에 따라 서로 다른 모델을 지정할 수 있습니다."
    )
    for role_key, role_label in AGENT_ROLES.items():
        desired_mode = ROLE_DEFAULT_MODE.get(role_key)
        default_index = 0
        for idx, (_, name, mode) in enumerate(catalogue):
            if desired_mode and mode == desired_mode:
                default_index = idx
                break
        selected_label = st.sidebar.selectbox(
            role_label,
            options=[entry[0] for entry in catalogue],
            index=default_index,
            key=f"role-select-{role_key}",
        )
        selected_model_name = selected_label.split(" - ", 1)[1]
        role_model_names[role_key] = selected_model_name
        model_meta = model_lookup.get(selected_model_name)
        if model_meta and model_meta.metadata.get("description"):
            st.sidebar.caption(f"{role_label}: {model_meta.metadata['description']}")

run_button = st.button(
    "A.I.D.A. 분석 시작",
    disabled=df is None or not role_model_names or any(name not in model_lookup for name in role_model_names.values()),
)

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
codex/implement-a.i.d.a.-agent-features-in-github-m21dss
summary_placeholder = st.container()




def _render_logs(logs: List[str]) -> None:
    formatted = "\n".join(logs[-200:])  # keep the latest 200 entries for readability
    codex/implement-a.i.d.a.-agent-features-in-github-m21dss
    if formatted:
        log_placeholder.markdown(f"```\n{formatted}\n```")
    else:
        log_placeholder.markdown("로그가 아직 없습니다.")


def _format_stage_log(log: StageLog) -> Dict[str, str]:
    role_label = AGENT_ROLES.get(log.role, "시스템") if log.role else "시스템"
    return {"단계": log.stage, "역할": role_label, "메시지": log.message}

    log_placeholder.markdown(f"```\n{formatted}\n```")



def _on_event(event: AgentEvent) -> None:
    st.session_state.logs.append(f"[{event.stage}] {event.message}")
    _render_logs(st.session_state.logs)


if run_button and df is not None:
    st.session_state.logs = []
    _render_logs(st.session_state.logs)

    with st.spinner("Agent가 분석을 수행 중입니다..."):
      codex/implement-a.i.d.a.-agent-features-in-github-m21dss
        role_configs = {role: initialize_llm(name) for role, name in role_model_names.items()}
        agent = AgentCore(role_configs)

        model_config = initialize_llm(selected_model_name)
        agent = AgentCore(model_config)

        artifacts: AgentRunArtifacts = agent.analyse(df, callback=_on_event)

    st.success("분석이 완료되었습니다. 아래 결과를 확인하세요.")

    report_placeholder.markdown(artifacts.final_report)
    codex/implement-a.i.d.a.-agent-features-in-github-m21dss
    st.download_button(
        "보고서 Markdown 다운로드",
        artifacts.final_report,
        file_name="aida_report.md",
        mime="text/markdown",
    )



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

        codex/implement-a.i.d.a.-agent-features-in-github-m21dss
    with summary_placeholder:
        st.subheader("역할별 엔진 구성")
        for role_key, model in artifacts.role_models.items():
            st.markdown(f"- **{AGENT_ROLES[role_key]}**: {model.name} ({model.mode})")

        st.subheader("외부 검색 결과")
        if artifacts.search_results:
            for item in artifacts.search_results:
                title = item.get("title") or "제목 없음"
                snippet = item.get("snippet") or ""
                link = item.get("link") or ""
                if link:
                    st.markdown(f"- **[{title}]({link})**: {snippet}")
                else:
                    st.markdown(f"- **{title}**: {snippet}")
        else:
            st.info("검색 결과가 없습니다.")

        if artifacts.stage_logs:
            st.subheader("단계별 로그 요약")
            stage_log_rows = [_format_stage_log(log) for log in artifacts.stage_logs]
            st.dataframe(pd.DataFrame(stage_log_rows))
else:
    st.info("CSV 파일을 업로드하고 각 역할에 사용할 엔진을 선택한 뒤 'A.I.D.A. 분석 시작' 버튼을 클릭하세요.")

else:
    st.info("CSV 파일을 업로드하고 분석 엔진을 선택한 뒤 'A.I.D.A. 분석 시작' 버튼을 클릭하세요.")

