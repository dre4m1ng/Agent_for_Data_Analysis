# A.I.D.A. (AI-driven Data Analyst) Agent

A.I.D.A.는 교체 가능한 LLM/SLM 분석 엔진을 기반으로 CSV 데이터를 자율 분석하는 Streamlit 애플리케이션입니다. 업로드된 데이터를 진단하고, 현실 세계 문제를 정의하며, 전처리-EDA-인사이트 도출을 자동화합니다.

## 주요 기능
- **데이터 업로드 & 미리보기**: CSV를 업로드하면 즉시 크기와 상위 5개 행을 확인할 수 있습니다.
- **역할별 엔진 구성**: 사이드바에서 문제 정의, 데이터 정제, EDA, 보고서 작성 등 각 단계를 담당할 LLM/SLM을 개별적으로 선택해 협업형 에이전트를 구성할 수 있습니다.
- **자율 분석 파이프라인**: 문제 정의 → 전처리 → EDA/시각화 → 인사이트 도출 → 최종 보고서를 순차적으로 수행합니다.
- **실시간 로그 스트리밍**: Agent 이벤트 로그가 실시간으로 표시되어 진행 상황을 추적할 수 있습니다.
- **시각화 & 보고서 출력**: 생성된 차트와 마크다운 보고서를 바로 확인할 수 있습니다.

## 프로젝트 구조
```
├── aida/
│   ├── agent/
│   │   ├── core.py          # 자율 분석 파이프라인 구현
│   │   └── models.py        # LLM/SLM 초기화 및 레지스트리
│   └── tools/
│       ├── code_executor.py # 안전한 코드 실행 도구
│       ├── chart_generator.py
│       ├── data_diagnostics.py
│       └── web_searcher.py
├── requirements.txt
└── streamlit_app.py         # Streamlit UI 엔트리 포인트
```

## 빠른 시작
1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
2. Streamlit 앱 실행
   ```bash
   streamlit run streamlit_app.py
   ```
3. 브라우저에서 표시되는 UI를 통해 CSV 파일을 업로드하고 각 역할에 사용할 엔진을 선택한 뒤 **A.I.D.A. 분석 시작** 버튼을 클릭하세요.

## 테스트 URL 안내

현재 저장소에는 배포된 데모 인스턴스가 제공되지 않습니다. 아래 절차에 따라 직접 실행한 뒤 로컬 또는 사내 네트워크에서 접속 URL을 확인해 주세요.

1. 터미널에서 `streamlit run streamlit_app.py`를 실행하면 기본적으로 `http://localhost:8501`에서 앱이 구동됩니다.
2. 동일 네트워크의 다른 장치에서 접속하려면 `streamlit run streamlit_app.py --server.address 0.0.0.0`으로 실행하고, 실행 로그에 표시되는 URL과 호스트 머신의 IP를 조합해 접속합니다.
3. 외부 공유가 필요할 경우 아래 "무료 배포 가이드"를 참고하여 호스팅 플랫폼에 배포하면 공개 URL을 받을 수 있습니다.

## 무료 배포 가이드

GitHub 저장소만으로는 Streamlit 앱을 직접 호스팅할 수 없습니다. GitHub Pages는 정적 사이트 전용이므로 Python 기반의 Streamlit 앱을 실행하지 못합니다. 아래 두 플랫폼은 무료 플랜을 제공하며, 이 저장소를 가장 간단하게 배포할 수 있는 방법입니다.

### 1. Streamlit Community Cloud (권장)
1. [Streamlit Community Cloud](https://streamlit.io/cloud)에 가입하고 GitHub 계정을 연결합니다.
2. "New app"을 선택하고 이 저장소(`Agent_for_Data_Analysis`)와 브랜치를 지정한 뒤, 메인 스크립트로 `streamlit_app.py`를 입력합니다.
3. 필요한 경우 "Advanced settings" → "Secrets"에 `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` 등 환경 변수를 추가합니다.
4. "Deploy"를 클릭하면 몇 분 내에 앱이 빌드되며, 배포가 완료되면 `https://<your-app-name>.streamlit.app` 형태의 공개 URL이 제공됩니다.

### 2. Hugging Face Spaces
1. [Hugging Face](https://huggingface.co/) 계정으로 로그인 후 "New Space"를 생성합니다.
2. Space 타입으로 **Streamlit**을 선택하고, 이 저장소를 복제하거나 새 Space의 리포지토리에 코드를 업로드합니다.
3. `requirements.txt`를 자동으로 읽어 의존성이 설치되며, `streamlit_app.py`가 기본 엔트리 포인트로 실행됩니다. 필요 시 "Files and versions"에서 `secrets.toml`을 생성해 API 키를 입력합니다.
4. Space가 빌드되면 `https://huggingface.co/spaces/<username>/<space-name>` 형태의 공개 URL이 즉시 제공됩니다.

두 플랫폼 모두 무료 티어에서 소규모 테스트에 충분하며, 빌드/실행 시간 제한이 있으므로 대규모 분석 시에는 유료 플랜 또는 자체 인프라를 고려하세요.

## 외부 검색 API 설정 (선택 사항)
Google Custom Search API를 사용하려면 다음 환경 변수를 설정하세요.
```bash
export GOOGLE_API_KEY="<YOUR-API-KEY>"
export GOOGLE_CSE_ID="<YOUR-SEARCH-ENGINE-ID>"
```
키가 설정되어 있지 않으면 검색 단계에서 안내 메시지가 출력됩니다.

## 모델 확장
`aida/agent/models.py`의 `_AVAILABLE_MODELS` 사전에 새로운 모델 구성을 추가하거나, `initialize_llm` 함수를 확장해 API 기반 호출을 연동할 수 있습니다. 추가된 모델은 Streamlit UI의 역할 선택에 자동으로 반영되므로, 단계별로 최적의 엔진 조합을 구성할 수 있습니다.

### 역할별 에이전트 개요

| 역할 | 담당 업무 | 기본 추천 모드 |
| --- | --- | --- |
| 문제 정의 & 계획 | 데이터 구조 파악, 외부 검색, 분석 목표 설정 | LLM |
| 데이터 정제 | 결측치 처리, 기본 전처리 코드 실행 | SLM |
| EDA & 시각화 | 핵심 지표 시각화, 상관관계 분석 | LLM |
| 인사이트 & 보고서 | 주요 통찰 정리, 최종 보고서 생성 | LLM |

## 라이선스
이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
