### yes24 관련 요청하는 프롬프트 
* @C:\webscraping_gemini\GEMINI.md 의 내용을 읽고 요청사항 대로 처리해 주세요

* @C:\webscraping_gemini\yes24\agent_scraping.md 해당 내용을 바탕으로 데이터를 수집하는 코드를 작성해 주세요.

* @C:\webscraping_gemini\yes24\data\yes24_ai.csv 파일을 불러와서 데이터 분석과 시각화를 진행하는 마크다운 파일을 생성해 주세요. 파일명은 agent_eda_yes24.md 로 합니다. 마크다운 문서만 작성하고 .py는 아직 작성하지 마세요.

* @C:\webscraping_gemini\yes24\agent_eda_yes24.md 문서의 내용에 따라 데이터 분석을 수행해 주세요.

### starbucks 관련 요청하는 프롬프트 
* @C:\webscraping_gemini\starbucks_stores\agent_prd_scraping.md 내용을 분석하여 데이터를 수집하고 csv 파일을 생성해 주세요

* @C:\webscraping_gemini\starbucks_stores\analysis_report.md 보고서에 아래의 내용이 반영되도록 eda_analyzer.py 수정해줘. 매장의 특징을 코드 대신 설명으로 보고서에 나타나도록 수정해야 함 예를 들어 T30 대신에 사이렌오더 출력해야 함 \
starbucks_service_codes = {
    "T30": "사이렌오더",
    "T20": "현금없는매장",
    "T17": "주차가능",
    "T05": "카드충전",
    "T65": "공기청정기",
    "T16": "친환경매장",
    "T08": "무선인터넷",
    "T32": "전자영수증",
    "T56": "디카페인",
    "T52": "에코매장",
    "T34": "딜리버스",
    "T21": "기프트카드",
    "T43": "현금영수증",
    "P80": "블론드",
    "P90": "티바나",
    "Z9999": "일반서비스"
}

```
uv pip uninstall altair
```
```
uv pip install altair==4.2.2
```
```
uv pip install standard-imghdr
```
```
streamlit --version
Streamlit, version 1.19.0
```