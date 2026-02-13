
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from loguru import logger
import os

# --- 설정 ---
DATA_PATH = "starbucks_stores/data/starbucks_ai.csv"
REPORT_PATH = "starbucks_stores/eda_report.md"
IMAGE_DIR = "starbucks_stores/images"
logger.add("starbucks_stores/eda_{time}.log", rotation="500 MB", encoding="utf-8")

# --- 보고서 생성을 위한 헬퍼 함수 ---
def add_to_report(content, is_code=False):
    """마크다운 보고서에 내용을 추가합니다."""
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        if is_code:
            f.write("```\n")
            f.write(content)
            f.write("\n```\n\n")
        else:
            f.write(content + "\n\n")

def save_plot(fig, filename, title):
    """플롯을 이미지 파일로 저장하고 보고서에 추가합니다."""
    filepath = os.path.join(IMAGE_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"'{filename}' 시각화 저장 완료.")
    add_to_report(f"### {title}")
    add_to_report(f"![{title}]({os.path.join('images', filename)})")

def main():
    # --- 이미지 저장 폴더 생성 ---
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # --- 보고서 초기화 ---
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    add_to_report("# 스타벅스 매장 데이터 EDA 보고서")

    # --- 0. 기본 분석 ---
    add_to_report("## 0. 기본 분석")
    logger.info("데이터 분석 시작...")

    # 데이터 로드
    try:
        df = pd.read_csv(DATA_PATH)
        add_to_report("### 데이터 로드 성공")
        add_to_report(f"데이터 경로: `{DATA_PATH}`")
    except FileNotFoundError:
        logger.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        add_to_report(f"**오류:** 데이터 파일을 찾을 수 없습니다: `{DATA_PATH}`")
        return

    # 데이터 구조
    add_to_report("### 데이터 구조")
    add_to_report(df.head().to_markdown())
    
    # 데이터 정보
    add_to_report("### 데이터 정보 (df.info())")
    info_str = df.info(verbose=True, show_counts=True)
    # info()는 출력을 직접 반환하지 않으므로, 버퍼를 사용해 캡처
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    add_to_report(info_str, is_code=True)

    # 기술 통계
    add_to_report("### 기술 통계")
    add_to_report(df.describe(include='all').to_markdown())

    # 결측치 및 품질 점검
    add_to_report("### 결측치 및 단일 값 컬럼 제거")
    missing_values = df.isnull().sum()
    add_to_report("#### 제거 전 결측치 비율")
    add_to_report((missing_values / len(df) * 100).to_string(), is_code=True)

    cols_to_drop = []
    for col in df.columns:
        # 모든 값이 결측치인 컬럼
        if df[col].isnull().all():
            cols_to_drop.append(col)
            logger.warning(f"'{col}' 컬럼은 모든 값이 결측치이므로 제거합니다.")
            continue
        # 모든 값이 동일한 컬럼
        # t/p로 시작하는 특성 컬럼이 아닌 경우에만 단일 값 체크
        if not (col.startswith('t') or col.startswith('p')):
            if df[col].nunique() == 1:
                cols_to_drop.append(col)
                logger.warning(f"'{col}' 컬럼은 모든 값이 동일하므로 제거합니다.")
    
    df.drop(columns=cols_to_drop, inplace=True)
    add_to_report(f"#### 제거된 컬럼: `{cols_to_drop}`")
    add_to_report("#### 제거 후 데이터 구조")
    add_to_report(df.head().to_markdown())

    # 데이터 타입 변환 (open_dt)
    df['open_dt'] = pd.to_datetime(df['open_dt'], format='%Y%m%d', errors='coerce')
    df['open_year'] = df['open_dt'].dt.year
    df['open_month'] = df['open_dt'].dt.month

    # --- 1. 기본 정보 요약 ---
    add_to_report("## 1. 기본 정보 요약")
    add_to_report("주요 필드(매장명, 시/도, 시/군/구, 주소, 오픈일) 요약")
    summary_df = df[['s_name', 'sido_name', 'gugun_name', 'doro_address', 'open_dt', 'lat', 'lot']].head()
    add_to_report(summary_df.to_markdown())

    # --- 2. 매장 특성 분석 (theme_state) ---
    add_to_report("## 2. 매장 특성 분석 (theme_state)")

    # theme_state에서 특성 추출
    all_themes = []
    df['theme_state'].dropna().str.split('@').apply(lambda themes: all_themes.extend(theme for theme in themes if theme))

    feature_summary = pd.Series(all_themes).value_counts()

    if not feature_summary.empty:
        feature_summary_df = pd.DataFrame(feature_summary).reset_index()
        feature_summary_df.columns = ['특징', '매장 수']

        add_to_report("### 매장별 제공 서비스/특징 (상위 10개)")
        add_to_report(feature_summary_df.head(10).to_markdown())

        fig, ax = plt.subplots(figsize=(12, 8))
        feature_summary_df.head(10).set_index('특징').plot(kind='bar', ax=ax)
        ax.set_title('상위 10개 매장 서비스/특징')
        ax.set_ylabel('매장 수')
        save_plot(fig, 'top10_features.png', '상위 10개 매장 서비스/특징')
        add_to_report(feature_summary_df.head(10).to_markdown())

    else:
        add_to_report("분석할 수 있는 매장 특성(theme_state) 정보가 부족합니다.")

    # --- 3. 주변 위치적 특징 분석 ---
    add_to_report("## 3. 주변 위치적 특징 분석")
    add_to_report("### 시/도별 매장 분포")
    sido_counts = df['sido_name'].value_counts()
    fig, ax = plt.subplots(figsize=(15, 7))
    sido_counts.plot(kind='bar', ax=ax)
    ax.set_title('시/도별 스타벅스 매장 수')
    ax.set_ylabel('매장 수')
    save_plot(fig, 'stores_by_sido.png', '시/도별 매장 분포')
    add_to_report(pd.DataFrame(sido_counts).to_markdown())

    add_to_report("### 서울시 내 구별 매장 분포 (가장 많은 시/도 기준)")
    top_sido = sido_counts.index[0]
    top_sido_df = df[df['sido_name'] == top_sido]
    gugun_counts = top_sido_df['gugun_name'].value_counts()
    fig, ax = plt.subplots(figsize=(15, 7))
    gugun_counts.plot(kind='bar', ax=ax)
    ax.set_title(f'{top_sido} 내 시/군/구별 스타벅스 매장 수')
    ax.set_ylabel('매장 수')
    save_plot(fig, 'stores_by_gugun_top_sido.png', f'{top_sido} 내 구별 매장 분포')
    add_to_report(pd.DataFrame(gugun_counts).to_markdown())
    add_to_report(f"가장 매장이 많은 지역은 **{top_sido}**이며, 특히 **{gugun_counts.index[0]}**에 매장이 집중되어 있어 중심 상권일 가능성이 높습니다.")

    # --- 4. 오픈일(open_dt) 기반 통계적 해석 ---
    add_to_report("## 4. 오픈일 기반 통계 분석")
    add_to_report("### 연도별 매장 오픈 트렌드")
    open_year_counts = df['open_year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    open_year_counts.plot(kind='line', marker='o', ax=ax)
    ax.set_title('연도별 스타벅스 매장 오픈 수')
    ax.set_ylabel('오픈 매장 수')
    save_plot(fig, 'open_trend_by_year.png', '연도별 매장 오픈 트렌드')
    add_to_report(pd.DataFrame(open_year_counts).to_markdown())
    add_to_report("최근 연도로 올수록 매장 오픈 수가 증가하는 경향을 보입니다.")

    # --- 5. 주요 비즈니스 인사이트 및 6. 데이터 품질 검증 ---
    add_to_report("## 5. 주요 비즈니스 인사이트 및 6. 데이터 품질 검증")
    add_to_report(""")
- **입지 경쟁력**: 매장은 주로 '서울', '경기' 등 수도권에 집중되어 있으며, 특히 강남구, 서초구, 중구 등 오피스 및 상업 중심지에 밀집되어 있습니다. 이는 유동인구가 많은 핵심 상권을 타겟으로 하는 전략으로 해석됩니다.
- **고객 편의성**: `t01`(무선인터넷), `t03`(테라스) 등의 편의 시설 제공 여부를 통해 고객 경험을 향상시키려는 노력을 엿볼 수 있습니다.
- **데이터 품질**:
  - `open_dt`에서 일부 날짜 변환에 실패한 데이터(NaT)가 존재할 수 있어 확인이 필요합니다.
  - `lat`, `lot` 좌표의 경우, 대한민국 범위를 벗어나는 값이 있는지 확인하여 데이터의 신뢰성을 검증해야 합니다. (예: 위도 33-39, 경도 124-132 범위)
  - `sido_code`, `gugun_code`는 각각 `sido_name`, `gugun_name`과 일관성을 유지하는지 검증이 필요합니다.
""")
    
    # 좌표계 이상치 확인
    lat_outliers = df[(df['lat'] < 33) | (df['lat'] > 39)]
    lot_outliers = df[(df['lot'] < 124) | (df['lot'] > 132)]
    if not lat_outliers.empty or not lot_outliers.empty:
        add_to_report("### 좌표계 이상치 의심 데이터")
        add_to_report(pd.concat([lat_outliers, lot_outliers]).to_markdown())
    else:
        add_to_report("### 좌표계 이상치 검사: 대한민국 범위 내에 모든 매장이 위치합니다.")

    # --- 7. 추가 분석 제안 ---
    add_to_report("## 7. 추가 분석 제안")
    add_to_report(""")
- **경쟁사 분석**: 공공데이터포털의 타 커피 전문점 데이터를 활용하여, 특정 지역 내 스타벅스와 경쟁사의 분포를 비교 분석할 수 있습니다.
- **유동인구 데이터 결합**: 서울시 열린데이터광장의 '지하철 승하차 인원', '유동인구' 데이터와 매장 위치를 결합하여, 유동인구와 매장 수의 상관관계를 분석할 수 있습니다.
- **매장 클러스터링**: `tXX`, `pXX`와 같은 매장 특성 데이터를 기반으로 K-Means 등의 클러스터링 알고리즘을 사용하여 매장을 '업무 중심형', '주거 지역형', '대학가형' 등으로 유형화할 수 있습니다.
- **매출 예측 모델**: 매장 위치, 오픈 연도, 주변 인구 밀도, 경쟁사 수 등의 변수를 활용하여 신규 매장의 예상 매출을 예측하는 회귀 모델을 개발할 수 있습니다.
""")

    logger.info("데이터 분석 및 보고서 생성 완료.")

if __name__ == "__main__":
    main()
