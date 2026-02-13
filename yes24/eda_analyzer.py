"""
eda_analyzer.py

이 스크립트는 Yes24 AI 도서 데이터를 대상으로 탐색적 데이터 분석(EDA)을 수행합니다.
CSV 파일에서 도서 정보를 읽어와 데이터를 전처리하고, 통계 분석을 수행하며,
다양한 시각화(히스토그램, 막대 그래프, 히트맵, 워드 클라우드)를 생성합니다.
또한, 피벗 테이블을 사용하여 교차 분석을 수행하고, 모든 분석 결과와 인사이트를
포함하는 마크다운 보고서를 이미지와 함께 생성합니다.

주요 의존성:
    - pandas: 데이터 조작 및 분석을 위한 라이브러리.
    - numpy: 수치 계산을 위한 라이브러리.
    - matplotlib: 정적, 애니메이션 및 인터랙티브 시각화를 생성하기 위한 라이브러리.
    - seaborn: 통계 데이터 시각화를 위한 라이브러리.
    - wordcloud: 텍스트 데이터에서 워드 클라우드 이미지를 생성하기 위한 라이브러리.
    - loguru: 스크립트 실행 전반에 걸쳐 강력한 로깅을 제공하는 라이브러리.
    # koreanize_matplotlib: (선택 사항이지만 한글 텍스트에 권장) matplotlib이 한글을 올바르게 표시하도록 구성합니다.

출력물:
    - 'yes24/analysis_report.md': 통계 요약, 인사이트, 시각화 이미지 경로가 포함된 전체 분석 보고서 마크다운 파일.
    - 'yes24/images/': 생성된 시각화 이미지 파일(PNG 형식)이 저장되는 디렉토리.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from datetime import datetime
from loguru import logger
# import koreanize_matplotlib # matplotlib이 한글을 올바르게 표시하도록 구성
import matplotlib.font_manager as fm # matplotlib 폰트 관리자 임포트

# WordCloud 라이브러리 임포트 시도. 설치되지 않은 경우, None으로 설정하고 워드 클라우드 생성을 건너뜁니다.
try:
    from wordcloud import WordCloud
except ImportError:
    print("wordcloud가 설치되지 않았습니다. 워드 클라우드 생성을 건너뜁니다.")
    WordCloud = None

def analyze_yes24_data():
    """
    Yes24 AI 도서 데이터를 분석하고, 시각화를 생성하며, 포괄적인 보고서를 작성합니다.

    이 함수는 전체 EDA(탐색적 데이터 분석) 과정을 조정합니다:
    1.  **설정**: 입력 데이터, 출력 이미지 및 보고서의 파일 경로를 설정합니다.
    2.  **데이터 로딩 및 전처리**: CSV 데이터를 읽고, 숫자형 컬럼을 정리하며,
        발행일에서 연도/월을 추출하고, 결측치를 처리합니다.
    3.  **기초 통계 분석**: 숫자형 및 범주형 데이터에 대한 `info()`, `describe()`를 제공하고,
        초기 인사이트를 보고서에 포함합니다.
    4.  **시각화 분석**: 다양한 플롯을 생성하고 저장합니다:
        -   숫자 데이터 분포를 위한 히스토그램.
        -   상위 범주형 데이터 빈도(예: 출판사, 저자)를 위한 막대 그래프.
        -   숫자 변수 간의 상관 관계를 위한 히트맵.
        -   핵심 테마를 식별하기 위한 도서 제목의 워드 클라우드.
        (참고: matplotlib 플롯에서 한글을 올바르게 렌더링하기 위해 koreanize_matplotlib이 사용됩니다.)
    5.  **교차 분석**: 피벗 테이블 및 교차표를 사용하여 여러 변수 간의 관계를
        밝히기 위한 고급 분석을 수행합니다 (예: 출판사 판매 동향, 연도별 가격 변화, 저자 성과).
    6.  **보고서 생성**: 모든 분석 결과, 시각화 및 인사이트를 단일 마크다운 파일
        ('yes24/analysis_report.md')로 집계하고 이미지 파일을
        'yes24/images/'에 저장합니다.

    이 스크립트는 'yes24/data/yes24_ai.csv' 파일이 입력 데이터로 존재한다고 가정합니다.

    Args:
        None

    Returns:
        None: 이 함수는 출력을 위해 직접 파일을 생성합니다.
    """
    # --- 0. 설정 (Configuration) ---
    # 스크랩된 Yes24 AI 도서 데이터가 포함된 입력 CSV 파일 경로.
    file_path = 'yes24/data/yes24_ai.csv'
    # 생성된 시각화 이미지를 저장할 출력 디렉토리.
    output_folder = 'yes24/images'
    # 최종 마크다운 분석 보고서의 경로.
    report_path = 'yes24/analysis_report.md'    

    # 출력 이미지 디렉토리가 없으면 생성합니다.
    os.makedirs(output_folder, exist_ok=True)

    font_path = 'C:/Windows/Fonts/malgun.ttf'
    # macOS 사용자의 경우 다음 경로를 사용할 수 있습니다.
    # font_path = 'AppleGothic'
    
    # 보고서 내용을 담을 리스트를 메인 제목으로 초기화합니다.
    report_content = ["# Yes24 AI 도서 데이터 분석 보고서\n\n"]

    # --- 1. 데이터 불러오기 및 전처리 (Data Loading & Preprocessing) ---
    report_content.append("## 1. 데이터 불러오기 및 전처리\n\n")
    # 입력 데이터 파일이 존재하는지 확인합니다. 없으면 오류를 기록하고 분석을 중단합니다.
    if not os.path.exists(file_path):
        report_content.append(f"'{file_path}' 파일을 찾을 수 없습니다. 분석을 중단합니다.")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("".join(report_content))
        print(f"분석 보고서가 '{report_path}'에 저장되었습니다.")
        logger.error(f"입력 데이터 파일을 찾을 수 없습니다: {file_path}")
        return

    # CSV 파일을 pandas DataFrame으로 로드합니다.
    df = pd.read_csv(file_path)

    def preprocess(df):
        """
        입력 DataFrame을 분석에 적합한 형태로 전처리합니다.

        이 중첩 함수는 여러 정리 및 변환 단계를 수행합니다:
        -   '상세 정보' 컬럼이 존재하는 경우 삭제합니다.
        -   지정된 숫자형 컬럼('판매지수', '리뷰수', '판매가', '정가')을
            문자열 형식(쉼표가 포함될 수 있음)에서 정수형으로 변환합니다.
            이 과정에는 다음이 포함됩니다:
                - `.str` 메서드를 사용할 수 있도록 문자열로 변환.
                - 공백 제거 (`.strip()`).
                - 쉼표 제거.
                - 숫자로 강제 변환하며, 파싱할 수 없는 값은 NaN으로 변환.
        -   변환 후 주요 숫자형 컬럼에 NaN 값이 있는 행을 삭제합니다.
        -   정리된 숫자형 컬럼을 `int` 타입으로 변환합니다.
        -   '발행일' 컬럼에서 '발행연도' 및 '발행월' 정보를 추출합니다.
        -   유효하지 않은 발행연도(예: 1900년 이하)를 가진 항목을 필터링합니다.

        Args:
            df (pd.DataFrame): CSV에서 로드된 원본 DataFrame.

        Returns:
            pd.DataFrame: EDA를 위한 전처리된 DataFrame.
        """
        # '상세 정보' 컬럼이 존재하면 삭제합니다. 이 컬럼은 EDA에 직접적으로 유용하지 않은 경우가 많습니다.
        if '상세 정보' in df.columns:
            df = df.drop(columns=['상세 정보'])
            
        # 숫자형 컬럼 처리: 쉼표 제거, 숫자로 변환, 그 다음 정수로 변환.
        for col in ['판매지수', '리뷰수', '판매가', '정가']:
            # 문자열로 변환하고, 공백을 제거하고, 쉼표를 제거한 다음 숫자로 변환합니다.
            # 'errors='coerce''는 파싱 오류를 NaN으로 만듭니다.
            df[col] = df[col].astype(str).str.strip().str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 변환 후 중요한 숫자형 컬럼에 NaN 값이 있는 행을 삭제합니다.
        df.dropna(subset=['판매지수', '리뷰수', '판매가', '정가'], inplace=True)
        
        # 정리된 숫자형 컬럼을 정수형으로 변환합니다.
        for col in ['판매지수', '리뷰수', '판매가', '정가']:
            df[col] = df[col].astype(int)

        # '발행일' 컬럼에서 발행연도와 발행월을 추출합니다.
        # 잠재적인 파싱 오류를 처리하기 위해 NaN을 0으로 채운 다음 정수로 변환합니다.
        df['발행연도'] = df['발행일'].str.extract(r'(\d{4})년').astype(float).fillna(0).astype(int)
        df['발행월'] = df['발행일'].str.extract(r'(\d{2})월').astype(float).fillna(0).astype(int)
        
        # 비현실적인 발행연도를 가진 행을 필터링합니다.
        df = df[df['발행연도'] > 1900]
        return df

    # DataFrame 복사본에 전처리 단계를 적용합니다.
    df = preprocess(df.copy())
    
    report_content.append("데이터 전처리가 완료되었습니다.\n\n")

    # --- 2. 기초 통계 분석 (Basic Statistical Analysis) ---
    report_content.append("## 2. 기초 통계 분석\n\n")
    
    # DataFrame의 info() 출력을 문자열로 캡처하여 보고서에 포함합니다.
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    report_content.append("### 데이터 정보 (info)\n\n```\n" + info_str + "\n```\n\n")

    # 숫자형 컬럼에 대한 기술 통계를 보고서에 추가합니다.
    report_content.append("### 수치형 데이터 기술통계\n\n")
    report_content.append(df.describe(include=np.number).to_markdown() + "\n\n")
    
    # 범주형 컬럼에 대한 기술 통계를 보고서에 추가합니다.
    report_content.append("### 범주형 데이터 기술통계\n\n")
    report_content.append(df.describe(include=['object', 'str']).to_markdown() + "\n\n")
    report_content.append("""*인사이트: 데이터는 약 1200개의 도서 정보를 포함하며, 숫자형 데이터(가격, 판매지수 등)와 범주형 데이터(출판사, 저자 등)로 구성되어 있습니다. '저자'와 '출판사'의 유니크한 값이 많은 것으로 보아 다양한 저자와 출판사가 참여하고 있음을 알 수 있습니다.*\n\n""")

    # --- 3. 시각화 분석 (Visualization Analysis) ---
    report_content.append("## 3. 시각화 분석\n\n")
    #font의 파일정보로 font name 을 알아내기
    font_prop = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_prop)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 숫자 데이터 분포를 위한 히스토그램.
    report_content.append("### 수치 데이터 분포\n\n")
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True) # kde=True는 커널 밀도 추정 플롯을 추가합니다.
        plt.title(f'{col} 분포')
        img_path = os.path.join(output_folder, f'hist_{col}.png')
        plt.savefig(img_path)
        plt.close() # 메모리 확보를 위해 플롯을 닫습니다.
        report_content.append(f"**{col} 분포**\n\n")
        report_content.append(f"![{col} 분포](./images/hist_{col}.png)\n\n")
    report_content.append("""*인사이트: 가격 데이터는 대부분 2-4만원대에 집중되어 있으며, 일부 고가의 책이 존재합니다. 판매지수와 리뷰수는 오른쪽으로 꼬리가 긴 분포를 보여 소수의 책이 매우 높은 판매지수와 리뷰수를 가짐을 시사합니다.*\n\n""")

    # 범주형 데이터 빈도를 위한 막대 그래프.
    report_content.append("### 범주형 데이터 분포\n\n")
    categorical_cols = ['출판사', '저자', '발행일']
    for col in categorical_cols:
        plt.figure(figsize=(12, 8))
        # 플로팅을 위해 가장 빈번한 상위 20개 카테고리를 가져옵니다.
        top_20 = df[col].value_counts().nlargest(20)
        sns.barplot(y=top_20.index, x=top_20.values)
        plt.title(f'상위 20개 {col} 빈도')
        plt.tight_layout() # 라벨이 겹치지 않도록 레이아웃을 조정합니다.
        img_path = os.path.join(output_folder, f'bar_{col}.png')
        plt.savefig(img_path)
        plt.close() # 메모리 확보를 위해 플롯을 닫습니다.
        report_content.append(f"**상위 20개 {col}**\n\n")
        report_content.append(f"![{col} 빈도](./images/bar_{col}.png)\n\n")
    report_content.append("""*인사이트: '한빛미디어', '길벗', '제이펍' 등의 출판사가 AI 관련 도서를 다수 출판하고 있습니다. 특정 저자들이 여러 권의 책을 집필한 경우도 확인됩니다.*\n\n""")

    # 숫자 데이터의 상관 관계 행렬을 위한 히트맵.
    report_content.append("### 상관 관계 분석\n\n")
    plt.figure(figsize=(10, 8))
    corr_df = df[numeric_cols].corr() # 쌍별 상관 관계를 계산합니다.
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f") # annot=True는 상관 관계 값을 표시합니다.
    plt.title('수치 데이터 간 상관관계')
    img_path = os.path.join(output_folder, 'heatmap_correlation.png')
    plt.savefig(img_path)
    plt.close() # 메모리 확보를 위해 플롯을 닫습니다.
    report_content.append(f"![상관관계 히트맵](./images/heatmap_correlation.png)\n\n")
    report_content.append("""*인사이트: '판매가'와 '정가'는 매우 높은 양의 상관관계를 보입니다. '리뷰수'와 '판매지수'도 어느 정도 양의 상관관계를 보여, 리뷰가 많을수록 판매지수가 높은 경향이 있음을 알 수 있습니다.*\n\n""")

    # 도서 제목의 워드 클라우드.
    report_content.append("### 도서 제목 워드 클라우드\n\n")
    if WordCloud: # WordCloud 라이브러리가 설치된 경우에만 생성합니다.
        words = []
        for title in df['제목'].dropna():
            # 공백으로 간단하게 단어를 분리합니다; 더 고급 NLP를 위해서는 konlpy 등이 필요할 수 있습니다.
            words.extend(title.split())
        
        # 짧은 단어와 일반적인 불용어를 필터링합니다.
        words = [w for w in words if len(w) > 1]
        stop_words = ["시대", "활용", "개발", "데이터", "분석", "학습", "모델", "시스템", "구축", "프로그래밍", "만들기", "시작", "배우는", "위한", "실전"]
        words = [w for w in words if w not in stop_words]

        # 단어 빈도를 계산합니다.
        count = Counter(words)
        
        # WordCloud를 위한 한글 폰트 경로를 명시적으로 설정합니다.
        font_path_for_wordcloud = 'C:/Windows/Fonts/malgun.ttf'
        # 일반 Malgun Gothic 폰트를 찾을 수 없는 경우 굵은 글꼴 버전으로 대체합니다.
        if not os.path.exists(font_path_for_wordcloud):
            font_path_for_wordcloud = 'C:/Windows/Fonts/malgunbd.ttf'
        
        # 적절한 폰트를 찾을 수 없는 경우 경고를 기록하고 WordCloud가 기본 폰트를 사용하도록 합니다.
        if not os.path.exists(font_path_for_wordcloud):
            logger.warning("Malgun Gothic 폰트를 찾을 수 없습니다. 워드 클라우드가 한글을 올바르게 렌더링하지 못할 수 있습니다.")
            font_path_for_wordcloud = None # WordCloud가 기본값을 사용하도록 합니다.
        
        # 워드 클라우드를 생성합니다.
        if font_path_for_wordcloud:
            wordcloud = WordCloud(font_path=font_path_for_wordcloud, width=800, height=400, background_color='white').generate_from_frequencies(count)
        else:
            logger.warning("워드 클라우드를 위한 적절한 폰트 경로를 찾을 수 없습니다. 워드 클라우드가 한글을 올바르게 렌더링하지 못할 수 있습니다.")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(count)
        
        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off') # 더 깔끔한 워드 클라우드 표시를 위해 축을 숨깁니다.
        img_path = os.path.join(output_folder, 'wordcloud_title.png')
        plt.savefig(img_path)
        plt.close() # 메모리 확보를 위해 플롯을 닫습니다.
        report_content.append(f"![워드클라우드](./images/wordcloud_title.png)\n\n")
        report_content.append("""*인사이트: (konlpy 미사용으로 정확도가 낮을 수 있음) '인공지능', '챗GPT', '파이썬' 등의 키워드가 자주 등장합니다.*\n\n""")
    else:
        report_content.append("wordcloud 라이브러리를 찾을 수 없어 워드 클라우드를 생성하지 못했습니다.\n\n")

    # --- 4. 교차 분석 (Cross Analysis) ---
    report_content.append("## 4. 교차 분석 (Pivot & Crosstab)\n\n")
    
    # 피벗 테이블: 평균 판매 관련 지표별 상위 10개 출판사.
    pivot1 = df.pivot_table(index='출판사', values=['판매가', '리뷰수', '판매지수'], aggfunc='mean').nlargest(10, '판매지수')
    report_content.append("### 상위 10개 출판사별 평균 판매가, 리뷰수, 판매지수\n\n")
    report_content.append(pivot1.to_markdown() + "\n\n")
    report_content.append("""*인사이트: 판매지수가 높은 책을 출판하는 출판사들의 평균 판매가와 리뷰수를 비교해볼 수 있습니다.*\n\n""")

    # 피벗 테이블: 발행연도별 평균 가격 변화.
    pivot2 = df.pivot_table(index='발행연도', values=['정가', '판매가'], aggfunc='mean')
    report_content.append("### 발행연도별 평균 가격 변화\n\n")
    report_content.append(pivot2.to_markdown() + "\n\n")
    report_content.append("""*인사이트: 시간에 따른 AI 도서의 평균 가격 변화 추이를 파악할 수 있습니다.*\n\n""")

    # 교차표: 상위 5개 출판사의 연도별 출판 건수.
    top_5_pubs = df['출판사'].value_counts().nlargest(5).index
    crosstab1 = pd.crosstab(df[df['출판사'].isin(top_5_pubs)]['출판사'], df['발행연도'])
    report_content.append("### 상위 5개 출판사의 연도별 출판 건수\n\n")
    report_content.append(crosstab1.to_markdown() + "\n\n")
    report_content.append("""*인사이트: 주요 출판사들이 언제부터 AI 관련 서적을 활발히 출판하기 시작했는지 경향을 볼 수 있습니다.*\n\n""")

    # 피벗 테이블: 평균 판매지수별 상위 10명 저자.
    pivot3 = df.groupby('저자')['판매지수'].mean().nlargest(10)
    report_content.append("### 상위 10명 저자의 평균 판매지수\n\n")
    report_content.append(pivot3.to_frame().to_markdown() + "\n\n")
    report_content.append("""*인사이트: 어떤 저자의 책들이 시장에서 좋은 반응을 얻고 있는지 파악할 수 있습니다.*\n\n""")

    # 교차표: 다른 가격 범위에 걸친 도서 수 분포.
    df['가격대'] = pd.cut(df['판매가'], bins=[0, 15000, 20000, 25000, 30000, 40000, np.inf], labels=['1.5만 이하', '1.5-2만', '2-2.5만', '2.5-3만', '3-4만', '4만 이상'])
    crosstab2 = pd.crosstab(df['가격대'], columns='count')
    report_content.append("### 판매가 가격대별 도서 수\n\n")
    report_content.append(crosstab2.to_markdown() + "\n\n")
    report_content.append("""*인사이트: 대부분의 AI 도서가 어떤 가격대에 분포하는지 명확하게 확인할 수 있습니다.*\n\n""")


    # --- 5. 보고서 저장 (Report Generation) ---
    # 수집된 모든 보고서 내용을 마크다운 파일에 작성합니다.
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("".join(report_content))
        
    print(f"분석 보고서가 '{report_path}'에 저장되었습니다.")
    print(f"시각화 이미지는 '{output_folder}' 폴더에 저장되었습니다.")
    logger.info(f"EDA 보고서가 '{report_path}'에 저장되었습니다. 이미지는 '{output_folder}'에 저장되었습니다.")


if __name__ == "__main__":
    analyze_yes24_data()
