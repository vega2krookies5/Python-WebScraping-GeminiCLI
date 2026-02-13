# Yes24 도서 데이터 탐색적 데이터 분석 (EDA) 및 시각화

## 1. 개요
이 문서는 Yes24에서 수집한 도서 데이터(`yes24/data/yes24_ai.csv`)에 대한 탐색적 데이터 분석(EDA) 및 시각화 과정을 설명합니다. 데이터의 구조를 이해하고, 주요 통계적 특성을 파악하며, 시각화를 통해 데이터에 숨겨진 패턴과 인사이트를 도출하는 것을 목표로 합니다.

## 2. 데이터 불러오기 및 초기 탐색

데이터 분석의 첫 단계로 CSV 파일을 불러와 데이터의 기본적인 구조와 내용을 확인합니다.

### 2.1. 데이터 로드

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib # 한글 폰트 설정을 위해 필요
import os

# 데이터 파일 경로
csv_filepath = "yes24/data/yes24_ai.csv"
plots_dir = "yes24/data/plots" # 시각화 이미지를 저장할 디렉토리

# 디렉토리가 없으면 생성
os.makedirs(plots_dir, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_filepath)
```

### 2.2. 데이터 초기 확인

`df.head()`, `df.info()`, `df.describe()` 등을 통해 데이터의 상위 몇 행, 컬럼 정보(데이터 타입, 결측치 등), 기술 통계량을 확인합니다.

```python
# 데이터프레임 상위 5행 확인
print(df.head())

# 데이터프레임 정보 확인 (컬럼명, Null 여부, 데이터 타입)
df.info()

# 데이터프레임 기술 통계량 확인
print(df.describe(include='all'))

# 데이터프레임의 크기 (행, 열) 확인
print(f"DataFrame Shape: {df.shape}")
# 데이터프레임의 컬럼 목록 확인
print(f"DataFrame Columns: {df.columns.tolist()}")
```

## 3. 데이터 전처리

수집된 데이터는 분석에 적합한 형태로 가공되어야 합니다. 주로 데이터 타입 변환, 결측치 처리, 새로운 파생 변수 생성 등의 작업을 수행합니다.

### 3.1. 컬럼명 정리

한글 컬럼명을 영어로 변경하여 코드에서 사용하기 용이하게 합니다.

```python
df.columns = ['title', 'author', 'publisher', 'publication_date', 'original_price', 'sales_price', 'review_count', 'sales_index', 'detail_info', 'description', 'product_url']
```

### 3.2. 결측치 및 데이터 타입 처리

`'N/A'` 문자열을 `NaN`으로 처리하고, 숫자형 데이터는 `int` 타입으로, 날짜 데이터는 `datetime` 타입으로 변환합니다. `발행일` 컬럼에서 `발행 연도`와 `발행 월`을 추출합니다.

```python
# 'N/A' 값을 NaN으로 대체
df.replace('N/A', pd.NA, inplace=True)

# 숫자형 컬럼 (정가, 판매가, 리뷰수, 판매지수) 전처리 및 타입 변환
for col in ['original_price', 'sales_price', 'review_count', 'sales_index']:
    if col in df.columns:
        # 숫자 이외의 문자 제거 후 numeric으로 변환
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# 'publication_date' 컬럼을 datetime으로 변환하고 연도/월 추출
if 'publication_date' in df.columns:
    df['publication_date'] = df['publication_date'].str.replace('년', '-').str.replace('월', '').str.strip()
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%Y-%m', errors='coerce')
    df['publication_year'] = df['publication_date'].dt.year
    df['publication_month'] = df['publication_date'].dt.month
```

### 3.3. 파생 변수 생성

할인율을 계산하여 `discount_rate` 컬럼을 추가합니다.

```python
# 할인율 계산 (정가 0인 경우를 방지)
df['discount_rate'] = ((df['original_price'] - df['sales_price']) / df['original_price'] * 100).round(2)
df.loc[df['original_price'] == 0, 'discount_rate'] = 0
```

## 4. 탐색적 데이터 분석 (EDA) 및 시각화

전처리된 데이터를 바탕으로 다양한 시각화를 통해 패턴과 관계를 분석합니다. 모든 시각화 결과는 `yes24/data/plots` 디렉토리에 PNG 파일로 저장됩니다.

### 4.1. 도서 판매가 분포

도서 판매 가격의 분포를 히스토그램과 KDE 플롯으로 확인합니다.

```python
plt.figure(figsize=(10, 6))
sns.histplot(df['sales_price'], kde=True)
plt.title('도서 판매가 분포')
plt.xlabel('판매가 (원)')
plt.ylabel('도서 수')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'sales_price_distribution.png'))
plt.close()
```

![도서 판매가 분포 예시](yes24/data/plots/sales_price_distribution.png)

### 4.2. 상위 10명 저자별 도서 수

가장 많은 도서를 출판한 상위 10명의 저자를 막대 그래프로 시각화합니다.

```python
top_authors = df['author'].value_counts().head(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=top_authors.index, y=top_authors.values)
plt.title('상위 10명 저자별 도서 수')
plt.xlabel('저자')
plt.ylabel('도서 수')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_authors.png'))
plt.close()
```

![상위 10명 저자별 도서 수 예시](yes24/data/plots/top_authors.png)

### 4.3. 상위 10개 출판사별 도서 수

가장 많은 도서를 출판한 상위 10개 출판사를 막대 그래프로 시각화합니다.

```python
top_publishers = df['publisher'].value_counts().head(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=top_publishers.index, y=top_publishers.values)
plt.title('상위 10개 출판사별 도서 수')
plt.xlabel('출판사')
plt.ylabel('도서 수')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_publishers.png'))
plt.close()
```

![상위 10개 출판사별 도서 수 예시](yes24/data/plots/top_publishers.png)

### 4.4. 연도별 도서 출판 수

연도별 도서 출판 수를 꺾은선 그래프로 추세를 확인합니다.

```python
if 'publication_year' in df.columns:
    books_per_year = df['publication_year'].value_counts().sort_index()
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=books_per_year.index, y=books_per_year.values, marker='o')
    plt.title('연도별 도서 출판 수')
    plt.xlabel('출판 연도')
    plt.ylabel('도서 수')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'books_per_year.png'))
    plt.close()
```

![연도별 도서 출판 수 예시](yes24/data/plots/books_per_year.png)

### 4.5. 판매지수 vs. 리뷰 수

판매지수와 리뷰 수 간의 관계를 산점도를 통해 살펴봅니다.

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sales_index', y='review_count', data=df)
plt.title('판매지수 vs. 리뷰 수')
plt.xlabel('판매지수')
plt.ylabel('리뷰 수')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'sales_index_vs_review_count.png'))
plt.close()
```

![판매지수 vs. 리뷰 수 예시](yes24/data/plots/sales_index_vs_review_count.png)

### 4.6. 도서 할인율 분포

도서 할인율의 분포를 히스토그램으로 확인합니다.

```python
plt.figure(figsize=(10, 6))
sns.histplot(df['discount_rate'], bins=20, kde=True)
plt.title('도서 할인율 분포')
plt.xlabel('할인율 (%)')
plt.ylabel('도서 수')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'discount_rate_distribution.png'))
plt.close()
```

![도서 할인율 분포 예시](yes24/data/plots/discount_rate_distribution.png)

### 4.7. 도서 제목 워드 클라우드

도서 제목에서 자주 등장하는 키워드를 워드 클라우드로 시각화합니다. (한글 폰트 설정 필수)

```python
from wordcloud import WordCloud
from collections import Counter
import re

# 모든 제목을 하나의 문자열로 결합
all_titles = ' '.join(df['title'].dropna().astype(str))

# 텍스트 정제 (한글, 영어, 공백만 유지)
def clean_text(text):
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_titles = clean_text(all_titles)

# 불용어 제거 및 두 글자 이상의 단어 필터링
korean_stopwords = ["책", "대한", "위한", "으로", "하는", "이다", "있습니다", "그리고", "대한민국", "with"] # 예시 불용어
words = cleaned_titles.split()
filtered_words = [word for word in words if word not in korean_stopwords and len(word) > 1]
text_for_wordcloud = ' '.join(filtered_words)

# 한글 폰트 경로 설정 (Windows 환경 기준, 사용자 시스템에 따라 다를 수 있음)
font_path = 'C:/Windows/Fonts/malgunbd.ttf'
if not os.path.exists(font_path):
    font_path = 'C:/Windows/Fonts/malgun.ttf'
if not os.path.exists(font_path):
    print("Warning: Korean font not found. Word cloud might not display Korean correctly.")
    font_path = None # WordCloud will use its default

if font_path:
    wordcloud = WordCloud(
        font_path=font_path,
        background_color="white",
        width=800,
        height=400,
        max_words=100
    ).generate(text_for_wordcloud)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('도서 제목 워드 클라우드')
    plt.savefig(os.path.join(plots_dir, 'title_wordcloud.png'))
    plt.close()
```

![도서 제목 워드 클라우드 예시](yes24/data/plots/title_wordcloud.png)

### 4.8. 상위 10개 출판사별 평균 판매가

출판사별 평균 판매가를 막대 그래프로 시각화합니다.

```python
avg_price_per_publisher = df.groupby('publisher')['sales_price'].mean().nlargest(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=avg_price_per_publisher.index, y=avg_price_per_publisher.values)
plt.title('상위 10개 출판사별 평균 판매가')
plt.xlabel('출판사')
plt.ylabel('평균 판매가 (원)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'avg_price_per_publisher.png'))
plt.close()
```

![상위 10개 출판사별 평균 판매가 예시](yes24/data/plots/avg_price_per_publisher.png)

## 5. 결론 및 인사이트

이 섹션에서는 위의 분석 결과를 바탕으로 도서 데이터에 대한 주요 인사이트를 요약합니다.

*   **가격 분포**: 대부분의 도서는 [X]원 ~ [Y]원 사이에 분포하며, [Z]원 부근에 가장 많은 도서가 집중되어 있습니다.
*   **저자 및 출판사**: [가장 많은 책을 쓴 저자]와 [가장 많은 책을 낸 출판사]가 시장에서 큰 영향력을 가집니다.
*   **출판 트렌드**: 연도별 출판 도서 수를 보면 [특정 연도에 출판이 급증/감소하는 추세]가 나타나 [관련 시장 변화]를 시사합니다.
*   **판매 지수와 리뷰**: 판매 지수가 높은 도서일수록 [리뷰 수가 많거나 특정 관계]를 보입니다.
*   **할인율**: 할인율 분포를 통해 [평균 할인율]과 [주요 할인 폭]을 파악할 수 있습니다.
*   **워드 클라우드**: 도서 제목에서 'AI', 'GPT', '인공지능' 등 [주요 키워드]가 빈번하게 나타나며 [현재 시장의 관심사]를 반영합니다.

이러한 분석은 Yes24의 도서 시장 동향을 이해하고, 마케팅 전략 수립 또는 도서 추천 시스템 개발에 기초 자료로 활용될 수 있습니다. 추가적으로 각 키워드별, 출판사별 상세 분석을 통해 더욱 깊이 있는 인사이트를 얻을 수 있을 것입니다.
