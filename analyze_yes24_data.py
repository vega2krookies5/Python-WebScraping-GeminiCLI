import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib # To enable Korean fonts in matplotlib
import os
from loguru import logger
from wordcloud import WordCloud
from collections import Counter
import re

# Configure loguru
logger.add("yes24_eda_{time}.log", rotation="500 MB")

# Define file paths
csv_filepath = "yes24/data/yes24_ai.csv"
plots_dir = "yes24/data/plots"
os.makedirs(plots_dir, exist_ok=True)

logger.info(f"Loading data from {csv_filepath}")
try:
    df = pd.read_csv(csv_filepath)
    logger.info("Data loaded successfully.")
except FileNotFoundError:
    logger.error(f"Error: The file {csv_filepath} was not found.")
    exit()

# --- Initial Data Inspection ---
logger.info("Performing initial data inspection.")
print("--- DataFrame Head ---")
print(df.head())
print("
--- DataFrame Info ---")
df.info()
print("
--- DataFrame Description ---")
print(df.describe(include='all'))
print(f"
DataFrame Shape: {df.shape}")
print(f"DataFrame Columns: {df.columns.tolist()}")

# --- Data Cleaning and Preprocessing ---
logger.info("Starting data cleaning and preprocessing.")

# Rename columns for easier access (optional, but good practice)
df.columns = ['title', 'author', 'publisher', 'publication_date', 'original_price', 'sales_price', 'review_count', 'sales_index', 'detail_info', 'description', 'product_url']

# Handle missing values - fill 'N/A' in 'detail_info' and 'description' with actual NaN
df.replace('N/A', pd.NA, inplace=True)

# Convert numeric columns
for col in ['original_price', 'sales_price', 'review_count', 'sales_index']:
    if col in df.columns:
        # Remove non-numeric characters (e.g., '원', commas) and convert to numeric
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        # For sales_index, handle empty strings after cleaning as 0 or NaN
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
logger.info("Numeric columns converted and cleaned.")


# Convert 'publication_date' to datetime and extract year
if 'publication_date' in df.columns:
    # Attempt to parse date, handling potential errors and different formats
    # First, try a common format, then a more flexible one if needed.
    df['publication_date'] = df['publication_date'].str.replace('년', '-').str.replace('월', '').str.strip()
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%Y-%m', errors='coerce')
    df['publication_year'] = df['publication_date'].dt.year
    df['publication_month'] = df['publication_date'].dt.month
    logger.info("Publication date processed and year/month extracted.")

# Calculate discount rate
df['discount_rate'] = ((df['original_price'] - df['sales_price']) / df['original_price'] * 100).round(2)
# Handle cases where original_price is 0 to avoid division by zero or NaN discount
df.loc[df['original_price'] == 0, 'discount_rate'] = 0
logger.info("Discount rate calculated.")

# --- Exploratory Data Analysis (EDA) and Visualization ---
logger.info("Starting EDA and visualization.")

# 1. Distribution of Sales Price
plt.figure(figsize=(10, 6))
sns.histplot(df['sales_price'], kde=True)
plt.title('도서 판매가 분포')
plt.xlabel('판매가 (원)')
plt.ylabel('도서 수')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'sales_price_distribution.png'))
plt.close()
logger.info("Generated sales_price_distribution.png")

# 2. Top 10 Authors by Number of Books
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
logger.info("Generated top_authors.png")

# 3. Top 10 Publishers by Number of Books
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
logger.info("Generated top_publishers.png")

# 4. Books Published per Year (if 'publication_year' exists and is cleaned)
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
    logger.info("Generated books_per_year.png")

# 5. Relationship between Sales Index and Review Count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sales_index', y='review_count', data=df)
plt.title('판매지수 vs. 리뷰 수')
plt.xlabel('판매지수')
plt.ylabel('리뷰 수')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'sales_index_vs_review_count.png'))
plt.close()
logger.info("Generated sales_index_vs_review_count.png")

# 6. Distribution of Discount Rate
plt.figure(figsize=(10, 6))
sns.histplot(df['discount_rate'], bins=20, kde=True)
plt.title('도서 할인율 분포')
plt.xlabel('할인율 (%)')
plt.ylabel('도서 수')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'discount_rate_distribution.png'))
plt.close()
logger.info("Generated discount_rate_distribution.png")

# 7. Word Cloud for Titles (requires proper Korean font and text processing)
# Combine all titles into a single string
all_titles = ' '.join(df['title'].dropna().astype(str))

# Basic text cleaning for word cloud
# Remove special characters, numbers, and common short words that might not be meaningful
def clean_text(text):
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text) # Keep Korean, English, spaces
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text

cleaned_titles = clean_text(all_titles)

# Filter out very common Korean stop words (manual list as an example)
# A more comprehensive list would be needed for a real-world scenario
korean_stopwords = ["책", "대한", "위한", "으로", "하는", "이다", "있습니다", "그리고", "대한민국"] 
words = cleaned_titles.split()
filtered_words = [word for word in words if word not in korean_stopwords and len(word) > 1]
text_for_wordcloud = ' '.join(filtered_words)


# Define a path to a Korean font available on Windows.
# This might need to be adjusted based on the user's system.
# Common Windows Korean fonts: 'Malgun Gothic', 'Batang', 'Gulim'
font_path = 'C:/Windows/Fonts/malgunbd.ttf' # Malgun Gothic Bold

# Check if the font path exists, otherwise try another common one
if not os.path.exists(font_path):
    font_path = 'C:/Windows/Fonts/malgun.ttf' # Malgun Gothic Regular
if not os.path.exists(font_path):
    logger.warning(f"Korean font not found at {font_path}. Word cloud might not display Korean correctly.")
    # Fallback to a generic font or provide instructions
    font_path = None # WordCloud will use its default, which might not support Korean

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
    logger.info("Generated title_wordcloud.png")
else:
    logger.warning("Skipping word cloud generation due to missing Korean font.")

logger.info("EDA and visualization completed. Plots saved to yes24/data/plots.")

# You can add more analysis and visualizations here.
# For example, average sales price per publisher, correlation matrix, etc.

# Example: Average sales price per publisher (top 10)
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
logger.info("Generated avg_price_per_publisher.png")

logger.info("All analysis and visualization steps completed.")
