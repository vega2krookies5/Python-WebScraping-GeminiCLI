import requests
from bs4 import BeautifulSoup
import pandas as pd
from loguru import logger
# import koreanize_matplotlib # Needed for the project setup, though not directly used in scraping logic.
import os

# Configure loguru
logger.add("yes24_scraper_{time}.log", rotation="500 MB")

class Yes24Scraper:
    BASE_URL = "https://www.yes24.com/product/category/CategoryProductContents"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        "Referer": "https://www.yes24.com/product/category/display/001001003032",
        "X-Requested-With": "XMLHttpRequest",
        "Host": "www.yes24.com"
    }

    def __init__(self):
        logger.info("Yes24Scraper initialized.")
    
    def _parse_book_data(self, item_unit):
        """Parses a single book item from the HTML and extracts relevant data."""
        book_data = {}

        # Book Title and URL
        title_element = item_unit.select_one(".gd_name")
        if title_element:
            book_data["title"] = title_element.get_text(strip=True)
            book_data["product_url"] = "https://www.yes24.com" + title_element["href"]
        else:
            book_data["title"] = None
            book_data["product_url"] = None

        # Image URL
        img_element = item_unit.select_one(".img_bdr img.lazy")
        if img_element:
            book_data["image_url"] = img_element.get("data-original") or img_element.get("src")
        else:
            book_data["image_url"] = None

        # Author
        author_element = item_unit.select_one(".info_auth a")
        book_data["author"] = author_element.get_text(strip=True) if author_element else None

        # Publisher
        publisher_element = item_unit.select_one(".info_pub a")
        book_data["publisher"] = publisher_element.get_text(strip=True) if publisher_element else None

        # Publication Date
        date_element = item_unit.select_one(".info_date")
        book_data["publication_date"] = date_element.get_text(strip=True) if date_element else None

        # Sales Price
        sales_price_element = item_unit.select_one(".txt_num .yes_b")
        book_data["sales_price"] = int(sales_price_element.get_text(strip=True).replace(",", "")) if sales_price_element else None
        
        # Original Price
        original_price_element = item_unit.select_one(".txt_num.dash .yes_m")
        book_data["original_price"] = int(original_price_element.get_text(strip=True).replace(",", "")) if original_price_element else None

        # Rating (리뷰 총점)
        rating_element = item_unit.select_one(".rating_grade .yes_b")
        book_data["rating"] = float(rating_element.get_text(strip=True)) if rating_element else None

        # Review Count
        review_count_element = item_unit.select_one(".rating_rvCount .txC_blue")
        if review_count_element:
            book_data["review_count"] = int(review_count_element.get_text(strip=True).replace("(", "").replace("건)", ""))
        else:
            book_data["review_count"] = 0

        logger.debug(f"Parsed book: {book_data['title']}")
        return book_data

    def scrape_category(self, dispNo: str, start_page: int = 1, end_page: int = 1):
        """
        Scrapes book data for a given category and page range.

        Args:
            dispNo (str): The category display number (e.g., "001001003032").
            start_page (int): The starting page number to scrape.
            end_page (int): The ending page number to scrape.

        Returns:
            list: A list of dictionaries, where each dictionary represents a book.
        """
        all_books_data = []
        logger.info(f"Starting scraping for dispNo={dispNo} from page {start_page} to {end_page}")

        for page in range(start_page, end_page + 1):
            params = {
                "dispNo": dispNo,
                "order": "SINDEX_ONLY",
                "addOptionTp": "0",
                "page": page,
                "size": "24",
                "statGbYn": "N",
                "viewMode": "",
                "_options": "",
                "directDelvYn": "",
                "usedTp": "0",
                "elemNo": "0",
                "elemSeq": "0",
                "seriesNumber": "0"
            }
            
            try:
                logger.info(f"Fetching page {page} for dispNo={dispNo}")
                response = requests.get(self.BASE_URL, headers=self.HEADERS, params=params)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                
                soup = BeautifulSoup(response.text, "html.parser")
                item_units = soup.select(".itemUnit")

                if not item_units:
                    logger.warning(f"No item units found on page {page}. Stopping.")
                    break

                for item_unit in item_units:
                    book = self._parse_book_data(item_unit)
                    all_books_data.append(book)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for page {page}: {e}")
                break
            except Exception as e:
                logger.error(f"An error occurred while parsing page {page}: {e}")
                break
        
        logger.info(f"Finished scraping. Total books collected: {len(all_books_data)}")
        return all_books_data

if __name__ == "__main__":
    scraper = Yes24Scraper()
    
    # Example usage: Scrape "IT/모바일" category (dispNo=001001003032) for the first 3 pages
    # You can change dispNo and the page range as needed.
    it_mobile_category_id = "001001003032"
    books = scraper.scrape_category(it_mobile_category_id, start_page=1, end_page=3)

    if books:
        df = pd.DataFrame(books)
        print("\n--- Scraped Book Data ---")
        print(df.head())
        output_dir = "yes24/data"
        output_filepath = os.path.join(output_dir, "yes24_ai.csv")
        os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

        df.to_csv(output_filepath, index=False, encoding="utf-8-sig")
        logger.info(f"Data saved to {output_filepath}")
    else:
        logger.info("No books were scraped.")