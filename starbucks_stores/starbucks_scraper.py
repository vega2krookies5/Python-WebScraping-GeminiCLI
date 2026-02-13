import requests
import pandas as pd
from loguru import logger
import os
import time

# Logger Setup
logger.add("starbucks_scraper_{time}.log", rotation="500 MB")

# Constants
BASE_URL = "https://www.starbucks.co.kr/store/getStore.do"
HEADERS = {
    "host": "www.starbucks.co.kr",
    "origin": "https://www.starbucks.co.kr",
    "referer": "https://www.starbucks.co.kr/store/store_map.do",
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest",
}
PAYLOAD_TEMPLATE = {
    "in_biz_cds": "0",
    "in_scodes": "0",
    "ins_lat": "37.56682",
    "ins_lng": "126.97865",
    "search_text": "",
    "p_sido_cd": "",  # This will be updated
    "p_gugun_cd": "",
    "isError": "true",
    "in_distance": "0",
    "in_biz_cd": "",
    "iend": "1000",
    "searchType": "C",
    "set_date": "",
    "rndCod": "9QQ7ILZT2H", # This might need to be dynamic, but trying static first
    "all_store": "0",
    "T03": "0", "T01": "0", "T27": "0", "T12": "0", "T09": "0", "T30": "0", "T05": "0", "T22": "0", "T21": "0", "T36": "0", "T43": "0", "Z9999": "0", "T64": "0", "T66": "0", "P02": "0", "P10": "0", "P50": "0", "P20": "0", "P60": "0", "P30": "0", "P70": "0", "P40": "0", "P80": "0", "whcroad_yn": "0", "P90": "0", "P01": "0", "new_bool": "0",
}

OUTPUT_DIR = "data"
OUTPUT_FILENAME = "starbucks_ai.csv"

def scrape_starbucks_stores():
    all_stores = []
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(1, 18):  # p_sido_cd from 01 to 17
        sido_cd = str(i).zfill(2)
        logger.info(f"Scraping stores for p_sido_cd: {sido_cd}")

        payload = PAYLOAD_TEMPLATE.copy()
        payload["p_sido_cd"] = sido_cd

        try:
            response = requests.post(BASE_URL, headers=HEADERS, data=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            json_data = response.json()

            if json_data and 'list' in json_data and json_data['list']:
                logger.debug(f"Found {len(json_data['list'])} stores for p_sido_cd: {sido_cd}")
                all_stores.extend(json_data['list'])
            else:
                logger.warning(f"No stores found or empty list for p_sido_cd: {sido_cd}")
            
            time.sleep(1) # Be polite and avoid overwhelming the server

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for p_sido_cd {sido_cd}: {e}")
        except ValueError as e:
            logger.error(f"Failed to decode JSON for p_sido_cd {sido_cd}: {e}")

    if not all_stores:
        logger.error("No store data collected. Exiting.")
        return

    df = pd.DataFrame(all_stores)
    df.to_csv(output_path, index=False, encoding="utf-8-sig") # Use utf-8-sig for proper Excel display
    logger.success(f"Successfully scraped {len(df)} stores and saved to {output_path}")

if __name__ == "__main__":
    # Change current working directory to starbucks_stores to ensure relative paths work
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    scrape_starbucks_stores()
