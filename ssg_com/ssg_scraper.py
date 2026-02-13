import requests
import pandas as pd
from datetime import datetime
from loguru import logger
import time

# 로거 설정
logger.add("ssg_com/scraping_{time}.log", rotation="500 MB")

def get_ssg_special_price_items(page: int):
    """
    SSG.com의 특가 페이지에서 상품 목록을 가져옵니다.
    """
    url = "https://frontapi.ssg.com/dp/api/v2/page/area"
    
    headers = {
        "referer": "https://www.ssg.com/page/pc/SpecialPrice.ssg",
        "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    }
    
    # 현재 시간을 기반으로 타임스탬프 생성
    current_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    
    payload = {
        "common": {
            "aplVer": "",
            "osCd": "",
            "ts": current_ts,
            "mobilAppNo": "99",
            "dispDvicDivCd": "10",
            "viewSiteNo": "6005"
        },
        "params": {
            "page": str(page),
            "state": "{}",
            "pageId": "100000007533",
            "pageSetId": "2",
            "dispCtgId": None,
            "pageCmptId": "4"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        logger.info(f"Successfully fetched data for page {page}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data for page {page}: {e}")
        return None

def main():
    """
    여러 페이지에서 상품 정보를 스크래핑하여 CSV 파일로 저장합니다.
    """
    all_items = []
    # 1페이지부터 5페이지까지 스크래핑합니다.
    for page_num in range(1, 6):
        logger.info(f"Scraping page {page_num}...")
        data = get_ssg_special_price_items(page_num)
        time.sleep(1) # 페이지 요청 사이에 1초 대기
        
        if data and data.get("resultCode") == "200":
            try:
                area_list = data.get("data", {}).get("areaList", [])
                for area in area_list:
                    for item_group in area:
                        item_list = item_group.get("itemList", [])
                        if item_list:
                            all_items.extend(item_list)
                        else:
                            logger.warning(f"No 'itemList' found on page {page_num} in an item_group.")
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing data on page {page_num}: {e}")
                logger.debug(f"Data structure: {data}")
        else:
            logger.warning(f"No data or error in response for page {page_num}.")

    if not all_items:
        logger.warning("No items were scraped. Exiting.")
        return

    # pandas DataFrame으로 변환
    df = pd.DataFrame(all_items)
    
    # CSV 파일로 저장
    output_filename = "ssg_com/data/ssg_special_price.csv"
    df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    
    logger.info(f"Scraped {len(df)} items and saved to '{output_filename}'")
    logger.info(f"DataFrame 길이 = {df.shape}")    
    #logger.info("First 5 rows of the dataframe:")
    #logger.info("\n" + df.head().to_string())

if __name__ == "__main__":
    main()
