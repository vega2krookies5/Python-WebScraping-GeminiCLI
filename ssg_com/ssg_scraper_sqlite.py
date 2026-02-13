
import requests
import pandas as pd
import sqlite3
from loguru import logger
import time
import json

# --- 로거 설정 ---
logger.add("ssg_com/ssg_scraping_{time}.log", rotation="10 MB", encoding="utf-8")

# --- 상수 정의 ---
DB_FILE = "ssg_com/ssg_items.db"
TABLE_NAME = "items"
API_URL = "https://frontapi.ssg.com/dp/api/v2/page/area"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json;charset=UTF-8",
    "origin": "https://www.ssg.com",
    "referer": "https://www.ssg.com/page/pc/SpecialPrice.ssg",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def create_connection_and_table() -> sqlite3.Connection:
    """데이터베이스에 연결하고 'items' 테이블이 없으면 생성합니다."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 테이블 생성 (- 주요 컬럼 외에는 TEXT로 저장)
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        itemId TEXT PRIMARY KEY,
        itemNm TEXT,
        uitemId TEXT,
        displayPrc INTEGER,
        strikeOutPrc INTEGER,
        soldOutYn TEXT,
        siteNo TEXT,
        salestrNo TEXT,
        brandId TEXT,
        brandNm TEXT,
        itemLnkd TEXT,
        itemImgUrl TEXT,
        benefitGrp0 TEXT,
        benefitGrp1 TEXT,
        benefitGrp2 TEXT,
        benefitGrp3 TEXT,
        benefitGrp4 TEXT,
        benefitGrp5 TEXT,
        itemTagNmAndItemCnt TEXT,
        all_data TEXT
    )
    """)
    conn.commit()
    logger.info(f"데이터베이스 '{DB_FILE}'에 연결되었고, '{TABLE_NAME}' 테이블이 준비되었습니다.")
    return conn

def fetch_items(page: int, retries: int = 3) -> list[dict]:
    """지정된 페이지에서 상품 목록을 가져옵니다."""
    payload = {
        "common": {
            "aplVer": "", "osCd": "", "ts": "20251118062034", "mobilAppNo": "99",
            "dispDvicDivCd": "10", "viewSiteNo": "6005"
        },
        "params": {
            "page": str(page), "state": "{}", "pageId": "100000007533",
            "pageSetId": "2", "dispCtgId": None, "pageCmptId": "4"
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
            response.raise_for_status()  # 200이 아닌 상태 코드에 대해 예외 발생
            
            data = response.json()
            all_items = []
            
            if "data" in data and "areaList" in data["data"]:
                for area_group in data["data"]["areaList"]:
                    for area in area_group:
                        if "itemList" in area:
                            all_items.extend(area["itemList"])
            
            logger.info(f"페이지 {page}: 상품 {len(all_items)}개 수집 완료. (URL: {API_URL})")
            return all_items
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"페이지 {page} 수집 중 오류 발생 (시도 {attempt + 1}/{retries}): {e}")
            time.sleep(2) # 재시도 전 잠시 대기
            
    logger.error(f"페이지 {page} 수집 최종 실패. 다음 페이지로 넘어갑니다.")
    return []

def items_to_df(items: list[dict]) -> pd.DataFrame:
    """상품 dict 리스트를 DataFrame으로 변환합니다."""
    if not items:
        return pd.DataFrame()

    # 모든 아이템을 JSON 문자열로 변환하여 'all_data' 컬럼에 저장
    for item in items:
        item['all_data'] = json.dumps(item, ensure_ascii=False)

    df = pd.json_normalize(items)
    
    # 리스트 타입 컬럼을 JSON 문자열로 변환 (해당하는 경우)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

    return df

def save_df_to_db(df: pd.DataFrame, conn: sqlite3.Connection):
    """DataFrame을 SQLite 데이터베이스에 UPSERT 방식으로 저장합니다."""
    if df.empty:
        logger.info("저장할 데이터가 없습니다.")
        return

    cursor = conn.cursor()
    
    # 테이블에 존재하는 컬럼만 선택
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    table_columns = [info[1] for info in cursor.fetchall()]
    df_columns = [col for col in df.columns if col in table_columns]
    
    df_to_save = df[df_columns]

    # INSERT OR IGNORE를 사용하여 데이터 삽입
    cols = ', '.join(df_to_save.columns)
    placeholders = ', '.join(['?'] * len(df_to_save.columns))
    
    sql = f"INSERT OR IGNORE INTO {TABLE_NAME} ({cols}) VALUES ({placeholders})"
    
    try:
        cursor.executemany(sql, df_to_save.to_records(index=False))
        conn.commit()
        logger.success(f"데이터 {len(df_to_save)}개 중 새로운 데이터가 DB에 저장되었습니다.")
        # 저장된 첫번째 아이템 이름 로깅
        if not df_to_save.empty and 'itemNm' in df_to_save.columns:
            logger.info(f"  -> 예시: '{df_to_save.iloc[0]['itemNm']}'")

    except sqlite3.Error as e:
        logger.error(f"DB 저장 중 오류 발생: {e}")
        conn.rollback()

def main(start_page: int = 1, end_page: int = 5):
    """메인 실행 함수: 지정된 범위의 페이지를 스크레이핑하고 DB에 저장합니다."""
    logger.info(f"SSG 상품 스크레이핑 시작 (페이지 {start_page}부터 {end_page}까지)")
    
    conn = None
    try:
        conn = create_connection_and_table()
        
        for page_num in range(start_page, end_page + 1):
            logger.info(f"--- 페이지 {page_num} 처리 시작 ---")
            
            # 1. 데이터 수집
            items = fetch_items(page_num)
            if not items:
                continue

            # 2. 데이터프레임 변환
            df = items_to_df(items)
            
            # 3. 데이터베이스에 저장
            save_df_to_db(df, conn)
            
            logger.info(f"--- 페이지 {page_num} 처리 완료 ---")
            time.sleep(1) # 페이지 요청 간 1초 대기

    except Exception as e:
        logger.critical(f"스크립트 실행 중 예기치 않은 오류 발생: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("데이터베이스 연결이 종료되었습니다.")
            
    logger.info("SSG 상품 스크레이핑 완료.")

if __name__ == "__main__":
    # 예: 12페이지부터 15페이지까지 수집
    main(start_page=6, end_page=10)
