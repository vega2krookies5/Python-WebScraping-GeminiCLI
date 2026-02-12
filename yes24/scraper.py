# 필요한 라이브러리를 가져옵니다.
import requests  # 웹 페이지에 HTTP 요청을 보내기 위한 라이브러리
from bs4 import BeautifulSoup  # HTML, XML 파일에서 데이터를 파싱하기 위한 라이브러리
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
import time  # 시간 관련 기능을 사용하기 위한 라이브러리
from datetime import datetime  # 날짜와 시간을 다루기 위한 라이브러리

# Yes24의 도서 정보를 스크랩하는 함수를 정의합니다.
def scrape_yes24():
    """
    Yes24 도서 정보를 스크랩하는 함수
    """
    # 스크래핑할 기본 URL을 설정합니다.
    base_url = "https://www.yes24.com/product/category/CategoryProductContents"
    # HTTP 요청에 사용할 헤더 정보를 설정합니다. (Referer, User-Agent 등)
    headers = {
        "Referer": "https://www.yes24.com/product/category/display/001001003032",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    # 모든 도서 정보를 저장할 리스트를 초기화합니다.
    all_books = []
    
    # 데이터 수집 시작을 알리는 메시지를 출력합니다.
    print("Yes24 도서 데이터 수집을 시작합니다.")
    
    # 1페이지부터 10페이지까지 반복합니다.
    for page in range(1, 11):
        # 요청에 필요한 파라미터(매개변수)를 설정합니다. (카테고리 번호, 페이지, 사이즈 등)
        params = {
            "dispNo": "001001003032",
            "order": "SINDEX_ONLY",
            "addOptionTp": "0",
            "page": page,
            "size": 120,
            "statGbYn": "N",
        }
        
        # 현재 수집 중인 페이지 번호를 출력합니다.
        print(f"{page}페이지 수집 중...")
        
        # 예외 처리를 위한 try-except 블록을 시작합니다.
        try:
            # 설정된 URL, 헤더, 파라미터로 GET 요청을 보냅니다.
            response = requests.get(base_url, headers=headers, params=params)
            # 요청이 성공하지 않았을 경우 예외를 발생시킵니다.
            response.raise_for_status()
            
            # 응답받은 HTML 텍스트를 BeautifulSoup 객체로 파싱합니다.
            soup = BeautifulSoup(response.text, 'html.parser')
            # 'itemUnit' 클래스를 가진 모든 'div' 태그(각 도서 항목)를 찾습니다.
            items = soup.find_all('div', class_='itemUnit')
            
            # 각 도서 항목에 대해 반복합니다.
            for item in items:
                # 개별 도서 정보 추출 시 발생할 수 있는 예외를 처리합니다.
                try:
                    # 'gd_name' 클래스를 가진 'a' 태그에서 제목 정보를 찾습니다.
                    title_tag = item.find('a', class_='gd_name')
                    # 제목 텍스트를 추출하고, 없으면 'N/A'로 설정합니다.
                    title = title_tag.text.strip() if title_tag else 'N/A'
                    # 상세 페이지 URL을 추출하고, 없으면 'N/A'로 설정합니다.
                    url = "https://www.yes24.com" + title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'N/A'
                    
                    # 'info_auth' 클래스를 가진 'span' 태그에서 저자 정보를 찾습니다.
                    author_tag = item.find('span', class_='info_auth')
                    # 저자 텍스트에서 '저'를 제거하고, 없으면 'N/A'로 설정합니다.
                    author = author_tag.text.strip().replace('저', '').strip() if author_tag else 'N/A'

                    # 'info_pub' 클래스를 가진 'span' 태그에서 출판사 정보를 찾습니다.
                    publisher_tag = item.find('span', class_='info_pub')
                    # 출판사 텍스트를 추출하고, 없으면 'N/A'로 설정합니다.
                    publisher = publisher_tag.text.strip() if publisher_tag else 'N/A'

                    # 'info_date' 클래스를 가진 'span' 태그에서 발행일 정보를 찾습니다.
                    date_tag = item.find('span', class_='info_date')
                    # 발행일 텍스트를 추출하고, 없으면 'N/A'로 설정합니다.
                    pub_date = date_tag.text.strip() if date_tag else 'N/A'

                    # CSS 선택자로 정가 정보를 찾습니다.
                    original_price_tag = item.select_one('.txt_num.dash em.yes_m')
                    # 정가 텍스트에서 쉼표를 제거하고, 없으면 'N/A'로 설정합니다.
                    original_price = original_price_tag.text.strip().replace(',', '') if original_price_tag else 'N/A'

                    # CSS 선택자로 판매가 정보를 찾습니다.
                    sale_price_tag = item.select_one('.txt_num em.yes_b')
                    # 판매가 텍스트에서 쉼표를 제거하고, 없으면 'N/A'로 설정합니다.
                    sale_price = sale_price_tag.text.strip().replace(',', '') if sale_price_tag else 'N/A'

                    # CSS 선택자로 리뷰 수 정보를 찾습니다.
                    review_count_tag = item.select_one('.rating_rvCount em.txC_blue')
                    # 리뷰 수 텍스트에서 괄호를 제거하고, 없으면 '0'으로 설정합니다.
                    review_count = review_count_tag.text.strip().replace('(', '').replace(')', '') if review_count_tag else '0'
                    
                    # 'saleNum' 클래스를 가진 'span' 태그에서 판매지수 정보를 찾습니다.
                    sale_num_tag = item.find('span', class_='saleNum')
                    # 판매지수 텍스트에서 '판매지수'를 제거하고, 없으면 'N/A'로 설정합니다.
                    sale_index = sale_num_tag.text.strip().replace('판매지수', '').strip() if sale_num_tag else 'N/A'
                    
                    # 'info_read' 클래스를 가진 'div' 태그에서 설명 정보를 찾습니다.
                    desc_tag = item.find('div', class_='info_read')
                    # 설명 텍스트를 추출하고, 없으면 'N/A'로 설정합니다.
                    description = desc_tag.text.strip() if desc_tag else 'N/A'

                    # 추출한 모든 정보를 딕셔너리 형태로 'all_books' 리스트에 추가합니다.
                    all_books.append({
                        "제목": title,
                        "저자": author,
                        "출판사": publisher,
                        "발행일": pub_date,
                        "정가": original_price,
                        "판매가": sale_price,
                        "리뷰수": review_count,
                        "판매지수": sale_index,
                        "상세정보": "N/A", # 상세 정보는 HTML 구조에서 특정하기 어려워 N/A로 처리
                        "설명": description,
                        "상세페이지URL": url
                    })
                # 개별 도서 정보 추출 중 오류 발생 시 메시지를 출력합니다.
                except Exception as e:
                    print(f"개별 도서 정보 추출 중 오류 발생: {e}")

            # 웹사이트에 과도한 부하를 주지 않기 위해 1초간 대기합니다.
            time.sleep(1)
            
        # HTTP 요청 관련 오류 발생 시 메시지를 출력하고 다음 페이지로 넘어갑니다.
        except requests.exceptions.RequestException as e:
            print(f"{page}페이지 수집 중 오류 발생: {e}")
            continue
            
    # 데이터 수집 완료 메시지를 출력합니다.
    print("데이터 수집 완료.")
    
    # 수집된 도서 정보가 있을 경우에만 다음을 실행합니다.
    if all_books:
        # 수집된 정보를 바탕으로 pandas DataFrame을 생성합니다.
        df = pd.DataFrame(all_books)
        
        # 수집된 데이터의 일부(상위 5개)를 출력합니다.
        print("\n수집된 데이터 일부:")
        print(df.head())
        
        # 오늘 날짜를 'YYYYMMDD' 형식의 문자열로 가져옵니다.
        #today = datetime.now().strftime("%Y%m%d")
        # 저장할 CSV 파일의 이름을 설정합니다.
        #filename = f"yes24\\data\\yes24_ai_{today}.csv"
        filename = f"yes24\\data\\yes24_ai.csv"
        
        # DataFrame을 CSV 파일로 저장합니다. (인덱스 제외, UTF-8-sig 인코딩)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        # 파일 저장 완료 메시지를 출력합니다.
        print(f"\n데이터가 '{filename}' 파일로 저장되었습니다.")
    # 수집된 정보가 없을 경우 메시지를 출력합니다.
    else:
        print("수집된 데이터가 없습니다.")

# 이 스크립트가 직접 실행될 때 scrape_yes24 함수를 호출합니다.
if __name__ == "__main__":
    scrape_yes24()