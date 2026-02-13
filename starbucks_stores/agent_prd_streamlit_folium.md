# 📌 Streamlit 스타벅스 전국 매장 분석 앱 개발 프롬프트

다음 요구사항을 충족하는 **Streamlit 기반 스타벅스 전국 매장 분석 웹앱**을 구현하라. 코드는 작성하지 말고 기능 위주의 구조와 설명을 제시하라.

데이터와 분석 내용은 다음 파일을 참고할 것 => starbucks_stores/starbucks_ai.csv
starbucks_stores 해당 경로의 분석 결과도 참고할 것 => starbucks_stores/eda_report.md

---

## 1. 📁 데이터 로드 및 기본 설정

* CSV 파일에는 스타벅스 전국 매장 정보가 포함되어 있으며, 최소한 다음 컬럼을 포함한 것으로 가정한다:
  `매장명, 시도명, 시군구명, 주소, 위도, 경도, 매장타입 등`
* Streamlit 앱 실행 시 CSV 파일을 로드하고 결측치 및 데이터 타입을 점검하는 초기 처리 설명 포함.
* 사이드바에 다음 주요 기능을 배치한다:

  * 시도 선택 필터
  * 시군구 선택 필터
  * 매장명 검색 입력창
  * 분석/시각화 메뉴 이동

---

## 2. 📊 기본 EDA(탭 구성)

기본 EDA는 탭 또는 메뉴 형태로 구성할 것. 각 탭에서 제공해야 하는 기능은 다음과 같다.

### ✔ 탭 1: 데이터 개요

* 전체 데이터 행·열 수 표시
* 컬럼별 데이터 타입
* 결측치 요약
* 시도별 매장 수 막대 그래프
* 시군구별 매장 수 막대 그래프

### ✔ 탭 2: 시도·시군구별 탐색 기능

* 시도/시군구 필터에 따라 매장 리스트를 동적으로 업데이트
* 선택한 지역의 매장 정보 테이블 제공
* 매장 타입별 분포 시각화
* 매장명 검색 시 자동 필터링

---

## 3. 🗺️ 지도 시각화 (Folium 활용)

지도 시각화 역시 탭 또는 메뉴로 구성.

### ✔ 탭 3: 전국 매장 지도

* Folium 지도에 모든 매장 마커 표시
* 선택된 시도/시군구만 지도에 반영되도록 필터 적용
* 매장 클릭 시 팝업으로 정보 표시 (매장명, 주소, 타입 등)

### ✔ 탭 4: 클러스터 맵

* Folium MarkerCluster 적용
* 전국 매장 클러스터링 시각화
* 지역 필터 적용 가능

---

## 4. 🤖 군집화(Clustering) 분석 기능

### ✔ 탭 5: 군집화 분석

* 위도/경도 기반 K-means 또는 DBSCAN 등 군집 알고리즘 적용
* 클러스터 개수 조절 슬라이더 제공
* 각 클러스터의 중심 정보 및 구성 매장 수 요약
* 결과를 Folium 지도에 색상별로 시각화
* 클러스터별 매장 특성 분석 내용 포함

  * 시도·시군구 분포
  * 매장 타입 분포
  * 주요 위치적 특성 설명

---

## 5. 🔎 매장 검색 기능

앱 전체에서 작동하는 **글로벌 매장 검색 기능** 구현을 명시한다.

* 사용자가 "매장명"을 입력하면 즉시 검색
* 검색된 매장이 포함된 지역을 자동 포커싱
* 해당 매장을 지도에서 강조 표시
* 검색 결과 테이블 제공

---

## 6. 🧭 사용자 인터페이스 구성 요구

* 상단 또는 좌측에 주요 메뉴(탭) 구성
* 사이드바에는 사용자 입력 컨트롤 배치

  * 시도 선택
  * 시군구 선택
  * 매장명 검색
  * 군집화 옵션 등
* 데이터 프레임/지도/그래프는 메인 화면에 표현
* 사용자 선택에 따라 모든 그래프와 지도가 실시간 업데이트되도록 구성

---

## 7. 📑 전체 구성 요약

앱은 다음과 같은 메뉴 또는 탭 구조를 가져야 한다:

1. **데이터 요약 (EDA 기본 정보)**
2. **지역별 분석 (시도·시군구별 탐색)**
3. **전국 매장 지도 시각화**
4. **클러스터맵 시각화**
5. **군집화 분석**
6. **매장 검색 기능**

각 화면은 Streamlit 환경에 맞게 동적 인터랙티브 UI 요소로 구성될 것.

실제 컬럼명은 csv 파일의 다음 일부 행의 내용을 참고할 것 

seq,sido_cd,sido_nm,gugun_cd,gugun_nm,code_order,view_yn,store_num,sido,gugun,address,new_img_nm,p_pro_seq,p_view_yn,p_sido_cd,p_gugun_cd,p_store_nm,p_theme_cd,p_wireless_yn,p_smoking_yn,p_book_yn,p_music_yn,p_terrace_yn,p_table_yn,p_takeout_yn,p_parking_yn,p_dollar_assent,p_card_recharge,p_subway_yn,stb_store_file_renew,stb_store_theme_renew,stb_store_time_renew,stb_store_lsm,s_code,s_name,tel,dlvry_call_cntr_phno,fax,sido_code,sido_name,gugun_code,gugun_name,addr,park_info,new_state,theme_state,new_bool,search_text,ins_lat,ins_lng,in_distance,out_distance,all_search_cnt,addr_search_cnt,store_search_cnt,rowCount,store_nm,store_cd,s_biz_code,new_icon,set_user,favorites,map_desc,notice,defaultimage,etcimage,in_biz_cd,in_store_cd,in_favorites,in_user_id,in_biz_cds,in_biz_arr,in_biz_arrdata,in_scodes,in_scode_arr,in_scode_arrdata,disp,set_date,hlytag,hlytag_msg,vSal,istart,iend,open_dt,gold_card,ip_lat,ip_long,espresso,new_store,premiere_food,doro_address,cold_blew,my_siren_order_store_yn,whcroad_yn,skuNo,skuName,skuImgUrl,stock_count,store_area_name,store_area_code,is_open,ev_open_yn,option,gift_stock_yn,lat,lot,t20,t04,t03,t01,t12,t09,t06,t10,p10,p50,p20,p60,p30,p70,p40,p80,t22,t21,p90,p01,t05,t30,t36,t27,t29,t43,t48,z9999,t64,t66,p02
0,,,,,,,,,,,,0,,,,,,,,,,,,,,,,,,,,,1509,역삼아레나빌딩,1522-3232,,02-568-3763,01,서울,0101,강남구,서울특별시 강남구 역삼동 721-13 아레나빌딩,,,Z9999@T05@T08@T16@T17@T20@T21@T30@T32@T36@@T43@T52@T56@T57@T64@T65@T69@T71@P80@P90,0,,,,0,9.26,-1,-1,-1,30,,0,3762,N,,0,,,/upload/store/2019/06/[3762]_20190612101444_j4839.JPG,,,,,,0,,,0,,,,,,,,1,60,20190613,0,,,,,,서울특별시 강남구 언주로 425 (역삼동),,N,WHCROAD,,,,0,,A01,,,,,37.501087,127.043069,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
