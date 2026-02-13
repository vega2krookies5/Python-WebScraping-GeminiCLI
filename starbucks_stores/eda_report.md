# 스타벅스 매장 데이터 EDA 보고서

## 0. 기본 분석

### 데이터 로드 성공

데이터 경로: `starbucks_stores/data/starbucks_ai.csv`

### 데이터 구조

|    |   seq |   sido_cd |   sido_nm |   gugun_cd |   gugun_nm |   code_order |   view_yn |   store_num |   sido |   gugun |   address |   new_img_nm |   p_pro_seq |   p_view_yn |   p_sido_cd |   p_gugun_cd |   p_store_nm |   p_theme_cd |   p_wireless_yn |   p_smoking_yn |   p_book_yn |   p_music_yn |   p_terrace_yn |   p_table_yn |   p_takeout_yn |   p_parking_yn |   p_dollar_assent |   p_card_recharge |   p_subway_yn |   stb_store_file_renew |   stb_store_theme_renew |   stb_store_time_renew |   stb_store_lsm |   s_code | s_name   | tel       |   dlvry_call_cntr_phno | fax          |   sido_code | sido_name   |   gugun_code | gugun_name   | addr                       |   park_info |   new_state | theme_state                                                                          |   new_bool |   search_text |   ins_lat |   ins_lng |   in_distance |   out_distance |   all_search_cnt |   addr_search_cnt |   store_search_cnt |   rowCount |   store_nm |   store_cd |   s_biz_code | new_icon   |   set_user |   favorites |   map_desc |   notice | defaultimage                                          |   etcimage |   in_biz_cd |   in_store_cd |   in_favorites |   in_user_id |   in_biz_cds |   in_biz_arr |   in_biz_arrdata |   in_scodes |   in_scode_arr |   in_scode_arrdata |   disp |   set_date |   hlytag |   hlytag_msg |   vSal |   istart |   iend |   open_dt |   gold_card |   ip_lat |   ip_long |   espresso |   new_store |   premiere_food | doro_address               |   cold_blew | my_siren_order_store_yn   | whcroad_yn   |   skuNo |   skuName |   skuImgUrl |   stock_count |   store_area_name | store_area_code   |   is_open |   ev_open_yn |   option |   gift_stock_yn |     lat |     lot |   t20 |   t04 |   t03 |   t01 |   t12 |   t09 |   t06 |   t10 |   p10 |   p50 |   p20 |   p60 |   p30 |   p70 |   p40 |   p80 |   t22 |   t21 |   p90 |   p01 |   t05 |   t30 |   t36 |   t27 |   t29 |   t43 |   t48 |   z9999 |   t64 |   t66 |   p02 |
|---:|------:|----------:|----------:|-----------:|-----------:|-------------:|----------:|------------:|-------:|--------:|----------:|-------------:|------------:|------------:|------------:|-------------:|-------------:|-------------:|----------------:|---------------:|------------:|-------------:|---------------:|-------------:|---------------:|---------------:|------------------:|------------------:|--------------:|-----------------------:|------------------------:|-----------------------:|----------------:|---------:|:---------|:----------|-----------------------:|:-------------|------------:|:------------|-------------:|:-------------|:---------------------------|------------:|------------:|:-------------------------------------------------------------------------------------|-----------:|--------------:|----------:|----------:|--------------:|---------------:|-----------------:|------------------:|-------------------:|-----------:|-----------:|-----------:|-------------:|:-----------|-----------:|------------:|-----------:|---------:|:------------------------------------------------------|-----------:|------------:|--------------:|---------------:|-------------:|-------------:|-------------:|-----------------:|------------:|---------------:|-------------------:|-------:|-----------:|---------:|-------------:|-------:|---------:|-------:|----------:|------------:|---------:|----------:|-----------:|------------:|----------------:|:---------------------------|------------:|:--------------------------|:-------------|--------:|----------:|------------:|--------------:|------------------:|:------------------|----------:|-------------:|---------:|----------------:|--------:|--------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|--------:|------:|------:|------:|
|  0 |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |     1509 | 역삼아레나빌딩  | 1522-3232 |                    nan | 02-568-3763  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 역삼동 721-13 아레나빌딩 |         nan |         nan | Z9999@T05@T08@T16@T17@T20@T21@T30@T32@@@T36@T43@T52@T56@T57@T64@T65@T69@T71@P80@P90  |          0 |           nan |       nan |       nan |             0 |           9.26 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |         3762 | N          |        nan |           0 |        nan |      nan | /upload/store/2019/06/[3762]_20190612101444_j4839.JPG |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |  20190613 |           0 |      nan |       nan |        nan |         nan |             nan | 서울특별시 강남구 언주로 425 (역삼동)    |         nan | N                         | WHCROAD      |     nan |       nan |         nan |             0 |               nan | A01               |       nan |          nan |      nan |             nan | 37.5011 | 127.043 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
|  1 |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |     1434 | 논현역사거리   | 1522-3232 |                    nan | 02-3442-3673 |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 논현동 142-2 정일빌딩   |         nan |         nan | Z9999@T05@T08@T16@T17@T20@T21@T22@T30@T32@T36@@T52@T56@T57@T64@T65@T69@P70@P80@P90   |          0 |           nan |       nan |       nan |             0 |           7.38 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |         3672 | N          |        nan |           0 |        nan |      nan | /upload/store/2025/07/3672_20250709034455_lnuoo.jpg   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |  20181123 |           0 |      nan |       nan |        nan |         nan |             nan | 서울특별시 강남구 강남대로 538 (논현동)   |         nan | N                         | nan          |     nan |       nan |         nan |             0 |               nan | A01               |       nan |          nan |      nan |             nan | 37.5102 | 127.022 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
|  2 |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |     1595 | 신사역성일빌딩  | 1522-3232 |                    nan | 02-547-3859  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 논현동 18-4 성일빌딩    |         nan |         nan | Z9999@T05@T08@T16@T17@T20@T21@T30@T32@T34@@@@T52@T56@T57@T64@T65@T68@T69@T73@P80@P90 |          0 |           nan |       nan |       nan |             0 |           6.95 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |         3858 | N          |        nan |           0 |        nan |      nan | /upload/store/2022/01/[3858]_20220127064243_5khni.jpg |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |  20191219 |           0 |      nan |       nan |        nan |         nan |             nan | 서울특별시 강남구 강남대로 584 (논현동)   |         nan | N                         | nan          |     nan |       nan |         nan |             0 |               nan | A01               |       nan |          nan |      nan |             nan | 37.5139 | 127.021 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
|  3 |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |     1527 | 국기원사거리   | 1522-3232 |                    nan | 02-568-3669  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 역삼동 648-22 동찬빌딩  |         nan |         nan | Z9999@T05@T08@T16@T17@T20@T21@T30@T32@T34@@@T36@T52@T56@T57@T64@T65@T69@P70@P90      |          0 |           nan |       nan |       nan |             0 |           8.82 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |         3669 | N          |        nan |           0 |        nan |      nan | /upload/store/2019/07/[3669]_20190730073527_5nk40.jpg |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |  20190731 |           0 |      nan |       nan |        nan |         nan |             nan | 서울특별시 강남구 테헤란로 125 (역삼동)   |         nan | N                         | WHCROAD      |     nan |       nan |         nan |             0 |               nan | A01               |       nan |          nan |      nan |             nan | 37.4995 | 127.031 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
|  4 |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |     1468 | 대치재경빌딩   | 1522-3232 |                    nan | 02-568-3705  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 대치동 599 대원빌딩     |         nan |         nan | Z9999@@T05@T08@T16@T17@T20@T21@T22@@@T30@T32@@@T43@T52@T56@T57@T64@T65@T69@T71@P90   |          0 |           nan |       nan |       nan |             0 |          10.92 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |         3703 | N          |        nan |           0 |        nan |      nan | /upload/store/2023/12/[3703]_20231217121347_eqnky.jpg |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |  20190214 |           0 |      nan |       nan |        nan |         nan |             nan | 서울특별시 강남구 남부순환로 2947 (대치동) |         nan | N                         | WHCROAD      |     nan |       nan |         nan |             0 |               nan | A01               |       nan |          nan |      nan |             nan | 37.4947 | 127.063 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |

### 데이터 정보 (df.info())

```
<class 'pandas.DataFrame'>
RangeIndex: 2114 entries, 0 to 2113
Columns: 137 entries, seq to p02
dtypes: float64(71), int64(52), str(14)
memory usage: 2.2 MB

```

### 기술 통계

|        |   seq |   sido_cd |   sido_nm |   gugun_cd |   gugun_nm |   code_order |   view_yn |   store_num |   sido |   gugun |   address |   new_img_nm |   p_pro_seq |   p_view_yn |   p_sido_cd |   p_gugun_cd |   p_store_nm |   p_theme_cd |   p_wireless_yn |   p_smoking_yn |   p_book_yn |   p_music_yn |   p_terrace_yn |   p_table_yn |   p_takeout_yn |   p_parking_yn |   p_dollar_assent |   p_card_recharge |   p_subway_yn |   stb_store_file_renew |   stb_store_theme_renew |   stb_store_time_renew |   stb_store_lsm |   s_code | s_name   | tel       |   dlvry_call_cntr_phno | fax   |   sido_code | sido_name   |   gugun_code | gugun_name   | addr                     |   park_info | new_state   | theme_state                                                                               |   new_bool |   search_text |   ins_lat |   ins_lng |   in_distance |   out_distance |   all_search_cnt |   addr_search_cnt |   store_search_cnt |   rowCount |   store_nm |   store_cd |   s_biz_code | new_icon   |   set_user |   favorites |   map_desc |   notice | defaultimage                                          |   etcimage |   in_biz_cd |   in_store_cd |   in_favorites |   in_user_id |   in_biz_cds |   in_biz_arr |   in_biz_arrdata |   in_scodes |   in_scode_arr |   in_scode_arrdata |   disp |   set_date |   hlytag |   hlytag_msg |   vSal |   istart |   iend |         open_dt |   gold_card |   ip_lat |   ip_long |   espresso |   new_store |   premiere_food | doro_address           |   cold_blew | my_siren_order_store_yn   | whcroad_yn   |   skuNo |   skuName |   skuImgUrl |   stock_count |   store_area_name | store_area_code   |   is_open |   ev_open_yn |   option |   gift_stock_yn |        lat |         lot |   t20 |   t04 |   t03 |   t01 |   t12 |   t09 |   t06 |   t10 |   p10 |   p50 |   p20 |   p60 |   p30 |   p70 |   p40 |   p80 |   t22 |   t21 |   p90 |   p01 |   t05 |   t30 |   t36 |   t27 |   t29 |   t43 |   t48 |   z9999 |   t64 |   t66 |   p02 |
|:-------|------:|----------:|----------:|-----------:|-----------:|-------------:|----------:|------------:|-------:|--------:|----------:|-------------:|------------:|------------:|------------:|-------------:|-------------:|-------------:|----------------:|---------------:|------------:|-------------:|---------------:|-------------:|---------------:|---------------:|------------------:|------------------:|--------------:|-----------------------:|------------------------:|-----------------------:|----------------:|---------:|:---------|:----------|-----------------------:|:------|------------:|:------------|-------------:|:-------------|:-------------------------|------------:|:------------|:------------------------------------------------------------------------------------------|-----------:|--------------:|----------:|----------:|--------------:|---------------:|-----------------:|------------------:|-------------------:|-----------:|-----------:|-----------:|-------------:|:-----------|-----------:|------------:|-----------:|---------:|:------------------------------------------------------|-----------:|------------:|--------------:|---------------:|-------------:|-------------:|-------------:|-----------------:|------------:|---------------:|-------------------:|-------:|-----------:|---------:|-------------:|-------:|---------:|-------:|----------------:|------------:|---------:|----------:|-----------:|------------:|----------------:|:-----------------------|------------:|:--------------------------|:-------------|--------:|----------:|------------:|--------------:|------------------:|:------------------|----------:|-------------:|---------:|----------------:|-----------:|------------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|--------:|------:|------:|------:|
| count  |  2114 |         0 |         0 |          0 |          0 |            0 |         0 |           0 |      0 |       0 |         0 |            0 |        2114 |           0 |           0 |            0 |            0 |            0 |               0 |              0 |           0 |            0 |              0 |            0 |              0 |              0 |                 0 |                 0 |             0 |                      0 |                       0 |                      0 |               0 |  2114    | 2114     | 2114      |                      0 | 2114  |  2114       | 2114        |     2114     | 2099         | 2114                     |           0 | 607         | 2114                                                                                      |       2114 |             0 |         0 |         0 |          2114 |       2114     |             2114 |              2114 |               2114 |       2114 |          0 |       2114 |      2114    | 2114       |          0 |        2114 |          0 |        0 | 2114                                                  |          0 |           0 |             0 |              0 |            0 |         2114 |            0 |                0 |        2114 |              0 |                  0 |      0 |          0 |        0 |            0 |      0 |     2114 |   2114 |  2114           |        2114 |        0 |         0 |          0 |           0 |               0 | 2114                   |           0 | 2114                      | 1693         |       0 |         0 |           0 |          2114 |                 0 | 2114              |         0 |            0 |        0 |               0 | 2114       | 2114        |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |  2114 |    2114 |  2114 |  2114 |  2114 |
| unique |   nan |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |         nan |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |   nan    | 2114     | 1         |                    nan | 1761  |   nan       | 18          |      nan     | 157          | 2101                     |         nan | 2           | 2025                                                                                      |        nan |           nan |       nan |       nan |           nan |        nan     |              nan |               nan |                nan |        nan |        nan |        nan |       nan    | 2          |        nan |         nan |        nan |      nan | 2114                                                  |        nan |         nan |           nan |            nan |          nan |          nan |          nan |              nan |         nan |            nan |                nan |    nan |        nan |      nan |          nan |    nan |      nan |    nan |   nan           |         nan |      nan |       nan |        nan |         nan |             nan | 2102                   |         nan | 1                         | 1            |     nan |       nan |         nan |           nan |               nan | 14                |       nan |          nan |      nan |             nan |  nan       |  nan        |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |     nan |   nan |   nan |   nan |
| top    |   nan |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |         nan |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |   nan    | 역삼아레나빌딩  | 1522-3232 |                    nan | --    |   nan       | 서울          |      nan     | 강남구          | 인천광역시 중구 운서동 2868 제2합동청사 |         nan | Y           | Z9999@T01@T05@T08@T16@T17@T20@T21@T30@T32@T34@T36@T43@T51@T52@T56@T58@T65@T69@T71@P90@T09 |        nan |           nan |       nan |       nan |           nan |        nan     |              nan |               nan |                nan |        nan |        nan |        nan |       nan    | N          |        nan |         nan |        nan |      nan | /upload/store/2019/06/[3762]_20190612101444_j4839.JPG |        nan |         nan |           nan |            nan |          nan |          nan |          nan |              nan |         nan |            nan |                nan |    nan |        nan |      nan |          nan |    nan |      nan |    nan |   nan           |         nan |      nan |       nan |        nan |         nan |             nan | 경기도 하남시 미사대로 750 (신장동) |         nan | N                         | WHCROAD      |     nan |       nan |         nan |           nan |               nan | A01               |       nan |          nan |      nan |             nan |  nan       |  nan        |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |     nan |   nan |   nan |   nan |
| freq   |   nan |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |         nan |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |   nan    | 1        | 2114      |                    nan | 349   |   nan       | 675         |      nan     | 102          | 3                        |         nan | 439         | 5                                                                                         |        nan |           nan |       nan |       nan |           nan |        nan     |              nan |               nan |                nan |        nan |        nan |        nan |       nan    | 2113       |        nan |         nan |        nan |      nan | 1                                                     |        nan |         nan |           nan |            nan |          nan |          nan |          nan |              nan |         nan |            nan |                nan |    nan |        nan |      nan |          nan |    nan |      nan |    nan |   nan           |         nan |      nan |       nan |        nan |         nan |             nan | 3                      |         nan | 2114                      | 1693         |     nan |       nan |         nan |           nan |               nan | 675               |       nan |          nan |      nan |             nan |  nan       |  nan        |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |     nan |   nan |   nan |   nan |
| mean   |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |  1412.4  | nan      | nan       |                    nan | nan   |     5.76632 | nan         |      589.451 | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |        106.971 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |      5226.67 | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |     2.01792e+07 |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |   36.7901  |  127.404    |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
| std    |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |   680.43 | nan      | nan       |                    nan | nan   |     4.36126 | nan         |      437.28  | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |        125.156 |                0 |                 0 |                  0 |          0 |        nan |          0 |      2474.69 | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        0 |      0 | 51976.5         |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |    1.03003 |    0.799346 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
| min    |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |     3    | nan      | nan       |                    nan | nan   |     1       | nan         |      101     | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |          0.06  |               -1 |                -1 |                 -1 |         30 |        nan |          0 |      3001    | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |     1.99907e+07 |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |   33.2068  |  126.24     |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
| 25%    |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |   865.25 | nan      | nan       |                    nan | nan   |     1       | nan         |      119     | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |         10.985 |               -1 |                -1 |                 -1 |         30 |        nan |          0 |      3602.25 | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |     2.01412e+07 |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |   35.8857  |  126.926    |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
| 50%    |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |  1458.5  | nan      | nan       |                    nan | nan   |     5       | nan         |      516     | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |         31.16  |               -1 |                -1 |                 -1 |         30 |        nan |          0 |      4151.5  | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |     2.01901e+07 |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |   37.3843  |  127.059    |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
| 75%    |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |  1992.75 | nan      | nan       |                    nan | nan   |     8       | nan         |      835     | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |        232.92  |               -1 |                -1 |                 -1 |         30 |        nan |          0 |      4710.75 | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |     2.02206e+07 |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |   37.5383  |  127.452    |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |
| max    |     0 |       nan |       nan |        nan |        nan |          nan |       nan |         nan |    nan |     nan |       nan |          nan |           0 |         nan |         nan |          nan |          nan |          nan |             nan |            nan |         nan |          nan |            nan |          nan |            nan |            nan |               nan |               nan |           nan |                    nan |                     nan |                    nan |             nan |  2523    | nan      | nan       |                    nan | nan   |    17       | nan         |     1701     | nan          | nan                      |         nan | nan         | nan                                                                                       |          0 |           nan |       nan |       nan |             0 |        488.81  |               -1 |                -1 |                 -1 |         30 |        nan |          0 |      9999    | nan        |        nan |           0 |        nan |      nan | nan                                                   |        nan |         nan |           nan |            nan |          nan |            0 |          nan |              nan |           0 |            nan |                nan |    nan |        nan |      nan |          nan |    nan |        1 |     60 |     2.02602e+07 |           0 |      nan |       nan |        nan |         nan |             nan | nan                    |         nan | nan                       | nan          |     nan |       nan |         nan |             0 |               nan | nan               |       nan |          nan |      nan |             nan |   38.2133  |  129.455    |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |       0 |     0 |     0 |     0 |

### 결측치 및 단일 값 컬럼 제거

#### 제거 전 결측치 비율

```
seq                          0.000000
sido_cd                    100.000000
sido_nm                    100.000000
gugun_cd                   100.000000
gugun_nm                   100.000000
code_order                 100.000000
view_yn                    100.000000
store_num                  100.000000
sido                       100.000000
gugun                      100.000000
address                    100.000000
new_img_nm                 100.000000
p_pro_seq                    0.000000
p_view_yn                  100.000000
p_sido_cd                  100.000000
p_gugun_cd                 100.000000
p_store_nm                 100.000000
p_theme_cd                 100.000000
p_wireless_yn              100.000000
p_smoking_yn               100.000000
p_book_yn                  100.000000
p_music_yn                 100.000000
p_terrace_yn               100.000000
p_table_yn                 100.000000
p_takeout_yn               100.000000
p_parking_yn               100.000000
p_dollar_assent            100.000000
p_card_recharge            100.000000
p_subway_yn                100.000000
stb_store_file_renew       100.000000
stb_store_theme_renew      100.000000
stb_store_time_renew       100.000000
stb_store_lsm              100.000000
s_code                       0.000000
s_name                       0.000000
tel                          0.000000
dlvry_call_cntr_phno       100.000000
fax                          0.000000
sido_code                    0.000000
sido_name                    0.000000
gugun_code                   0.000000
gugun_name                   0.709555
addr                         0.000000
park_info                  100.000000
new_state                   71.286660
theme_state                  0.000000
new_bool                     0.000000
search_text                100.000000
ins_lat                    100.000000
ins_lng                    100.000000
in_distance                  0.000000
out_distance                 0.000000
all_search_cnt               0.000000
addr_search_cnt              0.000000
store_search_cnt             0.000000
rowCount                     0.000000
store_nm                   100.000000
store_cd                     0.000000
s_biz_code                   0.000000
new_icon                     0.000000
set_user                   100.000000
favorites                    0.000000
map_desc                   100.000000
notice                     100.000000
defaultimage                 0.000000
etcimage                   100.000000
in_biz_cd                  100.000000
in_store_cd                100.000000
in_favorites               100.000000
in_user_id                 100.000000
in_biz_cds                   0.000000
in_biz_arr                 100.000000
in_biz_arrdata             100.000000
in_scodes                    0.000000
in_scode_arr               100.000000
in_scode_arrdata           100.000000
disp                       100.000000
set_date                   100.000000
hlytag                     100.000000
hlytag_msg                 100.000000
vSal                       100.000000
istart                       0.000000
iend                         0.000000
open_dt                      0.000000
gold_card                    0.000000
ip_lat                     100.000000
ip_long                    100.000000
espresso                   100.000000
new_store                  100.000000
premiere_food              100.000000
doro_address                 0.000000
cold_blew                  100.000000
my_siren_order_store_yn      0.000000
whcroad_yn                  19.914853
skuNo                      100.000000
skuName                    100.000000
skuImgUrl                  100.000000
stock_count                  0.000000
store_area_name            100.000000
store_area_code              0.000000
is_open                    100.000000
ev_open_yn                 100.000000
option                     100.000000
gift_stock_yn              100.000000
lat                          0.000000
lot                          0.000000
t20                          0.000000
t04                          0.000000
t03                          0.000000
t01                          0.000000
t12                          0.000000
t09                          0.000000
t06                          0.000000
t10                          0.000000
p10                          0.000000
p50                          0.000000
p20                          0.000000
p60                          0.000000
p30                          0.000000
p70                          0.000000
p40                          0.000000
p80                          0.000000
t22                          0.000000
t21                          0.000000
p90                          0.000000
p01                          0.000000
t05                          0.000000
t30                          0.000000
t36                          0.000000
t27                          0.000000
t29                          0.000000
t43                          0.000000
t48                          0.000000
z9999                        0.000000
t64                          0.000000
t66                          0.000000
p02                          0.000000
```

#### 제거된 컬럼: `['seq', 'sido_cd', 'sido_nm', 'gugun_cd', 'gugun_nm', 'code_order', 'view_yn', 'store_num', 'sido', 'gugun', 'address', 'new_img_nm', 'p_view_yn', 'p_sido_cd', 'p_gugun_cd', 'p_store_nm', 'p_theme_cd', 'p_wireless_yn', 'p_smoking_yn', 'p_book_yn', 'p_music_yn', 'p_terrace_yn', 'p_table_yn', 'p_takeout_yn', 'p_parking_yn', 'p_dollar_assent', 'p_card_recharge', 'p_subway_yn', 'stb_store_file_renew', 'stb_store_theme_renew', 'stb_store_time_renew', 'stb_store_lsm', 'dlvry_call_cntr_phno', 'park_info', 'new_bool', 'search_text', 'ins_lat', 'ins_lng', 'in_distance', 'all_search_cnt', 'addr_search_cnt', 'store_search_cnt', 'rowCount', 'store_nm', 'store_cd', 'set_user', 'favorites', 'map_desc', 'notice', 'etcimage', 'in_biz_cd', 'in_store_cd', 'in_favorites', 'in_user_id', 'in_biz_cds', 'in_biz_arr', 'in_biz_arrdata', 'in_scodes', 'in_scode_arr', 'in_scode_arrdata', 'disp', 'set_date', 'hlytag', 'hlytag_msg', 'vSal', 'istart', 'iend', 'gold_card', 'ip_lat', 'ip_long', 'espresso', 'new_store', 'premiere_food', 'cold_blew', 'my_siren_order_store_yn', 'whcroad_yn', 'skuNo', 'skuName', 'skuImgUrl', 'stock_count', 'store_area_name', 'is_open', 'ev_open_yn', 'option', 'gift_stock_yn', 'z9999']`

#### 제거 후 데이터 구조

|    |   p_pro_seq |   s_code | s_name   | tel       | fax          |   sido_code | sido_name   |   gugun_code | gugun_name   | addr                       |   new_state | theme_state                                                                          |   out_distance |   s_biz_code | new_icon   | defaultimage                                          |   open_dt | doro_address               | store_area_code   |     lat |     lot |   t20 |   t04 |   t03 |   t01 |   t12 |   t09 |   t06 |   t10 |   p10 |   p50 |   p20 |   p60 |   p30 |   p70 |   p40 |   p80 |   t22 |   t21 |   p90 |   p01 |   t05 |   t30 |   t36 |   t27 |   t29 |   t43 |   t48 |   t64 |   t66 |   p02 |
|---:|------------:|---------:|:---------|:----------|:-------------|------------:|:------------|-------------:|:-------------|:---------------------------|------------:|:-------------------------------------------------------------------------------------|---------------:|-------------:|:-----------|:------------------------------------------------------|----------:|:---------------------------|:------------------|--------:|--------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
|  0 |           0 |     1509 | 역삼아레나빌딩  | 1522-3232 | 02-568-3763  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 역삼동 721-13 아레나빌딩 |         nan | Z9999@T05@T08@T16@T17@T20@T21@T30@T32@@@T36@T43@T52@T56@T57@T64@T65@T69@T71@P80@P90  |           9.26 |         3762 | N          | /upload/store/2019/06/[3762]_20190612101444_j4839.JPG |  20190613 | 서울특별시 강남구 언주로 425 (역삼동)    | A01               | 37.5011 | 127.043 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |
|  1 |           0 |     1434 | 논현역사거리   | 1522-3232 | 02-3442-3673 |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 논현동 142-2 정일빌딩   |         nan | Z9999@T05@T08@T16@T17@T20@T21@T22@T30@T32@T36@@T52@T56@T57@T64@T65@T69@P70@P80@P90   |           7.38 |         3672 | N          | /upload/store/2025/07/3672_20250709034455_lnuoo.jpg   |  20181123 | 서울특별시 강남구 강남대로 538 (논현동)   | A01               | 37.5102 | 127.022 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |
|  2 |           0 |     1595 | 신사역성일빌딩  | 1522-3232 | 02-547-3859  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 논현동 18-4 성일빌딩    |         nan | Z9999@T05@T08@T16@T17@T20@T21@T30@T32@T34@@@@T52@T56@T57@T64@T65@T68@T69@T73@P80@P90 |           6.95 |         3858 | N          | /upload/store/2022/01/[3858]_20220127064243_5khni.jpg |  20191219 | 서울특별시 강남구 강남대로 584 (논현동)   | A01               | 37.5139 | 127.021 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |
|  3 |           0 |     1527 | 국기원사거리   | 1522-3232 | 02-568-3669  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 역삼동 648-22 동찬빌딩  |         nan | Z9999@T05@T08@T16@T17@T20@T21@T30@T32@T34@@@T36@T52@T56@T57@T64@T65@T69@P70@P90      |           8.82 |         3669 | N          | /upload/store/2019/07/[3669]_20190730073527_5nk40.jpg |  20190731 | 서울특별시 강남구 테헤란로 125 (역삼동)   | A01               | 37.4995 | 127.031 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |
|  4 |           0 |     1468 | 대치재경빌딩   | 1522-3232 | 02-568-3705  |           1 | 서울          |          101 | 강남구          | 서울특별시 강남구 대치동 599 대원빌딩     |         nan | Z9999@@T05@T08@T16@T17@T20@T21@T22@@@T30@T32@@@T43@T52@T56@T57@T64@T65@T69@T71@P90   |          10.92 |         3703 | N          | /upload/store/2023/12/[3703]_20231217121347_eqnky.jpg |  20190214 | 서울특별시 강남구 남부순환로 2947 (대치동) | A01               | 37.4947 | 127.063 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |

## 1. 기본 정보 요약

주요 필드(매장명, 시/도, 시/군/구, 주소, 오픈일) 요약

|    | s_name   | sido_name   | gugun_name   | doro_address               | open_dt             |     lat |     lot |
|---:|:---------|:------------|:-------------|:---------------------------|:--------------------|--------:|--------:|
|  0 | 역삼아레나빌딩  | 서울          | 강남구          | 서울특별시 강남구 언주로 425 (역삼동)    | 2019-06-13 00:00:00 | 37.5011 | 127.043 |
|  1 | 논현역사거리   | 서울          | 강남구          | 서울특별시 강남구 강남대로 538 (논현동)   | 2018-11-23 00:00:00 | 37.5102 | 127.022 |
|  2 | 신사역성일빌딩  | 서울          | 강남구          | 서울특별시 강남구 강남대로 584 (논현동)   | 2019-12-19 00:00:00 | 37.5139 | 127.021 |
|  3 | 국기원사거리   | 서울          | 강남구          | 서울특별시 강남구 테헤란로 125 (역삼동)   | 2019-07-31 00:00:00 | 37.4995 | 127.031 |
|  4 | 대치재경빌딩   | 서울          | 강남구          | 서울특별시 강남구 남부순환로 2947 (대치동) | 2019-02-14 00:00:00 | 37.4947 | 127.063 |

## 2. 매장 특성 분석 (theme_state)

### 매장별 제공 서비스/특징 (상위 10개)

|    | 특징   |   매장 수 |
|---:|:-----|-------:|
|  0 | T30  |   2103 |
|  1 | T20  |   2102 |
|  2 | T17  |   2100 |
|  3 | T05  |   2082 |
|  4 | T65  |   2069 |
|  5 | T16  |   2058 |
|  6 | T08  |   2048 |
|  7 | T32  |   2038 |
|  8 | T56  |   1953 |
|  9 | T52  |   1859 |

### 상위 10개 매장 서비스/특징

![상위 10개 매장 서비스/특징](images\top10_features.png)

|    | 특징   |   매장 수 |
|---:|:-----|-------:|
|  0 | T30  |   2103 |
|  1 | T20  |   2102 |
|  2 | T17  |   2100 |
|  3 | T05  |   2082 |
|  4 | T65  |   2069 |
|  5 | T16  |   2058 |
|  6 | T08  |   2048 |
|  7 | T32  |   2038 |
|  8 | T56  |   1953 |
|  9 | T52  |   1859 |

## 3. 주변 위치적 특징 분석

### 시/도별 매장 분포

### 시/도별 매장 분포

![시/도별 매장 분포](images\stores_by_sido.png)

| sido_name   |   count |
|:------------|--------:|
| 서울          |     675 |
| 경기          |     514 |
| 부산          |     146 |
| 대구          |      94 |
| 인천          |      92 |
| 경남          |      81 |
| 대전          |      73 |
| 광주          |      71 |
| 경북          |      64 |
| 충남          |      54 |
| 전북          |      44 |
| 충북          |      41 |
| 전남          |      37 |
| 울산          |      35 |
| 제주          |      35 |
| 강원          |      29 |
| 세종          |      15 |
| 강원도         |      14 |

### 서울시 내 구별 매장 분포 (가장 많은 시/도 기준)

### 서울 내 구별 매장 분포

![서울 내 구별 매장 분포](images\stores_by_gugun_top_sido.png)

| gugun_name   |   count |
|:-------------|--------:|
| 강남구          |     102 |
| 서초구          |      58 |
| 중구           |      54 |
| 영등포구         |      43 |
| 종로구          |      42 |
| 송파구          |      39 |
| 마포구          |      38 |
| 강서구          |      33 |
| 용산구          |      25 |
| 서대문구         |      24 |
| 광진구          |      23 |
| 강동구          |      20 |
| 성북구          |      17 |
| 양천구          |      17 |
| 은평구          |      16 |
| 금천구          |      15 |
| 노원구          |      15 |
| 성동구          |      15 |
| 관악구          |      14 |
| 동대문구         |      14 |
| 동작구          |      13 |
| 구로구          |      13 |
| 중랑구          |      10 |
| 도봉구          |       8 |
| 강북구          |       7 |

가장 매장이 많은 지역은 **서울**이며, 특히 **강남구**에 매장이 집중되어 있어 중심 상권일 가능성이 높습니다.

## 4. 오픈일 기반 통계 분석

### 연도별 매장 오픈 트렌드

### 연도별 매장 오픈 트렌드

![연도별 매장 오픈 트렌드](images\open_trend_by_year.png)

|   open_year |   count |
|------------:|--------:|
|        1999 |       1 |
|        2000 |       1 |
|        2001 |      10 |
|        2002 |      12 |
|        2003 |      10 |
|        2004 |      11 |
|        2005 |      14 |
|        2006 |      14 |
|        2007 |      21 |
|        2008 |      24 |
|        2009 |      17 |
|        2010 |      27 |
|        2011 |      55 |
|        2012 |      83 |
|        2013 |     105 |
|        2014 |     135 |
|        2015 |     118 |
|        2016 |     126 |
|        2017 |     141 |
|        2018 |     131 |
|        2019 |     140 |
|        2020 |     149 |
|        2021 |     155 |
|        2022 |     164 |
|        2023 |     149 |
|        2024 |     149 |
|        2025 |     140 |
|        2026 |      12 |

최근 연도로 올수록 매장 오픈 수가 증가하는 경향을 보입니다.

## 5. 주요 비즈니스 인사이트 및 6. 데이터 품질 검증

)
- **입지 경쟁력**: 매장은 주로 '서울', '경기' 등 수도권에 집중되어 있으며, 특히 강남구, 서초구, 중구 등 오피스 및 상업 중심지에 밀집되어 있습니다. 이는 유동인구가 많은 핵심 상권을 타겟으로 하는 전략으로 해석됩니다.
- **고객 편의성**: `t01`(무선인터넷), `t03`(테라스) 등의 편의 시설 제공 여부를 통해 고객 경험을 향상시키려는 노력을 엿볼 수 있습니다.
- **데이터 품질**:
  - `open_dt`에서 일부 날짜 변환에 실패한 데이터(NaT)가 존재할 수 있어 확인이 필요합니다.
  - `lat`, `lot` 좌표의 경우, 대한민국 범위를 벗어나는 값이 있는지 확인하여 데이터의 신뢰성을 검증해야 합니다. (예: 위도 33-39, 경도 124-132 범위)
  - `sido_code`, `gugun_code`는 각각 `sido_name`, `gugun_name`과 일관성을 유지하는지 검증이 필요합니다.


### 좌표계 이상치 검사: 대한민국 범위 내에 모든 매장이 위치합니다.

## 7. 추가 분석 제안

)
- **경쟁사 분석**: 공공데이터포털의 타 커피 전문점 데이터를 활용하여, 특정 지역 내 스타벅스와 경쟁사의 분포를 비교 분석할 수 있습니다.
- **유동인구 데이터 결합**: 서울시 열린데이터광장의 '지하철 승하차 인원', '유동인구' 데이터와 매장 위치를 결합하여, 유동인구와 매장 수의 상관관계를 분석할 수 있습니다.
- **매장 클러스터링**: `tXX`, `pXX`와 같은 매장 특성 데이터를 기반으로 K-Means 등의 클러스터링 알고리즘을 사용하여 매장을 '업무 중심형', '주거 지역형', '대학가형' 등으로 유형화할 수 있습니다.
- **매출 예측 모델**: 매장 위치, 오픈 연도, 주변 인구 밀도, 경쟁사 수 등의 변수를 활용하여 신규 매장의 예상 매출을 예측하는 회귀 모델을 개발할 수 있습니다.


