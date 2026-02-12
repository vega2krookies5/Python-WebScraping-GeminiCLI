# Yes24 데이터 수집 명세

## 1. 과업 목표
- Yes24 웹사이트에서 특정 카테고리의 도서 정보를 수집합니다.
- 수집된 데이터는 분석 및 시각화에 활용될 수 있도록 정제된 형태로 저장합니다.
- source 코드는 yes24\yes24_scraper.py 할 것
- 수집한 데이터는 yes24\data\yes24_ai.csv 파일로 작성할 것 

## 2. 수집 관련 정보

### 네트워크 메뉴를 통해 실제 데이터를 가져오는 URL
Request URL
https://www.yes24.com/product/category/CategoryProductContents?dispNo=001001003032&order=SINDEX_ONLY&addOptionTp=0&page=2&size=24&statGbYn=N&viewMode=&_options=&directDelvYn=&usedTp=0&elemNo=0&elemSeq=0&seriesNumber=0
Request Method
GET
Status Code
200 OK

### 해당 Request에 대한 Header 정보
host
www.yes24.com
referer
https://www.yes24.com/product/category/display/001001003032
rtt
50
sec-ch-ua
"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"
sec-ch-ua-mobile
?0
sec-ch-ua-platform
"Windows"
sec-fetch-dest
empty
sec-fetch-mode
cors
sec-fetch-site
same-origin
user-agent
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36
viewport-width
898
x-requested-with
XMLHttpRequest

### Payload
dispNo=001001003032&order=SINDEX_ONLY&addOptionTp=0&page=2&size=24&statGbYn=N&viewMode=&_options=&directDelvYn=&usedTp=0&elemNo=0&elemSeq=0&seriesNumber=0

### 응답 예시 (HTML, JSON 의 일부 정보)
<div class="itemUnit">
        <div class="item_img">
            <div class="img_canvas">
                
                <span class="img_item">
                    <span class="img_grp">
                        
                                                                                                <a href="/product/goods/146036009" class="lnk_img" onclick=" ">
                            <em class="img_bdr">
                                <img class="lazy" data-original="https://image.yes24.com/goods/146036009/L" src="https://image.yes24.com/goods/146036009/L" border="0" alt="Do it! LLM을 활용한 AI 에이전트 개발 입문" style="display: inline;">
                            </em>
                        </a>
                    </span>
                </span>
            </div>
            <div class="img_btn">
                
                        <a href="javascript:yes24GU.openPreviewCheck(146036009); " class="btnC btn_preview"><span class="bWrap"><em class="txt">미리보기</em></span></a>
                                            </div>
        </div>
        <div class="item_info">
            
                <div class="info_row info_keynote">




    <span id="spanGdKeynote" class="gd_keynote">
                

                                            <span class="iconC spring"><em class="txt">분철</em></span>
                                <a href="javascript:void(0);" onclick="openUrl('/product/category/series/001001003032?SeriesNumber=148352','Pcode','003_001')" class="lnk_series">이지스퍼블리싱-Do it! 시리즈</a>
        <!-- 클래스24 상품일 경우 -->
    </span>
    <script type="text/javascript">
        if ($('#spanGdKeynote').children().length == 0) {
            $('#spanGdKeynote').remove();
        }
    </script>


                </div>
            <div class="info_row info_name">
                
                                <a class="gd_name" href="/product/goods/146036009" onclick=" ">Do it! LLM을 활용한 AI 에이전트 개발 입문</a>
                
                    <span class="gd_nameE">GPT API+딥시크+라마+랭체인+랭그래프+RAG</span>
                                <a href="/product/goods/146036009" target="_blank" class="bgYUI ico_nWin" onclick=" ">Do it! LLM을 활용한 AI 에이전트 개발 입문 새창이동</a>
            </div>
                <div class="info_row info_pubGrp">
                    
                            <span class="authPub info_auth" onclick="">
                                <a href="https://www.yes24.com/product/search?domain=ALL&amp;query=%25EC%259D%25B4%25EC%2584%25B1%25EC%259A%25A9&amp;authorNo=325447&amp;author=이성용" target="_blank">이성용</a> 저
                            </span>
                    
                        <span class="authPub info_pub" onclick=""><a href="https://www.yes24.com/product/search?&amp;domain=ALL&amp;company=%ec%9d%b4%ec%a7%80%ec%8a%a4%ed%8d%bc%eb%b8%94%eb%a6%ac%ec%8b%b1&amp;query=%25EC%259D%25B4%25EC%25A7%2580%25EC%258A%25A4%25ED%258D%25BC%25EB%25B8%2594%25EB%25A6%25AC%25EC%258B%25B1">이지스퍼블리싱</a></span>
                                            <span class="authPub info_date">2025년 05월</span>
                </div>
                                                        <div class="info_row info_price">
                            <span class="txt_sale"><em class="num">10</em>%</span>
                        <strong class="txt_num"><em class="yes_b">31,500</em>원</strong>
                        <span class="txt_num dash"><em class="yes_m">35,000</em>원</span>
                                                    <span class="yPoint"><em class="bgYUI ico_point">포인트적립</em>1,750원</span>
                    </div>
                            <div class="info_row info_rating ">
                            <span class="saleNum">
                                판매지수 19,311
                            </span>
                                            <span class="rating_rvCount">
                            <a href="https://www.yes24.com/product/goods/146036009?ReviewYn=Y" onclick=""><em class="bit">회원리뷰</em>(<em class="txC_blue">21</em>건)</a>
                        </span>
                        <span class="rating_grade">
                            <span class="bgYUI tRating tRating_10">리뷰 총점</span><em class="yes_b">9.7</em>
                            <span class="moreRatingArea">
                                <span class="moreRatingBtn">
                                        <a href="javascript:void(0);" onclick="toggleLiCont(this,$('.sGLi'),event);" class="bgYUI">정보 더 보기/감추기</a>
                                </span>
                                <span class="moreRatingLi">
                                    <span class="moreRatingLiRow">
                                        <ul class="yesAlertLi">
                                            <li><em class="bl_dot bgYUI">&nbsp;</em>종이책 리뷰 (12건)</li>
                                            <li><em class="bl_dot bgYUI">&nbsp;</em>eBook 리뷰 (1건)</li>
                                            <li><em class="bl_dot bgYUI">&nbsp;</em>종이책 한줄평 (6건)</li>
                                            <li><em class="bl_dot bgYUI">&nbsp;</em>eBook 한줄평 (2건)</li>
                                        </ul>
                                    </span>
                                </span>
                            </span>
                        </span>
                </div>

                    <div class="info_row info_deli" name="delvTextArea"><span class="deli_des">21시까지 주문하면 </span><span class="deli_date"><strong class="deli_act">내일 아침 7시 전 (2/13, 금)</strong> 도착예정</span></div>

            
                                        <div class="info_row info_spring">
                    분철서비스 이용이 가능한 도서입니다.
                    <a href="https://www.yes24.com/campaign/01_book/2020/0304File.aspx" target="_blank" class="btnC s_size"><span class="bWrap"><em class="txt">자세히 보기</em><em class="bgYUI ico_goS"></em></span></a>
                </div>
                                                                                            <div class="info_row info_tag">
                                <span class="tag">
                                    <a href="https://www.yes24.com/product/search?domain=ALL&amp;query=%25ED%2581%25AC%25EB%25A0%2588%25EB%25A7%2588%25ED%2581%25B4%25EB%259F%25BD%25EC%2597%2590%25EC%259E%2588%25EC%2596%25B4%25EC%259A%2594&amp;hashNm=%ed%81%ac%eb%a0%88%eb%a7%88%ed%81%b4%eb%9f%bd%ec%97%90%ec%9e%88%ec%96%b4%ec%9a%94" onclick=" setTagClickExtraCode('100', '크레마클럽에있어요', '146036009','5392');">#크레마클럽에있어요</a>
                                </span>
                                <span class="tag">
                                    <a href="https://www.yes24.com/product/search?domain=ALL&amp;query=%25EB%25B6%2584%25EC%25B2%25A0&amp;hashNm=%eb%b6%84%ec%b2%a0" onclick=" setTagClickExtraCode('100', '분철', '146036009','2226');">#분철</a>
                                </span>
                                <span class="tag">
                                    <a href="https://www.yes24.com/product/search?domain=ALL&amp;query=%25ED%258C%258C%25EC%259D%25B4%25EC%258D%25AC&amp;hashNm=%ed%8c%8c%ec%9d%b4%ec%8d%ac" onclick=" setTagClickExtraCode('100', '파이썬', '146036009','826');">#파이썬</a>
                                </span>
                                <span class="tag">
                                    <a href="https://www.yes24.com/product/search?domain=ALL&amp;query=%25EC%25B1%2597GPT&amp;hashNm=%ec%b1%97GPT" onclick=" setTagClickExtraCode('100', '챗GPT', '146036009','6421');">#챗GPT</a>
                                </span>
                                <span class="tag">
                                    <a href="https://www.yes24.com/product/search?domain=ALL&amp;query=LLM&amp;hashNm=LLM" onclick=" setTagClickExtraCode('100', 'LLM', '146036009','6951');">#LLM</a>
                                </span>
                    </div>

            
                    <div class="info_row info_read">
                        GPT API를 활용한 업무 자동화부터 랭체인과 랭그래프를 활용한 멀티에이전트 개발까지!한 권으로 끝내는 AI 에이전트 개발 입문! AI가 모두의 일상을 바꾸고 있는 지금, AI 기술을 제대로 이해하고 활용하는 방법을...
                    </div>
                                        <div class="info_row info_event">
                                <div class="event">
                                        <span class="iconC freeD"><em class="txt">이벤트</em></span>
                                                                                                            <span class="iconC"><em class="txt">선착순</em></span>
<span class="iconC"><em class="txt">사은품</em></span>

                                    <span class="txt"><a href="https://event.yes24.com/template?EventNo=266882">월간 개발자 2026년 2월호</a></span>
                                    <span class="date">(26.01.30 ~ 26.02.28)</span>
                                </div>
                            </div>


            
                                        <div class="info_row info_relG">
                            관련상품 :
                            <span class="relG"><a href="https://www.yes24.com/Product/UsedShopHub/Hub/146036009">중고상품 <span class="relG_num">22개</span></a></span>
                                                <span class="relG"><a href="/product/goods/147123595">eBook <span class="relG_num">25,000원</span></a></span>
                </div>
                    </div>
            <div class="item_btnCol">
                
                


            <span class="btn_row">
                <span class="chkBox" style=""><label><input type="checkbox" name="ORD_GOODS_CHKBOX" id="ordChk_146036009" data-goodsno="146036009" class="basic" style=""><span class="bgYUI chk"></span></label></span>
                    <span class="numBox">
                        <span class="yesIpt ipt_wSizeF">
                            <input type="text" name="ORD_GOODS_CNT" id="ordCnt_146036009" style="ime-mode:disabled !important" title="수량설정" value="1" class="ac yes_m" onkeyup="return checkNumeric(this);" maxlength="3">
                        </span>
                        <button type="button" class="minus" onclick="order_payment.downOrderCount('146036009'); "><span class="bgYUI">수량감소</span></button>
                        <button type="button" class="plus" onclick="order_payment.upOrderCount('146036009'); "><span class="bgYUI">수량증가</span></button>
                    </span>
            </span>
                <a href="javascript:void(0);" onclick="order_payment.addCartV3('146036009', '', this, '', '', '', '', 'Search', true); " class="btnC btn_blue"><span class="bWrap"><em class="txt">카트에 넣기</em></span></a>
            <a href="javascript:void(0);" onclick="order_payment.orderDirectV3('146036009', '', this, '', '', '', '', 'Search'); " class="btnC btn_sBlue"><span class="bWrap"><em class="txt">바로구매</em></span></a>
        <a href="javascript:void(0);" class="btnC" name="btnList" onclick="order_payment.addMyListV3('146036009', '', '', true, ''); ;"><span class="bWrap"><em class="txt">리스트에 넣기</em></span></a>
<input type="hidden" name="ORD_GOODS_OPT" id="ordOpt_146036009" value="{&quot;goods_no&quot;:146036009,&quot;goods_seq&quot;:1,&quot;order_limit_yn&quot;:&quot;N&quot;,&quot;order_remain_count&quot;:0,&quot;event_no&quot;:0,&quot;add_cart_yn&quot;:&quot;Y&quot;,&quot;goods_state&quot;:&quot;02&quot;,&quot;order_limit_count&quot;:0,&quot;resource_key&quot;:&quot;01&quot;,&quot;limit_age_yn&quot;:&quot;N&quot;,&quot;limit_age&quot;:0,&quot;member_age&quot;:0,&quot;goods_name&quot;:&quot;Do it! LLM을 활용한 AI 에이전트 개발 입문&quot;,&quot;noint_quotamonth&quot;:0,&quot;min_cnt&quot;:0,&quot;max_cnt&quot;:0,&quot;opt_salepr&quot;:0,&quot;opt_yn&quot;:&quot;N&quot;,&quot;opt_inst_yn&quot;:&quot;N&quot;,&quot;flat_rate_yn&quot;:null,&quot;rent_goods_yn&quot;:&quot;N&quot;,&quot;bookclue_yn&quot;:&quot;N&quot;,&quot;goods_gb&quot;:&quot;01&quot;,&quot;goodsSortNo&quot;:&quot;001002&quot;,&quot;goodsSortNm&quot;:&quot;IT 모바일&quot;,&quot;goodsAuth&quot;:&quot;&lt;이성용&gt; 저&quot;,&quot;shopPrice&quot;:35000.00,&quot;salePrice&quot;:31500.00,&quot;discountShopPrice&quot;:3500.00}">

            </div>
    </div>