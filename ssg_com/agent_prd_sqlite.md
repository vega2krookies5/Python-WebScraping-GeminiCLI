## 예시 데이터의 일부

itemId,itemNm,uitemId,displayPrc,displayPrcYn,strikeOutPrc,minOnetOrdPsblQty,maxOnetOrdPsblQty,soldOutYn,stoppedSellingYn,siteNo,siteNm,salestrNo,salestrNm,sellpntNm,brandId,brandNm,brandEngNm,goItemDetailYn,goItemDetailYnNew,msgWhenGoToItemDetail,msgWhenGoToItemDetailNew,attnTgtIdnfNo1,attnTgtIdnfNo2,attnDivCd,needAdultCertification,itemLnkd,itemImgUrl,itemImgUrlList,imgSrchPsblYn,planDt,dataFileNm,pickuItemYn,qshppPsblYn,mltSellQty,itemOrdQty,itemOrdQtyType,tildeDispYn,recompPoint,recomRegCnt,sellUnitPrc,sellCapaUnitNmWithUnitCapa,replaceBtnDispYn,arrivalNotiBtnDispYn,advertAcctId,advertBidId,advertExtensTeryDivCd,advertBilngTypeCd,advertKindCd,priorAdvertAcctGrpId,molocoImpTrackers,molocoClickTrackers,cartPsblType,fundingInfo,preorderInfo,itemcTitle,itemcText,postngId,itemBadgList,abstExpsrTypeNm,flucType,flucOrdr,dptsFdshopOperYn,itemKindTypeNm,bothSsgMorningShppData,bothShppData,modelInfo,dispRankOrdr,itemChrctDivCd,benefitGrp0,benefitGrp1,benefitGrp2,benefitGrp3,benefitGrp4,benefitGrp5,itemTagNmAndItemCnt,replaceSellPrc,giftBtnShowType,giftBtnActType,giftBtnMsg,dealItemList,dealItemYn,firstPrc,prordPsblYn,lowItemPrc,originItemInfoList,itemAddInfo,shppcstInfo,oreItemId,omItemStatus,spclMallItemStatus,reviewAddtInfoData,diAdvertType,oreItemMallNm,adultItemAddImgYn,modifaceYn,certifyStatus,deliTypeNm,ssgInstallStatus,shppInfoLnkd,cartBtnYn,itemImgType,festaInfo,monthUsgFeeTxt,rentalItemYn,frebieImgUrl,uitemOptnNm,advertYn,advertMarkYn,itemVodDispYn,shppTypeDtlCd,bsplItemDivCd,adultItemTypeCd,usablInvQty,giftPackTypeCd,giftPackTypeNm,rstctInvQty,originItemNm,colorchipData,bndlInfoExpsrYn,dispStrtDts,dispEndDts,etcSellPrcNm,etcSellPrc,cartItemLnkd,commId,commType,itemRegDivCd,itemTagLnkd,certifyMark,svcTypeNm,maiTitleNm1,maiTitleNm2,subtitlNm1,subtitlNm2,mdMsgCntt,mdMsgCntt2,iconTagList,dispPiconImgFileNm,dispPiconCntxCntt,advertBadgeToolTip,soldOutDispTxt,mlinkUrl,benefitGrpSeparately,specialPriceItemImgFileNm,itemOrdQtyTxt
1000584541946,47년전통 전국4대떡집 쑥인절미2KG,00000,26010,Y,28900,1,10,N,N,6004,,6005,S.COM몰,,3000076957,남해중현떡집,Junghyun,Y,Y,"이 상품은 옵션이 있는 상품 입니다.
상품상세에서 옵션을 선택해주세요.","이 상품은 옵션이 있는 상품 입니다.

## sqlite 데이터베이스의 테이블로 저장
- 데이터베이스 초기화는 데이터베이스가 없을 때만 할 것, 데이터베이스 파일이 .py파일과 같은 경로에 생성되게 할 것
위 데이터를 참고해서 페이지마다 가져오는 데이터를 sqlite 데이터베이스의 테이블로 저장할 수 있게 할 것 
페이지 단위로 저장해서 중간에 연결이 끊기거나 수집을 멈추더라도 수집한 데이터까지는 저장이 되게 할 것

## 로그
수집하고 있는 과정을 로그로 남길것, 로그는 너무 자세히 남기지 말고 수집하는 url과, itemNm, 몇번째 페이지까지 수집했는지 정도만 남길것 

# SSG front API에서 상품 데이터를 수집하는 파이썬 코드를 작성해 주세요.

요구사항:
1. 요청 정보
ssg_com/agent_prd_scraping.md 파일을 참고
전체 페이지를 모두 수집하지 말고 3페이지만 먼저 수집해 주세요.

2. 응답 구조
   - 응답 JSON의 data.areaList 하위에 여러 area가 있고,
     각 area의 itemList에 상품 정보가 dict 형태로 들어 있습니다.
   - areaList는 리스트 안에 리스트가 중첩된 구조일 수 있으므로,
     이를 고려해서 모든 itemList의 item들을 평탄화 하여 수집해 주세요.

3. 데이터 처리
   - 모든 item dict를 수집한 후, pandas.json_normalize로 DataFrame을 생성합니다.
   - 중첩된 dict는 '필드명.하위필드명' 형태의 컬럼으로 풀어 주세요.
   - list 타입 필드는 json 문자열로 저장해 주세요.

4. DB 저장
   - sqlite DB 파일명: ssg_items.db
   - 테이블명: items
   - 기본 키: itemId(TEXT)
   - 주요 컬럼: itemId(TEXT), itemNm(TEXT), displayPrc(INTEGER), itemLnkd(TEXT),
                itemImgUrl(TEXT), salestrNo(TEXT), siteNo(TEXT)
   - 그 외 필드는 TEXT로 저장해도 됩니다.
   - itemId를 기준으로 이미 존재하면 INSERT를 무시하고, 없으면 INSERT 하는 방식(UPSERT)으로 구현해 주세요.

5. 코드 구조
   - fetch_items(page: int) -> list[dict]
   - items_to_df(items: list[dict]) -> pd.DataFrame
   - save_df_to_db(df: pd.DataFrame, conn: sqlite3.Connection) 함수로 분리해 주세요.
   - main 블록에서 1~5 페이지를 반복 호출하여 데이터를 수집하고 DB에 저장합니다.

6. 기타
   - requests, pandas, sqlite3 외의 서드파티 라이브러리는 사용하지 마세요.
   - 네트워크 오류나 응답 코드가 200이 아닐 경우 최대 3번 재시도하고,
     실패 시 해당 페이지를 건너뛰면서 경고 로그를 출력해 주세요.


## 데이터 수집 후 페이지마다 sqlite DB로 저장하는 새로운 .py 파일 만들기