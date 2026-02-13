### 네트워크 메뉴를 통해 실제 데이터를 가져오는 URL
Request URL
https://frontapi.ssg.com/dp/api/v2/page/area
Request Method
POST

 
### 해당 Request에 대한 Header 정보
referer
https://www.ssg.com/page/pc/SpecialPrice.ssg
sec-ch-ua
"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"
sec-ch-ua-mobile
?0
sec-ch-ua-platform
"macOS"
sec-fetch-dest
empty
sec-fetch-mode
cors
sec-fetch-site
same-site
user-agent
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.3


### Payload
{"common":{"aplVer":"","osCd":"","ts":"20251118062034","mobilAppNo":"99","dispDvicDivCd":"10","viewSiteNo":"6005"},"params":{"page":"5","state":"{}","pageId":"100000007533","pageSetId":"2","dispCtgId":null,"pageCmptId":"4"}}


### 응답 예시 (JSON 의 일부 정보)

{
    "cacheStatus": "NOT_STORED",
    "cacheStore": true,
    "resultCode": "200",
    "resultDtlCode": null,
    "resultMsg": "SUCCESS",
    "resultDtlMsg": null,
    "data": {
        "areaList": [
            [
                {
                    "cacheStatus": "NOT_STORED",
                    "cacheStore": true,
                    "unitType": "SPECIALDEAL_ITEM",
                    "dataType": "SPECIALPRICE_HOTDEAL",
                    "pageId": "100000007533",
                    "pageSetId": "2",
                    "pageCmptId": "4",
                    "reactPrefix": "특가|상품_5|",
                    "assocationParam": {
                        "actCtgId": null,
                        "dispCtgLowerId": null,
                        "hu": null,
                        "hr": null,
                        "reu": null,
                        "rer": null,
                        "fi": null,
                        "ru": null,
                        "re": null,
                        "pr": null,
                        "nrcq": null,
                        "dcim": null,
                        "nrcn": null,
                        "nrl": null,
                        "cornrSetId": null,
                        "brandId": null,
                        "dispCtgs": null,
                        "itemIds": null,
                        "frontItems": null,
                        "dti": null
                    },
                    "isAreaMoreFixedUnit": false,
                    "dispOrdr": 3,
                    "cornrSetId": "",
                    "itemList": [
                        {

응답 예시가 위와 같을 때 수집하고자 하는 데이터가  itemList 에 있을 때 itemList 하위의 모든 정보를 수집하는 코드를 작성하고 csv 파일로 저장 할 것                           