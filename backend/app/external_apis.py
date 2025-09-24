"""
실제 외부 API 연동 모듈
- 네이버 데이터랩 API
- LLM 기반 트렌드 생성 API
- 기상청 날씨 API
"""

import os
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp


class NaverTrendAPI:
    """네이버 데이터랩 트렌드 API 클라이언트"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/datalab"
        
    def get_search_trends(self, keywords: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """검색 트렌드 데이터 조회"""
        url = f"{self.base_url}/search"
        
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json"
        }
        
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "date",
            "keywordGroups": [
                {
                    "groupName": keyword,
                    "keywords": [keyword]
                } for keyword in keywords
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"네이버 트렌드 API 오류: {e}")
            return {"results": []}
    
    def get_shopping_trends(self, keywords: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """쇼핑 트렌드 데이터 조회"""
        url = f"{self.base_url}/shopping/category"
        
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json"
        }
        
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "date",
            "category": [
                {"name": keyword, "param": [keyword]} for keyword in keywords
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"네이버 쇼핑 트렌드 API 오류: {e}")
            return {"results": []}


class LLMTrendAPI:
    """LLM 기반 트렌드 키워드 생성 API"""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key

    async def get_trending_searches(self, geo: str = 'KR') -> List[Dict[str, Any]]:
        """LLM을 사용한 트렌딩 검색어 생성"""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.openai_api_key)

            current_date = datetime.now().strftime("%Y년 %m월 %d일")

            prompt = f"""
한국에서 {current_date} 현재 홈쇼핑과 관련하여 인기가 높거나 트렌딩될 것으로 예상되는 키워드 20개를 생성해주세요.

고려 요소:
- 계절적 요인 (현재 9월 중순)
- 한국의 소비 트렌드
- 홈쇼핑 특성상 인기 있는 카테고리들
- 최근 이슈나 관심사

각 키워드에 대해 1-100 사이의 트렌드 점수도 함께 제공해주세요.

응답 형식은 정확히 다음과 같이 해주세요:
키워드1,점수1
키워드2,점수2
...
키워드20,점수20
"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )

            llm_response = response.choices[0].message.content.strip()

            trends = []
            lines = llm_response.split('\n')

            for idx, line in enumerate(lines):
                line = line.strip()
                if ',' in line:
                    try:
                        keyword, score_str = line.split(',', 1)
                        keyword = keyword.strip()
                        score = int(float(score_str.strip()))

                        trends.append({
                            "keyword": keyword,
                            "rank": idx + 1,
                            "source": "llm_trending",
                            "timestamp": datetime.now().isoformat(),
                            "geo": geo,
                            "score": score
                        })
                    except (ValueError, IndexError):
                        continue

            return trends[:20]  # 상위 20개

        except Exception as e:
            print(f"LLM 트렌딩 검색어 API 오류: {e}")
            return []


class WeatherAPI:
    """기상청 날씨 API 클라이언트"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
        
    def get_current_weather(self, nx: int = 60, ny: int = 127) -> Dict[str, Any]:
        """현재 날씨 정보 조회 (서울 기준)"""
        url = f"{self.base_url}/getUltraSrtNcst"
        
        now = datetime.now()
        base_date = now.strftime("%Y%m%d")
        base_time = now.strftime("%H00")
        
        params = {
            "serviceKey": self.api_key,
            "numOfRows": "10",
            "pageNo": "1",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
            "dataType": "JSON"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("response", {}).get("header", {}).get("resultCode") == "00":
                items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                
                weather_data = {
                    "temperature": None,
                    "humidity": None,
                    "precipitation": None,
                    "wind_speed": None,
                    "timestamp": datetime.now().isoformat()
                }
                
                for item in items:
                    category = item.get("category")
                    value = float(item.get("obsrValue", 0))
                    
                    if category == "T1H":  # 기온
                        weather_data["temperature"] = value
                    elif category == "REH":  # 습도
                        weather_data["humidity"] = value
                    elif category == "RN1":  # 1시간 강수량
                        weather_data["precipitation"] = value
                    elif category == "WSD":  # 풍속
                        weather_data["wind_speed"] = value
                
                return weather_data
            else:
                print(f"기상청 API 오류: {data}")
                return self._get_mock_weather()
                
        except Exception as e:
            print(f"날씨 API 오류: {e}")
            return self._get_mock_weather()
    
    def _get_mock_weather(self) -> Dict[str, Any]:
        """API 오류 시 모의 날씨 데이터 반환"""
        return {
            "temperature": 22.5,
            "humidity": 65.0,
            "precipitation": 0.0,
            "wind_speed": 2.3,
            "timestamp": datetime.now().isoformat(),
            "mock": True
        }


class ExternalAPIManager:
    """외부 API 통합 관리 클래스"""

    def __init__(self):
        # 환경변수에서 API 키 로드
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # API 클라이언트 초기화
        self.naver_api = None
        self.llm_trend_api = None
        self.weather_api = None

        if self.naver_client_id and self.naver_client_secret:
            self.naver_api = NaverTrendAPI(self.naver_client_id, self.naver_client_secret)

        if self.openai_api_key:
            self.llm_trend_api = LLMTrendAPI(self.openai_api_key)

        if self.weather_api_key:
            self.weather_api = WeatherAPI(self.weather_api_key)

    async def collect_all_trends(self) -> List[Dict[str, Any]]:
        """모든 소스에서 트렌드 데이터 수집"""
        all_trends = []

        # LLM 기반 트렌딩 검색어 수집
        if self.llm_trend_api:
            try:
                llm_trends = await self.llm_trend_api.get_trending_searches()
                for trend in llm_trends:
                    all_trends.append({
                        "keyword": trend["keyword"],
                        "source": "llm",
                        "score": trend.get("score", 100 - trend["rank"]),
                        "timestamp": trend["timestamp"],
                        "category": self._categorize_keyword(trend["keyword"]),
                        "related_keywords": [],
                        "metadata": {"rank": trend["rank"], "llm_generated": True}
                    })
            except Exception as e:
                print(f"LLM 트렌드 수집 오류: {e}")

        # 네이버 트렌드 수집 (API 키가 있는 경우)
        if self.naver_api:
            try:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

                # 홈쇼핑 관련 키워드로 검색
                shopping_keywords = ["건강식품", "의류", "가전제품", "뷰티", "생활용품"]
                naver_data = self.naver_api.get_search_trends(shopping_keywords, start_date, end_date)

                for result in naver_data.get("results", []):
                    keyword = result.get("title", "")
                    if keyword and result.get("data"):
                        latest_data = result["data"][-1]  # 최신 데이터
                        all_trends.append({
                            "keyword": keyword,
                            "source": "naver",
                            "score": latest_data.get("ratio", 0),
                            "timestamp": datetime.now().isoformat(),
                            "category": self._categorize_keyword(keyword),
                            "related_keywords": [],
                            "metadata": {"period": f"{start_date}~{end_date}"}
                        })
            except Exception as e:
                print(f"네이버 트렌드 수집 오류: {e}")

        return all_trends
    
    def get_current_weather(self) -> Dict[str, Any]:
        """현재 날씨 정보 조회"""
        if self.weather_api:
            return self.weather_api.get_current_weather()
        else:
            # API 키가 없으면 모의 데이터 반환
            return {
                "temperature": 22.5,
                "humidity": 65.0,
                "precipitation": 0.0,
                "wind_speed": 2.3,
                "timestamp": datetime.now().isoformat(),
                "mock": True
            }
    
    def _categorize_keyword(self, keyword: str) -> str:
        """키워드를 카테고리로 분류"""
        categories = {
            "건강식품": ["비타민", "영양제", "건강", "다이어트", "보조제", "프로틴"],
            "의류": ["옷", "패션", "드레스", "셔츠", "바지", "치마", "코트"],
            "가전": ["에어프라이어", "선풍기", "청소기", "세탁기", "냉장고", "TV"],
            "뷰티": ["화장품", "스킨케어", "미용", "크림", "로션", "마스크"],
            "생활용품": ["주방", "욕실", "청소", "수납", "정리", "인테리어"]
        }
        
        keyword_lower = keyword.lower()
        for category, words in categories.items():
            if any(word in keyword_lower for word in words):
                return category
        
        return "기타"


# 전역 인스턴스
external_api_manager = ExternalAPIManager()
