"""
실제 외부 API 연동 모듈
- 네이버 데이터랩 API
- LLM 기반 트렌드 생성 API
- 기상청 날씨 API
"""

import os
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


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
        # DB 엔진 초기화 (공휴일 조회용)
        db_url = os.getenv("POSTGRES_URI")
        if db_url:
            self.engine = create_engine(db_url)
        else:
            logger.warning("POSTGRES_URI 환경 변수가 설정되지 않음 - 공휴일 조회 비활성화")
            self.engine = None

    async def get_trending_searches(self, geo: str = 'KR', weather_info: Dict[str, Any] = None, hour: Optional[int] = None) -> List[Dict[str, Any]]:
        """LLM을 사용한 트렌딩 검색어 생성"""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.openai_api_key)

            # 현재 날짜와 계절 정보
            now = datetime.now()
            current_date = now.strftime("%Y년 %m월 %d일")
            month = now.month
            season = self._get_season(month)
            
            # 요일 정보
            weekday_names = ["월", "화", "수", "목", "금", "토", "일"]
            weekday_kr = weekday_names[now.weekday()]
            
            # 시간대 정보 (hour가 없으면 현재 시각 사용)
            current_hour = hour if hour is not None else now.hour
            time_slot_info = self._get_time_slot_info(current_hour)

            # 시기적 특성 (공휴일 DB 활용)
            seasonal_context = self._get_seasonal_context(now)

            # 날씨 정보
            weather_condition = weather_info.get("weather", "맑음") if weather_info else "정보 없음"
            temperature = weather_info.get("temperature", "정보 없음") if weather_info else "정보 없음"
            precipitation = weather_info.get("precipitation", "정보 없음") if weather_info else "정보 없음"


            prompt = f"""
너는 한국 홈쇼핑 업계에서 15년차 경력을 가진 베테랑 상품기획자(MD)야. 
너의 임무는 아래 주어진 상황에 맞춰 **대박**을 터뜨릴 수 있는 잠재력 높은 상품 키워드 20개를 제안하는 것이다.

## 현재 상황
* **방송 날짜:** {current_date} ({weekday_kr}요일)
* **방송 시간대:** {time_slot_info['slot']}
* **타겟 시청자:** {time_slot_info['target']}
* **시간대 특성:** {time_slot_info['characteristic']}
* **계절/시기:** {season}, {seasonal_context}
* **날씨 상황:** {weather_condition}, 기온 {temperature}°C, 강수량 {precipitation}%

## 지시 사항
위 상황을 종합적으로 분석하여, 홈쇼핑에서 **높은 매출**을 기대할 수 있는 상품 키워드 20개를 제안해 줘.
- 타겟 고객의 심리와 니즈를 고려할 것
- 최근 소비 트렌드(편리미엄, 가성비, 건강 관리 등)를 반영할 것
- 계절과 시기적 특성을 적극 활용할 것 (예: 비 오는 날 → 제습기, 우산, 빨래건조기)

각 키워드에 대해 다음을 제공해 줘:
1. 트렌드 점수 (1-100)
2. 짧은 이유 태그 (5-10자, 핵심만)

**응답 형식 (정확히 지킬 것):**
키워드1,점수1,이유1
키워드2,점수2,이유2
...
키워드20,점수20,이유20

**예시:**
제습기,95,장마철 필수
선풍기,88,무더위 대비
건강즙,85,아침 활력
"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )

            llm_response = response.choices[0].message.content.strip()

            trends = []
            lines = llm_response.split('\n')

            for idx, line in enumerate(lines):
                line = line.strip()
                if ',' in line:
                    try:
                        parts = line.split(',')
                        if len(parts) >= 3:  # 키워드,점수,이유
                            keyword = parts[0].strip()
                            score = int(float(parts[1].strip()))
                            reason = parts[2].strip()
                            

                            trends.append({
                                "keyword": keyword,
                                "rank": idx + 1,
                                "source": "llm_trending",
                                "timestamp": datetime.now().isoformat(),
                                "geo": geo,
                                "score": score,
                                "reason": reason
                            })
                        elif len(parts) == 2:  # 이유 없는 경우?
                            keyword = parts[0].strip()
                            score = int(float(parts[1].strip()))
                            trends.append({
                                "keyword": keyword,
                                "rank": idx + 1,
                                "source": "llm_trending",
                                "timestamp": datetime.now().isoformat(),
                                "geo": geo,
                                "score": score,
                                "reason": ""
                            })
                    except (ValueError, IndexError) as e:
                        print(f"LLM 트렌딩 검색어 처리 오류: {e}")
                        continue

            return trends[:20]  # 상위 20개

        except Exception as e:
            print(f"LLM 트렌딩 검색어 API 오류: {e}")
            return []

    def _get_season(self, month: int) -> str:
        """계절 정보 반환"""
        if month in [12, 1, 2]:
            return "겨울"
        elif month in [3, 4, 5]:
            return "봄"
        elif month in [6, 7, 8]:
            return "여름"
        else:
            return "가을"
    
    def _get_seasonal_context(self, now: datetime) -> str:
        """시기적 특성 자동 생성 (공휴일 DB 활용)"""
        month = now.month
        day = now.day
        contexts = []
        
        # DB에서 공휴일 정보 확인 (명절 전후 판단)
        try:
            # 최근 7일 이내 명절이 있었는지 확인
            recent_holiday = self._check_recent_holiday(now)
            if recent_holiday:
                contexts.append(f"{recent_holiday} 직후")
            
            # 향후 7일 이내 명절이 있는지 확인
            upcoming_holiday = self._check_upcoming_holiday(now)
            if upcoming_holiday:
                contexts.append(f"{upcoming_holiday} 준비 시즌")
        except Exception as e:
            logger.warning(f"공휴일 확인 실패: {e}")
        
        # 계절 이벤트 (양력 기준으로 판단 가능한 것들만)
        if month in [10, 11]:
            contexts.append("김장 준비 시즌")
        elif month in [11, 12]:
            contexts.append("연말 대청소 시즌")
        elif month == 12 and day >= 20:
            contexts.append("크리스마스 시즌")
        elif month == 1 and day <= 10:
            contexts.append("새해 맞이")
        
        # 환절기 (계절 변화)
        if month in [3, 4]:
            contexts.append("봄맞이 환절기")
        elif month in [9, 10]:
            contexts.append("가을맞이 환절기")
        
        # 날씨 기반
        if month in [12, 1, 2]:
            contexts.append("본격 겨울 추위")
        elif month in [6, 7, 8]:
            contexts.append("여름 더위")
        elif month in [6, 7]:
            contexts.append("장마철")
        
        return ", ".join(contexts) if contexts else "일반 시즌"
    
    def _check_recent_holiday(self, now: datetime) -> Optional[str]:
        """최근 7일 이내 명절이 있었는지 DB에서 확인"""
        if not self.engine:
            return None
            
        try:
            start_date = (now - timedelta(days=7)).date()
            end_date = now.date()
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT holiday_name 
                    FROM TAIHOLIDAYS 
                    WHERE holiday_date BETWEEN :start_date AND :end_date
                    AND holiday_type = '법정공휴일'
                    ORDER BY holiday_date DESC
                    LIMIT 1
                """)
                result = conn.execute(query, {"start_date": start_date, "end_date": end_date})
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(f"최근 공휴일 확인 중 오류: {e}")
            return None
    
    def _check_upcoming_holiday(self, now: datetime) -> Optional[str]:
        """향후 7일 이내 명절이 있는지 DB에서 확인"""
        if not self.engine:
            return None
            
        try:
            start_date = now.date()
            end_date = (now + timedelta(days=7)).date()
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT holiday_name 
                    FROM TAIHOLIDAYS 
                    WHERE holiday_date BETWEEN :start_date AND :end_date
                    AND holiday_type = '법정공휴일'
                    ORDER BY holiday_date ASC
                    LIMIT 1
                """)
                result = conn.execute(query, {"start_date": start_date, "end_date": end_date})
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(f"향후 공휴일 확인 중 오류: {e}")
            return None

    def _get_time_slot_info(self, hour: int) -> Dict[str, str]:
        """시간대 정보 반환"""
        if 6 <= hour < 9:
            return {
                "slot": "아침 시간대 (06:00-09:00)",
                "target": "막 아침 집안일을 끝낸 30대~50대 주부",
                "characteristic": "건강과 활력에 대한 관심이 높습니다."
            }
        elif 9 <= hour < 12:
            return {
                "slot": "오전 시간대 (09:00-12:00)",
                "target": "주부층과 은퇴 노년층",
                "characteristic": "주방용품, 생활가전, 식품, 유아동 상품에 대한 관심이 높은 시간대"
            }
        elif 12 <= hour < 14:
            return {
                "slot": "점심 시간대 (12:00-14:00)",
                "target": "전 연령층",
                "characteristic": "음식 관련 상품(간편식, 밀키트, 반찬 등)에 대한 구매욕구가 매우 높습니다."
            }
        elif 14 <= hour < 18:
            return {
                "slot": "오후 시간대 (14:00-18:00)",
                "target": "주부층과 은퇴 노년층",
                "characteristic": "여유로운 쇼핑과 본인 및 가족을 위한 구매가 활발합니다."
            }
        elif 18 <= hour < 21:
            return {
                "slot": "저녁 시간대 (18:00-21:00)",
                "target": "가족 단위 시청층 (골든타임)",
                "characteristic": "저녁 식사 관련 상품과 건강식품, 생활가전에 대한 관심이 높습니다."
            }
        elif 21 <= hour < 24:
            return {
                "slot": "야간 시간대 (21:00-24:00)",
                "target": "직장인과 젊은층",
                "characteristic": "뷰티/화장품, 건강기능식품, 개인 관리 상품에 대한 관심이 높습니다."
            }
        else:  # 00:00-06:00
            return {
                "slot": "심야 시간대 (00:00-06:00)",
                "target": "불면증 시청자, 야간 근무자",
                "characteristic": "판매량이 극히 적은 시간대. 보험, 여행, 헬스케어 등 무형 상품 위주로 편성. 시청률 대비 전환율이 낮음."
            }


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
