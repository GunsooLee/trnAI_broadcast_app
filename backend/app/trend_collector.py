"""
트렌드 데이터 수집 모듈
- 네이버 데이터랩 실시간 급상승 검색어
- 구글 트렌드 데이터
- 뉴스 키워드 감지
- 날씨/재난 정보
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import os
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

@dataclass
class TrendKeyword:
    """트렌드 키워드 데이터 클래스"""
    keyword: str
    source: str  # 'naver', 'google', 'news', 'weather'
    score: float  # 트렌드 점수 (0-100)
    timestamp: datetime
    category: Optional[str] = None
    related_keywords: Optional[List[str]] = None
    metadata: Optional[Dict] = None

class TrendCollector:
    """트렌드 데이터 수집기"""
    
    def __init__(self):
        self.session = None
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_naver_realtime_trends(self) -> List[TrendKeyword]:
        """네이버 실시간 급상승 검색어 수집"""
        try:
            # 네이버 데이터랩 API는 공식적으로 실시간 급상승 검색어를 제공하지 않으므로
            # 대안으로 네이버 트렌드 페이지를 스크래핑하거나 모의 데이터 생성
            url = "https://datalab.naver.com/keyword/realtimeList.naver"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 실제 구현에서는 네이버 페이지 구조에 맞게 파싱
                    # 시간대별 현실적인 모의 데이터
                    current_hour = datetime.now().hour
                    season = self._get_current_season()
                    
                    # 시간대별 트렌드 키워드
                    if 6 <= current_hour < 12:  # 오전
                        mock_keywords = ["모닝커피", "아침식사대용", "비타민", "홈트레이닝", "요가매트"]
                    elif 12 <= current_hour < 18:  # 오후  
                        mock_keywords = ["다이어트식품", "건강간식", "에어프라이어", "공기청정기", "마스크팩"]
                    elif 18 <= current_hour < 24:  # 저녁
                        mock_keywords = ["저녁식사", "건강차", "수면용품", "아로마", "족욕기"]
                    else:  # 새벽
                        mock_keywords = ["불면증", "수면영양제", "아이마스크", "화이트노이즈", "가습기"]
                    
                    # 계절별 키워드 추가
                    seasonal_keywords = self._get_seasonal_keywords(season)
                    mock_keywords.extend(seasonal_keywords[:2])
                    
                    trends = []
                    for i, keyword in enumerate(mock_keywords[:5]):
                        trends.append(TrendKeyword(
                            keyword=keyword,
                            source="naver",
                            score=100 - i * 15 + (current_hour % 3) * 5,  # 시간에 따른 변동
                            timestamp=datetime.now(),
                            category="실시간급상승",
                            related_keywords=[f"{keyword}추천", f"{keyword}후기", f"{keyword}할인"]
                        ))
                    
                    return trends
                    
        except Exception as e:
            logger.error(f"네이버 트렌드 수집 실패: {e}")
            return []
    
    async def collect_google_trends(self, timeframe: str = "now 1-H") -> List[TrendKeyword]:
        """구글 트렌드 데이터 수집"""
        try:
            # 요일별, 시간대별 현실적인 구글 트렌드 모의 데이터
            current_hour = datetime.now().hour
            weekday = datetime.now().weekday()  # 0=월요일, 6=일요일
            
            # 평일/주말 구분
            if weekday < 5:  # 평일
                if 9 <= current_hour < 18:  # 업무시간
                    mock_trends = [("홈쇼핑", 92), ("건강식품", 85), ("다이어트", 78), ("운동용품", 71), ("마스크팩", 65)]
                else:  # 퇴근 후
                    mock_trends = [("저녁식사", 88), ("건강차", 82), ("수면용품", 76), ("아로마", 70), ("족욕기", 64)]
            else:  # 주말
                mock_trends = [("홈트레이닝", 95), ("요리용품", 89), ("청소용품", 83), ("반려동물용품", 77), ("취미용품", 71)]
            
            # 계절 보정
            season = self._get_current_season()
            if season == "겨울":
                mock_trends = [("난방용품", 98)] + mock_trends[:4]
            elif season == "여름":
                mock_trends = [("냉방용품", 96)] + mock_trends[:4]
            
            trends = []
            for keyword, base_score in mock_trends:
                # 시간에 따른 점수 변동 (±10점)
                score_variation = (current_hour % 7) - 3
                final_score = min(100, max(50, base_score + score_variation))
                
                trends.append(TrendKeyword(
                    keyword=keyword,
                    source="google",
                    score=final_score,
                    timestamp=datetime.now(),
                    category="검색트렌드",
                    metadata={"weekday": weekday, "hour": current_hour}
                ))
            
            return trends
            
        except Exception as e:
            logger.error(f"구글 트렌드 수집 실패: {e}")
            return []
    
    async def collect_news_keywords(self) -> List[TrendKeyword]:
        """뉴스 키워드 감지 - 모의 데이터"""
        try:
            # 시간대별 뉴스 트렌드 모의 데이터
            current_hour = datetime.now().hour
            
            # 시간대별 이슈 키워드
            if 6 <= current_hour < 12:  # 오전 - 건강/라이프스타일
                news_keywords = [("건강관리", 88), ("아침운동", 82), ("영양제", 76), ("다이어트", 70)]
            elif 12 <= current_hour < 18:  # 오후 - 쇼핑/생활
                news_keywords = [("홈쇼핑특가", 92), ("생활용품", 86), ("주방용품", 80), ("청소용품", 74)]
            elif 18 <= current_hour < 22:  # 저녁 - 휴식/케어
                news_keywords = [("스킨케어", 90), ("수면건강", 84), ("아로마테라피", 78), ("족욕", 72)]
            else:  # 밤/새벽 - 수면/건강
                news_keywords = [("불면증해결", 85), ("수면영양제", 79), ("야식대용", 73), ("건강차", 67)]
            
            trends = []
            for keyword, score in news_keywords:
                trends.append(TrendKeyword(
                    keyword=keyword,
                    source="news",
                    score=score,
                    timestamp=datetime.now(),
                    category="뉴스이슈",
                    related_keywords=[f"{keyword}추천", f"{keyword}리뷰"]
                ))
            
            return trends
            
        except Exception as e:
            logger.error(f"뉴스 키워드 수집 실패: {e}")
            return []
    
    async def collect_weather_alerts(self) -> List[TrendKeyword]:
        """날씨/재난 정보 수집"""
        try:
            # 기상청 API 또는 날씨 정보를 통한 상품 연관 키워드 생성
            weather_keywords = {
                "폭염": ["선풍기", "에어컨", "쿨매트", "아이스팩", "시원한음료"],
                "한파": ["난방용품", "전기장판", "핫팩", "따뜻한의류", "보온용품"],
                "미세먼지": ["공기청정기", "마스크", "공기정화식물", "실내운동용품"],
                "장마": ["제습기", "곰팡이제거제", "우산", "방수용품"]
            }
            
            # 계절별 현실적인 날씨 상황 모의 데이터
            season = self._get_current_season()
            month = datetime.now().month
            
            if season == "여름":
                current_weather = "폭염" if month in [7, 8] else "장마"
            elif season == "겨울":
                current_weather = "한파"
            elif season in ["봄", "가을"]:
                current_weather = "미세먼지"
            else:
                current_weather = "폭염"
            
            trends = []
            if current_weather in weather_keywords:
                for i, keyword in enumerate(weather_keywords[current_weather]):
                    trends.append(TrendKeyword(
                        keyword=keyword,
                        source="weather",
                        score=90 - i * 5,
                        timestamp=datetime.now(),
                        category="날씨연관",
                        metadata={"weather_condition": current_weather}
                    ))
            
            return trends
            
        except Exception as e:
            logger.error(f"날씨 정보 수집 실패: {e}")
            return []
    
    def _extract_keywords_from_news(self, news_items: List[Dict]) -> List[tuple]:
        """뉴스 아이템에서 키워드 추출"""
        keywords = {}
        
        for item in news_items:
            title = item.get("title", "")
            description = item.get("description", "")
            
            # HTML 태그 제거
            title = re.sub(r'<[^>]+>', '', title)
            description = re.sub(r'<[^>]+>', '', description)
            
            # 간단한 키워드 추출 (실제로는 더 정교한 NLP 처리 필요)
            text = f"{title} {description}"
            
            # 상품 관련 키워드 패턴
            product_patterns = [
                r'(\w+제품)', r'(\w+용품)', r'(\w+기기)', r'(\w+식품)',
                r'(\w+화장품)', r'(\w+의류)', r'(\w+가전)', r'(\w+건강)'
            ]
            
            for pattern in product_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) >= 2:  # 너무 짧은 키워드 제외
                        keywords[match] = keywords.get(match, 0) + 1
        
        # 빈도순으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [(k, min(v * 10, 100)) for k, v in sorted_keywords[:10]]
    
    def _get_current_season(self) -> str:
        """현재 계절 반환"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "겨울"
        elif month in [3, 4, 5]:
            return "봄"
        elif month in [6, 7, 8]:
            return "여름"
        else:
            return "가을"
    
    def _get_seasonal_keywords(self, season: str) -> List[str]:
        """계절별 키워드 반환"""
        seasonal_map = {
            "봄": ["알레르기", "미세먼지마스크", "봄나물", "환절기건강", "꽃가루"],
            "여름": ["에어컨", "선풍기", "아이스크림", "자외선차단", "물놀이용품"],
            "가을": ["환절기", "감기예방", "건조함", "보습", "면역력"],
            "겨울": ["난방", "보온", "건조", "감기", "비타민D"]
        }
        return seasonal_map.get(season, [])
    
    async def collect_all_trends(self) -> List[TrendKeyword]:
        """모든 소스에서 트렌드 데이터 수집"""
        all_trends = []
        
        # 병렬로 모든 트렌드 수집
        tasks = [
            self.collect_naver_realtime_trends(),
            self.collect_google_trends(),
            self.collect_news_keywords(),
            self.collect_weather_alerts()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_trends.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"트렌드 수집 중 오류: {result}")
        
        # 중복 제거 및 점수순 정렬
        unique_trends = self._deduplicate_trends(all_trends)
        return sorted(unique_trends, key=lambda x: x.score, reverse=True)
    
    def _deduplicate_trends(self, trends: List[TrendKeyword]) -> List[TrendKeyword]:
        """중복 트렌드 키워드 제거 및 점수 통합"""
        keyword_map = {}
        
        for trend in trends:
            key = trend.keyword.lower()
            if key in keyword_map:
                # 기존 트렌드와 점수 통합 (가중평균)
                existing = keyword_map[key]
                total_score = (existing.score + trend.score) / 2
                keyword_map[key] = TrendKeyword(
                    keyword=trend.keyword,
                    source=f"{existing.source},{trend.source}",
                    score=total_score,
                    timestamp=max(existing.timestamp, trend.timestamp),
                    category=existing.category or trend.category,
                    related_keywords=(existing.related_keywords or []) + (trend.related_keywords or [])
                )
            else:
                keyword_map[key] = trend
        
        return list(keyword_map.values())

class TrendProcessor:
    """트렌드 키워드 처리 및 상품 매칭"""
    
    def __init__(self, product_embedder):
        self.product_embedder = product_embedder
    
    async def match_trends_to_products(self, trends: List[TrendKeyword], top_k: int = 5) -> Dict:
        """트렌드 키워드를 상품과 매칭"""
        matched_results = {}
        
        for trend in trends:
            try:
                # 임베딩을 통한 상품 매칭
                if self.product_embedder:
                    similar_products = self.product_embedder.search_products(
                        trend_keywords=[trend.keyword],
                        top_k=top_k
                    )
                    
                    matched_results[trend.keyword] = {
                        "trend_info": {
                            "source": trend.source,
                            "score": trend.score,
                            "category": trend.category,
                            "timestamp": trend.timestamp.isoformat()
                        },
                        "matched_products": similar_products
                    }
                
            except Exception as e:
                logger.error(f"트렌드-상품 매칭 실패 ({trend.keyword}): {e}")
        
        return matched_results
    
    def calculate_trend_boost_factor(self, trend_score: float, base_factor: float = 1.0) -> float:
        """트렌드 점수를 기반으로 매출 예측 부스트 팩터 계산"""
        # 트렌드 점수가 높을수록 매출 예측에 긍정적 영향
        # 점수 0-100을 1.0-2.0 배수로 변환
        boost_factor = base_factor + (trend_score / 100.0)
        return min(boost_factor, 2.0)  # 최대 2배까지만 부스트
