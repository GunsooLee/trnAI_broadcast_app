"""
방송 편성 AI 추천 워크플로우
LangChain 기반 2단계 워크플로우: AI 방향 탐색 + 고속 랭킹
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .external_apis import ExternalAPIManager

from .dependencies import get_product_embedder
from . import broadcast_recommender as br
from .schemas import BroadcastResponse, BroadcastRecommendation, ProductInfo, BusinessMetrics, NaverProduct, CompetitorProduct, LastBroadcastMetrics
from .external_products_service import ExternalProductsService
from .services.broadcast_history_service import BroadcastHistoryService
from .netezza_config import netezza_conn

logger = logging.getLogger(__name__)

class BroadcastWorkflow:
    """방송 편성 AI 추천 워크플로우"""
    
    def __init__(self, model):
        self.model = model  # XGBoost 모델
        self.product_embedder = get_product_embedder()
        
        # AI 트렌드 캐시 (시간대별)
        self._ai_trends_cache = {}
        self._cache_ttl = 3600  # 1시간 (초)
        
        # LangChain LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # DB 연결
        self.engine = create_engine(os.getenv("POSTGRES_URI"))
        
        # 외부 상품 서비스
        self.external_products_service = ExternalProductsService()
        
        # 방송 이력 서비스 (Netezza)
        self.broadcast_history_service = BroadcastHistoryService()
    
    async def process_broadcast_recommendation(
        self, 
        broadcast_time: str, 
        recommendation_count: int = 5,
        trend_weight: float = 0.3,  # 트렌드 가중치 (0.3 = 30%)
        selling_weight: float = 0.7   # 매출 예측 가중치 (0.7 = 70%)
    ) -> BroadcastResponse:
        """메인 워크플로우: 방송 시간 기반 추천
        
        Args:
            broadcast_time: 방송 시간
            recommendation_count: 추천 개수
            trend_weight: 트렌드 가중치 (0.0~1.0, 기본 0.3)
            selling_weight: 매출 예측 가중치 (0.0~1.0, 기본 0.7)
                - 예: trend_weight=0.3, selling_weight=0.7 → 트렌드 30%, 매출 70%
                - 예: trend_weight=0.5, selling_weight=0.5 → 균형 (50:50)
        """
        
        import time
        workflow_start = time.time()
        
        print("=== [DEBUG] process_broadcast_recommendation 시작 ===")
        request_time = datetime.now().isoformat()
        logger.info(f"방송 추천 워크플로우 시작: {broadcast_time}")
        print(f"=== [DEBUG] broadcast_time: {broadcast_time}, recommendation_count: {recommendation_count} ===")
        
        try:
            # 1단계: 컨텍스트 수집 및 통합 키워드 생성
            step_start = time.time()
            print("=== [DEBUG] _collect_context_and_keywords 호출 ===")
            context = await self._collect_context_and_keywords(broadcast_time)
            print(f"⏱️  [1단계] 컨텍스트 수집: {time.time() - step_start:.2f}초")
            print(f"=== [DEBUG] 통합 키워드: {len(context.get('unified_keywords', []))}개 ===")
            
            # 2. 통합 검색 실행 (1회)
            step_start = time.time()
            print("=== [DEBUG] _execute_unified_search 호출 ===")
            search_result = await self._execute_unified_search(context, context.get("unified_keywords", []))
            print(f"⏱️  [2단계] 통합 검색: {time.time() - step_start:.2f}초")
            print(f"=== [DEBUG] 검색 완료 - 직접매칭: {len(search_result['direct_products'])}개, 카테고리: {len(search_result['category_groups'])}개 ===")
            
            # 검색에 사용된 키워드를 context에 저장
            context["search_keywords"] = search_result.get("search_keywords", [])
            
            # 3. 후보군 생성 (가중치 기반 비율 조정)
            step_start = time.time()
            print("=== [DEBUG] _generate_unified_candidates 호출 ===")
            max_trend = max(1, int(recommendation_count * trend_weight))  # 최소 1개
            max_sales = recommendation_count - max_trend + 3  # 여유분 추가
            print(f"=== [DEBUG] 가중치 적용: 트렌드 {max_trend}개 ({trend_weight:.0%}), 매출 {max_sales}개 ({selling_weight:.0%}) ===")
            
            candidate_products, category_scores = await self._generate_unified_candidates(
                search_result,
                context,
                max_trend_match=max_trend,
                max_sales_prediction=max_sales
            )
            print(f"⏱️  [3단계] 후보군 생성: {time.time() - step_start:.2f}초")
            print(f"=== [DEBUG] 후보군 생성 완료: {len(candidate_products)}개 ===")
            
            # 4. 최종 랭킹 계산
            step_start = time.time()
            ranked_products = await self._rank_final_candidates(
                candidate_products,
                category_scores=category_scores,
                context=context
            )
            print(f"⏱️  [4단계] 최종 랭킹: {time.time() - step_start:.2f}초")
            
            # 5. API 응답 생성
            step_start = time.time()
            response = await self._format_response(ranked_products[:recommendation_count], context)
            response.requestTime = request_time
            step_time = time.time() - step_start
            print(f"⏱️  [5단계] 응답 생성 총: {step_time:.2f}초")
            
            total_time = time.time() - workflow_start
            print(f"⏱️  ===== 워크플로우 총 시간: {total_time:.2f}초 =====")
            
            logger.info(f"방송 추천 완료: {len(ranked_products)}개 추천")
            return response
            
        except Exception as e:
            print(f"=== [DEBUG] 예외 발생: {type(e).__name__}: {e} ===")
            import traceback
            traceback.print_exc()
            logger.error(f"방송 추천 워크플로우 오류: {e}")
            # OpenAI API 관련 오류는 상위로 전파 (503 에러 반환용)
            if "AI 서비스" in str(e) or "OpenAI" in str(e) or "할당량" in str(e):
                raise e
            # 기타 내부 오류는 500 에러로 처리
            raise Exception(f"내부 서버 오류: {e}")
    
    async def _collect_context_and_keywords(self, broadcast_time: str) -> Dict[str, Any]:
        """컨텍스트 수집 및 통합 키워드 생성 (개선된 버전)"""
        
        # 방송 시간 파싱
        broadcast_dt = datetime.fromisoformat(broadcast_time.replace('Z', '+00:00'))
        
        # DB에서 공휴일 정보 조회
        holiday_name = await self._get_holiday_from_db(broadcast_dt.date())
        
        context = {
            "broadcast_time": broadcast_time,
            "broadcast_dt": broadcast_dt,
            "hour": broadcast_dt.hour,
            "weekday": broadcast_dt.weekday(),
            "season": self._get_season(broadcast_dt.month),
            "holiday_name": holiday_name  # 공휴일 정보 추가
        }
        
        # 날씨 정보 수집
        weather_info = br.get_weather_by_date(broadcast_dt.date())
        context["weather"] = weather_info

        # 시간대 정보
        time_slot = self._get_time_slot(broadcast_dt)
        day_type = "주말" if broadcast_dt.weekday() >= 5 else "평일"
        context["time_slot"] = time_slot
        context["day_type"] = day_type

        # AI 기반 트렌드 생성 (LLM API) - 캐싱 적용
        cache_key = f"{broadcast_dt.hour}_{weather_info.get('weather', 'Clear')}"
        current_time = datetime.now().timestamp()
        
        # 캐시 확인
        if cache_key in self._ai_trends_cache:
            cached_data, cached_time = self._ai_trends_cache[cache_key]
            if current_time - cached_time < self._cache_ttl:
                context["ai_trends"] = cached_data
                logger.info(f"✅ AI 트렌드 캐시 히트 ({cache_key}): {len(cached_data)}개 키워드")
            else:
                # 캐시 만료
                del self._ai_trends_cache[cache_key]
                logger.info(f"⏰ AI 트렌드 캐시 만료 ({cache_key})")
                context["ai_trends"] = None
        else:
            context["ai_trends"] = None
        
        # 캐시 미스 시 API 호출
        if context["ai_trends"] is None:
            api_manager = ExternalAPIManager()
            if api_manager.llm_trend_api:
                try:
                    import time
                    api_start = time.time()
                    # 방송 시간과 날씨 정보를 전달하여 맥락 기반 트렌드 생성
                    llm_trends = await api_manager.llm_trend_api.get_trending_searches(
                        hour=broadcast_dt.hour,
                        weather_info=weather_info
                    )
                    api_time = time.time() - api_start
                    # AI가 생성한 트렌드 키워드 추가
                    context["ai_trends"] = [t["keyword"] for t in llm_trends]
                    # 캐시 저장
                    self._ai_trends_cache[cache_key] = (context["ai_trends"], current_time)
                    logger.info(f"🔥 AI 트렌드 생성 완료 ({broadcast_dt.hour}시, {weather_info.get('weather', 'N/A')}): {len(llm_trends)}개 키워드 (소요: {api_time:.2f}초)")
                    logger.info(f"AI 트렌드: {context['ai_trends'][:5]}...")  # 상위 5개만 로그
                except Exception as e:
                    logger.error(f"AI 트렌드 생성 실패: {e}")
                    context["ai_trends"] = []
            else:
                logger.warning("OpenAI API 키 없음 - AI 트렌드 생성 건너뜀")
                context["ai_trends"] = []

        # 컨텍스트 로그 출력
        logger.info(f"컨텍스트 수집 완료 - 계절: {context['season']}, 시간대: {time_slot}, 요일: {day_type}")
        if holiday_name:
            logger.info(f"🎉 공휴일: {holiday_name}")
        logger.info(f"날씨: {weather_info.get('weather', 'N/A')}")
        
        # 통합 키워드 생성 (컨텍스트 우선, AI 트렌드는 보조)
        unified_keywords = []
        
        # 1. 컨텍스트 기반 키워드 생성 (날짜/시간/날씨 기반 - 우선순위 높음)
        context_keywords = await self._generate_context_keywords(context)
        if context_keywords:
            unified_keywords.extend(context_keywords)
            logger.info(f"[우선순위 1] 컨텍스트 키워드 {len(context_keywords)}개 추가")
        
        # 2. AI 트렌드 키워드 추가 (실시간 트렌드 - 보조 역할, 개수 제한)
        if context.get("ai_trends"):
            ai_trend_limit = 3  # 10개 → 3개로 축소
            unified_keywords.extend(context["ai_trends"][:ai_trend_limit])
            logger.info(f"[우선순위 2] AI 트렌드 키워드 {len(context['ai_trends'][:ai_trend_limit])}개 추가 (보조)")
        
        # 3. 중복 제거 및 저장
        context["unified_keywords"] = list(dict.fromkeys(unified_keywords))  # 순서 유지 중복 제거
        logger.info(f"통합 키워드 생성 완료: 총 {len(context['unified_keywords'])}개")
        logger.info(f"통합 키워드 (우선순위순): {context['unified_keywords']}")

        return context
    
    async def _get_holiday_from_db(self, target_date) -> Optional[str]:
        """DB에서 공휴일 정보 조회"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT holiday_name 
                    FROM TAIHOLIDAYS 
                    WHERE holiday_date = :target_date
                """)
                result = conn.execute(query, {"target_date": target_date})
                row = result.fetchone()
                
                if row:
                    holiday_name = row[0]
                    logger.info(f"공휴일 조회 성공: {target_date} -> {holiday_name}")
                    return holiday_name
                else:
                    return None
        except Exception as e:
            logger.error(f"공휴일 조회 오류: {e}")
            return None
    
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
    
    def _get_time_slot(self, dt: datetime) -> str:
        """시간대 정보 반환"""
        hour = dt.hour
        if 6 <= hour < 12:
            return "오전"
        elif 12 <= hour < 18:
            return "오후"
        elif 18 <= hour < 24:
            return "저녁"
        else:
            return "새벽"

    # _classify_keywords_with_langchain 함수 제거됨
    # 이제 _generate_base_context_keywords에서 키워드 생성과 확장을 통합 처리
    
    async def _execute_unified_search(self, context: Dict[str, Any], unified_keywords: List[str]) -> Dict[str, Any]:
        """다단계 Qdrant 검색: 키워드를 그룹별로 나눠서 검색하여 임베딩 희석 방지"""
        
        print(f"=== [DEBUG Multi-Stage Search] 시작, keywords: {len(unified_keywords)}개 ===")
        
        if not unified_keywords:
            logger.warning("통합 키워드 없음 - 빈 결과 반환")
            return {"direct_products": [], "category_groups": {}}
        
        try:
            # 키워드를 3개 그룹으로 나누기
            # 1단계: 핵심 키워드 (처음 5개)
            # 2단계: 중간 키워드 (다음 5개)
            # 3단계: 보완 키워드 (나머지)
            
            group1 = unified_keywords[:5]   # 핵심
            group2 = unified_keywords[5:10]  # 중간
            group3 = unified_keywords[10:]   # 보완
            
            all_results = []
            seen_products = set()
            
            # 1단계: 핵심 키워드 검색 (고유사도 기대)
            if group1:
                query1 = " ".join(group1)
                print(f"=== [1단계 검색] 핵심 키워드: {group1} ===")
                results1 = self.product_embedder.search_products(
                    trend_keywords=[query1],
                    top_k=30,  # 20 → 30 증가
                    score_threshold=0.4,
                    only_ready_products=True
                )
                for r in results1:
                    code = r.get("product_code")
                    if code not in seen_products:
                        all_results.append(r)
                        seen_products.add(code)
                print(f"  → {len(results1)}개 발견 (누적: {len(all_results)}개)")
            
            # 2단계: 중간 키워드 검색
            if group2:
                query2 = " ".join(group2)
                print(f"=== [2단계 검색] 중간 키워드: {group2} ===")
                results2 = self.product_embedder.search_products(
                    trend_keywords=[query2],
                    top_k=30,  # 20 → 30 증가
                    score_threshold=0.3,
                    only_ready_products=True
                )
                for r in results2:
                    code = r.get("product_code")
                    if code not in seen_products:
                        all_results.append(r)
                        seen_products.add(code)
                print(f"  → {len(results2)}개 발견 (누적: {len(all_results)}개)")
            
            # 3단계: 보완 키워드 검색
            if group3:
                query3 = " ".join(group3)
                print(f"=== [3단계 검색] 보완 키워드: {group3} ===")
                results3 = self.product_embedder.search_products(
                    trend_keywords=[query3],
                    top_k=25,  # 15 → 25 증가
                    score_threshold=0.3,
                    only_ready_products=True
                )
                for r in results3:
                    code = r.get("product_code")
                    if code not in seen_products:
                        all_results.append(r)
                        seen_products.add(code)
                print(f"  → {len(results3)}개 발견 (누적: {len(all_results)}개)")
            
            print(f"=== [다단계 검색 완료] 총 {len(all_results)}개 상품 ===")
            
            # 유사도 분포 확인 (디버깅)
            if all_results:
                similarities = [p.get("similarity_score", 0) for p in all_results]
                print(f"[유사도 분포] 최고: {max(similarities):.3f}, 평균: {sum(similarities)/len(similarities):.3f}, 최저: {min(similarities):.3f}")
                print(f"[상위 5개 유사도]")
                for i, p in enumerate(all_results[:5], 1):
                    sim = p.get("similarity_score", 0)
                    name = p.get("product_name", "")[:40]
                    tape = "📼" if (p.get("tape_code") and p.get("tape_name")) else "❌"
                    print(f"  {i}. {name} | 유사도: {sim:.3f} | 테이프: {tape}")
            
            # 유사도 기반 분류
            direct_products = []      # 고유사도: 직접 추천
            category_groups = {}      # 중유사도: 카테고리별 그룹
            
            # 유사도 임계값
            HIGH_SIMILARITY_THRESHOLD = 0.45  # 실제 유사도 분포(최고 0.498)에 맞춤
            
            for product in all_results:
                similarity = product.get("similarity_score", 0)
                category = product.get("category_main", "기타")
                
                # 고유사도 상품: 직접 매칭
                if similarity >= HIGH_SIMILARITY_THRESHOLD:
                    if product.get("tape_code") and product.get("tape_name"):
                        direct_products.append({
                            **product,
                            "source": "direct_match",
                            "similarity_score": similarity
                        })
                        print(f"  ✅ 직접매칭: {product.get('product_name')[:30]} (유사도: {similarity:.2f})")
                
                # 중유사도: 카테고리 그룹핑
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(product)
            
            print(f"=== [분류 완료] 직접매칭: {len(direct_products)}개, 카테고리: {len(category_groups)}개 ===")
            
            return {
                "direct_products": direct_products,
                "category_groups": category_groups,
                "search_keywords": unified_keywords[:5]
            }
            
        except Exception as e:
            logger.error(f"다단계 검색 오류: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            return {"direct_products": [], "category_groups": {}}
    
    async def _get_realtime_trend_keywords(self) -> List[str]:
        """실시간 트렌드 키워드 수집 (OpenAI Web Search)"""
        from openai import OpenAI
        
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = """당신은 20년차 한국 쇼핑 트렌드 분석 전문가입니다.

**임무: 지금 이 순간 한국에서 인기 있는 쇼핑 관련 키워드를 찾으세요**

웹 검색으로 다음 정보를 수집하세요:
- 한국 실시간 인기 검색어
- 현재 이슈가 되는 이벤트 (스포츠, 날씨 이슈, 사회 이벤트 등)
- 쇼핑 트렌드 키워드

**중요:**
- 쇼핑/상품과 연관 가능한 키워드만 추출
- 3-5개의 핵심 키워드만 선별
- 반드시 JSON 배열로만 반환: ["키워드1", "키워드2", ...]

**예시:**
- 가을야구 경기 중 → ["야구", "치킨", "맥주", "응원용품"]
- 한파주의보 → ["난방", "온열기", "핫팩"]
- 크리스마스 시즌 → ["크리스마스", "선물", "파티용품"]
"""
            
            print("=" * 80)
            print("[2단계 - OpenAI Web Search] 실시간 트렌드 수집 시작")
            print("=" * 80)
            print(f"[프롬프트]\n{prompt}")
            print("=" * 80)
            logger.info(f"[2단계] 실시간 트렌드 프롬프트: {prompt[:200]}...")
            
            response = client.responses.create(
                model="gpt-4o",
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "medium",
                    "user_location": {
                        "type": "approximate",
                        "country": "KR",
                        "timezone": "Asia/Seoul"
                    }
                }],
                input=prompt,
                max_output_tokens=200
            )
            
            result_text = response.output_text
            print("=" * 80)
            print(f"[2단계 - 응답] {result_text}")
            print("=" * 80)
            logger.info(f"[2단계] 실시간 트렌드 응답: {result_text}")
            
            # JSON 배열 추출
            import json
            import re
            json_match = re.search(r'\[.*?\]', result_text, re.DOTALL)
            if json_match:
                keywords = json.loads(json_match.group())
                print(f"[2단계 - 추출 성공] 키워드: {keywords}")
                logger.info(f"[2단계] 실시간 트렌드 키워드 추출 성공: {keywords}")
                return keywords[:5]  # 최대 5개만
            else:
                print("[2단계 - 실패] JSON 배열을 찾을 수 없음")
                logger.warning("[2단계] 실시간 트렌드에서 JSON 배열을 찾을 수 없음")
                return []
                
        except Exception as e:
            print("=" * 80)
            print(f"[2단계 - 오류] {e}")
            print("=" * 80)
            logger.error(f"[2단계] 실시간 트렌드 수집 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _generate_base_context_keywords(self, context: Dict[str, Any]) -> List[str]:
        """기본 컨텍스트 정보를 기반으로 LangChain으로 검색 키워드 생성"""
        
        # 컨텍스트 정보 추출 (안전하게)
        weather_info = context.get("weather", {})
        logger.info(f"weather_info type: {type(weather_info)}, value: {weather_info}")
        
        if isinstance(weather_info, dict):
            weather = weather_info.get("weather", "맑음")
            temperature = weather_info.get("temperature", 20)
        else:
            logger.warning(f"weather_info is not dict: {weather_info}")
            weather = "맑음"
            temperature = 20
        
        time_slot = context.get("time_slot", "저녁")
        day_type = context.get("day_type", "평일")
        holiday_name = context.get("holiday_name")  # 공휴일 정보
        
        # 날짜 정보 추출
        broadcast_dt = context.get("broadcast_dt")
        month = broadcast_dt.month if broadcast_dt else 11
        day = broadcast_dt.day if broadcast_dt else 19
        
        logger.info(f"추출된 정보 - weather: {weather}, temp: {temperature}, time_slot: {time_slot}, month: {month}, day: {day}, day_type: {day_type}, holiday: {holiday_name}")
        
        # LangChain 프롬프트 (키워드 생성 + 확장 통합)
        keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 20년차 홈쇼핑 상품 검색 전문가입니다. 
주어진 상황에 맞는 **구체적인 상품명 키워드**를 생성하고, 추상적인 키워드는 확장해주세요.

**2단계 작업:**
1. 상황에 맞는 키워드 10-15개 생성
2. 추상적 키워드를 구체적 상품명으로 확장

**핵심 원칙: 실제 상품명처럼 구체적으로!**

❌ 나쁜 예 (추상적):
- "겨울준비", "건강관리", "가족모임", "따뜻한", "편리한"

✅ 좋은 예 (구체적):
- "패딩", "기모바지", "담요", "오메가3", "락토핏", "온열기", "전기장판"

**시즌별 구체적 키워드 예시:**

11월-12월 (겨울):
- 의류: "패딩", "기모", "목도리", "장갑", "겨울코트"
- 가전: "온열기", "전기장판", "히터", "가습기"
- 건강: "오메가3", "유산균", "홍삼", "비타민", "면역"
- 식품: "군고구마", "호빵", "어묵", "핫팩"

7-8월 (여름):
- 의류: "반팔", "반바지", "원피스", "샌들"
- 가전: "선풍기", "에어컨", "제습기", "냉풍기"
- 건강: "수분크림", "자외선차단제", "비타민C"
- 식품: "아이스크림", "냉면", "수박", "음료"

**확장 규칙:**
- "수능 간식" → ["초콜릿", "견과류", "에너지바", "홍삼"]
- "블랙프라이데이" → ["할인", "특가", "세일"]
- "김장 재료" → ["김치냉장고", "절임배추", "고춧가루"]
- "겨울 패션" → ["패딩", "기모", "코트", "목도리"]
- 이미 구체적이면 확장 불필요

**중요 지침:**
1. 브랜드명도 포함 가능: "락토핏", "종근당", "쿠쿠", "해피콜"
2. 상품 카테고리명: "건강식품", "생활가전", "의류", "식품"
3. 시즌 특화 상품: 11-12월이면 "크리스마스", "연말선물", "수능간식"
4. 다양한 카테고리 포함 (최소 3개 이상 카테고리)

**시간대별 카테고리 우선순위 및 가중치 (중요!):**

🌅 아침 (06:00-09:59):
- 매우 적합 (1.2): 건강식품, 일반식품, 주방용품
- 보통 (0.9): 의류, 가전
- 부적합 (0.8): 패션잡화, 신발
- 예: "오메가3"(1.2), "유산균"(1.2), "커피"(1.2), "패딩"(0.9)

🌞 점심 (10:00-13:59):
- 매우 적합 (1.2): 일반식품, 주방용품, 생활용품
- 보통 (0.9): 의류, 가전, 건강식품
- 부적합 (0.8): 패션잡화, 신발
- 예: "간편식"(1.2), "도시락"(1.2), "청소용품"(1.2), "패딩"(0.9)

🌤️ 오후 (14:00-17:59):
- 매우 적합 (1.2): 가구/침구, 생활용품, 가전
- 보통 (1.0): 건강식품, 의류, 식품
- 예: "침대"(1.2), "매트리스"(1.2), "청소기"(1.2), "패딩"(1.0)

🌙 저녁/밤 (18:00-05:59):
- 매우 적합 (1.2): 의류, 패션잡화/보석, 신발, 화장품/뷰티
- 적합 (1.1): 건강식품, 가전, 가구/침구
- 보통 (0.9): 식품, 주방용품
- 예: "패딩"(1.2), "기모"(1.2), "목도리"(1.2), "스킨케어"(1.2), "화장품"(1.2), "홍삼"(1.1)

**가중치 규칙:**
- 1.2: 해당 시간대에 매우 적합한 카테고리
- 1.1: 적합한 카테고리
- 1.0: 보통 (기본값)
- 0.9: 다소 부적합
- 0.8: 부적합

JSON 형식으로 반환 (각 키워드에 가중치 포함):
{{
  "keywords": [
    {{"keyword": "키워드1", "weight": 1.2}},
    {{"keyword": "키워드2", "weight": 1.0}},
    {{"keyword": "키워드3", "weight": 0.9}}
  ],
  "expanded": {{
    "추상키워드1": ["구체1", "구체2", "구체3"],
    "추상키워드2": ["구체1", "구체2"]
  }}
}}"""),
            ("human", """날짜: {month}월 {day}일
날씨: {weather}
기온: {temperature}도
시간대: {time_slot}
요일 타입: {day_type}
공휴일: {holiday_name}

위 상황에 적합한 상품 검색 키워드를 생성해주세요. 
**특히 시간대({time_slot})를 고려해서 해당 시간대에 적합한 카테고리의 키워드를 우선적으로 생성하세요!**""")
        ])
        
        chain = keyword_prompt | self.llm | JsonOutputParser()
        
        try:
            # 프롬프트 로깅 (눈에 띄게)
            prompt_vars = {
                "month": month,
                "day": day,
                "weather": weather,
                "temperature": temperature,
                "time_slot": time_slot,
                "day_type": day_type,
                "holiday_name": holiday_name if holiday_name else "없음"
            }
            print("=" * 80)
            print("[1단계 - LangChain 프롬프트] 기본 컨텍스트 키워드 생성")
            print("=" * 80)
            print(f"입력 변수:")
            for key, value in prompt_vars.items():
                print(f"  - {key}: {value}")
            print("=" * 80)
            logger.info(f"[1단계] 기본 컨텍스트 프롬프트 변수: {prompt_vars}")
            
            result = await chain.ainvoke({
                "month": month,
                "day": day,
                "weather": weather,
                "temperature": temperature,
                "time_slot": time_slot,
                "day_type": day_type,
                "holiday_name": holiday_name if holiday_name else "없음"
            })
            
            # 결과 파싱
            keyword_weights = {}  # 키워드별 가중치 저장
            
            if isinstance(result, dict):
                keywords_data = result.get("keywords", [])
                expansion_map = result.get("expanded", {})
                
                # 키워드와 가중치 분리
                keywords = []
                for item in keywords_data:
                    if isinstance(item, dict):
                        kw = item.get("keyword", "")
                        weight = item.get("weight", 1.0)
                        keywords.append(kw)
                        keyword_weights[kw] = weight
                    else:
                        # 폴백: 문자열로 온 경우
                        keywords.append(item)
                        keyword_weights[item] = 1.0
            else:
                # 폴백: 리스트로 온 경우
                keywords = result if isinstance(result, list) else []
                expansion_map = {}
                for kw in keywords:
                    keyword_weights[kw] = 1.0
            
            print("=" * 80)
            print(f"[1단계 - 응답] LLM 생성 키워드 (가중치 포함):")
            for kw in keywords[:10]:
                weight = keyword_weights.get(kw, 1.0)
                print(f"  - {kw}: {weight}x")
            print(f"[1단계 - 결과] 총 {len(keywords)}개 키워드")
            print("=" * 80)
            
            # 확장 키워드 처리 및 매핑 생성
            expanded_keywords = []
            keyword_mapping = {}
            
            print(f"[1단계 - 확장] LLM 확장 결과:")
            for original_kw in keywords:
                # 원본 키워드 추가 (가중치 유지)
                expanded_keywords.append(original_kw)
                keyword_mapping[original_kw] = original_kw
                
                # 확장된 키워드 추가
                if original_kw in expansion_map:
                    expanded_list = expansion_map[original_kw]
                    print(f"  🔄 '{original_kw}' → {expanded_list}")
                    expanded_keywords.extend(expanded_list)
                    
                    # 매핑 저장
                    for exp_kw in expanded_list:
                        keyword_mapping[exp_kw] = original_kw
            
            # 중복 제거
            expanded_keywords = list(dict.fromkeys(expanded_keywords))
            
            print("=" * 80)
            print(f"[1단계 - LLM 확장 완료] 원본 {len(keywords)}개 → 확장 {len(expanded_keywords)}개")
            print(f"[1단계 - 확장 키워드] {expanded_keywords}")
            print("=" * 80)
            
            # RAG 방식: 실제 DB 상품명 기반 키워드 재확장
            rag_keywords = await self._extract_keywords_from_actual_products(expanded_keywords)
            
            # RAG 키워드도 매핑에 추가 (원본 키워드로 역추적)
            for rag_kw in rag_keywords:
                if rag_kw not in keyword_mapping:
                    # RAG로 추출된 키워드는 가장 관련 있는 원본 키워드로 매핑
                    # 간단하게 첫 번째 원본 키워드로 매핑 (개선 가능)
                    keyword_mapping[rag_kw] = keywords[0] if keywords else rag_kw
            
            # 최종 키워드 순서 최적화:
            # 1. RAG 키워드 (최우선! 실제 DB 상품명 기반)
            # 2. 원본 키워드 (LLM 생성)
            # 3. LLM 확장 키워드 (보완)
            final_keywords = []
            
            # 1순위: RAG 키워드 (최우선!)
            final_keywords.extend(rag_keywords)
            
            # 2순위: 원본 키워드 (RAG에 없는 것만)
            for orig_kw in keywords:
                if orig_kw not in final_keywords:
                    final_keywords.append(orig_kw)
            
            # 3순위: LLM 확장 키워드 (RAG/원본에 없는 것만)
            for exp_kw in expanded_keywords:
                if exp_kw not in final_keywords:
                    final_keywords.append(exp_kw)
            
            # context에 매핑 정보 및 가중치 저장
            context["keyword_mapping"] = keyword_mapping
            context["original_keywords"] = keywords
            context["keyword_weights"] = keyword_weights  # 시간대별 가중치
            
            print("=" * 80)
            print(f"[1단계 - 최종 완료] 원본 {len(keywords)}개 → LLM {len(expanded_keywords)}개 → RAG {len(rag_keywords)}개 → 최종 {len(final_keywords)}개")
            print(f"[키워드 순서 최적화 - RAG 최우선!]")
            print(f"  🥇 1순위 (RAG): {rag_keywords[:5]}...")
            print(f"  🥈 2순위 (원본): {[k for k in keywords if k not in rag_keywords][:5]}...")
            print(f"  🥉 3순위 (확장): {[k for k in expanded_keywords if k not in rag_keywords and k not in keywords][:5]}...")
            print(f"[1단계 - 최종 키워드 순서] {final_keywords[:20]}...")
            print(f"[1단계 - 매핑] {len(keyword_mapping)}개 매핑 저장")
            print("=" * 80)
            
            logger.info(f"[1단계] 컨텍스트 기반 키워드 생성 완료: {keywords}")
            logger.info(f"[1단계] LLM 확장: {expanded_keywords}")
            logger.info(f"[1단계] RAG 추출: {rag_keywords[:10]}")
            logger.info(f"[1단계] 최종 키워드: {final_keywords[:15]}")
            logger.info(f"[1단계] 키워드 매핑: {len(keyword_mapping)}개")
            return final_keywords
        except Exception as e:
            logger.error(f"컨텍스트 키워드 생성 오류: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            # 폴백: 시간대/계절 기반 실용적 키워드
            fallback_keywords = []
            
            # 시간대별 키워드
            if time_slot == "저녁":
                fallback_keywords.extend(["저녁식사", "실내활동", "휴식", "가족시간"])
            elif time_slot == "오전":
                fallback_keywords.extend(["아침", "출근", "활력", "건강"])
            elif time_slot == "오후":
                fallback_keywords.extend(["점심", "야외활동", "운동", "쇼핑"])
            else:
                fallback_keywords.extend(["밤", "수면", "휴식"])
            
            # 계절별 키워드
            if season == "겨울":
                fallback_keywords.extend(["따뜻한", "보온", "난방"])
            elif season == "여름":
                fallback_keywords.extend(["시원한", "냉방", "휴가"])
            elif season == "봄":
                fallback_keywords.extend(["신선한", "야외", "꽃"])
            else:
                fallback_keywords.extend(["가을", "건강", "환절기"])
            
            print(f"[1단계 - 폴백] 폴백 키워드 사용: {fallback_keywords}")
            logger.info(f"[1단계] 폴백 키워드 사용: {fallback_keywords}")
            logger.info(f"[1단계] 폴백 키워드 개수: {len(fallback_keywords)}")
            return fallback_keywords
    
    # 주석: _expand_keywords_to_product_terms 함수는 제거됨
    # 이제 _generate_base_context_keywords에서 키워드 생성과 확장을 한 번에 처리
    
    async def _extract_keywords_from_actual_products(self, trend_keywords: List[str]) -> List[str]:
        """
        RAG 방식: 실제 DB 상품명 기반 키워드 추출
        
        1. 트렌드 키워드로 느슨하게 검색
        2. 검색된 실제 상품명 분석
        3. LLM으로 유용한 키워드 추출
        
        Returns:
            실제 DB에 존재하는 상품 기반 키워드 리스트
        """
        
        print("=" * 80)
        print("[RAG 키워드 추출] 실제 상품명 기반 키워드 추출 시작")
        print("=" * 80)
        
        try:
            # 1단계: 느슨한 검색 (상위 5개 키워드만 사용)
            query = " ".join(trend_keywords[:5])
            print(f"[1단계] 느슨한 검색 쿼리: {query}")
            
            search_results = self.product_embedder.search_products(
                trend_keywords=[query],
                top_k=30,  # 충분한 샘플
                score_threshold=0.25,  # 매우 낮은 threshold
                only_ready_products=True
            )
            
            if not search_results:
                print("[RAG] 검색 결과 없음 - 원본 키워드 반환")
                return trend_keywords
            
            # 2단계: 실제 상품명 추출
            actual_product_names = [
                result.get("product_name", "")
                for result in search_results[:20]  # 상위 20개만
            ]
            
            print(f"[2단계] 검색된 상품 {len(actual_product_names)}개:")
            for i, name in enumerate(actual_product_names[:5], 1):
                print(f"  {i}. {name[:50]}")
            
            # 3단계: LLM으로 키워드 추출
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 홈쇼핑 상품 검색 전문가입니다.

**임무**: 실제 DB 상품명들을 분석해서 검색에 유용한 키워드를 추출하세요.

**추출 규칙**:
1. 브랜드명 추출 (예: "쿠쿠", "필립스", "락토핏")
2. 상품 카테고리 (예: "압력솥", "에어프라이어", "유산균")
3. 핵심 키워드 (예: "IH", "XXL", "프로바이오틱스")
4. 중복 제거

**예시**:
상품명: "쿠쿠 IH 10인용 압력밥솥"
추출: ["쿠쿠", "압력솥", "밥솥", "IH"]

상품명: "필립스 에어프라이어 XXL 7.3L"
추출: ["필립스", "에어프라이어", "튀김기", "XXL"]

JSON 형식:
{{"keywords": ["키워드1", "키워드2", ...]}}""")
,
                ("human", """트렌드 키워드: {trend_keywords}

우리 DB에서 검색된 실제 상품명들:
{product_names}

위 상품명들을 분석해서 검색에 유용한 키워드 15-20개를 추출하세요.""")
            ])
            
            chain = extraction_prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "trend_keywords": ", ".join(trend_keywords[:5]),
                "product_names": "\n".join([f"{i+1}. {name}" for i, name in enumerate(actual_product_names)])
            })
            
            extracted_keywords = result.get("keywords", [])
            
            print("=" * 80)
            print(f"[3단계] LLM 추출 완료: {len(extracted_keywords)}개 키워드")
            print(f"[추출 키워드] {extracted_keywords[:10]}...")
            print("=" * 80)
            
            return extracted_keywords
            
        except Exception as e:
            logger.error(f"RAG 키워드 추출 오류: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            
            # 폴백: 원본 키워드 반환
            print(f"[RAG 실패] 원본 키워드 사용: {trend_keywords}")
            return trend_keywords
    
    async def _generate_context_keywords(self, context: Dict[str, Any]) -> List[str]:
        """통합 키워드 생성: 1단계(기본 컨텍스트) + 2단계(실시간 트렌드)"""
        
        print("=" * 80)
        print("[통합 키워드 생성] 1단계: 기본 컨텍스트 키워드")
        print("=" * 80)
        
        # 1단계: 기본 컨텍스트 키워드 (날씨, 시간대, 계절, 공휴일)
        base_keywords = await self._generate_base_context_keywords(context)
        logger.info(f"1단계 기본 키워드: {base_keywords}")
        
        print("=" * 80)
        print("[통합 키워드 생성] 2단계: 실시간 트렌드 키워드")
        print("=" * 80)
        
        # 2단계: 실시간 트렌드 키워드 (OpenAI Web Search)
        realtime_keywords = await self._get_realtime_trend_keywords()
        logger.info(f"2단계 실시간 트렌드: {realtime_keywords}")
        
        # 통합: RAG 최우선 유지! (base_keywords 내부 순서: RAG → 원본 → 확장)
        # 웹 트렌드는 보완용으로 뒤에 배치
        combined_keywords = base_keywords + realtime_keywords
        
        # 중복 제거 (순서 유지)
        unique_keywords = list(dict.fromkeys(combined_keywords))
        
        print("=" * 80)
        print(f"[통합 키워드 순서] RAG 최우선 → 원본 → 확장 → 웹 트렌드")
        print(f"  1순위 (RAG): {base_keywords[:5]}...")
        print(f"  보완 (웹): {realtime_keywords[:3]}...")
        print(f"[통합 키워드] 최종 {len(unique_keywords)}개")
        print(f"[최종 순서] {unique_keywords[:15]}...")
        print("=" * 80)
        logger.info(f"통합 키워드 생성 완료: {unique_keywords[:20]}")
        
        return unique_keywords
    
    async def _generate_unified_candidates(
        self,
        search_result: Dict[str, Any],
        context: Dict[str, Any],
        max_trend_match: int = 8,  # 유사도 기반 최대 개수 (의류 편중 방지)
        max_sales_prediction: int = 32  # 매출예측 기반 최대 개수 (다양성 확보)
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """통합 후보군 생성 - 모든 상품 XGBoost 예측 후 가중치 조정"""
        
        candidates = []
        seen_products = set()
        
        print(f"=== [DEBUG Unified Candidates] 후보군 생성 시작 (목표: 최대 {max_trend_match + max_sales_prediction}개) ===")
        
        # 1. 모든 검색 결과를 하나로 통합
        all_products = []
        all_products.extend(search_result["direct_products"])  # 고유사도 상품
        
        # 카테고리 그룹의 모든 상품도 추가
        for category, products in search_result["category_groups"].items():
            all_products.extend(products)
        
        print(f"=== [DEBUG] 통합된 상품 수: {len(all_products)}개 ===")
        
        # 2. 중복 제거 (상품코드 + 소분류 + 브랜드)
        unique_products = {}
        seen_category_brand_pairs = set()  # (소분류, 브랜드) 조합
        
        for product in all_products:
            product_code = product.get("product_code")
            category_sub = product.get("category_sub", "")
            brand = product.get("brand", "")
            
            # 상품코드 중복 체크
            if product_code in unique_products:
                continue
            
            # 소분류 + 브랜드 조합 중복 체크 (다양성 보장)
            category_brand_key = (category_sub, brand)
            if category_sub and brand and category_brand_key in seen_category_brand_pairs:
                logger.info(f"소분류+브랜드 중복 제외: {product.get('product_name', '')[:30]} (소분류: {category_sub}, 브랜드: {brand})")
                continue
            
            # 통과한 경우 추가
            unique_products[product_code] = product
            if category_sub and brand:
                seen_category_brand_pairs.add(category_brand_key)
        
        print(f"=== [DEBUG] 중복 제거 후: {len(unique_products)}개 (소분류+브랜드 다양성 보장) ===")
        
        # 3. 배치 예측 준비 (상위 40개 - 의류 편중 방지)
        products_list = list(unique_products.values())[:40]
        print(f"=== [DEBUG] 배치 예측 대상: {len(products_list)}개 ===")
        
        # 4. 배치 XGBoost 예측 (한 번에 처리)
        predicted_sales_list = await self._predict_products_sales_batch(products_list, context)
        
        # 5. 예측 결과와 상품 매칭 + 점수 계산
        for i, product in enumerate(products_list):
            similarity = product.get("similarity_score", 0.5)
            predicted_sales = predicted_sales_list[i]
            
            # 점수 계산 (유사도 vs 매출 가중치 조정)
            if similarity >= 0.7:
                # 고유사도: 유사도 가중치 높임
                final_score = (
                    similarity * 0.7 +  # 유사도 70%
                    (predicted_sales / 100000000) * 0.3  # 매출 30% (정규화: 1억 기준)
                )
                source = "trend_match"
                print(f"  [고유사도] {product.get('product_name')[:20]}: 유사도={similarity:.2f}, 매출={predicted_sales/10000:.0f}만원, 점수={final_score:.3f}")
            else:
                # 저유사도: 매출 가중치 높임
                final_score = (
                    similarity * 0.3 +  # 유사도 30%
                    (predicted_sales / 100000000) * 0.7  # 매출 70%
                )
                source = "sales_prediction"
                print(f"  [저유사도] {product.get('product_name')[:20]}: 유사도={similarity:.2f}, 매출={predicted_sales/10000:.0f}만원, 점수={final_score:.3f}")
            
            candidates.append({
                "product": product,
                "source": source,
                "similarity_score": similarity,
                "predicted_sales": predicted_sales,
                "final_score": final_score
            })
        
        # 4. 점수순 정렬
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        print(f"=== [DEBUG] 총 {len(candidates)}개 후보 생성 완료, 점수순 정렬됨 ===")
        
        # 5. 카테고리별 점수 계산 (내부 사용용)
        category_scores = {}
        category_sales = {}
        for candidate in candidates:
            category = candidate["product"].get("category_main", "기타")
            if category == "기타" or not category:
                continue
            if category not in category_sales:
                category_sales[category] = []
            category_sales[category].append(candidate["predicted_sales"])
        
        for category, sales_list in category_sales.items():
            avg_sales = sum(sales_list) / len(sales_list)
            category_scores[category] = {"predicted_sales": avg_sales}
        
        return candidates, category_scores
    
    async def _predict_categories_with_xgboost(
        self, 
        category_groups: Dict[str, List[Dict]], 
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """카테고리별 XGBoost 매출 예측"""
        
        category_scores = {}
        broadcast_dt = context["broadcast_dt"]
        
        for category, products in category_groups.items():
            if not products:
                continue
            
            try:
                # 대표 상품으로 카테고리 매출 예측
                representative_product = products[0]
                predicted_sales = await self._predict_product_sales(representative_product, context)
                
                # 카테고리 내 상품 수로 보정
                adjusted_sales = predicted_sales * min(len(products) / 5, 2.0)
                
                category_scores[category] = {
                    "predicted_sales": adjusted_sales,
                    "product_count": len(products),
                    "avg_similarity": sum(p.get("similarity_score", 0) for p in products) / len(products)
                }
                
                print(f"  - 카테고리 '{category}': {int(adjusted_sales/10000)}만원 (상품: {len(products)}개)")
                
            except Exception as e:
                logger.error(f"카테고리 '{category}' 예측 실패: {e}")
                category_scores[category] = {
                    "predicted_sales": 10000000,  # 기본값 1000만원
                    "product_count": len(products),
                    "avg_similarity": 0.4
                }
        
        return category_scores
    
    async def _generate_candidates(self, promising_categories: List[Any], trend_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """후보군 생성 및 통합 (레거시, 사용 안 함)"""
        candidates = []
        
        # 유망 카테고리에서 에이스 상품 선발
        for category in promising_categories[:3]:
            ace_products = await self._get_ace_products_from_category(category.name, 5)
            
            for product in ace_products:
                candidates.append({
                    "product": product,
                    "source": "category",
                    "base_score": product.get("predicted_sales_score", 0.5),
                    "trend_boost": 1.0
                })
        
        return candidates
    
    async def _rank_final_candidates(self, candidates: List[Dict[str, Any]], category_scores: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """최종 랭킹 계산 - 시즌 적합성 + 카테고리+브랜드 다양성 적용"""
        
        print(f"=== [DEBUG _rank_final_candidates] 이미 점수순으로 정렬된 {len(candidates)}개 후보 수신 ===")
        
        # 0. 시즌 적합성 필터링 (LLM 배치 판단) - 상위 40개 후보에 대해
        top_candidates = candidates[:40]  # 충분한 후보군 준비 (시즌 필터 + 중복 제거 고려)
        print(f"\n=== [시즌 적합성 검사] 상위 {len(top_candidates)}개 후보 검사 시작 ===")
        
        season_filtered = await self._filter_by_season_suitability(top_candidates, context)
        print(f"=== [시즌 적합성 검사] {len(top_candidates)}개 → {len(season_filtered)}개 (부적합 {len(top_candidates) - len(season_filtered)}개 제거) ===\n")
        
        # 1. 카테고리+브랜드 중복 제거 + 대분류 카테고리 쿼터 제한
        category_brand_seen = set()
        category_count = {}  # 대분류 카테고리별 개수
        filtered_candidates = []
        
        for candidate in season_filtered:
            product = candidate["product"]
            category = product.get("category_main", "Unknown")
            brand = product.get("brand", "Unknown")
            key = f"{category}_{brand}"
            
            # 1-1. 같은 카테고리+브랜드 조합은 1개만 허용 (다양성 보장)
            if key in category_brand_seen:
                print(f"  ⚠️ 브랜드 중복 제거: {product.get('product_name')[:30]} (카테고리: {category}, 브랜드: {brand})")
                continue
            
            # 1-2. 같은 대분류 카테고리는 최대 4개까지만 허용
            current_count = category_count.get(category, 0)
            if current_count >= 4:
                print(f"  ⚠️ 카테고리 쿼터 초과: {product.get('product_name')[:30]} (카테고리: {category}, 이미 {current_count}개)")
                continue
            
            # 통과: 후보에 추가
            filtered_candidates.append(candidate)
            category_brand_seen.add(key)
            category_count[category] = current_count + 1
        
        print(f"=== [다양성 필터링] {len(season_filtered)}개 → {len(filtered_candidates)}개 (중복 {len(season_filtered) - len(filtered_candidates)}개 제거) ===")
        print(f"=== [카테고리 분포] {category_count} ===")
        
        for i, candidate in enumerate(filtered_candidates[:5]):
            product = candidate['product']
            print(f"  {i+1}위: {product.get('product_name')[:25]} | {product.get('category_main', 'N/A')} | {product.get('brand', 'N/A')} (점수: {candidate['final_score']:.3f})")
        
        return filtered_candidates
    
    async def _filter_by_season_suitability(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """시즌 적합성 필터링 - LLM 배치 판단"""
        
        if not candidates:
            return []
        
        # 현재 날짜 정보 추출
        broadcast_dt = context.get("broadcast_dt")
        month = broadcast_dt.month if broadcast_dt else 11
        day = broadcast_dt.day if broadcast_dt else 19
        holiday_name = context.get("holiday_name")
        
        # 상품 정보 준비 (상품명 + 테이프명)
        products_info = []
        for i, candidate in enumerate(candidates):
            product = candidate["product"]
            products_info.append({
                "index": i,
                "product_name": product.get("product_name", ""),
                "tape_name": product.get("tape_name", ""),
                "category": product.get("category_main", "")
            })
        
        # LLM 프롬프트
        season_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 20년차 홈쇼핑 방송 편성 전문가입니다.
현재 날짜/계절에 어울리지 않는 상품을 찾아주세요.

**제외 기준 (상품의 실제 특성 중심):**

1. 명절 불일치: 특정 명절 상품인데 현재 명절과 맞지 않는 경우
   - 예: 11월에 "신년특집", "설날", "추석" 포함 상품
   - 예: 7월에 "크리스마스" 포함 상품
   - 선행 판매 허용: 12월 말 신년특집 ⭕, 12월 중순 크리스마스 ⭕, 8월 말 추석 ⭕

2. 계절/날씨 부적합 상품 (상품 특성으로만 판단):
   
   **겨울철(11월~2월) - 추운 날씨에 제외할 것:**
   - 여름 냉방: "냉감", "쿨링", "시원한", "냉방", "피서용", "여름용"
   - 여름 의류: "반팔", "반바지", "민소매", "샌들" (실내용 제외)
   - 여름 침구: "냉감 패드", "쿨매트"
   - 예: "쿨드림 냉감패드", "여름 반팔티", "피서용 선풍기"
   
   **겨울철(11월~2월) - 추운 날씨에 적합 (허용):**
   - 난방 상품: "전기장판", "전기담요", "온열", "난방", "보온"
   - 겨울 의류: "패딩", "기모", "겨울", "코트", "목도리", "장갑", "두꺼운"
   - 예: "전기매트", "온열마사지기", "패딩", "기모바지" → 모두 OK!
   
   **여름철(6월~8월) - 더운 날씨에 제외할 것:**
   - 난방 상품: "전기장판", "전기담요", "온열", "난방"
   - 겨울 의류: "패딩", "기모", "겨울", "두꺼운 코트", "목도리"
   - 예: "겨울 패딩", "기모 바지", "전기장판"
   
   **봄/가을(3~5월, 9~10월) - 환절기:**
   - 3~5월: 겨울 난방 상품 제외, 여름 냉방 상품 OK
   - 9~10월: 여름 냉방 상품 제외, 겨울 난방 상품 OK

**중요 - 시즌 코드(SS/FW)는 무시하세요:**
- "25SS", "24FW" 같은 코드는 참고만 하고, 상품의 실제 특성으로 판단
- 예: "25SS 기모 바지" → 기모가 있으면 겨울에 OK
- 예: "24FW 반팔티" → 반팔이면 겨울에 제외
- 예: "23SS 패딩" → 패딩이면 여름에 제외

# 선행 판매는 허용 (1~2주 전)
- 12월 말 신년특집 ⭕
- 12월 중순 크리스마스 케이크 ⭕
- 8월 말 추석선물세트 ⭕

**제외하지 말 것:**
- 사계절 상품: 건강식품, 생활용품, 식품, 가전 등
- 시즌 키워드가 없는 일반 상품

JSON 형식으로 제외할 상품의 인덱스 배열을 반환하세요:
{{
  "exclude_indices": [인덱스 배열],
  "reasons": {{
    "인덱스": "제외 이유"
  }}
}}"""),
            ("human", """현재 정보:
- 날짜: {month}월 {day}일
- 공휴일: {holiday_name}

상품 목록:
{products_list}

위 상품 중 현재 날짜/시즌에 적합하지 않은 상품의 인덱스를 찾아주세요.
예: 11월 중순이면 겨울 상품은 OK, 추석/설날 상품은 제외""")
        ])
        
        # 상품 목록 문자열 생성 (상품명 + 테이프명)
        products_list_str = "\n".join([
            f"{p['index']}. {p['product_name']}\n   테이프명: {p['tape_name']}\n   카테고리: {p['category']}"
            for p in products_info
        ])
        
        chain = season_prompt | self.llm | JsonOutputParser()
        
        try:
            result = await chain.ainvoke({
                "month": month,
                "day": day,
                "holiday_name": holiday_name if holiday_name else "없음",
                "products_list": products_list_str
            })
            
            exclude_indices = set(result.get("exclude_indices", []))
            reasons = result.get("reasons", {})
            
            # 제외된 상품 로그
            for idx in exclude_indices:
                if idx < len(candidates):
                    product_name = candidates[idx]["product"].get("product_name", "")[:40]
                    reason = reasons.get(str(idx), "시즌 부적합")
                    print(f"  ❌ 제외: {product_name} - {reason}")
            
            # 필터링
            filtered = [c for i, c in enumerate(candidates) if i not in exclude_indices]
            return filtered
            
        except Exception as e:
            logger.error(f"시즌 적합성 판단 오류: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            # 오류 시 모든 후보 반환 (안전장치)
            return candidates
    
    def _calculate_competition_penalty(self, product: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> float:
        """경쟁 페널티 점수 계산"""
        category = product.get("category_main", "")
        same_category_count = sum(1 for c in all_candidates if c["product"].get("category_main") == category)
        
        # 같은 카테고리 상품이 많을수록 페널티
        if same_category_count <= 2:
            return 0.0
        elif same_category_count <= 4:
            return 0.1
        else:
            return 0.2
    
    async def _format_response(self, ranked_products: List[Dict[str, Any]], context: Dict[str, Any] = None) -> BroadcastResponse:
        """API 응답 생성 (비동기)"""
        print(f"=== [DEBUG _format_response] context keys: {context.keys() if context else 'None'} ===")
        if context:
            print(f"=== [DEBUG _format_response] generated_keywords: {context.get('generated_keywords', [])} ===")
        
        # 1. 테이프 코드 목록 추출
        tape_codes = [p["product"].get("tape_code") for p in ranked_products if p["product"].get("tape_code")]
        
        # 2. 최근 방송 실적 배치 조회 (Netezza)
        broadcast_history_map = {}
        if tape_codes:
            logger.info(f" {len(tape_codes)}개 테이프의 최근 방송 실적 조회 중...")
            broadcast_history_map = self.broadcast_history_service.get_latest_broadcasts_batch(tape_codes)
            logger.info(f" {sum(1 for v in broadcast_history_map.values() if v is not None)}개 테이프의 실적 조회 성공")
        
        recommendations = []
        
        # 순위 정보 추가 (배치 처리 전)
        for i, candidate in enumerate(ranked_products):
            candidate["rank"] = i + 1
            candidate["total_count"] = len(ranked_products)
        
        # [5-1단계] 배치로 모든 상품의 추천 근거 생성 (한 번의 LLM 호출)
        step_5_1_start = time.time()
        print("\n" + "=" * 80)
        print(f"[5-1단계] LLM 배치 처리 - {len(ranked_products)}개 상품의 추천 근거 생성")
        print("=" * 80)
        reasoning_list = await self._generate_batch_reasons_with_langchain(
            ranked_products,
            context or {"time_slot": "저녁", "weather": {"weather": "폭염"}}
        )
        print(f"⏱️  [5-1단계] 추천 근거 생성: {time.time() - step_5_1_start:.2f}초")
        
        for i, candidate in enumerate(ranked_products):
            product = candidate["product"]
            reasoning_summary = reasoning_list[i] if i < len(reasoning_list) else f"{product.get('category_main', '상품')} 추천"
            
            # 최근 방송 실적 조회
            tape_code = product.get("tape_code")
            last_broadcast_data = broadcast_history_map.get(tape_code) if tape_code else None
            last_broadcast = None
            
            if last_broadcast_data:
                try:
                    last_broadcast = LastBroadcastMetrics(**last_broadcast_data)
                    logger.debug(f"✅ 테이프 {tape_code}의 최근 방송 실적 추가")
                except Exception as e:
                    logger.warning(f"⚠️ 테이프 {tape_code}의 실적 데이터 파싱 실패: {e}")
            
            recommendation = BroadcastRecommendation(
                rank=i+1,
                productInfo=ProductInfo(
                    productId=product.get("product_code", "Unknown"),
                    productName=product.get("product_name", "Unknown"),
                    category=product.get("category_main", "Unknown"),
                    categoryMiddle=product.get("category_middle"),
                    categorySub=product.get("category_sub"),
                    brand=product.get("brand"),
                    price=product.get("price"),
                    tapeCode=product.get("tape_code"),
                    tapeName=product.get("tape_name")
                ),
                reasoning=reasoning_summary,
                businessMetrics=BusinessMetrics(
                    aiPredictedSales=f"{round(candidate['predicted_sales']/10000, 1):,.1f}만원",  # AI 예측 매출 (XGBoost, 소수점 1자리)
                    lastBroadcast=last_broadcast  # 최근 방송 실적 추가
                )
            )

            # 추천 결과 요약 로그 (시연/분석용) - 다른 단계 로그와 동일하게 print 사용
            try:
                print(
                    f"[RECOMMENDATION] #{recommendation.rank} "
                    f"{recommendation.productInfo.productName} | "
                    f"카테고리: {recommendation.productInfo.category} | "
                    f"예측매출: {recommendation.businessMetrics.aiPredictedSales} | "
                    f"최종점수: {candidate.get('final_score', 0.0):.3f} | "
                    f"근거: {recommendation.reasoning}"
                )
            except Exception:
                # 로깅 오류가 추천 로직에 영향 주지 않도록 방어
                pass

            recommendations.append(recommendation)
        
        # [5-2단계] 네이버/타사 편성 조회 및 AI 선택
        step_5_2_start = time.time()
        print("\n" + "=" * 80)
        print(f"[5-2단계] 네이버/타사 편성 조회 및 AI 선택")
        print("=" * 80)
        
        # 네이버 베스트 상품 조회
        naver_products_data = self.external_products_service.get_latest_best_products(limit=10)
        naver_products = [NaverProduct(**product) for product in naver_products_data]
        logger.info(f"✅ 네이버 상품 {len(naver_products)}개 수집")
        print(f"✅ 네이버 상품 {len(naver_products)}개 수집")
        
        # 타 홈쇼핑사 편성 상품 조회 - Netezza에서 실시간 조회
        try:
            broadcast_time_str = context.get("broadcast_time") if context else None
            if broadcast_time_str:
                competitor_data = await netezza_conn.get_competitor_schedules(broadcast_time_str)
                competitor_products = [CompetitorProduct(**comp) for comp in competitor_data]
                logger.info(f"✅ 타사 편성 {len(competitor_products)}개 수집")
                print(f"✅ 타사 편성 {len(competitor_products)}개 수집")
            else:
                logger.warning(f"⚠️ broadcast_time이 context에 없음")
                competitor_products = []
        except Exception as e:
            logger.warning(f"⚠️ 타사 편성 조회 실패: {str(e)}")
            competitor_products = []
        
        # AI 기반 네이버/타사 편성 10개 선택 및 통합
        selected_competitor_products = await self._select_and_merge_top_10(
            naver_products=naver_products,
            competitor_products=competitor_products,
            broadcast_time=broadcast_time_str,
            context=context
        )
        print(f"⏱️  [5-2단계] 네이버/타사 선택: {time.time() - step_5_2_start:.2f}초")
        
        return BroadcastResponse(
            requestTime="",  # 메인에서 설정
            recommendations=recommendations,
            competitorProducts=selected_competitor_products
        )
    
    async def _select_and_merge_top_10(
        self,
        naver_products: List[NaverProduct],
        competitor_products: List[CompetitorProduct],
        broadcast_time: str,
        context: Dict[str, Any] = None
    ) -> List[CompetitorProduct]:
        """
        AI를 활용하여 네이버/타사 편성 중 10개를 선택하고 통합
        네이버:타사 = 5:5 비율 유지 (한쪽이 부족하면 다른쪽으로 채움)
        """
        try:
            # 1. 네이버 상품을 타사 편성 형식으로 변환
            naver_as_competitor = [
                self._convert_naver_to_competitor(naver, idx)
                for idx, naver in enumerate(naver_products)
            ]
            
            # 2. AI에게 10개 선택 요청
            selected_indices = await self._ai_select_top_10(
                naver_products=naver_products,
                competitor_products=competitor_products,
                broadcast_time=broadcast_time,
                context=context
            )
            
            # 3. 선택된 항목 추출 (타사 편성 먼저, 네이버 나중)
            result = []
            
            # 타사 선택 항목 (우선 배치)
            for idx in selected_indices.get("competitor_indices", []):
                if 0 <= idx < len(competitor_products):
                    result.append(competitor_products[idx])
            
            # 네이버 선택 항목 (뒤에 배치)
            for idx in selected_indices.get("naver_indices", []):
                if 0 <= idx < len(naver_as_competitor):
                    result.append(naver_as_competitor[idx])
            
            logger.info(f"✅ AI 선택 완료: 네이버 {len(selected_indices.get('naver_indices', []))}개 + 타사 {len(selected_indices.get('competitor_indices', []))}개 = 총 {len(result)}개")
            
            return result[:10]  # 최대 10개
            
        except Exception as e:
            logger.error(f"⚠️ AI 선택 실패, 폴백 로직 사용: {str(e)}")
            # 폴백: 네이버 5개 + 타사 5개 단순 선택
            return self._fallback_select_top_10(naver_products, competitor_products)
    
    def _convert_naver_to_competitor(self, naver: NaverProduct, index: int) -> CompetitorProduct:
        """네이버 상품을 타사 편성 형식(CompetitorProduct)으로 변환"""
        return CompetitorProduct(
            company_name="네이버 스토어",
            broadcast_title=f"[네이버 인기 {index + 1}위] {naver.name[:50]}",
            start_time="",  # 빈칸
            end_time="",    # 빈칸
            duration_minutes=None,
            category_main=""  # 네이버 상품에는 카테고리 정보 없음
        )
    
    async def _ai_select_top_10(
        self,
        naver_products: List[NaverProduct],
        competitor_products: List[CompetitorProduct],
        broadcast_time: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, List[int]]:
        """
        AI를 활용하여 네이버/타사 편성 중 10개의 인덱스를 선택
        """
        # 프롬프트 구성
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 20년 경력의 홈쇼핑 방송 편성 전문가입니다.

# 데이터 이해
- **네이버 인기 상품**: 현재 시점의 시장 트렌드를 반영한 실시간 베스트 상품 (시간 무관)
- **타사 홈쇼핑 편성**: 특정 방송 시간대의 실제 편성 정보 (시간 기반)

# 선택 기준
1. **비율**: 네이버:타사 = 5:5를 최대한 유지 (한쪽 부족 시 다른쪽으로 채움)
2. **시간 적합성**: 요청된 방송 시간대에 적합한 상품/편성 선택
3. **트렌드 반영**: 네이버 인기 상품을 통해 현재 시장 트렌드 파악
4. **카테고리 균형**: 다양한 카테고리로 시청자 선택폭 확대
5. **경쟁 분석**: 타사 편성을 참고하여 차별화 또는 벤치마킹

# 선택 전략
- 네이버 인기 상품 중 방송 시간대와 어울리는 트렌드 상품 선택
- 타사 편성 중 해당 시간대에 검증된 상품 카테고리 참고
- 현재 트렌드(네이버)와 실제 편성(타사)의 균형 유지

JSON 형식으로 응답:
{{
  "naver_indices": [인덱스 배열],
  "competitor_indices": [인덱스 배열],
  "selection_summary": {{
    "time_match": "시간대 적합성 판단",
    "diversity": "선택한 상품들의 다양성 설명",
    "trend_analysis": "트렌드 반영 방식"
  }},
  "selection_reason": "전체 선택 근거 2-3문장"
}}"""),
            ("user", """방송 시간: {broadcast_time}

네이버 인기 상품 ({naver_count}개):
{naver_summary}

타사 홈쇼핑 편성 ({competitor_count}개):
{competitor_summary}

위 정보를 종합하여 방송 시간({broadcast_time})에 최적화된 10개를 선택하세요.""")
        ])
        
        # 네이버 상품 요약
        naver_summary = "\n".join([
            f"[{i}] {p.name[:40]} | 가격: {p.sale_price:,}원 | 할인: {p.discount_ratio}% | 판매량: {p.cumulation_sale_count}"
            for i, p in enumerate(naver_products[:20])  # 최대 20개만 전달
        ])
        
        # 타사 편성 요약
        competitor_summary = "\n".join([
            f"[{i}] {c.company_name} | {c.broadcast_title[:40]} | {c.start_time} ~ {c.end_time} | {c.category_main or '미분류'}"
            for i, c in enumerate(competitor_products[:20])  # 최대 20개만 전달
        ])
        
        # LLM 호출
        chain = prompt_template | self.llm | JsonOutputParser()
        
        result = await chain.ainvoke({
            "broadcast_time": broadcast_time or "미지정",
            "naver_count": len(naver_products),
            "competitor_count": len(competitor_products),
            "naver_summary": naver_summary or "없음",
            "competitor_summary": competitor_summary or "없음"
        })
        
        logger.info(f"AI 선택 근거: {result.get('selection_reason', '없음')}")
        
        return result
    
    def _fallback_select_top_10(
        self,
        naver_products: List[NaverProduct],
        competitor_products: List[CompetitorProduct]
    ) -> List[CompetitorProduct]:
        """AI 실패 시 폴백: 단순 5:5 선택 (타사 먼저, 네이버 나중)"""
        result = []
        
        # 타사 5개 (또는 가능한 만큼) - 우선 배치
        competitor_count = min(5, len(competitor_products))
        for i in range(competitor_count):
            result.append(competitor_products[i])
        
        # 네이버 5개 (또는 가능한 만큼) - 뒤에 배치
        naver_count = min(5, len(naver_products))
        for i in range(naver_count):
            result.append(self._convert_naver_to_competitor(naver_products[i], i))
        
        # 10개 미만이면 나머지로 채움
        if len(result) < 10:
            remaining = 10 - len(result)
            if competitor_count < len(competitor_products):
                for i in range(competitor_count, min(competitor_count + remaining, len(competitor_products))):
                    result.append(competitor_products[i])
            elif naver_count < len(naver_products):
                for i in range(naver_count, min(naver_count + remaining, len(naver_products))):
                    result.append(self._convert_naver_to_competitor(naver_products[i], i))
        
        logger.info(f"폴백 선택: 타사 우선, 총 {len(result)}개")
        return result[:10]
    
    def _generate_recommendation_reason(self, candidate: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """개선된 추천 근거 생성"""
        product = candidate["product"]
        source = candidate["source"]
        trend_boost = candidate.get("trend_boost", 1.0)
        predicted_sales = candidate.get("predicted_sales", 0)
        final_score = candidate.get("final_score", 0)
        
        # 기본 정보 추출
        category = product.get("category_main", "")
        product_name = product.get("product_name", "")
        trend_keyword = candidate.get("trend_keyword", "")
        tape_name = product.get("tape_name", "")
        
        # 시간대 정보
        time_slot = context.get("time_slot", "") if context else ""
        weather = context.get("weather", {}).get("weather", "") if context else ""
        
        # 근거 구성 요소들
        reasons = []
        
        # 1. 트렌드 관련 근거
        if source == "trend" and trend_keyword:
            if trend_boost > 1.3:
                reasons.append(f"'{trend_keyword}' 트렌드 급상승 반영")
            elif trend_boost > 1.1:
                reasons.append(f"'{trend_keyword}' 트렌드 상승세")
            else:
                reasons.append(f"'{trend_keyword}' 키워드 연관성")
        
        # 2. 카테고리 관련 근거
        elif source == "category":
            reasons.append(f"{category} 카테고리 유망 상품")
        
        # 3. 매출 예측 근거
        if predicted_sales > 80000000:  # 8천만원 이상
            reasons.append("높은 매출 예측")
        elif predicted_sales > 50000000:  # 5천만원 이상
            reasons.append("안정적 매출 예측")
        
        # 4. 시간대 적합성
        if time_slot and weather:
            if time_slot == "저녁" and category in ["건강식품", "화장품"]:
                reasons.append("저녁 시간대 최적")
            elif time_slot == "오후" and category in ["가전제품", "생활용품"]:
                reasons.append("오후 시간대 적합")
            elif weather == "폭염" and category in ["가전제품"] and "선풍기" in product_name:
                reasons.append("폭염 날씨 최적 상품")
        
        # 5. 방송테이프 정보
        if tape_name:
            reasons.append("방송테이프 준비 완료")
        
        # 6. AI 신뢰도
        if final_score > 0.8:
            reasons.append("AI 높은 신뢰도")
        elif final_score > 0.6:
            reasons.append("AI 추천 적합")
        
        # 근거가 없으면 기본 메시지
        if not reasons:
            reasons.append("종합 분석 결과 추천")
        
        # 최대 3개 근거만 사용
        return " + ".join(reasons[:3])
    
    def _generate_diverse_reason_templates(self, candidate: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """다양한 추천 근거 템플릿 생성"""
        product = candidate["product"]
        source = candidate["source"]
        trend_boost = candidate.get("trend_boost", 1.0)
        predicted_sales = candidate.get("predicted_sales", 0)
        
        # 기본 정보
        category = product.get("category_main", "")
        product_name = product.get("product_name", "")
        trend_keyword = candidate.get("trend_keyword", "")
        
        templates = []
        
        # 트렌드 기반 템플릿들
        if source == "trend" and trend_keyword:
            trend_templates = [
                f"'{trend_keyword}' 검색량 급증으로 높은 관심도 예상",
                f"실시간 '{trend_keyword}' 트렌드 반영한 타이밍 상품",
                f"'{trend_keyword}' 키워드 연관 상품으로 시청자 관심 집중",
                f"트렌드 '{trend_keyword}'와 완벽 매칭되는 최적 상품",
                f"'{trend_keyword}' 화제성 활용한 시의적절한 편성"
            ]
            templates.extend(trend_templates)
        
        # 매출 예측 기반 템플릿들
        sales_million = int(predicted_sales / 1000000)
        if sales_million > 80:
            sales_templates = [
                f"AI 예측 매출 {sales_million}백만원으로 최고 수익 기대",
                f"과거 데이터 분석 결과 {sales_million}백만원 매출 예상",
                f"머신러닝 모델 예측 {sales_million}백만원 고수익 상품"
            ]
        elif sales_million > 50:
            sales_templates = [
                f"안정적 {sales_million}백만원 매출 예측으로 리스크 최소화",
                f"검증된 {sales_million}백만원 수익 모델 상품",
                f"예측 매출 {sales_million}백만원으로 안전한 편성 선택"
            ]
        else:
            sales_templates = [
                "데이터 기반 매출 예측으로 검증된 상품",
                "AI 분석 결과 수익성 확인된 추천 상품",
                "과거 성과 데이터 기반 선별된 상품"
            ]
        templates.extend(sales_templates)
        
        # 카테고리 기반 템플릿들
        category_templates = [
            f"{category} 분야 베스트셀러 검증 상품",
            f"{category} 카테고리 내 경쟁력 1위 상품",
            f"{category} 시장에서 입증된 인기 상품",
            f"{category} 전문 상품으로 타겟 시청자 확보",
            f"{category} 분야 프리미엄 브랜드 상품"
        ]
        templates.extend(category_templates)
        
        # 날씨 기반 템플릿 (선택적, AI가 판단 못할 때만 사용)
        if context:
            weather = context.get("weather", {}).get("weather", "")
            
            # 극단적 날씨만 템플릿 제공 (AI 폴백용)
            if weather in ["폭염", "한파", "폭우", "폭설"]:
                weather_templates = [
                    f"{weather} 특수 상황 대응 상품",
                    f"현재 {weather} 상황에 필요한 아이템"
                ]
                templates.extend(weather_templates)
        
        # 방송테이프 기반 템플릿들
        tape_name = product.get("tape_name", "")
        if tape_name:
            tape_templates = [
                f"전용 방송테이프 '{tape_name}' 완벽 준비 완료",
                f"검증된 방송 콘텐츠로 시청자 몰입도 극대화",
                f"전문 제작 방송테이프로 상품 매력 완벽 전달"
            ]
            templates.extend(tape_templates)
        
        return templates
    
    async def _generate_fallback_response(self, request_time: str, recommendation_count: int) -> BroadcastResponse:
        """API 할당량 소진 시 임시 데이터로 추천 근거 시스템 테스트"""
        
        # 임시 상품 데이터 (데이터베이스에서 실제 존재하는 상품들)
        mock_products = [
            {
                "product_code": "P001",
                "product_name": "프리미엄 다이어트 보조제",
                "category_main": "건강식품",
                "tape_code": "T001",
                "tape_name": "프리미엄 다이어트 보조제"
            },
            {
                "product_code": "P002", 
                "product_name": "홈트레이닝 세트",
                "category_main": "스포츠용품",
                "tape_code": "T002",
                "tape_name": "홈트레이닝 세트 완전정복"
            },
            {
                "product_code": "P005",
                "product_name": "시원한 여름 선풍기",
                "category_main": "가전제품",
                "tape_code": "T005",
                "tape_name": "시원한 여름나기 선풍기"
            }
        ]
        
        # 임시 후보 데이터 생성
        mock_candidates = []
        for i, product in enumerate(mock_products[:recommendation_count]):
            candidate = {
                "product": product,
                "source": "trend" if i == 0 else "category",
                "base_score": 0.8 - i * 0.1,
                "trend_boost": 1.3 if i == 0 else 1.0,
                "predicted_sales": 85000000 - i * 15000000,
                "final_score": 0.85 - i * 0.1,
                "trend_keyword": "다이어트" if i == 0 else ""
            }
            mock_candidates.append(candidate)
        
        # 컨텍스트 생성
        context = {
            "time_slot": "저녁",
            "weather": {"weather": "폭염"},
            "competitors": []
        }
        
        # 개선된 추천 근거 시스템으로 응답 생성
        response = await self._format_response(mock_candidates, context)
        response.requestTime = request_time
        
        logger.info(f"폴백 응답 생성 완료: {len(mock_candidates)}개 추천 (추천 근거 시스템 테스트)")
        return response
    
    async def _generate_batch_reasons_with_langchain(self, candidates: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[str]:
        """배치로 여러 상품의 추천 근거를 한 번에 생성 (속도 개선)"""
        try:
            # 컨텍스트 정보
            time_slot = context.get("time_slot", "") if context else ""
            weather = context.get("weather", {}).get("weather", "") if context else ""
            holiday_name = context.get("holiday_name") if context else None
            
            # 키워드 매핑 정보 (확장된 키워드 → 원본 키워드)
            keyword_mapping = context.get("keyword_mapping", {}) if context else {}
            original_keywords = context.get("original_keywords", []) if context else []
            
            # 상품 정보 요약
            products_summary = []
            for candidate in candidates:
                product = candidate["product"]
                rank = candidate.get("rank", 0)
                predicted_sales = candidate.get("predicted_sales", 0)
                similarity_score = candidate.get("similarity_score", 0)
                final_score = candidate.get("final_score", 0)
                trend_keyword = candidate.get("trend_keyword", "")
                
                # 트렌드 키워드의 원본 키워드 찾기
                original_keyword = keyword_mapping.get(trend_keyword, trend_keyword) if trend_keyword else ""
                
                products_summary.append({
                    "rank": rank,
                    "product_name": product.get("product_name", ""),
                    "category": product.get("category_main", ""),
                    "predicted_sales": int(predicted_sales/10000) if predicted_sales else 0,
                    "similarity_score": f"{similarity_score:.3f}",
                    "final_score": f"{final_score:.3f}",
                    "trend_keyword": trend_keyword,
                    "original_keyword": original_keyword  # 원본 키워드 추가
                })
            
            # 배치 프롬프트 생성
            batch_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 홈쇼핑 방송 편성 전문가입니다.
여러 상품의 추천 근거를 한 번에 작성하세요.

# 핵심 원칙
1. **각 상품마다 100자 이내** 간결하게 작성
2. 전문적이고 객관적인 톤 유지
3. 구체적인 수치와 데이터 활용
4. **각 상품마다 완전히 다른 관점과 표현 사용**
5. 같은 패턴이나 문구 반복 절대 금지

# 활용 가능한 요소들
- 예측 매출 수치 (필수)
- 카테고리 특성 (필수)
- **원본 키워드** (매우 중요! 있을 경우 반드시 활용)
  * 예: 상품이 "초콜릿"이고 원본 키워드가 "수능 간식"이면
    → "수능 간식으로 적합한 초콜릿"처럼 표현
  * 예: 상품이 "패딩"이고 원본 키워드가 "겨울 패션"이면
    → "겨울 패션 트렌드에 맞는 패딩"처럼 표현
- 공휴일 (있을 경우 필수 언급)
- 시간대 특성 (저녁/오전/오후) - 신중하게 판단
- 날씨/계절 (선택적)

# 금지 사항
- "AI 분석 결과"로 시작하지 마세요
- 템플릿처럼 보이는 반복적 표현 금지
- 과장된 표현 금지
- 기술 용어 절대 사용 금지 (유사도, 점수, 비율 등)

JSON 형식으로 응답:
{{
  "reasons": [
    "1번 상품 추천 근거",
    "2번 상품 추천 근거",
    ...
  ]
}}""")
,
                ("human", """시간대: {time_slot}
날씨: {weather}
공휴일: {holiday_name}

추천 상품 목록:
{products_info}

위 {count}개 상품 각각에 대해 독창적인 추천 근거를 작성하세요.
**원본 키워드가 있으면 반드시 활용하세요!**""")
            ])
            
            # 상품 정보 포맷팅 (원본 키워드 포함)
            products_info = "\n".join([
                f"{p['rank']}. {p['product_name'][:40]} | 카테고리: {p['category']} | 예측매출: {p['predicted_sales']}만원 | 원본키워드: {p['original_keyword'] or '없음'}"
                for p in products_summary
            ])
            
            chain = batch_prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "time_slot": time_slot or "미지정",
                "weather": weather or "보통",
                "holiday_name": holiday_name if holiday_name else "없음",
                "products_info": products_info,
                "count": len(candidates)
            })
            
            reasons = result.get("reasons", [])
            print(f"✅ 배치 처리 완료: {len(reasons)}개 근거 생성")
            
            # 개수가 부족하면 기본 메시지로 채움
            while len(reasons) < len(candidates):
                idx = len(reasons)
                reasons.append(f"{candidates[idx]['product'].get('category_main', '상품')} 추천")
            
            return reasons[:len(candidates)]
            
        except Exception as e:
            logger.error(f"배치 근거 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            # 폴백: 개별 생성
            print("⚠️ 배치 처리 실패, 개별 생성으로 폴백...")
            return await self._generate_reasons_fallback(candidates, context)
    
    async def _generate_reasons_fallback(self, candidates: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[str]:
        """배치 실패 시 폴백: 개별 생성"""
        reasons = []
        for candidate in candidates:
            reason = await self._generate_dynamic_reason_with_langchain(candidate, context)
            reasons.append(reason)
        return reasons
    
    async def _generate_dynamic_reason_with_langchain(self, candidate: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """LangChain을 활용한 동적 추천 근거 생성 (개별, 폴백용)"""
        try:
            product = candidate["product"]
            source = candidate["source"]
            predicted_sales = candidate.get("predicted_sales", 0)
            similarity_score = candidate.get("similarity_score", 0)
            final_score = candidate.get("final_score", 0)
            rank = candidate.get("rank", 0)
            
            # 상품 정보
            category = product.get("category_main", "")
            product_name = product.get("product_name", "")
            trend_keyword = candidate.get("trend_keyword", "")
            
            # 컨텍스트 정보
            time_slot = context.get("time_slot", "") if context else ""
            weather = context.get("weather", {}).get("weather", "") if context else ""
            holiday_name = context.get("holiday_name") if context else None
            competitors = context.get("competitors", []) if context else []
            
            # 경쟁 상황 분석
            competitor_categories = [comp.get("category_main", "") for comp in competitors]
            has_competition = category in competitor_categories
            
            # 점수 분석 (실제 가중치 기반)
            if similarity_score >= 0.7:
                # 고유사도: 유사도 70%, 매출 30%
                similarity_ratio = 0.7
                sales_ratio = 0.3
            else:
                # 저유사도: 유사도 30%, 매출 70%
                similarity_ratio = 0.3
                sales_ratio = 0.7
            
            # 프롬프트 로깅 (눈에 띄게)
            print("=" * 80)
            print("[LLM 프롬프트] 추천 근거 생성")
            print("=" * 80)
            print(f"순위: {rank}위 | 추천 타입: {source}")
            print(f"상품: {product_name}, 카테고리: {category}")
            print(f"유사도: {similarity_score:.3f} | 매출: {int(predicted_sales/10000)}만원 | 최종점수: {final_score:.3f}")
            print(f"점수 구성: 유사도 {similarity_ratio*100:.0f}% / 매출 {sales_ratio*100:.0f}%")
            print(f"시간대: {time_slot}, 날씨: {weather}, 공휴일: {holiday_name or '없음'}")
            print("=" * 80)
            
            # 프롬프트 템플릿 생성
            reason_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 홈쇼핑 방송 편성 전문가입니다. 
주어진 데이터를 바탕으로 각 상품마다 독창적이고 설득력 있는 추천 근거를 작성하세요.

# 핵심 원칙
1. **100자 이내** 간결하게 작성
2. 전문적이고 객관적인 톤 유지
3. 구체적인 수치와 데이터 활용
4. **각 상품마다 완전히 다른 관점과 표현 사용**
5. 같은 패턴이나 문구 반복 절대 금지

# 활용 가능한 요소들
- 예측 매출 수치 (필수)
- 카테고리 특성 (필수)
- 점수 구성 비율 (유사도 vs 매출)
- 트렌드 키워드 (있을 경우)
- 공휴일 (있을 경우 필수 언급)
- 시간대 특성 (저녁/오전/오후) - **신중하게 판단**
  * 이 상품 카테고리가 해당 시간대에 실제로 적합한지 스스로 판단하세요
  * 예: 건강식품은 아침/저녁 적합, 의류는 낮 시간 적합, 가전은 저녁 적합
  * 확신이 없으면 시간대 언급하지 말고 다른 근거 사용
- 날씨/계절 (선택적, 과도한 반복 금지)

# 금지 사항 (답변에 절대 포함하지 말 것)
- "AI 분석 결과"로 시작하지 마세요
- 템플릿처럼 보이는 반복적 표현 금지
- 과장된 표현 (대박, 최고, 강추 등)
- 감정적 표현 (기쁘게, 행복하게 등)
- **기술 용어 절대 사용 금지**: 
  * "유사도", "유사도 점수", "similarity"
  * "매출 비중", "점수 구성", "70%", "30%", "비율"
  * "최종 점수", "final score"
  * 이런 내부 지표들을 절대 답변에 포함하지 마세요

# 창의적 작성 가이드
- **상품명의 특징을 활용** (브랜드, 수량, 특수성 등)
- 매출 수치를 다양한 방식으로 표현
- 시간대를 다르게 표현 (황금시간대, 주시청시간 등)
- 카테고리 특성을 창의적으로 활용
- 점수 구성에 따라 강조점을 다르게
- **각 상품마다 완전히 다른 각도에서 접근**
- **절대 이전 응답과 비슷한 패턴 사용 금지**"""),
    
    ("human", """
상품명: {product_name}
카테고리: {category}
추천 순위: {rank}위
추천 타입: {source}
예측 매출: {predicted_sales}만원
유사도 점수: {similarity_score}
최종 점수: {final_score}
점수 구성: 유사도 {similarity_ratio}% / 매출 {sales_ratio}%
시간대: {time_slot}
날씨: {weather}
공휴일: {holiday_name}
트렌드 키워드: {trend_keyword}

위 데이터를 분석하여 이 상품만의 독특한 추천 근거를 작성하세요.

**중요:**
- 다른 상품들과 완전히 다른 시작 문구 사용
- 같은 단어나 표현 반복 금지
- 공휴일이 있으면 반드시 언급
- 점수 구성 비율에 따라 강조점 다르게
- 100자 이내로 작성

추천 근거:""")
            ])
            
            chain = reason_prompt | self.llm
            
            result = await chain.ainvoke({
                "product_name": product_name,
                "category": category,
                "rank": rank,
                "source": source,  # "trend_match" 또는 "sales_prediction"
                "predicted_sales": int(predicted_sales/10000) if predicted_sales else "없음",
                "similarity_score": f"{similarity_score:.3f}",
                "final_score": f"{final_score:.3f}",
                "similarity_ratio": f"{similarity_ratio*100:.0f}",
                "sales_ratio": f"{sales_ratio*100:.0f}",
                "time_slot": time_slot or "미지정",
                "weather": weather or "보통",
                "holiday_name": holiday_name if holiday_name else "없음",
                "trend_keyword": trend_keyword or "없음"
            })
            
            return result.content.strip()
            
        except Exception as e:
            logger.error(f"동적 근거 생성 오류: {e}")
            import traceback
            traceback.print_exc()  # 에러 상세 로그
            # 폴백: 간단한 기본 메시지 (템플릿 아닌)
            return f"{candidate['product'].get('category_main', '상품')} 추천"
    
    def _prepare_features_for_product(self, product: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """1개 상품의 XGBoost feature 준비 (예측은 안 함)"""
        broadcast_dt = context["broadcast_dt"]
        
        print(f"=== [_prepare_features_for_product] 호출됨: {product.get('product_name', 'Unknown')[:30]} ===")
        
        # 로그 스케일링 적용 (학습 시와 동일)
        product_price = product.get("product_price", product.get("price", 100000))
        product_price_log = np.log1p(product_price)
        
        category_main = product.get("category_main", product.get("category", "Unknown"))
        time_slot = context["time_slot"]
        
        return {
            # Numeric features (단순화)
            "product_price_log": product_price_log,
            "hour": broadcast_dt.hour,
            "temperature": context["weather"].get("temperature", 20),
            "precipitation": context["weather"].get("precipitation", 0),
            
            # Categorical features
            "product_lgroup": category_main,
            "product_mgroup": product.get("category_middle", "Unknown"),
            "product_sgroup": product.get("category_sub", "Unknown"),
            "brand": product.get("brand", "Unknown"),
            "product_type": product.get("product_type", "유형"),
            "time_slot": time_slot,
            "day_of_week": ["월", "화", "수", "목", "금", "토", "일"][broadcast_dt.weekday()],
            "season": context["season"],
            "weather": context["weather"].get("weather", "Clear"),
            
            # Boolean features
            "is_weekend": 1 if broadcast_dt.weekday() >= 5 else 0,
            "is_holiday": 0
        }
    
    async def _predict_product_sales(self, product: Dict[str, Any], context: Dict[str, Any]) -> float:
        """개별 상품 XGBoost 매출 예측"""
        try:
            import pandas as pd
            
            # Feature 준비
            features = self._prepare_features_for_product(product, context)
            product_data = pd.DataFrame([features])
            
            logger.info(f"=== XGBoost 매출 예측 입력 데이터 ===")
            logger.info(f"상품: {product.get('product_name', 'Unknown')}")
            logger.info(f"카테고리: {product.get('category_main', 'Unknown')}")
            logger.info(f"가격: {product.get('product_price', 100000):,}원")
            logger.info(f"과거 평균 매출: {product.get('avg_sales', 30000000):,}원")
            logger.info(f"방송 시간: {context['broadcast_dt'].hour}시")
            logger.info(f"날씨: {context['weather'].get('weather', 'Clear')}, {context['weather'].get('temperature', 20)}°C")
            
            # XGBoost 파이프라인으로 예측 (전처리 포함)
            predicted_sales_log = self.model.predict(product_data)[0]
            # 로그 역변환 (학습 시 log1p 사용)
            predicted_sales = np.expm1(predicted_sales_log)
            logger.info(f"=== XGBoost 예측 결과 ===")
            logger.info(f"예측 매출: {predicted_sales:,.0f}원 ({predicted_sales/100000000:.2f}억)")
            
            return float(predicted_sales)
            
        except Exception as e:
            logger.error(f"상품 매출 예측 오류: {e}")
            logger.error(f"상품 정보: {product.get('product_name', 'Unknown')}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            return 30000000  # 기본값 (0.3억)
    
    async def _predict_products_sales_batch(self, products: List[Dict[str, Any]], context: Dict[str, Any]) -> List[float]:
        """여러 상품 XGBoost 매출 예측 (배치 처리)"""
        try:
            import pandas as pd
            
            if not products:
                return []
            
            # 모든 상품의 features를 한 번에 준비
            features_list = [
                self._prepare_features_for_product(product, context)
                for product in products
            ]
            
            batch_df = pd.DataFrame(features_list)
            
            print(f"=== [배치 예측] {len(products)}개 상품 일괄 예측 시작 ===")
            
            # 입력 피처 샘플 출력 (디버깅용)
            print(f"=== [입력 피처 샘플] ===")
            for i, (product, features) in enumerate(zip(products[:3], features_list[:3])):
                print(f"  상품 {i+1}: {product.get('product_name', '')[:30]}")
                print(f"    - product_price_log: {features['product_price_log']:.2f}")
                print(f"    - hour: {features['hour']}")
                print(f"    - 카테고리: {features['product_lgroup']}")
            
            # XGBoost 배치 예측 (한 번에 처리)
            predicted_sales_log = self.model.predict(batch_df)
            # 로그 역변환 (학습 시 log1p 사용)
            predicted_sales_array = np.expm1(predicted_sales_log)
            
            print(f"=== [배치 예측] 완료 ===")
            print(f"  평균: {predicted_sales_array.mean()/10000:.0f}만원")
            print(f"  최소: {predicted_sales_array.min()/10000:.0f}만원")
            print(f"  최대: {predicted_sales_array.max()/10000:.0f}만원")
            print(f"  표준편차: {predicted_sales_array.std()/10000:.0f}만원")
            
            # 예측 결과 샘플 출력
            print(f"=== [예측 결과 샘플] ===")
            for i, (product, sales) in enumerate(zip(products[:5], predicted_sales_array[:5])):
                print(f"  {i+1}. {product.get('product_name', '')[:30]:30s} → {sales/10000:.0f}만원")
            
            return [float(sales) for sales in predicted_sales_array]
            
        except Exception as e:
            logger.error(f"배치 매출 예측 오류: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            # 기본값 반환
            return [30000000.0] * len(products)
    
    async def _get_all_categories_from_db(self) -> List[str]:
        """PostgreSQL에서 모든 카테고리 조회"""
        try:
            query = text("""
                SELECT DISTINCT category_main
                FROM broadcast_training_dataset
                WHERE category_main IS NOT NULL
                ORDER BY category_main
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            
            categories = [row[0] for row in result]
            return categories
            
        except Exception as e:
            logger.error(f"전체 카테고리 조회 오류: {e}")
            return []
    
    async def _get_ace_products_from_category(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """카테고리별 에이스 상품 조회"""
        try:
            query = text("""
                SELECT product_code, product_name, category_main, category_middle, category_sub,
                       AVG(gross_profit) as avg_sales, COUNT(*) as broadcast_count,
                       tape_code, tape_name, MAX(price) as price, brand
                FROM broadcast_training_dataset 
                WHERE category_main = :category
                GROUP BY product_code, product_name, category_main, category_middle, category_sub,
                         tape_code, tape_name, brand
                ORDER BY avg_sales DESC 
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"category": category, "limit": limit}).fetchall()
                
            products = []
            for row in result:
                products.append({
                    "product_code": row[0],
                    "product_name": row[1],
                    "category_main": row[2],
                    "category_middle": row[3],
                    "category_sub": row[4],
                    "avg_sales": float(row[5]),
                    "broadcast_count": int(row[6]),
                    "tape_code": row[7],
                    "tape_name": row[8],
                    "price": float(row[9]) if row[9] else None,
                    "brand": row[10] if len(row) > 10 else None
                })
            
            return products
            
        except Exception as e:
            logger.error(f"에이스 상품 조회 오류: {e}")
            return []
    
    def _remove_duplicates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 - 같은 상품코드 및 같은 (소분류 + 브랜드) 조합 제거"""
        seen_products = set()
        seen_category_brand_pairs = set()  # (소분류, 브랜드) 조합
        unique_candidates = []
        
        for candidate in candidates:
            product_code = candidate.get("product_code", "")
            category_sub = candidate.get("category_sub", "")
            brand = candidate.get("brand", "")
            
            # 상품코드 중복 체크
            if product_code and product_code in seen_products:
                continue
            
            # 소분류 + 브랜드 조합 중복 체크
            category_brand_key = (category_sub, brand)
            if category_sub and brand and category_brand_key in seen_category_brand_pairs:
                logger.info(f"소분류+브랜드 중복 제외: {candidate.get('product_name', '')} (소분류: {category_sub}, 브랜드: {brand})")
                continue
            
            # 통과한 경우 추가
            if product_code:
                seen_products.add(product_code)
            if category_sub and brand:
                seen_category_brand_pairs.add(category_brand_key)
            unique_candidates.append(candidate)
        
        logger.info(f"중복 제거 완료: {len(candidates)}개 → {len(unique_candidates)}개 (소분류+브랜드 다양성 보장)")
        return unique_candidates
    
    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """후보 랭킹"""
        return sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)
    
    def _get_time_slot(self, dt: datetime) -> str:
        """시간대 분류"""
        hour = dt.hour
        if 6 <= hour < 9:
            return "아침"
        elif 9 <= hour < 12:
            return "오전"
        elif 12 <= hour < 14:
            return "점심"
        elif 14 <= hour < 18:
            return "오후"
        elif 18 <= hour < 22:
            return "저녁"
        else:
            return "야간"
    
    def _get_season(self, month: int) -> str:
        """계절 분류"""
        if 3 <= month <= 5:
            return "봄"
        elif 6 <= month <= 8:
            return "여름"
        elif 9 <= month <= 11:
            return "가을"
        else:
            return "겨울"
