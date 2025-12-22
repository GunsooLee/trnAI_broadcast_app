"""
방송 편성 AI 추천 워크플로우
LangChain 기반 2단계 워크플로우: AI 방향 탐색 + 고속 랭킹
"""

import asyncio
import calendar
import json
import logging
import time
from datetime import datetime, timedelta, date
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
from .schemas import BroadcastResponse, BroadcastRecommendation, ProductInfo, BusinessMetrics, NaverProduct, CompetitorProduct, LastBroadcastMetrics, RecommendationSource
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
                        weather_info=weather_info,
                        broadcast_date=broadcast_dt  # 방송 날짜 전달
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
        
        # 통합 키워드 생성 (컨텍스트 우선, 실시간 트렌드 보조)
        unified_keywords = []
        
        # 1. 컨텍스트 기반 키워드 생성 (날짜/시간/날씨 기반 - 우선순위 높음)
        context_keywords = await self._generate_context_keywords(context)
        if context_keywords:
            unified_keywords.extend(context_keywords)
            logger.info(f"[우선순위 1] 컨텍스트 키워드 {len(context_keywords)}개 추가")
        
        # 2. AI 트렌드 키워드 추가 (1단계 LLM - 시간대/날씨 기반 상품 키워드)
        if context.get("ai_trends"):
            ai_trend_limit = 10  # 3개 → 10개로 증가 (겨울 시즌 상품 반영)
            ai_keywords = context["ai_trends"][:ai_trend_limit]
            unified_keywords.extend(ai_keywords)
            print(f"[우선순위 2] AI 트렌드 키워드 {len(ai_keywords)}개 추가: {ai_keywords}")
            logger.info(f"[우선순위 2] AI 트렌드 키워드 {len(ai_keywords)}개 추가: {ai_keywords}")
        
        # 3. 실시간 웹 검색 트렌드 추가 (2단계 LLM - Web Search)
        print("=" * 80)
        print("[통합 키워드 생성] 2단계: 실시간 웹 검색 트렌드")
        print("=" * 80)
        try:
            realtime_result = await self._get_realtime_trend_keywords()
            # 튜플 반환 처리 (키워드, 뉴스출처)
            if isinstance(realtime_result, tuple):
                realtime_keywords, news_sources = realtime_result
            else:
                realtime_keywords = realtime_result if realtime_result else []
                news_sources = {}
            
            if realtime_keywords:
                unified_keywords.extend(realtime_keywords)
                context["realtime_trends"] = realtime_keywords  # 컨텍스트에도 저장
                context["news_sources"] = news_sources  # 뉴스 출처 정보 저장
                print(f"[2단계 완료] 실시간 웹 검색 키워드 {len(realtime_keywords)}개: {realtime_keywords}")
                print(f"[2단계 완료] 뉴스 출처 {len(news_sources)}개: {list(news_sources.keys())}")
                logger.info(f"[우선순위 3] 실시간 웹 검색 키워드 {len(realtime_keywords)}개 추가: {realtime_keywords}")
            else:
                context["realtime_trends"] = []
                context["news_sources"] = {}
        except Exception as e:
            print(f"[2단계 실패] {e}")
            logger.warning(f"실시간 웹 검색 트렌드 수집 실패 (무시): {e}")
            context["realtime_trends"] = []
            context["news_sources"] = {}
        
        # 4. 중복 제거 및 저장
        context["unified_keywords"] = list(dict.fromkeys(unified_keywords))  # 순서 유지 중복 제거
        
        # 통합 키워드 로그 출력
        print("=" * 80)
        print(f"[키워드 통합 완료] 총 {len(context['unified_keywords'])}개 키워드")
        print("=" * 80)
        print(f"[통합 키워드 전체]: {context['unified_keywords']}")
        print("=" * 80)
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
    
    def _get_historical_broadcast_periods(
        self, 
        target_date: date, 
        years_back: int = 5
    ) -> List[Tuple[date, date]]:
        """
        입력 날짜 기준 앞뒤 1개월씩 총 2개월 기간을 모든 과거 연도에 대해 생성
        
        Args:
            target_date: 기준 날짜
            years_back: 몇 년 전까지 조회할지 (기본 5년)
            
        Returns:
            [(시작일, 종료일), ...] 리스트
            
        Example:
            target_date = 2025-03-15 → 
            [(2024-02-15, 2024-04-15), (2023-02-15, 2023-04-15), ...]
        """
        periods = []
        current_year = target_date.year
        print(f"=== [DEBUG _get_historical_broadcast_periods] target_date: {target_date}, years_back: {years_back} ===")
        
        for year_offset in range(1, years_back + 1):
            target_year = current_year - year_offset
            
            try:
                # 1개월 전 계산
                start_month = target_date.month - 1
                start_year = target_year
                if start_month < 1:
                    start_month = 12
                    start_year -= 1
                
                # 1개월 후 계산
                end_month = target_date.month + 1
                end_year = target_year
                if end_month > 12:
                    end_month = 1
                    end_year += 1
                
                # 일자 처리 (월말 초과 방지)
                start_day = min(target_date.day, calendar.monthrange(start_year, start_month)[1])
                end_day = min(target_date.day, calendar.monthrange(end_year, end_month)[1])
                
                start_date = date(start_year, start_month, start_day)
                end_date = date(end_year, end_month, end_day)
                
                periods.append((start_date, end_date))
                
            except Exception as e:
                logger.warning(f"기간 계산 오류 (year_offset={year_offset}): {e}")
                print(f"=== [DEBUG] 기간 계산 오류: {e} ===")
                import traceback
                print(traceback.format_exc())
                continue
        
        print(f"=== [DEBUG _get_historical_broadcast_periods] 생성된 기간: {len(periods)}개 ===", flush=True)
        if periods:
            print(f"  첫 번째 기간: {periods[0]}", flush=True)
        return periods

    # _classify_keywords_with_langchain 함수 제거됨
    # 이제 _generate_base_context_keywords에서 키워드 생성과 확장을 통합 처리
    
    async def _execute_unified_search(self, context: Dict[str, Any], unified_keywords: List[str]) -> Dict[str, Any]:
        """다단계 Qdrant 검색: 과거 동일 시기 방송 상품 후보군 내에서 키워드 검색"""
        
        print(f"=== [DEBUG Multi-Stage Search] 시작, keywords: {len(unified_keywords)}개 ===")
        
        if not unified_keywords:
            logger.warning("통합 키워드 없음 - 빈 결과 반환")
            return {"direct_products": [], "category_groups": {}}
        
        try:
            # [주석처리] 0단계: 방송 기간 필터 계산 (입력 날짜 기준 앞뒤 1개월, 과거 5년)
            # TODO: 요구사항 확정 후 다시 활성화
            # broadcast_periods = None
            # target_date_str = context.get("broadcast_time", "")
            # print(f"=== [DEBUG] broadcast_time from context: {target_date_str} ===")
            # 
            # if target_date_str:
            #     try:
            #         # broadcast_time에서 날짜 추출 (예: "2025-03-15 20:00:00" → date(2025, 3, 15))
            #         if isinstance(target_date_str, str):
            #             target_dt = datetime.fromisoformat(target_date_str.replace("Z", "+00:00"))
            #         else:
            #             target_dt = target_date_str
            #         target_date = target_dt.date()
            #         
            #         # 과거 5년간 동일 시기 방송 기간 계산
            #         broadcast_periods = self._get_historical_broadcast_periods(target_date, years_back=5)
            #         
            #         if broadcast_periods:
            #             print(f"=== [방송 기간 필터] 기준일: {target_date}, {len(broadcast_periods)}개 기간 ===")
            #             for i, (start, end) in enumerate(broadcast_periods[:3]):
            #                 print(f"  - {start} ~ {end}")
            #             if len(broadcast_periods) > 3:
            #                 print(f"  - ... 외 {len(broadcast_periods) - 3}개 기간")
            #     except Exception as e:
            #         logger.warning(f"방송 기간 필터 계산 실패: {e}, 필터 없이 검색")
            #         broadcast_periods = None
            
            # 모든 키워드를 개별적으로 검색 (키워드별 다양성 확보)
            all_results = []
            seen_products = set()
            keyword_results = {}  # 키워드별 검색 결과 추적
            product_matched_keywords = {}  # 상품별 매칭된 키워드 추적
            
            print(f"=== [개별 키워드 검색] 총 {len(unified_keywords)}개 키워드 ===")
            
            for keyword in unified_keywords:
                # [주석처리] 방송 기간 필터 적용 로직 - 요구사항 확정 후 활성화
                # if broadcast_periods:
                #     results = self.product_embedder.search_products_with_broadcast_filter(
                #         trend_keywords=[keyword],
                #         broadcast_periods=broadcast_periods,
                #         top_k=10,
                #         score_threshold=0.3,
                #         only_ready_products=True
                #     )
                # else:
                results = self.product_embedder.search_products(
                    trend_keywords=[keyword],
                    top_k=5,
                    score_threshold=0.3,
                    only_ready_products=True
                )
                
                new_count = 0
                for r in results:
                    code = r.get("product_code")
                    if code not in seen_products:
                        # 상품에 매칭된 키워드 정보 추가
                        r["matched_keyword"] = keyword
                        all_results.append(r)
                        seen_products.add(code)
                        product_matched_keywords[code] = keyword
                        new_count += 1
                    elif code in product_matched_keywords:
                        # 이미 있는 상품이면 추가 키워드만 기록 (첫 번째 키워드가 가장 관련성 높음)
                        pass
                
                if new_count > 0:
                    keyword_results[keyword] = new_count
            
            # 검색 결과 요약 (전체 출력)
            print(f"=== [키워드별 검색 결과] 총 {len(keyword_results)}개 키워드 ===")
            for kw, count in keyword_results.items():
                print(f"  - {kw}: {count}개")
            
            # 유사도 기준 정렬
            all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            print(f"=== [검색 완료] 총 {len(all_results)}개 상품 (유사도순 정렬) ===")
            
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
        from datetime import datetime
        
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            
            # 1. [날짜 동적 계산]
            now = datetime.now()
            
            # 현재 연도/월
            current_year = now.year        # 2025

            one_month_ago = now - timedelta(days=30)
            search_start_date = one_month_ago.strftime("%Y-%m-%d")
            
            current_date_str = now.strftime("%Y년 %m월 %d일")


            # *** 제외할 과거 연도 계산 ***
            # (올해가 2025년이면 2024년 데이터는 배제하라고 시키기 위함)
            prev_year = current_year - 1   # 2024

            #target_period_str = f"{last_month_str} ~ {current_month_str}"
            
            prompt = f"""**[즉시 실행 명령]**
1. 지금 즉시 웹 검색 도구를 사용하여 뉴스 기사를 검색하세요.
2. 분석이나 사족 없이 **결과 JSON**만 출력하세요.


**[시점 정보 (자동 계산됨)]**
- **현재 시점:** {current_date_str}
- **검색 유효 시작일:** {search_start_date}

**[필수 제약 조건 1: 날짜 필터링]**
**날짜 필터링 (치명적):**
   - 검색 결과에서 **'{prev_year}년'** 또는 그 이전 날짜의 기사는 **절대 사용하지 마세요.**
   - 검색 쿼리를 생성할 때, 반드시 **`after:{search_start_date}`** 연산자를 쿼리 뒤에 붙여야 합니다. 
   (이 연산자를 쓰지 않으면 과거 기사가 검색되어 분석이 실패합니다.)

**[필수 제약 조건 2: DB 검색용 키워드 추출 (명사화)]**
- 뉴스 기사에서 **'구체적인 상품명'**, **'카테고리'**, **'브랜드'**만 추출하세요.
- **문장형 금지:** '~~하는 상황', '~~로 인한 품절' 같은 서술어와 조사를 모두 제거하세요.
- **하나의 기사(URL)에서 추출할 수 있는 키워드는 '최대 3개'로 제한합니다.**
- 5개의 키워드를 채우기 위해 최소 3개 이상의 서로 다른 기사를 참조하는 것을 권장합니다.

**[핵심 과제]**
당신은 대한민국 20년차 유통 전문 기자이자 홈쇼핑 MD입니다.
위 기간 동안 홈쇼핑 및 유통 업계에서 발생한 **가장 뜨거운 '트렌드 키워드' 5개**를 찾아내세요.

**[검색 키워드 조합 지침]**
정확한 단어 매칭뿐만 아니라, 아래와 같이 **연도 + 현상**을 조합하여 검색하세요.

*(권장 검색어 예시)*
- "홈쇼핑 주문 폭주 after:{search_start_date}"
- "유통 완판 대란 after:{search_start_date}"
- "홈쇼핑 인기 상품 after:{search_start_date}"

**[정답 필터링 & Fallback]**
1. **필수:** 반드시 '뉴스 기사'에 근거할 것. (블로그/카페 뇌피셜 제외)
2. **필수:** 하나의 기사(URL)에서 모든 키워드를 추출하지 마세요.
3. **Fallback:** 5개를 못 채워도 좋으니, **찾은 개수만큼만이라도** 출력하세요. (빈 배열 `[]` 금지)
4. **제외 대상:** 비실물 상품(앱, 주식, 부동산) 제외.
5. **제외 대상:** 기사 날짜를 확인하고 {search_start_date} 이전 기사일 경우 제외.
6. **제외 대상:** DB에서 상품 검색에 사용할 수 있는 **'구체적인 상품명'**, **'카테고리'**, **'브랜드'**만 추출. 아니면 제외.

**[출력 형식]**
반드시 아래 JSON 포맷을 지킬 것.
```json
{{
  "trend_keywords": ["키워드1", "키워드2", "키워드3"],
  "sources": [
    {{"keyword": "키워드1", "title": "기사제목...", "URL": "기사 출처 URL..."}},
    {{"keyword": "키워드2", "title": "기사제목...", "URL": "기사 출처 URL..."}}
  ]
}}

"""
            
            print("=" * 80)
            print("[2단계 - OpenAI Web Search] 실시간 트렌드 수집 시작")
            print("=" * 80)
            print(f"[프롬프트]\n{prompt}")
            print("=" * 80)
            logger.info(f"[2단계] 실시간 트렌드 프롬프트: {prompt[:200]}...")
            
            response = client.responses.create(
                model="gpt-5-nano",
                reasoning={"effort": "low"},
                instructions=f"당신은 한국어 데이터 분석가입니다. 반드시 `web_search` 도구를 사용하세요. 검색어 뒤에는 `after:{search_start_date}`를 붙여 최신 기사만 찾으세요. 결과는 오직 JSON만 출력하세요.",
                tools=[{
                    "type": "web_search",
                    "search_context_size": "high",
                    "user_location": {
                        "type": "approximate",
                        "country": "KR",
                        "timezone": "Asia/Seoul"
                    }
                }],
                tool_choice="required",  # 웹 검색 도구 사용 강제
                input=prompt,
                max_output_tokens=8000,
                max_tool_calls=15
            )
            
            result_text = response.output_text
            print("=" * 80)
            print(f"[2단계 - 응답 (전체)]")
            print("-" * 80)
            print(response.model_dump_json(indent=2))
            print("-" * 80)
            print(result_text)
            print("-" * 80)
            logger.info(f"[2단계] 실시간 트렌드 응답: {result_text}")
            
            # JSON 배열 추출 (```json 코드블록 내부 우선)
            import json
            import re
            
            # 1차: 전체 JSON 객체 파싱 시도
            try:
                # ```json 코드블록 제거
                clean_text = re.sub(r'```json\s*|\s*```', '', result_text)
                data = json.loads(clean_text)
                
                # trend_keywords 필드 추출
                if isinstance(data, dict) and 'trend_keywords' in data:
                    keywords = data['trend_keywords']
                    sources = data.get('sources', [])
                    print(f"[2단계 - 추출 성공] 키워드: {keywords}")
                    print(f"[2단계 - 뉴스 출처] {len(sources)}개 출처 정보")
                    
                    # 키워드별 뉴스 출처 매핑 반환 (튜플로)
                    keyword_sources = {}
                    for src in sources:
                        kw = src.get('keyword', '')
                        if kw:
                            keyword_sources[kw] = {
                                'news_title': src.get('title', ''),
                                'news_url': src.get('URL', src.get('url', ''))
                            }
                    
                    return keywords[:5], keyword_sources
                elif isinstance(data, list):
                    # 구버전 호환: 배열만 온 경우
                    print(f"[2단계 - 추출 성공] 키워드: {data}")
                    return data[:5], {}
            except json.JSONDecodeError:
                # 2차: 정규식으로 trend_keywords만 추출
                match = re.search(r'"trend_keywords"\s*:\s*(\[.*?\])', result_text, re.DOTALL)
                if match:
                    keywords = json.loads(match.group(1))
                    print(f"[2단계 - 추출 성공] 키워드: {keywords}")
                    return keywords[:5], {}
                
                print("[2단계 - 실패] JSON 추출 실패")
                return [], {}
                
        except Exception as e:
            print("=" * 80)
            print(f"[2단계 - 오류] {e}")
            print("=" * 80)
            logger.error(f"[2단계] 실시간 트렌드 수집 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], {}
    
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
            
            # 최종 키워드: 모든 키워드 통합 (중복 제거)
            final_keywords = []
            
            # 1순위: 원본 키워드 (LLM 생성)
            final_keywords.extend(keywords)
            
            # 2순위: LLM 확장 키워드
            for exp_kw in expanded_keywords:
                if exp_kw not in final_keywords:
                    final_keywords.append(exp_kw)
            
            # 3순위: RAG 키워드 (모두 포함)
            for rag_kw in rag_keywords:
                if rag_kw not in final_keywords:
                    final_keywords.append(rag_kw)
            
            # context에 매핑 정보 및 가중치 저장
            context["keyword_mapping"] = keyword_mapping
            context["original_keywords"] = keywords
            context["keyword_weights"] = keyword_weights  # 시간대별 가중치
            
            print("=" * 80)
            print(f"[1단계 - 최종 완료] 원본 {len(keywords)}개 + 확장 {len(expanded_keywords)}개 + RAG {len(rag_keywords)}개 → 최종 {len(final_keywords)}개")
            print(f"[키워드 통합 완료]")
            print(f"  - 원본: {keywords[:5]}...")
            print(f"  - 확장: {expanded_keywords[:5]}...")
            print(f"  - RAG: {rag_keywords[:5]}...")
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
1. 브랜드명 추출 (예: "쿠쿠", "필립스", "락토핏", "종근당건강")
2. 상품 카테고리 (예: "압력솥", "에어프라이어", "유산균", "오메가3")
3. 핵심 키워드 (예: "IH", "XXL", "프로바이오틱스", "알티지")
4. 중복 제거

**제외 규칙 (절대 포함하지 마세요)**:
- ❌ 숫자+단위 조합: "12개월", "15개월", "3박스", "6통", "18박스" 등
- ❌ 순수 숫자: "12", "15", "3" 등
- ❌ 접두사/접미사: "직_", "단_", "세일_", "[세일]", "[환원]" 등
- ❌ 의미없는 단어: "개월", "박스", "통", "개월분" 등

**예시**:
상품명: "종근당건강 프로메가 알티지비타민D 12개월"
추출: ["종근당건강", "프로메가", "알티지", "비타민D", "오메가3"]
(❌ "12개월" 제외)

상품명: "[세일]안국건강 초임계 알티지오메가3 12개월"
추출: ["안국건강", "초임계", "알티지", "오메가3"]
(❌ "[세일]", "12개월" 제외)

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
            
            # 후처리: 의미없는 키워드 필터링
            import re
            invalid_patterns = [
                r'^\d+개월분?$',      # "12개월", "15개월분"
                r'^\d+박스$',         # "3박스", "18박스"
                r'^\d+통$',           # "6통"
                r'^\d+$',             # 순수 숫자
                r'^[\[\(].*[\]\)]$',  # "[세일]", "(환원)" 등
                r'^직_',              # "직_" 접두사
                r'^단_',              # "단_" 접두사
                r'^세일_',            # "세일_" 접두사
            ]
            
            filtered_keywords = []
            removed_keywords = []
            for kw in extracted_keywords:
                is_invalid = False
                for pattern in invalid_patterns:
                    if re.match(pattern, kw):
                        is_invalid = True
                        removed_keywords.append(kw)
                        break
                if not is_invalid and len(kw) >= 2:  # 최소 2글자
                    filtered_keywords.append(kw)
            
            if removed_keywords:
                print(f"[3단계 - 필터링] 제거된 키워드: {removed_keywords}")
            
            print("=" * 80)
            print(f"[3단계] LLM 추출 완료: {len(extracted_keywords)}개 → 필터링 후 {len(filtered_keywords)}개")
            print(f"[추출 키워드] {filtered_keywords[:10]}...")
            print("=" * 80)
            
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"RAG 키워드 추출 오류: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            
            # 폴백: 원본 키워드 반환
            print(f"[RAG 실패] 원본 키워드 사용: {trend_keywords}")
            return trend_keywords
    
    async def _generate_context_keywords(self, context: Dict[str, Any]) -> List[str]:
        """컨텍스트 기반 키워드 생성 (1단계만 - 2단계는 상위에서 별도 호출)"""
        
        print("=" * 80)
        print("[통합 키워드 생성] 1단계: 기본 컨텍스트 키워드")
        print("=" * 80)
        
        # 1단계: 기본 컨텍스트 키워드 (날씨, 시간대, 계절, 공휴일)
        base_keywords = await self._generate_base_context_keywords(context)
        logger.info(f"1단계 기본 키워드: {base_keywords}")
        
        # 2단계(실시간 웹 검색)는 _collect_context_and_keywords()에서 별도 호출됨
        # 여기서는 1단계 결과만 반환
        
        print("=" * 80)
        print(f"[1단계 완료] 기본 컨텍스트 키워드 {len(base_keywords)}개")
        print(f"  키워드: {base_keywords[:10]}...")
        print("=" * 80)
        logger.info(f"컨텍스트 키워드 생성 완료: {base_keywords[:20]}")
        
        return base_keywords
    
    async def _generate_unified_candidates(
        self,
        search_result: Dict[str, Any],
        context: Dict[str, Any],
        max_trend_match: int = 8,  # 유사도 기반 최대 개수 (의류 편중 방지)
        max_sales_prediction: int = 32  # 매출예측 기반 최대 개수 (다양성 확보)
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """통합 후보군 생성 - Track A (키워드 매칭) + Track B (매출 상위) + Track C (과거 실적) 병합"""
        
        candidates = []
        seen_products = set()
        predicted_sales_cache = {}  # XGBoost 예측 캐시 (중복 예측 방지)
        
        broadcast_dt = context["broadcast_dt"]
        target_month = broadcast_dt.month
        target_hour = broadcast_dt.hour
        
        print(f"=== [DEBUG Unified Candidates] 후보군 생성 시작 (목표: 최대 {max_trend_match + max_sales_prediction}개) ===")
        print(f"=== [DEBUG] 대상 시간: {target_month}월 {target_hour}시 ===")
        
        # ========== Track A: 키워드 매칭 상품 ==========
        all_products = []
        # Track A 상품들에 source_tracks 초기화
        for product in search_result["direct_products"]:
            product["source_tracks"] = ["keyword"]  # 키워드 매칭
        all_products.extend(search_result["direct_products"])  # 고유사도 상품
        
        # 카테고리 그룹의 모든 상품도 추가
        for category, products in search_result["category_groups"].items():
            for product in products:
                product["source_tracks"] = ["keyword"]  # 키워드 매칭
            all_products.extend(products)
        
        print(f"=== [Track A] 키워드 매칭 상품: {len(all_products)}개 ===")
        
        # ========== Track B: 매출 예측 상위 상품 (키워드 무관) ==========
        sales_top_products = await self._get_sales_top_products(context, limit=20)
        print(f"=== [Track B] 매출 예측 상위 상품: {len(sales_top_products)}개 ===")
        
        # Track B 예측값 캐시에 저장 (중복 예측 방지)
        for product in sales_top_products:
            product_code = product.get("product_code")
            if product_code and "predicted_sales" in product:
                predicted_sales_cache[product_code] = product["predicted_sales"]
        
        # Track A + Track B 병합 (여러 출처 병합)
        for product in sales_top_products:
            product_code = product.get("product_code")
            existing = next((p for p in all_products if p.get("product_code") == product_code), None)
            if existing:
                # 기존 상품에 출처 추가 (리스트로 관리)
                if "source_tracks" not in existing:
                    existing["source_tracks"] = [existing.get("source_track", "keyword")]
                if "sales_top" not in existing["source_tracks"]:
                    existing["source_tracks"].append("sales_top")
            else:
                product["source_track"] = "sales_top"
                product["source_tracks"] = ["sales_top"]
                all_products.append(product)
        
        print(f"=== [DEBUG] Track A + B 통합: {len(all_products)}개 ===")
        
        # ========== Track C: 과거 유사 시간대/월 판매 실적 상품 ==========
        historical_products = self.product_embedder.get_historical_top_products(
            target_month=target_month,
            target_hour=target_hour,
            month_range=1,  # ±1개월 (예: 12월 → 11~1월)
            hour_range=1,   # ±1시간 (예: 9시 → 8~10시)
            limit=20
        )
        print(f"=== [Track C] 과거 유사 조건 상품: {len(historical_products)}개 (월: {target_month}±1, 시간: {target_hour}±1) ===")
        
        # Track C 상품 병합 (여러 출처 병합)
        track_c_added = 0
        track_c_updated = 0
        for product in historical_products:
            product_code = product.get("product_code")
            existing = next((p for p in all_products if p.get("product_code") == product_code), None)
            if existing:
                # 기존 상품에 출처 추가
                if "source_tracks" not in existing:
                    existing["source_tracks"] = [existing.get("source_track", "keyword")]
                if "historical" not in existing["source_tracks"]:
                    existing["source_tracks"].append("historical")
                existing["historical_avg_profit"] = product.get("historical_avg_profit", 0)
                existing["historical_broadcast_count"] = product.get("historical_broadcast_count", 0)
                track_c_updated += 1
            else:
                product["source_track"] = "historical"
                product["source_tracks"] = ["historical"]
                all_products.append(product)
                track_c_added += 1
        
        print(f"=== [DEBUG] Track A + B + C 통합: {len(all_products)}개 (Track C 신규: {track_c_added}개, 업데이트: {track_c_updated}개) ===")
        
        # 복합 출처 상품 디버그
        multi_source_products = [p for p in all_products if len(p.get("source_tracks", [])) > 1]
        if multi_source_products:
            print(f"=== [DEBUG] 복합 출처 상품 {len(multi_source_products)}개 ===")
            for p in multi_source_products[:3]:
                print(f"  - {p.get('product_name', '')[:25]}: {p.get('source_tracks', [])}")
        
        # ========== Track D: 경쟁사 편성 기반 RAG 검색 ==========
        competitor_products = await self._get_competitor_based_products(context, limit=15)
        print(f"=== [Track D] 경쟁사 대응 상품: {len(competitor_products)}개 ===")
        
        # Track D 상품 병합 (여러 출처 병합)
        track_d_added = 0
        track_d_updated = 0
        for product in competitor_products:
            product_code = product.get("product_code")
            existing = next((p for p in all_products if p.get("product_code") == product_code), None)
            if existing:
                # 기존 상품에 출처 추가
                if "source_tracks" not in existing:
                    existing["source_tracks"] = [existing.get("source_track", "keyword")]
                if "competitor" not in existing["source_tracks"]:
                    existing["source_tracks"].append("competitor")
                existing["competitor_info"] = product.get("competitor_info", {})
                track_d_updated += 1
            else:
                product["source_track"] = "competitor"
                product["source_tracks"] = ["competitor"]
                all_products.append(product)
                track_d_added += 1
        
        print(f"=== [DEBUG] Track A + B + C + D 통합: {len(all_products)}개 (Track D 신규: {track_d_added}개, 업데이트: {track_d_updated}개) ===")
        
        # 2. 중복 제거 전 정렬 (Track B/C/D 상품 우선)
        # 우선순위: competitor > sales_top > historical > keyword
        def get_track_priority(product):
            tracks = product.get("source_tracks", [])
            if "competitor" in tracks:
                return 0
            elif "sales_top" in tracks:
                return 1
            elif "historical" in tracks:
                return 2
            else:
                return 3
        
        all_products.sort(key=get_track_priority)
        
        # 2. 중복 제거 (상품코드 + 소분류 + 브랜드) - Track B/C/D 상품 우선 유지
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
        
        # 3. 배치 예측 준비 (상위 50개로 확대)
        products_list = list(unique_products.values())[:50]
        print(f"=== [DEBUG] 배치 예측 대상: {len(products_list)}개 ===")
        
        # 4. 배치 XGBoost 예측 (캐시 활용으로 중복 예측 방지)
        # Track B에서 이미 예측된 상품은 캐시에서 가져오고, 나머지만 새로 예측
        products_to_predict = []
        cached_indices = {}  # {index: cached_sales}
        
        for i, product in enumerate(products_list):
            product_code = product.get("product_code")
            if product_code in predicted_sales_cache:
                cached_indices[i] = predicted_sales_cache[product_code]
            else:
                products_to_predict.append((i, product))
        
        print(f"=== [DEBUG] 캐시 활용: {len(cached_indices)}개 캐시됨, {len(products_to_predict)}개 새로 예측 ===")
        
        # 새로 예측할 상품만 배치 예측
        predicted_sales_list = [0.0] * len(products_list)
        
        # 캐시된 값 먼저 채우기
        for idx, cached_sales in cached_indices.items():
            predicted_sales_list[idx] = cached_sales
        
        # 새로 예측할 상품이 있으면 배치 예측
        if products_to_predict:
            new_products = [p for _, p in products_to_predict]
            new_predictions = await self._predict_products_sales_batch(new_products, context)
            
            for j, (idx, _) in enumerate(products_to_predict):
                predicted_sales_list[idx] = new_predictions[j]
        
        # 5. 예측 결과와 상품 매칭 + 점수 계산 + 출처 정보 수집
        # 뉴스 출처 정보 가져오기
        news_sources = context.get("news_sources", {})
        realtime_trends = context.get("realtime_trends", [])
        ai_trends = context.get("ai_trends", [])
        context_keywords = context.get("context_keywords", [])  # 컨텍스트 기반 키워드
        keyword_mapping = context.get("keyword_mapping", {})
        unified_keywords = context.get("unified_keywords", [])
        
        print(f"=== [출처 추적] ai_trends: {ai_trends[:5]}... ===")
        print(f"=== [출처 추적] realtime_trends: {realtime_trends[:5]}... ===")
        print(f"=== [출처 추적] news_sources keys: {list(news_sources.keys())[:5]}... ===")
        
        for i, product in enumerate(products_list):
            similarity = product.get("similarity_score", 0.5)
            predicted_sales = predicted_sales_list[i]
            matched_keyword = product.get("matched_keyword", "")
            
            # 추천 출처 정보 수집
            recommendation_sources = []
            
            # 키워드 출처 판별
            keyword_source_type = "unknown"
            keyword_source_detail = ""
            
            if matched_keyword:
                # 출처 판별: 뉴스 > AI 트렌드 > 컨텍스트 순서로 확인
                if matched_keyword in news_sources:
                    keyword_source_type = "news"
                    keyword_source_detail = f"뉴스 트렌드에서 추출"
                elif matched_keyword in realtime_trends:
                    keyword_source_type = "news"
                    keyword_source_detail = f"실시간 웹 검색 트렌드"
                elif matched_keyword in ai_trends:
                    keyword_source_type = "ai"
                    keyword_source_detail = f"AI가 {context.get('time_slot', '')} {context.get('season', '')} 시즌에 맞게 생성"
                elif matched_keyword in context_keywords:
                    keyword_source_type = "context"
                    keyword_source_detail = f"날씨/시간대 기반 컨텍스트 키워드"
                else:
                    keyword_source_type = "ai"  # 기본값: AI 트렌드로 간주
                    keyword_source_detail = f"AI 트렌드 키워드"
                
                print(f"  [출처] {matched_keyword} → {keyword_source_type} ({keyword_source_detail})")
            
            # 1. RAG 매칭 출처 (키워드 출처 정보 포함)
            if matched_keyword:
                rag_source = {
                    "source_type": "rag_match",
                    "matched_keyword": matched_keyword,
                    "similarity_score": similarity,
                    "keyword_origin": keyword_source_type,  # 키워드가 어디서 왔는지
                    "keyword_origin_detail": keyword_source_detail
                }
                recommendation_sources.append(rag_source)
                
                # 2. 뉴스 트렌드 출처 (매칭된 키워드가 뉴스에서 온 경우)
                if matched_keyword in news_sources:
                    news_info = news_sources[matched_keyword]
                    news_source = {
                        "source_type": "news_trend",
                        "news_keyword": matched_keyword,
                        "news_title": news_info.get("news_title", ""),
                        "news_url": news_info.get("news_url", "")
                    }
                    recommendation_sources.append(news_source)
                
                # 3. AI 트렌드 출처 (매칭된 키워드가 AI 생성인 경우)
                if keyword_source_type == "ai" or matched_keyword in ai_trends:
                    ai_source = {
                        "source_type": "ai_trend",
                        "ai_keyword": matched_keyword,
                        "ai_reason": f"{context.get('time_slot', '')} 시간대 {context.get('season', '')} 시즌 트렌드 분석으로 생성"
                    }
                    recommendation_sources.append(ai_source)
            
            # 4. XGBoost 매출 예측 출처 (항상 추가)
            xgboost_source = {
                "source_type": "xgboost_sales",
                "xgboost_rank": i + 1,
                "predicted_sales": predicted_sales
            }
            recommendation_sources.append(xgboost_source)
            
            # 5. Track B 출처 (매출 예측 상위 - 키워드 무관)
            if product.get("source_track") == "sales_top":
                sales_top_source = {
                    "source_type": "sales_top",
                    "reason": "키워드 무관 매출 예측 상위 상품"
                }
                recommendation_sources.append(sales_top_source)
            
            # 6. Track C 출처 (과거 유사 시간대/월 판매 실적)
            if product.get("source_track") == "historical":
                historical_source = {
                    "source_type": "historical",
                    "reason": f"과거 {target_month}월±1, {target_hour}시±1 시간대에 실제로 잘 팔린 상품",
                    "historical_avg_profit": product.get("historical_avg_profit", 0),
                    "historical_broadcast_count": product.get("historical_broadcast_count", 0)
                }
                recommendation_sources.append(historical_source)
            
            # 7. Track D 출처 (경쟁사 편성 대응)
            if product.get("source_track") == "competitor":
                comp_info = product.get("competitor_info", {})
                competitor_source = {
                    "source_type": "competitor",
                    "competitor_company": comp_info.get("company", ""),
                    "competitor_title": comp_info.get("title", ""),
                    "competitor_keyword": product.get("matched_keyword", ""),
                    "reason": f"경쟁사 {comp_info.get('company', '')} 동시간대 편성 대응"
                }
                recommendation_sources.append(competitor_source)
            
            # 8. 컨텍스트 출처 (날씨, 시간대 등)
            context_factors = []
            if context.get("weather", {}).get("weather"):
                context_factors.append(f"날씨: {context['weather']['weather']}")
            if context.get("time_slot"):
                context_factors.append(f"시간대: {context['time_slot']}")
            if context.get("holiday_name"):
                context_factors.append(f"공휴일: {context['holiday_name']}")
            
            if context_factors:
                context_source = {
                    "source_type": "context",
                    "context_factor": ", ".join(context_factors)
                }
                recommendation_sources.append(context_source)
            
            # 점수 계산 (Track별 가산점 적용 - 여러 출처 합산)
            # 기본 점수: 매출 예측 기반 (유사도 비중 대폭 축소)
            base_score = (
                similarity * 0.2 +  # 유사도 20% (AI분석 비중 축소)
                (predicted_sales / 100000000) * 0.8  # 매출 80% (정규화: 1억 기준)
            )
            
            # 여러 출처에서 추천된 경우 가산점 합산
            source_tracks = product.get("source_tracks", [])
            track_bonus = 0.0
            source_labels = []  # 출처 라벨 리스트
            
            # Track별 가산점 (여러 출처면 합산)
            if "competitor" in source_tracks:
                track_bonus += 0.15
                source_labels.append("경쟁사")
            if "sales_top" in source_tracks:
                track_bonus += 0.12
                source_labels.append("매출상위")
            if "historical" in source_tracks:
                track_bonus += 0.10
                source_labels.append("과거실적")
            
            # keyword 출처 (뉴스 또는 AI분석) - 항상 표시
            # 뉴스 기준: 유사도 0.50 이상 (실제 트렌드 키워드 매칭)
            if "keyword" in source_tracks:
                if similarity >= 0.50:
                    track_bonus += 0.08
                    source_labels.append("뉴스")
                else:
                    # AI분석도 복합 출처로 표시 (가산점 없음)
                    source_labels.append("AI분석")
            
            # 출처가 없으면 AI분석 (패널티)
            if not source_labels:
                track_bonus = -0.05
                source_labels.append("AI분석")
            
            # 복합 출처 가산점 (2개 이상 출처면 추가 가산점)
            if len(source_labels) >= 2:
                track_bonus += 0.05 * (len(source_labels) - 1)  # 출처 1개 추가당 0.05
            
            # 대표 source 결정 (우선순위: 경쟁사 > 매출상위 > 과거실적 > 뉴스 > AI분석)
            if "competitor" in source_tracks:
                source = "competitor"
            elif "sales_top" in source_tracks:
                source = "sales_top"
            elif "historical" in source_tracks:
                source = "historical"
            elif similarity >= 0.7:
                source = "news_trend"
            else:
                source = "ai_trend"
            
            final_score = base_score + track_bonus
            
            # source_labels를 product에 저장 (추천 근거 생성용)
            product["source_labels"] = source_labels
            
            print(f"  [{'/'.join(source_labels)}] {product.get('product_name')[:20]}: 유사도={similarity:.2f}, 매출={predicted_sales/10000:.0f}만원, 가산점={track_bonus:.2f}, 점수={final_score:.3f}")
            
            candidates.append({
                "product": product,
                "source": source,
                "similarity_score": similarity,
                "predicted_sales": predicted_sales,
                "final_score": final_score,
                "recommendation_sources": recommendation_sources  # 추천 출처 정보 추가
            })
        
        # 4. 점수순 정렬
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        print(f"=== [DEBUG] 총 {len(candidates)}개 후보 생성 완료, 점수순 정렬됨 ===")
        
        # 4-1. Track별 최소 쿼터 보장 (source_labels 기반)
        # 라벨 → 쿼터 매핑
        label_quotas = {"경쟁사": 2, "매출상위": 2, "과거실적": 2, "뉴스": 2}
        label_counts = {"경쟁사": 0, "매출상위": 0, "과거실적": 0, "뉴스": 0, "AI분석": 0}
        
        final_candidates = []
        remaining_candidates = []
        
        # 먼저 각 라벨별로 쿼터만큼 선택 (source_labels 기반)
        for candidate in candidates:
            source_labels = candidate["product"].get("source_labels", ["AI분석"])
            selected = False
            
            # 쿼터가 남은 라벨이 있으면 선택
            for label in source_labels:
                if label in label_quotas and label_counts.get(label, 0) < label_quotas[label]:
                    final_candidates.append(candidate)
                    # 해당 상품의 모든 라벨 카운트 증가
                    for lbl in source_labels:
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    selected = True
                    break
            
            if not selected:
                remaining_candidates.append(candidate)
        
        # 나머지는 점수순으로 채움
        final_candidates.extend(remaining_candidates)
        
        # 다시 점수순 정렬
        final_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        print(f"=== [DEBUG] Track별 쿼터 적용 후: {len(final_candidates)}개 ===")
        print(f"  - 경쟁사: {label_counts.get('경쟁사', 0)}개, 매출상위: {label_counts.get('매출상위', 0)}개, 과거실적: {label_counts.get('과거실적', 0)}개, 뉴스: {label_counts.get('뉴스', 0)}개")
        
        candidates = final_candidates
        
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
    
    async def _get_sales_top_products(self, context: Dict[str, Any], limit: int = 20) -> List[Dict]:
        """
        Track B: 매출 예측 상위 상품 조회 (키워드 매칭 무관)
        방송테이프가 있는 전체 상품 중 XGBoost 매출 예측 상위 N개 반환
        """
        try:
            # 1. 방송테이프 있는 전체 상품 조회
            all_products = self.product_embedder.get_all_products_with_tape(limit=100)
            
            if not all_products:
                logger.warning("[Track B] 방송테이프 보유 상품 없음")
                return []
            
            print(f"=== [Track B] 방송테이프 보유 상품: {len(all_products)}개 ===")
            
            # 2. 배치 XGBoost 매출 예측
            predicted_sales_list = await self._predict_products_sales_batch(all_products, context)
            
            # 3. 매출 예측 결과와 상품 매칭
            products_with_sales = []
            for i, product in enumerate(all_products):
                product["predicted_sales"] = predicted_sales_list[i]
                products_with_sales.append(product)
            
            # 4. 매출 예측 내림차순 정렬
            products_with_sales.sort(key=lambda x: x.get("predicted_sales", 0), reverse=True)
            
            # 5. 상위 N개 반환
            top_products = products_with_sales[:limit]
            
            print(f"=== [Track B] 매출 예측 상위 {len(top_products)}개 선정 ===")
            for i, p in enumerate(top_products[:5], 1):
                print(f"  {i}. {p.get('product_name', '')[:30]} | 예측: {int(p.get('predicted_sales', 0)/10000)}만원")
            
            return top_products
            
        except Exception as e:
            logger.error(f"[Track B] 매출 상위 상품 조회 실패: {e}")
            return []
    
    async def _get_competitor_based_products(self, context: Dict[str, Any], limit: int = 15) -> List[Dict]:
        """
        Track D: 경쟁사 편성 기반 RAG 검색 (LLM 미사용)
        
        1. Netezza에서 경쟁사 편성 조회
        2. 편성 제목에서 키워드 추출 (코딩 방식)
        3. RAG 검색으로 유사 상품 찾기
        """
        try:
            broadcast_time_str = context.get("broadcast_time")
            if not broadcast_time_str:
                logger.warning("[Track D] broadcast_time이 context에 없음")
                return []
            
            # 1. 경쟁사 편성 조회
            competitor_data = await netezza_conn.get_competitor_schedules(broadcast_time_str)
            
            if not competitor_data:
                logger.info("[Track D] 경쟁사 편성 데이터 없음")
                return []
            
            print(f"=== [Track D] 경쟁사 편성 {len(competitor_data)}개 조회됨 ===")
            
            # 2. 편성 제목에서 키워드 추출 (LLM 미사용)
            competitor_keywords = []
            competitor_info = {}  # 키워드 → 경쟁사 정보 매핑
            
            for comp in competitor_data:
                title = comp.get("broadcast_title", "")
                company = comp.get("company_name", "")
                category = comp.get("category_main", "")
                start_time = comp.get("start_time", "")
                
                # 키워드 추출 (코딩 방식)
                keywords = self._extract_keywords_from_title(title, category)
                
                for kw in keywords:
                    if kw not in competitor_info:
                        competitor_keywords.append(kw)
                        competitor_info[kw] = {
                            "company": company,
                            "title": title[:40],  # 제목 40자로 제한
                            "start_time": str(start_time)[:16] if start_time else "",
                            "category": category,
                            "keyword": kw  # 매칭된 키워드 저장
                        }
            
            if not competitor_keywords:
                logger.info("[Track D] 경쟁사 편성에서 키워드 추출 실패")
                return []
            
            print(f"=== [Track D] 경쟁사 키워드 추출: {competitor_keywords[:10]}... ===")
            
            # 3. RAG 검색으로 유사 상품 찾기
            search_results = self.product_embedder.search_products(
                trend_keywords=competitor_keywords[:10],  # 상위 10개 키워드만
                top_k=limit,
                score_threshold=0.3,
                only_ready_products=True
            )
            
            # 4. 경쟁사 정보 추가 - 첫 번째 키워드 정보를 기본으로 사용
            first_keyword = competitor_keywords[0] if competitor_keywords else ""
            first_comp_info = competitor_info.get(first_keyword, {})
            
            for product in search_results:
                # 상품명에서 매칭되는 키워드 찾기
                product_name = product.get("product_name", "").lower()
                matched_kw = ""
                matched_info = {}
                
                for kw in competitor_keywords:
                    if kw.lower() in product_name or any(word in product_name for word in kw.lower().split()):
                        matched_kw = kw
                        matched_info = competitor_info.get(kw, {})
                        break
                
                # 매칭된 키워드가 없으면 첫 번째 키워드 정보 사용
                if not matched_kw:
                    matched_kw = first_keyword
                    matched_info = first_comp_info
                
                product["matched_keyword"] = matched_kw
                product["competitor_info"] = matched_info
                product["source_track"] = "competitor"
            
            print(f"=== [Track D] RAG 검색 결과: {len(search_results)}개 상품 ===")
            for i, p in enumerate(search_results[:3], 1):
                comp_info = p.get("competitor_info", {})
                print(f"  {i}. {p.get('product_name', '')[:25]} | 키워드: {p.get('matched_keyword', '')} | 경쟁사: {comp_info.get('company', 'N/A')} ({comp_info.get('start_time', '')[:16]})")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[Track D] 경쟁사 기반 상품 조회 실패: {e}")
            import traceback
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            return []
    
    def _extract_keywords_from_title(self, title: str, category: str = "") -> List[str]:
        """
        편성 제목에서 키워드 추출 (LLM 미사용, 코딩 방식)
        
        예시:
        - "겨울 패딩 특가전" → ["겨울 패딩", "패딩"]
        - "[특가] 로봇청소기 대전" → ["로봇청소기"]
        - "프리미엄 오메가3 12개월" → ["오메가3"]
        """
        import re
        
        if not title:
            return []
        
        # 불용어 (제거할 단어)
        stopwords = {
            # 프로모션 관련
            "특가", "특가전", "대전", "기획전", "세일", "할인", "프리미엄", "스페셜",
            "단독", "한정", "베스트", "인기", "추천", "신상", "신상품", "히트",
            # 수량/단위 관련
            "개월", "개월분", "박스", "세트", "팩", "통", "개", "매",
            # 방송 관련
            "방송", "홈쇼핑", "라이브", "생방송", "앵콜", "재방송",
            # 혜택 관련
            "무료배송", "사은품", "증정", "선물", "이벤트",
            # 시즌 코드 (의미없는 키워드)
            "24FW", "25FW", "24SS", "25SS", "23FW", "23SS", "22FW", "22SS",
            "FW", "SS", "AW", "봄", "여름", "가을", "겨울",
            # 기타 의미없는 키워드
            "신규", "런칭", "오픈", "리뉴얼", "업그레이드", "뉴", "NEW",
            "총", "전", "종", "구성", "더블", "트리플", "풀", "올"
        }
        
        # 1. 특수문자 및 괄호 내용 제거
        clean_title = re.sub(r'\[.*?\]|\(.*?\)|【.*?】', '', title)
        clean_title = re.sub(r'[^\w\s가-힣]', ' ', clean_title)
        
        # 2. 숫자+단위 패턴 제거 (12개월, 3박스 등)
        clean_title = re.sub(r'\d+\s*(개월분?|박스|세트|팩|통|개|매|g|kg|ml|L)', '', clean_title)
        
        # 3. 공백으로 분리
        words = clean_title.split()
        
        # 4. 불용어 제거 및 2글자 이상 필터링
        keywords = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word not in stopwords:
                keywords.append(word)
        
        # 5. 카테고리도 키워드로 추가
        if category and len(category) >= 2:
            keywords.append(category)
        
        # 6. 2단어 조합 키워드 추가 (예: "겨울 패딩")
        if len(keywords) >= 2:
            combined = f"{keywords[0]} {keywords[1]}"
            keywords.insert(0, combined)
        
        # 중복 제거
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:5]  # 최대 5개
    
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
            product_name = product.get("product_name", "")
            category = product.get("category_main", "Unknown")
            brand = product.get("brand", "Unknown")
            key = f"{category}_{brand}"
            
            # 1-1. 같은 카테고리+브랜드 조합은 1개만 허용 (다양성 보장)
            if key in category_brand_seen:
                print(f"  ⚠️ 브랜드 중복 제거: {product_name[:30]} (카테고리: {category}, 브랜드: {brand})")
                continue
            
            # 1-2. 같은 대분류 카테고리는 최대 4개까지만 허용
            current_count = category_count.get(category, 0)
            if current_count >= 4:
                print(f"  ⚠️ 카테고리 쿼터 초과: {product_name[:30]} (카테고리: {category}, 이미 {current_count}개)")
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
        
        # [5-1단계] 코딩 방식으로 추천 근거 생성 (LLM 미사용 - 속도/비용 최적화)
        step_5_1_start = time.time()
        print("\n" + "=" * 80)
        print(f"[5-1단계] 코딩 방식 - {len(ranked_products)}개 상품의 추천 근거 생성")
        print("=" * 80)
        
        reasoning_results = []
        for candidate in ranked_products:
            result = self._generate_reasoning_by_code(candidate, context or {})
            reasoning_results.append(result)
        
        reasoning_list = [r["reasoning"] for r in reasoning_results]
        print(f"⏱️  [5-1단계] 추천 근거 생성: {time.time() - step_5_1_start:.2f}초 (LLM 미사용)")
        
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
            
            # 추천 출처 정보는 내부 로그용으로만 사용 (API 응답에는 포함하지 않음)
            sources_for_log = candidate.get("recommendation_sources", [])
            
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
                # sources 필드 제거 - 추천 근거 생성에만 내부적으로 사용
            )

            # 추천 결과 요약 로그 (시연/분석용) - 출처 정보 포함
            try:
                # 출처 요약 생성 (딕셔너리 형태로 처리)
                source_summary = []
                for src in sources_for_log:
                    src_type = src.get("source_type", "")
                    if src_type == "rag_match":
                        origin = src.get("keyword_origin", "unknown")
                        source_summary.append(f"키워드({origin}): {src.get('matched_keyword', '')}")
                    elif src_type == "news_trend":
                        source_summary.append(f"뉴스: {src.get('news_keyword', '')}")
                    elif src_type == "ai_trend":
                        source_summary.append(f"AI트렌드: {src.get('ai_keyword', '')}")
                    elif src_type == "xgboost_sales":
                        pred_sales = src.get("predicted_sales", 0)
                        source_summary.append(f"매출예측: {int(pred_sales/10000):,}만원")
                
                print("=" * 100)
                print(
                    f"[최종 추천 #{recommendation.rank}] "
                    f"{recommendation.productInfo.productName[:40]}"
                )
                print(f"  [카테고리] {recommendation.productInfo.category}")
                print(f"  [예측매출] {recommendation.businessMetrics.aiPredictedSales}")
                print(f"  [점수] {candidate.get('final_score', 0.0):.3f}")
                print(f"  [출처] {' | '.join(source_summary)}")
                print(f"  [추천근거] {recommendation.reasoning[:80]}...")
            except Exception as e:
                print(f"[로그 오류] {e}")

            recommendations.append(recommendation)
        
        # [5-2단계] 네이버/타사 편성 조회 (LLM 없이 단순 조회)
        step_5_2_start = time.time()
        print("\n" + "=" * 80)
        print(f"[5-2단계] 네이버/타사 편성 조회")
        print("=" * 80)
        
        # 네이버 베스트 상품 조회 (상위 3개만)
        naver_products_data = self.external_products_service.get_latest_best_products(limit=3)
        naver_products = [NaverProduct(**product) for product in naver_products_data]
        logger.info(f"✅ 네이버 상품 상위 {len(naver_products)}개 수집")
        print(f"✅ 네이버 상품 상위 {len(naver_products)}개 수집")
        
        # 타 홈쇼핑사 편성 상품 조회 - Netezza에서 실시간 조회 (전체)
        competitor_products = []
        try:
            broadcast_time_str = context.get("broadcast_time") if context else None
            if broadcast_time_str:
                competitor_data = await netezza_conn.get_competitor_schedules(broadcast_time_str)
                competitor_products = [CompetitorProduct(**comp) for comp in competitor_data]
                logger.info(f"✅ 타사 편성 전체 {len(competitor_products)}개 수집")
                print(f"✅ 타사 편성 전체 {len(competitor_products)}개 수집")
            else:
                logger.warning(f"⚠️ broadcast_time이 context에 없음")
        except Exception as e:
            logger.warning(f"⚠️ 타사 편성 조회 실패: {str(e)}")
        
        # 네이버 상위 3개 + 타사 편성 전체 통합 (LLM 선택 없이)
        selected_competitor_products = []
        
        # 1. 타사 편성 전체 추가
        selected_competitor_products.extend(competitor_products)
        
        # 2. 네이버 상위 3개를 CompetitorProduct 형식으로 변환하여 추가
        for idx, naver in enumerate(naver_products[:3]):
            selected_competitor_products.append(self._convert_naver_to_competitor(naver, idx))
        
        print(f"⏱️  [5-2단계] 네이버/타사 조회 완료: {time.time() - step_5_2_start:.2f}초")
        print(f"  - 타사 편성: {len(competitor_products)}개")
        print(f"  - 네이버 상위: {len(naver_products)}개")
        
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
    
    # 출처 유형별 포맷 템플릿 (더 상세하게)
    SOURCE_TEMPLATES = {
        "news_trend": "[뉴스] '{keyword}' | {title} | URL: {url}",
        "ai_trend": "[AI트렌드] '{keyword}' - {reason}",
        "rag_match": "[키워드매칭] '{keyword}' ({origin})",
        "xgboost_sales": "[매출예측] {sales}만원 ({rank}위)",
        "context": "[컨텍스트] {factor}",
        "competitor": "[경쟁사] {name} {time} 편성 중"
    }
    
    # 키워드 출처 매핑
    KEYWORD_ORIGIN_MAP = {
        "news": "뉴스 트렌드",
        "ai": "AI 생성",
        "context": "컨텍스트",
        "unknown": "검색"
    }
    
    def _format_source_description(self, src: Dict[str, Any]) -> str:
        """출처 정보를 텍스트로 변환"""
        src_type = src.get("source_type", "")
        
        if src_type == "news_trend":
            keyword = src.get("news_keyword", "")
            title = src.get("news_title", "")[:50] if src.get("news_title") else "최근 기사"
            url = src.get("news_url", "") or "없음"
            return self.SOURCE_TEMPLATES["news_trend"].format(keyword=keyword, title=title, url=url) if keyword else ""
        
        elif src_type == "ai_trend":
            keyword = src.get("ai_keyword", "")
            reason = src.get("ai_reason", "시즌 트렌드")
            return self.SOURCE_TEMPLATES["ai_trend"].format(keyword=keyword, reason=reason) if keyword else ""
        
        elif src_type == "rag_match":
            keyword = src.get("matched_keyword", "")
            origin = self.KEYWORD_ORIGIN_MAP.get(src.get("keyword_origin", "unknown"), "검색")
            return self.SOURCE_TEMPLATES["rag_match"].format(keyword=keyword, origin=origin) if keyword else ""
        
        elif src_type == "xgboost_sales":
            sales = int(src.get("predicted_sales", 0) / 10000)
            rank = src.get("xgboost_rank", 0)
            return self.SOURCE_TEMPLATES["xgboost_sales"].format(sales=f"{sales:,}", rank=rank)
        
        elif src_type == "context":
            factor = src.get("context_factor", "")
            return self.SOURCE_TEMPLATES["context"].format(factor=factor) if factor else ""
        
        elif src_type == "competitor":
            name = src.get("competitor_name", "")
            time = src.get("competitor_time", "")
            return self.SOURCE_TEMPLATES["competitor"].format(name=name, time=time) if name else ""
        
        return ""
    
    def _format_product_info(self, rank: int, name: str, category: str, sales: int, sources: List[str]) -> str:
        """상품 정보를 프롬프트용 텍스트로 변환"""
        sources_str = " / ".join(sources) if sources else "출처 없음"
        return f"{rank}. {name[:50]} | {category} | {sales:,}만원\n   출처: {sources_str}"
    
    def _calculate_keyword_rankings(self, candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        """키워드 매칭 점수 기준 순위 계산"""
        # 키워드 매칭 점수 추출 (similarity_score 기준)
        scores = []
        for c in candidates:
            product_code = c.get("product", {}).get("product_code", "")
            similarity = c.get("similarity", 0)
            # recommendation_sources에서 rag_match의 similarity_score도 확인
            for src in c.get("recommendation_sources", []):
                if src.get("source_type") == "rag_match":
                    similarity = max(similarity, src.get("similarity_score", 0))
            scores.append((product_code, similarity))
        
        # 점수 내림차순 정렬 후 순위 부여
        scores.sort(key=lambda x: x[1], reverse=True)
        return {code: rank + 1 for rank, (code, _) in enumerate(scores)}
    
    def _calculate_sales_rankings(self, candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        """매출 예측 기준 순위 계산"""
        scores = []
        for c in candidates:
            product_code = c.get("product", {}).get("product_code", "")
            predicted_sales = c.get("predicted_sales", 0)
            scores.append((product_code, predicted_sales))
        
        # 매출 내림차순 정렬 후 순위 부여
        scores.sort(key=lambda x: x[1], reverse=True)
        return {code: rank + 1 for rank, (code, _) in enumerate(scores)}
    
    def _format_sources_with_rankings(self, sources: List[Dict], keyword_rank: int, sales_rank: int, total: int) -> List[str]:
        """출처 정보를 순위와 함께 포맷팅 (상위권만 순위 표시)"""
        result = []
        top_threshold = 10  # 상위 10위까지 순위 표시
        
        for src in sources:
            src_type = src.get("source_type", "")
            
            if src_type == "news_trend":
                keyword = src.get("news_keyword", "")
                title = src.get("news_title", "")[:50] if src.get("news_title") else "최근 기사"
                url = src.get("news_url", "") or "없음"
                if keyword:
                    result.append(f"[뉴스] '{keyword}' | {title} | URL: {url}")
            
            elif src_type == "ai_trend":
                keyword = src.get("ai_keyword", "")
                reason = src.get("ai_reason", "시즌 트렌드")
                if keyword:
                    # 상위권일 때만 순위 표시
                    if keyword_rank <= top_threshold:
                        result.append(f"[AI트렌드] '{keyword}' - {reason} (키워드 {keyword_rank}위)")
                    else:
                        result.append(f"[AI트렌드] '{keyword}' - {reason}")
            
            elif src_type == "rag_match":
                keyword = src.get("matched_keyword", "")
                origin = self.KEYWORD_ORIGIN_MAP.get(src.get("keyword_origin", "unknown"), "검색")
                if keyword:
                    if keyword_rank <= top_threshold:
                        result.append(f"[키워드매칭] '{keyword}' ({origin}) - 키워드 {keyword_rank}위")
                    else:
                        result.append(f"[키워드매칭] '{keyword}' ({origin})")
            
            elif src_type == "xgboost_sales":
                sales = int(src.get("predicted_sales", 0) / 10000)
                # 상위권일 때만 순위 표시
                if sales_rank <= top_threshold:
                    result.append(f"[매출예측] {sales:,}만원 (매출 {sales_rank}위)")
                else:
                    result.append(f"[매출예측] {sales:,}만원")
            
            elif src_type == "context":
                factor = src.get("context_factor", "")
                if factor:
                    result.append(f"[컨텍스트] {factor}")
            
            elif src_type == "competitor":
                name = src.get("competitor_name", "")
                time = src.get("competitor_time", "")
                if name:
                    result.append(f"[경쟁사] {name} {time} 편성 중")
            
            elif src_type == "sales_top":
                # Track B: 매출 예측 상위 (키워드 무관)
                if sales_rank <= top_threshold:
                    result.append(f"[매출상위] 키워드 무관 매출 예측 상위 (매출 {sales_rank}위)")
                else:
                    result.append(f"[매출상위] 키워드 무관 매출 예측 상위")
        
        return result
    
    def _validate_reasons_response(self, result: Any, expected_count: int) -> List[str]:
        """LLM 응답의 reasons 필드 검증"""
        # result가 None인 경우
        if result is None:
            logger.warning("[검증] LLM 응답이 None")
            return []
        
        # result가 dict가 아닌 경우
        if not isinstance(result, dict):
            logger.warning(f"[검증] LLM 응답이 dict가 아님: {type(result)}")
            return []
        
        # reasons 필드가 없는 경우
        reasons = result.get("reasons")
        if reasons is None:
            logger.warning("[검증] reasons 필드 없음")
            return []
        
        # reasons가 list가 아닌 경우
        if not isinstance(reasons, list):
            logger.warning(f"[검증] reasons가 list가 아님: {type(reasons)}")
            return []
        
        # 각 항목이 문자열인지 검증
        validated = []
        for i, reason in enumerate(reasons):
            if isinstance(reason, str) and reason.strip():
                validated.append(reason.strip())
            else:
                logger.warning(f"[검증] reasons[{i}]가 유효하지 않음: {type(reason)}")
        
        # 개수 검증
        if len(validated) != expected_count:
            logger.info(f"[검증] 개수 불일치: 기대 {expected_count}, 실제 {len(validated)}")
        
        return validated
    
    async def _generate_batch_reasons_with_langchain(self, candidates: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[str]:
        """배치로 여러 상품의 추천 근거를 한 번에 생성"""
        try:
            time_slot = context.get("time_slot", "") if context else ""
            weather = context.get("weather", {}).get("weather", "") if context else ""
            holiday_name = context.get("holiday_name") if context else None
            
            # 키워드 순위와 매출 순위를 따로 계산
            keyword_rankings = self._calculate_keyword_rankings(candidates)
            sales_rankings = self._calculate_sales_rankings(candidates)
            
            # 상품별 출처 정보 포맷팅 (순위 정보 포함)
            products_with_sources = []
            for candidate in candidates:
                product = candidate["product"]
                product_code = product.get("product_code", "")
                
                # 순위 정보 가져오기
                keyword_rank = keyword_rankings.get(product_code, 0)
                sales_rank = sales_rankings.get(product_code, 0)
                
                # 출처 정보 포맷팅 (순위 정보 포함)
                sources = self._format_sources_with_rankings(
                    candidate.get("recommendation_sources", []),
                    keyword_rank, sales_rank, len(candidates)
                )
                
                products_with_sources.append({
                    "rank": candidate.get("rank", 0),
                    "product_name": product.get("product_name", ""),
                    "category": product.get("category_main", ""),
                    "predicted_sales": int(candidate.get("predicted_sales", 0) / 10000),
                    "keyword_rank": keyword_rank,
                    "sales_rank": sales_rank,
                    "sources": sources
                })
            
            # 프롬프트 - 자연스럽고 구체적인 추천 근거 (순위 정보 포함)
            batch_prompt = ChatPromptTemplate.from_messages([
                ("system", """홈쇼핑 방송 편성 전문가로서 자연스러운 추천 근거를 작성하세요.

**오늘 날짜: {today_date}**

출처 유형별 작성 방법:

1. **[뉴스] 출처:**
   "최근 뉴스에 따르면 '프리미엄 여행상품' 관련 기사가 보도되었습니다. 이에 해당 상품의 편성을 추천합니다. (출처: URL)"

2. **[AI트렌드] 출처:**
   "{today_date} 트렌드 키워드 분석 결과 '겨울 의류' 키워드 1위로 적합한 상품입니다."

3. **[매출예측] 출처:**
   "AI 매출 예측 결과 1,135만원으로 매출 1위를 기록했습니다."

4. **[경쟁사] 출처:**
   "{today_date} {time_slot} 경쟁사 롯데홈쇼핑에서 유사 상품 판매 중으로, 해당 시간대 편성을 추천합니다."

5. **[매출상위] 출처:**
   "트렌드 키워드와 무관하게 AI 매출 예측 상위 상품으로, 안정적인 매출이 기대됩니다."

규칙:
- 각 상품 100-150자
- 출처에 순위가 표시된 경우에만 순위 언급 (상위 10위까지만 순위 표시됨)
- "[뉴스]", "[AI]" 태그를 자연어로 변환
- 뉴스 URL이 있으면 "(출처: URL)" 형태로 끝에 추가

JSON: {{"reasons": ["근거1", "근거2", ...]}}"""),
                ("human", """시간대: {time_slot} | 날씨: {weather}

{products_info}

{count}개 상품의 추천 근거를 자연스럽게 작성하세요.""")
            ])
            
            # 상품 정보 포맷팅
            products_info = "\n".join([
                self._format_product_info(p["rank"], p["product_name"], p["category"], p["predicted_sales"], p["sources"])
                for p in products_with_sources
            ])
            
            # 디버그: 실제 전달되는 상품 정보 출력
            print(f"\n[DEBUG] LLM에 전달되는 상품 정보:\n{products_info[:500]}...")
            
            chain = batch_prompt | self.llm | JsonOutputParser()
            
            # 오늘 날짜 생성
            from datetime import datetime
            today_date = datetime.now().strftime("%m월 %d일")
            
            result = await chain.ainvoke({
                "time_slot": time_slot or "미지정",
                "weather": weather or "보통",
                "today_date": today_date,
                "products_info": products_info,
                "count": len(candidates)
            })
            
            # JSON 파싱 검증
            reasons = self._validate_reasons_response(result, len(candidates))
            print(f"[배치 처리] {len(reasons)}개 근거 생성 완료")
            
            # 개수가 부족하면 출처 기반 기본 메시지로 채움
            while len(reasons) < len(candidates):
                idx = len(reasons)
                candidate = candidates[idx]
                sources = candidate.get("recommendation_sources", [])
                
                # 출처 기반 폴백 메시지 생성
                fallback_reason = self._generate_fallback_reason(candidate, sources)
                reasons.append(fallback_reason)
            
            return reasons[:len(candidates)]
            
        except Exception as e:
            logger.error(f"배치 근거 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            # 폴백: 출처 기반 기본 메시지
            print("⚠️ 배치 처리 실패, 출처 기반 폴백...")
            return [self._generate_fallback_reason(c, c.get("recommendation_sources", [])) for c in candidates]
    
    def _generate_reasoning_by_code(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        코딩 방식으로 추천 근거 생성 (LLM 미사용)
        
        Returns:
            {
                "reasoning": "추천 근거 텍스트",
                "scores": {
                    "total": 0.85,
                    "keyword_score": 0.70,
                    "sales_score": 0.15,
                    "historical_score": 0.00
                },
                "keyword_source": {
                    "type": "news" | "ai" | "context" | "historical",
                    "keyword": "키워드",
                    "news_url": "URL (뉴스인 경우)"
                }
            }
        """
        product = candidate.get("product", {})
        sources = candidate.get("recommendation_sources", [])
        predicted_sales = candidate.get("predicted_sales", 0)
        similarity_score = candidate.get("similarity_score", 0)
        final_score = candidate.get("final_score", 0)
        candidate_source = candidate.get("source", "")  # Track 출처 (competitor, sales_top, historical, news_trend, ai_trend)
        
        # 출처 정보 파싱
        news_info = None
        ai_info = None
        rag_info = None
        historical_info = None
        sales_top_info = None
        competitor_info = None
        context_info = None
        
        # candidate.source 기반으로 출처 정보 설정
        if candidate_source == "sales_top":
            sales_top_info = {"reason": "키워드 무관 매출 예측 상위 상품"}
        elif candidate_source == "historical":
            historical_info = {
                "avg_profit": product.get("historical_avg_profit", 0),
                "broadcast_count": product.get("historical_broadcast_count", 0),
                "reason": "과거 유사 시간대 실적 상위"
            }
        elif candidate_source == "competitor":
            comp_info = product.get("competitor_info", {})
            competitor_info = {
                "company": comp_info.get("company", "경쟁사"),
                "title": comp_info.get("title", ""),
                "keyword": product.get("matched_keyword", "")
            }
        
        for src in sources:
            src_type = src.get("source_type", "")
            if src_type == "news_trend" and not news_info:
                news_info = {
                    "keyword": src.get("news_keyword", ""),
                    "title": src.get("news_title", ""),
                    "url": src.get("news_url", "")
                }
            elif src_type == "ai_trend" and not ai_info:
                ai_info = {"keyword": src.get("ai_keyword", ""), "reason": src.get("ai_reason", "")}
            elif src_type == "rag_match" and not rag_info:
                rag_info = {
                    "keyword": src.get("matched_keyword", ""),
                    "similarity": src.get("similarity_score", 0),
                    "origin": src.get("keyword_origin", "unknown"),
                    "origin_detail": src.get("keyword_origin_detail", "")
                }
            elif src_type == "historical" and not historical_info:
                historical_info = {
                    "avg_profit": src.get("historical_avg_profit", 0),
                    "broadcast_count": src.get("historical_broadcast_count", 0),
                    "reason": src.get("reason", "")
                }
            elif src_type == "sales_top" and not sales_top_info:
                sales_top_info = {"reason": src.get("reason", "")}
            elif src_type == "competitor" and not competitor_info:
                competitor_info = {
                    "company": src.get("competitor_company", ""),
                    "title": src.get("competitor_title", ""),
                    "keyword": src.get("competitor_keyword", "")
                }
            elif src_type == "context" and not context_info:
                context_info = {"factor": src.get("context_factor", "")}
        
        # 점수 계산 (세분화)
        keyword_score = similarity_score * 0.7 if similarity_score >= 0.7 else similarity_score * 0.3
        sales_score = (predicted_sales / 100000000) * (0.3 if similarity_score >= 0.7 else 0.7)
        historical_score = 0.0
        if historical_info and historical_info["avg_profit"] > 0:
            historical_score = min(historical_info["avg_profit"] / 50000000, 0.2)  # 최대 0.2
        
        scores = {
            "total": round(final_score, 3),
            "keyword_score": round(keyword_score, 3),
            "sales_score": round(sales_score, 3),
            "historical_score": round(historical_score, 3)
        }
        
        # 키워드 출처 정보
        keyword_source = {"type": "unknown", "keyword": "", "news_url": None}
        
        if rag_info and rag_info["keyword"]:
            keyword_source["keyword"] = rag_info["keyword"]
            if rag_info["origin"] == "news" or news_info:
                keyword_source["type"] = "news"
                if news_info and news_info.get("url"):
                    keyword_source["news_url"] = news_info["url"]
            elif rag_info["origin"] == "ai" or ai_info:
                keyword_source["type"] = "ai"
            elif rag_info["origin"] == "context":
                keyword_source["type"] = "context"
            else:
                keyword_source["type"] = "ai"  # 기본값
        
        if historical_info:
            keyword_source["type"] = "historical"
        
        # 추천 근거 텍스트 생성
        parts = []
        
        # 0. 여러 출처 표시 (source_labels 활용)
        source_labels = product.get("source_labels", [])
        if source_labels:
            source_tag = "|".join(source_labels)
            parts.append(f"[{source_tag}]")
        
        # 1. 키워드 출처 상세 (뉴스 URL 등) - 복합 출처면 모두 표시
        # 단, 경쟁사 상품은 경쟁사 정보에서 키워드가 표시되므로 AI 트렌드 생략
        product_comp_info_check = product.get("competitor_info", {})
        is_competitor_only = product_comp_info_check and product_comp_info_check.get("company")
        
        if news_info and news_info["keyword"]:
            keyword_part = f"[뉴스] '{news_info['keyword']}' 트렌드"
            if news_info.get("url"):
                keyword_part += f" (출처: {news_info['url'][:50]}...)"
            parts.append(keyword_part)
        elif ai_info and ai_info["keyword"] and not is_competitor_only:
            # 경쟁사 상품이 아닌 경우에만 AI 트렌드 표시
            parts.append(f"[AI] '{ai_info['keyword']}' 시즌 트렌드")
        elif rag_info and rag_info["keyword"]:
            parts.append(f"[RAG] '{rag_info['keyword']}' 매칭 (유사도: {rag_info['similarity']:.0%})")
        
        # 2. Track C (과거 실적) 상세 - 최근 방송일자/매출 포함
        if historical_info and historical_info["avg_profit"] > 0:
            # 최근 방송 정보 (1건)
            last_broadcast_time = product.get("last_broadcast_time", "")
            last_profit = product.get("last_profit", 0)
            
            if last_broadcast_time and last_profit > 0:
                # 최근 방송일자와 매출 표시
                last_date = str(last_broadcast_time)[:10]  # YYYY-MM-DD
                last_time = str(last_broadcast_time)[11:16] if len(str(last_broadcast_time)) > 11 else ""  # HH:MM
                last_profit_str = f"{int(last_profit/10000):,}만원"
                hist_detail = f"[과거실적] 최근 {last_date} {last_time} 방송 매출 {last_profit_str}"
                # 평균 정보도 추가
                avg_profit_str = f"{int(historical_info['avg_profit']/10000):,}만원"
                hist_detail += f" (평균 {avg_profit_str}, {historical_info['broadcast_count']}회)"
            else:
                # 최근 정보가 없으면 평균만 표시
                avg_profit_str = f"{int(historical_info['avg_profit']/10000):,}만원"
                hist_detail = f"[과거실적] 유사 시간대 평균 {avg_profit_str} ({historical_info['broadcast_count']}회 방송)"
            
            parts.append(hist_detail)
        
        # 3. Track D (경쟁사 대응) 상세 - 방송사, 편성제목, 시간 포함
        product_comp_info = product.get("competitor_info", {})
        if product_comp_info and product_comp_info.get("company"):
            comp_company = product_comp_info.get("company", "")
            comp_title = product_comp_info.get("title", "")
            comp_time = product_comp_info.get("start_time", "")
            comp_keyword = product_comp_info.get("keyword", "")  # 경쟁사 편성에서 추출한 키워드
            
            # 상세 경쟁사 정보: "GS홈쇼핑 '로봇청소기 특가' (14:00) 키워드:'로봇청소기'"
            comp_part = f"[경쟁사대응] {comp_company}"
            if comp_title:
                comp_part += f" '{comp_title[:30]}'"
            if comp_time:
                time_str = comp_time[11:16] if len(comp_time) > 11 else comp_time
                comp_part += f" ({time_str})"
            if comp_keyword:
                comp_part += f" 키워드:'{comp_keyword}'"
            parts.append(comp_part)
        elif competitor_info and competitor_info.get("company"):
            comp_part = f"[경쟁사대응] {competitor_info['company']}"
            if competitor_info.get("keyword"):
                comp_part += f" '{competitor_info['keyword']}' 편성"
            parts.append(comp_part)
        
        # 4. 매출 예측
        sales_str = f"{int(predicted_sales/10000):,}만원"
        parts.append(f"[예측매출] {sales_str}")
        
        # 6. 최종 점수
        parts.append(f"[점수] 총점 {final_score:.3f} (키워드 {keyword_score:.3f} + 매출 {sales_score:.3f})")
        
        reasoning = " | ".join(parts)
        
        return {
            "reasoning": reasoning,
            "scores": scores,
            "keyword_source": keyword_source
        }
    
    def _generate_fallback_reason(self, candidate: Dict[str, Any], sources: List[Dict]) -> str:
        """출처 기반 폴백 추천 근거 생성 - 여러 출처 조합 (하위 호환용)"""
        result = self._generate_reasoning_by_code(candidate, {})
        return result["reasoning"]
    
    def _prepare_features_for_product(self, product: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """1개 상품의 XGBoost feature 준비 (예측은 안 함)
        
        2024-12-15 수정: 시간대/월 피처 강화, 날씨/가격 피처 제거
        - 시간: 9시에 팔린 상품 → 8~10시에 추천
        - 월: 11월에 팔린 상품 → 10~12월에 추천
        """
        broadcast_dt = context["broadcast_dt"]
        
        print(f"=== [_prepare_features_for_product] 호출됨: {product.get('product_name', 'Unknown')[:30]} ===")
        
        category_main = product.get("category_main", product.get("category", "Unknown"))
        time_slot = context["time_slot"]
        
        # 시간 피처: 사인/코사인 변환 (주기성 반영)
        hour = broadcast_dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # 월 피처: 사인/코사인 변환 (주기성 반영)
        month = broadcast_dt.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        return {
            # Numeric features - 시간대/월 강화 (날씨/가격 제거)
            "hour": hour,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month": month,
            "month_sin": month_sin,
            "month_cos": month_cos,
            
            # Categorical features (날씨/계절 제거)
            "product_lgroup": category_main,
            "product_mgroup": product.get("category_middle", "Unknown"),
            "product_sgroup": product.get("category_sub", "Unknown"),
            "brand": product.get("brand", "Unknown"),
            "product_type": product.get("product_type", "유형"),
            "time_slot": time_slot,  # 핵심 피처
            "day_of_week": ["월", "화", "수", "목", "금", "토", "일"][broadcast_dt.weekday()],  # 핵심 피처
            # "season" 제거: month 피처로 대체
            
            # Boolean features - 핵심 피처
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
        """여러 상품 XGBoost 매출 예측 (배치 처리) + 신상품 보정"""
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
                print(f"    - hour: {features['hour']}, month: {features['month']}")
                print(f"    - time_slot: {features['time_slot']}, day_of_week: {features['day_of_week']}")
                print(f"    - 카테고리: {features['product_lgroup']}")
            
            # XGBoost 배치 예측 (한 번에 처리)
            predicted_sales_log = self.model.predict(batch_df)
            # 로그 역변환 (학습 시 log1p 사용)
            predicted_sales_array = np.expm1(predicted_sales_log)
            
            # ========== 신상품 매출 보정 (2-A) ==========
            # 카테고리별 평균 매출 조회
            category_avg_sales = self.product_embedder.get_category_avg_sales()
            
            # 전체 평균 계산 (카테고리 평균이 없는 경우 대비)
            overall_avg = sum(category_avg_sales.values()) / len(category_avg_sales) if category_avg_sales else 30000000
            
            # 신상품 보정: 예측값이 너무 낮으면 카테고리 평균으로 대체
            MIN_SALES_THRESHOLD = 5000000  # 500만원 미만이면 신상품으로 간주
            corrected_count = 0
            
            for i, (product, sales) in enumerate(zip(products, predicted_sales_array)):
                if sales < MIN_SALES_THRESHOLD:
                    category = product.get("category_main", "")
                    category_avg = category_avg_sales.get(category, overall_avg)
                    # 카테고리 평균의 80%로 보정 (보수적 추정)
                    corrected_sales = category_avg * 0.8
                    predicted_sales_array[i] = corrected_sales
                    corrected_count += 1
                    print(f"  [신상품 보정] {product.get('product_name', '')[:25]} | {sales/10000:.0f}만원 → {corrected_sales/10000:.0f}만원 (카테고리 평균)")
            
            if corrected_count > 0:
                print(f"=== [신상품 보정] {corrected_count}개 상품 카테고리 평균으로 보정됨 ===")
            
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
