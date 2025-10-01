"""
방송 편성 AI 추천 워크플로우
LangChain 기반 2단계 워크플로우: AI 방향 탐색 + 고속 랭킹
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .external_apis import ExternalAPIManager

from .dependencies import get_product_embedder
from . import broadcast_recommender as br
from .schemas import BroadcastResponse, RecommendedCategory, BroadcastRecommendation, ProductInfo, Reasoning, BusinessMetrics

logger = logging.getLogger(__name__)

class BroadcastWorkflow:
    """방송 편성 AI 추천 워크플로우"""
    
    def __init__(self, model):
        self.model = model  # XGBoost 모델
        self.product_embedder = get_product_embedder()
        
        # LangChain LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # DB 연결
        self.engine = create_engine(os.getenv("POSTGRES_URI", os.getenv("DB_URI")))
    
    async def process_broadcast_recommendation(
        self, 
        broadcast_time: str, 
        recommendation_count: int = 5
    ) -> BroadcastResponse:
        """메인 워크플로우: 방송 시간 기반 추천"""
        
        print("=== [DEBUG] process_broadcast_recommendation 시작 ===")
        request_time = datetime.now().isoformat()
        logger.info(f"방송 추천 워크플로우 시작: {broadcast_time}")
        print(f"=== [DEBUG] broadcast_time: {broadcast_time}, recommendation_count: {recommendation_count} ===")
        
        try:
            # 1단계: AI의 방향 탐색 (숲 찾기)
            print("=== [DEBUG] _collect_context 호출 전 ===")
            context = await self._collect_context(broadcast_time)
            print(f"=== [DEBUG] _collect_context 완료, context keys: {context.keys()} ===")
            
            print("=== [DEBUG] _classify_keywords_with_langchain 호출 전 ===")
            classified_keywords = await self._classify_keywords_with_langchain(context)
            print(f"=== [DEBUG] _classify_keywords_with_langchain 완료, keys: {classified_keywords.keys()} ===")
            
            # 2. Track A, B 비동기 병렬 실행 (문서 명세 준수)
            print(f"=== [DEBUG] Track A/B 실행 전, category_keywords: {classified_keywords.get('category_keywords', [])}, product_keywords: {classified_keywords.get('product_keywords', [])} ===")
            track_a_result, track_b_result = await asyncio.gather(
                self._execute_track_a(context, classified_keywords.get("category_keywords", [])),
                self._execute_track_b(context, classified_keywords.get("product_keywords", []))
            )
            print(f"=== [DEBUG] Track A/B 완료, categories: {len(track_a_result.get('categories', []))}, products: {len(track_b_result.get('products', []))} ===")
            
            # 생성된 키워드를 context에 저장 (추천 근거에 사용)
            context["category_keywords"] = classified_keywords.get("category_keywords", [])
            context["product_keywords"] = classified_keywords.get("product_keywords", [])
            context["generated_keywords"] = track_b_result.get("generated_keywords", [])  # Track B에서 생성된 키워드
            print(f"=== [DEBUG] context에 키워드 저장 완료, generated_keywords: {context['generated_keywords']} ===")
            
            # 3. 후보군 생성 및 통합 (문서 명세 준수)
            candidate_products = await self._generate_candidates(
                promising_categories=track_a_result["categories"],
                trend_products=track_b_result["products"]
            )
            
            # 4. 최종 랭킹 계산 (문서 명세 준수)
            ranked_products = await self._rank_final_candidates(
                candidate_products,
                category_scores=track_a_result["scores"],
                context=context
            )
            
            # 5. API 응답 생성 (문서 명세 준수)
            response = await self._format_response(ranked_products[:recommendation_count], track_a_result["categories"][:3], context)
            response.requestTime = request_time
            
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
    
    async def _collect_context(self, broadcast_time: str) -> Dict[str, Any]:
        """컨텍스트 수집: 날씨, 트렌드, 시간 정보"""
        
        # 방송 시간 파싱
        broadcast_dt = datetime.fromisoformat(broadcast_time.replace('Z', '+00:00'))
        
        context = {
            "broadcast_time": broadcast_time,
            "broadcast_dt": broadcast_dt,
            "hour": broadcast_dt.hour,
            "weekday": broadcast_dt.weekday(),
            "season": self._get_season(broadcast_dt.month)
        }
        
        # 날씨 정보 수집
        weather_info = br.get_weather_by_date(broadcast_dt.date())
        context["weather"] = weather_info

        # 시간대 정보
        time_slot = self._get_time_slot(broadcast_dt)
        day_type = "주말" if broadcast_dt.weekday() >= 5 else "평일"
        context["time_slot"] = time_slot
        context["day_type"] = day_type

        # AI 기반 트렌드 생성 (LLM API)
        api_manager = ExternalAPIManager()
        if api_manager.llm_trend_api:
            try:
                llm_trends = await api_manager.llm_trend_api.get_trending_searches()
                # AI가 생성한 트렌드 키워드 추가
                context["ai_trends"] = [t["keyword"] for t in llm_trends]
                logger.info(f"AI 트렌드 생성 완료: {len(llm_trends)}개 키워드")
                logger.info(f"AI 트렌드: {context['ai_trends'][:5]}...")  # 상위 5개만 로그
            except Exception as e:
                logger.error(f"AI 트렌드 생성 실패: {e}")
                context["ai_trends"] = []
        else:
            logger.warning("OpenAI API 키 없음 - AI 트렌드 생성 건너뜀")
            context["ai_trends"] = []

        # 컨텍스트 로그 출력
        logger.info(f"컨텍스트 수집 완료 - 계절: {context['season']}, 시간대: {time_slot}, 요일: {day_type}")
        logger.info(f"날씨: {weather_info.get('weather', 'N/A')}")

        return context
    
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

    async def _classify_keywords_with_langchain(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """LangChain을 사용한 키워드 분류"""
        
        # 모든 키워드 수집
        all_keywords = []
        
        # 날씨 키워드
        if context["weather"].get("weather"):
            all_keywords.append(context["weather"]["weather"])
        
        # 시간/날짜 키워드
        all_keywords.extend([context["time_slot"], context["day_type"], context["season"]])
        
        # 🎉 AI 생성 트렌드 추가! (날씨/시간 기반 트렌드)
        if "ai_trends" in context and context["ai_trends"]:
            all_keywords.extend(context["ai_trends"][:10])  # 상위 10개만 포함
            logger.info(f"AI 트렌드 키워드 {len(context['ai_trends'][:10])}개 추가됨")

        # 수집된 키워드들 로그 출력
        logger.info(f"키워드 분류 시작 - 총 {len(all_keywords)}개 키워드: {all_keywords}")

        # LangChain 프롬프트
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 홈쇼핑 방송 편성 전문가입니다. 
주어진 키워드들을 다음 두 그룹으로 분류해주세요:

1. category_keywords: 상품 카테고리와 연관된 키워드 (예: 흐린날씨, 캠핑, 건강식품, 겨울)
2. product_keywords: 특정 상품을 지칭하는 키워드 (예: 아이폰, 라부부, 정관장)

JSON 형식으로 응답해주세요."""),
            ("human", "키워드 목록: {keywords}")
        ])
         
        chain = classification_prompt | self.llm | JsonOutputParser()
        
        try:
            result = await chain.ainvoke({"keywords": ", ".join(all_keywords)})
            logger.info(f"키워드 분류 완료: 카테고리 {len(result.get('category_keywords', []))}개, 상품 {len(result.get('product_keywords', []))}개")
            logger.info(f"분류된 카테고리 키워드: {result.get('category_keywords', [])}")
            logger.info(f"분류된 상품 키워드: {result.get('product_keywords', [])}")
            return result
        except Exception as e:
            logger.error(f"키워드 분류 오류: {e}")
            # OpenAI API 할당량 소진 또는 API 오류 시 예외 발생
            if "insufficient_quota" in str(e) or "429" in str(e):
                raise Exception(f"AI 서비스 일시 중단 - OpenAI API 할당량 소진: {e}")
            elif "api" in str(e).lower() or "openai" in str(e).lower():
                raise Exception(f"AI 서비스 연결 오류: {e}")
            else:
                # 기타 오류는 폴백 로직 사용
                return {
                    "category_keywords": all_keywords[:10],
                    "product_keywords": []
                }
    
    async def _execute_track_a(self, context: Dict[str, Any], category_keywords: List[str]) -> Dict[str, Any]:
        """Track A: 유망 카테고리 찾기"""
        
        print(f"=== [DEBUG Track A] 시작, category_keywords: {category_keywords} ===")
        if not category_keywords:
            print("=== [DEBUG Track A] category_keywords가 비어있음 ===")
            return {"categories": [], "scores": {}}
        
        # RAG 검색으로 관련 카테고리 찾기
        query = " ".join(category_keywords)
        print(f"=== [DEBUG Track A] Qdrant 검색 쿼리: '{query}' ===")
        
        try:
            # Qdrant에서 카테고리 검색 (방송 테이프 준비 완료 상품만)
            similar_products = self.product_embedder.search_products(
                trend_keywords=[query],
                top_k=50,
                score_threshold=0.3,
                only_ready_products=True
            )
            print(f"=== [DEBUG Track A] Qdrant 검색 결과: {len(similar_products)}개 상품 ===")
            if len(similar_products) > 0:
                print(f"=== [DEBUG Track A] 첫 번째 상품 예시: {similar_products[0]} ===")
            
            # 카테고리별 그룹핑
            category_scores = {}
            for product in similar_products:
                category = product.get('category_main', 'Unknown')
                score = product.get('similarity_score', 0)
                
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
            
            print(f"=== [DEBUG Track A] 카테고리 그룹핑 완료, 총 {len(category_scores)} 카테고리: {list(category_scores.keys())} ===")
            
            # Qdrant 결과가 없으면 전체 카테고리 조회
            if len(category_scores) == 0:
                print("=== [DEBUG Track A] Qdrant 결과 없음, PostgreSQL에서 전체 카테고리 조회 ===")
                all_categories = await self._get_all_categories_from_db()
                print(f"=== [DEBUG Track A] 전체 카테고리 {len(all_categories)}개 발견: {all_categories} ===")
                # 기본 점수 부여
                for category in all_categories:
                    category_scores[category] = [0.5]  # 기본 유사도 점수
            
            # 카테고리별 평균 점수 계산 및 XGBoost 예측
            promising_categories = []
            broadcast_dt = datetime.fromisoformat(context["broadcast_time"].replace('Z', '+00:00'))
            
            print(f"=== [DEBUG Track A] XGBoost 매출 예측 시작 ===")
            for category, scores in category_scores.items():
                print(f"=== [DEBUG Track A] 카테고리 '{category}' XGBoost 예측 중... ===")
                avg_score = sum(scores) / len(scores)
                
                # XGBoost로 해당 카테고리의 예상 매출 예측
                predicted_sales = await self._predict_category_sales(category, broadcast_dt)
                
                # 최종 점수 = RAG 점수 * 예상 매출
                final_score = avg_score * (predicted_sales / 1000000)  # 백만원 단위로 정규화
                
                promising_categories.append({
                    "category": category,
                    "rag_score": avg_score,
                    "predicted_sales": predicted_sales,
                    "final_score": final_score,
                    "reason": "AI 추천 유망 카테고리"
                })
            
            print(f"=== [DEBUG Track A] XGBoost 예측 완료, 총 {len(promising_categories)} 카테고리 ===")
            
            # 점수순 정렬
            promising_categories.sort(key=lambda x: x["final_score"], reverse=True)
            
            # RecommendedCategory 객체로 변환
            result = []
            for i, cat in enumerate(promising_categories[:5]):
                result.append(RecommendedCategory(
                    rank=i+1,
                    name=cat["category"],
                    reason=cat["reason"],
                    predictedSales=f"{cat['predicted_sales']/100000000:.1f}억"
                ))
            
            print(f"=== [DEBUG Track A] 최종 결과: {len(result)} 카테고리 ===")
            logger.info(f"Track A: 유망 카테고리 {len(result)}개 발견")
            return {"categories": result, "scores": category_scores}
            
        except Exception as e:
            logger.error(f"Track A 오류: {e}")
            return {"categories": [], "scores": {}}
    
    async def _generate_context_keywords(self, context: Dict[str, Any]) -> List[str]:
        """컨텍스트 정보를 기반으로 LangChain으로 검색 키워드 생성"""
        
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
        season = context.get("season", "봄")
        day_type = context.get("day_type", "평일")
        
        logger.info(f"추출된 정보 - weather: {weather}, temp: {temperature}, time_slot: {time_slot}, season: {season}, day_type: {day_type}")
        
        # LangChain 프롬프트
        keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 홈쇼핑 방송 편성 전문가입니다. 
주어진 컨텍스트 정보를 분석하여, 해당 시간/날씨/상황에 적합한 상품 검색 키워드를 생성해주세요.

예시:
- 날씨가 '비'이고 저녁 시간 → "우산", "방수", "실내활동", "따뜻한음식", "집콕", "요리도구"
- 날씨가 '맑음'이고 오후 시간 → "야외활동", "운동", "캠핑", "레저", "자외선차단"
- 겨울철 저녁 시간 → "난방", "보온", "따뜻한", "겨울의류", "온열", "찜질"

5-10개의 키워드를 JSON 배열로 반환해주세요."""),
            ("human", """날씨: {weather}
기온: {temperature}도
시간대: {time_slot}
계절: {season}
요일 타입: {day_type}

위 상황에 적합한 상품 검색 키워드를 생성해주세요.""")
        ])
        
        chain = keyword_prompt | self.llm | JsonOutputParser()
        
        try:
            result = await chain.ainvoke({
                "weather": weather,
                "temperature": temperature,
                "time_slot": time_slot,
                "season": season,
                "day_type": day_type
            })
            keywords = result.get("keywords", [])
            logger.info(f"컨텍스트 기반 키워드 생성 완료: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"컨텍스트 키워드 생성 오류: {e}")
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
            
            logger.info(f"폴백 키워드 사용: {fallback_keywords}")
            return fallback_keywords
    
    async def _execute_track_b(self, context: Dict[str, Any], product_keywords: List[str]) -> Dict[str, Any]:
        """Track B: 컨텍스트 기반 상품 찾기 (날씨/시간대 기반)"""
        
        print(f"=== [DEBUG Track B] 시작, product_keywords: {product_keywords} ===")
        
        generated_keywords = []  # 생성된 키워드 저장
        
        # 1. 트렌드 키워드가 없으면 컨텍스트에서 키워드 생성
        if not product_keywords:
            logger.info("Track B: 트렌드 키워드 없음 → 컨텍스트 기반 키워드 생성")
            product_keywords = await self._generate_context_keywords(context)
            generated_keywords = product_keywords  # 생성된 키워드 보관
            print(f"=== [DEBUG Track B] 생성된 컨텍스트 키워드: {product_keywords} ===")
        
        if not product_keywords:
            logger.info("Track B: 키워드 없음 → 빈 결과 반환")
            return {"products": [], "trend_scores": {}, "generated_keywords": []}
        
        # 2. 생성된 키워드로 상품 검색
        query = " ".join(product_keywords)
        print(f"=== [DEBUG Track B] Qdrant 검색 쿼리: '{query}' ===")
        
        try:
            # Qdrant에서 상품 검색 (방송 테이프 준비 완료 상품만)
            similar_products = self.product_embedder.search_products(
                trend_keywords=[query],
                top_k=20,
                score_threshold=0.3
            )
            
            print(f"=== [DEBUG Track B] Qdrant 검색 결과: {len(similar_products)}개 ===")
            
            if similar_products:
                trend_scores = {p["product_code"]: p.get("score", 0.5) for p in similar_products}
                logger.info(f"Track B 완료: {len(similar_products)}개 상품 발견")
                return {"products": similar_products, "trend_scores": trend_scores, "generated_keywords": generated_keywords}
            else:
                logger.info("Track B: 검색 결과 없음")
                return {"products": [], "trend_scores": {}, "generated_keywords": generated_keywords}
                
        except Exception as e:
            logger.error(f"Track B 오류: {e}")
            return {"products": [], "trend_scores": {}, "generated_keywords": []}
    
    async def _generate_candidates(self, promising_categories: List[RecommendedCategory], trend_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """후보군 생성 및 통합"""
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
    
    async def _rank_final_candidates(self, candidates: List[Dict[str, Any]], category_scores: Dict[str, List[float]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """최종 랭킹 계산 (가중치 기반 점수 공식)"""
        
        for candidate in candidates:
            product = candidate["product"]
            
            # 기본 점수 계산
            base_score = candidate["base_score"]
            trend_boost = candidate["trend_boost"]
            
            # XGBoost 매출 예측
            predicted_sales = await self._predict_product_sales(product, context)
            sales_score = min(predicted_sales / 100000000, 1.0)  # 1억 기준 정규화
            
            # 경쟁 페널티 계산
            competition_penalty = self._calculate_competition_penalty(product, candidates)
            
            # 최종 점수 = (기본점수 × 트렌드부스트 + 매출점수) × (1 - 경쟁페널티)
            final_score = (base_score * trend_boost + sales_score) * (1 - competition_penalty)
            
            candidate["final_score"] = final_score
            candidate["predicted_sales"] = predicted_sales
            candidate["competition_penalty"] = competition_penalty
        
        # 점수순 정렬
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
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
    
    async def _format_response(self, ranked_products: List[Dict[str, Any]], top_categories: List[RecommendedCategory], context: Dict[str, Any] = None) -> BroadcastResponse:
        """API 응답 생성 (비동기)"""
        print(f"=== [DEBUG _format_response] context keys: {context.keys() if context else 'None'} ===")
        if context:
            print(f"=== [DEBUG _format_response] generated_keywords: {context.get('generated_keywords', [])} ===")
        recommendations = []
        
        for i, candidate in enumerate(ranked_products):
            product = candidate["product"]
            
            # LangChain 기반 동적 근거 생성 (비동기)
            reasoning_summary = await self._generate_dynamic_reason_with_langchain(
                candidate, 
                context or {"time_slot": "저녁", "weather": {"weather": "폭염"}}
            )
            
            recommendation = BroadcastRecommendation(
                rank=i+1,
                productInfo=ProductInfo(
                    productId=product.get("product_code", "Unknown"),
                    productName=product.get("product_name", "Unknown"),
                    category=product.get("category_main", "Unknown"),
                    tapeCode=product.get("tape_code"),
                    tapeName=product.get("tape_name")
                ),
                reasoning=Reasoning(
                    summary=reasoning_summary,
                    linkedCategories=[product.get("category_main", "Unknown")],
                    matchedKeywords=context.get("generated_keywords", []) if context else []
                ),
                businessMetrics=BusinessMetrics(
                    pastAverageSales=f"{candidate['predicted_sales']/100000000:.1f}억",
                    marginRate=0.25,
                    stockLevel="High"
                )
            )
            recommendations.append(recommendation)
        
        return BroadcastResponse(
            requestTime="",  # 메인에서 설정
            recommendedCategories=top_categories,
            recommendations=recommendations
        )
    
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
        
        # 시간대/상황 기반 템플릿들
        if context:
            time_slot = context.get("time_slot", "")
            weather = context.get("weather", {}).get("weather", "")
            
            if time_slot == "저녁":
                time_templates = [
                    "저녁 시간대 시청자 특성에 최적화된 상품",
                    "퇴근 후 관심도 높은 저녁 타임 맞춤 상품",
                    "저녁 시간 구매 패턴 분석 결과 선정"
                ]
                templates.extend(time_templates)
            
            if weather == "폭염":
                weather_templates = [
                    "폭염 특수 수요 급증 예상 상품",
                    "무더위 해결사로 시의적절한 편성",
                    "폭염 대비 필수 아이템으로 구매 욕구 자극"
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
        
        # 임시 카테고리 데이터
        mock_categories = [
            RecommendedCategory(rank=1, name="건강식품", reason="트렌드 급상승", predictedSales="높음"),
            RecommendedCategory(rank=2, name="스포츠용품", reason="시즌 적합성", predictedSales="안정적"),
            RecommendedCategory(rank=3, name="가전제품", reason="날씨 연관성", predictedSales="보통")
        ]
        
        # 컨텍스트 생성
        context = {
            "time_slot": "저녁",
            "weather": {"weather": "폭염"},
            "competitors": []
        }
        
        # 개선된 추천 근거 시스템으로 응답 생성
        response = await self._format_response(mock_candidates, mock_categories, context)
        response.requestTime = request_time
        
        logger.info(f"폴백 응답 생성 완료: {len(mock_candidates)}개 추천 (추천 근거 시스템 테스트)")
        return response
    
    async def _generate_dynamic_reason_with_langchain(self, candidate: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """LangChain을 활용한 동적 추천 근거 생성"""
        try:
            product = candidate["product"]
            source = candidate["source"]
            trend_boost = candidate.get("trend_boost", 1.0)
            predicted_sales = candidate.get("predicted_sales", 0)
            
            # 상품 정보
            category = product.get("category_main", "")
            product_name = product.get("product_name", "")
            trend_keyword = candidate.get("trend_keyword", "")
            tape_name = product.get("tape_name", "")
            
            # 컨텍스트 정보
            time_slot = context.get("time_slot", "") if context else ""
            weather = context.get("weather", {}).get("weather", "") if context else ""
            competitors = context.get("competitors", []) if context else []
            
            # 경쟁 상황 분석
            competitor_categories = [comp.get("category_main", "") for comp in competitors]
            has_competition = category in competitor_categories
            
            # 프롬프트 템플릿 생성
            reason_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 홈쇼핑 방송 편성 전문가입니다. 
주어진 상품 정보와 데이터를 바탕으로 간결하고 설득력 있는 추천 근거를 생성해주세요.

다음 규칙을 따라주세요:
1. 한 문장으로 간결하게 작성 (최대 50자)
2. 구체적인 수치나 키워드 포함
3. 시청자가 이해하기 쉬운 표현 사용
4. 긍정적이고 확신에 찬 톤앤매너

근거에 포함할 요소들:
- 트렌드 키워드 활용
- 매출 예측 수치
- 시간대/날씨 적합성
- 경쟁 상황 (독점 편성 등)
- 방송테이프 준비 상태"""),
                ("human", """
상품명: {product_name}
카테고리: {category}
추천 소스: {source}
트렌드 키워드: {trend_keyword}
트렌드 부스트: {trend_boost}
예측 매출: {predicted_sales}만원
방송테이프: {tape_name}
시간대: {time_slot}
날씨: {weather}
경쟁 상황: {competition_status}

위 정보를 바탕으로 이 상품을 추천하는 핵심 근거를 한 문장으로 작성해주세요.
""")
            ])
            
            chain = reason_prompt | self.llm
            
            result = await chain.ainvoke({
                "product_name": product_name,
                "category": category,
                "source": "트렌드" if source == "trend" else "카테고리",
                "trend_keyword": trend_keyword or "없음",
                "trend_boost": f"{trend_boost:.1f}배",
                "predicted_sales": int(predicted_sales / 10000),  # 만원 단위
                "tape_name": tape_name or "미준비",
                "time_slot": time_slot or "미지정",
                "weather": weather or "보통",
                "competition_status": "경쟁 없음" if not has_competition else "경쟁 있음"
            })
            
            return result.content.strip()
            
        except Exception as e:
            logger.error(f"동적 근거 생성 오류: {e}")
            # 폴백: 기존 로직 사용
            return self._generate_recommendation_reason(candidate, context)
    
    async def _predict_category_sales(self, category: str, broadcast_dt: datetime) -> float:
        """카테고리별 XGBoost 매출 예측"""
        try:
            # XGBoost 모델이 요구하는 형식으로 데이터 준비
            import pandas as pd
            
            dummy_data = pd.DataFrame([{
                # Numeric features
                "product_price": 100000,
                "product_avg_profit": 30000000,
                "product_broadcast_count": 10,
                "category_timeslot_avg_profit": 25000000,
                "hour": broadcast_dt.hour,
                "temperature": 20,
                "precipitation": 0,
                
                # Categorical features
                "product_lgroup": category,
                "product_mgroup": category,
                "product_sgroup": category,
                "brand": "Unknown",
                "product_type": "유형",
                "time_slot": self._get_time_slot(broadcast_dt),
                "day_of_week": ["월", "화", "수", "목", "금", "토", "일"][broadcast_dt.weekday()],
                "season": self._get_season(broadcast_dt.month),
                "weather": "Clear",
                
                # Boolean features
                "is_weekend": 1 if broadcast_dt.weekday() >= 5 else 0,
                "is_holiday": 0
            }])
            
            # XGBoost 파이프라인으로 예측 (전처리 포함)
            predicted_sales = self.model.predict(dummy_data)[0]
            return float(predicted_sales)
            
        except Exception as e:
            logger.error(f"카테고리 매출 예측 오류 ({category}): {e}")
            return 50000000  # 기본값
    
    async def _predict_product_sales(self, product: Dict[str, Any], context: Dict[str, Any]) -> float:
        """개별 상품 XGBoost 매출 예측"""
        try:
            import pandas as pd
            
            # XGBoost 모델이 요구하는 형식으로 데이터 준비
            broadcast_dt = context["broadcast_dt"]
            
            product_data = pd.DataFrame([{
                # Numeric features
                "product_price": product.get("product_price", 100000),
                "product_avg_profit": product.get("avg_sales", 30000000),
                "product_broadcast_count": product.get("broadcast_count", 10),
                "category_timeslot_avg_profit": 25000000,
                "hour": broadcast_dt.hour,
                "temperature": context["weather"].get("temperature", 20),
                "precipitation": context["weather"].get("precipitation", 0),
                
                # Categorical features
                "product_lgroup": product.get("category_main", "Unknown"),
                "product_mgroup": product.get("category_middle", "Unknown"),
                "product_sgroup": product.get("category_sub", "Unknown"),
                "brand": "Unknown",
                "product_type": "유형",
                "time_slot": context["time_slot"],
                "day_of_week": ["월", "화", "수", "목", "금", "토", "일"][broadcast_dt.weekday()],
                "season": context["season"],
                "weather": context["weather"].get("weather", "Clear"),
                
                # Boolean features
                "is_weekend": 1 if broadcast_dt.weekday() >= 5 else 0,
                "is_holiday": 0
            }])
            
            # XGBoost 파이프라인으로 예측 (전처리 포함)
            predicted_sales = self.model.predict(product_data)[0]
            return float(predicted_sales)
            
        except Exception as e:
            logger.error(f"상품 매출 예측 오류: {e}")
            return 30000000  # 기본값
    
    async def _get_ace_products_from_category(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """카테고리별 에이스 상품 조회 (방송 테이프 준비 완료 상품만)"""
        try:
            # Qdrant에서 해당 카테고리 상품들 검색 (방송 테이프 준비 완료만)
            ace_products = self.product_embedder.search_products(
                trend_keywords=[category],
                top_k=limit * 3,  # 필터링으로 인한 결과 부족 방지
                score_threshold=0.3,
                only_ready_products=True  # 방송 테이프 준비 완료 상품만
            )
            
            # 카테고리 필터링 및 매출 예측 점수 추가
            filtered_products = []
            for product in ace_products:
                if product.get("category_main") == category:
                    # 과거 매출 실적 기반 점수 추가
                    product["predicted_sales_score"] = min(
                        product.get("product_avg_sales", 10000000) / 100000000, 1.0
                    )
                    filtered_products.append(product)
                    
                if len(filtered_products) >= limit:
                    break
            
            logger.info(f"카테고리 '{category}': {len(filtered_products)}개 방송 준비 완료 상품 발견")
            return filtered_products
            
        except Exception as e:
            logger.error(f"에이스 상품 조회 오류 ({category}): {e}")
            return []
    
    async def _generate_final_recommendations(
        self,
        category_candidates: List[RecommendedCategory],
        trend_products: List[Dict[str, Any]],
        broadcast_time: str,
        recommendation_count: int
    ) -> List[BroadcastRecommendation]:
        """2단계: 최종 후보 선정 및 고속 랭킹"""
        
        final_candidates = []
        
        # 1. 트렌드 상품 우선 포함
        for product in trend_products[:recommendation_count//2]:
            candidate = await self._create_recommendation_item(product, "trend", context)
            if candidate:
                final_candidates.append(candidate)
        
        # 2. 유망 카테고리에서 에이스 상품 선발
        for category in category_candidates[:3]:
            ace_products = await self._get_ace_products_from_category(category.name, 5)
            
            for product in ace_products:
                if len(final_candidates) >= recommendation_count:
                    break
                    
                candidate = await self._create_recommendation_item(product, "category", context)
                if candidate:
                    final_candidates.append(candidate)
        
        # 3. 중복 제거 및 랭킹
        unique_candidates = self._remove_duplicates(final_candidates)
        ranked_candidates = self._rank_candidates(unique_candidates)
        
        # 4. BroadcastRecommendation 객체로 변환
        recommendations = []
        for i, candidate in enumerate(ranked_candidates[:recommendation_count]):
            recommendations.append(BroadcastRecommendation(
                rank=i+1,
                productInfo=ProductInfo(
                    productId=candidate["product_code"],
                    productName=candidate["product_name"],
                    category=candidate.get("category_main", "Unknown")
                ),
                reasoning=Reasoning(
                    summary=candidate["reasoning"]["summary"],
                    linkedCategories=candidate["reasoning"]["linkedCategories"],
                    matchedKeywords=candidate["reasoning"]["matchedKeywords"]
                ),
                businessMetrics=BusinessMetrics(
                    pastAverageSales=f"{candidate['metrics']['pastAverageSales']/100000000:.1f}억",
                    marginRate=candidate['metrics']['marginRate'],
                    stockLevel=candidate['metrics']['stockLevel']
                )
            ))
        
        return recommendations
    
    async def _predict_category_sales(self, category: str, broadcast_dt: datetime) -> float:
        """카테고리별 매출 예측 (간단한 추정)"""
        try:
            # 과거 데이터에서 해당 카테고리의 평균 매출 조회
            query = text("""
                SELECT AVG(gross_profit) as avg_sales
                FROM broadcast_training_dataset 
                WHERE category_main = :category
                AND time_slot = :time_slot
            """)
            
            time_slot = self._get_time_slot(broadcast_dt)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"category": category, "time_slot": time_slot}).fetchone()
                
            return float(result[0]) if result and result[0] else 10000000.0  # 기본값 1천만원
            
        except Exception as e:
            logger.error(f"매출 예측 오류: {e}")
            return 10000000.0
    
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
                SELECT product_code, product_name, category_main, category_middle, 
                       AVG(gross_profit) as avg_sales, COUNT(*) as broadcast_count,
                       tape_code, tape_name
                FROM broadcast_training_dataset 
                WHERE category_main = :category
                GROUP BY product_code, product_name, category_main, category_middle,
                         tape_code, tape_name
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
                    "avg_sales": float(row[4]),
                    "broadcast_count": int(row[5]),
                    "tape_code": row[6],
                    "tape_name": row[7]
                })
            
            return products
            
        except Exception as e:
            logger.error(f"에이스 상품 조회 오류: {e}")
            return []
    
    async def _generate_detailed_summary(self, product: Dict[str, Any], source_type: str, context: Dict[str, Any] = None) -> str:
        """LangChain을 사용한 상세 추천 근거 생성"""
        try:
            # 컨텍스트 정보 준비
            category = product.get("category_main", "")
            avg_sales = product.get("avg_sales", 0)
            
            # 경쟁사 정보 수집
            competitors = context.get("competitors", []) if context else []
            competitor_categories = [comp.get("category_main", "") for comp in competitors]
            has_competition = category in competitor_categories
            
            # 트렌드 키워드 정보
            trend_keywords = context.get("trend_keywords", []) if context else []
            
            # 시간대 정보
            broadcast_time = context.get("broadcast_time", "") if context else ""
            time_period = self._get_time_period(broadcast_time)
            
            # LangChain 프롬프트로 상세 설명 생성
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 홈쇼핑 방송 편성 전문가입니다. 
주어진 정보를 바탕으로 상품 추천 근거를 구체적이고 설득력 있게 작성해주세요.

다음 요소들을 포함해서 작성하세요:
1. 카테고리의 매출 전망
2. 경쟁 상황 분석 (독점 방송 가능성 등)
3. 트렌드 키워드와의 연관성
4. 시간대 적합성

한 문장으로 간결하게 작성해주세요."""),
                ("human", """
상품 정보:
- 카테고리: {category}
- 예상 매출: {avg_sales}만원
- 방송 시간: {time_period}

경쟁 상황:
- 동시간대 경쟁사 카테고리: {competitor_categories}
- 경쟁 여부: {has_competition}

트렌드 키워드: {trend_keywords}
""")
            ])
            
            chain = summary_prompt | self.llm
            
            result = await chain.ainvoke({
                "category": category,
                "avg_sales": int(avg_sales / 10000),  # 만원 단위
                "time_period": time_period,
                "competitor_categories": ", ".join(competitor_categories) if competitor_categories else "없음",
                "has_competition": "있음" if has_competition else "없음",
                "trend_keywords": ", ".join(trend_keywords) if trend_keywords else "없음"
            })
            
            return result.content.strip()
            
        except Exception as e:
            logger.error(f"상세 설명 생성 오류: {e}")
            # 폴백: 기본 템플릿 사용
            if source_type == "trend":
                return f"'{product.get('trend_keyword', '')}' 트렌드와 관련된 인기 상품입니다."
            else:
                return f"'{product.get('category_main', '')}' 카테고리의 베스트셀러 상품입니다."
    
    async def _create_recommendation_item(self, product: Dict[str, Any], source_type: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """추천 아이템 생성"""
        try:
            # 기본 점수 계산
            base_score = product.get("avg_sales", 0) * 0.7
            
            if source_type == "trend":
                base_score *= 1.5  # 트렌드 보너스
                linked_categories = ["트렌드"]
                matched_keywords = [product.get("trend_keyword", "")]
                summary = await self._generate_detailed_summary(product, source_type, context)
            else:
                linked_categories = [product.get("category_main", "")]
                # context에서 생성된 키워드 가져오기
                matched_keywords = []
                if context:
                    matched_keywords = context.get("generated_keywords", []) or context.get("category_keywords", [])
                summary = await self._generate_detailed_summary(product, source_type, context)
            
            return {
                "product_code": product.get("product_code", ""),
                "product_name": product.get("product_name", ""),
                "category_main": product.get("category_main", ""),
                "final_score": base_score,
                "reasoning": {
                    "summary": summary,
                    "linkedCategories": linked_categories,
                    "matchedKeywords": matched_keywords
                },
                "metrics": {
                    "pastAverageSales": product.get("avg_sales", 0),
                    "marginRate": 0.25,  # 기본 마진율
                    "stockLevel": "High"  # 기본 재고 수준
                }
            }
            
        except Exception as e:
            logger.error(f"추천 아이템 생성 오류: {e}")
            return None
    
    def _remove_duplicates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거"""
        seen_products = set()
        unique_candidates = []
        
        for candidate in candidates:
            product_code = candidate.get("product_code", "")
            if product_code and product_code not in seen_products:
                seen_products.add(product_code)
                unique_candidates.append(candidate)
        
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
