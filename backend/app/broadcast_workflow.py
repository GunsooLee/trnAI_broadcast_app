"""
방송 편성 AI 추천 워크플로우
LangChain 기반 2단계 워크플로우: AI 방향 탐색 + 고속 랭킹
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import pandas as pd
from sqlalchemy import create_engine, text
import os

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .trend_collector import TrendCollector, TrendProcessor
from .product_embedder import ProductEmbedder
from . import broadcast_recommender as br
from .schemas import BroadcastResponse, RecommendedCategory, BroadcastRecommendation, ProductInfo, Reasoning, BusinessMetrics

logger = logging.getLogger(__name__)

class BroadcastWorkflow:
    """방송 편성 AI 추천 워크플로우"""
    
    def __init__(self, model, product_embedder: ProductEmbedder):
        self.model = model  # XGBoost 모델
        self.product_embedder = product_embedder
        self.trend_processor = TrendProcessor(product_embedder)
        
        # LangChain LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # DB 연결
        self.engine = create_engine(os.getenv("DB_URI"))
    
    async def process_broadcast_recommendation(
        self, 
        broadcast_time: str, 
        recommendation_count: int = 5
    ) -> BroadcastResponse:
        """메인 워크플로우: 방송 시간 기반 추천"""
        
        request_time = datetime.now().isoformat()
        logger.info(f"방송 추천 워크플로우 시작: {broadcast_time}")
        
        try:
            # 1단계: AI의 방향 탐색 (숲 찾기)
            context = await self._collect_context(broadcast_time)
            classified_keywords = await self._classify_keywords_with_langchain(context)
            
            # 2. Track A, B 비동기 병렬 실행 (문서 명세 준수)
            track_a_result, track_b_result = await asyncio.gather(
                self._execute_track_a(context, classified_keywords.get("category_keywords", [])),
                self._execute_track_b(context, classified_keywords.get("product_keywords", []))
            )
            
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
            response = self._format_response(ranked_products[:recommendation_count], track_a_result["categories"][:3])
            response.requestTime = request_time
            
            logger.info(f"방송 추천 완료: {len(ranked_products)}개 추천")
            return response
            
        except Exception as e:
            logger.error(f"방송 추천 워크플로우 오류: {e}")
            # 기본 응답 반환
            return BroadcastResponse(
                requestTime=request_time,
                recommendedCategories=[],
                recommendations=[]
            )
    
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
        
        # 트렌드 키워드 수집
        async with TrendCollector() as collector:
            trends = await collector.collect_all_trends()
        context["trends"] = trends
        
        # 시간대 정보
        time_slot = self._get_time_slot(broadcast_dt)
        day_type = "주말" if broadcast_dt.weekday() >= 5 else "평일"
        context["time_slot"] = time_slot
        context["day_type"] = day_type
        
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
        
        # 트렌드 키워드
        trend_keywords = [t["keyword"] for t in context["trends"]]
        all_keywords.extend(trend_keywords)
        
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
            return result
        except Exception as e:
            logger.error(f"키워드 분류 오류: {e}")
            return {
                "category_keywords": all_keywords[:10],
                "product_keywords": trend_keywords[:5]
            }
    
    async def _execute_track_a(self, context: Dict[str, Any], category_keywords: List[str]) -> Dict[str, Any]:
        """Track A: 유망 카테고리 찾기"""
        
        if not category_keywords:
            return {"categories": [], "scores": {}}
        
        # RAG 검색으로 관련 카테고리 찾기
        query = " ".join(category_keywords)
        
        try:
            # Qdrant에서 카테고리 검색 (상품 임베딩 활용)
            similar_products = self.product_embedder.search_products(
                trend_keywords=[query],
                top_k=50,
                score_threshold=0.3
            )
            
            # 카테고리별 그룹핑
            category_scores = {}
            for product in similar_products:
                category = product.get('category_main', 'Unknown')
                score = product.get('similarity_score', 0)
                
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
            
            # 카테고리별 평균 점수 계산 및 XGBoost 예측
            promising_categories = []
            broadcast_dt = datetime.fromisoformat(context["broadcast_time"].replace('Z', '+00:00'))
            
            for category, scores in category_scores.items():
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
                    "reason": f"'{query}' 키워드와 관련성 높음"
                })
            
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
            
            logger.info(f"Track A: 유망 카테고리 {len(result)}개 발견")
            return {"categories": result, "scores": category_scores}
            
        except Exception as e:
            logger.error(f"Track A 오류: {e}")
            return {"categories": [], "scores": {}}
    
    async def _execute_track_b(self, context: Dict[str, Any], product_keywords: List[str]) -> Dict[str, Any]:
        """Track B: 트렌드 상품 찾기"""
        
        if not product_keywords:
            return {"products": [], "trend_scores": {}}
        
        trend_products = []
        trend_scores = {}
        
        for keyword in product_keywords[:3]:  # 상위 3개 키워드만
            try:
                similar_products = self.product_embedder.search_products(
                    trend_keywords=[keyword],
                    top_k=10,
                    score_threshold=0.5
                )
                
                for product in similar_products:
                    product["trend_keyword"] = keyword
                    product["trend_boost"] = self._calculate_trend_boost(keyword, context)
                    trend_products.append(product)
                    
                    # 트렌드 점수 저장
                    product_id = product.get("product_id", "unknown")
                    trend_scores[product_id] = product["trend_boost"]
                    
            except Exception as e:
                logger.error(f"Track B 오류 ({keyword}): {e}")
        
        logger.info(f"Track B: 트렌드 상품 {len(trend_products)}개 발견")
        return {"products": trend_products, "trend_scores": trend_scores}
    
    def _calculate_trend_boost(self, keyword: str, context: Dict[str, Any]) -> float:
        """트렌드 부스트 점수 계산"""
        base_boost = 1.0
        
        # 날씨 기반 부스트
        weather = context.get("weather", {})
        temp = weather.get("temperature", 20)
        
        if keyword in ["아이스크림", "선풍기", "에어컨"] and temp > 25:
            base_boost += 0.3
        elif keyword in ["히터", "패딩", "난방"] and temp < 10:
            base_boost += 0.3
            
        # 시간대 기반 부스트
        hour = datetime.now().hour
        if keyword in ["커피", "모닝"] and 6 <= hour <= 10:
            base_boost += 0.2
        elif keyword in ["저녁", "디너"] and 17 <= hour <= 21:
            base_boost += 0.2
            
        return base_boost
    
    async def _generate_candidates(self, promising_categories: List[RecommendedCategory], trend_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """후보군 생성 및 통합"""
        candidates = []
        
        # 1. 트렌드 상품 우선 포함
        for product in trend_products:
            candidates.append({
                "product": product,
                "source": "trend",
                "base_score": product.get("similarity_score", 0.5),
                "trend_boost": product.get("trend_boost", 1.0)
            })
        
        # 2. 유망 카테고리에서 에이스 상품 선발
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
    
    def _format_response(self, ranked_products: List[Dict[str, Any]], top_categories: List[RecommendedCategory]) -> BroadcastResponse:
        """API 응답 생성"""
        recommendations = []
        
        for i, candidate in enumerate(ranked_products):
            product = candidate["product"]
            
            recommendation = BroadcastRecommendation(
                rank=i+1,
                productName=product.get("product_name", "Unknown"),
                category=product.get("category_main", "Unknown"),
                predictedSales=f"{candidate['predicted_sales']/100000000:.1f}억",
                confidence=f"{candidate['final_score']*100:.1f}%",
                reason=self._generate_recommendation_reason(candidate)
            )
            recommendations.append(recommendation)
        
        return BroadcastResponse(
            requestTime="",  # 메인에서 설정
            recommendedCategories=top_categories,
            recommendations=recommendations
        )
    
    def _generate_recommendation_reason(self, candidate: Dict[str, Any]) -> str:
        """추천 근거 생성"""
        source = candidate["source"]
        trend_boost = candidate["trend_boost"]
        
        if source == "trend" and trend_boost > 1.2:
            return "실시간 트렌드 급상승 + 높은 매출 예측"
        elif source == "trend":
            return "실시간 트렌드 반영"
        elif source == "category":
            return "유망 카테고리 내 에이스 상품"
        else:
            return "AI 종합 분석 결과"
    
    async def _predict_category_sales(self, category: str, broadcast_dt: datetime) -> float:
        """카테고리별 XGBoost 매출 예측"""
        try:
            # 가상의 카테고리 상품으로 예측 (실제로는 해당 카테고리의 대표 상품들 사용)
            dummy_product = {
                "product_name": f"{category} 대표상품",
                "category_main": category,
                "product_price": 50000,  # 평균 가격
                "broadcast_time": broadcast_dt.isoformat(),
                "weather": "맑음",
                "temperature": 20,
                "precipitation": 0
            }
            
            # XGBoost 예측 실행
            predicted_sales = br.predict_sales([dummy_product])
            return predicted_sales[0] if predicted_sales else 50000000  # 기본값 5천만원
            
        except Exception as e:
            logger.error(f"카테고리 매출 예측 오류 ({category}): {e}")
            return 50000000  # 기본값
    
    async def _predict_product_sales(self, product: Dict[str, Any], context: Dict[str, Any]) -> float:
        """개별 상품 XGBoost 매출 예측"""
        try:
            # 컨텍스트 정보를 상품 데이터에 추가
            enhanced_product = product.copy()
            enhanced_product.update({
                "broadcast_time": context["broadcast_time"],
                "weather": context["weather"].get("weather", "맑음"),
                "temperature": context["weather"].get("temperature", 20),
                "precipitation": context["weather"].get("precipitation", 0),
                "time_slot": context["time_slot"],
                "season": context["season"]
            })
            
            # XGBoost 예측 실행
            predicted_sales = br.predict_sales([enhanced_product])
            return predicted_sales[0] if predicted_sales else 30000000  # 기본값 3천만원
            
        except Exception as e:
            logger.error(f"상품 매출 예측 오류: {e}")
            return 30000000  # 기본값
    
    async def _get_ace_products_from_category(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """카테고리별 에이스 상품 조회"""
        try:
            # Qdrant에서 해당 카테고리 상품들 검색
            ace_products = self.product_embedder.search_products(
                trend_keywords=[category],
                top_k=limit * 2,  # 여유분 확보
                score_threshold=0.3
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
            candidate = await self._create_recommendation_item(product, "trend")
            if candidate:
                final_candidates.append(candidate)
        
        # 2. 유망 카테고리에서 에이스 상품 선발
        for category in category_candidates[:3]:
            ace_products = await self._get_ace_products_from_category(category.name, 5)
            
            for product in ace_products:
                if len(final_candidates) >= recommendation_count:
                    break
                    
                candidate = await self._create_recommendation_item(product, "category")
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
                SELECT AVG(sales_amount) as avg_sales
                FROM broadcast_training_dataset 
                WHERE product_mgroup = :category
                AND time_slot = :time_slot
            """)
            
            time_slot = self._get_time_slot(broadcast_dt)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"category": category, "time_slot": time_slot}).fetchone()
                
            return float(result[0]) if result and result[0] else 10000000.0  # 기본값 1천만원
            
        except Exception as e:
            logger.error(f"매출 예측 오류: {e}")
            return 10000000.0
    
    async def _get_ace_products_from_category(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """카테고리별 에이스 상품 조회"""
        try:
            query = text("""
                SELECT product_code, product_name, category_main, category_middle, 
                       AVG(sales_amount) as avg_sales, COUNT(*) as broadcast_count
                FROM broadcast_training_dataset 
                WHERE product_mgroup = :category
                GROUP BY product_code, product_name, category_main, category_middle
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
                    "broadcast_count": int(row[5])
                })
            
            return products
            
        except Exception as e:
            logger.error(f"에이스 상품 조회 오류: {e}")
            return []
    
    async def _create_recommendation_item(self, product: Dict[str, Any], source_type: str) -> Optional[Dict[str, Any]]:
        """추천 아이템 생성"""
        try:
            # 기본 점수 계산
            base_score = product.get("avg_sales", 0) * 0.7
            
            if source_type == "trend":
                base_score *= 1.5  # 트렌드 보너스
                linked_categories = ["트렌드"]
                matched_keywords = [product.get("trend_keyword", "")]
                summary = f"'{product.get('trend_keyword', '')}' 트렌드와 관련된 인기 상품입니다."
            else:
                linked_categories = [product.get("category_main", "")]
                matched_keywords = []
                summary = f"'{product.get('category_main', '')}' 카테고리의 베스트셀러 상품입니다."
            
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
