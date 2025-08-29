"""
트렌드 강화 추천 시스템
RAG + XGBoost + 트렌드 데이터를 통합한 추천 파이프라인
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import pandas as pd

from .trend_collector import TrendCollector, TrendProcessor, TrendKeyword
from .product_embedder import ProductEmbedder
from . import broadcast_recommender as br

logger = logging.getLogger(__name__)

class TrendEnhancedRecommender:
    """트렌드 데이터를 활용한 강화된 추천 시스템"""
    
    def __init__(self, model, product_embedder: ProductEmbedder):
        self.model = model
        self.product_embedder = product_embedder
        self.trend_processor = TrendProcessor(product_embedder)
        self.trend_cache = {}
        self.cache_expiry = timedelta(hours=1)  # 트렌드 캐시 유효시간
    
    async def get_cached_trends(self) -> List[TrendKeyword]:
        """캐시된 트렌드 데이터 반환 (만료시 새로 수집)"""
        now = datetime.now()
        
        if (self.trend_cache.get('timestamp') and 
            now - self.trend_cache['timestamp'] < self.cache_expiry and
            self.trend_cache.get('trends')):
            logger.info("캐시된 트렌드 데이터 사용")
            return self.trend_cache['trends']
        
        # 새로운 트렌드 데이터 수집
        logger.info("새로운 트렌드 데이터 수집 중...")
        async with TrendCollector() as collector:
            trends = await collector.collect_all_trends()
        
        self.trend_cache = {
            'trends': trends,
            'timestamp': now
        }
        
        return trends
    
    async def enhance_recommendations_with_trends(
        self, 
        base_recommendations: List[Dict],
        extracted_params: Dict[str, Any]
    ) -> List[Dict]:
        """기본 추천 결과를 트렌드 데이터로 강화"""
        
        try:
            # 트렌드 데이터 수집
            trends = await self.get_cached_trends()
            
            if not trends:
                logger.warning("트렌드 데이터가 없어 기본 추천 반환")
                return base_recommendations
            
            # 트렌드-상품 매칭
            matched_results = await self.trend_processor.match_trends_to_products(trends)
            
            # 추천 결과 강화
            enhanced_recommendations = []
            
            for rec in base_recommendations:
                enhanced_rec = rec.copy()
                
                # 상품 코드나 카테고리로 트렌드 매칭 확인
                product_code = rec.get('product_code', '')
                category = rec.get('category', '')
                
                trend_boost = 1.0
                matched_trends = []
                
                # 트렌드 매칭 확인
                for trend_keyword, match_data in matched_results.items():
                    matched_products = match_data.get('matched_products', [])
                    
                    # 상품 코드 또는 카테고리 매칭 확인
                    for product in matched_products:
                        if (product.get('product_code') == product_code or
                            category.lower() in product.get('category', '').lower()):
                            
                            # 트렌드 부스트 팩터 적용
                            trend_score = match_data['trend_info'].get('score', 0)
                            boost_factor = self.trend_processor.calculate_trend_boost_factor(trend_score)
                            trend_boost = max(trend_boost, boost_factor)
                            
                            matched_trends.append({
                                'keyword': trend_keyword,
                                'score': trend_score,
                                'source': match_data['trend_info'].get('source', ''),
                                'similarity': product.get('similarity', 0)
                            })
                
                # 예측 매출에 트렌드 부스트 적용
                original_sales = enhanced_rec.get('predicted_sales', 0)
                enhanced_rec['predicted_sales'] = original_sales * trend_boost
                enhanced_rec['trend_boost_factor'] = trend_boost
                enhanced_rec['matched_trends'] = matched_trends
                
                # 트렌드 관련 메타데이터 추가
                if matched_trends:
                    enhanced_rec['trend_enhanced'] = True
                    enhanced_rec['top_trend'] = max(matched_trends, key=lambda x: x['score'])
                else:
                    enhanced_rec['trend_enhanced'] = False
                
                enhanced_recommendations.append(enhanced_rec)
            
            # 트렌드 부스트가 적용된 순서로 재정렬
            enhanced_recommendations.sort(
                key=lambda x: x.get('predicted_sales', 0), 
                reverse=True
            )
            
            logger.info(f"트렌드 강화 완료: {len(enhanced_recommendations)}개 추천")
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"트렌드 강화 처리 중 오류: {e}")
            return base_recommendations
    
    async def get_trend_based_candidates(
        self, 
        extracted_params: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict]:
        """트렌드 기반 추가 후보 생성"""
        
        try:
            trends = await self.get_cached_trends()
            if not trends:
                return []
            
            # 상위 트렌드 키워드로 상품 검색
            trend_candidates = []
            
            for trend in trends[:5]:  # 상위 5개 트렌드만 사용
                if self.product_embedder:
                    similar_products = self.product_embedder.search_products(
                        trend_keywords=[trend.keyword],
                        top_k=3
                    )
                    
                    for product in similar_products:
                        # 트렌드 기반 후보 생성
                        candidate = {
                            'product_code': product.get('product_code', ''),
                            'category': product.get('category', ''),
                            'product_name': product.get('product_name', ''),
                            'trend_keyword': trend.keyword,
                            'trend_score': trend.score,
                            'trend_source': trend.source,
                            'similarity_score': product.get('similarity', 0),
                            'predicted_sales': 0,  # XGBoost로 예측 필요
                            'candidate_type': 'trend_based'
                        }
                        
                        trend_candidates.append(candidate)
            
            # 중복 제거 (상품 코드 기준)
            seen_products = set()
            unique_candidates = []
            
            for candidate in trend_candidates:
                product_code = candidate['product_code']
                if product_code not in seen_products:
                    seen_products.add(product_code)
                    unique_candidates.append(candidate)
            
            return unique_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"트렌드 기반 후보 생성 중 오류: {e}")
            return []
    
    async def recommend_with_trends(
        self, 
        extracted_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """트렌드를 반영한 통합 추천"""
        
        try:
            # 1. 기본 XGBoost 추천 실행
            logger.info("기본 XGBoost 추천 실행 중...")
            base_result = await self._get_base_recommendations(extracted_params)
            base_recommendations = base_result.get('recommendations', [])
            
            # 2. 트렌드 데이터로 추천 강화
            logger.info("트렌드 데이터로 추천 강화 중...")
            enhanced_recommendations = await self.enhance_recommendations_with_trends(
                base_recommendations, extracted_params
            )
            
            # 3. 트렌드 기반 추가 후보 생성
            logger.info("트렌드 기반 추가 후보 생성 중...")
            trend_candidates = await self.get_trend_based_candidates(extracted_params)
            
            # 4. 결과 통합
            final_recommendations = enhanced_recommendations.copy()
            
            # 트렌드 후보 중 기존 추천에 없는 것들 추가
            existing_products = {rec.get('product_code') for rec in final_recommendations}
            
            for candidate in trend_candidates:
                if candidate['product_code'] not in existing_products:
                    # XGBoost로 매출 예측 (간단한 추정)
                    candidate['predicted_sales'] = self._estimate_sales_for_trend_candidate(
                        candidate, extracted_params
                    )
                    final_recommendations.append(candidate)
            
            # 최종 정렬 (예측 매출 기준)
            final_recommendations.sort(
                key=lambda x: x.get('predicted_sales', 0), 
                reverse=True
            )
            
            # 시간대별로 상위 N개 선택
            time_slots = extracted_params.get('time_slots', ['오전'])
            recommendations_by_slot = {}
            
            for time_slot in time_slots:
                slot_recommendations = [
                    rec for rec in final_recommendations 
                    if rec.get('time_slot', '오전') == time_slot
                ][:5]  # 시간대별 상위 5개
                
                recommendations_by_slot[time_slot] = slot_recommendations
            
            # 트렌드 요약 정보 추가
            trends = await self.get_cached_trends()
            trend_summary = [
                {
                    'keyword': trend.keyword,
                    'score': trend.score,
                    'source': trend.source,
                    'category': trend.category
                }
                for trend in trends[:10]
            ]
            
            return {
                'extracted_params': extracted_params,
                'recommendations': final_recommendations[:10],  # 상위 10개
                'recommendations_by_slot': recommendations_by_slot,
                'trend_summary': trend_summary,
                'enhancement_applied': True,
                'total_candidates': len(final_recommendations)
            }
            
        except Exception as e:
            logger.error(f"트렌드 통합 추천 중 오류: {e}")
            # 오류 시 기본 추천 반환
            return await self._get_base_recommendations(extracted_params)
    
    async def _get_base_recommendations(self, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """기본 XGBoost 추천 실행"""
        try:
            # broadcast_recommender의 recommend 함수 호출
            result = br.recommend(
                target_date=extracted_params.get('date'),
                time_slots=extracted_params.get('time_slots', ['오전']),
                product_codes=extracted_params.get('products'),
                categories=extracted_params.get('categories'),
                weather_info=extracted_params.get('weather_info'),
                model=self.model
            )
            
            return {
                'extracted_params': extracted_params,
                'recommendations': result if isinstance(result, list) else [],
                'enhancement_applied': False
            }
            
        except Exception as e:
            logger.error(f"기본 추천 실행 중 오류: {e}")
            return {
                'extracted_params': extracted_params,
                'recommendations': [],
                'enhancement_applied': False
            }
    
    def _estimate_sales_for_trend_candidate(
        self, 
        candidate: Dict, 
        extracted_params: Dict[str, Any]
    ) -> float:
        """트렌드 후보의 매출 추정 (간단한 휴리스틱)"""
        
        # 기본 매출 추정치
        base_sales = 50000000  # 5천만원 기본값
        
        # 트렌드 점수 반영
        trend_score = candidate.get('trend_score', 0)
        trend_multiplier = 1 + (trend_score / 100)
        
        # 유사도 점수 반영
        similarity = candidate.get('similarity_score', 0)
        similarity_multiplier = 1 + (similarity * 0.5)
        
        estimated_sales = base_sales * trend_multiplier * similarity_multiplier
        
        return estimated_sales
