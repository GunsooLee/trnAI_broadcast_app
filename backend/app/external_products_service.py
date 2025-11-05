"""외부 상품 (네이버 베스트) 서비스"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
import os

logger = logging.getLogger(__name__)


class ExternalProductsService:
    """네이버 베스트 상품 조회 서비스"""
    
    def __init__(self):
        self.engine = create_engine(os.getenv("POSTGRES_URI"))
    
    def get_latest_best_products(self, limit: int = 20) -> List[Dict]:
        """
        가장 최근 수집된 베스트 상품 TOP 20 조회
        
        Args:
            limit: 조회할 상품 개수 (기본 20개)
        
        Returns:
            상품 리스트 (순위순)
        """
        try:
            query = text("""
                WITH latest_date AS (
                    SELECT MAX(collected_date) as max_date
                    FROM external_products
                )
                SELECT 
                    ep.product_id,
                    ep.name,
                    ep.rank_order,
                    ep.sale_price,
                    ep.discounted_price,
                    ep.discount_ratio,
                    ep.image_url,
                    ep.landing_url,
                    ep.mobile_landing_url,
                    ep.is_delivery_free,
                    ep.delivery_fee,
                    ep.cumulation_sale_count,
                    ep.review_count,
                    ep.review_score,
                    ep.mall_name,
                    ep.channel_no,
                    ep.collected_at,
                    ep.collected_date,
                    -- 전일 대비 순위 변동
                    COALESCE(
                        (SELECT rank_order 
                         FROM external_products ep2 
                         WHERE ep2.product_id = ep.product_id 
                           AND ep2.collected_date = (SELECT max_date FROM latest_date) - INTERVAL '1 day'
                         LIMIT 1
                        ), NULL
                    ) as prev_rank
                FROM external_products ep
                CROSS JOIN latest_date
                WHERE ep.collected_date = latest_date.max_date
                ORDER BY ep.rank_order ASC
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"limit": limit})
                rows = result.fetchall()
            
            products = []
            for row in rows:
                # 순위 변동 계산
                rank_change = None
                rank_change_text = "신규"
                if row.prev_rank is not None:
                    rank_change = row.prev_rank - row.rank_order  # 양수면 상승, 음수면 하락
                    if rank_change > 0:
                        rank_change_text = f"↑{rank_change}"
                    elif rank_change < 0:
                        rank_change_text = f"↓{abs(rank_change)}"
                    else:
                        rank_change_text = "→"
                
                products.append({
                    "product_id": row.product_id,
                    "name": row.name,
                    "rank": row.rank_order,
                    "rank_change": rank_change,
                    "rank_change_text": rank_change_text,
                    "sale_price": row.sale_price,
                    "discounted_price": row.discounted_price,
                    "discount_ratio": row.discount_ratio,
                    "image_url": row.image_url,
                    "landing_url": row.landing_url,
                    "mobile_landing_url": row.mobile_landing_url,
                    "is_delivery_free": row.is_delivery_free,
                    "delivery_fee": row.delivery_fee,
                    "cumulation_sale_count": row.cumulation_sale_count,
                    "review_count": row.review_count,
                    "review_score": row.review_score,
                    "mall_name": row.mall_name,
                    "channel_no": row.channel_no,
                    "collected_at": row.collected_at.isoformat() if row.collected_at else None,
                    "collected_date": row.collected_date.isoformat() if row.collected_date else None
                })
            
            logger.info(f"✅ 최신 베스트 상품 {len(products)}개 조회 완료")
            return products
            
        except Exception as e:
            logger.error(f"베스트 상품 조회 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        수집 통계 조회
        
        Returns:
            통계 정보 (최신 수집일, 총 상품 수 등)
        """
        try:
            query = text("""
                SELECT 
                    MAX(collected_date) as latest_date,
                    COUNT(DISTINCT product_id) as total_products,
                    COUNT(DISTINCT collected_date) as collection_days
                FROM external_products
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                row = result.fetchone()
            
            if row:
                return {
                    "latest_collection_date": row.latest_date.isoformat() if row.latest_date else None,
                    "total_unique_products": row.total_products,
                    "total_collection_days": row.collection_days
                }
            else:
                return {
                    "latest_collection_date": None,
                    "total_unique_products": 0,
                    "total_collection_days": 0
                }
                
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {
                "latest_collection_date": None,
                "total_unique_products": 0,
                "total_collection_days": 0
            }
