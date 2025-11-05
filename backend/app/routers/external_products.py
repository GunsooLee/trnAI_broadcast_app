"""외부 상품 크롤링 API 라우터"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict
import logging

from app.naver_shopping_api_crawler import NaverShoppingAPICrawler
from app.naver_best_crawler import NaverBestCrawler

router = APIRouter(prefix="/api/v1/external", tags=["external_products"])
logger = logging.getLogger(__name__)


@router.get("/crawl-naver-shopping")
async def crawl_naver_shopping(max_products: int = 100) -> Dict:
    """
    네이버 쇼핑 인기 상품 크롤링
    
    Args:
        max_products: 최대 수집 상품 개수 (기본값: 100)
    
    Returns:
        {
            "success": bool,
            "products": List[Dict],
            "count": int,
            "message": str
        }
    """
    try:
        logger.info(f"네이버 쇼핑 크롤링 시작 (max_products={max_products})")
        
        crawler = NaverShoppingAPICrawler()
        products = crawler.get_best_products(max_products=max_products)
        
        if not products:
            raise HTTPException(status_code=500, detail="상품 수집 실패")
        
        logger.info(f"✅ {len(products)}개 상품 수집 완료")
        
        return {
            "success": True,
            "products": products,
            "count": len(products),
            "message": f"{len(products)}개 상품 수집 완료"
        }
        
    except Exception as e:
        logger.error(f"크롤링 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crawl-naver-best")
async def crawl_naver_best(
    max_products: int = 100,
    category_id: str = 'A',
    sort_type: str = 'PRODUCT_BUY',
    period_type: str = 'DAILY'
) -> Dict:
    """
    네이버 쇼핑 베스트 상품 크롤링 (많이 구매한 상품)
    
    Args:
        max_products: 최대 수집 상품 개수 (기본값: 100)
        category_id: 카테고리 ID (A=전체, 50000000=패션의류 등)
        sort_type: 정렬 타입 (PRODUCT_BUY=많이구매, PRODUCT_REVIEW=리뷰많은)
        period_type: 기간 (DAILY=일간, WEEKLY=주간, MONTHLY=월간)
    
    Returns:
        {
            "success": bool,
            "products": List[Dict],
            "count": int,
            "message": str
        }
    """
    try:
        logger.info(f"네이버 베스트 크롤링 시작 (max_products={max_products}, category={category_id})")
        
        crawler = NaverBestCrawler()
        products = crawler.get_best_products(
            max_products=max_products,
            category_id=category_id,
            sort_type=sort_type,
            period_type=period_type
        )
        
        if not products:
            raise HTTPException(status_code=500, detail="상품 수집 실패")
        
        logger.info(f"✅ {len(products)}개 베스트 상품 수집 완료")
        
        return {
            "success": True,
            "products": products,
            "count": len(products),
            "message": f"{len(products)}개 베스트 상품 수집 완료"
        }
        
    except Exception as e:
        logger.error(f"베스트 크롤링 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """크롤러 헬스 체크"""
    return {"status": "ok", "service": "external_products_crawler"}
