"""네이버 쇼핑 베스트 상품 크롤러 (snxbest API 사용)"""
import requests
import logging
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class NaverBestCrawler:
    """네이버 쇼핑 베스트 상품 크롤러 (많이 구매한 상품)"""
    
    def __init__(self):
        self.base_url = "https://snxbest.naver.com/product/best/buy"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://shopping.naver.com/',
            'Accept-Language': 'ko-KR,ko;q=0.9',
        }
    
    def get_best_products(self, 
                         category_id: str = 'A',  # A=전체
                         sort_type: str = 'PRODUCT_BUY',  # 많이 구매한 상품
                         period_type: str = 'DAILY',  # DAILY, WEEKLY, MONTHLY
                         max_products: int = 100) -> List[Dict]:
        """
        베스트 상품 목록 가져오기
        
        Args:
            category_id: 카테고리 ID (A=전체, 50000000=패션의류 등)
            sort_type: 정렬 타입 (PRODUCT_BUY=많이구매, PRODUCT_REVIEW=리뷰많은)
            period_type: 기간 (DAILY=일간, WEEKLY=주간, MONTHLY=월간)
            max_products: 최대 상품 개수
        
        Returns:
            상품 리스트
        """
        try:
            params = {
                'categoryId': category_id,
                'sortType': sort_type,
                'periodType': period_type
            }
            
            logger.info(f"네이버 베스트 API 호출: {self.base_url}")
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API 호출 실패: {response.status_code}")
                return []
            
            # HTML에서 JSON 데이터 추출
            html = response.text
            
            import re
            import json
            
            # self.__next_f.push에서 JSON 데이터 찾기
            # 이스케이프된 JSON 패턴 (\"products\":[...])
            products_pattern = r'\\"products\\":\[(.*?)\],\\"syncDate\\"'
            products_match = re.search(products_pattern, html, re.DOTALL)
            
            products_data = []
            if products_match:
                try:
                    # 이스케이프 제거 및 JSON 배열 재구성
                    escaped_json = products_match.group(1)
                    # 백슬래시 이스케이프 제거
                    unescaped_json = escaped_json.replace('\\"', '"').replace('\\\\', '\\')
                    products_json = '[' + unescaped_json + ']'
                    products_data = json.loads(products_json)
                    logger.info(f"상품 데이터 발견: {len(products_data)}개")
                except Exception as e:
                    logger.error(f"JSON 파싱 실패: {e}")
            
            if not products_data:
                logger.error("상품 데이터를 찾을 수 없음")
                return []
            
            # 상품 파싱
            products = []
            collected_at = datetime.now().isoformat()
            
            for item in products_data[:max_products]:
                try:
                    product = {
                        'product_id': item.get('nvMid') or item.get('productId'),
                        'name': item.get('title', ''),
                        'image_url': item.get('imageUrl', ''),
                        'landing_url': item.get('linkUrl', ''),
                        'mobile_landing_url': item.get('linkUrl', ''),  # 동일
                        
                        # 가격 정보
                        'sale_price': item.get('priceValue', 0),
                        'discounted_price': item.get('discountPriceValue', 0),
                        'discount_ratio': int(item.get('discountRate', '0').replace('%', '') or 0),
                        
                        # 배송 정보
                        'is_delivery_free': item.get('deliveryFeeType') == 'FREE',
                        'delivery_fee': int(item.get('deliveryFee', '0').replace(',', '') or 0),
                        'is_today_dispatch': False,  # API에 없음
                        
                        # 판매 정보
                        'is_sold_out': False,  # API에 없음
                        'cumulation_sale_count': 0,  # API에 없음 (리뷰 수로 대체 가능)
                        'review_count': int(item.get('reviewCount', '0').replace(',', '').replace('+', '') or 0),
                        'review_score': float(item.get('reviewScore', '0') or 0),
                        
                        # 순위 정보
                        'rank_order': item.get('rank', 0),
                        
                        # 판매자 정보
                        'channel_no': str(item.get('chnlSeq', '')),
                        'landing_service': 'SMARTSTORE',
                        'mall_name': item.get('mallNm', ''),
                        
                        # 수집 정보
                        'collected_at': collected_at,
                        
                        # 추가 정보
                        'category_id': category_id,
                        'sort_type': sort_type,
                        'period_type': period_type
                    }
                    
                    products.append(product)
                    
                except Exception as e:
                    logger.error(f"상품 파싱 실패: {e}")
                    continue
            
            logger.info(f"총 {len(products)}개 상품 수집 완료")
            return products
            
        except Exception as e:
            logger.error(f"상품 수집 실패: {e}")
            return []


def main():
    """테스트 실행"""
    import sys
    import json
    
    # 커맨드라인 인자 확인
    json_output = '--json' in sys.argv
    
    if not json_output:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print("=" * 60)
        print("네이버 쇼핑 베스트 크롤러 (많이 구매한 상품)")
        print("=" * 60)
    else:
        # JSON 모드에서는 로그를 stderr로
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stderr
        )
    
    crawler = NaverBestCrawler()
    
    # 베스트 상품 수집
    if not json_output:
        print("\n🏆 많이 구매한 상품 수집 중...")
    
    products = crawler.get_best_products(max_products=100)
    
    if json_output:
        # JSON 형식으로 출력 (n8n용)
        print(json.dumps(products, ensure_ascii=False, indent=2))
    else:
        # 사람이 읽기 쉬운 형식
        if products:
            print(f"\n✅ 성공! {len(products)}개 상품 수집\n")
            
            # 상위 10개 출력
            for i, product in enumerate(products[:10], 1):
                print(f"{i}. {product['name'][:50]}")
                print(f"   순위: {product['rank_order']}위")
                print(f"   가격: {product['sale_price']:,}원", end='')
                if product['discount_ratio'] > 0:
                    print(f" → {product['discounted_price']:,}원 ({product['discount_ratio']}% 할인)")
                else:
                    print()
                print(f"   리뷰: ⭐{product['review_score']} ({product['review_count']:,}개)")
                print(f"   판매자: {product['mall_name']}")
                print()
        else:
            print("\n❌ 상품 수집 실패")
        
        print("=" * 60)
        print("완료!")
        print("=" * 60)


if __name__ == "__main__":
    main()
