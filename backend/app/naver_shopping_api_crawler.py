"""
네이버 쇼핑 API 크롤러
실제 API를 사용하여 인기 상품 데이터 수집
"""
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NaverShoppingAPICrawler:
    """네이버 쇼핑 API 크롤러 (동적 ID 추출)"""
    
    def __init__(self):
        self.layout_api_url = "https://shopping.naver.com/responsive/api/shopv/ns/layout"
        self.meta_api_url = "https://shopping.naver.com/_next/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://shopping.naver.com/ns/home/',
            'Accept-Language': 'ko-KR,ko;q=0.9',
        }
        
        # 캐시된 ID (같은 인스턴스에서 재사용)
        self._cached_build_id = None
        self._cached_tab_id = None
        self._cached_promotion_id = None
    
    def _get_dynamic_ids(self) -> Dict[str, str]:
        """
        동적으로 Build ID, Tab ID, Promotion ID 추출
        
        Returns:
            {'build_id': str, 'tab_id': str, 'promotion_id': str}
        """
        # 캐시 확인
        if self._cached_build_id and self._cached_tab_id and self._cached_promotion_id:
            logger.info("캐시된 ID 사용")
            return {
                'build_id': self._cached_build_id,
                'tab_id': self._cached_tab_id,
                'promotion_id': self._cached_promotion_id
            }
        
        try:
            # 1단계: 메인 페이지에서 Build ID 추출
            logger.info("Build ID 추출 중...")
            main_url = "https://shopping.naver.com/ns/home/"
            response = requests.get(main_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"메인 페이지 로드 실패: {response.status_code}")
                return {}
            
            # HTML에서 buildId 찾기
            html = response.text
            build_id = None
            
            # Next.js buildId는 보통 <script> 태그나 JSON에 있음
            import re
            match = re.search(r'"buildId":"([^"]+)"', html)
            if match:
                build_id = match.group(1)
                logger.info(f"Build ID 발견: {build_id}")
            else:
                logger.warning("Build ID를 찾을 수 없음, 기본값 사용")
                build_id = "C2g5NtExuzW7rvim22xyR"  # 폴백
            
            # 2단계: best.json에서 Tab ID와 Promotion ID 추출
            logger.info("Tab ID 및 Promotion ID 추출 중...")
            best_url = f"{self.meta_api_url}/{build_id}/ns/home/best.json"
            params = {
                'demo': 'ALL',
                'client': 'BROWSER',
                'deviceType': 'PC'
            }
            
            response = requests.get(best_url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"best.json 로드 실패: {response.status_code}")
                return {}
            
            data = response.json()
            
            # BEST 탭 찾기 (initialApolloState에서)
            page_props = data.get('pageProps', {})
            apollo_state = page_props.get('initialApolloState', {})
            
            tab_id = None
            promotion_id = None
            
            # 1순위: PROMOTION serviceCode를 가진 탭 찾기 (안정적으로 작동)
            for key, value in apollo_state.items():
                if isinstance(value, dict) and value.get('serviceCode') == 'PROMOTION':
                    tab_id = value.get('id')
                    tab_title = value.get('title', '프로모션')
                    # additionalInfo에서 promotionId 추출
                    additional_info = value.get('additionalInfo', {})
                    if additional_info:
                        promotion_id = additional_info.get('promotionId')
                    
                    if tab_id and promotion_id:
                        logger.info(f"✅ 프로모션 탭 발견 - {tab_title} (Tab: {tab_id}, Promotion: {promotion_id})")
                        break
            
            # 3순위: promotionInfo에서 가져오기 (PROMOTION 탭만 필요)
            if not tab_id:
                logger.warning("탭을 찾지 못함, promotionInfo 사용")
                promotion_info = page_props.get('promotionInfo', [])
                if promotion_info and len(promotion_info) > 0:
                    last_promo = promotion_info[-1]
                    promotion_id = last_promo.get('promotionId')
                    tab_id = page_props.get('activeTabId')
                    logger.info(f"폴백: Tab: {tab_id}, Promotion: {promotion_id}")
            
            if not tab_id:
                logger.error(f"ID 추출 실패 - tab_id: {tab_id}")
                return {}
            
            # serviceCode는 PROMOTION 고정
            service_code = 'PROMOTION'
            
            logger.info(f"✅ ID 추출 완료 - Tab: {tab_id}, Promotion: {promotion_id}")
            
            # 캐시 저장
            self._cached_build_id = build_id
            self._cached_tab_id = tab_id
            self._cached_promotion_id = promotion_id
            
            return {
                'build_id': build_id,
                'tab_id': tab_id,
                'promotion_id': promotion_id,
                'service_code': service_code
            }
            
        except Exception as e:
            logger.error(f"ID 추출 실패: {e}")
            return {}
    
    def get_best_products(self, max_products: int = 100) -> List[Dict]:
        """
        인기 상품 목록 가져오기 (동적 ID 사용)
        
        Args:
            max_products: 최대 상품 개수
        
        Returns:
            상품 리스트
        """
        try:
            # 동적으로 ID 가져오기
            ids = self._get_dynamic_ids()
            
            if not ids:
                logger.error("ID 추출 실패")
                return []
            
            tab_id = ids['tab_id']
            promotion_id = ids.get('promotion_id')
            service_code = ids.get('service_code', 'BEST')
            
            # API 파라미터 설정
            params = {
                'tabId': tab_id,
                'serviceCode': service_code,
                'panelId': 'null',
                'promotionId': promotion_id,
                'deviceType': 'PC',
                'clientType': 'BROWSER',
                'homeDemo': 'NOT_LOGIN',
                'bestDemo': 'ALL'
            }
            
            logger.info(f"네이버 쇼핑 API 호출: {self.layout_api_url}")
            response = requests.get(
                self.layout_api_url,
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API 호출 실패: {response.status_code}")
                return []
            
            data = response.json()
            
            if not data.get('isSuccess'):
                logger.error(f"API 응답 실패: {data}")
                return []
            
            # 상품 데이터 추출
            products = self._extract_products(data)
            
            logger.info(f"총 {len(products)}개 상품 수집 완료")
            return products[:max_products]
            
        except Exception as e:
            logger.error(f"상품 수집 실패: {e}")
            return []
    
    def _extract_products(self, data: Dict) -> List[Dict]:
        """
        API 응답에서 상품 데이터 추출
        
        Args:
            data: API 응답 데이터
        
        Returns:
            상품 리스트
        """
        products = []
        
        try:
            layout = data.get('data', {}).get('layout', {})
            page_data = layout.get('pageData', {})
            layers = page_data.get('layers', [])
            
            # 모든 레이어 순회
            for layer in layers:
                blocks = layer.get('blocks', [])
                
                for block in blocks:
                    items = block.get('items', [])
                    
                    for item in items:
                        contents = item.get('contents', [])
                        
                        for content in contents:
                            # productId가 있으면 상품 데이터
                            if 'productId' in content:
                                product = self._parse_product(content)
                                if product:
                                    products.append(product)
            
            return products
            
        except Exception as e:
            logger.error(f"상품 데이터 추출 실패: {e}")
            return []
    
    def _parse_product(self, content: Dict) -> Optional[Dict]:
        """
        상품 데이터 파싱
        
        Args:
            content: 상품 원본 데이터
        
        Returns:
            파싱된 상품 정보
        """
        try:
            product = {
                'product_id': content.get('productId'),
                'name': content.get('name'),
                'image_url': content.get('imageUrl'),
                'landing_url': content.get('landingUrl'),
                'mobile_landing_url': content.get('mobileLandingUrl'),
                
                # 가격 정보
                'sale_price': content.get('salePrice'),
                'discounted_price': content.get('discountedPrice'),
                'discount_ratio': content.get('discountedRatio', 0),
                
                # 배송 정보
                'is_delivery_free': content.get('isDeliveryFree', False),
                'delivery_fee': content.get('deliveryFee', 0),
                'is_today_dispatch': content.get('isTodayDispatch', False),
                
                # 상태 정보
                'is_sold_out': content.get('isSoldOut', False),
                'cumulation_sale_count': content.get('cumulationSaleCount', 0),
                
                # 기타
                'order': content.get('order'),  # 순위
                'channel_no': content.get('channelNo'),
                'landing_service': content.get('landingService'),
                
                # 수집 시간
                'collected_at': datetime.now().isoformat()
            }
            
            return product
            
        except Exception as e:
            logger.error(f"상품 파싱 실패: {e}")
            return None


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
        print("네이버 쇼핑 API 크롤러 테스트 (동적 ID)")
        print("=" * 60)
    else:
        # JSON 모드에서는 로그를 stderr로
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stderr
        )
    
    crawler = NaverShoppingAPICrawler()
    
    # 인기 상품 수집 (동적 ID 사용)
    if not json_output:
        print("\n1️⃣ 동적 ID 추출 및 상품 수집 중...")
    
    products = crawler.get_best_products(max_products=100)
    
    if json_output:
        # JSON 형식으로 출력 (n8n용)
        print(json.dumps(products, ensure_ascii=False, indent=2))
    else:
        # 사람이 읽기 쉬운 형식
        if products:
            print(f"\n✅ 성공! {len(products)}개 상품 수집\n")
            
            # 상위 5개 출력
            for i, product in enumerate(products[:5], 1):
                print(f"{i}. {product['name'][:50]}")
                print(f"   ID: {product['product_id']}")
                print(f"   가격: {product['sale_price']:,}원")
                if product['discount_ratio'] > 0:
                    print(f"   할인: {product['discount_ratio']}% → {product['discounted_price']:,}원")
                print(f"   무료배송: {'O' if product['is_delivery_free'] else 'X'}")
                print(f"   품절: {'O' if product['is_sold_out'] else 'X'}")
                print(f"   판매량: {product['cumulation_sale_count']}개")
                print()
        else:
            print("\n❌ 상품 수집 실패")
        
        print("=" * 60)
        print("완료!")
        print("=" * 60)


if __name__ == "__main__":
    main()
