"""
네이버 쇼핑 크롤러
Selenium + undetected-chromedriver를 사용하여 네이버 쇼핑 인기 상품 크롤링
"""

import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)


class NaverShoppingCrawler:
    """네이버 쇼핑 크롤러"""
    
    def __init__(self, headless: bool = True):
        """
        Args:
            headless: 헤드리스 모드 (True: 브라우저 창 안 띄움)
        """
        self.headless = headless
        self.driver = None
        
    def __enter__(self):
        """Context manager 진입"""
        self._init_driver()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()
        
    def _init_driver(self):
        """Chrome 드라이버 초기화 (강화된 봇 감지 우회)"""
        try:
            options = uc.ChromeOptions()
            
            if self.headless:
                options.add_argument('--headless=new')  # 새로운 헤드리스 모드
            
            # 기본 옵션
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--start-maximized')
            
            # 봇 감지 우회 옵션 추가
            options.add_argument('--disable-web-security')
            options.add_argument('--disable-features=IsolateOrigins,site-per-process')
            options.add_argument('--disable-site-isolation-trials')
            options.add_argument('--disable-features=BlockInsecurePrivateNetworkRequests')
            
            # 실제 사용자처럼 보이게
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-notifications')
            options.add_argument('--disable-popup-blocking')
            
            # User-Agent 설정 (최신 Chrome)
            options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36')
            
            # 언어 설정
            options.add_argument('--lang=ko-KR')
            options.add_experimental_option('prefs', {
                'intl.accept_languages': 'ko-KR,ko,en-US,en'
            })
            
            # Chrome 버전 자동 감지
            self.driver = uc.Chrome(
                options=options, 
                use_subprocess=True,
                driver_executable_path=None
            )
            
            # JavaScript로 webdriver 속성 숨기기
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['ko-KR', 'ko', 'en-US', 'en']
                    });
                    window.chrome = {
                        runtime: {}
                    };
                '''
            })
            
            logger.info("Chrome 드라이버 초기화 완료 (강화된 우회)")
            
        except Exception as e:
            logger.error(f"Chrome 드라이버 초기화 실패: {e}")
            raise
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Chrome 드라이버 종료")
            except Exception as e:
                logger.error(f"드라이버 종료 실패: {e}")
    
    def _random_sleep(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """랜덤 대기 (인간처럼)"""
        import random
        wait_time = random.uniform(min_sec, max_sec)
        time.sleep(wait_time)
    
    def _human_like_scroll(self):
        """인간처럼 스크롤"""
        import random
        # 랜덤한 속도로 스크롤
        scroll_height = random.randint(300, 800)
        self.driver.execute_script(f"window.scrollBy(0, {scroll_height});")
        self._random_sleep(0.5, 1.5)
    
    def search_products(
        self, 
        keyword: str, 
        max_items: int = 20,
        sort: str = "rel"  # rel: 관련도순, pop: 인기도순, price_asc: 낮은가격순, price_dsc: 높은가격순
    ) -> List[Dict]:
        """
        네이버 쇼핑에서 상품 검색
        
        Args:
            keyword: 검색 키워드
            max_items: 최대 수집 개수
            sort: 정렬 방식
        
        Returns:
            상품 정보 리스트
        """
        try:
            # 먼저 네이버 메인 페이지 방문 (더 자연스럽게)
            logger.info("네이버 메인 페이지 방문")
            self.driver.get("https://www.naver.com")
            self._random_sleep(1, 2)
            
            # 검색 URL
            search_url = f"https://search.shopping.naver.com/search/all?query={keyword}&sort={sort}"
            
            logger.info(f"네이버 쇼핑 검색 시작: {keyword}")
            self.driver.get(search_url)
            
            # 페이지 로딩 대기 (랜덤)
            self._random_sleep(3, 5)
            
            # 상품 리스트 수집
            products = []
            
            # 스크롤하면서 상품 수집
            scroll_count = 0
            max_scrolls = 5
            
            while len(products) < max_items and scroll_count < max_scrolls:
                # 여러 가능한 셀렉터 시도
                selectors = [
                    "div.product_item__MDtDF",
                    "div.product_item",
                    "div.basicList_item__2XT81",
                    "div[class*='product_item']",
                    "div[class*='basicList_item']"
                ]
                
                product_elements = []
                for selector in selectors:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        product_elements = elements
                        logger.info(f"셀렉터 '{selector}'로 {len(elements)}개 상품 발견")
                        break
                
                if not product_elements:
                    logger.warning(f"상품 요소를 찾을 수 없음 (스크롤 {scroll_count + 1})")
                
                for element in product_elements:
                    if len(products) >= max_items:
                        break
                    
                    try:
                        product_info = self._extract_product_info(element)
                        if product_info and product_info not in products:
                            products.append(product_info)
                            logger.debug(f"상품 수집: {product_info['product_name'][:30]}")
                    
                    except Exception as e:
                        logger.warning(f"상품 정보 추출 실패: {e}")
                        continue
                
                # 인간처럼 스크롤
                self._human_like_scroll()
                scroll_count += 1
            
            logger.info(f"총 {len(products)}개 상품 수집 완료")
            return products[:max_items]
            
        except Exception as e:
            logger.error(f"상품 검색 실패: {e}")
            return []
    
    def _extract_product_info(self, element) -> Optional[Dict]:
        """상품 요소에서 정보 추출"""
        try:
            # 상품명
            try:
                product_name = element.find_element(
                    By.CSS_SELECTOR, 
                    "div.product_title__Mmw2K a"
                ).text.strip()
            except:
                product_name = ""
            
            # 가격
            try:
                price_text = element.find_element(
                    By.CSS_SELECTOR, 
                    "span.price_num__S2p_v em"
                ).text.strip()
                price = int(price_text.replace(',', ''))
            except:
                price = 0
            
            # 상품 링크
            try:
                product_link = element.find_element(
                    By.CSS_SELECTOR, 
                    "div.product_title__Mmw2K a"
                ).get_attribute('href')
            except:
                product_link = ""
            
            # 판매처
            try:
                seller = element.find_element(
                    By.CSS_SELECTOR, 
                    "span.product_mall__Tpob9 a"
                ).text.strip()
            except:
                seller = ""
            
            # 리뷰 수
            try:
                review_text = element.find_element(
                    By.CSS_SELECTOR, 
                    "span.product_num__fafe5 em"
                ).text.strip()
                review_count = int(review_text.replace(',', ''))
            except:
                review_count = 0
            
            # 평점
            try:
                rating_text = element.find_element(
                    By.CSS_SELECTOR, 
                    "span.product_grade__IzyU3"
                ).text.strip()
                rating = float(rating_text.replace('별점', '').strip())
            except:
                rating = 0.0
            
            if not product_name or price == 0:
                return None
            
            return {
                'product_name': product_name,
                'price': price,
                'seller': seller,
                'review_count': review_count,
                'rating': rating,
                'product_link': product_link,
                'crawled_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"상품 정보 추출 오류: {e}")
            return None
    
    def get_trending_products(
        self, 
        category: Optional[str] = None,
        max_items: int = 20
    ) -> List[Dict]:
        """
        네이버 쇼핑 인기 상품 조회
        
        Args:
            category: 카테고리 (예: "패션의류", "식품", "가전디지털")
            max_items: 최대 수집 개수
        
        Returns:
            인기 상품 리스트
        """
        try:
            if category:
                url = f"https://shopping.naver.com/home/p/best/category/{category}"
            else:
                url = "https://shopping.naver.com/home/p/best"
            
            logger.info(f"네이버 쇼핑 인기 상품 조회: {category or '전체'}")
            self.driver.get(url)
            
            # 페이지 로딩 대기
            time.sleep(3)
            
            products = []
            
            # 인기 상품 요소 찾기
            product_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "div.product_item__MDtDF"
            )
            
            for element in product_elements[:max_items]:
                try:
                    product_info = self._extract_product_info(element)
                    if product_info:
                        products.append(product_info)
                except Exception as e:
                    logger.warning(f"인기 상품 정보 추출 실패: {e}")
                    continue
            
            logger.info(f"총 {len(products)}개 인기 상품 수집 완료")
            return products
            
        except Exception as e:
            logger.error(f"인기 상품 조회 실패: {e}")
            return []
    
    def get_category_rankings(self, max_categories: int = 10) -> List[Dict]:
        """
        네이버 쇼핑 카테고리별 인기 순위
        
        Args:
            max_categories: 최대 카테고리 수
        
        Returns:
            카테고리별 인기 상품 정보
        """
        try:
            url = "https://shopping.naver.com/home/p/best"
            
            logger.info("네이버 쇼핑 카테고리 랭킹 조회")
            self.driver.get(url)
            
            time.sleep(3)
            
            categories = []
            
            # 카테고리 탭 찾기
            category_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "a.tab_item__b5LS8"
            )
            
            for element in category_elements[:max_categories]:
                try:
                    category_name = element.text.strip()
                    if category_name:
                        categories.append({
                            'category': category_name,
                            'rank': len(categories) + 1
                        })
                except Exception as e:
                    logger.warning(f"카테고리 추출 실패: {e}")
                    continue
            
            logger.info(f"총 {len(categories)}개 카테고리 수집 완료")
            return categories
            
        except Exception as e:
            logger.error(f"카테고리 랭킹 조회 실패: {e}")
            return []


# 테스트 함수
def test_crawler():
    """크롤러 테스트"""
    print("=" * 60)
    print("네이버 쇼핑 크롤러 테스트")
    print("=" * 60)
    
    with NaverShoppingCrawler(headless=True) as crawler:
        # 1. 키워드 검색 테스트
        print("\n1️⃣ 키워드 검색 테스트: '건강식품'")
        products = crawler.search_products("건강식품", max_items=5, sort="pop")
        
        for i, product in enumerate(products, 1):
            print(f"\n{i}. {product['product_name'][:40]}")
            print(f"   가격: {product['price']:,}원")
            print(f"   판매처: {product['seller']}")
            print(f"   리뷰: {product['review_count']}개 | 평점: {product['rating']}")
        
        # 2. 인기 상품 조회 테스트
        print("\n\n2️⃣ 인기 상품 조회 테스트")
        trending = crawler.get_trending_products(max_items=5)
        
        for i, product in enumerate(trending, 1):
            print(f"\n{i}. {product['product_name'][:40]}")
            print(f"   가격: {product['price']:,}원")
        
        # 3. 카테고리 랭킹 테스트
        print("\n\n3️⃣ 카테고리 랭킹 테스트")
        categories = crawler.get_category_rankings(max_categories=5)
        
        for cat in categories:
            print(f"{cat['rank']}. {cat['category']}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_crawler()
