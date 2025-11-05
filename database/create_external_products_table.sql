-- 외부 상품 (네이버 쇼핑) 테이블 생성
-- 네이버 쇼핑에서 크롤링한 인기 상품 데이터 저장

CREATE TABLE IF NOT EXISTS external_products (
    -- 기본 정보
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,          -- 네이버 상품 ID
    name TEXT NOT NULL,                       -- 상품명
    
    -- 가격 정보
    sale_price INTEGER,                       -- 판매가
    discounted_price INTEGER,                 -- 할인가
    discount_ratio INTEGER DEFAULT 0,         -- 할인율 (%)
    
    -- 이미지 및 링크
    image_url TEXT,                           -- 상품 이미지 URL
    landing_url TEXT,                         -- PC 랜딩 URL
    mobile_landing_url TEXT,                  -- 모바일 랜딩 URL
    
    -- 배송 정보
    is_delivery_free BOOLEAN DEFAULT FALSE,   -- 무료배송 여부
    delivery_fee INTEGER DEFAULT 0,           -- 배송비
    is_today_dispatch BOOLEAN DEFAULT FALSE,  -- 오늘출발 여부
    
    -- 상태 정보
    is_sold_out BOOLEAN DEFAULT FALSE,        -- 품절 여부
    cumulation_sale_count INTEGER DEFAULT 0,  -- 누적 판매량
    
    -- 순위 및 메타 정보
    rank_order INTEGER,                       -- 순위 (order 필드)
    channel_no VARCHAR(50),                   -- 채널 번호
    landing_service VARCHAR(50),              -- 랜딩 서비스 (SMARTSTORE 등)
    
    -- 수집 정보
    collected_at TIMESTAMP NOT NULL,          -- 수집 시간
    collected_date DATE NOT NULL,             -- 수집 날짜 (YYYY-MM-DD)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 최초 생성 시간
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 최종 수정 시간
    
    -- UNIQUE 제약: 하루에 한 번만 INSERT, 같은 날 재실행 시 UPDATE
    CONSTRAINT unique_product_per_day UNIQUE (product_id, collected_date)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_external_products_product_id ON external_products(product_id);
CREATE INDEX IF NOT EXISTS idx_external_products_collected_at ON external_products(collected_at);
CREATE INDEX IF NOT EXISTS idx_external_products_collected_date ON external_products(collected_date);
CREATE INDEX IF NOT EXISTS idx_external_products_rank_order ON external_products(rank_order);
CREATE INDEX IF NOT EXISTS idx_external_products_sale_price ON external_products(sale_price);
CREATE INDEX IF NOT EXISTS idx_external_products_discount_ratio ON external_products(discount_ratio);

-- 업데이트 시간 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_external_products_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_external_products_updated_at
    BEFORE UPDATE ON external_products
    FOR EACH ROW
    EXECUTE FUNCTION update_external_products_updated_at();

-- 코멘트 추가
COMMENT ON TABLE external_products IS '네이버 쇼핑 베스트 상품 데이터 (외부 크롤링, 일별 이력 저장)';
COMMENT ON COLUMN external_products.product_id IS '네이버 쇼핑 상품 ID';
COMMENT ON COLUMN external_products.name IS '상품명';
COMMENT ON COLUMN external_products.sale_price IS '정상 판매가';
COMMENT ON COLUMN external_products.discounted_price IS '할인 적용가';
COMMENT ON COLUMN external_products.discount_ratio IS '할인율 (%)';
COMMENT ON COLUMN external_products.cumulation_sale_count IS '누적 판매량 (리뷰 수)';
COMMENT ON COLUMN external_products.collected_at IS '데이터 수집 시간';
COMMENT ON COLUMN external_products.collected_date IS '수집 날짜 (일별 이력 관리용)';
