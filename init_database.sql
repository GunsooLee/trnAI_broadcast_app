-- 홈쇼핑 방송 추천 시스템 데이터베이스 초기화


-- 1. 날씨 데이터 테이블
CREATE TABLE IF NOT EXISTS taiweather_daily (
    weather_date DATE PRIMARY KEY,
    weather VARCHAR(50),
    temperature DECIMAL(5,2),
    precipitation DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 상품 정보 테이블
CREATE TABLE IF NOT EXISTS TAIGOODS (
    product_code VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(200),
    category_main VARCHAR(100),
    category_middle VARCHAR(100),
    category_sub VARCHAR(100),
    price DECIMAL(10,2),
    brand VARCHAR(100),
    product_type VARCHAR(10), -- 유형/무형
    embedded_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2-1. 방송테이프 정보 테이블 (TAIPGMTAPE)
CREATE TABLE IF NOT EXISTS TAIPGMTAPE (
    tape_code VARCHAR(50) PRIMARY KEY,
    tape_name VARCHAR(200),
    product_code VARCHAR(50),
    production_status VARCHAR(20) DEFAULT 'ready', -- 'ready', 'in_production', 'archived'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_code) REFERENCES TAIGOODS(product_code)
);

-- 3. 방송 데이터 테이블
CREATE TABLE IF NOT EXISTS TAIBROADCASTS (
    id SERIAL PRIMARY KEY,
    tape_code VARCHAR(50),
    broadcast_start_timestamp TIMESTAMP,  -- 방송 시작 시간
    product_is_new BOOLEAN,              -- 신상품 여부 (True: 첫 방송, False: 재방송)
    gross_profit DECIMAL(15,2),          -- 매출 총이익
    sales_efficiency DECIMAL(10,2),      -- 매출 효율 (분당 매출 총이익)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tape_code) REFERENCES TAIPGMTAPE(tape_code)
);

-- 4. 공휴일 정보 테이블
CREATE TABLE IF NOT EXISTS TAIHOLIDAYS (
    holiday_date DATE PRIMARY KEY,
    holiday_name VARCHAR(100),
    holiday_type VARCHAR(50), -- 법정공휴일, 대체공휴일, 임시공휴일 등
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 경쟁사 방송 정보 테이블
CREATE TABLE IF NOT EXISTS TAICOMPETITOR_BROADCASTS (
    id SERIAL PRIMARY KEY,
    broadcast_date DATE,
    time_slot VARCHAR(20),
    competitor_name VARCHAR(100),
    category_main VARCHAR(100),
    category_middle VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. 모의 날씨 데이터 삽입
INSERT INTO taiweather_daily (weather_date, weather, temperature, precipitation) VALUES
('2025-08-18', 'Clear', 35.5, 0.0),
('2025-08-17', 'Clear', 32.0, 0.0),
('2025-08-16', 'Clouds', 28.5, 2.5),
('2025-08-15', 'Rain', 25.0, 15.8),
('2025-08-14', 'Clear', 30.2, 0.0)
ON CONFLICT (weather_date) DO NOTHING;

-- 5. 모의 상품 데이터 삽입
INSERT INTO TAIGOODS (product_code, product_name, category_main, category_middle, category_sub, price, brand, product_type) VALUES
('P001', '프리미엄 다이어트 보조제', '건강식품', '영양보조식품', '다이어트', 89000, '헬씨라이프', '유형'),
('P002', '홈트레이닝 세트', '운동용품', '헬스용품', '홈트레이닝', 150000, '피트니스코리아', '유형'),
('P003', '비타민C 1000mg', '건강식품', '영양보조식품', '비타민', 45000, '데일리뉴트리', '유형'),
('P004', '프리미엄 스킨케어 세트', '화장품', '기초화장품', '스킨케어', 120000, '뷰티랩', '유형'),
('P005', '무선 선풍기', '가전제품', '생활가전', '선풍기', 78000, '쿨윈드', '유형'),
('P006', '쿨매트 침대용', '생활용품', '침구류', '매트', 65000, '슬립웰', '유형'),
('P007', '프리미엄 에어프라이어', '가전제품', '주방가전', '에어프라이어', 180000, '이지쿡', '유형'),
('P008', '여름 원피스', '의류', '여성의류', '원피스', 85000, '썸머패션', '유형'),
('P009', '무선 이어폰', '전자제품', '음향기기', '이어폰', 95000, '사운드마스터', '유형'),
('P010', '마사지 건', '건강용품', '마사지용품', '마사지기', 135000, '릴렉스프로', '유형')
ON CONFLICT (product_code) DO NOTHING;

-- 5-1. 모의 방송테이프 데이터 삽입 (일부 상품만 테이프 제작 완료)
INSERT INTO TAIPGMTAPE (tape_code, tape_name, product_code, production_status) VALUES
('T001', '프리미엄 다이어트 보조제 방송테이프', 'P001', 'ready'),
('T002', '홈트레이닝 세트 완전정복', 'P002', 'ready'),
('T003', '비타민C 건강 특집', 'P003', 'ready'),
('T004', '스킨케어 뷰티 솔루션', 'P004', 'ready'),
('T005', '시원한 여름나기 선풍기', 'P005', 'ready'),
('T007', '에어프라이어 요리천국', 'P007', 'ready'),
('T009', '무선 이어폰 음악세상', 'P009', 'ready')
ON CONFLICT (tape_code) DO NOTHING;

-- 6. 모의 공휴일 데이터 삽입
INSERT INTO TAIHOLIDAYS (holiday_date, holiday_name, holiday_type) VALUES
('2025-01-01', '신정', '법정공휴일'),
('2025-02-09', '설날 연휴', '법정공휴일'),
('2025-02-10', '설날', '법정공휴일'),
('2025-02-11', '설날 연휴', '법정공휴일'),
('2025-03-01', '삼일절', '법정공휴일'),
('2025-05-05', '어린이날', '법정공휴일'),
('2025-05-06', '대체공휴일', '대체공휴일'),
('2025-06-06', '현충일', '법정공휴일'),
('2025-08-15', '광복절', '법정공휴일'),
('2025-09-28', '추석 연휴', '법정공휴일'),
('2025-09-29', '추석', '법정공휴일'),
('2025-09-30', '추석 연휴', '법정공휴일'),
('2025-10-03', '개천절', '법정공휴일'),
('2025-10-09', '한글날', '법정공휴일'),
('2025-12-25', '성탄절', '법정공휴일')
ON CONFLICT (holiday_date) DO NOTHING;

-- 7. 모의 경쟁사 방송 데이터 삽입
INSERT INTO TAICOMPETITOR_BROADCASTS (broadcast_date, time_slot, competitor_name, category_main, category_middle) VALUES
('2025-08-18', '20:00-22:00', 'A쇼핑', '건강식품', '영양보조식품'),
('2025-08-18', '20:00-22:00', 'B몰', '화장품', '기초화장품'),
('2025-08-18', '22:00-24:00', 'C샵', '가전제품', '생활가전'),
('2025-08-17', '20:00-22:00', 'A쇼핑', '건강식품', '영양보조식품'),
('2025-08-17', '18:00-20:00', 'D마트', '의류', '여성의류'),
('2025-08-16', '20:00-22:00', 'B몰', '가전제품', '생활가전'),
('2025-08-16', '22:00-24:00', 'C샵', '생활용품', '침구류'),
('2025-08-15', '20:00-22:00', 'A쇼핑', '가전제품', '주방가전'),
('2025-08-15', '18:00-20:00', 'D마트', '의류', '여성의류'),
('2025-08-14', '20:00-22:00', 'B몰', '전자제품', '음향기기');

-- 8. 모의 방송 데이터 삽입
-- P006, P008, P010은 방송테이프가 없으므로 방송 이력에서 제외
INSERT INTO TAIBROADCASTS (tape_code, broadcast_start_timestamp, product_is_new, gross_profit, sales_efficiency) VALUES
('T001', '2025-08-18 20:00:00', FALSE, 12000000, 100000.00),
('T002', '2025-08-18 22:00:00', TRUE, 6400000, 71111.11),
('T003', '2025-08-17 20:00:00', FALSE, 9600000, 128000.00),
('T004', '2025-08-17 18:00:00', TRUE, 16000000, 190476.19),
('T005', '2025-08-16 20:00:00', FALSE, 4800000, 120000.00),
('T007', '2025-08-15 20:00:00', TRUE, 14400000, 180000.00),
('T009', '2025-08-14 20:00:00', FALSE, 5600000, 140000.00);
-- ('2025-08-14', '22:00-24:00', 'T010', 11000000, 21000); -- T010은 없음

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_TAIBROADCASTS_timestamp ON TAIBROADCASTS(broadcast_start_timestamp);
CREATE INDEX IF NOT EXISTS idx_TAIGOODS_category_main ON TAIGOODS(category_main);
CREATE INDEX IF NOT EXISTS idx_TAIHOLIDAYS_date ON TAIHOLIDAYS(holiday_date);
CREATE INDEX IF NOT EXISTS idx_TAICOMPETITOR_BROADCASTS_date_slot ON TAICOMPETITOR_BROADCASTS(broadcast_date, time_slot);

-- 9. 트렌드 데이터 테이블 (n8n이 채워줄 데이터)
CREATE TABLE IF NOT EXISTS TAITRENDS (
    id SERIAL PRIMARY KEY,
    trend_date DATE NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    source VARCHAR(50) NOT NULL, -- 'NAVER', 'GOOGLE'
    score DECIMAL(8,5) DEFAULT 0,
    category VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trend_date, keyword, source)
);

-- 인덱스 생성 (트렌드 테이블용)
CREATE INDEX IF NOT EXISTS idx_TAITRENDS_date ON TAITRENDS(trend_date);
CREATE INDEX IF NOT EXISTS idx_TAITRENDS_score ON TAITRENDS(score DESC);
CREATE INDEX IF NOT EXISTS idx_TAITRENDS_category ON TAITRENDS(category);

-- 10. 모의 트렌드 데이터 삽입
INSERT INTO TAITRENDS (trend_date, keyword, source, score, category) VALUES
('2025-09-09', '다이어트', 'NAVER', 95, '건강식품'),
('2025-09-09', '캠핑', 'GOOGLE', 88, '레저/스포츠'),
('2025-09-09', '가을 옷', 'NAVER', 92, '의류'),
('2025-09-09', '스킨케어', 'NAVER', 90, '화장품'),
('2025-09-09', '홈트레이닝', 'GOOGLE', 87, '운동용품'),
('2025-09-08', '비타민', 'GOOGLE', 85, '건강식품'),
('2025-09-08', '에어프라이어', 'NAVER', 80, '가전제품'),
('2025-09-08', '선풍기', 'NAVER', 78, '가전제품'),
('2025-09-08', '원피스', 'GOOGLE', 82, '의류'),
('2025-09-07', '마사지', 'NAVER', 75, '건강용품')
ON CONFLICT (trend_date, keyword, source) DO NOTHING;
-- ============================================
-- 테이블: broadcast_training_dataset
-- 설명: XGBoost 매출 예측 모델 학습용 데이터셋
--       과거 방송 데이터를 기반으로 매출총이익을 예측하기 위한 피처와 타겟 변수를 저장
-- 작성일: 2025-09-30
-- ============================================

CREATE TABLE broadcast_training_dataset (
    -- ===== Primary Key =====
    id SERIAL PRIMARY KEY,
    
    -- ===== 시간 정보 =====
    broadcast_date DATE NOT NULL,
    hour INTEGER NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,
    time_slot VARCHAR(20),
    season VARCHAR(10),
    is_weekend BOOLEAN,
    
    -- ===== 상품 정보 =====
    product_code VARCHAR(50) NOT NULL,
    product_name VARCHAR(200),
    category_main VARCHAR(50) NOT NULL,
    category_middle VARCHAR(50),
    category_sub VARCHAR(50),
    price NUMERIC(12, 2),
    brand VARCHAR(100),
    product_type VARCHAR(10),
    product_is_new BOOLEAN,
    
    -- ===== 날씨 정보 =====
    weather VARCHAR(20),
    temperature FLOAT,
    precipitation FLOAT,
    
    -- ===== 외부 요인 =====
    is_holiday BOOLEAN NOT NULL,
    holiday_name VARCHAR(100),
    
    -- ===== 타겟 변수 =====
    gross_profit NUMERIC(15, 2) NOT NULL,
    sales_efficiency NUMERIC(10, 2),
    
    -- ===== 메타 정보 =====
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- ===== 제약 조건 =====
    CONSTRAINT check_hour CHECK (hour >= 0 AND hour < 24),
    CONSTRAINT check_price CHECK (price >= 0),
    CONSTRAINT check_gross_profit CHECK (gross_profit >= 0)
);

-- ============================================
-- 테이블 및 컬럼 코멘트
-- ============================================
COMMENT ON TABLE broadcast_training_dataset IS 'XGBoost 매출 예측 모델 학습용 데이터셋';

-- Primary Key
COMMENT ON COLUMN broadcast_training_dataset.id IS '자동 증가 Primary Key';

-- 시간 정보
COMMENT ON COLUMN broadcast_training_dataset.broadcast_date IS '방송 날짜';
COMMENT ON COLUMN broadcast_training_dataset.hour IS '방송 시작 시간 (0-23)';
COMMENT ON COLUMN broadcast_training_dataset.day_of_week IS '요일 (월/화/수/목/금/토/일)';
COMMENT ON COLUMN broadcast_training_dataset.time_slot IS '시간대 구분 (새벽/오전/오후/저녁/심야)';
COMMENT ON COLUMN broadcast_training_dataset.season IS '계절 (봄/여름/가을/겨울)';
COMMENT ON COLUMN broadcast_training_dataset.is_weekend IS '주말 여부';

-- 상품 정보
COMMENT ON COLUMN broadcast_training_dataset.product_code IS '상품 고유 코드';
COMMENT ON COLUMN broadcast_training_dataset.product_name IS '상품명';
COMMENT ON COLUMN broadcast_training_dataset.category_main IS '상품 대분류';
COMMENT ON COLUMN broadcast_training_dataset.category_middle IS '상품 중분류';
COMMENT ON COLUMN broadcast_training_dataset.category_sub IS '상품 소분류';
COMMENT ON COLUMN broadcast_training_dataset.price IS '상품 판매 가격 (원)';
COMMENT ON COLUMN broadcast_training_dataset.brand IS '브랜드명. 브랜드 파워에 따른 매출 차이 반영';
COMMENT ON COLUMN broadcast_training_dataset.product_type IS '상품 유형 (유형: 실물 상품, 무형: 상품권/여행권 등)';
COMMENT ON COLUMN broadcast_training_dataset.product_is_new IS '신상품 여부 (True: 첫 방송, False: 재방송)';

-- 날씨 정보
COMMENT ON COLUMN broadcast_training_dataset.weather IS '날씨 상태 (맑음/흐림/비/눈)';
COMMENT ON COLUMN broadcast_training_dataset.temperature IS '기온 (℃)';
COMMENT ON COLUMN broadcast_training_dataset.precipitation IS '강수량 (mm)';

-- 외부 요인
COMMENT ON COLUMN broadcast_training_dataset.is_holiday IS '공휴일 여부';

-- 타겟 변수
COMMENT ON COLUMN broadcast_training_dataset.gross_profit IS '매출총이익 (원). 예측 목표';
COMMENT ON COLUMN broadcast_training_dataset.sales_efficiency IS '매출효율 (원/분). 매출총이익 / 방송시간';

-- 메타 정보
COMMENT ON COLUMN broadcast_training_dataset.created_at IS '레코드 생성 시각';
COMMENT ON COLUMN broadcast_training_dataset.updated_at IS '레코드 최종 수정 시각';

-- ============================================
-- 인덱스 생성
-- ============================================
CREATE INDEX idx_broadcast_date ON broadcast_training_dataset(broadcast_date);
COMMENT ON INDEX idx_broadcast_date IS '날짜 기준 조회 성능 향상 (시계열 분석, 기간별 집계)';

CREATE INDEX idx_category_main ON broadcast_training_dataset(category_main);
COMMENT ON INDEX idx_category_main IS '대분류 카테고리 기준 조회 성능 향상 (카테고리별 매출 분석)';

CREATE INDEX idx_product_code ON broadcast_training_dataset(product_code);
COMMENT ON INDEX idx_product_code IS '상품 코드 기준 조회 성능 향상 (상품별 과거 성과 조회)';

CREATE INDEX idx_date_hour ON broadcast_training_dataset(broadcast_date, hour);
COMMENT ON INDEX idx_date_hour IS '날짜+시간 복합 인덱스. 특정 시간대의 과거 데이터 조회 성능 향상';

CREATE INDEX idx_brand ON broadcast_training_dataset(brand);
COMMENT ON INDEX idx_brand IS '브랜드별 성과 분석 및 조회 성능 향상';

CREATE INDEX idx_product_type ON broadcast_training_dataset(product_type);
COMMENT ON INDEX idx_product_type IS '유형/무형 상품별 매출 패턴 분석 성능 향상';

-- ============================================
-- 제약 조건 설명
-- ============================================
-- check_hour: 시간은 0~23 범위 내여야 함
-- check_price: 가격은 0 이상이어야 함
-- check_duration: 방송 시간은 양수여야 함
-- check_gross_profit: 매출총이익은 0 이상이어야 함