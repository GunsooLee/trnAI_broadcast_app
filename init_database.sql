-- 홈쇼핑 방송 추천 시스템 데이터베이스 초기화


-- 1. 날씨 데이터 테이블
CREATE TABLE IF NOT EXISTS taiweather_daily (
    weather_date TIMESTAMP PRIMARY KEY,
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
    embedded_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2-1. 방송테이프 정보 테이블 (TAIPGMTAPE)
CREATE TABLE IF NOT EXISTS TAIPGMTAPE (
    tape_code VARCHAR(50) PRIMARY KEY,
    tape_name VARCHAR(200),
    duration_minutes INTEGER,
    product_code VARCHAR(50),
    production_status VARCHAR(20) DEFAULT 'ready', -- 'ready', 'in_production', 'archived'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_code) REFERENCES TAIGOODS(product_code)
);

-- 3. 방송 데이터 테이블
CREATE TABLE IF NOT EXISTS TAIBROADCASTS (
    id SERIAL PRIMARY KEY,
    broadcast_date DATE,
    time_slot VARCHAR(20),
    tape_code VARCHAR(50),
    sales_amount DECIMAL(15,2),
    viewer_count INTEGER,
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
('2025-08-18', '폭염', 35.5, 0.0),
('2025-08-17', '맑음', 32.0, 0.0),
('2025-08-16', '구름많음', 28.5, 2.5),
('2025-08-15', '비', 25.0, 15.8),
('2025-08-14', '맑음', 30.2, 0.0)
ON CONFLICT (weather_date) DO NOTHING;

-- 5. 모의 상품 데이터 삽입
INSERT INTO TAIGOODS (product_code, product_name, category_main, category_middle, category_sub,  price) VALUES
('P001', '프리미엄 다이어트 보조제', '건강식품', '영양보조식품', '다이어트', 89000),
('P002', '홈트레이닝 세트', '운동용품', '헬스용품', '홈트레이닝', 150000),
('P003', '비타민C 1000mg', '건강식품', '영양보조식품', '비타민', 45000),
('P004', '프리미엄 스킨케어 세트', '화장품', '기초화장품', '스킨케어', 120000),
('P005', '무선 선풍기', '가전제품', '생활가전', '선풍기', 78000),
('P006', '쿨매트 침대용', '생활용품', '침구류', '매트', 65000),
('P007', '프리미엄 에어프라이어', '가전제품', '주방가전', '에어프라이어', 180000),
('P008', '여름 원피스', '의류', '여성의류', '원피스', 85000),
('P009', '무선 이어폰', '전자제품', '음향기기', '이어폰', 95000),
('P010', '마사지 건', '건강용품', '마사지용품', '마사지기', 135000)
ON CONFLICT (product_code) DO NOTHING;

-- 5-1. 모의 방송테이프 데이터 삽입 (일부 상품만 테이프 제작 완료)
INSERT INTO TAIPGMTAPE (tape_code, tape_name, duration_minutes, product_code, production_status) VALUES
('T001', '프리미엄 다이어트 보조제 방송테이프', 30, 'P001', 'ready'),
('T002', '홈트레이닝 세트 완전정복', 45, 'P002', 'ready'),
('T003', '비타민C 건강 특집', 25, 'P003', 'ready'),
('T004', '스킨케어 뷰티 솔루션', 35, 'P004', 'ready'),
('T005', '시원한 여름나기 선풍기', 20, 'P005', 'ready'),
('T007', '에어프라이어 요리천국', 40, 'P007', 'ready'),
('T009', '무선 이어폰 음악세상', 25, 'P009', 'ready')
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
INSERT INTO TAIBROADCASTS (broadcast_date, time_slot, tape_code, sales_amount, viewer_count) VALUES
('2025-08-18', '20:00-22:00', 'T001', 15000000, 25000),
('2025-08-18', '22:00-24:00', 'T002', 8000000, 18000),
('2025-08-17', '20:00-22:00', 'T003', 12000000, 22000),
('2025-08-17', '18:00-20:00', 'T004', 20000000, 30000),
('2025-08-16', '20:00-22:00', 'T005', 6000000, 15000),
-- ('2025-08-16', '22:00-24:00', 'T006', 4000000, 12000), -- T006은 없음
('2025-08-15', '20:00-22:00', 'T007', 18000000, 28000),
-- ('2025-08-15', '18:00-20:00', 'T008', 9000000, 20000), -- T008은 없음
('2025-08-14', '20:00-22:00', 'T009', 7000000, 16000);
-- ('2025-08-14', '22:00-24:00', 'T010', 11000000, 21000); -- T010은 없음

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_TAIBROADCASTS_date ON TAIBROADCASTS(broadcast_date);
CREATE INDEX IF NOT EXISTS idx_TAIBROADCASTS_timeslot ON TAIBROADCASTS(time_slot);
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

-- XGBoost 모델 학습용 방송 데이터셋 테이블
CREATE TABLE IF NOT EXISTS broadcast_training_dataset (
    id SERIAL PRIMARY KEY,
    broadcast_date DATE NOT NULL,
    time_slot VARCHAR(20) NOT NULL,
    product_code VARCHAR(50) NOT NULL,
    product_name VARCHAR(200),
    category_main VARCHAR(100),
    category_middle VARCHAR(100),
    category_sub VARCHAR(100),
    price DECIMAL(10,2),
    sales_amount DECIMAL(15,2) NOT NULL,
    viewer_count INTEGER,
    weather VARCHAR(50),
    temperature DECIMAL(5,2),
    precipitation DECIMAL(5,2),
    is_holiday BOOLEAN DEFAULT FALSE,
    competitor_count INTEGER DEFAULT 0,
    season VARCHAR(10),
    day_of_week INTEGER,
    hour INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 통합 시장 데이터 테이블 (n8n 크롤링용 - 홈쇼핑 랭킹 + 검색 트렌드)
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    data_type VARCHAR(50) NOT NULL,  -- 'homeshopping_ranking' 또는 'search_trend'
    source VARCHAR(100) NOT NULL,    -- 'CJ온스타일', 'GS샵', 'naver_shopping', 'google_trends' 등
    
    -- 홈쇼핑 랭킹 데이터
    product_name TEXT,
    rank_position INTEGER,
    price INTEGER,
    discount_rate DECIMAL(5,2),
    
    -- 검색 트렌드 데이터  
    keyword VARCHAR(200),
    trend_score INTEGER,  -- 1-100 점수
    search_volume BIGINT,
    
    -- 공통 필드
    category VARCHAR(100),
    related_keywords TEXT[],  -- 연관 키워드 배열
    metadata JSONB,  -- 추가 메타데이터
    original_url TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 중복 방지를 위한 유니크 제약
    UNIQUE(data_type, source, COALESCE(product_name, keyword), collected_at)
);

-- 인덱스 생성 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_broadcast_training_date ON broadcast_training_dataset(broadcast_time);
CREATE INDEX IF NOT EXISTS idx_broadcast_training_product ON broadcast_training_dataset(product_code);
CREATE INDEX IF NOT EXISTS idx_broadcast_training_category ON broadcast_training_dataset(category_main);

-- 시장 데이터 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_market_data_collected_at ON market_data(collected_at);
CREATE INDEX IF NOT EXISTS idx_market_data_type_source ON market_data(data_type, source);
CREATE INDEX IF NOT EXISTS idx_market_data_trend_score ON market_data(trend_score DESC) WHERE data_type = 'search_trend';
CREATE INDEX IF NOT EXISTS idx_market_data_rank ON market_data(rank_position) WHERE data_type = 'homeshopping_ranking';

-- 테이블 코멘트
COMMENT ON TABLE broadcast_training_dataset IS 'XGBoost 모델 학습을 위한 방송 매출 데이터셋';
COMMENT ON TABLE market_data IS 'n8n에서 크롤링한 통합 시장 데이터 (홈쇼핑 랭킹 + 검색 트렌드)';