-- 홈쇼핑 방송 추천 시스템 데이터베이스 초기화

-- 1. 날씨 데이터 테이블
CREATE TABLE IF NOT EXISTS weather_daily (
    weather_date DATE PRIMARY KEY,
    weather VARCHAR(50),
    temperature DECIMAL(5,2),
    precipitation DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 상품 정보 테이블
CREATE TABLE IF NOT EXISTS products (
    product_code VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(200),
    category_main VARCHAR(100),
    category_middle VARCHAR(100),
    category_sub VARCHAR(100),
    search_keywords VARCHAR(500),
    price DECIMAL(10,2),
    embedded_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 방송 데이터 테이블
CREATE TABLE IF NOT EXISTS broadcasts (
    id SERIAL PRIMARY KEY,
    broadcast_date DATE,
    time_slot VARCHAR(20),
    product_code VARCHAR(50),
    sales_amount DECIMAL(15,2),
    viewer_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_code) REFERENCES products(product_code)
);

-- 4. 공휴일 정보 테이블
CREATE TABLE IF NOT EXISTS holidays (
    holiday_date DATE PRIMARY KEY,
    holiday_name VARCHAR(100),
    holiday_type VARCHAR(50), -- 법정공휴일, 대체공휴일, 임시공휴일 등
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 경쟁사 방송 정보 테이블
CREATE TABLE IF NOT EXISTS competitor_broadcasts (
    id SERIAL PRIMARY KEY,
    broadcast_date DATE,
    time_slot VARCHAR(20),
    competitor_name VARCHAR(100),
    category_main VARCHAR(100),
    category_middle VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. 모의 날씨 데이터 삽입
INSERT INTO weather_daily (weather_date, weather, temperature, precipitation) VALUES
('2025-08-18', '폭염', 35.5, 0.0),
('2025-08-17', '맑음', 32.0, 0.0),
('2025-08-16', '구름많음', 28.5, 2.5),
('2025-08-15', '비', 25.0, 15.8),
('2025-08-14', '맑음', 30.2, 0.0)
ON CONFLICT (weather_date) DO NOTHING;

-- 5. 모의 상품 데이터 삽입
INSERT INTO products (product_code, product_name, category_main, category_middle, category_sub, search_keywords, price) VALUES
('P001', '프리미엄 다이어트 보조제', '건강식품', '영양보조식품', '다이어트', '프리미엄 다이어트 보조제 건강식품 영양보조식품 다이어트 체중감량 건강', 89000),
('P002', '홈트레이닝 세트', '운동용품', '헬스용품', '홈트레이닝', '홈트레이닝 세트 운동용품 헬스용품 홈트레이닝 운동 헬스', 150000),
('P003', '비타민C 1000mg', '건강식품', '영양보조식품', '비타민', '비타민C 1000mg 건강식품 영양보조식품 비타민 면역력 건강', 45000),
('P004', '프리미엄 스킨케어 세트', '화장품', '기초화장품', '스킨케어', '프리미엄 스킨케어 세트 화장품 기초화장품 스킨케어 미백 보습', 120000),
('P005', '무선 선풍기', '가전제품', '생활가전', '선풍기', '무선 선풍기 가전제품 생활가전 선풍기 시원함 여름', 78000),
('P006', '쿨매트 침대용', '생활용품', '침구류', '매트', '쿨매트 침대용 생활용품 침구류 매트 시원함 수면', 65000),
('P007', '프리미엄 에어프라이어', '가전제품', '주방가전', '에어프라이어', '프리미엄 에어프라이어 가전제품 주방가전 에어프라이어 요리 간편', 180000),
('P008', '여름 원피스', '의류', '여성의류', '원피스', '여름 원피스 의류 여성의류 원피스 여름 패션', 85000),
('P009', '무선 이어폰', '전자제품', '음향기기', '이어폰', '무선 이어폰 전자제품 음향기기 이어폰 무선 음악', 95000),
('P010', '마사지 건', '건강용품', '마사지용품', '마사지기', '마사지 건 건강용품 마사지용품 마사지기 마사지 근육 릴렉스', 135000)
ON CONFLICT (product_code) DO NOTHING;

-- 6. 모의 공휴일 데이터 삽입
INSERT INTO holidays (holiday_date, holiday_name, holiday_type) VALUES
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
INSERT INTO competitor_broadcasts (broadcast_date, time_slot, competitor_name, category_main, category_middle) VALUES
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
INSERT INTO broadcasts (broadcast_date, time_slot, product_code, sales_amount, viewer_count) VALUES
('2025-08-18', '20:00-22:00', 'P001', 15000000, 25000),
('2025-08-18', '22:00-24:00', 'P002', 8000000, 18000),
('2025-08-17', '20:00-22:00', 'P003', 12000000, 22000),
('2025-08-17', '18:00-20:00', 'P004', 20000000, 30000),
('2025-08-16', '20:00-22:00', 'P005', 6000000, 15000),
('2025-08-16', '22:00-24:00', 'P006', 4000000, 12000),
('2025-08-15', '20:00-22:00', 'P007', 18000000, 28000),
('2025-08-15', '18:00-20:00', 'P008', 9000000, 20000),
('2025-08-14', '20:00-22:00', 'P009', 7000000, 16000),
('2025-08-14', '22:00-24:00', 'P010', 11000000, 21000);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_broadcasts_date ON broadcasts(broadcast_date);
CREATE INDEX IF NOT EXISTS idx_broadcasts_timeslot ON broadcasts(time_slot);
CREATE INDEX IF NOT EXISTS idx_products_category_main ON products(category_main);
CREATE INDEX IF NOT EXISTS idx_holidays_date ON holidays(holiday_date);
CREATE INDEX IF NOT EXISTS idx_competitor_broadcasts_date_slot ON competitor_broadcasts(broadcast_date, time_slot);
