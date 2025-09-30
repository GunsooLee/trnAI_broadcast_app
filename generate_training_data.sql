-- ============================================
-- XGBoost 학습 데이터 생성 쿼리
-- 설명: 여러 테이블을 JOIN하여 broadcast_training_dataset에 학습용 데이터 삽입
-- 작성일: 2025-09-30
-- ============================================

-- broadcast_training_dataset 테이블에 학습 데이터 삽입
INSERT INTO broadcast_training_dataset (
    -- 시간 정보
    broadcast_date,
    hour,
    day_of_week,
    time_slot,
    season,
    is_weekend,
    
    -- 상품 정보
    product_code,
    product_name,
    category_main,
    category_middle,
    category_sub,
    price,
    brand,
    product_type,
    product_is_new,
    
    -- 날씨 정보
    weather,
    temperature,
    precipitation,
    
    -- 외부 요인
    is_holiday,
    holiday_name,
    
    -- 타겟 변수
    gross_profit,
    sales_efficiency
)
SELECT 
    -- ===== 시간 정보 =====
    -- broadcast_date: TIMESTAMP에서 DATE 추출
    b.broadcast_start_timestamp::DATE as broadcast_date,
    
    -- hour: 방송 시작 시간 (0-23)
    EXTRACT(HOUR FROM b.broadcast_start_timestamp)::INTEGER as hour,
    
    -- day_of_week: 요일 (월/화/수/목/금/토/일)
    CASE EXTRACT(DOW FROM b.broadcast_start_timestamp)
        WHEN 0 THEN '일'
        WHEN 1 THEN '월'
        WHEN 2 THEN '화'
        WHEN 3 THEN '수'
        WHEN 4 THEN '목'
        WHEN 5 THEN '금'
        WHEN 6 THEN '토'
    END as day_of_week,
    
    -- time_slot: 시간대 구분 (새벽/오전/오후/저녁/심야)
    CASE 
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 0 AND 5 THEN '새벽'
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 6 AND 11 THEN '오전'
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 12 AND 17 THEN '오후'
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 18 AND 23 THEN '저녁'
    END as time_slot,
    
    -- season: 계절 (봄/여름/가을/겨울)
    CASE 
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (3, 4, 5) THEN '봄'
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (6, 7, 8) THEN '여름'
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (9, 10, 11) THEN '가을'
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (12, 1, 2) THEN '겨울'
    END as season,
    
    -- is_weekend: 주말 여부
    EXTRACT(DOW FROM b.broadcast_start_timestamp) IN (0, 6) as is_weekend,
    
    -- ===== 상품 정보 =====
    g.product_code,
    g.product_name,
    g.category_main,
    g.category_middle,
    g.category_sub,
    g.price,
    g.brand,
    g.product_type,
    b.product_is_new,
    
    -- ===== 날씨 정보 =====
    w.weather,
    w.temperature,
    w.precipitation,
    
    -- ===== 외부 요인 =====
    -- is_holiday: 공휴일 여부
    CASE 
        WHEN h.holiday_date IS NOT NULL THEN TRUE 
        ELSE FALSE 
    END as is_holiday,
    
    -- holiday_name: 공휴일명
    h.holiday_name,
    
    -- ===== 타겟 변수 =====
    b.gross_profit,
    b.sales_efficiency

FROM TAIBROADCASTS b
-- 방송테이프 JOIN
INNER JOIN TAIPGMTAPE t ON b.tape_code = t.tape_code
-- 상품 정보 JOIN
INNER JOIN TAIGOODS g ON t.product_code = g.product_code
-- 날씨 정보 JOIN (LEFT: 날씨 데이터 없을 수도 있음)
LEFT JOIN taiweather_daily w 
    ON b.broadcast_start_timestamp::DATE = w.weather_date
-- 공휴일 정보 JOIN (LEFT: 평일이 대부분)
LEFT JOIN TAIHOLIDAYS h 
    ON b.broadcast_start_timestamp::DATE = h.holiday_date

WHERE b.broadcast_start_timestamp IS NOT NULL
  AND b.gross_profit IS NOT NULL
  AND b.product_is_new IS NOT NULL  -- product_is_new 필수
  AND g.product_type IS NOT NULL  -- product_type 필수
ORDER BY b.broadcast_start_timestamp;