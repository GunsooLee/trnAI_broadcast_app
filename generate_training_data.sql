-- ============================================
-- XGBoost 학습 데이터 생성 쿼리
-- 설명: 여러 테이블을 JOIN하여 broadcast_training_dataset에 학습용 데이터 삽입
-- 작성일: 2025-09-30
-- 수정일: 2025-10-16 (중복 방지 로직 추가)
-- ============================================

-- 기존 데이터와 중복되지 않도록 임시 테이블 사용
CREATE TEMP TABLE temp_training_data AS
SELECT 
    -- 시간 정보
    b.broadcast_start_timestamp::DATE as broadcast_date,
    EXTRACT(HOUR FROM b.broadcast_start_timestamp)::INTEGER as hour,
    CASE EXTRACT(DOW FROM b.broadcast_start_timestamp)
        WHEN 0 THEN '일'
        WHEN 1 THEN '월'
        WHEN 2 THEN '화'
        WHEN 3 THEN '수'
        WHEN 4 THEN '목'
        WHEN 5 THEN '금'
        WHEN 6 THEN '토'
    END as day_of_week,
    CASE 
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 0 AND 5 THEN '새벽'
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 6 AND 11 THEN '오전'
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 12 AND 17 THEN '오후'
        WHEN EXTRACT(HOUR FROM b.broadcast_start_timestamp) BETWEEN 18 AND 23 THEN '저녁'
    END as time_slot,
    CASE 
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (3, 4, 5) THEN '봄'
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (6, 7, 8) THEN '여름'
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (9, 10, 11) THEN '가을'
        WHEN EXTRACT(MONTH FROM b.broadcast_start_timestamp) IN (12, 1, 2) THEN '겨울'
    END as season,
    EXTRACT(DOW FROM b.broadcast_start_timestamp) IN (0, 6) as is_weekend,
    
    -- 상품 정보
    g.product_code,
    g.product_name,
    g.category_main,
    g.category_middle,
    g.category_sub,
    g.price,
    g.brand,
    g.product_type,
    b.product_is_new,
    
    -- 방송테이프 정보
    t.tape_code,
    t.tape_name,
    
    -- 날씨 정보
    w.weather,
    w.temperature,
    w.precipitation,
    
    -- 외부 요인
    CASE 
        WHEN h.holiday_date IS NOT NULL THEN TRUE 
        ELSE FALSE 
    END as is_holiday,
    h.holiday_name,
    
    -- 타겟 변수
    b.gross_profit,
    b.sales_efficiency

FROM TAIBROADCASTS b
INNER JOIN TAIPGMTAPE t ON b.tape_code = t.tape_code
INNER JOIN TAIGOODS g ON t.product_code = g.product_code
LEFT JOIN taiweather_daily w 
    ON b.broadcast_start_timestamp::DATE = w.weather_date
LEFT JOIN TAIHOLIDAYS h 
    ON b.broadcast_start_timestamp::DATE = h.holiday_date

WHERE b.broadcast_start_timestamp IS NOT NULL
  AND b.gross_profit IS NOT NULL
  AND b.product_is_new IS NOT NULL
  AND g.product_type IS NOT NULL;

-- 중복되지 않는 데이터만 삽입
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
    
    -- 방송테이프 정보
    tape_code,
    tape_name,
    
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
    broadcast_date,
    hour,
    day_of_week,
    time_slot,
    season,
    is_weekend,
    product_code,
    product_name,
    category_main,
    category_middle,
    category_sub,
    price,
    brand,
    product_type,
    product_is_new,
    tape_code,
    tape_name,
    weather,
    temperature,
    precipitation,
    is_holiday,
    holiday_name,
    gross_profit,
    sales_efficiency
FROM temp_training_data t
WHERE NOT EXISTS (
    SELECT 1 
    FROM broadcast_training_dataset b
    WHERE b.tape_code = t.tape_code
      AND b.broadcast_date = t.broadcast_date
      AND b.hour = t.hour
);

-- 삽입된 레코드 수 반환
SELECT COUNT(*) as inserted_count FROM temp_training_data t
WHERE NOT EXISTS (
    SELECT 1 
    FROM broadcast_training_dataset b
    WHERE b.tape_code = t.tape_code
      AND b.broadcast_date = t.broadcast_date
      AND b.hour = t.hour
);