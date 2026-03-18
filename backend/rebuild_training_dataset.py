#!/usr/bin/env python3
"""
broadcast_training_dataset 테이블 재생성
TAIBROADCASTS에서 직접 데이터 가져와서 정확한 학습 데이터 생성
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

print("=" * 100)
print("🔄 broadcast_training_dataset 재생성")
print("=" * 100)
print()

# PostgreSQL 연결
engine = create_engine('postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB')

# 1. 기존 테이블 백업
print("1️⃣ 기존 테이블 백업 중...")
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS broadcast_training_dataset_backup"))
    conn.execute(text("CREATE TABLE broadcast_training_dataset_backup AS SELECT * FROM broadcast_training_dataset"))
    conn.commit()
print("   ✅ 백업 완료")
print()

# 2. 기존 테이블 삭제
print("2️⃣ 기존 테이블 삭제 중...")
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS broadcast_training_dataset CASCADE"))
    conn.commit()
print("   ✅ 삭제 완료")
print()

# 3. 새 테이블 생성 (quantity_sold를 INTEGER로)
print("3️⃣ 새 테이블 생성 중...")
create_table_query = """
CREATE TABLE broadcast_training_dataset (
    id SERIAL PRIMARY KEY,
    broadcast_date DATE NOT NULL,
    hour INTEGER NOT NULL CHECK (hour >= 0 AND hour < 24),
    minute INTEGER DEFAULT 0,
    day_of_week VARCHAR(10) NOT NULL,
    time_slot VARCHAR(20),
    season VARCHAR(10),
    is_weekend BOOLEAN,
    product_code VARCHAR(50) NOT NULL,
    product_name VARCHAR(200),
    category_main VARCHAR(50) NOT NULL,
    category_middle VARCHAR(50),
    category_sub VARCHAR(50),
    price NUMERIC(12,2) CHECK (price >= 0),
    brand VARCHAR(100),
    product_type VARCHAR(10),
    product_is_new BOOLEAN,
    weather VARCHAR(20),
    temperature DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    is_holiday BOOLEAN NOT NULL,
    holiday_name VARCHAR(100),
    tape_code VARCHAR(50),
    tape_name VARCHAR(200),
    duration_minutes INTEGER DEFAULT 60,
    quantity_sold INTEGER DEFAULT 0,  -- INTEGER로 변경!
    gross_profit NUMERIC(15,2) NOT NULL CHECK (gross_profit >= 0),
    sales_efficiency NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_broadcast_date ON broadcast_training_dataset(broadcast_date);
CREATE INDEX idx_product_code ON broadcast_training_dataset(product_code);
CREATE INDEX idx_category_main ON broadcast_training_dataset(category_main);
CREATE INDEX idx_date_hour ON broadcast_training_dataset(broadcast_date, hour);
CREATE INDEX idx_brand ON broadcast_training_dataset(brand);
CREATE INDEX idx_product_type ON broadcast_training_dataset(product_type);
"""

with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()
print("   ✅ 테이블 생성 완료")
print()

# 4. TAIBROADCASTS에서 데이터 가져오기
print("4️⃣ TAIBROADCASTS에서 데이터 로딩 중...")
query = """
SELECT 
    tb.broadcast_start_timestamp,
    tb.broadcast_end_timestamp,
    tb.duration_minutes,
    tb.product_is_new,
    ROUND(tb.quantity_sold)::INTEGER as quantity_sold,  -- 반올림 후 INTEGER 변환
    tb.gross_profit,
    tb.sales_efficiency,
    tt.tape_code,
    tt.tape_name,
    tt.product_code,
    tg.product_name,
    tg.category_main,
    tg.category_middle,
    tg.category_sub,
    tg.price,
    tg.brand,
    tg.product_type
FROM taibroadcasts tb
JOIN taipgmtape tt ON tb.tape_code = tt.tape_code
JOIN taigoods tg ON tt.product_code = tg.product_code
WHERE tb.gross_profit > 0
  AND tb.broadcast_start_timestamp >= '2022-01-01'
ORDER BY tb.broadcast_start_timestamp
"""

df = pd.read_sql(query, engine)
print(f"   ✅ {len(df)}개 레코드 로딩 완료")
print()

# 5. 데이터 가공
print("5️⃣ 데이터 가공 중...")

# 날짜/시간 파생 변수
df['broadcast_date'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.date
df['hour'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.hour
df['minute'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.minute
df['day_of_week'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.day_name()

# 시간대 계산
def get_time_slot(hour):
    if 6 <= hour < 9:
        return "06:00-09:00"
    elif 9 <= hour < 12:
        return "09:00-12:00"
    elif 12 <= hour < 15:
        return "12:00-15:00"
    elif 15 <= hour < 18:
        return "15:00-18:00"
    elif 18 <= hour < 21:
        return "18:00-21:00"
    elif 21 <= hour < 24:
        return "21:00-24:00"
    else:
        return "00:00-06:00"

df['time_slot'] = df['hour'].apply(get_time_slot)

# 계절 계산
def get_season(date):
    month = pd.to_datetime(date).month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

df['season'] = df['broadcast_date'].apply(get_season)

# 주말 여부
df['is_weekend'] = pd.to_datetime(df['broadcast_date']).dt.dayofweek >= 5

# 공휴일 정보 (간단히 처리)
df['is_holiday'] = False
df['holiday_name'] = ''

# 날씨 정보 (임시로 NULL)
df['weather'] = None
df['temperature'] = None
df['precipitation'] = None

print("   ✅ 데이터 가공 완료")
print()

# 6. 데이터 통계
print("6️⃣ 데이터 통계:")
print(f"   총 레코드: {len(df):,}개")
print(f"   기간: {df['broadcast_date'].min()} ~ {df['broadcast_date'].max()}")
print(f"   quantity_sold 범위: {df['quantity_sold'].min()} ~ {df['quantity_sold'].max()}")
print(f"   quantity_sold 평균: {df['quantity_sold'].mean():.2f}")
print(f"   quantity_sold 타입: {df['quantity_sold'].dtype}")
print(f"   gross_profit 범위: {df['gross_profit'].min():,.0f} ~ {df['gross_profit'].max():,.0f}")
print(f"   gross_profit 평균: {df['gross_profit'].mean():,.0f}")
print()

# 7. PostgreSQL에 삽입
print("7️⃣ PostgreSQL에 데이터 삽입 중...")

# 필요한 컬럼만 선택
columns = [
    'broadcast_date', 'hour', 'minute', 'day_of_week', 'time_slot', 'season', 'is_weekend',
    'product_code', 'product_name', 'category_main', 'category_middle', 'category_sub',
    'price', 'brand', 'product_type', 'product_is_new',
    'weather', 'temperature', 'precipitation', 'is_holiday', 'holiday_name',
    'tape_code', 'tape_name', 'duration_minutes',
    'quantity_sold', 'gross_profit', 'sales_efficiency'
]

df_insert = df[columns].copy()

# NULL 처리
df_insert['product_is_new'] = df_insert['product_is_new'].fillna(False)
df_insert['duration_minutes'] = df_insert['duration_minutes'].fillna(60)

df_insert.to_sql(
    'broadcast_training_dataset',
    engine,
    if_exists='append',
    index=False,
    method='multi',
    chunksize=1000
)

print(f"   ✅ {len(df_insert)}개 레코드 삽입 완료")
print()

# 8. 검증
print("8️⃣ 데이터 검증:")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total,
            MIN(quantity_sold) as min_qty,
            MAX(quantity_sold) as max_qty,
            AVG(quantity_sold) as avg_qty,
            MIN(gross_profit) as min_profit,
            MAX(gross_profit) as max_profit,
            AVG(gross_profit) as avg_profit
        FROM broadcast_training_dataset
    """))
    row = result.fetchone()
    
    print(f"   총 레코드: {row[0]:,}개")
    print(f"   quantity_sold: {row[1]} ~ {row[2]} (평균: {row[3]:.2f})")
    print(f"   gross_profit: {row[4]:,.0f} ~ {row[5]:,.0f} (평균: {row[6]:,.0f})")

print()
print("=" * 100)
print("✅ broadcast_training_dataset 재생성 완료!")
print("=" * 100)
print()
print("다음 단계:")
print("  1. 모델 재학습: docker exec fastapi_backend python /app/train.py")
print("  2. API 재시작: docker restart fastapi_backend")
print()
