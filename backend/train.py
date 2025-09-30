import os
import sys
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# FastAPI 앱과 동일한 위치의 tokenizer_utils를 참조할 수 있도록 경로 추가
# 이렇게 하면 app.tokenizer_utils 형태가 아닌 tokenizer_utils로 바로 임포트 가능
sys.path.append(str(Path(__file__).parent / 'app'))
from tokenizer_utils import mecab_tokenizer

# --- 상수 정의 ---
MODEL_FILE_PROFIT = "xgb_broadcast_profit.joblib"
MODEL_FILE_EFFICIENCY = "xgb_broadcast_efficiency.joblib"
TABLE_NAME = "broadcast_training_dataset"

# --- DB 및 환경변수 설정 ---
def get_db_engine():
    """새로운 DB 엔진을 생성하여 반환합니다."""
    load_dotenv()
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise ValueError("DB_URI 환경변수가 설정되지 않았습니다.")
    return create_engine(db_uri)

# --- 데이터 로딩 및 전처리 ---
def _weekday_kr(date: dt.date) -> str:
    return ["월", "화", "수", "목", "금", "토", "일"][date.weekday()]

def _time_slot(hour: int) -> str:
    if 6 <= hour < 12:
        return "오전"
    elif 12 <= hour < 18:
        return "오후"
    elif 18 <= hour < 24:
        return "저녁"
    else:
        return "심야"

def load_data(engine: Engine) -> pd.DataFrame:
    """DB에서 학습용 데이터를 로드하고 기본 전처리를 수행합니다."""
    print("데이터 로딩 시작...")

    # broadcast_training_dataset에서 직접 데이터 로드
    query = f"""
    WITH base AS (
        SELECT
            product_code,
            category_main as product_lgroup,
            category_middle as product_mgroup,
            category_sub as product_sgroup,
            product_name,
            brand,
            product_type,
            time_slot,
            day_of_week,
            season,
            is_weekend,
            hour,
            weather,
            temperature,
            precipitation,
            is_holiday,
            holiday_name,
            gross_profit,
            sales_efficiency,
            price as product_price,
            broadcast_date
        FROM broadcast_training_dataset
        WHERE gross_profit IS NOT NULL
    ),
    product_stats AS (
        SELECT
            product_code,
            AVG(gross_profit) AS product_avg_profit,
            COUNT(*) AS product_broadcast_count
        FROM base
        GROUP BY product_code
    ),
    category_timeslot_stats AS (
        SELECT
            product_mgroup,
            time_slot,
            AVG(gross_profit) AS category_timeslot_avg_profit
        FROM base
        GROUP BY product_mgroup, time_slot
    )
    SELECT
        b.*,
        ps.product_avg_profit,
        ps.product_broadcast_count,
        cts.category_timeslot_avg_profit
    FROM base b
    LEFT JOIN product_stats ps ON b.product_code = ps.product_code
    LEFT JOIN category_timeslot_stats cts ON b.product_mgroup = cts.product_mgroup AND b.time_slot = cts.time_slot
    """

    df = pd.read_sql(query, engine)
    
    # NULL 값 처리
    df['product_avg_profit'] = df['product_avg_profit'].fillna(0)
    df['category_timeslot_avg_profit'] = df['category_timeslot_avg_profit'].fillna(0)
    df['product_broadcast_count'] = df['product_broadcast_count'].fillna(0)
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    df['precipitation'] = df['precipitation'].fillna(0)
    df['weather'] = df['weather'].fillna('Clear')
    df['brand'] = df['brand'].fillna('Unknown')
    df['product_type'] = df['product_type'].fillna('유형')
    df['holiday_name'] = df['holiday_name'].fillna('')
    
    print(f"데이터 로딩 완료. 총 {len(df)}개 행")
    return df

# --- 모델 파이프라인 빌드 ---
def build_pipeline() -> Pipeline:
    """Scikit-learn 파이프라인을 구축합니다."""
    print("모델 파이프라인 생성...")
    numeric_features = [
        "product_price",
        "product_avg_profit",
        "product_broadcast_count",
        "category_timeslot_avg_profit",
        "hour",
        "temperature",
        "precipitation",
    ]
    categorical_features = [
        "product_lgroup",
        "product_mgroup",
        "product_sgroup",
        "brand",
        "product_type",
        "time_slot",
        "day_of_week",
        "season",
        "weather",
    ]
    boolean_features = ["is_weekend", "is_holiday"]

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("bool", "passthrough", boolean_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    return Pipeline([("pre", preprocessor), ("model", model)])

# --- 모델 학습 실행 ---
def train() -> None:
    """전체 모델 학습 파이프라인을 실행합니다. (2개 타겟: gross_profit, sales_efficiency)"""
    engine = get_db_engine()
    df = load_data(engine)

    # 공통 제거 컬럼 (타겟 변수 제외)
    common_drop_cols = [
        "product_code",
        "product_name",
        "holiday_name",  # is_holiday로 충분
        "broadcast_date",  # 날짜는 피처로 사용하지 않음 (day_of_week, season으로 대체)
    ]
    
    # ========================================
    # 모델 1: gross_profit 예측 모델
    # ========================================
    print("\n" + "="*60)
    print("모델 1: 매출총이익(gross_profit) 예측 모델 학습")
    print("="*60)
    
    target1 = "gross_profit"
    drop_cols1 = common_drop_cols + ["sales_efficiency", target1]
    existing_drop_cols1 = [col for col in drop_cols1 if col in df.columns]
    
    X1 = df.drop(columns=existing_drop_cols1)
    y1 = df[target1]

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=42
    )

    print("모델 학습 시작...")
    pipe1 = build_pipeline()
    pipe1.fit(X1_train, y1_train)
    print("모델 학습 완료.")

    y1_pred = pipe1.predict(X1_test)
    print("\n=== 모델 1 평가 (gross_profit) ===")
    print(f"MAE : {mean_absolute_error(y1_test, y1_pred):,.2f} 원")
    print(f"RMSE: {np.sqrt(mean_squared_error(y1_test, y1_pred)):,.2f} 원")
    print(f"R2  : {r2_score(y1_test, y1_pred):.4f}\n")

    # 모델 1 저장
    model_path1 = Path(__file__).parent / 'app' / MODEL_FILE_PROFIT
    joblib.dump(pipe1, model_path1)
    print(f"✅ 모델 1이 '{model_path1}'에 저장되었습니다.")

    # ========================================
    # 모델 2: sales_efficiency 예측 모델
    # ========================================
    print("\n" + "="*60)
    print("모델 2: 매출효율(sales_efficiency) 예측 모델 학습")
    print("="*60)
    
    target2 = "sales_efficiency"
    
    # sales_efficiency가 NULL인 행 제거
    df_efficiency = df[df[target2].notna()].copy()
    print(f"sales_efficiency 유효 데이터: {len(df_efficiency)}개 행")
    
    drop_cols2 = common_drop_cols + ["gross_profit", target2]
    existing_drop_cols2 = [col for col in drop_cols2 if col in df_efficiency.columns]
    
    X2 = df_efficiency.drop(columns=existing_drop_cols2)
    y2 = df_efficiency[target2]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=42
    )

    print("모델 학습 시작...")
    pipe2 = build_pipeline()
    pipe2.fit(X2_train, y2_train)
    print("모델 학습 완료.")

    y2_pred = pipe2.predict(X2_test)
    print("\n=== 모델 2 평가 (sales_efficiency) ===")
    print(f"MAE : {mean_absolute_error(y2_test, y2_pred):,.2f} 원/분")
    print(f"RMSE: {np.sqrt(mean_squared_error(y2_test, y2_pred)):,.2f} 원/분")
    print(f"R2  : {r2_score(y2_test, y2_pred):.4f}\n")

    # 모델 2 저장
    model_path2 = Path(__file__).parent / 'app' / MODEL_FILE_EFFICIENCY
    joblib.dump(pipe2, model_path2)
    print(f"✅ 모델 2가 '{model_path2}'에 저장되었습니다.")
    
    print("\n" + "="*60)
    print("🎉 전체 모델 학습 완료!")
    print("="*60)

if __name__ == "__main__":
    train()
