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
MODEL_FILE = "xgb_broadcast_sales.joblib"
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
            '' as product_dgroup,
            '' as product_type,
            product_name,
            '' as tape_code,
            '' as tape_name,
            '' as keyword,
            time_slot,
            sales_amount,
            0 as order_count,
            price as product_price,
            broadcast_date
        FROM broadcast_training_dataset
        WHERE sales_amount IS NOT NULL
    ),
    product_stats AS (
        SELECT
            product_code,
            AVG(sales_amount) AS product_avg_sales,
            COUNT(*) AS product_broadcast_count
        FROM base
        GROUP BY product_code
    ),
    category_timeslot_stats AS (
        SELECT
            product_mgroup,
            time_slot,
            AVG(sales_amount) AS category_timeslot_avg_sales
        FROM base
        GROUP BY product_mgroup, time_slot
    )
    SELECT
        b.*,
        ps.product_avg_sales,
        ps.product_broadcast_count,
        cts.category_timeslot_avg_sales,
        COALESCE(b.competitor_count, 0) as competitor_count_same_category,
        CASE WHEN b.is_holiday THEN 1 ELSE 0 END as is_holiday
    FROM base b
    LEFT JOIN product_stats ps ON b.product_code = ps.product_code
    LEFT JOIN category_timeslot_stats cts ON b.product_mgroup = cts.product_mgroup AND b.time_slot = cts.time_slot
    """

    df = pd.read_sql(query, engine)
    df.fillna(0, inplace=True) # 통계값 없는 경우 0으로 채움
    print(f"데이터 로딩 완료. 총 {len(df)}개 행")
    return df

# --- 모델 파이프라인 빌드 ---
def build_pipeline() -> Pipeline:
    """Scikit-learn 파이프라인을 구축합니다."""
    print("모델 파이프라인 생성...")
    numeric_features = [
        "product_price",
        "product_avg_sales",
        "product_broadcast_count",
        "category_timeslot_avg_sales",
        "competitor_count_same_category",
        "is_holiday"
    ]
    categorical_features = ["product_lgroup", "product_mgroup", "time_slot"]
    # text_features 제거 - 메인 API에서 실제로 사용되지 않음

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
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
    """전체 모델 학습 파이프라인을 실행합니다."""
    engine = get_db_engine()
    df = load_data(engine)

    target = "sales_amount"
    drop_cols = [
        "product_sgroup",
        "product_dgroup",
        "product_type",
        "order_count",
        "broadcast_date",  # 날짜는 피처로 사용하지 않음
        target,
    ]
    X = df.drop(columns=drop_cols)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("모델 학습 시작...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    print("모델 학습 완료.")

    y_pred = pipe.predict(X_test)
    print("\n=== 모델 평가 ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2  : {r2_score(y_test, y_pred):.2f}\n")

    # 모델을 app 폴더 내에 저장합니다.
    model_path = Path(__file__).parent / 'app' / MODEL_FILE
    joblib.dump(pipe, model_path)
    print(f"모델이 '{model_path}'에 저장되었습니다.")

if __name__ == "__main__":
    train()
