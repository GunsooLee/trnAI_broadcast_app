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
    db_uri = os.getenv("DB_URI") or os.getenv("POSTGRES_URI")
    
    # Docker 컨테이너 내부에서는 호스트명이 다를 수 있음
    if not db_uri:
        db_uri = "postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB"
    
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

    # broadcast_training_dataset에서 직접 데이터 로드 (단순화)
    query = f"""
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
      AND gross_profit <= 50000000  -- 이상치 제거: 5천만원 이하만
      AND gross_profit >= 1000000   -- 하위 이상치 제거: 100만원 이상만
    """

    df = pd.read_sql(query, engine)
    
    # NULL 값 처리
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    df['precipitation'] = df['precipitation'].fillna(0)
    df['weather'] = df['weather'].fillna('Clear')
    df['brand'] = df['brand'].fillna('Unknown')
    df['product_type'] = df['product_type'].fillna('유형')
    df['holiday_name'] = df['holiday_name'].fillna('')
    
    # 과대예측 방지: 가격 로그 스케일링
    print("가격 피처 로그 스케일링 적용...")
    df['product_price_log'] = np.log1p(df['product_price'])
    print(f"  product_price: 원본 평균 {df['product_price'].mean():,.0f}원 → 로그 {df['product_price_log'].mean():.2f}")
    
    print(f"데이터 로딩 완료. 총 {len(df)}개 행")
    return df

# --- 모델 파이프라인 빌드 ---
def build_pipeline() -> Pipeline:
    """Scikit-learn 파이프라인을 구축합니다."""
    print("모델 파이프라인 생성...")
    numeric_features = [
        "product_broadcast_count",
        "category_timeslot_avg_profit_log",  # 로그 스케일링 버전 사용
        "product_price_log",  # 로그 스케일링 버전 사용
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
        max_depth=4,  # 6 → 4 (과적합 방지)
        min_child_weight=5,  # 1 → 5 (과적합 방지)
        gamma=0.2,  # 분할 최소 손실 감소
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,  # L1 정규화
        reg_lambda=2.0,  # L2 정규화 강화
        random_state=42,
    )

    return Pipeline([("pre", preprocessor), ("model", model)])

# --- 모델 학습 실행 ---
def train() -> dict:
    """전체 모델 학습 파이프라인을 실행합니다. (2개 타겟: gross_profit, sales_efficiency)
    
    Returns:
        dict: 학습 결과 통계
    """
    import time
    start_time = time.time()
    
    engine = get_db_engine()
    df = load_data(engine)
    
    training_stats = {
        "total_records": len(df),
        "models": {}
    }

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
    
    # 로그 변환 적용 (과대 예측 방지)
    print("타겟 변수 로그 변환 적용...")
    y1_log = np.log1p(y1)  # log(1 + y)
    print(f"  원본 평균: {y1.mean():,.0f}원, 로그 평균: {y1_log.mean():.2f}")

    X1_train, X1_test, y1_train_log, y1_test_log = train_test_split(
        X1, y1_log, test_size=0.2, random_state=42
    )
    
    # 원본 y 값도 분리 (평가용)
    _, _, y1_train_orig, y1_test_orig = train_test_split(
        X1, y1, test_size=0.2, random_state=42
    )

    print("모델 학습 시작...")
    pipe1 = build_pipeline()
    pipe1.fit(X1_train, y1_train_log)
    print("모델 학습 완료.")

    # 예측 후 역변환
    y1_pred_log = pipe1.predict(X1_test)
    y1_pred = np.expm1(y1_pred_log)  # exp(y) - 1
    
    mae1 = mean_absolute_error(y1_test_orig, y1_pred)
    rmse1 = np.sqrt(mean_squared_error(y1_test_orig, y1_pred))
    r2_1 = r2_score(y1_test_orig, y1_pred)
    
    print("\n=== 모델 1 평가 (gross_profit) ===")
    print(f"MAE : {mae1:,.2f} 원")
    print(f"RMSE: {rmse1:,.2f} 원")
    print(f"R2  : {r2_1:.4f}\n")

    # 모델 1 저장
    model_path1 = Path(__file__).parent / 'app' / MODEL_FILE_PROFIT
    joblib.dump(pipe1, model_path1)
    print(f"✅ 모델 1이 '{model_path1}'에 저장되었습니다.")
    
    # 통계 저장
    training_stats["models"]["profit_model"] = {
        "train_records": len(X1_train),
        "test_records": len(X1_test),
        "mae": round(mae1, 2),
        "rmse": round(rmse1, 2),
        "r2_score": round(r2_1, 4)
    }

    # ========================================
    # 모델 2: sales_efficiency 예측 모델 (사용 안 함 - 주석 처리)
    # ========================================
    # R2 Score가 0.38로 매우 낮고, 현재 API에서 사용하지 않음
    # 필요 시 주석 해제하여 재활성화 가능
    
    print("\n⏭️  모델 2 (sales_efficiency) 학습 스킵 (사용 안 함)")
    
    # 통계 저장 (빈 값)
    training_stats["models"]["efficiency_model"] = {
        "train_records": 0,
        "test_records": 0,
        "mae": 0,
        "rmse": 0,
        "r2_score": 0,
        "status": "skipped"
    }
    
    # # 아래 코드는 필요 시 주석 해제
    # print("\n" + "="*60)
    # print("모델 2: 매출효율(sales_efficiency) 예측 모델 학습")
    # print("="*60)
    # 
    # target2 = "sales_efficiency"
    # 
    # # sales_efficiency가 NULL인 행 제거
    # df_efficiency = df[df[target2].notna()].copy()
    # print(f"sales_efficiency 유효 데이터: {len(df_efficiency)}개 행")
    # 
    # drop_cols2 = common_drop_cols + ["gross_profit", target2]
    # existing_drop_cols2 = [col for col in drop_cols2 if col in df_efficiency.columns]
    # 
    # X2 = df_efficiency.drop(columns=existing_drop_cols2)
    # y2 = df_efficiency[target2]
    #
    # X2_train, X2_test, y2_train, y2_test = train_test_split(
    #     X2, y2, test_size=0.2, random_state=42
    # )
    #
    # print("모델 학습 시작...")
    # pipe2 = build_pipeline()
    # pipe2.fit(X2_train, y2_train)
    # print("모델 학습 완료.")
    #
    # y2_pred = pipe2.predict(X2_test)
    # mae2 = mean_absolute_error(y2_test, y2_pred)
    # rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
    # r2_2 = r2_score(y2_test, y2_pred)
    # 
    # print("\n=== 모델 2 평가 (sales_efficiency) ===")
    # print(f"MAE : {mae2:,.2f} 원/분")
    # print(f"RMSE: {rmse2:,.2f} 원/분")
    # print(f"R2  : {r2_2:.4f}\n")
    #
    # # 모델 2 저장
    # model_path2 = Path(__file__).parent / 'app' / MODEL_FILE_EFFICIENCY
    # joblib.dump(pipe2, model_path2)
    # print(f"✅ 모델 2가 '{model_path2}'에 저장되었습니다.")
    # 
    # # 통계 저장
    # training_stats["models"]["efficiency_model"] = {
    #     "train_records": len(X2_train),
    #     "test_records": len(X2_test),
    #     "mae": round(mae2, 2),
    #     "rmse": round(rmse2, 2),
    #     "r2_score": round(r2_2, 4)
    # }
    
    # 총 소요 시간
    elapsed_time = time.time() - start_time
    training_stats["training_time_seconds"] = round(elapsed_time, 2)
    
    print("\n" + "="*60)
    print("🎉 전체 모델 학습 완료!")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초")
    print("="*60)
    
    return training_stats

if __name__ == "__main__":
    train()
