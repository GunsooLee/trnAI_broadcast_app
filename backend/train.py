import os
import sys
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer

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
          AND gross_profit <= 50000000  -- 이상치 제거: 5천만원 이하만
          AND gross_profit >= 0         -- 마이너스 마진 방지, 하위 이상치 필터 제거
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
    ),
    category_overall_stats AS (
        SELECT
            product_mgroup,
            AVG(gross_profit) AS category_overall_avg_profit
        FROM base
        GROUP BY product_mgroup
    )
    SELECT 
        b.*,
        ps.product_avg_profit,
        ps.product_broadcast_count,
        cts.category_timeslot_avg_profit,
        COALESCE(cts.category_timeslot_avg_profit / NULLIF(co.category_overall_avg_profit, 0), 1) AS timeslot_specialty_score,
        b.time_slot || '_' || b.product_mgroup AS time_category_interaction
    FROM base b
    LEFT JOIN product_stats ps ON b.product_code = ps.product_code
    LEFT JOIN category_timeslot_stats cts ON b.product_mgroup = cts.product_mgroup AND b.time_slot = cts.time_slot
    LEFT JOIN category_overall_stats co ON b.product_mgroup = co.product_mgroup
    """

    df = pd.read_sql(query, engine)
    
    # NULL 값 처리
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    df['precipitation'] = df['precipitation'].fillna(0)
    df['weather'] = df['weather'].fillna('Clear')
    df['brand'] = df['brand'].fillna('Unknown')
    df['product_type'] = df['product_type'].fillna('유형')
    df['holiday_name'] = df['holiday_name'].fillna('')
    df['category_timeslot_avg_profit'] = df['category_timeslot_avg_profit'].fillna(0)
    df['product_avg_profit'] = df['product_avg_profit'].fillna(0)
    df['product_broadcast_count'] = df['product_broadcast_count'].fillna(0)
    
    # 월 컬럼 추출 (broadcast_date에서)
    df['month'] = pd.to_datetime(df['broadcast_date']).dt.month
    
    # 계절성-카테고리 특화 피처 추가 (C)
    print("계절성-카테고리 특화 피처 추가...")
    
    # 월 기반 부드러운 계절 전환 가중치
    def get_season_weights(month):
        """월별로 계절 가중치를 부드럽게 전환"""
        if month == 2:  # 2월: 겨울 70% + 봄 30%
            return {"겨울": 0.7, "봄": 0.3}
        elif month == 3:  # 3월: 겨울 30% + 봄 70%  
            return {"겨울": 0.3, "봄": 0.7}
        elif month == 5:  # 5월: 봄 70% + 여름 30%
            return {"봄": 0.7, "여름": 0.3}
        elif month == 8:  # 8월: 여름 70% + 가을 30%
            return {"여름": 0.7, "가을": 0.3}
        elif month == 9:  # 9월: 여름 30% + 가을 70%
            return {"여름": 0.3, "가을": 0.7}
        elif month == 11:  # 11월: 가을 70% + 겨울 30%
            return {"가을": 0.7, "겨울": 0.3}
        else:  # 명확한 계절
            seasons = {12: "겨울", 1: "겨울", 4: "봄", 6: "여름", 7: "여름", 10: "가을"}
            return {seasons.get(month, "봄"): 1.0}
    
    # 계절별 카테고리 평균 매출 계산
    season_category_stats = df.groupby(['season', 'product_mgroup'])['gross_profit'].mean().reset_index()
    season_category_stats.columns = ['season', 'product_mgroup', 'season_category_avg_profit']
    
    # 전체 카테고리 평균 매출 계산
    overall_category_stats = df.groupby('product_mgroup')['gross_profit'].mean().reset_index()
    overall_category_stats.columns = ['product_mgroup', 'overall_category_avg_profit']
    
    # 데이터프레임에 병합
    df = pd.merge(df, season_category_stats, on=['season', 'product_mgroup'], how='left')
    df = pd.merge(df, overall_category_stats, on='product_mgroup', how='left')
    
    # 계절-카테고리 특화 점수 계산
    df['season_category_specialty_score'] = (
        df['season_category_avg_profit'] / df['overall_category_avg_profit']
    ).fillna(1.0)
    
    # 월 기반 계절 가중치 적용하여 경계기 보정
    season_weights = df['month'].apply(lambda x: get_season_weights(x))
    
    # 각 계절 가중치를 별도 컬럼으로 추가
    df['spring_weight'] = season_weights.apply(lambda x: x.get('봄', 0.0))
    df['summer_weight'] = season_weights.apply(lambda x: x.get('여름', 0.0))
    df['autumn_weight'] = season_weights.apply(lambda x: x.get('가을', 0.0))
    df['winter_weight'] = season_weights.apply(lambda x: x.get('겨울', 0.0))
    
    # 계절-카테고리 상호작용 피처 생성
    df['season_category_interaction'] = df['season'] + '_' + df['product_mgroup']
    
    print(f"  계절-카테고리 조합: {len(df['season_category_interaction'].unique())}개")
    print(f"  계절 특화 점수 범위: {df['season_category_specialty_score'].min():.2f} ~ {df['season_category_specialty_score'].max():.2f}")
    print(f"  경계기 월 가중치 예시: 2월(겨울:{df[df['month']==2]['winter_weight'].iloc[0]:.1f}, 봄:{df[df['month']==2]['spring_weight'].iloc[0]:.1f})")
    
    # 과대예측 방지: 가격 로그 스케일링 (현재 미사용)
    print("가격 피처 로그 스케일링 적용...")
    df['product_price_log'] = np.log1p(df['product_price'])
    print(f"  product_price: 원본 평균 {df['product_price'].mean():,.0f}원 → 로그 {df['product_price_log'].mean():.2f}")
    
    # 시간대 피처 강화: 사인/코사인 변환 (주기성 반영)
    print("시간대 피처 강화 (사인/코사인 변환)...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print(f"  hour_sin 범위: [{df['hour_sin'].min():.2f}, {df['hour_sin'].max():.2f}]")
    print(f"  hour_cos 범위: [{df['hour_cos'].min():.2f}, {df['hour_cos'].max():.2f}]")
    
    # 월(month) 피처 추가: 11월에 팔린 상품은 10~12월에 추천되도록
    print("월(month) 피처 추가 (사인/코사인 변환)...")
    df['broadcast_date'] = pd.to_datetime(df['broadcast_date'])
    df['month'] = df['broadcast_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print(f"  month 분포: {df['month'].value_counts().sort_index().to_dict()}")
    print(f"  month_sin 범위: [{df['month_sin'].min():.2f}, {df['month_sin'].max():.2f}]")
    print(f"  month_cos 범위: [{df['month_cos'].min():.2f}, {df['month_cos'].max():.2f}]")
    
    print(f"데이터 로딩 완료. 총 {len(df)}개 행")
    return df

# --- 모델 파이프라인 빌드 ---
def build_pipeline() -> Pipeline:
    """Scikit-learn 파이프라인을 구축합니다."""
    print("모델 파이프라인 생성...")
    
    # 시간대/월 피처 강화, 날씨/가격 피처 제거 (2024-12-15 수정)
    numeric_features = [
        "product_price_log",            # 가격 (로그 스케일링)
        "product_avg_profit",           # 상품의 과거 평균 매출
        "product_broadcast_count",      # 상품의 과거 총 방송 횟수
        # 시간 피처 (9시에 팔린 상품 → 8~10시에 추천)
        "hour",                 # 시간 (0-23)
        "hour_sin",             # 시간 사인 변환 (주기성 반영)
        "hour_cos",             # 시간 코사인 변환 (주기성 반영)
        # 월 피처 (11월에 팔린 상품 → 10~12월에 추천)
        "month",                # 월 (1-12)
        "month_sin",            # 월 사인 변환 (주기성 반영)
        "month_cos",            # 월 코사인 변환 (주기성 반영)
        "category_timeslot_avg_profit",  # 해당 카테고리의 시간대별 평균 매출
        "timeslot_specialty_score",      # 해당 카테고리의 특정 시간대 특화 점수 (시간대평균 / 전체평균)
        # 계절성-카테고리 특화 피처 (C)
        "season_category_specialty_score", # 계절별 카테고리 특화 점수
        "spring_weight",        # 봄 계절 가중치 (경계기 보정)
        "summer_weight",        # 여름 계절 가중치 (경계기 보정)
        "autumn_weight",        # 가을 계절 가중치 (경계기 보정)
        "winter_weight",        # 겨울 계절 가중치 (경계기 보정)
        # "temperature",        # 제거: 날씨 영향 적음
        # "precipitation",      # 제거: 날씨 영향 적음
    ]
    categorical_features = [
        "product_lgroup",
        "product_mgroup",
        "product_sgroup",
        "brand",
        "product_type",
        "time_slot",            # 시간대 (오전/오후/저녁/심야) - 핵심 피처
        "day_of_week",          # 요일 - 핵심 피처
        "time_category_interaction", # 시간대와 카테고리 상호작용 피처
        "season_category_interaction", # 계절과 카테고리 상호작용 피처 (C)
        # "season",             # 제거: month 피처로 대체
        # "weather",            # 제거: 날씨 영향 적음
    ]
    boolean_features = ["is_weekend", "is_holiday"]  # 주말/공휴일 - 핵심 피처

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("bool", "passthrough", boolean_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    # XGBoost + LightGBM 스태킹 앙상블 모델
    base_models = [
        ('xgb', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
        )),
        ('lgb', LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1  # 경고 메시지 제거
        ))
    ]
    
    # 수정 1: 메타 모델을 가장 단순한 릿지 선형 회귀로 변경 (과적합 방지)
    final_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
    
    # 스태킹 앙상블 모델 생성 (TimeSeriesSplit은 StackingRegressor와 호환되지 않음)
    model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model,
        cv=5,  # 5-fold 교차검증 (데이터 시간순 정렬로 누수 최소화)
        passthrough=False  # 원본 특성을 메타 모델에 직접 전달 안 함
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

    # 시계열 데이터 누수 방지: 시간순으로 데이터 정렬
    print("시간순 데이터 정렬 (데이터 누수 방지)...")
    sort_indices = np.argsort(df['broadcast_date'].values)
    X1_sorted = X1.iloc[sort_indices].reset_index(drop=True)
    y1_log_sorted = y1_log.iloc[sort_indices].reset_index(drop=True)
    y1_orig_sorted = y1.iloc[sort_indices].reset_index(drop=True)
    print(f"  데이터 정렬 완료: {len(X1_sorted)}개 샘플")

    # 학습/테스트 분할 (시간순 유지, shuffle=False)
    X1_train, X1_test, y1_train_log, y1_test_log = train_test_split(
        X1_sorted, y1_log_sorted, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 원본 y 값도 분리 (평가용)
    _, _, y1_train_orig, y1_test_orig = train_test_split(
        X1_sorted, y1_orig_sorted, test_size=0.2, random_state=42, shuffle=False
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
