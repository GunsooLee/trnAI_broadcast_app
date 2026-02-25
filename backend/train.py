import os
import sys
import re
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

    # broadcast_training_dataset에서 직접 데이터 로드 (데이터 누수 완전 방지)
    query = """
    SELECT 
        b.*,
        -- 1. 상품별 누적 평균 매출 및 횟수, 가격
        COALESCE(AVG(gross_profit) OVER (
            PARTITION BY product_code 
            ORDER BY broadcast_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS product_avg_profit,
        
        COALESCE(COUNT(*) OVER (
            PARTITION BY product_code 
            ORDER BY broadcast_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS product_broadcast_count,
        
        COALESCE(AVG(price) OVER (
            PARTITION BY product_code 
            ORDER BY broadcast_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS product_price,

        -- 2. 시간대-카테고리 누적 평균 매출
        COALESCE(AVG(gross_profit) OVER (
            PARTITION BY category_middle, time_slot
            ORDER BY broadcast_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS category_timeslot_avg_profit,

        -- 3. 카테고리 전체 누적 평균 매출
        COALESCE(AVG(gross_profit) OVER (
            PARTITION BY category_middle
            ORDER BY broadcast_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS category_overall_avg_profit,

        -- 4. 계절-카테고리 누적 평균 매출 (파이썬에서 SQL로 이동!)
        COALESCE(AVG(gross_profit) OVER (
            PARTITION BY season, category_middle
            ORDER BY broadcast_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS season_category_avg_profit

    FROM broadcast_training_dataset b
    WHERE b.gross_profit IS NOT NULL
    ORDER BY b.broadcast_date
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

    # [추가] 텍스트 벡터화 에러 방지용 상품명 결측치 처리
    df['product_name'] = df['product_name'].fillna('')
    
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
    
    # 계절별 카테고리 평균 매출은 SQL에서 이미 계산됨 (데이터 누수 방지 완료)
    print("계절성-카테고리 특화 피처 추가...")

    # 0으로 나누어 무한대(inf) 값이 발생하는 것을 막기 위한 안전한 분모 생성
    safe_denominator = df['category_overall_avg_profit'].replace(0, np.nan)
    
    # 계절-카테고리 특화 점수 계산 (안전한 분모 사용 및 SQL 컬럼 통일)
    df['season_category_specialty_score'] = (
        df['season_category_avg_profit'] / safe_denominator
    ).fillna(1.0)
    
    # 월 기반 계절 가중치 적용하여 경계기 보정
    season_weights = df['month'].apply(lambda x: get_season_weights(x))
    
    # 각 계절 가중치를 별도 컬럼으로 추가
    df['spring_weight'] = season_weights.apply(lambda x: x.get('봄', 0.0))
    df['summer_weight'] = season_weights.apply(lambda x: x.get('여름', 0.0))
    df['autumn_weight'] = season_weights.apply(lambda x: x.get('가을', 0.0))
    df['winter_weight'] = season_weights.apply(lambda x: x.get('겨울', 0.0))
    
    # 계절-카테고리 상호작용 피처 생성
    df['season_category_interaction'] = df['season'] + '_' + df['category_middle']
    
    # 시간대별 특화 점수 계산 (안전한 분모 사용)
    df['timeslot_specialty_score'] = (
        df['category_timeslot_avg_profit'] / safe_denominator
    ).fillna(1.0)
    
    # 시간대-카테고리 상호작용 피처
    df['time_category_interaction'] = df['time_slot'] + '_' + df['category_middle']
    
    print(f"  계절-카테고리 조합: {len(df['season_category_interaction'].unique())}개")
    print(f"  계절 특화 점수 범위: {df['season_category_specialty_score'].min():.2f} ~ {df['season_category_specialty_score'].max():.2f}")
    print(f"  경계기 월 가중치 예시: 2월(겨울:{df[df['month']==2]['winter_weight'].iloc[0]:.1f}, 봄:{df[df['month']==2]['spring_weight'].iloc[0]:.1f})")
    
    # 과대예측 방지: 가격 로그 스케일링 (현재 미사용)
    print("가격 피처 로그 스케일링 적용...")
    df['product_price_log'] = np.log1p(df['product_price'])
    print(f"  product_price: 원본 평균 {df['product_price'].mean():,.0f}원 → 로그 {df['product_price_log'].mean():.2f}")
    
    # 시간대 피처 강화: hour + minute 통합 및 사인/코사인 변환 (주기성 반영)
    print("시간대 피처 강화 (hour_with_minute 사인/코사인 변환)...")
    # minute이 없는 경우 0으로 처리
    df['minute'] = df['minute'].fillna(0)
    # hour와 minute를 결합 (예: 13시 30분 = 13.5)
    df['hour_with_minute'] = df['hour'] + df['minute'] / 60.0
    df['hour_with_minute_sin'] = np.sin(2 * np.pi * df['hour_with_minute'] / 24)
    df['hour_with_minute_cos'] = np.cos(2 * np.pi * df['hour_with_minute'] / 24)
    print(f"  hour_with_minute 범위: [{df['hour_with_minute'].min():.2f}, {df['hour_with_minute'].max():.2f}]")
    print(f"  hour_with_minute_sin 범위: [{df['hour_with_minute_sin'].min():.2f}, {df['hour_with_minute_sin'].max():.2f}]")
    print(f"  hour_with_minute_cos 범위: [{df['hour_with_minute_cos'].min():.2f}, {df['hour_with_minute_cos'].max():.2f}]")
    
    # 월(month) 피처 추가: 11월에 팔린 상품은 10~12월에 추천되도록
    print("월(month) 피처 추가 (사인/코사인 변환)...")
    df['broadcast_date'] = pd.to_datetime(df['broadcast_date'])
    df['month'] = df['broadcast_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print(f"  month 분포: {df['month'].value_counts().sort_index().to_dict()}")
    print(f"  month_sin 범위: [{df['month_sin'].min():.2f}, {df['month_sin'].max():.2f}]")
    print(f"  month_cos 범위: [{df['month_cos'].min():.2f}, {df['month_cos'].max():.2f}]")
    
    # 방송 길이(duration) 피처 추가
    print("방송 길이(duration) 피처 추가...")
    df['duration_minutes'] = df['duration_minutes'].fillna(60)  # 기본값 60분
    df['duration_log'] = np.log1p(df['duration_minutes'])  # 로그 스케일링
    print(f"  duration_minutes 범위: [{df['duration_minutes'].min():.0f}, {df['duration_minutes'].max():.0f}]분")
    print(f"  duration_minutes 평균: {df['duration_minutes'].mean():.0f}분")
    print(f"  duration_log 범위: [{df['duration_log'].min():.2f}, {df['duration_log'].max():.2f}]")
    
    # 키워드 피처 추가 (매출 영향력 분석 기반)
    print("상품명 키워드 피처 생성...")
    df = add_keyword_features(df)
    
    print(f"데이터 로딩 완료. 총 {len(df)}개 행")
    return df

def add_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    상품명에서 매출 영향력이 높은 키워드를 추출하여 이진 피처로 변환
    
    분석 결과 기반 키워드:
    - 매출 비율 1.1배 이상 & 등장 횟수 50회 이상
    - 특집방송(1.82x), 두유대장(1.74x), 하나투어(1.62x) 등
    """
    
    # 매출 영향력 높은 한글 키워드 (37개)
    high_impact_keywords = [
        '특집방송', '두유대장', '두유제조기', '에어텔닷컴', '하나투어', 
        '마데카', '릴렉스', '가을', '고객감사', '추석', 
        '에버홈', '단하루', '동국제약', '비에날씬', '노랑풍선', 
        '임성근의', '여행', '갈비', '국내산', '크림', 
        '슬림', '지아잔틴', '배한호', '토비콤', '르까프', 
        '콘드로이친', '티셔츠', '카무트', '팬츠', '효소', 
        '흑염소진액', '세일', '첫날부터', '다재다능', '흥국생명', 
        '루테인', '암보험'
    ]
    
    # 숫자 패턴 (9개)
    number_patterns = ['1+1', '2+1', '6+3', '10+2', '6+6', '5+5', '6+1', '12+6', '20+1']
    
    # 한글 키워드 피처 생성
    for keyword in high_impact_keywords:
        feature_name = f'kw_{keyword}'
        df[feature_name] = df['product_name'].str.contains(keyword, na=False).astype(int)
    
    # 숫자 패턴 피처 생성 (정규식 이스케이프 필요)
    for pattern in number_patterns:
        feature_name = f'kw_{pattern.replace("+", "_")}'
        df[feature_name] = df['product_name'].str.contains(re.escape(pattern), na=False).astype(int)
    
    # 키워드 통계 출력
    total_keywords = len(high_impact_keywords) + len(number_patterns)
    keyword_cols = [col for col in df.columns if col.startswith('kw_')]
    keyword_hit_rate = (df[keyword_cols].sum(axis=1) > 0).mean() * 100
    
    print(f"  생성된 키워드 피처: {total_keywords}개")
    print(f"  키워드 포함 상품 비율: {keyword_hit_rate:.1f}%")
    
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
        # 시간 피처 (분 단위까지 반영한 정밀한 시간)
        "hour_with_minute",         # 시간 (0.0-23.99, 예: 13.5 = 13시 30분)
        "hour_with_minute_sin",     # 시간 사인 변환 (주기성 반영)
        "hour_with_minute_cos",     # 시간 코사인 변환 (주기성 반영)
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
        # 방송 길이 피처
        "duration_minutes",     # 방송 길이 (분)
        "duration_log",         # 방송 길이 (로그 스케일링)
        # "temperature",        # 제거: 날씨 영향 적음
        # "precipitation",      # 제거: 날씨 영향 적음
    ]
    
    # 키워드 피처 동적 추가 (매출 영향력 분석 기반)
    keyword_features = [
        # 한글 키워드 (37개)
        'kw_특집방송', 'kw_두유대장', 'kw_두유제조기', 'kw_에어텔닷컴', 'kw_하나투어',
        'kw_마데카', 'kw_릴렉스', 'kw_가을', 'kw_고객감사', 'kw_추석',
        'kw_에버홈', 'kw_단하루', 'kw_동국제약', 'kw_비에날씬', 'kw_노랑풍선',
        'kw_임성근의', 'kw_여행', 'kw_갈비', 'kw_국내산', 'kw_크림',
        'kw_슬림', 'kw_지아잔틴', 'kw_배한호', 'kw_토비콤', 'kw_르까프',
        'kw_콘드로이친', 'kw_티셔츠', 'kw_카무트', 'kw_팬츠', 'kw_효소',
        'kw_흑염소진액', 'kw_세일', 'kw_첫날부터', 'kw_다재다능', 'kw_흥국생명',
        'kw_루테인', 'kw_암보험',
        # 숫자 패턴 (9개)
        'kw_1_1', 'kw_2_1', 'kw_6_3', 'kw_10_2', 'kw_6_6',
        'kw_5_5', 'kw_6_1', 'kw_12_6', 'kw_20_1'
    ]
    
    # 키워드 피처를 numeric_features에 추가
    numeric_features.extend(keyword_features)
    categorical_features = [
        "category_main", "category_middle", "category_sub",
        "brand", "product_type", "time_slot", "day_of_week",
        "time_category_interaction", "season_category_interaction",
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

    # XGBoost + LightGBM 스태킹 앙상블 모델 (Optuna 최적화 파라미터 적용)
    base_models = [
        ('xgb', XGBRegressor(
            n_estimators=800, 
            learning_rate=0.024053100186204397, 
            max_depth=6, 
            min_child_weight=7, 
            subsample=0.7800951459285155, 
            colsample_bytree=0.8040394068614868,
            random_state=42,
            n_jobs=-1
        )),
        ('lgb', LGBMRegressor(
            n_estimators=500, 
            learning_rate=0.04312096438928548, 
            max_depth=7, 
            num_leaves=60, 
            subsample=0.8949895531136589, 
            colsample_bytree=0.7528423385351647,
            random_state=42,
            verbose=-1,
            n_jobs=-1
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

# --- Optuna 하이퍼파라미터 최적화 ---
import optuna
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore') # 불필요한 경고 메시지 숨김

def run_optuna_tuning(X_train, y_train, preprocessor):
    """XGBoost와 LightGBM의 최적 파라미터를 각각 찾습니다."""
    print("\n🚀 Optuna 하이퍼파라미터 최적화 시작...")
    
    # 데이터 누수 방지를 위한 시계열 교차 검증 객체
    tscv = TimeSeriesSplit(n_splits=5)

    # ==========================================
    # 1. XGBoost 목적 함수
    # ==========================================
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        model = XGBRegressor(**params)
        # 전처리기와 모델을 파이프라인으로 연결하여 누수 원천 차단
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])
        
        # 교차 검증 점수 계산 (MSE의 음수값을 반환하므로, 이를 다시 양수로 바꿔서 최소화)
        score = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
        return -score.mean()

    # ==========================================
    # 2. LightGBM 목적 함수
    # ==========================================
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        model = LGBMRegressor(**params)
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])
        
        score = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
        return -score.mean()

    # ==========================================
    # 3. 최적화 실행 (시간 절약을 위해 각각 20번씩만 탐색)
    # ==========================================
    print("\n[1/2] XGBoost 최적화 진행 중...")
    xgb_study = optuna.create_study(direction='minimize')
    xgb_study.optimize(xgb_objective, n_trials=20)
    
    print("\n[2/2] LightGBM 최적화 진행 중...")
    lgb_study = optuna.create_study(direction='minimize')
    lgb_study.optimize(lgb_objective, n_trials=20)

    # 최종 결과 출력
    print("\n" + "="*60)
    print("🏆 최적화 완료! 아래 파라미터를 복사해서 build_pipeline의 base_models에 붙여넣으십시오.")
    print("="*60)
    print(f"✅ XGBoost Best Params:\n{xgb_study.best_params}\n")
    print(f"✅ LightGBM Best Params:\n{lgb_study.best_params}\n")
    print("="*60)

# --- 모델 학습 실행 ---
def train() -> dict:
    """전체 모델 학습 파이프라인을 실행합니다. (타겟: quantity_sold)
    
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
    # 주의: price는 판매 수량 예측에 필수 피처이므로 제거하지 않음
    common_drop_cols = [
        "product_code",
        "product_name",  # 롤백: 텍스트 벡터화 제거
        "holiday_name",  # is_holiday로 충분
        "broadcast_date",  # 날짜는 피처로 사용하지 않음 (day_of_week, season으로 대체)
        "gross_profit",  # 타겟을 quantity_sold로 변경했으므로 gross_profit은 제거
    ]
    
    # ========================================
    # 모델 1: quantity_sold 예측 모델
    # ========================================
    print("\n" + "="*60)
    print("모델 1: 판매 수량(quantity_sold) 예측 모델 학습")
    print("="*60)
    
    target1 = "quantity_sold"
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
    
    # 전처리기만 따로 추출
    # Optuna 하이퍼파라미터 최적화 실행 (필요시 주석 해제)
    # preprocessor = build_pipeline().named_steps['pre']
    # run_optuna_tuning(X1_train, y1_train_log, preprocessor)
    # sys.exit(0)
    
    # 최적화된 파라미터로 파이프라인 생성
    pipe1 = build_pipeline()
    pipe1.fit(X1_train, y1_train_log)
    print("모델 학습 완료.")

    # 예측 후 역변환
    y1_pred_log = pipe1.predict(X1_test)
    y1_pred = np.expm1(y1_pred_log)  # exp(y) - 1
    
    # Smearing Estimator 계산 (로그 변환 과소예측 보정)
    print("\nSmearing Estimator 계산 중...")
    y1_train_pred_log = pipe1.predict(X1_train)
    residuals = y1_train_log - y1_train_pred_log
    smearing_factor = np.mean(np.exp(residuals))
    print(f"  Smearing Factor: {smearing_factor:.4f}")
    
    # Smearing Estimator 적용한 예측
    y1_pred_smeared = y1_pred * smearing_factor
    
    # 기본 예측 평가
    mae1 = mean_absolute_error(y1_test_orig, y1_pred)
    rmse1 = np.sqrt(mean_squared_error(y1_test_orig, y1_pred))
    r2_1 = r2_score(y1_test_orig, y1_pred)
    
    # Smearing 적용 후 평가
    mae1_smeared = mean_absolute_error(y1_test_orig, y1_pred_smeared)
    rmse1_smeared = np.sqrt(mean_squared_error(y1_test_orig, y1_pred_smeared))
    r2_1_smeared = r2_score(y1_test_orig, y1_pred_smeared)
    
    print("\n=== 모델 1 평가 (quantity_sold) ===")
    print(f"기본 예측:")
    print(f"  MAE : {mae1:,.2f} 개")
    print(f"  RMSE: {rmse1:,.2f} 개")
    print(f"  R2  : {r2_1:.4f}")
    print(f"\nSmearing Estimator 적용 후:")
    print(f"  MAE : {mae1_smeared:,.2f} 개")
    print(f"  RMSE: {rmse1_smeared:,.2f} 개")
    print(f"  R2  : {r2_1_smeared:.4f}")
    print(f"\n개선:")
    print(f"  MAE : {mae1 - mae1_smeared:+,.2f} 개 ({(mae1 - mae1_smeared)/mae1*100:+.1f}%)")
    print(f"  R2  : {r2_1_smeared - r2_1:+.4f}\n")

    # 모델 1 저장 (Smearing Factor 포함)
    model_path1 = Path(__file__).parent / 'app' / MODEL_FILE_PROFIT
    model_data = {
        'pipeline': pipe1,
        'smearing_factor': smearing_factor
    }
    joblib.dump(model_data, model_path1)
    print(f"✅ 모델 1이 '{model_path1}'에 저장되었습니다. (Smearing Factor: {smearing_factor:.4f})")
    
    # 통계 저장 (API 호환성을 위해 profit_model 키 사용)
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
