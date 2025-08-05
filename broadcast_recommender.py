#!/usr/bin/env python
"""broadcast_recommender.py

사용 예시(터미널에서 실행):
  # 1) 모델 학습 (새 데이터 적재 후 주기적으로 실행)
  python broadcast_recommender.py train

  # 2) 내일 방송편성 추천
  python broadcast_recommender.py recommend \
      --date 2025-07-18 \
      --time_slots "오전,오후,저녁" \

  # 3) 내일 방송편성 추천 (카테고리 단위)
  python broadcast_recommender.py recommend \
      --date 2025-07-18 \
      --time_slots "아침,오후,저녁" \
      --category

"""

import argparse
import datetime as dt
import joblib
import os
from typing import List, Dict
import re

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer_utils import mecab_tokenizer
from functools import lru_cache

# ---------------------------------------------------------------------------
# DB 설정 -----------------------------------------------------
# ---------------------------------------------------------------------------
DB_URI = "postgresql://TIKITAKA:TIKITAKA@TIKITAKA_postgres:5432/TIKITAKA_DB" # 서버
#DB_URI = "postgresql://TIKITAKA:TIKITAKA@175.106.97.27:5432/TIKITAKA_DB" # 로컬
TABLE_NAME = "broadcast_training_dataset"
MODEL_FILE = "xgb_broadcast_sales.joblib"

# ---------------------------------------------------------------------------
# 헬퍼: 키워드로 상품코드 조회 ---------------------------------------------
# ---------------------------------------------------------------------------

def _normalize_keywords(raw: list[str]) -> list[str]:
    """공백·쉼표·슬래시 등을 기준으로 분할 후 소문자 트림 & 중복 제거."""
    tokens: set[str] = set()
    for kw in raw:
        if not kw:
            continue
        for token in re.split(r"[\s,/]+", kw):
            token = token.strip().lower()
            if token:
                tokens.add(token)
    return list(tokens)


def search_product_codes_by_keywords(keywords: list[str]) -> list[str]:
    """product_name / keyword 컬럼 전체에 부분 매칭.

    - 입력 키워드를 공백·쉼표로 분할해 노멀라이즈.
    - product_name / keyword ILIKE 모두 검사.
    """

    norm_kw = _normalize_keywords(keywords)
    if not norm_kw:
        return []

    engine = create_engine(DB_URI)

    # OR 조건 구성  (ILIKE는 부분·대소문자 무시)
    clauses: list[str] = []
    params: dict[str, str] = {}
    for i, kw in enumerate(norm_kw):
        clauses.append(f"product_name ILIKE :kw{i} OR keyword ILIKE :kw{i}")
        params[f"kw{i}"] = f"%{kw}%"

    query = text(
        f"SELECT DISTINCT product_code FROM {TABLE_NAME} WHERE {' OR '.join(clauses)} LIMIT 200"
    )

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [r[0] for r in rows]

# ---------------------------------------------------------------------------
# 데이터 로딩 & 전처리 --------------------------------------------------------
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    단순 SELECT가 아닌, 윈도우 함수를 이용해 Feature Engineering이 적용된
    학습 데이터를 DataFrame으로 가져온다.
    """
    engine = create_engine(DB_URI)
    
    # 윈도우 함수를 사용한 SQL 쿼리
    # 각 행(방송)에 대해, 그 방송이 있기 전까지의 과거 데이터를 기반으로 평균값을 계산
    query = f"""
    WITH 
    -- 1) 방송 데이터
    base AS (
        SELECT 
            broadcast_id,
            broadcast_datetime,
            CAST(broadcast_datetime AS DATE) AS broadcast_date, -- <<< broadcast_datetime에서 날짜 추출
            broadcast_duration,
            product_code,
            product_lgroup,
            product_mgroup,
            product_sgroup,
            product_dgroup,
            product_type,
            product_name,
            keyword,
            time_slot,
            sales_amount,
            order_count,
            product_price,
            COALESCE(broadcast_showhost, 'NO_HOST') AS broadcast_showhost -- NULL을 'NO_HOST'로 처리
        FROM {TABLE_NAME}
        WHERE sales_amount IS NOT NULL
    ),
    -- 2) 상품별 통계 (전체 기간)
    product_stats AS (
        SELECT
            product_code,
            AVG(sales_amount) AS product_avg_sales,
            COUNT(*) AS product_broadcast_count
        FROM {TABLE_NAME}
        GROUP BY product_code
    ),
    -- 3) 카테고리-시간대별 통계 (전체 기간)
    category_timeslot_stats AS (
        SELECT
            product_mgroup,
            time_slot,
            AVG(sales_amount) AS category_timeslot_avg_sales
        FROM {TABLE_NAME}
        GROUP BY product_mgroup, time_slot
    ),
    -- 4) 카테고리 전체 평균 통계 (신규 추가)
    category_overall_stats AS (
        SELECT
            product_mgroup,
            AVG(sales_amount) AS category_overall_avg_sales
        FROM {TABLE_NAME}
        GROUP BY product_mgroup
    ),
    -- 5) 쇼호스트별 통계
    showhost_stats AS (
        SELECT
            COALESCE(broadcast_showhost, 'NO_HOST') AS broadcast_showhost,
            AVG(sales_amount) AS showhost_avg_sales,
            COUNT(*) AS showhost_broadcast_count
        FROM {TABLE_NAME}
        WHERE sales_amount IS NOT NULL
        GROUP BY COALESCE(broadcast_showhost, 'NO_HOST')
    ),
    -- 6) 쇼호스트-카테고리별 특화도 통계
    showhost_category_stats AS (
        SELECT
            COALESCE(broadcast_showhost, 'NO_HOST') AS broadcast_showhost,
            product_mgroup,
            AVG(sales_amount) AS showhost_category_avg_sales,
            COUNT(*) AS showhost_category_count
        FROM {TABLE_NAME}
        WHERE sales_amount IS NOT NULL
        GROUP BY COALESCE(broadcast_showhost, 'NO_HOST'), product_mgroup
    ),
    -- 7) 쇼호스트-시간대별 성과 통계
    showhost_timeslot_stats AS (
        SELECT
            COALESCE(broadcast_showhost, 'NO_HOST') AS broadcast_showhost,
            time_slot,
            AVG(sales_amount) AS showhost_timeslot_avg_sales,
            COUNT(*) AS showhost_timeslot_count
        FROM {TABLE_NAME}
        WHERE sales_amount IS NOT NULL
        GROUP BY COALESCE(broadcast_showhost, 'NO_HOST'), time_slot
    )
    -- 최종 학습 데이터셋 구성
    SELECT 
        b.*,
        w.temperature,
        w.precipitation,
        w.weather,
        p.product_avg_sales,
        p.product_broadcast_count,
        c.category_timeslot_avg_sales,
        -- 신규 특성: 시간대별 특화 점수 (division by zero 방지)
        COALESCE(c.category_timeslot_avg_sales / NULLIF(co.category_overall_avg_sales, 0), 1) AS timeslot_specialty_score,
        b.time_slot || '_' || b.product_mgroup AS time_category_interaction,
        -- 쇼호스트 관련 특성들
        hs.showhost_avg_sales,
        hs.showhost_broadcast_count,
        hcs.showhost_category_avg_sales,
        hcs.showhost_category_count,
        hts.showhost_timeslot_avg_sales,
        hts.showhost_timeslot_count,
        -- 쇼호스트 특화도 점수들
        COALESCE(hcs.showhost_category_avg_sales / NULLIF(hs.showhost_avg_sales, 0), 1) AS showhost_category_specialty,
        COALESCE(hts.showhost_timeslot_avg_sales / NULLIF(hs.showhost_avg_sales, 0), 1) AS showhost_timeslot_specialty
    FROM base b
    LEFT JOIN weather_daily w ON b.broadcast_date = w.weather_date
    LEFT JOIN product_stats p ON b.product_code = p.product_code
    LEFT JOIN category_timeslot_stats c ON b.product_mgroup = c.product_mgroup AND b.time_slot = c.time_slot
    LEFT JOIN category_overall_stats co ON b.product_mgroup = co.product_mgroup
    LEFT JOIN showhost_stats hs ON b.broadcast_showhost = hs.broadcast_showhost
    LEFT JOIN showhost_category_stats hcs ON b.broadcast_showhost = hcs.broadcast_showhost AND b.product_mgroup = hcs.product_mgroup
    LEFT JOIN showhost_timeslot_stats hts ON b.broadcast_showhost = hts.broadcast_showhost AND b.time_slot = hts.time_slot
    """
    
    df = pd.read_sql(query, engine)

    # --- 모델 학습에 필요한 피처 엔지니어링 ---
    # broadcast_datetime을 datetime 객체로 변환
    df['broadcast_datetime'] = pd.to_datetime(df['broadcast_datetime'])

    # 요일, 시즌, 시간대(숫자) 파생변수 생성
    df["weekday"] = df["broadcast_datetime"].dt.day_name().map({
        'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'
    })
    df["season"] = df["broadcast_datetime"].dt.month.apply(_season_kr)
    slot_map = {
        "심야": 2, "아침": 7, "오전": 10, "점심": 12, "오후": 15, "저녁": 18, "야간": 21
    }
    df["time_slot_int"] = df["time_slot"].map(slot_map)

    # NULL 값을 채우기 (LEFT JOIN으로 인해 발생 가능)
    df['product_avg_sales'] = df['product_avg_sales'].fillna(0)
    df['category_timeslot_avg_sales'] = df['category_timeslot_avg_sales'].fillna(0)
    df['product_broadcast_count'] = df['product_broadcast_count'].fillna(0)
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    df['precipitation'] = df['precipitation'].fillna(0)
    df['weather'] = df['weather'].fillna('정보없음') # 날씨 정보 없는 경우 대비
    
    # 쇼호스트 관련 NULL 값 처리
    df['showhost_avg_sales'] = df['showhost_avg_sales'].fillna(0)
    df['showhost_broadcast_count'] = df['showhost_broadcast_count'].fillna(0)
    df['showhost_category_avg_sales'] = df['showhost_category_avg_sales'].fillna(0)
    df['showhost_category_count'] = df['showhost_category_count'].fillna(0)
    df['showhost_timeslot_avg_sales'] = df['showhost_timeslot_avg_sales'].fillna(0)
    df['showhost_timeslot_count'] = df['showhost_timeslot_count'].fillna(0)
    df['showhost_category_specialty'] = df['showhost_category_specialty'].fillna(1)
    df['showhost_timeslot_specialty'] = df['showhost_timeslot_specialty'].fillna(1)
    
    return df


def build_pipeline() -> Pipeline:
    """수치/범주형/텍스트 전처리 + XGBoost 파이프라인 생성"""

    numeric_features = [
        "temperature",
        "precipitation",
        "product_price",
        "time_slot_int",
        "product_avg_sales",
        "product_broadcast_count",
        "category_timeslot_avg_sales",
        "timeslot_specialty_score",
        # 쇼호스트 관련 수치 특성
        "showhost_avg_sales",
        "showhost_broadcast_count",
        "showhost_category_avg_sales",
        "showhost_category_count",
        "showhost_timeslot_avg_sales",
        "showhost_timeslot_count",
        "showhost_category_specialty",
        "showhost_timeslot_specialty",
    ]

    categorical_features = [
        "weekday",
        "time_slot",
        "season",
        "weather",
        "product_lgroup",
        "product_mgroup",
        "product_sgroup",
        "product_dgroup",
        "product_type",
        "time_category_interaction",
        # 쇼호스트 관련 범주형 특성
        "broadcast_showhost",
    ]

    # ---- 텍스트 피처 ----------------------------------------
    text_transformers = [
        (
            "product_name_tfidf",
            TfidfVectorizer(
                tokenizer=mecab_tokenizer,
                lowercase=False,
                ngram_range=(1, 2),
                max_features=5000,
            ),
            "product_name",
        ),
        (
            "keyword_tfidf",
            TfidfVectorizer(
                tokenizer=mecab_tokenizer,
                lowercase=False,
                ngram_range=(1, 2),
                max_features=3000,
            ),
            "keyword",
        ),
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            *text_transformers,
        ]
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

# ---------------------------------------------------------------------------
# 모델 학습 -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def train() -> None:
    df = load_data()

    target = "sales_amount"
    # 모델 입력에 사용하지 않을 컬럼 지정.  
    # 텍스트 컬럼을 이제 학습에 사용하므로 삭제하지 않음
    drop_cols = [
        "broadcast_id",
        "broadcast_duration",        
        "broadcast_datetime",
        "order_count",  # 단일 타깃 학습을 위해 제외. 필요시 다중 타깃 학습 가능
        target,
    ]
    X = df.drop(columns=drop_cols)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\n=== 모델 평가 ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2  : {r2_score(y_test, y_pred):.2f}\n")

    joblib.dump(pipe, MODEL_FILE)
    print(f"모델이 '{MODEL_FILE}'에 저장되었습니다.")

# ---------------------------------------------------------------------------
# 추천 -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _weekday_kr(date: dt.date) -> str:
    return ["월", "화", "수", "목", "금", "토", "일"][date.weekday()]


def _season_kr(month: int) -> str:
    return (
        "겨울"
        if month in [12, 1, 2]
        else "봄"
        if month in [3, 4, 5]
        else "여름"
        if month in [6, 7, 8]
        else "가을"
    )


# 추천할 상품의 '전체 기간' 평균 실적을 조회하도록 수정
def fetch_product_info(product_codes: List[str]) -> pd.DataFrame:
    engine = create_engine(DB_URI)
    code_tuple = tuple(product_codes)
    query = f"""
        SELECT 
            product_code,
            -- 상품의 대표 카테고리 정보 (가장 마지막 값 사용)
            MAX(product_lgroup) AS product_lgroup,
            MAX(product_mgroup) AS product_mgroup,
            MAX(product_sgroup) AS product_sgroup,
            MAX(product_dgroup) AS product_dgroup,
            MAX(product_type) AS product_type,
            MAX(product_name)  AS product_name,   -- NEW
            MAX(keyword)       AS keyword,        -- NEW
            -- 모델에 필요한 피처들 (전체 기간 Groupby)
            COALESCE(AVG(product_price), 0) AS product_price,
            COALESCE(AVG(sales_amount), 0) AS product_avg_sales,
            COUNT(*) AS product_broadcast_count
        FROM {TABLE_NAME}
        WHERE product_code IN {code_tuple}
        GROUP BY product_code
    """
    return pd.read_sql(query, engine)

# 카테고리 모드일 때도 동일하게 '카테고리별 전체 기간' 평균 실적 조회
def fetch_category_info() -> pd.DataFrame:
    engine = create_engine(DB_URI)
    query = f"""
        SELECT 
            product_lgroup,
            product_mgroup,
            product_sgroup,
            product_dgroup,
            product_type,
            time_slot,
            COALESCE(AVG(product_price), 0) AS product_price,
            COALESCE(AVG(sales_amount), 0) AS category_timeslot_avg_sales
            -- 필요시 카테고리별 방송 횟수 등도 추가 가능
        FROM {TABLE_NAME}
        GROUP BY product_lgroup, product_mgroup, product_sgroup, product_dgroup, product_type, time_slot 
    """
    return pd.read_sql(query, engine)


def get_category_overall_avg_sales() -> Dict[str, float]:
    """카테고리(mgroup)별 전체 시간대 평균 매출액 조회"""
    engine = create_engine(DB_URI)
    query = f"""
    SELECT product_mgroup, AVG(sales_amount) as avg_sales
    FROM {TABLE_NAME}
    WHERE sales_amount IS NOT NULL
    GROUP BY product_mgroup
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    return {row[0]: row[1] for row in rows}


def get_showhost_stats() -> Dict[str, Dict[str, float]]:
    """쇼호스트별 기본 통계 조회"""
    engine = create_engine(DB_URI)
    query = f"""
    SELECT 
        COALESCE(broadcast_showhost, 'NO_HOST') as showhost_id,
        AVG(sales_amount) as avg_sales,
        COUNT(*) as broadcast_count
    FROM {TABLE_NAME}
    WHERE sales_amount IS NOT NULL
    GROUP BY COALESCE(broadcast_showhost, 'NO_HOST')
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    
    result = {}
    for row in rows:
        result[row[0]] = {
            'avg_sales': row[1],
            'broadcast_count': row[2]
        }
    return result


def get_showhost_category_stats() -> Dict[str, Dict[str, float]]:
    """쇼호스트-카테고리별 특화도 통계 조회"""
    engine = create_engine(DB_URI)
    query = f"""
    SELECT 
        COALESCE(broadcast_showhost, 'NO_HOST') as showhost_id,
        product_mgroup,
        AVG(sales_amount) as avg_sales,
        COUNT(*) as broadcast_count
    FROM {TABLE_NAME}
    WHERE sales_amount IS NOT NULL
    GROUP BY COALESCE(broadcast_showhost, 'NO_HOST'), product_mgroup
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    
    result = {}
    for row in rows:
        key = f"{row[0]}_{row[1]}"  # showhost_id_category
        result[key] = {
            'avg_sales': row[2],
            'broadcast_count': row[3]
        }
    return result


def get_showhost_timeslot_stats() -> Dict[str, Dict[str, float]]:
    """쇼호스트-시간대별 성과 통계 조회"""
    engine = create_engine(DB_URI)
    query = f"""
    SELECT 
        COALESCE(broadcast_showhost, 'NO_HOST') as showhost_id,
        time_slot,
        AVG(sales_amount) as avg_sales,
        COUNT(*) as broadcast_count
    FROM {TABLE_NAME}
    WHERE sales_amount IS NOT NULL
    GROUP BY COALESCE(broadcast_showhost, 'NO_HOST'), time_slot
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    
    result = {}
    for row in rows:
        key = f"{row[0]}_{row[1]}"  # showhost_id_timeslot
        result[key] = {
            'avg_sales': row[2],
            'broadcast_count': row[3]
        }
    return result


def prepare_candidate_row(
    date: dt.date,
    time_slot: str,
    product: pd.Series,
    weather_info: Dict[str, float],
    category_timeslot_sales_map: Dict[str, float],
    category_overall_sales_map: Dict[str, float],
    showhost_id: str = "NO_HOST",  # 쇼호스트 ID 추가
    showhost_stats: Dict[str, Dict[str, float]] = None,  # 쇼호스트 통계
    showhost_category_stats: Dict[str, Dict[str, float]] = None,  # 쇼호스트-카테고리 통계
    showhost_timeslot_stats: Dict[str, Dict[str, float]] = None,  # 쇼호스트-시간대 통계
) -> Dict:
    """모델이 요구하는 입력 형태(딕셔너리)로 변환"""

    # time_slot이 숫자("08")이면 그대로, 아니면 한글 구간명을 매핑
    if time_slot.isdigit():
        slot_int = int(time_slot)
    else:
        slot_map = {
            "심야": 2,   # 00-05
            "아침": 7,   # 06-08
            "오전": 10,  # 09-11
            "점심": 12,  # 12-13
            "오후": 15,  # 14-16
            "저녁": 18,  # 17-19
            "야간": 21,  # 20-23
        }
        slot_int = slot_map.get(time_slot, 12)

    category_key = f"{product['product_mgroup']}_{time_slot}"
    timeslot_avg = category_timeslot_sales_map.get(category_key, 0)
    overall_avg = category_overall_sales_map.get(product['product_mgroup'], 0)
    
    # 쇼호스트 관련 통계 계산
    showhost_stats = showhost_stats or {}
    showhost_category_stats = showhost_category_stats or {}
    showhost_timeslot_stats = showhost_timeslot_stats or {}
    
    showhost_basic = showhost_stats.get(showhost_id, {'avg_sales': 0, 'broadcast_count': 0})
    showhost_category_key = f"{showhost_id}_{product['product_mgroup']}"
    showhost_category = showhost_category_stats.get(showhost_category_key, {'avg_sales': 0, 'broadcast_count': 0})
    showhost_timeslot_key = f"{showhost_id}_{time_slot}"
    showhost_timeslot = showhost_timeslot_stats.get(showhost_timeslot_key, {'avg_sales': 0, 'broadcast_count': 0})
    
    # 쇼호스트 특화도 점수 계산
    showhost_avg_sales = showhost_basic['avg_sales']
    showhost_category_specialty = (showhost_category['avg_sales'] / showhost_avg_sales) if showhost_avg_sales > 0 else 1
    showhost_timeslot_specialty = (showhost_timeslot['avg_sales'] / showhost_avg_sales) if showhost_avg_sales > 0 else 1

    row = {
        # 날짜/시간 관련
        "broadcast_date": date.strftime("%Y-%m-%d"),
        "weekday": _weekday_kr(date),
        "time_slot": time_slot,
        "time_slot_int": slot_int,
        "season": _season_kr(date.month),
        # 날씨 정보
        "weather": weather_info["weather"],
        "temperature": weather_info["temperature"],
        "precipitation": weather_info["precipitation"],
        # 상품 정보
        "product_price": product.get("product_price", 0),
        "product_lgroup": product.get("product_lgroup"),
        "product_mgroup": product.get("product_mgroup"),
        "product_sgroup": product.get("product_sgroup"),
        "product_dgroup": product.get("product_dgroup"),
        "product_type": product.get("product_type"),
        "product_name": product.get("product_name", ""),
        "keyword": product.get("keyword", ""),
        "product_avg_sales": product.get("product_avg_sales", 0),
        "product_broadcast_count": product.get("product_broadcast_count", 0),
        "category_timeslot_avg_sales": timeslot_avg,
        "timeslot_specialty_score": timeslot_avg / overall_avg if overall_avg else 1,

        # 상호작용 특성
        "time_category_interaction": f"{time_slot}_{product.get('product_mgroup', '기타')}",
        
        # 쇼호스트 관련 특성
        "broadcast_showhost": showhost_id,
        "showhost_avg_sales": showhost_avg_sales,
        "showhost_broadcast_count": showhost_basic['broadcast_count'],
        "showhost_category_avg_sales": showhost_category['avg_sales'],
        "showhost_category_count": showhost_category['broadcast_count'],
        "showhost_timeslot_avg_sales": showhost_timeslot['avg_sales'],
        "showhost_timeslot_count": showhost_timeslot['broadcast_count'],
        "showhost_category_specialty": showhost_category_specialty,
        "showhost_timeslot_specialty": showhost_timeslot_specialty,
    }

    return row

def get_weather_by_date(date: dt.date) -> Dict[str, float]:
    """weather_daily 테이블에서 주어진 날짜의 날씨 정보를 반환한다.

    반환 형식: {"weather": "맑음", "temperature": 23.4, "precipitation": 0.0}
    값이 없으면 기본값을 반환한다.
    """
    engine = create_engine(DB_URI)
    query = text(
        """
        SELECT weather, temperature, precipitation
        FROM weather_daily
        WHERE weather_date = :d
        LIMIT 1
        """
    )

    df = pd.read_sql(query, engine, params={"d": date})
    if df.empty:
        return {"weather": "맑음", "temperature": 20.0, "precipitation": 0.0}
    row = df.iloc[0]
    return {
        "weather": row["weather"],
        "temperature": float(row["temperature"] or 20.0),
        "precipitation": float(row["precipitation"] or 0.0),
    }


def predict_category_performance(
    target_date: dt.date,
    time_slots: List[str],
    weather_info: Dict[str, float] | None = None,
    showhost_id: str = "NO_HOST",
    top_categories: int = 10
) -> pd.DataFrame:
    """시간대별 카테고리 성과를 예측하여 상위 카테고리를 반환한다."""
    
    # 모델 로드
    pipe: Pipeline = _load_model()
    
    # 통계 데이터 로드
    all_categories_info = fetch_category_info()
    category_timeslot_sales_map = all_categories_info.set_index(['product_mgroup', 'time_slot'])['category_timeslot_avg_sales'].to_dict()
    category_overall_sales_map = get_category_overall_avg_sales()
    showhost_stats = get_showhost_stats()
    showhost_category_stats = get_showhost_category_stats()
    showhost_timeslot_stats = get_showhost_timeslot_stats()
    
    if weather_info is None:
        weather_info = get_weather_by_date(target_date)
    
    # 카테고리별 대표 상품 선정 (각 카테고리에서 가장 성과가 좋은 상품)
    category_representatives = {}
    for mgroup in all_categories_info['product_mgroup'].unique():
        category_items = all_categories_info[all_categories_info['product_mgroup'] == mgroup]
        # 해당 카테고리에서 가장 평균 매출이 높은 상품 선택
        best_item = category_items.loc[category_items['category_timeslot_avg_sales'].idxmax()]
        category_representatives[mgroup] = best_item
    
    # 시간대별 카테고리 성과 예측
    category_predictions = []
    for slot in time_slots:
        for mgroup, representative in category_representatives.items():
            row_dict = prepare_candidate_row(
                target_date,
                slot,
                representative,
                weather_info,
                category_timeslot_sales_map,
                category_overall_sales_map,
                showhost_id=showhost_id,
                showhost_stats=showhost_stats,
                showhost_category_stats=showhost_category_stats,
                showhost_timeslot_stats=showhost_timeslot_stats
            )
            
            pred = pipe.predict(pd.DataFrame([row_dict]))[0]
            category_predictions.append({
                'time_slot': slot,
                'category': mgroup,
                'predicted_sales': pred,
                'representative_item': representative,
                'features': row_dict
            })
    
    # DataFrame으로 변환 후 시간대별 상위 카테고리 선정
    pred_df = pd.DataFrame(category_predictions)
    
    # 시간대별 상위 카테고리 선정
    top_categories_by_slot = []
    for slot in time_slots:
        slot_df = pred_df[pred_df['time_slot'] == slot]
        top_slot_categories = (
            slot_df.sort_values('predicted_sales', ascending=False)
            .head(top_categories)
        )
        top_categories_by_slot.append(top_slot_categories)
    
    if top_categories_by_slot:
        return pd.concat(top_categories_by_slot).reset_index(drop=True)
    else:
        return pd.DataFrame()


def recommend_category_first(
    target_date: dt.date,
    time_slots: List[str],
    weather_info: Dict[str, float] | None = None,
    showhost_id: str = "NO_HOST",
    top_categories_per_slot: int = 3,
    top_products_per_category: int = 1
) -> pd.DataFrame:
    """카테곦0리 우선 추천 방식
    
    1단계: 시간대별 최적 카테고리 선정
    2단계: 선정된 카테고리 내에서 최적 상품 추천
    """
    
    # 1단계: 카테고리 성과 예측
    category_performance = predict_category_performance(
        target_date=target_date,
        time_slots=time_slots,
        weather_info=weather_info,
        showhost_id=showhost_id,
        top_categories=top_categories_per_slot
    )
    
    if category_performance.empty:
        return pd.DataFrame()
    
    # 2단계: 선정된 카테곦0리 내에서 상품 추천
    final_recommendations = []
    
    for slot in time_slots:
        slot_categories = category_performance[category_performance['time_slot'] == slot]
        
        for _, category_row in slot_categories.head(top_categories_per_slot).iterrows():
            category = category_row['category']
            
            # 해당 카테곦0리의 상품들 조회
            category_products = get_products_by_category(category)
            
            if not category_products.empty:
                # 카테곦0리 내 최적 상품 예측
                best_products = recommend_products_in_category(
                    target_date=target_date,
                    time_slot=slot,
                    category=category,
                    products_df=category_products,
                    weather_info=weather_info,
                    showhost_id=showhost_id,
                    top_n=top_products_per_category
                )
                
                for _, product in best_products.iterrows():
                    final_recommendations.append({
                        'time_slot': slot,
                        'category': category,
                        'category_predicted_sales': category_row['predicted_sales'],
                        'product_code': product.get('product_code'),
                        'product_name': product.get('product_name'),
                        'product_predicted_sales': product.get('predicted_sales'),
                        'product_lgroup': product.get('product_lgroup'),
                        'product_mgroup': product.get('product_mgroup'),
                        'product_sgroup': product.get('product_sgroup'),
                        'product_dgroup': product.get('product_dgroup'),
                        'showhost_id': showhost_id,
                        'recommendation_reason': f"카테곦0리 '{category}' 예상매출: {category_row['predicted_sales']:.0f}원"
                    })
    
    return pd.DataFrame(final_recommendations)


def get_products_by_category(category: str) -> pd.DataFrame:
    """특정 카테곦0리의 상품들을 조회한다."""
    engine = create_engine(DB_URI)
    query = f"""
    SELECT DISTINCT
        product_code,
        product_name,
        product_lgroup,
        product_mgroup,
        product_sgroup,
        product_dgroup,
        product_type,
        keyword,
        AVG(product_price) as product_price,
        AVG(sales_amount) as product_avg_sales,
        COUNT(*) as product_broadcast_count
    FROM {TABLE_NAME}
    WHERE product_mgroup = :category
        AND sales_amount IS NOT NULL
    GROUP BY product_code, product_name, product_lgroup, product_mgroup, 
             product_sgroup, product_dgroup, product_type, keyword
    ORDER BY AVG(sales_amount) DESC
    LIMIT 50  -- 상위 50개 상품만 고려
    """
    return pd.read_sql(query, engine, params={'category': category})


def recommend_products_in_category(
    target_date: dt.date,
    time_slot: str,
    category: str,
    products_df: pd.DataFrame,
    weather_info: Dict[str, float],
    showhost_id: str = "NO_HOST",
    top_n: int = 1
) -> pd.DataFrame:
    """특정 카테곦0리 내에서 최적 상품들을 추천한다."""
    
    # 모델 로드
    pipe: Pipeline = _load_model()
    
    # 통계 데이터 로드
    all_categories_info = fetch_category_info()
    category_timeslot_sales_map = all_categories_info.set_index(['product_mgroup', 'time_slot'])['category_timeslot_avg_sales'].to_dict()
    category_overall_sales_map = get_category_overall_avg_sales()
    showhost_stats = get_showhost_stats()
    showhost_category_stats = get_showhost_category_stats()
    showhost_timeslot_stats = get_showhost_timeslot_stats()
    
    # 상품별 예측
    product_predictions = []
    for _, product in products_df.iterrows():
        row_dict = prepare_candidate_row(
            target_date,
            time_slot,
            product,
            weather_info,
            category_timeslot_sales_map,
            category_overall_sales_map,
            showhost_id=showhost_id,
            showhost_stats=showhost_stats,
            showhost_category_stats=showhost_category_stats,
            showhost_timeslot_stats=showhost_timeslot_stats
        )
        
        pred = pipe.predict(pd.DataFrame([row_dict]))[0]
        product_dict = product.to_dict()
        product_dict['predicted_sales'] = pred
        product_dict['features'] = row_dict
        product_predictions.append(product_dict)
    
    # 예측 매출 상위 N개 선정
    pred_df = pd.DataFrame(product_predictions)
    return pred_df.sort_values('predicted_sales', ascending=False).head(top_n)


def recommend(
    target_date: dt.date,
    time_slots: List[str],
    product_codes: List[str] | None = None,
    weather_info: Dict[str, float] | None = None,
    category_mode: bool = False,
    categories: List[str] | None = None,
    *,
    top_k_sample: int = 5,
    temp: float = 0.5,
    top_n: int | None = None,
    use_category_first: bool = True,  # 새로운 매개변수: 카테곦0리 우선 방식 사용 여부
    showhost_id: str = "NO_HOST",     # 쇼호스트 ID 매개변수 추가
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """시간대별 최적 상품(또는 카테고리)을 추천한다.

    use_category_first=True: 카테고리 우선 추천 방식 사용
    category_mode=True: 기존 카테고리 모드 사용
    """
    
    # 카테고리 우선 방식 사용
    if use_category_first:
        result = recommend_category_first(
            target_date=target_date,
            time_slots=time_slots,
            weather_info=weather_info,
            showhost_id=showhost_id,
            top_categories_per_slot=3,
            top_products_per_category=1
        )
        
        # 기존 형식과 맞추기 위해 컴럼 이름 조정
        if not result.empty:
            result = result.rename(columns={
                'product_predicted_sales': 'predicted_sales',
                'product_code': 'product_code',
                'product_name': 'product_name'
            })
            
            # 기존 형식에 맞는 컴럼 추가
            display_cols = [
                "time_slot",
                "predicted_sales", 
                "product_code",
                "product_name",
                "product_lgroup",
                "product_mgroup",
                "product_sgroup",
                "product_dgroup",
                "category",
                "showhost_id",
                "recommendation_reason"
            ]
            final_cols = [col for col in display_cols if col in result.columns]
            return result[final_cols]
        else:
            return pd.DataFrame()
    
    # 기존 방식 사용
    # 모델은 메모리에 1회만 로드해 재사용 -----------------------------
    pipe: Pipeline = _load_model()

    all_categories_info = fetch_category_info()
    category_timeslot_sales_map = all_categories_info.set_index(['product_mgroup', 'time_slot'])['category_timeslot_avg_sales'].to_dict()
    category_overall_sales_map = get_category_overall_avg_sales()
    
    # 쇼호스트 관련 통계 로드
    showhost_stats = get_showhost_stats()
    showhost_category_stats = get_showhost_category_stats()
    showhost_timeslot_stats = get_showhost_timeslot_stats()

    if category_mode:
        items_df = all_categories_info.copy()
        key_cols = [
            "product_lgroup",
            "product_mgroup",
            "product_sgroup",
            "product_dgroup",
            "product_type",
        ]
        label_col = "category"
        # 카테고리 식별자 문자열 생성
        items_df[label_col] = (
            items_df[key_cols]
            .astype(str)
            .agg("/".join, axis=1)
        )

        # 사용자가 특정 카테고리만 입력한 경우 필터링
        if categories:
            filtered = items_df[items_df[label_col].isin(categories)]
            # 지정 카테고리에 후보가 없으면 전체 풀로 되돌아가 추천 이어서 수행
            if not filtered.empty:
                items_df = filtered
    else:
        if not product_codes:
            raise ValueError("product_codes가 비어 있습니다. --category 플래그를 사용하거나 상품코드를 입력하세요.")
        items_df = fetch_product_info(product_codes)
        if items_df.empty:
            raise ValueError("입력한 product_codes에 해당하는 상품이 없습니다.")
        label_col = "product_code"

    if weather_info is None or not weather_info.get("weather"):
        weather_info = get_weather_by_date(target_date)

    candidates: List[Dict] = []
    for slot in time_slots:
        for _, item in items_df.iterrows():
            # 쇼호스트 ID는 매개변수로 받은 값 사용
            
            row_dict = prepare_candidate_row(
                target_date, 
                slot, 
                item, 
                weather_info, 
                category_timeslot_sales_map, 
                category_overall_sales_map,
                showhost_id=showhost_id,
                showhost_stats=showhost_stats,
                showhost_category_stats=showhost_category_stats,
                showhost_timeslot_stats=showhost_timeslot_stats
            )
            pred = pipe.predict(pd.DataFrame([row_dict]))[0]
            candidates.append({
                "time_slot": slot,
                label_col: item[label_col],
                "product_lgroup": item.get("product_lgroup"),
                "product_mgroup": item.get("product_mgroup"),
                "product_sgroup": item.get("product_sgroup"),
                "product_dgroup": item.get("product_dgroup"),
                "predicted_sales": pred,
                "showhost_id": showhost_id,  # 쇼호스트 ID 추가
                "features": row_dict, # <<< 예측에 사용된 파라미터 추가
            })

    cand_df = pd.DataFrame(candidates)

    # 시간대별로 예측 매출이 가장 높은 상위 N개 후보를 선택
    if not top_n:
        top_n = 1

    top_candidates = []
    for slot in time_slots:
        slot_df = cand_df[cand_df["time_slot"] == slot]
        if slot_df.empty:
            continue

        # 예측 매출 내림차순 정렬 후, label_col(카테고리/상품) 기준 중복 제거
        top_slot_candidates = (
            slot_df.sort_values("predicted_sales", ascending=False)
            .drop_duplicates(subset=[label_col], keep="first")
            .head(top_n)
        )
        top_candidates.append(top_slot_candidates)

    if not top_candidates:
        return pd.DataFrame()

    result_df = pd.concat(top_candidates).reset_index(drop=True)

    # 화면 표시에 필요한 컬럼들을 명시적으로 선택하여 반환
    display_cols = [
        "time_slot",
        "predicted_sales",
        "product_code",
        "product_lgroup",
        "product_mgroup",
        "product_sgroup",
        "product_dgroup",
        "category", # 단일 카테고리 열도 유지
        "features", # <<< 예측에 사용된 파라미터 추가
    ]
    # 반환할 df에 없는 컬럼은 제외하고, 존재하는 컬럼만 선택
    final_cols = [col for col in display_cols if col in result_df.columns]
    return result_df[final_cols]

# ---------------------------------------------------------------------------
# 모델 로딩 캐시 -------------------------------------------------------------
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> Pipeline:
    """디스크에서 학습된 파이프라인을 1회만 로드해 캐시한다."""
    return joblib.load(MODEL_FILE)

# ---------------------------------------------------------------------------
# CLI 인터페이스 -------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="방송 편성 추천기")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="모델 학습")

    rec_parser = sub.add_parser("recommend", help="편성 추천")
    rec_parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    rec_parser.add_argument(
        "--time_slots", default="아침,오전,점심,오후,저녁,야간", help="콤마로 구분된 시간대 문자열"
    )
    rec_parser.add_argument("--products", help="콤마로 구분된 상품코드 목록 (옵션)")
    rec_parser.add_argument(
        "--weather", default="맑음", help="해당 일자의 예상 날씨 (맑음/흐림 등)"
    )
    rec_parser.add_argument(
        "--temp", type=float, default=25.0, help="해당 일자 평균 기온"
    )
    rec_parser.add_argument(
        "--precip", type=float, default=0.0, help="해당 일자 예상 강수량(mm)"
    )
    rec_parser.add_argument("--category", action="store_true", help="카테고리 단위 추천 활성화")
    rec_parser.add_argument(
        "--categories",
        help="콤마로 구분된 카테고리 식별자 목록 (대/중/소/세/product_type). --category 플래그와 함께 사용",
    )
    rec_parser.add_argument("--top_k_sample", type=int, default=3, help="다양성 샘플링 크기")
    rec_parser.add_argument("--diversity_temp", type=float, default=0.5, help="다양성 샘플링 온도 (0=탐욕, ↑=랜덤)")
    rec_parser.add_argument("--top_n", type=int, help="상위 N개 후보 반환")

    args = parser.parse_args()

    if args.command == "train":
        train()
        return

    if args.command == "recommend":
        date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        slots = [s.strip() for s in args.time_slots.split(",") if s.strip()]
        products = [p.strip() for p in args.products.split(",") if p.strip()] if args.products else []
        categories = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
        weather_info = {
            "weather": args.weather,
            "temperature": args.temp,
            "precipitation": args.precip,
        }
        rec_df = recommend(
            date,
            slots,
            products,
            weather_info,
            category_mode=args.category or bool(categories),
            categories=categories,
            top_k_sample=args.top_k_sample,
            temp=args.diversity_temp,
            top_n=args.top_n,
        )
        if isinstance(rec_df, tuple):
            print("\n=== 추천 편성표 ===")
            print(rec_df[0].to_string(index=False))
            print("\n=== 상위 N개 후보 ===")
            print(rec_df[1].to_string(index=False))
        else:
            print("\n=== 추천 편성표 ===")
            print(rec_df.to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
