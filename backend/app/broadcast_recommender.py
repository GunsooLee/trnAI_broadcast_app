#!/usr/bin/env python
"""broadcast_recommender.py

사용 예시(터미널에서 실행):
  # 1) 모델 학습 (새 데이터 적재 후 주기적으로 실행)
  python train.py

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
from pathlib import Path

import numpy as np
import os
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
import sys

# ---------------------------------------------------------------------------
# DB 설정 -----------------------------------------------------
# ---------------------------------------------------------------------------
DB_URI = os.getenv("DB_URI", "postgresql://TIKITAKA:TIKITAKA@175.106.97.27:5432/TIKITAKA_DB")
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


def search_product_codes_by_keywords(keywords: list[str], engine: Engine | None = None) -> list[str]:
    """product_name / keyword 컬럼 전체에 부분 매칭.

    - 입력 키워드를 공백·쉼표로 분할해 노멀라이즈.
    - product_name / keyword ILIKE 모두 검사.
    """
    if engine is None:
        engine = get_db_engine()

    norm_kw = _normalize_keywords(keywords)
    if not norm_kw:
        return []

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
            product_price
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
        b.time_slot || '_' || b.product_mgroup AS time_category_interaction
    FROM base b
    LEFT JOIN weather_daily w ON b.broadcast_date = w.weather_date
    LEFT JOIN product_stats p ON b.product_code = p.product_code
    LEFT JOIN category_timeslot_stats c ON b.product_mgroup = c.product_mgroup AND b.time_slot = c.time_slot
    LEFT JOIN category_overall_stats co ON b.product_mgroup = co.product_mgroup -- 신규 조인
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
        "timeslot_specialty_score", # <<< 신규 특성 추가
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
        "time_category_interaction", # <<< 새로운 상호작용 특성 추가
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
def fetch_product_info(product_codes: List[str], engine: Engine | None = None) -> pd.DataFrame:
    """product_name / keyword 컬럼 전체에 부분 매칭.

    - 입력 키워드를 공백·쉼표로 분할해 노멀라이즈.
    - product_name / keyword ILIKE 모두 검사.
    """
    if engine is None:
        engine = get_db_engine()

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
    
    df = pd.read_sql(query, engine)

    return df

# 카테고리 모드일 때도 동일하게 '카테고리별 전체 기간' 평균 실적 조회
@lru_cache(maxsize=5)
def fetch_category_info(engine: Engine | None = None) -> pd.DataFrame:
    """[수정됨] 카테고리 추천 모드에서 사용할 모든 상품 정보를 DB에서 조회한다."""
    if engine is None:
        engine = get_db_engine()

    # KeyError 해결: 모델 예측에 필요한 모든 컬럼(product_code, name, keyword 등)을
    # 포함하도록 쿼리를 수정합니다. 카테고리별 집계가 아닌, 개별 상품 정보를 가져옵니다.
    query = f"""
        SELECT 
            p.product_code,
            -- 집계를 위해 MAX를 사용하지만, product_code로 그룹화하므로 사실상 개별 값을 가져옵니다.
            MAX(p.product_name) AS product_name,
            MAX(p.keyword) AS keyword,
            MAX(p.product_lgroup) AS product_lgroup,
            MAX(p.product_mgroup) AS product_mgroup,
            MAX(p.product_sgroup) AS product_sgroup,
            MAX(p.product_dgroup) AS product_dgroup,
            MAX(p.product_type) AS product_type,
            -- 상품의 평균 가격 및 통계
            COALESCE(AVG(p.product_price), 0) AS product_price,
            COALESCE(AVG(s.sales_amount), 0) AS product_avg_sales,
            COUNT(s.broadcast_id) AS product_broadcast_count
        FROM {TABLE_NAME} p
        -- 자기 자신과 조인하여 상품별 통계를 계산합니다.
        LEFT JOIN {TABLE_NAME} s ON p.product_code = s.product_code
        GROUP BY p.product_code
    """
    df = pd.read_sql(query, engine)
    
    # 이 함수는 이제 사실상 모든 상품 정보를 가져오므로,
    # 함수 이름을 좀 더 명확하게 바꾸는 것이 좋습니다. (예: fetch_all_product_info)
    # 하지만 일단은 최소한의 변경으로 오류를 해결합니다.
    return df


@lru_cache(maxsize=5)
def fetch_category_timeslot_sales(engine: Engine | None = None) -> pd.DataFrame:
    """카테고리(중분류)와 시간대별 평균 판매액 정보를 DB에서 조회한다."""
    if engine is None:
        engine = get_db_engine()

    query = f"""
        SELECT 
            product_mgroup,
            time_slot,
            COALESCE(AVG(sales_amount), 0) AS category_timeslot_avg_sales
        FROM {TABLE_NAME}
        WHERE sales_amount IS NOT NULL
        GROUP BY product_mgroup, time_slot
    """
    df = pd.read_sql(query, engine)
    return df


@lru_cache(maxsize=5)
def get_category_overall_avg_sales(engine: Engine | None = None) -> Dict[str, float]:
    """카테고리(mgroup)별 전체 시간대 평균 매출액 조회"""
    if engine is None:
        engine = get_db_engine()

    query = f"""
        SELECT product_mgroup, AVG(sales_amount) as avg_sales
        FROM {TABLE_NAME}
        GROUP BY product_mgroup
    """
    df = pd.read_sql(query, engine)
    return pd.Series(df.avg_sales.values, index=df.product_mgroup).to_dict()


def get_db_engine():
    """새로운 DB 엔진을 생성하여 반환합니다. (스레드 안전성 확보)"""
    if not DB_URI:
        raise ValueError("DB_URI environment variable is not set.")
    return create_engine(DB_URI)


def prepare_candidate_row(
    date: dt.date,
    time_slot: str,
    product: pd.Series,
    weather_info: Dict[str, float],
    category_timeslot_sales_map: Dict[str, float],
    category_overall_sales_map: Dict[str, float], # <<< 인자 추가
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

    row = {
        # 날짜/시간 관련
        "broadcast_date": date.strftime("%Y-%m-%d"),
        "weekday": _weekday_kr(date),
        "time_slot": time_slot,
        "time_slot_int": slot_int,
        "season": _season_kr(date.month),
        # 날씨 정보
        "weather": weather_info.get("weather", "없음"),
        "temperature": weather_info.get("temperature", 0.0),
        "precipitation": weather_info.get("precipitation", 0.0),
        # 상품 정보
        "product_price": product.get("product_price", 0),
        "product_lgroup": product.get("product_lgroup", "없음"),
        "product_mgroup": product.get("product_mgroup", "없음"),
        "product_sgroup": product.get("product_sgroup", "없음"),
        "product_dgroup": product.get("product_dgroup", "없음"),
        "product_type": product.get("product_type", "없음"),
        "product_name": product.get("product_name", ""),
        "keyword": product.get("keyword", ""),
        "product_avg_sales": product.get("product_avg_sales", 0),
        "product_broadcast_count": product.get("product_broadcast_count", 0),
        "category_timeslot_avg_sales": timeslot_avg,
        "timeslot_specialty_score": timeslot_avg / overall_avg if overall_avg else 1,

        # 상호작용 특성
        "time_category_interaction": f"{time_slot}_{product.get('product_mgroup', '없음')}",
    }

    return row

def get_weather_by_date(date: dt.date, engine: Engine | None = None) -> Dict[str, float]:
    """weather_daily 테이블에서 주어진 날짜의 날씨 정보를 반환한다.

    반환 형식: {"weather": "맑음", "temperature": 23.4, "precipitation": 0.0}
    값이 없으면 기본값을 반환한다.
    """
    if engine is None:
        engine = get_db_engine()

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


def recommend(
    model: Pipeline, 
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
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """시간대별 최적 상품(또는 카테고리)을 추천한다.

    category_mode=True 이면 product_codes를 무시하고 카테고리 단위로 추천한다.
    """
    print("  [br.recommend] 2.1. recommend 함수 시작")
    # 스레드 내에서 새로운 DB 엔진 생성
    engine = get_db_engine()
    print("  [br.recommend] 2.2. DB 엔진 생성 완료")

    # 모델은 더 이상 여기서 로드하지 않습니다.
    pipe = model

    print("  [br.recommend] 2.3. 카테고리 정보 조회 시작...")
    all_categories_info = fetch_category_info(engine)
    print("  [br.recommend] 2.4. 카테고리 정보 조회 완료")

    category_timeslot_sales_df = fetch_category_timeslot_sales(engine)
    category_timeslot_sales_map = category_timeslot_sales_df.set_index(['product_mgroup', 'time_slot'])['category_timeslot_avg_sales'].to_dict()
    
    print("  [br.recommend] 2.5. 카테고리 평균 판매액 조회 시작...")
    category_overall_sales_map = get_category_overall_avg_sales(engine)
    print("  [br.recommend] 2.6. 카테고리 평균 판매액 조회 완료")

    if category_mode:
        print("  [br.recommend] 2.7. 카테고리 모드 실행")
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
        print("  [br.recommend] 2.7. 상품 모드 실행")
        if not product_codes:
            raise ValueError("product_codes가 비어 있습니다. --category 플래그를 사용하거나 상품코드를 입력하세요.")
        print("  [br.recommend] 2.8. 상품 정보 조회 시작...")
        items_df = fetch_product_info(product_codes, engine)
        print("  [br.recommend] 2.9. 상품 정보 조회 완료")
        if items_df.empty:
            raise ValueError("입력한 product_codes에 해당하는 상품이 없습니다.")
        label_col = "product_code"

    if weather_info is None or not weather_info.get("weather"):
        print("  [br.recommend] 2.10. 날씨 정보 조회 시작...")
        weather_info = get_weather_by_date(target_date, engine)
        print("  [br.recommend] 2.11. 날씨 정보 조회 완료")

    print("  [br.recommend] 2.12. 예측 후보 데이터 생성 시작...")

    # 1. 모든 시간 슬롯과 상품의 조합을 생성 (벡터화)
    all_combinations = pd.MultiIndex.from_product(
        [time_slots, items_df.index],
        names=["time_slot_temp", "item_index"]  # <<< 컬럼명 충돌 방지를 위해 이름 변경
    ).to_frame(index=False)

    # 2. 상품 정보와 조합 병합
    cand_df = pd.merge(all_combinations, items_df, left_on="item_index", right_index=True)
    cand_df.drop(columns=["item_index"], inplace=True)

    # 3. 날짜/시간 관련 특성 추가 (벡터화)
    cand_df["broadcast_date"] = target_date.strftime("%Y-%m-%d")
    cand_df["weekday"] = _weekday_kr(target_date)
    cand_df["season"] = _season_kr(target_date.month)
    slot_map = {"심야": 2, "아침": 7, "오전": 10, "점심": 12, "오후": 15, "저녁": 18, "야간": 21}
    # 임시 컬럼명을 사용하여 파생 특성 생성
    cand_df["time_slot_int"] = cand_df["time_slot_temp"].map(slot_map).fillna(12).astype(int)

    # 4. 날씨 정보 추가 (모든 행에 동일 값)
    for key, value in weather_info.items():
        cand_df[key] = value

    # 5. 카테고리별 판매액 특성 추가 (벡터화)
    cand_df["category_key"] = cand_df["product_mgroup"] + "_" + cand_df["time_slot_temp"]
    cand_df["category_timeslot_avg_sales"] = cand_df["category_key"].map(category_timeslot_sales_map).fillna(0)
    cand_df["category_overall_avg_sales"] = cand_df["product_mgroup"].map(category_overall_sales_map).fillna(0)
    cand_df["timeslot_specialty_score"] = (cand_df["category_timeslot_avg_sales"] / cand_df["category_overall_avg_sales"]).fillna(1)

    # 6. 상호작용 특성 추가
    cand_df["time_category_interaction"] = cand_df["time_slot_temp"] + "_" + cand_df["product_mgroup"]

    # 7. 임시 컬럼명을 원래 이름으로 변경
    cand_df.rename(columns={"time_slot_temp": "time_slot"}, inplace=True)

    # [중요] 데이터 타입 오류를 막기 위해 모든 범주형 컬럼의 None 값을 "없음"으로 채웁니다.
    categorical_cols = [
        "weather", "product_lgroup", "product_mgroup", "product_sgroup",
        "product_dgroup", "product_type", "time_category_interaction", "weekday", "season", "time_slot"
    ]
    for col in categorical_cols:
        if col in cand_df.columns:
            cand_df[col] = cand_df[col].fillna("없음")

    print(f"  [br.recommend] 2.13. 예측 후보 데이터 {len(cand_df)}개 생성 완료")

    print("  [br.recommend] 2.14. 모델 예측(pipe.predict) 시작...")
    # pipe.predict에 필요한 feature만 전달
    predictions = pipe.predict(cand_df[pipe.feature_names_in_])
    print("  [br.recommend] 2.15. 모델 예측 완료")

    cand_df["predicted_sales"] = predictions

    # [옵션] 시간대 강화 포스트-부스팅: 재학습 없이도 슬롯 × 카테고리 특성을 강화할 수 있음
    # 환경변수 TIMESLOT_BOOST_ALPHA 로 세기 조절 (기본 1.0 = 변경 없음)
    try:
        _alpha = float(os.getenv("TIMESLOT_BOOST_ALPHA", "1.0"))
    except Exception:
        _alpha = 1.0
    if _alpha != 1.0 and "timeslot_specialty_score" in cand_df.columns:
        cand_df["predicted_sales"] = cand_df["predicted_sales"] * (
            cand_df["timeslot_specialty_score"].replace([np.inf, -np.inf], 1.0).fillna(1.0) ** _alpha
        )

    # 3.a 보정: product 모드 등에서 'category' 컬럼이 없을 경우 생성
    if "category" not in cand_df.columns:
        group_cols = [
            "product_lgroup",
            "product_mgroup",
            "product_sgroup",
            "product_dgroup",
            "product_type",
        ]
        existing = [c for c in group_cols if c in cand_df.columns]
        if existing:
            cand_df["category"] = (
                cand_df[existing].astype(str).agg("/".join, axis=1)
            )
        else:
            cand_df["category"] = "미정"

    # --- [400 Bad Request 해결] API 스키마에 맞게 결과 포맷팅 ---
    # 1. 상위 N개 필터링 및 정렬
    result_df = (
        cand_df.sort_values(by="predicted_sales", ascending=False)
        .head(top_n * len(time_slots))
    )

    # 2. 데이터 타입 변환 (JSON 호환)
    result_df["predicted_sales"] = result_df["predicted_sales"].astype(float)

    # 3. 'features' 딕셔너리 생성
    feature_cols = ["product_name", "keyword", "product_price", "product_avg_sales", "product_broadcast_count"]
    result_df["features"] = result_df[feature_cols].to_dict(orient="records")

    # 4. [중복 컬럼 문제 해결] features에 포함된 원본 컬럼들을 삭제합니다.
    result_df.drop(columns=feature_cols, inplace=True, errors='ignore')

    # 5. 최종 스키마에 맞는 컬럼만 선택 
    final_cols = ["time_slot", "predicted_sales", "product_code", "category", "features"]
    result_df = result_df[final_cols]

    print(f"  [br.recommend] 2.16. 최종 추천 결과 {len(result_df)}개 생성 완료")
    return result_df


def _load_model() -> Pipeline:
    """XGBoost 모델 파이프라인을 로드한다."""
    # Pickle이 모델을 로드할 때 'tokenizer_utils' 모듈을 찾을 수 있도록 임시 조치
    # 현재 프로젝트 구조(app.tokenizer_utils)를 pickle이 기억하는 경로(tokenizer_utils)에 연결
    import sys
    from . import tokenizer_utils
    sys.modules['tokenizer_utils'] = tokenizer_utils

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_broadcast_sales.joblib")
    print(f"--- Loading model from: {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # joblib.load는 동기 함수이므로, run_in_threadpool을 통해 호출되어야 합니다.
    pipe = joblib.load(MODEL_PATH)
    return pipe

class BroadcastRecommender:
    """방송 편성 추천 관련 비즈니스 로직을 담당하는 클래스"""
    pass

# ---------------------------------------------------------------------------
# CLI 인터페이스 -------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="방송 편성 추천기")
    sub = parser.add_subparsers(dest="command")

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
        model = _load_model()
        rec_df = recommend(
            model,
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
