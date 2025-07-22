#!/usr/bin/env python
"""broadcast_recommender.py

사용 예시(터미널에서 실행):
  # 1) 모델 학습 (새 데이터 적재 후 주기적으로 실행)
  python broadcast_recommender.py train

  # 2) 내일 방송편성 추천
  python broadcast_recommender.py recommend \
      --date 2025-07-18 \
      --time_slots "08,12,19,21" \
      --products "P1001,P2002,P3003,P4004"

  # 3) 내일 방송편성 추천 (카테고리 단위)
  python broadcast_recommender.py recommend \
      --date 2025-07-18 \
      --time_slots "08,12,19,21" \
      --category

"""

import argparse
import datetime as dt
import joblib
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# DB 설정 -----------------------------------------------------
# ---------------------------------------------------------------------------
DB_URI = "postgresql://TIKITAKA:TIKITAKA@TIKITAKA_postgres:5432/TIKITAKA_DB"  # WSL2 Docker 컨테이너(Postgres 16) 접속 정보
TABLE_NAME = "broadcast_training_dataset"
MODEL_FILE = "xgb_broadcast_sales.joblib"

# ---------------------------------------------------------------------------
# 데이터 로딩 & 전처리 --------------------------------------------------------
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """전체 학습 데이터를 DataFrame으로 가져온다."""
    engine = create_engine(DB_URI)
    query = f"SELECT * FROM {TABLE_NAME} WHERE sales_amount IS NOT NULL"
    return pd.read_sql(query, engine)


def build_pipeline() -> Pipeline:
    """수치/범주형 전처리 + XGBoost를 묶은 파이프라인을 생성한다."""

    numeric_features = [
        "temperature",
        "precipitation",
        "product_price",
        "time_slot_int",  # 새로 추가된 0~23 정수형 시간대
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
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
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
    # 텍스트 컬럼은 제외: 별도 텍스트 전처리를 적용해 실험 후, 성능 개선 시 포함하기로 함
    drop_cols = [
        "broadcast_id",
        "broadcast_duration",        "broadcast_datetime",
        "product_name",  # 텍스트 컬럼 제외
        "keyword",  # 텍스트 컬럼 제외
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


def fetch_product_info(product_codes: List[str]) -> pd.DataFrame:
    engine = create_engine(DB_URI)
    code_tuple = tuple(product_codes)
    query = (
        f"SELECT DISTINCT ON (product_code) * FROM {TABLE_NAME} "
        f"WHERE product_code IN {code_tuple}"
    )
    return pd.read_sql(query, engine)


def fetch_category_info() -> pd.DataFrame:
    """카테고리(대·중·소·세)별 평균 가격/방송시간 정보를 반환한다."""

    engine = create_engine(DB_URI)
    query = f"""
        SELECT 
            product_lgroup,
            product_mgroup,
            product_sgroup,
            product_dgroup,
            product_type,
            COALESCE(AVG(product_price), 0) AS product_price,
            AVG(broadcast_duration)  AS broadcast_duration
        FROM {TABLE_NAME}
        GROUP BY  product_lgroup, product_mgroup, product_sgroup, product_dgroup, product_type
    """
    return pd.read_sql(query, engine)


def prepare_candidate_row(
    date: dt.date,
    time_slot: str,
    product: pd.Series,
    weather_info: Dict[str, float],
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

    return {
        "broadcast_duration": product["broadcast_duration"],
        "temperature": weather_info["temperature"],
        "precipitation": weather_info["precipitation"],
        "product_price": product["product_price"],
        "weekday": _weekday_kr(date),
        "time_slot": time_slot,
        "time_slot_int": slot_int,
        "season": _season_kr(date.month),
        "weather": weather_info["weather"],
        "product_lgroup": product["product_lgroup"],
        "product_mgroup": product["product_mgroup"],
        "product_sgroup": product["product_sgroup"],
        "product_dgroup": product["product_dgroup"],
        "product_type": product["product_type"],
    }


def recommend(
    target_date: dt.date,
    time_slots: List[str],
    product_codes: List[str],
    weather_info: Dict[str, float],
    *,
    category_mode: bool = False,
    categories: List[str] | None = None,
) -> pd.DataFrame:
    """시간대별 최적 상품(또는 카테고리)을 추천한다.

    category_mode=True 이면 product_codes를 무시하고 카테고리 단위로 추천한다.
    """

    pipe: Pipeline = joblib.load(MODEL_FILE)

    if category_mode:
        items_df = fetch_category_info()
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

    candidates: List[Dict] = []
    for slot in time_slots:
        for _, item in items_df.iterrows():
            row_dict = prepare_candidate_row(target_date, slot, item, weather_info)
            pred = pipe.predict(pd.DataFrame([row_dict]))[0]
            candidates.append({
                "time_slot": slot,
                label_col: item[label_col],
                "predicted_sales": pred,
            })

    cand_df = pd.DataFrame(candidates)

    # 슬롯별로 예측 매출이 높은 후보를 선택하되, 전체 편성표에서 중복된 카테고리/상품이 나오지 않도록 한다.
    chosen_rows = []
    used_labels: set = set()

    for slot in time_slots:
        slot_df = cand_df[cand_df["time_slot"] == slot]
        if slot_df.empty:
            continue

        # 예측 매출 내림차순 정렬
        slot_df = slot_df.sort_values("predicted_sales", ascending=False)

        pick_row = None
        for _, row in slot_df.iterrows():
            label = row[label_col]
            if label not in used_labels:
                pick_row = row
                break

        # 모든 후보가 이미 사용됐다면 최고 매출 항목 선택
        if pick_row is None:
            pick_row = slot_df.iloc[0]

        chosen_rows.append(pick_row)
        used_labels.add(pick_row[label_col])

    best = pd.DataFrame(chosen_rows).reset_index(drop=True)
    return best

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
        )
        print("\n=== 추천 편성표 ===")
        print(rec_df.to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
