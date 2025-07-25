import datetime as dt
import zoneinfo  # 표준 타임존 처리 (Python 3.9+)
from typing import List

import streamlit as st
import pandas as pd
import broadcast_recommender as br
import json
import os
from openai import OpenAI
from functools import lru_cache

# OpenAI API 키는 환경변수 OPENAI_API_KEY 로 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Home Shopping Broadcast Recommender", page_icon="📺", layout="wide")

st.title("📺 홈쇼핑 방송편성 추천 챗봇")

# 초기 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []  # (role, content)

# 함수: OpenAI 로부터 파라미터 추출

def extract_params(user_msg: str) -> dict | None:
    """LLM을 호출해 파라미터(dict) 추출. 실패 시 None 반환"""

    # KST(Asia/Seoul) 고정 날짜·요일
    today_dt = dt.datetime.now(zoneinfo.ZoneInfo("Asia/Seoul"))
    today_date = today_dt.date()
    today_str = today_date.isoformat()
    today_weekday_kr = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"][today_date.weekday()]

    system_prompt = (
        "You are a machine that only returns JSON. Do not add any text before or after the JSON object. Your entire response must be only the JSON object itself.\n\n"
        "## Current Time Information\n"
        f"- Today's Date: {today_str} ({today_weekday_kr})\n\n"
        "## Instructions\n"
        "Extract the parameters from the user query and respond with a single JSON object that follows the schema below. If the user did not specify a value, use null (or an empty list for array types). Do not add any additional keys.\n\n"
        "{\n"
        "  \"date\": string | null,               # 방송 추천 대상 날짜 (YYYY-MM-DD)\n"
        "  \"time_slots\": string[] | null,       # 원하는 방송 시간대 배열 (예: [\"오전\", \"저녁\"])\n"
        "  \"weather\": string | null,            # 예상 날씨 (맑음/흐림 등)\n"
        "  \"temperature\": number | null,        # 평균 기온 (℃)\n"
        "  \"precipitation\": number | null,      # 예상 강수량 (mm)\n"
        "  \"season\": string | null,             # 계절 (봄/여름/가을/겨울)\n"
        "  \"day_type\": string | null,           # 평일/주말/공휴일\n"
        "  \"keywords\": string[] | null,         # 상품 키워드 배열\n"
        "  \"mode\": string | null,              # '카테고리' | '상품코드'\n"
        "  \"categories\": string[] | null,       # 카테고리 식별자 목록\n"
        "  \"products\": string[] | null,         # 상품코드 목록\n"
        "  \"gender\": string | null,             # 성별 (남성/여성)\n"
        "  \"age_group\": string | null           # 연령대 (예: '20대','30대','40대')\n"
        "}\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"파라미터 추출 실패: {e}")
        return None

# 캐시 유틸 ---------------------------------------------------------------
# ---------------------------------------------------------------------------

# --------------------------------- 헬퍼 ----------------------------------

TIME_SLOT_KEYWORDS = {
    "심야": ["심야", "새벽", "야간"],
    "아침": ["아침", "조식"],
    "오전": ["오전"],
    "점심": ["점심", "정오"],
    "오후": ["오후"],
    "저녁": ["저녁", "밤"],
}

def infer_time_slots(text: str) -> list[str] | None:
    """사용자 입력에서 대표 키워드를 찾아 time_slots 리스트 추정."""

    lowered = text.lower()
    hits: list[str] = []
    for slot, words in TIME_SLOT_KEYWORDS.items():
        if any(w in lowered for w in words):
            hits.append(slot)
    return hits or None

@st.cache_data(ttl=120, show_spinner=False)
def cached_recommend(
    target_date: dt.date,
    time_slots: list[str],
    product_codes: list[str],
    weather_info: dict,
    category_mode: bool,
    categories: list[str] | None,
    top_n: int = 1,
):
    """브로드캐스트 추천을 캐싱해 동일 요청 재호출 시 속도를 향상."""

    return br.recommend(
        target_date,
        time_slots,
        product_codes=product_codes,
        weather_info=weather_info,
        category_mode=category_mode,
        categories=categories,
        top_n=top_n,
    )

# 채팅 렌더링
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

# 입력창
if prompt := st.chat_input("편성 질문을 입력하세요…"):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append(("user", prompt))
    st.chat_message("user").write(prompt)

    # 파라미터 추출
    params = extract_params(prompt)
    assistant_msg = ""

    if params is None:
        assistant_msg = "죄송합니다. 파라미터를 이해하지 못했습니다. 다시 시도해 주세요."
        st.session_state.messages.append(("assistant", assistant_msg))
        st.chat_message("assistant").write(assistant_msg)
    else:
        # LLM이 time_slots 추출에 실패하면 휴리스틱으로 보정
        if params and not params.get("time_slots"):
            guess = infer_time_slots(prompt)
            if guess:
                params["time_slots"] = guess

        # 파라미터 JSON 먼저 사용자에게 즉시 보여주기 -----------------------------
        try:
            # ---------------- 날짜 파싱 ------------------------------------
            if params.get("date"):
                target_date = dt.date.fromisoformat(params["date"])
            else:
                # 날짜가 없고 요일/주말 언급만 있을 때 다음 해당 요일을 찾는다
                weekday_map = {
                    "월요일": 0,
                    "화요일": 1,
                    "수요일": 2,
                    "목요일": 3,
                    "금요일": 4,
                    "토요일": 5,
                    "일요일": 6,
                }

                kst_now = dt.datetime.now(zoneinfo.ZoneInfo("Asia/Seoul"))
                today = kst_now.date()
                widx = weekday_map.get(params.get("day_type"))
                if widx is not None:
                    days_ahead = (widx - today.weekday() + 7) % 7
                    days_ahead = 7 if days_ahead == 0 else days_ahead  # 다음 주 같은 요일
                    target_date = today + dt.timedelta(days=days_ahead)
                else:
                    # day_type이 "주말"/"평일"인 경우: 주말이면 다음 토요일, 평일이면 내일
                    if params.get("day_type") == "주말":
                        days_ahead = (5 - today.weekday()) % 7  # 토요일까지
                        days_ahead = 7 if days_ahead == 0 else days_ahead
                        target_date = today + dt.timedelta(days=days_ahead)
                    else:
                        target_date = today + dt.timedelta(days=1)  # 기본 내일
                # 추정한 날짜를 파라미터에도 반영
                params["date"] = target_date.isoformat()

            # --------- season 보정 ---------------------------------
            if not params.get("season"):
                def _infer_season(month: int) -> str:
                    if 3 <= month <= 5:
                        return "봄"
                    if 6 <= month <= 8:
                        return "여름"
                    if 9 <= month <= 11:
                        return "가을"
                    return "겨울"

                params["season"] = _infer_season(target_date.month)

            # ----- 파라미터 보정: time_slots, day_type ------------------------
            # time_slots가 없으면 전체 기본 슬롯 사용
            time_slots = params.get("time_slots") or [
                "아침",
                "오전",
                "점심",
                "오후",
                "저녁",
                "야간",
            ]

            # day_type이 없으면 평일/주말 계산
            if not params.get("day_type"):
                params["day_type"] = "주말" if target_date.weekday() >= 5 else "평일"

            # 날씨 기본 값 준비 (recommend 내부에서 보강될 수 있음)
            weather_info = {
                "weather": params.get("weather"),
                "temperature": params.get("temperature"),
                "precipitation": params.get("precipitation"),
            }

            # ---- 세부 단계 1: 날씨 정보 확인 -------------------------------
            with st.status("1/3 기본 파라미터 추출 중...", state="running") as w_status:  # type: ignore
                pass
            w_status.update(label="1/3 기본 파라미터 추출 완료", state="complete")  # type: ignore

            with st.status("2/3 날짜, 날씨 등 기타 정보 가져오는 중...", state="running") as w_status:  # type: ignore
                if not weather_info["weather"]:
                    fetched = br.get_weather_by_date(target_date)  # type: ignore
                    weather_info.update(fetched)
                w_status.update(label="2/3 날짜, 날씨 등 기타 정보 가져오기 완료", state="complete")  # type: ignore

            # 디스플레이용 파라미터 가공(날씨 갱신 포함)
            disp_params = params.copy()
            disp_params["weather"] = weather_info["weather"]
            disp_params["temperature"] = weather_info["temperature"]
            disp_params["precipitation"] = weather_info["precipitation"]
            disp_params["season"] = params.get("season")

            assistant_msg += (
                "### 추출된 파라미터\n````json\n"
                + json.dumps(disp_params, ensure_ascii=False, indent=2)
                + "\n````\n"
            )

            # 파라미터만 먼저 채팅에 표시
            st.session_state.messages.append(("assistant", assistant_msg))
            st.chat_message("assistant").write(assistant_msg)

            # 추천 결과는 placeholder에 나중에 채우기 -----------------------------
            result_placeholder = st.empty()

            with st.spinner("3/3 모델 예측 중..."):
                # 결과 테이블 초기화 (top_df가 정의되지 않을 경우 대비)
                top_df: pd.DataFrame | None = None

                # 상품코드를 주지 않았거나 모드가 "카테고리"이면 카테고리 추천으로 간주
                use_category = (
                    params.get("mode") == "카테고리" or not params.get("products")
                )

                # product_codes 결정 로직 -------------------------------
                product_codes: list[str] = params.get("products") or []

                # 제품 코드가 없고 키워드만 있을 때 키워드 → 상품코드 매핑
                if not use_category and not product_codes and params.get("keywords"):
                    product_codes = br.search_product_codes_by_keywords(params["keywords"])

                rec_df = cached_recommend(
                    target_date,
                    time_slots,
                    product_codes=product_codes,
                    weather_info=weather_info,
                    category_mode=use_category,
                    categories=params.get("categories"),
                    top_n=3,  # 시간대별 상위 3개 요청
                )

            # 스피너 종료 후 결과 표시
            # ----- 결과 포맷팅 및 한글 컬럼명 ------------------------------
            display_df = rec_df.copy()

            # 전체 카테고리 경로 생성 (대 > 중 > 소 > 세)
            cat_cols = ["product_lgroup", "product_mgroup", "product_sgroup", "product_dgroup"]
            # 데이터프레임에 존재하는 카테고리 컬럼만 필터링
            existing_cat_cols = [col for col in cat_cols if col in display_df.columns]

            if existing_cat_cols:
                display_df["전체 카테고리"] = display_df[existing_cat_cols].fillna('').agg(' > '.join, axis=1)
                # 양쪽 공백 및 '>' 문자 제거
                display_df["전체 카테고리"] = display_df["전체 카테고리"].str.strip(' > ').str.strip()

            # 숫자 -> 천단위 콤마 문자열 변환 (NumberColumn 포맷 오류 대응)
            if "predicted_sales" in display_df.columns:
                display_df["predicted_sales"] = (
                    display_df["predicted_sales"].round().astype(int).map("{:,}".format)
                )

            # 컬럼명 매핑
            col_name_map = {
                "time_slot": "시간대",
                "predicted_sales": "예상 매출(원)",
                "product_code": "상품코드",
                # "category": "카테고리", # 전체 카테고리 컬럼을 사용하므로 주석 처리
            }
            display_df = display_df.rename(columns={k: v for k, v in col_name_map.items() if k in display_df.columns})

            # 컬럼 정리: 불필요한 컬럼들 제거
            cols_to_drop = [
                "broadcast_tape_code", 
                "broadcast_showhost", 
                "category", # 기존 단일 카테고리 컬럼
            ] + cat_cols # 상세 카테고리 컬럼들
            
            for col in cols_to_drop:
                if col in display_df.columns:
                    display_df = display_df.drop(columns=[col])

            # 스피너 종료 후 결과 표시
            # 제목과 표를 하나의 컨테이너로 묶어 표시
            with result_placeholder.container():
                if not rec_df.empty:
                    st.subheader("📊 추천 편성표")
                    st.dataframe(
                        rec_df.drop(columns=["features"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    for index, row in rec_df.iterrows():
                        with st.expander(f"**{row['time_slot']} - {row['category']}** 상세 예측 정보 보기"):
                            st.json(row['features'])

                else:
                    st.warning("추천할 수 있는 상품을 찾지 못했습니다. 입력 조건을 변경해 보세요.")

        except Exception as e:
            assistant_msg = f"추천 실행 중 오류: {e}"
            st.session_state.messages.append(("assistant", assistant_msg))
            st.chat_message("assistant").write(assistant_msg)

# ---------------------------------------------------------------------------
# 사이드바 디버그 패널: 모델 피처 중요도 & 데이터 분포
# ---------------------------------------------------------------------------

import sys
from tokenizer_utils import mecab_tokenizer  # ensure module import for new models

# Compatibility shim for models pickled before tokenizer_utils module existed.
# They reference `mecab_tokenizer` inside the __main__ module that performed
# training (e.g., broadcast_recommender.py).  We expose the same symbol in the
# current top-level module so that joblib can resolve it during unpickling.
sys.modules[__name__].mecab_tokenizer = mecab_tokenizer

import joblib

with st.sidebar.expander("🛠️ 모델·데이터 통계", expanded=False):
    if st.button("Feature Importance / 분포 보기"):
        with st.spinner("모델 및 데이터 로딩 중..."):
            try:
                # 모델 로드 및 피처 중요도 계산
                pipe = br._load_model()  # type: ignore
                model = pipe.named_steps["model"]  # type: ignore
                importances = getattr(model, "feature_importances_", None)
                if importances is not None:
                    try:
                        feat_names = pipe.named_steps["pre"].get_feature_names_out()  # type: ignore
                    except Exception:
                        feat_names = [f"f{i}" for i in range(len(importances))]

                    imp_df = (
                        pd.DataFrame({"feature": feat_names, "importance": importances})
                        .sort_values("importance", ascending=False)
                        .head(40)
                    )
                    st.subheader("🔎 상위 Feature Importance (Top 40)")
                    st.bar_chart(imp_df.set_index("feature"))

                # 학습 데이터 분포 확인
                data_df = br.load_data()  # type: ignore
                if "time_slot_int" in data_df.columns:
                    st.subheader("📊 학습 데이터 시간대 분포 (time_slot_int)")
                    st.bar_chart(data_df["time_slot_int"].value_counts().sort_index())

                if "weekday" in data_df.columns:
                    st.subheader("📊 요일별 분포 (weekday)")
                    st.bar_chart(data_df["weekday"].value_counts())

            except Exception as ex:
                st.error(f"디버그 정보 생성 실패: {ex}")
