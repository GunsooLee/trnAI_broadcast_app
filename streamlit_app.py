import datetime as dt
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

    today = dt.date.today()
    today_str = today.isoformat()
    today_weekday_kr = ["월","화","수","목","금","토","일"][today.weekday()]

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

@st.cache_data(ttl=600, show_spinner=False)
def cached_recommend(
    target_date: dt.date,
    time_slots: list[str],
    product_codes: list[str],
    weather_info: dict,
    category_mode: bool,
    categories: list[str] | None,
):
    """브로드캐스트 추천을 캐싱해 동일 요청 재호출 시 속도를 향상."""

    return br.recommend(
        target_date,
        time_slots,
        product_codes=product_codes,
        weather_info=weather_info,
        category_mode=category_mode,
        categories=categories,
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
        # 파라미터 JSON 먼저 사용자에게 즉시 보여주기 -----------------------------
        try:
            target_date = dt.date.fromisoformat(params["date"])

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
            with st.status("1/3 파라미터 추출 중...", state="running") as w_status:  # type: ignore
                pass
            w_status.update(label="1/3 파라미터 추출 완료", state="complete")  # type: ignore

            with st.status("2/3 날짜, 날씨 등 기타 정보 확인 중...", state="running") as w_status:  # type: ignore
                if not weather_info["weather"]:
                    fetched = br.get_weather_by_date(target_date)  # type: ignore
                    weather_info.update(fetched)
                w_status.update(label="2/3 날짜, 날씨 등 기타 정보 확인 완료", state="complete")  # type: ignore

            # 디스플레이용 파라미터 가공(날씨 갱신 포함)
            disp_params = params.copy()
            disp_params["weather"] = weather_info["weather"]
            disp_params["temperature"] = weather_info["temperature"]
            disp_params["precipitation"] = weather_info["precipitation"]

            assistant_msg += (
                "### 1/3 추출된 파라미터\n````json\n"
                + json.dumps(disp_params, ensure_ascii=False, indent=2)
                + "\n````\n"
            )

            # 파라미터만 먼저 채팅에 표시
            st.session_state.messages.append(("assistant", assistant_msg))
            st.chat_message("assistant").write(assistant_msg)

            # 추천 결과는 placeholder에 나중에 채우기 -----------------------------
            result_placeholder = st.empty()

            with st.spinner("3/3 모델 예측 중..."):
                # 상품코드를 주지 않았거나 모드가 "카테고리"이면 카테고리 추천으로 간주
                use_category = (
                    params.get("mode") == "카테고리" or not params.get("products")
                )

                if use_category:
                    rec_df = cached_recommend(
                        target_date,
                        time_slots,
                        product_codes=[],
                        weather_info=weather_info,
                        category_mode=True,
                        categories=params.get("categories"),
                    )
                else:
                    rec_df = cached_recommend(
                        target_date,
                        time_slots,
                        product_codes=params.get("products", []),
                        weather_info=weather_info,
                        category_mode=False,
                        categories=None,
                    )

            # 스피너 종료 후 결과 표시
            # ----- 결과 포맷팅 및 한글 컬럼명 ------------------------------
            display_df = rec_df.copy()

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
                "category": "카테고리",
            }
            display_df = display_df.rename(columns={k: v for k, v in col_name_map.items() if k in display_df.columns})

            # 스피너 종료 후 결과 표시
            # 제목과 표를 하나의 컨테이너로 묶어 표시
            with result_placeholder.container():
                st.markdown("### 📊 매출 예측 결과")
                st.dataframe(display_df, hide_index=True)

        except Exception as e:
            assistant_msg = f"추천 실행 중 오류: {e}"
            st.session_state.messages.append(("assistant", assistant_msg))
            st.chat_message("assistant").write(assistant_msg)
