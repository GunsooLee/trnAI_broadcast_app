import datetime as dt
from typing import List

import streamlit as st
import pandas as pd
import broadcast_recommender as br
import json
import os
from openai import OpenAI

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

    system_prompt = (
        "너는 홈쇼핑 방송 추천 시스템의 파라미터 추출기다.\n"
        "사용자 입력을 읽고 JSON 으로만 답해야 한다.\n"
        "필드는 date(YYYY-MM-DD), time_slots(list[str]), weather(str), temperature(float), precipitation(float), day_type(str: '평일'|'주말'|'공휴일'), keywords(list[str]), mode(str: '카테고리'|'상품코드'), categories(list[str]), products(list[str]).\n"
        "없는 값은 null 혹은 빈 리스트로 채워라."
    )

    today_str = dt.date.today().isoformat()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"오늘 날짜는 {today_str} 입니다."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"파라미터 추출 실패: {e}")
        return None

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
        # 필수 파라미터 확인
        try:
            target_date = dt.date.fromisoformat(params["date"])
            time_slots = params["time_slots"]
            weather_info = {
                "weather": params.get("weather"),
                "temperature": params.get("temperature"),
                "precipitation": params.get("precipitation"),
            }

            # None 값이면 recommender 내부에서 DB 조회
            if params["mode"] == "카테고리":
                rec_df = br.recommend(
                    target_date,
                    time_slots,
                    product_codes=[],
                    weather_info=weather_info,
                    category_mode=True,
                    categories=params.get("categories"),
                )
            else:
                rec_df = br.recommend(
                    target_date,
                    time_slots,
                    product_codes=params.get("products", []),
                    weather_info=weather_info,
                    category_mode=False,
                )

            if not weather_info["weather"]:
                # broadcast_recommender가 날씨 채움, 우리는 화면 표시용으로도 사용
                fetched = br.get_weather_by_date(target_date)  # type: ignore
                weather_info.update(fetched)

            # 디스플레이용 파라미터 보강
            params["weather"] = weather_info["weather"]
            params["temperature"] = weather_info["temperature"]
            params["precipitation"] = weather_info["precipitation"]

            # 다시 렌더링 파라미터 JSON 블록
            assistant_msg += (
                "### 최종 파라미터\n````json\n"
                + json.dumps(params, ensure_ascii=False, indent=2)
                + "\n````"
            )

            assistant_msg += "\n\n추천 결과:"  # 표시 후 아래 데이터프레임 렌더링
            st.session_state.messages.append(("assistant", assistant_msg))
            st.chat_message("assistant").write(assistant_msg)
            st.dataframe(rec_df, hide_index=True)

        except Exception as e:
            assistant_msg = f"추천 실행 중 오류: {e}"
            st.session_state.messages.append(("assistant", assistant_msg))
            st.chat_message("assistant").write(assistant_msg)
