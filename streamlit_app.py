import datetime as dt
from typing import List

import streamlit as st
import pandas as pd

import broadcast_recommender as br

st.set_page_config(page_title="Home Shopping Broadcast Recommender", page_icon="📺", layout="wide")

st.title("📺 홈쇼핑 방송편성 추천 시스템")

# Sidebar inputs
with st.sidebar:
    st.header("🔧 입력 파라미터")

    target_date = st.date_input("방송 일자", value=dt.date.today() + dt.timedelta(days=1))

    slot_choices = ["아침", "오전", "점심", "오후", "저녁", "야간"]
    default_slots = ["아침", "점심", "저녁"]
    time_slots: List[str] = st.multiselect("편성 시간대", slot_choices, default_slots)

    st.subheader("🌦️ 날씨 정보")
    weather = st.selectbox("날씨", ["맑음", "흐림", "비", "눈"])
    temp = st.number_input("평균 기온(°C)", value=25.0)
    precip = st.number_input("강수량(mm)", value=0.0)

    mode = st.radio("추천 모드", ["카테고리", "상품코드"], horizontal=True)

    categories_input = st.text_input(
        "카테고리 목록 (콤마 구분)",
        help="대/중/소/세/product_type 형식, 예: 패션/의류/셔츠/반팔/일반상품"
    )
    products_input = st.text_input("상품코드 목록 (콤마 구분)")

    run_btn = st.button("🚀 추천 실행")

if run_btn:
    if not time_slots:
        st.error("최소 1개 이상의 시간대를 선택하세요.")
        st.stop()

    weather_info = {
        "weather": weather,
        "temperature": temp,
        "precipitation": precip,
    }

    try:
        if mode == "카테고리":
            categories = [c.strip() for c in categories_input.split(",") if c.strip()] if categories_input else None
            rec_df = br.recommend(
                target_date,
                time_slots,
                product_codes=[],
                weather_info=weather_info,
                category_mode=True,
                categories=categories,
            )
        else:
            products = [p.strip() for p in products_input.split(",") if p.strip()]
            if not products:
                st.error("상품코드를 입력하세요.")
                st.stop()
            rec_df = br.recommend(
                target_date,
                time_slots,
                product_codes=products,
                weather_info=weather_info,
                category_mode=False,
            )

        st.success("✅ 추천 완료")
        st.dataframe(rec_df, hide_index=True)

    except Exception as e:
        st.exception(e)
