import datetime as dt
import zoneinfo
import json
import os
from typing import Dict, Any, List

from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
import pandas as pd

# broadcast_recommender 모듈을 br로 임포트
from . import broadcast_recommender as br
from .schemas import RecommendResponse, RecommendationItem

# OpenAI API key is set as an environment variable OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper functions and logic from streamlit_app.py ---

TIME_SLOT_KEYWORDS = {
    "심야": ["심야", "새벽", "야간"], "아침": ["아침", "조식"], "오전": ["오전"],
    "점심": ["점심", "정오"], "오후": ["오후"], "저녁": ["저녁", "밤"],
}

def infer_time_slots(text: str) -> list[str] | None:
    """Infer time_slots list from user input using representative keywords."""
    lowered = text.lower()
    hits: list[str] = []
    for slot, words in TIME_SLOT_KEYWORDS.items():
        if any(w in lowered for w in words):
            hits.append(slot)
    return hits or None

def _infer_season(month: int) -> str:
    if 3 <= month <= 5: return "봄"
    if 6 <= month <= 8: return "여름"
    if 9 <= month <= 11: return "가을"
    return "겨울"

def extract_params_from_llm(user_msg: str) -> dict | None:
    """Call LLM to extract parameters as a dict."""
    today_dt = dt.datetime.now(zoneinfo.ZoneInfo("Asia/Seoul"))
    today_date = today_dt.date()
    today_str = today_date.isoformat()
    today_weekday_kr = ["월","화","수","목","금","토","일"][today_date.weekday()]

    system_prompt = (
        "You are a machine that only returns JSON. Do not add any text before or after the JSON object. Your entire response must be only the JSON object itself.\n\n"
        f"## Current Time Information\n- Today's Date: {today_str} ({today_weekday_kr})\n\n"
        "## Instructions\n"
        "Extract the parameters from the user query and respond with a single JSON object that follows the schema below. If the user did not specify a value, use null (or an empty list for array types). Do not add any additional keys.\n\n"
        "{\n"
        '  "date": string | null, "time_slots": string[] | null, "weather": string | null, '
        '  "temperature": number | null, "precipitation": number | null, "season": string | null, '
        '  "day_type": string | null, "keywords": string[] | null, "mode": string | null, '
        '  "categories": string[] | null, "products": string[] | null, "gender": string | null, "age_group": string | null\n'
        "}"
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
    except Exception:
        return None

def process_and_enrich_params(params: Dict[str, Any], user_query: str):
    """Process and enrich extracted parameters based on business logic."""
    
    if not params.get("time_slots"):
        if guess := infer_time_slots(user_query):
            params["time_slots"] = guess

    if params.get("date"):
        target_date = dt.date.fromisoformat(params["date"])
    else:
        weekday_map = {"월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, "금요일": 4, "토요일": 5, "일요일": 6}
        today = dt.datetime.now(zoneinfo.ZoneInfo("Asia/Seoul")).date()
        widx = weekday_map.get(params.get("day_type"))
        if widx is not None:
            days_ahead = (widx - today.weekday() + 7) % 7 or 7
            target_date = today + dt.timedelta(days=days_ahead)
        elif params.get("day_type") == "주말":
            days_ahead = (5 - today.weekday() + 7) % 7 or 7
            target_date = today + dt.timedelta(days=days_ahead)
        else:
            target_date = today + dt.timedelta(days=1)
        params["date"] = target_date.isoformat()

    if not params.get("season"):
        params["season"] = _infer_season(target_date.month)
    if not params.get("time_slots"):
        params["time_slots"] = ["아침", "오전", "점심", "오후", "저녁", "야간"]
    if not params.get("day_type"):
        params["day_type"] = "주말" if target_date.weekday() >= 5 else "평일"

    weather_info = {
        "weather": params.get("weather"),
        "temperature": params.get("temperature"),
        "precipitation": params.get("precipitation"),
    }
    if not weather_info["weather"]:
        fetched = br.get_weather_by_date(target_date)
        weather_info.update(fetched)
    
    params.update(weather_info)
    return params, target_date, weather_info


async def load_model_async():
    """서버 시작 시 모델을 비동기적으로 로드하기 위한 함수"""
    return await run_in_threadpool(br._load_model)


async def get_recommendations(user_query: str, model: br.Pipeline) -> RecommendResponse:
    print("--- 1. Recommendation service started ---")
    """
    Main service function to get recommendations from a user query.
    """
    # 동기 함수들을 run_in_threadpool을 사용하여 비동기적으로 실행
    params = await run_in_threadpool(extract_params_from_llm, user_query)
    if params is None:
        raise ValueError("Failed to extract parameters from the query.")

    enriched_params, target_date, weather_info = await run_in_threadpool(
        process_and_enrich_params, params, user_query
    )

    use_category = enriched_params.get("mode") != "상품" and (not enriched_params.get("products") or enriched_params.get("mode") == "카테고리")
    product_codes = enriched_params.get("products") or []
    if not use_category and not product_codes and enriched_params.get("keywords"):
        product_codes = await run_in_threadpool(br.search_product_codes_by_keywords, enriched_params["keywords"])

    print("--- 2. Calling broadcast recommender ---")
    # recommend 함수는 이제 동기적으로 호출해도 안전합니다 (모델 로딩이 없음).
    # 하지만 DB I/O 등 다른 동기 작업이 있을 수 있으므로 threadpool 사용을 유지합니다.
    rec_df = await run_in_threadpool(
        br.recommend,
        model=model, # 미리 로드된 모델을 전달
        target_date=target_date,
        time_slots=enriched_params["time_slots"],
        product_codes=product_codes,
        weather_info=weather_info,
        category_mode=use_category,
        categories=enriched_params.get("categories"),
        top_n=3,
        use_category_first=True,  # 카테고리 우선 방식 사용
        showhost_id="NO_HOST",   # 기본 쇼호스트 ID
    )

    recommendations = [RecommendationItem(**row) for row in rec_df.to_dict('records')] if not rec_df.empty else []

    return RecommendResponse(
        extracted_params=enriched_params,
        recommendations=recommendations
    )

async def extract_and_enrich_params(user_query: str) -> Dict[str, Any]:
    """
    사용자 질문에서 파라미터만 추출하고 보강합니다.
    """
    print("--- 1. Parameter extraction service started ---")
    
    # 동기 함수들을 run_in_threadpool을 사용하여 비동기적으로 실행
    params = await run_in_threadpool(extract_params_from_llm, user_query)
    if params is None:
        raise ValueError("Failed to extract parameters from the query.")

    enriched_params, target_date, weather_info = await run_in_threadpool(
        process_and_enrich_params, params, user_query
    )
    
    print("--- 2. Parameter extraction completed ---")
    return enriched_params

async def get_recommendations_with_params(params: Dict[str, Any], model: br.Pipeline) -> RecommendResponse:
    """
    수정된 파라미터로 추천을 생성합니다.
    """
    print("--- 1. Recommendation with params service started ---")
    
    # 파라미터에서 필요한 값들 추출
    target_date = dt.date.fromisoformat(params["date"])
    weather_info = {
        "weather": params.get("weather"),
        "temperature": params.get("temperature"),
        "precipitation": params.get("precipitation"),
    }
    
    use_category = params.get("mode") != "상품" and (not params.get("products") or params.get("mode") == "카테고리")
    product_codes = params.get("products") or []
    
    if not use_category and not product_codes and params.get("keywords"):
        product_codes = await run_in_threadpool(br.search_product_codes_by_keywords, params["keywords"])

    print("--- 2. Calling broadcast recommender with params ---")
    rec_df = await run_in_threadpool(
        br.recommend,
        model=model,
        target_date=target_date,
        time_slots=params["time_slots"],
        product_codes=product_codes,
        weather_info=weather_info,
        category_mode=use_category,
        categories=params.get("categories"),
        top_n=3,
        use_category_first=True,  # 카테고리 우선 방식 사용
        showhost_id="NO_HOST",   # 기본 쇼호스트 ID
    )

    recommendations = [RecommendationItem(**row) for row in rec_df.to_dict('records')] if not rec_df.empty else []

    return RecommendResponse(
        extracted_params=params,
        recommendations=recommendations
    )
