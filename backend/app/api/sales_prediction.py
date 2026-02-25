"""
매출 예측 API 엔드포인트
단일 상품 또는 날짜별 편성표 기반 매출 예측
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from datetime import datetime
import pandas as pd
import numpy as np

router = APIRouter(prefix="/api/v1/sales", tags=["sales"])
logger = logging.getLogger(__name__)


class SingleProductPredictionRequest(BaseModel):
    """단일 상품 매출 예측 요청"""
    tape_code: str = Field(..., description="방송 테이프 코드", example="0000012179")
    broadcast_start_time: str = Field(..., description="방송 시작 일시 (YYYY-MM-DD HH:MM:SS)", example="2026-02-22 14:00:00")
    broadcast_end_time: Optional[str] = Field(None, description="방송 종료 일시 (YYYY-MM-DD HH:MM:SS, 선택적)", example="2026-02-22 15:00:00")


class SingleProductPredictionResponse(BaseModel):
    """단일 상품 매출 예측 응답"""
    product_code: str
    product_name: str
    broadcast_datetime: str
    predicted_sales: float
    confidence: float
    features_used: dict


@router.post("/predict-single", response_model=SingleProductPredictionResponse)
async def predict_single_product_sales(payload: SingleProductPredictionRequest):
    """📊 단일 상품 매출 예측 API
    
    방송 테이프 코드와 방송 일시를 입력하면 해당 조건에서의 예상 매출을 예측합니다.
    train.py의 피처 엔지니어링 로직을 그대로 사용하여 정확한 예측을 제공합니다.
    """
    import time
    start_time = time.time()
    
    logger.info(f"단일 상품 예측 요청: {payload.tape_code} @ {payload.broadcast_start_time}")
    
    try:
        # 1. 방송 시작 시간 파싱 (초 단위 포함 또는 미포함 모두 지원)
        try:
            broadcast_dt = datetime.strptime(payload.broadcast_start_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                broadcast_dt = datetime.strptime(payload.broadcast_start_time, "%Y-%m-%d %H:%M")
            except ValueError:
                raise HTTPException(status_code=400, detail="방송 시작 일시 형식이 올바르지 않습니다. (YYYY-MM-DD HH:MM:SS)")
        
        # 2. 방송 길이 계산
        if payload.broadcast_end_time:
            # 종료 시간이 제공된 경우, duration 계산
            try:
                broadcast_end_dt = datetime.strptime(payload.broadcast_end_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    broadcast_end_dt = datetime.strptime(payload.broadcast_end_time, "%Y-%m-%d %H:%M")
                except ValueError:
                    raise HTTPException(status_code=400, detail="방송 종료 일시 형식이 올바르지 않습니다. (YYYY-MM-DD HH:MM:SS)")
            
            duration_seconds = (broadcast_end_dt - broadcast_dt).total_seconds()
            if duration_seconds <= 0:
                raise HTTPException(status_code=400, detail="방송 종료 시간이 시작 시간보다 이전입니다.")
            
            duration_minutes = int(duration_seconds / 60)
        else:
            # 종료 시간이 없으면 기본값 60분 사용
            duration_minutes = 60
        
        logger.info(f"방송 길이: {duration_minutes}분")
        
        # 3. DB 연결
        from train import get_db_engine
        engine = get_db_engine()
        
        # 3. 테이프 코드로 상품 코드 조회
        tape_query = f"""
        SELECT product_code
        FROM TAIPGMTAPE
        WHERE tape_code = '{payload.tape_code}'
        LIMIT 1
        """
        
        tape_df = pd.read_sql(tape_query, engine)
        
        if tape_df.empty:
            raise HTTPException(status_code=404, detail=f"방송 테이프를 찾을 수 없습니다: {payload.tape_code}")
        
        product_code = tape_df.iloc[0]['product_code']
        
        # 4. 상품 정보 조회
        product_query = f"""
        SELECT 
            product_code,
            product_name,
            category_main,
            category_middle,
            category_sub,
            price,
            brand,
            product_type
        FROM TAIGOODS
        WHERE product_code = '{product_code}'
        LIMIT 1
        """
        
        product_df = pd.read_sql(product_query, engine)
        
        if product_df.empty:
            raise HTTPException(status_code=404, detail=f"상품을 찾을 수 없습니다: {product_code}")
        
        product_info = product_df.iloc[0]
        
        # 5. 피처 생성 (train.py 로직 재사용)
        features = await _create_features_for_prediction(
            product_code=product_code,
            product_info=product_info,
            broadcast_dt=broadcast_dt,
            duration_minutes=duration_minutes,
            engine=engine
        )
        
        # 6. 모델 로드 및 예측
        import joblib
        from pathlib import Path
        
        model_path = Path(__file__).parent.parent / 'xgb_broadcast_profit.joblib'
        model = joblib.load(model_path)
        
        # DataFrame 생성
        pred_df = pd.DataFrame([features])
        
        # 예측 (로그 스케일)
        predicted_sales_log = model.predict(pred_df)[0]
        predicted_sales = np.expm1(predicted_sales_log)
        predicted_sales = max(0, predicted_sales)
        
        elapsed_time = time.time() - start_time
        logger.info(f"예측 완료: {predicted_sales:,.0f}원 (소요시간: {elapsed_time:.2f}초)")
        
        return SingleProductPredictionResponse(
            product_code=product_code,
            product_name=product_info['product_name'],
            broadcast_datetime=broadcast_dt.isoformat(),
            predicted_sales=float(predicted_sales),
            confidence=0.85,
            features_used={
                "tape_code": payload.tape_code,
                "category_main": features.get('category_main', ''),
                "time_slot": features.get('time_slot', ''),
                "is_weekend": features.get('is_weekend', False),
                "season": features.get('season', '')
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매출 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"매출 예측 중 오류가 발생했습니다: {str(e)}")


async def _create_features_for_prediction(
    product_code: str,
    product_info: pd.Series,
    broadcast_dt: datetime,
    duration_minutes: int,
    engine
) -> dict:
    """train.py의 피처 엔지니어링 로직을 재사용하여 예측용 피처 생성"""
    
    # 기본 시간 피처
    hour = broadcast_dt.hour
    minute = broadcast_dt.minute
    # 분을 반영한 정확한 시간 (예: 13:30 = 13.5, 13:45 = 13.75)
    hour_with_minute = hour + minute / 60.0
    month = broadcast_dt.month
    day_of_week_num = broadcast_dt.weekday()
    
    day_of_week_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    day_of_week = day_of_week_map[day_of_week_num]
    
    # 시간대
    if 0 <= hour < 6:
        time_slot = '새벽'
    elif 6 <= hour < 12:
        time_slot = '오전'
    elif 12 <= hour < 18:
        time_slot = '오후'
    else:
        time_slot = '저녁'
    
    # 계절
    if month in [3, 4, 5]:
        season = '봄'
    elif month in [6, 7, 8]:
        season = '여름'
    elif month in [9, 10, 11]:
        season = '가을'
    else:
        season = '겨울'
    
    # 주말 여부
    is_weekend = day_of_week_num in [5, 6]
    
    # 공휴일 확인
    holiday_query = f"""
    SELECT holiday_name
    FROM TAIHOLIDAYS
    WHERE holiday_date = '{broadcast_dt.date()}'
    LIMIT 1
    """
    holiday_df = pd.read_sql(holiday_query, engine)
    is_holiday = not holiday_df.empty
    holiday_name = holiday_df.iloc[0]['holiday_name'] if is_holiday else ''
    
    # 날씨 정보
    weather_query = f"""
    SELECT weather, temperature, precipitation
    FROM taiweather_daily
    WHERE weather_date = '{broadcast_dt.date()}'
    LIMIT 1
    """
    weather_df = pd.read_sql(weather_query, engine)
    
    if not weather_df.empty:
        weather = weather_df.iloc[0]['weather']
        temperature = weather_df.iloc[0]['temperature']
        precipitation = weather_df.iloc[0]['precipitation']
    else:
        weather = 'Clear'
        temperature = 15.0
        precipitation = 0.0
    
    # 상품 과거 실적 (누적 평균)
    past_performance_query = f"""
    SELECT 
        COALESCE(AVG(gross_profit), 0) as avg_profit,
        COUNT(*) as broadcast_count
    FROM broadcast_training_dataset
    WHERE product_code = '{product_code}'
      AND broadcast_date < '{broadcast_dt.date()}'
    """
    past_df = pd.read_sql(past_performance_query, engine)
    
    product_avg_profit = past_df.iloc[0]['avg_profit'] if not past_df.empty else 0
    product_broadcast_count = past_df.iloc[0]['broadcast_count'] if not past_df.empty else 0
    
    # 카테고리-시간대 평균
    category_timeslot_query = f"""
    SELECT COALESCE(AVG(gross_profit), 0) as avg_profit
    FROM broadcast_training_dataset
    WHERE category_middle = '{product_info['category_middle']}'
      AND time_slot = '{time_slot}'
      AND broadcast_date < '{broadcast_dt.date()}'
    """
    cat_time_df = pd.read_sql(category_timeslot_query, engine)
    category_timeslot_avg_profit = cat_time_df.iloc[0]['avg_profit'] if not cat_time_df.empty else 0
    
    # 카테고리 전체 평균
    category_overall_query = f"""
    SELECT COALESCE(AVG(gross_profit), 0) as avg_profit
    FROM broadcast_training_dataset
    WHERE category_middle = '{product_info['category_middle']}'
      AND broadcast_date < '{broadcast_dt.date()}'
    """
    cat_overall_df = pd.read_sql(category_overall_query, engine)
    category_overall_avg_profit = cat_overall_df.iloc[0]['avg_profit'] if not cat_overall_df.empty else 0
    
    # 시간대 특화 점수
    if category_overall_avg_profit > 0:
        timeslot_specialty_score = category_timeslot_avg_profit / category_overall_avg_profit
    else:
        timeslot_specialty_score = 1.0
    
    # 계절-카테고리 평균
    season_category_query = f"""
    SELECT COALESCE(AVG(gross_profit), 0) as avg_profit
    FROM broadcast_training_dataset
    WHERE category_middle = '{product_info['category_middle']}'
      AND season = '{season}'
      AND broadcast_date < '{broadcast_dt.date()}'
    """
    season_cat_df = pd.read_sql(season_category_query, engine)
    season_category_avg_profit = season_cat_df.iloc[0]['avg_profit'] if not season_cat_df.empty else 0
    
    # 계절 특화 점수
    if category_overall_avg_profit > 0:
        season_category_specialty_score = season_category_avg_profit / category_overall_avg_profit
    else:
        season_category_specialty_score = 1.0
    
    # 계절 가중치 (경계기 보정)
    spring_weight = 0.0
    summer_weight = 0.0
    autumn_weight = 0.0
    winter_weight = 0.0
    
    if month == 2:
        winter_weight = 0.7
        spring_weight = 0.3
    elif month == 3:
        winter_weight = 0.3
        spring_weight = 0.7
    elif month == 5:
        spring_weight = 0.7
        summer_weight = 0.3
    elif month == 6:
        spring_weight = 0.3
        summer_weight = 0.7
    elif month == 8:
        summer_weight = 0.7
        autumn_weight = 0.3
    elif month == 9:
        summer_weight = 0.3
        autumn_weight = 0.7
    elif month == 11:
        autumn_weight = 0.7
        winter_weight = 0.3
    elif month == 12:
        autumn_weight = 0.3
        winter_weight = 0.7
    else:
        if season == '봄':
            spring_weight = 1.0
        elif season == '여름':
            summer_weight = 1.0
        elif season == '가을':
            autumn_weight = 1.0
        else:
            winter_weight = 1.0
    
    # 가격 로그 스케일링
    price = product_info.get('price', 0)
    product_price_log = np.log1p(price) if price > 0 else 0
    
    # 방송 길이 로그 스케일링
    duration_log = np.log1p(duration_minutes)
    
    # 시간 사인/코사인 변환 (분까지 반영)
    hour_sin = np.sin(2 * np.pi * hour_with_minute / 24)
    hour_cos = np.cos(2 * np.pi * hour_with_minute / 24)
    
    # 월 사인/코사인 변환
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # 상호작용 피처
    time_category_interaction = f"{time_slot}_{product_info['category_middle']}"
    season_category_interaction = f"{season}_{product_info['category_middle']}"
    
    # 키워드 피처 (상품명 기반)
    product_name = product_info['product_name']
    
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
    
    number_patterns = ['1+1', '2+1', '6+3', '10+2', '6+6', '5+5', '6+1', '12+6', '20+1']
    
    keyword_features = {}
    for keyword in high_impact_keywords:
        keyword_features[f'kw_{keyword}'] = 1 if keyword in product_name else 0
    
    for pattern in number_patterns:
        safe_pattern = pattern.replace('+', '_')
        keyword_features[f'kw_{safe_pattern}'] = 1 if pattern in product_name else 0
    
    # 최종 피처 딕셔너리
    features = {
        # 가격
        'product_price_log': product_price_log,
        
        # 상품 과거 실적
        'product_avg_profit': product_avg_profit,
        'product_broadcast_count': product_broadcast_count,
        
        # 시간 피처 (분 단위까지 반영)
        'hour_with_minute': hour_with_minute,
        'hour_with_minute_sin': hour_sin,
        'hour_with_minute_cos': hour_cos,
        
        # 월 피처
        'month': month,
        'month_sin': month_sin,
        'month_cos': month_cos,
        
        # 카테고리-시간대 피처
        'category_timeslot_avg_profit': category_timeslot_avg_profit,
        'timeslot_specialty_score': timeslot_specialty_score,
        
        # 계절 피처
        'season_category_specialty_score': season_category_specialty_score,
        'spring_weight': spring_weight,
        'summer_weight': summer_weight,
        'autumn_weight': autumn_weight,
        'winter_weight': winter_weight,
        
        # 방송 길이 피처
        'duration_minutes': duration_minutes,
        'duration_log': duration_log,
        
        # 카테고리 피처
        'category_main': product_info['category_main'],
        'category_middle': product_info['category_middle'],
        'category_sub': product_info['category_sub'],
        
        # 브랜드/타입
        'brand': product_info.get('brand', 'Unknown'),
        'product_type': product_info.get('product_type', '유형'),
        
        # 시간대/요일
        'time_slot': time_slot,
        'day_of_week': day_of_week,
        
        # 상호작용 피처
        'time_category_interaction': time_category_interaction,
        'season_category_interaction': season_category_interaction,
        
        # 주말/공휴일
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        
        # 계절
        'season': season,
    }
    
    # 키워드 피처 추가
    features.update(keyword_features)
    
    return features
