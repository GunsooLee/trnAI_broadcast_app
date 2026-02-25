#!/usr/bin/env python3
"""
최근 3일치 방송 데이터로 모델 예측 정확도 평가
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import joblib
from pathlib import Path
import sys

# train.py에서 피처 생성 함수 임포트
sys.path.append('/app')
from train import get_db_engine, load_data, build_pipeline


def get_recent_broadcasts(engine, days=3):
    """최근 N일간의 방송 데이터 조회"""
    query = f"""
    SELECT 
        b.tape_code,
        t.product_code,
        b.broadcast_start_timestamp,
        b.broadcast_end_timestamp,
        b.duration_minutes,
        b.gross_profit as actual_sales,
        b.sales_efficiency,
        g.product_name,
        g.category_main,
        g.category_middle,
        g.category_sub,
        g.price
    FROM TAIBROADCASTS b
    JOIN TAIPGMTAPE t ON b.tape_code = t.tape_code
    JOIN TAIGOODS g ON t.product_code = g.product_code
    WHERE b.broadcast_start_timestamp >= CURRENT_DATE - INTERVAL '{days} days'
      AND b.gross_profit IS NOT NULL
      AND g.category_main IS NOT NULL
    ORDER BY b.broadcast_start_timestamp DESC
    """
    
    df = pd.read_sql(query, engine)
    print(f"✅ 최근 {days}일간 방송 데이터: {len(df)}건")
    print(f"   기간: {df['broadcast_start_timestamp'].min()} ~ {df['broadcast_start_timestamp'].max()}")
    print(f"   상품 수: {df['product_code'].nunique()}개")
    print()
    
    return df


def create_features_for_evaluation(df, engine):
    """평가용 피처 생성 (train.py 로직 완전 재현)"""
    
    print("피처 생성 중...")
    
    # 1. 날짜/시간 피처
    df['broadcast_date'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.date
    df['hour'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.hour
    df['minute'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.minute
    df['day_of_week'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.day_name()
    df['month'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.month
    
    # 2. hour_with_minute (분 단위 포함)
    df['hour_with_minute'] = df['hour'] + df['minute'] / 60.0
    df['hour_with_minute_sin'] = np.sin(2 * np.pi * df['hour_with_minute'] / 24)
    df['hour_with_minute_cos'] = np.cos(2 * np.pi * df['hour_with_minute'] / 24)
    
    # 3. month 사인/코사인 변환
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 4. 시간대
    df['time_slot'] = df['hour'].apply(lambda h: 
        'morning' if 6 <= h < 12 else
        'afternoon' if 12 <= h < 18 else
        'evening' if 18 <= h < 23 else 'night'
    )
    
    # 5. 계절
    df['season'] = df['month'].apply(lambda m:
        'spring' if m in [3, 4, 5] else
        'summer' if m in [6, 7, 8] else
        'fall' if m in [9, 10, 11] else 'winter'
    )
    
    # 6. 주말 여부
    df['is_weekend'] = pd.to_datetime(df['broadcast_start_timestamp']).dt.dayofweek.isin([5, 6])
    
    # 7. 공휴일 (기본값 False)
    df['is_holiday'] = False
    
    # 8. brand, product_type (기본값)
    df['brand'] = df.get('brand', 'Unknown')
    df['product_type'] = df.get('product_type', 'Unknown')
    
    # 9. 가격 로그 스케일링
    df['product_price'] = df['price'].fillna(df['price'].median())
    df['product_price_log'] = np.log1p(df['product_price'])
    
    # 10. duration 피처
    df['duration_minutes'] = df['duration_minutes'].fillna(60)
    df['duration_log'] = np.log1p(df['duration_minutes'])
    
    # 11. 상품별 과거 통계 (전체 학습 데이터에서 계산)
    product_stats_query = """
    SELECT 
        product_code,
        AVG(gross_profit) as product_avg_profit,
        COUNT(*) as product_broadcast_count
    FROM broadcast_training_dataset
    GROUP BY product_code
    """
    product_stats = pd.read_sql(product_stats_query, engine)
    df = df.merge(product_stats, on='product_code', how='left')
    df['product_avg_profit'] = df['product_avg_profit'].fillna(df['actual_sales'].median())
    df['product_broadcast_count'] = df['product_broadcast_count'].fillna(1)
    
    # 12. 카테고리-시간대별 평균 매출
    category_timeslot_query = """
    SELECT 
        category_main,
        time_slot,
        AVG(gross_profit) as category_timeslot_avg_profit
    FROM broadcast_training_dataset
    GROUP BY category_main, time_slot
    """
    category_timeslot_stats = pd.read_sql(category_timeslot_query, engine)
    df = df.merge(category_timeslot_stats, on=['category_main', 'time_slot'], how='left')
    df['category_timeslot_avg_profit'] = df['category_timeslot_avg_profit'].fillna(df['actual_sales'].median())
    
    # 13. 시간대 특화 점수
    overall_avg = df['actual_sales'].mean()
    df['timeslot_specialty_score'] = df['category_timeslot_avg_profit'] / overall_avg
    
    # 14. 계절-카테고리 특화 피처
    season_category_query = """
    SELECT 
        season,
        category_main,
        AVG(gross_profit) as season_category_avg_profit
    FROM broadcast_training_dataset
    GROUP BY season, category_main
    """
    season_category_stats = pd.read_sql(season_category_query, engine)
    df = df.merge(season_category_stats, on=['season', 'category_main'], how='left')
    df['season_category_specialty_score'] = df['season_category_avg_profit'].fillna(overall_avg) / overall_avg
    
    # 15. 계절별 가중치 (경계기 처리)
    df['spring_weight'] = df['month'].apply(lambda m: 0.3 if m == 2 else (0.7 if m == 5 else (1.0 if m in [3,4] else 0.0)))
    df['summer_weight'] = df['month'].apply(lambda m: 0.3 if m == 5 else (0.7 if m == 8 else (1.0 if m in [6,7] else 0.0)))
    df['autumn_weight'] = df['month'].apply(lambda m: 0.3 if m == 8 else (0.7 if m == 11 else (1.0 if m in [9,10] else 0.0)))
    df['winter_weight'] = df['month'].apply(lambda m: 0.3 if m == 11 else (0.7 if m == 2 else (1.0 if m in [12,1] else 0.0)))
    
    # 16. 인터랙션 피처
    df['time_category_interaction'] = df['time_slot'] + '_' + df['category_main']
    df['season_category_interaction'] = df['season'] + '_' + df['category_main']
    
    # 17. 키워드 피처 (train.py와 동일)
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
    
    import re
    for keyword in high_impact_keywords:
        feature_name = f'kw_{keyword}'
        df[feature_name] = df['product_name'].fillna('').str.contains(keyword, na=False).astype(int)
    
    for pattern in number_patterns:
        feature_name = f'kw_{pattern.replace("+", "_")}'
        df[feature_name] = df['product_name'].str.contains(re.escape(pattern), na=False).astype(int)
    
    print(f"✅ 피처 생성 완료: {len(df)}건")
    print()
    
    return df


def evaluate_predictions(df, model_path, engine):
    """모델 예측 및 평가"""
    
    print("=" * 70)
    print("모델 예측 및 평가")
    print("=" * 70)
    
    # 모델 로드
    model = joblib.load(model_path)
    print(f"✅ 모델 로드: {model_path}")
    
    # 피처 생성
    df_features = create_features_for_evaluation(df.copy(), engine)
    
    # 예측에 필요한 피처 컬럼 (train.py와 완전히 동일)
    numeric_features = [
        "product_price_log",
        "product_avg_profit",
        "product_broadcast_count",
        "hour_with_minute",
        "hour_with_minute_sin",
        "hour_with_minute_cos",
        "month",
        "month_sin",
        "month_cos",
        "category_timeslot_avg_profit",
        "timeslot_specialty_score",
        "season_category_specialty_score",
        "spring_weight",
        "summer_weight",
        "autumn_weight",
        "winter_weight",
        "duration_minutes",
        "duration_log",
    ]
    
    keyword_features = [
        'kw_특집방송', 'kw_두유대장', 'kw_두유제조기', 'kw_에어텔닷컴', 'kw_하나투어',
        'kw_마데카', 'kw_릴렉스', 'kw_가을', 'kw_고객감사', 'kw_추석',
        'kw_에버홈', 'kw_단하루', 'kw_동국제약', 'kw_비에날씬', 'kw_노랑풍선',
        'kw_임성근의', 'kw_여행', 'kw_갈비', 'kw_국내산', 'kw_크림',
        'kw_슬림', 'kw_지아잔틴', 'kw_배한호', 'kw_토비콤', 'kw_르까프',
        'kw_콘드로이친', 'kw_티셔츠', 'kw_카무트', 'kw_팬츠', 'kw_효소',
        'kw_흑염소진액', 'kw_세일', 'kw_첫날부터', 'kw_다재다능', 'kw_흥국생명',
        'kw_루테인', 'kw_암보험',
        'kw_1_1', 'kw_2_1', 'kw_6_3', 'kw_10_2', 'kw_6_6',
        'kw_5_5', 'kw_6_1', 'kw_12_6', 'kw_20_1'
    ]
    
    numeric_features.extend(keyword_features)
    
    categorical_features = [
        "category_main", "category_middle", "category_sub",
        "brand", "product_type", "time_slot", "day_of_week",
        "time_category_interaction", "season_category_interaction",
    ]
    
    boolean_features = ["is_weekend", "is_holiday"]
    
    # 모든 피처 결합
    feature_columns = numeric_features + categorical_features + boolean_features
    
    # 예측 실행
    X = df_features[feature_columns]
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)  # 로그 역변환
    
    # 실제값
    y_actual = df_features['actual_sales'].values
    
    # 평가 지표 계산
    mae = np.mean(np.abs(y_actual - y_pred))
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    
    # R² 계산
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print()
    print("=" * 70)
    print("📊 예측 정확도 평가 결과")
    print("=" * 70)
    print(f"총 방송 건수: {len(df_features)}건")
    print(f"평가 기간: {df_features['broadcast_start_timestamp'].min()} ~ {df_features['broadcast_start_timestamp'].max()}")
    print()
    print(f"MAE  (평균 절대 오차)    : {mae:,.0f}원")
    print(f"RMSE (평균 제곱근 오차)  : {rmse:,.0f}원")
    print(f"MAPE (평균 절대 백분율 오차): {mape:.1f}%")
    print(f"R²   (결정 계수)         : {r2:.4f}")
    print()
    
    # 실제값 vs 예측값 통계
    print("=" * 70)
    print("실제 매출 vs 예측 매출 비교")
    print("=" * 70)
    print(f"실제 매출 평균: {np.mean(y_actual):,.0f}원")
    print(f"예측 매출 평균: {np.mean(y_pred):,.0f}원")
    print(f"실제 매출 중앙값: {np.median(y_actual):,.0f}원")
    print(f"예측 매출 중앙값: {np.median(y_pred):,.0f}원")
    print()
    
    # 오차 분포
    errors = y_actual - y_pred
    error_pct = (errors / y_actual) * 100
    
    print("=" * 70)
    print("오차 분포")
    print("=" * 70)
    print(f"오차 평균: {np.mean(errors):,.0f}원 ({np.mean(error_pct):.1f}%)")
    print(f"오차 중앙값: {np.median(errors):,.0f}원 ({np.median(error_pct):.1f}%)")
    print(f"오차 표준편차: {np.std(errors):,.0f}원")
    print()
    
    # 정확도 구간별 분포
    print("=" * 70)
    print("예측 정확도 구간별 분포")
    print("=" * 70)
    
    accuracy_ranges = [
        (0, 10, "±10% 이내"),
        (10, 20, "±10~20%"),
        (20, 30, "±20~30%"),
        (30, 50, "±30~50%"),
        (50, 100, "±50% 이상")
    ]
    
    for min_pct, max_pct, label in accuracy_ranges:
        count = np.sum((np.abs(error_pct) >= min_pct) & (np.abs(error_pct) < max_pct))
        percentage = (count / len(error_pct)) * 100
        print(f"{label:15s}: {count:3d}건 ({percentage:5.1f}%)")
    
    print()
    
    # 상위/하위 예측 사례
    print("=" * 70)
    print("예측이 가장 정확한 상위 10건 (오차가 가장 적은 건)")
    print("=" * 70)
    
    # inf 값 제거하고 절대값 기준으로 정렬
    df_valid = df_results[np.isfinite(df_results['error_pct'])].copy()
    df_valid['abs_error_pct'] = np.abs(df_valid['error_pct'])
    
    top_accurate = df_valid.nsmallest(10, 'abs_error_pct')[
        ['broadcast_start_timestamp', 'product_name', 'actual_sales', 'predicted_sales', 'error_pct']
    ]
    
    for i, (idx, row) in enumerate(top_accurate.iterrows(), 1):
        print(f"{i}. {row['broadcast_start_timestamp']}")
        print(f"   상품: {row['product_name'][:60]}")
        print(f"   실제: {row['actual_sales']:,.0f}원 | 예측: {row['predicted_sales']:,.0f}원 | 오차: {row['error_pct']:.1f}%")
        print()
    
    print("=" * 70)
    print("예측 오차가 가장 큰 상위 5건")
    print("=" * 70)
    
    df_results['abs_error_pct'] = np.abs(df_results['error_pct'])
    worst_accurate = df_results.nlargest(5, 'abs_error_pct')[
        ['broadcast_start_timestamp', 'product_name', 'actual_sales', 'predicted_sales', 'error_pct']
    ]
    
    for idx, row in worst_accurate.iterrows():
        print(f"• {row['broadcast_start_timestamp']}")
        print(f"  상품: {row['product_name'][:50]}")
        print(f"  실제: {row['actual_sales']:,.0f}원 | 예측: {row['predicted_sales']:,.0f}원 | 오차: {row['error_pct']:.1f}%")
        print()
    
    return df_results


def main():
    """메인 실행 함수"""
    
    print("\n" + "=" * 70)
    print("🎯 최근 3일 방송 데이터 예측 정확도 평가")
    print("=" * 70)
    print()
    
    # DB 연결
    engine = get_db_engine()
    
    # 최근 3일 데이터 조회
    df = get_recent_broadcasts(engine, days=3)
    
    if len(df) == 0:
        print("❌ 평가할 데이터가 없습니다.")
        return
    
    # 모델 경로
    model_path = Path('/app/app/xgb_broadcast_profit.joblib')
    
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 예측 및 평가
    df_results = evaluate_predictions(df, model_path, engine)
    
    # 결과 저장 (선택적)
    output_path = Path('/app/recent_predictions_evaluation.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 평가 결과 저장: {output_path}")
    print()


if __name__ == "__main__":
    main()
