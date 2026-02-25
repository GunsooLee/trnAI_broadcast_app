#!/usr/bin/env python3
"""
오늘 방송의 실제 매출과 예측 매출 비교
"""

import pandas as pd
import numpy as np

# 실제 매출 데이터 (사용자 제공)
actual_data = [
    {"no": 1, "time": "00:52", "product": "투라 리프팅 여신 속눈썹 10박스", "actual_sales": 16920000, "orders": 242},
    {"no": 2, "time": "01:52", "product": "[1등염색제] 최신상 모나리스타 컬러크림", "actual_sales": 7890000, "orders": 100},
    {"no": 3, "time": "02:51", "product": "[방송에서만]지니라이프 듀얼 맥스 콘드로이친", "actual_sales": 8810000, "orders": 89},
    {"no": 4, "time": "03:57", "product": "안국약품 토비콤 루테인 지아잔틴", "actual_sales": 952000, "orders": 4},
    {"no": 5, "time": "04:16", "product": "뉴트리원 카무트효소 골드 12박스", "actual_sales": 660000, "orders": 2},
    {"no": 6, "time": "04:37", "product": "욕실의여왕 고체형 실내탈취제", "actual_sales": 7720000, "orders": 92},
    {"no": 7, "time": "05:36", "product": "[엘르스포츠] ELLE SPORT 여성 반터틀넥 니트", "actual_sales": 29760000, "orders": 432},
    {"no": 8, "time": "06:36", "product": "슈리오 소가죽 밸런스 키높이 스니커즈", "actual_sales": 58340000, "orders": 1084},
    {"no": 9, "time": "07:36", "product": "리비에라앤바 메디포트 달임마스터", "actual_sales": 20900000, "orders": 162},
    {"no": 10, "time": "08:35", "product": "[단독구성] 배한호 원장의 알부민", "actual_sales": 33860000, "orders": 114},
]

# 예측 데이터 로드
predicted_df = pd.read_csv('/app/today_broadcast_predictions.csv')

# 실제 데이터 DataFrame 생성
actual_df = pd.DataFrame(actual_data)

# 예측 데이터에서 필요한 컬럼만 추출 (no 기준으로 매칭)
comparison = actual_df.merge(
    predicted_df[['no', 'tape_code', 'product_name', 'predicted_sales']], 
    on='no', 
    how='left'
)

# 오차 계산
comparison['error'] = comparison['actual_sales'] - comparison['predicted_sales']
comparison['error_pct'] = (comparison['error'] / comparison['actual_sales']) * 100
comparison['abs_error_pct'] = np.abs(comparison['error_pct'])

# 결과 출력
print("=" * 100)
print("📊 오늘(2026-02-25) 방송 실제 매출 vs 예측 매출 비교")
print("=" * 100)
print()

for idx, row in comparison.iterrows():
    print(f"{row['no']:2d}. {row['time']} - {row['product'][:50]}")
    print(f"    실제 매출: {row['actual_sales']:>12,d}원 | 예측 매출: {row['predicted_sales']:>12,.0f}원")
    print(f"    오차: {row['error']:>12,.0f}원 ({row['error_pct']:>6.1f}%)")
    
    # 정확도 평가
    if abs(row['error_pct']) <= 10:
        accuracy = "🟢 매우 정확"
    elif abs(row['error_pct']) <= 20:
        accuracy = "🟢 정확"
    elif abs(row['error_pct']) <= 30:
        accuracy = "🟡 양호"
    elif abs(row['error_pct']) <= 50:
        accuracy = "🟡 보통"
    else:
        accuracy = "🔴 부정확"
    
    print(f"    평가: {accuracy}")
    print()

# 전체 통계
print("=" * 100)
print("📈 전체 통계")
print("=" * 100)
print()

total_actual = comparison['actual_sales'].sum()
total_predicted = comparison['predicted_sales'].sum()
mae = comparison['error'].abs().mean()
rmse = np.sqrt((comparison['error'] ** 2).mean())
mape = comparison['abs_error_pct'].mean()

# R² 계산
ss_res = np.sum(comparison['error'] ** 2)
ss_tot = np.sum((comparison['actual_sales'] - comparison['actual_sales'].mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"총 실제 매출:  {total_actual:>15,d}원")
print(f"총 예측 매출:  {total_predicted:>15,.0f}원")
print(f"총 오차:       {total_actual - total_predicted:>15,.0f}원 ({((total_actual - total_predicted)/total_actual)*100:>6.1f}%)")
print()
print(f"MAE  (평균 절대 오차):        {mae:>12,.0f}원")
print(f"RMSE (평균 제곱근 오차):      {rmse:>12,.0f}원")
print(f"MAPE (평균 절대 백분율 오차): {mape:>12.1f}%")
print(f"R²   (결정 계수):             {r2:>12.4f}")
print()

# 정확도 구간별 분포
print("=" * 100)
print("정확도 구간별 분포")
print("=" * 100)
print()

ranges = [
    (0, 10, "±10% 이내"),
    (10, 20, "±10~20%"),
    (20, 30, "±20~30%"),
    (30, 50, "±30~50%"),
    (50, 100, "±50% 이상")
]

for min_pct, max_pct, label in ranges:
    count = len(comparison[(comparison['abs_error_pct'] >= min_pct) & (comparison['abs_error_pct'] < max_pct)])
    percentage = (count / len(comparison)) * 100
    print(f"{label:15s}: {count:2d}건 ({percentage:5.1f}%)")

print()

# 가장 정확한 예측
print("=" * 100)
print("가장 정확한 예측 Top 3")
print("=" * 100)
print()

top_3 = comparison.nsmallest(3, 'abs_error_pct')
for i, (idx, row) in enumerate(top_3.iterrows(), 1):
    print(f"{i}. {row['product'][:50]}")
    print(f"   실제: {row['actual_sales']:,d}원 | 예측: {row['predicted_sales']:,.0f}원 | 오차: {row['error_pct']:.1f}%")
    print()

# 가장 부정확한 예측
print("=" * 100)
print("가장 부정확한 예측 Top 3")
print("=" * 100)
print()

worst_3 = comparison.nlargest(3, 'abs_error_pct')
for i, (idx, row) in enumerate(worst_3.iterrows(), 1):
    print(f"{i}. {row['product'][:50]}")
    print(f"   실제: {row['actual_sales']:,d}원 | 예측: {row['predicted_sales']:,.0f}원 | 오차: {row['error_pct']:.1f}%")
    print()

# CSV 저장
output_file = "/app/today_actual_vs_predicted.csv"
comparison.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 비교 결과 저장: {output_file}")
print()
