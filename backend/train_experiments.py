#!/usr/bin/env python3
"""
다양한 방법으로 모델 성능 개선 실험
1. 최근 1년 데이터만 사용
2. Sample Weight 적용
3. 조합
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
from pathlib import Path
import sys

# train.py의 함수들 import
sys.path.append('/app')
from train import load_data, build_pipeline

print("=" * 100)
print("🧪 모델 성능 개선 실험")
print("=" * 100)
print()

# 실제 매출 데이터 (테스트용)
actual_data = [
    {"no": 1, "actual_sales": 16920000, "actual_qty": 242},
    {"no": 2, "actual_sales": 7890000, "actual_qty": 100},
    {"no": 3, "actual_sales": 8810000, "actual_qty": 89},
    {"no": 4, "actual_sales": 952000, "actual_qty": 4},
    {"no": 5, "actual_sales": 660000, "actual_qty": 2},
    {"no": 6, "actual_sales": 7720000, "actual_qty": 92},
    {"no": 7, "actual_sales": 29760000, "actual_qty": 432},
    {"no": 8, "actual_sales": 58340000, "actual_qty": 1084},
    {"no": 9, "actual_sales": 20900000, "actual_qty": 162},
    {"no": 10, "actual_sales": 33860000, "actual_qty": 114},
]
actual_df = pd.DataFrame(actual_data)

# 엔진 생성
engine = create_engine('postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB')

# 결과 저장
results = []

# ==========================================
# 실험 1: 기본 모델 (전체 데이터)
# ==========================================
print("=" * 100)
print("실험 1: 기본 모델 (전체 데이터)")
print("=" * 100)
print()

df_all = load_data(engine)
print(f"전체 데이터: {len(df_all)}건")
print(f"기간: {df_all['broadcast_date'].min()} ~ {df_all['broadcast_date'].max()}")
print()

# 학습 데이터 준비
X_all = df_all.drop(columns=['gross_profit', 'sales_efficiency', 'quantity_sold'])
y_all = df_all['quantity_sold']
y_all_log = np.log1p(y_all)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_all, y_all_log, test_size=0.2, random_state=42, shuffle=False
)
_, _, y_train_orig, y_test_orig = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, shuffle=False
)

# 모델 학습
pipe = build_pipeline()
pipe.fit(X_train, y_train_log)

# 예측
y_pred_log = pipe.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Smearing Factor 계산
y_train_pred_log = pipe.predict(X_train)
residuals = y_train_log - y_train_pred_log
smearing_factor = np.mean(np.exp(residuals))
y_pred_smeared = y_pred * smearing_factor

# 평가
mae = mean_absolute_error(y_test_orig, y_pred_smeared)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_smeared))
r2 = r2_score(y_test_orig, y_pred_smeared)

print(f"테스트 데이터 성능:")
print(f"  MAE:  {mae:.2f}개")
print(f"  RMSE: {rmse:.2f}개")
print(f"  R²:   {r2:.4f}")
print(f"  Smearing Factor: {smearing_factor:.4f}")
print()

results.append({
    'method': '기본 (전체 데이터)',
    'data_size': len(df_all),
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'smearing_factor': smearing_factor
})

# ==========================================
# 실험 2: 최근 1년 데이터만 사용
# ==========================================
print("=" * 100)
print("실험 2: 최근 1년 데이터만 사용 (2025년 이후)")
print("=" * 100)
print()

df_recent = df_all[df_all['broadcast_date'] >= '2025-01-01'].copy()
print(f"최근 1년 데이터: {len(df_recent)}건")
print(f"기간: {df_recent['broadcast_date'].min()} ~ {df_recent['broadcast_date'].max()}")
print()

if len(df_recent) < 1000:
    print("⚠️  데이터가 너무 적음 (< 1000건). 6개월 데이터로 재시도...")
    df_recent = df_all[df_all['broadcast_date'] >= '2025-07-01'].copy()
    print(f"최근 6개월 데이터: {len(df_recent)}건")
    print()

# 학습 데이터 준비
X_recent = df_recent.drop(columns=['gross_profit', 'sales_efficiency', 'quantity_sold'])
y_recent = df_recent['quantity_sold']
y_recent_log = np.log1p(y_recent)

X_train_r, X_test_r, y_train_log_r, y_test_log_r = train_test_split(
    X_recent, y_recent_log, test_size=0.2, random_state=42, shuffle=False
)
_, _, y_train_orig_r, y_test_orig_r = train_test_split(
    X_recent, y_recent, test_size=0.2, random_state=42, shuffle=False
)

# 모델 학습
pipe_r = build_pipeline()
pipe_r.fit(X_train_r, y_train_log_r)

# 예측
y_pred_log_r = pipe_r.predict(X_test_r)
y_pred_r = np.expm1(y_pred_log_r)

# Smearing Factor 계산
y_train_pred_log_r = pipe_r.predict(X_train_r)
residuals_r = y_train_log_r - y_train_pred_log_r
smearing_factor_r = np.mean(np.exp(residuals_r))
y_pred_smeared_r = y_pred_r * smearing_factor_r

# 평가
mae_r = mean_absolute_error(y_test_orig_r, y_pred_smeared_r)
rmse_r = np.sqrt(mean_squared_error(y_test_orig_r, y_pred_smeared_r))
r2_r = r2_score(y_test_orig_r, y_pred_smeared_r)

print(f"테스트 데이터 성능:")
print(f"  MAE:  {mae_r:.2f}개")
print(f"  RMSE: {rmse_r:.2f}개")
print(f"  R²:   {r2_r:.4f}")
print(f"  Smearing Factor: {smearing_factor_r:.4f}")
print()

results.append({
    'method': '최근 1년 데이터',
    'data_size': len(df_recent),
    'mae': mae_r,
    'rmse': rmse_r,
    'r2': r2_r,
    'smearing_factor': smearing_factor_r
})

# ==========================================
# 실험 3: Sample Weight 적용
# ==========================================
print("=" * 100)
print("실험 3: Sample Weight 적용 (고매출 상품 가중치)")
print("=" * 100)
print()

# 가중치 계산: 75% 이상 고매출 상품에 2배 가중치
q75 = y_train_orig.quantile(0.75)
sample_weights = np.where(y_train_orig > q75, 2.0, 1.0)

print(f"Q75 (75% 분위수): {q75:.0f}개")
print(f"고매출 상품 ({q75:.0f}개 이상): {(sample_weights == 2.0).sum()}건 (가중치 2.0)")
print(f"일반 상품: {(sample_weights == 1.0).sum()}건 (가중치 1.0)")
print()

# 모델 학습 (Sample Weight 적용)
pipe_w = build_pipeline()
pipe_w.fit(X_train, y_train_log, model__sample_weight=sample_weights)

# 예측
y_pred_log_w = pipe_w.predict(X_test)
y_pred_w = np.expm1(y_pred_log_w)

# Smearing Factor 계산
y_train_pred_log_w = pipe_w.predict(X_train)
residuals_w = y_train_log - y_train_pred_log_w
smearing_factor_w = np.mean(np.exp(residuals_w))
y_pred_smeared_w = y_pred_w * smearing_factor_w

# 평가
mae_w = mean_absolute_error(y_test_orig, y_pred_smeared_w)
rmse_w = np.sqrt(mean_squared_error(y_test_orig, y_pred_smeared_w))
r2_w = r2_score(y_test_orig, y_pred_smeared_w)

print(f"테스트 데이터 성능:")
print(f"  MAE:  {mae_w:.2f}개")
print(f"  RMSE: {rmse_w:.2f}개")
print(f"  R²:   {r2_w:.4f}")
print(f"  Smearing Factor: {smearing_factor_w:.4f}")
print()

results.append({
    'method': 'Sample Weight',
    'data_size': len(df_all),
    'mae': mae_w,
    'rmse': rmse_w,
    'r2': r2_w,
    'smearing_factor': smearing_factor_w
})

# ==========================================
# 실험 4: 최근 1년 + Sample Weight
# ==========================================
print("=" * 100)
print("실험 4: 최근 1년 데이터 + Sample Weight")
print("=" * 100)
print()

# 가중치 계산
q75_r = y_train_orig_r.quantile(0.75)
sample_weights_r = np.where(y_train_orig_r > q75_r, 2.0, 1.0)

print(f"Q75 (75% 분위수): {q75_r:.0f}개")
print(f"고매출 상품: {(sample_weights_r == 2.0).sum()}건 (가중치 2.0)")
print(f"일반 상품: {(sample_weights_r == 1.0).sum()}건 (가중치 1.0)")
print()

# 모델 학습
pipe_rw = build_pipeline()
pipe_rw.fit(X_train_r, y_train_log_r, model__sample_weight=sample_weights_r)

# 예측
y_pred_log_rw = pipe_rw.predict(X_test_r)
y_pred_rw = np.expm1(y_pred_log_rw)

# Smearing Factor 계산
y_train_pred_log_rw = pipe_rw.predict(X_train_r)
residuals_rw = y_train_log_r - y_train_pred_log_rw
smearing_factor_rw = np.mean(np.exp(residuals_rw))
y_pred_smeared_rw = y_pred_rw * smearing_factor_rw

# 평가
mae_rw = mean_absolute_error(y_test_orig_r, y_pred_smeared_rw)
rmse_rw = np.sqrt(mean_squared_error(y_test_orig_r, y_pred_smeared_rw))
r2_rw = r2_score(y_test_orig_r, y_pred_smeared_rw)

print(f"테스트 데이터 성능:")
print(f"  MAE:  {mae_rw:.2f}개")
print(f"  RMSE: {rmse_rw:.2f}개")
print(f"  R²:   {r2_rw:.4f}")
print(f"  Smearing Factor: {smearing_factor_rw:.4f}")
print()

results.append({
    'method': '최근 1년 + Sample Weight',
    'data_size': len(df_recent),
    'mae': mae_rw,
    'rmse': rmse_rw,
    'r2': r2_rw,
    'smearing_factor': smearing_factor_rw
})

# ==========================================
# 결과 비교
# ==========================================
print("=" * 100)
print("📊 실험 결과 비교 (테스트 데이터)")
print("=" * 100)
print()

results_df = pd.DataFrame(results)
print(f"{'방법':<25} {'데이터 크기':>12} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 100)

for _, row in results_df.iterrows():
    print(f"{row['method']:<25} {row['data_size']:>12,}건 {row['mae']:>10.2f} {row['rmse']:>10.2f} {row['r2']:>10.4f}")

print()

# 최고 성능 모델 선택
best_idx = results_df['r2'].idxmax()
best_method = results_df.loc[best_idx, 'method']
best_r2 = results_df.loc[best_idx, 'r2']

print(f"✅ 최고 성능: {best_method} (R² {best_r2:.4f})")
print()

# 최고 성능 모델 저장
print("=" * 100)
print("💾 최고 성능 모델 저장")
print("=" * 100)
print()

if best_method == '기본 (전체 데이터)':
    best_pipe = pipe
    best_smearing = smearing_factor
elif best_method == '최근 1년 데이터':
    best_pipe = pipe_r
    best_smearing = smearing_factor_r
elif best_method == 'Sample Weight':
    best_pipe = pipe_w
    best_smearing = smearing_factor_w
else:  # 최근 1년 + Sample Weight
    best_pipe = pipe_rw
    best_smearing = smearing_factor_rw

model_path = Path('/app/app/xgb_broadcast_profit.joblib')
model_data = {
    'pipeline': best_pipe,
    'smearing_factor': best_smearing,
    'method': best_method
}
joblib.dump(model_data, model_path)

print(f"✅ 최고 성능 모델 저장 완료: {best_method}")
print(f"   - Smearing Factor: {best_smearing:.4f}")
print(f"   - 경로: {model_path}")
print()

print("=" * 100)
print("🎉 실험 완료!")
print("=" * 100)
