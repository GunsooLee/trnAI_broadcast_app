#!/usr/bin/env python3
"""
오늘(2026-02-25) 방송 편성에 대한 매출 예측
"""

import requests
import pandas as pd
from datetime import datetime
import json

# 오늘 방송 편성 데이터 (이미지에서 추출)
broadcasts = [
    {"no": 1, "tape_code": "0000015603", "start": "2026-02-25 00:52:31", "end": "2026-02-25 01:52:01"},
    {"no": 2, "tape_code": "0000015470", "start": "2026-02-25 01:52:01", "end": "2026-02-25 02:51:31"},
    {"no": 3, "tape_code": "0000016333", "start": "2026-02-25 02:51:31", "end": "2026-02-25 03:51:01"},
    {"no": 4, "tape_code": "0000015454", "start": "2026-02-25 03:57:02", "end": "2026-02-25 04:16:52"},
    {"no": 5, "tape_code": "0000015615", "start": "2026-02-25 04:16:52", "end": "2026-02-25 04:36:42"},
    {"no": 6, "tape_code": "0000016272", "start": "2026-02-25 04:36:42", "end": "2026-02-25 05:36:37"},
    {"no": 7, "tape_code": "0000016218", "start": "2026-02-25 05:36:37", "end": "2026-02-25 06:36:07"},
    {"no": 8, "tape_code": "0000016377", "start": "2026-02-25 06:36:32", "end": "2026-02-25 07:36:02"},
    {"no": 9, "tape_code": "0000016102", "start": "2026-02-25 07:36:12", "end": "2026-02-25 08:35:32"},
    {"no": 10, "tape_code": "0000016138", "start": "2026-02-25 08:35:57", "end": "2026-02-25 09:35:27"},
    {"no": 11, "tape_code": "0000016364", "start": "2026-02-25 09:35:27", "end": "2026-02-25 10:34:57"},
    {"no": 12, "tape_code": "0000016222", "start": "2026-02-25 10:35:22", "end": "2026-02-25 11:34:52"},
    {"no": 13, "tape_code": "0000015603", "start": "2026-02-25 11:34:52", "end": "2026-02-25 12:34:22"},
    {"no": 14, "tape_code": "0000015775", "start": "2026-02-25 12:34:47", "end": "2026-02-25 13:34:17"},
    {"no": 15, "tape_code": "0000015629", "start": "2026-02-25 13:34:17", "end": "2026-02-25 14:33:47"},
    {"no": 16, "tape_code": "0000016057", "start": "2026-02-25 14:34:12", "end": "2026-02-25 15:33:42"},
    {"no": 17, "tape_code": "0000015727", "start": "2026-02-25 15:33:42", "end": "2026-02-25 16:33:12"},
    {"no": 18, "tape_code": "0000016039", "start": "2026-02-25 16:33:37", "end": "2026-02-25 17:33:07"},
    {"no": 19, "tape_code": "0000016275", "start": "2026-02-25 17:33:07", "end": "2026-02-25 18:32:37"},
    {"no": 20, "tape_code": "0000016242", "start": "2026-02-25 18:33:02", "end": "2026-02-25 19:32:32"},
    {"no": 21, "tape_code": "0000016152", "start": "2026-02-25 19:33:27", "end": "2026-02-25 20:32:57"},
    {"no": 22, "tape_code": "0000016364", "start": "2026-02-25 20:33:52", "end": "2026-02-25 21:33:22"},
    {"no": 23, "tape_code": "0000016376", "start": "2026-02-25 21:34:17", "end": "2026-02-25 22:33:47"},
    {"no": 24, "tape_code": "0000016304", "start": "2026-02-25 22:33:47", "end": "2026-02-25 23:33:17"},
    {"no": 25, "tape_code": "0000016340", "start": "2026-02-25 23:33:17", "end": "2026-02-26 00:32:47"},
]

API_URL = "http://localhost:8501/api/v1/sales/predict-single"

def predict_broadcast(broadcast):
    """단일 방송에 대한 매출 예측"""
    payload = {
        "tape_code": broadcast["tape_code"],
        "broadcast_start_time": broadcast["start"],
        "broadcast_end_time": broadcast["end"]
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        return {
            "no": broadcast["no"],
            "tape_code": broadcast["tape_code"],
            "product_code": result.get("product_code", ""),
            "product_name": result.get("product_name", ""),
            "broadcast_start": broadcast["start"],
            "broadcast_end": broadcast["end"],
            "predicted_sales": result.get("predicted_sales", 0),
            "status": "success"
        }
    except Exception as e:
        return {
            "no": broadcast["no"],
            "tape_code": broadcast["tape_code"],
            "product_code": "",
            "product_name": "",
            "broadcast_start": broadcast["start"],
            "broadcast_end": broadcast["end"],
            "predicted_sales": 0,
            "status": f"error: {str(e)}"
        }

def main():
    print("=" * 80)
    print("📊 2026-02-25 방송 편성 매출 예측")
    print("=" * 80)
    print(f"총 {len(broadcasts)}건의 방송 예측 시작...")
    print()
    
    results = []
    
    for i, broadcast in enumerate(broadcasts, 1):
        print(f"[{i}/{len(broadcasts)}] {broadcast['tape_code']} 예측 중...", end=" ")
        result = predict_broadcast(broadcast)
        results.append(result)
        
        if result["status"] == "success":
            print(f"✅ {result['predicted_sales']:,.0f}원")
        else:
            print(f"❌ {result['status']}")
    
    # DataFrame 생성
    df = pd.DataFrame(results)
    
    # 결과 출력
    print()
    print("=" * 80)
    print("📊 예측 결과 요약")
    print("=" * 80)
    print()
    
    success_count = len(df[df['status'] == 'success'])
    total_predicted_sales = df[df['status'] == 'success']['predicted_sales'].sum()
    
    print(f"성공: {success_count}/{len(broadcasts)}건")
    print(f"총 예측 매출: {total_predicted_sales:,.0f}원")
    print(f"평균 예측 매출: {total_predicted_sales/success_count:,.0f}원")
    print()
    
    # 상위 5건
    print("=" * 80)
    print("예측 매출 상위 5건")
    print("=" * 80)
    top_5 = df[df['status'] == 'success'].nlargest(5, 'predicted_sales')
    
    for idx, row in top_5.iterrows():
        print(f"{row['no']:2d}. {row['broadcast_start']} ~ {row['broadcast_end']}")
        print(f"    테이프: {row['tape_code']} | 상품: {row['product_name'][:50]}")
        print(f"    예측 매출: {row['predicted_sales']:,.0f}원")
        print()
    
    # CSV 저장
    output_file = "/app/today_broadcast_predictions.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 예측 결과 저장: {output_file}")
    print()
    
    # 실제 매출 비교용 템플릿 생성
    comparison_df = df[['no', 'tape_code', 'product_name', 'broadcast_start', 'broadcast_end', 'predicted_sales']].copy()
    comparison_df['actual_sales'] = ''  # 실제 매출 입력란
    comparison_df['error'] = ''  # 오차 계산란
    comparison_df['error_pct'] = ''  # 오차율 계산란
    
    comparison_file = "/app/today_broadcast_comparison_template.csv"
    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f"✅ 실제 매출 비교용 템플릿 저장: {comparison_file}")
    print("   (actual_sales 컬럼에 실제 매출을 입력하여 비교하세요)")
    print()

if __name__ == "__main__":
    main()
