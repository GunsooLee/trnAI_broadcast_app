#!/usr/bin/env python3
"""
quantity_sold 복구 후 API 성능 비교 테스트
이전 FINAL_REPORT.md의 테스트 데이터와 동일한 상품으로 재테스트
"""

import requests
import json
from datetime import datetime

API_URL = "http://localhost:8501/api/v1/sales/predict-single"

# FINAL_REPORT.md에 있던 테스트 케이스 (2026-02-24 데이터)
test_cases = [
    {
        "name": "배한호 알부민",
        "tape_code": "0000016138",
        "broadcast_start_time": "2026-02-24 08:36:01",
        "actual_quantity": 179,
        "actual_gross_profit": 22823833,
        "price": 297000,
        "simple_calc": 179 * 297000  # 53,163,000
    },
    {
        "name": "슈리오 스니커즈",
        "tape_code": "0000015661",
        "broadcast_start_time": "2026-02-24 06:36:36",
        "actual_quantity": 1405,
        "actual_gross_profit": 15710127,
        "price": 49000,
        "simple_calc": 1405 * 49000  # 68,845,000
    },
    {
        "name": "투라 속눈썹",
        "tape_code": "0000015603",
        "broadcast_start_time": "2026-02-23 11:34:54",
        "actual_quantity": 620,
        "actual_gross_profit": 15887292,
        "price": 69900,
        "simple_calc": 620 * 69900  # 43,338,000
    }
]

print("=" * 100)
print("📊 quantity_sold 복구 후 API 성능 테스트")
print("=" * 100)
print()

results = []

for idx, case in enumerate(test_cases, 1):
    print(f"\n{'='*100}")
    print(f"테스트 케이스 {idx}: {case['name']}")
    print(f"{'='*100}")
    
    payload = {
        "tape_code": case["tape_code"],
        "broadcast_start_time": case["broadcast_start_time"]
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        predicted_quantity = data.get("predicted_quantity", 0)
        predicted_sales = data.get("predicted_sales", 0)
        
        # 실제값 vs 예측값 비교
        quantity_error = abs(predicted_quantity - case["actual_quantity"])
        quantity_error_pct = (quantity_error / case["actual_quantity"] * 100) if case["actual_quantity"] > 0 else 0
        
        profit_error = abs(predicted_sales - case["actual_gross_profit"])
        profit_error_pct = (profit_error / case["actual_gross_profit"] * 100) if case["actual_gross_profit"] > 0 else 0
        
        print(f"\n📦 판매 수량 (quantity_sold):")
        print(f"   실제: {case['actual_quantity']:,}개")
        print(f"   예측: {predicted_quantity:,.1f}개")
        print(f"   오차: {quantity_error:,.1f}개 ({quantity_error_pct:.1f}%)")
        
        print(f"\n💰 매출액 (gross_profit):")
        print(f"   실제 gross_profit: {case['actual_gross_profit']:,}원")
        print(f"   예측 매출액: {predicted_sales:,.0f}원")
        print(f"   오차: {profit_error:,.0f}원 ({profit_error_pct:.1f}%)")
        
        print(f"\n📐 참고 계산:")
        print(f"   단순 계산 (수량×가격): {case['simple_calc']:,}원")
        print(f"   gross_profit / 단순계산: {(case['actual_gross_profit'] / case['simple_calc'] * 100):.1f}%")
        print(f"   → gross_profit은 할인/비용 반영된 실제 매출로 추정")
        
        results.append({
            "name": case["name"],
            "actual_quantity": case["actual_quantity"],
            "predicted_quantity": predicted_quantity,
            "quantity_error_pct": quantity_error_pct,
            "actual_profit": case["actual_gross_profit"],
            "predicted_profit": predicted_sales,
            "profit_error_pct": profit_error_pct
        })
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   상세: {e.response.text[:500]}")

print("\n" + "=" * 100)
print("📈 종합 결과 요약")
print("=" * 100)

if results:
    avg_qty_error = sum(r["quantity_error_pct"] for r in results) / len(results)
    avg_profit_error = sum(r["profit_error_pct"] for r in results) / len(results)
    
    print(f"\n평균 판매 수량 오차율: {avg_qty_error:.1f}%")
    print(f"평균 매출액 오차율: {avg_profit_error:.1f}%")
    
    print("\n상세 결과:")
    print(f"{'상품명':<20} {'실제수량':>10} {'예측수량':>10} {'수량오차':>10} {'실제매출':>15} {'예측매출':>15} {'매출오차':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<20} {r['actual_quantity']:>10,} {r['predicted_quantity']:>10,.0f} {r['quantity_error_pct']:>9.1f}% {r['actual_profit']:>15,} {r['predicted_profit']:>15,.0f} {r['profit_error_pct']:>9.1f}%")

print("\n" + "=" * 100)
print("✅ 테스트 완료")
print("=" * 100)
