#!/usr/bin/env python3
"""추가 테스트 데이터로 API 성능 검증"""

import requests
import json

API_URL = "http://localhost:8501/api/v1/sales/predict-single"

# 추가 테스트 케이스 (2026-02-24 실제 방송 데이터)
test_cases = [
    {
        "name": "참좋은여행 태항산 2월",
        "product_code": "27776109",
        "tape_code": "0000016361",
        "broadcast_datetime": "2026-02-24 10:00:00",
        "actual_gross_profit": 19590000,
        "actual_quantity": 170
    },
    {
        "name": "라이나생명 암보험",
        "product_code": "22172389",
        "tape_code": "0000016140",
        "broadcast_datetime": "2026-02-24 11:00:00",
        "actual_gross_profit": 26550000,
        "actual_quantity": 40
    },
    {
        "name": "바디프랜드 팔콘",
        "product_code": "27776104",
        "tape_code": "0000016304",
        "broadcast_datetime": "2026-02-24 14:00:00",
        "actual_gross_profit": 26200000,
        "actual_quantity": 145
    },
    {
        "name": "비에날씬 9박스",
        "product_code": "24911939",
        "tape_code": "0000015827",
        "broadcast_datetime": "2026-02-24 09:00:00",
        "actual_gross_profit": 20435075,
        "actual_quantity": 89
    },
    {
        "name": "더창 볼륨핏 인모가발",
        "product_code": "27147947",
        "tape_code": "0000016241",
        "broadcast_datetime": "2026-02-24 15:00:00",
        "actual_gross_profit": 16171925,
        "actual_quantity": 608
    },
]

print("=" * 70)
print("📊 추가 테스트 데이터로 API 성능 검증 (Quantile 0.85)")
print("=" * 70)

results = []

for tc in test_cases:
    payload = {
        "product_code": tc["product_code"],
        "tape_code": tc["tape_code"],
        "broadcast_start_time": tc["broadcast_datetime"]
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            predicted_sales = data.get("predicted_sales", 0)
            actual_sales = tc["actual_gross_profit"]
            error = abs(actual_sales - predicted_sales)
            error_rate = (error / actual_sales) * 100 if actual_sales > 0 else 0
            
            # 과소예측 vs 과대예측 판단
            direction = "🔺 과대" if predicted_sales > actual_sales else "🔻 과소"
            
            results.append({
                "name": tc["name"],
                "actual": actual_sales,
                "predicted": predicted_sales,
                "error": error,
                "error_rate": error_rate,
                "direction": direction
            })
            
            print(f"\n{'=' * 70}")
            print(f"테스트: {tc['name']}")
            print(f"{'=' * 70}")
            print(f"💰 실제 매출액: {actual_sales:,.0f}원")
            print(f"🤖 예측 매출액: {predicted_sales:,.0f}원")
            print(f"📊 오차: {error:,.0f}원 ({error_rate:.1f}%) {direction}")
        else:
            print(f"\n❌ {tc['name']}: API 오류 - {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"\n❌ {tc['name']}: 요청 실패 - {e}")

# 종합 결과
print("\n" + "=" * 70)
print("📈 종합 결과 요약")
print("=" * 70)

if results:
    avg_error_rate = sum(r["error_rate"] for r in results) / len(results)
    over_predictions = sum(1 for r in results if "과대" in r["direction"])
    under_predictions = sum(1 for r in results if "과소" in r["direction"])
    
    print(f"\n평균 오차율: {avg_error_rate:.1f}%")
    print(f"과대예측: {over_predictions}건 / 과소예측: {under_predictions}건")
    
    print(f"\n{'상품명':<25} {'실제매출':>15} {'예측매출':>15} {'오차율':>10} {'방향':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} {r['actual']:>15,.0f} {r['predicted']:>15,.0f} {r['error_rate']:>9.1f}% {r['direction']:>8}")

print("\n" + "=" * 70)
print("✅ 추가 테스트 완료")
print("=" * 70)
