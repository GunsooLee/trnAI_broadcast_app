#!/usr/bin/env python3
"""최근 3일(2/22-2/24) 데이터로 전체 API 성능 검증"""

import requests
import json

API_URL = "http://localhost:8501/api/v1/sales/predict-single"

# 최근 3일 테스트 케이스 (실제 방송 데이터)
test_cases = [
    # 2월 24일 데이터
    {"name": "참좋은여행 태항산 2월", "product_code": "27776109", "tape_code": "0000016361", "broadcast_datetime": "2026-02-24 10:00:00", "actual_gross_profit": 19590000, "actual_quantity": 170},
    {"name": "라이나생명 암보험", "product_code": "22172389", "tape_code": "0000016140", "broadcast_datetime": "2026-02-24 11:00:00", "actual_gross_profit": 26550000, "actual_quantity": 40},
    {"name": "바디프랜드 팔콘", "product_code": "27776104", "tape_code": "0000016304", "broadcast_datetime": "2026-02-24 14:00:00", "actual_gross_profit": 26200000, "actual_quantity": 145},
    {"name": "연세 알부민 골드", "product_code": "26571430", "tape_code": "0000016153", "broadcast_datetime": "2026-02-24 15:00:00", "actual_gross_profit": 8267075, "actual_quantity": 119},
    {"name": "여성 몽크로스 티셔츠", "product_code": "27787801", "tape_code": "0000016327", "broadcast_datetime": "2026-02-24 16:00:00", "actual_gross_profit": 8386459, "actual_quantity": 411},
    {"name": "슈리오 스니커즈", "product_code": "24229485", "tape_code": "0000015661", "broadcast_datetime": "2026-02-24 17:00:00", "actual_gross_profit": 15710127, "actual_quantity": 1405},
    {"name": "비에날씬 9박스", "product_code": "24911939", "tape_code": "0000015827", "broadcast_datetime": "2026-02-24 18:00:00", "actual_gross_profit": 20435075, "actual_quantity": 89},
    {"name": "배한호 알부민", "product_code": "26570242", "tape_code": "0000016138", "broadcast_datetime": "2026-02-24 19:00:00", "actual_gross_profit": 22823833, "actual_quantity": 179},
    {"name": "더창 볼륨핏 인모가발", "product_code": "27147947", "tape_code": "0000016241", "broadcast_datetime": "2026-02-24 20:00:00", "actual_gross_profit": 16171925, "actual_quantity": 608},
    {"name": "세일 토비콤", "product_code": "23514612", "tape_code": "0000015492", "broadcast_datetime": "2026-02-24 21:00:00", "actual_gross_profit": 20884170, "actual_quantity": 309},
    
    # 2월 23일 데이터
    {"name": "리체나 염색제", "product_code": "15327129", "tape_code": "0000015430", "broadcast_datetime": "2026-02-23 10:00:00", "actual_gross_profit": 12388959, "actual_quantity": 344},
    {"name": "마담4060 니트", "product_code": "24420032", "tape_code": "0000016375", "broadcast_datetime": "2026-02-23 11:00:00", "actual_gross_profit": 16235951, "actual_quantity": 1314},
    {"name": "포티니 만능 선반", "product_code": "24734451", "tape_code": "0000016220", "broadcast_datetime": "2026-02-23 14:00:00", "actual_gross_profit": 14753249, "actual_quantity": 816},
    {"name": "흥국생명 암보험", "product_code": "25543592", "tape_code": "0000016037", "broadcast_datetime": "2026-02-23 15:00:00", "actual_gross_profit": 15407361, "actual_quantity": 409},
    {"name": "바로빗 염색약", "product_code": "27796765", "tape_code": "0000016334", "broadcast_datetime": "2026-02-23 16:00:00", "actual_gross_profit": 15900000, "actual_quantity": 113},
    {"name": "대원제약 알부민", "product_code": "27549551", "tape_code": "0000016321", "broadcast_datetime": "2026-02-23 17:00:00", "actual_gross_profit": 17946302, "actual_quantity": 519},
    {"name": "란체티 카라니트", "product_code": "26754206", "tape_code": "0000016314", "broadcast_datetime": "2026-02-23 18:00:00", "actual_gross_profit": 26779268, "actual_quantity": 291},
    {"name": "김하진 갈비탕", "product_code": "23416732", "tape_code": "0000016355", "broadcast_datetime": "2026-02-23 19:00:00", "actual_gross_profit": 24170644, "actual_quantity": 702},
    {"name": "라쉬반 드로즈", "product_code": "26868841", "tape_code": "0000015450", "broadcast_datetime": "2026-02-23 20:00:00", "actual_gross_profit": 15403846, "actual_quantity": 1071},
    {"name": "하라즈 앰플염색제", "product_code": "25967753", "tape_code": "0000016242", "broadcast_datetime": "2026-02-23 21:00:00", "actual_gross_profit": 7516600, "actual_quantity": 139},
    
    # 2월 22일 데이터
    {"name": "헤스티지 양가죽 코트", "product_code": "27583554", "tape_code": "0000015950", "broadcast_datetime": "2026-02-22 10:00:00", "actual_gross_profit": 7212405, "actual_quantity": 195},
    {"name": "지니라이프 콘드로이친", "product_code": "27579746", "tape_code": "0000016371", "broadcast_datetime": "2026-02-22 11:00:00", "actual_gross_profit": 12570593, "actual_quantity": 318},
    {"name": "블라우풍트 이어폰", "product_code": "26441024", "tape_code": "0000016333", "broadcast_datetime": "2026-02-22 14:00:00", "actual_gross_profit": 19826906, "actual_quantity": 380},
    {"name": "뉴트리원 카무트", "product_code": "20311238", "tape_code": "0000015439", "broadcast_datetime": "2026-02-22 15:00:00", "actual_gross_profit": 14910748, "actual_quantity": 598},
    {"name": "나에노 액체세제", "product_code": "23983442", "tape_code": "0000016216", "broadcast_datetime": "2026-02-22 16:00:00", "actual_gross_profit": 17306639, "actual_quantity": 744},
    {"name": "투라 속눈썹", "product_code": "21033338", "tape_code": "0000015603", "broadcast_datetime": "2026-02-22 17:00:00", "actual_gross_profit": 15887292, "actual_quantity": 620},
    {"name": "홈앤코튼 화장지", "product_code": "26494023", "tape_code": "0000016259", "broadcast_datetime": "2026-02-22 18:00:00", "actual_gross_profit": 20555648, "actual_quantity": 666},
    {"name": "제시카 헤어큐", "product_code": "26135530", "tape_code": "0000016365", "broadcast_datetime": "2026-02-22 19:00:00", "actual_gross_profit": 18444593, "actual_quantity": 285},
    {"name": "레드캠프 남성 자켓", "product_code": "24680045", "tape_code": "0000016057", "broadcast_datetime": "2026-02-22 20:00:00", "actual_gross_profit": 11637723, "actual_quantity": 549},
    {"name": "26SS 여성 워킹화", "product_code": "27782650", "tape_code": "0000016349", "broadcast_datetime": "2026-02-22 21:00:00", "actual_gross_profit": 16492601, "actual_quantity": 631},
]

print("=" * 80)
print("📊 최근 3일(2/22-2/24) 데이터로 API 성능 검증 (Quantile 0.85)")
print("=" * 80)

results = []
success_count = 0
error_count = 0

for i, tc in enumerate(test_cases, 1):
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
            
            success_count += 1
            
            if i <= 5:  # 처음 5개만 상세 출력
                print(f"\n[{i:2d}] {tc['name']}")
                print(f"💰 실제: {actual_sales:,.0f}원 | 🤖 예측: {predicted_sales:,.0f}원")
                print(f"📊 오차: {error:,.0f}원 ({error_rate:.1f}%) {direction}")
        else:
            print(f"\n❌ [{i:2d}] {tc['name']}: API 오류 - {response.status_code}")
            error_count += 1
    except Exception as e:
        print(f"\n❌ [{i:2d}] {tc['name']}: 요청 실패 - {e}")
        error_count += 1

# 종합 결과
print("\n" + "=" * 80)
print("📈 종합 결과 요약")
print("=" * 80)

if results:
    avg_error_rate = sum(r["error_rate"] for r in results) / len(results)
    over_predictions = sum(1 for r in results if "과대" in r["direction"])
    under_predictions = sum(1 for r in results if "과소" in r["direction"])
    
    print(f"\n📊 성능 지표:")
    print(f"   전체 테스트: {len(test_cases)}건 (성공: {success_count}건, 실패: {error_count}건)")
    print(f"   평균 오차율: {avg_error_rate:.1f}%")
    print(f"   과대예측: {over_predictions}건 / 과소예측: {under_predictions}건")
    
    # 오차율 분포
    excellent = sum(1 for r in results if r["error_rate"] <= 10)
    good = sum(1 for r in results if 10 < r["error_rate"] <= 20)
    fair = sum(1 for r in results if 20 < r["error_rate"] <= 30)
    poor = sum(1 for r in results if r["error_rate"] > 30)
    
    print(f"\n📊 오차율 분포:")
    print(f"   우수 (≤10%): {excellent}건")
    print(f"   양호 (11-20%): {good}건") 
    print(f"   보통 (21-30%): {fair}건")
    print(f"   부족 (>30%): {poor}건")
    
    print(f"\n{'상품명':<20} {'실제매출':>12} {'예측매출':>12} {'오차율':>8} {'방향':>8}")
    print("-" * 65)
    for r in results[:15]:  # 상위 15개만 표시
        print(f"{r['name']:<20} {r['actual']:>12,.0f} {r['predicted']:>12,.0f} {r['error_rate']:>7.1f}% {r['direction']:>8}")

print("\n" + "=" * 80)
print("✅ 최근 3일 데이터 테스트 완료")
print("=" * 80)
