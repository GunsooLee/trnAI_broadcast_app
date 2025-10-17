#!/usr/bin/env python3
"""
방송편성 추천 API 테스트 스크립트
"""
import requests
import json
from datetime import datetime, timedelta

API_URL = "http://localhost:8501/api/v1/broadcast/recommendations"

def test_recommendation(broadcast_time, recommendation_count=5, trend_ratio=0.3, description=""):
    """추천 API 테스트"""
    print("\n" + "="*80)
    print(f"🧪 테스트: {description}")
    print("="*80)
    print(f"방송 시간: {broadcast_time}")
    print(f"추천 개수: {recommendation_count}")
    print(f"트렌드 비율: {trend_ratio:.0%}")
    print("-"*80)
    
    payload = {
        "broadcastTime": broadcast_time,
        "recommendationCount": recommendation_count,
        "trendRatio": trend_ratio
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ 성공 (응답 시간: {response.elapsed.total_seconds():.2f}초)")
            print(f"\n📊 추천 카테고리: {len(data.get('recommendedCategories', []))}개")
            for cat in data.get('recommendedCategories', [])[:3]:
                print(f"  {cat['rank']}. {cat['name']} - {cat['predictedSales']}")
            
            print(f"\n🎁 추천 상품: {len(data.get('recommendations', []))}개")
            for rec in data.get('recommendations', [])[:5]:
                info = rec['productInfo']
                metrics = rec['businessMetrics']
                print(f"\n  {rec['rank']}. {info['productName'][:50]}")
                print(f"     📂 카테고리: {info['category']}")
                print(f"     🎯 추천 타입: {rec['recommendationType']}")
                print(f"     💰 매출 예상: {metrics['pastAverageSales']}")
                print(f"     📝 근거: {rec['reasoning']['summary'][:60]}...")
            
            return True
        else:
            print(f"❌ 실패: HTTP {response.status_code}")
            print(f"응답: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def main():
    print("\n" + "🎬 방송편성 추천 API 테스트 시작" + "\n")
    
    # 현재 시간 기준
    now = datetime.now()
    
    test_cases = [
        {
            "broadcast_time": "2024-12-25T19:00:00Z",
            "recommendation_count": 5,
            "trend_ratio": 0.3,
            "description": "크리스마스 저녁 (트렌드 30%)"
        },
        {
            "broadcast_time": "2024-07-15T14:00:00Z",
            "recommendation_count": 5,
            "trend_ratio": 0.5,
            "description": "여름 오후 (트렌드 50%)"
        },
        {
            "broadcast_time": "2024-01-10T10:00:00Z",
            "recommendation_count": 3,
            "trend_ratio": 0.7,
            "description": "겨울 오전 (트렌드 70%)"
        },
        {
            "broadcast_time": "2024-10-01T20:00:00Z",
            "recommendation_count": 7,
            "trend_ratio": 0.0,
            "description": "가을 저녁 (매출예측만 100%)"
        },
    ]
    
    results = []
    for test in test_cases:
        success = test_recommendation(**test)
        results.append(success)
    
    # 최종 결과
    print("\n" + "="*80)
    print("📊 테스트 결과 요약")
    print("="*80)
    print(f"총 테스트: {len(results)}개")
    print(f"성공: {sum(results)}개")
    print(f"실패: {len(results) - sum(results)}개")
    
    if all(results):
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패")
    
    print("="*80)

if __name__ == '__main__':
    main()
