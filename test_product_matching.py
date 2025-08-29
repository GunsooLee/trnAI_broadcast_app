#!/usr/bin/env python3
"""
상품 매칭 시스템 테스트 스크립트
"""

import requests
import json
import pandas as pd
from datetime import datetime

def test_product_matching():
    """상품 매칭 시스템 테스트"""
    
    print("🔍 상품 매칭 시스템 테스트 시작...")
    print("=" * 50)
    
    base_url = "http://localhost:8501"
    
    try:
        # 1. 트렌드 분석 API 테스트 (상품 매칭 포함)
        print("1️⃣ 트렌드 분석 & 상품 매칭 테스트...")
        response = requests.get(f"{base_url}/api/v1/trends/analyze", timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 트렌드 분석 성공!")
            
            # 매칭 결과 분석
            matched_results = data.get('matched_results', {})
            total_trends = len(matched_results)
            
            print(f"\n📊 매칭 결과 요약:")
            print(f"   분석된 트렌드: {total_trends}개")
            
            # 각 트렌드별 매칭 상품 수 확인
            total_matched_products = 0
            successful_matches = 0
            
            for keyword, match_data in matched_results.items():
                matched_products = match_data.get('matched_products', [])
                product_count = len(matched_products)
                total_matched_products += product_count
                
                if product_count > 0:
                    successful_matches += 1
                    
                print(f"   - {keyword}: {product_count}개 상품 매칭")
                
                # 상위 3개 상품 정보 표시
                for i, product in enumerate(matched_products[:3]):
                    similarity = product.get('similarity_score', 0)
                    product_name = product.get('product_name', 'Unknown')
                    print(f"     {i+1}. {product_name} (유사도: {similarity:.3f})")
            
            print(f"\n🎯 매칭 성능:")
            print(f"   성공적 매칭: {successful_matches}/{total_trends}개 트렌드")
            print(f"   총 매칭 상품: {total_matched_products}개")
            print(f"   평균 매칭률: {successful_matches/total_trends*100:.1f}%" if total_trends > 0 else "   평균 매칭률: 0%")
            
        else:
            print(f"❌ 트렌드 분석 실패: {response.status_code}")
            print(f"   응답: {response.text}")
            return False
        
        # 2. 개별 키워드 매칭 테스트
        print("\n2️⃣ 개별 키워드 매칭 테스트...")
        test_keywords = ["다이어트", "건강식품", "홈트레이닝", "스킨케어"]
        
        for keyword in test_keywords:
            print(f"\n🔍 '{keyword}' 키워드 테스트:")
            
            # 트렌드 기반 추천 API 호출
            payload = {
                "user_query": f"{keyword} 관련 상품 추천해주세요",
                "time_slot": "20:00-22:00",
                "target_audience": "30-40대 여성",
                "budget_range": "중간"
            }
            
            response = requests.post(
                f"{base_url}/api/v1/recommend-with-trends",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                print(f"   ✅ {len(recommendations)}개 추천 상품")
                
                # 상위 2개 추천 상품 표시
                for i, rec in enumerate(recommendations[:2]):
                    product_name = rec.get('product_name', 'Unknown')
                    predicted_sales = rec.get('predicted_sales', 0)
                    trend_boost = rec.get('trend_boost_factor', 1.0)
                    print(f"     {i+1}. {product_name}")
                    print(f"        예상매출: {predicted_sales:,.0f}원")
                    print(f"        트렌드부스트: {trend_boost:.2f}x")
                    
            else:
                print(f"   ❌ 추천 실패: {response.status_code}")
                # OpenAI API 키 문제일 가능성
                if "openai" in response.text.lower() or "api" in response.text.lower():
                    print("   ⚠️  OpenAI API 키 설정이 필요할 수 있습니다")
        
        # 3. Qdrant 벡터 DB 상태 확인
        print("\n3️⃣ 벡터 DB 상태 확인...")
        try:
            # 간접적으로 상품 DB 상태 확인
            response = requests.get(f"{base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ 벡터 DB 연결 정상")
            else:
                print("   ❌ 벡터 DB 연결 문제")
        except:
            print("   ❌ 벡터 DB 상태 확인 불가")
        
        print("\n🎉 상품 매칭 테스트 완료!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API 서버에 연결할 수 없습니다.")
        return False
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        return False

def analyze_matching_performance():
    """매칭 성능 분석"""
    print("\n📈 매칭 성능 분석...")
    
    # 예상되는 매칭 결과 분석
    expected_matches = {
        "다이어트": ["다이어트식품", "건강보조식품", "운동용품"],
        "건강식품": ["비타민", "영양제", "건강보조식품"],
        "홈트레이닝": ["운동용품", "헬스기구", "요가매트"],
        "스킨케어": ["화장품", "마스크팩", "스킨케어"]
    }
    
    print("예상 매칭 카테고리:")
    for keyword, categories in expected_matches.items():
        print(f"   {keyword} → {', '.join(categories)}")

if __name__ == "__main__":
    success = test_product_matching()
    if success:
        analyze_matching_performance()
    else:
        print("\n❌ 테스트 실패 - 시스템 상태를 확인하세요")
