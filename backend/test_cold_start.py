import asyncio
import os
import joblib
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

# Adjust path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.broadcast_workflow import BroadcastWorkflow
from app.product_embedder import ProductEmbedder

async def run_test():
    print("=== 신상품(Cold Start) 매출 예측 테스트 ===")
    
    # 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), 'app', 'xgb_broadcast_profit.joblib')
    model = joblib.load(model_path)
    
    embedder = ProductEmbedder(
        openai_api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        qdrant_host="localhost"
    )
    
    workflow = BroadcastWorkflow(model=model)
    workflow.product_embedder = embedder # 주입
    
    # 가상의 신상품 생성 (DB에 없는 완전 새로운 상품)
    new_product = {
        'product_code': 'NEW_001',
        'product_name': '다이슨 에어랩 멀티 스타일러 컴플리트',
        'category_main': '가전',
        'category_middle': '이·미용가전',
        'category_sub': '헤어스타일러',
        'brand': '다이슨',
        'product_type': '유형',
        'product_price': 749000,
        'product_avg_profit': 0, # 신상품이므로 실적 0
        'product_broadcast_count': 0 # 신상품이므로 방송횟수 0
    }
    
    print(f"\n📦 테스트 신상품: {new_product['product_name']} ({new_product['category_main']}/{new_product['category_middle']})")
    print(f"💰 단가: {new_product['product_price']:,.0f}원 | 📊 실적: 없음 (신상품)")
    
    # 테스트 날짜 및 시간대
    target_date = datetime(2025, 11, 19) # 평일
    hour = 21 # 프라임타임
    
    broadcast_dt = target_date.replace(hour=hour, minute=0, second=0)
    context = {
        'broadcast_dt': broadcast_dt,
        'time_slot': '저녁',
        'weather': {'weather': 'Clear', 'temperature': 15},
        'is_holiday': 0
    }
    
    print("\n🔍 Vector DB를 통한 유사 상품 검색 및 실적 추정 진행 중...")
    pred_sales = await workflow._predict_product_sales(new_product, context)
    
    print(f"\n🎯 최종 예측 매출: {pred_sales:,.0f}원")

if __name__ == "__main__":
    asyncio.run(run_test())
