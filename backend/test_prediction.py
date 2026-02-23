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

async def run_test():
    print("=== 예측 모델 테스트 스크립트 ===")
    
    # 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), 'app', 'xgb_broadcast_profit.joblib')
    model = joblib.load(model_path)
    workflow = BroadcastWorkflow(model=model)
    
    # 다양한 특징을 가진 상품 3개 선택 (DB에서)
    query = text("""
        SELECT 
            product_code, 
            MAX(product_name) as product_name, 
            MAX(category_main) as category_main, 
            MAX(category_middle) as category_middle, 
            MAX(category_sub) as category_sub, 
            MAX(brand) as brand, 
            MAX(product_type) as product_type,
            COALESCE(AVG(price), 0) as product_price, 
            COALESCE(AVG(gross_profit), 0) as avg_sales, 
            COUNT(*) as product_broadcast_count
        FROM broadcast_training_dataset
        GROUP BY product_code
        HAVING COUNT(*) > 10
        ORDER BY RANDOM()
        LIMIT 3
    """)
    
    with workflow.engine.connect() as conn:
        result = conn.execute(query)
        products = [dict(row._mapping) for row in result]
        
    # 시간대별 테스트
    test_hours = [3, 9, 14, 21]
    time_slot_names = {3: "새벽", 9: "오전", 14: "오후", 21: "저녁"}
    
    # 테스트 날짜: 평일
    target_date = datetime(2025, 11, 19) # 수요일
    
    for p in products:
        print(f"\n📦 상품: {p['product_name']} ({p['category_main']}/{p['category_middle']})")
        print(f"💰 단가: {p['product_price']:,.0f}원 | 📊 과거 평균 매출: {p['avg_sales']:,.0f}원 | 📺 방송 횟수: {p['product_broadcast_count']}회")
        
        for hour in test_hours:
            broadcast_dt = target_date.replace(hour=hour, minute=0, second=0)
            context = {
                'broadcast_dt': broadcast_dt,
                'time_slot': time_slot_names[hour],
                'weather': {'weather': 'Clear', 'temperature': 15},
                'is_holiday': 0
            }
            
            # workflow 내부의 예측 메서드 직접 호출
            pred_sales = await workflow._predict_product_sales(p, context)
            
            # 출력 포맷 맞추기
            slot_name = time_slot_names[hour]
            print(f"  🕒 [{slot_name} {hour:02d}:00] 예측 매출: {pred_sales:,.0f}원")

if __name__ == "__main__":
    asyncio.run(run_test())
