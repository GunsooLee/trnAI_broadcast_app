#!/usr/bin/env python3
"""네이버 베스트 상품 수동 업데이트 스크립트"""
import requests
import psycopg2
from datetime import datetime
import os

# API 호출
print("=== 네이버 베스트 상품 크롤링 ===")
response = requests.get('http://localhost:8501/api/v1/external/crawl-naver-best?max_products=20')
data = response.json()

products = data.get('products', [])
print(f'✅ {len(products)}개 상품 수집 완료\n')

# DB 연결
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='TRNAI_DB',
    user='TRN_AI',
    password=os.getenv('POSTGRES_PASSWORD', 'trn1234!')
)
cur = conn.cursor()

today = datetime.now().strftime('%Y-%m-%d')
success_count = 0
error_count = 0

print("=== DB 저장 시작 ===")
for i, p in enumerate(products, 1):
    try:
        cur.execute("""
            INSERT INTO external_products (
                product_id, name, sale_price, discounted_price, discount_ratio,
                image_url, landing_url, mobile_landing_url,
                is_delivery_free, delivery_fee, is_today_dispatch, is_sold_out,
                cumulation_sale_count, rank_order, channel_no, landing_service,
                review_count, review_score, mall_name,
                collected_at, collected_date
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (product_id, collected_date) DO UPDATE SET
                name = EXCLUDED.name,
                sale_price = EXCLUDED.sale_price,
                discounted_price = EXCLUDED.discounted_price,
                discount_ratio = EXCLUDED.discount_ratio,
                image_url = EXCLUDED.image_url,
                landing_url = EXCLUDED.landing_url,
                mobile_landing_url = EXCLUDED.mobile_landing_url,
                is_delivery_free = EXCLUDED.is_delivery_free,
                delivery_fee = EXCLUDED.delivery_fee,
                rank_order = EXCLUDED.rank_order,
                review_count = EXCLUDED.review_count,
                review_score = EXCLUDED.review_score,
                mall_name = EXCLUDED.mall_name,
                collected_at = EXCLUDED.collected_at,
                updated_at = CURRENT_TIMESTAMP
        """, (
            p['product_id'],
            p['name'],
            p['sale_price'],
            p['discounted_price'],
            p['discount_ratio'],
            p['image_url'],
            p['landing_url'],
            p['mobile_landing_url'],
            p['is_delivery_free'],
            p['delivery_fee'],
            p.get('is_today_dispatch', False),
            p.get('is_sold_out', False),
            p.get('cumulation_sale_count', 0),
            p['rank_order'],
            p['channel_no'],
            p['landing_service'],
            p.get('review_count', 0),
            p.get('review_score', 0.0),
            p.get('mall_name'),
            p['collected_at'],
            today
        ))
        
        success_count += 1
        print(f"  {i}. [{p['rank_order']}위] {p['name'][:40]} ✅")
        
    except Exception as e:
        error_count += 1
        print(f"  {i}. [{p['rank_order']}위] {p['name'][:40]} ❌ {e}")

conn.commit()
cur.close()
conn.close()

print(f"\n=== 완료 ===")
print(f"성공: {success_count}개")
print(f"실패: {error_count}개")
