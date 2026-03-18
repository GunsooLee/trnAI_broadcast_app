import requests
import json
from datetime import datetime

url = "http://localhost:8501/api/v1/broadcast/recommendations"
headers = {"Content-Type": "application/json"}
payload = {
    "broadcastTime": "2026-02-25T14:00:00",
    "recommendationCount": 5
}

try:
    print(f"API 요청 전송: {url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    print("\n[API 응답 결과]")
    if "recommendedCategories" in data:
        print(f"추천 카테고리: {data['recommendedCategories']}")
    
    if "recommendations" in data:
        for idx, rec in enumerate(data['recommendations'], 1):
            product = rec.get('product', {})
            tape = rec.get('broadcastTape', {})
            metrics = rec.get('predictedMetrics', {})
            
            print(f"\n{idx}. {product.get('productName', 'Unknown')} ({product.get('productCode', 'Unknown')})")
            print(f"   - 카테고리: {product.get('categoryMain', '')} > {product.get('categoryMiddle', '')}")
            print(f"   - 가격: {product.get('price', 0):,}원")
            if tape:
                print(f"   - 테이프: {tape.get('tapeName', '')} ({tape.get('durationMinutes', 0)}분)")
            print(f"   - 예측 판매량: {metrics.get('predictedQuantity', 0):,.1f}개")
            print(f"   - 예측 매출액: {metrics.get('predictedProfit', 0):,.0f}원")
            print(f"   - 추천 점수: {rec.get('score', 0):.2f}")
            print(f"   - 추천 근거: {rec.get('reason', '')}")
            
except Exception as e:
    print(f"API 호출 실패: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"상세 에러: {e.response.text}")
