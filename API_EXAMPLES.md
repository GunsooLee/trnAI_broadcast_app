# 홈쇼핑 방송 추천 시스템 API

## 🎯 방송 편성 AI 추천 API

**POST** `/api/v1/broadcast/recommendations`

### 입력값
```json
{
  "broadcastTime": "2025-09-15T22:40:00+09:00",
  "recommendationCount": 5
}
```

### 출력값
```json
{
  "requestTime": "2025-09-08T11:19:26+09:00",
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "건강식품",
      "reason": "'다이어트' 키워드와 관련성 높음",
      "predictedSales": "8.5억"
    },
    {
      "rank": 2,
      "name": "의류",
      "reason": "계절 트렌드 반영",
      "predictedSales": "6.2억"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "P001",
        "productName": "프리미엄 다이어트 보조제",
        "category": "건강식품",
        "tapeCode": "T001",
        "tapeName": "프리미엄 다이어트 보조제 방송테이프",
        "durationMinutes": 30
      },
      "reasoning": {
        "summary": "AI 예측 매출 85백만원으로 최고 수익 기대",
        "linkedCategories": ["건강식품", "다이어트"],
        "matchedKeywords": ["다이어트", "건강", "체중감량"]
      },
      "businessMetrics": {
        "pastAverageSales": "8.5억",
        "marginRate": 0.25,
        "stockLevel": "High"
      }
    },
    {
      "rank": 2,
      "productInfo": {
        "productId": "P002",
        "productName": "홈트레이닝 세트",
        "category": "스포츠용품",
        "tapeCode": "T002",
        "tapeName": "홈트레이닝 세트 완전정복",
        "durationMinutes": 45
      },
      "reasoning": {
        "summary": "트렌드 키워드 '운동'과 높은 연관성",
        "linkedCategories": ["스포츠용품", "건강"],
        "matchedKeywords": ["운동", "홈트", "피트니스"]
      },
      "businessMetrics": {
        "pastAverageSales": "7.0억",
        "marginRate": 0.30,
        "stockLevel": "Medium"
      }
    }
  ]
}
```

## 🔧 **시스템 아키텍처**

```
n8n (30분마다) → 외부 API 수집 → PostgreSQL 저장
                                        ↓
PD 웹페이지 → FastAPI → 트렌드 DB 조회 → XGBoost 예측 → 추천 결과
```

## 📊 **주요 특징**

1. **실시간 트렌드**: 30분마다 네이버/구글에서 수집
2. **AI 예측**: XGBoost로 매출 예측 (억 단위)
3. **2단계 워크플로우**: Track A(카테고리) + Track B(상품) 병렬 처리
4. **동적 근거**: LangChain으로 추천 이유 생성
5. **방송테이프 관리**: TPGMTAPE 테이블로 방송 가능 상품만 필터링
