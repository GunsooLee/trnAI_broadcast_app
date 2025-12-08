# 프론트엔드 API 가이드

## 방송 편성 추천 API

### Endpoint
```
POST /api/v1/broadcast/recommendations
```

### Request

```typescript
interface BroadcastRequest {
  broadcastTime: string;        // ISO 8601 형식
  recommendationCount: number;  // 추천 개수 (기본값: 10)
  trendWeight?: number;         // 트렌드 가중치 0.0~1.0 (기본값: 0.3)
  sellingWeight?: number;       // 매출 가중치 0.0~1.0 (기본값: 0.7)
}
```

**예시:**
```json
{
  "broadcastTime": "2025-12-08T22:00:00+09:00",
  "recommendationCount": 10,
  "trendWeight": 0.3,
  "sellingWeight": 0.7
}
```

---

### Response

```typescript
interface BroadcastResponse {
  requestTime: string;
  recommendations: BroadcastRecommendation[];
  competitorProducts: CompetitorProduct[] | null;  // 네이버 + 타사 통합
}

interface BroadcastRecommendation {
  rank: number;
  productInfo: ProductInfo;
  reasoning: string;  // 추천 근거 (100자 이내)
  businessMetrics: BusinessMetrics;
}

interface ProductInfo {
  productId: string;
  productName: string;
  category: string;
  categoryMiddle?: string;
  categorySub?: string;
  brand?: string;
  price?: number;
  tapeCode?: string;
  tapeName?: string;
}

interface BusinessMetrics {
  aiPredictedSales: string;
  lastBroadcast?: LastBroadcastMetrics;
}

interface LastBroadcastMetrics {
  broadcastStartTime: string;
  orderQuantity: number;
  totalProfit: number;
  profitEfficiency: number;
  conversionWorth: number;
  conversionRate: number;
  realFee: number;
  mixFee: number;
}

interface CompetitorProduct {
  company_name: string;            // "네이버 스토어" 또는 타사명
  broadcast_title: string;
  start_time: string | null;       // 타사만
  end_time: string | null;         // 타사만
  duration_minutes: number | null; // 타사만
  category_main: string;
}
```

---

## UI 구현 예시

### 추천 상품 카드

```tsx
export const RecommendationCard = ({ recommendation }) => {
  const { productInfo, reasoning, businessMetrics } = recommendation;
  
  return (
    <div className="recommendation-card">
      <span className="rank-badge">#{recommendation.rank}</span>
      <h3>{productInfo.productName}</h3>
      <p className="category">{productInfo.category}</p>
      <p className="reasoning">{reasoning}</p>
      <div className="metrics">
        <span>AI 예측: {businessMetrics.aiPredictedSales}</span>
        {businessMetrics.lastBroadcast && (
          <span>최근 실적: {businessMetrics.lastBroadcast.totalProfit.toLocaleString()}원</span>
        )}
      </div>
    </div>
  );
};
```

### API 호출

```typescript
const API_BASE_URL = 'http://localhost:8501';

export const getBroadcastRecommendations = async (
  broadcastTime: string,
  recommendationCount: number = 10
): Promise<BroadcastResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/broadcast/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      broadcastTime,
      recommendationCount,
      trendWeight: 0.3,
      sellingWeight: 0.7,
    }),
  });
  return response.json();
};
```

---

## 에러 처리

| 상태 코드 | 원인 | 처리 |
|----------|------|------|
| 400 | 잘못된 요청 (가중치 합 ≠ 1.0) | 입력값 검증 |
| 500 | 서버 오류 | 재시도 안내 |
| 503 | AI 서비스 일시 중단 | 잠시 후 재시도 |

---

## 체크리스트

- [ ] `lastBroadcast` 필드 null 처리
- [ ] 날짜/시간 포맷팅 (ISO 8601 → 한국 시간)
- [ ] 숫자 포맷팅 (천 단위 콤마)
- [ ] 반응형 디자인

---

## 관련 문서

- `docs/API_RESPONSE_EXAMPLE.json` - 실제 응답 예시
- `docs/API_결과_필드_설명서.md` - 현업 담당자용 설명서
