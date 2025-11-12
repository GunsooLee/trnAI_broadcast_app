# 프론트엔드 개발자를 위한 API 가이드

## 📡 방송 편성 추천 API

### Endpoint
```
POST /api/v1/broadcast/recommendations
```

### Request

```typescript
interface BroadcastRequest {
  broadcastTime: string;        // ISO 8601 형식 (예: "2025-11-11T22:00:00+09:00")
  recommendationCount: number;  // 추천 개수 (기본값: 5)
  trendWeight?: number;         // 트렌드 가중치 0.0~1.0 (기본값: 0.3)
  sellingWeight?: number;       // 매출 가중치 0.0~1.0 (기본값: 0.7)
}
```

**예시:**
```json
{
  "broadcastTime": "2025-11-11T22:00:00+09:00",
  "recommendationCount": 3,
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
  naverProducts: NaverProduct[] | null;
  competitorProducts: CompetitorProduct[] | null;
}

interface BroadcastRecommendation {
  rank: number;
  productInfo: ProductInfo;
  reasoning: Reasoning;
  businessMetrics: BusinessMetrics;
}

interface ProductInfo {
  productId: string;
  productName: string;
  category: string;
  brand?: string;
  price?: number;
  tapeCode?: string;
  tapeName?: string;
}

interface Reasoning {
  summary: string;           // LangChain이 생성한 추천 근거 (50자 이내)
}

interface BusinessMetrics {
  aiPredictedSales: string;  // AI 예측 매출 (예: "850.0만원")
  lastBroadcast?: LastBroadcastMetrics;  // 최근 방송 실적 (Netezza 조회)
}

interface LastBroadcastMetrics {
  broadcastStartTime: string;  // 방송시작일시
  orderQuantity: number;       // 주문수량
  totalProfit: number;         // 매출총이익(실적)
  profitEfficiency: number;    // ONAIR매출총이익(효율)
  conversionWorth: number;     // 환산가치값(분리송출)
  conversionRate: number;      // 적용전환율
  realFee: number;             // 실질수수료
  mixFee: number;              // 혼합수수료
}

interface NaverProduct {
  product_id: string;
  name: string;
  rank: number;
  rank_change: number | null;
  rank_change_text: string;    // "↑2", "↓3", "신규", "-"
  sale_price: number;
  discounted_price: number;
  discount_ratio: number;
  image_url: string;
  landing_url: string;
  mobile_landing_url: string;
  is_delivery_free: boolean;
  delivery_fee: number;
  cumulation_sale_count: number;
  review_count: number | null;
  review_score: number | null;
  mall_name: string | null;
  channel_no: string | null;
  collected_at: string | null;
  collected_date: string | null;
}

interface CompetitorProduct {
  // TODO: 크롤링 서버에서 데이터 받으면 필드 정의 예정
}
```

---

## 🎨 UI 구현 예시

### 1. 추천 상품 카드

```tsx
import React from 'react';

interface RecommendationCardProps {
  recommendation: BroadcastRecommendation;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation }) => {
  const { productInfo, reasoning, businessMetrics } = recommendation;
  
  return (
    <div className="recommendation-card">
      {/* 헤더 */}
      <div className="card-header">
        <span className="rank-badge">#{recommendation.rank}</span>
      </div>

      {/* 상품 정보 */}
      <div className="product-info">
        <h3>{productInfo.productName}</h3>
        <p className="category">{productInfo.category}</p>
        {productInfo.brand && <p className="brand">{productInfo.brand}</p>}
        {productInfo.price && <p className="price">{productInfo.price.toLocaleString()}원</p>}
      </div>

      {/* 추천 근거 */}
      <div className="reasoning">
        <p className="summary">{reasoning.summary}</p>
      </div>

      {/* 비즈니스 지표 */}
      <div className="business-metrics">
        <div className="metric">
          <span className="label">AI 예측 매출</span>
          <span className="value">{businessMetrics.aiPredictedSales}</span>
        </div>
        
        {/* 최근 방송 실적 */}
        {businessMetrics.lastBroadcast && (
          <div className="last-broadcast">
            <h4>최근 방송 실적</h4>
            <div className="broadcast-date">
              {new Date(businessMetrics.lastBroadcast.broadcastStartTime).toLocaleDateString('ko-KR')}
            </div>
            <div className="metrics-grid">
              <div className="metric">
                <span className="label">주문수량</span>
                <span className="value">{businessMetrics.lastBroadcast.orderQuantity.toLocaleString()}개</span>
              </div>
              <div className="metric">
                <span className="label">매출총이익</span>
                <span className="value">{businessMetrics.lastBroadcast.totalProfit.toLocaleString()}원</span>
              </div>
              <div className="metric">
                <span className="label">효율</span>
                <span className="value">{businessMetrics.lastBroadcast.profitEfficiency}</span>
              </div>
              <div className="metric">
                <span className="label">전환율</span>
                <span className="value">{businessMetrics.lastBroadcast.conversionRate}%</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 방송테이프 정보 */}
      {productInfo.tapeCode && (
        <div className="tape-info">
          <span className="tape-code">{productInfo.tapeCode}</span>
          <span className="tape-name">{productInfo.tapeName}</span>
        </div>
      )}
    </div>
  );
};
```

### 2. 외부 상품 (네이버 베스트) 카드

```tsx
interface ExternalProductCardProps {
  product: ExternalProduct;
}

export const ExternalProductCard: React.FC<ExternalProductCardProps> = ({ product }) => {
  return (
    <div className="external-product-card">
      {/* 순위 및 변동 */}
      <div className="rank-section">
        <span className="rank">{product.rank}위</span>
        <span className={`rank-change ${product.rank_change_text}`}>
          {product.rank_change_text}
        </span>
      </div>

      {/* 상품 이미지 */}
      <img src={product.image_url} alt={product.name} />

      {/* 상품 정보 */}
      <h4>{product.name}</h4>
      
      {/* 가격 정보 */}
      <div className="price-section">
        {product.discount_ratio > 0 && (
          <>
            <span className="original-price">{product.sale_price.toLocaleString()}원</span>
            <span className="discount-badge">{product.discount_ratio}%</span>
          </>
        )}
        <span className="discounted-price">{product.discounted_price.toLocaleString()}원</span>
      </div>

      {/* 배송 정보 */}
      <div className="delivery-info">
        {product.is_delivery_free && <span className="badge">무료배송</span>}
      </div>

      {/* 리뷰 정보 */}
      {product.review_count && (
        <div className="review-info">
          <span className="rating">⭐ {product.review_score}</span>
          <span className="count">({product.review_count.toLocaleString()})</span>
        </div>
      )}

      {/* 판매량 */}
      <div className="sales-info">
        누적 판매: {product.cumulation_sale_count.toLocaleString()}개
      </div>

      {/* 링크 */}
      <a href={product.landing_url} target="_blank" rel="noopener noreferrer">
        상품 보기
      </a>
    </div>
  );
};
```

### 3. API 호출 예시

```typescript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8501';

export const getBroadcastRecommendations = async (
  broadcastTime: string,
  recommendationCount: number = 5,
  trendWeight: number = 0.3,
  sellingWeight: number = 0.7
): Promise<BroadcastResponse> => {
  try {
    const response = await axios.post<BroadcastResponse>(
      `${API_BASE_URL}/api/v1/broadcast/recommendations`,
      {
        broadcastTime,
        recommendationCount,
        trendWeight,
        sellingWeight,
      }
    );
    return response.data;
  } catch (error) {
    console.error('API 호출 실패:', error);
    throw error;
  }
};

// 사용 예시
const fetchRecommendations = async () => {
  const broadcastTime = '2025-11-11T22:00:00+09:00';
  const data = await getBroadcastRecommendations(broadcastTime, 5);
  
  console.log('추천 상품:', data.recommendations);
  console.log('네이버 상품:', data.naverProducts);
  console.log('경쟁사 상품:', data.competitorProducts);
};
```

---

## 💡 주요 포인트

### 1. AI 예측 vs 실제 매출 비교
```tsx
const ComparisonView = ({ metrics }: { metrics: BusinessMetrics }) => {
  const aiPrediction = parseFloat(metrics.aiPredictedSales.replace(/[^0-9.]/g, ''));
  const actualSales = metrics.lastBroadcast 
    ? metrics.lastBroadcast.totalProfit / 10000 
    : null;
  
  return (
    <div className="comparison">
      <div className="ai-prediction">
        <span>AI 예측</span>
        <strong>{metrics.aiPredictedSales}</strong>
      </div>
      {actualSales && (
        <div className="actual-sales">
          <span>최근 실적</span>
          <strong>{actualSales.toFixed(0)}만원</strong>
        </div>
      )}
    </div>
  );
};
```

### 2. 순위 변동 표시
```tsx
const RankChangeIcon = ({ text }: { text: string }) => {
  if (text === '신규') return <span className="new-badge">NEW</span>;
  if (text.startsWith('↑')) return <span className="rank-up">{text}</span>;
  if (text.startsWith('↓')) return <span className="rank-down">{text}</span>;
  return <span className="rank-same">-</span>;
};
```

---

## 📝 실제 응답 예시

전체 응답 예시는 [`API_RESPONSE_EXAMPLE.json`](./API_RESPONSE_EXAMPLE.json) 파일을 참고하세요.

**현업 담당자를 위한 한국어 설명서**: [`API_결과_필드_설명서.md`](./API_결과_필드_설명서.md)

---

## 🔧 에러 처리

```typescript
try {
  const data = await getBroadcastRecommendations(broadcastTime, 5);
  // 성공 처리
} catch (error) {
  if (axios.isAxiosError(error)) {
    if (error.response?.status === 400) {
      // 잘못된 요청 (예: trendWeight + salesWeight != 1.0)
      alert('가중치 합이 1.0이 되어야 합니다.');
    } else if (error.response?.status === 500) {
      // 서버 오류
      alert('서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
    }
  }
}
```

---

## 🎯 체크리스트

- [ ] `lastBroadcast` 필드가 `null`일 수 있음을 고려한 UI 처리
- [ ] 날짜/시간 포맷팅 (ISO 8601 → 한국 시간)
- [ ] 숫자 포맷팅 (천 단위 콤마)
- [ ] 이미지 로딩 실패 처리
- [ ] 외부 링크 새 탭에서 열기 (`target="_blank"`)
- [ ] 반응형 디자인 (모바일/태블릿/데스크톱)
- [ ] `competitorProducts`는 현재 빈 배열로 반환됨 (향후 데이터 추가 예정)
