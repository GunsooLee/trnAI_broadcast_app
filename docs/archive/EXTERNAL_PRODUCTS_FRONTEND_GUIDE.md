# 외부 상품 (네이버 베스트) - 프론트엔드 개발 가이드

## 📋 개요

방송 편성 추천 API에서 반환하는 `externalProducts` 필드를 프론트엔드에서 표시하는 방법을 안내합니다.

---

## 🎯 API 엔드포인트

```
POST /api/v1/broadcast/recommendations
```

### 요청 예시

```typescript
const response = await fetch('http://localhost:8501/api/v1/broadcast/recommendations', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    broadcastTime: '2025-11-06T14:00:00+09:00',
    recommendationCount: 5,
    trendWeight: 0.3,
    salesWeight: 0.7
  })
});

const data = await response.json();
```

---

## 📊 응답 구조

```typescript
interface BroadcastResponse {
  requestTime: string;
  recommendedCategories: RecommendedCategory[];
  recommendations: BroadcastRecommendation[];
  externalProducts?: ExternalProduct[];  // ← 새로 추가된 필드
}

interface ExternalProduct {
  // 기본 정보
  product_id: string;
  name: string;
  
  // 순위 정보
  rank: number;                    // 1~20
  rank_change: number | null;      // 양수=상승, 음수=하락, null=신규
  rank_change_text: string;        // "↑3", "↓2", "→", "신규"
  
  // 가격 정보
  sale_price: number;              // 정가 (원)
  discounted_price: number;        // 할인가 (원)
  discount_ratio: number;          // 할인율 (%)
  
  // 배송 정보
  is_delivery_free: boolean;
  delivery_fee: number;            // 배송비 (원)
  
  // 리뷰 정보
  review_count: number;
  review_score: number;            // 0.0 ~ 5.0
  
  // 판매자 정보
  mall_name: string | null;
  channel_no: string;
  
  // 링크 정보
  image_url: string;
  landing_url: string;
  mobile_landing_url: string;
  
  // 기타
  cumulation_sale_count: number;
  collected_at: string;            // ISO 8601
  collected_date: string;          // YYYY-MM-DD
}
```

---

## 🎨 UI 컴포넌트 예시

### 1. 외부 상품 카드 (React)

```tsx
import React from 'react';

interface ExternalProductCardProps {
  product: ExternalProduct;
}

export const ExternalProductCard: React.FC<ExternalProductCardProps> = ({ product }) => {
  const hasDiscount = product.discount_ratio > 0;
  const hasReviews = product.review_count > 0;
  
  return (
    <div className="product-card">
      {/* 순위 배지 */}
      <div className="rank-badge">
        <span className="rank">{product.rank}위</span>
        <span className={`rank-change ${getRankChangeClass(product.rank_change)}`}>
          {product.rank_change_text}
        </span>
      </div>
      
      {/* 상품 이미지 */}
      <a href={product.landing_url} target="_blank" rel="noopener noreferrer">
        <img 
          src={product.image_url} 
          alt={product.name}
          className="product-image"
        />
      </a>
      
      {/* 상품명 */}
      <h3 className="product-name" title={product.name}>
        {truncateText(product.name, 50)}
      </h3>
      
      {/* 가격 정보 */}
      <div className="price-section">
        {hasDiscount ? (
          <>
            <span className="original-price">
              {product.sale_price.toLocaleString()}원
            </span>
            <div className="discount-info">
              <span className="discount-ratio">{product.discount_ratio}%</span>
              <span className="discounted-price">
                {product.discounted_price.toLocaleString()}원
              </span>
            </div>
          </>
        ) : (
          <span className="price">
            {product.sale_price.toLocaleString()}원
          </span>
        )}
      </div>
      
      {/* 리뷰 정보 */}
      {hasReviews && (
        <div className="review-section">
          <span className="review-score">⭐ {product.review_score}</span>
          <span className="review-count">
            ({product.review_count.toLocaleString()}개)
          </span>
        </div>
      )}
      
      {/* 배송 정보 */}
      <div className="delivery-section">
        {product.is_delivery_free ? (
          <span className="badge badge-success">무료배송</span>
        ) : (
          <span className="badge">배송비 {product.delivery_fee.toLocaleString()}원</span>
        )}
      </div>
      
      {/* 판매자 정보 */}
      {product.mall_name && (
        <div className="seller-info">
          <span className="seller-name">{product.mall_name}</span>
        </div>
      )}
    </div>
  );
};

// 헬퍼 함수들
function getRankChangeClass(rankChange: number | null): string {
  if (rankChange === null) return 'new';
  if (rankChange > 0) return 'up';
  if (rankChange < 0) return 'down';
  return 'same';
}

function truncateText(text: string, maxLength: number): string {
  return text.length > maxLength 
    ? text.substring(0, maxLength) + '...' 
    : text;
}
```

### 2. 외부 상품 리스트

```tsx
interface ExternalProductsListProps {
  products: ExternalProduct[];
}

export const ExternalProductsList: React.FC<ExternalProductsListProps> = ({ products }) => {
  if (!products || products.length === 0) {
    return (
      <div className="empty-state">
        <p>외부 상품 데이터가 없습니다.</p>
      </div>
    );
  }
  
  return (
    <section className="external-products-section">
      <div className="section-header">
        <h2>🌐 네이버 베스트 상품 TOP 20</h2>
        <p className="subtitle">실시간 인기 상품을 참고하세요</p>
      </div>
      
      <div className="products-grid">
        {products.map((product) => (
          <ExternalProductCard 
            key={product.product_id} 
            product={product} 
          />
        ))}
      </div>
    </section>
  );
};
```

---

## 🎨 CSS 스타일 예시

```css
/* 외부 상품 섹션 */
.external-products-section {
  margin-top: 40px;
  padding: 24px;
  background: #f8f9fa;
  border-radius: 12px;
}

.section-header {
  margin-bottom: 24px;
}

.section-header h2 {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 8px;
}

.section-header .subtitle {
  color: #6c757d;
  font-size: 14px;
}

/* 상품 그리드 */
.products-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
}

/* 상품 카드 */
.product-card {
  background: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s, box-shadow 0.2s;
  position: relative;
}

.product-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

/* 순위 배지 */
.rank-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  z-index: 1;
}

.rank-change.up {
  color: #ff4444;
}

.rank-change.down {
  color: #4444ff;
}

.rank-change.new {
  color: #ffaa00;
}

.rank-change.same {
  color: #888;
}

/* 상품 이미지 */
.product-image {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 12px;
}

/* 상품명 */
.product-name {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 8px;
  line-height: 1.4;
  height: 40px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

/* 가격 정보 */
.price-section {
  margin-bottom: 8px;
}

.original-price {
  font-size: 12px;
  color: #999;
  text-decoration: line-through;
  display: block;
  margin-bottom: 4px;
}

.discount-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.discount-ratio {
  font-size: 14px;
  font-weight: bold;
  color: #ff4444;
}

.discounted-price,
.price {
  font-size: 16px;
  font-weight: bold;
  color: #333;
}

/* 리뷰 정보 */
.review-section {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  margin-bottom: 8px;
}

.review-score {
  font-weight: bold;
  color: #ffa500;
}

.review-count {
  color: #666;
}

/* 배송 정보 */
.delivery-section {
  margin-bottom: 8px;
}

.badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  background: #e9ecef;
  color: #495057;
}

.badge-success {
  background: #d4edda;
  color: #155724;
}

/* 판매자 정보 */
.seller-info {
  font-size: 12px;
  color: #666;
  padding-top: 8px;
  border-top: 1px solid #eee;
}

.seller-name {
  font-weight: 500;
}

/* 빈 상태 */
.empty-state {
  text-align: center;
  padding: 40px;
  color: #999;
}
```

---

## 🔍 필터링 및 정렬 예시

### 1. 급상승 상품 필터링

```typescript
function getHotProducts(products: ExternalProduct[]): ExternalProduct[] {
  return products.filter(p => 
    p.rank_change !== null && p.rank_change > 5
  );
}
```

### 2. 고평가 상품 필터링

```typescript
function getHighRatedProducts(products: ExternalProduct[]): ExternalProduct[] {
  return products.filter(p => 
    p.review_score >= 4.8 && p.review_count > 10000
  );
}
```

### 3. 할인 상품 필터링

```typescript
function getDiscountedProducts(products: ExternalProduct[]): ExternalProduct[] {
  return products.filter(p => p.discount_ratio > 30);
}
```

### 4. 무료배송 상품 필터링

```typescript
function getFreeShippingProducts(products: ExternalProduct[]): ExternalProduct[] {
  return products.filter(p => p.is_delivery_free);
}
```

### 5. 정렬

```typescript
// 순위순 (기본)
products.sort((a, b) => a.rank - b.rank);

// 리뷰 많은 순
products.sort((a, b) => b.review_count - a.review_count);

// 평점 높은 순
products.sort((a, b) => b.review_score - a.review_score);

// 할인율 높은 순
products.sort((a, b) => b.discount_ratio - a.discount_ratio);

// 가격 낮은 순
products.sort((a, b) => a.discounted_price - b.discounted_price);
```

---

## 📱 반응형 디자인

```css
/* 모바일 (< 768px) */
@media (max-width: 767px) {
  .products-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  
  .product-card {
    padding: 12px;
  }
  
  .product-name {
    font-size: 13px;
  }
}

/* 태블릿 (768px ~ 1024px) */
@media (min-width: 768px) and (max-width: 1024px) {
  .products-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

/* 데스크톱 (> 1024px) */
@media (min-width: 1025px) {
  .products-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

---

## 🎯 사용자 경험 개선 팁

### 1. 로딩 상태

```tsx
{isLoading ? (
  <div className="loading-skeleton">
    {[...Array(20)].map((_, i) => (
      <div key={i} className="skeleton-card" />
    ))}
  </div>
) : (
  <ExternalProductsList products={externalProducts} />
)}
```

### 2. 에러 처리

```tsx
{error ? (
  <div className="error-state">
    <p>외부 상품을 불러오는 중 오류가 발생했습니다.</p>
    <button onClick={retry}>다시 시도</button>
  </div>
) : (
  <ExternalProductsList products={externalProducts} />
)}
```

### 3. 상품 상세 모달

```tsx
const [selectedProduct, setSelectedProduct] = useState<ExternalProduct | null>(null);

// 카드 클릭 시
<div onClick={() => setSelectedProduct(product)}>
  <ExternalProductCard product={product} />
</div>

// 모달
{selectedProduct && (
  <ProductDetailModal 
    product={selectedProduct}
    onClose={() => setSelectedProduct(null)}
  />
)}
```

### 4. 외부 링크 추적

```tsx
function handleProductClick(product: ExternalProduct) {
  // 분석 이벤트 전송
  analytics.track('external_product_click', {
    product_id: product.product_id,
    rank: product.rank,
    name: product.name
  });
  
  // 새 탭에서 열기
  window.open(product.landing_url, '_blank');
}
```

---

## 🚀 성능 최적화

### 1. 이미지 레이지 로딩

```tsx
<img 
  src={product.image_url} 
  alt={product.name}
  loading="lazy"
/>
```

### 2. 가상 스크롤 (많은 상품 표시 시)

```tsx
import { FixedSizeGrid } from 'react-window';

<FixedSizeGrid
  columnCount={4}
  columnWidth={220}
  height={600}
  rowCount={Math.ceil(products.length / 4)}
  rowHeight={350}
  width={900}
>
  {({ columnIndex, rowIndex, style }) => (
    <div style={style}>
      <ExternalProductCard 
        product={products[rowIndex * 4 + columnIndex]} 
      />
    </div>
  )}
</FixedSizeGrid>
```

---

## 📊 분석 및 추적

```typescript
// 외부 상품 섹션 노출
useEffect(() => {
  if (externalProducts && externalProducts.length > 0) {
    analytics.track('external_products_viewed', {
      count: externalProducts.length,
      top_product: externalProducts[0].name
    });
  }
}, [externalProducts]);

// 상품 클릭
function trackProductClick(product: ExternalProduct) {
  analytics.track('external_product_clicked', {
    product_id: product.product_id,
    rank: product.rank,
    name: product.name,
    price: product.discounted_price,
    has_discount: product.discount_ratio > 0
  });
}
```

---

## 🎯 결론

- ✅ `externalProducts` 배열을 받아서 UI에 표시
- ✅ 순위, 가격, 리뷰, 배송 정보를 명확하게 표시
- ✅ 반응형 디자인으로 모바일/태블릿/데스크톱 대응
- ✅ 필터링/정렬 기능으로 사용자 경험 향상
- ✅ 로딩/에러 상태 처리
- ✅ 성능 최적화 (레이지 로딩, 가상 스크롤)

**PD와 사용자가 외부 트렌드를 쉽게 파악할 수 있는 직관적인 UI를 구현하세요!** 🎨
