# 외부 상품 - 프론트엔드 가이드

> 프론트엔드 개발자용

---

## API 응답

방송 추천 API 응답의 `competitorProducts` 필드에서 외부 상품 데이터를 받습니다.

```typescript
interface CompetitorProduct {
  company_name: string;            // "네이버 스토어" 또는 타사명
  broadcast_title: string;         // 상품명/방송 제목
  start_time: string | null;       // 타사만
  end_time: string | null;         // 타사만
  duration_minutes: number | null; // 타사만
  category_main: string;
}
```

**구분 방법:**
- 네이버 상품: `company_name` = "네이버 스토어", 시간 필드 null
- 타사 편성: `company_name` = 실제 홈쇼핑사명, 시간 정보 포함

---

## UI 컴포넌트 예시

### 외부 상품 카드

```tsx
interface CompetitorProductCardProps {
  product: CompetitorProduct;
}

export const CompetitorProductCard = ({ product }: CompetitorProductCardProps) => {
  const isNaverProduct = product.company_name === '네이버 스토어';
  
  return (
    <div className="competitor-card">
      <div className="source-badge">
        {isNaverProduct ? '네이버 인기' : product.company_name}
      </div>
      <h4>{product.broadcast_title}</h4>
      <span className="category">{product.category_main}</span>
      {!isNaverProduct && product.start_time && (
        <div className="time-info">
          {formatTime(product.start_time)} ~ {formatTime(product.end_time)}
          ({product.duration_minutes}분)
        </div>
      )}
    </div>
  );
};

const formatTime = (isoString: string | null) => {
  if (!isoString) return '';
  return new Date(isoString).toLocaleTimeString('ko-KR', { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
};
```

### 외부 상품 리스트

```tsx
export const CompetitorProductsList = ({ products }: { products: CompetitorProduct[] }) => {
  if (!products?.length) {
    return <p>외부 상품 데이터가 없습니다.</p>;
  }

  const naverProducts = products.filter(p => p.company_name === '네이버 스토어');
  const competitorProducts = products.filter(p => p.company_name !== '네이버 스토어');

  return (
    <div className="competitor-products">
      {competitorProducts.length > 0 && (
        <section>
          <h3>타사 편성</h3>
          <div className="product-grid">
            {competitorProducts.map((p, i) => (
              <CompetitorProductCard key={i} product={p} />
            ))}
          </div>
        </section>
      )}
      {naverProducts.length > 0 && (
        <section>
          <h3>네이버 인기 상품</h3>
          <div className="product-grid">
            {naverProducts.map((p, i) => (
              <CompetitorProductCard key={i} product={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
```

---

## CSS 스타일

```css
.competitor-card {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
  background: white;
}

.source-badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  background: #f5f5f5;
  margin-bottom: 8px;
}

.competitor-card h4 {
  margin: 8px 0;
  font-size: 14px;
  line-height: 1.4;
}

.category {
  color: #666;
  font-size: 12px;
}

.time-info {
  margin-top: 8px;
  font-size: 12px;
  color: #888;
}

.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}
```

---

## 활용 팁

1. **타사 편성 우선 표시**: 경쟁사 분석이 더 중요하므로 타사 편성을 먼저 배치
2. **시간대 강조**: 현재 방송 시간과 겹치는 타사 편성은 강조 표시
3. **카테고리 필터**: 카테고리별 필터링 기능 제공
4. **반응형 디자인**: 모바일에서는 카드를 세로로 배치

---

## 관련 문서

- `docs/FRONTEND_API_GUIDE.md` - 전체 API 가이드
- `docs/API_RESPONSE_EXAMPLE.json` - 응답 예시
