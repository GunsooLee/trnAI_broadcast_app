# 외부 상품 (네이버 베스트) API 문서

## 📋 개요

방송 편성 추천 API에 **네이버 쇼핑 베스트 상품**을 추가하여, PD들이 외부 트렌드를 참고할 수 있도록 구현했습니다.

---

## 🎯 주요 기능

### 1. **자동 포함**
- 방송 편성 추천 API 호출 시 **자동으로 네이버 베스트 상품 TOP 20** 포함
- 입력 파라미터(`broadcastTime`, `recommendationCount` 등)와 **무관하게 항상 최신 데이터** 제공

### 2. **일별 이력 관리**
- 매일 새벽 2시 자동 크롤링 (n8n 워크플로우)
- 하루에 한 번만 INSERT, 같은 날 재실행 시 UPDATE
- 순위 변동 추적 가능 (전일 대비 상승/하락/유지)

### 3. **PD에게 유용한 정보**
- ✅ 순위 및 순위 변동 (↑3, ↓2, →, 신규)
- ✅ 상품명, 이미지, 링크
- ✅ 가격 정보 (정가, 할인가, 할인율)
- ✅ 배송 정보 (무료배송 여부, 배송비)
- ✅ 리뷰 정보 (평점, 리뷰 수)
- ✅ 판매자 정보

---

## 📊 API 응답 구조

### 기존 응답 (내부 상품)
```json
{
  "requestTime": "2025-11-06T14:00:00+09:00",
  "recommendedCategories": [...],
  "recommendations": [...]
}
```

### 신규 추가 (외부 상품)
```json
{
  "requestTime": "2025-11-06T14:00:00+09:00",
  "recommendedCategories": [...],
  "recommendations": [...],
  "externalProducts": [
    {
      "product_id": "83021087183",
      "name": "윤남텍 간편세척 초음파 가습기 / YN-101",
      "rank": 1,
      "rank_change": 0,
      "rank_change_text": "→",
      "sale_price": 88000,
      "discounted_price": 88000,
      "discount_ratio": 0,
      "image_url": "https://shopping-phinf.pstatic.net/...",
      "landing_url": "https://smartstore.naver.com/...",
      "mobile_landing_url": "https://smartstore.naver.com/...",
      "is_delivery_free": true,
      "delivery_fee": 0,
      "cumulation_sale_count": 0,
      "review_count": 34897,
      "review_score": 4.9,
      "mall_name": "윤남텍",
      "channel_no": "100123456",
      "collected_at": "2025-11-04T02:00:00",
      "collected_date": "2025-11-04"
    }
  ]
}
```

---

## 📖 externalProducts 필드 상세 설명

### 🔑 기본 정보

#### `product_id` (string)
- **설명**: 네이버 쇼핑 상품 고유 ID
- **예시**: `"83021087183"`
- **용도**: 상품 식별, 중복 제거, 링크 생성

#### `name` (string)
- **설명**: 상품명 (네이버에서 표시되는 전체 이름)
- **예시**: `"윤남텍 간편세척 초음파 가습기 / YN-101 / 네이버"`
- **특징**: 브랜드, 모델명, 옵션 등 포함

---

### 📈 순위 정보

#### `rank` (integer)
- **설명**: 현재 베스트 순위 (1~20위)
- **예시**: `1` (1위)
- **용도**: 인기도 파악, 정렬

#### `rank_change` (integer | null)
- **설명**: 전일 대비 순위 변동 (양수=상승, 음수=하락)
- **예시**: 
  - `3` → 3단계 상승 (4위 → 1위)
  - `-2` → 2단계 하락 (3위 → 5위)
  - `0` → 순위 유지
  - `null` → 신규 진입 (전날 데이터 없음)

#### `rank_change_text` (string)
- **설명**: 순위 변동 텍스트 (UI 표시용)
- **예시**: 
  - `"↑3"` → 3단계 상승
  - `"↓2"` → 2단계 하락
  - `"→"` → 순위 유지
  - `"신규"` → 신규 진입
- **용도**: 프론트엔드에서 바로 표시 가능

---

### 💰 가격 정보

#### `sale_price` (integer)
- **설명**: 정상 판매가 (정가)
- **예시**: `88000` (88,000원)
- **단위**: 원 (KRW)

#### `discounted_price` (integer)
- **설명**: 할인 적용가 (실제 판매가)
- **예시**: `88000` (88,000원)
- **특징**: 할인이 없으면 `sale_price`와 동일

#### `discount_ratio` (integer)
- **설명**: 할인율 (%)
- **예시**: 
  - `0` → 할인 없음
  - `30` → 30% 할인
- **계산**: `(sale_price - discounted_price) / sale_price * 100`

---

### 🚚 배송 정보

#### `is_delivery_free` (boolean)
- **설명**: 무료배송 여부
- **예시**: 
  - `true` → 무료배송
  - `false` → 배송비 있음
- **용도**: 배송비 무료 상품 필터링

#### `delivery_fee` (integer)
- **설명**: 배송비 (원)
- **예시**: 
  - `0` → 무료배송
  - `3000` → 배송비 3,000원
- **특징**: `is_delivery_free=true`이면 항상 `0`

---

### ⭐ 리뷰 정보

#### `review_count` (integer)
- **설명**: 리뷰 개수
- **예시**: `35093` (35,093개 리뷰)
- **특징**: 
  - `99999+`는 `99999`로 저장
  - 리뷰가 많을수록 신뢰도 높음
- **용도**: 인기도 지표, 신뢰도 평가

#### `review_score` (float)
- **설명**: 평균 리뷰 평점 (5점 만점)
- **예시**: `4.9` (⭐4.9)
- **범위**: `0.0` ~ `5.0`
- **용도**: 품질 지표, 고평가 상품 필터링

---

### 🏪 판매자 정보

#### `mall_name` (string | null)
- **설명**: 판매자 스토어명
- **예시**: `"윤남텍"`, `"아르뫼"`
- **특징**: 
  - 공식 브랜드 스토어인 경우 브랜드명
  - 일반 판매자인 경우 스토어명
  - 데이터 없으면 `null`

#### `channel_no` (string)
- **설명**: 네이버 스마트스토어 채널 번호
- **예시**: `"101135970"`
- **용도**: 판매자 식별, 스토어 페이지 링크 생성

---

### 🔗 링크 정보

#### `image_url` (string)
- **설명**: 상품 이미지 URL
- **예시**: `"https://shopping-phinf.pstatic.net/main_8302108/83021087183.jpg?type=f450"`
- **특징**: 네이버 CDN 이미지 (450x450 크기)
- **용도**: 썸네일 표시

#### `landing_url` (string)
- **설명**: PC 상품 페이지 URL
- **예시**: `"https://smartstore.naver.com/main/products/5476592524"`
- **용도**: PC에서 상품 상세 페이지로 이동

#### `mobile_landing_url` (string)
- **설명**: 모바일 상품 페이지 URL
- **예시**: `"https://smartstore.naver.com/main/products/5476592524"`
- **특징**: 대부분 `landing_url`과 동일 (반응형 웹)
- **용도**: 모바일에서 상품 상세 페이지로 이동

---

### 📊 판매 정보

#### `cumulation_sale_count` (integer)
- **설명**: 누적 판매량 (또는 리뷰 수)
- **예시**: `35093`
- **특징**: 
  - 네이버 API에서 직접 제공하지 않음
  - 현재는 `review_count`로 대체
- **용도**: 인기도 지표

---

### 📅 수집 정보

#### `collected_at` (string, ISO 8601)
- **설명**: 데이터 수집 시각 (타임스탬프)
- **예시**: `"2025-11-05T17:00:28.764253"`
- **형식**: `YYYY-MM-DDTHH:MM:SS.ffffff`
- **용도**: 데이터 신선도 확인

#### `collected_date` (string, ISO 8601)
- **설명**: 데이터 수집 날짜
- **예시**: `"2025-11-05"`
- **형식**: `YYYY-MM-DD`
- **용도**: 일별 이력 관리, 순위 변동 추적

---

### 📋 필드 요약표

| 카테고리 | 필드 | 타입 | 설명 |
|---------|------|------|------|
| **기본** | `product_id` | string | 상품 ID |
| | `name` | string | 상품명 |
| **순위** | `rank` | int | 현재 순위 (1~20) |
| | `rank_change` | int\|null | 순위 변동 |
| | `rank_change_text` | string | 변동 텍스트 (↑3, ↓2, →, 신규) |
| **가격** | `sale_price` | int | 정가 (원) |
| | `discounted_price` | int | 할인가 (원) |
| | `discount_ratio` | int | 할인율 (%) |
| **배송** | `is_delivery_free` | bool | 무료배송 여부 |
| | `delivery_fee` | int | 배송비 (원) |
| **리뷰** | `review_count` | int | 리뷰 개수 |
| | `review_score` | float | 평점 (0~5) |
| **판매자** | `mall_name` | string\|null | 스토어명 |
| | `channel_no` | string | 채널 번호 |
| **링크** | `image_url` | string | 이미지 URL |
| | `landing_url` | string | PC 링크 |
| | `mobile_landing_url` | string | 모바일 링크 |
| **기타** | `cumulation_sale_count` | int | 누적 판매량 |
| | `collected_at` | string | 수집 시각 |
| | `collected_date` | string | 수집 날짜 |

**총 20개 필드**

---

### 🎯 프론트엔드 개발자를 위한 활용 예시

#### 1. 순위 배지 표시
```typescript
function getRankBadge(product: ExternalProduct) {
  const { rank, rank_change_text } = product;
  return `${rank}위 ${rank_change_text}`;
}
// 출력: "1위 ↑3"
```

#### 2. 가격 표시 (할인 여부)
```typescript
function getPriceDisplay(product: ExternalProduct) {
  if (product.discount_ratio > 0) {
    return (
      <>
        <del>{product.sale_price.toLocaleString()}원</del>
        <strong>{product.discounted_price.toLocaleString()}원</strong>
        <span className="discount">{product.discount_ratio}% 할인</span>
      </>
    );
  }
  return <strong>{product.sale_price.toLocaleString()}원</strong>;
}
```

#### 3. 리뷰 평점 표시
```typescript
function getReviewDisplay(product: ExternalProduct) {
  if (product.review_count > 0) {
    return `⭐${product.review_score} (${product.review_count.toLocaleString()}개)`;
  }
  return "리뷰 없음";
}
```

#### 4. 배송 정보 표시
```typescript
function getDeliveryBadge(product: ExternalProduct) {
  return product.is_delivery_free 
    ? <Badge color="green">무료배송</Badge>
    : <Badge>배송비 {product.delivery_fee.toLocaleString()}원</Badge>;
}
```

#### 5. 급상승 상품 필터링
```typescript
function getHotProducts(products: ExternalProduct[]) {
  return products.filter(p => 
    p.rank_change !== null && p.rank_change > 5
  );
}
```

---

## 🔧 구현 상세

### 1. **데이터베이스 스키마**

**테이블**: `external_products`

```sql
CREATE TABLE external_products (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    name TEXT NOT NULL,
    rank_order INTEGER,
    sale_price INTEGER,
    discounted_price INTEGER,
    discount_ratio INTEGER DEFAULT 0,
    image_url TEXT,
    landing_url TEXT,
    mobile_landing_url TEXT,
    is_delivery_free BOOLEAN DEFAULT FALSE,
    delivery_fee INTEGER DEFAULT 0,
    cumulation_sale_count INTEGER DEFAULT 0,
    review_count INTEGER DEFAULT 0,
    review_score NUMERIC(3,1) DEFAULT 0.0,
    mall_name VARCHAR(200),
    channel_no VARCHAR(50),
    collected_at TIMESTAMP NOT NULL,
    collected_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 하루에 한 번만 INSERT
    CONSTRAINT unique_product_per_day UNIQUE (product_id, collected_date)
);
```

### 2. **서비스 클래스**

**파일**: `/backend/app/external_products_service.py`

```python
class ExternalProductsService:
    """네이버 베스트 상품 조회 서비스"""
    
    def get_latest_best_products(self, limit: int = 20) -> List[Dict]:
        """가장 최근 수집된 베스트 상품 TOP 20 조회"""
        # 최신 collected_date의 상품들을 rank_order 순으로 조회
        # 전일 대비 순위 변동 계산 (LEFT JOIN)
```

**주요 기능**:
- 최신 수집 날짜 자동 감지
- 순위순 정렬
- 전일 대비 순위 변동 계산 (`rank_change`, `rank_change_text`)

### 3. **워크플로우 통합**

**파일**: `/backend/app/broadcast_workflow.py`

```python
async def _format_response(...) -> BroadcastResponse:
    # 내부 상품 추천 생성
    recommendations = [...]
    
    # 외부 상품 조회 (항상 실행)
    external_products_data = self.external_products_service.get_latest_best_products(limit=20)
    external_products = [ExternalProduct(**product) for product in external_products_data]
    
    return BroadcastResponse(
        requestTime="",
        recommendedCategories=top_categories,
        recommendations=recommendations,
        externalProducts=external_products if external_products else None
    )
```

---

## 📈 데이터 수집 프로세스

### n8n 워크플로우

**파일**: `/n8n_workflows/naver_shopping_crawler_final.json`

**스케줄**: 매일 새벽 2시

**프로세스**:
1. FastAPI 엔드포인트 호출: `GET /api/v1/external/crawl-naver-best?max_products=20`
2. 네이버 베스트 API 크롤링 (`snxbest.naver.com`)
3. JSON 데이터 파싱
4. PostgreSQL UPSERT 쿼리 생성 및 실행
   ```sql
   ON CONFLICT (product_id, collected_date) DO UPDATE SET ...
   ```

---

## 🧪 테스트

### 1. API 테스트

```bash
curl -X POST "http://localhost:8501/api/v1/broadcast/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-11-06T14:00:00+09:00",
    "recommendationCount": 5,
    "trendWeight": 0.3,
    "salesWeight": 0.7
  }'
```

**응답 확인**:
- `externalProducts` 배열에 20개 상품 포함
- 각 상품의 순위, 가격, 리뷰 정보 확인

### 2. 데이터 확인

```sql
-- 최신 수집 데이터 확인
SELECT 
    rank_order,
    name,
    sale_price,
    discounted_price,
    discount_ratio,
    review_score,
    review_count,
    collected_date
FROM external_products
WHERE collected_date = (SELECT MAX(collected_date) FROM external_products)
ORDER BY rank_order ASC
LIMIT 20;

-- 순위 변동 추적
SELECT 
    p1.name,
    p1.rank_order as today_rank,
    p2.rank_order as yesterday_rank,
    (p2.rank_order - p1.rank_order) as rank_change
FROM external_products p1
LEFT JOIN external_products p2 
    ON p1.product_id = p2.product_id 
    AND p2.collected_date = CURRENT_DATE - INTERVAL '1 day'
WHERE p1.collected_date = CURRENT_DATE
ORDER BY p1.rank_order ASC;
```

---

## 📊 PD가 활용할 수 있는 정보

### 1. **트렌드 파악**
- 현재 네이버 쇼핑에서 가장 인기 있는 상품 TOP 20
- 순위 변동을 통한 급상승/하락 상품 식별

### 2. **가격 정보**
- 할인율이 높은 상품 (소비자 관심도 높음)
- 가격대별 인기 상품 분포

### 3. **리뷰 분석**
- 평점 4.5 이상 고평가 상품
- 리뷰 수가 많은 검증된 상품

### 4. **카테고리 트렌드**
- 어떤 카테고리의 상품이 베스트에 많이 올라오는지
- 계절/시기별 인기 카테고리 변화

### 5. **경쟁사 분석**
- 외부 시장에서 인기 있는 상품과 내부 상품 비교
- 가격 경쟁력 분석

---

## 🔄 유지보수

### 데이터 정리

```sql
-- 90일 이상 된 데이터 삭제 (선택적)
DELETE FROM external_products 
WHERE collected_date < CURRENT_DATE - INTERVAL '90 days';
```

### 크롤러 재실행

```bash
# 수동 크롤링
curl "http://localhost:8501/api/v1/external/crawl-naver-best?max_products=20"
```

---

## 📝 향후 개선 사항

1. **카테고리별 필터링**: 특정 카테고리만 조회
2. **순위 변동 알림**: 급상승/급하락 상품 알림
3. **가격 추적**: 가격 변동 이력 저장
4. **다른 플랫폼 추가**: 쿠팡, 11번가 등

---

## 🎯 결론

- ✅ 방송 편성 추천 API에 외부 상품 섹션 추가 완료
- ✅ 입력 파라미터와 무관하게 항상 최신 TOP 20 제공
- ✅ PD들이 외부 트렌드를 참고하여 편성 결정 가능
- ✅ 일별 이력 관리로 순위 변동 추적 가능

**PD들은 이제 내부 상품 추천 + 외부 베스트 상품을 함께 보고 최적의 편성을 결정할 수 있습니다!** 🎊
