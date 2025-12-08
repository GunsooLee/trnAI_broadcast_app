# 외부 상품 API 문서

> 백엔드 개발자 및 시스템 관리자용

---

## 개요

방송 편성 추천 API에 네이버 쇼핑 베스트 상품 + 타사 편성 정보가 `competitorProducts` 필드로 통합 제공됩니다.

**주요 특징:**
- 방송 추천 API 호출 시 자동 포함
- AI가 네이버 + 타사 중 10개 선택 (5:5 비율 유지)
- 매일 새벽 2시 자동 크롤링 (n8n)

---

## API 응답 구조

```json
{
  "requestTime": "2025-12-08T14:00:00+09:00",
  "recommendations": [...],
  "competitorProducts": [
    {
      "company_name": "네이버 스토어",
      "broadcast_title": "윤남텍 초음파 가습기",
      "start_time": null,
      "end_time": null,
      "duration_minutes": null,
      "category_main": "생활가전"
    },
    {
      "company_name": "GS홈쇼핑",
      "broadcast_title": "프리미엄 건강식품 세트",
      "start_time": "2025-12-08T20:00:00+09:00",
      "end_time": "2025-12-08T21:00:00+09:00",
      "duration_minutes": 60,
      "category_main": "건강식품"
    }
  ]
}
```

---

## 필드 설명

| 필드 | 타입 | 설명 |
|-----|------|------|
| `company_name` | string | 출처 ("네이버 스토어" 또는 타사명) |
| `broadcast_title` | string | 상품명 또는 방송 제목 |
| `start_time` | string\|null | 방송 시작 시간 (타사만) |
| `end_time` | string\|null | 방송 종료 시간 (타사만) |
| `duration_minutes` | int\|null | 방송 길이 (분, 타사만) |
| `category_main` | string | 대분류 카테고리 |

**구분 방법:**
- 네이버 상품: `company_name` = "네이버 스토어", 시간 필드 null
- 타사 편성: `company_name` = 실제 홈쇼핑사명, 시간 정보 포함

---

## DB 스키마

### external_products 테이블

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
    is_delivery_free BOOLEAN DEFAULT FALSE,
    delivery_fee INTEGER DEFAULT 0,
    review_count INTEGER DEFAULT 0,
    review_score NUMERIC(3,1) DEFAULT 0.0,
    mall_name VARCHAR(200),
    collected_at TIMESTAMP NOT NULL,
    collected_date DATE NOT NULL,
    CONSTRAINT unique_product_per_day UNIQUE (product_id, collected_date)
);
```

---

## 데이터 수집

### n8n 워크플로우

- **파일**: `n8n_workflows/naver_shopping_crawler_final.json`
- **스케줄**: 매일 새벽 2시
- **프로세스**:
  1. FastAPI 엔드포인트 호출
  2. 네이버 베스트 API 크롤링
  3. PostgreSQL UPSERT

### 수동 크롤링

```bash
curl "http://localhost:8501/api/v1/external/crawl-naver-best?max_products=20"
```

---

## 관련 파일

| 파일 | 설명 |
|-----|------|
| `backend/app/naver_best_crawler.py` | 크롤러 |
| `backend/app/external_products_service.py` | 서비스 |
| `backend/app/routers/external_products.py` | API 라우터 |
| `backend/app/broadcast_workflow.py` | 워크플로우 통합 |

---

## 유지보수

### 데이터 정리

```sql
-- 90일 이상 된 데이터 삭제
DELETE FROM external_products 
WHERE collected_date < CURRENT_DATE - INTERVAL '90 days';
```

### 데이터 확인

```sql
SELECT rank_order, name, sale_price, discount_ratio, collected_date
FROM external_products
WHERE collected_date = (SELECT MAX(collected_date) FROM external_products)
ORDER BY rank_order ASC
LIMIT 20;
```
