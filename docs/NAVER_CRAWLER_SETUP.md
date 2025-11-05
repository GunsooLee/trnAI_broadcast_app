# 네이버 쇼핑 크롤러 설정 가이드

## 🚀 빠른 시작 (3단계)

### 1️⃣ 데이터베이스 테이블 생성

```bash
docker exec -i postgres_db psql -U postgres -d trnai < database/create_external_products_table.sql
```

### 2️⃣ FastAPI 재시작

```bash
docker restart fastapi_backend
sleep 5

# 헬스 체크
curl http://localhost:8501/api/v1/external/health
```

### 3️⃣ n8n 워크플로우 설정

1. n8n 접속: http://localhost:5678
2. **Import from File** → `n8n_workflows/naver_shopping_crawler_final.json`
3. **PostgreSQL Credential** 설정:
   - Host: `postgres_db`
   - Database: `trnai`
   - User: `postgres`
   - Password: (환경변수 참조)
4. **Active** 토글 ON

---

## 📊 워크플로우 구조

```
스케줄 트리거 (매일 새벽 2시)
    ↓
크롤러 API 호출 (FastAPI)
    ↓
쿼리 생성 (Code 노드)
    ↓
쿼리 실행 (PostgreSQL)
    ↓
결과 집계
    ↓
오래된 데이터 삭제 (7일 이상)
```

---

## 🧪 테스트

### API 직접 호출

```bash
# 5개 상품 크롤링
curl "http://localhost:8501/api/v1/external/crawl-naver-shopping?max_products=5"
```

### 데이터 확인

```bash
docker exec -it postgres_db psql -U postgres -d trnai
```

```sql
-- 최신 데이터 확인
SELECT product_id, name, sale_price, discount_ratio, collected_at 
FROM external_products 
ORDER BY collected_at DESC 
LIMIT 5;

-- 인기 상품 TOP 10
SELECT name, sale_price, discount_ratio, cumulation_sale_count
FROM external_products
WHERE collected_at >= NOW() - INTERVAL '1 day'
ORDER BY rank_order ASC
LIMIT 10;
```

---

## 📁 파일 구조

```
/home/trn/trnAi/
├── backend/app/
│   ├── naver_shopping_api_crawler.py       # 크롤러 핵심 로직
│   └── routers/
│       └── external_products.py            # FastAPI 라우터
├── database/
│   └── create_external_products_table.sql  # DB 스키마
├── n8n_workflows/
│   └── naver_shopping_crawler_final.json   # n8n 워크플로우
└── docs/
    └── NAVER_CRAWLER_SETUP.md              # 이 문서
```

---

## 🔧 트러블슈팅

### 문제: n8n에서 "there is no parameter $1" 에러

**원인:** PostgreSQL 노드의 파라미터 바인딩 문제

**해결:** Code 노드에서 SQL 쿼리를 직접 생성하도록 수정 (완료)

### 문제: API 호출 실패

```bash
# FastAPI 로그 확인
docker logs fastapi_backend --tail 50

# 컨테이너 상태 확인
docker ps | grep fastapi
```

### 문제: 데이터가 저장되지 않음

```sql
-- 테이블 존재 확인
\dt external_products

-- 권한 확인
GRANT ALL PRIVILEGES ON TABLE external_products TO postgres;
```

---

## 📈 유용한 쿼리

### 수집 통계

```sql
-- 일별 수집 상품 수
SELECT 
    DATE(collected_at) as date,
    COUNT(*) as count
FROM external_products
GROUP BY DATE(collected_at)
ORDER BY date DESC
LIMIT 7;

-- 평균 할인율
SELECT 
    AVG(discount_ratio) as avg_discount,
    MAX(discount_ratio) as max_discount
FROM external_products
WHERE collected_at >= NOW() - INTERVAL '1 day';
```

### 할인율 높은 상품

```sql
SELECT 
    name,
    sale_price,
    discounted_price,
    discount_ratio,
    landing_url
FROM external_products
WHERE discount_ratio > 30
ORDER BY discount_ratio DESC
LIMIT 10;
```

---

## ⚙️ 설정 변경

### 수집 주기 변경

n8n 워크플로우의 스케줄 트리거 수정:

- `0 2 * * *` - 매일 새벽 2시 (기본값)
- `0 */6 * * *` - 6시간마다
- `0 2,14 * * *` - 매일 2시, 14시

### 수집 상품 개수 변경

API URL 파라미터 수정:
```
http://fastapi_backend:8501/api/v1/external/crawl-naver-shopping?max_products=200
```

### 데이터 보관 기간 변경

"오래된 데이터 삭제" 노드의 쿼리 수정:
```sql
DELETE FROM external_products 
WHERE collected_at < NOW() - INTERVAL '30 days';  -- 30일로 변경
```

---

## ✅ 체크리스트

- [ ] PostgreSQL 테이블 생성
- [ ] FastAPI 재시작 및 헬스 체크
- [ ] n8n 워크플로우 임포트
- [ ] PostgreSQL Credential 설정
- [ ] 워크플로우 활성화
- [ ] 수동 테스트 실행
- [ ] 데이터 확인

**모두 완료하면 자동 크롤링 시작!** 🎉

---

## 🎯 다음 단계

1. ✅ 크롤러 설정 완료
2. 🔜 외부상품추천 API 개발
3. 🔜 프론트엔드 연동
4. 🔜 알림 시스템 구축
