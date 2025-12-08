# 외부 상품 (네이버 베스트) 문서

## 문서 목록

| 문서 | 대상 | 내용 |
|-----|------|------|
| `EXTERNAL_PRODUCTS_API.md` | 백엔드 개발자 | API 구조, DB 스키마, n8n 워크플로우 |
| `EXTERNAL_PRODUCTS_FRONTEND_GUIDE.md` | 프론트엔드 개발자 | UI 컴포넌트, 스타일, 필터링 |
| `EXTERNAL_PRODUCTS_PD_GUIDE.md` | PD/편성팀 | 트렌드 파악, 편성 전략, 활용 방법 |

---

## 빠른 시작

### API 테스트

```bash
curl -X POST "http://localhost:8501/api/v1/broadcast/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-12-08T14:00:00+09:00",
    "recommendationCount": 10
  }'
```

응답의 `competitorProducts` 필드에서 네이버 인기 상품 + 타사 편성 확인

---

## 데이터 수집

- **자동 수집**: 매일 새벽 2시 (n8n 워크플로우)
- **데이터 소스**: 네이버 쇼핑 베스트
- **수집 개수**: TOP 20

---

## 관련 파일

| 파일 | 설명 |
|-----|------|
| `backend/app/naver_best_crawler.py` | 크롤러 |
| `backend/app/external_products_service.py` | 서비스 |
| `backend/app/routers/external_products.py` | API 라우터 |
| `n8n_workflows/naver_shopping_crawler_final.json` | n8n 워크플로우 |
