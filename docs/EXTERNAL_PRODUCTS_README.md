# 외부 상품 (네이버 베스트) 문서 모음

## 📚 문서 목록

### 1. **EXTERNAL_PRODUCTS_API.md** (15KB)
**대상**: 백엔드 개발자, 시스템 관리자

**내용**:
- 📋 API 개요 및 주요 기능
- 📊 API 응답 구조 (JSON)
- 📖 **externalProducts 필드 상세 설명** (20개 필드)
  - 기본 정보 (product_id, name)
  - 순위 정보 (rank, rank_change, rank_change_text)
  - 가격 정보 (sale_price, discounted_price, discount_ratio)
  - 배송 정보 (is_delivery_free, delivery_fee)
  - 리뷰 정보 (review_count, review_score)
  - 판매자 정보 (mall_name, channel_no)
  - 링크 정보 (image_url, landing_url, mobile_landing_url)
  - 수집 정보 (collected_at, collected_date)
- 🔧 구현 상세 (DB 스키마, 서비스 클래스, 워크플로우)
- 📈 데이터 수집 프로세스 (n8n 워크플로우)
- 🧪 테스트 방법
- 🔄 유지보수

**추천 독자**: 시스템 아키텍처를 이해하고 싶은 개발자

---

### 2. **EXTERNAL_PRODUCTS_FRONTEND_GUIDE.md** (14KB)
**대상**: 프론트엔드 개발자

**내용**:
- 🎯 API 엔드포인트 및 요청/응답 예시
- 📊 TypeScript 인터페이스 정의
- 🎨 UI 컴포넌트 예시 (React)
  - ExternalProductCard (상품 카드)
  - ExternalProductsList (상품 리스트)
- 🎨 CSS 스타일 예시 (완전한 스타일시트)
- 🔍 필터링 및 정렬 예시
  - 급상승 상품, 고평가 상품, 할인 상품, 무료배송 상품
  - 순위순, 리뷰순, 평점순, 할인율순, 가격순
- 📱 반응형 디자인 (모바일/태블릿/데스크톱)
- 🎯 사용자 경험 개선 팁
  - 로딩 상태, 에러 처리, 상품 상세 모달, 외부 링크 추적
- 🚀 성능 최적화
  - 이미지 레이지 로딩, 가상 스크롤
- 📊 분석 및 추적

**추천 독자**: UI를 구현할 프론트엔드 개발자

---

### 3. **EXTERNAL_PRODUCTS_PD_GUIDE.md** (9.7KB)
**대상**: PD (프로그램 디렉터), 편성팀

**내용**:
- 📋 기능 개요 (비기술적 설명)
- 📊 화면 구성 예시
- 🎯 활용 방법
  - 트렌드 파악 (급상승 상품, 신규 진입 상품)
  - 가격 경쟁력 분석 (할인율, 가격대별 분포)
  - 리뷰 분석 (고평가 상품, 신제품 vs 검증 상품)
  - 카테고리 트렌드 (인기 카테고리, 계절/시기별 변화)
  - 경쟁사 분석 (외부 vs 내부 상품 비교)
- 📊 실전 활용 시나리오
  - 시나리오 1: 오후 2시 방송 편성
  - 시나리오 2: 긴급 편성 변경
  - 시나리오 3: 주간 편성 회의
- 🎯 주의사항
- 📋 체크리스트

**추천 독자**: 방송 편성을 담당하는 PD

---

## 🎯 문서 선택 가이드

### 당신은 누구인가요?

#### 👨‍💻 **백엔드 개발자**
→ `EXTERNAL_PRODUCTS_API.md` 읽기  
→ DB 스키마, API 구조, 데이터 수집 프로세스 이해

#### 🎨 **프론트엔드 개발자**
→ `EXTERNAL_PRODUCTS_FRONTEND_GUIDE.md` 읽기  
→ UI 컴포넌트, 스타일, 필터링/정렬 구현

#### 📺 **PD / 편성팀**
→ `EXTERNAL_PRODUCTS_PD_GUIDE.md` 읽기  
→ 트렌드 파악, 편성 전략, 실전 활용 방법

#### 🔧 **시스템 관리자**
→ `EXTERNAL_PRODUCTS_API.md` 읽기  
→ n8n 워크플로우, 데이터 수집, 유지보수

#### 📊 **데이터 분석가**
→ `EXTERNAL_PRODUCTS_API.md` + `EXTERNAL_PRODUCTS_PD_GUIDE.md` 읽기  
→ 데이터 구조 + 비즈니스 활용 방법

---

## 📖 빠른 참조

### 주요 필드 (20개)

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

---

## 🚀 빠른 시작

### 개발자

```bash
# 1. API 테스트
curl -X POST "http://localhost:8501/api/v1/broadcast/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-11-06T14:00:00+09:00",
    "recommendationCount": 5,
    "trendWeight": 0.3,
    "salesWeight": 0.7
  }'

# 2. 응답에서 externalProducts 확인
# 3. 프론트엔드 가이드 참고하여 UI 구현
```

### PD

```
1. 방송 편성 추천 화면 접속
2. "네이버 베스트 상품 TOP 20" 섹션 확인
3. 급상승 상품 (↑5 이상) 체크
4. 신규 진입 상품 (신규 배지) 체크
5. 내부 추천과 비교하여 최종 편성 결정
```

---

## 📊 데이터 업데이트

- **자동 업데이트**: 매일 새벽 2시
- **수동 업데이트**: `curl "http://localhost:8501/api/v1/external/crawl-naver-best?max_products=20"`
- **데이터 소스**: 네이버 쇼핑 베스트 (snxbest.naver.com)
- **수집 개수**: TOP 20

---

## 🔗 관련 파일

### 백엔드
- `/backend/app/naver_best_crawler.py` - 크롤러
- `/backend/app/external_products_service.py` - 서비스
- `/backend/app/routers/external_products.py` - API 라우터
- `/backend/app/broadcast_workflow.py` - 워크플로우 통합
- `/backend/app/schemas.py` - Pydantic 스키마

### 데이터베이스
- `/database/create_external_products_table.sql` - 테이블 스키마

### 워크플로우
- `/n8n_workflows/naver_shopping_crawler_final.json` - n8n 워크플로우

### 스크립트
- `/scripts/update_external_products.py` - 수동 업데이트 스크립트

---

## 📞 문의

- **기술 문의**: 백엔드 개발팀
- **UI 문의**: 프론트엔드 개발팀
- **데이터 문의**: 데이터 분석팀
- **편성 문의**: 편성팀장

---

## 🎯 결론

**3개의 문서로 모든 역할을 커버합니다**:

1. **API 문서** → 개발자, 시스템 관리자
2. **프론트엔드 가이드** → UI 개발자
3. **PD 가이드** → 편성팀, 비즈니스 사용자

**각자의 역할에 맞는 문서를 읽고 활용하세요!** 📚✨
