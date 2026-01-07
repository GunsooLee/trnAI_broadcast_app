# AI 기반 홈쇼핑 방송 편성 추천 시스템

최종 업데이트: 2026-01-06
문서 버전: 2.0 (소스 코드 기반 검증 완료)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [추천 워크플로우](#3-추천-워크플로우)
4. [API 명세](#4-api-명세)
5. [점수 계산 로직](#5-점수-계산-로직)
6. [추천 근거 형식](#6-추천-근거-형식)
7. [데이터베이스 스키마](#7-데이터베이스-스키마)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [설치 및 실행](#9-설치-및-실행)
10. [n8n 워크플로우](#10-n8n-워크플로우)

---

## 1. 프로젝트 개요

### 1.1 목적

홈쇼핑 PD가 방송 편성표에서 빈 시간대를 발견하면, AI가 해당 시간대에 최적의 상품을 추천한다.

### 1.2 추천 근거

| 근거 | 설명 |
|------|------|
| 실시간 뉴스 트렌드 | OpenAI Web Search로 최근 7일 뉴스에서 트렌드 키워드 추출 |
| XGBoost 매출 예측 | 과거 방송 데이터 기반 매출 예측 모델 |
| 과거 방송 실적 | 유사 시간대/월에 실제로 잘 팔린 상품 |
| 경쟁사 편성 대응 | 타사 방송 편성에 대응하는 자사 상품 추천 |
| 방송테이프 필터링 | 즉시 방송 가능한 상품만 추천 (production_status = 'ready') |
| 날씨/시간대/공휴일 | 컨텍스트 기반 추천 보정 |

### 1.3 기술 스택

| 구분 | 기술 | 상세 |
|------|------|------|
| Backend | FastAPI | Python 기반 비동기 REST API 서버, Pydantic 스키마 검증, 자동 OpenAPI 문서 생성 |
| AI/ML | OpenAI Embedding | text-embedding-3-small 모델로 상품 텍스트를 1536차원 벡터로 변환, Qdrant 벡터 검색에 활용 |
| | XGBoost | 과거 방송 데이터 기반 매출 예측 모델, 시간대/카테고리/가격/요일 등 피처 활용 |
| | LangChain + GPT-4o-mini | 컨텍스트(날씨/시간대/공휴일) 기반 검색 키워드 생성, 추천 근거 텍스트 생성 |
| | OpenAI Responses API | gpt-5-nano 모델 + Web Search 도구로 최근 7일 뉴스에서 실시간 트렌드 키워드 수집 |
| Vector DB | Qdrant | 상품 임베딩 저장 및 코사인 유사도 기반 벡터 검색, 방송테이프 보유 상품 필터링 지원 |
| Database | PostgreSQL | 상품 마스터, 방송테이프, 방송실적, 경쟁사 편성, 날씨 데이터 등 비즈니스 데이터 저장 |
| 외부 데이터 | OpenWeatherMap API | 서울 기준 기온/습도/날씨상태/강수량 등 실시간 날씨 데이터 수집 (3시간 주기) |
| | BI 시스템 (Netezza) | 운영계 데이터베이스에서 상품/방송테이프/방송실적/타사편성 원천 데이터 마이그레이션 |
| 자동화 | n8n | 데이터 수집, BI 시스템 마이그레이션, 모델 재학습 등 배치 작업 스케줄링 및 자동화 |
| 인프라 | Docker Compose | 멀티 컨테이너 환경 구성, 서비스 간 네트워크 연결, 볼륨 마운트 및 환경변수 관리 |

---

## 2. 시스템 아키텍처

### 2.1 컴포넌트 구성

```
+------------------------------------------------------------------+
|                    Backend API (FastAPI)                          |
|                         Port: 8501                                |
|  +------------------------------------------------------------+  |
|  |  broadcast_workflow.py (BroadcastWorkflow 클래스)          |  |
|  |  - process_broadcast_recommendation(): 메인 워크플로우     |  |
|  |  - _collect_context_and_keywords(): 컨텍스트 수집          |  |
|  |  - _get_realtime_trend_keywords(): 뉴스 트렌드 수집        |  |
|  |  - _generate_unified_candidates(): 4-Track 후보군 생성     |  |
|  |  - _calculate_scores_and_assign_labels(): 점수 계산        |  |
|  |  - _generate_reasoning_by_code(): 추천 근거 생성           |  |
|  +------------------------------------------------------------+  |
+-------+---------------+---------------+---------------+----------+
        |               |               |               |
        v               v               v               v
+----------+    +------------+    +----------+    +----------+
|  Qdrant  |    | PostgreSQL |    |  OpenAI  |    |   n8n    |
|  :6333   |    |   :5432    |    |   API    |    |  :5678   |
+----------+    +------------+    +----------+    +----------+
                     ^
                     |
              +-------------+
              | BI 시스템   |
              | (Netezza)   |
              +-------------+
```

### 2.2 포트 정보

| 컴포넌트 | 컨테이너명 | 포트 | 역할 |
|----------|------------|------|------|
| FastAPI Backend | fastapi_backend | 8501 | AI 추천 API 서버 |
| PostgreSQL | trnAi_postgres | 5432 | 비즈니스 데이터 저장 |
| Qdrant | qdrant_vector_db | 6333 | 상품 벡터 임베딩 저장 |
| n8n | trnAi_n8n | 5678 | 자동화 워크플로우 |

---

## 3. 추천 워크플로우

### 3.1 전체 흐름

```
[API 요청] POST /api/v1/broadcast/recommendations
    |
    v
[1단계] 컨텍스트 수집 (_collect_context_and_keywords)
    - 방송 시간 파싱
    - 날씨 정보 조회
    - 공휴일 정보 조회
    - 시간대/요일/시즌 분석
    |
    v
[2단계] 트렌드 키워드 수집 (_get_realtime_trend_keywords)
    - OpenAI Responses API + Web Search
    - 최근 7일 뉴스에서 트렌드 키워드 5개 추출
    - 뉴스 제목/URL 저장
    |
    v
[3단계] 4-Track 후보군 생성 (_generate_unified_candidates)
    |
    +-- Track A: 키워드 매칭 (RAG 벡터 검색)
    +-- Track B: 매출 예측 상위 20개 (XGBoost)
    +-- Track C: 과거 유사 조건 실적 상위 20개
    +-- Track D: 경쟁사 편성 대응 12개 (RAG 검색)
    |
    v
[4단계] 병합 + 중복 제거 + 점수 계산
    - 상품코드 기준 중복 제거
    - 소분류+브랜드 조합 중복 제거 (다양성 보장)
    - Track별 가산점 적용
    - 복합 출처 가산점 적용
    |
    v
[5단계] 최종 랭킹 + 응답 생성
    - 점수순 정렬
    - 추천 근거 생성
    - API 응답 포맷팅
```

### 3.2 Track별 상세

#### Track A: 키워드 매칭 (뉴스/AI 트렌드)

- 소스: `broadcast_workflow.py` - `_execute_unified_search()`
- 방식: Qdrant 벡터 검색 (코사인 유사도)
- 입력: 뉴스 트렌드 키워드 5개
- 출력: 키워드별 유사 상품 (score_threshold: 0.3)

#### Track B: 매출 예측 상위

- 소스: `broadcast_workflow.py` - `_get_sales_top_products()`
- 방식: 방송테이프 보유 전체 상품 대상 XGBoost 매출 예측
- 출력: 예측 매출 상위 20개

#### Track C: 과거 유사 조건 실적

- 소스: `product_embedder.py` - `get_historical_top_products()`
- 조건:
  - 월 범위: 대상 월 ±1개월 (예: 12월 → 11월~1월)
  - 시간 범위: 대상 시간 ±1시간 (예: 9시 → 8시~10시)
- 출력: 평균 매출 상위 20개 (최근 방송일자/매출 포함)

#### Track D: 경쟁사 편성 대응

- 소스: `broadcast_workflow.py` - `_get_competitor_based_products()`
- 방식:
  1. 방송 시간 ±1시간 범위의 경쟁사 편성 조회
  2. 경쟁사 편성 제목에서 키워드 추출
  3. 키워드별 개별 RAG 검색 (score_threshold: 0.4)
- 출력: 경쟁사 대응 상품 12개

### 3.3 병합 규칙

1. Track 우선순위: competitor > sales_top > historical > keyword
2. 기존 상품에 새 Track 추가 시:
   - Track B/C: source_tracks에 추가
   - Track D: competitor_info만 저장 (트랙 추가 안 함, 잘못된 라벨 방지)
3. 중복 제거:
   - 상품코드 기준 중복 제거
   - 소분류+브랜드 조합 중복 제거 (다양성 보장)

---

## 4. API 명세

### 4.1 메인 API

#### POST /api/v1/broadcast/recommendations

방송 시간대에 최적의 상품 추천

**Request Body:**

```json
{
  "broadcastTime": "2025-12-08T22:00:00+09:00",
  "recommendationCount": 10,
  "trendWeight": 0.3,
  "sellingWeight": 0.7
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| broadcastTime | string | O | - | ISO 8601 형식 방송 시간 |
| recommendationCount | int | X | 10 | 추천 개수 |
| trendWeight | float | X | 0.3 | 트렌드 가중치 (0.0~1.0) |
| sellingWeight | float | X | 0.7 | 매출 가중치 (0.0~1.0) |

- trendWeight + sellingWeight = 1.0 이어야 함

**Response Body:**

```json
{
  "requestTime": "2025-12-08T20:00:00+09:00",
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "13918293",
        "productName": "잭필드 23 WINTER 남성 숨쉬는바지 3종",
        "category": "의류",
        "categoryMiddle": "일반의류",
        "categorySub": "일반의류-하의",
        "brand": "잭필드",
        "price": 79800.0,
        "tapeCode": "0000012179",
        "tapeName": "[23FW 최신상] 잭필드 겨울 숨쉬는바지 3종"
      },
      "reasoning": "[과거실적] | [과거실적] 최근 2024-01-04 22:33 방송 매출 2,005만원 (평균 2,005만원, 1회) | [예측매출] 2,005만원 | [점수] 총점 0.450 (키워드 0.100 + 매출 0.350)",
      "businessMetrics": {
        "aiPredictedSales": "2,005.1만원",
        "lastBroadcast": {
          "broadcastStartTime": "2024-01-04 22:33:28",
          "orderQuantity": 395,
          "totalProfit": 7513059.0,
          "profitEfficiency": 5.8,
          "conversionWorth": 0.0,
          "conversionRate": 78.34,
          "realFee": 0.0,
          "mixFee": 0.0
        }
      }
    }
  ],
  "competitorProducts": [
    {
      "company_name": "네이버 스토어",
      "broadcast_title": "[네이버 인기 1위] 포항 구룡포 과메기 야채세트",
      "start_time": "",
      "end_time": "",
      "duration_minutes": null,
      "category_main": ""
    }
  ]
}
```

### 4.2 기타 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 | 소스 파일 |
|-----------|--------|------|-----------|
| /api/v1/health | GET | 헬스체크 | main.py |
| /api/v1/tapes/sync | POST | BI 시스템(Netezza) → PostgreSQL 방송테이프 동기화 | main.py |
| /api/v1/migration/* | - | 데이터 마이그레이션 | api/migration.py |
| /api/v1/embeddings/* | - | 상품 임베딩 생성 | api/embeddings.py |
| /api/v1/training/* | - | XGBoost 모델 학습 | api/training.py |
| /api/v1/external-products/* | - | 외부 상품 크롤링 | routers/external_products.py |

---

## 5. 점수 계산 로직

소스: `broadcast_workflow.py` - `_calculate_scores_and_assign_labels()`

### 5.1 기본 점수

```
기본점수 = (유사도 × 0.2) + ((예측매출 / 1억) × 0.8)
```

### 5.2 Track별 가산점

| Track | 라벨 | 가산점 | 조건 |
|-------|------|--------|------|
| D | 경쟁사 | +0.12 | source_tracks에 "competitor" 포함 |
| B | 매출상위 | +0.12 | source_tracks에 "sales_top" 포함 |
| C | 과거실적 | +0.10 | source_tracks에 "historical" 포함 |
| A | 뉴스 | +0.08 | recommendation_sources에 news_trend 타입 존재 |
| A | AI분석 | 0 | RAG 매칭 유사도 >= 0.5 (라벨만 표시) |
| - | 출처없음 | -0.05 | source_labels가 비어있는 경우 |

### 5.3 복합 출처 가산점

```
복합 가산점 = 0.05 × (출처 개수 - 1)
```

| 출처 개수 | 가산점 | 예시 |
|-----------|--------|------|
| 1개 | 0 | [경쟁사] |
| 2개 | +0.05 | [경쟁사|매출상위] |
| 3개 | +0.10 | [경쟁사|매출상위|과거실적] |

### 5.4 최종 점수

```
최종점수 = 기본점수 + Track별 가산점 합계 + 복합 가산점
```

### 5.5 유사도 임계값

| 항목 | 임계값 | 설명 |
|------|--------|------|
| RAG 매칭 표시 | >= 0.5 | 유사도 50% 이상일 때만 [RAG] 정보 표시 |
| AI분석 라벨 | >= 0.5 | 유사도 50% 이상일 때만 [AI분석] 라벨 표시 |
| Track A 검색 | >= 0.3 | 키워드 매칭 기본 임계값 |
| Track D 검색 | >= 0.4 | 경쟁사 대응 RAG 검색 임계값 |

---

## 6. 추천 근거 형식

소스: `broadcast_workflow.py` - `_generate_reasoning_by_code()`

### 6.1 형식

```
[출처태그] | [상세정보1] | [상세정보2] | [예측매출] | [점수]
```

### 6.2 출처 태그

| 태그 | 설명 |
|------|------|
| [경쟁사] | 경쟁사 편성 기반 추천 |
| [매출상위] | XGBoost 매출 예측 상위 |
| [과거실적] | 유사 시간대 과거 방송 실적 |
| [뉴스] | 뉴스 트렌드 키워드 매칭 |
| [AI분석] | AI 시즌 트렌드 분석 (유사도 >= 0.5) |
| [경쟁사|매출상위] | 복합 출처 |

### 6.3 상세 정보 형식

**뉴스 트렌드:**
```
[뉴스] '키워드' 트렌드 (출처: URL...)
```

**RAG 매칭 (유사도 >= 0.5):**
```
[RAG] '키워드' 매칭 (유사도: 85%)
```

**과거 실적:**
```
[과거실적] 최근 2024-11-11 09:31 방송 매출 2,729만원 (평균 2,729만원, 1회)
```

**경쟁사 대응:**
```
[경쟁사대응] hmallplus '에이앤디 사가폭스 밍크카라 패딩코트' (08:00) 키워드:'사가폭스'
```

**예측 매출:**
```
[예측매출] 1,770만원
```

**점수:**
```
[점수] 총점 0.527 (키워드 0.150 + 매출 0.377)
```

### 6.4 전체 예시

```
[경쟁사|뉴스] | [뉴스] '밍크코트' 트렌드 (출처: https://news.example.com/...) | [경쟁사대응] hmallplus '에이앤디 사가폭스 밍크카라 패딩코트' (08:00) 키워드:'사가폭스' | [예측매출] 1,770만원 | [점수] 총점 0.527 (키워드 0.150 + 매출 0.377)
```

---

## 7. 데이터베이스 스키마

### 7.1 주요 테이블

| 테이블 | 설명 |
|--------|------|
| taigoods | 상품 마스터 |
| taipgmtape | 방송테이프 정보 |
| taibroadcasts | 방송 실적 데이터 |
| external_products | 네이버 베스트 상품 |
| competitor_schedules | 타사 편성 정보 |

### 7.2 taigoods (상품)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| product_code | VARCHAR | 상품코드 (PK) |
| product_name | VARCHAR | 상품명 |
| category_main | VARCHAR | 대분류 |
| category_middle | VARCHAR | 중분류 |
| category_sub | VARCHAR | 소분류 |
| brand | VARCHAR | 브랜드 |
| price | NUMERIC | 가격 |

### 7.3 taipgmtape (방송테이프)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| tape_code | VARCHAR | 테이프코드 (PK) |
| tape_name | VARCHAR | 테이프명 |
| product_code | VARCHAR | 상품코드 (FK) |
| production_status | VARCHAR | 제작상태 ('ready', 'in_production', 'archived') |

### 7.4 taibroadcasts (방송실적)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| broadcast_id | SERIAL | 방송ID (PK) |
| tape_code | VARCHAR | 테이프코드 (FK) |
| broadcast_start_timestamp | TIMESTAMP | 방송시작일시 |
| gross_profit | NUMERIC | 매출총이익 |
| order_quantity | INTEGER | 주문수량 |
| product_is_new | BOOLEAN | 신상품여부 |

---

## 8. 프로젝트 구조

```
trnAi/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI 엔트리포인트
│   │   ├── broadcast_workflow.py        # 핵심 추천 워크플로우
│   │   ├── broadcast_recommender.py     # 추천 유틸리티
│   │   ├── product_embedder.py          # 상품 임베딩/RAG 검색
│   │   ├── schemas.py                   # Pydantic 스키마
│   │   ├── dependencies.py              # 의존성 주입
│   │   ├── external_apis.py             # 외부 API 연동
│   │   ├── netezza_config.py            # BI 시스템(Netezza) 연결
│   │   ├── api/
│   │   │   ├── migration.py             # 마이그레이션 API
│   │   │   ├── embeddings.py            # 임베딩 API
│   │   │   └── training.py              # 학습 API
│   │   ├── routers/
│   │   │   └── external_products.py     # 외부 상품 API
│   │   └── services/
│   │       ├── external_products_service.py
│   │       └── broadcast_history_service.py
│   ├── train.py                         # XGBoost 모델 학습
│   ├── requirements.txt
│   └── .env                             # 환경변수
├── n8n_workflows/
│   ├── naver_shopping_crawler_final.json
│   ├── netezza_full_migration_with_competitor.json
│   ├── netezza_migration_workflow_http.json
│   ├── weather_collection_workflow.json
│   └── xgboost_training_workflow.json
├── docs/                                # 상세 문서
├── docker-compose.yml
├── Dockerfile
└── PROJECT_GUIDE.md                     # 이 문서
```

### 8.1 핵심 파일 설명

| 파일 | 주요 기능 |
|------|-----------|
| broadcast_workflow.py | 추천 워크플로우 전체, 4-Track 후보군 생성, 점수 계산, 근거 생성 |
| product_embedder.py | 상품 임베딩, Qdrant 검색, 과거 실적 조회 |
| schemas.py | API 요청/응답 스키마, RecommendationSource |
| main.py | FastAPI 앱, 라우터 등록 |
| train.py | XGBoost 모델 학습 스크립트 |
| netezza_config.py | BI 시스템(Netezza) 연결 설정 |

---

## 9. 설치 및 실행

### 9.1 사전 요구사항

- Docker & Docker Compose
- OpenAI API Key

### 9.2 환경변수 설정

`backend/.env` 파일 생성:

```env
POSTGRES_URI=postgresql://TRN_AI:TRN_AI@postgres:5432/TRNAI_DB
OPENAI_API_KEY=sk-your-openai-api-key
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 9.3 서비스 실행

```bash
# 네트워크 생성 (최초 1회)
docker network create shopping-network

# 서비스 시작
docker-compose up -d

# 로그 확인
docker logs -f fastapi_backend
```

### 9.4 초기화

```bash
# 상품 임베딩 생성
docker exec -it fastapi_backend python app/setup_product_embeddings.py

# XGBoost 모델 학습
docker exec -it fastapi_backend python train.py
```

### 9.5 API 테스트

```bash
curl -X POST http://localhost:8501/api/v1/broadcast/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-12-08T22:00:00+09:00",
    "recommendationCount": 10
  }'
```

---

## 10. n8n 워크플로우

n8n은 데이터 수집, 마이그레이션, 모델 학습 등의 배치 작업을 자동화하는 워크플로우 도구이다.

### 10.1 워크플로우 상세

#### naver_shopping_crawler_final.json

- **목적**: 네이버 쇼핑 베스트 상품 TOP 20을 수집하여 PostgreSQL에 저장
- **스케줄**: 매일 02:00
- **동작 흐름**:
  1. 네이버 쇼핑 베스트 페이지 크롤링
  2. 상품명, 가격, 순위, 이미지 URL 등 추출
  3. external_products 테이블에 Upsert
- **활용**: 경쟁사 대응 추천 시 네이버 인기 상품 참고

#### weather_collection_workflow.json

- **목적**: OpenWeatherMap API를 통해 날씨 데이터를 수집하여 추천 컨텍스트로 활용
- **스케줄**: 3시간마다
- **동작 흐름**:
  1. OpenWeatherMap API 호출 (서울 기준)
  2. 기온, 습도, 날씨 상태, 강수량 등 추출
  3. weather_data 테이블에 저장
- **활용**: 비 오는 날 실내용품, 더운 날 냉방용품 등 날씨 기반 추천

#### netezza_full_migration_with_competitor.json

- **목적**: BI 시스템(Netezza)에서 상품, 방송테이프, 방송실적, 타사 편성 데이터를 PostgreSQL로 마이그레이션
- **스케줄**: 수동 또는 일일 스케줄
- **동작 흐름**:
  1. Netezza DB 연결
  2. taigoods(상품), taipgmtape(방송테이프), taibroadcasts(방송실적) 데이터 조회
  3. competitor_schedules(타사 편성) 데이터 조회
  4. PostgreSQL에 Upsert
- **활용**: 추천 시스템의 기반 데이터 동기화

#### netezza_migration_workflow_http.json

- **목적**: HTTP 요청을 통해 특정 데이터만 선택적으로 마이그레이션
- **트리거**: HTTP Webhook (POST /webhook/migration)
- **동작 흐름**:
  1. HTTP 요청 수신 (table_name, date_range 파라미터)
  2. 지정된 테이블만 Netezza에서 조회
  3. PostgreSQL에 Upsert
  4. 결과 반환
- **활용**: 특정 날짜 범위나 테이블만 선택적 동기화 필요 시

#### xgboost_training_workflow.json

- **목적**: XGBoost 매출 예측 모델을 최신 데이터로 재학습
- **스케줄**: 수동 또는 주간 스케줄
- **동작 흐름**:
  1. FastAPI 학습 API 호출 (POST /api/v1/training/start)
  2. PostgreSQL에서 학습 데이터 조회
  3. XGBoost 모델 학습 실행
  4. 모델 파일 저장 (backend/models/xgboost_model.pkl)
  5. 학습 결과 로깅
- **활용**: 새로운 방송 데이터 축적 후 모델 성능 개선

### 10.2 설정 방법

1. http://localhost:5678 접속
2. Workflows > Import from File
3. n8n_workflows/*.json 파일 선택
4. Credentials 설정:
   - PostgreSQL: TRNAI_DB 연결 정보
   - Netezza: BI 시스템 연결 정보 (마이그레이션용)
   - OpenWeatherMap: API Key (날씨 수집용)
5. Active 토글 ON

---

문서 끝
