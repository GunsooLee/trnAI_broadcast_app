# 🎯 홈쇼핑 AI 매출 예측 & 방송 편성 시스템

> XGBoost 머신러닝 기반 매출 예측 + 트렌드 분석을 결합한 AI 방송 편성 최적화 시스템

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

## 📋 목차
- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [빠른 시작](#빠른-시작)
- [API 사용법](#api-사용법)
- [문서](#문서)
- [기술 스택](#기술-스택)

---

## 개요

### 🎯 시스템 목적

홈쇼핑 방송 편성 담당자와 PD를 위한 **데이터 기반 의사결정 지원 시스템**입니다.

**핵심 가치:**
- ✅ **매출 예측**: 특정 상품을 특정 시간에 방송하면 얼마나 팔릴지 AI가 예측
- ✅ **시간대 최적화**: 여러 시간대를 비교하여 최고 매출 시간대 추천
- ✅ **신상품 지원**: 과거 방송 이력이 없어도 카테고리 평균 기반 예측
- ✅ **트렌드 반영**: 실시간 트렌드 키워드 기반 상품 추천

### 📊 예측 정확도

- **R² Score**: 0.603 (60.3% 설명력)
- **평균 오차**: ±300만원 (MAE)
- **신뢰도**: 방송 횟수에 따라 60~95%

### 💡 활용 시나리오

**시나리오 1: 신상품 편성 결정**
```
Q: 새로운 건강식품을 언제 방송하면 좋을까?
A: 여러 시간대 예측 → 저녁 9시 최고 (1,736만원) → 해당 시간 편성
```

**시나리오 2: 재방송 최적화**
```
Q: 기존 상품을 다른 시간대에 방송하면?
A: 오후 2시 (1,026만원) vs 저녁 9시 (1,736만원) → 시간 변경으로 +710만원
```

**시나리오 3: 상품 선택**
```
Q: 같은 시간대에 여러 상품 중 어떤 것을 방송할까?
A: 각 상품별 예측 비교 → 최고 매출 상품 선택
```

## 주요 기능

### 1️⃣ 매출 예측 API

**단일 상품 예측** (`POST /api/v1/sales/predict-single`)
- 특정 상품의 특정 날짜/시간 매출 예측
- 신상품도 예측 가능 (카테고리 평균 기반)
- 응답 시간: 100~300ms

**날짜별 편성표 예측** (`POST /api/v1/sales/predict`)
- 특정 날짜의 모든 편성 방송 일괄 예측
- 하루 전체 예상 매출 확인

### 2️⃣ 방송 편성 추천 API

**트렌드 기반 추천** (`POST /api/v1/broadcast/recommendations`)
- 실시간 트렌드 키워드 분석
- RAG 벡터 검색으로 관련 상품 매칭
- XGBoost 매출 예측 결합
- LLM 기반 추천 근거 생성

### 3️⃣ 데이터 파이프라인

**자동 데이터 수집** (n8n 워크플로우)
- NETEZZA → PostgreSQL 일일 동기화
- 상품 마스터, 방송 테이프, 방송 이력
- 경쟁사 편성 정보 수집

**모델 자동 학습**
- 최신 데이터로 XGBoost 모델 재학습
- 예측 정확도 지속적 향상

### 4️⃣ 예측 모델 상세

**XGBoost Stacking Ensemble**
- Base Models: XGBoost, LightGBM, CatBoost
- Meta Learner: RidgeCV
- 입력 피처: 100개 이상
  - 상품 정보 (가격, 카테고리, 브랜드)
  - 시간 정보 (시간대, 요일, 계절, 공휴일)
  - 과거 실적 (상품별/카테고리별 평균)
  - 키워드 피처 (46개 고영향 키워드)
  - 날씨 정보 (기온, 강수량)
  - 카테고리-시간 상호작용

**예측 정확도 (방송 횟수별)**

| 방송 횟수 | 예측 방법 | 신뢰도 | 평균 오차 |
|----------|----------|--------|----------|
| 0회 (신상품) | 카테고리 평균 | 60~70% | ±500만원 |
| 1~5회 | 상품 평균 + 카테고리 | 70~85% | ±400만원 |
| 6~20회 | 상품 과거 실적 | 85~90% | ±300만원 |
| 20회 이상 | 상품 과거 실적 (충분) | 90~95% | ±200만원 |

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js:3001)                  │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                  Backend API (FastAPI:8501)                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ broadcast_workflow.py                                  │ │
│  │ - 컨텍스트 수집 (날씨, 시간대, 공휴일)                 │ │
│  │ - LangChain 트렌드 키워드 생성                         │ │
│  │ - Qdrant 벡터 검색                                     │ │
│  │ - XGBoost 배치 예측                                    │ │
│  │ - LangChain 추천 근거 생성                             │ │
│  └────────────────────────────────────────────────────────┘ │
└───┬─────────────┬─────────────┬─────────────┬───────────────┘
    ↓             ↓             ↓             ↓
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Qdrant │  │PostgreSQL│  │ OpenAI   │  │   n8n    │
│ :6333  │  │  :5432   │  │   API    │  │  :5678   │
└────────┘  └──────────┘  └──────────┘  └──────────┘
```

| 컴포넌트 | 포트 | 역할 |
|---------|------|------|
| FastAPI Backend | 8501 | AI 추천 API 서버 |
| Next.js Frontend | 3001 | 사용자 인터페이스 |
| PostgreSQL | 5432 | 상품/매출/방송 데이터 |
| Qdrant | 6333 | 상품 임베딩 벡터 DB |
| n8n | 5678 | 자동화 워크플로우 |

## 기술 스택

| 구분 | 기술 | 역할 |
|-----|------|------|
| AI/ML | OpenAI API | 임베딩 생성 (768차원) |
| | Qdrant | 벡터 유사도 검색 |
| | XGBoost | 매출 예측 모델 |
| | LangChain | GPT-4o-mini 기반 추천 근거 생성 |
| Backend | FastAPI + Python 3.11 | 비동기 API 서버 |
| | PostgreSQL | 데이터 저장소 |
| Frontend | Next.js + React + TypeScript | 웹 UI |
| 자동화 | n8n | 크롤링/데이터 수집 워크플로우 |
| 인프라 | Docker Compose | 컨테이너 오케스트레이션 |

## 빠른 시작

### 사전 준비
- Docker & Docker Compose
- OpenAI API Key (선택사항 - 트렌드 추천 기능용)

### 1. 환경변수 설정

`backend/.env` 파일 생성:
```env
# PostgreSQL 연결
POSTGRES_URI=postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB

# NETEZZA 연결 (데이터 수집용)
NETEZZA_HOST=your_netezza_host
NETEZZA_PORT=5480
NETEZZA_DATABASE=your_database
NETEZZA_USER=your_username
NETEZZA_PASSWORD=your_password

# OpenAI API (트렌드 추천용, 선택사항)
OPENAI_API_KEY=your_openai_api_key

# Qdrant (트렌드 추천용, 선택사항)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 2. 서비스 실행

```bash
# Docker 네트워크 생성
docker network create shopping-network

# 모든 서비스 시작
docker-compose up -d

# 서비스 상태 확인
docker-compose ps
```

### 3. 데이터 초기화

```bash
# 1. NETEZZA → PostgreSQL 데이터 마이그레이션
curl -X POST "http://localhost:8501/api/v1/migration/start-sync" \
  -H "Content-Type: application/json" \
  -d '{"full_sync": true}'

# 2. XGBoost 모델 학습
curl -X POST "http://localhost:8501/api/v1/training/start-sync" \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true}'

# 3. 상품 임베딩 생성 (트렌드 추천용, 선택사항)
curl -X POST "http://localhost:8501/api/v1/embeddings/generate-sync" \
  -H "Content-Type: application/json" \
  -d '{"force_all": true}'
```

### 4. API 테스트

**매출 예측 테스트:**
```bash
# 단일 상품 매출 예측
curl -X POST "http://localhost:8501/api/v1/sales/predict-single" \
  -H "Content-Type: application/json" \
  -d '{
    "tape_code": "0000012179",
    "broadcast_time": "2026-02-22 14:00"
  }'

# 응답 예시:
# {
#   "product_code": "13918293",
#   "product_name": "잭필드 23 WINTER 남성 숨쉬는바지 3종",
#   "predicted_sales": 22112319.68,
#   "confidence": 0.85
# }
```

**트렌드 추천 테스트:**
```bash
curl -X POST "http://localhost:8501/api/v1/broadcast/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2026-02-25T21:00:00+09:00",
    "recommendationCount": 5
  }'
```

## API 사용법

### 📊 매출 예측 API

#### 1. 단일 상품 매출 예측

**Endpoint:** `POST /api/v1/sales/predict-single`

**Request:**
```json
{
  "tape_code": "0000012179",
  "broadcast_time": "2026-02-22 14:00"
}
```

**Response:**
```json
{
  "product_code": "13918293",
  "product_name": "잭필드 23 WINTER 남성 숨쉬는바지 3종",
  "broadcast_datetime": "2026-02-22T14:00:00",
  "predicted_sales": 22112319.68,
  "confidence": 0.85,
  "features_used": {
    "tape_code": "0000012179",
    "category_main": "의류",
    "time_slot": "오후",
    "is_weekend": true,
    "season": "겨울"
  }
}
```

#### 2. 날짜별 편성표 예측

**Endpoint:** `POST /api/v1/sales/predict`

**Request:**
```json
{
  "date": "2026-02-22"
}
```

**Response:**
```json
{
  "date": "2026-02-22",
  "predictions": [
    {
      "product_code": "15750903",
      "product_name": "세일 토비콤 루테인 지아잔틴 12박스",
      "broadcast_time": "09:00",
      "predicted_sales": 28727542.0,
      "confidence": 0.85
    }
  ]
}
```

### 🎯 트렌드 추천 API

**Endpoint:** `POST /api/v1/broadcast/recommendations`

**Request:**
```json
{
  "broadcastTime": "2026-02-25T21:00:00+09:00",
  "recommendationCount": 5
}
```

**Response:** 트렌드 키워드 기반 상품 추천 + 매출 예측 + 추천 근거

### 🔧 관리 API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/v1/health` | GET | 서비스 상태 확인 |
| `/api/v1/migration/start-sync` | POST | NETEZZA → PostgreSQL 데이터 동기화 |
| `/api/v1/training/start-sync` | POST | XGBoost 모델 재학습 |
| `/api/v1/embeddings/generate-sync` | POST | 상품 임베딩 생성 |

**상세 API 문서:** [API_DOCUMENTATION.md](./docs/API_DOCUMENTATION.md)

## 프로젝트 구조

```
trnAi/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 엔트리포인트
│   │   ├── broadcast_workflow.py      # 핵심 추천 워크플로우
│   │   │   ├── _generate_unified_candidates()  # Track A + B 후보군 생성
│   │   │   ├── _get_sales_top_products()       # Track B: 매출 상위 상품
│   │   │   ├── _predict_products_sales_batch() # XGBoost 배치 예측 + 신상품 보정
│   │   │   ├── _generate_batch_reasons_with_langchain()  # LLM 추천 근거 생성
│   │   │   └── _format_sources_with_rankings() # 출처 포맷팅 (순위 포함)
│   │   ├── broadcast_recommender.py   # 추천 로직
│   │   ├── product_embedder.py        # 상품 임베딩
│   │   │   ├── get_all_products_with_tape()    # 방송테이프 보유 상품 조회
│   │   │   └── get_category_avg_sales()        # 카테고리별 평균 매출 조회
│   │   ├── external_apis.py           # 외부 API 연동
│   │   ├── netezza_config.py          # Netezza DB 연결
│   │   ├── schemas.py                 # Pydantic 스키마 (RecommendationSource 포함)
│   │   ├── api/                       # API 라우터
│   │   ├── routers/                   # 추가 라우터
│   │   └── services/                  # 비즈니스 서비스
│   ├── train.py                       # XGBoost 모델 학습
│   └── requirements.txt
├── frontend/                          # Next.js 프론트엔드
├── n8n_workflows/                     # n8n 워크플로우 JSON
├── docs/                              # 상세 문서
├── docker-compose.yml
└── README.md
```

## n8n 워크플로우

### 데이터 수집

| 워크플로우 | 스케줄 | 설명 |
|-----------|--------|------|
| `naver_shopping_crawler_final.json` | 매일 02:00 | 네이버 베스트 TOP 20 수집 |
| `weather_collection_workflow.json` | 3시간마다 | 날씨 데이터 수집 |

### 데이터 마이그레이션

| 워크플로우 | 트리거 | 설명 |
|-----------|--------|------|
| `netezza_full_migration_with_competitor.json` | 수동/스케줄 | Netezza → PostgreSQL 전체 마이그레이션 (타사 편성 포함) |
| `netezza_full_migration_workflow.json` | 수동/스케줄 | Netezza → PostgreSQL 전체 마이그레이션 |
| `netezza_migration_workflow.json` | 수동 | 기본 마이그레이션 |
| `netezza_migration_workflow_http.json` | HTTP 호출 | API 트리거 마이그레이션 |

### 모델 학습

| 워크플로우 | 트리거 | 설명 |
|-----------|--------|------|
| `xgboost_training_workflow.json` | 수동/스케줄 | XGBoost 매출 예측 모델 재학습 |

### 설정 방법

1. http://localhost:5678 접속
2. Workflows → Import from File
3. `n8n_workflows/*.json` 파일 선택
4. Credentials 설정:
   - PostgreSQL (TRNAI_DB)
   - Netezza (운영 DB, 필요시)
   - OpenWeatherMap API Key (날씨 수집용)
5. Active 토글 ON

## 문서

### 📚 사용자 가이드

- **[사용자 가이드 (비개발자용)](./docs/USER_GUIDE.md)** ⭐
  - 기획자, 방송편성 담당자를 위한 쉬운 설명
  - 실무 활용 시나리오
  - FAQ 및 용어 설명

- **[API 문서 (개발자용)](./docs/API_DOCUMENTATION.md)** ⭐
  - 전체 API 엔드포인트 상세 설명
  - 코드 예시 (Python, JavaScript, cURL)
  - 에러 처리 가이드

### 🔧 기술 문서

- `docs/DATA_PIPELINE.md` - 데이터 파이프라인 구조
- `docs/MODEL_TRAINING.md` - XGBoost 모델 학습 가이드
- `docs/ARCHITECTURE.md` - 시스템 아키텍처 상세

### 📋 참고 자료

- `docs/API_RESPONSE_EXAMPLE.json` - API 응답 예시
- `docs/FRONTEND_API_GUIDE.md` - 프론트엔드 연동 가이드
- `docs/NAVER_CRAWLER_SETUP.md` - 네이버 크롤러 설정

---

## 데이터 현황

### 📊 수집 데이터 (2026년 2월 기준)

| 테이블 | 레코드 수 | 기간 | 설명 |
|--------|----------|------|------|
| **TAIGOODS** | 5,019개 | 2022~ | 방송 테이프 보유 상품 |
| **TAIPGMTAPE** | 7,034개 | 2022~ | 방송 테이프 (52.6%는 미방송) |
| **TAIBROADCASTS** | 18,908건 | 2022~ | 방송 이력 (학습 데이터) |
| **broadcast_training_dataset** | 18,908건 | 2022~ | 피처 엔지니어링 완료 데이터 |

### 🔄 자동 업데이트

- **n8n 워크플로우**: 매일 새벽 12시 자동 실행
- **증분 업데이트**: 전날 변경분만 수집
- **모델 재학습**: 주 1회 자동 실행 (선택사항)

---