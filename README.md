# AI 기반 홈쇼핑 방송 편성 추천 시스템

실시간 트렌드 분석과 XGBoost 매출 예측을 결합한 AI 방송 편성 추천 시스템

## 개요

홈쇼핑 PD가 방송 편성표에서 빈 시간대를 발견하면, AI가 해당 시간대에 최적의 상품을 추천합니다.

**추천 근거:**
- 날씨 데이터 (비오는 날 실내용품, 더운 날 냉방용품)
- XGBoost 기반 매출 예측
- 방송테이프 준비 상태 (즉시 방송 가능 상품만)
- 실시간 트렌드 키워드
- 시간대별 최적화 (저녁 주방용품, 심야 건강식품)

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

## 시작하기

### 사전 준비
- Docker & Docker Compose
- OpenAI API Key

### 1. 환경변수 설정

`backend/.env` 파일 생성:
```env
POSTGRES_URI=postgresql://TRN_AI:TRN_AI@postgres:5432/TRNAI_DB
OPENAI_API_KEY=your_openai_api_key
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 2. 서비스 실행

```bash
# 네트워크 생성
docker network create shopping-network

# 서비스 시작
docker-compose up -d
```

### 3. 초기화

```bash
# 상품 임베딩 생성
docker exec -it fastapi_backend python app/setup_product_embeddings.py

# XGBoost 모델 학습
docker exec -it fastapi_backend python train.py
```

### 4. API 테스트

```bash
curl -X POST http://localhost:8501/api/v1/broadcast/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-12-08T22:00:00+09:00",
    "recommendationCount": 5
  }'
```

## API 명세

### POST `/api/v1/broadcast/recommendations`

방송 시간대에 최적의 상품 추천

**Request:**
```json
{
  "broadcastTime": "2025-12-08T22:00:00+09:00",
  "recommendationCount": 5,
  "trendWeight": 0.3,
  "sellingWeight": 0.7
}
```

**Response:**
```json
{
  "requestTime": "2025-12-08T14:30:00+09:00",
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "주방용품",
      "reason": "저녁 시간대 최적 카테고리"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "11388995",
        "productName": "[해피콜] 다이아몬드 프라이팬 3종 세트",
        "category": "생활 > 주방용품",
        "tapeCode": "T001"
      },
      "reasoning": {
        "summary": "저녁 시간대 최적화된 주방용품, 8,500만원 매출 예상"
      },
      "businessMetrics": {
        "aiPredictedSales": "8,948.1만원",
        "lastBroadcast": {
          "totalProfit": 16466355.0,
          "orderQuantity": 287
        }
      }
    }
  ],
  "externalProducts": [...]
}
```

### 기타 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/v1/health` | GET | 헬스체크 |
| `/api/v1/tapes/sync` | POST | Netezza → PostgreSQL 방송테이프 동기화 |
| `/api/v1/migration/*` | - | 데이터 마이그레이션 (n8n 연동) |
| `/api/v1/embeddings/*` | - | 상품 임베딩 생성 (n8n 연동) |
| `/api/v1/training/*` | - | XGBoost 모델 학습 (n8n 연동) |
| `/api/v1/external-products/*` | - | 외부 상품 크롤링 (n8n 연동) |

## 프로젝트 구조

```
trnAi/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 엔트리포인트
│   │   ├── broadcast_workflow.py      # 핵심 추천 워크플로우
│   │   ├── broadcast_recommender.py   # 추천 로직
│   │   ├── product_embedder.py        # 상품 임베딩
│   │   ├── external_apis.py           # 외부 API 연동
│   │   ├── netezza_config.py          # Netezza DB 연결
│   │   ├── schemas.py                 # Pydantic 스키마
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

- `docs/API_RESPONSE_EXAMPLE.json` - API 응답 예시
- `docs/API_결과_필드_설명서.md` - 응답 필드 상세 설명
- `docs/FRONTEND_API_GUIDE.md` - 프론트엔드 연동 가이드
- `docs/EXTERNAL_PRODUCTS_*.md` - 외부 상품 API 관련 문서
- `docs/NAVER_CRAWLER_SETUP.md` - 네이버 크롤러 설정

---