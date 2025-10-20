# 🎯 AI 기반 홈쇼핑 방송 편성 추천 시스템

**실시간 트렌드 분석**과 **XGBoost 매출 예측**을 결합한 AI 방송 편성 추천 시스템

> **최신 업데이트 (2025-10-17):** n8n 날씨 수집 워크플로우 추가, 프로젝트 구조 안정화

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [주요 기능](#-주요-기능)
3. [시스템 아키텍처](#-시스템-아키텍처)
4. [기술 스택](#-기술-스택)
5. [시작하기](#-시작하기)
6. [API 명세](#-api-명세)
7. [데이터베이스 구조](#-데이터베이스-구조)
8. [n8n 워크플로우](#-n8n-워크플로우)

---

## 💡 프로젝트 개요

### 무엇을 하는 시스템인가요?

홈쇼핑 PD가 방송 편성표에서 **빈 시간대**를 발견하면, AI가 해당 시간대에 **최적의 상품**을 추천해줍니다.

**추천 근거:**
- 🌡️ **날씨 데이터**: 비오는 날에는 실내용품, 더운 날에는 냉방용품
- 📈 **과거 매출 데이터**: XGBoost 모델로 매출 예측
- 🎬 **방송테이프 준비 상태**: 즉시 방송 가능한 상품만 추천
- 🔥 **트렌드 키워드**: 실시간 인기 키워드와 연관된 상품
- ⏰ **시간대 분석**: 저녁에는 주방용품, 심야에는 건강식품

### 사용자 시나리오

```
1. PD가 방송 편성표에서 빈 시간대 발견
   ↓
2. 'AI 추천' 버튼 클릭
   ↓
3. AI가 해당 시간대에 최적의 상품 5~10개 추천
   ↓
4. PD가 추천 근거를 확인하고 최종 선택
   ↓
5. 선택한 상품을 편성표에 추가
```

---

## ⚡ 주요 기능

### 1. 실시간 트렌드 기반 상품 추천
- **RAG (Retrieval-Augmented Generation)**: 트렌드 키워드와 상품 임베딩의 벡터 유사도 검색
- **Qdrant 벡터 DB**: 상품 정보를 1536차원 벡터로 변환하여 저장
- **OpenAI Embedding**: text-embedding-3-small 모델 사용

### 2. XGBoost 매출 예측
- **배치 예측**: 최대 30개 상품을 1번에 예측 (10~20배 성능 향상)
- **피처**: 카테고리, 시간대, 요일, 날씨, 공휴일 등
- **타겟**: 매출총이익 (gross_profit)

### 3. 동적 추천 근거 생성
- **LangChain**: GPT-4를 활용한 자연어 추천 근거 자동 생성
- **개인화**: 각 상품별로 구체적이고 설득력 있는 근거 제공
- **예시**: "저녁 시간대에 최적화된 주방용품으로, 과거 패턴 분석 결과 8,500만원의 매출이 예상됩니다."

### 4. 방송테이프 필터링
- **TAIPGMTAPE 테이블**: 방송테이프 제작 상태 관리
- **production_status='ready'**: 즉시 방송 가능한 상품만 추천
- **INNER JOIN**: 방송테이프가 없는 상품은 자동 제외

### 5. 유사도 기반 가중치 조정
```python
if similarity >= 0.7:
    # 고유사도: 트렌드 중시
    final_score = similarity * 0.7 + (predicted_sales / 1억) * 0.3
    recommendationType = "trend_match"
else:
    # 저유사도: 매출 중시
    final_score = similarity * 0.3 + (predicted_sales / 1억) * 0.7
    recommendationType = "sales_prediction"
```

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                  │
│                      http://localhost:3001                  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP Request
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)                    │
│                      http://localhost:8501                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  broadcast_workflow.py (핵심 추천 로직)              │  │
│  │  - 컨텍스트 수집 (날씨, 시간대, 공휴일)              │  │
│  │  - 트렌드 키워드 생성 (LangChain)                   │  │
│  │  - Qdrant 벡터 검색                                  │  │
│  │  - XGBoost 배치 예측                                 │  │
│  │  - 추천 근거 생성 (LangChain)                       │  │
│  └──────────────────────────────────────────────────────┘  │
└───┬─────────────┬─────────────┬─────────────┬──────────────┘
    │             │             │             │
    ↓             ↓             ↓             ↓
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Qdrant  │  │PostgreSQL│  │ OpenAI   │  │  n8n     │
│Vector  │  │  (RDB)   │  │   API    │  │(Workflow)│
│  DB    │  │          │  │          │  │          │
│:6333   │  │  :5432   │  │          │  │  :5678   │
└────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 주요 구성요소

| 구성요소 | 역할 | 포트 |
|---------|------|------|
| **FastAPI Backend** | AI 추천 API 서버 | 8501 |
| **Next.js Frontend** | 사용자 인터페이스 | 3001 |
| **PostgreSQL** | 상품/매출/방송 데이터 저장 | 5432 |
| **Qdrant** | 상품 임베딩 벡터 저장 | 6333 |
| **n8n** | 날씨 수집 자동화 워크플로우 | 5678 |
| **OpenAI API** | 임베딩 생성, LLM 추론 | - |

---

## 🛠️ 기술 스택

### Backend
- **Python 3.11+**: AI/ML 생태계 표준
- **FastAPI**: 고성능 비동기 API 서버
- **LangChain**: RAG 및 LLM 워크플로우 관리
- **XGBoost**: 매출 예측 ML 모델
- **SQLAlchemy**: ORM 및 데이터베이스 연동

### Database
- **PostgreSQL 16**: 정형 데이터 저장 (상품, 매출, 방송)
- **Qdrant**: 벡터 검색 엔진 (상품 임베딩)

### AI/ML
- **OpenAI API**: text-embedding-3-small, GPT-4
- **XGBoost**: Gradient Boosting 매출 예측 모델
- **scikit-learn**: 데이터 전처리 및 평가

### DevOps
- **Docker & Docker Compose**: 컨테이너 기반 배포
- **n8n**: 워크플로우 자동화 (날씨 수집)

---

## 🚀 시작하기

### 사전 준비
- Docker & Docker Compose
- OpenAI API Key
- (선택) OpenWeatherMap API Key

### 1. 저장소 복제
```bash
git clone https://github.com/your-repo/trnAi.git
cd trnAi
```

### 2. 환경변수 설정
`backend/.env` 파일 생성:
```env
# Database
POSTGRES_URI=postgresql://TRN_AI:TRN_AI_PASSWORD@postgres:5432/TRNAI_DB

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# OpenWeatherMap (선택)
OPENWEATHER_API_KEY=your_openweather_api_key
```

### 3. Docker 네트워크 생성
```bash
docker network create shopping-network
```

### 4. 서비스 실행
```bash
docker-compose up -d
```

**실행되는 서비스:**
- FastAPI Backend: http://localhost:8501
- Next.js Frontend: http://localhost:3001
- PostgreSQL: localhost:5432
- Qdrant: http://localhost:6333
- n8n: http://localhost:5678

### 5. 데이터베이스 초기화
```bash
# 테이블 생성 (자동 실행됨)
docker exec -it fastapi_backend python app/init_db.py
```

### 6. 상품 임베딩 생성
```bash
# 방송테이프 있는 상품만 임베딩
docker exec -it fastapi_backend python app/setup_product_embeddings.py
```

**임베딩 대상:**
- TAIGOODS INNER JOIN TAIPGMTAPE
- production_status='ready'만 포함
- 텍스트: 상품명 + 테이프명 + 카테고리

### 7. XGBoost 모델 학습
```bash
docker exec -it fastapi_backend python train.py
```

**생성되는 모델:**
- `xgb_broadcast_profit.joblib` - 매출총이익 예측 (사용 중)
- `xgb_broadcast_efficiency.joblib` - 매출효율 예측 (미사용)

### 8. API 테스트
```bash
curl -X POST http://localhost:8501/api/v1/broadcast/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-10-17T22:00:00+09:00",
    "recommendationCount": 5
  }'
```

---

## 📡 API 명세

### POST `/api/v1/broadcast/recommendations`
방송 시간대에 최적의 상품 추천

#### Request
```json
{
  "broadcastTime": "2025-10-17T22:00:00+09:00",
  "recommendationCount": 5,
  "trendRatio": 0.3  // 선택사항, 기본값 0.3
}
```

#### Response (200 OK)
```json
{
  "requestTime": "2025-10-17T14:30:00+09:00",
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "주방용품",
      "reason": "저녁 시간대 최적 카테고리",
      "predictedSales": "8.5억"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "11388995",
        "productName": "[해피콜] 다이아몬드 프라이팬 3종 세트",
        "category": "생활 > 주방용품",
        "tapeCode": "T001",
        "tapeName": "다이아몬드 프라이팬 방송테이프"
      },
      "reasoning": {
        "summary": "저녁 시간대에 최적화된 주방용품으로, 과거 패턴 분석 결과 8,500만원의 매출이 예상됩니다.",
        "linkedCategories": ["주방용품"],
        "matchedKeywords": ["요리", "저녁식사"]
      },
      "businessMetrics": {
        "pastAverageSales": "8,500만원",
        "marginRate": 0.35,
        "stockLevel": "High"
      },
      "recommendationType": "trend_match"
    }
  ]
}
```

#### 응답 필드 설명
- `recommendationType`: 추천 타입
  - `"trend_match"`: 트렌드 연관성 높음 (유사도 ≥ 0.7)
  - `"sales_prediction"`: 매출 예측 기반 (유사도 < 0.7)

---

## 🗄️ 데이터베이스 구조

### 주요 테이블

#### 1. TAIGOODS (상품 마스터)
```sql
CREATE TABLE taigoods (
    product_code VARCHAR(50) PRIMARY KEY,
    product_name TEXT,
    category_main_name VARCHAR(100),
    category_middle_name VARCHAR(100),
    category_sub_name VARCHAR(100),
    price DECIMAL(15,2),
    search_keywords TEXT
);
```

#### 2. TAIPGMTAPE (방송테이프)
```sql
CREATE TABLE taipgmtape (
    tape_code VARCHAR(50) PRIMARY KEY,
    tape_name VARCHAR(200),
    duration_minutes INTEGER,
    product_code VARCHAR(50) REFERENCES taigoods(product_code),
    production_status VARCHAR(20)  -- 'ready', 'in_production', 'archived'
);
```

#### 3. TAIBROADCASTS (방송 이력)
```sql
CREATE TABLE taibroadcasts (
    broadcast_id SERIAL PRIMARY KEY,
    product_code VARCHAR(50),
    broadcast_timestamp TIMESTAMP,
    actual_sales_amount DECIMAL(15,2),
    gross_profit DECIMAL(15,2)
);
```

#### 4. TAIWEATHER_DAILY (날씨 데이터)
```sql
CREATE TABLE taiweather_daily (
    weather_date DATE PRIMARY KEY,
    weather VARCHAR(50),
    temperature DECIMAL(5,2),
    precipitation DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 5. BROADCAST_TRAINING_DATASET (XGBoost 학습 데이터)
```sql
CREATE TABLE broadcast_training_dataset (
    id SERIAL PRIMARY KEY,
    broadcast_timestamp TIMESTAMP,
    category_name VARCHAR(100),
    day_of_week VARCHAR(20),
    is_holiday INTEGER,
    temperature DECIMAL(5,2),
    actual_sales_amount DECIMAL(15,2),
    gross_profit DECIMAL(15,2)
);
```

---

## 🔄 n8n 워크플로우

### 날씨 수집 워크플로우 (3시간마다)

**파일:** `n8n_workflows/weather_collection_workflow.json`

**워크플로우 구조:**
```
[Every 3 Hours Trigger]
    ↓
[Get Weather from OpenWeatherMap API]
    ↓
[Transform Weather Data]
    ↓
[Save to PostgreSQL (taiweather_daily)]
    ↓
[Log Success]
```

**설정 방법:**

1. **n8n 접속**: http://localhost:5678
2. **계정 생성** (최초 1회)
3. **워크플로우 Import**:
   - Workflows → Import from File
   - 파일 선택: `n8n_workflows/weather_collection_workflow.json`
4. **Credentials 설정**:
   - **OpenWeatherMap API**:
     - Type: HTTP Query Auth
     - Parameter Name: `appid`
     - Parameter Value: `YOUR_API_KEY`
   - **PostgreSQL**:
     - Host: `postgres`
     - Database: `TRNAI_DB`
     - User: `TRN_AI`
     - Password: `TRN_AI_PASSWORD`
     - Port: `5432`
5. **워크플로우 활성화**: Active 토글 ON

**수집 데이터:**
- 위치: Seoul, KR
- 온도 (°C)
- 날씨 상태 (Clear, Rain, Snow 등)
- 강수량 (mm)

---

## 🔮 향후 개발 계획

### 단기 (1~2개월)
- [ ] 실시간 트렌드 수집 (네이버 DataLab, Google Trends)
- [ ] 경쟁사 편성 데이터 수집 및 분석
- [ ] 프론트엔드 UI 완성



**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
