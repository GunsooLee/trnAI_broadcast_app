# AI 기반 홈쇼핑 방송 편성 추천 시스템 (v2.0 - 상세 구현 명세)

데이터 기반의 AI 예측을 통해 방송 편성 효율을 극대화하는 백엔드 시스템

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [사용자 시나리오](#사용자-시나리오)
3. [상세 워크플로우](#상세-워크플로우)
4. [상세 알고리즘 명세](#상세-알고리즘-명세)
   - 4.1. [메인 컨트롤러 의사코드](#41-메인-컨트롤러-의사코드)
   - 4.2. [Track A/B 병렬 처리 및 결과 통합](#42-track-ab-병렬-처리-및-결과-통합)
   - 4.3. [최종 상품 랭킹 점수 공식](#43-최종-상품-랭킹-점수-공식)
   - 4.4. [RAG 검색 파라미터 설정](#44-rag-검색-파라미터-설정)
5. [시스템 아키텍처](#시스템-아키텍처)
6. [기술 스택](#기술-스택)
7. [데이터 플로우 및 형식](#데이터-플로우-및-형식)
8. [성능 기준 및 캐싱 전략](#성능-기준-및-캐싱-전략)
9. [LangChain 워크플로우 상세](#langchain-워크플로우-상세)
10. [XGBoost 모델 학습 데이터](#xgboost-모델-학습-데이터)
11. [API 명세서](#api-명세서)
12. [시작하기 (개발 환경 설정)](#시작하기-개발-환경-설정)

---

## 프로젝트 개요

### 개요
본 프로젝트는 **최신 트렌드, 날씨, 경쟁사 편성 현황, 과거 판매 데이터** 등 다양한 요소를 종합적으로 분석하는 AI 모델을 통해, 주어진 공백 시간에 가장 높은 매출을 기대할 수 있는 **최적의 상품을 자동으로 추천**하는 시스템을 구축

### 실제 적용 화면(예상)

![홈쇼핑 방송 편성 시스템 화면](./image_2025-09-05_10-07-28.png)

위 화면은 실제 홈쇼핑 방송 편성 시스템에서 **AI 편성 추천** 버튼을 통해 공백 시간대에 최적의 상품을 추천받는 과정을 보여줍니다. PD가 특정 시간대의 빈 슬롯에서 AI 추천 기능을 활용하여 데이터 기반의 상품 편성 결정을 내림.



### 09.03 회의 추가내용

팝업 화면은 '내부상품', '외부상품' 탭으로 구분하여 '외부상품'도 추천

'내부상품'에 대한 추천 기준은 매출액 뿐만아니라 '취급효율', '매총효율', '전환율', 환산가치시간'등을 확인

---

## 사용자 시나리오

1. **공백 시간 확인**: PD가 방송 편성 웹페이지에서 비어있는 편성 슬롯을 확인

2. **AI 추천 요청**: 해당 슬롯의 'AI 추천' 버튼을 클릭

3. **추천 결과 확인**: 잠시 후, 추천 상품 리스트가 담긴 팝업창이 나타남. 리스트는 AI가 매긴 추천 점수 순으로 정렬되어 있음

4. **AI 생성 추천 근거 확인**: 각 상품별로 AI가 자동 생성한 상세하고 전문적인 추천 근거(매출 전망, 경쟁 분석, 트렌드 연관성, 시간대 적합성)를 확인

5. **최종 편성 확정**: PD는 제시된 후보군 중에서 가장 적합하다고 판단하는 상품을 선택하여 편성을 최종 확정

---

## 상세 워크플로우

사용자 요청부터 응답까지 AI 백엔드 서버 내부에서 일어나는 상세한 동작 과정을 5단계로 나누어 설명

### 1단계: 요청 접수 및 컨텍스트 수집

**입력:** PD가 특정 시간대(예: 2024-10-01 22:00)의 'AI 추천' 버튼 클릭

**처리 과정:**
- API 요청으로 `broadcastTime` 파라미터를 받음
- 해당 시간을 기준으로 다양한 컨텍스트 데이터를 수집:
  - 날씨 정보 (기온, 날씨 상태)
  - 공휴일/특별일 정보
  - 경쟁사 편성 현황 (동시간대 방송 상품)
  - 실시간 트렌드 키워드 (네이버, 구글 등)

**출력:** 통합된 컨텍스트 객체

### 2단계: 키워드 분류 및 방향성 결정

**처리 과정:**
- 수집된 모든 트렌드 키워드를 LangChain(GPT-4)에 전달
- AI가 키워드를 두 그룹으로 자동 분류:
  - **카테고리 연관 키워드**: "겨울", "추위", "건강" 등 일반적인 상황/라이프스타일
  - **상품 특정 키워드**: "아이폰", "나이키" 등 구체적인 브랜드/상품명

**출력:** 분류된 키워드 그룹

### 3단계: 병렬 검색 (Track A/B 동시 실행)

두 개의 독립적인 검색 트랙을 `asyncio.gather`로 병렬 실행하여 성능 최적화

**Track A - 카테고리 기반 검색 (2단계 처리):**

*3-1단계: RAG 검색*
1. 카테고리 연관 키워드로 Qdrant에서 유사한 카테고리들을 RAG 검색
2. 벡터 유사도 기반으로 관련성 높은 카테고리 후보군 추출 (예: 5-10개)

*3-2단계: XGBoost 매출 예측*
1. RAG로 찾은 카테고리 후보군에 대해 XGBoost 모델로 매출 예측 수행
2. 각 카테고리별 예상 매출액 계산 (경쟁사 편성, 날씨, 시간대 등 고려)
3. 유사도 점수 + 매출 예측값을 결합하여 **'유망 카테고리 TOP 3'** 최종 선정

**Track B - 상품 직접 검색 (매출 예측 없음):**
1. 상품 특정 키워드로 Qdrant에서 관련 상품들을 RAG 검색
2. 벡터 유사도가 높은 **'특정 상품 리스트'** 추출
3. ⚠️ Track B에서는 XGBoost 매출 예측을 수행하지 않음 (RAG 검색만)

**출력:** Track A 결과 (매출 예측된 카테고리 리스트) + Track B 결과 (상품 리스트)

### 4단계: 후보군 통합 및 랭킹

**후보군 생성:**
- Track B의 상품들을 최종 후보군에 먼저 추가
- Track A의 각 유망 카테고리별로 PostgreSQL에서 베스트셀러 상품들을 SQL 조회하여 추가
- 중복 상품 제거 (product_id 기준)

**최종 점수 계산:**
```
상품 최종 점수 = (카테고리 점수 × 0.6) + 개별 상품 점수 - (경쟁 페널티 × 0.3)
```

- **카테고리 점수**: Track A에서 해당 카테고리가 받은 점수
- **개별 상품 점수**: 과거 매출, 마진율, 재고 수준을 정규화한 값
- **경쟁 페널티**: 동시간대 경쟁사가 같은 카테고리 방송 시 감점

**출력:** 점수순으로 정렬된 상품 리스트

### 5단계: 추천 근거 생성 및 응답

**추천 근거 자동 생성:**
- 각 상품별로 LangChain을 활용하여 구체적인 추천 이유를 생성
- 포함 요소: 매출 전망, 경쟁 상황 분석, 트렌드 연관성, 시간대 적합성
- 예시: "주방용품 카테고리의 예상 매출이 높고, 동시간대 경쟁이 없어 독점 방송이 가능합니다."

**최종 API 응답:**
```json
{
  "recommendations": [
    {
      "productId": "P001",
      "productName": "프리미엄 다이어트 보조제",
      "finalScore": 0.87,
      "reasoning": "AI 생성 추천 근거",
      "category": "건강식품"
    }
  ],
  "recommendedCategories": ["건강식품", "주방용품", "패션의류"]
}
```

**출력:** PD에게 전달되는 최종 추천 결과

---

## 시스템 아키텍처

본 시스템은 **마이크로서비스 아키텍처(MSA)**를 채택하여, 실시간 요청을 처리하는 AI 백엔드 서버와 주기적인 데이터 수집을 담당하는 배치 서버의 역할을 명확히 분리하여 안정성과 확장성을 확보

### 주요 구성요소

- **AI 백엔드 서버 (FastAPI)**: 사용자의 실시간 추천 요청을 받아 AI 연산 및 비즈니스 로직을 수행하는 핵심 서버

- **배치 서버 (n8n)**: 주기적으로 외부 웹사이트 및 API를 통해 최신 트렌드와 경쟁사 편성 데이터를 수집하고 RDB에 저장하는 역할을 담당

- **RDB (PostgreSQL)**: 상품 정보, 과거 매출 데이터, 경쟁사 편성 데이터 등 모든 정형 데이터를 저장하고 관리

- **Vector DB (Qdrant)**: 상품 및 카테고리 정보의 텍스트를 벡터로 변환하여 저장하고, 의미 기반 검색(RAG)을 수행

---

## 기술 스택

| 구분 | 기술 | 역할 및 이유 |
|------|------|-------------|
| **AI 백엔드 서버** | Python & FastAPI | AI/ML 생태계의 표준. 빠르고 현대적인 API 서버 구축에 최적화 |
| **AI 프레임워크** | LangChain | RAG, 모델 호출, 비즈니스 로직 등 복잡한 AI 워크플로우를 지휘하는 역할 |
| **머신러닝 모델** | XGBoost | 카테고리별 매출 예측을 위한 고성능 ML 모델 |
| **RDB** | PostgreSQL | 상품, 매출, 경쟁사 데이터 등 핵심 정형 데이터를 안정적으로 관리 |
| **Vector DB** | Qdrant | 고성능 벡터 검색 엔진. Rust 기반으로 빠르고 가벼워 초기 구축에 유리 |
| **배치 서버** | n8n | 주기적인 데이터 수집 워크플로우를 시각적으로 쉽게 구축하고 관리 |
| **DevOps** | Docker, GitHub Actions | 개발 환경 통일 및 CI/CD 자동화 (권장) |

---

## 상세 알고리즘 명세

본 섹션은 시스템의 핵심 로직을 구체적인 의사코드와 공식, 파라미터 값으로 명세

### 4.1. 메인 컨트롤러 의사코드

FastAPI의 API 엔드포인트에서 호출될 메인 함수의 논리적 흐름

```python
# main.py - /api/v1/broadcast/recommendations

async def get_recommendations(request: Request):
    # 1. 컨텍스트 수집 및 키워드 분류
    context = await gather_context(request.broadcastTime)
    classified_keywords = await classify_keywords(context.trends)

    # 2. Track A, B 비동기 병렬 실행
    # asyncio.gather를 사용하여 두 트랙을 동시에 실행
    track_a_result, track_b_result = await asyncio.gather(
        execute_track_a(context, classified_keywords.category_keywords),
        execute_track_b(context, classified_keywords.product_keywords)
    )

    # 3. 후보군 생성 및 통합
    candidate_products = await generate_candidates(
        promising_categories=track_a_result.categories,
        trend_products=track_b_result.products
    )

    # 4. 최종 랭킹 계산
    ranked_products = await rank_final_candidates(
        candidate_products,
        category_scores=track_a_result.scores,
        context=context
    )

    # 5. API 응답 생성
    return format_response(ranked_products, track_a_result.categories)
```

### 4.2. Track A/B 병렬 처리 및 결과 통합

**병렬 처리 방법:** Python의 `asyncio.gather`를 사용하여 두 개의 비동기 함수(`execute_track_a`, `execute_track_b`)를 동시에 실행. 이를 통해 I/O 바운드 작업(DB 조회, API 호출) 대기 시간을 최소화

**결과 통합 로직:**
1. `execute_track_b`에서 반환된 '상품 특정' 상품 리스트를 최종 후보군에 먼저 추가
2. `execute_track_a`에서 반환된 '유망 카테고리' 리스트를 순회하며, 각 카테고리별로 RDB에서 '에이스 상품'(판매량 상위 100개 등)을 SQL로 조회
3. 두 리스트를 합친 후, `product_id`를 기준으로 중복을 제거하여 최종 후보군(Candidate Pool)을 생성

### 4.3. 최종 상품 랭킹 점수 공식

**가중치 (W1, W2):** 초기값은 비즈니스 요구사항에 따라 설정하며, 향후 A/B 테스트를 통해 최적화
- **W1 (카테고리 적합도 가중치):** 0.6
- **W2 (경쟁 상황 가중치):** 0.3

**최종 점수 공식:**
```
Final_Score = (Category_Score * W1) + Individual_Score - (Competition_Penalty * W2)
```

**각 점수 계산 방법:**
- **Category_Score (0~1):** 상품이 속한 카테고리가 Track A에서 받은 최종 점수. (XGBoost 예측값과 RAG 관련성 점수를 합산 후 Min-Max 정규화)
- **Individual_Score (0~1):** 상품 개별 지표를 합산한 점수.
  ```
  (Normalized_Past_Sales * 0.5) + (Normalized_Margin_Rate * 0.3) + (Normalized_Stock_Level * 0.2)
  ```
- **Competition_Penalty (0 또는 1):** 동시간대 경쟁사가 동일 카테고리(중분류 기준) 상품을 방송할 경우 1, 아닐 경우 0.

**정규화:** 모든 개별 지표(과거 매출, 마진율 등)는 후보군 내에서 Min-Max Scaling을 사용하여 0과 1 사이의 값으로 정규화 `(value - min) / (max - min)`

### 4.4. RAG 검색 파라미터 설정

Qdrant(Vector DB) 검색 시 사용할 구체적인 파라미터

**검색 대상:** 카테고리 인덱스, 상품 인덱스

**top_k (반환할 결과 수):**
- 카테고리 검색 시: `k = 5`
- 상품 검색 시: `k = 20`

**score_threshold (최소 유사도 임계값):**
- `threshold = 0.7` (코사인 유사도 기준, 이 값보다 낮으면 결과에서 제외)

**필터링 조건 (Metadata Filtering):**
- 모든 상품 검색 시 `is_active = true`, `stock_level > 0` 과 같은 필터를 기본으로 적용하여 판매 불가능한 상품을 사전에 제외

---

## 데이터 플로우 및 형식

**데이터 플로우 다이어그램 (텍스트 기반):**
```
Request (broadcastTime) -> [컨트롤러] -> gather_context -> Context Object 
-> [Track A | Track B] -> [Category_Result | Product_Result] 
-> generate_candidates -> List[Product] -> rank_candidates 
-> List[Ranked_Product] -> [API 응답]
```

**주요 데이터 형식 (Pydantic 모델 예시):**
```python
class Context:
    broadcast_time: datetime
    weather: str
    is_holiday: bool
    trends: List[str]
    competitors: List[CompetitorInfo]

class RankedProduct:
    product_id: str
    product_name: str
    final_score: float
    reasoning: Dict[str, Any]
```

**에러 핸들링:**
- Vector DB/RDB 연결 실패: `503 Service Unavailable` 응답
- XGBoost 모델 로드 실패: `500 Internal Server Error` 응답
- 유효하지 않은 broadcastTime: `400 Bad Request` 응답

---

## 성능 기준 및 캐싱 전략

**API 응답 시간 목표:** 최종 사용자(PD)의 경험을 위해, API 요청부터 응답까지 평균 2초, 최대 3초를 목표로 설정

**추천 정확도 기준 (초기):** 추천된 상위 5개 상품 중 1개 이상이 실제 편성으로 이어지는 비율(Hit Rate @5)을 30% 이상으로 목표 설정하고, 피드백을 통해 지속적으로 개선

**캐싱 전략:**
- **날씨, 공휴일 정보:** 외부 API 호출 결과를 1시간 주기로 캐싱 (Redis 또는 인메모리 캐시 사용)
- **트렌드 키워드:** n8n이 수 시간 주기로 업데이트하므로, API 서버 시작 시 메모리에 로드하여 사용
- **Vector DB 연결:** 애플리케이션 시작 시 커넥션 풀을 생성하여 재사용

---

## LangChain 워크플로우 상세

**프롬프트 템플릿 (키워드 분류기):**
```
You are a helpful assistant. Classify the given keywords into 'category_keywords' which are general situations or lifestyles, and 'product_keywords' which are specific brand or product names.

Keywords: [{keywords}]

Respond ONLY in JSON format like this:
{
  "category_keywords": ["list of keywords"],
  "product_keywords": ["list of keywords"]
}
```

**Chain 구성 (LCEL - LangChain Expression Language):**
```python
# 의사코드 예시
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4-turbo")
parser = JsonOutputParser()

classification_chain = prompt | model | parser
```

**에러 처리:** LLM 응답이 JSON 형식이 아니거나, API 호출에 실패할 경우 재시도(Retry) 로직을 2회 수행하고, 최종 실패 시 모든 키워드를 '카테고리 키워드'로 간주하여 시스템 중단을 방지

### AI 기반 추천 근거 생성

선정된 상품에 대해 **왜 이 상품을 추천했는지** 구체적이고 설득력 있는 근거를 자동 생성

**프롬프트 템플릿 (추천 근거 생성기):**
```
당신은 홈쇼핑 방송 편성 전문가입니다. 
주어진 정보를 바탕으로 상품 추천 근거를 구체적이고 설득력 있게 작성해주세요.

다음 요소들을 포함해서 작성하세요:
1. 카테고리의 매출 전망
2. 경쟁 상황 분석 (독점 방송 가능성 등)
3. 트렌드 키워드와의 연관성
4. 시간대 적합성

한 문장으로 간결하게 작성해주세요.

상품 정보:
- 카테고리: {category}
- 예상 매출: {avg_sales}만원
- 방송 시간: {time_period}

경쟁 상황:
- 동시간대 경쟁사 카테고리: {competitor_categories}
- 경쟁 여부: {has_competition}

트렌드 키워드: {trend_keywords}
```

**생성 예시:**
- **입력**: 카테고리="주방용품", 매출=8500만원, 경쟁="없음", 트렌드="주말저녁,요리"
- **출력**: "'주방용품' 카테고리의 예상 매출이 높고, 동시간대 경쟁이 없어 독점 방송이 가능합니다. 또한, '주말저녁'과 '요리' 트렌드와 관련성이 높아 매출 증가 가능성이 있습니다."

**폴백 처리:** API 오류 시 기본 템플릿(`"'{카테고리}' 카테고리의 베스트셀러 상품입니다."`)을 사용하여 시스템 안정성 보장

---

## XGBoost 모델 학습 데이터

{{ ... }}

### 주요 피처 (Features)

| 피처명 | 설명 | 예시 |
|--------|------|------|
| `broadcast_timestamp` | 방송 날짜 및 시간 | 2024-10-01 22:00 |
| `day_of_week` | 요일 | 화요일 |
| `category_name` | 방송된 상품의 카테고리 | 패션의류, 주방용품 |
| `is_holiday` | 공휴일 여부 (1 or 0) | 1 (공휴일), 0 (평일) |
| `temperature` | 당시 기온 | 23.5°C |
| `competitor_count_same_category` | 동시간대 동일 카테고리를 방송하는 경쟁사 수 | 2개사 |

### 타겟 변수 (Label)

- **`actual_sales_amount`**: 해당 방송의 실제 매출액 (예측 목표)

---

## 📋 API 명세서

### 방송 추천 API

**Endpoint:** `POST /api/v1/broadcast/recommendations`

#### Request Body
```json
{
  "broadcastTime": "2025-09-15T22:40:00+09:00",
  "recommendationCount": 5
}
```

#### Response Body (Success: 200 OK)
```json
{
  "requestTime": "2025-08-25T14:01:44+09:00",
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "주방용품",
      "reason": "경쟁사 부재 및 '주말 저녁' 키워드와 관련성 높음",
      "predictedSales": "9.8억"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "P300123",
        "productName": "[해피콜] 다이아몬드 프라이팬 3종 세트",
        "category": "생활 > 주방용품"
      },
      "reasoning": {
        "summary": "'주방용품' 카테고리의 예상 매출이 높고, 동시간대 경쟁이 없어 독점 방송이 가능합니다.",
        "linkedCategories": ["주방용품"],
        "matchedKeywords": ["주말 저녁", "요리"]
      },
      "businessMetrics": {
        "pastAverageSales": "8.5억",
        "marginRate": 0.35,
        "stockLevel": "Good"
      }
    }
  ]
}
```

---

## 🚀 시작하기 (개발 환경 설정)

### 사전 준비
- Python 3.11+
- PostgreSQL 14+
- Docker & Docker Compose
- OpenAI API Key

### 설치 및 실행

#### 1. 저장소 복제
```bash
git clone https://github.com/your-repo/trnAi.git
cd trnAi
```

#### 2. Docker 환경 설정
```bash
# Docker 네트워크 생성
docker network create shopping-network

# 서비스 실행
docker-compose up -d
```

#### 3. 환경변수 설정
`backend/.env` 파일을 생성하고 다음 정보를 입력합니다:
```env
DB_URI=postgresql://TRN_AI:TRN_AI@localhost:5432/TRNAI_DB
OPENAI_API_KEY=your_openai_api_key_here
```

#### 4. 데이터베이스 초기화
```bash
# PostgreSQL 컨테이너에서 초기 스키마 실행
docker exec -i trnAi_postgres psql -U TRN_AI -d TRNAI_DB < init_database.sql
```

#### 5. XGBoost 모델 학습
```bash
cd backend
python train.py
```

---
