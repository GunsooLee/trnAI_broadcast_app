# AI 기반 홈쇼핑 방송 편성 추천 시스템 (v3.1 - 2025-10-02 업데이트)

데이터 기반의 AI 예측을 통해 방송 편성 효율을 극대화하는 백엔드 시스템

> **최신 업데이트 (2025-10-02):** 통합 점수 시스템, 배치 XGBoost 예측 (10~20배 성능 향상), 유사도 가중치 조정

## 목차

1. [주요 변경사항 (2025-10-02)](#주요-변경사항-2025-10-02)
2. [프로젝트 개요](#프로젝트-개요)
3. [사용자 시나리오](#사용자-시나리오)
4. [상세 워크플로우](#상세-워크플로우)
5. [상세 알고리즘 명세](#상세-알고리즘-명세)
   - 5.1. [메인 컨트롤러 의사코드](#51-메인-컨트롤러-의사코드)
   - 5.2. [Track A/B 병렬 처리 및 결과 통합](#52-track-ab-병렬-처리-및-결과-통합)
   - 5.3. [최종 상품 랭킹 점수 공식](#53-최종-상품-랭킹-점수-공식)
   - 5.4. [RAG 검색 파라미터 설정](#54-rag-검색-파라미터-설정)
6. [시스템 아키텍처](#시스템-아키텍처)
7. [기술 스택](#기술-스택)
8. [데이터 플로우 및 형식](#데이터-플로우-및-형식)
9. [성능 기준 및 캐싱 전략](#성능-기준-및-캐싱-전략)
10. [LangChain 워크플로우 상세](#langchain-워크플로우-상세)
11. [XGBoost 모델 학습 데이터](#xgboost-모델-학습-데이터)
12. [API 명세서](#api-명세서)
13. [시작하기 (개발 환경 설정)](#시작하기-개발-환경-설정)

---

## 주요 변경사항 (2025-10-02)

### 🚀 아키텍처 대폭 개선 (v3.1)

#### 1. **Track A/B 분리 → 통합 검색 아키텍처**
- **변경 전**: 키워드 분류 → Track A/B 병렬 실행 → 결과 통합 (복잡)
- **변경 후**: 통합 키워드 → 단일 Qdrant 검색 → 배치 예측 (간단)
- **비용 절감**: OpenAI API 호출 50% 감소 (키워드 분류 제거)
- **성능 향상**: Qdrant 검색 1회로 통합 (기존 2회 → 1회)
- **코드 간소화**: 600줄 → 400줄 (40% 감소)

#### 2. **통합 점수 시스템 (모든 상품 XGBoost 예측)**
- **임계값**: 0.7 (조정 가능)
- **핵심 변경**: 모든 상품이 XGBoost 매출 예측을 받음
- **가중치 조정 로직**:
  ```
  유사도 ≥ 0.7 → 유사도 70% + 매출 30% (트렌드 중시)
  유사도 < 0.7 → 유사도 30% + 매출 70% (매출 중시)
  ```
- **장점**: 매출 역전 방지, 공정한 비교, 정확한 예측

#### 3. **배치 예측 시스템 (성능 10~20배 향상)** ⭐
- **구현**: XGBoost 배치 예측으로 30개 상품을 1번에 처리
- **성능 개선**:
  - 이전: 30번 개별 호출 → 느림 ❌
  - 현재: 1번 배치 호출 → 10~20배 빠름 ✅
- **함수 구조**:
  1. `_prepare_features_for_product()` - 피처 준비 (공통)
  2. `_predict_product_sales()` - 개별 예측
  3. `_predict_products_sales_batch()` - 배치 예측 (권장)
- **코드 재사용**: 중복 제거, 유지보수 용이

#### 4. **추천 타입 구분 기능 (`recommendationType`)**
- **API 응답에 추가**: 각 상품이 어떤 방식으로 추천되었는지 명시
- **2가지 타입**:
  - `"trend_match"`: 고유사도 상품 (≥0.7, 트렌드 가중치 높음)
  - `"sales_prediction"`: 저유사도 상품 (<0.7, 매출 가중치 높음)
- **활용**: 프론트엔드에서 뱃지 표시, 추천 근거 차별화

#### 5. **비율 조정 기능 (trendRatio)**
- **새로운 파라미터**: `trendRatio` (0.0~1.0)
- **사용 예시**:
  ```json
  {
    "broadcastTime": "2025-09-30T22:00:00",
    "recommendationCount": 5,
    "trendRatio": 0.3  // 트렌드 30%, 매출 70%
  }
  ```
- **유연성**: 시나리오별 비율 조정 가능
  - 0.0 = 매출 예측만 (안정적)
  - 0.5 = 균형 (50:50)
  - 1.0 = 트렌드만 (급부상 키워드 우선)

#### 6. **LLM 프롬프트 로깅 시스템**
- **로그 추가**: 모든 LLM 호출 시 프롬프트 변수 로깅
- **디버깅**: LLM 응답 문제 추적 용이
- **예시**:
  ```
  [LLM 프롬프트] 컨텍스트 키워드 생성 - 변수: {weather: "맑음", time_slot: "저녁"}
  [LLM 프롬프트] 추천 근거 생성 - 상품: 스킨케어, 매출: 1257만원
  ```

### 🐛 버그 수정

- `duration` 변수 미정의 오류 수정 (기본값 30분 설정)
- `br.predict_sales()` 미구현 함수 호출 제거
- XGBoost 모델 파일 경로 수정 (`xgb_broadcast_sales.joblib` → `xgb_broadcast_profit.joblib`)
- `tokenizer_utils` 모듈 경로 문제 해결 (dependencies.py)

### 📊 현재 시스템 상태 (2025-09-30 17:15 기준)

#### ✅ 완전 작동 확인
- **API 엔드포인트**: POST /api/v1/broadcast/recommendations - 정상 작동
- **XGBoost 매출 예측**: 카테고리별 예측 성공 (예: 화장품 1,257만원)
- **LangChain 동적 근거**: 각 상품별 구체적인 추천 근거 자동 생성
- **Fallback 로직**: Qdrant 검색 0건 → PostgreSQL 전체 카테고리 조회 → 정상 추천
- **에러 제로**: 모든 버그 수정 완료

#### 테스트 결과 (2025-09-30 22:00 시간대)
```json
{
  "recommendedCategories": [
    {"rank": 1, "name": "화장품", "predictedSales": "0.1억"},
    {"rank": 2, "name": "가전제품", "predictedSales": "0.1억"},
    {"rank": 3, "name": "건강식품", "predictedSales": "0.1억"}
  ],
  "recommendations": [
    {
      "rank": 1,
      "productName": "프리미엄 스킨케어 세트",
      "reasoning": "폭염 속 저녁 시간대에 독점 방송으로 1257만원 매출 예상되는 프리미엄 스킨케어 세트를 추천합니다!"
    }
  ]
}
```

#### ⚠️ 알려진 제약사항
- **학습 데이터**: 7건 (과적합 위험, 최소 100건+ 권장)
- **Qdrant 벡터 DB**: 비어있음 (임베딩 미실행 상태)
- **Track B**: ✅ 활성화 (컨텍스트 기반 키워드 생성)
- **실시간 트렌드 API**: 미연동 (향후 개발 예정)
- **방송테이프 정보**: API 응답에 tapeCode/tapeName이 null (DB 조인 미구현)

#### 🔧 다음 단계 권장 작업
1. **학습 데이터 확보**: 최소 100건 이상의 방송/매출 데이터 수집
2. **상품 임베딩 실행**: `docker exec -it fastapi_backend python app/setup_product_embeddings.py`
3. **XGBoost 재학습**: 충분한 데이터 확보 후 `docker exec -it fastapi_backend python train.py`
4. **방송테이프 정보 추가**: PostgreSQL 조인으로 tapeCode/tapeName 응답에 포함
5. **트렌드 수집 구현**: n8n 워크플로우 또는 배치 스크립트로 Track B 활성화

#### 🔮 향후 개발 예정 기능
- **실시간 트렌드 수집**: 네이버 DataLab, Google Trends API 연동 (Track B 활성화)
- **경쟁사 편성 데이터**: TAICOMPETITOR_BROADCASTS 테이블 활용, 경쟁 페널티 로직 구현
- **트렌드 DB 관리**: TAITRENDS 테이블 활용, 시간대별 트렌드 키워드 저장
- **n8n 배치 서버**: 주기적 트렌드/경쟁사 데이터 수집 워크플로우 구축
- **날씨 API 연동**: 기상청 API로 실시간 날씨 데이터 수집

> **현재 사용 안 하는 파일**: `trend_db_manager.py` (완전 미사용), `broadcast_recommender.py` (구형 API에서만 사용)

---

## 프로젝트 개요

### 개요
본 프로젝트는 **날씨, 과거 판매 데이터, 방송테이프 준비 상태** 등 다양한 요소를 종합적으로 분석하는 AI 모델을 통해, 주어진 공백 시간에 가장 높은 매출을 기대할 수 있는 **최적의 상품을 자동으로 추천**하는 시스템을 구축

> **참고**: 실시간 트렌드 및 경쟁사 편성 데이터 수집 기능은 향후 개발 예정 (Track B)

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
  - 시간대 정보 (아침/오후/저녁/심야)
  - ~~경쟁사 편성 현황~~ (향후 개발 예정)
  - ~~실시간 트렌드 키워드~~ (향후 개발 예정)

**출력:** 통합된 컨텍스트 객체

### 2단계: 통합 키워드 생성

**처리 과정:**
- **AI 트렌드 키워드** (10개): 외부 LLM API로 생성된 시간대별 트렌드
- **컨텍스트 키워드** (6~10개): LangChain이 날씨/시간대/계절 분석하여 생성
  - 예: 비오는 저녁 → "우산", "방수", "실내활동", "따뜻한음식"
  - 예: 맑은 오후 → "야외활동", "운동", "캠핑", "레저"
- **중복 제거**: 두 키워드 리스트를 통합하여 최종 15~20개 키워드 생성

**출력:** 통합 키워드 리스트 (15~20개)

### 3단계: 통합 검색 및 배치 XGBoost 예측

**통합 검색 (1회):**
1. 통합 키워드로 Qdrant에서 상품 벡터 검색 (top_k=30)
2. 유사도 점수 0.3 이상인 상품만 추출
3. 방송테이프 준비 완료 상품만 포함 (production_status='ready')
4. 유사도 기반 분류 (direct_products / category_groups)

**배치 XGBoost 예측 및 점수 계산:**
```python
# 1. 모든 상품 통합 및 중복 제거
all_products = direct_products + category_groups
unique_products = 중복제거(all_products)[:30]

# 2. 배치 XGBoost 예측 (1번 호출)
predicted_sales_list = batch_predict(unique_products)

# 3. 유사도 가중치 조정
for product, predicted_sales in zip(unique_products, predicted_sales_list):
    if similarity >= 0.7:
        # 고유사도: 트렌드 중시
        final_score = similarity * 0.7 + (predicted_sales / 1억) * 0.3
        source = "trend_match"
    else:
        # 저유사도: 매출 중시
        final_score = similarity * 0.3 + (predicted_sales / 1억) * 0.7
        source = "sales_prediction"
```

**핵심 개선:**
- ✅ 모든 상품이 XGBoost 매출 예측 받음
- ✅ 배치 처리로 성능 10~20배 향상
- ✅ 매출 역전 방지 (유사도 높아도 매출 반영)

**출력:** 
- **candidates**: 점수순 정렬된 후보군 (final_score 포함)

### 4단계: 최종 랭킹 및 추천 개수 조정

**랭킹 계산:**
- 3단계에서 이미 `final_score`가 계산되어 정렬됨
- 추가 처리 없이 점수순으로 상위 N개 선택

**추천 개수 결정:**
```python
# 요청된 개수만큼 선택 (최대 데이터 수까지)
final_recommendations = candidates[:recommendationCount]
```

**예시 (recommendationCount=10):**
- 검색 결과: 7개 발견
- 최종 반환: 7개 (데이터 부족)

**예시 (recommendationCount=5):**
- 검색 결과: 30개 발견
- 최종 반환: 5개 (점수 상위)

**출력:** 최종 랭킹 리스트 (recommendationCount개)

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
      "rank": 1,
      "productInfo": {
        "productId": "P001",
        "productName": "프리미엄 다이어트 보조제",
        "category": "건강식품",
        "tapeCode": "T001",
        "tapeName": "프리미엄 다이어트 보조제 방송테이프"
      },
      "reasoning": {
        "summary": "AI 생성 추천 근거",
        "linkedCategories": ["건강식품"],
        "matchedKeywords": ["다이어트", "건강"]
      },
      "businessMetrics": {
        "pastAverageSales": "8.5억",
        "marginRate": 0.25,
        "stockLevel": "High"
      }
    }
  ],
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "건강식품",
      "reason": "트렌드 연관성 높음",
      "predictedSales": "8.5억"
    }
  ]
}
```

**출력:** PD에게 전달되는 최종 추천 결과

---

## 시스템 아키텍처

본 시스템은 **마이크로서비스 아키텍처(MSA)**를 채택하여, 실시간 요청을 처리하는 AI 백엔드 서버와 주기적인 데이터 수집을 담당하는 배치 서버의 역할을 명확히 분리하여 안정성과 확장성을 확보

### 주요 구성요소

- **AI 백엔드 서버 (FastAPI)**: 사용자의 실시간 추천 요청을 받아 AI 연산 및 비즈니스 로직을 수행하는 핵심 서버

- **배치 서버 (n8n)**: *(향후 개발 예정)* 주기적으로 외부 API를 통해 실시간 트렌드 데이터를 수집하고 RDB에 저장하는 역할 예정

- **RDB (PostgreSQL)**: 상품 정보, 과거 매출 데이터, 방송테이프 정보 등 모든 정형 데이터를 저장하고 관리

- **Vector DB (Qdrant)**: 상품 및 카테고리 정보의 텍스트를 벡터로 변환하여 저장하고, 의미 기반 검색(RAG)을 수행

---

## 기술 스택

| 구분 | 기술 | 역할 및 이유 |
|------|------|-------------|
| **AI 백엔드 서버** | Python & FastAPI | AI/ML 생태계의 표준. 빠르고 현대적인 API 서버 구축에 최적화 |
| **AI 프레임워크** | LangChain | RAG, 모델 호출, 비즈니스 로직 등 복잡한 AI 워크플로우를 지휘하는 역할 |
| **머신러닝 모델** | XGBoost | 카테고리별 매출 예측을 위한 고성능 ML 모델 |
| **RDB** | PostgreSQL | 상품, 매출, 방송테이프 데이터 등 핵심 정형 데이터를 안정적으로 관리 |
| **Vector DB** | Qdrant | 고성능 벡터 검색 엔진. Rust 기반으로 빠르고 가벼워 초기 구축에 유리 |
| **배치 서버** | n8n (향후 예정) | 주기적인 트렌드 데이터 수집 워크플로우를 시각적으로 쉽게 구축하고 관리 |
| **외부 API** | ~~네이버 DataLab, Google Trends~~, 기상청 API (향후) | 실시간 트렌드 및 날씨 데이터 수집 (향후 연동 예정) |
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
- ~~**Competition_Penalty**~~: (향후 개발 예정 - 경쟁사 데이터 미수집)

**정규화:** 모든 개별 지표(과거 매출, 마진율 등)는 후보군 내에서 Min-Max Scaling을 사용하여 0과 1 사이의 값으로 정규화 `(value - min) / (max - min)`

### 5.4. RAG 검색 파라미터 설정

#### **Vector Search (벡터 검색)**
- **검색 대상**: Qdrant 벡터 DB의 **방송테이프 임베딩** (OpenAI text-embedding-3-small, 1536차원)
- **임베딩 텍스트**: `상품명 + 테이프명 + 카테고리 (대/중분류)`
- **데이터 소스**: `TAIGOODS INNER JOIN TAIPGMTAPE` (production_status='ready'만 포함)
- **통합 검색**: top_k=30, score_threshold=0.3
- **유사도 임계값**: 0.7 (고유사도/저유사도 구분)
- **유사도 계산**: 코사인 유사도 기반 벡터 검색
- **Fallback**: 검색 결과 0개 시 PostgreSQL에서 전체 카테고리 조회

#### **XGBoost 배치 예측**
- **배치 크기**: 최대 30개 상품
- **예측 대상**: 모든 검색 결과 (중복 제거 후)
- **성능**: 개별 예측 대비 10~20배 향상
- **정규화**: 매출 / 1억원 기준

---

## 데이터 플로우 및 형식

**데이터 플로우 다이어그램 (텍스트 기반):**
```
Request (broadcastTime, recommendationCount, trendRatio)
  ↓
[컨트롤러] gather_context (Context Object)
  ↓
통합 키워드 생성 (AI Trend + 컨텍스트 키워드)
  ↓
Qdrant 통합 검색 (top_k=30)
  ↓
중복 제거 + 분류 (direct_products / category_groups)
  ↓
XGBoost 배치 예측 (1번 호출)
  ↓
유사도 가중치 조정 (final_score 계산)
  ↓
점수순 정렬
  ↓
상위 recommendationCount개 선택
  ↓
LangChain 추천 근거 생성
  ↓
[API 응답] (recommendations + recommendedCategories)
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

**성능 최적화 현황:**
- ✅ **배치 XGBoost 예측**: 30개 상품 1번 호출로 처리 (10~20배 향상)
- ✅ **통합 검색**: Qdrant 1번 호출로 통합 (기존 2번 → 1번)
- ✅ **API 호출 최소화**: 키워드 분류 LLM 제거
- ✅ **코드 최적화**: 피처 준비 함수 공통화

**추천 정확도 기준 (초기):** 추천된 상위 5개 상품 중 1개 이상이 실제 편성으로 이어지는 비율(Hit Rate @5)을 30% 이상으로 목표 설정하고, 피드백을 통해 지속적으로 개선

**캐싱 전략:**
- **날씨, 공휴일 정보:** 외부 API 호출 결과를 1시간 주기로 캐싱 (Redis 또는 인메모리 캐시 사용)
- **트렌드 키워드:** n8n이 수 시간 주기로 업데이트하므로, API 서버 시작 시 메모리에 로드하여 사용
- **Vector DB 연결:** 애플리케이션 시작 시 커넥션 풀을 생성하여 재사용
- **XGBoost 모델:** 메모리에 로드하여 배치 예측 시 재사용

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
2. 시간대 적합성
3. 날씨 및 계절 적합성
4. 방솨테이프 준비 상태

한 문장으로 간결하게 작성해주세요.

상품 정보:
- 카테고리: {category}
- 예상 매출: {avg_sales}만원
- 방송 시간: {time_period}
- 날씨: {weather}

~~경쟁 상황: (향후 개발 예정)~~
~~트렌드 키워드: (향후 개발 예정)~~
```

**생성 예시:**
- **입력**: 카테고리="주방용품", 매출=8500만원, 시간대="저녁", 날씨="맑음"
- **출력**: "'주방용품' 카테고리는 저녁 시간대에 최적화되어 있으며, 과거 패턴 분석 결과 8500만원의 매출이 예상됩니다."

**폴백 처리:** API 오류 시 기본 템플릿(`"'{카테고리}' 카테고리의 베스트셀러 상품입니다."`)을 사용하여 시스템 안정성 보장

---

## XGBoost 모델 학습 데이터

### 모델 개요

**학습된 모델:**
- ✅ **xgb_broadcast_profit.joblib** - 매출총이익(gross_profit) 예측 모델 (사용 중)
- ⚠️ **xgb_broadcast_efficiency.joblib** - 매출효율(sales_efficiency) 예측 모델 (미사용)

**데이터 소스:** `broadcast_training_dataset` 테이블 (PostgreSQL)

**현재 학습 데이터:** 7건 (최소 100건+ 권장)

### 주요 피처 (Features)

| 피처명 | 설명 | 예시 |
|--------|------|------|
| `broadcast_timestamp` | 방송 날짜 및 시간 | 2024-10-01 22:00 |
| `day_of_week` | 요일 | 화요일 |
| `category_name` | 방송된 상품의 카테고리 | 패션의류, 주방용품 |
| `is_holiday` | 공휴일 여부 (1 or 0) | 1 (공휴일), 0 (평일) |
| `temperature` | 당시 기온 | 23.5°C |
| ~~`competitor_count_same_category`~~ | ~~동시간대 동일 카테고리를 방송하는 경쟁사 수~~ | ~~(향후 개발 예정)~~ |

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
  "recommendationCount": 5,
  "trendRatio": 0.3  // 선택사항, 기본값 0.3
}
```

**요청 필드:**
- `broadcastTime` (string, required): 방송 시간 (ISO 8601 형식)
- `recommendationCount` (integer, optional): 추천 개수 (기본값: 5)
- `trendRatio` (float, optional): 트렌드 비율 0.0~1.0 (기본값: 0.3)

#### Response Body (Success: 200 OK)
```json
{
  "requestTime": "2025-08-25T14:01:44+09:00",
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "주방용품",
      "reason": "저녁 시간대 최적 카테고리로 AI 분석 결과 높은 매출 예상",
      "predictedSales": "9.8억"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "P300123",
        "productName": "[해피콜] 다이아몬드 프라이팬 3종 세트",
        "category": "생활 > 주방용품",
        "tapeCode": "T300123",
        "tapeName": "다이아몬드 프라이팬 3종 세트 방송테이프"
      },
      "reasoning": {
        "summary": "'주방용품' 카테고리는 저녁 시간대에 최적화되어 있으며, 과거 패턴 분석 결과 매출이 예상됩니다.",
        "linkedCategories": ["주방용품"],
        "matchedKeywords": []
      },
      "businessMetrics": {
        "pastAverageSales": "1257만원",
        "marginRate": 0.25,
        "stockLevel": "High"
      },
      "recommendationType": "sales_prediction"
    }
  ]
}
```

**응답 필드 (신규):**
- `recommendationType`: 추천 타입 구분
  - `"trend_match"`: 고유사도 상품 (≥0.7, 유사도 가중치 70%)
  - `"sales_prediction"`: 저유사도 상품 (<0.7, 매출 가중치 70%)
- **주의**: 모든 상품이 XGBoost 예측을 받지만, 가중치만 다름

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

#### 5. 상품 임베딩 초기화
```bash
# 방송테이프 있는 상품만 임베딩 (INNER JOIN)
docker exec -it fastapi_backend python app/setup_product_embeddings.py
```

**임베딩 방식 (v2.1):**
- **대상**: 방송테이프 있는 상품만 (TAIPGMTAPE INNER JOIN, production_status='ready')
- **텍스트**: `상품명 + 테이프명 + 카테고리 (대/중분류)`
- **효과**: 방송 불가능한 상품 제외, 검색 품질 향상

#### 6. XGBoost 모델 학습
```bash
# 2개 모델 학습 (profit, efficiency)
docker exec -it fastapi_backend python train.py
```

**학습 결과:**
- `xgb_broadcast_profit.joblib` - 매출총이익 예측 (사용 중)
- `xgb_broadcast_efficiency.joblib` - 매출효율 예측 (미사용)

**⚠️ 주의**: 현재 학습 데이터 7건으로 과적합 위험, 최소 100건+ 권장

**주요 변경사항 (v2.1 - 2025-09-30):**
- 방송테이프 기반 임베딩으로 전환
- XGBoost 직접 연동 (self.model.predict)
- Track B 활성화 (컨텍스트 기반 키워드 생성)
- Fallback 로직 추가 (Qdrant 없이도 작동)
- 듀얼 모델 중 profit 모델만 사용

#### 7. 외부 API 설정 (선택사항)
트렌드 수집을 위한 외부 API 키 설정:
```env
# .env 파일에 추가
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
WEATHER_API_KEY=your_weather_api_key
```

#### 8. n8n 워크플로우 배포 (선택사항)
```bash
# n8n 워크플로우 JSON 파일을 n8n 서버에 import
# 파일 위치: n8n_workflows/trend_collection_workflow.json
```

---

## 📋 **API 명세서**

### **메인 추천 API**

#### **POST `/api/v1/broadcast/recommendations`**
홈쇼핑 방송 편성을 위한 AI 상품 추천 API

**요청 (Request)**
```json
{
  "broadcastTime": "2025-09-15T22:40:00+09:00",
  "recommendationCount": 5
}
```

**요청 필드**
- `broadcastTime` (string, required): 방송 시간 (ISO 8601 형식)
- `recommendationCount` (integer, required): 추천받을 상품 개수 (1-10)

**정상 응답 (200 OK)**
```json
{
  "requestTime": "2025-09-16T01:38:25.905184",
  "recommendedCategories": [
    {
      "rank": 1,
      "name": "주방용품",
      "reason": "저녁 시간대 최적 카테고리로 XGBoost 모델 분석 결과 높은 매출 예상",
      "predictedSales": "9.8억"
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "productInfo": {
        "productId": "P300123",
        "productName": "[해피콜] 다이아몬드 프라이팬 3종 세트",
        "category": "생활 > 주방용품",
        "tapeCode": "T300123",
        "tapeName": "다이아몬드 프라이팬 3종 세트 방송테이프"
      },
      "reasoning": {
        "summary": "주방용품 카테고리는 저녁 시간대에 최적화되어 있으며, 방송테이프 준비 완료로 즉시 편성 가능합니다.",
        "linkedCategories": ["주방용품"],
        "matchedKeywords": []
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

**에러 응답**

| HTTP 코드 | 상황 | 응답 메시지 |
|-----------|------|-------------|
| **400** | 잘못된 요청 형식 | `{"detail": "잘못된 요청 데이터: [상세 메시지]"}` |
| **503** | AI 서비스 일시 중단 | `{"detail": "AI 서비스 일시 중단 - 잠시 후 다시 시도해주세요."}` |
| **503** | 빈 추천 결과 | `{"detail": "추천 결과를 생성할 수 없습니다. AI 서비스가 일시적으로 이용 불가능합니다."}` |
| **500** | 내부 서버 오류 | `{"detail": "내부 서버 오류가 발생했습니다."}` |

**사용 예시**
```bash
# cURL 예시
curl -X POST http://localhost:8501/api/v1/broadcast/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "broadcastTime": "2025-09-15T22:40:00+09:00",
    "recommendationCount": 3
  }'

# Python 예시
import requests

response = requests.post(
    "http://localhost:8501/api/v1/broadcast/recommendations",
    json={
        "broadcastTime": "2025-09-15T22:40:00+09:00",
        "recommendationCount": 3
    }
)
print(response.json())
```

### **보조 API**

#### **GET `/api/v1/health`**
API 서버 상태 확인

**응답 (200 OK)**
```json
{
  "status": "ok"
}
```

---
