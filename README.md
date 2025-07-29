# 🛍️ 홈쇼핑 방송 매출 예측 & 편성 추천

**링크 하나만 열면 바로 체험할 수 있어요!**

[➡️ 데모 바로가기](http://175.106.97.27:8501/) _(PC·모바일 모두 지원)_

---

## ✨ 무엇을 할 수 있나요?
1. **질문만 입력**하면, AI가 날짜·시간대·상품 키워드를 이해해
2. 과거 매출 데이터를 학습한 모델이 **시간대별 예상 매출**을 계산하고
3. 가장 잘 팔릴 것으로 예측되는 **상품(또는 카테고리)** 편성을 추천해 줍니다.

예)  
“다음 주 토요일 오전에 다이어트 보조제 방송 추천해 줘” →  📋 추천 편성표 + 예상 매출 표시

---

## ⚡️ FastAPI & Next.js 기반 실행/운영 가이드 (2025년 최신)

### 1. 전체 아키텍처
- **Backend:** Python FastAPI (API 서버)
- **Frontend:** Next.js (React 기반 SPA, 포트 3001)
- **DB:** PostgreSQL
- **(구 Streamlit → 완전 대체됨!)**

### 2. 개발/로컬 실행 방법

#### 2-1. 백엔드(FastAPI) 실행
```bash
cd backend
# 가상환경 활성화 및 의존성 설치
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# FastAPI 서버 실행 (포트 8501)
uvicorn app.main:app --host 0.0.0.0 --port 8501 --reload
```
- 환경변수: `.env` 파일에 `DB_URI`, `OPENAI_API_KEY` 등 필요
- API 문서: [http://localhost:8501/docs](http://localhost:8501/docs)

#### 2-2. 프론트엔드(Next.js) 실행
```bash
cd frontend
npm install
npm run dev   # http://localhost:3001
```
- 환경변수 필요시 `.env.local` 사용 (ex: API base url)

#### 2-3. 전체 연동
- 프론트엔드가 백엔드의 8501 포트로 API 요청
- CORS/Proxy 설정은 이미 적용됨

### 3. 운영 서버 배포/실행 방법

#### 3-1. 백엔드(FastAPI) Docker 빌드/실행
- **(중요) 기존 Dockerfile/compose는 Streamlit 기준 → FastAPI용으로 수정 필요!**
- 아래는 FastAPI 기준 예시:

**Dockerfile (backend/ 디렉토리 기준 예시)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8501"]
```

**docker-compose.yml (예시)**
```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    container_name: fastapi_backend
    ports:
      - "8501:8501"
    env_file: ./backend/.env
    restart: unless-stopped
  frontend:
    build: ./frontend
    container_name: nextjs_frontend
    ports:
      - "3001:3001"
    restart: unless-stopped
networks:
  default:
    external: false
```

- **운영 배포 절차**
  1. 서버에 소스 업로드 (혹은 git pull)
  2. `.env`, `frontend/.env.local` 등 환경파일 세팅
  3. `docker compose up -d --build`로 전체 서비스 기동
  4. (DB/Postgres는 별도 운영 필요)

#### 3-2. 프론트엔드(Next.js) Docker 빌드/실행
- `frontend/Dockerfile` 예시
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3001
CMD ["npm", "start"]
```
- Next.js는 `npm run build` 후 `npm start`로 운영

### 4. 기타 참고
- **운영 서버 오픈 포트:** 8501(FastAPI), 3001(Next.js)
- **DB 연결:** 운영 DB URI를 `.env`에 반드시 명시
- **모델 파일:** 학습 후 `backend/app/xgb_broadcast_sales.joblib` 위치에 존재해야 함
- **모든 서비스는 Docker로 통합 배포 가능**

---

## 🧑‍💻 개발자용 가이드
아래 내용은 직접 학습·배포하고 싶은 분들을 위한 상세 설명입니다. 사용만 해보려면 건너뛰어도 괜찮아요.

<details>
<summary>클릭해서 펼치기</summary>

### 환경 구성
```bash
# Python 3.11 권장 (mecab-python3 wheel 지원)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Mecab 사전은 `mecab-python3` wheel 에 포함되어 추가 설정이 필요 없습니다.

### 학습
```bash
python broadcast_recommender.py train \
    --db-uri postgresql://USER:PASS@HOST:PORT/DB  # (옵션) 환경변수/파일 설정 가능
```
출력 예시
```
=== 모델 평가 ===
MAE : 7.1M
RMSE: 11.9M
R2  : 0.83
```

### 로컬 추천 예시
```python
import datetime as dt
import broadcast_recommender as br

date = dt.date.today() + dt.timedelta(days=1)
result = br.recommend(
    target_date=date,
    time_slots=["아침", "오전"],
    product_codes=["A00123"],
    weather_info={"weather": "맑음", "temperature": 25, "precipitation": 0},
)
print(result)
```

### 주요 파일 구조
```
├── broadcast_recommender.py  # 학습 + 추천 백엔드
├── tokenizer_utils.py        # Mecab 토크나이저 모듈 (joblib 호환)
├── streamlit_app.py          # 챗봇 UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### 기여 / TODO
- 하이퍼파라미터 튜닝 & 모델 앙상블
- 모델 모니터링 지표 대시보드
- API 서버(FastAPI) 분리 배포

</details>

## 🧑‍💻 개발자용 상세 가이드

### 1. 모델 학습(Training) 파이프라인

#### 데이터 소스 및 테이블
- **주 테이블:** `broadcast_training_dataset`
- **날씨 테이블:** `weather_daily` (조인)
- **사용 컬럼:**
  - 방송 정보: `broadcast_id`, `broadcast_datetime`, `broadcast_duration`
  - 상품 정보: `product_code`, `product_lgroup`, `product_mgroup`, `product_sgroup`, `product_dgroup`, `product_type`, `product_name`, `keyword`, `product_price`
  - 매출 정보: `sales_amount`, `order_count`
  - 시간대 정보: `time_slot`
  - 외부 정보: `temperature`, `precipitation`, `weather` (날씨)

#### 주요 Feature Engineering
- **상품별 통계:**
  - `product_avg_sales`: 상품별 전체 기간 평균 매출
  - `product_broadcast_count`: 상품별 방송 횟수
- **카테고리-시간대별 통계:**
  - `category_timeslot_avg_sales`: (중분류, 시간대)별 평균 매출
  - `category_overall_avg_sales`: 중분류 전체 기간 평균 매출
  - `timeslot_specialty_score`: 시간대별 특화 점수 (category_timeslot_avg_sales / category_overall_avg_sales)
- **파생 변수:**
  - `weekday`: 방송 요일(월~일)
  - `season`: 방송 월로부터 계절 추출(봄/여름/가을/겨울)
  - `time_slot_int`: 시간대를 숫자로 변환
  - `time_category_interaction`: 시간대와 카테고리의 조합
- **결측치 처리:** 평균/0/‘정보없음’ 등으로 채움

#### SQL 예시 (학습 데이터 생성)
```sql
WITH base AS (
    SELECT ... FROM broadcast_training_dataset WHERE sales_amount IS NOT NULL
),
product_stats AS (
    SELECT product_code, AVG(sales_amount) AS product_avg_sales, COUNT(*) AS product_broadcast_count
    FROM broadcast_training_dataset GROUP BY product_code
),
category_timeslot_stats AS (
    SELECT product_mgroup, time_slot, AVG(sales_amount) AS category_timeslot_avg_sales
    FROM broadcast_training_dataset GROUP BY product_mgroup, time_slot
),
category_overall_stats AS (
    SELECT product_mgroup, AVG(sales_amount) AS category_overall_avg_sales
    FROM broadcast_training_dataset GROUP BY product_mgroup
)
SELECT
    b.*, w.temperature, w.precipitation, w.weather,
    p.product_avg_sales, p.product_broadcast_count,
    c.category_timeslot_avg_sales,
    COALESCE(c.category_timeslot_avg_sales / NULLIF(co.category_overall_avg_sales, 0), 1) AS timeslot_specialty_score,
    b.time_slot || '_' || b.product_mgroup AS time_category_interaction
FROM base b
LEFT JOIN weather_daily w ON b.broadcast_date = w.weather_date
LEFT JOIN product_stats p ON b.product_code = p.product_code
LEFT JOIN category_timeslot_stats c ON b.product_mgroup = c.product_mgroup AND b.time_slot = c.time_slot
LEFT JOIN category_overall_stats co ON b.product_mgroup = co.product_mgroup
```

### 전처리 및 파이프라인
- **수치형 특성:**
  - `product_price`, `product_avg_sales`, `product_broadcast_count`, `category_timeslot_avg_sales`, `timeslot_specialty_score`, `temperature`, `precipitation`, `time_slot_int`
- **범주형 특성:**
  - `weekday`, `season`, `weather`, `product_lgroup`, `product_mgroup`, `product_sgroup`, `product_dgroup`, `product_type`, `time_slot`, `time_category_interaction`
- **텍스트 특성:**
  - `product_name`, `keyword` (TF-IDF + Mecab 형태소 분석기 사용)
- **모델:**
  - `XGBRegressor` (n_estimators=500, learning_rate=0.05 등 하이퍼파라미터)
- **전체 파이프라인:**
  - Scikit-learn `Pipeline`
    - ColumnTransformer로 수치/범주/텍스트 특성 각각 처리
    - 최종적으로 XGBoost 회귀 모델에 입력
- **학습 실행:**
  - `python train.py`
  - 학습 완료 후 `backend/app/xgb_broadcast_sales.joblib`에 모델 저장

---

### 2. 예측(추천) 파이프라인

#### 입력 파라미터
- **날짜:** `date` (YYYY-MM-DD)
- **시간대:** `time_slots` (예: "오전,오후,저녁")
- **상품 코드:** `product_codes` (or 카테고리)
- **날씨 정보:** (없으면 자동으로 조회)

#### 파라미터 처리 및 후보 생성
- **카테고리 모드/상품 모드:**
  - 카테고리 모드: 중분류/소분류 등 카테고리별 추천
  - 상품 모드: 개별 상품별 추천
- **후보 생성:**
  - 입력받은 모든 시간대 × 상품/카테고리 조합을 생성
  - 각 후보에 대해 날짜, 요일, 계절, 시간대(숫자), 날씨, 카테고리별 통계 등 feature를 벡터화하여 추가
- **Feature Engineering:**
  - 학습과 동일하게 각종 통계/파생변수 계산
  - 결측값은 동일하게 처리

#### 예측 및 결과 포맷
- **예측:**
  - 후보 DataFrame에서 학습된 파이프라인의 feature만 추출
  - `model.predict()`로 매출 예측
- **정렬 및 상위 N개 선택:**
  - 예측 매출 기준 내림차순 정렬
  - 시간대별 상위 N개 후보 선택
- **최종 반환 구조:**
  - `time_slot`: 추천 시간대
  - `predicted_sales`: 예측 매출
  - `product_code` (or `category`): 추천 상품/카테고리
  - `features`: 추천 후보의 상세 정보(딕셔너리)

#### API/CLI 사용 예시
- **API:**
  - `/api/v1/recommend`
  - Request: `{ "user_query": "내일 오전에 건강식품 뭐 팔면 좋을까?" }`
- **CLI:**
  - `python broadcast_recommender.py recommend --date 2025-07-18 --time_slots "오전,오후,저녁" --products "P001,P002"`

---

## 📝 최근 변경사항 (2025-07-24)

| 구분 | 내용 |
|------|------|
| 모델 피처 | • `broadcast_tape_code` 완전 제거<br>• `broadcast_showhost` 학습/예측엔 사용하지만 **UI 출력에서 제외** |
| 추천 로직 | • 동일 `product_lgroup` 편성 **최대 2회** 제한 → 카테고리 다양성 강화<br>• `top_k_sample` softmax 샘플링 온도(`--diversity_temp`) 추가 |
| 카테고리 전용 모드 | • `--category` 플래그 및 `--categories` 인자 지원 → 특정 카테고리(예: 식품)로만 후보 제한 |
| CLI 인자 | `--top_k_sample`, `--diversity_temp`, `--top_n` 등 세분화 옵션 추가 |
| 배포 가이드 | Docker 재배포 추천 순서<br>```bash
docker compose down --remove-orphans
git pull
docker compose build --no-cache
docker compose up -d
```|

위 변경으로 추천 결과 다양성이 향상되고, 특정 카테고리 전용 편성도 손쉽게 요청할 수 있습니다.

## 🔍 어떻게 질문을 이해하나요?
사용자가 입력한 문장은 OpenAI GPT로 전송되어 **날짜, 시간대, 키워드, 상품코드, 카테고리 등**을 추출한 JSON 형태의 파라미터로 변환됩니다.

예시 입력 → 추출 JSON

```text
"내일 저녁에 루테인 제품 방송하면 얼마나 팔릴까?"
```

```json
{
  "date": "2025-07-24",
  "time_slots": ["저녁"],
  "keywords": ["루테인"],
  "mode": null,
  "products": null,
  "categories": null
}
```

이 JSON 이 `recommend()` 함수로 전달되어 모델 예측에 활용됩니다.

---

## 🚧 향후 개선 로드맵
- 🔬 **모델 고도화**: 하이퍼파라미터 튜닝, LightGBM/TabNet 앙상블 실험
- 🗣️ **질문 이해 향상**: 키워드, 상품명 외에 프로모션·할인 조건 등 추가 파싱
- 📈 **모니터링**: 실시간 예측 정확도·매출 대비 그래프 대시보드(Grafana)
- 🌐 **REST API**: FastAPI 기반 추천/학습 엔드포인트 분리 제공
- ☁️ **배포 자동화**: GitHub Actions + Docker Hub, Kubernetes Helm 차트

---
