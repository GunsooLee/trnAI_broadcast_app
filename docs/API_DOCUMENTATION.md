# 📊 홈쇼핑 매출 예측 API 문서

## 목차
- [개요](#개요)
- [API 엔드포인트](#api-엔드포인트)
- [개발자용 상세 가이드](#개발자용-상세-가이드)
- [사용 예시](#사용-예시)
- [에러 처리](#에러-처리)
- [성능 및 제약사항](#성능-및-제약사항)

---

## 개요

### 시스템 목적
홈쇼핑 방송 편성을 위한 AI 기반 매출 예측 시스템입니다. XGBoost 머신러닝 모델을 사용하여 상품, 시간대, 날씨, 공휴일 등 다양한 요인을 고려한 정확한 매출 예측을 제공합니다.

### 주요 기능
- ✅ 단일 상품 매출 예측 (시간대별)
- ✅ 날짜별 편성표 기반 일괄 예측
- ✅ 신상품 예측 (과거 실적 없어도 가능)
- ✅ 시간대 최적화 추천

### 기술 스택
- **Backend**: FastAPI (Python 3.11)
- **ML Model**: XGBoost (Stacking Ensemble)
- **Database**: PostgreSQL 14
- **Container**: Docker

---

## API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:8501` (또는 배포된 서버 주소)
- **Content-Type**: `application/json`
- **인증**: 현재 버전에서는 인증 불필요

---

## 1. 단일 상품 매출 예측 API

### `POST /api/v1/sales/predict-single`

방송 테이프 코드와 방송 일시를 입력하면 해당 조건에서의 예상 매출을 예측합니다.

#### Request

```json
{
  "tape_code": "0000012179",
  "broadcast_start_time": "2026-02-22 14:00:00",
  "broadcast_end_time": "2026-02-22 15:00:00"
}
```

**Parameters:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|------|------|------|
| `tape_code` | string | ✅ | 방송 테이프 코드 | "0000012179" |
| `broadcast_start_time` | string | ✅ | 방송 시작 일시 (YYYY-MM-DD HH:MM:SS) | "2026-02-22 14:00:00" |
| `broadcast_end_time` | string | ⭕ | 방송 종료 일시 (YYYY-MM-DD HH:MM:SS, 선택적) | "2026-02-22 15:00:00" |

#### Response (200 OK)

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

**Response Fields:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `product_code` | string | 상품 코드 |
| `product_name` | string | 상품명 |
| `broadcast_datetime` | string | 방송 일시 (ISO 8601) |
| `predicted_sales` | float | 예측 매출액 (원) |
| `confidence` | float | 예측 신뢰도 (0~1) |
| `features_used` | object | 예측에 사용된 주요 피처 |

#### Error Responses

**404 Not Found** - 방송 테이프를 찾을 수 없음
```json
{
  "detail": "방송 테이프를 찾을 수 없습니다: 0000099999"
}
```

**400 Bad Request** - 잘못된 날짜/시간 형식
```json
{
  "detail": "방송 시작 일시 형식이 올바르지 않습니다. (YYYY-MM-DD HH:MM:SS)"
}
```

---

## 2. 날짜별 편성표 기반 예측 API

### `POST /api/v1/sales/predict`

특정 날짜의 모든 편성 방송에 대한 매출을 일괄 예측합니다.

#### Request

```json
{
  "date": "2026-02-22"
}
```

**Parameters:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|------|------|------|
| `date` | string | ✅ | 예측할 날짜 (YYYY-MM-DD) | "2026-02-22" |

#### Response (200 OK)

```json
{
  "date": "2026-02-22",
  "predictions": [
    {
      "product_code": "15750903",
      "product_name": "세일 토비콤 루테인 지아잔틴 12박스",
      "broadcast_time": "09:00",
      "duration_minutes": 30,
      "predicted_sales": 28727542.0,
      "confidence": 0.85
    },
    {
      "product_code": "20294307",
      "product_name": "[세일]코지마 안마의자 데코르 CMC-A35",
      "broadcast_time": "10:00",
      "duration_minutes": 30,
      "predicted_sales": 28833246.0,
      "confidence": 0.85
    }
  ]
}
```

**Response Fields:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `date` | string | 예측 날짜 |
| `predictions` | array | 상품별 예측 결과 배열 |
| `predictions[].product_code` | string | 상품 코드 |
| `predictions[].product_name` | string | 상품명 |
| `predictions[].broadcast_time` | string | 방송 시간 (HH:MM) |
| `predictions[].duration_minutes` | integer | 방송 시간(분) |
| `predictions[].predicted_sales` | float | 예측 매출액 (원) |
| `predictions[].confidence` | float | 예측 신뢰도 (0~1) |

**빈 결과 (편성 없음):**
```json
{
  "date": "2026-03-01",
  "predictions": []
}
```

---

## 3. 모델 학습 API

### `POST /api/v1/training/start-sync`

XGBoost 모델을 최신 데이터로 재학습합니다.

#### Request

```json
{
  "force_retrain": false
}
```

**Parameters:**

| 필드 | 타입 | 필수 | 설명 | 기본값 |
|------|------|------|------|--------|
| `force_retrain` | boolean | ❌ | 강제 재학습 여부 | false |

#### Response (200 OK)

```json
{
  "status": "success",
  "message": "모델 학습 완료",
  "metrics": {
    "r2_score": 0.603,
    "mae": 3000000,
    "rmse": 6500000
  },
  "training_data_count": 18908,
  "model_path": "/app/xgb_broadcast_profit.joblib"
}
```

---

## 4. 데이터 마이그레이션 API

### `POST /api/v1/migration/start-sync`

NETEZZA에서 PostgreSQL로 데이터를 동기화합니다.

#### Request

```json
{
  "full_sync": false,
  "tables": "TAIGOODS,TAIPGMTAPE"
}
```

**Parameters:**

| 필드 | 타입 | 필수 | 설명 | 기본값 |
|------|------|------|------|--------|
| `full_sync` | boolean | ❌ | 전체 재동기화 여부 | false |
| `tables` | string | ❌ | 동기화할 테이블 (쉼표 구분) | 모든 테이블 |

**사용 가능한 테이블:**
- `TAIGOODS` - 상품 마스터
- `TAIPGMTAPE` - 방송 테이프
- `TAIBROADCASTS` - 방송 이력
- `TAICOMPETITOR_BROADCASTS` - 경쟁사 편성

#### Response (200 OK)

```json
{
  "status": "success",
  "message": "마이그레이션 완료",
  "total_rows": 12053,
  "tables": {
    "TAIGOODS": {
      "rows": 5019,
      "status": "완료"
    },
    "TAIPGMTAPE": {
      "rows": 7034,
      "status": "완료"
    }
  }
}
```

---

## 개발자용 상세 가이드

### 1. 환경 설정

#### Docker 환경에서 실행

```bash
# 컨테이너 시작
docker-compose up -d

# API 서버 상태 확인
curl http://localhost:8501/api/v1/health

# 응답: {"status": "ok"}
```

#### 로컬 개발 환경

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집 (DB 연결 정보 등)

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8501 --reload
```

### 2. 데이터베이스 스키마

#### TAIGOODS (상품 마스터)
```sql
CREATE TABLE TAIGOODS (
    product_code VARCHAR(20) PRIMARY KEY,
    product_name TEXT NOT NULL,
    category_main VARCHAR(100),
    category_middle VARCHAR(100),
    category_sub VARCHAR(100),
    price NUMERIC(15,2),
    brand VARCHAR(100),
    product_type VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### TAIPGMTAPE (방송 테이프)
```sql
CREATE TABLE TAIPGMTAPE (
    tape_code VARCHAR(20) PRIMARY KEY,
    tape_name TEXT,
    product_code VARCHAR(20) REFERENCES TAIGOODS(product_code),
    production_status VARCHAR(20),
    created_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### TAIBROADCASTS (방송 이력)
```sql
CREATE TABLE TAIBROADCASTS (
    id SERIAL PRIMARY KEY,
    tape_code VARCHAR(20) REFERENCES TAIPGMTAPE(tape_code),
    broadcast_start_timestamp TIMESTAMP NOT NULL,
    product_is_new BOOLEAN,
    gross_profit NUMERIC(15,2),
    sales_efficiency NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. 모델 피처 설명

#### 입력 피처 (총 100개 이상)

**카테고리 피처:**
- `category_main` - 대분류 (의류, 건강식품 등)
- `category_middle` - 중분류
- `category_sub` - 소분류

**시간 피처:**
- `hour` - 방송 시간 (0~23)
- `hour_sin`, `hour_cos` - 시간 순환 인코딩
- `time_slot` - 시간대 (새벽/오전/오후/저녁)
- `day_of_week` - 요일
- `is_weekend` - 주말 여부

**계절 피처:**
- `season` - 계절 (봄/여름/가을/겨울)
- `spring_weight`, `summer_weight`, `autumn_weight`, `winter_weight` - 계절 경계 가중치

**상품 피처:**
- `product_price_log` - 가격 (로그 스케일)
- `product_avg_profit` - 상품 과거 평균 매출
- `product_broadcast_count` - 상품 방송 횟수

**카테고리-시간 상호작용:**
- `category_timeslot_avg_profit` - 카테고리별 시간대 평균 매출
- `timeslot_specialty_score` - 시간대 특화 점수

**키워드 피처 (46개):**
- `kw_특집방송`, `kw_세일`, `kw_루테인` 등 - 상품명 키워드 이진 피처

**날씨 피처:**
- `weather` - 날씨 상태
- `temperature` - 기온
- `precipitation` - 강수량

**공휴일 피처:**
- `is_holiday` - 공휴일 여부
- `holiday_name` - 공휴일명

### 4. 예측 정확도

#### 모델 성능 지표
- **R² Score**: 0.603 (60.3% 설명력)
- **MAE (평균 절대 오차)**: 약 300만원
- **RMSE (평균 제곱근 오차)**: 약 650만원

#### 방송 횟수별 신뢰도

| 방송 횟수 | 예측 방법 | 신뢰도 | MAE |
|----------|----------|--------|-----|
| 0회 (신상품) | 카테고리 평균 | 60~70% | ±500만원 |
| 1~5회 | 상품 평균 + 카테고리 | 70~85% | ±400만원 |
| 6~20회 | 상품 과거 실적 | 85~90% | ±300만원 |
| 20회 이상 | 상품 과거 실적 (충분) | 90~95% | ±200만원 |

---

## 사용 예시

### Python 예시

```python
import requests
import json

# 1. 단일 상품 예측
def predict_single_product(tape_code, broadcast_start_time):
    url = "http://localhost:8501/api/v1/sales/predict-single"
    payload = {
        "tape_code": tape_code,
        "broadcast_start_time": broadcast_start_time
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"상품: {data['product_name']}")
        print(f"예측 매출: {data['predicted_sales']:,.0f}원")
        print(f"신뢰도: {data['confidence']:.0%}")
        return data
    else:
        print(f"오류: {response.status_code} - {response.text}")
        return None

# 사용 예시
result = predict_single_product("0000012179", "2026-02-22 14:00:00")


# 2. 여러 시간대 비교
def find_best_timeslot(tape_code, date):
    times = ["09:00", "14:00", "18:00", "21:00"]
    results = []
    
    for time in times:
        broadcast_start_time = f"{date} {time}"
        result = predict_single_product(tape_code, broadcast_start_time)
        if result:
            results.append({
                "time": time,
                "sales": result["predicted_sales"]
            })
    
    # 최고 매출 시간대 찾기
    best = max(results, key=lambda x: x["sales"])
    print(f"\n최적 시간대: {best['time']} - {best['sales']:,.0f}원")
    return best

# 사용 예시
best_time = find_best_timeslot("0000012179", "2026-02-22")


# 3. 날짜별 편성표 예측
def predict_daily_schedule(date):
    url = "http://localhost:8501/api/v1/sales/predict"
    payload = {"date": date}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"날짜: {data['date']}")
        print(f"총 {len(data['predictions'])}개 방송 예측")
        
        total_sales = sum(p["predicted_sales"] for p in data["predictions"])
        print(f"예상 총 매출: {total_sales:,.0f}원")
        
        return data
    else:
        print(f"오류: {response.status_code}")
        return None

# 사용 예시
schedule = predict_daily_schedule("2026-02-22")
```

### JavaScript (Node.js) 예시

```javascript
const axios = require('axios');

// 1. 단일 상품 예측
async function predictSingleProduct(tapeCode, broadcastStartTime) {
  try {
    const response = await axios.post(
      'http://localhost:8501/api/v1/sales/predict-single',
      {
        tape_code: tapeCode,
        broadcast_start_time: broadcastStartTime
      }
    );
    
    const data = response.data;
    console.log(`상품: ${data.product_name}`);
    console.log(`예측 매출: ${data.predicted_sales.toLocaleString()}원`);
    console.log(`신뢰도: ${(data.confidence * 100).toFixed(0)}%`);
    
    return data;
  } catch (error) {
    console.error('오류:', error.response?.status, error.response?.data);
    return null;
  }
}

// 사용 예시
predictSingleProduct('0000012179', '2026-02-22 14:00');


// 2. 여러 시간대 비교
async function findBestTimeslot(tapeCode, date) {
  const times = ['09:00', '14:00', '18:00', '21:00'];
  const results = [];
  
  for (const time of times) {
    const broadcastTime = `${date} ${time}`;
    const result = await predictSingleProduct(tapeCode, broadcastTime);
    if (result) {
      results.push({
        time: time,
        sales: result.predicted_sales
      });
    }
  }
  
  // 최고 매출 시간대 찾기
  const best = results.reduce((max, r) => 
    r.sales > max.sales ? r : max
  );
  
  console.log(`\n최적 시간대: ${best.time} - ${best.sales.toLocaleString()}원`);
  return best;
}

// 사용 예시
findBestTimeslot('0000012179', '2026-02-22');
```

### cURL 예시

```bash
# 1. 단일 상품 예측
curl -X POST "http://localhost:8501/api/v1/sales/predict-single" \
  -H "Content-Type: application/json" \
  -d '{
    "tape_code": "0000012179",
    "broadcast_start_time": "2026-02-22 14:00:00"
  }'

# 2. 날짜별 편성표 예측
curl -X POST "http://localhost:8501/api/v1/sales/predict" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-02-22"}'

# 3. 헬스 체크
curl http://localhost:8501/api/v1/health
```

---

## 에러 처리

### HTTP 상태 코드

| 코드 | 의미 | 원인 | 해결 방법 |
|------|------|------|----------|
| 200 | 성공 | 정상 처리 | - |
| 400 | Bad Request | 잘못된 요청 데이터 | 요청 형식 확인 |
| 404 | Not Found | 상품을 찾을 수 없음 | 상품 코드 확인 |
| 408 | Timeout | 처리 시간 초과 | 재시도 또는 관리자 문의 |
| 500 | Internal Server Error | 서버 내부 오류 | 로그 확인 또는 관리자 문의 |
| 503 | Service Unavailable | 서비스 일시 중단 | 잠시 후 재시도 |

### 에러 응답 예시

```json
{
  "detail": "상품을 찾을 수 없습니다: 99999999"
}
```

### 에러 처리 권장 사항

```python
import requests
from requests.exceptions import Timeout, ConnectionError

def safe_predict(product_code, date, time, max_retries=3):
    url = "http://localhost:8501/api/v1/sales/predict-single"
    payload = {
        "product_code": product_code,
        "broadcast_date": date,
        "broadcast_time": time
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"상품을 찾을 수 없습니다: {product_code}")
                return None
            elif response.status_code == 400:
                print(f"잘못된 요청: {response.json()['detail']}")
                return None
            else:
                print(f"오류 {response.status_code}: {response.text}")
                
        except Timeout:
            print(f"타임아웃 (시도 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2)  # 2초 대기 후 재시도
        except ConnectionError:
            print("서버 연결 실패")
            return None
    
    return None
```

---

## 성능 및 제약사항

### 성능

- **응답 시간**: 평균 100~300ms (단일 예측)
- **처리량**: 초당 약 10~20 요청
- **동시 접속**: 최대 50 동시 연결

### 제약사항

1. **상품 데이터**
   - TAIGOODS 테이블에 등록된 상품만 예측 가능
   - 현재 약 5,019개 상품 지원

2. **날짜 범위**
   - 과거 데이터: 2022년 1월 1일 이후
   - 미래 예측: 제한 없음 (단, 계절/공휴일 정보 필요)

3. **예측 정확도**
   - 방송 횟수가 적을수록 정확도 낮음
   - 신상품(방송 0회): 카테고리 평균 기반 예측

4. **데이터 업데이트**
   - n8n 워크플로우를 통해 매일 자동 업데이트
   - 수동 업데이트: `/api/v1/migration/start-sync` 호출

### 권장 사항

1. **캐싱 사용**
   - 같은 요청은 결과를 캐싱하여 재사용
   - Redis 등 캐시 서버 활용 권장

2. **배치 처리**
   - 여러 상품 예측 시 `/api/v1/sales/predict` 사용
   - 개별 API 여러 번 호출보다 효율적

3. **에러 처리**
   - 타임아웃, 재시도 로직 구현
   - 404 에러 시 상품 존재 여부 확인

4. **모니터링**
   - 응답 시간, 에러율 모니터링
   - 로그 수집 및 분석

---

## 추가 리소스

### API 문서
- Swagger UI: `http://localhost:8501/docs`
- ReDoc: `http://localhost:8501/redoc`

### 관련 문서
- [비개발자용 가이드](./USER_GUIDE.md)
- [시스템 아키텍처](./ARCHITECTURE.md)
- [데이터 파이프라인](./DATA_PIPELINE.md)

### 지원
- GitHub Issues: [프로젝트 저장소]
- 이메일: [담당자 이메일]
- Slack: [팀 채널]

---

**문서 버전**: 1.1.0  
**최종 업데이트**: 2026-03-18  
**작성자**: AI Development Team
