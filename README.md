# 🛍️ Home Shopping Broadcast Sales Prediction & Recommender

한국어 홈쇼핑 방송의 편성·매출 예측 파이프라인입니다. 숫자·범주형 피처뿐 아니라 **상품명(`product_name`) / 키워드(`keyword`)** 텍스트를 Mecab + TF-IDF 로 벡터화하여 XGBoost 모델이 예측 정확도를 높입니다.

## 주요 기능
1. **학습 스크립트** `broadcast_recommender.py`
   - PostgreSQL 에서 학습 데이터 로드
   - 파이프라인 구성: 수치형, 범주형, 텍스트(Mecab tokenizer)
   - 모델 평가(MAE / RMSE / R²) 후 `xgb_broadcast_sales.joblib` 저장
2. **추천 API** `recommend()`
   - 날짜·시간대·상품코드(또는 카테고리)·날씨를 입력받아 예상 매출을 기반으로 방송 편성 추천
   - 키워드만 주어도 **`product_name` / `keyword` 컬럼** 부분 매칭으로 후보 상품 검색
3. **Streamlit 챗봇** `streamlit_app.py`
   - 자연어 질문 → OpenAI LLM 으로 파라미터(JSON) 추출
   - 추천 결과를 표 형식으로 시각화
4. **Docker 배포**
   - `python:3.11-slim` 기반, `requirements.txt` 단일 관리
   - `docker-compose.yml` 에 DB(PostgreSQL) + 앱 서비스 예시 포함

## 환경 구성
```bash
# Python 3.11 권장 (mecab-python3 wheel 지원)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Mecab 사전은 `mecab-python3` wheel 에 포함되어 추가 설정이 필요 없습니다.

## 학습
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

## 로컬 추천 예시
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

## Streamlit 실행
```bash
streamlit run streamlit_app.py
```
환경변수 `OPENAI_API_KEY` 가 필요합니다.
👉 데모 서버가 구동 중이라면 브라우저에서 [http://175.106.97.27:8501/](http://175.106.97.27:8501/) 로 바로 접속해 체험할 수 있습니다.

## Docker 실행 (학습은 로컬, 추천만 컨테이너로)
```bash
# 빌드 및 백그라운드 기동
docker compose up -d --build

# 로그 확인
docker compose logs -f app
```

## 주요 파일 구조
```
├── broadcast_recommender.py  # 학습 + 추천 백엔드
├── tokenizer_utils.py        # Mecab 토크나이저 모듈 (joblib 호환)
├── streamlit_app.py          # 챗봇 UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 기여 / TODO
- 하이퍼파라미터 튜닝 & 모델 앙상블
- 모델 모니터링 지표 대시보드
- API 서버(FastAPI) 분리 배포

---
© 2025 Windsurf & GunsooLee/trnAI project.
