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

## 🚀 지금 바로 써보기 (1분)
1. 위의 **데모 바로가기** 링크를 클릭합니다.
2. 챗봇 입력창에 궁금한 점을 자연어로 물어보세요.  
   • 예) _"이번 주말 저녁에 4050 여성 화장품 뭐 팔면 좋을까?"_
3. 결과 표를 확인하고, 필요하면 조건을 바꿔 다시 질문하면 끝!

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

### Streamlit 실행
```bash
streamlit run streamlit_app.py
```
환경변수 `OPENAI_API_KEY` 가 필요합니다.

### Docker 실행 (학습은 로컬, 추천만 컨테이너로)
```bash
# 빌드 및 백그라운드 기동
docker compose up -d --build

# 로그 확인
docker compose logs -f app
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
