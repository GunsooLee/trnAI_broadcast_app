# 홈쇼핑 방송 편성 추천 시스템 📺

AI 모델이 방송 시간대별 예상 매출을 예측하고, 가장 높은 매출이 기대되는 **상품 또는 상품 카테고리**를 자동으로 추천합니다.

* Python 3.13 · XGBoost
* Streamlit 웹 UI (포트 8501)
* Docker 한-방 배포

---

## 1. 서버 빠른 시작 (Docker)

```bash
# 코드 내려받기
cd /opt
git clone https://github.com/<YOUR_ORG>/broadcast_recommender.git
cd broadcast_recommender

# 컨테이너 빌드 & 실행
docker compose up -d --build
```

브라우저에서 `http://<서버IP>:8501` 접속 후 날짜·시간대·날씨를 입력하고 **🚀 추천 실행** 버튼을 누르면 결과가 표시됩니다.

### 1.1 중지 / 재배포
```bash
docker compose down      # 중지
git pull                 # 최신 코드 반영
docker compose up -d --build
```

---

## 2. 로컬 개발 (Docker 없이)

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 모델 학습 (DB에서 데이터 로드 후 모델 저장)
python broadcast_recommender.py train

# Streamlit 실행
streamlit run streamlit_app.py
```

---

## 3. CLI 사용 예시

```bash
# 모델 학습
python broadcast_recommender.py train

# 상품 코드 기반 추천
python broadcast_recommender.py recommend \
    --date 2025-07-24 \
    --time_slots "아침,점심,저녁" \
    --products "P1001,P2002,P3003"

# 카테고리 기반 추천
python broadcast_recommender.py recommend \
    --date 2025-07-24 \
    --time_slots "아침,점심,저녁" \
    --category
```

---

## 4. 프로젝트 구조

```
├─ broadcast_recommender.py   # 학습 & 추천 로직
├─ streamlit_app.py           # 웹 UI
├─ requirements.txt           # 의존 패키지 목록
├─ Dockerfile                 # 컨테이너 빌드 정의
├─ docker-compose.yml         # 배포 구성
└─ README.md                  # 이 파일
```

---

## 5. 문제 해결
* 컨테이너가 DB에 연결되지 않을 때 → `docker logs` 로 오류 확인 후 `docker-compose.yml` 의 환경변수, 네트워크 설정을 점검하세요.
* 모델 파일이 없을 때 → 컨테이너 안에서 한 번 `python broadcast_recommender.py train` 실행하거나 로컬에서 학습된 `.joblib` 파일을 복사합니다.

즐거운 방송 편성 자동화 되세요! 🎉
