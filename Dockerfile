# --- Dockerfile for Home Shopping FastAPI Recommender (운영 서버용) ---
FROM python:3.11-slim

# 필수 시스템 패키지 설치 (psycopg2, 빌드툴 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /app

# requirements 우선 복사 및 설치 (레이어 캐싱)
COPY backend/requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 전체 백엔드 코드 복사
COPY backend/. /app

# 모델 파일도 반드시 포함 (이미 학습된 경우)
# COPY backend/app/xgb_broadcast_sales.joblib /app/app/xgb_broadcast_sales.joblib
# (이미 backend/app/ 하위면 위 COPY로 포함됨)

# 포트 오픈 (FastAPI)
EXPOSE 8501

# 운영 기본 명령: FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8501"]
