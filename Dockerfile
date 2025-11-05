# --- Dockerfile for Home Shopping FastAPI Recommender (운영 서버용) ---
FROM python:3.11-slim

# 필수 시스템 패키지 설치 (psycopg2, 빌드툴, ODBC 드라이버, Chrome 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        unixodbc \
        unixodbc-dev \
        libodbcinst2 \
        unixodbc-common \
        odbcinst \
        wget \
        curl \
        gnupg \
        ca-certificates && \
    # Chrome 설치
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# NETEZZA ODBC 드라이버 설치 (선택사항 - 드라이버 파일이 있는 경우)
# COPY netezza_odbc_driver.tar.gz /tmp/
# RUN cd /tmp && tar -xzf netezza_odbc_driver.tar.gz && \
#     cp -r lib/* /usr/lib/ && \
#     cp -r bin/* /usr/bin/ && \
#     rm -rf /tmp/netezza_odbc_driver.tar.gz

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
