# --- Dockerfile for Home Shopping Next.js Frontend (운영 서버용) ---
FROM node:20-alpine

# 작업 디렉토리 생성
WORKDIR /app

# package.json, package-lock.json 우선 복사 (캐싱 최적화)
COPY package*.json ./

# 의존성 설치
RUN npm install

# 전체 프론트엔드 코드 복사
COPY . .

# Next.js 빌드
RUN npm run build

# 3001 포트 오픈
EXPOSE 3001

# 운영 기본 명령: Next.js 서버 실행
CMD ["npm", "start"]
