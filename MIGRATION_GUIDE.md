# NETEZZA → PostgreSQL 매일 자동 마이그레이션 가이드

## 📋 목차
1. [환경 설정](#1-환경-설정)
2. [Docker 재빌드](#2-docker-재빌드)
3. [수동 실행 테스트](#3-수동-실행-테스트)
4. [매일 자동 실행 설정](#4-매일-자동-실행-설정)
5. [모니터링 및 트러블슈팅](#5-모니터링-및-트러블슈팅)

---

## 1. 환경 설정

### 1-1. NETEZZA 연결 정보 설정

`backend/.env` 파일에 NETEZZA 연결 정보를 추가합니다:

```bash
# backend/.env 파일 편집
nano backend/.env
```

```env
# NETEZZA 연결 정보
NETEZZA_HOST=10.x.x.x  # 실제 NETEZZA 서버 IP
NETEZZA_PORT=5480
NETEZZA_DATABASE=실제_데이터베이스명
NETEZZA_USER=실제_사용자명
NETEZZA_PASSWORD=실제_비밀번호

# PostgreSQL 연결 정보 (Docker 내부)
DB_URI=postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB

# OpenAI API (임베딩용)
OPENAI_API_KEY=실제_OpenAI_API_키
```

### 1-2. 마이그레이션 스크립트 수정

`backend/app/migrate_netezza_to_postgres.py` 파일을 열어서 다음 부분을 **실제 NETEZZA 테이블명**으로 수정:

```python
# 67~78줄 수정
query = """
SELECT 
    product_code,
    product_name,
    category_main_name AS category_main,
    category_middle_name AS category_middle,
    category_sub_name AS category_sub,
    price,
    brand,
    '유형' AS product_type,
    CURRENT_TIMESTAMP AS created_at
FROM 
    실제_상품_테이블명  -- 여기 수정!
WHERE 
    상태필드 = '활성'  -- 필터 조건 수정!
ORDER BY 
    product_code
"""
```

---

## 2. Docker 재빌드

### 2-1. Docker 재빌드 및 재시작

```bash
cd /home/trn/trnAi

# 기존 컨테이너 중지 및 삭제
docker-compose down

# 이미지 재빌드 및 컨테이너 재시작
docker-compose up -d --build

# 빌드 로그 확인
docker-compose logs -f fastapi_backend
```

### 2-2. 패키지 설치 확인

```bash
# pyodbc 설치 확인
docker exec -it fastapi_backend pip list | grep pyodbc

# 결과: pyodbc 5.2.0 (설치됨)
```

### 2-3. ODBC 드라이버 확인

```bash
# ODBC 드라이버 목록 확인
docker exec -it fastapi_backend odbcinst -q -d

# NETEZZA ODBC 드라이버가 없으면 별도 설치 필요
```

---

## 3. 수동 실행 테스트

### 3-1. 첫 실행 (테스트)

```bash
# Docker 컨테이너 내부에서 스크립트 실행
docker exec -it fastapi_backend python app/migrate_netezza_to_postgres.py
```

**예상 출력:**
```
============================================================
NETEZZA → PostgreSQL 데이터 마이그레이션
============================================================
✅ NETEZZA 연결 성공
✅ PostgreSQL 연결 성공
📥 NETEZZA에서 상품 데이터 추출 중...
   추출 완료: 1234개 상품
🧹 데이터 정제 중...
   정제 후: 1234개 상품
📤 PostgreSQL taigoods 테이블에 적재 중...
   ✅ 1234개 레코드 적재 완료
============================================================
✅ 마이그레이션 완료!
============================================================
   상품 데이터: 1234개
```

### 3-2. 결과 확인

```bash
# PostgreSQL 접속
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB

# 데이터 확인
SELECT COUNT(*) FROM taigoods;
SELECT category_main, COUNT(*) FROM taigoods GROUP BY category_main;

# 최신 10개 상품
SELECT product_code, product_name, category_main FROM taigoods ORDER BY created_at DESC LIMIT 10;
```

### 3-3. 임베딩 생성 (필수!)

```bash
# 상품 임베딩 생성 (Qdrant에 벡터 저장)
docker exec -it fastapi_backend python app/setup_product_embeddings.py
```

---

## 4. 매일 자동 실행 설정

### 방법 1: 리눅스 크론 사용 (권장) ⭐

#### 4-1. 자동 설정 스크립트 실행

```bash
# 실행 권한 부여
chmod +x /home/trn/trnAi/setup_daily_migration.sh

# 크론 자동 설정
bash /home/trn/trnAi/setup_daily_migration.sh
```

#### 4-2. 수동 크론 설정 (또는)

```bash
# 크론 편집기 열기
crontab -e

# 다음 라인 추가 (매일 새벽 2시 실행)
0 2 * * * cd /home/trn/trnAi && docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py >> /var/log/netezza_migration.log 2>&1
```

**크론 시간 설정 예시:**
- `0 2 * * *` - 매일 새벽 2시
- `0 */6 * * *` - 6시간마다
- `0 0 * * 0` - 매주 일요일 자정
- `0 1 * * 1-5` - 평일 새벽 1시

#### 4-3. 크론 작동 확인

```bash
# 크론 작업 목록 확인
crontab -l

# 크론 서비스 상태 확인
sudo systemctl status cron  # Ubuntu/Debian
sudo systemctl status crond  # CentOS/RHEL
```

### 방법 2: Docker 컨테이너 내부 크론

#### 4-2-1. Dockerfile에 크론 추가

```dockerfile
# Dockerfile에 추가
RUN apt-get update && apt-get install -y cron

# 크론 파일 복사
COPY migration_cron /etc/cron.d/migration_cron
RUN chmod 0644 /etc/cron.d/migration_cron && crontab /etc/cron.d/migration_cron
```

#### 4-2-2. 크론 파일 생성 (`migration_cron`)

```bash
# migration_cron 파일 내용
0 2 * * * root cd /app && python app/migrate_netezza_to_postgres.py >> /var/log/migration.log 2>&1
```

---

## 5. 모니터링 및 트러블슈팅

### 5-1. 로그 확인

```bash
# 마이그레이션 로그 확인
tail -f /var/log/netezza_migration.log

# 최근 50줄 확인
tail -50 /var/log/netezza_migration.log

# 오류만 필터링
grep "❌" /var/log/netezza_migration.log
```

### 5-2. 수동 실행 (긴급)

```bash
# 즉시 마이그레이션 실행
docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py

# 임베딩도 함께 실행
docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py && \
docker exec -i fastapi_backend python app/setup_product_embeddings.py
```

### 5-3. 일반적인 오류 및 해결

#### 오류 1: NETEZZA 연결 실패

```
❌ NETEZZA 연결 실패: [08001] Socket closed
```

**해결:**
1. NETEZZA 서버 IP/포트 확인
2. 방화벽 규칙 확인
3. ODBC 드라이버 설치 확인

#### 오류 2: PostgreSQL 중복 키 오류

```
❌ 적재 실패: duplicate key value violates unique constraint
```

**해결:**
- `if_exists='append'` → `if_exists='replace'` 변경 (전체 재생성)
- 또는 증분 업데이트 로직 구현

#### 오류 3: OpenAI API 키 없음

```
❌ OpenAI API 키가 설정되지 않았습니다.
```

**해결:**
- `backend/.env` 파일에 `OPENAI_API_KEY` 추가
- Docker 재시작: `docker-compose restart fastapi_backend`

### 5-4. 성능 모니터링

```bash
# 마이그레이션 실행 시간 측정
time docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py

# 데이터베이스 크기 확인
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename IN ('taigoods', 'taipgmtape')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

---

## 6. 추가 최적화 (선택사항)

### 6-1. 증분 업데이트 구현

마이그레이션 스크립트에서 `updated_at` 기준으로 변경된 데이터만 가져오기:

```python
# WHERE 조건 추가
WHERE updated_at > (SELECT MAX(updated_at) FROM taigoods_temp)
```

### 6-2. 병렬 처리

대용량 데이터의 경우 멀티프로세싱 활용:

```python
from multiprocessing import Pool

with Pool(4) as p:
    p.map(process_batch, batches)
```

### 6-3. 실패 시 알림 (이메일/슬랙)

```python
# 스크립트에 추가
import requests

def send_slack_alert(message):
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    requests.post(webhook_url, json={'text': message})
```

---

## 7. 체크리스트

- [ ] NETEZZA 연결 정보 설정 (`.env`)
- [ ] 테이블명 및 쿼리 수정 (`migrate_netezza_to_postgres.py`)
- [ ] Docker 재빌드 완료
- [ ] 수동 실행 테스트 성공
- [ ] 데이터 확인 (PostgreSQL)
- [ ] 임베딩 생성 완료
- [ ] 크론 작업 등록
- [ ] 로그 확인 설정
- [ ] 첫 자동 실행 확인 (다음날)

---

## 📞 문제 발생 시

1. **로그 확인**: `/var/log/netezza_migration.log`
2. **수동 실행**: `docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py`
3. **환경변수 확인**: `docker exec -it fastapi_backend env | grep NETEZZA`
4. **연결 테스트**: NETEZZA/PostgreSQL 직접 접속 확인

---

**작성일**: 2025-10-13  
**버전**: 1.0
