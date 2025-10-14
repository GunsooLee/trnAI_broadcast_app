# ✅ NETEZZA → PostgreSQL 매일 자동 마이그레이션 완료!

## 📋 수정된 내용

### 1. **실제 NETEZZA 쿼리 적용**
- FBD_PGMCPF_M, DST_GDS_MST_EXT, DST_GDS_CLS, DST_BRND_MST 테이블 연동
- 실제 컬럼명으로 매핑 완료

### 2. **증분 업데이트 구현**
- 매일 배치 실행 시 **어제 이후 수정된 데이터만** 가져오기
- `DGM.REG_DTTM >= 어제날짜` 조건 자동 추가

### 3. **UPSERT 방식 적용**
- 중복 데이터 발생 시 자동 업데이트
- `ON CONFLICT DO UPDATE` 사용

### 4. **자동 임베딩 생성**
- 마이그레이션 후 10분 뒤 자동으로 임베딩 생성
- 크론 2개 등록: 2:00 마이그레이션 → 2:10 임베딩

---

## 🚀 지금 바로 실행하기

### Step 1: NETEZZA 연결 정보 확인
```bash
# backend/.env 파일에 이미 추가했는지 확인
cat backend/.env | grep NETEZZA
```

### Step 2: 수동 테스트 (필수!)
```bash
# Docker 재빌드 (Dockerfile 수정됨)
docker-compose down
docker-compose up -d --build

# 첫 마이그레이션 테스트 (전체 데이터)
docker exec -it fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"
```

**예상 출력:**
```
============================================================
NETEZZA → PostgreSQL 데이터 마이그레이션
모드: 전체 재처리
============================================================
✅ NETEZZA 연결 성공
✅ PostgreSQL 연결 성공
📥 NETEZZA에서 상품 데이터 추출 중 (전체)...
   추출 완료: 1234개 상품
🧹 데이터 정제 중...
   정제 후: 1234개 상품
📤 PostgreSQL taigoods 테이블에 UPSERT 중...
   ✅ 1234개 레코드 UPSERT 완료

============================================================
✅ 마이그레이션 완료!
============================================================
   상품 데이터: 1234개
   테이프 데이터: 567개
```

### Step 3: 데이터 확인
```bash
# PostgreSQL 접속
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB

# 데이터 확인
SELECT COUNT(*) FROM taigoods;
SELECT category_main, COUNT(*) FROM taigoods GROUP BY category_main;
SELECT * FROM taigoods LIMIT 5;
```

### Step 4: 임베딩 생성
```bash
docker exec -it fastapi_backend python app/setup_product_embeddings.py
```

### Step 5: 매일 자동 실행 설정
```bash
# 크론 자동 설정
chmod +x /home/trn/trnAi/setup_daily_migration.sh
bash /home/trn/trnAi/setup_daily_migration.sh

# 크론 확인
crontab -l
```

---

## 📊 SQL 쿼리 구조

### 상품 데이터 (TAIGOODS)
```sql
SELECT A.REP_GDS_CD AS product_code
     , DGM.GDS_NM   AS product_name
     , DGC.GDS_LCLS_NM  AS category_main
     , DGC.GDS_MCLS_NM  AS category_middle
     , DGC.GDS_SCLS_NM  AS category_sub
     , DGM.GDS_SEL_UPRC AS price
     , DBM.BRND_NM AS brand
  FROM (
    SELECT DISTINCT FPM.REP_GDS_CD AS REP_GDS_CD
      FROM SNTDM.SNTADM.FBD_PGMCPF_M FPM
     WHERE FPM.STRD_YMD >= '20240101'
       AND FPM.PROG_TAPE_CD LIKE '0000%'
       ) A
  JOIN SNTDM.SNTADM.DST_GDS_MST_EXT DGM
    ON (A.REP_GDS_CD = DGM.GDS_CD)
  JOIN SNTDM.SNTADM.DST_GDS_CLS DGC
    ON (DGM.GDS_USCLS_CD = DGC.GDS_USCLS_CD)
  JOIN SNTDM.SNTADM.DST_BRND_MST DBM
    ON (DGM.BRND_CD = DBM.BRND_CD)
 WHERE DGM.REG_DTTM >= '20251012'  -- 어제 날짜 (자동 계산)
```

### 방송테이프 데이터 (TAIPGMTAPE)
```sql
SELECT DISTINCT
       FPM.PROG_TAPE_CD AS tape_code,
       FPM.PGMC_NM AS tape_name,
       FPM.REP_GDS_CD AS product_code,
       'ready' AS production_status
  FROM SNTDM.SNTADM.FBD_PGMCPF_M FPM
 WHERE FPM.STRD_YMD >= '20240101'
   AND FPM.PROG_TAPE_CD LIKE '0000%'
   AND FPM.STRD_YMD >= '20251012'  -- 어제 날짜 (자동 계산)
```

---

## ⏰ 크론 작업 (매일 자동 실행)

### 등록된 크론 작업
```bash
# 매일 새벽 2:00 - 데이터 마이그레이션
0 2 * * * cd /home/trn/trnAi && docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py >> /var/log/netezza_migration.log 2>&1

# 매일 새벽 2:10 - 임베딩 생성
10 2 * * * cd /home/trn/trnAi && docker exec -i fastapi_backend python app/setup_product_embeddings.py >> /var/log/product_embedding.log 2>&1
```

### 로그 확인
```bash
# 실시간 로그
tail -f /var/log/netezza_migration.log
tail -f /var/log/product_embedding.log

# 최근 50줄
tail -50 /var/log/netezza_migration.log

# 오류만 필터링
grep "❌" /var/log/netezza_migration.log
```

---

## 🔧 모드 변경

### 증분 업데이트 (기본값)
```bash
# 어제 이후 수정된 데이터만
docker exec -it fastapi_backend python app/migrate_netezza_to_postgres.py
```

### 전체 재처리
```bash
# 모든 데이터 다시 가져오기
docker exec -it fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"
```

---

## 📌 중요 포인트

### 1. **증분 업데이트 조건**
- 상품: `DGM.REG_DTTM >= 어제날짜`
- 테이프: `FPM.STRD_YMD >= 어제날짜`

### 2. **UPSERT 동작**
- 상품코드(`product_code`) 중복 시 → 업데이트
- 테이프코드(`tape_code`) 중복 시 → 업데이트
- 신규 데이터 → 삽입

### 3. **필터 조건**
- `STRD_YMD >= '20240101'` - 2024년 이후 데이터만
- `PROG_TAPE_CD LIKE '0000%'` - 특정 테이프 코드 패턴

### 4. **자동화 흐름**
```
매일 새벽 2:00 → 마이그레이션 실행 (어제 수정분)
    ↓ (10분 대기)
매일 새벽 2:10 → 임베딩 생성 (신규/수정 상품만)
    ↓
추천 시스템에서 사용 가능!
```

---

## 🐛 트러블슈팅

### 오류 1: NETEZZA 연결 실패
```
❌ NETEZZA 연결 실패: [HY000] [IBM] ...
```
**해결:** `.env` 파일의 NETEZZA_HOST, NETEZZA_PORT, NETEZZA_USER 확인

### 오류 2: ODBC 드라이버 없음
```
❌ ('01000', "[01000] [unixODBC][Driver Manager]Can't open lib 'NetezzaSQL'")
```
**해결:** Dockerfile에 NETEZZA ODBC 드라이버 설치 필요 (별도 제공)

### 오류 3: 데이터 0건
```
📥 NETEZZA에서 상품 데이터 추출 중...
   추출 완료: 0개 상품
```
**해결:** 
1. SQL 쿼리의 날짜 필터 확인 (`STRD_YMD >= '20240101'`)
2. 테이블명 오타 확인
3. NETEZZA에서 직접 쿼리 실행해보기

---

## ✅ 완료 체크리스트

- [x] `backend/.env`에 NETEZZA 연결 정보 추가
- [x] `migrate_netezza_to_postgres.py` 실제 쿼리 적용
- [x] Dockerfile ODBC 드라이버 지원 추가
- [x] Docker 재빌드
- [ ] 수동 테스트 (전체 동기화)
- [ ] 데이터 확인 (PostgreSQL)
- [ ] 임베딩 생성 확인
- [ ] 크론 작업 등록
- [ ] 내일 자동 실행 확인

---

**작성일:** 2025-10-13  
**버전:** 2.0 (실제 쿼리 적용)
