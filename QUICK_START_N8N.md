# 🚀 NETEZZA 마이그레이션 빠른 시작 (n8n)

## ✅ 완료된 작업

1. **범용 마이그레이션 시스템 구축**
   - 설정 파일 기반 (테이블 추가 간편)
   - UPSERT 방식 (중복 자동 처리)
   - 증분 업데이트 (어제 수정분만)

2. **n8n 워크플로우 생성**
   - 매일 새벽 2시 자동 실행
   - 마이그레이션 → 임베딩 생성 자동화
   - 실패 시 로그 기록

3. **현재 활성화된 테이블**
   - ✅ TAIGOODS (상품 마스터)
   - ✅ TAIPGMTAPE (방송테이프)
   - ✅ TAIBROADCASTS (방송 이력)

---

## 🔧 초기 설정 (5분)

### 1. 환경변수 설정 (`.env` 파일 확인)

```bash
# backend/.env
NETEZZA_HOST=실제_IP
NETEZZA_PORT=5480
NETEZZA_DATABASE=실제_DB명
NETEZZA_USER=실제_사용자
NETEZZA_PASSWORD=실제_비번
```

### 2. Docker 재빌드

```bash
cd /home/trn/trnAi
docker-compose down
docker-compose up -d --build
```

### 3. 첫 테스트 (전체 동기화)

```bash
# 모든 테이블 전체 가져오기
docker exec -it fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"
```

**예상 출력:**
```
======================================================================
NETEZZA → PostgreSQL 데이터 마이그레이션 (범용)
======================================================================
모드: 전체 재처리
대상 테이블: 모든 활성화된 테이블
======================================================================

📋 마이그레이션 대상: 3개 테이블
   - TAIGOODS: 상품 마스터 데이터
   - TAIPGMTAPE: 방송테이프 정보
   - TAIBROADCASTS: 방송 이력 데이터

======================================================================
🔄 TAIGOODS 마이그레이션 시작...
======================================================================
📥 NETEZZA에서 TAIGOODS 추출 중 (전체 데이터)...
   ✅ 추출 완료: 1234개 레코드
🧹 데이터 정제 중...
   정제 후: 1234개 상품
📤 PostgreSQL TAIGOODS 테이블에 UPSERT 중...
   ✅ 1234개 레코드 UPSERT 완료
✅ TAIGOODS: 1234개 레코드 처리 완료

======================================================================
📊 마이그레이션 결과 요약
======================================================================
성공: 3/3 테이블
총 레코드: 2,567개

✅ TAIGOODS                 1,234개  (완료)
✅ TAIPGMTAPE                 567개  (완료)
✅ TAIBROADCASTS              766개  (완료)
======================================================================

✅ 모든 테이블 마이그레이션 완료!
```

### 4. 데이터 확인

```bash
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB

# PostgreSQL에서
SELECT COUNT(*) FROM taigoods;
SELECT COUNT(*) FROM taipgmtape;
SELECT COUNT(*) FROM taibroadcasts;
```

### 5. 임베딩 생성

```bash
docker exec -it fastapi_backend python app/setup_product_embeddings.py
```

---

## 📅 n8n 워크플로우 설정

### 1. n8n 접속

```bash
# n8n 실행 확인
docker ps | grep n8n

# 브라우저에서
http://localhost:5678
```

### 2. 워크플로우 Import

1. **Workflows** → **Import from File**
2. 파일 선택: `/home/trn/trnAi/n8n_workflows/netezza_migration_workflow.json`
3. Import 클릭

### 3. 워크플로우 구조 확인

```
[매일 새벽 2시 실행]
    ↓
[데이터 마이그레이션 실행]
    ↓
[마이그레이션 성공?]
    ├─ 성공 → [임베딩 생성] → [성공 로그]
    └─ 실패 → [실패 로그]
```

### 4. 테스트 실행

1. 워크플로우 화면에서 **Execute Workflow** 버튼 클릭
2. 각 노드의 실행 결과 확인
3. 모두 초록색이면 성공!

### 5. 활성화

워크플로우 상단의 **Active** 토글 ON → 매일 자동 실행!

---

## 📋 테이블 추가 방법 (3분)

### 예시: 날씨 데이터 추가

**1. `backend/app/migrate_tables_config.py` 편집:**

```python
"TAIWEATHER_DAILY": {
    "enabled": True,  # ← 활성화
    "description": "일별 날씨 데이터",
    "primary_key": "weather_date",
    "incremental_column": "REG_DTTM",
    "query": lambda incremental: f"""
        SELECT 
               WEATHER_DATE AS weather_date,
               WEATHER_TYPE AS weather,
               TEMPERATURE AS temperature,
               PRECIPITATION AS precipitation,
               REG_DTTM AS created_at
          FROM SNTDM.SNTADM.WEATHER_DAILY
         WHERE 1=1
           {f"AND WEATHER_DATE >= '{get_yesterday()}'" if incremental else ""}
    """
},
```

**2. Docker 재시작:**

```bash
docker-compose restart fastapi_backend
```

**3. 테스트:**

```bash
# 새 테이블만 테스트
docker exec -it fastapi_backend bash -c "TABLES=TAIWEATHER_DAILY python app/migrate_netezza_to_postgres.py"
```

**4. n8n은 자동 반영!**

다음 실행부터 자동으로 날씨 데이터도 마이그레이션됩니다.

---

## 🔍 모니터링

### n8n 대시보드

- **Executions** 탭에서 실행 이력 확인
- 실패 시 stderr 로그 확인

### PostgreSQL 데이터 확인

```bash
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB -c "
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size('public.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE 'tai%'
ORDER BY pg_total_relation_size('public.'||tablename) DESC;
"
```

---

## 🎯 특수 실행 모드

### 전체 재처리

```bash
docker exec -it fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"
```

### 특정 테이블만

```bash
docker exec -it fastapi_backend bash -c "TABLES=TAIGOODS,TAIPGMTAPE python app/migrate_netezza_to_postgres.py"
```

### n8n에서 실행

n8n 워크플로우 노드의 Command 수정:
```bash
# 전체 재처리
docker exec -i fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"

# 특정 테이블
docker exec -i fastapi_backend bash -c "TABLES=TAIGOODS python app/migrate_netezza_to_postgres.py"
```

---

## 📚 상세 문서

- **N8N_MIGRATION_GUIDE.md** - n8n 워크플로우 상세 설정
- **NETEZZA_MIGRATION_README.md** - 마이그레이션 시스템 상세
- **migrate_tables_config.py** - 테이블 설정 파일

---

## ✅ 완료 체크리스트

- [ ] `.env` 파일에 NETEZZA 연결 정보 입력
- [ ] Docker 재빌드 완료
- [ ] 첫 전체 동기화 테스트 성공
- [ ] 데이터 확인 (PostgreSQL)
- [ ] 임베딩 생성 완료
- [ ] n8n 워크플로우 Import
- [ ] n8n 테스트 실행 성공
- [ ] n8n 워크플로우 활성화
- [ ] 다음날 자동 실행 확인

---

**🎉 완료! 이제 매일 새벽 2시에 자동으로 데이터가 동기화됩니다!**

**추가 테이블이 필요하면 `migrate_tables_config.py`만 수정하세요!**
