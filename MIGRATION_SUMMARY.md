# 📊 NETEZZA 마이그레이션 시스템 완성!

## ✅ 완료된 구조

### 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    NETEZZA (원본 DB)                         │
│  - FBD_PGMCPF_M (방송 프로그램)                              │
│  - DST_GDS_MST_EXT (상품 마스터)                             │
│  - DST_GDS_CLS (상품 분류)                                   │
│  - DST_BRND_MST (브랜드 마스터)                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 매일 새벽 2시
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            n8n 워크플로우 (자동 실행)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Schedule Trigger (매일 02:00)                     │   │
│  │ 2. 데이터 마이그레이션 (증분 업데이트)                │   │
│  │ 3. 성공/실패 체크                                    │   │
│  │ 4. 임베딩 생성 (성공 시)                             │   │
│  │ 5. 로그 기록                                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│    Python 마이그레이션 스크립트 (범용)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ migrate_tables_config.py                            │   │
│  │  - 테이블별 쿼리 정의                                │   │
│  │  - Primary Key, 증분 컬럼 설정                       │   │
│  │  - enabled True/False                               │   │
│  │                                                      │   │
│  │ migrate_netezza_to_postgres.py                      │   │
│  │  - 설정 파일 읽기                                    │   │
│  │  - NETEZZA 연결 & 데이터 추출                        │   │
│  │  - PostgreSQL UPSERT                                │   │
│  │  - 결과 리포팅                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL (타겟 DB)                            │
│  ┌───────────────────┬──────────────────┬────────────────┐  │
│  │ TAIGOODS          │ TAIPGMTAPE       │ TAIBROADCASTS  │  │
│  │ (상품 마스터)      │ (방송테이프)      │ (방송 이력)     │  │
│  ├───────────────────┼──────────────────┼────────────────┤  │
│  │ - product_code    │ - tape_code      │ - id           │  │
│  │ - product_name    │ - tape_name      │ - tape_code    │  │
│  │ - category_*      │ - product_code   │ - broadcast_*  │  │
│  │ - price           │ - status         │ - gross_profit │  │
│  │ - brand           │                  │                │  │
│  └───────────────────┴──────────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│     setup_product_embeddings.py (자동 실행)                  │
│  - 신규/수정 상품 임베딩 생성                                 │
│  - Qdrant 벡터 DB 저장                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               추천 시스템 (RAG + XGBoost)                    │
│  - 실시간 트렌드 기반 상품 검색                               │
│  - 매출 예측 및 방송 편성 추천                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 주요 파일 구조

```
/home/trn/trnAi/
│
├── backend/app/
│   ├── migrate_tables_config.py         ⭐ 테이블 설정 (여기서 테이블 추가/수정)
│   ├── migrate_netezza_to_postgres.py   ⭐ 범용 마이그레이션 스크립트
│   └── setup_product_embeddings.py      ⭐ 임베딩 생성
│
├── n8n_workflows/
│   └── netezza_migration_workflow.json  ⭐ n8n 워크플로우 정의
│
├── N8N_MIGRATION_GUIDE.md               📖 n8n 설정 상세 가이드
├── QUICK_START_N8N.md                   📖 빠른 시작 가이드
├── NETEZZA_MIGRATION_README.md          📖 마이그레이션 시스템 상세
└── MIGRATION_SUMMARY.md                 📖 이 문서 (요약)
```

---

## 🎯 핵심 기능

### 1. **범용 테이블 마이그레이션**

**테이블 추가는 설정 파일만 수정!**

```python
# backend/app/migrate_tables_config.py

MIGRATION_TABLES = {
    "테이블명": {
        "enabled": True,           # 활성화 여부
        "description": "설명",
        "primary_key": "PK컬럼",
        "incremental_column": "증분컬럼",
        "query": lambda incremental: f"""
            SELECT ... FROM ...
            WHERE ...
            {f"AND 증분조건" if incremental else ""}
        """
    },
}
```

**장점:**
- 코드 수정 불필요
- 테이블 추가/제거 간편
- 모든 테이블이 동일한 로직 사용

### 2. **증분 업데이트 (Incremental Update)**

매일 **어제 수정된 데이터만** 가져옴:
```python
# 자동 계산
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

# 쿼리에 자동 추가
WHERE DGM.REG_DTTM >= '20251012'  # 어제 날짜
```

**장점:**
- 빠른 실행 시간
- 네트워크 부하 최소화
- 전체 재처리도 가능 (`FULL_SYNC=true`)

### 3. **UPSERT 방식**

중복 데이터 자동 업데이트:
```sql
INSERT INTO taigoods (...)
VALUES (...)
ON CONFLICT (product_code) DO UPDATE SET
    product_name = EXCLUDED.product_name,
    price = EXCLUDED.price,
    updated_at = CURRENT_TIMESTAMP
```

**장점:**
- 중복 없음
- 자동 업데이트
- 데이터 무결성 보장

### 4. **n8n 자동화**

```
매일 새벽 2:00 → 마이그레이션 → 성공 시 임베딩 생성 → 로그 기록
```

**장점:**
- 크론보다 관리 편함
- 시각적 워크플로우
- 실패 시 자동 알림 (추가 가능)
- 실행 이력 추적

---

## 🚀 사용 방법

### 일반 실행 (매일 자동)

**n8n 워크플로우가 알아서 실행**
- 시간: 매일 새벽 2:00
- 모드: 증분 업데이트 (어제 수정분)
- 후속: 임베딩 자동 생성

### 수동 실행

**전체 재처리:**
```bash
docker exec -it fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"
```

**특정 테이블만:**
```bash
docker exec -it fastapi_backend bash -c "TABLES=TAIGOODS python app/migrate_netezza_to_postgres.py"
```

**증분 업데이트 (기본값):**
```bash
docker exec -it fastapi_backend python app/migrate_netezza_to_postgres.py
```

---

## 📊 현재 마이그레이션 테이블

| 테이블 | 상태 | 설명 | 레코드 수 (예시) |
|-------|------|------|-----------------|
| **TAIGOODS** | ✅ 활성 | 상품 마스터 | ~1,234건 |
| **TAIPGMTAPE** | ✅ 활성 | 방송테이프 | ~567건 |
| **TAIBROADCASTS** | ✅ 활성 | 방송 이력 | ~766건 |
| TAIHOLIDAYS | ⏸️ 비활성 | 공휴일 (수동 업데이트) | - |
| TAIWEATHER | ⏸️ 비활성 | 날씨 (추후 구현) | - |
| TAITRENDS | ⏸️ 비활성 | 트렌드 (n8n 별도 수집) | - |

---

## 🔧 테이블 추가 3단계

### 1단계: 설정 추가

`backend/app/migrate_tables_config.py`에 추가:
```python
"TAINEWDATA": {
    "enabled": True,
    "description": "새 데이터",
    "primary_key": "data_id",
    "incremental_column": "updated_at",
    "query": lambda incremental: f"""
        SELECT * FROM NEW_TABLE
        WHERE 1=1
        {f"AND updated_at >= '{get_yesterday()}'" if incremental else ""}
    """
},
```

### 2단계: Docker 재시작

```bash
docker-compose restart fastapi_backend
```

### 3단계: 테스트

```bash
docker exec -it fastapi_backend bash -c "TABLES=TAINEWDATA python app/migrate_netezza_to_postgres.py"
```

**끝! n8n 워크플로우는 자동 반영됩니다!**

---

## 📈 모니터링

### n8n 대시보드

```
http://localhost:5678
→ Executions 탭
→ 실행 이력 확인
```

### PostgreSQL 확인

```bash
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB -c "
SELECT tablename, pg_size_pretty(pg_total_relation_size('public.'||tablename))
FROM pg_tables WHERE tablename LIKE 'tai%';
"
```

### 로그 확인

```bash
# n8n 워크플로우 로그
docker logs n8n_container -f

# FastAPI 로그
docker logs fastapi_backend -f
```

---

## 🎓 주요 개념

### 증분 업데이트 (Incremental)
- **매일:** 어제 이후 수정된 데이터만
- **빠름:** 소량 데이터만 전송
- **안전:** 전체 재처리도 가능

### UPSERT
- **INSERT:** 새 데이터 삽입
- **UPDATE:** 중복 시 업데이트
- **자동:** ON CONFLICT 활용

### 범용 시스템
- **설정 파일 기반:** 코드 수정 불필요
- **확장 가능:** 테이블 무한 추가
- **유지보수 편함:** 한 곳에서 관리

---

## 🐛 트러블슈팅

### 문제: 데이터 0건

**원인:**
- 날짜 필터가 너무 좁음
- 테이블명 오타
- 조인 조건 오류

**해결:**
```bash
# 전체 재처리로 확인
FULL_SYNC=true python app/migrate_netezza_to_postgres.py

# 쿼리 직접 실행 (NETEZZA)
SELECT COUNT(*) FROM SNTDM.SNTADM.DST_GDS_MST_EXT;
```

### 문제: UPSERT 실패

**원인:**
- Primary Key 중복
- 컬럼명 불일치
- 데이터 타입 오류

**해결:**
```python
# migrate_tables_config.py에서
"primary_key": "product_code",  # ← 확인!
```

### 문제: n8n 실행 안 됨

**원인:**
- Docker 명령 권한 없음
- 컨테이너명 불일치

**해결:**
```bash
# 컨테이너명 확인
docker ps | grep fastapi

# n8n 노드 수정
Command: docker exec -i 실제_컨테이너명 python app/...
```

---

## ✅ 최종 체크리스트

- [x] 범용 마이그레이션 스크립트 작성
- [x] 테이블 설정 파일 분리
- [x] UPSERT 로직 구현
- [x] 증분 업데이트 구현
- [x] n8n 워크플로우 생성
- [x] 자동 임베딩 생성 연동
- [ ] `.env` 파일 실제 값 입력
- [ ] 첫 전체 동기화 테스트
- [ ] n8n 워크플로우 Import
- [ ] n8n 활성화
- [ ] 다음날 자동 실행 확인

---

## 📚 다음 단계

### 1. 추가 테이블 마이그레이션

- [ ] TAIHOLIDAYS (공휴일)
- [ ] TAIWEATHER_DAILY (날씨)
- [ ] TAICOMPETITOR_BROADCASTS (경쟁사 방송)

### 2. 트렌드 수집 자동화

별도 n8n 워크플로우:
- 네이버 DataLab API
- Google Trends
- TAITRENDS 테이블 저장

### 3. XGBoost 재학습 자동화

주 1회 또는 월 1회 자동 재학습

### 4. 알림 시스템

- Slack 알림
- 이메일 알림
- 대시보드 구축

---

## 🎉 완성!

**이제 매일 새벽 2시에 자동으로:**
1. NETEZZA에서 어제 수정된 데이터 가져오기
2. PostgreSQL에 UPSERT
3. 신규/수정 상품 임베딩 생성
4. 추천 시스템에서 사용 가능!

**테이블 추가는 설정 파일만 수정하면 됩니다!**

---

**작성일:** 2025-10-13  
**작성자:** Cascade AI  
**버전:** 3.0 (n8n 자동화 완성)
