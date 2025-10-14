# 🔄 n8n 배치 워크플로우 설정 가이드

## 📋 개요

NETEZZA → PostgreSQL 데이터 마이그레이션을 **n8n 워크플로우**로 자동화합니다.

---

## 🎯 구조

### 범용 마이그레이션 시스템

```
migrate_tables_config.py  ← 테이블 설정 (쿼리, 키 정의)
        ↓
migrate_netezza_to_postgres.py  ← 범용 실행 스크립트
        ↓
n8n 워크플로우  ← 스케줄링 및 자동화
        ↓
PostgreSQL + 임베딩 생성
```

### 테이블 추가/수정 방법

**`backend/app/migrate_tables_config.py`** 파일에서:

```python
MIGRATION_TABLES = {
    
    # 새 테이블 추가
    "TAINEWDATA": {
        "enabled": True,  # True면 자동 마이그레이션
        "description": "새로운 데이터 테이블",
        "primary_key": "data_id",
        "incremental_column": "updated_at",
        "query": lambda incremental: f"""
            SELECT 
                   data_id,
                   data_name,
                   created_at
              FROM SNTDM.SNTADM.NEW_DATA_TABLE
             WHERE 1=1
               {f"AND updated_at >= '{get_yesterday()}'" if incremental else ""}
        """
    },
    
}
```

**설정 항목:**
- `enabled`: True/False (활성화 여부)
- `description`: 테이블 설명
- `primary_key`: Primary Key 컬럼명 (UPSERT용)
- `incremental_column`: 증분 업데이트 기준 컬럼
- `query`: 람다 함수 (incremental 파라미터 받음)

---

## 🚀 n8n 워크플로우 설정

### 1단계: n8n 접속

```bash
# n8n 실행 확인
docker ps | grep n8n

# 접속
http://localhost:5678
```

### 2단계: 워크플로우 Import

1. n8n 대시보드 접속
2. **Workflows** → **Import from File** 클릭
3. `/home/trn/trnAi/n8n_workflows/netezza_migration_workflow.json` 업로드
4. 워크플로우 이름 확인: "NETEZZA → PostgreSQL 매일 데이터 마이그레이션"

### 3단계: 노드 설정 확인

#### 📅 Schedule Trigger (매일 새벽 2시)
```
Cron Expression: 0 2 * * *
Timezone: Asia/Seoul
```

#### 🐳 데이터 마이그레이션 실행
```bash
Command: docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py
Working Directory: /home/trn/trnAi
```

#### 🔍 마이그레이션 성공?
```
Condition: {{$json.exitCode}} equals 0
```

#### 🎯 임베딩 생성 (성공 시)
```bash
Command: docker exec -i fastapi_backend python app/setup_product_embeddings.py
Working Directory: /home/trn/trnAi
```

### 4단계: 테스트 실행

1. 워크플로우 화면에서 **Execute Workflow** 클릭
2. 실행 결과 확인:
   - ✅ 모든 노드가 초록색이면 성공
   - ❌ 빨간색 노드가 있으면 로그 확인

### 5단계: 활성화

1. 워크플로우 상단의 **Active** 토글 ON
2. 매일 새벽 2시에 자동 실행됨

---

## 🔧 환경변수 설정

### n8n 노드에서 환경변수 전달

**전체 재처리:**
```bash
docker exec -i fastapi_backend bash -c "FULL_SYNC=true python app/migrate_netezza_to_postgres.py"
```

**특정 테이블만:**
```bash
docker exec -i fastapi_backend bash -c "TABLES=TAIGOODS,TAIPGMTAPE python app/migrate_netezza_to_postgres.py"
```

### n8n 노드 수정 방법

1. "데이터 마이그레이션 실행" 노드 더블클릭
2. Command 필드 수정
3. Save 클릭

---

## 📊 현재 마이그레이션 대상 테이블

| 테이블명 | 활성화 | 설명 | 증분 기준 |
|---------|--------|------|-----------|
| **TAIGOODS** | ✅ | 상품 마스터 | DGM.REG_DTTM |
| **TAIPGMTAPE** | ✅ | 방송테이프 정보 | FPM.STRD_YMD |
| **TAIBROADCASTS** | ✅ | 방송 이력 | FPM.STRD_YMD |
| **TAIHOLIDAYS** | ❌ | 공휴일 정보 | (수동 업데이트) |
| **TAICOMPETITOR_BROADCASTS** | ❌ | 경쟁사 방송 | (추후 구현) |
| **TAITRENDS** | ❌ | 트렌드 데이터 | (n8n 별도 수집) |

---

## 📝 테이블 추가 가이드

### 예시: 날씨 데이터 테이블 추가

#### 1. `migrate_tables_config.py`에 추가

```python
"TAIWEATHER_DAILY": {
    "enabled": True,
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
         ORDER BY WEATHER_DATE
    """
},
```

#### 2. PostgreSQL 테이블 존재 확인

```sql
-- 이미 init_database.sql에 정의되어 있음
CREATE TABLE IF NOT EXISTS taiweather_daily (
    weather_date DATE PRIMARY KEY,
    weather VARCHAR(50),
    temperature DECIMAL(5,2),
    precipitation DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. 테스트

```bash
# Docker 재빌드 (설정 파일 업데이트)
docker-compose restart fastapi_backend

# 수동 실행
docker exec -it fastapi_backend bash -c "TABLES=TAIWEATHER_DAILY python app/migrate_netezza_to_postgres.py"
```

#### 4. n8n 워크플로우는 자동 반영

설정 파일만 수정하면 n8n 워크플로우는 수정 없이 자동으로 새 테이블 마이그레이션!

---

## 📈 모니터링

### n8n 실행 이력 확인

1. n8n 대시보드 → **Executions** 탭
2. 각 실행 클릭하여 상세 로그 확인
3. 실패 시 stderr 확인

### 로그 파일 (선택사항)

```bash
# 워크플로우 로그 저장 설정 (n8n 노드에 추가)
Command: docker exec -i fastapi_backend python app/migrate_netezza_to_postgres.py >> /var/log/migration.log 2>&1

# 로그 확인
tail -f /var/log/migration.log
```

### PostgreSQL 확인

```bash
docker exec -it trnAi_postgres psql -U TRN_AI -d TRNAI_DB -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE 'tai%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

---

## 🔔 알림 추가 (선택사항)

### Slack 알림 노드 추가

1. n8n 워크플로우 편집
2. "성공 로그 처리" 또는 "실패 로그 처리" 뒤에 **Slack** 노드 추가
3. Webhook URL 설정
4. 메시지 템플릿:

```json
{
  "text": "{{$json.status === 'SUCCESS' ? '✅' : '❌'}} 데이터 마이그레이션 {{$json.status}}",
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*시간:* {{$json.timestamp}}\n*상태:* {{$json.status}}\n*메시지:* {{$json.message}}"
      }
    }
  ]
}
```

### 이메일 알림 노드 추가

1. **Send Email** 노드 추가
2. SMTP 설정 (Gmail, AWS SES 등)
3. 실패 시에만 알림 전송

---

## 🐛 트러블슈팅

### 문제 1: n8n에서 Docker 명령 실행 안 됨

```
Error: docker: command not found
```

**해결:**
```bash
# n8n 컨테이너에 Docker 설치 또는
# 호스트에서 n8n 실행 (Docker 없이)
```

### 문제 2: 권한 오류

```
Error: Permission denied
```

**해결:**
```bash
# n8n 실행 사용자에게 Docker 실행 권한 부여
sudo usermod -aG docker n8n_user
```

### 문제 3: 테이블 추가했는데 마이그레이션 안 됨

**확인:**
1. `enabled: True`로 설정했는지
2. Docker 재시작했는지
3. 쿼리 문법 오류 없는지

---

## ✅ 완료 체크리스트

- [ ] `migrate_tables_config.py`에 필요한 테이블 모두 추가
- [ ] 각 테이블의 쿼리 테스트 완료
- [ ] n8n 워크플로우 Import 완료
- [ ] 테스트 실행 성공
- [ ] 워크플로우 활성화 (Active)
- [ ] 다음날 자동 실행 확인
- [ ] 모니터링 대시보드 확인
- [ ] (선택) Slack/이메일 알림 설정

---

## 🎯 다음 단계

### 1. 다른 테이블 추가

`migrate_tables_config.py`에서 다음 테이블 활성화:
- TAIBROADCASTS (방송 이력)
- TAIHOLIDAYS (공휴일)
- TAIWEATHER_DAILY (날씨)

### 2. 트렌드 데이터 수집

별도 n8n 워크플로우 생성:
- 네이버 DataLab API 호출
- Google Trends 수집
- TAITRENDS 테이블에 저장

### 3. XGBoost 재학습 자동화

주 1회 또는 월 1회 자동 재학습 워크플로우 추가

---

**작성일:** 2025-10-13  
**버전:** 2.0 (n8n 워크플로우 기반)
