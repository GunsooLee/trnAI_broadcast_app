"""
NETEZZA → PostgreSQL 마이그레이션 API 엔드포인트
n8n에서 HTTP 요청으로 호출 가능
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import logging

router = APIRouter(prefix="/api/v1/migration", tags=["migration"])
logger = logging.getLogger(__name__)


class MigrationRequest(BaseModel):
    full_sync: bool = False  # True면 전체 동기화, False면 증분 업데이트
    tables: Optional[str] = None  # 특정 테이블만 (예: "TAIGOODS,TAIPGMTAPE")


class MigrationResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None
    total_rows: Optional[int] = None  # 총 마이그레이션된 row 수
    tables: Optional[dict] = None  # 테이블별 상세 정보


@router.post("/start", response_model=MigrationResponse)
async def start_migration(
    request: MigrationRequest,
    background_tasks: BackgroundTasks
):
    """
    NETEZZA → PostgreSQL 마이그레이션 시작
    
    - **full_sync**: True면 전체 재처리, False면 증분 업데이트 (기본값)
    - **tables**: 특정 테이블만 마이그레이션 (예: "TAIGOODS,TAIPGMTAPE")
    
    백그라운드에서 실행되며, 즉시 응답을 반환합니다.
    """
    
    try:
        # 환경변수 설정
        env = os.environ.copy()
        
        if request.full_sync:
            env['FULL_SYNC'] = 'true'
        
        if request.tables:
            env['TABLES'] = request.tables
        
        # 백그라운드에서 마이그레이션 실행
        background_tasks.add_task(
            run_migration_task,
            env
        )
        
        return MigrationResponse(
            status="started",
            message="마이그레이션이 백그라운드에서 시작되었습니다.",
            job_id="migration_job_001"
        )
        
    except Exception as e:
        logger.error(f"마이그레이션 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-sync", response_model=MigrationResponse)
async def start_migration_sync(request: MigrationRequest):
    """
    NETEZZA → PostgreSQL 마이그레이션 시작 (동기 실행)
    
    완료될 때까지 대기하고 결과를 반환합니다.
    n8n에서 결과를 즉시 확인할 때 사용하세요.
    """
    
    try:
        # 환경변수 설정
        env = os.environ.copy()
        
        if request.full_sync:
            env['FULL_SYNC'] = 'true'
        
        if request.tables:
            env['TABLES'] = request.tables
        
        # 마이그레이션 스크립트 실행
        script_path = "/app/app/migrate_netezza_to_postgres.py"
        
        result = subprocess.run(
            ["python", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )
        
        if result.returncode == 0:
            # stdout에서 결과 파싱
            total_rows, tables_info = parse_migration_output(result.stdout)
            
            return MigrationResponse(
                status="success",
                message="마이그레이션 완료",
                job_id="migration_sync_001",
                total_rows=total_rows,
                tables=tables_info
            )
        else:
            return MigrationResponse(
                status="failed",
                message=f"마이그레이션 실패: {result.stderr}",
                job_id="migration_sync_001"
            )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="마이그레이션 타임아웃 (10분 초과)")
    except Exception as e:
        logger.error(f"마이그레이션 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_migration_output(output: str) -> tuple:
    """마이그레이션 스크립트 출력에서 row 수 추출
    
    Returns:
        tuple: (total_rows, tables_info)
    """
    import re
    
    total_rows = 0
    tables_info = {}
    
    try:
        # "총 레코드: 1,234개" 패턴 찾기
        total_match = re.search(r'총 레코드:\s*([\d,]+)개', output)
        if total_match:
            total_rows = int(total_match.group(1).replace(',', ''))
        
        # 각 테이블별 레코드 수 추출
        # 패턴: "✅ TAIGOODS                   1,234개  (완료)"
        table_pattern = r'[✅❌]\s+(\w+)\s+([\d,]+)개\s+\((.+?)\)'
        for match in re.finditer(table_pattern, output):
            table_name = match.group(1)
            row_count = int(match.group(2).replace(',', ''))
            status = match.group(3)
            
            tables_info[table_name] = {
                "rows": row_count,
                "status": status
            }
    
    except Exception as e:
        logger.warning(f"출력 파싱 실패: {e}")
    
    return total_rows, tables_info


def run_migration_task(env: dict):
    """백그라운드 마이그레이션 태스크"""
    try:
        script_path = "/app/app/migrate_netezza_to_postgres.py"
        
        result = subprocess.run(
            ["python", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            logger.info("✅ 백그라운드 마이그레이션 성공")
            logger.info(result.stdout)
        else:
            logger.error("❌ 백그라운드 마이그레이션 실패")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"백그라운드 마이그레이션 오류: {e}")


@router.get("/status")
async def get_migration_status():
    """마이그레이션 상태 확인 (추후 구현)"""
    return {
        "status": "not_implemented",
        "message": "상태 조회 기능은 추후 구현 예정입니다."
    }
