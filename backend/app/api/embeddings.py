"""
상품 임베딩 생성 API 엔드포인트
n8n에서 HTTP 요청으로 호출 가능
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import logging

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    force_all: bool = False  # True면 전체 재생성, False면 신규/수정분만
    batch_size: int = 100  # 배치 크기


class EmbeddingResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None


@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks
):
    """
    상품 임베딩 생성
    
    - **force_all**: True면 전체 재생성, False면 신규/수정분만 (기본값)
    - **batch_size**: 배치 크기 (기본 100)
    
    백그라운드에서 실행되며, 즉시 응답을 반환합니다.
    """
    
    try:
        # 환경변수 설정
        env = os.environ.copy()
        
        if request.force_all:
            env['FORCE_ALL'] = 'true'
        
        env['BATCH_SIZE'] = str(request.batch_size)
        
        # 백그라운드에서 임베딩 생성 실행
        background_tasks.add_task(
            run_embedding_task,
            env
        )
        
        return EmbeddingResponse(
            status="started",
            message="임베딩 생성이 백그라운드에서 시작되었습니다.",
            job_id="embedding_job_001"
        )
        
    except Exception as e:
        logger.error(f"임베딩 생성 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-sync", response_model=EmbeddingResponse)
async def generate_embeddings_sync(request: EmbeddingRequest):
    """
    상품 임베딩 생성 (동기 실행)
    
    완료될 때까지 대기하고 결과를 반환합니다.
    n8n에서 결과를 즉시 확인할 때 사용하세요.
    """
    
    try:
        # 환경변수 설정
        env = os.environ.copy()
        
        if request.force_all:
            env['FORCE_ALL'] = 'true'
        
        env['BATCH_SIZE'] = str(request.batch_size)
        
        # 임베딩 스크립트 실행
        script_path = "/app/app/setup_product_embeddings.py"
        
        result = subprocess.run(
            ["python", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30분 타임아웃
        )
        
        if result.returncode == 0:
            # stdout에서 임베딩된 개수 추출 시도
            output_lines = result.stdout.strip().split('\n')
            embedded_count = "알 수 없음"
            
            for line in output_lines:
                if "임베딩 완료" in line or "embedded" in line.lower():
                    embedded_count = line
                    break
            
            return EmbeddingResponse(
                status="success",
                message=f"임베딩 생성 완료: {embedded_count}",
                job_id="embedding_sync_001"
            )
        else:
            return EmbeddingResponse(
                status="failed",
                message=f"임베딩 생성 실패: {result.stderr}",
                job_id="embedding_sync_001"
            )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="임베딩 생성 타임아웃 (30분 초과)")
    except Exception as e:
        logger.error(f"임베딩 생성 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_embedding_task(env: dict):
    """백그라운드 임베딩 생성 태스크"""
    try:
        script_path = "/app/app/setup_product_embeddings.py"
        
        result = subprocess.run(
            ["python", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode == 0:
            logger.info("✅ 백그라운드 임베딩 생성 성공")
            logger.info(result.stdout)
        else:
            logger.error("❌ 백그라운드 임베딩 생성 실패")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"백그라운드 임베딩 생성 오류: {e}")


@router.get("/status")
async def get_embedding_status():
    """임베딩 상태 확인"""
    try:
        from sqlalchemy import create_engine, text
        
        # PostgreSQL 연결 (setup_product_embeddings.py와 동일한 환경변수 사용)
        postgres_url = os.getenv("DB_URI")
        if not postgres_url:
            # Fallback: docker-compose.yml의 설정 사용
            postgres_url = "postgresql://TRN_AI:TRN_AI@postgres:5432/TRNAI_DB"
        
        engine = create_engine(postgres_url)
        
        # 통계 조회
        with engine.connect() as conn:
            # 전체 상품 수
            total_result = conn.execute(text("SELECT COUNT(*) FROM taigoods"))
            total_count = total_result.scalar()
            
            # 임베딩된 상품 수
            embedded_result = conn.execute(text("SELECT COUNT(*) FROM taigoods WHERE embedded_at IS NOT NULL"))
            embedded_count = embedded_result.scalar()
            
            # 방송테이프 있는 상품 수
            tape_result = conn.execute(text("""
                SELECT COUNT(DISTINCT g.product_code) 
                FROM taigoods g
                INNER JOIN taipgmtape t ON g.product_code = t.product_code
                WHERE t.production_status = 'ready'
            """))
            tape_count = tape_result.scalar()
            
            # 임베딩 필요 (방송테이프 있지만 임베딩 안된 상품)
            need_embedding_result = conn.execute(text("""
                SELECT COUNT(DISTINCT g.product_code) 
                FROM taigoods g
                INNER JOIN taipgmtape t ON g.product_code = t.product_code
                WHERE t.production_status = 'ready'
                AND (g.embedded_at IS NULL OR g.embedded_at < g.updated_at)
            """))
            need_embedding_count = need_embedding_result.scalar()
        
        return {
            "status": "ok",
            "total_products": total_count,
            "embedded_products": embedded_count,
            "products_with_tape": tape_count,
            "need_embedding": need_embedding_count,
            "embedding_rate": f"{(embedded_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
        }
        
    except Exception as e:
        logger.error(f"임베딩 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
