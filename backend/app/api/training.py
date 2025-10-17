"""
XGBoost 모델 학습 API 엔드포인트
n8n에서 HTTP 요청으로 호출 가능
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import logging

router = APIRouter(prefix="/api/v1/training", tags=["training"])
logger = logging.getLogger(__name__)


class TrainingRequest(BaseModel):
    async_mode: bool = True  # True면 백그라운드 실행, False면 동기 실행


class TrainingResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None
    models_updated: Optional[dict] = None


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    XGBoost 모델 학습 시작 (비동기)
    
    - **async_mode**: True면 백그라운드 실행 (기본값), False면 동기 실행
    
    백그라운드에서 실행되며, 즉시 응답을 반환합니다.
    학습 완료까지 시간이 걸릴 수 있습니다 (데이터 양에 따라 다름).
    """
    
    try:
        if request.async_mode:
            # 백그라운드에서 학습 실행
            background_tasks.add_task(run_training_task)
            
            return TrainingResponse(
                status="started",
                message="XGBoost 모델 학습이 백그라운드에서 시작되었습니다.",
                job_id="training_job_001"
            )
        else:
            # 동기 실행
            result = run_training_sync()
            return result
        
    except Exception as e:
        logger.error(f"모델 학습 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-sync", response_model=TrainingResponse)
async def start_training_sync(request: TrainingRequest):
    """
    XGBoost 모델 학습 시작 (동기 실행)
    
    완료될 때까지 대기하고 결과를 반환합니다.
    n8n에서 결과를 즉시 확인할 때 사용하세요.
    """
    
    try:
        result = run_training_sync()
        return result
        
    except Exception as e:
        logger.error(f"모델 학습 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_training_sync():
    """동기 모델 학습 실행"""
    try:
        # Python 스크립트를 직접 실행하여 결과 받기
        import sys
        sys.path.insert(0, '/app')
        from train import train
        
        # 학습 실행 및 통계 수집
        training_stats = train()
        
        logger.info("✅ XGBoost 모델 학습 성공")
        logger.info(f"학습 데이터: {training_stats['total_records']}건")
        logger.info(f"소요 시간: {training_stats['training_time_seconds']}초")
        
        # 모델 파일 업데이트 시간 확인
        models_info = get_models_info()
        
        # 학습 통계와 모델 정보 결합
        models_info["profit_model"].update(training_stats["models"]["profit_model"])
        models_info["efficiency_model"].update(training_stats["models"]["efficiency_model"])
        
        return TrainingResponse(
            status="success",
            message=f"XGBoost 모델 학습 완료 ({training_stats['total_records']}건, {training_stats['training_time_seconds']}초)",
            job_id="training_sync_001",
            models_updated={
                "total_records": training_stats["total_records"],
                "training_time_seconds": training_stats["training_time_seconds"],
                "models": models_info
            }
        )
        
    except ImportError as e:
        logger.error(f"모듈 import 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모듈 import 실패: {str(e)}")
    except Exception as e:
        logger.error(f"모델 학습 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_training_task():
    """백그라운드 모델 학습 태스크"""
    try:
        script_path = "/app/train.py"
        
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode == 0:
            logger.info("✅ 백그라운드 모델 학습 성공")
            logger.info(result.stdout)
        else:
            logger.error("❌ 백그라운드 모델 학습 실패")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"백그라운드 모델 학습 오류: {e}")


def get_models_info():
    """현재 모델 파일 정보 반환"""
    model_profit_path = "/app/app/xgb_broadcast_profit.joblib"
    model_efficiency_path = "/app/app/xgb_broadcast_efficiency.joblib"
    
    profit_exists = os.path.exists(model_profit_path)
    efficiency_exists = os.path.exists(model_efficiency_path)
    
    models_info = {}
    
    if profit_exists:
        from datetime import datetime
        profit_mtime = os.path.getmtime(model_profit_path)
        models_info["profit_model"] = {
            "exists": True,
            "last_updated": datetime.fromtimestamp(profit_mtime).isoformat()
        }
    else:
        models_info["profit_model"] = {"exists": False}
    
    if efficiency_exists:
        from datetime import datetime
        efficiency_mtime = os.path.getmtime(model_efficiency_path)
        models_info["efficiency_model"] = {
            "exists": True,
            "last_updated": datetime.fromtimestamp(efficiency_mtime).isoformat()
        }
    else:
        models_info["efficiency_model"] = {"exists": False}
    
    return models_info


@router.get("/status")
async def get_training_status():
    """모델 학습 상태 및 모델 파일 정보 확인"""
    
    models_info = get_models_info()
    
    profit_exists = models_info.get("profit_model", {}).get("exists", False)
    efficiency_exists = models_info.get("efficiency_model", {}).get("exists", False)
    
    if profit_exists and efficiency_exists:
        return {
            "status": "models_exist",
            "message": "XGBoost 모델이 존재합니다",
            "models": models_info
        }
    else:
        return {
            "status": "models_missing",
            "message": "일부 또는 전체 모델이 없습니다. 학습이 필요합니다.",
            "models": models_info
        }
