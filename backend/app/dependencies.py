import os
import sys
import joblib
from functools import lru_cache
from pathlib import Path

# tokenizer_utils 모듈을 찾을 수 있도록 경로 추가 (main.py와 동일)
sys.path.append(str(Path(__file__).parent))

from .product_embedder import ProductEmbedder

# --- 캐시된 의존성 함수들 ---

@lru_cache(maxsize=1)
def get_product_embedder() -> ProductEmbedder:
    """ProductEmbedder 싱글턴 인스턴스를 반환합니다."""
    print("--- Initializing ProductEmbedder... ---")
    return ProductEmbedder(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qdrant_host="qdrant_vector_db",
        qdrant_port=6333
    )

@lru_cache(maxsize=1)
def get_xgboost_model():
    """XGBoost 모델 싱글턴 인스턴스를 반환합니다."""
    print("--- Loading XGBoost model... ---")
    model_path = Path(__file__).parent / "xgb_broadcast_profit.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

@lru_cache(maxsize=1)
def get_broadcast_workflow():
    from .broadcast_workflow import BroadcastWorkflow # 순환 참조 방지를 위한 지역 임포트
    """BroadcastWorkflow 싱글턴 인스턴스를 반환합니다."""
    print("--- Initializing BroadcastWorkflow... ---")
    return BroadcastWorkflow(
        model=get_xgboost_model()
    )
