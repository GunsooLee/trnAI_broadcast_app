import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 모델 로드 시 'tokenizer_utils' 모듈을 찾을 수 있도록 경로 추가
# train.py와 동일한 로직을 사용하여 app 폴더를 경로에 추가합니다.
sys.path.append(str(Path(__file__).parent))
print("--- sys.path updated ---")
import pprint
pprint.pprint(sys.path)

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

from .schemas import BroadcastRequest, BroadcastResponse, TapeCollectionResponse, BroadcastTapeInfo
from .broadcast_workflow import BroadcastWorkflow
from .dependencies import get_broadcast_workflow
from .netezza_config import netezza_conn

app = FastAPI(
    title="Home Shopping Broadcast Recommender API",
    description="An API to get broadcast schedule recommendations based on user queries.",
    version="1.0.0"
)

# CORS 설정: Next.js 프론트엔드(기본 포트 3000)에서의 요청을 허용
origins = [
    "http://localhost:3001",  # Next.js 프론트엔드 새 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# 🎯 핵심 API 엔드포인트
# ========================================

@app.post("/api/v1/broadcast/recommendations", response_model=BroadcastResponse)
async def broadcast_recommendations(payload: BroadcastRequest, workflow: BroadcastWorkflow = Depends(get_broadcast_workflow)):
    """🚀 메인 방송 편성 AI 추천 API - LangChain 기반 2단계 워크플로우
    
    실시간 트렌드 분석 + XGBoost 매출 예측 + 방송테이프 필터링을 통한
    최적의 홈쇼핑 방송 편성을 추천합니다.
    """
    print(f"--- API Endpoint /api/v1/broadcast/recommendations received a request: {payload.broadcastTime} ---")
    print(f"--- 가중치 설정: 트렌드 {payload.trendWeight:.0%} / 매출 {payload.salesWeight:.0%} ---")
    try:
        response_data = await workflow.process_broadcast_recommendation(
            payload.broadcastTime,
            payload.recommendationCount,
            payload.trendWeight,
            payload.salesWeight
        )

        # 빈 추천 결과 체크
        has_recommendations = response_data.recommendations and len(response_data.recommendations) > 0
        has_categories = response_data.recommendedCategories and len(response_data.recommendedCategories) > 0

        if not has_recommendations and not has_categories:
            print(f"--- 빈 결과 감지: recommendations={len(response_data.recommendations) if response_data.recommendations else 0}, categories={len(response_data.recommendedCategories) if response_data.recommendedCategories else 0} ---")
            raise HTTPException(
                status_code=503,
                detail="추천 결과를 생성할 수 없습니다. AI 서비스가 일시적으로 이용 불가능합니다."
            )

        return response_data

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"잘못된 요청 데이터: {str(e)}")
    except Exception as e:
        print(f"--- ERROR IN /api/v1/broadcast/recommendations ---")
        import traceback
        traceback.print_exc()

        # OpenAI API 관련 오류는 503으로 처리
        if any(keyword in str(e) for keyword in ["AI 서비스", "OpenAI", "할당량", "insufficient_quota", "429"]):
            raise HTTPException(
                status_code=503,
                detail="AI 서비스 일시 중단 - 잠시 후 다시 시도해주세요."
            )

        # 기타 내부 오류는 500으로 처리
        raise HTTPException(status_code=500, detail="내부 서버 오류가 발생했습니다.")

@app.get("/api/v1/health", summary="Health Check")
def health_check():
    """API 서버의 상태를 확인합니다."""
    return {"status": "ok"}

# ========================================
# 🎬 방송테이프 수집 API 엔드포인트
# ========================================

@app.post("/api/v1/tapes/sync", response_model=TapeCollectionResponse)
async def sync_broadcast_tapes():
    """📺 Netezza DB에서 방송테이프 정보를 동기화합니다 (Upsert 방식)

    n8n에서 주기적으로 호출하여 방송테이프 정보를 PostgreSQL DB에 동기화합니다.
    """
    from datetime import datetime

    print(f"--- API Endpoint /api/v1/tapes/sync received request ---")

    try:
        collection_timestamp = datetime.now().isoformat()

        # Netezza에서 모든 방송테이프 정보 가져오기
        print(f"--- Fetching all broadcast tapes from Netezza ---")
        raw_tapes = await netezza_conn.get_all_broadcast_tapes()

        # 결과를 스키마에 맞게 변환
        tapes = []
        for tape_data in raw_tapes:
            tape_info = BroadcastTapeInfo(
                tape_id=str(tape_data.get('tape_id', '')),
                product_code=str(tape_data.get('product_code', '')),
                product_name=str(tape_data.get('product_name', '')),
                category=str(tape_data.get('category', '')),
                broadcast_date=str(tape_data.get('broadcast_date', '')),
                broadcast_time=str(tape_data.get('broadcast_time', '')) if tape_data.get('broadcast_time') else None,
                duration_minutes=tape_data.get('duration_minutes'),
                status=str(tape_data.get('status', '')),
                created_at=str(tape_data.get('created_at', '')),
                updated_at=str(tape_data.get('updated_at', ''))
            )
            tapes.append(tape_info)

        # PostgreSQL에 Upsert 수행
        upserted_count = await netezza_conn.upsert_tapes_to_postgres(tapes)

        response = TapeCollectionResponse(
            tapes=tapes,
            collection_timestamp=collection_timestamp,
            total_count=len(tapes),
            upserted_count=upserted_count
        )

        print(f"--- Successfully synced {len(tapes)} tapes, upserted {upserted_count} records ---")
        return response

    except Exception as e:
        print(f"--- ERROR IN /api/v1/tapes/sync ---")
        import traceback
        traceback.print_exc()

        # Netezza 연결 오류는 503으로 처리
        if "Netezza" in str(e) or "connection" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail=f"Netezza 데이터베이스 연결 오류: {str(e)}"
            )

        # 기타 내부 오류는 500으로 처리
        raise HTTPException(
            status_code=500,
            detail=f"방송테이프 동기화 중 오류가 발생했습니다: {str(e)}"
        )


# ========================================
# 🔄 데이터 마이그레이션 API (n8n 연동용)
# ========================================
from .api.migration import router as migration_router
app.include_router(migration_router)

# ========================================
# 🎨 상품 임베딩 생성 API (n8n 연동용)
# ========================================
from .api.embeddings import router as embeddings_router
app.include_router(embeddings_router)

# ========================================
# 🤖 XGBoost 모델 학습 API (n8n 연동용)
# ========================================
from .api.training import router as training_router
app.include_router(training_router)
