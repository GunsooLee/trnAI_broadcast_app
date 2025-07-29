from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from .schemas import RecommendRequest, RecommendResponse
from . import services
from . import broadcast_recommender as br # broadcast_recommender 임포트

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 모델을 비동기적으로 로드합니다.
    print("--- Loading model on startup... ---")
    model = await services.load_model_async()
    app.state.model = model
    print("--- Model loaded successfully. ---")
    yield
    # 애플리케이션 종료 시 정리 (필요 시)
    app.state.model = None

app = FastAPI(
    title="Home Shopping Broadcast Recommender API",
    description="An API to get broadcast schedule recommendations based on user queries.",
    version="1.0.0",
    lifespan=lifespan # lifespan 이벤트 핸들러 다시 활성화
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

@app.post("/api/v1/recommend", response_model=RecommendResponse)
async def recommend_broadcast(payload: RecommendRequest, request: Request):
    print("--- API Endpoint /api/v1/recommend received a request ---")
    """
    사용자 질문에 기반해 방송 편성을 추천합니다.
    - 시작 시 로드된 모델을 app.state에서 가져와 사용합니다.
    """
    try:
        # request.app.state에서 미리 로드된 모델을 가져옵니다.
        model = request.app.state.model
        response_data = await services.get_recommendations(payload.user_query, model)
        return response_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- ERROR IN /api/v1/recommend ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/v1/health", summary="Health Check")
def health_check():
    """API 서버의 상태를 확인합니다."""
    return {"status": "ok"}
