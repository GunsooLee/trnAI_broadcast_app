from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from .schemas import RecommendRequest, RecommendResponse, CandidatesResponse, TrendCollectionResponse, TrendAnalysisResponse, BroadcastRequest, BroadcastResponse
from . import services
from . import broadcast_recommender as br # broadcast_recommender 임포트
from .product_embedder import ProductEmbedder
from .trend_collector import TrendCollector, TrendProcessor
from .broadcast_workflow import BroadcastWorkflow
from .trend_db_manager import trend_db_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 모델을 비동기적으로 로드합니다.
    print("--- Loading model on startup... ---")
    model = await services.load_model_async()
    app.state.model = model
    
    # ProductEmbedder 초기화
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        app.state.product_embedder = ProductEmbedder(
            openai_api_key=openai_api_key,
            qdrant_host="qdrant_vector_db" if os.getenv("DOCKER_ENV") else "localhost"
        )
        print("--- ProductEmbedder initialized ---")
        
        # TrendProcessor 초기화
        app.state.trend_processor = TrendProcessor(app.state.product_embedder)
        print("--- TrendProcessor initialized ---")
        
        # BroadcastWorkflow 초기화
        app.state.broadcast_workflow = BroadcastWorkflow(model, app.state.product_embedder)
        print("--- BroadcastWorkflow initialized ---")
    else:
        print("--- Warning: OPENAI_API_KEY not found, ProductEmbedder not initialized ---")
        app.state.product_embedder = None
        app.state.trend_processor = None
        app.state.broadcast_workflow = None
    
    print("--- Model loaded successfully. ---")
    yield
    # 애플리케이션 종료 시 정리 (필요 시)
    app.state.model = None
    app.state.product_embedder = None
    app.state.trend_processor = None
    app.state.broadcast_workflow = None

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

@app.post("/api/v1/extract-params")
async def extract_params(payload: RecommendRequest):
    """
    사용자 질문에서 파라미터만 추출합니다.
    """
    try:
        extracted_params = await services.extract_and_enrich_params(payload.user_query)
        return {"extracted_params": extracted_params}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- ERROR IN /api/v1/extract-params ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/v1/recommend-with-params")
async def recommend_with_params(payload: dict, request: Request):
    """
    수정된 파라미터로 방송 편성을 추천합니다.
    """
    try:
        model = request.app.state.model
        response_data = await services.get_recommendations_with_params(payload, model)
        return response_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- ERROR IN /api/v1/recommend-with-params ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/v1/recommend-candidates", response_model=CandidatesResponse)
async def recommend_candidates(payload: RecommendRequest, request: Request, top_k: int = 5):
    """시간대별 Top-k 후보 리스트를 반환합니다. 기본 k=5"""
    try:
        model = request.app.state.model
        response_data = await services.get_candidates(payload.user_query, model, top_k=top_k)
        return response_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- ERROR IN /api/v1/recommend-candidates ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/v1/trends/collect", response_model=TrendCollectionResponse)
async def collect_trends(request: Request):
    """트렌드 데이터 수집 (배치 처리 - DB 저장)"""
    print("--- API Endpoint /api/v1/trends/collect received a request ---")
    try:
        # 외부 API에서 트렌드 수집 후 DB 저장
        result = await trend_db_manager.collect_and_save_trends()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"트렌드 수집 실패: {result['error']}")
        
        # DB에서 최신 트렌드 조회
        latest_trends = await trend_db_manager.get_latest_trends(limit=50, hours_back=1)
        
        # 스키마에 맞게 변환
        trend_schemas = []
        for trend in latest_trends:
            trend_schemas.append({
                "keyword": trend["keyword"],
                "source": trend["source"],
                "score": trend["score"],
                "timestamp": trend["collected_at"],
                "category": trend["category"],
                "related_keywords": trend["related_keywords"] or [],
                "metadata": trend["metadata"] or {}
            })
        
        return TrendCollectionResponse(
            trends=trend_schemas,
            collection_timestamp=result["timestamp"],
            total_count=result["saved_count"]
        )
        
    except Exception as e:
        print(f"--- ERROR IN /api/v1/trends/collect ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/v1/trends/analyze", response_model=TrendAnalysisResponse)
async def analyze_trends(request: Request):
    """트렌드를 분석하고 상품과 매칭합니다. (DB에서 조회)"""
    try:
        trend_processor = request.app.state.trend_processor
        
        if not trend_processor:
            raise HTTPException(status_code=503, detail="TrendProcessor가 초기화되지 않았습니다.")
        
        # DB에서 최신 트렌드 조회
        latest_trends = await trend_db_manager.get_latest_trends(limit=30, hours_back=6)
        
        if not latest_trends:
            return TrendAnalysisResponse(
                trends=[],
                matched_results={},
                analysis_timestamp=datetime.now().isoformat()
            )
        
        # 상품 매칭
        matched_results = {}
        for trend_data in latest_trends:
            keyword = trend_data["keyword"]
            matching_result = await trend_processor.match_trend_to_products(keyword)
            if matching_result["matched_products"]:
                matched_results[keyword] = matching_result
        
        # 스키마에 맞게 변환
        trend_schemas = []
        matching_responses = {}
        
        for trend_data in latest_trends:
            trend_schemas.append({
                "keyword": trend_data["keyword"],
                "source": trend_data["source"],
                "score": trend_data["score"],
                "timestamp": trend_data["collected_at"],
                "category": trend_data["category"],
                "related_keywords": trend_data["related_keywords"] or [],
                "metadata": trend_data["metadata"] or {}
            })
            
            keyword = trend_data["keyword"]
            if keyword in matched_results:
                boost_factor = trend_processor.calculate_trend_boost_factor(trend_data["score"])
                matching_responses[keyword] = {
                    "trend_keyword": keyword,
                    "trend_info": matched_results[keyword]["trend_info"],
                    "matched_products": matched_results[keyword]["matched_products"],
                    "boost_factor": boost_factor
                }
        
        return TrendAnalysisResponse(
            trends=trend_schemas,
            matched_results=matching_responses,
            analysis_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"--- ERROR IN /api/v1/trends/analyze ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/v1/recommend-with-trends", response_model=RecommendResponse)
async def recommend_with_trends(payload: RecommendRequest, request: Request):
    """트렌드 데이터를 반영한 강화된 방송 편성 추천"""
    print("--- API Endpoint /api/v1/recommend-with-trends received a request ---")
    try:
        model = request.app.state.model
        product_embedder = request.app.state.product_embedder
        
        if not product_embedder:
            raise HTTPException(status_code=503, detail="ProductEmbedder가 초기화되지 않았습니다.")
        
        response_data = await services.get_trend_enhanced_recommendations(
            payload.user_query, model, product_embedder
        )
        return response_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- ERROR IN /api/v1/recommend-with-trends ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/v1/broadcast/recommendations", response_model=BroadcastResponse)
async def broadcast_recommendations(payload: BroadcastRequest, request: Request):
    """새로운 방송 편성 AI 추천 API - LangChain 기반 2단계 워크플로우"""
    print(f"--- API Endpoint /api/v1/broadcast/recommendations received a request: {payload.broadcastTime} ---")
    try:
        broadcast_workflow = request.app.state.broadcast_workflow
        
        if not broadcast_workflow:
            raise HTTPException(status_code=503, detail="BroadcastWorkflow가 초기화되지 않았습니다.")
        
        response_data = await broadcast_workflow.process_broadcast_recommendation(
            payload.broadcastTime, 
            payload.recommendationCount
        )
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- ERROR IN /api/v1/broadcast/recommendations ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/v1/health", summary="Health Check")
def health_check():
    """API 서버의 상태를 확인합니다."""
    return {"status": "ok"}
