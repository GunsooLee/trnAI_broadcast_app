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

from .schemas import (
    BroadcastRequest, BroadcastResponse, TapeCollectionResponse, BroadcastTapeInfo,
    SalesPredictionRequest, SalesPredictionResponse, ProductSalesPrediction
)
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
    import time
    start_time = time.time()
    
    print(f"--- API Endpoint /api/v1/broadcast/recommendations received a request: {payload.broadcastTime} ---")
    print(f"--- 가중치 설정: 트렌드 {payload.trendWeight:.0%} / 매출 {payload.sellingWeight:.0%} ---")
    try:
        response_data = await workflow.process_broadcast_recommendation(
            payload.broadcastTime,
            payload.recommendationCount,
            payload.trendWeight,
            payload.sellingWeight
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  총 응답 시간: {elapsed_time:.2f}초")

        # 빈 추천 결과도 정상 응답으로 처리 (200 OK)
        if not response_data.recommendations or len(response_data.recommendations) == 0:
            print(f"⚠️  추천 결과 0개 - 정상 응답 반환 (빈 배열)")

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
# � 매출 예측 API 엔드포인트
# ========================================

@app.post("/api/v1/sales/predict", response_model=SalesPredictionResponse)
async def predict_sales_by_date(payload: SalesPredictionRequest, workflow: BroadcastWorkflow = Depends(get_broadcast_workflow)):
    """📊 특정 날짜의 편성표 기반 매출 예측 API
    
    입력된 날짜에 편성된 방송 상품들의 매출을 XGBoost 모델로 예측합니다.
    """
    import time
    start_time = time.time()
    
    print(f"--- API Endpoint /api/v1/sales/predict received request for date: {payload.date} ---")
    
    try:
        from datetime import datetime
        import pandas as pd
        
        # 1. 날짜 형식 검증
        try:
            target_date = datetime.strptime(payload.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요."
            )
        
        # 2. Netezza에서 해당 날짜의 편성표 조회
        print(f"--- Fetching broadcast schedule for {payload.date} ---")
        schedule_data = await netezza_conn.get_broadcast_schedule_by_date(payload.date)
        
        if not schedule_data:
            print(f"⚠️  해당 날짜에 편성된 방송이 없습니다: {payload.date}")
            return SalesPredictionResponse(
                date=payload.date,
                predictions=[]
            )
        
        print(f"--- Found {len(schedule_data)} broadcasts for {payload.date} ---")
        
        # 3. 각 방송 상품에 대해 매출 예측
        import numpy as np
        from . import broadcast_recommender as br
        
        predictions = []
        
        for item in schedule_data:
            product_code = item.get('product_code', '')
            product_name = item.get('product_name', '')
            broadcast_start = item.get('broadcast_start_time', '')
            duration_minutes = item.get('duration_minutes', 0)
            
            # 방송 시간 추출 (HH:MM 형식)
            try:
                if isinstance(broadcast_start, str):
                    broadcast_time = broadcast_start.split(' ')[1][:5] if ' ' in broadcast_start else broadcast_start[:5]
                else:
                    broadcast_time = broadcast_start.strftime('%H:%M')
            except:
                broadcast_time = "00:00"
            
            # 4. XGBoost 모델로 매출 예측
            try:
                # 상품 정보 조회
                product_df = br.fetch_product_info([product_code], workflow.engine)
                
                if product_df.empty:
                    print(f"⚠️  상품 정보를 찾을 수 없습니다: {product_code}")
                    continue
                
                # 상품 딕셔너리 생성
                product_dict = product_df.iloc[0].to_dict()
                product_dict['product_code'] = product_code
                product_dict['product_name'] = product_name
                
                # context 생성 (broadcast_workflow와 동일한 형식)
                hour = int(broadcast_time.split(':')[0])
                broadcast_dt = target_date.replace(hour=hour, minute=0, second=0)
                
                # DB에서 공휴일 정보 조회
                holiday_name = await workflow._get_holiday_from_db(broadcast_dt.date())
                
                # 날씨 정보 수집
                weather_info = br.get_weather_by_date(broadcast_dt.date())
                
                context = {
                    'broadcast_dt': broadcast_dt,
                    'time_slot': _get_time_slot(broadcast_time),
                    'weather': weather_info,
                    'holiday_name': holiday_name
                }
                
                # workflow의 _prepare_features_for_product 사용
                features = workflow._prepare_features_for_product(product_dict, context)
                pred_df = pd.DataFrame([features])
                
                # XGBoost 모델로 예측
                predicted_sales_log = workflow.model.predict(pred_df)[0]
                predicted_sales = np.expm1(predicted_sales_log)
                
                # 음수 예측값 방지
                predicted_sales = max(0, predicted_sales)
                
                predictions.append(ProductSalesPrediction(
                    product_code=product_code,
                    product_name=product_name,
                    broadcast_time=broadcast_time,
                    duration_minutes=duration_minutes,
                    predicted_sales=float(predicted_sales),
                    confidence=0.85
                ))
                
                print(f"  ✅ {product_code} ({product_name[:20]}...): {predicted_sales:,.0f}원")
                
            except Exception as e:
                print(f"  ⚠️  매출 예측 실패 ({product_code}): {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  총 응답 시간: {elapsed_time:.2f}초")
        print(f"� 예측 완료: {len(predictions)}개 상품")
        
        return SalesPredictionResponse(
            date=payload.date,
            predictions=predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"--- ERROR IN /api/v1/sales/predict ---")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"매출 예측 중 오류가 발생했습니다: {str(e)}"
        )

def _get_time_slot(time_str: str) -> str:
    """시간 문자열을 시간대로 변환"""
    try:
        hour = int(time_str.split(':')[0])
        if 6 <= hour < 12:
            return "아침"
        elif 12 <= hour < 18:
            return "오후"
        elif 18 <= hour < 24:
            return "저녁"
        else:
            return "새벽"
    except:
        return "오후"

# ========================================
# �🔄 데이터 마이그레이션 API (n8n 연동용)
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

# ========================================
# 🛒 외부 상품 크롤링 API (n8n 연동용)
# ========================================
from .routers.external_products import router as external_products_router
app.include_router(external_products_router)

# ========================================
# 📊 매출 예측 API (단일 상품 예측)
# ========================================
from .api.sales_prediction import router as sales_prediction_router
app.include_router(sales_prediction_router)
