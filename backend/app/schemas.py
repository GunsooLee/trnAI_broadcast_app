from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime as dt

class RecommendRequest(BaseModel):
    """사용자의 추천 요청 스키마"""
    user_query: str = Field(..., description="사용자가 입력한 자연어 질문", example="내일 저녁에 방송할 만한 거 추천해줘")

class RecommendationItem(BaseModel):
    """개별 추천 결과 항목 스키마"""
    time_slot: str
    predicted_sales: float
    product_code: str
    category: str
    features: Dict[str, Any]

class RecommendResponse(BaseModel):
    """추천 API 응답 스키마"""
    extracted_params: Dict[str, Any] = Field(description="사용자 질문에서 추출 및 보강된 파라미터")
    recommendations: List[RecommendationItem] = Field(description="추천된 방송 편성 목록")

# --- Top-k 후보 응답 스키마 ---
class TimeSlotCandidates(BaseModel):
    time_slot: str
    items: List[RecommendationItem]

class CandidatesResponse(BaseModel):
    extracted_params: Dict[str, Any]
    candidates: List[TimeSlotCandidates]
