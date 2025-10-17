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

# --- 트렌드 관련 스키마 ---
class TrendKeywordSchema(BaseModel):
    """트렌드 키워드 스키마"""
    keyword: str
    source: str
    score: float
    timestamp: str
    category: Optional[str] = None
    related_keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class TrendCollectionResponse(BaseModel):
    """트렌드 수집 응답 스키마"""
    trends: List[TrendKeywordSchema]
    collection_timestamp: str
    total_count: int

class TrendMatchingResponse(BaseModel):
    """트렌드-상품 매칭 응답 스키마"""
    trend_keyword: str
    trend_info: Dict[str, Any]
    matched_products: List[Dict[str, Any]]
    boost_factor: float

class TrendAnalysisResponse(BaseModel):
    """트렌드 분석 응답 스키마"""
    trends: List[TrendKeywordSchema]
    matched_results: Dict[str, TrendMatchingResponse]
    analysis_timestamp: str

# --- 새로운 방송 추천 API 스키마 ---
class BroadcastRequest(BaseModel):
    """방송 추천 요청 스키마"""
    broadcastTime: str = Field(..., description="방송 시간", example="2025-09-15T22:40:00+09:00")
    recommendationCount: int = Field(default=5, description="추천 개수", example=5)
    trendRatio: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="트렌드 매칭 비율 (0.0~1.0). 0.3=트렌드30%+매출70%, 0.5=균형, 1.0=트렌드만",
        example=0.3
    )

class RecommendedCategory(BaseModel):
    """추천 카테고리 스키마"""
    rank: int
    name: str
    reason: str
    predictedSales: str

class ProductInfo(BaseModel):
    """상품 정보 스키마"""
    productId: str
    productName: str
    category: str
    price: Optional[float] = Field(default=None, description="상품 가격", example=99000.0)
    tapeCode: Optional[str] = Field(default=None, description="방송테이프 코드", example="T001")
    tapeName: Optional[str] = Field(default=None, description="방송테이프명", example="프리미엄 다이어트 보조제 방송테이프")

class Reasoning(BaseModel):
    """추천 근거 스키마"""
    summary: str
    linkedCategories: List[str]
    matchedKeywords: List[str]

class BusinessMetrics(BaseModel):
    """비즈니스 지표 스키마"""
    pastAverageSales: str
    marginRate: float
    stockLevel: str

class BroadcastRecommendation(BaseModel):
    """방송 추천 항목 스키마"""
    rank: int
    productInfo: ProductInfo
    reasoning: Reasoning
    businessMetrics: BusinessMetrics
    recommendationType: str  # "trend_match" (유사도) or "sales_prediction" (매출예측)

class BroadcastResponse(BaseModel):
    """방송 추천 응답 스키마"""
    requestTime: str
    recommendedCategories: List[RecommendedCategory]
    recommendations: List[BroadcastRecommendation]

# --- 방송테이프 관련 스키마 ---
class BroadcastTapeInfo(BaseModel):
    """방송테이프 정보 스키마"""
    tape_id: str
    product_code: str
    product_name: str
    category: str
    broadcast_date: str
    broadcast_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    status: str
    created_at: str
    updated_at: str

class TapeCollectionResponse(BaseModel):
    """테이프 수집 응답 스키마"""
    tapes: List[BroadcastTapeInfo]
    collection_timestamp: str
    total_count: int
    upserted_count: int
