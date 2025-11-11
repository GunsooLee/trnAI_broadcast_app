from pydantic import BaseModel, Field, validator
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
    trendWeight: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="트렌드 가중치 (0.0~1.0). 예: 0.3=트렌드 30%",
        example=0.3
    )
    salesWeight: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="매출 예측 가중치 (0.0~1.0). 예: 0.7=매출 70%",
        example=0.7
    )
    
    @validator('salesWeight')
    def validate_weights_sum(cls, v, values):
        """트렌드 + 매출 가중치 합이 1.0인지 검증"""
        if 'trendWeight' in values:
            total = values['trendWeight'] + v
            if not (0.99 <= total <= 1.01):  # 부동소수점 오차 허용
                raise ValueError(f"trendWeight + salesWeight는 1.0이어야 합니다 (현재: {total:.2f})")
        return v

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
    brand: Optional[str] = Field(default=None, description="브랜드", example="해피콜")
    price: Optional[float] = Field(default=None, description="상품 가격", example=99000.0)
    tapeCode: Optional[str] = Field(default=None, description="방송테이프 코드", example="T001")
    tapeName: Optional[str] = Field(default=None, description="방송테이프명", example="프리미엄 다이어트 보조제 방송테이프")

class Reasoning(BaseModel):
    """추천 근거 스키마"""
    summary: str
    linkedCategories: List[str]

class LastBroadcastMetrics(BaseModel):
    """최근 방송 실적 스키마"""
    broadcastStartTime: str = Field(description="방송시작일시")
    orderQuantity: int = Field(description="주문수량")
    totalProfit: float = Field(description="매출총이익(실적)")
    profitEfficiency: float = Field(description="ONAIR매출총이익(효율)")
    conversionWorth: float = Field(description="환산가치값(분리송출)")
    conversionRate: float = Field(description="적용전환율")
    realFee: float = Field(description="실질수수료")
    mixFee: float = Field(description="혼합수수료")

class BusinessMetrics(BaseModel):
    """비즈니스 지표 스키마"""
    aiPredictedSales: str = Field(description="AI 예측 매출 (XGBoost 모델)")
    marginRate: float
    stockLevel: str
    lastBroadcast: Optional[LastBroadcastMetrics] = Field(
        default=None,
        description="가장 최근 방송의 실제 실적 데이터"
    )

class BroadcastRecommendation(BaseModel):
    """방송 추천 항목 스키마"""
    rank: int
    productInfo: ProductInfo
    reasoning: Reasoning
    businessMetrics: BusinessMetrics
    recommendationType: str  # "trend_match" (유사도) or "sales_prediction" (매출예측)

class ExternalProduct(BaseModel):
    """외부 상품 (네이버 베스트) 스키마"""
    product_id: str
    name: str
    rank: int
    rank_change: Optional[int] = None
    rank_change_text: str
    sale_price: int
    discounted_price: int
    discount_ratio: int
    image_url: str
    landing_url: str
    mobile_landing_url: str
    is_delivery_free: bool
    delivery_fee: int
    cumulation_sale_count: int
    review_count: Optional[int] = None
    review_score: Optional[float] = None
    mall_name: Optional[str] = None
    channel_no: Optional[str] = None
    collected_at: Optional[str] = None
    collected_date: Optional[str] = None

class BroadcastResponse(BaseModel):
    """방송 추천 응답 스키마"""
    requestTime: str
    recommendedCategories: List[RecommendedCategory]
    recommendations: List[BroadcastRecommendation]
    externalProducts: Optional[List[ExternalProduct]] = Field(
        default=None,
        description="네이버 베스트 상품 (외부 상품)"
    )

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
