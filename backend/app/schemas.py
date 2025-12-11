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
    broadcastTime: str = Field(..., description="방송 시간", example="2025-11-19T14:00:00+09:00")
    recommendationCount: int = Field(default=10, description="추천 개수", example=10)
    trendWeight: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="트렌드 가중치 (0.0~1.0). 예: 0.3=트렌드 30%",
        example=0.3
    )
    sellingWeight: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="매출 예측 가중치 (0.0~1.0). 예: 0.7=매출 70%",
        example=0.7
    )
    
    @validator('sellingWeight')
    def validate_weights_sum(cls, v, values):
        """트렌드 + 매출 가중치 합이 1.0인지 검증"""
        if 'trendWeight' in values:
            total = values['trendWeight'] + v
            if not (0.99 <= total <= 1.01):  # 부동소수점 오차 허용
                raise ValueError(f"trendWeight + sellingWeight는 1.0이어야 합니다 (현재: {total:.2f})")
        return v

class RecommendationSource(BaseModel):
    """추천 출처 정보 스키마 - 왜 이 상품이 추천됐는지 추적"""
    source_type: str = Field(
        description="추천 출처 유형",
        example="news_trend"
    )  # news_trend, ai_trend, context, xgboost_sales, competitor, rag_match
    
    # 뉴스 트렌드 출처
    news_keyword: Optional[str] = Field(default=None, description="뉴스에서 추출한 키워드", example="손난로")
    news_title: Optional[str] = Field(default=None, description="뉴스 기사 제목", example="한파에 손난로 판매 급증")
    news_url: Optional[str] = Field(default=None, description="뉴스 기사 URL")
    
    # AI 트렌드 출처 (LLM 생성 키워드)
    ai_keyword: Optional[str] = Field(default=None, description="AI가 생성한 트렌드 키워드", example="겨울 패딩")
    ai_reason: Optional[str] = Field(default=None, description="AI가 키워드를 생성한 이유", example="12월 저녁 시간대 겨울 의류 수요")
    
    # 컨텍스트 기반 출처 (날씨, 시간대, 공휴일 등)
    context_factor: Optional[str] = Field(default=None, description="컨텍스트 요인", example="날씨: 한파, 시간대: 저녁")
    
    # RAG 검색 출처
    matched_keyword: Optional[str] = Field(default=None, description="RAG 검색에 사용된 키워드", example="전기장판")
    similarity_score: Optional[float] = Field(default=None, description="벡터 유사도 점수", example=0.85)
    keyword_origin: Optional[str] = Field(default=None, description="키워드 출처 유형 (news/ai/context)", example="ai")
    keyword_origin_detail: Optional[str] = Field(default=None, description="키워드 출처 상세 설명", example="AI가 저녁 겨울 시즌에 맞게 생성")
    
    # XGBoost 매출 예측 출처
    xgboost_rank: Optional[int] = Field(default=None, description="XGBoost 매출 예측 순위", example=3)
    predicted_sales: Optional[float] = Field(default=None, description="예측 매출액", example=15000000)
    
    # 경쟁사 편성 출처
    competitor_name: Optional[str] = Field(default=None, description="경쟁사명", example="롯데홈쇼핑")
    competitor_time: Optional[str] = Field(default=None, description="경쟁사 편성 시간", example="14:00-15:00")

class ProductInfo(BaseModel):
    """상품 정보 스키마"""
    productId: str
    productName: str
    category: str = Field(description="대분류 카테고리", example="건강식품")
    categoryMiddle: Optional[str] = Field(default=None, description="중분류 카테고리", example="건강보조식품")
    categorySub: Optional[str] = Field(default=None, description="소분류 카테고리", example="다이어트")
    brand: Optional[str] = Field(default=None, description="브랜드", example="해피콜")
    price: Optional[float] = Field(default=None, description="상품 가격", example=99000.0)
    tapeCode: Optional[str] = Field(default=None, description="방송테이프 코드", example="T001")
    tapeName: Optional[str] = Field(default=None, description="방송테이프명", example="프리미엄 다이어트 보조제 방송테이프")

# Reasoning 스키마 제거 - 단순 문자열로 변경

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
    lastBroadcast: Optional[LastBroadcastMetrics] = Field(
        default=None,
        description="가장 최근 방송의 실제 실적 데이터"
    )

class BroadcastRecommendation(BaseModel):
    """방송 추천 항목 스키마"""
    rank: int
    productInfo: ProductInfo
    reasoning: str = Field(description="추천 근거")
    businessMetrics: BusinessMetrics
    # sources 필드 제거 - 추천 출처 정보는 내부적으로 reasoning 생성에만 사용

class NaverProduct(BaseModel):
    """네이버 베스트 상품 스키마"""
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

class CompetitorProduct(BaseModel):
    """타 홈쇼핑사 편성 상품 스키마 (네이버 상품 포함)"""
    company_name: str = Field(description="경쟁사명 또는 '네이버 스토어'", example="CJ온스타일")
    broadcast_title: str = Field(description="방송 제목 또는 상품명", example="프리미엄 건강식품 특가전")
    start_time: Optional[str] = Field(default="", description="방송 시작 시간 (네이버 상품은 빈칸)", example="2025-11-19 14:00:00")
    end_time: Optional[str] = Field(default="", description="방송 종료 시간 (네이버 상품은 빈칸)", example="2025-11-19 15:00:00")
    duration_minutes: Optional[int] = Field(default=None, description="방송 시간(분)", example=60)
    category_main: Optional[str] = Field(default="", description="대분류 카테고리", example="건강식품")

class BroadcastResponse(BaseModel):
    """방송 추천 응답 스키마"""
    requestTime: str
    recommendations: List[BroadcastRecommendation]
    competitorProducts: Optional[List[CompetitorProduct]] = Field(
        default=None,
        description="네이버 인기 상품 + 타사 편성 통합 (AI 선택 10개, 5:5 비율 유지)"
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
