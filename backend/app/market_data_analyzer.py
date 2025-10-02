"""
시장 데이터 분석기 - DB에서 크롤링된 홈쇼핑/검색 트렌드 데이터를 가져와서 LLM 분석
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

class MarketDataAnalyzer:
    """시장 데이터 분석기 - 크롤링된 데이터를 LLM으로 분석"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.engine = create_engine(os.getenv("POSTGRES_URI"))
    
    async def get_latest_market_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """DB에서 최신 시장 데이터 수집"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        market_data = {
            "homeshopping_rankings": [],
            "search_trends": [],
            "category_insights": [],
            "collected_at": datetime.now().isoformat()
        }
        
        try:
            with self.engine.connect() as conn:
                # 홈쇼핑 사이트 랭킹 데이터
                homeshopping_query = text("""
                    SELECT source, product_name, rank_position, category, price, collected_at
                    FROM market_data 
                    WHERE data_type = 'homeshopping_ranking' 
                    AND collected_at >= :cutoff_time 
                    ORDER BY source, rank_position 
                    LIMIT 50
                """)
                
                result = conn.execute(homeshopping_query, {"cutoff_time": cutoff_time})
                for row in result:
                    market_data["homeshopping_rankings"].append({
                        "site_name": row.source,
                        "product_name": row.product_name,
                        "rank": row.rank_position,
                        "category": row.category,
                        "price": row.price,
                        "collected_at": row.collected_at.isoformat()
                    })
                
                # 검색 트렌드 데이터
                search_trends_query = text("""
                    SELECT source, keyword, category, trend_score, search_volume, collected_at
                    FROM market_data 
                    WHERE data_type = 'search_trend' 
                    AND collected_at >= :cutoff_time 
                    ORDER BY trend_score DESC 
                    LIMIT 30
                """)
                
                result = conn.execute(search_trends_query, {"cutoff_time": cutoff_time})
                for row in result:
                    market_data["search_trends"].append({
                        "source": row.source,  # 'naver_shopping', 'google_trends' 등
                        "keyword": row.keyword,
                        "category": row.category,
                        "trend_score": row.trend_score,
                        "search_volume": row.search_volume,
                        "collected_at": row.collected_at.isoformat()
                    })
                
                logger.info(f"시장 데이터 수집 완료: 홈쇼핑 {len(market_data['homeshopping_rankings'])}개, 검색트렌드 {len(market_data['search_trends'])}개")
                
        except Exception as e:
            logger.error(f"시장 데이터 수집 실패: {e}")
            # 실패 시 빈 데이터 반환
        
        return market_data
    
    def format_market_data_for_llm(self, market_data: Dict[str, Any]) -> str:
        """시장 데이터를 LLM 프롬프트용 텍스트로 포맷팅"""
        
        formatted_text = "[수집된 데이터]\n\n"
        
        # 홈쇼핑 사이트별 랭킹 정리
        if market_data["homeshopping_rankings"]:
            sites = {}
            for item in market_data["homeshopping_rankings"]:
                site = item["site_name"]
                if site not in sites:
                    sites[site] = []
                sites[site].append(item)
            
            for site_name, items in sites.items():
                formatted_text += f"- {site_name} 실시간 랭킹:\n"
                for item in items[:10]:  # 상위 10개만
                    formatted_text += f"  {item['rank']}위: {item['product_name']}"
                    if item['category']:
                        formatted_text += f" ({item['category']})"
                    if item['price']:
                        formatted_text += f" - {item['price']:,}원"
                    formatted_text += "\n"
                formatted_text += "\n"
        
        # 검색 트렌드 정리
        if market_data["search_trends"]:
            sources = {}
            for item in market_data["search_trends"]:
                source = item["source"]
                if source not in sources:
                    sources[source] = []
                sources[source].append(item)
            
            source_names = {
                "naver_shopping": "네이버 쇼핑인사이트",
                "google_trends": "구글 트렌드",
                "naver_datalab": "네이버 데이터랩"
            }
            
            for source, items in sources.items():
                display_name = source_names.get(source, source)
                
                # 카테고리별로 그룹화
                categories = {}
                for item in items:
                    cat = item.get('category', '기타')
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(item)
                
                for category, cat_items in categories.items():
                    formatted_text += f"- {display_name} '{category}' 분야 급상승 검색어:\n"
                    for i, item in enumerate(cat_items[:5], 1):  # 상위 5개만
                        formatted_text += f"  {i}위: {item['keyword']}"
                        if item.get('trend_score'):
                            formatted_text += f" (트렌드점수: {item['trend_score']})"
                        formatted_text += "\n"
                    formatted_text += "\n"
        
        return formatted_text
    
    async def analyze_market_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """시장 데이터를 분석하여 트렌드 키워드 추천"""
        
        # 1. DB에서 최신 시장 데이터 수집
        market_data = await self.get_latest_market_data(hours_back)
        
        if not market_data["homeshopping_rankings"] and not market_data["search_trends"]:
            # 데이터가 없으면 기본 응답 반환
            return {
                "success": False,
                "error": "분석할 시장 데이터가 없습니다. n8n 크롤링이 정상 작동하는지 확인해주세요.",
                "data_status": {
                    "homeshopping_count": 0,
                    "search_trends_count": 0,
                    "last_updated": market_data["collected_at"]
                }
            }
        
        # 2. 데이터를 LLM 프롬프트용으로 포맷팅
        formatted_data = self.format_market_data_for_llm(market_data)
        
        # 3. LLM에게 분석 요청
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        prompt = f"""너는 대한민국 최고의 홈쇼핑 MD야. 아래는 방금 수집한 최신 시장 데이터야.

{formatted_data}

[너의 임무]
1. 위 데이터를 바탕으로 현재 가장 중요한 홈쇼핑 트렌드 테마 3가지를 요약해줘.
2. 이 데이터들을 종합적으로 고려하여, 지금 당장 방송에 편성해야 할 가장 유망한 키워드 15개를 선정해줘.
3. 각 키워드마다 왜 유망한지 근거 데이터를 함께 제시해줘.
   (예: '일월 온수매트': CJ온스타일 실시간 랭킹 상위권이며, 9월 중순 환절기 시즌 수요 급증 예상)
4. 최종 결과는 아래 JSON 형식으로 반환해줘:

{{
  "trend_summary": ["테마1", "테마2", "테마3"],
  "recommended_keywords": [
    {{
      "keyword": "키워드1",
      "trend_score": 95,
      "reason": "근거1"
    }},
    {{
      "keyword": "키워드2", 
      "trend_score": 90,
      "reason": "근거2"
    }}
  ]
}}

현재 날짜: {current_date}
계절적 요인: 9월 중순 (환절기, 가을 시즌 진입)
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                # JSON 부분만 추출 (```json 태그 제거)
                if "```json" in llm_response:
                    json_start = llm_response.find("```json") + 7
                    json_end = llm_response.find("```", json_start)
                    json_text = llm_response[json_start:json_end].strip()
                elif "{" in llm_response:
                    json_start = llm_response.find("{")
                    json_end = llm_response.rfind("}") + 1
                    json_text = llm_response[json_start:json_end]
                else:
                    json_text = llm_response
                
                analysis_result = json.loads(json_text)
                
                # 결과에 메타데이터 추가
                analysis_result.update({
                    "success": True,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_sources": {
                        "homeshopping_count": len(market_data["homeshopping_rankings"]),
                        "search_trends_count": len(market_data["search_trends"]),
                        "data_freshness_hours": hours_back
                    },
                    "raw_llm_response": llm_response
                })
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                logger.error(f"LLM 응답 JSON 파싱 실패: {e}")
                return {
                    "success": False,
                    "error": "LLM 응답을 JSON으로 파싱할 수 없습니다.",
                    "raw_response": llm_response,
                    "analysis_timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"LLM 분석 요청 실패: {e}")
            return {
                "success": False,
                "error": f"LLM 분석 중 오류 발생: {str(e)}",
                "analysis_timestamp": datetime.now().isoformat()
            }

# 사용 예시
if __name__ == "__main__":
    import asyncio
    
    async def test_analyzer():
        analyzer = MarketDataAnalyzer(os.getenv("OPENAI_API_KEY"))
        result = await analyzer.analyze_market_trends(hours_back=24)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    asyncio.run(test_analyzer())
