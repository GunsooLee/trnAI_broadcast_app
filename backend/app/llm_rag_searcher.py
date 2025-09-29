"""
LLM 기반 RAG 검색 시스템
자연어 질문을 받아 LLM이 상품 데이터베이스를 검색하고 추천하는 방식
"""

import openai
from typing import List, Dict, Any
import json
from sqlalchemy import create_engine, text
import os
import logging

logger = logging.getLogger(__name__)

class LLMRAGSearcher:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.engine = create_engine(os.getenv("POSTGRES_URI", os.getenv("DB_URI")))
    
    def get_product_context(self, limit: int = 50) -> str:
        """상품 데이터베이스에서 컨텍스트 정보 조회"""
        query = """
        SELECT p.product_code, p.product_name, p.category_main, p.category_middle,
               p.category_sub, p.price,
               t.tape_code, t.tape_name
        FROM TAIGOODS p
        LEFT JOIN TAIPGMTAPE t ON p.product_code = t.product_code
        WHERE t.production_status = 'ready' OR t.production_status IS NULL
        ORDER BY p.product_code
        LIMIT %s
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), (limit,)).fetchall()
        
        # 상품 정보를 텍스트로 변환
        products_text = []
        for row in results:
            tape_status = "방송테이프 준비완료" if row[6] else "방송테이프 없음"
            product_info = f"""
상품코드: {row[0]}
상품명: {row[1]}
카테고리: {row[2]} > {row[3]} > {row[4]}
가격: {row[5]:,}원
방송상태: {tape_status}
"""
            products_text.append(product_info.strip())
        
        return "\n\n".join(products_text)
    
    def build_candidate_context(self, candidate_products: List[Dict]) -> str:
        """벡터 검색 결과를 LLM 컨텍스트로 변환"""
        products_text = []
        for i, product in enumerate(candidate_products, 1):
            tape_status = "방송테이프 준비완료" if product.get('tape_code') else "방송테이프 없음"
            similarity = product.get('similarity_score', 0.0)
            
            product_info = f"""
{i}. 상품코드: {product.get('product_code', 'N/A')}
   상품명: {product.get('product_name', 'N/A')}
   카테고리: {product.get('category_main', 'N/A')}
   방송상태: {tape_status}
   유사도: {similarity:.3f}
"""
            products_text.append(product_info.strip())
        
        return "\n\n".join(products_text)
    
    def search_with_llm(self, user_query: str, trend_context: str = None) -> Dict[str, Any]:
        """LLM을 사용한 2단계 하이브리드 검색 (벡터 검색 + LLM 분석)"""
        
        # 1️⃣ 벡터 검색으로 후보 상품 필터링 (전체 → 30개)
        from product_embedder import ProductEmbedder
        
        try:
            embedder = ProductEmbedder(
                openai_api_key=self.openai_client.api_key,
                qdrant_host="qdrant_vector_db" if os.getenv("DOCKER_ENV") else "localhost"
            )
            
            # 벡터 검색으로 관련 상품 30개 추출
            candidate_products = embedder.search_products(
                trend_keywords=[user_query],
                top_k=30,
                score_threshold=0.3,
                only_ready_products=True
            )
            
            if not candidate_products:
                return {
                    "analysis": "관련 상품을 찾을 수 없습니다",
                    "recommendations": [],
                    "summary": "검색 조건에 맞는 상품이 없습니다"
                }
            
            # 2️⃣ 후보 상품들을 LLM 컨텍스트로 변환
            product_context = self.build_candidate_context(candidate_products)
            
        except Exception as e:
            logger.warning(f"벡터 검색 실패, 전체 상품 조회로 대체: {e}")
            # 벡터 검색 실패 시 전체 상품에서 제한적으로 조회
            product_context = self.get_product_context(limit=30)
        
        # 트렌드 컨텍스트 추가
        trend_info = f"\n\n현재 트렌드 정보:\n{trend_context}" if trend_context else ""
        
        # LLM 프롬프트 구성
        prompt = f"""
당신은 홈쇼핑 방송 상품 추천 전문가입니다.
사용자의 질문을 분석하여 가장 적합한 상품들을 추천해주세요.

상품 데이터베이스:
{product_context}
{trend_info}

사용자 질문: {user_query}

다음 JSON 형식으로 응답해주세요:
{{
    "analysis": "사용자 질문 분석 결과",
    "recommendations": [
        {{
            "product_code": "상품코드",
            "product_name": "상품명",
            "category": "카테고리",
            "price": "가격",
            "reason": "추천 이유",
            "confidence": 0.95,
            "broadcast_ready": true/false
        }}
    ],
    "summary": "전체 추천 요약"
}}

중요사항:
1. 방송테이프가 준비된 상품을 우선 추천
2. 트렌드와 연관성이 높은 상품 선택
3. 구체적인 추천 이유 제시
4. 최대 5개 상품 추천
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 홈쇼핑 방송 상품 추천 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # JSON 응답 파싱
            response_text = response.choices[0].message.content
            
            # JSON 추출 (```json 블록 처리)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            
            logger.info(f"LLM RAG 검색 완료: {len(result.get('recommendations', []))}개 상품 추천")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return {
                "analysis": "응답 파싱 오류",
                "recommendations": [],
                "summary": "LLM 응답을 파싱할 수 없습니다."
            }
        except Exception as e:
            logger.error(f"LLM RAG 검색 실패: {e}")
            return {
                "analysis": "검색 오류",
                "recommendations": [],
                "summary": f"검색 중 오류가 발생했습니다: {str(e)}"
            }
    
    def search_with_trend_analysis(self, trend_keywords: List[str], user_context: str = None) -> Dict[str, Any]:
        """트렌드 기반 LLM 검색"""
        
        trend_text = ", ".join(trend_keywords)
        base_query = f"현재 '{trend_text}' 트렌드에 맞는 상품을 추천해주세요."
        
        if user_context:
            full_query = f"{base_query} 추가 요구사항: {user_context}"
        else:
            full_query = base_query
        
        return self.search_with_llm(full_query, f"인기 키워드: {trend_text}")

# 사용 예시
if __name__ == "__main__":
    # 테스트 코드
    searcher = LLMRAGSearcher(os.getenv('OPENAI_API_KEY'))
    
    # 예시 1: 자연어 질문
    result1 = searcher.search_with_llm("여름철에 시원하게 지낼 수 있는 상품 추천해줘")
    print("=== 자연어 검색 결과 ===")
    print(json.dumps(result1, ensure_ascii=False, indent=2))
    
    # 예시 2: 트렌드 기반 검색
    result2 = searcher.search_with_trend_analysis(["다이어트", "건강"], "저녁 시간대 방송용")
    print("\n=== 트렌드 기반 검색 결과 ===")
    print(json.dumps(result2, ensure_ascii=False, indent=2))
