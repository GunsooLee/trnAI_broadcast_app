#!/usr/bin/env python3
"""
상품 임베딩 초기화 스크립트
PostgreSQL의 상품 데이터를 OpenAI로 임베딩하여 Qdrant에 저장
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
import logging

# 백엔드 앱 모듈 추가
sys.path.append('/app')
from app.product_embedder import ProductEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """상품 임베딩 초기화 실행"""
    
    print("🔄 상품 임베딩 초기화 시작...")
    
    # 환경변수 로드
    openai_api_key = os.getenv('OPENAI_API_KEY')
    db_uri = os.getenv('DB_URI')
    
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        return False
    
    if not db_uri:
        print("❌ 데이터베이스 URI가 설정되지 않았습니다.")
        return False
    
    try:
        # 1. PostgreSQL에서 상품 데이터 조회
        print("1️⃣ PostgreSQL에서 상품 데이터 조회...")
        engine = create_engine(db_uri)
        
        query = """
        SELECT 
            product_code,
            product_name,
            product_mgroup,
            keyword
        FROM TAIGOODS
        ORDER BY product_code
        """
        
        products_df = pd.read_sql(query, engine)
        print(f"   📊 총 {len(products_df)}개 상품 조회됨")
        
        if len(products_df) == 0:
            print("❌ 상품 데이터가 없습니다.")
            return False
        
        # 2. ProductEmbedder 초기화
        print("2️⃣ ProductEmbedder 초기화...")
        embedder = ProductEmbedder(
            openai_api_key=openai_api_key,
            qdrant_host="qdrant_vector_db",  # Docker 컨테이너명
            qdrant_port=6333
        )
        
        # 3. Qdrant 컬렉션 설정
        print("3️⃣ Qdrant 컬렉션 설정...")
        embedder.setup_collection()
        
        # 4. 상품 임베딩 및 인덱싱
        print("4️⃣ 상품 임베딩 및 인덱싱...")
        processed_count = embedder.build_product_index(products_df, batch_size=5)
        
        # 5. 결과 확인
        print("5️⃣ 결과 확인...")
        collection_info = embedder.get_collection_info()
        
        print(f"\n✅ 상품 임베딩 완료!")
        print(f"   처리된 상품: {processed_count}개")
        print(f"   벡터 개수: {collection_info.get('vectors_count', 0)}개")
        print(f"   인덱싱된 벡터: {collection_info.get('indexed_vectors_count', 0)}개")
        
        # 6. 테스트 검색
        print("\n6️⃣ 테스트 검색...")
        test_keywords = ["다이어트", "건강식품", "스킨케어"]
        
        for keyword in test_keywords:
            results = embedder.search_products([keyword], top_k=3, score_threshold=0.5)
            print(f"   '{keyword}' 검색 결과: {len(results)}개")
            for i, product in enumerate(results[:2]):
                print(f"     {i+1}. {product['product_name']} (유사도: {product['similarity_score']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"상품 임베딩 초기화 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 상품 임베딩 초기화 성공!")
    else:
        print("\n💥 상품 임베딩 초기화 실패!")
        sys.exit(1)
