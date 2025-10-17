#!/usr/bin/env python3
"""
벡터 검색 테스트 스크립트
"""
import sys
import os
sys.path.insert(0, '/app/app')

from product_embedder import ProductEmbedder

def main():
    # ProductEmbedder 초기화
    openai_api_key = os.getenv('OPENAI_API_KEY', 'sk-test-key')
    embedder = ProductEmbedder(
        openai_api_key=openai_api_key,
        qdrant_host='qdrant_vector_db',  # Docker 네트워크 내부 호스트명
        qdrant_port=6333
    )
    
    # 컬렉션 정보 확인
    print("📊 Qdrant 컬렉션 정보:")
    info = embedder.get_collection_info()
    print(f"   벡터 개수: {info.get('vectors_count', 0):,}개")
    print(f"   인덱싱 완료: {info.get('indexed_vectors_count', 0):,}개")
    print(f"   상태: {info.get('status', 'unknown')}")
    
    # 테스트 검색어
    test_queries = [
        '다이어트 건강식품',
        '겨울 패딩 점퍼',
        '무선 청소기',
        '스킨케어 화장품',
        '프라이팬 냄비',
    ]
    
    print("\n" + "="*60)
    for query in test_queries:
        print(f'\n🔍 검색어: "{query}"')
        print("-" * 60)
        
        try:
            results = embedder.search_products(query, top_k=5)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['product_name']}")
                    print(f"     📦 상품코드: {result.get('product_code', 'N/A')}")
                    print(f"     📂 카테고리: {result['category_main']} > {result['category_middle']}")
                    print(f"     🎯 유사도: {result['score']:.3f}")
                    if i < len(results):
                        print()
            else:
                print('  ❌ 검색 결과 없음')
        except Exception as e:
            print(f'  ❌ 검색 실패: {e}')
    
    print("\n" + "="*60)
    print("✅ 벡터 검색 테스트 완료!")

if __name__ == '__main__':
    main()
