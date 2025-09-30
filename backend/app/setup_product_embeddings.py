#!/usr/bin/env python3
"""
상품 임베딩 초기화 스크립트 (증분 처리)
PostgreSQL의 신규/수정된 상품 데이터만 OpenAI로 임베딩하여 Qdrant에 저장
"""

import os
import sys
import uuid
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
from qdrant_client.models import PointStruct

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from product_embedder import ProductEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_products_to_embed(engine, force_all=False):
    """임베딩이 필요한 상품들을 조회"""
    
    if force_all:
        # 전체 재처리 모드 (방송테이프 있는 상품만)
        query = """
        SELECT
            g.product_code,
            g.product_name,
            g.category_main,
            g.category_middle,
            g.category_sub,
            t.tape_code,
            t.tape_name,
            g.created_at,
            g.updated_at,
            g.embedded_at
        FROM
            TAIGOODS g
        INNER JOIN
            taipgmtape t ON g.product_code = t.product_code
        WHERE t.production_status = 'ready'
        ORDER BY
            g.product_code
        """
        print("🔄 전체 방송테이프 재처리 모드 (방송테이프 있는 상품만)")
    else:
        # 증분 처리 모드: 신규 또는 수정된 방송테이프만
        query = """
        SELECT
            g.product_code,
            g.product_name,
            g.category_main,
            g.category_middle,
            g.category_sub,
            t.tape_code,
            t.tape_name,
            g.created_at,
            g.updated_at,
            g.embedded_at
        FROM
            TAIGOODS g
        INNER JOIN
            taipgmtape t ON g.product_code = t.product_code
        WHERE t.production_status = 'ready'
          AND (g.embedded_at IS NULL OR g.updated_at > g.embedded_at)
        ORDER BY g.product_code
        """
        print("🔄 증분 처리 모드: 신규/수정 방송테이프만")
    
    return pd.read_sql(query, engine)

def update_embedded_timestamp(engine, product_codes):
    """임베딩 완료된 상품들의 embedded_at 업데이트"""
    
    if not product_codes:
        return
    
    placeholders = ','.join([f"'{code}'" for code in product_codes])
    query = f"""
    UPDATE TAIGOODS 
    SET embedded_at = CURRENT_TIMESTAMP
    WHERE product_code IN ({placeholders})
    """
    
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()
    
    print(f"   ✅ {len(product_codes)}개 상품의 embedded_at 업데이트 완료")

def main():
    """상품 임베딩 초기화 실행"""
    
    print("🔄 상품 임베딩 초기화 시작...")
    
    # 환경변수 로드
    openai_api_key = os.getenv('OPENAI_API_KEY')
    db_uri = os.getenv('DB_URI')
    
    # 전체 재처리 여부 확인 (환경변수로 제어)
    force_all = os.getenv('FORCE_ALL_EMBEDDING', 'false').lower() == 'true'
    
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        return False
    
    if not db_uri:
        print("❌ 데이터베이스 URI가 설정되지 않았습니다.")
        return False
    
    try:
        # 1. PostgreSQL에서 임베딩 필요한 상품 데이터 조회
        print("1️⃣ 임베딩 필요한 상품 데이터 조회...")
        engine = create_engine(db_uri)
        
        products_df = get_products_to_embed(engine, force_all)
        
        if len(products_df) == 0:
            print("✅ 임베딩이 필요한 상품이 없습니다.")
            return True
        
        print(f"   📊 임베딩 대상: {len(products_df)}개 상품")
        
        # 상품 상태 분석
        new_products = products_df[products_df['embedded_at'].isna()]
        updated_products = products_df[products_df['embedded_at'].notna()]
        
        print(f"   - 신규 상품: {len(new_products)}개")
        print(f"   - 수정된 상품: {len(updated_products)}개")
        
        # 상품별 상세 정보 출력
        for _, row in products_df.iterrows():
            status = "신규" if pd.isna(row.get('embedded_at')) else "수정됨"
            category_info = f"{row.get('category_main', 'N/A')}"
            print(f"     - {row.get('product_code', 'Unknown')}: {row.get('product_name', 'Unknown')[:30]}... ({status})")
        
        # 2. ProductEmbedder 초기화
        print("2️⃣ ProductEmbedder 초기화...")
        embedder = ProductEmbedder(
            openai_api_key=openai_api_key,
            qdrant_host="qdrant_vector_db",  # Docker 컨테이너명
            qdrant_port=6333
        )
        
        # 3. Qdrant 컬렉션 설정 (신규 상품이 있을 때만)
        if len(new_products) > 0 or force_all:
            print("3️⃣ Qdrant 컬렉션 설정...")
            if force_all:
                embedder.setup_collection()  # 전체 재생성
            else:
                # 컬렉션이 없으면 생성
                try:
                    embedder.get_collection_info()
                except:
                    embedder.setup_collection()
        
        # 4. 상품 임베딩 및 인덱싱
        print("4️⃣ 상품 임베딩 및 인덱싱...")
        print("   ⚠️  OpenAI API 호출로 시간이 걸릴 수 있습니다...")
        
        # 배치 처리로 임베딩
        batch_size = 3
        processed_products = []
        
        for i in range(0, len(products_df), batch_size):
            batch_df = products_df.iloc[i:i+batch_size]
            print(f"   배치 {i//batch_size + 1}/{(len(products_df)-1)//batch_size + 1} 처리 중...")
            
            # 개별 상품 임베딩
            for idx, row in batch_df.iterrows():
                try:
                    # 방송테이프 정보 결합 (상품명 + 테이프명 + 카테고리)
                    product_name = str(row.get('product_name', ''))
                    tape_name = str(row.get('tape_name', ''))
                    category_main = str(row.get('category_main', ''))
                    category_middle = str(row.get('category_middle', ''))
                    
                    text = f"{product_name} {tape_name} {category_main} > {category_middle}".strip()
                    
                    if not text:
                        print(f"     ⚠️  빈 텍스트 건너뜀: {row.get('product_code', 'Unknown')}")
                        continue
                    
                    # OpenAI 임베딩 생성
                    embedding = embedder.get_embedding(text)
                    
                    # Qdrant에 저장/업데이트 (상품코드를 UUID로 변환하여 ID로 사용)
                    product_code = row['product_code']
                    # 네임스페이스 기반 UUID 생성으로 항상 동일한 상품코드는 동일한 UUID를 갖도록 보장
                    namespace_uuid = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                    point_id = str(uuid.uuid5(namespace_uuid, product_code))
                    point = {
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "product_code": str(row.get('product_code', '')),
                            "product_name": product_name,
                            "category_main": category_main,
                            "category_middle": category_middle,
                            "category_sub": category_sub,
                            "tape_code": str(row.get('tape_code', '')),
                            "tape_name": str(row.get('tape_name', '')),
                            "text": text,
                            "updated_at": datetime.now().isoformat()
                        }
                    }
                    
                    # 개별 업서트
                    embedder.qdrant_client.upsert(
                        collection_name=embedder.collection_name,
                        points=[PointStruct(**point)]
                    )
                    
                    processed_products.append(row['product_code'])
                    print(f"     ✅ {row['product_code']}: {product_name}")
                    
                except Exception as e:
                    print(f"     ❌ {row.get('product_code', 'Unknown')} 임베딩 실패: {e}")
                    continue
        
        # 5. PostgreSQL embedded_at 업데이트
        if processed_products:
            print("5️⃣ 임베딩 완료 상태 업데이트...")
            update_embedded_timestamp(engine, processed_products)
        
        # 6. 결과 확인
        print("6️⃣ 결과 확인...")
        collection_info = embedder.get_collection_info()
        
        print(f"\n✅ 상품 임베딩 완료!")
        print(f"   처리된 상품: {len(processed_products)}개")
        print(f"   전체 벡터 개수: {collection_info.get('vectors_count', 0)}개")
        print(f"   인덱싱된 벡터: {collection_info.get('indexed_vectors_count', 0)}개")
        
        # 7. 테스트 검색
        if processed_products:
            print("\n7️⃣ 테스트 검색...")
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
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 상품 임베딩 초기화 성공!")
    else:
        print("\n💥 상품 임베딩 초기화 실패!")
        sys.exit(1)
