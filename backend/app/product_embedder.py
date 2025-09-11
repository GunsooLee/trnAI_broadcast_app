import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class ProductEmbedder:
    def __init__(self, openai_api_key: str, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.qdrant_client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = "products"
        self.engine = create_engine(os.getenv("POSTGRES_URI", os.getenv("DB_URI")))
        
    def setup_collection(self):
        """컬렉션 초기화 (기존 컬렉션이 있으면 삭제 후 재생성)"""
        try:
            # 기존 컬렉션 확인 및 삭제
            collections = self.qdrant_client.get_collections()
            if any(col.name == self.collection_name for col in collections.collections):
                self.qdrant_client.delete_collection(self.collection_name)
                logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제됨")
            
            # 새 컬렉션 생성
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # OpenAI embedding 차원
            )
            logger.info(f"컬렉션 '{self.collection_name}' 생성 완료")
            
        except Exception as e:
            logger.error(f"컬렉션 설정 실패: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """OpenAI API로 임베딩 생성"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",  # 비용 효율적
                input=text.strip()
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {text[:50]}... - {e}")
            raise
    
    def build_product_index(self, products_df: pd.DataFrame, batch_size: int = 100):
        """상품 데이터를 임베딩하여 Qdrant에 저장"""
        logger.info(f"총 {len(products_df)}개 상품 임베딩 시작")
        
        points = []
        processed = 0
        
        for idx, row in products_df.iterrows():
            try:
                # 상품 정보 결합 (None 값 처리)
                product_name = str(row.get('product_name', ''))
                keyword = str(row.get('keyword', ''))
                category = str(row.get('product_mgroup', ''))
                
                text = f"{product_name} {keyword} {category}".strip()
                
                if not text:
                    logger.warning(f"빈 텍스트 건너뜀: {row.get('product_code', 'Unknown')}")
                    continue
                
                # OpenAI 임베딩 생성
                embedding = self.get_embedding(text)
                
                point = PointStruct(
                    id=processed,
                    vector=embedding,
                    payload={
                        "product_code": str(row.get('product_code', '')),
                        "product_name": product_name,
                        "category": category,
                        "keyword": keyword,
                        "text": text,
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
                processed += 1
                
                # 배치 단위로 업로드
                if len(points) >= batch_size:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"{processed}개 상품 임베딩 완료")
                    points = []
                    
            except Exception as e:
                logger.error(f"상품 임베딩 실패 (idx: {idx}): {e}")
                continue
        
        # 남은 포인트들 업로드
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(f"전체 임베딩 완료: {processed}개 상품")
        return processed
    
    def search_products(self, trend_keywords: List[str], top_k: int = 10, score_threshold: float = 0.7, only_ready_products: bool = True) -> List[Dict]:
        """트렌드 키워드로 관련 상품 검색 (방송 테이프 존재하는 상품만)"""
        if not trend_keywords:
            return []
        
        try:
            query_text = " ".join(trend_keywords)
            query_embedding = self.get_embedding(query_text)
            
            # Qdrant에서 벡터 검색 (필터링 없이)
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k * 3,  # 여유분 확보
                score_threshold=score_threshold
            )
            
            # 검색된 상품 코드들 추출
            product_codes = [hit.payload.get('product_code') for hit in results if hit.payload.get('product_code')]
            
            if not product_codes or not only_ready_products:
                # 방송테이프 필터링이 불필요한 경우
                products = []
                for hit in results[:top_k]:
                    product = hit.payload.copy()
                    product['similarity_score'] = hit.score
                    products.append(product)
                return products
            
            # TPGMTAPE 테이블과 조인하여 방송테이프 존재 여부 확인
            placeholders = ','.join([f"'{code}'" for code in product_codes])
            query = f"""
            SELECT DISTINCT p.product_code, p.product_name, p.category_main, p.category_middle, p.category_sub,
                   p.search_keywords, t.tape_code, t.tape_name, t.duration_minutes
            FROM TAIGOODS p
            INNER JOIN TAIPGMTAPE t ON p.product_code = t.product_code
            WHERE p.product_code IN ({placeholders})
              AND t.production_status = 'ready'
            """
            
            with self.engine.connect() as conn:
                db_results = conn.execute(text(query)).fetchall()
            
            # 방송테이프가 있는 상품 코드 세트
            tape_product_codes = {row[0] for row in db_results}
            
            # 벡터 검색 결과를 방송테이프 존재 여부로 필터링
            filtered_products = []
            for hit in results:
                product_code = hit.payload.get('product_code')
                if product_code in tape_product_codes:
                    product = hit.payload.copy()
                    product['similarity_score'] = hit.score
                    
                    # 방송테이프 정보 추가
                    tape_info = next((row for row in db_results if row[0] == product_code), None)
                    if tape_info:
                        product['tape_code'] = tape_info[6]
                        product['tape_name'] = tape_info[7]
                        product['duration_minutes'] = tape_info[8]
                    
                    filtered_products.append(product)
                    
                if len(filtered_products) >= top_k:
                    break
            
            filter_msg = "(방송테이프 존재)" if only_ready_products else ""
            logger.info(f"키워드 '{query_text}'로 {len(filtered_products)}개 상품 검색됨 {filter_msg}")
            return filtered_products
            
        except Exception as e:
            logger.error(f"상품 검색 실패: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": info.config.params.name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
