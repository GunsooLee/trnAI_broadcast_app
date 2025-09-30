import openai
import uuid
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
                category_main = str(row.get('category_main', ''))
                category_middle = str(row.get('category_middle', ''))
                category_sub = str(row.get('category_sub', ''))
                brand = str(row.get('brand', ''))
                
                # 임베딩 텍스트: 상품명 + 카테고리 + 브랜드
                text_parts = [product_name, category_main, category_middle, category_sub, brand]
                text = " ".join([part for part in text_parts if part and part != 'nan']).strip()
                
                if not text:
                    logger.warning(f"빈 텍스트 건너뜀: {row.get('product_code', 'Unknown')}")
                    continue
                
                # OpenAI 임베딩 생성
                embedding = self.get_embedding(text)
                
                product_code = str(row.get('product_code', ''))
                namespace_uuid = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                point_id = str(uuid.uuid5(namespace_uuid, product_code))

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "product_code": product_code,
                        "product_name": product_name,
                        "category_main": category_main,
                        "category_middle": category_middle,
                        "category_sub": category_sub,
                        "brand": brand,
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
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Qdrant 필터 조건 생성 (방송테이프 코드가 있는 상품만)
            qdrant_filter = None
            if only_ready_products:
                qdrant_filter = Filter(
                    must_not=[
                        FieldCondition(key="payload.tape_code", match=MatchValue(value=""))
                    ]
                )

            # Qdrant에서 벡터 검색
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # 결과 포맷팅
            filtered_products = []
            for hit in results:
                product = hit.payload.copy()
                product['similarity_score'] = hit.score
                filtered_products.append(product)
            
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
