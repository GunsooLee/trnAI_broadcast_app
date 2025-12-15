import openai
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import os
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class ProductEmbedder:
    def __init__(self, openai_api_key: str, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.qdrant_client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = "products"
        self.engine = create_engine(os.getenv("POSTGRES_URI"))
        
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
        total = len(products_df)
        print(f"=== 임베딩 시작: 총 {total}개 상품 ===")
        logger.info(f"총 {total}개 상품 임베딩 시작")
        
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
                        "price": float(row.get('price', 0)) if row.get('price') else 0,
                        "text": text,
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
                processed += 1
                
                # 진행 상황 출력 (10개마다)
                if processed % 10 == 0:
                    print(f"진행 중: {processed}/{total} ({processed*100//total}%)")
                
                # 배치 단위로 업로드
                if len(points) >= batch_size:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"=== 배치 업로드: {processed}개 완료 ({processed*100//total}%) ===")
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
    
    def search_products(
        self, 
        trend_keywords: List[str], 
        top_k: int = 10, 
        score_threshold: float = 0.7, 
        only_ready_products: bool = True
    ) -> List[Dict]:
        """
        트렌드 키워드로 관련 상품 검색 (방송 테이프 존재하는 상품만)
        
        Args:
            trend_keywords: 검색 키워드 리스트
            top_k: 반환할 상품 개수
            score_threshold: 유사도 임계값
            only_ready_products: 방송테이프 존재 상품만 필터링
        """
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
                # 디버깅: price 확인
                if 'price' not in product or product.get('price') is None:
                    logger.warning(f"⚠️ Qdrant 검색 결과에 price 없음: {product.get('product_name', 'Unknown')[:30]}")
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
    
    def get_all_products_with_tape(self, limit: int = 100) -> List[Dict]:
        """
        방송테이프가 있는 전체 상품 조회 (매출 예측용)
        
        Args:
            limit: 최대 조회 개수
            
        Returns:
            방송테이프가 있는 상품 리스트
        """
        query = """
        SELECT DISTINCT 
            g.product_code,
            g.product_name,
            g.category_main,
            g.category_middle,
            g.category_sub,
            g.brand,
            g.price,
            t.tape_code,
            t.tape_name
        FROM taigoods g
        INNER JOIN taipgmtape t ON g.product_code = t.product_code
        WHERE t.production_status = 'ready'
        ORDER BY g.product_code
        LIMIT :limit
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"limit": limit})
                products = []
                for row in result.fetchall():
                    products.append({
                        "product_code": row[0],
                        "product_name": row[1],
                        "category_main": row[2],
                        "category_middle": row[3],
                        "category_sub": row[4],
                        "brand": row[5],
                        "price": float(row[6]) if row[6] else 0,
                        "tape_code": row[7],
                        "tape_name": row[8],
                        "similarity_score": 0.0  # 키워드 매칭 없음
                    })
                
            logger.info(f"방송테이프 보유 상품 조회: {len(products)}개")
            return products
            
        except Exception as e:
            logger.error(f"방송테이프 상품 조회 실패: {e}")
            return []
    
    def get_category_avg_sales(self) -> Dict[str, float]:
        """
        카테고리별 평균 매출 조회 (신상품 매출 예측 보정용)
        
        Returns:
            카테고리별 평균 매출 딕셔너리
        """
        query = """
        SELECT 
            g.category_main,
            AVG(b.gross_profit) as avg_sales
        FROM taibroadcasts b
        JOIN taipgmtape t ON b.tape_code = t.tape_code
        JOIN taigoods g ON t.product_code = g.product_code
        WHERE b.gross_profit > 0
          AND b.broadcast_start_timestamp >= NOW() - INTERVAL '90 days'
        GROUP BY g.category_main
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                category_sales = {}
                for row in result.fetchall():
                    if row[0] and row[1]:
                        category_sales[row[0]] = float(row[1])
                
            logger.info(f"카테고리별 평균 매출 조회: {len(category_sales)}개 카테고리")
            return category_sales
            
        except Exception as e:
            logger.error(f"카테고리별 평균 매출 조회 실패: {e}")
            return {}
    
    def get_products_by_broadcast_period(
        self,
        start_date: date,
        end_date: date
    ) -> List[str]:
        """
        특정 기간에 방송된 상품코드 목록 조회
        
        Args:
            start_date: 시작일 (예: date(2024, 12, 1))
            end_date: 종료일 (예: date(2024, 12, 31))
            
        Returns:
            해당 기간에 방송된 상품코드 리스트
        """
        query = """
        SELECT DISTINCT g.product_code
        FROM taibroadcasts b
        JOIN taipgmtape t ON b.tape_code = t.tape_code
        JOIN taigoods g ON t.product_code = g.product_code
        WHERE b.broadcast_start_timestamp >= :start_date
          AND b.broadcast_start_timestamp < :end_date + INTERVAL '1 day'
        ORDER BY g.product_code
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"start_date": start_date, "end_date": end_date}
                )
                product_codes = [row[0] for row in result.fetchall()]
                
            logger.info(f"기간 {start_date} ~ {end_date} 방송 상품: {len(product_codes)}개")
            return product_codes
            
        except Exception as e:
            logger.error(f"방송 기간별 상품 조회 실패: {e}")
            return []
    
    def get_products_by_multiple_periods(
        self,
        periods: List[Tuple[date, date]]
    ) -> List[str]:
        """
        여러 기간에 방송된 상품코드 목록 조회 (합집합)
        
        Args:
            periods: (시작일, 종료일) 튜플 리스트
                예: [(date(2024, 10, 15), date(2024, 11, 15)),
                     (date(2023, 10, 15), date(2023, 11, 15)),
                     (date(2022, 10, 15), date(2022, 11, 15))]
            
        Returns:
            해당 기간들에 방송된 상품코드 리스트 (중복 제거)
            
        Example:
            # 최근 3년간 10월 15일 ~ 11월 15일 방송 상품 조회
            periods = [
                (date(2024, 10, 15), date(2024, 11, 15)),
                (date(2023, 10, 15), date(2023, 11, 15)),
                (date(2022, 10, 15), date(2022, 11, 15)),
            ]
            products = embedder.get_products_by_multiple_periods(periods)
        """
        if not periods:
            return []
        
        # 동적 SQL 생성: OR 조건으로 여러 기간 결합
        conditions = []
        params = {}
        
        for i, (start, end) in enumerate(periods):
            conditions.append(
                f"(b.broadcast_start_timestamp >= :start_{i} "
                f"AND b.broadcast_start_timestamp < :end_{i} + INTERVAL '1 day')"
            )
            params[f"start_{i}"] = start
            params[f"end_{i}"] = end
        
        query = f"""
        SELECT DISTINCT g.product_code
        FROM taibroadcasts b
        JOIN taipgmtape t ON b.tape_code = t.tape_code
        JOIN taigoods g ON t.product_code = g.product_code
        WHERE {' OR '.join(conditions)}
        ORDER BY g.product_code
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                product_codes = [row[0] for row in result.fetchall()]
            
            period_strs = [f"{s}~{e}" for s, e in periods]
            print(f"=== [다중 기간 필터] {len(periods)}개 기간 → 방송 상품: {len(product_codes)}개 ===", flush=True)
            logger.info(f"다중 기간 {period_strs} 방송 상품: {len(product_codes)}개")
            return product_codes
            
        except Exception as e:
            logger.error(f"다중 기간 상품 조회 실패: {e}")
            return []
    
    def get_same_period_across_years(
        self,
        start_month: int,
        start_day: int,
        end_month: int,
        end_day: int,
        years: List[int]
    ) -> List[str]:
        """
        여러 해의 동일 기간에 방송된 상품 조회 (편의 메서드)
        
        Args:
            start_month: 시작 월
            start_day: 시작 일
            end_month: 종료 월
            end_day: 종료 일
            years: 조회할 연도 리스트
            
        Returns:
            해당 기간들에 방송된 상품코드 리스트
            
        Example:
            # 2022~2024년 10월 15일 ~ 11월 15일 방송 상품
            products = embedder.get_same_period_across_years(
                start_month=10, start_day=15,
                end_month=11, end_day=15,
                years=[2022, 2023, 2024]
            )
        """
        periods = [
            (date(year, start_month, start_day), date(year, end_month, end_day))
            for year in years
        ]
        return self.get_products_by_multiple_periods(periods)
    
    def search_products_with_broadcast_filter(
        self,
        trend_keywords: List[str],
        broadcast_start_date: Optional[date] = None,
        broadcast_end_date: Optional[date] = None,
        broadcast_periods: Optional[List[Tuple[date, date]]] = None,
        top_k: int = 10,
        score_threshold: float = 0.5,
        only_ready_products: bool = True
    ) -> List[Dict]:
        """
        방송 기간 필터링 + 벡터 검색 통합 메서드
        
        Args:
            trend_keywords: 검색 키워드 리스트
            broadcast_start_date: 방송 시작일 필터 (단일 기간용, None이면 필터 안함)
            broadcast_end_date: 방송 종료일 필터 (단일 기간용, None이면 필터 안함)
            broadcast_periods: 여러 기간 필터 [(시작일, 종료일), ...] (다중 기간용)
            top_k: 반환할 상품 개수
            score_threshold: 유사도 임계값
            only_ready_products: 방송테이프 존재 상품만 필터링
            
        Returns:
            필터링된 상품 리스트 (유사도 점수 포함)
            
        Example:
            # 단일 기간: 2024년 12월에 방송한 상품 중 "다이어트" 관련 검색
            results = embedder.search_products_with_broadcast_filter(
                trend_keywords=["다이어트", "건강"],
                broadcast_start_date=date(2024, 12, 1),
                broadcast_end_date=date(2024, 12, 31),
                top_k=5
            )
            
            # 다중 기간: 최근 3년간 10월~11월 방송 상품 중 검색
            results = embedder.search_products_with_broadcast_filter(
                trend_keywords=["다이어트", "건강"],
                broadcast_periods=[
                    (date(2024, 10, 15), date(2024, 11, 15)),
                    (date(2023, 10, 15), date(2023, 11, 15)),
                    (date(2022, 10, 15), date(2022, 11, 15)),
                ],
                top_k=5
            )
        """
        if not trend_keywords:
            return []
        
        try:
            # 1단계: 방송 기간 필터가 있으면 PostgreSQL에서 상품코드 목록 조회
            allowed_product_codes = None
            filter_description = ""
            
            # 다중 기간 우선
            if broadcast_periods:
                allowed_product_codes = self.get_products_by_multiple_periods(broadcast_periods)
                filter_description = f"{len(broadcast_periods)}개 기간"
            # 단일 기간
            elif broadcast_start_date and broadcast_end_date:
                allowed_product_codes = self.get_products_by_broadcast_period(
                    broadcast_start_date, broadcast_end_date
                )
                filter_description = f"{broadcast_start_date}~{broadcast_end_date}"
            
            if allowed_product_codes is not None and not allowed_product_codes:
                logger.info(f"기간 필터({filter_description})에 해당하는 방송 상품 없음")
                return []
            
            if allowed_product_codes:
                logger.info(f"방송 기간 필터({filter_description}): {len(allowed_product_codes)}개 상품 후보")
            
            # 2단계: Qdrant 벡터 검색 (필터 조건 포함)
            query_text = " ".join(trend_keywords)
            query_embedding = self.get_embedding(query_text)
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
            
            # Qdrant 필터 조건 구성
            filter_conditions = []
            
            # 방송테이프 존재 필터
            if only_ready_products:
                filter_conditions.append(
                    FieldCondition(key="tape_code", match=MatchValue(value=""))
                )
            
            # 방송 기간 필터 (상품코드 목록으로 필터링)
            must_conditions = []
            if allowed_product_codes:
                must_conditions.append(
                    FieldCondition(
                        key="product_code",
                        match=MatchAny(any=allowed_product_codes)
                    )
                )
            
            # 필터 구성
            qdrant_filter = None
            if filter_conditions or must_conditions:
                qdrant_filter = Filter(
                    must=must_conditions if must_conditions else None,
                    must_not=filter_conditions if filter_conditions and only_ready_products else None
                )
            
            # Qdrant 검색 실행
            # 방송 기간 필터가 있으면 더 많은 결과를 가져와서 필터링
            search_limit = top_k * 3 if allowed_product_codes else top_k
            
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=search_limit,
                score_threshold=score_threshold
            )
            
            # 3단계: 결과 포맷팅
            filtered_products = []
            for hit in results:
                product = hit.payload.copy()
                product['similarity_score'] = hit.score
                
                # 방송 기간 필터 적용 (Qdrant 필터가 작동하지 않을 경우 대비)
                if allowed_product_codes:
                    if product.get('product_code') not in allowed_product_codes:
                        continue
                
                filtered_products.append(product)
                
                if len(filtered_products) >= top_k:
                    break
            
            # 로그
            filter_info = []
            if broadcast_periods:
                filter_info.append(f"방송기간: {len(broadcast_periods)}개 기간")
            elif broadcast_start_date and broadcast_end_date:
                filter_info.append(f"방송기간: {broadcast_start_date}~{broadcast_end_date}")
            if only_ready_products:
                filter_info.append("방송테이프 존재")
            filter_msg = f"({', '.join(filter_info)})" if filter_info else ""
            
            logger.info(f"키워드 '{query_text}'로 {len(filtered_products)}개 상품 검색됨 {filter_msg}")
            return filtered_products
            
        except Exception as e:
            logger.error(f"방송 기간 필터 검색 실패: {e}")
            return []
    
    def get_broadcast_history(
        self,
        product_code: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        특정 상품의 방송 이력 조회
        
        Args:
            product_code: 상품코드
            limit: 최대 조회 건수
            
        Returns:
            방송 이력 리스트 (최신순)
        """
        query = """
        SELECT 
            b.broadcast_start_timestamp,
            b.product_is_new,
            b.gross_profit,
            b.sales_efficiency,
            t.tape_code,
            t.tape_name
        FROM taibroadcasts b
        JOIN taipgmtape t ON b.tape_code = t.tape_code
        WHERE t.product_code = :product_code
        ORDER BY b.broadcast_start_timestamp DESC
        LIMIT :limit
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"product_code": product_code, "limit": limit}
                )
                
                history = []
                for row in result.fetchall():
                    history.append({
                        "broadcast_timestamp": row[0].isoformat() if row[0] else None,
                        "is_new": row[1],
                        "gross_profit": float(row[2]) if row[2] else 0,
                        "sales_efficiency": float(row[3]) if row[3] else 0,
                        "tape_code": row[4],
                        "tape_name": row[5]
                    })
                
            return history
            
        except Exception as e:
            logger.error(f"방송 이력 조회 실패 ({product_code}): {e}")
            return []
    
    def get_same_period_last_year_products(
        self,
        target_month: int,
        target_year: Optional[int] = None
    ) -> List[str]:
        """
        작년 동월에 방송된 상품코드 목록 조회 (편의 메서드)
        
        Args:
            target_month: 대상 월 (1-12)
            target_year: 대상 연도 (None이면 작년)
            
        Returns:
            해당 기간에 방송된 상품코드 리스트
            
        Example:
            # 작년 12월에 방송한 상품 조회
            products = embedder.get_same_period_last_year_products(12)
        """
        if target_year is None:
            target_year = datetime.now().year - 1
        
        start_date = date(target_year, target_month, 1)
        
        # 다음 달 1일 계산
        if target_month == 12:
            end_date = date(target_year + 1, 1, 1)
        else:
            end_date = date(target_year, target_month + 1, 1)
        
        # 마지막 날로 조정
        end_date = end_date - timedelta(days=1)
        
        return self.get_products_by_broadcast_period(start_date, end_date)
