"""
Netezza 데이터베이스 연결 설정
"""
import os
import asyncio
import nzpy
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg

logger = logging.getLogger(__name__)

class NetezzaConnection:
    def __init__(self):
        self.host = os.getenv("NETEZZA_HOST", "localhost")
        self.port = os.getenv("NETEZZA_PORT", "5480")
        self.database = os.getenv("NETEZZA_DATABASE", "SYSTEM")
        self.username = os.getenv("NETEZZA_USER", "admin")
        self.password = os.getenv("NETEZZA_PASSWORD", "password")
        self.connection_params = self._get_connection_params()

    def _get_connection_params(self) -> dict:
        """Netezza nzpy 연결 파라미터 생성"""
        return {
            'host': self.host,
            'port': int(self.port),
            'database': self.database,
            'user': self.username,
            'password': self.password,
        }

    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """비동기적으로 Netezza 쿼리 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync_query, query, params)

    def _execute_sync_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """동기적으로 Netezza 쿼리 실행"""
        try:
            with nzpy.connect(**self.connection_params) as conn:
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # 컬럼 이름 가져오기 (소문자로 변환)
                columns = [desc[0].lower() for desc in cursor.description]

                # 결과를 딕셔너리 리스트로 변환
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))

                logger.info(f"Netezza query executed successfully. Returned {len(results)} rows.")
                return results

        except Exception as e:
            logger.error(f"Netezza query failed: {str(e)}")
            raise Exception(f"Netezza 데이터베이스 연결 실패: {str(e)}")

    async def get_all_broadcast_tapes(self) -> List[Dict[str, Any]]:
        """모든 방송테이프 정보를 가져옵니다"""
        query = """
        SELECT
            tape_id,
            product_code,
            product_name,
            category,
            broadcast_date,
            broadcast_time,
            duration_minutes,
            status,
            created_at,
            updated_at
        FROM DST_PROG_TAPE
        ORDER BY created_at DESC
        """

        return await self.execute_query(query)

    async def get_competitor_schedules(self, broadcast_time: str, limit: int = 10) -> List[Dict[str, Any]]:
        """특정 시간대의 타사 편성 정보를 가져옵니다
        
        Args:
            broadcast_time: 방송 시간 (ISO 8601 형식, 예: "2025-09-15T22:40:00+09:00")
            limit: 최대 조회 개수 (기본값: 10)
        
        Returns:
            타사 편성 정보 리스트
        """
        # ISO 8601 형식을 Netezza TIMESTAMP 형식으로 변환
        # 예: "2025-09-15T22:40:00+09:00" -> "2025-09-15 22:40:00"
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(broadcast_time)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.error(f"Invalid broadcast_time format: {broadcast_time}, error: {str(e)}")
            formatted_time = broadcast_time.replace('T', ' ').split('+')[0]
        
        query = """
        SELECT
            CMPNY_NM as company_name,
            BDCAST_SUBJ as broadcast_title,
            STRT_DTTM as start_time,
            END_DTTM as end_time,
            BDCAST_MIN as duration_minutes,
            LCLS_CTGR as category_main
        FROM SNTDM.SNTADM.FBD_OTENT_RST_ANAL_D_NEW
        WHERE STRT_DTTM <= ?
          AND END_DTTM >= ?
        ORDER BY STRT_DTTM
        LIMIT ?
        """
        
        logger.info(f"Fetching competitor schedules for time: {formatted_time}")
        return await self.execute_query(query, (formatted_time, formatted_time, limit))

    async def upsert_tapes_to_postgres(self, tapes: List) -> int:
        """PostgreSQL TAIPGMTAPE 테이블에 방송테이프 정보를 Upsert합니다"""
        if not tapes:
            return 0

        # PostgreSQL 연결 정보 가져오기
        db_uri = os.getenv("DB_URI", "postgresql://TRN_AI:TRN_AI@localhost:5432/TRNAI_DB")

        try:
            # asyncpg를 사용한 PostgreSQL 연결
            conn = await asyncpg.connect(db_uri)

            upserted_count = 0

            for tape in tapes:
                # TAIPGMTAPE 테이블 스키마에 맞게 매핑
                # Netezza의 tape_id를 PostgreSQL의 tape_code로 사용
                tape_code = tape.tape_id
                tape_name = tape.product_name  # 또는 적절한 테이프명
                product_code = tape.product_code
                production_status = 'ready'  # 기본값

                # UPSERT 쿼리 (ON CONFLICT DO UPDATE)
                upsert_query = """
                INSERT INTO TAIPGMTAPE (tape_code, tape_name, product_code, production_status, updated_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT (tape_code)
                DO UPDATE SET
                    tape_name = EXCLUDED.tape_name,
                    product_code = EXCLUDED.product_code,
                    production_status = EXCLUDED.production_status,
                    updated_at = CURRENT_TIMESTAMP
                """

                await conn.execute(upsert_query, tape_code, tape_name, product_code, production_status)
                upserted_count += 1

            await conn.close()
            logger.info(f"Successfully upserted {upserted_count} tapes to TAIPGMTAPE")
            return upserted_count

        except Exception as e:
            logger.error(f"PostgreSQL upsert failed: {str(e)}")
            raise Exception(f"PostgreSQL 데이터 저장 실패: {str(e)}")

# 전역 인스턴스
netezza_conn = NetezzaConnection()