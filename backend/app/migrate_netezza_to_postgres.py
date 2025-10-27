#!/usr/bin/env python3
"""
NETEZZA → PostgreSQL 데이터 마이그레이션 스크립트 (범용)
모든 테이블을 설정 파일 기반으로 자동 마이그레이션
"""

import os
import sys
import pandas as pd
try:
    import nzpy  # NETEZZA 연결용 (Pure Python, ODBC 불필요)
    USE_NZPY = True
except ImportError:
    import pyodbc  # ODBC 대체
    USE_NZPY = False
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
from dotenv import load_dotenv

# 설정 파일 임포트
try:
    from migrate_tables_config import get_enabled_tables, get_table_config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from migrate_tables_config import get_enabled_tables, get_table_config

# 환경변수 로드
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def connect_netezza():
    """NETEZZA 데이터베이스 연결 (nzpy 또는 pyodbc)"""
    # 환경변수에서 NETEZZA 연결 정보 가져오기
    netezza_host = os.getenv('NETEZZA_HOST', 'your_netezza_host')
    netezza_port = int(os.getenv('NETEZZA_PORT', '5480'))
    netezza_db = os.getenv('NETEZZA_DATABASE', 'your_database')
    netezza_user = os.getenv('NETEZZA_USER', 'your_username')
    netezza_pwd = os.getenv('NETEZZA_PASSWORD', 'your_password')
    
    try:
        logger.info(f"🔌 NETEZZA 연결 시도: {netezza_host}:{netezza_port}/{netezza_db}")
        
        if USE_NZPY:
            # nzpy 사용 (Pure Python, ODBC 불필요)
            logger.info("   연결 방식: nzpy (Pure Python)")
            conn = nzpy.connect(
                host=netezza_host,
                port=netezza_port,
                database=netezza_db,
                user=netezza_user,
                password=netezza_pwd,
                securityLevel=0,  # 보안 수준 (0=비암호화)
                logLevel=0
            )
            logger.info("✅ NETEZZA 연결 성공 (nzpy)")
        else:
            # pyodbc 사용 (ODBC 드라이버 필요)
            logger.info("   연결 방식: pyodbc (ODBC)")
            netezza_driver = os.getenv('NETEZZA_DRIVER', 'NetezzaSQL')
            conn_string = (
                f"DRIVER={{{netezza_driver}}};"
                f"SERVER={netezza_host};"
                f"PORT={netezza_port};"
                f"DATABASE={netezza_db};"
                f"UID={netezza_user};"
                f"PWD={netezza_pwd};"
            )
            conn = pyodbc.connect(conn_string)
            logger.info("✅ NETEZZA 연결 성공 (pyodbc)")
        
        return conn
        
    except Exception as e:
        logger.error(f"❌ NETEZZA 연결 실패: {e}")
        logger.error(f"   호스트: {netezza_host}:{netezza_port}")
        logger.error(f"   데이터베이스: {netezza_db}")
        logger.error(f"   사용자: {netezza_user}")
        raise


def connect_postgres():
    """PostgreSQL 데이터베이스 연결"""
    # 환경변수에서 DB URI 가져오기
    db_uri = os.getenv('DB_URI') or os.getenv('POSTGRES_URI')
    
    # Docker 컨테이너 내부에서는 호스트명이 다를 수 있음
    if not db_uri:
        db_uri = "postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB"
    
    try:
        engine = create_engine(db_uri)
        logger.info("✅ PostgreSQL 연결 성공")
        return engine
    except Exception as e:
        logger.error(f"❌ PostgreSQL 연결 실패: {e}")
        raise


def extract_table_from_netezza(netezza_conn, table_name, incremental=True):
    """NETEZZA에서 테이블 데이터 추출 (범용)
    
    Args:
        netezza_conn: NETEZZA 데이터베이스 연결
        table_name: 테이블명 (예: TAIGOODS, TAIPGMTAPE)
        incremental: True면 어제 이후 수정된 데이터만, False면 전체 데이터
    """
    
    # 테이블 설정 가져오기
    config = get_table_config(table_name)
    if not config:
        logger.error(f"❌ 테이블 설정을 찾을 수 없음: {table_name}")
        return None
    
    # 쿼리 생성 (설정 파일의 람다 함수 실행)
    query = config["query"](incremental)
    
    # 로그 출력
    mode = f"증분 업데이트" if incremental else "전체 데이터"
    logger.info(f"📥 NETEZZA에서 {table_name} 추출 중 ({mode})...")
    logger.debug(f"   쿼리: {query[:200]}...")
    
    try:
        df = pd.read_sql(query, netezza_conn)
        logger.info(f"   ✅ 추출 완료: {len(df)}개 레코드")
        return df
    except Exception as e:
        logger.error(f"   ❌ {table_name} 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def upsert_to_postgres(df, table_name, postgres_engine, key_column='product_code'):
    """PostgreSQL에 데이터 UPSERT (존재하면 업데이트, 없으면 삽입)
    
    Args:
        df: 적재할 데이터프레임
        table_name: 테이블명 (taigoods 또는 taipgmtape)
        postgres_engine: PostgreSQL 엔진
        key_column: Primary Key 컬럼명
    """
    
    logger.info(f"📤 PostgreSQL {table_name} 테이블에 UPSERT 중...")
    
    if len(df) == 0:
        logger.warning("   ⚠️  적재할 데이터 없음")
        return True
    
    try:
        from sqlalchemy import text
        
        # 임시 테이블에 데이터 적재 (PostgreSQL은 소문자로 통일)
        temp_table = f"{table_name.lower()}_temp"
        df.to_sql(
            temp_table,
            postgres_engine,
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=1000
        )
        
        # UPSERT 쿼리 실행 (ON CONFLICT DO UPDATE)
        if table_name.lower() == 'taigoods':
            upsert_query = f"""
            INSERT INTO taigoods (product_code, product_name, category_main, category_middle, 
                                  category_sub, price, brand, product_type, created_at, updated_at)
            SELECT product_code, product_name, category_main, category_middle, 
                   category_sub, price::NUMERIC, brand, product_type, 
                   created_at::TIMESTAMP, CURRENT_TIMESTAMP
            FROM {temp_table}
            ON CONFLICT (product_code) DO UPDATE SET
                product_name = EXCLUDED.product_name,
                category_main = EXCLUDED.category_main,
                category_middle = EXCLUDED.category_middle,
                category_sub = EXCLUDED.category_sub,
                price = EXCLUDED.price,
                brand = EXCLUDED.brand,
                product_type = EXCLUDED.product_type,
                updated_at = CURRENT_TIMESTAMP
            """
        elif table_name.lower() == 'taipgmtape':
            upsert_query = f"""
            INSERT INTO taipgmtape (tape_code, tape_name, product_code, production_status, 
                                    created_at, updated_at)
            SELECT tape_code, tape_name, product_code, production_status, 
                   created_at::TIMESTAMP, CURRENT_TIMESTAMP
            FROM {temp_table}
            ON CONFLICT (tape_code) DO UPDATE SET
                tape_name = EXCLUDED.tape_name,
                product_code = EXCLUDED.product_code,
                production_status = EXCLUDED.production_status,
                updated_at = CURRENT_TIMESTAMP
            """
        elif table_name.lower() == 'taibroadcasts':
            # TAIPGMTAPE에 존재하는 tape_code만 삽입 (Foreign Key 제약 준수)
            upsert_query = f"""
            INSERT INTO taibroadcasts (tape_code, broadcast_start_timestamp, product_is_new, 
                                       gross_profit, sales_efficiency, created_at)
            SELECT t.tape_code, 
                   t.broadcast_start_timestamp::TIMESTAMP, 
                   CASE WHEN t.product_is_new IS NULL THEN FALSE 
                        ELSE t.product_is_new::BOOLEAN 
                   END, 
                   t.gross_profit::NUMERIC, 
                   t.sales_efficiency::NUMERIC, 
                   t.created_at::TIMESTAMP
            FROM {temp_table} t
            WHERE EXISTS (SELECT 1 FROM taipgmtape WHERE tape_code = t.tape_code)
            """
        elif table_name.lower() == 'taicompetitor_broadcasts':
            # 경쟁사 방송 정보 UPSERT
            # 1. 고유 인덱스 생성 (없으면 생성)
            create_index_query = """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_taicomp_brdcst_date_slot_comp 
            ON taicompetitor_broadcasts (broadcast_date, time_slot, competitor_name)
            """
            with postgres_engine.connect() as conn:
                conn.execute(text(create_index_query))
                conn.commit()
            
            # 2. UPSERT 실행
            upsert_query = f"""
            INSERT INTO taicompetitor_broadcasts (broadcast_date, time_slot, competitor_name, 
                                                   category_main, category_middle, created_at)
            SELECT broadcast_date::DATE, 
                   time_slot, 
                   competitor_name, 
                   category_main, 
                   category_middle,
                   CURRENT_TIMESTAMP
            FROM {temp_table}
            ON CONFLICT (broadcast_date, time_slot, competitor_name)
            DO UPDATE SET
                category_main = EXCLUDED.category_main,
                category_middle = EXCLUDED.category_middle,
                created_at = CURRENT_TIMESTAMP
            """
        else:
            raise ValueError(f"지원하지 않는 테이블: {table_name}")
        
        with postgres_engine.connect() as conn:
            conn.execute(text(upsert_query))
            conn.commit()
        
        # 임시 테이블 삭제
        with postgres_engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
        
        logger.info(f"   ✅ {len(df)}개 레코드 UPSERT 완료")
        return True
    except Exception as e:
        logger.error(f"   ❌ UPSERT 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_time_slot(start_time, end_time):
    """시작/종료 시간을 time_slot으로 변환
    
    Args:
        start_time: 시작 시간 (예: "2025-10-15 14:30:00.000")
        end_time: 종료 시간 (예: "2025-10-15 15:30:00.000")
    
    Returns:
        time_slot: 시간대 문자열 (예: "14:00-16:00")
    """
    try:
        # 문자열을 datetime으로 변환
        if pd.isna(start_time) or pd.isna(end_time):
            return "00:00-01:00"
        
        start_str = str(start_time)
        end_str = str(end_time)
        
        # 시간 추출 (여러 형식 대응)
        if len(start_str) >= 19:  # "2025-10-15 14:30:00" 형식
            start_hour = int(start_str[11:13])
            end_hour = int(end_str[11:13])
            end_minute = int(end_str[14:16])
            
            # 종료 시간이 정각이 아니면 다음 시간으로 올림
            if end_minute > 0:
                end_hour += 1
            
            # 24시를 넘어가는 경우 처리
            if end_hour > 24:
                end_hour = 24
            
            return f"{start_hour:02d}:00-{end_hour:02d}:00"
        else:
            return "00:00-01:00"
            
    except Exception as e:
        logger.warning(f"시간 변환 실패: {start_time} ~ {end_time}, 에러: {e}")
        return "00:00-01:00"


def data_cleansing(df, table_type='products'):
    """데이터 정제"""
    
    logger.info("🧹 데이터 정제 중...")
    
    # NETEZZA는 컬럼명을 대문자로 반환하므로 소문자로 변환
    df.columns = df.columns.str.lower()
    
    if table_type == 'products':
        # NULL 처리
        df['product_name'] = df['product_name'].fillna('')
        df['brand'] = df['brand'].fillna('미지정')
        df['price'] = df['price'].fillna(0)
        
        # 데이터 타입 변환
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        
        # 상품코드 중복 제거 (최신 데이터 우선)
        df = df.drop_duplicates(subset=['product_code'], keep='last')
        
        # 빈 상품코드 제거
        df = df[df['product_code'].notna()]
        df = df[df['product_code'] != '']
        
        logger.info(f"   정제 후: {len(df)}개 상품")
    
    elif table_type == 'competitors':
        # 경쟁사 방송 데이터 정제
        logger.info("   경쟁사 방송 데이터 정제 중...")
        
        # 컬럼 리네임
        df = df.rename(columns={
            'bdcast_dt': 'broadcast_date',
            'strt_dttm': 'start_time',
            'end_dttm': 'end_time',
            'cmpny_nm': 'competitor_name',
            'lcls_ctgr': 'category_main',
            'mcls_ctgr': 'category_middle'
        })
        
        # time_slot 계산
        df['time_slot'] = df.apply(
            lambda row: convert_time_slot(row.get('start_time'), row.get('end_time')), 
            axis=1
        )
        
        # NULL 처리
        df['competitor_name'] = df['competitor_name'].fillna('미지정')
        df['category_main'] = df['category_main'].fillna('기타')
        df['category_middle'] = df['category_middle'].fillna('기타')
        
        # 필요한 컬럼만 선택
        df = df[['broadcast_date', 'time_slot', 'competitor_name', 'category_main', 'category_middle']]
        
        # 중복 제거 (같은 날짜/시간대/경쟁사는 최신 데이터만)
        df = df.drop_duplicates(subset=['broadcast_date', 'time_slot', 'competitor_name'], keep='last')
        
        logger.info(f"   정제 후: {len(df)}개 경쟁사 방송")
    
    return df


def migrate_single_table(netezza_conn, postgres_engine, table_name, incremental=True):
    """단일 테이블 마이그레이션
    
    Args:
        netezza_conn: NETEZZA 연결
        postgres_engine: PostgreSQL 엔진
        table_name: 테이블명
        incremental: 증분 업데이트 여부
    
    Returns:
        dict: 마이그레이션 결과 정보
    """
    
    config = get_table_config(table_name)
    if not config:
        return {"success": False, "table": table_name, "message": "테이블 설정 없음", "count": 0}
    
    try:
        # 1. 데이터 추출
        df = extract_table_from_netezza(netezza_conn, table_name, incremental)
        
        if df is None or len(df) == 0:
            logger.warning(f"⚠️  {table_name}: 추출된 데이터 없음")
            return {"success": True, "table": table_name, "message": "데이터 없음", "count": 0}
        
        # 2. 데이터 정제
        if table_name == 'TAIGOODS':
            table_type = 'products'
        elif table_name == 'TAICOMPETITOR_BROADCASTS':
            table_type = 'competitors'
        else:
            table_type = 'tapes'
        
        df = data_cleansing(df, table_type)
        
        # 3. UPSERT
        primary_key = config["primary_key"]
        success = upsert_to_postgres(df, table_name, postgres_engine, key_column=primary_key)
        
        return {
            "success": success,
            "table": table_name,
            "message": "완료" if success else "실패",
            "count": len(df)
    }
        
    except Exception as e:
        logger.error(f"❌ {table_name} 마이그레이션 실패: {e}")
        return {"success": False, "table": table_name, "message": str(e), "count": 0}


def main():
    """메인 실행 함수 - 모든 활성화된 테이블 마이그레이션"""
    
    # 환경변수로 전체 재처리 여부 결정 (기본값: 증분 업데이트)
    full_sync = os.getenv('FULL_SYNC', 'false').lower() == 'true'
    incremental = not full_sync
    
    # 특정 테이블만 마이그레이션 (환경변수)
    target_tables = os.getenv('TABLES', '').split(',') if os.getenv('TABLES') else None
    
    print("=" * 70)
    print("NETEZZA → PostgreSQL 데이터 마이그레이션 (범용)")
    print("=" * 70)
    print(f"모드: {'전체 재처리' if not incremental else '증분 업데이트 (어제 이후)'}")
    if target_tables:
        print(f"대상 테이블: {', '.join(target_tables)}")
    else:
        print(f"대상 테이블: 모든 활성화된 테이블")
    print("=" * 70)
    
    try:
        # 1. 데이터베이스 연결
        netezza_conn = connect_netezza()
        postgres_engine = connect_postgres()
        
        # 2. 마이그레이션할 테이블 목록 가져오기
        enabled_tables = get_enabled_tables()
        
        if target_tables:
            # 특정 테이블만 필터링
            enabled_tables = {k: v for k, v in enabled_tables.items() if k in target_tables}
        
        if not enabled_tables:
            logger.error("❌ 마이그레이션할 테이블이 없습니다")
            return
        
        logger.info(f"\n📋 마이그레이션 대상: {len(enabled_tables)}개 테이블")
        for table_name, config in enabled_tables.items():
            logger.info(f"   - {table_name}: {config['description']}")
        
        # 3. 각 테이블 마이그레이션
        print("\n" + "=" * 70)
        results = []
        
        for table_name in enabled_tables.keys():
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 {table_name} 마이그레이션 시작...")
            logger.info(f"{'='*70}")
            
            result = migrate_single_table(netezza_conn, postgres_engine, table_name, incremental)
            results.append(result)
            
            if result["success"]:
                logger.info(f"✅ {table_name}: {result['count']}개 레코드 처리 완료")
            else:
                logger.error(f"❌ {table_name}: {result['message']}")
        
        # 4. 결과 요약
        print("\n" + "=" * 70)
        print("📊 마이그레이션 결과 요약")
        print("=" * 70)
        
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        total_records = sum(r["count"] for r in results)
        
        print(f"성공: {success_count}/{total_count} 테이블")
        print(f"총 레코드: {total_records:,}개")
        print()
        
        for result in results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['table']:<25} {result['count']:>8,}개  ({result['message']})")
        
        print("=" * 70)
        
        if success_count == total_count:
            print("\n✅ 모든 테이블 마이그레이션 완료!")
            print("\n다음 단계:")
            print("  1. 데이터 확인: psql로 PostgreSQL 접속")
            print("  2. 임베딩 생성: docker exec -it fastapi_backend python app/setup_product_embeddings.py")
        else:
            print("\n⚠️  일부 테이블 마이그레이션 실패 - 로그 확인 필요")
        
        # 연결 종료
        netezza_conn.close()
        postgres_engine.dispose()
        
        # 종료 코드 반환 (n8n에서 확인 가능)
        return 0 if success_count == total_count else 1
        
    except Exception as e:
        logger.error(f"❌ 마이그레이션 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
