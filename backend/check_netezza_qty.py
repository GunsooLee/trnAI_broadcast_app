import os
import sys
import pandas as pd

# 백엔드 컨테이너 내부의 마이그레이션 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.migrate_netezza_to_postgres import connect_netezza

print("Netezza DB 연결 중...")
try:
    conn = connect_netezza()
    
    query = """
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN ORD_QTY > 0 THEN 1 ELSE 0 END) as rows_with_qty,
        MIN(ORD_QTY) as min_qty,
        MAX(ORD_QTY) as max_qty,
        AVG(ORD_QTY) as avg_qty
    FROM SNTDM.SNTADM.FBD_PGMCPF_M
    WHERE STRD_YMD >= '20240101'
      AND PROG_TAPE_CD LIKE '0000%'
    """
    
    print("Netezza 원본 데이터(ORD_QTY) 쿼리 실행 중...")
    df = pd.read_sql(query, conn)
    print("\n[Netezza 원본 ORD_QTY 확인 결과]")
    print(df.to_string(index=False))
    
    query_sample = """
    SELECT STRD_YMD, PROG_TAPE_CD, ORD_QTY, PRDC_MOD_SAMT 
    FROM SNTDM.SNTADM.FBD_PGMCPF_M 
    WHERE STRD_YMD >= '20240101' 
      AND ORD_QTY > 0
    LIMIT 5
    """
    df_sample = pd.read_sql(query_sample, conn)
    print("\n[ORD_QTY > 0 인 샘플 데이터 5건]")
    print(df_sample.to_string(index=False))
    
except Exception as e:
    print(f"오류 발생: {e}")
