import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/app')
from migrate_netezza_to_postgres import connect_netezza

print("Netezza 연결 중...")
conn = connect_netezza()

query = """
SELECT FPM.PROG_TAPE_CD AS tape_code,
       FPM.BDCAST_STRT_DTTM AS broadcast_start_timestamp,
       SUM(FPM.ORD_QTY) AS quantity_sold,
       ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT), 2) + SUM(FPM.ADV_COST), 0) AS gross_profit
  FROM SNTDM.SNTADM.FBD_PGMCPF_M FPM
  JOIN SNTDM.SNTADM.FBD_BCPGMGOAL_D FBD
    ON FPM.STRD_YMD = FBD.STRD_YMD
   AND FPM.PROG_TAPE_CD = FBD.PROG_TAPE_CD
   AND FPM.BCPGM_MTHD_CD = FBD.BCPGM_MTHD_CD
   AND FPM.BDCAST_STRT_DTTM = FBD.BDCAST_STRT_DTTM
   AND FPM.BDCAST_END_DTTM = FBD.BDCAST_END_DTTM
   AND FPM.REP_GDS_CD = FBD.REP_GDS_CD
   AND FPM.PLATFORM_CD = FBD.PLATFORM_CD
   AND FPM.PLATFORM_CD_TOT = FBD.PLATFORM_CD_TOT
 WHERE FPM.STRD_YMD >= '20240101'
   AND FPM.STRD_YMD <= '20240110'
   AND FPM.PROG_TAPE_CD LIKE '0000%'
 GROUP BY FPM.STRD_YMD,
          FPM.BDCAST_STRT_DTTM,
          FPM.BDCAST_END_DTTM,
          FPM.BCPGM_MTHD_CD,
          FPM.PROG_TAPE_CD,
          FPM.REP_GDS_CD,
          FPM.MD_CD_REP_GDS
 LIMIT 10
"""

df = pd.read_sql(query, conn)
print("\n[추출 쿼리 결과 테스트]")
print(df.to_string())
