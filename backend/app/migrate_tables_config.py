"""
NETEZZA → PostgreSQL 테이블 마이그레이션 설정
각 테이블별 쿼리 및 매핑 정의
"""

from datetime import datetime, timedelta

def get_yesterday():
    """어제 날짜 계산 (YYYYMMDD 형식)"""
    return (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

def get_yesterday_timestamp():
    """어제 날짜 계산 (타임스탬프 형식)"""
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')


# ============================================
# 테이블 마이그레이션 설정
# ============================================

MIGRATION_TABLES = {
    
    # 1. 상품 마스터 테이블
    "TAIGOODS": {
        "enabled": True,
        "description": "상품 마스터 데이터 (2024년 이후 방송된 상품만)",
        "primary_key": "product_code",
        "incremental_column": "WTI.REG_DTTM",  # 증분 업데이트 기준 컬럼
        "query": lambda incremental: f"""
            SELECT SUB.product_code AS product_code
                 , DGM.GDS_NM   AS product_name
                 , DGC.GDS_LCLS_NM  AS category_main
                 , DGC.GDS_MCLS_NM  AS category_middle
                 , DGC.GDS_SCLS_NM  AS category_sub
                 , DGM.GDS_SEL_UPRC AS price
                 , DBM.BRND_NM AS brand
                 , CASE WHEN DGM.CMPSTN_YN = '1' THEN '유형' ELSE '무형' END AS product_type
                 , DGM.REG_DTTM AS created_at
              FROM (
                SELECT DISTINCT WTGI.GDS_CD AS product_code
                  FROM SNTDW.SNTADM.WBD_TAPE_INFO WTI
                  JOIN SNTDW.SNTADM.WBD_TAPE_GDS_INFO WTGI
                    ON (WTI.PROG_TAPE_CD = WTGI.PROG_TAPE_CD)
                  JOIN SNTDM.SNTADM.FBD_PGMCPF_M FPM
                    ON (WTI.PROG_TAPE_CD = FPM.PROG_TAPE_CD)
                 WHERE WTGI.MN_YN = '1'
                   AND WTGI.GDS_CD <> '00000000'
                   AND FPM.STRD_YMD >= '20240101'
                   AND FPM.PROG_TAPE_CD LIKE '0000%'
                   {f"AND WTI.REG_DTTM >= '{get_yesterday_timestamp()}'" if incremental else ""}
              ) SUB
              JOIN SNTDM.SNTADM.DST_GDS_MST_EXT DGM
                ON (SUB.product_code = DGM.GDS_CD)
              JOIN SNTDM.SNTADM.DST_GDS_CLS DGC
                ON (DGM.GDS_USCLS_CD = DGC.GDS_USCLS_CD)
              JOIN SNTDM.SNTADM.DST_BRND_MST DBM
                ON (DGM.BRND_CD = DBM.BRND_CD)
             ORDER BY SUB.product_code
        """
    },
    
    # 2. 방송테이프 테이블
    "TAIPGMTAPE": {
        "enabled": True,
        "description": "방송테이프 정보 (2024년 이후 방송된 테이프만)",
        "primary_key": "tape_code",
        "incremental_column": "WTI.REG_DTTM",
        "query": lambda incremental: f"""
            SELECT tape_code,
                   tape_name,
                   product_code,
                   production_status,
                   created_at
              FROM (
                SELECT WTI.PROG_TAPE_CD AS tape_code,
                       WTI.PROG_TAPE_NM AS tape_name,
                       WTGI.GDS_CD AS product_code,
                       CASE WTI.TAPE_USE_DVCD
                           WHEN '00' THEN 'ready'
                           WHEN '10' THEN 'inactive'
                           WHEN '99' THEN 'archived'
                           ELSE 'inactive'
                       END AS production_status,
                       WTI.REG_DTTM AS created_at,
                       ROW_NUMBER() OVER (PARTITION BY WTI.PROG_TAPE_CD ORDER BY WTGI.MOD_DTTM DESC) AS rn
                  FROM SNTDW.SNTADM.WBD_TAPE_INFO WTI
                  JOIN SNTDW.SNTADM.WBD_TAPE_GDS_INFO WTGI
                    ON (WTI.PROG_TAPE_CD = WTGI.PROG_TAPE_CD)
                  JOIN SNTDM.SNTADM.FBD_PGMCPF_M FPM
                    ON (WTI.PROG_TAPE_CD = FPM.PROG_TAPE_CD)
                 WHERE WTGI.MN_YN = '1'
                   AND WTGI.GDS_CD <> '00000000'
                   AND FPM.STRD_YMD >= '20240101'
                   AND FPM.PROG_TAPE_CD LIKE '0000%'
                   {f"AND WTI.REG_DTTM >= '{get_yesterday_timestamp()}'" if incremental else ""}
              ) sub
             WHERE rn = 1
             ORDER BY tape_code
        """
    },
    
    # 3. 방송 이력 테이블
    "TAIBROADCASTS": {
        "enabled": True,
        "description": "방송 이력 데이터",
        "primary_key": "id",  # SERIAL (자동 생성)
        "incremental_column": "FPM.STRD_YMD",
        "query": lambda incremental: f"""
            SELECT FPM.PROG_TAPE_CD AS tape_code,
                   FPM.BDCAST_STRT_DTTM AS broadcast_start_timestamp,
                   NULL AS product_is_new,
                   ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT), 2) + SUM(FPM.ADV_COST), 0) AS gross_profit,
                   ROUND((ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT), 2) + SUM(FPM.ADV_COST), 0) / MAX(FBD.CONV_WORTH_VAL)) / 1000000, 1) AS sales_efficiency,
                   CURRENT_TIMESTAMP AS created_at
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
               AND FPM.PROG_TAPE_CD LIKE '0000%'
               {f"AND FPM.STRD_YMD >= '{get_yesterday()}'" if incremental else ""}
             GROUP BY FPM.STRD_YMD,
                      FPM.BDCAST_STRT_DTTM,
                      FPM.BDCAST_END_DTTM,
                      FPM.BCPGM_MTHD_CD,
                      FPM.PROG_TAPE_CD,
                      FPM.REP_GDS_CD,
                      FPM.MD_CD_REP_GDS
             ORDER BY FPM.BDCAST_STRT_DTTM
        """
    },
    
    # 4. 공휴일 테이블 (연 1회 또는 수동 업데이트)
    "TAIHOLIDAYS": {
        "enabled": False,  # 수동 업데이트
        "description": "공휴일 정보 (수동 업데이트 권장)",
        "primary_key": "holiday_date",
        "incremental_column": None,
        "query": lambda incremental: """
            SELECT 
                   HOLIDAY_DATE AS holiday_date,
                   HOLIDAY_NAME AS holiday_name,
                   HOLIDAY_TYPE AS holiday_type,
                   CURRENT_TIMESTAMP AS created_at
              FROM SNTDM.SNTADM.HOLIDAY_MST
             WHERE HOLIDAY_DATE >= CURRENT_DATE
             ORDER BY HOLIDAY_DATE
        """
    },
    
    # 5. 경쟁사 방송 정보 (추가 예정)
    "TAICOMPETITOR_BROADCASTS": {
        "enabled": False,  # 추후 구현
        "description": "경쟁사 방송 정보 (추후 구현)",
        "primary_key": "id",
        "incremental_column": None,
        "query": lambda incremental: """
            -- 경쟁사 데이터 수집 쿼리 (추후 정의)
            SELECT 1 WHERE 1=0
        """
    },
    
    # 6. 트렌드 데이터 (n8n 별도 수집)
    "TAITRENDS": {
        "enabled": False,  # n8n에서 별도 수집
        "description": "트렌드 데이터 (n8n 워크플로우에서 수집)",
        "primary_key": "id",
        "incremental_column": None,
        "query": lambda incremental: """
            -- n8n에서 별도 수집
            SELECT 1 WHERE 1=0
        """
    },
}


def get_enabled_tables():
    """활성화된 테이블 목록 반환"""
    return {k: v for k, v in MIGRATION_TABLES.items() if v["enabled"]}


def get_table_config(table_name):
    """특정 테이블의 설정 반환"""
    return MIGRATION_TABLES.get(table_name.upper())
