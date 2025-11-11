"""
방송 이력 조회 서비스 (Netezza)
"""
import os
import logging
from typing import Optional, List, Dict, Any
import nzpy

logger = logging.getLogger(__name__)

class BroadcastHistoryService:
    """방송 이력 조회 서비스 (Netezza DB)"""
    
    def __init__(self):
        """기존 netezza_config의 환경변수 사용"""
        self.host = os.getenv("NETEZZA_HOST")
        self.port = os.getenv("NETEZZA_PORT", "5480")
        self.database = os.getenv("NETEZZA_DATABASE", "SYSTEM")
        self.username = os.getenv("NETEZZA_USER")
        self.password = os.getenv("NETEZZA_PASSWORD")
        
        if not all([self.host, self.username, self.password]):
            logger.warning("⚠️ Netezza 환경변수가 설정되지 않았습니다. 최근 방송 실적 조회가 비활성화됩니다.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Netezza 연결 설정 완료: {self.host}:{self.port}/{self.database}")
    
    def get_latest_broadcast_by_tape_code(self, tape_code: str) -> Optional[Dict[str, Any]]:
        """
        방송테이프 코드로 가장 최근 방송 실적 조회
        
        Args:
            tape_code: 프로그램테이프코드 (예: '0000011413')
            
        Returns:
            최근 방송 실적 딕셔너리 또는 None
        """
        if not self.enabled:
            logger.debug(f"Netezza가 비활성화되어 테이프 {tape_code}의 실적 조회를 건너뜁니다.")
            return None
        
        if not tape_code:
            return None
        
        query = """
        SELECT FPM.STRD_YMD 
             , FPM.BDCAST_STRT_DTTM
             , SUM(FPM.ORD_QTY) as ORD_QTY
             , ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT),2) + SUM(FPM.ADV_COST),0) as SAL_TOT_PRFT
             , ROUND((ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT),2) + SUM(FPM.ADV_COST),0) / MAX(FBD.CONV_WORTH_VAL)) /1000000 ,1) as SAL_TOT_PRFT_EFF
             , MAX(FBD.CONV_WORTH_VAL) as CONV_WORTH_VAL
             , AVG(FPM.MOD_CNVS_RT) as MOD_CNVS_RT
             , Sum(FPM.NAME_SAMT) - Sum(FPM.OURCO_BRD_DC_AMT) as REAL_FEE
             , Sum(FPM.MIX_FEE) as MIX_FEE
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
           AND FPM.PROG_TAPE_CD = ?
         GROUP BY FPM.STRD_YMD, FPM.BDCAST_STRT_DTTM
         ORDER BY FPM.STRD_YMD DESC, FPM.BDCAST_STRT_DTTM DESC
         LIMIT 1
        """
        
        try:
            with nzpy.connect(
                host=self.host,
                port=int(self.port),
                database=self.database,
                user=self.username,
                password=self.password
            ) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (tape_code,))
                row = cursor.fetchone()
                
                if row:
                    logger.info(f"✅ 테이프 {tape_code}의 최근 방송 실적 조회 성공")
                    # nzpy는 컬럼명으로 접근 가능
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, row))
                    
                    return {
                        "broadcastStartTime": str(row_dict.get("BDCAST_STRT_DTTM", "")),
                        "orderQuantity": int(row_dict.get("ORD_QTY", 0) or 0),
                        "totalProfit": float(row_dict.get("SAL_TOT_PRFT", 0) or 0),
                        "profitEfficiency": float(row_dict.get("SAL_TOT_PRFT_EFF", 0) or 0),
                        "conversionWorth": float(row_dict.get("CONV_WORTH_VAL", 0) or 0),
                        "conversionRate": float(row_dict.get("MOD_CNVS_RT", 0) or 0),
                        "realFee": float(row_dict.get("REAL_FEE", 0) or 0),
                        "mixFee": float(row_dict.get("MIX_FEE", 0) or 0)
                    }
                else:
                    logger.debug(f"테이프 {tape_code}의 최근 방송 실적이 없습니다.")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Netezza 쿼리 실행 오류 (테이프: {tape_code}): {e}")
            return None
    
    def get_latest_broadcasts_batch(self, tape_codes: list[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        여러 방송테이프 코드의 최근 방송 실적을 배치로 조회 (성능 최적화)
        
        Args:
            tape_codes: 프로그램테이프코드 리스트
            
        Returns:
            {tape_code: 실적 딕셔너리} 매핑
        """
        if not self.enabled or not tape_codes:
            return {code: None for code in tape_codes}
        
        # IN 절을 위한 플레이스홀더 생성
        placeholders = ','.join(['?' for _ in tape_codes])
        
        query = f"""
        SELECT * FROM (
            SELECT 
                 FPM.PROG_TAPE_CD
                 , FPM.STRD_YMD 
                 , FPM.BDCAST_STRT_DTTM
                 , SUM(FPM.ORD_QTY) as ORD_QTY
                 , ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT),2) + SUM(FPM.ADV_COST),0) as SAL_TOT_PRFT
                 , ROUND((ROUND(ROUND(SUM(FPM.PRDC_MOD_SAMT) + SUM(FPM.INTNGB_SAMT),2) + SUM(FPM.ADV_COST),0) / MAX(FBD.CONV_WORTH_VAL)) /1000000 ,1) as SAL_TOT_PRFT_EFF
                 , MAX(FBD.CONV_WORTH_VAL) as CONV_WORTH_VAL
                 , AVG(FPM.MOD_CNVS_RT) as MOD_CNVS_RT
                 , Sum(FPM.NAME_SAMT) - Sum(FPM.OURCO_BRD_DC_AMT) as REAL_FEE
                 , Sum(FPM.MIX_FEE) as MIX_FEE
                 , ROW_NUMBER() OVER (PARTITION BY FPM.PROG_TAPE_CD ORDER BY FPM.STRD_YMD DESC, FPM.BDCAST_STRT_DTTM DESC) as rn
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
               AND FPM.PROG_TAPE_CD IN ({placeholders})
             GROUP BY FPM.PROG_TAPE_CD, FPM.STRD_YMD, FPM.BDCAST_STRT_DTTM
        ) ranked
        WHERE rn = 1
        """
        
        result = {code: None for code in tape_codes}
        
        try:
            with nzpy.connect(
                host=self.host,
                port=int(self.port),
                database=self.database,
                user=self.username,
                password=self.password
            ) as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(tape_codes))
                rows = cursor.fetchall()
                
                # 컬럼명 가져오기
                columns = [desc[0] for desc in cursor.description]
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    tape_code = row_dict.get("PROG_TAPE_CD")
                    
                    if tape_code:
                        result[tape_code] = {
                            "broadcastStartTime": str(row_dict.get("BDCAST_STRT_DTTM", "")),
                            "orderQuantity": int(row_dict.get("ORD_QTY", 0) or 0),
                            "totalProfit": float(row_dict.get("SAL_TOT_PRFT", 0) or 0),
                            "profitEfficiency": float(row_dict.get("SAL_TOT_PRFT_EFF", 0) or 0),
                            "conversionWorth": float(row_dict.get("CONV_WORTH_VAL", 0) or 0),
                            "conversionRate": float(row_dict.get("MOD_CNVS_RT", 0) or 0),
                            "realFee": float(row_dict.get("REAL_FEE", 0) or 0),
                            "mixFee": float(row_dict.get("MIX_FEE", 0) or 0)
                        }
                
                logger.info(f"✅ {len(rows)}개 테이프의 최근 방송 실적 배치 조회 성공")
                return result
                
        except Exception as e:
            logger.error(f"❌ Netezza 배치 쿼리 실행 오류: {e}")
            return result
