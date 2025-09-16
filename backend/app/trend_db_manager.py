"""
트렌드 데이터베이스 관리 모듈
- 트렌드 데이터 저장/조회
- 배치 수집 데이터 관리
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import json
from .trend_collector import TrendKeyword
# from .external_apis import external_api_manager  # n8n이 API 호출 담당하므로 불필요


class TrendDatabaseManager:
    """트렌드 데이터베이스 관리 클래스"""
    
    def __init__(self):
        # 환경변수에서 DB 연결 정보 가져오기
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'shopping_broadcast'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
    
    def get_connection(self):
        """데이터베이스 연결 생성"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"데이터베이스 연결 오류: {e}")
            raise
    
    async def save_trends_batch(self, trends: List[TrendKeyword]) -> int:
        """트렌드 데이터 배치 저장"""
        if not trends:
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 데이터 준비
            trend_data = []
            for trend in trends:
                trend_data.append((
                    trend.keyword,
                    trend.source,
                    float(trend.score),
                    trend.category,
                    trend.related_keywords,
                    json.dumps(trend.metadata) if trend.metadata else None,
                    trend.timestamp
                ))
            
            # 배치 INSERT
            insert_query = """
                INSERT INTO trends (keyword, source, score, category, related_keywords, metadata, collected_at)
                VALUES %s
                ON CONFLICT (keyword, source, collected_at::date) 
                DO UPDATE SET 
                    score = EXCLUDED.score,
                    category = EXCLUDED.category,
                    related_keywords = EXCLUDED.related_keywords,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """
            
            execute_values(cursor, insert_query, trend_data)
            conn.commit()
            
            saved_count = len(trend_data)
            print(f"트렌드 데이터 {saved_count}개 저장 완료")
            return saved_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"트렌드 저장 오류: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    async def get_latest_trends(self, 
                               limit: int = 50, 
                               category: Optional[str] = None,
                               min_score: float = 0.0,
                               hours_back: int = 24) -> List[Dict[str, Any]]:
        """최신 트렌드 데이터 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 기본 쿼리
            base_query = """
                SELECT DISTINCT ON (keyword) 
                    keyword, source, score, category, 
                    related_keywords, metadata, collected_at
                FROM trends 
                WHERE collected_at >= NOW() - INTERVAL '%s hours'
                    AND score >= %s
            """
            
            params = [hours_back, min_score]
            
            # 카테고리 필터 추가
            if category:
                base_query += " AND category = %s"
                params.append(category)
            
            # 정렬 및 제한
            base_query += """
                ORDER BY keyword, score DESC, collected_at DESC
                LIMIT %s
            """
            params.append(limit)
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            
            # Dict 형태로 변환
            trends = []
            for row in results:
                trend_dict = dict(row)
                trend_dict['collected_at'] = trend_dict['collected_at'].isoformat()
                trends.append(trend_dict)
            
            print(f"최신 트렌드 {len(trends)}개 조회 완료")
            return trends
            
        except Exception as e:
            print(f"트렌드 조회 오류: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    async def get_trending_by_category(self, hours_back: int = 6) -> Dict[str, List[Dict[str, Any]]]:
        """카테고리별 트렌드 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT category, keyword, AVG(score) as avg_score, COUNT(*) as frequency
                FROM trends 
                WHERE collected_at >= NOW() - INTERVAL '%s hours'
                    AND category IS NOT NULL
                    AND category != '기타'
                GROUP BY category, keyword
                HAVING AVG(score) >= 10
                ORDER BY category, avg_score DESC
            """
            
            cursor.execute(query, [hours_back])
            results = cursor.fetchall()
            
            # 카테고리별로 그룹화
            category_trends = {}
            for row in results:
                category = row['category']
                if category not in category_trends:
                    category_trends[category] = []
                
                category_trends[category].append({
                    'keyword': row['keyword'],
                    'avg_score': float(row['avg_score']),
                    'frequency': row['frequency']
                })
            
            # 각 카테고리별로 상위 10개만 유지
            for category in category_trends:
                category_trends[category] = category_trends[category][:10]
            
            return category_trends
            
        except Exception as e:
            print(f"카테고리별 트렌드 조회 오류: {e}")
            return {}
        finally:
            if conn:
                conn.close()
    
    async def get_trend_history(self, keyword: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """특정 키워드의 트렌드 히스토리 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT DATE(collected_at) as trend_date, 
                       AVG(score) as avg_score,
                       MAX(score) as max_score,
                       COUNT(*) as data_points
                FROM trends 
                WHERE keyword ILIKE %s
                    AND collected_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(collected_at)
                ORDER BY trend_date DESC
            """
            
            cursor.execute(query, [f'%{keyword}%', days_back])
            results = cursor.fetchall()
            
            history = []
            for row in results:
                history.append({
                    'date': row['trend_date'].isoformat(),
                    'avg_score': float(row['avg_score']),
                    'max_score': float(row['max_score']),
                    'data_points': row['data_points']
                })
            
            return history
            
        except Exception as e:
            print(f"트렌드 히스토리 조회 오류: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    async def cleanup_old_trends(self, days_to_keep: int = 7) -> int:
        """오래된 트렌드 데이터 정리"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                DELETE FROM trends 
                WHERE collected_at < NOW() - INTERVAL '%s days'
            """
            
            cursor.execute(query, [days_to_keep])
            deleted_count = cursor.rowcount
            conn.commit()
            
            print(f"오래된 트렌드 데이터 {deleted_count}개 삭제 완료")
            return deleted_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"트렌드 정리 오류: {e}")
            return 0
        finally:
            if conn:
                conn.close()
    
    async def collect_and_save_trends(self) -> Dict[str, Any]:
        """n8n 배치가 수집한 트렌드 데이터 처리 (DB에서 최신 데이터 조회)"""
        print("--- 트렌드 데이터 처리 시작 (DB 기반) ---")

        try:
            # DB에서 최신 트렌드 데이터 조회 (n8n이 이미 저장한 데이터)
            latest_trends = await self.get_latest_trends(limit=100, hours_back=1)

            # 오래된 데이터 정리
            cleaned_count = await self.cleanup_old_trends()

            result = {
                "success": True,
                "collected_count": len(latest_trends),
                "saved_count": len(latest_trends),  # 이미 저장된 데이터 개수
                "cleaned_count": cleaned_count,
                "timestamp": datetime.now().isoformat()
            }

            print(f"--- 트렌드 데이터 처리 완료: {result} ---")
            return result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f"--- 트렌드 데이터 처리 실패: {error_result} ---")
            return error_result


# 전역 인스턴스
trend_db_manager = TrendDatabaseManager()
