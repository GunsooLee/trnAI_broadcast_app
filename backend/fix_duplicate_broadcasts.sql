-- TAIBROADCASTS 테이블 중복 데이터 제거 스크립트
-- 실행 전 백업 권장!

-- 1. 중복 데이터 확인
SELECT 
    tape_code,
    broadcast_start_timestamp,
    COUNT(*) as duplicate_count,
    STRING_AGG(id::TEXT, ', ') as duplicate_ids
FROM TAIBROADCASTS
GROUP BY tape_code, broadcast_start_timestamp
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 20;

-- 2. 중복 제거 (각 그룹에서 id가 가장 작은 것만 남기고 삭제)
DELETE FROM TAIBROADCASTS
WHERE id IN (
    SELECT id
    FROM (
        SELECT 
            id,
            ROW_NUMBER() OVER (
                PARTITION BY tape_code, broadcast_start_timestamp 
                ORDER BY id
            ) as rn
        FROM TAIBROADCASTS
    ) t
    WHERE rn > 1
);

-- 3. 유니크 제약 조건 추가 (중복 방지)
CREATE UNIQUE INDEX IF NOT EXISTS idx_taibroadcasts_unique 
ON TAIBROADCASTS (tape_code, broadcast_start_timestamp);

-- 4. 결과 확인
SELECT 
    COUNT(*) as total_broadcasts,
    COUNT(DISTINCT (tape_code, broadcast_start_timestamp)) as unique_broadcasts
FROM TAIBROADCASTS;
