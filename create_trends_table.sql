-- 트렌드 키워드 테이블 생성
CREATE TABLE IF NOT EXISTS trends (
    id SERIAL PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL,
    source VARCHAR(50) NOT NULL, -- 'naver', 'google', 'news', 'weather'
    score DECIMAL(10,2) NOT NULL, -- 트렌드 점수 (0-100)
    category VARCHAR(100), -- '건강식품', '의류', '가전', '뷰티', '생활용품', '기타'
    related_keywords TEXT[], -- 연관 키워드 배열
    metadata JSONB, -- 추가 메타데이터 (JSON 형태)
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- 인덱스 생성 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_trends_keyword ON trends(keyword);
CREATE INDEX IF NOT EXISTS idx_trends_source ON trends(source);
CREATE INDEX IF NOT EXISTS idx_trends_category ON trends(category);
CREATE INDEX IF NOT EXISTS idx_trends_collected_at ON trends(collected_at);
CREATE INDEX IF NOT EXISTS idx_trends_score ON trends(score DESC);

-- 복합 인덱스 (자주 사용되는 쿼리 패턴)
CREATE INDEX IF NOT EXISTS idx_trends_keyword_collected_at ON trends(keyword, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_trends_category_score ON trends(category, score DESC);

-- 트렌드 데이터 정리를 위한 파티션 (선택사항)
-- 7일 이상 된 데이터 자동 삭제 함수
CREATE OR REPLACE FUNCTION cleanup_old_trends()
RETURNS void AS $$
BEGIN
    DELETE FROM trends 
    WHERE collected_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- 매일 자정에 실행되는 정리 작업 (cron job 필요)
-- SELECT cron.schedule('cleanup-trends', '0 0 * * *', 'SELECT cleanup_old_trends();');

COMMENT ON TABLE trends IS '실시간 트렌드 키워드 데이터 저장 테이블';
COMMENT ON COLUMN trends.keyword IS '트렌드 키워드';
COMMENT ON COLUMN trends.source IS '데이터 소스 (naver, google, news, weather)';
COMMENT ON COLUMN trends.score IS '트렌드 점수 (0-100, 높을수록 인기)';
COMMENT ON COLUMN trends.category IS '상품 카테고리 분류';
COMMENT ON COLUMN trends.related_keywords IS '연관 키워드 목록';
COMMENT ON COLUMN trends.metadata IS '소스별 추가 메타데이터 (JSON)';
COMMENT ON COLUMN trends.collected_at IS '데이터 수집 시각';
