#!/usr/bin/env python3
"""
매출 상위 방송의 상품명 키워드 분석 스크립트
"""

import os
import re
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
from sqlalchemy import create_engine

# 환경 설정
load_dotenv()
db_url = os.getenv("POSTGRES_URI", "postgresql://TRN_AI:TRN_AI@trnAi_postgres:5432/TRNAI_DB")
engine = create_engine(db_url)

# 데이터 로드
query = """
SELECT 
    product_name,
    gross_profit,
    broadcast_date
FROM broadcast_training_dataset
WHERE gross_profit IS NOT NULL
ORDER BY broadcast_date
"""

print("데이터 로딩 중...")
df = pd.read_sql(query, engine)
print(f"총 {len(df):,}개 방송 데이터 로드 완료")

# 매출 상위 10-20% 필터링
top_10_threshold = df['gross_profit'].quantile(0.90)
top_20_threshold = df['gross_profit'].quantile(0.80)

top_10_df = df[df['gross_profit'] >= top_10_threshold]
top_20_df = df[df['gross_profit'] >= top_20_threshold]

print(f"\n매출 상위 10%: {len(top_10_df):,}개 방송 (매출 {top_10_threshold:,.0f}원 이상)")
print(f"매출 상위 20%: {len(top_20_df):,}개 방송 (매출 {top_20_threshold:,.0f}원 이상)")

# 한글 키워드 추출 함수
def extract_korean_keywords(text):
    """상품명에서 한글 키워드 추출 (2글자 이상)"""
    if pd.isna(text):
        return []
    # 한글만 추출 (2글자 이상)
    keywords = re.findall(r'[가-힣]{2,}', str(text))
    return keywords

# 숫자+한글 패턴 추출 (예: 1+1, 2+1)
def extract_number_patterns(text):
    """숫자+기호+숫자 패턴 추출"""
    if pd.isna(text):
        return []
    patterns = re.findall(r'\d+\+\d+', str(text))
    return patterns

# 상위 10% 키워드 분석
print("\n=== 매출 상위 10% 방송 키워드 분석 ===")
top_10_keywords = []
top_10_patterns = []

for name in top_10_df['product_name']:
    top_10_keywords.extend(extract_korean_keywords(name))
    top_10_patterns.extend(extract_number_patterns(name))

keyword_counter_10 = Counter(top_10_keywords)
pattern_counter_10 = Counter(top_10_patterns)

print("\n[한글 키워드 Top 30]")
for keyword, count in keyword_counter_10.most_common(30):
    freq = count / len(top_10_df) * 100
    print(f"{keyword:15s} : {count:4d}회 ({freq:5.1f}%)")

print("\n[숫자 패턴 Top 10]")
for pattern, count in pattern_counter_10.most_common(10):
    freq = count / len(top_10_df) * 100
    print(f"{pattern:10s} : {count:4d}회 ({freq:5.1f}%)")

# 상위 20% 키워드 분석
print("\n=== 매출 상위 20% 방송 키워드 분석 ===")
top_20_keywords = []
top_20_patterns = []

for name in top_20_df['product_name']:
    top_20_keywords.extend(extract_korean_keywords(name))
    top_20_patterns.extend(extract_number_patterns(name))

keyword_counter_20 = Counter(top_20_keywords)
pattern_counter_20 = Counter(top_20_patterns)

print("\n[한글 키워드 Top 30]")
for keyword, count in keyword_counter_20.most_common(30):
    freq = count / len(top_20_df) * 100
    print(f"{keyword:15s} : {count:4d}회 ({freq:5.1f}%)")

print("\n[숫자 패턴 Top 10]")
for pattern, count in pattern_counter_20.most_common(10):
    freq = count / len(top_20_df) * 100
    print(f"{pattern:10s} : {count:4d}회 ({freq:5.1f}%)")

# 매출 상관관계가 높은 키워드 발굴
print("\n=== 매출 상관관계 분석 ===")

# 주요 키워드 후보 (빈도 기반)
candidate_keywords = [kw for kw, _ in keyword_counter_20.most_common(50)]
candidate_patterns = [pt for pt, _ in pattern_counter_20.most_common(10)]

# 각 키워드별 평균 매출 계산
keyword_profit_analysis = []

for keyword in candidate_keywords:
    has_keyword = df['product_name'].str.contains(keyword, na=False)
    if has_keyword.sum() >= 10:  # 최소 10회 이상 등장
        avg_profit_with = df[has_keyword]['gross_profit'].mean()
        avg_profit_without = df[~has_keyword]['gross_profit'].mean()
        profit_ratio = avg_profit_with / avg_profit_without if avg_profit_without > 0 else 0
        
        keyword_profit_analysis.append({
            'keyword': keyword,
            'count': has_keyword.sum(),
            'avg_profit_with': avg_profit_with,
            'avg_profit_without': avg_profit_without,
            'profit_ratio': profit_ratio
        })

# 매출 비율 기준 정렬
keyword_profit_df = pd.DataFrame(keyword_profit_analysis)
keyword_profit_df = keyword_profit_df.sort_values('profit_ratio', ascending=False)

print("\n[매출 영향력 Top 20 키워드]")
print(f"{'키워드':<15s} {'등장횟수':>8s} {'포함시 평균매출':>15s} {'미포함시 평균매출':>15s} {'매출비율':>8s}")
print("=" * 80)
for _, row in keyword_profit_df.head(20).iterrows():
    print(f"{row['keyword']:<15s} {row['count']:>8.0f} {row['avg_profit_with']:>15,.0f} {row['avg_profit_without']:>15,.0f} {row['profit_ratio']:>8.2f}x")

# 숫자 패턴 분석
pattern_profit_analysis = []
for pattern in candidate_patterns:
    has_pattern = df['product_name'].str.contains(re.escape(pattern), na=False)
    if has_pattern.sum() >= 5:
        avg_profit_with = df[has_pattern]['gross_profit'].mean()
        avg_profit_without = df[~has_pattern]['gross_profit'].mean()
        profit_ratio = avg_profit_with / avg_profit_without if avg_profit_without > 0 else 0
        
        pattern_profit_analysis.append({
            'pattern': pattern,
            'count': has_pattern.sum(),
            'avg_profit_with': avg_profit_with,
            'avg_profit_without': avg_profit_without,
            'profit_ratio': profit_ratio
        })

pattern_profit_df = pd.DataFrame(pattern_profit_analysis)
pattern_profit_df = pattern_profit_df.sort_values('profit_ratio', ascending=False)

print("\n[매출 영향력 숫자 패턴]")
print(f"{'패턴':<10s} {'등장횟수':>8s} {'포함시 평균매출':>15s} {'미포함시 평균매출':>15s} {'매출비율':>8s}")
print("=" * 80)
for _, row in pattern_profit_df.iterrows():
    print(f"{row['pattern']:<10s} {row['count']:>8.0f} {row['avg_profit_with']:>15,.0f} {row['avg_profit_without']:>15,.0f} {row['profit_ratio']:>8.2f}x")

# 최종 추천 키워드
print("\n" + "=" * 80)
print("🎯 최종 추천 키워드 (매출 비율 1.1배 이상 & 등장 횟수 50회 이상)")
print("=" * 80)

recommended_keywords = keyword_profit_df[
    (keyword_profit_df['profit_ratio'] >= 1.1) & 
    (keyword_profit_df['count'] >= 50)
]['keyword'].tolist()

recommended_patterns = pattern_profit_df[
    pattern_profit_df['profit_ratio'] >= 1.0
]['pattern'].tolist()

print(f"\n한글 키워드 ({len(recommended_keywords)}개):")
print(recommended_keywords)

print(f"\n숫자 패턴 ({len(recommended_patterns)}개):")
print(recommended_patterns)

print("\n✅ 분석 완료!")
