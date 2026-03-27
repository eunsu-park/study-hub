# 19. 전문 검색(Full-Text Search)

**이전**: [테이블 파티셔닝](./18_Table_Partitioning.md) | **다음**: [보안과 접근 제어](./20_Security_Access_Control.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 인덱싱, 형태소 분석(stemming), 랭킹 측면에서 전문 검색(Full-Text Search)이 LIKE/ILIKE보다 우수한 이유를 설명할 수 있다
2. tsvector(문서 표현) 및 tsquery(쿼리 표현) 데이터 타입을 생성하고 조작할 수 있다
3. 텍스트 검색 사전(dictionary), 불용어(stop word), 형태소 분석을 다양한 언어에 맞게 구성할 수 있다
4. tsvector 컬럼에 GIN 및 GiST 인덱스를 구축하고 두 방식의 트레이드오프를 비교할 수 있다
5. ts_rank, ts_rank_cd, 그리고 setweight를 활용한 가중치 검색으로 검색 결과를 랭킹할 수 있다
6. 구문 검색(phrase search), 접두어 매칭(prefix matching), 하이라이트 스니펫(highlighted snippet) 등 고급 검색 기능을 구현할 수 있다
7. pg_trgm 확장을 사용하여 퍼지 매칭(fuzzy matching), 오타 허용, LIKE 쿼리 가속화를 구현할 수 있다

---

사용자 생성 콘텐츠가 있는 애플리케이션은 결국 검색 기능이 필요합니다. LIKE 쿼리는 소규모 데이터셋에서는 작동하지만, 대용량 테이블에서는 매우 느려지고 언어를 이해하지 못합니다 — "running"은 "run"과 매칭되지 않으며, 결과를 관련성으로 랭킹할 방법도 없습니다. PostgreSQL의 내장 전문 검색(Full-Text Search)은 형태소 분석, 불용어 제거, 불리언 연산자, 관련성 랭킹을 갖춘 언어 인식 인덱싱을 제공합니다 — Elasticsearch와 같은 외부 검색 엔진 없이도 말이죠. 많은 애플리케이션에서 이 내장 기능으로 충분하며, 아키텍처를 단순하게 유지할 수 있습니다.

## 목차
1. [전문 검색 개요](#1-전문-검색-개요)
2. [tsvector와 tsquery](#2-tsvector와-tsquery)
3. [검색 구성](#3-검색-구성)
4. [GIN과 GiST 인덱스](#4-gin과-gist-인덱스)
5. [랭킹과 가중치](#5-랭킹과-가중치)
6. [고급 검색 기법](#6-고급-검색-기법)
7. [퍼지 매칭을 위한 pg_trgm](#7-퍼지-매칭을-위한-pg_trgm)
8. [연습 문제](#8-연습-문제)

---

## 1. 전문 검색 개요

### 1.1 왜 전문 검색인가?

```
┌─────────────────────────────────────────────────────────────────┐
│              LIKE vs 전문 검색(Full-Text Search)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LIKE / ILIKE:                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ SELECT * FROM articles                   │                   │
│  │ WHERE body ILIKE '%database%';           │                   │
│  │                                          │                   │
│  │ ✗ 인덱스 사용 불가 (순차 스캔)            │                   │
│  │ ✗ 언어 인식 기능 없음                     │                   │
│  │ ✗ 랭킹 기능 없음                          │                   │
│  │ ✗ "databases"가 "database"와 매치 안 됨   │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  전문 검색(Full-Text Search):                                    │
│  ┌──────────────────────────────────────────┐                   │
│  │ SELECT * FROM articles                   │                   │
│  │ WHERE to_tsvector(body)                  │                   │
│  │       @@ to_tsquery('database');          │                   │
│  │                                          │                   │
│  │ ✓ GIN 인덱스 지원 (빠름)                  │                   │
│  │ ✓ 형태소 분석 (databases → database)      │                   │
│  │ ✓ 관련성에 따른 랭킹                      │                   │
│  │ ✓ 불리언 연산자 (AND, OR, NOT)            │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│              전문 검색 파이프라인                                │
│                                                                 │
│  문서 텍스트                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   Parser         │──▶ 단어로 토큰화                          │
│  └─────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │  Dictionaries    │──▶ 정규화 (불용어, 어간)                  │
│  └─────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   tsvector       │──▶ 정렬된 렉심 리스트 + 위치              │
│  └─────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │  GIN Index       │──▶ 렉심별 빠른 검색                       │
│  └─────────────────┘                                            │
│                                                                 │
│  쿼리 텍스트 ──▶ tsquery ──▶ tsvector와 매치                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. tsvector와 tsquery

### 2.1 tsvector — 문서 표현

```sql
-- tsvector는 텍스트를 언어학적으로 정규화: "foxes" → "fox", "jumped" → "jump"
-- LIKE '%fox%'가 놓치는 단어 변형을 FTS가 매칭할 수 있는 이유
SELECT to_tsvector('english', 'The quick brown foxes jumped over the lazy dogs');
-- 결과: 'brown':3 'dog':9 'fox':4 'jump':5 'lazi':8 'quick':2

-- 주의: 불용어 제거됨 ("the", "over"), 단어 어간 추출됨 ("foxes" → "fox")

-- tsvector는 렉심을 위치와 함께 저장
SELECT to_tsvector('english', 'A fat cat sat on a mat. A fat cat ate a fat rat.');
-- 결과: 'ate':10 'cat':3,8 'fat':2,7,11 'mat':6 'rat':12 'sat':4
```

### 2.2 tsquery — 쿼리 표현

```sql
-- 기본 쿼리
SELECT to_tsquery('english', 'cats & dogs');
-- 결과: 'cat' & 'dog'

-- 불리언 연산자
SELECT to_tsquery('english', 'cat | dog');        -- OR
SELECT to_tsquery('english', 'cat & !dog');       -- AND NOT
SELECT to_tsquery('english', 'cat <-> dog');      -- FOLLOWED BY (구문)
SELECT to_tsquery('english', 'cat <2> dog');      -- 2단어 이내

-- plainto_tsquery: 더 간단한 구문 (암묵적 AND)
SELECT plainto_tsquery('english', 'fat cats');
-- 결과: 'fat' & 'cat'

-- phraseto_tsquery: 정확한 구문 매칭
SELECT phraseto_tsquery('english', 'fat cats');
-- 결과: 'fat' <-> 'cat'

-- websearch_to_tsquery: 웹 스타일 구문 (PostgreSQL 11+)
SELECT websearch_to_tsquery('english', '"fat cats" -dogs');
-- 결과: 'fat' <-> 'cat' & !'dog'

SELECT websearch_to_tsquery('english', 'cats or dogs');
-- 결과: 'cat' | 'dog'
```

### 2.3 매치 연산자(@@)

```sql
-- @@는 FTS 핵심 연산자: 문서(tsvector)가 쿼리(tsquery)를 만족하는지 검사
-- LIKE와 달리 @@ 는 GIN 인덱스를 사용하고 언어적 동등성(어간 매칭)을 이해함
SELECT to_tsvector('english', 'The fat cats sat on the mat')
       @@ to_tsquery('english', 'cat & mat');
-- 결과: true

SELECT to_tsvector('english', 'The fat cats sat on the mat')
       @@ to_tsquery('english', 'cat & dog');
-- 결과: false

-- 실용적인 예제
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO articles (title, body) VALUES
('PostgreSQL Full-Text Search', 'PostgreSQL provides powerful full-text search capabilities using tsvector and tsquery.'),
('Database Indexing Guide', 'Indexes are crucial for database performance. B-tree, GIN, and GiST are common index types.'),
('Introduction to SQL', 'SQL is a standard language for managing relational databases. SELECT, INSERT, UPDATE, DELETE.'),
('Advanced Query Optimization', 'Query optimization involves analyzing execution plans and choosing efficient access paths.');

-- 문서 검색
SELECT title, ts_rank(to_tsvector('english', body), query) AS rank
FROM articles, to_tsquery('english', 'database & index') AS query
WHERE to_tsvector('english', body) @@ query
ORDER BY rank DESC;
```

### 2.4 저장된 tsvector 컬럼

```sql
-- STORED 컬럼에 tsvector 사전 계산 — 매 쿼리마다 텍스트를 재파싱하는 비용 제거.
-- setweight로 제목 매치에 'A'(최고), 본문에 'B' 할당하여 제목 히트가 더 높은 랭크
ALTER TABLE articles ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(body, '')), 'B')
    ) STORED;

-- 이제 쿼리는 미리 계산된 컬럼을 사용
SELECT title
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database');
```

---

## 3. 검색 구성

### 3.1 텍스트 검색 구성

```sql
-- 사용 가능한 구성 목록
SELECT cfgname FROM pg_ts_config;

-- 일반적인 구성:
-- 'simple'    — 형태소 분석 없음, 불용어 없음
-- 'english'   — 영어 형태소 분석 및 불용어
-- 'german'    — 독일어 형태소 분석
-- 'spanish'   — 스페인어 형태소 분석

-- 현재 기본값 표시
SHOW default_text_search_config;

-- 기본값 설정
SET default_text_search_config = 'english';

-- 구성 비교
SELECT to_tsvector('simple', 'The running dogs are quickly jumping');
-- 결과: 'are':4 'dogs':3 'jumping':6 'quickly':5 'running':2 'the':1

SELECT to_tsvector('english', 'The running dogs are quickly jumping');
-- 결과: 'dog':3 'jump':6 'quick':5 'run':2
```

### 3.2 사전과 불용어

```sql
-- 구성에 대한 사전 표시
SELECT * FROM ts_debug('english', 'The quick brown foxes are jumping');

-- 출력은 토큰 → 사전 → 렉심 매핑을 표시
-- Token     | Dictionary   | Lexemes
-- ----------|-------------|--------
-- The       | english_stem | {stop word}
-- quick     | english_stem | {quick}
-- brown     | english_stem | {brown}
-- foxes     | english_stem | {fox}
-- are       | english_stem | {stop word}
-- jumping   | english_stem | {jump}
```

### 3.3 사용자 정의 구성

```sql
-- 영어 기반 사용자 정의 구성 생성
CREATE TEXT SEARCH CONFIGURATION my_english (COPY = english);

-- 동의어 사전 추가
CREATE TEXT SEARCH DICTIONARY my_synonyms (
    TEMPLATE = synonym,
    SYNONYMS = my_synonyms  -- $SHAREDIR/tsearch_data/my_synonyms.syn 참조
);

-- 구성에 추가
ALTER TEXT SEARCH CONFIGURATION my_english
    ALTER MAPPING FOR asciiword
    WITH my_synonyms, english_stem;
```

---

## 4. GIN과 GiST 인덱스

### 4.1 전문 검색을 위한 GIN 인덱스

```sql
-- GIN 인덱스는 각 렉심을 포함하는 행에 매핑 — @@를 순차 스캔에서
-- 비트맵 인덱스 스캔으로 변환, 수백만 행 테이블에서 FTS 실용화
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- 표현식 인덱스: 저장된 컬럼 불필요하나 인덱스 생성 시 tsvector 재구축
-- 사전 계산된 가중치(A/B/C/D) 활용 불가
CREATE INDEX idx_articles_body_gin ON articles
    USING GIN (to_tsvector('english', body));

-- 쿼리는 자동으로 인덱스 사용
EXPLAIN ANALYZE
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'database');
-- Bitmap Index Scan on idx_articles_search
```

### 4.2 GIN vs GiST 비교

```
┌─────────────────────────────────────────────────────────────────┐
│              전문 검색을 위한 GIN vs GiST                        │
├────────────────┬──────────────────┬──────────────────────────────┤
│                │ GIN              │ GiST                         │
├────────────────┼──────────────────┼──────────────────────────────┤
│ 빌드 속도       │ 느림             │ 빠름                          │
│ 검색 속도       │ 빠름 (3배)       │ 느림                          │
│ 인덱스 크기     │ 큼               │ 작음                          │
│ 업데이트 비용   │ 높음             │ 낮음                          │
│ 정확한 결과     │ 예               │ 거짓 긍정 가능                 │
│ 최적화         │ 정적 데이터,      │ 자주 업데이트되는 데이터,      │
│                │ 읽기 위주         │ 쓰기 위주                     │
└────────────────┴──────────────────┴──────────────────────────────┘
```

```sql
-- GiST 인덱스 (대안)
CREATE INDEX idx_articles_search_gist ON articles USING GiST (search_vector);

-- fastupdate가 있는 GIN (기본값 on, 배치 삽입에 좋음)
CREATE INDEX idx_articles_search_gin ON articles
    USING GIN (search_vector) WITH (fastupdate = on);
```

### 4.3 인덱스 유지보수

```sql
-- 인덱스 크기 확인
SELECT pg_size_pretty(pg_relation_size('idx_articles_search'));

-- 필요시 재인덱싱
REINDEX INDEX idx_articles_search;

-- 쿼리 플래너를 위한 분석
ANALYZE articles;
```

---

## 5. 랭킹과 가중치

### 5.1 ts_rank

```sql
-- 기본 랭킹
SELECT
    title,
    ts_rank(search_vector, to_tsquery('english', 'database')) AS rank
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database')
ORDER BY rank DESC;

-- 정규화 옵션 (비트마스크)
-- 0  = 기본값 (문서 길이 무시)
-- 1  = 1 + log(문서 길이)로 나눔
-- 2  = 문서 길이로 나눔
-- 4  = 범위 간 평균 조화 거리로 나눔
-- 8  = 고유 단어 수로 나눔
-- 16 = 1 + log(고유 단어)로 나눔
-- 32 = 자기 자신 + 1로 나눔

SELECT
    title,
    ts_rank(search_vector, query, 2) AS rank_normalized  -- 길이로 정규화
FROM articles, to_tsquery('english', 'database') AS query
WHERE search_vector @@ query
ORDER BY rank_normalized DESC;
```

### 5.2 ts_rank_cd (커버 밀도)

```sql
-- 커버 밀도 랭킹은 매칭 용어의 근접성을 고려
SELECT
    title,
    ts_rank_cd(search_vector, to_tsquery('english', 'database & index')) AS rank_cd
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database & index')
ORDER BY rank_cd DESC;
```

### 5.3 가중치 검색

```sql
-- 가중치: A (1.0), B (0.4), C (0.2), D (0.1)
-- 사용자 정의 가중치 배열: {D, C, B, A}

-- 사용자 정의 가중치 적용
SELECT
    title,
    ts_rank('{0.1, 0.2, 0.4, 1.0}', search_vector, query) AS weighted_rank
FROM articles, to_tsquery('english', 'database') AS query
WHERE search_vector @@ query
ORDER BY weighted_rank DESC;

-- 가중치가 있는 tsvector 생성
SELECT
    setweight(to_tsvector('english', 'PostgreSQL Guide'), 'A') ||
    setweight(to_tsvector('english', 'A comprehensive database tutorial'), 'B') ||
    setweight(to_tsvector('english', 'Learn SQL queries and optimization'), 'C');
```

### 5.4 검색 결과 하이라이팅

```sql
-- ts_headline은 매칭 용어를 하이라이트
SELECT
    title,
    ts_headline('english', body, to_tsquery('english', 'database'),
        'StartSel=<b>, StopSel=</b>, MaxWords=35, MinWords=15'
    ) AS highlighted
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database');

-- 옵션:
-- StartSel / StopSel — 하이라이트 마커
-- MaxWords / MinWords — 컨텍스트 윈도우 크기
-- ShortWord — 표시할 최소 단어 길이
-- MaxFragments — 조각 수 (0 = 전체 문서)
-- FragmentDelimiter — 조각 간 구분자
```

---

## 6. 고급 검색 기법

### 6.1 다중 컬럼 검색

```sql
-- 다른 가중치로 여러 컬럼에 걸쳐 검색
CREATE TABLE blog_posts (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    body TEXT NOT NULL,
    tags TEXT[],
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(body, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(array_to_string(tags, ' '), '')), 'A')
    ) STORED
);

CREATE INDEX idx_blog_search ON blog_posts USING GIN (search_vector);
```

### 6.2 구문 검색

```sql
-- 정확한 구문: "full text search"
SELECT title FROM articles
WHERE search_vector @@ phraseto_tsquery('english', 'full text search');

-- 근접성: N개 위치 내의 단어
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'full <3> search');
-- "full"과 "search"가 서로 3단어 이내
```

### 6.3 필터가 있는 검색

```sql
-- 전문 검색과 일반 WHERE 절 결합
SELECT title, ts_rank(search_vector, query) AS rank
FROM articles, to_tsquery('english', 'database') AS query
WHERE search_vector @@ query
  AND created_at >= NOW() - INTERVAL '30 days'
ORDER BY rank DESC
LIMIT 10;

-- 결합 검색을 위한 복합 인덱스
CREATE INDEX idx_articles_date_search ON articles
    USING GIN (search_vector) WHERE created_at >= '2024-01-01';
```

### 6.4 자동 완성 / 접두어 검색

```sql
-- :*를 사용한 접두어 매칭
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'dat:*');
-- 매치: database, data, date 등

-- 실용적인 자동 완성 함수
CREATE OR REPLACE FUNCTION search_autocomplete(search_term TEXT, max_results INT DEFAULT 10)
RETURNS TABLE(title TEXT, rank REAL) AS $$
BEGIN
    RETURN QUERY
    SELECT a.title, ts_rank(a.search_vector, query) AS rank
    FROM articles a, to_tsquery('english', search_term || ':*') AS query
    WHERE a.search_vector @@ query
    ORDER BY rank DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

SELECT * FROM search_autocomplete('post');
```

### 6.5 검색 통계

```sql
-- 코퍼스에서 가장 흔한 단어
SELECT word, ndoc, nentry
FROM ts_stat('SELECT search_vector FROM articles')
ORDER BY nentry DESC
LIMIT 20;

-- 특정 쿼리에 대한 단어 빈도
SELECT word, ndoc
FROM ts_stat('SELECT search_vector FROM articles')
WHERE word LIKE 'data%'
ORDER BY ndoc DESC;
```

---

## 7. 퍼지 매칭을 위한 pg_trgm

### 7.1 트라이그램 기초

```sql
-- pg_trgm은 FTS를 보완: FTS는 정확한 렉심 매치(어간 분석 후) 필요하지만,
-- pg_trgm은 3글자 부분 문자열을 사용하여 오타와 부분 문자열을 처리
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 트라이그램 표시
SELECT show_trgm('PostgreSQL');
-- 결과: {"  p"," po","gre","osq","pos","ostg","pgr","ql ","res","sql","stg","tgr"}

-- 유사도 점수 (0~1)
SELECT similarity('PostgreSQL', 'Postgresql');  -- ~0.75
SELECT similarity('PostgreSQL', 'MySQL');       -- ~0.09
```

### 7.2 트라이그램 인덱스

```sql
-- GIN 트라이그램 인덱스
CREATE INDEX idx_articles_title_trgm ON articles USING GIN (title gin_trgm_ops);

-- GiST 트라이그램 인덱스 (거리 연산자 지원)
CREATE INDEX idx_articles_title_trgm_gist ON articles USING GiST (title gist_trgm_ops);

-- 유사도 검색
SELECT title, similarity(title, 'Postgre') AS sim
FROM articles
WHERE title % 'Postgre'  -- % 연산자는 유사도 임계값 사용
ORDER BY sim DESC;

-- 유사도 임계값 설정 (기본값 0.3)
SET pg_trgm.similarity_threshold = 0.2;

-- 거리 연산자 (GiST만)
SELECT title, title <-> 'Postgre' AS distance
FROM articles
ORDER BY title <-> 'Postgre'
LIMIT 5;
```

### 7.3 트라이그램 인덱스를 사용한 LIKE/ILIKE

```sql
-- 트라이그램 인덱스는 LIKE 쿼리도 가속화
CREATE INDEX idx_articles_body_trgm ON articles USING GIN (body gin_trgm_ops);

-- 이제 이 쿼리들은 인덱스를 사용
SELECT title FROM articles WHERE body LIKE '%database%';
SELECT title FROM articles WHERE body ILIKE '%DATABASE%';

-- 정규 표현식 쿼리도 이점 얻음
SELECT title FROM articles WHERE body ~ 'data(base|set)';
```

### 7.4 FTS와 pg_trgm 결합

```sql
-- 양쪽 장점: 관련성은 전문 검색, 오타 허용은 트라이그램
CREATE OR REPLACE FUNCTION smart_search(search_term TEXT, max_results INT DEFAULT 10)
RETURNS TABLE(id INT, title TEXT, score REAL) AS $$
BEGIN
    -- 먼저 정확한 전문 검색 시도
    RETURN QUERY
    SELECT a.id, a.title,
           ts_rank(a.search_vector, websearch_to_tsquery('english', search_term)) AS score
    FROM articles a
    WHERE a.search_vector @@ websearch_to_tsquery('english', search_term)
    ORDER BY score DESC
    LIMIT max_results;

    -- 결과가 없으면 퍼지 매칭으로 폴백
    IF NOT FOUND THEN
        RETURN QUERY
        SELECT a.id, a.title, similarity(a.title, search_term) AS score
        FROM articles a
        WHERE a.title % search_term OR a.body % search_term
        ORDER BY score DESC
        LIMIT max_results;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

---

## 8. 연습 문제

### 연습 1: 제품 검색
전자상거래 제품 카탈로그를 위한 전문 검색 시스템을 구축하세요.

```sql
-- 예시 답안
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    price NUMERIC(10,2),
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(category, '')), 'C')
    ) STORED
);

CREATE INDEX idx_products_search ON products USING GIN (search_vector);

-- 카테고리 필터가 있는 검색 함수
CREATE OR REPLACE FUNCTION search_products(
    search_term TEXT,
    category_filter TEXT DEFAULT NULL,
    max_results INT DEFAULT 20
)
RETURNS TABLE(id INT, name TEXT, price NUMERIC, rank REAL) AS $$
BEGIN
    RETURN QUERY
    SELECT p.id, p.name, p.price,
           ts_rank(p.search_vector, websearch_to_tsquery('english', search_term)) AS rank
    FROM products p
    WHERE p.search_vector @@ websearch_to_tsquery('english', search_term)
      AND (category_filter IS NULL OR p.category = category_filter)
    ORDER BY rank DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;
```

### 연습 2: 하이라이팅이 있는 검색
하이라이트된 매치가 있는 검색 결과를 생성하세요.

```sql
-- 예시 답안
SELECT
    id,
    ts_headline('english', name, query,
        'StartSel=<mark>, StopSel=</mark>') AS highlighted_name,
    ts_headline('english', description, query,
        'StartSel=<mark>, StopSel=</mark>, MaxFragments=2, FragmentDelimiter= ... ') AS highlighted_desc,
    ts_rank(search_vector, query) AS relevance
FROM products, websearch_to_tsquery('english', 'wireless bluetooth') AS query
WHERE search_vector @@ query
ORDER BY relevance DESC;
```

### 연습 3: 다국어 검색
영어와 simple(형태소 분석 없음) 텍스트를 모두 처리하는 검색 시스템을 설계하세요.

```sql
-- 예시 답안
CREATE TABLE multilingual_docs (
    id SERIAL PRIMARY KEY,
    content_en TEXT,
    content_raw TEXT,
    search_en tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content_en, ''))
    ) STORED,
    search_simple tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(content_raw, ''))
    ) STORED
);

CREATE INDEX idx_ml_en ON multilingual_docs USING GIN (search_en);
CREATE INDEX idx_ml_simple ON multilingual_docs USING GIN (search_simple);

-- 양쪽에 걸쳐 검색
SELECT id, content_en
FROM multilingual_docs
WHERE search_en @@ websearch_to_tsquery('english', 'search term')
   OR search_simple @@ websearch_to_tsquery('simple', 'search term');
```

---

**이전**: [테이블 파티셔닝](./18_Table_Partitioning.md) | **다음**: [보안과 접근 제어](./20_Security_Access_Control.md)

## 참고 자료
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- [pg_trgm Module](https://www.postgresql.org/docs/current/pgtrgm.html)
- [Text Search Functions](https://www.postgresql.org/docs/current/functions-textsearch.html)
