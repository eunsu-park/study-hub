# 19. Full-Text Search

**Previous**: [Table Partitioning](./18_Table_Partitioning.md) | **Next**: [Security and Access Control](./20_Security_Access_Control.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why full-text search outperforms LIKE/ILIKE for text retrieval in terms of indexing, stemming, and ranking
2. Create and manipulate tsvector (document representation) and tsquery (query representation) data types
3. Configure text search dictionaries, stop words, and stemming for different languages
4. Build GIN and GiST indexes on tsvector columns and compare their trade-offs
5. Rank search results using ts_rank, ts_rank_cd, and weighted search with setweight
6. Implement advanced search features including phrase search, prefix matching, and highlighted snippets
7. Use the pg_trgm extension for fuzzy matching, typo tolerance, and LIKE query acceleration

---

Every application with user-generated content eventually needs a search feature. While LIKE queries work for small datasets, they become painfully slow on large tables and cannot understand language -- "running" will not match "run," and there is no way to rank results by relevance. PostgreSQL's built-in full-text search provides linguistic-aware indexing with stemming, stop-word removal, boolean operators, and relevance ranking -- all without an external search engine like Elasticsearch. For many applications, this built-in capability is more than sufficient, and it keeps your architecture simple.

## Table of Contents

Before the FTS reference, read [**Theory & Principles**](#theory--principles) — what `tsvector` actually stores (an inverted-index-friendly tokenized form), how GIN's posting lists make `@@` queries fast, the math behind `ts_rank`, and how `pg_trgm` extends FTS to fuzzy matching.

1. [Full-Text Search Overview](#1-full-text-search-overview)
2. [tsvector and tsquery](#2-tsvector-and-tsquery)
3. [Search Configuration](#3-search-configuration)
4. [GIN and GiST Indexes](#4-gin-and-gist-indexes)
5. [Ranking and Weights](#5-ranking-and-weights)
6. [Advanced Search Techniques](#6-advanced-search-techniques)
7. [pg_trgm for Fuzzy Matching](#7-pg_trgm-for-fuzzy-matching)
8. [Practice Problems](#8-practice-problems)

---

## Theory & Principles

Full-text search is an entirely different kind of indexing problem from "look up a row by id". You are matching natural-language tokens, possibly stemmed, possibly with fuzzy spelling, and ranking by some notion of relevance. PostgreSQL implements this with **`tsvector`** (a sorted token list with positions), **`tsquery`** (a parsed query expression), the `@@` match operator, GIN as the underlying inverted index, `ts_rank` as the relevance score, and the `pg_trgm` extension for fuzzy/prefix matching that the standard FTS cannot express.

This section covers:

- **(A)** Tokenization: how text becomes a `tsvector` (parser → dictionaries → stemmed lexemes).
- **(B)** GIN as inverted index for FTS: posting lists, the `@@` operator's evaluation algorithm.
- **(C)** Ranking: `ts_rank` and `ts_rank_cd`, the cover-density formula, and weights A/B/C/D.
- **(D)** Trigrams: `pg_trgm`, similarity, and how it accelerates `LIKE '%substring%'`.

### A. Tokenization — From Text to `tsvector`

A `tsvector` is a *sorted, deduplicated list of normalized tokens* (called **lexemes**) with positional metadata. Producing one from raw text goes through three stages controlled by a **text search configuration** (e.g., `english`, `simple`, `korean`).

#### A.1 Parser — text into raw tokens

The parser splits the input into tokens classified by type — `word`, `email`, `url`, `number`, etc. For English, "The quick brown fox" yields four word tokens.

#### A.2 Dictionaries — tokens into lexemes

Each token is processed by an ordered chain of **dictionaries** based on its type. Common operations:

- **stop-word removal** — drop "the", "a", "is" (English `english_stop`).
- **stemming** — collapse "running", "runs", "ran" to lexeme "run" (English snowball stemmer).
- **synonyms / thesaurus** — replace "USA" with "united states".
- **simple lowercase** — for `simple` config that does no stemming.

#### A.3 Result — sorted lexemes with positions

```sql
SELECT to_tsvector('english', 'The quick brown foxes jumped');
-- 'brown':3 'fox':4 'jump':5 'quick':2
```

"The" was dropped (stop word), "foxes" became "fox" (stemmed), "jumped" became "jump". Each lexeme keeps a position number — required for phrase queries.

### B. GIN and the `@@` Match

#### B.1 Storage

A GIN index on a `tsvector` column stores, for each lexeme, the list of row TIDs whose `tsvector` contains that lexeme. This is exactly the JSONB-GIN structure of lesson 14, specialized for text.

#### B.2 The `@@` operator

`tsvector @@ tsquery` evaluates the query expression against the document. For `'fox & jumped'::tsquery`, the executor:

1. Parse the query into an expression tree: `AND(fox, jump)` (after stemming).
2. For each leaf (lexeme), look up the GIN posting list.
3. Combine the posting lists per the operator: AND = intersection, OR = union, NOT = difference.
4. Return the matching TIDs.

`tsquery` supports `&` (AND), `|` (OR), `!` (NOT), `<->` (phrase: tokens adjacent), `<N>` (tokens N positions apart), and grouping with parentheses.

#### B.3 GiST as the alternative

`USING gin (tsv)` is the typical choice — fast lookups, slow inserts. `USING gist (tsv)` is a *signature-based* index that is faster to insert but produces *false positives* requiring recheck against the actual `tsvector`. Useful for very write-heavy workloads where index size or insert latency matters more than lookup speed.

### C. Ranking — `ts_rank` and `ts_rank_cd`

Returning matching rows is only half the job; you usually want them ranked by relevance.

#### C.1 `ts_rank` — frequency-based

```sql
ts_rank(tsv, query, normalization) → float4
```

Implements a TF-IDF-flavored score: a lexeme that appears multiple times in the document boosts the score; the score is normalized by document length depending on the `normalization` flag. Default is no length normalization.

#### C.2 `ts_rank_cd` — cover-density

`cd` = cover density. Considers how *close together* the matching lexemes appear: a document where all query terms are clustered in one paragraph scores higher than one where they are scattered. More expensive to compute but generally produces better-feeling rankings for multi-word queries.

#### C.3 Weights — boosting parts of the document

Documents often have structure (title, summary, body, tags). You can store these in one `tsvector` with **weights**:

```sql
setweight(to_tsvector('english', title), 'A') ||
setweight(to_tsvector('english', body), 'B')
```

`A` is the highest weight, `D` the lowest. `ts_rank` accepts a `weights` array (`{0.1, 0.2, 0.4, 1.0}` for D, C, B, A) that multiplies the contribution of lexemes by their weight.

### D. Trigrams — `pg_trgm`

FTS is great for whole-word matching but fails for typos, partial words, and substring search. `pg_trgm` is the standard extension that fills these gaps.

#### D.1 Trigram extraction

A **trigram** is a 3-character substring. `pg_trgm` extracts all trigrams from a string:

```
'apple' → {'  a', ' ap', 'app', 'ppl', 'ple', 'le '}
```

Two leading and one trailing space normalize word boundaries.

#### D.2 Similarity

`similarity(s1, s2)` returns the Jaccard similarity of the two trigram sets: `|intersection| / |union|`, in [0, 1]. `'apple' % 'apples'` returns true if their similarity exceeds `pg_trgm.similarity_threshold` (default 0.3).

#### D.3 GIN/GiST on trigrams

`CREATE INDEX ON t USING gin (col gin_trgm_ops);` stores the trigram inverted index. This makes `WHERE col LIKE '%substr%'` (which is non-sargable for normal B-tree, lesson 5 §B.2) sargable: the planner extracts trigrams from `'%substr%'`, looks them up in GIN, intersects, and returns candidate TIDs.

This is the standard solution for accelerating arbitrary `LIKE '%X%'` queries.

### From Theory to the SQL Below

Each of the following sections is one of these mechanisms made concrete:

- **`to_tsvector('english', text)`** — runs the §A pipeline.
- **`to_tsquery('english', 'fox & jumped')`** — parses query syntax, applies same stemming.
- **`tsv @@ tsq`** — match operator (§B.2).
- **`CREATE INDEX ON t USING gin (tsv)`** — accelerates `@@` (§B.1).
- **`ts_rank`, `ts_rank_cd`** — relevance scores (§C).
- **`setweight(tsv, 'A')`** — assign weights to document parts (§C.3).
- **`pg_trgm` extension + `gin_trgm_ops`** — fuzzy/substring matching (§D).

---

## 1. Full-Text Search Overview

### 1.1 Why Full-Text Search?

```
┌─────────────────────────────────────────────────────────────────┐
│              LIKE vs Full-Text Search                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LIKE / ILIKE:                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ SELECT * FROM articles                   │                   │
│  │ WHERE body ILIKE '%database%';           │                   │
│  │                                          │                   │
│  │ ✗ No index usage (sequential scan)       │                   │
│  │ ✗ No linguistic awareness                │                   │
│  │ ✗ No ranking capability                  │                   │
│  │ ✗ "databases" won't match "database"     │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  Full-Text Search:                                              │
│  ┌──────────────────────────────────────────┐                   │
│  │ SELECT * FROM articles                   │                   │
│  │ WHERE to_tsvector(body)                  │                   │
│  │       @@ to_tsquery('database');          │                   │
│  │                                          │                   │
│  │ ✓ GIN index support (fast)               │                   │
│  │ ✓ Stemming (databases → database)        │                   │
│  │ ✓ Ranking by relevance                   │                   │
│  │ ✓ Boolean operators (AND, OR, NOT)       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Full-Text Search Pipeline                           │
│                                                                 │
│  Document Text                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   Parser         │──▶ Tokenize into words                   │
│  └─────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │  Dictionaries    │──▶ Normalize (stop words, stems)         │
│  └─────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   tsvector       │──▶ Sorted list of lexemes + positions    │
│  └─────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │  GIN Index       │──▶ Fast lookup by lexeme                 │
│  └─────────────────┘                                            │
│                                                                 │
│  Query Text ──▶ tsquery ──▶ Match against tsvector             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. tsvector and tsquery

### 2.1 tsvector — Document Representation

```sql
-- tsvector normalizes text linguistically: "foxes" → "fox", "jumped" → "jump"
-- This is why FTS matches word variants that LIKE '%fox%' would miss entirely
SELECT to_tsvector('english', 'The quick brown foxes jumped over the lazy dogs');
-- Result: 'brown':3 'dog':9 'fox':4 'jump':5 'lazi':8 'quick':2

-- Notice: stop words removed ("the", "over"), words stemmed ("foxes" → "fox")

-- tsvector stores lexemes with positions
SELECT to_tsvector('english', 'A fat cat sat on a mat. A fat cat ate a fat rat.');
-- Result: 'ate':10 'cat':3,8 'fat':2,7,11 'mat':6 'rat':12 'sat':4
```

### 2.2 tsquery — Query Representation

```sql
-- Basic query
SELECT to_tsquery('english', 'cats & dogs');
-- Result: 'cat' & 'dog'

-- Boolean operators
SELECT to_tsquery('english', 'cat | dog');        -- OR
SELECT to_tsquery('english', 'cat & !dog');       -- AND NOT
SELECT to_tsquery('english', 'cat <-> dog');      -- FOLLOWED BY (phrase)
SELECT to_tsquery('english', 'cat <2> dog');      -- within 2 words

-- plainto_tsquery: simpler syntax (implicit AND)
SELECT plainto_tsquery('english', 'fat cats');
-- Result: 'fat' & 'cat'

-- phraseto_tsquery: exact phrase matching
SELECT phraseto_tsquery('english', 'fat cats');
-- Result: 'fat' <-> 'cat'

-- websearch_to_tsquery: web-style syntax (PostgreSQL 11+)
SELECT websearch_to_tsquery('english', '"fat cats" -dogs');
-- Result: 'fat' <-> 'cat' & !'dog'

SELECT websearch_to_tsquery('english', 'cats or dogs');
-- Result: 'cat' | 'dog'
```

### 2.3 The Match Operator (@@)

```sql
-- @@ is the core FTS operator: it checks whether a document (tsvector) satisfies a query (tsquery)
-- Unlike LIKE, @@ can use GIN indexes and understands linguistic equivalence (stem matching)
SELECT to_tsvector('english', 'The fat cats sat on the mat')
       @@ to_tsquery('english', 'cat & mat');
-- Result: true

SELECT to_tsvector('english', 'The fat cats sat on the mat')
       @@ to_tsquery('english', 'cat & dog');
-- Result: false

-- Practical example
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

-- Search articles
SELECT title, ts_rank(to_tsvector('english', body), query) AS rank
FROM articles, to_tsquery('english', 'database & index') AS query
WHERE to_tsvector('english', body) @@ query
ORDER BY rank DESC;
```

### 2.4 Stored tsvector Column

```sql
-- Pre-compute tsvector in a STORED column — avoids re-parsing text on every query.
-- setweight assigns 'A' (highest) to title matches and 'B' to body, so title hits rank higher.
ALTER TABLE articles ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(body, '')), 'B')
    ) STORED;

-- Now queries use the pre-computed column
SELECT title
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database');
```

---

## 3. Search Configuration

### 3.1 Text Search Configurations

```sql
-- List available configurations
SELECT cfgname FROM pg_ts_config;

-- Common configurations:
-- 'simple'    — no stemming, no stop words
-- 'english'   — English stemming and stop words
-- 'german'    — German stemming
-- 'spanish'   — Spanish stemming

-- Show current default
SHOW default_text_search_config;

-- Set default
SET default_text_search_config = 'english';

-- Compare configurations
SELECT to_tsvector('simple', 'The running dogs are quickly jumping');
-- Result: 'are':4 'dogs':3 'jumping':6 'quickly':5 'running':2 'the':1

SELECT to_tsvector('english', 'The running dogs are quickly jumping');
-- Result: 'dog':3 'jump':6 'quick':5 'run':2
```

### 3.2 Dictionaries and Stop Words

```sql
-- Show dictionaries for a configuration
SELECT * FROM ts_debug('english', 'The quick brown foxes are jumping');

-- Output shows token → dictionary → lexeme mapping
-- Token     | Dictionary   | Lexemes
-- ----------|-------------|--------
-- The       | english_stem | {stop word}
-- quick     | english_stem | {quick}
-- brown     | english_stem | {brown}
-- foxes     | english_stem | {fox}
-- are       | english_stem | {stop word}
-- jumping   | english_stem | {jump}
```

### 3.3 Custom Configuration

```sql
-- Create custom configuration based on English
CREATE TEXT SEARCH CONFIGURATION my_english (COPY = english);

-- Add synonym dictionary
CREATE TEXT SEARCH DICTIONARY my_synonyms (
    TEMPLATE = synonym,
    SYNONYMS = my_synonyms  -- references $SHAREDIR/tsearch_data/my_synonyms.syn
);

-- Add to configuration
ALTER TEXT SEARCH CONFIGURATION my_english
    ALTER MAPPING FOR asciiword
    WITH my_synonyms, english_stem;
```

---

## 4. GIN and GiST Indexes

### 4.1 GIN Index for Full-Text Search

```sql
-- GIN index maps each lexeme to the rows containing it — turns @@ from a sequential scan
-- into a bitmap index scan, making FTS viable on million-row tables
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- Expression index: no stored column needed, but rebuilds tsvector during index creation
-- and cannot benefit from pre-computed weights (A/B/C/D)
CREATE INDEX idx_articles_body_gin ON articles
    USING GIN (to_tsvector('english', body));

-- Query uses the index automatically
EXPLAIN ANALYZE
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'database');
-- Bitmap Index Scan on idx_articles_search
```

### 4.2 GIN vs GiST Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│              GIN vs GiST for Full-Text Search                   │
├────────────────┬──────────────────┬──────────────────────────────┤
│                │ GIN              │ GiST                         │
├────────────────┼──────────────────┼──────────────────────────────┤
│ Build speed    │ Slower           │ Faster                       │
│ Search speed   │ Faster (3x)      │ Slower                       │
│ Index size     │ Larger           │ Smaller                      │
│ Update cost    │ Higher           │ Lower                        │
│ Exact results  │ Yes              │ May have false positives      │
│ Best for       │ Static data,     │ Frequently updated data,     │
│                │ read-heavy       │ write-heavy                  │
└────────────────┴──────────────────┴──────────────────────────────┘
```

```sql
-- GiST index (alternative)
CREATE INDEX idx_articles_search_gist ON articles USING GiST (search_vector);

-- GIN with fastupdate (default on, good for batch inserts)
CREATE INDEX idx_articles_search_gin ON articles
    USING GIN (search_vector) WITH (fastupdate = on);
```

### 4.3 Index Maintenance

```sql
-- Check index size
SELECT pg_size_pretty(pg_relation_size('idx_articles_search'));

-- Reindex if needed
REINDEX INDEX idx_articles_search;

-- Analyze for query planner
ANALYZE articles;
```

---

## 5. Ranking and Weights

### 5.1 ts_rank

```sql
-- Basic ranking
SELECT
    title,
    ts_rank(search_vector, to_tsquery('english', 'database')) AS rank
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database')
ORDER BY rank DESC;

-- Normalization options (bitmask)
-- 0  = default (document length ignored)
-- 1  = divide by 1 + log(document length)
-- 2  = divide by document length
-- 4  = divide by mean harmonic distance between extents
-- 8  = divide by number of unique words
-- 16 = divide by 1 + log(unique words)
-- 32 = divide by itself + 1

SELECT
    title,
    ts_rank(search_vector, query, 2) AS rank_normalized  -- normalize by length
FROM articles, to_tsquery('english', 'database') AS query
WHERE search_vector @@ query
ORDER BY rank_normalized DESC;
```

### 5.2 ts_rank_cd (Cover Density)

```sql
-- Cover density ranking considers proximity of matching terms
SELECT
    title,
    ts_rank_cd(search_vector, to_tsquery('english', 'database & index')) AS rank_cd
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database & index')
ORDER BY rank_cd DESC;
```

### 5.3 Weighted Search

```sql
-- Weights: A (1.0), B (0.4), C (0.2), D (0.1)
-- Custom weights array: {D, C, B, A}

-- Apply custom weights
SELECT
    title,
    ts_rank('{0.1, 0.2, 0.4, 1.0}', search_vector, query) AS weighted_rank
FROM articles, to_tsquery('english', 'database') AS query
WHERE search_vector @@ query
ORDER BY weighted_rank DESC;

-- Build weighted tsvector
SELECT
    setweight(to_tsvector('english', 'PostgreSQL Guide'), 'A') ||
    setweight(to_tsvector('english', 'A comprehensive database tutorial'), 'B') ||
    setweight(to_tsvector('english', 'Learn SQL queries and optimization'), 'C');
```

### 5.4 Highlighting Search Results

```sql
-- ts_headline highlights matching terms
SELECT
    title,
    ts_headline('english', body, to_tsquery('english', 'database'),
        'StartSel=<b>, StopSel=</b>, MaxWords=35, MinWords=15'
    ) AS highlighted
FROM articles
WHERE search_vector @@ to_tsquery('english', 'database');

-- Options:
-- StartSel / StopSel — highlight markers
-- MaxWords / MinWords — context window size
-- ShortWord — minimum word length to show
-- MaxFragments — number of fragments (0 = whole document)
-- FragmentDelimiter — separator between fragments
```

---

## 6. Advanced Search Techniques

### 6.1 Multi-Column Search

```sql
-- Search across multiple columns with different weights
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

### 6.2 Phrase Search

```sql
-- Exact phrase: "full text search"
SELECT title FROM articles
WHERE search_vector @@ phraseto_tsquery('english', 'full text search');

-- Proximity: words within N positions
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'full <3> search');
-- "full" and "search" within 3 words of each other
```

### 6.3 Search with Filters

```sql
-- Combine full-text search with regular WHERE clauses
SELECT title, ts_rank(search_vector, query) AS rank
FROM articles, to_tsquery('english', 'database') AS query
WHERE search_vector @@ query
  AND created_at >= NOW() - INTERVAL '30 days'
ORDER BY rank DESC
LIMIT 10;

-- Composite index for combined search
CREATE INDEX idx_articles_date_search ON articles
    USING GIN (search_vector) WHERE created_at >= '2024-01-01';
```

### 6.4 Auto-Complete / Prefix Search

```sql
-- Prefix matching with :*
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'dat:*');
-- Matches: database, data, date, etc.

-- Practical autocomplete function
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

### 6.5 Search Statistics

```sql
-- Most common words in a corpus
SELECT word, ndoc, nentry
FROM ts_stat('SELECT search_vector FROM articles')
ORDER BY nentry DESC
LIMIT 20;

-- Word frequency for specific query
SELECT word, ndoc
FROM ts_stat('SELECT search_vector FROM articles')
WHERE word LIKE 'data%'
ORDER BY ndoc DESC;
```

---

## 7. pg_trgm for Fuzzy Matching

### 7.1 Trigram Basics

```sql
-- pg_trgm complements FTS: FTS requires exact lexeme matches (after stemming),
-- but pg_trgm handles typos and partial strings using 3-character subsequences
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Show trigrams
SELECT show_trgm('PostgreSQL');
-- Result: {"  p"," po","gre","osq","pos","ostg","pgr","ql ","res","sql","stg","tgr"}

-- Similarity score (0 to 1)
SELECT similarity('PostgreSQL', 'Postgresql');  -- ~0.75
SELECT similarity('PostgreSQL', 'MySQL');       -- ~0.09
```

### 7.2 Trigram Indexes

```sql
-- GIN trigram index
CREATE INDEX idx_articles_title_trgm ON articles USING GIN (title gin_trgm_ops);

-- GiST trigram index (supports distance operator)
CREATE INDEX idx_articles_title_trgm_gist ON articles USING GiST (title gist_trgm_ops);

-- Similarity search
SELECT title, similarity(title, 'Postgre') AS sim
FROM articles
WHERE title % 'Postgre'  -- % operator uses similarity threshold
ORDER BY sim DESC;

-- Set similarity threshold (default 0.3)
SET pg_trgm.similarity_threshold = 0.2;

-- Distance operator (GiST only)
SELECT title, title <-> 'Postgre' AS distance
FROM articles
ORDER BY title <-> 'Postgre'
LIMIT 5;
```

### 7.3 LIKE/ILIKE with Trigram Index

```sql
-- Trigram index accelerates LIKE queries too
CREATE INDEX idx_articles_body_trgm ON articles USING GIN (body gin_trgm_ops);

-- These queries now use the index
SELECT title FROM articles WHERE body LIKE '%database%';
SELECT title FROM articles WHERE body ILIKE '%DATABASE%';

-- Regular expression queries also benefit
SELECT title FROM articles WHERE body ~ 'data(base|set)';
```

### 7.4 Combining FTS and pg_trgm

```sql
-- Best of both: full-text search for relevance, trigrams for typo tolerance
CREATE OR REPLACE FUNCTION smart_search(search_term TEXT, max_results INT DEFAULT 10)
RETURNS TABLE(id INT, title TEXT, score REAL) AS $$
BEGIN
    -- Try exact full-text search first
    RETURN QUERY
    SELECT a.id, a.title,
           ts_rank(a.search_vector, websearch_to_tsquery('english', search_term)) AS score
    FROM articles a
    WHERE a.search_vector @@ websearch_to_tsquery('english', search_term)
    ORDER BY score DESC
    LIMIT max_results;

    -- If no results, fall back to fuzzy matching
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

## 8. Practice Problems

### Exercise 1: Product Search
Build a full-text search system for an e-commerce product catalog.

```sql
-- Example answer
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

-- Search function with category filter
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

### Exercise 2: Search with Highlighting
Create a search result with highlighted matches.

```sql
-- Example answer
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

### Exercise 3: Multilingual Search
Design a search system that handles both English and simple (no stemming) text.

```sql
-- Example answer
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

-- Search across both
SELECT id, content_en
FROM multilingual_docs
WHERE search_en @@ websearch_to_tsquery('english', 'search term')
   OR search_simple @@ websearch_to_tsquery('simple', 'search term');
```

---

## References
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- [pg_trgm Module](https://www.postgresql.org/docs/current/pgtrgm.html)
- [Text Search Functions](https://www.postgresql.org/docs/current/functions-textsearch.html)

---

**Previous**: [Table Partitioning](./18_Table_Partitioning.md) | **Next**: [Security and Access Control](./20_Security_Access_Control.md)
