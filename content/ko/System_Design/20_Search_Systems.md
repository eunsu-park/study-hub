# 20. 검색 시스템

**이전**: [관측 가능성과 모니터링](./19_Observability_Monitoring.md)

난이도: ⭐⭐⭐⭐

## 학습 목표(Learning Objectives)

**난이도**: ⭐⭐⭐⭐ (고급)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 역색인(Inverted Index)이 어떻게 작동하는지 설명하고, 수십억 개의 문서에서 1초 미만의 전문 검색(full-text search)을 가능하게 하는 이유를 설명할 수 있다
2. 텍스트 분석 파이프라인(토큰화, 정규화, 어간 추출, 불용어 처리)을 설명하고 검색 품질에 미치는 영향을 설명할 수 있다
3. 적절한 샤드 수, 레플리카 전략, 노드 역할을 고려하여 Elasticsearch 클러스터 아키텍처를 설계할 수 있다
4. match, bool, multi-match, function-score 쿼리를 사용하여 다양한 검색 요구 사항을 충족하는 Elasticsearch 쿼리를 작성할 수 있다
5. BM25 랭킹 알고리즘과 관련성 향상 기법(필드 가중치, 감쇠 함수, 커스텀 스코어링)을 설명할 수 있다
6. 인덱스 샤딩(index sharding), 라우팅, 캐싱, 계층형 스토리지 전략을 통해 검색 시스템을 수평 확장할 수 있다
7. 엣지 n-그램(edge n-gram)과 완성 제안기(completion suggester)를 사용하여 자동완성 및 입력 중 검색(search-as-you-type) 기능을 구현할 수 있다

---

검색은 소프트웨어 사용 방식에 너무 깊이 내재되어 있어서, 관련 없는 결과가 나오거나 느리게 느껴지거나 명백한 일치 항목을 놓치는 등 제대로 작동하지 않으면 사용자는 즉시 알아챕니다. 이커머스 상품 검색부터 Kibana의 로그 탐색까지, 모든 검색 창 뒤에는 역색인과 신중하게 조율된 랭킹 알고리즘이 자리하고 있습니다. 검색 내부 구조를 이해하면 코퍼스(corpus)가 수억 개의 문서로 증가하더라도 밀리초 안에 올바른 결과를 반환하는 시스템을 설계할 수 있습니다.

---

## 목차

1. [검색 엔진의 기초](#1-검색-엔진의-기초)
2. [역색인](#2-역색인)
3. [Elasticsearch 아키텍처](#3-elasticsearch-아키텍처)
4. [인덱싱과 매핑](#4-인덱싱과-매핑)
5. [검색 쿼리](#5-검색-쿼리)
6. [랭킹과 관련성](#6-랭킹과-관련성)
7. [검색 시스템 확장](#7-검색-시스템-확장)
8. [연습 문제](#8-연습-문제)

---

## 1. 검색 엔진의 기초

### 1.1 검색 시스템 구성 요소

```
┌─────────────────────────────────────────────────────────────────┐
│              Search System Architecture                          │
│                                                                 │
│  ┌────────────┐     ┌────────────┐     ┌────────────────────┐   │
│  │  Crawling / │     │  Indexing   │     │  Query Processing  │   │
│  │  Ingestion  │────▶│  Pipeline   │────▶│  & Retrieval       │   │
│  └────────────┘     └────────────┘     └────────────────────┘   │
│       │                   │                      │              │
│       ▼                   ▼                      ▼              │
│  ┌────────────┐     ┌────────────┐     ┌────────────────────┐   │
│  │ Fetch docs  │     │ Tokenize   │     │ Parse query        │   │
│  │ Extract text│     │ Analyze    │     │ Match documents    │   │
│  │ Normalize   │     │ Build index│     │ Rank results       │   │
│  │ Detect lang │     │ Store      │     │ Highlight          │   │
│  └────────────┘     └────────────┘     └────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 전문 검색 vs 데이터베이스 검색

```
┌──────────────────────┬────────────────────┬──────────────────────┐
│                      │ Database (SQL)     │ Search Engine (ES)   │
├──────────────────────┼────────────────────┼──────────────────────┤
│ Primary use          │ Transactional      │ Search & analytics   │
│ Data model           │ Structured rows    │ Semi-structured docs │
│ Query language       │ SQL                │ DSL (JSON)           │
│ Text matching        │ LIKE / FTS         │ Full-text + facets   │
│ Relevance scoring    │ Limited            │ Advanced (BM25)      │
│ Horizontal scaling   │ Complex (sharding) │ Native (shards)      │
│ Real-time updates    │ ACID guarantees    │ Near real-time       │
│ Aggregations         │ GROUP BY           │ Rich aggregations    │
│ Consistency          │ Strong             │ Eventual             │
└──────────────────────┴────────────────────┴──────────────────────┘
```

---

## 2. 역색인

### 2.1 작동 원리

```
┌─────────────────────────────────────────────────────────────────┐
│              Inverted Index                                      │
│                                                                 │
│  Documents:                                                     │
│  Doc1: "the quick brown fox"                                    │
│  Doc2: "the lazy brown dog"                                     │
│  Doc3: "the quick dog jumps"                                    │
│                                                                 │
│  Forward Index:                    Inverted Index:              │
│  Doc1 → [the, quick, brown, fox]  brown → [Doc1, Doc2]         │
│  Doc2 → [the, lazy, brown, dog]   dog   → [Doc2, Doc3]         │
│  Doc3 → [the, quick, dog, jumps]  fox   → [Doc1]               │
│                                   jumps → [Doc3]               │
│                                   lazy  → [Doc2]               │
│                                   quick → [Doc1, Doc3]         │
│                                   the   → [Doc1, Doc2, Doc3]   │
│                                                                 │
│  Query: "quick brown"                                           │
│  quick → {Doc1, Doc3}                                           │
│  brown → {Doc1, Doc2}                                           │
│  Intersection: {Doc1}  ← AND query result                      │
│  Union: {Doc1, Doc2, Doc3} ← OR query result                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 텍스트 분석 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│              Text Analysis Pipeline                              │
│                                                                 │
│  Input: "The Quick Brown Foxes are JUMPING!"                    │
│                                                                 │
│  1. Character Filters                                           │
│     ├── HTML strip: remove <tags>                               │
│     └── Pattern replace: normalize chars                        │
│     Result: "The Quick Brown Foxes are JUMPING!"                │
│                                                                 │
│  2. Tokenizer                                                   │
│     ├── Standard: split on word boundaries                      │
│     └── Result: ["The", "Quick", "Brown", "Foxes", "are",      │
│                  "JUMPING"]                                     │
│                                                                 │
│  3. Token Filters                                               │
│     ├── Lowercase: "the", "quick", "brown", "foxes", ...       │
│     ├── Stop words: remove "the", "are"                         │
│     ├── Stemmer: "foxes" → "fox", "jumping" → "jump"           │
│     └── Result: ["quick", "brown", "fox", "jump"]              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 위치를 포함한 포스팅 리스트

```
┌─────────────────────────────────────────────────────────────────┐
│              Posting List with Term Frequency & Positions        │
│                                                                 │
│  Term: "database"                                               │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ Doc1: tf=3, positions=[5, 12, 45]                   │        │
│  │ Doc3: tf=1, positions=[8]                           │        │
│  │ Doc7: tf=5, positions=[2, 10, 15, 30, 42]          │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
│  Uses:                                                          │
│  • tf (term frequency) → relevance scoring                     │
│  • positions → phrase queries, proximity queries                │
│  • doc frequency → IDF calculation                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Elasticsearch 아키텍처

### 3.1 클러스터 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│              Elasticsearch Cluster                               │
│                                                                 │
│  Cluster: "production"                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Node 1 (Master + Data)                                │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               │     │
│  │  │ Shard 0  │ │ Shard 2  │ │Replica 1 │               │     │
│  │  │ (Primary)│ │ (Primary)│ │          │               │     │
│  │  └──────────┘ └──────────┘ └──────────┘               │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Node 2 (Data)                                         │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               │     │
│  │  │ Shard 1  │ │Replica 0 │ │Replica 2 │               │     │
│  │  │ (Primary)│ │          │ │          │               │     │
│  │  └──────────┘ └──────────┘ └──────────┘               │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Node 3 (Data)                                         │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               │     │
│  │  │Replica 0 │ │Replica 1 │ │Replica 2 │               │     │
│  │  └──────────┘ └──────────┘ └──────────┘               │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  Index: "products" → 3 primary shards, 1 replica each          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 인덱스, 샤드, 세그먼트

```
┌─────────────────────────────────────────────────────────────────┐
│              Index → Shard → Segment                             │
│                                                                 │
│  Index (logical)                                                │
│  ├── Shard 0 (Lucene index)                                    │
│  │   ├── Segment 1 (immutable)                                 │
│  │   │   ├── Inverted index                                    │
│  │   │   ├── Stored fields                                     │
│  │   │   └── Doc values (columnar)                             │
│  │   ├── Segment 2 (immutable)                                 │
│  │   └── Segment 3 (immutable)                                 │
│  ├── Shard 1                                                   │
│  │   └── ...                                                   │
│  └── Shard 2                                                   │
│      └── ...                                                   │
│                                                                 │
│  New docs → in-memory buffer → refresh (1s) → new segment      │
│  Segments merge periodically (background)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 쓰기 및 읽기 경로

```
┌─────────────────────────────────────────────────────────────────┐
│  Write Path:                                                    │
│                                                                 │
│  Client ──▶ Coordinating Node ──▶ Primary Shard                │
│                                       │                        │
│                                       ├──▶ In-memory buffer    │
│                                       ├──▶ Translog (WAL)      │
│                                       └──▶ Replica Shards      │
│                                                                 │
│  Refresh (every 1s): buffer → searchable segment               │
│  Flush (every 30m):  translog → disk, clear translog           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Read Path:                                                     │
│                                                                 │
│  Client ──▶ Coordinating Node                                  │
│                    │                                            │
│            ┌───────┼───────┐                                    │
│            ▼       ▼       ▼                                    │
│         Shard 0  Shard 1  Shard 2  (scatter)                   │
│            │       │       │                                    │
│            └───────┼───────┘                                    │
│                    ▼                                            │
│           Merge & rank results (gather)                         │
│                    │                                            │
│                    ▼                                            │
│                 Client                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 인덱싱과 매핑

### 4.1 인덱스 생성

```json
PUT /products
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "product_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "english_stemmer", "english_stop"]
        }
      },
      "filter": {
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        },
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "analyzer": "product_analyzer",
        "fields": {
          "keyword": { "type": "keyword" },
          "suggest": {
            "type": "completion"
          }
        }
      },
      "description": {
        "type": "text",
        "analyzer": "product_analyzer"
      },
      "category": { "type": "keyword" },
      "price": { "type": "float" },
      "rating": { "type": "float" },
      "created_at": { "type": "date" },
      "in_stock": { "type": "boolean" },
      "tags": { "type": "keyword" }
    }
  }
}
```

### 4.2 필드 타입

```
┌────────────────┬──────────────────────────────────────────────────┐
│ Type           │ Use Case                                         │
├────────────────┼──────────────────────────────────────────────────┤
│ text           │ Full-text search (analyzed, tokenized)            │
│ keyword        │ Exact match, sorting, aggregations                │
│ integer/long   │ Numeric values, range queries                     │
│ float/double   │ Decimal numbers                                   │
│ date           │ Timestamps, date math                             │
│ boolean        │ true/false filters                                │
│ object         │ Nested JSON (flattened internally)                │
│ nested         │ Array of objects (maintains relationships)        │
│ geo_point      │ Latitude/longitude                                │
│ completion     │ Auto-complete suggestions                         │
│ dense_vector   │ ML embeddings, similarity search                  │
└────────────────┴──────────────────────────────────────────────────┘
```

---

## 5. 검색 쿼리

### 5.1 매치 쿼리

```json
// Full-text match (analyzed)
GET /products/_search
{
  "query": {
    "match": {
      "description": {
        "query": "wireless bluetooth headphones",
        "operator": "and"
      }
    }
  }
}

// Multi-match across fields
GET /products/_search
{
  "query": {
    "multi_match": {
      "query": "sony headphones",
      "fields": ["name^3", "description", "tags^2"],
      "type": "best_fields"
    }
  }
}
```

### 5.2 불린 쿼리

```json
// Compound boolean query
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "description": "wireless headphones" } }
      ],
      "filter": [
        { "term": { "category": "electronics" } },
        { "range": { "price": { "gte": 50, "lte": 200 } } },
        { "term": { "in_stock": true } }
      ],
      "should": [
        { "term": { "tags": "premium" } },
        { "range": { "rating": { "gte": 4.5 } } }
      ],
      "must_not": [
        { "term": { "tags": "refurbished" } }
      ],
      "minimum_should_match": 1
    }
  }
}
```

### 5.3 집계

```json
// Faceted search with aggregations
GET /products/_search
{
  "size": 0,
  "aggs": {
    "categories": {
      "terms": { "field": "category", "size": 20 }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 50, "key": "budget" },
          { "from": 50, "to": 200, "key": "mid-range" },
          { "from": 200, "key": "premium" }
        ]
      }
    },
    "avg_rating": {
      "avg": { "field": "rating" }
    },
    "top_rated_by_category": {
      "terms": { "field": "category" },
      "aggs": {
        "avg_rating": { "avg": { "field": "rating" } },
        "top_products": {
          "top_hits": {
            "size": 3,
            "sort": [{ "rating": "desc" }],
            "_source": ["name", "rating", "price"]
          }
        }
      }
    }
  }
}
```

### 5.4 자동완성 / 제안

```json
// Completion suggester
POST /products/_search
{
  "suggest": {
    "product-suggest": {
      "prefix": "wire",
      "completion": {
        "field": "name.suggest",
        "size": 5,
        "fuzzy": {
          "fuzziness": "AUTO"
        }
      }
    }
  }
}

// Search-as-you-type with match_phrase_prefix
GET /products/_search
{
  "query": {
    "match_phrase_prefix": {
      "name": {
        "query": "wireless blue",
        "max_expansions": 10
      }
    }
  }
}
```

---

## 6. 랭킹과 관련성

### 6.1 BM25 스코어링

```
┌─────────────────────────────────────────────────────────────────┐
│              BM25 Scoring (Elasticsearch default)                │
│                                                                 │
│  score(q, d) = Σ IDF(qi) × [f(qi,d) × (k1+1)]                │
│                              ────────────────────────           │
│                              f(qi,d) + k1×(1-b+b×|d|/avgdl)   │
│                                                                 │
│  Where:                                                         │
│  • IDF(qi) = inverse document frequency of term qi              │
│    (rare terms → higher score)                                  │
│  • f(qi,d) = term frequency of qi in document d                │
│    (more occurrences → higher, but with diminishing returns)   │
│  • |d| = document length                                       │
│  • avgdl = average document length                              │
│  • k1 = term saturation parameter (default 1.2)                │
│  • b = length normalization (default 0.75)                      │
│                                                                 │
│  Key insight: BM25 has diminishing returns for term frequency   │
│  (unlike TF-IDF which grows linearly)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 부스팅과 커스텀 스코어링

```json
// Field boosting
GET /products/_search
{
  "query": {
    "multi_match": {
      "query": "headphones",
      "fields": ["name^5", "description^2", "tags^3"]
    }
  }
}

// Function score for custom ranking
GET /products/_search
{
  "query": {
    "function_score": {
      "query": { "match": { "description": "headphones" } },
      "functions": [
        {
          "field_value_factor": {
            "field": "rating",
            "modifier": "log1p",
            "factor": 2
          }
        },
        {
          "gauss": {
            "created_at": {
              "origin": "now",
              "scale": "30d",
              "decay": 0.5
            }
          }
        }
      ],
      "score_mode": "multiply",
      "boost_mode": "multiply"
    }
  }
}
```

### 6.3 설명 API

```json
// Understand why a document scored the way it did
GET /products/_explain/doc_id
{
  "query": {
    "match": { "description": "wireless headphones" }
  }
}
```

---

## 7. 검색 시스템 확장

### 7.1 샤드 크기 전략

```
┌─────────────────────────────────────────────────────────────────┐
│              Shard Sizing Guidelines                              │
│                                                                 │
│  Rules of thumb:                                                │
│  • Target shard size: 10-50 GB                                  │
│  • Max shards per node: ~20 per GB of heap                      │
│  • Typical node heap: 32 GB → ~640 shards max                  │
│                                                                 │
│  Example calculation:                                           │
│  • Data: 500 GB, 1 replica                                     │
│  • Total: 1 TB (with replicas)                                  │
│  • Shard size target: 25 GB                                     │
│  • Primary shards: 500/25 = 20                                  │
│  • Total shards: 20 × 2 = 40                                   │
│  • Nodes needed: 40/640 ≈ 1, but 3 for HA                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 인덱스 수명 주기 관리 (ILM)

```
┌─────────────────────────────────────────────────────────────────┐
│              Index Lifecycle (Hot-Warm-Cold-Delete)               │
│                                                                 │
│  Hot Phase (0-7 days)                                           │
│  ├── Fast SSDs, high write throughput                           │
│  ├── Rollover at 50GB or 7 days                                │
│  └── Primary for indexing + search                              │
│                                                                 │
│  Warm Phase (7-30 days)                                         │
│  ├── Standard disks, read-only                                  │
│  ├── Force merge to 1 segment per shard                         │
│  └── Shrink replica count                                       │
│                                                                 │
│  Cold Phase (30-90 days)                                        │
│  ├── Object storage (S3) via searchable snapshots               │
│  ├── Minimal resources                                          │
│  └── Slower queries acceptable                                  │
│                                                                 │
│  Delete Phase (> 90 days)                                       │
│  └── Automatically delete expired indices                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 검색 성능 최적화

```
Performance checklist:

1. Mapping optimization
   - Use keyword for exact match (not text)
   - Disable _source for write-heavy indices
   - Use doc_values for sorting/aggregations

2. Query optimization
   - Use filter context (cacheable, no scoring)
   - Avoid wildcard queries on large fields
   - Use routing for tenant-specific data

3. Index optimization
   - Appropriate shard count (avoid over-sharding)
   - Force merge read-only indices
   - Use index sorting for common sort patterns

4. Hardware
   - 50% memory for OS file cache
   - SSDs for hot data
   - Dedicated master nodes in production
```

---

## 8. 연습 문제

### 문제 1: 이커머스 검색
천만 개의 제품이 있는 이커머스 플랫폼을 위한 검색 시스템을 설계하세요.

```
요구 사항:
- 이름, 설명, 브랜드에 대한 전문 검색
- 카테고리 패싯(faceting)
- 가격 범위 필터링
- 관련성, 가격, 평점, 최신순으로 정렬
- 자동완성 제안
- 오타 허용

주요 결정 사항:
- 인덱스 설계: 5개의 프라이머리 샤드 (1000만 문서 ≈ 20GB)
- 동의어가 있는 커스텀 분석기 (예: "phone" = "smartphone")
- 자동완성을 위한 완성 제안기(Completion suggester)
- 인기 제품 부스팅을 위한 함수 스코어
- 오타 허용을 위한 Fuzziness
```

### 문제 2: 로그 검색 플랫폼
하루 50TB의 로그를 위한 중앙 집중식 로그 검색 시스템을 설계하세요.

```
주요 고려 사항:
- 쓰기 중심 워크로드 (50TB/일 = ~580MB/초)
- 시간 기반 인덱스 (일별 또는 시간별 롤오버)
- Hot-warm-cold 아키텍처
- 7일 검색 윈도우 (hot), 30일 아카이브 (cold)
- 구조화된 필드: timestamp, level, service, message

아키텍처:
- 수집 버퍼로서 Kafka
- 파싱을 위한 Logstash/Vector
- Hot 노드: 10 × (32GB RAM, 2TB NVMe SSD)
- Warm 노드: 6 × (32GB RAM, 8TB HDD)
- ILM 정책: 50GB에서 롤오버, 24시간 후 warm으로 이동
- ELK Cloud vs Loki 비교 및 예상 비용
```

### 문제 3: 자동완성 시스템
50ms 미만으로 결과를 반환하는 검색 자동완성을 설계하세요.

```
접근법:
1. 접두사 매칭을 위한 완성 제안기(Completion suggester)
2. 부분 단어 매칭을 위한 엣지 n-gram 분석기
3. 제안을 위한 별도의 경량 인덱스
4. Redis에 인기 쿼리 캐시 (TTL: 5분)
5. 클라이언트 측 디바운싱 (300ms)

인덱스 설계:
{
  "suggest_text": {
    "type": "text",
    "analyzer": "edge_ngram_analyzer",
    "search_analyzer": "standard",
    "fields": {
      "suggest": { "type": "completion" }
    }
  },
  "popularity": { "type": "integer" }
}

쿼리 파이프라인:
사용자 입력 → 디바운스 → edge-ngram 매치 + 완성 제안
→ 인기도로 부스트 → 상위 10개 반환
```

---

**이전**: [관측 가능성과 모니터링](./19_Observability_Monitoring.md)

---

## 참고 자료
- [Elasticsearch: The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Lucene Scoring (BM25)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Elasticsearch Performance Tuning](https://www.elastic.co/guide/en/elasticsearch/reference/current/tune-for-search-speed.html)
- [Designing Data-Intensive Applications (Ch. 3)](https://dataintensive.net/)
