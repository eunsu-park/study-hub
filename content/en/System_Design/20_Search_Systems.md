# 20. Search Systems

**Previous**: [Observability and Monitoring](./19_Observability_Monitoring.md)

Difficulty: ⭐⭐⭐⭐

## Learning Objectives

1. Explain how an inverted index works and why it enables sub-second full-text search over billions of documents
2. Describe the text analysis pipeline (tokenization, normalization, stemming, stop words) and its impact on search quality
3. Design an Elasticsearch cluster architecture with appropriate shard counts, replica strategies, and node roles
4. Construct Elasticsearch queries using match, bool, multi-match, and function-score queries to meet different search requirements
5. Explain the BM25 ranking algorithm and techniques for boosting relevance (field weighting, decay functions, custom scoring)
6. Scale a search system horizontally through index sharding, routing, caching, and tiered storage strategies
7. Implement autocomplete and search-as-you-type features using edge n-grams and completion suggesters

---

Search is so deeply embedded in how we use software that users notice immediately when it does not work well -- returning irrelevant results, feeling slow, or missing obvious matches. Behind every search box, from e-commerce product search to log exploration in Kibana, sits an inverted index coupled with a carefully tuned ranking algorithm. Understanding search internals lets you design systems that return the right results in milliseconds, even as the corpus grows to hundreds of millions of documents.

## Table of Contents

1. [Search Engine Fundamentals](#1-search-engine-fundamentals)
2. [Inverted Index](#2-inverted-index)
3. [Elasticsearch Architecture](#3-elasticsearch-architecture)
4. [Indexing and Mapping](#4-indexing-and-mapping)
5. [Search Queries](#5-search-queries)
6. [Ranking and Relevance](#6-ranking-and-relevance)
7. [Scaling Search Systems](#7-scaling-search-systems)
8. [Practice Problems](#8-practice-problems)

---

## 1. Search Engine Fundamentals

### 1.1 Search System Components

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

### 1.2 Full-Text Search vs Database Search

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

## 2. Inverted Index

### 2.1 How It Works

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

### 2.2 Text Analysis Pipeline

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

### 2.3 Posting List with Positions

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

## 3. Elasticsearch Architecture

### 3.1 Cluster Architecture

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

### 3.2 Index, Shard, and Segment

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

### 3.3 Write and Read Paths

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

## 4. Indexing and Mapping

### 4.1 Index Creation

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

### 4.2 Field Types

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

## 5. Search Queries

### 5.1 Match Queries

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

### 5.2 Boolean Queries

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

### 5.3 Aggregations

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

### 5.4 Autocomplete / Suggestions

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

## 6. Ranking and Relevance

### 6.1 BM25 Scoring

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

### 6.2 Boosting and Custom Scoring

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

### 6.3 Explain API

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

## 7. Scaling Search Systems

### 7.1 Shard Sizing Strategy

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

### 7.2 Index Lifecycle Management (ILM)

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

### 7.3 Search Performance Optimization

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

## 8. Practice Problems

### Problem 1: E-Commerce Search
Design a search system for an e-commerce platform with 10M products.

```
Requirements:
- Full-text search across name, description, brand
- Category faceting
- Price range filtering
- Sort by relevance, price, rating, newest
- Autocomplete suggestions
- Typo tolerance

Key decisions:
- Index design: 5 primary shards (10M docs ≈ 20GB)
- Custom analyzer with synonyms (e.g., "phone" = "smartphone")
- Completion suggester for autocomplete
- Function score boosting popular products
- Fuzziness for typo tolerance
```

### Problem 2: Log Search Platform
Design a centralized log search system for 50TB/day of logs.

```
Key considerations:
- Write-heavy workload (50TB/day = ~580MB/s)
- Time-based indices (daily or hourly rollover)
- Hot-warm-cold architecture
- 7-day search window (hot), 30-day archive (cold)
- Structured fields: timestamp, level, service, message

Architecture:
- Kafka as ingestion buffer
- Logstash/Vector for parsing
- Hot nodes: 10 × (32GB RAM, 2TB NVMe SSD)
- Warm nodes: 6 × (32GB RAM, 8TB HDD)
- ILM policy: rollover at 50GB, move to warm at 24h
- Estimated cost vs ELK Cloud vs Loki comparison
```

### Problem 3: Autocomplete System
Design a search autocomplete that returns results in < 50ms.

```
Approach:
1. Completion suggester for prefix matching
2. Edge n-gram analyzer for partial word matching
3. Separate lightweight index for suggestions
4. Cache popular queries in Redis (TTL: 5min)
5. Client-side debouncing (300ms)

Index design:
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

Query pipeline:
User types → debounce → edge-ngram match + completion suggest
→ boost by popularity → return top 10
```

---

**Previous**: [Observability and Monitoring](./19_Observability_Monitoring.md)

---

## References
- [Elasticsearch: The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Lucene Scoring (BM25)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Elasticsearch Performance Tuning](https://www.elastic.co/guide/en/elasticsearch/reference/current/tune-for-search-speed.html)
- [Designing Data-Intensive Applications (Ch. 3)](https://dataintensive.net/)
