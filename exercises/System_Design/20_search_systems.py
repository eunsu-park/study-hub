"""
Exercises for Lesson 20: Search Systems
Topic: System_Design

Solutions to practice problems from the lesson.
Covers e-commerce search design, log search platform, and autocomplete system.
"""

import math
import time
import random
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# === Exercise 1: E-Commerce Search ===
# Problem: Design search for 10M products with full-text, faceting, filters, autocomplete.

@dataclass
class Product:
    product_id: str
    name: str
    description: str
    brand: str
    category: str
    price: float
    rating: float
    popularity: int  # Sales count


class InvertedIndex:
    """Simplified inverted index for full-text search."""

    def __init__(self):
        self.index = defaultdict(set)  # token -> set of doc_ids
        self.documents = {}  # doc_id -> Product
        self.synonyms = {}  # word -> canonical form

    def add_synonym(self, word, canonical):
        self.synonyms[word] = canonical

    def _tokenize(self, text):
        """Simple tokenizer with lowercasing and synonym resolution."""
        tokens = text.lower().split()
        resolved = [self.synonyms.get(t, t) for t in tokens]
        return resolved

    def index_product(self, product):
        """Index a product across searchable fields."""
        self.documents[product.product_id] = product

        # Index name, description, brand
        for token in self._tokenize(product.name):
            self.index[token].add(product.product_id)
        for token in self._tokenize(product.description):
            self.index[token].add(product.product_id)
        for token in self._tokenize(product.brand):
            self.index[token].add(product.product_id)

    def search(self, query, filters=None, sort_by="relevance", limit=10):
        """Search with BM25-like scoring, filtering, and sorting."""
        tokens = self._tokenize(query)

        # Find matching documents
        if not tokens:
            return []

        # Intersection of posting lists (AND query)
        matching_ids = None
        for token in tokens:
            # Fuzzy matching: also check tokens with edit distance 1
            candidates = self._fuzzy_lookup(token)
            if matching_ids is None:
                matching_ids = candidates
            else:
                matching_ids = matching_ids & candidates

        if not matching_ids:
            return []

        # Apply filters
        results = [self.documents[pid] for pid in matching_ids
                    if pid in self.documents]

        if filters:
            if "category" in filters:
                results = [p for p in results if p.category == filters["category"]]
            if "price_min" in filters:
                results = [p for p in results if p.price >= filters["price_min"]]
            if "price_max" in filters:
                results = [p for p in results if p.price <= filters["price_max"]]
            if "min_rating" in filters:
                results = [p for p in results if p.rating >= filters["min_rating"]]

        # Score and sort
        scored = []
        for product in results:
            if sort_by == "relevance":
                # BM25-like: term frequency + popularity boost
                tf = sum(1 for t in tokens
                         if t in product.name.lower() or t in product.description.lower())
                score = tf * 10 + product.popularity * 0.01 + product.rating
            elif sort_by == "price_asc":
                score = -product.price
            elif sort_by == "price_desc":
                score = product.price
            elif sort_by == "rating":
                score = product.rating
            elif sort_by == "newest":
                score = product.popularity  # Simplified
            else:
                score = 0
            scored.append((product, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:limit]]

    def _fuzzy_lookup(self, token):
        """Find documents matching token with fuzzy matching."""
        exact = self.index.get(token, set())
        # Simple prefix matching for typo tolerance
        prefix_matches = set()
        if len(token) >= 3:
            for idx_token, doc_ids in self.index.items():
                if idx_token.startswith(token[:3]) or token.startswith(idx_token[:3]):
                    prefix_matches |= doc_ids
        return exact | prefix_matches

    def get_facets(self, query, facet_field="category"):
        """Get facet counts for search results."""
        tokens = self._tokenize(query)
        matching_ids = None
        for token in tokens:
            candidates = self.index.get(token, set())
            if matching_ids is None:
                matching_ids = candidates
            else:
                matching_ids = matching_ids & candidates

        if not matching_ids:
            return {}

        facets = defaultdict(int)
        for pid in matching_ids:
            product = self.documents.get(pid)
            if product:
                facets[getattr(product, facet_field, "unknown")] += 1

        return dict(sorted(facets.items(), key=lambda x: x[1], reverse=True))


def exercise_1():
    """E-commerce search system."""
    print("E-Commerce Search System (10M products):")
    print("=" * 60)

    index = InvertedIndex()

    # Add synonyms
    index.add_synonym("phone", "smartphone")
    index.add_synonym("mobile", "smartphone")
    index.add_synonym("laptop", "notebook")

    # Create sample products
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    brands = ["Samsung", "Apple", "Sony", "Nike", "Adidas"]

    random.seed(42)
    products = []
    for i in range(1000):
        cat = random.choice(categories)
        brand = random.choice(brands)
        product = Product(
            product_id=f"P{i:06d}",
            name=f"{brand} {cat} Product {i}",
            description=f"High quality {cat.lower()} item from {brand}. "
                        f"smartphone compatible wireless bluetooth",
            brand=brand,
            category=cat,
            price=round(random.uniform(10, 500), 2),
            rating=round(random.uniform(3.0, 5.0), 1),
            popularity=random.randint(0, 10000),
        )
        products.append(product)
        index.index_product(product)

    # Full-text search
    print("\n  --- Full-text search: 'smartphone' ---")
    results = index.search("smartphone", limit=5)
    for p in results:
        print(f"    {p.product_id}: {p.name} (${p.price}, {p.rating}*)")

    # Filtered search
    print("\n  --- Filtered: 'smartphone' in Electronics, $50-200 ---")
    results = index.search("smartphone",
                            filters={"category": "Electronics",
                                     "price_min": 50, "price_max": 200},
                            limit=5)
    for p in results:
        print(f"    {p.product_id}: {p.name} (${p.price}, {p.rating}*)")

    # Sort by price
    print("\n  --- Sorted by price (ascending) ---")
    results = index.search("smartphone", sort_by="price_asc", limit=5)
    for p in results:
        print(f"    {p.product_id}: {p.name} (${p.price})")

    # Category facets
    print("\n  --- Category facets for 'bluetooth' ---")
    facets = index.get_facets("bluetooth")
    for cat, count in facets.items():
        print(f"    {cat}: {count} products")

    # Architecture
    print("\n  Index Design (for 10M products):")
    print("    - 5 primary shards (~2M docs each, ~4GB per shard)")
    print("    - 1 replica per shard (for HA)")
    print("    - Custom analyzer: lowercase + synonym + stemming")
    print("    - Completion suggester for autocomplete")
    print("    - Function score: boost by popularity and recency")


# === Exercise 2: Log Search Platform ===
# Problem: Design centralized log search for 50TB/day.

@dataclass
class LogEntry:
    timestamp: float
    level: str
    service: str
    message: str
    trace_id: str = ""


class TimeBasedLogIndex:
    """Simulates time-based log indexing (like Elasticsearch ILM)."""

    def __init__(self, max_index_size_gb=50):
        self.indices = defaultdict(list)  # index_name -> [LogEntry]
        self.max_index_size = max_index_size_gb
        self.hot_indices = set()
        self.warm_indices = set()
        self.cold_indices = set()

    def ingest(self, entry):
        """Ingest log entry into appropriate index."""
        # Index per hour
        hour = int(entry.timestamp / 3600)
        index_name = f"logs-{hour}"
        self.indices[index_name].append(entry)
        self.hot_indices.add(index_name)

    def search(self, query, time_start=None, time_end=None, level=None,
               service=None, limit=10):
        """Search across indices."""
        results = []
        for index_name, entries in self.indices.items():
            for entry in entries:
                if time_start and entry.timestamp < time_start:
                    continue
                if time_end and entry.timestamp > time_end:
                    continue
                if level and entry.level != level:
                    continue
                if service and entry.service != service:
                    continue
                if query and query.lower() not in entry.message.lower():
                    continue
                results.append(entry)

        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def apply_ilm(self, current_time):
        """Apply Index Lifecycle Management."""
        for index_name in list(self.hot_indices):
            hour = int(index_name.split("-")[1])
            age_hours = (current_time / 3600) - hour

            if age_hours > 24:
                self.hot_indices.discard(index_name)
                self.warm_indices.add(index_name)
            if age_hours > 168:  # 7 days
                self.warm_indices.discard(index_name)
                self.cold_indices.add(index_name)
            if age_hours > 720:  # 30 days
                self.cold_indices.discard(index_name)
                del self.indices[index_name]


def exercise_2():
    """Log search platform design."""
    print("Log Search Platform (50TB/day):")
    print("=" * 60)

    print("\n  Capacity Planning:")
    daily_volume_tb = 50
    write_rate_mb_s = daily_volume_tb * 1024 * 1024 / 86400
    print(f"    Daily volume: {daily_volume_tb} TB")
    print(f"    Write rate: {write_rate_mb_s:.0f} MB/s (~580 MB/s)")
    print(f"    7-day hot storage: {daily_volume_tb * 7} TB")
    print(f"    30-day total: {daily_volume_tb * 30} TB = {daily_volume_tb * 30 / 1024:.1f} PB")

    print("\n  Architecture:")
    print("    Ingestion: Kafka (buffer) -> Logstash/Vector (parse) -> Elasticsearch")
    print("    Hot nodes:  10 x (32GB RAM, 2TB NVMe SSD) = 20TB hot")
    print("    Warm nodes: 6 x (32GB RAM, 8TB HDD) = 48TB warm")
    print("    Cold:       S3 (compressed, searchable snapshots)")

    print("\n  Index Strategy:")
    print("    - Time-based indices: logs-YYYY.MM.DD-HH")
    print("    - Rollover at 50GB or 1 hour")
    print("    - ILM: hot (0-24h) -> warm (1-7d) -> cold/S3 (7-30d) -> delete")
    print("    - Force merge warm indices to 1 segment")

    # Simulate
    log_index = TimeBasedLogIndex()
    services = ["web", "api", "auth", "payment", "catalog"]
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]

    random.seed(42)
    base_time = time.time()
    for i in range(1000):
        entry = LogEntry(
            timestamp=base_time + random.uniform(0, 3600),
            level=random.choices(levels, weights=[60, 20, 10, 10])[0],
            service=random.choice(services),
            message=f"Request processed in {random.randint(1, 500)}ms "
                    f"status={random.choice([200, 200, 200, 200, 500])}",
            trace_id=f"trace-{random.randint(1000, 9999)}",
        )
        log_index.ingest(entry)

    # Search examples
    print("\n  --- Search: errors in payment service ---")
    results = log_index.search("", level="ERROR", service="payment", limit=3)
    for r in results:
        print(f"    [{r.level}] {r.service}: {r.message[:50]}")

    print(f"\n  --- Search: 'status=500' ---")
    results = log_index.search("status=500", limit=3)
    for r in results:
        print(f"    [{r.level}] {r.service}: {r.message[:50]}")

    # Cost comparison
    print("\n  Cost Comparison (monthly):")
    print(f"    {'Solution':<25} {'Estimated Cost':>15}")
    print("    " + "-" * 42)
    print(f"    {'ELK Cloud (managed)':<25} {'$80,000':>15}")
    print(f"    {'Self-hosted ELK':<25} {'$40,000':>15}")
    print(f"    {'Grafana Loki + S3':<25} {'$15,000':>15}")
    print(f"    {'ClickHouse':<25} {'$20,000':>15}")


# === Exercise 3: Autocomplete System ===
# Problem: Design autocomplete returning results in <50ms.

class AutocompleteIndex:
    """Autocomplete using edge n-grams and popularity scoring."""

    def __init__(self):
        self.suggestions = {}  # text -> popularity
        self.ngram_index = defaultdict(set)  # ngram -> set of texts
        self.cache = {}  # prefix -> results (LRU cache simulation)
        self.cache_ttl = 300  # 5 minutes

    def add_suggestion(self, text, popularity=1):
        """Add a suggestion with popularity score."""
        text_lower = text.lower()
        self.suggestions[text_lower] = popularity

        # Generate edge n-grams (prefixes of length 1 to len)
        for n in range(1, len(text_lower) + 1):
            ngram = text_lower[:n]
            self.ngram_index[ngram].add(text_lower)

    def search(self, prefix, limit=10):
        """Search suggestions by prefix."""
        prefix_lower = prefix.lower()

        # Check cache
        cache_key = prefix_lower
        if cache_key in self.cache:
            cached_results, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_results

        # Look up in n-gram index
        candidates = self.ngram_index.get(prefix_lower, set())

        # Score by popularity
        scored = []
        for text in candidates:
            popularity = self.suggestions.get(text, 0)
            # Boost exact prefix matches
            boost = 2.0 if text.startswith(prefix_lower) else 1.0
            scored.append((text, popularity * boost))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = [text for text, _ in scored[:limit]]

        # Cache results
        self.cache[cache_key] = (results, time.time())

        return results


def exercise_3():
    """Autocomplete system design."""
    print("Autocomplete System (<50ms response):")
    print("=" * 60)

    autocomplete = AutocompleteIndex()

    # Populate with product searches
    search_terms = [
        ("iphone 15 pro", 50000),
        ("iphone 15 case", 30000),
        ("iphone charger", 25000),
        ("ipad pro", 20000),
        ("ipad mini", 15000),
        ("samsung galaxy s24", 40000),
        ("samsung galaxy tab", 18000),
        ("sony headphones", 35000),
        ("sony playstation", 28000),
        ("wireless earbuds", 45000),
        ("wireless mouse", 22000),
        ("wireless keyboard", 19000),
        ("laptop stand", 12000),
        ("laptop bag", 10000),
        ("laptop charger", 8000),
        ("macbook pro", 38000),
        ("macbook air", 32000),
    ]

    for term, popularity in search_terms:
        autocomplete.add_suggestion(term, popularity)

    # Simulate autocomplete as user types
    print("\n  Simulating user typing 'iph':")
    for i in range(1, 4):
        prefix = "iph"[:i]
        results = autocomplete.search(prefix, limit=5)
        print(f"    '{prefix}' -> {results}")

    print("\n  Simulating user typing 'wireless':")
    for prefix in ["w", "wi", "wir", "wire", "wirel", "wirele", "wireless"]:
        results = autocomplete.search(prefix, limit=3)
        print(f"    '{prefix}' -> {results[:3]}")

    # Architecture
    print("\n  Architecture:")
    print("    1. Separate lightweight index for suggestions")
    print("    2. Edge n-gram analyzer for partial word matching")
    print("    3. Cache popular queries in Redis (TTL: 5min)")
    print("    4. Client-side debouncing (300ms)")
    print("    5. Boost by search popularity (query analytics)")

    print("\n  Performance optimizations:")
    print("    - Completion suggester (Elasticsearch) for prefix matching")
    print("    - Pre-computed top-10 for common prefixes")
    print("    - CDN caching for top 1000 queries")
    print("    - Client sends request only after 2+ characters")

    print("\n  Query pipeline:")
    print("    User types -> debounce (300ms) -> prefix lookup")
    print("    -> edge-ngram match + completion suggest")
    print("    -> boost by popularity -> return top 10")

    # Cache effectiveness
    print("\n  Cache effectiveness simulation:")
    random.seed(42)
    cache_hits = 0
    total_queries = 0

    for _ in range(1000):
        # Zipf-like: popular prefixes queried more often
        prefix = random.choice(["ip", "sam", "wir", "lap", "mac",
                                 "son", "iph", "gal", "ear"])
        total_queries += 1
        results = autocomplete.search(prefix)
        if prefix in autocomplete.cache:
            cache_hits += 1

    # After first round, most are cached
    for _ in range(1000):
        prefix = random.choice(["ip", "sam", "wir", "lap", "mac",
                                 "son", "iph", "gal", "ear"])
        total_queries += 1
        if prefix in autocomplete.cache:
            cache_hits += 1

    print(f"    Total queries: {total_queries}")
    print(f"    Cache hits: {cache_hits}")
    print(f"    Cache hit rate: {cache_hits/total_queries:.1%}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: E-Commerce Search ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Log Search Platform ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Autocomplete System ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
