"""
Exercise Solutions: Lesson 23 - Production Vector Search

Covers:
  - Exercise 1: Hybrid Search Fusion (RRF and Linear Combination)
  - Exercise 2: Filter Benchmark (Pre-filtering vs Post-filtering)
  - Exercise 3: Reranking Pipeline (Bi-encoder + Cross-encoder simulation)
  - Exercise 4: Capacity Planner (Multi-cloud cost estimation)
  - Exercise 5: Monitoring Dashboard (Prometheus metrics simulation)

Note: Pure Python simulation without requiring actual vector DB or
      model library installations.
"""

import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def random_vector(dim: int) -> list[float]:
    """Generate a random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# =====================================================================
# Exercise 1: Hybrid Search Fusion
# =====================================================================

def bm25_score(query_terms: list[str], doc_terms: list[str],
               doc_lengths: dict[str, int], avg_dl: float,
               k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 scoring."""
    score = 0.0
    dl = len(doc_terms)
    for term in query_terms:
        tf = doc_terms.count(term)
        if tf == 0:
            continue
        idf = 1.0  # simplified
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator
    return score


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked lists using RRF."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def linear_combination_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    alpha: float = 0.7,
) -> list[tuple[str, float]]:
    """Combine results with weighted normalized scores."""
    def normalize(scores):
        vals = [s for _, s in scores]
        if not vals:
            return scores
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [(doc_id, 1.0) for doc_id, _ in scores]
        return [(doc_id, (s - mn) / (mx - mn)) for doc_id, s in scores]

    dense_norm = normalize(dense_results)
    sparse_norm = normalize(sparse_results)

    combined: dict[str, float] = {}
    for doc_id, score in dense_norm:
        combined[doc_id] = alpha * score
    for doc_id, score in sparse_norm:
        combined[doc_id] = combined.get(doc_id, 0.0) + (1 - alpha) * score

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def exercise_1_hybrid_search_fusion():
    """Compare RRF and linear combination fusion."""
    print("=" * 70)
    print("Exercise 1: Hybrid Search Fusion")
    print("=" * 70)

    dim = 16
    n_docs = 200

    # Create documents with both text terms and vectors
    categories = ["code E-4021", "troubleshooting guide", "error patterns",
                   "system diagnostics", "E-4021 patch", "general FAQ",
                   "API reference", "configuration", "monitoring setup",
                   "release notes E-4021"]

    docs = {}
    for i in range(n_docs):
        cat = categories[i % len(categories)]
        docs[f"doc-{i:03d}"] = {
            "text": f"{cat} document {i}",
            "terms": cat.lower().split(),
            "vector": random_vector(dim),
        }

    # Query where hybrid should outperform either alone
    query_text = "E-4021 troubleshooting"
    query_terms = query_text.lower().split()
    query_vector = random_vector(dim)

    # Dense search (vector similarity)
    dense_scores = []
    for doc_id, doc in docs.items():
        score = cosine_similarity(query_vector, doc["vector"])
        dense_scores.append((doc_id, score))
    dense_scores.sort(key=lambda x: x[1], reverse=True)
    dense_ranked = [doc_id for doc_id, _ in dense_scores[:50]]

    # Sparse search (BM25-like term matching)
    avg_dl = sum(len(d["terms"]) for d in docs.values()) / len(docs)
    sparse_scores = []
    for doc_id, doc in docs.items():
        score = bm25_score(query_terms, doc["terms"], {}, avg_dl)
        sparse_scores.append((doc_id, score))
    sparse_scores.sort(key=lambda x: x[1], reverse=True)
    sparse_ranked = [doc_id for doc_id, _ in sparse_scores[:50]]

    # Fuse with RRF
    rrf_results = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=60)

    # Fuse with linear combination
    lc_results = linear_combination_fusion(
        dense_scores[:50], sparse_scores[:50], alpha=0.5,
    )

    # Ground truth: documents mentioning E-4021 (exact match matters)
    relevant = set(
        doc_id for doc_id, doc in docs.items()
        if "e-4021" in " ".join(doc["terms"])
    )

    # Compute precision at k=10
    def precision_at_k(results, relevant_set, k=10):
        top_k = set(r[0] if isinstance(r, tuple) else r for r in results[:k])
        return len(top_k & relevant_set) / k

    p_dense = precision_at_k(dense_ranked, relevant)
    p_sparse = precision_at_k(sparse_ranked, relevant)
    p_rrf = precision_at_k(rrf_results, relevant)
    p_lc = precision_at_k(lc_results, relevant)

    print(f"\nQuery: '{query_text}'")
    print(f"Relevant documents (mentioning E-4021): {len(relevant)}")
    print(f"\n{'Method':<25} {'P@10':<10} {'Top-3 results'}")
    print("-" * 70)
    print(f"{'Dense only':<25} {p_dense:<10.3f} "
          f"{', '.join(dense_ranked[:3])}")
    print(f"{'Sparse only':<25} {p_sparse:<10.3f} "
          f"{', '.join(sparse_ranked[:3])}")
    print(f"{'RRF (k=60)':<25} {p_rrf:<10.3f} "
          f"{', '.join(r[0] for r in rrf_results[:3])}")
    print(f"{'Linear (alpha=0.5)':<25} {p_lc:<10.3f} "
          f"{', '.join(r[0] for r in lc_results[:3])}")

    # Test different alpha values
    print(f"\nAlpha sensitivity (linear combination):")
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        lc = linear_combination_fusion(dense_scores[:50], sparse_scores[:50], alpha)
        p = precision_at_k(lc, relevant)
        bar = "#" * int(p * 50)
        print(f"  alpha={alpha:.1f}: P@10={p:.3f} {bar}")


# =====================================================================
# Exercise 2: Filter Benchmark
# =====================================================================

def exercise_2_filter_benchmark():
    """Benchmark pre-filtering vs post-filtering at different selectivities."""
    print("\n" + "=" * 70)
    print("Exercise 2: Filter Benchmark")
    print("=" * 70)

    dim = 16
    n_vectors = 5000
    n_queries = 20
    k = 10

    # Create dataset with metadata
    categories = ["electronics", "clothing", "home", "sports", "books",
                   "food", "toys", "automotive", "health", "garden"]
    data = []
    for i in range(n_vectors):
        data.append({
            "id": i,
            "vector": random_vector(dim),
            "category": categories[i % len(categories)],
            "price": random.uniform(5.0, 500.0),
            "rating": random.uniform(1.0, 5.0),
        })

    queries = [random_vector(dim) for _ in range(n_queries)]

    # Ground truth: brute-force search on full dataset
    def brute_force_search(query, dataset, k):
        scored = [(d["id"], cosine_similarity(query, d["vector"])) for d in dataset]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # Pre-filtering: filter first, then search
    def pre_filter_search(query, dataset, filter_fn, k):
        filtered = [d for d in dataset if filter_fn(d)]
        return brute_force_search(query, filtered, k)

    # Post-filtering: search first, then filter
    def post_filter_search(query, dataset, filter_fn, k, overfetch=5):
        all_results = brute_force_search(query, dataset, k * overfetch)
        filtered = [(doc_id, score) for doc_id, score in all_results
                     if filter_fn(next(d for d in dataset if d["id"] == doc_id))]
        return filtered[:k]

    # Test at different selectivity levels
    selectivities = [
        ("90% (1 category)", lambda d: d["category"] == "electronics"),  # ~10% data
        ("50% (5 categories)", lambda d: d["category"] in categories[:5]),
        ("10% (high rating)", lambda d: d["rating"] >= 4.5),
        ("1% (expensive+rated)", lambda d: d["price"] > 450 and d["rating"] > 4.5),
    ]

    print(f"\nDataset: {n_vectors} vectors, {dim}d, {n_queries} queries, top-{k}")
    print(f"\n{'Selectivity':<25} {'Method':<15} {'Avg ms':<10} "
          f"{'Avg results':<14} {'Recall vs full':<14}")
    print("-" * 80)

    for sel_name, filter_fn in selectivities:
        # Count filtered data
        filtered_count = sum(1 for d in data if filter_fn(d))
        actual_selectivity = filtered_count / n_vectors

        # Pre-filter benchmark
        pre_latencies = []
        pre_result_counts = []
        pre_recalls = []

        post_latencies = []
        post_result_counts = []
        post_recalls = []

        for query in queries:
            # Ground truth (filter then exact search)
            gt = pre_filter_search(query, data, filter_fn, k)
            gt_ids = set(r[0] for r in gt)

            # Pre-filter
            start = time.perf_counter()
            pre_res = pre_filter_search(query, data, filter_fn, k)
            pre_latencies.append((time.perf_counter() - start) * 1000)
            pre_result_counts.append(len(pre_res))
            pre_ids = set(r[0] for r in pre_res)
            pre_recalls.append(len(pre_ids & gt_ids) / len(gt_ids) if gt_ids else 1.0)

            # Post-filter
            start = time.perf_counter()
            post_res = post_filter_search(query, data, filter_fn, k, overfetch=5)
            post_latencies.append((time.perf_counter() - start) * 1000)
            post_result_counts.append(len(post_res))
            post_ids = set(r[0] for r in post_res)
            post_recalls.append(len(post_ids & gt_ids) / len(gt_ids) if gt_ids else 1.0)

        # Print results
        print(f"{sel_name:<25} {'Pre-filter':<15} "
              f"{statistics.mean(pre_latencies):<10.2f} "
              f"{statistics.mean(pre_result_counts):<14.1f} "
              f"{statistics.mean(pre_recalls):<14.3f}")
        print(f"{'  ('+str(filtered_count)+' docs)':<25} {'Post-filter':<15} "
              f"{statistics.mean(post_latencies):<10.2f} "
              f"{statistics.mean(post_result_counts):<14.1f} "
              f"{statistics.mean(post_recalls):<14.3f}")


# =====================================================================
# Exercise 3: Reranking Pipeline
# =====================================================================

def exercise_3_reranking_pipeline():
    """Build a two-stage retrieval pipeline with simulated reranking."""
    print("\n" + "=" * 70)
    print("Exercise 3: Reranking Pipeline")
    print("=" * 70)

    dim = 16
    n_docs = 500
    n_queries = 20
    k_retrieve = 50
    k_final = 10

    # Create documents with relevance scores
    docs = {}
    for i in range(n_docs):
        docs[f"doc-{i:03d}"] = {
            "vector": random_vector(dim),
            "quality": random.random(),  # simulated document quality
        }

    queries = []
    ground_truths = []
    for _ in range(n_queries):
        q_vec = random_vector(dim)
        # Ground truth: top-k by combined vector similarity + quality
        scored = []
        for doc_id, doc in docs.items():
            sim = cosine_similarity(q_vec, doc["vector"])
            # "True" relevance considers both similarity and quality
            true_score = 0.6 * sim + 0.4 * doc["quality"]
            scored.append((doc_id, true_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        queries.append(q_vec)
        ground_truths.append([doc_id for doc_id, _ in scored[:k_final]])

    def mrr(predicted_lists, truth_lists):
        """Mean Reciprocal Rank."""
        rrs = []
        for pred, truth in zip(predicted_lists, truth_lists):
            best_truth = truth[0]  # most relevant document
            for rank, doc_id in enumerate(pred, 1):
                if doc_id == best_truth:
                    rrs.append(1.0 / rank)
                    break
            else:
                rrs.append(0.0)
        return sum(rrs) / len(rrs)

    def recall_at_k(predicted_lists, truth_lists, k):
        recalls = []
        for pred, truth in zip(predicted_lists, truth_lists):
            pred_set = set(pred[:k])
            truth_set = set(truth[:k])
            recalls.append(len(pred_set & truth_set) / len(truth_set) if truth_set else 0)
        return sum(recalls) / len(recalls)

    # Stage 1: Bi-encoder retrieval (vector similarity only)
    stage1_results = []
    for q_vec in queries:
        scored = []
        for doc_id, doc in docs.items():
            sim = cosine_similarity(q_vec, doc["vector"])
            scored.append((doc_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        stage1_results.append([doc_id for doc_id, _ in scored[:k_retrieve]])

    # Stage 2: Cross-encoder reranking (simulated — uses quality signal)
    stage2_results = []
    for i, (q_vec, candidates) in enumerate(zip(queries, stage1_results)):
        reranked = []
        for doc_id in candidates:
            doc = docs[doc_id]
            sim = cosine_similarity(q_vec, doc["vector"])
            # Cross-encoder captures quality (simulated interaction)
            cross_score = 0.5 * sim + 0.5 * doc["quality"]
            reranked.append((doc_id, cross_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        stage2_results.append([doc_id for doc_id, _ in reranked[:k_final]])

    # Evaluate
    stage1_final = [r[:k_final] for r in stage1_results]

    mrr_stage1 = mrr(stage1_final, ground_truths)
    mrr_stage2 = mrr(stage2_results, ground_truths)
    recall_stage1 = recall_at_k(stage1_final, ground_truths, k_final)
    recall_stage2 = recall_at_k(stage2_results, ground_truths, k_final)

    print(f"\nPipeline: {n_docs} docs, {dim}d, retrieve {k_retrieve} → rerank to {k_final}")
    print(f"\n{'Stage':<30} {'MRR@{k_final}':<15} {'Recall@{k_final}':<15}")
    print("-" * 60)
    print(f"{'Bi-encoder only':<30} {mrr_stage1:<15.4f} {recall_stage1:<15.4f}")
    print(f"{'Bi-encoder + Cross-encoder':<30} {mrr_stage2:<15.4f} {recall_stage2:<15.4f}")
    print(f"{'Improvement':<30} {(mrr_stage2-mrr_stage1)/mrr_stage1*100:+.1f}%"
          f"{'':>8} {(recall_stage2-recall_stage1)/recall_stage1*100:+.1f}%")

    # Impact of retrieve depth on final quality
    print(f"\nImpact of retrieval depth (k_retrieve) on reranked quality:")
    for k_ret in [10, 20, 50, 100, 200]:
        reranked_lists = []
        for i, q_vec in enumerate(queries):
            scored = []
            for doc_id, doc in docs.items():
                sim = cosine_similarity(q_vec, doc["vector"])
                scored.append((doc_id, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [doc_id for doc_id, _ in scored[:k_ret]]

            reranked = []
            for doc_id in candidates:
                doc = docs[doc_id]
                sim = cosine_similarity(q_vec, doc["vector"])
                cross_score = 0.5 * sim + 0.5 * doc["quality"]
                reranked.append((doc_id, cross_score))
            reranked.sort(key=lambda x: x[1], reverse=True)
            reranked_lists.append([doc_id for doc_id, _ in reranked[:k_final]])

        m = mrr(reranked_lists, ground_truths)
        r = recall_at_k(reranked_lists, ground_truths, k_final)
        print(f"  k_retrieve={k_ret:<5d} MRR={m:.4f}  Recall@10={r:.4f}")


# =====================================================================
# Exercise 4: Capacity Planner (Multi-Cloud)
# =====================================================================

@dataclass
class CloudInstance:
    name: str
    cloud: str
    vcpus: int
    ram_gb: float
    hourly_cost: float
    storage_cost_gb_month: float


# Common instance types across clouds
INSTANCES = [
    CloudInstance("r6i.xlarge", "AWS", 4, 32, 0.252, 0.10),
    CloudInstance("r6i.2xlarge", "AWS", 8, 64, 0.504, 0.10),
    CloudInstance("r6i.4xlarge", "AWS", 16, 128, 1.008, 0.10),
    CloudInstance("n2-highmem-4", "GCP", 4, 32, 0.262, 0.08),
    CloudInstance("n2-highmem-8", "GCP", 8, 64, 0.524, 0.08),
    CloudInstance("n2-highmem-16", "GCP", 16, 128, 1.048, 0.08),
    CloudInstance("E4as_v5 (4vcpu)", "Azure", 4, 32, 0.252, 0.09),
    CloudInstance("E8as_v5 (8vcpu)", "Azure", 8, 64, 0.504, 0.09),
    CloudInstance("E16as_v5 (16vcpu)", "Azure", 16, 128, 1.008, 0.09),
]


def estimate_capacity(
    n_vectors: int,
    dim: int,
    quantization: str = "none",
    index_type: str = "HNSW",
    qps_target: int = 1000,
    replication_factor: int = 2,
) -> dict:
    """Estimate resources needed for vector search deployment."""
    bpe = {"none": 4, "float16": 2, "int8": 1, "pq96": 96 / dim}.get(quantization, 4)
    vector_gb = (n_vectors * dim * bpe) / (1024 ** 3)

    hnsw_m = 32
    index_gb = {
        "HNSW": (n_vectors * hnsw_m * 2 * 8) / (1024 ** 3),
        "IVF": 0.1 * vector_gb,
        "Flat": 0,
    }.get(index_type, 0)

    metadata_gb = (n_vectors * 200) / (1024 ** 3)
    total_gb = vector_gb + index_gb + metadata_gb

    # QPS per core estimate
    qps_per_core = {"HNSW": 3000, "IVF": 5000, "Flat": 50}.get(index_type, 1000)
    cores_needed = max(4, int(qps_target / qps_per_core) + 1)

    return {
        "vector_gb": vector_gb,
        "index_gb": index_gb,
        "metadata_gb": metadata_gb,
        "total_gb": total_gb,
        "cores_needed": cores_needed,
        "replication_factor": replication_factor,
    }


def find_best_instance(total_gb: float, cores_needed: int,
                       replication_factor: int,
                       cloud: str | None = None) -> list[dict]:
    """Find best instance type for given requirements."""
    candidates = INSTANCES if cloud is None else [i for i in INSTANCES if i.cloud == cloud]
    results = []

    for inst in candidates:
        if inst.ram_gb * 0.7 < total_gb:  # need 30% headroom
            continue
        if inst.vcpus < cores_needed:
            continue

        n_nodes = replication_factor  # 1 shard fits in one node
        if total_gb > inst.ram_gb * 0.7:
            n_shards = math.ceil(total_gb / (inst.ram_gb * 0.7))
            n_nodes = n_shards * replication_factor

        monthly_compute = inst.hourly_cost * 730 * n_nodes  # 730 hrs/month
        monthly_storage = inst.storage_cost_gb_month * total_gb * n_nodes

        results.append({
            "instance": inst.name,
            "cloud": inst.cloud,
            "n_nodes": n_nodes,
            "monthly_compute": monthly_compute,
            "monthly_storage": monthly_storage,
            "monthly_total": monthly_compute + monthly_storage,
        })

    results.sort(key=lambda x: x["monthly_total"])
    return results


def exercise_4_capacity_planner():
    """Multi-cloud capacity planning for vector search."""
    print("\n" + "=" * 70)
    print("Exercise 4: Capacity Planner (Multi-Cloud)")
    print("=" * 70)

    scenarios = [
        {"name": "Small (1M, 768d, int8)", "n": 1_000_000, "d": 768,
         "q": "int8", "qps": 500},
        {"name": "Medium (10M, 768d, int8)", "n": 10_000_000, "d": 768,
         "q": "int8", "qps": 2000},
        {"name": "Large (100M, 768d, pq96)", "n": 100_000_000, "d": 768,
         "q": "pq96", "qps": 5000},
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        cap = estimate_capacity(
            n_vectors=scenario["n"],
            dim=scenario["d"],
            quantization=scenario["q"],
            qps_target=scenario["qps"],
            replication_factor=2,
        )

        print(f"  Vector data: {cap['vector_gb']:.1f} GB")
        print(f"  Index overhead: {cap['index_gb']:.1f} GB")
        print(f"  Metadata: {cap['metadata_gb']:.1f} GB")
        print(f"  Total per shard: {cap['total_gb']:.1f} GB")
        print(f"  Cores needed: {cap['cores_needed']}")
        print(f"\n  {'Cloud':<8} {'Instance':<22} {'Nodes':<8} "
              f"{'Compute/mo':<14} {'Storage/mo':<14} {'Total/mo'}")
        print(f"  {'-'*80}")

        for cloud in ["AWS", "GCP", "Azure"]:
            options = find_best_instance(
                cap["total_gb"], cap["cores_needed"],
                cap["replication_factor"], cloud=cloud,
            )
            if options:
                best = options[0]
                print(f"  {best['cloud']:<8} {best['instance']:<22} "
                      f"{best['n_nodes']:<8} "
                      f"${best['monthly_compute']:<13,.0f} "
                      f"${best['monthly_storage']:<13,.0f} "
                      f"${best['monthly_total']:,.0f}")
            else:
                print(f"  {cloud:<8} No suitable instance found "
                      f"(data too large for single node)")


# =====================================================================
# Exercise 5: Monitoring Dashboard (Prometheus Simulation)
# =====================================================================

@dataclass
class MetricSample:
    timestamp: datetime
    value: float
    labels: dict = field(default_factory=dict)


class SimulatedHistogram:
    """Simulates a Prometheus Histogram for latency tracking."""

    def __init__(self, name: str, buckets: list[float]):
        self.name = name
        self.buckets = sorted(buckets)
        self.samples: list[float] = []

    def observe(self, value: float):
        self.samples.append(value)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


class SimulatedCounter:
    """Simulates a Prometheus Counter."""

    def __init__(self, name: str):
        self.name = name
        self.values: dict[str, float] = defaultdict(float)

    def inc(self, labels: str = "default", amount: float = 1.0):
        self.values[labels] += amount

    def total(self, labels: str = "default") -> float:
        return self.values[labels]


class SimulatedGauge:
    """Simulates a Prometheus Gauge."""

    def __init__(self, name: str):
        self.name = name
        self.value = 0.0

    def set(self, value: float):
        self.value = value

    def get(self) -> float:
        return self.value


class VectorSearchMonitor:
    """Complete monitoring setup for vector search service."""

    def __init__(self):
        self.search_latency = SimulatedHistogram(
            "vector_search_latency_ms",
            buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500],
        )
        self.rerank_latency = SimulatedHistogram(
            "rerank_latency_ms",
            buckets=[10, 25, 50, 100, 200, 500],
        )
        self.search_count = SimulatedCounter("vector_search_total")
        self.error_count = SimulatedCounter("vector_search_errors_total")
        self.result_count = SimulatedHistogram(
            "vector_search_result_count",
            buckets=[0, 1, 5, 10, 20, 50],
        )
        self.index_size = SimulatedGauge("vector_index_size")
        self.memory_usage_pct = SimulatedGauge("memory_usage_pct")
        self.recall_gauge = SimulatedGauge("recall_at_10")
        self.alerts: list[dict] = []

    def record_search(self, latency_ms: float, n_results: int,
                      success: bool = True):
        self.search_latency.observe(latency_ms)
        self.result_count.observe(n_results)
        if success:
            self.search_count.inc("success")
        else:
            self.search_count.inc("error")
            self.error_count.inc()

    def check_alerts(self):
        """Check metric thresholds and generate alerts."""
        p99 = self.search_latency.percentile(99)
        if p99 > 100:
            self.alerts.append({
                "severity": "warning",
                "metric": "search_latency_p99",
                "value": p99,
                "threshold": 100,
                "message": f"p99 latency {p99:.1f}ms > 100ms threshold",
            })

        error_rate = (self.error_count.total() /
                      max(1, self.search_count.total("success") +
                          self.error_count.total()))
        if error_rate > 0.01:
            self.alerts.append({
                "severity": "critical",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": 0.01,
                "message": f"Error rate {error_rate:.3f} > 1% threshold",
            })

        recall = self.recall_gauge.get()
        if recall > 0 and recall < 0.90:
            self.alerts.append({
                "severity": "warning",
                "metric": "recall_at_10",
                "value": recall,
                "threshold": 0.90,
                "message": f"Recall {recall:.3f} < 0.90 threshold",
            })

    def dashboard_summary(self) -> str:
        lines = []
        lines.append("┌─────────────── Vector Search Dashboard ───────────────┐")
        lines.append("│")
        lines.append(f"│  Search QPS:      {self.search_count.total('success'):.0f} total")
        lines.append(f"│  p50 Latency:     {self.search_latency.percentile(50):.1f} ms")
        lines.append(f"│  p95 Latency:     {self.search_latency.percentile(95):.1f} ms")
        lines.append(f"│  p99 Latency:     {self.search_latency.percentile(99):.1f} ms")
        lines.append(f"│  Error Rate:      "
                     f"{self.error_count.total() / max(1, self.search_count.total('success')):.3f}")
        lines.append(f"│  Avg Results:     {self.result_count.mean():.1f}")
        lines.append(f"│  Index Size:      {self.index_size.get():,.0f} vectors")
        lines.append(f"│  Memory Usage:    {self.memory_usage_pct.get():.1f}%")
        lines.append(f"│  Recall@10:       {self.recall_gauge.get():.3f}")
        lines.append("│")

        if self.alerts:
            lines.append("│  ALERTS:")
            for alert in self.alerts:
                icon = "!!" if alert["severity"] == "critical" else "!"
                lines.append(f"│    [{icon}] {alert['message']}")
        else:
            lines.append("│  ALERTS: None")

        lines.append("│")
        lines.append("└────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


def exercise_5_monitoring_dashboard():
    """Simulate vector search monitoring with alerts."""
    print("\n" + "=" * 70)
    print("Exercise 5: Monitoring Dashboard")
    print("=" * 70)

    monitor = VectorSearchMonitor()
    monitor.index_size.set(5_000_000)
    monitor.memory_usage_pct.set(72.3)
    monitor.recall_gauge.set(0.96)

    # Simulate normal traffic
    print("\n--- Simulating normal traffic (1000 searches) ---")
    for _ in range(1000):
        latency = random.gauss(8, 3)  # ~8ms avg, 3ms std
        latency = max(0.5, latency)
        n_results = random.randint(5, 10)
        success = random.random() > 0.002  # 0.2% error rate
        monitor.record_search(latency, n_results, success)

    monitor.check_alerts()
    print(monitor.dashboard_summary())

    # Simulate degraded state
    print("\n--- Simulating degraded state (slow queries + low recall) ---")
    monitor2 = VectorSearchMonitor()
    monitor2.index_size.set(5_000_000)
    monitor2.memory_usage_pct.set(92.1)
    monitor2.recall_gauge.set(0.82)  # Below threshold

    for _ in range(1000):
        latency = random.gauss(45, 30)  # Much higher latency
        latency = max(1.0, latency)
        n_results = random.randint(0, 8)
        success = random.random() > 0.03  # 3% error rate
        monitor2.record_search(latency, n_results, success)

    monitor2.check_alerts()
    print(monitor2.dashboard_summary())


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    random.seed(42)
    exercise_1_hybrid_search_fusion()
    exercise_2_filter_benchmark()
    exercise_3_reranking_pipeline()
    exercise_4_capacity_planner()
    exercise_5_monitoring_dashboard()
    print("\nAll exercises completed.")
