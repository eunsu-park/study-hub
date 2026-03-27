"""
Exercise Solutions: Lesson 22 - Vector Storage and Indexing

Covers:
  - Exercise 1: FAISS Index Comparison (Flat, IVF, HNSW, IVFPQ)
  - Exercise 2: Milvus Collection Schema Design
  - Exercise 3: Qdrant vs Chroma API Comparison
  - Exercise 4: Index Versioning Pipeline
  - Exercise 5: Benchmark Runner

Note: Pure Python simulation of vector operations, index types, and
      database interactions without requiring actual library installations.
"""

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Simulated Vector Operations
# ---------------------------------------------------------------------------

def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def random_vector(dim: int) -> list[float]:
    """Generate a random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# =====================================================================
# Exercise 1: FAISS Index Comparison
# =====================================================================

class FlatIndex:
    """Brute-force exact search index."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: list[list[float]] = []
        self.ids: list[int] = []

    def add(self, vectors: list[list[float]]) -> None:
        for v in vectors:
            self.ids.append(len(self.ids))
            self.vectors.append(v)

    def search(self, query: list[float], k: int = 10) -> list[tuple[int, float]]:
        distances = []
        for idx, v in zip(self.ids, self.vectors):
            d = euclidean_distance(query, v)
            distances.append((idx, d))
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def memory_bytes(self) -> int:
        return self.ntotal * self.dim * 4  # float32


class IVFIndex:
    """Simulated IVF (Inverted File) index with coarse quantizer."""

    def __init__(self, dim: int, nlist: int):
        self.dim = dim
        self.nlist = nlist
        self.centroids: list[list[float]] = []
        self.inverted_lists: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
        self.ntotal = 0
        self.nprobe = 1

    def train(self, vectors: list[list[float]]) -> None:
        """Simulate k-means training to find centroids."""
        # Simplified: pick random vectors as centroids
        indices = random.sample(range(len(vectors)), min(self.nlist, len(vectors)))
        self.centroids = [vectors[i] for i in indices]

    def add(self, vectors: list[list[float]]) -> None:
        for v in vectors:
            # Assign to nearest centroid
            best_cell = min(
                range(len(self.centroids)),
                key=lambda c: euclidean_distance(v, self.centroids[c]),
            )
            self.inverted_lists[best_cell].append((self.ntotal, v))
            self.ntotal += 1

    def search(self, query: list[float], k: int = 10) -> list[tuple[int, float]]:
        # Find nearest centroids
        centroid_dists = [
            (i, euclidean_distance(query, c))
            for i, c in enumerate(self.centroids)
        ]
        centroid_dists.sort(key=lambda x: x[1])
        probe_cells = [c[0] for c in centroid_dists[: self.nprobe]]

        # Search within selected cells
        candidates = []
        for cell in probe_cells:
            for idx, v in self.inverted_lists[cell]:
                d = euclidean_distance(query, v)
                candidates.append((idx, d))

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def memory_bytes(self) -> int:
        return self.ntotal * self.dim * 4 + self.nlist * self.dim * 4


class HNSWIndex:
    """Simplified HNSW index simulation."""

    def __init__(self, dim: int, M: int = 16):
        self.dim = dim
        self.M = M
        self.vectors: list[list[float]] = []
        self.ids: list[int] = []
        self.graph: dict[int, list[int]] = defaultdict(list)
        self.ef_search = 32

    def add(self, vectors: list[list[float]]) -> None:
        for v in vectors:
            idx = len(self.ids)
            self.ids.append(idx)
            self.vectors.append(v)
            # Connect to M random existing nodes (simplified)
            if idx > 0:
                neighbors = random.sample(
                    range(idx), min(self.M, idx)
                )
                self.graph[idx] = neighbors
                for n in neighbors:
                    if len(self.graph[n]) < self.M * 2:
                        self.graph[n].append(idx)

    def search(self, query: list[float], k: int = 10) -> list[tuple[int, float]]:
        """Simplified greedy HNSW search."""
        if not self.ids:
            return []

        # Start from random entry point
        visited = set()
        entry = random.choice(self.ids)
        candidates = [(euclidean_distance(query, self.vectors[entry]), entry)]
        visited.add(entry)

        # Greedy traversal
        steps = 0
        while steps < self.ef_search:
            if not candidates:
                break
            candidates.sort()
            current_dist, current = candidates[0]

            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = euclidean_distance(query, self.vectors[neighbor])
                    candidates.append((d, neighbor))
                    steps += 1

            candidates.sort()
            candidates = candidates[: self.ef_search]

        candidates.sort()
        return [(idx, dist) for dist, idx in candidates[:k]]

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def memory_bytes(self) -> int:
        vector_mem = self.ntotal * self.dim * 4
        graph_mem = self.ntotal * self.M * 2 * 8  # bidirectional links
        return vector_mem + graph_mem


def exercise_1_faiss_index_comparison():
    """Compare Flat, IVF, and HNSW indexes on random vectors."""
    print("=" * 70)
    print("Exercise 1: FAISS Index Comparison")
    print("=" * 70)

    dim = 32  # Small dim for simulation speed
    n_vectors = 2000
    n_queries = 10
    k = 10

    # Generate data
    data = [random_vector(dim) for _ in range(n_vectors)]
    queries = [random_vector(dim) for _ in range(n_queries)]

    # --- Flat Index (ground truth) ---
    flat = FlatIndex(dim)
    start = time.perf_counter()
    flat.add(data)
    build_time_flat = time.perf_counter() - start

    start = time.perf_counter()
    flat_results = [flat.search(q, k) for q in queries]
    search_time_flat = (time.perf_counter() - start) / n_queries

    # Ground truth IDs
    ground_truth = [set(r[0] for r in res) for res in flat_results]

    # --- IVF Index ---
    nlist = 16
    ivf = IVFIndex(dim, nlist)
    start = time.perf_counter()
    ivf.train(data)
    ivf.add(data)
    build_time_ivf = time.perf_counter() - start

    ivf.nprobe = 4
    start = time.perf_counter()
    ivf_results = [ivf.search(q, k) for q in queries]
    search_time_ivf = (time.perf_counter() - start) / n_queries

    # --- HNSW Index ---
    hnsw = HNSWIndex(dim, M=8)
    start = time.perf_counter()
    hnsw.add(data)
    build_time_hnsw = time.perf_counter() - start

    start = time.perf_counter()
    hnsw_results = [hnsw.search(q, k) for q in queries]
    search_time_hnsw = (time.perf_counter() - start) / n_queries

    # Compute recall
    def recall_at_k(results, truth):
        recalls = []
        for res, gt in zip(results, truth):
            pred = set(r[0] for r in res)
            recalls.append(len(pred & gt) / len(gt) if gt else 0)
        return sum(recalls) / len(recalls)

    recall_ivf = recall_at_k(ivf_results, ground_truth)
    recall_hnsw = recall_at_k(hnsw_results, ground_truth)

    print(f"\nDataset: {n_vectors} vectors, {dim} dimensions, top-{k}")
    print(f"\n{'Index':<12} {'Build(s)':<12} {'Search(ms)':<14} "
          f"{'Recall@{k}':<12} {'Memory(KB)':<12}")
    print("-" * 62)
    print(f"{'Flat':<12} {build_time_flat:<12.4f} {search_time_flat*1000:<14.2f} "
          f"{'1.000':<12} {flat.memory_bytes()/1024:<12.1f}")
    print(f"{'IVF':<12} {build_time_ivf:<12.4f} {search_time_ivf*1000:<14.2f} "
          f"{recall_ivf:<12.3f} {ivf.memory_bytes()/1024:<12.1f}")
    print(f"{'HNSW':<12} {build_time_hnsw:<12.4f} {search_time_hnsw*1000:<14.2f} "
          f"{recall_hnsw:<12.3f} {hnsw.memory_bytes()/1024:<12.1f}")


# =====================================================================
# Exercise 2: Milvus Collection Schema Design
# =====================================================================

@dataclass
class FieldSchema:
    name: str
    dtype: str
    is_primary: bool = False
    auto_id: bool = False
    max_length: int | None = None
    dim: int | None = None

@dataclass
class CollectionSchema:
    name: str
    fields: list[FieldSchema]
    description: str = ""

    def validate(self) -> list[str]:
        """Validate schema design and return warnings."""
        warnings = []
        has_primary = any(f.is_primary for f in self.fields)
        has_vector = any(f.dtype == "FLOAT_VECTOR" for f in self.fields)

        if not has_primary:
            warnings.append("ERROR: No primary key field defined")
        if not has_vector:
            warnings.append("ERROR: No vector field defined")

        primary_count = sum(1 for f in self.fields if f.is_primary)
        if primary_count > 1:
            warnings.append("ERROR: Multiple primary keys not allowed")

        vector_fields = [f for f in self.fields if f.dtype == "FLOAT_VECTOR"]
        for vf in vector_fields:
            if vf.dim and vf.dim > 32768:
                warnings.append(f"WARNING: Dimension {vf.dim} is very high")

        varchar_fields = [f for f in self.fields if f.dtype == "VARCHAR"]
        for vf in varchar_fields:
            if not vf.max_length:
                warnings.append(f"WARNING: VARCHAR field '{vf.name}' missing max_length")

        return warnings

    def display(self) -> str:
        lines = [f"Collection: {self.name}", f"Description: {self.description}", ""]
        lines.append(f"{'Field':<20} {'Type':<15} {'Primary':<10} {'Extra'}")
        lines.append("-" * 65)
        for f in self.fields:
            extra = ""
            if f.auto_id:
                extra += "auto_id "
            if f.max_length:
                extra += f"max_length={f.max_length} "
            if f.dim:
                extra += f"dim={f.dim} "
            lines.append(f"{f.name:<20} {f.dtype:<15} {str(f.is_primary):<10} {extra}")
        return "\n".join(lines)


def exercise_2_milvus_schema_design():
    """Design a Milvus schema for e-commerce product search."""
    print("\n" + "=" * 70)
    print("Exercise 2: Milvus Collection Schema Design (E-commerce)")
    print("=" * 70)

    # Design schema for product search with category and price filtering
    schema = CollectionSchema(
        name="products",
        description="E-commerce product search with embeddings",
        fields=[
            FieldSchema(
                name="product_id",
                dtype="INT64",
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="title",
                dtype="VARCHAR",
                max_length=512,
            ),
            FieldSchema(
                name="description",
                dtype="VARCHAR",
                max_length=2048,
            ),
            FieldSchema(
                name="category",
                dtype="VARCHAR",
                max_length=128,
            ),
            FieldSchema(
                name="subcategory",
                dtype="VARCHAR",
                max_length=128,
            ),
            FieldSchema(
                name="brand",
                dtype="VARCHAR",
                max_length=256,
            ),
            FieldSchema(
                name="price",
                dtype="FLOAT",
            ),
            FieldSchema(
                name="in_stock",
                dtype="BOOL",
            ),
            FieldSchema(
                name="rating",
                dtype="FLOAT",
            ),
            FieldSchema(
                name="embedding",
                dtype="FLOAT_VECTOR",
                dim=768,
            ),
        ],
    )

    # Validate and display
    warnings = schema.validate()
    print(f"\n{schema.display()}")

    if warnings:
        print(f"\nValidation warnings:")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\nValidation: PASSED (no warnings)")

    # Index recommendation
    print(f"\nRecommended indexes:")
    print(f"  Vector field 'embedding': IVF_SQ8 (nlist=2048)")
    print(f"  Scalar field 'category': Inverted index")
    print(f"  Scalar field 'price': STL_SORT index")
    print(f"  Scalar field 'in_stock': Inverted index")

    # Partition recommendation
    print(f"\nRecommended partitions:")
    print(f"  By top-level category: electronics, clothing, home, sports, ...")
    print(f"  Benefit: partition pruning reduces search space by ~5-10x")

    # Example search
    print(f"\nExample search expression:")
    print(f'  expr=\'category == "electronics" and price >= 50.0 '
          f'and price <= 500.0 and in_stock == true\'')


# =====================================================================
# Exercise 3: Qdrant vs Chroma API Comparison
# =====================================================================

class SimulatedQdrant:
    """Simulated Qdrant client for API comparison."""

    def __init__(self):
        self.collections: dict[str, dict] = {}

    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine"):
        self.collections[name] = {
            "config": {"size": vector_size, "distance": distance},
            "points": {},
        }
        return {"status": "ok"}

    def upsert(self, collection: str, points: list[dict]):
        coll = self.collections[collection]
        for p in points:
            coll["points"][p["id"]] = {
                "vector": p["vector"],
                "payload": p.get("payload", {}),
            }
        return {"status": "ok", "count": len(points)}

    def search(self, collection: str, query_vector: list[float],
               limit: int = 10, query_filter: dict | None = None):
        coll = self.collections[collection]
        results = []
        for pid, point in coll["points"].items():
            # Apply filter
            if query_filter:
                if not self._match_filter(point["payload"], query_filter):
                    continue
            score = cosine_similarity(query_vector, point["vector"])
            results.append({"id": pid, "score": score, "payload": point["payload"]})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _match_filter(self, payload: dict, query_filter: dict) -> bool:
        for key, condition in query_filter.items():
            if key not in payload:
                return False
            if isinstance(condition, dict):
                if "$eq" in condition and payload[key] != condition["$eq"]:
                    return False
                if "$gte" in condition and payload[key] < condition["$gte"]:
                    return False
            elif payload[key] != condition:
                return False
        return True


class SimulatedChroma:
    """Simulated Chroma client for API comparison."""

    def __init__(self):
        self.collections: dict[str, dict] = {}

    def get_or_create_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = {"documents": {}}
        return ChromaCollection(name, self.collections[name])


class ChromaCollection:
    def __init__(self, name: str, data: dict):
        self.name = name
        self.data = data

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str] | None = None,
            metadatas: list[dict] | None = None):
        for i, doc_id in enumerate(ids):
            self.data["documents"][doc_id] = {
                "embedding": embeddings[i],
                "document": documents[i] if documents else None,
                "metadata": metadatas[i] if metadatas else {},
            }

    def query(self, query_embeddings: list[list[float]], n_results: int = 10,
              where: dict | None = None):
        query_vec = query_embeddings[0]
        results = []
        for doc_id, doc in self.data["documents"].items():
            if where:
                if not self._match_where(doc["metadata"], where):
                    continue
            score = cosine_similarity(query_vec, doc["embedding"])
            results.append({
                "id": doc_id,
                "distance": 1 - score,
                "document": doc["document"],
                "metadata": doc["metadata"],
            })
        results.sort(key=lambda x: x["distance"])
        return {"ids": [[r["id"] for r in results[:n_results]]],
                "distances": [[r["distance"] for r in results[:n_results]]]}

    def _match_where(self, metadata: dict, where: dict) -> bool:
        for key, condition in where.items():
            if key not in metadata:
                return False
            if isinstance(condition, dict):
                if "$gte" in condition and metadata[key] < condition["$gte"]:
                    return False
                if "$eq" in condition and metadata[key] != condition["$eq"]:
                    return False
            elif metadata[key] != condition:
                return False
        return True


def exercise_3_qdrant_vs_chroma():
    """Compare Qdrant and Chroma APIs for the same search task."""
    print("\n" + "=" * 70)
    print("Exercise 3: Qdrant vs Chroma API Comparison")
    print("=" * 70)

    dim = 16
    n_docs = 100

    # Generate sample data
    docs = []
    for i in range(n_docs):
        docs.append({
            "id": f"doc-{i:03d}",
            "text": f"Document {i} about {'AI' if i % 3 == 0 else 'data' if i % 3 == 1 else 'systems'}",
            "category": ["AI", "data", "systems"][i % 3],
            "year": 2020 + (i % 5),
            "embedding": random_vector(dim),
        })

    query = random_vector(dim)

    # --- Qdrant ---
    print("\n--- Qdrant API ---")
    qdrant = SimulatedQdrant()

    qdrant.create_collection("articles", vector_size=dim, distance="Cosine")
    qdrant.upsert("articles", [
        {"id": d["id"], "vector": d["embedding"],
         "payload": {"category": d["category"], "year": d["year"]}}
        for d in docs
    ])

    start = time.perf_counter()
    qdrant_results = qdrant.search(
        "articles", query, limit=5,
        query_filter={"category": {"$eq": "AI"}, "year": {"$gte": 2023}},
    )
    qdrant_time = time.perf_counter() - start

    print(f"  Search time: {qdrant_time*1000:.2f}ms")
    print(f"  Results: {len(qdrant_results)}")
    for r in qdrant_results[:3]:
        print(f"    {r['id']}: score={r['score']:.4f}, payload={r['payload']}")

    # --- Chroma ---
    print("\n--- Chroma API ---")
    chroma = SimulatedChroma()
    collection = chroma.get_or_create_collection("articles")

    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=[d["embedding"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"category": d["category"], "year": d["year"]} for d in docs],
    )

    start = time.perf_counter()
    chroma_results = collection.query(
        query_embeddings=[query], n_results=5,
        where={"category": "AI"},
    )
    chroma_time = time.perf_counter() - start

    print(f"  Search time: {chroma_time*1000:.2f}ms")
    print(f"  Results: {len(chroma_results['ids'][0])}")
    for doc_id, dist in zip(chroma_results['ids'][0][:3],
                             chroma_results['distances'][0][:3]):
        print(f"    {doc_id}: distance={dist:.4f}")

    # Comparison
    print("\n--- API Comparison ---")
    print(f"  {'Feature':<25} {'Qdrant':<25} {'Chroma'}")
    print(f"  {'-'*75}")
    print(f"  {'Filter syntax':<25} {'must/should/must_not':<25} {'where dict'}")
    print(f"  {'Result format':<25} {'score (higher=better)':<25} {'distance (lower=better)'}")
    print(f"  {'Metadata field':<25} {'payload':<25} {'metadatas'}")
    print(f"  {'Collection create':<25} {'create_collection()':<25} {'get_or_create_collection()'}")
    print(f"  {'Insert method':<25} {'upsert()':<25} {'add()'}")


# =====================================================================
# Exercise 4: Index Versioning Pipeline
# =====================================================================

@dataclass
class IndexVersion:
    version: str
    n_vectors: int
    dim: int
    index_type: str
    created_at: str
    metadata: dict = field(default_factory=dict)


class IndexVersionStore:
    """Manages versioned FAISS-like index saves with metadata."""

    def __init__(self, base_path: str = "/tmp/index_versions"):
        self.base_path = base_path
        self.versions: list[IndexVersion] = []
        self.current_version: str | None = None

    def save_version(
        self,
        index_data: dict,
        metadata: dict | None = None,
    ) -> IndexVersion:
        """Save a new index version."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        iv = IndexVersion(
            version=version,
            n_vectors=index_data.get("ntotal", 0),
            dim=index_data.get("dim", 0),
            index_type=index_data.get("index_type", "unknown"),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self.versions.append(iv)
        self.current_version = version
        return iv

    def rollback(self, target_version: str) -> bool:
        """Rollback to a previous version."""
        matching = [v for v in self.versions if v.version == target_version]
        if not matching:
            return False
        self.current_version = target_version
        return True

    def list_versions(self) -> list[IndexVersion]:
        return list(self.versions)

    def get_current(self) -> IndexVersion | None:
        if not self.current_version:
            return None
        matching = [v for v in self.versions if v.version == self.current_version]
        return matching[0] if matching else None

    def diff(self, v1: str, v2: str) -> dict:
        """Compare two versions."""
        ver1 = next((v for v in self.versions if v.version == v1), None)
        ver2 = next((v for v in self.versions if v.version == v2), None)
        if not ver1 or not ver2:
            return {"error": "Version not found"}
        return {
            "v1": v1,
            "v2": v2,
            "vector_count_delta": ver2.n_vectors - ver1.n_vectors,
            "index_type_changed": ver1.index_type != ver2.index_type,
            "dim_changed": ver1.dim != ver2.dim,
        }


def exercise_4_index_versioning():
    """Demonstrate index versioning with rollback."""
    print("\n" + "=" * 70)
    print("Exercise 4: Index Versioning Pipeline")
    print("=" * 70)

    store = IndexVersionStore()

    # Simulate daily index builds
    builds = [
        {"ntotal": 1_000_000, "dim": 768, "index_type": "IVF4096,SQ8"},
        {"ntotal": 1_050_000, "dim": 768, "index_type": "IVF4096,SQ8"},
        {"ntotal": 500_000, "dim": 768, "index_type": "IVF4096,SQ8"},  # Bug!
        {"ntotal": 1_100_000, "dim": 768, "index_type": "IVF4096,SQ8"},
    ]

    metadata_list = [
        {"recall_at_10": 0.95, "p99_latency_ms": 3.2, "status": "healthy"},
        {"recall_at_10": 0.96, "p99_latency_ms": 3.1, "status": "healthy"},
        {"recall_at_10": 0.72, "p99_latency_ms": 2.8, "status": "degraded"},
        {"recall_at_10": 0.95, "p99_latency_ms": 3.3, "status": "healthy"},
    ]

    print("\nBuilding index versions:")
    for i, (build, meta) in enumerate(zip(builds, metadata_list)):
        v = store.save_version(build, meta)
        status = "OK" if meta["recall_at_10"] >= 0.90 else "ALERT"
        print(f"  [{status}] v{v.version}: {v.n_vectors:,} vectors, "
              f"recall={meta['recall_at_10']:.2f}")

    # Detect recall drop and rollback
    print(f"\nDetecting recall drop in v3...")
    versions = store.list_versions()
    for v in versions:
        if v.metadata.get("recall_at_10", 1.0) < 0.90:
            print(f"  Recall {v.metadata['recall_at_10']:.2f} < threshold 0.90")
            # Find last healthy version
            healthy = [
                ver for ver in versions
                if ver.metadata.get("status") == "healthy"
                and ver.version < v.version
            ]
            if healthy:
                target = healthy[-1]
                store.rollback(target.version)
                print(f"  Rolled back to v{target.version} "
                      f"({target.n_vectors:,} vectors, "
                      f"recall={target.metadata['recall_at_10']:.2f})")

    print(f"\nCurrent active version: v{store.get_current().version}")
    print(f"  Vectors: {store.get_current().n_vectors:,}")


# =====================================================================
# Exercise 5: Benchmark Runner
# =====================================================================

@dataclass
class BenchmarkResult:
    index_type: str
    n_vectors: int
    dim: int
    build_time_s: float
    qps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    recall_at_10: float
    memory_mb: float


def run_benchmark(
    index,
    queries: list[list[float]],
    ground_truth: list[set[int]],
    k: int = 10,
) -> dict:
    """Run benchmark on an index and collect metrics."""
    latencies = []
    recalls = []

    for i, query in enumerate(queries):
        start = time.perf_counter()
        results = index.search(query, k)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        if ground_truth:
            pred_ids = set(r[0] for r in results)
            gt = ground_truth[i]
            recall = len(pred_ids & gt) / len(gt) if gt else 0
            recalls.append(recall)

    latencies.sort()
    n = len(latencies)

    return {
        "qps": 1000.0 / (sum(latencies) / n) if latencies else 0,
        "latency_p50_ms": latencies[n // 2] if latencies else 0,
        "latency_p95_ms": latencies[int(n * 0.95)] if latencies else 0,
        "latency_p99_ms": latencies[int(n * 0.99)] if latencies else 0,
        "recall_at_10": sum(recalls) / len(recalls) if recalls else 0,
        "memory_mb": index.memory_bytes() / (1024 * 1024),
    }


def exercise_5_benchmark_runner():
    """Run benchmarks across index types and display comparison."""
    print("\n" + "=" * 70)
    print("Exercise 5: Benchmark Runner")
    print("=" * 70)

    dim = 32
    n_vectors = 1500
    n_queries = 50
    k = 10

    data = [random_vector(dim) for _ in range(n_vectors)]
    queries = [random_vector(dim) for _ in range(n_queries)]

    # Build ground truth with Flat index
    flat = FlatIndex(dim)
    flat.add(data)
    ground_truth = [
        set(r[0] for r in flat.search(q, k))
        for q in queries
    ]

    # Benchmark each index type
    results = []

    # Flat
    start = time.perf_counter()
    flat_new = FlatIndex(dim)
    flat_new.add(data)
    build_time = time.perf_counter() - start
    metrics = run_benchmark(flat_new, queries, ground_truth, k)
    results.append(BenchmarkResult(
        index_type="Flat", n_vectors=n_vectors, dim=dim,
        build_time_s=build_time, **metrics,
    ))

    # IVF
    start = time.perf_counter()
    ivf = IVFIndex(dim, nlist=16)
    ivf.train(data)
    ivf.add(data)
    ivf.nprobe = 4
    build_time = time.perf_counter() - start
    metrics = run_benchmark(ivf, queries, ground_truth, k)
    results.append(BenchmarkResult(
        index_type="IVF(nlist=16,nprobe=4)", n_vectors=n_vectors, dim=dim,
        build_time_s=build_time, **metrics,
    ))

    # HNSW
    start = time.perf_counter()
    hnsw = HNSWIndex(dim, M=8)
    hnsw.add(data)
    hnsw.ef_search = 32
    build_time = time.perf_counter() - start
    metrics = run_benchmark(hnsw, queries, ground_truth, k)
    results.append(BenchmarkResult(
        index_type="HNSW(M=8,ef=32)", n_vectors=n_vectors, dim=dim,
        build_time_s=build_time, **metrics,
    ))

    # Display results
    print(f"\nBenchmark: {n_vectors} vectors, {dim}d, top-{k}, {n_queries} queries\n")
    header = (f"{'Index Type':<28} {'Build(s)':<10} {'QPS':<10} "
              f"{'p50(ms)':<10} {'p95(ms)':<10} {'p99(ms)':<10} "
              f"{'Recall@10':<10} {'Mem(MB)':<10}")
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r.index_type:<28} {r.build_time_s:<10.4f} {r.qps:<10.0f} "
              f"{r.latency_p50_ms:<10.3f} {r.latency_p95_ms:<10.3f} "
              f"{r.latency_p99_ms:<10.3f} {r.recall_at_10:<10.3f} "
              f"{r.memory_mb:<10.3f}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    random.seed(42)
    exercise_1_faiss_index_comparison()
    exercise_2_milvus_schema_design()
    exercise_3_qdrant_vs_chroma()
    exercise_4_index_versioning()
    exercise_5_benchmark_runner()
    print("\nAll exercises completed.")
