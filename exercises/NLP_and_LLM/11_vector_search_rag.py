"""
Exercises for Lesson 11: Vector Databases
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional, Any


# ============================================================
# Shared utilities: lightweight vector store simulation
# (No external dependencies like chromadb or faiss required)
# ============================================================

class SimpleVectorStore:
    """
    Minimal in-memory vector store for exercise demonstrations.
    Simulates core Chroma/FAISS behavior with numpy.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding from text (hash-based)."""
        h = hashlib.sha256(text.encode()).digest()
        # Use hash bytes as seeds for a reproducible vector
        rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
        vec = rng.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def add(self, documents: List[str], ids: List[str],
            metadatas: Optional[List[Dict]] = None,
            embeddings: Optional[List[np.ndarray]] = None):
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            self.ids.append(doc_id)
            self.documents.append(doc)
            self.metadatas.append(metadatas[i] if metadatas else {})
            if embeddings is not None:
                self.vectors.append(np.array(embeddings[i], dtype=np.float32))
            else:
                self.vectors.append(self._embed(doc))

    def get(self, ids: List[str]) -> Dict:
        result_ids = []
        result_docs = []
        result_meta = []
        for doc_id in ids:
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                result_ids.append(doc_id)
                result_docs.append(self.documents[idx])
                result_meta.append(self.metadatas[idx])
        return {"ids": result_ids, "documents": result_docs, "metadatas": result_meta}

    def query(self, query_texts: List[str], n_results: int = 5,
              where: Optional[Dict] = None) -> Dict:
        """Cosine-similarity search with optional metadata filtering."""
        query_vec = self._embed(query_texts[0])

        # Filter by metadata
        candidates = list(range(len(self.ids)))
        if where:
            candidates = [i for i in candidates if self._match_filter(self.metadatas[i], where)]

        if not candidates:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

        # Compute cosine distances
        vecs = np.array([self.vectors[i] for i in candidates])
        sims = vecs @ query_vec
        distances = 1.0 - sims  # cosine distance

        top_k = min(n_results, len(candidates))
        top_indices = np.argsort(distances)[:top_k]

        return {
            "ids": [[self.ids[candidates[i]] for i in top_indices]],
            "documents": [[self.documents[candidates[i]] for i in top_indices]],
            "distances": [[float(distances[i]) for i in top_indices]],
            "metadatas": [[self.metadatas[candidates[i]] for i in top_indices]],
        }

    def delete(self, ids: List[str]):
        for doc_id in ids:
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.ids.pop(idx)
                self.vectors.pop(idx)
                self.documents.pop(idx)
                self.metadatas.pop(idx)

    def _match_filter(self, meta: Dict, where: Dict) -> bool:
        """Evaluate Chroma-style metadata filter."""
        for key, condition in where.items():
            if key == "$and":
                return all(self._match_filter(meta, c) for c in condition)
            if key == "$or":
                return any(self._match_filter(meta, c) for c in condition)
            if isinstance(condition, dict):
                for op, val in condition.items():
                    actual = meta.get(key)
                    if actual is None:
                        return False
                    if op == "$eq" and actual != val:
                        return False
                    if op == "$ne" and actual == val:
                        return False
                    if op == "$gt" and not (actual > val):
                        return False
                    if op == "$gte" and not (actual >= val):
                        return False
                    if op == "$lt" and not (actual < val):
                        return False
                    if op == "$lte" and not (actual <= val):
                        return False
                    if op == "$in" and actual not in val:
                        return False
                    if op == "$nin" and actual in val:
                        return False
            else:
                # Implicit $eq
                if meta.get(key) != condition:
                    return False
        return True

    @property
    def count(self):
        return len(self.ids)


# === Exercise 1: FAISS Index Type Selection ===
# Problem: For each scenario, choose the appropriate FAISS index type
# and explain why. Then demonstrate the memory/accuracy trade-offs
# using a simulated comparison.

def exercise_1():
    """FAISS index type selection for different scenarios."""
    print("=" * 60)
    print("Exercise 1: FAISS Index Type Selection")
    print("=" * 60)

    # Scenario definitions
    scenarios = {
        "A. Medical record search": {
            "dataset_size": 50_000,
            "constraints": "100% accuracy required",
            "best_index": "IndexFlatL2",
            "reasoning": (
                "Exact search (no approximation). 50K vectors x 384 dims x 4 bytes "
                "= ~73MB — manageable. Medical decisions require precision."
            ),
        },
        "B. Real-time product search": {
            "dataset_size": 10_000_000,
            "constraints": "<50ms latency required",
            "best_index": "IndexHNSWFlat",
            "reasoning": (
                "98%+ accuracy at millisecond latency. No training required. "
                "Best recall/latency trade-off for real-time serving."
            ),
        },
        "C. Mobile app embedding search": {
            "dataset_size": 500_000,
            "constraints": "500MB memory limit",
            "best_index": "IndexIVFPQ",
            "reasoning": (
                "PQ compresses vectors from 1536 bytes to ~64 bytes (24x). "
                "500K x 64 bytes = ~32MB, well within budget."
            ),
        },
        "D. Nightly batch recommendation": {
            "dataset_size": 2_000_000,
            "constraints": "Accuracy >95%, training time OK",
            "best_index": "IndexIVFFlat",
            "reasoning": (
                "Good accuracy with fast approximate search. Training is one-time. "
                "nprobe tunable for accuracy/speed trade-off."
            ),
        },
    }

    print("\nScenario Analysis:")
    print("-" * 60)
    for scenario, details in scenarios.items():
        print(f"\n  {scenario}")
        print(f"    Dataset:    {details['dataset_size']:,} vectors")
        print(f"    Constraint: {details['constraints']}")
        print(f"    Best Index: {details['best_index']}")
        print(f"    Reasoning:  {details['reasoning']}")

    # Demonstrate memory calculations
    print("\n\nMemory Comparison (384-dim, float32):")
    print("-" * 60)
    dimension = 384
    n_vectors = 500_000

    index_types = {
        "Flat (exact)": dimension * 4,              # 4 bytes per float32
        "IVF (approximate)": dimension * 4,          # same storage, adds cluster centroids
        "HNSW": dimension * 4 + 32 * 8,             # vectors + graph edges (M=32)
        "PQ (m=8, nbits=8)": 8 * 1,                 # 8 sub-vectors, 1 byte each
        "IVF+PQ": 8 * 1,                            # compressed within clusters
    }

    for name, bytes_per_vec in index_types.items():
        total_mb = n_vectors * bytes_per_vec / (1024 ** 2)
        print(f"  {name:<25} {bytes_per_vec:>4} bytes/vec  →  {total_mb:>8.1f} MB total")

    print("\n  Key insight: PQ achieves ~24x compression vs Flat,")
    print("  making it feasible for memory-constrained environments.")


# === Exercise 2: Chroma Metadata Filtering ===
# Problem: Write Chroma queries for three requirements involving
# year, category, and citation filters.

def exercise_2():
    """Chroma metadata filtering with complex queries."""
    print("\n" + "=" * 60)
    print("Exercise 2: Chroma Metadata Filtering")
    print("=" * 60)

    # Create collection with sample research papers
    store = SimpleVectorStore(dimension=64)

    papers = [
        ("Attention Is All You Need", {"year": 2017, "category": "nlp", "citations": 50000}),
        ("BERT: Pre-training of Bidirectional Transformers", {"year": 2018, "category": "nlp", "citations": 30000}),
        ("ResNet: Deep Residual Learning", {"year": 2016, "category": "cv", "citations": 80000}),
        ("GPT-4 Technical Report", {"year": 2023, "category": "nlp", "citations": 2000}),
        ("Scaling Laws for Neural Language Models", {"year": 2020, "category": "ml", "citations": 3000}),
        ("Vision Transformer (ViT)", {"year": 2021, "category": "cv", "citations": 12000}),
        ("LLaMA: Open Foundation Models", {"year": 2023, "category": "ml", "citations": 5000}),
        ("Diffusion Models Beat GANs", {"year": 2021, "category": "cv", "citations": 4000}),
        ("DPO: Direct Preference Optimization", {"year": 2024, "category": "ml", "citations": 800}),
        ("Mamba: Linear-Time Sequence Modeling", {"year": 2024, "category": "nlp", "citations": 600}),
        ("Segment Anything Model", {"year": 2023, "category": "cv", "citations": 7000}),
        ("Constitutional AI", {"year": 2022, "category": "ml", "citations": 1500}),
    ]

    store.add(
        documents=[p[0] for p in papers],
        ids=[f"p{i}" for i in range(len(papers))],
        metadatas=[p[1] for p in papers],
    )

    # Query 1: NLP papers from 2022 or later
    print("\nQuery 1: NLP papers from 2022+")
    print("-" * 40)
    results_1 = store.query(
        query_texts=["transformer architecture"],
        n_results=10,
        where={
            "$and": [
                {"year": {"$gte": 2022}},
                {"category": {"$eq": "nlp"}},
            ]
        },
    )
    for doc, meta in zip(results_1["documents"][0], results_1["metadatas"][0]):
        print(f"  [{meta['year']}] [{meta['category']}] {doc}")

    # Query 2: ML or CV with >100 citations
    print("\nQuery 2: ML or CV papers with >100 citations")
    print("-" * 40)
    results_2 = store.query(
        query_texts=["neural network"],
        n_results=10,
        where={
            "$and": [
                {"category": {"$in": ["ml", "cv"]}},
                {"citations": {"$gt": 100}},
            ]
        },
    )
    for doc, meta in zip(results_2["documents"][0], results_2["metadatas"][0]):
        print(f"  [{meta['year']}] [{meta['category']}] citations={meta['citations']:,}  {doc}")

    # Query 3: 2023-2025, not CV
    print("\nQuery 3: Papers from 2023-2025, NOT in 'cv' category")
    print("-" * 40)
    results_3 = store.query(
        query_texts=["deep learning"],
        n_results=10,
        where={
            "$and": [
                {"year": {"$gte": 2023}},
                {"year": {"$lte": 2025}},
                {"category": {"$ne": "cv"}},
            ]
        },
    )
    for doc, meta in zip(results_3["documents"][0], results_3["metadatas"][0]):
        print(f"  [{meta['year']}] [{meta['category']}] {doc}")

    # Reference: Chroma filter operators
    print("\n  Chroma filter operators reference:")
    print("    $eq, $ne:  equal, not equal")
    print("    $gt, $gte, $lt, $lte:  numeric comparisons")
    print("    $in, $nin:  membership in list")
    print("    $and, $or:  logical combinations")
    print("\n  Common pitfall: $and/$or take a list; ALL conditions must")
    print("  be at the same nesting level within the where clause.")


# === Exercise 3: Deduplication with Content Hashing ===
# Problem: Extend upsert_documents to handle document updates:
# if same logical key has different content, update; if identical, skip.

def exercise_3():
    """Deduplication with content hashing and change detection."""
    print("\n" + "=" * 60)
    print("Exercise 3: Deduplication with Content Hashing")
    print("=" * 60)

    def get_content_hash(text: str) -> str:
        """Generate a stable ID from content."""
        return hashlib.md5(text.encode()).hexdigest()

    def smart_upsert(
        texts: List[str],
        doc_keys: List[str],
        collection: SimpleVectorStore,
        metadatas: Optional[List[Dict]] = None,
    ) -> Dict[str, int]:
        """
        Upsert documents with change detection.

        Strategy: store both the logical key AND content hash in metadata.
        When re-indexing, check if the content hash changed.

        Returns: {"added": N, "updated": N, "skipped": N}
        """
        stats = {"added": 0, "updated": 0, "skipped": 0}

        for i, (text, key) in enumerate(zip(texts, doc_keys)):
            new_hash = get_content_hash(text)
            meta = dict(metadatas[i]) if metadatas else {}
            meta["doc_key"] = key
            meta["content_hash"] = new_hash

            # Check if this logical key already exists (search metadata)
            existing_indices = [
                j for j, m in enumerate(collection.metadatas)
                if m.get("doc_key") == key
            ]

            if not existing_indices:
                # New document
                collection.add(
                    documents=[text],
                    ids=[new_hash],
                    metadatas=[meta],
                )
                stats["added"] += 1

            elif collection.metadatas[existing_indices[0]].get("content_hash") == new_hash:
                # Identical content — skip
                stats["skipped"] += 1

            else:
                # Content changed — delete old, add new
                old_id = collection.ids[existing_indices[0]]
                collection.delete(ids=[old_id])
                collection.add(
                    documents=[text],
                    ids=[new_hash],
                    metadatas=[meta],
                )
                stats["updated"] += 1

        return stats

    # Test the smart_upsert function
    store = SimpleVectorStore(dimension=32)

    print("\nRound 1: Adding initial documents")
    result1 = smart_upsert(
        texts=["Version 1 of doc A", "Version 1 of doc B"],
        doc_keys=["doc_A", "doc_B"],
        collection=store,
    )
    print(f"  Result: {result1}")
    print(f"  Store count: {store.count}")
    assert result1 == {"added": 2, "updated": 0, "skipped": 0}

    print("\nRound 2: A changed, B same")
    result2 = smart_upsert(
        texts=["Version 2 of doc A", "Version 1 of doc B"],
        doc_keys=["doc_A", "doc_B"],
        collection=store,
    )
    print(f"  Result: {result2}")
    print(f"  Store count: {store.count}")
    assert result2 == {"added": 0, "updated": 1, "skipped": 1}

    print("\nRound 3: Both same (no changes)")
    result3 = smart_upsert(
        texts=["Version 2 of doc A", "Version 1 of doc B"],
        doc_keys=["doc_A", "doc_B"],
        collection=store,
    )
    print(f"  Result: {result3}")
    assert result3 == {"added": 0, "updated": 0, "skipped": 2}

    print("\nRound 4: Add new doc C")
    result4 = smart_upsert(
        texts=["Version 2 of doc A", "Version 1 of doc B", "Version 1 of doc C"],
        doc_keys=["doc_A", "doc_B", "doc_C"],
        collection=store,
    )
    print(f"  Result: {result4}")
    print(f"  Store count: {store.count}")
    assert result4 == {"added": 1, "updated": 0, "skipped": 2}

    print("\n  Why content-hash IDs matter:")
    print("    If you used sequential IDs, you'd have to compare document text")
    print("    on every upsert. With content hashes, unchanged documents always")
    print("    generate the same ID — no text comparison needed.")
    print("\n  All assertions passed!")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
