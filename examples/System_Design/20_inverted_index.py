"""
Inverted Index & BM25 Scoring

Demonstrates:
- Inverted index construction
- Boolean search (AND, OR)
- TF-IDF scoring
- BM25 ranking algorithm
- Query processing pipeline

Theory:
- An inverted index maps terms to the documents containing them.
  Core data structure for full-text search engines.
- TF-IDF: Term Frequency × Inverse Document Frequency.
  High for terms that are frequent in a document but rare overall.
- BM25: Probabilistic ranking function. Improves on TF-IDF with
  term frequency saturation and document length normalization.
  BM25(D, Q) = Σ IDF(qi) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × |D|/avgdl))
  Typical: k1 = 1.2, b = 0.75

Adapted from System Design Lesson 20.
"""

import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field


# ── Text Processing ────────────────────────────────────────────────────

# Why: Stop words appear in nearly every document, so they contribute almost zero
# discriminative power to search ranking (IDF is near zero). Removing them reduces
# index size by ~30-40% and speeds up query processing significantly.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
    "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ── Inverted Index ─────────────────────────────────────────────────────

# Why: Storing positions alongside frequency enables phrase queries and proximity
# search (e.g., "machine learning" as an exact phrase, not just two separate words).
# Without positions, the index can only answer bag-of-words queries.
@dataclass
class Posting:
    doc_id: int
    term_freq: int
    positions: list[int] = field(default_factory=list)


# Why: The inverted index reverses the document→terms relationship into
# term→documents, enabling O(1) lookup of which documents contain a query term.
# This is the core data structure behind every full-text search engine
# (Elasticsearch, Lucene, PostgreSQL FTS).
class InvertedIndex:
    """Inverted index with TF-IDF and BM25 scoring."""

    def __init__(self):
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.documents: dict[int, str] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_count = 0
        self.avg_doc_length = 0.0

    def add_document(self, doc_id: int, text: str) -> None:
        """Index a document."""
        self.documents[doc_id] = text
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

        # Update average document length
        total = sum(self.doc_lengths.values())
        self.avg_doc_length = total / self.doc_count

        # Count term frequencies and positions
        term_positions: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        # Add to index
        for term, positions in term_positions.items():
            self.index[term].append(
                Posting(doc_id, len(positions), positions)
            )

    def search_boolean(self, query: str, op: str = "AND") -> list[int]:
        """Boolean search: AND or OR of query terms."""
        terms = tokenize(query)
        if not terms:
            return []

        result_sets = []
        for term in terms:
            doc_ids = {p.doc_id for p in self.index.get(term, [])}
            result_sets.append(doc_ids)

        if op == "AND":
            result = result_sets[0]
            for s in result_sets[1:]:
                result &= s
        else:  # OR
            result = set()
            for s in result_sets:
                result |= s

        return sorted(result)

    def _idf(self, term: str) -> float:
        """Inverse Document Frequency."""
        # Why: IDF penalizes common terms and boosts rare ones. A term appearing
        # in 1 out of 1000 docs is far more informative than one in 900 out of
        # 1000. The +0.5 smoothing prevents division by zero and log(negative).
        df = len(self.index.get(term, []))
        if df == 0:
            return 0.0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def search_tfidf(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """TF-IDF ranked search."""
        terms = tokenize(query)
        scores: dict[int, float] = defaultdict(float)

        for term in terms:
            idf = self._idf(term)
            for posting in self.index.get(term, []):
                # Why: log(1 + tf) dampens the effect of raw term frequency.
                # A document mentioning "python" 100 times is not 100x more
                # relevant than one mentioning it once — diminishing returns.
                tf = math.log(1 + posting.term_freq)
                scores[posting.doc_id] += tf * idf

        # Normalize by document length
        for doc_id in scores:
            scores[doc_id] /= math.sqrt(self.doc_lengths.get(doc_id, 1))

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def search_bm25(self, query: str, top_k: int = 10,
                    k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
        """BM25 ranked search."""
        # Why: BM25 improves on TF-IDF in two key ways: (1) term frequency
        # saturates — the 10th occurrence of a word adds much less score than
        # the 1st (controlled by k1), and (2) document length normalization
        # (controlled by b) prevents long documents from unfairly dominating.
        terms = tokenize(query)
        scores: dict[int, float] = defaultdict(float)

        for term in terms:
            idf = self._idf(term)
            for posting in self.index.get(term, []):
                tf = posting.term_freq
                dl = self.doc_lengths[posting.doc_id]
                avgdl = self.avg_doc_length

                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                scores[posting.doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def get_term_stats(self, term: str) -> dict:
        """Get statistics for a term."""
        postings = self.index.get(term, [])
        return {
            "term": term,
            "document_frequency": len(postings),
            "total_occurrences": sum(p.term_freq for p in postings),
            "idf": self._idf(term),
        }


# ── Demos ──────────────────────────────────────────────────────────────

SAMPLE_DOCS = {
    1: "The quick brown fox jumps over the lazy dog",
    2: "A fast brown dog runs across the green field",
    3: "The fox is quick and clever, outsmarting the lazy hound",
    4: "Python is a popular programming language for data science",
    5: "Data structures and algorithms are fundamental to programming",
    6: "Machine learning algorithms process large datasets efficiently",
    7: "The brown bear sleeps in the forest during winter",
    8: "Search engines use inverted indexes for fast text retrieval",
    9: "Information retrieval systems rank documents by relevance",
    10: "Natural language processing enables text understanding",
}


def demo_index_construction():
    print("=" * 60)
    print("INVERTED INDEX CONSTRUCTION")
    print("=" * 60)

    idx = InvertedIndex()
    for doc_id, text in SAMPLE_DOCS.items():
        idx.add_document(doc_id, text)

    print(f"\n  Indexed {idx.doc_count} documents")
    print(f"  Unique terms: {len(idx.index)}")
    print(f"  Avg document length: {idx.avg_doc_length:.1f} tokens")

    # Show index entries for selected terms
    print(f"\n  Sample index entries:")
    for term in ["quick", "brown", "programming", "algorithms"]:
        postings = idx.index.get(term, [])
        doc_ids = [p.doc_id for p in postings]
        stats = idx.get_term_stats(term)
        print(f"    '{term}': docs={doc_ids}, df={stats['document_frequency']}, "
              f"idf={stats['idf']:.3f}")


def demo_boolean_search():
    print("\n" + "=" * 60)
    print("BOOLEAN SEARCH")
    print("=" * 60)

    idx = InvertedIndex()
    for doc_id, text in SAMPLE_DOCS.items():
        idx.add_document(doc_id, text)

    queries = [
        ("brown fox", "AND"),
        ("brown fox", "OR"),
        ("programming language", "AND"),
        ("data algorithms", "OR"),
    ]

    for query, op in queries:
        results = idx.search_boolean(query, op)
        print(f"\n  '{query}' ({op}): {len(results)} result(s)")
        for doc_id in results:
            print(f"    Doc {doc_id}: {SAMPLE_DOCS[doc_id][:60]}")


def demo_tfidf_vs_bm25():
    print("\n" + "=" * 60)
    print("TF-IDF vs BM25 RANKING")
    print("=" * 60)

    idx = InvertedIndex()
    for doc_id, text in SAMPLE_DOCS.items():
        idx.add_document(doc_id, text)

    query = "fast text retrieval algorithms"
    print(f"\n  Query: '{query}'\n")

    # TF-IDF
    tfidf_results = idx.search_tfidf(query, top_k=5)
    print(f"  TF-IDF ranking:")
    print(f"    {'Rank':>5}  {'Doc':>4}  {'Score':>8}  Text")
    print(f"    {'-'*5}  {'-'*4}  {'-'*8}  {'-'*40}")
    for rank, (doc_id, score) in enumerate(tfidf_results, 1):
        print(f"    {rank:>5}  {doc_id:>4}  {score:>8.4f}  "
              f"{SAMPLE_DOCS[doc_id][:40]}")

    # BM25
    bm25_results = idx.search_bm25(query, top_k=5)
    print(f"\n  BM25 ranking:")
    print(f"    {'Rank':>5}  {'Doc':>4}  {'Score':>8}  Text")
    print(f"    {'-'*5}  {'-'*4}  {'-'*8}  {'-'*40}")
    for rank, (doc_id, score) in enumerate(bm25_results, 1):
        print(f"    {rank:>5}  {doc_id:>4}  {score:>8.4f}  "
              f"{SAMPLE_DOCS[doc_id][:40]}")


def demo_bm25_parameters():
    print("\n" + "=" * 60)
    print("BM25 PARAMETER SENSITIVITY")
    print("=" * 60)

    idx = InvertedIndex()
    for doc_id, text in SAMPLE_DOCS.items():
        idx.add_document(doc_id, text)

    query = "programming data"
    print(f"\n  Query: '{query}'")
    print(f"  Varying k1 (term saturation) and b (length normalization)\n")

    configs = [
        (0.5, 0.75, "Low k1: fast saturation"),
        (1.2, 0.75, "Default k1: standard"),
        (2.0, 0.75, "High k1: slower saturation"),
        (1.2, 0.0, "b=0: no length norm"),
        (1.2, 0.5, "b=0.5: moderate length norm"),
        (1.2, 1.0, "b=1.0: full length norm"),
    ]

    for k1, b, desc in configs:
        results = idx.search_bm25(query, top_k=3, k1=k1, b=b)
        top_docs = [f"D{did}({score:.3f})" for did, score in results]
        print(f"  k1={k1:.1f}, b={b:.2f} [{desc}]")
        print(f"    Top 3: {', '.join(top_docs)}")


def demo_term_stats():
    print("\n" + "=" * 60)
    print("TERM STATISTICS")
    print("=" * 60)

    idx = InvertedIndex()
    for doc_id, text in SAMPLE_DOCS.items():
        idx.add_document(doc_id, text)

    # Collect all terms and sort by IDF
    all_terms = sorted(idx.index.keys())
    stats = [(t, idx.get_term_stats(t)) for t in all_terms]
    stats.sort(key=lambda x: x[1]["idf"])

    print(f"\n  Terms sorted by IDF (low → high rarity):\n")
    print(f"    {'Term':<18} {'DF':>4} {'IDF':>7} {'Occurrences':>12}")
    print(f"    {'-'*18} {'-'*4} {'-'*7} {'-'*12}")
    for term, s in stats[:20]:
        print(f"    {term:<18} {s['document_frequency']:>4} "
              f"{s['idf']:>7.3f} {s['total_occurrences']:>12}")


if __name__ == "__main__":
    demo_index_construction()
    demo_boolean_search()
    demo_tfidf_vs_bm25()
    demo_bm25_parameters()
    demo_term_stats()
