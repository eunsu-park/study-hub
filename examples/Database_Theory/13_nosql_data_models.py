"""
NoSQL Data Models — In-Memory Simulations

Demonstrates the four major NoSQL data model paradigms without external dependencies:
- Key-Value Store: hash-based storage with TTL and atomic operations
- Document Store: JSON documents with nested queries (SQLite JSON)
- Wide-Column Store: column-family storage with sparse columns
- Graph Database: nodes, edges, and traversal algorithms

Theory:
- Each NoSQL model optimizes for a different access pattern
- Key-Value: O(1) lookups, simplest API, opaque values
- Document: rich queries on semi-structured data, schema flexibility
- Wide-Column: efficient for sparse, wide rows with column-family grouping
- Graph: relationship-centric queries, traversals, shortest paths
- CAP theorem forces trade-offs; each model makes different choices

All examples use only the Python standard library + sqlite3.
"""

import sqlite3
import json
import time
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# 1. KEY-VALUE STORE
# ============================================================

class KeyValueStore:
    """In-memory key-value store with TTL, atomic increment, and batch ops.

    Simulates Redis-like functionality: GET, PUT, DELETE, EXPIRE, INCR, MGET.
    Uses a Python dict as the underlying hash table.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}  # key -> expiry timestamp

    def _is_expired(self, key: str) -> bool:
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]
            return True
        return False

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a key-value pair with optional TTL in seconds."""
        self._data[key] = value
        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            del self._expiry[key]

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key. Returns None if missing or expired."""
        if self._is_expired(key):
            return None
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if key existed."""
        self._expiry.pop(key, None)
        return self._data.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        if self._is_expired(key):
            return False
        return key in self._data

    def incr(self, key: str, amount: int = 1) -> int:
        """Atomic increment. Initializes to 0 if key does not exist."""
        if self._is_expired(key) or key not in self._data:
            self._data[key] = 0
        self._data[key] += amount
        return self._data[key]

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Batch retrieval of multiple keys."""
        return [self.get(k) for k in keys]

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """List keys, optionally filtering by prefix."""
        result = []
        for k in list(self._data.keys()):
            if self._is_expired(k):
                continue
            if pattern is None or k.startswith(pattern):
                result.append(k)
        return result


def demonstrate_key_value_store():
    """Demonstrate key-value store operations."""
    print("=" * 60)
    print("1. KEY-VALUE STORE")
    print("=" * 60)
    print()

    kv = KeyValueStore()

    # Basic CRUD
    print("1.1 Basic CRUD Operations")
    print("-" * 60)
    kv.put("user:1001", {"name": "Alice", "email": "alice@example.com"})
    kv.put("user:1002", {"name": "Bob", "email": "bob@example.com"})
    kv.put("session:abc123", {"user_id": 1001, "role": "admin"})

    print(f"  GET user:1001   -> {kv.get('user:1001')}")
    print(f"  GET session:abc -> {kv.get('session:abc123')}")
    print(f"  EXISTS user:1002 -> {kv.exists('user:1002')}")
    print(f"  EXISTS user:9999 -> {kv.exists('user:9999')}")

    kv.delete("user:1002")
    print(f"  DELETE user:1002, EXISTS -> {kv.exists('user:1002')}")

    # Atomic increment (counter pattern)
    print("\n1.2 Atomic Increment (Page View Counter)")
    print("-" * 60)
    for _ in range(5):
        kv.incr("page:views:home")
    print(f"  page:views:home = {kv.get('page:views:home')}")

    kv.incr("page:views:home", 10)
    print(f"  After INCR by 10 = {kv.get('page:views:home')}")

    # Batch retrieval
    print("\n1.3 Batch Retrieval (MGET)")
    print("-" * 60)
    kv.put("config:max_retries", 3)
    kv.put("config:timeout", 30)
    kv.put("config:debug", False)
    results = kv.mget(["config:max_retries", "config:timeout", "config:debug"])
    for key, val in zip(["max_retries", "timeout", "debug"], results):
        print(f"  {key} = {val}")

    # Key patterns (namespace simulation)
    print("\n1.4 Key Namespacing")
    print("-" * 60)
    user_keys = kv.keys("user:")
    config_keys = kv.keys("config:")
    print(f"  user:* keys   -> {user_keys}")
    print(f"  config:* keys -> {config_keys}")

    # TTL demonstration
    print("\n1.5 TTL (Time-To-Live)")
    print("-" * 60)
    kv.put("cache:result", "computed_value", ttl=0)  # expires immediately
    time.sleep(0.01)
    print(f"  cache:result after expiry -> {kv.get('cache:result')}")
    print("  (TTL=0 causes immediate expiration)")

    print()


# ============================================================
# 2. DOCUMENT STORE (SQLite JSON)
# ============================================================

def demonstrate_document_store():
    """Demonstrate document store with nested queries and aggregations."""
    print("=" * 60)
    print("2. DOCUMENT STORE (SQLite JSON)")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create collections as tables
    cursor.execute('''
        CREATE TABLE collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at REAL DEFAULT (julianday('now'))
        )
    ''')
    cursor.execute('''
        CREATE INDEX idx_collection ON collections(collection)
    ''')

    # Insert documents with flexible schemas
    print("2.1 Schema Flexibility (Different Document Structures)")
    print("-" * 60)

    orders = [
        {
            "_id": "order_001",
            "customer": {"name": "Alice", "tier": "gold"},
            "items": [
                {"product": "Laptop", "qty": 1, "price": 999.99},
                {"product": "Mouse", "qty": 2, "price": 29.99}
            ],
            "total": 1059.97,
            "status": "shipped",
            "tags": ["electronics", "priority"]
        },
        {
            "_id": "order_002",
            "customer": {"name": "Bob", "tier": "silver"},
            "items": [
                {"product": "Book", "qty": 3, "price": 19.99}
            ],
            "total": 59.97,
            "status": "pending",
            "discount_code": "SAVE10"  # Extra field not in order_001
        },
        {
            "_id": "order_003",
            "customer": {"name": "Alice", "tier": "gold"},
            "items": [
                {"product": "Keyboard", "qty": 1, "price": 149.99},
                {"product": "Monitor", "qty": 1, "price": 399.99}
            ],
            "total": 549.98,
            "status": "shipped",
            "notes": "Gift wrap requested"  # Extra field
        }
    ]

    for order in orders:
        cursor.execute(
            "INSERT INTO collections (collection, data) VALUES (?, ?)",
            ("orders", json.dumps(order))
        )
    print(f"  Inserted {len(orders)} order documents (each with different fields)")

    # Nested field queries
    print("\n2.2 Nested Field Queries")
    print("-" * 60)

    print("  Query: Find orders by customer.name = 'Alice'")
    cursor.execute("""
        SELECT json_extract(data, '$._id') as id,
               json_extract(data, '$.total') as total,
               json_extract(data, '$.status') as status
        FROM collections
        WHERE collection = 'orders'
          AND json_extract(data, '$.customer.name') = 'Alice'
    """)
    for row in cursor.fetchall():
        print(f"    {row[0]}: total=${row[1]:.2f}, status={row[2]}")

    # Array element queries
    print("\n2.3 Array Element Queries")
    print("-" * 60)

    print("  Query: Find orders containing 'Laptop'")
    cursor.execute("""
        SELECT DISTINCT json_extract(c.data, '$._id') as id,
               json_extract(c.data, '$.total') as total
        FROM collections c, json_each(json_extract(c.data, '$.items')) as items
        WHERE c.collection = 'orders'
          AND json_extract(items.value, '$.product') = 'Laptop'
    """)
    for row in cursor.fetchall():
        print(f"    {row[0]}: total=${row[1]:.2f}")

    # Aggregation
    print("\n2.4 Aggregation (Revenue by Customer Tier)")
    print("-" * 60)
    cursor.execute("""
        SELECT json_extract(data, '$.customer.tier') as tier,
               COUNT(*) as order_count,
               SUM(json_extract(data, '$.total')) as total_revenue
        FROM collections
        WHERE collection = 'orders'
        GROUP BY tier
        ORDER BY total_revenue DESC
    """)
    for row in cursor.fetchall():
        print(f"    Tier: {row[0]:8} | Orders: {row[1]} | Revenue: ${row[2]:.2f}")

    # Embedding vs Referencing
    print("\n2.5 Embedding vs Referencing Pattern")
    print("-" * 60)
    print("  EMBEDDED (denormalized): customer data inside order document")
    print("    Pro: Single read fetches order + customer info")
    print("    Con: Customer data duplicated across orders")
    print()
    print("  REFERENCED (normalized): customer_id links to separate doc")

    # Demonstrate referenced pattern
    cursor.execute('''
        CREATE TABLE customers (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE orders_ref (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            data TEXT NOT NULL
        )
    ''')

    cursor.execute("INSERT INTO customers VALUES (?, ?)",
                   ("C001", json.dumps({"name": "Alice", "tier": "gold"})))
    cursor.execute("INSERT INTO orders_ref VALUES (?, ?, ?)",
                   ("O001", "C001", json.dumps({"total": 1059.97})))

    # JOIN equivalent ($lookup in MongoDB)
    cursor.execute("""
        SELECT c.data, o.data
        FROM orders_ref o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.id = 'O001'
    """)
    row = cursor.fetchone()
    print(f"    Referenced lookup: customer={row[0]}, order={row[1]}")

    conn.close()
    print()


# ============================================================
# 3. WIDE-COLUMN STORE
# ============================================================

class ColumnFamily:
    """A column family: a group of related columns stored together.

    In wide-column stores (Cassandra, HBase, Bigtable):
    - Rows are identified by a row key
    - Columns are grouped into column families
    - Each row can have different columns (sparse)
    - Columns are sorted within each row
    """

    def __init__(self, name: str):
        self.name = name
        # row_key -> {column_name: value}
        self._rows: Dict[str, Dict[str, Any]] = {}

    def put(self, row_key: str, column: str, value: Any) -> None:
        if row_key not in self._rows:
            self._rows[row_key] = {}
        self._rows[row_key][column] = value

    def get(self, row_key: str, column: Optional[str] = None) -> Any:
        if row_key not in self._rows:
            return None
        if column is None:
            return dict(self._rows[row_key])  # entire row
        return self._rows[row_key].get(column)

    def delete(self, row_key: str, column: Optional[str] = None) -> None:
        if row_key in self._rows:
            if column is None:
                del self._rows[row_key]
            else:
                self._rows[row_key].pop(column, None)

    def scan(self, start_key: str = "", end_key: str = "~",
             columns: Optional[List[str]] = None) -> List[Tuple[str, Dict]]:
        """Range scan over row keys (sorted order)."""
        results = []
        for key in sorted(self._rows.keys()):
            if start_key <= key <= end_key:
                row = self._rows[key]
                if columns:
                    row = {c: v for c, v in row.items() if c in columns}
                if row:
                    results.append((key, row))
        return results

    def row_count(self) -> int:
        return len(self._rows)


class WideColumnStore:
    """Simulates a wide-column store with multiple column families.

    Models the Bigtable/Cassandra data model:
    - Table contains multiple column families
    - Each column family stores a group of related columns
    - Row keys are globally sorted (enables efficient range scans)
    """

    def __init__(self):
        self._families: Dict[str, ColumnFamily] = {}

    def create_column_family(self, name: str) -> ColumnFamily:
        cf = ColumnFamily(name)
        self._families[name] = cf
        return cf

    def get_family(self, name: str) -> Optional[ColumnFamily]:
        return self._families.get(name)

    def families(self) -> List[str]:
        return list(self._families.keys())


def demonstrate_wide_column_store():
    """Demonstrate wide-column store concepts."""
    print("=" * 60)
    print("3. WIDE-COLUMN STORE")
    print("=" * 60)
    print()

    store = WideColumnStore()

    # Create column families (like Cassandra/HBase)
    profile_cf = store.create_column_family("profile")
    activity_cf = store.create_column_family("activity")
    metrics_cf = store.create_column_family("metrics")

    print("3.1 Column Family Structure")
    print("-" * 60)
    print(f"  Column families: {store.families()}")
    print("  Each family groups related columns (stored together on disk)")

    # Insert user data across column families
    print("\n3.2 Inserting Data (Sparse Columns)")
    print("-" * 60)

    # User 1001: has all profile fields
    profile_cf.put("user:1001", "name", "Alice")
    profile_cf.put("user:1001", "email", "alice@example.com")
    profile_cf.put("user:1001", "city", "New York")
    profile_cf.put("user:1001", "tier", "gold")

    # User 1002: different set of columns (sparse!)
    profile_cf.put("user:1002", "name", "Bob")
    profile_cf.put("user:1002", "email", "bob@example.com")
    profile_cf.put("user:1002", "phone", "555-1234")  # No city, but has phone

    # User 1003: minimal profile
    profile_cf.put("user:1003", "name", "Charlie")

    print("  user:1001 -> name, email, city, tier  (4 columns)")
    print("  user:1002 -> name, email, phone        (3 columns, different!)")
    print("  user:1003 -> name                      (1 column, sparse)")
    print("  Each row can have different columns — no wasted storage")

    # Activity tracking (time-series style)
    print("\n3.3 Time-Series Pattern (Activity Tracking)")
    print("-" * 60)

    activity_cf.put("user:1001", "2024-01-15:login", "web")
    activity_cf.put("user:1001", "2024-01-15:purchase", "order_001")
    activity_cf.put("user:1001", "2024-01-16:login", "mobile")
    activity_cf.put("user:1001", "2024-01-16:view", "product_42")

    activity_cf.put("user:1002", "2024-01-15:login", "web")
    activity_cf.put("user:1002", "2024-01-15:search", "laptop")

    print("  Row key: user:1001")
    row = activity_cf.get("user:1001")
    for col, val in sorted(row.items()):
        print(f"    {col}: {val}")

    # Range scan
    print("\n3.4 Range Scan")
    print("-" * 60)
    print("  Scan all users in activity family:")
    for key, cols in activity_cf.scan():
        print(f"    {key}: {len(cols)} activity entries")

    # Metrics with counters
    print("\n3.5 Counter Columns (Metrics)")
    print("-" * 60)
    pages = ["home", "products", "checkout", "home", "home", "products"]
    for page in pages:
        current = metrics_cf.get("page_views", page) or 0
        metrics_cf.put("page_views", page, current + 1)

    print("  Page view counts:")
    row = metrics_cf.get("page_views")
    for page, count in sorted(row.items(), key=lambda x: -x[1]):
        print(f"    {page:15} = {count}")

    # When to use wide-column
    print("\n3.6 Wide-Column Store Characteristics")
    print("-" * 60)
    print("  Strengths:")
    print("    - Efficient range scans on sorted row keys")
    print("    - Sparse columns (no storage wasted on NULLs)")
    print("    - Column families allow physical co-location")
    print("    - Excellent for time-series and IoT data")
    print("  Weaknesses:")
    print("    - No JOINs or complex queries")
    print("    - Schema must be designed around query patterns")
    print("    - No secondary indexes (in pure model)")

    print()


# ============================================================
# 4. GRAPH DATABASE
# ============================================================

class GraphDatabase:
    """In-memory property graph database.

    Implements the labeled property graph model used by Neo4j:
    - Nodes have labels and properties
    - Edges (relationships) have types, direction, and properties
    - Supports traversal queries (BFS, shortest path)
    """

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}   # id -> {label, props}
        self._edges: List[Dict[str, Any]] = []         # list of edge dicts
        self._adjacency: Dict[str, List[int]] = defaultdict(list)  # node_id -> edge indices
        self._reverse_adj: Dict[str, List[int]] = defaultdict(list)

    def add_node(self, node_id: str, label: str, **properties) -> None:
        self._nodes[node_id] = {"label": label, "properties": properties}

    def add_edge(self, source: str, target: str, rel_type: str,
                 **properties) -> None:
        idx = len(self._edges)
        self._edges.append({
            "source": source,
            "target": target,
            "type": rel_type,
            "properties": properties
        })
        self._adjacency[source].append(idx)
        self._reverse_adj[target].append(idx)

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str, rel_type: Optional[str] = None,
                  direction: str = "outgoing") -> List[Tuple[str, str, Dict]]:
        """Get neighbors: returns list of (neighbor_id, rel_type, edge_props)."""
        results = []
        if direction in ("outgoing", "both"):
            for idx in self._adjacency.get(node_id, []):
                edge = self._edges[idx]
                if rel_type is None or edge["type"] == rel_type:
                    results.append((edge["target"], edge["type"],
                                    edge["properties"]))
        if direction in ("incoming", "both"):
            for idx in self._reverse_adj.get(node_id, []):
                edge = self._edges[idx]
                if rel_type is None or edge["type"] == rel_type:
                    results.append((edge["source"], edge["type"],
                                    edge["properties"]))
        return results

    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """BFS shortest path between two nodes."""
        if start not in self._nodes or end not in self._nodes:
            return None
        visited: Set[str] = {start}
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            for neighbor_id, _, _ in self.neighbors(current, direction="both"):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        return None

    def find_by_label(self, label: str) -> List[Tuple[str, Dict]]:
        """Find all nodes with a given label."""
        return [(nid, data) for nid, data in self._nodes.items()
                if data["label"] == label]

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self._nodes), "edges": len(self._edges)}


def demonstrate_graph_database():
    """Demonstrate graph database with social network and shortest path."""
    print("=" * 60)
    print("4. GRAPH DATABASE")
    print("=" * 60)
    print()

    graph = GraphDatabase()

    # Build a social network graph
    print("4.1 Building a Social Network Graph")
    print("-" * 60)

    # People
    people = [
        ("alice", "Person", {"name": "Alice", "age": 30, "city": "New York"}),
        ("bob", "Person", {"name": "Bob", "age": 28, "city": "London"}),
        ("charlie", "Person", {"name": "Charlie", "age": 35, "city": "Tokyo"}),
        ("diana", "Person", {"name": "Diana", "age": 32, "city": "Berlin"}),
        ("eve", "Person", {"name": "Eve", "age": 27, "city": "New York"}),
    ]
    for pid, label, props in people:
        graph.add_node(pid, label, **props)

    # Companies
    graph.add_node("techcorp", "Company", name="TechCorp", industry="Technology")
    graph.add_node("datalab", "Company", name="DataLab", industry="Data Science")

    # Skills
    for skill in ["Python", "SQL", "GraphDB", "ML"]:
        graph.add_node(skill.lower(), "Skill", name=skill)

    # Relationships
    graph.add_edge("alice", "bob", "KNOWS", since=2020)
    graph.add_edge("bob", "charlie", "KNOWS", since=2019)
    graph.add_edge("charlie", "diana", "KNOWS", since=2021)
    graph.add_edge("alice", "eve", "KNOWS", since=2022)
    graph.add_edge("diana", "eve", "KNOWS", since=2023)

    graph.add_edge("alice", "techcorp", "WORKS_AT", role="Engineer", since=2021)
    graph.add_edge("bob", "techcorp", "WORKS_AT", role="Manager", since=2018)
    graph.add_edge("charlie", "datalab", "WORKS_AT", role="Scientist", since=2020)

    graph.add_edge("alice", "python", "HAS_SKILL", level="expert")
    graph.add_edge("alice", "sql", "HAS_SKILL", level="advanced")
    graph.add_edge("bob", "python", "HAS_SKILL", level="intermediate")
    graph.add_edge("charlie", "ml", "HAS_SKILL", level="expert")
    graph.add_edge("charlie", "python", "HAS_SKILL", level="advanced")

    stats = graph.stats()
    print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    # Traversal queries
    print("\n4.2 Traversal: Who Does Alice Know?")
    print("-" * 60)
    for neighbor, rel_type, props in graph.neighbors("alice", "KNOWS"):
        node = graph.get_node(neighbor)
        print(f"  Alice --[KNOWS since {props.get('since')}]--> "
              f"{node['properties']['name']} ({node['properties']['city']})")

    print("\n4.3 Traversal: Who Works at TechCorp?")
    print("-" * 60)
    for neighbor, rel_type, props in graph.neighbors("techcorp", "WORKS_AT",
                                                      direction="incoming"):
        node = graph.get_node(neighbor)
        print(f"  {node['properties']['name']} --[WORKS_AT role={props['role']}]--> TechCorp")

    print("\n4.4 Multi-Hop: Alice's Skills")
    print("-" * 60)
    for skill_id, rel_type, props in graph.neighbors("alice", "HAS_SKILL"):
        skill_node = graph.get_node(skill_id)
        print(f"  Alice --[HAS_SKILL level={props['level']}]--> "
              f"{skill_node['properties']['name']}")

    # Shortest path
    print("\n4.5 Shortest Path: Alice to Diana")
    print("-" * 60)
    path = graph.shortest_path("alice", "diana")
    if path:
        names = [graph.get_node(n)["properties"].get("name", n) for n in path]
        print(f"  Path: {' -> '.join(names)}")
        print(f"  Hops: {len(path) - 1}")
    else:
        print("  No path found")

    # Another shortest path
    print("\n4.6 Shortest Path: Bob to Eve")
    print("-" * 60)
    path = graph.shortest_path("bob", "eve")
    if path:
        names = [graph.get_node(n)["properties"].get("name", n) for n in path]
        print(f"  Path: {' -> '.join(names)}")
        print(f"  Hops: {len(path) - 1}")

    # Find by label
    print("\n4.7 Find All People")
    print("-" * 60)
    for node_id, data in graph.find_by_label("Person"):
        p = data["properties"]
        print(f"  {p['name']:10} age={p['age']}, city={p['city']}")

    # Graph vs relational comparison
    print("\n4.8 Graph vs Relational for Relationship Queries")
    print("-" * 60)
    print("  Relational: 'Find friends of friends'")
    print("    SELECT p3.name FROM people p1")
    print("    JOIN knows k1 ON p1.id = k1.person_id")
    print("    JOIN knows k2 ON k1.friend_id = k2.person_id")
    print("    JOIN people p3 ON k2.friend_id = p3.id")
    print("    WHERE p1.name = 'Alice';")
    print("    (Multiple JOINs, performance degrades with depth)")
    print()
    print("  Graph: MATCH (a:Person {name:'Alice'})-[:KNOWS*2]->(fof)")
    print("    RETURN fof.name")
    print("    (Constant time per hop, regardless of graph size)")

    print()


# ============================================================
# 5. COMPARISON AND DECISION FRAMEWORK
# ============================================================

def demonstrate_comparison():
    """Compare all four NoSQL models with a decision framework."""
    print("=" * 60)
    print("5. COMPARISON: WHEN TO USE WHICH MODEL")
    print("=" * 60)
    print()

    comparison = [
        ("Dimension", "Key-Value", "Document", "Wide-Column", "Graph"),
        ("─" * 12, "─" * 12, "─" * 12, "─" * 12, "─" * 12),
        ("Query", "GET/PUT", "Rich JSON", "Scan/Get", "Traversal"),
        ("Schema", "None", "Flexible", "Semi-fixed", "Flexible"),
        ("Scalability", "Excellent", "Good", "Excellent", "Moderate"),
        ("Relationships", "None", "Embedded", "Denormalized", "Native"),
        ("Best for", "Cache/Session", "CMS/Catalog", "Time-series", "Social/Fraud"),
        ("Example DB", "Redis", "MongoDB", "Cassandra", "Neo4j"),
        ("CAP choice", "AP typical", "CP or AP", "AP typical", "CP typical"),
    ]

    for row in comparison:
        print(f"  {row[0]:14} {row[1]:14} {row[2]:14} {row[3]:14} {row[4]:14}")

    print()
    print("  Decision Framework:")
    print("  ┌─ Need sub-ms latency, simple lookups? ──── Key-Value")
    print("  ├─ Need flexible schemas, rich queries?  ──── Document")
    print("  ├─ Need time-series, wide sparse rows?   ──── Wide-Column")
    print("  ├─ Need relationship traversals?         ──── Graph")
    print("  └─ Need ACID + SQL + scale?              ──── NewSQL (Lesson 15)")

    print()
    print("  Polyglot Persistence: Use multiple models in one application")
    print("    e.g., Redis (cache) + MongoDB (catalog) + Neo4j (recommendations)")

    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          NoSQL DATA MODELS — In-Memory Simulations           ║
║  Key-Value, Document, Wide-Column, Graph                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_key_value_store()
    demonstrate_document_store()
    demonstrate_wide_column_store()
    demonstrate_graph_database()
    demonstrate_comparison()

    print("=" * 60)
    print("SUMMARY: NoSQL DATA MODELS")
    print("=" * 60)
    print("Key takeaways:")
    print("  1. Each model optimizes for different access patterns")
    print("  2. Key-Value: simplest, fastest for point lookups")
    print("  3. Document: flexible schemas, rich queries on nested data")
    print("  4. Wide-Column: sparse data, efficient range scans")
    print("  5. Graph: relationship-centric, constant-time traversals")
    print("  6. CAP theorem forces trade-offs in distributed settings")
    print("  7. Polyglot persistence: combine models for best results")
    print("=" * 60)
