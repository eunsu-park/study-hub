"""
Exercises for Lesson 09: Indexing
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers B+Tree calculations, bitmap indexes, index design strategies,
extendible hashing, and linear hashing.
"""

import math


# === Exercise 1: Dense vs Sparse Index ===
# Problem: Explain why secondary indexes must be dense.

def exercise_1():
    """Explain dense vs sparse index requirement."""
    print("Why must a secondary index be dense while a primary index can be sparse?")
    print()
    print("PRIMARY INDEX (on ordering field):")
    print("  - Data file is physically sorted by the indexed attribute.")
    print("  - Records with the same key value are clustered in consecutive blocks.")
    print("  - One index entry per BLOCK is sufficient (block anchor).")
    print("  - To find a record: binary search on index -> go to block -> scan within block.")
    print("  - This is a SPARSE index.")
    print()
    print("SECONDARY INDEX (on non-ordering field):")
    print("  - Data file is NOT sorted by the indexed attribute.")
    print("  - Records with the same key value can be scattered across many blocks.")
    print("  - One entry per BLOCK is useless -- the block doesn't tell you where")
    print("    a particular key value's records are.")
    print("  - Must have one entry per RECORD (or per unique key value + pointer list).")
    print("  - This is a DENSE index.")
    print()
    print("Analogy:")
    print("  - Sparse index = table of contents (works because chapters are in order)")
    print("  - Dense index = back-of-book index (needed because topics appear anywhere)")


# === Exercise 2: B+Tree Calculations ===
# Problem: Calculate B+Tree properties for a given table.

def exercise_2():
    """B+Tree calculations for a 500,000-record table."""
    n_records = 500_000
    block_size = 4096  # 4 KB
    record_size = 200  # bytes
    key_size = 8       # bytes
    pointer_size = 8   # bytes

    print(f"Given:")
    print(f"  Records: {n_records:,}")
    print(f"  Block size: {block_size:,} bytes ({block_size // 1024} KB)")
    print(f"  Record size: {record_size} bytes")
    print(f"  Key size: {key_size} bytes")
    print(f"  Pointer size: {pointer_size} bytes")
    print()

    # (a) Records per data block
    records_per_block = block_size // record_size
    print(f"(a) Records per data block: {block_size} / {record_size} = {records_per_block}")
    print()

    # (b) Number of data blocks
    n_data_blocks = math.ceil(n_records / records_per_block)
    print(f"(b) Data blocks: ceil({n_records:,} / {records_per_block}) = {n_data_blocks:,}")
    print()

    # (c) Maximum fan-out (order) of B+Tree
    # Internal node: (order-1) keys + order pointers
    # (order-1) * key_size + order * pointer_size <= block_size
    # order * (key_size + pointer_size) - key_size <= block_size
    # order <= (block_size + key_size) / (key_size + pointer_size)
    order = (block_size + key_size) // (key_size + pointer_size)
    print(f"(c) Maximum fan-out (order m):")
    print(f"    Internal node: (m-1) keys + m pointers")
    print(f"    (m-1) x {key_size} + m x {pointer_size} <= {block_size}")
    print(f"    m x ({key_size} + {pointer_size}) <= {block_size} + {key_size}")
    print(f"    m <= ({block_size} + {key_size}) / ({key_size} + {pointer_size})")
    print(f"    m <= {(block_size + key_size) / (key_size + pointer_size):.1f}")
    print(f"    m = {order}")
    print()

    # (d) Maximum height
    # With n keys, height h <= ceil(log_{ceil(m/2)}(n))
    min_fanout = math.ceil(order / 2)
    height = math.ceil(math.log(n_records) / math.log(min_fanout))
    print(f"(d) Maximum height:")
    print(f"    Minimum fan-out: ceil({order}/2) = {min_fanout}")
    print(f"    Height h <= ceil(log_{min_fanout}({n_records:,}))")
    print(f"    h <= ceil({math.log(n_records) / math.log(min_fanout):.2f})")
    print(f"    h = {height}")
    print()

    # (e) Disk I/Os: index search vs full table scan
    index_ios = height + 1  # traverse tree + 1 data block access
    scan_ios = n_data_blocks
    print(f"(e) Disk I/Os for equality search:")
    print(f"    Using B+Tree index: {height} (tree traversal) + 1 (data block) = {index_ios} I/Os")
    print(f"    Full table scan: {n_data_blocks:,} I/Os")
    print(f"    Speedup: {n_data_blocks / index_ios:.0f}x")


# === Exercise 4: Bitmap Index ===
# Problem: Calculate bitmap index properties.

def exercise_4():
    """Bitmap index calculations."""
    n_rows = 10_000_000
    n_colors = 8
    purple_pct = 0.001  # 0.1%

    print(f"Given: table with {n_rows:,} rows, 'color' column with {n_colors} distinct values")
    print()

    # (a) Total uncompressed size
    bits_per_bitmap = n_rows
    bytes_per_bitmap = math.ceil(bits_per_bitmap / 8)
    total_bytes = bytes_per_bitmap * n_colors
    total_mb = total_bytes / (1024 * 1024)

    print(f"(a) Uncompressed bitmap index size:")
    print(f"    Each bitmap: {n_rows:,} bits = {bytes_per_bitmap:,} bytes = {bytes_per_bitmap / (1024*1024):.1f} MB")
    print(f"    Total ({n_colors} colors): {total_bytes:,} bytes = {total_mb:.1f} MB")
    print()

    # (b) Run-length encoding compression for 'purple'
    n_purple = int(n_rows * purple_pct)
    n_zeros_between = int(1 / purple_pct) - 1  # average gap between 1s

    print(f"(b) Purple bitmap compression (RLE):")
    print(f"    Only {n_purple:,} out of {n_rows:,} rows are purple ({purple_pct*100}%)")
    print(f"    Average run of zeros between 1s: ~{n_zeros_between}")
    print(f"    Uncompressed: {bytes_per_bitmap:,} bytes ({bytes_per_bitmap / 1024:.0f} KB)")

    # Rough RLE estimate: each run encoded as (count, value)
    # With 0.1% density, about n_purple runs of 0s + n_purple single 1s
    # Each run needs ~4 bytes (count) + 1 bit
    rle_estimate = n_purple * (4 + 1)  # conservative estimate
    compression_ratio = bytes_per_bitmap / rle_estimate

    print(f"    RLE compressed estimate: ~{rle_estimate:,} bytes ({rle_estimate / 1024:.0f} KB)")
    print(f"    Compression ratio: ~{compression_ratio:.0f}x")
    print(f"    Very compressible! Sparse bitmaps (few 1s) compress extremely well.")
    print()

    # (c) Comparison with B+Tree
    print(f"(c) B+Tree vs Bitmap on 'color' column:")
    print(f"    B+Tree on color ({n_colors} distinct values, {n_rows:,} rows):")
    print(f"      - Low selectivity: each color matches ~{n_rows // n_colors:,} rows (12.5%)")
    print(f"      - Index scan returns many row pointers -> random I/O")
    print(f"      - Often WORSE than sequential scan for low-selectivity queries")
    print()
    print(f"    Bitmap on color:")
    print(f"      - Bitmap AND/OR operations are CPU-fast (bitwise ops)")
    print(f"      - Ideal for multi-predicate queries: color='red' AND size='L'")
    print(f"      - Count queries: popcount(bitmap) without touching data pages")
    print(f"      - Poor for high-cardinality columns or frequent updates")


# === Exercise 5: Index Strategy Design ===
# Problem: Design indexes for a query workload.

def exercise_5():
    """Design indexing strategy for a query workload."""
    queries = [
        {
            "query": "Q1 (90%): SELECT * FROM users WHERE email = ?",
            "index": "CREATE UNIQUE INDEX idx_users_email ON users(email);",
            "type": "B+Tree (hash index also viable)",
            "strategy": "Unique index since email is unique. Covers 90% of traffic."
        },
        {
            "query": "Q2 (5%): SELECT name, created_at FROM users WHERE created_at > ? ORDER BY created_at DESC LIMIT 20",
            "index": "CREATE INDEX idx_users_created_at ON users(created_at DESC) INCLUDE (name);",
            "type": "B+Tree, covering index",
            "strategy": "DESC order matches ORDER BY DESC. INCLUDE(name) makes it covering -- "
                        "no need to access heap for name column."
        },
        {
            "query": "Q3 (3%): SELECT * FROM users WHERE country = ? AND age BETWEEN ? AND ?",
            "index": "CREATE INDEX idx_users_country_age ON users(country, age);",
            "type": "B+Tree, composite index",
            "strategy": "country = ? (equality) first, then age BETWEEN (range) second. "
                        "This order is optimal: equality prefix narrows to one country, "
                        "then range scan within that partition."
        },
        {
            "query": "Q4 (2%): SELECT * FROM users WHERE name ILIKE '%smith%'",
            "index": "CREATE INDEX idx_users_name_trgm ON users USING gin(name gin_trgm_ops);",
            "type": "GIN with trigram extension (pg_trgm)",
            "strategy": "B+Tree cannot handle leading wildcard (%smith%). "
                        "GIN trigram index breaks text into 3-character grams and indexes them. "
                        "Supports LIKE/ILIKE with leading wildcards."
        }
    ]

    for q in queries:
        print(f"{q['query']}")
        print(f"  Index:    {q['index']}")
        print(f"  Type:     {q['type']}")
        print(f"  Strategy: {q['strategy']}")
        print()


# === Exercise 6: Extendible Hashing ===
# Problem: Insert into extendible hash directory.

def exercise_6():
    """Extendible hashing: insert and handle split."""
    print("Extendible Hash Directory (global depth = 2, bucket capacity = 2)")
    print()

    # Initial state
    directory = {
        "00": {"name": "A", "data": ["00110", "00010"], "local_depth": 2},
        "01": {"name": "B", "data": ["01100"], "local_depth": 2},
        "10": {"name": "C", "data": ["10001", "10110"], "local_depth": 2},
        "11": {"name": "D", "data": ["11000"], "local_depth": 2},
    }

    def print_directory(directory, global_depth):
        print(f"  Global depth: {global_depth}")
        for prefix in sorted(directory.keys()):
            bucket = directory[prefix]
            print(f"    {prefix} -> Bucket {bucket['name']} {bucket['data']} (local depth {bucket['local_depth']})")
        print()

    print("Initial state:")
    print_directory(directory, 2)

    # Insert h = 00001
    new_hash = "00001"
    prefix = new_hash[:2]  # "00"
    target_bucket = directory[prefix]

    print(f"Insert h = {new_hash}")
    print(f"  Prefix (2 bits): '{prefix}' -> Bucket {target_bucket['name']}")
    print(f"  Bucket {target_bucket['name']} has {len(target_bucket['data'])} items (capacity = 2)")
    print(f"  Bucket is FULL! Must split.")
    print()

    # Since local_depth == global_depth, must double directory
    print("  local_depth (2) == global_depth (2) -> DOUBLE the directory")
    print()

    # After doubling: global depth = 3
    # 000 -> was 00, 001 -> was 00 (split bucket A)
    # 010 -> was 01, 011 -> was 01
    # 100 -> was 10, 101 -> was 10
    # 110 -> was 11, 111 -> was 11

    # Split bucket A: redistribute based on 3rd bit
    all_items = target_bucket["data"] + [new_hash]
    bucket_a_new = [h for h in all_items if h[2] == '0']  # 3rd bit = 0
    bucket_e_new = [h for h in all_items if h[2] == '1']  # 3rd bit = 1

    print(f"  Items to redistribute: {all_items}")
    print(f"  Split on 3rd bit:")
    print(f"    3rd bit = 0: {bucket_a_new} -> Bucket A' (local depth 3)")
    print(f"    3rd bit = 1: {bucket_e_new} -> Bucket E  (local depth 3)")
    print()

    print("Final state (global depth = 3):")
    final_dir = {
        "000": {"name": "A'", "data": bucket_a_new, "local_depth": 3},
        "001": {"name": "E",  "data": bucket_e_new, "local_depth": 3},
        "010": {"name": "B",  "data": ["01100"], "local_depth": 2},
        "011": {"name": "B",  "data": ["01100"], "local_depth": 2},
        "100": {"name": "C",  "data": ["10001", "10110"], "local_depth": 2},
        "101": {"name": "C",  "data": ["10001", "10110"], "local_depth": 2},
        "110": {"name": "D",  "data": ["11000"], "local_depth": 2},
        "111": {"name": "D",  "data": ["11000"], "local_depth": 2},
    }
    print_directory(final_dir, 3)
    print("  Note: 010/011 share Bucket B, 100/101 share C, 110/111 share D (local depth 2 < global depth 3)")


# === Exercise 7: Linear Hashing ===
# Problem: Insert keys into a linear hashing scheme.

def exercise_7():
    """Linear hashing: insert keys step by step."""
    print("Linear Hashing: N=4 initial buckets, capacity=2")
    print("h0(K) = K mod 4, h1(K) = K mod 8")
    print("Split pointer s = 0")
    print()

    # State: buckets as lists, overflow as separate lists
    buckets = {
        0: [8, 16],   # full
        1: [5],
        2: [10],
        3: [7, 15],   # full
    }
    overflow = {}
    s = 0  # split pointer
    n = 4  # current number of buckets (before any expansion)

    def print_state(buckets, overflow, s, label=""):
        if label:
            print(f"  {label}")
        for b in sorted(buckets.keys()):
            ovf = overflow.get(b, [])
            ovf_str = f" -> overflow: {ovf}" if ovf else ""
            print(f"    Bucket {b}: {buckets[b]}{ovf_str}")
        print(f"    Split pointer: s={s}")
        print()

    print("Initial state:")
    print_state(buckets, overflow, s)

    def h(key, level):
        return key % (4 * (2 ** level))

    def insert_key(key, buckets, overflow, s, n):
        """Insert a key and handle splits."""
        # Determine target bucket
        bucket_idx = h(key, 0)  # h0(K) = K mod 4
        if bucket_idx < s:
            bucket_idx = h(key, 1)  # h1(K) = K mod 8

        print(f"  Insert key {key}:")
        print(f"    h0({key}) = {key} mod 4 = {key % 4}", end="")
        if key % 4 < s:
            print(f" (< s={s}, use h1)")
            print(f"    h1({key}) = {key} mod 8 = {key % 8}")
        else:
            print(f" (>= s={s}, use h0)")

        # Insert into target bucket
        if len(buckets.get(bucket_idx, [])) < 2:
            buckets.setdefault(bucket_idx, []).append(key)
            print(f"    Placed in Bucket {bucket_idx}")
        else:
            # Overflow
            overflow.setdefault(bucket_idx, []).append(key)
            print(f"    Bucket {bucket_idx} full! Added to overflow chain")

            # Split bucket s
            print(f"    Trigger split of Bucket {s}")
            old_items = buckets.pop(s, []) + overflow.pop(s, [])
            new_bucket_idx = s + n
            buckets[s] = []
            buckets[new_bucket_idx] = []

            for item in old_items:
                new_idx = h(item, 1)  # h1 = K mod 8
                if len(buckets.get(new_idx, [])) < 2:
                    buckets.setdefault(new_idx, []).append(item)
                else:
                    overflow.setdefault(new_idx, []).append(item)

            print(f"    Bucket {s} items {old_items} redistributed using h1:")
            for item in old_items:
                print(f"      h1({item}) = {item} mod 8 = {h(item, 1)}")

            s += 1
            if s >= n:
                s = 0
                n *= 2

        return s, n

    # Insert 12
    s, n = insert_key(12, buckets, overflow, s, n)
    print_state(buckets, overflow, s)

    # Insert 9
    s, n = insert_key(9, buckets, overflow, s, n)
    print_state(buckets, overflow, s)

    # Insert 3
    s, n = insert_key(3, buckets, overflow, s, n)
    print_state(buckets, overflow, s)


# === Exercise 9: Partial Index Recommendation ===
# Problem: Index strategy for status='pending' COUNT query.

def exercise_9():
    """Index recommendation for COUNT query optimization."""
    n_rows = 50_000_000
    n_statuses = 5

    print(f"Query: SELECT COUNT(*) FROM orders WHERE status = 'pending'")
    print(f"Table: orders ({n_rows:,} rows), status has {n_statuses} distinct values")
    print(f"Current time: 2 seconds")
    print()

    # Option 1: Full B+Tree on status
    rows_per_status = n_rows // n_statuses
    print("Option 1: B+Tree index on status")
    print(f"  Selectivity: 1/{n_statuses} = {1/n_statuses:.1%}")
    print(f"  Index scan returns ~{rows_per_status:,} row pointers")
    print(f"  Problem: Low selectivity -> may be WORSE than seq scan due to random I/O")
    print()

    # Option 2: Partial index
    print("Option 2: Partial index (PostgreSQL)")
    print("  CREATE INDEX idx_orders_pending ON orders(status)")
    print("  WHERE status = 'pending';")
    print()
    pending_pct = 0.10  # assume 10% are pending
    n_pending = int(n_rows * pending_pct)
    print(f"  Assuming ~{pending_pct:.0%} of orders are 'pending': {n_pending:,} rows")
    print(f"  Index size: much smaller (only {pending_pct:.0%} of data)")
    print(f"  For COUNT(*): index-only scan -- just count leaf entries")
    print(f"  Estimated time: < 100ms (index-only scan of small index)")
    print()

    # Option 3: Covering index for index-only scan
    print("Option 3: Covering index for index-only scan")
    print("  CREATE INDEX idx_orders_status ON orders(status) INCLUDE (id);")
    print("  COUNT(*) can be answered from index alone.")
    print()

    print("Recommendation: Option 2 (partial index)")
    print(f"  Improvement: ~{2.0 / 0.1:.0f}x speedup (2s -> ~100ms)")
    print("  Also useful if 'pending' orders are frequently queried and represent a small fraction.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Dense vs Sparse Index ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: B+Tree Calculations ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 4: Bitmap Index ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Index Strategy Design ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Extendible Hashing ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 7: Linear Hashing ===")
    print("=" * 70)
    exercise_7()

    print("=" * 70)
    print("=== Exercise 9: Partial Index Recommendation ===")
    print("=" * 70)
    exercise_9()

    print("\nAll exercises completed!")
