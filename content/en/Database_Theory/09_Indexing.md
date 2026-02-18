# 09. Indexing

**Previous**: [Query Processing and Optimization](./08_Query_Processing.md) | **Next**: [Transaction Theory](./10_Transaction_Theory.md)

---

## Learning Objectives

- Understand why indexing is critical for database performance
- Distinguish between ordered indices: primary, clustering, and secondary
- Master B-Tree and B+Tree structures, operations, and complexity
- Learn hash-based indexing techniques: static, extendible, and linear hashing
- Understand bitmap indices and multi-dimensional index structures
- Apply index design guidelines in practice

---

## 1. Why Indexing Matters

### The Fundamental Problem

Consider a table `employees` with 1,000,000 rows stored on disk. Each disk block holds 10 rows, so the table occupies 100,000 blocks. To find a single employee by ID:

```
Without index:  Scan all 100,000 blocks  →  O(n) disk I/Os
With index:     Follow index pointers     →  O(log n) disk I/Os
```

For 100,000 blocks:
- **Sequential scan**: up to 100,000 block reads
- **B+Tree index** (branching factor 200): approximately log_200(1,000,000) ≈ 3 block reads

This is a **33,000x improvement** in I/O cost.

### Sequential vs. Indexed Access

```
Sequential Access (Table Scan):
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Blk1 │ Blk2 │ Blk3 │ Blk4 │ Blk5 │ ...  │ BlkN │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  ↑ Read every block sequentially

Indexed Access:
┌─────────────────┐
│   Index Root     │  ← 1 I/O
│  [50│100│150]    │
└──┬──────┬──────┬─┘
   ↓      ↓      ↓
 ┌────┐ ┌────┐ ┌────┐
 │Leaf│ │Leaf│ │Leaf│  ← 1 I/O (follow pointer)
 └──┬─┘ └────┘ └────┘
    ↓
 ┌──────┐
 │ Data │  ← 1 I/O (fetch actual record)
 │ Block│
 └──────┘
```

### When Not to Use an Index

Indexes are not always beneficial:

- **Small tables**: A sequential scan of a few blocks is faster than index traversal overhead
- **High selectivity queries**: If a query returns most rows (e.g., >15-20% of the table), a full scan is cheaper
- **Frequent writes**: Every INSERT, UPDATE, or DELETE must also update the index
- **Low-cardinality columns**: Indexing a boolean column with a B+Tree is wasteful (bitmap index may be appropriate)

### Cost Model Basics

The cost of disk access dominates query performance. We measure in terms of **block transfers** (reading/writing a disk block) and **seeks** (moving the disk head).

Let:
- `b` = number of blocks in the file
- `n` = number of records
- `f` = blocking factor (records per block), so `b = ⌈n/f⌉`

| Access Method | Average Cost (equality search) |
|---|---|
| Linear scan (unsorted) | b/2 block transfers |
| Linear scan (sorted) | ⌈log₂(b)⌉ (binary search) |
| Primary index (sorted) | ⌈log₂(b_i)⌉ + 1, where b_i = index blocks |
| B+Tree | ⌈log_p(b)⌉ + 1, where p = branching factor |
| Hash index | 1 (ideal) to 2 (with overflow) |

---

## 2. Ordered Indices

### Index Structure Fundamentals

An **index** is a data structure that maps **search key values** to **pointers** (record identifiers or block addresses). The search key need not be a primary key -- any attribute or set of attributes can serve as a search key.

```
Index Entry Format:
┌─────────────┬─────────────┐
│ Search Key   │ Pointer(s)  │
│ Value        │ to Records  │
└─────────────┴─────────────┘
```

### 2.1 Primary Index

A **primary index** is built on the **ordering key** of a sequentially ordered file. The file is physically sorted on the indexed attribute.

```
Primary Index on Employee ID (file sorted by ID):

Index:                         Data File:
┌─────┬──────┐                 ┌───────────────────┐
│ 100 │  ──────────────────→  │ 100 │ Alice │ ... │ Block 1
├─────┼──────┤                 │ 105 │ Bob   │ ... │
│ 200 │  ──────┐               ├───────────────────┤
├─────┼──────┤  └───────────→  │ 200 │ Carol │ ... │ Block 2
│ 300 │  ──────┐               │ 210 │ Dave  │ ... │
└─────┴──────┘  │              ├───────────────────┤
                 └───────────→ │ 300 │ Eve   │ ... │ Block 3
                               │ 350 │ Frank │ ... │
                               └───────────────────┘
```

**Properties:**
- One index entry per **block** (not per record) -- this is a sparse index
- The search key is the same attribute the file is sorted on
- Very efficient for equality and range queries
- Only **one** primary index can exist per table (since the file can only be sorted one way)

### 2.2 Clustering Index

A **clustering index** is built on a non-key attribute that the file is physically ordered on. Multiple records may share the same search key value.

```
Clustering Index on Department:

Index:                         Data File (sorted by dept):
┌──────────┬──────┐            ┌───────────────────────────┐
│ Acctg    │  ──────────────→ │ Acctg  │ Alice │ 60000   │
├──────────┼──────┤            │ Acctg  │ Bob   │ 55000   │
│ Eng      │  ──────┐          ├───────────────────────────┤
├──────────┼──────┤  └──────→ │ Eng    │ Carol │ 80000   │
│ Sales    │  ──────┐          │ Eng    │ Dave  │ 75000   │
└──────────┴──────┘  │         │ Eng    │ Eve   │ 72000   │
                      │        ├───────────────────────────┤
                      └─────→  │ Sales  │ Frank │ 50000   │
                               │ Sales  │ Grace │ 52000   │
                               └───────────────────────────┘
```

**Properties:**
- One index entry per **distinct value** of the search key
- Each pointer leads to the first block containing that value
- Efficient for grouping and aggregation queries

### 2.3 Secondary Index

A **secondary index** provides an alternative access path on a non-ordering attribute. The file is **not** sorted on this attribute.

```
Secondary Index on Salary:

Index (sorted by salary):      Data File (sorted by ID, NOT salary):
┌───────┬──────┐               ┌───────────────────────────┐
│ 50000 │  ──────────────────→ │ 100 │ Alice │ 60000     │ Block 1
├───────┼──────┤               │ 105 │ Bob   │ 50000     │
│ 55000 │  ──────┐              ├───────────────────────────┤
├───────┼──────┤  │            │ 200 │ Carol │ 80000     │ Block 2
│ 60000 │  ──────┐│            │ 210 │ Dave  │ 55000     │
├───────┼──────┤  ││           ├───────────────────────────┤
│ 72000 │  ──┐    ││           │ 300 │ Eve   │ 72000     │ Block 3
│ ...   │    │    ││           │ 350 │ Frank │ 75000     │
└───────┴────┘    ││           └───────────────────────────┘
                  │││  (pointers cross block boundaries)
```

**Properties:**
- Must be a **dense index** (one entry per record, not per block) since data is not sorted by the search key
- Multiple secondary indices can exist on the same table
- Less efficient for range queries (random I/O pattern due to non-sequential data)
- An extra level of indirection (bucket of pointers) is sometimes used for duplicate key values

### Dense vs. Sparse Indices

**Dense Index**: One index entry for **every record** in the data file.

```
Dense Index:
┌─────┬───┐     ┌──────────────┐
│ 100 │ ──┼───→ │ 100 │ Alice  │
├─────┼───┤     │ 105 │ Bob    │ ←── entry for 105 exists
│ 105 │ ──┼───→ │              │
├─────┼───┤     ├──────────────┤
│ 200 │ ──┼───→ │ 200 │ Carol  │
├─────┼───┤     │ 210 │ Dave   │
│ 210 │ ──┼───→ │              │
└─────┴───┘     └──────────────┘
```

**Sparse Index**: One index entry for **some records** (typically one per block).

```
Sparse Index:
┌─────┬───┐     ┌──────────────┐
│ 100 │ ──┼───→ │ 100 │ Alice  │  Block 1
│     │   │     │ 105 │ Bob    │  (no entry for 105)
├─────┼───┤     ├──────────────┤
│ 200 │ ──┼───→ │ 200 │ Carol  │  Block 2
│     │   │     │ 210 │ Dave   │  (no entry for 210)
└─────┴───┘     └──────────────┘
```

**Comparison:**

| Aspect | Dense Index | Sparse Index |
|---|---|---|
| Space | Larger (entry per record) | Smaller (entry per block) |
| Search | Direct lookup possible | May need to scan within block |
| Requirement | Works on unsorted files | Requires sorted data file |
| Maintenance | More updates on insert/delete | Fewer updates |
| Use case | Secondary indices | Primary indices |

### Multi-Level Indices

When the index itself is too large to fit in memory, we can build an **index on the index**:

```
Level 2 (outer index):
┌──────┬───┐
│  1   │ ──┼───→ ┌──────┬───┐
│ 500  │ ──┼─┐   │  1   │ ──┼───→ Data Block 1
└──────┴───┘ │   │ 100  │ ──┼───→ Data Block 2
              │   │ 200  │ ──┼───→ Data Block 3
              │   │ ...  │   │
              │   └──────┴───┘
              │   Level 1 (inner index, part 1)
              │
              └→ ┌──────┬───┐
                 │ 500  │ ──┼───→ Data Block K
                 │ 600  │ ──┼───→ Data Block K+1
                 │ ...  │   │
                 └──────┴───┘
                 Level 1 (inner index, part 2)
```

This naturally leads us to tree-structured indices, specifically B-Trees and B+Trees.

---

## 3. B-Tree

### 3.1 Structure

A **B-Tree** of order `m` (also called a B-Tree of degree `m`) satisfies:

1. Every node has at most `m` children
2. Every internal node (except root) has at least `⌈m/2⌉` children
3. The root has at least 2 children (if it is not a leaf)
4. All leaves appear at the same level
5. A non-leaf node with `k` children contains `k-1` keys

```
B-Tree of order 4 (each node holds up to 3 keys):

                    ┌─────────┐
                    │   30     │
                    └──┬───┬──┘
                  ┌────┘   └────┐
           ┌──────┴──┐    ┌────┴──────┐
           │ 10 │ 20 │    │ 40 │ 50   │
           └┬───┬───┬┘    └┬───┬────┬─┘
            ↓   ↓   ↓      ↓   ↓    ↓
          ┌──┐┌──┐┌──┐  ┌──┐┌──┐ ┌───┐
          │5 ││15││25│  │35││45│ │55  │
          │8 ││  ││  │  │  ││  │ │60  │
          └──┘└──┘└──┘  └──┘└──┘ └───┘

Note: In a B-Tree (unlike B+Tree), data pointers
exist at EVERY node, not just leaves.
```

**Node structure** (internal node with `n` keys):

```
┌────┬─────┬────┬─────┬────┬─────┬────┐
│ P₁ │ K₁  │ P₂ │ K₂  │ P₃ │ ... │ Pₙ₊₁│
│    │ D₁  │    │ D₂  │    │     │    │
└────┴─────┴────┴─────┴────┴─────┴────┘
  ↓          ↓          ↓            ↓
subtree    subtree    subtree      subtree
< K₁     [K₁,K₂)   [K₂,K₃)      ≥ Kₙ

Where:
  Pᵢ = pointer to child node (or data block for leaf)
  Kᵢ = search key value
  Dᵢ = pointer to the data record with key Kᵢ
```

### 3.2 Search

Searching a B-Tree for key `K`:

```
BTREE-SEARCH(node, K):
    i = 1
    while i ≤ node.n and K > node.key[i]:
        i = i + 1

    if i ≤ node.n and K == node.key[i]:
        return (node, i)           // Found at this node

    if node is a leaf:
        return NOT_FOUND

    // Read child from disk
    DISK-READ(node.child[i])
    return BTREE-SEARCH(node.child[i], K)
```

**Complexity**: O(log_m n) disk I/Os, where n is the number of keys and m is the order.

### 3.3 Insertion

Insert key `K` into a B-Tree of order `m`:

1. **Find** the appropriate leaf node using search
2. **Insert** the key in sorted order within the leaf
3. If the node overflows (has `m` keys):
   - **Split** the node into two nodes at the median key
   - **Promote** the median key to the parent
   - Recursively handle parent overflow if needed

**Example: Insert 25 into a B-Tree of order 3 (max 2 keys per node):**

```
Before:
        ┌────┐
        │ 20 │
        └─┬──┘
     ┌────┴─────┐
  ┌─────┐   ┌──────┐
  │10│15│   │22│30 │  ← Node is full
  └─────┘   └──────┘

Step 1: Key 25 belongs in the right leaf [22, 30].
Step 2: Insert → [22, 25, 30] — overflow! (3 keys > max 2)
Step 3: Split at median (25). Promote 25 to parent.

After:
        ┌──────┐
        │20│25 │
        └┬──┬──┘
    ┌────┘  │  └────┐
 ┌─────┐ ┌──┐  ┌──┐
 │10│15│ │22│  │30│
 └─────┘ └──┘  └──┘
```

### 3.4 Deletion

Delete key `K` from a B-Tree of order `m`:

1. **Find** the node containing `K`
2. If `K` is in a **leaf**: remove it directly
3. If `K` is in an **internal node**: replace with predecessor (or successor) from leaf, then delete from leaf
4. If the node **underflows** (fewer than `⌈m/2⌉ - 1` keys):
   - **Redistribute** (borrow) from a sibling if possible
   - Otherwise, **merge** with a sibling and pull down a key from the parent
   - Recursively handle parent underflow

**Example: Delete 20 from above tree (order 3, min 1 key per non-root node):**

```
Before:
        ┌──────┐
        │20│25 │
        └┬──┬──┘
    ┌────┘  │  └────┐
 ┌─────┐ ┌──┐  ┌──┐
 │10│15│ │22│  │30│
 └─────┘ └──┘  └──┘

Step 1: Key 20 is in an internal node.
Step 2: Replace with in-order predecessor (15).
Step 3: Delete 15 from the leaf.

After replacement:
        ┌──────┐
        │15│25 │
        └┬──┬──┘
    ┌────┘  │  └────┐
 ┌──┐   ┌──┐   ┌──┐
 │10│   │22│   │30│
 └──┘   └──┘   └──┘
```

### 3.5 Complexity Analysis

For a B-Tree of order `m` with `n` keys:

| Operation | Disk I/Os | CPU Time |
|---|---|---|
| Search | O(log_m n) | O(m · log_m n) |
| Insert | O(log_m n) | O(m · log_m n) |
| Delete | O(log_m n) | O(m · log_m n) |

**Height bound**: For `n ≥ 1` keys and minimum degree `t = ⌈m/2⌉`:

```
h ≤ log_t((n+1)/2)
```

With a typical branching factor of 100-200, a tree of height 3-4 can index **billions** of records.

---

## 4. B+Tree

### 4.1 Structure and Properties

The **B+Tree** is the most widely used index structure in relational databases (PostgreSQL, MySQL InnoDB, Oracle, SQL Server all use B+Trees). It differs from the B-Tree in crucial ways:

1. **All data pointers are in leaf nodes** -- internal nodes only store keys and child pointers
2. **Leaf nodes are linked** in a doubly-linked list for efficient range scans
3. **Keys in internal nodes are duplicated** in leaves (internal keys serve only as guides)

```
B+Tree Structure:

Internal Nodes (guide keys only):
                    ┌────────────┐
                    │  30  │  60  │
                    └──┬───┬──┬──┘
              ┌────────┘   │  └────────┐
              ↓            ↓           ↓
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ 10 │ 20 │  │ 40 │ 50 │  │ 70 │ 80 │
        └─┬──┬──┬─┘  └─┬──┬──┬─┘  └─┬──┬──┬─┘
          ↓  ↓  ↓      ↓  ↓  ↓      ↓  ↓  ↓

Leaf Nodes (actual data, linked):
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ 5,8 │↔│10,15│↔│20,25│↔│30,35│↔│40,45│↔│50,55│↔│60,65│↔│70,75│↔│80,85│
│     │  │     │  │     │  │     │  │     │  │     │  │     │  │     │  │     │
│ D*  │  │ D*  │  │ D*  │  │ D*  │  │ D*  │  │ D*  │  │ D*  │  │ D*  │  │ D*  │
└─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
  ↑                                                                          ↑
Head                                              Leaf chain (doubly linked) Tail

D* = pointer to actual data record
```

### Node Format

**Internal node** (order `m`, at most `m` pointers, `m-1` keys):

```
┌────┬─────┬────┬─────┬────┬─────┬────┐
│ P₁ │ K₁  │ P₂ │ K₂  │ P₃ │ ... │ Pₘ│
└────┴─────┴────┴─────┴────┴─────┴────┘
All keys in subtree(P₁) < K₁
K₁ ≤ all keys in subtree(P₂) < K₂
...
Kₘ₋₁ ≤ all keys in subtree(Pₘ)
```

**Leaf node** (at most `m-1` key-pointer pairs, plus sibling pointer):

```
┌──────┬──────┬──────┬──────┬─────────┐
│K₁,D₁│K₂,D₂│K₃,D₃│ ...  │ Pₙₑₓₜ  │
└──────┴──────┴──────┴──────┴─────────┘
Dᵢ = pointer to record with key Kᵢ
Pₙₑₓₜ = pointer to next leaf node
```

### 4.2 B+Tree vs. B-Tree Comparison

| Feature | B-Tree | B+Tree |
|---|---|---|
| Data pointers | At every node | Only at leaf nodes |
| Key duplication | No duplicates | Internal keys duplicated in leaves |
| Leaf linking | Not linked | Doubly linked list |
| Range queries | Must traverse tree multiple times | Follow leaf chain |
| Fan-out | Lower (data pointers take space) | Higher (internal nodes are slimmer) |
| Equality search | May terminate early (at internal node) | Always goes to leaf |
| Space usage | Slightly less total | Slightly more (key duplication) |
| Practical use | Rarely used in databases | Standard in all major RDBMS |

### 4.3 Search in B+Tree

**Equality search** for key `K`:

```
BPLUS-SEARCH(root, K):
    node = root
    while node is not a leaf:
        i = smallest index such that K < node.key[i]
        if no such i exists:
            node = node.child[last]   // rightmost child
        else:
            node = node.child[i]      // go to child i

    // Now at leaf: scan for K
    for each (key, pointer) in node:
        if key == K:
            return pointer
    return NOT_FOUND
```

**Range search** for keys in `[lo, hi]`:

```
BPLUS-RANGE-SEARCH(root, lo, hi):
    // Step 1: Find the leaf containing lo
    leaf = find leaf for lo (using equality search logic)

    // Step 2: Scan leaves via linked list
    results = []
    while leaf is not null:
        for each (key, pointer) in leaf:
            if key > hi:
                return results
            if key >= lo:
                results.append(pointer)
        leaf = leaf.next    // Follow leaf chain!
    return results
```

This is why B+Trees excel at range queries -- once you find the starting leaf, you simply follow the linked list.

### 4.4 Insertion in B+Tree

Insert key `K` with data pointer `D`:

```
BPLUS-INSERT(root, K, D):
    leaf = find appropriate leaf node

    if leaf has room:
        insert (K, D) in sorted order in leaf
    else:
        // Leaf overflow: split
        Create new leaf L'
        Distribute entries evenly: first ⌈m/2⌉ stay, rest go to L'

        // Copy up: the first key of L' goes to parent
        parent-key = L'.key[1]
        INSERT-IN-PARENT(leaf, parent-key, L')

    Update leaf linked list pointers

INSERT-IN-PARENT(left, key, right):
    if parent has room:
        insert (key, right pointer) in parent
    else:
        // Internal node overflow: split
        Create new internal node N'
        Distribute: first ⌈m/2⌉ pointers stay, rest go to N'

        // Push up: the middle key goes to parent (NOT copied)
        middle-key = the key that separates the two halves
        INSERT-IN-PARENT(parent, middle-key, N')
```

**Important distinction:**
- Leaf split: **copy up** (the separator key exists in both the leaf and the parent)
- Internal split: **push up** (the separator key moves to the parent and is removed from the splitting node)

**Example: Insert 7 into a B+Tree of order 3 (max 2 keys per leaf):**

```
Before:
           ┌────┐
           │ 5  │
           └┬───┘
       ┌────┘  └────┐
    ┌─────┐     ┌──────┐
    │ 3,4 │ ──→ │ 5,6  │
    └─────┘     └──────┘

Step 1: Key 7 belongs in leaf [5, 6]. Leaf is full.
Step 2: Split leaf: [5] and [6, 7]. Copy up key 6 to parent.

After:
           ┌──────┐
           │ 5 │ 6│
           └┬──┬──┘
       ┌───┘  │  └───┐
    ┌─────┐ ┌───┐ ┌─────┐
    │ 3,4 │→│ 5 │→│ 6,7 │
    └─────┘ └───┘ └─────┘
```

### 4.5 Deletion in B+Tree

Delete key `K`:

```
BPLUS-DELETE(root, K):
    leaf = find leaf containing K
    Remove K from leaf

    if leaf has enough keys (≥ ⌈(m-1)/2⌉):
        done (may need to update parent key if first key changed)
    else:
        // Underflow
        if sibling has extra keys:
            redistribute (borrow from sibling)
            update parent key
        else:
            merge with sibling
            delete entry from parent
            recursively handle parent underflow
```

**Leaf merge** removes a key from the parent. **Internal node merge** pulls down a key from the parent. When the root is left with a single child, the child becomes the new root (tree height decreases).

### 4.6 Bulk Loading

Building a B+Tree one insertion at a time from an empty tree is inefficient. **Bulk loading** is used when creating an index on existing data:

```
BULK-LOAD(sorted_data, m):
    // Step 1: Sort data by search key (external sort if needed)

    // Step 2: Fill leaf nodes sequentially
    for each key in sorted_data:
        add to current leaf
        if leaf full:
            start new leaf
            add separator to parent level

    // Step 3: Build internal levels bottom-up
    repeat for each level until root is created
```

**Advantages of bulk loading:**
- Sequential I/O instead of random I/O
- Fill factor can be controlled (e.g., 90% full for future inserts)
- O(n log_m n) total, but with much better constants

**PostgreSQL:**
```sql
-- CREATE INDEX uses bulk loading internally when table has data
CREATE INDEX idx_emp_salary ON employees(salary);

-- REINDEX rebuilds an index (useful after heavy updates)
REINDEX INDEX idx_emp_salary;
```

### 4.7 Height and Performance Analysis

For a B+Tree of order `m` with `n` search key values:

**Maximum height:**
```
h ≤ ⌈log_{⌈m/2⌉}(n)⌉
```

**Practical example:**

Given:
- Block size: 4 KB
- Key size: 8 bytes, pointer size: 8 bytes
- Internal node fan-out: m = 4096 / (8 + 8) = 256
- Leaf entries: (m-1) = 255 per leaf

For n = 100,000,000 (100 million) records:

```
Height = ⌈log₁₂₈(100,000,000)⌉ = ⌈log₁₂₈(10⁸)⌉ ≈ ⌈8/2.1⌉ ≈ 4
```

So **4 disk reads** suffice to find any record among 100 million. In practice, the root and top levels are cached in memory, reducing this to **1-2 disk reads**.

---

## 5. Hash-Based Indexing

### 5.1 Static Hashing

A **hash index** uses a hash function `h(K)` to map search key values directly to bucket addresses.

```
Hash Function h maps key K to bucket number:

Key: "Alice" → h("Alice") = 2
Key: "Bob"   → h("Bob")   = 0
Key: "Carol" → h("Carol") = 2  (collision!)

Bucket Array:
┌─────────┐
│ Bucket 0 │ → [Bob, ...]
├─────────┤
│ Bucket 1 │ → [empty]
├─────────┤
│ Bucket 2 │ → [Alice, ...] → [Carol, ...]  (overflow chain)
├─────────┤
│ Bucket 3 │ → [Dave, ...]
└─────────┘
```

**Properties:**
- **Ideal case**: O(1) lookup -- one disk read
- **Overflow handling**: Chaining (linked list of overflow buckets)
- **Weakness**: Fixed number of buckets. Performance degrades as file grows

**Hash function requirements:**
1. **Uniform distribution**: Keys should be spread evenly across buckets
2. **Deterministic**: Same key always maps to the same bucket
3. **Fast computation**: O(1) time

### 5.2 Extendible Hashing

**Extendible hashing** adapts to data growth by using a **directory** that doubles in size as needed, without reorganizing the entire file.

**Key concepts:**
- **Global depth** (`d`): Number of bits of the hash value used by the directory
- **Local depth** (`dᵢ`): Number of bits used by bucket `i`
- Directory size = `2^d` entries

```
Example with global depth d = 2:

Hash values (binary):        Directory (d=2):
h(K₁) = 01001...            ┌────┬───────────┐
h(K₂) = 10110...            │ 00 │ ──→ Bucket A (local depth 2)
h(K₃) = 00101...            │ 01 │ ──→ Bucket B (local depth 2)
h(K₄) = 11010...            │ 10 │ ──→ Bucket C (local depth 1)
h(K₅) = 10011...            │ 11 │ ──→ Bucket C (local depth 1)
                             └────┴───────────┘

Note: Entries 10 and 11 both point to Bucket C
because C's local depth (1) < global depth (2).
Only the first bit matters for Bucket C.
```

**Bucket split when bucket overflows:**

```
If Bucket B (local depth 2) overflows:

Case 1: local depth < global depth
  → Split bucket, increase local depth
  → Redistribute entries
  → Update directory pointers

Case 2: local depth == global depth
  → Double the directory (global depth increases by 1)
  → Split bucket, increase local depth
  → Redistribute entries
  → Update directory pointers
```

**Advantages:**
- No overflow chains (splits handle growth)
- At most 2 disk accesses for any lookup (1 for directory, 1 for bucket)
- Grows gracefully

**Disadvantages:**
- Directory can become large if data is skewed
- Directory doubling is expensive (though infrequent)

### 5.3 Linear Hashing

**Linear hashing** avoids the directory entirely. Buckets are split in a **round-robin** fashion, one at a time, controlled by a split pointer.

**Key idea:**
- Maintain a **split pointer** `s` that tracks the next bucket to split
- Use two hash functions: `h₀(K) = K mod N` and `h₁(K) = K mod 2N`
- When a bucket overflows, split the bucket at the split pointer (not necessarily the overflowing bucket)

```
State with N=4 buckets, split pointer s=1:

Buckets:
┌──────────┐
│ Bucket 0 │ → [records with h₁(K)=0]    (already split, uses h₁)
├──────────┤
│ Bucket 1 │ → [records with h₀(K)=1]    ← split pointer s
├──────────┤
│ Bucket 2 │ → [records with h₀(K)=2]    (not yet split, uses h₀)
├──────────┤
│ Bucket 3 │ → [records with h₀(K)=3]    (not yet split, uses h₀)
├──────────┤
│ Bucket 4 │ → [records with h₁(K)=4]    (created by splitting Bucket 0)
└──────────┘

Lookup algorithm:
  b = h₀(K)          // e.g., K mod 4
  if b < s:           // bucket already split
      b = h₁(K)      // use K mod 8 instead
  search bucket b
```

**When to split:**
- Split is triggered when any bucket overflows
- The bucket at the split pointer is split (not the overflowing bucket!)
- The overflowing bucket uses overflow chains temporarily
- Split pointer advances: `s = s + 1`
- When `s = N`, a round is complete: `N = 2N`, `s = 0`, advance hash functions

**Advantages:**
- No directory needed
- Smooth growth (one bucket at a time)
- Guaranteed O(1) average access

**Disadvantages:**
- Temporary overflow chains
- Splitting may not relieve the actual overflowing bucket immediately

### Hash vs. B+Tree Comparison

| Criteria | Hash Index | B+Tree Index |
|---|---|---|
| Equality query | O(1) -- excellent | O(log n) -- good |
| Range query | Not supported | Excellent (leaf chain) |
| Ordered traversal | Not supported | Natural (leaf scan) |
| Dynamic growth | Extendible/linear | Splits and merges |
| Space usage | Can waste space (empty buckets) | Good utilization (~67%) |
| Implementation | Simpler | More complex |
| Common in RDBMS | PostgreSQL (hash), Redis | All major RDBMS |

---

## 6. Bitmap Index

### 6.1 Concept

A **bitmap index** creates a bit vector for each distinct value of an indexed attribute. Each bit corresponds to a row in the table.

```
Table: employees (8 rows)
┌─────┬───────┬────────┬────────┐
│ RID │ Name  │ Dept   │ Gender │
├─────┼───────┼────────┼────────┤
│  0  │ Alice │ Eng    │ F      │
│  1  │ Bob   │ Sales  │ M      │
│  2  │ Carol │ Eng    │ F      │
│  3  │ Dave  │ HR     │ M      │
│  4  │ Eve   │ Eng    │ F      │
│  5  │ Frank │ Sales  │ M      │
│  6  │ Grace │ HR     │ F      │
│  7  │ Hank  │ Eng    │ M      │
└─────┴───────┴────────┴────────┘

Bitmap Index on Dept:
  Eng:   [1, 0, 1, 0, 1, 0, 0, 1]  (rows 0,2,4,7)
  Sales: [0, 1, 0, 0, 0, 1, 0, 0]  (rows 1,5)
  HR:    [0, 0, 0, 1, 0, 0, 1, 0]  (rows 3,6)

Bitmap Index on Gender:
  F:     [1, 0, 1, 0, 1, 0, 1, 0]  (rows 0,2,4,6)
  M:     [0, 1, 0, 1, 0, 1, 0, 1]  (rows 1,3,5,7)
```

### 6.2 Bitmap Operations

Queries can be answered by combining bitmaps using **bitwise operations**:

```
Query: Find all female engineers

  Dept=Eng:  [1, 0, 1, 0, 1, 0, 0, 1]
  AND
  Gender=F:  [1, 0, 1, 0, 1, 0, 1, 0]
  ─────────────────────────────────────
  Result:    [1, 0, 1, 0, 1, 0, 0, 0]  → rows 0, 2, 4
                                         (Alice, Carol, Eve)
```

```
Query: Find all employees in Sales OR HR

  Dept=Sales: [0, 1, 0, 0, 0, 1, 0, 0]
  OR
  Dept=HR:    [0, 0, 0, 1, 0, 0, 1, 0]
  ─────────────────────────────────────
  Result:     [0, 1, 0, 1, 0, 1, 1, 0]  → rows 1, 3, 5, 6
```

```
Query: Find all non-engineers

  NOT Dept=Eng: NOT [1, 0, 1, 0, 1, 0, 0, 1]
  ─────────────────────────────────────────────
  Result:           [0, 1, 0, 1, 0, 1, 1, 0]  → rows 1, 3, 5, 6
```

### 6.3 When to Use Bitmap Indices

**Ideal for:**
- **Low cardinality** attributes (Gender: 2 values, Status: 3-5 values)
- **Data warehousing** with mostly read-only access
- **Complex multi-attribute queries** (AND/OR combinations)
- **Counting queries** (COUNT with WHERE) -- just count set bits
- **Star schema** fact tables with many dimension foreign keys

**Not ideal for:**
- **High cardinality** attributes (e.g., unique IDs -- one bitmap per value!)
- **OLTP** with frequent updates (updating a bitmap on every insert is expensive)
- **Highly concurrent** environments (bitmap locks affect many rows)

### 6.4 Bitmap Compression

For large tables, bitmaps can become very large. Compression techniques help:

**Run-Length Encoding (RLE):**
```
Original:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
RLE:       (0, 11), (1, 1), (0, 4)    → "11 zeros, 1 one, 4 zeros"
```

**Word-Aligned Hybrid (WAH):**
Used in Oracle and other systems. Compresses sequences of identical words (32 or 64 bits) while still allowing efficient bitwise operations on the compressed form.

**Roaring Bitmaps:**
A modern compression technique that partitions the bitmap into chunks and uses the optimal representation (array, bitmap, or run) for each chunk. Widely used in Apache Lucene, Apache Spark, and other systems.

### 6.5 Space Analysis

For a table with `n` rows and an attribute with `c` distinct values:

```
Uncompressed bitmap size = n × c bits = nc/8 bytes

Compared to B+Tree:
B+Tree index size ≈ n × (key_size + pointer_size) bytes
```

**Example**: 1 million rows, attribute with 5 values:
- Bitmap: 1,000,000 x 5 / 8 = 625,000 bytes ≈ 610 KB
- B+Tree: 1,000,000 x (8 + 8) = 16,000,000 bytes ≈ 15.3 MB

Bitmap is **25x smaller** for low-cardinality attributes.

---

## 7. Multi-Dimensional Indexing

### 7.1 The Problem

Standard B+Trees and hash indices work well for **one-dimensional** search keys. But many queries involve **multiple dimensions**:

```sql
-- Spatial query
SELECT * FROM restaurants
WHERE latitude BETWEEN 37.7 AND 37.8
  AND longitude BETWEEN -122.5 AND -122.4;

-- Multi-attribute range query
SELECT * FROM products
WHERE price BETWEEN 10 AND 50
  AND weight BETWEEN 0.5 AND 2.0;
```

A B+Tree on `(latitude, longitude)` can efficiently filter on `latitude` but then must scan all matching entries for `longitude`. True multi-dimensional indices address both dimensions simultaneously.

### 7.2 R-Tree

The **R-Tree** (Rectangle Tree) is the most widely used spatial index. It organizes data using **minimum bounding rectangles (MBRs)**.

```
R-Tree Structure:

Root: [MBR₁, MBR₂]
       ┌──────────────────────────────────────┐
       │   ┌───────────┐    ┌──────────────┐  │
       │   │   MBR₁    │    │    MBR₂      │  │
       │   │ ┌──┐ ┌──┐ │    │ ┌───┐  ┌──┐  │  │
       │   │ │P1│ │P2│ │    │ │P3 │  │P4│  │  │
       │   │ └──┘ └──┘ │    │ └───┘  └──┘  │  │
       │   │      ┌──┐ │    │     ┌──┐     │  │
       │   │      │P5│ │    │     │P6│     │  │
       │   │      └──┘ │    │     └──┘     │  │
       │   └───────────┘    └──────────────┘  │
       └──────────────────────────────────────┘

Each internal node: [MBR₁, ptr₁, MBR₂, ptr₂, ...]
Each leaf node: [MBR₁, oid₁, MBR₂, oid₂, ...]
```

**Properties:**
- Balanced tree (all leaves at the same level)
- Each node contains between `⌈m/2⌉` and `m` entries
- MBRs at a given level may overlap
- Search may need to follow multiple paths (unlike B+Tree)

**Operations:**
- **Search**: Start at root, descend into all children whose MBR overlaps the query rectangle
- **Insert**: Choose the subtree whose MBR needs the least enlargement; split if necessary
- **Used in**: PostgreSQL (GiST indices), Oracle Spatial, PostGIS

### 7.3 kd-Tree

The **kd-tree** (k-dimensional tree) partitions space by alternating splitting dimensions at each level.

```
2D kd-tree example (splitting on x, then y, then x, ...):

Data points: (2,3), (5,4), (9,6), (4,7), (8,1), (7,2)

                    (7,2)          Split on x=7
                   /     \
              (5,4)       (9,6)    Split on y=4, y=6
             /    \          \
          (2,3)  (4,7)      (8,1)  Split on x=2, x=4, x=8

Spatial partitioning:
┌───────────────────────┐
│           │            │
│  (2,3)    │   (9,6)   │
│    ·      │     ·     │
│───────(5,4)──────     │
│    ·  │   │           │
│ (4,7) │   │  (8,1)    │
│       │   │    ·      │
└───────┴───┴───────────┘
        x=7    (split line)
```

**Properties:**
- Binary tree (each node splits space in half)
- Efficient for low-dimensional data (d ≤ 20)
- Search: O(n^(1-1/d) + k) where k is the number of results
- Not balanced for dynamic data (use variants like k-d-B tree)

**Comparison of spatial indices:**

| Feature | R-Tree | kd-Tree |
|---|---|---|
| Dimensions | Works for any d | Best for low d |
| Dynamic updates | Good (designed for it) | Poor (may unbalance) |
| Disk-based | Yes (designed for it) | Originally in-memory |
| Overlap | Allows overlap | No overlap |
| Use case | GIS, spatial databases | k-NN search, in-memory |

---

## 8. Index Selection and Design Guidelines

### 8.1 Index Selection Problem

Choosing the right indices is critical for performance. The **index selection problem** asks: given a workload (set of queries), which indices should be created?

**Factors to consider:**
1. **Query patterns**: Which columns appear in WHERE, JOIN, ORDER BY, GROUP BY?
2. **Update frequency**: How often are INSERT, UPDATE, DELETE executed?
3. **Data distribution**: Cardinality and selectivity of each column
4. **Storage budget**: Each index costs disk space and maintenance overhead
5. **Correlation**: Columns that are frequently queried together

### 8.2 Covering Index

A **covering index** includes all columns needed by a query, so the database can answer the query from the index alone without accessing the table (**index-only scan**).

```sql
-- Query:
SELECT name, salary FROM employees WHERE department = 'Eng';

-- Non-covering index (requires table access):
CREATE INDEX idx_dept ON employees(department);
-- Finds matching rows via index, then fetches name, salary from table

-- Covering index (no table access needed):
CREATE INDEX idx_dept_covering ON employees(department, name, salary);
-- All needed columns are in the index
```

**PostgreSQL INCLUDE syntax:**

```sql
-- INCLUDE columns are stored in the leaf level but not in the search key
CREATE INDEX idx_dept_incl ON employees(department) INCLUDE (name, salary);
-- department is the search key; name and salary are payload only
```

**Benefits:**
- Eliminates table access (huge I/O savings)
- Particularly effective for frequently-run queries

**Trade-offs:**
- Larger index size
- More columns to maintain on updates

### 8.3 Composite (Multi-Column) Index

A **composite index** is built on multiple columns. The column order matters significantly.

```sql
CREATE INDEX idx_dept_salary ON employees(department, salary);
```

This index is useful for:

```sql
-- Uses full index (both columns):
SELECT * FROM employees WHERE department = 'Eng' AND salary > 80000;

-- Uses index prefix (department only):
SELECT * FROM employees WHERE department = 'Eng';

-- Does NOT use this index efficiently:
SELECT * FROM employees WHERE salary > 80000;  -- salary is not a prefix
```

**Leftmost prefix rule**: A composite index on `(A, B, C)` can be used for queries on:
- `A`
- `A, B`
- `A, B, C`

But NOT efficiently for:
- `B` alone
- `C` alone
- `B, C`

### 8.4 Partial Index

A **partial index** indexes only a subset of rows, defined by a predicate.

```sql
-- Index only active employees (if 90% are inactive, index is much smaller)
CREATE INDEX idx_active_emp ON employees(name)
WHERE status = 'active';

-- Index only recent orders
CREATE INDEX idx_recent_orders ON orders(customer_id)
WHERE order_date > '2025-01-01';

-- Index only non-null values
CREATE INDEX idx_email ON users(email)
WHERE email IS NOT NULL;
```

**Benefits:**
- Smaller index size (fewer entries)
- Faster to maintain (fewer updates)
- Better cache utilization

### 8.5 Expression Index

Some databases support indexing computed expressions:

```sql
-- PostgreSQL: index on expression
CREATE INDEX idx_lower_email ON users(lower(email));

-- Useful for case-insensitive searches:
SELECT * FROM users WHERE lower(email) = 'alice@example.com';

-- Index on year extracted from timestamp:
CREATE INDEX idx_order_year ON orders(EXTRACT(YEAR FROM order_date));
```

### 8.6 Practical Guidelines

**Rule 1: Index columns in WHERE, JOIN, ORDER BY**
```sql
-- This query benefits from indices on customer_id, order_date, status
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date > '2025-01-01'
ORDER BY o.status;
```

**Rule 2: Consider selectivity**
- High selectivity (few matching rows) = good index candidate
- Low selectivity (most rows match) = poor index candidate
- Rule of thumb: Index is useful if it selects < 15-20% of rows

**Rule 3: Avoid over-indexing**
```
Each index costs:
- Disk space (size of the index structure)
- Insert overhead: ~1 B+Tree insert per index
- Update overhead: delete old + insert new per index
- Vacuum/maintenance overhead

Typical guideline: 3-5 indices per table, rarely more than 10
```

**Rule 4: Use EXPLAIN to verify index usage**
```sql
EXPLAIN ANALYZE SELECT * FROM employees WHERE department = 'Eng';

-- Look for:
-- "Index Scan" or "Index Only Scan" → index is being used
-- "Seq Scan" → index is NOT being used (or doesn't exist)
-- "Bitmap Index Scan" → bitmap index used for multi-condition queries
```

**Rule 5: Monitor and maintain indices**

```sql
-- PostgreSQL: check index usage statistics
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
-- Indices with idx_scan = 0 are unused and can be dropped

-- Check index size
SELECT pg_size_pretty(pg_relation_size('idx_emp_salary'));

-- Rebuild fragmented indices
REINDEX INDEX idx_emp_salary;
```

---

## 9. Index Structures in Major Databases

### PostgreSQL

```sql
-- B+Tree (default)
CREATE INDEX idx_btree ON table(column);

-- Hash (equality only, WAL-logged since v10)
CREATE INDEX idx_hash ON table USING hash(column);

-- GiST (Generalized Search Tree -- R-Tree, full-text, etc.)
CREATE INDEX idx_gist ON table USING gist(geom_column);

-- GIN (Generalized Inverted Index -- arrays, JSONB, full-text)
CREATE INDEX idx_gin ON table USING gin(jsonb_column);

-- BRIN (Block Range Index -- large sorted tables)
CREATE INDEX idx_brin ON table USING brin(timestamp_column);

-- SP-GiST (Space-Partitioned GiST -- kd-tree, radix tree)
CREATE INDEX idx_spgist ON table USING spgist(point_column);
```

### MySQL InnoDB

```
- Primary key = clustered B+Tree index (data stored in leaf nodes)
- Secondary indices store the primary key value (not a row pointer)
- This means secondary index lookup requires two B+Tree traversals:
  1. Secondary index → primary key value
  2. Primary index → actual row data
```

### Summary Table

| Index Type | Best For | Limitation |
|---|---|---|
| B+Tree | General purpose, range, ORDER BY | Not great for multi-dimension |
| Hash | Exact equality lookup | No range support |
| Bitmap | Low cardinality, analytics | Poor for OLTP, high cardinality |
| GiST/R-Tree | Spatial, geometric data | Overlap can slow search |
| GIN | Full-text, arrays, JSONB | Large index, slow updates |
| BRIN | Very large, naturally ordered tables | Low precision |

---

## 10. Exercises

### Conceptual Questions

**Exercise 1**: Explain why a secondary index must be dense (one entry per record) while a primary index can be sparse (one entry per block).

**Exercise 2**: A table has 500,000 records stored in blocks of 4 KB each. Each record is 200 bytes. A B+Tree index on the primary key uses 8-byte keys and 8-byte pointers. Each index node is one block (4 KB).

(a) How many records fit in one data block?
(b) How many data blocks does the table occupy?
(c) What is the maximum fan-out (order) of the B+Tree?
(d) What is the maximum height of the B+Tree?
(e) How many disk I/Os are needed for an equality search using the index vs. a full table scan?

**Exercise 3**: Compare B-Tree and B+Tree. Why do virtually all database systems use B+Trees instead of B-Trees?

**Exercise 4**: A bitmap index is created on a `color` column with 8 distinct values in a table with 10 million rows.

(a) What is the total uncompressed size of the bitmap index?
(b) If only 0.1% of rows have color = 'purple', how compressible is the purple bitmap using run-length encoding?
(c) Compare this to a B+Tree index on the same column.

### Practical Questions

**Exercise 5**: For the following query workload, design an indexing strategy:

```sql
-- Q1 (90% of traffic): exact lookup
SELECT * FROM users WHERE email = ?;

-- Q2 (5% of traffic): range scan with sort
SELECT name, created_at FROM users
WHERE created_at > ? ORDER BY created_at DESC LIMIT 20;

-- Q3 (3% of traffic): multi-column filter
SELECT * FROM users
WHERE country = ? AND age BETWEEN ? AND ?;

-- Q4 (2% of traffic): text search
SELECT * FROM users WHERE name ILIKE '%smith%';
```

For each query, specify:
(a) What index (or indices) would you create?
(b) What type of index (B+Tree, hash, GIN, etc.)?
(c) Would you use a composite, covering, or partial index?

**Exercise 6**: Given the following extendible hash directory with global depth 2 and bucket capacity 2:

```
Directory:          Buckets:
00 → Bucket A [h=00110, h=00010]  (local depth 2)
01 → Bucket B [h=01100]            (local depth 2)
10 → Bucket C [h=10001, h=10110]  (local depth 2)
11 → Bucket D [h=11000]            (local depth 2)
```

Show the state of the directory and buckets after inserting a record with hash value `h = 00001`.

**Exercise 7**: Consider a linear hashing scheme with 4 initial buckets (N=4), split pointer s=0, and bucket capacity 2. The hash functions are h₀(K) = K mod 4 and h₁(K) = K mod 8. Starting with:

```
Bucket 0: [8, 16]   (full)
Bucket 1: [5]
Bucket 2: [10]
Bucket 3: [7, 15]   (full)
```

Show the state after inserting keys 12, 9, and 3 (one at a time).

### Analysis Questions

**Exercise 8**: Prove that the height of a B+Tree with order `m` and `n` keys is at most `⌈log_{⌈m/2⌉}(n)⌉`.

**Exercise 9**: A database administrator notices that a query `SELECT COUNT(*) FROM orders WHERE status = 'pending'` takes 2 seconds on a table with 50 million rows. The `status` column has 5 distinct values. Recommend an indexing strategy and estimate the improvement.

**Exercise 10**: You have a table `sensor_data` with 1 billion rows, columns `(sensor_id, timestamp, value, location_x, location_y)`, and the following query patterns:

- Time-range queries: `WHERE timestamp BETWEEN ? AND ?`
- Sensor-specific queries: `WHERE sensor_id = ? AND timestamp BETWEEN ? AND ?`
- Spatial queries: `WHERE location_x BETWEEN ? AND ? AND location_y BETWEEN ? AND ?`
- Aggregations: `SELECT sensor_id, AVG(value) FROM ... GROUP BY sensor_id`

Design a comprehensive indexing strategy. Consider B+Tree, BRIN, GiST, and composite indices. Justify your choices.

---

**Previous**: [Query Processing and Optimization](./08_Query_Processing.md) | **Next**: [Transaction Theory](./10_Transaction_Theory.md)
