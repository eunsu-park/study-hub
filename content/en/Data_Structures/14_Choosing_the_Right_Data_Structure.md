# Choosing the Right Data Structure

**Previous**: [Searching Fundamentals](./13_Searching_Fundamentals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare data structures across multiple dimensions (time, space, use case)
2. Apply a systematic decision framework to choose the right data structure
3. Identify common problem patterns and their optimal data structures
4. Understand time-space trade-offs in data structure selection
5. Avoid common mistakes in data structure selection
6. Map real-world problems to appropriate data structures
7. Combine multiple data structures for complex requirements

---

Choosing the right data structure is one of the most impactful decisions in software engineering. The wrong choice can turn an O(n) operation into O(n^2), or waste megabytes of memory. This lesson synthesizes everything from the previous 13 lessons into a practical decision guide.

## The Big Picture: Operation Complexity

### Linear Structures

| Operation | Array/List | Linked List | Stack | Queue | Deque |
|-----------|-----------|-------------|-------|-------|-------|
| Access by index | **O(1)** | O(n) | -- | -- | O(n) |
| Insert at front | O(n) | **O(1)** | -- | -- | **O(1)** |
| Insert at back | O(1)* | O(1)** | **O(1)** | **O(1)** | **O(1)** |
| Delete at front | O(n) | **O(1)** | -- | **O(1)** | **O(1)** |
| Delete at back | O(1) | O(n)*** | **O(1)** | -- | **O(1)** |
| Search | O(n) | O(n) | O(n) | O(n) | O(n) |

*Amortized | **With tail pointer | ***O(1) for doubly linked

### Associative Structures

| Operation | Hash Table | BST (balanced) | Trie | Sorted Array |
|-----------|-----------|----------------|------|-------------|
| Insert | **O(1)** | O(log n) | O(m) | O(n) |
| Delete | **O(1)** | O(log n) | O(m) | O(n) |
| Search | **O(1)** | O(log n) | O(m) | O(log n) |
| Min/Max | O(n) | **O(log n)** | -- | **O(1)** |
| Range query | O(n) | **O(log n + k)** | -- | O(log n + k) |
| Prefix search | O(n) | O(n) | **O(m)** | O(log n + k) |
| Ordered iteration | O(n log n) | **O(n)** | O(n) | **O(n)** |

### Priority Structures

| Operation | Unsorted Array | Sorted Array | Heap |
|-----------|---------------|-------------|------|
| Insert | **O(1)** | O(n) | O(log n) |
| Find min/max | O(n) | **O(1)** | **O(1)** |
| Extract min/max | O(n) | O(1)* | O(log n) |
| Build | **O(n)** | O(n log n) | **O(n)** |

*O(n) if removing from front

## Decision Framework

### Step 1: What Operations Do You Need?

```
                    What is your primary operation?
                              |
            +--------+--------+--------+--------+
            |        |        |        |        |
         Access   Insert/   Search   Order    Priority
         by key   Delete            matters?
            |        |        |        |        |
          dict    Where?    Type?    Yes/No    Heap
            |        |        |        |
            |   +----+----+  |     +--+--+
            |   |    |    |  |     |     |
            |  Front Mid End |    BST  Sorted
            |   |    |    |  |         Array
            |  deque LL list |
            |              |  |
            |              v  v
            v           Hash Table
         Hash Table    (exact match)
         or dict       BST (range)
```

### Step 2: Ask These Questions

| Question | If Yes | If No |
|----------|--------|-------|
| Do you need random access by index? | Array/list | Consider linked structures |
| Do you need fast insert/delete at both ends? | Deque | Array may work |
| Do you need key-value associations? | Dict/hash table | -- |
| Do you need sorted order? | BST, sorted array, heap | Hash table is fine |
| Do you need fast min/max? | Heap | -- |
| Do you need prefix matching? | Trie | -- |
| Is memory a concern? | Array (compact) | Linked structures OK |
| Is the data size fixed? | Array | Dynamic structures |
| Do you need LIFO access? | Stack | -- |
| Do you need FIFO access? | Queue/deque | -- |

## Common Problem Patterns

### Pattern 1: Frequency Counting

**Problem**: Count occurrences of elements.

**Best structure**: `Counter` (specialized dict)

```python
from collections import Counter
counts = Counter(items)
most_common = counts.most_common(k)
```

### Pattern 2: Deduplication

**Problem**: Remove duplicates while preserving order.

**Best structure**: `dict` (preserves insertion order in Python 3.7+)

```python
unique = list(dict.fromkeys(items))
```

### Pattern 3: Fast Lookup + Insertion Order

**Problem**: Need O(1) lookup and maintain insertion/access order.

**Best structure**: `OrderedDict` or `dict` + doubly linked list (LRU cache)

### Pattern 4: Top-K Elements

**Problem**: Find the K largest/smallest elements.

**Best structure**: Heap (min-heap for K largest, max-heap for K smallest)

```python
import heapq
top_k = heapq.nlargest(k, items, key=...)
```

### Pattern 5: Range Queries on Sorted Data

**Problem**: Find all elements in range [a, b].

**Best structure**: BST or sorted array with `bisect`

### Pattern 6: Graph/Network Relationships

**Problem**: Model connections between entities.

**Best structure**: Adjacency list (`dict` of `list`s)

### Pattern 7: Undo/Redo

**Problem**: Track state changes for undo/redo.

**Best structure**: Two stacks (undo stack, redo stack)

### Pattern 8: Task Scheduling by Priority

**Problem**: Process tasks by priority.

**Best structure**: Heap-based priority queue

### Pattern 9: Autocomplete/Prefix Search

**Problem**: Find all words matching a prefix.

**Best structure**: Trie

### Pattern 10: Sliding Window Statistics

**Problem**: Track min/max/sum in a sliding window.

**Best structure**: Monotonic deque or two heaps

## Time-Space Trade-offs

```
More Space                              Less Space
<---------------------------------------------->
Hash table   BST       Sorted array    Linear search
  O(1)      O(log n)    O(log n)         O(n)
  O(n)      O(n)        O(n)             O(1) extra

Preprocessing time vs query time:
No prep + O(n) query  vs  O(n) prep + O(1) query
(single query)            (many queries)
```

### When to Preprocess

| Scenario | Strategy |
|----------|---------|
| One query | Linear search (no preprocessing overhead) |
| Many queries, exact match | Build a hash set/dict |
| Many queries, range | Sort + binary search |
| Many queries, prefix | Build a trie |
| Dynamic data | BST or hash table (supports insert/delete) |

## Common Mistakes

### Mistake 1: Using a List as a Queue

```python
# BAD: O(n) dequeue
queue = [1, 2, 3]
queue.pop(0)  # Shifts all elements

# GOOD: O(1) dequeue
from collections import deque
queue = deque([1, 2, 3])
queue.popleft()
```

### Mistake 2: Linear Search in a Loop

```python
# BAD: O(n*m) -- linear search for each query
for query in queries:
    if query in large_list:  # O(n) each time
        process(query)

# GOOD: O(n+m) -- convert to set first
lookup = set(large_list)
for query in queries:
    if query in lookup:  # O(1) each time
        process(query)
```

### Mistake 3: Sorting for Every Query

```python
# BAD: O(n log n) per query
for _ in range(q):
    data.sort()
    result = data[k]

# GOOD: Sort once or use a heap
data.sort()  # Once
# Or use heapq for dynamic data
```

### Mistake 4: Ignoring Built-in Data Structures

Python provides highly optimized built-ins. Prefer them over custom implementations:

| Need | Use | Not |
|------|-----|-----|
| Stack | `list` or `deque` | Custom linked list stack |
| Queue | `deque` | Custom implementation |
| Priority queue | `heapq` | Custom heap |
| Hash map | `dict` | Custom hash table |
| Sorted container | `bisect` + `list` | Custom BST |
| Counting | `Counter` | Manual dict counting |

## Real-World Mapping

| Real-World Problem | Data Structure |
|-------------------|----------------|
| Browser history (back/forward) | Two stacks |
| Print queue | Queue |
| Auto-complete search bar | Trie |
| Database index | B-tree / B+ tree |
| Social network friends | Graph (adjacency list) |
| Leaderboard / rankings | Heap or balanced BST |
| Spell checker | Hash set + edit distance |
| File system | Tree (n-ary) |
| CPU task scheduler | Priority queue (heap) |
| Cache (LRU/LFU) | Hash map + doubly linked list |
| Undo in text editor | Stack |
| Version control | DAG (directed acyclic graph) |
| DNS lookup | Hash table (with caching) |

## Decision Flowchart Summary

```
Start
  |
  v
Need key-value pairs? --Yes--> dict (hash map)
  |
  No
  |
  v
Need ordering/sorting? --Yes--> Sorted array, BST, or heap
  |                              |
  No                          Need dynamic insert/delete?
  |                              |           |
  v                            Yes          No
Need unique elements? ------+  BST      sorted array + bisect
  |                         |
  Yes --> set               Need min/max efficiently?
  |                              |
  No                           Yes --> heap
  |
  v
Need LIFO? --Yes--> stack (list)
  |
  No
  |
  v
Need FIFO? --Yes--> queue (deque)
  |
  No
  |
  v
Need fast insert at both ends? --Yes--> deque
  |
  No
  |
  v
Default: list (array)
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| No universal best | Each structure excels at specific operations |
| Know the operations | Identify what operations dominate your use case |
| Preprocessing | Build indices (hash, sort, trie) for repeated queries |
| Python built-ins | `dict`, `set`, `deque`, `heapq`, `bisect`, `Counter` |
| Common patterns | Counting, dedup, top-K, sliding window, graph |
| Trade-offs | Time vs space, build cost vs query cost |
| Real-world mapping | Match the problem structure to the data structure |

---

**License**: Content licensed under CC BY-NC 4.0
