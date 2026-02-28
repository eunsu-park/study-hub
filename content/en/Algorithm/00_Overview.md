# Algorithm Learning Guide

## Introduction

This folder contains materials for systematically learning algorithms and data structures. You can study step-by-step from complexity analysis to advanced algorithms, which is helpful for coding interview preparation and algorithm competition training.

**Target Audience**: Developers with programming fundamentals, coding test preparation candidates

---

## Learning Roadmap

```
[Basic]                   [Intermediate]            [Advanced]
  │                         │                         │
  ▼                         ▼                         ▼
Complexity Analysis ───▶ Divide & Conquer ───▶ Dynamic Programming
  │                    │                      │
  ▼                    ▼                      ▼
Arrays/Strings ─────▶ Trees/BST ───────────▶ Segment Tree
  │                    │                      │
  ▼                    ▼                      ▼
Stacks/Queues ──────▶ Graph Basics ────────▶ Network Flow
  │                    │                      │
  ▼                    ▼                      ▼
Hash Tables ────────▶ Shortest Path/MST ──▶ Advanced DP Optimization
```

---

## Prerequisites

- Programming fundamentals (variables, control flow, functions)
- Basic data structures (arrays, lists)
- At least one language among C, C++, or Python

---

## File List

### Basic Data Structures (01-05)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [01_Complexity_Analysis.md](./01_Complexity_Analysis.md) | ⭐ | Big O, time/space complexity, analysis techniques |
| [02_Arrays_and_Strings.md](./02_Arrays_and_Strings.md) | ⭐ | Two pointers, sliding window, prefix sum |
| [03_Stacks_and_Queues.md](./03_Stacks_and_Queues.md) | ⭐ | Parenthesis checking, postfix notation, monotonic stack |
| [04_Hash_Tables.md](./04_Hash_Tables.md) | ⭐⭐ | Hash functions, collision resolution, hashmap/set implementation |
| [05_Sorting_Algorithms.md](./05_Sorting_Algorithms.md) | ⭐⭐ | Bubble/selection/insertion/merge/quick/heap sort |

### Search and Divide & Conquer (06-08)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [06_Searching_Algorithms.md](./06_Searching_Algorithms.md) | ⭐⭐ | Binary search, parametric search, hash search |
| [07_Divide_and_Conquer.md](./07_Divide_and_Conquer.md) | ⭐⭐ | Merge sort, quick sort, fast exponentiation |
| [08_Backtracking.md](./08_Backtracking.md) | ⭐⭐⭐ | N-Queens, permutations/combinations, pruning |

### Tree Data Structures (09-11)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [09_Trees_and_BST.md](./09_Trees_and_BST.md) | ⭐⭐ | Tree traversal, BST operations, balanced trees |
| [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) | ⭐⭐ | Heap structure, heap sort, kth element |
| [11_Trie.md](./11_Trie.md) | ⭐⭐⭐ | Prefix tree, autocomplete, XOR trie |

### Graphs (12-17)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [12_Graph_Basics.md](./12_Graph_Basics.md) | ⭐⭐ | Graph representation, DFS, BFS |
| [13_Topological_Sort.md](./13_Topological_Sort.md) | ⭐⭐⭐ | Kahn, DFS-based, cycle detection |
| [14_Shortest_Path.md](./14_Shortest_Path.md) | ⭐⭐⭐ | Dijkstra, Bellman-Ford, Floyd-Warshall |
| [15_Minimum_Spanning_Tree.md](./15_Minimum_Spanning_Tree.md) | ⭐⭐⭐ | Kruskal, Prim, Union-Find |
| [16_LCA_and_Tree_Queries.md](./16_LCA_and_Tree_Queries.md) | ⭐⭐⭐ | Binary Lifting, Sparse Table |
| [17_Strongly_Connected_Components.md](./17_Strongly_Connected_Components.md) | ⭐⭐⭐⭐ | Tarjan, Kosaraju, 2-SAT |

### DP and Mathematics (18-22)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [18_Dynamic_Programming.md](./18_Dynamic_Programming.md) | ⭐⭐⭐ | Memoization, knapsack, LCS, LIS |
| [19_Greedy_Algorithms.md](./19_Greedy_Algorithms.md) | ⭐⭐ | Activity selection, Huffman coding, greedy vs DP |
| [20_Bitmask_DP.md](./20_Bitmask_DP.md) | ⭐⭐⭐ | Bit operations, TSP, set DP |
| [21_Math_and_Number_Theory.md](./21_Math_and_Number_Theory.md) | ⭐⭐⭐ | Modular arithmetic, GCD/LCM, primes, combinatorics |
| [22_String_Algorithms.md](./22_String_Algorithms.md) | ⭐⭐⭐ | KMP, Rabin-Karp, Z-algorithm |

### Advanced Data Structures (23-24)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [23_Segment_Tree.md](./23_Segment_Tree.md) | ⭐⭐⭐⭐ | Range queries, Lazy Propagation |
| [24_Fenwick_Tree.md](./24_Fenwick_Tree.md) | ⭐⭐⭐ | BIT, range sum, 2D Fenwick |

### Advanced Graphs (25)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [25_Network_Flow.md](./25_Network_Flow.md) | ⭐⭐⭐⭐ | Ford-Fulkerson, bipartite matching |

### Special Topics (26-28)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [26_Computational_Geometry.md](./26_Computational_Geometry.md) | ⭐⭐⭐ | CCW, convex hull, line segment intersection |
| [27_Game_Theory.md](./27_Game_Theory.md) | ⭐⭐⭐ | Nim game, Sprague-Grundy, minimax |
| [28_Advanced_DP_Optimization.md](./28_Advanced_DP_Optimization.md) | ⭐⭐⭐⭐ | CHT, D&C optimization, Knuth optimization |

### Advanced Data Structures II (30-32)

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [30_Heavy_Light_Decomposition.md](./30_Heavy_Light_Decomposition.md) | ⭐⭐⭐⭐⭐ | HLD, path queries on trees |
| [31_Link_Cut_Tree.md](./31_Link_Cut_Tree.md) | ⭐⭐⭐⭐⭐ | Dynamic tree connectivity, access lemma |
| [32_Persistent_Segment_Tree.md](./32_Persistent_Segment_Tree.md) | ⭐⭐⭐⭐⭐ | Persistent data structures, version control on segment trees |

### Wrap-up

| Filename | Difficulty | Key Topics |
|----------|-----------|-----------|
| [29_Problem_Solving.md](./29_Problem_Solving.md) | ⭐⭐⭐⭐ | Problem type approaches, LeetCode/BOJ recommendations |

---

## Recommended Learning Sequence

### Phase 1: Basic Data Structures (1 week)
```
01 → 02 → 03 → 04
```

### Phase 2: Sorting and Searching (1 week)
```
05 → 06 → 07 → 08
```

### Phase 3: Trees (1 week)
```
09 → 10 → 11
```

### Phase 4: Graph Basics (1-2 weeks)
```
12 → 13 → 14 → 15
```

### Phase 5: Advanced Graphs (1 week)
```
16 → 17
```

### Phase 6: DP and Mathematics (2 weeks)
```
18 → 19 → 20 → 21 → 22
```

### Phase 7: Advanced Data Structures (1 week)
```
23 → 24
```

### Phase 8: Advanced Graphs and Special Topics (2+ weeks)
```
25 → 26 → 27 → 28
```

### Wrap-up
```
29 (recommended to proceed in parallel with other phases)
```

---

## Practice Environment

### Language-specific Setup

```bash
# C/C++ (GCC)
gcc --version
g++ --version

# Python
python3 --version

# Online environments
# - https://www.onlinegdb.com/
# - https://replit.com/
```

### Recommended Tools

- **IDE**: VS Code, CLion, PyCharm
- **Online Judge**: BOJ (Baekjoon), LeetCode, Programmers
- **Visualization**: https://visualgo.net/

---

## Complexity Quick Reference

| Complexity | Name | Example | ~10^6 operations |
|-----------|------|---------|-----------------|
| O(1) | Constant | Array access | Instant |
| O(log n) | Logarithmic | Binary search | ~20 ops |
| O(n) | Linear | Sequential search | 10^6 ops |
| O(n log n) | Linearithmic | Merge sort | ~2×10^7 ops |
| O(n²) | Quadratic | Bubble sort | 10^12 ops (caution) |
| O(2^n) | Exponential | Subsets | Infeasible |

---

## Related Resources

### Integration with Other Folders

| Folder | Related Content |
|--------|----------------|
| [C_Programming/](../C_Programming/00_Overview.md) | Dynamic arrays, linked lists, hash table implementation |
| [CPP/](../CPP/00_Overview.md) | STL containers, algorithm header usage |
| [Python/](../Python/00_Overview.md) | List comprehension, itertools usage |

### External Resources

- [Baekjoon Online Judge](https://www.acmicpc.net/)
- [LeetCode](https://leetcode.com/)
- [Programmers](https://programmers.co.kr/)
- [VisuAlgo - Algorithm Visualization](https://visualgo.net/)

---

## Learning Tips

1. **Implement Yourself**: Try implementing before using libraries
2. **Complexity Analysis**: Calculate time/space complexity for all code
3. **Multiple Languages**: Implement the same algorithm in C, C++, and Python
4. **Problem Solving**: Solve 3-5 related problems after each lesson
5. **Error Log**: Record why you got problems wrong

---

## Coding Test Preparation Guide

### Goals by Level

| Level | Goal | Required Lessons |
|-------|------|-----------------|
| Beginner | Solve basic problems | 01~08 |
| Intermediate | Major company coding tests | 01~17 |
| Advanced | Algorithm competitions | 01~25 |
| Expert | ICPC/IOI level | All (including 26~29) |

### Time Limit Guidelines

- **1 second**: O(n log n) or better (n ≤ 10^6)
- **2 seconds**: O(n²) feasible (n ≤ 5000)
- **5 seconds**: O(n²) feasible (n ≤ 10000)
