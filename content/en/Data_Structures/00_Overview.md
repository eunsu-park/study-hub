# Data Structures

Data structures are the building blocks of efficient software. They determine how data is organized, accessed, and modified, directly impacting the performance and scalability of every program you write. This topic provides a thorough introduction to fundamental data structures, their implementations in Python, and the reasoning behind choosing one structure over another.

## What You'll Learn

This topic provides hands-on coverage of:
- **Arrays and Lists**: Dynamic arrays, indexing, slicing, and amortized time complexity
- **Linked Lists**: Singly, doubly, and circular linked lists with pointer manipulation
- **Stacks and Queues**: LIFO/FIFO abstractions, implementations, and real-world applications
- **Hash Tables**: Hashing functions, collision resolution, and Python's dict internals
- **Trees**: Binary trees, traversals, binary search trees, and balanced tree concepts
- **Heaps**: Min-heaps, max-heaps, and priority queue operations
- **Graphs**: Representations, terminology, BFS, and DFS
- **Sets and Maps**: Mathematical set operations, map abstractions, and implementations
- **String Data Structures**: Tries, suffix arrays, and pattern matching fundamentals
- **Sorting and Searching**: Classic algorithms with complexity analysis
- **Choosing Data Structures**: Trade-off analysis and decision frameworks

## Prerequisites

- [Programming](../Programming/00_Overview.md) -- General programming concepts (variables, control flow, functions)
- [Python Basics](../Python_Basics/00_Overview.md) -- Python syntax, built-in types, classes, and basic OOP

No prior knowledge of data structures or algorithms is required. If you can write a Python class and use lists and dictionaries, you are ready.

## Learning Roadmap

```
                       Data Structures -- Learning Path
  +-----------------------------------------------------------------------+
  |                                                                         |
  |  +--------------+   +------------------+   +------------------------+  |
  |  | 01 Arrays &  |-->| 02 Linked Lists  |-->| 03 Stacks              |  |
  |  |    Lists     |   |                  |   |                        |  |
  |  +--------------+   +------------------+   +------------+-----------+  |
  |                                                          |              |
  |                                                          v              |
  |  +--------------+   +------------------+   +------------------------+  |
  |  | 06 Trees     |<--| 05 Hash Tables   |<--| 04 Queues              |  |
  |  |    Basics    |   |                  |   |                        |  |
  |  +------+-------+   +------------------+   +------------------------+  |
  |         |                                                               |
  |         v                                                               |
  |  +--------------+   +------------------+   +------------------------+  |
  |  | 07 Binary    |-->| 08 Heaps         |-->| 09 Graphs Basics       |  |
  |  |    Search    |   |                  |   |                        |  |
  |  |    Trees     |   |                  |   |                        |  |
  |  +--------------+   +------------------+   +------------+-----------+  |
  |                                                          |              |
  |                                                          v              |
  |  +--------------+   +------------------+   +------------------------+  |
  |  | 12 Sorting   |<--| 11 Strings as DS |<--| 10 Sets & Maps         |  |
  |  |              |   |                  |   |                        |  |
  |  +------+-------+   +------------------+   +------------------------+  |
  |         |                                                               |
  |         v                                                               |
  |  +--------------+   +------------------+                               |
  |  | 13 Searching |-->| 14 Choosing the  |                               |
  |  |              |   |    Right DS      |                               |
  |  +--------------+   +------------------+                               |
  |                                                                         |
  +-----------------------------------------------------------------------+
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Arrays and Lists](01_Arrays_and_Lists.md) | ⭐ | Dynamic arrays, indexing, slicing, time complexity |
| 02 | [Linked Lists](02_Linked_Lists.md) | ⭐ | Singly, doubly, circular linked lists |
| 03 | [Stacks](03_Stacks.md) | ⭐ | LIFO, array/linked-list implementations, applications |
| 04 | [Queues](04_Queues.md) | ⭐ | FIFO, circular queue, deque |
| 05 | [Hash Tables](05_Hash_Tables.md) | ⭐⭐ | Hashing, collision resolution, Python dict internals |
| 06 | [Trees Basics](06_Trees_Basics.md) | ⭐⭐ | Binary trees, traversals (inorder, preorder, postorder, level-order) |
| 07 | [Binary Search Trees](07_Binary_Search_Trees.md) | ⭐⭐ | Insert, delete, search, BST properties |
| 08 | [Heaps](08_Heaps.md) | ⭐⭐ | Min-heap, max-heap, priority queue, heapify |
| 09 | [Graphs Basics](09_Graphs_Basics.md) | ⭐⭐ | Adjacency list/matrix, BFS, DFS, terminology |
| 10 | [Sets and Maps](10_Sets_and_Maps.md) | ⭐ | Set operations, map/dictionary implementations |
| 11 | [Strings as Data Structures](11_Strings_as_Data_Structures.md) | ⭐⭐ | Trie, pattern matching, string hashing |
| 12 | [Sorting Fundamentals](12_Sorting_Fundamentals.md) | ⭐⭐ | Bubble, selection, insertion, merge, quick sort |
| 13 | [Searching Fundamentals](13_Searching_Fundamentals.md) | ⭐ | Linear search, binary search, hash-based search |
| 14 | [Choosing the Right Data Structure](14_Choosing_the_Right_Data_Structure.md) | ⭐⭐ | Trade-offs, comparison tables, decision guide |

## Recommended Learning Order

Follow the lessons sequentially from 01 through 14. Each lesson builds on concepts introduced in the previous one:

1. **Linear Structures (Lessons 1-4)**: Arrays, linked lists, stacks, and queues form the foundation of all data structures
2. **Associative Structures (Lesson 5)**: Hash tables power dictionaries and sets -- the workhorses of practical programming
3. **Hierarchical Structures (Lessons 6-8)**: Trees and heaps introduce recursive data organization
4. **Graph Structures (Lesson 9)**: Graphs generalize trees to model networks and relationships
5. **Specialized Structures (Lessons 10-11)**: Sets, maps, and string-specific structures
6. **Algorithms on Structures (Lessons 12-13)**: Sorting and searching tie everything together
7. **Synthesis (Lesson 14)**: Learn to choose the right tool for each job

## Practice Environment

All examples use Python 3.10+ with no external dependencies:

```bash
python3 --version
# Python 3.12.x (or newer)
```

Example code for each lesson is available in `examples/Data_Structures/`.
Exercise stubs with test cases are available in `exercises/Data_Structures/`.

## Related Materials

- [Python Basics](../Python_Basics/00_Overview.md) -- Core Python syntax and built-in data types
- [Algorithm](../Algorithm/00_Overview.md) -- Advanced algorithm design and analysis
- [Python Advanced](../Python_Advanced/00_Overview.md) -- Generators, iterators, and advanced Python patterns
- [Database Theory](../Database_Theory/00_Overview.md) -- How data structures underpin database engines

---

**License**: Content licensed under CC BY-NC 4.0
