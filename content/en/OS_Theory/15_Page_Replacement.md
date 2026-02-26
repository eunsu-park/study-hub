# Page Replacement ⭐⭐⭐⭐

**Previous**: [Virtual Memory](./14_Virtual_Memory.md) | **Next**: [File System Basics](./16_File_System_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why page replacement is necessary when physical memory is over-allocated
2. Implement FIFO, LRU, and Clock page replacement algorithms
3. Demonstrate Belady's anomaly with a concrete FIFO example
4. Compare page replacement algorithms by miss rate and implementation complexity
5. Explain thrashing and how working set models prevent it
6. Analyze the relationship between page frame allocation and page fault rate
7. Distinguish stack algorithms from non-stack algorithms and explain why this matters
8. Evaluate the Page Fault Frequency (PFF) method for dynamic frame allocation

---

When physical memory is full and a new page is needed, the OS must choose a victim page to evict. This choice directly affects your system's performance -- a bad replacement policy can cause thrashing, where the OS spends more time swapping pages than running your programs. The algorithms in this lesson range from the theoretically optimal (but unimplementable) to practical approximations used in real operating systems like Linux.

## Table of Contents

1. [Need for Page Replacement](#1-need-for-page-replacement)
2. [FIFO Algorithm](#2-fifo-algorithm)
3. [Optimal Algorithm](#3-optimal-algorithm)
4. [LRU Algorithm](#4-lru-algorithm)
5. [LRU Approximation Algorithms](#5-lru-approximation-algorithms)
6. [Belady's Anomaly](#6-beladys-anomaly)
7. [Thrashing and Working Set](#7-thrashing-and-working-set)
8. [Practice Problems](#practice-problems)

---

## 1. Need for Page Replacement

### 1.1 Over-allocation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Memory Over-allocation                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Physical Memory: 8GB (2,097,152 frames @ 4KB)                        │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │  Process 1: 2GB virtual address space                             │ │
│   │  Process 2: 4GB virtual address space                             │ │
│   │  Process 3: 3GB virtual address space                             │ │
│   │  Process 4: 2GB virtual address space                             │ │
│   │  ...                                                               │ │
│   │  Total virtual address space: 20GB                                │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│   20GB > 8GB → Cannot hold all pages in memory!                        │
│                                                                          │
│   Solution: Demand paging + Page replacement                            │
│   - Only actively used pages in memory                                  │
│   - Rest in disk swap space                                             │
│   - Replace when needed                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Page Replacement Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Page Replacement Process                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Page fault occurs (needed page not in memory)                      │
│      │                                                                   │
│      ▼                                                                   │
│   2. Find free frame                                                    │
│      ├── If available: use that frame                                   │
│      └── If not: page replacement needed                                │
│          │                                                               │
│          ▼                                                               │
│   3. Select victim page (using page replacement algorithm)              │
│      │                                                                   │
│      ▼                                                                   │
│   4. Handle victim page                                                 │
│      ├── Modified (Dirty): write to disk                                │
│      └── Not modified: discard                                          │
│      │                                                                   │
│      ▼                                                                   │
│   5. Load new page                                                      │
│      - Read from disk to frame                                          │
│      │                                                                   │
│      ▼                                                                   │
│   6. Update tables                                                      │
│      - Victim page: valid=0                                             │
│      - New page: valid=1, record frame number                           │
│      │                                                                   │
│      ▼                                                                   │
│   7. Restart instruction                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Dirty Bit

```c
// Using Dirty Bit in page replacement
void replace_page(int victim_frame) {
    PageTableEntry* victim_pte = get_pte_for_frame(victim_frame);

    if (victim_pte->dirty) {
        // Modified page - must write to disk
        write_to_swap(victim_frame, victim_pte->swap_slot);
        // 2 I/Os: read + write
    } else {
        // Not modified - disk copy is valid
        // Just discard, no disk write needed
        // 1 I/O: read only
    }

    // Load new page
    read_from_swap(victim_frame, new_page_swap_slot);
}
```

---

## 2. FIFO Algorithm

### 2.1 Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIFO (First-In, First-Out)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Rule: Replace the oldest page                                         │
│                                                                          │
│   Implementation: Use queue                                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   Page load order:  A → B → C → D → ...                        │   │
│   │                                                                  │   │
│   │   Memory (3 frames):                                            │   │
│   │   ┌───┬───┬───┐                                                │   │
│   │   │ A │ B │ C │  ← Oldest: A                                   │   │
│   │   └─┬─┴───┴───┘                                                │   │
│   │     │                                                            │   │
│   │     ▼ On D access                                               │   │
│   │   ┌───┬───┬───┐                                                │   │
│   │   │ D │ B │ C │  ← A replaced                                  │   │
│   │   └───┴─┬─┴───┘                                                │   │
│   │         │                                                        │   │
│   │         ▼ On E access                                           │   │
│   │   ┌───┬───┬───┐                                                │   │
│   │   │ D │ E │ C │  ← B replaced (now C is oldest)                │   │
│   │   └───┴───┴───┘                                                │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Advantages: Simple implementation                                      │
│   Disadvantages: May replace frequently used pages                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 FIFO Implementation

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX_FRAMES 10

typedef struct {
    int pages[MAX_FRAMES];
    int front;
    int rear;
    int count;
    int capacity;
} FIFOQueue;

void fifo_init(FIFOQueue* q, int capacity) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->capacity = capacity;
}

bool fifo_contains(FIFOQueue* q, int page) {
    for (int i = 0; i < q->count; i++) {
        int idx = (q->front + i) % q->capacity;
        if (q->pages[idx] == page) return true;
    }
    return false;
}

int fifo_access(FIFOQueue* q, int page) {
    // Hit if page already exists
    if (fifo_contains(q, page)) {
        printf("Page %d: hit\n", page);
        return 0;  // No page fault
    }

    // Page fault
    printf("Page %d: fault", page);

    if (q->count < q->capacity) {
        // Free frame available
        q->pages[q->rear] = page;
        q->rear = (q->rear + 1) % q->capacity;
        q->count++;
        printf(" (using free frame)\n");
    } else {
        // Replace oldest page
        int victim = q->pages[q->front];
        q->pages[q->front] = page;
        q->front = (q->front + 1) % q->capacity;
        printf(" (replaced page %d)\n", victim);
    }

    return 1;  // Page fault
}

void fifo_print(FIFOQueue* q) {
    printf("Memory: [");
    for (int i = 0; i < q->count; i++) {
        int idx = (q->front + i) % q->capacity;
        printf("%d", q->pages[idx]);
        if (i < q->count - 1) printf(", ");
    }
    printf("]\n\n");
}

int main() {
    FIFOQueue q;
    fifo_init(&q, 3);

    int reference_string[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    int n = sizeof(reference_string) / sizeof(reference_string[0]);

    int page_faults = 0;
    for (int i = 0; i < n; i++) {
        page_faults += fifo_access(&q, reference_string[i]);
        fifo_print(&q);
    }

    printf("Total page faults: %d\n", page_faults);
    return 0;
}
```

### 2.3 Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│               FIFO Example (3 frames)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2         │
│                                                                          │
│   Access  Frame 0   Frame 1   Frame 2    Fault?                         │
│   ─────────────────────────────────────────────────                     │
│    7      7         -          -          fault                         │
│    0      7         0          -          fault                         │
│    1      7         0          1          fault                         │
│    2      2         0          1          fault (replaced 7)            │
│    0      2         0          1          hit                           │
│    3      2         3          1          fault (replaced 0)            │
│    0      2         3          0          fault (replaced 1)            │
│    4      4         3          0          fault (replaced 2)            │
│    2      4         2          0          fault (replaced 3)            │
│    3      4         2          3          fault (replaced 0)            │
│    0      0         2          3          fault (replaced 4)            │
│    3      0         2          3          hit                           │
│    2      0         2          3          hit                           │
│    1      0         1          3          fault (replaced 2)            │
│    2      0         1          2          fault (replaced 3)            │
│                                                                          │
│   Total page faults: 12                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Optimal Algorithm

### 3.1 Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Optimal (OPT) Algorithm                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Rule: Replace the page that will be unused for the longest time       │
│         (Requires future knowledge - ideal algorithm)                   │
│                                                                          │
│   Example:                                                               │
│   Future reference: ... D, E, F, A, B, C, A ...                         │
│                      ↑                                                   │
│                   Current position                                       │
│                                                                          │
│   Memory: [A, B, C]                                                     │
│   Need to access D, which to replace?                                   │
│                                                                          │
│   - A: used 4 accesses later                                            │
│   - B: used 5 accesses later                                            │
│   - C: used 6 accesses later ← Used latest                             │
│                                                                          │
│   → Replacing C is optimal                                              │
│                                                                          │
│   Advantages: Guarantees minimum page faults (comparison baseline)      │
│   Disadvantages: Cannot predict future → Impossible to implement        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Optimal Implementation

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX_FRAMES 10
#define MAX_REFS 100

int frames[MAX_FRAMES];
int frame_count = 0;
int capacity;

int reference_string[MAX_REFS];
int ref_length;

// Check if page is in memory
int find_page(int page) {
    for (int i = 0; i < frame_count; i++) {
        if (frames[i] == page) return i;
    }
    return -1;
}

// Find page that will be used latest in the future
int find_victim(int current_index) {
    int victim = 0;
    int farthest = -1;

    for (int i = 0; i < frame_count; i++) {
        int page = frames[i];
        int next_use = ref_length;  // Max value if not used

        // Find when this page is used in future references
        for (int j = current_index + 1; j < ref_length; j++) {
            if (reference_string[j] == page) {
                next_use = j;
                break;
            }
        }

        if (next_use > farthest) {
            farthest = next_use;
            victim = i;
        }
    }

    return victim;
}

int optimal_access(int page, int current_index) {
    int idx = find_page(page);

    if (idx != -1) {
        printf("Page %d: hit\n", page);
        return 0;
    }

    printf("Page %d: fault", page);

    if (frame_count < capacity) {
        frames[frame_count++] = page;
        printf(" (using free frame)\n");
    } else {
        int victim_idx = find_victim(current_index);
        printf(" (replaced page %d)\n", frames[victim_idx]);
        frames[victim_idx] = page;
    }

    return 1;
}

int main() {
    capacity = 3;

    int refs[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    ref_length = sizeof(refs) / sizeof(refs[0]);
    for (int i = 0; i < ref_length; i++) {
        reference_string[i] = refs[i];
    }

    int page_faults = 0;
    for (int i = 0; i < ref_length; i++) {
        page_faults += optimal_access(reference_string[i], i);
    }

    printf("\nTotal page faults: %d\n", page_faults);
    return 0;
}
```

### 3.3 Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Optimal Example (3 frames)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2         │
│                                                                          │
│   Access  Frame 0   Frame 1   Frame 2    Fault?   Reason                │
│   ─────────────────────────────────────────────────────────────────────│
│    7      7         -          -          fault                         │
│    0      7         0          -          fault                         │
│    1      7         0          1          fault                         │
│    2      2         0          1          fault    7: not used          │
│    0      2         0          1          hit                           │
│    3      2         0          3          fault    1: used latest       │
│    0      2         0          3          hit                           │
│    4      2         4          3          fault    0: used latest       │
│    2      2         4          3          hit                           │
│    3      2         4          3          hit                           │
│    0      0         4          3          fault    2: used latest       │
│    3      0         4          3          hit                           │
│    2      0         2          3          fault    4: not used          │
│    1      0         2          1          fault    3: not used          │
│    2      0         2          1          hit                           │
│                                                                          │
│   Total page faults: 9 (3 fewer than FIFO!)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. LRU Algorithm

### 4.1 Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LRU (Least Recently Used)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Rule: Replace the page that has not been used for the longest time    │
│         (Recent use = likely to be used in near future)                 │
│                                                                          │
│   Comparison with Optimal:                                               │
│   - Optimal: Looks at future (will be unused longest)                   │
│   - LRU: Looks at past (unused longest so far)                          │
│                                                                          │
│   Utilizes Temporal Locality:                                            │
│   "Recently used data is likely to be used again in near future"        │
│                                                                          │
│   Advantages:                                                            │
│   - Performance close to Optimal                                         │
│   - Actually implementable                                               │
│                                                                          │
│   Disadvantages:                                                         │
│   - Complex implementation (need to track all access times)              │
│   - High overhead without hardware support                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 LRU Implementation Methods

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LRU Implementation Methods                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Counter-based                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Store timestamp for each page                                   │   │
│   │                                                                  │   │
│   │  On page access:                                                 │   │
│   │  page.last_used = ++global_counter;                             │   │
│   │                                                                  │   │
│   │  On replacement: select page with smallest counter value        │   │
│   │                                                                  │   │
│   │  Disadvantage: O(n) table search, overflow handling needed      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   2. Stack-based                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Manage pages with doubly linked list                           │   │
│   │                                                                  │   │
│   │  On page access: move that page to stack top                    │   │
│   │  On replacement: remove page at stack bottom                    │   │
│   │                                                                  │   │
│   │    Top                                                          │   │
│   │     ↓                                                           │   │
│   │   ┌───┐                                                         │   │
│   │   │ A │ ← Most recently used                                   │   │
│   │   ├───┤                                                         │   │
│   │   │ C │                                                         │   │
│   │   ├───┤                                                         │   │
│   │   │ B │ ← Least recently used (replacement target)             │   │
│   │   └───┘                                                         │   │
│   │   Bottom                                                        │   │
│   │                                                                  │   │
│   │  Advantage: O(1) replacement                                    │   │
│   │  Disadvantage: Pointer updates needed on move                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 LRU Stack Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Node {
    int page;
    struct Node* prev;
    struct Node* next;
} Node;

typedef struct {
    Node* head;   // Most recently used
    Node* tail;   // Least recently used
    int count;
    int capacity;
} LRUCache;

LRUCache* lru_create(int capacity) {
    LRUCache* cache = malloc(sizeof(LRUCache));
    cache->head = NULL;
    cache->tail = NULL;
    cache->count = 0;
    cache->capacity = capacity;
    return cache;
}

// Find page
Node* lru_find(LRUCache* cache, int page) {
    Node* current = cache->head;
    while (current) {
        if (current->page == page) return current;
        current = current->next;
    }
    return NULL;
}

// Move node to front
void move_to_front(LRUCache* cache, Node* node) {
    if (node == cache->head) return;  // Already at front

    // Remove from list
    if (node->prev) node->prev->next = node->next;
    if (node->next) node->next->prev = node->prev;
    if (node == cache->tail) cache->tail = node->prev;

    // Insert at front
    node->prev = NULL;
    node->next = cache->head;
    if (cache->head) cache->head->prev = node;
    cache->head = node;
    if (!cache->tail) cache->tail = node;
}

// Insert new node at front
void insert_front(LRUCache* cache, int page) {
    Node* node = malloc(sizeof(Node));
    node->page = page;
    node->prev = NULL;
    node->next = cache->head;

    if (cache->head) cache->head->prev = node;
    cache->head = node;
    if (!cache->tail) cache->tail = node;

    cache->count++;
}

// Remove tail node
int remove_tail(LRUCache* cache) {
    if (!cache->tail) return -1;

    Node* victim = cache->tail;
    int page = victim->page;

    cache->tail = victim->prev;
    if (cache->tail) cache->tail->next = NULL;
    else cache->head = NULL;

    free(victim);
    cache->count--;

    return page;
}

int lru_access(LRUCache* cache, int page) {
    Node* node = lru_find(cache, page);

    if (node) {
        // Hit: move to front
        printf("Page %d: hit\n", page);
        move_to_front(cache, node);
        return 0;
    }

    // Fault
    printf("Page %d: fault", page);

    if (cache->count >= cache->capacity) {
        // Replace least recently used page
        int victim = remove_tail(cache);
        printf(" (replaced page %d)", victim);
    }

    insert_front(cache, page);
    printf("\n");

    return 1;
}

void lru_print(LRUCache* cache) {
    printf("Memory (MRU→LRU): [");
    Node* current = cache->head;
    while (current) {
        printf("%d", current->page);
        if (current->next) printf(", ");
        current = current->next;
    }
    printf("]\n\n");
}

int main() {
    LRUCache* cache = lru_create(3);

    int refs[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    int n = sizeof(refs) / sizeof(refs[0]);

    int page_faults = 0;
    for (int i = 0; i < n; i++) {
        page_faults += lru_access(cache, refs[i]);
        lru_print(cache);
    }

    printf("Total page faults: %d\n", page_faults);
    return 0;
}
```

### 4.4 Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│               LRU Example (3 frames)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2         │
│                                                                          │
│   Access  Stack (MRU→LRU)    Fault?                                     │
│   ──────────────────────────────────────                                │
│    7      [7]              fault                                        │
│    0      [0, 7]           fault                                        │
│    1      [1, 0, 7]        fault                                        │
│    2      [2, 1, 0]        fault (replaced 7 - LRU)                     │
│    0      [0, 2, 1]        hit (moved 0 to MRU)                         │
│    3      [3, 0, 2]        fault (replaced 1 - LRU)                     │
│    0      [0, 3, 2]        hit (moved 0 to MRU)                         │
│    4      [4, 0, 3]        fault (replaced 2 - LRU)                     │
│    2      [2, 4, 0]        fault (replaced 3 - LRU)                     │
│    3      [3, 2, 4]        fault (replaced 0 - LRU)                     │
│    0      [0, 3, 2]        fault (replaced 4 - LRU)                     │
│    3      [3, 0, 2]        hit (moved 3 to MRU)                         │
│    2      [2, 3, 0]        hit (moved 2 to MRU)                         │
│    1      [1, 2, 3]        fault (replaced 0 - LRU)                     │
│    2      [2, 1, 3]        hit (moved 2 to MRU)                         │
│                                                                          │
│   Total page faults: 10 (FIFO: 12, OPT: 9)                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. LRU Approximation Algorithms

### 5.1 Second-Chance (Clock) Algorithm

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Second-Chance Algorithm                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Uses reference bit:                                                    │
│   - On page access: reference bit = 1                                   │
│   - On replacement: select page with reference bit = 0                  │
│                                                                          │
│   Implemented as circular queue (clock algorithm):                       │
│                                                                          │
│              ┌───────────────────┐                                      │
│              │                   │                                      │
│              ▼                   │                                      │
│       ┌─────────┐   ┌─────────┐  │  ┌─────────┐                        │
│       │Page A   │───│Page B   │──┼──│Page C   │─────┐                  │
│       │Ref: 1   │   │Ref: 0   │  │  │Ref: 1   │     │                  │
│       └─────────┘   └─────────┘  │  └─────────┘     │                  │
│           ▲                      │                   │                  │
│           │          ┌───────────┘                   │                  │
│           │          │                               │                  │
│           │      ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│           └──────│Page F   │───│Page E   │───│Page D   │              │
│                  │Ref: 0   │   │Ref: 1   │   │Ref: 0   │              │
│                  └─────────┘   └─────────┘   └─────────┘              │
│                       ↑                                                 │
│                    Pointer                                              │
│                                                                          │
│   Replacement process:                                                   │
│   1. Examine page pointed to                                            │
│   2. If Ref=1: set Ref=0, move to next (second chance)                 │
│   3. If Ref=0: replace this page                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Enhanced Second-Chance

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Enhanced Second-Chance Algorithm                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Uses reference bit + modify bit                                       │
│                                                                          │
│   Priority by (Ref, Mod) combination:                                   │
│   ┌────────────────┬────────────────────────────────────┬────────────┐  │
│   │    (Ref, Mod)  │              Description           │   Priority │  │
│   ├────────────────┼────────────────────────────────────┼────────────┤  │
│   │     (0, 0)     │ Not recently used, not modified    │   1 (Best) │  │
│   │     (0, 1)     │ Not recently used, modified        │   2        │  │
│   │     (1, 0)     │ Recently used, not modified        │   3        │  │
│   │     (1, 1)     │ Recently used, modified            │   4 (Worst)│  │
│   └────────────────┴────────────────────────────────────┴────────────┘  │
│                                                                          │
│   Replacement algorithm:                                                 │
│   1. Search for (0,0) page → replace if found                           │
│   2. Search for (0,1) page, set Ref=0 while passing                     │
│   3. Start over, search for (0,0)                                       │
│   4. Search for (0,1) → replace if found                                │
│                                                                          │
│   Advantage: Minimize dirty page replacement → Reduce I/O               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Clock Algorithm Implementation

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX_FRAMES 10

typedef struct {
    int page;
    bool reference_bit;
    bool valid;
} Frame;

Frame frames[MAX_FRAMES];
int clock_hand = 0;
int capacity;
int count = 0;

void clock_init(int cap) {
    capacity = cap;
    for (int i = 0; i < MAX_FRAMES; i++) {
        frames[i].valid = false;
    }
}

int find_page(int page) {
    for (int i = 0; i < capacity; i++) {
        if (frames[i].valid && frames[i].page == page) {
            return i;
        }
    }
    return -1;
}

int find_empty() {
    for (int i = 0; i < capacity; i++) {
        if (!frames[i].valid) return i;
    }
    return -1;
}

int clock_access(int page) {
    int idx = find_page(page);

    if (idx != -1) {
        // Hit: set reference bit
        printf("Page %d: hit\n", page);
        frames[idx].reference_bit = true;
        return 0;
    }

    // Fault
    printf("Page %d: fault", page);

    int empty = find_empty();
    if (empty != -1) {
        // Use free frame
        frames[empty].page = page;
        frames[empty].reference_bit = true;
        frames[empty].valid = true;
        count++;
        printf(" (using free frame %d)\n", empty);
        return 1;
    }

    // Select victim with clock algorithm
    while (true) {
        if (!frames[clock_hand].reference_bit) {
            // Replace if reference bit is 0
            printf(" (replaced page %d in frame %d)\n",
                   frames[clock_hand].page, clock_hand);

            frames[clock_hand].page = page;
            frames[clock_hand].reference_bit = true;

            clock_hand = (clock_hand + 1) % capacity;
            return 1;
        }

        // If reference bit is 1, set to 0 and move to next
        frames[clock_hand].reference_bit = false;
        clock_hand = (clock_hand + 1) % capacity;
    }
}

void clock_print() {
    printf("Memory: [");
    for (int i = 0; i < capacity; i++) {
        if (frames[i].valid) {
            printf("%d(R:%d)", frames[i].page,
                   frames[i].reference_bit ? 1 : 0);
        } else {
            printf("-");
        }
        if (i < capacity - 1) printf(", ");
    }
    printf("] Pointer: %d\n\n", clock_hand);
}

int main() {
    clock_init(3);

    int refs[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3};
    int n = sizeof(refs) / sizeof(refs[0]);

    int page_faults = 0;
    for (int i = 0; i < n; i++) {
        page_faults += clock_access(refs[i]);
        clock_print();
    }

    printf("Total page faults: %d\n", page_faults);
    return 0;
}
```

---

## 6. Belady's Anomaly

### 6.1 Phenomenon

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Belady's Anomaly                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Intuition: More frames should mean fewer page faults.                 │
│   Belady's Anomaly: With FIFO, increasing frames can increase faults!   │
│                                                                          │
│   Example: Reference string 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5          │
│                                                                          │
│   3 frames:                                                              │
│   1    [1, -, -]        fault                                           │
│   2    [1, 2, -]        fault                                           │
│   3    [1, 2, 3]        fault                                           │
│   4    [4, 2, 3]        fault                                           │
│   1    [4, 1, 3]        fault                                           │
│   2    [4, 1, 2]        fault                                           │
│   5    [5, 1, 2]        fault                                           │
│   1    [5, 1, 2]        hit                                             │
│   2    [5, 1, 2]        hit                                             │
│   3    [3, 1, 2]        fault                                           │
│   4    [3, 4, 2]        fault                                           │
│   5    [3, 4, 5]        fault  → Total 9 faults                         │
│                                                                          │
│   4 frames:                                                              │
│   1    [1, -, -, -]     fault                                           │
│   2    [1, 2, -, -]     fault                                           │
│   3    [1, 2, 3, -]     fault                                           │
│   4    [1, 2, 3, 4]     fault                                           │
│   1    [1, 2, 3, 4]     hit                                             │
│   2    [1, 2, 3, 4]     hit                                             │
│   5    [5, 2, 3, 4]     fault                                           │
│   1    [5, 1, 3, 4]     fault                                           │
│   2    [5, 1, 2, 4]     fault                                           │
│   3    [5, 1, 2, 3]     fault                                           │
│   4    [4, 1, 2, 3]     fault                                           │
│   5    [4, 5, 2, 3]     fault  → Total 10 faults                        │
│                                                                          │
│   3 frames: 9 faults < 4 frames: 10 faults  ← Anomaly!                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Stack Algorithms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Stack Algorithm                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Definition: Page set for n frames is always subset of n+1 frames      │
│        M(n) ⊆ M(n+1)                                                    │
│                                                                          │
│   Stack Algorithm = No Belady's Anomaly                                 │
│                                                                          │
│   Example: LRU                                                           │
│   3 frames: Memory = {A, B, C}                                          │
│   4 frames: Memory = {A, B, C, D}                                       │
│   → {A, B, C} ⊂ {A, B, C, D} always holds                               │
│                                                                          │
│   FIFO is not a stack algorithm:                                        │
│   At time t, 3 frames: {1, 2, 3}                                        │
│   At time t, 4 frames: {2, 3, 4}                                        │
│   → {1, 2, 3} ⊄ {2, 3, 4} possible                                      │
│                                                                          │
│   Stack algorithm examples: LRU, Optimal, LFU                            │
│   Non-stack algorithm: FIFO                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Thrashing and Working Set

### 7.1 Thrashing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Thrashing                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Definition: Process spends more time on page replacement than execution│
│                                                                          │
│   CPU Utilization                                                        │
│      ↑                                                                   │
│      │        ╱──╲                                                      │
│      │       ╱    ╲                                                     │
│      │      ╱      ╲                                                    │
│      │     ╱        ╲                                                   │
│      │    ╱          ╲                                                  │
│      │   ╱            ╲    Thrashing                                    │
│      │  ╱              ╲   threshold                                    │
│      │ ╱                ╲                                               │
│      │╱                  ╲__________                                    │
│      └──────────────────────────────▶ Degree of multiprogramming        │
│                                                                          │
│   Thrashing occurrence:                                                  │
│   1. Low CPU utilization → OS runs more processes                       │
│   2. More processes → fewer frames per process                          │
│   3. More page faults → more I/O waiting                                │
│   4. CPU utilization decreases → OS runs more processes... (vicious cycle)│
│                                                                          │
│   Result: System nearly stops working                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Working Set Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Working Set Model                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Working Set: Set of pages referenced during time window (Δ)           │
│                                                                          │
│   Time window Δ = 10                                                    │
│                                                                          │
│   Reference sequence: ... 1 2 3 4 5 1 2 3 1 2 | 7 8 9 0 7 8 9 0 7 8     │
│                               ↑ Current time                             │
│                                                                          │
│   Working Set(Δ=10) = {1, 2, 3, 4, 5}                                   │
│                                                                          │
│   Working Set size change:                                               │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │  WSS                                                            │   │
│   │   ↑      ╭──╮     ╭──╮                                         │   │
│   │   │     ╱    ╲   ╱    ╲     Locality transition                │   │
│   │   │────╱      ╲─╱      ╲────                                   │   │
│   │   │   Stable   Stable   Stable                                 │   │
│   │   └─────────────────────────────────▶ Time                     │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Principle:                                                             │
│   - WSS(i) = Process i's Working Set size                               │
│   - D = Σ WSS(i) = Total frame demand                                   │
│   - D > Total frames → Thrashing occurs                                 │
│                                                                          │
│   Solution: If D > m, suspend one process                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Page Fault Frequency (PFF)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Page Fault Frequency Method                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Concept: Adjust frame allocation based on page fault frequency        │
│                                                                          │
│   Page fault rate                                                        │
│      ↑                                                                   │
│      │  ────────── Upper threshold                                      │
│      │       ↑                                                          │
│      │       │ Allocate more frames                                     │
│      │       │                                                          │
│      │   ─ ─ ─ ─ ─ ─  Target range                                     │
│      │       │                                                          │
│      │       │ Reclaim frames                                           │
│      │       ↓                                                          │
│      │  ────────── Lower threshold                                      │
│      └─────────────────────────▶ Allocated frames                       │
│                                                                          │
│   Operation:                                                             │
│   - Fault rate > upper: allocate more frames                            │
│   - Fault rate < lower: reclaim some frames                             │
│   - Insufficient frames: swap out some processes                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Preventing Thrashing in Linux

```bash
# Check swap usage
$ free -h
              total        used        free      shared  buff/cache   available
Mem:           15Gi       10Gi       500Mi       100Mi       5.0Gi       4.5Gi
Swap:          8.0Gi       2.0Gi       6.0Gi  ← Swap in use

# Check swappiness (0-100, higher = more aggressive swap)
$ cat /proc/sys/vm/swappiness
60

# Adjust swappiness (lower to prevent thrashing)
$ sudo sysctl vm.swappiness=10

# Check OOM Killer logs
$ dmesg | grep -i "out of memory"

# Memory usage by process
$ ps aux --sort=-%mem | head -10

# Limit memory with cgroups (containers)
$ cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

---

## Practice Problems

### Problem 1: Algorithm Comparison
Calculate page faults for reference string 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5 with 3 frames using FIFO, LRU, and Optimal.

<details>
<summary>Show Answer</summary>

```
FIFO:
1: [1,-,-] fault    5: [5,2,3] fault    4: [4,5,2] fault
2: [1,2,-] fault    1: [5,1,3] fault    5: [4,5,2] hit
3: [1,2,3] fault    2: [5,1,2] fault
4: [4,2,3] fault    3: [3,1,2] fault
1: [4,1,3] fault
2: [4,1,2] fault
Total: 9 faults

LRU:
1: [1] fault        5: [5,1,2] fault    4: [4,3,2] fault
2: [2,1] fault      1: [1,5,2] hit      5: [5,4,3] fault
3: [3,2,1] fault    2: [2,1,5] hit
4: [4,3,2] fault    3: [3,2,1] fault
1: [1,4,3] fault
2: [2,1,4] fault
Total: 10 faults

Optimal:
1: [1,-,-] fault    5: [5,1,2] fault    4: [4,1,2] fault
2: [1,2,-] fault    1: [5,1,2] hit      5: [5,1,2] fault
3: [1,2,3] fault    2: [5,1,2] hit
4: [4,2,3] fault    3: [3,1,2] fault
1: [4,1,3] fault
2: [4,1,2] fault
Total: 7 faults
```

</details>

### Problem 2: Second-Chance
With 4 frames in the following state, which page gets replaced when inserting page E?

```
Pointer → [A, R=1] → [B, R=0] → [C, R=1] → [D, R=0]
```

<details>
<summary>Show Answer</summary>

```
1. Pointer at A, R=1
   → Set R=0, move to next

2. Pointer at B, R=0
   → B gets replaced!

Result: [A, R=0] → [E, R=1] → [C, R=1] → [D, R=0]
                  ↑ New page

Pointer now points to C
```

</details>

### Problem 3: Working Set
Calculate Working Set at time t=10 with Δ = 5.

```
Time:   1  2  3  4  5  6  7  8  9  10
Page:   1  2  3  1  2  1  3  4  5   2
```

<details>
<summary>Show Answer</summary>

```
References within Δ=5 before t=10:
t=6: Page 1
t=7: Page 3
t=8: Page 4
t=9: Page 5
t=10: Page 2

Working Set(t=10, Δ=5) = {1, 2, 3, 4, 5}
WSS = 5

This process needs at least 5 frames
```

</details>

### Problem 4: Belady's Anomaly Proof
Calculate page faults for FIFO with 3 frames and 4 frames to verify Belady's Anomaly.

Reference string: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5

<details>
<summary>Show Answer</summary>

```
3 frames (FIFO):
1: [1,-,-] F    1: [4,1,3] F    3: [3,1,2] F
2: [1,2,-] F    2: [4,1,2] F    4: [3,4,2] F
3: [1,2,3] F    5: [5,1,2] F    5: [3,4,5] F
4: [4,2,3] F    1: [5,1,2] H
                2: [5,1,2] H
Total: 9 faults

4 frames (FIFO):
1: [1,-,-,-] F    5: [5,2,3,4] F    4: [4,1,2,3] F
2: [1,2,-,-] F    1: [5,1,3,4] F    5: [4,5,2,3] F
3: [1,2,3,-] F    2: [5,1,2,4] F
4: [1,2,3,4] F    3: [5,1,2,3] F
1: [1,2,3,4] H
2: [1,2,3,4] H
Total: 10 faults

Belady's Anomaly confirmed:
3 frames: 9 faults
4 frames: 10 faults
→ More frames but more faults!
```

</details>

### Problem 5: Thrashing Solution
The following symptoms are observed in the system. Analyze the cause and propose solutions.

- CPU utilization: 5%
- Disk I/O: 95%
- Memory: Nearly full
- Many processes waiting

<details>
<summary>Show Answer</summary>

```
Cause Analysis:
- Classic thrashing state
- Too many processes competing for insufficient memory
- Most time spent on page fault handling (disk I/O)

Solutions:

1. Immediate fix:
   - Suspend some processes
   - Swap out to free frames for other processes

2. System configuration:
   - Lower swappiness (vm.swappiness=10)
   - Limit memory overcommit (vm.overcommit_memory=2)

3. Long-term solutions:
   - Add physical memory
   - Implement Working Set-based scheduling
   - Monitor PFF (Page Fault Frequency)
   - Limit per-process memory with cgroups

4. Application level:
   - Check for memory leaks
   - Adjust cache sizes
   - Terminate unnecessary processes

Monitoring commands:
$ vmstat 1   # Check si/so (swap in/out)
$ sar -B 1   # Check pgpgin/pgpgout
```

</details>

---

## Next Steps

Continue to [16_File_System_Basics.md](./16_File_System_Basics.md) to learn file system basics!

---

## References

- Silberschatz, "Operating System Concepts" Chapter 10
- Tanenbaum, "Modern Operating Systems" Chapter 3
- Linux kernel source: `mm/vmscan.c`, `mm/workingset.c`
- Belady, L.A. "A study of replacement algorithms for virtual-storage computer"
