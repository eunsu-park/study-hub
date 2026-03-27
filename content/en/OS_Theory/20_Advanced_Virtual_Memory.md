[Previous: I/O and IPC](./18_IO_and_IPC.md)

---

# 20. Advanced Virtual Memory

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain TLB management strategies including multi-level TLBs and TLB shootdown
2. Implement huge page allocation and describe its performance benefits
3. Analyze NUMA topology and its impact on memory allocation decisions
4. Describe memory-mapped I/O mechanisms and their use in device drivers
5. Profile and optimize application memory access patterns for modern hardware

---

## Table of Contents

1. [TLB Deep Dive](#1-tlb-deep-dive)
2. [Huge Pages](#2-huge-pages)
3. [NUMA Architecture](#3-numa-architecture)
4. [Memory-Mapped I/O](#4-memory-mapped-io)
5. [Kernel Memory Management](#5-kernel-memory-management)
6. [Copy-on-Write and Memory Sharing](#6-copy-on-write-and-memory-sharing)
7. [Memory Performance Optimization](#7-memory-performance-optimization)
8. [Exercises](#8-exercises)

---

## 1. TLB Deep Dive

### 1.1 TLB Structure and Operation

```
TLB (Translation Lookaside Buffer):
  A cache for page table entries (virtual → physical mappings).

Without TLB:
  Virtual address → Walk page table (4 memory accesses for 4-level!)
  → Physical address
  Time: ~200 cycles

With TLB (hit):
  Virtual address → TLB lookup (1 cycle) → Physical address
  Time: ~1 cycle

TLB hit rate is CRITICAL for performance.
Typical hit rate: 99%+ for well-behaved workloads.

TLB Structure:
  ┌──────────────────────────────────────────┐
  │ VPN (Virtual Page Number) │ PPN │ Flags  │
  ├──────────────────────────────────────────┤
  │ 0x7fff1000                │ 0x3a2│ RWXU  │
  │ 0x400000                  │ 0x1f8│ RX-U  │
  │ 0x601000                  │ 0x2c1│ RW-U  │
  │ ...                       │ ...  │ ...   │
  └──────────────────────────────────────────┘
```

### 1.2 Multi-Level TLBs

```
Modern CPUs use hierarchical TLBs:

L1 ITLB (Instructions):  64-128 entries, 1 cycle
L1 DTLB (Data):          64-128 entries, 1 cycle
L2 Unified TLB:          1024-4096 entries, ~7 cycles
Page Table Walk:          ~200 cycles (via page walker hardware)

                    ┌────────┐
   Virtual Addr ───▶│ L1 TLB │──hit──▶ Physical Addr (1 cycle)
                    └───┬────┘
                       miss
                    ┌───▼────┐
                    │ L2 TLB │──hit──▶ Physical Addr (7 cycles)
                    └───┬────┘
                       miss
                    ┌───▼──────────┐
                    │ Page Walker  │──▶ Physical Addr (200 cycles)
                    │ (Hardware)   │
                    └──────────────┘
```

### 1.3 TLB Shootdown

```c
/*
 * TLB Shootdown: When a page mapping changes on a multiprocessor system,
 * ALL cores caching that mapping must invalidate it.
 *
 * This requires an Inter-Processor Interrupt (IPI) - expensive!
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * Simulate TLB impact measurement.
 * In real kernels, TLB shootdown uses:
 *   - invlpg instruction (invalidate single entry)
 *   - cr3 reload (flush entire TLB)
 *   - IPI to notify other cores
 */

void demonstrate_tlb_impact(void) {
    const size_t PAGE_SIZE = 4096;
    const size_t NUM_PAGES = 256;
    const size_t TOTAL_SIZE = PAGE_SIZE * NUM_PAGES;

    /* Allocate memory */
    char *mem = mmap(NULL, TOTAL_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        return;
    }

    /* Sequential access: good TLB behavior (few unique pages) */
    volatile char sink;
    for (int iter = 0; iter < 1000; iter++) {
        for (size_t i = 0; i < TOTAL_SIZE; i += 64) {
            sink = mem[i];
        }
    }

    /* Random access: poor TLB behavior (many unique pages) */
    size_t *random_offsets = malloc(NUM_PAGES * sizeof(size_t));
    for (size_t i = 0; i < NUM_PAGES; i++) {
        random_offsets[i] = (rand() % NUM_PAGES) * PAGE_SIZE;
    }

    for (int iter = 0; iter < 1000; iter++) {
        for (size_t i = 0; i < NUM_PAGES; i++) {
            sink = mem[random_offsets[i]];
        }
    }

    free(random_offsets);
    munmap(mem, TOTAL_SIZE);
}
```

---

## 2. Huge Pages

### 2.1 Why Huge Pages?

```
Standard page size: 4 KB
Huge page sizes: 2 MB (x86), 1 GB (x86), 64 KB (ARM)

Problem with small pages:
  Application uses 1 GB of memory:
  4 KB pages: 262,144 pages → 262,144 TLB entries needed
  But TLB only holds ~1000-4000 entries!
  → Constant TLB misses → constant page walks → SLOW

With huge pages:
  2 MB pages: 512 pages → 512 TLB entries
  1 GB pages: 1 page → 1 TLB entry!
  → TLB covers entire working set → FAST

Performance improvement: 10-30% for memory-intensive workloads.
```

### 2.2 Using Huge Pages in Linux

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)  /* 2 MB */

/*
 * Method 1: mmap with MAP_HUGETLB
 */
void *allocate_huge_mmap(size_t size) {
    /* Round up to huge page boundary */
    size_t aligned = (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);

    void *ptr = mmap(NULL, aligned,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                     -1, 0);

    if (ptr == MAP_FAILED) {
        perror("mmap(MAP_HUGETLB)");
        return NULL;
    }

    printf("Allocated %zu bytes with huge pages at %p\n", aligned, ptr);
    return ptr;
}

/*
 * Method 2: madvise with MADV_HUGEPAGE (Transparent Huge Pages)
 */
void *allocate_thp(size_t size) {
    size_t aligned = (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);

    void *ptr = mmap(NULL, aligned,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);

    if (ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    /* Hint to kernel: use transparent huge pages */
    if (madvise(ptr, aligned, MADV_HUGEPAGE) != 0) {
        perror("madvise(MADV_HUGEPAGE)");
    }

    printf("Allocated %zu bytes with THP hint at %p\n", aligned, ptr);
    return ptr;
}

/*
 * Benchmark: compare 4K pages vs huge pages
 */
void benchmark_page_sizes(void) {
    const size_t SIZE = 256 * 1024 * 1024;  /* 256 MB */
    const int ITERATIONS = 10;

    /* Allocate with regular pages */
    char *regular = mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    /* Allocate with huge pages */
    char *huge = allocate_huge_mmap(SIZE);

    if (!regular || !huge) return;

    /* Touch all pages (fault them in) */
    memset(regular, 0, SIZE);
    if (huge) memset(huge, 0, SIZE);

    /* Random access benchmark */
    volatile char sink;
    size_t stride = 4096;  /* Access one per page */

    printf("Random access benchmark (%zu MB):\n", SIZE / (1024*1024));

    /* Regular pages */
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (size_t off = 0; off < SIZE; off += stride) {
            size_t idx = ((off * 2654435761UL) % SIZE) & ~(stride - 1);
            sink = regular[idx];
        }
    }
    printf("  Regular pages: done\n");

    /* Huge pages */
    if (huge) {
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (size_t off = 0; off < SIZE; off += stride) {
                size_t idx = ((off * 2654435761UL) % SIZE) & ~(stride - 1);
                sink = huge[idx];
            }
        }
        printf("  Huge pages: done\n");
    }

    munmap(regular, SIZE);
    if (huge) munmap(huge, SIZE);
}

int main(void) {
    benchmark_page_sizes();
    return 0;
}
```

---

## 3. NUMA Architecture

### 3.1 NUMA Topology

```
UMA (Uniform Memory Access) - older systems:
  All CPUs access all memory with EQUAL latency.

    CPU0  CPU1  CPU2  CPU3
      \    |    |    /
       ┌───┴────┴───┐
       │  Memory Bus │
       └───────┬─────┘
               │
          ┌────┴────┐
          │ Memory   │
          └──────────┘

NUMA (Non-Uniform Memory Access) - modern servers:
  Each CPU has LOCAL memory (fast) and REMOTE memory (slow).

    ┌─────────────┐         ┌─────────────┐
    │   Node 0     │ QPI/UPI │   Node 1     │
    │ CPU0  CPU1  │◄────────▶│ CPU2  CPU3  │
    │ ┌─────────┐ │         │ ┌─────────┐ │
    │ │ Memory  │ │         │ │ Memory  │ │
    │ │ (Local) │ │         │ │ (Local) │ │
    │ └─────────┘ │         │ └─────────┘ │
    └─────────────┘         └─────────────┘

  Local access:  ~80 ns
  Remote access: ~140 ns (1.75x slower!)
```

### 3.2 NUMA-Aware Programming

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <sched.h>

/*
 * Compile: gcc -o numa_demo numa_demo.c -lnuma
 */

void demonstrate_numa(void) {
    if (numa_available() < 0) {
        printf("NUMA not available\n");
        return;
    }

    /* Query NUMA topology */
    int num_nodes = numa_num_configured_nodes();
    int num_cpus = numa_num_configured_cpus();
    printf("NUMA nodes: %d, CPUs: %d\n", num_nodes, num_cpus);

    for (int node = 0; node < num_nodes; node++) {
        long free_mem;
        long total = numa_node_size(node, &free_mem);
        printf("Node %d: %ld MB total, %ld MB free\n",
               node, total / (1024*1024), free_mem / (1024*1024));

        /* Which CPUs belong to this node? */
        struct bitmask *cpus = numa_allocate_cpumask();
        numa_node_to_cpus(node, cpus);
        printf("  CPUs: ");
        for (int cpu = 0; cpu < num_cpus; cpu++) {
            if (numa_bitmask_isbitset(cpus, cpu)) {
                printf("%d ", cpu);
            }
        }
        printf("\n");
        numa_free_cpumask(cpus);
    }

    /* Allocate on specific NUMA node */
    size_t size = 1024 * 1024;  /* 1 MB */

    void *local = numa_alloc_onnode(size, 0);
    printf("\nAllocated on node 0: %p\n", local);

    void *interleaved = numa_alloc_interleaved(size);
    printf("Interleaved allocation: %p\n", interleaved);

    /* Bind thread to specific node */
    numa_run_on_node(0);
    printf("Thread bound to node 0\n");

    numa_free(local, size);
    numa_free(interleaved, size);
}

/*
 * NUMA-aware data structure layout
 */
typedef struct {
    int *data;
    size_t size;
    int numa_node;
} numa_array_t;

numa_array_t *create_numa_array(size_t count, int node) {
    numa_array_t *arr = malloc(sizeof(numa_array_t));
    arr->size = count;
    arr->numa_node = node;

    /* Allocate data on specific NUMA node */
    arr->data = numa_alloc_onnode(count * sizeof(int), node);
    if (!arr->data) {
        free(arr);
        return NULL;
    }

    return arr;
}
```

---

## 4. Memory-Mapped I/O

### 4.1 mmap for File I/O

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

/*
 * Memory-mapped file I/O:
 * Map file contents directly into virtual address space.
 * No explicit read()/write() calls needed!
 *
 * Process virtual memory:
 * ┌──────────────┐
 * │    Stack      │
 * ├──────────────┤
 * │    ...        │
 * ├──────────────┤
 * │  mmap region  │◄── File contents mapped here
 * │  (file.dat)   │    Reads/writes go directly to file
 * ├──────────────┤
 * │    Heap       │
 * ├──────────────┤
 * │    Code       │
 * └──────────────┘
 */

void mmap_file_example(const char *filename) {
    /* Open file */
    int fd = open(filename, O_RDWR);
    if (fd < 0) {
        perror("open");
        return;
    }

    /* Get file size */
    struct stat sb;
    fstat(fd, &sb);
    size_t size = sb.st_size;

    /* Map file into memory */
    char *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    /* Now we can access file contents as if it were memory! */
    printf("First 100 bytes: %.100s\n", mapped);

    /* Modify file by writing to memory */
    mapped[0] = 'H';
    mapped[1] = 'i';

    /* Flush changes to disk */
    msync(mapped, size, MS_SYNC);

    /* Cleanup */
    munmap(mapped, size);
    close(fd);
}

/*
 * Performance comparison: read() vs mmap
 */
void compare_io_methods(const char *filename) {
    struct stat sb;
    stat(filename, &sb);
    size_t size = sb.st_size;

    /* Method 1: Traditional read() */
    int fd = open(filename, O_RDONLY);
    char *buf = malloc(size);
    read(fd, buf, size);

    /* Process: count newlines */
    long count1 = 0;
    for (size_t i = 0; i < size; i++) {
        if (buf[i] == '\n') count1++;
    }
    free(buf);
    close(fd);

    /* Method 2: mmap */
    fd = open(filename, O_RDONLY);
    char *mapped = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    /* Hint: we'll read sequentially */
    madvise(mapped, size, MADV_SEQUENTIAL);

    long count2 = 0;
    for (size_t i = 0; i < size; i++) {
        if (mapped[i] == '\n') count2++;
    }

    munmap(mapped, size);
    close(fd);

    printf("read():  %ld newlines\n", count1);
    printf("mmap():  %ld newlines\n", count2);
}
```

### 4.2 Device Memory-Mapped I/O

```c
/*
 * Device MMIO: Hardware registers mapped into virtual address space.
 * Used by device drivers to communicate with hardware.
 *
 * Physical memory map:
 * 0x00000000 - 0x7FFFFFFF: RAM
 * 0xE0000000 - 0xE0000FFF: GPU registers (MMIO)
 * 0xF0000000 - 0xF000FFFF: Network card registers (MMIO)
 *
 * Reading/writing to these addresses talks to hardware!
 */

#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

/* Example: Reading a hardware register via MMIO */
void read_device_register(off_t phys_addr, size_t size) {
    int fd = open("/dev/mem", O_RDONLY);
    if (fd < 0) {
        perror("open /dev/mem");
        return;
    }

    /* Map physical address into our virtual space */
    volatile uint32_t *regs = mmap(NULL, size,
                                    PROT_READ,
                                    MAP_SHARED,
                                    fd, phys_addr);
    if (regs == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    /* Read register (volatile ensures no optimization) */
    uint32_t status = regs[0];
    printf("Device status register: 0x%08x\n", status);

    munmap((void *)regs, size);
    close(fd);
}
```

---

## 5. Kernel Memory Management

### 5.1 Slab Allocator

```
Linux kernel memory allocators:

Buddy System:
  Allocates memory in power-of-2 page chunks.
  Good for large allocations, wastes memory for small objects.

Slab Allocator:
  Caches frequently-allocated objects of the same size.
  Avoids repeated initialization/destruction.

  ┌──────────────────────────────────┐
  │          kmem_cache               │
  │  (e.g., "task_struct" cache)      │
  ├──────────────────────────────────┤
  │  Slab 1: [obj][obj][obj][obj]    │ ← Full
  │  Slab 2: [obj][obj][   ][   ]    │ ← Partial
  │  Slab 3: [   ][   ][   ][   ]    │ ← Empty
  └──────────────────────────────────┘

  Benefits:
  - No fragmentation for fixed-size objects
  - Cached objects avoid re-initialization
  - NUMA-aware: per-node slab lists
```

### 5.2 OOM Killer

```c
/*
 * Linux OOM (Out of Memory) Killer:
 * When system runs out of memory, kernel kills processes.
 *
 * OOM score: /proc/<pid>/oom_score
 *   Higher score = more likely to be killed
 *
 * Factors:
 *   - Memory usage (higher = more likely)
 *   - oom_score_adj (-1000 to 1000, user-configurable)
 *   - Process age (newer processes more likely)
 */

#include <stdio.h>
#include <stdlib.h>

void show_oom_info(pid_t pid) {
    char path[256];
    FILE *fp;

    /* Read OOM score */
    snprintf(path, sizeof(path), "/proc/%d/oom_score", pid);
    fp = fopen(path, "r");
    if (fp) {
        int score;
        fscanf(fp, "%d", &score);
        printf("PID %d OOM score: %d\n", pid, score);
        fclose(fp);
    }

    /* Read OOM adjustment */
    snprintf(path, sizeof(path), "/proc/%d/oom_score_adj", pid);
    fp = fopen(path, "r");
    if (fp) {
        int adj;
        fscanf(fp, "%d", &adj);
        printf("PID %d OOM adj: %d\n", pid, adj);
        fclose(fp);
    }
}

/*
 * Protect critical process from OOM:
 * echo -1000 > /proc/<pid>/oom_score_adj
 */
```

---

## 6. Copy-on-Write and Memory Sharing

### 6.1 COW Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

/*
 * Copy-on-Write (COW):
 * After fork(), parent and child SHARE the same physical pages.
 * Pages are marked read-only.
 * Only when one process WRITES does the kernel copy the page.
 *
 * Before write:
 *   Parent VAS ──▶ Physical Page ◀── Child VAS
 *                  (read-only)
 *
 * After child writes:
 *   Parent VAS ──▶ Physical Page (original)
 *   Child VAS  ──▶ Physical Page (COPY, now writable)
 */

void demonstrate_cow(void) {
    /* Allocate 100 MB */
    size_t size = 100 * 1024 * 1024;
    char *data = malloc(size);
    memset(data, 'A', size);  /* Touch all pages */

    printf("Parent: allocated %zu MB\n", size / (1024*1024));

    pid_t pid = fork();

    if (pid == 0) {
        /* Child process */
        /* At this point, child shares ALL pages with parent via COW */
        printf("Child: shares pages with parent (COW)\n");

        /* Only modify first page - only this page gets copied */
        data[0] = 'B';
        printf("Child: modified 1 page (1 physical copy)\n");

        /* The other 25599 pages are still shared! */
        printf("Child: 99.996%% of memory still shared\n");

        free(data);
        _exit(0);
    } else {
        wait(NULL);
        printf("Parent: data[0] still = '%c' (unchanged)\n", data[0]);
        free(data);
    }
}

int main(void) {
    demonstrate_cow();
    return 0;
}
```

---

## 7. Memory Performance Optimization

### 7.1 Cache-Friendly Access Patterns

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096

/*
 * Row-major vs column-major access:
 * Arrays are stored row-by-row in memory.
 * Accessing by row = sequential = cache-friendly.
 * Accessing by column = strided = cache-unfriendly.
 */

void row_major_access(int matrix[N][N]) {
    long sum = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += matrix[i][j];  /* Sequential access: FAST */
        }
    }
}

void col_major_access(int matrix[N][N]) {
    long sum = 0;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            sum += matrix[i][j];  /* Strided access: SLOW */
        }
    }
}

void benchmark_access_patterns(void) {
    int (*matrix)[N] = malloc(N * N * sizeof(int));

    /* Initialize */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = i + j;

    clock_t start, end;

    start = clock();
    for (int iter = 0; iter < 10; iter++)
        row_major_access(matrix);
    end = clock();
    printf("Row-major: %.3f s\n",
           (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    for (int iter = 0; iter < 10; iter++)
        col_major_access(matrix);
    end = clock();
    printf("Col-major: %.3f s\n",
           (double)(end - start) / CLOCKS_PER_SEC);

    free(matrix);
}

int main(void) {
    benchmark_access_patterns();
    return 0;
}
```

### 7.2 Memory Prefetching

```c
#include <immintrin.h>

/*
 * Software prefetching hints:
 * Tell the CPU to start loading data before we need it.
 */
void prefetch_example(int *data, size_t n) {
    long sum = 0;
    const int PREFETCH_DISTANCE = 16;  /* 16 elements ahead */

    for (size_t i = 0; i < n; i++) {
        /* Prefetch data we'll need soon */
        if (i + PREFETCH_DISTANCE < n) {
            _mm_prefetch(&data[i + PREFETCH_DISTANCE], _MM_HINT_T0);
        }
        sum += data[i];
    }
}
```

---

## 8. Exercises

### Exercise 1: TLB Performance Measurement

Measure TLB impact on your system:
1. Write a program that accesses N pages in random order
2. Vary N from 10 to 100,000 pages
3. Measure access time per element
4. Plot: access time vs number of unique pages
5. Identify the TLB capacity (where performance drops)

### Exercise 2: Huge Page Benchmark

Compare regular vs huge pages:
1. Allocate 1 GB with regular pages (4 KB)
2. Allocate 1 GB with huge pages (2 MB)
3. Perform random access benchmark on both
4. Use `perf stat` to measure TLB misses for each
5. Calculate the performance improvement from huge pages

### Exercise 3: NUMA-Aware Allocator

Build a NUMA-aware memory allocator:
1. Detect NUMA topology using libnuma
2. Implement `numa_malloc(size, node)` that allocates on specific node
3. Benchmark: local allocation vs remote allocation
4. Implement interleaved allocation for large shared buffers
5. Show the latency difference between local and remote access

### Exercise 4: Memory-Mapped File Processing

Build a high-performance file processor using mmap:
1. Create a 1 GB test file with random integers
2. Process with read(): count values > threshold
3. Process with mmap(): same operation
4. Compare: throughput, system calls (strace), page faults
5. Add madvise() hints and measure improvement

### Exercise 5: COW Fork Analyzer

Analyze copy-on-write behavior:
1. Allocate varying amounts of memory (100 MB, 500 MB, 1 GB)
2. Fork and measure fork() time
3. In child: modify 0%, 10%, 50%, 100% of pages
4. Monitor RSS (resident set size) of parent and child
5. Plot memory usage over time showing COW page faults

---

*End of Lesson 20*
