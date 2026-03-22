# Advanced Memory Management

**Previous**: [Advanced Pointers](./01_Advanced_Pointers.md) | **Next**: [Bit Operations](./03_Bit_Operations.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Diagram the process memory layout (text, data, BSS, heap, stack)
2. Implement a simple memory pool allocator for fixed-size objects
3. Use memory-mapped files with mmap for efficient file access
4. Apply Valgrind and AddressSanitizer to detect memory corruption
5. Explain fragmentation (internal/external) and mitigation strategies

---

Every C program runs inside a process whose address space is divided into well-defined segments: text, data, BSS, heap, and stack. Understanding this layout -- and the trade-offs between stack allocation, heap allocation, and memory-mapped files -- is essential for writing programs that are both correct and efficient. This lesson takes you beyond `malloc` and `free` into the territory of custom allocators, memory-mapped I/O, and professional debugging tools.

**Difficulty**: Advanced

---

## 1. Process Memory Layout

### The Five Segments

```
High Address
+---------------------------+
|        Stack              |  <- Local variables, function frames
|        (grows down)       |     Automatic allocation/deallocation
+---------------------------+
|           |               |
|           v               |
|                           |
|           ^               |
|           |               |
+---------------------------+
|        Heap               |  <- malloc/free, dynamic allocation
|        (grows up)         |     Programmer-managed lifetime
+---------------------------+
|        BSS                |  <- Uninitialized global/static variables
|        (zero-initialized) |     e.g., static int count;
+---------------------------+
|        Data               |  <- Initialized global/static variables
|        (read-write)       |     e.g., int limit = 100;
+---------------------------+
|        Text               |  <- Compiled machine code
|        (read-only)        |     String literals also here
+---------------------------+
Low Address
```

### Verifying the Layout

```c
#include <stdio.h>
#include <stdlib.h>

int global_init = 42;            // Data segment
int global_uninit;               // BSS segment
static int static_var = 10;      // Data segment

int main(void) {
    int stack_var = 1;           // Stack
    int *heap_var = malloc(4);   // Heap

    printf("Text  (main):         %p\n", (void*)main);
    printf("Data  (global_init):  %p\n", (void*)&global_init);
    printf("Data  (static_var):   %p\n", (void*)&static_var);
    printf("BSS   (global_uninit):%p\n", (void*)&global_uninit);
    printf("Heap  (heap_var):     %p\n", (void*)heap_var);
    printf("Stack (stack_var):    %p\n", (void*)&stack_var);

    free(heap_var);
    return 0;
}
```

---

## 2. Stack vs Heap Deep Dive

### Stack Frame Anatomy

Each function call creates a stack frame containing:

```
+---------------------------+
|  Return address           |  <- Where to resume after return
+---------------------------+
|  Saved frame pointer      |  <- Caller's base pointer (rbp)
+---------------------------+
|  Local variables          |  <- int x, char buf[64], etc.
+---------------------------+
|  Function arguments       |  <- Parameters passed to callee
+---------------------------+
```

```c
#include <stdio.h>

void deep_call(int depth) {
    char buffer[1024];  // 1 KB per frame
    printf("Depth %d: buffer at %p\n", depth, (void*)buffer);

    if (depth < 10) {
        deep_call(depth + 1);  // Each call adds ~1 KB to stack
    }
}

int main(void) {
    deep_call(0);
    return 0;
}
```

### Stack Overflow

```c
// Dangerous: unbounded recursion
void infinite_recursion(void) {
    char buffer[4096];  // 4 KB per frame
    infinite_recursion();  // Stack overflow!
}

// Dangerous: large stack allocation
void large_stack_alloc(void) {
    char huge[10 * 1024 * 1024];  // 10 MB on stack -> likely crash
    huge[0] = 'A';
}
```

### Stack vs Heap Comparison

| Property | Stack | Heap |
|----------|-------|------|
| Allocation speed | Very fast (pointer bump) | Slower (free-list search) |
| Deallocation | Automatic (scope exit) | Manual (`free`) |
| Size limit | Small (typically 1-8 MB) | Large (limited by RAM + swap) |
| Fragmentation | None | Internal and external |
| Thread safety | Each thread has own stack | Shared, needs synchronization |
| Lifetime | Tied to function scope | Until explicitly freed |

---

## 3. Memory-Mapped Files

### mmap/munmap Basics

Memory-mapped files let you access file contents as if they were in memory, avoiding explicit `read`/`write` calls.

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main(void) {
    // Open file
    int fd = open("example.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return 1;
    }

    // Map file into memory
    char *mapped = mmap(NULL, sb.st_size,
                        PROT_READ,       // Read-only access
                        MAP_PRIVATE,     // Private copy-on-write
                        fd, 0);          // File descriptor, offset
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // Access file contents directly through pointer
    printf("First 100 bytes:\n");
    write(STDOUT_FILENO, mapped, sb.st_size < 100 ? sb.st_size : 100);
    printf("\n");

    // Clean up
    munmap(mapped, sb.st_size);
    close(fd);
    return 0;
}
```

### Shared vs Private Mappings

| Flag | Behavior | Use Case |
|------|----------|----------|
| `MAP_PRIVATE` | Copy-on-write; changes are private | Reading files, loading libraries |
| `MAP_SHARED` | Changes are visible to other processes and written to file | IPC, database files |
| `MAP_ANONYMOUS` | Not backed by a file; initialized to zero | Custom allocators, large buffers |

### Anonymous Mapping (Large Allocation)

```c
#include <sys/mman.h>
#include <stdio.h>

int main(void) {
    size_t size = 1024 * 1024;  // 1 MB

    // Allocate 1 MB of zero-initialized memory without a file
    void *block = mmap(NULL, size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
    if (block == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // Use the memory
    int *arr = (int *)block;
    arr[0] = 42;
    printf("arr[0] = %d\n", arr[0]);

    // Release
    munmap(block, size);
    return 0;
}
```

---

## 4. Custom Allocators

### Arena (Bump) Allocator

The simplest allocator: bump a pointer forward for each allocation, free everything at once.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *buffer;    // Backing memory
    size_t capacity; // Total size
    size_t offset;   // Current position
} Arena;

Arena *arena_create(size_t capacity) {
    Arena *arena = malloc(sizeof(Arena));
    if (!arena) return NULL;

    arena->buffer = malloc(capacity);
    if (!arena->buffer) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->offset = 0;
    return arena;
}

void *arena_alloc(Arena *arena, size_t size) {
    // Align to 8 bytes
    size_t aligned = (size + 7) & ~7;

    if (arena->offset + aligned > arena->capacity) {
        return NULL;  // Out of memory
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += aligned;
    return ptr;
}

void arena_reset(Arena *arena) {
    arena->offset = 0;  // "Free" everything at once
}

void arena_destroy(Arena *arena) {
    free(arena->buffer);
    free(arena);
}

int main(void) {
    Arena *arena = arena_create(4096);

    // Allocate from arena -- no individual free needed
    int *nums = arena_alloc(arena, 10 * sizeof(int));
    char *name = arena_alloc(arena, 64);

    for (int i = 0; i < 10; i++) nums[i] = i * i;
    strcpy(name, "Arena allocator demo");

    printf("nums[5] = %d\n", nums[5]);
    printf("name = %s\n", name);
    printf("Used: %zu / %zu bytes\n", arena->offset, arena->capacity);

    arena_reset(arena);  // Free all allocations at once
    printf("After reset: %zu / %zu bytes\n", arena->offset, arena->capacity);

    arena_destroy(arena);
    return 0;
}
```

### Memory Pool for Fixed-Size Blocks

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct PoolBlock {
    struct PoolBlock *next;  // Free list link
} PoolBlock;

typedef struct {
    void *memory;          // Backing buffer
    PoolBlock *free_list;  // Head of free list
    size_t block_size;     // Size of each block
    size_t block_count;    // Total number of blocks
    size_t used_count;     // Currently allocated blocks
} MemoryPool;

MemoryPool *pool_create(size_t block_size, size_t block_count) {
    // Ensure block_size is at least sizeof(PoolBlock*)
    if (block_size < sizeof(PoolBlock)) {
        block_size = sizeof(PoolBlock);
    }

    MemoryPool *pool = malloc(sizeof(MemoryPool));
    if (!pool) return NULL;

    pool->memory = malloc(block_size * block_count);
    if (!pool->memory) {
        free(pool);
        return NULL;
    }

    pool->block_size = block_size;
    pool->block_count = block_count;
    pool->used_count = 0;

    // Build free list: chain all blocks together
    pool->free_list = NULL;
    for (size_t i = 0; i < block_count; i++) {
        PoolBlock *block = (PoolBlock *)((char *)pool->memory + i * block_size);
        block->next = pool->free_list;
        pool->free_list = block;
    }

    return pool;
}

void *pool_alloc(MemoryPool *pool) {
    if (!pool->free_list) return NULL;  // Pool exhausted

    PoolBlock *block = pool->free_list;
    pool->free_list = block->next;
    pool->used_count++;

    memset(block, 0, pool->block_size);  // Zero-initialize
    return block;
}

void pool_free(MemoryPool *pool, void *ptr) {
    if (!ptr) return;

    PoolBlock *block = (PoolBlock *)ptr;
    block->next = pool->free_list;
    pool->free_list = block;
    pool->used_count--;
}

void pool_destroy(MemoryPool *pool) {
    free(pool->memory);
    free(pool);
}

// Example: pool of fixed-size structs
typedef struct {
    int id;
    double value;
    char name[32];
} Record;

int main(void) {
    MemoryPool *pool = pool_create(sizeof(Record), 100);

    Record *r1 = pool_alloc(pool);
    Record *r2 = pool_alloc(pool);
    Record *r3 = pool_alloc(pool);

    r1->id = 1; r1->value = 3.14; strcpy(r1->name, "Alpha");
    r2->id = 2; r2->value = 2.71; strcpy(r2->name, "Beta");
    r3->id = 3; r3->value = 1.41; strcpy(r3->name, "Gamma");

    printf("Pool usage: %zu / %zu blocks\n", pool->used_count, pool->block_count);

    pool_free(pool, r2);  // Return r2 to pool
    printf("After free: %zu / %zu blocks\n", pool->used_count, pool->block_count);

    Record *r4 = pool_alloc(pool);  // Reuses r2's memory
    r4->id = 4;
    printf("r4->id = %d (reused block)\n", r4->id);

    pool_destroy(pool);
    return 0;
}
```

---

## 5. Memory Fragmentation

### Internal vs External Fragmentation

```
External Fragmentation:
+------+----+------+----+------+----+
| Used | -- | Used | -- | Used | -- |   Free gaps too small
+------+----+------+----+------+----+   for new allocation
         4B          8B          4B     even though 16B total free

Internal Fragmentation:
+----------+----------+----------+
| Used: 3B | Used: 5B | Used: 1B |  Allocator rounds up to
| Pad:  5B | Pad:  3B | Pad:  7B |  8-byte boundaries
+----------+----------+----------+  15 bytes wasted in padding
```

### Mitigation Strategies

| Strategy | Targets | How It Works |
|----------|---------|--------------|
| Memory pools | External | Fixed-size blocks eliminate external fragmentation |
| Slab allocation | Both | Pre-allocate pools for common object sizes |
| Buddy system | External | Split/merge power-of-2 blocks |
| Compaction | External | Move objects to consolidate free space (requires handle indirection) |
| Arena allocator | Both | Bulk free eliminates fragmentation entirely |

---

## 6. Memory Debugging Tools

### Valgrind Memcheck

```bash
# Compile with debug info
gcc -g -O0 -o program program.c

# Run under Valgrind
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./program
```

**Example Valgrind output for a leak**:
```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 100 bytes in 1 blocks
==12345==   total heap usage: 5 allocs, 4 frees, 500 bytes allocated
==12345==
==12345== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2BBAF: malloc (vg_replace_malloc.c:299)
==12345==    by 0x400547: main (program.c:10)
```

### AddressSanitizer (ASan)

```bash
# Compile with ASan
gcc -fsanitize=address -fno-omit-frame-pointer -g -o program program.c

# Run normally -- ASan instruments the binary
./program
```

ASan detects:
- Heap buffer overflow/underflow
- Stack buffer overflow
- Use-after-free
- Double free
- Memory leaks (with `ASAN_OPTIONS=detect_leaks=1`)

### LeakSanitizer (LSan)

```bash
# Standalone leak detection
gcc -fsanitize=leak -g -o program program.c
./program
```

### Common Memory Errors and Their Symptoms

| Error | Symptom | Detection Tool |
|-------|---------|---------------|
| Use-after-free | Crash or corrupt data | ASan, Valgrind |
| Double free | Crash (heap corruption) | ASan, Valgrind |
| Buffer overflow | Silent corruption or crash | ASan, Valgrind |
| Memory leak | Growing RSS over time | LSan, Valgrind |
| Uninitialized read | Non-deterministic behavior | Valgrind (`--track-origins=yes`) |
| Stack overflow | Segfault | `ulimit -s`, ASan |

---

## 7. Practical Patterns

### RAII-like Cleanup in C

C lacks destructors, but you can emulate RAII with `goto` cleanup:

```c
#include <stdio.h>
#include <stdlib.h>

int process_file(const char *path) {
    int result = -1;
    FILE *fp = NULL;
    char *buffer = NULL;

    fp = fopen(path, "r");
    if (!fp) goto cleanup;

    buffer = malloc(4096);
    if (!buffer) goto cleanup;

    // Do work with fp and buffer...
    result = 0;

cleanup:
    free(buffer);       // free(NULL) is safe
    if (fp) fclose(fp); // fclose(NULL) is not safe
    return result;
}
```

### Ownership Conventions

Establish clear ownership rules in your API:

```c
// Convention: caller owns returned pointer and must free it
char *create_greeting(const char *name) {
    char *buf = malloc(256);
    if (buf) snprintf(buf, 256, "Hello, %s!", name);
    return buf;  // Caller must free
}

// Convention: callee borrows pointer, does not free
void print_greeting(const char *greeting) {
    printf("%s\n", greeting);  // Read-only, no ownership transfer
}

// Convention: callee takes ownership (consumes the pointer)
void log_and_free(char *message) {
    fprintf(stderr, "[LOG] %s\n", message);
    free(message);  // Callee frees -- caller must not use after this
}
```

### Resource Table Pattern

For managing many resources of the same type:

```c
#include <stdio.h>
#include <stdlib.h>

#define MAX_RESOURCES 64

typedef struct {
    void *resources[MAX_RESOURCES];
    int count;
} ResourceTable;

void rt_init(ResourceTable *rt) {
    rt->count = 0;
}

void *rt_alloc(ResourceTable *rt, size_t size) {
    if (rt->count >= MAX_RESOURCES) return NULL;
    void *ptr = malloc(size);
    if (ptr) {
        rt->resources[rt->count++] = ptr;
    }
    return ptr;
}

void rt_free_all(ResourceTable *rt) {
    for (int i = 0; i < rt->count; i++) {
        free(rt->resources[i]);
    }
    rt->count = 0;
}
```

---

## Exercises

### Exercise 1: Memory Layout Explorer
Write a program that prints the addresses of variables in each segment (text, data, BSS, heap, stack) and verifies they appear in the expected order.

### Exercise 2: Arena Allocator with Reset
Extend the arena allocator to support a "save point" and "restore" mechanism, allowing partial rollback of allocations.

### Exercise 3: Pool Allocator Stress Test
Create a memory pool for a `Connection` struct. Simulate 10,000 allocate/free cycles and verify that the pool never leaks and that freed blocks are correctly reused.

### Exercise 4: mmap Word Counter
Write a program that memory-maps a text file and counts the number of words without using `fread` or `fgets`.

### Exercise 5: Leak Detector
Implement a simple leak detector that wraps `malloc` and `free` using macros, recording file/line of each allocation. At exit, print any unfreed allocations.

```c
#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#define free(ptr)    debug_free(ptr, __FILE__, __LINE__)
```

---

## Next Steps

Once you understand memory management internals, proceed to:
- [03. Bit Operations](./03_Bit_Operations.md) - Master bit-level manipulation for systems programming
