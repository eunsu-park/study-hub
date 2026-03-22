/*
 * Exercises for Lesson 02: Advanced Memory Management
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex02 02_advanced_memory_management.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* === Exercise 1: Implement Memory Pool === */
/* Problem: Create a fixed-size memory pool allocator that avoids
 *          fragmentation and provides O(1) allocation/deallocation. */

#define POOL_BLOCK_SIZE 64
#define POOL_NUM_BLOCKS 16

typedef struct {
    unsigned char memory[POOL_BLOCK_SIZE * POOL_NUM_BLOCKS];
    int free_list[POOL_NUM_BLOCKS];  /* stack of free block indices */
    int free_top;                     /* top of free stack (-1 = empty) */
    int allocated;                    /* count of allocated blocks */
} MemoryPool;

void pool_init(MemoryPool *pool) {
    pool->free_top = POOL_NUM_BLOCKS - 1;
    pool->allocated = 0;
    /* Initialize free list: all blocks available */
    for (int i = 0; i < POOL_NUM_BLOCKS; i++) {
        pool->free_list[i] = i;
    }
    memset(pool->memory, 0, sizeof(pool->memory));
}

void *pool_alloc(MemoryPool *pool) {
    if (pool->free_top < 0) {
        fprintf(stderr, "pool_alloc: pool exhausted!\n");
        return NULL;
    }
    int block_idx = pool->free_list[pool->free_top--];
    pool->allocated++;
    return &pool->memory[block_idx * POOL_BLOCK_SIZE];
}

void pool_free(MemoryPool *pool, void *ptr) {
    if (!ptr) return;
    /* Calculate which block this pointer belongs to */
    size_t offset = (unsigned char *)ptr - pool->memory;
    if (offset >= sizeof(pool->memory) || offset % POOL_BLOCK_SIZE != 0) {
        fprintf(stderr, "pool_free: invalid pointer!\n");
        return;
    }
    int block_idx = (int)(offset / POOL_BLOCK_SIZE);
    pool->free_list[++pool->free_top] = block_idx;
    pool->allocated--;
}

void exercise_1(void) {
    printf("=== Exercise 1: Memory Pool Allocator ===\n");

    MemoryPool pool;
    pool_init(&pool);
    printf("Pool initialized: %d blocks of %d bytes each\n",
           POOL_NUM_BLOCKS, POOL_BLOCK_SIZE);

    /* Allocate several blocks */
    void *blocks[8];
    for (int i = 0; i < 8; i++) {
        blocks[i] = pool_alloc(&pool);
        if (blocks[i]) {
            sprintf((char *)blocks[i], "Block %d data", i);
        }
    }
    printf("Allocated 8 blocks (used=%d, free=%d)\n",
           pool.allocated, pool.free_top + 1);

    /* Verify data */
    for (int i = 0; i < 8; i++) {
        printf("  blocks[%d]: \"%s\"\n", i, (char *)blocks[i]);
    }

    /* Free some blocks */
    pool_free(&pool, blocks[2]);
    pool_free(&pool, blocks[5]);
    printf("Freed blocks 2 and 5 (used=%d, free=%d)\n",
           pool.allocated, pool.free_top + 1);

    /* Reallocate — should reuse freed blocks */
    void *reused1 = pool_alloc(&pool);
    void *reused2 = pool_alloc(&pool);
    sprintf((char *)reused1, "Reused A");
    sprintf((char *)reused2, "Reused B");
    printf("Reallocated: \"%s\", \"%s\"\n",
           (char *)reused1, (char *)reused2);

    /*
     * Advantages of pool allocator:
     * - O(1) alloc and free (stack-based free list)
     * - Zero fragmentation (fixed block sizes)
     * - Cache-friendly (contiguous memory)
     * - No system calls after initialization
     *
     * Disadvantages:
     * - Fixed block size wastes memory for small allocations
     * - Fixed capacity — cannot grow
     */
}

/* === Exercise 2: mmap File Reader === */
/* Problem: Read a file using mmap instead of fread, useful for large files. */

void exercise_2(void) {
    printf("\n=== Exercise 2: mmap File Reader ===\n");

    /* Create a test file */
    const char *path = "/tmp/c_advanced_mmap_test.txt";
    FILE *fp = fopen(path, "w");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "Line 1: Hello from mmap!\n");
    fprintf(fp, "Line 2: Memory-mapped file I/O.\n");
    fprintf(fp, "Line 3: Efficient for large files.\n");
    fclose(fp);

    /* Open and mmap the file */
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return; }

    size_t file_size = (size_t)st.st_size;
    printf("File size: %zu bytes\n", file_size);

    char *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }
    close(fd);  /* fd can be closed after mmap */

    /* Process the mapped data — count lines */
    int line_count = 0;
    for (size_t i = 0; i < file_size; i++) {
        if (mapped[i] == '\n') line_count++;
    }
    printf("Line count: %d\n", line_count);

    /* Print content (mapped region is read-only) */
    printf("Content:\n");
    printf("%.*s", (int)file_size, mapped);

    /* Clean up */
    munmap(mapped, file_size);
    remove(path);

    /*
     * mmap vs fread:
     * - mmap: OS handles paging, good for random access / large files
     * - fread: buffered I/O, better for sequential small reads
     * - mmap avoids copying data from kernel to user space
     * - Must munmap() when done to release the mapping
     */
    printf("File unmapped and cleaned up.\n");
}

/* === Exercise 3: Detect Memory Leaks === */
/* Problem: Implement a simple leak detector that tracks allocations. */

#define MAX_ALLOCS 128

static struct {
    void *ptr;
    size_t size;
    const char *file;
    int line;
} alloc_table[MAX_ALLOCS];
static int alloc_count = 0;

void *tracked_malloc(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr && alloc_count < MAX_ALLOCS) {
        alloc_table[alloc_count].ptr = ptr;
        alloc_table[alloc_count].size = size;
        alloc_table[alloc_count].file = file;
        alloc_table[alloc_count].line = line;
        alloc_count++;
    }
    return ptr;
}

void tracked_free(void *ptr) {
    if (!ptr) return;
    for (int i = 0; i < alloc_count; i++) {
        if (alloc_table[i].ptr == ptr) {
            free(ptr);
            /* Remove entry by swapping with last */
            alloc_table[i] = alloc_table[--alloc_count];
            return;
        }
    }
    fprintf(stderr, "tracked_free: unknown pointer %p (possible double-free)\n",
            ptr);
}

void report_leaks(void) {
    if (alloc_count == 0) {
        printf("No memory leaks detected.\n");
        return;
    }
    printf("LEAK REPORT: %d allocation(s) not freed:\n", alloc_count);
    for (int i = 0; i < alloc_count; i++) {
        printf("  %zu bytes at %p (allocated at %s:%d)\n",
               alloc_table[i].size, alloc_table[i].ptr,
               alloc_table[i].file, alloc_table[i].line);
        free(alloc_table[i].ptr);  /* clean up for demo purposes */
    }
    alloc_count = 0;
}

/* Macros to auto-capture file and line */
#define TMALLOC(size) tracked_malloc((size), __FILE__, __LINE__)
#define TFREE(ptr)    tracked_free((ptr))

void exercise_3(void) {
    printf("\n=== Exercise 3: Leak Detector ===\n");

    /* Allocate some memory — intentionally leak some */
    char *a = TMALLOC(100);
    char *b = TMALLOC(200);
    char *c = TMALLOC(300);  /* will be "leaked" */
    int  *d = TMALLOC(sizeof(int) * 10);  /* will be "leaked" */

    strcpy(a, "properly freed");
    strcpy(b, "also freed");
    strcpy(c, "leaked!");
    d[0] = 42;

    TFREE(a);
    TFREE(b);
    /* c and d not freed — intentional leak for demo */

    printf("After freeing a and b (but not c and d):\n");
    report_leaks();
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
