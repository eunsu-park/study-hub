/*
 * memory_pool_demo.c
 *
 * Simple fixed-size block memory pool allocator.
 * Pre-allocates a large arena and hands out equal-sized blocks
 * via a free-list, avoiding per-allocation malloc overhead.
 *
 * Build:  gcc -Wall -Wextra -std=c11 -o memory_pool_demo memory_pool_demo.c
 * Run:    ./memory_pool_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Pool configuration ───────────────────────────────────────── */
#define BLOCK_SIZE  64   /* bytes per block (must be >= sizeof(void*)) */
#define POOL_BLOCKS 16   /* number of blocks in the pool */

typedef struct MemPool {
    uint8_t  arena[BLOCK_SIZE * POOL_BLOCKS];  /* backing memory */
    void    *free_head;                        /* singly-linked free list */
    size_t   alloc_count;
} MemPool;

/* ── Initialise: chain every block into the free list ─────────── */
static void pool_init(MemPool *p)
{
    p->free_head   = NULL;
    p->alloc_count = 0;

    for (int i = POOL_BLOCKS - 1; i >= 0; i--) {
        void *block = p->arena + i * BLOCK_SIZE;
        *(void **)block = p->free_head;   /* store next-ptr in block */
        p->free_head = block;
    }
}

/* ── Allocate one block ───────────────────────────────────────── */
static void *pool_alloc(MemPool *p)
{
    if (!p->free_head) {
        fprintf(stderr, "pool_alloc: pool exhausted\n");
        return NULL;
    }
    void *block  = p->free_head;
    p->free_head = *(void **)block;
    p->alloc_count++;
    memset(block, 0, BLOCK_SIZE);
    return block;
}

/* ── Return one block to the pool ─────────────────────────────── */
static void pool_free(MemPool *p, void *block)
{
    if (!block) return;
    *(void **)block = p->free_head;
    p->free_head = block;
    p->alloc_count--;
}

/* ── Diagnostics ──────────────────────────────────────────────── */
static void pool_stats(const MemPool *p)
{
    size_t free_count = 0;
    for (void *cur = p->free_head; cur; cur = *(void **)cur)
        free_count++;
    printf("  allocated: %zu / %d  |  free: %zu\n",
           p->alloc_count, POOL_BLOCKS, free_count);
}

int main(void)
{
    MemPool pool;
    pool_init(&pool);

    printf("=== Memory Pool Demo ===\n");
    printf("Block size: %d bytes, capacity: %d blocks\n\n", BLOCK_SIZE, POOL_BLOCKS);

    printf("After init:\n");
    pool_stats(&pool);

    /* Allocate several blocks and write data */
    void *blocks[5];
    for (int i = 0; i < 5; i++) {
        blocks[i] = pool_alloc(&pool);
        snprintf((char *)blocks[i], BLOCK_SIZE, "block-%d payload", i);
    }

    printf("\nAfter 5 allocations:\n");
    pool_stats(&pool);

    for (int i = 0; i < 5; i++)
        printf("  blocks[%d]: \"%s\"\n", i, (char *)blocks[i]);

    /* Free two blocks */
    pool_free(&pool, blocks[1]);
    pool_free(&pool, blocks[3]);
    printf("\nAfter freeing blocks[1] and blocks[3]:\n");
    pool_stats(&pool);

    /* Re-allocate — should reuse freed blocks */
    void *reused = pool_alloc(&pool);
    snprintf((char *)reused, BLOCK_SIZE, "reused block");
    printf("\nAfter re-alloc:\n");
    pool_stats(&pool);
    printf("  reused: \"%s\"\n", (char *)reused);

    pool_free(&pool, reused);
    for (int i = 0; i < 5; i++) {
        if (i != 1 && i != 3)
            pool_free(&pool, blocks[i]);
    }

    printf("\nAfter freeing all:\n");
    pool_stats(&pool);

    return 0;
}
