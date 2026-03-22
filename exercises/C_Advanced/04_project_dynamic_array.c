/*
 * Exercises for Lesson 06: Project Dynamic Array
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex06 06_project_dynamic_array.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Exercise 1: Basic Dynamic Array === */
/* Problem: Implement a dynamic array of ints with automatic growth. */

typedef struct {
    int *data;
    size_t size;      /* Number of elements currently stored */
    size_t capacity;  /* Allocated capacity */
} DynArray;

DynArray *dynarray_create(size_t initial_capacity) {
    /*
     * Separate size vs capacity is the key insight:
     * - size: how many elements are in use
     * - capacity: how many elements are allocated
     * - When size == capacity, we need to grow
     *
     * Initial capacity should be > 0 to avoid edge cases with realloc.
     */
    DynArray *arr = malloc(sizeof(DynArray));
    if (!arr) return NULL;

    if (initial_capacity == 0) initial_capacity = 4;
    arr->data = malloc(initial_capacity * sizeof(int));
    if (!arr->data) { free(arr); return NULL; }

    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

void dynarray_free(DynArray *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Basic Dynamic Array ===\n");

    DynArray *arr = dynarray_create(4);
    if (!arr) { printf("Allocation failed\n"); return; }

    printf("Initial: size=%zu, capacity=%zu\n", arr->size, arr->capacity);

    /* Manually add elements to demonstrate the concept */
    for (int i = 0; i < 4; i++) {
        arr->data[arr->size++] = (i + 1) * 10;
    }

    printf("After 4 inserts: size=%zu, capacity=%zu\n", arr->size, arr->capacity);
    printf("Contents: ");
    for (size_t i = 0; i < arr->size; i++) {
        printf("%d ", arr->data[i]);
    }
    printf("\n");

    printf("\nMemory layout explanation:\n");
    printf("  data pointer:  %p\n", (void *)arr->data);
    printf("  sizeof(int):   %zu bytes\n", sizeof(int));
    printf("  Total alloc:   %zu bytes (capacity * sizeof(int))\n",
           arr->capacity * sizeof(int));

    dynarray_free(arr);
}

/* === Exercise 2: Push/Pop/Insert Operations === */
/* Problem: Implement push_back, pop_back, and insert_at with bounds checking. */

static int dynarray_grow(DynArray *arr) {
    /*
     * Growth strategy: double the capacity.
     * realloc may move the data to a new location if there isn't
     * enough contiguous space. Always assign to a temp pointer first
     * to avoid losing the original data if realloc fails.
     */
    size_t new_cap = arr->capacity * 2;
    int *new_data = realloc(arr->data, new_cap * sizeof(int));
    if (!new_data) return -1;

    arr->data = new_data;
    arr->capacity = new_cap;
    return 0;
}

int dynarray_push(DynArray *arr, int value) {
    if (arr->size == arr->capacity) {
        if (dynarray_grow(arr) != 0) return -1;
    }
    arr->data[arr->size++] = value;
    return 0;
}

int dynarray_pop(DynArray *arr, int *out) {
    if (arr->size == 0) return -1; /* Underflow */
    *out = arr->data[--arr->size];
    return 0;
}

int dynarray_insert(DynArray *arr, size_t index, int value) {
    if (index > arr->size) return -1; /* Out of bounds */
    if (arr->size == arr->capacity) {
        if (dynarray_grow(arr) != 0) return -1;
    }

    /* Shift elements right to make room -- O(n) operation */
    memmove(&arr->data[index + 1], &arr->data[index],
            (arr->size - index) * sizeof(int));
    arr->data[index] = value;
    arr->size++;
    return 0;
}

static void print_array(const DynArray *arr) {
    printf("[");
    for (size_t i = 0; i < arr->size; i++) {
        printf("%d%s", arr->data[i], i < arr->size - 1 ? ", " : "");
    }
    printf("] (size=%zu, cap=%zu)\n", arr->size, arr->capacity);
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Push/Pop/Insert Operations ===\n");

    DynArray *arr = dynarray_create(2);
    if (!arr) return;

    printf("Push 10, 20, 30, 40, 50:\n");
    for (int v = 10; v <= 50; v += 10) {
        dynarray_push(arr, v);
        printf("  push(%d): ", v);
        print_array(arr);
    }

    printf("\nInsert 25 at index 2:\n");
    dynarray_insert(arr, 2, 25);
    printf("  ");
    print_array(arr);

    printf("\nInsert 5 at index 0 (front):\n");
    dynarray_insert(arr, 0, 5);
    printf("  ");
    print_array(arr);

    printf("\nPop 3 elements:\n");
    for (int i = 0; i < 3; i++) {
        int val;
        if (dynarray_pop(arr, &val) == 0) {
            printf("  popped %d: ", val);
            print_array(arr);
        }
    }

    /* Edge case: pop from empty */
    DynArray *empty = dynarray_create(1);
    int val;
    printf("\nPop from empty array: %s\n",
           dynarray_pop(empty, &val) == -1 ? "ERROR (correct)" : "unexpected");

    dynarray_free(arr);
    dynarray_free(empty);
}

/* === Exercise 3: Resize Strategy Comparison === */
/* Problem: Compare doubling vs additive growth strategies. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Resize Strategy Comparison ===\n");

    /*
     * Two common growth strategies:
     * 1. Doubling (multiplicative): new_cap = old_cap * 2
     *    - Amortized O(1) per push
     *    - Wastes up to 50% of allocated memory
     *
     * 2. Additive: new_cap = old_cap + FIXED_INCREMENT
     *    - Amortized O(n) per push (many more reallocations)
     *    - More memory-efficient
     *
     * For n insertions:
     * - Doubling: ~log2(n) reallocations, total copies ~2n
     * - Additive (k): ~n/k reallocations, total copies ~n^2/(2k)
     */

    int n = 1000;
    printf("Inserting %d elements:\n\n", n);
    printf("%-20s  %-12s  %-15s\n", "Strategy", "Reallocs", "Total Copies");
    printf("--------------------  ------------  ---------------\n");

    /* Simulate doubling strategy */
    int cap = 1, reallocs = 0;
    long total_copies = 0;
    for (int i = 0; i < n; i++) {
        if (i == cap) {
            total_copies += cap; /* All elements must be copied */
            cap *= 2;
            reallocs++;
        }
    }
    printf("%-20s  %-12d  %-15ld\n", "Doubling (x2)", reallocs, total_copies);

    /* Simulate 1.5x growth (used by MSVC's std::vector) */
    cap = 1; reallocs = 0; total_copies = 0;
    int size = 0;
    for (int i = 0; i < n; i++) {
        if (size == cap) {
            total_copies += size;
            cap = cap + cap / 2;
            if (cap <= size) cap = size + 1;
            reallocs++;
        }
        size++;
    }
    printf("%-20s  %-12d  %-15ld\n", "1.5x growth", reallocs, total_copies);

    /* Simulate additive (+10) */
    cap = 1; reallocs = 0; total_copies = 0;
    for (int i = 0; i < n; i++) {
        if (i == cap) {
            total_copies += cap;
            cap += 10;
            reallocs++;
        }
    }
    printf("%-20s  %-12d  %-15ld\n", "Additive (+10)", reallocs, total_copies);

    /* Simulate additive (+100) */
    cap = 1; reallocs = 0; total_copies = 0;
    for (int i = 0; i < n; i++) {
        if (i == cap) {
            total_copies += cap;
            cap += 100;
            reallocs++;
        }
    }
    printf("%-20s  %-12d  %-15ld\n", "Additive (+100)", reallocs, total_copies);

    printf("\nConclusion: Doubling minimizes reallocs and total copies.\n");
    printf("Trade-off: up to 50%% wasted memory vs O(1) amortized push.\n");
}

/* === Exercise 4: Memory Leak Detection === */
/* Problem: Demonstrate common memory leak patterns and how to detect them. */
void exercise_4(void) {
    printf("\n=== Exercise 4: Memory Leak Detection ===\n");

    /*
     * Common memory leak patterns in dynamic arrays:
     * 1. Forgetting to free the data pointer before freeing the struct
     * 2. Losing the pointer by reassigning without freeing
     * 3. Early return without cleanup
     * 4. realloc failure: losing original pointer
     */

    /* Pattern 1: Correct cleanup order */
    printf("Pattern 1: Correct cleanup\n");
    int *data = malloc(10 * sizeof(int));
    if (data) {
        printf("  Allocated %zu bytes at %p\n", 10 * sizeof(int), (void *)data);
        free(data);
        data = NULL; /* Prevent use-after-free */
        printf("  Freed and nullified pointer\n");
    }

    /* Pattern 2: realloc failure handling */
    printf("\nPattern 2: Safe realloc\n");
    int *buf = malloc(4 * sizeof(int));
    if (buf) {
        buf[0] = 42;
        printf("  Original: buf=%p, buf[0]=%d\n", (void *)buf, buf[0]);

        /* WRONG: buf = realloc(buf, 8 * sizeof(int));
         * If realloc fails, buf becomes NULL and the original memory leaks!
         */

        /* CORRECT: use a temporary pointer */
        int *tmp = realloc(buf, 8 * sizeof(int));
        if (tmp) {
            buf = tmp;
            printf("  After realloc: buf=%p, buf[0]=%d (preserved)\n",
                   (void *)buf, buf[0]);
        } else {
            printf("  realloc failed, original buf preserved at %p\n",
                   (void *)buf);
        }
        free(buf);
    }

    /* Pattern 3: Tracking allocations */
    printf("\nPattern 3: Allocation tracking\n");
    static int alloc_count = 0;
    static int free_count = 0;

    for (int i = 0; i < 5; i++) {
        int *p = malloc(sizeof(int) * (size_t)(i + 1));
        alloc_count++;
        if (p) {
            free(p);
            free_count++;
        }
    }
    printf("  Allocations: %d, Frees: %d, Leaked: %d\n",
           alloc_count, free_count, alloc_count - free_count);

    printf("\nTool recommendation: Use Valgrind to detect leaks:\n");
    printf("  valgrind --leak-check=full --show-leak-kinds=all ./program\n");
}

/* === Exercise 5: Generic Container Using void* === */
/* Problem: Create a type-agnostic dynamic array using void pointers. */

typedef struct {
    void *data;
    size_t elem_size;  /* Size of each element */
    size_t size;
    size_t capacity;
} GenericArray;

GenericArray *generic_create(size_t elem_size, size_t initial_cap) {
    GenericArray *arr = malloc(sizeof(GenericArray));
    if (!arr) return NULL;
    if (initial_cap == 0) initial_cap = 4;

    arr->data = malloc(elem_size * initial_cap);
    if (!arr->data) { free(arr); return NULL; }

    arr->elem_size = elem_size;
    arr->size = 0;
    arr->capacity = initial_cap;
    return arr;
}

int generic_push(GenericArray *arr, const void *elem) {
    if (arr->size == arr->capacity) {
        size_t new_cap = arr->capacity * 2;
        void *new_data = realloc(arr->data, arr->elem_size * new_cap);
        if (!new_data) return -1;
        arr->data = new_data;
        arr->capacity = new_cap;
    }

    /*
     * memcpy to the correct offset: base + (size * elem_size)
     * We cast data to char* for byte-level arithmetic since
     * void* arithmetic is undefined in standard C.
     */
    char *dest = (char *)arr->data + arr->size * arr->elem_size;
    memcpy(dest, elem, arr->elem_size);
    arr->size++;
    return 0;
}

void *generic_get(const GenericArray *arr, size_t index) {
    if (index >= arr->size) return NULL;
    return (char *)arr->data + index * arr->elem_size;
}

void generic_free(GenericArray *arr) {
    if (arr) { free(arr->data); free(arr); }
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Generic Container Using void* ===\n");

    /* Use with int */
    printf("Generic array of int:\n");
    GenericArray *ints = generic_create(sizeof(int), 4);
    for (int i = 1; i <= 5; i++) {
        generic_push(ints, &i);
    }
    printf("  ");
    for (size_t i = 0; i < ints->size; i++) {
        int *val = (int *)generic_get(ints, i);
        printf("%d ", *val);
    }
    printf("(size=%zu, cap=%zu)\n", ints->size, ints->capacity);
    generic_free(ints);

    /* Use with double */
    printf("\nGeneric array of double:\n");
    GenericArray *doubles = generic_create(sizeof(double), 4);
    double vals[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    for (int i = 0; i < 5; i++) {
        generic_push(doubles, &vals[i]);
    }
    printf("  ");
    for (size_t i = 0; i < doubles->size; i++) {
        double *val = (double *)generic_get(doubles, i);
        printf("%.1f ", *val);
    }
    printf("(size=%zu, cap=%zu)\n", doubles->size, doubles->capacity);
    generic_free(doubles);

    /* Use with struct */
    typedef struct { int x, y; } Point;
    printf("\nGeneric array of Point structs:\n");
    GenericArray *points = generic_create(sizeof(Point), 2);
    Point pts[] = {{1, 2}, {3, 4}, {5, 6}};
    for (int i = 0; i < 3; i++) {
        generic_push(points, &pts[i]);
    }
    printf("  ");
    for (size_t i = 0; i < points->size; i++) {
        Point *p = (Point *)generic_get(points, i);
        printf("(%d,%d) ", p->x, p->y);
    }
    printf("(size=%zu, cap=%zu)\n", points->size, points->capacity);
    generic_free(points);

    /*
     * Trade-off of void* generics:
     * + Type-agnostic, works with any data type
     * - No type safety (compiler can't catch type errors)
     * - Requires casting on every access
     * - Can't store heterogeneous types without tagged unions
     *
     * C11 _Generic can add some type safety at the macro level.
     */
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
