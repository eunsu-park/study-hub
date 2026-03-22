# Advanced C Pointers

**Previous**: [C Advanced](./00_Overview.md) | **Next**: [Advanced Memory Management](./02_Advanced_Memory_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Perform pointer arithmetic to traverse arrays and compute element distances
2. Distinguish between pointer arrays (`int *arr[]`) and array pointers (`int (*p)[N]`)
3. Use double pointers to modify a caller's pointer from within a function
4. Declare, assign, and invoke function pointers, including with `typedef` and `qsort`
5. Apply `const` correctly with pointers to express read-only intent in function interfaces
6. Implement common data structures (linked list, dynamic 2D array) using pointer-based allocation
7. Write variadic functions using `<stdarg.h>` and apply the `restrict` qualifier for optimization

---

Pointers are simultaneously the most powerful and the most dangerous feature of C. They give you direct access to memory, enabling efficient data structures, zero-copy interfaces, and hardware control -- but a single misplaced dereference can crash your program or silently corrupt data. This lesson moves beyond the basics to build the deep, practical understanding of pointers that separates confident C programmers from cautious ones.

**Difficulty**: Advanced

---

## 1. Pointer Arithmetic

### Pointer Increment/Decrement

Adding 1 to a pointer increases the address by **the size of the type it points to**.

```c
int arr[] = {10, 20, 30, 40, 50};
int *p = arr;

printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[0] = 10
p++;
printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[1] = 20
p += 2;
printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[3] = 40
```

### Array Traversal with Pointers

```c
int arr[] = {1, 2, 3, 4, 5};
int n = sizeof(arr) / sizeof(arr[0]);

// Method 1: Using index
for (int i = 0; i < n; i++) {
    printf("%d ", arr[i]);
}

// Method 2: Pointer arithmetic
for (int *p = arr; p < arr + n; p++) {
    printf("%d ", *p);
}

// Method 3: Mixed pointer and index
int *p = arr;
for (int i = 0; i < n; i++) {
    printf("%d ", *(p + i));  // Same as p[i]
}
```

### Pointer Subtraction

Returns the **number of elements** between two pointers.

```c
int arr[] = {10, 20, 30, 40, 50};
int *start = &arr[0];
int *end = &arr[4];

ptrdiff_t diff = end - start;  // 4 (element count, not bytes)
printf("Element count: %td\n", diff);
```

### Pointer Comparison

```c
int arr[] = {1, 2, 3, 4, 5};
int *p1 = &arr[1];
int *p2 = &arr[3];

if (p1 < p2) {
    printf("p1 is at a lower address\n");  // This line prints
}

// Only compare pointers within the same array
// Comparing pointers to different arrays is undefined behavior
```

---

## 2. Arrays and Pointers

### The Truth About Array Indexing

`arr[i]` is syntactic sugar for `*(arr + i)`.

```c
int arr[] = {10, 20, 30};

// All equivalent
printf("%d\n", arr[1]);       // 20
printf("%d\n", *(arr + 1));   // 20
printf("%d\n", *(1 + arr));   // 20
printf("%d\n", 1[arr]);       // 20 (strange but legal!)
```

### Pointer Array vs Array Pointer

```c
// Pointer array: array of pointers
int *ptr_arr[3];  // Array holding 3 int*

int a = 1, b = 2, c = 3;
ptr_arr[0] = &a;
ptr_arr[1] = &b;
ptr_arr[2] = &c;

// Array pointer: pointer to an array
int (*arr_ptr)[4];  // Pointer to int[4] array

int arr[4] = {1, 2, 3, 4};
arr_ptr = &arr;

printf("%d\n", (*arr_ptr)[2]);  // 3
```

**How to read declarations**:
```c
int *ptr_arr[3];   // [3] first -> ptr_arr is array of size 3
                   // * next -> elements are pointers
                   // int -> pointers to int

int (*arr_ptr)[4]; // * first (parentheses) -> arr_ptr is pointer
                   // [4] next -> points to array of size 4
                   // int -> int array
```

### 2D Arrays and Pointer Relationships

```c
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// Element access
printf("%d\n", matrix[1][2]);           // 7
printf("%d\n", *(*(matrix + 1) + 2));   // 7

// matrix is converted to pointer to int[4] array
// matrix[i] is address of first element in row i
```

---

## 3. Multiple Indirection

### Double Pointer (Pointer to Pointer)

```c
int x = 42;
int *p = &x;
int **pp = &p;

printf("x:   %d\n", x);       // 42
printf("*p:  %d\n", *p);      // 42
printf("**pp: %d\n", **pp);   // 42

// Address relationships
printf("&x:  %p\n", (void*)&x);   // Address of x
printf("p:   %p\n", (void*)p);    // Address of x
printf("&p:  %p\n", (void*)&p);   // Address of p
printf("pp:  %p\n", (void*)pp);   // Address of p
```

### Double Pointer Use: Modifying Pointer in Function

```c
#include <stdio.h>
#include <stdlib.h>

// Wrong way: copy of pointer is passed
void allocate_wrong(int *p, int size) {
    p = malloc(size * sizeof(int));  // Only modifies local p
    // Caller's pointer is not changed
}

// Correct way: use double pointer
void allocate_correct(int **pp, int size) {
    *pp = malloc(size * sizeof(int));  // Modifies caller's pointer
}

int main(void) {
    int *arr = NULL;

    allocate_wrong(arr, 5);
    printf("wrong: %p\n", (void*)arr);  // NULL

    allocate_correct(&arr, 5);
    printf("correct: %p\n", (void*)arr);  // Valid address

    free(arr);
    return 0;
}
```

### Dynamic 2D Array

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int rows = 3, cols = 4;

    // Method 1: Pointer array (separate allocation per row)
    int **matrix = malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(int));
    }

    // Usage
    matrix[1][2] = 42;
    printf("%d\n", matrix[1][2]);

    // Free (in reverse order!)
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);

    // Method 2: Contiguous memory allocation (cache efficient)
    int *flat = malloc(rows * cols * sizeof(int));
    // Access as flat[i * cols + j]
    flat[1 * cols + 2] = 42;
    free(flat);

    return 0;
}
```

### String Array (Command Line Arguments)

```c
#include <stdio.h>

int main(int argc, char *argv[]) {
    // argv is array of char*
    // argv[0]: program name
    // argv[1] ~ argv[argc-1]: arguments

    printf("Argument count: %d\n", argc);

    for (int i = 0; i < argc; i++) {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    return 0;
}
```

```c
// Creating a string array directly
char *fruits[] = {"apple", "banana", "cherry"};
int n = sizeof(fruits) / sizeof(fruits[0]);

for (int i = 0; i < n; i++) {
    printf("%s\n", fruits[i]);
}
```

---

## 4. Function Pointers

### Basic Declaration and Usage

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int mul(int a, int b) { return a * b; }

int main(void) {
    // Function pointer declaration
    int (*fp)(int, int);

    // Assign function address
    fp = add;  // or fp = &add;
    printf("add: %d\n", fp(3, 4));  // 7

    fp = sub;
    printf("sub: %d\n", fp(3, 4));  // -1

    fp = mul;
    printf("mul: %d\n", fp(3, 4));  // 12

    return 0;
}
```

### Improving Readability with typedef

```c
// Define function pointer type
typedef int (*Operation)(int, int);

int add(int a, int b) { return a + b; }

int main(void) {
    Operation op = add;
    printf("%d\n", op(5, 3));  // 8

    // Array of function pointers
    Operation ops[] = {add, sub, mul};
    for (int i = 0; i < 3; i++) {
        printf("%d\n", ops[i](10, 3));
    }

    return 0;
}
```

### Callback Functions

```c
#include <stdio.h>

// Define callback type
typedef void (*Callback)(int);

void process_array(int *arr, int size, Callback cb) {
    for (int i = 0; i < size; i++) {
        cb(arr[i]);
    }
}

void print_value(int x) {
    printf("%d ", x);
}

void print_double(int x) {
    printf("%d ", x * 2);
}

int main(void) {
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original: ");
    process_array(arr, n, print_value);
    printf("\n");

    printf("Doubled: ");
    process_array(arr, n, print_double);
    printf("\n");

    return 0;
}
```

### Using qsort

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Comparison function: ascending
int compare_int_asc(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}

// Comparison function: descending
int compare_int_desc(const void *a, const void *b) {
    return *(int*)b - *(int*)a;
}

// String comparison
int compare_str(const void *a, const void *b) {
    return strcmp(*(char**)a, *(char**)b);
}

int main(void) {
    // Sort integers
    int nums[] = {3, 1, 4, 1, 5, 9, 2, 6};
    int n = sizeof(nums) / sizeof(nums[0]);

    qsort(nums, n, sizeof(int), compare_int_asc);

    for (int i = 0; i < n; i++) {
        printf("%d ", nums[i]);
    }
    printf("\n");  // 1 1 2 3 4 5 6 9

    // Sort strings
    char *words[] = {"banana", "apple", "cherry"};
    int wn = sizeof(words) / sizeof(words[0]);

    qsort(words, wn, sizeof(char*), compare_str);

    for (int i = 0; i < wn; i++) {
        printf("%s ", words[i]);
    }
    printf("\n");  // apple banana cherry

    return 0;
}
```

---

## 5. void Pointers and Generic Programming

A generic pointer that can point to any type.

```c
void *generic;

int x = 42;
double d = 3.14;
char c = 'A';

generic = &x;  // OK
generic = &d;  // OK
generic = &c;  // OK

// Casting required for dereference
printf("%d\n", *(int*)generic);  // Cast then dereference
```

**void pointer uses**:
- Return type of `malloc()`
- Writing generic functions (e.g., `qsort`, `memcpy`)
- Implementing polymorphic interfaces in C

### Generic Swap Function

```c
#include <stdio.h>
#include <string.h>

void generic_swap(void *a, void *b, size_t size) {
    unsigned char temp[size];  // VLA as temp buffer
    memcpy(temp, a, size);
    memcpy(a, b, size);
    memcpy(b, temp, size);
}

int main(void) {
    int x = 10, y = 20;
    generic_swap(&x, &y, sizeof(int));
    printf("x=%d, y=%d\n", x, y);  // x=20, y=10

    double a = 1.5, b = 2.5;
    generic_swap(&a, &b, sizeof(double));
    printf("a=%.1f, b=%.1f\n", a, b);  // a=2.5, b=1.5

    return 0;
}
```

---

## 6. const and Pointers

### Four Combinations

```c
int x = 10;
int y = 20;

// 1. Regular pointer
int *p1 = &x;
*p1 = 30;   // OK: can modify value
p1 = &y;    // OK: can point to different address

// 2. const int* (pointer to const int)
// = int const *
const int *p2 = &x;
// *p2 = 30;  // Error: cannot modify value
p2 = &y;      // OK: can point to different address

// 3. int* const (const pointer to int)
int *const p3 = &x;
*p3 = 30;     // OK: can modify value
// p3 = &y;   // Error: cannot point to different address

// 4. const int* const (const pointer to const int)
const int *const p4 = &x;
// *p4 = 30;  // Error: cannot modify value
// p4 = &y;   // Error: cannot point to different address
```

### How to Read

Read from right to left:

```c
const int *p;      // p is pointer, points to int const
int *const p;      // p is const pointer, points to int
const int *const p; // p is const pointer, points to int const
```

### const in Function Parameters

```c
// Input only: indicates value won't be modified
void print_array(const int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
        // arr[i] = 0;  // Compile error!
    }
}

// Always receive strings as const char*
void print_str(const char *str) {
    while (*str) {
        putchar(*str++);
    }
}
```

---

## 7. Self-referential Structures

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;  // Pointer to itself
} Node;

// Create node
Node *create_node(int data) {
    Node *node = malloc(sizeof(Node));
    if (node) {
        node->data = data;
        node->next = NULL;
    }
    return node;
}

// Add to front
void push_front(Node **head, int data) {
    Node *new_node = create_node(data);
    if (new_node) {
        new_node->next = *head;
        *head = new_node;
    }
}

// Print
void print_list(Node *head) {
    while (head) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

// Free all
void free_list(Node *head) {
    while (head) {
        Node *temp = head;
        head = head->next;
        free(temp);
    }
}

int main(void) {
    Node *list = NULL;

    push_front(&list, 3);
    push_front(&list, 2);
    push_front(&list, 1);

    print_list(list);  // 1 -> 2 -> 3 -> NULL

    free_list(list);
    return 0;
}
```

---

## 8. Variadic Functions and the `restrict` Qualifier

### Variadic Functions with `<stdarg.h>`

C supports functions that accept a variable number of arguments through the `<stdarg.h>` header. This is how `printf`, `scanf`, and similar functions work internally.

```c
#include <stdio.h>
#include <stdarg.h>

/*
 * va_list  - Type that holds the state needed to traverse the argument list
 * va_start - Initialize va_list to point to the first variadic argument
 * va_arg   - Retrieve the next argument, advancing the internal pointer
 * va_end   - Clean up (required for portability; some ABIs allocate memory)
 */

/* Sum a variable number of integers.
 * The caller must pass the count as the first argument -- there is no way
 * for the function to discover how many arguments were supplied. */
int sum(int count, ...) {
    va_list args;
    va_start(args, count);  /* Initialize: 'count' is the last named parameter */

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);  /* Retrieve next int */
    }

    va_end(args);  /* Always call va_end to avoid undefined behavior */
    return total;
}

int main(void) {
    printf("Sum: %d\n", sum(3, 10, 20, 30));   /* 60 */
    printf("Sum: %d\n", sum(5, 1, 2, 3, 4, 5)); /* 15 */
    return 0;
}
```

### Implementing a printf-like Function

A common real-world pattern is wrapping `printf` for logging:

```c
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

/* A logging function that prepends a timestamp.
 * The format string + variadic args are forwarded to vfprintf,
 * which is the va_list version of fprintf. */
void log_message(const char *level, const char *fmt, ...) {
    /* Print timestamp */
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    fprintf(stderr, "[%02d:%02d:%02d] [%s] ",
            t->tm_hour, t->tm_min, t->tm_sec, level);

    /* Forward variadic arguments to vfprintf.
     * Why vfprintf instead of fprintf?  Because we already consumed
     * the variadic args into a va_list -- fprintf cannot accept va_list. */
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
}

int main(void) {
    log_message("INFO",  "Server started on port %d", 8080);
    log_message("ERROR", "Failed to open file: %s", "config.yaml");
    return 0;
}
```

### Type Safety Issues with Variadic Functions

Variadic functions are inherently **type-unsafe**: the compiler cannot verify that the arguments match the expected types.

**Key dangers**:
- No compiler type-checking on variadic arguments
- `va_arg` with the wrong type reads incorrect bytes (UB)
- Passing fewer arguments than expected reads stack garbage
- Default argument promotions apply: `float` -> `double`, `char`/`short` -> `int`

### The `restrict` Qualifier

The `restrict` qualifier (C99) is a promise from the programmer to the compiler: **the pointer is the only way to access the memory it points to** during its lifetime. This enables the compiler to perform optimizations that would otherwise be impossible due to aliasing concerns.

```c
#include <stdio.h>
#include <string.h>

/* Without restrict: the compiler must assume a and b might overlap.
 * Every write to *a could change *b, forcing re-reads. */
void add_arrays_slow(int *a, const int *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];  /* Must re-read b[i] every iteration if a==b possible */
    }
}

/* With restrict: we promise a and b do NOT overlap.
 * The compiler can vectorize aggressively (SIMD), reorder loads/stores,
 * and keep values in registers without re-reading from memory. */
void add_arrays_fast(int *restrict a, const int *restrict b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];  /* Safe to cache b[i] and vectorize */
    }
}

int main(void) {
    int x[] = {1, 2, 3, 4};
    int y[] = {10, 20, 30, 40};

    /* Correct: x and y are separate arrays */
    add_arrays_fast(x, y, 4);

    /* WRONG: passing overlapping memory with restrict -- UB!
     * add_arrays_fast(x, x+1, 3);  <- violates restrict contract */

    for (int i = 0; i < 4; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");  /* 11 22 33 44 */

    return 0;
}
```

### restrict in Standard Library Functions

The C standard library uses `restrict` extensively. Compare the signatures of `memcpy` and `memmove`:

```c
/* memcpy: source and destination must NOT overlap.
 * restrict tells the compiler this, enabling optimized block copy. */
void *memcpy(void *restrict dest, const void *restrict src, size_t n);

/* memmove: source and destination MAY overlap.
 * No restrict -> compiler must handle overlap (copy through temp buffer). */
void *memmove(void *dest, const void *src, size_t n);

/* This is why memcpy is faster than memmove:
 * restrict allows the compiler to use wider loads/stores without
 * worrying about overwriting source data before it is read. */
```

**Guidelines for using `restrict`**:
1. Use it on function parameters when you can guarantee no aliasing
2. Violating the `restrict` contract is undefined behavior -- the compiler trusts you
3. `restrict` only exists in C (C99+), not in standard C++ (though compilers offer `__restrict`)
4. Profile before and after: the optimization benefit depends on the loop and target architecture

---

## Exercises

### Exercise 1: Reverse Array

Write a function that reverses an array in place using only pointers.

```c
void reverse_array(int *arr, int size);

// Example: {1, 2, 3, 4, 5} -> {5, 4, 3, 2, 1}
```

### Exercise 2: Reverse Words in String

Convert "Hello World" to "World Hello" using pointer manipulation.

### Exercise 3: Reverse Linked List

Write a function that reverses a singly linked list.

```c
Node *reverse_list(Node *head);
```

### Exercise 4: Function Pointer Calculator

Implement the four arithmetic operations using a function pointer array.

```c
// Input: "3 + 4" -> Output: 7
```

### Exercise 5: Generic Binary Search

Implement a generic binary search function using `void*` and a comparison callback, similar to `bsearch` from `<stdlib.h>`.

```c
void *generic_bsearch(const void *key, const void *base,
                      size_t nmemb, size_t size,
                      int (*compar)(const void *, const void *));
```

---

## Next Steps

Once you've mastered advanced pointers, proceed to:
- [02. Advanced Memory Management](./02_Advanced_Memory_Management.md) - Deep dive into memory layout, custom allocators, and debugging tools
