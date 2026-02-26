# Advanced C Pointers

**Previous**: [Advanced Embedded Protocols](./19_Advanced_Embedded_Protocols.md) | **Next**: [Network Programming in C](./21_Network_Programming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Perform pointer arithmetic to traverse arrays and compute element distances
2. Distinguish between pointer arrays (`int *arr[]`) and array pointers (`int (*p)[N]`)
3. Use double pointers to modify a caller's pointer from within a function
4. Declare, assign, and invoke function pointers, including with `typedef` and `qsort`
5. Manage dynamic memory safely with `malloc`, `calloc`, `realloc`, and `free`, avoiding leaks and dangling references
6. Apply `const` correctly with pointers to express read-only intent in function interfaces
7. Implement common data structures (linked list, dynamic 2D array) using pointer-based allocation
8. Detect and fix pointer bugs (dangling pointer, double free, buffer overflow) with Valgrind and AddressSanitizer

---

Pointers are simultaneously the most powerful and the most dangerous feature of C. They give you direct access to memory, enabling efficient data structures, zero-copy interfaces, and hardware control -- but a single misplaced dereference can crash your program or silently corrupt data. This lesson moves beyond the basics to build the deep, practical understanding of pointers that separates confident C programmers from cautious ones.

**Difficulty**: Intermediate

---

## 1. Pointer Basics Review

### Memory and Addresses

Computer memory is a contiguous space with byte-addressed locations.

```c
#include <stdio.h>

int main(void) {
    int x = 42;

    printf("Value: %d\n", x);           // 42
    printf("Address: %p\n", (void*)&x); // 0x7ffd12345678 (example)
    printf("Size: %zu bytes\n", sizeof(x)); // 4

    return 0;
}
```

### Pointer Declaration and Initialization

```c
int x = 10;
int *p;      // Pointer declaration
p = &x;      // Assign address

// Declare and initialize at once (recommended)
int *q = &x;

// Uninitialized pointer is dangerous!
int *danger; // Garbage value - don't use
```

### Dereference Operator (*)

```c
int x = 42;
int *p = &x;

printf("Value pointed to by p: %d\n", *p);  // 42

*p = 100;  // x's value changed to 100
printf("New value of x: %d\n", x);          // 100
```

### NULL Pointer

```c
int *p = NULL;  // Points to nothing

// NULL check is essential!
if (p != NULL) {
    printf("%d\n", *p);
} else {
    printf("Pointer is NULL\n");
}

// nullptr is also available in C11 (some compilers)
```

### void Pointer

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

---

## 2. Pointer Arithmetic

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

## 3. Arrays and Pointers

### Meaning of Array Name

In most contexts, an array name is converted to **the address of the first element**.

```c
int arr[5] = {1, 2, 3, 4, 5};

printf("arr:     %p\n", (void*)arr);      // Same address
printf("&arr[0]: %p\n", (void*)&arr[0]);  // Same address

int *p = arr;  // Same as int *p = &arr[0];
```

**Exceptions**:
```c
// sizeof returns total array size
printf("sizeof(arr): %zu\n", sizeof(arr));  // 20 (5 * 4 bytes)

// &arr is address of entire array (different type)
printf("arr:  %p\n", (void*)arr);           // int* type
printf("&arr: %p\n", (void*)&arr);          // int(*)[5] type

// Same address but +1 means different things
printf("arr + 1:  %p\n", (void*)(arr + 1));   // Increases by 4 bytes
printf("&arr + 1: %p\n", (void*)(&arr + 1));  // Increases by 20 bytes
```

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

### 2D Arrays

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
int *ptr_arr[3];   // [3] first → ptr_arr is array of size 3
                   // * next → elements are pointers
                   // int → pointers to int

int (*arr_ptr)[4]; // * first (parentheses) → arr_ptr is pointer
                   // [4] next → points to array of size 4
                   // int → int array
```

---

## 4. Multiple Indirection

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

## 5. Function Pointers

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

## 6. Dynamic Memory Management

### malloc, calloc, realloc, free

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    // malloc: allocate without initialization
    int *arr1 = malloc(5 * sizeof(int));
    // Values are garbage! Initialization needed

    // calloc: allocate with zero initialization
    int *arr2 = calloc(5, sizeof(int));
    // All values are 0

    // realloc: resize
    arr1 = realloc(arr1, 10 * sizeof(int));
    // Original values preserved, additional space is uninitialized

    // NULL check is essential!
    if (arr1 == NULL || arr2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Free after use
    free(arr1);
    free(arr2);

    // Set to NULL after free (optional but recommended)
    arr1 = NULL;
    arr2 = NULL;

    return 0;
}
```

### Preventing Memory Leaks

```c
// Wrong pattern: memory leak
void memory_leak(void) {
    int *p = malloc(100);
    // Function ends without free → leak!
}

// Correct pattern
void no_leak(void) {
    int *p = malloc(100);
    if (p == NULL) return;

    // Do work...

    free(p);  // Always free
}

// Be careful with error handling
int process(void) {
    int *a = malloc(100);
    int *b = malloc(200);

    if (a == NULL || b == NULL) {
        free(a);  // free(NULL) is safe
        free(b);
        return -1;
    }

    // Do work...

    free(a);
    free(b);
    return 0;
}
```

### Safe realloc Usage

```c
// Dangerous pattern
p = realloc(p, new_size);  // Original address lost on failure!

// Safe pattern
int *temp = realloc(p, new_size);
if (temp == NULL) {
    // p is still valid
    free(p);
    return NULL;
}
p = temp;
```

---

## 7. const and Pointers

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

## 8. Strings and Pointers

### String Literal vs Character Array

```c
// String literal: read-only memory
char *str1 = "Hello";
// str1[0] = 'h';  // Undefined behavior! (usually crashes)

// Character array: modifiable
char str2[] = "Hello";
str2[0] = 'h';  // OK

// Using const is recommended
const char *str3 = "Hello";  // Intent is clear
```

### Implementing String Functions

```c
#include <stdio.h>

// strlen implementation
size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

// strcpy implementation
char *my_strcpy(char *dest, const char *src) {
    char *ret = dest;
    while ((*dest++ = *src++));
    return ret;
}

// strcmp implementation
int my_strcmp(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(unsigned char*)s1 - *(unsigned char*)s2;
}

// strcat implementation
char *my_strcat(char *dest, const char *src) {
    char *ret = dest;
    while (*dest) dest++;  // Move to end
    while ((*dest++ = *src++));
    return ret;
}

int main(void) {
    char buffer[100] = "Hello";

    printf("Length: %zu\n", my_strlen(buffer));  // 5

    my_strcat(buffer, " World");
    printf("%s\n", buffer);  // Hello World

    return 0;
}
```

### String Arrays

```c
// Method 1: Pointer array (different lengths possible)
const char *names1[] = {
    "Alice",
    "Bob",
    "Charlie"
};

// Method 2: 2D array (fixed length)
char names2[][10] = {
    "Alice",
    "Bob",
    "Charlie"
};

// Difference
printf("sizeof(names1[0]): %zu\n", sizeof(names1[0]));  // 8 (pointer size)
printf("sizeof(names2[0]): %zu\n", sizeof(names2[0]));  // 10 (array size)
```

---

## 9. Structures and Pointers

### Structure Pointer Basics

```c
#include <stdio.h>
#include <string.h>

typedef struct {
    char name[50];
    int age;
    double height;
} Person;

int main(void) {
    Person p1 = {"Alice", 25, 165.5};
    Person *ptr = &p1;

    // Member access: -> operator
    printf("Name: %s\n", ptr->name);      // Same as (*ptr).name
    printf("Age: %d\n", ptr->age);

    // Modify value
    ptr->age = 26;

    return 0;
}
```

### Dynamic Structures

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *name;  // Dynamically allocated string
    int age;
} Person;

Person *create_person(const char *name, int age) {
    Person *p = malloc(sizeof(Person));
    if (p == NULL) return NULL;

    p->name = malloc(strlen(name) + 1);
    if (p->name == NULL) {
        free(p);
        return NULL;
    }

    strcpy(p->name, name);
    p->age = age;

    return p;
}

void free_person(Person *p) {
    if (p) {
        free(p->name);
        free(p);
    }
}

int main(void) {
    Person *alice = create_person("Alice", 25);
    if (alice) {
        printf("%s, %d\n", alice->name, alice->age);
        free_person(alice);
    }
    return 0;
}
```

### Self-referential Structure (Linked List)

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

## 10. Common Mistakes and Debugging

### Dangling Pointer

A pointer pointing to freed memory.

```c
// Dangerous code
int *p = malloc(sizeof(int));
*p = 42;
free(p);
// p still points to same address (dangling pointer)
printf("%d\n", *p);  // Undefined behavior!

// Solution
free(p);
p = NULL;  // Explicitly set to NULL

if (p != NULL) {
    printf("%d\n", *p);  // Protected by NULL check
}
```

### Use After Free

```c
// Dangerous pattern
char *str = malloc(100);
strcpy(str, "Hello");
free(str);
// ...
printf("%s\n", str);  // Accessing freed memory!
```

### Double Free

```c
// Dangerous code
int *p = malloc(sizeof(int));
free(p);
free(p);  // Freeing same memory twice → may crash

// Solution
free(p);
p = NULL;
free(p);  // free(NULL) is safe
```

### Buffer Overflow

```c
// Dangerous code
char buffer[10];
strcpy(buffer, "This is a very long string");  // Overflow!

// Safe code
char buffer[10];
strncpy(buffer, "This is a very long string", sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';

// Or use snprintf
snprintf(buffer, sizeof(buffer), "%s", "This is a very long string");
```

### Finding Memory Errors with Valgrind

```bash
# Compile (include debug info)
gcc -g -o myprogram myprogram.c

# Run Valgrind
valgrind --leak-check=full ./myprogram
```

**Example Valgrind output**:
```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 100 bytes in 1 blocks
==12345==   total heap usage: 5 allocs, 4 frees, 500 bytes allocated
==12345==
==12345== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2BBAF: malloc (vg_replace_malloc.c:299)
==12345==    by 0x400547: main (myprogram.c:10)
```

### Debugging Tips

1. **Print pointers**
```c
printf("ptr = %p, *ptr = %d\n", (void*)ptr, ptr ? *ptr : -1);
```

2. **Use assert**
```c
#include <assert.h>

void process(int *arr, int size) {
    assert(arr != NULL);
    assert(size > 0);
    // ...
}
```

3. **Use AddressSanitizer** (GCC/Clang)
```bash
gcc -fsanitize=address -g myprogram.c -o myprogram
./myprogram
```

---

## 11. Variadic Functions and the `restrict` Qualifier

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

```c
#include <stdio.h>
#include <stdarg.h>

double average(int count, ...) {
    va_list args;
    va_start(args, count);

    double total = 0.0;
    for (int i = 0; i < count; i++) {
        /* DANGER: if the caller passes an int where we expect double,
         * va_arg reads the wrong number of bytes → garbage value.
         * The compiler will NOT warn about this mismatch! */
        total += va_arg(args, double);
    }

    va_end(args);
    return total / count;
}

int main(void) {
    /* Correct: passing doubles */
    printf("%.2f\n", average(3, 1.0, 2.0, 3.0));  /* 2.00 */

    /* BUG: passing ints instead of doubles -- undefined behavior!
     * average(3, 1, 2, 3);  ← int is 4 bytes, double is 8 bytes */

    return 0;
}
```

**Key dangers**:
- No compiler type-checking on variadic arguments
- `va_arg` with the wrong type reads incorrect bytes (UB)
- Passing fewer arguments than expected reads stack garbage
- Default argument promotions apply: `float` → `double`, `char`/`short` → `int`

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
     * add_arrays_fast(x, x+1, 3);  ← violates restrict contract */

    for (int i = 0; i < 4; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");  /* 11 22 33 44 */

    return 0;
}
```

### restrict and Aliasing: Why It Matters

```c
#include <stdio.h>

/* Classic example: without restrict, this function is pessimized.
 * Consider what happens if a == b: writing *a changes *b! */
void multiply(int *a, int *b, int *result) {
    *result = *a * *b;
    /* If the compiler needs *a again later, it must re-read from memory
     * because writing *result might have changed *a (if result == a). */
}

/* With restrict: the compiler knows a, b, result are all distinct.
 * It can keep *a and *b in registers and skip re-reads. */
void multiply_fast(int *restrict a, int *restrict b, int *restrict result) {
    *result = *a * *b;
    /* Compiler can trust that *a and *b haven't changed. */
}
```

### restrict in Standard Library Functions

The C standard library uses `restrict` extensively. Compare the signatures of `memcpy` and `memmove`:

```c
/* memcpy: source and destination must NOT overlap.
 * restrict tells the compiler this, enabling optimized block copy. */
void *memcpy(void *restrict dest, const void *restrict src, size_t n);

/* memmove: source and destination MAY overlap.
 * No restrict → compiler must handle overlap (copy through temp buffer). */
void *memmove(void *dest, const void *src, size_t n);

/* This is why memcpy is faster than memmove:
 * restrict allows the compiler to use wider loads/stores without
 * worrying about overwriting source data before it is read. */
```

### Practical restrict Usage Patterns

```c
#include <stddef.h>

/* Pattern 1: Function parameters -- most common use */
void process(float *restrict output,
             const float *restrict input,
             size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * 2.0f;
    }
}

/* Pattern 2: Structure members (rare but valid in C99) */
struct Buffer {
    float *restrict data;  /* Only this pointer accesses the buffer */
    size_t size;
};

/* Pattern 3: Local variables */
void compute(float *base, size_t n) {
    /* Tell the compiler these local views don't alias each other */
    float *restrict first_half  = base;
    float *restrict second_half = base + n / 2;
    /* Only valid if we don't access the same elements through both */
}
```

**Guidelines for using `restrict`**:
1. Use it on function parameters when you can guarantee no aliasing
2. Violating the `restrict` contract is undefined behavior -- the compiler trusts you
3. `restrict` only exists in C (C99+), not in standard C++ (though compilers offer `__restrict`)
4. Profile before and after: the optimization benefit depends on the loop and target architecture

---

## Practice Problems

### Problem 1: Reverse Array

Write a function that reverses an array in place using only pointers.

```c
void reverse_array(int *arr, int size);

// Example: {1, 2, 3, 4, 5} → {5, 4, 3, 2, 1}
```

### Problem 2: Reverse Words in String

Convert "Hello World" to "World Hello".

### Problem 3: Reverse Linked List

Write a function that reverses a singly linked list.

```c
Node *reverse_list(Node *head);
```

### Problem 4: Function Pointer Calculator

Implement the four arithmetic operations using a function pointer array.

```c
// Input: "3 + 4" → Output: 7
```

---

## Summary

| Concept | Key Points |
|------|------------|
| Pointer basics | `&`(address), `*`(dereference), NULL check essential |
| Pointer arithmetic | Increases/decreases by type size |
| Arrays and pointers | `arr[i] == *(arr + i)` |
| Multiple indirection | Use when modifying pointer in function |
| Function pointers | Callbacks, qsort comparison function |
| Dynamic memory | malloc/free, leak prevention, safe realloc pattern |
| const pointers | `const int*` vs `int* const` |
| Debugging | Valgrind, AddressSanitizer |

---

## References

- [C Programming: A Modern Approach (K.N. King)](http://knking.com/books/c2/)
- [The C Programming Language (K&R)](https://en.wikipedia.org/wiki/The_C_Programming_Language)
- [Valgrind Documentation](https://valgrind.org/docs/manual/quick-start.html)
- [cdecl: C declaration decoder](https://cdecl.org/)
