# Dynamic Memory

**Previous**: [Structs and Unions](./08_Structs_and_Unions.md) | **Next**: [File I/O](./10_File_IO.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Allocate heap memory using `malloc` and `calloc` and explain when each is appropriate
2. Resize allocations with `realloc` and handle the returned pointer correctly
3. Release memory with `free` and set pointers to `NULL` to prevent dangling references
4. Identify and prevent common memory errors: leaks, double-free, use-after-free
5. Apply a consistent allocate-check-use-free pattern in programs

---

Stack variables are simple and fast, but they have two critical limitations: their size must be known at compile time, and they are destroyed when the function returns. Dynamic memory allocation lets you request memory at runtime, control its lifetime, and build data structures that grow and shrink as needed. The cost is that **you** become responsible for releasing that memory.

## 1. Stack vs Heap

| Property | Stack | Heap |
|----------|-------|------|
| Allocation | Automatic (function entry) | Manual (`malloc`, `calloc`) |
| Deallocation | Automatic (function exit) | Manual (`free`) |
| Size | Fixed at compile time | Determined at runtime |
| Typical limit | 1-8 MB (OS default) | Limited by available RAM |
| Speed | Very fast (pointer bump) | Slower (bookkeeping overhead) |
| Fragmentation | None | Possible over time |

**When to use the heap**:
- You don't know the size until runtime (user input, file data)
- The data must outlive the function that creates it
- The data is too large for the stack (large arrays, buffers)

---

## 2. malloc

`malloc` (memory allocate) requests a block of uninitialized bytes from the heap. It returns a `void *` pointer to the block, or `NULL` if the allocation fails.

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    printf("How many numbers? ");
    scanf("%d", &n);

    /* Allocate array of n ints */
    int *arr = malloc(n * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Use the memory */
    for (int i = 0; i < n; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    /* Release the memory */
    free(arr);
    arr = NULL;

    return 0;
}
```

### Key Points

- Always use `sizeof` with the pointed-to type: `malloc(n * sizeof(int))` or the safer idiom `malloc(n * sizeof(*arr))` which stays correct even if you change the type of `arr`.
- **Always check the return value**. `malloc` returns `NULL` on failure.
- In C, casting the return of `malloc` is unnecessary (and discouraged by many style guides) because `void *` converts implicitly to any pointer type. In C++ it is required, but you should not use `malloc` in C++ anyway.

```c
/* Preferred: sizeof applied to the variable, not the type */
int *data = malloc(count * sizeof(*data));
```

---

## 3. calloc

`calloc` (clear allocate) works like `malloc` but takes two arguments (count and element size) and **zero-initializes** the memory.

```c
#include <stdlib.h>

int main(void) {
    /* Allocate 100 ints, all initialized to 0 */
    int *arr = calloc(100, sizeof(int));
    if (arr == NULL) {
        return 1;
    }

    /* arr[0] through arr[99] are all 0 */

    free(arr);
    arr = NULL;
    return 0;
}
```

| Function | Arguments | Initialized? | Use When |
|----------|-----------|-------------|----------|
| `malloc` | Total bytes | No (garbage) | You will immediately write to all bytes |
| `calloc` | Count, element size | Yes (zeroed) | You need zero-initialized memory |

`calloc` also has a safety advantage: it checks for integer overflow in `count * size` internally, while `malloc(count * size)` can silently overflow and allocate too little memory.

---

## 4. realloc

`realloc` resizes a previously allocated block. It may move the data to a new location if there is not enough room to expand in place.

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    size_t capacity = 4;
    size_t size = 0;
    int *arr = malloc(capacity * sizeof(*arr));
    if (arr == NULL) return 1;

    /* Simulate adding elements */
    for (int i = 0; i < 20; i++) {
        if (size == capacity) {
            capacity *= 2;
            int *temp = realloc(arr, capacity * sizeof(*temp));
            if (temp == NULL) {
                fprintf(stderr, "realloc failed\n");
                free(arr);   /* free the original block */
                return 1;
            }
            arr = temp;
            printf("Grew to capacity %zu\n", capacity);
        }
        arr[size++] = i * 10;
    }

    for (size_t i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);
    arr = NULL;
    return 0;
}
```

### Critical Rule: Never Do This

```c
arr = realloc(arr, new_size);   /* DANGEROUS */
```

If `realloc` fails and returns `NULL`, you have just overwritten your only pointer to the original block — that memory is now **leaked**. Always use a temporary pointer:

```c
int *temp = realloc(arr, new_size);
if (temp == NULL) {
    /* handle error — arr is still valid */
} else {
    arr = temp;
}
```

### realloc Special Cases

| Call | Behavior |
|------|----------|
| `realloc(NULL, size)` | Equivalent to `malloc(size)` |
| `realloc(ptr, 0)` | Implementation-defined (may free or return small block) — avoid |
| `realloc(ptr, smaller)` | May shrink in place, may return new pointer |

---

## 5. free

`free` returns a block of dynamically allocated memory to the system. After freeing, set the pointer to `NULL` to prevent accidental reuse.

```c
int *p = malloc(sizeof(int));
*p = 42;

free(p);
p = NULL;   /* good practice — prevents dangling pointer use */
```

**Rules**:
- Only free memory that was returned by `malloc`, `calloc`, or `realloc`.
- Do not free stack variables, global variables, or string literals.
- Do not free the same pointer twice.
- `free(NULL)` is safe and does nothing — this is why setting freed pointers to `NULL` helps.

---

## 6. Common Memory Errors

### Memory Leak

Allocated memory that is never freed. The program's memory usage grows without bound.

```c
void process(void) {
    char *buf = malloc(1024);
    if (some_condition) {
        return;   /* BUG: buf is leaked on this path */
    }
    /* ... use buf ... */
    free(buf);
}
```

### Double Free

Freeing the same block twice corrupts the memory allocator's internal data structures.

```c
int *p = malloc(sizeof(int));
free(p);
free(p);   /* UNDEFINED BEHAVIOR — heap corruption */
```

**Fix**: Set pointer to `NULL` after free. `free(NULL)` is a no-op.

### Use-After-Free

Accessing memory after it has been freed. The memory may have been reallocated for a different purpose.

```c
int *p = malloc(sizeof(int));
*p = 42;
free(p);
printf("%d\n", *p);   /* UNDEFINED BEHAVIOR */
```

### Buffer Overflow

Writing past the end of an allocated block.

```c
int *arr = malloc(5 * sizeof(int));
arr[5] = 99;   /* UNDEFINED BEHAVIOR — out of bounds */
```

| Error | Cause | Symptom | Prevention |
|-------|-------|---------|------------|
| Leak | Forgot to `free` | Growing memory usage | Free on all paths |
| Double free | `free` called twice | Crash / corruption | Set to NULL after free |
| Use-after-free | Access freed memory | Garbage data / crash | Set to NULL, don't use |
| Overflow | Write past allocation | Corruption / crash | Track sizes carefully |

---

## 7. Memory Management Patterns

### Ownership

Establish clear ownership: the function (or module) that allocates is responsible for freeing.

```c
/* Caller owns the returned memory */
char *create_greeting(const char *name) {
    size_t len = strlen("Hello, ") + strlen(name) + 2;
    char *buf = malloc(len);
    if (buf == NULL) return NULL;
    snprintf(buf, len, "Hello, %s!", name);
    return buf;   /* caller must free */
}

int main(void) {
    char *msg = create_greeting("Alice");
    if (msg) {
        printf("%s\n", msg);
        free(msg);   /* caller frees */
    }
    return 0;
}
```

### Goto Cleanup Pattern

When a function makes multiple allocations that depend on each other, use `goto cleanup` to ensure all are freed on any error:

```c
#include <stdio.h>
#include <stdlib.h>

int process_data(size_t n) {
    int *buffer = NULL;
    char *name = NULL;
    int result = -1;

    buffer = malloc(n * sizeof(*buffer));
    if (buffer == NULL) goto cleanup;

    name = malloc(256);
    if (name == NULL) goto cleanup;

    /* ... do work with buffer and name ... */

    result = 0;  /* success */

cleanup:
    free(name);     /* free(NULL) is safe */
    free(buffer);
    return result;
}
```

This pattern is widely used in the Linux kernel and other system-level C code.

---

## 8. Dynamic Arrays of Structs

A practical example combining `malloc`, structs, and proper cleanup:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[50];
    int age;
    double score;
} Student;

Student *create_students(size_t count) {
    Student *students = calloc(count, sizeof(Student));
    return students;  /* NULL if allocation failed */
}

void print_students(const Student *students, size_t count) {
    printf("%-20s %5s %7s\n", "Name", "Age", "Score");
    printf("%-20s %5s %7s\n", "----", "---", "-----");
    for (size_t i = 0; i < count; i++) {
        printf("%-20s %5d %7.2f\n",
               students[i].name,
               students[i].age,
               students[i].score);
    }
}

int main(void) {
    size_t n = 3;
    Student *roster = create_students(n);
    if (roster == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Populate */
    strcpy(roster[0].name, "Alice");
    roster[0].age = 20;
    roster[0].score = 95.5;

    strcpy(roster[1].name, "Bob");
    roster[1].age = 22;
    roster[1].score = 88.0;

    strcpy(roster[2].name, "Charlie");
    roster[2].age = 21;
    roster[2].score = 92.3;

    print_students(roster, n);

    /* Grow the array to add one more student */
    n = 4;
    Student *temp = realloc(roster, n * sizeof(Student));
    if (temp == NULL) {
        fprintf(stderr, "realloc failed\n");
        free(roster);
        return 1;
    }
    roster = temp;

    strcpy(roster[3].name, "Diana");
    roster[3].age = 23;
    roster[3].score = 97.1;

    printf("\nAfter adding Diana:\n");
    print_students(roster, n);

    free(roster);
    roster = NULL;
    return 0;
}
```

Output:

```
Name                   Age   Score
----                   ---   -----
Alice                   20   95.50
Bob                     22   88.00
Charlie                 21   92.30

After adding Diana:
Name                   Age   Score
----                   ---   -----
Alice                   20   95.50
Bob                     22   88.00
Charlie                 21   92.30
Diana                   23   97.10
```

---

## Exercises

**Exercise 1 — Dynamic Integer Array**: Write a program that reads integers from the user until they enter -1. Store them in a dynamically growing array (start with capacity 4, double when full). Print all values and free the memory.

**Exercise 2 — String Duplicator**: Write a function `char *my_strdup(const char *s)` that allocates memory, copies the string into it, and returns the new string. The caller is responsible for freeing. Test with several strings.

**Exercise 3 — Matrix Allocation**: Write functions to dynamically allocate a 2D matrix (`int **`), fill it with the multiplication table (row * column), print it, and free all memory. The dimensions should be provided at runtime.

**Exercise 4 — Struct Array from File**: Write a program that asks the user how many students to enter, allocates an array of `Student` structs, fills them from user input, finds the student with the highest score, and frees all memory.

**Exercise 5 — Memory Error Hunt**: The following program contains three memory errors. Identify and fix each one:

```c
#include <stdlib.h>
#include <string.h>

int main(void) {
    int *a = malloc(5 * sizeof(int));
    a[5] = 100;

    char *s = malloc(10);
    strcpy(s, "Hello, World!");

    int *b = malloc(sizeof(int));
    free(b);
    *b = 42;

    free(a);
    free(s);
    return 0;
}
```

---

## Next Steps

You now have full control over memory allocation and deallocation in C. In the next lesson, [File I/O](./10_File_IO.md), you will learn how to read and write data to files — combining dynamic memory with file operations to build programs that persist data between runs.
