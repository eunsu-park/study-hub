# Arrays and Strings

**Previous**: [Functions and Scope](./05_Functions_and_Scope.md) | **Next**: [Pointers Fundamentals](./07_Pointers_Fundamentals.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare, initialize, and iterate over one-dimensional arrays
2. Work with multi-dimensional arrays and understand memory layout
3. Compute array size using `sizeof` and pass arrays to functions
4. Manipulate C strings using standard library functions (`strlen`, `strcpy`, `strcat`, `strcmp`, `strncpy`, `snprintf`)
5. Explain the null terminator and distinguish string literals from char arrays

---

Arrays let you store a fixed number of values of the same type in contiguous memory. Strings in C are simply arrays of characters terminated by a special null byte. Understanding both is essential because nearly every non-trivial C program relies on them for data storage and text processing.

## 1. Array Declaration and Initialization

An array declaration specifies the element type and a compile-time constant size.

```c
int scores[5];               /* uninitialized — contains garbage values */
int primes[5] = {2, 3, 5, 7, 11};  /* fully initialized */
int zeros[5] = {0};          /* partial init — first element 0, rest auto-zeroed */
int partial[5] = {10, 20};   /* 10, 20, 0, 0, 0 */
```

When you provide an initializer list, the compiler can infer the size:

```c
int data[] = {1, 2, 3, 4};   /* compiler deduces size = 4 */
```

**C99 Designated Initializers** let you set specific indices:

```c
int sparse[10] = {
    [0] = 100,
    [5] = 500,
    [9] = 900
};
/* Elements at other indices are zero */
```

---

## 2. Accessing and Modifying Elements

Array elements are accessed with zero-based indexing using the `[]` operator.

```c
#include <stdio.h>

int main(void) {
    int temps[7] = {22, 25, 19, 28, 31, 27, 23};

    /* Read */
    printf("Monday: %d°C\n", temps[0]);

    /* Write */
    temps[2] = 21;

    /* Iterate */
    for (int i = 0; i < 7; i++) {
        printf("Day %d: %d°C\n", i + 1, temps[i]);
    }
    return 0;
}
```

**No bounds checking**: C does not verify that your index is within the declared size. Accessing `temps[7]` or `temps[-1]` is undefined behavior — the program may crash, produce wrong results, or appear to work until it doesn't.

| Common Bug | Description |
|------------|-------------|
| Off-by-one | Looping `i <= size` instead of `i < size` |
| Uninitialized read | Reading from an array that was never initialized |
| Negative index | Using a signed variable that becomes negative |

---

## 3. Multi-Dimensional Arrays

A 2D array is an array of arrays. C stores them in **row-major order** — all elements of row 0 come first in memory, then row 1, and so on.

```c
#include <stdio.h>

int main(void) {
    int matrix[3][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    /* Print the matrix */
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            printf("%3d ", matrix[r][c]);
        }
        printf("\n");
    }
    return 0;
}
```

Memory layout for `matrix[3][4]`:

```
Address:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11]
Element:   1    2    3    4    5    6    7    8    9   10   11   12
Row:      |--- row 0 ---|--- row 1 ---|--- row 2 ---|
```

The element `matrix[r][c]` is located at offset `r * 4 + c` from the start.

---

## 4. Array Size

The `sizeof` operator returns the total byte count of an array when applied to the array name directly.

```c
#include <stdio.h>

int main(void) {
    int data[] = {10, 20, 30, 40, 50};

    size_t total_bytes = sizeof(data);         /* e.g. 20 on a system where int = 4 bytes */
    size_t element_size = sizeof(data[0]);     /* 4 */
    size_t count = sizeof(data) / sizeof(data[0]);  /* 5 */

    printf("Array has %zu elements\n", count);
    return 0;
}
```

**Why this fails with pointers**: When an array is passed to a function, it decays to a pointer. `sizeof(pointer)` gives the pointer size (typically 8 bytes on 64-bit), not the array size. This is one of the most common C pitfalls.

```c
void print_size(int arr[]) {
    /* sizeof(arr) == sizeof(int *) == 8, NOT the array size */
    printf("Inside function: %zu\n", sizeof(arr));  /* 8 */
}
```

---

## 5. Passing Arrays to Functions

Because arrays decay to pointers when passed to functions, you must always pass the size as a separate parameter.

```c
#include <stdio.h>

double average(const int arr[], size_t n) {
    long sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (double)sum / n;
}

int main(void) {
    int scores[] = {85, 92, 78, 96, 88};
    size_t n = sizeof(scores) / sizeof(scores[0]);

    printf("Average: %.1f\n", average(scores, n));
    return 0;
}
```

| Parameter Style | Equivalent To | Notes |
|----------------|---------------|-------|
| `int arr[]` | `int *arr` | Size information lost |
| `int arr[5]` | `int *arr` | The `5` is ignored by compiler |
| `const int arr[]` | `const int *arr` | Prevents modification |

For 2D arrays, you must specify the column count:

```c
void print_matrix(int rows, int cols, int mat[][4]) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%d ", mat[r][c]);
        }
        printf("\n");
    }
}
```

---

## 6. Character Arrays and Strings

A **C string** is a character array whose last meaningful character is followed by a **null terminator** `'\0'` (the byte value 0).

```c
char greeting[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
char greeting2[] = "Hello";  /* compiler adds '\0' automatically, size = 6 */
```

**String literals** like `"Hello"` are stored in read-only memory. Assigning one to a `char *` gives you a pointer to that immutable data:

```c
char arr[] = "Hello";    /* mutable copy on the stack — safe to modify */
char *ptr  = "Hello";    /* pointer to read-only literal — modifying is UB */

arr[0] = 'J';   /* OK: arr is now "Jello" */
/* ptr[0] = 'J';  UNDEFINED BEHAVIOR — do not modify string literals */
```

| Aspect | `char arr[] = "Hi"` | `char *ptr = "Hi"` |
|--------|---------------------|---------------------|
| Storage | Stack (mutable copy) | Read-only section |
| `sizeof` | 3 (includes `'\0'`) | 8 (pointer size) |
| Modifiable | Yes | No (undefined behavior) |

---

## 7. String Functions

Include `<string.h>` for all string functions. Always ensure destination buffers are large enough.

### Length

```c
#include <string.h>

char msg[] = "Hello";
size_t len = strlen(msg);  /* 5 — does NOT count '\0' */
```

### Copy

```c
char dest[20];
strcpy(dest, "Hello");        /* copies "Hello\0" to dest */
strncpy(dest, "Hello", 19);   /* copies at most 19 chars, may NOT null-terminate */
dest[19] = '\0';              /* always null-terminate after strncpy */
```

### Concatenate

```c
char buf[50] = "Hello";
strcat(buf, ", ");            /* buf = "Hello, " */
strcat(buf, "World!");        /* buf = "Hello, World!" */

/* Safer version with length limit */
strncat(buf, " Bye", sizeof(buf) - strlen(buf) - 1);
```

### Compare

```c
int result = strcmp("apple", "banana");
/* result < 0: "apple" comes before "banana" */
/* result == 0: strings are equal */
/* result > 0: first string comes after second */

/* Compare at most n characters */
int cmp = strncmp("Hello", "Help", 3);  /* 0 — first 3 chars match */
```

### Formatted Printing to Strings

```c
char buf[100];
int age = 30;
snprintf(buf, sizeof(buf), "Age: %d years", age);
/* buf = "Age: 30 years" */
/* snprintf never writes past the buffer size — always prefer over sprintf */
```

### Memory Functions

`<string.h>` also provides three essential functions that operate on raw bytes rather than null-terminated strings. They work on any data type, not just characters.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `memcpy` | `void *memcpy(void *dest, const void *src, size_t n)` | Copy `n` bytes; regions must **not** overlap |
| `memmove` | `void *memmove(void *dest, const void *src, size_t n)` | Copy `n` bytes; safe even if regions overlap |
| `memset` | `void *memset(void *dest, int val, size_t n)` | Fill `n` bytes with `val` |

```c
#include <string.h>
#include <stdio.h>

int main(void) {
    int src[5] = {1, 2, 3, 4, 5};
    int dst[5];

    memcpy(dst, src, sizeof(src));   /* fast copy — src and dst do not overlap */

    /* Zero-initialize an array — the most common use of memset */
    int arr[100];
    memset(arr, 0, sizeof(arr));     /* every byte set to 0 */

    /* Overlapping copy: shift elements right by one position */
    /* memcpy here would be undefined behavior — use memmove */
    memmove(src + 1, src, 4 * sizeof(int));  /* safe overlap */
    src[0] = 0;
    /* src is now {0, 1, 2, 3, 4} */

    return 0;
}
```

> **Why the distinction matters**: `memcpy` may be implemented with SIMD instructions that read/write chunks larger than a byte, making it faster but unsafe when source and destination overlap. `memmove` guarantees correctness for overlapping regions, typically by copying through a temporary buffer or choosing copy direction based on address order.

---

## 8. String Input

### fgets (Recommended)

`fgets` reads up to `n-1` characters and always null-terminates. It includes the newline character if there is room.

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char name[50];

    printf("Enter your name: ");
    if (fgets(name, sizeof(name), stdin) != NULL) {
        /* Remove trailing newline if present */
        name[strcspn(name, "\n")] = '\0';
        printf("Hello, %s!\n", name);
    }
    return 0;
}
```

### scanf with Strings (Risky)

```c
char word[20];
scanf("%19s", word);  /* reads one word, max 19 chars + '\0' */
/* Without the width limit, scanf can overflow the buffer */
```

| Function | Buffer Overflow? | Reads Spaces? | Newline Handling |
|----------|-----------------|---------------|------------------|
| `fgets` | No (if size correct) | Yes | Included in buffer |
| `scanf("%s", ...)` | Yes (without width) | No (stops at whitespace) | Left in input buffer |
| `gets` | **Always** | Yes | Removed — **NEVER USE** |

The `gets` function was removed from the C11 standard because it cannot prevent buffer overflow under any circumstances.

---

## Exercises

**Exercise 1 — Array Statistics**: Write a program that reads 10 integers into an array and prints the minimum, maximum, and average.

**Exercise 2 — Reverse Array**: Write a function `void reverse(int arr[], size_t n)` that reverses an array in place. Test it with arrays of both odd and even length.

**Exercise 3 — Matrix Transpose**: Write a function that takes a 3x3 integer matrix and prints its transpose (rows become columns).

**Exercise 4 — String Reversal**: Write a function `void str_reverse(char s[])` that reverses a C string in place without using any library functions. Handle the empty string case.

**Exercise 5 — Word Counter**: Write a program that reads a line of text with `fgets` and counts the number of words (sequences of non-space characters). Handle multiple consecutive spaces correctly.

---

## Next Steps

You now know how to store collections of data in arrays and manipulate text with C strings. In the next lesson, [Pointers Fundamentals](./07_Pointers_Fundamentals.md), you will learn how pointers work under the hood — the mechanism that makes arrays, strings, and dynamic memory possible in C.
