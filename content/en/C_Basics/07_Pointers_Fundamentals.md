# Pointers Fundamentals

**Previous**: [Arrays and Strings](./06_Arrays_and_Strings.md) | **Next**: [Structs and Unions](./08_Structs_and_Unions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare pointer variables and use the address-of (`&`) and dereference (`*`) operators
2. Draw memory diagrams showing pointer relationships
3. Explain pointer arithmetic and the equivalence of `arr[i]` and `*(arr + i)`
4. Implement pass-by-reference using pointer parameters
5. Use NULL pointers safely and explain dangling pointer risks
6. Distinguish pointers to different types and understand pointer size

---

Pointers are the single most important concept in C. A pointer is a variable that holds the memory address of another variable. Every dynamic data structure, every function that modifies its arguments, and every system-level API in C depends on pointers. This lesson builds your understanding from the ground up with diagrams, examples, and common pitfalls.

## 1. What Is a Pointer?

Every variable in a running program occupies one or more bytes of memory, and each byte has a unique numerical **address**. A pointer is a variable that stores such an address.

Think of memory as a long row of numbered mailboxes. Each mailbox (byte) has an address printed on it. A pointer is a slip of paper on which you write down one of those addresses — it tells you *where* to look, not what is stored there.

```
Memory (simplified):
Address:  0x1000   0x1004   0x1008   0x100C
          +--------+--------+--------+--------+
          |   42   |  0x1000|  ...   |  ...   |
          +--------+--------+--------+--------+
Variable:    x         p
```

In this diagram, `x` lives at address `0x1000` and holds the value `42`. The pointer `p` lives at `0x1004` and holds the value `0x1000` — the address of `x`.

---

## 2. Declaring and Initializing Pointers

A pointer variable is declared by placing `*` between the base type and the variable name.

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;   /* p stores the address of x */

    printf("Value of x:       %d\n", x);        /* 42 */
    printf("Address of x:     %p\n", (void *)&x); /* e.g. 0x7ffd1234abcd */
    printf("Value of p:       %p\n", (void *)p);  /* same address */
    printf("Value at *p:      %d\n", *p);        /* 42 */

    return 0;
}
```

| Symbol | Meaning | Example |
|--------|---------|---------|
| `int *p` | Declare `p` as a pointer to `int` | `int *p;` |
| `&x` | Address-of operator — get the address of `x` | `p = &x;` |
| `*p` | Dereference operator — access the value at the address `p` holds | `printf("%d", *p);` |

**Initializing to NULL**: When you don't yet have an address to assign, initialize the pointer to `NULL` (defined in `<stddef.h>` or `<stdio.h>`).

```c
int *p = NULL;   /* safe — clearly indicates "points to nothing" */
```

---

## 3. Dereferencing

Dereferencing a pointer means following the address to read or write the value stored there.

```c
#include <stdio.h>

int main(void) {
    int a = 10;
    int *p = &a;

    /* Read through pointer */
    printf("a = %d\n", *p);   /* 10 */

    /* Write through pointer */
    *p = 25;
    printf("a = %d\n", a);    /* 25 — a was modified via *p */

    return 0;
}
```

### const with Pointers

The `const` keyword can protect the pointed-to value, the pointer itself, or both.

```c
int x = 10, y = 20;

const int *p1 = &x;     /* pointer to const int — cannot modify *p1 */
/* *p1 = 30;  ERROR */
p1 = &y;                /* OK — can change where p1 points */

int *const p2 = &x;     /* const pointer to int — cannot change p2 itself */
*p2 = 30;               /* OK — can modify the value */
/* p2 = &y;  ERROR */

const int *const p3 = &x;  /* const pointer to const int — nothing changeable */
```

Read declarations **right to left**: `const int *p` → "p is a pointer to int that is const."

---

## 4. Pointer Arithmetic

When you add an integer `n` to a pointer `p`, the compiler advances the address by `n * sizeof(*p)` bytes, not `n` bytes. This is what makes pointer traversal of arrays so natural.

```c
#include <stdio.h>

int main(void) {
    int arr[] = {10, 20, 30, 40, 50};
    int *p = arr;  /* points to arr[0] */

    printf("*p       = %d\n", *p);       /* 10 */
    printf("*(p + 1) = %d\n", *(p + 1)); /* 20 */
    printf("*(p + 4) = %d\n", *(p + 4)); /* 50 */

    /* Advance the pointer */
    p += 2;
    printf("*p after p+=2: %d\n", *p);   /* 30 */

    return 0;
}
```

| Expression | Result | Explanation |
|-----------|--------|-------------|
| `p + 1` | Address of next `int` | Advances by `sizeof(int)` bytes |
| `p - 1` | Address of previous `int` | Goes back by `sizeof(int)` bytes |
| `p2 - p1` | Number of elements between | Returns `ptrdiff_t`, not byte count |
| `p++` | Advance `p` to next element | Post-increment |

---

## 5. Arrays and Pointers

An array name, in most expressions, **decays** to a pointer to its first element. This means `arr` and `&arr[0]` produce the same address.

```c
#include <stdio.h>

int main(void) {
    int arr[] = {100, 200, 300};

    /* These are equivalent */
    printf("%d\n", arr[1]);        /* 200 */
    printf("%d\n", *(arr + 1));    /* 200 */

    int *p = arr;
    printf("%d\n", p[2]);          /* 300 — pointer indexing works too */
    printf("%d\n", *(p + 2));      /* 300 */

    return 0;
}
```

The equivalence `arr[i]` == `*(arr + i)` is not just a convenience — it is the **definition** of the `[]` operator in C. In fact, `i[arr]` is also legal (since addition is commutative), though you should never write code that way.

**Key difference**: An array name is **not** a modifiable lvalue. You cannot do `arr++` or `arr = p`. A pointer variable can be reassigned freely.

---

## 6. Pass-by-Reference

C passes all arguments **by value** — the function receives a copy. To let a function modify the caller's variable, pass a **pointer** to that variable.

```c
#include <stdio.h>

/* Classic swap using pointers */
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main(void) {
    int x = 10, y = 20;
    printf("Before: x=%d, y=%d\n", x, y);  /* 10, 20 */

    swap(&x, &y);
    printf("After:  x=%d, y=%d\n", x, y);  /* 20, 10 */

    return 0;
}
```

Another common use — a function that returns an error code and writes its result through a pointer:

```c
#include <stdio.h>
#include <stdbool.h>

bool safe_divide(double numerator, double denominator, double *result) {
    if (denominator == 0.0) {
        return false;
    }
    *result = numerator / denominator;
    return true;
}

int main(void) {
    double answer;
    if (safe_divide(10.0, 3.0, &answer)) {
        printf("Result: %.4f\n", answer);  /* 3.3333 */
    } else {
        printf("Division by zero!\n");
    }
    return 0;
}
```

---

## 7. Pointer to Pointer

A pointer to pointer stores the address of another pointer variable. Declared with `**`.

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;
    int **pp = &p;

    printf("x   = %d\n", x);     /* 42 */
    printf("*p  = %d\n", *p);    /* 42 */
    printf("**pp = %d\n", **pp); /* 42 */

    **pp = 99;
    printf("x   = %d\n", x);     /* 99 */

    return 0;
}
```

Memory diagram:

```
  pp            p             x
+--------+   +--------+   +--------+
| &p     |-->| &x     |-->|   99   |
+--------+   +--------+   +--------+
```

**Common use cases**:
- Modifying a pointer inside a function (e.g., `void alloc(int **out)`)
- Arrays of strings (`char *argv[]` is equivalent to `char **argv`)
- Dynamic 2D arrays (array of pointers to rows)

---

## 8. Common Pointer Mistakes

### Uninitialized Pointer

```c
int *p;          /* p holds garbage — some random address */
*p = 10;         /* UNDEFINED BEHAVIOR — writing to unknown memory */
```

**Fix**: Always initialize pointers to `NULL` or a valid address.

### Dangling Pointer

A pointer that refers to memory that has been freed or has gone out of scope.

```c
int *get_local(void) {
    int local = 42;
    return &local;  /* WARNING: local is destroyed when function returns */
}

int main(void) {
    int *p = get_local();
    printf("%d\n", *p);  /* UNDEFINED BEHAVIOR — dangling pointer */
    return 0;
}
```

**Fix**: Never return the address of a local variable. Use `static`, pass a buffer, or allocate on the heap.

### NULL Dereference

```c
int *p = NULL;
*p = 10;         /* CRASH — segmentation fault on most systems */
```

**Fix**: Always check for `NULL` before dereferencing.

```c
if (p != NULL) {
    *p = 10;
}
```

### Wild Pointer

A pointer that has been freed but not set to `NULL`, then used again.

```c
#include <stdlib.h>

int *p = malloc(sizeof(int));
*p = 42;
free(p);
/* p still holds the old address — it's "wild" */
*p = 10;    /* UNDEFINED BEHAVIOR — use-after-free */

/* Fix: set to NULL immediately after free */
free(p);
p = NULL;
```

| Mistake | Symptom | Prevention |
|---------|---------|------------|
| Uninitialized pointer | Random crashes | Initialize to `NULL` or `&var` |
| Dangling pointer | Intermittent corruption | Don't return addresses of locals |
| NULL dereference | Segfault | Check before dereferencing |
| Wild pointer | Use-after-free | Set to `NULL` after `free` |

---

## Exercises

**Exercise 1 — Pointer Basics**: Write a program that declares an `int`, a `float`, and a `char`. For each, print the value, address, and the size of a pointer to it. Verify that all pointer sizes are the same on your platform.

**Exercise 2 — Array Traversal with Pointers**: Write a function `int sum_array(const int *arr, size_t n)` that computes the sum of an array using pointer arithmetic (no `[]` operator). Test it with arrays of various sizes.

**Exercise 3 — Swap Three**: Write a function `void rotate_three(int *a, int *b, int *c)` that rotates three values so that `a` gets `b`'s old value, `b` gets `c`'s old value, and `c` gets `a`'s old value. Call it from `main` and print before/after.

**Exercise 4 — Find in Array**: Write a function `int *find(int *arr, size_t n, int target)` that returns a pointer to the first occurrence of `target` in the array, or `NULL` if not found. In `main`, use the returned pointer to modify the found element.

**Exercise 5 — Pointer to Pointer Modification**: Write a function `void allocate_and_set(int **pp, int value)` that allocates memory for a single `int` using `malloc`, stores `value` there, and updates the caller's pointer through `pp`. The caller should print the value and free the memory.

---

## Next Steps

Pointers are the foundation for everything that follows. In the next lesson, [Structs and Unions](./08_Structs_and_Unions.md), you will learn how to group related data into custom types — and use pointers to access their members efficiently with the arrow operator.
