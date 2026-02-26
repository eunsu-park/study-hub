# C Language Basics Quick Review

**Previous**: [C Language Environment Setup](./01_Environment_Setup.md) | **Next**: [Project 1: Basic Arithmetic Calculator](./03_Project_Calculator.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish C's static typing, manual memory management, and compiled execution from Python or JavaScript equivalents
2. Explain the role of `#include`, `main`, semicolons, and curly braces in a minimal C program
3. Identify the correct format specifier for each basic data type when using `printf` and `scanf`
4. Implement pointer declaration, address-of (`&`), and dereference (`*`) to pass values by reference
5. Build and iterate over fixed-size arrays, and explain the equivalence of `arr[i]` and `*(arr + i)`
6. Apply `strlen`, `strcpy`, `strcat`, and `strcmp` for safe string manipulation in C
7. Design structs with `typedef`, access members through both dot and arrow operators, and pass structs to functions via pointers
8. Apply `malloc`, `free`, and NULL-check patterns to allocate and release heap memory without leaks

---

If you already know Python, JavaScript, or another high-level language, most programming concepts are familiar -- variables, loops, functions, and data structures. What makes C different is how close it sits to the hardware: you choose the exact size of every integer, manage every byte of memory yourself, and talk directly to the operating system. This lesson gives you a rapid-fire tour of C's core syntax so you can start building real projects in the next lesson.

> A summary of core C syntax for those with experience in other programming languages

## 1. Characteristics of C

### Comparison with Other Languages

| Feature | Python/JS | C |
|------|-----------|---|
| **Memory Management** | Automatic (GC) | Manual (malloc/free) |
| **Type System** | Dynamic typing | Static typing |
| **Execution** | Interpreter | Compiled |
| **Abstraction Level** | High | Low (close to hardware) |

### Why Learn C

- Systems programming (OS, drivers)
- Embedded systems
- Performance-critical applications
- Understanding foundations of other languages (Python, Ruby are written in C)

---

## 2. Basic Structure

```c
#include <stdio.h>    // Include header file (preprocessor directive)

// main function: Program entry point
int main(void) {
    printf("Hello, C!\n");
    return 0;         // 0 = normal exit
}
```

### Comparison with Python

```python
# Python
print("Hello, Python!")
```

```c
// C
#include <stdio.h>
int main(void) {
    printf("Hello, C!\n");
    return 0;
}
```

**C Characteristics:**
- Semicolon `;` required
- Curly braces `{}` for block delimiting
- Explicit main function
- Header file include required

---

## 3. Data Types

### Basic Data Types

```c
#include <stdio.h>

int main(void) {
    // Integer types
    char c = 'A';           // 1 byte (-128 ~ 127)
    short s = 100;          // 2 bytes
    int i = 1000;           // 4 bytes (typically)
    long l = 100000L;       // 4 or 8 bytes
    long long ll = 100000000000LL;  // 8 bytes

    // Unsigned integers
    unsigned int ui = 4000000000U;

    // Floating-point types
    float f = 3.14f;        // 4 bytes
    double d = 3.14159265;  // 8 bytes

    // Output
    printf("char: %c (%d)\n", c, c);  // A (65)
    printf("int: %d\n", i);
    printf("float: %f\n", f);
    printf("double: %.8f\n", d);

    return 0;
}
```

### Format Specifiers (printf)

| Specifier | Type | Example |
|--------|------|------|
| `%d` | int | `printf("%d", 42)` |
| `%u` | unsigned int | `printf("%u", 42)` |
| `%ld` | long | `printf("%ld", 42L)` |
| `%f` | float/double | `printf("%f", 3.14)` |
| `%c` | char | `printf("%c", 'A')` |
| `%s` | string | `printf("%s", "hello")` |
| `%p` | pointer address | `printf("%p", &x)` |
| `%x` | hexadecimal | `printf("%x", 255)` → ff |

### sizeof Operator

```c
printf("int size: %zu bytes\n", sizeof(int));
printf("double size: %zu bytes\n", sizeof(double));
printf("pointer size: %zu bytes\n", sizeof(int*));
```

---

## 4. Pointers (Core of C!)

### What is a Pointer?

**A variable that stores a memory address.**

```
Memory:
Address    Value
0x1000     42      ← int x = 42;
0x1004     0x1000  ← int *p = &x;  (stores address of x)
```

### Basic Syntax

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;      // p stores the address of x

    printf("Value of x: %d\n", x);        // 42
    printf("Address of x: %p\n", &x);     // 0x7fff...
    printf("Value of p (address): %p\n", p); // 0x7fff... (same address)
    printf("Value pointed by p: %d\n", *p);  // 42 (dereferencing)

    // Modify value through pointer
    *p = 100;
    printf("New value of x: %d\n", x);     // 100

    return 0;
}
```

### Pointer Operators

| Operator | Meaning | Example |
|--------|------|------|
| `&` | Address operator | `&x` → address of x |
| `*` | Dereference operator | `*p` → value pointed by p |

### Why Do We Need Pointers?

```c
// Problem: C passes values by copy (call by value)
void wrong_swap(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    // Original values unchanged!
}

// Solution: Pass addresses using pointers
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    // Original values changed!
}

int main(void) {
    int x = 10, y = 20;

    wrong_swap(x, y);
    printf("After wrong_swap: x=%d, y=%d\n", x, y);  // 10, 20 (no change)

    swap(&x, &y);
    printf("After swap: x=%d, y=%d\n", x, y);  // 20, 10

    return 0;
}
```

---

## 5. Arrays

### Basic Arrays

```c
#include <stdio.h>

int main(void) {
    // Array declaration and initialization
    int numbers[5] = {10, 20, 30, 40, 50};

    // Access
    printf("%d\n", numbers[0]);  // 10
    printf("%d\n", numbers[4]);  // 50

    // Size
    int size = sizeof(numbers) / sizeof(numbers[0]);
    printf("Array size: %d\n", size);  // 5

    // Iteration
    for (int i = 0; i < size; i++) {
        printf("numbers[%d] = %d\n", i, numbers[i]);
    }

    return 0;
}
```

### Relationship Between Arrays and Pointers

```c
int arr[5] = {1, 2, 3, 4, 5};

// Array name is the address of the first element
printf("%p\n", arr);      // Address of first element
printf("%p\n", &arr[0]);  // Same address

// Pointer arithmetic
int *p = arr;
printf("%d\n", *p);       // 1 (arr[0])
printf("%d\n", *(p + 1)); // 2 (arr[1])
printf("%d\n", *(p + 2)); // 3 (arr[2])

// arr[i] == *(arr + i)
```

### Strings (char arrays)

```c
#include <stdio.h>
#include <string.h>  // String functions

int main(void) {
    // String is char array + null terminator '\0'
    char str1[] = "Hello";        // Automatically adds '\0'
    char str2[10] = "World";
    char str3[] = {'H', 'i', '\0'};

    printf("%s\n", str1);         // Hello
    printf("Length: %zu\n", strlen(str1));  // 5

    // String copy
    char dest[20];
    strcpy(dest, str1);           // dest = "Hello"

    // String concatenation
    strcat(dest, " ");
    strcat(dest, str2);           // dest = "Hello World"
    printf("%s\n", dest);

    // String comparison
    if (strcmp(str1, "Hello") == 0) {
        printf("Equal!\n");
    }

    return 0;
}
```

---

## 6. Functions

### Basic Functions

```c
#include <stdio.h>

// Function declaration (prototype)
int add(int a, int b);
void greet(const char *name);

int main(void) {
    int result = add(3, 5);
    printf("3 + 5 = %d\n", result);

    greet("Alice");
    return 0;
}

// Function definition
int add(int a, int b) {
    return a + b;
}

void greet(const char *name) {
    printf("Hello, %s!\n", name);
}
```

### Passing Arrays to Functions

```c
// Arrays are passed as pointers (no size information)
void print_array(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Or use this notation (same meaning)
void print_array2(int arr[], int size) {
    // ...
}

int main(void) {
    int nums[] = {1, 2, 3, 4, 5};
    print_array(nums, 5);
    return 0;
}
```

---

## 7. Structures

### Basic Structure

```c
#include <stdio.h>
#include <string.h>

// Structure definition
struct Person {
    char name[50];
    int age;
    float height;
};

int main(void) {
    // Structure variable declaration and initialization
    struct Person p1 = {"John Doe", 25, 175.5};

    // Member access (. operator)
    printf("Name: %s\n", p1.name);
    printf("Age: %d\n", p1.age);

    // Modify member
    p1.age = 26;
    strcpy(p1.name, "Jane Smith");

    return 0;
}
```

### Simplify with typedef

```c
typedef struct {
    char name[50];
    int age;
} Person;  // Now use without 'struct' keyword

int main(void) {
    Person p1 = {"John Doe", 25};
    printf("%s\n", p1.name);
    return 0;
}
```

### Pointers and Structures

```c
typedef struct {
    char name[50];
    int age;
} Person;

void birthday(Person *p) {
    p->age++;  // Use -> operator for pointers
    // Same as (*p).age++;
}

int main(void) {
    Person p1 = {"John Doe", 25};

    birthday(&p1);
    printf("Age: %d\n", p1.age);  // 26

    // Access via pointer
    Person *ptr = &p1;
    printf("Name: %s\n", ptr->name);

    return 0;
}
```

---

## 8. Dynamic Memory Allocation

### malloc / free

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // Dynamically allocate one integer
    int *p = (int *)malloc(sizeof(int));
    if (p == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    *p = 42;
    printf("%d\n", *p);
    free(p);  // Free memory (required!)

    // Dynamically allocate array
    int n = 5;
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        return 1;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);  // Must free array too!

    return 0;
}
```

### Beware of Memory Leaks

```c
// Bad example: Memory leak
void bad(void) {
    int *p = malloc(sizeof(int));
    *p = 42;
    // No free(p); → Memory leak!
}

// Good example
void good(void) {
    int *p = malloc(sizeof(int));
    if (p == NULL) return;
    *p = 42;
    // After use...
    free(p);
    p = NULL;  // Prevent dangling pointer
}
```

---

## 9. Header Files

### Header File Structure

```c
// utils.h
#ifndef UTILS_H      // include guard
#define UTILS_H

// Function declarations
int add(int a, int b);
int subtract(int a, int b);

// Structure definition
typedef struct {
    int x, y;
} Point;

#endif
```

```c
// utils.c
#include "utils.h"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}
```

```c
// main.c
#include <stdio.h>
#include "utils.h"

int main(void) {
    printf("%d\n", add(3, 5));
    Point p = {10, 20};
    printf("(%d, %d)\n", p.x, p.y);
    return 0;
}
```

### Compilation

```bash
gcc main.c utils.c -o program
```

---

## 10. Key Differences Summary (Python → C)

| Python | C |
|--------|---|
| `print("Hello")` | `printf("Hello\n");` |
| `x = 10` | `int x = 10;` |
| `if x > 5:` | `if (x > 5) {` |
| `for i in range(5):` | `for (int i = 0; i < 5; i++) {` |
| `def func(x):` | `int func(int x) {` |
| `class Person:` | `struct Person {` |
| Automatic memory | `malloc()` / `free()` |
| `len(arr)` | `sizeof(arr)/sizeof(arr[0])` |

---

## Exercises

### Exercise 1: Data Type Sizes and Format Specifiers

Write a program that prints the size (in bytes) and a sample value for every data type covered in Section 3: `char`, `short`, `int`, `long`, `long long`, `unsigned int`, `float`, and `double`. Use the correct format specifier for each type. Then answer:

1. Why is `%zu` used for `sizeof` output rather than `%d`?
2. What happens if you use `%d` to print a `long long` value on a 64-bit system?

### Exercise 2: Pointer Swap Dissection

Copy the `wrong_swap` / `swap` example from Section 4 and extend it:

1. Add `printf` statements inside `wrong_swap` to print the *addresses* of `a` and `b` using `%p`. Do the same inside `swap` for the pointer parameters.
2. Call both functions in `main` and print the addresses of `x` and `y` before each call.
3. Confirm in writing: why do the addresses printed inside `wrong_swap` differ from the addresses of `x` and `y`, while those inside `swap` are the same?

### Exercise 3: Array and Pointer Arithmetic

Write a program that declares `int arr[] = {10, 20, 30, 40, 50}` and then:

1. Iterates through the array using an index (`arr[i]`), printing each element.
2. Iterates again using a pointer (`int *p = arr; ... *(p + i)`), printing each element.
3. Iterates a third time by advancing the pointer itself (`p++`) without using an index.
4. Uses `sizeof` to compute the number of elements and confirm all three loops print identical output.

### Exercise 4: Dynamic String Builder

Implement a function `char *build_greeting(const char *name)` that:

1. Dynamically allocates exactly the right amount of memory to hold the string `"Hello, <name>!"` (use `strlen` and `malloc`).
2. Constructs the string using `strcpy` and `strcat`.
3. Returns the pointer to the caller, who is responsible for calling `free`.

Write `main` to call `build_greeting`, print the result, and free it. Run the program under Valgrind (if available) or add a manual check to confirm no memory is leaked.

### Exercise 5: Struct-Based Student Record

Define a `typedef struct` called `Student` with fields `char name[64]`, `int id`, and `float gpa`. Then:

1. Dynamically allocate an array of 3 `Student` structs using `malloc`.
2. Write a function `void print_student(const Student *s)` that prints a student's details using the `->` operator.
3. Write a function `void raise_gpa(Student *s, float delta)` that adds `delta` to the student's GPA (cap at 4.0).
4. Call `raise_gpa` on each student, then call `print_student` to verify, and finally `free` the array.

---

## Next Steps

Now let's build actual projects!

[Project 1: Basic Arithmetic Calculator](./03_Project_Calculator.md) → Start the first project!
