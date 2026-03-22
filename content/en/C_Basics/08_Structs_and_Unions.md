# Structs and Unions

**Previous**: [Pointers Fundamentals](./07_Pointers_Fundamentals.md) | **Next**: [Dynamic Memory](./09_Dynamic_Memory.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define structs with multiple fields and access members using dot and arrow operators
2. Use `typedef` to create convenient type aliases for structs
3. Differentiate structs from unions and explain memory layout differences
4. Define and use enumerations for named integer constants
5. Apply bit fields for compact data representation

---

So far, every variable you have used holds a single value — one integer, one float, one character. Real programs model entities that have multiple attributes: a student has a name, an ID, and a GPA; a pixel has red, green, and blue components. Structs let you group these related pieces of data into a single custom type.

## 1. Defining Structs

The `struct` keyword introduces a new composite type. Each piece of data inside is called a **field** or **member**.

```c
struct Point {
    double x;
    double y;
};

struct Student {
    char name[50];
    int id;
    double gpa;
};
```

You can declare and initialize variables of a struct type:

```c
#include <stdio.h>

struct Point {
    double x;
    double y;
};

int main(void) {
    /* Declaration and initialization */
    struct Point origin = {0.0, 0.0};
    struct Point p1 = {3.5, -2.1};

    /* Declaration then assignment (field by field) */
    struct Point p2;
    p2.x = 1.0;
    p2.y = 4.0;

    printf("p1 = (%.1f, %.1f)\n", p1.x, p1.y);
    return 0;
}
```

---

## 2. Accessing Members

Use the **dot operator** (`.`) to access members of a struct variable.

```c
#include <stdio.h>
#include <string.h>

struct Student {
    char name[50];
    int id;
    double gpa;
};

int main(void) {
    struct Student s;
    strcpy(s.name, "Alice");
    s.id = 1001;
    s.gpa = 3.85;

    printf("Name: %s, ID: %d, GPA: %.2f\n", s.name, s.id, s.gpa);
    return 0;
}
```

### C99 Designated Initializers

You can name the fields during initialization, in any order:

```c
struct Student s = {
    .name = "Bob",
    .gpa = 3.92,
    .id = 1002
};
```

This is especially valuable when a struct has many fields and you want to make the code self-documenting.

### Struct Assignment

Assigning one struct to another copies all members:

```c
struct Point a = {1.0, 2.0};
struct Point b = a;  /* b.x = 1.0, b.y = 2.0 — full copy */
```

**Caution**: If a struct contains a pointer, the copy is **shallow** — both structs point to the same memory.

---

## 3. typedef

Writing `struct Point` everywhere is verbose. The `typedef` keyword creates an alias.

```c
typedef struct {
    double x;
    double y;
} Point;

/* Now you can write: */
Point p = {1.0, 2.0};
```

A common convention for self-referencing structs (e.g., linked lists) requires naming the struct tag:

```c
typedef struct Node {
    int data;
    struct Node *next;   /* must use "struct Node" here, not "Node" */
} Node;
```

| Style | Declaration | Usage |
|-------|------------|-------|
| Without typedef | `struct Point { ... };` | `struct Point p;` |
| With typedef | `typedef struct { ... } Point;` | `Point p;` |
| Both tag and typedef | `typedef struct Point { ... } Point;` | Either form works |

---

## 4. Structs and Pointers

When you have a pointer to a struct, you access members with the **arrow operator** (`->`).

```c
#include <stdio.h>
#include <math.h>

typedef struct {
    double x;
    double y;
} Point;

double distance(const Point *a, const Point *b) {
    double dx = a->x - b->x;   /* equivalent to (*a).x - (*b).x */
    double dy = a->y - b->y;
    return sqrt(dx * dx + dy * dy);
}

int main(void) {
    Point p1 = {0.0, 0.0};
    Point p2 = {3.0, 4.0};

    printf("Distance: %.2f\n", distance(&p1, &p2));  /* 5.00 */
    return 0;
}
```

The arrow operator `p->member` is syntactic sugar for `(*p).member`. The parentheses are necessary because `.` has higher precedence than `*`.

### Passing Structs to Functions

| Method | Syntax | Copies Data? | Can Modify Original? |
|--------|--------|-------------|---------------------|
| By value | `void f(Point p)` | Yes — full copy | No |
| By pointer | `void f(Point *p)` | No — just the address | Yes |
| By const pointer | `void f(const Point *p)` | No | No (read-only) |

For small structs (2-3 fields), passing by value is fine. For larger structs, pass by `const` pointer to avoid expensive copies.

---

## 5. Nested Structs

Structs can contain other structs, modeling hierarchical data naturally.

```c
#include <stdio.h>

typedef struct {
    int day;
    int month;
    int year;
} Date;

typedef struct {
    char name[50];
    int id;
    Date hire_date;
    Date birth_date;
} Employee;

int main(void) {
    Employee emp = {
        .name = "Alice",
        .id = 42,
        .hire_date = {15, 3, 2023},
        .birth_date = {.day = 10, .month = 7, .year = 1995}
    };

    printf("%s was hired on %02d/%02d/%04d\n",
           emp.name,
           emp.hire_date.day,
           emp.hire_date.month,
           emp.hire_date.year);

    return 0;
}
```

With a pointer: `emp_ptr->hire_date.day` (arrow for the first level, dot for the nested struct since `hire_date` is not itself a pointer).

---

## 6. Unions

A **union** looks like a struct but all members **share the same memory**. The union is only as large as its largest member. Only one member holds a valid value at any given time.

```c
#include <stdio.h>

union Data {
    int i;
    float f;
    char str[20];
};

int main(void) {
    union Data d;

    printf("Size of union: %zu bytes\n", sizeof(d));  /* 20 — size of str */

    d.i = 42;
    printf("d.i = %d\n", d.i);   /* 42 */

    d.f = 3.14f;
    printf("d.f = %.2f\n", d.f); /* 3.14 */
    printf("d.i = %d\n", d.i);   /* garbage — overwritten by d.f */

    return 0;
}
```

### Tagged Union Pattern

To know which member is currently valid, pair a union with an enum "tag":

```c
#include <stdio.h>

typedef enum { VAL_INT, VAL_FLOAT, VAL_STRING } ValueType;

typedef struct {
    ValueType type;
    union {
        int i;
        float f;
        char s[32];
    } data;
} Value;

void print_value(const Value *v) {
    switch (v->type) {
        case VAL_INT:    printf("int: %d\n", v->data.i);    break;
        case VAL_FLOAT:  printf("float: %.2f\n", v->data.f); break;
        case VAL_STRING: printf("string: %s\n", v->data.s);  break;
    }
}

int main(void) {
    Value v1 = {.type = VAL_INT, .data.i = 42};
    Value v2 = {.type = VAL_STRING, .data.s = "Hello"};

    print_value(&v1);  /* int: 42 */
    print_value(&v2);  /* string: Hello */
    return 0;
}
```

| Feature | struct | union |
|---------|--------|-------|
| Memory | Sum of all member sizes (+ padding) | Size of largest member |
| Active members | All at once | One at a time |
| Use case | Group related data | Variant/polymorphic data |

---

## 7. Enumerations

An `enum` defines a set of named integer constants.

```c
#include <stdio.h>

enum Direction { NORTH, EAST, SOUTH, WEST };
/* NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3 */

enum HttpStatus {
    HTTP_OK         = 200,
    HTTP_NOT_FOUND  = 404,
    HTTP_SERVER_ERR = 500
};

int main(void) {
    enum Direction dir = NORTH;

    switch (dir) {
        case NORTH: printf("Going north\n"); break;
        case EAST:  printf("Going east\n");  break;
        case SOUTH: printf("Going south\n"); break;
        case WEST:  printf("Going west\n");  break;
    }

    printf("HTTP OK = %d\n", HTTP_OK);  /* 200 */
    return 0;
}
```

Enums are type-safe documentation. Instead of magic numbers scattered through your code, give them meaningful names.

| Feature | Description |
|---------|-------------|
| Default values | Start at 0, auto-increment |
| Explicit values | Assign with `= value` |
| Underlying type | `int` (in standard C) |
| Scope | Global (not scoped to enum name) |

---

## 8. Bit Fields

Bit fields let you specify the exact number of bits a struct member should occupy. Useful for flags, hardware register maps, and memory-constrained environments.

```c
#include <stdio.h>

typedef struct {
    unsigned int is_active : 1;   /* 1 bit: 0 or 1 */
    unsigned int priority  : 3;   /* 3 bits: 0-7 */
    unsigned int category  : 4;   /* 4 bits: 0-15 */
} TaskFlags;

int main(void) {
    TaskFlags task = {
        .is_active = 1,
        .priority = 5,
        .category = 12
    };

    printf("Active: %u, Priority: %u, Category: %u\n",
           task.is_active, task.priority, task.category);

    printf("Size of TaskFlags: %zu bytes\n", sizeof(TaskFlags));
    /* Likely 4 bytes — the compiler packs bits into an unsigned int */

    return 0;
}
```

### Hardware Register Example

```c
typedef struct {
    unsigned int enable     : 1;
    unsigned int mode       : 2;
    unsigned int interrupt  : 1;
    unsigned int reserved   : 4;
} ControlRegister;
```

**Portability notes**:
- The order of bit fields within a byte is implementation-defined (may differ between compilers and architectures).
- You cannot take the address of a bit field member (`&task.priority` is illegal).
- Bit fields are best for internal data structures where portability across compilers is not critical. For wire protocols, use explicit bitwise operations instead.

---

## Exercises

**Exercise 1 — Rectangle Struct**: Define a `Rectangle` struct with `width` and `height` (both `double`). Write functions `double area(const Rectangle *r)` and `double perimeter(const Rectangle *r)`. Create several rectangles and print their areas and perimeters.

**Exercise 2 — Student Records**: Define a `Student` struct with name, ID, and an array of 5 grades. Write a function that takes a `const Student *` and returns the average grade. Create an array of 3 students and print a report.

**Exercise 3 — Tagged Union Calculator**: Create a tagged union `Number` that can hold either an `int` or a `double`. Write a function `void print_number(const Number *n)` and a function `Number add_numbers(const Number *a, const Number *b)` that adds two Numbers (promoting to double if either is a double).

**Exercise 4 — Color Enum and Struct**: Define an enum `Color` with RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA. Define a struct `Pixel` with `int x`, `int y`, and `Color color`. Write a function that prints a pixel's info with the color name (not number).

**Exercise 5 — Packed Flags**: Define a bit field struct `FilePermissions` with read, write, execute bits for owner, group, and others (9 bits total). Write functions to set, clear, and display permissions in `rwxrwxrwx` format (like `ls -l`).

---

## Next Steps

You can now create custom data types that model real-world entities. In the next lesson, [Dynamic Memory](./09_Dynamic_Memory.md), you will learn how to allocate structs and arrays on the heap at runtime — essential for building data structures whose size is not known at compile time.
