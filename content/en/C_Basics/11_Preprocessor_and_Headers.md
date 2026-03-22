# Preprocessor and Headers

**Previous**: [File I/O](./10_File_IO.md) | **Next**: [Build Tools and Debugging](./12_Build_Tools_and_Debugging.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the preprocessor phase and how it transforms source code before compilation
2. Use `#include` for standard and user-defined headers
3. Define and use object-like and function-like macros with `#define`
4. Apply conditional compilation with `#ifdef`, `#ifndef`, `#if`, `#else`, `#endif`
5. Write header files with include guards to prevent multiple inclusion

---

The C preprocessor is a text-transformation engine that runs before the compiler ever sees your code. It handles file inclusion, macro expansion, and conditional compilation — three mechanisms that are fundamental to organizing C projects beyond a single file. Understanding the preprocessor is essential because its behavior explains many C-specific idioms, error messages, and project structures.

## 1. What Is the Preprocessor?

When you compile a C file, the translation happens in several phases:

```
Source Code (.c)
      │
      ▼
  Preprocessor  ──── Phase 1: text substitution (#include, #define, #if)
      │
      ▼
  Compiler      ──── Phase 2: parse C code, generate assembly
      │
      ▼
  Assembler     ──── Phase 3: generate object code (.o)
      │
      ▼
  Linker        ──── Phase 4: combine objects into executable
      │
      ▼
  Executable
```

The preprocessor operates on **text** — it knows nothing about C syntax, types, or scope. Every directive starts with `#` and must be the first non-whitespace character on its line.

You can see the preprocessor output with:

```bash
gcc -E main.c -o main.i    # output preprocessed source
```

This expands all `#include`, `#define`, and conditional directives, producing a single file that the compiler then processes.

---

## 2. #include

The `#include` directive copies the entire contents of another file into the current file at the point of the directive.

### Angle Brackets vs Quotes

```c
#include <stdio.h>       /* search system include paths */
#include "myheader.h"    /* search current directory first, then system paths */
```

| Syntax | Search Order | Use For |
|--------|-------------|---------|
| `<header.h>` | System include directories only | Standard library headers |
| `"header.h"` | Current directory, then system paths | Your own project headers |

### Commonly Used Standard Headers

| Header | Provides |
|--------|----------|
| `<stdio.h>` | `printf`, `scanf`, `fopen`, `FILE` |
| `<stdlib.h>` | `malloc`, `free`, `atoi`, `exit`, `rand` |
| `<string.h>` | `strlen`, `strcpy`, `strcmp`, `memcpy` |
| `<math.h>` | `sqrt`, `sin`, `cos`, `pow` (link with `-lm`) |
| `<stdbool.h>` | `bool`, `true`, `false` (C99) |
| `<stdint.h>` | `int32_t`, `uint8_t`, fixed-width types (C99) |
| `<stddef.h>` | `size_t`, `NULL`, `ptrdiff_t` |
| `<ctype.h>` | `isalpha`, `isdigit`, `toupper`, `tolower` |
| `<errno.h>` | `errno`, error codes |
| `<assert.h>` | `assert` macro for debugging checks |
| `<limits.h>` | `INT_MAX`, `INT_MIN`, `CHAR_BIT` |

---

## 3. Object-Like Macros

An object-like macro associates a name with a replacement text. By convention, macro names use `ALL_CAPS`.

```c
#define PI          3.14159265358979
#define MAX_SIZE    1024
#define AUTHOR      "Alice"
#define DEBUG_MODE  1
```

Every occurrence of the macro name in the source is replaced with its definition before compilation:

```c
#include <stdio.h>

#define BUFFER_SIZE 256
#define VERSION     "1.2.0"

int main(void) {
    char buf[BUFFER_SIZE];           /* becomes: char buf[256]; */
    printf("Version: %s\n", VERSION); /* becomes: printf("Version: %s\n", "1.2.0"); */
    return 0;
}
```

### Advantages Over Magic Numbers

- **Readability**: `BUFFER_SIZE` is more descriptive than `256`
- **Maintainability**: Change the value in one place, and it updates everywhere
- **No memory used**: Macros are compile-time text replacements, not variables

### When to Prefer `const` or `enum` Instead

In modern C (C99+), `const` variables and `enum` values are often better choices because they are type-safe and visible to the debugger:

```c
static const double PI = 3.14159265358979;   /* type-safe constant */
enum { MAX_SIZE = 1024 };                     /* integer constant */
```

However, `#define` is still needed for string constants, header guards, and conditional compilation.

---

## 4. Function-Like Macros

A function-like macro takes parameters, enclosed in parentheses immediately after the macro name (no space).

```c
#define SQUARE(x)    ((x) * (x))
#define MAX(a, b)    ((a) > (b) ? (a) : (b))
#define MIN(a, b)    ((a) < (b) ? (a) : (b))
#define ABS(x)       ((x) < 0 ? -(x) : (x))
```

### Parenthesization Rules

**Always parenthesize every parameter and the entire expression.** Without parentheses, operator precedence can cause subtle bugs.

```c
/* BAD — missing parentheses */
#define SQUARE_BAD(x) x * x

int result = SQUARE_BAD(2 + 3);
/* Expands to: 2 + 3 * 2 + 3 = 2 + 6 + 3 = 11 (wrong!) */

/* GOOD — fully parenthesized */
#define SQUARE(x) ((x) * (x))

int result = SQUARE(2 + 3);
/* Expands to: ((2 + 3) * (2 + 3)) = 25 (correct) */
```

### Double Evaluation Pitfall

Macro arguments are substituted textually, so they can be evaluated more than once:

```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int x = 5, y = 3;
int z = MAX(x++, y);
/* Expands to: ((x++) > (y) ? (x++) : (y))
   x is incremented TWICE if x > y — almost certainly a bug */
```

**Rule**: Never pass expressions with side effects to function-like macros. If you need to, use an inline function instead:

```c
static inline int max_int(int a, int b) {
    return a > b ? a : b;
}
```

### Multi-Line Macros

Use the backslash `\` to continue a macro across lines:

```c
#define PRINT_ARRAY(arr, n)          \
    do {                             \
        for (int i = 0; i < (n); i++) \
            printf("%d ", (arr)[i]); \
        printf("\n");                \
    } while (0)
```

The `do { ... } while (0)` idiom ensures the macro works correctly in all contexts (e.g., after an `if` without braces).

---

## 5. Conditional Compilation

Conditional directives let you include or exclude code based on compile-time conditions. This is essential for platform-specific code, debug modes, and feature toggles.

### #ifdef and #ifndef

```c
#define DEBUG

#ifdef DEBUG
    printf("Debug: x = %d\n", x);   /* included only if DEBUG is defined */
#endif

#ifndef RELEASE
    printf("Not a release build\n");  /* included only if RELEASE is NOT defined */
#endif
```

### #if, #elif, #else, #endif

```c
#define VERSION 3

#if VERSION == 1
    printf("Version 1\n");
#elif VERSION == 2
    printf("Version 2\n");
#elif VERSION >= 3
    printf("Version 3 or later\n");
#else
    printf("Unknown version\n");
#endif
```

### Platform-Specific Code

```c
#include <stdio.h>

void clear_screen(void) {
#ifdef _WIN32
    system("cls");
#elif defined(__APPLE__) || defined(__linux__)
    system("clear");
#else
    printf("\033[2J\033[H");  /* ANSI escape fallback */
#endif
}
```

### Compile-Time Feature Flags

You can define macros from the command line:

```bash
gcc -DDEBUG -DVERSION=3 main.c -o main
```

This is equivalent to writing `#define DEBUG` and `#define VERSION 3` at the top of the file.

| Directive | Purpose |
|-----------|---------|
| `#ifdef NAME` | True if `NAME` is defined |
| `#ifndef NAME` | True if `NAME` is not defined |
| `#if expr` | True if constant expression is non-zero |
| `#elif expr` | Else-if chain |
| `#else` | Default branch |
| `#endif` | Ends conditional block |
| `defined(NAME)` | Operator usable in `#if` / `#elif` |

---

## 6. Header Files

A header file (`.h`) declares the **interface** that other source files can use: function prototypes, type definitions, macros, and extern variable declarations.

### Include Guards

Without protection, including the same header twice causes duplicate definition errors. **Include guards** prevent this:

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

double circle_area(double radius);
double circle_circumference(double radius);

typedef struct {
    double x;
    double y;
} Point;

#endif /* MATH_UTILS_H */
```

The first time `math_utils.h` is included, `MATH_UTILS_H` is not defined, so the contents are processed and the macro is defined. On subsequent inclusions, the `#ifndef` test fails and the entire file is skipped.

### #pragma once (Non-Standard but Widely Supported)

```c
#pragma once

double circle_area(double radius);
double circle_circumference(double radius);
```

Most modern compilers (GCC, Clang, MSVC) support `#pragma once`. It is simpler but not part of the C standard. Many projects use both for maximum compatibility:

```c
#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#pragma once

/* ... declarations ... */

#endif
```

### What Goes in a Header

| Belongs in `.h` | Belongs in `.c` |
|-----------------|-----------------|
| Function prototypes | Function definitions (bodies) |
| `typedef`, `struct`, `enum` definitions | Static (file-scope) functions |
| `#define` macros and constants | Global variable definitions |
| `extern` variable declarations | `#include` of own header |
| Inline function definitions | Implementation details |

---

## 7. Multi-File Compilation

Real C programs span multiple `.c` files. Each file is compiled independently into an object file (`.o`), then the linker combines them.

### Example Project

```
project/
├── main.c
├── math_utils.h
└── math_utils.c
```

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#define PI 3.14159265358979

double circle_area(double radius);
double circle_circumference(double radius);

#endif
```

```c
/* math_utils.c */
#include "math_utils.h"

double circle_area(double radius) {
    return PI * radius * radius;
}

double circle_circumference(double radius) {
    return 2.0 * PI * radius;
}
```

```c
/* main.c */
#include <stdio.h>
#include "math_utils.h"

int main(void) {
    double r = 5.0;
    printf("Area: %.2f\n", circle_area(r));
    printf("Circumference: %.2f\n", circle_circumference(r));
    return 0;
}
```

### Compilation Steps

```bash
gcc -c math_utils.c -o math_utils.o   # compile to object file
gcc -c main.c -o main.o               # compile to object file
gcc math_utils.o main.o -o program    # link into executable

# Or all at once:
gcc main.c math_utils.c -o program
```

### Declaration vs Definition

- A **declaration** tells the compiler that something exists and what its type is.
- A **definition** allocates storage or provides the function body.

```c
/* Declaration (in header) */
extern int global_count;        /* variable exists somewhere */
double compute(double x);       /* function exists somewhere */

/* Definition (in .c file) */
int global_count = 0;           /* allocates storage */
double compute(double x) {      /* provides the body */
    return x * x;
}
```

The `extern` keyword says "this variable is defined in another file." Without it, each `.c` file would create its own copy, and the linker would report duplicate symbols.

---

## 8. Other Directives

### #undef

Removes a previously defined macro:

```c
#define TEMP 100
/* ... use TEMP ... */
#undef TEMP
/* TEMP is no longer defined */
```

### #error

Forces a compilation error with a custom message:

```c
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
#error "This code requires C99 or later"
#endif
```

### #pragma

Compiler-specific instructions:

```c
#pragma pack(push, 1)   /* disable struct padding (GCC, MSVC) */
typedef struct {
    char a;
    int b;
} Packed;
#pragma pack(pop)
```

### Predefined Macros

The compiler automatically defines several useful macros:

| Macro | Expands To | Example |
|-------|-----------|---------|
| `__FILE__` | Current filename | `"main.c"` |
| `__LINE__` | Current line number | `42` |
| `__DATE__` | Compilation date | `"Mar 17 2026"` |
| `__TIME__` | Compilation time | `"14:30:00"` |
| `__func__` | Current function name (C99) | `"main"` |
| `__STDC__` | 1 if compiler conforms to ISO C | `1` |
| `__STDC_VERSION__` | C standard version | `201112L` (C11) |

Useful for logging and debugging:

```c
#define LOG(msg) fprintf(stderr, "[%s:%d] %s: %s\n", \
    __FILE__, __LINE__, __func__, msg)

void process(void) {
    LOG("starting process");
    /* Output: [main.c:25] process: starting process */
}
```

### Stringification and Token Pasting

Two special preprocessor operators:

```c
/* # — converts a macro argument to a string literal */
#define STRINGIFY(x) #x
printf("%s\n", STRINGIFY(Hello World));  /* prints: Hello World */

/* ## — concatenates two tokens */
#define MAKE_VAR(prefix, num) prefix##num
int MAKE_VAR(value, 1) = 10;  /* becomes: int value1 = 10; */
int MAKE_VAR(value, 2) = 20;  /* becomes: int value2 = 20; */
```

---

## Exercises

**Exercise 1 — Header and Source Split**: Take any single-file C program from a previous lesson and split it into three files: a header (`.h`) with declarations and include guards, an implementation (`.c`) with function bodies, and a `main.c` that uses them. Compile with separate `gcc -c` commands and link.

**Exercise 2 — Debug Macro**: Write a `DEBUG_PRINT(fmt, ...)` macro using variadic macros that prints file, line, and a formatted message — but only when a `DEBUG` macro is defined. When `DEBUG` is not defined, the macro should expand to nothing.

**Exercise 3 — Cross-Platform Utility**: Write a header `platform.h` that defines `PLATFORM_NAME` as a string ("Windows", "macOS", or "Linux") using conditional compilation with `_WIN32`, `__APPLE__`, and `__linux__`. Write a `main.c` that includes it and prints the platform name.

**Exercise 4 — Generic MAX Macro**: Write a `GENERIC_MAX(type, a, b)` macro that declares a helper function `max_##type` using token pasting. Use it to generate `max_int`, `max_float`, and `max_double` functions. Test all three from `main`.

**Exercise 5 — Build System**: Create a project with 4 files: `main.c`, `utils.h`, `utils.c`, and `math_ops.h`/`math_ops.c`. Each header should have proper include guards. Write a sequence of `gcc` commands to compile and link them. Verify that include guards prevent errors when a header is included multiple times.

---

## Next Steps

You now understand how the preprocessor transforms your source code and how to organize multi-file C projects with headers. In the next lesson, [Build Tools and Debugging](./12_Build_Tools_and_Debugging.md), you will learn how to automate compilation with Makefiles and track down bugs with debugging tools.
