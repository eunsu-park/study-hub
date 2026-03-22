# Build Tools and Debugging

**Previous**: [Preprocessor and Headers](./11_Preprocessor_and_Headers.md) | **Next**: [Project: Calculator](./13_Project_Calculator.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write Makefiles with variables, pattern rules, automatic variables, and phony targets
2. Select appropriate compiler warning flags and optimization levels
3. Debug programs using printf tracing and systematic binary search
4. Use GDB to set breakpoints, inspect variables, and step through code
5. Organize multi-file projects with src/, include/, and build/ directories

---

Knowing how to write C code is only half the story -- you also need to compile it efficiently, catch bugs quickly, and organize files so the project stays manageable as it grows. This lesson covers the practical tooling that turns a collection of `.c` files into a reliable, debuggable program. Master these skills now and every future project will go smoother.

## 1. Compiler Flags Deep Dive

### Warning Flags

The compiler can catch many bugs before your program ever runs, but only if you ask it to:

```c
// compile.sh — recommended development flags
gcc -Wall -Wextra -Werror -std=c11 -g -O0 main.c -o main
```

| Flag | Purpose |
|------|---------|
| `-Wall` | Enable most common warnings (unused variables, implicit declarations, etc.) |
| `-Wextra` | Additional warnings beyond `-Wall` (unused parameters, sign comparison) |
| `-Werror` | Treat all warnings as errors -- forces you to fix them |
| `-std=c11` | Use the C11 standard (modern, portable) |
| `-g` | Include debug symbols (required for GDB) |
| `-O0` | No optimization -- easiest to debug |

### Optimization Levels

| Level | Effect | When to Use |
|-------|--------|-------------|
| `-O0` | No optimization | Development and debugging |
| `-O1` | Basic optimization | Moderate speedup, still debuggable |
| `-O2` | Standard optimization | Release builds |
| `-O3` | Aggressive optimization | Performance-critical code |
| `-Os` | Optimize for size | Embedded systems |

### Sanitizers

Address Sanitizer catches memory errors at runtime:

```bash
gcc -Wall -Wextra -std=c11 -g -fsanitize=address -fsanitize=undefined main.c -o main
```

| Sanitizer | Catches |
|-----------|---------|
| `-fsanitize=address` | Buffer overflow, use-after-free, memory leaks |
| `-fsanitize=undefined` | Signed overflow, null pointer dereference, alignment |

```c
// This bug is silent without sanitizers but caught with -fsanitize=address
#include <stdio.h>

int main(void) {
    int arr[5] = {1, 2, 3, 4, 5};
    printf("%d\n", arr[10]);  // Out-of-bounds read -- ASan catches this
    return 0;
}
```

---

## 2. Makefile Essentials

A Makefile automates compilation so you do not have to retype long `gcc` commands. The basic structure is **target: prerequisites**, followed by a tab-indented recipe:

```makefile
# Variables
CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -std=c11 -g
LDFLAGS =

# Default target
all: calculator

# Link step
calculator: main.o calc.o utils.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compile step
main.o: main.c calc.h utils.h
	$(CC) $(CFLAGS) -c main.c

calc.o: calc.c calc.h
	$(CC) $(CFLAGS) -c calc.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

# Cleanup
clean:
	rm -f *.o calculator

.PHONY: all clean
```

### Key Concepts

| Term | Meaning |
|------|---------|
| **Target** | The file to build (left of `:`) |
| **Prerequisites** | Files the target depends on (right of `:`) |
| **Recipe** | Shell commands to produce the target (must be tab-indented) |
| **Variable** | `CC = gcc` defines a variable; `$(CC)` expands it |
| `.PHONY` | Declares targets that are not real files |

### Common Variables

| Variable | Convention |
|----------|-----------|
| `CC` | C compiler (`gcc`, `clang`) |
| `CFLAGS` | Compiler flags (`-Wall -g`) |
| `LDFLAGS` | Linker flags (`-lm`, `-lpthread`) |
| `CPPFLAGS` | Preprocessor flags (`-I./include`, `-DDEBUG`) |

---

## 3. Automatic Variables

Automatic variables eliminate repetition in rules:

| Variable | Expands To |
|----------|------------|
| `$@` | The target filename |
| `$<` | The first prerequisite |
| `$^` | All prerequisites (space-separated) |
| `$*` | The stem matched by `%` in a pattern rule |

### Pattern Rules

Instead of writing a separate rule for each `.o` file, use a pattern rule:

```makefile
# Pattern rule: any .o depends on matching .c
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# This single rule replaces:
#   main.o: main.c       ->  gcc ... -c main.c -o main.o
#   calc.o: calc.c       ->  gcc ... -c calc.c -o calc.o
#   utils.o: utils.c     ->  gcc ... -c utils.c -o utils.o
```

Full example using automatic variables:

```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g

SRCS    = main.c calc.c utils.c
OBJS    = $(SRCS:.c=.o)
TARGET  = calculator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

## 4. Advanced Makefile Features

### Automatic Dependency Generation

When a header changes, all files that include it must recompile. Let the compiler generate dependency files:

```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g -MMD -MP

SRCS    = main.c calc.c utils.c
OBJS    = $(SRCS:.c=.o)
DEPS    = $(OBJS:.o=.d)
TARGET  = calculator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean
```

| Flag | Purpose |
|------|---------|
| `-MMD` | Generate `.d` dependency files alongside `.o` files |
| `-MP` | Add phony targets for each header (avoids errors when headers are deleted) |
| `-include` | Include `.d` files if they exist (the `-` silences errors on first build) |

### Multiple Targets

```makefile
TESTS = test_calc test_utils

all: calculator $(TESTS)

calculator: main.o calc.o utils.o
	$(CC) $(CFLAGS) -o $@ $^

test_calc: test_calc.o calc.o
	$(CC) $(CFLAGS) -o $@ $^

test_utils: test_utils.o utils.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: test
test: $(TESTS)
	./test_calc
	./test_utils
```

---

## 5. printf Debugging

The simplest debugging technique is printing values at strategic points:

```c
#include <stdio.h>

// DEBUG macro: prints only when compiled with -DDEBUG
#ifdef DEBUG
  #define DBG(fmt, ...) fprintf(stderr, "[DBG %s:%d] " fmt "\n", \
                                __FILE__, __LINE__, ##__VA_ARGS__)
#else
  #define DBG(fmt, ...)  // Expands to nothing
#endif

int binary_search(int arr[], int n, int target) {
    int lo = 0, hi = n - 1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        DBG("lo=%d mid=%d hi=%d arr[mid]=%d", lo, mid, hi, arr[mid]);

        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else                   hi = mid - 1;
    }

    DBG("target %d not found", target);
    return -1;
}
```

```bash
# Debug build: prints DBG messages
gcc -Wall -std=c11 -DDEBUG search.c -o search_dbg

# Release build: DBG messages compiled out
gcc -Wall -std=c11 -O2 search.c -o search
```

### Systematic Binary Search Debugging

When you have a bug but no idea where it is:

1. Add a print at the midpoint of the suspicious code
2. Determine which half contains the bug
3. Add a print at the midpoint of that half
4. Repeat until the bug is localized

This is O(log n) in the number of code lines -- far faster than reading every line.

---

## 6. GDB Basics

GDB (GNU Debugger) lets you pause execution, inspect variables, and step through code line by line.

### Starting GDB

```bash
# Compile with debug symbols
gcc -Wall -std=c11 -g -O0 program.c -o program

# Start GDB
gdb ./program

# Or start with arguments
gdb --args ./program arg1 arg2
```

### Essential Commands

| Command | Short | Action |
|---------|-------|--------|
| `run` | `r` | Start the program |
| `break main` | `b main` | Set breakpoint at function `main` |
| `break file.c:42` | `b file.c:42` | Set breakpoint at line 42 |
| `next` | `n` | Execute next line (step over function calls) |
| `step` | `s` | Execute next line (step into function calls) |
| `finish` | `fin` | Run until current function returns |
| `continue` | `c` | Resume execution until next breakpoint |
| `print x` | `p x` | Print variable value |
| `print *ptr` | `p *ptr` | Dereference and print pointer |
| `print arr[0]@10` | | Print 10 elements of array |
| `watch x` | | Break when variable `x` changes |
| `backtrace` | `bt` | Show call stack |
| `info locals` | | Show all local variables |
| `quit` | `q` | Exit GDB |

### Example GDB Session

```
$ gdb ./calculator
(gdb) break main
Breakpoint 1 at 0x4011a0: file main.c, line 15.
(gdb) run
Starting program: ./calculator

Breakpoint 1, main () at main.c:15
15      double num1, num2, result;
(gdb) next
16      char operator;
(gdb) break calculate
Breakpoint 2 at 0x401250: file calc.c, line 8.
(gdb) continue
Continuing.
Enter expression: 10 / 0

Breakpoint 2, calculate (num1=10, op='/', num2=0, result=0x7ffd...) at calc.c:8
8       switch (op) {
(gdb) print num2
$1 = 0
(gdb) next
18          if (num2 == 0) {
(gdb) quit
```

### Watchpoints

Watchpoints pause execution whenever a variable changes -- useful for tracking down corruption:

```
(gdb) watch contact_count
Hardware watchpoint 1: contact_count
(gdb) run
...
Hardware watchpoint 1: contact_count
Old value = 3
New value = 4
add_contact (ab=0x7ffd...) at addressbook.c:95
```

---

## 7. Common Bug Patterns

### Segmentation Fault

A segfault means you accessed memory you should not have:

```c
// Null pointer dereference
int *p = NULL;
*p = 42;            // SEGFAULT

// Array out of bounds
int arr[5];
arr[100] = 42;      // SEGFAULT (or silent corruption)

// Use after free
int *p = malloc(sizeof(int));
free(p);
*p = 42;            // SEGFAULT (use-after-free)
```

**Diagnosis**: Run with `-fsanitize=address` or use GDB `backtrace` to find the exact line.

### Off-by-One Errors

```c
// Bug: writes past end of array
int arr[10];
for (int i = 0; i <= 10; i++) {   // Should be i < 10
    arr[i] = i;
}

// Bug: string missing null terminator
char buf[5];
strncpy(buf, "Hello", 5);    // No room for '\0'!
// Fix: strncpy(buf, "Hello", sizeof(buf) - 1); buf[sizeof(buf)-1] = '\0';
```

### Uninitialized Variables

```c
int sum;               // Not initialized -- contains garbage!
for (int i = 0; i < 10; i++) {
    sum += i;          // Undefined behavior
}
// Fix: int sum = 0;
```

### Buffer Overflow

```c
char name[10];
scanf("%s", name);     // User types "Alexander" (9 chars + '\0' = 10, just fits)
                       // User types "Christopher" -> OVERFLOW

// Fix: limit input length
scanf("%9s", name);    // Read at most 9 chars, leave room for '\0'

// Better: use fgets
fgets(name, sizeof(name), stdin);
name[strcspn(name, "\n")] = '\0';
```

---

## 8. Project Organization

As projects grow beyond a handful of files, a consistent directory layout prevents chaos:

```
my_project/
├── Makefile
├── include/           # Header files (.h)
│   ├── calc.h
│   └── utils.h
├── src/               # Source files (.c)
│   ├── main.c
│   ├── calc.c
│   └── utils.c
├── build/             # Object files and dependencies (generated)
│   ├── main.o
│   ├── calc.o
│   └── utils.o
└── tests/             # Test files
    ├── test_calc.c
    └── test_utils.c
```

### Makefile for This Layout

```makefile
CC       = gcc
CFLAGS   = -Wall -Wextra -std=c11 -g -MMD -MP
CPPFLAGS = -Iinclude

SRC_DIR  = src
BUILD_DIR = build
INC_DIR  = include

SRCS     = $(wildcard $(SRC_DIR)/*.c)
OBJS     = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS     = $(OBJS:.o=.d)
TARGET   = calculator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
```

### Header Guard Convention

```c
// include/calc.h
#ifndef CALC_H
#define CALC_H

int calculate(double num1, char op, double num2, double *result);

#endif  // CALC_H
```

### Compilation Commands

```bash
# Build everything
make

# Build with debug output
make CFLAGS="-Wall -Wextra -std=c11 -g -O0 -DDEBUG"

# Clean and rebuild
make clean && make

# Build only if changed
make    # make tracks timestamps automatically
```

---

## Exercises

1. **Makefile from scratch**: You have three files: `main.c`, `math_ops.c`, and `math_ops.h`. Write a complete Makefile with variables (`CC`, `CFLAGS`), a pattern rule for `.o` files, automatic dependency generation (`-MMD -MP`), and `clean`/`all` phony targets.

2. **Sanitizer detective**: The following program has a hidden bug. Compile it with `-fsanitize=address` and `-fsanitize=undefined`, run it, read the sanitizer output, and fix the bug:
   ```c
   #include <stdio.h>
   #include <string.h>
   int main(void) {
       char buf[8];
       strcpy(buf, "overflow!");
       printf("%s\n", buf);
       return 0;
   }
   ```

3. **DEBUG macro**: Write a `debug.h` header that defines a `DBG(fmt, ...)` macro printing the file name, line number, and a formatted message to stderr when `DEBUG` is defined, and expanding to nothing otherwise. Use it in a small program and verify both `-DDEBUG` and release builds work correctly.

4. **GDB practice**: Write a program that computes the factorial of a user-supplied number using a loop. Compile with `-g -O0`, start GDB, set a breakpoint inside the loop, and use `next` and `print` to watch the accumulator grow each iteration. Record the GDB commands you used.

5. **Project layout**: Take any previous exercise (e.g., the calculator) and reorganize it into the `src/`, `include/`, `build/`, `tests/` directory structure. Write a Makefile that compiles sources from `src/`, places object files in `build/`, and includes headers from `include/`. Verify that `make clean && make` produces a working binary.

---

## Next Steps

[Project: Calculator](./13_Project_Calculator.md) -- Put your build-tool knowledge into practice by building an interactive calculator, step by step.
