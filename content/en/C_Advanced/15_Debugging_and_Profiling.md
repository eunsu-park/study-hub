# Debugging and Profiling

**Previous**: [Embedded Systems](./14_Embedded_Systems.md) | **Next**: [Cross-Platform Development](./16_Cross_Platform_Development.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use advanced GDB features: conditional breakpoints, watchpoints, reverse debugging, core dumps
2. Detect memory errors using Valgrind Memcheck and interpret its output
3. Apply compile-time sanitizers (ASan, UBSan, TSan) to catch runtime bugs
4. Profile program performance using gprof and Valgrind's Callgrind
5. Write unit tests using Unity framework and assert-based testing
6. Perform static analysis using cppcheck and clang-tidy

---

Bugs are inevitable, but the speed at which you find and fix them separates productive programmers from frustrated ones. A segmentation fault at 2 AM is far less terrifying when you know how to fire up GDB, set a breakpoint, and inspect the call stack. Correctness without performance is frustrating; performance without correctness is dangerous. This lesson equips you with a professional toolkit for both -- from interactive debugging and automated memory analysis to CPU profiling and unit testing.

**Difficulty**: Advanced

**Prerequisites**: Pointers, dynamic memory allocation

---

## 1. Advanced GDB

### Starting GDB

```bash
# Compile with debug symbols
gcc -g -O0 -Wall -Wextra program.c -o program

# Launch GDB
gdb ./program

# With arguments
gdb --args ./program arg1 arg2

# Attach to running process
gdb -p <pid>

# Analyze core dump
gdb ./program core
```

### Conditional Breakpoints

```bash
# Break only when condition is true
(gdb) break main.c:42 if i == 100
(gdb) break process_item if ptr == NULL
(gdb) break sort.c:15 if n > 1000

# Add condition to existing breakpoint
(gdb) condition 3 x > 0

# Ignore first N hits
(gdb) ignore 2 50  # Skip breakpoint 2 for 50 hits
```

### Watchpoints

Stop execution when a variable changes:

```bash
# Break when variable is written
(gdb) watch counter
(gdb) watch arr[5]
(gdb) watch *ptr

# Break when variable is read
(gdb) rwatch x

# Break on read or write
(gdb) awatch x

# List watchpoints
(gdb) info watchpoints
```

### Core Dump Analysis

```bash
# Enable core dumps
$ ulimit -c unlimited

# Run program (crashes and generates core file)
$ ./buggy_program
Segmentation fault (core dumped)

# Analyze the core
$ gdb ./buggy_program core
(gdb) bt           # Backtrace shows where it crashed
(gdb) frame 0      # Select the crash frame
(gdb) info locals  # See local variables
(gdb) print *ptr   # Inspect the offending pointer
```

### TUI Mode

Debug while viewing source code:

```bash
# Start TUI
(gdb) tui enable
# Or at startup
$ gdb -tui ./program

# Change layout
(gdb) layout src    # Source code
(gdb) layout asm    # Assembly
(gdb) layout split  # Source + Assembly
(gdb) layout regs   # Registers

# Exit TUI
(gdb) tui disable
```

### GDB Practical Example

```c
// buggy.c
#include <stdio.h>
#include <stdlib.h>

int sum_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // Bug: <= should be <
        sum += arr[i];
    }
    return sum;
}

int main(void) {
    int *numbers = malloc(5 * sizeof(int));
    for (int i = 0; i < 5; i++) {
        numbers[i] = i + 1;
    }
    int total = sum_array(numbers, 5);
    printf("Total: %d\n", total);
    free(numbers);
    return 0;
}
```

```bash
$ gcc -g -O0 buggy.c -o buggy
$ gdb ./buggy
(gdb) break sum_array
(gdb) run
(gdb) print size
$1 = 5
(gdb) watch sum
(gdb) continue
# Watchpoint fires each time sum changes
# On the 6th iteration (i=5), we read past the array
```

---

## 2. Valgrind Memcheck

### Basic Usage

```bash
# Run memory check
valgrind ./program

# Detailed leak report
valgrind --leak-check=full ./program

# Track origin of uninitialized values
valgrind --leak-check=full --track-origins=yes ./program

# Save to log file
valgrind --log-file=valgrind.log ./program
```

### Detecting Memory Leaks

```c
// leak.c
#include <stdlib.h>
#include <string.h>

void create_leak(void) {
    int *ptr = malloc(100 * sizeof(int));
    ptr[0] = 42;
    // free(ptr); missing!
}

char *duplicate_string(const char *str) {
    char *copy = malloc(strlen(str) + 1);
    strcpy(copy, str);
    return copy;  // Caller must free
}

int main(void) {
    create_leak();
    char *str = duplicate_string("Hello");
    // free(str); missing!
    return 0;
}
```

```bash
$ valgrind --leak-check=full ./leak

==12345== HEAP SUMMARY:
==12345==     in use at exit: 406 bytes in 2 blocks
==12345==   total heap usage: 2 allocs, 0 frees, 406 bytes allocated
==12345==
==12345== 6 bytes in 1 blocks are definitely lost
==12345==    at 0x4C2FB0F: malloc
==12345==    by 0x10871B: duplicate_string (leak.c:11)
==12345==    by 0x108751: main (leak.c:18)
==12345==
==12345== 400 bytes in 1 blocks are definitely lost
==12345==    at 0x4C2FB0F: malloc
==12345==    by 0x1086E2: create_leak (leak.c:5)
==12345==    by 0x108745: main (leak.c:16)
```

### Leak Types

| Type | Description |
|------|-------------|
| definitely lost | Pointer to block is completely lost |
| indirectly lost | Block reachable only through a lost block |
| possibly lost | Pointer points to middle of block |
| still reachable | Block still accessible at program exit |

### Invalid Memory Access

Valgrind catches out-of-bounds access, use-after-free, double-free, and uninitialized reads:

```bash
$ valgrind --track-origins=yes ./invalid

==12345== Invalid write of size 4
==12345==    at 0x1086A1: main (invalid.c:11)
==12345==  Address 0x522d054 is 0 bytes after a block of size 20 alloc'd
```

---

## 3. Address Sanitizer (ASan)

### Using ASan

```bash
# Compile with ASan
gcc -fsanitize=address -g -fno-omit-frame-pointer program.c -o program

# Run normally -- ASan reports errors at runtime
./program
```

### ASan vs Valgrind

| Feature | Valgrind | ASan |
|---------|----------|------|
| Speed | 10-50x slower | 2x slower |
| Memory | 2x usage | 3x usage |
| Stack overflow | No | Yes |
| Global overflow | No | Yes |
| Recompilation | Not needed | Required |

### ASan Example

```c
// asan_test.c
#include <stdlib.h>

int main(void) {
    int *arr = malloc(10 * sizeof(int));
    arr[10] = 42;  // Heap buffer overflow
    free(arr);
    arr[0] = 100;  // Use after free
    return 0;
}
```

```bash
$ gcc -fsanitize=address -g asan_test.c -o asan_test
$ ./asan_test

ERROR: AddressSanitizer: heap-buffer-overflow on address 0x604000000028
WRITE of size 4 at 0x604000000028 thread T0
    #0 0x4011a3 in main asan_test.c:5
```

---

## 4. UBSan and TSan

### Undefined Behavior Sanitizer

```bash
gcc -fsanitize=undefined -g program.c -o program
```

Catches: signed integer overflow, null dereference, shift by negative amount, division by zero, out-of-bounds array access.

```c
// ubsan_test.c
#include <limits.h>

int main(void) {
    int x = INT_MAX;
    int y = x + 1;  // Signed overflow (undefined behavior!)
    return y;
}
```

```bash
$ gcc -fsanitize=undefined -g ubsan_test.c -o ubsan_test
$ ./ubsan_test
ubsan_test.c:5:15: runtime error: signed integer overflow:
2147483647 + 1 cannot be represented in type 'int'
```

### Thread Sanitizer

```bash
gcc -fsanitize=thread -g program.c -o program -pthread
```

Detects data races between threads. Cannot be combined with ASan.

---

## 5. Profiling with gprof

### Workflow

```bash
# 1. Compile with profiling flags
gcc -pg -O2 -o program program.c

# 2. Run (generates gmon.out)
./program

# 3. Analyze
gprof program gmon.out > profile.txt
less profile.txt
```

### Reading gprof Output

```
Flat profile:

  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
 45.2     0.85     0.85     1000     0.85     1.20  sort_array
 30.1     1.42     0.57  1000000     0.00     0.00  compare
 15.0     1.70     0.28     1000     0.28     0.28  copy_array
  9.7     1.89     0.18        1   180.00  1890.00  main
```

Key columns:
- **% time**: fraction of total execution
- **self seconds**: time in this function only (excluding callees)
- **calls**: number of invocations
- **self ms/call**: average time per call (excluding children)
- **total ms/call**: average time per call (including children)

---

## 6. Valgrind Callgrind

Instruction-level profiling without recompilation:

```bash
# Run with callgrind
valgrind --tool=callgrind ./program

# View results as text
callgrind_annotate callgrind.out.<pid>

# Visualize with KCachegrind (GUI)
kcachegrind callgrind.out.<pid>
```

### Callgrind vs gprof

| Feature | gprof | Callgrind |
|---------|-------|-----------|
| Requires recompilation | Yes (`-pg`) | No |
| Overhead | Low (~5%) | High (~20-50x) |
| Granularity | Function | Instruction |
| Cache simulation | No | Yes |
| Call graph | Basic | Detailed |

---

## 7. Unit Testing

### assert.h -- Minimal Approach

```c
#include <assert.h>
#include <string.h>
#include <stdio.h>

// Simple test runner macro
#define RUN(test) do { \
    printf("  %-40s", #test); \
    test(); \
    printf("PASS\n"); \
} while(0)

void test_strlen_basic(void) {
    assert(strlen("hello") == 5);
    assert(strlen("") == 0);
}

void test_strcmp_equal(void) {
    assert(strcmp("abc", "abc") == 0);
}

void test_strcmp_less(void) {
    assert(strcmp("abc", "abd") < 0);
}

int main(void) {
    printf("Running tests:\n");
    RUN(test_strlen_basic);
    RUN(test_strcmp_equal);
    RUN(test_strcmp_less);
    printf("All tests passed!\n");
    return 0;
}
```

### Unity Framework

[Unity](https://github.com/ThrowTheSwitch/Unity) is a lightweight C testing framework (single `.c` and `.h` file):

```c
// test_math.c
#include "unity.h"
#include "math_utils.h"

void setUp(void) { }
void tearDown(void) { }

void test_add(void) {
    TEST_ASSERT_EQUAL_INT(5, add(2, 3));
    TEST_ASSERT_EQUAL_INT(0, add(-1, 1));
    TEST_ASSERT_EQUAL_INT(-3, add(-1, -2));
}

void test_divide(void) {
    TEST_ASSERT_EQUAL_FLOAT(2.5f, divide(5.0f, 2.0f), 0.001f);
}

void test_divide_by_zero(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, divide(5.0f, 0.0f), 0.001f);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_add);
    RUN_TEST(test_divide);
    RUN_TEST(test_divide_by_zero);
    return UNITY_END();
}
```

### Common Unity Assertions

| Assertion | Purpose |
|-----------|---------|
| `TEST_ASSERT_EQUAL_INT(exp, act)` | Integer equality |
| `TEST_ASSERT_EQUAL_FLOAT(exp, act, delta)` | Float with tolerance |
| `TEST_ASSERT_EQUAL_STRING(exp, act)` | String comparison |
| `TEST_ASSERT_NULL(ptr)` | Pointer is NULL |
| `TEST_ASSERT_NOT_NULL(ptr)` | Pointer is not NULL |
| `TEST_ASSERT_TRUE(cond)` | Boolean condition |
| `TEST_ASSERT_EQUAL_MEMORY(exp, act, len)` | Memory comparison |

### Writing Testable C Code

```c
// BAD: I/O mixed with logic
int process_file(const char *filename) {
    FILE *f = fopen(filename, "r");
    int sum = 0, val;
    while (fscanf(f, "%d", &val) == 1) sum += val;
    fclose(f);
    return sum;
}

// GOOD: Pure logic separated from I/O
int sum_array(const int *arr, size_t len) {
    int sum = 0;
    for (size_t i = 0; i < len; i++) sum += arr[i];
    return sum;
}
```

---

## 8. Static Analysis

### Compiler Warnings

```bash
# Maximum warnings
gcc -Wall -Wextra -Wpedantic -Werror program.c

# Even more
gcc -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion \
    -Wformat=2 -Wnull-dereference -Wuninitialized program.c
```

### cppcheck

```bash
# Static analysis
cppcheck --enable=all program.c

# With suppression
cppcheck --enable=all --suppress=missingInclude .
```

### clang-tidy

```bash
# Linting
clang-tidy program.c -- -Wall

# Fix automatically
clang-tidy --fix program.c -- -Wall
```

### scan-build (Clang Static Analyzer)

```bash
scan-build gcc -o program program.c
```

---

## The Test-Profile-Optimize Cycle

1. **Write Tests** -- Ensure correctness before optimizing
2. **Profile** -- Identify actual bottlenecks (gprof / callgrind)
3. **Optimize the Hot Path** -- Algorithm > data structure > micro-optimization
4. **Re-test** -- Verify correctness is preserved
5. **Re-profile** -- Quantify the speedup

**Golden rule**: Never optimize without profiling first. The bottleneck is rarely where you think it is.

---

## Tool Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| GDB | Interactive debugging | Crash investigation, logic errors |
| Valgrind Memcheck | Memory error detection | Memory leaks, invalid access |
| ASan | Fast memory error detection | Development builds |
| UBSan | Undefined behavior detection | Development builds |
| TSan | Data race detection | Multithreaded programs |
| gprof | CPU profiling | Release-mode performance |
| Callgrind | Instruction-level profiling | Detailed analysis |
| Unity | Unit testing | Ongoing development |
| cppcheck | Static analysis | CI/CD pipeline |
| `-Wall -Wextra` | Compiler warnings | Every compilation |

---

## Exercises

### Exercise 1: Find Memory Leaks

Find and fix all memory leaks in the following code using Valgrind:

```c
typedef struct {
    char *name;
    int *scores;
    int num_scores;
} Student;

Student *create_student(const char *name, int num_scores) {
    Student *s = malloc(sizeof(Student));
    s->name = malloc(strlen(name) + 1);
    strcpy(s->name, name);
    s->scores = malloc(num_scores * sizeof(int));
    s->num_scores = num_scores;
    return s;
}

void process_students(void) {
    Student *students[3];
    students[0] = create_student("Alice", 5);
    students[1] = create_student("Bob", 3);
    students[2] = create_student("Charlie", 4);
    // No cleanup!
}
```

### Exercise 2: Profile a Sorting Algorithm

Create a program that sorts 1 million random integers using both bubble sort and quicksort. Profile with `gprof` and answer:
1. What percentage of time does `compare` take in each?
2. How many function calls does each algorithm make?
3. What is the speedup ratio of quicksort over bubble sort?

### Exercise 3: Unit Test a String Library

Write `my_strlen`, `my_strcpy`, and `my_strrev` functions. Create tests using `assert.h` with at least 3 test cases per function, including edge cases (empty string, single character, NULL pointer).

### Exercise 4: Sanitizer Bug Hunt

Compile the following code with `-fsanitize=address,undefined` and fix all issues reported:

```c
int main(void) {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i <= 5; i++) sum += arr[i];

    int x = 2147483647;
    x = x + 1;

    int *p = malloc(10 * sizeof(int));
    free(p);
    p[0] = 42;

    return sum + x;
}
```

### Exercise 5: Cache-Friendly Matrix Multiplication

Implement naive and cache-friendly (tiled/blocked) matrix multiplication for 512x512 matrices. Profile both versions and compare cache miss rates, IPC, and wall-clock time.

---

## Next Steps

With debugging and profiling mastered, proceed to:
- [Cross-Platform Development](./16_Cross_Platform_Development.md) -- Writing portable C that compiles on Linux, macOS, and Windows
