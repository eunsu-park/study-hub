# Lesson 23: Testing and Profiling in C

**Previous**: [Inter-Process Communication and Signals](./22_IPC_and_Signals.md) | **Next**: [Cross-Platform Development](./24_Cross_Platform_Development.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up and write unit tests using the Unity testing framework for C
2. Structure C code for testability by separating logic from I/O and using dependency injection
3. Validate function behavior with assertion macros covering normal, boundary, and error cases
4. Profile CPU-intensive programs with gprof to identify the hottest functions
5. Perform instruction-level profiling with Valgrind's Callgrind and visualize results
6. Analyze heap usage over time with Valgrind's Massif to find memory bottlenecks
7. Apply the test-profile-optimize cycle to improve performance without sacrificing correctness
8. Run static analysis tools (cppcheck, clang-tidy) and compiler warnings to catch bugs before runtime

---

Correctness without performance is frustrating; performance without correctness is dangerous. Professional C development requires both -- and the key is knowing which to pursue first. This lesson introduces a disciplined workflow: write tests to lock in correct behavior, then profile to find the real bottleneck, then optimize surgically, and finally re-test to make sure you did not break anything. It is a cycle that scales from hobby projects to production systems.

---

## 1. Why Testing Matters in C

C programs are particularly vulnerable to:
- **Memory errors**: buffer overflows, use-after-free, double-free
- **Undefined behavior**: signed overflow, null dereference, uninitialized reads
- **Logic errors**: off-by-one, integer truncation, wrong pointer arithmetic

Testing catches these issues before they become security vulnerabilities or production crashes.

### Testing Pyramid for C

```
        ┌──────┐
        │ E2E  │  System tests (shell scripts, expect)
       ┌┴──────┴┐
       │ Integ. │  Multi-module tests, file I/O, IPC
      ┌┴────────┴┐
      │  Unit    │  Individual function tests (Unity, CMocka)
     ┌┴──────────┴┐
     │  Static   │  Compiler warnings, clang-tidy, cppcheck
    └──────────────┘
```

---

## 2. Unit Testing with Unity

[Unity](https://github.com/ThrowTheSwitch/Unity) is a lightweight C testing framework — a single `.c` and `.h` file.

### 2.1 Setting Up Unity

```c
// test_math.c
#include "unity.h"
#include "math_utils.h"   // Module under test

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

// Test cases
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

### 2.2 Common Unity Assertions

| Assertion | Purpose |
|-----------|---------|
| `TEST_ASSERT_EQUAL_INT(expected, actual)` | Integer equality |
| `TEST_ASSERT_EQUAL_FLOAT(exp, act, delta)` | Float with tolerance |
| `TEST_ASSERT_EQUAL_STRING(exp, act)` | String comparison |
| `TEST_ASSERT_NULL(ptr)` | Pointer is NULL |
| `TEST_ASSERT_NOT_NULL(ptr)` | Pointer is not NULL |
| `TEST_ASSERT_TRUE(cond)` | Boolean condition |
| `TEST_ASSERT_EQUAL_MEMORY(exp, act, len)` | Memory comparison |
| `TEST_FAIL_MESSAGE("msg")` | Force failure |

---

## 3. Writing Testable C Code

### 3.1 Separation of Concerns

```c
// BAD: I/O mixed with logic — hard to test
int process_file(const char *filename) {
    FILE *f = fopen(filename, "r");
    int sum = 0;
    int val;
    while (fscanf(f, "%d", &val) == 1) {
        sum += val;
    }
    fclose(f);
    printf("Sum: %d\n", sum);
    return sum;
}

// GOOD: Pure logic separated from I/O
int sum_array(const int *arr, size_t len) {
    int sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

// I/O wrapper calls testable logic
int process_file(const char *filename) {
    // ... read into array ...
    int result = sum_array(values, count);
    printf("Sum: %d\n", result);
    return result;
}
```

### 3.2 Dependency Injection via Function Pointers

```c
// Inject allocator for testing
typedef void *(*allocator_fn)(size_t);
typedef void (*free_fn)(void *);

typedef struct {
    allocator_fn alloc;
    free_fn free;
} Allocator;

// Production allocator
static Allocator default_alloc = { malloc, free };

// Test allocator (tracks allocations)
static int alloc_count = 0;
static void *test_alloc(size_t size) {
    alloc_count++;
    return malloc(size);
}
static void test_free(void *ptr) {
    alloc_count--;
    free(ptr);
}
static Allocator test_allocator = { test_alloc, test_free };
```

---

## 4. Testing with assert.h (Minimal Approach)

For quick tests without frameworks:

```c
#include <assert.h>
#include <string.h>

// Simple test runner
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

---

## 5. Profiling with gprof

gprof shows which functions consume the most CPU time.

### 5.1 Workflow

```bash
# 1. Compile with profiling flags
gcc -pg -O2 -o program program.c

# 2. Run the program (generates gmon.out)
./program

# 3. Analyze the profile
gprof program gmon.out > profile.txt

# 4. Read flat profile and call graph
less profile.txt
```

### 5.2 Reading gprof Output

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
- **self seconds**: time in this function only
- **calls**: number of invocations
- **self ms/call**: average time per call (excluding children)
- **total ms/call**: average time per call (including children)

---

## 6. Profiling with Valgrind (Callgrind)

Callgrind provides instruction-level profiling without recompilation.

```bash
# Run with callgrind
valgrind --tool=callgrind ./program

# View results
callgrind_annotate callgrind.out.<pid>

# Or use KCachegrind for visualization
kcachegrind callgrind.out.<pid>
```

### 6.1 Callgrind vs gprof

| Feature | gprof | Callgrind |
|---------|-------|-----------|
| Requires recompilation | Yes (`-pg`) | No |
| Overhead | Low (~5%) | High (~20-50x) |
| Granularity | Function | Instruction |
| Cache simulation | No | Yes |
| Call graph | Basic | Detailed |

---

## 7. Memory Profiling with Valgrind (Massif)

```bash
# Heap profiler
valgrind --tool=massif ./program
ms_print massif.out.<pid>
```

Output shows heap usage over time — useful for finding memory leaks and peak usage.

---

## 8. Performance Optimization Techniques

### 8.1 Common Optimizations

After profiling identifies bottlenecks:

```c
// 1. Cache-friendly access (row-major traversal)
// BAD: column-major (cache-unfriendly)
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        sum += matrix[i][j];

// GOOD: row-major (cache-friendly)
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        sum += matrix[i][j];

// 2. Avoid repeated computation
// BAD
for (int i = 0; i < n; i++)
    result[i] = arr[i] / sqrt(sum_of_squares(arr, n));

// GOOD: compute once
double norm = sqrt(sum_of_squares(arr, n));
for (int i = 0; i < n; i++)
    result[i] = arr[i] / norm;

// 3. Strength reduction
// BAD
x = y * 4;     // multiply
// GOOD
x = y << 2;    // shift (compiler usually does this)

// 4. Branch prediction hints (GCC)
if (__builtin_expect(error_condition, 0)) {
    handle_error();  // Unlikely path
}
```

### 8.2 Compiler Optimization Levels

| Flag | Description |
|------|-------------|
| `-O0` | No optimization (default, best for debugging) |
| `-O1` | Basic optimizations |
| `-O2` | Standard optimizations (recommended for release) |
| `-O3` | Aggressive (may increase binary size) |
| `-Os` | Optimize for size |
| `-Ofast` | `-O3` + fast-math (may break IEEE compliance) |

```bash
# Compare optimization impact
gcc -O0 -o prog_debug program.c
gcc -O2 -o prog_release program.c
gcc -O3 -march=native -o prog_fast program.c

time ./prog_debug
time ./prog_release
time ./prog_fast
```

---

## 9. Static Analysis

### 9.1 Compiler Warnings

```bash
# Maximum warnings
gcc -Wall -Wextra -Wpedantic -Werror program.c

# Even more warnings
gcc -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion \
    -Wformat=2 -Wnull-dereference -Wuninitialized program.c
```

### 9.2 Static Analysis Tools

```bash
# cppcheck — static analysis
cppcheck --enable=all program.c

# clang-tidy — linting and modernization
clang-tidy program.c -- -Wall

# scan-build — Clang static analyzer
scan-build gcc -o program program.c
```

---

## 10. Putting It Together: Test-Profile-Optimize Cycle

```
┌─────────────────────────────────────────────┐
│  1. Write Tests (Unity / assert.h)          │
│     → Ensure correctness before optimizing  │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  2. Profile (gprof / callgrind / massif)    │
│     → Identify actual bottlenecks           │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  3. Optimize the Hot Path                   │
│     → Algorithm > data structure > micro    │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  4. Re-test (regression check)              │
│     → Verify correctness is preserved       │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  5. Re-profile (measure improvement)        │
│     → Quantify the speedup                  │
└─────────────────────────────────────────────┘
```

**Golden rule**: Never optimize without profiling first. The bottleneck is rarely where you think it is.

---

## Practice Exercises

### Exercise 1: Unit Testing a String Library

Write a simple string library (`mystring.h` / `mystring.c`) with:
- `my_strlen(const char *s)` — returns string length
- `my_strcpy(char *dst, const char *src)` — copies string
- `my_strrev(char *s)` — reverses string in-place

Then write a test file using `assert.h` with at least 3 test cases per function, including edge cases (empty string, single character, NULL pointer).

### Exercise 2: Profile a Sorting Algorithm

Create a program that sorts 1 million random integers using both bubble sort and quicksort. Profile with `gprof` and answer:
1. What percentage of time does `compare` take in each?
2. How many function calls does each algorithm make?
3. What is the speedup ratio of quicksort over bubble sort?

### Exercise 3: Cache-Friendly Matrix Multiplication

Implement naive and cache-friendly (tiled/blocked) matrix multiplication for 512x512 matrices. Profile both versions with `perf stat` and compare:
1. Cache miss rates (`perf stat -e cache-misses,cache-references`)
2. Instructions per cycle (IPC)
3. Wall-clock time

---

## Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| Unity | Unit testing framework | Ongoing development |
| assert.h | Quick sanity checks | Prototyping, small programs |
| gprof | CPU profiling | Release-mode performance |
| Valgrind (callgrind) | Instruction-level profiling | Detailed analysis |
| Valgrind (massif) | Heap profiling | Memory optimization |
| cppcheck | Static analysis | CI/CD pipeline |
| `-Wall -Wextra` | Compiler warnings | Every compilation |

---
