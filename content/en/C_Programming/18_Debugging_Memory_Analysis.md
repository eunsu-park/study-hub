# Debugging and Memory Analysis

**Previous**: [Project 16: Serial Communication](./17_Project_Serial_Communication.md) | **Next**: [Advanced Embedded Protocols](./19_Advanced_Embedded_Protocols.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Navigate GDB to set breakpoints, inspect variables, and step through code
2. Detect memory leaks and invalid accesses using Valgrind's Memcheck tool
3. Interpret Valgrind output to distinguish definitely-lost, indirectly-lost, and still-reachable blocks
4. Compile with AddressSanitizer to catch buffer overflows, use-after-free, and double-free at runtime
5. Apply defensive programming techniques such as `assert`, logging levels, and NULL-after-free
6. Diagnose common memory bugs including uninitialized reads, stack overflow, and dangling pointers
7. Choose the right debugging tool for a given situation (GDB vs. Valgrind vs. ASan)

---

Bugs are inevitable, but the speed at which you find and fix them separates productive programmers from frustrated ones. A segmentation fault at 2 AM is far less terrifying when you know how to fire up GDB, set a breakpoint, and inspect the call stack. This lesson equips you with a professional debugging toolkit -- from interactive stepping to automated memory analysis -- so that you spend less time guessing and more time shipping correct code.

**Difficulty**: Advanced

**Prerequisites**: Pointers, dynamic memory allocation

---

## Table of Contents

1. [Debugging Basics](#debugging-basics)
2. [GDB Debugger](#gdb-debugger)
3. [Valgrind Memory Analysis](#valgrind-memory-analysis)
4. [AddressSanitizer](#addresssanitizer)
5. [Common Memory Bugs](#common-memory-bugs)
6. [Debugging Strategies](#debugging-strategies)

---

## Debugging Basics

### Debug Build

To debug, you must compile with the `-g` flag.

```bash
# Include debug symbols
gcc -g -Wall -Wextra program.c -o program

# Without optimization (easier debugging)
gcc -g -O0 -Wall -Wextra program.c -o program

# Debug + optimization (tracking release bugs)
gcc -g -O2 -Wall -Wextra program.c -o program
```

### printf Debugging

The most basic debugging method.

```c
#include <stdio.h>

#define DEBUG 1

#if DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

int calculate(int a, int b) {
    DEBUG_PRINT("calculate called: a=%d, b=%d", a, b);
    int result = a * b + a;
    DEBUG_PRINT("result=%d", result);
    return result;
}

int main(void) {
    int x = calculate(5, 3);
    DEBUG_PRINT("main: x=%d", x);
    return 0;
}
```

### assert Macro

Used for precondition verification.

```c
#include <assert.h>
#include <stdlib.h>

int divide(int a, int b) {
    assert(b != 0 && "Division by zero!");
    return a / b;
}

void process_array(int *arr, size_t size) {
    assert(arr != NULL && "Array is NULL!");
    assert(size > 0 && "Size must be positive!");

    for (size_t i = 0; i < size; i++) {
        arr[i] *= 2;
    }
}
```

> **Note**: To disable assert in release builds, use the `-DNDEBUG` flag

---

## GDB Debugger

### Starting GDB

```bash
# Load program
gdb ./program

# With arguments
gdb --args ./program arg1 arg2

# Attach to running process
gdb -p <pid>

# Analyze core dump
gdb ./program core
```

### Basic Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `run` | `r` | Run program |
| `continue` | `c` | Continue execution |
| `next` | `n` | Next line (don't enter functions) |
| `step` | `s` | Next line (enter functions) |
| `finish` | `fin` | Run to end of current function |
| `quit` | `q` | Exit GDB |

### Breakpoints

```bash
# Set at line number
(gdb) break main.c:15
(gdb) b 15

# Set at function
(gdb) break main
(gdb) break calculate

# Conditional breakpoint
(gdb) break 20 if i == 5
(gdb) break process if ptr == NULL

# List breakpoints
(gdb) info breakpoints
(gdb) info b

# Delete breakpoint
(gdb) delete 1        # Delete #1
(gdb) delete          # Delete all

# Disable/enable breakpoint
(gdb) disable 2
(gdb) enable 2
```

### Inspecting Variables

```bash
# Print variable
(gdb) print x
(gdb) p x
(gdb) p arr[0]
(gdb) p *ptr
(gdb) p ptr->field

# Expressions
(gdb) p x + y
(gdb) p sizeof(struct data)

# Print array
(gdb) p *arr@10        # 10 elements

# Format specifiers
(gdb) p/x value        # Hexadecimal
(gdb) p/t value        # Binary
(gdb) p/d value        # Decimal
(gdb) p/c value        # Character

# Print all local variables
(gdb) info locals

# Global variables
(gdb) info variables
```

### Memory Inspection

```bash
# Examine memory contents
(gdb) x/10xb ptr       # 10 bytes, hex
(gdb) x/10dw ptr       # 10 words, decimal
(gdb) x/s str          # String
(gdb) x/10i func       # 10 instructions (disassemble)

# Format: x/[count][format][size]
# Format: x(hex), d(decimal), s(string), i(instruction)
# Size: b(byte), h(2 bytes), w(4 bytes), g(8 bytes)
```

### Stack Trace

```bash
# Backtrace (call stack)
(gdb) backtrace
(gdb) bt

# Select specific frame
(gdb) frame 2
(gdb) f 2

# Frame info
(gdb) info frame

# Move up/down frames
(gdb) up
(gdb) down
```

### Watchpoints

Stops when a variable is modified.

```bash
# Write watchpoint
(gdb) watch x
(gdb) watch arr[5]
(gdb) watch *ptr

# Read watchpoint
(gdb) rwatch x

# Read/write watchpoint
(gdb) awatch x

# List watchpoints
(gdb) info watchpoints
```

### GDB Practical Example

```c
// buggy.c - Code with a bug
#include <stdio.h>
#include <stdlib.h>

int sum_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // Bug: should use < instead of <=
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
(gdb) print arr[0]
$2 = 1
(gdb) next
(gdb) print i
$3 = 0
(gdb) watch sum
(gdb) continue
# Stops each time sum changes
```

### GDB TUI Mode

Debug while viewing source code with text user interface.

```bash
# Start TUI mode
(gdb) tui enable
# Or at startup
$ gdb -tui ./program

# Change layout
(gdb) layout src       # Source code
(gdb) layout asm       # Assembly
(gdb) layout split     # Source + Assembly
(gdb) layout regs      # Registers

# Exit TUI mode
(gdb) tui disable
```

---

## Valgrind Memory Analysis

### Installing Valgrind

```bash
# Ubuntu/Debian
sudo apt install valgrind

# macOS (Intel only)
brew install valgrind

# CentOS/RHEL
sudo yum install valgrind
```

### Basic Usage

```bash
# Run memory check
valgrind ./program

# Detailed output
valgrind --leak-check=full ./program

# More detailed info
valgrind --leak-check=full --show-leak-kinds=all ./program

# Save to log file
valgrind --log-file=valgrind.log ./program
```

### Memcheck Tool

The most commonly used Valgrind tool.

```bash
valgrind --tool=memcheck --leak-check=full \
         --track-origins=yes ./program
```

| Option | Description |
|--------|-------------|
| `--leak-check=full` | Detailed leak information |
| `--show-leak-kinds=all` | Show all types of leaks |
| `--track-origins=yes` | Track origin of uninitialized values |
| `--verbose` | Verbose output |

### Memory Leak Example

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
$ gcc -g leak.c -o leak
$ valgrind --leak-check=full ./leak

==12345== HEAP SUMMARY:
==12345==     in use at exit: 406 bytes in 2 blocks
==12345==   total heap usage: 2 allocs, 0 frees, 406 bytes allocated
==12345==
==12345== 6 bytes in 1 blocks are definitely lost in loss record 1 of 2
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x10871B: duplicate_string (leak.c:11)
==12345==    by 0x108751: main (leak.c:18)
==12345==
==12345== 400 bytes in 1 blocks are definitely lost in loss record 2 of 2
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x1086E2: create_leak (leak.c:5)
==12345==    by 0x108745: main (leak.c:16)
```

### Leak Types

| Type | Description |
|------|-------------|
| definitely lost | Definite leak (pointer lost) |
| indirectly lost | Indirect leak (was accessible through another block) |
| possibly lost | Possible leak (pointer points to middle of block) |
| still reachable | Still accessible at program exit |

### Invalid Memory Access

```c
// invalid.c
#include <stdlib.h>

int main(void) {
    int *arr = malloc(5 * sizeof(int));

    // Reading uninitialized value
    int x = arr[0];

    // Out of bounds write
    arr[5] = 100;

    // Out of bounds read
    int y = arr[10];

    free(arr);

    // Use After Free
    arr[0] = 42;

    // Double free
    free(arr);

    return x + y;
}
```

```bash
$ valgrind --track-origins=yes ./invalid

==12345== Conditional jump or move depends on uninitialised value(s)
==12345==    at 0x108691: main (invalid.c:8)
==12345==  Uninitialised value was created by a heap allocation
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x108671: main (invalid.c:5)
==12345==
==12345== Invalid write of size 4
==12345==    at 0x1086A1: main (invalid.c:11)
==12345==  Address 0x522d054 is 0 bytes after a block of size 20 alloc'd
```

---

## AddressSanitizer

### Using ASan

Add flags when compiling.

```bash
# GCC
gcc -fsanitize=address -g program.c -o program

# Clang
clang -fsanitize=address -g program.c -o program

# Additional options
gcc -fsanitize=address -fno-omit-frame-pointer -g program.c -o program
```

### ASan Advantages

| Feature | Valgrind | ASan |
|---------|----------|------|
| Speed | 10-50x slower | 2x slower |
| Memory usage | 2x | 3x |
| Stack overflow | X | O |
| Global variable overflow | X | O |
| Recompilation required | X | O |

### ASan Example

```c
// asan_test.c
#include <stdlib.h>

int main(void) {
    int *arr = malloc(10 * sizeof(int));

    // Heap buffer overflow
    arr[10] = 42;

    free(arr);

    // Use After Free
    arr[0] = 100;

    return 0;
}
```

```bash
$ gcc -fsanitize=address -g asan_test.c -o asan_test
$ ./asan_test

=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x604000000028
WRITE of size 4 at 0x604000000028 thread T0
    #0 0x4011a3 in main asan_test.c:8
    #1 0x7f123456789a in __libc_start_main

0x604000000028 is located 0 bytes to the right of 40-byte region
allocated by thread T0 here:
    #0 0x7f1234567890 in malloc
    #1 0x401157 in main asan_test.c:5
```

### Other Sanitizers

```bash
# Undefined behavior check
gcc -fsanitize=undefined -g program.c -o program

# Thread error check
gcc -fsanitize=thread -g program.c -o program -pthread

# Combine multiple sanitizers
gcc -fsanitize=address,undefined -g program.c -o program
```

### UBSan Example

```c
// ubsan_test.c
#include <stdio.h>
#include <limits.h>

int main(void) {
    int x = INT_MAX;
    int y = x + 1;  // Integer overflow (undefined behavior)

    int arr[5] = {1, 2, 3, 4, 5};
    int z = arr[10];  // Array out of bounds

    int *ptr = NULL;
    // *ptr = 42;  // NULL dereference

    printf("%d %d\n", y, z);
    return 0;
}
```

```bash
$ gcc -fsanitize=undefined -g ubsan_test.c -o ubsan_test
$ ./ubsan_test

ubsan_test.c:7:15: runtime error: signed integer overflow:
2147483647 + 1 cannot be represented in type 'int'
```

---

## Common Memory Bugs

### 1. Buffer Overflow

```c
// Problem
char buffer[10];
strcpy(buffer, "This is too long!");  // Overflow

// Solution
char buffer[10];
strncpy(buffer, "This is too long!", sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';

// Or use snprintf
snprintf(buffer, sizeof(buffer), "%s", "This is too long!");
```

### 2. Use After Free

```c
// Problem
int *ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);
printf("%d\n", *ptr);  // Accessing freed memory

// Solution
int *ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);
ptr = NULL;  // Set to NULL after free
```

### 3. Double Free

```c
// Problem
int *ptr = malloc(sizeof(int));
free(ptr);
free(ptr);  // Double free

// Solution
int *ptr = malloc(sizeof(int));
free(ptr);
ptr = NULL;  // free(NULL) is safe
free(ptr);   // OK
```

### 4. Memory Leak

```c
// Problem
void process(void) {
    int *data = malloc(100);
    if (error_condition) {
        return;  // Leak!
    }
    // ...
    free(data);
}

// Solution 1: Using goto
void process(void) {
    int *data = malloc(100);
    if (error_condition) {
        goto cleanup;
    }
    // ...
cleanup:
    free(data);
}

// Solution 2: Structured
void process(void) {
    int *data = malloc(100);
    if (!error_condition) {
        // Normal processing
    }
    free(data);
}
```

### 5. Uninitialized Memory

```c
// Problem
int x;
printf("%d\n", x);  // Garbage value

int *arr = malloc(10 * sizeof(int));
printf("%d\n", arr[0]);  // Garbage value

// Solution
int x = 0;

int *arr = calloc(10, sizeof(int));  // Initialized to 0
// Or
int *arr = malloc(10 * sizeof(int));
memset(arr, 0, 10 * sizeof(int));
```

### 6. Stack Overflow

```c
// Problem: Infinite recursion
void infinite(void) {
    infinite();  // Stack overflow
}

// Problem: Large local variable
void big_local(void) {
    int huge_array[1000000];  // Potential stack overflow
}

// Solution: Dynamic allocation
void use_heap(void) {
    int *array = malloc(1000000 * sizeof(int));
    // ...
    free(array);
}
```

---

## Debugging Strategies

### 1. Make It Reproducible

```c
// Fix random seed
srand(12345);  // Always same results

// Log inputs
void log_input(const char *input) {
    FILE *f = fopen("input.log", "a");
    fprintf(f, "%s\n", input);
    fclose(f);
}
```

### 2. Binary Search for Bug Location

```c
// Comment out half the code and check if bug occurs
// Repeat by halving the section with the bug

// Using git bisect
// $ git bisect start
// $ git bisect bad HEAD
// $ git bisect good v1.0
```

### 3. Defensive Programming

```c
// Validate at function start
int process_data(int *data, size_t size) {
    // Precondition checks
    if (data == NULL) {
        fprintf(stderr, "Error: data is NULL\n");
        return -1;
    }
    if (size == 0) {
        fprintf(stderr, "Error: size is 0\n");
        return -1;
    }

    // Processing logic...

    return 0;
}
```

### 4. Logging Levels

```c
typedef enum {
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG
} LogLevel;

LogLevel current_level = LOG_INFO;

void log_message(LogLevel level, const char *fmt, ...) {
    if (level > current_level) return;

    const char *level_str[] = {"ERROR", "WARN", "INFO", "DEBUG"};

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[%s] ", level_str[level]);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// Usage
log_message(LOG_DEBUG, "Processing item %d", i);
log_message(LOG_ERROR, "Failed to open file: %s", filename);
```

---

## Exercises

### Problem 1: Find Memory Leaks

Find and fix all memory leaks in the following code.

```c
#include <stdlib.h>
#include <string.h>

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

    // No cleanup after processing!
}

int main(void) {
    process_students();
    return 0;
}
```

<details>
<summary>View Solution</summary>

```c
void free_student(Student *s) {
    if (s) {
        free(s->name);
        free(s->scores);
        free(s);
    }
}

void process_students(void) {
    Student *students[3];

    students[0] = create_student("Alice", 5);
    students[1] = create_student("Bob", 3);
    students[2] = create_student("Charlie", 4);

    // Cleanup
    for (int i = 0; i < 3; i++) {
        free_student(students[i]);
    }
}
```

</details>

### Problem 2: Using GDB

Debug the following code that causes a segmentation fault using GDB.

```c
#include <stdio.h>

void recursive(int n) {
    int arr[1000];
    arr[0] = n;
    if (n > 0) {
        recursive(n - 1);
    }
}

int main(void) {
    recursive(10000);
    return 0;
}
```

<details>
<summary>Solution</summary>

```bash
$ gcc -g -O0 program.c -o program
$ gdb ./program
(gdb) run
# Segmentation fault occurs
(gdb) bt
# Stack overflow confirmed - recursive function called too many times

# Solution: Reduce recursion depth or use iteration
```

</details>

---

## Next Step

- [Advanced Embedded Protocols](./19_Advanced_Embedded_Protocols.md) - PWM, I2C, SPI

---

## References

- [GDB Official Documentation](https://www.gnu.org/software/gdb/documentation/)
- [Valgrind Official Documentation](https://valgrind.org/docs/manual/)
- [AddressSanitizer Wiki](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [Debugging with GDB (Book)](https://sourceware.org/gdb/current/onlinedocs/gdb/)
