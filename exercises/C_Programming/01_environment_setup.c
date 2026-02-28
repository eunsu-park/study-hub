/*
 * Exercises for Lesson 01: Environment Setup
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex01 01_environment_setup.c
 */
#include <stdio.h>
#include <stdlib.h>

/* === Exercise 1: Verify Your Toolchain === */
/* Problem: Print sizes of fundamental C types using sizeof. */
void exercise_1(void) {
    printf("=== Exercise 1: Verify Your Toolchain ===\n");

    /* Print sizes of all fundamental types to understand your platform */
    printf("Type Sizes on This Platform:\n");
    printf("  char      : %zu bytes\n", sizeof(char));
    printf("  short     : %zu bytes\n", sizeof(short));
    printf("  int       : %zu bytes\n", sizeof(int));
    printf("  long      : %zu bytes\n", sizeof(long));
    printf("  long long : %zu bytes\n", sizeof(long long));
    printf("  float     : %zu bytes\n", sizeof(float));
    printf("  double    : %zu bytes\n", sizeof(double));
    printf("  void *    : %zu bytes\n", sizeof(void *));

    /*
     * Notes:
     * - On most 64-bit Linux systems: int=4, long=8 (LP64 model)
     * - On 64-bit Windows (LLP64): int=4, long=4, long long=8
     * - sizeof returns size_t, so %zu is the correct format specifier
     */

    printf("\nPointer size indicates %zu-bit platform\n",
           sizeof(void *) * 8);
}

/* === Exercise 2: Compiler Flags Exploration === */
/* Problem: Demonstrate the buggy program and explain what each flag catches. */
void exercise_2(void) {
    printf("\n=== Exercise 2: Compiler Flags Exploration ===\n");

    /*
     * The buggy program from the exercise:
     *
     *   int x;                     // Uninitialized variable
     *   float ratio = 1 / 3;      // Integer division (likely a bug)
     *   printf("%d %f\n", x, ratio);
     *
     * Compilation results with different flags:
     *
     * 1. gcc buggy.c -o buggy
     *    -> No warnings. Compiles silently.
     *
     * 2. gcc -Wall buggy.c -o buggy
     *    -> Warning: 'x' is used uninitialized
     *    -> -Wall enables common warnings like uninitialized variables.
     *
     * 3. gcc -Wall -Wextra buggy.c -o buggy
     *    -> Same as above, plus potential extra warnings.
     *    -> -Wextra adds warnings for things like unused parameters,
     *       signed/unsigned comparisons, etc.
     *
     * 4. gcc -Wall -Wextra -std=c11 buggy.c -o buggy
     *    -> Same warnings, but now enforces C11 standard conformance.
     *    -> Disallows some GCC extensions; stricter parsing.
     *
     * Why -Wextra catches more than -Wall:
     *   -Wall is NOT "all warnings" -- it enables a commonly useful subset.
     *   -Wextra adds additional checks like comparison between signed and
     *   unsigned, missing field initializers, unused parameters, and more.
     *   Always use both in development.
     */

    /* Demonstrate the fixed version of the buggy code */
    int x = 0;                     /* Fix: initialize variable */
    float ratio = 1.0f / 3.0f;    /* Fix: use float literals for float division */
    printf("Fixed program output: x=%d, ratio=%f\n", x, ratio);
    printf("(Integer division 1/3 = %d, float division 1.0/3.0 = %f)\n",
           1 / 3, 1.0 / 3.0);
}

/* === Exercise 3: Multi-File Makefile === */
/* Problem: Create a two-file project with Makefile. Solution shown inline. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Multi-File Makefile ===\n");

    /*
     * File: math_utils.h
     * -------------------
     * #ifndef MATH_UTILS_H
     * #define MATH_UTILS_H
     *
     * int square(int n);
     * int cube(int n);
     *
     * #endif
     *
     * File: math_utils.c
     * -------------------
     * #include "math_utils.h"
     *
     * int square(int n) { return n * n; }
     * int cube(int n) { return n * n * n; }
     *
     * File: main.c
     * -------------------
     * #include <stdio.h>
     * #include "math_utils.h"
     *
     * int main(void) {
     *     int n;
     *     printf("Enter a number: ");
     *     scanf("%d", &n);
     *     printf("Square: %d\n", square(n));
     *     printf("Cube:   %d\n", cube(n));
     *     return 0;
     * }
     *
     * File: Makefile
     * -------------------
     * CC      = gcc
     * CFLAGS  = -Wall -Wextra -std=c11
     * TARGET  = mathprog
     * SRCS    = main.c math_utils.c
     * OBJS    = $(SRCS:.c=.o)
     *
     * all: $(TARGET)
     *
     * $(TARGET): $(OBJS)
     * 	$(CC) $(CFLAGS) -o $@ $^
     *
     * %.o: %.c
     * 	$(CC) $(CFLAGS) -c $< -o $@
     *
     * .PHONY: clean
     * clean:
     * 	rm -f $(OBJS) $(TARGET)
     */

    /* Inline demonstration of math_utils functionality */
    int n = 5;
    printf("square(%d) = %d\n", n, n * n);
    printf("cube(%d)   = %d\n", n, n * n * n);
}

/* === Exercise 4: GDB Step-Through === */
/* Problem: Write a factorial program suitable for GDB debugging. */

/* Factorial function designed for GDB practice */
long factorial(int n) {
    long product = 1;
    for (int i = 1; i <= n; i++) {
        /* Set breakpoint here: (gdb) break factorial.c:XX */
        product *= i;
        /* After each step, inspect with:
         *   (gdb) print i
         *   (gdb) print product
         * To change i mid-execution:
         *   (gdb) set variable i = 8
         *   (gdb) continue
         * This is useful for skipping iterations or testing edge cases
         * without recompiling.
         */
    }
    return product;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: GDB Step-Through ===\n");

    int n = 10;
    long result = factorial(n);
    printf("factorial(%d) = %ld\n", n, result);

    /*
     * GDB session walkthrough:
     *
     * $ gcc -g -O0 -std=c11 -o ex01 01_environment_setup.c
     * $ gdb ./ex01
     * (gdb) break factorial          # Set breakpoint at function entry
     * (gdb) run                      # Start execution
     * (gdb) break 93                 # Break inside the loop (product *= i line)
     * (gdb) continue                 # Hit first iteration
     * (gdb) print i                  # Should be 1
     * (gdb) print product            # Should be 1
     * (gdb) next                     # Step to next iteration
     * (gdb) print i                  # Should be 2
     * (gdb) print product            # Should be 2
     * (gdb) next                     # i=3
     * (gdb) print product            # Should be 6
     * (gdb) set variable i = 8       # Skip ahead to i=8
     * (gdb) continue                 # Watch product jump from 6 * 8 = 48
     * (gdb) print product            # Observe altered result
     *
     * Why modifying variables mid-execution is useful:
     * - Skip slow or repetitive iterations during debugging
     * - Test boundary conditions without changing source code
     * - Force specific execution paths to reproduce rare bugs
     */
}

/* === Exercise 5: Project Structure Scaffold === */
/* Problem: Describe the project structure and Makefile with auto-dependencies. */
void exercise_5(void) {
    printf("\n=== Exercise 5: Project Structure Scaffold ===\n");

    /*
     * Directory structure:
     *
     * string_utils/
     * ├── include/
     * │   └── string_utils.h
     * ├── src/
     * │   ├── string_utils.c
     * │   └── main.c
     * ├── tests/
     * │   └── test_utils.c
     * ├── build/
     * └── Makefile
     *
     * Makefile with auto-dependency generation:
     * -------------------------------------------
     * CC       = gcc
     * CFLAGS   = -Wall -Wextra -std=c11 -Iinclude
     * DEPFLAGS = -MMD -MP
     *
     * SRCDIR   = src
     * BUILDDIR = build
     * TESTDIR  = tests
     *
     * SRCS     = $(wildcard $(SRCDIR)/*.c)
     * OBJS     = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))
     * DEPS     = $(OBJS:.o=.d)
     * TARGET   = $(BUILDDIR)/string_utils
     *
     * all: $(TARGET)
     *
     * $(TARGET): $(OBJS)
     * 	$(CC) $(CFLAGS) -o $@ $^
     *
     * $(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
     * 	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
     *
     * $(BUILDDIR):
     * 	mkdir -p $(BUILDDIR)
     *
     * .PHONY: test clean
     *
     * test: $(BUILDDIR)/test_utils
     * 	./$(BUILDDIR)/test_utils
     *
     * $(BUILDDIR)/test_utils: $(TESTDIR)/test_utils.c $(BUILDDIR)/string_utils.o
     * 	$(CC) $(CFLAGS) -o $@ $^
     *
     * clean:
     * 	rm -rf $(BUILDDIR)
     *
     * -include $(DEPS)
     * -------------------------------------------
     *
     * Key features:
     * - $(DEPFLAGS) = -MMD -MP generates .d dependency files alongside .o files
     * - -MMD: Generate dependency file for user headers (not system headers)
     * - -MP: Add phony targets for headers to prevent errors when headers are deleted
     * - "-include $(DEPS)": Include .d files silently (no error if missing on first build)
     * - Changing any header auto-triggers recompilation of dependent .c files
     */

    printf("See source comments for complete project structure and Makefile.\n");
    printf("Key feature: -MMD -MP flags generate auto-dependency files (.d)\n");
    printf("so changing a header recompiles all dependent sources.\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
