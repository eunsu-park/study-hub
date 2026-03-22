/*
 * Exercises for Lesson 12: Build Tools and Debugging
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex12 12_build_tools_and_debugging.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Exercise 1: Find the Bugs === */
/* Problem: Each buggy function has one or more defects. The fixed versions
 *          are provided alongside explanations. */

/* Bug 1: Off-by-one error */
void bug1_fixed(void) {
    printf("  Bug 1: Off-by-one in array access\n");

    int arr[5] = {10, 20, 30, 40, 50};

    /*
     * BUGGY VERSION:
     *   for (int i = 0; i <= 5; i++)  // accesses arr[5] -> out of bounds
     *       printf("%d ", arr[i]);
     *
     * FIX: use i < 5 (strict less-than for array size)
     */
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/* Bug 2: Dangling pointer / use-after-free */
void bug2_fixed(void) {
    printf("  Bug 2: Use-after-free\n");

    /*
     * BUGGY VERSION:
     *   char *str = malloc(20);
     *   strcpy(str, "hello");
     *   free(str);
     *   printf("%s\n", str);  // use-after-free!
     *
     * FIX: Don't access memory after freeing it.
     *      Set pointer to NULL after free.
     */
    char *str = malloc(20);
    if (!str) return;
    strcpy(str, "hello");
    printf("  Before free: %s\n", str);
    free(str);
    str = NULL;  /* Prevent accidental reuse */
    printf("  After free: pointer set to NULL (safe)\n");
}

/* Bug 3: Buffer overflow in string copy */
void bug3_fixed(void) {
    printf("  Bug 3: Buffer overflow in strcpy\n");

    /*
     * BUGGY VERSION:
     *   char dest[5];
     *   strcpy(dest, "This is a very long string");  // overflow!
     *
     * FIX: Use strncpy with proper size limiting, or check length first.
     */
    char dest[20];
    const char *src = "This is a very long string";
    size_t max_len = sizeof(dest) - 1;

    if (strlen(src) < sizeof(dest)) {
        strcpy(dest, src);
    } else {
        strncpy(dest, src, max_len);
        dest[max_len] = '\0';  /* strncpy doesn't guarantee null termination */
    }
    printf("  Safe copy: \"%s\"\n", dest);
}

/* Bug 4: Memory leak */
void bug4_fixed(void) {
    printf("  Bug 4: Memory leak\n");

    /*
     * BUGGY VERSION:
     *   for (int i = 0; i < 1000; i++) {
     *       int *p = malloc(sizeof(int) * 100);
     *       p[0] = i;
     *       // forgot to free(p) -> leaks 400KB total!
     *   }
     *
     * FIX: Always free allocated memory when done.
     */
    for (int i = 0; i < 5; i++) {  /* reduced iterations for demo */
        int *p = malloc(sizeof(int) * 100);
        if (!p) continue;
        p[0] = i;
        printf("  Allocated and used p[0]=%d, now freeing\n", p[0]);
        free(p);  /* FIX: free in the same scope */
    }
}

/* Bug 5: Integer sign comparison */
void bug5_fixed(void) {
    printf("  Bug 5: Signed/unsigned comparison\n");

    /*
     * BUGGY VERSION:
     *   int len = -1;
     *   unsigned int size = 5;
     *   if (len < size) printf("correct");  // -1 becomes huge unsigned!
     *
     * FIX: Cast or use same types for comparison.
     */
    int len = -1;
    unsigned int size = 5;

    /* Buggy comparison: len is promoted to unsigned, -1 becomes UINT_MAX */
    if ((unsigned int)len < size) {
        printf("  Unsigned comparison: -1 < 5 is FALSE (wrong!)\n");
    } else {
        printf("  Unsigned comparison: -1 >= 5 is TRUE (unexpected!)\n");
    }

    /* Fixed: compare as signed */
    if (len < (int)size) {
        printf("  Signed comparison: -1 < 5 is TRUE (correct)\n");
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Find the Bugs ===\n\n");
    bug1_fixed();
    bug2_fixed();
    bug3_fixed();
    bug4_fixed();
    bug5_fixed();
}

/* === Exercise 2: Makefile Writing === */
/* Problem: Write Makefiles for different project structures. */
void exercise_2(void) {
    printf("\n=== Exercise 2: Makefile Writing ===\n");

    /*
     * Makefile 1: Simple single-file project
     * ----------------------------------------
     * CC      = gcc
     * CFLAGS  = -Wall -Wextra -std=c11 -g
     * TARGET  = myprogram
     *
     * all: $(TARGET)
     *
     * $(TARGET): main.c
     * 	$(CC) $(CFLAGS) -o $@ $<
     *
     * .PHONY: clean run
     * clean:
     * 	rm -f $(TARGET)
     * run: $(TARGET)
     * 	./$(TARGET)
     */
    printf("Makefile 1: Single-file project\n");
    printf("  Pattern: direct source-to-binary rule\n");

    /*
     * Makefile 2: Multi-file project with separate compilation
     * ---------------------------------------------------------
     * CC       = gcc
     * CFLAGS   = -Wall -Wextra -std=c11 -g
     * LDFLAGS  = -lm
     *
     * SRCDIR   = src
     * OBJDIR   = obj
     * INCDIR   = include
     *
     * SRCS     = $(wildcard $(SRCDIR)/*.c)
     * OBJS     = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
     * DEPS     = $(OBJS:.o=.d)
     * TARGET   = build/app
     *
     * all: $(TARGET)
     *
     * $(TARGET): $(OBJS) | build
     * 	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
     *
     * $(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
     * 	$(CC) $(CFLAGS) -I$(INCDIR) -MMD -MP -c $< -o $@
     *
     * $(OBJDIR) build:
     * 	mkdir -p $@
     *
     * -include $(DEPS)
     *
     * .PHONY: clean
     * clean:
     * 	rm -rf $(OBJDIR) build
     */
    printf("Makefile 2: Multi-file with auto-dependency tracking\n");
    printf("  Key features: separate obj dir, -MMD -MP for deps\n");

    /*
     * Makefile 3: Project with tests
     * --------------------------------
     * (extends Makefile 2)
     *
     * TESTDIR   = tests
     * TEST_SRCS = $(wildcard $(TESTDIR)/*.c)
     * TEST_BINS = $(patsubst $(TESTDIR)/%.c,build/test_%,$(TEST_SRCS))
     * LIB_OBJS  = $(filter-out $(OBJDIR)/main.o,$(OBJS))
     *
     * .PHONY: test
     * test: $(TEST_BINS)
     * 	@for t in $^; do echo "Running $$t..."; $$t || exit 1; done
     * 	@echo "All tests passed!"
     *
     * build/test_%: $(TESTDIR)/%.c $(LIB_OBJS) | build
     * 	$(CC) $(CFLAGS) -I$(INCDIR) -o $@ $^ $(LDFLAGS)
     */
    printf("Makefile 3: Adds test target with per-test binaries\n");
    printf("  LIB_OBJS filters out main.o so tests provide own main()\n");
}

/* === Exercise 3: Debugging Workflow Reference === */
/* Problem: Document common debugging tool commands and workflows. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Debugging Workflow Reference ===\n");

    printf("\n--- GDB Essentials ---\n");
    printf("  gcc -g -O0 -o app app.c     # compile with debug info\n");
    printf("  gdb ./app                    # start debugger\n");
    printf("  (gdb) break main             # set breakpoint\n");
    printf("  (gdb) run                    # start execution\n");
    printf("  (gdb) next / step            # step over / into\n");
    printf("  (gdb) print var              # inspect variable\n");
    printf("  (gdb) backtrace              # show call stack\n");
    printf("  (gdb) watch var              # break when var changes\n");

    printf("\n--- Valgrind for Memory Errors ---\n");
    printf("  valgrind --leak-check=full ./app\n");
    printf("  Detects: leaks, use-after-free, uninitialized reads,\n");
    printf("           double-free, buffer overflows (on heap)\n");

    printf("\n--- AddressSanitizer (compile-time) ---\n");
    printf("  gcc -fsanitize=address -g -o app app.c\n");
    printf("  ./app  # crashes with detailed report on memory errors\n");
    printf("  Faster than Valgrind, catches stack overflows too\n");

    printf("\n--- UndefinedBehaviorSanitizer ---\n");
    printf("  gcc -fsanitize=undefined -g -o app app.c\n");
    printf("  Catches: signed overflow, null deref, shift errors\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
