/*
 * Exercises for Lesson 18: Debugging and Memory Analysis
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex18 18_debugging_memory_analysis.c
 *
 * For memory analysis:
 *   gcc -g -O0 -std=c11 -o ex18_debug 18_debugging_memory_analysis.c
 *   valgrind --leak-check=full ./ex18_debug
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Exercise 1: Find Memory Leaks === */
/* Problem: The original code creates Student structs with nested allocations
 * but never frees them. Fix by adding proper cleanup functions. */

typedef struct {
    char *name;
    int *scores;
    int num_scores;
} Student;

Student *create_student(const char *name, int num_scores) {
    Student *s = malloc(sizeof(Student));
    if (!s) return NULL;

    s->name = malloc(strlen(name) + 1);
    if (!s->name) {
        free(s);
        return NULL;
    }
    strcpy(s->name, name);

    s->scores = malloc((size_t)num_scores * sizeof(int));
    if (!s->scores) {
        free(s->name);
        free(s);
        return NULL;
    }
    s->num_scores = num_scores;

    /* Initialize scores to 0 for safety */
    for (int i = 0; i < num_scores; i++) {
        s->scores[i] = 0;
    }

    return s;
}

/*
 * Solution: free_student deallocates all nested memory in the correct order.
 *
 * Why order matters:
 * - If we free(s) first, s->name and s->scores become dangling references
 *   and we can no longer access them to free them.
 * - Always free innermost (child) allocations first, then the parent struct.
 *
 * The NULL check prevents double-free or free-of-NULL errors.
 */
void free_student(Student *s) {
    if (s) {
        free(s->name);    /* Free the name string */
        free(s->scores);  /* Free the scores array */
        free(s);          /* Free the struct itself */
    }
}

void print_student(const Student *s) {
    printf("  Student: %-10s (%d scores: ", s->name, s->num_scores);
    for (int i = 0; i < s->num_scores; i++) {
        printf("%d%s", s->scores[i], (i < s->num_scores - 1) ? ", " : "");
    }
    printf(")\n");
}

void exercise_1(void) {
    printf("=== Exercise 1: Find Memory Leaks (Fixed) ===\n");

    /*
     * Original buggy code had no cleanup:
     *
     *   void process_students(void) {
     *       Student *students[3];
     *       students[0] = create_student("Alice", 5);
     *       students[1] = create_student("Bob", 3);
     *       students[2] = create_student("Charlie", 4);
     *       // No cleanup after processing! <-- 3 LEAKS
     *   }
     *
     * Leak analysis (original code):
     * - Each Student has 3 allocations: struct, name, scores
     * - 3 students x 3 allocations = 9 memory blocks leaked
     *
     * Valgrind would report:
     *   LEAK SUMMARY:
     *     definitely lost: 9 blocks
     */

    Student *students[3];

    students[0] = create_student("Alice", 5);
    students[1] = create_student("Bob", 3);
    students[2] = create_student("Charlie", 4);

    /* Set some scores for demonstration */
    if (students[0]) {
        int alice_scores[] = {95, 88, 92, 78, 85};
        memcpy(students[0]->scores, alice_scores, 5 * sizeof(int));
    }
    if (students[1]) {
        int bob_scores[] = {72, 81, 90};
        memcpy(students[1]->scores, bob_scores, 3 * sizeof(int));
    }
    if (students[2]) {
        int charlie_scores[] = {88, 94, 76, 91};
        memcpy(students[2]->scores, charlie_scores, 4 * sizeof(int));
    }

    /* Print all students */
    for (int i = 0; i < 3; i++) {
        if (students[i]) {
            print_student(students[i]);
        }
    }

    /* FIX: Proper cleanup -- free all students */
    for (int i = 0; i < 3; i++) {
        free_student(students[i]);
        students[i] = NULL; /* Nullify after free to prevent use-after-free */
    }

    printf("  All students freed successfully. No memory leaks.\n");
}

/* === Exercise 2: Using GDB (Stack Overflow Diagnosis) === */
/* Problem: Debug a program that causes a segmentation fault due to stack overflow.
 * The original has recursive(10000) with a 1000-int local array per call. */

/*
 * Buggy version (causes stack overflow):
 *
 *   void recursive(int n) {
 *       int arr[1000];          // 4000 bytes per stack frame
 *       arr[0] = n;
 *       if (n > 0) {
 *           recursive(n - 1);   // 10000 recursive calls
 *       }
 *   }
 *
 * Analysis:
 * - Each call allocates ~4000 bytes (1000 ints) on the stack
 * - 10000 calls x 4000 bytes = ~40 MB of stack
 * - Default stack size is typically 8 MB (Linux) or 1 MB (some systems)
 * - Result: SIGSEGV (segmentation fault) from stack overflow
 *
 * GDB session:
 *   $ gcc -g -O0 program.c -o program
 *   $ gdb ./program
 *   (gdb) run
 *   Program received signal SIGSEGV, Segmentation fault.
 *   (gdb) bt
 *   #0  recursive (n=7845) at program.c:3
 *   #1  recursive (n=7846) at program.c:5
 *   ...  (thousands of frames)
 *   -> Stack overflow confirmed
 */

/* Fixed version 1: Iterative approach (eliminates stack overflow) */
void fixed_iterative(int n) {
    /*
     * Instead of 10000 stack frames each with a 1000-int array,
     * we use a single heap allocation and iterate.
     */
    int *arr = malloc((size_t)n * sizeof(int));
    if (!arr) {
        fprintf(stderr, "  malloc failed for %d elements\n", n);
        return;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = n - i;
    }

    printf("  First element: %d, Last element: %d\n", arr[0], arr[n - 1]);
    free(arr);
}

/* Fixed version 2: Reduced recursion depth (safe for small n) */
void fixed_recursive(int n) {
    if (n <= 0) return;
    int arr[10]; /* Much smaller local array */
    arr[0] = n;
    printf("  Level %d: arr[0]=%d\n", n, arr[0]);
    if (n > 1) {
        fixed_recursive(n - 1);
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Using GDB (Stack Overflow Fix) ===\n");

    printf("\nFix 1 - Iterative approach (handles n=10000 easily):\n");
    fixed_iterative(10000);

    printf("\nFix 2 - Safe recursion (small depth, small local array):\n");
    fixed_recursive(5);

    /*
     * Key lessons:
     * 1. Large local arrays in recursive functions are dangerous
     * 2. Use heap allocation (malloc) for large data structures
     * 3. Convert deep recursion to iteration when possible
     * 4. Use GDB's 'bt' (backtrace) to identify stack overflow
     * 5. 'ulimit -s' shows/sets stack size limit on Linux
     */
    printf("\n  Lessons learned:\n");
    printf("  - Avoid large local arrays in recursive functions\n");
    printf("  - Use malloc for large data; convert deep recursion to iteration\n");
    printf("  - GDB 'bt' command reveals stack overflow immediately\n");
}

int main(void) {
    exercise_1();
    exercise_2();

    printf("\nAll exercises completed!\n");
    return 0;
}
