/*
 * Exercises for Lesson 02: C Basics Review
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex02 02_c_basics_review.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Exercise 1: Data Type Sizes and Format Specifiers === */
/* Problem: Print size and sample value for every fundamental data type. */
void exercise_1(void) {
    printf("=== Exercise 1: Data Type Sizes and Format Specifiers ===\n");

    char c = 'A';
    short s = 32767;
    int i = 2147483647;
    long l = 1234567890L;
    long long ll = 9223372036854775807LL;
    unsigned int ui = 4294967295U;
    float f = 3.14f;
    double d = 3.141592653589793;

    printf("%-12s size: %zu bytes, value: %c (%d)\n",
           "char", sizeof(char), c, c);
    printf("%-12s size: %zu bytes, value: %hd\n",
           "short", sizeof(short), s);
    printf("%-12s size: %zu bytes, value: %d\n",
           "int", sizeof(int), i);
    printf("%-12s size: %zu bytes, value: %ld\n",
           "long", sizeof(long), l);
    printf("%-12s size: %zu bytes, value: %lld\n",
           "long long", sizeof(long long), ll);
    printf("%-12s size: %zu bytes, value: %u\n",
           "unsigned int", sizeof(unsigned int), ui);
    printf("%-12s size: %zu bytes, value: %f\n",
           "float", sizeof(float), f);
    printf("%-12s size: %zu bytes, value: %.15f\n",
           "double", sizeof(double), d);

    /*
     * Q1: Why use %zu for sizeof output rather than %d?
     * A: sizeof returns size_t, which is an unsigned type (typically unsigned long
     *    on 64-bit). Using %d would cause undefined behavior because %d expects
     *    int. On 64-bit systems, size_t is 8 bytes but int is only 4 bytes,
     *    so %d could print garbage or negative values for large sizes.
     *
     * Q2: What happens if you use %d to print a long long on a 64-bit system?
     * A: Undefined behavior. %d reads only 4 bytes from the argument, leaving
     *    the remaining 4 bytes misaligned on the stack. This corrupts subsequent
     *    format specifiers. The correct specifier is %lld.
     */
    printf("\n--- Demonstration of format specifier mismatch ---\n");
    printf("Correct:   sizeof(int)  = %zu\n", sizeof(int));
    printf("Correct:   long long    = %lld\n", ll);
}

/* === Exercise 2: Pointer Swap Dissection === */
/* Problem: Show why wrong_swap fails and swap succeeds by printing addresses. */

void wrong_swap(int a, int b) {
    printf("  Inside wrong_swap: &a=%p, &b=%p\n", (void *)&a, (void *)&b);
    int temp = a;
    a = b;
    b = temp;
    /* a and b are local copies -- changes do not affect caller's variables */
}

void swap(int *a, int *b) {
    printf("  Inside swap:       a=%p,  b=%p\n", (void *)a, (void *)b);
    int temp = *a;
    *a = *b;
    *b = temp;
    /* a and b point to caller's actual variables, so changes persist */
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Pointer Swap Dissection ===\n");

    int x = 10, y = 20;
    printf("Before:              &x=%p, &y=%p\n", (void *)&x, (void *)&y);
    printf("                     x=%d, y=%d\n", x, y);

    printf("\nCalling wrong_swap(x, y):\n");
    wrong_swap(x, y);
    printf("After wrong_swap:    x=%d, y=%d (unchanged!)\n", x, y);

    printf("\nCalling swap(&x, &y):\n");
    swap(&x, &y);
    printf("After swap:          x=%d, y=%d (swapped!)\n", x, y);

    /*
     * Explanation:
     * - wrong_swap receives copies of x and y. The addresses &a and &b inside
     *   the function are different from &x and &y in main because they are
     *   separate local variables on a new stack frame. Swapping a and b only
     *   modifies these local copies.
     *
     * - swap receives pointers to x and y. The pointer values a and b hold
     *   the same addresses as &x and &y, so dereferencing and swapping
     *   modifies the actual variables in the caller's stack frame.
     */
}

/* === Exercise 3: Array and Pointer Arithmetic === */
/* Problem: Three different iteration methods over an array, all producing same output. */
void exercise_3(void) {
    printf("\n=== Exercise 3: Array and Pointer Arithmetic ===\n");

    int arr[] = {10, 20, 30, 40, 50};
    size_t n = sizeof(arr) / sizeof(arr[0]);

    printf("Array has %zu elements (sizeof(arr)=%zu, sizeof(arr[0])=%zu)\n\n",
           n, sizeof(arr), sizeof(arr[0]));

    /* Method 1: Index-based iteration */
    printf("Method 1 (index):     ");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    /* Method 2: Pointer offset arithmetic */
    printf("Method 2 (ptr+offset): ");
    int *p = arr;
    for (size_t i = 0; i < n; i++) {
        printf("%d ", *(p + i));
    }
    printf("\n");

    /* Method 3: Pointer increment */
    printf("Method 3 (ptr++):      ");
    for (int *q = arr; q < arr + n; q++) {
        printf("%d ", *q);
    }
    printf("\n");

    /*
     * All three methods are equivalent:
     * - arr[i] is syntactic sugar for *(arr + i)
     * - Pointer arithmetic advances by sizeof(int) per increment
     * - arr decays to &arr[0] when used in expressions
     */
}

/* === Exercise 4: Dynamic String Builder === */
/* Problem: Dynamically allocate and construct a greeting string. */

char *build_greeting(const char *name) {
    /* "Hello, " (7) + name + "!" (1) + '\0' (1) */
    size_t len = strlen("Hello, ") + strlen(name) + strlen("!") + 1;
    char *greeting = malloc(len);
    if (!greeting) {
        fprintf(stderr, "malloc failed\n");
        return NULL;
    }

    strcpy(greeting, "Hello, ");
    strcat(greeting, name);
    strcat(greeting, "!");

    return greeting; /* Caller is responsible for calling free() */
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Dynamic String Builder ===\n");

    const char *names[] = {"Alice", "Bob", "World"};

    for (int i = 0; i < 3; i++) {
        char *greeting = build_greeting(names[i]);
        if (greeting) {
            printf("%s\n", greeting);
            free(greeting); /* Prevent memory leak */
            greeting = NULL; /* Good practice: nullify after free */
        }
    }

    /*
     * To verify no memory leaks with Valgrind:
     *   gcc -g -std=c11 -o ex02 02_c_basics_review.c
     *   valgrind --leak-check=full ./ex02
     *
     * Expected output should show:
     *   All heap blocks were freed -- no leaks are possible
     */
}

/* === Exercise 5: Struct-Based Student Record === */
/* Problem: Dynamic struct array with accessor and mutator functions. */

typedef struct {
    char name[64];
    int id;
    float gpa;
} Student;

void print_student(const Student *s) {
    /* Use -> operator to access members through a pointer */
    printf("  [ID %d] %-10s GPA: %.2f\n", s->id, s->name, s->gpa);
}

void raise_gpa(Student *s, float delta) {
    s->gpa += delta;
    /* Cap GPA at 4.0 */
    if (s->gpa > 4.0f) {
        s->gpa = 4.0f;
    }
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Struct-Based Student Record ===\n");

    /* Dynamically allocate array of 3 students */
    Student *students = malloc(3 * sizeof(Student));
    if (!students) {
        fprintf(stderr, "malloc failed\n");
        return;
    }

    /* Initialize students */
    strcpy(students[0].name, "Alice");
    students[0].id = 1001;
    students[0].gpa = 3.5f;

    strcpy(students[1].name, "Bob");
    students[1].id = 1002;
    students[1].gpa = 3.8f;

    strcpy(students[2].name, "Charlie");
    students[2].id = 1003;
    students[2].gpa = 3.9f;

    printf("Before GPA raise:\n");
    for (int i = 0; i < 3; i++) {
        print_student(&students[i]);
    }

    /* Raise each student's GPA by 0.3 */
    for (int i = 0; i < 3; i++) {
        raise_gpa(&students[i], 0.3f);
    }

    printf("\nAfter GPA raise (+0.3, capped at 4.0):\n");
    for (int i = 0; i < 3; i++) {
        print_student(&students[i]);
    }

    /* Free the dynamically allocated array */
    free(students);
    students = NULL;
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
