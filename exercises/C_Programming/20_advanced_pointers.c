/*
 * Exercises for Lesson 20: Advanced Pointers
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex20 20_advanced_pointers.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* === Exercise 1: Reverse Array === */
/* Problem: Reverse an array in place using only pointers (no indexing). */

void reverse_array(int *arr, int size) {
    /*
     * Two-pointer technique:
     * - 'left' starts at the beginning
     * - 'right' starts at the last element
     * - Swap elements and move pointers inward until they meet
     *
     * This is O(n/2) swaps, O(1) extra space.
     */
    int *left = arr;
    int *right = arr + size - 1;

    while (left < right) {
        /* Swap using pointer dereference */
        int temp = *left;
        *left = *right;
        *right = temp;

        left++;
        right--;
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Reverse Array ===\n");

    int arr[] = {1, 2, 3, 4, 5};
    int size = (int)(sizeof(arr) / sizeof(arr[0]));

    printf("Before: ");
    for (int i = 0; i < size; i++) printf("%d ", arr[i]);
    printf("\n");

    reverse_array(arr, size);

    printf("After:  ");
    for (int i = 0; i < size; i++) printf("%d ", arr[i]);
    printf("\n");

    /* Test with even-length array */
    int arr2[] = {10, 20, 30, 40};
    int size2 = 4;

    printf("\nBefore: ");
    for (int i = 0; i < size2; i++) printf("%d ", arr2[i]);
    printf("\n");

    reverse_array(arr2, size2);

    printf("After:  ");
    for (int i = 0; i < size2; i++) printf("%d ", arr2[i]);
    printf("\n");
}

/* === Exercise 2: Reverse Words in String === */
/* Problem: Convert "Hello World" to "World Hello". */

/* Helper: reverse a portion of a string in place */
static void reverse_str_range(char *start, char *end) {
    while (start < end) {
        char temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

void reverse_words(char *str) {
    /*
     * Three-step algorithm (in-place, O(n) time, O(1) space):
     *
     * 1. Reverse the entire string:
     *    "Hello World" -> "dlroW olleH"
     *
     * 2. Reverse each word individually:
     *    "dlroW" -> "World"
     *    "olleH" -> "Hello"
     *    Result: "World Hello"
     *
     * This works because reversing twice cancels out for each word,
     * but the word ORDER gets reversed by the first full reversal.
     */
    if (!str || !*str) return;

    int len = (int)strlen(str);

    /* Step 1: Reverse entire string */
    reverse_str_range(str, str + len - 1);

    /* Step 2: Reverse each word */
    char *word_start = str;
    for (char *p = str; ; p++) {
        if (*p == ' ' || *p == '\0') {
            /* Reverse the word from word_start to p-1 */
            if (word_start < p) {
                reverse_str_range(word_start, p - 1);
            }
            if (*p == '\0') break;
            word_start = p + 1;
        }
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Reverse Words in String ===\n");

    char str1[] = "Hello World";
    printf("Before: \"%s\"\n", str1);
    reverse_words(str1);
    printf("After:  \"%s\"\n", str1);

    char str2[] = "The quick brown fox";
    printf("\nBefore: \"%s\"\n", str2);
    reverse_words(str2);
    printf("After:  \"%s\"\n", str2);

    char str3[] = "SingleWord";
    printf("\nBefore: \"%s\"\n", str3);
    reverse_words(str3);
    printf("After:  \"%s\"\n", str3);
}

/* === Exercise 3: Reverse Linked List === */
/* Problem: Reverse a singly linked list. */

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node *create_node(int data) {
    Node *n = malloc(sizeof(Node));
    if (!n) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    n->data = data;
    n->next = NULL;
    return n;
}

Node *reverse_list(Node *head) {
    /*
     * Iterative reversal using three pointers:
     *
     *   prev -> NULL
     *   curr -> 1 -> 2 -> 3 -> NULL
     *
     * Each iteration:
     *   1. Save curr->next in 'next'
     *   2. Point curr->next to prev (reverse the link)
     *   3. Advance prev and curr forward
     *
     * After all iterations:
     *   NULL <- 1 <- 2 <- 3
     *                     ^
     *                    prev (new head)
     *
     * Time: O(n), Space: O(1)
     */
    Node *prev = NULL;
    Node *curr = head;

    while (curr != NULL) {
        Node *next = curr->next;  /* Save next node */
        curr->next = prev;        /* Reverse the link */
        prev = curr;              /* Advance prev */
        curr = next;              /* Advance curr */
    }

    return prev; /* prev is the new head */
}

static void print_list(const Node *head) {
    for (const Node *n = head; n != NULL; n = n->next) {
        printf("%d", n->data);
        if (n->next) printf(" -> ");
    }
    printf(" -> NULL\n");
}

static void free_list(Node *head) {
    while (head) {
        Node *next = head->next;
        free(head);
        head = next;
    }
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Reverse Linked List ===\n");

    /* Build list: 1 -> 2 -> 3 -> 4 -> 5 -> NULL */
    Node *head = create_node(1);
    head->next = create_node(2);
    head->next->next = create_node(3);
    head->next->next->next = create_node(4);
    head->next->next->next->next = create_node(5);

    printf("Before: ");
    print_list(head);

    head = reverse_list(head);

    printf("After:  ");
    print_list(head);

    free_list(head);
}

/* === Exercise 4: Function Pointer Calculator === */
/* Problem: Implement four arithmetic operations using a function pointer array. */

typedef double (*ArithOp)(double, double);

static double add(double a, double b) { return a + b; }
static double sub(double a, double b) { return a - b; }
static double mul(double a, double b) { return a * b; }
static double divide(double a, double b) {
    if (b == 0.0) {
        fprintf(stderr, "  Error: division by zero\n");
        return 0.0;
    }
    return a / b;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Function Pointer Calculator ===\n");

    /*
     * Function pointer array maps operator characters to implementations.
     *
     * Why function pointers?
     * - Eliminates long if-else or switch chains
     * - Easily extensible: add new operations by adding entries
     * - Same pattern used by qsort(), signal(), atexit() in C stdlib
     * - Foundation for callback-based designs
     */

    /* Map operator character to function */
    struct {
        char op;
        const char *name;
        ArithOp func;
    } operations[] = {
        {'+', "add",      add},
        {'-', "subtract", sub},
        {'*', "multiply", mul},
        {'/', "divide",   divide},
    };

    int num_ops = (int)(sizeof(operations) / sizeof(operations[0]));

    /* Test expressions */
    struct {
        double a;
        char op;
        double b;
    } tests[] = {
        {3.0, '+', 4.0},
        {10.0, '-', 3.0},
        {6.0, '*', 7.0},
        {15.0, '/', 4.0},
        {10.0, '/', 0.0},  /* Division by zero test */
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    for (int t = 0; t < num_tests; t++) {
        double a = tests[t].a;
        char op = tests[t].op;
        double b = tests[t].b;

        /* Find the matching operation */
        ArithOp func = NULL;
        for (int i = 0; i < num_ops; i++) {
            if (operations[i].op == op) {
                func = operations[i].func;
                break;
            }
        }

        if (func) {
            double result = func(a, b);
            printf("  %.0f %c %.0f = %.2f\n", a, op, b, result);
        } else {
            printf("  Unknown operator: '%c'\n", op);
        }
    }

    /*
     * Alternative: Direct array indexing by ASCII value (compact but less readable)
     *
     *   ArithOp ops[256] = {0};
     *   ops['+'] = add;
     *   ops['-'] = sub;
     *   ops['*'] = mul;
     *   ops['/'] = divide;
     *
     *   // Usage:
     *   if (ops[(unsigned char)op])
     *       result = ops[(unsigned char)op](a, b);
     */
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();

    printf("\nAll exercises completed!\n");
    return 0;
}
