/*
 * Exercises for Lesson 15: Debugging and Profiling
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex15 15_debugging_and_profiling.c -lm
 * Profile: gcc -Wall -Wextra -std=c11 -pg -o ex15 15_debugging_and_profiling.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

/* === Exercise 1: Bug Hunting === */
/* Problem: Find and fix bugs in the following functions. Each has at least
 *          one subtle defect. */

/* Bug 1: Binary search — original has off-by-one causing infinite loop */
int binary_search(const int *arr, int size, int target) {
    int lo = 0, hi = size - 1;

    /*
     * BUGGY VERSION:
     *   while (lo < hi) {           // should be lo <= hi
     *       int mid = (lo + hi) / 2; // can overflow for large lo+hi
     *
     * FIX: lo <= hi, and use safe midpoint calculation.
     */
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;  /* safe: no overflow */
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Bug 2: String concatenation — original doesn't account for null terminator */
char *safe_strcat(const char *s1, const char *s2) {
    /*
     * BUGGY VERSION:
     *   char *result = malloc(strlen(s1) + strlen(s2));  // missing +1 for '\0'
     *   strcpy(result, s1);
     *   strcat(result, s2);
     *
     * FIX: allocate strlen(s1) + strlen(s2) + 1
     */
    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);
    char *result = malloc(len1 + len2 + 1);
    if (!result) return NULL;
    memcpy(result, s1, len1);
    memcpy(result + len1, s2, len2 + 1);  /* includes null terminator */
    return result;
}

/* Bug 3: Linked list reversal — original loses nodes */
typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node *make_node(int data) {
    Node *n = malloc(sizeof(Node));
    if (n) { n->data = data; n->next = NULL; }
    return n;
}

Node *reverse_list(Node *head) {
    /*
     * BUGGY VERSION:
     *   Node *prev = NULL, *curr = head;
     *   while (curr) {
     *       curr->next = prev;   // loses reference to next node!
     *       prev = curr;
     *       curr = curr->next;   // curr->next already changed!
     *   }
     *
     * FIX: Save next pointer before modifying curr->next.
     */
    Node *prev = NULL, *curr = head;
    while (curr) {
        Node *next = curr->next;  /* save next before overwriting */
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

void free_list(Node *head) {
    while (head) {
        Node *next = head->next;
        free(head);
        head = next;
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Bug Hunting ===\n");

    /* Test binary search */
    printf("\nBinary Search:\n");
    int sorted[] = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};
    int n = sizeof(sorted) / sizeof(sorted[0]);
    printf("  Find 23: index=%d (expected 5)\n", binary_search(sorted, n, 23));
    printf("  Find 2:  index=%d (expected 0)\n", binary_search(sorted, n, 2));
    printf("  Find 91: index=%d (expected 9)\n", binary_search(sorted, n, 91));
    printf("  Find 50: index=%d (expected -1)\n", binary_search(sorted, n, 50));

    /* Test safe_strcat */
    printf("\nSafe String Concatenation:\n");
    char *result = safe_strcat("Hello, ", "World!");
    printf("  \"%s\"\n", result);
    free(result);

    /* Test list reversal */
    printf("\nLinked List Reversal:\n");
    Node *list = make_node(1);
    list->next = make_node(2);
    list->next->next = make_node(3);
    list->next->next->next = make_node(4);

    printf("  Before: ");
    for (Node *p = list; p; p = p->next) printf("%d -> ", p->data);
    printf("NULL\n");

    list = reverse_list(list);
    printf("  After:  ");
    for (Node *p = list; p; p = p->next) printf("%d -> ", p->data);
    printf("NULL\n");

    free_list(list);
}

/* === Exercise 2: Profiling Analysis === */
/* Problem: Compare different algorithm implementations and measure
 *          performance to understand bottlenecks. */

/* Naive string search: O(n*m) */
int search_naive(const char *text, const char *pattern) {
    int n = (int)strlen(text);
    int m = (int)strlen(pattern);
    int count = 0;

    for (int i = 0; i <= n - m; i++) {
        int match = 1;
        for (int j = 0; j < m; j++) {
            if (text[i + j] != pattern[j]) {
                match = 0;
                break;
            }
        }
        if (match) count++;
    }
    return count;
}

/* Generate a test string for benchmarking */
char *generate_text(int size) {
    char *text = malloc(size + 1);
    if (!text) return NULL;
    for (int i = 0; i < size; i++) {
        text[i] = 'a' + (rand() % 4);  /* only a-d for higher match rate */
    }
    text[size] = '\0';
    return text;
}

/* Bubble sort for comparison */
void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

/* Comparison function for qsort */
int cmp_int(const void *a, const void *b) {
    return *(const int *)a - *(const int *)b;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Profiling Analysis ===\n");

    srand(42);

    /* String search benchmark */
    printf("\nString search (100K chars, pattern 'abab'):\n");
    char *text = generate_text(100000);
    if (text) {
        clock_t start = clock();
        int matches = search_naive(text, "abab");
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        printf("  Naive: %d matches in %.2f ms\n", matches, elapsed);
        free(text);
    }

    /* Sort benchmark */
    printf("\nSort comparison (10000 elements):\n");
    int n = 10000;
    int *arr1 = malloc(n * sizeof(int));
    int *arr2 = malloc(n * sizeof(int));
    if (arr1 && arr2) {
        for (int i = 0; i < n; i++) {
            arr1[i] = rand() % 100000;
            arr2[i] = arr1[i];  /* same data */
        }

        clock_t start = clock();
        bubble_sort(arr1, n);
        clock_t end = clock();
        printf("  Bubble sort: %.2f ms\n",
               (double)(end - start) / CLOCKS_PER_SEC * 1000.0);

        start = clock();
        qsort(arr2, n, sizeof(int), cmp_int);
        end = clock();
        printf("  qsort:       %.2f ms\n",
               (double)(end - start) / CLOCKS_PER_SEC * 1000.0);

        /* Verify both produce same result */
        int match = 1;
        for (int i = 0; i < n; i++) {
            if (arr1[i] != arr2[i]) { match = 0; break; }
        }
        printf("  Results match: %s\n", match ? "yes" : "no");
    }
    free(arr1);
    free(arr2);

    /*
     * Profiling workflow:
     * 1. Compile with -pg: gcc -pg -O2 -o app app.c
     * 2. Run: ./app (generates gmon.out)
     * 3. Analyze: gprof ./app gmon.out > analysis.txt
     * 4. Look for: % time, cumulative seconds, self seconds
     * 5. Focus optimization on the hottest functions
     *
     * Alternative tools:
     * - perf stat ./app          (Linux: CPU counters)
     * - perf record ./app        (Linux: sampling profiler)
     * - valgrind --tool=callgrind ./app  (call graph profiler)
     * - Instruments (macOS)
     */
}

/* === Exercise 3: Unit Test Writing === */
/* Problem: Write a minimal test framework and tests for utility functions. */

/* Minimal test framework */
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST %-40s ", #name); \
    name(); \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL (%s:%d: %d != %d)\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        return; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    if (strcmp((a), (b)) != 0) { \
        printf("FAIL (%s:%d: \"%s\" != \"%s\")\n", __FILE__, __LINE__, (a), (b)); \
        return; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL (%s:%d: condition false)\n", __FILE__, __LINE__); \
        return; \
    } \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

/* Functions under test */
int my_abs(int x) { return x < 0 ? -x : x; }
int my_max(int a, int b) { return a > b ? a : b; }
int my_clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Test cases */
void test_abs_positive(void) {
    ASSERT_EQ(my_abs(5), 5);
    ASSERT_EQ(my_abs(100), 100);
    PASS();
}

void test_abs_negative(void) {
    ASSERT_EQ(my_abs(-5), 5);
    ASSERT_EQ(my_abs(-100), 100);
    PASS();
}

void test_abs_zero(void) {
    ASSERT_EQ(my_abs(0), 0);
    PASS();
}

void test_max_basic(void) {
    ASSERT_EQ(my_max(3, 7), 7);
    ASSERT_EQ(my_max(10, 2), 10);
    PASS();
}

void test_max_equal(void) {
    ASSERT_EQ(my_max(5, 5), 5);
    PASS();
}

void test_max_negative(void) {
    ASSERT_EQ(my_max(-3, -7), -3);
    PASS();
}

void test_clamp_within(void) {
    ASSERT_EQ(my_clamp(5, 0, 10), 5);
    PASS();
}

void test_clamp_below(void) {
    ASSERT_EQ(my_clamp(-5, 0, 10), 0);
    PASS();
}

void test_clamp_above(void) {
    ASSERT_EQ(my_clamp(15, 0, 10), 10);
    PASS();
}

void test_binary_search_found(void) {
    int arr[] = {1, 3, 5, 7, 9};
    ASSERT_EQ(binary_search(arr, 5, 5), 2);
    ASSERT_EQ(binary_search(arr, 5, 1), 0);
    ASSERT_EQ(binary_search(arr, 5, 9), 4);
    PASS();
}

void test_binary_search_not_found(void) {
    int arr[] = {1, 3, 5, 7, 9};
    ASSERT_EQ(binary_search(arr, 5, 4), -1);
    ASSERT_EQ(binary_search(arr, 5, 0), -1);
    ASSERT_EQ(binary_search(arr, 5, 10), -1);
    PASS();
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Unit Test Writing ===\n\n");

    TEST(test_abs_positive);
    TEST(test_abs_negative);
    TEST(test_abs_zero);
    TEST(test_max_basic);
    TEST(test_max_equal);
    TEST(test_max_negative);
    TEST(test_clamp_within);
    TEST(test_clamp_below);
    TEST(test_clamp_above);
    TEST(test_binary_search_found);
    TEST(test_binary_search_not_found);

    printf("\n  Results: %d/%d passed\n", tests_passed, tests_run);

    /*
     * For real projects, use a proper test framework:
     * - Unity (https://github.com/ThrowTheSwitch/Unity) — simple, embedded-friendly
     * - Check (https://libcheck.github.io/check/) — supports forking
     * - CMocka — mock support, SAL-based
     *
     * Test best practices:
     * - One assertion per test when possible
     * - Test boundary conditions (0, -1, MAX, empty)
     * - Test both success and failure paths
     * - Use setup/teardown for resource management
     */
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
