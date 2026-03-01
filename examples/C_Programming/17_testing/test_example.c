/**
 * Unit Testing in C — Using assert.h
 *
 * Demonstrates:
 *   - Writing testable functions
 *   - Simple test framework with macros
 *   - Edge case testing
 *   - Test reporting
 *
 * This example uses only assert.h (no external dependencies).
 * For production code, consider Unity or CMocka.
 *
 * Build & Run:
 *   make test
 *   # or: gcc -Wall -Wextra -o test_example test_example.c && ./test_example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* ── Test Framework Macros ──────────────────────────────────────── */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-45s", #name); \
    name(); \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

#define ASSERT_EQ(a, b)  assert((a) == (b))
#define ASSERT_NEQ(a, b) assert((a) != (b))
#define ASSERT_STR_EQ(a, b) assert(strcmp((a), (b)) == 0)
#define ASSERT_NULL(p)   assert((p) == NULL)
#define ASSERT_NOT_NULL(p) assert((p) != NULL)
#define ASSERT_NEAR(a, b, tol) assert(fabs((a) - (b)) < (tol))

/* ── Module Under Test: String Utilities ────────────────────────── */

/**
 * Returns length of a string (like strlen, but handles NULL).
 */
size_t my_strlen(const char *s) {
    if (s == NULL) return 0;
    size_t len = 0;
    while (s[len] != '\0') len++;
    return len;
}

/**
 * Reverses a string in-place.
 */
void my_strrev(char *s) {
    if (s == NULL) return;
    size_t len = my_strlen(s);
    for (size_t i = 0; i < len / 2; i++) {
        char tmp = s[i];
        s[i] = s[len - 1 - i];
        s[len - 1 - i] = tmp;
    }
}

/**
 * Counts occurrences of character c in string s.
 */
int my_strcount(const char *s, char c) {
    if (s == NULL) return 0;
    int count = 0;
    while (*s) {
        if (*s == c) count++;
        s++;
    }
    return count;
}

/**
 * Trims leading and trailing whitespace (returns new allocation).
 * Caller must free the result.
 */
char *my_strtrim(const char *s) {
    if (s == NULL) return NULL;

    /* Skip leading whitespace */
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;

    size_t len = my_strlen(s);
    if (len == 0) {
        char *result = malloc(1);
        result[0] = '\0';
        return result;
    }

    /* Find end (excluding trailing whitespace) */
    const char *end = s + len - 1;
    while (end > s && (*end == ' ' || *end == '\t' || *end == '\n')) end--;

    size_t new_len = (size_t)(end - s + 1);
    char *result = malloc(new_len + 1);
    memcpy(result, s, new_len);
    result[new_len] = '\0';
    return result;
}

/* ── Module Under Test: Integer Math ────────────────────────────── */

int int_max(int a, int b) { return a > b ? a : b; }
int int_min(int a, int b) { return a < b ? a : b; }
int int_clamp(int val, int lo, int hi) {
    return int_max(lo, int_min(val, hi));
}
int gcd(int a, int b) {
    a = abs(a); b = abs(b);
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

/* ── Tests: my_strlen ───────────────────────────────────────────── */

void test_strlen_basic(void) {
    ASSERT_EQ(my_strlen("hello"), 5);
    ASSERT_EQ(my_strlen(""), 0);
    ASSERT_EQ(my_strlen("a"), 1);
}

void test_strlen_null(void) {
    ASSERT_EQ(my_strlen(NULL), 0);
}

void test_strlen_whitespace(void) {
    ASSERT_EQ(my_strlen("  "), 2);
    ASSERT_EQ(my_strlen("\t\n"), 2);
}

/* ── Tests: my_strrev ───────────────────────────────────────────── */

void test_strrev_basic(void) {
    char s[] = "hello";
    my_strrev(s);
    ASSERT_STR_EQ(s, "olleh");
}

void test_strrev_single(void) {
    char s[] = "x";
    my_strrev(s);
    ASSERT_STR_EQ(s, "x");
}

void test_strrev_empty(void) {
    char s[] = "";
    my_strrev(s);
    ASSERT_STR_EQ(s, "");
}

void test_strrev_palindrome(void) {
    char s[] = "racecar";
    my_strrev(s);
    ASSERT_STR_EQ(s, "racecar");
}

void test_strrev_null(void) {
    my_strrev(NULL);  /* Should not crash */
}

/* ── Tests: my_strcount ─────────────────────────────────────────── */

void test_strcount_basic(void) {
    ASSERT_EQ(my_strcount("hello", 'l'), 2);
    ASSERT_EQ(my_strcount("hello", 'h'), 1);
    ASSERT_EQ(my_strcount("hello", 'z'), 0);
}

void test_strcount_empty(void) {
    ASSERT_EQ(my_strcount("", 'a'), 0);
    ASSERT_EQ(my_strcount(NULL, 'a'), 0);
}

/* ── Tests: my_strtrim ──────────────────────────────────────────── */

void test_strtrim_basic(void) {
    char *r = my_strtrim("  hello  ");
    ASSERT_STR_EQ(r, "hello");
    free(r);
}

void test_strtrim_no_whitespace(void) {
    char *r = my_strtrim("hello");
    ASSERT_STR_EQ(r, "hello");
    free(r);
}

void test_strtrim_all_whitespace(void) {
    char *r = my_strtrim("   ");
    ASSERT_STR_EQ(r, "");
    free(r);
}

void test_strtrim_tabs_newlines(void) {
    char *r = my_strtrim("\t  hello world  \n");
    ASSERT_STR_EQ(r, "hello world");
    free(r);
}

void test_strtrim_null(void) {
    ASSERT_NULL(my_strtrim(NULL));
}

/* ── Tests: Integer Math ────────────────────────────────────────── */

void test_int_max(void) {
    ASSERT_EQ(int_max(3, 5), 5);
    ASSERT_EQ(int_max(-1, -5), -1);
    ASSERT_EQ(int_max(0, 0), 0);
}

void test_int_clamp(void) {
    ASSERT_EQ(int_clamp(5, 0, 10), 5);
    ASSERT_EQ(int_clamp(-5, 0, 10), 0);
    ASSERT_EQ(int_clamp(15, 0, 10), 10);
}

void test_gcd(void) {
    ASSERT_EQ(gcd(12, 8), 4);
    ASSERT_EQ(gcd(17, 13), 1);
    ASSERT_EQ(gcd(100, 25), 25);
    ASSERT_EQ(gcd(0, 5), 5);
    ASSERT_EQ(gcd(-12, 8), 4);
}

/* ── Main Test Runner ───────────────────────────────────────────── */

int main(void) {
    printf("========================================\n");
    printf("  Unit Tests — String Utilities & Math\n");
    printf("========================================\n\n");

    printf("my_strlen:\n");
    TEST(test_strlen_basic);
    TEST(test_strlen_null);
    TEST(test_strlen_whitespace);

    printf("\nmy_strrev:\n");
    TEST(test_strrev_basic);
    TEST(test_strrev_single);
    TEST(test_strrev_empty);
    TEST(test_strrev_palindrome);
    TEST(test_strrev_null);

    printf("\nmy_strcount:\n");
    TEST(test_strcount_basic);
    TEST(test_strcount_empty);

    printf("\nmy_strtrim:\n");
    TEST(test_strtrim_basic);
    TEST(test_strtrim_no_whitespace);
    TEST(test_strtrim_all_whitespace);
    TEST(test_strtrim_tabs_newlines);
    TEST(test_strtrim_null);

    printf("\nInteger Math:\n");
    TEST(test_int_max);
    TEST(test_int_clamp);
    TEST(test_gcd);

    printf("\n========================================\n");
    printf("  Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("========================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
