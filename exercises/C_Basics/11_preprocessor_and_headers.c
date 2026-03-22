/*
 * Exercises for Lesson 11: Preprocessor and Headers
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex11 11_preprocessor_and_headers.c
 * Debug:   gcc -Wall -Wextra -std=c11 -DDEBUG_MODE -o ex11 11_preprocessor_and_headers.c
 */
#include <stdio.h>
#include <string.h>

/* === Exercise 1: Debug Macro === */
/* Problem: Create debug-print macros that include file, line, and function. */

#ifdef DEBUG_MODE
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[DEBUG %s:%d %s] " fmt "\n", \
                __FILE__, __LINE__, __func__, ##__VA_ARGS__)
    #define DEBUG_ENTER() \
        fprintf(stderr, "[DEBUG %s:%d] --> %s()\n", \
                __FILE__, __LINE__, __func__)
    #define DEBUG_EXIT() \
        fprintf(stderr, "[DEBUG %s:%d] <-- %s()\n", \
                __FILE__, __LINE__, __func__)
#else
    #define DEBUG_PRINT(fmt, ...) ((void)0)
    #define DEBUG_ENTER()         ((void)0)
    #define DEBUG_EXIT()          ((void)0)
#endif

int compute_factorial(int n) {
    DEBUG_ENTER();
    DEBUG_PRINT("n = %d", n);

    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
        DEBUG_PRINT("i=%d, result=%d", i, result);
    }

    DEBUG_EXIT();
    return result;
}

void exercise_1(void) {
    printf("=== Exercise 1: Debug Macro ===\n");

#ifdef DEBUG_MODE
    printf("DEBUG_MODE is ON — debug messages go to stderr\n");
#else
    printf("DEBUG_MODE is OFF — recompile with -DDEBUG_MODE to enable\n");
#endif

    int r = compute_factorial(5);
    printf("factorial(5) = %d\n", r);

    /*
     * When compiled with -DDEBUG_MODE, debug output appears on stderr.
     * When compiled normally, all DEBUG macros expand to nothing ((void)0),
     * producing zero runtime overhead.
     *
     * The ## before __VA_ARGS__ is a GCC extension that removes the
     * trailing comma when no variadic args are provided.
     */
}

/* === Exercise 2: MIN/MAX and Utility Macros === */
/* Problem: Implement type-generic MIN, MAX, CLAMP, and SWAP macros. */

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN(MAX((x), (lo)), (hi)))
#define SWAP(a, b) do { \
    __typeof__(a) _tmp = (a); \
    (a) = (b); \
    (b) = _tmp; \
} while (0)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define STRINGIFY(x) #x
#define CONCAT(a, b) a ## b

void exercise_2(void) {
    printf("\n=== Exercise 2: MIN/MAX and Utility Macros ===\n");

    /* MIN / MAX */
    printf("MIN(3, 7) = %d\n", MIN(3, 7));
    printf("MAX(3, 7) = %d\n", MAX(3, 7));
    printf("MIN(3.14, 2.72) = %.2f\n", MIN(3.14, 2.72));

    /* CLAMP */
    printf("CLAMP(15, 0, 10) = %d\n", CLAMP(15, 0, 10));
    printf("CLAMP(-5, 0, 10) = %d\n", CLAMP(-5, 0, 10));
    printf("CLAMP(5, 0, 10) = %d\n", CLAMP(5, 0, 10));

    /* SWAP */
    int x = 10, y = 20;
    printf("Before SWAP: x=%d, y=%d\n", x, y);
    SWAP(x, y);
    printf("After SWAP:  x=%d, y=%d\n", x, y);

    /* ARRAY_SIZE */
    int arr[] = {1, 2, 3, 4, 5};
    printf("ARRAY_SIZE(arr) = %zu\n", ARRAY_SIZE(arr));

    /* STRINGIFY */
    printf("STRINGIFY(Hello) = \"%s\"\n", STRINGIFY(Hello));

    /*
     * Macro pitfalls to remember:
     * - Always parenthesize macro parameters: (a) not a
     * - MIN/MAX evaluate args twice — avoid side effects like MIN(i++, j++)
     * - Use do { } while(0) for multi-statement macros
     * - __typeof__ is a GCC/Clang extension, not standard C
     */
}

/* === Exercise 3: Conditional Compilation === */
/* Problem: Use preprocessor directives for platform-aware and feature-toggle code. */

/* Platform detection */
#if defined(_WIN32) || defined(_WIN64)
    #define PLATFORM "Windows"
    #define PATH_SEP '\\'
#elif defined(__APPLE__)
    #define PLATFORM "macOS"
    #define PATH_SEP '/'
#elif defined(__linux__)
    #define PLATFORM "Linux"
    #define PATH_SEP '/'
#else
    #define PLATFORM "Unknown"
    #define PATH_SEP '/'
#endif

/* Feature toggles */
#ifndef FEATURE_LOGGING
    #define FEATURE_LOGGING 1
#endif

#ifndef FEATURE_METRICS
    #define FEATURE_METRICS 0
#endif

#define LOG_IF_ENABLED(msg) do { \
    if (FEATURE_LOGGING) printf("[LOG] %s\n", msg); \
} while (0)

/* Compile-time assertions (C11 _Static_assert) */
_Static_assert(sizeof(int) >= 4, "int must be at least 32 bits");
_Static_assert(sizeof(void *) == 4 || sizeof(void *) == 8,
               "only 32-bit and 64-bit platforms supported");

void exercise_3(void) {
    printf("\n=== Exercise 3: Conditional Compilation ===\n");

    printf("Platform: %s\n", PLATFORM);
    printf("Path separator: '%c'\n", PATH_SEP);
    printf("Compiler: ");
#if defined(__clang__)
    printf("Clang %d.%d.%d\n", __clang_major__, __clang_minor__,
           __clang_patchlevel__);
#elif defined(__GNUC__)
    printf("GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("Unknown\n");
#endif

    printf("C standard: ");
#if __STDC_VERSION__ >= 201710L
    printf("C18\n");
#elif __STDC_VERSION__ >= 201112L
    printf("C11\n");
#elif __STDC_VERSION__ >= 199901L
    printf("C99\n");
#else
    printf("C89/C90\n");
#endif

    printf("FEATURE_LOGGING: %s\n", FEATURE_LOGGING ? "enabled" : "disabled");
    printf("FEATURE_METRICS: %s\n", FEATURE_METRICS ? "enabled" : "disabled");

    LOG_IF_ENABLED("Application started");
    LOG_IF_ENABLED("Processing request");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
