/*
 * preprocessor_demo.c — Macros, conditional compilation, include guards.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o preprocessor_demo preprocessor_demo.c
 * Run:     ./preprocessor_demo
 *
 * Try: gcc -DFEATURE_DEBUG=0 ... to disable debug output at compile time.
 */

#include <stdio.h>
#include "config.h"

/* Object-like macros */
#define MAX_BUFFER 1024
#define PI         3.14159265358979

/* Function-like macros */
#define MAX(a, b)     ((a) > (b) ? (a) : (b))
#define MIN(a, b)     ((a) < (b) ? (a) : (b))
#define SQUARE(x)     ((x) * (x))
#define ARRAY_LEN(a)  (sizeof(a) / sizeof((a)[0]))

/* Stringification and token pasting */
#define STRINGIFY(x)  #x
#define CONCAT(a, b)  a##b

/* Debug logging macro */
#if FEATURE_DEBUG
    #define DEBUG_LOG(fmt, ...) \
        fprintf(stderr, "[DEBUG %s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define DEBUG_LOG(fmt, ...)  /* no-op */
#endif

/* Conditional logging */
#if FEATURE_LOGGING
    #define LOG(msg)  printf("[LOG] %s\n", msg)
#else
    #define LOG(msg)
#endif

int main(void)
{
    /* Header info */
    printf("=== Include Header ===\n");
    printf("App: %s v%s\n", APP_NAME, APP_VERSION);
    printf("Platform: %s\n", PLATFORM);

    /* Object-like macros */
    printf("\n=== Object-like Macros ===\n");
    printf("MAX_BUFFER = %d\n", MAX_BUFFER);
    printf("PI = %.15f\n", PI);

    /* Function-like macros */
    printf("\n=== Function-like Macros ===\n");
    printf("MAX(3, 7)   = %d\n", MAX(3, 7));
    printf("MIN(3, 7)   = %d\n", MIN(3, 7));
    printf("SQUARE(5)   = %d\n", SQUARE(5));

    int arr[] = {10, 20, 30, 40};
    printf("ARRAY_LEN   = %zu\n", ARRAY_LEN(arr));

    /* Stringification */
    printf("\n=== Stringification ===\n");
    printf("STRINGIFY(hello) = \"%s\"\n", STRINGIFY(hello));

    /* Token pasting */
    printf("\n=== Token Pasting ===\n");
    int CONCAT(my, Var) = 42;  /* creates variable 'myVar' */
    printf("myVar = %d\n", myVar);

    /* Predefined macros */
    printf("\n=== Predefined Macros ===\n");
    printf("__FILE__    = %s\n", __FILE__);
    printf("__LINE__    = %d\n", __LINE__);
    printf("__DATE__    = %s\n", __DATE__);
    printf("__TIME__    = %s\n", __TIME__);
    printf("__STDC__    = %d\n", __STDC__);

    /* Conditional compilation */
    printf("\n=== Conditional Compilation ===\n");
    LOG("Logging is enabled");
    DEBUG_LOG("x = %d, name = %s", 42, "test");

    return 0;
}
