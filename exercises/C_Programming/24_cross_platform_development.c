/**
 * Exercises for Lesson 24: Cross-Platform Development in C
 * Topic: C_Programming
 *
 * Solutions to practice problems covering platform detection,
 * portable data types, endianness handling, cross-platform sleep,
 * file operations, and platform abstraction layers.
 *
 * Compile:
 *   Linux/macOS: gcc -std=c11 -Wall -o ex24 24_cross_platform_development.c -lpthread
 *   Windows:     cl /W4 24_cross_platform_development.c ws2_32.lib
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

/* ---- Platform detection ---- */
#if defined(_WIN32) || defined(_WIN64)
    #define PLAT_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
    #define PLAT_MACOS   1
#elif defined(__linux__)
    #define PLAT_LINUX   1
#endif

#if PLAT_WINDOWS
    #include <windows.h>
#else
    #include <unistd.h>
    #include <pthread.h>
#endif


/* ===========================================================================
 * Exercise 1: Platform Info Reporter
 *
 * Print OS, architecture, compiler, endianness, and key type sizes
 * using only preprocessor macros and <stdint.h>.
 * =========================================================================== */
void exercise_1(void) {
    printf("=== Exercise 1: Platform Info Reporter ===\n\n");

    /* OS */
    const char *os =
    #if PLAT_WINDOWS
        "Windows";
    #elif PLAT_MACOS
        "macOS";
    #elif PLAT_LINUX
        "Linux";
    #else
        "Unknown";
    #endif
    printf("Operating System: %s\n", os);

    /* Architecture */
    const char *arch =
    #if defined(__x86_64__) || defined(_M_X64)
        "x86_64 (64-bit)";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        "ARM64 (64-bit)";
    #elif defined(__i386__) || defined(_M_IX86)
        "x86 (32-bit)";
    #elif defined(__arm__)
        "ARM (32-bit)";
    #else
        "Unknown";
    #endif
    printf("Architecture:     %s\n", arch);

    /* Compiler */
    #if defined(_MSC_VER)
        printf("Compiler:         MSVC %d\n", _MSC_VER);
    #elif defined(__clang__)
        printf("Compiler:         Clang %d.%d.%d\n",
               __clang_major__, __clang_minor__, __clang_patchlevel__);
    #elif defined(__GNUC__)
        printf("Compiler:         GCC %d.%d.%d\n",
               __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #else
        printf("Compiler:         Unknown\n");
    #endif

    /* Endianness -- runtime check */
    uint16_t test = 0x0102;
    uint8_t *bytes = (uint8_t *)&test;
    printf("Byte Order:       %s-endian\n",
           (bytes[0] == 0x01) ? "big" : "little");

    /* Type sizes */
    printf("\nType Sizes:\n");
    printf("  sizeof(int):       %zu bytes\n", sizeof(int));
    printf("  sizeof(long):      %zu bytes\n", sizeof(long));
    printf("  sizeof(long long): %zu bytes\n", sizeof(long long));
    printf("  sizeof(void *):    %zu bytes\n", sizeof(void *));
    printf("  sizeof(size_t):    %zu bytes\n", sizeof(size_t));
    printf("\n");
}


/* ===========================================================================
 * Exercise 2: Portable sleep_ms with timing verification
 *
 * Implement sleep_ms using nanosleep (POSIX) or Sleep (Windows),
 * then measure accuracy to within ±50ms.
 * =========================================================================== */

static void portable_sleep_ms(unsigned int ms) {
#if PLAT_WINDOWS
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

void exercise_2(void) {
    printf("=== Exercise 2: Portable sleep_ms ===\n\n");

    const unsigned int target_ms = 500;

    /* Measure wall-clock time around the sleep */
    clock_t start = clock();
    portable_sleep_ms(target_ms);
    clock_t end = clock();

    /* clock() measures CPU time, not wall time, so it may show ~0 during sleep.
       For a better wall-clock measurement, we use platform-specific timers. */
#if PLAT_WINDOWS
    /* On Windows, we re-measure with GetTickCount for wall time */
    DWORD t0 = GetTickCount();
    portable_sleep_ms(target_ms);
    DWORD elapsed_wall = GetTickCount() - t0;
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    portable_sleep_ms(target_ms);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long elapsed_wall = (t1.tv_sec - t0.tv_sec) * 1000 +
                        (t1.tv_nsec - t0.tv_nsec) / 1000000;
#endif

    double cpu_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("Target sleep:     %u ms\n", target_ms);
    printf("CPU time:         %.1f ms (low because CPU is idle during sleep)\n", cpu_ms);
    printf("Wall-clock time:  %ld ms\n", (long)elapsed_wall);

    long diff = (long)elapsed_wall - (long)target_ms;
    bool within_tolerance = (diff >= -50 && diff <= 50);
    printf("Deviation:        %+ld ms (%s)\n", diff,
           within_tolerance ? "PASS: within ±50ms" : "FAIL: outside ±50ms");
    printf("\n");
}


/* ===========================================================================
 * Exercise 3: Cross-Platform File Copy (standard C version)
 *
 * Copy a file using only fopen/fread/fwrite (works on any platform).
 * =========================================================================== */

int file_copy_portable(const char *src, const char *dst) {
    FILE *in  = fopen(src, "rb");
    if (!in) return -1;

    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }

    /* 64 KB buffer -- balances memory usage vs syscall overhead */
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, n, out) != n) {
            fclose(in);
            fclose(out);
            return -1;
        }
    }

    int err = ferror(in);
    fclose(in);
    fclose(out);
    return err ? -1 : 0;
}

void exercise_3(void) {
    printf("=== Exercise 3: Cross-Platform File Copy ===\n\n");

    /* Create a test file */
    const char *src_path = "/tmp/xplat_copy_src.txt";
    const char *dst_path = "/tmp/xplat_copy_dst.txt";

    FILE *f = fopen(src_path, "w");
    if (!f) {
        printf("Cannot create test file at %s\n\n", src_path);
        return;
    }
    /* Write some recognizable content */
    for (int i = 0; i < 1000; i++) {
        fprintf(f, "Line %04d: Cross-platform file copy test data\n", i);
    }
    fclose(f);

    /* Copy */
    int result = file_copy_portable(src_path, dst_path);
    printf("Copy result: %s\n", result == 0 ? "SUCCESS" : "FAILED");

    /* Verify */
    FILE *s = fopen(src_path, "rb");
    FILE *d = fopen(dst_path, "rb");
    if (s && d) {
        fseek(s, 0, SEEK_END);
        fseek(d, 0, SEEK_END);
        long src_size = ftell(s);
        long dst_size = ftell(d);
        printf("Source size: %ld bytes\n", src_size);
        printf("Dest size:   %ld bytes\n", dst_size);
        printf("Size match:  %s\n", src_size == dst_size ? "YES" : "NO");
    }
    if (s) fclose(s);
    if (d) fclose(d);

    /* Cleanup */
    remove(src_path);
    remove(dst_path);
    printf("\n");
}


/* ===========================================================================
 * Exercise 4: Minimal PAL (mutex + thread + sleep)
 *
 * 4 threads each increment a shared counter 1,000,000 times under mutex.
 * Final count must be exactly 4,000,000.
 * =========================================================================== */

/* --- Mutex abstraction --- */
#if PLAT_WINDOWS
typedef CRITICAL_SECTION pal_mutex_t;
#else
typedef pthread_mutex_t  pal_mutex_t;
#endif

static int pal_mutex_init(pal_mutex_t *m) {
#if PLAT_WINDOWS
    InitializeCriticalSection(m);
    return 0;
#else
    return pthread_mutex_init(m, NULL);
#endif
}

static int pal_mutex_lock(pal_mutex_t *m) {
#if PLAT_WINDOWS
    EnterCriticalSection(m); return 0;
#else
    return pthread_mutex_lock(m);
#endif
}

static int pal_mutex_unlock(pal_mutex_t *m) {
#if PLAT_WINDOWS
    LeaveCriticalSection(m); return 0;
#else
    return pthread_mutex_unlock(m);
#endif
}

static void pal_mutex_destroy(pal_mutex_t *m) {
#if PLAT_WINDOWS
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}

/* --- Shared state --- */
static pal_mutex_t g_mutex;
static int64_t     g_counter = 0;
#define INCREMENTS_PER_THREAD 1000000

/* --- Thread function --- */
#if PLAT_WINDOWS
static DWORD WINAPI worker_fn(LPVOID arg) {
    (void)arg;
    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {
        pal_mutex_lock(&g_mutex);
        g_counter++;
        pal_mutex_unlock(&g_mutex);
    }
    return 0;
}
#else
static void *worker_fn(void *arg) {
    (void)arg;
    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {
        pal_mutex_lock(&g_mutex);
        g_counter++;
        pal_mutex_unlock(&g_mutex);
    }
    return NULL;
}
#endif

void exercise_4(void) {
    printf("=== Exercise 4: PAL Mutex + Thread Test ===\n\n");

    const int num_threads = 4;
    g_counter = 0;
    pal_mutex_init(&g_mutex);

#if PLAT_WINDOWS
    HANDLE threads[4];
    for (int i = 0; i < num_threads; i++) {
        threads[i] = CreateThread(NULL, 0, worker_fn, NULL, 0, NULL);
    }
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);
    for (int i = 0; i < num_threads; i++) {
        CloseHandle(threads[i]);
    }
#else
    pthread_t threads[4];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_fn, NULL);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
#endif

    pal_mutex_destroy(&g_mutex);

    int64_t expected = (int64_t)num_threads * INCREMENTS_PER_THREAD;
    printf("Threads:   %d\n", num_threads);
    printf("Per-thread: %d increments\n", INCREMENTS_PER_THREAD);
    printf("Expected:  %" PRId64 "\n", expected);
    printf("Actual:    %" PRId64 "\n", g_counter);
    printf("Result:    %s\n",
           g_counter == expected ? "PASS (mutex works correctly)" : "FAIL");
    printf("\n");
}


/* ===========================================================================
 * Main
 * =========================================================================== */
int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();

    printf("=== All exercises completed ===\n");
    return 0;
}
