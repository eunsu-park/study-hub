/**
 * cross_platform_demo.c
 *
 * Demonstrates cross-platform C programming techniques:
 * - Platform detection via preprocessor macros
 * - Portable data types (stdint.h, inttypes.h)
 * - Endianness detection and byte swapping
 * - Cross-platform sleep and timing
 * - Platform abstraction for directory listing
 *
 * Compile:
 *   Linux/macOS: gcc -std=c11 -Wall -o xplat cross_platform_demo.c -lpthread
 *   Windows:     cl /W4 cross_platform_demo.c ws2_32.lib
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

/* ===================================================================
 * 1. Platform Detection
 * =================================================================== */

#if defined(_WIN32) || defined(_WIN64)
    #define PLAT_WINDOWS 1
    #define PLAT_NAME    "Windows"
    #include <windows.h>
    #include <direct.h>   /* _getcwd */
#elif defined(__APPLE__) && defined(__MACH__)
    #define PLAT_MACOS   1
    #define PLAT_NAME    "macOS"
    #include <unistd.h>
    #include <dirent.h>
    #include <sys/stat.h>
#elif defined(__linux__)
    #define PLAT_LINUX   1
    #define PLAT_NAME    "Linux"
    #include <unistd.h>
    #include <dirent.h>
    #include <sys/stat.h>
#else
    #define PLAT_NAME    "Unknown"
#endif

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
    #define ARCH_NAME "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_NAME "ARM64"
#elif defined(__i386__) || defined(_M_IX86)
    #define ARCH_NAME "x86"
#elif defined(__arm__) || defined(_M_ARM)
    #define ARCH_NAME "ARM"
#else
    #define ARCH_NAME "Unknown"
#endif

/* Compiler detection */
#if defined(_MSC_VER)
    #define COMP_NAME "MSVC"
    #define COMP_VER  _MSC_VER
#elif defined(__clang__)
    #define COMP_NAME "Clang"
    #define COMP_VER  (__clang_major__ * 100 + __clang_minor__)
#elif defined(__GNUC__)
    #define COMP_NAME "GCC"
    #define COMP_VER  (__GNUC__ * 100 + __GNUC_MINOR__)
#else
    #define COMP_NAME "Unknown"
    #define COMP_VER  0
#endif

/* ===================================================================
 * 2. Portable Sleep
 * =================================================================== */

/**
 * Sleep for the specified number of milliseconds.
 * Uses Sleep() on Windows, nanosleep() on POSIX.
 */
void sleep_ms(unsigned int ms) {
#if PLAT_WINDOWS
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

/* ===================================================================
 * 3. Endianness Detection and Byte Swap
 * =================================================================== */

bool is_little_endian(void) {
    uint16_t val = 1;
    uint8_t *bytes = (uint8_t *)&val;
    return bytes[0] == 1;
}

uint16_t swap16(uint16_t v) {
    return (v >> 8) | (v << 8);
}

uint32_t swap32(uint32_t v) {
    return ((v >> 24) & 0x000000FFu) |
           ((v >>  8) & 0x0000FF00u) |
           ((v <<  8) & 0x00FF0000u) |
           ((v << 24) & 0xFF000000u);
}

/* Convert host byte order to big-endian (network byte order) */
uint32_t to_big_endian_32(uint32_t host_val) {
    if (is_little_endian()) {
        return swap32(host_val);
    }
    return host_val;
}

/* ===================================================================
 * 4. Cross-Platform Directory Listing
 * =================================================================== */

void list_directory(const char *path) {
    printf("\nDirectory listing for: %s\n", path);
    printf("%-30s %s\n", "Name", "Type");
    printf("%-30s %s\n", "----", "----");

#if PLAT_WINDOWS
    WIN32_FIND_DATAA fdata;
    char pattern[260];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);

    HANDLE h = FindFirstFileA(pattern, &fdata);
    if (h == INVALID_HANDLE_VALUE) {
        printf("  (cannot open directory)\n");
        return;
    }
    do {
        const char *type = (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                           ? "DIR" : "FILE";
        printf("  %-28s %s\n", fdata.cFileName, type);
    } while (FindNextFileA(h, &fdata));
    FindClose(h);

#else
    DIR *d = opendir(path);
    if (!d) {
        printf("  (cannot open directory)\n");
        return;
    }
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        /* Skip hidden . and .. entries */
        if (ent->d_name[0] == '.') continue;

        const char *type;
        /* d_type is a BSD/Linux extension; fall back to stat if unavailable */
    #if defined(DT_DIR)
        type = (ent->d_type == DT_DIR) ? "DIR" : "FILE";
    #else
        struct stat st;
        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        stat(full, &st);
        type = S_ISDIR(st.st_mode) ? "DIR" : "FILE";
    #endif
        printf("  %-28s %s\n", ent->d_name, type);
    }
    closedir(d);
#endif
}

/* ===================================================================
 * 5. Portable High-Resolution Timer
 * =================================================================== */

/**
 * Return current time in milliseconds (monotonic clock).
 * Used for benchmarking, not wall-clock time.
 */
double get_time_ms(void) {
#if PLAT_WINDOWS
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#endif
}

/* ===================================================================
 * 6. Main: Platform Info Report
 * =================================================================== */

int main(void) {
    printf("=== Cross-Platform C Demo ===\n\n");

    /* Platform info */
    printf("Platform Info:\n");
    printf("  OS:           %s\n", PLAT_NAME);
    printf("  Architecture: %s\n", ARCH_NAME);
    printf("  Compiler:     %s (version %d)\n", COMP_NAME, COMP_VER);
    printf("  C Standard:   %ld\n", (long)__STDC_VERSION__);
    printf("\n");

    /* Data type sizes -- these vary across platforms */
    printf("Data Type Sizes:\n");
    printf("  sizeof(char):      %zu bytes\n", sizeof(char));
    printf("  sizeof(short):     %zu bytes\n", sizeof(short));
    printf("  sizeof(int):       %zu bytes\n", sizeof(int));
    printf("  sizeof(long):      %zu bytes\n", sizeof(long));
    printf("  sizeof(long long): %zu bytes\n", sizeof(long long));
    printf("  sizeof(float):     %zu bytes\n", sizeof(float));
    printf("  sizeof(double):    %zu bytes\n", sizeof(double));
    printf("  sizeof(void *):    %zu bytes\n", sizeof(void *));
    printf("  sizeof(size_t):    %zu bytes\n", sizeof(size_t));
    printf("\n");

    /* Fixed-width types are portable */
    printf("Fixed-Width Types (always the same):\n");
    printf("  sizeof(uint8_t):   %zu\n", sizeof(uint8_t));
    printf("  sizeof(uint16_t):  %zu\n", sizeof(uint16_t));
    printf("  sizeof(uint32_t):  %zu\n", sizeof(uint32_t));
    printf("  sizeof(uint64_t):  %zu\n", sizeof(uint64_t));
    printf("\n");

    /* Endianness */
    printf("Byte Order:\n");
    printf("  System is %s-endian\n",
           is_little_endian() ? "little" : "big");
    uint32_t val = 0xDEADBEEF;
    uint32_t swapped = swap32(val);
    printf("  Original:  0x%08" PRIX32 "\n", val);
    printf("  Swapped:   0x%08" PRIX32 "\n", swapped);
    printf("  Network:   0x%08" PRIX32 " (big-endian)\n",
           to_big_endian_32(val));
    printf("\n");

    /* Sleep test with timing */
    printf("Sleep Accuracy Test:\n");
    double start = get_time_ms();
    sleep_ms(100);
    double elapsed = get_time_ms() - start;
    printf("  Requested: 100 ms\n");
    printf("  Actual:    %.1f ms\n", elapsed);
    printf("\n");

    /* Directory listing */
    list_directory(".");

    return 0;
}
