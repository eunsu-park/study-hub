# Cross-Platform Development in C

**Previous**: [Debugging and Profiling](./15_Debugging_and_Profiling.md) | **Next**: [Project: Snake Game](./17_Project_Snake_Game.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify platform-specific differences in data types, headers, APIs, and toolchains
2. Use preprocessor macros and conditional compilation to isolate platform-dependent code
3. Design a platform abstraction layer (PAL) that hides OS differences behind a uniform API
4. Configure CMake to build cross-platform C projects for Windows, Linux, and macOS
5. Apply portable coding practices (fixed-width integers, standard library usage, endianness handling)

---

Virtually every non-trivial C program touches something that varies by operating system: file paths use `/` on Unix and `\` on Windows; threads come from pthreads or the Win32 API; sockets differ in initialization and error codes. Writing portable C means isolating these differences so that business logic remains untouched when a new platform arrives. This lesson shows you how, starting from simple `#ifdef` guards and ending with a full CMake-based build that compiles on all three major desktops.

---

## Table of Contents

1. [Why Cross-Platform C Is Hard](#1-why-cross-platform-c-is-hard)
2. [Platform Detection Macros](#2-platform-detection-macros)
3. [Conditional Compilation](#3-conditional-compilation)
4. [Portable Data Types](#4-portable-data-types)
5. [Byte Order and Endianness](#5-byte-order-and-endianness)
6. [Platform Abstraction Layer](#6-platform-abstraction-layer)
7. [Cross-Platform File and Path Handling](#7-cross-platform-file-and-path-handling)
8. [Cross-Platform Networking](#8-cross-platform-networking)
9. [Cross-Platform Threading](#9-cross-platform-threading)
10. [Building with CMake](#10-building-with-cmake)
11. [Practice Problems](#11-practice-problems)

---

## 1. Why Cross-Platform C Is Hard

C is standardized (C11, C17, C23), but the standard only covers the *language* and a minimal *standard library*. Everything outside that -- threads (before C11), sockets, file-system traversal, dynamic loading, GUI -- is platform-specific.

### Key Difference Areas

| Area | Linux / macOS | Windows |
|------|--------------|---------|
| Path separator | `/` | `\` (also accepts `/`) |
| Line endings | `\n` (LF) | `\r\n` (CRLF) |
| Dynamic libraries | `.so` / `.dylib` | `.dll` |
| Thread API | `pthread` | Win32 Threads / `_beginthreadex` |
| Socket init | none needed | `WSAStartup()` required |
| Directory listing | `opendir/readdir` | `FindFirstFile/FindNextFile` |
| Shared memory | `mmap`, `shm_open` | `CreateFileMapping` |
| Compiler | GCC, Clang | MSVC, Clang, MinGW-GCC |

---

## 2. Platform Detection Macros

```c
/* platform_detect.h */
#ifndef PLATFORM_DETECT_H
#define PLATFORM_DETECT_H

/* --- Operating System --- */
#if defined(_WIN32) || defined(_WIN64)
    #define PLAT_WINDOWS  1
#elif defined(__APPLE__) && defined(__MACH__)
    #define PLAT_MACOS    1
#elif defined(__linux__)
    #define PLAT_LINUX    1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
    #define PLAT_BSD      1
#else
    #error "Unsupported platform"
#endif

/* --- Compiler --- */
#if defined(_MSC_VER)
    #define COMP_MSVC     1
#elif defined(__clang__)
    #define COMP_CLANG    1
#elif defined(__GNUC__)
    #define COMP_GCC      1
#endif

/* --- Architecture --- */
#if defined(__x86_64__) || defined(_M_X64)
    #define ARCH_X64      1
#elif defined(__i386__) || defined(_M_IX86)
    #define ARCH_X86      1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_ARM64    1
#elif defined(__arm__) || defined(_M_ARM)
    #define ARCH_ARM      1
#endif

#endif /* PLATFORM_DETECT_H */
```

---

## 3. Conditional Compilation

### Simple #ifdef Guards

```c
#include "platform_detect.h"

void sleep_ms(int ms) {
#if PLAT_WINDOWS
    Sleep(ms);                    /* Windows: <windows.h> */
#else
    struct timespec ts;           /* POSIX: <time.h> */
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}
```

### Source-File Isolation

For larger blocks, put each implementation in a separate file and let the build system choose:

```
src/
├── sleep.h           /* Public API: void sleep_ms(int ms); */
├── sleep_posix.c     /* POSIX implementation */
└── sleep_win32.c     /* Windows implementation */
```

In CMake:

```cmake
if(WIN32)
    target_sources(myapp PRIVATE src/sleep_win32.c)
else()
    target_sources(myapp PRIVATE src/sleep_posix.c)
endif()
```

---

## 4. Portable Data Types

### Fixed-Width Integers

```c
#include <stdint.h>

uint8_t   byte_val;      /* Exactly 8 bits  */
int16_t   short_val;     /* Exactly 16 bits */
uint32_t  crc;           /* Exactly 32 bits */
int64_t   timestamp_us;  /* Exactly 64 bits */

/* For printf, use <inttypes.h> format macros */
#include <inttypes.h>
printf("CRC: %" PRIu32 "\n", crc);
printf("Time: %" PRId64 " us\n", timestamp_us);
```

### size_t and ptrdiff_t

- `size_t`: unsigned, for array sizes and `sizeof` results
- `ptrdiff_t`: signed, for pointer arithmetic results

```c
for (size_t i = 0; i < count; i++) {
    process(arr[i]);
}
printf("Count: %zu\n", count);
```

---

## 5. Byte Order and Endianness

### Detecting Endianness

```c
int is_little_endian(void) {
    uint16_t val = 1;
    uint8_t *bytes = (uint8_t *)&val;
    return bytes[0] == 1;
}
```

### Portable Byte Swapping

```c
static inline uint16_t swap16(uint16_t v) {
    return (v >> 8) | (v << 8);
}

static inline uint32_t swap32(uint32_t v) {
    return ((v >> 24) & 0x000000FF) |
           ((v >>  8) & 0x0000FF00) |
           ((v <<  8) & 0x00FF0000) |
           ((v << 24) & 0xFF000000);
}
```

### Struct Packing

```c
/* GCC / Clang */
typedef struct __attribute__((packed)) {
    uint8_t  type;
    uint16_t length;
    uint32_t sequence;
} PacketHeader;

/* MSVC */
#pragma pack(push, 1)
typedef struct {
    uint8_t  type;
    uint16_t length;
    uint32_t sequence;
} PacketHeader;
#pragma pack(pop)
```

---

## 6. Platform Abstraction Layer

A PAL defines a uniform API; each platform gets its own implementation.

### Thread Abstraction Example

```c
/* pal_thread.h */
#ifndef PAL_THREAD_H
#define PAL_THREAD_H

typedef struct pal_thread pal_thread_t;
typedef void *(*pal_thread_fn)(void *arg);

int  pal_thread_create(pal_thread_t **thread, pal_thread_fn fn, void *arg);
int  pal_thread_join(pal_thread_t *thread, void **result);
void pal_thread_destroy(pal_thread_t *thread);

#endif
```

```c
/* thread_posix.c */
#include "pal_thread.h"
#include <pthread.h>
#include <stdlib.h>

struct pal_thread {
    pthread_t handle;
};

int pal_thread_create(pal_thread_t **thread, pal_thread_fn fn, void *arg) {
    *thread = malloc(sizeof(pal_thread_t));
    if (!*thread) return -1;
    if (pthread_create(&(*thread)->handle, NULL, fn, arg) != 0) {
        free(*thread);
        *thread = NULL;
        return -1;
    }
    return 0;
}

int pal_thread_join(pal_thread_t *thread, void **result) {
    return pthread_join(thread->handle, result) == 0 ? 0 : -1;
}

void pal_thread_destroy(pal_thread_t *thread) {
    free(thread);
}
```

```c
/* thread_win32.c */
#include "pal_thread.h"
#include <windows.h>
#include <stdlib.h>

struct pal_thread {
    HANDLE handle;
    pal_thread_fn fn;
    void *arg;
    void *result;
};

static DWORD WINAPI thread_wrapper(LPVOID param) {
    pal_thread_t *t = (pal_thread_t *)param;
    t->result = t->fn(t->arg);
    return 0;
}

int pal_thread_create(pal_thread_t **thread, pal_thread_fn fn, void *arg) {
    *thread = malloc(sizeof(pal_thread_t));
    if (!*thread) return -1;
    (*thread)->fn = fn;
    (*thread)->arg = arg;
    (*thread)->result = NULL;
    (*thread)->handle = CreateThread(NULL, 0, thread_wrapper, *thread, 0, NULL);
    if (!(*thread)->handle) {
        free(*thread);
        *thread = NULL;
        return -1;
    }
    return 0;
}

int pal_thread_join(pal_thread_t *thread, void **result) {
    if (WaitForSingleObject(thread->handle, INFINITE) != WAIT_OBJECT_0)
        return -1;
    CloseHandle(thread->handle);
    if (result) *result = thread->result;
    return 0;
}

void pal_thread_destroy(pal_thread_t *thread) {
    free(thread);
}
```

---

## 7. Cross-Platform File and Path Handling

### Path Separator

```c
#if PLAT_WINDOWS
    #define PATH_SEP '\\'
#else
    #define PATH_SEP '/'
#endif
```

### Directory Listing

```c
#if PLAT_WINDOWS
#include <windows.h>
void list_dir(const char *path) {
    WIN32_FIND_DATAA fdata;
    char pattern[MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    HANDLE h = FindFirstFileA(pattern, &fdata);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        printf("%s\n", fdata.cFileName);
    } while (FindNextFileA(h, &fdata));
    FindClose(h);
}
#else
#include <dirent.h>
void list_dir(const char *path) {
    DIR *d = opendir(path);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        printf("%s\n", ent->d_name);
    }
    closedir(d);
}
#endif
```

---

## 8. Cross-Platform Networking

Windows requires `WSAStartup` before any socket call:

```c
#if PLAT_WINDOWS
    #include <winsock2.h>
    #include <ws2tcpip.h>
    typedef SOCKET sock_t;
    #define SOCK_INVALID INVALID_SOCKET
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    typedef int sock_t;
    #define SOCK_INVALID (-1)
#endif

int net_init(void) {
#if PLAT_WINDOWS
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
#else
    return 0;
#endif
}

void net_cleanup(void) {
#if PLAT_WINDOWS
    WSACleanup();
#endif
}

int net_close(sock_t s) {
#if PLAT_WINDOWS
    return closesocket(s);
#else
    return close(s);
#endif
}
```

---

## 9. Cross-Platform Threading

### Mutex Abstraction

```c
/* pal_mutex.h */
#include "platform_detect.h"

#if PLAT_WINDOWS
    #include <windows.h>
    typedef CRITICAL_SECTION pal_mutex_t;
#else
    #include <pthread.h>
    typedef pthread_mutex_t pal_mutex_t;
#endif

static inline int pal_mutex_init(pal_mutex_t *m) {
#if PLAT_WINDOWS
    InitializeCriticalSection(m);
    return 0;
#else
    return pthread_mutex_init(m, NULL);
#endif
}

static inline int pal_mutex_lock(pal_mutex_t *m) {
#if PLAT_WINDOWS
    EnterCriticalSection(m);
    return 0;
#else
    return pthread_mutex_lock(m);
#endif
}

static inline int pal_mutex_unlock(pal_mutex_t *m) {
#if PLAT_WINDOWS
    LeaveCriticalSection(m);
    return 0;
#else
    return pthread_mutex_unlock(m);
#endif
}

static inline void pal_mutex_destroy(pal_mutex_t *m) {
#if PLAT_WINDOWS
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}
```

---

## 10. Building with CMake

### Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(cross_demo C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common sources
set(SOURCES src/main.c src/app.c)

# Platform-specific sources
if(WIN32)
    list(APPEND SOURCES src/pal/thread_win32.c src/pal/net_win32.c)
else()
    list(APPEND SOURCES src/pal/thread_posix.c src/pal/net_posix.c)
endif()

add_executable(cross_demo ${SOURCES})
target_include_directories(cross_demo PRIVATE include)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(cross_demo ws2_32)
else()
    target_link_libraries(cross_demo pthread m)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(cross_demo PRIVATE /W4 /WX)
else()
    target_compile_options(cross_demo PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
```

### Building

```bash
# Linux / macOS
mkdir build && cd build
cmake ..
cmake --build .

# Windows (Visual Studio)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Feature Detection

```cmake
include(CheckIncludeFile)
include(CheckFunctionExists)

check_include_file("threads.h" HAVE_C11_THREADS)
check_function_exists(epoll_create1 HAVE_EPOLL)
check_function_exists(kqueue HAVE_KQUEUE)

configure_file(config.h.in config.h)
```

```c
/* config.h.in */
#cmakedefine HAVE_C11_THREADS
#cmakedefine HAVE_EPOLL
#cmakedefine HAVE_KQUEUE
```

---

## 11. Practice Problems

### Problem 1: Platform Info Reporter

Write a program that prints:
- Operating system name and architecture
- Compiler name and version
- Whether the system is little-endian or big-endian
- `sizeof(int)`, `sizeof(long)`, `sizeof(void *)`

### Problem 2: Portable sleep_ms

Implement `void sleep_ms(unsigned int ms)` that works on both POSIX (use `nanosleep`) and Windows (use `Sleep`). Write a test that sleeps for 500ms and measures elapsed time with `clock()` to verify accuracy within +/-50ms.

### Problem 3: Cross-Platform File Copy

Write `int file_copy(const char *src, const char *dst)` using only standard C (`fopen/fread/fwrite`). Then write a second version using platform-specific APIs (`sendfile` on Linux, `copyfile` on macOS, `CopyFileA` on Windows) for better performance.

### Problem 4: Platform Abstraction Layer

Design and implement a minimal PAL with:
- `pal_mutex_t`: init, lock, unlock, destroy
- `pal_thread_t`: create, join, destroy
- `pal_sleep_ms(unsigned int ms)`

Write a test that spawns 4 threads incrementing a shared counter 1,000,000 times under mutex protection. The final count must equal 4,000,000 regardless of platform.

### Problem 5: CMake Feature Detection

Create a CMakeLists.txt that:
1. Detects whether `<threads.h>` is available
2. If yes, builds with C11 threads
3. If no, falls back to pthreads (POSIX) or Win32 threads
4. Generates a `config.h` with `#define HAS_C11_THREADS` accordingly

---

## Next Steps

You now have the tools to write portable C. Put it all together in the capstone project:
- [Project: Snake Game](./17_Project_Snake_Game.md) -- A terminal game combining everything you have learned
