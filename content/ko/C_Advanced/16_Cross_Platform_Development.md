# C 크로스 플랫폼 개발

**이전**: [디버깅과 프로파일링](./15_Debugging_and_Profiling.md) | **다음**: [프로젝트: 스네이크 게임](./17_Project_Snake_Game.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 데이터 타입, 헤더, API, 툴체인에서 플랫폼별 차이를 식별할 수 있다
2. 전처리기 매크로와 조건부 컴파일을 사용하여 플랫폼 의존적 코드를 격리할 수 있다
3. OS 차이를 균일한 API 뒤에 숨기는 플랫폼 추상화 계층(PAL)을 설계할 수 있다
4. CMake를 설정하여 Windows, Linux, macOS용 크로스 플랫폼 C 프로젝트를 빌드할 수 있다
5. 고정 너비 정수, 표준 라이브러리 사용, 엔디언 처리 등 이식 가능한 코딩 방법을 적용할 수 있다

---

사실상 모든 비자명 C 프로그램은 운영체제에 따라 달라지는 무언가를 다룹니다: 파일 경로는 Unix에서 `/`를, Windows에서 `\`를 사용하고; 스레드는 pthreads 또는 Win32 API에서 가져오고; 소켓은 초기화와 오류 코드가 다릅니다. 이식 가능한 C를 작성한다는 것은 새로운 플랫폼이 등장해도 비즈니스 로직이 변경되지 않도록 이러한 차이점을 격리하는 것입니다. 이 레슨에서는 간단한 `#ifdef` 가드부터 세 가지 주요 데스크탑 모두에서 컴파일되는 완전한 CMake 기반 빌드까지 보여줍니다.

---

## 목차

1. [크로스 플랫폼 C가 어려운 이유](#1-크로스-플랫폼-c가-어려운-이유)
2. [플랫폼 감지 매크로](#2-플랫폼-감지-매크로)
3. [조건부 컴파일](#3-조건부-컴파일)
4. [이식 가능한 데이터 타입](#4-이식-가능한-데이터-타입)
5. [바이트 순서와 엔디언](#5-바이트-순서와-엔디언)
6. [플랫폼 추상화 계층](#6-플랫폼-추상화-계층)
7. [크로스 플랫폼 파일 및 경로 처리](#7-크로스-플랫폼-파일-및-경로-처리)
8. [크로스 플랫폼 네트워킹](#8-크로스-플랫폼-네트워킹)
9. [크로스 플랫폼 스레딩](#9-크로스-플랫폼-스레딩)
10. [CMake로 빌드하기](#10-cmake로-빌드하기)
11. [연습 문제](#11-연습-문제)

---

## 1. 크로스 플랫폼 C가 어려운 이유

C는 표준화되어 있지만(C11, C17, C23), 표준은 *언어*와 최소한의 *표준 라이브러리*만 다룹니다. 그 외의 모든 것 -- 스레드(C11 이전), 소켓, 파일시스템 탐색, 동적 로딩, GUI -- 은 플랫폼별입니다.

### 주요 차이 영역

| 영역 | Linux / macOS | Windows |
|------|--------------|---------|
| 경로 구분자 | `/` | `\` (`/`도 허용) |
| 줄 끝 | `\n` (LF) | `\r\n` (CRLF) |
| 동적 라이브러리 | `.so` / `.dylib` | `.dll` |
| 스레드 API | `pthread` | Win32 Threads / `_beginthreadex` |
| 소켓 초기화 | 불필요 | `WSAStartup()` 필요 |
| 디렉토리 목록 | `opendir/readdir` | `FindFirstFile/FindNextFile` |
| 공유 메모리 | `mmap`, `shm_open` | `CreateFileMapping` |
| 컴파일러 | GCC, Clang | MSVC, Clang, MinGW-GCC |

---

## 2. 플랫폼 감지 매크로

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

## 3. 조건부 컴파일

### 간단한 #ifdef 가드

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

### 소스 파일 분리

큰 블록의 경우 각 구현을 별도의 파일에 넣고 빌드 시스템이 선택하게 합니다:

```
src/
├── sleep.h           /* Public API: void sleep_ms(int ms); */
├── sleep_posix.c     /* POSIX implementation */
└── sleep_win32.c     /* Windows implementation */
```

CMake에서:

```cmake
if(WIN32)
    target_sources(myapp PRIVATE src/sleep_win32.c)
else()
    target_sources(myapp PRIVATE src/sleep_posix.c)
endif()
```

---

## 4. 이식 가능한 데이터 타입

### 고정 너비 정수

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

### size_t와 ptrdiff_t

- `size_t`: 부호 없음, 배열 크기와 `sizeof` 결과용
- `ptrdiff_t`: 부호 있음, 포인터 연산 결과용

```c
for (size_t i = 0; i < count; i++) {
    process(arr[i]);
}
printf("Count: %zu\n", count);
```

---

## 5. 바이트 순서와 엔디언

### 엔디언 감지

```c
int is_little_endian(void) {
    uint16_t val = 1;
    uint8_t *bytes = (uint8_t *)&val;
    return bytes[0] == 1;
}
```

### 이식 가능한 바이트 스왑

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

### 구조체 패킹

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

## 6. 플랫폼 추상화 계층

PAL은 균일한 API를 정의하고, 각 플랫폼은 자체 구현을 가집니다.

### 스레드 추상화 예제

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

## 7. 크로스 플랫폼 파일 및 경로 처리

### 경로 구분자

```c
#if PLAT_WINDOWS
    #define PATH_SEP '\\'
#else
    #define PATH_SEP '/'
#endif
```

### 디렉토리 목록

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

## 8. 크로스 플랫폼 네트워킹

Windows에서는 소켓 호출 전에 `WSAStartup`이 필요합니다:

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

## 9. 크로스 플랫폼 스레딩

### 뮤텍스 추상화

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

## 10. CMake로 빌드하기

### 최소 CMakeLists.txt

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

### 빌드

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

### 기능 감지

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

## 11. 연습 문제

### 연습 문제 1: 플랫폼 정보 리포터

다음을 출력하는 프로그램을 작성하세요:
- 운영체제 이름과 아키텍처
- 컴파일러 이름과 버전
- 시스템이 리틀 엔디언인지 빅 엔디언인지
- `sizeof(int)`, `sizeof(long)`, `sizeof(void *)`

### 연습 문제 2: 이식 가능한 sleep_ms

POSIX(`nanosleep` 사용)와 Windows(`Sleep` 사용) 모두에서 동작하는 `void sleep_ms(unsigned int ms)`를 구현하세요. 500ms 동안 대기하고 `clock()`으로 경과 시간을 측정하여 +/-50ms 이내의 정확성을 검증하는 테스트를 작성하세요.

### 연습 문제 3: 크로스 플랫폼 파일 복사

표준 C(`fopen/fread/fwrite`)만 사용하여 `int file_copy(const char *src, const char *dst)`를 작성하세요. 그런 다음 더 나은 성능을 위해 플랫폼별 API(Linux의 `sendfile`, macOS의 `copyfile`, Windows의 `CopyFileA`)를 사용하는 두 번째 버전을 작성하세요.

### 연습 문제 4: 플랫폼 추상화 계층

다음을 갖춘 최소 PAL을 설계하고 구현하세요:
- `pal_mutex_t`: init, lock, unlock, destroy
- `pal_thread_t`: create, join, destroy
- `pal_sleep_ms(unsigned int ms)`

뮤텍스 보호 하에 공유 카운터를 1,000,000번 증가시키는 4개의 스레드를 생성하는 테스트를 작성하세요. 플랫폼에 관계없이 최종 카운트는 4,000,000이어야 합니다.

### 연습 문제 5: CMake 기능 감지

다음을 수행하는 CMakeLists.txt를 만드세요:
1. `<threads.h>`가 사용 가능한지 감지
2. 사용 가능하면 C11 스레드로 빌드
3. 불가능하면 pthreads(POSIX) 또는 Win32 스레드로 폴백
4. `#define HAS_C11_THREADS`가 포함된 `config.h` 생성

---

## 다음 단계

이제 이식 가능한 C를 작성하는 도구를 갖추었습니다. 캡스톤 프로젝트에서 모든 것을 합쳐 보세요:
- [프로젝트: 스네이크 게임](./17_Project_Snake_Game.md) -- 배운 모든 것을 결합한 터미널 게임
