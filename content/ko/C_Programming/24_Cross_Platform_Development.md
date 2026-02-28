# 레슨 24: C 언어의 크로스 플랫폼 개발(Cross-Platform Development)

**이전**: [테스팅과 프로파일링](./23_Testing_and_Profiling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 데이터 타입, 헤더, API, 툴체인의 플랫폼별 차이점을 파악한다
2. 전처리기 매크로(preprocessor macro)와 조건부 컴파일(conditional compilation)을 사용하여 플랫폼 의존 코드를 격리한다
3. OS 차이를 균일한 API 뒤에 숨기는 플랫폼 추상화 계층(PAL, Platform Abstraction Layer)을 설계한다
4. CMake를 이용해 Windows, Linux, macOS 대상의 크로스 플랫폼 C 프로젝트를 빌드하도록 설정한다
5. 이식성 있는 코딩 관행(고정 폭 정수, 표준 라이브러리 사용, 엔디언 처리)을 적용한다

---

사실상 모든 비자명한 C 프로그램은 운영체제에 따라 달라지는 무언가를 건드린다: 파일 경로는 Unix에서 `/`를 사용하고 Windows에서 `\`를 사용하며, 스레드(thread)는 pthreads나 Win32 API에서 오고, 소켓(socket)은 초기화와 오류 코드가 서로 다르다. 이식성 있는 C를 작성한다는 것은 이러한 차이점을 격리하여 새 플랫폼이 추가될 때 비즈니스 로직이 변경되지 않도록 하는 것을 의미한다. 이 레슨에서는 단순한 `#ifdef` 가드(guard)에서 시작하여 세 가지 주요 데스크톱 운영체제에서 모두 컴파일되는 완전한 CMake 기반 빌드까지 그 방법을 보여준다.

---

## 목차

1. [크로스 플랫폼 C가 어려운 이유](#1-크로스-플랫폼-c가-어려운-이유)
2. [플랫폼 감지 매크로](#2-플랫폼-감지-매크로)
3. [조건부 컴파일](#3-조건부-컴파일)
4. [이식성 있는 데이터 타입](#4-이식성-있는-데이터-타입)
5. [바이트 순서와 엔디언](#5-바이트-순서와-엔디언)
6. [플랫폼 추상화 계층](#6-플랫폼-추상화-계층)
7. [크로스 플랫폼 파일 및 경로 처리](#7-크로스-플랫폼-파일-및-경로-처리)
8. [크로스 플랫폼 네트워킹](#8-크로스-플랫폼-네트워킹)
9. [크로스 플랫폼 스레딩](#9-크로스-플랫폼-스레딩)
10. [CMake로 빌드하기](#10-cmake로-빌드하기)
11. [연습 문제](#11-연습-문제)

---

## 1. 크로스 플랫폼 C가 어려운 이유

C는 표준화(C11, C17, C23)되어 있지만, 표준은 *언어*와 최소한의 *표준 라이브러리*만 다룬다. 그 외의 모든 것 -- 스레드(C11 이전), 소켓, 파일 시스템 탐색, 동적 로딩, GUI -- 은 플랫폼별로 다르다.

### 주요 차이점 영역

| 영역 | Linux / macOS | Windows |
|------|--------------|---------|
| 경로 구분자 | `/` | `\` (하지만 `/`도 허용) |
| 줄 끝 문자 | `\n` (LF) | `\r\n` (CRLF) |
| 동적 라이브러리 | `.so` / `.dylib` | `.dll` |
| 스레드 API | `pthread` | Win32 Threads / `_beginthreadex` |
| 소켓 초기화 | 불필요 | `WSAStartup()` 필요 |
| 디렉터리 목록 | `opendir/readdir` | `FindFirstFile/FindNextFile` |
| 공유 메모리 | `mmap`, `shm_open` | `CreateFileMapping` |
| 컴파일러 | GCC, Clang | MSVC, Clang, MinGW-GCC |

POSIX 시스템 내에서도 차이점이 존재한다: macOS에는 `epoll`이 없고(`kqueue` 사용), Linux에는 `dispatch`(GCD)가 없으며, BSD 변형들은 각자의 시스템 호출이 있다.

---

## 2. 플랫폼 감지 매크로

컴파일러는 대상 플랫폼을 식별하는 매크로를 미리 정의한다. 신뢰할 수 있는 감지 헤더는 다음과 같다:

```c
/* platform_detect.h -- OS와 컴파일러 감지 */
#ifndef PLATFORM_DETECT_H
#define PLATFORM_DETECT_H

/* --- 운영체제 --- */
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

/* --- 컴파일러 --- */
#if defined(_MSC_VER)
    #define COMP_MSVC     1
#elif defined(__clang__)
    #define COMP_CLANG    1
#elif defined(__GNUC__)
    #define COMP_GCC      1
#endif

/* --- 아키텍처 --- */
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

> **64비트에도 `_WIN32`를 쓰는 이유?** Windows에서는 64비트 대상에서도 `_WIN32`가 정의된다. `_WIN64`는 64비트에서 추가로 정의되지만, `_WIN32` 하나만 확인해도 두 경우를 모두 포괄한다.

---

## 3. 조건부 컴파일

### 단순한 `#ifdef` 가드

플랫폼별 코드를 감싸는 가장 직접적인 방법:

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

더 큰 블록의 경우, 각 구현을 별도의 파일에 넣고 빌드 시스템이 선택하도록 한다:

```
src/
├── sleep.h           /* 공개 API: void sleep_ms(int ms); */
├── sleep_posix.c     /* POSIX 구현 */
└── sleep_win32.c     /* Windows 구현 */
```

CMake에서:

```cmake
if(WIN32)
    target_sources(myapp PRIVATE src/sleep_win32.c)
else()
    target_sources(myapp PRIVATE src/sleep_posix.c)
endif()
```

이렇게 하면 각 파일을 깔끔하게 유지할 수 있지만 -- `#ifdef` 스파게티가 없음 -- 함수 시그니처가 중복된다는 단점이 있다.

---

## 4. 이식성 있는 데이터 타입

### 고정 폭 정수(`<stdint.h>`)

`int`가 32비트라고 가정하지 말라. 명시적인 폭을 사용하라:

```c
#include <stdint.h>

uint8_t   byte_val;      /* 정확히 8비트  */
int16_t   short_val;     /* 정확히 16비트 */
uint32_t  crc;           /* 정확히 32비트 */
int64_t   timestamp_us;  /* 정확히 64비트 */

/* printf에는 <inttypes.h> 형식 매크로 사용 */
#include <inttypes.h>
printf("CRC: %" PRIu32 "\n", crc);
printf("Time: %" PRId64 " us\n", timestamp_us);
```

### `size_t`와 `ptrdiff_t`

- `size_t`: 부호 없는 정수, 배열 크기와 `sizeof` 결과에 사용. 32비트 플랫폼에서 32비트, 64비트 플랫폼에서 64비트.
- `ptrdiff_t`: 부호 있는 정수, 포인터 산술 결과에 사용.

```c
/* 배열을 이식성 있게 순회 */
for (size_t i = 0; i < count; i++) {
    process(arr[i]);
}

/* Printf: size_t에는 %zu 사용 */
printf("Count: %zu\n", count);
```

### `bool` (`<stdbool.h>`)

C99에서 `<stdbool.h>`를 도입했다. C23 이전에는 `bool`이 `_Bool`의 매크로였다. C23에서는 `bool`, `true`, `false`가 키워드가 되었다.

```c
#include <stdbool.h>

bool is_valid(const char *s) {
    return s != NULL && s[0] != '\0';
}
```

---

## 5. 바이트 순서와 엔디언

네트워크 프로토콜과 이진 파일 형식은 명시적인 엔디언(endianness) 처리를 필요로 한다.

### 엔디언 감지

```c
#include <stdint.h>

int is_little_endian(void) {
    /* 정수 1은 리틀 엔디언(LE)에서 {0x01, 0x00, ...}로 저장되고,
       빅 엔디언(BE)에서 {0x00, ..., 0x01}로 저장된다 */
    uint16_t val = 1;
    uint8_t *bytes = (uint8_t *)&val;
    return bytes[0] == 1;
}
```

### 이식성 있는 바이트 스와핑(byte swapping)

```c
/* 수동 바이트 스왑 (어디서든 작동) */
static inline uint16_t swap16(uint16_t v) {
    return (v >> 8) | (v << 8);
}

static inline uint32_t swap32(uint32_t v) {
    return ((v >> 24) & 0x000000FF) |
           ((v >>  8) & 0x0000FF00) |
           ((v <<  8) & 0x00FF0000) |
           ((v << 24) & 0xFF000000);
}

/* 네트워크 바이트 순서(빅 엔디언) 변환 */
#if PLAT_WINDOWS
    #include <winsock2.h>   /* htonl, htons, ntohl, ntohs */
#else
    #include <arpa/inet.h>  /* POSIX에서도 동일한 함수 */
#endif
```

### 구조체 패킹(struct packing)

컴파일러는 정렬을 위해 패딩(padding)을 삽입한다. 이진 프로토콜에는 패킹된 구조체를 사용한다:

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

/* 크로스 플랫폼 매크로 */
#if COMP_MSVC
    #define PACKED_STRUCT  __pragma(pack(push, 1))
    #define PACKED_END     __pragma(pack(pop))
#else
    #define PACKED_STRUCT
    #define PACKED_END     __attribute__((packed))
#endif
```

---

## 6. 플랫폼 추상화 계층

PAL(Platform Abstraction Layer)은 균일한 API를 정의하고, 각 플랫폼은 자체 구현을 가진다. 이것이 대형 C 프로젝트(SQLite, libuv, CPython)가 이식성을 유지하는 방법이다.

### 설계 패턴

```
include/
└── pal/
    ├── pal.h           /* 마스터 인클루드 */
    ├── pal_thread.h    /* 스레드 API */
    ├── pal_fs.h        /* 파일시스템 API */
    └── pal_net.h       /* 네트워킹 API */

src/pal/
├── thread_posix.c
├── thread_win32.c
├── fs_posix.c
├── fs_win32.c
├── net_posix.c
└── net_win32.c
```

### 예시: 스레드 추상화

```c
/* pal_thread.h */
#ifndef PAL_THREAD_H
#define PAL_THREAD_H

typedef struct pal_thread pal_thread_t;
typedef void *(*pal_thread_fn)(void *arg);

/* 새 스레드를 생성하고 시작한다.
   성공 시 0, 실패 시 -1을 반환한다. */
int  pal_thread_create(pal_thread_t **thread, pal_thread_fn fn, void *arg);

/* 스레드가 완료될 때까지 기다린다. 스레드 반환값을 *result에 저장한다. */
int  pal_thread_join(pal_thread_t *thread, void **result);

/* 스레드 리소스를 해제한다. join 후에 호출해야 한다. */
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

/* Windows 스레드 시그니처가 POSIX와 다르므로 래퍼를 사용한다 */
static DWORD WINAPI thread_wrapper(LPVOID param) {
    pal_thread_t *t = (pal_thread_t *)param;
    t->result = t->fn(t->arg);
    return 0;
}

int pal_thread_create(pal_thread_t **thread, pal_thread_fn fn, void *arg) {
    *thread = malloc(sizeof(pal_thread_t));
    if (!*thread) return -1;
    (*thread)->fn     = fn;
    (*thread)->arg    = arg;
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

호출 코드는 `pthread_t`나 `HANDLE`을 전혀 보지 못하고, `pal_thread_*` 함수만 사용한다.

---

## 7. 크로스 플랫폼 파일 및 경로 처리

### 경로 구분자

```c
#if PLAT_WINDOWS
    #define PATH_SEP '\\'
    #define PATH_SEP_STR "\\"
#else
    #define PATH_SEP '/'
    #define PATH_SEP_STR "/"
#endif
```

실제로는 Windows API 호출에서도 슬래시 `/`가 작동한다(셸 명령어 제외). 많은 프로젝트가 그냥 `/`를 모든 곳에 사용한다.

### 디렉터리 목록 조회

```c
#include "platform_detect.h"
#include <stdio.h>

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

### 홈 디렉터리

```c
const char *get_home_dir(void) {
#if PLAT_WINDOWS
    /* %USERPROFILE%는 보통 C:\Users\<이름> */
    return getenv("USERPROFILE");
#else
    return getenv("HOME");
#endif
}
```

---

## 8. 크로스 플랫폼 네트워킹

### 소켓 초기화

Windows는 소켓 호출 전에 `WSAStartup`이 필요하다:

```c
#include "platform_detect.h"

#if PLAT_WINDOWS
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
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
    return 0;  /* POSIX에서는 초기화 불필요 */
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

### 이식성 있는 오류 처리

```c
int net_last_error(void) {
#if PLAT_WINDOWS
    return WSAGetLastError();   /* Windows 전용 오류 코드 */
#else
    return errno;               /* POSIX errno */
#endif
}
```

---

## 9. 크로스 플랫폼 스레딩

C11에서 `<threads.h>`를 도입했지만, 지원이 고르지 않다(MSVC는 VS 2022에서야 추가). 최대 이식성을 위한 세 가지 옵션:

| 옵션 | 이식성 | 복잡도 |
|--------|------------|------------|
| C11 `<threads.h>` | GCC 12+, Clang 17+, MSVC 2022+ | 낮음 |
| PAL 래퍼 (6절) | 모든 컴파일러 | 중간 |
| 서드파티 (tinycthread) | 모든 컴파일러 | 낮음 |

### C11 스레드 (사용 가능한 경우)

```c
#include <threads.h>
#include <stdio.h>

int worker(void *arg) {
    int id = *(int *)arg;
    printf("Worker %d running\n", id);
    return 0;
}

int main(void) {
    thrd_t t;
    int id = 42;
    thrd_create(&t, worker, &id);
    thrd_join(t, NULL);
    return 0;
}
```

### 뮤텍스(mutex) 추상화

```c
/* pal_mutex.h */
#ifndef PAL_MUTEX_H
#define PAL_MUTEX_H

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

#endif
```

---

## 10. CMake로 빌드하기

CMake는 크로스 플랫폼 C/C++ 빌드의 사실상(de facto) 표준이다.

### 최소한의 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(cross_demo C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 공통 소스
set(SOURCES
    src/main.c
    src/app.c
)

# 플랫폼별 소스
if(WIN32)
    list(APPEND SOURCES
        src/pal/thread_win32.c
        src/pal/fs_win32.c
        src/pal/net_win32.c
    )
elseif(APPLE)
    list(APPEND SOURCES
        src/pal/thread_posix.c
        src/pal/fs_posix.c
        src/pal/net_posix.c
    )
    # macOS 전용: 하드웨어 감지를 위해 IOKit 링크
    find_library(IOKIT_LIB IOKit)
    list(APPEND EXTRA_LIBS ${IOKIT_LIB})
else()
    list(APPEND SOURCES
        src/pal/thread_posix.c
        src/pal/fs_posix.c
        src/pal/net_posix.c
    )
endif()

add_executable(cross_demo ${SOURCES})
target_include_directories(cross_demo PRIVATE include)

# 플랫폼별 라이브러리
if(WIN32)
    target_link_libraries(cross_demo ws2_32)
else()
    target_link_libraries(cross_demo pthread m ${EXTRA_LIBS})
endif()

# 컴파일러 경고 (모든 컴파일러)
if(MSVC)
    target_compile_options(cross_demo PRIVATE /W4 /WX)
else()
    target_compile_options(cross_demo PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
```

### 각 플랫폼에서 빌드하기

```bash
# Linux / macOS
mkdir build && cd build
cmake ..
cmake --build .

# Windows (Visual Studio)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release

# ARM용 크로스 컴파일 (x86 Linux에서)
cmake .. -DCMAKE_TOOLCHAIN_FILE=toolchains/arm-linux.cmake
cmake --build .
```

### `check_include_file`을 사용한 기능 감지

OS를 추측하는 대신, 실제 기능 가용성을 테스트한다:

```cmake
include(CheckIncludeFile)
include(CheckFunctionExists)

check_include_file("threads.h" HAVE_C11_THREADS)
check_function_exists(epoll_create1 HAVE_EPOLL)
check_function_exists(kqueue HAVE_KQUEUE)

# 설정 헤더 생성
configure_file(config.h.in config.h)
```

```c
/* config.h.in (CMake 템플릿) */
#cmakedefine HAVE_C11_THREADS
#cmakedefine HAVE_EPOLL
#cmakedefine HAVE_KQUEUE
```

```c
/* 코드에서의 사용 */
#include "config.h"

#if HAVE_EPOLL
    #include <sys/epoll.h>
    /* epoll 기반 이벤트 루프 사용 */
#elif HAVE_KQUEUE
    #include <sys/event.h>
    /* kqueue 기반 이벤트 루프 사용 */
#else
    /* select()로 폴백 */
    #include <sys/select.h>
#endif
```

---

## 11. 연습 문제

### 문제 1: 플랫폼 정보 출력기

다음을 출력하는 프로그램을 작성하라:
- 운영체제 이름과 아키텍처
- 컴파일러 이름과 버전
- 시스템이 리틀 엔디언인지 빅 엔디언인지
- `sizeof(int)`, `sizeof(long)`, `sizeof(void *)`

런타임 OS 감지 API 없이 전처리기 매크로와 `<stdint.h>`만 사용하라.

### 문제 2: 이식성 있는 `sleep_ms`

POSIX(`nanosleep` 사용)와 Windows(`Sleep` 사용) 양쪽에서 동작하는 `void sleep_ms(unsigned int ms)`를 구현하라. 500ms 동안 슬립한 후 `clock()`으로 경과 시간을 측정하여 ±50ms 이내의 정확도를 검증하는 테스트를 작성하라.

### 문제 3: 크로스 플랫폼 파일 복사

표준 C만 사용(`fopen/fread/fwrite`)하는 `int file_copy(const char *src, const char *dst)`를 작성하라. 그런 다음 플랫폼별 API(Linux의 `sendfile`, macOS의 `copyfile`, Windows의 `CopyFileA`)를 사용하여 더 높은 성능을 내는 두 번째 버전을 작성하라. 100MB 파일에서 처리량을 비교하라.

### 문제 4: 플랫폼 추상화 계층

다음을 포함하는 최소한의 PAL을 설계하고 구현하라:
- `pal_mutex_t`: init, lock, unlock, destroy
- `pal_thread_t`: create, join, destroy
- `pal_sleep_ms(unsigned int ms)`

뮤텍스 보호 하에 공유 카운터를 각각 1,000,000번 증가시키는 스레드 4개를 생성하는 테스트 프로그램을 작성하라. 최종 카운트는 플랫폼에 관계없이 4,000,000이어야 한다.

### 문제 5: CMake 기능 감지

다음을 수행하는 CMakeLists.txt를 작성하라:
1. `<threads.h>`가 사용 가능한지 감지한다
2. 사용 가능하면 C11 스레드로 빌드한다
3. 사용 불가능하면 pthreads(POSIX) 또는 Win32 스레드로 폴백한다
4. 그에 따라 `#define HAS_C11_THREADS`가 포함된 `config.h`를 생성한다
5. 동일한 소스 코드가 수정 없이 Linux, macOS, Windows에서 컴파일된다

---

*레슨 24 끝*
