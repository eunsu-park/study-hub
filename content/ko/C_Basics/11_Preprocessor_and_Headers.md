# 전처리기와 헤더

**이전**: [파일 입출력](./10_File_IO.md) | **다음**: [빌드 도구와 디버깅](./12_Build_Tools_and_Debugging.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 전처리기 단계를 설명하고 컴파일 전에 소스 코드를 어떻게 변환하는지 이해하기
2. 표준 및 사용자 정의 헤더에 `#include` 사용하기
3. `#define`으로 객체형 매크로와 함수형 매크로를 정의하고 사용하기
4. `#ifdef`, `#ifndef`, `#if`, `#else`, `#endif`로 조건부 컴파일 적용하기
5. 다중 포함을 방지하는 인클루드 가드가 있는 헤더 파일 작성하기

---

C 전처리기는 컴파일러가 코드를 보기 전에 실행되는 텍스트 변환 엔진입니다. 파일 포함, 매크로 확장, 조건부 컴파일을 처리합니다 — 단일 파일을 넘어 C 프로젝트를 구성하는 데 기본이 되는 세 가지 메커니즘입니다. C 특유의 관용구, 오류 메시지, 프로젝트 구조의 많은 부분이 전처리기의 동작으로 설명되므로, 전처리기를 이해하는 것은 필수적입니다.

## 1. 전처리기란?

C 파일을 컴파일할 때, 번역은 여러 단계를 거칩니다:

```
Source Code (.c)
      │
      ▼
  Preprocessor  ──── Phase 1: text substitution (#include, #define, #if)
      │
      ▼
  Compiler      ──── Phase 2: parse C code, generate assembly
      │
      ▼
  Assembler     ──── Phase 3: generate object code (.o)
      │
      ▼
  Linker        ──── Phase 4: combine objects into executable
      │
      ▼
  Executable
```

전처리기는 **텍스트**에 대해 작동합니다 — C 문법, 타입, 스코프에 대해 아무것도 모릅니다. 모든 지시자는 `#`으로 시작하며, 해당 줄에서 공백이 아닌 첫 번째 문자여야 합니다.

전처리기 출력을 다음과 같이 확인할 수 있습니다:

```bash
gcc -E main.c -o main.i    # output preprocessed source
```

이 명령은 모든 `#include`, `#define`, 조건부 지시자를 확장하여, 컴파일러가 이후 처리하는 단일 파일을 생성합니다.

---

## 2. #include

`#include` 지시자는 다른 파일의 전체 내용을 지시자 위치에 복사합니다.

### 꺽쇠괄호 vs 따옴표

```c
#include <stdio.h>       /* search system include paths */
#include "myheader.h"    /* search current directory first, then system paths */
```

| 문법 | 검색 순서 | 사용 대상 |
|--------|-------------|---------|
| `<header.h>` | 시스템 인클루드 디렉토리만 | 표준 라이브러리 헤더 |
| `"header.h"` | 현재 디렉토리 먼저, 그다음 시스템 경로 | 자신의 프로젝트 헤더 |

### 자주 사용하는 표준 헤더

| 헤더 | 제공 내용 |
|--------|----------|
| `<stdio.h>` | `printf`, `scanf`, `fopen`, `FILE` |
| `<stdlib.h>` | `malloc`, `free`, `atoi`, `exit`, `rand` |
| `<string.h>` | `strlen`, `strcpy`, `strcmp`, `memcpy` |
| `<math.h>` | `sqrt`, `sin`, `cos`, `pow` (`-lm`으로 링크) |
| `<stdbool.h>` | `bool`, `true`, `false` (C99) |
| `<stdint.h>` | `int32_t`, `uint8_t`, 고정 너비 타입 (C99) |
| `<stddef.h>` | `size_t`, `NULL`, `ptrdiff_t` |
| `<ctype.h>` | `isalpha`, `isdigit`, `toupper`, `tolower` |
| `<errno.h>` | `errno`, 오류 코드 |
| `<assert.h>` | 디버깅 검사용 `assert` 매크로 |
| `<limits.h>` | `INT_MAX`, `INT_MIN`, `CHAR_BIT` |

---

## 3. 객체형 매크로 (Object-Like Macros)

객체형 매크로는 이름을 대체 텍스트와 연결합니다. 관례상 매크로 이름은 `ALL_CAPS`를 사용합니다.

```c
#define PI          3.14159265358979
#define MAX_SIZE    1024
#define AUTHOR      "Alice"
#define DEBUG_MODE  1
```

소스에서 매크로 이름이 나타날 때마다 컴파일 전에 정의로 대체됩니다:

```c
#include <stdio.h>

#define BUFFER_SIZE 256
#define VERSION     "1.2.0"

int main(void) {
    char buf[BUFFER_SIZE];           /* becomes: char buf[256]; */
    printf("Version: %s\n", VERSION); /* becomes: printf("Version: %s\n", "1.2.0"); */
    return 0;
}
```

### 매직 넘버 대비 장점

- **가독성**: `BUFFER_SIZE`가 `256`보다 더 설명적
- **유지보수성**: 한 곳에서 값을 변경하면 모든 곳에 반영
- **메모리 사용 없음**: 매크로는 컴파일 타임 텍스트 대체이며 변수가 아님

### `const`나 `enum`을 대신 사용하는 경우

최신 C(C99+)에서는 타입 안전하고 디버거에서 볼 수 있는 `const` 변수와 `enum` 값이 더 나은 선택인 경우가 많습니다:

```c
static const double PI = 3.14159265358979;   /* type-safe constant */
enum { MAX_SIZE = 1024 };                     /* integer constant */
```

그러나 `#define`은 문자열 상수, 헤더 가드, 조건부 컴파일에 여전히 필요합니다.

---

## 4. 함수형 매크로 (Function-Like Macros)

함수형 매크로는 매크로 이름 바로 뒤에(공백 없이) 괄호로 묶인 매개변수를 받습니다.

```c
#define SQUARE(x)    ((x) * (x))
#define MAX(a, b)    ((a) > (b) ? (a) : (b))
#define MIN(a, b)    ((a) < (b) ? (a) : (b))
#define ABS(x)       ((x) < 0 ? -(x) : (x))
```

### 괄호 규칙

**모든 매개변수와 전체 표현식에 항상 괄호를 씌우세요.** 괄호가 없으면 연산자 우선순위 때문에 미묘한 버그가 발생할 수 있습니다.

```c
/* BAD — missing parentheses */
#define SQUARE_BAD(x) x * x

int result = SQUARE_BAD(2 + 3);
/* Expands to: 2 + 3 * 2 + 3 = 2 + 6 + 3 = 11 (wrong!) */

/* GOOD — fully parenthesized */
#define SQUARE(x) ((x) * (x))

int result = SQUARE(2 + 3);
/* Expands to: ((2 + 3) * (2 + 3)) = 25 (correct) */
```

### 이중 평가 함정 (Double Evaluation Pitfall)

매크로 인수는 텍스트로 대체되므로 두 번 이상 평가될 수 있습니다:

```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int x = 5, y = 3;
int z = MAX(x++, y);
/* Expands to: ((x++) > (y) ? (x++) : (y))
   x is incremented TWICE if x > y — almost certainly a bug */
```

**규칙**: 부작용이 있는 표현식을 함수형 매크로에 절대 전달하지 마세요. 필요한 경우 인라인 함수를 대신 사용하세요:

```c
static inline int max_int(int a, int b) {
    return a > b ? a : b;
}
```

### 여러 줄 매크로

백슬래시 `\`를 사용하여 매크로를 여러 줄에 걸쳐 계속합니다:

```c
#define PRINT_ARRAY(arr, n)          \
    do {                             \
        for (int i = 0; i < (n); i++) \
            printf("%d ", (arr)[i]); \
        printf("\n");                \
    } while (0)
```

`do { ... } while (0)` 관용구는 매크로가 모든 맥락에서(예: 중괄호 없는 `if` 뒤에서) 올바르게 작동하도록 합니다.

---

## 5. 조건부 컴파일

조건부 지시자를 사용하면 컴파일 타임 조건에 따라 코드를 포함하거나 제외할 수 있습니다. 플랫폼별 코드, 디버그 모드, 기능 토글에 필수적입니다.

### #ifdef와 #ifndef

```c
#define DEBUG

#ifdef DEBUG
    printf("Debug: x = %d\n", x);   /* included only if DEBUG is defined */
#endif

#ifndef RELEASE
    printf("Not a release build\n");  /* included only if RELEASE is NOT defined */
#endif
```

### #if, #elif, #else, #endif

```c
#define VERSION 3

#if VERSION == 1
    printf("Version 1\n");
#elif VERSION == 2
    printf("Version 2\n");
#elif VERSION >= 3
    printf("Version 3 or later\n");
#else
    printf("Unknown version\n");
#endif
```

### 플랫폼별 코드

```c
#include <stdio.h>

void clear_screen(void) {
#ifdef _WIN32
    system("cls");
#elif defined(__APPLE__) || defined(__linux__)
    system("clear");
#else
    printf("\033[2J\033[H");  /* ANSI escape fallback */
#endif
}
```

### 컴파일 타임 기능 플래그

명령줄에서 매크로를 정의할 수 있습니다:

```bash
gcc -DDEBUG -DVERSION=3 main.c -o main
```

이는 파일 상단에 `#define DEBUG`와 `#define VERSION 3`을 쓰는 것과 동일합니다.

| 지시자 | 용도 |
|-----------|---------|
| `#ifdef NAME` | `NAME`이 정의되었으면 참 |
| `#ifndef NAME` | `NAME`이 정의되지 않았으면 참 |
| `#if expr` | 상수 표현식이 0이 아니면 참 |
| `#elif expr` | else-if 체인 |
| `#else` | 기본 분기 |
| `#endif` | 조건부 블록 종료 |
| `defined(NAME)` | `#if` / `#elif`에서 사용 가능한 연산자 |

---

## 6. 헤더 파일

헤더 파일(`.h`)은 다른 소스 파일이 사용할 수 있는 **인터페이스**를 선언합니다: 함수 프로토타입, 타입 정의, 매크로, extern 변수 선언.

### 인클루드 가드 (Include Guards)

보호 장치가 없으면 같은 헤더를 두 번 포함하면 중복 정의 오류가 발생합니다. **인클루드 가드**가 이를 방지합니다:

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

double circle_area(double radius);
double circle_circumference(double radius);

typedef struct {
    double x;
    double y;
} Point;

#endif /* MATH_UTILS_H */
```

`math_utils.h`가 처음 포함되면 `MATH_UTILS_H`가 정의되지 않았으므로 내용이 처리되고 매크로가 정의됩니다. 이후 포함에서는 `#ifndef` 테스트가 실패하고 전체 파일이 건너뛰어집니다.

### #pragma once (비표준이지만 널리 지원됨)

```c
#pragma once

double circle_area(double radius);
double circle_circumference(double radius);
```

대부분의 현대 컴파일러(GCC, Clang, MSVC)가 `#pragma once`를 지원합니다. 더 간단하지만 C 표준의 일부가 아닙니다. 많은 프로젝트에서 최대 호환성을 위해 둘 다 사용합니다:

```c
#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#pragma once

/* ... declarations ... */

#endif
```

### 헤더에 넣어야 할 것

| `.h`에 속하는 것 | `.c`에 속하는 것 |
|-----------------|-----------------|
| 함수 프로토타입 | 함수 정의 (본문) |
| `typedef`, `struct`, `enum` 정의 | 정적 (파일 스코프) 함수 |
| `#define` 매크로와 상수 | 전역 변수 정의 |
| `extern` 변수 선언 | 자체 헤더의 `#include` |
| 인라인 함수 정의 | 구현 세부사항 |

---

## 7. 다중 파일 컴파일

실제 C 프로그램은 여러 `.c` 파일에 걸쳐 있습니다. 각 파일은 독립적으로 오브젝트 파일(`.o`)로 컴파일된 다음, 링커가 이들을 결합합니다.

### 예시 프로젝트

```
project/
├── main.c
├── math_utils.h
└── math_utils.c
```

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#define PI 3.14159265358979

double circle_area(double radius);
double circle_circumference(double radius);

#endif
```

```c
/* math_utils.c */
#include "math_utils.h"

double circle_area(double radius) {
    return PI * radius * radius;
}

double circle_circumference(double radius) {
    return 2.0 * PI * radius;
}
```

```c
/* main.c */
#include <stdio.h>
#include "math_utils.h"

int main(void) {
    double r = 5.0;
    printf("Area: %.2f\n", circle_area(r));
    printf("Circumference: %.2f\n", circle_circumference(r));
    return 0;
}
```

### 컴파일 단계

```bash
gcc -c math_utils.c -o math_utils.o   # compile to object file
gcc -c main.c -o main.o               # compile to object file
gcc math_utils.o main.o -o program    # link into executable

# Or all at once:
gcc main.c math_utils.c -o program
```

### 선언 vs 정의

- **선언(declaration)**은 무언가가 존재하고 타입이 무엇인지 컴파일러에게 알려줍니다.
- **정의(definition)**는 저장 공간을 할당하거나 함수 본문을 제공합니다.

```c
/* Declaration (in header) */
extern int global_count;        /* variable exists somewhere */
double compute(double x);       /* function exists somewhere */

/* Definition (in .c file) */
int global_count = 0;           /* allocates storage */
double compute(double x) {      /* provides the body */
    return x * x;
}
```

`extern` 키워드는 "이 변수는 다른 파일에 정의되어 있다"고 말합니다. 이것이 없으면 각 `.c` 파일이 자체 복사본을 생성하고, 링커가 중복 심볼을 보고합니다.

---

## 8. 기타 지시자

### #undef

이전에 정의된 매크로를 제거합니다:

```c
#define TEMP 100
/* ... use TEMP ... */
#undef TEMP
/* TEMP is no longer defined */
```

### #error

사용자 정의 메시지와 함께 컴파일 오류를 강제합니다:

```c
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
#error "This code requires C99 or later"
#endif
```

### #pragma

컴파일러별 명령:

```c
#pragma pack(push, 1)   /* disable struct padding (GCC, MSVC) */
typedef struct {
    char a;
    int b;
} Packed;
#pragma pack(pop)
```

### 미리 정의된 매크로

컴파일러가 자동으로 여러 유용한 매크로를 정의합니다:

| 매크로 | 확장 결과 | 예시 |
|-------|-----------|---------|
| `__FILE__` | 현재 파일명 | `"main.c"` |
| `__LINE__` | 현재 줄 번호 | `42` |
| `__DATE__` | 컴파일 날짜 | `"Mar 17 2026"` |
| `__TIME__` | 컴파일 시간 | `"14:30:00"` |
| `__func__` | 현재 함수명 (C99) | `"main"` |
| `__STDC__` | 컴파일러가 ISO C를 준수하면 1 | `1` |
| `__STDC_VERSION__` | C 표준 버전 | `201112L` (C11) |

로깅과 디버깅에 유용합니다:

```c
#define LOG(msg) fprintf(stderr, "[%s:%d] %s: %s\n", \
    __FILE__, __LINE__, __func__, msg)

void process(void) {
    LOG("starting process");
    /* Output: [main.c:25] process: starting process */
}
```

### 문자열화와 토큰 붙이기 (Stringification and Token Pasting)

두 가지 특별한 전처리기 연산자:

```c
/* # — converts a macro argument to a string literal */
#define STRINGIFY(x) #x
printf("%s\n", STRINGIFY(Hello World));  /* prints: Hello World */

/* ## — concatenates two tokens */
#define MAKE_VAR(prefix, num) prefix##num
int MAKE_VAR(value, 1) = 10;  /* becomes: int value1 = 10; */
int MAKE_VAR(value, 2) = 20;  /* becomes: int value2 = 20; */
```

---

## 연습문제

**연습문제 1 — 헤더와 소스 분리**: 이전 레슨의 단일 파일 C 프로그램을 세 파일로 분리하세요: 선언과 인클루드 가드가 있는 헤더(`.h`), 함수 본문이 있는 구현(`.c`), 이를 사용하는 `main.c`. 별도의 `gcc -c` 명령으로 컴파일하고 링크하세요.

**연습문제 2 — 디버그 매크로**: 가변 매크로를 사용하여 파일, 줄, 서식화된 메시지를 출력하는 `DEBUG_PRINT(fmt, ...)` 매크로를 작성하세요 — 단, `DEBUG` 매크로가 정의된 경우에만. `DEBUG`가 정의되지 않은 경우 매크로는 아무것도 확장하지 않아야 합니다.

**연습문제 3 — 크로스 플랫폼 유틸리티**: `_WIN32`, `__APPLE__`, `__linux__`를 사용한 조건부 컴파일로 `PLATFORM_NAME`을 문자열("Windows", "macOS", "Linux")로 정의하는 `platform.h` 헤더를 작성하세요. 이를 포함하고 플랫폼 이름을 출력하는 `main.c`를 작성하세요.

**연습문제 4 — 제네릭 MAX 매크로**: 토큰 붙이기를 사용하여 헬퍼 함수 `max_##type`을 선언하는 `GENERIC_MAX(type, a, b)` 매크로를 작성하세요. 이를 사용하여 `max_int`, `max_float`, `max_double` 함수를 생성하세요. `main`에서 세 가지를 모두 테스트하세요.

**연습문제 5 — 빌드 시스템**: 4개의 파일로 프로젝트를 만드세요: `main.c`, `utils.h`, `utils.c`, `math_ops.h`/`math_ops.c`. 각 헤더에는 적절한 인클루드 가드가 있어야 합니다. 컴파일하고 링크하는 `gcc` 명령 시퀀스를 작성하세요. 헤더가 여러 번 포함되어도 인클루드 가드가 오류를 방지하는지 확인하세요.

---

## 다음 단계

이제 전처리기가 소스 코드를 어떻게 변환하는지, 그리고 헤더로 다중 파일 C 프로젝트를 어떻게 구성하는지 이해했습니다. 다음 레슨 [빌드 도구와 디버깅](./12_Build_Tools_and_Debugging.md)에서는 Makefile로 컴파일을 자동화하고 디버깅 도구로 버그를 추적하는 방법을 배웁니다.
