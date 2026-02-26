# 레슨 23: C 테스팅과 프로파일링(Testing and Profiling)

**이전**: [프로세스 간 통신과 시그널](./22_IPC_and_Signals.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C용 Unity 테스팅 프레임워크(Unity testing framework)를 설치하고 단위 테스트 작성
2. 로직을 I/O와 분리하고 의존성 주입(dependency injection)을 사용하여 테스트 가능한 C 코드 구조화
3. 일반, 경계, 오류 케이스를 포함하는 어서션 매크로(assertion macros)로 함수 동작 검증
4. gprof로 CPU 집약적 프로그램을 프로파일링(profiling)하여 가장 시간이 많이 걸리는 함수 식별
5. Valgrind의 Callgrind로 명령어 수준(instruction-level) 프로파일링 수행 및 결과 시각화
6. Valgrind의 Massif로 시간에 따른 힙(heap) 사용량을 분석하여 메모리 병목 발견
7. 테스트-프로파일-최적화 사이클을 적용하여 정확성을 희생하지 않고 성능 향상
8. 정적 분석 도구(static analysis tools)(cppcheck, clang-tidy)와 컴파일러 경고로 런타임 이전에 버그 검출

---

정확성 없는 성능은 위험하고, 성능 없는 정확성은 답답합니다. 전문적인 C 개발에는 둘 다 필요하며, 핵심은 어느 것을 먼저 추구해야 하는지 아는 것입니다. 이 레슨에서는 체계적인 워크플로우를 소개합니다: 먼저 테스트를 작성하여 올바른 동작을 고정하고, 프로파일링으로 실제 병목을 찾은 뒤, 외과적으로 최적화하고, 마지막으로 아무것도 망가지지 않았는지 재테스트합니다. 이 사이클은 취미 프로젝트부터 프로덕션 시스템까지 모두 적용할 수 있습니다.

---

## 1. C에서 테스팅이 중요한 이유

C 프로그램은 다음과 같은 취약점에 특히 노출되어 있습니다:
- **메모리 오류(Memory Error)**: 버퍼 오버플로우(buffer overflow), 해제 후 사용(use-after-free), 이중 해제(double-free)
- **미정의 동작(Undefined Behavior)**: 부호 있는 정수 오버플로우, 널 역참조, 초기화되지 않은 메모리 읽기
- **논리 오류(Logic Error)**: 경계값 오류(off-by-one), 정수 잘림(integer truncation), 잘못된 포인터 연산

테스팅은 이러한 문제가 보안 취약점이나 프로덕션 장애로 이어지기 전에 발견할 수 있게 해줍니다.

### C를 위한 테스팅 피라미드(Testing Pyramid)

```
        ┌──────┐
        │ E2E  │  시스템 테스트 (shell scripts, expect)
       ┌┴──────┴┐
       │ Integ. │  멀티 모듈 테스트, 파일 I/O, IPC
      ┌┴────────┴┐
      │  Unit    │  개별 함수 테스트 (Unity, CMocka)
     ┌┴──────────┴┐
     │  Static   │  컴파일러 경고, clang-tidy, cppcheck
    └──────────────┘
```

---

## 2. Unity로 단위 테스트(Unit Testing)

[Unity](https://github.com/ThrowTheSwitch/Unity)는 단일 `.c`와 `.h` 파일로 구성된 경량 C 테스팅 프레임워크입니다.

### 2.1 Unity 설정

```c
// test_math.c
#include "unity.h"
#include "math_utils.h"   // Module under test

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

// Test cases
void test_add(void) {
    TEST_ASSERT_EQUAL_INT(5, add(2, 3));
    TEST_ASSERT_EQUAL_INT(0, add(-1, 1));
    TEST_ASSERT_EQUAL_INT(-3, add(-1, -2));
}

void test_divide(void) {
    TEST_ASSERT_EQUAL_FLOAT(2.5f, divide(5.0f, 2.0f), 0.001f);
}

void test_divide_by_zero(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, divide(5.0f, 0.0f), 0.001f);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_add);
    RUN_TEST(test_divide);
    RUN_TEST(test_divide_by_zero);
    return UNITY_END();
}
```

### 2.2 주요 Unity 어서션(Assertion)

| 어서션 | 용도 |
|--------|------|
| `TEST_ASSERT_EQUAL_INT(expected, actual)` | 정수 동등 비교 |
| `TEST_ASSERT_EQUAL_FLOAT(exp, act, delta)` | 허용 오차를 포함한 부동소수점 비교 |
| `TEST_ASSERT_EQUAL_STRING(exp, act)` | 문자열 비교 |
| `TEST_ASSERT_NULL(ptr)` | 포인터가 NULL인지 확인 |
| `TEST_ASSERT_NOT_NULL(ptr)` | 포인터가 NULL이 아닌지 확인 |
| `TEST_ASSERT_TRUE(cond)` | 불리언 조건 확인 |
| `TEST_ASSERT_EQUAL_MEMORY(exp, act, len)` | 메모리 내용 비교 |
| `TEST_FAIL_MESSAGE("msg")` | 강제 실패 처리 |

---

## 3. 테스트 가능한 C 코드 작성

### 3.1 관심사의 분리(Separation of Concerns)

```c
// BAD: I/O mixed with logic — hard to test
int process_file(const char *filename) {
    FILE *f = fopen(filename, "r");
    int sum = 0;
    int val;
    while (fscanf(f, "%d", &val) == 1) {
        sum += val;
    }
    fclose(f);
    printf("Sum: %d\n", sum);
    return sum;
}

// GOOD: Pure logic separated from I/O
int sum_array(const int *arr, size_t len) {
    int sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

// I/O wrapper calls testable logic
int process_file(const char *filename) {
    // ... read into array ...
    int result = sum_array(values, count);
    printf("Sum: %d\n", result);
    return result;
}
```

### 3.2 함수 포인터를 통한 의존성 주입(Dependency Injection)

```c
// Inject allocator for testing
typedef void *(*allocator_fn)(size_t);
typedef void (*free_fn)(void *);

typedef struct {
    allocator_fn alloc;
    free_fn free;
} Allocator;

// Production allocator
static Allocator default_alloc = { malloc, free };

// Test allocator (tracks allocations)
static int alloc_count = 0;
static void *test_alloc(size_t size) {
    alloc_count++;
    return malloc(size);
}
static void test_free(void *ptr) {
    alloc_count--;
    free(ptr);
}
static Allocator test_allocator = { test_alloc, test_free };
```

---

## 4. assert.h를 이용한 테스팅 (간단한 방법)

프레임워크 없이 빠르게 테스트하고 싶을 때:

```c
#include <assert.h>
#include <string.h>

// Simple test runner
#define RUN(test) do { \
    printf("  %-40s", #test); \
    test(); \
    printf("PASS\n"); \
} while(0)

void test_strlen_basic(void) {
    assert(strlen("hello") == 5);
    assert(strlen("") == 0);
}

void test_strcmp_equal(void) {
    assert(strcmp("abc", "abc") == 0);
}

void test_strcmp_less(void) {
    assert(strcmp("abc", "abd") < 0);
}

int main(void) {
    printf("Running tests:\n");
    RUN(test_strlen_basic);
    RUN(test_strcmp_equal);
    RUN(test_strcmp_less);
    printf("All tests passed!\n");
    return 0;
}
```

---

## 5. gprof를 이용한 프로파일링(Profiling)

gprof는 어떤 함수가 CPU 시간을 가장 많이 소모하는지 보여줍니다.

### 5.1 사용 절차

```bash
# 1. Compile with profiling flags
gcc -pg -O2 -o program program.c

# 2. Run the program (generates gmon.out)
./program

# 3. Analyze the profile
gprof program gmon.out > profile.txt

# 4. Read flat profile and call graph
less profile.txt
```

### 5.2 gprof 출력 해석

```
Flat profile:

  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
 45.2     0.85     0.85     1000     0.85     1.20  sort_array
 30.1     1.42     0.57  1000000     0.00     0.00  compare
 15.0     1.70     0.28     1000     0.28     0.28  copy_array
  9.7     1.89     0.18        1   180.00  1890.00  main
```

주요 열 설명:
- **% time**: 전체 실행 시간에서 차지하는 비율
- **self seconds**: 해당 함수 자체에서 소비한 시간
- **calls**: 함수 호출 횟수
- **self ms/call**: 호출당 평균 시간 (자식 함수 제외)
- **total ms/call**: 호출당 평균 시간 (자식 함수 포함)

---

## 6. Valgrind(Callgrind)를 이용한 프로파일링

Callgrind는 재컴파일 없이 명령어(Instruction) 수준의 프로파일링을 제공합니다.

```bash
# Run with callgrind
valgrind --tool=callgrind ./program

# View results
callgrind_annotate callgrind.out.<pid>

# Or use KCachegrind for visualization
kcachegrind callgrind.out.<pid>
```

### 6.1 Callgrind vs gprof 비교

| 특성 | gprof | Callgrind |
|------|-------|-----------|
| 재컴파일 필요 여부 | 필요 (`-pg`) | 불필요 |
| 오버헤드(Overhead) | 낮음 (~5%) | 높음 (~20-50배) |
| 측정 단위 | 함수 | 명령어(Instruction) |
| 캐시 시뮬레이션(Cache Simulation) | 미지원 | 지원 |
| 콜 그래프(Call Graph) | 기본 수준 | 상세 수준 |

---

## 7. Valgrind(Massif)를 이용한 메모리 프로파일링

```bash
# Heap profiler
valgrind --tool=massif ./program
ms_print massif.out.<pid>
```

출력 결과는 시간에 따른 힙(Heap) 사용량을 보여주며, 메모리 누수와 최대 메모리 사용량을 파악하는 데 유용합니다.

---

## 8. 성능 최적화 기법

### 8.1 주요 최적화 패턴

프로파일링으로 병목을 확인한 후 적용합니다:

```c
// 1. Cache-friendly access (row-major traversal)
// BAD: column-major (cache-unfriendly)
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        sum += matrix[i][j];

// GOOD: row-major (cache-friendly)
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        sum += matrix[i][j];

// 2. Avoid repeated computation
// BAD
for (int i = 0; i < n; i++)
    result[i] = arr[i] / sqrt(sum_of_squares(arr, n));

// GOOD: compute once
double norm = sqrt(sum_of_squares(arr, n));
for (int i = 0; i < n; i++)
    result[i] = arr[i] / norm;

// 3. Strength reduction
// BAD
x = y * 4;     // multiply
// GOOD
x = y << 2;    // shift (compiler usually does this)

// 4. Branch prediction hints (GCC)
if (__builtin_expect(error_condition, 0)) {
    handle_error();  // Unlikely path
}
```

### 8.2 컴파일러 최적화 레벨

| 플래그 | 설명 |
|--------|------|
| `-O0` | 최적화 없음 (기본값, 디버깅에 적합) |
| `-O1` | 기본 최적화 |
| `-O2` | 표준 최적화 (릴리즈 빌드 권장) |
| `-O3` | 적극적 최적화 (바이너리 크기 증가 가능) |
| `-Os` | 크기 최소화 최적화 |
| `-Ofast` | `-O3` + 빠른 수학 연산 (IEEE 표준 준수 미보장) |

```bash
# Compare optimization impact
gcc -O0 -o prog_debug program.c
gcc -O2 -o prog_release program.c
gcc -O3 -march=native -o prog_fast program.c

time ./prog_debug
time ./prog_release
time ./prog_fast
```

---

## 9. 정적 분석(Static Analysis)

### 9.1 컴파일러 경고

```bash
# Maximum warnings
gcc -Wall -Wextra -Wpedantic -Werror program.c

# Even more warnings
gcc -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion \
    -Wformat=2 -Wnull-dereference -Wuninitialized program.c
```

### 9.2 정적 분석 도구

```bash
# cppcheck — static analysis
cppcheck --enable=all program.c

# clang-tidy — linting and modernization
clang-tidy program.c -- -Wall

# scan-build — Clang static analyzer
scan-build gcc -o program program.c
```

---

## 10. 통합 적용: 테스트-프로파일-최적화 사이클

```
┌─────────────────────────────────────────────┐
│  1. 테스트 작성 (Unity / assert.h)          │
│     → 최적화 전에 정확성 먼저 확보          │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  2. 프로파일링 (gprof / callgrind / massif) │
│     → 실제 병목 구간 파악                   │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  3. 핫 패스(Hot Path) 최적화                │
│     → 알고리즘 > 자료구조 > 마이크로 최적화 │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  4. 재테스트 (회귀 검사)                    │
│     → 정확성이 유지되는지 검증              │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  5. 재프로파일링 (개선 측정)                │
│     → 속도 향상 효과 정량화                 │
└─────────────────────────────────────────────┘
```

**황금 법칙**: 프로파일링 없이 절대로 최적화하지 마세요. 병목은 여러분이 생각하는 곳에 있는 경우가 드뭅니다.

---

## 연습 문제

### 연습 1: 문자열 라이브러리 단위 테스트

다음 함수를 포함하는 간단한 문자열 라이브러리(`mystring.h` / `mystring.c`)를 작성하세요:
- `my_strlen(const char *s)` — 문자열 길이 반환
- `my_strcpy(char *dst, const char *src)` — 문자열 복사
- `my_strrev(char *s)` — 문자열 제자리 뒤집기

그 다음 `assert.h`를 사용한 테스트 파일을 작성하여, 각 함수마다 빈 문자열·단일 문자·NULL 포인터 등의 경계 조건을 포함한 최소 3개의 테스트 케이스를 작성하세요.

### 연습 2: 정렬 알고리즘 프로파일링

100만 개의 임의 정수를 버블 정렬과 퀵 정렬 각각으로 정렬하는 프로그램을 작성하세요. `gprof`로 프로파일링한 후 다음 질문에 답하세요:
1. 각 알고리즘에서 `compare` 함수가 차지하는 시간 비율은?
2. 각 알고리즘의 전체 함수 호출 횟수는?
3. 퀵 정렬이 버블 정렬보다 몇 배 빠른가?

### 연습 3: 캐시 친화적(Cache-Friendly) 행렬 곱셈

512×512 행렬에 대해 단순(Naive) 방식과 타일링(Tiled/Blocked) 방식의 행렬 곱셈을 각각 구현하세요. `perf stat`으로 두 버전을 프로파일링하여 비교하세요:
1. 캐시 미스(Cache Miss) 비율 (`perf stat -e cache-misses,cache-references`)
2. IPC (Instructions Per Cycle, 사이클당 명령어 수)
3. 실제 실행 시간(Wall-clock Time)

---

## 요약

| 도구 | 용도 | 사용 시점 |
|------|------|-----------|
| Unity | 단위 테스트 프레임워크 | 지속적 개발 과정 |
| assert.h | 간단한 정합성 검사 | 프로토타이핑, 소규모 프로그램 |
| gprof | CPU 프로파일링 | 릴리즈 모드 성능 분석 |
| Valgrind (callgrind) | 명령어 수준 프로파일링 | 정밀 분석 |
| Valgrind (massif) | 힙(Heap) 프로파일링 | 메모리 최적화 |
| cppcheck | 정적 분석 | CI/CD 파이프라인 |
| `-Wall -Wextra` | 컴파일러 경고 | 모든 컴파일 시 |

---

**이전**: [프로세스 간 통신과 시그널](./22_IPC_and_Signals.md)
