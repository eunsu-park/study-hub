# 디버깅과 프로파일링

**이전**: [임베디드 시스템](./14_Embedded_Systems.md) | **다음**: [크로스 플랫폼 개발](./16_Cross_Platform_Development.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GDB의 고급 기능인 조건부 중단점, 감시점, 역방향 디버깅, 코어 덤프를 사용할 수 있다
2. Valgrind Memcheck를 사용하여 메모리 오류를 감지하고 그 출력을 해석할 수 있다
3. 컴파일 타임 새니타이저(ASan, UBSan, TSan)를 적용하여 런타임 버그를 잡을 수 있다
4. gprof과 Valgrind의 Callgrind를 사용하여 프로그램 성능을 프로파일링할 수 있다
5. Unity 프레임워크와 assert 기반 테스팅을 사용하여 단위 테스트를 작성할 수 있다
6. cppcheck와 clang-tidy를 사용하여 정적 분석을 수행할 수 있다

---

버그는 불가피하지만, 그것을 찾고 수정하는 속도가 생산적인 프로그래머와 좌절하는 프로그래머를 구분합니다. 새벽 2시의 세그멘테이션 오류는 GDB를 실행하고 중단점을 설정하고 호출 스택을 검사하는 방법을 알면 훨씬 덜 무섭습니다. 성능 없는 정확성은 답답하고, 정확성 없는 성능은 위험합니다. 이 레슨은 대화형 디버깅과 자동 메모리 분석부터 CPU 프로파일링과 단위 테스팅까지 전문 도구 킷을 제공합니다.

**난이도**: 고급

**사전 지식**: 포인터, 동적 메모리 할당

---

## 1. 고급 GDB

### GDB 시작

```bash
# Compile with debug symbols
gcc -g -O0 -Wall -Wextra program.c -o program

# Launch GDB
gdb ./program

# With arguments
gdb --args ./program arg1 arg2

# Attach to running process
gdb -p <pid>

# Analyze core dump
gdb ./program core
```

### 조건부 중단점

```bash
# Break only when condition is true
(gdb) break main.c:42 if i == 100
(gdb) break process_item if ptr == NULL
(gdb) break sort.c:15 if n > 1000

# Add condition to existing breakpoint
(gdb) condition 3 x > 0

# Ignore first N hits
(gdb) ignore 2 50  # Skip breakpoint 2 for 50 hits
```

### 감시점 (Watchpoint)

변수가 변경될 때 실행을 중지합니다:

```bash
# Break when variable is written
(gdb) watch counter
(gdb) watch arr[5]
(gdb) watch *ptr

# Break when variable is read
(gdb) rwatch x

# Break on read or write
(gdb) awatch x

# List watchpoints
(gdb) info watchpoints
```

### 코어 덤프 분석

```bash
# Enable core dumps
$ ulimit -c unlimited

# Run program (crashes and generates core file)
$ ./buggy_program
Segmentation fault (core dumped)

# Analyze the core
$ gdb ./buggy_program core
(gdb) bt           # Backtrace shows where it crashed
(gdb) frame 0      # Select the crash frame
(gdb) info locals  # See local variables
(gdb) print *ptr   # Inspect the offending pointer
```

### TUI 모드

소스 코드를 보면서 디버깅합니다:

```bash
# Start TUI
(gdb) tui enable
# Or at startup
$ gdb -tui ./program

# Change layout
(gdb) layout src    # Source code
(gdb) layout asm    # Assembly
(gdb) layout split  # Source + Assembly
(gdb) layout regs   # Registers

# Exit TUI
(gdb) tui disable
```

### GDB 실용 예제

```c
// buggy.c
#include <stdio.h>
#include <stdlib.h>

int sum_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // Bug: <= should be <
        sum += arr[i];
    }
    return sum;
}

int main(void) {
    int *numbers = malloc(5 * sizeof(int));
    for (int i = 0; i < 5; i++) {
        numbers[i] = i + 1;
    }
    int total = sum_array(numbers, 5);
    printf("Total: %d\n", total);
    free(numbers);
    return 0;
}
```

```bash
$ gcc -g -O0 buggy.c -o buggy
$ gdb ./buggy
(gdb) break sum_array
(gdb) run
(gdb) print size
$1 = 5
(gdb) watch sum
(gdb) continue
# Watchpoint fires each time sum changes
# On the 6th iteration (i=5), we read past the array
```

---

## 2. Valgrind Memcheck

### 기본 사용법

```bash
# Run memory check
valgrind ./program

# Detailed leak report
valgrind --leak-check=full ./program

# Track origin of uninitialized values
valgrind --leak-check=full --track-origins=yes ./program

# Save to log file
valgrind --log-file=valgrind.log ./program
```

### 메모리 누수 감지

```c
// leak.c
#include <stdlib.h>
#include <string.h>

void create_leak(void) {
    int *ptr = malloc(100 * sizeof(int));
    ptr[0] = 42;
    // free(ptr); missing!
}

char *duplicate_string(const char *str) {
    char *copy = malloc(strlen(str) + 1);
    strcpy(copy, str);
    return copy;  // Caller must free
}

int main(void) {
    create_leak();
    char *str = duplicate_string("Hello");
    // free(str); missing!
    return 0;
}
```

```bash
$ valgrind --leak-check=full ./leak

==12345== HEAP SUMMARY:
==12345==     in use at exit: 406 bytes in 2 blocks
==12345==   total heap usage: 2 allocs, 0 frees, 406 bytes allocated
==12345==
==12345== 6 bytes in 1 blocks are definitely lost
==12345==    at 0x4C2FB0F: malloc
==12345==    by 0x10871B: duplicate_string (leak.c:11)
==12345==    by 0x108751: main (leak.c:18)
==12345==
==12345== 400 bytes in 1 blocks are definitely lost
==12345==    at 0x4C2FB0F: malloc
==12345==    by 0x1086E2: create_leak (leak.c:5)
==12345==    by 0x108745: main (leak.c:16)
```

### 누수 유형

| 유형 | 설명 |
|------|------|
| definitely lost | 블록에 대한 포인터가 완전히 손실됨 |
| indirectly lost | 손실된 블록을 통해서만 접근 가능한 블록 |
| possibly lost | 포인터가 블록 중간을 가리킴 |
| still reachable | 프로그램 종료 시 여전히 접근 가능한 블록 |

### 잘못된 메모리 접근

Valgrind는 범위 밖 접근, 해제 후 사용, 이중 해제, 초기화되지 않은 읽기를 포착합니다:

```bash
$ valgrind --track-origins=yes ./invalid

==12345== Invalid write of size 4
==12345==    at 0x1086A1: main (invalid.c:11)
==12345==  Address 0x522d054 is 0 bytes after a block of size 20 alloc'd
```

---

## 3. 주소 새니타이저 (ASan)

### ASan 사용

```bash
# Compile with ASan
gcc -fsanitize=address -g -fno-omit-frame-pointer program.c -o program

# Run normally -- ASan reports errors at runtime
./program
```

### ASan vs Valgrind

| 특성 | Valgrind | ASan |
|------|----------|------|
| 속도 | 10-50배 느림 | 2배 느림 |
| 메모리 | 2배 사용 | 3배 사용 |
| 스택 오버플로우 | 불가 | 가능 |
| 전역 오버플로우 | 불가 | 가능 |
| 재컴파일 | 불필요 | 필요 |

### ASan 예제

```c
// asan_test.c
#include <stdlib.h>

int main(void) {
    int *arr = malloc(10 * sizeof(int));
    arr[10] = 42;  // Heap buffer overflow
    free(arr);
    arr[0] = 100;  // Use after free
    return 0;
}
```

```bash
$ gcc -fsanitize=address -g asan_test.c -o asan_test
$ ./asan_test

ERROR: AddressSanitizer: heap-buffer-overflow on address 0x604000000028
WRITE of size 4 at 0x604000000028 thread T0
    #0 0x4011a3 in main asan_test.c:5
```

---

## 4. UBSan과 TSan

### 정의되지 않은 동작 새니타이저 (UBSan)

```bash
gcc -fsanitize=undefined -g program.c -o program
```

포착 항목: 부호 있는 정수 오버플로우, null 역참조, 음수 시프트, 0으로 나누기, 범위 밖 배열 접근.

```c
// ubsan_test.c
#include <limits.h>

int main(void) {
    int x = INT_MAX;
    int y = x + 1;  // Signed overflow (undefined behavior!)
    return y;
}
```

```bash
$ gcc -fsanitize=undefined -g ubsan_test.c -o ubsan_test
$ ./ubsan_test
ubsan_test.c:5:15: runtime error: signed integer overflow:
2147483647 + 1 cannot be represented in type 'int'
```

### 스레드 새니타이저 (TSan)

```bash
gcc -fsanitize=thread -g program.c -o program -pthread
```

스레드 간 데이터 경쟁을 감지합니다. ASan과 함께 사용할 수 없습니다.

---

## 5. gprof으로 프로파일링

### 워크플로우

```bash
# 1. Compile with profiling flags
gcc -pg -O2 -o program program.c

# 2. Run (generates gmon.out)
./program

# 3. Analyze
gprof program gmon.out > profile.txt
less profile.txt
```

### gprof 출력 읽기

```
Flat profile:

  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
 45.2     0.85     0.85     1000     0.85     1.20  sort_array
 30.1     1.42     0.57  1000000     0.00     0.00  compare
 15.0     1.70     0.28     1000     0.28     0.28  copy_array
  9.7     1.89     0.18        1   180.00  1890.00  main
```

주요 열:
- **% time**: 총 실행 시간의 비율
- **self seconds**: 이 함수에서만 소요된 시간 (호출된 함수 제외)
- **calls**: 호출 횟수
- **self ms/call**: 호출당 평균 시간 (자식 제외)
- **total ms/call**: 호출당 평균 시간 (자식 포함)

---

## 6. Valgrind Callgrind

재컴파일 없이 명령어 수준 프로파일링:

```bash
# Run with callgrind
valgrind --tool=callgrind ./program

# View results as text
callgrind_annotate callgrind.out.<pid>

# Visualize with KCachegrind (GUI)
kcachegrind callgrind.out.<pid>
```

### Callgrind vs gprof

| 특성 | gprof | Callgrind |
|------|-------|-----------|
| 재컴파일 필요 | 예 (`-pg`) | 아니오 |
| 오버헤드 | 낮음 (~5%) | 높음 (~20-50배) |
| 세분화 | 함수 | 명령어 |
| 캐시 시뮬레이션 | 불가 | 가능 |
| 호출 그래프 | 기본 | 상세 |

---

## 7. 단위 테스팅

### assert.h -- 최소 접근 방식

```c
#include <assert.h>
#include <string.h>
#include <stdio.h>

// Simple test runner macro
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

### Unity 프레임워크

[Unity](https://github.com/ThrowTheSwitch/Unity)는 경량 C 테스팅 프레임워크입니다 (단일 `.c`와 `.h` 파일):

```c
// test_math.c
#include "unity.h"
#include "math_utils.h"

void setUp(void) { }
void tearDown(void) { }

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

### 주요 Unity 어서션

| 어서션 | 용도 |
|--------|------|
| `TEST_ASSERT_EQUAL_INT(exp, act)` | 정수 동등성 |
| `TEST_ASSERT_EQUAL_FLOAT(exp, act, delta)` | 허용 오차를 포함한 실수 |
| `TEST_ASSERT_EQUAL_STRING(exp, act)` | 문자열 비교 |
| `TEST_ASSERT_NULL(ptr)` | 포인터가 NULL |
| `TEST_ASSERT_NOT_NULL(ptr)` | 포인터가 NULL이 아님 |
| `TEST_ASSERT_TRUE(cond)` | 불리언 조건 |
| `TEST_ASSERT_EQUAL_MEMORY(exp, act, len)` | 메모리 비교 |

### 테스트 가능한 C 코드 작성

```c
// BAD: I/O mixed with logic
int process_file(const char *filename) {
    FILE *f = fopen(filename, "r");
    int sum = 0, val;
    while (fscanf(f, "%d", &val) == 1) sum += val;
    fclose(f);
    return sum;
}

// GOOD: Pure logic separated from I/O
int sum_array(const int *arr, size_t len) {
    int sum = 0;
    for (size_t i = 0; i < len; i++) sum += arr[i];
    return sum;
}
```

---

## 8. 정적 분석

### 컴파일러 경고

```bash
# Maximum warnings
gcc -Wall -Wextra -Wpedantic -Werror program.c

# Even more
gcc -Wall -Wextra -Wshadow -Wconversion -Wdouble-promotion \
    -Wformat=2 -Wnull-dereference -Wuninitialized program.c
```

### cppcheck

```bash
# Static analysis
cppcheck --enable=all program.c

# With suppression
cppcheck --enable=all --suppress=missingInclude .
```

### clang-tidy

```bash
# Linting
clang-tidy program.c -- -Wall

# Fix automatically
clang-tidy --fix program.c -- -Wall
```

### scan-build (Clang 정적 분석기)

```bash
scan-build gcc -o program program.c
```

---

## 테스트-프로파일-최적화 사이클

1. **테스트 작성** -- 최적화 전에 정확성 보장
2. **프로파일링** -- 실제 병목 지점 식별 (gprof / callgrind)
3. **핫 경로 최적화** -- 알고리즘 > 자료구조 > 마이크로 최적화
4. **재테스트** -- 정확성이 보존되었는지 확인
5. **재프로파일링** -- 속도 향상 정량화

**황금률**: 프로파일링 없이 절대 최적화하지 마세요. 병목 지점은 당신이 생각하는 곳에 거의 없습니다.

---

## 도구 요약

| 도구 | 용도 | 사용 시기 |
|------|------|----------|
| GDB | 대화형 디버깅 | 충돌 조사, 로직 오류 |
| Valgrind Memcheck | 메모리 오류 감지 | 메모리 누수, 잘못된 접근 |
| ASan | 빠른 메모리 오류 감지 | 개발 빌드 |
| UBSan | 정의되지 않은 동작 감지 | 개발 빌드 |
| TSan | 데이터 경쟁 감지 | 멀티스레드 프로그램 |
| gprof | CPU 프로파일링 | 릴리스 모드 성능 |
| Callgrind | 명령어 수준 프로파일링 | 상세 분석 |
| Unity | 단위 테스팅 | 지속적 개발 |
| cppcheck | 정적 분석 | CI/CD 파이프라인 |
| `-Wall -Wextra` | 컴파일러 경고 | 모든 컴파일 시 |

---

## 연습 문제

### 연습 문제 1: 메모리 누수 찾기

Valgrind를 사용하여 다음 코드의 모든 메모리 누수를 찾고 수정하세요:

```c
typedef struct {
    char *name;
    int *scores;
    int num_scores;
} Student;

Student *create_student(const char *name, int num_scores) {
    Student *s = malloc(sizeof(Student));
    s->name = malloc(strlen(name) + 1);
    strcpy(s->name, name);
    s->scores = malloc(num_scores * sizeof(int));
    s->num_scores = num_scores;
    return s;
}

void process_students(void) {
    Student *students[3];
    students[0] = create_student("Alice", 5);
    students[1] = create_student("Bob", 3);
    students[2] = create_student("Charlie", 4);
    // No cleanup!
}
```

### 연습 문제 2: 정렬 알고리즘 프로파일링

100만 개의 랜덤 정수를 버블 정렬과 퀵 정렬로 정렬하는 프로그램을 만드세요. `gprof`으로 프로파일링하고 답하세요:
1. 각 알고리즘에서 `compare`가 차지하는 시간 비율은?
2. 각 알고리즘의 함수 호출 횟수는?
3. 퀵 정렬의 버블 정렬 대비 속도 향상 비율은?

### 연습 문제 3: 문자열 라이브러리 단위 테스트

`my_strlen`, `my_strcpy`, `my_strrev` 함수를 작성하세요. `assert.h`를 사용하여 함수당 최소 3개의 테스트 케이스를 만드세요 (빈 문자열, 단일 문자, NULL 포인터 등 엣지 케이스 포함).

### 연습 문제 4: 새니타이저 버그 사냥

다음 코드를 `-fsanitize=address,undefined`로 컴파일하고 보고된 모든 문제를 수정하세요:

```c
int main(void) {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i <= 5; i++) sum += arr[i];

    int x = 2147483647;
    x = x + 1;

    int *p = malloc(10 * sizeof(int));
    free(p);
    p[0] = 42;

    return sum + x;
}
```

### 연습 문제 5: 캐시 친화적 행렬 곱셈

512x512 행렬에 대한 순진한 방식과 캐시 친화적(타일/블록) 행렬 곱셈을 구현하세요. 두 버전을 프로파일링하고 캐시 미스율, IPC, 실제 소요 시간을 비교하세요.

---

## 다음 단계

디버깅과 프로파일링을 마스터했다면 다음으로 진행하세요:
- [크로스 플랫폼 개발](./16_Cross_Platform_Development.md) -- Linux, macOS, Windows에서 컴파일되는 이식 가능한 C 작성
