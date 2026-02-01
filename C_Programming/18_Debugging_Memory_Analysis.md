# 디버깅과 메모리 분석

## 개요

효과적인 디버깅은 프로그래머의 핵심 역량입니다. 이 장에서는 GDB 디버거, Valgrind 메모리 분석 도구, 그리고 AddressSanitizer를 사용하여 버그를 찾고 메모리 문제를 해결하는 방법을 학습합니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 포인터, 동적 메모리 할당

---

## 목차

1. [디버깅 기초](#디버깅-기초)
2. [GDB 디버거](#gdb-디버거)
3. [Valgrind 메모리 분석](#valgrind-메모리-분석)
4. [AddressSanitizer](#addresssanitizer)
5. [일반적인 메모리 버그](#일반적인-메모리-버그)
6. [디버깅 전략](#디버깅-전략)

---

## 디버깅 기초

### 디버그 빌드

디버깅을 위해서는 `-g` 플래그로 컴파일해야 합니다.

```bash
# 디버그 심볼 포함
gcc -g -Wall -Wextra program.c -o program

# 최적화 없이 (디버깅 용이)
gcc -g -O0 -Wall -Wextra program.c -o program

# 디버그 + 최적화 (릴리스 버그 추적)
gcc -g -O2 -Wall -Wextra program.c -o program
```

### printf 디버깅

가장 기본적인 디버깅 방법입니다.

```c
#include <stdio.h>

#define DEBUG 1

#if DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

int calculate(int a, int b) {
    DEBUG_PRINT("calculate called: a=%d, b=%d", a, b);
    int result = a * b + a;
    DEBUG_PRINT("result=%d", result);
    return result;
}

int main(void) {
    int x = calculate(5, 3);
    DEBUG_PRINT("main: x=%d", x);
    return 0;
}
```

### assert 매크로

전제 조건 검증에 사용합니다.

```c
#include <assert.h>
#include <stdlib.h>

int divide(int a, int b) {
    assert(b != 0 && "Division by zero!");
    return a / b;
}

void process_array(int *arr, size_t size) {
    assert(arr != NULL && "Array is NULL!");
    assert(size > 0 && "Size must be positive!");

    for (size_t i = 0; i < size; i++) {
        arr[i] *= 2;
    }
}
```

> **참고**: 릴리스 빌드에서 assert를 비활성화하려면 `-DNDEBUG` 플래그 사용

---

## GDB 디버거

### GDB 시작

```bash
# 프로그램 로드
gdb ./program

# 인자와 함께
gdb --args ./program arg1 arg2

# 실행 중인 프로세스 연결
gdb -p <pid>

# 코어 덤프 분석
gdb ./program core
```

### 기본 명령어

| 명령어 | 단축키 | 설명 |
|--------|--------|------|
| `run` | `r` | 프로그램 실행 |
| `continue` | `c` | 실행 계속 |
| `next` | `n` | 다음 줄 (함수 안으로 들어가지 않음) |
| `step` | `s` | 다음 줄 (함수 안으로 들어감) |
| `finish` | `fin` | 현재 함수 끝까지 실행 |
| `quit` | `q` | GDB 종료 |

### 브레이크포인트

```bash
# 줄 번호에 설정
(gdb) break main.c:15
(gdb) b 15

# 함수에 설정
(gdb) break main
(gdb) break calculate

# 조건부 브레이크포인트
(gdb) break 20 if i == 5
(gdb) break process if ptr == NULL

# 브레이크포인트 목록
(gdb) info breakpoints
(gdb) info b

# 브레이크포인트 삭제
(gdb) delete 1        # 1번 삭제
(gdb) delete          # 모두 삭제

# 브레이크포인트 비활성화/활성화
(gdb) disable 2
(gdb) enable 2
```

### 변수 검사

```bash
# 변수 출력
(gdb) print x
(gdb) p x
(gdb) p arr[0]
(gdb) p *ptr
(gdb) p ptr->field

# 표현식
(gdb) p x + y
(gdb) p sizeof(struct data)

# 배열 출력
(gdb) p *arr@10        # 10개 요소

# 형식 지정
(gdb) p/x value        # 16진수
(gdb) p/t value        # 2진수
(gdb) p/d value        # 10진수
(gdb) p/c value        # 문자

# 지역 변수 모두 출력
(gdb) info locals

# 전역 변수
(gdb) info variables
```

### 메모리 검사

```bash
# 메모리 내용 확인
(gdb) x/10xb ptr       # 10바이트, 16진수
(gdb) x/10dw ptr       # 10워드, 10진수
(gdb) x/s str          # 문자열
(gdb) x/10i func       # 10개 명령어 (역어셈블)

# 형식: x/[개수][형식][크기]
# 형식: x(16진수), d(10진수), s(문자열), i(명령어)
# 크기: b(바이트), h(2바이트), w(4바이트), g(8바이트)
```

### 스택 추적

```bash
# 백트레이스 (호출 스택)
(gdb) backtrace
(gdb) bt

# 특정 프레임 선택
(gdb) frame 2
(gdb) f 2

# 프레임 정보
(gdb) info frame

# 상위/하위 프레임 이동
(gdb) up
(gdb) down
```

### 워치포인트

변수가 변경될 때 멈춥니다.

```bash
# 쓰기 워치포인트
(gdb) watch x
(gdb) watch arr[5]
(gdb) watch *ptr

# 읽기 워치포인트
(gdb) rwatch x

# 읽기/쓰기 워치포인트
(gdb) awatch x

# 워치포인트 목록
(gdb) info watchpoints
```

### GDB 실전 예제

```c
// buggy.c - 버그가 있는 코드
#include <stdio.h>
#include <stdlib.h>

int sum_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // 버그: <= 대신 < 사용해야 함
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
(gdb) print arr[0]
$2 = 1
(gdb) next
(gdb) print i
$3 = 0
(gdb) watch sum
(gdb) continue
# sum이 변경될 때마다 멈춤
```

### GDB TUI 모드

텍스트 사용자 인터페이스로 소스 코드를 보면서 디버깅합니다.

```bash
# TUI 모드 시작
(gdb) tui enable
# 또는 시작 시
$ gdb -tui ./program

# 레이아웃 변경
(gdb) layout src       # 소스 코드
(gdb) layout asm       # 어셈블리
(gdb) layout split     # 소스 + 어셈블리
(gdb) layout regs      # 레지스터

# TUI 모드 종료
(gdb) tui disable
```

---

## Valgrind 메모리 분석

### Valgrind 설치

```bash
# Ubuntu/Debian
sudo apt install valgrind

# macOS (Intel만 지원)
brew install valgrind

# CentOS/RHEL
sudo yum install valgrind
```

### 기본 사용법

```bash
# 메모리 검사 실행
valgrind ./program

# 상세 출력
valgrind --leak-check=full ./program

# 더 자세한 정보
valgrind --leak-check=full --show-leak-kinds=all ./program

# 로그 파일로 저장
valgrind --log-file=valgrind.log ./program
```

### Memcheck 도구

가장 많이 사용되는 Valgrind 도구입니다.

```bash
valgrind --tool=memcheck --leak-check=full \
         --track-origins=yes ./program
```

| 옵션 | 설명 |
|------|------|
| `--leak-check=full` | 상세 누수 정보 |
| `--show-leak-kinds=all` | 모든 종류의 누수 표시 |
| `--track-origins=yes` | 초기화되지 않은 값의 출처 추적 |
| `--verbose` | 상세 출력 |

### 메모리 누수 예제

```c
// leak.c
#include <stdlib.h>
#include <string.h>

void create_leak(void) {
    int *ptr = malloc(100 * sizeof(int));
    ptr[0] = 42;
    // free(ptr); 누락!
}

char *duplicate_string(const char *str) {
    char *copy = malloc(strlen(str) + 1);
    strcpy(copy, str);
    return copy;  // 호출자가 free해야 함
}

int main(void) {
    create_leak();

    char *str = duplicate_string("Hello");
    // free(str); 누락!

    return 0;
}
```

```bash
$ gcc -g leak.c -o leak
$ valgrind --leak-check=full ./leak

==12345== HEAP SUMMARY:
==12345==     in use at exit: 406 bytes in 2 blocks
==12345==   total heap usage: 2 allocs, 0 frees, 406 bytes allocated
==12345==
==12345== 6 bytes in 1 blocks are definitely lost in loss record 1 of 2
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x10871B: duplicate_string (leak.c:11)
==12345==    by 0x108751: main (leak.c:18)
==12345==
==12345== 400 bytes in 1 blocks are definitely lost in loss record 2 of 2
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x1086E2: create_leak (leak.c:5)
==12345==    by 0x108745: main (leak.c:16)
```

### 누수 종류

| 종류 | 설명 |
|------|------|
| definitely lost | 확실히 누수 (포인터 손실) |
| indirectly lost | 간접 누수 (다른 블록을 통해 접근 가능했음) |
| possibly lost | 가능한 누수 (포인터가 블록 중간을 가리킴) |
| still reachable | 프로그램 종료 시 아직 접근 가능 |

### 잘못된 메모리 접근

```c
// invalid.c
#include <stdlib.h>

int main(void) {
    int *arr = malloc(5 * sizeof(int));

    // 초기화되지 않은 값 읽기
    int x = arr[0];

    // 경계 초과 쓰기
    arr[5] = 100;

    // 경계 초과 읽기
    int y = arr[10];

    free(arr);

    // 해제 후 사용 (Use After Free)
    arr[0] = 42;

    // 이중 해제
    free(arr);

    return x + y;
}
```

```bash
$ valgrind --track-origins=yes ./invalid

==12345== Conditional jump or move depends on uninitialised value(s)
==12345==    at 0x108691: main (invalid.c:8)
==12345==  Uninitialised value was created by a heap allocation
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x108671: main (invalid.c:5)
==12345==
==12345== Invalid write of size 4
==12345==    at 0x1086A1: main (invalid.c:11)
==12345==  Address 0x522d054 is 0 bytes after a block of size 20 alloc'd
```

---

## AddressSanitizer

### ASan 사용법

컴파일 시 플래그를 추가합니다.

```bash
# GCC
gcc -fsanitize=address -g program.c -o program

# Clang
clang -fsanitize=address -g program.c -o program

# 추가 옵션
gcc -fsanitize=address -fno-omit-frame-pointer -g program.c -o program
```

### ASan 장점

| 특징 | Valgrind | ASan |
|------|----------|------|
| 속도 | 10-50배 느림 | 2배 느림 |
| 메모리 사용 | 2배 | 3배 |
| 스택 오버플로우 | X | O |
| 전역 변수 오버플로우 | X | O |
| 재컴파일 필요 | X | O |

### ASan 예제

```c
// asan_test.c
#include <stdlib.h>

int main(void) {
    int *arr = malloc(10 * sizeof(int));

    // 힙 버퍼 오버플로우
    arr[10] = 42;

    free(arr);

    // Use After Free
    arr[0] = 100;

    return 0;
}
```

```bash
$ gcc -fsanitize=address -g asan_test.c -o asan_test
$ ./asan_test

=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x604000000028
WRITE of size 4 at 0x604000000028 thread T0
    #0 0x4011a3 in main asan_test.c:8
    #1 0x7f123456789a in __libc_start_main

0x604000000028 is located 0 bytes to the right of 40-byte region
allocated by thread T0 here:
    #0 0x7f1234567890 in malloc
    #1 0x401157 in main asan_test.c:5
```

### 기타 Sanitizer

```bash
# 정의되지 않은 동작 검사
gcc -fsanitize=undefined -g program.c -o program

# 스레드 오류 검사
gcc -fsanitize=thread -g program.c -o program -pthread

# 여러 sanitizer 조합
gcc -fsanitize=address,undefined -g program.c -o program
```

### UBSan 예제

```c
// ubsan_test.c
#include <stdio.h>
#include <limits.h>

int main(void) {
    int x = INT_MAX;
    int y = x + 1;  // 정수 오버플로우 (정의되지 않은 동작)

    int arr[5] = {1, 2, 3, 4, 5};
    int z = arr[10];  // 배열 경계 초과

    int *ptr = NULL;
    // *ptr = 42;  // NULL 역참조

    printf("%d %d\n", y, z);
    return 0;
}
```

```bash
$ gcc -fsanitize=undefined -g ubsan_test.c -o ubsan_test
$ ./ubsan_test

ubsan_test.c:7:15: runtime error: signed integer overflow:
2147483647 + 1 cannot be represented in type 'int'
```

---

## 일반적인 메모리 버그

### 1. 버퍼 오버플로우

```c
// 문제
char buffer[10];
strcpy(buffer, "This is too long!");  // 오버플로우

// 해결
char buffer[10];
strncpy(buffer, "This is too long!", sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';

// 또는 snprintf 사용
snprintf(buffer, sizeof(buffer), "%s", "This is too long!");
```

### 2. Use After Free

```c
// 문제
int *ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);
printf("%d\n", *ptr);  // 해제된 메모리 접근

// 해결
int *ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);
ptr = NULL;  // 해제 후 NULL 설정
```

### 3. 이중 해제

```c
// 문제
int *ptr = malloc(sizeof(int));
free(ptr);
free(ptr);  // 이중 해제

// 해결
int *ptr = malloc(sizeof(int));
free(ptr);
ptr = NULL;  // free(NULL)은 안전함
free(ptr);   // OK
```

### 4. 메모리 누수

```c
// 문제
void process(void) {
    int *data = malloc(100);
    if (error_condition) {
        return;  // 누수!
    }
    // ...
    free(data);
}

// 해결 1: goto 사용
void process(void) {
    int *data = malloc(100);
    if (error_condition) {
        goto cleanup;
    }
    // ...
cleanup:
    free(data);
}

// 해결 2: 구조화
void process(void) {
    int *data = malloc(100);
    if (!error_condition) {
        // 정상 처리
    }
    free(data);
}
```

### 5. 초기화되지 않은 메모리

```c
// 문제
int x;
printf("%d\n", x);  // 쓰레기 값

int *arr = malloc(10 * sizeof(int));
printf("%d\n", arr[0]);  // 쓰레기 값

// 해결
int x = 0;

int *arr = calloc(10, sizeof(int));  // 0으로 초기화
// 또는
int *arr = malloc(10 * sizeof(int));
memset(arr, 0, 10 * sizeof(int));
```

### 6. 스택 오버플로우

```c
// 문제: 무한 재귀
void infinite(void) {
    infinite();  // 스택 오버플로우
}

// 문제: 큰 지역 변수
void big_local(void) {
    int huge_array[1000000];  // 스택 오버플로우 가능
}

// 해결: 동적 할당
void use_heap(void) {
    int *array = malloc(1000000 * sizeof(int));
    // ...
    free(array);
}
```

---

## 디버깅 전략

### 1. 재현 가능하게 만들기

```c
// 랜덤 시드 고정
srand(12345);  // 항상 같은 결과

// 입력 기록
void log_input(const char *input) {
    FILE *f = fopen("input.log", "a");
    fprintf(f, "%s\n", input);
    fclose(f);
}
```

### 2. 이진 탐색으로 버그 위치 찾기

```c
// 코드 절반을 주석 처리하고 버그 발생 확인
// 버그 있는 절반을 다시 반으로 나누어 반복

// git bisect 사용
// $ git bisect start
// $ git bisect bad HEAD
// $ git bisect good v1.0
```

### 3. 방어적 프로그래밍

```c
// 함수 시작 시 검증
int process_data(int *data, size_t size) {
    // 전제 조건 검사
    if (data == NULL) {
        fprintf(stderr, "Error: data is NULL\n");
        return -1;
    }
    if (size == 0) {
        fprintf(stderr, "Error: size is 0\n");
        return -1;
    }

    // 처리 로직...

    return 0;
}
```

### 4. 로깅 레벨

```c
typedef enum {
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG
} LogLevel;

LogLevel current_level = LOG_INFO;

void log_message(LogLevel level, const char *fmt, ...) {
    if (level > current_level) return;

    const char *level_str[] = {"ERROR", "WARN", "INFO", "DEBUG"};

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[%s] ", level_str[level]);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// 사용
log_message(LOG_DEBUG, "Processing item %d", i);
log_message(LOG_ERROR, "Failed to open file: %s", filename);
```

---

## 연습 문제

### 문제 1: 메모리 누수 찾기

다음 코드에서 모든 메모리 누수를 찾아 수정하세요.

```c
#include <stdlib.h>
#include <string.h>

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

    // 처리 후 정리 없음!
}

int main(void) {
    process_students();
    return 0;
}
```

<details>
<summary>정답 보기</summary>

```c
void free_student(Student *s) {
    if (s) {
        free(s->name);
        free(s->scores);
        free(s);
    }
}

void process_students(void) {
    Student *students[3];

    students[0] = create_student("Alice", 5);
    students[1] = create_student("Bob", 3);
    students[2] = create_student("Charlie", 4);

    // 정리
    for (int i = 0; i < 3; i++) {
        free_student(students[i]);
    }
}
```

</details>

### 문제 2: GDB 사용

세그멘테이션 폴트가 발생하는 다음 코드를 GDB로 디버깅하세요.

```c
#include <stdio.h>

void recursive(int n) {
    int arr[1000];
    arr[0] = n;
    if (n > 0) {
        recursive(n - 1);
    }
}

int main(void) {
    recursive(10000);
    return 0;
}
```

<details>
<summary>해결 방법</summary>

```bash
$ gcc -g -O0 program.c -o program
$ gdb ./program
(gdb) run
# Segmentation fault 발생
(gdb) bt
# 스택 오버플로우 확인 - recursive 함수가 너무 많이 호출됨

# 해결: 재귀 깊이 줄이기 또는 반복문 사용
```

</details>

---

## 다음 단계

- [19_Advanced_Embedded_Protocols.md](./19_Advanced_Embedded_Protocols.md) - PWM, I2C, SPI

---

## 참고 자료

- [GDB 공식 문서](https://www.gnu.org/software/gdb/documentation/)
- [Valgrind 공식 문서](https://valgrind.org/docs/manual/)
- [AddressSanitizer Wiki](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [Debugging with GDB (책)](https://sourceware.org/gdb/current/onlinedocs/gdb/)
