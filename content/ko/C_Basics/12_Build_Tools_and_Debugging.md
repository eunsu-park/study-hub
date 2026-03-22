# 빌드 도구와 디버깅

**이전**: [전처리기와 헤더](./11_Preprocessor_and_Headers.md) | **다음**: [프로젝트: 계산기](./13_Project_Calculator.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 변수, 패턴 규칙, 자동 변수, 가짜 타겟으로 Makefile 작성하기
2. 적절한 컴파일러 경고 플래그와 최적화 수준 선택하기
3. printf 트레이싱과 체계적 이진 탐색으로 프로그램 디버깅하기
4. GDB를 사용하여 브레이크포인트 설정, 변수 검사, 코드 한 줄씩 실행하기
5. src/, include/, build/ 디렉토리로 다중 파일 프로젝트 구성하기

---

C 코드를 작성하는 방법을 아는 것은 이야기의 절반에 불과합니다 -- 효율적으로 컴파일하고, 버그를 빠르게 잡고, 프로젝트가 커져도 관리하기 쉽도록 파일을 구성해야 합니다. 이 레슨에서는 `.c` 파일 모음을 신뢰할 수 있고 디버깅 가능한 프로그램으로 변환하는 실용적인 도구를 다룹니다. 이 기술들을 지금 익히면 앞으로의 모든 프로젝트가 더 순탄해질 것입니다.

## 1. 컴파일러 플래그 심화

### 경고 플래그

컴파일러는 프로그램이 실행되기 전에 많은 버그를 잡을 수 있지만, 요청해야만 가능합니다:

```c
// compile.sh — recommended development flags
gcc -Wall -Wextra -Werror -std=c11 -g -O0 main.c -o main
```

| 플래그 | 용도 |
|------|---------|
| `-Wall` | 가장 흔한 경고 활성화 (미사용 변수, 암시적 선언 등) |
| `-Wextra` | `-Wall`을 넘어서는 추가 경고 (미사용 매개변수, 부호 비교) |
| `-Werror` | 모든 경고를 오류로 처리 -- 반드시 수정해야 함 |
| `-std=c11` | C11 표준 사용 (현대적, 이식 가능) |
| `-g` | 디버그 심볼 포함 (GDB에 필수) |
| `-O0` | 최적화 없음 -- 디버깅이 가장 쉬움 |

### 최적화 수준

| 수준 | 효과 | 사용 시기 |
|-------|--------|-------------|
| `-O0` | 최적화 없음 | 개발 및 디버깅 |
| `-O1` | 기본 최적화 | 적당한 속도 향상, 여전히 디버깅 가능 |
| `-O2` | 표준 최적화 | 릴리스 빌드 |
| `-O3` | 공격적 최적화 | 성능이 중요한 코드 |
| `-Os` | 크기 최적화 | 임베디드 시스템 |

### 새니타이저 (Sanitizers)

Address Sanitizer는 런타임에 메모리 오류를 잡습니다:

```bash
gcc -Wall -Wextra -std=c11 -g -fsanitize=address -fsanitize=undefined main.c -o main
```

| 새니타이저 | 잡는 것 |
|-----------|---------|
| `-fsanitize=address` | 버퍼 오버플로, 해제 후 사용, 메모리 누수 |
| `-fsanitize=undefined` | 부호 있는 오버플로, 널 포인터 역참조, 정렬 |

```c
// This bug is silent without sanitizers but caught with -fsanitize=address
#include <stdio.h>

int main(void) {
    int arr[5] = {1, 2, 3, 4, 5};
    printf("%d\n", arr[10]);  // Out-of-bounds read -- ASan catches this
    return 0;
}
```

---

## 2. Makefile 기초

Makefile은 긴 `gcc` 명령을 다시 입력할 필요가 없도록 컴파일을 자동화합니다. 기본 구조는 **타겟: 전제조건**이며, 탭으로 들여쓴 레시피가 뒤따릅니다:

```makefile
# Variables
CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -std=c11 -g
LDFLAGS =

# Default target
all: calculator

# Link step
calculator: main.o calc.o utils.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compile step
main.o: main.c calc.h utils.h
	$(CC) $(CFLAGS) -c main.c

calc.o: calc.c calc.h
	$(CC) $(CFLAGS) -c calc.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

# Cleanup
clean:
	rm -f *.o calculator

.PHONY: all clean
```

### 핵심 개념

| 용어 | 의미 |
|------|---------|
| **타겟 (Target)** | 빌드할 파일 (`:`의 왼쪽) |
| **전제조건 (Prerequisites)** | 타겟이 의존하는 파일 (`:`의 오른쪽) |
| **레시피 (Recipe)** | 타겟을 생성하는 셸 명령 (반드시 탭으로 들여쓰기) |
| **변수 (Variable)** | `CC = gcc`로 변수 정의; `$(CC)`로 확장 |
| `.PHONY` | 실제 파일이 아닌 타겟 선언 |

### 일반 변수

| 변수 | 관례 |
|----------|-----------|
| `CC` | C 컴파일러 (`gcc`, `clang`) |
| `CFLAGS` | 컴파일러 플래그 (`-Wall -g`) |
| `LDFLAGS` | 링커 플래그 (`-lm`, `-lpthread`) |
| `CPPFLAGS` | 전처리기 플래그 (`-I./include`, `-DDEBUG`) |

---

## 3. 자동 변수

자동 변수는 규칙에서 반복을 제거합니다:

| 변수 | 확장 결과 |
|----------|------------|
| `$@` | 타겟 파일명 |
| `$<` | 첫 번째 전제조건 |
| `$^` | 모든 전제조건 (공백으로 구분) |
| `$*` | 패턴 규칙에서 `%`에 매칭된 스템 |

### 패턴 규칙

각 `.o` 파일에 별도의 규칙을 작성하는 대신 패턴 규칙을 사용합니다:

```makefile
# Pattern rule: any .o depends on matching .c
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# This single rule replaces:
#   main.o: main.c       ->  gcc ... -c main.c -o main.o
#   calc.o: calc.c       ->  gcc ... -c calc.c -o calc.o
#   utils.o: utils.c     ->  gcc ... -c utils.c -o utils.o
```

자동 변수를 사용하는 완전한 예시:

```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g

SRCS    = main.c calc.c utils.c
OBJS    = $(SRCS:.c=.o)
TARGET  = calculator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

## 4. 고급 Makefile 기능

### 자동 의존성 생성

헤더가 변경되면 이를 포함하는 모든 파일을 다시 컴파일해야 합니다. 컴파일러가 의존성 파일을 생성하도록 하세요:

```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g -MMD -MP

SRCS    = main.c calc.c utils.c
OBJS    = $(SRCS:.c=.o)
DEPS    = $(OBJS:.o=.d)
TARGET  = calculator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean
```

| 플래그 | 용도 |
|------|---------|
| `-MMD` | `.o` 파일과 함께 `.d` 의존성 파일 생성 |
| `-MP` | 각 헤더에 가짜 타겟 추가 (헤더 삭제 시 오류 방지) |
| `-include` | `.d` 파일이 있으면 포함 (`-`는 첫 빌드 시 오류를 조용히 처리) |

### 다중 타겟

```makefile
TESTS = test_calc test_utils

all: calculator $(TESTS)

calculator: main.o calc.o utils.o
	$(CC) $(CFLAGS) -o $@ $^

test_calc: test_calc.o calc.o
	$(CC) $(CFLAGS) -o $@ $^

test_utils: test_utils.o utils.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: test
test: $(TESTS)
	./test_calc
	./test_utils
```

---

## 5. printf 디버깅

가장 간단한 디버깅 기법은 전략적 위치에서 값을 출력하는 것입니다:

```c
#include <stdio.h>

// DEBUG macro: prints only when compiled with -DDEBUG
#ifdef DEBUG
  #define DBG(fmt, ...) fprintf(stderr, "[DBG %s:%d] " fmt "\n", \
                                __FILE__, __LINE__, ##__VA_ARGS__)
#else
  #define DBG(fmt, ...)  // Expands to nothing
#endif

int binary_search(int arr[], int n, int target) {
    int lo = 0, hi = n - 1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        DBG("lo=%d mid=%d hi=%d arr[mid]=%d", lo, mid, hi, arr[mid]);

        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else                   hi = mid - 1;
    }

    DBG("target %d not found", target);
    return -1;
}
```

```bash
# Debug build: prints DBG messages
gcc -Wall -std=c11 -DDEBUG search.c -o search_dbg

# Release build: DBG messages compiled out
gcc -Wall -std=c11 -O2 search.c -o search
```

### 체계적 이진 탐색 디버깅

버그가 있지만 어디에 있는지 모를 때:

1. 의심되는 코드의 중간 지점에 출력문 추가
2. 어느 쪽 절반에 버그가 있는지 판단
3. 그 절반의 중간 지점에 출력문 추가
4. 버그가 특정될 때까지 반복

이 방법은 코드 줄 수에 대해 O(log n)으로 — 모든 줄을 읽는 것보다 훨씬 빠릅니다.

---

## 6. GDB 기초

GDB(GNU Debugger)를 사용하면 실행을 일시 중지하고, 변수를 검사하고, 코드를 한 줄씩 실행할 수 있습니다.

### GDB 시작하기

```bash
# Compile with debug symbols
gcc -Wall -std=c11 -g -O0 program.c -o program

# Start GDB
gdb ./program

# Or start with arguments
gdb --args ./program arg1 arg2
```

### 필수 명령어

| 명령어 | 축약 | 동작 |
|---------|-------|--------|
| `run` | `r` | 프로그램 시작 |
| `break main` | `b main` | `main` 함수에 브레이크포인트 설정 |
| `break file.c:42` | `b file.c:42` | 42번 줄에 브레이크포인트 설정 |
| `next` | `n` | 다음 줄 실행 (함수 호출 건너뛰기) |
| `step` | `s` | 다음 줄 실행 (함수 내부로 진입) |
| `finish` | `fin` | 현재 함수가 반환될 때까지 실행 |
| `continue` | `c` | 다음 브레이크포인트까지 실행 재개 |
| `print x` | `p x` | 변수 값 출력 |
| `print *ptr` | `p *ptr` | 포인터 역참조하여 출력 |
| `print arr[0]@10` | | 배열의 10개 요소 출력 |
| `watch x` | | 변수 `x`가 변경되면 중단 |
| `backtrace` | `bt` | 호출 스택 표시 |
| `info locals` | | 모든 지역 변수 표시 |
| `quit` | `q` | GDB 종료 |

### GDB 세션 예시

```
$ gdb ./calculator
(gdb) break main
Breakpoint 1 at 0x4011a0: file main.c, line 15.
(gdb) run
Starting program: ./calculator

Breakpoint 1, main () at main.c:15
15      double num1, num2, result;
(gdb) next
16      char operator;
(gdb) break calculate
Breakpoint 2 at 0x401250: file calc.c, line 8.
(gdb) continue
Continuing.
Enter expression: 10 / 0

Breakpoint 2, calculate (num1=10, op='/', num2=0, result=0x7ffd...) at calc.c:8
8       switch (op) {
(gdb) print num2
$1 = 0
(gdb) next
18          if (num2 == 0) {
(gdb) quit
```

### 워치포인트 (Watchpoints)

워치포인트는 변수가 변경될 때마다 실행을 일시 중지합니다 -- 손상을 추적하는 데 유용합니다:

```
(gdb) watch contact_count
Hardware watchpoint 1: contact_count
(gdb) run
...
Hardware watchpoint 1: contact_count
Old value = 3
New value = 4
add_contact (ab=0x7ffd...) at addressbook.c:95
```

---

## 7. 일반적인 버그 패턴

### 세그멘테이션 폴트 (Segmentation Fault)

세그폴트는 접근해서는 안 되는 메모리에 접근했다는 의미입니다:

```c
// Null pointer dereference
int *p = NULL;
*p = 42;            // SEGFAULT

// Array out of bounds
int arr[5];
arr[100] = 42;      // SEGFAULT (or silent corruption)

// Use after free
int *p = malloc(sizeof(int));
free(p);
*p = 42;            // SEGFAULT (use-after-free)
```

**진단**: `-fsanitize=address`로 실행하거나 GDB의 `backtrace`로 정확한 줄을 찾으세요.

### 오프바이원 오류 (Off-by-One Errors)

```c
// Bug: writes past end of array
int arr[10];
for (int i = 0; i <= 10; i++) {   // Should be i < 10
    arr[i] = i;
}

// Bug: string missing null terminator
char buf[5];
strncpy(buf, "Hello", 5);    // No room for '\0'!
// Fix: strncpy(buf, "Hello", sizeof(buf) - 1); buf[sizeof(buf)-1] = '\0';
```

### 초기화되지 않은 변수

```c
int sum;               // Not initialized -- contains garbage!
for (int i = 0; i < 10; i++) {
    sum += i;          // Undefined behavior
}
// Fix: int sum = 0;
```

### 버퍼 오버플로

```c
char name[10];
scanf("%s", name);     // User types "Alexander" (9 chars + '\0' = 10, just fits)
                       // User types "Christopher" -> OVERFLOW

// Fix: limit input length
scanf("%9s", name);    // Read at most 9 chars, leave room for '\0'

// Better: use fgets
fgets(name, sizeof(name), stdin);
name[strcspn(name, "\n")] = '\0';
```

---

## 8. 프로젝트 구성

프로젝트가 몇 개 파일 이상으로 커지면, 일관된 디렉토리 레이아웃이 혼란을 방지합니다:

```
my_project/
├── Makefile
├── include/           # Header files (.h)
│   ├── calc.h
│   └── utils.h
├── src/               # Source files (.c)
│   ├── main.c
│   ├── calc.c
│   └── utils.c
├── build/             # Object files and dependencies (generated)
│   ├── main.o
│   ├── calc.o
│   └── utils.o
└── tests/             # Test files
    ├── test_calc.c
    └── test_utils.c
```

### 이 레이아웃을 위한 Makefile

```makefile
CC       = gcc
CFLAGS   = -Wall -Wextra -std=c11 -g -MMD -MP
CPPFLAGS = -Iinclude

SRC_DIR  = src
BUILD_DIR = build
INC_DIR  = include

SRCS     = $(wildcard $(SRC_DIR)/*.c)
OBJS     = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS     = $(OBJS:.o=.d)
TARGET   = calculator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
```

### 헤더 가드 관례

```c
// include/calc.h
#ifndef CALC_H
#define CALC_H

int calculate(double num1, char op, double num2, double *result);

#endif  // CALC_H
```

### 컴파일 명령

```bash
# Build everything
make

# Build with debug output
make CFLAGS="-Wall -Wextra -std=c11 -g -O0 -DDEBUG"

# Clean and rebuild
make clean && make

# Build only if changed
make    # make tracks timestamps automatically
```

---

## 연습문제

1. **처음부터 Makefile 작성**: `main.c`, `math_ops.c`, `math_ops.h` 세 파일이 있습니다. 변수(`CC`, `CFLAGS`), `.o` 파일을 위한 패턴 규칙, 자동 의존성 생성(`-MMD -MP`), `clean`/`all` 가짜 타겟이 있는 완전한 Makefile을 작성하세요.

2. **새니타이저 탐정**: 다음 프로그램에는 숨겨진 버그가 있습니다. `-fsanitize=address`와 `-fsanitize=undefined`로 컴파일하고, 실행하고, 새니타이저 출력을 읽고, 버그를 수정하세요:
   ```c
   #include <stdio.h>
   #include <string.h>
   int main(void) {
       char buf[8];
       strcpy(buf, "overflow!");
       printf("%s\n", buf);
       return 0;
   }
   ```

3. **DEBUG 매크로**: `DEBUG`가 정의되면 파일명, 줄 번호, 서식화된 메시지를 stderr에 출력하고, 그렇지 않으면 아무것도 확장하지 않는 `DBG(fmt, ...)` 매크로를 정의하는 `debug.h` 헤더를 작성하세요. 작은 프로그램에서 사용하고 `-DDEBUG`와 릴리스 빌드 모두 올바르게 작동하는지 확인하세요.

4. **GDB 연습**: 루프를 사용하여 사용자가 입력한 수의 팩토리얼을 계산하는 프로그램을 작성하세요. `-g -O0`으로 컴파일하고, GDB를 시작하고, 루프 안에 브레이크포인트를 설정하고, `next`와 `print`로 각 반복에서 누적기가 증가하는 것을 관찰하세요. 사용한 GDB 명령을 기록하세요.

5. **프로젝트 레이아웃**: 이전 연습문제(예: 계산기)를 `src/`, `include/`, `build/`, `tests/` 디렉토리 구조로 재구성하세요. `src/`에서 소스를 컴파일하고, `build/`에 오브젝트 파일을 배치하고, `include/`에서 헤더를 포함하는 Makefile을 작성하세요. `make clean && make`가 작동하는 바이너리를 생성하는지 확인하세요.

---

## 다음 단계

[프로젝트: 계산기](./13_Project_Calculator.md) -- 대화형 계산기를 단계별로 구축하여 빌드 도구 지식을 실전에 적용하세요.
