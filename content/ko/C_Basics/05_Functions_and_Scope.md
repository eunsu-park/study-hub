# 함수와 스코프

**이전**: [제어 흐름](./04_Control_Flow.md) | **다음**: [배열과 문자열](./06_Arrays_and_Strings.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 함수 프로토타입을 선언하고 매개변수와 반환값이 있는 함수를 정의한다
2. 값에 의한 호출(call-by-value) 의미를 설명하고 포인터를 사용하여 참조에 의한 호출(call-by-reference)을 시뮬레이션한다
3. 스코프 규칙(블록, 함수, 파일)과 지속성을 위한 `static` 키워드를 적용한다
4. 적절한 기저 조건(base case)을 가진 재귀 함수를 작성한다
5. 가변 인자 함수의 기초를 사용한다 (`stdarg.h` 간단 소개)

---

함수는 C 코드를 관리 가능하고 재사용 가능한 조각으로 구성하는 주요 메커니즘입니다. 잘 설계된 함수는 한 가지 일을 하고, 그것을 잘 수행하며, 이름과 매개변수를 통해 목적을 전달합니다. C가 인자를 전달하는 방식, 변수가 스코프되는 방식, 값이 메모리에 저장되는 위치를 이해하면 훨씬 효과적인 C 프로그래머가 될 것입니다.

## 1. 함수 선언과 정의

C 함수는 두 부분으로 구성됩니다: 존재를 알리는 **선언**(프로토타입)과 구현을 제공하는 **정의**.

### 구문

```c
/* 선언 (프로토타입) — 컴파일러에 함수의 시그니처를 알림 */
return_type function_name(parameter_list);

/* 정의 — 실제 구현 제공 */
return_type function_name(parameter_list) {
    /* 본문 */
    return value;  /* void 함수에서는 생략 */
}
```

### 완전한 예제

```c
#include <stdio.h>

/* 선언 (프로토타입) */
int add(int a, int b);
void greet(const char *name);

int main(void) {
    int sum = add(3, 4);
    printf("3 + 4 = %d\n", sum);  /* 7 */

    greet("Alice");  /* Hello, Alice! */

    return 0;
}

/* 정의 */
int add(int a, int b) {
    return a + b;
}

void greet(const char *name) {
    printf("Hello, %s!\n", name);
}
```

### void 함수

아무것도 반환하지 않는 함수는 반환 타입으로 `void`를 사용합니다. 매개변수가 없는 함수는 매개변수 목록에 `void`를 사용합니다.

```c
#include <stdio.h>

/* 아무것도 받지 않고, 아무것도 반환하지 않음 */
void print_separator(void) {
    printf("========================\n");
}

/* 매개변수를 받고, 아무것도 반환하지 않음 */
void print_range(int start, int end) {
    for (int i = start; i <= end; i++) {
        printf("%d ", i);
    }
    printf("\n");
}

int main(void) {
    print_separator();
    print_range(1, 10);
    print_separator();
    return 0;
}
```

### 여러 return 문

함수는 여러 `return` 문을 가질 수 있습니다. 도달한 첫 번째 문에서 실행이 종료됩니다.

```c
int absolute(int n) {
    if (n >= 0) {
        return n;
    }
    return -n;
}

char grade(int score) {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
}
```

---

## 2. 매개변수와 반환값

### 값에 의한 전달(Pass by Value)

C는 항상 인자를 **값에 의해** 전달합니다. 함수는 각 인자의 복사본을 받으므로, 함수 내에서 매개변수를 수정해도 원래 변수에는 영향이 없습니다.

```c
#include <stdio.h>

void try_to_modify(int x) {
    x = 999;  /* 지역 복사본만 수정 */
    printf("Inside function: x = %d\n", x);
}

int main(void) {
    int num = 42;
    try_to_modify(num);
    printf("After function:  num = %d\n", num);  /* 여전히 42 */
    return 0;
}
```

### 포인터를 사용한 참조에 의한 전달 시뮬레이션

호출자의 변수를 수정하려면 **포인터**를 전달합니다. (포인터는 레슨 07에서 자세히 다루지만, 패턴은 여기서 소개합니다.)

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main(void) {
    int x = 10, y = 20;
    printf("Before: x=%d y=%d\n", x, y);

    swap(&x, &y);  /* 주소 전달 */
    printf("After:  x=%d y=%d\n", x, y);  /* x=20 y=10 */

    return 0;
}
```

### 포인터를 통한 여러 값 반환

함수는 직접 하나의 값만 반환할 수 있으므로, 추가 결과를 반환하려면 출력 매개변수(포인터)를 사용합니다.

```c
#include <stdio.h>

void divide(int dividend, int divisor, int *quotient, int *remainder) {
    *quotient  = dividend / divisor;
    *remainder = dividend % divisor;
}

int main(void) {
    int q, r;
    divide(17, 5, &q, &r);
    printf("17 / 5 = %d remainder %d\n", q, r);  /* 3 remainder 2 */
    return 0;
}
```

### 구조체 반환

관련 반환값을 그룹화하려면 구조체를 반환할 수 있습니다 (레슨 08에서 다룸):

```c
#include <stdio.h>

typedef struct {
    int quot;
    int rem;
} DivResult;

DivResult divide2(int a, int b) {
    DivResult result = { a / b, a % b };
    return result;
}

int main(void) {
    DivResult dr = divide2(17, 5);
    printf("17 / 5 = %d remainder %d\n", dr.quot, dr.rem);
    return 0;
}
```

---

## 3. 함수 프로토타입

**프로토타입**은 함수가 정의되기 전에 컴파일러에 함수의 반환 타입과 매개변수 타입을 알립니다. 이는 소스 파일에서 정의가 나타나기 전에 함수가 호출될 때 필요합니다.

### 프로토타입이 중요한 이유

```c
#include <stdio.h>

/* 프로토타입이 없으면, main()에서 호출을 만났을 때
   컴파일러가 add()를 모릅니다. C89에서는 암시적 선언이
   발생했지만 (C99 이후 제거됨), C99 이후에서는 오류입니다. */

/* 프로토타입 */
int add(int a, int b);

int main(void) {
    printf("%d\n", add(3, 4));  /* OK — 컴파일러가 add의 시그니처를 앎 */
    return 0;
}

int add(int a, int b) {
    return a + b;
}
```

### 헤더 파일 관례

다중 파일 프로젝트에서는 프로토타입을 헤더 파일(`.h`)에, 정의를 소스 파일(`.c`)에 넣습니다.

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);
double average(const int *arr, int n);

#endif /* MATH_UTILS_H */
```

```c
/* math_utils.c */
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

double average(const int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (double)sum / n;
}
```

```c
/* main.c */
#include <stdio.h>
#include "math_utils.h"

int main(void) {
    printf("3 + 4 = %d\n", add(3, 4));
    printf("3 * 4 = %d\n", multiply(3, 4));

    int data[] = {10, 20, 30, 40, 50};
    printf("average = %.1f\n", average(data, 5));
    return 0;
}
```

---

## 4. 스코프 규칙

**스코프**는 변수가 어디에서 보이고 접근 가능한지를 결정합니다. C에는 여러 수준의 스코프가 있습니다.

### 블록 스코프

블록 `{}` 안에서 선언된 변수는 해당 블록 내에서만 보입니다.

```c
#include <stdio.h>

int main(void) {
    int x = 10;

    {
        int y = 20;           /* y는 이 블록에서만 보임 */
        printf("x=%d y=%d\n", x, y);  /* OK */
    }

    /* printf("y=%d\n", y);  — 오류: y가 여기서 보이지 않음 */

    /* for 루프 변수는 블록 스코프를 가짐 (C99 이후) */
    for (int i = 0; i < 3; i++) {
        printf("%d ", i);
    }
    /* printf("%d\n", i);  — 오류: i가 스코프 밖 */
    printf("\n");

    return 0;
}
```

### 함수 스코프

레이블(`goto`와 함께 사용)은 함수 스코프를 가집니다 -- 블록 중첩에 관계없이 전체 함수에서 보입니다.

### 파일 스코프 (전역 변수)

모든 함수 밖에서 선언된 변수는 **파일 스코프**를 가집니다. 선언 지점부터 파일 끝까지 접근 가능합니다.

```c
#include <stdio.h>

int global_count = 0;   /* 파일 스코프 — 아래 어디서든 접근 가능 */

void increment(void) {
    global_count++;
}

int main(void) {
    increment();
    increment();
    printf("count = %d\n", global_count);  /* 2 */
    return 0;
}
```

> **모범 사례**: 전역 변수의 사용을 최소화하세요. 코드의 추론, 테스트, 유지보수가 어려워집니다. 함수 매개변수를 통해 데이터를 전달하는 것을 선호하세요.

### 섀도잉(Shadowing)

내부 스코프는 외부 스코프와 같은 이름의 변수를 선언하여 **섀도잉**할 수 있습니다.

```c
#include <stdio.h>

int x = 100;  /* 전역 */

int main(void) {
    int x = 50;  /* 전역 x를 섀도잉 */
    printf("x = %d\n", x);  /* 50 */

    {
        int x = 10;  /* main()의 x를 섀도잉 */
        printf("x = %d\n", x);  /* 10 */
    }

    printf("x = %d\n", x);  /* 50 — main의 x로 복귀 */
    return 0;
}
```

> **경고**: 섀도잉은 합법적이지만 혼란을 줍니다. `-Wshadow`로 컴파일하면 이에 대한 경고를 활성화합니다.

---

## 5. 저장 클래스(Storage Class)

저장 클래스는 변수의 **수명**과 **링키지**를 제어합니다.

| 키워드 | 스코프 | 수명 | 기본값 | 비고 |
|--------|--------|------|--------|------|
| `auto` | 블록 | 블록 지속 | 미정의 (쓰레기) | 지역 변수의 기본값; 키워드는 거의 사용하지 않음 |
| `static` (지역) | 블록 | 프로그램 지속 | 0 | 호출 간 값 유지 |
| `static` (파일) | 파일 | 프로그램 지속 | 0 | 파일 외부에서 보이지 않음 |
| `extern` | 파일+ | 프로그램 지속 | 0 | 한 파일에서 선언, 다른 파일에서 접근 가능 |
| `register` | 블록 | 블록 지속 | 미정의 | CPU 레지스터에 저장 힌트 (오늘날 거의 사용하지 않음) |

### static 지역 변수

`static` 지역 변수는 한 번만 초기화되며 함수 호출 간에 유지됩니다.

```c
#include <stdio.h>

int next_id(void) {
    static int id = 0;  /* 한 번만 초기화; 호출 간 유지 */
    id++;
    return id;
}

int main(void) {
    printf("ID: %d\n", next_id());  /* 1 */
    printf("ID: %d\n", next_id());  /* 2 */
    printf("ID: %d\n", next_id());  /* 3 */
    return 0;
}
```

### 파일 스코프에서의 static

`static` 전역 변수나 함수는 해당 번역 단위(소스 파일)에 **내부적**입니다. 다른 파일에서는 `extern`으로도 접근할 수 없습니다.

```c
/* helpers.c */
static int internal_counter = 0;  /* helpers.c에서만 보임 */

static void helper(void) {        /* helpers.c에서만 호출 가능 */
    internal_counter++;
}

void public_function(void) {      /* 다른 파일에서 보임 */
    helper();
}
```

### extern

`extern`은 다른 파일에서 **정의된** 변수를 선언합니다.

```c
/* config.c */
int max_connections = 100;  /* 정의 */

/* main.c */
#include <stdio.h>

extern int max_connections;  /* 선언 — config.c의 정의를 사용 */

int main(void) {
    printf("Max connections: %d\n", max_connections);
    return 0;
}
```

두 파일을 함께 컴파일: `gcc main.c config.c -o app`

---

## 6. 재귀

재귀 함수는 자기 자신을 호출합니다. 모든 재귀 함수에는 다음이 필요합니다:

1. 재귀를 멈추는 **기저 조건**.
2. 기저 조건을 향해 나아가는 **재귀 조건**.

### 팩토리얼

```c
#include <stdio.h>

long long factorial(int n) {
    if (n <= 1) {
        return 1;          /* 기저 조건 */
    }
    return n * factorial(n - 1);  /* 재귀 조건 */
}

int main(void) {
    for (int i = 0; i <= 10; i++) {
        printf("%2d! = %lld\n", i, factorial(i));
    }
    return 0;
}
```

### 피보나치

```c
#include <stdio.h>

/* 단순 재귀 피보나치 — 지수 시간, 설명용 */
int fib(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return fib(n - 1) + fib(n - 2);
}

/* 반복 버전 — 선형 시간 */
int fib_iter(int n) {
    if (n <= 0) return 0;
    int prev = 0, curr = 1;
    for (int i = 2; i <= n; i++) {
        int next = prev + curr;
        prev = curr;
        curr = next;
    }
    return curr;
}

int main(void) {
    printf("Recursive: fib(10) = %d\n", fib(10));       /* 55 */
    printf("Iterative: fib(10) = %d\n", fib_iter(10));  /* 55 */
    return 0;
}
```

### 스택 사용과 한계

각 재귀 호출은 호출 스택에 **스택 프레임**을 추가합니다. 너무 많은 중첩 호출은 **스택 오버플로우**를 유발합니다.

```c
#include <stdio.h>

void count_down(int n) {
    printf("%d\n", n);
    if (n > 0) {
        count_down(n - 1);
    }
}

int main(void) {
    count_down(10);        /* 정상 */
    /* count_down(1000000);  — 스택 오버플로우! */
    return 0;
}
```

### 꼬리 재귀(Tail Recursion)

재귀 호출이 마지막 연산일 때 이를 **꼬리 재귀**라 합니다. 일부 컴파일러는 (최적화 시) 꼬리 재귀를 루프로 변환하여 스택 성장을 제거할 수 있습니다.

```c
/* 꼬리 재귀 팩토리얼 */
long long factorial_tail(int n, long long acc) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);  /* 꼬리 위치 */
}

/* 래퍼 */
long long factorial2(int n) {
    return factorial_tail(n, 1);
}
```

> **참고**: C 표준은 꼬리 호출 최적화를 요구하지 않지만, GCC와 Clang은 `-O2` 이상에서 이를 수행합니다.

---

## 7. 가변 인자 함수(Variadic Function)

가변 인자 함수는 가변 개수의 인자를 받습니다. 가장 친숙한 예는 `printf`입니다. 직접 작성하려면 `<stdarg.h>`를 사용합니다.

### stdarg.h 매크로

| 매크로 | 용도 |
|--------|------|
| `va_list` | 가변 인자 상태를 보유하는 타입 |
| `va_start(ap, last_fixed)` | 마지막 고정 매개변수 이후 `ap` 초기화 |
| `va_arg(ap, type)` | 다음 인자를 `type`으로 가져옴 |
| `va_end(ap)` | 정리 |

### 예제: 가변 인자 합계

```c
#include <stdio.h>
#include <stdarg.h>

/* count: 뒤따르는 정수의 수 */
int sum(int count, ...) {
    va_list ap;
    va_start(ap, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(ap, int);
    }

    va_end(ap);
    return total;
}

int main(void) {
    printf("sum(3, 10, 20, 30)  = %d\n", sum(3, 10, 20, 30));   /* 60 */
    printf("sum(5, 1,2,3,4,5)   = %d\n", sum(5, 1, 2, 3, 4, 5)); /* 15 */
    return 0;
}
```

### 예제: 커스텀 로거

```c
#include <stdio.h>
#include <stdarg.h>

void log_message(const char *level, const char *fmt, ...) {
    printf("[%s] ", level);

    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);   /* vprintf는 va_list를 받음 */
    va_end(ap);

    printf("\n");
}

int main(void) {
    log_message("INFO",  "Server started on port %d", 8080);
    log_message("WARN",  "Memory usage at %d%%", 85);
    log_message("ERROR", "Failed to open '%s': code %d", "data.csv", -1);
    return 0;
}
```

> **주의**: 가변 인자 함수는 가변 인자에 대한 타입 검사가 없습니다. 잘못된 타입을 전달하면 미정의 동작이 됩니다. 드물게 사용하고 예상되는 타입을 주의 깊게 문서화하세요.

---

## 8. 함수 포인터(Function Pointer)

**함수 포인터**는 함수의 주소를 저장하고 간접적으로 호출할 수 있게 합니다. 콜백(callback), 디스패치 테이블, 플러그 가능한 동작을 위한 C의 메커니즘입니다.

### 선언과 기본 사용

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

int main(void) {
    /* 함수 포인터 선언: 반환 타입 (*이름)(매개변수 타입) */
    int (*fp)(int, int) = add;   /* fp가 add를 가리킴 */

    printf("add via pointer: %d\n", fp(3, 4));  /* 7 */

    fp = mul;                    /* 다른 함수로 재할당 */
    printf("mul via pointer: %d\n", fp(3, 4));  /* 12 */

    return 0;
}
```

### 가독성을 위한 typedef

실제 코드에서 함수 포인터 문법은 다루기 불편해집니다. `typedef`로 타입에 깔끔한 이름을 붙일 수 있습니다.

```c
typedef int (*operation_t)(int, int);   /* operation_t가 타입이 됨 */

operation_t op = add;
printf("%d\n", op(10, 5));  /* 15 */
```

### 콜백(Callback) 패턴

함수 포인터를 다른 함수에 전달하는 것이 **콜백 패턴**입니다 — `qsort`, `bsearch`, 시그널 핸들러 및 많은 라이브러리 API의 기반입니다.

```c
#include <stdio.h>

typedef int (*operation_t)(int, int);

int compute(int x, int y, operation_t op) {
    return op(x, y);
}

int subtract(int a, int b) { return a - b; }

int main(void) {
    printf("compute(10, 3, add)      = %d\n", compute(10, 3, add));       /* 13 */
    printf("compute(10, 3, subtract) = %d\n", compute(10, 3, subtract));  /* 7  */
    printf("compute(10, 3, mul)      = %d\n", compute(10, 3, mul));       /* 30 */
    return 0;
}
```

> **표준 라이브러리와의 연결**: `qsort`는 정확히 이 패턴을 사용합니다 — 비교 함수 포인터 `int (*compar)(const void *, const void *)`를 받아 어떤 데이터 타입이든 원하는 순서 규칙으로 정렬할 수 있습니다.

---

## 연습문제

### 연습문제 1: 수학 라이브러리

다음 함수를 작성하고 `main`에서 테스트하세요:

1. `int power(int base, int exp)` — 루프를 사용하여 base^exp 계산 (exp >= 0 가정).
2. `int gcd(int a, int b)` — 유클리드 알고리즘(반복)으로 최대공약수 계산.
3. `int is_prime(int n)` — `n`이 소수이면 1, 아니면 0 반환.
4. `void print_primes(int start, int end)` — [start, end] 범위의 모든 소수 출력.

파일 상단이나 별도의 헤더에 프로토타입을 배치하세요.

### 연습문제 2: 스코프 탐정

실행하지 않고 이 프로그램의 출력을 예측한 다음 확인하세요:

```c
#include <stdio.h>

int x = 1;

void f(void) {
    int x = 10;
    printf("f: x = %d\n", x);
    {
        int x = 20;
        printf("f inner: x = %d\n", x);
    }
    printf("f after block: x = %d\n", x);
}

void g(void) {
    printf("g: x = %d\n", x);
    x = 5;
}

int main(void) {
    printf("main: x = %d\n", x);
    f();
    printf("main after f: x = %d\n", x);
    g();
    printf("main after g: x = %d\n", x);
    return 0;
}
```

각 `printf` 옆에 예상 출력과 어떤 `x`가 참조되는지 설명하는 주석을 작성하세요.

### 연습문제 3: 정적 카운터

`static` 지역 변수를 사용하여 호출할 때마다 다른 정수를 반환하는 함수 `int unique_id(void)` (1, 2, 3, ...)를 작성하세요. 그런 다음 카운터를 0으로 리셋하는 두 번째 함수 `void reset_id(void)`를 작성하세요. `reset_id`가 `unique_id` 안의 `static` 변수에 직접 접근할 수 있습니까? 불가능하다면, 리셋을 지원하기 위해 코드를 어떻게 재구성하겠습니까?

### 연습문제 4: 재귀적 거듭제곱

거듭제곱 함수의 재귀 버전을 작성하세요: `long long power_rec(int base, int exp)`. 세 가지 경우를 처리하세요:

1. `exp == 0`이면 1 반환
2. `exp`가 짝수: `base^exp = (base^(exp/2))^2`
3. `exp`가 홀수: `base^exp = base * base^(exp-1)`

이것은 **제곱에 의한 거듭제곱**(exponentiation by squaring)이라 하며 O(log n) 시간에 실행됩니다. `power_rec(2, 30)` (예상: 1073741824)으로 테스트하세요.

### 연습문제 5: 미니 printf

세 가지 형식 지정자만 지원하는 간소화된 `my_printf(const char *fmt, ...)`를 작성하세요:

- `%d` — `int` 출력
- `%s` — `char *` 출력
- `%c` — `char` 출력

`stdarg.h`를 사용하여 형식 문자열을 문자 단위로 순회하세요. `%`를 만나면 다음 문자를 읽어 타입을 판단하고, `va_arg`로 값을 가져와 출력하세요. 다음으로 테스트하세요:

```c
my_printf("Name: %s, Age: %d, Grade: %c\n", "Alice", 25, 'A');
```

---

## 다음 단계

함수를 통해 프로그램을 재사용 가능한 구성 요소로 구조화할 수 있게 되었습니다. 다음으로 [배열과 문자열](./06_Arrays_and_Strings.md)에서 데이터 컬렉션을 다루는 방법을 배워봅시다!
