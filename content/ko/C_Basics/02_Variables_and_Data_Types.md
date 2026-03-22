# 변수와 데이터 타입

**이전**: [환경 설정](./01_Environment_Setup.md) | **다음**: [연산자와 표현식](./03_Operators_and_Expressions.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C의 모든 기본 타입으로 변수를 선언하고 초기화한다
2. `char`, `short`, `int`, `long`, `long long`, `float`, `double`의 크기와 범위를 설명한다
3. unsigned 수식어를 적용하고 2의 보수(Two's Complement) 표현을 이해한다
4. `sizeof`로 타입 크기를 확인하고 `const`/`volatile` 한정자를 사용한다
5. 암시적/명시적 타입 변환을 수행하고 잘림(truncation) 위험을 식별한다
6. 각 데이터 타입에 맞는 올바른 `printf` 형식 지정자를 선택한다

---

모든 C 프로그램은 메모리에 저장된 데이터를 다룹니다. 변수가 언제든지 타입을 바꿀 수 있는 동적 타입 언어와 달리, C에서는 사용하기 전에 모든 변수의 타입을 선언해야 합니다. 이러한 규율은 컴파일러가 적절한 크기의 메모리를 할당하고 효율적인 기계어 코드를 생성하는 데 필요한 정보를 제공합니다. C의 타입 시스템을 마스터하는 것은 포인터, 구조체, 그리고 이후의 모든 내용을 이해하기 위한 기초입니다.

## 1. 변수와 선언

C에서 변수는 특정 타입을 가진 이름이 붙은 메모리 영역입니다. 변수를 사용하기 전에 반드시 선언해야 합니다.

### 선언 구문

```c
type name;            /* 선언만 (초기화되지 않음) */
type name = value;    /* 선언과 동시에 초기화  */
```

```c
#include <stdio.h>

int main(void) {
    int age;              /* 선언만 — 쓰레기 값 포함 */
    int score = 95;       /* 선언 및 초기화 */
    double pi = 3.14159;  /* 부동소수점 변수 */
    char grade = 'A';     /* 단일 문자 */

    age = 25;             /* 선언 후 대입 */

    printf("age = %d, score = %d\n", age, score);
    printf("pi = %f, grade = %c\n", pi, grade);
    return 0;
}
```

### 이름 규칙

| 규칙 | 예시 | 유효? |
|------|------|-------|
| 문자 또는 밑줄로 시작해야 함 | `count`, `_temp` | 예 |
| 문자, 숫자, 밑줄 포함 가능 | `score2`, `max_val` | 예 |
| 숫자로 시작할 수 없음 | `2nd_place` | 아니오 |
| C 키워드를 사용할 수 없음 | `int`, `return` | 아니오 |
| 대소문자 구분 | `Count`와 `count`는 다른 변수 | -- |

### 이름 규약

```c
int student_count;    /* snake_case — C에서 일반적 */
int studentCount;     /* camelCase  — C에서는 덜 일반적 */
#define MAX_SIZE 100  /* UPPER_CASE — 상수와 매크로에 사용 */
```

### 다중 선언

```c
int a, b, c;           /* 세 개의 int, 모두 초기화되지 않음 */
int x = 1, y = 2;     /* 두 개의 int, 모두 초기화됨 */
double width = 10.5, height = 20.0;
```

> **경고**: 초기화되지 않은 지역 변수는 예측할 수 없는 쓰레기 값을 포함합니다. 변수를 읽기 전에 항상 초기화하세요.

---

## 2. 정수 타입

C는 크기와 범위가 다른 여러 정수 타입을 제공합니다. 정확한 크기는 플랫폼에 따라 다르지만, C 표준은 최소 범위를 보장합니다.

### 정수 타입 요약

| 타입 | 최소 크기 | 일반 크기 (64비트) | 일반 범위 |
|------|----------|-------------------|----------|
| `char` | 1바이트 | 1바이트 | -128 ~ 127 |
| `short` | 2바이트 | 2바이트 | -32,768 ~ 32,767 |
| `int` | 2바이트 | 4바이트 | -2,147,483,648 ~ 2,147,483,647 |
| `long` | 4바이트 | 8바이트 (Linux/macOS), 4바이트 (Windows) | 플랫폼 의존 |
| `long long` | 8바이트 | 8바이트 | -9.2 x 10^18 ~ 9.2 x 10^18 |

### 보장된 순서

C 표준은 다음을 보장합니다:

```
sizeof(char) <= sizeof(short) <= sizeof(int) <= sizeof(long) <= sizeof(long long)
```

### 정수 다루기

```c
#include <stdio.h>
#include <limits.h>   /* INT_MIN, INT_MAX 등 */

int main(void) {
    char   c = 'Z';           /* 1바이트 — 작은 정수도 저장 가능 */
    short  s = 1000;          /* 최소 2바이트 */
    int    i = 42;            /* 최소 2바이트, 보통 4바이트 */
    long   l = 100000L;       /* 최소 4바이트 — L 접미사에 주목 */
    long long ll = 9000000000000LL;  /* 최소 8바이트 — LL 접미사 */

    printf("char:      %c  (value: %d, size: %zu bytes)\n", c, c, sizeof(c));
    printf("short:     %hd (size: %zu bytes)\n", s, sizeof(s));
    printf("int:       %d  (size: %zu bytes)\n", i, sizeof(i));
    printf("long:      %ld (size: %zu bytes)\n", l, sizeof(l));
    printf("long long: %lld (size: %zu bytes)\n", ll, sizeof(ll));

    printf("\nint range: %d to %d\n", INT_MIN, INT_MAX);
    return 0;
}
```

### 정수 오버플로우(Overflow)

정수가 범위를 초과하면, signed인지 unsigned인지에 따라 동작이 달라집니다:

```c
#include <stdio.h>
#include <limits.h>

int main(void) {
    int max = INT_MAX;
    printf("INT_MAX     = %d\n", max);
    printf("INT_MAX + 1 = %d\n", max + 1);  /* signed의 경우 미정의 동작! */

    return 0;
}
```

> **중요**: 부호 있는 정수(signed integer) 오버플로우는 C에서 **미정의 동작(undefined behavior)**입니다. 컴파일러는 오버플로우 검사를 최적화해서 제거하는 것을 포함하여 무엇이든 할 수 있습니다. 부호 있는 오버플로우가 순환(wrap around)한다고 절대 가정하지 마세요.

---

## 2a. 고정 폭 정수 타입(Fixed-Width Integer Types)

`int`, `long` 등의 크기는 플랫폼에 따라 다르기 때문에 정확한 크기가 중요한 경우 이식성 문제가 생깁니다. `<stdint.h>` 헤더(C99 이상)는 크기가 보장된 타입을 제공합니다.

### 타입과 한계

```c
#include <stdint.h>
#include <inttypes.h>  /* PRId32, PRIu64 등 — 형식 지정자 매크로 */
#include <stdio.h>

int main(void) {
    int8_t   a = -128;          /* 정확히 8비트, 부호 있음  */
    uint8_t  b = 255;           /* 정확히 8비트, 부호 없음 */
    int16_t  c = 32767;
    int32_t  d = INT32_MAX;     /* 2,147,483,647            */
    uint64_t e = UINT64_MAX;    /* 18,446,744,073,709,551,615 */

    printf("int32_t max : %" PRId32 "\n", d);
    printf("uint64_t max: %" PRIu64 "\n", e);

    return 0;
}
```

| 부호 있음 | 부호 없음 | 폭 | 범위 (부호 있음) |
|-----------|-----------|----|--------------------|
| `int8_t` | `uint8_t` | 8비트 | -128 ~ 127 |
| `int16_t` | `uint16_t` | 16비트 | -32,768 ~ 32,767 |
| `int32_t` | `uint32_t` | 32비트 | ±2.1 × 10⁹ |
| `int64_t` | `uint64_t` | 64비트 | ±9.2 × 10¹⁸ |

### 고정 폭 타입을 사용하는 경우

- **바이너리 파일 형식**: 4바이트 필드를 읽고 쓸 때 `int`가 아닌 `int32_t`를 사용해야 합니다.
- **네트워크 프로토콜**: 프로토콜 헤더는 정확한 바이트 폭을 지정하며, `uint16_t`는 2바이트 포트 번호에 해당합니다.
- **하드웨어 레지스터**: 32비트 메모리 매핑 레지스터는 `uint32_t`로 접근해야 합니다.
- **크로스 플랫폼 코드**: `int`가 2바이트인지 4바이트인지에 따라 동작이 달라지는 모든 곳.

> **참고**: 정확한 크기가 중요하지 않은 일반 산술 연산에는 평범한 `int`가 여전히 권장됩니다 — 컴파일러가 가장 효율적인 네이티브 크기를 선택할 수 있습니다. 정확한 표현이 핵심인 상황에만 고정 폭 타입을 사용하세요.

---

## 3. 부호 없는 정수(Unsigned Integer)

`unsigned` 키워드는 정수를 음이 아닌 값으로 제한하여, 양수 범위를 사실상 두 배로 늘립니다.

### 부호 없는 타입 범위

| 타입 | 일반 크기 | 범위 |
|------|----------|------|
| `unsigned char` | 1바이트 | 0 ~ 255 |
| `unsigned short` | 2바이트 | 0 ~ 65,535 |
| `unsigned int` | 4바이트 | 0 ~ 4,294,967,295 |
| `unsigned long` | 4 또는 8바이트 | 0 ~ 2^32-1 또는 2^64-1 |
| `unsigned long long` | 8바이트 | 0 ~ 18,446,744,073,709,551,615 |

### 2의 보수(Two's Complement)

현대 시스템은 부호 있는 정수를 표현하기 위해 **2의 보수**를 사용합니다:

- 최상위 비트(MSB)가 부호 비트입니다: 0 = 양수, 1 = 음수.
- 숫자를 부정하려면: 모든 비트를 반전시킨 다음 1을 더합니다.
- 8비트 `char`의 경우: `01111111` = 127, `10000000` = -128.

```c
#include <stdio.h>

int main(void) {
    unsigned int u = 0;
    printf("u     = %u\n", u);
    printf("u - 1 = %u\n", u - 1);  /* 4294967295로 순환 (정의된 동작!) */

    /* unsigned 오버플로우는 정의된 동작: 2^N으로 나눈 나머지로 순환 */
    unsigned char byte = 255;
    byte = byte + 1;
    printf("255 + 1 as unsigned char = %u\n", byte);  /* 0 */

    return 0;
}
```

### unsigned를 사용하는 경우

- 비트 조작과 플래그
- 배열 인덱스 (`size_t` 사용을 권장)
- 본질적으로 음이 아닌 값 (예: 바이트 수)
- unsigned 타입을 사용하는 API와의 인터페이스

> **주의**: signed와 unsigned를 비교에서 혼합하면 놀라운 결과가 발생할 수 있습니다:
>
> ```c
> int a = -1;
> unsigned int b = 1;
> if (a < b) {
>     printf("Expected\n");
> } else {
>     printf("Surprise!\n");  /* 이것이 출력됨! -1이 큰 unsigned 값으로 변환됨 */
> }
> ```

---

## 4. 부동소수점 타입

C는 실수를 표현하기 위한 세 가지 부동소수점 타입을 제공합니다.

| 타입 | 일반 크기 | 정밀도 | 범위 (근사) |
|------|----------|--------|------------|
| `float` | 4바이트 | ~7자리 | ±3.4 x 10^38 |
| `double` | 8바이트 | ~15자리 | ±1.7 x 10^308 |
| `long double` | 8-16바이트 | ~18-21자리 | 플랫폼 의존 |

### IEEE 754 기초

부동소수점 숫자는 세 부분으로 저장됩니다: **부호**, **지수**, **가수**(유효숫자).

- `float`: 부호 1비트 + 지수 8비트 + 가수 23비트 = 32비트
- `double`: 부호 1비트 + 지수 11비트 + 가수 52비트 = 64비트

### 부동소수점 다루기

```c
#include <stdio.h>
#include <float.h>   /* FLT_MIN, FLT_MAX, DBL_EPSILON 등 */

int main(void) {
    float  f = 3.14f;        /* float 리터럴에는 f 접미사 */
    double d = 3.141592653589793;  /* 기본 리터럴 타입은 double */
    long double ld = 3.14159265358979323846L;  /* L 접미사 */

    printf("float:       %.7f  (size: %zu bytes)\n", f, sizeof(f));
    printf("double:      %.15f (size: %zu bytes)\n", d, sizeof(d));
    printf("long double: %.18Lf (size: %zu bytes)\n", ld, sizeof(ld));

    /* 정밀도 한계 */
    printf("\nfloat precision:  %d digits\n", FLT_DIG);
    printf("double precision: %d digits\n", DBL_DIG);
    return 0;
}
```

### 부동소수점 주의사항

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    /* 동등 비교는 신뢰할 수 없음 */
    double a = 0.1 + 0.2;
    double b = 0.3;
    printf("0.1 + 0.2 == 0.3? %d\n", a == b);  /* 0 (거짓!) */

    /* 비교에는 엡실론을 사용 */
    double epsilon = 1e-9;
    if (fabs(a - b) < epsilon) {
        printf("Approximately equal\n");  /* 이것이 출력됨 */
    }

    /* 정수 나눗셈 함정 */
    double ratio = 1 / 3;       /* 0.000000 — 정수 나눗셈! */
    double correct = 1.0 / 3.0; /* 0.333333 — 부동소수점 나눗셈 */
    printf("1/3   = %f\n", ratio);
    printf("1.0/3 = %f\n", correct);

    return 0;
}
```

---

## 5. 타입 한정자(Type Qualifier)

타입 한정자는 변수가 접근되거나 최적화되는 방식을 수정합니다.

### const

`const` 한정자는 초기화 후 변수를 읽기 전용으로 만듭니다.

```c
#include <stdio.h>

int main(void) {
    const int MAX_STUDENTS = 100;
    const double PI = 3.14159265358979;

    printf("Max students: %d\n", MAX_STUDENTS);
    /* MAX_STUDENTS = 200;  — 컴파일러 오류: const 변수에 대입 */

    /* const와 포인터 (포인터 레슨에서 자세히 다룸) */
    const char *greeting = "Hello";  /* const char를 가리키는 포인터 */
    /* greeting[0] = 'h';  — 오류: const 데이터를 수정할 수 없음 */

    return 0;
}
```

### volatile

`volatile` 한정자는 변수가 언제든지 변경될 수 있음을 컴파일러에 알립니다(예: 하드웨어 레지스터, 시그널 핸들러). 따라서 읽기를 최적화하여 제거하지 않아야 합니다.

```c
volatile int sensor_value;  /* 하드웨어에 의해 변경될 수 있음 */

/* 컴파일러는 매번 sensor_value를 다시 읽으며, 캐시하지 않음 */
while (sensor_value == 0) {
    /* 센서가 트리거될 때까지 대기 */
}
```

### static 지역 변수

`static` 지역 변수는 함수 호출 간에 값을 유지합니다.

```c
#include <stdio.h>

void counter(void) {
    static int count = 0;  /* 한 번만 초기화됨 */
    count++;
    printf("Called %d times\n", count);
}

int main(void) {
    counter();  /* Called 1 times */
    counter();  /* Called 2 times */
    counter();  /* Called 3 times */
    return 0;
}
```

---

## 6. 타입 변환

C는 두 가지 방법으로 타입 변환을 수행합니다: **암시적**(자동)과 **명시적**(캐스트).

### 암시적 변환 (승격)

서로 다른 타입의 피연산자가 표현식에 나타나면, 컴파일러는 "작은" 타입을 "큰" 타입으로 승격시킵니다.

```
char/short → int → unsigned int → long → unsigned long → long long → float → double → long double
```

```c
#include <stdio.h>

int main(void) {
    int    i = 42;
    double d = 3.14;

    /* 덧셈 전에 i가 double로 승격됨 */
    double result = i + d;
    printf("%f\n", result);  /* 45.140000 */

    /* 산술에서 char가 int로 승격됨 */
    char c = 'A';         /* 65 */
    int  n = c + 1;       /* 66 */
    printf("%c\n", (char)n);  /* 'B' */

    return 0;
}
```

### 명시적 변환 (캐스팅)

의도적으로 타입 간 변환을 하려면 캐스트를 사용합니다.

```c
#include <stdio.h>

int main(void) {
    int a = 7, b = 2;

    /* 캐스트 없이: 정수 나눗셈 */
    double bad  = a / b;          /* 3.000000 */

    /* 캐스트 사용: 부동소수점 나눗셈 */
    double good = (double)a / b;  /* 3.500000 */

    printf("bad  = %f\n", bad);
    printf("good = %f\n", good);

    /* 잘림 위험: double에서 int로 */
    double pi = 3.99;
    int truncated = (int)pi;  /* 3 — 소수 부분이 버려짐 */
    printf("truncated = %d\n", truncated);

    return 0;
}
```

### 일반적인 잘림 위험

| 변환 | 위험 |
|------|------|
| `double`에서 `float` | 정밀도 손실 |
| `double`에서 `int` | 소수 부분 버려짐 |
| `long long`에서 `int` | 값이 `INT_MAX`를 초과하면 상위 비트 손실 |
| `int`에서 `char` | 최하위 8비트만 보존 |
| `unsigned`에서 `signed` | 값이 `TYPE_MAX`보다 크면 재해석 |

```c
#include <stdio.h>

int main(void) {
    long long big = 5000000000LL;
    int small = (int)big;
    printf("big = %lld, small = %d\n", big, small);
    /* small은 쓰레기 값 — 50억은 INT_MAX를 초과함 */

    unsigned int u = 3000000000U;
    int s = (int)u;
    printf("unsigned %u -> signed %d\n", u, s);
    /* 음수 — 비트 패턴의 재해석 */

    return 0;
}
```

---

## 7. 형식 지정자(Format Specifier)

`printf`와 `scanf` 계열 함수는 각 인자의 타입에 맞는 형식 지정자를 사용합니다.

### 종합 형식 지정자 표

| 지정자 | 타입 | 예시 |
|--------|------|------|
| `%d` 또는 `%i` | `int` (부호 있는 10진수) | `printf("%d", 42)` |
| `%u` | `unsigned int` | `printf("%u", 42U)` |
| `%ld` | `long` | `printf("%ld", 100000L)` |
| `%lld` | `long long` | `printf("%lld", 9000000000LL)` |
| `%lu` | `unsigned long` | `printf("%lu", 100000UL)` |
| `%llu` | `unsigned long long` | `printf("%llu", val)` |
| `%hd` | `short` | `printf("%hd", (short)10)` |
| `%f` | `double` (printf) / `float` (scanf) | `printf("%f", 3.14)` |
| `%lf` | `double` (scanf 전용) | `scanf("%lf", &d)` |
| `%e` / `%E` | 과학적 표기법 | `printf("%e", 0.001)` → `1.000000e-03` |
| `%g` | `%f`와 `%e` 중 짧은 것 | `printf("%g", 3.14)` |
| `%c` | `char` | `printf("%c", 'A')` |
| `%s` | `char *` (문자열) | `printf("%s", "hello")` |
| `%p` | 포인터 (주소) | `printf("%p", (void *)&x)` |
| `%x` / `%X` | 부호 없는 16진수 | `printf("%x", 255)` → `ff` |
| `%o` | 부호 없는 8진수 | `printf("%o", 8)` → `10` |
| `%zu` | `size_t` | `printf("%zu", sizeof(int))` |
| `%%` | 리터럴 `%` | `printf("100%%")` |

### 너비와 정밀도

```c
#include <stdio.h>

int main(void) {
    int n = 42;
    double pi = 3.14159265;

    printf("[%10d]\n", n);      /* [        42] — 오른쪽 정렬, 너비 10 */
    printf("[%-10d]\n", n);     /* [42        ] — 왼쪽 정렬 */
    printf("[%05d]\n", n);      /* [00042]      — 0으로 패딩 */

    printf("[%.2f]\n", pi);     /* [3.14]       — 소수점 이하 2자리 */
    printf("[%10.4f]\n", pi);   /* [    3.1416] — 너비 10, 소수점 4자리 */

    printf("[%.5s]\n", "Hello, World");  /* [Hello] — 문자열에서 최대 5문자 */

    return 0;
}
```

> **경고**: 잘못된 형식 지정자는 **미정의 동작**을 유발합니다. `long long`을 출력하는 데 `%d`를 사용하거나 `int`를 출력하는 데 `%f`를 사용하면 쓰레기 출력이나 충돌이 발생할 수 있습니다.

---

## 8. sizeof 연산자

`sizeof` 연산자는 타입이나 변수의 바이트 단위 크기를 반환합니다. 가변 길이 배열을 제외하면 **컴파일 시간**에 평가됩니다.

```c
#include <stdio.h>

int main(void) {
    /* 타입에 대한 sizeof */
    printf("char:        %zu bytes\n", sizeof(char));        /* 항상 1 */
    printf("short:       %zu bytes\n", sizeof(short));
    printf("int:         %zu bytes\n", sizeof(int));
    printf("long:        %zu bytes\n", sizeof(long));
    printf("long long:   %zu bytes\n", sizeof(long long));
    printf("float:       %zu bytes\n", sizeof(float));
    printf("double:      %zu bytes\n", sizeof(double));
    printf("long double: %zu bytes\n", sizeof(long double));
    printf("void *:      %zu bytes\n", sizeof(void *));

    printf("\n");

    /* 변수에 대한 sizeof */
    int arr[10];
    printf("arr:         %zu bytes\n", sizeof(arr));          /* 40 (10 * 4) */
    printf("arr elements: %zu\n", sizeof(arr) / sizeof(arr[0])); /* 10 */

    /* 표현식에 대한 sizeof — 표현식은 평가되지 않음 */
    int x = 5;
    printf("sizeof(x++): %zu\n", sizeof(x++));  /* x는 여전히 5 */
    printf("x = %d\n", x);                       /* 5, 6이 아님! */

    return 0;
}
```

### 이식성 있는 코드를 위한 sizeof 사용

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n = 10;

    /* 메모리 할당 — sizeof는 어떤 플랫폼에서든 올바른 크기를 보장 */
    int *arr = malloc(n * sizeof(*arr));  /* 권장: sizeof(*arr) */
    if (arr == NULL) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = i * i;
    }

    /* 배열 길이 계산 (실제 배열에만 작동, 포인터에는 안됨) */
    int fixed[5] = {10, 20, 30, 40, 50};
    size_t len = sizeof(fixed) / sizeof(fixed[0]);
    printf("fixed has %zu elements\n", len);  /* 5 */

    free(arr);
    return 0;
}
```

---

## 연습문제

### 연습문제 1: 타입 크기 탐색기

모든 기본 정수 및 부동소수점 타입의 크기(바이트 단위)와 최솟값/최댓값을 출력하는 프로그램을 작성하세요. 정수 한계에는 `<limits.h>`를, 부동소수점 한계에는 `<float.h>`를 사용하세요. 출력을 깔끔한 표 형식으로 포맷하세요:

```
Type              Size    Min                  Max
char              1       -128                 127
unsigned char     1       0                    255
short             2       ...                  ...
...
```

### 연습문제 2: 오버플로우 탐정

다음을 보여주는 프로그램을 작성하세요:

1. `INT_MAX + 1`로 부호 있는 정수 오버플로우 (`-Wall`로 컴파일하고 경고를 확인).
2. `0U - 1`로 부호 없는 정수 순환.
3. `16777217` (2^24 + 1)을 `float`에 저장하고 출력하여 float 정밀도 손실.

각 경우에 대해 변환 전후 값을 출력하고 무슨 일이 일어났는지 설명하는 주석을 작성하세요.

### 연습문제 3: 온도 변환기

`scanf`를 사용하여 화씨 온도를 (`double`로) 입력받고 섭씨 등가를 출력하는 프로그램을 작성하세요. 공식은 `C = (F - 32) * 5.0 / 9.0`입니다. 결과를 정확히 소수점 이하 2자리로 출력하세요. 테스트: 32.0 (예상: 0.00), 212.0 (예상: 100.00), -40.0 (예상: -40.00).

### 연습문제 4: 타입 변환 함정

각 줄의 출력을 예측한 다음, 프로그램을 실행하여 확인하세요:

```c
printf("%d\n", (int)3.9);
printf("%d\n", (int)-3.9);
printf("%u\n", (unsigned int)-1);
printf("%d\n", (char)300);
printf("%f\n", 7 / 2);
printf("%f\n", 7.0 / 2);
```

각 줄 옆에 해당 출력이 나오는 이유를 설명하는 주석을 작성하세요.

### 연습문제 5: 형식 지정자 연습

각 기본 타입(`char`, `short`, `int`, `long`, `long long`, `unsigned int`, `float`, `double`)의 변수를 하나씩 선언하고 다음을 사용하여 각각 출력하는 단일 프로그램을 작성하세요:

1. 올바른 형식 지정자.
2. 의도적으로 잘못된 지정자 (예: `double`에 `%d`).

`-Wall -Wextra`로 컴파일하고 컴파일러가 어떤 불일치를 감지하는지 확인하세요. 발견한 내용을 주석으로 문서화하세요.

---

## 다음 단계

이제 C가 메모리에 데이터를 저장하는 방법을 이해했으니, [연산자와 표현식](./03_Operators_and_Expressions.md)에서 데이터를 결합하고 변환하는 방법을 살펴봅시다!
