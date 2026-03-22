# 연산자와 표현식

**이전**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md) | **다음**: [제어 흐름](./04_Control_Flow.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 산술, 관계, 논리, 대입 연산자를 표현식에 적용한다
2. C의 연산자 우선순위와 결합 방향 규칙을 사용하여 표현식을 평가한다
3. 전위 증가와 후위 증가를 구별하고 부작용(side effect) 순서를 설명한다
4. 비트 연산자를 사용하여 플래그 조작과 마스킹을 수행한다
5. 삼항 연산자와 쉼표 연산자를 간결한 표현식에 적용한다

---

연산자는 프로그래밍 언어의 동사입니다 -- 데이터로 무엇을 할지 컴파일러에 지시합니다. C에는 기본 산술부터 비트 수준 조작까지 다양한 연산자가 있습니다. 이러한 연산자가 어떻게 작동하는지, 그리고 어떤 순서로 평가되는지 이해하는 것은 정확하고 효율적인 C 코드를 작성하는 데 필수적입니다.

## 1. 산술 연산자

산술 연산자는 숫자 피연산자에 대해 수학적 계산을 수행합니다.

| 연산자 | 이름 | 예시 | 결과 |
|--------|------|------|------|
| `+` | 덧셈 | `7 + 3` | `10` |
| `-` | 뺄셈 | `7 - 3` | `4` |
| `*` | 곱셈 | `7 * 3` | `21` |
| `/` | 나눗셈 | `7 / 3` | `2` (정수) |
| `%` | 나머지 (모듈로) | `7 % 3` | `1` |
| `+` | 단항 양수 | `+5` | `5` |
| `-` | 단항 음수 | `-5` | `-5` |

### 정수 나눗셈 vs 부동소수점 나눗셈

```c
#include <stdio.h>

int main(void) {
    /* 정수 나눗셈은 0 방향으로 잘림 */
    printf("7 / 3   = %d\n", 7 / 3);     /* 2  */
    printf("-7 / 3  = %d\n", -7 / 3);    /* -2 (C99 이후 0 방향으로 잘림) */

    /* 부동소수점 나눗셈은 소수 부분을 보존 */
    printf("7.0 / 3 = %f\n", 7.0 / 3);   /* 2.333333 */

    /* 캐스트로 float 나눗셈 강제 */
    int a = 7, b = 3;
    printf("(double)a / b = %f\n", (double)a / b);  /* 2.333333 */

    return 0;
}
```

### 나머지 연산자

`%` 연산자는 정수 나눗셈 후 나머지를 반환합니다. 정수 타입에서만 작동합니다.

```c
#include <stdio.h>

int main(void) {
    printf("10 %% 3 = %d\n", 10 % 3);   /* 1 */
    printf("10 %% 5 = %d\n", 10 % 5);   /* 0 */
    printf("-7 %% 3 = %d\n", -7 % 3);   /* -1 (C99 이후 부호는 피제수를 따름) */

    /* 일반적인 사용 */
    int n = 42;
    if (n % 2 == 0) {
        printf("%d is even\n", n);
    }

    /* 마지막 자릿수 추출 */
    int last_digit = 12345 % 10;  /* 5 */
    printf("Last digit of 12345: %d\n", last_digit);

    return 0;
}
```

---

## 2. 관계 및 동등 연산자

관계 연산자는 두 값을 비교하고 `int` 결과를 생성합니다: `1` (참) 또는 `0` (거짓). C99 이전에는 내장 불리언 타입이 없었습니다; C99에서 `_Bool` (또는 `<stdbool.h>`를 통한 `bool`)이 추가되었습니다.

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `==` | 같음 | `a == b` |
| `!=` | 같지 않음 | `a != b` |
| `<` | 미만 | `a < b` |
| `>` | 초과 | `a > b` |
| `<=` | 이하 | `a <= b` |
| `>=` | 이상 | `a >= b` |

```c
#include <stdio.h>
#include <stdbool.h>   /* C99: bool, true, false */

int main(void) {
    int x = 10, y = 20;

    printf("x == y: %d\n", x == y);  /* 0 */
    printf("x != y: %d\n", x != y);  /* 1 */
    printf("x < y:  %d\n", x < y);   /* 1 */
    printf("x >= y: %d\n", x >= y);  /* 0 */

    /* bool 사용 (C99) */
    bool is_positive = (x > 0);
    printf("is_positive: %d\n", is_positive);  /* 1 */

    return 0;
}
```

> **흔한 실수**: `==` (비교) 대신 `=` (대입) 사용:
>
> ```c
> if (x = 5) {  /* 버그: x에 5를 대입, 항상 참! */
>     printf("This always executes\n");
> }
> ```
>
> 일부 프로그래머는 `5 == x` (요다 조건문)으로 작성하여 이 실수를 방지합니다. `5 = x`는 컴파일러 오류가 되기 때문입니다. `-Wall`로 컴파일하면 이 패턴에 대해 경고합니다.

---

## 3. 논리 연산자

논리 연산자는 불리언 표현식을 결합합니다. 0이 아닌 모든 값을 참으로, 0을 거짓으로 취급합니다.

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `&&` | 논리 AND | `a && b` |
| `\|\|` | 논리 OR | `a \|\| b` |
| `!` | 논리 NOT | `!a` |

### 단락 평가(Short-Circuit Evaluation)

C는 논리 표현식을 **왼쪽에서 오른쪽**으로 평가하고, 결과가 결정되는 즉시 멈춥니다:

- `&&`는 왼쪽 피연산자가 거짓이면 멈춥니다 (전체 표현식이 거짓).
- `||`는 왼쪽 피연산자가 참이면 멈춥니다 (전체 표현식이 참).

```c
#include <stdio.h>

int main(void) {
    int a = 5, b = 0;

    /* 단락 평가: b가 0(거짓)이므로 (b && ...)는 즉시 거짓 */
    if (b != 0 && a / b > 2) {
        printf("This is safe\n");
    } else {
        printf("Division by zero avoided!\n");  /* 이것이 출력됨 */
    }

    /* 단락 평가가 없으면 a/b는 충돌 */

    /* 논리 NOT */
    int logged_in = 0;
    if (!logged_in) {
        printf("Please log in\n");  /* 이것이 출력됨 */
    }

    /* 진리표 시연 */
    printf("\nTruth Table for && and ||\n");
    printf("0 && 0 = %d\n", 0 && 0);  /* 0 */
    printf("0 && 1 = %d\n", 0 && 1);  /* 0 */
    printf("1 && 0 = %d\n", 1 && 0);  /* 0 */
    printf("1 && 1 = %d\n", 1 && 1);  /* 1 */
    printf("0 || 0 = %d\n", 0 || 0);  /* 0 */
    printf("0 || 1 = %d\n", 0 || 1);  /* 1 */
    printf("1 || 0 = %d\n", 1 || 0);  /* 1 */
    printf("1 || 1 = %d\n", 1 || 1);  /* 1 */

    return 0;
}
```

### 실용 예제: 입력 유효성 검사

```c
#include <stdio.h>

int main(void) {
    int age;
    printf("Enter age: ");
    scanf("%d", &age);

    if (age >= 0 && age <= 150) {
        printf("Valid age: %d\n", age);
    } else {
        printf("Invalid age\n");
    }

    /* 논리 OR를 이용한 범위 검사 */
    char grade;
    printf("Enter grade (A-F): ");
    scanf(" %c", &grade);

    if (grade < 'A' || grade > 'F') {
        printf("Invalid grade\n");
    }

    return 0;
}
```

---

## 4. 대입 연산자

대입 연산자는 변수에 값을 저장합니다. 복합 대입 연산자는 연산과 대입을 결합합니다.

| 연산자 | 동등 표현 | 예시 |
|--------|----------|------|
| `=` | -- | `x = 5` |
| `+=` | `x = x + n` | `x += 3` |
| `-=` | `x = x - n` | `x -= 3` |
| `*=` | `x = x * n` | `x *= 3` |
| `/=` | `x = x / n` | `x /= 3` |
| `%=` | `x = x % n` | `x %= 3` |
| `&=` | `x = x & n` | `x &= 0xFF` |
| `\|=` | `x = x \| n` | `x \|= 0x01` |
| `^=` | `x = x ^ n` | `x ^= mask` |
| `<<=` | `x = x << n` | `x <<= 2` |
| `>>=` | `x = x >> n` | `x >>= 2` |

```c
#include <stdio.h>

int main(void) {
    int x = 10;

    x += 5;   /* x = 15 */
    x -= 3;   /* x = 12 */
    x *= 2;   /* x = 24 */
    x /= 4;   /* x = 6  */
    x %= 5;   /* x = 1  */

    printf("x = %d\n", x);  /* 1 */

    /* 대입은 표현식 — 대입된 값을 반환 */
    int a, b, c;
    a = b = c = 0;  /* 오른쪽에서 왼쪽: c=0, b=0, a=0 */
    printf("a=%d b=%d c=%d\n", a, b, c);

    return 0;
}
```

---

## 5. 증가와 감소

`++`와 `--` 연산자는 1을 더하거나 뺍니다. 두 가지 형태가 있으며 중요한 차이가 있습니다.

| 형태 | 이름 | 동작 |
|------|------|------|
| `++x` | 전위 증가 | 먼저 증가, 그 다음 새 값 사용 |
| `x++` | 후위 증가 | 현재 값 사용, 그 다음 증가 |
| `--x` | 전위 감소 | 먼저 감소, 그 다음 새 값 사용 |
| `x--` | 후위 감소 | 현재 값 사용, 그 다음 감소 |

```c
#include <stdio.h>

int main(void) {
    int a = 5;
    int b;

    /* 전위 증가: 증가 후 대입 */
    b = ++a;
    printf("++a: a=%d, b=%d\n", a, b);  /* a=6, b=6 */

    a = 5;  /* 초기화 */

    /* 후위 증가: 대입 후 증가 */
    b = a++;
    printf("a++: a=%d, b=%d\n", a, b);  /* a=6, b=5 */

    return 0;
}
```

### 표현식 내 부작용

> **경고**: 하나의 표현식에서 같은 변수에 `++`나 `--`를 여러 번 사용하면 **미정의 동작**입니다:
>
> ```c
> int i = 5;
> int result = i++ + ++i;  /* 미정의 동작 — 이렇게 하지 마세요! */
> ```
>
> 컴파일러는 `i++`와 `++i`를 어떤 순서로든 평가할 수 있습니다. 다른 컴파일러(또는 최적화 수준)에서 다른 결과가 나올 수 있습니다.

### 언제 어떤 것을 사용할까

- **독립 문장** (`i++;` 또는 `++i;`): 차이 없음; 둘 다 `i`를 1만큼 증가.
- **표현식 내에서**: 이전 값이 특별히 필요하지 않다면 전위 증가(`++i`) 사용.
- **`for` 루프에서**: `for (int i = 0; i < n; i++)` — 어떤 형태든 작동하지만, C에서는 `i++`가 관례.

---

## 6. 비트 연산자

비트 연산자는 정수 값의 개별 비트에 대해 작동합니다. 시스템 프로그래밍, 임베디드 개발, 성능 중심 코드에 필수적입니다.

| 연산자 | 이름 | 설명 |
|--------|------|------|
| `&` | 비트 AND | 두 비트가 모두 1이면 1로 설정 |
| `\|` | 비트 OR | 어느 한쪽 비트가 1이면 1로 설정 |
| `^` | 비트 XOR | 비트가 다르면 1로 설정 |
| `~` | 비트 NOT | 모든 비트 반전 |
| `<<` | 왼쪽 시프트 | 비트를 왼쪽으로 이동, 0으로 채움 |
| `>>` | 오른쪽 시프트 | 비트를 오른쪽으로 이동 (채움은 부호에 따라 다름) |

### AND, OR, XOR 진리표

| A | B | A & B | A \| B | A ^ B |
|---|---|-------|--------|-------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

### 실용 예제

```c
#include <stdio.h>

int main(void) {
    unsigned char a = 0b11001010;  /* 10진수로 202 */
    unsigned char b = 0b10110101;  /* 10진수로 181 */

    printf("a & b  = 0x%02X\n", a & b);   /* 0x80 = 10000000 */
    printf("a | b  = 0x%02X\n", a | b);   /* 0xFF = 11111111 */
    printf("a ^ b  = 0x%02X\n", a ^ b);   /* 0x7F = 01111111 */
    printf("~a     = 0x%02X\n", (unsigned char)~a);  /* 0x35 = 00110101 */

    /* 시프트 연산자 */
    unsigned int x = 1;
    printf("1 << 3 = %u\n", x << 3);   /* 8  (2^3을 곱함) */
    printf("8 >> 2 = %u\n", 8U >> 2);  /* 2  (2^2로 나눔)  */

    return 0;
}
```

### 플래그 조작

비트 연산자는 플래그를 관리하는 데 자주 사용됩니다 -- 켜짐/꺼짐 상태를 나타내는 개별 비트입니다.

```c
#include <stdio.h>

/* 플래그를 2의 거듭제곱으로 정의 */
#define FLAG_READ    (1 << 0)   /* 0001 = 1 */
#define FLAG_WRITE   (1 << 1)   /* 0010 = 2 */
#define FLAG_EXECUTE (1 << 2)   /* 0100 = 4 */
#define FLAG_DELETE  (1 << 3)   /* 1000 = 8 */

int main(void) {
    unsigned int permissions = 0;

    /* 플래그 설정 */
    permissions |= FLAG_READ;          /* 읽기 켜기 */
    permissions |= FLAG_WRITE;         /* 쓰기 켜기 */
    printf("After set: %u\n", permissions);  /* 3 (0011) */

    /* 플래그 확인 */
    if (permissions & FLAG_READ) {
        printf("Read permission is ON\n");
    }
    if (!(permissions & FLAG_EXECUTE)) {
        printf("Execute permission is OFF\n");
    }

    /* 플래그 해제 */
    permissions &= ~FLAG_WRITE;        /* 쓰기 끄기 */
    printf("After clear write: %u\n", permissions);  /* 1 (0001) */

    /* 플래그 토글 */
    permissions ^= FLAG_EXECUTE;       /* 실행 토글 */
    printf("After toggle execute: %u\n", permissions);  /* 5 (0101) */

    return 0;
}
```

### 비트 마스킹

```c
#include <stdio.h>

int main(void) {
    /* 32비트 값에서 특정 바이트 추출 */
    unsigned int color = 0xFF8040A0;   /* RGBA: R=FF, G=80, B=40, A=A0 */

    unsigned char r = (color >> 24) & 0xFF;
    unsigned char g = (color >> 16) & 0xFF;
    unsigned char b = (color >>  8) & 0xFF;
    unsigned char a = (color >>  0) & 0xFF;

    printf("R=%u G=%u B=%u A=%u\n", r, g, b, a);
    /* R=255 G=128 B=64 A=160 */

    /* 바이트를 32비트 값으로 패킹 */
    unsigned int packed = ((unsigned int)r << 24) |
                          ((unsigned int)g << 16) |
                          ((unsigned int)b <<  8) |
                          ((unsigned int)a);
    printf("Packed: 0x%08X\n", packed);  /* 0xFF8040A0 */

    return 0;
}
```

---

## 7. 삼항 연산자와 쉼표 연산자

### 삼항 연산자

삼항 연산자 `condition ? expr_if_true : expr_if_false`는 간단한 표현식에서 `if-else`의 간결한 대안입니다.

```c
#include <stdio.h>

int main(void) {
    int x = 10, y = 20;

    /* if-else 대신 */
    int max = (x > y) ? x : y;
    printf("max = %d\n", max);  /* 20 */

    /* printf 내에서 인라인으로 */
    int score = 75;
    printf("Result: %s\n", (score >= 60) ? "Pass" : "Fail");  /* Pass */

    /* 중첩 삼항 (가독성이 떨어지므로 자제) */
    int val = 0;
    const char *sign = (val > 0) ? "positive"
                     : (val < 0) ? "negative"
                     : "zero";
    printf("val is %s\n", sign);  /* zero */

    /* 절댓값 */
    int n = -42;
    int abs_n = (n >= 0) ? n : -n;
    printf("|%d| = %d\n", n, abs_n);

    return 0;
}
```

### 쉼표 연산자

쉼표 연산자는 왼쪽에서 오른쪽으로 두 표현식을 평가하고 가장 오른쪽 표현식의 값을 반환합니다. `for` 루프 헤더에서 가장 많이 볼 수 있습니다.

```c
#include <stdio.h>

int main(void) {
    /* for 루프에서 쉼표 — 여러 변수 */
    for (int i = 0, j = 10; i < j; i++, j--) {
        printf("i=%d j=%d\n", i, j);
    }

    /* 연산자로서의 쉼표 (for 루프 외에서는 거의 사용되지 않음) */
    int x = (1, 2, 3);  /* x = 3 — 마지막 표현식의 값 */
    printf("x = %d\n", x);

    return 0;
}
```

---

## 8. 연산자 우선순위 표

여러 연산자가 표현식에 나타나면, 우선순위와 결합 방향이 평가 순서를 결정합니다. 높은 우선순위의 연산자가 더 강하게 결합합니다.

| 우선순위 | 연산자 | 설명 | 결합 방향 |
|---------|--------|------|----------|
| 1 (최고) | `()` `[]` `->` `.` | 함수 호출, 첨자, 멤버 접근 | 왼쪽에서 오른쪽 |
| 2 | `++` `--` (후위) | 후위 증가/감소 | 왼쪽에서 오른쪽 |
| 3 | `++` `--` (전위) `+` `-` `!` `~` `*` `&` `sizeof` `(type)` | 단항 연산자, 캐스트, sizeof | 오른쪽에서 왼쪽 |
| 4 | `*` `/` `%` | 곱셈류 | 왼쪽에서 오른쪽 |
| 5 | `+` `-` | 덧셈류 | 왼쪽에서 오른쪽 |
| 6 | `<<` `>>` | 비트 시프트 | 왼쪽에서 오른쪽 |
| 7 | `<` `<=` `>` `>=` | 관계 | 왼쪽에서 오른쪽 |
| 8 | `==` `!=` | 동등 | 왼쪽에서 오른쪽 |
| 9 | `&` | 비트 AND | 왼쪽에서 오른쪽 |
| 10 | `^` | 비트 XOR | 왼쪽에서 오른쪽 |
| 11 | `\|` | 비트 OR | 왼쪽에서 오른쪽 |
| 12 | `&&` | 논리 AND | 왼쪽에서 오른쪽 |
| 13 | `\|\|` | 논리 OR | 왼쪽에서 오른쪽 |
| 14 | `?:` | 삼항 조건 | 오른쪽에서 왼쪽 |
| 15 | `=` `+=` `-=` `*=` `/=` `%=` `<<=` `>>=` `&=` `^=` `\|=` | 대입 | 오른쪽에서 왼쪽 |
| 16 (최저) | `,` | 쉼표 | 왼쪽에서 오른쪽 |

### 우선순위 예제

```c
#include <stdio.h>

int main(void) {
    /* 곱셈이 덧셈보다 먼저 */
    int a = 2 + 3 * 4;     /* 2 + (3*4) = 14, (2+3)*4 = 20이 아님 */
    printf("2 + 3 * 4 = %d\n", a);

    /* 관계가 논리보다 먼저 */
    int x = 5, y = 10;
    int result = x > 3 && y < 20;  /* (x>3) && (y<20) = 1 */
    printf("x > 3 && y < 20 = %d\n", result);

    /* 비트 AND는 동등보다 우선순위가 낮음 — 흔한 함정! */
    int flags = 5;        /* 0101 */
    int mask  = 4;        /* 0100 */

    /* 잘못: ==가 &보다 더 강하게 결합 */
    if (flags & mask == 4) {
        printf("Bug: this tests flags & (mask == 4)\n");
    }

    /* 올바름: 괄호 사용 */
    if ((flags & mask) == 4) {
        printf("Correct: bit 2 is set\n");
    }

    /* 확실하지 않으면 괄호를 사용하세요! */
    int b = (2 + 3) * 4;  /* 20 — 명시적 그룹화 */
    printf("(2 + 3) * 4 = %d\n", b);

    return 0;
}
```

> **모범 사례**: 평가 순서가 즉시 명확하지 않을 때는 괄호를 추가하세요. 런타임 비용은 없으며 모든 독자에게 의도를 명확히 합니다.

---

## 연습문제

### 연습문제 1: 표현식 계산기

코드를 실행하지 않고 각 표현식을 수동으로 계산하세요. 그런 다음 프로그램을 작성하여 확인하세요:

```c
int a = 10, b = 3, c = 7;
printf("%d\n", a + b * c);           /* ? */
printf("%d\n", (a + b) * c);         /* ? */
printf("%d\n", a % b + c / b);       /* ? */
printf("%d\n", a > b && c > a);      /* ? */
printf("%d\n", !0 + !1);            /* ? */
printf("%d\n", a & b | c);          /* ? */
```

### 연습문제 2: 임시 변수 없이 교환

두 정수 변수를 교환하는 프로그램을 작성하세요:

1. XOR (`^`) — 세 문장, 임시 변수 없음.
2. 덧셈과 뺄셈 — 세 문장, 임시 변수 없음.

교환 전후 값을 출력하세요. XOR 방법이 실패할 수 있는 경우를 설명하세요 (힌트: 두 변수가 같은 객체일 때는?).

### 연습문제 3: 권한 검사기

비트 시프트를 사용하여 네 개의 권한 플래그(`READ`, `WRITE`, `EXECUTE`, `ADMIN`)를 정의하세요. 활성화된 권한을 출력하는 함수 `void print_permissions(unsigned int perms)`를 작성하세요. 그런 다음 다음을 수행하는 main 함수를 작성하세요:

1. READ와 WRITE 부여.
2. 모든 활성 권한 확인 및 출력.
3. WRITE 취소.
4. ADMIN 부여.
5. 다시 확인 및 출력.

### 연습문제 4: 비트 카운터

`unsigned int n`에서 1인 비트의 수를 반환하는 함수 `int count_set_bits(unsigned int n)`를 작성하세요. 두 가지 방법으로 구현하세요:

1. 최하위 비트를 확인하고 오른쪽 시프트하는 루프 사용.
2. Brian Kernighan 트릭 사용: `n = n & (n - 1)`은 가장 낮은 설정 비트를 지웁니다.

값 0, 1, 255, 0xDEADBEEF로 두 구현을 테스트하세요.

### 연습문제 5: RGBA 색상 믹서

다음을 수행하는 프로그램을 작성하세요:

1. 두 RGBA 색상을 `unsigned int` 값으로 정의 (예: 빨강 `0xFF0000FF`, 초록 `0x00FF00FF`).
2. 비트 연산자를 사용하여 각 색상의 R, G, B, A 성분 추출.
3. 각 채널의 평균을 구하여 두 색상의 50% 블렌드 계산.
4. 블렌드된 채널을 단일 `unsigned int`로 패킹.
5. 세 색상 모두를 `0xRRGGBBAA` 형식으로 출력.

---

## 다음 단계

이제 데이터로 계산하는 방법을 알게 되었습니다. 다음으로 [제어 흐름](./04_Control_Flow.md)에서 결정을 내리고 동작을 반복하는 방법을 배워봅시다!
