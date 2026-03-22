# 제어 흐름

**이전**: [연산자와 표현식](./03_Operators_and_Expressions.md) | **다음**: [함수와 스코프](./05_Functions_and_Scope.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `if`, `else if`, `else` 문을 사용하여 분기 로직을 작성한다
2. `switch`-`case`로 다중 분기를 대체하고 fall-through 동작을 설명한다
3. `for`로 카운트 루프를, `while`과 `do-while`로 조건 루프를 구현한다
4. `break`, `continue`, 중첩 루프 패턴으로 루프 실행을 제어한다
5. C에 `goto`가 존재하는 이유와 사용이 적절한 경우를 설명한다

---

프로그램은 결정을 내리고 동작을 반복할 수 있을 때 유용해집니다. 제어 흐름 문은 조건에 따라 실행을 분기하고 기준이 충족될 때까지 코드 블록을 반복할 수 있게 합니다. C는 한번 마스터하면 프로그램 실행을 정밀하게 제어할 수 있는 간결한 제어 흐름 구성 요소 세트를 제공합니다.

## 1. if / else if / else

`if` 문은 가장 기본적인 분기 구조입니다. 조건이 참(0이 아닌 값)일 때만 코드 블록을 실행합니다.

### 기본 구문

```c
if (condition) {
    /* 조건이 0이 아닐 때 (참) 실행 */
}

if (condition) {
    /* 참 분기 */
} else {
    /* 거짓 분기 */
}

if (condition1) {
    /* ... */
} else if (condition2) {
    /* ... */
} else if (condition3) {
    /* ... */
} else {
    /* 위 어느 것에도 해당하지 않는 경우 */
}
```

### 예제

```c
#include <stdio.h>

int main(void) {
    int temperature = 22;

    if (temperature > 30) {
        printf("It's hot outside\n");
    } else if (temperature > 20) {
        printf("It's warm outside\n");    /* 이것이 출력됨 */
    } else if (temperature > 10) {
        printf("It's cool outside\n");
    } else {
        printf("It's cold outside\n");
    }

    return 0;
}
```

### 중첩

```c
#include <stdio.h>

int main(void) {
    int age = 25;
    int has_license = 1;

    if (age >= 18) {
        if (has_license) {
            printf("You can drive\n");
        } else {
            printf("Get a license first\n");
        }
    } else {
        printf("Too young to drive\n");
    }

    return 0;
}
```

### 흔한 실수

```c
#include <stdio.h>

int main(void) {
    int x = 5;

    /* 실수 1: = vs == */
    if (x = 10) {              /* 버그: x에 10을 대입, 항상 참 */
        printf("x is now %d\n", x);
    }

    /* 실수 2: 매달린 else (Dangling else) */
    int a = 1, b = 0;
    if (a)
        if (b)
            printf("a and b\n");
    else                       /* 이 else는 외부가 아닌 내부 if에 속함 */
        printf("This might surprise you\n");  /* a=1이고 b=0일 때 출력됨 */

    /* 해결: 항상 중괄호 사용 */
    if (a) {
        if (b) {
            printf("a and b\n");
        }
    } else {
        printf("not a\n");
    }

    /* 실수 3: 빈 문장 */
    if (x > 0);  /* 경고: 세미콜론이 이것을 아무 동작 없는 문으로 만듦 */
    {
        printf("This always executes regardless of x\n");
    }

    return 0;
}
```

> **모범 사례**: 단일 문장 본문에도 항상 중괄호 `{}`를 사용하세요. 매달린 else 문제를 방지하고 향후 수정을 더 안전하게 합니다.

---

## 2. switch-case

`switch` 문은 정수 표현식의 값에 따라 여러 대안 중에서 선택합니다. 긴 `if-else if` 체인보다 깔끔한 경우가 많습니다.

### 구문

```c
switch (expression) {
    case constant1:
        /* 문장들 */
        break;
    case constant2:
        /* 문장들 */
        break;
    default:
        /* 위 어느 것에도 해당하지 않는 경우 */
        break;
}
```

### 예제: 메뉴 선택

```c
#include <stdio.h>

int main(void) {
    int choice;
    printf("Menu:\n");
    printf("1. New Game\n");
    printf("2. Load Game\n");
    printf("3. Settings\n");
    printf("4. Quit\n");
    printf("Enter choice: ");
    scanf("%d", &choice);

    switch (choice) {
        case 1:
            printf("Starting new game...\n");
            break;
        case 2:
            printf("Loading saved game...\n");
            break;
        case 3:
            printf("Opening settings...\n");
            break;
        case 4:
            printf("Goodbye!\n");
            break;
        default:
            printf("Invalid choice\n");
            break;
    }

    return 0;
}
```

### Fall-Through 동작

`break` 없이는 실행이 다음 case로 **떨어집니다(fall-through)**. 때로는 이것이 의도적입니다.

```c
#include <stdio.h>

int main(void) {
    char grade = 'B';

    switch (grade) {
        case 'A':
        case 'B':            /* fall-through: A와 B 모두 "Good" 출력 */
            printf("Good job!\n");
            break;
        case 'C':
            printf("Average\n");
            break;
        case 'D':
        case 'F':            /* fall-through: D와 F 모두 "Needs improvement" 출력 */
            printf("Needs improvement\n");
            break;
        default:
            printf("Invalid grade\n");
            break;
    }

    return 0;
}
```

### 의도적 Fall-Through: 월의 일수

```c
#include <stdio.h>

int main(void) {
    int month = 2, year = 2024;
    int days;

    switch (month) {
        case 2:
            days = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))
                   ? 29 : 28;
            break;
        case 4: case 6: case 9: case 11:  /* 30일인 달 */
            days = 30;
            break;
        default:                            /* 31일인 달 */
            days = 31;
            break;
    }

    printf("Month %d in year %d has %d days\n", month, year, days);
    return 0;
}
```

### 제약 사항

- `switch` 표현식은 정수 타입(`int`, `char`, `enum` 등)이어야 합니다 -- `float`, `double`, `char *`는 **불가**.
- 각 `case` 레이블은 **컴파일 시간 상수**여야 합니다.
- 중복된 case 값은 허용되지 않습니다.

---

## 3. for 루프

`for` 루프는 카운트 반복을 위한 C의 핵심 도구입니다. 초기화, 조건, 갱신을 한 줄에 담습니다.

### 구문

```c
for (initialization; condition; update) {
    /* 본문 */
}
```

실행 순서:
1. **초기화** — 루프 시작 전 한 번 실행.
2. **조건** — 각 반복 전에 검사; 거짓이면 루프 종료.
3. **본문** — 조건이 참이면 실행.
4. **갱신** — 각 반복 후 실행, 그 다음 단계 2로.

### 예제

```c
#include <stdio.h>

int main(void) {
    /* 0부터 4까지 카운트 */
    for (int i = 0; i < 5; i++) {
        printf("%d ", i);
    }
    printf("\n");  /* 0 1 2 3 4 */

    /* 역순 카운트 */
    for (int i = 10; i > 0; i--) {
        printf("%d ", i);
    }
    printf("\n");  /* 10 9 8 7 6 5 4 3 2 1 */

    /* 2씩 증가 */
    for (int i = 0; i <= 20; i += 2) {
        printf("%d ", i);
    }
    printf("\n");  /* 0 2 4 6 8 10 12 14 16 18 20 */

    /* 1부터 100까지 합 */
    int sum = 0;
    for (int i = 1; i <= 100; i++) {
        sum += i;
    }
    printf("Sum 1..100 = %d\n", sum);  /* 5050 */

    return 0;
}
```

### for 루프에서 여러 변수

```c
#include <stdio.h>

int main(void) {
    /* 수렴하는 두 루프 변수 */
    for (int lo = 0, hi = 10; lo < hi; lo++, hi--) {
        printf("lo=%d hi=%d\n", lo, hi);
    }
    /* lo=0 hi=10, lo=1 hi=9, ..., lo=4 hi=6 */

    return 0;
}
```

### 무한 루프

```c
/* 무한 루프 — break 또는 return으로 종료해야 함 */
for (;;) {
    printf("Running forever...\n");
    break;  /* 이 예제에서는 즉시 종료 */
}
```

---

## 4. while 루프

`while` 루프는 조건이 참인 동안 블록을 반복합니다. **진입 제어** 루프입니다: 조건이 처음부터 거짓이면 본문은 실행되지 않습니다.

### 구문

```c
while (condition) {
    /* 본문 */
}
```

### 예제

```c
#include <stdio.h>

int main(void) {
    /* 위로 카운트 */
    int i = 0;
    while (i < 5) {
        printf("%d ", i);
        i++;
    }
    printf("\n");  /* 0 1 2 3 4 */

    /* 감시 값: -1이 나올 때까지 읽기 */
    int num, total = 0, count = 0;
    printf("Enter numbers (-1 to stop): ");
    scanf("%d", &num);

    while (num != -1) {
        total += num;
        count++;
        scanf("%d", &num);
    }

    if (count > 0) {
        printf("Average: %.2f\n", (double)total / count);
    }

    return 0;
}
```

### 자릿수 카운터

```c
#include <stdio.h>

int main(void) {
    int number = 123456;
    int digits = 0;
    int temp = number;

    if (temp == 0) {
        digits = 1;
    } else {
        while (temp != 0) {
            temp /= 10;
            digits++;
        }
    }

    printf("%d has %d digits\n", number, digits);  /* 123456 has 6 digits */
    return 0;
}
```

---

## 5. do-while 루프

`do-while` 루프는 **출구 제어** 루프입니다: 조건이 검사되기 전에 본문이 항상 최소 한 번 실행됩니다.

### 구문

```c
do {
    /* 본문 — 항상 최소 한 번 실행 */
} while (condition);  /* 세미콜론에 주의! */
```

### 입력 유효성 검사 패턴

`do-while`의 가장 일반적인 용도는 입력 유효성 검사입니다: 사용자에게 입력을 요청하고, 유효하지 않으면 반복합니다.

```c
#include <stdio.h>

int main(void) {
    int choice;

    do {
        printf("Enter a number between 1 and 10: ");
        scanf("%d", &choice);

        if (choice < 1 || choice > 10) {
            printf("Invalid! Try again.\n");
        }
    } while (choice < 1 || choice > 10);

    printf("You chose: %d\n", choice);
    return 0;
}
```

### 메뉴 루프

```c
#include <stdio.h>

int main(void) {
    int option;

    do {
        printf("\n--- Menu ---\n");
        printf("1. Say Hello\n");
        printf("2. Say Goodbye\n");
        printf("0. Exit\n");
        printf("Choice: ");
        scanf("%d", &option);

        switch (option) {
            case 1: printf("Hello!\n"); break;
            case 2: printf("Goodbye!\n"); break;
            case 0: printf("Exiting...\n"); break;
            default: printf("Unknown option\n"); break;
        }
    } while (option != 0);

    return 0;
}
```

### 비교: while vs do-while

| 특성 | `while` | `do-while` |
|------|---------|------------|
| 조건 검사 | 본문 전 | 본문 후 |
| 최소 실행 횟수 | 0 | 1 |
| 사용 경우 | 일반 루프 | 입력 유효성 검사, 메뉴 루프 |

---

## 6. break와 continue

### break

`break`는 가장 안쪽의 `for`, `while`, `do-while` 또는 `switch`를 즉시 빠져나갑니다.

```c
#include <stdio.h>

int main(void) {
    /* 50보다 큰 첫 번째 7의 배수 찾기 */
    for (int i = 51; ; i++) {
        if (i % 7 == 0) {
            printf("Found: %d\n", i);  /* 56 */
            break;
        }
    }

    /* 배열 검색 */
    int data[] = {10, 25, 37, 42, 58};
    int target = 37;
    int found = 0;

    for (int i = 0; i < 5; i++) {
        if (data[i] == target) {
            printf("Found %d at index %d\n", target, i);
            found = 1;
            break;
        }
    }
    if (!found) {
        printf("%d not found\n", target);
    }

    return 0;
}
```

### continue

`continue`는 현재 반복의 나머지를 건너뛰고 다음 반복으로 점프합니다.

```c
#include <stdio.h>

int main(void) {
    /* 홀수만 출력 */
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            continue;  /* 짝수 건너뛰기 */
        }
        printf("%d ", i);
    }
    printf("\n");  /* 1 3 5 7 9 */

    /* 양수만 합산, 음수 건너뛰기 */
    int values[] = {3, -1, 4, -1, 5, -9, 2, 6};
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        if (values[i] < 0) {
            continue;
        }
        sum += values[i];
    }
    printf("Sum of positives: %d\n", sum);  /* 20 */

    return 0;
}
```

### while 루프에서의 break와 continue

```c
#include <stdio.h>

int main(void) {
    int i = 0;
    while (i < 100) {
        i++;
        if (i % 3 != 0) {
            continue;   /* 3의 배수가 아닌 것 건너뛰기 */
        }
        if (i > 20) {
            break;       /* 20 이후 중지 */
        }
        printf("%d ", i);
    }
    printf("\n");  /* 3 6 9 12 15 18 */

    return 0;
}
```

---

## 7. 중첩 루프

루프는 다른 루프 안에 넣을 수 있습니다. 2차원 데이터 작업, 패턴 생성, 검색에 자주 사용됩니다.

### 구구단

```c
#include <stdio.h>

int main(void) {
    printf("    ");
    for (int j = 1; j <= 9; j++) {
        printf("%4d", j);
    }
    printf("\n    ------------------------------------\n");

    for (int i = 1; i <= 9; i++) {
        printf("%2d |", i);
        for (int j = 1; j <= 9; j++) {
            printf("%4d", i * j);
        }
        printf("\n");
    }

    return 0;
}
```

### 삼각형 패턴

```c
#include <stdio.h>

int main(void) {
    int rows = 5;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
        printf("\n");
    }
    /*
    *
    * *
    * * *
    * * * *
    * * * * *
    */

    return 0;
}
```

### 중첩 루프에서 조기 탈출

`break`는 가장 안쪽 루프만 빠져나갑니다. 여러 단계를 빠져나가려면 플래그 변수 또는 `goto`를 사용합니다.

```c
#include <stdio.h>

int main(void) {
    /* 방법 1: 플래그 변수 */
    int found = 0;
    int matrix[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int target = 5;

    for (int i = 0; i < 3 && !found; i++) {
        for (int j = 0; j < 3 && !found; j++) {
            if (matrix[i][j] == target) {
                printf("Found %d at [%d][%d]\n", target, i, j);
                found = 1;
            }
        }
    }

    return 0;
}
```

---

## 8. goto

`goto` 문은 같은 함수 내의 레이블이 붙은 문장으로 무조건 점프합니다. 일반적인 프로그래밍에서는 널리 권장되지 않지만, C에서 잘 확립된 사용 사례가 하나 있습니다: 중앙화된 오류 정리.

### 구문

```c
goto label;

/* ... */

label:
    /* 문장들 */
```

### 오류 정리 패턴

함수가 여러 리소스(메모리, 파일, 락)를 획득할 때, `goto`는 오류가 발생하면 역순으로 리소스를 해제하는 깔끔한 방법을 제공합니다.

```c
#include <stdio.h>
#include <stdlib.h>

int process_file(const char *path) {
    FILE *fp = NULL;
    char *buffer = NULL;
    int result = -1;  /* 실패를 가정 */

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open file\n");
        goto cleanup;
    }

    buffer = malloc(1024);
    if (buffer == NULL) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    /* ... fp와 buffer로 작업 수행 ... */
    if (fgets(buffer, 1024, fp) == NULL) {
        fprintf(stderr, "Read failed\n");
        goto cleanup;
    }

    printf("Read: %s", buffer);
    result = 0;  /* 성공 */

cleanup:
    free(buffer);       /* free(NULL)은 안전 */
    if (fp != NULL) {
        fclose(fp);
    }
    return result;
}

int main(void) {
    process_file("test.txt");
    return 0;
}
```

### goto를 일반적으로 피해야 하는 이유

- 제어 흐름을 따라가기 어렵게 만듭니다 ("스파게티 코드").
- 구조적 프로그래밍 구조를 우회합니다.
- 변수 초기화를 건너뛸 수 있어 버그를 유발합니다.

> **경험 규칙**: `goto`는 함수 끝의 단일 정리 레이블로 향하는 전방 점프에만 사용하세요. 절대 역방향으로 점프하지 마세요 (그것은 루프의 역할입니다). 함수에 리소스 정리가 필요 없다면 `goto`가 거의 확실히 필요하지 않습니다.

---

## 연습문제

### 연습문제 1: 성적 분류기

숫자 점수(0-100)를 읽고 학점을 출력하는 프로그램을 작성하세요:

- 90-100: A
- 80-89: B
- 70-79: C
- 60-69: D
- 60 미만: F

`if-else if-else`를 사용하세요. 유효하지 않은 입력(0 미만 또는 100 초과)은 오류 메시지로 처리하세요. 그런 다음 `score / 10`에 대한 `switch` 문으로 다시 작성하세요.

### 연습문제 2: FizzBuzz

1부터 100까지 숫자를 출력합니다. 3의 배수에는 "Fizz"를, 5의 배수에는 "Buzz"를, 둘 다의 배수에는 "FizzBuzz"를 출력합니다. `for` 루프를 사용하세요. 그런 다음 `while`만 사용하는 두 번째 버전을 구현하세요.

### 연습문제 3: 숫자 맞히기 루프

고정된 "비밀" 숫자(예: 42)를 생성하고 사용자에게 반복적으로 추측하도록 요청하는 프로그램을 작성하세요. 각 추측 후 "Too high", "Too low", 또는 "Correct!"를 출력합니다. `do-while` 루프를 사용하세요. 추측 횟수를 세고 사용자가 맞추면 출력하세요.

### 연습문제 4: 소수 찾기

2부터 200까지의 모든 소수를 출력하는 프로그램을 작성하세요. 중첩 루프를 사용합니다: 외부 루프는 후보 숫자를 반복하고, 내부 루프는 2부터 후보의 제곱근까지 나눗셈 가능성을 검사합니다. 인수가 발견되면 내부 루프를 조기 종료하기 위해 `break`를 사용하세요. 소수가 아닌 것을 건너뛰기 위해 외부 루프에서 `continue`를 사용하세요.

### 연습문제 5: 패턴 출력기

주어진 홀수 `n`(예: `n = 7`)에 대해 다음 다이아몬드 패턴을 출력하는 프로그램을 작성하세요:

```
   *
  ***
 *****
*******
 *****
  ***
   *
```

공백과 별표에 중첩 루프를 사용하세요. 프로그램은 모든 홀수 값의 `n`에서 작동해야 합니다.

---

## 다음 단계

이제 프로그램의 흐름을 지시할 수 있습니다. 다음으로 [함수와 스코프](./05_Functions_and_Scope.md)에서 코드를 재사용 가능한 블록으로 구성하는 방법을 배워봅시다!
