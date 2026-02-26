# 프로젝트 1: 사칙연산 계산기

**이전**: [C 언어 기초 빠른 복습](./02_C_Basics_Review.md) | **다음**: [프로젝트 2: 숫자 맞추기 게임](./04_Project_Number_Guessing.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `scanf`와 형식 지정자(format specifier), 주소 연산자(address-of operator)를 사용하여 사용자 입력을 읽을 수 있습니다
2. `switch-case` 문을 사용하여 여러 분기로 프로그램 흐름을 제어할 수 있습니다
3. 포인터 매개변수(pointer parameter)를 받고 상태 코드(status code)를 반환하는 함수를 정의하고 호출할 수 있습니다
4. 입력 검증(input validation)과 반환 코드(return code)를 사용하여 런타임 오류(runtime error)를 우아하게 처리할 수 있습니다
5. 사용자가 종료를 선택할 때까지 반복 실행되는 프로그램을 설계할 수 있습니다
6. 입력, 계산, 출력을 별도의 함수로 분리하여 관심사 분리(separation of concerns)를 구현할 수 있습니다

---

계산기는 인터랙티브 프로그램의 "Hello World"라고 할 수 있습니다. 쓰레기 값이 들어올 수 있는 사용자 입력 읽기, 올바른 연산 선택, 0으로 나누기 같은 오류 보고 등 현실에서 마주치는 지저분한 세부 사항을 한꺼번에 처리해야 합니다. 이 프로젝트를 마치면 작지만 완전한 커맨드라인 도구를 갖게 되며, 더 중요하게는 어떤 인터랙티브 C 프로그램을 구조화할 때도 재사용할 수 있는 패턴을 익히게 됩니다.

## 1단계: 기본 계산기

### 요구사항

```
두 수와 연산자를 입력받아 결과 출력
예: 10 + 5 → 결과: 15
```

### 핵심 문법: scanf

```c
#include <stdio.h>

int main(void) {
    int num;
    printf("Enter a number: ");
    scanf("%d", &num);        // & required! (pass address)
    printf("You entered: %d\n", num);

    // Multiple values
    int a, b;
    printf("Enter two numbers (space-separated): ");
    scanf("%d %d", &a, &b);
    printf("a=%d, b=%d\n", a, b);

    // Character input
    char op;
    printf("Enter operator: ");
    scanf(" %c", &op);        // Space before %c: ignore previous newline
    printf("Operator: %c\n", op);

    return 0;
}
```

### 핵심 문법: switch-case

```c
char grade = 'B';

switch (grade) {
    case 'A':
        printf("Excellent\n");
        break;
    case 'B':
        printf("Good\n");
        break;
    case 'C':
        printf("Average\n");
        break;
    default:
        printf("Other\n");
        break;
}
```

### 구현

```c
// calculator_v1.c
#include <stdio.h>

int main(void) {
    double num1, num2;
    char operator;

    printf("=== Simple Calculator ===\n");
    printf("Enter expression (e.g., 10 + 5): ");
    scanf("%lf %c %lf", &num1, &operator, &num2);

    double result;

    switch (operator) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        case '/':
            result = num1 / num2;
            break;
        default:
            printf("Error: Unsupported operator.\n");
            return 1;
    }

    printf("Result: %.2f %c %.2f = %.2f\n", num1, operator, num2, result);

    return 0;
}
```

### 실행 예시

```
$ ./calculator_v1
=== Simple Calculator ===
Enter expression (e.g., 10 + 5): 10 + 5
Result: 10.00 + 5.00 = 15.00

$ ./calculator_v1
Enter expression (e.g., 10 + 5): 20 / 4
Result: 20.00 / 4.00 = 5.00
```

---

## 2단계: 에러 처리 추가

### 문제점

```
20 / 0 → 결과: inf (무한대) 또는 에러
```

### 개선된 코드

```c
// calculator_v2.c
#include <stdio.h>

int main(void) {
    double num1, num2;
    char operator;

    printf("=== Calculator v2 ===\n");
    printf("Enter expression (e.g., 10 + 5): ");

    // Input validation
    if (scanf("%lf %c %lf", &num1, &operator, &num2) != 3) {
        printf("Error: Invalid input format.\n");
        return 1;
    }

    double result;
    int error = 0;

    switch (operator) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        case '/':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                error = 1;
            } else {
                result = num1 / num2;
            }
            break;
        default:
            printf("Error: '%c' is not a supported operator.\n", operator);
            error = 1;
            break;
    }

    if (!error) {
        printf("Result: %.2f %c %.2f = %.2f\n", num1, operator, num2, result);
    }

    return error;
}
```

---

## 3단계: 함수로 분리

### 구조

```
main() → get_input() → 입력 받기
       → calculate() → 계산 수행
       → 결과 출력
```

### 완성 코드

```c
// calculator_v3.c
#include <stdio.h>

// 함수 선언
int get_input(double *num1, char *op, double *num2);
int calculate(double num1, char op, double num2, double *result);
void print_result(double num1, char op, double num2, double result);

int main(void) {
    double num1, num2, result;
    char operator;

    printf("=== Calculator v3 ===\n");

    // 입력 받기
    if (get_input(&num1, &operator, &num2) != 0) {
        printf("Error: Invalid input format.\n");
        return 1;
    }

    // 계산
    if (calculate(num1, operator, num2, &result) != 0) {
        return 1;
    }

    // 결과 출력
    print_result(num1, operator, num2, result);

    return 0;
}

// 입력 함수
int get_input(double *num1, char *op, double *num2) {
    printf("Enter expression (e.g., 10 + 5): ");
    if (scanf("%lf %c %lf", num1, op, num2) != 3) {
        return -1;  // 에러
    }
    return 0;  // 성공
}

// 계산 함수
int calculate(double num1, char op, double num2, double *result) {
    switch (op) {
        case '+':
            *result = num1 + num2;
            break;
        case '-':
            *result = num1 - num2;
            break;
        case '*':
            *result = num1 * num2;
            break;
        case '/':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = num1 / num2;
            break;
        case '%':
            // 정수 나머지 연산
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = (int)num1 % (int)num2;
            break;
        default:
            printf("Error: '%c' is not a supported operator.\n", op);
            return -1;
    }
    return 0;
}

// 출력 함수
void print_result(double num1, char op, double num2, double result) {
    printf("Result: %.2f %c %.2f = %.2f\n", num1, op, num2, result);
}
```

---

## 4단계: 반복 계산 (최종 버전)

### 완성 코드

```c
// calculator.c (최종)
#include <stdio.h>
#include <stdlib.h>

// 함수 선언
int get_input(double *num1, char *op, double *num2);
int calculate(double num1, char op, double num2, double *result);
void print_result(double num1, char op, double num2, double result);
void print_help(void);
void clear_input_buffer(void);

int main(void) {
    double num1, num2, result;
    char operator;
    char continue_calc;

    printf("=============================\n");
    printf("     Simple Calculator v4    \n");
    printf("=============================\n");
    print_help();

    do {
        // 입력 받기
        if (get_input(&num1, &operator, &num2) != 0) {
            printf("Error: Invalid input format.\n");
            clear_input_buffer();
            continue;
        }

        // 계산
        if (calculate(num1, operator, num2, &result) == 0) {
            // 결과 출력
            print_result(num1, operator, num2, result);
        }

        // 계속 여부
        printf("\nContinue? (y/n): ");
        scanf(" %c", &continue_calc);
        clear_input_buffer();
        printf("\n");

    } while (continue_calc == 'y' || continue_calc == 'Y');

    printf("Exiting calculator.\n");
    return 0;
}

int get_input(double *num1, char *op, double *num2) {
    printf("\nEnter expression: ");
    if (scanf("%lf %c %lf", num1, op, num2) != 3) {
        return -1;
    }
    return 0;
}

int calculate(double num1, char op, double num2, double *result) {
    switch (op) {
        case '+':
            *result = num1 + num2;
            break;
        case '-':
            *result = num1 - num2;
            break;
        case '*':
        case 'x':
        case 'X':
            *result = num1 * num2;
            break;
        case '/':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = num1 / num2;
            break;
        case '%':
            if (num2 == 0) {
                printf("Error: Cannot divide by zero.\n");
                return -1;
            }
            *result = (int)num1 % (int)num2;
            break;
        case '^':
            // 간단한 거듭제곱 (양의 정수만)
            *result = 1;
            for (int i = 0; i < (int)num2; i++) {
                *result *= num1;
            }
            break;
        default:
            printf("Error: '%c' is not a supported operator.\n", op);
            return -1;
    }
    return 0;
}

void print_result(double num1, char op, double num2, double result) {
    printf(">>> %.4g %c %.4g = %.4g\n", num1, op, num2, result);
}

void print_help(void) {
    printf("\nSupported operators: + - * / %% ^\n");
    printf("Input format: number operator number\n");
    printf("Examples: 10 + 5, 20 / 4, 2 ^ 10\n");
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

### 실행 예시

```
=============================
     Simple Calculator v4
=============================

Supported operators: + - * / % ^
Input format: number operator number
Examples: 10 + 5, 20 / 4, 2 ^ 10

Enter expression: 100 + 250
>>> 100 + 250 = 350

Continue? (y/n): y

Enter expression: 2 ^ 10
>>> 2 ^ 10 = 1024

Continue? (y/n): y

Enter expression: 10 / 0
Error: Cannot divide by zero.

Continue? (y/n): n

Exiting calculator.
```

---

## 컴파일 및 실행

```bash
# 컴파일
gcc -Wall -Wextra -std=c11 calculator.c -o calculator

# 실행
./calculator
```

---

## 배운 내용 정리

| 개념 | 설명 |
|------|------|
| `scanf` | 형식에 맞게 입력 받기 |
| `switch-case` | 값에 따른 분기 처리 |
| 함수 분리 | 코드 구조화, 재사용성 |
| 포인터 매개변수 | 함수에서 값 변경하기 |
| 에러 처리 | 반환값으로 성공/실패 표시 |

---

## 연습 문제

1. **제곱근 연산 추가**: `sqrt` 연산자 추가 (힌트: `#include <math.h>`, `sqrt()`)

2. **계산 이력 저장**: 최근 10개 계산 결과를 배열에 저장하고 출력하는 기능 추가

3. **괄호 지원**: `(10 + 5) * 2` 같은 수식 처리 (어려움 주의!)

---

## 다음 단계

[프로젝트 2: 숫자 맞추기 게임](./04_Project_Number_Guessing.md) → 게임을 만들어봅시다!

---

**이전**: [C 언어 기초 빠른 복습](./02_C_Basics_Review.md) | **다음**: [프로젝트 2: 숫자 맞추기 게임](./04_Project_Number_Guessing.md)
