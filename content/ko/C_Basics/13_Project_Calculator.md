# 프로젝트: 기본 산술 계산기

**이전**: [빌드 도구와 디버깅](./12_Build_Tools_and_Debugging.md) | **다음**: [프로젝트: 숫자 맞추기 게임](./14_Project_Number_Guessing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 적절한 서식 지정자와 주소 연산자를 사용하여 `scanf`로 사용자 입력 읽기
2. `switch-case` 문을 사용하여 여러 분기로 프로그램 흐름 라우팅하기
3. 포인터 매개변수를 받고 상태 코드를 반환하는 함수 정의 및 호출하기
4. 입력 유효성 검사와 반환 코드를 사용하여 런타임 오류를 우아하게 처리하기
5. 사용자가 종료를 선택할 때까지 연속으로 반복하는 프로그램 설계하기
6. 입력, 계산, 출력을 별도의 함수로 추출하여 관심사 분리하기

---

계산기는 대화형 프로그램의 "Hello World"입니다. 한꺼번에 모든 실세계의 지저분한 세부사항을 처리해야 합니다 -- 쓰레기일 수 있는 사용자 입력 읽기, 올바른 연산 선택, 0으로 나누기 같은 오류 보고. 이 프로젝트를 마치면 작지만 완전한 명령줄 도구를 갖게 되며, 더 중요한 것은 모든 대화형 C 프로그램을 구조화하는 재사용 가능한 패턴을 갖게 됩니다.

## 단계 1: 기본 계산기

### 요구사항

```
두 숫자와 연산자를 입력받아 결과를 출력
예시: 10 + 5 → Result: 15
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

## 단계 2: 오류 처리 추가

### 문제

```
20 / 0 → Result: inf (infinity) 또는 오류
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

## 단계 3: 함수로 분리

### 구조

```
main() → get_input() → 입력 받기
       → calculate() → 계산 수행
       → 결과 출력
```

### 완전한 코드

```c
// calculator_v3.c
#include <stdio.h>

// Function declarations
int get_input(double *num1, char *op, double *num2);
int calculate(double num1, char op, double num2, double *result);
void print_result(double num1, char op, double num2, double result);

int main(void) {
    double num1, num2, result;
    char operator;

    printf("=== Calculator v3 ===\n");

    // Get input
    if (get_input(&num1, &operator, &num2) != 0) {
        printf("Error: Invalid input format.\n");
        return 1;
    }

    // Calculate
    if (calculate(num1, operator, num2, &result) != 0) {
        return 1;
    }

    // Print result
    print_result(num1, operator, num2, result);

    return 0;
}

// Input function
int get_input(double *num1, char *op, double *num2) {
    printf("Enter expression (e.g., 10 + 5): ");
    if (scanf("%lf %c %lf", num1, op, num2) != 3) {
        return -1;  // Error
    }
    return 0;  // Success
}

// Calculate function
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
            // Integer modulo operation
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

// Output function
void print_result(double num1, char op, double num2, double result) {
    printf("Result: %.2f %c %.2f = %.2f\n", num1, op, num2, result);
}
```

---

## 단계 4: 반복 계산 (최종 버전)

### 완전한 코드

```c
// calculator.c (final)
#include <stdio.h>
#include <stdlib.h>

// Function declarations
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
        // Get input
        if (get_input(&num1, &operator, &num2) != 0) {
            printf("Error: Invalid input format.\n");
            clear_input_buffer();
            continue;
        }

        // Calculate
        if (calculate(num1, operator, num2, &result) == 0) {
            // Print result
            print_result(num1, operator, num2, result);
        }

        // Continue?
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
            // Simple exponentiation (positive integers only)
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

## 컴파일과 실행

```bash
# Compile
gcc -Wall -Wextra -std=c11 calculator.c -o calculator

# Run
./calculator
```

---

## 배운 내용 정리

| 개념 | 설명 |
|------|------|
| `scanf` | 지정된 형식으로 입력 읽기 |
| `switch-case` | 값에 따라 분기 |
| 함수 분리 | 코드 구조화, 재사용성 |
| 포인터 매개변수 | 함수에서 값 수정 |
| 오류 처리 | 반환 값을 사용하여 성공/실패 표시 |

---

## 연습문제

1. **제곱근 연산 추가**: `sqrt` 연산자 추가 (힌트: `#include <math.h>`, `sqrt()`)

2. **계산 히스토리**: 마지막 10개의 계산 결과를 배열에 저장하고 표시

3. **괄호 지원**: `(10 + 5) * 2`와 같은 표현식 처리 (도전 과제!)

4. **Makefile 통합**: `-Wall -Wextra -std=c11 -g`로 계산기를 컴파일하고 `clean` 타겟을 포함하는 Makefile 작성

5. **다중 파일 분리**: 계산기를 `main.c`, `calc.c`/`calc.h`, `io.c`/`io.h`로 분리한 후 모든 파일을 컴파일하도록 Makefile 업데이트

---

## 다음 단계

[프로젝트: 숫자 맞추기 게임](./14_Project_Number_Guessing.md) -- 루프, 조건문, 난수 생성을 강화하는 게임을 만들어 보세요.
