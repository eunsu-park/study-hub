# C 언어 기초 빠른 복습

**이전**: [C 언어 환경 설정](./01_Environment_Setup.md) | **다음**: [프로젝트 1: 사칙연산 계산기](./03_Project_Calculator.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C의 정적 타입(static typing), 수동 메모리 관리(manual memory management), 컴파일 실행(compiled execution)을 Python 또는 JavaScript의 동등 개념과 구별하기
2. 최소한의 C 프로그램에서 `#include`, `main`, 세미콜론, 중괄호의 역할 설명하기
3. `printf`와 `scanf` 사용 시 각 기본 자료형에 맞는 올바른 형식 지정자(format specifier) 식별하기
4. 포인터 선언(pointer declaration), 주소 연산자(`&`), 역참조 연산자(`*`)를 구현하여 값을 참조로 전달하기
5. 고정 크기 배열(fixed-size array)을 만들고 순회하며, `arr[i]`와 `*(arr + i)`의 동등성 설명하기
6. C에서 안전한 문자열 조작을 위해 `strlen`, `strcpy`, `strcat`, `strcmp` 적용하기
7. `typedef`를 사용하여 구조체(struct)를 설계하고, 점(`.`) 연산자와 화살표(`->`) 연산자 모두를 통해 멤버에 접근하며, 포인터로 구조체를 함수에 전달하기
8. `malloc`, `free`, NULL 검사 패턴을 적용하여 메모리 누수(memory leak) 없이 힙(heap) 메모리를 할당하고 해제하기

---

Python, JavaScript, 또는 다른 고급 언어를 이미 알고 있다면, 변수, 반복문, 함수, 자료구조 등 대부분의 프로그래밍 개념은 이미 익숙할 것입니다. C가 특별한 이유는 하드웨어와 얼마나 가까이 있는가에 있습니다: 모든 정수의 정확한 크기를 직접 선택하고, 모든 메모리 바이트를 직접 관리하며, 운영체제와 직접 소통합니다. 이 레슨에서는 C의 핵심 문법을 빠르게 훑어봄으로써, 다음 레슨에서 바로 실제 프로젝트를 시작할 수 있도록 합니다.

> 다른 프로그래밍 언어 경험이 있는 분을 위한 C 핵심 문법 정리

## 1. C 언어의 특징

### 다른 언어와의 비교

| 특징 | Python/JS | C |
|------|-----------|---|
| **메모리 관리** | 자동 (GC) | 수동 (malloc/free) |
| **타입 시스템** | 동적 타입(dynamic typing) | 정적 타입(static typing) |
| **실행 방식** | 인터프리터(interpreter) | 컴파일(compiled) |
| **추상화 수준** | 높음 | 낮음 (하드웨어 가까움) |

### C 언어를 배워야 하는 이유

- 시스템 프로그래밍(systems programming) (OS, 드라이버)
- 임베디드 시스템(embedded systems)
- 성능이 중요한 애플리케이션(performance-critical applications)
- 다른 언어의 기반 이해 (Python, Ruby는 C로 작성)

---

## 2. 기본 구조

```c
#include <stdio.h>    // 헤더 파일 포함 (전처리기 지시문)

// main 함수: 프로그램 시작점
int main(void) {
    printf("Hello, C!\n");
    return 0;         // 0 = 정상 종료
}
```

### Python과 비교

```python
# Python
print("Hello, Python!")
```

```c
// C
#include <stdio.h>
int main(void) {
    printf("Hello, C!\n");
    return 0;
}
```

**C의 특징:**
- 세미콜론 `;` 필수
- 중괄호 `{}` 로 블록 구분
- 명시적인 main 함수
- 헤더 파일 include 필요

---

## 3. 자료형

### 기본 자료형

```c
#include <stdio.h>

int main(void) {
    // 정수형
    char c = 'A';           // 1바이트 (-128 ~ 127)
    short s = 100;          // 2바이트
    int i = 1000;           // 4바이트 (보통)
    long l = 100000L;       // 4 또는 8바이트
    long long ll = 100000000000LL;  // 8바이트

    // 부호 없는 정수
    unsigned int ui = 4000000000U;

    // 실수형
    float f = 3.14f;        // 4바이트
    double d = 3.14159265;  // 8바이트

    // 출력
    printf("char: %c (%d)\n", c, c);  // A (65)
    printf("int: %d\n", i);
    printf("float: %f\n", f);
    printf("double: %.8f\n", d);

    return 0;
}
```

### 형식 지정자(Format Specifier) (printf)

| 지정자 | 타입 | 예시 |
|--------|------|------|
| `%d` | int | `printf("%d", 42)` |
| `%u` | unsigned int | `printf("%u", 42)` |
| `%ld` | long | `printf("%ld", 42L)` |
| `%f` | float/double | `printf("%f", 3.14)` |
| `%c` | char | `printf("%c", 'A')` |
| `%s` | 문자열 | `printf("%s", "hello")` |
| `%p` | 포인터 주소 | `printf("%p", &x)` |
| `%x` | 16진수 | `printf("%x", 255)` → ff |

### sizeof 연산자

```c
printf("int 크기: %zu 바이트\n", sizeof(int));
printf("double 크기: %zu 바이트\n", sizeof(double));
printf("포인터 크기: %zu 바이트\n", sizeof(int*));
```

---

## 4. 포인터(Pointer) (C의 핵심!)

### 포인터란?

**메모리 주소(memory address)를 저장하는 변수**입니다.

```
메모리:
주소        값
0x1000     42      ← int x = 42;
0x1004     0x1000  ← int *p = &x;  (x의 주소 저장)
```

### 기본 문법

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;      // p는 x의 주소를 저장

    printf("x의 값: %d\n", x);        // 42
    printf("x의 주소: %p\n", &x);     // 0x7fff...
    printf("p의 값 (주소): %p\n", p); // 0x7fff... (같은 주소)
    printf("p가 가리키는 값: %d\n", *p);  // 42 (역참조)

    // 포인터로 값 변경
    *p = 100;
    printf("x의 새 값: %d\n", x);     // 100

    return 0;
}
```

### 포인터 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `&` | 주소 연산자(address operator) | `&x` → x의 주소 |
| `*` | 역참조 연산자(dereference operator) | `*p` → p가 가리키는 값 |

### 왜 포인터가 필요한가?

```c
// 문제: C에서 함수는 값을 복사해서 전달 (call by value)
void wrong_swap(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    // 원본은 변경되지 않음!
}

// 해결: 포인터로 주소 전달
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    // 원본이 변경됨!
}

int main(void) {
    int x = 10, y = 20;

    wrong_swap(x, y);
    printf("wrong_swap 후: x=%d, y=%d\n", x, y);  // 10, 20 (변화 없음)

    swap(&x, &y);
    printf("swap 후: x=%d, y=%d\n", x, y);  // 20, 10

    return 0;
}
```

---

## 5. 배열

### 기본 배열

```c
#include <stdio.h>

int main(void) {
    // 배열 선언 및 초기화
    int numbers[5] = {10, 20, 30, 40, 50};

    // 접근
    printf("%d\n", numbers[0]);  // 10
    printf("%d\n", numbers[4]);  // 50

    // 크기
    int size = sizeof(numbers) / sizeof(numbers[0]);
    printf("배열 크기: %d\n", size);  // 5

    // 순회
    for (int i = 0; i < size; i++) {
        printf("numbers[%d] = %d\n", i, numbers[i]);
    }

    return 0;
}
```

### 배열과 포인터의 관계

```c
int arr[5] = {1, 2, 3, 4, 5};

// 배열 이름은 첫 번째 요소의 주소
printf("%p\n", arr);      // 첫 번째 요소 주소
printf("%p\n", &arr[0]);  // 같은 주소

// 포인터 연산
int *p = arr;
printf("%d\n", *p);       // 1 (arr[0])
printf("%d\n", *(p + 1)); // 2 (arr[1])
printf("%d\n", *(p + 2)); // 3 (arr[2])

// arr[i] == *(arr + i)
```

### 문자열 (char 배열)

```c
#include <stdio.h>
#include <string.h>  // 문자열 함수

int main(void) {
    // 문자열은 char 배열 + 널 종료 문자 '\0'
    char str1[] = "Hello";        // 자동으로 '\0' 추가
    char str2[10] = "World";
    char str3[] = {'H', 'i', '\0'};

    printf("%s\n", str1);         // Hello
    printf("길이: %zu\n", strlen(str1));  // 5

    // 문자열 복사
    char dest[20];
    strcpy(dest, str1);           // dest = "Hello"

    // 문자열 연결
    strcat(dest, " ");
    strcat(dest, str2);           // dest = "Hello World"
    printf("%s\n", dest);

    // 문자열 비교
    if (strcmp(str1, "Hello") == 0) {
        printf("같음!\n");
    }

    return 0;
}
```

---

## 6. 함수

### 기본 함수

```c
#include <stdio.h>

// 함수 선언 (프로토타입)
int add(int a, int b);
void greet(const char *name);

int main(void) {
    int result = add(3, 5);
    printf("3 + 5 = %d\n", result);

    greet("Alice");
    return 0;
}

// 함수 정의
int add(int a, int b) {
    return a + b;
}

void greet(const char *name) {
    printf("Hello, %s!\n", name);
}
```

### 배열을 함수에 전달

```c
// 배열은 포인터로 전달됨 (크기 정보 없음)
void print_array(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 또는 이렇게 표기 (동일한 의미)
void print_array2(int arr[], int size) {
    // ...
}

int main(void) {
    int nums[] = {1, 2, 3, 4, 5};
    print_array(nums, 5);
    return 0;
}
```

---

## 7. 구조체(Struct)

### 기본 구조체

```c
#include <stdio.h>
#include <string.h>

// 구조체 정의
struct Person {
    char name[50];
    int age;
    float height;
};

int main(void) {
    // 구조체 변수 선언 및 초기화
    struct Person p1 = {"John Doe", 25, 175.5};

    // 멤버 접근 (. 연산자)
    printf("이름: %s\n", p1.name);
    printf("나이: %d\n", p1.age);

    // 멤버 수정
    p1.age = 26;
    strcpy(p1.name, "Jane Smith");

    return 0;
}
```

### typedef로 간단하게

```c
typedef struct {
    char name[50];
    int age;
} Person;  // 이제 'struct' 키워드 없이 사용

int main(void) {
    Person p1 = {"John Doe", 25};
    printf("%s\n", p1.name);
    return 0;
}
```

### 포인터와 구조체

```c
typedef struct {
    char name[50];
    int age;
} Person;

void birthday(Person *p) {
    p->age++;  // 포인터는 -> 연산자 사용
    // (*p).age++; 와 동일
}

int main(void) {
    Person p1 = {"John Doe", 25};

    birthday(&p1);
    printf("나이: %d\n", p1.age);  // 26

    // 포인터로 접근
    Person *ptr = &p1;
    printf("이름: %s\n", ptr->name);

    return 0;
}
```

---

## 8. 동적 메모리 할당(Dynamic Memory Allocation)

### malloc / free

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // 정수 하나 동적 할당
    int *p = (int *)malloc(sizeof(int));
    if (p == NULL) {
        printf("메모리 할당 실패\n");
        return 1;
    }
    *p = 42;
    printf("%d\n", *p);
    free(p);  // 메모리 해제 (필수!)

    // 배열 동적 할당
    int n = 5;
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        return 1;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);  // 배열도 해제 필수!

    return 0;
}
```

### 메모리 누수(Memory Leak) 주의

```c
// 나쁜 예: 메모리 누수
void bad(void) {
    int *p = malloc(sizeof(int));
    *p = 42;
    // free(p); 없음 → 메모리 누수!
}

// 좋은 예
void good(void) {
    int *p = malloc(sizeof(int));
    if (p == NULL) return;
    *p = 42;
    // 사용 후...
    free(p);
    p = NULL;  // 댕글링 포인터(dangling pointer) 방지
}
```

---

## 9. 헤더 파일(Header File)

### 헤더 파일 구조

```c
// utils.h
#ifndef UTILS_H      // include guard
#define UTILS_H

// 함수 선언
int add(int a, int b);
int subtract(int a, int b);

// 구조체 정의
typedef struct {
    int x, y;
} Point;

#endif
```

```c
// utils.c
#include "utils.h"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}
```

```c
// main.c
#include <stdio.h>
#include "utils.h"

int main(void) {
    printf("%d\n", add(3, 5));
    Point p = {10, 20};
    printf("(%d, %d)\n", p.x, p.y);
    return 0;
}
```

### 컴파일

```bash
gcc main.c utils.c -o program
```

---

## 10. 주요 차이점 요약 (Python → C)

| Python | C |
|--------|---|
| `print("Hello")` | `printf("Hello\n");` |
| `x = 10` | `int x = 10;` |
| `if x > 5:` | `if (x > 5) {` |
| `for i in range(5):` | `for (int i = 0; i < 5; i++) {` |
| `def func(x):` | `int func(int x) {` |
| `class Person:` | `struct Person {` |
| 자동 메모리 관리 | `malloc()` / `free()` |
| `len(arr)` | `sizeof(arr)/sizeof(arr[0])` |

---

## 연습 문제

### 연습 1: 자료형 크기와 형식 지정자(Format Specifier)

3절에서 다룬 모든 자료형(`char`, `short`, `int`, `long`, `long long`, `unsigned int`, `float`, `double`)의 크기(바이트 단위)와 샘플 값을 출력하는 프로그램을 작성하세요. 각 타입에 맞는 올바른 형식 지정자를 사용하세요. 그런 다음 다음에 답하세요:

1. `sizeof` 출력에 `%d` 대신 `%zu`를 사용하는 이유는 무엇인가요?
2. 64비트 시스템에서 `long long` 값을 출력할 때 `%d`를 사용하면 어떤 일이 발생하나요?

### 연습 2: 포인터 swap 해부

4절의 `wrong_swap` / `swap` 예제를 복사하여 다음과 같이 확장하세요:

1. `wrong_swap` 내부에 `%p`를 사용하여 `a`와 `b`의 *주소(address)*를 출력하는 `printf` 문을 추가하세요. `swap` 내부의 포인터 매개변수에도 동일하게 하세요.
2. `main`에서 두 함수를 호출하고, 각 호출 전에 `x`와 `y`의 주소를 출력하세요.
3. 다음을 서면으로 확인하세요: `wrong_swap` 내부에서 출력된 주소가 `x`와 `y`의 주소와 다른 이유는 무엇이고, `swap` 내부의 주소는 왜 동일한가요?

### 연습 3: 배열과 포인터 산술(Pointer Arithmetic)

`int arr[] = {10, 20, 30, 40, 50}`을 선언하는 프로그램을 작성하고:

1. 인덱스(`arr[i]`)를 사용하여 배열을 순회하며 각 요소를 출력하세요.
2. 포인터(`int *p = arr; ... *(p + i)`)를 사용하여 다시 순회하며 각 요소를 출력하세요.
3. 인덱스 없이 포인터 자체를 증가시켜(`p++`) 세 번째로 순회하세요.
4. `sizeof`를 사용하여 요소 개수를 계산하고, 세 가지 루프가 동일한 출력을 생성하는지 확인하세요.

### 연습 4: 동적 문자열 빌더(Dynamic String Builder)

다음 기능을 하는 `char *build_greeting(const char *name)` 함수를 구현하세요:

1. `"Hello, <name>!"` 문자열을 저장하기에 정확히 적당한 메모리를 동적으로 할당하세요(`strlen`과 `malloc` 사용).
2. `strcpy`와 `strcat`을 사용하여 문자열을 구성하세요.
3. 호출자에게 포인터를 반환하며, 호출자가 `free`를 호출할 책임을 집니다.

`build_greeting`을 호출하고, 결과를 출력하고, 메모리를 해제하는 `main`을 작성하세요. Valgrind가 있으면 프로그램을 실행하거나, 수동 확인을 추가하여 메모리 누수(memory leak)가 없음을 확인하세요.

### 연습 5: 구조체(Struct) 기반 학생 레코드

`char name[64]`, `int id`, `float gpa` 필드를 갖는 `Student`라는 `typedef struct`를 정의하세요. 그런 다음:

1. `malloc`을 사용하여 3개의 `Student` 구조체 배열을 동적으로 할당하세요.
2. `->` 연산자를 사용하여 학생의 세부 정보를 출력하는 `void print_student(const Student *s)` 함수를 작성하세요.
3. `delta`를 학생의 GPA에 더하는(최대 4.0으로 제한) `void raise_gpa(Student *s, float delta)` 함수를 작성하세요.
4. 각 학생에 `raise_gpa`를 호출한 후 `print_student`로 결과를 확인하고, 최종적으로 배열을 `free`하세요.

---

## 다음 단계

이제 실제 프로젝트를 만들어보겠습니다!

[프로젝트 1: 사칙연산 계산기](./03_Project_Calculator.md) → 첫 번째 프로젝트 시작!

**이전**: [C 언어 환경 설정](./01_Environment_Setup.md) | **다음**: [프로젝트 1: 사칙연산 계산기](./03_Project_Calculator.md)
