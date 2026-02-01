# C 언어 기초 빠른 복습

> 다른 프로그래밍 언어 경험이 있는 분을 위한 C 핵심 문법 정리

## 1. C 언어의 특징

### 다른 언어와의 비교

| 특징 | Python/JS | C |
|------|-----------|---|
| **메모리 관리** | 자동 (GC) | 수동 (malloc/free) |
| **타입 시스템** | 동적 타입 | 정적 타입 |
| **실행 방식** | 인터프리터 | 컴파일 |
| **추상화 수준** | 높음 | 낮음 (하드웨어 가까움) |

### C 언어를 배워야 하는 이유

- 시스템 프로그래밍 (OS, 드라이버)
- 임베디드 시스템
- 성능이 중요한 애플리케이션
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

### 형식 지정자 (printf)

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

## 4. 포인터 (C의 핵심!)

### 포인터란?

**메모리 주소를 저장하는 변수**입니다.

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
| `&` | 주소 연산자 | `&x` → x의 주소 |
| `*` | 역참조 연산자 | `*p` → p가 가리키는 값 |

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

## 7. 구조체

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
    struct Person p1 = {"홍길동", 25, 175.5};

    // 멤버 접근 (. 연산자)
    printf("이름: %s\n", p1.name);
    printf("나이: %d\n", p1.age);

    // 멤버 수정
    p1.age = 26;
    strcpy(p1.name, "김철수");

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
    Person p1 = {"홍길동", 25};
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
    Person p1 = {"홍길동", 25};

    birthday(&p1);
    printf("나이: %d\n", p1.age);  // 26

    // 포인터로 접근
    Person *ptr = &p1;
    printf("이름: %s\n", ptr->name);

    return 0;
}
```

---

## 8. 동적 메모리 할당

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

### 메모리 누수 주의

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
    p = NULL;  // dangling pointer 방지
}
```

---

## 9. 헤더 파일

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

## 다음 단계

이제 실제 프로젝트를 만들어보겠습니다!

[03_Project_Calculator.md](./03_Project_Calculator.md) → 첫 번째 프로젝트 시작!
