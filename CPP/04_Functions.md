# 함수

## 1. 함수란?

함수는 특정 작업을 수행하는 코드 블록입니다.

```cpp
#include <iostream>

// 함수 정의
void sayHello() {
    std::cout << "Hello!" << std::endl;
}

int main() {
    sayHello();  // 함수 호출
    sayHello();
    return 0;
}
```

### 함수의 구조

```cpp
반환타입 함수이름(매개변수) {
    // 함수 본문
    return 값;  // 반환값 (void면 생략 가능)
}
```

---

## 2. 함수 선언과 정의

### 선언 (프로토타입)

```cpp
#include <iostream>

// 함수 선언 (프로토타입)
int add(int a, int b);

int main() {
    std::cout << add(3, 5) << std::endl;  // 8
    return 0;
}

// 함수 정의
int add(int a, int b) {
    return a + b;
}
```

### 헤더 파일에 선언

```cpp
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);

#endif
```

```cpp
// math_utils.cpp
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

```cpp
// main.cpp
#include <iostream>
#include "math_utils.h"

int main() {
    std::cout << add(3, 5) << std::endl;
    std::cout << multiply(3, 5) << std::endl;
    return 0;
}
```

---

## 3. 매개변수 전달 방식

### 값에 의한 전달 (Pass by Value)

```cpp
#include <iostream>

void increment(int n) {  // n은 복사본
    n++;
    std::cout << "함수 내: " << n << std::endl;
}

int main() {
    int x = 10;
    increment(x);
    std::cout << "함수 후: " << x << std::endl;  // 10 (변경 안 됨)
    return 0;
}
```

### 참조에 의한 전달 (Pass by Reference)

```cpp
#include <iostream>

void increment(int& n) {  // n은 원본의 참조
    n++;
    std::cout << "함수 내: " << n << std::endl;
}

int main() {
    int x = 10;
    increment(x);
    std::cout << "함수 후: " << x << std::endl;  // 11 (변경됨)
    return 0;
}
```

### 포인터에 의한 전달 (Pass by Pointer)

```cpp
#include <iostream>

void increment(int* n) {  // n은 주소
    (*n)++;
    std::cout << "함수 내: " << *n << std::endl;
}

int main() {
    int x = 10;
    increment(&x);  // 주소 전달
    std::cout << "함수 후: " << x << std::endl;  // 11
    return 0;
}
```

### const 참조 (읽기 전용)

```cpp
#include <iostream>
#include <string>

// 복사 없이 읽기만 (효율적)
void printLength(const std::string& str) {
    std::cout << "길이: " << str.length() << std::endl;
    // str[0] = 'x';  // 에러! const이므로 수정 불가
}

int main() {
    std::string name = "Hello";
    printLength(name);
    return 0;
}
```

### 언제 어떤 방식을 사용할까?

| 상황 | 권장 방식 |
|------|----------|
| 작은 타입 (int, double) 읽기 | 값 전달 |
| 큰 타입 읽기 | `const T&` |
| 수정 필요 | `T&` |
| nullptr 허용 필요 | `T*` |

---

## 4. 반환값

### 단일 값 반환

```cpp
int square(int n) {
    return n * n;
}
```

### 참조 반환 (주의 필요)

```cpp
#include <iostream>

int& getElement(int arr[], int index) {
    return arr[index];  // 배열 요소의 참조 반환
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    getElement(arr, 2) = 100;  // arr[2] = 100
    std::cout << arr[2] << std::endl;  // 100

    return 0;
}
```

### 여러 값 반환 (C++17 구조적 바인딩)

```cpp
#include <iostream>
#include <tuple>

std::tuple<int, int, int> getStats(int arr[], int size) {
    int sum = 0, min = arr[0], max = arr[0];
    for (int i = 0; i < size; i++) {
        sum += arr[i];
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    return {sum, min, max};
}

int main() {
    int arr[] = {5, 2, 8, 1, 9};

    // C++17 구조적 바인딩
    auto [sum, min, max] = getStats(arr, 5);
    std::cout << "합: " << sum << ", 최소: " << min << ", 최대: " << max << std::endl;

    return 0;
}
```

---

## 5. 기본 매개변수 (Default Parameters)

```cpp
#include <iostream>

void greet(std::string name = "Guest", int times = 1) {
    for (int i = 0; i < times; i++) {
        std::cout << "Hello, " << name << "!" << std::endl;
    }
}

int main() {
    greet();                // Hello, Guest!
    greet("Alice");         // Hello, Alice!
    greet("Bob", 3);        // Hello, Bob! (3번)
    return 0;
}
```

### 규칙

```cpp
// 기본값은 오른쪽부터
void func(int a, int b = 10, int c = 20);  // OK
// void func(int a = 5, int b, int c = 20);  // 에러!

// 선언과 정의 중 한 곳에만
void func(int a, int b = 10);  // 선언에 기본값
void func(int a, int b) { }    // 정의에는 기본값 없음
```

---

## 6. 함수 오버로딩 (Function Overloading)

같은 이름, 다른 매개변수를 가진 여러 함수를 정의할 수 있습니다.

```cpp
#include <iostream>

// 정수 덧셈
int add(int a, int b) {
    return a + b;
}

// 실수 덧셈
double add(double a, double b) {
    return a + b;
}

// 세 개 덧셈
int add(int a, int b, int c) {
    return a + b + c;
}

int main() {
    std::cout << add(3, 5) << std::endl;        // int 버전: 8
    std::cout << add(3.5, 2.5) << std::endl;    // double 버전: 6.0
    std::cout << add(1, 2, 3) << std::endl;     // 세 개 버전: 6
    return 0;
}
```

### 오버로딩 규칙

```cpp
// 매개변수 타입이 다르면 OK
void print(int n);
void print(double n);
void print(std::string s);

// 매개변수 개수가 다르면 OK
void print(int a);
void print(int a, int b);

// 반환 타입만 다르면 오버로딩 불가!
// int func(int a);
// double func(int a);  // 에러!
```

---

## 7. inline 함수

짧은 함수의 호출 오버헤드를 줄입니다.

```cpp
#include <iostream>

inline int square(int n) {
    return n * n;
}

int main() {
    std::cout << square(5) << std::endl;  // 컴파일러가 25로 대체할 수 있음
    return 0;
}
```

### 사용 시점

- 함수 본문이 짧을 때 (1~2줄)
- 자주 호출되는 함수
- 컴파일러가 최종 결정 (inline은 힌트)

---

## 8. 재귀 함수 (Recursive Function)

자기 자신을 호출하는 함수입니다.

### 팩토리얼

```cpp
#include <iostream>

int factorial(int n) {
    if (n <= 1) return 1;       // 기저 조건
    return n * factorial(n - 1); // 재귀 호출
}

int main() {
    std::cout << "5! = " << factorial(5) << std::endl;  // 120
    return 0;
}
```

### 피보나치

```cpp
#include <iostream>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    for (int i = 0; i < 10; i++) {
        std::cout << fibonacci(i) << " ";
    }
    std::cout << std::endl;  // 0 1 1 2 3 5 8 13 21 34
    return 0;
}
```

### 재귀 vs 반복

```cpp
// 반복 버전 (효율적)
int factorialLoop(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

---

## 9. 함수 포인터

함수를 변수처럼 저장하고 전달할 수 있습니다.

```cpp
#include <iostream>

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

int main() {
    // 함수 포인터 선언
    int (*operation)(int, int);

    operation = add;
    std::cout << "Add: " << operation(5, 3) << std::endl;  // 8

    operation = subtract;
    std::cout << "Subtract: " << operation(5, 3) << std::endl;  // 2

    operation = multiply;
    std::cout << "Multiply: " << operation(5, 3) << std::endl;  // 15

    return 0;
}
```

### 콜백 함수

```cpp
#include <iostream>

void processArray(int arr[], int size, int (*func)(int)) {
    for (int i = 0; i < size; i++) {
        arr[i] = func(arr[i]);
    }
}

int doubleIt(int n) { return n * 2; }
int squareIt(int n) { return n * n; }

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    processArray(arr, 5, doubleIt);
    for (int n : arr) std::cout << n << " ";  // 2 4 6 8 10
    std::cout << std::endl;

    processArray(arr, 5, squareIt);
    for (int n : arr) std::cout << n << " ";  // 4 16 36 64 100
    std::cout << std::endl;

    return 0;
}
```

---

## 10. 람다 표현식 (Lambda) - 미리보기

C++11부터 익명 함수를 만들 수 있습니다.

```cpp
#include <iostream>

int main() {
    // 기본 람다
    auto add = [](int a, int b) {
        return a + b;
    };

    std::cout << add(3, 5) << std::endl;  // 8

    // 캡처
    int multiplier = 10;
    auto multiply = [multiplier](int n) {
        return n * multiplier;
    };

    std::cout << multiply(5) << std::endl;  // 50

    return 0;
}
```

---

## 11. main 함수의 매개변수

```cpp
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "인자 개수: " << argc << std::endl;

    for (int i = 0; i < argc; i++) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }

    return 0;
}
```

실행:
```bash
./program hello world
```

출력:
```
인자 개수: 3
argv[0]: ./program
argv[1]: hello
argv[2]: world
```

---

## 12. 실습 예제

### 최대공약수 (유클리드 알고리즘)

```cpp
#include <iostream>

int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int main() {
    std::cout << "gcd(48, 18) = " << gcd(48, 18) << std::endl;  // 6
    std::cout << "gcd(56, 98) = " << gcd(56, 98) << std::endl;  // 14
    return 0;
}
```

### 두 값 교환

```cpp
#include <iostream>

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    std::cout << "전: x=" << x << ", y=" << y << std::endl;

    swap(x, y);
    std::cout << "후: x=" << x << ", y=" << y << std::endl;

    return 0;
}
```

---

## 13. 요약

| 개념 | 설명 |
|------|------|
| 함수 선언 | 함수의 시그니처 (프로토타입) |
| 함수 정의 | 함수의 본문 |
| 값 전달 | 복사본 전달, 원본 변경 안 됨 |
| 참조 전달 | 원본 전달, 변경 가능 |
| const 참조 | 읽기 전용 원본 전달 |
| 기본 매개변수 | 생략 가능한 인자 |
| 오버로딩 | 같은 이름, 다른 매개변수 |
| inline | 함수 호출 대신 코드 삽입 |
| 재귀 | 자기 자신 호출 |

---

## 다음 단계

[05_Arrays_and_Strings.md](./05_Arrays_and_Strings.md)에서 배열과 문자열을 배워봅시다!
