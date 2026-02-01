# 변수와 자료형

## 1. 변수란?

변수는 데이터를 저장하는 메모리 공간에 붙인 이름입니다.

```cpp
#include <iostream>

int main() {
    int age = 25;           // 정수형 변수
    double height = 175.5;  // 실수형 변수
    char grade = 'A';       // 문자형 변수

    std::cout << "나이: " << age << std::endl;
    std::cout << "키: " << height << std::endl;
    std::cout << "등급: " << grade << std::endl;

    return 0;
}
```

---

## 2. 기본 자료형

### 정수형 (Integer Types)

| 타입 | 크기 | 범위 |
|------|------|------|
| `short` | 2바이트 | -32,768 ~ 32,767 |
| `int` | 4바이트 | 약 -21억 ~ 21억 |
| `long` | 4/8바이트 | 시스템에 따라 다름 |
| `long long` | 8바이트 | 약 -922경 ~ 922경 |

```cpp
#include <iostream>

int main() {
    short s = 32767;
    int i = 2147483647;
    long l = 2147483647L;
    long long ll = 9223372036854775807LL;

    std::cout << "short: " << s << std::endl;
    std::cout << "int: " << i << std::endl;
    std::cout << "long: " << l << std::endl;
    std::cout << "long long: " << ll << std::endl;

    return 0;
}
```

### 부호 없는 정수 (Unsigned)

```cpp
unsigned int positive = 4294967295;  // 0 ~ 약 42억
unsigned short us = 65535;           // 0 ~ 65535
```

### 실수형 (Floating Point Types)

| 타입 | 크기 | 정밀도 |
|------|------|--------|
| `float` | 4바이트 | 약 7자리 |
| `double` | 8바이트 | 약 15자리 |
| `long double` | 8~16바이트 | 시스템에 따라 다름 |

```cpp
#include <iostream>
#include <iomanip>  // setprecision

int main() {
    float f = 3.14159265358979f;
    double d = 3.14159265358979;

    std::cout << std::setprecision(15);
    std::cout << "float: " << f << std::endl;
    std::cout << "double: " << d << std::endl;

    return 0;
}
```

출력:
```
float: 3.14159274101257
double: 3.14159265358979
```

### 문자형 (Character Type)

```cpp
#include <iostream>

int main() {
    char letter = 'A';
    char newline = '\n';
    char tab = '\t';

    std::cout << "문자: " << letter << std::endl;
    std::cout << "ASCII 값: " << (int)letter << std::endl;  // 65

    // 이스케이프 시퀀스
    std::cout << "Tab:\tAfter tab" << std::endl;
    std::cout << "Quote: \"Hello\"" << std::endl;

    return 0;
}
```

### 이스케이프 시퀀스

| 시퀀스 | 의미 |
|--------|------|
| `\n` | 줄바꿈 |
| `\t` | 탭 |
| `\\` | 백슬래시 |
| `\"` | 큰따옴표 |
| `\'` | 작은따옴표 |

### 불리언형 (Boolean Type)

```cpp
#include <iostream>

int main() {
    bool isTrue = true;
    bool isFalse = false;

    std::cout << "true: " << isTrue << std::endl;   // 1
    std::cout << "false: " << isFalse << std::endl; // 0

    // 조건 표현식
    bool result = (5 > 3);  // true
    std::cout << "5 > 3: " << result << std::endl;

    return 0;
}
```

---

## 3. 변수 선언과 초기화

### 선언과 초기화 방식

```cpp
#include <iostream>

int main() {
    // 선언만 (초기화되지 않음 - 쓰레기값)
    int a;

    // 선언과 동시에 초기화
    int b = 10;

    // 중괄호 초기화 (C++11, 권장)
    int c{20};

    // 복사 초기화
    int d = {30};

    // 여러 변수 동시 선언
    int x = 1, y = 2, z = 3;

    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    std::cout << "d: " << d << std::endl;

    return 0;
}
```

### 중괄호 초기화의 장점

```cpp
int a = 3.14;   // 가능 (3으로 잘림, 경고 없을 수 있음)
int b{3.14};    // 컴파일 에러! (좁히기 변환 금지)
int c{3};       // 정확한 값
```

---

## 4. 상수

### const 상수

```cpp
#include <iostream>

int main() {
    const int MAX_SIZE = 100;
    const double PI = 3.14159;

    std::cout << "MAX_SIZE: " << MAX_SIZE << std::endl;
    std::cout << "PI: " << PI << std::endl;

    // MAX_SIZE = 200;  // 에러! const는 수정 불가

    return 0;
}
```

### constexpr (컴파일 타임 상수)

```cpp
#include <iostream>

constexpr int square(int x) {
    return x * x;
}

int main() {
    constexpr int SIZE = 10;
    constexpr int AREA = square(5);  // 컴파일 시 계산

    int arr[SIZE];  // 배열 크기로 사용 가능

    std::cout << "SIZE: " << SIZE << std::endl;
    std::cout << "AREA: " << AREA << std::endl;

    return 0;
}
```

### const vs constexpr

| 구분 | const | constexpr |
|------|-------|-----------|
| 초기화 시점 | 런타임 가능 | 컴파일 타임 필수 |
| 배열 크기 | 일부 컴파일러만 | 항상 가능 |
| 함수 적용 | 불가 | 가능 |

---

## 5. auto 키워드 (C++11)

컴파일러가 타입을 자동으로 추론합니다.

```cpp
#include <iostream>

int main() {
    auto i = 42;        // int
    auto d = 3.14;      // double
    auto c = 'A';       // char
    auto b = true;      // bool
    auto s = "Hello";   // const char*

    std::cout << "i type: int, value: " << i << std::endl;
    std::cout << "d type: double, value: " << d << std::endl;

    // 타입 확인 (디버깅용)
    // typeid(i).name() 으로 확인 가능

    return 0;
}
```

### auto 사용 시 주의

```cpp
auto x = 10;       // int (리터럴 기본값)
auto y = 10.0;     // double
auto z = 10.0f;    // float (f 접미사)
auto ll = 10LL;    // long long
```

---

## 6. 형변환 (Type Casting)

### 암시적 형변환 (자동)

```cpp
#include <iostream>

int main() {
    int i = 10;
    double d = i;  // int → double (안전)

    double pi = 3.14;
    int truncated = pi;  // double → int (소수점 손실!)

    std::cout << "d: " << d << std::endl;         // 10
    std::cout << "truncated: " << truncated << std::endl;  // 3

    return 0;
}
```

### 명시적 형변환

```cpp
#include <iostream>

int main() {
    double pi = 3.14159;

    // C 스타일 (비권장)
    int a = (int)pi;

    // C++ 함수 스타일
    int b = int(pi);

    // static_cast (권장)
    int c = static_cast<int>(pi);

    std::cout << "a: " << a << std::endl;  // 3
    std::cout << "b: " << b << std::endl;  // 3
    std::cout << "c: " << c << std::endl;  // 3

    return 0;
}
```

### C++ 캐스트 연산자

| 캐스트 | 용도 |
|--------|------|
| `static_cast<T>` | 일반적인 타입 변환 |
| `const_cast<T>` | const 제거/추가 |
| `dynamic_cast<T>` | 다형성 클래스 변환 |
| `reinterpret_cast<T>` | 비트 수준 재해석 |

---

## 7. 크기 확인: sizeof

```cpp
#include <iostream>

int main() {
    std::cout << "char: " << sizeof(char) << " bytes" << std::endl;
    std::cout << "short: " << sizeof(short) << " bytes" << std::endl;
    std::cout << "int: " << sizeof(int) << " bytes" << std::endl;
    std::cout << "long: " << sizeof(long) << " bytes" << std::endl;
    std::cout << "long long: " << sizeof(long long) << " bytes" << std::endl;
    std::cout << "float: " << sizeof(float) << " bytes" << std::endl;
    std::cout << "double: " << sizeof(double) << " bytes" << std::endl;
    std::cout << "bool: " << sizeof(bool) << " bytes" << std::endl;

    int arr[10];
    std::cout << "int[10]: " << sizeof(arr) << " bytes" << std::endl;

    return 0;
}
```

---

## 8. 리터럴 (Literals)

### 정수 리터럴

```cpp
int decimal = 42;       // 10진수
int octal = 052;        // 8진수 (0으로 시작)
int hex = 0x2A;         // 16진수 (0x로 시작)
int binary = 0b101010;  // 2진수 (C++14, 0b로 시작)

long l = 42L;
unsigned u = 42U;
long long ll = 42LL;
unsigned long long ull = 42ULL;
```

### 실수 리터럴

```cpp
double d1 = 3.14;
double d2 = 3.14e2;    // 314.0 (과학적 표기법)
double d3 = 3.14e-2;   // 0.0314

float f = 3.14f;       // float (f 접미사)
long double ld = 3.14L; // long double (L 접미사)
```

### 숫자 구분자 (C++14)

```cpp
int million = 1'000'000;        // 가독성 향상
long long big = 1'234'567'890LL;
double pi = 3.141'592'653;
```

---

## 9. 타입 별칭

### typedef (전통적 방식)

```cpp
typedef unsigned int uint;
typedef long long int64;

uint a = 100;
int64 b = 1234567890123LL;
```

### using (C++11, 권장)

```cpp
using uint = unsigned int;
using int64 = long long;

uint a = 100;
int64 b = 1234567890123LL;
```

---

## 10. 표준 고정 크기 타입

`<cstdint>` 헤더에 정의된 플랫폼 독립적인 타입입니다.

```cpp
#include <iostream>
#include <cstdint>

int main() {
    int8_t a = 127;          // 정확히 8비트
    int16_t b = 32767;       // 정확히 16비트
    int32_t c = 2147483647;  // 정확히 32비트
    int64_t d = 9223372036854775807LL;  // 정확히 64비트

    uint8_t ua = 255;        // unsigned 8비트
    uint16_t ub = 65535;     // unsigned 16비트

    std::cout << "int8_t max: " << (int)a << std::endl;
    std::cout << "int16_t max: " << b << std::endl;
    std::cout << "int32_t max: " << c << std::endl;
    std::cout << "int64_t max: " << d << std::endl;

    return 0;
}
```

---

## 11. 요약

| 분류 | 타입 | 크기 |
|------|------|------|
| 정수 | `int` | 4바이트 |
| 정수 | `long long` | 8바이트 |
| 실수 | `double` | 8바이트 |
| 문자 | `char` | 1바이트 |
| 불리언 | `bool` | 1바이트 |

| 키워드 | 용도 |
|--------|------|
| `const` | 런타임 상수 |
| `constexpr` | 컴파일 타임 상수 |
| `auto` | 타입 자동 추론 |
| `static_cast` | 안전한 형변환 |

---

## 12. 연습 문제

### 연습 1: 변수 출력

다양한 타입의 변수를 선언하고 값을 출력하세요.

### 연습 2: 형변환

섭씨 온도를 입력받아 화씨로 변환하세요. (F = C × 9/5 + 32)

### 연습 3: sizeof

모든 기본 타입의 크기를 출력하는 프로그램을 작성하세요.

---

## 다음 단계

[03_Operators_and_Control_Flow.md](./03_Operators_and_Control_Flow.md)에서 연산자와 제어문을 배워봅시다!
