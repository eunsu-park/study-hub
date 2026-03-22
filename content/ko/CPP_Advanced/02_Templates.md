# 템플릿

**이전**: [이동 의미론 심화](./01_Move_Semantics_Deep_Dive.md) | **다음**: [템플릿 메타프로그래밍](./03_Template_Metaprogramming.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일 및 다중 타입 매개변수로 함수 템플릿과 클래스 템플릿 정의하기
2. 암묵적 타입 추론과 명시적 템플릿 인수 지정을 구분하기
3. 비타입 템플릿 매개변수를 사용하여 컴파일 시간 상수를 템플릿에 포함하기
4. 완전 특수화와 부분 특수화를 작성하여 특정 타입에 대한 동작 커스터마이즈하기
5. 가변 인수 템플릿(variadic templates)과 폴드 표현식(fold expressions)을 구현하여 임의 개수의 인수를 받는 함수 작성하기
6. SFINAE와 `if constexpr`를 적용하여 템플릿 인스턴스화를 조건부로 활성화하기
7. 헤더 전용 컴파일 모델을 사용하여 템플릿 코드를 올바르게 구성하기

---

> **비유 -- 컴파일러의 쿠키 커터**: C++ 템플릿은 쿠키 커터와 같습니다. 모양을 한 번 정의하면 컴파일러가 사용하는 모든 타입(`int`, `double`, `std::string`)에 대해 구체적인 버전을 찍어냅니다. 커터 자체는 쿠키가 아니라, 컴파일러가 컴파일 시간에 실제 타입 안전한 코드를 생성하기 위해 사용하는 청사진이며, 런타임 오버헤드는 제로입니다.

템플릿은 C++의 제네릭 프로그래밍의 기초입니다. 템플릿 없이는 지원하려는 각 데이터 타입에 대해 모든 함수와 클래스를 복제해야 합니다 -- 유지보수의 악몽이죠. 템플릿을 사용하면 컴파일러가 복제를 대신하여 특수화되고 타입 안전한 코드를 생성하면서 로직은 한 번만 작성하면 됩니다. 템플릿은 전체 STL의 기반이며 현대 메타프로그래밍 기법의 토대를 이룹니다.

## 1. 템플릿이란?

템플릿은 타입에 독립적인 제네릭 코드를 작성할 수 있는 C++의 강력한 기능입니다.

> **템플릿(Template)**
>
> - 타입을 매개변수로 받는 코드
> - 컴파일 시간에 실제 타입으로 대체됨
> - 코드 재사용성을 극대화
>
> **함수 템플릿(Function Template)** | **클래스 템플릿(Class Template)**

### 왜 템플릿인가?

```cpp
// 템플릿 없이 오버로딩 사용
int max(int a, int b) { return (a > b) ? a : b; }
double max(double a, double b) { return (a > b) ? a : b; }
char max(char a, char b) { return (a > b) ? a : b; }
// ... 모든 타입에 대해 반복 필요

// 단일 템플릿으로 해결
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
```

---

## 2. 함수 템플릿

### 기본 문법

```cpp
#include <iostream>

// 함수 템플릿 정의
template<typename T>
T add(T a, T b) {
    return a + b;
}

// typename 대신 class도 사용 가능 (같은 의미)
template<class T>
T multiply(T a, T b) {
    return a * b;
}

int main() {
    // 명시적 타입 지정
    std::cout << add<int>(3, 5) << std::endl;        // 8
    std::cout << add<double>(3.5, 2.5) << std::endl; // 6

    // 타입 추론 (컴파일러가 자동으로 타입 결정)
    std::cout << add(10, 20) << std::endl;           // 30 (int)
    std::cout << add(1.5, 2.5) << std::endl;         // 4 (double)

    std::cout << multiply(4, 5) << std::endl;        // 20

    return 0;
}
```

### 다중 타입 매개변수

```cpp
#include <iostream>
#include <string>

template<typename T, typename U>
void printPair(T first, U second) {
    std::cout << first << ", " << second << std::endl;
}

// 반환 타입도 템플릿으로
template<typename T, typename U>
auto addDifferent(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14: 간단한 auto 반환
template<typename T, typename U>
auto addSimple(T a, U b) {
    return a + b;
}

int main() {
    printPair(1, "Hello");         // 1, Hello
    printPair(3.14, 100);          // 3.14, 100
    printPair("Name", std::string("Alice"));  // Name, Alice

    std::cout << addDifferent(10, 3.5) << std::endl;  // 13.5 (double)
    std::cout << addSimple(5, 2.5) << std::endl;      // 7.5

    return 0;
}
```

### 비타입 템플릿 매개변수

```cpp
#include <iostream>
#include <array>

// 정수 값을 템플릿 매개변수로
template<typename T, int Size>
class FixedArray {
private:
    T data[Size];
public:
    T& operator[](int index) { return data[index]; }
    const T& operator[](int index) const { return data[index]; }
    int size() const { return Size; }
};

// 함수에서도 사용 가능
template<int N>
int factorial() {
    return N * factorial<N - 1>();
}

template<>
int factorial<0>() {
    return 1;
}

int main() {
    FixedArray<int, 5> arr;
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << " ";  // 0 10 20 30 40
    }
    std::cout << std::endl;

    // 컴파일 시간에 계산
    std::cout << "5! = " << factorial<5>() << std::endl;  // 120

    return 0;
}
```

---

## 3. 클래스 템플릿

### 기본 문법

```cpp
#include <iostream>

template<typename T>
class Box {
private:
    T value;

public:
    Box(T v) : value(v) {}

    T getValue() const { return value; }
    void setValue(T v) { value = v; }

    void display() const {
        std::cout << "Box: " << value << std::endl;
    }
};

int main() {
    Box<int> intBox(42);
    intBox.display();  // Box: 42

    Box<double> doubleBox(3.14);
    doubleBox.display();  // Box: 3.14

    Box<std::string> stringBox("Hello");
    stringBox.display();  // Box: Hello

    return 0;
}
```

### 클래스 외부 멤버 함수 정의

```cpp
#include <iostream>

template<typename T>
class Calculator {
private:
    T value;

public:
    Calculator(T v);
    T add(T x);
    T subtract(T x);
    void display() const;
};

// 외부 정의 시 템플릿 선언 필요
template<typename T>
Calculator<T>::Calculator(T v) : value(v) {}

template<typename T>
T Calculator<T>::add(T x) {
    return value + x;
}

template<typename T>
T Calculator<T>::subtract(T x) {
    return value - x;
}

template<typename T>
void Calculator<T>::display() const {
    std::cout << "Value: " << value << std::endl;
}

int main() {
    Calculator<int> calc(10);
    std::cout << calc.add(5) << std::endl;      // 15
    std::cout << calc.subtract(3) << std::endl;  // 7
    calc.display();  // Value: 10

    return 0;
}
```

### 다중 타입 매개변수

```cpp
#include <iostream>
#include <string>

template<typename K, typename V>
class Pair {
private:
    K key;
    V value;

public:
    Pair(K k, V v) : key(k), value(v) {}

    K getKey() const { return key; }
    V getValue() const { return value; }

    void display() const {
        std::cout << key << ": " << value << std::endl;
    }
};

int main() {
    Pair<std::string, int> age("Alice", 25);
    age.display();  // Alice: 25

    Pair<int, std::string> student(1001, "Bob");
    student.display();  // 1001: Bob

    Pair<std::string, double> price("Apple", 1.99);
    price.display();  // Apple: 1.99

    return 0;
}
```

### 기본 템플릿 인수

```cpp
#include <iostream>
#include <vector>

template<typename T = int, int Size = 10>
class Array {
private:
    T data[Size];
    int count = 0;

public:
    void add(T value) {
        if (count < Size) {
            data[count++] = value;
        }
    }

    void display() const {
        for (int i = 0; i < count; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

    int capacity() const { return Size; }
};

int main() {
    Array<> arr1;  // int, 10 (기본값)
    arr1.add(1);
    arr1.add(2);
    arr1.display();  // 1 2

    Array<double> arr2;  // double, 10
    arr2.add(1.5);
    arr2.add(2.5);
    arr2.display();  // 1.5 2.5

    Array<std::string, 5> arr3;  // string, 5
    arr3.add("Hello");
    arr3.add("World");
    arr3.display();  // Hello World

    return 0;
}
```

---

## 4. 템플릿 특수화

### 완전 특수화

```cpp
#include <iostream>
#include <cstring>

// 기본 템플릿
template<typename T>
class DataHolder {
private:
    T data;
public:
    DataHolder(T d) : data(d) {}
    void display() const {
        std::cout << "General: " << data << std::endl;
    }
};

// char*에 대한 완전 특수화
template<>
class DataHolder<char*> {
private:
    char* data;
public:
    DataHolder(const char* d) {
        data = new char[strlen(d) + 1];
        strcpy(data, d);
    }
    ~DataHolder() { delete[] data; }
    void display() const {
        std::cout << "char*: " << data << std::endl;
    }
};

// bool에 대한 완전 특수화
template<>
class DataHolder<bool> {
private:
    bool data;
public:
    DataHolder(bool d) : data(d) {}
    void display() const {
        std::cout << "bool: " << (data ? "true" : "false") << std::endl;
    }
};

int main() {
    DataHolder<int> h1(42);
    h1.display();  // General: 42

    DataHolder<char*> h2("Hello");
    h2.display();  // char*: Hello

    DataHolder<bool> h3(true);
    h3.display();  // bool: true

    return 0;
}
```

### 부분 특수화

```cpp
#include <iostream>

// 기본 템플릿
template<typename T, typename U>
class Pair {
public:
    void info() const {
        std::cout << "General Pair<T, U>" << std::endl;
    }
};

// 두 타입이 같을 때의 부분 특수화
template<typename T>
class Pair<T, T> {
public:
    void info() const {
        std::cout << "Same type Pair<T, T>" << std::endl;
    }
};

// 두 번째가 int일 때의 부분 특수화
template<typename T>
class Pair<T, int> {
public:
    void info() const {
        std::cout << "Pair<T, int>" << std::endl;
    }
};

// 포인터 타입에 대한 부분 특수화
template<typename T, typename U>
class Pair<T*, U*> {
public:
    void info() const {
        std::cout << "Pointer Pair<T*, U*>" << std::endl;
    }
};

int main() {
    Pair<double, char> p1;
    p1.info();  // General Pair<T, U>

    Pair<double, double> p2;
    p2.info();  // Same type Pair<T, T>

    Pair<double, int> p3;
    p3.info();  // Pair<T, int>

    Pair<int*, double*> p4;
    p4.info();  // Pointer Pair<T*, U*>

    return 0;
}
```

### 함수 템플릿 특수화

```cpp
#include <iostream>
#include <cstring>

// 기본 템플릿
template<typename T>
bool isEqual(T a, T b) {
    return a == b;
}

// char* 특수화
template<>
bool isEqual<const char*>(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

int main() {
    std::cout << std::boolalpha;

    std::cout << isEqual(10, 10) << std::endl;           // true
    std::cout << isEqual(3.14, 3.14) << std::endl;       // true

    const char* s1 = "Hello";
    const char* s2 = "Hello";
    std::cout << isEqual(s1, s2) << std::endl;           // true (내용 비교)

    return 0;
}
```

---

## 5. 가변 인수 템플릿

### 기본 문법

```cpp
#include <iostream>

// 재귀 종료 (기저 사례)
void print() {
    std::cout << std::endl;
}

// 가변 인수 템플릿
template<typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first;
    if (sizeof...(args) > 0) {
        std::cout << ", ";
    }
    print(args...);  // 재귀 호출
}

int main() {
    print(1, 2, 3);                    // 1, 2, 3
    print("Hello", 3.14, 42, 'A');     // Hello, 3.14, 42, A
    print("Name:", "Alice", "Age:", 25);  // Name:, Alice, Age:, 25

    return 0;
}
```

### 폴드 표현식 (C++17)

```cpp
#include <iostream>

// C++17 폴드 표현식 (간소화)
template<typename... Args>
auto sumFold(Args... args) {
    return (args + ...);  // 우측 폴드
}

template<typename... Args>
void printFold(Args... args) {
    ((std::cout << args << " "), ...);  // 쉼표 연산자 폴드
    std::cout << std::endl;
}

template<typename... Args>
bool allTrue(Args... args) {
    return (args && ...);  // 모두 참인가?
}

template<typename... Args>
bool anyTrue(Args... args) {
    return (args || ...);  // 하나라도 참인가?
}

int main() {
    std::cout << sumFold(1, 2, 3, 4, 5) << std::endl;  // 15

    printFold(1, "Hello", 3.14);  // 1 Hello 3.14

    std::cout << std::boolalpha;
    std::cout << allTrue(true, true, true) << std::endl;   // true
    std::cout << allTrue(true, false, true) << std::endl;  // false
    std::cout << anyTrue(false, false, true) << std::endl; // true

    return 0;
}
```

---

## 6. SFINAE와 if constexpr

SFINAE(Substitution Failure Is Not An Error): 템플릿 인수 치환 실패는 에러가 아닙니다 -- 컴파일러가 해당 오버로드를 단순히 버립니다.

### enable_if를 사용한 기본 SFINAE

```cpp
#include <iostream>
#include <type_traits>

// 정수 타입에 대해서만 활성화
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process(T value) {
    std::cout << "Integer: " << value << std::endl;
}

// 부동소수점 타입에 대해서만 활성화
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process(T value) {
    std::cout << "Float: " << value << std::endl;
}

int main() {
    process(42);      // Integer: 42
    process(3.14);    // Float: 3.14
    // process("Hi"); // 컴파일 에러 (어느 것도 매칭되지 않음)

    return 0;
}
```

### C++17 if constexpr (선호)

```cpp
#include <iostream>
#include <type_traits>

template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value * 2 << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << value / 2 << std::endl;
    } else {
        std::cout << "Other: " << value << std::endl;
    }
}

int main() {
    process(10);        // Integer: 20
    process(5.0);       // Float: 2.5
    process("Hello");   // Other: Hello

    return 0;
}
```

---

## 7. 타입 특성(Type Traits)

```cpp
#include <iostream>
#include <type_traits>

int main() {
    std::cout << std::boolalpha;

    // 타입 검사
    std::cout << "is_integral<int>: "
              << std::is_integral<int>::value << std::endl;  // true
    std::cout << "is_floating_point<double>: "
              << std::is_floating_point<double>::value << std::endl;  // true
    std::cout << "is_pointer<int*>: "
              << std::is_pointer<int*>::value << std::endl;  // true

    // 타입 변환
    std::cout << "is_same<int, int>: "
              << std::is_same<int, int>::value << std::endl;  // true

    using NoRef = std::remove_reference<int&>::type;
    std::cout << "is_same<NoRef, int>: "
              << std::is_same<NoRef, int>::value << std::endl;  // true

    // 조건부 타입 선택
    using Type1 = std::conditional<true, int, double>::type;
    std::cout << "Type1 is int: "
              << std::is_same<Type1, int>::value << std::endl;  // true

    return 0;
}
```

---

## 8. 템플릿 컴파일 모델

### 헤더에 정의하는 이유

| | 일반 함수 | 템플릿 |
|---|---|---|
| **header.h** | 선언 | 선언 + 정의 |
| **source.cpp** | 정의 | (사용 위치에서 인스턴스화) |

### 올바른 템플릿 구조

```cpp
// mytemplate.h
#ifndef MYTEMPLATE_H
#define MYTEMPLATE_H

template<typename T>
class MyContainer {
private:
    T* data;
    size_t size;

public:
    MyContainer(size_t n);
    ~MyContainer();
    T& operator[](size_t index);
    size_t getSize() const;
};

// 템플릿 정의도 헤더에
template<typename T>
MyContainer<T>::MyContainer(size_t n)
    : data(new T[n]), size(n) {}

template<typename T>
MyContainer<T>::~MyContainer() {
    delete[] data;
}

template<typename T>
T& MyContainer<T>::operator[](size_t index) {
    return data[index];
}

template<typename T>
size_t MyContainer<T>::getSize() const {
    return size;
}

#endif
```

### 명시적 인스턴스화 (선택 사항)

```cpp
// mytemplate.cpp
#include "mytemplate.h"

// 특정 타입에 대한 명시적 인스턴스화
template class MyContainer<int>;
template class MyContainer<double>;
template class MyContainer<std::string>;
```

---

## 9. 실용적 템플릿 예제

### 제네릭 스택

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

template<typename T>
class Stack {
private:
    std::vector<T> data;

public:
    void push(const T& value) {
        data.push_back(value);
    }

    T pop() {
        if (empty()) {
            throw std::runtime_error("Stack is empty");
        }
        T value = data.back();
        data.pop_back();
        return value;
    }

    T& top() {
        if (empty()) {
            throw std::runtime_error("Stack is empty");
        }
        return data.back();
    }

    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

int main() {
    Stack<int> intStack;
    intStack.push(1);
    intStack.push(2);
    intStack.push(3);

    while (!intStack.empty()) {
        std::cout << intStack.pop() << " ";  // 3 2 1
    }
    std::cout << std::endl;

    Stack<std::string> strStack;
    strStack.push("Hello");
    strStack.push("World");
    std::cout << strStack.top() << std::endl;  // World

    return 0;
}
```

### 완벽한 전달을 사용하는 팩토리 함수

```cpp
#include <iostream>
#include <memory>
#include <string>

template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(std::move(n)), age(a) {}

    void introduce() const {
        std::cout << name << ", " << age << " years old" << std::endl;
    }
};

int main() {
    auto p = make<Person>("Alice", 25);
    p->introduce();  // Alice, 25 years old

    return 0;
}
```

---

## 10. 요약

| 개념 | 설명 |
|------|------|
| 함수 템플릿 | 타입에 독립적인 함수 |
| 클래스 템플릿 | 타입에 독립적인 클래스 |
| 템플릿 특수화 | 특정 타입에 대한 특별한 구현 |
| 부분 특수화 | 부분적 조건에 대한 특수화 |
| 가변 인수 템플릿 | 임의 개수의 인수 처리 |
| SFINAE | 치환 실패는 에러가 아님 |
| if constexpr | 컴파일 시간 분기 (C++17) |
| 비타입 매개변수 | 값을 템플릿 인수로 |

---

## 연습문제

### 연습문제 1: 가변 인수 Min/Max 함수

폴드 표현식을 사용하여 임의 개수의 인수에서 최솟값과 최댓값을 반환하는 함수 템플릿을 작성하세요.

### 연습문제 2: 제네릭 큐

Stack 예제를 참고하여 `enqueue`, `dequeue`, `front`, `empty`, `size` 연산을 갖춘 Queue 클래스 템플릿을 작성하세요.

### 연습문제 3: 타입별 직렬화

`if constexpr`를 사용하여 다양한 타입(기본 타입, 컨테이너 등)을 문자열로 변환하는 `serialize` 함수 템플릿을 작성하세요.

### 연습문제 4: 컴파일 시간 행렬

비타입 템플릿 매개변수를 사용하여 `Matrix<T, Rows, Cols>` 클래스 템플릿을 구현하세요. 컴파일 시간 차원 검사를 통한 덧셈과 곱셈을 지원하세요.

### 연습문제 5: Tuple 구현

가변 인수 템플릿과 재귀 상속을 사용하여 간소화된 `Tuple` 클래스 템플릿을 구현하세요.

---

## 다음 단계

SFINAE, 타입 특성, 컴파일 시간 계산을 포함한 템플릿 메타프로그래밍 기법을 더 깊이 살펴봅시다: [03_Template_Metaprogramming.md](./03_Template_Metaprogramming.md).
