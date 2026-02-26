# 템플릿

**이전**: [STL 알고리즘과 반복자](./11_STL_Algorithms_Iterators.md) | **다음**: [예외 처리와 파일 입출력](./13_Exceptions_and_File_IO.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일 및 다중 타입 매개변수를 사용하여 함수 템플릿(Function Template)과 클래스 템플릿(Class Template)을 정의한다
2. 암묵적 타입 추론(Type Deduction)과 명시적 템플릿 인자 지정을 구별한다
3. 비타입 템플릿 매개변수(Non-Type Template Parameter)를 사용하여 컴파일 타임 상수를 템플릿에 내장한다
4. 특정 타입에 대한 동작을 커스터마이즈하기 위해 전체 특수화(Full Specialization)와 부분 특수화(Partial Specialization)를 작성한다
5. 임의의 인자 개수를 받는 함수를 위한 가변 인자 템플릿(Variadic Template)과 폴드 표현식(Fold Expression)을 구현한다
6. SFINAE와 `if constexpr`을 적용하여 조건부로 템플릿 인스턴스화(Instantiation)를 활성화한다
7. C++20 개념(Concept)을 정의하고 사용하여 명확한 오류 메시지와 함께 템플릿 매개변수를 제약한다
8. 헤더 전용(Header-Only) 컴파일 모델을 사용하여 템플릿 코드를 올바르게 구성한다

---

> **비유 — 컴파일러의 쿠키 커터**: C++ 템플릿은 쿠키 커터(틀)와 같습니다. 모양을 한 번 정의해 두면, 컴파일러가 사용하는 모든 타입(`int`, `double`, `std::string`)에 대해 구체적인 버전을 찍어냅니다. 틀 자체는 쿠키가 아닙니다 — 컴파일러가 컴파일 타임에 실제로 타입 안전한 코드를 생성하기 위해 사용하는 청사진으로, 런타임 오버헤드가 전혀 없습니다.

템플릿은 C++에서 제네릭 프로그래밍(Generic Programming)의 토대입니다. 템플릿 없이는 지원하고 싶은 모든 데이터 타입에 대해 모든 함수와 클래스를 중복으로 작성해야 합니다 — 유지보수의 악몽이죠. 템플릿을 사용하면 컴파일러가 그 중복 작업을 대신 해주어 타입 안전한 특수화된 코드를 생성하고, 개발자는 로직을 한 번만 작성하면 됩니다. 템플릿은 전체 STL의 기반이며 현대적인 메타프로그래밍(Metaprogramming) 기법의 토대를 형성합니다.

## 1. 템플릿이란?

템플릿은 타입에 독립적인 일반화된 코드를 작성하는 C++의 강력한 기능입니다.

```
┌─────────────────────────────────────────────┐
│           템플릿 (Template)                   │
├─────────────────────────────────────────────┤
│  • 타입을 매개변수로 받는 코드                   │
│  • 컴파일 타임에 실제 타입으로 대체               │
│  • 코드 재사용성 극대화                         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┬─────────────────┐
│  함수 템플릿      │  클래스 템플릿    │
│  (Function)      │  (Class)        │
└─────────────────┴─────────────────┘
```

### 왜 템플릿이 필요한가?

```cpp
// 템플릿 없이 오버로딩으로 구현
int max(int a, int b) { return (a > b) ? a : b; }
double max(double a, double b) { return (a > b) ? a : b; }
char max(char a, char b) { return (a > b) ? a : b; }
// ... 모든 타입마다 반복 필요

// 템플릿으로 한 번에 해결
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

// typename 대신 class도 사용 가능 (의미 동일)
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

### 여러 타입 매개변수

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

// 정수값을 템플릿 매개변수로
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

    // 컴파일 타임에 계산됨
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

### 멤버 함수 외부 정의

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

// 멤버 함수 외부 정의 시 template 선언 필요
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

### 여러 타입 매개변수

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

### 기본 템플릿 인자

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

### 전체 특수화 (Full Specialization)

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
        std::cout << "일반: " << data << std::endl;
    }
};

// char* 타입에 대한 전체 특수화
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

// bool 타입에 대한 전체 특수화
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
    h1.display();  // 일반: 42

    DataHolder<char*> h2("Hello");
    h2.display();  // char*: Hello

    DataHolder<bool> h3(true);
    h3.display();  // bool: true

    return 0;
}
```

### 부분 특수화 (Partial Specialization)

```cpp
#include <iostream>

// 기본 템플릿
template<typename T, typename U>
class Pair {
public:
    void info() const {
        std::cout << "일반 Pair<T, U>" << std::endl;
    }
};

// 두 타입이 같을 때 부분 특수화
template<typename T>
class Pair<T, T> {
public:
    void info() const {
        std::cout << "같은 타입 Pair<T, T>" << std::endl;
    }
};

// 두 번째가 int일 때 부분 특수화
template<typename T>
class Pair<T, int> {
public:
    void info() const {
        std::cout << "Pair<T, int>" << std::endl;
    }
};

// 포인터 타입 부분 특수화
template<typename T, typename U>
class Pair<T*, U*> {
public:
    void info() const {
        std::cout << "포인터 Pair<T*, U*>" << std::endl;
    }
};

int main() {
    Pair<double, char> p1;
    p1.info();  // 일반 Pair<T, U>

    Pair<double, double> p2;
    p2.info();  // 같은 타입 Pair<T, T>

    Pair<double, int> p3;
    p3.info();  // Pair<T, int>

    Pair<int*, double*> p4;
    p4.info();  // 포인터 Pair<T*, U*>

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
    std::cout << isEqual("Hello", "Hello") << std::endl; // true (포인터 주소 비교)

    const char* s1 = "Hello";
    const char* s2 = "Hello";
    std::cout << isEqual(s1, s2) << std::endl;           // true (문자열 내용 비교)

    return 0;
}
```

---

## 5. 가변 인자 템플릿 (Variadic Templates)

### 기본 문법

```cpp
#include <iostream>

// 재귀 종료 조건 (base case)
void print() {
    std::cout << std::endl;
}

// 가변 인자 템플릿
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

### 합계 계산

```cpp
#include <iostream>

// 재귀 종료
template<typename T>
T sum(T value) {
    return value;
}

// 가변 인자
template<typename T, typename... Args>
T sum(T first, Args... args) {
    return first + sum(args...);
}

int main() {
    std::cout << sum(1, 2, 3, 4, 5) << std::endl;     // 15
    std::cout << sum(1.5, 2.5, 3.0) << std::endl;     // 7
    std::cout << sum(10) << std::endl;                 // 10

    return 0;
}
```

### sizeof... 연산자

```cpp
#include <iostream>

template<typename... Args>
void countArgs(Args... args) {
    std::cout << "인자 개수: " << sizeof...(Args) << std::endl;
    std::cout << "인자 개수: " << sizeof...(args) << std::endl;  // 같은 결과
}

int main() {
    countArgs();                 // 인자 개수: 0
    countArgs(1);                // 인자 개수: 1
    countArgs(1, 2, 3);          // 인자 개수: 3
    countArgs("a", 1, 3.14, 'c'); // 인자 개수: 4

    return 0;
}
```

### 폴드 표현식 (C++17)

```cpp
#include <iostream>

// C++17 폴드 표현식으로 간단하게
template<typename... Args>
auto sumFold(Args... args) {
    return (args + ...);  // 우측 폴드
}

template<typename... Args>
void printFold(Args... args) {
    ((std::cout << args << " "), ...);  // 콤마 연산자 폴드
    std::cout << std::endl;
}

template<typename... Args>
bool allTrue(Args... args) {
    return (args && ...);  // 모두 true인지
}

template<typename... Args>
bool anyTrue(Args... args) {
    return (args || ...);  // 하나라도 true인지
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

## 6. SFINAE

SFINAE (Substitution Failure Is Not An Error): 템플릿 인자 대체 실패는 에러가 아닙니다.

### 기본 개념

```cpp
#include <iostream>
#include <type_traits>

// 정수 타입일 때만 활성화
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process(T value) {
    std::cout << "정수: " << value << std::endl;
}

// 부동소수점 타입일 때만 활성화
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process(T value) {
    std::cout << "실수: " << value << std::endl;
}

int main() {
    process(42);      // 정수: 42
    process(3.14);    // 실수: 3.14
    // process("Hi"); // 컴파일 에러 (둘 다 매칭 안 됨)

    return 0;
}
```

### C++17 if constexpr

```cpp
#include <iostream>
#include <type_traits>

template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "정수: " << value * 2 << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "실수: " << value / 2 << std::endl;
    } else {
        std::cout << "기타: " << value << std::endl;
    }
}

int main() {
    process(10);        // 정수: 20
    process(5.0);       // 실수: 2.5
    process("Hello");   // 기타: Hello

    return 0;
}
```

---

## 7. 타입 특성 (Type Traits)

### 기본 타입 특성

```cpp
#include <iostream>
#include <type_traits>

int main() {
    std::cout << std::boolalpha;

    // 타입 검사
    std::cout << "is_integral<int>: "
              << std::is_integral<int>::value << std::endl;  // true
    std::cout << "is_integral<double>: "
              << std::is_integral<double>::value << std::endl;  // false

    std::cout << "is_floating_point<double>: "
              << std::is_floating_point<double>::value << std::endl;  // true

    std::cout << "is_pointer<int*>: "
              << std::is_pointer<int*>::value << std::endl;  // true

    std::cout << "is_class<std::string>: "
              << std::is_class<std::string>::value << std::endl;  // true

    // 타입 변환
    std::cout << "is_same<int, int>: "
              << std::is_same<int, int>::value << std::endl;  // true

    using NoRef = std::remove_reference<int&>::type;
    std::cout << "is_same<NoRef, int>: "
              << std::is_same<NoRef, int>::value << std::endl;  // true

    return 0;
}
```

### 조건부 타입 선택

```cpp
#include <iostream>
#include <type_traits>

template<bool Condition, typename T, typename F>
struct MyConditional {
    using type = T;
};

template<typename T, typename F>
struct MyConditional<false, T, F> {
    using type = F;
};

int main() {
    // std::conditional 사용
    using Type1 = std::conditional<true, int, double>::type;
    using Type2 = std::conditional<false, int, double>::type;

    std::cout << std::boolalpha;
    std::cout << "Type1 is int: "
              << std::is_same<Type1, int>::value << std::endl;     // true
    std::cout << "Type2 is double: "
              << std::is_same<Type2, double>::value << std::endl;  // true

    // 크기에 따른 타입 선택
    using SmallType = std::conditional<(sizeof(int) > 4), long, int>::type;
    std::cout << "SmallType size: " << sizeof(SmallType) << std::endl;

    return 0;
}
```

---

## 8. Concepts (C++20)

### 기본 문법

```cpp
#include <iostream>
#include <concepts>

// concept 정의
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

// concept 사용
template<Numeric T>
T square(T x) {
    return x * x;
}

// requires 절 사용
template<typename T>
requires Addable<T>
T add(T a, T b) {
    return a + b;
}

// 축약형
auto multiply(Numeric auto a, Numeric auto b) {
    return a * b;
}

int main() {
    std::cout << square(5) << std::endl;      // 25
    std::cout << square(3.5) << std::endl;    // 12.25
    // square("Hi");  // 에러: Numeric 제약 불만족

    std::cout << add(10, 20) << std::endl;    // 30
    std::cout << multiply(3, 4) << std::endl; // 12

    return 0;
}
```

### 표준 Concepts

```cpp
#include <iostream>
#include <concepts>
#include <string>

// 표준 concept 사용
template<std::integral T>
void processInt(T value) {
    std::cout << "정수: " << value << std::endl;
}

template<std::floating_point T>
void processFloat(T value) {
    std::cout << "실수: " << value << std::endl;
}

template<std::convertible_to<std::string> T>
void processString(T value) {
    std::string s = value;
    std::cout << "문자열: " << s << std::endl;
}

int main() {
    processInt(42);
    processFloat(3.14);
    processString("Hello");

    return 0;
}
```

---

## 9. 실용적인 템플릿 예제

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
            throw std::runtime_error("스택이 비어있습니다");
        }
        T value = data.back();
        data.pop_back();
        return value;
    }

    T& top() {
        if (empty()) {
            throw std::runtime_error("스택이 비어있습니다");
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

### 팩토리 함수

```cpp
#include <iostream>
#include <memory>
#include <string>

// make 함수 템플릿
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(n), age(a) {
        std::cout << "Person 생성: " << name << std::endl;
    }

    void introduce() const {
        std::cout << name << ", " << age << "세" << std::endl;
    }
};

int main() {
    auto p = make<Person>("Alice", 25);
    p->introduce();  // Alice, 25세

    auto nums = make<std::vector<int>>(std::initializer_list<int>{1, 2, 3});
    for (int n : *nums) {
        std::cout << n << " ";  // 1 2 3
    }
    std::cout << std::endl;

    return 0;
}
```

### 타입 안전 printf

```cpp
#include <iostream>
#include <sstream>
#include <string>

// 재귀 종료
void safePrint(std::ostream& os, const char* format) {
    while (*format) {
        if (*format == '%' && *(format + 1) != '%') {
            throw std::runtime_error("인자 부족");
        }
        if (*format == '%' && *(format + 1) == '%') {
            format++;  // %% 건너뛰기
        }
        os << *format++;
    }
}

// 가변 인자 처리
template<typename T, typename... Args>
void safePrint(std::ostream& os, const char* format, T value, Args... args) {
    while (*format) {
        if (*format == '%') {
            if (*(format + 1) == '%') {
                os << '%';
                format += 2;
                continue;
            }
            os << value;
            safePrint(os, format + 1, args...);
            return;
        }
        os << *format++;
    }
    throw std::runtime_error("인자가 너무 많음");
}

template<typename... Args>
std::string format(const char* fmt, Args... args) {
    std::ostringstream oss;
    safePrint(oss, fmt, args...);
    return oss.str();
}

int main() {
    std::cout << format("이름: %, 나이: %세", "Alice", 25) << std::endl;
    // 이름: Alice, 나이: 25세

    std::cout << format("% + % = %", 10, 20, 30) << std::endl;
    // 10 + 20 = 30

    return 0;
}
```

---

## 10. 템플릿 컴파일 모델

### 헤더에 정의해야 하는 이유

```
일반 함수:                    템플릿:
┌─────────────┐              ┌─────────────┐
│ header.h    │              │ header.h    │
│ 선언만      │              │ 선언 + 정의  │
└─────────────┘              └─────────────┘
       │                            │
       ▼                            ▼
┌─────────────┐              ┌─────────────┐
│ source.cpp  │              │ (사용처에서  │
│ 정의        │              │  인스턴스화) │
└─────────────┘              └─────────────┘
```

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

// 템플릿 정의도 헤더에 포함
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

### 명시적 인스턴스화 (선택적)

```cpp
// mytemplate.cpp
#include "mytemplate.h"

// 특정 타입에 대해 명시적 인스턴스화
template class MyContainer<int>;
template class MyContainer<double>;
template class MyContainer<std::string>;
```

---

## 11. 요약

| 개념 | 설명 |
|------|------|
| 함수 템플릿 | 타입에 독립적인 함수 |
| 클래스 템플릿 | 타입에 독립적인 클래스 |
| 템플릿 특수화 | 특정 타입에 대한 특별 구현 |
| 부분 특수화 | 일부 조건에 대한 특수화 |
| 가변 인자 템플릿 | 임의 개수의 인자 처리 |
| SFINAE | 대체 실패는 에러 아님 |
| Concepts (C++20) | 템플릿 제약 조건 |
| 비타입 매개변수 | 값을 템플릿 인자로 |

---

## 12. 연습 문제

### 연습 1: 최소/최대값 함수

임의 개수의 인자에서 최소값과 최대값을 반환하는 함수 템플릿을 작성하세요.

### 연습 2: 제네릭 Queue

Stack 예제를 참고하여 Queue 클래스 템플릿을 작성하세요.

### 연습 3: 타입별 직렬화

다양한 타입을 문자열로 변환하는 `serialize` 함수 템플릿을 작성하세요. (기본 타입, 컨테이너 등)

---

## 다음 단계

[예외 처리와 파일 입출력](./13_Exceptions_and_File_IO.md)에서 예외 처리와 파일 I/O를 배워봅시다!

---

**이전**: [STL 알고리즘과 반복자](./11_STL_Algorithms_Iterators.md) | **다음**: [예외 처리와 파일 입출력](./13_Exceptions_and_File_IO.md)
