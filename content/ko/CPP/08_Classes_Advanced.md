# 클래스 심화

**이전**: [클래스 기초](./07_Classes_Basics.md) | **다음**: [상속과 다형성](./09_Inheritance_and_Polymorphism.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 산술, 비교, 복합 대입, 스트림 연산자에 대한 연산자 오버로딩(operator overloading)을 구현할 수 있습니다
2. 멤버 함수 방식과 friend 함수 방식의 연산자 오버로딩을 구별할 수 있습니다
3. 얕은 복사(shallow copy)와 깊은 복사(deep copy)의 차이를 설명하고 깊은 복사 생성자를 구현할 수 있습니다
4. 자기 대입(self-assignment) 보호를 포함한 올바른 복사 대입 연산자(copy assignment operator)를 설계할 수 있습니다
5. 이동 시맨틱(move semantics)(이동 생성자와 이동 대입 연산자)을 적용하여 불필요한 복사를 제거할 수 있습니다
6. 5의 규칙(Rule of Five)을 파악하고 클래스가 다섯 가지 특수 멤버 함수를 모두 정의해야 하는 시점을 설명할 수 있습니다
7. 클래스 전체가 공유하는 상태를 위해 `static` 멤버 변수와 함수를 구현할 수 있습니다
8. 의도치 않은 암시적 변환을 방지하기 위해 `explicit`을 적용할 수 있습니다

---

클래스를 만들 수 있게 되면, 다음 질문은 복사, 이동, 비교, 출력 시 클래스가 어떻게 *동작*해야 하는가입니다. 연산자 오버로딩, 복사/이동 시맨틱(copy/move semantics), 5의 규칙(Rule of Five)과 같은 고급 클래스 메커니즘은 사용자 정의 타입(user-defined type)을 내장 타입처럼 자연스럽게 느껴지게 만드는 요소입니다. 이 기법들은 학문적인 호기심의 대상이 아닙니다. `std::string`부터 `std::vector`까지 표준 라이브러리(Standard Library) 전체의 근간을 이루며, 이를 마스터하는 것은 전문적인 수준의 C++ 코드를 작성하는 데 필수적입니다.

## 1. 연산자 오버로딩

클래스에 대해 연산자의 동작을 정의할 수 있습니다.

### 기본 구문

```cpp
반환타입 operator연산자(매개변수) {
    // 구현
}
```

### 산술 연산자 오버로딩

```cpp
#include <iostream>

class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // + 연산자 (멤버 함수)
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    // - 연산자
    Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }

    // * 연산자 (스칼라 곱)
    Vector2D operator*(double scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }

    void print() const {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }
};

int main() {
    Vector2D v1(3, 4);
    Vector2D v2(1, 2);

    Vector2D v3 = v1 + v2;  // operator+ 호출
    v3.print();  // (4, 6)

    Vector2D v4 = v1 - v2;
    v4.print();  // (2, 2)

    Vector2D v5 = v1 * 2;
    v5.print();  // (6, 8)

    return 0;
}
```

### 비교 연산자 오버로딩

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(n), age(a) {}

    // == 연산자
    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }

    // != 연산자
    bool operator!=(const Person& other) const {
        return !(*this == other);
    }

    // < 연산자 (나이 기준)
    bool operator<(const Person& other) const {
        return age < other.age;
    }
};

int main() {
    Person p1("Alice", 25);
    Person p2("Alice", 25);
    Person p3("Bob", 30);

    std::cout << std::boolalpha;
    std::cout << (p1 == p2) << std::endl;  // true
    std::cout << (p1 != p3) << std::endl;  // true
    std::cout << (p1 < p3) << std::endl;   // true

    return 0;
}
```

### 복합 대입 연산자

```cpp
class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // += 연산자
    Vector2D& operator+=(const Vector2D& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    // -= 연산자
    Vector2D& operator-=(const Vector2D& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
};
```

### 증감 연산자

```cpp
#include <iostream>

class Counter {
private:
    int value;

public:
    Counter(int v = 0) : value(v) {}

    // 전위 증가 (++c)
    Counter& operator++() {
        ++value;
        return *this;
    }

    // 후위 증가 (c++)
    Counter operator++(int) {  // int는 구분용 더미
        Counter temp = *this;
        ++value;
        return temp;
    }

    int getValue() const { return value; }
};

int main() {
    Counter c(5);

    std::cout << (++c).getValue() << std::endl;  // 6
    std::cout << (c++).getValue() << std::endl;  // 6
    std::cout << c.getValue() << std::endl;      // 7

    return 0;
}
```

### 입출력 연산자 (friend)

```cpp
#include <iostream>

class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // << 연산자 (friend 함수)
    friend std::ostream& operator<<(std::ostream& os, const Vector2D& v) {
        os << "(" << v.x << ", " << v.y << ")";
        return os;
    }

    // >> 연산자
    friend std::istream& operator>>(std::istream& is, Vector2D& v) {
        is >> v.x >> v.y;
        return is;
    }
};

int main() {
    Vector2D v(3, 4);
    std::cout << "벡터: " << v << std::endl;  // 벡터: (3, 4)

    Vector2D v2;
    std::cout << "x y 입력: ";
    std::cin >> v2;
    std::cout << "입력: " << v2 << std::endl;

    return 0;
}
```

### 함수 호출 연산자 ()

```cpp
#include <iostream>

class Adder {
private:
    int base;

public:
    Adder(int b) : base(b) {}

    // () 연산자 - 함수처럼 호출 가능
    int operator()(int x) const {
        return base + x;
    }

    int operator()(int x, int y) const {
        return base + x + y;
    }
};

int main() {
    Adder add10(10);

    std::cout << add10(5) << std::endl;     // 15
    std::cout << add10(5, 3) << std::endl;  // 18

    return 0;
}
```

### 첨자 연산자 []

```cpp
#include <iostream>
#include <stdexcept>

class SafeArray {
private:
    int* data;
    int size;

public:
    SafeArray(int s) : size(s) {
        data = new int[size]();
    }

    ~SafeArray() {
        delete[] data;
    }

    // [] 연산자 (읽기/쓰기)
    int& operator[](int index) {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // const 버전 (읽기 전용)
    const int& operator[](int index) const {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }
};

int main() {
    SafeArray arr(5);
    arr[0] = 10;
    arr[1] = 20;

    std::cout << arr[0] << std::endl;  // 10
    std::cout << arr[1] << std::endl;  // 20

    // arr[10] = 100;  // 예외 발생!

    return 0;
}
```

---

## 2. 복사 생성자

객체를 복사할 때 호출됩니다.

### 기본 복사

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(n), age(a) {
        std::cout << "일반 생성자" << std::endl;
    }

    // 복사 생성자
    Person(const Person& other) : name(other.name), age(other.age) {
        std::cout << "복사 생성자" << std::endl;
    }
};

int main() {
    Person p1("Alice", 25);    // 일반 생성자
    Person p2(p1);             // 복사 생성자
    Person p3 = p1;            // 복사 생성자

    return 0;
}
```

### 얕은 복사 vs 깊은 복사

```cpp
#include <iostream>
#include <cstring>

class String {
private:
    char* data;
    int length;

public:
    String(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }

    // 깊은 복사 생성자
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];  // 새 메모리 할당
        strcpy(data, other.data);     // 내용 복사
        std::cout << "깊은 복사" << std::endl;
    }

    ~String() {
        delete[] data;
    }

    void print() const {
        std::cout << data << std::endl;
    }
};

int main() {
    String s1("Hello");
    String s2 = s1;  // 깊은 복사

    s1.print();  // Hello
    s2.print();  // Hello

    return 0;
}
```

---

## 3. 복사 대입 연산자

이미 존재하는 객체에 다른 객체를 대입할 때 호출됩니다.

```cpp
#include <iostream>
#include <cstring>

class String {
private:
    char* data;
    int length;

public:
    String(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }

    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
    }

    // 복사 대입 연산자
    String& operator=(const String& other) {
        if (this != &other) {  // 자기 대입 체크
            delete[] data;     // 기존 메모리 해제

            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        return *this;
    }

    ~String() {
        delete[] data;
    }

    void print() const {
        std::cout << data << std::endl;
    }
};

int main() {
    String s1("Hello");
    String s2("World");

    s2 = s1;  // 복사 대입 연산자

    s1.print();  // Hello
    s2.print();  // Hello

    return 0;
}
```

---

## 4. 이동 시맨틱 (C++11)

임시 객체의 리소스를 "이동"하여 불필요한 복사를 방지합니다.

### 이동 생성자

```cpp
#include <iostream>
#include <cstring>
#include <utility>  // std::move

class String {
private:
    char* data;
    int length;

public:
    String(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        std::cout << "일반 생성자" << std::endl;
    }

    // 복사 생성자
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
        std::cout << "복사 생성자" << std::endl;
    }

    // 이동 생성자
    String(String&& other) noexcept {
        data = other.data;      // 포인터만 복사
        length = other.length;
        other.data = nullptr;   // 원본 무효화
        other.length = 0;
        std::cout << "이동 생성자" << std::endl;
    }

    ~String() {
        delete[] data;
    }

    void print() const {
        if (data) std::cout << data << std::endl;
        else std::cout << "(empty)" << std::endl;
    }
};

int main() {
    String s1("Hello");           // 일반 생성자
    String s2 = s1;               // 복사 생성자
    String s3 = std::move(s1);    // 이동 생성자

    s1.print();  // (empty) - 이동됨
    s2.print();  // Hello
    s3.print();  // Hello

    return 0;
}
```

### 이동 대입 연산자

```cpp
// 이동 대입 연산자
String& operator=(String&& other) noexcept {
    if (this != &other) {
        delete[] data;          // 기존 메모리 해제

        data = other.data;      // 포인터 이동
        length = other.length;

        other.data = nullptr;   // 원본 무효화
        other.length = 0;
    }
    std::cout << "이동 대입" << std::endl;
    return *this;
}
```

### Rule of Five

리소스를 관리하는 클래스는 5가지를 모두 정의해야 합니다:

1. 소멸자
2. 복사 생성자
3. 복사 대입 연산자
4. 이동 생성자
5. 이동 대입 연산자

```cpp
class Resource {
public:
    Resource();                                    // 생성자
    ~Resource();                                   // 1. 소멸자
    Resource(const Resource& other);              // 2. 복사 생성자
    Resource& operator=(const Resource& other);   // 3. 복사 대입
    Resource(Resource&& other) noexcept;          // 4. 이동 생성자
    Resource& operator=(Resource&& other) noexcept; // 5. 이동 대입
};
```

---

## 5. static 멤버

클래스의 모든 객체가 공유하는 멤버입니다.

### static 멤버 변수

```cpp
#include <iostream>

class Counter {
private:
    static int count;  // 선언

public:
    Counter() {
        count++;
    }

    ~Counter() {
        count--;
    }

    static int getCount() {  // static 멤버 함수
        return count;
    }
};

// 정의 (클래스 외부)
int Counter::count = 0;

int main() {
    std::cout << "개수: " << Counter::getCount() << std::endl;  // 0

    Counter c1;
    Counter c2;
    std::cout << "개수: " << Counter::getCount() << std::endl;  // 2

    {
        Counter c3;
        std::cout << "개수: " << Counter::getCount() << std::endl;  // 3
    }

    std::cout << "개수: " << Counter::getCount() << std::endl;  // 2

    return 0;
}
```

### static 멤버 함수

```cpp
#include <iostream>

class Math {
public:
    static int add(int a, int b) {
        return a + b;
    }

    static int multiply(int a, int b) {
        return a * b;
    }

    static const double PI;
};

const double Math::PI = 3.14159;

int main() {
    // 객체 없이 호출
    std::cout << Math::add(3, 5) << std::endl;       // 8
    std::cout << Math::multiply(3, 5) << std::endl;  // 15
    std::cout << Math::PI << std::endl;              // 3.14159

    return 0;
}
```

---

## 6. friend

클래스의 private 멤버에 접근할 수 있는 외부 함수나 클래스입니다.

### friend 함수

```cpp
#include <iostream>

class Box {
private:
    double width;

public:
    Box(double w) : width(w) {}

    // friend 함수 선언
    friend void printWidth(const Box& b);
    friend double addWidths(const Box& a, const Box& b);
};

// friend 함수 정의
void printWidth(const Box& b) {
    std::cout << "Width: " << b.width << std::endl;  // private 접근 가능
}

double addWidths(const Box& a, const Box& b) {
    return a.width + b.width;
}

int main() {
    Box b1(10), b2(20);

    printWidth(b1);  // Width: 10
    std::cout << "합: " << addWidths(b1, b2) << std::endl;  // 합: 30

    return 0;
}
```

### friend 클래스

```cpp
#include <iostream>

class Engine {
private:
    int horsepower;

public:
    Engine(int hp) : horsepower(hp) {}

    friend class Car;  // Car가 Engine의 private에 접근 가능
};

class Car {
private:
    Engine engine;

public:
    Car(int hp) : engine(hp) {}

    void showHorsepower() const {
        std::cout << "마력: " << engine.horsepower << std::endl;
    }
};

int main() {
    Car car(300);
    car.showHorsepower();  // 마력: 300

    return 0;
}
```

---

## 7. explicit

암시적 형변환을 방지합니다.

```cpp
#include <iostream>

class Fraction {
private:
    int numerator;
    int denominator;

public:
    // explicit 없으면 Fraction f = 5; 가능
    explicit Fraction(int n, int d = 1) : numerator(n), denominator(d) {}

    void print() const {
        std::cout << numerator << "/" << denominator << std::endl;
    }
};

void printFraction(const Fraction& f) {
    f.print();
}

int main() {
    Fraction f1(3, 4);
    f1.print();  // 3/4

    Fraction f2(5);  // 명시적 호출 OK
    f2.print();  // 5/1

    // Fraction f3 = 5;  // 에러! explicit
    // printFraction(10);  // 에러! 암시적 변환 불가

    printFraction(Fraction(10));  // OK: 명시적 변환

    return 0;
}
```

---

## 8. 실습 예제: 완전한 String 클래스

```cpp
#include <iostream>
#include <cstring>
#include <utility>

class String {
private:
    char* data;
    size_t length;

public:
    // 기본 생성자
    String() : data(nullptr), length(0) {
        data = new char[1];
        data[0] = '\0';
    }

    // 문자열 생성자
    String(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }

    // 복사 생성자
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
    }

    // 이동 생성자
    String(String&& other) noexcept
        : data(other.data), length(other.length) {
        other.data = nullptr;
        other.length = 0;
    }

    // 소멸자
    ~String() {
        delete[] data;
    }

    // 복사 대입
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        return *this;
    }

    // 이동 대입
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }

    // + 연산자
    String operator+(const String& other) const {
        char* newData = new char[length + other.length + 1];
        strcpy(newData, data);
        strcat(newData, other.data);
        String result(newData);
        delete[] newData;
        return result;
    }

    // == 연산자
    bool operator==(const String& other) const {
        return strcmp(data, other.data) == 0;
    }

    // [] 연산자
    char& operator[](size_t index) {
        return data[index];
    }

    const char& operator[](size_t index) const {
        return data[index];
    }

    // << 연산자
    friend std::ostream& operator<<(std::ostream& os, const String& s) {
        return os << s.data;
    }

    size_t size() const { return length; }
    const char* c_str() const { return data; }
};

int main() {
    String s1("Hello");
    String s2(" World");
    String s3 = s1 + s2;

    std::cout << s3 << std::endl;  // Hello World
    std::cout << "길이: " << s3.size() << std::endl;  // 11

    return 0;
}
```

---

## 9. 요약

| 개념 | 설명 |
|------|------|
| 연산자 오버로딩 | 클래스에 연산자 정의 |
| 복사 생성자 | `T(const T&)` |
| 복사 대입 | `T& operator=(const T&)` |
| 이동 생성자 | `T(T&&)` |
| 이동 대입 | `T& operator=(T&&)` |
| `static` | 클래스 공유 멤버 |
| `friend` | private 접근 허용 |
| `explicit` | 암시적 변환 방지 |

---

**이전**: [클래스 기초](./07_Classes_Basics.md) | **다음**: [상속과 다형성](./09_Inheritance_and_Polymorphism.md)
