# 클래스 기초

**이전**: [포인터와 참조](./06_Pointers_and_References.md) | **다음**: [클래스 심화](./08_Classes_Advanced.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 클래스(class)가 무엇인지, 데이터와 동작을 하나의 타입으로 어떻게 결합하는지 설명할 수 있습니다
2. `public`, `private`, `protected` 접근 지정자(access specifier)를 적용하여 캡슐화(encapsulation)를 강제할 수 있습니다
3. 생성자(constructor)(기본, 매개변수 있는)를 구현하고 멤버 초기화 목록(member initializer list)을 활용할 수 있습니다
4. 객체가 범위를 벗어날 때 자원을 해제하는 소멸자(destructor)를 설계할 수 있습니다
5. `this` 포인터를 모호성 해소와 메서드 체이닝(method chaining)에 적용할 수 있습니다
6. `const` 멤버 함수와 비-`const` 멤버 함수를 구별하고, getter가 `const`여야 하는 이유를 설명할 수 있습니다
7. 클래스 선언을 헤더(`.h`)와 구현(`.cpp`) 파일에 분리하여 구성할 수 있습니다
8. C++에서 `struct`와 `class`를 비교하고 관례에 따른 사용법을 설명할 수 있습니다

---

클래스(class)는 C++에서 객체 지향 프로그래밍(object-oriented programming)으로 나아가는 관문입니다. 은행 계좌, 센서, 게임 캐릭터와 같은 실세계 개체를 데이터와 그 데이터를 다루는 연산을 하나로 묶은 독립적인 단위로 모델링할 수 있게 해줍니다. 생성자가 객체를 초기화하는 방식, 접근 지정자가 불변 조건(invariant)을 보호하는 방식, 소멸자가 자원을 정리하는 방식을 이해하면, 앞으로 마주치게 될 사실상 모든 C++ 라이브러리와 프레임워크의 개념적 토대를 갖추게 됩니다.

## 1. 클래스란?

클래스는 데이터와 함수를 하나로 묶은 사용자 정의 타입입니다.

```cpp
#include <iostream>
#include <string>

class Person {
public:
    // 멤버 변수 (데이터)
    std::string name;
    int age;

    // 멤버 함수 (동작)
    void introduce() {
        std::cout << "안녕하세요, " << name << "입니다. "
                  << age << "살입니다." << std::endl;
    }
};

int main() {
    // 객체 생성
    Person p1;
    p1.name = "Alice";
    p1.age = 25;
    p1.introduce();  // 안녕하세요, Alice입니다. 25살입니다.

    Person p2;
    p2.name = "Bob";
    p2.age = 30;
    p2.introduce();

    return 0;
}
```

---

## 2. 접근 지정자

| 지정자 | 클래스 내부 | 파생 클래스 | 외부 |
|--------|------------|------------|------|
| `public` | O | O | O |
| `protected` | O | O | X |
| `private` | O | X | X |

```cpp
#include <iostream>

class Example {
public:
    int publicVar = 1;

protected:
    int protectedVar = 2;

private:
    int privateVar = 3;

public:
    void showAll() {
        // 클래스 내부에서는 모두 접근 가능
        std::cout << publicVar << std::endl;
        std::cout << protectedVar << std::endl;
        std::cout << privateVar << std::endl;
    }
};

int main() {
    Example ex;

    std::cout << ex.publicVar << std::endl;  // OK
    // std::cout << ex.protectedVar << std::endl;  // 에러!
    // std::cout << ex.privateVar << std::endl;  // 에러!

    ex.showAll();  // 내부에서는 모두 접근 가능

    return 0;
}
```

### 캡슐화

```cpp
#include <iostream>
#include <string>

class BankAccount {
private:
    std::string owner;
    double balance;

public:
    // Getter
    std::string getOwner() const {
        return owner;
    }

    double getBalance() const {
        return balance;
    }

    // Setter
    void setOwner(const std::string& name) {
        owner = name;
    }

    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
};

int main() {
    BankAccount account;
    account.setOwner("Alice");
    account.deposit(1000);

    std::cout << account.getOwner() << ": $"
              << account.getBalance() << std::endl;

    account.withdraw(300);
    std::cout << "출금 후: $" << account.getBalance() << std::endl;

    return 0;
}
```

---

## 3. 생성자 (Constructor)

객체가 생성될 때 자동으로 호출되는 특별한 함수입니다.

### 기본 생성자

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    // 기본 생성자
    Person() {
        name = "Unknown";
        age = 0;
        std::cout << "기본 생성자 호출" << std::endl;
    }
};

int main() {
    Person p;  // 기본 생성자 호출
    std::cout << p.name << ", " << p.age << std::endl;
    return 0;
}
```

### 매개변수가 있는 생성자

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    // 기본 생성자
    Person() : name("Unknown"), age(0) {}

    // 매개변수 있는 생성자
    Person(std::string n, int a) {
        name = n;
        age = a;
    }
};

int main() {
    Person p1;                    // 기본 생성자
    Person p2("Alice", 25);       // 매개변수 생성자
    Person p3 = {"Bob", 30};      // C++11 초기화

    std::cout << p1.name << ", " << p1.age << std::endl;
    std::cout << p2.name << ", " << p2.age << std::endl;
    std::cout << p3.name << ", " << p3.age << std::endl;

    return 0;
}
```

### 멤버 초기화 리스트

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;
    const int id;  // const 멤버

public:
    // 멤버 초기화 리스트 (권장)
    Person(std::string n, int a, int i)
        : name(n), age(a), id(i)  // 초기화 리스트
    {
        // 본문
        std::cout << "Person 생성: " << name << std::endl;
    }

    void show() const {
        std::cout << "ID: " << id << ", " << name << ", " << age << std::endl;
    }
};

int main() {
    Person p("Alice", 25, 1001);
    p.show();
    return 0;
}
```

### 초기화 리스트를 사용해야 하는 경우

1. **const 멤버** 초기화
2. **참조 멤버** 초기화
3. 부모 클래스 생성자 호출
4. 성능 (불필요한 기본 생성 방지)

---

## 4. 소멸자 (Destructor)

객체가 소멸될 때 자동으로 호출됩니다.

```cpp
#include <iostream>

class Resource {
private:
    int* data;

public:
    Resource(int size) {
        data = new int[size];
        std::cout << "리소스 할당" << std::endl;
    }

    ~Resource() {  // 소멸자
        delete[] data;
        std::cout << "리소스 해제" << std::endl;
    }
};

int main() {
    {
        Resource r(100);
        // 블록 끝에서 r 소멸
    }
    std::cout << "블록 종료 후" << std::endl;

    return 0;
}
```

출력:
```
리소스 할당
리소스 해제
블록 종료 후
```

---

## 5. this 포인터

현재 객체를 가리키는 포인터입니다.

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string name, int age) {
        // this로 멤버와 매개변수 구분
        this->name = name;
        this->age = age;
    }

    // 메서드 체이닝을 위한 반환
    Person& setName(std::string name) {
        this->name = name;
        return *this;  // 자기 자신 반환
    }

    Person& setAge(int age) {
        this->age = age;
        return *this;
    }

    void show() const {
        std::cout << name << ", " << age << std::endl;
    }
};

int main() {
    Person p("Alice", 25);

    // 메서드 체이닝
    p.setName("Bob").setAge(30);
    p.show();  // Bob, 30

    return 0;
}
```

---

## 6. const 멤버 함수

객체를 수정하지 않는 함수입니다.

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string n, int a) : name(n), age(a) {}

    // const 멤버 함수: 객체 수정 불가
    std::string getName() const {
        // name = "Other";  // 에러!
        return name;
    }

    int getAge() const {
        return age;
    }

    // non-const 멤버 함수: 객체 수정 가능
    void setAge(int a) {
        age = a;
    }
};

void printPerson(const Person& p) {
    // const 객체는 const 멤버 함수만 호출 가능
    std::cout << p.getName() << ", " << p.getAge() << std::endl;
    // p.setAge(30);  // 에러!
}

int main() {
    Person p("Alice", 25);
    printPerson(p);
    return 0;
}
```

---

## 7. 클래스와 헤더 파일

### person.h

```cpp
#ifndef PERSON_H
#define PERSON_H

#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string n, int a);

    std::string getName() const;
    int getAge() const;
    void setAge(int a);
    void introduce() const;
};

#endif
```

### person.cpp

```cpp
#include "person.h"
#include <iostream>

Person::Person(std::string n, int a) : name(n), age(a) {}

std::string Person::getName() const {
    return name;
}

int Person::getAge() const {
    return age;
}

void Person::setAge(int a) {
    age = a;
}

void Person::introduce() const {
    std::cout << "안녕하세요, " << name << "입니다." << std::endl;
}
```

### main.cpp

```cpp
#include <iostream>
#include "person.h"

int main() {
    Person p("Alice", 25);
    p.introduce();
    return 0;
}
```

### 컴파일

```bash
g++ -c person.cpp -o person.o
g++ -c main.cpp -o main.o
g++ person.o main.o -o program
```

---

## 8. 구조체 vs 클래스

C++에서 `struct`와 `class`의 유일한 차이는 기본 접근 지정자입니다.

```cpp
struct MyStruct {
    int x;  // 기본: public
};

class MyClass {
    int x;  // 기본: private
};
```

### 관례

```cpp
// struct: 데이터 위주, 메서드 거의 없음
struct Point {
    int x;
    int y;
};

// class: 동작 포함, 캡슐화
class Rectangle {
private:
    Point topLeft;
    Point bottomRight;

public:
    int getArea() const;
};
```

---

## 9. 기본 생성자 규칙

```cpp
#include <iostream>

class A {
public:
    int value;
    // 생성자가 없으면 컴파일러가 기본 생성자 자동 생성
};

class B {
public:
    int value;
    B(int v) : value(v) {}
    // 사용자 정의 생성자가 있으면 기본 생성자 없음
};

class C {
public:
    int value;
    C(int v) : value(v) {}
    C() = default;  // 기본 생성자 명시적 생성
};

class D {
public:
    D() = delete;  // 기본 생성자 삭제
};

int main() {
    A a;           // OK
    // B b;        // 에러! 기본 생성자 없음
    B b(10);       // OK
    C c;           // OK (default로 명시)
    C c2(20);      // OK
    // D d;        // 에러! 삭제됨

    return 0;
}
```

---

## 10. 인라인 멤버 함수

```cpp
class Calculator {
public:
    // 클래스 내부 정의: 자동으로 inline
    int add(int a, int b) {
        return a + b;
    }

    // 외부 정의 함수의 inline
    int multiply(int a, int b);
};

// 클래스 외부에서 inline 지정
inline int Calculator::multiply(int a, int b) {
    return a * b;
}
```

---

## 11. 실습 예제

### Rectangle 클래스

```cpp
#include <iostream>

class Rectangle {
private:
    double width;
    double height;

public:
    // 생성자
    Rectangle() : width(0), height(0) {}
    Rectangle(double w, double h) : width(w), height(h) {}

    // Getter/Setter
    double getWidth() const { return width; }
    double getHeight() const { return height; }

    void setWidth(double w) {
        if (w >= 0) width = w;
    }

    void setHeight(double h) {
        if (h >= 0) height = h;
    }

    // 기능
    double getArea() const {
        return width * height;
    }

    double getPerimeter() const {
        return 2 * (width + height);
    }

    void display() const {
        std::cout << "Rectangle(" << width << " x " << height << ")" << std::endl;
        std::cout << "  넓이: " << getArea() << std::endl;
        std::cout << "  둘레: " << getPerimeter() << std::endl;
    }
};

int main() {
    Rectangle r1;
    r1.setWidth(5);
    r1.setHeight(3);
    r1.display();

    Rectangle r2(10, 4);
    r2.display();

    return 0;
}
```

### Student 클래스

```cpp
#include <iostream>
#include <string>
#include <vector>

class Student {
private:
    std::string name;
    int id;
    std::vector<int> scores;

public:
    Student(std::string n, int i) : name(n), id(i) {}

    void addScore(int score) {
        if (score >= 0 && score <= 100) {
            scores.push_back(score);
        }
    }

    double getAverage() const {
        if (scores.empty()) return 0;

        int sum = 0;
        for (int s : scores) {
            sum += s;
        }
        return static_cast<double>(sum) / scores.size();
    }

    void display() const {
        std::cout << "학번: " << id << ", 이름: " << name << std::endl;
        std::cout << "성적: ";
        for (int s : scores) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        std::cout << "평균: " << getAverage() << std::endl;
    }
};

int main() {
    Student s("Alice", 20210001);
    s.addScore(85);
    s.addScore(90);
    s.addScore(78);
    s.display();

    return 0;
}
```

---

## 12. 요약

| 개념 | 설명 |
|------|------|
| `class` | 사용자 정의 타입 |
| `public` | 어디서든 접근 가능 |
| `private` | 클래스 내부에서만 접근 |
| `protected` | 클래스와 파생 클래스에서 접근 |
| 생성자 | 객체 생성 시 호출 |
| 소멸자 | 객체 소멸 시 호출 |
| `this` | 현재 객체 포인터 |
| `const` 메서드 | 객체 수정 불가 |

---

**이전**: [포인터와 참조](./06_Pointers_and_References.md) | **다음**: [클래스 심화](./08_Classes_Advanced.md)
