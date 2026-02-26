# 상속과 다형성

**이전**: [클래스 심화](./08_Classes_Advanced.md) | **다음**: [STL 컨테이너](./10_STL_Containers.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 상속의 개념을 설명하고 `public`, `protected`, `private` 상속을 사용하여 파생 클래스를 구현할 수 있습니다
2. 상속 계층 구조에서 생성자와 소멸자의 호출 순서를 파악할 수 있습니다
3. 함수 오버라이딩(`virtual` 없이)과 다형적 디스패치(`virtual` 사용)를 구별할 수 있습니다
4. `override`와 `final` 키워드(C++11)를 적용하여 의도를 명확히 하고 컴파일 타임 오류를 잡을 수 있습니다
5. 순수 가상 함수를 사용하여 추상 클래스를 설계하고 인터페이스 패턴을 구현할 수 있습니다
6. 기반 클래스 소멸자가 `virtual`이어야 하는 이유를 설명하고, 그렇지 않을 때 발생하는 메모리 누수 시나리오를 식별할 수 있습니다
7. 다중 상속에서 발생하는 다이아몬드 문제를 파악하고 가상 상속으로 해결할 수 있습니다
8. `dynamic_cast`와 `typeid`를 적용하여 안전한 런타임 타입 식별(RTTI)을 수행할 수 있습니다

---

상속과 다형성은 일반적인 인터페이스에 맞게 코드를 작성하고, 어떤 구체적인 구현과도 원활하게 동작하도록 만드는 메커니즘입니다. 이는 플러그인 아키텍처, GUI 프레임워크, 게임 엔진 등 기존 코드를 재작성하지 않고 새로운 동작을 추가해야 하는 모든 곳의 기반이 됩니다. 내부적으로 가상 디스패치 테이블이 어떻게 동작하는지 이해하면 다형성이 적절한 도구인지, 아니면 더 단순한 대안으로 충분한지에 대해 정보에 입각한 결정을 내리는 데 도움이 됩니다.

## 1. 상속이란?

상속은 기존 클래스(부모)의 속성과 메서드를 새 클래스(자식)가 물려받는 것입니다.

```cpp
#include <iostream>
#include <string>

// 부모 클래스 (기반 클래스)
class Animal {
public:
    std::string name;

    void eat() {
        std::cout << name << "이(가) 먹습니다." << std::endl;
    }

    void sleep() {
        std::cout << name << "이(가) 잠을 잡니다." << std::endl;
    }
};

// 자식 클래스 (파생 클래스)
class Dog : public Animal {
public:
    void bark() {
        std::cout << name << ": 멍멍!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void meow() {
        std::cout << name << ": 야옹!" << std::endl;
    }
};

int main() {
    Dog dog;
    dog.name = "바둑이";
    dog.eat();    // 상속받은 메서드
    dog.bark();   // Dog만의 메서드

    Cat cat;
    cat.name = "나비";
    cat.sleep();  // 상속받은 메서드
    cat.meow();   // Cat만의 메서드

    return 0;
}
```

---

## 2. 상속 접근 지정자

```cpp
class Base {
public:
    int publicVar;
protected:
    int protectedVar;
private:
    int privateVar;
};

// public 상속 (가장 일반적)
class PublicDerived : public Base {
    // publicVar → public
    // protectedVar → protected
    // privateVar → 접근 불가
};

// protected 상속
class ProtectedDerived : protected Base {
    // publicVar → protected
    // protectedVar → protected
    // privateVar → 접근 불가
};

// private 상속
class PrivateDerived : private Base {
    // publicVar → private
    // protectedVar → private
    // privateVar → 접근 불가
};
```

### 접근 지정자 요약

| 부모 멤버 | public 상속 | protected 상속 | private 상속 |
|----------|------------|----------------|--------------|
| public | public | protected | private |
| protected | protected | protected | private |
| private | 접근 불가 | 접근 불가 | 접근 불가 |

---

## 3. 생성자와 소멸자 호출 순서

```cpp
#include <iostream>

class Base {
public:
    Base() {
        std::cout << "Base 생성자" << std::endl;
    }
    ~Base() {
        std::cout << "Base 소멸자" << std::endl;
    }
};

class Derived : public Base {
public:
    Derived() {
        std::cout << "Derived 생성자" << std::endl;
    }
    ~Derived() {
        std::cout << "Derived 소멸자" << std::endl;
    }
};

int main() {
    Derived d;
    return 0;
}
```

출력:
```
Base 생성자
Derived 생성자
Derived 소멸자
Base 소멸자
```

### 부모 생성자 호출

```cpp
#include <iostream>
#include <string>

class Person {
protected:
    std::string name;
    int age;

public:
    Person(std::string n, int a) : name(n), age(a) {
        std::cout << "Person 생성자" << std::endl;
    }
};

class Student : public Person {
private:
    int studentId;

public:
    // 부모 생성자 호출
    Student(std::string n, int a, int id)
        : Person(n, a), studentId(id) {  // 초기화 리스트에서 호출
        std::cout << "Student 생성자" << std::endl;
    }

    void show() const {
        std::cout << "이름: " << name << ", 나이: " << age
                  << ", 학번: " << studentId << std::endl;
    }
};

int main() {
    Student s("Alice", 20, 20210001);
    s.show();
    return 0;
}
```

---

## 4. 함수 오버라이딩

자식 클래스에서 부모의 함수를 재정의합니다.

```cpp
#include <iostream>

class Animal {
public:
    void speak() {
        std::cout << "동물이 소리를 냅니다." << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() {  // 오버라이딩
        std::cout << "멍멍!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void speak() {  // 오버라이딩
        std::cout << "야옹!" << std::endl;
    }
};

int main() {
    Animal a;
    Dog d;
    Cat c;

    a.speak();  // 동물이 소리를 냅니다.
    d.speak();  // 멍멍!
    c.speak();  // 야옹!

    return 0;
}
```

### 부모 함수 호출

```cpp
class Dog : public Animal {
public:
    void speak() {
        Animal::speak();  // 부모 함수 호출
        std::cout << "멍멍!" << std::endl;
    }
};
```

---

## 5. 가상 함수 (virtual)

런타임에 적절한 함수를 호출합니다 (동적 바인딩).

### 문제: 정적 바인딩

```cpp
#include <iostream>

class Animal {
public:
    void speak() {
        std::cout << "동물 소리" << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() {
        std::cout << "멍멍!" << std::endl;
    }
};

int main() {
    Dog dog;
    Animal* ptr = &dog;

    ptr->speak();  // "동물 소리" (문제!)

    return 0;
}
```

### 해결: virtual 키워드

```cpp
#include <iostream>

class Animal {
public:
    virtual void speak() {  // virtual 추가
        std::cout << "동물 소리" << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() override {  // override (C++11, 선택적)
        std::cout << "멍멍!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void speak() override {
        std::cout << "야옹!" << std::endl;
    }
};

int main() {
    Dog dog;
    Cat cat;

    Animal* ptr1 = &dog;
    Animal* ptr2 = &cat;

    ptr1->speak();  // 멍멍! (올바름!)
    ptr2->speak();  // 야옹! (올바름!)

    return 0;
}
```

### override 키워드 (C++11)

```cpp
class Base {
public:
    virtual void foo(int x) {}
};

class Derived : public Base {
public:
    void foo(int x) override {}     // OK
    // void foo(double x) override {}  // 에러! 시그니처 불일치
    // void bar() override {}          // 에러! 부모에 bar 없음
};
```

### final 키워드 (C++11)

```cpp
class Base {
public:
    virtual void foo() final {}  // 더 이상 오버라이드 불가
};

class Derived : public Base {
public:
    // void foo() override {}  // 에러! final 함수
};

// 클래스 상속 금지
class FinalClass final {};
// class Derived2 : public FinalClass {};  // 에러!
```

---

## 6. 가상 소멸자

기반 클래스의 소멸자는 반드시 virtual이어야 합니다.

```cpp
#include <iostream>

class Base {
public:
    Base() { std::cout << "Base 생성" << std::endl; }
    virtual ~Base() { std::cout << "Base 소멸" << std::endl; }  // virtual!
};

class Derived : public Base {
private:
    int* data;

public:
    Derived() {
        data = new int[100];
        std::cout << "Derived 생성" << std::endl;
    }
    ~Derived() {
        delete[] data;
        std::cout << "Derived 소멸" << std::endl;
    }
};

int main() {
    Base* ptr = new Derived();
    delete ptr;  // virtual 덕분에 Derived 소멸자도 호출됨

    return 0;
}
```

출력:
```
Base 생성
Derived 생성
Derived 소멸
Base 소멸
```

---

## 7. 순수 가상 함수와 추상 클래스

### 순수 가상 함수

```cpp
class Shape {
public:
    // 순수 가상 함수 (= 0)
    virtual double getArea() const = 0;
    virtual double getPerimeter() const = 0;

    virtual ~Shape() = default;
};

// Shape shape;  // 에러! 추상 클래스는 인스턴스 생성 불가
```

### 추상 클래스 구현

```cpp
#include <iostream>
#include <cmath>

class Shape {
public:
    virtual double getArea() const = 0;
    virtual double getPerimeter() const = 0;
    virtual void draw() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double getArea() const override {
        return M_PI * radius * radius;
    }

    double getPerimeter() const override {
        return 2 * M_PI * radius;
    }

    void draw() const override {
        std::cout << "원을 그립니다. 반지름: " << radius << std::endl;
    }
};

class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double getArea() const override {
        return width * height;
    }

    double getPerimeter() const override {
        return 2 * (width + height);
    }

    void draw() const override {
        std::cout << "사각형을 그립니다. "
                  << width << " x " << height << std::endl;
    }
};

int main() {
    Circle c(5);
    Rectangle r(4, 3);

    Shape* shapes[] = {&c, &r};

    for (Shape* s : shapes) {
        s->draw();
        std::cout << "  넓이: " << s->getArea() << std::endl;
        std::cout << "  둘레: " << s->getPerimeter() << std::endl;
    }

    return 0;
}
```

---

## 8. 다중 상속

여러 부모 클래스를 상속받을 수 있습니다.

```cpp
#include <iostream>

class Flyable {
public:
    void fly() {
        std::cout << "날아갑니다." << std::endl;
    }
};

class Swimmable {
public:
    void swim() {
        std::cout << "헤엄칩니다." << std::endl;
    }
};

class Duck : public Flyable, public Swimmable {
public:
    void quack() {
        std::cout << "꽥꽥!" << std::endl;
    }
};

int main() {
    Duck duck;
    duck.fly();    // Flyable로부터
    duck.swim();   // Swimmable로부터
    duck.quack();  // Duck 고유

    return 0;
}
```

### 다이아몬드 문제

```cpp
#include <iostream>

class Animal {
public:
    int age;
};

class Mammal : public Animal {};
class Bird : public Animal {};

// Diamond 문제
class Bat : public Mammal, public Bird {
    // age가 두 번 상속됨!
};

int main() {
    Bat bat;
    // bat.age = 5;  // 에러! 모호함
    bat.Mammal::age = 5;  // 명시적 지정
    bat.Bird::age = 10;

    return 0;
}
```

### 가상 상속으로 해결

```cpp
#include <iostream>

class Animal {
public:
    int age;
};

class Mammal : virtual public Animal {};  // virtual 상속
class Bird : virtual public Animal {};    // virtual 상속

class Bat : public Mammal, public Bird {
    // age가 하나만 존재
};

int main() {
    Bat bat;
    bat.age = 5;  // OK!
    std::cout << bat.age << std::endl;

    return 0;
}
```

---

## 9. 인터페이스 패턴

순수 가상 함수만 있는 클래스입니다.

```cpp
#include <iostream>
#include <string>

// 인터페이스
class Printable {
public:
    virtual void print() const = 0;
    virtual ~Printable() = default;
};

class Serializable {
public:
    virtual std::string serialize() const = 0;
    virtual void deserialize(const std::string& data) = 0;
    virtual ~Serializable() = default;
};

// 여러 인터페이스 구현
class Document : public Printable, public Serializable {
private:
    std::string content;

public:
    Document(const std::string& c) : content(c) {}

    void print() const override {
        std::cout << "문서 내용: " << content << std::endl;
    }

    std::string serialize() const override {
        return "DOC:" + content;
    }

    void deserialize(const std::string& data) override {
        if (data.substr(0, 4) == "DOC:") {
            content = data.substr(4);
        }
    }
};

int main() {
    Document doc("Hello, World!");

    Printable* p = &doc;
    p->print();

    Serializable* s = &doc;
    std::cout << s->serialize() << std::endl;

    return 0;
}
```

---

## 10. RTTI (런타임 타입 정보)

### dynamic_cast

```cpp
#include <iostream>

class Base {
public:
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void derivedOnly() {
        std::cout << "Derived 전용 함수" << std::endl;
    }
};

int main() {
    Base* base = new Derived();

    // 안전한 다운캐스팅
    Derived* derived = dynamic_cast<Derived*>(base);
    if (derived) {
        derived->derivedOnly();
    }

    Base* base2 = new Base();
    Derived* derived2 = dynamic_cast<Derived*>(base2);
    if (derived2 == nullptr) {
        std::cout << "캐스팅 실패" << std::endl;
    }

    delete base;
    delete base2;

    return 0;
}
```

### typeid

```cpp
#include <iostream>
#include <typeinfo>

class Animal {
public:
    virtual ~Animal() = default;
};

class Dog : public Animal {};
class Cat : public Animal {};

int main() {
    Animal* a1 = new Dog();
    Animal* a2 = new Cat();

    std::cout << typeid(*a1).name() << std::endl;  // Dog 관련
    std::cout << typeid(*a2).name() << std::endl;  // Cat 관련

    if (typeid(*a1) == typeid(Dog)) {
        std::cout << "a1은 Dog입니다." << std::endl;
    }

    delete a1;
    delete a2;

    return 0;
}
```

---

## 11. 요약

| 개념 | 설명 |
|------|------|
| `class Derived : public Base` | 상속 |
| `virtual` | 가상 함수 (동적 바인딩) |
| `override` | 오버라이드 명시 |
| `final` | 상속/오버라이드 금지 |
| `= 0` | 순수 가상 함수 |
| 추상 클래스 | 순수 가상 함수 포함 |
| `virtual ~Base()` | 가상 소멸자 |
| `dynamic_cast` | 안전한 다운캐스팅 |

---

## 연습 문제

### 연습 1: 생성자/소멸자(Constructor/Destructor) 호출 순서

세 단계의 상속 계층을 만드세요: `Vehicle` → `Car` → `ElectricCar`. 각 클래스는 생성자와 소멸자에서 메시지를 출력해야 합니다. `main`에서 `ElectricCar` 객체를 스택에 생성하고 출력 순서가 예상과 일치하는지 확인하세요 (생성 시 기반 클래스 먼저, 소멸 시 파생 클래스 먼저). 그 다음 변수를 힙에 할당된 `ElectricCar`를 가리키는 `Vehicle*`로 변경하고, 삭제한 후 `Vehicle`의 소멸자가 `virtual`이 아닐 때 어떻게 변하는지 관찰하세요. 이유를 설명하세요.

### 연습 2: 다형성(Polymorphism)을 활용한 도형 면적 계산기

순수 가상 메서드(pure virtual method) `double area() const`를 가진 추상 기반 클래스(abstract base class) `Shape`를 설계하세요. 최소 세 가지 구체 클래스를 파생하세요: `Circle`, `Rectangle`, `Triangle`. 세 개 모두 `std::vector<Shape*>`에 저장하고 반복하면서 기반 클래스 포인터를 사용하여 각 도형의 면적을 출력하세요. 각 객체에 대해 올바른 `area()` 구현이 호출되는지 확인하세요 (다형적 디스패치(polymorphic dispatch)).

```cpp
#include <iostream>
#include <vector>
#include <cmath>

class Shape {
public:
    virtual double area() const = 0;
    virtual std::string name() const = 0;
    virtual ~Shape() = default;
};

// Circle, Rectangle, Triangle 여기에 구현하세요 ...

int main() {
    std::vector<Shape*> shapes;
    // 객체를 추가하고 반복하세요 ...
    return 0;
}
```

### 연습 3: override와 final 안전성

다음 기반 클래스가 주어졌을 때, `override` 키워드를 사용하여 `log()`를 오버라이드하는 파생 클래스 `FastLogger`를 추가하세요. 그 다음 의도적으로 시그니처 불일치를 도입하고(예: 매개변수 타입 변경) 컴파일 오류를 관찰하세요. 마지막으로 `FastLogger::log()`를 `final`로 표시하고 세 번째 클래스에서 오버라이드를 시도하여 컴파일 타임 오류를 확인하세요.

```cpp
class Logger {
public:
    virtual void log(const std::string& message) {
        std::cout << "[LOG] " << message << std::endl;
    }
    virtual ~Logger() = default;
};
```

런타임이 아닌 컴파일 타임에 이러한 오류를 잡는 것이 왜 더 바람직한지 주석으로 설명하세요.

### 연습 4: 인터페이스(Interface) 조합

두 개의 순수 가상 인터페이스를 정의하세요: `Drawable` (`void draw() const` 포함)와 `Resizable` (`void resize(double factor)` 포함). `std::vector<Drawable*>`를 저장하고 각 요소에 `draw()`를 호출하는 `Canvas` 클래스를 구현하세요. 그런 다음 두 인터페이스를 모두 구현하는 `Square` 클래스를 만드세요. 여러 `Square` 객체를 `Canvas`에 추가하고, 일부는 크기를 조정하고, 모두 그려보세요.

### 연습 5: 다이아몬드 문제(Diamond Problem) 해결

클래식 다이아몬드 계층을 만드세요: `Person` → `Employee`, `Person` → `Student`, `Employee` + `Student` → `WorkingStudent`. `Person` 클래스는 `std::string name` 멤버와 `greet()` 메서드를 가져야 합니다. 가상 상속(virtual inheritance) 없이 모호성 오류를 확인하세요. 그런 다음 `virtual public` 상속으로 수정하고 `WorkingStudent`가 모호함 없이 `greet()`를 호출할 수 있는지 확인하세요.

---

## 다음 단계

[STL 컨테이너](./10_STL_Containers.md)에서 STL 컨테이너를 배워봅시다!
