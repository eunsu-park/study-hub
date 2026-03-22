# 상속과 다형성

**이전**: [클래스 고급](./09_Classes_Advanced.md) | **다음**: [STL 컨테이너](./11_STL_Containers.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 상속을 설명하고 `public`, `protected`, `private` 상속으로 파생 클래스를 구현한다
2. 상속 계층에서 생성자와 소멸자 호출 순서를 파악한다
3. 함수 오버라이딩(`virtual` 없이)과 다형적 디스패치(`virtual` 포함)를 구분한다
4. `override`와 `final` 키워드(C++11)를 적용하여 의도를 명시하고 컴파일 타임에 오류를 잡는다
5. 순수 가상 함수를 사용하여 추상 클래스를 설계하고 인터페이스 패턴을 구현한다
6. 기본 클래스 소멸자가 왜 `virtual`이어야 하는지와 그렇지 않을 때의 메모리 누수 시나리오를 설명한다
7. 다중 상속의 다이아몬드 문제를 식별하고 가상 상속으로 해결한다
8. 안전한 런타임 타입 식별(RTTI)을 위해 `dynamic_cast`와 `typeid`를 적용한다

---

상속과 다형성은 일반적인 인터페이스에 대해 코드를 작성하면서 모든 특정 구현과 원활하게 작동하게 하는 메커니즘입니다. 이는 플러그인 아키텍처, GUI 프레임워크, 게임 엔진의 기반입니다 -- 기존 코드를 다시 작성하지 않고 새로운 동작을 추가해야 하는 모든 곳에서 사용됩니다.

## 1. 상속이란?

상속은 새 클래스(자식)가 기존 클래스(부모)의 속성과 메서드를 물려받는 것입니다.

```cpp
#include <iostream>
#include <string>

// 부모 클래스 (기본 클래스)
class Animal {
public:
    std::string name;

    void eat() { std::cout << name << " is eating." << std::endl; }
    void sleep() { std::cout << name << " is sleeping." << std::endl; }
};

// 자식 클래스 (파생 클래스)
class Dog : public Animal {
public:
    void bark() { std::cout << name << ": Woof woof!" << std::endl; }
};

class Cat : public Animal {
public:
    void meow() { std::cout << name << ": Meow!" << std::endl; }
};

int main() {
    Dog dog;
    dog.name = "Buddy";
    dog.eat();    // 상속된 메서드
    dog.bark();   // Dog 고유 메서드

    Cat cat;
    cat.name = "Whiskers";
    cat.sleep();  // 상속된 메서드
    cat.meow();   // Cat 고유 메서드

    return 0;
}
```

---

## 2. 상속 접근 지정자

| 부모 멤버 | public 상속 | protected 상속 | private 상속 |
|-----------|------------|---------------|-------------|
| public | public | protected | private |
| protected | protected | protected | private |
| private | 접근 불가 | 접근 불가 | 접근 불가 |

---

## 3. 생성자와 소멸자 호출 순서

```cpp
#include <iostream>

class Base {
public:
    Base() { std::cout << "Base constructor" << std::endl; }
    ~Base() { std::cout << "Base destructor" << std::endl; }
};

class Derived : public Base {
public:
    Derived() { std::cout << "Derived constructor" << std::endl; }
    ~Derived() { std::cout << "Derived destructor" << std::endl; }
};

int main() {
    Derived d;
    return 0;
}
```

출력:
```
Base constructor
Derived constructor
Derived destructor
Base destructor
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
        std::cout << "Person constructor" << std::endl;
    }
};

class Student : public Person {
private:
    int studentId;
public:
    Student(std::string n, int a, int id)
        : Person(n, a), studentId(id) {  // 초기화 리스트에서 호출
        std::cout << "Student constructor" << std::endl;
    }

    void show() const {
        std::cout << "Name: " << name << ", Age: " << age
                  << ", Student ID: " << studentId << std::endl;
    }
};

int main() {
    Student s("Alice", 20, 20210001);
    s.show();
    return 0;
}
```

---

## 4. 함수 오버라이딩(Function Overriding)

자식 클래스가 부모의 함수를 재정의합니다.

```cpp
#include <iostream>

class Animal {
public:
    void speak() { std::cout << "Animal makes a sound." << std::endl; }
};

class Dog : public Animal {
public:
    void speak() { std::cout << "Woof woof!" << std::endl; }  // 오버라이드
};

int main() {
    Animal a;
    Dog d;
    a.speak();  // Animal makes a sound.
    d.speak();  // Woof woof!
    return 0;
}
```

---

## 5. 가상 함수(Virtual Function)

런타임에 적절한 함수를 호출합니다 (동적 바인딩).

### 문제: 정적 바인딩

```cpp
#include <iostream>

class Animal {
public:
    void speak() { std::cout << "Animal sound" << std::endl; }
};

class Dog : public Animal {
public:
    void speak() { std::cout << "Woof woof!" << std::endl; }
};

int main() {
    Dog dog;
    Animal* ptr = &dog;
    ptr->speak();  // "Animal sound" (문제!)
    return 0;
}
```

### 해결: virtual 키워드

```cpp
#include <iostream>

class Animal {
public:
    virtual void speak() { std::cout << "Animal sound" << std::endl; }
};

class Dog : public Animal {
public:
    void speak() override { std::cout << "Woof woof!" << std::endl; }
};

class Cat : public Animal {
public:
    void speak() override { std::cout << "Meow!" << std::endl; }
};

int main() {
    Dog dog;
    Cat cat;
    Animal* ptr1 = &dog;
    Animal* ptr2 = &cat;
    ptr1->speak();  // Woof woof! (올바름!)
    ptr2->speak();  // Meow! (올바름!)
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
    // void foo(double x) override {}  // 오류! 시그니처 불일치
    // void bar() override {}          // 오류! 부모에 bar 없음
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
    // void foo() override {}  // 오류! final 함수
};

// 클래스 상속 방지
class FinalClass final {};
// class Derived2 : public FinalClass {};  // 오류!
```

---

## 6. 가상 소멸자(Virtual Destructor)

기본 클래스 소멸자는 반드시 virtual이어야 합니다.

```cpp
#include <iostream>

class Base {
public:
    Base() { std::cout << "Base created" << std::endl; }
    virtual ~Base() { std::cout << "Base destroyed" << std::endl; }  // virtual!
};

class Derived : public Base {
private:
    int* data;
public:
    Derived() { data = new int[100]; std::cout << "Derived created" << std::endl; }
    ~Derived() { delete[] data; std::cout << "Derived destroyed" << std::endl; }
};

int main() {
    Base* ptr = new Derived();
    delete ptr;  // virtual 덕분에 Derived 소멸자도 호출됨
    return 0;
}
```

출력:
```
Base created
Derived created
Derived destroyed
Base destroyed
```

---

## 7. 순수 가상 함수와 추상 클래스

```cpp
#include <iostream>
#include <cmath>

class Shape {
public:
    virtual double getArea() const = 0;       // 순수 가상 함수
    virtual double getPerimeter() const = 0;
    virtual void draw() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
private:
    double radius;
public:
    Circle(double r) : radius(r) {}
    double getArea() const override { return M_PI * radius * radius; }
    double getPerimeter() const override { return 2 * M_PI * radius; }
    void draw() const override { std::cout << "Drawing circle. Radius: " << radius << std::endl; }
};

class Rectangle : public Shape {
private:
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double getArea() const override { return width * height; }
    double getPerimeter() const override { return 2 * (width + height); }
    void draw() const override { std::cout << "Drawing rectangle. " << width << " x " << height << std::endl; }
};

int main() {
    Circle c(5);
    Rectangle r(4, 3);
    Shape* shapes[] = {&c, &r};

    for (Shape* s : shapes) {
        s->draw();
        std::cout << "  Area: " << s->getArea() << std::endl;
        std::cout << "  Perimeter: " << s->getPerimeter() << std::endl;
    }
    return 0;
}
```

---

## 8. 다중 상속(Multiple Inheritance)

```cpp
#include <iostream>

class Flyable {
public:
    void fly() { std::cout << "Flying." << std::endl; }
};

class Swimmable {
public:
    void swim() { std::cout << "Swimming." << std::endl; }
};

class Duck : public Flyable, public Swimmable {
public:
    void quack() { std::cout << "Quack quack!" << std::endl; }
};

int main() {
    Duck duck;
    duck.fly();    // Flyable에서
    duck.swim();   // Swimmable에서
    duck.quack();  // Duck 고유
    return 0;
}
```

### 다이아몬드 문제

```cpp
class Animal { public: int age; };
class Mammal : public Animal {};
class Bird : public Animal {};

class Bat : public Mammal, public Bird {
    // age가 두 번 상속됨!
};
```

### 가상 상속으로 해결

```cpp
#include <iostream>

class Animal { public: int age; };
class Mammal : virtual public Animal {};  // 가상 상속
class Bird : virtual public Animal {};    // 가상 상속

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

class Document : public Printable, public Serializable {
private:
    std::string content;
public:
    Document(const std::string& c) : content(c) {}
    void print() const override { std::cout << "Document content: " << content << std::endl; }
    std::string serialize() const override { return "DOC:" + content; }
    void deserialize(const std::string& data) override {
        if (data.substr(0, 4) == "DOC:") content = data.substr(4);
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
    void derivedOnly() { std::cout << "Derived only function" << std::endl; }
};

int main() {
    Base* base = new Derived();
    Derived* derived = dynamic_cast<Derived*>(base);
    if (derived) { derived->derivedOnly(); }

    Base* base2 = new Base();
    Derived* derived2 = dynamic_cast<Derived*>(base2);
    if (derived2 == nullptr) { std::cout << "Cast failed" << std::endl; }

    delete base;
    delete base2;
    return 0;
}
```

### typeid

```cpp
#include <iostream>
#include <typeinfo>

class Animal { public: virtual ~Animal() = default; };
class Dog : public Animal {};
class Cat : public Animal {};

int main() {
    Animal* a1 = new Dog();
    Animal* a2 = new Cat();

    std::cout << typeid(*a1).name() << std::endl;
    std::cout << typeid(*a2).name() << std::endl;

    if (typeid(*a1) == typeid(Dog)) {
        std::cout << "a1 is a Dog." << std::endl;
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
| `override` | 명시적 오버라이드 |
| `final` | 상속/오버라이드 방지 |
| `= 0` | 순수 가상 함수 |
| 추상 클래스 | 순수 가상 함수를 포함 |
| `virtual ~Base()` | 가상 소멸자 |
| `dynamic_cast` | 안전한 다운캐스팅 |

---

## 연습문제

### 연습문제 1: 생성자/소멸자 순서

3단계 상속 체인을 만드세요: `Vehicle` -> `Car` -> `ElectricCar`. 각 클래스는 생성자와 소멸자에서 메시지를 출력해야 합니다. `main`에서 `ElectricCar` 객체를 스택에 생성하고 출력 순서가 예상과 일치하는지 확인하세요. 그런 다음 변수를 힙에 할당된 `ElectricCar`를 가리키는 `Vehicle*`로 변경하고 삭제한 후 `Vehicle`의 소멸자가 `virtual`이 아닌 경우 무엇이 변하는지 관찰하세요. 이유를 설명하세요.

### 연습문제 2: 다형적 도형 면적 계산기

순수 가상 메서드 `double area() const`를 가진 추상 기본 클래스 `Shape`를 설계하세요. 최소 3개의 구체 클래스를 파생시키세요: `Circle`, `Rectangle`, `Triangle`. 셋 모두를 `std::vector<Shape*>`에 저장하고 반복하며 기본 클래스 포인터를 사용하여 각 도형의 면적을 출력하세요.

### 연습문제 3: override와 final 안전성

기본 클래스 `Logger`의 `log()` 메서드를 `override`로 오버라이드하는 파생 클래스 `FastLogger`를 추가하세요. 의도적으로 시그니처 불일치를 만들어 컴파일러 오류를 관찰하세요. 마지막으로 `FastLogger::log()`를 `final`로 표시하고 세 번째 클래스에서 오버라이드를 시도하세요.

### 연습문제 4: 인터페이스 합성

두 개의 순수 가상 인터페이스를 정의하세요: `Drawable`(`void draw() const`)와 `Resizable`(`void resize(double factor)`). 두 인터페이스를 모두 구현하는 `Square` 클래스를 만드세요. `Canvas` 클래스가 `std::vector<Drawable*>`를 저장하고 각 요소에 `draw()`를 호출하게 하세요.

### 연습문제 5: 다이아몬드 문제 해결

고전적인 다이아몬드 계층을 만드세요: `Person` -> `Employee`, `Person` -> `Student`, `Employee` + `Student` -> `WorkingStudent`. 가상 상속 없이 모호성 오류를 보여주고, `virtual public` 상속으로 수정하세요.

---

## 다음 단계

[STL 컨테이너](./11_STL_Containers.md)에서 STL 컨테이너에 대해 알아봅시다!
