# Inheritance and Polymorphism

**Previous**: [Advanced Classes](./08_Classes_Advanced.md) | **Next**: [STL Containers](./10_STL_Containers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain inheritance and implement derived classes using `public`, `protected`, and `private` inheritance
2. Identify the constructor and destructor call order in an inheritance hierarchy
3. Distinguish between function overriding (without `virtual`) and polymorphic dispatch (with `virtual`)
4. Apply the `override` and `final` keywords (C++11) to make intent explicit and catch errors at compile time
5. Design abstract classes using pure virtual functions and implement interface patterns
6. Explain why base-class destructors should be `virtual` and identify the memory-leak scenario when they are not
7. Identify the diamond problem in multiple inheritance and resolve it with virtual inheritance
8. Apply `dynamic_cast` and `typeid` for safe runtime type identification (RTTI)

---

Inheritance and polymorphism are the mechanisms that let you write code against a general interface and have it work seamlessly with any specific implementation. This is the foundation of plugin architectures, GUI frameworks, and game engines -- anywhere you need to add new behavior without rewriting existing code. Understanding how the virtual dispatch table works under the hood will also help you make informed decisions about when polymorphism is the right tool and when simpler alternatives suffice.

## 1. What is Inheritance?

Inheritance is when a new class (child) inherits properties and methods from an existing class (parent).

```cpp
#include <iostream>
#include <string>

// Parent class (base class)
class Animal {
public:
    std::string name;

    void eat() {
        std::cout << name << " is eating." << std::endl;
    }

    void sleep() {
        std::cout << name << " is sleeping." << std::endl;
    }
};

// Child class (derived class)
class Dog : public Animal {
public:
    void bark() {
        std::cout << name << ": Woof woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void meow() {
        std::cout << name << ": Meow!" << std::endl;
    }
};

int main() {
    Dog dog;
    dog.name = "Buddy";
    dog.eat();    // Inherited method
    dog.bark();   // Dog's own method

    Cat cat;
    cat.name = "Whiskers";
    cat.sleep();  // Inherited method
    cat.meow();   // Cat's own method

    return 0;
}
```

---

## 2. Inheritance Access Specifiers

```cpp
class Base {
public:
    int publicVar;
protected:
    int protectedVar;
private:
    int privateVar;
};

// public inheritance (most common)
class PublicDerived : public Base {
    // publicVar -> public
    // protectedVar -> protected
    // privateVar -> inaccessible
};

// protected inheritance
class ProtectedDerived : protected Base {
    // publicVar -> protected
    // protectedVar -> protected
    // privateVar -> inaccessible
};

// private inheritance
class PrivateDerived : private Base {
    // publicVar -> private
    // protectedVar -> private
    // privateVar -> inaccessible
};
```

### Access Specifier Summary

| Parent Member | public inheritance | protected inheritance | private inheritance |
|---------------|-------------------|----------------------|---------------------|
| public | public | protected | private |
| protected | protected | protected | private |
| private | inaccessible | inaccessible | inaccessible |

---

## 3. Constructor and Destructor Call Order

```cpp
#include <iostream>

class Base {
public:
    Base() {
        std::cout << "Base constructor" << std::endl;
    }
    ~Base() {
        std::cout << "Base destructor" << std::endl;
    }
};

class Derived : public Base {
public:
    Derived() {
        std::cout << "Derived constructor" << std::endl;
    }
    ~Derived() {
        std::cout << "Derived destructor" << std::endl;
    }
};

int main() {
    Derived d;
    return 0;
}
```

Output:
```
Base constructor
Derived constructor
Derived destructor
Base destructor
```

### Calling Parent Constructor

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
    // Call parent constructor
    Student(std::string n, int a, int id)
        : Person(n, a), studentId(id) {  // Call in initializer list
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

## 4. Function Overriding

Child class redefines parent's function.

```cpp
#include <iostream>

class Animal {
public:
    void speak() {
        std::cout << "Animal makes a sound." << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() {  // Override
        std::cout << "Woof woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void speak() {  // Override
        std::cout << "Meow!" << std::endl;
    }
};

int main() {
    Animal a;
    Dog d;
    Cat c;

    a.speak();  // Animal makes a sound.
    d.speak();  // Woof woof!
    c.speak();  // Meow!

    return 0;
}
```

### Calling Parent Function

```cpp
class Dog : public Animal {
public:
    void speak() {
        Animal::speak();  // Call parent function
        std::cout << "Woof woof!" << std::endl;
    }
};
```

---

## 5. Virtual Functions

Call the appropriate function at runtime (dynamic binding).

### Problem: Static Binding

```cpp
#include <iostream>

class Animal {
public:
    void speak() {
        std::cout << "Animal sound" << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() {
        std::cout << "Woof woof!" << std::endl;
    }
};

int main() {
    Dog dog;
    Animal* ptr = &dog;

    ptr->speak();  // "Animal sound" (problem!)

    return 0;
}
```

### Solution: virtual Keyword

```cpp
#include <iostream>

class Animal {
public:
    virtual void speak() {  // Add virtual
        std::cout << "Animal sound" << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() override {  // override (C++11, optional)
        std::cout << "Woof woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void speak() override {
        std::cout << "Meow!" << std::endl;
    }
};

int main() {
    Dog dog;
    Cat cat;

    Animal* ptr1 = &dog;
    Animal* ptr2 = &cat;

    ptr1->speak();  // Woof woof! (correct!)
    ptr2->speak();  // Meow! (correct!)

    return 0;
}
```

### override Keyword (C++11)

```cpp
class Base {
public:
    virtual void foo(int x) {}
};

class Derived : public Base {
public:
    void foo(int x) override {}     // OK
    // void foo(double x) override {}  // Error! Signature mismatch
    // void bar() override {}          // Error! No bar in parent
};
```

### final Keyword (C++11)

```cpp
class Base {
public:
    virtual void foo() final {}  // Cannot override further
};

class Derived : public Base {
public:
    // void foo() override {}  // Error! final function
};

// Prevent class inheritance
class FinalClass final {};
// class Derived2 : public FinalClass {};  // Error!
```

---

## 6. Virtual Destructor

Base class destructor must be virtual.

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
    Derived() {
        data = new int[100];
        std::cout << "Derived created" << std::endl;
    }
    ~Derived() {
        delete[] data;
        std::cout << "Derived destroyed" << std::endl;
    }
};

int main() {
    Base* ptr = new Derived();
    delete ptr;  // Thanks to virtual, Derived destructor is also called

    return 0;
}
```

Output:
```
Base created
Derived created
Derived destroyed
Base destroyed
```

---

## 7. Pure Virtual Functions and Abstract Classes

### Pure Virtual Functions

```cpp
class Shape {
public:
    // Pure virtual function (= 0)
    virtual double getArea() const = 0;
    virtual double getPerimeter() const = 0;

    virtual ~Shape() = default;
};

// Shape shape;  // Error! Cannot instantiate abstract class
```

### Implementing Abstract Classes

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
        std::cout << "Drawing circle. Radius: " << radius << std::endl;
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
        std::cout << "Drawing rectangle. "
                  << width << " x " << height << std::endl;
    }
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

## 8. Multiple Inheritance

Can inherit from multiple parent classes.

```cpp
#include <iostream>

class Flyable {
public:
    void fly() {
        std::cout << "Flying." << std::endl;
    }
};

class Swimmable {
public:
    void swim() {
        std::cout << "Swimming." << std::endl;
    }
};

class Duck : public Flyable, public Swimmable {
public:
    void quack() {
        std::cout << "Quack quack!" << std::endl;
    }
};

int main() {
    Duck duck;
    duck.fly();    // From Flyable
    duck.swim();   // From Swimmable
    duck.quack();  // Duck's own

    return 0;
}
```

### Diamond Problem

```cpp
#include <iostream>

class Animal {
public:
    int age;
};

class Mammal : public Animal {};
class Bird : public Animal {};

// Diamond problem
class Bat : public Mammal, public Bird {
    // age is inherited twice!
};

int main() {
    Bat bat;
    // bat.age = 5;  // Error! Ambiguous
    bat.Mammal::age = 5;  // Explicit specification
    bat.Bird::age = 10;

    return 0;
}
```

### Solution with Virtual Inheritance

```cpp
#include <iostream>

class Animal {
public:
    int age;
};

class Mammal : virtual public Animal {};  // virtual inheritance
class Bird : virtual public Animal {};    // virtual inheritance

class Bat : public Mammal, public Bird {
    // Only one age exists
};

int main() {
    Bat bat;
    bat.age = 5;  // OK!
    std::cout << bat.age << std::endl;

    return 0;
}
```

---

## 9. Interface Pattern

A class with only pure virtual functions.

```cpp
#include <iostream>
#include <string>

// Interface
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

// Implement multiple interfaces
class Document : public Printable, public Serializable {
private:
    std::string content;

public:
    Document(const std::string& c) : content(c) {}

    void print() const override {
        std::cout << "Document content: " << content << std::endl;
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

## 10. RTTI (Run-Time Type Information)

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
        std::cout << "Derived only function" << std::endl;
    }
};

int main() {
    Base* base = new Derived();

    // Safe downcasting
    Derived* derived = dynamic_cast<Derived*>(base);
    if (derived) {
        derived->derivedOnly();
    }

    Base* base2 = new Base();
    Derived* derived2 = dynamic_cast<Derived*>(base2);
    if (derived2 == nullptr) {
        std::cout << "Cast failed" << std::endl;
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

    std::cout << typeid(*a1).name() << std::endl;  // Dog related
    std::cout << typeid(*a2).name() << std::endl;  // Cat related

    if (typeid(*a1) == typeid(Dog)) {
        std::cout << "a1 is a Dog." << std::endl;
    }

    delete a1;
    delete a2;

    return 0;
}
```

---

## 11. Summary

| Concept | Description |
|---------|-------------|
| `class Derived : public Base` | Inheritance |
| `virtual` | Virtual function (dynamic binding) |
| `override` | Explicit override |
| `final` | Prevent inheritance/override |
| `= 0` | Pure virtual function |
| Abstract class | Contains pure virtual functions |
| `virtual ~Base()` | Virtual destructor |
| `dynamic_cast` | Safe downcasting |

---

## Exercises

### Exercise 1: Constructor/Destructor Order

Create a three-level inheritance chain: `Vehicle` → `Car` → `ElectricCar`. Each class should print a message in its constructor and destructor. In `main`, create an `ElectricCar` object on the stack and verify the output order matches your expectations (base first for construction, derived first for destruction). Then change the variable to a `Vehicle*` pointing to a heap-allocated `ElectricCar`, delete it, and observe what changes if `Vehicle`'s destructor is not `virtual`. Explain why.

### Exercise 2: Polymorphic Shape Area Calculator

Design an abstract base class `Shape` with a pure virtual method `double area() const`. Derive at least three concrete classes: `Circle`, `Rectangle`, and `Triangle`. Store all three in a `std::vector<Shape*>`, iterate over it, and print each shape's area using the base-class pointer. Verify that the correct `area()` implementation is called for each object (polymorphic dispatch).

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

// Implement Circle, Rectangle, Triangle here ...

int main() {
    std::vector<Shape*> shapes;
    // Add objects and iterate ...
    return 0;
}
```

### Exercise 3: override and final Safety

Given the following base class, add a derived class `FastLogger` that overrides `log()` using the `override` keyword. Then intentionally introduce a signature mismatch (e.g., change the parameter type) and observe the compiler error. Finally, mark `FastLogger::log()` as `final` and attempt to override it in a third class to see the compile-time error.

```cpp
class Logger {
public:
    virtual void log(const std::string& message) {
        std::cout << "[LOG] " << message << std::endl;
    }
    virtual ~Logger() = default;
};
```

Explain in a comment why catching these errors at compile time is preferable to discovering them at runtime.

### Exercise 4: Interface Composition

Define two pure-virtual interfaces: `Drawable` (with `void draw() const`) and `Resizable` (with `void resize(double factor)`). Implement a `Canvas` class that stores a `std::vector<Drawable*>` and calls `draw()` on each element. Then create a `Square` class that implements both interfaces. Add several `Square` objects to a `Canvas`, resize some of them, and draw all of them.

### Exercise 5: Resolving the Diamond Problem

Create the classic diamond hierarchy: `Person` → `Employee`, `Person` → `Student`, `Employee` + `Student` → `WorkingStudent`. The `Person` class should have a `std::string name` member and a `greet()` method. Without virtual inheritance, demonstrate the ambiguity error. Then fix it with `virtual public` inheritance and confirm that `WorkingStudent` can call `greet()` unambiguously.

---

## Next Step

Let's learn about STL containers in [10_STL_Containers.md](./10_STL_Containers.md)!
