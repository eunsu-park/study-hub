# Class Basics

**Previous**: [Pointers and References](./06_Pointers_and_References.md) | **Next**: [Advanced Classes](./08_Classes_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a class is and how it combines data and behavior into a single type
2. Apply `public`, `private`, and `protected` access specifiers to enforce encapsulation
3. Implement constructors (default, parameterized) and use member initializer lists
4. Design destructors to release resources when objects go out of scope
5. Apply the `this` pointer for disambiguation and method chaining
6. Distinguish between `const` and non-`const` member functions and explain why getters should be `const`
7. Organize class declarations across header (`.h`) and implementation (`.cpp`) files
8. Compare `struct` and `class` in C++ and explain convention-based usage

---

Classes are the gateway to object-oriented programming in C++. They let you model real-world entities -- a bank account, a sensor, a game character -- as self-contained units that bundle their data with the operations that act on it. Once you internalize how constructors initialize objects, how access specifiers protect invariants, and how destructors clean up resources, you will have the conceptual foundation for virtually every C++ library and framework you encounter.

## 1. What is a Class?

A class is a user-defined type that combines data and functions.

```cpp
#include <iostream>
#include <string>

class Person {
public:
    // Member variables (data)
    std::string name;
    int age;

    // Member function (behavior)
    void introduce() {
        std::cout << "Hello, I am " << name << ". "
                  << "I am " << age << " years old." << std::endl;
    }
};

int main() {
    // Create object
    Person p1;
    p1.name = "Alice";
    p1.age = 25;
    p1.introduce();  // Hello, I am Alice. I am 25 years old.

    Person p2;
    p2.name = "Bob";
    p2.age = 30;
    p2.introduce();

    return 0;
}
```

---

## 2. Access Specifiers

| Specifier | Inside Class | Derived Class | Outside |
|-----------|--------------|---------------|---------|
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
        // All accessible inside class
        std::cout << publicVar << std::endl;
        std::cout << protectedVar << std::endl;
        std::cout << privateVar << std::endl;
    }
};

int main() {
    Example ex;

    std::cout << ex.publicVar << std::endl;  // OK
    // std::cout << ex.protectedVar << std::endl;  // Error!
    // std::cout << ex.privateVar << std::endl;  // Error!

    ex.showAll();  // All accessible internally

    return 0;
}
```

### Encapsulation

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
    std::cout << "After withdrawal: $" << account.getBalance() << std::endl;

    return 0;
}
```

---

## 3. Constructor

A special function automatically called when an object is created.

### Default Constructor

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    // Default constructor
    Person() {
        name = "Unknown";
        age = 0;
        std::cout << "Default constructor called" << std::endl;
    }
};

int main() {
    Person p;  // Default constructor called
    std::cout << p.name << ", " << p.age << std::endl;
    return 0;
}
```

### Parameterized Constructor

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    // Default constructor
    Person() : name("Unknown"), age(0) {}

    // Parameterized constructor
    Person(std::string n, int a) {
        name = n;
        age = a;
    }
};

int main() {
    Person p1;                    // Default constructor
    Person p2("Alice", 25);       // Parameterized constructor
    Person p3 = {"Bob", 30};      // C++11 initialization

    std::cout << p1.name << ", " << p1.age << std::endl;
    std::cout << p2.name << ", " << p2.age << std::endl;
    std::cout << p3.name << ", " << p3.age << std::endl;

    return 0;
}
```

### Member Initializer List

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;
    const int id;  // const member

public:
    // Member initializer list (recommended)
    Person(std::string n, int a, int i)
        : name(n), age(a), id(i)  // Initializer list
    {
        // Body
        std::cout << "Person created: " << name << std::endl;
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

### When to Use Initializer List

1. **const member** initialization
2. **Reference member** initialization
3. Calling parent class constructor
4. Performance (avoid unnecessary default construction)

---

## 4. Destructor

Automatically called when an object is destroyed.

```cpp
#include <iostream>

class Resource {
private:
    int* data;

public:
    Resource(int size) {
        data = new int[size];
        std::cout << "Resource allocated" << std::endl;
    }

    ~Resource() {  // Destructor
        delete[] data;
        std::cout << "Resource freed" << std::endl;
    }
};

int main() {
    {
        Resource r(100);
        // r destroyed at end of block
    }
    std::cout << "After block ends" << std::endl;

    return 0;
}
```

Output:
```
Resource allocated
Resource freed
After block ends
```

---

## 5. this Pointer

A pointer to the current object.

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string name, int age) {
        // Use this to distinguish member from parameter
        this->name = name;
        this->age = age;
    }

    // Return for method chaining
    Person& setName(std::string name) {
        this->name = name;
        return *this;  // Return self
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

    // Method chaining
    p.setName("Bob").setAge(30);
    p.show();  // Bob, 30

    return 0;
}
```

---

## 6. const Member Functions

Functions that do not modify the object.

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string n, int a) : name(n), age(a) {}

    // const member function: Cannot modify object
    std::string getName() const {
        // name = "Other";  // Error!
        return name;
    }

    int getAge() const {
        return age;
    }

    // non-const member function: Can modify object
    void setAge(int a) {
        age = a;
    }
};

void printPerson(const Person& p) {
    // const object can only call const member functions
    std::cout << p.getName() << ", " << p.getAge() << std::endl;
    // p.setAge(30);  // Error!
}

int main() {
    Person p("Alice", 25);
    printPerson(p);
    return 0;
}
```

---

## 7. Classes and Header Files

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
    std::cout << "Hello, I am " << name << "." << std::endl;
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

### Compilation

```bash
g++ -c person.cpp -o person.o
g++ -c main.cpp -o main.o
g++ person.o main.o -o program
```

---

## 8. Struct vs Class

In C++, the only difference between `struct` and `class` is the default access specifier.

```cpp
struct MyStruct {
    int x;  // Default: public
};

class MyClass {
    int x;  // Default: private
};
```

### Convention

```cpp
// struct: Data-centric, few methods
struct Point {
    int x;
    int y;
};

// class: Includes behavior, encapsulation
class Rectangle {
private:
    Point topLeft;
    Point bottomRight;

public:
    int getArea() const;
};
```

---

## 9. Default Constructor Rules

```cpp
#include <iostream>

class A {
public:
    int value;
    // No constructor - compiler generates default constructor
};

class B {
public:
    int value;
    B(int v) : value(v) {}
    // User-defined constructor - no default constructor
};

class C {
public:
    int value;
    C(int v) : value(v) {}
    C() = default;  // Explicitly generate default constructor
};

class D {
public:
    D() = delete;  // Delete default constructor
};

int main() {
    A a;           // OK
    // B b;        // Error! No default constructor
    B b(10);       // OK
    C c;           // OK (default explicitly created)
    C c2(20);      // OK
    // D d;        // Error! Deleted

    return 0;
}
```

---

## 10. Inline Member Functions

```cpp
class Calculator {
public:
    // Defined inside class: Automatically inline
    int add(int a, int b) {
        return a + b;
    }

    // External definition with inline
    int multiply(int a, int b);
};

// Specify inline outside class
inline int Calculator::multiply(int a, int b) {
    return a * b;
}
```

---

## 11. Practice Examples

### Rectangle Class

```cpp
#include <iostream>

class Rectangle {
private:
    double width;
    double height;

public:
    // Constructors
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

    // Functions
    double getArea() const {
        return width * height;
    }

    double getPerimeter() const {
        return 2 * (width + height);
    }

    void display() const {
        std::cout << "Rectangle(" << width << " x " << height << ")" << std::endl;
        std::cout << "  Area: " << getArea() << std::endl;
        std::cout << "  Perimeter: " << getPerimeter() << std::endl;
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

### Student Class

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
        std::cout << "ID: " << id << ", Name: " << name << std::endl;
        std::cout << "Scores: ";
        for (int s : scores) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        std::cout << "Average: " << getAverage() << std::endl;
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

## 12. Summary

| Concept | Description |
|---------|-------------|
| `class` | User-defined type |
| `public` | Accessible from anywhere |
| `private` | Accessible only within class |
| `protected` | Accessible in class and derived classes |
| Constructor | Called when object is created |
| Destructor | Called when object is destroyed |
| `this` | Pointer to current object |
| `const` method | Cannot modify object |

---

## Next Step

Let's learn about operator overloading and copy/move in [08_Classes_Advanced.md](./08_Classes_Advanced.md)!
