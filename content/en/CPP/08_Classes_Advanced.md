# Advanced Classes

**Previous**: [Class Basics](./07_Classes_Basics.md) | **Next**: [Inheritance and Polymorphism](./09_Inheritance_and_Polymorphism.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement operator overloading for arithmetic, comparison, compound assignment, and stream operators
2. Distinguish between member-function and friend-function operator overloads
3. Explain the difference between shallow copy and deep copy and implement a deep-copy constructor
4. Design a correct copy assignment operator with self-assignment protection
5. Apply move semantics (move constructor and move assignment operator) to eliminate unnecessary copies
6. Identify the Rule of Five and explain when a class must define all five special member functions
7. Implement `static` member variables and functions for class-wide shared state
8. Apply `explicit` to prevent unintended implicit conversions

---

Once you can create classes, the next question is how they should *behave* when copied, moved, compared, or printed. Advanced class mechanics -- operator overloading, copy/move semantics, and the Rule of Five -- are what make user-defined types feel as natural as built-in types. These techniques are not academic curiosities; they underpin the entire Standard Library, from `std::string` to `std::vector`, and mastering them is essential for writing professional-quality C++.

## 1. Operator Overloading

You can define how operators work with your classes.

### Basic Syntax

```cpp
return_type operator symbol(parameters) {
    // implementation
}
```

### Arithmetic Operator Overloading

```cpp
#include <iostream>

class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // + operator (member function)
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    // - operator
    Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }

    // * operator (scalar multiplication)
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

    Vector2D v3 = v1 + v2;  // operator+ called
    v3.print();  // (4, 6)

    Vector2D v4 = v1 - v2;
    v4.print();  // (2, 2)

    Vector2D v5 = v1 * 2;
    v5.print();  // (6, 8)

    return 0;
}
```

### Comparison Operator Overloading

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(n), age(a) {}

    // == operator
    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }

    // != operator
    bool operator!=(const Person& other) const {
        return !(*this == other);
    }

    // < operator (by age)
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

### Compound Assignment Operators

```cpp
class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // += operator
    Vector2D& operator+=(const Vector2D& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    // -= operator
    Vector2D& operator-=(const Vector2D& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
};
```

### Increment/Decrement Operators

```cpp
#include <iostream>

class Counter {
private:
    int value;

public:
    Counter(int v = 0) : value(v) {}

    // Prefix increment (++c)
    Counter& operator++() {
        ++value;
        return *this;
    }

    // Postfix increment (c++)
    Counter operator++(int) {  // int is a dummy to distinguish
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

### Stream Operators (friend)

```cpp
#include <iostream>

class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // << operator (friend function)
    friend std::ostream& operator<<(std::ostream& os, const Vector2D& v) {
        os << "(" << v.x << ", " << v.y << ")";
        return os;
    }

    // >> operator
    friend std::istream& operator>>(std::istream& is, Vector2D& v) {
        is >> v.x >> v.y;
        return is;
    }
};

int main() {
    Vector2D v(3, 4);
    std::cout << "Vector: " << v << std::endl;  // Vector: (3, 4)

    Vector2D v2;
    std::cout << "Enter x y: ";
    std::cin >> v2;
    std::cout << "Input: " << v2 << std::endl;

    return 0;
}
```

### Function Call Operator ()

```cpp
#include <iostream>

class Adder {
private:
    int base;

public:
    Adder(int b) : base(b) {}

    // () operator - can be called like a function
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

### Subscript Operator []

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

    // [] operator (read/write)
    int& operator[](int index) {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // const version (read-only)
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

    // arr[10] = 100;  // Exception thrown!

    return 0;
}
```

---

## 2. Copy Constructor

Called when an object is copied.

### Basic Copy

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(n), age(a) {
        std::cout << "Regular constructor" << std::endl;
    }

    // Copy constructor
    Person(const Person& other) : name(other.name), age(other.age) {
        std::cout << "Copy constructor" << std::endl;
    }
};

int main() {
    Person p1("Alice", 25);    // Regular constructor
    Person p2(p1);             // Copy constructor
    Person p3 = p1;            // Copy constructor

    return 0;
}
```

### Shallow Copy vs Deep Copy

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

    // Deep copy constructor
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];  // Allocate new memory
        strcpy(data, other.data);     // Copy contents
        std::cout << "Deep copy" << std::endl;
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
    String s2 = s1;  // Deep copy

    s1.print();  // Hello
    s2.print();  // Hello

    return 0;
}
```

---

## 3. Copy Assignment Operator

Called when assigning to an existing object.

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

    // Copy assignment operator
    String& operator=(const String& other) {
        if (this != &other) {  // Self-assignment check
            delete[] data;     // Free existing memory

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

    s2 = s1;  // Copy assignment operator

    s1.print();  // Hello
    s2.print();  // Hello

    return 0;
}
```

---

## 4. Move Semantics (C++11)

"Move" resources from temporary objects to avoid unnecessary copies.

### Move Constructor

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
        std::cout << "Regular constructor" << std::endl;
    }

    // Copy constructor
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
        std::cout << "Copy constructor" << std::endl;
    }

    // Move constructor
    String(String&& other) noexcept {
        data = other.data;      // Just copy pointer
        length = other.length;
        other.data = nullptr;   // Invalidate original
        other.length = 0;
        std::cout << "Move constructor" << std::endl;
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
    String s1("Hello");           // Regular constructor
    String s2 = s1;               // Copy constructor
    String s3 = std::move(s1);    // Move constructor
    // s1 is now empty

    s1.print();  // (empty) - moved
    s2.print();  // Hello
    s3.print();  // Hello

    return 0;
}
```

### Move Assignment Operator

```cpp
// Move assignment operator
String& operator=(String&& other) noexcept {
    if (this != &other) {
        delete[] data;          // Free existing memory

        data = other.data;      // Move pointer
        length = other.length;

        other.data = nullptr;   // Invalidate original
        other.length = 0;
    }
    std::cout << "Move assignment" << std::endl;
    return *this;
}
```

### Rule of Five

Classes that manage resources should define all 5:

1. Destructor
2. Copy constructor
3. Copy assignment operator
4. Move constructor
5. Move assignment operator

```cpp
class Resource {
public:
    Resource();                                    // Constructor
    ~Resource();                                   // 1. Destructor
    Resource(const Resource& other);              // 2. Copy constructor
    Resource& operator=(const Resource& other);   // 3. Copy assignment
    Resource(Resource&& other) noexcept;          // 4. Move constructor
    Resource& operator=(Resource&& other) noexcept; // 5. Move assignment
};
```

---

## 5. static Members

Members shared by all objects of a class.

### static Member Variables

```cpp
#include <iostream>

class Counter {
private:
    static int count;  // Declaration

public:
    Counter() {
        count++;
    }

    ~Counter() {
        count--;
    }

    static int getCount() {  // static member function
        return count;
    }
};

// Definition (outside class)
int Counter::count = 0;

int main() {
    std::cout << "Count: " << Counter::getCount() << std::endl;  // 0

    Counter c1;
    Counter c2;
    std::cout << "Count: " << Counter::getCount() << std::endl;  // 2

    {
        Counter c3;
        std::cout << "Count: " << Counter::getCount() << std::endl;  // 3
    }

    std::cout << "Count: " << Counter::getCount() << std::endl;  // 2

    return 0;
}
```

### static Member Functions

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
    // Call without object
    std::cout << Math::add(3, 5) << std::endl;       // 8
    std::cout << Math::multiply(3, 5) << std::endl;  // 15
    std::cout << Math::PI << std::endl;              // 3.14159

    return 0;
}
```

---

## 6. friend

External functions or classes that can access private members.

### friend Function

```cpp
#include <iostream>

class Box {
private:
    double width;

public:
    Box(double w) : width(w) {}

    // friend function declaration
    friend void printWidth(const Box& b);
    friend double addWidths(const Box& a, const Box& b);
};

// friend function definition
void printWidth(const Box& b) {
    std::cout << "Width: " << b.width << std::endl;  // Can access private
}

double addWidths(const Box& a, const Box& b) {
    return a.width + b.width;
}

int main() {
    Box b1(10), b2(20);

    printWidth(b1);  // Width: 10
    std::cout << "Sum: " << addWidths(b1, b2) << std::endl;  // Sum: 30

    return 0;
}
```

### friend Class

```cpp
#include <iostream>

class Engine {
private:
    int horsepower;

public:
    Engine(int hp) : horsepower(hp) {}

    friend class Car;  // Car can access Engine's private members
};

class Car {
private:
    Engine engine;

public:
    Car(int hp) : engine(hp) {}

    void showHorsepower() const {
        std::cout << "Horsepower: " << engine.horsepower << std::endl;
    }
};

int main() {
    Car car(300);
    car.showHorsepower();  // Horsepower: 300

    return 0;
}
```

---

## 7. explicit

Prevents implicit conversions.

```cpp
#include <iostream>

class Fraction {
private:
    int numerator;
    int denominator;

public:
    // Without explicit, Fraction f = 5; would work
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

    Fraction f2(5);  // Explicit call OK
    f2.print();  // 5/1

    // Fraction f3 = 5;  // Error! explicit
    // printFraction(10);  // Error! No implicit conversion

    printFraction(Fraction(10));  // OK: Explicit conversion

    return 0;
}
```

---

## 8. Practice Example: Complete String Class

```cpp
#include <iostream>
#include <cstring>
#include <utility>

class String {
private:
    char* data;
    size_t length;

public:
    // Default constructor
    String() : data(nullptr), length(0) {
        data = new char[1];
        data[0] = '\0';
    }

    // String constructor
    String(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }

    // Copy constructor
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
    }

    // Move constructor
    String(String&& other) noexcept
        : data(other.data), length(other.length) {
        other.data = nullptr;
        other.length = 0;
    }

    // Destructor
    ~String() {
        delete[] data;
    }

    // Copy assignment
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        return *this;
    }

    // Move assignment
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

    // + operator
    String operator+(const String& other) const {
        char* newData = new char[length + other.length + 1];
        strcpy(newData, data);
        strcat(newData, other.data);
        String result(newData);
        delete[] newData;
        return result;
    }

    // == operator
    bool operator==(const String& other) const {
        return strcmp(data, other.data) == 0;
    }

    // [] operator
    char& operator[](size_t index) {
        return data[index];
    }

    const char& operator[](size_t index) const {
        return data[index];
    }

    // << operator
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
    std::cout << "Length: " << s3.size() << std::endl;  // 11

    return 0;
}
```

---

## 9. Summary

| Concept | Description |
|---------|-------------|
| Operator overloading | Define operators for classes |
| Copy constructor | `T(const T&)` |
| Copy assignment | `T& operator=(const T&)` |
| Move constructor | `T(T&&)` |
| Move assignment | `T& operator=(T&&)` |
| `static` | Shared class member |
| `friend` | Allow private access |
| `explicit` | Prevent implicit conversion |

---

## Next Step

Let's learn about inheritance and polymorphism in [09_Inheritance_and_Polymorphism.md](./09_Inheritance_and_Polymorphism.md)!
