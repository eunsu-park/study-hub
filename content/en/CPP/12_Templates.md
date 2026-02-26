# Templates

**Previous**: [STL Algorithms and Iterators](./11_STL_Algorithms_Iterators.md) | **Next**: [Exception Handling and File I/O](./13_Exceptions_and_File_IO.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define function templates and class templates with single and multiple type parameters
2. Distinguish between implicit type deduction and explicit template argument specification
3. Use non-type template parameters to embed compile-time constants into templates
4. Write full and partial specializations to customize behavior for specific types
5. Implement variadic templates and fold expressions for functions accepting arbitrary argument counts
6. Apply SFINAE and `if constexpr` to conditionally enable template instantiations
7. Define and use C++20 concepts to constrain template parameters with clear error messages
8. Organize template code correctly using the header-only compilation model

---

> **Analogy -- The Compiler's Cookie Cutter**: A C++ template is like a cookie cutter: you define the shape once, and the compiler stamps out a concrete version for every type you use (`int`, `double`, `std::string`). The cutter itself isn't a cookie--it's a blueprint the compiler uses at compile time to generate real, type-safe code with zero runtime overhead.

Templates are the foundation of generic programming in C++. Without them, you would need to duplicate every function and class for each data type you want to support--a maintenance nightmare. With templates, the compiler does the duplication for you, producing specialized, type-safe code while you write the logic only once. They underpin the entire STL and form the basis for modern metaprogramming techniques.

## 1. What are Templates?

Templates are a powerful feature of C++ that allows writing type-independent, generic code.

```
┌─────────────────────────────────────────────┐
│           Template                           │
├─────────────────────────────────────────────┤
│  • Code that receives types as parameters    │
│  • Replaced with actual types at compile time│
│  • Maximizes code reusability                │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┬─────────────────┐
│  Function       │  Class          │
│  Template       │  Template       │
└─────────────────┴─────────────────┘
```

### Why Templates?

```cpp
// Without templates, using overloading
int max(int a, int b) { return (a > b) ? a : b; }
double max(double a, double b) { return (a > b) ? a : b; }
char max(char a, char b) { return (a > b) ? a : b; }
// ... Need to repeat for every type

// Solved with a single template
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
```

---

## 2. Function Templates

### Basic Syntax

```cpp
#include <iostream>

// Function template definition
template<typename T>
T add(T a, T b) {
    return a + b;
}

// Can use class instead of typename (same meaning)
template<class T>
T multiply(T a, T b) {
    return a * b;
}

int main() {
    // Explicit type specification
    std::cout << add<int>(3, 5) << std::endl;        // 8
    std::cout << add<double>(3.5, 2.5) << std::endl; // 6

    // Type deduction (compiler automatically determines type)
    std::cout << add(10, 20) << std::endl;           // 30 (int)
    std::cout << add(1.5, 2.5) << std::endl;         // 4 (double)

    std::cout << multiply(4, 5) << std::endl;        // 20

    return 0;
}
```

### Multiple Type Parameters

```cpp
#include <iostream>
#include <string>

template<typename T, typename U>
void printPair(T first, U second) {
    std::cout << first << ", " << second << std::endl;
}

// Return type also as template
template<typename T, typename U>
auto addDifferent(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14: Simple auto return
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

### Non-Type Template Parameters

```cpp
#include <iostream>
#include <array>

// Integer value as template parameter
template<typename T, int Size>
class FixedArray {
private:
    T data[Size];
public:
    T& operator[](int index) { return data[index]; }
    const T& operator[](int index) const { return data[index]; }
    int size() const { return Size; }
};

// Can also use in functions
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

    // Computed at compile time
    std::cout << "5! = " << factorial<5>() << std::endl;  // 120

    return 0;
}
```

---

## 3. Class Templates

### Basic Syntax

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

### Member Function Outside Definition

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

// Need template declaration for outside definition
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

### Multiple Type Parameters

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

### Default Template Arguments

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
    Array<> arr1;  // int, 10 (defaults)
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

## 4. Template Specialization

### Full Specialization

```cpp
#include <iostream>
#include <cstring>

// Base template
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

// Full specialization for char*
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

// Full specialization for bool
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

### Partial Specialization

```cpp
#include <iostream>

// Base template
template<typename T, typename U>
class Pair {
public:
    void info() const {
        std::cout << "General Pair<T, U>" << std::endl;
    }
};

// Partial specialization when both types are same
template<typename T>
class Pair<T, T> {
public:
    void info() const {
        std::cout << "Same type Pair<T, T>" << std::endl;
    }
};

// Partial specialization when second is int
template<typename T>
class Pair<T, int> {
public:
    void info() const {
        std::cout << "Pair<T, int>" << std::endl;
    }
};

// Partial specialization for pointer types
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

### Function Template Specialization

```cpp
#include <iostream>
#include <cstring>

// Base template
template<typename T>
bool isEqual(T a, T b) {
    return a == b;
}

// char* specialization
template<>
bool isEqual<const char*>(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

int main() {
    std::cout << std::boolalpha;

    std::cout << isEqual(10, 10) << std::endl;           // true
    std::cout << isEqual(3.14, 3.14) << std::endl;       // true
    std::cout << isEqual("Hello", "Hello") << std::endl; // true (pointer address comparison)

    const char* s1 = "Hello";
    const char* s2 = "Hello";
    std::cout << isEqual(s1, s2) << std::endl;           // true (string content comparison)

    return 0;
}
```

---

## 5. Variadic Templates

### Basic Syntax

```cpp
#include <iostream>

// Recursive termination (base case)
void print() {
    std::cout << std::endl;
}

// Variadic template
template<typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first;
    if (sizeof...(args) > 0) {
        std::cout << ", ";
    }
    print(args...);  // Recursive call
}

int main() {
    print(1, 2, 3);                    // 1, 2, 3
    print("Hello", 3.14, 42, 'A');     // Hello, 3.14, 42, A
    print("Name:", "Alice", "Age:", 25);  // Name:, Alice, Age:, 25

    return 0;
}
```

### Sum Calculation

```cpp
#include <iostream>

// Recursive termination
template<typename T>
T sum(T value) {
    return value;
}

// Variadic arguments
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

### sizeof... Operator

```cpp
#include <iostream>

template<typename... Args>
void countArgs(Args... args) {
    std::cout << "Argument count: " << sizeof...(Args) << std::endl;
    std::cout << "Argument count: " << sizeof...(args) << std::endl;  // Same result
}

int main() {
    countArgs();                 // Argument count: 0
    countArgs(1);                // Argument count: 1
    countArgs(1, 2, 3);          // Argument count: 3
    countArgs("a", 1, 3.14, 'c'); // Argument count: 4

    return 0;
}
```

### Fold Expressions (C++17)

```cpp
#include <iostream>

// C++17 fold expression (simplified)
template<typename... Args>
auto sumFold(Args... args) {
    return (args + ...);  // Right fold
}

template<typename... Args>
void printFold(Args... args) {
    ((std::cout << args << " "), ...);  // Comma operator fold
    std::cout << std::endl;
}

template<typename... Args>
bool allTrue(Args... args) {
    return (args && ...);  // All true?
}

template<typename... Args>
bool anyTrue(Args... args) {
    return (args || ...);  // Any true?
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

SFINAE (Substitution Failure Is Not An Error): Template argument substitution failure is not an error.

### Basic Concept

```cpp
#include <iostream>
#include <type_traits>

// Enabled only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process(T value) {
    std::cout << "Integer: " << value << std::endl;
}

// Enabled only for floating point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process(T value) {
    std::cout << "Float: " << value << std::endl;
}

int main() {
    process(42);      // Integer: 42
    process(3.14);    // Float: 3.14
    // process("Hi"); // Compile error (neither matches)

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

## 7. Type Traits

### Basic Type Traits

```cpp
#include <iostream>
#include <type_traits>

int main() {
    std::cout << std::boolalpha;

    // Type checking
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

    // Type conversion
    std::cout << "is_same<int, int>: "
              << std::is_same<int, int>::value << std::endl;  // true

    using NoRef = std::remove_reference<int&>::type;
    std::cout << "is_same<NoRef, int>: "
              << std::is_same<NoRef, int>::value << std::endl;  // true

    return 0;
}
```

### Conditional Type Selection

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
    // Using std::conditional
    using Type1 = std::conditional<true, int, double>::type;
    using Type2 = std::conditional<false, int, double>::type;

    std::cout << std::boolalpha;
    std::cout << "Type1 is int: "
              << std::is_same<Type1, int>::value << std::endl;     // true
    std::cout << "Type2 is double: "
              << std::is_same<Type2, double>::value << std::endl;  // true

    // Type selection based on size
    using SmallType = std::conditional<(sizeof(int) > 4), long, int>::type;
    std::cout << "SmallType size: " << sizeof(SmallType) << std::endl;

    return 0;
}
```

---

## 8. Concepts (C++20)

### Basic Syntax

```cpp
#include <iostream>
#include <concepts>

// Concept definition
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

// Using concept
template<Numeric T>
T square(T x) {
    return x * x;
}

// Using requires clause
template<typename T>
requires Addable<T>
T add(T a, T b) {
    return a + b;
}

// Abbreviated form
auto multiply(Numeric auto a, Numeric auto b) {
    return a * b;
}

int main() {
    std::cout << square(5) << std::endl;      // 25
    std::cout << square(3.5) << std::endl;    // 12.25
    // square("Hi");  // Error: Numeric constraint not satisfied

    std::cout << add(10, 20) << std::endl;    // 30
    std::cout << multiply(3, 4) << std::endl; // 12

    return 0;
}
```

### Standard Concepts

```cpp
#include <iostream>
#include <concepts>
#include <string>

// Using standard concepts
template<std::integral T>
void processInt(T value) {
    std::cout << "Integer: " << value << std::endl;
}

template<std::floating_point T>
void processFloat(T value) {
    std::cout << "Float: " << value << std::endl;
}

template<std::convertible_to<std::string> T>
void processString(T value) {
    std::string s = value;
    std::cout << "String: " << s << std::endl;
}

int main() {
    processInt(42);
    processFloat(3.14);
    processString("Hello");

    return 0;
}
```

---

## 9. Practical Template Examples

### Generic Stack

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

### Factory Function

```cpp
#include <iostream>
#include <memory>
#include <string>

// make function template
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

class Person {
public:
    std::string name;
    int age;

    Person(std::string n, int a) : name(n), age(a) {
        std::cout << "Person created: " << name << std::endl;
    }

    void introduce() const {
        std::cout << name << ", " << age << " years old" << std::endl;
    }
};

int main() {
    auto p = make<Person>("Alice", 25);
    p->introduce();  // Alice, 25 years old

    auto nums = make<std::vector<int>>(std::initializer_list<int>{1, 2, 3});
    for (int n : *nums) {
        std::cout << n << " ";  // 1 2 3
    }
    std::cout << std::endl;

    return 0;
}
```

### Type-Safe printf

```cpp
#include <iostream>
#include <sstream>
#include <string>

// Recursive termination
void safePrint(std::ostream& os, const char* format) {
    while (*format) {
        if (*format == '%' && *(format + 1) != '%') {
            throw std::runtime_error("Insufficient arguments");
        }
        if (*format == '%' && *(format + 1) == '%') {
            format++;  // Skip %%
        }
        os << *format++;
    }
}

// Variadic argument handling
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
    throw std::runtime_error("Too many arguments");
}

template<typename... Args>
std::string format(const char* fmt, Args... args) {
    std::ostringstream oss;
    safePrint(oss, fmt, args...);
    return oss.str();
}

int main() {
    std::cout << format("Name: %, Age: % years", "Alice", 25) << std::endl;
    // Name: Alice, Age: 25 years

    std::cout << format("% + % = %", 10, 20, 30) << std::endl;
    // 10 + 20 = 30

    return 0;
}
```

---

## 10. Template Compilation Model

### Why Define in Headers

```
Regular function:             Template:
┌─────────────┐              ┌─────────────┐
│ header.h    │              │ header.h    │
│ Declaration │              │ Decl + Def  │
└─────────────┘              └─────────────┘
       │                            │
       ▼                            ▼
┌─────────────┐              ┌─────────────┐
│ source.cpp  │              │ (Instantiated│
│ Definition  │              │  at use site)│
└─────────────┘              └─────────────┘
```

### Correct Template Structure

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

// Template definitions also in header
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

### Explicit Instantiation (Optional)

```cpp
// mytemplate.cpp
#include "mytemplate.h"

// Explicit instantiation for specific types
template class MyContainer<int>;
template class MyContainer<double>;
template class MyContainer<std::string>;
```

---

## 11. Summary

| Concept | Description |
|---------|-------------|
| Function template | Type-independent function |
| Class template | Type-independent class |
| Template specialization | Special implementation for specific types |
| Partial specialization | Specialization for partial conditions |
| Variadic template | Handling arbitrary number of arguments |
| SFINAE | Substitution failure is not an error |
| Concepts (C++20) | Template constraints |
| Non-type parameter | Value as template argument |

---

## 12. Exercises

### Exercise 1: Min/Max Function

Write a function template that returns the minimum and maximum values from an arbitrary number of arguments.

### Exercise 2: Generic Queue

Referencing the Stack example, write a Queue class template.

### Exercise 3: Type-Specific Serialization

Write a `serialize` function template that converts various types to strings (basic types, containers, etc.).

---

## Next Steps

Let's learn about exception handling and file I/O in [13_Exceptions_and_File_IO.md](./13_Exceptions_and_File_IO.md)!
