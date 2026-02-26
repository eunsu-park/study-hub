# Variables and Types

**Previous**: [Environment Setup and First Program](./01_Environment_Setup.md) | **Next**: [Operators and Control Flow](./03_Operators_and_Control_Flow.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a variable is and how it maps to memory
2. Distinguish between integer, floating-point, character, and boolean data types
3. Compare copy initialization, direct initialization, and brace initialization (C++11)
4. Apply `const` and `constexpr` to declare compile-time and runtime constants
5. Implement automatic type deduction using the `auto` keyword
6. Identify narrowing conversions and use `static_cast` for safe explicit casting
7. Apply `sizeof` to inspect the memory footprint of types
8. Design type aliases with `using` and choose appropriate fixed-width integer types

---

Variables are the building blocks of every program -- they give names to the data your code manipulates. Understanding how C++ represents integers, floating-point numbers, characters, and booleans at the hardware level is what separates someone who merely writes code from someone who writes *efficient* code. Mastering types early also prevents an entire category of subtle bugs caused by overflow, precision loss, and unintended conversions.

## 1. What are Variables?

Variables are named memory locations that store data.

```cpp
#include <iostream>

int main() {
    int age = 25;           // Integer variable
    double height = 175.5;  // Floating-point variable
    char grade = 'A';       // Character variable

    std::cout << "Age: " << age << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Grade: " << grade << std::endl;

    return 0;
}
```

---

## 2. Basic Data Types

### Integer Types

| Type | Size | Range |
|------|------|-------|
| `short` | 2 bytes | -32,768 ~ 32,767 |
| `int` | 4 bytes | Approx -2.1B ~ 2.1B |
| `long` | 4/8 bytes | System dependent |
| `long long` | 8 bytes | Approx -9.2 quintillion ~ 9.2 quintillion |

```cpp
#include <iostream>

int main() {
    short s = 32767;
    int i = 2147483647;
    long l = 2147483647L;
    long long ll = 9223372036854775807LL;

    std::cout << "short: " << s << std::endl;
    std::cout << "int: " << i << std::endl;
    std::cout << "long: " << l << std::endl;
    std::cout << "long long: " << ll << std::endl;

    return 0;
}
```

### Unsigned Integers

```cpp
unsigned int positive = 4294967295;  // 0 ~ approx 4.2B
unsigned short us = 65535;           // 0 ~ 65535
```

### Floating-Point Types

| Type | Size | Precision |
|------|------|-----------|
| `float` | 4 bytes | ~7 digits |
| `double` | 8 bytes | ~15 digits |
| `long double` | 8~16 bytes | System dependent |

```cpp
#include <iostream>
#include <iomanip>  // setprecision

int main() {
    float f = 3.14159265358979f;
    double d = 3.14159265358979;

    std::cout << std::setprecision(15);
    std::cout << "float: " << f << std::endl;
    std::cout << "double: " << d << std::endl;

    return 0;
}
```

Output:
```
float: 3.14159274101257
double: 3.14159265358979
```

### Character Type

```cpp
#include <iostream>

int main() {
    char letter = 'A';
    char newline = '\n';
    char tab = '\t';

    std::cout << "Character: " << letter << std::endl;
    std::cout << "ASCII value: " << (int)letter << std::endl;  // 65

    // Escape sequences
    std::cout << "Tab:\tAfter tab" << std::endl;
    std::cout << "Quote: \"Hello\"" << std::endl;

    return 0;
}
```

### Escape Sequences

| Sequence | Meaning |
|----------|---------|
| `\n` | Newline |
| `\t` | Tab |
| `\\` | Backslash |
| `\"` | Double quote |
| `\'` | Single quote |

### Boolean Type

```cpp
#include <iostream>

int main() {
    bool isTrue = true;
    bool isFalse = false;

    std::cout << "true: " << isTrue << std::endl;   // 1
    std::cout << "false: " << isFalse << std::endl; // 0

    // Conditional expression
    bool result = (5 > 3);  // true
    std::cout << "5 > 3: " << result << std::endl;

    return 0;
}
```

---

## 3. Variable Declaration and Initialization

### Declaration and Initialization Methods

```cpp
#include <iostream>

int main() {
    // Declaration only (uninitialized - garbage value)
    int a;

    // Declaration with initialization
    int b = 10;

    // Brace initialization (C++11, recommended)
    int c{20};

    // Copy initialization
    int d = {30};

    // Multiple variable declaration
    int x = 1, y = 2, z = 3;

    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    std::cout << "d: " << d << std::endl;

    return 0;
}
```

### Advantages of Brace Initialization

```cpp
int a = 3.14;   // OK (truncated to 3, may not warn)
int b{3.14};    // Compile error! (prevents narrowing conversion)
int c{3};       // Exact value
```

---

## 4. Constants

### const Constants

```cpp
#include <iostream>

int main() {
    const int MAX_SIZE = 100;
    const double PI = 3.14159;

    std::cout << "MAX_SIZE: " << MAX_SIZE << std::endl;
    std::cout << "PI: " << PI << std::endl;

    // MAX_SIZE = 200;  // Error! const cannot be modified

    return 0;
}
```

### constexpr (Compile-time Constants)

```cpp
#include <iostream>

constexpr int square(int x) {
    return x * x;
}

int main() {
    constexpr int SIZE = 10;
    constexpr int AREA = square(5);  // Computed at compile time

    int arr[SIZE];  // Can be used as array size

    std::cout << "SIZE: " << SIZE << std::endl;
    std::cout << "AREA: " << AREA << std::endl;

    return 0;
}
```

### const vs constexpr

| Feature | const | constexpr |
|---------|-------|-----------|
| Initialization time | Runtime allowed | Compile-time required |
| Array size | Some compilers only | Always allowed |
| Function application | Not possible | Possible |

---

## 5. auto Keyword (C++11)

The compiler automatically deduces the type.

```cpp
#include <iostream>

int main() {
    auto i = 42;        // int
    auto d = 3.14;      // double
    auto c = 'A';       // char
    auto b = true;      // bool
    auto s = "Hello";   // const char*

    std::cout << "i type: int, value: " << i << std::endl;
    std::cout << "d type: double, value: " << d << std::endl;

    // Type checking (for debugging)
    // Can use typeid(i).name()

    return 0;
}
```

### auto Usage Notes

```cpp
auto x = 10;       // int (literal default)
auto y = 10.0;     // double
auto z = 10.0f;    // float (f suffix)
auto ll = 10LL;    // long long
```

---

## 6. Type Casting

### Implicit Conversion (Automatic)

```cpp
#include <iostream>

int main() {
    int i = 10;
    double d = i;  // int → double (safe)

    double pi = 3.14;
    int truncated = pi;  // double → int (decimal loss!)

    std::cout << "d: " << d << std::endl;         // 10
    std::cout << "truncated: " << truncated << std::endl;  // 3

    return 0;
}
```

### Explicit Conversion

```cpp
#include <iostream>

int main() {
    double pi = 3.14159;

    // C style (not recommended)
    int a = (int)pi;

    // C++ function style
    int b = int(pi);

    // static_cast (recommended)
    int c = static_cast<int>(pi);

    std::cout << "a: " << a << std::endl;  // 3
    std::cout << "b: " << b << std::endl;  // 3
    std::cout << "c: " << c << std::endl;  // 3

    return 0;
}
```

### C++ Cast Operators

| Cast | Purpose |
|------|---------|
| `static_cast<T>` | General type conversion |
| `const_cast<T>` | Add/remove const |
| `dynamic_cast<T>` | Polymorphic class conversion |
| `reinterpret_cast<T>` | Bit-level reinterpretation |

---

## 7. Size Check: sizeof

```cpp
#include <iostream>

int main() {
    std::cout << "char: " << sizeof(char) << " bytes" << std::endl;
    std::cout << "short: " << sizeof(short) << " bytes" << std::endl;
    std::cout << "int: " << sizeof(int) << " bytes" << std::endl;
    std::cout << "long: " << sizeof(long) << " bytes" << std::endl;
    std::cout << "long long: " << sizeof(long long) << " bytes" << std::endl;
    std::cout << "float: " << sizeof(float) << " bytes" << std::endl;
    std::cout << "double: " << sizeof(double) << " bytes" << std::endl;
    std::cout << "bool: " << sizeof(bool) << " bytes" << std::endl;

    int arr[10];
    std::cout << "int[10]: " << sizeof(arr) << " bytes" << std::endl;

    return 0;
}
```

---

## 8. Literals

### Integer Literals

```cpp
int decimal = 42;       // Decimal
int octal = 052;        // Octal (starts with 0)
int hex = 0x2A;         // Hexadecimal (starts with 0x)
int binary = 0b101010;  // Binary (C++14, starts with 0b)

long l = 42L;
unsigned u = 42U;
long long ll = 42LL;
unsigned long long ull = 42ULL;
```

### Floating-Point Literals

```cpp
double d1 = 3.14;
double d2 = 3.14e2;    // 314.0 (scientific notation)
double d3 = 3.14e-2;   // 0.0314

float f = 3.14f;       // float (f suffix)
long double ld = 3.14L; // long double (L suffix)
```

### Digit Separators (C++14)

```cpp
int million = 1'000'000;        // Improved readability
long long big = 1'234'567'890LL;
double pi = 3.141'592'653;
```

---

## 9. Type Aliases

### typedef (Traditional Method)

```cpp
typedef unsigned int uint;
typedef long long int64;

uint a = 100;
int64 b = 1234567890123LL;
```

### using (C++11, Recommended)

```cpp
using uint = unsigned int;
using int64 = long long;

uint a = 100;
int64 b = 1234567890123LL;
```

---

## 10. Standard Fixed-Width Types

Platform-independent types defined in `<cstdint>` header.

```cpp
#include <iostream>
#include <cstdint>

int main() {
    int8_t a = 127;          // Exactly 8 bits
    int16_t b = 32767;       // Exactly 16 bits
    int32_t c = 2147483647;  // Exactly 32 bits
    int64_t d = 9223372036854775807LL;  // Exactly 64 bits

    uint8_t ua = 255;        // unsigned 8 bits
    uint16_t ub = 65535;     // unsigned 16 bits

    std::cout << "int8_t max: " << (int)a << std::endl;
    std::cout << "int16_t max: " << b << std::endl;
    std::cout << "int32_t max: " << c << std::endl;
    std::cout << "int64_t max: " << d << std::endl;

    return 0;
}
```

---

## 11. Summary

| Category | Type | Size |
|----------|------|------|
| Integer | `int` | 4 bytes |
| Integer | `long long` | 8 bytes |
| Float | `double` | 8 bytes |
| Character | `char` | 1 byte |
| Boolean | `bool` | 1 byte |

| Keyword | Purpose |
|---------|---------|
| `const` | Runtime constant |
| `constexpr` | Compile-time constant |
| `auto` | Automatic type deduction |
| `static_cast` | Safe type conversion |

---

## 12. Practice Exercises

### Exercise 1: Variable Output

Declare variables of various types and print their values.

### Exercise 2: Type Conversion

Write a program that takes Celsius temperature as input and converts it to Fahrenheit. (F = C × 9/5 + 32)

### Exercise 3: sizeof

Write a program that prints the size of all basic types.

---

## Next Steps

Let's learn about operators and control flow in [03_Operators_and_Control_Flow.md](./03_Operators_and_Control_Flow.md)!
