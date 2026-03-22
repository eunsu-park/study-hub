# Namespaces and IO Streams

**Previous**: [Pointers and References](./06_Pointers_and_References.md) | **Next**: [Classes Basics](./08_Classes_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Organize code into namespaces and resolve name collisions
2. Perform formatted input and output using iostream and iomanip
3. Parse and build strings using stringstream
4. Read entire lines with `std::getline` and handle mixed input
5. Use the `using` directive and declaration safely

---

As programs grow beyond a handful of files, the risk of name collisions grows with them. Namespaces give you a way to partition identifiers into logical groups so that two libraries can each define a `log()` function without conflict. Equally important is fluent I/O: knowing how to format numbers, align columns, and robustly read user input turns rough prototypes into polished, professional tools.

## 1. Namespaces

A namespace is a declarative region that groups identifiers under a name, preventing collisions.

### Defining a Namespace

```cpp
#include <iostream>

namespace math {
    double PI = 3.14159265358979;

    double circleArea(double r) {
        return PI * r * r;
    }
}

namespace physics {
    double PI = 3.14159;  // No conflict with math::PI

    double sphereVolume(double r) {
        return (4.0 / 3.0) * PI * r * r * r;
    }
}

int main() {
    std::cout << "Circle area: " << math::circleArea(5.0) << std::endl;
    std::cout << "Sphere volume: " << physics::sphereVolume(5.0) << std::endl;
    return 0;
}
```

### Nested Namespaces

```cpp
// Traditional syntax
namespace company {
    namespace project {
        void init() { /* ... */ }
    }
}

// C++17 shorthand
namespace company::project {
    void shutdown() { /* ... */ }
}

int main() {
    company::project::init();
    company::project::shutdown();
    return 0;
}
```

### Anonymous (Unnamed) Namespaces

Identifiers in an anonymous namespace have internal linkage -- visible only within the current translation unit, similar to `static` at file scope.

```cpp
namespace {
    int helperCounter = 0;  // Only visible in this .cpp file

    void increment() {
        helperCounter++;
    }
}
```

### using Directive vs using Declaration

```cpp
#include <iostream>
#include <string>

// using declaration: imports one name
using std::cout;
using std::endl;

int main() {
    cout << "Hello" << endl;  // OK

    // std::string still requires std:: unless also declared
    std::string name = "World";
    cout << name << endl;

    return 0;
}
```

```cpp
// using directive: imports ALL names from a namespace
using namespace std;  // Pulls in everything from std

int main() {
    cout << "Hello" << endl;     // OK
    string name = "World";       // OK
    return 0;
}
```

### Safety Comparison

| Approach | Scope | Risk |
|----------|-------|------|
| `std::cout` (fully qualified) | None | Safest |
| `using std::cout;` (declaration) | Imports one name | Low risk |
| `using namespace std;` (directive) | Imports all names | Higher risk of collisions |

**Rule of thumb**: Never put `using namespace` in a header file. In source files, prefer `using` declarations for the specific names you need.

---

## 2. Standard Output

### std::cout and the Insertion Operator

```cpp
#include <iostream>

int main() {
    // Chain multiple insertions
    std::cout << "Name: " << "Alice" << ", Age: " << 30 << std::endl;

    // '\n' vs std::endl
    std::cout << "Line 1\n";          // Newline only (faster)
    std::cout << "Line 2" << std::endl; // Newline + flush buffer

    // Explicit flush without newline
    std::cout << "Processing..." << std::flush;

    return 0;
}
```

### When to Use endl vs '\n'

| Method | Effect | Use When |
|--------|--------|----------|
| `'\n'` | Inserts newline | Default choice (faster) |
| `std::endl` | Inserts newline + flushes buffer | Need guaranteed output (debugging, logging) |
| `std::flush` | Flushes buffer only | Progress indicators |

For high-throughput output (e.g., printing millions of lines), prefer `'\n'` to avoid the overhead of flushing on every line.

---

## 3. Standard Input

### std::cin and the Extraction Operator

```cpp
#include <iostream>

int main() {
    int age;
    double height;

    std::cout << "Enter age and height: ";
    std::cin >> age >> height;  // Reads two values separated by whitespace

    std::cout << "Age: " << age << ", Height: " << height << std::endl;

    return 0;
}
```

### Input Failure and Recovery

When `std::cin` encounters data that does not match the expected type, it enters a fail state.

```cpp
#include <iostream>
#include <limits>

int main() {
    int number;

    while (true) {
        std::cout << "Enter an integer: ";
        if (std::cin >> number) {
            break;  // Success
        }

        // Input failed (e.g., user typed "abc")
        std::cout << "Invalid input. Try again.\n";
        std::cin.clear();  // Clear the fail flag
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard bad input
    }

    std::cout << "You entered: " << number << std::endl;
    return 0;
}
```

### Key Recovery Functions

| Function | Purpose |
|----------|---------|
| `cin.clear()` | Reset error flags |
| `cin.ignore(n, delim)` | Discard up to `n` characters or until `delim` |
| `cin.fail()` | Returns `true` if last extraction failed |
| `cin.good()` | Returns `true` if stream is in a good state |

---

## 4. Formatted Output with iomanip

The `<iomanip>` header provides manipulators that control how values are printed.

### Width and Fill

```cpp
#include <iostream>
#include <iomanip>

int main() {
    // setw: minimum field width (applies to next output only)
    std::cout << std::setw(10) << 42 << std::endl;       // "        42"
    std::cout << std::setw(10) << "Hello" << std::endl;   // "     Hello"

    // setfill: character used for padding
    std::cout << std::setfill('0') << std::setw(5) << 42 << std::endl;  // "00042"
    std::cout << std::setfill('.') << std::setw(20) << "Menu" << std::endl;
    // "................Menu"

    return 0;
}
```

### Alignment

```cpp
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::left  << std::setw(15) << "Name"
              << std::right << std::setw(10) << "Score" << std::endl;
    std::cout << std::left  << std::setw(15) << "Alice"
              << std::right << std::setw(10) << 95 << std::endl;
    std::cout << std::left  << std::setw(15) << "Bob"
              << std::right << std::setw(10) << 87 << std::endl;
    std::cout << std::left  << std::setw(15) << "Charlie"
              << std::right << std::setw(10) << 92 << std::endl;

    return 0;
}
```

Output:
```
Name                 Score
Alice                   95
Bob                     87
Charlie                 92
```

### Floating-Point Precision

```cpp
#include <iostream>
#include <iomanip>

int main() {
    double pi = 3.14159265358979;

    // Default precision (6 significant digits)
    std::cout << pi << std::endl;                  // 3.14159

    // setprecision: number of significant digits
    std::cout << std::setprecision(3) << pi << std::endl;  // 3.14

    // fixed: digits after decimal point
    std::cout << std::fixed << std::setprecision(2) << pi << std::endl;  // 3.14

    // scientific notation
    std::cout << std::scientific << std::setprecision(4) << pi << std::endl;
    // 3.1416e+00

    // Reset to default
    std::cout << std::defaultfloat;

    return 0;
}
```

### Number Base and Boolean Format

```cpp
#include <iostream>
#include <iomanip>

int main() {
    int num = 255;

    std::cout << "Decimal:     " << std::dec << num << std::endl;  // 255
    std::cout << "Hexadecimal: " << std::hex << num << std::endl;  // ff
    std::cout << "Octal:       " << std::oct << num << std::endl;  // 377

    // Show base prefix
    std::cout << std::showbase;
    std::cout << "Hex: " << std::hex << num << std::endl;  // 0xff
    std::cout << "Oct: " << std::oct << num << std::endl;  // 0377
    std::cout << std::noshowbase << std::dec;  // Reset

    // Boolean output
    bool flag = true;
    std::cout << flag << std::endl;                    // 1
    std::cout << std::boolalpha << flag << std::endl;  // true
    std::cout << std::noboolalpha;  // Reset

    return 0;
}
```

### Manipulator Summary

| Manipulator | Header | Effect | Sticky? |
|-------------|--------|--------|---------|
| `setw(n)` | `<iomanip>` | Minimum field width | No (next output only) |
| `setfill(c)` | `<iomanip>` | Padding character | Yes |
| `setprecision(n)` | `<iomanip>` | Digit precision | Yes |
| `fixed` | `<iostream>` | Fixed-point notation | Yes |
| `scientific` | `<iostream>` | Scientific notation | Yes |
| `left` / `right` | `<iostream>` | Alignment | Yes |
| `dec` / `hex` / `oct` | `<iostream>` | Number base | Yes |
| `boolalpha` | `<iostream>` | Print true/false | Yes |
| `showbase` | `<iostream>` | Show 0x or 0 prefix | Yes |

---

## 5. String Streams

`<sstream>` provides stream classes that operate on `std::string` objects, letting you parse strings as if they were input streams or build strings as if they were output streams.

### Parsing with istringstream

```cpp
#include <iostream>
#include <sstream>
#include <string>

int main() {
    std::string data = "Alice 90 85 92";

    std::istringstream iss(data);
    std::string name;
    int s1, s2, s3;

    iss >> name >> s1 >> s2 >> s3;

    std::cout << name << "'s average: "
              << (s1 + s2 + s3) / 3.0 << std::endl;
    // Alice's average: 89

    return 0;
}
```

### Building with ostringstream

```cpp
#include <iostream>
#include <sstream>
#include <iomanip>

int main() {
    std::ostringstream oss;

    oss << "Total: $" << std::fixed << std::setprecision(2) << 1234.5;
    std::string result = oss.str();

    std::cout << result << std::endl;  // Total: $1234.50

    return 0;
}
```

### Splitting a Delimited String

```cpp
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(s);
    std::string token;

    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main() {
    auto parts = split("one:two:three:four", ':');
    for (const auto& p : parts) {
        std::cout << "[" << p << "] ";
    }
    std::cout << std::endl;
    // [one] [two] [three] [four]

    return 0;
}
```

---

## 6. getline

### Reading Full Lines

```cpp
#include <iostream>
#include <string>

int main() {
    std::string line;

    std::cout << "Enter a sentence: ";
    std::getline(std::cin, line);

    std::cout << "You said: " << line << std::endl;

    return 0;
}
```

### Mixing cin >> and getline

A common pitfall: after `cin >>`, a trailing newline remains in the buffer. The next `getline` reads an empty string.

```cpp
#include <iostream>
#include <string>

int main() {
    int age;
    std::string name;

    std::cout << "Enter age: ";
    std::cin >> age;

    // WRONG: getline reads the leftover '\n'
    // std::getline(std::cin, name);  // Gets empty string!

    // FIX: discard the leftover newline first
    std::cin.ignore();  // or std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << "Enter full name: ";
    std::getline(std::cin, name);

    std::cout << "Name: " << name << ", Age: " << age << std::endl;

    return 0;
}
```

### Custom Delimiter

```cpp
#include <iostream>
#include <string>

int main() {
    std::string field;

    // Read until '|' instead of newline
    std::cout << "Enter pipe-delimited data: ";
    while (std::getline(std::cin, field, '|')) {
        std::cout << "  Field: [" << field << "]\n";
        if (std::cin.peek() == '\n') break;  // Stop at end of line
    }

    return 0;
}
```

---

## 7. Error Streams

C++ provides three standard output streams.

| Stream | Purpose | Buffered? |
|--------|---------|-----------|
| `std::cout` | Normal output | Yes |
| `std::cerr` | Error messages | No (immediate) |
| `std::clog` | Diagnostic/log messages | Yes |

```cpp
#include <iostream>

int main() {
    std::cout << "Normal output" << std::endl;
    std::cerr << "Error: something went wrong" << std::endl;
    std::clog << "Log: operation completed" << std::endl;

    return 0;
}
```

### Redirecting Output

```bash
# Redirect stdout only (errors still appear on screen)
./program > output.txt

# Redirect stderr only
./program 2> errors.txt

# Redirect both separately
./program > output.txt 2> errors.txt

# Redirect both to the same file
./program > all.txt 2>&1
```

---

## 8. Best Practices

### Avoid using namespace std in Headers

```cpp
// BAD: header.h
#pragma once
using namespace std;  // Pollutes every file that includes this header

// GOOD: header.h
#pragma once
#include <string>
std::string formatName(const std::string& first, const std::string& last);
```

### Namespace Aliases

When fully qualified names are long, create a short alias.

```cpp
namespace fs = std::filesystem;  // C++17
namespace chrono = std::chrono;

// Now use the alias
auto start = chrono::steady_clock::now();
```

### Argument-Dependent Lookup (ADL)

ADL allows the compiler to find functions in the namespace of their arguments without explicit qualification.

```cpp
#include <iostream>
#include <string>

namespace geometry {
    struct Point { double x, y; };

    // Found via ADL when called with a Point argument
    std::ostream& operator<<(std::ostream& os, const Point& p) {
        return os << "(" << p.x << ", " << p.y << ")";
    }
}

int main() {
    geometry::Point p{3.0, 4.0};
    std::cout << p << std::endl;  // ADL finds geometry::operator<<
    return 0;
}
```

### Input Robustness Checklist

1. Always check whether extraction succeeded (`if (std::cin >> x)`)
2. Use `std::cin.ignore()` after `>>` before `getline`
3. Use `std::cin.clear()` + `std::cin.ignore(...)` to recover from bad input
4. Prefer `getline` + `istringstream` for complex parsing

---

## 9. Summary

| Concept | Key Points |
|---------|------------|
| Namespace | Groups identifiers to prevent collisions |
| Nested namespace | `namespace A::B { }` (C++17) |
| Anonymous namespace | Internal linkage (file-local) |
| `using` declaration | Imports one name |
| `using` directive | Imports all names (use cautiously) |
| `std::cout` | Buffered standard output |
| `std::cin` | Standard input with `>>` extraction |
| `std::cerr` | Unbuffered error output |
| `<iomanip>` | Formatted output (width, precision, fill) |
| `std::istringstream` | Parse strings as input |
| `std::ostringstream` | Build strings as output |
| `std::getline` | Read full lines (custom delimiter optional) |
| Namespace alias | `namespace fs = std::filesystem;` |

---

## Exercises

### Exercise 1: Formatted Table

Write a program that prints a multiplication table (1-5) x (1-5) with each cell right-aligned in a field of width 5. Use `setw` for alignment.

### Exercise 2: Receipt Formatter

Using `ostringstream` and `iomanip`, build a receipt string with item names left-aligned (20 chars), quantities right-aligned (5 chars), and prices right-aligned (10 chars, 2 decimal places). Print the total at the bottom.

### Exercise 3: Robust Input Loop

Write a program that repeatedly asks the user for a `double` until a valid number is entered. Handle non-numeric input gracefully using `cin.clear()` and `cin.ignore()`. After a valid entry, print the value with exactly 4 decimal places.

### Exercise 4: Namespace Collision Resolution

Create two namespaces (`audio` and `video`) that each define a `void play()` function printing a different message. In `main`, call both functions using fully qualified names, then use a `using` declaration for one and call it unqualified.

### Exercise 5: CSV Line Parser

Write a function `std::vector<std::string> parseCSV(const std::string& line)` that uses `istringstream` and `getline` with `','` as the delimiter. Test it with `"Alice,30,Engineering,95.5"` and print each field on its own line.

---

## Next Steps

Let's learn about classes in [08_Classes_Basics.md](./08_Classes_Basics.md)!
