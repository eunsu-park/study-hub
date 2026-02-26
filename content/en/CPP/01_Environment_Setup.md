# Environment Setup and First Program

**Previous**: [C++ Learning Guide](./00_Overview.md) | **Next**: [Variables and Types](./02_Variables_and_Types.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what C++ is and identify its key features and version history
2. Install a C++ development environment on Windows, macOS, or Linux
3. Implement and compile a "Hello, World!" program using g++
4. Apply common compiler options such as `-std=c++17`, `-Wall`, and `-Wextra`
5. Distinguish between `std::cout` and `std::cin` for basic input/output operations
6. Compare C and C++ approaches to I/O, memory management, and string handling
7. Configure VS Code or Visual Studio for C++ development

---

Every programming journey begins with a working toolchain. Before you can explore classes, templates, or the Standard Library, you need a compiler that turns your source code into an executable and an editor where you can write it comfortably. This lesson walks you through that essential first step and rewards you with the most satisfying milestone in any language: seeing your first program run.

## 1. What is C++?

C++ is a general-purpose programming language developed by Bjarne Stroustrup in 1979, extending the C language.

### Features of C++

| Feature | Description |
|---------|-------------|
| Object-Oriented | Supports classes, inheritance, polymorphism |
| High Performance | Low-level control close to hardware |
| Multi-Paradigm | Procedural, object-oriented, functional programming |
| Compatibility | Mostly compatible with C code |
| STL | Powerful Standard Template Library |

### C++ Version History

```
C++98 ──▶ C++03 ──▶ C++11 ──▶ C++14 ──▶ C++17 ──▶ C++20 ──▶ C++23
 │                     │
 │                     └── Beginning of "Modern C++"
 └── First standard
```

---

## 2. Development Environment Installation

### Windows

**Method 1: MinGW-w64 (Recommended)**

1. Install [MSYS2](https://www.msys2.org/)
2. Run in MSYS2 terminal:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc
   ```
3. Add to PATH environment variable: `C:\msys64\ucrt64\bin`

**Method 2: Visual Studio**

1. Install [Visual Studio Community](https://visualstudio.microsoft.com/)
2. Select "Desktop development with C++" workload

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Or install GCC via Homebrew
brew install gcc
```

### Linux (Ubuntu/Debian)

```bash
# Install GCC
sudo apt update
sudo apt install g++ build-essential

# Check version
g++ --version
```

### Linux (CentOS/RHEL)

```bash
# Install GCC
sudo dnf install gcc-c++

# Check version
g++ --version
```

---

## 3. First Program: Hello World

### Writing Code

Create a `hello.cpp` file:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### Code Explanation

```cpp
#include <iostream>    // Include I/O library
                       // <> means standard library

int main() {           // Program entry point
                       // int is return type

    std::cout          // Standard output (console)
              << "Hello, World!"  // Send string via output operator
              << std::endl;       // Newline + flush buffer

    return 0;          // Return 0 = normal termination
}
```

### Compile and Run

```bash
# Compile
g++ hello.cpp -o hello

# Run
./hello      # Linux/macOS
hello.exe    # Windows
```

Output:
```
Hello, World!
```

### Compiler Options

| Option | Description |
|--------|-------------|
| `-o filename` | Specify output file name |
| `-std=c++17` | Specify C++ standard version |
| `-Wall` | Enable all warnings |
| `-Wextra` | Enable extra warnings |
| `-g` | Include debugging information |

```bash
# Recommended compile command
g++ -std=c++17 -Wall -Wextra hello.cpp -o hello
```

---

## 4. Basic Input/Output

### Output: std::cout

```cpp
#include <iostream>

int main() {
    // String output
    std::cout << "Hello" << std::endl;

    // Multiple values
    std::cout << "Number: " << 42 << std::endl;

    // Multiple lines
    std::cout << "Line 1\n"
              << "Line 2\n"
              << "Line 3" << std::endl;

    return 0;
}
```

### Input: std::cin

```cpp
#include <iostream>

int main() {
    int age;
    std::cout << "Enter your age: ";
    std::cin >> age;
    std::cout << "You are " << age << " years old." << std::endl;

    return 0;
}
```

### String Input

```cpp
#include <iostream>
#include <string>

int main() {
    std::string name;

    std::cout << "Enter your name: ";
    std::cin >> name;  // Reads until whitespace

    std::cout << "Hello, " << name << "!" << std::endl;

    return 0;
}
```

### Reading Entire Line

```cpp
#include <iostream>
#include <string>

int main() {
    std::string fullName;

    std::cout << "Enter your name: ";
    std::getline(std::cin, fullName);  // Reads entire line

    std::cout << "Hello, " << fullName << "!" << std::endl;

    return 0;
}
```

---

## 5. using namespace std

To avoid typing `std::` every time:

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello" << endl;  // std:: can be omitted
    return 0;
}
```

### Considerations

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| `std::cout` | Prevents name collisions | More typing |
| `using namespace std;` | Concise | Possible name collisions |
| `using std::cout;` | Compromise | Declare only what's needed |

**Recommendation**: Use `std::` explicitly in header files, and use `using` only in source files.

---

## 6. Comments

```cpp
#include <iostream>

int main() {
    // Single-line comment

    /*
     * Multi-line comment
     * Also called block comment
     */

    std::cout << "Hello" << std::endl;  // Comment after code

    return 0;
}
```

---

## 7. IDE Setup

### VS Code

1. Install C/C++ extension (Microsoft)
2. Install Code Runner extension (optional)
3. Configure `tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [{
        "label": "C++ Build",
        "type": "shell",
        "command": "g++",
        "args": [
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "${file}",
            "-o",
            "${fileDirname}/${fileBasenameNoExtension}"
        ],
        "group": {
            "kind": "build",
            "isDefault": true
        }
    }]
}
```

### Visual Studio

1. File → New Project → Console App
2. Automatically builds/runs

---

## 8. Differences Between C and C++

### Header Files

```cpp
// C style (usable but not recommended)
#include <stdio.h>
#include <stdlib.h>

// C++ style (recommended)
#include <cstdio>    // C++ version of C header
#include <cstdlib>
#include <iostream>  // C++ specific
```

### I/O Comparison

```cpp
// C style
#include <cstdio>

int main() {
    int num;
    printf("Number: ");
    scanf("%d", &num);
    printf("Input: %d\n", num);
    return 0;
}
```

```cpp
// C++ style
#include <iostream>

int main() {
    int num;
    std::cout << "Number: ";
    std::cin >> num;
    std::cout << "Input: " << num << std::endl;
    return 0;
}
```

### Key Differences

| Item | C | C++ |
|------|---|-----|
| I/O | printf/scanf | cout/cin |
| Memory | malloc/free | new/delete |
| Strings | char[] | std::string |
| bool | None (use int) | bool type |
| Overloading | Not possible | Possible |
| Classes | Structs only | Class support |

---

## 9. Practice Example

### Simple Calculator

```cpp
#include <iostream>

int main() {
    double num1, num2;
    char op;

    std::cout << "First number: ";
    std::cin >> num1;

    std::cout << "Operator (+, -, *, /): ";
    std::cin >> op;

    std::cout << "Second number: ";
    std::cin >> num2;

    double result;
    switch (op) {
        case '+': result = num1 + num2; break;
        case '-': result = num1 - num2; break;
        case '*': result = num1 * num2; break;
        case '/': result = num1 / num2; break;
        default:
            std::cout << "Invalid operator." << std::endl;
            return 1;
    }

    std::cout << num1 << " " << op << " " << num2
              << " = " << result << std::endl;

    return 0;
}
```

Execution:
```
First number: 10
Operator (+, -, *, /): +
Second number: 5
10 + 5 = 15
```

---

## 10. Summary

| Concept | Description |
|---------|-------------|
| `#include` | Include header file |
| `main()` | Program entry point |
| `std::cout` | Standard output |
| `std::cin` | Standard input |
| `std::endl` | Newline + buffer flush |
| `\n` | Newline character |
| `g++` | GNU C++ compiler |

---

## Next Steps

Let's learn about variables and types in [02_Variables_and_Types.md](./02_Variables_and_Types.md)!
