# CPP_Basics Examples

Example code for C++ Basics course (Tier 2).

## Directory Structure

| Directory | Topic | Description |
|-----------|-------|-------------|
| `02_variables/` | Variables & Types | Fundamental types, auto, const, constexpr |
| `03_operators/` | Operators & Control Flow | Arithmetic, logical, bitwise, if/switch/loops |
| `04_functions/` | Functions | Overloading, default params, pass-by-ref, lambda |
| `05_arrays_strings/` | Arrays & Strings | C arrays, std::array, std::string operations |
| `06_pointers/` | Pointers & References | Pointers, references, new/delete, const variants |
| `07_namespaces_io/` | Namespaces & I/O | Namespace, cin/cout, iomanip, stringstream |
| `08_classes/` | Classes Basics | Class definition, constructor, destructor |
| `09_classes_advanced/` | Classes Advanced | Operator overloading, copy constructor, Rule of Three |
| `10_inheritance/` | Inheritance & Polymorphism | Virtual functions, abstract class, dynamic_cast |
| `11_stl_containers/` | STL Containers | vector, map, set, deque, unordered_map |
| `12_stl_algorithms/` | STL Algorithms | sort, find, transform, accumulate, lambda |
| `13_exceptions_file_io/` | Exceptions & File I/O | try/catch, custom exceptions, file read/write |
| `14_cmake_basics/` | CMake Basics | CMakeLists.txt, project structure, testing |
| `15_student_management/` | Project: Student Management | Complete CRUD application |

## Building

```bash
# Build all single-file examples
make all

# Build a specific example
cd 02_variables && g++ -std=c++20 -Wall -Wextra -o variables_demo variables_demo.cpp

# Clean
make clean
```

## Requirements

- GCC 12+ or Clang 15+ with C++20 support
- CMake 3.20+ (for cmake_basics example)
