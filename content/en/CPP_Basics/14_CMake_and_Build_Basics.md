# CMake and Build Basics

**Previous**: [Exceptions and File I/O](./13_Exceptions_and_File_IO.md) | **Next**: [Project: Student Management System](./15_Project_Student_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why manual compilation does not scale and how CMake solves dependency tracking, platform differences, and reproducibility
2. Write a minimal `CMakeLists.txt` with `cmake_minimum_required`, `project`, and `add_executable`
3. Create library targets with `add_library` and link them using `target_link_libraries` with correct visibility (`PUBLIC`, `PRIVATE`, `INTERFACE`)
4. Set compiler warnings and C++ standard flags using `target_compile_options` and generator expressions
5. Structure a simple multi-file project with separate source and include directories
6. Configure Debug and Release builds using `CMAKE_BUILD_TYPE`

---

A C++ program that compiles on your machine means nothing if it cannot be reliably built by a teammate, a CI server, or your future self on a different OS. CMake has become the industry standard for C++ build configuration precisely because it abstracts away platform-specific toolchain details while giving you fine-grained control over targets, dependencies, and testing. Mastering CMake is not a detour from learning C++--it is the skill that turns your code into shippable software.

---

## 1. Why Build Systems?

For anything beyond a single-file program, manually running `g++` becomes impractical:

```bash
# This doesn't scale
g++ -std=c++17 -Wall -I./include \
    src/main.cpp src/math.cpp src/utils.cpp \
    -lsqlite3 -lpthread -o myapp
```

Problems:
- **Dependency tracking**: which files changed? what needs recompilation?
- **Ordering**: libraries must be linked after object files
- **Platform differences**: Linux vs macOS vs Windows flags differ
- **Reproducibility**: every developer must use the same flags

### Build System Landscape

| Tool | Type | Description |
|------|------|-------------|
| Make | Build tool | Rule-based, UNIX-centric |
| CMake | Meta-build system | Generates Makefiles, Ninja, VS solutions |
| Meson | Meta-build system | Python-based, fast |
| Bazel | Build system | Google, hermetic builds |
| Ninja | Build tool | Low-level, designed for generators |

**CMake** is the de facto standard for C++ projects.

---

## 2. Minimal CMakeLists.txt

```cmake
# Minimum CMake version required
cmake_minimum_required(VERSION 3.16)

# Project name, version, and languages
project(MyApp VERSION 1.0.0 LANGUAGES CXX)

# Create an executable target
add_executable(myapp src/main.cpp)
```

### Build Commands

```bash
# Configure (generates build files)
cmake -B build

# Build
cmake --build build

# Run
./build/myapp
```

---

## 3. Project Structure

A typical C++ project layout:

```
myproject/
├── CMakeLists.txt          # Root CMake file
├── src/
│   ├── main.cpp
│   ├── math.cpp
│   └── math.hpp
├── include/
│   └── myproject/
│       └── utils.hpp       # Public headers
└── build/                  # Out-of-source build directory
```

---

## 4. Targets, Properties, and Modern CMake

Modern CMake is **target-based** -- every compile flag, include path, and dependency is attached to a target.

### 4.1 Executable and Library Targets

```cmake
cmake_minimum_required(VERSION 3.16)
project(Calculator VERSION 1.0 LANGUAGES CXX)

# Set C++ standard for the whole project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create a library
add_library(mathlib
    src/math.cpp
    src/utils.cpp
)

# Specify include directories for the library
target_include_directories(mathlib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include    # Users of mathlib see these
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src        # Only mathlib itself sees these
)

# Create executable that uses the library
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE mathlib)
```

### 4.2 PUBLIC, PRIVATE, INTERFACE

| Keyword | This target | Consumers of this target |
|---------|-------------|--------------------------|
| PUBLIC | Yes | Yes |
| PRIVATE | Yes | No |
| INTERFACE | No | Yes |

```cmake
# mathlib uses Eigen internally (PRIVATE)
# mathlib exposes nlohmann_json in its API (PUBLIC)
target_link_libraries(mathlib
    PRIVATE Eigen3::Eigen
    PUBLIC  nlohmann_json::nlohmann_json
)
```

---

## 5. Compiler Warnings and Flags

```cmake
# Add warnings to a specific target
target_compile_options(calculator PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Build-type specific flags are handled automatically:
# CMAKE_BUILD_TYPE=Debug    -> -g (debug symbols)
# CMAKE_BUILD_TYPE=Release  -> -O3 -DNDEBUG
# CMAKE_BUILD_TYPE=RelWithDebInfo -> -O2 -g
```

### Configuring Build Type

```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

---

## 6. Finding External Libraries

### find_package (System-installed)

```cmake
# Find installed libraries
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

target_link_libraries(myapp PRIVATE
    Threads::Threads
    SQLite::SQLite3
)
```

---

## 7. Header-Only Libraries

```cmake
# Header-only library (no .cpp files)
add_library(myheaders INTERFACE)
target_include_directories(myheaders
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Consumers just link
target_link_libraries(consumer PRIVATE myheaders)
```

---

## 8. Complete Basic Example

```cmake
cmake_minimum_required(VERSION 3.16)
project(Calculator
    VERSION 1.0.0
    DESCRIPTION "A simple calculator library"
    LANGUAGES CXX
)

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Library ──────────────────────────────
add_library(calclib
    src/calculator.cpp
)

target_include_directories(calclib
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(calclib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)

# ── Executable ───────────────────────────
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE calclib)
```

---

## 9. Summary

| Concept | Modern CMake Practice |
|---------|----------------------|
| Include paths | `target_include_directories()` |
| Compile flags | `target_compile_options()` |
| Linking | `target_link_libraries()` |
| C++ standard | `set(CMAKE_CXX_STANDARD 17)` |
| Dependencies | `find_package()` |
| Build type | `-DCMAKE_BUILD_TYPE=Release` |

**Anti-patterns to avoid:**
- `include_directories()` -- use `target_include_directories()` instead
- `link_libraries()` -- use `target_link_libraries()` instead
- `add_compile_options()` -- use `target_compile_options()` instead
- In-source builds -- always use `cmake -B build`

---

## Practice Exercises

### Exercise 1: Build a Multi-File Project

Create a project with:
- A `stringutils` library (in `src/stringutils.cpp`) with `to_upper()`, `to_lower()`, `trim()` functions
- A `main.cpp` that uses the library
- A proper `CMakeLists.txt` with `add_library` and `add_executable`
- Build in a separate `build/` directory

### Exercise 2: Add Include Directories

Extend Exercise 1:
- Move the header file to `include/stringutils.hpp`
- Use `target_include_directories` with `PUBLIC` so both the library and executable can find the header
- Verify that the project builds correctly from a clean `build/` directory

### Exercise 3: Cross-Platform Warnings

Modify your CMakeLists.txt to:
- Use generator expressions for compiler-specific warnings (GCC/Clang: `-Wall -Wextra`, MSVC: `/W4`)
- Support both Debug and Release builds
- Verify that Debug builds include debug symbols (`-g`) and Release builds enable optimization (`-O3`)

---

For advanced CMake features (FetchContent, CTest, packaging), see [External Libraries and Build](../CPP_Advanced/17_External_Libraries_and_Build.md).

---

## Next Steps

Let's put everything together in a capstone project: [Project: Student Management System](./15_Project_Student_Management.md)!
