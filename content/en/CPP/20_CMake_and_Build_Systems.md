# Lesson 20: CMake and Build Systems

**Previous**: [Project: Student Management System](./19_Project_Student_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why manual compilation does not scale and how CMake solves dependency tracking, platform differences, and reproducibility
2. Write a minimal `CMakeLists.txt` with `cmake_minimum_required`, `project`, and `add_executable`
3. Create library targets with `add_library` and link them using `target_link_libraries` with correct visibility (`PUBLIC`, `PRIVATE`, `INTERFACE`)
4. Set compiler warnings and C++ standard flags using `target_compile_options` and generator expressions
5. Integrate external dependencies via `find_package` and `FetchContent`
6. Configure testing with CTest and Google Test, and run tests from the command line
7. Define install rules and build-type configurations for portable, production-ready builds

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
├── tests/
│   ├── CMakeLists.txt
│   └── test_math.cpp
└── build/                  # Out-of-source build directory
```

---

## 4. Targets, Properties, and Modern CMake

Modern CMake is **target-based** — every compile flag, include path, and dependency is attached to a target.

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
# CMAKE_BUILD_TYPE=Debug    → -g (debug symbols)
# CMAKE_BUILD_TYPE=Release  → -O3 -DNDEBUG
# CMAKE_BUILD_TYPE=RelWithDebInfo → -O2 -g
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

### 6.1 find_package (System-installed)

```cmake
# Find installed libraries
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

target_link_libraries(myapp PRIVATE
    Threads::Threads
    SQLite::SQLite3
)
```

### 6.2 FetchContent (Download at configure time)

```cmake
include(FetchContent)

# Download Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Download nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

target_link_libraries(myapp PRIVATE nlohmann_json::nlohmann_json)
```

---

## 7. Testing with CTest

### 7.1 Basic CTest Integration

```cmake
# In root CMakeLists.txt
enable_testing()
add_subdirectory(tests)
```

```cmake
# tests/CMakeLists.txt
add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE mathlib)

# Register test with CTest
add_test(NAME MathTests COMMAND test_math)
```

### 7.2 With Google Test

```cmake
# tests/CMakeLists.txt
include(GoogleTest)

add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE
    mathlib
    GTest::gtest_main
)

# Auto-discover tests from GTest
gtest_discover_tests(test_math)
```

### Running Tests

```bash
# Build and run all tests
cmake --build build
cd build && ctest --output-on-failure

# Run with verbose output
ctest -V

# Run specific tests
ctest -R MathTests
```

---

## 8. Header-Only Libraries

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

## 9. Install Rules

```cmake
# Install targets
install(TARGETS myapp mathlib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
)
```

```bash
# Install to default prefix (/usr/local)
cmake --install build

# Install to custom prefix
cmake --install build --prefix /opt/myapp
```

---

## 10. Complete Example

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

# ── Testing ──────────────────────────────
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(test_calculator tests/test_calculator.cpp)
    target_link_libraries(test_calculator PRIVATE
        calclib
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(test_calculator)
endif()

# ── Install ──────────────────────────────
install(TARGETS calculator calclib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
```

---

## Practice Exercises

### Exercise 1: Build a Multi-File Project

Create a project with:
- A `stringutils` library (in `src/stringutils.cpp`) with `to_upper()`, `to_lower()`, `trim()` functions
- A `main.cpp` that uses the library
- A proper `CMakeLists.txt` with `add_library` and `add_executable`
- Build in a separate `build/` directory

### Exercise 2: Add Google Test

Extend Exercise 1:
- Add Google Test via `FetchContent`
- Write tests for all three string functions
- Register tests with `gtest_discover_tests`
- Verify with `ctest --output-on-failure`

### Exercise 3: Cross-Platform Build

Modify your CMakeLists.txt to:
- Use generator expressions for compiler-specific warnings
- Support both Debug and Release builds
- Add an `option()` to enable/disable tests
- Add `install()` rules for the executable and headers

---

## Summary

| Concept | Modern CMake Practice |
|---------|----------------------|
| Include paths | `target_include_directories()` |
| Compile flags | `target_compile_options()` |
| Linking | `target_link_libraries()` |
| C++ standard | `set(CMAKE_CXX_STANDARD 17)` |
| Dependencies | `find_package()` or `FetchContent` |
| Testing | `enable_testing()` + CTest |
| Build type | `-DCMAKE_BUILD_TYPE=Release` |

**Anti-patterns to avoid:**
- `include_directories()` — use `target_include_directories()` instead
- `link_libraries()` — use `target_link_libraries()` instead
- `add_compile_options()` — use `target_compile_options()` instead
- In-source builds — always use `cmake -B build`

---

## Navigation

**Previous**: [Project: Student Management System](./19_Project_Student_Management.md)
