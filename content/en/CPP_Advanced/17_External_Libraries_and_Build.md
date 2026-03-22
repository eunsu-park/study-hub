# External Libraries and Advanced Build Systems

**Previous**: [C++23 Features](./16_CPP23_Features.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and manage C++ dependencies using vcpkg and Conan package managers
2. Use `FetchContent` to download and integrate third-party libraries at CMake configure time
3. Configure testing with CTest and Google Test, and package builds with CPack
4. Apply CMake presets for reproducible, shareable build configurations
5. Integrate popular libraries (Boost, fmt, spdlog, nlohmann/json, Eigen) into C++ projects
6. Evaluate trade-offs between package managers, `FetchContent`, and manual dependency management

---

C++ has one of the richest library ecosystems in programming, but historically, dependency management has been its weakest point. Modern tooling -- vcpkg, Conan, CMake's `FetchContent`, and CMake presets -- has largely solved this problem. This lesson covers both the libraries you should know and the build system techniques that make using them practical in real projects. Mastering these tools is what transforms a collection of source files into portable, testable, shippable software.

---

## Table of Contents

1. [Package Managers: vcpkg and Conan](#1-package-managers-vcpkg-and-conan)
2. [CMake FetchContent](#2-cmake-fetchcontent)
3. [CTest and Testing Integration](#3-ctest-and-testing-integration)
4. [CPack and Packaging](#4-cpack-and-packaging)
5. [CMake Presets](#5-cmake-presets)
6. [Popular Libraries](#6-popular-libraries)

---

## 1. Package Managers: vcpkg and Conan

### vcpkg (Microsoft)

```bash
# Install vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh

# Install libraries
./vcpkg install nlohmann-json boost-asio fmt spdlog eigen3

# Use with CMake (toolchain file)
cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

**Manifest mode** (recommended): create `vcpkg.json` in your project root:

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": [
    "nlohmann-json",
    "boost-asio",
    "fmt",
    "spdlog",
    "eigen3"
  ]
}
```

### Conan

```bash
# Install Conan
pip install conan

# Create conanfile.txt
cat > conanfile.txt << 'EOF'
[requires]
nlohmann_json/3.11.3
boost/1.84.0
fmt/10.2.1
spdlog/1.13.0

[generators]
CMakeDeps
CMakeToolchain
EOF

# Install dependencies
conan install . --build=missing
```

### vcpkg vs Conan

| Feature | vcpkg | Conan |
|---------|-------|-------|
| Package count | ~2,500 | ~1,700 |
| Binary caching | Yes | Yes |
| Custom repos | Port overlays | Remotes |
| CMake integration | Toolchain file | Generator |
| Platform support | Windows/Linux/macOS | Cross-platform + embedded |
| Reproducibility | Manifest + baseline | Lockfile |

---

## 2. CMake FetchContent

`FetchContent` downloads and builds dependencies at CMake configure time -- no external package manager needed.

### Basic Usage

```cmake
include(FetchContent)

# Declare dependencies
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.13.0
)

# Download and make available
FetchContent_MakeAvailable(json fmt spdlog)

# Link against them
add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE
    nlohmann_json::nlohmann_json
    fmt::fmt
    spdlog::spdlog
)
```

### Controlling FetchContent Behavior

```cmake
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    # Prevent installing gtest when 'cmake --install' is run
    OVERRIDE_FIND_PACKAGE
)

# Optionally set options before MakeAvailable
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest spdlog)
```

### FetchContent vs Package Managers

| Scenario | Recommended Approach |
|----------|---------------------|
| CI/CD pipeline with many deps | vcpkg manifest mode |
| Quick prototype | FetchContent |
| Large enterprise project | Conan + CMake |
| Header-only library | Direct include or FetchContent |
| System dependency (OpenSSL, etc.) | `find_package` |

---

## 3. CTest and Testing Integration

### Basic CTest

```cmake
# In root CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(MyProject LANGUAGES CXX)

enable_testing()

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math PRIVATE mathlib)

# Register test
add_test(NAME MathTests COMMAND test_math)
```

### With Google Test

```cmake
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math PRIVATE
    mathlib
    GTest::gtest_main
)

# Auto-discover tests from GTest macros
gtest_discover_tests(test_math)
```

### Writing Tests

```cpp
// tests/test_math.cpp
#include <gtest/gtest.h>
#include "math.h"

TEST(MathTest, Addition) {
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-1, 1), 0);
    EXPECT_EQ(add(0, 0), 0);
}

TEST(MathTest, Division) {
    EXPECT_DOUBLE_EQ(divide(10.0, 3.0), 10.0 / 3.0);
    EXPECT_THROW(divide(1.0, 0.0), std::invalid_argument);
}

TEST(MathTest, Fibonacci) {
    EXPECT_EQ(fibonacci(0), 0);
    EXPECT_EQ(fibonacci(1), 1);
    EXPECT_EQ(fibonacci(10), 55);
}
```

### Running Tests

```bash
# Build and run all tests
cmake --build build
cd build && ctest --output-on-failure

# Verbose output
ctest -V

# Run specific tests by name regex
ctest -R MathTest

# Run tests in parallel
ctest -j$(nproc)
```

---

## 4. CPack and Packaging

CPack generates distributable packages (DEB, RPM, ZIP, NSIS, DMG) from your CMake project.

### Basic CPack Configuration

```cmake
# At the end of your root CMakeLists.txt
install(TARGETS myapp
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# CPack settings
set(CPACK_PACKAGE_NAME "MyApp")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_CONTACT "dev@example.com")
set(CPACK_PACKAGE_DESCRIPTION "My awesome C++ application")

# Generator-specific settings
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libstdc++6")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

include(CPack)
```

### Building Packages

```bash
# Build the project
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Generate packages
cd build
cpack -G ZIP     # ZIP archive
cpack -G DEB     # Debian package
cpack -G RPM     # RPM package
cpack -G NSIS    # Windows installer
cpack -G DragNDrop  # macOS DMG
```

---

## 5. CMake Presets

CMake presets (introduced in CMake 3.19) provide a standardized way to share build configurations.

### CMakePresets.json

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },
  "configurePresets": [
    {
      "name": "dev",
      "displayName": "Development",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/dev",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_CXX_STANDARD": "20",
        "BUILD_TESTS": "ON",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    },
    {
      "name": "release",
      "displayName": "Release",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_STANDARD": "20",
        "BUILD_TESTS": "OFF"
      }
    },
    {
      "name": "vcpkg",
      "displayName": "vcpkg Integration",
      "inherits": "dev",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "dev",
      "configurePreset": "dev"
    },
    {
      "name": "release",
      "configurePreset": "release"
    }
  ],
  "testPresets": [
    {
      "name": "dev",
      "configurePreset": "dev",
      "output": { "outputOnFailure": true },
      "execution": { "jobs": 4 }
    }
  ]
}
```

### Using Presets

```bash
# List available presets
cmake --list-presets

# Configure with a preset
cmake --preset dev

# Build with a preset
cmake --build --preset dev

# Test with a preset
ctest --preset dev
```

### User-Local Presets

Create `CMakeUserPresets.json` (gitignored) for personal overrides:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "my-local",
      "inherits": "dev",
      "cacheVariables": {
        "CMAKE_CXX_COMPILER": "/opt/gcc-14/bin/g++"
      }
    }
  ]
}
```

---

## 6. Popular Libraries

### Boost

The "Swiss Army Knife" of C++ -- 160+ peer-reviewed libraries. Many eventually become part of the standard.

```cpp
#include <boost/asio.hpp>
#include <iostream>

namespace asio = boost::asio;

// Simple TCP echo server
void echo_server() {
    asio::io_context io;
    asio::ip::tcp::acceptor acceptor(
        io, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 8080));

    asio::ip::tcp::socket socket(io);
    acceptor.accept(socket);

    char buf[1024];
    boost::system::error_code ec;
    size_t len = socket.read_some(asio::buffer(buf), ec);
    if (!ec) {
        asio::write(socket, asio::buffer(buf, len));
    }
}
```

### nlohmann/json

Intuitive, STL-like JSON library:

```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

void demo() {
    // Create JSON
    json config = {
        {"name", "MyApp"},
        {"version", 2},
        {"ports", {8080, 8443}},
        {"database", {{"host", "localhost"}, {"port", 5432}}}
    };

    // Access
    std::string name = config["name"];
    int port = config["database"]["port"];

    // Serialize
    std::string pretty = config.dump(2);
}

// Auto serialization
struct Person {
    std::string name;
    int age;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Person, name, age)
};
```

### fmt

The library that inspired `std::format` -- still widely used for its extra features:

```cpp
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

void demo() {
    fmt::println("x = {}, y = {:.2f}", 42, 3.14159);

    // Format containers
    std::vector v = {1, 2, 3};
    fmt::println("v = {}", v);               // [1, 2, 3]
    fmt::println("v = {}", fmt::join(v, ", ")); // 1, 2, 3

    // Chrono
    auto now = std::chrono::system_clock::now();
    fmt::println("Time: {:%Y-%m-%d %H:%M:%S}", now);

    // Color
    fmt::print(fg(fmt::color::green), "Success!\n");
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "Error!\n");
}
```

### spdlog

Fast, structured logging built on fmt:

```cpp
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>

void demo() {
    spdlog::info("Welcome to spdlog!");
    spdlog::warn("Easy padding: {:08d}", 42);
    spdlog::error("Error code: {:#x}", 0xDEAD);

    // File logger with rotation
    auto logger = spdlog::rotating_logger_mt(
        "file_logger", "logs/app.log",
        1024 * 1024 * 5, 3);
    logger->info("Logged to file");

    // Custom pattern
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
}
```

### Eigen

Linear algebra library -- the standard choice for scientific computing:

```cpp
#include <Eigen/Dense>
#include <iostream>

void demo() {
    // Matrix operations
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 10;

    Eigen::Vector3d b(3, 3, 4);

    // Solve Ax = b
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
    std::cout << "Solution:\n" << x << "\n";

    // Eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(A);
    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n";
}
```

### CMake Integration for All Libraries

```cmake
cmake_minimum_required(VERSION 3.20)
project(FullExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find installed libraries (via vcpkg or system)
find_package(Boost 1.80 REQUIRED COMPONENTS system)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(fmt 10.0 REQUIRED)
find_package(spdlog 1.12 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE
    Boost::system
    nlohmann_json::nlohmann_json
    fmt::fmt
    spdlog::spdlog
    Eigen3::Eigen
)
```

### Library Selection Guide

| Need | Recommended Library |
|------|-------------------|
| JSON | nlohmann/json (ease) or simdjson (speed) |
| HTTP client | cpp-httplib (simple) or Boost.Beast (full) |
| Logging | spdlog |
| String formatting | fmt (or std::format if C++20 suffices) |
| Linear algebra | Eigen |
| Testing | Google Test or Catch2 |
| CLI parsing | CLI11 or cxxopts |
| Async I/O / Networking | Boost.Asio |

---

## Exercises

### Exercise 1: JSON Configuration System

Build a configuration system using nlohmann/json:
1. Define a `Config` struct with nested settings
2. Load from a JSON file with validation
3. Support default values for missing fields
4. Implement config merging (file defaults + user overrides)

### Exercise 2: CMake Multi-Library Project

Create a CMake project that:
1. Uses `FetchContent` for nlohmann/json and fmt
2. Uses `find_package` for Boost
3. Has a library target and an executable target
4. Includes Google Test with `gtest_discover_tests`
5. Has presets for Debug, Release, and CI builds

### Exercise 3: Custom fmt Formatter

Create custom `fmt` formatters for:
1. A `Matrix` class (pretty-print with alignment)
2. A `Duration` class (format as "2h 30m 15s")
3. A `Color` class (format as hex "#RRGGBB" or RGB "(r, g, b)")

### Exercise 4: Logging Library

Build a logging library using spdlog as the backend:
1. Singleton logger with lazy initialization
2. Console + rotating file sinks
3. Structured log fields (key=value pairs)
4. CMake library target that consumers can link against

### Exercise 5: Scientific Calculator

Build a scientific calculator using Eigen for matrix operations and nlohmann/json for input:
1. Read matrix operations from a JSON file
2. Support: add, multiply, inverse, eigenvalue decomposition
3. Output results in JSON format
4. Package with CPack as a ZIP distribution

---

## Next Steps

Congratulations on completing C++ Advanced! You now have deep expertise in modern C++, from move semantics and template metaprogramming through C++20/23 features to design patterns and build systems. Explore [CUDA](../CUDA/00_Overview.md) for GPU programming or [DL_Scratch_C](../DL_Scratch_C/00_Overview.md) for building deep learning from scratch.
