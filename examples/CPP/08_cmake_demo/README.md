# CMake Demo: Calculator Library

A simple calculator project demonstrating modern CMake practices for C++17.

## Project Structure

```
08_cmake_demo/
├── CMakeLists.txt              # Root build configuration
├── include/
│   └── calculator.hpp          # Public header
├── src/
│   ├── calculator.cpp          # Library implementation
│   └── main.cpp                # Application entry point
├── tests/
│   ├── CMakeLists.txt          # Test configuration
│   └── test_calculator.cpp     # Unit tests
└── README.md
```

## Build & Run

```bash
# Configure
cmake -B build

# Build
cmake --build build

# Run the application
./build/calculator

# Run tests
cd build && ctest --output-on-failure
```

## Build Types

```bash
# Debug (with debug symbols, no optimization)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release (optimized)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Disable tests
cmake -B build -DBUILD_TESTS=OFF
```

## Key CMake Concepts Demonstrated

| Concept | Usage |
|---------|-------|
| `add_library` | Creates `calclib` library target |
| `add_executable` | Creates `calculator` executable |
| `target_include_directories` | PUBLIC/PRIVATE include paths |
| `target_link_libraries` | Links library to executable |
| `target_compile_options` | Per-target compiler warnings |
| `enable_testing` + `add_test` | CTest integration |
| Generator expressions | Cross-platform compiler flags |
| `install` rules | Installation targets |
