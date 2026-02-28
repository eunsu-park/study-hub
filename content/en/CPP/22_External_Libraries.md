# Lesson 22: External Libraries (Boost, nlohmann/json, fmt)

**Previous**: [C++23 Features](./21_CPP23_Features.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and manage C++ dependencies using vcpkg and Conan package managers
2. Use Boost.Asio for asynchronous I/O and networking with callbacks and coroutines
3. Parse, create, and manipulate JSON data with nlohmann/json using intuitive syntax
4. Apply `fmt` for type-safe, fast string formatting and understand its relationship to `std::format`
5. Integrate external libraries into CMake projects using `find_package` and `FetchContent`

---

## Table of Contents

1. [Why External Libraries?](#1-why-external-libraries)
2. [Package Managers: vcpkg and Conan](#2-package-managers-vcpkg-and-conan)
3. [Boost: The Swiss Army Knife](#3-boost-the-swiss-army-knife)
4. [nlohmann/json: JSON for Modern C++](#4-nlohmannjson-json-for-modern-c)
5. [fmt: Fast Formatting](#5-fmt-fast-formatting)
6. [spdlog: Structured Logging](#6-spdlog-structured-logging)
7. [CMake Integration Patterns](#7-cmake-integration-patterns)
8. [Exercises](#8-exercises)

---

## 1. Why External Libraries?

C++ has a rich ecosystem of libraries that extend the standard library:

| Need | Standard Library | Popular External Library |
|------|-----------------|------------------------|
| JSON | — | nlohmann/json, simdjson, rapidjson |
| HTTP client | — | Boost.Beast, cpp-httplib, cpr |
| Logging | — | spdlog, glog |
| String formatting | `std::format` (C++20) | fmt (the original) |
| Async I/O | — | Boost.Asio (basis for Networking TS) |
| Testing | — | Google Test, Catch2, doctest |
| CLI parsing | — | CLI11, argparse, cxxopts |

**Why not "just write it yourself"?**

These libraries represent thousands of person-hours of development, testing, and optimization. Using them means:
- Fewer bugs (battle-tested code)
- Better performance (expert-optimized)
- Less maintenance burden
- Shared knowledge (team members know the library)

---

## 2. Package Managers: vcpkg and Conan

### vcpkg (Microsoft)

```bash
# Install vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh

# Install a library
./vcpkg install nlohmann-json boost-asio fmt spdlog

# Use with CMake (toolchain file)
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

**Manifest mode** (recommended): create `vcpkg.json` in your project:

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": [
    "nlohmann-json",
    "boost-asio",
    "fmt",
    "spdlog"
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

---

## 3. Boost: The Swiss Army Knife

Boost is a collection of 160+ peer-reviewed libraries. Many Boost libraries eventually become part of the C++ standard (e.g., `shared_ptr`, `filesystem`, `optional`, `variant`).

### Boost.Asio — Asynchronous I/O

Boost.Asio is the foundation for networked and async programming in C++. It's the basis for the upcoming C++ Networking TS.

```cpp
#include <boost/asio.hpp>
#include <iostream>

namespace asio = boost::asio;

// Simple TCP echo server
void echo_server() {
    asio::io_context io;
    asio::ip::tcp::acceptor acceptor(
        io, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 8080));

    while (true) {
        asio::ip::tcp::socket socket(io);
        acceptor.accept(socket);

        // Read data
        char buf[1024];
        boost::system::error_code ec;
        size_t len = socket.read_some(asio::buffer(buf), ec);
        if (!ec) {
            // Echo back
            asio::write(socket, asio::buffer(buf, len));
        }
    }
}
```

### Boost.Asio with Coroutines (C++20)

```cpp
#include <boost/asio.hpp>
#include <boost/asio/co_spawn.hpp>

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

asio::awaitable<void> handle_client(tcp::socket socket) {
    char buf[1024];
    while (true) {
        auto n = co_await socket.async_read_some(
            asio::buffer(buf), asio::use_awaitable);
        co_await asio::async_write(
            socket, asio::buffer(buf, n), asio::use_awaitable);
    }
}

asio::awaitable<void> listener(asio::io_context& io) {
    tcp::acceptor acceptor(io, {tcp::v4(), 8080});
    while (true) {
        auto socket = co_await acceptor.async_accept(asio::use_awaitable);
        asio::co_spawn(io, handle_client(std::move(socket)), asio::detached);
    }
}
```

### Boost.Beast — HTTP/WebSocket

```cpp
#include <boost/beast.hpp>
#include <boost/asio.hpp>

namespace beast = boost::beast;
namespace http = beast::http;

// Simple HTTP GET request
void http_get(const std::string& host, const std::string& target) {
    asio::io_context io;
    asio::ip::tcp::resolver resolver(io);
    beast::tcp_stream stream(io);

    auto results = resolver.resolve(host, "80");
    stream.connect(results);

    http::request<http::string_body> req{http::verb::get, target, 11};
    req.set(http::field::host, host);
    http::write(stream, req);

    beast::flat_buffer buffer;
    http::response<http::string_body> res;
    http::read(stream, buffer, res);

    std::cout << res.body() << std::endl;
}
```

---

## 4. nlohmann/json: JSON for Modern C++

nlohmann/json is the most popular C++ JSON library. Its key feature: intuitive, STL-like syntax.

### Basic Usage

```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

void basics() {
    // Create JSON from initializer list
    json config = {
        {"name", "MyApp"},
        {"version", 2},
        {"debug", false},
        {"ports", {8080, 8443}},
        {"database", {
            {"host", "localhost"},
            {"port", 5432}
        }}
    };

    // Access values
    std::string name = config["name"];
    int port = config["database"]["port"];
    bool debug = config.value("debug", true);  // with default

    // Iterate
    for (auto& [key, val] : config.items()) {
        std::cout << key << ": " << val << "\n";
    }

    // Serialize
    std::string pretty = config.dump(2);  // 2-space indent
    std::cout << pretty << "\n";
}
```

### Parse and Serialize

```cpp
void parse_demo() {
    // Parse from string
    auto j = json::parse(R"({"name": "Alice", "age": 30})");

    // Parse from file
    std::ifstream f("config.json");
    json file_json = json::parse(f);

    // Safe parsing (no exceptions)
    auto result = json::parse("invalid json", nullptr, false);
    if (result.is_discarded()) {
        std::cerr << "Parse error!\n";
    }
}
```

### Automatic Serialization with `NLOHMANN_DEFINE_TYPE`

```cpp
struct Person {
    std::string name;
    int age;
    std::vector<std::string> hobbies;

    // Auto-generate to_json/from_json
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Person, name, age, hobbies)
};

void struct_demo() {
    Person alice{"Alice", 30, {"reading", "coding"}};

    // Serialize struct → JSON
    json j = alice;
    std::cout << j.dump(2) << "\n";

    // Deserialize JSON → struct
    auto bob = j.get<Person>();
    bob.name = "Bob";
}
```

### JSON Pointer and Patch

```cpp
void advanced_demo() {
    json j = {{"a", {{"b", {{"c", 42}}}}}};

    // JSON Pointer (RFC 6901)
    int val = j["/a/b/c"_json_pointer];  // 42

    // JSON Patch (RFC 6902)
    json patch = json::array({
        {{"op", "replace"}, {"path", "/a/b/c"}, {"value", 100}}
    });
    json patched = j.patch(patch);
}
```

---

## 5. fmt: Fast Formatting

`fmt` is the library that inspired `std::format`. It's still widely used because:
- It compiles faster than `<format>`
- It works with C++11 and later
- It has features not yet in the standard (color output, `fmt::join`)

```cpp
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

void fmt_demo() {
    // Basic formatting (same syntax as std::format)
    std::string s = fmt::format("Hello, {}!", "world");

    // Print directly (like std::print)
    fmt::println("x = {}, y = {:.2f}", 42, 3.14159);

    // Format containers directly
    std::vector v = {1, 2, 3, 4, 5};
    fmt::println("v = {}", v);            // v = [1, 2, 3, 4, 5]
    fmt::println("v = {}", fmt::join(v, ", "));  // v = 1, 2, 3, 4, 5

    // Chrono formatting
    auto now = std::chrono::system_clock::now();
    fmt::println("Time: {:%Y-%m-%d %H:%M:%S}", now);

    // Colored output
    fmt::print(fg(fmt::color::green), "Success: {}\n", "all tests passed");
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "Error: {}\n", "failed");

    // Named arguments
    fmt::println("{name} scored {score} points",
                 fmt::arg("name", "Alice"),
                 fmt::arg("score", 95));
}
```

### Custom Formatters

```cpp
struct Point { double x, y; };

template<>
struct fmt::formatter<Point> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const Point& p, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "({:.2f}, {:.2f})", p.x, p.y);
    }
};

// Now works with fmt::format, fmt::print, etc.
Point p{1.5, 2.7};
fmt::println("Point: {}", p);  // Point: (1.50, 2.70)
```

---

## 6. spdlog: Structured Logging

spdlog builds on `fmt` to provide fast, structured logging:

```cpp
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>

void logging_demo() {
    // Default logger (stdout)
    spdlog::info("Welcome to spdlog!");
    spdlog::warn("Easy padding in numbers: {:08d}", 42);
    spdlog::error("Error code: {:#x}", 0xDEAD);

    // Set log level
    spdlog::set_level(spdlog::level::debug);
    spdlog::debug("This message is now visible");

    // File logger with rotation
    auto logger = spdlog::rotating_logger_mt(
        "file_logger",
        "logs/app.log",
        1024 * 1024 * 5,  // 5 MB max
        3                   // 3 rotated files
    );
    logger->info("Logged to file");

    // Custom pattern
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
}
```

### Structured Logging

```cpp
// Use spdlog with JSON formatter for structured logging
#include <spdlog/sinks/stdout_color_sinks.h>

void structured_demo() {
    auto logger = spdlog::stdout_color_mt("app");

    // Structured context via key-value pairs
    logger->info("User login: user={} ip={} status={}",
                 "alice", "192.168.1.1", "success");

    // Output: [2024-01-15 10:30:00] [info] User login: user=alice ip=192.168.1.1 status=success
}
```

---

## 7. CMake Integration Patterns

### Pattern 1: `find_package` (system-installed or vcpkg)

```cmake
cmake_minimum_required(VERSION 3.20)
project(myapp LANGUAGES CXX)

find_package(Boost 1.80 REQUIRED COMPONENTS system)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(fmt 10.0 REQUIRED)
find_package(spdlog 1.12 REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE
    Boost::system
    nlohmann_json::nlohmann_json
    fmt::fmt
    spdlog::spdlog
)
target_compile_features(myapp PRIVATE cxx_std_20)
```

### Pattern 2: `FetchContent` (download at configure time)

```cmake
include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
)

FetchContent_MakeAvailable(json fmt)

target_link_libraries(myapp PRIVATE nlohmann_json::nlohmann_json fmt::fmt)
```

### Pattern 3: Header-Only Libraries

```cmake
# For single-header libraries, just add include path
target_include_directories(myapp PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
```

### Best Practice: Choose the Right Pattern

| Scenario | Recommended Approach |
|----------|---------------------|
| CI/CD pipeline | vcpkg manifest mode |
| Quick prototype | FetchContent |
| Large project | Conan + CMake |
| Header-only lib | Direct include |
| System dependency | `find_package` |

---

## 8. Exercises

### Exercise 1: JSON Configuration System

Build a configuration system using nlohmann/json:
1. Define a `Config` struct with nested settings
2. Load from a JSON file with validation
3. Support default values for missing fields
4. Implement config merging (file defaults + user overrides)

### Exercise 2: Async TCP Server with Boost.Asio

Write a multi-client echo server using Boost.Asio:
1. Accept multiple connections concurrently
2. Echo received data back to each client
3. Log connections using spdlog
4. Handle client disconnection gracefully

### Exercise 3: Custom fmt Formatter

Create custom `fmt` formatters for:
1. A `Matrix` class (pretty-print with alignment)
2. A `Duration` class (format as "2h 30m 15s")
3. A `Color` class (format as hex "#RRGGBB" or RGB "(r, g, b)")

### Exercise 4: Library Comparison

Write the same program three different ways:
1. Using only the standard library
2. Using Boost + nlohmann/json + fmt
3. Measure and compare: binary size, compile time, code readability

### Exercise 5: CMake Multi-Library Project

Create a CMake project that:
1. Uses `FetchContent` for nlohmann/json
2. Uses `find_package` for Boost
3. Has a library target and an executable target
4. Includes unit tests with Google Test
5. Builds on Linux, macOS, and Windows (use generator expressions)

---

## Navigation

**Previous**: [C++23 Features](./21_CPP23_Features.md)
