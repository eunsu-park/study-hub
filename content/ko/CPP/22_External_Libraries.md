# 레슨 22: 외부 라이브러리 (Boost, nlohmann/json, fmt)

**이전**: [C++23 기능](./21_CPP23_Features.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. vcpkg와 Conan 패키지 관리자를 사용하여 C++ 의존성을 설치하고 관리한다
2. Boost.Asio를 사용하여 콜백과 코루틴으로 비동기 I/O와 네트워킹을 구현한다
3. nlohmann/json으로 직관적인 문법을 사용하여 JSON 데이터를 파싱, 생성, 조작한다
4. `fmt`를 적용하여 타입 안전하고 빠른 문자열 포맷팅을 수행하고, `std::format`과의 관계를 이해한다
5. `find_package`와 `FetchContent`를 사용하여 외부 라이브러리를 CMake 프로젝트에 통합한다

---

## 목차

1. [왜 외부 라이브러리인가?](#1-왜-외부-라이브러리인가)
2. [패키지 관리자: vcpkg와 Conan](#2-패키지-관리자-vcpkg와-conan)
3. [Boost: 스위스 아미 나이프](#3-boost-스위스-아미-나이프)
4. [nlohmann/json: 모던 C++을 위한 JSON](#4-nlohmannjson-모던-c을-위한-json)
5. [fmt: 빠른 포맷팅](#5-fmt-빠른-포맷팅)
6. [spdlog: 구조화된 로깅](#6-spdlog-구조화된-로깅)
7. [CMake 통합 패턴](#7-cmake-통합-패턴)
8. [연습문제(Exercises)](#8-연습문제exercises)

---

## 1. 왜 외부 라이브러리인가?

C++에는 표준 라이브러리를 확장하는 풍부한 라이브러리 생태계가 있습니다:

| 필요 기능 | 표준 라이브러리 | 인기 외부 라이브러리 |
|----------|----------------|---------------------|
| JSON | — | nlohmann/json, simdjson, rapidjson |
| HTTP 클라이언트 | — | Boost.Beast, cpp-httplib, cpr |
| 로깅 | — | spdlog, glog |
| 문자열 포맷팅 | `std::format` (C++20) | fmt (원조) |
| 비동기 I/O | — | Boost.Asio (Networking TS의 기반) |
| 테스팅 | — | Google Test, Catch2, doctest |
| CLI 파싱 | — | CLI11, argparse, cxxopts |

**"직접 구현하면 되지 않나요?"**

이 라이브러리들은 수천 시간의 개발, 테스트, 최적화의 결정체입니다. 이를 사용하면 다음과 같은 이점이 있습니다:
- 버그 감소 (실전 검증된 코드)
- 더 나은 성능 (전문가가 최적화)
- 유지 보수 부담 감소
- 공유된 지식 (팀원이 라이브러리를 알고 있음)

---

## 2. 패키지 관리자: vcpkg와 Conan

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

**매니페스트 모드(Manifest mode)** (권장): 프로젝트에 `vcpkg.json`을 생성하세요:

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

| 기능 | vcpkg | Conan |
|------|-------|-------|
| 패키지 수 | ~2,500 | ~1,700 |
| 바이너리 캐싱 | 예 | 예 |
| 커스텀 저장소 | Port overlays | Remotes |
| CMake 통합 | Toolchain file | Generator |
| 플랫폼 지원 | Windows/Linux/macOS | 크로스 플랫폼 + 임베디드 |

---

## 3. Boost: 스위스 아미 나이프

Boost는 160개 이상의 동료 검토(peer-reviewed) 라이브러리 모음입니다. 많은 Boost 라이브러리가 결국 C++ 표준에 포함되었습니다 (예: `shared_ptr`, `filesystem`, `optional`, `variant`).

### Boost.Asio — 비동기 I/O

Boost.Asio는 C++ 네트워크 및 비동기 프로그래밍의 기반입니다. 향후 C++ Networking TS의 기초가 됩니다.

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

### 코루틴과 함께 사용하는 Boost.Asio (C++20)

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

## 4. nlohmann/json: 모던 C++을 위한 JSON

nlohmann/json은 가장 인기 있는 C++ JSON 라이브러리입니다. 핵심 특징은 STL과 유사한 직관적인 문법입니다.

### 기본 사용법

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

### 파싱과 직렬화(Parse and Serialize)

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

### `NLOHMANN_DEFINE_TYPE`를 이용한 자동 직렬화

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

### JSON 포인터와 패치(JSON Pointer and Patch)

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

## 5. fmt: 빠른 포맷팅

`fmt`는 `std::format`에 영감을 준 라이브러리입니다. 다음과 같은 이유로 여전히 널리 사용됩니다:
- `<format>`보다 빠르게 컴파일됨
- C++11 이상에서 동작
- 표준에 아직 없는 기능 제공 (색상 출력, `fmt::join`)

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

### 커스텀 포맷터(Custom Formatters)

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

## 6. spdlog: 구조화된 로깅

spdlog는 `fmt` 위에 구축된 빠른 구조화 로깅 라이브러리입니다:

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

### 구조화된 로깅(Structured Logging)

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

## 7. CMake 통합 패턴

### 패턴 1: `find_package` (시스템 설치 또는 vcpkg)

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

### 패턴 2: `FetchContent` (구성 시점에 다운로드)

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

### 패턴 3: 헤더 전용 라이브러리(Header-Only Libraries)

```cmake
# For single-header libraries, just add include path
target_include_directories(myapp PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
```

### 모범 사례: 올바른 패턴 선택

| 상황 | 권장 방식 |
|------|----------|
| CI/CD 파이프라인 | vcpkg 매니페스트 모드 |
| 빠른 프로토타입 | FetchContent |
| 대규모 프로젝트 | Conan + CMake |
| 헤더 전용 라이브러리 | 직접 include |
| 시스템 의존성 | `find_package` |

---

## 8. 연습문제(Exercises)

### 연습문제 1: JSON 구성 시스템

nlohmann/json을 사용하여 구성 시스템을 구축하세요:
1. 중첩 설정이 있는 `Config` 구조체를 정의하세요
2. 검증과 함께 JSON 파일에서 로드하세요
3. 누락된 필드에 대한 기본값을 지원하세요
4. 구성 병합(파일 기본값 + 사용자 재정의)을 구현하세요

### 연습문제 2: Boost.Asio를 이용한 비동기 TCP 서버

Boost.Asio를 사용하여 다중 클라이언트 에코 서버를 작성하세요:
1. 여러 연결을 동시에 수락하세요
2. 수신한 데이터를 각 클라이언트에게 에코(echo) 하세요
3. spdlog를 사용하여 연결을 로깅하세요
4. 클라이언트 연결 해제를 우아하게 처리하세요

### 연습문제 3: 커스텀 fmt 포맷터

다음을 위한 커스텀 `fmt` 포맷터를 만드세요:
1. `Matrix` 클래스 (정렬이 맞춰진 예쁜 출력)
2. `Duration` 클래스 ("2h 30m 15s" 형식으로 포맷)
3. `Color` 클래스 (hex "#RRGGBB" 또는 RGB "(r, g, b)" 형식으로 포맷)

### 연습문제 4: 라이브러리 비교

같은 프로그램을 세 가지 방식으로 작성하세요:
1. 표준 라이브러리만 사용
2. Boost + nlohmann/json + fmt 사용
3. 바이너리 크기, 컴파일 시간, 코드 가독성을 측정하고 비교하세요

### 연습문제 5: CMake 다중 라이브러리 프로젝트

다음을 수행하는 CMake 프로젝트를 만드세요:
1. nlohmann/json에 `FetchContent` 사용
2. Boost에 `find_package` 사용
3. 라이브러리 타겟과 실행 파일 타겟 포함
4. Google Test를 이용한 단위 테스트 포함
5. Linux, macOS, Windows에서 빌드 (제너레이터 표현식(generator expressions) 사용)

---

## 내비게이션

**이전**: [C++23 기능](./21_CPP23_Features.md)
