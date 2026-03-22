# 외부 라이브러리와 고급 빌드 시스템

**이전**: [C++23 기능](./16_CPP23_Features.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. vcpkg와 Conan 패키지 매니저를 사용하여 C++ 의존성을 설치하고 관리할 수 있다
2. `FetchContent`를 사용하여 CMake 구성 시점에 서드파티 라이브러리를 다운로드하고 통합할 수 있다
3. CTest와 Google Test로 테스트를 구성하고, CPack으로 빌드를 패키징할 수 있다
4. 재현 가능하고 공유 가능한 빌드 구성을 위해 CMake 프리셋을 적용할 수 있다
5. 인기 라이브러리(Boost, fmt, spdlog, nlohmann/json, Eigen)를 C++ 프로젝트에 통합할 수 있다
6. 패키지 매니저, `FetchContent`, 수동 의존성 관리 간의 트레이드오프를 평가할 수 있다

---

C++는 프로그래밍에서 가장 풍부한 라이브러리 생태계 중 하나를 갖고 있지만, 역사적으로 의존성 관리가 가장 약한 점이었습니다. 현대 도구 -- vcpkg, Conan, CMake의 `FetchContent`, CMake 프리셋 -- 이 이 문제를 대부분 해결했습니다. 이 레슨은 알아야 할 라이브러리와 이를 실제 프로젝트에서 실용적으로 사용하게 하는 빌드 시스템 기법을 모두 다룹니다. 이 도구들을 마스터하는 것이 소스 파일 모음을 이식 가능하고, 테스트 가능하고, 출하 가능한 소프트웨어로 변환하는 것입니다.

---

## 목차

1. [패키지 매니저: vcpkg와 Conan](#1-패키지-매니저-vcpkg와-conan)
2. [CMake FetchContent](#2-cmake-fetchcontent)
3. [CTest와 테스트 통합](#3-ctest와-테스트-통합)
4. [CPack과 패키징](#4-cpack과-패키징)
5. [CMake 프리셋](#5-cmake-프리셋)
6. [인기 라이브러리](#6-인기-라이브러리)

---

## 1. 패키지 매니저: vcpkg와 Conan

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

**매니페스트 모드** (권장): 프로젝트 루트에 `vcpkg.json`을 생성합니다:

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

| 기능 | vcpkg | Conan |
|------|-------|-------|
| 패키지 수 | ~2,500 | ~1,700 |
| 바이너리 캐싱 | 예 | 예 |
| 커스텀 저장소 | 포트 오버레이 | 리모트 |
| CMake 통합 | 툴체인 파일 | 제너레이터 |
| 플랫폼 지원 | Windows/Linux/macOS | 크로스 플랫폼 + 임베디드 |
| 재현성 | 매니페스트 + 베이스라인 | 락파일 |

---

## 2. CMake FetchContent

`FetchContent`는 CMake 구성 시점에 의존성을 다운로드하고 빌드합니다 -- 외부 패키지 매니저가 필요 없습니다.

### 기본 사용법

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

### FetchContent 동작 제어

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

### FetchContent vs 패키지 매니저

| 시나리오 | 권장 접근법 |
|---------|------------|
| 다수의 의존성이 있는 CI/CD 파이프라인 | vcpkg 매니페스트 모드 |
| 빠른 프로토타입 | FetchContent |
| 대규모 엔터프라이즈 프로젝트 | Conan + CMake |
| 헤더 전용 라이브러리 | 직접 include 또는 FetchContent |
| 시스템 의존성 (OpenSSL 등) | `find_package` |

---

## 3. CTest와 테스트 통합

### 기본 CTest

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

### Google Test와 함께

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

### 테스트 작성

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

### 테스트 실행

```bash
# 빌드 및 모든 테스트 실행
cmake --build build
cd build && ctest --output-on-failure

# 자세한 출력
ctest -V

# 이름 정규식으로 특정 테스트 실행
ctest -R MathTest

# 병렬 테스트 실행
ctest -j$(nproc)
```

---

## 4. CPack과 패키징

CPack은 CMake 프로젝트에서 배포 가능한 패키지(DEB, RPM, ZIP, NSIS, DMG)를 생성합니다.

### 기본 CPack 구성

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

### 패키지 빌드

```bash
# 프로젝트 빌드
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 패키지 생성
cd build
cpack -G ZIP     # ZIP archive
cpack -G DEB     # Debian package
cpack -G RPM     # RPM package
cpack -G NSIS    # Windows installer
cpack -G DragNDrop  # macOS DMG
```

---

## 5. CMake 프리셋

CMake 프리셋(CMake 3.19에서 도입)은 빌드 구성을 공유하는 표준화된 방법을 제공합니다.

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

### 프리셋 사용

```bash
# 사용 가능한 프리셋 나열
cmake --list-presets

# 프리셋으로 구성
cmake --preset dev

# 프리셋으로 빌드
cmake --build --preset dev

# 프리셋으로 테스트
ctest --preset dev
```

### 사용자 로컬 프리셋

개인 오버라이드를 위해 `CMakeUserPresets.json`을 생성합니다 (gitignore에 추가):

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

## 6. 인기 라이브러리

### Boost

C++의 "만능 도구" -- 160+ 피어 리뷰된 라이브러리. 많은 것이 결국 표준의 일부가 됩니다.

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

직관적이고 STL과 유사한 JSON 라이브러리:

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

// 자동 직렬화
struct Person {
    std::string name;
    int age;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Person, name, age)
};
```

### fmt

`std::format`에 영감을 준 라이브러리 -- 추가 기능으로 여전히 널리 사용됩니다:

```cpp
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

void demo() {
    fmt::println("x = {}, y = {:.2f}", 42, 3.14159);

    // 컨테이너 포매팅
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

fmt 기반의 빠른 구조화된 로깅:

```cpp
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>

void demo() {
    spdlog::info("Welcome to spdlog!");
    spdlog::warn("Easy padding: {:08d}", 42);
    spdlog::error("Error code: {:#x}", 0xDEAD);

    // 로테이션이 있는 파일 로거
    auto logger = spdlog::rotating_logger_mt(
        "file_logger", "logs/app.log",
        1024 * 1024 * 5, 3);
    logger->info("Logged to file");

    // 커스텀 패턴
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
}
```

### Eigen

선형 대수 라이브러리 -- 과학 컴퓨팅의 표준 선택:

```cpp
#include <Eigen/Dense>
#include <iostream>

void demo() {
    // 행렬 연산
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 10;

    Eigen::Vector3d b(3, 3, 4);

    // Ax = b 풀기
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
    std::cout << "Solution:\n" << x << "\n";

    // 고유값
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(A);
    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n";
}
```

### 모든 라이브러리의 CMake 통합

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

### 라이브러리 선택 가이드

| 필요 | 권장 라이브러리 |
|------|----------------|
| JSON | nlohmann/json (편의성) 또는 simdjson (속도) |
| HTTP 클라이언트 | cpp-httplib (간단) 또는 Boost.Beast (풀) |
| 로깅 | spdlog |
| 문자열 포매팅 | fmt (또는 C++20이면 std::format) |
| 선형 대수 | Eigen |
| 테스팅 | Google Test 또는 Catch2 |
| CLI 파싱 | CLI11 또는 cxxopts |
| 비동기 I/O / 네트워킹 | Boost.Asio |

---

## 연습 문제

### 연습 1: JSON 구성 시스템

nlohmann/json을 사용하여 구성 시스템을 구축하세요:
1. 중첩 설정을 가진 `Config` 구조체 정의
2. 검증과 함께 JSON 파일에서 로드
3. 누락된 필드에 대한 기본값 지원
4. 구성 병합 구현 (파일 기본값 + 사용자 오버라이드)

### 연습 2: CMake 다중 라이브러리 프로젝트

다음과 같은 CMake 프로젝트를 만드세요:
1. nlohmann/json과 fmt에 `FetchContent` 사용
2. Boost에 `find_package` 사용
3. 라이브러리 타겟과 실행 파일 타겟 포함
4. `gtest_discover_tests`를 사용한 Google Test 포함
5. Debug, Release, CI 빌드를 위한 프리셋 포함

### 연습 3: 커스텀 fmt 포매터

다음에 대한 커스텀 `fmt` 포매터를 만드세요:
1. `Matrix` 클래스 (정렬된 예쁜 출력)
2. `Duration` 클래스 ("2h 30m 15s" 형식)
3. `Color` 클래스 (hex "#RRGGBB" 또는 RGB "(r, g, b)" 형식)

### 연습 4: 로깅 라이브러리

spdlog를 백엔드로 사용하여 로깅 라이브러리를 구축하세요:
1. 지연 초기화를 가진 싱글톤 로거
2. 콘솔 + 로테이팅 파일 싱크
3. 구조화된 로그 필드 (key=value 쌍)
4. 소비자가 링크할 수 있는 CMake 라이브러리 타겟

### 연습 5: 과학 계산기

행렬 연산에 Eigen, 입력에 nlohmann/json을 사용하여 과학 계산기를 구축하세요:
1. JSON 파일에서 행렬 연산 읽기
2. 지원: 덧셈, 곱셈, 역행렬, 고유값 분해
3. JSON 형식으로 결과 출력
4. CPack으로 ZIP 배포로 패키징

---

## 다음 단계

C++ Advanced를 완료한 것을 축하합니다! 이제 이동 의미론과 템플릿 메타프로그래밍에서 C++20/23 기능, 디자인 패턴, 빌드 시스템에 이르기까지 현대 C++에 대한 깊은 전문성을 갖추었습니다. GPU 프로그래밍을 위한 [CUDA](../CUDA/00_Overview.md)나 딥러닝을 처음부터 구축하는 [DL_Scratch_C](../DL_Scratch_C/00_Overview.md)를 탐구하세요.
