# 레슨 20: CMake와 빌드 시스템(Build Systems)

**이전**: [프로젝트: 학생 관리 시스템](./19_Project_Student_Management.md) | **다음**: [C++23 기능](./21_CPP23_Features.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수동 컴파일이 확장되지 않는 이유와 CMake가 의존성 추적(dependency tracking), 플랫폼 차이, 재현성(reproducibility) 문제를 해결하는 방법을 설명한다
2. `cmake_minimum_required`, `project`, `add_executable`을 사용하여 최소한의 `CMakeLists.txt`를 작성한다
3. `add_library`로 라이브러리 타깃(target)을 생성하고, 올바른 가시성(`PUBLIC`, `PRIVATE`, `INTERFACE`)으로 `target_link_libraries`를 사용해 연결한다
4. `target_compile_options`와 생성기 표현식(generator expression)을 사용해 컴파일러 경고와 C++ 표준 플래그를 설정한다
5. `find_package`와 `FetchContent`를 통해 외부 의존성을 통합한다
6. CTest와 Google Test로 테스트를 구성하고, 커맨드 라인에서 테스트를 실행한다
7. 이식 가능하고 프로덕션에 적합한 빌드를 위한 설치 규칙과 빌드 타입 구성을 정의한다

---

여러분의 컴퓨터에서 컴파일되는 C++ 프로그램은 팀원, CI 서버, 또는 다른 OS를 사용하는 미래의 자신이 안정적으로 빌드할 수 없다면 의미가 없습니다. CMake가 C++ 빌드 구성의 업계 표준이 된 이유는 정확히, 플랫폼별 툴체인(toolchain) 세부 사항을 추상화하면서도 타깃, 의존성, 테스트에 대한 세밀한 제어권을 제공하기 때문입니다. CMake를 마스터하는 것은 C++을 배우는 길에서 벗어난 우회로가 아닙니다 — 그것은 코드를 출시 가능한 소프트웨어로 만들어주는 기술입니다.

---

## 1. 빌드 시스템이 필요한 이유

단일 파일 프로그램을 넘어서는 규모의 프로젝트에서는 `g++`을 직접 실행하는 방식이 금방 한계에 부딪힙니다.

```bash
# This doesn't scale
g++ -std=c++17 -Wall -I./include \
    src/main.cpp src/math.cpp src/utils.cpp \
    -lsqlite3 -lpthread -o myapp
```

주요 문제점:
- **의존성 추적(Dependency tracking)**: 어떤 파일이 변경되었는가? 무엇을 다시 컴파일해야 하는가?
- **빌드 순서(Ordering)**: 라이브러리는 오브젝트 파일(object file) 이후에 링크(link)되어야 함
- **플랫폼 차이(Platform differences)**: Linux, macOS, Windows 간 컴파일 플래그(flag)가 다름
- **재현성(Reproducibility)**: 모든 개발자가 동일한 플래그를 사용해야 함

### 빌드 시스템 종류

| 도구 | 종류 | 설명 |
|------|------|-------------|
| Make | 빌드 도구(Build tool) | 규칙(rule) 기반, UNIX 중심 |
| CMake | 메타 빌드 시스템(Meta-build system) | Makefile, Ninja, VS 솔루션 생성 |
| Meson | 메타 빌드 시스템(Meta-build system) | Python 기반, 빠른 속도 |
| Bazel | 빌드 시스템(Build system) | Google 개발, 허메틱 빌드(hermetic build) |
| Ninja | 빌드 도구(Build tool) | 저수준, 생성기(generator)용으로 설계 |

**CMake**는 C++ 프로젝트의 사실상 표준(de facto standard)입니다.

---

## 2. 최소한의 CMakeLists.txt

```cmake
# Minimum CMake version required
cmake_minimum_required(VERSION 3.16)

# Project name, version, and languages
project(MyApp VERSION 1.0.0 LANGUAGES CXX)

# Create an executable target
add_executable(myapp src/main.cpp)
```

### 빌드 명령어

```bash
# Configure (generates build files)
cmake -B build

# Build
cmake --build build

# Run
./build/myapp
```

---

## 3. 프로젝트 구조(Project Structure)

일반적인 C++ 프로젝트 레이아웃(layout):

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

## 4. 타깃(Targets), 속성(Properties), 모던 CMake

모던 CMake는 **타깃(target) 기반**입니다 — 컴파일 플래그, 인클루드 경로(include path), 의존성 모두가 타깃에 연결됩니다.

### 4.1 실행 파일(Executable)과 라이브러리(Library) 타깃

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

| 키워드 | 이 타깃 | 이 타깃을 사용하는 측 |
|---------|-------------|--------------------------|
| PUBLIC | 적용됨 | 적용됨 |
| PRIVATE | 적용됨 | 적용 안 됨 |
| INTERFACE | 적용 안 됨 | 적용됨 |

```cmake
# mathlib uses Eigen internally (PRIVATE)
# mathlib exposes nlohmann_json in its API (PUBLIC)
target_link_libraries(mathlib
    PRIVATE Eigen3::Eigen
    PUBLIC  nlohmann_json::nlohmann_json
)
```

---

## 5. 컴파일러 경고(Compiler Warnings)와 플래그(Flags)

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

### 빌드 타입(Build Type) 설정

```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

---

## 6. 외부 라이브러리(External Libraries) 찾기

### 6.1 find_package (시스템에 설치된 라이브러리)

```cmake
# Find installed libraries
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

target_link_libraries(myapp PRIVATE
    Threads::Threads
    SQLite::SQLite3
)
```

### 6.2 FetchContent (설정 시 자동 다운로드)

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

## 7. CTest로 테스트하기

### 7.1 기본 CTest 통합

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

### 7.2 Google Test와 함께 사용

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

### 테스트 실행

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

## 8. 헤더 전용 라이브러리(Header-Only Libraries)

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

## 9. 설치 규칙(Install Rules)

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

## 10. 완성된 예제

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

## 연습 문제

### 연습 1: 다중 파일 프로젝트 빌드

다음을 포함하는 프로젝트를 생성하세요:
- `to_upper()`, `to_lower()`, `trim()` 함수가 담긴 `stringutils` 라이브러리 (`src/stringutils.cpp`)
- 해당 라이브러리를 사용하는 `main.cpp`
- `add_library`와 `add_executable`이 포함된 적절한 `CMakeLists.txt`
- 별도의 `build/` 디렉토리에서 빌드

### 연습 2: Google Test 추가

연습 1을 확장하세요:
- `FetchContent`를 통해 Google Test 추가
- 세 가지 문자열 함수에 대한 테스트 작성
- `gtest_discover_tests`로 테스트 등록
- `ctest --output-on-failure`로 검증

### 연습 3: 크로스 플랫폼(Cross-Platform) 빌드

CMakeLists.txt를 다음과 같이 수정하세요:
- 컴파일러별 경고를 위한 생성기 표현식(generator expression) 사용
- Debug 및 Release 빌드 모두 지원
- 테스트 활성화/비활성화를 위한 `option()` 추가
- 실행 파일과 헤더에 대한 `install()` 규칙 추가

---

## 요약

| 개념 | 모던 CMake 관행 |
|---------|----------------------|
| 인클루드 경로(Include paths) | `target_include_directories()` |
| 컴파일 플래그(Compile flags) | `target_compile_options()` |
| 링킹(Linking) | `target_link_libraries()` |
| C++ 표준(C++ standard) | `set(CMAKE_CXX_STANDARD 17)` |
| 의존성(Dependencies) | `find_package()` 또는 `FetchContent` |
| 테스트(Testing) | `enable_testing()` + CTest |
| 빌드 타입(Build type) | `-DCMAKE_BUILD_TYPE=Release` |

**피해야 할 안티패턴(Anti-patterns):**
- `include_directories()` — 대신 `target_include_directories()` 사용
- `link_libraries()` — 대신 `target_link_libraries()` 사용
- `add_compile_options()` — 대신 `target_compile_options()` 사용
- 소스 내 빌드(in-source build) — 항상 `cmake -B build` 사용

---

## 탐색

**이전**: [프로젝트: 학생 관리 시스템](./19_Project_Student_Management.md) | **다음**: [C++23 기능](./21_CPP23_Features.md)
