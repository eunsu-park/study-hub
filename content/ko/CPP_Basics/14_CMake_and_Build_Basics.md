# CMake와 빌드 기초

**이전**: [예외와 파일 I/O](./13_Exceptions_and_File_IO.md) | **다음**: [프로젝트: 학생 관리 시스템](./15_Project_Student_Management.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수동 컴파일이 확장되지 않는 이유와 CMake가 의존성 추적, 플랫폼 차이, 재현성을 어떻게 해결하는지 설명한다
2. `cmake_minimum_required`, `project`, `add_executable`로 최소 `CMakeLists.txt`를 작성한다
3. `add_library`로 라이브러리 타겟을 만들고 `target_link_libraries`로 올바른 가시성(`PUBLIC`, `PRIVATE`, `INTERFACE`)으로 링크한다
4. `target_compile_options`와 생성기 표현식으로 컴파일러 경고와 C++ 표준 플래그를 설정한다
5. 별도의 소스와 인클루드 디렉토리로 간단한 다중 파일 프로젝트를 구성한다
6. `CMAKE_BUILD_TYPE`을 사용하여 Debug와 Release 빌드를 설정한다

---

여러분의 컴퓨터에서 컴파일되는 C++ 프로그램도 팀원, CI 서버, 또는 다른 OS의 미래의 자신이 안정적으로 빌드할 수 없다면 의미가 없습니다. CMake는 플랫폼별 도구 체인 세부 사항을 추상화하면서 타겟, 의존성, 테스트에 대한 세밀한 제어를 제공하기 때문에 C++ 빌드 설정의 업계 표준이 되었습니다.

---

## 1. 빌드 시스템이 필요한 이유

단일 파일 프로그램 이상에서는 수동으로 `g++`를 실행하는 것이 비실용적입니다:

```bash
# 이것은 확장되지 않습니다
g++ -std=c++17 -Wall -I./include \
    src/main.cpp src/math.cpp src/utils.cpp \
    -lsqlite3 -lpthread -o myapp
```

문제점:
- **의존성 추적**: 어떤 파일이 변경되었나? 무엇을 다시 컴파일해야 하나?
- **순서**: 라이브러리는 오브젝트 파일 뒤에 링크되어야 함
- **플랫폼 차이**: Linux vs macOS vs Windows 플래그가 다름
- **재현성**: 모든 개발자가 같은 플래그를 사용해야 함

### 빌드 시스템 현황

| 도구 | 유형 | 설명 |
|------|------|------|
| Make | 빌드 도구 | 규칙 기반, UNIX 중심 |
| CMake | 메타 빌드 시스템 | Makefile, Ninja, VS 솔루션 생성 |
| Meson | 메타 빌드 시스템 | Python 기반, 빠름 |
| Bazel | 빌드 시스템 | Google, 밀폐형 빌드 |
| Ninja | 빌드 도구 | 저수준, 생성기용 설계 |

**CMake**가 C++ 프로젝트의 사실상 표준입니다.

---

## 2. 최소 CMakeLists.txt

```cmake
# 필요한 최소 CMake 버전
cmake_minimum_required(VERSION 3.16)

# 프로젝트 이름, 버전, 언어
project(MyApp VERSION 1.0.0 LANGUAGES CXX)

# 실행 파일 타겟 생성
add_executable(myapp src/main.cpp)
```

### 빌드 명령

```bash
# 설정 (빌드 파일 생성)
cmake -B build

# 빌드
cmake --build build

# 실행
./build/myapp
```

---

## 3. 프로젝트 구조

일반적인 C++ 프로젝트 레이아웃:

```
myproject/
├── CMakeLists.txt          # 루트 CMake 파일
├── src/
│   ├── main.cpp
│   ├── math.cpp
│   └── math.hpp
├── include/
│   └── myproject/
│       └── utils.hpp       # 공개 헤더
└── build/                  # 소스 밖 빌드 디렉토리
```

---

## 4. 타겟, 프로퍼티, 모던 CMake

모던 CMake는 **타겟 기반**입니다 -- 모든 컴파일 플래그, 인클루드 경로, 의존성이 타겟에 부착됩니다.

### 4.1 실행 파일과 라이브러리 타겟

```cmake
cmake_minimum_required(VERSION 3.16)
project(Calculator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 라이브러리 생성
add_library(mathlib src/math.cpp src/utils.cpp)

# 라이브러리의 인클루드 디렉토리 지정
target_include_directories(mathlib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include    # mathlib 사용자도 볼 수 있음
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src        # mathlib 자체만 볼 수 있음
)

# 라이브러리를 사용하는 실행 파일 생성
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE mathlib)
```

### 4.2 PUBLIC, PRIVATE, INTERFACE

| 키워드 | 이 타겟 | 이 타겟의 소비자 |
|--------|--------|----------------|
| PUBLIC | 예 | 예 |
| PRIVATE | 예 | 아니오 |
| INTERFACE | 아니오 | 예 |

---

## 5. 컴파일러 경고와 플래그

```cmake
# 특정 타겟에 경고 추가
target_compile_options(calculator PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
```

### 빌드 타입 설정

```bash
# Debug 빌드
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release 빌드
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

---

## 6. 외부 라이브러리 찾기

```cmake
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

target_link_libraries(myapp PRIVATE Threads::Threads SQLite::SQLite3)
```

---

## 7. 헤더 전용 라이브러리

```cmake
add_library(myheaders INTERFACE)
target_include_directories(myheaders INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(consumer PRIVATE myheaders)
```

---

## 8. 완전한 기본 예제

```cmake
cmake_minimum_required(VERSION 3.16)
project(Calculator VERSION 1.0.0 DESCRIPTION "A simple calculator library" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── 라이브러리 ──────────────────────────
add_library(calclib src/calculator.cpp)
target_include_directories(calclib
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(calclib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)

# ── 실행 파일 ───────────────────────────
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE calclib)
```

---

## 9. 요약

| 개념 | 모던 CMake 방식 |
|------|----------------|
| 인클루드 경로 | `target_include_directories()` |
| 컴파일 플래그 | `target_compile_options()` |
| 링킹 | `target_link_libraries()` |
| C++ 표준 | `set(CMAKE_CXX_STANDARD 17)` |
| 의존성 | `find_package()` |
| 빌드 타입 | `-DCMAKE_BUILD_TYPE=Release` |

**피해야 할 안티 패턴:**
- `include_directories()` -- 대신 `target_include_directories()` 사용
- `link_libraries()` -- 대신 `target_link_libraries()` 사용
- `add_compile_options()` -- 대신 `target_compile_options()` 사용
- 소스 내 빌드 -- 항상 `cmake -B build` 사용

---

## 실습 연습문제

### 연습문제 1: 다중 파일 프로젝트 빌드
`stringutils` 라이브러리와 이를 사용하는 `main.cpp`로 프로젝트를 만들고 적절한 `CMakeLists.txt`를 작성하세요.

### 연습문제 2: 인클루드 디렉토리 추가
헤더 파일을 `include/` 디렉토리로 이동하고 `target_include_directories`와 `PUBLIC`을 사용하세요.

### 연습문제 3: 크로스 플랫폼 경고
생성기 표현식으로 컴파일러별 경고를 설정하고 Debug와 Release 빌드를 모두 지원하세요.

---

고급 CMake 기능(FetchContent, CTest, 패키징)은 [외부 라이브러리와 빌드](../CPP_Advanced/17_External_Libraries_and_Build.md)를 참조하세요.

---

## 다음 단계

모든 것을 캡스톤 프로젝트에 종합합시다: [프로젝트: 학생 관리 시스템](./15_Project_Student_Management.md)!
