# C++ 고급

## 소개

이 토픽은 고급 C++ 프로그래밍을 다룹니다: 템플릿과 메타프로그래밍, 모던 C++ 표준(C++11부터 C++23까지), 동시성과 멀티스레딩, 디자인 패턴, 빌드 시스템 통합. 이 레슨들은 기본 C++ 지식을 바탕으로 언어의 가장 강력한 기능을 마스터하는 데 초점을 맞춥니다.

**선수 과목**: [CPP 기초](../CPP_Basics/00_Overview.md) (또는 클래스, 상속, STL 컨테이너, 기본 템플릿에 대한 동등한 지식)

---

## 학습 로드맵

```
[템플릿 & 메모리]                  [모던 표준]                      [시스템 & 패턴]
  |                                   |                               |
  v                                   v                               v
이동 의미론 ---------+          모던 C++11/14 ----+            멀티스레딩
  |                  |            |                 |              |
  v                  |            v                 |              v
템플릿               |          모던 C++17          |            고급 동시성
  |                  |            |                 |              |
  v                  |            v                 |              v
템플릿 메타프로그    |          C++20 개념           |            디자인 패턴
  |                  |            |                 |              (생성/구조)
  v                  |            v                 |              |
스마트 포인터        |          C++20 범위           |              v
  & RAII             |            |                 |            디자인 패턴
  |                  |            v                 |              (행위/관용구)
  v                  |          C++20 코루틴        |              |
에러 처리 ----------+            |                 |              v
                                  v                 |            외부 라이브러리
                                모듈 & C++20 -------+              & 빌드
                                  유틸리티
                                  |
                                  v
                                C++23 기능
```

---

## 파일 목록

| # | 제목 | 난이도 | 주요 내용 |
|---|------|--------|----------|
| [01](./01_Move_Semantics_Deep_Dive.md) | 이동 의미론 심화 | ⭐⭐⭐ | rvalue 참조, std::move, 전달, 5의 법칙/0의 법칙 |
| [02](./02_Templates.md) | 템플릿 | ⭐⭐⭐ | 함수/클래스 템플릿, 특수화 |
| [03](./03_Template_Metaprogramming.md) | 템플릿 메타프로그래밍 | ⭐⭐⭐⭐ | SFINAE, type_traits, if constexpr |
| [04](./04_Smart_Pointers_and_RAII.md) | 스마트 포인터와 RAII | ⭐⭐⭐⭐ | unique_ptr, shared_ptr, weak_ptr, RAII |
| [05](./05_Error_Handling_Patterns.md) | 에러 처리 패턴 | ⭐⭐⭐ | noexcept, 예외 안전성, std::expected |
| [06](./06_Modern_CPP_11_14.md) | 모던 C++ (C++11/14) | ⭐⭐⭐ | auto, 람다, constexpr, 균일 초기화 |
| [07](./07_Modern_CPP_17.md) | 모던 C++ (C++17) | ⭐⭐⭐ | 구조적 바인딩, optional/variant/any, 파일시스템 |
| [08](./08_CPP20_Concepts.md) | C++20 개념(Concepts) | ⭐⭐⭐⭐ | concepts, requires, 제약된 auto |
| [09](./09_CPP20_Ranges.md) | C++20 범위(Ranges) | ⭐⭐⭐⭐ | 뷰, 어댑터, 파이프라인 합성 |
| [10](./10_CPP20_Coroutines.md) | C++20 코루틴 | ⭐⭐⭐⭐⭐ | co_await, co_yield, 제너레이터 |
| [11](./11_Modules_and_CPP20_Utilities.md) | 모듈과 C++20 유틸리티 | ⭐⭐⭐ | export/import, std::format, std::span |
| [12](./12_Multithreading.md) | 멀티스레딩 | ⭐⭐⭐⭐ | std::thread, mutex, async/future |
| [13](./13_Concurrency_Advanced.md) | 고급 동시성 | ⭐⭐⭐⭐⭐ | latch/barrier, 락 프리, memory_order |
| [14](./14_Design_Patterns_Creational_Structural.md) | 디자인 패턴 (생성/구조) | ⭐⭐⭐⭐ | SOLID, 싱글턴, 팩토리, 어댑터, 데코레이터 |
| [15](./15_Design_Patterns_Behavioral_Idioms.md) | 디자인 패턴 (행위/관용구) | ⭐⭐⭐⭐ | 옵저버, 전략, CRTP, PIMPL |
| [16](./16_CPP23_Features.md) | C++23 기능 | ⭐⭐⭐⭐⭐ | std::expected, std::print, deducing this |
| [17](./17_External_Libraries_and_Build.md) | 외부 라이브러리와 빌드 | ⭐⭐⭐ | Conan, vcpkg, FetchContent, CTest, Boost/fmt |

---

## 권장 학습 순서

### 경로 1: 템플릿 & 메모리
1. 이동 의미론 심화 -> 템플릿 -> 템플릿 메타프로그래밍 -> 스마트 포인터와 RAII -> 에러 처리 패턴

### 경로 2: 모던 표준
2. 모던 C++11/14 -> 모던 C++17 -> C++20 개념 -> C++20 범위 -> C++20 코루틴 -> 모듈과 C++20 유틸리티 -> C++23 기능

### 경로 3: 시스템 & 패턴
3. 멀티스레딩 -> 고급 동시성 -> 디자인 패턴 (생성/구조) -> 디자인 패턴 (행위/관용구) -> 외부 라이브러리와 빌드

---

## 실습 환경

```bash
# 컴파일러 버전 확인 (대부분의 레슨에 C++20 지원 필요)
g++ --version
clang++ --version

# C++20과 경고 옵션으로 컴파일
g++ -std=c++20 -Wall -Wextra -pedantic -g program.cpp -o program

# C++23 기능으로 컴파일 (GCC 13+ / Clang 17+)
g++ -std=c++23 -Wall -Wextra -pedantic -g program.cpp -o program

# AddressSanitizer로 컴파일
g++ -std=c++20 -fsanitize=address -g program.cpp -o program

# ThreadSanitizer로 컴파일 (동시성 레슨용)
g++ -std=c++20 -fsanitize=thread -g program.cpp -o program -pthread
```

---

## 관련 자료

- [CPP 기초](../CPP_Basics/00_Overview.md) - C++ 기초 (변수, OOP, STL, 기본 템플릿)
- [C_Advanced/](../C_Advanced/00_Overview.md) - 고급 C 프로그래밍 (포인터, 시스템 프로그래밍)
- [Algorithm/](../Algorithm/00_Overview.md) - 자료구조와 알고리즘
- [Software_Engineering/](../Software_Engineering/00_Overview.md) - 소프트웨어 설계 원칙
- [System_Design/](../System_Design/00_Overview.md) - 시스템 아키텍처와 설계
