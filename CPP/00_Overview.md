# C++ 학습 가이드

## 소개

이 폴더는 C++ 프로그래밍을 처음부터 체계적으로 학습하기 위한 자료를 담고 있습니다. 기초 문법부터 모던 C++까지 단계별로 학습할 수 있습니다.

**대상 독자**: 프로그래밍 입문자 ~ 중급자

---

## 학습 로드맵

```
[입문]              [기초]              [중급]              [고급]
  │                   │                   │                   │
  ▼                   ▼                   ▼                   ▼
환경설정 ─────▶ 함수 ─────────▶ 클래스기초 ────▶ 예외처리
  │                   │                   │                   │
  ▼                   ▼                   ▼                   ▼
변수/타입 ────▶ 배열/문자열 ──▶ 클래스심화 ────▶ 스마트포인터
  │                   │                   │                   │
  ▼                   ▼                   ▼                   ▼
제어문 ───────▶ 포인터/참조 ──▶ 상속/다형성 ───▶ 모던C++
                                          │
                                          ▼
                                    STL ─────▶ 템플릿
```

---

## 선수 지식

- 기본적인 컴퓨터 사용법
- 터미널/명령 프롬프트 사용 경험 (권장)

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Environment_Setup.md](./01_Environment_Setup.md) | ⭐ | 개발 환경, Hello World |
| [02_Variables_and_Types.md](./02_Variables_and_Types.md) | ⭐ | 기본 타입, 상수, 형변환 |
| [03_Operators_and_Control_Flow.md](./03_Operators_and_Control_Flow.md) | ⭐ | 연산자, if/switch, 반복문 |
| [04_Functions.md](./04_Functions.md) | ⭐⭐ | 함수 정의, 오버로딩, 기본값 |
| [05_Arrays_and_Strings.md](./05_Arrays_and_Strings.md) | ⭐⭐ | 배열, C문자열, string |
| [06_Pointers_and_References.md](./06_Pointers_and_References.md) | ⭐⭐ | 포인터, 참조자, 동적 메모리 |
| [07_Classes_Basics.md](./07_Classes_Basics.md) | ⭐⭐⭐ | 클래스, 생성자, 소멸자 |
| [08_Classes_Advanced.md](./08_Classes_Advanced.md) | ⭐⭐⭐ | 연산자 오버로딩, 복사/이동 |
| [09_Inheritance_and_Polymorphism.md](./09_Inheritance_and_Polymorphism.md) | ⭐⭐⭐ | 상속, 가상함수, 추상클래스 |
| [10_STL_Containers.md](./10_STL_Containers.md) | ⭐⭐⭐ | vector, map, set |
| [11_STL_Algorithms_Iterators.md](./11_STL_Algorithms_Iterators.md) | ⭐⭐⭐ | algorithm, iterator |
| [12_Templates.md](./12_Templates.md) | ⭐⭐⭐ | 함수/클래스 템플릿 |
| [13_Exceptions_and_File_IO.md](./13_Exceptions_and_File_IO.md) | ⭐⭐⭐⭐ | try/catch, fstream |
| [14_Smart_Pointers_Memory.md](./14_Smart_Pointers_Memory.md) | ⭐⭐⭐⭐ | unique_ptr, shared_ptr |
| [15_Modern_CPP.md](./15_Modern_CPP.md) | ⭐⭐⭐⭐ | C++11/14/17/20 기능 |
| [16_Multithreading_Concurrency.md](./16_Multithreading_Concurrency.md) | ⭐⭐⭐⭐ | std::thread, mutex, async/future |
| [17_CPP20_Advanced.md](./17_CPP20_Advanced.md) | ⭐⭐⭐⭐⭐ | Concepts, Ranges, Coroutines |
| [18_Design_Patterns.md](./18_Design_Patterns.md) | ⭐⭐⭐⭐ | Singleton, Factory, Observer, CRTP |

---

## 추천 학습 순서

### 입문 (프로그래밍 첫걸음)
1. 환경설정과 첫프로그램 → 변수와 자료형 → 연산자와 제어문

### 기초 (핵심 문법)
2. 함수 → 배열과 문자열 → 포인터와 참조

### 중급 (객체지향/STL)
3. 클래스 기초 → 클래스 심화 → 상속과 다형성
4. STL 컨테이너 → STL 알고리즘과 반복자 → 템플릿

### 고급 (전문가 수준)
5. 예외처리와 파일입출력 → 스마트포인터와 메모리 → 모던C++

### 심화 (전문가)
6. 멀티스레딩과 동시성 → C++20 심화 → 디자인패턴

---

## 실습 환경

```bash
# 컴파일러 버전 확인
g++ --version

# C++17 표준으로 컴파일
g++ -std=c++17 -Wall -Wextra program.cpp -o program

# 실행
./program
```

### 권장 도구
- **컴파일러**: g++ (GCC), clang++
- **IDE**: VS Code + C/C++ 확장, CLion, Visual Studio
- **빌드 시스템**: CMake (대규모 프로젝트용)

---

## 관련 자료

- [C_Programming/](../C_Programming/00_Overview.md) - C 언어 기초
- [Linux/](../Linux/00_Overview.md) - 리눅스 개발 환경
- [Python/](../Python/00_Overview.md) - 다른 언어와 비교
