# 환경 설정과 첫 번째 프로그램

**이전**: [C++ 기초](./00_Overview.md) | **다음**: [변수와 타입](./02_Variables_and_Types.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C++이 무엇인지 설명하고 주요 특징과 버전 역사를 파악한다
2. Windows, macOS 또는 Linux에 C++ 개발 환경을 설치한다
3. g++을 사용하여 "Hello, World!" 프로그램을 구현하고 컴파일한다
4. `-std=c++17`, `-Wall`, `-Wextra` 등 주요 컴파일러 옵션을 적용한다
5. 기본 입출력 연산에서 `std::cout`과 `std::cin`을 구분한다
6. I/O, 메모리 관리, 문자열 처리에서 C와 C++ 접근 방식을 비교한다
7. VS Code 또는 Visual Studio를 C++ 개발용으로 설정한다

---

모든 프로그래밍 여정은 작동하는 도구 체인에서 시작됩니다. 클래스, 템플릿, 표준 라이브러리를 탐구하기 전에, 소스 코드를 실행 파일로 변환하는 컴파일러와 편안하게 코드를 작성할 편집기가 필요합니다. 이 레슨에서는 이 필수적인 첫 단계를 안내하고, 어떤 언어에서든 가장 만족스러운 이정표인 첫 번째 프로그램 실행을 보상으로 드립니다.

## 1. C++란 무엇인가?

C++는 1979년 Bjarne Stroustrup이 C 언어를 확장하여 개발한 범용 프로그래밍 언어입니다.

### C++의 특징

| 특징 | 설명 |
|------|------|
| 객체지향(Object-Oriented) | 클래스, 상속, 다형성 지원 |
| 고성능(High Performance) | 하드웨어에 가까운 저수준 제어 |
| 다중 패러다임(Multi-Paradigm) | 절차적, 객체지향, 함수형 프로그래밍 |
| 호환성(Compatibility) | C 코드와 대부분 호환 |
| STL | 강력한 표준 템플릿 라이브러리 |

### C++ 버전 역사

```
C++98 ──> C++03 ──> C++11 ──> C++14 ──> C++17 ──> C++20 ──> C++23
 |                     |
 |                     +-- "모던 C++"의 시작
 +-- 최초 표준
```

---

## 2. 개발 환경 설치

### Windows

**방법 1: MinGW-w64 (권장)**

1. [MSYS2](https://www.msys2.org/) 설치
2. MSYS2 터미널에서 실행:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc
   ```
3. PATH 환경 변수에 추가: `C:\msys64\ucrt64\bin`

**방법 2: Visual Studio**

1. [Visual Studio Community](https://visualstudio.microsoft.com/) 설치
2. "C++를 사용한 데스크톱 개발" 워크로드 선택

### macOS

```bash
# Xcode 명령줄 도구 설치
xcode-select --install

# 또는 Homebrew로 GCC 설치
brew install gcc
```

### Linux (Ubuntu/Debian)

```bash
# GCC 설치
sudo apt update
sudo apt install g++ build-essential

# 버전 확인
g++ --version
```

### Linux (CentOS/RHEL)

```bash
# GCC 설치
sudo dnf install gcc-c++

# 버전 확인
g++ --version
```

---

## 3. 첫 번째 프로그램: Hello World

### 코드 작성

`hello.cpp` 파일을 생성합니다:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### 코드 설명

```cpp
#include <iostream>    // I/O 라이브러리 포함
                       // <>는 표준 라이브러리를 의미

int main() {           // 프로그램 진입점
                       // int는 반환 타입

    std::cout          // 표준 출력 (콘솔)
              << "Hello, World!"  // 출력 연산자로 문자열 전송
              << std::endl;       // 줄바꿈 + 버퍼 비우기

    return 0;          // 0 반환 = 정상 종료
}
```

### 컴파일과 실행

```bash
# 컴파일
g++ hello.cpp -o hello

# 실행
./hello      # Linux/macOS
hello.exe    # Windows
```

출력:
```
Hello, World!
```

### 컴파일러 옵션

| 옵션 | 설명 |
|------|------|
| `-o filename` | 출력 파일 이름 지정 |
| `-std=c++17` | C++ 표준 버전 지정 |
| `-Wall` | 모든 경고 활성화 |
| `-Wextra` | 추가 경고 활성화 |
| `-g` | 디버깅 정보 포함 |

```bash
# 권장 컴파일 명령
g++ -std=c++17 -Wall -Wextra hello.cpp -o hello
```

---

## 4. 기본 입출력

### 출력: std::cout

```cpp
#include <iostream>

int main() {
    // 문자열 출력
    std::cout << "Hello" << std::endl;

    // 여러 값 출력
    std::cout << "Number: " << 42 << std::endl;

    // 여러 줄 출력
    std::cout << "Line 1\n"
              << "Line 2\n"
              << "Line 3" << std::endl;

    return 0;
}
```

### 입력: std::cin

```cpp
#include <iostream>

int main() {
    int age;
    std::cout << "Enter your age: ";
    std::cin >> age;
    std::cout << "You are " << age << " years old." << std::endl;

    return 0;
}
```

### 문자열 입력

```cpp
#include <iostream>
#include <string>

int main() {
    std::string name;

    std::cout << "Enter your name: ";
    std::cin >> name;  // 공백까지 읽음

    std::cout << "Hello, " << name << "!" << std::endl;

    return 0;
}
```

### 한 줄 전체 읽기

```cpp
#include <iostream>
#include <string>

int main() {
    std::string fullName;

    std::cout << "Enter your name: ";
    std::getline(std::cin, fullName);  // 한 줄 전체 읽기

    std::cout << "Hello, " << fullName << "!" << std::endl;

    return 0;
}
```

---

## 5. using namespace std

매번 `std::`를 입력하지 않으려면:

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello" << endl;  // std:: 생략 가능
    return 0;
}
```

### 고려사항

| 방법 | 장점 | 단점 |
|------|------|------|
| `std::cout` | 이름 충돌 방지 | 타이핑이 많음 |
| `using namespace std;` | 간결함 | 이름 충돌 가능성 |
| `using std::cout;` | 절충안 | 필요한 것만 선언 |

**권장**: 헤더 파일에서는 `std::`를 명시적으로 사용하고, 소스 파일에서만 `using`을 사용합니다.

---

## 6. 주석

```cpp
#include <iostream>

int main() {
    // 한 줄 주석

    /*
     * 여러 줄 주석
     * 블록 주석이라고도 함
     */

    std::cout << "Hello" << std::endl;  // 코드 뒤 주석

    return 0;
}
```

---

## 7. IDE 설정

### VS Code

1. C/C++ 확장 프로그램 설치 (Microsoft)
2. Code Runner 확장 프로그램 설치 (선택사항)
3. `tasks.json` 설정:

```json
{
    "version": "2.0.0",
    "tasks": [{
        "label": "C++ Build",
        "type": "shell",
        "command": "g++",
        "args": [
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "${file}",
            "-o",
            "${fileDirname}/${fileBasenameNoExtension}"
        ],
        "group": {
            "kind": "build",
            "isDefault": true
        }
    }]
}
```

### Visual Studio

1. 파일 -> 새 프로젝트 -> 콘솔 앱
2. 자동으로 빌드/실행

---

## 8. C와 C++의 차이점

### 헤더 파일

```cpp
// C 스타일 (사용 가능하지만 권장하지 않음)
#include <stdio.h>
#include <stdlib.h>

// C++ 스타일 (권장)
#include <cstdio>    // C 헤더의 C++ 버전
#include <cstdlib>
#include <iostream>  // C++ 전용
```

### I/O 비교

```cpp
// C 스타일
#include <cstdio>

int main() {
    int num;
    printf("Number: ");
    scanf("%d", &num);
    printf("Input: %d\n", num);
    return 0;
}
```

```cpp
// C++ 스타일
#include <iostream>

int main() {
    int num;
    std::cout << "Number: ";
    std::cin >> num;
    std::cout << "Input: " << num << std::endl;
    return 0;
}
```

### 주요 차이점

| 항목 | C | C++ |
|------|---|-----|
| I/O | printf/scanf | cout/cin |
| 메모리 | malloc/free | new/delete |
| 문자열 | char[] | std::string |
| bool | 없음 (int 사용) | bool 타입 |
| 오버로딩 | 불가능 | 가능 |
| 클래스 | 구조체만 | 클래스 지원 |

---

## 9. 실습 예제

### 간단한 계산기

```cpp
#include <iostream>

int main() {
    double num1, num2;
    char op;

    std::cout << "First number: ";
    std::cin >> num1;

    std::cout << "Operator (+, -, *, /): ";
    std::cin >> op;

    std::cout << "Second number: ";
    std::cin >> num2;

    double result;
    switch (op) {
        case '+': result = num1 + num2; break;
        case '-': result = num1 - num2; break;
        case '*': result = num1 * num2; break;
        case '/': result = num1 / num2; break;
        default:
            std::cout << "Invalid operator." << std::endl;
            return 1;
    }

    std::cout << num1 << " " << op << " " << num2
              << " = " << result << std::endl;

    return 0;
}
```

실행:
```
First number: 10
Operator (+, -, *, /): +
Second number: 5
10 + 5 = 15
```

---

## 10. 요약

| 개념 | 설명 |
|------|------|
| `#include` | 헤더 파일 포함 |
| `main()` | 프로그램 진입점 |
| `std::cout` | 표준 출력 |
| `std::cin` | 표준 입력 |
| `std::endl` | 줄바꿈 + 버퍼 비우기 |
| `\n` | 줄바꿈 문자 |
| `g++` | GNU C++ 컴파일러 |

---

## 연습문제

### 연습문제 1: Hello 커스터마이징

`std::cout`을 사용하여 이름, 나이, 좋아하는 프로그래밍 언어를 각각 별도의 줄에 출력하는 프로그램을 작성하세요.

### 연습문제 2: 컴파일러 플래그

`-Wall -Wextra`로 프로그램을 컴파일하고 의도적으로 경고를 발생시키세요 (예: 사용하지 않는 변수). 컴파일러 출력을 관찰하고 경고를 수정하세요.

### 연습문제 3: 입력 에코

사용자에게 이름과 출생 연도를 물어본 후, 대략적인 나이를 포함한 인사말을 출력하는 프로그램을 작성하세요.

### 연습문제 4: C vs C++ I/O

섹션 9의 간단한 계산기를 `cout`/`cin` 대신 C 스타일 `printf`/`scanf`로 다시 작성하세요. 두 버전을 비교하세요.

### 연습문제 5: 여러 줄 출력

`std::cout` 문과 `\n` 이스케이프 시퀀스만 사용하여 (`std::endl` 없이) `*` 문자로 구성된 상자 (가로 5, 세로 3)를 출력하는 프로그램을 작성하세요.

---

## 다음 단계

[02_Variables_and_Types.md](./02_Variables_and_Types.md)에서 변수와 타입에 대해 알아봅시다!
