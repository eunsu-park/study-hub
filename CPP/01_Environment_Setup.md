# 환경설정과 첫 프로그램

## 1. C++란?

C++는 Bjarne Stroustrup이 1979년에 C 언어를 확장하여 개발한 범용 프로그래밍 언어입니다.

### C++의 특징

| 특징 | 설명 |
|------|------|
| 객체지향 | 클래스, 상속, 다형성 지원 |
| 고성능 | 하드웨어에 가까운 저수준 제어 |
| 멀티패러다임 | 절차적, 객체지향, 함수형 프로그래밍 |
| 호환성 | C 코드와 대부분 호환 |
| STL | 강력한 표준 템플릿 라이브러리 |

### C++ 버전 역사

```
C++98 ──▶ C++03 ──▶ C++11 ──▶ C++14 ──▶ C++17 ──▶ C++20 ──▶ C++23
 │                     │
 │                     └── "모던 C++"의 시작
 └── 첫 표준
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
3. 환경 변수 PATH에 추가: `C:\msys64\ucrt64\bin`

**방법 2: Visual Studio**

1. [Visual Studio Community](https://visualstudio.microsoft.com/) 설치
2. "C++를 사용한 데스크톱 개발" 워크로드 선택

### macOS

```bash
# Xcode Command Line Tools 설치
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
#include <iostream>    // 입출력 라이브러리 포함
                       // <> 는 표준 라이브러리를 의미

int main() {           // 프로그램 시작점 (진입점)
                       // int는 정수 반환 타입

    std::cout          // 표준 출력 (콘솔)
              << "Hello, World!"  // 출력 연산자로 문자열 전달
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

### 컴파일 옵션

| 옵션 | 설명 |
|------|------|
| `-o 파일명` | 출력 파일 이름 지정 |
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
    std::cout << "나이를 입력하세요: ";
    std::cin >> age;
    std::cout << "당신은 " << age << "살입니다." << std::endl;

    return 0;
}
```

### 문자열 입력

```cpp
#include <iostream>
#include <string>

int main() {
    std::string name;

    std::cout << "이름을 입력하세요: ";
    std::cin >> name;  // 공백 전까지만 읽음

    std::cout << "안녕하세요, " << name << "님!" << std::endl;

    return 0;
}
```

### 한 줄 전체 입력

```cpp
#include <iostream>
#include <string>

int main() {
    std::string fullName;

    std::cout << "이름을 입력하세요: ";
    std::getline(std::cin, fullName);  // 줄 전체 읽음

    std::cout << "안녕하세요, " << fullName << "님!" << std::endl;

    return 0;
}
```

---

## 5. using namespace std

`std::`를 매번 쓰는 것이 번거롭다면:

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello" << endl;  // std:: 생략 가능
    return 0;
}
```

### 주의사항

| 방법 | 장점 | 단점 |
|------|------|------|
| `std::cout` | 이름 충돌 방지 | 타이핑 많음 |
| `using namespace std;` | 간결함 | 이름 충돌 가능 |
| `using std::cout;` | 절충안 | 필요한 것만 선언 |

**권장**: 헤더 파일에서는 `std::`를 명시적으로 사용하고, 소스 파일에서만 `using`을 사용하세요.

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

1. C/C++ 확장 설치 (Microsoft)
2. Code Runner 확장 설치 (선택)
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

1. 파일 → 새 프로젝트 → 콘솔 앱 선택
2. 자동으로 빌드/실행 가능

---

## 8. C와 C++의 차이

### 헤더 파일

```cpp
// C 스타일 (사용 가능하지만 비권장)
#include <stdio.h>
#include <stdlib.h>

// C++ 스타일 (권장)
#include <cstdio>    // C 헤더의 C++ 버전
#include <cstdlib>
#include <iostream>  // C++ 전용
```

### 입출력 비교

```cpp
// C 스타일
#include <cstdio>

int main() {
    int num;
    printf("숫자: ");
    scanf("%d", &num);
    printf("입력: %d\n", num);
    return 0;
}
```

```cpp
// C++ 스타일
#include <iostream>

int main() {
    int num;
    std::cout << "숫자: ";
    std::cin >> num;
    std::cout << "입력: " << num << std::endl;
    return 0;
}
```

### 주요 차이점

| 항목 | C | C++ |
|------|---|-----|
| 입출력 | printf/scanf | cout/cin |
| 메모리 | malloc/free | new/delete |
| 문자열 | char[] | std::string |
| bool | 없음 (int 사용) | bool 타입 |
| 오버로딩 | 불가 | 가능 |
| 클래스 | 구조체만 | 클래스 지원 |

---

## 9. 실습 예제

### 간단한 계산기

```cpp
#include <iostream>

int main() {
    double num1, num2;
    char op;

    std::cout << "첫 번째 숫자: ";
    std::cin >> num1;

    std::cout << "연산자 (+, -, *, /): ";
    std::cin >> op;

    std::cout << "두 번째 숫자: ";
    std::cin >> num2;

    double result;
    switch (op) {
        case '+': result = num1 + num2; break;
        case '-': result = num1 - num2; break;
        case '*': result = num1 * num2; break;
        case '/': result = num1 / num2; break;
        default:
            std::cout << "잘못된 연산자입니다." << std::endl;
            return 1;
    }

    std::cout << num1 << " " << op << " " << num2
              << " = " << result << std::endl;

    return 0;
}
```

실행:
```
첫 번째 숫자: 10
연산자 (+, -, *, /): +
두 번째 숫자: 5
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
| `std::endl` | 줄바꿈 + 버퍼 플러시 |
| `\n` | 줄바꿈 문자 |
| `g++` | GNU C++ 컴파일러 |

---

## 다음 단계

[02_Variables_and_Types.md](./02_Variables_and_Types.md)에서 C++의 변수와 자료형을 배워봅시다!
