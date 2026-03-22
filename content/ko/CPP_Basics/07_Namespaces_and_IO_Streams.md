# 네임스페이스와 IO 스트림

**이전**: [포인터와 참조](./06_Pointers_and_References.md) | **다음**: [클래스 기초](./08_Classes_Basics.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 코드를 네임스페이스로 구성하고 이름 충돌을 해결한다
2. iostream과 iomanip을 사용하여 서식 있는 입출력을 수행한다
3. stringstream을 사용하여 문자열을 파싱하고 구성한다
4. `std::getline`으로 전체 줄을 읽고 혼합 입력을 처리한다
5. `using` 지시문과 선언을 안전하게 사용한다

---

프로그램이 몇 개의 파일을 넘어서면 이름 충돌의 위험이 커집니다. 네임스페이스는 식별자를 논리적 그룹으로 분할하여 두 라이브러리가 각각 `log()` 함수를 정의하더라도 충돌하지 않게 합니다. 마찬가지로 유창한 I/O -- 숫자 서식 지정, 열 정렬, 견고한 사용자 입력 읽기 -- 를 아는 것은 거친 프로토타입을 세련되고 전문적인 도구로 변환합니다.

## 1. 네임스페이스(Namespace)

네임스페이스는 식별자를 이름 아래에 그룹화하여 충돌을 방지하는 선언적 영역입니다.

### 네임스페이스 정의

```cpp
#include <iostream>

namespace math {
    double PI = 3.14159265358979;

    double circleArea(double r) {
        return PI * r * r;
    }
}

namespace physics {
    double PI = 3.14159;  // math::PI와 충돌 없음

    double sphereVolume(double r) {
        return (4.0 / 3.0) * PI * r * r * r;
    }
}

int main() {
    std::cout << "Circle area: " << math::circleArea(5.0) << std::endl;
    std::cout << "Sphere volume: " << physics::sphereVolume(5.0) << std::endl;
    return 0;
}
```

### 중첩 네임스페이스

```cpp
// 전통적 문법
namespace company {
    namespace project {
        void init() { /* ... */ }
    }
}

// C++17 축약 문법
namespace company::project {
    void shutdown() { /* ... */ }
}

int main() {
    company::project::init();
    company::project::shutdown();
    return 0;
}
```

### 익명(이름 없는) 네임스페이스

익명 네임스페이스의 식별자는 내부 링크를 가집니다 -- 현재 번역 단위에서만 보이며, 파일 범위의 `static`과 유사합니다.

```cpp
namespace {
    int helperCounter = 0;  // 이 .cpp 파일에서만 보임

    void increment() {
        helperCounter++;
    }
}
```

### using 지시문 vs using 선언

```cpp
#include <iostream>
#include <string>

// using 선언: 하나의 이름만 가져옴
using std::cout;
using std::endl;

int main() {
    cout << "Hello" << endl;  // OK

    // std::string은 선언하지 않았으므로 여전히 std:: 필요
    std::string name = "World";
    cout << name << endl;

    return 0;
}
```

```cpp
// using 지시문: 네임스페이스의 모든 이름을 가져옴
using namespace std;  // std의 모든 것을 가져옴

int main() {
    cout << "Hello" << endl;     // OK
    string name = "World";       // OK
    return 0;
}
```

### 안전성 비교

| 접근 방식 | 범위 | 위험도 |
|-----------|------|--------|
| `std::cout` (완전 한정) | 없음 | 가장 안전 |
| `using std::cout;` (선언) | 하나의 이름만 가져옴 | 낮은 위험 |
| `using namespace std;` (지시문) | 모든 이름 가져옴 | 높은 충돌 위험 |

**경험 법칙**: 헤더 파일에 `using namespace`를 절대 넣지 마세요. 소스 파일에서는 필요한 특정 이름에 대한 `using` 선언을 선호하세요.

---

## 2. 표준 출력

### std::cout과 삽입 연산자

```cpp
#include <iostream>

int main() {
    // 여러 삽입 체인
    std::cout << "Name: " << "Alice" << ", Age: " << 30 << std::endl;

    // '\n' vs std::endl
    std::cout << "Line 1\n";          // 줄바꿈만 (더 빠름)
    std::cout << "Line 2" << std::endl; // 줄바꿈 + 버퍼 비우기

    // 줄바꿈 없이 명시적 비우기
    std::cout << "Processing..." << std::flush;

    return 0;
}
```

### endl vs '\n' 사용 시기

| 방법 | 효과 | 사용 시기 |
|------|------|----------|
| `'\n'` | 줄바꿈 삽입 | 기본 선택 (더 빠름) |
| `std::endl` | 줄바꿈 + 버퍼 비우기 | 출력 보장 필요 시 (디버깅, 로깅) |
| `std::flush` | 버퍼만 비우기 | 진행 표시기 |

고처리량 출력(예: 수백만 줄 출력)에서는 매 줄마다 비우기 오버헤드를 피하기 위해 `'\n'`을 선호하세요.

---

## 3. 표준 입력

### std::cin과 추출 연산자

```cpp
#include <iostream>

int main() {
    int age;
    double height;

    std::cout << "Enter age and height: ";
    std::cin >> age >> height;  // 공백으로 구분된 두 값 읽기

    std::cout << "Age: " << age << ", Height: " << height << std::endl;

    return 0;
}
```

### 입력 실패와 복구

`std::cin`이 예상 타입과 맞지 않는 데이터를 만나면 실패 상태에 들어갑니다.

```cpp
#include <iostream>
#include <limits>

int main() {
    int number;

    while (true) {
        std::cout << "Enter an integer: ";
        if (std::cin >> number) {
            break;  // 성공
        }

        // 입력 실패 (예: 사용자가 "abc" 입력)
        std::cout << "Invalid input. Try again.\n";
        std::cin.clear();  // 실패 플래그 초기화
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // 잘못된 입력 버리기
    }

    std::cout << "You entered: " << number << std::endl;
    return 0;
}
```

### 주요 복구 함수

| 함수 | 용도 |
|------|------|
| `cin.clear()` | 오류 플래그 초기화 |
| `cin.ignore(n, delim)` | `n`개 문자 또는 `delim`까지 버림 |
| `cin.fail()` | 마지막 추출 실패 시 `true` 반환 |
| `cin.good()` | 스트림이 정상 상태면 `true` 반환 |

---

## 4. iomanip을 사용한 서식 있는 출력

`<iomanip>` 헤더는 값의 출력 방식을 제어하는 조작자를 제공합니다.

### 너비와 채우기

```cpp
#include <iostream>
#include <iomanip>

int main() {
    // setw: 최소 필드 너비 (다음 출력에만 적용)
    std::cout << std::setw(10) << 42 << std::endl;       // "        42"
    std::cout << std::setw(10) << "Hello" << std::endl;   // "     Hello"

    // setfill: 채우기에 사용되는 문자
    std::cout << std::setfill('0') << std::setw(5) << 42 << std::endl;  // "00042"
    std::cout << std::setfill('.') << std::setw(20) << "Menu" << std::endl;
    // "................Menu"

    return 0;
}
```

### 정렬

```cpp
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::left  << std::setw(15) << "Name"
              << std::right << std::setw(10) << "Score" << std::endl;
    std::cout << std::left  << std::setw(15) << "Alice"
              << std::right << std::setw(10) << 95 << std::endl;
    std::cout << std::left  << std::setw(15) << "Bob"
              << std::right << std::setw(10) << 87 << std::endl;
    std::cout << std::left  << std::setw(15) << "Charlie"
              << std::right << std::setw(10) << 92 << std::endl;

    return 0;
}
```

출력:
```
Name                 Score
Alice                   95
Bob                     87
Charlie                 92
```

### 부동소수점 정밀도

```cpp
#include <iostream>
#include <iomanip>

int main() {
    double pi = 3.14159265358979;

    // 기본 정밀도 (유효 숫자 6자리)
    std::cout << pi << std::endl;                  // 3.14159

    // setprecision: 유효 숫자 수
    std::cout << std::setprecision(3) << pi << std::endl;  // 3.14

    // fixed: 소수점 이하 자릿수
    std::cout << std::fixed << std::setprecision(2) << pi << std::endl;  // 3.14

    // 과학적 표기법
    std::cout << std::scientific << std::setprecision(4) << pi << std::endl;
    // 3.1416e+00

    // 기본값으로 초기화
    std::cout << std::defaultfloat;

    return 0;
}
```

### 진법과 불리언 형식

```cpp
#include <iostream>
#include <iomanip>

int main() {
    int num = 255;

    std::cout << "Decimal:     " << std::dec << num << std::endl;  // 255
    std::cout << "Hexadecimal: " << std::hex << num << std::endl;  // ff
    std::cout << "Octal:       " << std::oct << num << std::endl;  // 377

    // 진법 접두사 표시
    std::cout << std::showbase;
    std::cout << "Hex: " << std::hex << num << std::endl;  // 0xff
    std::cout << "Oct: " << std::oct << num << std::endl;  // 0377
    std::cout << std::noshowbase << std::dec;  // 초기화

    // 불리언 출력
    bool flag = true;
    std::cout << flag << std::endl;                    // 1
    std::cout << std::boolalpha << flag << std::endl;  // true
    std::cout << std::noboolalpha;  // 초기화

    return 0;
}
```

### 조작자 요약

| 조작자 | 헤더 | 효과 | 지속? |
|--------|------|------|-------|
| `setw(n)` | `<iomanip>` | 최소 필드 너비 | 아니오 (다음 출력만) |
| `setfill(c)` | `<iomanip>` | 채우기 문자 | 예 |
| `setprecision(n)` | `<iomanip>` | 자릿수 정밀도 | 예 |
| `fixed` | `<iostream>` | 고정 소수점 표기 | 예 |
| `scientific` | `<iostream>` | 과학적 표기법 | 예 |
| `left` / `right` | `<iostream>` | 정렬 | 예 |
| `dec` / `hex` / `oct` | `<iostream>` | 진법 | 예 |
| `boolalpha` | `<iostream>` | true/false 출력 | 예 |
| `showbase` | `<iostream>` | 0x 또는 0 접두사 표시 | 예 |

---

## 5. 문자열 스트림

`<sstream>`은 `std::string` 객체에서 작동하는 스트림 클래스를 제공하여 문자열을 입력 스트림처럼 파싱하거나 출력 스트림처럼 구성할 수 있게 합니다.

### istringstream으로 파싱

```cpp
#include <iostream>
#include <sstream>
#include <string>

int main() {
    std::string data = "Alice 90 85 92";

    std::istringstream iss(data);
    std::string name;
    int s1, s2, s3;

    iss >> name >> s1 >> s2 >> s3;

    std::cout << name << "'s average: "
              << (s1 + s2 + s3) / 3.0 << std::endl;
    // Alice's average: 89

    return 0;
}
```

### ostringstream으로 구성

```cpp
#include <iostream>
#include <sstream>
#include <iomanip>

int main() {
    std::ostringstream oss;

    oss << "Total: $" << std::fixed << std::setprecision(2) << 1234.5;
    std::string result = oss.str();

    std::cout << result << std::endl;  // Total: $1234.50

    return 0;
}
```

### 구분자로 문자열 분할

```cpp
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(s);
    std::string token;

    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main() {
    auto parts = split("one:two:three:four", ':');
    for (const auto& p : parts) {
        std::cout << "[" << p << "] ";
    }
    std::cout << std::endl;
    // [one] [two] [three] [four]

    return 0;
}
```

---

## 6. getline

### 전체 줄 읽기

```cpp
#include <iostream>
#include <string>

int main() {
    std::string line;

    std::cout << "Enter a sentence: ";
    std::getline(std::cin, line);

    std::cout << "You said: " << line << std::endl;

    return 0;
}
```

### cin >>와 getline 혼합

일반적인 함정: `cin >>` 후에 남은 줄바꿈이 버퍼에 남아 있어 다음 `getline`이 빈 문자열을 읽습니다.

```cpp
#include <iostream>
#include <string>

int main() {
    int age;
    std::string name;

    std::cout << "Enter age: ";
    std::cin >> age;

    // 잘못됨: getline이 남은 '\n'을 읽음
    // std::getline(std::cin, name);  // 빈 문자열!

    // 수정: 남은 줄바꿈을 먼저 버림
    std::cin.ignore();  // 또는 std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << "Enter full name: ";
    std::getline(std::cin, name);

    std::cout << "Name: " << name << ", Age: " << age << std::endl;

    return 0;
}
```

### 사용자 정의 구분자

```cpp
#include <iostream>
#include <string>

int main() {
    std::string field;

    // 줄바꿈 대신 '|'까지 읽기
    std::cout << "Enter pipe-delimited data: ";
    while (std::getline(std::cin, field, '|')) {
        std::cout << "  Field: [" << field << "]\n";
        if (std::cin.peek() == '\n') break;  // 줄 끝에서 중단
    }

    return 0;
}
```

---

## 7. 오류 스트림

C++는 세 가지 표준 출력 스트림을 제공합니다.

| 스트림 | 용도 | 버퍼링? |
|--------|------|---------|
| `std::cout` | 일반 출력 | 예 |
| `std::cerr` | 오류 메시지 | 아니오 (즉시) |
| `std::clog` | 진단/로그 메시지 | 예 |

```cpp
#include <iostream>

int main() {
    std::cout << "Normal output" << std::endl;
    std::cerr << "Error: something went wrong" << std::endl;
    std::clog << "Log: operation completed" << std::endl;

    return 0;
}
```

### 출력 리디렉션

```bash
# stdout만 리디렉션 (오류는 화면에 표시)
./program > output.txt

# stderr만 리디렉션
./program 2> errors.txt

# 둘 다 별도로 리디렉션
./program > output.txt 2> errors.txt

# 둘 다 같은 파일로 리디렉션
./program > all.txt 2>&1
```

---

## 8. 모범 사례

### 헤더에서 using namespace std 피하기

```cpp
// 나쁨: header.h
#pragma once
using namespace std;  // 이 헤더를 포함하는 모든 파일을 오염

// 좋음: header.h
#pragma once
#include <string>
std::string formatName(const std::string& first, const std::string& last);
```

### 네임스페이스 별칭

완전 한정 이름이 길 때 짧은 별칭을 만듭니다.

```cpp
namespace fs = std::filesystem;  // C++17
namespace chrono = std::chrono;

// 이제 별칭 사용
auto start = chrono::steady_clock::now();
```

### 인자 의존 검색(ADL, Argument-Dependent Lookup)

ADL은 컴파일러가 명시적 한정 없이 인자의 네임스페이스에서 함수를 찾을 수 있게 합니다.

```cpp
#include <iostream>
#include <string>

namespace geometry {
    struct Point { double x, y; };

    // Point 인자와 함께 호출될 때 ADL로 발견됨
    std::ostream& operator<<(std::ostream& os, const Point& p) {
        return os << "(" << p.x << ", " << p.y << ")";
    }
}

int main() {
    geometry::Point p{3.0, 4.0};
    std::cout << p << std::endl;  // ADL이 geometry::operator<<를 찾음
    return 0;
}
```

### 입력 견고성 체크리스트

1. 추출이 성공했는지 항상 확인 (`if (std::cin >> x)`)
2. `>>` 후 `getline` 전에 `std::cin.ignore()` 사용
3. 잘못된 입력에서 복구하려면 `std::cin.clear()` + `std::cin.ignore(...)` 사용
4. 복잡한 파싱에는 `getline` + `istringstream` 선호

---

## 9. 요약

| 개념 | 핵심 포인트 |
|------|------------|
| 네임스페이스 | 충돌 방지를 위해 식별자를 그룹화 |
| 중첩 네임스페이스 | `namespace A::B { }` (C++17) |
| 익명 네임스페이스 | 내부 링크 (파일 범위) |
| `using` 선언 | 하나의 이름 가져오기 |
| `using` 지시문 | 모든 이름 가져오기 (주의하여 사용) |
| `std::cout` | 버퍼링된 표준 출력 |
| `std::cin` | `>>` 추출을 사용한 표준 입력 |
| `std::cerr` | 버퍼링되지 않은 오류 출력 |
| `<iomanip>` | 서식 있는 출력 (너비, 정밀도, 채우기) |
| `std::istringstream` | 문자열을 입력으로 파싱 |
| `std::ostringstream` | 문자열을 출력으로 구성 |
| `std::getline` | 전체 줄 읽기 (사용자 정의 구분자 옵션) |
| 네임스페이스 별칭 | `namespace fs = std::filesystem;` |

---

## 연습문제

### 연습문제 1: 서식 있는 테이블

각 셀이 너비 5의 필드에 오른쪽 정렬된 구구단(1-5) x (1-5)을 출력하는 프로그램을 작성하세요. 정렬에 `setw`를 사용하세요.

### 연습문제 2: 영수증 포매터

`ostringstream`과 `iomanip`을 사용하여 항목 이름은 왼쪽 정렬(20자), 수량은 오른쪽 정렬(5자), 가격은 오른쪽 정렬(10자, 소수점 2자리)인 영수증 문자열을 구성하세요. 하단에 합계를 출력하세요.

### 연습문제 3: 견고한 입력 루프

유효한 숫자가 입력될 때까지 사용자에게 `double`을 반복적으로 요청하는 프로그램을 작성하세요. `cin.clear()`와 `cin.ignore()`를 사용하여 숫자가 아닌 입력을 우아하게 처리하세요. 유효한 항목 후에 정확히 소수점 4자리로 값을 출력하세요.

### 연습문제 4: 네임스페이스 충돌 해결

각각 다른 메시지를 출력하는 `void play()` 함수를 정의하는 두 개의 네임스페이스(`audio`와 `video`)를 만드세요. `main`에서 완전 한정 이름을 사용하여 두 함수를 호출한 다음 하나에 대해 `using` 선언을 사용하고 한정 없이 호출하세요.

### 연습문제 5: CSV 줄 파서

`','`를 구분자로 `istringstream`과 `getline`을 사용하는 함수 `std::vector<std::string> parseCSV(const std::string& line)`를 작성하세요. `"Alice,30,Engineering,95.5"`로 테스트하고 각 필드를 별도의 줄에 출력하세요.

---

## 다음 단계

[08_Classes_Basics.md](./08_Classes_Basics.md)에서 클래스에 대해 알아봅시다!
