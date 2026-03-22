# 연산자와 제어 흐름

**이전**: [변수와 타입](./02_Variables_and_Types.md) | **다음**: [함수](./04_Functions.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C++ 표현식에서 산술, 대입, 비교, 논리, 비트 연산자를 적용한다
2. 전위 및 후위 증감 연산자의 동작을 구분한다
3. 논리 표현식에서의 단축 평가(Short-circuit Evaluation)를 설명하고 부수 효과를 파악한다
4. `if`, `else if`, `else`, `switch` 문을 사용하여 분기 로직을 구현한다
5. `for`, `while`, `do-while`과 범위 기반 `for` (C++11)를 사용하여 반복문을 설계한다
6. `break`와 `continue`를 적용하여 반복문 실행 흐름을 제어한다
7. 연산자 우선순위 규칙을 파악하고 의도를 명확히 하기 위해 괄호를 사용한다

---

연산자와 제어 흐름은 모든 프로그램의 핸들과 가속 페달입니다. 연산자 없이는 값을 계산, 비교, 결합할 수 없고, 제어 흐름 없이는 결정을 내리거나 작업을 반복할 수 없습니다. 이 둘이 함께 정적인 선언 목록을 동적이고 반응하는 로직으로 변환합니다. 앞으로 만나게 될 모든 고급 C++ 기능은 궁극적으로 이러한 기본 요소 위에 구축됩니다.

## 1. 산술 연산자

### 기본 산술 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `+` | 덧셈 | `a + b` |
| `-` | 뺄셈 | `a - b` |
| `*` | 곱셈 | `a * b` |
| `/` | 나눗셈 | `a / b` |
| `%` | 나머지 | `a % b` |

```cpp
#include <iostream>

int main() {
    int a = 17, b = 5;

    std::cout << "a + b = " << a + b << std::endl;  // 22
    std::cout << "a - b = " << a - b << std::endl;  // 12
    std::cout << "a * b = " << a * b << std::endl;  // 85
    std::cout << "a / b = " << a / b << std::endl;  // 3 (정수 나눗셈)
    std::cout << "a % b = " << a % b << std::endl;  // 2

    return 0;
}
```

### 정수 나눗셈 vs 부동소수점 나눗셈

```cpp
#include <iostream>

int main() {
    int a = 7, b = 2;

    // 정수 나눗셈 (소수점 절삭)
    std::cout << "7 / 2 = " << a / b << std::endl;  // 3

    // 부동소수점 나눗셈
    std::cout << "7.0 / 2 = " << 7.0 / 2 << std::endl;  // 3.5
    std::cout << "(double)7 / 2 = " << static_cast<double>(a) / b << std::endl;  // 3.5

    return 0;
}
```

### 증감 연산자

```cpp
#include <iostream>

int main() {
    int a = 5;

    std::cout << "a = " << a << std::endl;    // 5
    std::cout << "++a = " << ++a << std::endl; // 6 (전위: 먼저 증가)
    std::cout << "a++ = " << a++ << std::endl; // 6 (후위: 나중에 증가)
    std::cout << "a = " << a << std::endl;    // 7

    return 0;
}
```

---

## 2. 대입 연산자

### 복합 대입 연산자

```cpp
#include <iostream>

int main() {
    int a = 10;

    a += 5;   // a = a + 5
    std::cout << "a += 5: " << a << std::endl;  // 15

    a -= 3;   // a = a - 3
    std::cout << "a -= 3: " << a << std::endl;  // 12

    a *= 2;   // a = a * 2
    std::cout << "a *= 2: " << a << std::endl;  // 24

    a /= 4;   // a = a / 4
    std::cout << "a /= 4: " << a << std::endl;  // 6

    a %= 4;   // a = a % 4
    std::cout << "a %= 4: " << a << std::endl;  // 2

    return 0;
}
```

---

## 3. 비교 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `==` | 같음 | `a == b` |
| `!=` | 같지 않음 | `a != b` |
| `<` | 작음 | `a < b` |
| `>` | 큼 | `a > b` |
| `<=` | 작거나 같음 | `a <= b` |
| `>=` | 크거나 같음 | `a >= b` |

```cpp
#include <iostream>

int main() {
    int a = 5, b = 10;

    std::cout << std::boolalpha;  // true/false로 출력
    std::cout << "a == b: " << (a == b) << std::endl;  // false
    std::cout << "a != b: " << (a != b) << std::endl;  // true
    std::cout << "a < b: " << (a < b) << std::endl;    // true
    std::cout << "a > b: " << (a > b) << std::endl;    // false
    std::cout << "a <= b: " << (a <= b) << std::endl;  // true
    std::cout << "a >= b: " << (a >= b) << std::endl;  // false

    return 0;
}
```

---

## 4. 논리 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `&&` | AND | `a && b` |
| `\|\|` | OR | `a \|\| b` |
| `!` | NOT | `!a` |

```cpp
#include <iostream>

int main() {
    bool a = true, b = false;

    std::cout << std::boolalpha;
    std::cout << "a && b: " << (a && b) << std::endl;  // false
    std::cout << "a || b: " << (a || b) << std::endl;  // true
    std::cout << "!a: " << (!a) << std::endl;          // false
    std::cout << "!b: " << (!b) << std::endl;          // true

    // 복합 조건
    int age = 25;
    bool isStudent = true;

    bool discount = (age < 20) || isStudent;  // 학생이거나 20세 미만
    std::cout << "Discount applied: " << discount << std::endl;  // true

    return 0;
}
```

### 단축 평가(Short-circuit Evaluation)

```cpp
#include <iostream>

int main() {
    int x = 0;

    // &&: 첫 번째가 false면 두 번째는 평가되지 않음
    if (false && (++x > 0)) {
        // x는 증가하지 않음
    }
    std::cout << "x after &&: " << x << std::endl;  // 0

    // ||: 첫 번째가 true면 두 번째는 평가되지 않음
    if (true || (++x > 0)) {
        // x는 증가하지 않음
    }
    std::cout << "x after ||: " << x << std::endl;  // 0

    return 0;
}
```

---

## 5. 비트 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `&` | AND | `a & b` |
| `\|` | OR | `a \| b` |
| `^` | XOR | `a ^ b` |
| `~` | NOT | `~a` |
| `<<` | 왼쪽 시프트 | `a << n` |
| `>>` | 오른쪽 시프트 | `a >> n` |

```cpp
#include <iostream>

int main() {
    int a = 5;  // 0101
    int b = 3;  // 0011

    std::cout << "a & b = " << (a & b) << std::endl;  // 1 (0001)
    std::cout << "a | b = " << (a | b) << std::endl;  // 7 (0111)
    std::cout << "a ^ b = " << (a ^ b) << std::endl;  // 6 (0110)
    std::cout << "~a = " << (~a) << std::endl;        // -6

    std::cout << "a << 1 = " << (a << 1) << std::endl;  // 10 (1010)
    std::cout << "a >> 1 = " << (a >> 1) << std::endl;  // 2 (0010)

    return 0;
}
```

---

## 6. 삼항 연산자

```cpp
condition ? value_if_true : value_if_false
```

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;

    // if-else 대안
    int max = (a > b) ? a : b;
    std::cout << "Maximum: " << max << std::endl;  // 20

    // 문자열 선택
    int score = 85;
    std::string result = (score >= 60) ? "Pass" : "Fail";
    std::cout << "Result: " << result << std::endl;  // Pass

    // 중첩 (가독성 주의)
    int num = 0;
    std::string sign = (num > 0) ? "positive" : (num < 0) ? "negative" : "zero";
    std::cout << "Sign: " << sign << std::endl;  // zero

    return 0;
}
```

---

## 7. if 문

### 기본 if 문

```cpp
#include <iostream>

int main() {
    int age = 18;

    if (age >= 18) {
        std::cout << "You are an adult." << std::endl;
    }

    return 0;
}
```

### if-else 문

```cpp
#include <iostream>

int main() {
    int score = 75;

    if (score >= 60) {
        std::cout << "Pass" << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
```

### if-else if-else 문

```cpp
#include <iostream>

int main() {
    int score = 85;

    if (score >= 90) {
        std::cout << "A" << std::endl;
    } else if (score >= 80) {
        std::cout << "B" << std::endl;
    } else if (score >= 70) {
        std::cout << "C" << std::endl;
    } else if (score >= 60) {
        std::cout << "D" << std::endl;
    } else {
        std::cout << "F" << std::endl;
    }

    return 0;
}
```

### if 문 내 변수 선언 (C++17)

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};

    // C++17: if 문 내 변수 선언
    if (auto it = scores.find("Alice"); it != scores.end()) {
        std::cout << "Alice's score: " << it->second << std::endl;
    }

    return 0;
}
```

---

## 8. switch 문

### 기본 switch 문

```cpp
#include <iostream>

int main() {
    int day = 3;

    switch (day) {
        case 1:
            std::cout << "Monday" << std::endl;
            break;
        case 2:
            std::cout << "Tuesday" << std::endl;
            break;
        case 3:
            std::cout << "Wednesday" << std::endl;
            break;
        case 4:
            std::cout << "Thursday" << std::endl;
            break;
        case 5:
            std::cout << "Friday" << std::endl;
            break;
        case 6:
        case 7:
            std::cout << "Weekend" << std::endl;
            break;
        default:
            std::cout << "Invalid value" << std::endl;
    }

    return 0;
}
```

### 폴스루(Fall-through, 의도적 생략)

```cpp
#include <iostream>

int main() {
    char grade = 'B';

    switch (grade) {
        case 'A':
        case 'B':
        case 'C':
            std::cout << "Pass" << std::endl;
            break;
        case 'D':
        case 'F':
            std::cout << "Fail" << std::endl;
            break;
        default:
            std::cout << "Invalid grade" << std::endl;
    }

    return 0;
}
```

### switch 문 주의사항

```cpp
// switch는 정수 타입, 문자 타입, 열거형에서만 동작
// 문자열은 허용되지 않음 (C++에서)

// 변수 선언 시 중괄호 필요
switch (value) {
    case 1: {
        int x = 10;  // 중괄호로 스코프 정의
        // ...
        break;
    }
    case 2:
        // ...
        break;
}
```

---

## 9. for 반복문

### 기본 for 반복문

```cpp
#include <iostream>

int main() {
    // 1부터 5까지 출력
    for (int i = 1; i <= 5; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 1 2 3 4 5

    return 0;
}
```

### 역순 for 반복문

```cpp
#include <iostream>

int main() {
    for (int i = 5; i >= 1; i--) {
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

### 중첩 for 반복문

```cpp
#include <iostream>

int main() {
    // 2단 구구단
    for (int i = 1; i <= 9; i++) {
        std::cout << "2 x " << i << " = " << 2 * i << std::endl;
    }

    // 별 삼각형
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= i; j++) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

출력:
```
*
**
***
****
*****
```

### 범위 기반 for 반복문 (C++11)

```cpp
#include <iostream>
#include <vector>

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    // 배열 순회
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 참조로 수정
    for (int& num : arr) {
        num *= 2;
    }

    // 벡터 순회
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    for (const auto& name : names) {
        std::cout << name << std::endl;
    }

    return 0;
}
```

---

## 10. while 반복문

### 기본 while 반복문

```cpp
#include <iostream>

int main() {
    int count = 1;

    while (count <= 5) {
        std::cout << count << " ";
        count++;
    }
    std::cout << std::endl;  // 1 2 3 4 5

    return 0;
}
```

### 무한 루프와 탈출

```cpp
#include <iostream>

int main() {
    int num;

    while (true) {
        std::cout << "Enter a number (0 to exit): ";
        std::cin >> num;

        if (num == 0) {
            break;  // 루프 탈출
        }

        std::cout << "Input: " << num << std::endl;
    }

    std::cout << "Exited" << std::endl;

    return 0;
}
```

---

## 11. do-while 반복문

최소 한 번 실행됩니다.

```cpp
#include <iostream>

int main() {
    int num;

    do {
        std::cout << "Enter a number between 1 and 10: ";
        std::cin >> num;
    } while (num < 1 || num > 10);  // 조건이 참이면 반복

    std::cout << "You entered: " << num << std::endl;

    return 0;
}
```

### while vs do-while

```cpp
#include <iostream>

int main() {
    int x = 0;

    // while: 조건을 먼저 확인
    while (x > 0) {
        std::cout << "while executed" << std::endl;
        x--;
    }
    // 출력 없음

    // do-while: 최소 한 번 실행
    do {
        std::cout << "do-while executed" << std::endl;
        x--;
    } while (x > 0);
    // "do-while executed" 출력됨

    return 0;
}
```

---

## 12. break와 continue

### break

즉시 반복문을 종료합니다.

```cpp
#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        if (i == 5) {
            break;  // 5에서 종료
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 1 2 3 4

    return 0;
}
```

### continue

현재 반복을 건너뜁니다.

```cpp
#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) {
            continue;  // 짝수 건너뛰기
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;  // 1 3 5 7 9

    return 0;
}
```

---

## 13. 연산자 우선순위

| 우선순위 | 연산자 |
|----------|--------|
| 1 (최고) | `()`, `[]`, `->`, `.` |
| 2 | `!`, `~`, `++`, `--`, `sizeof` |
| 3 | `*`, `/`, `%` |
| 4 | `+`, `-` |
| 5 | `<<`, `>>` |
| 6 | `<`, `<=`, `>`, `>=` |
| 7 | `==`, `!=` |
| 8 | `&` |
| 9 | `^` |
| 10 | `\|` |
| 11 | `&&` |
| 12 | `\|\|` |
| 13 | `?:` |
| 14 (최저) | `=`, `+=`, `-=` 등 |

**팁**: 의심스러울 때는 괄호를 사용하세요!

---

## 14. 요약

| 카테고리 | 연산자 |
|----------|--------|
| 산술 | `+`, `-`, `*`, `/`, `%` |
| 비교 | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| 논리 | `&&`, `\|\|`, `!` |
| 비트 | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| 대입 | `=`, `+=`, `-=`, `*=`, `/=` |

| 제어 흐름 | 용도 |
|-----------|------|
| `if-else` | 조건 분기 |
| `switch` | 다중 분기 |
| `for` | 횟수 기반 반복 |
| `while` | 조건 기반 반복 |
| `do-while` | 최소 한 번 실행 |

---

## 연습문제

### 연습문제 1: 연산자 평가 예측

코드를 실행하지 않고 각 문의 출력을 예측하세요. 그런 다음 컴파일하여 답을 확인하세요.

```cpp
#include <iostream>

int main() {
    int a = 10, b = 3;

    // 실행 전에 각 출력을 예측하세요
    std::cout << a / b << std::endl;          // ?
    std::cout << a % b << std::endl;          // ?
    std::cout << (double)a / b << std::endl;  // ?

    int x = 5;
    std::cout << x++ << std::endl;  // ?
    std::cout << x   << std::endl;  // ?
    std::cout << ++x << std::endl;  // ?

    // 예측: 다음은 0 또는 1을 출력할까요?
    int counter = 0;
    if (false && (++counter > 0)) {}
    if (true  || (++counter > 0)) {}
    std::cout << counter << std::endl;  // ?

    return 0;
}
```

예측 후, `counter`가 그 값을 갖는 이유를 자신의 말로 설명하세요 (단축 평가).

### 연습문제 2: 비트 플래그 연산

비트 연산자를 사용하여 간단한 권한 시스템을 구현하세요. 세 가지 플래그(`READ = 1`, `WRITE = 2`, `EXECUTE = 4`)를 정의하고 다음을 수행하는 프로그램을 작성하세요:

1. READ와 WRITE가 설정된 권한 변수를 생성합니다.
2. EXECUTE가 설정되어 있는지 확인합니다 (설정되어 있지 않아야 합니다).
3. 비트 OR를 사용하여 EXECUTE 권한을 부여합니다.
4. 비트 AND와 NOT(`&= ~WRITE`)을 사용하여 WRITE 권한을 해제합니다.
5. 최종 권한을 사람이 읽을 수 있는 형태로 출력합니다.

```cpp
#include <iostream>

int main() {
    const int READ    = 1;  // 001
    const int WRITE   = 2;  // 010
    const int EXECUTE = 4;  // 100

    int perms = READ | WRITE;  // 시작: READ + WRITE

    // 여기에 단계를 추가하세요 ...

    return 0;
}
```

### 연습문제 3: switch를 사용한 학점 계산기

숫자 점수(0-100)를 `score / 10`에 대한 `switch` 문을 사용하여 문자 학점으로 변환하는 함수 `char letterGrade(int score)`를 작성하세요. 10과 9는 `'A'`, 8은 `'B'`, 7은 `'C'`, 6은 `'D'`, 나머지는 `'F'`로 매핑합니다. 경계값(59, 60, 89, 90, 100)을 포함하여 최소 5개의 점수로 테스트하세요.

### 연습문제 4: 반복문 패턴 도전

중첩 `for` 반복문을 사용하여 주어진 `size`에 대해 다음 다이아몬드 패턴을 출력하세요 (여기서는 `size = 4`):

```
   *
  ***
 *****
*******
 *****
  ***
   *
```

윗부분(가운데 행 포함)은 `2*i - 1`개의 별(`i`는 1부터 `size`까지)이어야 하고, 아랫부분은 이를 반영합니다. `std::cout`, 반복문, 필요하면 `continue` 또는 `break`만 사용하세요. 문자열 조작 함수는 사용하지 마세요.

### 연습문제 5: 입력 유효성 검사 반복문

`do-while` 반복문을 사용하여 사용자에게 [1, 100] 범위의 정수를 반복적으로 입력받는 완전한 프로그램을 작성하세요. 유효한 값이 입력될 때까지 반복해야 합니다. 유효한 숫자를 받으면 삼항 연산자를 사용하여 `"low"` (1-33), `"medium"` (34-66), `"high"` (67-100)로 분류한 후 분류 결과를 출력하세요.

---

## 다음 단계

[04_Functions.md](./04_Functions.md)에서 함수에 대해 알아봅시다!
