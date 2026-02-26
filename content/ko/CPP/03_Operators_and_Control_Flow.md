# 연산자와 제어문

**이전**: [변수와 자료형](./02_Variables_and_Types.md) | **다음**: [함수](./04_Functions.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C++ 표현식에서 산술(arithmetic), 대입(assignment), 비교(comparison), 논리(logical), 비트(bitwise) 연산자를 적용할 수 있습니다
2. 전위(prefix)와 후위(postfix) 증감 연산자의 동작 차이를 구분할 수 있습니다
3. 논리 표현식에서의 단락 평가(short-circuit evaluation)를 설명하고 부수 효과(side effect) 영향을 파악할 수 있습니다
4. `if`, `else if`, `else`, `switch` 문을 사용해 분기(branching) 로직을 구현할 수 있습니다
5. `for`, `while`, `do-while` 반복문(C++11의 범위 기반 `for` 포함)을 설계할 수 있습니다
6. `break`와 `continue`를 사용해 반복 실행 흐름을 제어할 수 있습니다
7. 연산자 우선순위(operator precedence) 규칙을 파악하고 괄호로 의도를 명확히 표현할 수 있습니다

---

연산자와 제어문은 모든 프로그램의 핸들과 엑셀러레이터입니다. 연산자 없이는 값을 계산하거나 비교하거나 조합할 수 없고, 제어문 없이는 결정을 내리거나 작업을 반복할 수 없습니다. 이 둘이 함께 정적인 선언 목록을 동적이고 반응적인 로직으로 변환하며, 이후에 접하게 될 모든 고급 C++ 기능은 궁극적으로 이 기본 요소들 위에 구축됩니다.

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

### 정수 나눗셈 vs 실수 나눗셈

```cpp
#include <iostream>

int main() {
    int a = 7, b = 2;

    // 정수 나눗셈 (소수점 버림)
    std::cout << "7 / 2 = " << a / b << std::endl;  // 3

    // 실수 나눗셈
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
| `==` | 같다 | `a == b` |
| `!=` | 다르다 | `a != b` |
| `<` | 작다 | `a < b` |
| `>` | 크다 | `a > b` |
| `<=` | 작거나 같다 | `a <= b` |
| `>=` | 크거나 같다 | `a >= b` |

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
| `&&` | AND (그리고) | `a && b` |
| `\|\|` | OR (또는) | `a \|\| b` |
| `!` | NOT (부정) | `!a` |

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
    std::cout << "할인 적용: " << discount << std::endl;  // true

    return 0;
}
```

### 단락 평가 (Short-circuit Evaluation)

```cpp
#include <iostream>

int main() {
    int x = 0;

    // &&: 첫 번째가 false면 두 번째 평가 안 함
    if (false && (++x > 0)) {
        // x는 증가하지 않음
    }
    std::cout << "x after &&: " << x << std::endl;  // 0

    // ||: 첫 번째가 true면 두 번째 평가 안 함
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
조건 ? 참일_때_값 : 거짓일_때_값
```

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;

    // if-else 대체
    int max = (a > b) ? a : b;
    std::cout << "최댓값: " << max << std::endl;  // 20

    // 문자열 선택
    int score = 85;
    std::string result = (score >= 60) ? "합격" : "불합격";
    std::cout << "결과: " << result << std::endl;  // 합격

    // 중첩 (가독성 주의)
    int num = 0;
    std::string sign = (num > 0) ? "양수" : (num < 0) ? "음수" : "영";
    std::cout << "부호: " << sign << std::endl;  // 영

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
        std::cout << "성인입니다." << std::endl;
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
        std::cout << "합격" << std::endl;
    } else {
        std::cout << "불합격" << std::endl;
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

### if 문에서 변수 선언 (C++17)

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};

    // C++17: if문 내 변수 선언
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
            std::cout << "월요일" << std::endl;
            break;
        case 2:
            std::cout << "화요일" << std::endl;
            break;
        case 3:
            std::cout << "수요일" << std::endl;
            break;
        case 4:
            std::cout << "목요일" << std::endl;
            break;
        case 5:
            std::cout << "금요일" << std::endl;
            break;
        case 6:
        case 7:
            std::cout << "주말" << std::endl;
            break;
        default:
            std::cout << "잘못된 값" << std::endl;
    }

    return 0;
}
```

### fall-through (의도적 생략)

```cpp
#include <iostream>

int main() {
    char grade = 'B';

    switch (grade) {
        case 'A':
        case 'B':
        case 'C':
            std::cout << "합격" << std::endl;
            break;
        case 'D':
        case 'F':
            std::cout << "불합격" << std::endl;
            break;
        default:
            std::cout << "잘못된 등급" << std::endl;
    }

    return 0;
}
```

### switch 문 주의사항

```cpp
// switch는 정수형, 문자형, enum만 사용 가능
// 문자열은 불가 (C++에서)

// 변수 선언 시 중괄호 필요
switch (value) {
    case 1: {
        int x = 10;  // 중괄호로 범위 지정
        // ...
        break;
    }
    case 2:
        // ...
        break;
}
```

---

## 9. for 루프

### 기본 for 루프

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

### 역순 for 루프

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

### 중첩 for 루프

```cpp
#include <iostream>

int main() {
    // 구구단 2단
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

### 범위 기반 for 루프 (C++11)

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

    // vector 순회
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    for (const auto& name : names) {
        std::cout << name << std::endl;
    }

    return 0;
}
```

---

## 10. while 루프

### 기본 while 루프

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
        std::cout << "숫자 입력 (0 종료): ";
        std::cin >> num;

        if (num == 0) {
            break;  // 루프 탈출
        }

        std::cout << "입력: " << num << std::endl;
    }

    std::cout << "종료" << std::endl;

    return 0;
}
```

---

## 11. do-while 루프

최소 한 번은 실행됩니다.

```cpp
#include <iostream>

int main() {
    int num;

    do {
        std::cout << "1~10 사이 숫자 입력: ";
        std::cin >> num;
    } while (num < 1 || num > 10);  // 조건이 참이면 반복

    std::cout << "입력한 숫자: " << num << std::endl;

    return 0;
}
```

### while vs do-while

```cpp
#include <iostream>

int main() {
    int x = 0;

    // while: 조건 먼저 검사
    while (x > 0) {
        std::cout << "while 실행" << std::endl;
        x--;
    }
    // 출력 없음

    // do-while: 최소 한 번 실행
    do {
        std::cout << "do-while 실행" << std::endl;
        x--;
    } while (x > 0);
    // "do-while 실행" 출력됨

    return 0;
}
```

---

## 12. break와 continue

### break

루프를 즉시 탈출합니다.

```cpp
#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        if (i == 5) {
            break;  // 5에서 탈출
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
|---------|--------|
| 1 (높음) | `()`, `[]`, `->`, `.` |
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
| 14 (낮음) | `=`, `+=`, `-=` 등 |

**팁**: 헷갈리면 괄호를 사용하세요!

---

## 14. 요약

| 분류 | 연산자 |
|------|--------|
| 산술 | `+`, `-`, `*`, `/`, `%` |
| 비교 | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| 논리 | `&&`, `\|\|`, `!` |
| 비트 | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| 대입 | `=`, `+=`, `-=`, `*=`, `/=` |

| 제어문 | 용도 |
|--------|------|
| `if-else` | 조건 분기 |
| `switch` | 다중 분기 |
| `for` | 횟수 기반 반복 |
| `while` | 조건 기반 반복 |
| `do-while` | 최소 1회 실행 반복 |

---

## 연습 문제

### 연습 1: 연산자 평가 결과 예측

코드를 실행하지 않고 각 출력문의 결과를 예측하세요. 그 후 컴파일하여 답을 확인하세요.

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

    // 예측: 다음 코드는 0을 출력할까요, 1을 출력할까요?
    int counter = 0;
    if (false && (++counter > 0)) {}
    if (true  || (++counter > 0)) {}
    std::cout << counter << std::endl;  // ?

    return 0;
}
```

예측 후, `counter`가 그 값을 갖는 이유를 단락 평가(short-circuit evaluation)를 사용해 설명하세요.

### 연습 2: 비트 연산자(Bitwise Operator)를 이용한 권한 플래그

비트 연산자를 사용하여 간단한 권한 시스템을 구현하세요. 세 가지 플래그(`READ = 1`, `WRITE = 2`, `EXECUTE = 4`)를 정의하고 다음을 수행하는 프로그램을 작성하세요:

1. READ와 WRITE가 설정된 권한 변수를 생성합니다.
2. EXECUTE가 설정되어 있는지 확인합니다 (설정되지 않아야 합니다).
3. 비트 OR를 사용하여 EXECUTE 권한을 부여합니다.
4. `&= ~WRITE`를 사용하여 WRITE 권한을 제거합니다.
5. 최종 권한을 사람이 읽기 좋은 형식으로 출력합니다.

```cpp
#include <iostream>

int main() {
    const int READ    = 1;  // 001
    const int WRITE   = 2;  // 010
    const int EXECUTE = 4;  // 100

    int perms = READ | WRITE;  // 초기: READ + WRITE

    // 여기에 단계를 추가하세요 ...

    return 0;
}
```

### 연습 3: switch를 이용한 성적 계산기

`score / 10`에 대한 `switch` 문을 사용하여 숫자 점수(0–100)를 문자 등급으로 변환하는 함수 `char letterGrade(int score)`를 작성하세요. 10과 9는 `'A'`, 8은 `'B'`, 7은 `'C'`, 6은 `'D'`, 나머지는 `'F'`로 매핑합니다. 경계값(59, 60, 89, 90, 100)을 포함한 최소 다섯 가지 다른 점수로 테스트하세요.

### 연습 4: 반복문 패턴 도전

중첩된 `for` 반복문을 사용하여 주어진 `size`에 대해 다음 다이아몬드 패턴을 출력하세요 (여기서는 `size = 4`의 경우):

```
   *
  ***
 *****
*******
 *****
  ***
   *
```

윗 절반(중간 행 포함)은 `2*i - 1`개의 별로 된 행을 가져야 하고(`i`는 1부터 `size`까지), 아랫 절반은 이를 대칭으로 반영합니다. `std::cout`, 반복문, 그리고 필요한 경우 `continue` 또는 `break`만 사용하세요 — 문자열 조작 함수는 사용하지 마세요.

### 연습 5: 입력 검증 반복문

`do-while` 반복문을 사용하여 사용자가 [1, 100] 범위의 정수를 입력할 때까지 반복적으로 입력을 요청하는 완전한 프로그램을 작성하세요. 유효한 값이 입력되면 삼항 연산자(ternary operator)를 사용하여 `"낮음"(1–33)`, `"중간"(34–66)`, `"높음"(67–100)`으로 분류하고 결과를 출력하세요.

---

## 다음 단계

[함수](./04_Functions.md)에서 함수를 배워봅시다!
