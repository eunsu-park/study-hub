# 연산자와 제어문

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

## 다음 단계

[04_Functions.md](./04_Functions.md)에서 함수를 배워봅시다!
