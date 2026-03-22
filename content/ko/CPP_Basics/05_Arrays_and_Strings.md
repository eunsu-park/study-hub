# 배열과 문자열

**이전**: [함수](./04_Functions.md) | **다음**: [포인터와 참조](./06_Pointers_and_References.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C 스타일 배열과 다차원 배열에서 요소를 선언, 초기화, 접근한다
2. C 스타일 배열과 `std::array`를 비교하고 경계 검사 접근의 안전성 장점을 설명한다
3. C 스타일 문자열(`char[]`)과 `std::string`을 구분하고 각각이 적절한 상황을 파악한다
4. 핵심 `std::string` 연산을 적용한다: 연결, 검색, 부분 문자열 추출, 삽입, 교체
5. `std::stoi`, `std::stod`, `std::to_string`을 사용하여 문자열-숫자 변환을 구현한다
6. `std::stringstream`을 사용하여 문자열 분할 로직을 설계한다
7. `std::string_view` (C++17)가 문자열 데이터를 읽을 때 불필요한 복사를 어떻게 방지하는지 설명한다

---

배열과 문자열은 데이터 저장의 핵심입니다. 간단한 로그 파서부터 고빈도 거래 시스템까지 거의 모든 실제 프로그램은 값이나 문자의 시퀀스를 생성, 검색, 변환하는 데 많은 시간을 보냅니다. 어떤 컨테이너를 선택할지(원시 배열, `std::array`, `std::string`, `std::string_view`)와 성능 트레이드오프를 이해하는 것은 C++ 경력 전반에 걸쳐 큰 도움이 됩니다.

## 1. 배열 기초

배열은 동일한 타입의 여러 값을 연속된 메모리에 저장합니다.

### 배열 선언과 초기화

```cpp
#include <iostream>

int main() {
    // 크기 지정
    int arr1[5];  // 미초기화 (쓰레기 값)

    // 초기화 리스트
    int arr2[5] = {1, 2, 3, 4, 5};

    // 부분 초기화 (나머지는 0)
    int arr3[5] = {1, 2};  // {1, 2, 0, 0, 0}

    // 모두 0으로 초기화
    int arr4[5] = {};  // {0, 0, 0, 0, 0}

    // 크기 자동 결정
    int arr5[] = {1, 2, 3};  // 크기 3

    // 출력
    for (int i = 0; i < 5; i++) {
        std::cout << arr2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 배열 접근

```cpp
#include <iostream>

int main() {
    int arr[5] = {10, 20, 30, 40, 50};

    // 읽기
    std::cout << "First: " << arr[0] << std::endl;  // 10
    std::cout << "Third: " << arr[2] << std::endl;  // 30

    // 쓰기
    arr[1] = 200;
    std::cout << "After modification: " << arr[1] << std::endl;  // 200

    // 범위 기반 for
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 배열 크기

```cpp
#include <iostream>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};

    // sizeof로 크기 계산
    int size = sizeof(arr) / sizeof(arr[0]);
    std::cout << "Array size: " << size << std::endl;  // 5

    // C++17: std::size
    // #include <iterator>
    // std::cout << std::size(arr) << std::endl;

    return 0;
}
```

---

## 2. 다차원 배열

### 2차원 배열

```cpp
#include <iostream>

int main() {
    // 3행, 4열
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    // 접근
    std::cout << "matrix[1][2] = " << matrix[1][2] << std::endl;  // 7

    // 전체 출력
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

### 3차원 배열

```cpp
#include <iostream>

int main() {
    int cube[2][3][4] = {
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        },
        {
            {13, 14, 15, 16},
            {17, 18, 19, 20},
            {21, 22, 23, 24}
        }
    };

    std::cout << "cube[1][2][3] = " << cube[1][2][3] << std::endl;  // 24

    return 0;
}
```

---

## 3. std::array (C++11)

안전한 고정 크기 배열입니다.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // 크기
    std::cout << "Size: " << arr.size() << std::endl;

    // 접근
    std::cout << "First: " << arr[0] << std::endl;
    std::cout << "Last: " << arr.back() << std::endl;

    // 경계 검사 접근
    std::cout << "arr.at(2): " << arr.at(2) << std::endl;
    // arr.at(10);  // 예외 발생!

    // 범위 기반 for
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 채우기
    arr.fill(0);

    return 0;
}
```

### 배열 vs std::array

| 기능 | C 배열 | std::array |
|------|--------|------------|
| 크기 확인 | sizeof 계산 | .size() |
| 경계 검사 | 없음 | .at() |
| 복사 | 불가능 | 가능 |
| 함수 전달 | 포인터로 변환 | 값/참조 전달 |

---

## 4. C 스타일 문자열

문자 배열로 문자열을 표현합니다.

```cpp
#include <iostream>
#include <cstring>  // strlen, strcpy 등

int main() {
    // 문자열 리터럴
    char str1[] = "Hello";  // {'H', 'e', 'l', 'l', 'o', '\0'}
    char str2[10] = "World";

    // 길이
    std::cout << "Length: " << strlen(str1) << std::endl;  // 5

    // 출력
    std::cout << str1 << std::endl;

    // 문자별 접근
    for (int i = 0; str1[i] != '\0'; i++) {
        std::cout << str1[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### C 문자열 함수

```cpp
#include <iostream>
#include <cstring>

int main() {
    char str1[20] = "Hello";
    char str2[20] = "World";
    char dest[40];

    // 복사
    strcpy(dest, str1);
    std::cout << "strcpy: " << dest << std::endl;  // Hello

    // 연결
    strcat(dest, " ");
    strcat(dest, str2);
    std::cout << "strcat: " << dest << std::endl;  // Hello World

    // 비교
    if (strcmp(str1, str2) < 0) {
        std::cout << str1 << " < " << str2 << std::endl;
    }

    // 찾기
    char* pos = strstr(dest, "World");
    if (pos != nullptr) {
        std::cout << "Found: " << pos << std::endl;  // World
    }

    return 0;
}
```

---

## 5. std::string

C++ 문자열 클래스입니다.

### 기본 사용법

```cpp
#include <iostream>
#include <string>

int main() {
    // 생성
    std::string s1 = "Hello";
    std::string s2("World");
    std::string s3(5, 'x');  // "xxxxx"

    // 출력
    std::cout << s1 << " " << s2 << std::endl;

    // 길이
    std::cout << "Length: " << s1.length() << std::endl;  // 5
    std::cout << "Size: " << s1.size() << std::endl;      // 5 (동일)

    // 빈 문자열 확인
    std::string empty;
    std::cout << "Is empty: " << empty.empty() << std::endl;  // true

    return 0;
}
```

### 문자열 연산

```cpp
#include <iostream>
#include <string>

int main() {
    std::string s1 = "Hello";
    std::string s2 = "World";

    // 연결
    std::string s3 = s1 + " " + s2;
    std::cout << s3 << std::endl;  // Hello World

    // += 연산자
    s1 += "!";
    std::cout << s1 << std::endl;  // Hello!

    // append
    s1.append(" C++");
    std::cout << s1 << std::endl;  // Hello! C++

    // 비교
    if (s1 == "Hello! C++") {
        std::cout << "Equal" << std::endl;
    }

    if (s1 < s2) {  // 사전순 비교
        std::cout << s1 << " < " << s2 << std::endl;
    }

    return 0;
}
```

### 문자열 접근

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello";

    // 인덱스 접근
    std::cout << "First character: " << str[0] << std::endl;  // H
    std::cout << "Last: " << str.back() << std::endl;  // o

    // 경계 검사 접근
    std::cout << "at(1): " << str.at(1) << std::endl;  // e

    // 수정
    str[0] = 'h';
    std::cout << str << std::endl;  // hello

    // 범위 기반 for
    for (char c : str) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 부분 문자열

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // 부분 문자열 추출
    std::string sub = str.substr(7, 5);  // 위치 7부터 5글자
    std::cout << sub << std::endl;  // World

    // 위치부터 끝까지
    std::string rest = str.substr(7);
    std::cout << rest << std::endl;  // World!

    return 0;
}
```

### 검색

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // 찾기
    size_t pos = str.find("World");
    if (pos != std::string::npos) {
        std::cout << "Position: " << pos << std::endl;  // 7
    }

    // 문자 찾기
    pos = str.find('o');
    std::cout << "First o: " << pos << std::endl;  // 4

    // 끝에서부터 찾기
    pos = str.rfind('o');
    std::cout << "Last o: " << pos << std::endl;  // 8

    // 찾지 못한 경우
    pos = str.find("xyz");
    if (pos == std::string::npos) {
        std::cout << "Not found" << std::endl;
    }

    return 0;
}
```

### 수정

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // 삽입
    str.insert(7, "Beautiful ");
    std::cout << str << std::endl;  // Hello, Beautiful World!

    // 삭제
    str.erase(7, 10);  // 위치 7부터 10글자 삭제
    std::cout << str << std::endl;  // Hello, World!

    // 교체
    str.replace(7, 5, "C++");  // World를 C++로 교체
    std::cout << str << std::endl;  // Hello, C++!

    // 비우기
    str.clear();
    std::cout << "Is empty: " << str.empty() << std::endl;

    return 0;
}
```

---

## 6. 문자열 변환

### 숫자 <-> 문자열

```cpp
#include <iostream>
#include <string>

int main() {
    // 숫자 -> 문자열
    int num = 42;
    std::string str1 = std::to_string(num);
    std::cout << "to_string: " << str1 << std::endl;

    double pi = 3.14159;
    std::string str2 = std::to_string(pi);
    std::cout << "to_string: " << str2 << std::endl;

    // 문자열 -> 숫자
    std::string s1 = "123";
    int n1 = std::stoi(s1);
    std::cout << "stoi: " << n1 << std::endl;

    std::string s2 = "3.14";
    double d1 = std::stod(s2);
    std::cout << "stod: " << d1 << std::endl;

    // 기타 변환 함수
    // std::stol - long
    // std::stoll - long long
    // std::stof - float

    return 0;
}
```

### 문자 변환

```cpp
#include <iostream>
#include <cctype>
#include <string>

int main() {
    char c = 'a';

    // 대소문자 변환
    std::cout << "Uppercase: " << (char)std::toupper(c) << std::endl;  // A

    c = 'Z';
    std::cout << "Lowercase: " << (char)std::tolower(c) << std::endl;  // z

    // 문자 검사
    std::cout << std::boolalpha;
    std::cout << "isalpha('A'): " << (bool)std::isalpha('A') << std::endl;  // true
    std::cout << "isdigit('5'): " << (bool)std::isdigit('5') << std::endl;  // true
    std::cout << "isspace(' '): " << (bool)std::isspace(' ') << std::endl;  // true

    // 전체 문자열 대문자 변환
    std::string str = "Hello World";
    for (char& c : str) {
        c = std::toupper(c);
    }
    std::cout << str << std::endl;  // HELLO WORLD

    return 0;
}
```

---

## 7. 문자열 입력

```cpp
#include <iostream>
#include <string>

int main() {
    std::string word;
    std::string line;

    // 단어 입력 (공백까지)
    std::cout << "Enter a word: ";
    std::cin >> word;
    std::cout << "Input: " << word << std::endl;

    // 버퍼 비우기
    std::cin.ignore();

    // 한 줄 전체 입력
    std::cout << "Enter a sentence: ";
    std::getline(std::cin, line);
    std::cout << "Input: " << line << std::endl;

    return 0;
}
```

---

## 8. 문자열 분할

```cpp
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

int main() {
    std::string str = "apple,banana,cherry,date";

    // stringstream 사용
    std::stringstream ss(str);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    for (const auto& t : tokens) {
        std::cout << t << std::endl;
    }

    return 0;
}
```

---

## 9. string_view (C++17)

복사 없이 문자열을 참조합니다.

```cpp
#include <iostream>
#include <string>
#include <string_view>

void print(std::string_view sv) {
    std::cout << sv << std::endl;
}

int main() {
    std::string str = "Hello, World!";
    const char* cstr = "C-style string";

    // 다양한 문자열 타입을 받을 수 있음
    print(str);
    print(cstr);
    print("Literal");

    // 복사 없는 부분 문자열
    std::string_view sv = str;
    std::cout << sv.substr(0, 5) << std::endl;  // Hello

    return 0;
}
```

---

## 10. 실습 예제

### 문자열 반전

```cpp
#include <iostream>
#include <string>
#include <algorithm>

int main() {
    std::string str = "Hello";

    // 방법 1: reverse 함수
    std::reverse(str.begin(), str.end());
    std::cout << str << std::endl;  // olleH

    // 방법 2: 수동 구현
    str = "World";
    int len = str.length();
    for (int i = 0; i < len / 2; i++) {
        std::swap(str[i], str[len - 1 - i]);
    }
    std::cout << str << std::endl;  // dlroW

    return 0;
}
```

### 회문(Palindrome) 검사

```cpp
#include <iostream>
#include <string>
#include <algorithm>

bool isPalindrome(const std::string& str) {
    std::string reversed = str;
    std::reverse(reversed.begin(), reversed.end());
    return str == reversed;
}

int main() {
    std::cout << std::boolalpha;
    std::cout << isPalindrome("radar") << std::endl;  // true
    std::cout << isPalindrome("hello") << std::endl;  // false
    return 0;
}
```

### 단어 수 세기

```cpp
#include <iostream>
#include <string>
#include <sstream>

int countWords(const std::string& str) {
    std::stringstream ss(str);
    std::string word;
    int count = 0;

    while (ss >> word) {
        count++;
    }

    return count;
}

int main() {
    std::string text = "Hello World this is C++";
    std::cout << "Word count: " << countWords(text) << std::endl;  // 5
    return 0;
}
```

---

## 11. 요약

| 타입 | 특징 |
|------|------|
| C 배열 `T[]` | 고정 크기, 경계 검사 없음 |
| `std::array<T, N>` | 고정 크기, 안전함 |
| C 문자열 `char[]` | 널 종료, 수동 관리 |
| `std::string` | 동적 크기, 자동 관리 |
| `std::string_view` | 읽기 전용 참조 |

| std::string 메서드 | 설명 |
|--------------------|------|
| `length()`, `size()` | 길이 |
| `empty()` | 비어 있는지 확인 |
| `substr(pos, len)` | 부분 문자열 |
| `find(str)` | 검색 |
| `replace(pos, len, str)` | 교체 |
| `insert(pos, str)` | 삽입 |
| `erase(pos, len)` | 삭제 |

---

## 연습문제

### 연습문제 1: 배열 통계

`std::array<double, 10>`을 선언하고 사용자 입력으로 채운 후 최솟값, 최댓값, 평균값을 출력하는 프로그램을 작성하세요.

### 연습문제 2: 시저 암호

모든 소문자를 `shift`만큼 이동시키는 함수 `std::string encrypt(const std::string& text, int shift)`를 작성하세요 ('z'에서 'a'로 순환). 대응하는 `decrypt` 함수를 작성하고 왕복 변환이 되는지 확인하세요.

### 연습문제 3: CSV 파서

사용자로부터 단일 CSV 줄(예: `"Alice,90,85,92"`)을 읽고 쉼표로 분할하는 프로그램을 작성하세요. 이름과 숫자 필드의 평균을 출력하세요.

### 연습문제 4: string_view 성능

`const std::string&`을 받는 함수와 `std::string_view`를 받는 함수를 작성하여 인자의 모음을 세세요. 각각을 문자열 리터럴로 호출하고 어느 것이 임시 `std::string` 생성을 피하는지 설명하세요.

### 연습문제 5: 행렬 전치

3x4 `int` 행렬을 선언하고 1-12 값으로 초기화한 후, 4x3 행렬로 전치하는 함수를 작성하세요. 원본 행렬과 전치된 행렬을 모두 출력하세요.

---

## 다음 단계

[06_Pointers_and_References.md](./06_Pointers_and_References.md)에서 포인터와 참조에 대해 알아봅시다!
