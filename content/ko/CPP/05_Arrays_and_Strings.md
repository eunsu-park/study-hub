# 배열과 문자열

**이전**: [함수](./04_Functions.md) | **다음**: [포인터와 참조](./06_Pointers_and_References.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C 스타일 배열과 다차원 배열을 선언, 초기화하고 원소에 접근할 수 있다
2. C 스타일 배열과 `std::array`를 비교하고, 경계 검사 접근의 안전성 이점을 설명할 수 있다
3. C 스타일 문자열(`char[]`)과 `std::string`의 차이를 구분하고, 각각의 적합한 사용 상황을 식별할 수 있다
4. `std::string`의 핵심 연산(연결, 검색, 부분 문자열 추출, 삽입, 교체)을 적용할 수 있다
5. `std::stoi`, `std::stod`, `std::to_string`을 사용해 문자열-숫자 간 변환을 구현할 수 있다
6. `std::stringstream`을 활용한 문자열 분리(splitting) 로직을 설계할 수 있다
7. `std::string_view`(C++17)가 문자열 데이터를 읽을 때 불필요한 복사를 어떻게 피하는지 설명할 수 있다

---

배열과 문자열은 데이터 저장의 핵심 도구입니다. 단순한 로그 파서부터 고빈도 거래 시스템까지, 거의 모든 실제 프로그램은 값이나 문자의 시퀀스를 생성하고, 검색하고, 변환하는 데 많은 시간을 소비합니다. 어떤 컨테이너를 선택해야 하는지(원시 배열, `std::array`, `std::string`, `std::string_view`)를 파악하고 성능 트레이드오프를 이해하는 것은 C++ 경력 전반에 걸쳐 큰 도움이 될 것입니다.

## 1. 배열 기초

배열은 같은 타입의 여러 값을 연속된 메모리에 저장합니다.

### 배열 선언과 초기화

```cpp
#include <iostream>

int main() {
    // 크기 지정
    int arr1[5];  // 초기화되지 않음 (쓰레기값)

    // 초기화 목록
    int arr2[5] = {1, 2, 3, 4, 5};

    // 부분 초기화 (나머지는 0)
    int arr3[5] = {1, 2};  // {1, 2, 0, 0, 0}

    // 전체 0으로 초기화
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
    std::cout << "첫 번째: " << arr[0] << std::endl;  // 10
    std::cout << "세 번째: " << arr[2] << std::endl;  // 30

    // 쓰기
    arr[1] = 200;
    std::cout << "수정 후: " << arr[1] << std::endl;  // 200

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
    std::cout << "배열 크기: " << size << std::endl;  // 5

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
    // 3행 4열
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

크기가 고정된 안전한 배열입니다.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // 크기
    std::cout << "크기: " << arr.size() << std::endl;

    // 접근
    std::cout << "첫 번째: " << arr[0] << std::endl;
    std::cout << "마지막: " << arr.back() << std::endl;

    // 범위 검사 접근
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

| 특징 | C 배열 | std::array |
|------|--------|------------|
| 크기 확인 | sizeof 계산 | .size() |
| 범위 검사 | 없음 | .at() |
| 복사 | 불가 | 가능 |
| 함수 전달 | 포인터로 변환 | 값/참조 전달 가능 |

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
    std::cout << "길이: " << strlen(str1) << std::endl;  // 5

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
        std::cout << "찾음: " << pos << std::endl;  // World
    }

    return 0;
}
```

---

## 5. std::string

C++의 문자열 클래스입니다.

### 기본 사용

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
    std::cout << "길이: " << s1.length() << std::endl;  // 5
    std::cout << "크기: " << s1.size() << std::endl;    // 5 (동일)

    // 빈 문자열 체크
    std::string empty;
    std::cout << "비어있음: " << empty.empty() << std::endl;  // true

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

    // += 연산
    s1 += "!";
    std::cout << s1 << std::endl;  // Hello!

    // append
    s1.append(" C++");
    std::cout << s1 << std::endl;  // Hello! C++

    // 비교
    if (s1 == "Hello! C++") {
        std::cout << "같음" << std::endl;
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
    std::cout << "첫 문자: " << str[0] << std::endl;  // H
    std::cout << "마지막: " << str.back() << std::endl;  // o

    // 범위 검사 접근
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
        std::cout << "위치: " << pos << std::endl;  // 7
    }

    // 문자 찾기
    pos = str.find('o');
    std::cout << "첫 번째 o: " << pos << std::endl;  // 4

    // 뒤에서 찾기
    pos = str.rfind('o');
    std::cout << "마지막 o: " << pos << std::endl;  // 8

    // 못 찾은 경우
    pos = str.find("xyz");
    if (pos == std::string::npos) {
        std::cout << "찾지 못함" << std::endl;
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
    str.replace(7, 5, "C++");  // World를 C++로
    std::cout << str << std::endl;  // Hello, C++!

    // 지우기
    str.clear();
    std::cout << "비어있음: " << str.empty() << std::endl;

    return 0;
}
```

---

## 6. 문자열 변환

### 숫자 ↔ 문자열

```cpp
#include <iostream>
#include <string>

int main() {
    // 숫자 → 문자열
    int num = 42;
    std::string str1 = std::to_string(num);
    std::cout << "to_string: " << str1 << std::endl;

    double pi = 3.14159;
    std::string str2 = std::to_string(pi);
    std::cout << "to_string: " << str2 << std::endl;

    // 문자열 → 숫자
    std::string s1 = "123";
    int n1 = std::stoi(s1);
    std::cout << "stoi: " << n1 << std::endl;

    std::string s2 = "3.14";
    double d1 = std::stod(s2);
    std::cout << "stod: " << d1 << std::endl;

    // 다른 변환 함수
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
    std::cout << "대문자: " << (char)std::toupper(c) << std::endl;  // A

    c = 'Z';
    std::cout << "소문자: " << (char)std::tolower(c) << std::endl;  // z

    // 문자 검사
    std::cout << std::boolalpha;
    std::cout << "isalpha('A'): " << (bool)std::isalpha('A') << std::endl;  // true
    std::cout << "isdigit('5'): " << (bool)std::isdigit('5') << std::endl;  // true
    std::cout << "isspace(' '): " << (bool)std::isspace(' ') << std::endl;  // true

    // 문자열 전체 대문자로
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

    // 단어 입력 (공백 전까지)
    std::cout << "단어 입력: ";
    std::cin >> word;
    std::cout << "입력: " << word << std::endl;

    // 버퍼 비우기
    std::cin.ignore();

    // 줄 전체 입력
    std::cout << "문장 입력: ";
    std::getline(std::cin, line);
    std::cout << "입력: " << line << std::endl;

    return 0;
}
```

---

## 8. 문자열 분리

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

문자열을 복사 없이 참조합니다.

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

    // 부분 문자열도 복사 없이
    std::string_view sv = str;
    std::cout << sv.substr(0, 5) << std::endl;  // Hello

    return 0;
}
```

---

## 10. 실습 예제

### 문자열 뒤집기

```cpp
#include <iostream>
#include <string>
#include <algorithm>

int main() {
    std::string str = "Hello";

    // 방법 1: reverse 함수
    std::reverse(str.begin(), str.end());
    std::cout << str << std::endl;  // olleH

    // 방법 2: 직접 구현
    str = "World";
    int len = str.length();
    for (int i = 0; i < len / 2; i++) {
        std::swap(str[i], str[len - 1 - i]);
    }
    std::cout << str << std::endl;  // dlroW

    return 0;
}
```

### 팰린드롬 검사

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

### 단어 개수 세기

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
    std::cout << "단어 수: " << countWords(text) << std::endl;  // 5
    return 0;
}
```

---

## 11. 요약

| 종류 | 특징 |
|------|------|
| C 배열 `T[]` | 고정 크기, 경계 검사 없음 |
| `std::array<T, N>` | 고정 크기, 안전함 |
| C 문자열 `char[]` | null 종료, 수동 관리 |
| `std::string` | 동적 크기, 자동 관리 |
| `std::string_view` | 읽기 전용 참조 |

| std::string 메서드 | 설명 |
|-------------------|------|
| `length()`, `size()` | 길이 |
| `empty()` | 비어있는지 |
| `substr(pos, len)` | 부분 문자열 |
| `find(str)` | 검색 |
| `replace(pos, len, str)` | 교체 |
| `insert(pos, str)` | 삽입 |
| `erase(pos, len)` | 삭제 |

---

## 다음 단계

[포인터와 참조](./06_Pointers_and_References.md)에서 포인터와 참조를 배워봅시다!
