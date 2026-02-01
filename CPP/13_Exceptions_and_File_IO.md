# 예외 처리와 파일 입출력

## 1. 예외 처리란?

예외(Exception)는 프로그램 실행 중 발생하는 비정상적인 상황입니다. C++은 try-catch 구문으로 예외를 처리합니다.

```
┌─────────────────────────────────────────────┐
│              예외 처리 흐름                   │
└─────────────────────────────────────────────┘
                    │
     ┌──────────────┴──────────────┐
     ▼                              ▼
┌─────────┐                   ┌─────────┐
│   try   │ ───예외 발생───▶  │  throw  │
│  블록   │                   │         │
└─────────┘                   └─────────┘
     │                              │
     │ 예외 없음                     │ 예외 전파
     ▼                              ▼
┌─────────┐                   ┌─────────┐
│  정상   │                   │  catch  │
│  종료   │                   │  블록   │
└─────────┘                   └─────────┘
```

---

## 2. try, throw, catch

### 기본 문법

```cpp
#include <iostream>
#include <string>

double divide(double a, double b) {
    if (b == 0) {
        throw std::string("0으로 나눌 수 없습니다");  // 예외 발생
    }
    return a / b;
}

int main() {
    try {
        std::cout << divide(10, 2) << std::endl;  // 5
        std::cout << divide(10, 0) << std::endl;  // 예외 발생!
        std::cout << "이 줄은 실행되지 않음" << std::endl;
    }
    catch (const std::string& e) {
        std::cout << "에러: " << e << std::endl;
    }

    std::cout << "프로그램 계속 실행" << std::endl;

    return 0;
}
```

출력:
```
5
에러: 0으로 나눌 수 없습니다
프로그램 계속 실행
```

### 여러 catch 블록

```cpp
#include <iostream>
#include <stdexcept>

void process(int value) {
    if (value < 0) {
        throw std::invalid_argument("음수는 허용되지 않습니다");
    }
    if (value > 100) {
        throw std::out_of_range("100을 초과할 수 없습니다");
    }
    if (value == 0) {
        throw 0;  // int 타입 예외
    }
    std::cout << "값: " << value << std::endl;
}

int main() {
    int tests[] = {50, -10, 150, 0};

    for (int val : tests) {
        try {
            process(val);
        }
        catch (const std::invalid_argument& e) {
            std::cout << "잘못된 인자: " << e.what() << std::endl;
        }
        catch (const std::out_of_range& e) {
            std::cout << "범위 초과: " << e.what() << std::endl;
        }
        catch (int e) {
            std::cout << "정수 예외: " << e << std::endl;
        }
        catch (...) {  // 모든 예외 포착
            std::cout << "알 수 없는 예외" << std::endl;
        }
    }

    return 0;
}
```

출력:
```
값: 50
잘못된 인자: 음수는 허용되지 않습니다
범위 초과: 100을 초과할 수 없습니다
정수 예외: 0
```

---

## 3. 표준 예외 클래스

```
                std::exception
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
  logic_error    runtime_error   bad_alloc
       │              │
   ┌───┴───┐      ┌───┴───┐
   ▼       ▼      ▼       ▼
invalid_  out_of_ overflow_ underflow_
argument  range   error     error
```

### 주요 예외 클래스

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include <new>

int main() {
    // logic_error 계열 (프로그래머 실수)
    try {
        throw std::invalid_argument("잘못된 인자");
    } catch (const std::exception& e) {
        std::cout << "invalid_argument: " << e.what() << std::endl;
    }

    try {
        throw std::out_of_range("범위 초과");
    } catch (const std::exception& e) {
        std::cout << "out_of_range: " << e.what() << std::endl;
    }

    try {
        throw std::length_error("길이 오류");
    } catch (const std::exception& e) {
        std::cout << "length_error: " << e.what() << std::endl;
    }

    // runtime_error 계열 (실행 시 오류)
    try {
        throw std::runtime_error("런타임 오류");
    } catch (const std::exception& e) {
        std::cout << "runtime_error: " << e.what() << std::endl;
    }

    try {
        throw std::overflow_error("오버플로우");
    } catch (const std::exception& e) {
        std::cout << "overflow_error: " << e.what() << std::endl;
    }

    // bad_alloc (메모리 할당 실패)
    try {
        throw std::bad_alloc();
    } catch (const std::exception& e) {
        std::cout << "bad_alloc: " << e.what() << std::endl;
    }

    return 0;
}
```

### exception 클래스 상속

```cpp
#include <iostream>
#include <exception>
#include <string>

// 커스텀 예외 클래스
class FileNotFoundError : public std::exception {
private:
    std::string message;

public:
    FileNotFoundError(const std::string& filename)
        : message("파일을 찾을 수 없음: " + filename) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

class InvalidFormatError : public std::exception {
private:
    std::string message;

public:
    InvalidFormatError(const std::string& detail)
        : message("잘못된 형식: " + detail) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

void readConfig(const std::string& filename) {
    if (filename.empty()) {
        throw FileNotFoundError("(빈 파일명)");
    }
    if (filename.find(".cfg") == std::string::npos) {
        throw InvalidFormatError("확장자는 .cfg여야 합니다");
    }
    std::cout << filename << " 읽기 성공" << std::endl;
}

int main() {
    std::string files[] = {"", "data.txt", "config.cfg"};

    for (const auto& f : files) {
        try {
            readConfig(f);
        }
        catch (const FileNotFoundError& e) {
            std::cout << "[파일 오류] " << e.what() << std::endl;
        }
        catch (const InvalidFormatError& e) {
            std::cout << "[형식 오류] " << e.what() << std::endl;
        }
    }

    return 0;
}
```

---

## 4. 예외 재발생과 noexcept

### 예외 재발생

```cpp
#include <iostream>
#include <stdexcept>

void lowLevel() {
    throw std::runtime_error("저수준 오류");
}

void midLevel() {
    try {
        lowLevel();
    }
    catch (const std::exception& e) {
        std::cout << "[중간층] 예외 감지: " << e.what() << std::endl;
        throw;  // 예외 재발생 (상위로 전달)
    }
}

void highLevel() {
    try {
        midLevel();
    }
    catch (const std::exception& e) {
        std::cout << "[상위층] 최종 처리: " << e.what() << std::endl;
    }
}

int main() {
    highLevel();
    return 0;
}
```

출력:
```
[중간층] 예외 감지: 저수준 오류
[상위층] 최종 처리: 저수준 오류
```

### noexcept 지정자

```cpp
#include <iostream>

// 예외를 던지지 않음을 보장
void safeFunction() noexcept {
    // 예외를 던지면 std::terminate() 호출
    std::cout << "안전한 함수" << std::endl;
}

// 조건부 noexcept
template<typename T>
void process(T& obj) noexcept(noexcept(obj.doSomething())) {
    obj.doSomething();
}

class Safe {
public:
    void doSomething() noexcept {
        std::cout << "Safe::doSomething" << std::endl;
    }
};

class Unsafe {
public:
    void doSomething() {
        throw std::runtime_error("오류");
    }
};

int main() {
    std::cout << std::boolalpha;

    // noexcept 여부 확인
    std::cout << "safeFunction noexcept: "
              << noexcept(safeFunction()) << std::endl;  // true

    Safe s;
    Unsafe u;

    std::cout << "Safe noexcept: "
              << noexcept(process(s)) << std::endl;    // true
    std::cout << "Unsafe noexcept: "
              << noexcept(process(u)) << std::endl;    // false

    safeFunction();

    return 0;
}
```

---

## 5. 예외 안전성

### 예외 안전성 수준

| 수준 | 설명 |
|------|------|
| No-throw | 예외를 절대 던지지 않음 |
| Strong | 예외 발생 시 원래 상태로 복원 |
| Basic | 예외 발생 후에도 유효한 상태 유지 |
| No guarantee | 예외 발생 시 상태 불명 |

### RAII로 예외 안전성 확보

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

// RAII 클래스
class FileHandler {
private:
    FILE* file;

public:
    FileHandler(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (!file) {
            throw std::runtime_error("파일 열기 실패");
        }
        std::cout << "파일 열림" << std::endl;
    }

    ~FileHandler() {
        if (file) {
            fclose(file);
            std::cout << "파일 닫힘" << std::endl;
        }
    }

    void write(const char* data) {
        if (fputs(data, file) == EOF) {
            throw std::runtime_error("쓰기 실패");
        }
    }

    // 복사 금지
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};

void processFile() {
    FileHandler fh("test.txt", "w");  // RAII: 생성자에서 열기
    fh.write("Hello, World!\n");
    throw std::runtime_error("중간에 예외 발생!");
    fh.write("이 줄은 실행되지 않음");
}  // RAII: 소멸자에서 자동으로 닫힘

int main() {
    try {
        processFile();
    }
    catch (const std::exception& e) {
        std::cout << "예외: " << e.what() << std::endl;
    }

    return 0;
}
```

출력:
```
파일 열림
파일 닫힘
예외: 중간에 예외 발생!
```

---

## 6. 파일 입출력 기초

### 파일 스트림 클래스

| 클래스 | 용도 |
|--------|------|
| `ifstream` | 파일 읽기 |
| `ofstream` | 파일 쓰기 |
| `fstream` | 읽기/쓰기 |

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // 파일 쓰기
    std::ofstream outFile("example.txt");
    if (outFile.is_open()) {
        outFile << "Hello, File!" << std::endl;
        outFile << "Line 2" << std::endl;
        outFile << 42 << " " << 3.14 << std::endl;
        outFile.close();
        std::cout << "파일 쓰기 완료" << std::endl;
    }

    // 파일 읽기
    std::ifstream inFile("example.txt");
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            std::cout << "읽음: " << line << std::endl;
        }
        inFile.close();
    }

    return 0;
}
```

### 파일 열기 모드

```cpp
#include <iostream>
#include <fstream>

int main() {
    // 쓰기 모드 (기본: 덮어쓰기)
    std::ofstream f1("test.txt");
    f1 << "새 내용" << std::endl;
    f1.close();

    // 추가 모드
    std::ofstream f2("test.txt", std::ios::app);
    f2 << "추가된 내용" << std::endl;
    f2.close();

    // 바이너리 모드
    std::ofstream f3("data.bin", std::ios::binary);
    int num = 12345;
    f3.write(reinterpret_cast<char*>(&num), sizeof(num));
    f3.close();

    // 읽기+쓰기 모드
    std::fstream f4("test.txt", std::ios::in | std::ios::out);

    // 파일 끝에서 시작 (추가)
    std::ofstream f5("test.txt", std::ios::ate);

    // 기존 내용 삭제 후 쓰기
    std::ofstream f6("test.txt", std::ios::trunc);

    return 0;
}
```

| 모드 | 설명 |
|------|------|
| `ios::in` | 읽기 |
| `ios::out` | 쓰기 |
| `ios::app` | 끝에 추가 |
| `ios::ate` | 파일 끝에서 시작 |
| `ios::trunc` | 기존 내용 삭제 |
| `ios::binary` | 바이너리 모드 |

---

## 7. 파일 읽기 방법

### 다양한 읽기 방법

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

int main() {
    // 테스트 파일 생성
    std::ofstream out("data.txt");
    out << "Alice 25 90.5\n";
    out << "Bob 30 85.0\n";
    out << "Charlie 28 92.3\n";
    out.close();

    // 방법 1: >> 연산자 (공백 기준)
    std::ifstream f1("data.txt");
    std::string name;
    int age;
    double score;
    std::cout << "=== >> 연산자 ===" << std::endl;
    while (f1 >> name >> age >> score) {
        std::cout << name << ", " << age << ", " << score << std::endl;
    }
    f1.close();

    // 방법 2: getline (한 줄씩)
    std::ifstream f2("data.txt");
    std::string line;
    std::cout << "\n=== getline ===" << std::endl;
    while (std::getline(f2, line)) {
        std::cout << "줄: " << line << std::endl;
    }
    f2.close();

    // 방법 3: getline + stringstream
    std::ifstream f3("data.txt");
    std::cout << "\n=== stringstream ===" << std::endl;
    while (std::getline(f3, line)) {
        std::istringstream iss(line);
        iss >> name >> age >> score;
        std::cout << "이름=" << name << ", 나이=" << age
                  << ", 점수=" << score << std::endl;
    }
    f3.close();

    // 방법 4: 전체 파일 읽기
    std::ifstream f4("data.txt");
    std::stringstream buffer;
    buffer << f4.rdbuf();
    std::string content = buffer.str();
    std::cout << "\n=== 전체 내용 ===" << std::endl;
    std::cout << content;
    f4.close();

    return 0;
}
```

### 문자 단위 읽기

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ofstream out("chars.txt");
    out << "ABC\nDEF";
    out.close();

    std::ifstream in("chars.txt");
    char c;

    // get()으로 한 문자씩
    std::cout << "문자별: ";
    while (in.get(c)) {
        if (c == '\n') {
            std::cout << "[LF]";
        } else {
            std::cout << c;
        }
    }
    std::cout << std::endl;

    // peek()으로 미리보기
    in.clear();
    in.seekg(0);

    std::cout << "Peek: ";
    while (in.peek() != EOF) {
        char peeked = in.peek();
        char got;
        in.get(got);
        std::cout << "(" << (int)peeked << ")";
    }
    std::cout << std::endl;

    in.close();
    return 0;
}
```

---

## 8. 바이너리 파일

### 바이너리 읽기/쓰기

```cpp
#include <iostream>
#include <fstream>
#include <vector>

struct Record {
    int id;
    char name[50];
    double score;
};

int main() {
    // 바이너리 쓰기
    std::ofstream out("records.bin", std::ios::binary);

    Record r1 = {1, "Alice", 95.5};
    Record r2 = {2, "Bob", 87.0};
    Record r3 = {3, "Charlie", 91.2};

    out.write(reinterpret_cast<char*>(&r1), sizeof(Record));
    out.write(reinterpret_cast<char*>(&r2), sizeof(Record));
    out.write(reinterpret_cast<char*>(&r3), sizeof(Record));
    out.close();

    std::cout << "Record 크기: " << sizeof(Record) << " bytes" << std::endl;

    // 바이너리 읽기
    std::ifstream in("records.bin", std::ios::binary);

    Record record;
    std::cout << "\n=== 레코드 읽기 ===" << std::endl;
    while (in.read(reinterpret_cast<char*>(&record), sizeof(Record))) {
        std::cout << "ID: " << record.id
                  << ", Name: " << record.name
                  << ", Score: " << record.score << std::endl;
    }
    in.close();

    // 특정 레코드 접근 (랜덤 액세스)
    std::ifstream in2("records.bin", std::ios::binary);

    // 두 번째 레코드로 이동 (0부터 시작)
    in2.seekg(1 * sizeof(Record));
    in2.read(reinterpret_cast<char*>(&record), sizeof(Record));
    std::cout << "\n두 번째 레코드: " << record.name << std::endl;

    in2.close();

    return 0;
}
```

### 벡터 저장/로드

```cpp
#include <iostream>
#include <fstream>
#include <vector>

void saveVector(const std::string& filename, const std::vector<int>& vec) {
    std::ofstream out(filename, std::ios::binary);

    // 크기 먼저 저장
    size_t size = vec.size();
    out.write(reinterpret_cast<char*>(&size), sizeof(size));

    // 데이터 저장
    out.write(reinterpret_cast<const char*>(vec.data()),
              size * sizeof(int));
    out.close();
}

std::vector<int> loadVector(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);

    // 크기 읽기
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    // 데이터 읽기
    std::vector<int> vec(size);
    in.read(reinterpret_cast<char*>(vec.data()),
            size * sizeof(int));
    in.close();

    return vec;
}

int main() {
    std::vector<int> original = {10, 20, 30, 40, 50};

    saveVector("vector.bin", original);
    std::cout << "저장 완료" << std::endl;

    std::vector<int> loaded = loadVector("vector.bin");
    std::cout << "로드된 데이터: ";
    for (int n : loaded) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## 9. 파일 위치 제어

### seekg, seekp, tellg, tellp

```cpp
#include <iostream>
#include <fstream>

int main() {
    // 파일 생성
    std::ofstream out("position.txt");
    out << "0123456789ABCDEF";
    out.close();

    // 읽기 위치 제어
    std::ifstream in("position.txt");

    // 현재 위치 확인
    std::cout << "시작 위치: " << in.tellg() << std::endl;

    // 5번째 위치로 이동 (처음부터)
    in.seekg(5, std::ios::beg);
    char c;
    in.get(c);
    std::cout << "위치 5의 문자: " << c << std::endl;

    // 현재 위치에서 3칸 뒤로
    in.seekg(3, std::ios::cur);
    in.get(c);
    std::cout << "3칸 뒤: " << c << std::endl;

    // 끝에서 2칸 앞
    in.seekg(-2, std::ios::end);
    in.get(c);
    std::cout << "끝에서 2칸 앞: " << c << std::endl;

    in.close();

    // 쓰기 위치 제어
    std::fstream file("position.txt", std::ios::in | std::ios::out);

    file.seekp(10);  // 10번째 위치로 이동
    file << "XYZ";   // ABC를 XYZ로 덮어씀

    file.seekg(0);   // 처음으로
    std::string content;
    std::getline(file, content);
    std::cout << "수정 후: " << content << std::endl;

    file.close();

    return 0;
}
```

### 파일 크기 확인

```cpp
#include <iostream>
#include <fstream>

long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return -1;
    }
    return file.tellg();
}

int main() {
    // 테스트 파일 생성
    std::ofstream out("size_test.txt");
    out << "Hello, World!";
    out.close();

    long size = getFileSize("size_test.txt");
    std::cout << "파일 크기: " << size << " bytes" << std::endl;

    return 0;
}
```

---

## 10. 스트림 상태 확인

### 상태 플래그

```cpp
#include <iostream>
#include <fstream>
#include <sstream>

void checkStreamState(std::ios& stream) {
    std::cout << "good(): " << stream.good() << std::endl;
    std::cout << "eof():  " << stream.eof() << std::endl;
    std::cout << "fail(): " << stream.fail() << std::endl;
    std::cout << "bad():  " << stream.bad() << std::endl;
}

int main() {
    std::cout << std::boolalpha;

    // 정상 상태
    std::istringstream ss1("100");
    int num;
    ss1 >> num;
    std::cout << "=== 정상 읽기 후 ===" << std::endl;
    checkStreamState(ss1);

    // EOF 상태
    ss1 >> num;
    std::cout << "\n=== EOF 후 ===" << std::endl;
    checkStreamState(ss1);

    // 실패 상태
    std::istringstream ss2("abc");
    ss2 >> num;
    std::cout << "\n=== 잘못된 형식 ===" << std::endl;
    checkStreamState(ss2);

    // 상태 초기화
    ss2.clear();
    std::cout << "\n=== clear() 후 ===" << std::endl;
    checkStreamState(ss2);

    // 파일 열기 실패
    std::ifstream file("nonexistent.txt");
    std::cout << "\n=== 존재하지 않는 파일 ===" << std::endl;
    checkStreamState(file);

    return 0;
}
```

### 예외 활성화

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ifstream file;

    // 스트림 예외 활성화
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        file.open("nonexistent_file.txt");
        // 파일이 없으면 예외 발생
    }
    catch (const std::ios_base::failure& e) {
        std::cout << "파일 열기 실패: " << e.what() << std::endl;
    }

    return 0;
}
```

---

## 11. 문자열 스트림

### stringstream 활용

```cpp
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    // 문자열 -> 숫자 변환
    std::string numStr = "42 3.14 100";
    std::istringstream iss(numStr);

    int i;
    double d;
    int j;
    iss >> i >> d >> j;
    std::cout << "파싱: " << i << ", " << d << ", " << j << std::endl;

    // 숫자 -> 문자열 변환
    std::ostringstream oss;
    oss << "결과: " << 123 << " + " << 456 << " = " << (123 + 456);
    std::string result = oss.str();
    std::cout << result << std::endl;

    // CSV 파싱
    std::string csv = "Alice,25,90.5";
    std::istringstream csvStream(csv);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(csvStream, token, ',')) {
        tokens.push_back(token);
    }

    std::cout << "CSV 파싱: ";
    for (const auto& t : tokens) {
        std::cout << "[" << t << "] ";
    }
    std::cout << std::endl;

    // stringstream 재사용
    std::stringstream ss;
    ss << "Hello";
    std::cout << "1: " << ss.str() << std::endl;

    ss.str("");  // 내용 비우기
    ss.clear();  // 상태 초기화
    ss << "World";
    std::cout << "2: " << ss.str() << std::endl;

    return 0;
}
```

---

## 12. 실용적인 예제

### 설정 파일 파서

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

class ConfigParser {
private:
    std::map<std::string, std::string> config;

public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // 빈 줄, 주석 건너뛰기
            if (line.empty() || line[0] == '#') continue;

            // '=' 기준으로 분리
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // 공백 제거 (간단한 버전)
                config[key] = value;
            }
        }
        return true;
    }

    std::string get(const std::string& key,
                    const std::string& defaultValue = "") const {
        auto it = config.find(key);
        if (it != config.end()) {
            return it->second;
        }
        return defaultValue;
    }

    int getInt(const std::string& key, int defaultValue = 0) const {
        auto it = config.find(key);
        if (it != config.end()) {
            return std::stoi(it->second);
        }
        return defaultValue;
    }

    void display() const {
        for (const auto& [key, value] : config) {
            std::cout << key << " = " << value << std::endl;
        }
    }
};

int main() {
    // 설정 파일 생성
    std::ofstream out("config.ini");
    out << "# 서버 설정\n";
    out << "host=localhost\n";
    out << "port=8080\n";
    out << "max_connections=100\n";
    out << "debug=true\n";
    out.close();

    // 설정 파일 읽기
    ConfigParser config;
    if (config.load("config.ini")) {
        std::cout << "=== 설정 파일 ===" << std::endl;
        config.display();

        std::cout << "\n=== 개별 접근 ===" << std::endl;
        std::cout << "Host: " << config.get("host") << std::endl;
        std::cout << "Port: " << config.getInt("port") << std::endl;
        std::cout << "Timeout: " << config.getInt("timeout", 30) << std::endl;
    }

    return 0;
}
```

### CSV 파일 처리

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct Student {
    std::string name;
    int age;
    double score;
};

class CSVHandler {
public:
    static void write(const std::string& filename,
                      const std::vector<Student>& students) {
        std::ofstream file(filename);

        // 헤더
        file << "name,age,score\n";

        // 데이터
        for (const auto& s : students) {
            file << s.name << "," << s.age << "," << s.score << "\n";
        }
    }

    static std::vector<Student> read(const std::string& filename) {
        std::vector<Student> students;
        std::ifstream file(filename);

        std::string line;
        std::getline(file, line);  // 헤더 건너뛰기

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            Student s;
            std::string field;

            std::getline(iss, s.name, ',');
            std::getline(iss, field, ',');
            s.age = std::stoi(field);
            std::getline(iss, field, ',');
            s.score = std::stod(field);

            students.push_back(s);
        }

        return students;
    }
};

int main() {
    // CSV 쓰기
    std::vector<Student> students = {
        {"Alice", 20, 95.5},
        {"Bob", 22, 87.0},
        {"Charlie", 21, 91.2}
    };

    CSVHandler::write("students.csv", students);
    std::cout << "CSV 저장 완료" << std::endl;

    // CSV 읽기
    auto loaded = CSVHandler::read("students.csv");

    std::cout << "\n=== 학생 목록 ===" << std::endl;
    for (const auto& s : loaded) {
        std::cout << s.name << " (" << s.age << "세): "
                  << s.score << "점" << std::endl;
    }

    return 0;
}
```

---

## 13. 요약

| 개념 | 설명 |
|------|------|
| `try-catch` | 예외 처리 블록 |
| `throw` | 예외 발생 |
| `noexcept` | 예외를 던지지 않음 보장 |
| `std::exception` | 표준 예외 기본 클래스 |
| `ifstream` | 파일 읽기 스트림 |
| `ofstream` | 파일 쓰기 스트림 |
| `fstream` | 읽기/쓰기 스트림 |
| `stringstream` | 문자열 스트림 |
| `seekg/seekp` | 파일 위치 이동 |
| `tellg/tellp` | 현재 위치 확인 |

---

## 14. 연습 문제

### 연습 1: 로그 파일 클래스

날짜/시간과 함께 메시지를 기록하는 Logger 클래스를 작성하세요.

### 연습 2: 예외 계층 구조

데이터베이스 관련 예외 클래스 계층구조를 설계하세요.
(ConnectionError, QueryError, AuthenticationError 등)

### 연습 3: JSON 파서 (간단 버전)

간단한 key-value JSON을 파싱하는 클래스를 작성하세요.
(예: `{"name": "Alice", "age": 25}`)

---

## 다음 단계

[14_Smart_Pointers_Memory.md](./14_Smart_Pointers_Memory.md)에서 스마트 포인터를 배워봅시다!
