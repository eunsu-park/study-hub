# 예외 처리와 파일 I/O

**이전**: [STL 알고리즘과 반복자](./12_STL_Algorithms_and_Iterators.md) | **다음**: [CMake와 빌드 기초](./14_CMake_and_Build_Basics.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `try`, `throw`, `catch`를 사용하여 프로그램 충돌 없이 예외 상황을 처리한다
2. `std::exception`을 상속하여 커스텀 예외 클래스를 설계한다
3. `noexcept`를 적용하여 예외를 던지지 않는다고 선언한다
4. 4단계 예외 안전성을 설명하고 RAII를 구현하여 강한 보장을 달성한다
5. `ifstream`, `ofstream`, `fstream`과 다양한 열기 모드로 텍스트 파일을 읽고 쓴다
6. `read()`/`write()`로 바이너리 파일 I/O를 수행하고 `seekg`/`seekp`로 파일 위치를 제어한다
7. `stringstream`과 `getline`을 사용하여 구조화된 데이터(CSV, 설정 파일)를 파싱한다

---

모든 실세계 프로그램은 예상치 못한 상황을 다뤄야 합니다. 예외 처리는 오류 감지 코드와 오류 복구 코드를 분리하는 구조화된 방법을 제공하여 주요 로직을 깔끔하게 유지합니다. 파일 I/O와 결합하면 데이터를 영속화하고 다른 시스템과 통신하며 상태를 조용히 오염시키는 대신 실패에서 우아하게 복구할 수 있습니다.

## 1. 예외 처리란?

예외는 프로그램 실행 중 발생하는 비정상적인 상황입니다. C++은 try-catch 구문으로 예외를 처리합니다.

> **예외 처리 흐름**
>
> - **try 블록** -- 예외 발생 시 --> **throw** --> 예외 전파 --> **catch 블록**
> - **try 블록** -- 예외 없음 --> **정상 종료**

---

## 2. try, throw, catch

### 기본 구문

```cpp
#include <iostream>
#include <string>

double divide(double a, double b) {
    if (b == 0) {
        throw std::string("Cannot divide by zero");  // 예외 던지기
    }
    return a / b;
}

int main() {
    try {
        std::cout << divide(10, 2) << std::endl;  // 5
        std::cout << divide(10, 0) << std::endl;  // 예외 발생!
        std::cout << "이 줄은 실행되지 않습니다" << std::endl;
    }
    catch (const std::string& e) {
        std::cout << "Error: " << e << std::endl;
    }

    std::cout << "Program continues" << std::endl;

    return 0;
}
```

출력:
```
5
Error: Cannot divide by zero
Program continues
```

### 다중 catch 블록

```cpp
#include <iostream>
#include <stdexcept>

void process(int value) {
    if (value < 0) {
        throw std::invalid_argument("Negative numbers not allowed");
    }
    if (value > 100) {
        throw std::out_of_range("Cannot exceed 100");
    }
    if (value == 0) {
        throw 0;  // int 타입 예외
    }
    std::cout << "Value: " << value << std::endl;
}

int main() {
    int tests[] = {50, -10, 150, 0};

    for (int val : tests) {
        try {
            process(val);
        }
        catch (const std::invalid_argument& e) {
            std::cout << "Invalid argument: " << e.what() << std::endl;
        }
        catch (const std::out_of_range& e) {
            std::cout << "Out of range: " << e.what() << std::endl;
        }
        catch (int e) {
            std::cout << "Integer exception: " << e << std::endl;
        }
        catch (...) {  // 모든 예외 포착
            std::cout << "Unknown exception" << std::endl;
        }
    }

    return 0;
}
```

출력:
```
Value: 50
Invalid argument: Negative numbers not allowed
Out of range: Cannot exceed 100
Integer exception: 0
```

---

## 3. 표준 예외 클래스

> **std::exception**
>
> - **logic_error**
>   - invalid_argument
>   - out_of_range
> - **runtime_error**
>   - overflow_error
>   - underflow_error
> - **bad_alloc**

### 주요 예외 클래스

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include <new>

int main() {
    // logic_error 계열 (프로그래머 실수)
    try {
        throw std::invalid_argument("Invalid argument");
    } catch (const std::exception& e) {
        std::cout << "invalid_argument: " << e.what() << std::endl;
    }

    try {
        throw std::out_of_range("Out of range");
    } catch (const std::exception& e) {
        std::cout << "out_of_range: " << e.what() << std::endl;
    }

    try {
        throw std::length_error("Length error");
    } catch (const std::exception& e) {
        std::cout << "length_error: " << e.what() << std::endl;
    }

    // runtime_error 계열 (런타임 오류)
    try {
        throw std::runtime_error("Runtime error");
    } catch (const std::exception& e) {
        std::cout << "runtime_error: " << e.what() << std::endl;
    }

    try {
        throw std::overflow_error("Overflow");
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
        : message("File not found: " + filename) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

class InvalidFormatError : public std::exception {
private:
    std::string message;

public:
    InvalidFormatError(const std::string& detail)
        : message("Invalid format: " + detail) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

void readConfig(const std::string& filename) {
    if (filename.empty()) {
        throw FileNotFoundError("(empty filename)");
    }
    if (filename.find(".cfg") == std::string::npos) {
        throw InvalidFormatError("Extension must be .cfg");
    }
    std::cout << filename << " read successfully" << std::endl;
}

int main() {
    std::string files[] = {"", "data.txt", "config.cfg"};

    for (const auto& f : files) {
        try {
            readConfig(f);
        }
        catch (const FileNotFoundError& e) {
            std::cout << "[File Error] " << e.what() << std::endl;
        }
        catch (const InvalidFormatError& e) {
            std::cout << "[Format Error] " << e.what() << std::endl;
        }
    }

    return 0;
}
```

---

## 4. 예외 재던지기와 noexcept

### 예외 재던지기

```cpp
#include <iostream>
#include <stdexcept>

void lowLevel() {
    throw std::runtime_error("Low-level error");
}

void midLevel() {
    try {
        lowLevel();
    }
    catch (const std::exception& e) {
        std::cout << "[Mid-level] Exception detected: " << e.what() << std::endl;
        throw;  // 예외 재던지기 (상위로 전파)
    }
}

void highLevel() {
    try {
        midLevel();
    }
    catch (const std::exception& e) {
        std::cout << "[High-level] Final handling: " << e.what() << std::endl;
    }
}

int main() {
    highLevel();
    return 0;
}
```

출력:
```
[Mid-level] Exception detected: Low-level error
[High-level] Final handling: Low-level error
```

### noexcept 지정자

```cpp
#include <iostream>

// 예외를 던지지 않음을 보장
void safeFunction() noexcept {
    // 예외를 던지면 std::terminate() 호출
    std::cout << "Safe function" << std::endl;
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
        throw std::runtime_error("Error");
    }
};

int main() {
    std::cout << std::boolalpha;

    // noexcept 확인
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
| Strong(강한) | 예외 시 원래 상태로 복원 |
| Basic(기본) | 예외 후에도 유효한 상태 유지 |
| No guarantee | 예외 시 미정의 상태 |

### 예외 안전성을 위한 RAII

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
            throw std::runtime_error("Failed to open file");
        }
        std::cout << "File opened" << std::endl;
    }

    ~FileHandler() {
        if (file) {
            fclose(file);
            std::cout << "File closed" << std::endl;
        }
    }

    void write(const char* data) {
        if (fputs(data, file) == EOF) {
            throw std::runtime_error("Write failed");
        }
    }

    // 복사 금지
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};

void processFile() {
    FileHandler fh("test.txt", "w");  // RAII: 생성자에서 열기
    fh.write("Hello, World!\n");
    throw std::runtime_error("Exception in middle!");
    fh.write("이 줄은 실행되지 않습니다");
}  // RAII: 소멸자에서 자동으로 닫힘

int main() {
    try {
        processFile();
    }
    catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
```

출력:
```
File opened
File closed
Exception: Exception in middle!
```

---

## 6. 파일 I/O 기초

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
        std::cout << "File write complete" << std::endl;
    }

    // 파일 읽기
    std::ifstream inFile("example.txt");
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            std::cout << "Read: " << line << std::endl;
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
    f1 << "New content" << std::endl;
    f1.close();

    // 추가 모드
    std::ofstream f2("test.txt", std::ios::app);
    f2 << "Appended content" << std::endl;
    f2.close();

    // 바이너리 모드
    std::ofstream f3("data.bin", std::ios::binary);
    int num = 12345;
    f3.write(reinterpret_cast<char*>(&num), sizeof(num));
    f3.close();

    // 읽기+쓰기 모드
    std::fstream f4("test.txt", std::ios::in | std::ios::out);

    // 끝에서 시작 (추가)
    std::ofstream f5("test.txt", std::ios::ate);

    // 기존 내용 삭제
    std::ofstream f6("test.txt", std::ios::trunc);

    return 0;
}
```

| 모드 | 설명 |
|------|------|
| `ios::in` | 읽기 |
| `ios::out` | 쓰기 |
| `ios::app` | 끝에 추가 |
| `ios::ate` | 끝에서 시작 |
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

    // 방법 1: >> 연산자 (공백 구분)
    std::ifstream f1("data.txt");
    std::string name;
    int age;
    double score;
    std::cout << "=== >> operator ===" << std::endl;
    while (f1 >> name >> age >> score) {
        std::cout << name << ", " << age << ", " << score << std::endl;
    }
    f1.close();

    // 방법 2: getline (줄 단위)
    std::ifstream f2("data.txt");
    std::string line;
    std::cout << "\n=== getline ===" << std::endl;
    while (std::getline(f2, line)) {
        std::cout << "Line: " << line << std::endl;
    }
    f2.close();

    // 방법 3: getline + stringstream
    std::ifstream f3("data.txt");
    std::cout << "\n=== stringstream ===" << std::endl;
    while (std::getline(f3, line)) {
        std::istringstream iss(line);
        iss >> name >> age >> score;
        std::cout << "Name=" << name << ", Age=" << age
                  << ", Score=" << score << std::endl;
    }
    f3.close();

    // 방법 4: 파일 전체 읽기
    std::ifstream f4("data.txt");
    std::stringstream buffer;
    buffer << f4.rdbuf();
    std::string content = buffer.str();
    std::cout << "\n=== Full content ===" << std::endl;
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
    std::cout << "Character by character: ";
    while (in.get(c)) {
        if (c == '\n') {
            std::cout << "[LF]";
        } else {
            std::cout << c;
        }
    }
    std::cout << std::endl;

    // peek()으로 미리 보기
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

    std::cout << "Record size: " << sizeof(Record) << " bytes" << std::endl;

    // 바이너리 읽기
    std::ifstream in("records.bin", std::ios::binary);

    Record record;
    std::cout << "\n=== Reading records ===" << std::endl;
    while (in.read(reinterpret_cast<char*>(&record), sizeof(Record))) {
        std::cout << "ID: " << record.id
                  << ", Name: " << record.name
                  << ", Score: " << record.score << std::endl;
    }
    in.close();

    // 특정 레코드에 임의 접근
    std::ifstream in2("records.bin", std::ios::binary);

    // 두 번째 레코드로 이동 (0부터 시작)
    in2.seekg(1 * sizeof(Record));
    in2.read(reinterpret_cast<char*>(&record), sizeof(Record));
    std::cout << "\nSecond record: " << record.name << std::endl;

    in2.close();

    return 0;
}
```

### Vector 저장/로드

```cpp
#include <iostream>
#include <fstream>
#include <vector>

void saveVector(const std::string& filename, const std::vector<int>& vec) {
    std::ofstream out(filename, std::ios::binary);

    // 먼저 크기 저장
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
    std::cout << "Save complete" << std::endl;

    std::vector<int> loaded = loadVector("vector.bin");
    std::cout << "Loaded data: ";
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
    std::cout << "Start position: " << in.tellg() << std::endl;

    // 위치 5로 이동 (처음부터)
    in.seekg(5, std::ios::beg);
    char c;
    in.get(c);
    std::cout << "Character at position 5: " << c << std::endl;

    // 현재에서 3칸 앞으로 이동
    in.seekg(3, std::ios::cur);
    in.get(c);
    std::cout << "3 positions forward: " << c << std::endl;

    // 끝에서 2칸 앞
    in.seekg(-2, std::ios::end);
    in.get(c);
    std::cout << "2 before end: " << c << std::endl;

    in.close();

    // 쓰기 위치 제어
    std::fstream file("position.txt", std::ios::in | std::ios::out);

    file.seekp(10);  // 위치 10으로 이동
    file << "XYZ";   // ABC를 XYZ로 덮어쓰기

    file.seekg(0);   // 처음으로
    std::string content;
    std::getline(file, content);
    std::cout << "After modification: " << content << std::endl;

    file.close();

    return 0;
}
```

### 파일 크기 얻기

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
    std::cout << "File size: " << size << " bytes" << std::endl;

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
    std::cout << "=== After normal read ===" << std::endl;
    checkStreamState(ss1);

    // EOF 상태
    ss1 >> num;
    std::cout << "\n=== After EOF ===" << std::endl;
    checkStreamState(ss1);

    // 실패 상태
    std::istringstream ss2("abc");
    ss2 >> num;
    std::cout << "\n=== Invalid format ===" << std::endl;
    checkStreamState(ss2);

    // 상태 초기화
    ss2.clear();
    std::cout << "\n=== After clear() ===" << std::endl;
    checkStreamState(ss2);

    // 파일 열기 실패
    std::ifstream file("nonexistent.txt");
    std::cout << "\n=== Non-existent file ===" << std::endl;
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
        std::cout << "Failed to open file: " << e.what() << std::endl;
    }

    return 0;
}
```

---

## 11. 문자열 스트림

### stringstream 사용법

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
    std::cout << "Parsed: " << i << ", " << d << ", " << j << std::endl;

    // 숫자 -> 문자열 변환
    std::ostringstream oss;
    oss << "Result: " << 123 << " + " << 456 << " = " << (123 + 456);
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

    std::cout << "CSV parsed: ";
    for (const auto& t : tokens) {
        std::cout << "[" << t << "] ";
    }
    std::cout << std::endl;

    // stringstream 재사용
    std::stringstream ss;
    ss << "Hello";
    std::cout << "1: " << ss.str() << std::endl;

    ss.str("");  // 내용 초기화
    ss.clear();  // 상태 초기화
    ss << "World";
    std::cout << "2: " << ss.str() << std::endl;

    return 0;
}
```

---

## 12. 실용 예제

### 설정 파일 파서

```cpp
class ConfigParser {
private:
    std::map<std::string, std::string> config;
public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                config[line.substr(0, pos)] = line.substr(pos + 1);
            }
        }
        return true;
    }
    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = config.find(key);
        return (it != config.end()) ? it->second : def;
    }
};
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
    std::cout << "CSV saved" << std::endl;

    // CSV 읽기
    auto loaded = CSVHandler::read("students.csv");

    std::cout << "\n=== Student list ===" << std::endl;
    for (const auto& s : loaded) {
        std::cout << s.name << " (" << s.age << " years old): "
                  << s.score << " points" << std::endl;
    }

    return 0;
}
```

---

## 13. 요약

| 개념 | 설명 |
|------|------|
| `try-catch` | 예외 처리 블록 |
| `throw` | 예외 던지기 |
| `noexcept` | 예외 안 던짐 보장 |
| `std::exception` | 표준 예외 기본 클래스 |
| `ifstream` | 파일 읽기 스트림 |
| `ofstream` | 파일 쓰기 스트림 |
| `fstream` | 읽기/쓰기 스트림 |
| `stringstream` | 문자열 스트림 |
| `seekg/seekp` | 파일 위치 이동 |
| `tellg/tellp` | 현재 위치 확인 |

---

## 14. 연습문제

### 연습문제 1: 로그 파일 클래스
날짜/시간과 함께 메시지를 기록하는 Logger 클래스를 작성하세요.

### 연습문제 2: 예외 계층 구조
데이터베이스 관련 예외 클래스 계층을 설계하세요. (ConnectionError, QueryError, AuthenticationError 등)

### 연습문제 3: JSON 파서 (간단 버전)
간단한 키-값 JSON을 파싱하는 클래스를 작성하세요. (예: `{"name": "Alice", "age": 25}`)

---

## 다음 단계

[CMake와 빌드 기초](./14_CMake_and_Build_Basics.md)에서 CMake와 빌드 기초에 대해 알아봅시다!
