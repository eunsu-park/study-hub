# Exception Handling and File I/O

**Previous**: [Templates](./12_Templates.md) | **Next**: [Smart Pointers and Memory Management](./14_Smart_Pointers_Memory.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `try`, `throw`, and `catch` to handle exceptional conditions without crashing the program
2. Design custom exception classes by inheriting from `std::exception`
3. Apply `noexcept` to declare functions that guarantee not to throw
4. Explain the four levels of exception safety and implement RAII to achieve the strong guarantee
5. Read and write text files using `ifstream`, `ofstream`, and `fstream` with various open modes
6. Perform binary file I/O with `read()`/`write()` and control file position using `seekg`/`seekp`
7. Parse structured data (CSV, config files) using `stringstream` and `getline`

---

Every real-world program must deal with the unexpected--files that do not exist, networks that drop, and users who enter invalid data. Exception handling gives you a structured way to separate error-detection code from error-recovery code, keeping your main logic clean. Paired with file I/O, these skills let your programs persist data, communicate with other systems, and recover gracefully from failure instead of silently corrupting state.

## 1. What is Exception Handling?

Exceptions are abnormal situations that occur during program execution. C++ handles exceptions using try-catch syntax.

```
┌─────────────────────────────────────────────┐
│           Exception Handling Flow            │
└─────────────────────────────────────────────┘
                    │
     ┌──────────────┴──────────────┐
     ▼                              ▼
┌─────────┐                   ┌─────────┐
│   try   │ ─Exception─────▶  │  throw  │
│  block  │                   │         │
└─────────┘                   └─────────┘
     │                              │
     │ No exception                  │ Exception propagation
     ▼                              ▼
┌─────────┐                   ┌─────────┐
│ Normal  │                   │  catch  │
│  exit   │                   │  block  │
└─────────┘                   └─────────┘
```

---

## 2. try, throw, catch

### Basic Syntax

```cpp
#include <iostream>
#include <string>

double divide(double a, double b) {
    if (b == 0) {
        throw std::string("Cannot divide by zero");  // Throw exception
    }
    return a / b;
}

int main() {
    try {
        std::cout << divide(10, 2) << std::endl;  // 5
        std::cout << divide(10, 0) << std::endl;  // Exception thrown!
        std::cout << "This line won't execute" << std::endl;
    }
    catch (const std::string& e) {
        std::cout << "Error: " << e << std::endl;
    }

    std::cout << "Program continues" << std::endl;

    return 0;
}
```

Output:
```
5
Error: Cannot divide by zero
Program continues
```

### Multiple catch Blocks

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
        throw 0;  // int type exception
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
        catch (...) {  // Catch all exceptions
            std::cout << "Unknown exception" << std::endl;
        }
    }

    return 0;
}
```

Output:
```
Value: 50
Invalid argument: Negative numbers not allowed
Out of range: Cannot exceed 100
Integer exception: 0
```

---

## 3. Standard Exception Classes

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

### Main Exception Classes

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include <new>

int main() {
    // logic_error family (programmer mistakes)
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

    // runtime_error family (runtime errors)
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

    // bad_alloc (memory allocation failure)
    try {
        throw std::bad_alloc();
    } catch (const std::exception& e) {
        std::cout << "bad_alloc: " << e.what() << std::endl;
    }

    return 0;
}
```

### Inheriting exception Class

```cpp
#include <iostream>
#include <exception>
#include <string>

// Custom exception class
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

## 4. Exception Rethrowing and noexcept

### Exception Rethrowing

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
        throw;  // Rethrow exception (propagate upwards)
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

Output:
```
[Mid-level] Exception detected: Low-level error
[High-level] Final handling: Low-level error
```

### noexcept Specifier

```cpp
#include <iostream>

// Guarantees not to throw exceptions
void safeFunction() noexcept {
    // Throwing exception calls std::terminate()
    std::cout << "Safe function" << std::endl;
}

// Conditional noexcept
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

    // Check noexcept
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

## 5. Exception Safety

### Exception Safety Levels

| Level | Description |
|-------|-------------|
| No-throw | Never throws exceptions |
| Strong | Restores original state on exception |
| Basic | Maintains valid state after exception |
| No guarantee | Undefined state on exception |

### RAII for Exception Safety

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

// RAII class
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

    // Delete copy
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};

void processFile() {
    FileHandler fh("test.txt", "w");  // RAII: open in constructor
    fh.write("Hello, World!\n");
    throw std::runtime_error("Exception in middle!");
    fh.write("This line won't execute");
}  // RAII: automatically closed in destructor

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

Output:
```
File opened
File closed
Exception: Exception in middle!
```

---

## 6. File I/O Basics

### File Stream Classes

| Class | Purpose |
|-------|---------|
| `ifstream` | Read files |
| `ofstream` | Write files |
| `fstream` | Read/write |

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Write file
    std::ofstream outFile("example.txt");
    if (outFile.is_open()) {
        outFile << "Hello, File!" << std::endl;
        outFile << "Line 2" << std::endl;
        outFile << 42 << " " << 3.14 << std::endl;
        outFile.close();
        std::cout << "File write complete" << std::endl;
    }

    // Read file
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

### File Open Modes

```cpp
#include <iostream>
#include <fstream>

int main() {
    // Write mode (default: overwrite)
    std::ofstream f1("test.txt");
    f1 << "New content" << std::endl;
    f1.close();

    // Append mode
    std::ofstream f2("test.txt", std::ios::app);
    f2 << "Appended content" << std::endl;
    f2.close();

    // Binary mode
    std::ofstream f3("data.bin", std::ios::binary);
    int num = 12345;
    f3.write(reinterpret_cast<char*>(&num), sizeof(num));
    f3.close();

    // Read+write mode
    std::fstream f4("test.txt", std::ios::in | std::ios::out);

    // Start at end (append)
    std::ofstream f5("test.txt", std::ios::ate);

    // Truncate existing content
    std::ofstream f6("test.txt", std::ios::trunc);

    return 0;
}
```

| Mode | Description |
|------|-------------|
| `ios::in` | Read |
| `ios::out` | Write |
| `ios::app` | Append to end |
| `ios::ate` | Start at end |
| `ios::trunc` | Delete existing content |
| `ios::binary` | Binary mode |

---

## 7. File Reading Methods

### Various Reading Methods

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

int main() {
    // Create test file
    std::ofstream out("data.txt");
    out << "Alice 25 90.5\n";
    out << "Bob 30 85.0\n";
    out << "Charlie 28 92.3\n";
    out.close();

    // Method 1: >> operator (whitespace separated)
    std::ifstream f1("data.txt");
    std::string name;
    int age;
    double score;
    std::cout << "=== >> operator ===" << std::endl;
    while (f1 >> name >> age >> score) {
        std::cout << name << ", " << age << ", " << score << std::endl;
    }
    f1.close();

    // Method 2: getline (line by line)
    std::ifstream f2("data.txt");
    std::string line;
    std::cout << "\n=== getline ===" << std::endl;
    while (std::getline(f2, line)) {
        std::cout << "Line: " << line << std::endl;
    }
    f2.close();

    // Method 3: getline + stringstream
    std::ifstream f3("data.txt");
    std::cout << "\n=== stringstream ===" << std::endl;
    while (std::getline(f3, line)) {
        std::istringstream iss(line);
        iss >> name >> age >> score;
        std::cout << "Name=" << name << ", Age=" << age
                  << ", Score=" << score << std::endl;
    }
    f3.close();

    // Method 4: Read entire file
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

### Character-wise Reading

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ofstream out("chars.txt");
    out << "ABC\nDEF";
    out.close();

    std::ifstream in("chars.txt");
    char c;

    // get() one character at a time
    std::cout << "Character by character: ";
    while (in.get(c)) {
        if (c == '\n') {
            std::cout << "[LF]";
        } else {
            std::cout << c;
        }
    }
    std::cout << std::endl;

    // peek() to preview
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

## 8. Binary Files

### Binary Read/Write

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
    // Binary write
    std::ofstream out("records.bin", std::ios::binary);

    Record r1 = {1, "Alice", 95.5};
    Record r2 = {2, "Bob", 87.0};
    Record r3 = {3, "Charlie", 91.2};

    out.write(reinterpret_cast<char*>(&r1), sizeof(Record));
    out.write(reinterpret_cast<char*>(&r2), sizeof(Record));
    out.write(reinterpret_cast<char*>(&r3), sizeof(Record));
    out.close();

    std::cout << "Record size: " << sizeof(Record) << " bytes" << std::endl;

    // Binary read
    std::ifstream in("records.bin", std::ios::binary);

    Record record;
    std::cout << "\n=== Reading records ===" << std::endl;
    while (in.read(reinterpret_cast<char*>(&record), sizeof(Record))) {
        std::cout << "ID: " << record.id
                  << ", Name: " << record.name
                  << ", Score: " << record.score << std::endl;
    }
    in.close();

    // Random access to specific record
    std::ifstream in2("records.bin", std::ios::binary);

    // Move to second record (0-indexed)
    in2.seekg(1 * sizeof(Record));
    in2.read(reinterpret_cast<char*>(&record), sizeof(Record));
    std::cout << "\nSecond record: " << record.name << std::endl;

    in2.close();

    return 0;
}
```

### Save/Load Vector

```cpp
#include <iostream>
#include <fstream>
#include <vector>

void saveVector(const std::string& filename, const std::vector<int>& vec) {
    std::ofstream out(filename, std::ios::binary);

    // Save size first
    size_t size = vec.size();
    out.write(reinterpret_cast<char*>(&size), sizeof(size));

    // Save data
    out.write(reinterpret_cast<const char*>(vec.data()),
              size * sizeof(int));
    out.close();
}

std::vector<int> loadVector(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);

    // Read size
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Read data
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

## 9. File Position Control

### seekg, seekp, tellg, tellp

```cpp
#include <iostream>
#include <fstream>

int main() {
    // Create file
    std::ofstream out("position.txt");
    out << "0123456789ABCDEF";
    out.close();

    // Read position control
    std::ifstream in("position.txt");

    // Check current position
    std::cout << "Start position: " << in.tellg() << std::endl;

    // Move to position 5 (from beginning)
    in.seekg(5, std::ios::beg);
    char c;
    in.get(c);
    std::cout << "Character at position 5: " << c << std::endl;

    // Move 3 positions forward from current
    in.seekg(3, std::ios::cur);
    in.get(c);
    std::cout << "3 positions forward: " << c << std::endl;

    // 2 positions before end
    in.seekg(-2, std::ios::end);
    in.get(c);
    std::cout << "2 before end: " << c << std::endl;

    in.close();

    // Write position control
    std::fstream file("position.txt", std::ios::in | std::ios::out);

    file.seekp(10);  // Move to position 10
    file << "XYZ";   // Overwrite ABC with XYZ

    file.seekg(0);   // To beginning
    std::string content;
    std::getline(file, content);
    std::cout << "After modification: " << content << std::endl;

    file.close();

    return 0;
}
```

### Get File Size

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
    // Create test file
    std::ofstream out("size_test.txt");
    out << "Hello, World!";
    out.close();

    long size = getFileSize("size_test.txt");
    std::cout << "File size: " << size << " bytes" << std::endl;

    return 0;
}
```

---

## 10. Stream State Checking

### State Flags

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

    // Normal state
    std::istringstream ss1("100");
    int num;
    ss1 >> num;
    std::cout << "=== After normal read ===" << std::endl;
    checkStreamState(ss1);

    // EOF state
    ss1 >> num;
    std::cout << "\n=== After EOF ===" << std::endl;
    checkStreamState(ss1);

    // Failed state
    std::istringstream ss2("abc");
    ss2 >> num;
    std::cout << "\n=== Invalid format ===" << std::endl;
    checkStreamState(ss2);

    // State reset
    ss2.clear();
    std::cout << "\n=== After clear() ===" << std::endl;
    checkStreamState(ss2);

    // File open failure
    std::ifstream file("nonexistent.txt");
    std::cout << "\n=== Non-existent file ===" << std::endl;
    checkStreamState(file);

    return 0;
}
```

### Enable Exceptions

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ifstream file;

    // Enable stream exceptions
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        file.open("nonexistent_file.txt");
        // Exception thrown if file doesn't exist
    }
    catch (const std::ios_base::failure& e) {
        std::cout << "Failed to open file: " << e.what() << std::endl;
    }

    return 0;
}
```

---

## 11. String Streams

### stringstream Usage

```cpp
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    // String -> number conversion
    std::string numStr = "42 3.14 100";
    std::istringstream iss(numStr);

    int i;
    double d;
    int j;
    iss >> i >> d >> j;
    std::cout << "Parsed: " << i << ", " << d << ", " << j << std::endl;

    // Number -> string conversion
    std::ostringstream oss;
    oss << "Result: " << 123 << " + " << 456 << " = " << (123 + 456);
    std::string result = oss.str();
    std::cout << result << std::endl;

    // CSV parsing
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

    // stringstream reuse
    std::stringstream ss;
    ss << "Hello";
    std::cout << "1: " << ss.str() << std::endl;

    ss.str("");  // Clear content
    ss.clear();  // Reset state
    ss << "World";
    std::cout << "2: " << ss.str() << std::endl;

    return 0;
}
```

---

## 12. Practical Examples

### Config File Parser

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
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            // Split by '='
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // Simple whitespace removal
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
    // Create config file
    std::ofstream out("config.ini");
    out << "# Server configuration\n";
    out << "host=localhost\n";
    out << "port=8080\n";
    out << "max_connections=100\n";
    out << "debug=true\n";
    out.close();

    // Read config file
    ConfigParser config;
    if (config.load("config.ini")) {
        std::cout << "=== Config file ===" << std::endl;
        config.display();

        std::cout << "\n=== Individual access ===" << std::endl;
        std::cout << "Host: " << config.get("host") << std::endl;
        std::cout << "Port: " << config.getInt("port") << std::endl;
        std::cout << "Timeout: " << config.getInt("timeout", 30) << std::endl;
    }

    return 0;
}
```

### CSV File Processing

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

        // Header
        file << "name,age,score\n";

        // Data
        for (const auto& s : students) {
            file << s.name << "," << s.age << "," << s.score << "\n";
        }
    }

    static std::vector<Student> read(const std::string& filename) {
        std::vector<Student> students;
        std::ifstream file(filename);

        std::string line;
        std::getline(file, line);  // Skip header

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
    // Write CSV
    std::vector<Student> students = {
        {"Alice", 20, 95.5},
        {"Bob", 22, 87.0},
        {"Charlie", 21, 91.2}
    };

    CSVHandler::write("students.csv", students);
    std::cout << "CSV saved" << std::endl;

    // Read CSV
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

## 13. Summary

| Concept | Description |
|---------|-------------|
| `try-catch` | Exception handling block |
| `throw` | Throw exception |
| `noexcept` | Guarantee not to throw |
| `std::exception` | Standard exception base class |
| `ifstream` | File read stream |
| `ofstream` | File write stream |
| `fstream` | Read/write stream |
| `stringstream` | String stream |
| `seekg/seekp` | Move file position |
| `tellg/tellp` | Check current position |

---

## 14. Exercises

### Exercise 1: Log File Class

Write a Logger class that records messages with date/time.

### Exercise 2: Exception Hierarchy

Design a database-related exception class hierarchy.
(ConnectionError, QueryError, AuthenticationError, etc.)

### Exercise 3: JSON Parser (Simple Version)

Write a class that parses simple key-value JSON.
(Example: `{"name": "Alice", "age": 25}`)

---

## Next Steps

Let's learn about smart pointers in [14_Smart_Pointers_Memory.md](./14_Smart_Pointers_Memory.md)!
