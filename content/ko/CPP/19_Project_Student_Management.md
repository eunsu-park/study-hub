# 19. 프로젝트: 학생 관리 시스템

**이전**: [C++ 디자인 패턴](./18_Design_Patterns.md) | **다음**: [CMake와 빌드 시스템](./20_CMake_and_Build_Systems.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 관심사의 명확한 분리(데이터, 저장소, UI)를 갖춘 멀티 클래스 C++ 애플리케이션을 설계한다
2. 효율적인 조회, 정렬, 고유성 보장을 위해 STL 컨테이너(`map`, `vector`, `set`)를 적용한다
3. 입력 검증(input validation)과 의미 있는 오류 메시지를 포함한 CRUD 연산을 구현한다
4. `stringstream`을 사용하여 구조화된 데이터를 CSV 파일로 직렬화(serialize)하고 역직렬화(deserialize)한다
5. 예외 처리(exception handling)로 잘못된 입력을 우아하게 처리하는 메뉴 기반 CLI를 구축한다
6. 연산자 오버로딩(operator overloading)(`<<`, `<`, `==`)을 사용하여 사용자 정의 타입을 STL 알고리즘 및 I/O와 통합한다
7. 완전한 프로젝트에서 const 정확성(const correctness), RAII, 모던 C++ 모범 사례를 연습한다

---

지금까지 각 레슨은 C++ 기능을 개별적으로 다루었습니다. 이 프로젝트는 그것들을 하나의 완결된 작동 애플리케이션으로 통합합니다. 학생 관리 시스템을 처음부터 구축하면 클래스를 어떻게 구조화할지, 어떤 컨테이너를 선택할지, 오류를 어떻게 처리할지와 같은 설계 결정을 직접 내려야 하며, 이는 실제 소프트웨어 개발의 도전과 유사합니다. 이러한 프로젝트를 통해 개별적으로 습득한 지식이 진정한 실력으로 발전합니다.

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [요구사항](#2-요구사항)
3. [클래스 설계](#3-클래스-설계)
4. [Student 클래스](#4-student-클래스)
5. [Database 클래스](#5-database-클래스)
6. [파일 I/O 및 직렬화](#6-파일-io-및-직렬화)
7. [예외 처리](#7-예외-처리)
8. [메뉴 인터페이스](#8-메뉴-인터페이스)
9. [전체 구현](#9-전체-구현)
10. [테스트 및 사용법](#10-테스트-및-사용법)

---

## 1. 프로젝트 개요

사용자가 다음을 수행할 수 있는 **학생 관리 시스템**을 구축합니다:
- 학생 레코드 추가, 제거, 업데이트
- 이름, ID 또는 GPA로 학생 검색 및 필터링
- 통계 계산(평균 GPA, 상위 학생)
- 파일에 데이터 저장/로드
- 에러를 우아하게 처리

```
┌─────────────────────────────────────────────────────────────┐
│           Student Management System Architecture            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                                          │
│  │     main()    │  (Menu-driven CLI)                       │
│  └───────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐       ┌─────────────────┐                │
│  │   Database    │◄──────┤ StudentDatabase │                │
│  │   (singleton) │       │ (manages all    │                │
│  └───────┬───────┘       │  students)      │                │
│          │               └─────────────────┘                │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐                                          │
│  │    Student    │  (Data class)                            │
│  │  - id, name   │                                          │
│  │  - age, gpa   │                                          │
│  └───────────────┘                                          │
│                                                             │
│  STL Usage:                                                 │
│    - map<int, Student>  (ID → Student lookup)               │
│    - vector<Student>    (sorted results)                    │
│    - set<string>        (unique names)                      │
│                                                             │
│  File I/O:                                                  │
│    - Save to text file (CSV-like format)                    │
│    - Load from file (deserialization)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 요구사항

### 기능 요구사항
1. **CRUD 연산:** 학생 레코드 생성(Create), 읽기(Read), 업데이트(Update), 삭제(Delete)
2. **검색:** ID, 이름 또는 GPA 범위로 학생 찾기
3. **통계:** 평균 GPA 계산, 상위 N명 학생 찾기
4. **영속성(Persistence):** 텍스트 파일로부터 데이터 저장/로드
5. **검증:** 유효한 ID, GPA(0.0–4.0), 나이(> 0) 보장

### 기술 요구사항
- **STL 컨테이너** 사용 (map, vector, set)
- **스마트 포인터** 사용 (해당되는 경우 shared_ptr 또는 unique_ptr)
- **예외 처리** 구현 (사용자 정의 예외)
- **RAII** 원칙 따르기
- **const 정확성** 사용
- **연산자 오버로딩** 구현 (Student 비교용)

---

## 3. 클래스 설계

### 3.1 Student 클래스
속성을 가진 단일 학생을 표현:
- `id` (int, 고유값)
- `name` (string)
- `age` (int)
- `gpa` (double, 0.0–4.0)

**메서드:**
- 검증을 포함한 생성자
- Getter/Setter
- 정렬 및 검색을 위한 연산자 오버로딩(<, ==)
- I/O를 위한 Friend 함수

### 3.2 StudentDatabase 클래스
학생 컬렉션 관리:
- ID로 O(log N) 조회를 위한 `std::map<int, Student>`
- 메서드: add, remove, update, search, save, load
- 잘못된 연산에 대한 예외 처리

---

## 4. Student 클래스

### 4.1 헤더 파일 (Student.h)

```cpp
#ifndef STUDENT_H
#define STUDENT_H

#include <string>
#include <iostream>
#include <stdexcept>

class Student {
private:
    int id;
    std::string name;
    int age;
    double gpa;

public:
    // Constructor with validation
    Student(int id, const std::string& name, int age, double gpa);

    // Default constructor (for map insertion)
    Student() : id(0), name(""), age(0), gpa(0.0) {}

    // Getters
    int getId() const { return id; }
    std::string getName() const { return name; }
    int getAge() const { return age; }
    double getGpa() const { return gpa; }

    // Setters (with validation)
    void setName(const std::string& newName);
    void setAge(int newAge);
    void setGpa(double newGpa);

    // Operator overloading
    bool operator<(const Student& other) const;  // Compare by GPA (descending)
    bool operator==(const Student& other) const; // Compare by ID

    // Friend function for output
    friend std::ostream& operator<<(std::ostream& os, const Student& s);

    // Serialization
    std::string serialize() const;
    static Student deserialize(const std::string& line);
};

#endif // STUDENT_H
```

### 4.2 구현 파일 (Student.cpp)

```cpp
#include "Student.h"
#include <sstream>
#include <iomanip>

Student::Student(int id, const std::string& name, int age, double gpa)
    : id(id), name(name), age(age), gpa(gpa) {
    // Validation
    if (id <= 0) {
        throw std::invalid_argument("ID must be positive");
    }
    if (name.empty()) {
        throw std::invalid_argument("Name cannot be empty");
    }
    if (age <= 0 || age > 120) {
        throw std::invalid_argument("Age must be between 1 and 120");
    }
    if (gpa < 0.0 || gpa > 4.0) {
        throw std::invalid_argument("GPA must be between 0.0 and 4.0");
    }
}

void Student::setName(const std::string& newName) {
    if (newName.empty()) {
        throw std::invalid_argument("Name cannot be empty");
    }
    name = newName;
}

void Student::setAge(int newAge) {
    if (newAge <= 0 || newAge > 120) {
        throw std::invalid_argument("Age must be between 1 and 120");
    }
    age = newAge;
}

void Student::setGpa(double newGpa) {
    if (newGpa < 0.0 || newGpa > 4.0) {
        throw std::invalid_argument("GPA must be between 0.0 and 4.0");
    }
    gpa = newGpa;
}

bool Student::operator<(const Student& other) const {
    // Sort by GPA descending
    return gpa > other.gpa;
}

bool Student::operator==(const Student& other) const {
    return id == other.id;
}

std::ostream& operator<<(std::ostream& os, const Student& s) {
    os << "ID: " << std::setw(5) << s.id
       << " | Name: " << std::setw(20) << std::left << s.name
       << " | Age: " << std::setw(3) << s.age
       << " | GPA: " << std::fixed << std::setprecision(2) << s.gpa;
    return os;
}

std::string Student::serialize() const {
    std::ostringstream oss;
    oss << id << "," << name << "," << age << "," << std::fixed << std::setprecision(2) << gpa;
    return oss.str();
}

Student Student::deserialize(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    int id, age;
    double gpa;
    std::string name;

    // Parse CSV: id,name,age,gpa
    std::getline(iss, token, ',');
    id = std::stoi(token);

    std::getline(iss, name, ',');

    std::getline(iss, token, ',');
    age = std::stoi(token);

    std::getline(iss, token, ',');
    gpa = std::stod(token);

    return Student(id, name, age, gpa);
}
```

---

## 5. Database 클래스

### 5.1 헤더 파일 (StudentDatabase.h)

```cpp
#ifndef STUDENTDATABASE_H
#define STUDENTDATABASE_H

#include "Student.h"
#include <map>
#include <vector>
#include <memory>
#include <fstream>

class StudentDatabase {
private:
    std::map<int, Student> students;  // ID → Student
    std::string filename;

public:
    StudentDatabase(const std::string& filename = "students.csv")
        : filename(filename) {}

    // CRUD operations
    void addStudent(const Student& student);
    void removeStudent(int id);
    void updateStudent(int id, const Student& updatedStudent);
    Student getStudent(int id) const;

    // Search and filter
    std::vector<Student> searchByName(const std::string& name) const;
    std::vector<Student> filterByGpa(double minGpa, double maxGpa) const;

    // Statistics
    double averageGpa() const;
    std::vector<Student> topNStudents(int n) const;

    // Display
    void displayAll() const;

    // File I/O
    void saveToFile() const;
    void loadFromFile();

    // Utility
    size_t size() const { return students.size(); }
    bool empty() const { return students.empty(); }
};

#endif // STUDENTDATABASE_H
```

### 5.2 구현 파일 (StudentDatabase.cpp)

```cpp
#include "StudentDatabase.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

void StudentDatabase::addStudent(const Student& student) {
    int id = student.getId();
    if (students.find(id) != students.end()) {
        throw std::runtime_error("Student with ID " + std::to_string(id) + " already exists");
    }
    students[id] = student;
    std::cout << "Student added successfully.\n";
}

void StudentDatabase::removeStudent(int id) {
    auto it = students.find(id);
    if (it == students.end()) {
        throw std::runtime_error("Student with ID " + std::to_string(id) + " not found");
    }
    students.erase(it);
    std::cout << "Student removed successfully.\n";
}

void StudentDatabase::updateStudent(int id, const Student& updatedStudent) {
    auto it = students.find(id);
    if (it == students.end()) {
        throw std::runtime_error("Student with ID " + std::to_string(id) + " not found");
    }
    it->second = updatedStudent;
    std::cout << "Student updated successfully.\n";
}

Student StudentDatabase::getStudent(int id) const {
    auto it = students.find(id);
    if (it == students.end()) {
        throw std::runtime_error("Student with ID " + std::to_string(id) + " not found");
    }
    return it->second;
}

std::vector<Student> StudentDatabase::searchByName(const std::string& name) const {
    std::vector<Student> results;
    for (const auto& [id, student] : students) {
        if (student.getName().find(name) != std::string::npos) {
            results.push_back(student);
        }
    }
    return results;
}

std::vector<Student> StudentDatabase::filterByGpa(double minGpa, double maxGpa) const {
    std::vector<Student> results;
    for (const auto& [id, student] : students) {
        double gpa = student.getGpa();
        if (gpa >= minGpa && gpa <= maxGpa) {
            results.push_back(student);
        }
    }
    return results;
}

double StudentDatabase::averageGpa() const {
    if (students.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& [id, student] : students) {
        sum += student.getGpa();
    }
    return sum / students.size();
}

std::vector<Student> StudentDatabase::topNStudents(int n) const {
    std::vector<Student> all;
    for (const auto& [id, student] : students) {
        all.push_back(student);
    }

    // Sort by GPA descending (using operator<)
    std::sort(all.begin(), all.end());

    // Return top N
    if (n > static_cast<int>(all.size())) {
        n = all.size();
    }
    return std::vector<Student>(all.begin(), all.begin() + n);
}

void StudentDatabase::displayAll() const {
    if (students.empty()) {
        std::cout << "No students in database.\n";
        return;
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Total Students: " << students.size() << "\n";
    std::cout << std::string(70, '=') << "\n";
    for (const auto& [id, student] : students) {
        std::cout << student << "\n";
    }
    std::cout << std::string(70, '=') << "\n\n";
}

void StudentDatabase::saveToFile() const {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    for (const auto& [id, student] : students) {
        ofs << student.serialize() << "\n";
    }

    std::cout << "Data saved to " << filename << "\n";
}

void StudentDatabase::loadFromFile() {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cout << "No existing file found. Starting with empty database.\n";
        return;
    }

    students.clear();
    std::string line;
    int count = 0;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        try {
            Student s = Student::deserialize(line);
            students[s.getId()] = s;
            count++;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << " (" << e.what() << ")\n";
        }
    }

    std::cout << "Loaded " << count << " students from " << filename << "\n";
}
```

---

## 6. 파일 I/O 및 직렬화

데이터는 CSV 형식으로 저장됩니다:

```
1,Alice Johnson,20,3.85
2,Bob Smith,22,3.20
3,Carol Lee,19,3.95
```

각 줄: `id,name,age,gpa`

`serialize()` 및 `deserialize()` 메서드가 변환을 처리합니다.

---

## 7. 예외 처리

```cpp
// Example usage with exception handling
try {
    Student s(101, "John Doe", 21, 3.75);
    db.addStudent(s);
} catch (const std::invalid_argument& e) {
    std::cerr << "Validation error: " << e.what() << "\n";
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << "\n";
} catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << "\n";
}
```

더 세밀한 에러 처리를 위해 사용자 정의 예외를 정의할 수 있습니다:

```cpp
class DatabaseException : public std::runtime_error {
public:
    explicit DatabaseException(const std::string& msg) : std::runtime_error(msg) {}
};
```

---

## 8. 메뉴 인터페이스

### 8.1 메뉴 표시

```cpp
void displayMenu() {
    std::cout << "\n========== Student Management System ==========\n";
    std::cout << "1. Add Student\n";
    std::cout << "2. Remove Student\n";
    std::cout << "3. Update Student\n";
    std::cout << "4. Display All Students\n";
    std::cout << "5. Search by Name\n";
    std::cout << "6. Filter by GPA Range\n";
    std::cout << "7. Show Average GPA\n";
    std::cout << "8. Show Top N Students\n";
    std::cout << "9. Save to File\n";
    std::cout << "10. Load from File\n";
    std::cout << "0. Exit\n";
    std::cout << "===============================================\n";
    std::cout << "Enter choice: ";
}
```

### 8.2 입력 헬퍼 함수

```cpp
int getIntInput(const std::string& prompt) {
    int value;
    std::cout << prompt;
    while (!(std::cin >> value)) {
        std::cin.clear();
        std::cin.ignore(10000, '\n');
        std::cout << "Invalid input. " << prompt;
    }
    std::cin.ignore(10000, '\n');
    return value;
}

double getDoubleInput(const std::string& prompt) {
    double value;
    std::cout << prompt;
    while (!(std::cin >> value)) {
        std::cin.clear();
        std::cin.ignore(10000, '\n');
        std::cout << "Invalid input. " << prompt;
    }
    std::cin.ignore(10000, '\n');
    return value;
}

std::string getStringInput(const std::string& prompt) {
    std::string value;
    std::cout << prompt;
    std::getline(std::cin, value);
    return value;
}
```

---

## 9. 전체 구현

### 9.1 메인 프로그램 (main.cpp)

```cpp
#include "Student.h"
#include "StudentDatabase.h"
#include <iostream>
#include <limits>

// Input helper functions
int getIntInput(const std::string& prompt);
double getDoubleInput(const std::string& prompt);
std::string getStringInput(const std::string& prompt);
void displayMenu();

int main() {
    StudentDatabase db("students.csv");

    // Load existing data
    try {
        db.loadFromFile();
    } catch (const std::exception& e) {
        std::cerr << "Error loading file: " << e.what() << "\n";
    }

    int choice;
    do {
        displayMenu();
        choice = getIntInput("");

        try {
            switch (choice) {
                case 1: { // Add Student
                    int id = getIntInput("Enter ID: ");
                    std::string name = getStringInput("Enter Name: ");
                    int age = getIntInput("Enter Age: ");
                    double gpa = getDoubleInput("Enter GPA: ");

                    Student s(id, name, age, gpa);
                    db.addStudent(s);
                    break;
                }
                case 2: { // Remove Student
                    int id = getIntInput("Enter ID to remove: ");
                    db.removeStudent(id);
                    break;
                }
                case 3: { // Update Student
                    int id = getIntInput("Enter ID to update: ");
                    Student oldStudent = db.getStudent(id);
                    std::cout << "Current record:\n" << oldStudent << "\n";

                    std::string name = getStringInput("Enter new Name (or press Enter to keep): ");
                    if (name.empty()) name = oldStudent.getName();

                    int age = getIntInput("Enter new Age (or 0 to keep): ");
                    if (age == 0) age = oldStudent.getAge();

                    double gpa = getDoubleInput("Enter new GPA (or -1 to keep): ");
                    if (gpa < 0) gpa = oldStudent.getGpa();

                    Student updatedStudent(id, name, age, gpa);
                    db.updateStudent(id, updatedStudent);
                    break;
                }
                case 4: { // Display All
                    db.displayAll();
                    break;
                }
                case 5: { // Search by Name
                    std::string name = getStringInput("Enter name to search: ");
                    auto results = db.searchByName(name);
                    std::cout << "\nFound " << results.size() << " student(s):\n";
                    for (const auto& s : results) {
                        std::cout << s << "\n";
                    }
                    break;
                }
                case 6: { // Filter by GPA
                    double minGpa = getDoubleInput("Enter minimum GPA: ");
                    double maxGpa = getDoubleInput("Enter maximum GPA: ");
                    auto results = db.filterByGpa(minGpa, maxGpa);
                    std::cout << "\nFound " << results.size() << " student(s):\n";
                    for (const auto& s : results) {
                        std::cout << s << "\n";
                    }
                    break;
                }
                case 7: { // Average GPA
                    double avg = db.averageGpa();
                    std::cout << "\nAverage GPA: " << std::fixed << std::setprecision(2) << avg << "\n";
                    break;
                }
                case 8: { // Top N Students
                    int n = getIntInput("Enter number of top students: ");
                    auto top = db.topNStudents(n);
                    std::cout << "\nTop " << top.size() << " student(s):\n";
                    for (const auto& s : top) {
                        std::cout << s << "\n";
                    }
                    break;
                }
                case 9: { // Save
                    db.saveToFile();
                    break;
                }
                case 10: { // Load
                    db.loadFromFile();
                    break;
                }
                case 0: { // Exit
                    std::cout << "Exiting...\n";
                    break;
                }
                default:
                    std::cout << "Invalid choice. Try again.\n";
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Validation error: " << e.what() << "\n";
        } catch (const std::runtime_error& e) {
            std::cerr << "Runtime error: " << e.what() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }

    } while (choice != 0);

    // Auto-save on exit
    try {
        db.saveToFile();
    } catch (const std::exception& e) {
        std::cerr << "Failed to save data: " << e.what() << "\n";
    }

    return 0;
}

// Input helper implementations
int getIntInput(const std::string& prompt) {
    int value;
    std::cout << prompt;
    while (!(std::cin >> value)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. " << prompt;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return value;
}

double getDoubleInput(const std::string& prompt) {
    double value;
    std::cout << prompt;
    while (!(std::cin >> value)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. " << prompt;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return value;
}

std::string getStringInput(const std::string& prompt) {
    std::string value;
    std::cout << prompt;
    std::getline(std::cin, value);
    return value;
}

void displayMenu() {
    std::cout << "\n========== Student Management System ==========\n";
    std::cout << "1. Add Student\n";
    std::cout << "2. Remove Student\n";
    std::cout << "3. Update Student\n";
    std::cout << "4. Display All Students\n";
    std::cout << "5. Search by Name\n";
    std::cout << "6. Filter by GPA Range\n";
    std::cout << "7. Show Average GPA\n";
    std::cout << "8. Show Top N Students\n";
    std::cout << "9. Save to File\n";
    std::cout << "10. Load from File\n";
    std::cout << "0. Exit\n";
    std::cout << "===============================================\n";
    std::cout << "Enter choice: ";
}
```

### 9.2 Makefile

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET = student_mgmt
OBJS = main.o Student.o StudentDatabase.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp Student.h StudentDatabase.h
	$(CXX) $(CXXFLAGS) -c main.cpp

Student.o: Student.cpp Student.h
	$(CXX) $(CXXFLAGS) -c Student.cpp

StudentDatabase.o: StudentDatabase.cpp StudentDatabase.h Student.h
	$(CXX) $(CXXFLAGS) -c StudentDatabase.cpp

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

## 10. 테스트 및 사용법

### 10.1 컴파일

```bash
make
```

### 10.2 사용 예시

```
$ ./student_mgmt

========== Student Management System ==========
1. Add Student
2. Remove Student
...
0. Exit
===============================================
Enter choice: 1
Enter ID: 101
Enter Name: Alice Johnson
Enter Age: 20
Enter GPA: 3.85
Student added successfully.

Enter choice: 1
Enter ID: 102
Enter Name: Bob Smith
Enter Age: 22
Enter GPA: 3.20
Student added successfully.

Enter choice: 4

======================================================================
Total Students: 2
======================================================================
ID:   101 | Name: Alice Johnson       | Age:  20 | GPA: 3.85
ID:   102 | Name: Bob Smith           | Age:  22 | GPA: 3.20
======================================================================

Enter choice: 7
Average GPA: 3.52

Enter choice: 8
Enter number of top students: 1
Top 1 student(s):
ID:   101 | Name: Alice Johnson       | Age:  20 | GPA: 3.85

Enter choice: 9
Data saved to students.csv

Enter choice: 0
Exiting...
Data saved to students.csv
```

### 10.3 파일 내용 (students.csv)

```
101,Alice Johnson,20,3.85
102,Bob Smith,22,3.20
```

---

## 연습 문제

### 연습 1: 수강 과목 등록 기능 추가

`Student` 클래스를 확장하여 수강 과목 목록을 지원하세요. 다음을 추가하세요:
- `std::vector<std::string> courses` 멤버.
- `enrollCourse(const std::string& course)` — 이미 등록되지 않은 경우에만 과목을 추가합니다 (`std::find` 사용).
- `dropCourse(const std::string& course)` — 과목을 제거합니다; 등록되지 않은 경우 `std::runtime_error`를 던집니다.
- `getCourses() const` — 벡터를 반환합니다.
- CSV 필드에서 과목을 `|`로 구분하여 지속 저장하도록 `serialize()` / `deserialize()`를 업데이트합니다.

두 과목을 등록하고 하나를 삭제한 후 직렬화(serialization) 왕복이 올바르게 동작하는지 확인하여 테스트하세요.

### 연습 2: StudentDatabase에 통계 기능 확장

`StudentDatabase`에 다음 메서드를 추가하세요:

1. `std::map<std::string, double> gpaDistribution() const` — 문자 등급(`"A"`, `"B"`, `"C"`, `"D"`, `"F"`)을 해당 범위의 학생 비율에 매핑한 맵을 반환합니다.
2. `double medianGpa() const` — 모든 학생의 중앙값(median) GPA를 반환합니다 (GPA 복사본을 정렬하고 중간값을 선택하세요; 짝수 크기 컬렉션도 처리하세요).
3. `std::vector<Student> belowAverage() const` — GPA가 학급 평균보다 엄격하게 낮은 모든 학생을 반환합니다.

`main.cpp`에 메뉴 옵션 11, 12, 13을 추가하여 이 메서드를 호출하고 결과를 표시하세요.

### 연습 3: 사용자 정의 예외(Custom Exception) 계층

프로젝트에서 `std::runtime_error`와 `std::invalid_argument`의 사용을 사용자 정의 예외 계층으로 교체하세요:

```cpp
class StudentException : public std::exception { ... };
class StudentNotFoundException : public StudentException { ... };
class DuplicateStudentException : public StudentException { ... };
class InvalidStudentDataException : public StudentException { ... };
```

각 예외는 설명적인 메시지를 포함해야 하며, 해당되는 경우 관련 학생 ID도 포함해야 합니다. `Student.cpp`와 `StudentDatabase.cpp` 모두에서 모든 throw 위치와 catch 블록을 업데이트하세요. `StudentException`을 잡으면 세 가지 파생 타입이 모두 처리되는지 확인하세요.

### 연습 4: 정렬 및 필터링 기능 향상

`StudentDatabase`에 세 가지 새로운 쿼리 메서드를 추가하세요:

1. `std::vector<Student> sortedByName() const` — 이름 알파벳순으로 정렬된 모든 학생을 반환합니다 (대소문자 무시).
2. `std::vector<Student> sortedByAge() const` — 나이 오름차순으로 정렬된 모든 학생을 반환합니다.
3. `std::vector<Student> filterByAgeRange(int minAge, int maxAge) const` — 나이가 범위 내에 있는 학생을 반환합니다 (경계 포함).

람다 비교자/술어(predicate)와 함께 `std::sort` 또는 `std::copy_if`를 사용하여 각 메서드를 구현하세요. 대응하는 메뉴 항목을 추가하세요.

### 연습 5: CSV 일괄 가져오기

`StudentDatabase`에 `int importFromCsv(const std::string& filename)` 메서드를 추가하세요:
1. 지정된 파일을 열고 모든 줄을 읽습니다.
2. 각 줄을 `Student`로 역직렬화(deserialize)하려고 시도합니다.
3. 검증에 실패한 줄은 건너뜁니다 (예외를 잡아 경고를 로깅).
4. ID가 이미 데이터베이스에 있는 학생은 건너뜁니다 (덮어쓰지 않음).
5. 성공적으로 가져온 학생 수를 반환합니다.

유효한 레코드 5개와 유효하지 않은 레코드 2개(예: 음수 GPA, 빈 이름)가 있는 테스트 CSV 파일을 작성하고, 가져오기를 실행하여 올바른 개수와 기존 레코드가 변경되지 않았음을 확인하세요.

---

## 네비게이션

**이전**: [C++ 디자인 패턴](./18_Design_Patterns.md) | **다음**: [CMake와 빌드 시스템](./20_CMake_and_Build_Systems.md)
