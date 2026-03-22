# 프로젝트: 학생 관리 시스템

**이전**: [CMake와 빌드 기초](./14_CMake_and_Build_Basics.md) | **다음**: (마지막 레슨)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 명확한 관심사 분리(데이터, 저장소, UI)로 다중 클래스 C++ 애플리케이션을 설계한다
2. 효율적인 검색, 정렬, 고유성 보장을 위해 STL 컨테이너(`map`, `vector`, `set`)를 적용한다
3. 입력 검증과 의미 있는 오류 메시지를 포함한 CRUD 연산을 구현한다
4. `stringstream`을 사용하여 구조화된 데이터를 CSV 파일로 직렬화 및 역직렬화한다
5. 예외 처리로 잘못된 입력을 우아하게 처리하는 메뉴 기반 CLI를 구축한다
6. 연산자 오버로딩(`<<`, `<`, `==`)으로 커스텀 타입을 STL 알고리즘 및 I/O와 통합한다
7. 완전한 프로젝트에서 const 정확성, RAII, 모던 C++ 모범 사례를 실습한다

---

지금까지 각 레슨은 개별 C++ 기능을 분리하여 가르쳤습니다. 이 프로젝트는 그 모든 것을 응집력 있는 작동하는 애플리케이션으로 통합합니다. 처음부터 학생 관리 시스템을 구축하면 실제 소프트웨어 개발의 도전을 반영하는 설계 결정을 내려야 합니다.

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [요구사항](#2-요구사항)
3. [클래스 설계](#3-클래스-설계)
4. [Student 클래스](#4-student-클래스)
5. [Database 클래스](#5-database-클래스)
6. [파일 I/O와 직렬화](#6-파일-io와-직렬화)
7. [예외 처리](#7-예외-처리)
8. [메뉴 인터페이스](#8-메뉴-인터페이스)
9. [전체 구현](#9-전체-구현)
10. [테스트와 사용법](#10-테스트와-사용법)

---

## 1. 프로젝트 개요

다음 기능을 갖춘 **학생 관리 시스템**을 구축합니다:
- 학생 레코드 추가, 삭제, 수정
- 이름, ID, GPA로 학생 검색 및 필터링
- 통계 계산 (평균 GPA, 상위 학생)
- 파일로 데이터 저장/로드
- 오류 우아한 처리

> **아키텍처**
>
> - **main()** (메뉴 기반 CLI) --> **StudentDatabase** (모든 학생 관리) --> **Student** (데이터 클래스)
> - **STL 사용**: `map<int, Student>` (ID 검색), `vector<Student>` (정렬 결과), `set<string>` (고유 이름)
> - **파일 I/O**: 텍스트 파일(CSV 형식)로 저장/로드

---

## 2. 요구사항

### 기능 요구사항
1. **CRUD 연산:** 학생 레코드 생성, 읽기, 수정, 삭제
2. **검색:** ID, 이름, GPA 범위로 학생 찾기
3. **통계:** 평균 GPA 계산, 상위 N명 학생 찾기
4. **영속성:** 텍스트 파일에서 데이터 저장/로드
5. **검증:** 유효한 ID, GPA(0.0-4.0), 나이(> 0) 보장

### 기술 요구사항
- **STL 컨테이너** 사용 (map, vector, set)
- **예외 처리** 구현 (커스텀 예외)
- **RAII** 원칙 준수
- **const 정확성** 사용
- **연산자 오버로딩** 구현 (Student 비교)

---

## 3. 클래스 설계

### 3.1 Student 클래스
단일 학생을 나타냅니다:
- `id` (int, 고유), `name` (string), `age` (int), `gpa` (double, 0.0-4.0)
- 검증 포함 생성자, 게터/세터, 연산자 오버로딩, 직렬화

### 3.2 StudentDatabase 클래스
학생 컬렉션을 관리합니다:
- `std::map<int, Student>`로 O(log N) ID 검색
- 메서드: add, remove, update, search, save, load

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
    Student(int id, const std::string& name, int age, double gpa);
    Student() : id(0), name(""), age(0), gpa(0.0) {}

    int getId() const { return id; }
    std::string getName() const { return name; }
    int getAge() const { return age; }
    double getGpa() const { return gpa; }

    void setName(const std::string& newName);
    void setAge(int newAge);
    void setGpa(double newGpa);

    bool operator<(const Student& other) const;
    bool operator==(const Student& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Student& s);

    std::string serialize() const;
    static Student deserialize(const std::string& line);
};

#endif
```

### 4.2 구현 파일 (Student.cpp)

```cpp
#include "Student.h"
#include <sstream>
#include <iomanip>

Student::Student(int id, const std::string& name, int age, double gpa)
    : id(id), name(name), age(age), gpa(gpa) {
    if (id <= 0) throw std::invalid_argument("ID must be positive");
    if (name.empty()) throw std::invalid_argument("Name cannot be empty");
    if (age <= 0 || age > 120) throw std::invalid_argument("Age must be between 1 and 120");
    if (gpa < 0.0 || gpa > 4.0) throw std::invalid_argument("GPA must be between 0.0 and 4.0");
}

void Student::setName(const std::string& newName) {
    if (newName.empty()) throw std::invalid_argument("Name cannot be empty");
    name = newName;
}

void Student::setAge(int newAge) {
    if (newAge <= 0 || newAge > 120) throw std::invalid_argument("Age must be between 1 and 120");
    age = newAge;
}

void Student::setGpa(double newGpa) {
    if (newGpa < 0.0 || newGpa > 4.0) throw std::invalid_argument("GPA must be between 0.0 and 4.0");
    gpa = newGpa;
}

bool Student::operator<(const Student& other) const { return gpa > other.gpa; }
bool Student::operator==(const Student& other) const { return id == other.id; }

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
    int id, age; double gpa; std::string name;

    std::getline(iss, token, ','); id = std::stoi(token);
    std::getline(iss, name, ',');
    std::getline(iss, token, ','); age = std::stoi(token);
    std::getline(iss, token, ','); gpa = std::stod(token);

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
#include <fstream>

class StudentDatabase {
private:
    std::map<int, Student> students;
    std::string filename;

public:
    StudentDatabase(const std::string& filename = "students.csv") : filename(filename) {}

    void addStudent(const Student& student);
    void removeStudent(int id);
    void updateStudent(int id, const Student& updatedStudent);
    Student getStudent(int id) const;

    std::vector<Student> searchByName(const std::string& name) const;
    std::vector<Student> filterByGpa(double minGpa, double maxGpa) const;

    double averageGpa() const;
    std::vector<Student> topNStudents(int n) const;

    void displayAll() const;
    void saveToFile() const;
    void loadFromFile();

    size_t size() const { return students.size(); }
    bool empty() const { return students.empty(); }
};

#endif
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

    // GPA 내림차순 정렬 (operator< 사용)
    std::sort(all.begin(), all.end());

    // 상위 N명 반환
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

## 6. 파일 I/O와 직렬화

데이터는 CSV 형식으로 저장됩니다:

```
1,Alice Johnson,20,3.85
2,Bob Smith,22,3.20
3,Carol Lee,19,3.95
```

각 줄: `id,name,age,gpa`

`serialize()`와 `deserialize()` 메서드가 변환을 처리합니다.

---

## 7. 예외 처리

```cpp
try {
    Student s(101, "John Doe", 21, 3.75);
    db.addStudent(s);
} catch (const std::invalid_argument& e) {
    std::cerr << "Validation error: " << e.what() << "\n";
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << "\n";
}
```

---

## 8. 메뉴 인터페이스

```cpp
void displayMenu() {
    std::cout << "\n========== 학생 관리 시스템 ==========\n";
    std::cout << "1. 학생 추가\n";
    std::cout << "2. 학생 삭제\n";
    std::cout << "3. 학생 수정\n";
    std::cout << "4. 전체 학생 표시\n";
    std::cout << "5. 이름으로 검색\n";
    std::cout << "6. GPA 범위로 필터\n";
    std::cout << "7. 평균 GPA 표시\n";
    std::cout << "8. 상위 N명 학생 표시\n";
    std::cout << "9. 파일에 저장\n";
    std::cout << "10. 파일에서 로드\n";
    std::cout << "0. 종료\n";
    std::cout << "======================================\n";
    std::cout << "Enter choice: ";
}
```

---

## 9. 전체 구현

### 9.1 Main 프로그램 (main.cpp)

```cpp
#include "Student.h"
#include "StudentDatabase.h"
#include <iostream>
#include <limits>

// 입력 헬퍼 함수
int getIntInput(const std::string& prompt);
double getDoubleInput(const std::string& prompt);
std::string getStringInput(const std::string& prompt);
void displayMenu();

int main() {
    StudentDatabase db("students.csv");

    // 기존 데이터 로드
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
                case 1: { // 학생 추가
                    int id = getIntInput("Enter ID: ");
                    std::string name = getStringInput("Enter Name: ");
                    int age = getIntInput("Enter Age: ");
                    double gpa = getDoubleInput("Enter GPA: ");

                    Student s(id, name, age, gpa);
                    db.addStudent(s);
                    break;
                }
                case 2: { // 학생 삭제
                    int id = getIntInput("Enter ID to remove: ");
                    db.removeStudent(id);
                    break;
                }
                case 3: { // 학생 수정
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
                case 4: { // 전체 표시
                    db.displayAll();
                    break;
                }
                case 5: { // 이름으로 검색
                    std::string name = getStringInput("Enter name to search: ");
                    auto results = db.searchByName(name);
                    std::cout << "\nFound " << results.size() << " student(s):\n";
                    for (const auto& s : results) {
                        std::cout << s << "\n";
                    }
                    break;
                }
                case 6: { // GPA로 필터
                    double minGpa = getDoubleInput("Enter minimum GPA: ");
                    double maxGpa = getDoubleInput("Enter maximum GPA: ");
                    auto results = db.filterByGpa(minGpa, maxGpa);
                    std::cout << "\nFound " << results.size() << " student(s):\n";
                    for (const auto& s : results) {
                        std::cout << s << "\n";
                    }
                    break;
                }
                case 7: { // 평균 GPA
                    double avg = db.averageGpa();
                    std::cout << "\nAverage GPA: " << std::fixed << std::setprecision(2) << avg << "\n";
                    break;
                }
                case 8: { // 상위 N명
                    int n = getIntInput("Enter number of top students: ");
                    auto top = db.topNStudents(n);
                    std::cout << "\nTop " << top.size() << " student(s):\n";
                    for (const auto& s : top) {
                        std::cout << s << "\n";
                    }
                    break;
                }
                case 9: { // 저장
                    db.saveToFile();
                    break;
                }
                case 10: { // 로드
                    db.loadFromFile();
                    break;
                }
                case 0: { // 종료
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

    // 종료 시 자동 저장
    try {
        db.saveToFile();
    } catch (const std::exception& e) {
        std::cerr << "Failed to save data: " << e.what() << "\n";
    }

    return 0;
}

// 입력 헬퍼 구현
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
    std::cout << "\n========== 학생 관리 시스템 ==========\n";
    std::cout << "1. 학생 추가\n";
    std::cout << "2. 학생 삭제\n";
    std::cout << "3. 학생 수정\n";
    std::cout << "4. 전체 학생 표시\n";
    std::cout << "5. 이름으로 검색\n";
    std::cout << "6. GPA 범위로 필터\n";
    std::cout << "7. 평균 GPA 표시\n";
    std::cout << "8. 상위 N명 학생 표시\n";
    std::cout << "9. 파일에 저장\n";
    std::cout << "10. 파일에서 로드\n";
    std::cout << "0. 종료\n";
    std::cout << "======================================\n";
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

## 10. 테스트와 사용법

### 10.1 컴파일

```bash
make
```

### 10.2 사용 예시

```
$ ./student_mgmt

========== 학생 관리 시스템 ==========
1. 학생 추가
2. 학생 삭제
...
0. 종료
======================================
Enter choice: 1
Enter ID: 101
Enter Name: Alice Johnson
Enter Age: 20
Enter GPA: 3.85
Student added successfully.

Enter choice: 4

======================================================================
Total Students: 1
======================================================================
ID:   101 | Name: Alice Johnson       | Age:  20 | GPA: 3.85
======================================================================
```

### 10.3 파일 내용 (students.csv)

```
101,Alice Johnson,20,3.85
102,Bob Smith,22,3.20
```

---

## 연습문제

### 연습문제 1: 수강 과목 등록 기능 추가
`Student` 클래스에 수강 과목 목록을 지원하도록 확장하세요. `enrollCourse()`, `dropCourse()`, `getCourses()`를 추가하고 직렬화/역직렬화를 업데이트하세요.

### 연습문제 2: StudentDatabase 통계 확장
`gpaDistribution()` (학점별 비율), `medianGpa()` (중앙값), `belowAverage()` (평균 미만 학생) 메서드를 추가하세요.

### 연습문제 3: 커스텀 예외 계층
`StudentException`, `StudentNotFoundException`, `DuplicateStudentException`, `InvalidStudentDataException` 커스텀 예외 계층을 구현하세요.

### 연습문제 4: 정렬 및 필터링 개선
`sortedByName()`, `sortedByAge()`, `filterByAgeRange()` 메서드를 `std::sort`와 `std::copy_if`로 구현하세요.

### 연습문제 5: CSV 일괄 가져오기
유효성 검사 실패를 건너뛰고 기존 ID 중복을 피하면서 CSV 파일에서 학생을 일괄 가져오는 `importFromCsv()` 메서드를 추가하세요.

---

## 다음 단계

C++ 기초 과정을 완료하셨습니다! [C++ 고급](../CPP_Advanced/00_Overview.md)에서 템플릿, 모던 C++ 표준, 동시성, 디자인 패턴을 마스터하며 C++ 여정을 이어가세요.
