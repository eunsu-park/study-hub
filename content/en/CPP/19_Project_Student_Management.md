# 19. Project: Student Management System

**Previous**: [C++ Design Patterns](./18_Design_Patterns.md) | **Next**: [CMake and Build Systems](./20_CMake_and_Build_Systems.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a multi-class C++ application with clear separation of concerns (data, storage, UI)
2. Apply STL containers (`map`, `vector`, `set`) for efficient lookup, sorting, and uniqueness guarantees
3. Implement CRUD operations with input validation and meaningful error messages
4. Serialize and deserialize structured data to and from CSV files using `stringstream`
5. Build a menu-driven CLI that handles invalid input gracefully with exception handling
6. Use operator overloading (`<<`, `<`, `==`) to integrate custom types with STL algorithms and I/O
7. Practice const correctness, RAII, and modern C++ best practices in a complete project

---

Up to this point, each lesson has taught individual C++ features in isolation. This project ties them all together into a cohesive, working application. Building a student management system from scratch forces you to make design decisions--how to structure classes, which containers to choose, how to handle errors--that mirror the challenges of real software development. It is in projects like this that isolated knowledge becomes genuine skill.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Requirements](#2-requirements)
3. [Class Design](#3-class-design)
4. [Student Class](#4-student-class)
5. [Database Class](#5-database-class)
6. [File I/O and Serialization](#6-file-io-and-serialization)
7. [Exception Handling](#7-exception-handling)
8. [Menu Interface](#8-menu-interface)
9. [Complete Implementation](#9-complete-implementation)
10. [Testing and Usage](#10-testing-and-usage)

---

## 1. Project Overview

We'll build a **Student Management System** that allows users to:
- Add, remove, and update student records
- Search and filter students by name, ID, or GPA
- Calculate statistics (average GPA, top students)
- Save/load data to/from files
- Handle errors gracefully

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

## 2. Requirements

### Functional Requirements
1. **CRUD Operations:** Create, Read, Update, Delete student records
2. **Search:** Find students by ID, name, or GPA range
3. **Statistics:** Compute average GPA, find top N students
4. **Persistence:** Save/load data from a text file
5. **Validation:** Ensure valid IDs, GPAs (0.0–4.0), ages (> 0)

### Technical Requirements
- Use **STL containers** (map, vector, set)
- Use **smart pointers** (shared_ptr or unique_ptr where applicable)
- Implement **exception handling** (custom exceptions)
- Follow **RAII** principles
- Use **const correctness**
- Implement **operator overloading** (for Student comparison)

---

## 3. Class Design

### 3.1 Student Class
Represents a single student with attributes:
- `id` (int, unique)
- `name` (string)
- `age` (int)
- `gpa` (double, 0.0–4.0)

**Methods:**
- Constructor with validation
- Getters/setters
- Operator overloading (<, ==) for sorting and searching
- Friend function for I/O

### 3.2 StudentDatabase Class
Manages the collection of students:
- `std::map<int, Student>` for O(log N) lookup by ID
- Methods: add, remove, update, search, save, load
- Exception handling for invalid operations

---

## 4. Student Class

### 4.1 Header File (Student.h)

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

### 4.2 Implementation File (Student.cpp)

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

## 5. Database Class

### 5.1 Header File (StudentDatabase.h)

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

### 5.2 Implementation File (StudentDatabase.cpp)

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

## 6. File I/O and Serialization

Data is saved in CSV format:

```
1,Alice Johnson,20,3.85
2,Bob Smith,22,3.20
3,Carol Lee,19,3.95
```

Each line: `id,name,age,gpa`

The `serialize()` and `deserialize()` methods handle conversion.

---

## 7. Exception Handling

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

Custom exceptions could be defined for more granular error handling:

```cpp
class DatabaseException : public std::runtime_error {
public:
    explicit DatabaseException(const std::string& msg) : std::runtime_error(msg) {}
};
```

---

## 8. Menu Interface

### 8.1 Menu Display

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

### 8.2 Input Helper Functions

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

## 9. Complete Implementation

### 9.1 Main Program (main.cpp)

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

## 10. Testing and Usage

### 10.1 Compilation

```bash
make
```

### 10.2 Sample Usage

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

### 10.3 File Contents (students.csv)

```
101,Alice Johnson,20,3.85
102,Bob Smith,22,3.20
```

---

## Exercises

### Exercise 1: Add a Course Enrollment Feature

Extend the `Student` class to support a list of enrolled courses. Add:
- A `std::vector<std::string> courses` member.
- `enrollCourse(const std::string& course)` — adds the course only if not already enrolled (use `std::find`).
- `dropCourse(const std::string& course)` — removes the course; throws `std::runtime_error` if not enrolled.
- `getCourses() const` — returns the vector.
- Update `serialize()` / `deserialize()` to persist courses (courses can be joined with `|` as a separator in the CSV field).

Test by enrolling two courses, dropping one, and verifying serialization round-trips correctly.

### Exercise 2: Extend StudentDatabase with Statistics

Add the following methods to `StudentDatabase`:

1. `std::map<std::string, double> gpaDistribution() const` — returns a map from letter grade (`"A"`, `"B"`, `"C"`, `"D"`, `"F"`) to the percentage of students in that range.
2. `double medianGpa() const` — returns the median GPA across all students (sort a copy of the GPAs and pick the middle value; handle even-sized collections).
3. `std::vector<Student> belowAverage() const` — returns all students whose GPA is strictly below the class average.

Add menu options 11, 12, and 13 to `main.cpp` to invoke these methods and display the results.

### Exercise 3: Custom Exception Hierarchy

Replace the use of `std::runtime_error` and `std::invalid_argument` in the project with a custom exception hierarchy:

```cpp
class StudentException : public std::exception { ... };
class StudentNotFoundException : public StudentException { ... };
class DuplicateStudentException : public StudentException { ... };
class InvalidStudentDataException : public StudentException { ... };
```

Each exception should carry a descriptive message and the relevant student ID where applicable. Update all throw sites and catch blocks in both `Student.cpp` and `StudentDatabase.cpp`. Verify that catching `StudentException` handles all three derived types.

### Exercise 4: Sorting and Filtering Enhancements

Add three new query methods to `StudentDatabase`:

1. `std::vector<Student> sortedByName() const` — returns all students sorted alphabetically by name (case-insensitive).
2. `std::vector<Student> sortedByAge() const` — returns all students sorted by age ascending.
3. `std::vector<Student> filterByAgeRange(int minAge, int maxAge) const` — returns students whose age is within the range (inclusive).

Implement each method using `std::sort` or `std::copy_if` with a lambda comparator/predicate. Add corresponding menu entries.

### Exercise 5: Batch Import from CSV

Add a method `int importFromCsv(const std::string& filename)` to `StudentDatabase` that:
1. Opens the specified file and reads every line.
2. Attempts to deserialize each line into a `Student`.
3. Skips lines that fail validation (catching exceptions and logging a warning).
4. Skips students whose ID already exists in the database (no overwrite).
5. Returns the count of successfully imported students.

Write a test CSV file with five valid records and two invalid ones (e.g., a negative GPA and an empty name), run the import, and confirm the correct count and the existing records are untouched.

---

## Navigation

**Previous**: [C++ Design Patterns](./18_Design_Patterns.md) | **Next**: [CMake and Build Systems](./20_CMake_and_Build_Systems.md)
