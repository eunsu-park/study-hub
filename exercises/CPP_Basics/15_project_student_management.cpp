// Exercise 15: Project — Student Management System
// Build a complete student management application using all C++ basics.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex15 15_project_student_management.cpp && ./ex15

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cassert>

// TODO 1: Implement the Student class
// - Private: name_ (string), id_ (int), grades_ (vector<double>)
// - Constructor, getters
// - add_grade(double), average() const, letter_grade() const
// - operator<< for formatted output

class Student {
    // TODO: Implement
};

// TODO 2: Implement the StudentDatabase class
// - Private: vector<Student>
// - add_student(Student)
// - remove_student(int id) -> bool
// - find_by_id(int id) -> Student* (nullptr if not found)
// - find_by_name(const string& partial) -> vector<Student*>
// - top_students(int n) -> vector of top n students by average
// - class_average() -> double
// - sort_by_average() (descending)
// - save_to_csv(const string& filename)
// - load_from_csv(const string& filename)

class StudentDatabase {
    // TODO: Implement
};

// TODO 3: Implement a report generator function
// that takes a StudentDatabase and returns a formatted string report:
//   === Student Report ===
//   Total students: X
//   Class average: XX.X
//   Highest: Name (XX.X)
//   Lowest: Name (XX.X)
//   Grade distribution: A:X B:X C:X D:X F:X

// std::string generate_report(const StudentDatabase& db) { ... }

int main() {
    std::cout << "=== Exercise 15: Student Management System ===\n\n";

    // Test 1: Student class
    // Student s1("Alice", 1001);
    // s1.add_grade(90); s1.add_grade(85); s1.add_grade(92);
    // assert(std::abs(s1.average() - 89.0) < 0.1);
    // assert(s1.letter_grade() == "B");
    // std::cout << s1 << '\n';
    // std::cout << "Test 1 passed: Student class\n";

    // Test 2: StudentDatabase
    // StudentDatabase db;
    // db.add_student(Student("Alice", 1001));
    // db.add_student(Student("Bob", 1002));
    // db.add_student(Student("Carol", 1003));
    // auto* found = db.find_by_id(1002);
    // assert(found != nullptr);
    // found->add_grade(95); found->add_grade(88);
    // assert(db.remove_student(9999) == false);
    // std::cout << "Test 2 passed: StudentDatabase\n";

    // Test 3: CSV save/load
    // db.save_to_csv("/tmp/students.csv");
    // StudentDatabase db2;
    // db2.load_from_csv("/tmp/students.csv");
    // std::cout << "Test 3 passed: CSV save/load\n";

    // Test 4: Report
    // std::string report = generate_report(db);
    // std::cout << report << '\n';
    // std::cout << "Test 4 passed: Report generation\n";

    std::cout << "Uncomment tests as you implement each part.\n";
    std::cout << "This is a capstone exercise — use all C++ basics!\n";
    return 0;
}
