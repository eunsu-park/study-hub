#ifndef STUDENT_H
#define STUDENT_H

#include <string>
#include <iostream>

/**
 * @brief Represents a student with basic academic information
 *
 * This class demonstrates:
 * - Encapsulation with private members and public accessors
 * - Operator overloading for comparison and output
 * - String serialization for persistence
 */
class Student {
private:
    int id;
    std::string name;
    std::string major;
    double gpa;

public:
    // Constructor
    // Why: passing strings by const reference avoids copying potentially large strings
    // while preventing accidental modification of the caller's data
    Student(int id, const std::string& name, const std::string& major, double gpa);

    // Default constructor
    Student() : id(0), name(""), major(""), gpa(0.0) {}

    // Getters
    int getId() const { return id; }
    std::string getName() const { return name; }
    std::string getMajor() const { return major; }
    double getGpa() const { return gpa; }

    // Setters
    void setName(const std::string& newName) { name = newName; }
    void setMajor(const std::string& newMajor) { major = newMajor; }
    void setGpa(double newGpa) { gpa = newGpa; }

    // Why: operator< enables this type to work with std::sort, std::set, and std::map
    // without requiring a custom comparator — STL algorithms rely on this by default
    bool operator<(const Student& other) const { return id < other.id; }
    bool operator==(const Student& other) const { return id == other.id; }

    // Serialization
    std::string toCSV() const;
    static Student fromCSV(const std::string& csvLine);

    // Why: friend grants operator<< access to private members while keeping it a free
    // function — this preserves the natural stream syntax (cout << student) rather than member syntax
    friend std::ostream& operator<<(std::ostream& os, const Student& student);
};

#endif // STUDENT_H
