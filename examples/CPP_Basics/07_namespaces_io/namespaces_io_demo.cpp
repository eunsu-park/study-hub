// namespaces_io_demo.cpp — Namespaces, cin/cout, iomanip, stringstream
// Compile: g++ -std=c++20 -Wall -Wextra -o namespaces_io_demo namespaces_io_demo.cpp

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>

// --- Custom namespaces ---
namespace math {
    constexpr double PI = 3.14159265358979;
    double circle_area(double r) { return PI * r * r; }
}

namespace physics {
    constexpr double G = 9.81;  // m/s^2
    double fall_time(double height) {
        return std::sqrt(2.0 * height / G);
    }
}

// Nested namespace (C++17)
namespace company::team::project {
    std::string name() { return "Widget v2"; }
}

int main() {
    // --- Namespace usage ---
    std::cout << "=== Namespaces ===\n";
    std::cout << "Circle area (r=5): " << math::circle_area(5.0) << '\n';
    std::cout << "Fall time (h=10m): " << physics::fall_time(10.0) << " s\n";
    std::cout << "Project: " << company::team::project::name() << '\n';

    // using declaration
    using math::PI;
    std::cout << "PI = " << PI << '\n';

    // --- cout formatting with iomanip ---
    std::cout << "\n=== iomanip Formatting ===\n";

    // Precision
    double val = 3.141592653589793;
    std::cout << "default:    " << val << '\n';
    std::cout << "fixed(4):   " << std::fixed << std::setprecision(4) << val << '\n';
    std::cout << "scientific: " << std::scientific << val << '\n';
    std::cout << std::defaultfloat;  // reset

    // Width and alignment
    std::cout << "\n--- Width & Fill ---\n";
    std::cout << std::setw(15) << std::left  << "Name" << "| Score\n";
    std::cout << std::setfill('-') << std::setw(22) << "" << '\n';
    std::cout << std::setfill(' ');
    std::cout << std::setw(15) << std::left << "Alice" << "| " << 95 << '\n';
    std::cout << std::setw(15) << std::left << "Bob" << "| " << 87 << '\n';

    // Number bases
    std::cout << "\n--- Number Bases ---\n";
    int num = 255;
    std::cout << "dec: " << std::dec << num << '\n';
    std::cout << "hex: " << std::hex << std::showbase << num << '\n';
    std::cout << "oct: " << std::oct << num << '\n';
    std::cout << std::dec << std::noshowbase;  // reset

    // Boolalpha
    std::cout << "bool: " << std::boolalpha << true << ", " << false << '\n';

    // --- stringstream ---
    std::cout << "\n=== stringstream ===\n";

    // Building a string
    std::ostringstream oss;
    oss << "Score: " << 95 << " / " << 100;
    std::string result = oss.str();
    std::cout << result << '\n';

    // Parsing a string
    std::string data = "42 3.14 hello";
    std::istringstream iss(data);
    int i;
    double d;
    std::string s;
    iss >> i >> d >> s;
    std::cout << "Parsed: int=" << i << " double=" << d
              << " string=" << s << '\n';

    // CSV parsing
    std::string csv_line = "Alice,95,A";
    std::istringstream csv(csv_line);
    std::string name, grade;
    int score;
    std::getline(csv, name, ',');
    csv >> score;
    csv.ignore(1);  // skip comma
    std::getline(csv, grade);
    std::cout << "CSV: name=" << name << " score=" << score
              << " grade=" << grade << '\n';

    return 0;
}
