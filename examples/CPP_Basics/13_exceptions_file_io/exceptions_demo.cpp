// exceptions_demo.cpp — try/catch, custom exceptions, file read/write
// Compile: g++ -std=c++20 -Wall -Wextra -o exceptions_demo exceptions_demo.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

// --- Custom exception ---
class ValidationError : public std::runtime_error {
    int code_;
public:
    ValidationError(const std::string& msg, int code)
        : std::runtime_error(msg), code_(code) {}
    int code() const { return code_; }
};

// Function that may throw
double safe_divide(double a, double b) {
    if (b == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}

int parse_age(const std::string& input) {
    int age = std::stoi(input);  // may throw std::invalid_argument
    if (age < 0 || age > 150) {
        throw ValidationError("Age out of range: " + input, 1001);
    }
    return age;
}

int main() {
    // --- Basic try/catch ---
    std::cout << "=== Basic Exception Handling ===\n";
    try {
        std::cout << "10 / 3 = " << safe_divide(10, 3) << '\n';
        std::cout << "10 / 0 = " << safe_divide(10, 0) << '\n';
    } catch (const std::invalid_argument& e) {
        std::cerr << "Caught: " << e.what() << '\n';
    }

    // --- Custom exception ---
    std::cout << "\n=== Custom Exception ===\n";
    std::vector<std::string> test_ages = {"25", "abc", "-5", "200"};
    for (const auto& input : test_ages) {
        try {
            int age = parse_age(input);
            std::cout << "\"" << input << "\" -> age=" << age << '\n';
        } catch (const ValidationError& e) {
            std::cerr << "Validation error (code " << e.code()
                      << "): " << e.what() << '\n';
        } catch (const std::invalid_argument& e) {
            std::cerr << "Parse error for \"" << input << "\": " << e.what() << '\n';
        }
    }

    // --- Multiple catch + catch-all ---
    std::cout << "\n=== Catch hierarchy ===\n";
    try {
        throw std::out_of_range("index 99 out of bounds");
    } catch (const std::out_of_range& e) {
        std::cerr << "out_of_range: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "Unknown exception\n";
    }

    // --- File writing ---
    std::cout << "\n=== File I/O ===\n";
    const std::string filename = "/tmp/cpp_demo_output.txt";

    {
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Cannot open file for writing\n";
            return 1;
        }
        ofs << "Line 1: Hello from C++\n";
        ofs << "Line 2: Exception handling demo\n";
        ofs << "Line 3: File I/O works!\n";
        std::cout << "Wrote to " << filename << '\n';
    }  // file closed automatically

    // --- File reading ---
    {
        std::ifstream ifs(filename);
        if (!ifs) {
            std::cerr << "Cannot open file for reading\n";
            return 1;
        }
        std::string line;
        int line_num = 0;
        while (std::getline(ifs, line)) {
            std::cout << "  [" << ++line_num << "] " << line << '\n';
        }
    }

    // --- Exception with file I/O ---
    std::cout << "\n=== File Exception ===\n";
    try {
        std::ifstream ifs("/nonexistent/path/file.txt");
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open /nonexistent/path/file.txt");
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "File error: " << e.what() << '\n';
    }

    std::cout << "\nAll demos completed.\n";
    return 0;
}
