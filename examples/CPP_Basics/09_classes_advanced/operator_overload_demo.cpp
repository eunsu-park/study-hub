// operator_overload_demo.cpp — Operator overloading, copy constructor, rule of three
// Compile: g++ -std=c++20 -Wall -Wextra -o operator_overload_demo operator_overload_demo.cpp

#include <iostream>
#include <cstring>

class Vector2D {
private:
    double x_, y_;

public:
    Vector2D(double x = 0.0, double y = 0.0) : x_(x), y_(y) {}

    // Getters
    double x() const { return x_; }
    double y() const { return y_; }

    // Arithmetic operators
    Vector2D operator+(const Vector2D& rhs) const {
        return {x_ + rhs.x_, y_ + rhs.y_};
    }

    Vector2D operator-(const Vector2D& rhs) const {
        return {x_ - rhs.x_, y_ - rhs.y_};
    }

    Vector2D operator*(double scalar) const {
        return {x_ * scalar, y_ * scalar};
    }

    // Compound assignment
    Vector2D& operator+=(const Vector2D& rhs) {
        x_ += rhs.x_;
        y_ += rhs.y_;
        return *this;
    }

    // Comparison
    bool operator==(const Vector2D& rhs) const {
        return x_ == rhs.x_ && y_ == rhs.y_;
    }

    // Stream insertion (friend)
    friend std::ostream& operator<<(std::ostream& os, const Vector2D& v) {
        return os << "(" << v.x_ << ", " << v.y_ << ")";
    }
};

// Non-member operator: scalar * vector
Vector2D operator*(double scalar, const Vector2D& v) {
    return v * scalar;
}

// --- Rule of Three demo: manual resource management ---
class DynArray {
private:
    int* data_;
    size_t size_;

public:
    explicit DynArray(size_t n) : data_(new int[n]{}), size_(n) {
        std::cout << "  [Ctor] DynArray size=" << n << '\n';
    }

    // Copy constructor (deep copy)
    DynArray(const DynArray& other) : data_(new int[other.size_]), size_(other.size_) {
        std::memcpy(data_, other.data_, size_ * sizeof(int));
        std::cout << "  [Copy ctor] DynArray size=" << size_ << '\n';
    }

    // Copy assignment operator
    DynArray& operator=(const DynArray& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::memcpy(data_, other.data_, size_ * sizeof(int));
            std::cout << "  [Copy assign] DynArray size=" << size_ << '\n';
        }
        return *this;
    }

    // Destructor
    ~DynArray() {
        delete[] data_;
        std::cout << "  [Dtor] DynArray\n";
    }

    // Subscript operator
    int& operator[](size_t idx) { return data_[idx]; }
    const int& operator[](size_t idx) const { return data_[idx]; }

    size_t size() const { return size_; }

    friend std::ostream& operator<<(std::ostream& os, const DynArray& a) {
        os << "[";
        for (size_t i = 0; i < a.size_; ++i) {
            if (i > 0) os << ", ";
            os << a.data_[i];
        }
        return os << "]";
    }
};

int main() {
    std::cout << "=== Vector2D Operator Overloading ===\n";
    Vector2D a(3.0, 4.0), b(1.0, 2.0);
    std::cout << "a = " << a << '\n';
    std::cout << "b = " << b << '\n';
    std::cout << "a + b = " << (a + b) << '\n';
    std::cout << "a - b = " << (a - b) << '\n';
    std::cout << "a * 2 = " << (a * 2.0) << '\n';
    std::cout << "3 * b = " << (3.0 * b) << '\n';

    Vector2D c = a;
    c += b;
    std::cout << "c = a; c += b => " << c << '\n';
    std::cout << "a == b? " << std::boolalpha << (a == b) << '\n';

    std::cout << "\n=== DynArray (Rule of Three) ===\n";
    DynArray arr(5);
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = static_cast<int>(i * 10);
    }
    std::cout << "arr = " << arr << '\n';

    DynArray copy = arr;          // copy constructor
    copy[0] = 999;
    std::cout << "copy = " << copy << '\n';
    std::cout << "arr  = " << arr << " (unchanged — deep copy)\n";

    DynArray assigned(3);
    assigned = arr;               // copy assignment
    std::cout << "assigned = " << assigned << '\n';

    std::cout << "\n--- End of main ---\n";
    return 0;
}
