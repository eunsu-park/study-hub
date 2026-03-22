// inheritance_demo.cpp — Virtual functions, abstract class, polymorphism
// Compile: g++ -std=c++20 -Wall -Wextra -o inheritance_demo inheritance_demo.cpp

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// --- Abstract base class ---
class Shape {
protected:
    std::string name_;

public:
    explicit Shape(const std::string& name) : name_(name) {}
    virtual ~Shape() = default;

    // Pure virtual functions
    virtual double area() const = 0;
    virtual double perimeter() const = 0;

    // Virtual function with default implementation
    virtual void describe() const {
        std::cout << name_ << ": area=" << area()
                  << " perimeter=" << perimeter() << '\n';
    }

    const std::string& name() const { return name_; }
};

class Circle : public Shape {
    double radius_;

public:
    explicit Circle(double r) : Shape("Circle"), radius_(r) {}

    double area() const override { return M_PI * radius_ * radius_; }
    double perimeter() const override { return 2.0 * M_PI * radius_; }
};

class Rectangle : public Shape {
    double w_, h_;

public:
    Rectangle(double w, double h) : Shape("Rectangle"), w_(w), h_(h) {}

    double area() const override { return w_ * h_; }
    double perimeter() const override { return 2.0 * (w_ + h_); }
};

class Triangle : public Shape {
    double a_, b_, c_;

public:
    Triangle(double a, double b, double c)
        : Shape("Triangle"), a_(a), b_(b), c_(c) {}

    double area() const override {
        double s = (a_ + b_ + c_) / 2.0;
        return std::sqrt(s * (s - a_) * (s - b_) * (s - c_));
    }

    double perimeter() const override { return a_ + b_ + c_; }
};

// --- Multiple inheritance example ---
class Printable {
public:
    virtual ~Printable() = default;
    virtual std::string to_string() const = 0;
};

class Serializable {
public:
    virtual ~Serializable() = default;
    virtual std::string serialize() const = 0;
};

class Document : public Printable, public Serializable {
    std::string title_;
    std::string content_;

public:
    Document(const std::string& title, const std::string& content)
        : title_(title), content_(content) {}

    std::string to_string() const override {
        return "[" + title_ + "] " + content_;
    }

    std::string serialize() const override {
        return "{\"title\":\"" + title_ + "\",\"content\":\"" + content_ + "\"}";
    }
};

int main() {
    std::cout << "=== Polymorphism via Base Pointer ===\n";
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 6.0));
    shapes.push_back(std::make_unique<Triangle>(3.0, 4.0, 5.0));

    for (const auto& s : shapes) {
        s->describe();  // dynamic dispatch
    }

    std::cout << "\n=== dynamic_cast ===\n";
    Shape* raw = shapes[0].get();
    if (auto* circ = dynamic_cast<Circle*>(raw)) {
        std::cout << "Successfully cast to Circle, area=" << circ->area() << '\n';
    }
    if (auto* rect = dynamic_cast<Rectangle*>(raw)) {
        std::cout << "Cast to Rectangle\n";  // won't execute
    } else {
        std::cout << "Not a Rectangle\n";
    }

    std::cout << "\n=== Multiple Inheritance ===\n";
    Document doc("Report", "Q4 financial results");
    std::cout << "Print: " << doc.to_string() << '\n';
    std::cout << "JSON:  " << doc.serialize() << '\n';

    // Using interface pointers
    Printable* p = &doc;
    std::cout << "Via Printable*: " << p->to_string() << '\n';

    return 0;
}
