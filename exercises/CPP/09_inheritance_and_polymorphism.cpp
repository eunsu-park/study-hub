/*
 * Exercises for Lesson 09: Inheritance and Polymorphism
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex09 09_inheritance_and_polymorphism.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
using namespace std;

// === Exercise 1: Constructor/Destructor Order ===
// Problem: Create a three-level hierarchy Vehicle -> Car -> ElectricCar.
//          Observe construction order (base first) and destruction order
//          (derived first). Using a virtual destructor ensures proper
//          cleanup through base pointers.

class Vehicle {
public:
    Vehicle() { cout << "  Vehicle constructor" << endl; }
    virtual ~Vehicle() { cout << "  Vehicle destructor" << endl; }
    virtual void describe() const { cout << "  I am a Vehicle" << endl; }
};

class Car : public Vehicle {
public:
    Car() { cout << "  Car constructor" << endl; }
    ~Car() override { cout << "  Car destructor" << endl; }
    void describe() const override { cout << "  I am a Car" << endl; }
};

class ElectricCar : public Car {
public:
    ElectricCar() { cout << "  ElectricCar constructor" << endl; }
    ~ElectricCar() override { cout << "  ElectricCar destructor" << endl; }
    void describe() const override { cout << "  I am an ElectricCar" << endl; }
};

void exercise_1() {
    cout << "=== Exercise 1: Constructor/Destructor Order ===" << endl;

    // Stack allocation: full construction and destruction chain
    cout << "\n--- Stack-allocated ElectricCar ---" << endl;
    {
        ElectricCar ec;
        ec.describe();
        cout << "  (leaving scope...)" << endl;
    }
    // Construction: Vehicle -> Car -> ElectricCar
    // Destruction:  ElectricCar -> Car -> Vehicle

    // Heap allocation via base pointer (virtual destructor is critical)
    cout << "\n--- Heap-allocated via Vehicle* (virtual dtor) ---" << endl;
    {
        Vehicle* v = new ElectricCar();
        v->describe();  // Polymorphic: prints "I am an ElectricCar"
        delete v;       // Virtual destructor ensures all dtors run
    }

    // Explanation: Without virtual destructor on Vehicle, deleting through
    // Vehicle* would only call Vehicle's destructor, leaking resources
    // from Car and ElectricCar. The virtual keyword ensures the most-derived
    // destructor is called first, then the chain proceeds up to the base.
    cout << "\nNote: If Vehicle's destructor were NOT virtual, deleting" << endl;
    cout << "through Vehicle* would skip ElectricCar and Car destructors." << endl;
}

// === Exercise 2: Polymorphic Shape Area Calculator ===
// Problem: Abstract base class Shape with pure virtual area(). Derive Circle,
//          Rectangle, Triangle. Store in vector<Shape*> and compute areas.

class Shape {
public:
    virtual double area() const = 0;
    virtual string name() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
    double radius_;
public:
    explicit Circle(double r) : radius_(r) {}
    double area() const override { return M_PI * radius_ * radius_; }
    string name() const override { return "Circle(r=" + to_string(radius_) + ")"; }
};

class Rectangle : public Shape {
    double width_, height_;
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    double area() const override { return width_ * height_; }
    string name() const override {
        return "Rectangle(" + to_string(width_) + "x" + to_string(height_) + ")";
    }
};

class Triangle : public Shape {
    double base_, height_;
public:
    Triangle(double b, double h) : base_(b), height_(h) {}
    double area() const override { return 0.5 * base_ * height_; }
    string name() const override {
        return "Triangle(b=" + to_string(base_) + ",h=" + to_string(height_) + ")";
    }
};

void exercise_2() {
    cout << "\n=== Exercise 2: Polymorphic Shape Area Calculator ===" << endl;

    // Using raw pointers as specified in the exercise, with manual cleanup
    vector<Shape*> shapes;
    shapes.push_back(new Circle(5.0));
    shapes.push_back(new Rectangle(4.0, 6.0));
    shapes.push_back(new Triangle(3.0, 8.0));

    double totalArea = 0.0;
    for (const Shape* s : shapes) {
        // Polymorphic dispatch: correct area() is called for each type
        double a = s->area();
        cout << "  " << s->name() << " -> area = " << a << endl;
        totalArea += a;
    }
    cout << "  Total area: " << totalArea << endl;

    // Cleanup
    for (Shape* s : shapes) {
        delete s;
    }
}

// === Exercise 3: override and final Safety ===
// Problem: Derive FastLogger from Logger using override. Mark it final
//          so no further overrides are possible.

class Logger {
public:
    virtual void log(const string& message) {
        cout << "  [LOG] " << message << endl;
    }
    virtual ~Logger() = default;
};

class FastLogger final : public Logger {
    // 'override' catches signature mismatches at compile time.
    // For example, writing void log(string message) (missing const&)
    // with 'override' would produce a compiler error because it doesn't
    // match the base signature.
    //
    // 'final' on the class prevents any further inheritance.
    // If we only wanted to prevent overriding log(), we could put
    // 'final' on the method instead.
public:
    void log(const string& message) override {
        // Simulating a "fast" logger that skips the [LOG] prefix
        cout << "  [FAST] " << message << endl;
    }
};

// This would fail to compile because FastLogger is marked final:
// class UltraLogger : public FastLogger { };

void exercise_3() {
    cout << "\n=== Exercise 3: override and final Safety ===" << endl;

    Logger base;
    FastLogger fast;

    base.log("Regular logging");
    fast.log("Fast logging");

    // Polymorphic usage
    Logger* ptr = &fast;
    ptr->log("Via base pointer - dispatches to FastLogger");

    cout << "\n  Why compile-time errors are better than runtime:" << endl;
    cout << "  - A signature mismatch without 'override' silently creates" << endl;
    cout << "    a new virtual function instead of overriding the base one." << endl;
    cout << "  - The bug only manifests when the base pointer calls the wrong" << endl;
    cout << "    version, which can be hard to debug." << endl;
    cout << "  - 'override' and 'final' make the intent explicit and let the" << endl;
    cout << "    compiler enforce it." << endl;
}

// === Exercise 4: Interface Composition ===
// Problem: Two interfaces (Drawable, Resizable). Canvas holds Drawable*.
//          Square implements both. Demonstrate composition of interfaces.

class Drawable {
public:
    virtual void draw() const = 0;
    virtual ~Drawable() = default;
};

class Resizable {
public:
    virtual void resize(double factor) = 0;
    virtual ~Resizable() = default;
};

class Square : public Drawable, public Resizable {
    double side_;
    string label_;
public:
    Square(double side, const string& label)
        : side_(side), label_(label) {}

    void draw() const override {
        cout << "  Drawing " << label_ << " (side=" << side_ << ")" << endl;
    }

    void resize(double factor) override {
        side_ *= factor;
        cout << "  Resized " << label_ << " by " << factor
             << "x -> side=" << side_ << endl;
    }

    double side() const { return side_; }
};

class Canvas {
    vector<Drawable*> elements_;
public:
    void add(Drawable* d) { elements_.push_back(d); }

    void drawAll() const {
        cout << "  --- Canvas drawing all elements ---" << endl;
        for (const auto* elem : elements_) {
            elem->draw();
        }
    }
};

void exercise_4() {
    cout << "\n=== Exercise 4: Interface Composition ===" << endl;

    Square s1(5.0, "Square-A");
    Square s2(3.0, "Square-B");
    Square s3(7.0, "Square-C");

    Canvas canvas;
    canvas.add(&s1);
    canvas.add(&s2);
    canvas.add(&s3);

    canvas.drawAll();

    // Resize some squares through the Resizable interface
    s1.resize(2.0);
    s3.resize(0.5);

    canvas.drawAll();
}

// === Exercise 5: Resolving the Diamond Problem ===
// Problem: Person -> Employee, Person -> Student, both -> WorkingStudent.
//          Use virtual inheritance to resolve ambiguity.

class Person {
protected:
    string name_;
public:
    explicit Person(const string& name = "Unknown") : name_(name) {
        cout << "  Person(" << name_ << ") constructed" << endl;
    }
    virtual ~Person() = default;

    void greet() const {
        cout << "  Hello, I'm " << name_ << endl;
    }
};

// Virtual inheritance ensures only one Person sub-object exists
class Employee : virtual public Person {
protected:
    string company_;
public:
    Employee(const string& name = "Unknown", const string& company = "N/A")
        : Person(name), company_(company) {
        cout << "  Employee at " << company_ << " constructed" << endl;
    }
};

class Student : virtual public Person {
protected:
    string school_;
public:
    Student(const string& name = "Unknown", const string& school = "N/A")
        : Person(name), school_(school) {
        cout << "  Student at " << school_ << " constructed" << endl;
    }
};

// WorkingStudent inherits from both Employee and Student.
// With virtual inheritance, there is exactly one Person sub-object,
// so greet() is unambiguous.
class WorkingStudent : public Employee, public Student {
public:
    WorkingStudent(const string& name, const string& company, const string& school)
        : Person(name), Employee(name, company), Student(name, school) {
        // Note: With virtual inheritance, the most-derived class
        // must initialize the virtual base (Person) directly.
        cout << "  WorkingStudent constructed" << endl;
    }

    void info() const {
        cout << "  " << name_ << " works at " << company_
             << " and studies at " << school_ << endl;
    }
};

void exercise_5() {
    cout << "\n=== Exercise 5: Diamond Problem Resolution ===" << endl;

    cout << "\nConstructing WorkingStudent:" << endl;
    WorkingStudent ws("Alice", "TechCorp", "MIT");

    cout << "\nCalling greet() (unambiguous with virtual inheritance):" << endl;
    ws.greet();  // No ambiguity: single Person sub-object
    ws.info();

    cout << "\nWithout virtual inheritance, ws.greet() would be ambiguous" << endl;
    cout << "because the compiler wouldn't know whether to call" << endl;
    cout << "Employee::Person::greet() or Student::Person::greet()." << endl;
    cout << "Virtual inheritance ensures only one Person exists." << endl;
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
