// behavioral_demo.cpp — Observer, Strategy, CRTP patterns
// Compile: g++ -std=c++20 -Wall -Wextra -o behavioral_demo behavioral_demo.cpp

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <memory>

// ============================================================
// Pattern 1: Observer
// ============================================================

class EventEmitter {
public:
    using Callback = std::function<void(const std::string&)>;

    void on(const std::string& event, Callback cb) {
        listeners_.push_back({event, std::move(cb)});
    }

    void emit(const std::string& event, const std::string& data) const {
        for (const auto& [ev, cb] : listeners_) {
            if (ev == event) cb(data);
        }
    }

private:
    struct Listener {
        std::string event;
        Callback callback;
    };
    std::vector<Listener> listeners_;
};

void observer_demo() {
    std::cout << "=== Observer Pattern ===\n";

    EventEmitter emitter;

    // Subscribe
    emitter.on("login", [](const std::string& user) {
        std::cout << "  [Logger] User logged in: " << user << '\n';
    });
    emitter.on("login", [](const std::string& user) {
        std::cout << "  [Analytics] Track login: " << user << '\n';
    });
    emitter.on("error", [](const std::string& msg) {
        std::cout << "  [Alert] Error occurred: " << msg << '\n';
    });

    // Emit events
    emitter.emit("login", "alice");
    emitter.emit("login", "bob");
    emitter.emit("error", "connection timeout");
}

// ============================================================
// Pattern 2: Strategy
// ============================================================

// Strategy interface
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) const = 0;
    virtual std::string name() const = 0;
};

class BubbleSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) const override {
        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = 0; j + 1 < data.size() - i; ++j)
                if (data[j] > data[j + 1])
                    std::swap(data[j], data[j + 1]);
    }
    std::string name() const override { return "BubbleSort"; }
};

class QuickSort : public SortStrategy {
    void qsort(std::vector<int>& d, int lo, int hi) const {
        if (lo >= hi) return;
        int pivot = d[hi], i = lo;
        for (int j = lo; j < hi; ++j)
            if (d[j] < pivot) std::swap(d[i++], d[j]);
        std::swap(d[i], d[hi]);
        qsort(d, lo, i - 1);
        qsort(d, i + 1, hi);
    }
public:
    void sort(std::vector<int>& data) const override {
        if (!data.empty())
            qsort(data, 0, static_cast<int>(data.size()) - 1);
    }
    std::string name() const override { return "QuickSort"; }
};

class Sorter {
    std::unique_ptr<SortStrategy> strategy_;
public:
    void set_strategy(std::unique_ptr<SortStrategy> s) {
        strategy_ = std::move(s);
    }
    void sort(std::vector<int>& data) const {
        if (strategy_) {
            std::cout << "  Sorting with " << strategy_->name() << '\n';
            strategy_->sort(data);
        }
    }
};

void strategy_demo() {
    std::cout << "\n=== Strategy Pattern ===\n";

    Sorter sorter;
    std::vector<int> data = {5, 2, 8, 1, 9, 3};

    auto print = [](const std::vector<int>& v) {
        for (int x : v) std::cout << x << ' ';
        std::cout << '\n';
    };

    // Use BubbleSort
    auto data1 = data;
    sorter.set_strategy(std::make_unique<BubbleSort>());
    sorter.sort(data1);
    std::cout << "  Result: ";
    print(data1);

    // Switch to QuickSort
    auto data2 = data;
    sorter.set_strategy(std::make_unique<QuickSort>());
    sorter.sort(data2);
    std::cout << "  Result: ";
    print(data2);
}

// ============================================================
// Pattern 3: CRTP (Curiously Recurring Template Pattern)
// ============================================================

// Static polymorphism via CRTP
template <typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->area_impl();
    }
    void describe() const {
        std::cout << "  Shape: area = " << area() << '\n';
    }
};

class Circle : public Shape<Circle> {
    double r_;
public:
    explicit Circle(double r) : r_(r) {}
    double area_impl() const { return 3.14159 * r_ * r_; }
};

class Rect : public Shape<Rect> {
    double w_, h_;
public:
    Rect(double w, double h) : w_(w), h_(h) {}
    double area_impl() const { return w_ * h_; }
};

// CRTP mixin: add comparison operators
template <typename Derived>
class Comparable {
public:
    bool operator>(const Derived& other) const {
        return static_cast<const Derived*>(this)->value() > other.value();
    }
    bool operator<(const Derived& other) const {
        return static_cast<const Derived*>(this)->value() < other.value();
    }
    bool operator==(const Derived& other) const {
        return static_cast<const Derived*>(this)->value() == other.value();
    }
};

class Temperature : public Comparable<Temperature> {
    double celsius_;
public:
    explicit Temperature(double c) : celsius_(c) {}
    double value() const { return celsius_; }
};

void crtp_demo() {
    std::cout << "\n=== CRTP Pattern ===\n";

    Circle c(5.0);
    Rect r(4.0, 6.0);
    c.describe();  // no virtual dispatch
    r.describe();

    std::cout << "\n  CRTP Mixin (Comparable):\n";
    Temperature t1(100.0), t2(37.0);
    std::cout << "  100 > 37? " << std::boolalpha << (t1 > t2) << '\n';
    std::cout << "  100 < 37? " << (t1 < t2) << '\n';
    std::cout << "  100 == 100? " << (t1 == t1) << '\n';
}

int main() {
    observer_demo();
    strategy_demo();
    crtp_demo();
    return 0;
}
