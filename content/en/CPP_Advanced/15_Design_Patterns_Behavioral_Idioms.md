# Design Patterns: Behavioral and C++ Idioms

**Previous**: [Design Patterns: Creational and Structural](./14_Design_Patterns_Creational_Structural.md) | **Next**: [C++23 Features](./16_CPP23_Features.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the Observer pattern using both interface-based and modern signal/slot approaches
2. Apply the Strategy pattern to encapsulate interchangeable algorithms, including lambda-based variants
3. Build undo/redo systems using the Command pattern with macro command composition
4. Use the State and Template Method patterns to manage complex control flow and algorithm skeletons
5. Implement CRTP for static polymorphism, counter mixins, and fluent interfaces
6. Apply the PIMPL idiom to create compilation firewalls that reduce build dependencies
7. Use type erasure to store heterogeneous objects in a single container without a common base class

---

Behavioral patterns define how objects communicate and distribute responsibility. Unlike creational and structural patterns which focus on *what* objects are and *how* they're composed, behavioral patterns focus on *what objects do* and *how they interact*. C++ adds its own idioms -- CRTP, PIMPL, type erasure, and NVI -- that exploit templates, value semantics, and compilation model features unavailable in other languages. Mastering both the GoF behavioral patterns and the C++ idioms gives you a complete toolkit for designing maintainable architectures.

---

## Table of Contents

1. [Observer](#1-observer)
2. [Strategy](#2-strategy)
3. [Command](#3-command)
4. [State](#4-state)
5. [Template Method](#5-template-method)
6. [Iterator](#6-iterator)
7. [CRTP (Curiously Recurring Template Pattern)](#7-crtp)
8. [PIMPL (Pointer to Implementation)](#8-pimpl)
9. [Type Erasure](#9-type-erasure)
10. [NVI (Non-Virtual Interface)](#10-nvi)

---

## 1. Observer

Notifies multiple objects when a subject's state changes.

### Modern Signal/Slot Approach

```cpp
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

template<typename... Args>
class Signal {
public:
    using Slot = std::function<void(Args...)>;
    using SlotId = size_t;

    SlotId connect(Slot slot) {
        slots_.push_back({nextId_, std::move(slot)});
        return nextId_++;
    }

    void disconnect(SlotId id) {
        slots_.erase(
            std::remove_if(slots_.begin(), slots_.end(),
                [id](const auto& p) { return p.first == id; }),
            slots_.end());
    }

    void emit(Args... args) {
        for (auto& [id, slot] : slots_) {
            slot(args...);
        }
    }

private:
    std::vector<std::pair<SlotId, Slot>> slots_;
    SlotId nextId_ = 0;
};

// Usage: Stock price monitoring
class Stock {
public:
    Stock(std::string symbol, double price)
        : symbol_(std::move(symbol)), price_(price) {}

    void setPrice(double price) {
        double old = price_;
        price_ = price;
        priceChanged.emit(symbol_, old, price_);
    }

    Signal<std::string, double, double> priceChanged;

private:
    std::string symbol_;
    double price_;
};

int main() {
    Stock apple("AAPL", 150.0);

    apple.priceChanged.connect(
        [](const std::string& sym, double old_p, double new_p) {
            std::cout << sym << ": $" << old_p << " -> $" << new_p << "\n";
        });

    apple.setPrice(155.0);  // AAPL: $150 -> $155
    apple.setPrice(148.0);  // AAPL: $155 -> $148
    return 0;
}
```

### Traditional Interface-Based Observer

```cpp
#include <memory>
#include <vector>
#include <string>
#include <iostream>

class IObserver {
public:
    virtual ~IObserver() = default;
    virtual void update(const std::string& message) = 0;
};

class ISubject {
public:
    virtual ~ISubject() = default;
    virtual void attach(std::shared_ptr<IObserver> obs) = 0;
    virtual void detach(std::shared_ptr<IObserver> obs) = 0;
    virtual void notify() = 0;
};

class NewsAgency : public ISubject {
    std::vector<std::weak_ptr<IObserver>> observers_;
    std::string news_;

public:
    void attach(std::shared_ptr<IObserver> obs) override {
        observers_.push_back(obs);
    }

    void detach(std::shared_ptr<IObserver> obs) override {
        observers_.erase(
            std::remove_if(observers_.begin(), observers_.end(),
                [&](const std::weak_ptr<IObserver>& wp) {
                    auto sp = wp.lock();
                    return !sp || sp == obs;
                }),
            observers_.end());
    }

    void notify() override {
        for (auto it = observers_.begin(); it != observers_.end();) {
            if (auto obs = it->lock()) {
                obs->update(news_);
                ++it;
            } else {
                it = observers_.erase(it);
            }
        }
    }

    void setNews(const std::string& news) {
        news_ = news;
        notify();
    }
};

class NewsChannel : public IObserver {
    std::string name_;
public:
    NewsChannel(std::string name) : name_(std::move(name)) {}
    void update(const std::string& msg) override {
        std::cout << name_ << " received: " << msg << "\n";
    }
};
```

---

## 2. Strategy

Encapsulates algorithms to make them interchangeable.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>

// Classic approach: interface-based
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
    virtual std::string getName() const = 0;
};

class BubbleSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = 0; j < data.size() - i - 1; ++j)
                if (data[j] > data[j + 1])
                    std::swap(data[j], data[j + 1]);
    }
    std::string getName() const override { return "Bubble Sort"; }
};

class QuickSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        qsort(data, 0, static_cast<int>(data.size()) - 1);
    }
    std::string getName() const override { return "Quick Sort"; }
private:
    void qsort(std::vector<int>& a, int lo, int hi) {
        if (lo >= hi) return;
        int pivot = a[hi], i = lo - 1;
        for (int j = lo; j < hi; ++j)
            if (a[j] < pivot) std::swap(a[++i], a[j]);
        std::swap(a[i + 1], a[hi]);
        qsort(a, lo, i);
        qsort(a, i + 2, hi);
    }
};

class Sorter {
    std::unique_ptr<SortStrategy> strategy_;
public:
    void setStrategy(std::unique_ptr<SortStrategy> s) {
        strategy_ = std::move(s);
    }
    void sort(std::vector<int>& data) {
        if (strategy_) {
            std::cout << "Sorting with " << strategy_->getName() << "\n";
            strategy_->sort(data);
        }
    }
};
```

### Modern Approach: std::function

```cpp
class ModernSorter {
    using Strategy = std::function<void(std::vector<int>&)>;
    Strategy strategy_;
public:
    void setStrategy(Strategy s) { strategy_ = std::move(s); }
    void sort(std::vector<int>& data) { if (strategy_) strategy_(data); }
};

int main() {
    ModernSorter sorter;
    std::vector<int> data = {5, 3, 1, 4, 2};

    // Lambda strategy
    sorter.setStrategy([](std::vector<int>& v) {
        std::sort(v.begin(), v.end());
    });
    sorter.sort(data);

    for (int x : data) std::cout << x << " ";  // 1 2 3 4 5
    std::cout << "\n";
    return 0;
}
```

---

## 3. Command

Encapsulates requests as objects, enabling undo/redo.

```cpp
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
#include <string>

class Document {
    std::string content_;
public:
    void write(const std::string& text) {
        content_ += text;
    }
    void erase(size_t count) {
        if (count <= content_.size())
            content_.erase(content_.size() - count);
    }
    std::string getContent() const { return content_; }
};

class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

class WriteCommand : public Command {
    Document& doc_;
    std::string text_;
public:
    WriteCommand(Document& doc, std::string text)
        : doc_(doc), text_(std::move(text)) {}
    void execute() override { doc_.write(text_); }
    void undo() override { doc_.erase(text_.size()); }
};

class EraseCommand : public Command {
    Document& doc_;
    size_t count_;
    std::string erased_;
public:
    EraseCommand(Document& doc, size_t count) : doc_(doc), count_(count) {}
    void execute() override {
        auto content = doc_.getContent();
        if (count_ <= content.size())
            erased_ = content.substr(content.size() - count_);
        doc_.erase(count_);
    }
    void undo() override { doc_.write(erased_); }
};

// Invoker with Undo/Redo
class CommandManager {
    std::stack<std::unique_ptr<Command>> undoStack_;
    std::stack<std::unique_ptr<Command>> redoStack_;
public:
    void execute(std::unique_ptr<Command> cmd) {
        cmd->execute();
        undoStack_.push(std::move(cmd));
        while (!redoStack_.empty()) redoStack_.pop();
    }
    void undo() {
        if (undoStack_.empty()) return;
        auto cmd = std::move(undoStack_.top());
        undoStack_.pop();
        cmd->undo();
        redoStack_.push(std::move(cmd));
    }
    void redo() {
        if (redoStack_.empty()) return;
        auto cmd = std::move(redoStack_.top());
        redoStack_.pop();
        cmd->execute();
        undoStack_.push(std::move(cmd));
    }
};

// Macro command (composite)
class MacroCommand : public Command {
    std::vector<std::unique_ptr<Command>> commands_;
public:
    void addCommand(std::unique_ptr<Command> cmd) {
        commands_.push_back(std::move(cmd));
    }
    void execute() override {
        for (auto& cmd : commands_) cmd->execute();
    }
    void undo() override {
        for (auto it = commands_.rbegin(); it != commands_.rend(); ++it)
            (*it)->undo();
    }
};

int main() {
    Document doc;
    CommandManager mgr;

    mgr.execute(std::make_unique<WriteCommand>(doc, "Hello"));
    mgr.execute(std::make_unique<WriteCommand>(doc, " World"));
    std::cout << doc.getContent() << "\n";  // Hello World

    mgr.undo();
    std::cout << doc.getContent() << "\n";  // Hello

    mgr.redo();
    std::cout << doc.getContent() << "\n";  // Hello World
    return 0;
}
```

---

## 4. State

Allows an object to alter its behavior when its internal state changes.

```cpp
#include <iostream>
#include <memory>

class TrafficLight;

class State {
public:
    virtual ~State() = default;
    virtual void handle(TrafficLight& light) = 0;
    virtual std::string name() const = 0;
};

class TrafficLight {
    std::unique_ptr<State> state_;
public:
    TrafficLight(std::unique_ptr<State> initial)
        : state_(std::move(initial)) {}

    void setState(std::unique_ptr<State> s) { state_ = std::move(s); }
    void change() { state_->handle(*this); }
    std::string currentState() const { return state_->name(); }
};

class GreenState : public State {
public:
    void handle(TrafficLight& light) override;
    std::string name() const override { return "GREEN"; }
};

class YellowState : public State {
public:
    void handle(TrafficLight& light) override;
    std::string name() const override { return "YELLOW"; }
};

class RedState : public State {
public:
    void handle(TrafficLight& light) override {
        std::cout << "RED -> GREEN\n";
        light.setState(std::make_unique<GreenState>());
    }
    std::string name() const override { return "RED"; }
};

void GreenState::handle(TrafficLight& light) {
    std::cout << "GREEN -> YELLOW\n";
    light.setState(std::make_unique<YellowState>());
}

void YellowState::handle(TrafficLight& light) {
    std::cout << "YELLOW -> RED\n";
    light.setState(std::make_unique<RedState>());
}

int main() {
    TrafficLight light(std::make_unique<RedState>());
    for (int i = 0; i < 6; ++i) {
        std::cout << "Current: " << light.currentState() << " -> ";
        light.change();
    }
    return 0;
}
```

---

## 5. Template Method

Defines the skeleton of an algorithm; subclasses fill in specific steps.

```cpp
#include <iostream>
#include <string>

class DataParser {
public:
    virtual ~DataParser() = default;

    // Template method — fixed algorithm structure
    void parseFile(const std::string& filename) {
        std::cout << "=== Parsing " << filename << " ===\n";
        openFile(filename);
        extractData();
        parseData();
        analyzeData();  // Hook: optional override
        closeFile();
        std::cout << "=== Done ===\n\n";
    }

protected:
    virtual void openFile(const std::string& f) {
        std::cout << "Opening: " << f << "\n";
    }
    virtual void closeFile() { std::cout << "Closing file\n"; }

    // Mandatory steps — subclasses must implement
    virtual void extractData() = 0;
    virtual void parseData() = 0;

    // Hook — optional override
    virtual void analyzeData() {}
};

class CSVParser : public DataParser {
protected:
    void extractData() override { std::cout << "Extracting CSV rows\n"; }
    void parseData() override { std::cout << "Splitting by commas\n"; }
    void analyzeData() override { std::cout << "Counting rows/columns\n"; }
};

class JSONParser : public DataParser {
protected:
    void extractData() override { std::cout << "Extracting JSON objects\n"; }
    void parseData() override { std::cout << "Building object tree\n"; }
    void analyzeData() override { std::cout << "Validating schema\n"; }
};

int main() {
    CSVParser csv;
    csv.parseFile("data.csv");

    JSONParser json;
    json.parseFile("data.json");
    return 0;
}
```

---

## 6. Iterator

Provides sequential access to elements without exposing internal structure. In modern C++, this is mostly handled by ranges and iterators built into the STL, but custom iterators remain useful for user-defined containers.

```cpp
#include <iostream>
#include <iterator>
#include <cstddef>

template<typename T, size_t N>
class FixedArray {
    T data_[N];

public:
    // Iterator type
    class iterator {
        T* ptr_;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        explicit iterator(T* p) : ptr_(p) {}

        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        iterator& operator++() { ++ptr_; return *this; }
        iterator operator++(int) { auto tmp = *this; ++ptr_; return tmp; }
        iterator& operator--() { --ptr_; return *this; }
        iterator operator+(difference_type n) const { return iterator(ptr_ + n); }
        difference_type operator-(const iterator& o) const { return ptr_ - o.ptr_; }
        bool operator==(const iterator& o) const { return ptr_ == o.ptr_; }
        bool operator!=(const iterator& o) const { return ptr_ != o.ptr_; }
        bool operator<(const iterator& o) const { return ptr_ < o.ptr_; }
    };

    T& operator[](size_t i) { return data_[i]; }
    size_t size() const { return N; }

    iterator begin() { return iterator(data_); }
    iterator end() { return iterator(data_ + N); }
};

int main() {
    FixedArray<int, 5> arr;
    for (size_t i = 0; i < arr.size(); ++i) arr[i] = static_cast<int>(i * 10);

    // Range-based for works because we have begin/end
    for (int val : arr) {
        std::cout << val << " ";  // 0 10 20 30 40
    }
    std::cout << "\n";
    return 0;
}
```

---

## 7. CRTP

The Curiously Recurring Template Pattern provides static (compile-time) polymorphism.

### Static Polymorphism

```cpp
#include <iostream>

template<typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->areaImpl();
    }
    void draw() const {
        static_cast<const Derived*>(this)->drawImpl();
    }
};

class Circle : public Shape<Circle> {
    double radius_;
public:
    Circle(double r) : radius_(r) {}
    double areaImpl() const { return 3.14159 * radius_ * radius_; }
    void drawImpl() const {
        std::cout << "Drawing circle r=" << radius_ << "\n";
    }
};

class Rect : public Shape<Rect> {
    double w_, h_;
public:
    Rect(double w, double h) : w_(w), h_(h) {}
    double areaImpl() const { return w_ * h_; }
    void drawImpl() const {
        std::cout << "Drawing rect " << w_ << "x" << h_ << "\n";
    }
};

template<typename T>
void printArea(const Shape<T>& s) {
    std::cout << "Area: " << s.area() << "\n";
}
```

### Counter Mixin

```cpp
template<typename Derived>
class Counter {
    static inline int count_ = 0;
protected:
    Counter() { ++count_; }
    ~Counter() { --count_; }
public:
    static int getCount() { return count_; }
};

class Widget : public Counter<Widget> {};
class Gadget : public Counter<Gadget> {};
// Widget::getCount() and Gadget::getCount() are independent
```

### Fluent Interface

```cpp
template<typename Derived>
class Builder {
protected:
    std::string name_;
public:
    Derived& setName(const std::string& n) {
        name_ = n;
        return static_cast<Derived&>(*this);
    }
};

class PersonBuilder : public Builder<PersonBuilder> {
    int age_ = 0;
public:
    PersonBuilder& setAge(int a) { age_ = a; return *this; }
    void build() {
        std::cout << name_ << ", age " << age_ << "\n";
    }
};

// PersonBuilder().setName("Alice").setAge(30).build();
```

### CRTP vs Virtual Dispatch

| CRTP (static) | Virtual (dynamic) |
|---------------|-------------------|
| Resolved at compile time | Resolved at runtime |
| No vtable overhead | vtable pointer per object |
| Cannot store in heterogeneous container | Polymorphic containers work |
| Inlinable | Harder to inline |

---

## 8. PIMPL

The Pointer to Implementation idiom hides implementation details from headers, reducing compile-time dependencies.

```cpp
// === widget.h ===
#pragma once
#include <memory>
#include <string>

class Widget {
public:
    Widget();
    ~Widget();

    Widget(Widget&&) noexcept;
    Widget& operator=(Widget&&) noexcept;

    Widget(const Widget&);
    Widget& operator=(const Widget&);

    void setName(const std::string& name);
    std::string getName() const;
    void doSomething();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

// === widget.cpp ===
#include "widget.h"
#include <iostream>

class Widget::Impl {
public:
    std::string name;
    int counter = 0;

    void process() {
        std::cout << "Processing: " << name << " (" << ++counter << ")\n";
    }
};

Widget::Widget() : pImpl_(std::make_unique<Impl>()) {}
Widget::~Widget() = default;
Widget::Widget(Widget&&) noexcept = default;
Widget& Widget::operator=(Widget&&) noexcept = default;

Widget::Widget(const Widget& other)
    : pImpl_(std::make_unique<Impl>(*other.pImpl_)) {}

Widget& Widget::operator=(const Widget& other) {
    if (this != &other)
        pImpl_ = std::make_unique<Impl>(*other.pImpl_);
    return *this;
}

void Widget::setName(const std::string& name) { pImpl_->name = name; }
std::string Widget::getName() const { return pImpl_->name; }
void Widget::doSomething() { pImpl_->process(); }
```

### PIMPL Benefits

| Benefit | Explanation |
|---------|-------------|
| Compilation firewall | Changing Impl doesn't recompile clients |
| ABI stability | Adding Impl members doesn't change Widget's size |
| Hidden dependencies | Impl can #include anything without polluting the header |
| Clean headers | Header shows only the public API |

---

## 9. Type Erasure

Store heterogeneous objects in a single container without requiring a common base class.

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Drawable {
    // Inner concept (abstract interface)
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    // Inner model (wraps any T with draw())
    template<typename T>
    struct Model : Concept {
        T object_;
        Model(T obj) : object_(std::move(obj)) {}
        void draw() const override { object_.draw(); }
        std::unique_ptr<Concept> clone() const override {
            return std::make_unique<Model>(*this);
        }
    };

    std::unique_ptr<Concept> pImpl_;

public:
    template<typename T>
    Drawable(T obj) : pImpl_(std::make_unique<Model<T>>(std::move(obj))) {}

    Drawable(const Drawable& other) : pImpl_(other.pImpl_->clone()) {}
    Drawable& operator=(const Drawable& other) {
        pImpl_ = other.pImpl_->clone();
        return *this;
    }
    Drawable(Drawable&&) = default;
    Drawable& operator=(Drawable&&) = default;

    void draw() const { pImpl_->draw(); }
};

// Types with draw() — no common base class needed
struct Circle {
    double r;
    void draw() const { std::cout << "Circle(r=" << r << ")\n"; }
};

struct Square {
    double s;
    void draw() const { std::cout << "Square(s=" << s << ")\n"; }
};

struct Text {
    std::string content;
    void draw() const { std::cout << "Text(\"" << content << "\")\n"; }
};

int main() {
    // Heterogeneous container — no inheritance required
    std::vector<Drawable> shapes;
    shapes.push_back(Circle{5.0});
    shapes.push_back(Square{3.0});
    shapes.push_back(Text{"Hello"});

    for (const auto& s : shapes) {
        s.draw();
    }
    return 0;
}
```

### Type Erasure vs Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| Inheritance + virtual | Simple, well-known | Requires base class, heap allocation, no value semantics |
| `std::variant` | No heap allocation, value semantics | Closed set of types, visitor boilerplate |
| Type erasure | Open set, value semantics, no base class | More complex implementation |
| `std::any` | Stores anything | No interface, must cast to use |

---

## 10. NVI

The Non-Virtual Interface idiom makes all public methods non-virtual. Virtual functions are private, called by the public interface. This gives the base class control over pre/post conditions.

```cpp
#include <iostream>
#include <string>

class Logger {
public:
    // Non-virtual public interface
    void log(const std::string& message) {
        if (shouldLog(message)) {        // Pre-condition (non-virtual)
            doLog(formatMessage(message)); // Customization point (virtual)
            ++messageCount_;              // Post-condition (non-virtual)
        }
    }

    int messageCount() const { return messageCount_; }
    virtual ~Logger() = default;

private:
    // Customization points — subclasses override these
    virtual void doLog(const std::string& msg) = 0;
    virtual std::string formatMessage(const std::string& msg) {
        return "[LOG] " + msg;
    }
    virtual bool shouldLog(const std::string&) { return true; }

    int messageCount_ = 0;
};

class ConsoleLogger : public Logger {
    void doLog(const std::string& msg) override {
        std::cout << msg << "\n";
    }
};

class FilteredLogger : public Logger {
    std::string filter_;
    void doLog(const std::string& msg) override {
        std::cout << msg << "\n";
    }
    bool shouldLog(const std::string& msg) override {
        return msg.find(filter_) != std::string::npos;
    }
public:
    FilteredLogger(std::string filter) : filter_(std::move(filter)) {}
};

int main() {
    ConsoleLogger console;
    console.log("Application started");
    console.log("Processing data");
    std::cout << "Messages: " << console.messageCount() << "\n";

    FilteredLogger filtered("error");
    filtered.log("info: all good");    // Filtered out
    filtered.log("error: disk full");  // Logged
    std::cout << "Messages: " << filtered.messageCount() << "\n";  // 1
    return 0;
}
```

### NVI Benefits

- Base class controls invariants (logging, locking, validation)
- Subclasses focus only on the varying behavior
- Adding pre/post conditions doesn't break derived classes
- Public API is stable; virtual customization points can evolve

---

## Exercises

### Exercise 1: Event System

Implement a type-safe event system using the Observer pattern. Support multiple event types (`MouseClick`, `KeyPress`, `WindowResize`). Subscribers register for specific event types. Test with at least three subscribers listening to different events.

### Exercise 2: Strategy with Compile-Time Selection

Implement a `Compressor` class that uses the Strategy pattern. Provide `GzipStrategy`, `LZ4Strategy`, and `NoOpStrategy`. Then implement a compile-time variant using CRTP where the strategy is a template parameter. Compare the runtime and compile-time approaches.

### Exercise 3: Text Editor with Command Pattern

Build a text editor with full undo/redo support. Commands: `Insert(pos, text)`, `Delete(pos, length)`, `Replace(pos, length, text)`. Implement macro recording: start recording, execute commands, stop recording, replay the macro.

### Exercise 4: PIMPL Library

Create a `Database` class using PIMPL. The header exposes only `connect()`, `query()`, `disconnect()`. The implementation uses SQLite internally. Show that changing the implementation to a mock database requires zero changes to client code.

### Exercise 5: Type-Erased Function

Implement a simplified `MyFunction<R(Args...)>` using type erasure (similar to `std::function`). Support callable objects, lambdas, and function pointers. Test by storing different callables in a `vector<MyFunction<int(int)>>`.

---

## Next Steps

With both behavioral patterns and C++ idioms in your toolkit, you can design systems that are flexible, maintainable, and idiomatic. The next lesson covers C++23 features that bring further improvements to the language.

- [C++23 Features](./16_CPP23_Features.md)
