# 디자인 패턴: 행위 및 C++ 이디엄

**이전**: [디자인 패턴: 생성 및 구조](./14_Design_Patterns_Creational_Structural.md) | **다음**: [C++23 기능](./16_CPP23_Features.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 인터페이스 기반과 현대적 시그널/슬롯 접근법 모두를 사용하여 옵저버 패턴을 구현할 수 있다
2. 람다 기반 변형을 포함하여 교환 가능한 알고리즘을 캡슐화하는 전략 패턴을 적용할 수 있다
3. 매크로 커맨드 합성과 함께 커맨드 패턴을 사용하여 실행 취소/재실행 시스템을 구축할 수 있다
4. 상태 패턴과 템플릿 메서드 패턴을 사용하여 복잡한 제어 흐름과 알고리즘 골격을 관리할 수 있다
5. 정적 다형성, 카운터 믹스인, 플루언트 인터페이스를 위한 CRTP를 구현할 수 있다
6. PIMPL 이디엄을 적용하여 빌드 의존성을 줄이는 컴파일 방화벽을 만들 수 있다
7. 타입 소거(type erasure)를 사용하여 공통 기반 클래스 없이 이질적인 객체를 단일 컨테이너에 저장할 수 있다

---

행위 패턴은 객체가 어떻게 통신하고 책임을 분배하는지 정의합니다. 객체가 *무엇인지*와 *어떻게 구성되는지*에 초점을 맞추는 생성 및 구조 패턴과 달리, 행위 패턴은 *객체가 무엇을 하는지*와 *어떻게 상호작용하는지*에 초점을 맞춥니다. C++는 자체적인 이디엄 -- CRTP, PIMPL, 타입 소거, NVI -- 을 추가하며, 이는 다른 언어에서 사용할 수 없는 템플릿, 값 의미론, 컴파일 모델 기능을 활용합니다. GoF 행위 패턴과 C++ 이디엄을 모두 마스터하면 유지보수 가능한 아키텍처를 설계하기 위한 완전한 도구 키트를 갖게 됩니다.

---

## 목차

1. [옵저버](#1-옵저버)
2. [전략](#2-전략)
3. [커맨드](#3-커맨드)
4. [상태](#4-상태)
5. [템플릿 메서드](#5-템플릿-메서드)
6. [반복자](#6-반복자)
7. [CRTP (Curiously Recurring Template Pattern)](#7-crtp)
8. [PIMPL (Pointer to Implementation)](#8-pimpl)
9. [타입 소거 (Type Erasure)](#9-타입-소거)
10. [NVI (Non-Virtual Interface)](#10-nvi)

---

## 1. 옵저버

주체(subject)의 상태가 변경될 때 여러 객체에 알립니다.

### 현대적 시그널/슬롯 접근법

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

// 사용: 주가 모니터링
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

### 전통적 인터페이스 기반 옵저버

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

## 2. 전략

알고리즘을 캡슐화하여 교환 가능하게 만듭니다.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>

// 클래식 접근법: 인터페이스 기반
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

### 현대적 접근법: std::function

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

    // 람다 전략
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

## 3. 커맨드

요청을 객체로 캡슐화하여 실행 취소/재실행을 가능하게 합니다.

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

// 실행 취소/재실행을 가진 인보커
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

// 매크로 커맨드 (컴포지트)
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

## 4. 상태

내부 상태가 변경될 때 객체의 동작을 변경할 수 있게 합니다.

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

## 5. 템플릿 메서드

알고리즘의 골격을 정의하고, 하위 클래스가 특정 단계를 채웁니다.

```cpp
#include <iostream>
#include <string>

class DataParser {
public:
    virtual ~DataParser() = default;

    // 템플릿 메서드 — 고정된 알고리즘 구조
    void parseFile(const std::string& filename) {
        std::cout << "=== Parsing " << filename << " ===\n";
        openFile(filename);
        extractData();
        parseData();
        analyzeData();  // 훅: 선택적 오버라이드
        closeFile();
        std::cout << "=== Done ===\n\n";
    }

protected:
    virtual void openFile(const std::string& f) {
        std::cout << "Opening: " << f << "\n";
    }
    virtual void closeFile() { std::cout << "Closing file\n"; }

    // 필수 단계 — 하위 클래스가 반드시 구현
    virtual void extractData() = 0;
    virtual void parseData() = 0;

    // 훅 — 선택적 오버라이드
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

## 6. 반복자

내부 구조를 노출하지 않고 요소에 순차적으로 접근합니다. 현대 C++에서는 대부분 STL에 내장된 레인지와 반복자로 처리되지만, 사용자 정의 컨테이너에는 커스텀 반복자가 여전히 유용합니다.

```cpp
#include <iostream>
#include <iterator>
#include <cstddef>

template<typename T, size_t N>
class FixedArray {
    T data_[N];

public:
    // 반복자 타입
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

    // begin/end가 있으므로 범위 기반 for가 동작
    for (int val : arr) {
        std::cout << val << " ";  // 0 10 20 30 40
    }
    std::cout << "\n";
    return 0;
}
```

---

## 7. CRTP

CRTP(Curiously Recurring Template Pattern)는 정적(컴파일 타임) 다형성을 제공합니다.

### 정적 다형성

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

### 카운터 믹스인

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
// Widget::getCount()와 Gadget::getCount()는 독립적
```

### 플루언트 인터페이스

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

### CRTP vs 가상 디스패치

| CRTP (정적) | 가상 (동적) |
|-------------|------------|
| 컴파일 타임에 결정 | 런타임에 결정 |
| vtable 오버헤드 없음 | 객체당 vtable 포인터 |
| 이질적 컨테이너에 저장 불가 | 다형적 컨테이너 동작 |
| 인라인 가능 | 인라인 어려움 |

---

## 8. PIMPL

PIMPL(Pointer to Implementation) 이디엄은 헤더에서 구현 세부 사항을 숨겨 컴파일 타임 의존성을 줄입니다.

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

### PIMPL의 장점

| 장점 | 설명 |
|------|------|
| 컴파일 방화벽 | Impl 변경이 클라이언트 재컴파일을 유발하지 않음 |
| ABI 안정성 | Impl 멤버 추가가 Widget 크기를 변경하지 않음 |
| 의존성 숨김 | Impl이 헤더를 오염시키지 않고 무엇이든 #include 가능 |
| 깔끔한 헤더 | 헤더가 공개 API만 표시 |

---

## 9. 타입 소거

공통 기반 클래스 없이 이질적인 객체를 단일 컨테이너에 저장합니다.

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Drawable {
    // 내부 컨셉 (추상 인터페이스)
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    // 내부 모델 (draw()를 가진 모든 T를 래핑)
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

// draw()를 가진 타입들 — 공통 기반 클래스 불필요
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
    // 이질적 컨테이너 — 상속 불필요
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

### 타입 소거 vs 대안

| 접근법 | 장점 | 단점 |
|--------|------|------|
| 상속 + virtual | 간단, 잘 알려진 | 기반 클래스 필요, 힙 할당, 값 의미론 없음 |
| `std::variant` | 힙 할당 없음, 값 의미론 | 폐쇄 타입 집합, 방문자 보일러플레이트 |
| 타입 소거 | 개방 타입 집합, 값 의미론, 기반 클래스 없음 | 구현이 더 복잡 |
| `std::any` | 무엇이든 저장 | 인터페이스 없음, 사용하려면 캐스트 필요 |

---

## 10. NVI

NVI(Non-Virtual Interface) 이디엄은 모든 공개 메서드를 비가상으로 만듭니다. 가상 함수는 private이며 공개 인터페이스에 의해 호출됩니다. 이를 통해 기반 클래스가 사전/사후 조건을 제어할 수 있습니다.

```cpp
#include <iostream>
#include <string>

class Logger {
public:
    // 비가상 공개 인터페이스
    void log(const std::string& message) {
        if (shouldLog(message)) {        // 사전 조건 (비가상)
            doLog(formatMessage(message)); // 커스터마이즈 포인트 (가상)
            ++messageCount_;              // 사후 조건 (비가상)
        }
    }

    int messageCount() const { return messageCount_; }
    virtual ~Logger() = default;

private:
    // 커스터마이즈 포인트 — 하위 클래스가 이것을 오버라이드
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
    filtered.log("info: all good");    // 필터링됨
    filtered.log("error: disk full");  // 로깅됨
    std::cout << "Messages: " << filtered.messageCount() << "\n";  // 1
    return 0;
}
```

### NVI의 장점

- 기반 클래스가 불변조건을 제어 (로깅, 락, 검증)
- 하위 클래스는 변하는 동작에만 집중
- 사전/사후 조건 추가가 파생 클래스를 깨뜨리지 않음
- 공개 API가 안정적; 가상 커스터마이즈 포인트가 진화 가능

---

## 연습 문제

### 연습 1: 이벤트 시스템

옵저버 패턴을 사용하여 타입 안전 이벤트 시스템을 구현하세요. 여러 이벤트 타입(`MouseClick`, `KeyPress`, `WindowResize`)을 지원하세요. 구독자가 특정 이벤트 타입에 등록합니다. 최소 세 명의 구독자가 다른 이벤트를 청취하도록 테스트하세요.

### 연습 2: 컴파일 타임 선택이 가능한 전략

전략 패턴을 사용하는 `Compressor` 클래스를 구현하세요. `GzipStrategy`, `LZ4Strategy`, `NoOpStrategy`를 제공하세요. 그런 다음 전략이 템플릿 매개변수인 CRTP를 사용한 컴파일 타임 변형을 구현하세요. 런타임과 컴파일 타임 접근법을 비교하세요.

### 연습 3: 커맨드 패턴을 사용한 텍스트 편집기

완전한 실행 취소/재실행 지원이 있는 텍스트 편집기를 구축하세요. 커맨드: `Insert(pos, text)`, `Delete(pos, length)`, `Replace(pos, length, text)`. 매크로 기록을 구현하세요: 기록 시작, 커맨드 실행, 기록 중지, 매크로 재생.

### 연습 4: PIMPL 라이브러리

PIMPL을 사용한 `Database` 클래스를 만드세요. 헤더는 `connect()`, `query()`, `disconnect()`만 노출합니다. 구현은 내부적으로 SQLite를 사용합니다. 구현을 모의 데이터베이스로 변경해도 클라이언트 코드에 변경이 필요 없음을 보이세요.

### 연습 5: 타입 소거 함수

타입 소거를 사용하여 간소화된 `MyFunction<R(Args...)>`를 구현하세요 (`std::function`과 유사). 호출 가능 객체, 람다, 함수 포인터를 지원하세요. `vector<MyFunction<int(int)>>`에 다른 호출 가능 객체를 저장하여 테스트하세요.

---

## 다음 단계

행위 패턴과 C++ 이디엄을 모두 갖추면 유연하고 유지보수 가능하며 관용적인 시스템을 설계할 수 있습니다. 다음 레슨은 언어에 추가적인 개선을 가져오는 C++23 기능을 다룹니다.

- [C++23 기능](./16_CPP23_Features.md)
