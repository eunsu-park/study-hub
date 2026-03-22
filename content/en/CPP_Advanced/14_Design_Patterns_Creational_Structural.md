# Design Patterns: Creational and Structural

**Previous**: [Advanced Concurrency](./13_Concurrency_Advanced.md) | **Next**: [Design Patterns: Behavioral and C++ Idioms](./15_Design_Patterns_Behavioral_Idioms.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the SOLID principles and their role as the foundation for applying design patterns effectively
2. Implement the Singleton pattern using Meyers' Singleton and template-based approaches
3. Apply the Factory Method and Abstract Factory patterns to decouple object creation from usage
4. Use the Builder pattern with fluent interfaces and optional Director classes for complex object construction
5. Implement the Prototype pattern for polymorphic cloning
6. Apply structural patterns (Adapter, Decorator, Facade, Composite, Proxy) to compose flexible object hierarchies
7. Evaluate when a design pattern adds value versus when simpler alternatives suffice

---

Design patterns are not academic exercises--they are battle-tested solutions to problems that every non-trivial C++ project encounters. Knowing when a Factory simplifies object creation, when a Decorator adds behavior without subclassing, or when a Facade tames a complex subsystem lets you write architectures that are flexible without being over-engineered. Combined with modern C++ features like smart pointers, lambdas, and templates, these patterns become lighter and more expressive than their textbook forms.

---

## Table of Contents

1. [Design Patterns Overview](#1-design-patterns-overview)
2. [SOLID Principles](#2-solid-principles)
3. [Singleton](#3-singleton)
4. [Factory Method](#4-factory-method)
5. [Abstract Factory](#5-abstract-factory)
6. [Builder](#6-builder)
7. [Prototype](#7-prototype)
8. [Adapter](#8-adapter)
9. [Decorator](#9-decorator)
10. [Facade](#10-facade)
11. [Composite](#11-composite)
12. [Proxy](#12-proxy)

---

## 1. Design Patterns Overview

### Classification

| Creational Patterns | Structural Patterns | Behavioral Patterns |
|---|---|---|
| Singleton | Adapter | Observer |
| Factory Method | Decorator | Strategy |
| Abstract Factory | Facade | Command |
| Builder | Composite | State |
| Prototype | Proxy | Iterator |
| | Bridge | Template Method |
| | Flyweight | |

This lesson covers Creational and Structural patterns. Behavioral patterns and C++ idioms are in the next lesson.

---

## 2. SOLID Principles

```cpp
// S - Single Responsibility Principle
// A class should have only one reason to change.

// Bad: multiple responsibilities mixed
class BadUserManager {
public:
    void createUser(const std::string& name) { /* ... */ }
    void saveToDatabase() { /* ... */ }     // DB responsibility
    void sendEmail() { /* ... */ }          // Email responsibility
};

// Good: separated responsibilities
class User {
public:
    User(const std::string& name) : name_(name) {}
    std::string getName() const { return name_; }
private:
    std::string name_;
};

class UserRepository {
public:
    void save(const User& user) { /* Save to DB */ }
};

class EmailService {
public:
    void sendWelcome(const User& user) { /* Send email */ }
};
```

```cpp
// O - Open/Closed Principle
// Open for extension, closed for modification.

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
};

class Rectangle : public Shape {
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
private:
    double width, height;
};

// Adding Circle doesn't modify Shape or Rectangle
class Circle : public Shape {
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
private:
    double radius;
};
```

```cpp
// L - Liskov Substitution Principle
// Subtypes must be substitutable for their base types.

// I - Interface Segregation Principle
// Clients should not depend on interfaces they don't use.

// Bad: bloated interface
class IBadWorker {
public:
    virtual void work() = 0;
    virtual void eat() = 0;    // Robots don't eat
    virtual void sleep() = 0;  // Robots don't sleep
};

// Good: segregated interfaces
class IWorkable { public: virtual void work() = 0; };
class IFeedable { public: virtual void eat() = 0; };

// D - Dependency Inversion Principle
// Depend on abstractions, not concretions.

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const std::string& message) = 0;
};

class Application {
public:
    Application(std::shared_ptr<ILogger> logger) : logger_(logger) {}
    void run() { logger_->log("Application started"); }
private:
    std::shared_ptr<ILogger> logger_;
};
```

---

## 3. Singleton

Ensures only one instance exists.

```cpp
#include <mutex>
#include <memory>
#include <iostream>

// Meyers' Singleton — Thread-safe since C++11
class Singleton {
public:
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;

    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

    void doSomething() {
        std::cout << "Singleton doing something\n";
    }

private:
    Singleton() { std::cout << "Singleton created\n"; }
    ~Singleton() { std::cout << "Singleton destroyed\n"; }
};
```

### Template Singleton

```cpp
template<typename T>
class SingletonBase {
public:
    SingletonBase(const SingletonBase&) = delete;
    SingletonBase& operator=(const SingletonBase&) = delete;

    static T& getInstance() {
        static T instance;
        return instance;
    }

protected:
    SingletonBase() = default;
    ~SingletonBase() = default;
};

class Logger : public SingletonBase<Logger> {
    friend class SingletonBase<Logger>;
public:
    void log(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[LOG] " << msg << "\n";
    }
private:
    Logger() = default;
    std::mutex mutex_;
};

int main() {
    Singleton::getInstance().doSomething();
    Logger::getInstance().log("Hello, Singleton!");
    return 0;
}
```

---

## 4. Factory Method

Delegates object creation to subclasses or a registry.

```cpp
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <iostream>

// Product interface
class Document {
public:
    virtual ~Document() = default;
    virtual void open() = 0;
    virtual void save() = 0;
    virtual std::string getType() const = 0;
};

// Concrete products
class PDFDocument : public Document {
public:
    void open() override { std::cout << "Opening PDF document\n"; }
    void save() override { std::cout << "Saving PDF document\n"; }
    std::string getType() const override { return "PDF"; }
};

class WordDocument : public Document {
public:
    void open() override { std::cout << "Opening Word document\n"; }
    void save() override { std::cout << "Saving Word document\n"; }
    std::string getType() const override { return "Word"; }
};

// Self-registering factory
class DocumentFactory {
public:
    using Creator = std::function<std::unique_ptr<Document>()>;

    static void registerType(const std::string& type, Creator creator) {
        getRegistry()[type] = std::move(creator);
    }

    static std::unique_ptr<Document> create(const std::string& type) {
        auto it = getRegistry().find(type);
        if (it != getRegistry().end()) {
            return it->second();
        }
        throw std::runtime_error("Unknown document type: " + type);
    }

private:
    static std::unordered_map<std::string, Creator>& getRegistry() {
        static std::unordered_map<std::string, Creator> registry;
        return registry;
    }
};

// Auto-registration helper
template<typename T>
struct DocumentRegistrar {
    DocumentRegistrar(const std::string& type) {
        DocumentFactory::registerType(type, [] {
            return std::make_unique<T>();
        });
    }
};

static DocumentRegistrar<PDFDocument> pdfReg("pdf");
static DocumentRegistrar<WordDocument> wordReg("word");

int main() {
    auto doc = DocumentFactory::create("pdf");
    doc->open();
    return 0;
}
```

---

## 5. Abstract Factory

Creates families of related objects without specifying concrete classes.

```cpp
#include <memory>
#include <iostream>

// Abstract products
class Button {
public:
    virtual ~Button() = default;
    virtual void render() = 0;
};

class TextBox {
public:
    virtual ~TextBox() = default;
    virtual void render() = 0;
};

// Concrete products: Dark theme
class DarkButton : public Button {
public:
    void render() override { std::cout << "[Dark Button]\n"; }
};

class DarkTextBox : public TextBox {
public:
    void render() override { std::cout << "[Dark TextBox]\n"; }
};

// Concrete products: Light theme
class LightButton : public Button {
public:
    void render() override { std::cout << "[Light Button]\n"; }
};

class LightTextBox : public TextBox {
public:
    void render() override { std::cout << "[Light TextBox]\n"; }
};

// Abstract factory
class UIFactory {
public:
    virtual ~UIFactory() = default;
    virtual std::unique_ptr<Button> createButton() = 0;
    virtual std::unique_ptr<TextBox> createTextBox() = 0;
};

class DarkThemeFactory : public UIFactory {
public:
    std::unique_ptr<Button> createButton() override {
        return std::make_unique<DarkButton>();
    }
    std::unique_ptr<TextBox> createTextBox() override {
        return std::make_unique<DarkTextBox>();
    }
};

class LightThemeFactory : public UIFactory {
public:
    std::unique_ptr<Button> createButton() override {
        return std::make_unique<LightButton>();
    }
    std::unique_ptr<TextBox> createTextBox() override {
        return std::make_unique<LightTextBox>();
    }
};

// Client code — depends only on abstract factory
void buildUI(UIFactory& factory) {
    auto btn = factory.createButton();
    auto txt = factory.createTextBox();
    btn->render();
    txt->render();
}

int main() {
    DarkThemeFactory dark;
    LightThemeFactory light;

    std::cout << "Dark theme:\n";
    buildUI(dark);

    std::cout << "Light theme:\n";
    buildUI(light);

    return 0;
}
```

---

## 6. Builder

Builds complex objects step by step with a fluent interface.

```cpp
#include <string>
#include <optional>
#include <iostream>

class Computer {
public:
    void showSpecs() const {
        std::cout << "CPU: " << cpu << "\n";
        std::cout << "RAM: " << ram << "GB\n";
        std::cout << "Storage: " << storage << "GB " << storageType << "\n";
        std::cout << "GPU: " << gpu.value_or("Integrated") << "\n";
        std::cout << "OS: " << os.value_or("None") << "\n";
    }

    friend class ComputerBuilder;

private:
    std::string cpu;
    int ram = 0;
    int storage = 0;
    std::string storageType;
    std::optional<std::string> gpu;
    std::optional<std::string> os;
};

class ComputerBuilder {
public:
    ComputerBuilder& setCPU(const std::string& cpu) {
        computer_.cpu = cpu; return *this;
    }
    ComputerBuilder& setRAM(int gb) {
        computer_.ram = gb; return *this;
    }
    ComputerBuilder& setStorage(int gb, const std::string& type = "SSD") {
        computer_.storage = gb; computer_.storageType = type; return *this;
    }
    ComputerBuilder& setGPU(const std::string& gpu) {
        computer_.gpu = gpu; return *this;
    }
    ComputerBuilder& setOS(const std::string& os) {
        computer_.os = os; return *this;
    }

    Computer build() {
        if (computer_.cpu.empty()) throw std::runtime_error("CPU is required");
        if (computer_.ram <= 0) throw std::runtime_error("RAM must be positive");
        return std::move(computer_);
    }

private:
    Computer computer_;
};

// Director: predefined configurations
class ComputerDirector {
public:
    static Computer buildGamingPC() {
        return ComputerBuilder()
            .setCPU("Intel i9-13900K").setRAM(64)
            .setStorage(2000, "NVMe SSD").setGPU("NVIDIA RTX 4090")
            .setOS("Windows 11").build();
    }

    static Computer buildOfficePC() {
        return ComputerBuilder()
            .setCPU("Intel i5-13400").setRAM(16)
            .setStorage(512, "SSD").setOS("Windows 11").build();
    }
};

int main() {
    auto gaming = ComputerDirector::buildGamingPC();
    gaming.showSpecs();
    return 0;
}
```

---

## 7. Prototype

Creates new objects by cloning existing ones. Useful when construction is expensive or when you need polymorphic copies.

```cpp
#include <memory>
#include <iostream>
#include <unordered_map>

class Shape {
public:
    virtual ~Shape() = default;
    virtual std::unique_ptr<Shape> clone() const = 0;
    virtual void draw() const = 0;
};

class Circle : public Shape {
    double radius;
    std::string color;

public:
    Circle(double r, const std::string& c) : radius(r), color(c) {}

    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Circle>(*this);
    }

    void draw() const override {
        std::cout << "Circle(r=" << radius << ", color=" << color << ")\n";
    }
};

class Rectangle : public Shape {
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Rectangle>(*this);
    }

    void draw() const override {
        std::cout << "Rectangle(" << width << "x" << height << ")\n";
    }
};

// Prototype registry
class ShapeRegistry {
    std::unordered_map<std::string, std::unique_ptr<Shape>> prototypes_;

public:
    void registerShape(const std::string& name, std::unique_ptr<Shape> proto) {
        prototypes_[name] = std::move(proto);
    }

    std::unique_ptr<Shape> create(const std::string& name) const {
        auto it = prototypes_.find(name);
        if (it != prototypes_.end()) {
            return it->second->clone();
        }
        return nullptr;
    }
};

int main() {
    ShapeRegistry registry;
    registry.registerShape("red_circle",
        std::make_unique<Circle>(5.0, "red"));
    registry.registerShape("unit_rect",
        std::make_unique<Rectangle>(1.0, 1.0));

    auto s1 = registry.create("red_circle");
    auto s2 = registry.create("unit_rect");
    s1->draw();  // Circle(r=5, color=red)
    s2->draw();  // Rectangle(1x1)

    return 0;
}
```

---

## 8. Adapter

Connects incompatible interfaces.

```cpp
#include <memory>
#include <iostream>

// Target interface (what the client expects)
class MediaPlayer {
public:
    virtual ~MediaPlayer() = default;
    virtual void play(const std::string& filename) = 0;
};

// Adaptee (existing class with incompatible interface)
class AdvancedPlayer {
public:
    void playMP4(const std::string& f) {
        std::cout << "Playing MP4: " << f << "\n";
    }
    void playMKV(const std::string& f) {
        std::cout << "Playing MKV: " << f << "\n";
    }
};

// Adapter (wraps Adaptee to match Target)
class MediaAdapter : public MediaPlayer {
    AdvancedPlayer player_;

public:
    void play(const std::string& filename) override {
        if (filename.ends_with(".mp4")) {
            player_.playMP4(filename);
        } else if (filename.ends_with(".mkv")) {
            player_.playMKV(filename);
        } else {
            std::cout << "Unsupported format\n";
        }
    }
};

int main() {
    MediaAdapter player;
    player.play("movie.mp4");  // Playing MP4: movie.mp4
    player.play("video.mkv");  // Playing MKV: video.mkv
    return 0;
}
```

---

## 9. Decorator

Dynamically adds functionality to objects without modifying their class.

```cpp
#include <memory>
#include <iostream>
#include <string>

class Coffee {
public:
    virtual ~Coffee() = default;
    virtual std::string getDescription() const = 0;
    virtual double getCost() const = 0;
};

class Espresso : public Coffee {
public:
    std::string getDescription() const override { return "Espresso"; }
    double getCost() const override { return 2.00; }
};

class CoffeeDecorator : public Coffee {
protected:
    std::unique_ptr<Coffee> coffee_;
public:
    explicit CoffeeDecorator(std::unique_ptr<Coffee> c) : coffee_(std::move(c)) {}
    std::string getDescription() const override { return coffee_->getDescription(); }
    double getCost() const override { return coffee_->getCost(); }
};

class Milk : public CoffeeDecorator {
public:
    explicit Milk(std::unique_ptr<Coffee> c) : CoffeeDecorator(std::move(c)) {}
    std::string getDescription() const override {
        return coffee_->getDescription() + ", Milk";
    }
    double getCost() const override { return coffee_->getCost() + 0.50; }
};

class Mocha : public CoffeeDecorator {
public:
    explicit Mocha(std::unique_ptr<Coffee> c) : CoffeeDecorator(std::move(c)) {}
    std::string getDescription() const override {
        return coffee_->getDescription() + ", Mocha";
    }
    double getCost() const override { return coffee_->getCost() + 0.80; }
};

class Whip : public CoffeeDecorator {
public:
    explicit Whip(std::unique_ptr<Coffee> c) : CoffeeDecorator(std::move(c)) {}
    std::string getDescription() const override {
        return coffee_->getDescription() + ", Whip";
    }
    double getCost() const override { return coffee_->getCost() + 0.70; }
};

int main() {
    std::unique_ptr<Coffee> order = std::make_unique<Espresso>();
    order = std::make_unique<Milk>(std::move(order));
    order = std::make_unique<Mocha>(std::move(order));
    order = std::make_unique<Whip>(std::move(order));

    std::cout << order->getDescription() << " $" << order->getCost() << "\n";
    // Espresso, Milk, Mocha, Whip $4.00
    return 0;
}
```

---

## 10. Facade

Provides a simplified interface to a complex subsystem.

```cpp
#include <iostream>
#include <memory>

class CPU {
public:
    void freeze() { std::cout << "CPU: Freezing\n"; }
    void jump(long addr) { std::cout << "CPU: Jumping to " << addr << "\n"; }
    void execute() { std::cout << "CPU: Executing\n"; }
};

class Memory {
public:
    void load(long addr, const std::string& data) {
        std::cout << "Memory: Loading '" << data << "' at " << addr << "\n";
    }
};

class HardDrive {
public:
    std::string read(long lba, int size) {
        std::cout << "HDD: Reading " << size << " bytes from sector " << lba << "\n";
        return "boot_data";
    }
};

// Facade
class ComputerFacade {
    std::unique_ptr<CPU> cpu_ = std::make_unique<CPU>();
    std::unique_ptr<Memory> mem_ = std::make_unique<Memory>();
    std::unique_ptr<HardDrive> hdd_ = std::make_unique<HardDrive>();

public:
    void start() {
        std::cout << "=== Starting Computer ===\n";
        cpu_->freeze();
        mem_->load(0x00, hdd_->read(0, 512));
        cpu_->jump(0x00);
        cpu_->execute();
        std::cout << "=== Ready ===\n";
    }
};

int main() {
    ComputerFacade computer;
    computer.start();
    return 0;
}
```

---

## 11. Composite

Treats individual objects and compositions uniformly through a tree structure.

```cpp
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <numeric>

class FileSystemEntry {
public:
    virtual ~FileSystemEntry() = default;
    virtual std::string name() const = 0;
    virtual size_t size() const = 0;
    virtual void print(int indent = 0) const = 0;
};

class File : public FileSystemEntry {
    std::string name_;
    size_t size_;

public:
    File(std::string n, size_t s) : name_(std::move(n)), size_(s) {}
    std::string name() const override { return name_; }
    size_t size() const override { return size_; }
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << name_ << " (" << size_ << " bytes)\n";
    }
};

class Directory : public FileSystemEntry {
    std::string name_;
    std::vector<std::unique_ptr<FileSystemEntry>> children_;

public:
    explicit Directory(std::string n) : name_(std::move(n)) {}
    std::string name() const override { return name_; }

    size_t size() const override {
        size_t total = 0;
        for (const auto& child : children_) {
            total += child->size();
        }
        return total;
    }

    void add(std::unique_ptr<FileSystemEntry> entry) {
        children_.push_back(std::move(entry));
    }

    void print(int indent = 0) const override {
        std::cout << std::string(indent, ' ') << name_ << "/\n";
        for (const auto& child : children_) {
            child->print(indent + 2);
        }
    }
};

int main() {
    auto root = std::make_unique<Directory>("root");
    root->add(std::make_unique<File>("readme.md", 1024));

    auto src = std::make_unique<Directory>("src");
    src->add(std::make_unique<File>("main.cpp", 2048));
    src->add(std::make_unique<File>("utils.cpp", 512));
    root->add(std::move(src));

    root->print();
    std::cout << "Total size: " << root->size() << " bytes\n";
    return 0;
}
```

---

## 12. Proxy

Controls access to another object.

```cpp
#include <memory>
#include <iostream>
#include <unordered_map>

class Image {
public:
    virtual ~Image() = default;
    virtual void display() = 0;
};

// Real subject — expensive to create
class RealImage : public Image {
    std::string filename_;

public:
    explicit RealImage(const std::string& f) : filename_(f) {
        loadFromDisk();
    }

    void display() override {
        std::cout << "Displaying " << filename_ << "\n";
    }

private:
    void loadFromDisk() {
        std::cout << "Loading " << filename_ << " from disk...\n";
    }
};

// Proxy — lazy loading
class ProxyImage : public Image {
    std::string filename_;
    std::unique_ptr<RealImage> real_;

public:
    explicit ProxyImage(const std::string& f) : filename_(f) {}

    void display() override {
        if (!real_) {
            real_ = std::make_unique<RealImage>(filename_);
        }
        real_->display();
    }
};

// Caching proxy
class CachingProxy : public Image {
    std::string filename_;
    static std::unordered_map<std::string, std::shared_ptr<RealImage>> cache_;

public:
    explicit CachingProxy(const std::string& f) : filename_(f) {}

    void display() override {
        auto it = cache_.find(filename_);
        if (it == cache_.end()) {
            cache_[filename_] = std::make_shared<RealImage>(filename_);
        }
        cache_[filename_]->display();
    }
};
std::unordered_map<std::string, std::shared_ptr<RealImage>> CachingProxy::cache_;

int main() {
    ProxyImage img("photo.jpg");
    // Image not loaded yet

    img.display();  // Loads then displays
    img.display();  // Just displays (already loaded)

    return 0;
}
```

---

## Exercises

### Exercise 1: Plugin System with Factory

Design a plugin system using the Factory Method pattern. Define an `IPlugin` interface with `name()`, `version()`, and `execute()`. Create a `PluginFactory` with self-registration. Implement at least three plugins and a main program that loads them by name from command-line arguments.

### Exercise 2: Logger with Decorator

Implement a logging system using the Decorator pattern. The base `Logger` writes plain text. Decorators add: timestamp prefix, log level (INFO, WARN, ERROR), and file output. Show that decorators can be combined in any order.

### Exercise 3: Composite Expression Tree

Build an expression tree using the Composite pattern. `Leaf` nodes hold numeric values. `BinaryOp` nodes hold an operator (+, -, *, /) and two children. Implement `evaluate()` and `toString()` on both. Parse "3 + 4 * 2" into the tree and evaluate it.

### Exercise 4: Smart Proxy

Implement a `LoggingProxy<T>` template that wraps any object and logs every method call (method name, arguments, return value). Use `operator->` overloading to intercept calls transparently.

### Exercise 5: Abstract Factory for Database Layers

Design an Abstract Factory for database access. The factory creates `Connection`, `Command`, and `ResultSet` objects. Implement two concrete families: `SQLite` and `Mock` (in-memory). Write client code that works with either family without modification.

---

## Next Steps

This lesson covered Creational and Structural patterns. The next lesson completes the picture with Behavioral patterns (Observer, Strategy, Command, State, Template Method, Iterator) and C++ idioms (CRTP, PIMPL, type erasure, NVI).

- [Design Patterns: Behavioral and C++ Idioms](./15_Design_Patterns_Behavioral_Idioms.md)
