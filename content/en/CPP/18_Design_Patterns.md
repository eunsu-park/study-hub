# 18. C++ Design Patterns

**Previous**: [Advanced C++20](./17_CPP20_Advanced.md) | **Next**: [Project: Student Management System](./19_Project_Student_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose and classification of GoF design patterns (Creational, Structural, Behavioral)
2. Implement creational patterns (Singleton, Factory Method, Builder) using modern C++ idioms
3. Apply structural patterns (Adapter, Decorator, Facade) to compose flexible object hierarchies
4. Use behavioral patterns (Observer, Strategy, Command, Template Method) to encapsulate varying behavior
5. Leverage C++-specific idioms including CRTP for static polymorphism and PIMPL for compilation firewalls
6. Evaluate trade-offs between pattern-based and lambda/function-based approaches in modern C++

---

Design patterns are not academic exercises--they are battle-tested solutions to problems that every non-trivial C++ project encounters. Knowing when a Factory simplifies object creation, when an Observer decouples modules, or when CRTP eliminates virtual-call overhead lets you write architectures that are flexible without being over-engineered. Combined with modern C++ features like smart pointers, lambdas, and templates, these patterns become lighter and more expressive than their textbook forms.

## Table of Contents
1. [Design Patterns Overview](#1-design-patterns-overview)
2. [Creational Patterns](#2-creational-patterns)
3. [Structural Patterns](#3-structural-patterns)
4. [Behavioral Patterns](#4-behavioral-patterns)
5. [C++-Specific Idioms](#5-c-specific-idioms)
6. [Practice Problems](#6-practice-problems)

---

## 1. Design Patterns Overview

### 1.1 What are Design Patterns?

Design patterns are reusable solutions to commonly occurring problems in software design.

```
┌─────────────────────────────────────────────────────────────┐
│                    GoF Design Pattern Categories            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Creational  │  │ Structural  │  │ Behavioral  │         │
│  │  Patterns   │  │  Patterns   │  │  Patterns   │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │ Singleton   │  │ Adapter     │  │ Observer    │         │
│  │ Factory     │  │ Decorator   │  │ Strategy    │         │
│  │ Builder     │  │ Facade      │  │ Command     │         │
│  │ Prototype   │  │ Composite   │  │ State       │         │
│  │ Abstract    │  │ Proxy       │  │ Iterator    │         │
│  │   Factory   │  │ Bridge      │  │ Template    │         │
│  │             │  │ Flyweight   │  │   Method    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 SOLID Principles

```cpp
// SOLID Principles - Foundation of good object-oriented design

// S - Single Responsibility Principle
// A class should have only one responsibility

// Bad example: multiple responsibilities mixed
class BadUserManager {
public:
    void createUser(const std::string& name) { /* ... */ }
    void saveToDatabase() { /* ... */ }     // DB responsibility
    void sendEmail() { /* ... */ }          // Email responsibility
    void generateReport() { /* ... */ }     // Report responsibility
};

// Good example: separated responsibilities
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

// O - Open/Closed Principle
// Open for extension, closed for modification

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;  // No need to modify existing code when adding new shapes
};

class Rectangle : public Shape {
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
private:
    double width, height;
};

class Circle : public Shape {
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
private:
    double radius;
};

// L - Liskov Substitution Principle
// Subclasses should be substitutable for their base classes

// I - Interface Segregation Principle
// Clients should not depend on interfaces they don't use

// Bad example: bloated interface
class IBadWorker {
public:
    virtual void work() = 0;
    virtual void eat() = 0;    // Robots don't eat
    virtual void sleep() = 0;  // Robots don't sleep
};

// Good example: segregated interfaces
class IWorkable {
public:
    virtual void work() = 0;
};

class IFeedable {
public:
    virtual void eat() = 0;
};

// D - Dependency Inversion Principle
// High-level modules should not depend on low-level modules

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const std::string& message) = 0;
};

class FileLogger : public ILogger {
public:
    void log(const std::string& message) override {
        // Write log to file
    }
};

class Application {
public:
    // Depend on interface, not concrete class
    Application(std::shared_ptr<ILogger> logger) : logger_(logger) {}

    void run() {
        logger_->log("Application started");
    }
private:
    std::shared_ptr<ILogger> logger_;
};
```

---

## 2. Creational Patterns

### 2.1 Singleton

Ensures only one instance exists.

```cpp
#include <mutex>
#include <memory>

// Basic Singleton (C++11+, thread-safe)
class Singleton {
public:
    // Prevent copy/move
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;

    // Meyers' Singleton - Thread-safe guaranteed since C++11
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

    void doSomething() {
        std::cout << "Singleton doing something\n";
    }

private:
    Singleton() {
        std::cout << "Singleton created\n";
    }
    ~Singleton() {
        std::cout << "Singleton destroyed\n";
    }
};

// Template Singleton
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

// Usage
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

// Usage example
int main() {
    Singleton::getInstance().doSomething();
    Logger::getInstance().log("Hello, Singleton!");
    return 0;
}
```

### 2.2 Factory Method

Delegates object creation to subclasses.

```cpp
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

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
    void open() override {
        std::cout << "Opening PDF document\n";
    }
    void save() override {
        std::cout << "Saving PDF document\n";
    }
    std::string getType() const override { return "PDF"; }
};

class WordDocument : public Document {
public:
    void open() override {
        std::cout << "Opening Word document\n";
    }
    void save() override {
        std::cout << "Saving Word document\n";
    }
    std::string getType() const override { return "Word"; }
};

class ExcelDocument : public Document {
public:
    void open() override {
        std::cout << "Opening Excel spreadsheet\n";
    }
    void save() override {
        std::cout << "Saving Excel spreadsheet\n";
    }
    std::string getType() const override { return "Excel"; }
};

// Factory class
class DocumentFactory {
public:
    using Creator = std::function<std::unique_ptr<Document>()>;

    // Register creator
    static void registerType(const std::string& type, Creator creator) {
        getRegistry()[type] = std::move(creator);
    }

    // Create object
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
        DocumentFactory::registerType(type, []() {
            return std::make_unique<T>();
        });
    }
};

// Registration
static DocumentRegistrar<PDFDocument> pdfReg("pdf");
static DocumentRegistrar<WordDocument> wordReg("word");
static DocumentRegistrar<ExcelDocument> excelReg("excel");

// Usage example
int main() {
    auto doc1 = DocumentFactory::create("pdf");
    doc1->open();

    auto doc2 = DocumentFactory::create("word");
    doc2->open();

    return 0;
}
```

### 2.3 Builder

Builds complex objects step by step.

```cpp
#include <string>
#include <vector>
#include <optional>
#include <iostream>

// Complex object
class Computer {
public:
    void showSpecs() const {
        std::cout << "=== Computer Specs ===\n";
        std::cout << "CPU: " << cpu << "\n";
        std::cout << "RAM: " << ram << "GB\n";
        std::cout << "Storage: " << storage << "GB " << storageType << "\n";
        std::cout << "GPU: " << gpu.value_or("Integrated") << "\n";
        std::cout << "OS: " << os.value_or("None") << "\n";
    }

    // Declare Builder as friend for direct member access
    friend class ComputerBuilder;

private:
    std::string cpu;
    int ram;
    int storage;
    std::string storageType;
    std::optional<std::string> gpu;
    std::optional<std::string> os;
};

// Fluent Builder
class ComputerBuilder {
public:
    ComputerBuilder& setCPU(const std::string& cpu) {
        computer_.cpu = cpu;
        return *this;
    }

    ComputerBuilder& setRAM(int gb) {
        computer_.ram = gb;
        return *this;
    }

    ComputerBuilder& setStorage(int gb, const std::string& type = "SSD") {
        computer_.storage = gb;
        computer_.storageType = type;
        return *this;
    }

    ComputerBuilder& setGPU(const std::string& gpu) {
        computer_.gpu = gpu;
        return *this;
    }

    ComputerBuilder& setOS(const std::string& os) {
        computer_.os = os;
        return *this;
    }

    Computer build() {
        // Validation
        if (computer_.cpu.empty()) {
            throw std::runtime_error("CPU is required");
        }
        if (computer_.ram <= 0) {
            throw std::runtime_error("RAM must be positive");
        }
        return std::move(computer_);
    }

private:
    Computer computer_;
};

// Director (optional) - predefined configurations
class ComputerDirector {
public:
    static Computer buildGamingPC() {
        return ComputerBuilder()
            .setCPU("Intel i9-13900K")
            .setRAM(64)
            .setStorage(2000, "NVMe SSD")
            .setGPU("NVIDIA RTX 4090")
            .setOS("Windows 11")
            .build();
    }

    static Computer buildOfficePC() {
        return ComputerBuilder()
            .setCPU("Intel i5-13400")
            .setRAM(16)
            .setStorage(512, "SSD")
            .setOS("Windows 11")
            .build();
    }

    static Computer buildDeveloperWorkstation() {
        return ComputerBuilder()
            .setCPU("AMD Ryzen 9 7950X")
            .setRAM(128)
            .setStorage(4000, "NVMe SSD")
            .setGPU("NVIDIA RTX 4080")
            .setOS("Ubuntu 22.04")
            .build();
    }
};

// Usage example
int main() {
    // Direct build
    auto custom = ComputerBuilder()
        .setCPU("AMD Ryzen 7 7800X3D")
        .setRAM(32)
        .setStorage(1000)
        .setGPU("AMD RX 7900 XTX")
        .build();
    custom.showSpecs();

    std::cout << "\n";

    // Using Director
    auto gaming = ComputerDirector::buildGamingPC();
    gaming.showSpecs();

    return 0;
}
```

---

## 3. Structural Patterns

### 3.1 Adapter

Connects incompatible interfaces.

```cpp
#include <memory>
#include <iostream>

// Existing interface (Target)
class MediaPlayer {
public:
    virtual ~MediaPlayer() = default;
    virtual void play(const std::string& filename) = 0;
};

// Class to adapt (Adaptee)
class AdvancedMediaPlayer {
public:
    virtual ~AdvancedMediaPlayer() = default;
    virtual void playVLC(const std::string& filename) = 0;
    virtual void playMP4(const std::string& filename) = 0;
};

class VLCPlayer : public AdvancedMediaPlayer {
public:
    void playVLC(const std::string& filename) override {
        std::cout << "Playing VLC file: " << filename << "\n";
    }
    void playMP4(const std::string& filename) override {
        // Not supported
    }
};

class MP4Player : public AdvancedMediaPlayer {
public:
    void playVLC(const std::string& filename) override {
        // Not supported
    }
    void playMP4(const std::string& filename) override {
        std::cout << "Playing MP4 file: " << filename << "\n";
    }
};

// Adapter (Object Adapter)
class MediaAdapter : public MediaPlayer {
public:
    MediaAdapter(const std::string& audioType) {
        if (audioType == "vlc") {
            player_ = std::make_unique<VLCPlayer>();
        } else if (audioType == "mp4") {
            player_ = std::make_unique<MP4Player>();
        }
    }

    void play(const std::string& filename) override {
        std::string ext = getExtension(filename);
        if (ext == "vlc") {
            player_->playVLC(filename);
        } else if (ext == "mp4") {
            player_->playMP4(filename);
        }
    }

private:
    std::string getExtension(const std::string& filename) {
        size_t pos = filename.rfind('.');
        if (pos != std::string::npos) {
            return filename.substr(pos + 1);
        }
        return "";
    }

    std::unique_ptr<AdvancedMediaPlayer> player_;
};

// Unified player
class AudioPlayer : public MediaPlayer {
public:
    void play(const std::string& filename) override {
        std::string ext = getExtension(filename);

        if (ext == "mp3") {
            std::cout << "Playing MP3 file: " << filename << "\n";
        } else if (ext == "vlc" || ext == "mp4") {
            MediaAdapter adapter(ext);
            adapter.play(filename);
        } else {
            std::cout << "Unsupported format: " << ext << "\n";
        }
    }

private:
    std::string getExtension(const std::string& filename) {
        size_t pos = filename.rfind('.');
        if (pos != std::string::npos) {
            return filename.substr(pos + 1);
        }
        return "";
    }
};

// Function adapter (using lambda)
template<typename OldFunc, typename NewFunc>
class FunctionAdapter {
public:
    FunctionAdapter(OldFunc old, NewFunc adapter)
        : oldFunc_(old), adapter_(adapter) {}

    template<typename... Args>
    auto operator()(Args&&... args) {
        return adapter_(oldFunc_, std::forward<Args>(args)...);
    }

private:
    OldFunc oldFunc_;
    NewFunc adapter_;
};

// Usage example
int main() {
    AudioPlayer player;

    player.play("song.mp3");
    player.play("movie.vlc");
    player.play("video.mp4");
    player.play("image.png");

    return 0;
}
```

### 3.2 Decorator

Dynamically adds functionality to objects.

```cpp
#include <memory>
#include <iostream>
#include <string>

// Component interface
class Coffee {
public:
    virtual ~Coffee() = default;
    virtual std::string getDescription() const = 0;
    virtual double getCost() const = 0;
};

// Base implementations
class Espresso : public Coffee {
public:
    std::string getDescription() const override {
        return "Espresso";
    }
    double getCost() const override {
        return 2.00;
    }
};

class Americano : public Coffee {
public:
    std::string getDescription() const override {
        return "Americano";
    }
    double getCost() const override {
        return 2.50;
    }
};

// Decorator base class
class CoffeeDecorator : public Coffee {
public:
    explicit CoffeeDecorator(std::unique_ptr<Coffee> coffee)
        : coffee_(std::move(coffee)) {}

    std::string getDescription() const override {
        return coffee_->getDescription();
    }

    double getCost() const override {
        return coffee_->getCost();
    }

protected:
    std::unique_ptr<Coffee> coffee_;
};

// Concrete decorators
class Milk : public CoffeeDecorator {
public:
    explicit Milk(std::unique_ptr<Coffee> coffee)
        : CoffeeDecorator(std::move(coffee)) {}

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Milk";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.50;
    }
};

class Whip : public CoffeeDecorator {
public:
    explicit Whip(std::unique_ptr<Coffee> coffee)
        : CoffeeDecorator(std::move(coffee)) {}

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Whip";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.70;
    }
};

class Mocha : public CoffeeDecorator {
public:
    explicit Mocha(std::unique_ptr<Coffee> coffee)
        : CoffeeDecorator(std::move(coffee)) {}

    std::string getDescription() const override {
        return coffee_->getDescription() + ", Mocha";
    }

    double getCost() const override {
        return coffee_->getCost() + 0.80;
    }
};

// Helper function (factory style)
template<typename Decorator, typename... Args>
std::unique_ptr<Coffee> addDecorator(std::unique_ptr<Coffee> coffee, Args&&... args) {
    return std::make_unique<Decorator>(std::move(coffee), std::forward<Args>(args)...);
}

// Usage example
int main() {
    // Basic espresso
    std::unique_ptr<Coffee> coffee1 = std::make_unique<Espresso>();
    std::cout << coffee1->getDescription() << " $" << coffee1->getCost() << "\n";

    // Mocha latte
    std::unique_ptr<Coffee> coffee2 = std::make_unique<Espresso>();
    coffee2 = std::make_unique<Milk>(std::move(coffee2));
    coffee2 = std::make_unique<Mocha>(std::move(coffee2));
    std::cout << coffee2->getDescription() << " $" << coffee2->getCost() << "\n";

    // Fully loaded
    std::unique_ptr<Coffee> coffee3 = std::make_unique<Americano>();
    coffee3 = std::make_unique<Milk>(std::move(coffee3));
    coffee3 = std::make_unique<Mocha>(std::move(coffee3));
    coffee3 = std::make_unique<Whip>(std::move(coffee3));
    std::cout << coffee3->getDescription() << " $" << coffee3->getCost() << "\n";

    return 0;
}
```

### 3.3 Facade

Provides a simple interface to a complex subsystem.

```cpp
#include <iostream>
#include <memory>

// Complex subsystems
class CPU {
public:
    void freeze() { std::cout << "CPU: Freezing...\n"; }
    void jump(long position) {
        std::cout << "CPU: Jumping to " << position << "\n";
    }
    void execute() { std::cout << "CPU: Executing...\n"; }
};

class Memory {
public:
    void load(long position, const std::string& data) {
        std::cout << "Memory: Loading '" << data
                  << "' at position " << position << "\n";
    }
};

class HardDrive {
public:
    std::string read(long lba, int size) {
        std::cout << "HardDrive: Reading " << size
                  << " bytes from sector " << lba << "\n";
        return "boot_data";
    }
};

class Display {
public:
    void turnOn() { std::cout << "Display: Turning on\n"; }
    void showLogo() { std::cout << "Display: Showing boot logo\n"; }
};

// Facade - provides simple interface
class ComputerFacade {
public:
    ComputerFacade()
        : cpu_(std::make_unique<CPU>())
        , memory_(std::make_unique<Memory>())
        , hardDrive_(std::make_unique<HardDrive>())
        , display_(std::make_unique<Display>()) {}

    void start() {
        std::cout << "=== Computer Starting ===\n";
        display_->turnOn();
        display_->showLogo();
        cpu_->freeze();
        memory_->load(BOOT_ADDRESS, hardDrive_->read(BOOT_SECTOR, SECTOR_SIZE));
        cpu_->jump(BOOT_ADDRESS);
        cpu_->execute();
        std::cout << "=== Computer Ready ===\n";
    }

    void shutdown() {
        std::cout << "=== Computer Shutting Down ===\n";
        // Shutdown logic...
    }

private:
    static constexpr long BOOT_ADDRESS = 0x00000;
    static constexpr long BOOT_SECTOR = 0;
    static constexpr int SECTOR_SIZE = 512;

    std::unique_ptr<CPU> cpu_;
    std::unique_ptr<Memory> memory_;
    std::unique_ptr<HardDrive> hardDrive_;
    std::unique_ptr<Display> display_;
};

// Usage example
int main() {
    ComputerFacade computer;
    computer.start();
    // ... use ...
    computer.shutdown();
    return 0;
}
```

---

## 4. Behavioral Patterns

### 4.1 Observer

Notifies other objects of state changes.

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <string>

// Modern C++ Observer (function-based)
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
            slots_.end()
        );
    }

    void emit(Args... args) {
        for (auto& [id, slot] : slots_) {
            slot(args...);
        }
    }

    void operator()(Args... args) {
        emit(args...);
    }

private:
    std::vector<std::pair<SlotId, Slot>> slots_;
    SlotId nextId_ = 0;
};

// Usage example: Stock price monitoring
class Stock {
public:
    Stock(const std::string& symbol, double price)
        : symbol_(symbol), price_(price) {}

    const std::string& getSymbol() const { return symbol_; }
    double getPrice() const { return price_; }

    void setPrice(double price) {
        double oldPrice = price_;
        price_ = price;
        priceChanged.emit(symbol_, oldPrice, price_);
    }

    Signal<std::string, double, double> priceChanged;

private:
    std::string symbol_;
    double price_;
};

// Traditional Observer pattern (interface-based)
class IObserver {
public:
    virtual ~IObserver() = default;
    virtual void update(const std::string& message) = 0;
};

class ISubject {
public:
    virtual ~ISubject() = default;
    virtual void attach(std::shared_ptr<IObserver> observer) = 0;
    virtual void detach(std::shared_ptr<IObserver> observer) = 0;
    virtual void notify() = 0;
};

class NewsAgency : public ISubject {
public:
    void attach(std::shared_ptr<IObserver> observer) override {
        observers_.push_back(observer);
    }

    void detach(std::shared_ptr<IObserver> observer) override {
        observers_.erase(
            std::remove_if(observers_.begin(), observers_.end(),
                [&observer](const std::weak_ptr<IObserver>& wp) {
                    auto sp = wp.lock();
                    return !sp || sp == observer;
                }),
            observers_.end()
        );
    }

    void notify() override {
        for (auto it = observers_.begin(); it != observers_.end();) {
            if (auto observer = it->lock()) {
                observer->update(news_);
                ++it;
            } else {
                it = observers_.erase(it);  // Remove expired observer
            }
        }
    }

    void setNews(const std::string& news) {
        news_ = news;
        notify();
    }

private:
    std::vector<std::weak_ptr<IObserver>> observers_;
    std::string news_;
};

class NewsChannel : public IObserver {
public:
    NewsChannel(const std::string& name) : name_(name) {}

    void update(const std::string& message) override {
        std::cout << name_ << " received: " << message << "\n";
    }

private:
    std::string name_;
};

// Usage example
int main() {
    // Modern Signal/Slot approach
    Stock apple("AAPL", 150.0);

    auto id1 = apple.priceChanged.connect(
        [](const std::string& symbol, double oldPrice, double newPrice) {
            std::cout << symbol << ": $" << oldPrice << " -> $" << newPrice;
            if (newPrice > oldPrice) {
                std::cout << " (+)\n";
            } else {
                std::cout << " (-)\n";
            }
        }
    );

    apple.setPrice(155.0);
    apple.setPrice(148.0);

    std::cout << "\n--- Traditional Observer ---\n";

    // Traditional approach
    auto agency = std::make_shared<NewsAgency>();
    auto cnn = std::make_shared<NewsChannel>("CNN");
    auto bbc = std::make_shared<NewsChannel>("BBC");

    agency->attach(cnn);
    agency->attach(bbc);

    agency->setNews("Breaking: Major tech announcement!");

    return 0;
}
```

### 4.2 Strategy

Encapsulates algorithms to make them interchangeable.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>

// Strategy interface
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
    virtual std::string getName() const = 0;
};

// Concrete strategies
class BubbleSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
    std::string getName() const override { return "Bubble Sort"; }
};

class QuickSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        quickSort(data, 0, data.size() - 1);
    }
    std::string getName() const override { return "Quick Sort"; }

private:
    void quickSort(std::vector<int>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    int partition(std::vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }
};

class MergeSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        mergeSort(data, 0, data.size() - 1);
    }
    std::string getName() const override { return "Merge Sort"; }

private:
    void mergeSort(std::vector<int>& arr, int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }

    void merge(std::vector<int>& arr, int l, int m, int r) {
        std::vector<int> left(arr.begin() + l, arr.begin() + m + 1);
        std::vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

        int i = 0, j = 0, k = l;
        while (i < left.size() && j < right.size()) {
            arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
        }
        while (i < left.size()) arr[k++] = left[i++];
        while (j < right.size()) arr[k++] = right[j++];
    }
};

// Context
class Sorter {
public:
    void setStrategy(std::unique_ptr<SortStrategy> strategy) {
        strategy_ = std::move(strategy);
    }

    void sort(std::vector<int>& data) {
        if (strategy_) {
            std::cout << "Sorting with " << strategy_->getName() << "\n";
            strategy_->sort(data);
        }
    }

private:
    std::unique_ptr<SortStrategy> strategy_;
};

// Modern C++ approach: using function objects
class ModernSorter {
public:
    using Strategy = std::function<void(std::vector<int>&)>;

    void setStrategy(Strategy strategy) {
        strategy_ = std::move(strategy);
    }

    void sort(std::vector<int>& data) {
        if (strategy_) {
            strategy_(data);
        }
    }

private:
    Strategy strategy_;
};

// Usage example
int main() {
    std::vector<int> data = {64, 34, 25, 12, 22, 11, 90};

    // Traditional strategy pattern
    Sorter sorter;

    auto data1 = data;
    sorter.setStrategy(std::make_unique<BubbleSort>());
    sorter.sort(data1);

    auto data2 = data;
    sorter.setStrategy(std::make_unique<QuickSort>());
    sorter.sort(data2);

    // Modern approach - using lambda
    ModernSorter modernSorter;

    auto data3 = data;
    modernSorter.setStrategy([](std::vector<int>& v) {
        std::sort(v.begin(), v.end());  // Using STL
    });
    modernSorter.sort(data3);

    // Print result
    std::cout << "Sorted: ";
    for (int x : data3) std::cout << x << " ";
    std::cout << "\n";

    return 0;
}
```

### 4.3 Command

Encapsulates requests as objects.

```cpp
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
#include <functional>

// Receiver
class Document {
public:
    void write(const std::string& text) {
        content_ += text;
        std::cout << "Writing: '" << text << "'\n";
    }

    void erase(size_t count) {
        if (count <= content_.size()) {
            std::string erased = content_.substr(content_.size() - count);
            content_.erase(content_.size() - count);
            std::cout << "Erasing: '" << erased << "'\n";
        }
    }

    std::string getContent() const { return content_; }

    void setContent(const std::string& content) {
        content_ = content;
    }

private:
    std::string content_;
};

// Command interface
class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

// Concrete commands
class WriteCommand : public Command {
public:
    WriteCommand(Document& doc, const std::string& text)
        : doc_(doc), text_(text) {}

    void execute() override {
        doc_.write(text_);
    }

    void undo() override {
        doc_.erase(text_.size());
    }

private:
    Document& doc_;
    std::string text_;
};

class EraseCommand : public Command {
public:
    EraseCommand(Document& doc, size_t count)
        : doc_(doc), count_(count) {}

    void execute() override {
        // Save state before execution
        std::string content = doc_.getContent();
        if (count_ <= content.size()) {
            erasedText_ = content.substr(content.size() - count_);
        }
        doc_.erase(count_);
    }

    void undo() override {
        doc_.write(erasedText_);
    }

private:
    Document& doc_;
    size_t count_;
    std::string erasedText_;
};

// Invoker - supports Undo/Redo
class CommandManager {
public:
    void execute(std::unique_ptr<Command> cmd) {
        cmd->execute();
        undoStack_.push(std::move(cmd));
        // Clear redo stack when new command is executed
        while (!redoStack_.empty()) redoStack_.pop();
    }

    void undo() {
        if (!undoStack_.empty()) {
            auto cmd = std::move(undoStack_.top());
            undoStack_.pop();
            cmd->undo();
            redoStack_.push(std::move(cmd));
        }
    }

    void redo() {
        if (!redoStack_.empty()) {
            auto cmd = std::move(redoStack_.top());
            redoStack_.pop();
            cmd->execute();
            undoStack_.push(std::move(cmd));
        }
    }

    bool canUndo() const { return !undoStack_.empty(); }
    bool canRedo() const { return !redoStack_.empty(); }

private:
    std::stack<std::unique_ptr<Command>> undoStack_;
    std::stack<std::unique_ptr<Command>> redoStack_;
};

// Macro command (composite command)
class MacroCommand : public Command {
public:
    void addCommand(std::unique_ptr<Command> cmd) {
        commands_.push_back(std::move(cmd));
    }

    void execute() override {
        for (auto& cmd : commands_) {
            cmd->execute();
        }
    }

    void undo() override {
        for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
            (*it)->undo();
        }
    }

private:
    std::vector<std::unique_ptr<Command>> commands_;
};

// Usage example
int main() {
    Document doc;
    CommandManager manager;

    // Execute commands
    manager.execute(std::make_unique<WriteCommand>(doc, "Hello"));
    manager.execute(std::make_unique<WriteCommand>(doc, " World"));
    manager.execute(std::make_unique<WriteCommand>(doc, "!"));

    std::cout << "Content: " << doc.getContent() << "\n\n";

    // Undo
    std::cout << "--- Undo ---\n";
    manager.undo();
    std::cout << "Content: " << doc.getContent() << "\n\n";

    manager.undo();
    std::cout << "Content: " << doc.getContent() << "\n\n";

    // Redo
    std::cout << "--- Redo ---\n";
    manager.redo();
    std::cout << "Content: " << doc.getContent() << "\n\n";

    return 0;
}
```

### 4.4 Template Method

Defines the skeleton of an algorithm and lets subclasses implement some steps.

```cpp
#include <iostream>
#include <string>
#include <fstream>

// Abstract class - defines template method
class DataParser {
public:
    virtual ~DataParser() = default;

    // Template method - algorithm skeleton
    void parseFile(const std::string& filename) {
        std::cout << "=== Parsing " << filename << " ===\n";

        openFile(filename);
        extractData();
        parseData();
        analyzeData();
        closeFile();

        std::cout << "=== Done ===\n\n";
    }

protected:
    // Default implementation
    virtual void openFile(const std::string& filename) {
        std::cout << "Opening file: " << filename << "\n";
    }

    virtual void closeFile() {
        std::cout << "Closing file\n";
    }

    // Pure virtual functions - must be implemented by subclasses
    virtual void extractData() = 0;
    virtual void parseData() = 0;

    // Hook method - optionally override
    virtual void analyzeData() {
        // Default: do nothing
    }
};

// Concrete classes
class CSVParser : public DataParser {
protected:
    void extractData() override {
        std::cout << "Extracting CSV rows and columns\n";
    }

    void parseData() override {
        std::cout << "Parsing CSV: splitting by commas\n";
    }

    void analyzeData() override {
        std::cout << "CSV Analysis: counting rows and columns\n";
    }
};

class JSONParser : public DataParser {
protected:
    void extractData() override {
        std::cout << "Extracting JSON objects and arrays\n";
    }

    void parseData() override {
        std::cout << "Parsing JSON: building object tree\n";
    }

    void analyzeData() override {
        std::cout << "JSON Analysis: validating schema\n";
    }
};

class XMLParser : public DataParser {
protected:
    void extractData() override {
        std::cout << "Extracting XML elements and attributes\n";
    }

    void parseData() override {
        std::cout << "Parsing XML: building DOM tree\n";
    }
    // Not using analyzeData() (uses default implementation)
};

// Usage example
int main() {
    CSVParser csvParser;
    csvParser.parseFile("data.csv");

    JSONParser jsonParser;
    jsonParser.parseFile("data.json");

    XMLParser xmlParser;
    xmlParser.parseFile("config.xml");

    return 0;
}
```

---

## 5. C++-Specific Idioms

### 5.1 CRTP (Curiously Recurring Template Pattern)

A technique for implementing static polymorphism.

```cpp
#include <iostream>
#include <memory>
#include <vector>

// Basic CRTP form
template<typename Derived>
class Base {
public:
    void interface() {
        // Call derived class implementation (static polymorphism)
        static_cast<Derived*>(this)->implementation();
    }

    // Default implementation
    void implementation() {
        std::cout << "Base implementation\n";
    }
};

class Derived1 : public Base<Derived1> {
public:
    void implementation() {
        std::cout << "Derived1 implementation\n";
    }
};

class Derived2 : public Base<Derived2> {
public:
    void implementation() {
        std::cout << "Derived2 implementation\n";
    }
};

// CRTP Application 1: Mixin
template<typename Derived>
class Counter {
public:
    static int getCount() { return count_; }

protected:
    Counter() { ++count_; }
    ~Counter() { --count_; }

private:
    static inline int count_ = 0;
};

class Widget : public Counter<Widget> {
public:
    Widget(int id) : id_(id) {}
private:
    int id_;
};

class Gadget : public Counter<Gadget> {
public:
    Gadget(const std::string& name) : name_(name) {}
private:
    std::string name_;
};

// CRTP Application 2: Static Interface (Static Polymorphism)
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
public:
    Circle(double r) : radius_(r) {}

    double areaImpl() const {
        return 3.14159 * radius_ * radius_;
    }

    void drawImpl() const {
        std::cout << "Drawing circle with radius " << radius_ << "\n";
    }

private:
    double radius_;
};

class Rectangle : public Shape<Rectangle> {
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}

    double areaImpl() const {
        return width_ * height_;
    }

    void drawImpl() const {
        std::cout << "Drawing rectangle " << width_ << "x" << height_ << "\n";
    }

private:
    double width_, height_;
};

// Template function utilizing static polymorphism
template<typename T>
void printArea(const Shape<T>& shape) {
    std::cout << "Area: " << shape.area() << "\n";
}

// CRTP Application 3: Fluent Interface / Method Chaining
template<typename Derived>
class Builder {
public:
    Derived& setName(const std::string& name) {
        name_ = name;
        return static_cast<Derived&>(*this);
    }

protected:
    std::string name_;
};

class PersonBuilder : public Builder<PersonBuilder> {
public:
    PersonBuilder& setAge(int age) {
        age_ = age;
        return *this;
    }

    PersonBuilder& setCity(const std::string& city) {
        city_ = city;
        return *this;
    }

    void build() {
        std::cout << "Person: " << name_ << ", " << age_ << ", " << city_ << "\n";
    }

private:
    int age_ = 0;
    std::string city_;
};

// Usage example
int main() {
    // Basic CRTP
    Derived1 d1;
    Derived2 d2;
    d1.interface();
    d2.interface();

    std::cout << "\n--- Counter Mixin ---\n";
    {
        Widget w1(1), w2(2), w3(3);
        Gadget g1("A"), g2("B");

        std::cout << "Widgets: " << Widget::getCount() << "\n";
        std::cout << "Gadgets: " << Gadget::getCount() << "\n";
    }
    std::cout << "After scope - Widgets: " << Widget::getCount() << "\n";

    std::cout << "\n--- Static Polymorphism ---\n";
    Circle circle(5.0);
    Rectangle rect(3.0, 4.0);

    printArea(circle);
    printArea(rect);

    std::cout << "\n--- Fluent Builder ---\n";
    PersonBuilder()
        .setName("John")
        .setAge(30)
        .setCity("Seoul")
        .build();

    return 0;
}
```

### 5.2 PIMPL (Pointer to Implementation)

A technique for hiding implementation from headers.

```cpp
// === widget.h ===
#ifndef WIDGET_H
#define WIDGET_H

#include <memory>
#include <string>

class Widget {
public:
    Widget();
    ~Widget();

    // Move operations
    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;

    // Copy operations (if needed)
    Widget(const Widget& other);
    Widget& operator=(const Widget& other);

    // Public interface
    void setName(const std::string& name);
    std::string getName() const;
    void doSomething();

private:
    // Forward declaration only - hides implementation
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

#endif

// === widget.cpp ===
// #include "widget.h"
#include <iostream>

// Implementation class definition
class Widget::Impl {
public:
    std::string name;
    int internalData = 0;

    void internalProcess() {
        std::cout << "Internal processing for: " << name << "\n";
        internalData++;
    }
};

// Widget implementation
Widget::Widget() : pImpl_(std::make_unique<Impl>()) {}

Widget::~Widget() = default;

Widget::Widget(Widget&& other) noexcept = default;
Widget& Widget::operator=(Widget&& other) noexcept = default;

Widget::Widget(const Widget& other)
    : pImpl_(std::make_unique<Impl>(*other.pImpl_)) {}

Widget& Widget::operator=(const Widget& other) {
    if (this != &other) {
        pImpl_ = std::make_unique<Impl>(*other.pImpl_);
    }
    return *this;
}

void Widget::setName(const std::string& name) {
    pImpl_->name = name;
}

std::string Widget::getName() const {
    return pImpl_->name;
}

void Widget::doSomething() {
    pImpl_->internalProcess();
}

// === main.cpp ===
int main() {
    Widget w;
    w.setName("MyWidget");
    w.doSomething();
    std::cout << "Widget name: " << w.getName() << "\n";

    // Move
    Widget w2 = std::move(w);
    w2.doSomething();

    return 0;
}
```

### 5.3 Type Erasure

Handle different types through a single interface.

```cpp
#include <iostream>
#include <memory>
#include <vector>

// Type Erasure pattern implementation
class Drawable {
public:
    template<typename T>
    Drawable(T obj) : pImpl_(std::make_shared<Model<T>>(std::move(obj))) {}

    void draw() const {
        pImpl_->draw();
    }

private:
    // Concept - interface
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
    };

    // Model - wraps concrete type
    template<typename T>
    struct Model : Concept {
        Model(T obj) : object_(std::move(obj)) {}

        void draw() const override {
            object_.draw();  // T must have a draw() method
        }

        T object_;
    };

    std::shared_ptr<const Concept> pImpl_;
};

// Various types with draw() method
class Circle {
public:
    Circle(double r) : radius_(r) {}
    void draw() const {
        std::cout << "Circle with radius " << radius_ << "\n";
    }
private:
    double radius_;
};

class Square {
public:
    Square(double s) : side_(s) {}
    void draw() const {
        std::cout << "Square with side " << side_ << "\n";
    }
private:
    double side_;
};

class Text {
public:
    Text(std::string t) : text_(std::move(t)) {}
    void draw() const {
        std::cout << "Text: " << text_ << "\n";
    }
private:
    std::string text_;
};

// std::function is also an example of Type Erasure
// std::any is also an example of Type Erasure

// Usage example
int main() {
    // Store various types in one container
    std::vector<Drawable> shapes;

    shapes.emplace_back(Circle(5.0));
    shapes.emplace_back(Square(3.0));
    shapes.emplace_back(Text("Hello"));

    // Handle through same interface
    for (const auto& shape : shapes) {
        shape.draw();
    }

    return 0;
}
```

### 5.4 RAII (Resource Acquisition Is Initialization)

Links resource management to constructor/destructor.

```cpp
#include <iostream>
#include <fstream>
#include <mutex>
#include <memory>

// File handle RAII wrapper
class File {
public:
    explicit File(const std::string& filename, const std::string& mode = "r") {
        file_ = fopen(filename.c_str(), mode.c_str());
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::cout << "File opened: " << filename << "\n";
    }

    ~File() {
        if (file_) {
            fclose(file_);
            std::cout << "File closed\n";
        }
    }

    // Prevent copy
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    // Allow move
    File(File&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }

    File& operator=(File&& other) noexcept {
        if (this != &other) {
            if (file_) fclose(file_);
            file_ = other.file_;
            other.file_ = nullptr;
        }
        return *this;
    }

    FILE* get() const { return file_; }

private:
    FILE* file_ = nullptr;
};

// Lock guard (similar to std::lock_guard)
template<typename Mutex>
class LockGuard {
public:
    explicit LockGuard(Mutex& mutex) : mutex_(mutex) {
        mutex_.lock();
    }

    ~LockGuard() {
        mutex_.unlock();
    }

    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    Mutex& mutex_;
};

// Scope guard (generic cleanup)
template<typename Func>
class ScopeGuard {
public:
    explicit ScopeGuard(Func&& func)
        : func_(std::forward<Func>(func)), active_(true) {}

    ~ScopeGuard() {
        if (active_) {
            func_();
        }
    }

    void dismiss() { active_ = false; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ScopeGuard(ScopeGuard&& other) noexcept
        : func_(std::move(other.func_)), active_(other.active_) {
        other.active_ = false;
    }

private:
    Func func_;
    bool active_;
};

// Helper function
template<typename Func>
ScopeGuard<Func> makeScopeGuard(Func&& func) {
    return ScopeGuard<Func>(std::forward<Func>(func));
}

// Usage example
int main() {
    // RAII file
    try {
        File file("/tmp/test.txt", "w");
        fprintf(file.get(), "Hello RAII!\n");
        // File automatically closed when scope ends
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // Scope guard
    std::cout << "\n--- Scope Guard ---\n";
    {
        auto guard = makeScopeGuard([]() {
            std::cout << "Cleanup executed!\n";
        });

        std::cout << "Doing work...\n";
        // Cleanup executed when scope ends
    }

    // Using dismiss
    {
        auto guard = makeScopeGuard([]() {
            std::cout << "This won't run\n";
        });

        guard.dismiss();  // Cancel cleanup
    }

    return 0;
}
```

---

## 6. Practice Problems

### Exercise 1: Logging System (Singleton + Strategy)
```cpp
// Requirements:
// 1. Implement a singleton Logger
// 2. Various output strategies (Console, File, Network)
// 3. Log levels (DEBUG, INFO, WARNING, ERROR)

// Hint:
class ILogStrategy {
public:
    virtual void write(const std::string& message) = 0;
};

class Logger {
public:
    static Logger& getInstance();
    void setStrategy(std::unique_ptr<ILogStrategy> strategy);
    void log(LogLevel level, const std::string& message);
};
```

### Exercise 2: UI Components (Composite + Decorator)
```cpp
// Requirements:
// 1. UI component hierarchy (Window, Panel, Button)
// 2. Add borders and scrollbars via decorators
// 3. Implement containers via composite

// Hint:
class UIComponent {
public:
    virtual void render() = 0;
    virtual void add(std::shared_ptr<UIComponent> component) {}
};
```

### Exercise 3: Document Editor (Command + Memento)
```cpp
// Requirements:
// 1. Implement text insert, delete, format commands
// 2. Support unlimited Undo/Redo
// 3. Snapshot save/restore (Memento)

// Hint:
class Memento {
    friend class Document;
    std::string state_;
};

class Document {
public:
    Memento createMemento();
    void restore(const Memento& memento);
};
```

### Exercise 4: Plugin System (Factory + Type Erasure)
```cpp
// Requirements:
// 1. Register/unregister plugins at runtime
// 2. Support various plugin types
// 3. Execute through common interface

// Hint:
class Plugin {
public:
    template<typename T>
    Plugin(T plugin);
    void execute();
};

class PluginManager {
    std::unordered_map<std::string, Plugin> plugins_;
};
```

---

## Next Steps

- [15_C++17_Features](15_C++17_Features.md) - Review modern C++ features
- Refer to advanced design pattern books (GoF, Head First)
- Practice applying patterns in real projects

## References

- "Design Patterns: Elements of Reusable Object-Oriented Software" (GoF)
- "Modern C++ Design" - Andrei Alexandrescu
- [Refactoring.Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

---

**Previous**: [Advanced C++20](./17_CPP20_Advanced.md) | **Next**: [Project: Student Management System](./19_Project_Student_Management.md)
