# 디자인 패턴: 생성 및 구조

**이전**: [고급 동시성](./13_Concurrency_Advanced.md) | **다음**: [디자인 패턴: 행위 및 C++ 이디엄](./15_Design_Patterns_Behavioral_Idioms.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. SOLID 원칙과 디자인 패턴을 효과적으로 적용하기 위한 기반 역할을 설명할 수 있다
2. 마이어스 싱글톤(Meyers' Singleton)과 템플릿 기반 접근법을 사용하여 싱글톤 패턴을 구현할 수 있다
3. 팩토리 메서드와 추상 팩토리 패턴을 적용하여 객체 생성과 사용을 분리할 수 있다
4. 플루언트 인터페이스와 선택적 디렉터 클래스를 사용한 빌더 패턴으로 복잡한 객체를 구성할 수 있다
5. 다형적 복제를 위한 프로토타입 패턴을 구현할 수 있다
6. 구조 패턴(어댑터, 데코레이터, 파사드, 컴포지트, 프록시)을 적용하여 유연한 객체 계층을 구성할 수 있다
7. 디자인 패턴이 가치를 더하는 경우와 더 간단한 대안이 충분한 경우를 평가할 수 있다

---

디자인 패턴은 학술적 연습이 아닙니다 -- 모든 비사소적(non-trivial) C++ 프로젝트가 마주치는 문제에 대한 실전 검증된 해결책입니다. 팩토리가 객체 생성을 단순화하는 시점, 데코레이터가 서브클래싱 없이 동작을 추가하는 시점, 파사드가 복잡한 하위 시스템을 길들이는 시점을 아는 것은 과도한 설계 없이 유연한 아키텍처를 작성할 수 있게 합니다. 스마트 포인터, 람다, 템플릿 같은 현대 C++ 기능과 결합하면 이 패턴들은 교과서적 형태보다 더 가볍고 표현력이 풍부해집니다.

---

## 목차

1. [디자인 패턴 개요](#1-디자인-패턴-개요)
2. [SOLID 원칙](#2-solid-원칙)
3. [싱글톤](#3-싱글톤)
4. [팩토리 메서드](#4-팩토리-메서드)
5. [추상 팩토리](#5-추상-팩토리)
6. [빌더](#6-빌더)
7. [프로토타입](#7-프로토타입)
8. [어댑터](#8-어댑터)
9. [데코레이터](#9-데코레이터)
10. [파사드](#10-파사드)
11. [컴포지트](#11-컴포지트)
12. [프록시](#12-프록시)

---

## 1. 디자인 패턴 개요

### 분류

| 생성 패턴 | 구조 패턴 | 행위 패턴 |
|----------|----------|----------|
| 싱글톤 (Singleton) | 어댑터 (Adapter) | 옵저버 (Observer) |
| 팩토리 메서드 (Factory Method) | 데코레이터 (Decorator) | 전략 (Strategy) |
| 추상 팩토리 (Abstract Factory) | 파사드 (Facade) | 커맨드 (Command) |
| 빌더 (Builder) | 컴포지트 (Composite) | 상태 (State) |
| 프로토타입 (Prototype) | 프록시 (Proxy) | 반복자 (Iterator) |
| | 브릿지 (Bridge) | 템플릿 메서드 (Template Method) |
| | 플라이웨이트 (Flyweight) | |

이 레슨은 생성 패턴과 구조 패턴을 다룹니다. 행위 패턴과 C++ 이디엄은 다음 레슨에서 다룹니다.

---

## 2. SOLID 원칙

```cpp
// S - 단일 책임 원칙 (Single Responsibility Principle)
// 클래스는 변경할 이유가 하나만 있어야 합니다.

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
// O - 개방-폐쇄 원칙 (Open/Closed Principle)
// 확장에는 열려 있고, 수정에는 닫혀 있어야 합니다.

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

// Circle 추가가 Shape이나 Rectangle을 수정하지 않음
class Circle : public Shape {
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
private:
    double radius;
};
```

```cpp
// L - 리스코프 치환 원칙 (Liskov Substitution Principle)
// 하위 타입은 기반 타입을 대체할 수 있어야 합니다.

// I - 인터페이스 분리 원칙 (Interface Segregation Principle)
// 클라이언트는 사용하지 않는 인터페이스에 의존하지 않아야 합니다.

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

// D - 의존성 역전 원칙 (Dependency Inversion Principle)
// 구체가 아닌 추상에 의존해야 합니다.

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

## 3. 싱글톤

인스턴스가 하나만 존재하도록 보장합니다.

```cpp
#include <mutex>
#include <memory>
#include <iostream>

// Meyers' Singleton — C++11 이후 스레드 안전
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

### 템플릿 싱글톤

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

## 4. 팩토리 메서드

객체 생성을 하위 클래스나 레지스트리에 위임합니다.

```cpp
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <iostream>

// 제품 인터페이스
class Document {
public:
    virtual ~Document() = default;
    virtual void open() = 0;
    virtual void save() = 0;
    virtual std::string getType() const = 0;
};

// 구체 제품
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

// 자기 등록 팩토리
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

// 자동 등록 헬퍼
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

## 5. 추상 팩토리

구체 클래스를 지정하지 않고 관련 객체의 패밀리를 생성합니다.

```cpp
#include <memory>
#include <iostream>

// 추상 제품
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

// 구체 제품: 다크 테마
class DarkButton : public Button {
public:
    void render() override { std::cout << "[Dark Button]\n"; }
};

class DarkTextBox : public TextBox {
public:
    void render() override { std::cout << "[Dark TextBox]\n"; }
};

// 구체 제품: 라이트 테마
class LightButton : public Button {
public:
    void render() override { std::cout << "[Light Button]\n"; }
};

class LightTextBox : public TextBox {
public:
    void render() override { std::cout << "[Light TextBox]\n"; }
};

// 추상 팩토리
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

// 클라이언트 코드 — 추상 팩토리에만 의존
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

## 6. 빌더

플루언트 인터페이스를 사용하여 복잡한 객체를 단계별로 구축합니다.

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

// 디렉터: 미리 정의된 구성
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

## 7. 프로토타입

기존 객체를 복제하여 새 객체를 생성합니다. 생성이 비용이 크거나 다형적 복사가 필요할 때 유용합니다.

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

// 프로토타입 레지스트리
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

## 8. 어댑터

호환되지 않는 인터페이스를 연결합니다.

```cpp
#include <memory>
#include <iostream>

// 대상 인터페이스 (클라이언트가 기대하는 것)
class MediaPlayer {
public:
    virtual ~MediaPlayer() = default;
    virtual void play(const std::string& filename) = 0;
};

// 피적응자 (호환되지 않는 인터페이스의 기존 클래스)
class AdvancedPlayer {
public:
    void playMP4(const std::string& f) {
        std::cout << "Playing MP4: " << f << "\n";
    }
    void playMKV(const std::string& f) {
        std::cout << "Playing MKV: " << f << "\n";
    }
};

// 어댑터 (피적응자를 래핑하여 대상에 맞춤)
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

## 9. 데코레이터

클래스를 수정하지 않고 동적으로 객체에 기능을 추가합니다.

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

## 10. 파사드

복잡한 하위 시스템에 단순화된 인터페이스를 제공합니다.

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

// 파사드
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

## 11. 컴포지트

트리 구조를 통해 개별 객체와 합성 객체를 균일하게 처리합니다.

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

## 12. 프록시

다른 객체에 대한 접근을 제어합니다.

```cpp
#include <memory>
#include <iostream>
#include <unordered_map>

class Image {
public:
    virtual ~Image() = default;
    virtual void display() = 0;
};

// 실제 주체 — 생성 비용이 큼
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

// 프록시 — 지연 로딩
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

// 캐싱 프록시
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
    // 이미지가 아직 로드되지 않음

    img.display();  // 로드 후 표시
    img.display();  // 표시만 (이미 로드됨)

    return 0;
}
```

---

## 연습 문제

### 연습 1: 팩토리를 이용한 플러그인 시스템

팩토리 메서드 패턴을 사용하여 플러그인 시스템을 설계하세요. `name()`, `version()`, `execute()`를 가진 `IPlugin` 인터페이스를 정의하세요. 자기 등록을 가진 `PluginFactory`를 만드세요. 최소 세 개의 플러그인을 구현하고 명령줄 인수에서 이름으로 로드하는 메인 프로그램을 작성하세요.

### 연습 2: 데코레이터를 이용한 로거

데코레이터 패턴을 사용하여 로깅 시스템을 구현하세요. 기본 `Logger`는 일반 텍스트를 작성합니다. 데코레이터가 추가하는 것: 타임스탬프 접두사, 로그 레벨(INFO, WARN, ERROR), 파일 출력. 데코레이터를 임의의 순서로 결합할 수 있음을 보이세요.

### 연습 3: 컴포지트 표현식 트리

컴포지트 패턴을 사용하여 표현식 트리를 구축하세요. `Leaf` 노드는 숫자 값을 갖습니다. `BinaryOp` 노드는 연산자(+, -, *, /)와 두 자식을 갖습니다. 양쪽에 `evaluate()`와 `toString()`을 구현하세요. "3 + 4 * 2"를 트리로 파싱하고 평가하세요.

### 연습 4: 스마트 프록시

모든 객체를 래핑하고 모든 메서드 호출(메서드 이름, 인수, 반환 값)을 로깅하는 `LoggingProxy<T>` 템플릿을 구현하세요. `operator->` 오버로딩을 사용하여 투명하게 호출을 가로채세요.

### 연습 5: 데이터베이스 계층을 위한 추상 팩토리

데이터베이스 접근을 위한 추상 팩토리를 설계하세요. 팩토리가 `Connection`, `Command`, `ResultSet` 객체를 생성합니다. 두 개의 구체 패밀리를 구현하세요: `SQLite`와 `Mock` (인메모리). 수정 없이 어떤 패밀리와도 동작하는 클라이언트 코드를 작성하세요.

---

## 다음 단계

이 레슨은 생성 패턴과 구조 패턴을 다루었습니다. 다음 레슨은 행위 패턴(옵저버, 전략, 커맨드, 상태, 템플릿 메서드, 반복자)과 C++ 이디엄(CRTP, PIMPL, 타입 소거, NVI)으로 그림을 완성합니다.

- [디자인 패턴: 행위 및 C++ 이디엄](./15_Design_Patterns_Behavioral_Idioms.md)
