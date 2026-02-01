# 18. C++ 디자인 패턴

## 학습 목표
- GoF 디자인 패턴의 핵심 패턴 이해
- 모던 C++를 활용한 패턴 구현
- CRTP, PIMPL 등 C++ 특화 이디엄 습득
- 패턴 선택 기준과 적용 사례 학습

## 목차
1. [디자인 패턴 개요](#1-디자인-패턴-개요)
2. [생성 패턴](#2-생성-패턴)
3. [구조 패턴](#3-구조-패턴)
4. [행동 패턴](#4-행동-패턴)
5. [C++ 특화 이디엄](#5-c-특화-이디엄)
6. [연습 문제](#6-연습-문제)

---

## 1. 디자인 패턴 개요

### 1.1 디자인 패턴이란?

디자인 패턴은 소프트웨어 설계에서 자주 발생하는 문제에 대한 재사용 가능한 해결책입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    GoF 디자인 패턴 분류                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  생성 패턴   │  │  구조 패턴   │  │  행동 패턴   │         │
│  │ Creational  │  │ Structural  │  │ Behavioral  │         │
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

### 1.2 SOLID 원칙

```cpp
// SOLID 원칙 - 좋은 객체지향 설계의 기초

// S - Single Responsibility Principle (단일 책임 원칙)
// 클래스는 하나의 책임만 가져야 함

// 나쁜 예: 여러 책임이 혼재
class BadUserManager {
public:
    void createUser(const std::string& name) { /* ... */ }
    void saveToDatabase() { /* ... */ }     // DB 책임
    void sendEmail() { /* ... */ }          // 이메일 책임
    void generateReport() { /* ... */ }     // 리포트 책임
};

// 좋은 예: 책임 분리
class User {
public:
    User(const std::string& name) : name_(name) {}
    std::string getName() const { return name_; }
private:
    std::string name_;
};

class UserRepository {
public:
    void save(const User& user) { /* DB 저장 */ }
};

class EmailService {
public:
    void sendWelcome(const User& user) { /* 이메일 발송 */ }
};

// O - Open/Closed Principle (개방-폐쇄 원칙)
// 확장에는 열려있고, 수정에는 닫혀있어야 함

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;  // 새 도형 추가 시 기존 코드 수정 불필요
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

// L - Liskov Substitution Principle (리스코프 치환 원칙)
// 자식 클래스는 부모 클래스를 대체할 수 있어야 함

// I - Interface Segregation Principle (인터페이스 분리 원칙)
// 클라이언트는 사용하지 않는 인터페이스에 의존하면 안 됨

// 나쁜 예: 거대한 인터페이스
class IBadWorker {
public:
    virtual void work() = 0;
    virtual void eat() = 0;    // 로봇은 먹지 않음
    virtual void sleep() = 0;  // 로봇은 자지 않음
};

// 좋은 예: 분리된 인터페이스
class IWorkable {
public:
    virtual void work() = 0;
};

class IFeedable {
public:
    virtual void eat() = 0;
};

// D - Dependency Inversion Principle (의존성 역전 원칙)
// 고수준 모듈이 저수준 모듈에 의존하면 안 됨

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const std::string& message) = 0;
};

class FileLogger : public ILogger {
public:
    void log(const std::string& message) override {
        // 파일에 로그 기록
    }
};

class Application {
public:
    // 구체 클래스가 아닌 인터페이스에 의존
    Application(std::shared_ptr<ILogger> logger) : logger_(logger) {}

    void run() {
        logger_->log("Application started");
    }
private:
    std::shared_ptr<ILogger> logger_;
};
```

---

## 2. 생성 패턴

### 2.1 Singleton (싱글톤)

인스턴스가 하나만 존재하도록 보장합니다.

```cpp
#include <mutex>
#include <memory>

// 기본 싱글톤 (C++11 이상, 스레드 안전)
class Singleton {
public:
    // 복사/이동 방지
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;

    // Meyers' Singleton - C++11부터 스레드 안전 보장
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

// 템플릿 싱글톤
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

// 사용
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

// 사용 예
int main() {
    Singleton::getInstance().doSomething();
    Logger::getInstance().log("Hello, Singleton!");
    return 0;
}
```

### 2.2 Factory Method (팩토리 메서드)

객체 생성을 서브클래스에 위임합니다.

```cpp
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

// 제품 인터페이스
class Document {
public:
    virtual ~Document() = default;
    virtual void open() = 0;
    virtual void save() = 0;
    virtual std::string getType() const = 0;
};

// 구체 제품들
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

// 팩토리 클래스
class DocumentFactory {
public:
    using Creator = std::function<std::unique_ptr<Document>()>;

    // 생성자 등록
    static void registerType(const std::string& type, Creator creator) {
        getRegistry()[type] = std::move(creator);
    }

    // 객체 생성
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
        DocumentFactory::registerType(type, []() {
            return std::make_unique<T>();
        });
    }
};

// 등록
static DocumentRegistrar<PDFDocument> pdfReg("pdf");
static DocumentRegistrar<WordDocument> wordReg("word");
static DocumentRegistrar<ExcelDocument> excelReg("excel");

// 사용 예
int main() {
    auto doc1 = DocumentFactory::create("pdf");
    doc1->open();

    auto doc2 = DocumentFactory::create("word");
    doc2->open();

    return 0;
}
```

### 2.3 Builder (빌더)

복잡한 객체를 단계별로 생성합니다.

```cpp
#include <string>
#include <vector>
#include <optional>
#include <iostream>

// 복잡한 객체
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

    // Builder가 직접 멤버에 접근하도록 friend 선언
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
        // 유효성 검사
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

// Director (선택적) - 미리 정의된 구성
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

// 사용 예
int main() {
    // 직접 빌드
    auto custom = ComputerBuilder()
        .setCPU("AMD Ryzen 7 7800X3D")
        .setRAM(32)
        .setStorage(1000)
        .setGPU("AMD RX 7900 XTX")
        .build();
    custom.showSpecs();

    std::cout << "\n";

    // Director 사용
    auto gaming = ComputerDirector::buildGamingPC();
    gaming.showSpecs();

    return 0;
}
```

---

## 3. 구조 패턴

### 3.1 Adapter (어댑터)

호환되지 않는 인터페이스를 연결합니다.

```cpp
#include <memory>
#include <iostream>

// 기존 인터페이스 (Target)
class MediaPlayer {
public:
    virtual ~MediaPlayer() = default;
    virtual void play(const std::string& filename) = 0;
};

// 적응해야 할 클래스 (Adaptee)
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
        // 지원하지 않음
    }
};

class MP4Player : public AdvancedMediaPlayer {
public:
    void playVLC(const std::string& filename) override {
        // 지원하지 않음
    }
    void playMP4(const std::string& filename) override {
        std::cout << "Playing MP4 file: " << filename << "\n";
    }
};

// 어댑터 (Object Adapter)
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

// 통합 플레이어
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

// 함수 어댑터 (람다 사용)
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

// 사용 예
int main() {
    AudioPlayer player;

    player.play("song.mp3");
    player.play("movie.vlc");
    player.play("video.mp4");
    player.play("image.png");

    return 0;
}
```

### 3.2 Decorator (데코레이터)

객체에 동적으로 기능을 추가합니다.

```cpp
#include <memory>
#include <iostream>
#include <string>

// 컴포넌트 인터페이스
class Coffee {
public:
    virtual ~Coffee() = default;
    virtual std::string getDescription() const = 0;
    virtual double getCost() const = 0;
};

// 기본 구현
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

// 데코레이터 기본 클래스
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

// 구체 데코레이터들
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

// 헬퍼 함수 (팩토리 스타일)
template<typename Decorator, typename... Args>
std::unique_ptr<Coffee> addDecorator(std::unique_ptr<Coffee> coffee, Args&&... args) {
    return std::make_unique<Decorator>(std::move(coffee), std::forward<Args>(args)...);
}

// 사용 예
int main() {
    // 기본 에스프레소
    std::unique_ptr<Coffee> coffee1 = std::make_unique<Espresso>();
    std::cout << coffee1->getDescription() << " $" << coffee1->getCost() << "\n";

    // 모카 라떼
    std::unique_ptr<Coffee> coffee2 = std::make_unique<Espresso>();
    coffee2 = std::make_unique<Milk>(std::move(coffee2));
    coffee2 = std::make_unique<Mocha>(std::move(coffee2));
    std::cout << coffee2->getDescription() << " $" << coffee2->getCost() << "\n";

    // 풀옵션
    std::unique_ptr<Coffee> coffee3 = std::make_unique<Americano>();
    coffee3 = std::make_unique<Milk>(std::move(coffee3));
    coffee3 = std::make_unique<Mocha>(std::move(coffee3));
    coffee3 = std::make_unique<Whip>(std::move(coffee3));
    std::cout << coffee3->getDescription() << " $" << coffee3->getCost() << "\n";

    return 0;
}
```

### 3.3 Facade (퍼사드)

복잡한 서브시스템에 단순한 인터페이스를 제공합니다.

```cpp
#include <iostream>
#include <memory>

// 복잡한 서브시스템들
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

// 퍼사드 - 간단한 인터페이스 제공
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
        // 종료 로직...
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

// 사용 예
int main() {
    ComputerFacade computer;
    computer.start();
    // ... 사용 ...
    computer.shutdown();
    return 0;
}
```

---

## 4. 행동 패턴

### 4.1 Observer (옵저버)

객체 상태 변화를 다른 객체에 통지합니다.

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <string>

// 모던 C++ 옵저버 (함수 기반)
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

// 사용 예: 주식 가격 모니터링
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

// 전통적인 옵저버 패턴 (인터페이스 기반)
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
                it = observers_.erase(it);  // 만료된 옵저버 제거
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

// 사용 예
int main() {
    // 모던 Signal/Slot 방식
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

    // 전통적인 방식
    auto agency = std::make_shared<NewsAgency>();
    auto cnn = std::make_shared<NewsChannel>("CNN");
    auto bbc = std::make_shared<NewsChannel>("BBC");

    agency->attach(cnn);
    agency->attach(bbc);

    agency->setNews("Breaking: Major tech announcement!");

    return 0;
}
```

### 4.2 Strategy (전략)

알고리즘을 캡슐화하여 교체 가능하게 만듭니다.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>

// 전략 인터페이스
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
    virtual std::string getName() const = 0;
};

// 구체 전략들
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

// 컨텍스트
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

// 모던 C++ 방식: 함수 객체 사용
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

// 사용 예
int main() {
    std::vector<int> data = {64, 34, 25, 12, 22, 11, 90};

    // 전통적인 전략 패턴
    Sorter sorter;

    auto data1 = data;
    sorter.setStrategy(std::make_unique<BubbleSort>());
    sorter.sort(data1);

    auto data2 = data;
    sorter.setStrategy(std::make_unique<QuickSort>());
    sorter.sort(data2);

    // 모던 방식 - 람다 사용
    ModernSorter modernSorter;

    auto data3 = data;
    modernSorter.setStrategy([](std::vector<int>& v) {
        std::sort(v.begin(), v.end());  // STL 사용
    });
    modernSorter.sort(data3);

    // 결과 출력
    std::cout << "Sorted: ";
    for (int x : data3) std::cout << x << " ";
    std::cout << "\n";

    return 0;
}
```

### 4.3 Command (커맨드)

요청을 객체로 캡슐화합니다.

```cpp
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
#include <functional>

// 수신자 (Receiver)
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

// 커맨드 인터페이스
class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

// 구체 커맨드들
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
        // 실행 전 상태 저장
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

// 인보커 (Invoker) - Undo/Redo 지원
class CommandManager {
public:
    void execute(std::unique_ptr<Command> cmd) {
        cmd->execute();
        undoStack_.push(std::move(cmd));
        // 새 명령 실행 시 redo 스택 클리어
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

// 매크로 커맨드 (복합 커맨드)
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

// 사용 예
int main() {
    Document doc;
    CommandManager manager;

    // 명령 실행
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

### 4.4 Template Method (템플릿 메서드)

알고리즘의 골격을 정의하고 일부 단계를 서브클래스에서 구현합니다.

```cpp
#include <iostream>
#include <string>
#include <fstream>

// 추상 클래스 - 템플릿 메서드 정의
class DataParser {
public:
    virtual ~DataParser() = default;

    // 템플릿 메서드 - 알고리즘 골격
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
    // 기본 구현
    virtual void openFile(const std::string& filename) {
        std::cout << "Opening file: " << filename << "\n";
    }

    virtual void closeFile() {
        std::cout << "Closing file\n";
    }

    // 순수 가상 함수 - 서브클래스에서 반드시 구현
    virtual void extractData() = 0;
    virtual void parseData() = 0;

    // Hook 메서드 - 선택적으로 오버라이드
    virtual void analyzeData() {
        // 기본적으로 아무것도 하지 않음
    }
};

// 구체 클래스들
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
    // analyzeData() 사용하지 않음 (기본 구현 사용)
};

// 사용 예
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

## 5. C++ 특화 이디엄

### 5.1 CRTP (Curiously Recurring Template Pattern)

정적 다형성을 구현하는 기법입니다.

```cpp
#include <iostream>
#include <memory>
#include <vector>

// CRTP 기본 형태
template<typename Derived>
class Base {
public:
    void interface() {
        // 파생 클래스의 구현 호출 (정적 다형성)
        static_cast<Derived*>(this)->implementation();
    }

    // 기본 구현
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

// CRTP 활용 1: 믹스인 (Mixin)
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

// CRTP 활용 2: 정적 인터페이스 (Static Polymorphism)
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

// 템플릿 함수로 정적 다형성 활용
template<typename T>
void printArea(const Shape<T>& shape) {
    std::cout << "Area: " << shape.area() << "\n";
}

// CRTP 활용 3: Fluent Interface / Method Chaining
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

// 사용 예
int main() {
    // 기본 CRTP
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

구현을 헤더에서 숨기는 기법입니다.

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

    // 이동 연산
    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;

    // 복사 연산 (필요시)
    Widget(const Widget& other);
    Widget& operator=(const Widget& other);

    // 공개 인터페이스
    void setName(const std::string& name);
    std::string getName() const;
    void doSomething();

private:
    // 전방 선언만 - 구현 숨김
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

#endif

// === widget.cpp ===
// #include "widget.h"
#include <iostream>

// 구현 클래스 정의
class Widget::Impl {
public:
    std::string name;
    int internalData = 0;

    void internalProcess() {
        std::cout << "Internal processing for: " << name << "\n";
        internalData++;
    }
};

// Widget 구현
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

    // 이동
    Widget w2 = std::move(w);
    w2.doSomething();

    return 0;
}
```

### 5.3 Type Erasure (타입 소거)

다양한 타입을 단일 인터페이스로 다룹니다.

```cpp
#include <iostream>
#include <memory>
#include <vector>

// Type Erasure 패턴 구현
class Drawable {
public:
    template<typename T>
    Drawable(T obj) : pImpl_(std::make_shared<Model<T>>(std::move(obj))) {}

    void draw() const {
        pImpl_->draw();
    }

private:
    // 개념 (Concept) - 인터페이스
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
    };

    // 모델 (Model) - 구체 타입 래핑
    template<typename T>
    struct Model : Concept {
        Model(T obj) : object_(std::move(obj)) {}

        void draw() const override {
            object_.draw();  // T가 draw() 메서드를 가져야 함
        }

        T object_;
    };

    std::shared_ptr<const Concept> pImpl_;
};

// draw() 메서드를 가진 다양한 타입들
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

// std::function도 Type Erasure의 예시
// std::any도 Type Erasure의 예시

// 사용 예
int main() {
    // 다양한 타입을 하나의 컨테이너에 저장
    std::vector<Drawable> shapes;

    shapes.emplace_back(Circle(5.0));
    shapes.emplace_back(Square(3.0));
    shapes.emplace_back(Text("Hello"));

    // 동일한 인터페이스로 다루기
    for (const auto& shape : shapes) {
        shape.draw();
    }

    return 0;
}
```

### 5.4 RAII (Resource Acquisition Is Initialization)

리소스 관리를 생성자/소멸자에 연결합니다.

```cpp
#include <iostream>
#include <fstream>
#include <mutex>
#include <memory>

// 파일 핸들 RAII 래퍼
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

    // 복사 금지
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    // 이동 허용
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

// 락 가드 (std::lock_guard 유사)
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

// 스코프 가드 (범용 정리 작업)
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

// 헬퍼 함수
template<typename Func>
ScopeGuard<Func> makeScopeGuard(Func&& func) {
    return ScopeGuard<Func>(std::forward<Func>(func));
}

// 사용 예
int main() {
    // RAII 파일
    try {
        File file("/tmp/test.txt", "w");
        fprintf(file.get(), "Hello RAII!\n");
        // 스코프 종료 시 자동으로 파일 닫힘
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // 스코프 가드
    std::cout << "\n--- Scope Guard ---\n";
    {
        auto guard = makeScopeGuard([]() {
            std::cout << "Cleanup executed!\n";
        });

        std::cout << "Doing work...\n";
        // 스코프 종료 시 cleanup 실행
    }

    // dismiss 사용
    {
        auto guard = makeScopeGuard([]() {
            std::cout << "This won't run\n";
        });

        guard.dismiss();  // cleanup 취소
    }

    return 0;
}
```

---

## 6. 연습 문제

### 연습 1: 로그 시스템 (Singleton + Strategy)
```cpp
// 요구사항:
// 1. 싱글톤 Logger 구현
// 2. 다양한 출력 전략 (Console, File, Network)
// 3. 로그 레벨 (DEBUG, INFO, WARNING, ERROR)

// 힌트:
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

### 연습 2: UI 컴포넌트 (Composite + Decorator)
```cpp
// 요구사항:
// 1. UI 컴포넌트 계층 구조 (Window, Panel, Button)
// 2. 데코레이터로 테두리, 스크롤바 추가
// 3. 컴포지트로 컨테이너 구현

// 힌트:
class UIComponent {
public:
    virtual void render() = 0;
    virtual void add(std::shared_ptr<UIComponent> component) {}
};
```

### 연습 3: 문서 편집기 (Command + Memento)
```cpp
// 요구사항:
// 1. 텍스트 삽입, 삭제, 서식 명령 구현
// 2. 무제한 Undo/Redo 지원
// 3. 스냅샷 저장/복원 (Memento)

// 힌트:
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

### 연습 4: 플러그인 시스템 (Factory + Type Erasure)
```cpp
// 요구사항:
// 1. 런타임에 플러그인 등록/해제
// 2. 다양한 플러그인 타입 지원
// 3. 공통 인터페이스로 실행

// 힌트:
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

## 다음 단계

- [15_C++17_기능](15_C++17_기능.md) - 모던 C++ 기능 복습
- 디자인 패턴 심화 서적 참고 (GoF, Head First)
- 실제 프로젝트에 패턴 적용 연습

## 참고 자료

- "Design Patterns: Elements of Reusable Object-Oriented Software" (GoF)
- "Modern C++ Design" - Andrei Alexandrescu
- [Refactoring.Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

---

[← 이전: C++20 심화](17_C++20_심화.md) | [목차](00_Overview.md)
