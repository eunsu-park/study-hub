# 디자인 패턴

> **주제**: Programming
> **레슨**: 7 of 16
> **선수지식**: OOP 원칙(캡슐화, 상속, 다형성) 이해, 기본 프로그래밍 경험
> **목표**: 일반적인 디자인 패턴(Gang of Four 및 그 이상)을 마스터하고, 언제 적용할지 인식하며, 트레이드오프 이해

## 소개

디자인 패턴(Design Pattern)은 소프트웨어 설계에서 일반적으로 발생하는 문제에 대한 재사용 가능한 솔루션입니다. 개발자를 위한 공유 어휘를 제공하고 수십 년간 다듬어진 모범 사례를 캡슐화합니다. 이 레슨에서는 고전적인 Gang of Four(GoF) 패턴, 언제 사용할지, 그리고 언제 과도할 수 있는지를 다룹니다.

## 디자인 패턴이란?

**디자인 패턴** = 이름 + 문제 + 솔루션 + 결과

- **이름**: 공유 어휘 (예: "여기서 Factory 패턴을 사용하세요")
- **문제**: 패턴을 적용할 때
- **솔루션**: 문제를 해결하는 일반적인 설계
- **결과**: 트레이드오프, 비용, 이점

### 역사: Gang of Four (1994)

Gamma, Helm, Johnson, Vlissides의 책 *Design Patterns: Elements of Reusable Object-Oriented Software*는 23개의 패턴을 소개했으며, 세 가지 범주로 구성됩니다:

1. **생성(Creational)**: 객체 생성 메커니즘
2. **구조(Structural)**: 클래스와 객체의 조합
3. **행동(Behavioral)**: 객체 간 통신

### 패턴을 사용해야 할 때

**좋은 이유:**
- 문제가 패턴의 의도와 일치
- 패턴이 설계를 단순화
- 팀이 패턴을 이해

**나쁜 이유 (안티패턴):**
- "더 많은 디자인 패턴이 필요하다" (패턴 열병)
- 이력서 채우기 위한 패턴 사용
- 단순한 문제를 과도하게 엔지니어링
- 맞지 않는 곳에 패턴 강제

**기억하세요:** 패턴은 도구이지 목표가 아닙니다. 단순한 코드 > 불필요하게 패턴화된 코드.

## 생성 패턴(Creational Patterns)

이 패턴들은 객체 생성을 다루며, 인스턴스화 프로세스를 추상화합니다.

### 싱글턴(Singleton)

**의도:** 클래스가 하나의 인스턴스만 갖도록 보장하고 전역 접근 지점 제공.

**사용 시기:**
- 공유 리소스 관리 (데이터베이스 연결 풀, 로거)
- 시스템 전체 작업 조정

**Java (스레드 안전):**
```java
public class DatabaseConnection {
    private static volatile DatabaseConnection instance;
    private Connection connection;

    // Private constructor prevents external instantiation
    private DatabaseConnection() {
        // Initialize connection
        connection = DriverManager.getConnection("jdbc:...");
    }

    // Double-checked locking for thread safety
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }

    public Connection getConnection() {
        return connection;
    }
}

// Usage
DatabaseConnection db = DatabaseConnection.getInstance();
```

**Python:**
```python
class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.log_file = open('app.log', 'a')
            self.initialized = True

    def log(self, message):
        self.log_file.write(f"{message}\n")
        self.log_file.flush()

# Usage
logger1 = Logger()
logger2 = Logger()
assert logger1 is logger2  # Same instance
```

**C++ (Meyer's Singleton):**
```cpp
class Config {
public:
    static Config& getInstance() {
        static Config instance;  // Thread-safe in C++11+
        return instance;
    }

    // Delete copy constructor and assignment
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    void setValue(const std::string& key, const std::string& value) {
        settings[key] = value;
    }

    std::string getValue(const std::string& key) {
        return settings[key];
    }

private:
    Config() {}  // Private constructor
    std::map<std::string, std::string> settings;
};

// Usage
Config::getInstance().setValue("timeout", "30");
```

**왜 종종 안티패턴인가:**
- 전역 상태가 테스트를 어렵게 만듦
- 강한 결합
- 단일 책임 원칙 위반
- 모킹이나 대체가 어려움

**더 나은 대안:** 의존성 주입(Dependency Injection)

```java
// Instead of Singleton
public class UserService {
    private final Database database;

    public UserService(Database database) {  // Injected dependency
        this.database = database;
    }
}

// Easy to test with mock
Database mockDb = new MockDatabase();
UserService service = new UserService(mockDb);
```

### 팩토리 메서드(Factory Method)

**의도:** 객체 생성을 위한 인터페이스를 정의하되, 서브클래스가 인스턴스화할 클래스를 결정하도록 함.

**JavaScript:**
```javascript
// Product interface
class Button {
    render() {
        throw new Error('Must implement render()');
    }
}

class WindowsButton extends Button {
    render() {
        return '<button class="windows">Click me</button>';
    }
}

class MacButton extends Button {
    render() {
        return '<button class="mac">Click me</button>';
    }
}

// Creator
class Dialog {
    render() {
        const button = this.createButton();  // Factory method
        return `<div>${button.render()}</div>`;
    }

    createButton() {
        throw new Error('Must implement createButton()');
    }
}

class WindowsDialog extends Dialog {
    createButton() {
        return new WindowsButton();
    }
}

class MacDialog extends Dialog {
    createButton() {
        return new MacButton();
    }
}

// Usage
const os = detectOS();
let dialog;

if (os === 'Windows') {
    dialog = new WindowsDialog();
} else if (os === 'Mac') {
    dialog = new MacDialog();
}

console.log(dialog.render());
```

**Python (Simple Factory vs Factory Method):**
```python
from abc import ABC, abstractmethod

# Simple Factory (not GoF pattern, but useful)
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type):
        if shape_type == 'circle':
            return Circle()
        elif shape_type == 'rectangle':
            return Rectangle()
        else:
            raise ValueError(f"Unknown shape: {shape_type}")

# Factory Method (GoF pattern)
class Document(ABC):
    @abstractmethod
    def create_page(self):
        pass

    def print_document(self):
        page = self.create_page()  # Factory method
        print(f"Printing: {page.render()}")

class PDFDocument(Document):
    def create_page(self):
        return PDFPage()

class WordDocument(Document):
    def create_page(self):
        return WordPage()

class Page(ABC):
    @abstractmethod
    def render(self):
        pass

class PDFPage(Page):
    def render(self):
        return "PDF page content"

class WordPage(Page):
    def render(self):
        return "Word page content"
```

### 빌더(Builder)

**의도:** 복잡한 객체의 생성을 표현과 분리하여 단계별 생성 허용.

**Java (플루언트 인터페이스):**
```java
public class HttpRequest {
    private String method;
    private String url;
    private Map<String, String> headers;
    private String body;

    private HttpRequest() {
        headers = new HashMap<>();
    }

    public static class Builder {
        private HttpRequest request;

        public Builder() {
            request = new HttpRequest();
        }

        public Builder method(String method) {
            request.method = method;
            return this;
        }

        public Builder url(String url) {
            request.url = url;
            return this;
        }

        public Builder header(String key, String value) {
            request.headers.put(key, value);
            return this;
        }

        public Builder body(String body) {
            request.body = body;
            return this;
        }

        public HttpRequest build() {
            // Validation
            if (request.method == null || request.url == null) {
                throw new IllegalStateException("method and url are required");
            }
            return request;
        }
    }

    // Getters...
}

// Usage
HttpRequest request = new HttpRequest.Builder()
    .method("POST")
    .url("https://api.example.com/users")
    .header("Content-Type", "application/json")
    .header("Authorization", "Bearer token123")
    .body("{\"name\": \"Alice\"}")
    .build();
```

**C++ (SQL 쿼리 빌더):**
```cpp
class SQLQuery {
private:
    std::string table;
    std::vector<std::string> columns;
    std::string whereClause;
    std::string orderBy;
    int limitValue = -1;

public:
    class Builder {
    private:
        SQLQuery query;

    public:
        Builder& select(const std::vector<std::string>& cols) {
            query.columns = cols;
            return *this;
        }

        Builder& from(const std::string& tbl) {
            query.table = tbl;
            return *this;
        }

        Builder& where(const std::string& condition) {
            query.whereClause = condition;
            return *this;
        }

        Builder& orderBy(const std::string& column) {
            query.orderBy = column;
            return *this;
        }

        Builder& limit(int n) {
            query.limitValue = n;
            return *this;
        }

        SQLQuery build() {
            if (query.table.empty()) {
                throw std::runtime_error("table is required");
            }
            return query;
        }
    };

    std::string toSQL() const {
        std::string sql = "SELECT ";

        if (columns.empty()) {
            sql += "*";
        } else {
            for (size_t i = 0; i < columns.size(); i++) {
                if (i > 0) sql += ", ";
                sql += columns[i];
            }
        }

        sql += " FROM " + table;

        if (!whereClause.empty()) {
            sql += " WHERE " + whereClause;
        }

        if (!orderBy.empty()) {
            sql += " ORDER BY " + orderBy;
        }

        if (limitValue > 0) {
            sql += " LIMIT " + std::to_string(limitValue);
        }

        return sql;
    }
};

// Usage
SQLQuery query = SQLQuery::Builder()
    .select({"id", "name", "email"})
    .from("users")
    .where("age > 18")
    .orderBy("name")
    .limit(10)
    .build();

std::cout << query.toSQL() << "\n";
```

### 프로토타입(Prototype)

**의도:** 기존 객체(프로토타입)를 복사하여 새 객체 생성.

**JavaScript (프로토타입 상속):**
```javascript
const carPrototype = {
    drive() {
        console.log(`Driving a ${this.make} ${this.model}`);
    },

    clone() {
        return Object.create(Object.getPrototypeOf(this),
            Object.getOwnPropertyDescriptors(this));
    }
};

function createCar(make, model, year) {
    const car = Object.create(carPrototype);
    car.make = make;
    car.model = model;
    car.year = year;
    return car;
}

const tesla = createCar('Tesla', 'Model 3', 2024);
const teslaClone = tesla.clone();
teslaClone.year = 2025;

tesla.drive();      // Driving a Tesla Model 3
teslaClone.drive(); // Driving a Tesla Model 3
```

## 구조 패턴(Structural Patterns)

이 패턴들은 객체 조합을 다루며, 유연성을 유지하면서 더 큰 구조를 형성합니다.

### 어댑터(Adapter)

**의도:** 클래스의 인터페이스를 클라이언트가 기대하는 다른 인터페이스로 변환.

**Python:**
```python
# Legacy system
class OldPaymentProcessor:
    def process_payment(self, amount):
        print(f"Old system processing ${amount}")

# New interface
class PaymentProcessor:
    def pay(self, amount):
        raise NotImplementedError

# Adapter
class PaymentAdapter(PaymentProcessor):
    def __init__(self, old_processor):
        self.old_processor = old_processor

    def pay(self, amount):
        # Adapt new interface to old interface
        self.old_processor.process_payment(amount)

# Client code expects new interface
def checkout(payment_processor: PaymentProcessor, amount):
    payment_processor.pay(amount)

# Usage
old_system = OldPaymentProcessor()
adapter = PaymentAdapter(old_system)
checkout(adapter, 99.99)  # Works with new interface
```

**Java (객체 어댑터):**
```java
// Target interface
interface MediaPlayer {
    void play(String filename);
}

// Adaptee (incompatible interface)
class AdvancedMediaPlayer {
    public void playVlc(String filename) {
        System.out.println("Playing VLC file: " + filename);
    }

    public void playMp4(String filename) {
        System.out.println("Playing MP4 file: " + filename);
    }
}

// Adapter
class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedPlayer;

    public MediaAdapter() {
        advancedPlayer = new AdvancedMediaPlayer();
    }

    @Override
    public void play(String filename) {
        if (filename.endsWith(".vlc")) {
            advancedPlayer.playVlc(filename);
        } else if (filename.endsWith(".mp4")) {
            advancedPlayer.playMp4(filename);
        }
    }
}

// Usage
MediaPlayer player = new MediaAdapter();
player.play("video.mp4");
```

### 데코레이터(Decorator)

**의도:** 객체에 동적으로 추가 책임을 부여하여 서브클래싱의 유연한 대안 제공.

**Java (I/O 스트림):**
```java
// Classic example: Java I/O
InputStream fileStream = new FileInputStream("data.txt");
InputStream bufferedStream = new BufferedInputStream(fileStream);
InputStream compressedStream = new GZIPInputStream(bufferedStream);

// Each decorator adds functionality
```

**Python (함수 데코레이터):**
```python
import time
from functools import wraps

# Timing decorator
def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

# Logging decorator
def logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Stack decorators
@timing
@logging
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(10)
```

**C++ (커피숍 예제):**
```cpp
// Component
class Coffee {
public:
    virtual ~Coffee() = default;
    virtual std::string getDescription() const = 0;
    virtual double cost() const = 0;
};

// Concrete Component
class SimpleCoffee : public Coffee {
public:
    std::string getDescription() const override {
        return "Simple coffee";
    }

    double cost() const override {
        return 2.0;
    }
};

// Decorator
class CoffeeDecorator : public Coffee {
protected:
    Coffee* coffee;

public:
    CoffeeDecorator(Coffee* c) : coffee(c) {}
    virtual ~CoffeeDecorator() { delete coffee; }
};

// Concrete Decorators
class Milk : public CoffeeDecorator {
public:
    Milk(Coffee* c) : CoffeeDecorator(c) {}

    std::string getDescription() const override {
        return coffee->getDescription() + ", milk";
    }

    double cost() const override {
        return coffee->cost() + 0.5;
    }
};

class Sugar : public CoffeeDecorator {
public:
    Sugar(Coffee* c) : CoffeeDecorator(c) {}

    std::string getDescription() const override {
        return coffee->getDescription() + ", sugar";
    }

    double cost() const override {
        return coffee->cost() + 0.2;
    }
};

// Usage
Coffee* myCoffee = new SimpleCoffee();
myCoffee = new Milk(myCoffee);
myCoffee = new Sugar(myCoffee);

std::cout << myCoffee->getDescription() << " costs $" << myCoffee->cost() << "\n";
// Output: Simple coffee, milk, sugar costs $2.7

delete myCoffee;
```

### 퍼사드(Facade)

**의도:** 복잡한 하위 시스템에 통합되고 단순화된 인터페이스 제공.

**JavaScript:**
```javascript
// Complex subsystem
class CPU {
    freeze() { console.log('CPU: Freezing...'); }
    jump(position) { console.log(`CPU: Jumping to ${position}`); }
    execute() { console.log('CPU: Executing...'); }
}

class Memory {
    load(position, data) {
        console.log(`Memory: Loading ${data} at ${position}`);
    }
}

class HardDrive {
    read(sector, size) {
        console.log(`HDD: Reading ${size} bytes from sector ${sector}`);
        return 'boot data';
    }
}

// Facade
class ComputerFacade {
    constructor() {
        this.cpu = new CPU();
        this.memory = new Memory();
        this.hdd = new HardDrive();
    }

    start() {
        this.cpu.freeze();
        const bootData = this.hdd.read(0, 1024);
        this.memory.load(0, bootData);
        this.cpu.jump(0);
        this.cpu.execute();
    }
}

// Client code (simple!)
const computer = new ComputerFacade();
computer.start();  // Hides complex subsystem
```

### 프록시(Proxy)

**의도:** 다른 객체에 대한 대리자 또는 자리 표시자를 제공하여 접근 제어.

**유형:**
- **가상 프록시(Virtual Proxy)**: 지연 초기화 (필요할 때만 비싼 객체 생성)
- **보호 프록시(Protection Proxy)**: 접근 제어
- **원격 프록시(Remote Proxy)**: 다른 주소 공간의 객체 표현
- **캐시 프록시(Cache Proxy)**: 결과 캐싱

**Python (캐싱을 포함한 가상 프록시):**
```python
class Image:
    def __init__(self, filename):
        self.filename = filename
        self._load()

    def _load(self):
        print(f"Loading image from {self.filename}")
        # Expensive operation
        time.sleep(2)
        self.data = f"Image data from {self.filename}"

    def display(self):
        print(f"Displaying {self.data}")

# Proxy with lazy loading and caching
class ImageProxy:
    def __init__(self, filename):
        self.filename = filename
        self._image = None  # Not loaded yet

    def display(self):
        if self._image is None:
            self._image = Image(self.filename)  # Lazy load
        self._image.display()

# Usage
image = ImageProxy("photo.jpg")
# Image not loaded yet

image.display()  # Loads now (2 second delay)
image.display()  # Uses cached image (instant)
```

**Java (보호 프록시):**
```java
interface Document {
    void display();
    void edit(String content);
}

class RealDocument implements Document {
    private String content;

    public void display() {
        System.out.println("Displaying: " + content);
    }

    public void edit(String newContent) {
        content = newContent;
        System.out.println("Document edited");
    }
}

class ProtectedDocument implements Document {
    private RealDocument document;
    private String userRole;

    public ProtectedDocument(String role) {
        document = new RealDocument();
        userRole = role;
    }

    public void display() {
        document.display();  // Anyone can view
    }

    public void edit(String content) {
        if (userRole.equals("ADMIN")) {
            document.edit(content);
        } else {
            System.out.println("Access denied: insufficient permissions");
        }
    }
}

// Usage
Document doc = new ProtectedDocument("USER");
doc.display();        // OK
doc.edit("new text"); // Denied
```

### 컴포지트(Composite)

**의도:** 객체들을 트리 구조로 조합하여 부분-전체 계층 구조를 표현하고, 개별 객체와 복합 객체를 동일하게 처리.

**C++:**
```cpp
#include <vector>
#include <memory>

// Component
class Graphic {
public:
    virtual ~Graphic() = default;
    virtual void draw() const = 0;
    virtual void add(std::shared_ptr<Graphic> g) {
        throw std::runtime_error("Cannot add to leaf");
    }
};

// Leaf
class Circle : public Graphic {
public:
    void draw() const override {
        std::cout << "Drawing circle\n";
    }
};

class Rectangle : public Graphic {
public:
    void draw() const override {
        std::cout << "Drawing rectangle\n";
    }
};

// Composite
class CompositeGraphic : public Graphic {
private:
    std::vector<std::shared_ptr<Graphic>> children;

public:
    void add(std::shared_ptr<Graphic> g) override {
        children.push_back(g);
    }

    void draw() const override {
        for (const auto& child : children) {
            child->draw();
        }
    }
};

// Usage
auto circle1 = std::make_shared<Circle>();
auto circle2 = std::make_shared<Circle>();
auto rect = std::make_shared<Rectangle>();

auto group1 = std::make_shared<CompositeGraphic>();
group1->add(circle1);
group1->add(rect);

auto group2 = std::make_shared<CompositeGraphic>();
group2->add(circle2);
group2->add(group1);  // Nested composite

group2->draw();  // Draws entire tree
```

## 행동 패턴(Behavioral Patterns)

이 패턴들은 객체 간 통신에 초점을 맞춥니다.

### 옵저버(Observer)

**의도:** 일대다 의존성을 정의하여 한 객체가 상태를 변경하면 모든 종속 객체에 알림.

**JavaScript (이벤트 시스템):**
```javascript
class EventEmitter {
    constructor() {
        this.listeners = {};
    }

    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event]
                .filter(cb => cb !== callback);
        }
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }
}

// Subject
class Stock extends EventEmitter {
    constructor(symbol) {
        super();
        this.symbol = symbol;
        this.price = 0;
    }

    setPrice(price) {
        this.price = price;
        this.emit('priceChange', { symbol: this.symbol, price });
    }
}

// Observers
class Investor {
    constructor(name) {
        this.name = name;
    }

    update(data) {
        console.log(`${this.name} notified: ${data.symbol} is now $${data.price}`);
    }
}

// Usage
const apple = new Stock('AAPL');

const investor1 = new Investor('Alice');
const investor2 = new Investor('Bob');

apple.on('priceChange', data => investor1.update(data));
apple.on('priceChange', data => investor2.update(data));

apple.setPrice(150);  // Both investors notified
apple.setPrice(155);  // Both investors notified
```

**Python (속성 옵저버):**
```python
class Observable:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, data):
        for observer in self._observers:
            observer.update(data)

class Subject(Observable):
    def __init__(self):
        super().__init__()
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify(value)  # Notify observers on change

class ConcreteObserver:
    def __init__(self, name):
        self.name = name

    def update(self, data):
        print(f"{self.name} received update: {data}")

# Usage
subject = Subject()

obs1 = ConcreteObserver("Observer 1")
obs2 = ConcreteObserver("Observer 2")

subject.attach(obs1)
subject.attach(obs2)

subject.state = "new state"  # Both observers notified
```

### 전략(Strategy)

**의도:** 알고리즘 패밀리를 정의하고 각각을 캡슐화하여 상호 교환 가능하게 만듦.

**Java:**
```java
// Strategy interface
interface PaymentStrategy {
    void pay(int amount);
}

// Concrete strategies
class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(int amount) {
        System.out.println("Paid $" + amount + " with credit card " + cardNumber);
    }
}

class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(int amount) {
        System.out.println("Paid $" + amount + " via PayPal account " + email);
    }
}

class CryptoPayment implements PaymentStrategy {
    private String walletAddress;

    public CryptoPayment(String wallet) {
        this.walletAddress = wallet;
    }

    @Override
    public void pay(int amount) {
        System.out.println("Paid $" + amount + " to crypto wallet " + walletAddress);
    }
}

// Context
class ShoppingCart {
    private PaymentStrategy paymentStrategy;

    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }

    public void checkout(int amount) {
        paymentStrategy.pay(amount);
    }
}

// Usage
ShoppingCart cart = new ShoppingCart();

cart.setPaymentStrategy(new CreditCardPayment("1234-5678"));
cart.checkout(100);

cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
cart.checkout(50);

cart.setPaymentStrategy(new CryptoPayment("0x1234..."));
cart.checkout(200);
```

### 커맨드(Command)

**의도:** 요청을 객체로 캡슐화하여 매개변수화, 큐잉, 로깅, 실행 취소 작업 허용.

**C++ (실행 취소/재실행이 있는 텍스트 편집기):**
```cpp
#include <stack>
#include <memory>

// Command interface
class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

// Receiver
class TextEditor {
private:
    std::string text;

public:
    void insertText(const std::string& str, size_t pos) {
        text.insert(pos, str);
    }

    void deleteText(size_t pos, size_t length) {
        text.erase(pos, length);
    }

    std::string getText() const {
        return text;
    }
};

// Concrete Commands
class InsertCommand : public Command {
private:
    TextEditor* editor;
    std::string textToInsert;
    size_t position;

public:
    InsertCommand(TextEditor* ed, const std::string& text, size_t pos)
        : editor(ed), textToInsert(text), position(pos) {}

    void execute() override {
        editor->insertText(textToInsert, position);
    }

    void undo() override {
        editor->deleteText(position, textToInsert.length());
    }
};

class DeleteCommand : public Command {
private:
    TextEditor* editor;
    std::string deletedText;
    size_t position;
    size_t length;

public:
    DeleteCommand(TextEditor* ed, size_t pos, size_t len)
        : editor(ed), position(pos), length(len) {
        deletedText = editor->getText().substr(pos, len);
    }

    void execute() override {
        editor->deleteText(position, length);
    }

    void undo() override {
        editor->insertText(deletedText, position);
    }
};

// Invoker
class CommandManager {
private:
    std::stack<std::shared_ptr<Command>> undoStack;
    std::stack<std::shared_ptr<Command>> redoStack;

public:
    void executeCommand(std::shared_ptr<Command> cmd) {
        cmd->execute();
        undoStack.push(cmd);
        // Clear redo stack on new command
        while (!redoStack.empty()) {
            redoStack.pop();
        }
    }

    void undo() {
        if (!undoStack.empty()) {
            auto cmd = undoStack.top();
            undoStack.pop();
            cmd->undo();
            redoStack.push(cmd);
        }
    }

    void redo() {
        if (!redoStack.empty()) {
            auto cmd = redoStack.top();
            redoStack.pop();
            cmd->execute();
            undoStack.push(cmd);
        }
    }
};

// Usage
TextEditor editor;
CommandManager manager;

auto insert1 = std::make_shared<InsertCommand>(&editor, "Hello", 0);
manager.executeCommand(insert1);
std::cout << editor.getText() << "\n";  // Hello

auto insert2 = std::make_shared<InsertCommand>(&editor, " World", 5);
manager.executeCommand(insert2);
std::cout << editor.getText() << "\n";  // Hello World

manager.undo();
std::cout << editor.getText() << "\n";  // Hello

manager.redo();
std::cout << editor.getText() << "\n";  // Hello World
```

### 반복자(Iterator)

**의도:** 기본 표현을 노출하지 않고 컬렉션의 요소에 순차적으로 접근하는 방법 제공.

**Python (사용자 정의 반복자):**
```python
class LinkedListNode:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        if not self.head:
            self.head = LinkedListNode(value)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = LinkedListNode(value)

    def __iter__(self):
        return LinkedListIterator(self.head)

class LinkedListIterator:
    def __init__(self, head):
        self.current = head

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is None:
            raise StopIteration
        value = self.current.value
        self.current = self.current.next
        return value

# Usage
linked_list = LinkedList()
linked_list.append(1)
linked_list.append(2)
linked_list.append(3)

for value in linked_list:
    print(value)  # 1, 2, 3
```

### 상태(State)

**의도:** 객체의 내부 상태가 변경될 때 객체의 동작을 변경할 수 있도록 함.

**JavaScript (TCP 연결):**
```javascript
// State interface
class ConnectionState {
    open(connection) { throw new Error('Not implemented'); }
    close(connection) { throw new Error('Not implemented'); }
    read(connection) { throw new Error('Not implemented'); }
    write(connection, data) { throw new Error('Not implemented'); }
}

// Concrete States
class ClosedState extends ConnectionState {
    open(connection) {
        console.log('Opening connection...');
        connection.setState(new OpenState());
    }

    close(connection) {
        console.log('Already closed');
    }

    read(connection) {
        console.log('Cannot read: connection closed');
    }

    write(connection, data) {
        console.log('Cannot write: connection closed');
    }
}

class OpenState extends ConnectionState {
    open(connection) {
        console.log('Already open');
    }

    close(connection) {
        console.log('Closing connection...');
        connection.setState(new ClosedState());
    }

    read(connection) {
        console.log('Reading data...');
        return 'data';
    }

    write(connection, data) {
        console.log(`Writing: ${data}`);
    }
}

// Context
class TCPConnection {
    constructor() {
        this.state = new ClosedState();
    }

    setState(state) {
        this.state = state;
    }

    open() { this.state.open(this); }
    close() { this.state.close(this); }
    read() { return this.state.read(this); }
    write(data) { this.state.write(this, data); }
}

// Usage
const conn = new TCPConnection();
conn.read();        // Cannot read: connection closed
conn.open();        // Opening connection...
conn.write('Hello'); // Writing: Hello
conn.close();       // Closing connection...
```

### 템플릿 메서드(Template Method)

**의도:** 메서드에서 알고리즘의 골격을 정의하고 일부 단계를 서브클래스에 위임.

**Python:**
```python
from abc import ABC, abstractmethod

class DataParser(ABC):
    # Template method
    def parse(self, filename):
        data = self.read_file(filename)
        parsed = self.parse_data(data)
        validated = self.validate(parsed)
        self.use_data(validated)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            return f.read()

    @abstractmethod
    def parse_data(self, data):
        pass

    def validate(self, data):
        # Default validation (can be overridden)
        return data

    @abstractmethod
    def use_data(self, data):
        pass

class CSVParser(DataParser):
    def parse_data(self, data):
        lines = data.strip().split('\n')
        return [line.split(',') for line in lines]

    def use_data(self, data):
        print(f"Processing {len(data)} CSV rows")

class JSONParser(DataParser):
    def parse_data(self, data):
        import json
        return json.loads(data)

    def validate(self, data):
        if not isinstance(data, list):
            raise ValueError("Expected list")
        return data

    def use_data(self, data):
        print(f"Processing {len(data)} JSON objects")

# Usage
csv_parser = CSVParser()
csv_parser.parse('data.csv')

json_parser = JSONParser()
json_parser.parse('data.json')
```

## 안티패턴(Anti-Patterns)

**피해야 할 일반적인 안티패턴:**

- **신 객체(God Object)**: 모든 것을 하는 하나의 클래스
- **스파게티 코드(Spaghetti Code)**: 얽힌 제어 흐름
- **황금 망치(Golden Hammer)**: 모든 문제에 하나의 패턴 사용
- **용암 흐름(Lava Flow)**: "혹시 몰라서" 유지하는 죽은 코드
- **복사-붙여넣기 프로그래밍**: 추상화 대신 중복

## 현대적 관점

많은 GoF 패턴들은 현대 언어에서 덜 필요합니다:

- **전략/커맨드**: 일급 함수가 클래스의 필요성 제거
- **반복자**: 대부분의 언어에 내장 (Python 제너레이터, Java Streams)
- **싱글턴**: 의존성 주입 선호
- **방문자**: 함수형 언어의 패턴 매칭

**예제 (Python에서 클래스 없는 전략):**
```python
# Old (class-based)
class Strategy:
    def execute(self):
        pass

# New (function-based)
def strategy_a(data):
    return data * 2

def strategy_b(data):
    return data ** 2

strategies = {'a': strategy_a, 'b': strategy_b}
result = strategies['a'](10)
```

## 요약

| 패턴 | 범주 | 목적 | 사용 시기 |
|---------|----------|---------|-------------|
| 싱글턴(Singleton) | 생성 | 하나의 인스턴스 | 공유 리소스 (대신 DI 사용) |
| 팩토리 메서드(Factory Method) | 생성 | 인스턴스화 지연 | 객체 생성이 다양함 |
| 빌더(Builder) | 생성 | 복잡한 생성 | 많은 선택적 매개변수 |
| 어댑터(Adapter) | 구조 | 인터페이스 변환 | 비호환 코드 통합 |
| 데코레이터(Decorator) | 구조 | 동적으로 동작 추가 | 서브클래싱의 유연한 대안 |
| 퍼사드(Facade) | 구조 | 인터페이스 단순화 | 복잡한 하위 시스템 숨기기 |
| 프록시(Proxy) | 구조 | 접근 제어 | 지연 로딩, 접근 제어 |
| 옵저버(Observer) | 행동 | 일대다 알림 | 이벤트 시스템 |
| 전략(Strategy) | 행동 | 교환 가능한 알고리즘 | 런타임에 알고리즘 변경 |
| 커맨드(Command) | 행동 | 요청 캡슐화 | 실행 취소/재실행, 큐잉 |

## 연습 문제

### 연습 문제 1: 패턴 식별
각 시나리오에서 어떤 디자인 패턴이 사용되는지 식별하세요:

1. Java의 `InputStream` 계층 구조 (BufferedInputStream, GZIPInputStream)
2. Spring Framework의 의존성 주입
3. GUI 이벤트 리스너 (button.onClick)
4. Python의 `@property` 데코레이터
5. SQL 쿼리 빌더

### 연습 문제 2: 옵저버 구현
온도가 변경될 때 여러 디스플레이에 알리는 기상 관측소를 구현하세요. 포함할 것:
- `WeatherStation` (subject)
- 여러 디스플레이 유형 (현재 조건, 통계, 예보)
- 동적으로 디스플레이 추가/제거 기능

### 연습 문제 3: 전략으로 리팩토링
이 코드를 전략 패턴을 사용하도록 리팩토링하세요:

```java
class Order {
    public double calculateShipping(String method, double weight) {
        if (method.equals("standard")) {
            return weight * 0.5;
        } else if (method.equals("express")) {
            return weight * 1.5;
        } else if (method.equals("overnight")) {
            return weight * 3.0;
        }
        return 0;
    }
}
```

### 연습 문제 4: 커맨드 시스템 구축
커맨드 패턴을 사용하여 스마트 홈 자동화 시스템을 만드세요:
- 명령 지원: 조명 켜기/끄기, 온도 조절, 문 잠금/잠금 해제
- 매크로 명령 구현 (예: "외출" = 조명 끄기, 문 잠금, 온도 조절)
- 모든 명령에 대한 실행 취소 지원

### 연습 문제 5: 패턴을 사용하지 말아야 할 때
각 시나리오에 대해 디자인 패턴 사용이 과도한 이유를 설명하세요:

1. 파일을 읽고 내용을 출력하는 스크립트
2. 더하기/빼기/곱하기/나누기가 있는 간단한 계산기
3. 3개 화면이 있는 할 일 목록 앱
4. 5개 설정이 있는 구성 파일

---

**이전**: [함수형 프로그래밍 개념](06_Functional_Programming.md)
**다음**: [클린 코드와 코드 스멜](08_Clean_Code.md)
