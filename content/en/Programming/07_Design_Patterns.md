# Design Patterns

**Previous**: [Functional Programming Concepts](./06_Functional_Programming.md) | **Next**: [Clean Code & Code Smells](./08_Clean_Code.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define what a design pattern is and name its four essential elements (name, problem, solution, consequences)
2. Classify patterns into creational, structural, and behavioral categories
3. Implement key creational patterns (Singleton, Factory Method, Builder) and explain when each is appropriate
4. Implement key structural patterns (Adapter, Decorator, Facade, Proxy, Composite) and identify real-world examples
5. Implement key behavioral patterns (Observer, Strategy, Command, Iterator, State, Template Method) and contrast their intents
6. Evaluate when a design pattern adds value versus when it introduces unnecessary complexity
7. Recognize common anti-patterns (God Object, Golden Hammer, Lava Flow) and explain how to avoid them
8. Explain how modern language features (first-class functions, pattern matching) reduce the need for certain classical patterns

---

Design patterns are reusable solutions to commonly occurring problems in software design. They provide a shared vocabulary for developers and encapsulate best practices refined over decades. This lesson covers the classic Gang of Four (GoF) patterns, when to use them, and when they might be overkill.

> **📌 Analogy:** Design patterns are like a chef's recipe book: proven combinations of ingredients and techniques for common dishes. You do not reinvent how to make a roux every time -- you apply the known recipe and focus your creativity on what makes your dish unique.

## What Are Design Patterns?

**Design Pattern** = Name + Problem + Solution + Consequences

- **Name**: A shared vocabulary (e.g., "use the Factory pattern here")
- **Problem**: When to apply the pattern
- **Solution**: General design that solves the problem
- **Consequences**: Trade-offs, costs, and benefits

### History: Gang of Four (1994)

The book *Design Patterns: Elements of Reusable Object-Oriented Software* by Gamma, Helm, Johnson, and Vlissides introduced 23 patterns, organized into three categories:

1. **Creational**: Object creation mechanisms
2. **Structural**: Composition of classes and objects
3. **Behavioral**: Communication between objects

### When to Use Patterns

**Good reasons:**
- Problem matches pattern's intent
- Pattern simplifies design
- Team understands the pattern

**Bad reasons (anti-patterns):**
- "We need more design patterns" (pattern fever)
- Using patterns for resume padding
- Overengineering simple problems
- Forcing patterns where they don't fit

**Remember:** Patterns are tools, not goals. Simple code > unnecessarily patterned code.

## Creational Patterns

These patterns deal with object creation, abstracting the instantiation process.

### Singleton

**Intent:** Ensure a class has only one instance and provide a global access point.

**When to use:**
- Managing shared resources (database connection pool, logger)
- Coordinating system-wide actions

**Java (thread-safe):**
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

**Why it's often an anti-pattern:**
- Global state makes testing difficult
- Tight coupling
- Violates Single Responsibility Principle
- Difficult to mock or replace

**Better alternative:** Dependency injection

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

### Factory Method

**Intent:** Define an interface for creating objects, but let subclasses decide which class to instantiate.

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

### Builder

**Intent:** Separate construction of a complex object from its representation, allowing step-by-step construction.

**Java (Fluent Interface):**
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

**C++ (SQL Query Builder):**
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

### Prototype

**Intent:** Create new objects by copying existing objects (prototypes).

**JavaScript (prototypal inheritance):**
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

## Structural Patterns

These patterns deal with object composition, forming larger structures while keeping them flexible.

### Adapter

**Intent:** Convert the interface of a class into another interface clients expect.

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

**Java (Object Adapter):**
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

### Decorator

**Intent:** Attach additional responsibilities to an object dynamically, providing a flexible alternative to subclassing.

**Java (I/O Streams):**
```java
// Classic example: Java I/O
InputStream fileStream = new FileInputStream("data.txt");
InputStream bufferedStream = new BufferedInputStream(fileStream);
InputStream compressedStream = new GZIPInputStream(bufferedStream);

// Each decorator adds functionality
```

**Python (Function Decorators):**
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

**C++ (Coffee Shop Example):**
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

### Facade

**Intent:** Provide a unified, simplified interface to a complex subsystem.

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

### Proxy

**Intent:** Provide a surrogate or placeholder for another object to control access.

**Types:**
- **Virtual Proxy**: Lazy initialization (create expensive object only when needed)
- **Protection Proxy**: Access control
- **Remote Proxy**: Represent object in different address space
- **Cache Proxy**: Cache results

**Python (Virtual Proxy with Caching):**
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

**Java (Protection Proxy):**
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

### Composite

**Intent:** Compose objects into tree structures to represent part-whole hierarchies, treating individual and composite objects uniformly.

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

## Behavioral Patterns

These patterns focus on communication between objects.

### Observer

**Intent:** Define a one-to-many dependency so that when one object changes state, all dependents are notified.

**JavaScript (Event System):**
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

**Python (Property Observer):**
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

### Strategy

**Intent:** Define a family of algorithms, encapsulate each one, and make them interchangeable.

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

### Command

**Intent:** Encapsulate a request as an object, allowing parameterization, queuing, logging, and undo operations.

**C++ (Text Editor with Undo/Redo):**
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

### Iterator

**Intent:** Provide a way to access elements of a collection sequentially without exposing the underlying representation.

**Python (Custom Iterator):**
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

### State

**Intent:** Allow an object to alter its behavior when its internal state changes.

**JavaScript (TCP Connection):**
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

### Template Method

**Intent:** Define the skeleton of an algorithm in a method, deferring some steps to subclasses.

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

## Anti-Patterns

**Common anti-patterns to avoid:**

- **God Object**: One class that does everything
- **Spaghetti Code**: Tangled control flow
- **Golden Hammer**: Using one pattern for every problem
- **Lava Flow**: Dead code kept "just in case"
- **Copy-Paste Programming**: Duplication instead of abstraction

## Modern Perspective

Many GoF patterns are less necessary in modern languages:

- **Strategy/Command**: First-class functions eliminate need for classes
- **Iterator**: Built into most languages (Python generators, Java Streams)
- **Singleton**: Dependency injection preferred
- **Visitor**: Pattern matching in functional languages

**Example (Strategy without classes in Python):**
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

## Summary

| Pattern | Category | Purpose | When to Use |
|---------|----------|---------|-------------|
| Singleton | Creational | One instance | Shared resources (use DI instead) |
| Factory Method | Creational | Defer instantiation | Object creation varies |
| Builder | Creational | Complex construction | Many optional parameters |
| Adapter | Structural | Interface conversion | Integrate incompatible code |
| Decorator | Structural | Add behavior dynamically | Flexible alternatives to subclassing |
| Facade | Structural | Simplify interface | Hide complex subsystems |
| Proxy | Structural | Control access | Lazy loading, access control |
| Observer | Behavioral | One-to-many notifications | Event systems |
| Strategy | Behavioral | Swappable algorithms | Algorithm varies at runtime |
| Command | Behavioral | Encapsulate requests | Undo/redo, queuing |

## Exercises

### Exercise 1: Identify Patterns
Identify which design pattern(s) are used in each scenario:

1. Java's `InputStream` hierarchy (BufferedInputStream, GZIPInputStream)
2. Spring Framework's dependency injection
3. GUI event listeners (button.onClick)
4. Python's `@property` decorator
5. SQL query builders

### Exercise 2: Implement Observer
Implement a weather station that notifies multiple displays when temperature changes. Include:
- `WeatherStation` (subject)
- Multiple display types (current conditions, statistics, forecast)
- Ability to add/remove displays dynamically

### Exercise 3: Refactor with Strategy
Refactor this code to use the Strategy pattern:

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

### Exercise 4: Build a Command System
Create a smart home automation system using the Command pattern:
- Support commands: turn on/off lights, adjust thermostat, lock/unlock doors
- Implement macro commands (e.g., "leaving home" turns off lights, locks doors, adjusts thermostat)
- Support undo for all commands

### Exercise 5: When NOT to Use Patterns
For each scenario, explain why using a design pattern would be overkill:

1. A script that reads a file and prints its contents
2. A simple calculator with add/subtract/multiply/divide
3. A todo list app with 3 screens
4. A config file with 5 settings

---

**Previous**: [Functional Programming Concepts](./06_Functional_Programming.md) | **Next**: [Clean Code & Code Smells](./08_Clean_Code.md)
