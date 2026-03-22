# OOP 원칙

> **주제**: Programming
> **레슨**: 5 of 16
> **선수지식**: 클래스와 객체에 대한 기본 이해, 객체지향 언어 하나 이상 숙지
> **목표**: OOP의 네 가지 기둥(캡슐화, 추상화, 상속, 다형성)을 마스터하고 SOLID 원칙을 적용하여 유지보수 가능하고 확장 가능한 코드 작성

## 소개

객체지향 프로그래밍(OOP, Object-Oriented Programming)은 코드를 구조화하는 방법을 안내하는 기본 원칙들 위에 구축됩니다. 이러한 원칙들을 피상적으로가 아니라 깊이 이해하는 것은 유지보수 가능하고 확장 가능한 소프트웨어를 작성하는 데 필수적입니다. 이 레슨에서는 OOP의 네 가지 기둥, SOLID 원칙, 그리고 각 개념을 언제 적용하고(혹은 피해야) 하는지 탐구합니다.

## OOP의 네 가지 기둥

### 1. 캡슐화(Encapsulation)

**캡슐화**는 데이터와 그 데이터를 조작하는 메서드를 단일 단위(클래스) 내에 묶는 것이며, 객체의 일부 컴포넌트에 대한 직접 접근을 제한합니다.

#### 데이터 은닉과 접근 제한자(Access Modifiers)

언어마다 캡슐화를 다르게 구현합니다:

**Java:**
```java
public class BankAccount {
    private double balance;  // Private: only accessible within this class
    private String accountNumber;

    // Public interface for controlled access
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }

    public boolean withdraw(double amount) {
        if (amount > 0 && balance >= amount) {
            balance -= amount;
            return true;
        }
        return false;
    }

    public double getBalance() {
        return balance;
    }
}
```

**Python:**
```python
class BankAccount:
    def __init__(self, account_number):
        self.__balance = 0.0  # Name mangling (weak privacy)
        self.__account_number = account_number

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def withdraw(self, amount):
        if amount > 0 and self.__balance >= amount:
            self.__balance -= amount
            return True
        return False

    @property
    def balance(self):
        return self.__balance
```

**C++:**
```cpp
class BankAccount {
private:
    double balance;
    std::string accountNumber;

public:
    BankAccount(const std::string& num) : balance(0.0), accountNumber(num) {}

    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }

    bool withdraw(double amount) {
        if (amount > 0 && balance >= amount) {
            balance -= amount;
            return true;
        }
        return false;
    }

    double getBalance() const {
        return balance;
    }
};
```

#### 정보 은닉 원칙(Information Hiding Principle)

핵심 아이디어: **구현 세부사항을 숨기고 필요한 것만 노출합니다**. 이를 통해 클래스를 사용하는 코드를 깨뜨리지 않고 내부 구현을 변경할 수 있습니다.

**나쁜 예 (캡슐화 위반):**
```java
public class User {
    public String name;  // Direct access
    public List<String> permissions;  // Can be modified directly
}

// Client code
user.permissions.add("ADMIN");  // Bypasses any validation
```

**좋은 예:**
```java
public class User {
    private String name;
    private Set<String> permissions;  // Use Set to prevent duplicates

    public void grantPermission(String permission) {
        if (isValidPermission(permission)) {
            permissions.add(permission);
            auditLog.log("Permission granted: " + permission);
        }
    }

    public boolean hasPermission(String permission) {
        return permissions.contains(permission);
    }

    private boolean isValidPermission(String permission) {
        // Validation logic
        return true;
    }
}
```

### 2. 추상화(Abstraction)

**추상화**는 복잡성을 숨기고 필수적인 특징만 보여주는 것을 의미합니다. 복잡한 구현을 숨기는 단순한 인터페이스를 만드는 것입니다.

#### 추상 클래스 vs 인터페이스

**추상 클래스 예제 (Java):**
```java
public abstract class Shape {
    protected String color;

    public abstract double calculateArea();
    public abstract double calculatePerimeter();

    // Concrete method shared by all shapes
    public void setColor(String color) {
        this.color = color;
    }
}

public class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }
}

public class Rectangle extends Shape {
    private double width, height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return width * height;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }
}
```

**인터페이스 예제 (C++):**
```cpp
// Pure abstract interface
class Drawable {
public:
    virtual void draw() = 0;  // Pure virtual function
    virtual ~Drawable() = default;
};

class Movable {
public:
    virtual void move(int x, int y) = 0;
    virtual ~Movable() = default;
};

// Class implementing multiple interfaces
class GameCharacter : public Drawable, public Movable {
private:
    int x, y;

public:
    void draw() override {
        std::cout << "Drawing character at (" << x << ", " << y << ")\n";
    }

    void move(int newX, int newY) override {
        x = newX;
        y = newY;
    }
};
```

### 3. 상속(Inheritance)

**상속**은 클래스가 다른 클래스로부터 속성과 메서드를 획득하여 "is-a" 관계를 통해 코드 재사용을 촉진합니다.

#### 단일 상속(Single Inheritance)

**Python:**
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement speak()")

    def sleep(self):
        print(f"{self.name} is sleeping")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
print(dog.speak())  # Buddy says Woof!
dog.sleep()         # Buddy is sleeping
```

#### 다이아몬드 문제(Diamond Problem, 다중 상속)

**Python (다중 상속 지원):**
```python
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):  # Diamond inheritance
    pass

d = D()
d.method()  # Prints "B" (MRO: D -> B -> C -> A)
print(D.__mro__)  # Shows Method Resolution Order
```

**C++ (잠재적 문제가 있는 다중 상속 허용):**
```cpp
class Device {
protected:
    std::string name;
};

class Printer : public Device {
public:
    void print() { std::cout << "Printing...\n"; }
};

class Scanner : public Device {
public:
    void scan() { std::cout << "Scanning...\n"; }
};

// Diamond problem: two copies of Device
class MultiFunctionPrinter : public Printer, public Scanner {
    // Ambiguity: which 'name' to use?
};

// Solution: Virtual inheritance
class PrinterV : virtual public Device { };
class ScannerV : virtual public Device { };
class MultiFunctionPrinterV : public PrinterV, public ScannerV { };
```

#### 상속을 사용하지 말아야 할 때

**안티패턴:**
```java
// Inheritance used for code reuse (wrong reason)
class Stack extends ArrayList<Object> {
    public void push(Object item) {
        add(item);
    }

    public Object pop() {
        return remove(size() - 1);
    }
}

// Problem: Stack inherits all ArrayList methods
stack.add(0, "item");  // Can insert at arbitrary position - breaks stack contract!
```

**더 나은 방법 (컴포지션):**
```java
class Stack {
    private List<Object> elements = new ArrayList<>();

    public void push(Object item) {
        elements.add(item);
    }

    public Object pop() {
        if (elements.isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.remove(elements.size() - 1);
    }

    // Only expose stack operations
}
```

### 4. 다형성(Polymorphism)

**다형성**은 "많은 형태"를 의미하며, 하나의 인터페이스에 여러 구현이 있는 것입니다.

#### 컴파일 타임 다형성(Compile-Time Polymorphism, 메서드 오버로딩)

**Java:**
```java
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public double add(double a, double b) {
        return a + b;
    }

    public int add(int a, int b, int c) {
        return a + b + c;
    }
}

Calculator calc = new Calculator();
calc.add(5, 3);        // Calls int version
calc.add(5.5, 3.2);    // Calls double version
calc.add(1, 2, 3);     // Calls three-argument version
```

#### 런타임 다형성(Runtime Polymorphism, 메서드 오버라이딩)

**C++:**
```cpp
class PaymentProcessor {
public:
    virtual void processPayment(double amount) {
        std::cout << "Processing payment: $" << amount << "\n";
    }

    virtual ~PaymentProcessor() = default;
};

class CreditCardProcessor : public PaymentProcessor {
public:
    void processPayment(double amount) override {
        std::cout << "Processing credit card payment: $" << amount << "\n";
        // Credit card-specific logic
    }
};

class PayPalProcessor : public PaymentProcessor {
public:
    void processPayment(double amount) override {
        std::cout << "Processing PayPal payment: $" << amount << "\n";
        // PayPal-specific logic
    }
};

void checkout(PaymentProcessor* processor, double amount) {
    processor->processPayment(amount);  // Runtime polymorphism
}

// Usage
CreditCardProcessor creditCard;
PayPalProcessor paypal;

checkout(&creditCard, 99.99);  // Uses CreditCardProcessor
checkout(&paypal, 49.99);       // Uses PayPalProcessor
```

#### 덕 타이핑(Duck Typing, 동적 언어)

**Python:**
```python
# "If it walks like a duck and quacks like a duck, it's a duck"
class Duck:
    def quack(self):
        print("Quack!")

    def fly(self):
        print("Flying!")

class Person:
    def quack(self):
        print("I'm imitating a duck!")

    def fly(self):
        print("I'm flapping my arms!")

def make_it_quack(duck):
    duck.quack()  # No type checking needed

make_it_quack(Duck())    # Works
make_it_quack(Person())  # Also works!
```

## 상속보다 컴포지션(Composition Over Inheritance)

현대 설계는 **상속(is-a)**보다 **컴포지션(has-a)**을 선호하는데, 더 유연하기 때문입니다.

**상속 (경직됨):**
```java
class Bird {
    void fly() { }
}

class Penguin extends Bird {
    @Override
    void fly() {
        throw new UnsupportedOperationException("Penguins can't fly!");
    }
}
```

**컴포지션 (유연함):**
```java
interface FlyBehavior {
    void fly();
}

class CanFly implements FlyBehavior {
    public void fly() {
        System.out.println("Flying!");
    }
}

class CannotFly implements FlyBehavior {
    public void fly() {
        System.out.println("Can't fly");
    }
}

class Bird {
    private FlyBehavior flyBehavior;

    public Bird(FlyBehavior flyBehavior) {
        this.flyBehavior = flyBehavior;
    }

    public void performFly() {
        flyBehavior.fly();
    }

    // Can change behavior at runtime
    public void setFlyBehavior(FlyBehavior fb) {
        this.flyBehavior = fb;
    }
}

Bird eagle = new Bird(new CanFly());
Bird penguin = new Bird(new CannotFly());
```

## SOLID 원칙

### S: 단일 책임 원칙(Single Responsibility Principle, SRP)

**클래스는 변경해야 할 이유가 하나만 있어야 합니다.**

**위반:**
```python
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def save_to_database(self):
        # Database logic (reason 1 to change)
        pass

    def send_welcome_email(self):
        # Email logic (reason 2 to change)
        pass

    def generate_report(self):
        # Reporting logic (reason 3 to change)
        pass
```

**수정:**
```python
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user):
        # Database logic only
        pass

class EmailService:
    def send_welcome_email(self, user):
        # Email logic only
        pass

class UserReportGenerator:
    def generate(self, user):
        # Reporting logic only
        pass
```

### O: 개방-폐쇄 원칙(Open/Closed Principle, OCP)

**소프트웨어 엔티티는 확장에는 열려 있어야 하지만 수정에는 닫혀 있어야 합니다.**

**위반:**
```java
class AreaCalculator {
    public double calculateArea(Object shape) {
        if (shape instanceof Circle) {
            Circle circle = (Circle) shape;
            return Math.PI * circle.radius * circle.radius;
        } else if (shape instanceof Rectangle) {
            Rectangle rect = (Rectangle) shape;
            return rect.width * rect.height;
        }
        // Must modify this method to add new shapes!
        return 0;
    }
}
```

**수정:**
```java
interface Shape {
    double calculateArea();
}

class Circle implements Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }
}

class Rectangle implements Shape {
    private double width, height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return width * height;
    }
}

class AreaCalculator {
    public double calculateArea(Shape shape) {
        return shape.calculateArea();  // No modification needed for new shapes
    }
}

// Adding Triangle requires no changes to AreaCalculator
class Triangle implements Shape {
    private double base, height;

    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return 0.5 * base * height;
    }
}
```

### L: 리스코프 치환 원칙(Liskov Substitution Principle, LSP)

**서브타입은 프로그램 정확성을 해치지 않고 기본 타입으로 대체 가능해야 합니다.**

**위반:**
```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def set_width(self, width):
        self.width = width
        self.height = width  # Maintains square invariant

    def set_height(self, height):
        self.width = height
        self.height = height

# LSP violation
def test_rectangle(rect):
    rect.set_width(5)
    rect.set_height(4)
    assert rect.area() == 20  # Fails for Square!

rectangle = Rectangle(0, 0)
test_rectangle(rectangle)  # Pass

square = Square(0, 0)
test_rectangle(square)  # Fail! area() returns 16, not 20
```

**수정:**
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

# No longer substitutable, which is correct—they're different shapes
```

### I: 인터페이스 분리 원칙(Interface Segregation Principle, ISP)

**클라이언트별 세분화된 인터페이스가 범용 인터페이스보다 낫습니다.**

**위반:**
```java
interface Worker {
    void work();
    void eat();
    void sleep();
}

class HumanWorker implements Worker {
    public void work() { /* ... */ }
    public void eat() { /* ... */ }
    public void sleep() { /* ... */ }
}

class RobotWorker implements Worker {
    public void work() { /* ... */ }
    public void eat() { /* Robot doesn't eat! */ }
    public void sleep() { /* Robot doesn't sleep! */ }
}
```

**수정:**
```java
interface Workable {
    void work();
}

interface Eatable {
    void eat();
}

interface Sleepable {
    void sleep();
}

class HumanWorker implements Workable, Eatable, Sleepable {
    public void work() { /* ... */ }
    public void eat() { /* ... */ }
    public void sleep() { /* ... */ }
}

class RobotWorker implements Workable {
    public void work() { /* ... */ }
    // Only implements what it needs
}
```

### D: 의존성 역전 원칙(Dependency Inversion Principle, DIP)

**추상화에 의존하고 구체화에 의존하지 마세요. 상위 수준 모듈은 하위 수준 모듈에 의존하지 않아야 합니다.**

**위반:**
```cpp
class MySQLDatabase {
public:
    void save(const std::string& data) {
        std::cout << "Saving to MySQL: " << data << "\n";
    }
};

class UserService {
private:
    MySQLDatabase database;  // Tight coupling to concrete class

public:
    void createUser(const std::string& username) {
        database.save(username);  // Can't switch databases
    }
};
```

**수정:**
```cpp
// Abstraction
class Database {
public:
    virtual void save(const std::string& data) = 0;
    virtual ~Database() = default;
};

// Concrete implementations
class MySQLDatabase : public Database {
public:
    void save(const std::string& data) override {
        std::cout << "Saving to MySQL: " << data << "\n";
    }
};

class PostgreSQLDatabase : public Database {
public:
    void save(const std::string& data) override {
        std::cout << "Saving to PostgreSQL: " << data << "\n";
    }
};

// High-level module depends on abstraction
class UserService {
private:
    Database* database;  // Depends on abstraction

public:
    UserService(Database* db) : database(db) {}

    void createUser(const std::string& username) {
        database->save(username);  // Works with any Database implementation
    }
};

// Usage
MySQLDatabase mysql;
UserService service1(&mysql);
service1.createUser("alice");

PostgreSQLDatabase postgres;
UserService service2(&postgres);
service2.createUser("bob");
```

## 데메테르의 법칙(Law of Demeter, 최소 지식 원칙)

**"낯선 이에게 말하지 마세요."** 객체는 다음에만 메서드를 호출해야 합니다:
1. 자기 자신
2. 파라미터로 전달된 객체
3. 자신이 생성한 객체
4. 직접 구성 요소인 객체

**위반:**
```java
class Car {
    private Engine engine;

    public Engine getEngine() {
        return engine;
    }
}

class Engine {
    private FuelPump fuelPump;

    public FuelPump getFuelPump() {
        return fuelPump;
    }
}

class FuelPump {
    public void pump() { }
}

// Client code
car.getEngine().getFuelPump().pump();  // Chains through three objects!
```

**수정:**
```java
class Car {
    private Engine engine;

    public void refuel() {
        engine.refuel();  // Car tells Engine, doesn't reach into it
    }
}

class Engine {
    private FuelPump fuelPump;

    public void refuel() {
        fuelPump.pump();  // Engine tells FuelPump
    }
}

class FuelPump {
    public void pump() { }
}

// Client code
car.refuel();  // Simple and decoupled
```

## 실전 예제

### 완전한 SOLID 예제 (JavaScript)

```javascript
// S: Single Responsibility
class Product {
    constructor(name, price) {
        this.name = name;
        this.price = price;
    }
}

class ShoppingCart {
    constructor() {
        this.items = [];
    }

    addItem(product, quantity) {
        this.items.push({ product, quantity });
    }

    removeItem(productName) {
        this.items = this.items.filter(item => item.product.name !== productName);
    }

    getItems() {
        return this.items;
    }
}

// O: Open/Closed - discount strategies
class DiscountStrategy {
    calculate(total) {
        return total;
    }
}

class NoDiscount extends DiscountStrategy {
    calculate(total) {
        return total;
    }
}

class PercentageDiscount extends DiscountStrategy {
    constructor(percentage) {
        super();
        this.percentage = percentage;
    }

    calculate(total) {
        return total * (1 - this.percentage / 100);
    }
}

class FixedDiscount extends DiscountStrategy {
    constructor(amount) {
        super();
        this.amount = amount;
    }

    calculate(total) {
        return Math.max(0, total - this.amount);
    }
}

// D: Dependency Inversion
class OrderService {
    constructor(paymentProcessor, discountStrategy) {
        this.paymentProcessor = paymentProcessor;  // Inject dependency
        this.discountStrategy = discountStrategy;
    }

    checkout(cart) {
        const total = cart.getItems().reduce((sum, item) =>
            sum + item.product.price * item.quantity, 0);

        const discounted = this.discountStrategy.calculate(total);
        return this.paymentProcessor.process(discounted);
    }
}

// Usage
const cart = new ShoppingCart();
cart.addItem(new Product("Laptop", 1000), 1);
cart.addItem(new Product("Mouse", 25), 2);

const discount = new PercentageDiscount(10);
const payment = new CreditCardProcessor();
const orderService = new OrderService(payment, discount);

orderService.checkout(cart);
```

## 요약

네 가지 기둥과 SOLID 원칙은 함께 작동합니다:

| 원칙 | 방지하는 것 | 주요 이점 |
|-----------|------------------|-------------|
| 캡슐화(Encapsulation) | 데이터 손상 | 데이터 무결성 |
| 추상화(Abstraction) | 불필요한 복잡성 | 단순성 |
| 상속(Inheritance) | 코드 중복 | 재사용성 |
| 다형성(Polymorphism) | 경직된 코드 | 유연성 |
| SRP | 신 클래스(God classes) | 유지보수성 |
| OCP | 수정 파급 | 안정성 |
| LSP | 깨진 계층 구조 | 정확성 |
| ISP | 비대한 인터페이스 | 결합도 감소 |
| DIP | 강한 결합 | 테스트 용이성 |

## 연습 문제

### 연습 문제 1: 위반 사항 식별
이 코드를 검토하고 어떤 SOLID 원칙이 위반되었는지 식별하세요:

```python
class EmailService:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.port = 587

    def send_email(self, recipient, subject, body):
        # Connect to SMTP
        # Send email
        pass

    def save_user(self, user):
        # Save to database
        pass

    def generate_invoice(self, order):
        # Generate PDF invoice
        pass

    def calculate_tax(self, amount, country):
        if country == "US":
            return amount * 0.07
        elif country == "UK":
            return amount * 0.20
        elif country == "DE":
            return amount * 0.19
        # ... more countries
```

### 연습 문제 2: 컴포지션으로 리팩토링
상속 기반 설계를 컴포지션을 사용하도록 리팩토링하세요:

```java
class Employee {
    protected String name;
    protected double salary;

    public void work() { }
    public void attendMeeting() { }
}

class Manager extends Employee {
    public void managePeople() { }
}

class Developer extends Employee {
    public void writeCode() { }
}

class ManagerDeveloper extends ??? {  // Diamond problem!
    // Needs both managePeople() and writeCode()
}
```

### 연습 문제 3: 리스코프 치환 원칙 적용
이 LSP 위반을 수정하세요:

```cpp
class Bird {
public:
    virtual void fly() {
        std::cout << "Flying\n";
    }
};

class Ostrich : public Bird {
public:
    void fly() override {
        throw std::logic_error("Ostriches can't fly!");
    }
};

void makeBirdFly(Bird* bird) {
    bird->fly();  // May throw exception!
}
```

### 연습 문제 4: SOLID 설계 구현
다음 요구사항을 만족하는 알림 시스템을 설계하세요:
- 이메일, SMS, 푸시 알림 지원
- 우선순위별 알림 필터링 허용
- 새로운 알림 유형으로 쉽게 확장 가능
- 모든 SOLID 원칙 준수

### 연습 문제 5: 코드 리뷰
코드베이스(자신의 것 또는 오픈소스)를 검토하고 다음을 식별하세요:
1. 캡슐화 위반 1개
2. 개선 가능한 추상화 1개
3. 부적절한 상속 사용 1개
4. 다형성이 도움이 될 곳 1개
5. SOLID 위반 최소 3개

---

**이전**: [04_Programming_Paradigms.md](04_Programming_Paradigms.md)
**다음**: [함수형 프로그래밍 개념](06_Functional_Programming.md)
