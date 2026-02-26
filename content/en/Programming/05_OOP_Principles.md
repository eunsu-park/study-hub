# OOP Principles

**Previous**: [Control Flow Patterns](./04_Control_Flow_Patterns.md) | **Next**: [Functional Programming Concepts](./06_Functional_Programming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the four pillars of OOP -- encapsulation, abstraction, inheritance, and polymorphism -- with concrete examples
2. Implement encapsulation using access modifiers and the information hiding principle
3. Design class hierarchies using inheritance and recognize when composition is a better alternative
4. Apply compile-time polymorphism (overloading) and runtime polymorphism (overriding/duck typing)
5. State and apply each of the five SOLID principles (SRP, OCP, LSP, ISP, DIP)
6. Identify the diamond problem in multiple inheritance and describe solutions (MRO, virtual inheritance)
7. Apply the Law of Demeter to reduce coupling between objects
8. Refactor code from inheritance-based to composition-based design when appropriate

---

Object-Oriented Programming (OOP) is built on fundamental principles that guide how we structure code. Understanding these principles deeply -- not just superficially -- is crucial for writing maintainable, scalable software. This lesson explores the four pillars of OOP, SOLID principles, and when to apply (or avoid) each concept.

> **📌 Analogy:** A class is an architect's blueprint that specifies the structure (attributes) and functions (methods) of a building. Every building constructed from the same blueprint shares the design, but each has its own address and occupants -- just as each object has its own state.

## The Four Pillars of OOP

### 1. Encapsulation

**Encapsulation** is the bundling of data and methods that operate on that data within a single unit (class), while restricting direct access to some of the object's components.

#### Data Hiding and Access Modifiers

Different languages implement encapsulation differently:

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

#### Information Hiding Principle

The key idea: **hide implementation details, expose only what's necessary**. This allows you to change internal implementation without breaking code that uses your class.

**Bad (violates encapsulation):**
```java
public class User {
    public String name;  // Direct access
    public List<String> permissions;  // Can be modified directly
}

// Client code
user.permissions.add("ADMIN");  // Bypasses any validation
```

**Good:**
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

### 2. Abstraction

**Abstraction** means hiding complexity and showing only essential features. It's about creating simple interfaces that hide complex implementations.

#### Abstract Classes vs Interfaces

**Abstract Class Example (Java):**
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

**Interface Example (C++):**
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

### 3. Inheritance

**Inheritance** allows a class to acquire properties and methods from another class, promoting code reuse through an "is-a" relationship.

#### Single Inheritance

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

#### The Diamond Problem (Multiple Inheritance)

**Python (supports multiple inheritance):**
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

**C++ (allows multiple inheritance with potential issues):**
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

#### When NOT to Use Inheritance

**Anti-pattern:**
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

**Better (composition):**
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

### 4. Polymorphism

**Polymorphism** means "many forms"—one interface with multiple implementations.

#### Compile-Time Polymorphism (Method Overloading)

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

#### Runtime Polymorphism (Method Overriding)

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

#### Duck Typing (Dynamic Languages)

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

## Composition Over Inheritance

Modern design often favors **composition** (has-a) over **inheritance** (is-a) because it's more flexible.

**Inheritance (rigid):**
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

**Composition (flexible):**
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

## SOLID Principles

### S: Single Responsibility Principle (SRP)

**A class should have only one reason to change.**

**Violation:**
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

**Fixed:**
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

### O: Open/Closed Principle (OCP)

**Software entities should be open for extension, but closed for modification.**

**Violation:**
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

**Fixed:**
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

### L: Liskov Substitution Principle (LSP)

**Subtypes must be substitutable for their base types without altering program correctness.**

**Violation:**
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

**Fixed:**
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

### I: Interface Segregation Principle (ISP)

**Many client-specific interfaces are better than one general-purpose interface.**

**Violation:**
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

**Fixed:**
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

### D: Dependency Inversion Principle (DIP)

**Depend on abstractions, not concretions. High-level modules should not depend on low-level modules.**

**Violation:**
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

**Fixed:**
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

## Law of Demeter (Principle of Least Knowledge)

**"Don't talk to strangers."** An object should only call methods on:
1. Itself
2. Objects passed as parameters
3. Objects it creates
4. Its direct component objects

**Violation:**
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

**Fixed:**
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

## Practical Examples

### Complete SOLID Example (JavaScript)

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

## Summary

The four pillars and SOLID principles work together:

| Principle | What It Prevents | Key Benefit |
|-----------|------------------|-------------|
| Encapsulation | Data corruption | Data integrity |
| Abstraction | Unnecessary complexity | Simplicity |
| Inheritance | Code duplication | Reusability |
| Polymorphism | Rigid code | Flexibility |
| SRP | God classes | Maintainability |
| OCP | Modification ripples | Stability |
| LSP | Broken hierarchies | Correctness |
| ISP | Fat interfaces | Decoupling |
| DIP | Tight coupling | Testability |

## Exercises

### Exercise 1: Identify Violations
Review this code and identify which SOLID principles are violated:

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

### Exercise 2: Refactor to Composition
Refactor this inheritance-based design to use composition:

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

### Exercise 3: Apply Liskov Substitution
Fix this LSP violation:

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

### Exercise 4: Implement SOLID Design
Design a notification system that:
- Supports email, SMS, and push notifications
- Allows filtering notifications by priority
- Can be easily extended with new notification types
- Follows all SOLID principles

### Exercise 5: Code Review
Review a codebase (your own or open-source) and identify:
1. One encapsulation violation
2. One abstraction that could be improved
3. One inappropriate use of inheritance
4. One place where polymorphism would help
5. At least three SOLID violations

---

**Previous**: [Control Flow Patterns](./04_Control_Flow_Patterns.md) | **Next**: [Functional Programming Concepts](./06_Functional_Programming.md)
