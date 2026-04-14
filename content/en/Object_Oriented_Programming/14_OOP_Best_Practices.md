# Lesson 14: OOP Best Practices

## Learning Objectives

By the end of this lesson, you will be able to:
1. Identify and avoid common OOP anti-patterns
2. Apply naming conventions that communicate intent
3. Design classes with appropriate granularity (not too big, not too small)
4. Refactor procedural code into clean OOP designs
5. Write testable classes using dependency injection
6. Balance OOP with pragmatism — know when NOT to use OOP
7. Apply a practical OOP design checklist to your code

## Anti-Patterns to Avoid

### Anti-Pattern 1: The God Class

A class that does everything — violates SRP:

```python
# BAD: God class
class Application:
    def __init__(self):
        self.users = []
        self.orders = []
        self.products = []

    def register_user(self, name, email): ...
    def authenticate_user(self, email, password): ...
    def create_order(self, user_id, products): ...
    def process_payment(self, order_id, card): ...
    def send_email(self, to, subject, body): ...
    def generate_report(self, report_type): ...
    def backup_database(self): ...
    def render_template(self, name, context): ...
    # 50 more methods...
```

```python
# GOOD: Split into focused classes
class UserService:
    """Handles user registration and authentication."""
    def register(self, name, email): ...
    def authenticate(self, email, password): ...

class OrderService:
    """Handles order creation and management."""
    def create(self, user_id, products): ...
    def cancel(self, order_id): ...

class PaymentService:
    """Handles payment processing."""
    def process(self, order_id, card): ...
    def refund(self, order_id): ...

class EmailService:
    """Handles email delivery."""
    def send(self, to, subject, body): ...
```

### Anti-Pattern 2: Feature Envy

A method that uses more data from another class than its own:

```python
# BAD: Feature envy — Order's method accesses Customer's data heavily
class Order:
    def calculate_discount(self, customer):
        if customer.membership_level == "gold":
            if customer.years_active > 5:
                if customer.total_purchases > 10000:
                    return 0.20
                return 0.15
            return 0.10
        return 0.0

# GOOD: Move the method to where the data lives
class Customer:
    def get_discount_rate(self):
        """Customer knows its own discount logic."""
        if self.membership_level == "gold":
            if self.years_active > 5:
                if self.total_purchases > 10000:
                    return 0.20
                return 0.15
            return 0.10
        return 0.0

class Order:
    def calculate_discount(self):
        return self.customer.get_discount_rate() * self.total
```

### Anti-Pattern 3: Yo-Yo Inheritance

Deep inheritance hierarchies that force you to bounce up and down:

```
                    Animal
                      │
                  Vertebrate
                      │
                   Mammal
                      │
                  Carnivore
                      │
                  Canidae
                      │
                    Dog         <-- To understand Dog, you must read
                      │             6 levels of classes!
                  GermanShepherd
```

```
BETTER: Shallow hierarchy + composition

    Animal
      │
   ┌──┴──┐
  Dog   Cat        <-- 2 levels max

  Dog has:
    - DietBehavior (carnivore)
    - BreedInfo (German Shepherd)
```

### Anti-Pattern 4: Premature Abstraction

Creating abstractions before you understand the problem:

```python
# BAD: Over-engineering from the start
class AbstractDataProcessor(ABC):
    @abstractmethod
    def preprocess(self): ...
    @abstractmethod
    def process(self): ...
    @abstractmethod
    def postprocess(self): ...
    @abstractmethod
    def validate(self): ...

# ... for a program that only processes CSVs one way

# GOOD: Start simple, extract abstractions when patterns emerge
class CSVProcessor:
    def process(self, filepath):
        data = self._read_csv(filepath)
        cleaned = self._clean_data(data)
        return cleaned
```

### Anti-Pattern 5: Anemic Domain Model

Classes that are just data bags with no behavior:

```python
# BAD: Anemic — all logic is outside the class
class User:
    def __init__(self):
        self.name = ""
        self.email = ""
        self.is_active = True

def validate_user(user): ...  # Logic lives outside
def activate_user(user): ...
def deactivate_user(user): ...

# GOOD: Rich domain model — behavior lives with data
class User:
    def __init__(self, name, email):
        self._validate(name, email)
        self.name = name
        self.email = email
        self.is_active = True

    def _validate(self, name, email):
        if not name:
            raise ValueError("Name required")
        if "@" not in email:
            raise ValueError("Invalid email")

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True
```

## Naming Conventions

### Class Names

```python
# Classes: PascalCase, nouns that describe the entity
class UserAccount: ...
class PaymentProcessor: ...
class HTTPResponse: ...  # Acronyms stay uppercase
class DatabaseConnection: ...

# Bad names
class Manager: ...       # Too vague — manager of what?
class Utils: ...         # Grab-bag, probably violates SRP
class Data: ...          # Everything is data
class MyClass: ...       # No meaning
```

### Method Names

```python
class Order:
    # Actions: verb phrases
    def calculate_total(self): ...
    def apply_discount(self, rate): ...
    def submit(self): ...

    # Queries: is_/has_/can_ for booleans
    def is_valid(self): ...
    def has_items(self): ...
    def can_cancel(self): ...

    # Properties: noun phrases
    @property
    def total(self): ...

    @property
    def item_count(self): ...
```

## Class Design Guidelines

### Guideline 1: Keep Classes Focused

```
┌─────────────────────────────────────────────────┐
│  Rule of thumb:                                 │
│                                                 │
│  If you can't describe what a class does in a   │
│  single sentence WITHOUT using "and" or "or",   │
│  it probably does too much.                     │
│                                                 │
│  Good: "UserRepository persists user data"      │
│  Bad:  "AppManager handles users AND orders     │
│         AND sends emails OR notifications"      │
└─────────────────────────────────────────────────┘
```

### Guideline 2: Prefer Small Public Interfaces

```python
class EmailSender:
    """Small, focused public interface."""

    def send(self, to, subject, body):
        """The ONE public method — clean interface."""
        message = self._build_message(to, subject, body)
        self._validate(message)
        self._deliver(message)

    # Implementation details are private
    def _build_message(self, to, subject, body): ...
    def _validate(self, message): ...
    def _deliver(self, message): ...
```

### Guideline 3: Design for Testability

```python
# BAD: Hard to test — creates its own dependencies
class OrderProcessor:
    def __init__(self):
        self.db = PostgresDatabase()       # Hard-coded
        self.emailer = SMTPEmailService()   # Hard-coded

    def process(self, order):
        self.db.save(order)
        self.emailer.send(order.customer.email, "Order confirmed")


# GOOD: Dependencies injected — easy to test with mocks
class OrderProcessor:
    def __init__(self, db, emailer):
        self.db = db             # Injected
        self.emailer = emailer   # Injected

    def process(self, order):
        self.db.save(order)
        self.emailer.send(order.customer.email, "Order confirmed")


# In tests:
class FakeDB:
    def __init__(self):
        self.saved = []
    def save(self, item):
        self.saved.append(item)

class FakeEmailer:
    def __init__(self):
        self.sent = []
    def send(self, to, msg):
        self.sent.append((to, msg))

# Easy to test!
processor = OrderProcessor(FakeDB(), FakeEmailer())
```

### Guideline 4: Use Composition by Default

```python
# Start with composition; use inheritance only for true "is-a"

class Logger:
    def log(self, message): ...

class Validator:
    def validate(self, data): ...

class Repository:
    def save(self, entity): ...


class UserService:
    """Composed of focused components."""

    def __init__(self, repo, validator, logger):
        self._repo = repo
        self._validator = validator
        self._logger = logger

    def create_user(self, data):
        self._validator.validate(data)
        user = User(**data)
        self._repo.save(user)
        self._logger.log(f"User created: {user.name}")
        return user
```

## Refactoring Checklist

When reviewing OOP code, ask these questions:

```
┌─────────────────────────────────────────────────────────────┐
│  OOP Design Checklist                                       │
├─────────────────────────────────────────────────────────────┤
│  [ ] Does each class have a single, clear responsibility?   │
│  [ ] Is the class name a noun that describes its purpose?   │
│  [ ] Are methods verbs that describe actions?               │
│  [ ] Is the public interface minimal and focused?           │
│  [ ] Are dependencies injected, not hard-coded?             │
│  [ ] Is inheritance used only for true "is-a"?              │
│  [ ] Are hierarchies shallow (2-3 levels max)?              │
│  [ ] Do subclasses satisfy Liskov Substitution?             │
│  [ ] Are mutable defaults avoided in __init__?              │
│  [ ] Is internal state protected with _ or __?              │
│  [ ] Are invariants enforced through @property?             │
│  [ ] Is the code testable without real databases/network?   │
│  [ ] Would a function be simpler than a class here?         │
└─────────────────────────────────────────────────────────────┘
```

## When NOT to Use OOP

OOP is not always the best choice:

```python
# Functions are fine for stateless transformations
def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32

# Don't wrap it in a class just because you can:
class TemperatureConverter:  # Unnecessary!
    def convert(self, celsius):
        return celsius * 9 / 5 + 32


# Simple scripts don't need classes
# Just write functions and call them

# Data pipelines may be cleaner with functions
def extract(): ...
def transform(data): ...
def load(data): ...

# Compose with function calls
load(transform(extract()))
```

### The "Is This a Class?" Test

```
┌─────────────────────────────────────────────────┐
│  Use a CLASS when:                              │
│  - You have data + behavior that belong together│
│  - You need multiple instances with shared logic│
│  - You need to maintain state across method calls│
│  - You need inheritance or polymorphism         │
│                                                 │
│  Use a FUNCTION when:                           │
│  - The operation is stateless                   │
│  - Input -> output with no side state           │
│  - It's a one-off transformation                │
│  - A class would have only one public method    │
└─────────────────────────────────────────────────┘
```

## Practical Refactoring Example

```python
# BEFORE: Procedural spaghetti
def process_orders(orders, db_conn, smtp_server):
    for order in orders:
        # Validation
        if not order.get("items"):
            print(f"Order {order['id']} has no items")
            continue
        if order.get("total", 0) <= 0:
            print(f"Order {order['id']} has invalid total")
            continue

        # Calculate total
        total = sum(item["price"] * item["qty"] for item in order["items"])
        tax = total * 0.08
        grand_total = total + tax

        # Save to database
        db_conn.execute(f"INSERT INTO orders VALUES ({order['id']}, {grand_total})")

        # Send confirmation
        smtp_server.send(order["email"], f"Order {order['id']} confirmed: ${grand_total:.2f}")


# AFTER: Clean OOP
from dataclasses import dataclass, field


@dataclass
class OrderItem:
    name: str
    price: float
    quantity: int

    @property
    def subtotal(self):
        return self.price * self.quantity


@dataclass
class Order:
    id: int
    email: str
    items: list[OrderItem] = field(default_factory=list)
    tax_rate: float = 0.08

    @property
    def subtotal(self):
        return sum(item.subtotal for item in self.items)

    @property
    def tax(self):
        return self.subtotal * self.tax_rate

    @property
    def total(self):
        return self.subtotal + self.tax

    def is_valid(self):
        return len(self.items) > 0 and self.subtotal > 0


class OrderProcessor:
    def __init__(self, repository, notifier):
        self._repo = repository
        self._notifier = notifier

    def process(self, order: Order):
        if not order.is_valid():
            raise ValueError(f"Order {order.id} is invalid")

        self._repo.save(order)
        self._notifier.send_confirmation(order)
        return order.total
```

## Summary

- Avoid anti-patterns: God Class, Feature Envy, Yo-Yo Inheritance, Premature Abstraction, Anemic Domain Model
- Follow naming conventions: PascalCase for classes, verb phrases for methods, `is_`/`has_` for booleans
- Keep classes focused with small public interfaces
- Design for testability using dependency injection
- Prefer composition over inheritance by default
- Use OOP for stateful, behavior-rich entities; use functions for stateless transformations
- Apply the design checklist before finalizing class designs
- Start simple and refactor to patterns when complexity demands it

## Course Conclusion

Congratulations on completing the Object-Oriented Programming course! You now have a thorough understanding of OOP from foundational concepts to advanced design principles. Key takeaways:

1. **Core concepts**: Classes, objects, constructors, and the object lifecycle
2. **Four pillars**: Encapsulation, inheritance, polymorphism, and abstraction
3. **Design principles**: SOLID, composition over inheritance, design patterns
4. **Python-specific**: Magic methods, dataclasses, protocols, and modern idioms
5. **Pragmatism**: Know when to use OOP and when simpler approaches work better

Continue practicing by building projects that combine these concepts. The best way to internalize OOP is to design, implement, and refactor real systems.
