# Clean Code & Code Smells

**Previous**: [Design Patterns](./07_Design_Patterns.md) | **Next**: [Error Handling](./09_Error_Handling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of technical debt and describe its impact on long-term software maintenance
2. Apply intention-revealing naming conventions for variables, functions, and classes
3. Write small, focused functions that follow the single-responsibility principle and have minimal arguments
4. Identify at least eight common code smells (Long Method, God Class, Feature Envy, Data Clumps, Shotgun Surgery, Primitive Obsession, Switch Statements, Dead Code)
5. Apply standard refactoring techniques (Extract Method, Rename, Introduce Explaining Variable, Move Method, Replace Magic Number)
6. Distinguish between good and bad comments and prefer self-documenting code
7. Apply the DRY, KISS, YAGNI, and Boy Scout Rule principles to everyday coding decisions
8. Evaluate code formatting practices (vertical/horizontal) and explain the value of team conventions and automated formatters

---

"Any fool can write code that a computer can understand. Good programmers write code that humans can understand." -- Martin Fowler. Clean code is code that is easy to read, understand, and modify. It minimizes cognitive load, reduces bugs, and makes collaboration easier. This lesson covers principles from Robert C. Martin's *Clean Code*, Martin Fowler's *Refactoring*, and other industry best practices.

## The Cost of Bad Code

**Technical debt** accumulates when we take shortcuts:

```python
# Technical debt example
def process(data):  # What does this process?
    x = []  # What is x?
    for d in data:  # Single-letter variable
        if d[0] > 100:  # Magic number
            x.append(d[1] * 1.2)  # What is 1.2?
    return x

# The "interest" we pay:
# - Hard to understand
# - Hard to modify
# - Hard to test
# - Bugs hide easily
```

**The clean version:**
```python
PRICE_MARKUP = 1.2
MINIMUM_QUANTITY = 100

def calculate_marked_up_prices(orders):
    """Extract prices from high-quantity orders and apply markup."""
    marked_up_prices = []

    for order in orders:
        quantity = order[0]
        price = order[1]

        if quantity > MINIMUM_QUANTITY:
            marked_up_price = price * PRICE_MARKUP
            marked_up_prices.append(marked_up_price)

    return marked_up_prices
```

## Meaningful Names

Names are the primary way we communicate intent. Good names make code self-documenting.

### Intention-Revealing Names

**Bad:**
```java
int d;  // elapsed time in days
```

**Good:**
```java
int elapsedTimeInDays;
int daysSinceCreation;
int daysSinceModification;
```

### Avoid Disinformation

**Bad:**
```javascript
const accountList = {};  // Not a list, it's an object!

function hp() { }  // What does hp mean?

const theAccounts = [];  // "the" adds no information
```

**Good:**
```javascript
const accountMap = {};

function calculateHorsePower() { }
// Or if context is clear:
function calculatePower() { }

const accounts = [];
```

### Use Pronounceable Names

**Bad:**
```python
genymdhms  # generation year, month, day, hour, minute, second
```

**Good:**
```python
generation_timestamp
```

### Use Searchable Names

**Bad:**
```cpp
for (int i = 0; i < 7; i++) {  // What is 7?
    // ...
}
```

**Good:**
```cpp
const int DAYS_IN_WEEK = 7;

for (int day = 0; day < DAYS_IN_WEEK; day++) {
    // ...
}
```

### Class Names: Nouns

Classes represent things (nouns):

```java
// Good
class Customer { }
class Account { }
class PaymentProcessor { }

// Bad
class Manager { }  // Too vague
class Data { }     // Meaningless
class Info { }     // Meaningless
```

### Method Names: Verbs

Methods perform actions (verbs):

```python
# Good
def calculate_total():
    pass

def send_email():
    pass

def is_valid():  # Boolean: is_, has_, can_
    pass

# Bad
def total():  # Noun (ambiguous)
    pass

def email():  # Noun
    pass
```

### Avoid Encodings

**Bad (Hungarian notation):**
```cpp
int iCount;
string strName;
bool bIsActive;
```

**Good:**
```cpp
int count;
string name;
bool isActive;
```

### One Word Per Concept

Pick one term and stick with it:

**Inconsistent:**
```java
class UserFetcher { }
class AccountRetriever { }
class ProductGetter { }
```

**Consistent:**
```java
class UserRepository { }
class AccountRepository { }
class ProductRepository { }
```

### Avoid Noise Words

**Bad:**
```javascript
const productData = {};
const productInfo = {};
const theProduct = {};

function getProductData() { }
function getProductInfo() { }
```

What's the difference between `Data` and `Info`? None!

**Good:**
```javascript
const product = {};

function getProduct() { }
```

## Functions

### Small Functions

Functions should do **one thing**, do it **well**, and do it **only**.

**Bad (does too much):**
```python
def process_order(order):
    # Validate
    if not order.get('customer_id'):
        raise ValueError("Missing customer ID")
    if not order.get('items'):
        raise ValueError("No items")

    # Calculate total
    total = 0
    for item in order['items']:
        total += item['price'] * item['quantity']

    # Apply discount
    if order.get('coupon'):
        discount = get_discount(order['coupon'])
        total -= total * discount

    # Save to database
    db.save('orders', order)

    # Send confirmation email
    send_email(order['customer_id'], f"Order confirmed: ${total}")

    return total
```

**Good (single responsibilities):**
```python
def process_order(order):
    validate_order(order)
    total = calculate_order_total(order)
    save_order(order)
    send_confirmation_email(order, total)
    return total

def validate_order(order):
    if not order.get('customer_id'):
        raise ValueError("Missing customer ID")
    if not order.get('items'):
        raise ValueError("No items")

def calculate_order_total(order):
    subtotal = sum(item['price'] * item['quantity']
                   for item in order['items'])
    discount = get_discount_amount(order, subtotal)
    return subtotal - discount

def get_discount_amount(order, subtotal):
    if coupon := order.get('coupon'):
        discount_rate = get_discount(coupon)
        return subtotal * discount_rate
    return 0

def save_order(order):
    db.save('orders', order)

def send_confirmation_email(order, total):
    customer_id = order['customer_id']
    message = f"Order confirmed: ${total}"
    send_email(customer_id, message)
```

### Ideal Argument Count

**0 arguments (niladic)**: Best
**1 argument (monadic)**: Good
**2 arguments (dyadic)**: OK
**3 arguments (triadic)**: Questionable
**4+ arguments (polyadic)**: Avoid

**Bad:**
```java
public void createUser(String firstName, String lastName, String email,
                       String phone, String address, String city,
                       String state, String zipCode) {
    // Too many parameters!
}
```

**Good:**
```java
public class UserData {
    private String firstName;
    private String lastName;
    private String email;
    private String phone;
    private Address address;

    // Getters and setters...
}

public class Address {
    private String street;
    private String city;
    private String state;
    private String zipCode;
}

public void createUser(UserData userData) {
    // Single parameter (object)
}
```

### No Side Effects

A function should do what its name says, nothing more.

**Bad (hidden side effect):**
```javascript
function checkPassword(username, password) {
    const user = db.getUser(username);
    if (user.password === hash(password)) {
        Session.initialize();  // Side effect! Not mentioned in name
        return true;
    }
    return false;
}
```

**Good:**
```javascript
function isPasswordCorrect(username, password) {
    const user = db.getUser(username);
    return user.password === hash(password);
}

// Caller controls side effects
if (isPasswordCorrect(username, password)) {
    Session.initialize();
    redirectToHomePage();
}
```

### Command-Query Separation

A function should either **do something** (command) or **return something** (query), but not both.

**Bad (does both):**
```python
def set_and_check_attribute(name, value):
    """Sets attribute and returns True if it was changed."""
    old_value = get_attribute(name)
    set_attribute(name, value)
    return old_value != value

# Confusing usage:
if set_and_check_attribute('status', 'active'):
    # Did we set it? Or just check it?
    pass
```

**Good (separate command and query):**
```python
def set_attribute(name, value):
    """Sets attribute (command)."""
    attributes[name] = value

def has_attribute_changed(name, new_value):
    """Checks if value would change (query)."""
    return get_attribute(name) != new_value

# Clear usage:
if has_attribute_changed('status', 'active'):
    set_attribute('status', 'active')
```

### DRY: Don't Repeat Yourself

**Bad:**
```java
public void printUserReport(User user) {
    System.out.println("Name: " + user.getFirstName() + " " + user.getLastName());
    System.out.println("Email: " + user.getEmail());
    System.out.println("Phone: " + user.getPhone());
}

public void emailUserDetails(User user) {
    String details = "Name: " + user.getFirstName() + " " + user.getLastName() + "\n";
    details += "Email: " + user.getEmail() + "\n";
    details += "Phone: " + user.getPhone();
    sendEmail(user.getEmail(), details);
}
```

**Good:**
```java
public String formatUserDetails(User user) {
    return String.format("Name: %s %s\nEmail: %s\nPhone: %s",
        user.getFirstName(), user.getLastName(),
        user.getEmail(), user.getPhone());
}

public void printUserReport(User user) {
    System.out.println(formatUserDetails(user));
}

public void emailUserDetails(User user) {
    sendEmail(user.getEmail(), formatUserDetails(user));
}
```

## Code Smells

Code smells are indicators of potential problems. They don't always mean bad code, but they deserve attention.

### Long Method

**Smell:**
```python
def process_payment(order):
    # 200 lines of code...
    pass
```

**Refactoring:** Extract Method

```python
def process_payment(order):
    validate_payment_info(order)
    charge_amount = calculate_charge(order)
    transaction_id = charge_credit_card(charge_amount, order.card)
    update_inventory(order)
    send_confirmation(order, transaction_id)
    return transaction_id
```

### Large Class (God Class)

**Smell:**
```java
public class User {
    // Authentication
    public boolean login(String password) { }
    public void logout() { }

    // Profile management
    public void updateProfile() { }
    public void uploadAvatar() { }

    // Permissions
    public boolean hasPermission(String permission) { }
    public void grantPermission(String permission) { }

    // Email
    public void sendWelcomeEmail() { }
    public void sendPasswordResetEmail() { }

    // Analytics
    public void trackLogin() { }
    public void trackPageView() { }

    // ... 50 more methods
}
```

**Refactoring:** Extract Class

```java
public class User {
    private UserProfile profile;
    private UserAuthentication auth;
    private UserPermissions permissions;

    // Minimal interface
}

public class UserAuthentication {
    public boolean login(String password) { }
    public void logout() { }
}

public class UserProfile {
    public void update() { }
    public void uploadAvatar() { }
}

public class UserPermissions {
    public boolean has(String permission) { }
    public void grant(String permission) { }
}
```

### Feature Envy

A method that uses data/methods from another class more than its own.

**Smell:**
```python
class Order:
    def __init__(self, customer):
        self.customer = customer

    def get_discount(self):
        # Uses customer data extensively
        if self.customer.is_premium():
            if self.customer.years_active > 5:
                return 0.20
            else:
                return 0.10
        else:
            return 0.05
```

**Refactoring:** Move Method

```python
class Customer:
    def get_discount_rate(self):
        if self.is_premium():
            if self.years_active > 5:
                return 0.20
            else:
                return 0.10
        else:
            return 0.05

class Order:
    def __init__(self, customer):
        self.customer = customer

    def get_discount(self):
        return self.customer.get_discount_rate()
```

### Data Clumps

Groups of data that always appear together.

**Smell:**
```javascript
function createUser(firstName, lastName, email, street, city, state, zip) {
    // ...
}

function updateUser(userId, firstName, lastName, email, street, city, state, zip) {
    // ...
}
```

**Refactoring:** Introduce Parameter Object

```javascript
class Address {
    constructor(street, city, state, zip) {
        this.street = street;
        this.city = city;
        this.state = state;
        this.zip = zip;
    }
}

class UserInfo {
    constructor(firstName, lastName, email, address) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.address = address;
    }
}

function createUser(userInfo) {
    // ...
}

function updateUser(userId, userInfo) {
    // ...
}
```

### Shotgun Surgery

A single change requires modifications in many classes.

**Smell:**
```java
// Adding a new payment method requires changes in:
class Order { }          // Add payment type check
class PaymentForm { }    // Add UI for new payment
class PaymentValidator { } // Add validation
class Receipt { }        // Add receipt format
class Analytics { }      // Add tracking
```

**Refactoring:** Move related changes to one place (Strategy pattern)

```java
interface PaymentMethod {
    void process(double amount);
    String getReceiptFormat();
    void validate();
    void track();
}

class CreditCardPayment implements PaymentMethod { }
class PayPalPayment implements PaymentMethod { }
class CryptoPayment implements PaymentMethod { }

// Now adding a new payment method only requires one new class
```

### Divergent Change

One class is commonly changed in different ways for different reasons.

**Smell:**
```python
class Report:
    def generate(self):
        data = self.fetch_data()  # Database logic (reason 1)
        formatted = self.format(data)  # Formatting logic (reason 2)
        self.send_email(formatted)  # Email logic (reason 3)
```

**Refactoring:** Extract Class (Single Responsibility Principle)

```python
class ReportDataFetcher:
    def fetch(self):
        # Database logic only
        pass

class ReportFormatter:
    def format(self, data):
        # Formatting logic only
        pass

class ReportEmailer:
    def send(self, report):
        # Email logic only
        pass

class ReportGenerator:
    def __init__(self, fetcher, formatter, emailer):
        self.fetcher = fetcher
        self.formatter = formatter
        self.emailer = emailer

    def generate(self):
        data = self.fetcher.fetch()
        formatted = self.formatter.format(data)
        self.emailer.send(formatted)
```

### Primitive Obsession

Using primitives instead of small objects for simple tasks.

**Smell:**
```cpp
void sendMessage(std::string phoneNumber) {
    // Is phoneNumber valid? No type safety
    if (phoneNumber.length() != 10) {
        throw std::invalid_argument("Invalid phone");
    }
    // ...
}

sendMessage("invalid");  // Compiles, but wrong
```

**Refactoring:** Replace Data Value with Object

```cpp
class PhoneNumber {
private:
    std::string number;

public:
    PhoneNumber(const std::string& num) {
        if (num.length() != 10 || !isDigits(num)) {
            throw std::invalid_argument("Invalid phone number");
        }
        number = num;
    }

    std::string toString() const { return number; }

private:
    bool isDigits(const std::string& str) const {
        return std::all_of(str.begin(), str.end(), ::isdigit);
    }
};

void sendMessage(const PhoneNumber& phoneNumber) {
    // Type safety! phoneNumber is guaranteed valid
}

// sendMessage("invalid");  // Compiler error!
sendMessage(PhoneNumber("1234567890"));  // OK
```

### Switch Statements

**Smell:**
```java
public double calculatePay(Employee employee) {
    switch (employee.type) {
        case HOURLY:
            return employee.hours * employee.hourlyRate;
        case SALARIED:
            return employee.monthlySalary;
        case COMMISSIONED:
            return employee.baseSalary + employee.commission;
        default:
            throw new IllegalArgumentException("Unknown employee type");
    }
}

// Problem: This switch appears in multiple places!
// If we add a new employee type, we must update all switches.
```

**Refactoring:** Replace Conditional with Polymorphism

```java
abstract class Employee {
    public abstract double calculatePay();
}

class HourlyEmployee extends Employee {
    private double hours;
    private double hourlyRate;

    public double calculatePay() {
        return hours * hourlyRate;
    }
}

class SalariedEmployee extends Employee {
    private double monthlySalary;

    public double calculatePay() {
        return monthlySalary;
    }
}

class CommissionedEmployee extends Employee {
    private double baseSalary;
    private double commission;

    public double calculatePay() {
        return baseSalary + commission;
    }
}

// Adding a new type only requires a new class
```

### Speculative Generality

**Smell:**
```python
class AbstractFactoryProviderSingleton:
    """Might need this someday..."""
    pass

def calculate_with_future_algorithm_support(data, algorithm='current'):
    # Supports 5 algorithms we might need in the future
    pass
```

**Refactoring:** YAGNI (You Aren't Gonna Need It) — delete it!

```python
def calculate(data):
    # Implement what you need now
    pass

# Add algorithms when actually needed
```

### Dead Code

**Smell:**
```javascript
function processOrder(order) {
    // validateAddress(order.address);  // Commented out
    calculateTotal(order);
    // if (false) {  // Always false
    //     applyOldDiscountLogic(order);
    // }
}
```

**Refactoring:** Delete it! Version control remembers.

```javascript
function processOrder(order) {
    calculateTotal(order);
}
```

## Refactoring Techniques

### Extract Method

**Before:**
```python
def print_invoice(invoice):
    print("**** Invoice ****")
    print(f"Customer: {invoice['customer']}")
    print(f"Amount: ${invoice['amount']}")

    # Calculate discount
    discount = 0
    if invoice['amount'] > 1000:
        discount = invoice['amount'] * 0.1

    print(f"Discount: ${discount}")
    print(f"Total: ${invoice['amount'] - discount}")
```

**After:**
```python
def print_invoice(invoice):
    print_header()
    print_customer_info(invoice)
    discount = calculate_discount(invoice)
    print_amounts(invoice, discount)

def print_header():
    print("**** Invoice ****")

def print_customer_info(invoice):
    print(f"Customer: {invoice['customer']}")
    print(f"Amount: ${invoice['amount']}")

def calculate_discount(invoice):
    if invoice['amount'] > 1000:
        return invoice['amount'] * 0.1
    return 0

def print_amounts(invoice, discount):
    print(f"Discount: ${discount}")
    print(f"Total: ${invoice['amount'] - discount}")
```

### Rename Variable/Method

**Before:**
```java
public void calc(int d) {
    int r = d * 0.1;
    System.out.println(r);
}
```

**After:**
```java
public void calculateDiscount(int orderAmount) {
    int discountAmount = orderAmount * 0.1;
    System.out.println(discountAmount);
}
```

### Introduce Explaining Variable

**Before:**
```javascript
if ((platform.toUpperCase().includes('MAC') ||
     platform.toUpperCase().includes('LINUX')) &&
    browser.toUpperCase().includes('CHROME')) {
    // ...
}
```

**After:**
```javascript
const isMacOrLinux = platform.toUpperCase().includes('MAC') ||
                     platform.toUpperCase().includes('LINUX');
const isChrome = browser.toUpperCase().includes('CHROME');

if (isMacOrLinux && isChrome) {
    // ...
}
```

### Replace Magic Number with Named Constant

**Before:**
```cpp
double calculateMonthlyPayment(double principal, double rate, int years) {
    return principal * rate / 12 / (1 - pow(1 + rate / 12, -years * 12));
}
```

**After:**
```cpp
const int MONTHS_PER_YEAR = 12;

double calculateMonthlyPayment(double principal, double annualRate, int years) {
    double monthlyRate = annualRate / MONTHS_PER_YEAR;
    int totalMonths = years * MONTHS_PER_YEAR;
    return principal * monthlyRate / (1 - pow(1 + monthlyRate, -totalMonths));
}
```

### Move Method

Move a method to the class where it belongs.

**Before:**
```python
class Customer:
    def __init__(self, name):
        self.name = name

class Order:
    def __init__(self, customer, amount):
        self.customer = customer
        self.amount = amount

    def get_discount(self):
        # Uses customer data, not order data
        if self.customer.name.startswith('VIP'):
            return 0.2
        return 0.1
```

**After:**
```python
class Customer:
    def __init__(self, name):
        self.name = name

    def get_discount_rate(self):
        if self.name.startswith('VIP'):
            return 0.2
        return 0.1

class Order:
    def __init__(self, customer, amount):
        self.customer = customer
        self.amount = amount

    def get_discount(self):
        return self.amount * self.customer.get_discount_rate()
```

## Principles

### DRY: Don't Repeat Yourself

Every piece of knowledge should have a single, unambiguous representation.

### KISS: Keep It Simple, Stupid

Simplicity should be a key goal. Avoid unnecessary complexity.

**Complex:**
```python
class AbstractFactoryBuilderProvider:
    @staticmethod
    def create_instance_with_dependency_injection(config):
        # 50 lines of complex logic
        pass
```

**Simple:**
```python
def create_user(name, email):
    return User(name, email)
```

### YAGNI: You Aren't Gonna Need It

Don't implement features until they're actually needed.

**YAGNI violation:**
```java
class User {
    // Just in case we need these someday...
    private String backupEmail;
    private String alternatePhone;
    private String preferredLanguage;  // We only support English
    private boolean isVerified;  // No verification process yet
}
```

### Principle of Least Surprise

Code should behave as users expect.

**Surprising:**
```javascript
function getUsers() {
    const users = db.query('SELECT * FROM users');
    users.forEach(u => u.lastAccessed = Date.now());  // Side effect!
    db.update(users);
    return users;
}
```

**Unsurprising:**
```javascript
function getUsers() {
    return db.query('SELECT * FROM users');
}

function updateLastAccessed(users) {
    users.forEach(u => u.lastAccessed = Date.now());
    db.update(users);
}
```

### Boy Scout Rule

**"Leave the code better than you found it."**

Even small improvements accumulate:
- Rename a confusing variable
- Extract a long method
- Add a missing comment
- Delete dead code

## Comments

### Good Comments

**Legal comments:**
```java
// Copyright (C) 2024 Company Inc. All rights reserved.
```

**Informative comments:**
```python
# Returns a list of tuples: (user_id, username, last_login)
def get_active_users():
    pass
```

**Explanation of intent:**
```cpp
// We use a hash instead of array because lookup needs to be O(1)
std::unordered_map<int, User> userCache;
```

**Clarification:**
```javascript
// Third-party API returns timestamps in milliseconds, not seconds
const timestampSeconds = apiResponse.timestamp / 1000;
```

**TODO comments:**
```python
# TODO: Add caching to improve performance (issue #1234)
def fetch_user_data(user_id):
    pass
```

### Bad Comments

**Redundant comments:**
```java
// Get the name
public String getName() {
    return name;
}

// Increment i
i++;
```

**Misleading comments:**
```python
# Returns the user's age
def get_user_info(user_id):
    return db.get_user(user_id)  # Returns entire user object, not just age!
```

**Noise comments:**
```cpp
/**
 * Constructor.
 */
public User() {
}
```

**Commented-out code:**
```javascript
function processOrder(order) {
    // validateAddress(order);  // Don't comment out, delete it!
    // applyDiscount(order);
    calculateTotal(order);
}
```

**Journal comments:**
```java
// 2024-01-15: Added validation - John
// 2024-01-20: Fixed bug - Sarah
// 2024-02-01: Refactored - Mike
public void processPayment() {
    // Use version control for history!
}
```

### Self-Documenting Code

Prefer code that explains itself:

**Needs comments:**
```python
# Check if user is eligible for discount
if u.t == 'premium' and u.yrs > 2:
    pass
```

**Self-documenting:**
```python
def is_eligible_for_loyalty_discount(user):
    return user.type == 'premium' and user.years_active > 2

if is_eligible_for_loyalty_discount(user):
    pass
```

## Formatting

Consistent formatting improves readability.

### Vertical Formatting

**Newspaper metaphor**: Most important information at top.

```python
# Public interface first
class OrderService:
    def process_order(self, order):
        self._validate(order)
        total = self._calculate_total(order)
        self._save(order)
        return total

    # Private helpers below
    def _validate(self, order):
        pass

    def _calculate_total(self, order):
        pass

    def _save(self, order):
        pass
```

### Horizontal Formatting

Keep lines short (80-120 characters).

**Bad:**
```java
public void createUser(String firstName, String lastName, String email, String phone, Address address, List<String> permissions) { }
```

**Good:**
```java
public void createUser(
    String firstName,
    String lastName,
    String email,
    String phone,
    Address address,
    List<String> permissions
) {
    // ...
}
```

### Team Conventions

Use linters and formatters:
- **JavaScript**: ESLint, Prettier
- **Python**: pylint, black
- **Java**: Checkstyle, Google Java Format
- **C++**: clang-format

## Summary

| Principle | Guideline |
|-----------|-----------|
| Names | Reveal intent, avoid disinformation, use pronounceable names |
| Functions | Small, do one thing, few arguments, no side effects |
| Comments | Explain why, not what; prefer self-documenting code |
| DRY | Don't repeat yourself |
| KISS | Keep it simple |
| YAGNI | You aren't gonna need it |
| Boy Scout Rule | Leave code better than you found it |

**Code smells to watch for:**
- Long methods/large classes
- Feature envy
- Data clumps
- Shotgun surgery
- Primitive obsession
- Switch statements (consider polymorphism)
- Dead code

## Exercises

### Exercise 1: Refactor This Code

```python
def p(o):
    t = 0
    for i in o['items']:
        t += i['p'] * i['q']
    if o.get('c'):
        d = 0.1 if o['c'] == 'SAVE10' else 0.2 if o['c'] == 'SAVE20' else 0
        t = t - (t * d)
    return t
```

Refactor this to follow clean code principles.

### Exercise 2: Identify Code Smells

Review this code and identify at least 5 code smells:

```java
public class UserManager {
    public void process(String type, String id, String data) {
        if (type.equals("create")) {
            // 50 lines of user creation logic
        } else if (type.equals("update")) {
            // 50 lines of user update logic
        } else if (type.equals("delete")) {
            // 50 lines of user deletion logic
        }
    }

    public String getUserFirstName(String id) {
        String[] data = db.query("SELECT * FROM users WHERE id = " + id);
        return data[0];
    }

    public String getUserLastName(String id) {
        String[] data = db.query("SELECT * FROM users WHERE id = " + id);
        return data[1];
    }

    public String getUserEmail(String id) {
        String[] data = db.query("SELECT * FROM users WHERE id = " + id);
        return data[2];
    }
}
```

### Exercise 3: Write Clean Functions

Rewrite this function to follow the principles of clean code:

```javascript
function validateAndProcessForm(formData) {
    if (!formData.email || !formData.email.includes('@')) {
        alert('Invalid email');
        return false;
    }
    if (!formData.password || formData.password.length < 8) {
        alert('Password too short');
        return false;
    }
    if (formData.password !== formData.confirmPassword) {
        alert('Passwords do not match');
        return false;
    }

    const user = {
        email: formData.email,
        password: hash(formData.password),
        created: Date.now()
    };

    db.save(user);
    sendWelcomeEmail(user.email);
    redirectToHomePage();
    return true;
}
```

### Exercise 4: Apply Boy Scout Rule

Find a file in your codebase and make it slightly better:
1. Rename one poorly-named variable
2. Extract one long method
3. Remove one piece of dead code
4. Add one explanatory variable

Document your changes.

### Exercise 5: Code Review Checklist

Create a code review checklist based on this lesson. Include at least:
- 5 naming checks
- 5 function checks
- 5 code smell checks
- 3 comment checks

---

**Previous**: [Design Patterns](./07_Design_Patterns.md) | **Next**: [Error Handling](./09_Error_Handling.md)
