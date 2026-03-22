# 클린 코드와 코드 스멜

> **주제**: Programming
> **레슨**: 8 of 16
> **선수지식**: 최소 하나 이상의 언어에서 프로그래밍 경험, 함수와 클래스 숙지
> **목표**: 모범 사례를 따르고 코드 스멜을 인식하며 효과적인 리팩토링 기법을 적용하여 깨끗하고 유지보수 가능한 코드 작성 학습

## 소개

"바보도 컴퓨터가 이해할 수 있는 코드를 작성할 수 있습니다. 좋은 프로그래머는 사람이 이해할 수 있는 코드를 작성합니다." — Martin Fowler

클린 코드(Clean Code)는 읽기 쉽고, 이해하기 쉬우며, 수정하기 쉬운 코드입니다. 인지 부하를 최소화하고, 버그를 줄이며, 협업을 더 쉽게 만듭니다. 이 레슨에서는 Robert C. Martin의 *Clean Code*, Martin Fowler의 *Refactoring*, 그리고 기타 업계 모범 사례의 원칙들을 다룹니다.

## 나쁜 코드의 비용

**기술 부채(Technical Debt)**는 지름길을 택할 때 누적됩니다:

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

**클린 버전:**
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

## 의미 있는 이름

이름은 우리가 의도를 전달하는 주요 방법입니다. 좋은 이름은 코드를 자체 문서화하게 만듭니다.

### 의도를 드러내는 이름(Intention-Revealing Names)

**나쁜 예:**
```java
int d;  // elapsed time in days
```

**좋은 예:**
```java
int elapsedTimeInDays;
int daysSinceCreation;
int daysSinceModification;
```

### 잘못된 정보 피하기(Avoid Disinformation)

**나쁜 예:**
```javascript
const accountList = {};  // Not a list, it's an object!

function hp() { }  // What does hp mean?

const theAccounts = [];  // "the" adds no information
```

**좋은 예:**
```javascript
const accountMap = {};

function calculateHorsePower() { }
// Or if context is clear:
function calculatePower() { }

const accounts = [];
```

### 발음 가능한 이름 사용(Use Pronounceable Names)

**나쁜 예:**
```python
genymdhms  # generation year, month, day, hour, minute, second
```

**좋은 예:**
```python
generation_timestamp
```

### 검색 가능한 이름 사용(Use Searchable Names)

**나쁜 예:**
```cpp
for (int i = 0; i < 7; i++) {  // What is 7?
    // ...
}
```

**좋은 예:**
```cpp
const int DAYS_IN_WEEK = 7;

for (int day = 0; day < DAYS_IN_WEEK; day++) {
    // ...
}
```

### 클래스 이름: 명사

클래스는 사물(명사)을 나타냅니다:

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

### 메서드 이름: 동사

메서드는 동작(동사)을 수행합니다:

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

### 인코딩 피하기(Avoid Encodings)

**나쁜 예 (헝가리안 표기법):**
```cpp
int iCount;
string strName;
bool bIsActive;
```

**좋은 예:**
```cpp
int count;
string name;
bool isActive;
```

### 개념당 한 단어(One Word Per Concept)

하나의 용어를 선택하고 일관되게 사용하세요:

**일관성 없음:**
```java
class UserFetcher { }
class AccountRetriever { }
class ProductGetter { }
```

**일관성 있음:**
```java
class UserRepository { }
class AccountRepository { }
class ProductRepository { }
```

### 불필요한 단어 피하기(Avoid Noise Words)

**나쁜 예:**
```javascript
const productData = {};
const productInfo = {};
const theProduct = {};

function getProductData() { }
function getProductInfo() { }
```

`Data`와 `Info`의 차이는? 없습니다!

**좋은 예:**
```javascript
const product = {};

function getProduct() { }
```

## 함수

### 작은 함수(Small Functions)

함수는 **한 가지만** 해야 하고, **잘** 해야 하며, **오직 그것만** 해야 합니다.

**나쁜 예 (너무 많은 일을 함):**
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

**좋은 예 (단일 책임):**
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

### 이상적인 인자 개수(Ideal Argument Count)

**0개 인자 (무항, niladic)**: 최상
**1개 인자 (단항, monadic)**: 좋음
**2개 인자 (이항, dyadic)**: 괜찮음
**3개 인자 (삼항, triadic)**: 의심스러움
**4개 이상 인자 (다항, polyadic)**: 피하기

**나쁜 예:**
```java
public void createUser(String firstName, String lastName, String email,
                       String phone, String address, String city,
                       String state, String zipCode) {
    // Too many parameters!
}
```

**좋은 예:**
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

### 부수 효과 없음(No Side Effects)

함수는 이름이 말하는 것만 해야 하며, 그 이상은 하지 말아야 합니다.

**나쁜 예 (숨겨진 부수 효과):**
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

**좋은 예:**
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

### 명령-쿼리 분리(Command-Query Separation)

함수는 **무언가를 하거나** (명령, command) **무언가를 반환하거나** (쿼리, query) 해야 하며, 둘 다 하면 안 됩니다.

**나쁜 예 (둘 다 함):**
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

**좋은 예 (명령과 쿼리 분리):**
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

### DRY: 반복하지 마세요(Don't Repeat Yourself)

**나쁜 예:**
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

**좋은 예:**
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

## 코드 스멜(Code Smells)

코드 스멜은 잠재적 문제의 지표입니다. 항상 나쁜 코드를 의미하는 것은 아니지만 주의가 필요합니다.

### 긴 메서드(Long Method)

**스멜:**
```python
def process_payment(order):
    # 200 lines of code...
    pass
```

**리팩토링:** 메서드 추출(Extract Method)

```python
def process_payment(order):
    validate_payment_info(order)
    charge_amount = calculate_charge(order)
    transaction_id = charge_credit_card(charge_amount, order.card)
    update_inventory(order)
    send_confirmation(order, transaction_id)
    return transaction_id
```

### 거대한 클래스(Large Class, God Class)

**스멜:**
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

**리팩토링:** 클래스 추출(Extract Class)

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

### 기능 선망(Feature Envy)

자신의 클래스보다 다른 클래스의 데이터/메서드를 더 많이 사용하는 메서드.

**스멜:**
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

**리팩토링:** 메서드 이동(Move Method)

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

### 데이터 덩어리(Data Clumps)

항상 함께 나타나는 데이터 그룹.

**스멜:**
```javascript
function createUser(firstName, lastName, email, street, city, state, zip) {
    // ...
}

function updateUser(userId, firstName, lastName, email, street, city, state, zip) {
    // ...
}
```

**리팩토링:** 매개변수 객체 도입(Introduce Parameter Object)

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

### 산탄총 수술(Shotgun Surgery)

하나의 변경이 많은 클래스의 수정을 요구.

**스멜:**
```java
// Adding a new payment method requires changes in:
class Order { }          // Add payment type check
class PaymentForm { }    // Add UI for new payment
class PaymentValidator { } // Add validation
class Receipt { }        // Add receipt format
class Analytics { }      // Add tracking
```

**리팩토링:** 관련 변경을 한 곳으로 이동 (전략 패턴)

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

### 발산적 변경(Divergent Change)

하나의 클래스가 다른 이유로 여러 방식으로 자주 변경됨.

**스멜:**
```python
class Report:
    def generate(self):
        data = self.fetch_data()  # Database logic (reason 1)
        formatted = self.format(data)  # Formatting logic (reason 2)
        self.send_email(formatted)  # Email logic (reason 3)
```

**리팩토링:** 클래스 추출 (단일 책임 원칙)

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

### 기본형 집착(Primitive Obsession)

단순한 작업에 작은 객체 대신 원시 타입 사용.

**스멜:**
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

**리팩토링:** 값 객체로 대체(Replace Data Value with Object)

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

### Switch 문(Switch Statements)

**스멜:**
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

**리팩토링:** 다형성으로 조건문 대체(Replace Conditional with Polymorphism)

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

### 추측성 일반화(Speculative Generality)

**스멜:**
```python
class AbstractFactoryProviderSingleton:
    """Might need this someday..."""
    pass

def calculate_with_future_algorithm_support(data, algorithm='current'):
    # Supports 5 algorithms we might need in the future
    pass
```

**리팩토링:** YAGNI (필요하지 않을 것이다) — 삭제하세요!

```python
def calculate(data):
    # Implement what you need now
    pass

# Add algorithms when actually needed
```

### 죽은 코드(Dead Code)

**스멜:**
```javascript
function processOrder(order) {
    // validateAddress(order.address);  // Commented out
    calculateTotal(order);
    // if (false) {  // Always false
    //     applyOldDiscountLogic(order);
    // }
}
```

**리팩토링:** 삭제하세요! 버전 관리가 기억합니다.

```javascript
function processOrder(order) {
    calculateTotal(order);
}
```

## 리팩토링 기법

### 메서드 추출(Extract Method)

**이전:**
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

**이후:**
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

### 변수/메서드 이름 변경(Rename Variable/Method)

**이전:**
```java
public void calc(int d) {
    int r = d * 0.1;
    System.out.println(r);
}
```

**이후:**
```java
public void calculateDiscount(int orderAmount) {
    int discountAmount = orderAmount * 0.1;
    System.out.println(discountAmount);
}
```

### 설명 변수 도입(Introduce Explaining Variable)

**이전:**
```javascript
if ((platform.toUpperCase().includes('MAC') ||
     platform.toUpperCase().includes('LINUX')) &&
    browser.toUpperCase().includes('CHROME')) {
    // ...
}
```

**이후:**
```javascript
const isMacOrLinux = platform.toUpperCase().includes('MAC') ||
                     platform.toUpperCase().includes('LINUX');
const isChrome = browser.toUpperCase().includes('CHROME');

if (isMacOrLinux && isChrome) {
    // ...
}
```

### 매직 넘버를 명명된 상수로 대체(Replace Magic Number with Named Constant)

**이전:**
```cpp
double calculateMonthlyPayment(double principal, double rate, int years) {
    return principal * rate / 12 / (1 - pow(1 + rate / 12, -years * 12));
}
```

**이후:**
```cpp
const int MONTHS_PER_YEAR = 12;

double calculateMonthlyPayment(double principal, double annualRate, int years) {
    double monthlyRate = annualRate / MONTHS_PER_YEAR;
    int totalMonths = years * MONTHS_PER_YEAR;
    return principal * monthlyRate / (1 - pow(1 + monthlyRate, -totalMonths));
}
```

### 메서드 이동(Move Method)

메서드를 속한 클래스로 이동.

**이전:**
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

**이후:**
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

## 원칙

### DRY: 반복하지 마세요(Don't Repeat Yourself)

모든 지식 조각은 단일하고 명확한 표현을 가져야 합니다.

### KISS: 단순하게 유지하세요(Keep It Simple, Stupid)

단순성은 핵심 목표여야 합니다. 불필요한 복잡성을 피하세요.

**복잡함:**
```python
class AbstractFactoryBuilderProvider:
    @staticmethod
    def create_instance_with_dependency_injection(config):
        # 50 lines of complex logic
        pass
```

**단순함:**
```python
def create_user(name, email):
    return User(name, email)
```

### YAGNI: 필요하지 않을 것이다(You Aren't Gonna Need It)

실제로 필요할 때까지 기능을 구현하지 마세요.

**YAGNI 위반:**
```java
class User {
    // Just in case we need these someday...
    private String backupEmail;
    private String alternatePhone;
    private String preferredLanguage;  // We only support English
    private boolean isVerified;  // No verification process yet
}
```

### 최소 놀람 원칙(Principle of Least Surprise)

코드는 사용자가 예상하는 대로 동작해야 합니다.

**놀라움:**
```javascript
function getUsers() {
    const users = db.query('SELECT * FROM users');
    users.forEach(u => u.lastAccessed = Date.now());  // Side effect!
    db.update(users);
    return users;
}
```

**놀람 없음:**
```javascript
function getUsers() {
    return db.query('SELECT * FROM users');
}

function updateLastAccessed(users) {
    users.forEach(u => u.lastAccessed = Date.now());
    db.update(users);
}
```

### 보이스카우트 규칙(Boy Scout Rule)

**"발견한 코드보다 더 나은 상태로 남겨두세요."**

작은 개선도 누적됩니다:
- 혼란스러운 변수 이름 변경
- 긴 메서드 추출
- 누락된 주석 추가
- 죽은 코드 삭제

## 주석(Comments)

### 좋은 주석

**법적 주석:**
```java
// Copyright (C) 2024 Company Inc. All rights reserved.
```

**정보 제공 주석:**
```python
# Returns a list of tuples: (user_id, username, last_login)
def get_active_users():
    pass
```

**의도 설명:**
```cpp
// We use a hash instead of array because lookup needs to be O(1)
std::unordered_map<int, User> userCache;
```

**명확화:**
```javascript
// Third-party API returns timestamps in milliseconds, not seconds
const timestampSeconds = apiResponse.timestamp / 1000;
```

**TODO 주석:**
```python
# TODO: Add caching to improve performance (issue #1234)
def fetch_user_data(user_id):
    pass
```

### 나쁜 주석

**중복 주석:**
```java
// Get the name
public String getName() {
    return name;
}

// Increment i
i++;
```

**오해의 소지가 있는 주석:**
```python
# Returns the user's age
def get_user_info(user_id):
    return db.get_user(user_id)  # Returns entire user object, not just age!
```

**잡음 주석:**
```cpp
/**
 * Constructor.
 */
public User() {
}
```

**주석 처리된 코드:**
```javascript
function processOrder(order) {
    // validateAddress(order);  // Don't comment out, delete it!
    // applyDiscount(order);
    calculateTotal(order);
}
```

**일지 주석:**
```java
// 2024-01-15: Added validation - John
// 2024-01-20: Fixed bug - Sarah
// 2024-02-01: Refactored - Mike
public void processPayment() {
    // Use version control for history!
}
```

### 자체 문서화 코드(Self-Documenting Code)

스스로 설명하는 코드를 선호하세요:

**주석이 필요함:**
```python
# Check if user is eligible for discount
if u.t == 'premium' and u.yrs > 2:
    pass
```

**자체 문서화:**
```python
def is_eligible_for_loyalty_discount(user):
    return user.type == 'premium' and user.years_active > 2

if is_eligible_for_loyalty_discount(user):
    pass
```

## 형식(Formatting)

일관된 형식은 가독성을 향상시킵니다.

### 수직 형식(Vertical Formatting)

**신문 은유**: 가장 중요한 정보가 위에.

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

### 수평 형식(Horizontal Formatting)

줄을 짧게 유지 (80-120자).

**나쁜 예:**
```java
public void createUser(String firstName, String lastName, String email, String phone, Address address, List<String> permissions) { }
```

**좋은 예:**
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

### 팀 규약(Team Conventions)

린터와 포매터 사용:
- **JavaScript**: ESLint, Prettier
- **Python**: pylint, black
- **Java**: Checkstyle, Google Java Format
- **C++**: clang-format

## 요약

| 원칙 | 지침 |
|-----------|-----------|
| 이름 | 의도 드러내기, 잘못된 정보 피하기, 발음 가능한 이름 사용 |
| 함수 | 작게, 한 가지만, 적은 인자, 부수 효과 없음 |
| 주석 | 왜인지 설명, 무엇인지는 설명 안 함; 자체 문서화 코드 선호 |
| DRY | 반복하지 마세요 |
| KISS | 단순하게 유지 |
| YAGNI | 필요하지 않을 것이다 |
| 보이스카우트 규칙 | 발견한 것보다 나은 상태로 남겨두기 |

**주의할 코드 스멜:**
- 긴 메서드/거대한 클래스
- 기능 선망
- 데이터 덩어리
- 산탄총 수술
- 기본형 집착
- Switch 문 (다형성 고려)
- 죽은 코드

## 연습 문제

### 연습 문제 1: 이 코드 리팩토링

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

클린 코드 원칙을 따르도록 리팩토링하세요.

### 연습 문제 2: 코드 스멜 식별

이 코드를 검토하고 최소 5개의 코드 스멜을 식별하세요:

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

### 연습 문제 3: 클린 함수 작성

클린 코드 원칙을 따르도록 이 함수를 다시 작성하세요:

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

### 연습 문제 4: 보이스카우트 규칙 적용

코드베이스에서 파일을 찾아 약간 개선하세요:
1. 이름이 잘못 지어진 변수 하나 이름 변경
2. 긴 메서드 하나 추출
3. 죽은 코드 하나 제거
4. 설명 변수 하나 추가

변경 사항을 문서화하세요.

### 연습 문제 5: 코드 리뷰 체크리스트

이 레슨을 기반으로 코드 리뷰 체크리스트를 만드세요. 최소한 포함할 것:
- 5개의 이름 지정 검사
- 5개의 함수 검사
- 5개의 코드 스멜 검사
- 3개의 주석 검사

---

**이전**: [디자인 패턴](07_Design_Patterns.md)
**다음**: [에러 처리 전략(Error Handling Strategies)](09_Error_Handling.md)
