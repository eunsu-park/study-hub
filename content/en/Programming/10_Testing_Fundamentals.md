# Testing Fundamentals

**Previous**: [Error Handling Strategies](./09_Error_Handling.md) | **Next**: [Debugging & Profiling](./11_Debugging_and_Profiling.md)

> **Topic**: Programming
> **Lesson**: 10 of 16
> **Prerequisites**: Functions and Methods, Error Handling, Object-Oriented Programming

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the test pyramid and allocate testing effort appropriately across unit, integration, and end-to-end tests
2. Write unit tests following the Arrange-Act-Assert pattern with descriptive naming conventions in Python, JavaScript, and Java
3. Apply Test-Driven Development (TDD) using the Red-Green-Refactor cycle to implement new features
4. Distinguish between test double types (dummy, stub, mock, spy, fake) and select the right one for each testing scenario
5. Write integration tests for database interactions and REST API endpoints
6. Set up end-to-end tests using browser automation tools (Selenium, Playwright, Cypress)
7. Interpret code coverage metrics (line, branch, path) and set realistic coverage targets
8. Use property-based testing to discover edge cases that example-based tests miss

---

## Introduction

Testing is not just about finding bugs – it's about building confidence in your code, enabling refactoring, providing documentation, and improving design. Well-tested code is easier to change, maintain, and understand.

This lesson covers testing fundamentals across multiple languages and testing paradigms, from unit tests to end-to-end tests, from traditional testing to Test-Driven Development, and from mocks to property-based testing.

---

## Why Test?

### 1. Confidence

Tests give you confidence that your code works as intended and continues to work after changes.

### 2. Regression Prevention

Tests catch bugs when you introduce them, not weeks later in production.

```python
# Without tests: You change something, deploy, and hope it works
def calculate_discount(price, customer_tier):
    if customer_tier == "gold":
        return price * 0.8  # Changed from 0.9 - did we break something?
    return price

# With tests: You know immediately if you break something
def test_gold_discount():
    assert calculate_discount(100, "gold") == 80
```

### 3. Documentation

Tests document how your code is supposed to be used:

```javascript
// This test documents the API behavior better than comments
test('User registration with valid data creates a new user', async () => {
    const userData = {
        username: 'alice',
        email: 'alice@example.com',
        password: 'securepassword123'
    };

    const user = await registerUser(userData);

    expect(user.id).toBeDefined();
    expect(user.username).toBe('alice');
    expect(user.email).toBe('alice@example.com');
    expect(user.hashedPassword).not.toBe('securepassword123'); // Password should be hashed
});
```

### 4. Design Feedback

If code is hard to test, it's often poorly designed:
- Too many dependencies → tight coupling
- Functions doing too much → violation of Single Responsibility
- Hard to mock → dependency inversion needed

Testing drives better design.

---

## The Test Pyramid

The test pyramid guides how to distribute your testing effort:

```
         /\
        /  \  E2E Tests (few, slow, expensive)
       /____\
      /      \
     / Integ. \ Integration Tests (some, medium speed)
    /__________\
   /            \
  /     Unit     \ Unit Tests (many, fast, cheap)
 /________________\
```

**Distribution:**
- **70% Unit tests**: Test individual functions/classes in isolation
- **20% Integration tests**: Test components working together
- **10% E2E tests**: Test the entire system from user perspective

**Why this ratio?**
- Unit tests are fast, pinpoint failures, easy to maintain
- Integration tests catch interface issues
- E2E tests catch real user workflows but are slow and brittle

---

## Unit Testing

Unit tests verify that individual units (functions, methods, classes) work correctly in isolation.

### What is a Unit?

A "unit" is the smallest testable part of your application:
- A function
- A method
- A class
- A module (in some contexts)

### Structure: Arrange-Act-Assert (AAA)

Also called Given-When-Then:

```python
def test_shopping_cart_total():
    # Arrange (Given): Set up test data and preconditions
    cart = ShoppingCart()
    cart.add_item(Product("Book", 10.00), quantity=2)
    cart.add_item(Product("Pen", 1.50), quantity=3)

    # Act (When): Execute the behavior you're testing
    total = cart.calculate_total()

    # Assert (Then): Verify the result
    assert total == 24.50
```

### Naming Conventions

Good test names describe what they test and what the expected outcome is:

```python
# Convention: test_<what>_<condition>_<expected_result>

def test_divide_by_zero_raises_exception():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_user_login_with_wrong_password_returns_error():
    result = login("alice", "wrongpassword")
    assert result.success is False
    assert "Invalid password" in result.error_message

def test_empty_cart_has_zero_total():
    cart = ShoppingCart()
    assert cart.calculate_total() == 0
```

Alternative convention (BDD-style):

```javascript
describe('ShoppingCart', () => {
    describe('when empty', () => {
        it('should have a total of 0', () => {
            const cart = new ShoppingCart();
            expect(cart.calculateTotal()).toBe(0);
        });
    });

    describe('when items are added', () => {
        it('should calculate the correct total', () => {
            const cart = new ShoppingCart();
            cart.addItem({ name: 'Book', price: 10 }, 2);
            cart.addItem({ name: 'Pen', price: 1.5 }, 3);
            expect(cart.calculateTotal()).toBe(24.5);
        });
    });
});
```

### Examples in Multiple Languages

**Python (pytest):**

```python
import pytest
from calculator import Calculator

def test_addition():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5

def test_subtraction():
    calc = Calculator()
    result = calc.subtract(10, 4)
    assert result == 6

def test_division_by_zero():
    calc = Calculator()
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300)
])
def test_addition_parametrized(a, b, expected):
    calc = Calculator()
    assert calc.add(a, b) == expected
```

**JavaScript (Jest):**

```javascript
const Calculator = require('./calculator');

describe('Calculator', () => {
    let calc;

    beforeEach(() => {
        calc = new Calculator();
    });

    test('adds two numbers correctly', () => {
        expect(calc.add(2, 3)).toBe(5);
    });

    test('subtracts two numbers correctly', () => {
        expect(calc.subtract(10, 4)).toBe(6);
    });

    test('throws error when dividing by zero', () => {
        expect(() => calc.divide(10, 0)).toThrow('Cannot divide by zero');
    });

    test.each([
        [2, 3, 5],
        [0, 0, 0],
        [-1, 1, 0],
        [100, 200, 300]
    ])('add(%i, %i) should return %i', (a, b, expected) => {
        expect(calc.add(a, b)).toBe(expected);
    });
});
```

**Java (JUnit 5):**

```java
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    private Calculator calc;

    @BeforeEach
    void setUp() {
        calc = new Calculator();
    }

    @Test
    void testAddition() {
        assertEquals(5, calc.add(2, 3));
    }

    @Test
    void testSubtraction() {
        assertEquals(6, calc.subtract(10, 4));
    }

    @Test
    void testDivisionByZero() {
        Exception exception = assertThrows(
            IllegalArgumentException.class,
            () -> calc.divide(10, 0)
        );
        assertTrue(exception.getMessage().contains("Cannot divide by zero"));
    }

    @ParameterizedTest
    @CsvSource({
        "2, 3, 5",
        "0, 0, 0",
        "-1, 1, 0",
        "100, 200, 300"
    })
    void testAdditionParameterized(int a, int b, int expected) {
        assertEquals(expected, calc.add(a, b));
    }
}
```

---

## Test-Driven Development (TDD)

TDD is a development methodology where you write tests **before** writing the implementation code.

### The Red-Green-Refactor Cycle

```
1. RED:     Write a failing test
     ↓
2. GREEN:   Write minimal code to make it pass
     ↓
3. REFACTOR: Improve the code without changing behavior
     ↓
     (repeat)
```

### Example: Implementing a Stack with TDD

**Step 1: RED – Write a failing test**

```python
import pytest
from stack import Stack

def test_new_stack_is_empty():
    stack = Stack()
    assert stack.is_empty() == True
```

Run: ❌ FAIL (Stack class doesn't exist)

**Step 2: GREEN – Make it pass (minimal code)**

```python
class Stack:
    def is_empty(self):
        return True
```

Run: ✅ PASS

**Step 3: Add another test (RED)**

```python
def test_push_adds_element():
    stack = Stack()
    stack.push(5)
    assert stack.is_empty() == False
```

Run: ❌ FAIL (push doesn't exist, is_empty always returns True)

**Step 4: GREEN – Make it pass**

```python
class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)
```

Run: ✅ PASS

**Step 5: Add more tests and implement pop, peek, etc.**

```python
def test_pop_removes_and_returns_top_element():
    stack = Stack()
    stack.push(5)
    stack.push(10)
    assert stack.pop() == 10
    assert stack.pop() == 5
    assert stack.is_empty() == True

def test_pop_on_empty_stack_raises_error():
    stack = Stack()
    with pytest.raises(IndexError):
        stack.pop()

def test_peek_returns_top_without_removing():
    stack = Stack()
    stack.push(5)
    stack.push(10)
    assert stack.peek() == 10
    assert stack.peek() == 10  # Still there
    assert not stack.is_empty()
```

**Step 6: REFACTOR**

Now that tests are passing, refactor for clarity:

```python
class Stack:
    """A last-in-first-out (LIFO) stack data structure."""

    def __init__(self):
        self._items = []

    def is_empty(self):
        """Return True if the stack has no elements."""
        return len(self._items) == 0

    def push(self, item):
        """Add an item to the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def size(self):
        """Return the number of items in the stack."""
        return len(self._items)
```

Tests still pass ✅, but code is cleaner.

### Benefits of TDD

1. **Better design**: Writing tests first forces you to think about the interface
2. **Complete coverage**: Every line of code has a test (you wrote the test first!)
3. **Confidence**: You know the code works because you've tested it continuously
4. **Documentation**: Tests document the expected behavior
5. **Less debugging**: Bugs are caught immediately

### When TDD Works Well

- Algorithmic code with clear inputs/outputs
- Utility functions and libraries
- Bug fixes (write a test that reproduces the bug, then fix it)
- Refactoring (tests ensure behavior doesn't change)

### When TDD is Challenging

- Exploratory coding (you don't know what you're building yet)
- UI code (hard to test interactions before they exist)
- Complex integrations (many unknowns)
- Rapid prototyping

**Note:** TDD is a tool, not a religion. Use it when it helps, skip it when it hinders.

---

## Behavior-Driven Development (BDD)

BDD extends TDD with a focus on behavior from the user's perspective.

### Given/When/Then Syntax

```gherkin
Feature: User Login

  Scenario: Successful login with valid credentials
    Given a user with username "alice" and password "secret123"
    When the user attempts to log in with correct credentials
    Then the user should be logged in
    And the user should see their dashboard

  Scenario: Failed login with invalid password
    Given a user with username "alice" and password "secret123"
    When the user attempts to log in with password "wrongpassword"
    Then the user should not be logged in
    And the user should see an error message "Invalid password"
```

### Tools: Cucumber, Behave, Jest

**Python (Behave):**

```gherkin
# features/login.feature
Feature: User Login
  Scenario: Successful login
    Given a user "alice" with password "secret123"
    When I login with username "alice" and password "secret123"
    Then I should be logged in
```

```python
# features/steps/login_steps.py
from behave import given, when, then

@given('a user "{username}" with password "{password}"')
def step_create_user(context, username, password):
    context.users = {username: password}

@when('I login with username "{username}" and password "{password}"')
def step_login(context, username, password):
    expected_password = context.users.get(username)
    context.login_success = (expected_password == password)

@then('I should be logged in')
def step_verify_login(context):
    assert context.login_success, "Login should succeed"
```

**JavaScript (Jest with describe/it):**

```javascript
describe('User Login', () => {
    describe('given a user with valid credentials', () => {
        let user;

        beforeEach(() => {
            user = createUser('alice', 'secret123');
        });

        describe('when the user logs in with correct credentials', () => {
            let result;

            beforeEach(() => {
                result = login('alice', 'secret123');
            });

            it('should succeed', () => {
                expect(result.success).toBe(true);
            });

            it('should return the user object', () => {
                expect(result.user.username).toBe('alice');
            });
        });
    });
});
```

### User Stories → Acceptance Criteria → Tests

```
User Story:
  As a customer
  I want to add items to my shopping cart
  So that I can purchase multiple items at once

Acceptance Criteria:
  ✓ Items can be added to the cart
  ✓ The cart displays the correct quantity
  ✓ The cart shows the correct total price
  ✓ Items can be removed from the cart

Tests:
  test_add_item_to_cart()
  test_cart_displays_quantity()
  test_cart_calculates_total()
  test_remove_item_from_cart()
```

---

## Test Doubles

Test doubles are objects that stand in for real dependencies during testing.

### Types of Test Doubles

#### 1. Dummy

Placeholder object, never actually used:

```python
def test_send_email():
    # We don't care about the logger, but the function requires one
    dummy_logger = None
    send_email("alice@example.com", "Hello", logger=dummy_logger)
```

#### 2. Stub

Returns predetermined data:

```python
class StubUserRepository:
    def find_by_id(self, user_id):
        # Always returns the same user, regardless of ID
        return User(id=1, name="Alice", email="alice@example.com")

def test_user_service():
    user_repo = StubUserRepository()
    service = UserService(user_repo)
    user = service.get_user_details(999)  # ID doesn't matter
    assert user.name == "Alice"
```

**JavaScript:**
```javascript
const stubDatabase = {
    findUser: (id) => ({ id: 1, name: 'Alice' })  // Always returns Alice
};

test('getUserDetails returns user from database', () => {
    const service = new UserService(stubDatabase);
    const user = service.getUserDetails(999);
    expect(user.name).toBe('Alice');
});
```

#### 3. Mock

Verifies that specific methods are called with expected arguments:

```python
from unittest.mock import Mock

def test_send_welcome_email():
    # Create a mock email service
    mock_email_service = Mock()

    # Use it in the code under test
    user_service = UserService(email_service=mock_email_service)
    user_service.register_user("alice@example.com")

    # Verify the email service was called correctly
    mock_email_service.send.assert_called_once_with(
        to="alice@example.com",
        subject="Welcome!",
        body="Thank you for registering"
    )
```

**JavaScript (Jest):**
```javascript
test('registerUser sends a welcome email', () => {
    const mockEmailService = {
        send: jest.fn()
    };

    const userService = new UserService(mockEmailService);
    userService.registerUser('alice@example.com');

    expect(mockEmailService.send).toHaveBeenCalledWith({
        to: 'alice@example.com',
        subject: 'Welcome!',
        body: 'Thank you for registering'
    });
});
```

#### 4. Spy

Records how it was called, allowing post-execution verification:

```python
from unittest.mock import MagicMock

def test_logger_spy():
    logger = MagicMock()
    process_order(order_id=123, logger=logger)

    # Verify logger was called
    assert logger.info.call_count == 2
    logger.info.assert_any_call("Processing order 123")
    logger.info.assert_any_call("Order 123 completed")
```

**JavaScript:**
```javascript
test('processOrder logs progress', () => {
    const spyLogger = {
        info: jest.fn()
    };

    processOrder(123, spyLogger);

    expect(spyLogger.info).toHaveBeenCalledTimes(2);
    expect(spyLogger.info).toHaveBeenCalledWith('Processing order 123');
    expect(spyLogger.info).toHaveBeenCalledWith('Order 123 completed');
});
```

#### 5. Fake

A working implementation, but simpler than the real one:

```python
class FakeDatabase:
    """In-memory database for testing"""
    def __init__(self):
        self.users = {}
        self.next_id = 1

    def save_user(self, user):
        user.id = self.next_id
        self.users[user.id] = user
        self.next_id += 1
        return user

    def find_user(self, user_id):
        return self.users.get(user_id)

def test_user_service_with_fake_db():
    fake_db = FakeDatabase()
    service = UserService(database=fake_db)

    user = service.create_user("Alice", "alice@example.com")
    assert user.id is not None

    retrieved = service.get_user(user.id)
    assert retrieved.name == "Alice"
```

### When to Use Each

| Type | Use When | Example |
|------|----------|---------|
| **Dummy** | Parameter is required but not used | Logger that's never called |
| **Stub** | You need predetermined responses | Database returning test data |
| **Mock** | You want to verify interactions | Email service should be called |
| **Spy** | You want to observe behavior | Recording logger calls |
| **Fake** | You need a working but simpler implementation | In-memory database |

---

## Integration Testing

Integration tests verify that components work together correctly.

### Database Tests

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    # Use an in-memory SQLite database for testing
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_and_retrieve_user(db_session):
    # Create a user
    user = User(username="alice", email="alice@example.com")
    db_session.add(user)
    db_session.commit()

    # Retrieve the user
    retrieved = db_session.query(User).filter_by(username="alice").first()
    assert retrieved is not None
    assert retrieved.email == "alice@example.com"
```

### API Tests

```python
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app({'TESTING': True})
    with app.test_client() as client:
        yield client

def test_get_users_endpoint(client):
    response = client.get('/api/users')
    assert response.status_code == 200
    assert 'users' in response.json

def test_create_user_endpoint(client):
    response = client.post('/api/users', json={
        'username': 'alice',
        'email': 'alice@example.com'
    })
    assert response.status_code == 201
    assert response.json['username'] == 'alice'
```

**JavaScript (Express + Supertest):**

```javascript
const request = require('supertest');
const app = require('./app');

describe('User API', () => {
    test('GET /api/users returns list of users', async () => {
        const response = await request(app)
            .get('/api/users')
            .expect(200);

        expect(Array.isArray(response.body.users)).toBe(true);
    });

    test('POST /api/users creates a new user', async () => {
        const response = await request(app)
            .post('/api/users')
            .send({ username: 'alice', email: 'alice@example.com' })
            .expect(201);

        expect(response.body.username).toBe('alice');
    });
});
```

### Test Containers

Use Docker containers for integration tests with real databases:

```python
from testcontainers.postgres import PostgresContainer

def test_with_real_postgres():
    with PostgresContainer("postgres:14") as postgres:
        connection_url = postgres.get_connection_url()
        engine = create_engine(connection_url)
        # Run tests against real Postgres
```

---

## End-to-End (E2E) Testing

E2E tests simulate real user interactions with the entire system.

### Browser Automation: Selenium, Playwright, Cypress

**Selenium (Python):**

```python
from selenium import webdriver
from selenium.webdriver.common.by import By

def test_user_registration():
    driver = webdriver.Chrome()
    driver.get("http://localhost:3000/register")

    # Fill in the form
    driver.find_element(By.ID, "username").send_keys("alice")
    driver.find_element(By.ID, "email").send_keys("alice@example.com")
    driver.find_element(By.ID, "password").send_keys("secret123")

    # Submit
    driver.find_element(By.ID, "submit-button").click()

    # Verify success
    success_message = driver.find_element(By.CLASS_NAME, "success-message")
    assert "Registration successful" in success_message.text

    driver.quit()
```

**Playwright (JavaScript):**

```javascript
const { test, expect } = require('@playwright/test');

test('user can register successfully', async ({ page }) => {
    await page.goto('http://localhost:3000/register');

    await page.fill('#username', 'alice');
    await page.fill('#email', 'alice@example.com');
    await page.fill('#password', 'secret123');

    await page.click('#submit-button');

    await expect(page.locator('.success-message')).toContainText('Registration successful');
});
```

**Cypress (JavaScript):**

```javascript
describe('User Registration', () => {
    it('successfully registers a new user', () => {
        cy.visit('/register');

        cy.get('#username').type('alice');
        cy.get('#email').type('alice@example.com');
        cy.get('#password').type('secret123');

        cy.get('#submit-button').click();

        cy.get('.success-message').should('contain', 'Registration successful');
    });
});
```

### When E2E is Worth the Cost

E2E tests are:
- **Slow**: Starting browsers, loading pages, waiting for elements
- **Brittle**: Break when UI changes
- **Expensive**: Require infrastructure, maintenance

**Use E2E tests for:**
- Critical user journeys (checkout, payment, signup)
- Cross-browser compatibility
- Acceptance testing before release

**Don't use E2E tests for:**
- Testing every edge case (use unit tests)
- Testing business logic (use unit/integration tests)
- Rapid feedback during development (too slow)

---

## Code Coverage

Code coverage measures which lines of code are executed during tests.

### Types of Coverage

1. **Line coverage**: What % of lines were executed?
2. **Branch coverage**: What % of if/else branches were taken?
3. **Path coverage**: What % of possible execution paths were tested?

```python
def discount(price, is_member):
    if is_member:
        return price * 0.9  # Line 3
    else:
        return price  # Line 5

# Test 1: Line coverage 80% (lines 1,2,3), branch coverage 50%
assert discount(100, True) == 90

# Test 2: Line coverage 100% (all lines), branch coverage 100%
assert discount(100, True) == 90
assert discount(100, False) == 100
```

### The 100% Coverage Myth

**High coverage ≠ Good tests**

```python
# 100% line coverage, but terrible test!
def test_bad():
    add(2, 3)  # Function is called, but result is not checked!
```

**Good coverage targets:**
- Critical business logic: aim for 90-100%
- Utility functions: 80-90%
- UI code: 60-70% (harder to test)
- **Total project:** 70-80% is reasonable

**Focus on:**
- Testing behavior, not coverage percentage
- Meaningful assertions
- Edge cases and error handling

---

## Property-Based Testing

Instead of writing specific test cases, describe properties that should always hold, and generate random test cases.

### Hypothesis (Python)

```python
from hypothesis import given
import hypothesis.strategies as st

# Traditional approach: write specific examples
def test_reverse_twice_is_identity_manual():
    assert reverse(reverse([1, 2, 3])) == [1, 2, 3]
    assert reverse(reverse([5])) == [5]
    assert reverse(reverse([])) == []

# Property-based: test with random lists
@given(st.lists(st.integers()))
def test_reverse_twice_is_identity(lst):
    assert reverse(reverse(lst)) == lst

# Hypothesis generates many random lists:
# [], [0], [1, 2, 3], [999, -42], etc.
```

**More examples:**

```python
@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    assert a + b == b + a

@given(st.lists(st.integers()))
def test_sorted_list_is_in_order(lst):
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]
```

### fast-check (JavaScript)

```javascript
const fc = require('fast-check');

test('reversing a string twice gives the original', () => {
    fc.assert(
        fc.property(fc.string(), (str) => {
            return reverse(reverse(str)) === str;
        })
    );
});

test('addition is commutative', () => {
    fc.assert(
        fc.property(fc.integer(), fc.integer(), (a, b) => {
            return a + b === b + a;
        })
    );
});
```

**Benefits:**
- Discovers edge cases you didn't think of
- Tests properties, not specific values
- Generates minimal failing examples (shrinking)

---

## Mutation Testing

Mutation testing tests your tests by introducing bugs (mutations) and checking if tests catch them.

```python
# Original code
def is_even(n):
    return n % 2 == 0

# Mutation 1: Change == to !=
def is_even(n):
    return n % 2 != 0  # Bug introduced

# If tests still pass, your tests are weak!
```

**Tools:**
- **Python:** mutmut, mutpy
- **JavaScript:** Stryker
- **Java:** PIT

---

## Summary

**Key Principles:**
1. **Test behavior, not implementation** – Tests should verify what code does, not how
2. **Follow the test pyramid** – Many unit tests, some integration tests, few E2E tests
3. **Keep tests fast** – Slow tests won't be run
4. **Keep tests independent** – One test shouldn't affect another
5. **Make tests readable** – Tests are documentation
6. **Use descriptive names** – Test names should explain what they test
7. **Don't test private methods** – Test public APIs
8. **Avoid test duplication** – Use setup/teardown, fixtures, helper functions

---

## Exercises

### Exercise 1: Write Unit Tests

Write comprehensive unit tests for this `BankAccount` class:

```python
class BankAccount:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

    def transfer(self, amount, target_account):
        self.withdraw(amount)
        target_account.deposit(amount)
```

Cover:
- Normal operations
- Edge cases (zero balance, exact balance withdrawal)
- Error cases (negative amounts, insufficient funds)
- Transfer between accounts

### Exercise 2: Apply TDD

Use TDD to implement a `PriorityQueue` class with these operations:
- `enqueue(item, priority)`: Add item with priority (higher number = higher priority)
- `dequeue()`: Remove and return highest-priority item
- `is_empty()`: Check if queue is empty
- `peek()`: Return highest-priority item without removing

Write tests first, then implement.

### Exercise 3: Test Doubles

Refactor this code to be testable using test doubles:

```python
def send_order_confirmation(order_id):
    order = database.get_order(order_id)  # Database call
    user = database.get_user(order.user_id)  # Database call
    email_service.send(
        to=user.email,
        subject=f"Order {order_id} confirmed",
        body=f"Your order for ${order.total} has been confirmed"
    )  # External email service
    logger.log(f"Confirmation sent for order {order_id}")  # Logging
```

Write tests using:
- Stubs for the database
- Mock for the email service
- Spy for the logger

### Exercise 4: Integration Test

Write an integration test for a simple REST API with these endpoints:
- `POST /api/tasks` – Create a task
- `GET /api/tasks` – List all tasks
- `GET /api/tasks/:id` – Get a specific task
- `PUT /api/tasks/:id` – Update a task
- `DELETE /api/tasks/:id` – Delete a task

Test the complete workflow: create, read, update, delete.

### Exercise 5: Property-Based Testing

Write property-based tests for a `merge_sorted_lists` function that merges two sorted lists:

Properties to test:
- Result length equals sum of input lengths
- Result is sorted
- All elements from inputs appear in result
- Works with empty lists

---

---

**Previous**: [Error Handling Strategies](./09_Error_Handling.md) | **Next**: [Debugging & Profiling](./11_Debugging_and_Profiling.md)
