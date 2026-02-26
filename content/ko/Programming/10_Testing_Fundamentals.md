# 테스팅 기초(Testing Fundamentals)

> **주제**: Programming
> **레슨**: 16 중 10
> **선수 지식**: 함수와 메서드, 에러 처리, 객체 지향 프로그래밍
> **목표**: 다양한 유형의 테스트(단위, 통합, E2E)를 마스터하고, TDD와 BDD를 이해하며, 테스트 더블을 효과적으로 사용하는 방법을 배우고, 취약성 없이 신뢰를 제공하는 테스트를 작성합니다.

---

## 소개

테스팅은 단순히 버그를 찾는 것만이 아닙니다 – 코드에 대한 신뢰를 구축하고, 리팩토링을 가능하게 하며, 문서를 제공하고, 설계를 개선하는 것입니다. 잘 테스트된 코드는 변경, 유지보수, 이해가 더 쉽습니다.

이 레슨은 여러 언어와 테스팅 패러다임에 걸친 테스팅 기초를 다루며, 단위 테스트부터 종단 간 테스트까지, 전통적인 테스팅부터 테스트 주도 개발까지, 목부터 속성 기반 테스팅까지 포함합니다.

---

## 왜 테스트를 하는가?

### 1. 신뢰(Confidence)

테스트는 코드가 의도한 대로 작동하고 변경 후에도 계속 작동한다는 신뢰를 줍니다.

### 2. 회귀 방지(Regression Prevention)

테스트는 버그를 도입할 때 잡아내며, 몇 주 후 프로덕션에서가 아닙니다.

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

### 3. 문서화(Documentation)

테스트는 코드가 어떻게 사용되어야 하는지를 문서화합니다:

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

### 4. 설계 피드백(Design Feedback)

코드를 테스트하기 어렵다면, 종종 잘못 설계된 것입니다:
- 의존성이 너무 많음 → 강한 결합
- 함수가 너무 많은 일을 함 → 단일 책임 위반
- 목 만들기 어려움 → 의존성 역전 필요

테스팅은 더 나은 설계를 유도합니다.

---

## 테스트 피라미드(Test Pyramid)

테스트 피라미드는 테스팅 노력을 어떻게 분배할지 안내합니다:

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

**분포:**
- **70% 단위 테스트**: 개별 함수/클래스를 고립하여 테스트
- **20% 통합 테스트**: 컴포넌트가 함께 작동하는지 테스트
- **10% E2E 테스트**: 사용자 관점에서 전체 시스템 테스트

**왜 이 비율인가?**
- 단위 테스트는 빠르고, 실패를 정확히 지적하며, 유지보수가 쉬움
- 통합 테스트는 인터페이스 문제를 잡아냄
- E2E 테스트는 실제 사용자 워크플로를 잡지만 느리고 취약함

---

## 단위 테스팅(Unit Testing)

단위 테스트는 개별 단위(함수, 메서드, 클래스)가 고립되어 올바르게 작동하는지 검증합니다.

### 단위란 무엇인가?

"단위"는 애플리케이션의 테스트 가능한 가장 작은 부분입니다:
- 함수
- 메서드
- 클래스
- 모듈 (일부 맥락에서)

### 구조: 배치-실행-단언(Arrange-Act-Assert, AAA)

주어진-언제-그러면(Given-When-Then)이라고도 합니다:

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

### 명명 규칙(Naming Conventions)

좋은 테스트 이름은 무엇을 테스트하고 예상 결과가 무엇인지 설명합니다:

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

대체 규칙 (BDD 스타일):

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

### 다양한 언어의 예제

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

## 테스트 주도 개발(Test-Driven Development, TDD)

TDD는 구현 코드를 작성하기 **전에** 테스트를 먼저 작성하는 개발 방법론입니다.

### 빨강-초록-리팩토링 주기(Red-Green-Refactor Cycle)

```
1. RED:     Write a failing test
     ↓
2. GREEN:   Write minimal code to make it pass
     ↓
3. REFACTOR: Improve the code without changing behavior
     ↓
     (repeat)
```

### 예제: TDD로 스택 구현하기

**단계 1: RED – 실패하는 테스트 작성**

```python
import pytest
from stack import Stack

def test_new_stack_is_empty():
    stack = Stack()
    assert stack.is_empty() == True
```

실행: ❌ FAIL (Stack 클래스가 존재하지 않음)

**단계 2: GREEN – 통과하게 만들기 (최소 코드)**

```python
class Stack:
    def is_empty(self):
        return True
```

실행: ✅ PASS

**단계 3: 다른 테스트 추가 (RED)**

```python
def test_push_adds_element():
    stack = Stack()
    stack.push(5)
    assert stack.is_empty() == False
```

실행: ❌ FAIL (push가 존재하지 않고, is_empty가 항상 True 반환)

**단계 4: GREEN – 통과하게 만들기**

```python
class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)
```

실행: ✅ PASS

**단계 5: 더 많은 테스트 추가 및 pop, peek 등 구현**

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

**단계 6: REFACTOR**

이제 테스트가 통과하므로, 명확성을 위해 리팩토링합니다:

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

테스트는 여전히 통과 ✅하지만, 코드가 더 깔끔해졌습니다.

### TDD의 이점

1. **더 나은 설계**: 테스트를 먼저 작성하면 인터페이스에 대해 생각하게 됨
2. **완전한 커버리지**: 모든 코드 라인에 테스트가 있음 (테스트를 먼저 작성했으니까!)
3. **신뢰**: 코드가 작동하는 것을 알고 있음 (지속적으로 테스트했으니까)
4. **문서화**: 테스트가 예상 동작을 문서화함
5. **디버깅 감소**: 버그가 즉시 잡힘

### TDD가 잘 작동하는 경우

- 명확한 입력/출력이 있는 알고리즘 코드
- 유틸리티 함수와 라이브러리
- 버그 수정 (버그를 재현하는 테스트를 작성한 후 수정)
- 리팩토링 (테스트가 동작이 변경되지 않았음을 보장)

### TDD가 도전적인 경우

- 탐색적 코딩 (무엇을 만들고 있는지 아직 모름)
- UI 코드 (존재하기 전에 상호작용을 테스트하기 어려움)
- 복잡한 통합 (많은 미지수)
- 빠른 프로토타이핑

**참고:** TDD는 도구이지 종교가 아닙니다. 도움이 될 때 사용하고, 방해가 될 때는 건너뜁니다.

---

## 행동 주도 개발(Behavior-Driven Development, BDD)

BDD는 사용자 관점에서의 행동에 초점을 맞춰 TDD를 확장합니다.

### 주어진/언제/그러면 구문(Given/When/Then Syntax)

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

### 도구: Cucumber, Behave, Jest

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

### 사용자 스토리 → 수락 기준 → 테스트

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

## 테스트 더블(Test Doubles)

테스트 더블은 테스트 중에 실제 의존성을 대신하는 객체입니다.

### 테스트 더블의 유형

#### 1. 더미(Dummy)

자리 표시 객체, 실제로 사용되지 않음:

```python
def test_send_email():
    # We don't care about the logger, but the function requires one
    dummy_logger = None
    send_email("alice@example.com", "Hello", logger=dummy_logger)
```

#### 2. 스텁(Stub)

미리 정해진 데이터를 반환:

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

#### 3. 목(Mock)

특정 메서드가 예상된 인자로 호출되었는지 검증:

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

#### 4. 스파이(Spy)

호출 방법을 기록하여 실행 후 검증 가능:

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

#### 5. 페이크(Fake)

작동하는 구현이지만 실제보다 단순함:

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

### 각각 언제 사용하는가

| 유형 | 사용 시기 | 예제 |
|------|----------|---------|
| **더미(Dummy)** | 파라미터가 필요하지만 사용되지 않을 때 | 절대 호출되지 않는 로거 |
| **스텁(Stub)** | 미리 정해진 응답이 필요할 때 | 테스트 데이터를 반환하는 데이터베이스 |
| **목(Mock)** | 상호작용을 검증하고 싶을 때 | 이메일 서비스 호출 확인 |
| **스파이(Spy)** | 행동을 관찰하고 싶을 때 | 로거 호출 기록 |
| **페이크(Fake)** | 작동하지만 단순한 구현이 필요할 때 | 인메모리 데이터베이스 |

---

## 통합 테스팅(Integration Testing)

통합 테스트는 컴포넌트가 함께 올바르게 작동하는지 검증합니다.

### 데이터베이스 테스트

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

### API 테스트

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

### 테스트 컨테이너(Test Containers)

실제 데이터베이스로 통합 테스트를 위해 Docker 컨테이너 사용:

```python
from testcontainers.postgres import PostgresContainer

def test_with_real_postgres():
    with PostgresContainer("postgres:14") as postgres:
        connection_url = postgres.get_connection_url()
        engine = create_engine(connection_url)
        # Run tests against real Postgres
```

---

## 종단 간(End-to-End, E2E) 테스팅

E2E 테스트는 전체 시스템과의 실제 사용자 상호작용을 시뮬레이션합니다.

### 브라우저 자동화: Selenium, Playwright, Cypress

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

### E2E가 비용 대비 가치가 있을 때

E2E 테스트는:
- **느림**: 브라우저 시작, 페이지 로딩, 요소 대기
- **취약함**: UI 변경 시 깨짐
- **비용 높음**: 인프라, 유지보수 필요

**E2E 테스트 사용:**
- 중요한 사용자 여정 (체크아웃, 결제, 가입)
- 크로스 브라우저 호환성
- 릴리스 전 수락 테스트

**E2E 테스트 사용하지 않음:**
- 모든 엣지 케이스 테스트 (단위 테스트 사용)
- 비즈니스 로직 테스트 (단위/통합 테스트 사용)
- 개발 중 빠른 피드백 (너무 느림)

---

## 코드 커버리지(Code Coverage)

코드 커버리지는 테스트 중에 실행된 코드 라인의 비율을 측정합니다.

### 커버리지 유형

1. **라인 커버리지**: 몇 %의 라인이 실행되었는가?
2. **브랜치 커버리지**: 몇 %의 if/else 브랜치가 실행되었는가?
3. **경로 커버리지**: 몇 %의 가능한 실행 경로가 테스트되었는가?

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

### 100% 커버리지 신화

**높은 커버리지 ≠ 좋은 테스트**

```python
# 100% line coverage, but terrible test!
def test_bad():
    add(2, 3)  # Function is called, but result is not checked!
```

**좋은 커버리지 목표:**
- 중요한 비즈니스 로직: 90-100% 목표
- 유틸리티 함수: 80-90%
- UI 코드: 60-70% (테스트하기 더 어려움)
- **전체 프로젝트:** 70-80%가 합리적

**초점:**
- 커버리지 비율이 아닌 행동 테스트
- 의미 있는 단언문
- 엣지 케이스와 에러 처리

---

## 속성 기반 테스팅(Property-Based Testing)

특정 테스트 케이스를 작성하는 대신, 항상 유지되어야 하는 속성을 설명하고 무작위 테스트 케이스를 생성합니다.

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

**더 많은 예제:**

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

**이점:**
- 생각하지 못한 엣지 케이스 발견
- 특정 값이 아닌 속성 테스트
- 최소 실패 예제 생성 (축소)

---

## 변이 테스팅(Mutation Testing)

변이 테스팅은 버그(변이)를 도입하여 테스트가 잡아내는지 확인함으로써 테스트를 테스트합니다.

```python
# Original code
def is_even(n):
    return n % 2 == 0

# Mutation 1: Change == to !=
def is_even(n):
    return n % 2 != 0  # Bug introduced

# If tests still pass, your tests are weak!
```

**도구:**
- **Python:** mutmut, mutpy
- **JavaScript:** Stryker
- **Java:** PIT

---

## 요약

**핵심 원칙:**
1. **구현이 아닌 행동 테스트** – 테스트는 코드가 무엇을 하는지 검증해야 하며, 어떻게 하는지가 아님
2. **테스트 피라미드 따르기** – 많은 단위 테스트, 일부 통합 테스트, 적은 E2E 테스트
3. **테스트를 빠르게 유지** – 느린 테스트는 실행되지 않음
4. **테스트를 독립적으로 유지** – 한 테스트가 다른 테스트에 영향을 주지 않아야 함
5. **테스트를 읽기 쉽게** – 테스트는 문서임
6. **설명적인 이름 사용** – 테스트 이름이 무엇을 테스트하는지 설명해야 함
7. **private 메서드 테스트하지 않기** – public API 테스트
8. **테스트 중복 피하기** – setup/teardown, fixtures, 헬퍼 함수 사용

---

## 연습 문제

### 연습 문제 1: 단위 테스트 작성

이 `BankAccount` 클래스에 대한 포괄적인 단위 테스트를 작성하세요:

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

커버:
- 정상 작동
- 엣지 케이스 (잔액 0, 정확한 잔액 인출)
- 에러 케이스 (음수 금액, 잔액 부족)
- 계정 간 이체

### 연습 문제 2: TDD 적용

다음 작업으로 `PriorityQueue` 클래스를 TDD를 사용하여 구현하세요:
- `enqueue(item, priority)`: 우선순위로 항목 추가 (높은 숫자 = 높은 우선순위)
- `dequeue()`: 최고 우선순위 항목 제거 및 반환
- `is_empty()`: 큐가 비어 있는지 확인
- `peek()`: 최고 우선순위 항목을 제거하지 않고 반환

테스트를 먼저 작성한 후 구현하세요.

### 연습 문제 3: 테스트 더블

테스트 더블을 사용하여 이 코드를 테스트 가능하게 리팩토링하세요:

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

다음을 사용하여 테스트를 작성하세요:
- 데이터베이스용 스텁
- 이메일 서비스용 목
- 로거용 스파이

### 연습 문제 4: 통합 테스트

다음 엔드포인트가 있는 간단한 REST API에 대한 통합 테스트를 작성하세요:
- `POST /api/tasks` – 작업 생성
- `GET /api/tasks` – 모든 작업 나열
- `GET /api/tasks/:id` – 특정 작업 가져오기
- `PUT /api/tasks/:id` – 작업 업데이트
- `DELETE /api/tasks/:id` – 작업 삭제

완전한 워크플로 테스트: 생성, 읽기, 업데이트, 삭제.

### 연습 문제 5: 속성 기반 테스팅

두 개의 정렬된 리스트를 병합하는 `merge_sorted_lists` 함수에 대한 속성 기반 테스트를 작성하세요:

테스트할 속성:
- 결과 길이는 입력 길이의 합과 같음
- 결과가 정렬됨
- 입력의 모든 요소가 결과에 나타남
- 빈 리스트와 작동함

---

## 내비게이션

**이전 레슨**: [에러 처리 전략(Error Handling Strategies)](09_Error_Handling.md)
**다음 레슨**: [디버깅 및 프로파일링(Debugging & Profiling)](11_Debugging_and_Profiling.md)
