# 엔드투엔드 테스트 (End-to-End Testing)

**이전**: [API 테스트](./08_API_Testing.md) | **다음**: [속성 기반 테스트](./10_Property_Based_Testing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Python용 Playwright를 사용하여 브라우저 자동화 테스트를 작성할 수 있다
2. 유지보수하기 쉬운 E2E 테스트를 위해 Page Object 패턴을 적용할 수 있다
3. 셀렉터, 어서션, 대기 전략을 효과적으로 사용할 수 있다
4. 디버깅을 위한 스크린샷 및 비디오 녹화를 캡처할 수 있다
5. E2E 테스트를 CI 파이프라인에 통합할 수 있다
6. E2E 테스트와 통합 테스트의 사용 시점을 판단할 수 있다

---

## 엔드투엔드 테스트란 무엇인가?

엔드투엔드(E2E) 테스트는 사용자의 관점에서 전체 애플리케이션 스택을 실행합니다. 웹 애플리케이션의 경우, 이는 실제 브라우저를 실행하고, 페이지를 탐색하며, 폼을 채우고, 버튼을 클릭하고, 사용자에게 보이는 것을 검증하는 것을 의미합니다.

```
E2E Test Flow:
Browser ──→ Frontend ──→ API ──→ Business Logic ──→ Database
  ↑                                                     │
  └────────────── Assertions on visible output ─────────┘
```

E2E 테스트는 시스템이 전체적으로 작동하는지에 대해 가장 높은 신뢰도를 제공하지만, 작성, 유지보수, 실행 비용도 가장 높습니다. 핵심 사용자 여정에 대해 선별적으로 사용합니다.

---

## Python용 Playwright

Playwright는 Microsoft가 개발한 브라우저 자동화 라이브러리입니다. Chromium, Firefox, WebKit을 지원하며, Python 바인딩을 공식 지원합니다.

### 설치

```bash
pip install playwright
playwright install  # Downloads browser binaries
```

pytest 통합을 위해:

```bash
pip install pytest-playwright
```

### 첫 번째 Playwright 테스트

```python
# test_homepage.py
from playwright.sync_api import Page


def test_homepage_title(page: Page):
    page.goto("https://example.com")
    assert page.title() == "Example Domain"


def test_homepage_has_heading(page: Page):
    page.goto("https://example.com")
    heading = page.locator("h1")
    assert heading.text_content() == "Example Domain"
```

`page` 픽스처는 `pytest-playwright`가 제공합니다. 각 테스트는 새로운 브라우저 페이지를 받습니다.

### Playwright 테스트 실행하기

```bash
# Run normally (headless)
pytest test_homepage.py

# Run with visible browser (for debugging)
pytest test_homepage.py --headed

# Run in slow motion (see each action)
pytest test_homepage.py --headed --slowmo=500

# Run in specific browser
pytest test_homepage.py --browser firefox
pytest test_homepage.py --browser webkit
```

---

## 셀렉터와 로케이터

Playwright는 요소를 찾기 위한 여러 전략을 제공합니다. 사용자가 페이지를 인식하는 방식과 일치하는 사용자 중심 로케이터를 선호합니다.

### 권장 로케이터 (좋은 순서에서 나쁜 순서로)

```python
def test_selector_strategies(page: Page):
    page.goto("http://localhost:8000")

    # 1. Role-based (best — accessible and stable)
    page.get_by_role("button", name="Submit")
    page.get_by_role("heading", name="Welcome")
    page.get_by_role("link", name="Sign Up")

    # 2. Text-based (good for visible text)
    page.get_by_text("Add to Cart")
    page.get_by_label("Email address")
    page.get_by_placeholder("Enter your name")

    # 3. Test ID (good when no semantic selector exists)
    page.get_by_test_id("checkout-button")
    # Requires: <button data-testid="checkout-button">

    # 4. CSS selector (acceptable, but brittle if classes change)
    page.locator(".product-card")
    page.locator("#main-content")

    # 5. XPath (avoid — verbose and fragile)
    page.locator("xpath=//div[@class='product']//button")
```

### 로케이터 체이닝

```python
def test_locator_chaining(page: Page):
    page.goto("http://localhost:8000/products")

    # Find the "Add to Cart" button within a specific product card
    product = page.locator(".product-card").filter(has_text="Laptop")
    product.get_by_role("button", name="Add to Cart").click()

    # Verify cart count updated
    cart_count = page.get_by_test_id("cart-count")
    assert cart_count.text_content() == "1"
```

---

## 인터랙션과 어서션

### 일반적인 인터랙션

```python
def test_form_interactions(page: Page):
    page.goto("http://localhost:8000/register")

    # Text input
    page.get_by_label("Username").fill("alice")
    page.get_by_label("Email").fill("alice@example.com")
    page.get_by_label("Password").fill("SecureP@ss1")

    # Checkbox
    page.get_by_label("I agree to terms").check()

    # Dropdown
    page.get_by_label("Country").select_option("US")

    # Radio button
    page.get_by_label("Monthly billing").click()

    # Submit
    page.get_by_role("button", name="Create Account").click()

    # Wait for navigation
    page.wait_for_url("**/dashboard")
```

### Playwright 어서션

Playwright는 조건이 충족되거나 타임아웃이 만료될 때까지 재시도하는 자동 대기 어서션을 제공합니다:

```python
from playwright.sync_api import expect


def test_assertions(page: Page):
    page.goto("http://localhost:8000")

    # Element visibility
    expect(page.get_by_role("heading", name="Dashboard")).to_be_visible()
    expect(page.get_by_text("Loading...")).to_be_hidden()

    # Text content
    expect(page.locator(".welcome")).to_have_text("Hello, Alice")
    expect(page.locator(".welcome")).to_contain_text("Alice")

    # Element count
    expect(page.locator(".notification")).to_have_count(3)

    # CSS and attributes
    expect(page.locator(".active-tab")).to_have_class("tab active-tab")
    expect(page.get_by_role("link", name="Profile")).to_have_attribute("href", "/profile")

    # Input values
    expect(page.get_by_label("Search")).to_have_value("initial query")

    # Page URL and title
    expect(page).to_have_url("http://localhost:8000/dashboard")
    expect(page).to_have_title("My Dashboard")
```

### 대기 전략

Playwright는 인터랙션 전에 요소를 자동으로 대기하지만, 때로는 명시적 대기가 필요합니다:

```python
def test_async_content(page: Page):
    page.goto("http://localhost:8000/reports")

    # Click generates a report asynchronously
    page.get_by_role("button", name="Generate Report").click()

    # Wait for a specific element to appear
    page.wait_for_selector(".report-complete", timeout=30_000)

    # Wait for network requests to finish
    with page.expect_response("**/api/report") as response_info:
        page.get_by_role("button", name="Generate Report").click()
    response = response_info.value
    assert response.status == 200

    # Wait for a specific condition
    page.wait_for_function("document.querySelector('.progress').style.width === '100%'")
```

---

## Page Object 패턴

Page Object 패턴은 페이지별 셀렉터와 액션을 전용 클래스에 캡슐화합니다. 이를 통해 테스트 코드를 깔끔하게 유지하고, 셀렉터 변경을 단일 지점에서 업데이트할 수 있습니다.

### Page Object 없이 (취약한 방식)

```python
def test_login_flow(page: Page):
    page.goto("http://localhost:8000/login")
    page.locator("#username-input").fill("alice")
    page.locator("#password-input").fill("secret123")
    page.locator("button.login-btn").click()
    page.wait_for_url("**/dashboard")
    assert page.locator("h1.dashboard-title").text_content() == "Welcome, Alice"
```

셀렉터가 변경되면(예: `#username-input`이 `#email-input`으로 변경), 이를 참조하는 모든 테스트를 업데이트해야 합니다.

### Page Object 사용 (유지보수하기 쉬운 방식)

```python
# pages/login_page.py
from playwright.sync_api import Page, expect


class LoginPage:
    def __init__(self, page: Page):
        self.page = page
        self.username_input = page.get_by_label("Username")
        self.password_input = page.get_by_label("Password")
        self.submit_button = page.get_by_role("button", name="Log In")
        self.error_message = page.get_by_role("alert")

    def goto(self):
        self.page.goto("http://localhost:8000/login")
        return self

    def login(self, username: str, password: str):
        self.username_input.fill(username)
        self.password_input.fill(password)
        self.submit_button.click()
        return self

    def expect_error(self, message: str):
        expect(self.error_message).to_contain_text(message)
        return self
```

```python
# pages/dashboard_page.py
from playwright.sync_api import Page, expect


class DashboardPage:
    def __init__(self, page: Page):
        self.page = page
        self.heading = page.get_by_role("heading", level=1)
        self.logout_button = page.get_by_role("button", name="Log Out")
        self.notification_list = page.locator(".notifications li")

    def expect_welcome_message(self, name: str):
        expect(self.heading).to_have_text(f"Welcome, {name}")
        return self

    def expect_notification_count(self, count: int):
        expect(self.notification_list).to_have_count(count)
        return self

    def logout(self):
        self.logout_button.click()
        return LoginPage(self.page)
```

```python
# test_login.py
from pages.login_page import LoginPage
from pages.dashboard_page import DashboardPage


def test_successful_login(page):
    login_page = LoginPage(page).goto()
    login_page.login("alice", "secret123")

    dashboard = DashboardPage(page)
    dashboard.expect_welcome_message("Alice")


def test_invalid_credentials(page):
    login_page = LoginPage(page).goto()
    login_page.login("alice", "wrong_password")
    login_page.expect_error("Invalid credentials")


def test_logout(page):
    LoginPage(page).goto().login("alice", "secret123")
    dashboard = DashboardPage(page)
    login_page = dashboard.logout()

    expect(page).to_have_url("**/login")
```

---

## 스크린샷과 비디오 녹화

### 실패 시 스크린샷

```python
# conftest.py
import pytest
from pathlib import Path


@pytest.fixture
def page(page, request):
    """Override the default page fixture to capture screenshots on failure."""
    yield page
    if request.node.rep_call.failed:
        screenshot_dir = Path("test-results/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=screenshot_dir / f"{request.node.name}.png",
            full_page=True,
        )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test result available to fixtures."""
    import pytest
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
```

### 수동 스크린샷

```python
def test_visual_state(page: Page):
    page.goto("http://localhost:8000/dashboard")

    # Capture full page
    page.screenshot(path="dashboard.png", full_page=True)

    # Capture specific element
    chart = page.locator(".revenue-chart")
    chart.screenshot(path="revenue-chart.png")
```

### 비디오 녹화

`pytest.ini` 또는 `conftest.py`에서 비디오 녹화를 설정합니다:

```python
# conftest.py
import pytest


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "record_video_dir": "test-results/videos/",
        "record_video_size": {"width": 1280, "height": 720},
    }
```

또는 커맨드 라인에서:

```bash
pytest --video=on               # Record all tests
pytest --video=retain-on-failure # Record only failing tests
```

### 트레이싱

Playwright 트레이스는 액션, 스크린샷, 네트워크 요청의 타임라인을 캡처합니다:

```python
def test_with_trace(page: Page, browser):
    context = browser.new_context()
    context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    page.goto("http://localhost:8000")
    page.get_by_role("button", name="Submit").click()

    context.tracing.stop(path="test-results/trace.zip")
    context.close()

    # View trace: playwright show-trace test-results/trace.zip
```

```bash
# View the trace in a browser UI
playwright show-trace test-results/trace.zip
```

---

## CI 통합

### GitHub Actions 예제

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-playwright
          playwright install --with-deps chromium

      - name: Start application
        run: |
          flask run --port 8000 &
          sleep 3  # Wait for server to start

      - name: Run E2E tests
        run: pytest tests/e2e/ --video=retain-on-failure

      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: e2e-results
          path: test-results/
```

### 실용적인 CI 고려사항

- **브라우저 설치**: `playwright install --with-deps`로 시스템 종속성을 처리합니다
- **서버 시작**: 테스트 실행 전에 백그라운드에서 앱을 시작합니다
- **타임아웃**: CI에서는 기본 타임아웃을 늘립니다 (로컬 머신보다 느림)
- **아티팩트**: 실패 시 디버깅을 위해 스크린샷과 비디오를 업로드합니다
- **병렬 처리**: `pytest-xdist`의 `--numprocesses`로 병렬 E2E 실행을 합니다

```python
# conftest.py — CI-aware configuration
import os

# Increase timeout in CI
DEFAULT_TIMEOUT = 30_000 if os.getenv("CI") else 10_000

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
    }
```

---

## E2E 테스트 vs 통합 테스트

E2E 테스트는 비용이 높습니다. 더 저렴한 테스트로 충분한 신뢰도를 얻을 수 없는 시나리오에만 사용합니다.

### E2E 테스트를 사용해야 하는 경우

- **핵심 사용자 여정**: 로그인, 결제, 온보딩 플로우
- **시스템 간 워크플로우**: 프론트엔드 -> API -> DB -> 외부 서비스
- **시각적 정확성**: 레이아웃 렌더링, 반응형 디자인
- **JavaScript 의존 동작**: 클라이언트 측 유효성 검증, 동적 UI

### 통합 테스트를 대신 사용해야 하는 경우

- **API 정확성**: 브라우저 대신 테스트 클라이언트(Flask/FastAPI) 사용
- **데이터베이스 쿼리**: SQLAlchemy를 사용한 리포지토리 테스트
- **비즈니스 로직**: 적절한 mock을 사용한 단위 테스트
- **성능**: 통합 테스트가 E2E보다 10~100배 빠름

### 결정 프레임워크

```
Does this test REQUIRE a browser?
├── Yes (visual, JS behavior, cross-page flow) ──→ E2E test
└── No
    ├── Does it cross multiple services? ──→ Integration test
    └── Can it be tested in isolation? ──→ Unit test
```

### 권장 E2E 테스트 수

일반적인 웹 애플리케이션의 경우:

| 애플리케이션 규모 | E2E 테스트 | 통합 테스트 | 단위 테스트 |
|-----------------|-----------|------------|-----------|
| 소규모 (MVP)     | 5-10      | 20-50      | 100+      |
| 중규모           | 15-30     | 50-150     | 500+      |
| 대규모           | 30-50     | 150-500    | 2000+     |

정확한 수치는 다양하지만, E2E 테스트는 항상 전체 테스트 스위트의 작은 비율이어야 합니다.

---

## 실용 예제: Todo 애플리케이션 테스트

모든 개념을 현실적인 E2E 테스트 스위트로 통합합니다:

```python
# pages/todo_page.py
from playwright.sync_api import Page, expect


class TodoPage:
    URL = "http://localhost:8000/todos"

    def __init__(self, page: Page):
        self.page = page
        self.new_todo_input = page.get_by_placeholder("What needs to be done?")
        self.todo_items = page.get_by_test_id("todo-item")
        self.clear_completed = page.get_by_role("button", name="Clear completed")
        self.items_left = page.get_by_test_id("todo-count")

    def goto(self):
        self.page.goto(self.URL)
        return self

    def add_todo(self, text: str):
        self.new_todo_input.fill(text)
        self.new_todo_input.press("Enter")
        return self

    def complete_todo(self, text: str):
        item = self.todo_items.filter(has_text=text)
        item.get_by_role("checkbox").check()
        return self

    def delete_todo(self, text: str):
        item = self.todo_items.filter(has_text=text)
        item.hover()  # Delete button appears on hover
        item.get_by_role("button", name="Delete").click()
        return self

    def expect_todo_count(self, count: int):
        expect(self.todo_items).to_have_count(count)
        return self

    def expect_items_left(self, count: int):
        expect(self.items_left).to_contain_text(str(count))
        return self
```

```python
# test_todo_e2e.py
from pages.todo_page import TodoPage


class TestTodoApp:
    def test_add_single_todo(self, page):
        todo = TodoPage(page).goto()
        todo.add_todo("Buy milk")
        todo.expect_todo_count(1)
        todo.expect_items_left(1)

    def test_add_multiple_todos(self, page):
        todo = TodoPage(page).goto()
        todo.add_todo("Buy milk")
        todo.add_todo("Walk the dog")
        todo.add_todo("Read a book")
        todo.expect_todo_count(3)
        todo.expect_items_left(3)

    def test_complete_todo(self, page):
        todo = TodoPage(page).goto()
        todo.add_todo("Buy milk")
        todo.add_todo("Walk the dog")
        todo.complete_todo("Buy milk")
        todo.expect_items_left(1)

    def test_delete_todo(self, page):
        todo = TodoPage(page).goto()
        todo.add_todo("Buy milk")
        todo.add_todo("Walk the dog")
        todo.delete_todo("Buy milk")
        todo.expect_todo_count(1)

    def test_full_workflow(self, page):
        """Critical path: add, complete, clear completed."""
        todo = TodoPage(page).goto()

        # Add items
        todo.add_todo("Task 1")
        todo.add_todo("Task 2")
        todo.add_todo("Task 3")
        todo.expect_todo_count(3)

        # Complete one
        todo.complete_todo("Task 2")
        todo.expect_items_left(2)

        # Clear completed
        todo.page.get_by_role("button", name="Clear completed").click()
        todo.expect_todo_count(2)
```

---

## 연습 문제

1. **Page Object 작성하기**: 로그인 페이지와 대시보드 페이지를 위한 Page Object 클래스를 생성합니다. 로그인 페이지에는 `login(username, password)`와 `expect_error(message)` 메서드가 있어야 합니다. 대시보드에는 `expect_welcome(name)`과 `navigate_to(section)` 메서드가 있어야 합니다. 이 Page Object를 사용하여 5개의 E2E 테스트를 작성합니다.

2. **실패 시 스크린샷**: 테스트가 실패할 때 자동으로 전체 페이지 스크린샷을 캡처하는 `conftest.py`를 설정합니다. 스크린샷을 테스트 이름을 파일명으로 사용하여 `test-results/` 디렉토리에 저장합니다. 의도적으로 실패하는 테스트를 실행하여 작동하는지 확인합니다.

3. **E2E vs 통합 테스트 결정**: 다음 테스트 시나리오 목록에 대해 각각을 단위, 통합, E2E로 분류하고 선택 이유를 설명합니다:
   - 사용자가 제품을 검색하고 결과를 볼 수 있음
   - 골드 회원에 대해 할인 계산이 올바르게 적용됨
   - 장바구니부터 확인 페이지까지 결제 플로우가 작동함
   - 유효하지 않은 이메일이 제출되면 API가 422를 반환함
   - 모바일 뷰포트에서 네비게이션 메뉴가 축소됨

---

**License**: CC BY-NC 4.0
