# End-to-End Testing

**Previous**: [API Testing](./08_API_Testing.md) | **Next**: [Property-Based Testing](./10_Property_Based_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write browser automation tests using Playwright for Python
2. Apply the Page Object pattern to create maintainable E2E tests
3. Use selectors, assertions, and waiting strategies effectively
4. Capture screenshots and video recordings for debugging
5. Integrate E2E tests into CI pipelines
6. Decide when to use E2E tests versus integration tests

---

## What Are End-to-End Tests?

End-to-end (E2E) tests exercise the full application stack from the user's perspective. For a web application, this means launching a real browser, navigating pages, filling forms, clicking buttons, and verifying what the user sees.

```
E2E Test Flow:
Browser ──→ Frontend ──→ API ──→ Business Logic ──→ Database
  ↑                                                     │
  └────────────── Assertions on visible output ─────────┘
```

E2E tests provide the highest confidence that the system works as a whole, but they are also the most expensive to write, maintain, and run. Use them sparingly for critical user journeys.

---

## Playwright for Python

Playwright is a browser automation library developed by Microsoft. It supports Chromium, Firefox, and WebKit, and has first-class Python bindings.

### Installation

```bash
pip install playwright
playwright install  # Downloads browser binaries
```

For pytest integration:

```bash
pip install pytest-playwright
```

### Your First Playwright Test

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

The `page` fixture is provided by `pytest-playwright`. Each test gets a fresh browser page.

### Running Playwright Tests

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

## Selectors and Locators

Playwright provides multiple strategies for finding elements. Prefer user-facing locators that match how users perceive the page.

### Recommended Locators (Best to Worst)

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

### Chaining Locators

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

## Interactions and Assertions

### Common Interactions

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

### Playwright Assertions

Playwright provides auto-waiting assertions that retry until the condition is met or a timeout expires:

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

### Waiting Strategies

Playwright auto-waits for elements before interactions, but sometimes you need explicit waits:

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

## The Page Object Pattern

The Page Object pattern encapsulates page-specific selectors and actions in dedicated classes. This keeps test code clean and makes selector changes a single-point update.

### Without Page Objects (Brittle)

```python
def test_login_flow(page: Page):
    page.goto("http://localhost:8000/login")
    page.locator("#username-input").fill("alice")
    page.locator("#password-input").fill("secret123")
    page.locator("button.login-btn").click()
    page.wait_for_url("**/dashboard")
    assert page.locator("h1.dashboard-title").text_content() == "Welcome, Alice"
```

If any selector changes (e.g., `#username-input` becomes `#email-input`), you must update every test that references it.

### With Page Objects (Maintainable)

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

## Screenshots and Video Recording

### Screenshots on Failure

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

### Manual Screenshots

```python
def test_visual_state(page: Page):
    page.goto("http://localhost:8000/dashboard")

    # Capture full page
    page.screenshot(path="dashboard.png", full_page=True)

    # Capture specific element
    chart = page.locator(".revenue-chart")
    chart.screenshot(path="revenue-chart.png")
```

### Video Recording

Configure video recording in `pytest.ini` or `conftest.py`:

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

Or via command line:

```bash
pytest --video=on               # Record all tests
pytest --video=retain-on-failure # Record only failing tests
```

### Tracing

Playwright traces capture a timeline of actions, screenshots, and network requests:

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

## CI Integration

### GitHub Actions Example

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

### Practical CI Considerations

- **Browser installation**: `playwright install --with-deps` handles system dependencies
- **Server startup**: Start your app in the background before running tests
- **Timeouts**: Increase default timeouts in CI (slower than local machines)
- **Artifacts**: Upload screenshots and videos on failure for debugging
- **Parallelization**: Use `pytest-xdist` with `--numprocesses` for parallel E2E runs

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

## When E2E vs Integration Tests

E2E tests are expensive. Reserve them for scenarios where no cheaper test provides sufficient confidence.

### Use E2E Tests For

- **Critical user journeys**: Login, checkout, onboarding flows
- **Cross-system workflows**: Frontend -> API -> DB -> external service
- **Visual correctness**: Layout rendering, responsive design
- **JavaScript-dependent behavior**: Client-side validation, dynamic UI

### Use Integration Tests Instead For

- **API correctness**: Use test clients (Flask/FastAPI), not a browser
- **Database queries**: Use repository tests with SQLAlchemy
- **Business logic**: Use unit tests with appropriate mocks
- **Performance**: Integration tests are 10-100x faster than E2E

### Decision Framework

```
Does this test REQUIRE a browser?
├── Yes (visual, JS behavior, cross-page flow) ──→ E2E test
└── No
    ├── Does it cross multiple services? ──→ Integration test
    └── Can it be tested in isolation? ──→ Unit test
```

### Recommended E2E Test Count

For a typical web application:

| Application Size | E2E Tests | Integration Tests | Unit Tests |
|-----------------|-----------|-------------------|------------|
| Small (MVP)     | 5-10      | 20-50             | 100+       |
| Medium          | 15-30     | 50-150            | 500+       |
| Large           | 30-50     | 150-500           | 2000+      |

The exact numbers vary, but E2E tests should always be a small fraction of your total test suite.

---

## Practical Example: Testing a Todo Application

Bringing together all concepts in a realistic E2E test suite:

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

## Exercises

1. **Write Page Objects**: Create Page Object classes for a login page and a dashboard page. The login page should have `login(username, password)` and `expect_error(message)` methods. The dashboard should have `expect_welcome(name)` and `navigate_to(section)` methods. Write 5 E2E tests using these page objects.

2. **Screenshot-on-failure**: Configure a `conftest.py` that automatically captures a full-page screenshot whenever a test fails. Store screenshots in a `test-results/` directory with the test name as the filename. Run some intentionally failing tests to verify it works.

3. **E2E vs integration decision**: Given this list of test scenarios, classify each as unit, integration, or E2E and justify your choice:
   - User can search for products and see results
   - The discount calculation applies correctly for gold members
   - The checkout flow works from cart to confirmation page
   - The API returns 422 when an invalid email is submitted
   - The navigation menu collapses on mobile viewports

---

**License**: CC BY-NC 4.0
