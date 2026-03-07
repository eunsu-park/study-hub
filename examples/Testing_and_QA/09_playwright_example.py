#!/usr/bin/env python3
"""Example: Playwright Browser Automation Testing

Demonstrates Playwright for end-to-end browser testing: page navigation,
element interaction, assertions, network interception, and visual testing.
Related lesson: 12_UI_and_E2E_Testing.md
"""

# =============================================================================
# WHY PLAYWRIGHT?
# Playwright is a modern browser automation framework that:
#   1. Supports Chromium, Firefox, and WebKit from a single API
#   2. Auto-waits for elements — no explicit waits/sleeps needed
#   3. Supports network interception for mocking API responses
#   4. Runs headless by default (perfect for CI/CD)
#   5. Provides codegen tool to record user interactions as test code
#
# Playwright vs Selenium:
#   - Playwright: auto-waits, faster, better API, built-in assertions
#   - Selenium: wider browser support, larger ecosystem, more mature
# =============================================================================

import pytest

try:
    from playwright.sync_api import sync_playwright, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# =============================================================================
# NOTE: These tests require Playwright browsers to be installed:
#   pip install playwright
#   playwright install
#
# The tests below demonstrate patterns and are designed to be educational.
# Some tests use example.com which is a reserved domain for documentation.
# =============================================================================

pytestmark = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE,
    reason="Playwright not installed (pip install playwright && playwright install)"
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def browser_instance():
    """Launch browser once per session (expensive operation).
    scope='session' means all tests share one browser process."""
    with sync_playwright() as p:
        # headless=True for CI, set to False for debugging
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser_instance):
    """Create a fresh browser context and page for each test.
    Context isolation means cookies, localStorage, etc. don't leak between tests."""
    context = browser_instance.new_context(
        viewport={"width": 1280, "height": 720},
        # Uncomment for mobile testing:
        # **p.devices["iPhone 13"]
    )
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture
def authenticated_page(page):
    """Page with pre-set authentication state.
    In real apps, you would set cookies or localStorage tokens."""
    # Simulate login by setting a cookie
    page.context.add_cookies([{
        "name": "session_token",
        "value": "test-token-123",
        "domain": "localhost",
        "path": "/",
    }])
    return page


# =============================================================================
# 1. BASIC NAVIGATION AND ASSERTIONS
# =============================================================================

class TestBasicNavigation:
    """Fundamental Playwright operations."""

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_page_title(self, page):
        """Navigate to a page and verify its title."""
        page.goto("https://example.com")

        # Playwright's expect provides auto-retry assertions
        expect(page).to_have_title("Example Domain")

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_page_url(self, page):
        """Verify the current URL after navigation."""
        page.goto("https://example.com")
        expect(page).to_have_url("https://example.com/")

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_element_visibility(self, page):
        """Check that elements are visible on the page.
        Playwright auto-waits for elements — no explicit wait needed."""
        page.goto("https://example.com")

        heading = page.locator("h1")
        expect(heading).to_be_visible()
        expect(heading).to_have_text("Example Domain")


# =============================================================================
# 2. LOCATOR STRATEGIES
# =============================================================================
# Playwright locators are the foundation of element interaction.
# They auto-wait and auto-retry, making tests more stable.

class TestLocators:
    """Demonstrate different ways to find elements."""

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_css_selector(self, page):
        """CSS selectors — familiar and powerful."""
        page.goto("https://example.com")
        element = page.locator("h1")
        expect(element).to_be_visible()

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_text_selector(self, page):
        """Find elements by their visible text content.
        More resilient to markup changes than CSS selectors."""
        page.goto("https://example.com")
        link = page.get_by_text("More information")
        expect(link).to_be_visible()

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_role_selector(self, page):
        """Find elements by ARIA role — best for accessibility.
        page.get_by_role uses the accessibility tree, not the DOM."""
        page.goto("https://example.com")
        link = page.get_by_role("link", name="More information")
        expect(link).to_be_visible()

    # These are the RECOMMENDED locator priorities (most to least preferred):
    # 1. get_by_role()        — reflects user perception, best for a11y
    # 2. get_by_text()        — visible text, resilient to markup changes
    # 3. get_by_label()       — form elements by label text
    # 4. get_by_placeholder() — inputs by placeholder text
    # 5. get_by_test_id()     — data-testid attributes, last resort


# =============================================================================
# 3. FORM INTERACTION PATTERNS
# =============================================================================
# Demonstrate how to interact with common form elements.

class TestFormInteraction:
    """These tests use page.set_content to create local HTML for testing
    without needing a running server."""

    def test_text_input(self, page):
        """Fill text inputs and verify values."""
        page.set_content('''
            <form>
                <label for="name">Name:</label>
                <input id="name" type="text" />
                <label for="email">Email:</label>
                <input id="email" type="email" />
            </form>
        ''')

        page.get_by_label("Name:").fill("Alice")
        page.get_by_label("Email:").fill("alice@example.com")

        expect(page.get_by_label("Name:")).to_have_value("Alice")
        expect(page.get_by_label("Email:")).to_have_value("alice@example.com")

    def test_checkbox(self, page):
        """Check/uncheck checkboxes."""
        page.set_content('''
            <label><input type="checkbox" id="agree" /> I agree</label>
        ''')

        checkbox = page.get_by_role("checkbox")
        expect(checkbox).not_to_be_checked()

        checkbox.check()
        expect(checkbox).to_be_checked()

        checkbox.uncheck()
        expect(checkbox).not_to_be_checked()

    def test_dropdown(self, page):
        """Select dropdown options."""
        page.set_content('''
            <label for="color">Color:</label>
            <select id="color">
                <option value="">Choose...</option>
                <option value="red">Red</option>
                <option value="blue">Blue</option>
                <option value="green">Green</option>
            </select>
        ''')

        page.get_by_label("Color:").select_option("blue")
        expect(page.get_by_label("Color:")).to_have_value("blue")

    def test_form_submission(self, page):
        """Test form submission and result."""
        page.set_content('''
            <form id="myform">
                <input name="query" type="text" />
                <button type="submit">Search</button>
            </form>
            <div id="result" style="display:none">Results shown</div>
            <script>
                document.getElementById('myform').addEventListener('submit', (e) => {
                    e.preventDefault();
                    document.getElementById('result').style.display = 'block';
                });
            </script>
        ''')

        page.locator('input[name="query"]').fill("test query")
        page.get_by_role("button", name="Search").click()

        expect(page.locator("#result")).to_be_visible()


# =============================================================================
# 4. NETWORK INTERCEPTION
# =============================================================================
# Intercept and mock network requests — essential for:
#   1. Testing error states (500, 404, timeout)
#   2. Testing with specific API responses
#   3. Making tests fast by avoiding real network calls

class TestNetworkInterception:
    """Mock API responses using Playwright's route interception."""

    def test_mock_api_response(self, page):
        """Intercept API calls and return mock data."""
        # Set up route interception BEFORE navigation
        page.route("**/api/users", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='[{"id": 1, "name": "Mock User"}]'
        ))

        page.set_content('''
            <div id="users"></div>
            <script>
                fetch('/api/users')
                    .then(r => r.json())
                    .then(users => {
                        document.getElementById('users').textContent =
                            users.map(u => u.name).join(', ');
                    });
            </script>
        ''')

        expect(page.locator("#users")).to_have_text("Mock User")

    def test_mock_api_error(self, page):
        """Test UI behavior when API returns an error."""
        page.route("**/api/data", lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body='{"error": "Internal Server Error"}'
        ))

        page.set_content('''
            <div id="status">Loading...</div>
            <script>
                fetch('/api/data')
                    .then(r => {
                        if (!r.ok) throw new Error('API Error');
                        return r.json();
                    })
                    .then(data => {
                        document.getElementById('status').textContent = 'OK';
                    })
                    .catch(err => {
                        document.getElementById('status').textContent = 'Error: ' + err.message;
                    });
            </script>
        ''')

        expect(page.locator("#status")).to_have_text("Error: API Error")

    def test_wait_for_network(self, page):
        """Wait for specific network requests to complete."""
        page.route("**/api/slow", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"loaded": true}'
        ))

        page.set_content('''
            <button onclick="loadData()">Load</button>
            <div id="result"></div>
            <script>
                async function loadData() {
                    const r = await fetch('/api/slow');
                    const d = await r.json();
                    document.getElementById('result').textContent = d.loaded ? 'Loaded!' : 'Failed';
                }
            </script>
        ''')

        # Click and wait for the network response
        with page.expect_response("**/api/slow"):
            page.get_by_role("button", name="Load").click()

        expect(page.locator("#result")).to_have_text("Loaded!")


# =============================================================================
# 5. SCREENSHOT AND VISUAL COMPARISON
# =============================================================================

class TestVisualTesting:
    """Demonstrate screenshot capabilities."""

    def test_take_screenshot(self, page, tmp_path):
        """Capture a screenshot for visual inspection or comparison."""
        page.set_content('''
            <h1 style="color: navy; font-family: Arial;">Hello Playwright</h1>
            <p>This is a test page for visual testing.</p>
        ''')

        screenshot_path = tmp_path / "screenshot.png"
        page.screenshot(path=str(screenshot_path))
        assert screenshot_path.exists()
        assert screenshot_path.stat().st_size > 0

    def test_element_screenshot(self, page, tmp_path):
        """Capture a screenshot of a specific element only."""
        page.set_content('''
            <div id="card" style="border: 2px solid blue; padding: 20px; width: 200px;">
                <h2>Card Title</h2>
                <p>Card content here.</p>
            </div>
        ''')

        screenshot_path = tmp_path / "card.png"
        page.locator("#card").screenshot(path=str(screenshot_path))
        assert screenshot_path.exists()


# =============================================================================
# 6. PAGE OBJECT MODEL PATTERN
# =============================================================================
# The Page Object Model encapsulates page interactions into classes,
# making tests more maintainable and readable.

class LoginPage:
    """Page Object for a login page.
    Encapsulates selectors and interactions — if the UI changes,
    only this class needs updating, not every test."""

    def __init__(self, page):
        self.page = page
        # Define locators once
        self.username_input = page.get_by_label("Username")
        self.password_input = page.get_by_label("Password")
        self.submit_button = page.get_by_role("button", name="Login")
        self.error_message = page.locator(".error-message")

    def navigate(self):
        self.page.set_content('''
            <form id="login-form">
                <label for="username">Username</label>
                <input id="username" type="text" />
                <label for="password">Password</label>
                <input id="password" type="password" />
                <button type="submit">Login</button>
                <div class="error-message" style="display:none; color:red"></div>
            </form>
            <div id="dashboard" style="display:none">Welcome!</div>
            <script>
                document.getElementById('login-form').addEventListener('submit', (e) => {
                    e.preventDefault();
                    const u = document.getElementById('username').value;
                    const p = document.getElementById('password').value;
                    if (u === 'admin' && p === 'password') {
                        document.getElementById('dashboard').style.display = 'block';
                        document.getElementById('login-form').style.display = 'none';
                    } else {
                        const err = document.querySelector('.error-message');
                        err.textContent = 'Invalid credentials';
                        err.style.display = 'block';
                    }
                });
            </script>
        ''')
        return self

    def login(self, username: str, password: str):
        self.username_input.fill(username)
        self.password_input.fill(password)
        self.submit_button.click()
        return self


class TestPageObjectModel:
    """Tests using the Page Object Model pattern."""

    @pytest.fixture
    def login_page(self, page):
        return LoginPage(page).navigate()

    def test_successful_login(self, login_page, page):
        """Test reads like a user story: navigate, login, verify."""
        login_page.login("admin", "password")
        expect(page.locator("#dashboard")).to_be_visible()

    def test_failed_login(self, login_page):
        """Failed login shows error message."""
        login_page.login("wrong", "wrong")
        expect(login_page.error_message).to_be_visible()
        expect(login_page.error_message).to_have_text("Invalid credentials")

    def test_empty_submission(self, login_page):
        """Submitting empty form shows error."""
        login_page.login("", "")
        expect(login_page.error_message).to_be_visible()


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pip install playwright pytest
# playwright install chromium
# pytest 09_playwright_example.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
