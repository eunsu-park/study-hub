# Lesson 17: Testing Legacy Code

**Previous**: [Database Testing](./16_Database_Testing.md) | **Next**: [Test Strategy and Planning](./18_Test_Strategy_and_Planning.md)

---

Legacy code is code without tests. That is not a moral judgment — it is a practical definition from Michael Feathers' *Working Effectively with Legacy Code*. Without tests, you cannot change the code with confidence. Without confidence, changes become risky, slow, and error-prone. The irony is that the code most in need of change is often the code most dangerous to change. This lesson addresses the central challenge: how do you add tests to code that was never designed to be tested?

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**:
- Strong pytest skills (Lessons 02–06)
- Understanding of test doubles (Lesson 14)
- Familiarity with refactoring concepts
- Experience maintaining existing codebases

## Learning Objectives

After completing this lesson, you will be able to:

1. Write characterization tests (Golden Master tests) to capture existing behavior
2. Identify and exploit seams to inject test doubles into untestable code
3. Apply dependency-breaking techniques to make code testable
4. Use approval testing to lock in complex outputs
5. Plan an incremental strategy for introducing tests to a legacy codebase

---

## 1. The Legacy Code Dilemma

The fundamental problem is circular:

```
To change code safely → you need tests
To write tests → you need to change the code (make it testable)
To change the code → you need tests to do it safely
```

Breaking this cycle requires careful, minimal changes that make code testable without altering its behavior. Every technique in this lesson is designed to minimize the risk of that initial change.

### 1.1 Why Legacy Code Resists Testing

Common characteristics of untestable code:

```python
# Everything in one function, external calls embedded throughout
def process_monthly_report():
    # Direct database access — no dependency injection
    conn = psycopg2.connect("postgresql://prod:secret@db-server/reports")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions WHERE month = current_month()")
    rows = cursor.fetchall()

    # Business logic mixed with I/O
    total = 0
    for row in rows:
        amount = row[3] * get_exchange_rate(row[4])  # HTTP call
        total += amount

    # Direct file system access
    with open(f"/reports/{datetime.now().strftime('%Y-%m')}.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    # Direct email sending
    send_email("finance@company.com", f"Monthly Total: ${total:.2f}")

    return total
```

This function is impossible to unit test: it connects to a database, makes HTTP calls, writes files, and sends emails. Testing it would require all those systems to be running.

---

## 2. Characterization Tests (Golden Master)

Before refactoring, you need to know what the code currently does — not what it should do, but what it actually does. Characterization tests capture the current behavior, including any bugs.

### 2.1 The Process

1. Call the code with a known input
2. Record the output (whatever it is)
3. Write a test that asserts the output matches the recorded value

```python
def test_characterize_calculate_discount():
    """
    Characterization test: captures CURRENT behavior, not DESIRED behavior.
    If this test breaks after a refactor, investigate whether the change
    was intentional or accidental.
    """
    # These expected values were determined by running the function
    # and recording what it returned — NOT from a specification.
    assert calculate_discount(100, "premium") == 15.0
    assert calculate_discount(100, "standard") == 5.0
    assert calculate_discount(100, "unknown_type") == 0.0
    assert calculate_discount(0, "premium") == 0.0
    assert calculate_discount(-50, "premium") == -7.5  # Bug? Maybe. But it's current behavior.
```

### 2.2 Characterization Tests for Complex Output

For functions with complex output (reports, HTML, data structures), capture the full output:

```python
import json
import os


def capture_golden_master(func, *args, filename, **kwargs):
    """Run the function and save its output as a golden master file."""
    result = func(*args, **kwargs)
    golden_path = os.path.join("tests", "golden", filename)
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    with open(golden_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def test_report_generation():
    """Compare against golden master output."""
    golden_path = "tests/golden/monthly_report.json"

    result = generate_report(year=2024, month=1, data=SAMPLE_DATA)

    if not os.path.exists(golden_path):
        # First run: create the golden master
        with open(golden_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        pytest.skip("Golden master created. Review and re-run.")

    with open(golden_path) as f:
        expected = json.load(f)

    assert result == expected, (
        "Output differs from golden master. "
        "If this change is intentional, delete the golden master and re-run."
    )
```

### 2.3 When to Use Characterization Tests

- **Before refactoring**: Lock in current behavior before changing code structure
- **Before adding features**: Ensure existing functionality is not broken
- **When documentation is missing**: The tests become living documentation
- **When you do not understand the code**: The tests reveal what it does

---

## 3. Seam-Based Testing

A **seam** is a place in the code where you can change behavior without editing the code itself. Michael Feathers identifies three types of seams:

### 3.1 Object Seams (Dependency Injection)

Replace a dependency by passing it as a parameter:

```python
# BEFORE: Hard dependency, untestable
class OrderProcessor:
    def process(self, order):
        gateway = StripeGateway()  # Hardcoded dependency
        result = gateway.charge(order.total)
        db = PostgresDB()          # Another hardcoded dependency
        db.save_order(order, result)
        return result


# AFTER: Object seam via constructor injection
class OrderProcessor:
    def __init__(self, gateway=None, db=None):
        self.gateway = gateway or StripeGateway()
        self.db = db or PostgresDB()

    def process(self, order):
        result = self.gateway.charge(order.total)
        self.db.save_order(order, result)
        return result


# Now testable:
def test_order_processing():
    mock_gateway = Mock()
    mock_gateway.charge.return_value = {"status": "success"}
    mock_db = Mock()

    processor = OrderProcessor(gateway=mock_gateway, db=mock_db)
    result = processor.process(Order(total=50.00))

    mock_gateway.charge.assert_called_once_with(50.00)
    mock_db.save_order.assert_called_once()
```

### 3.2 Link Seams (Module-Level Substitution)

Replace a module-level dependency using `unittest.mock.patch`:

```python
# legacy_module.py — cannot be easily modified
import requests


def fetch_user_data(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()


# Test using a link seam (patch at the import level)
from unittest.mock import patch


def test_fetch_user_data():
    mock_response = Mock()
    mock_response.json.return_value = {"id": 1, "name": "alice"}
    mock_response.status_code = 200

    with patch("legacy_module.requests.get", return_value=mock_response):
        result = fetch_user_data(1)
        assert result["name"] == "alice"
```

### 3.3 Preprocessor Seams (Environment-Based)

Use environment variables or configuration to alter behavior:

```python
import os


def get_database_url():
    """Seam: behavior changes based on environment."""
    if os.environ.get("TESTING"):
        return "sqlite:///:memory:"
    return os.environ.get("DATABASE_URL", "postgresql://localhost/prod")


# Test:
def test_with_test_database(monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    url = get_database_url()
    assert "sqlite" in url
```

---

## 4. Dependency Breaking Techniques

When code has deep dependencies that make testing impossible, apply these targeted techniques to introduce seams.

### 4.1 Extract and Override

Extract the problematic code into a method, then override it in a test subclass:

```python
# BEFORE: Untestable — datetime.now() is non-deterministic
class ReportGenerator:
    def generate(self, data):
        timestamp = datetime.now()
        # ... complex logic using timestamp ...
        return {"timestamp": timestamp, "data": processed_data}


# AFTER: Extract datetime.now() into an overridable method
class ReportGenerator:
    def _get_current_time(self):
        return datetime.now()

    def generate(self, data):
        timestamp = self._get_current_time()
        # ... same complex logic ...
        return {"timestamp": timestamp, "data": processed_data}


# Test subclass overrides just the time method
class TestableReportGenerator(ReportGenerator):
    def _get_current_time(self):
        return datetime(2024, 6, 15, 12, 0, 0)


def test_report_generation():
    generator = TestableReportGenerator()
    result = generator.generate(sample_data)
    assert result["timestamp"] == datetime(2024, 6, 15, 12, 0, 0)
```

### 4.2 Wrap and Delegate

Wrap legacy code in a new class with a testable interface:

```python
# Legacy code you cannot modify (third-party, generated, etc.)
class LegacyPaymentSystem:
    def make_payment(self, amount, card, cvv, exp, name, addr1, addr2, city, state, zip_code):
        # 500 lines of untestable code
        ...


# Wrapper with a clean interface
class PaymentGateway:
    def __init__(self, legacy_system=None):
        self._system = legacy_system or LegacyPaymentSystem()

    def charge(self, amount: float, card_info: CardInfo) -> PaymentResult:
        try:
            result = self._system.make_payment(
                amount,
                card_info.number,
                card_info.cvv,
                card_info.expiry,
                card_info.holder_name,
                card_info.address.line1,
                card_info.address.line2,
                card_info.address.city,
                card_info.address.state,
                card_info.address.zip_code,
            )
            return PaymentResult(success=True, transaction_id=result)
        except Exception as e:
            return PaymentResult(success=False, error=str(e))


# Test the wrapper, mock the legacy system
def test_charge_success():
    mock_legacy = Mock()
    mock_legacy.make_payment.return_value = "TX123"

    gateway = PaymentGateway(legacy_system=mock_legacy)
    result = gateway.charge(50.00, card_info)

    assert result.success is True
    assert result.transaction_id == "TX123"
```

### 4.3 Sprout Method

When you need to add new functionality to a legacy function, do not modify it — add the new logic in a separate, testable method:

```python
# Legacy function — 200 lines, complex, fragile
def process_order(order_data):
    # ... 200 lines of untested code ...
    total = complex_calculation(order_data)
    # New requirement: apply loyalty discount
    # DON'T add logic here — sprout a new method
    total = apply_loyalty_discount(total, order_data["customer_id"])
    return total


# New, testable method
def apply_loyalty_discount(total: float, customer_id: int) -> float:
    """Sprout method: new logic isolated from legacy code."""
    loyalty_tier = get_loyalty_tier(customer_id)
    discount_rates = {"gold": 0.10, "silver": 0.05, "bronze": 0.02}
    discount = discount_rates.get(loyalty_tier, 0)
    return total * (1 - discount)


# Fully testable
def test_gold_loyalty_discount():
    with patch("myapp.orders.get_loyalty_tier", return_value="gold"):
        result = apply_loyalty_discount(100.00, customer_id=42)
        assert result == 90.00
```

---

## 5. The Strangler Fig Pattern for Test Migration

The Strangler Fig pattern (originally for system migration) applies to test migration: gradually replace untested legacy code with tested new code, one piece at a time.

### 5.1 The Process

```
Phase 1: Characterize (do not change code)
├── Write characterization tests for the most critical paths
├── Cover the highest-risk areas first
└── Build a safety net before any changes

Phase 2: Extract (minimal changes)
├── Identify seams in the legacy code
├── Extract dependencies into injectable parameters
├── Write unit tests for the extracted pieces
└── Verify characterization tests still pass

Phase 3: Replace (incremental)
├── Rewrite one module at a time with tests
├── Keep the old code running alongside (feature flags)
├── Compare outputs between old and new
└── Remove old code only when confident

Phase 4: Maintain
├── All new code requires tests (enforce in code review)
├── Every bug fix includes a regression test
└── Gradually expand coverage as you touch legacy code
```

### 5.2 Implementation Example

```python
# Phase 1: Characterize the legacy pricing function
def test_characterize_legacy_pricing():
    """Capture current behavior of the legacy pricing engine."""
    assert legacy_calculate_price("SKU001", 1) == 29.99
    assert legacy_calculate_price("SKU001", 10) == 269.91  # 10% volume discount
    assert legacy_calculate_price("SKU001", 100) == 2399.20  # 20% volume discount


# Phase 2: Extract the pricing logic into a testable module
class PricingEngine:
    def __init__(self, catalog=None):
        self.catalog = catalog or ProductCatalog()

    def calculate_price(self, sku: str, quantity: int) -> float:
        base_price = self.catalog.get_price(sku)
        discount = self._volume_discount(quantity)
        return round(base_price * quantity * (1 - discount), 2)

    def _volume_discount(self, quantity: int) -> float:
        if quantity >= 100:
            return 0.20
        elif quantity >= 10:
            return 0.10
        return 0.0


# Phase 3: Verify new implementation matches legacy
def test_new_pricing_matches_legacy():
    """Ensure the new engine produces identical results."""
    catalog = Mock()
    catalog.get_price.return_value = 29.99
    engine = PricingEngine(catalog=catalog)

    assert engine.calculate_price("SKU001", 1) == 29.99
    assert engine.calculate_price("SKU001", 10) == 269.91
    assert engine.calculate_price("SKU001", 100) == 2399.20
```

---

## 6. Approval Testing

Approval testing (also called snapshot testing) automates the golden master approach. The [approvaltests](https://github.com/approvals/ApprovalTests.Python) library handles file comparison and approval workflow.

### 6.1 Installation

```bash
pip install approvaltests
```

### 6.2 Basic Approval Test

```python
from approvaltests import verify, verify_as_json


def test_report_output():
    """Verify the report output matches the approved version."""
    report = generate_report(sample_data)
    verify(report)  # Compares against .approved.txt file


def test_api_response():
    """Verify the API response structure."""
    response = get_user_profile(user_id=1)
    verify_as_json(response)  # Compares JSON against approved file
```

### 6.3 How Approval Testing Works

```
First run:
  1. Test generates output → saves to test_name.received.txt
  2. No .approved.txt exists → test fails
  3. Developer reviews .received.txt
  4. If correct: rename to .approved.txt (approve it)

Subsequent runs:
  1. Test generates output → saves to test_name.received.txt
  2. Compares against test_name.approved.txt
  3. If identical → test passes
  4. If different → test fails, showing the diff
```

### 6.4 Approval Testing for Legacy Refactoring

```python
def test_approve_legacy_html_output():
    """Lock in the current HTML output before refactoring the template."""
    html = render_dashboard(
        user=sample_user,
        transactions=sample_transactions,
        settings=default_settings,
    )
    verify(html)


# After refactoring the template engine:
# - If the HTML output is identical → test passes → refactor is safe
# - If the HTML output differs → test fails → investigate the difference
```

---

## 7. Introducing Tests Incrementally

### 7.1 The Boy Scout Rule for Tests

"Leave the code better tested than you found it." Every time you touch legacy code:

1. Write a characterization test for the current behavior
2. Make your change
3. Write a test for the new behavior
4. Verify the characterization test still passes (unless your change intentionally altered behavior)

### 7.2 Prioritizing What to Test First

| Priority | Criteria | Rationale |
|---|---|---|
| 1 | Code that changes frequently | High change rate = high risk |
| 2 | Code with known bugs | Bugs indicate fragile logic |
| 3 | Code in critical paths | Failures have maximum impact |
| 4 | Code with upcoming changes | Investment pays off immediately |
| 5 | Code that is simple to test | Quick wins build momentum |

### 7.3 Tracking Progress

```python
# conftest.py — coverage tracking for legacy code improvement
import json
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    """Record coverage trend over time."""
    trend_file = Path("tests/coverage_trend.json")
    if trend_file.exists():
        trend = json.loads(trend_file.read_text())
    else:
        trend = []

    trend.append({
        "date": datetime.now().isoformat(),
        "total_tests": session.testscollected,
        "passed": session.testsfailed == 0,
    })
    trend_file.write_text(json.dumps(trend, indent=2))
```

---

## 8. Common Mistakes When Testing Legacy Code

1. **Trying to reach 100% coverage immediately**: Focus on high-risk areas first. Coverage grows incrementally over months.

2. **Rewriting instead of refactoring**: A big rewrite without the safety net of tests is the most dangerous approach. Refactor in small, tested steps.

3. **Treating characterization tests as permanent**: They capture current behavior, including bugs. As you fix bugs, update or replace the characterization tests with proper specification tests.

4. **Changing too much at once**: Each change should be small enough that if a test breaks, the cause is obvious.

5. **Not communicating with the team**: Legacy code testing is a team effort. Share the strategy, track progress visibly, and celebrate milestones.

---

## Exercises

1. **Characterization Test**: Take a function from a real project (or the sample code) and write characterization tests for it. Run it with at least 10 different inputs and record the outputs. Include at least one case where the behavior seems wrong.

2. **Seam Identification**: Given a 50-line legacy function with three external dependencies (database, HTTP API, file system), identify all seams and propose the least invasive way to make it testable.

3. **Extract and Override**: Take a class with a hardcoded dependency on `datetime.now()` and apply the Extract and Override technique. Write tests for time-dependent behavior.

4. **Sprout Method**: Given a legacy function that needs new functionality (e.g., adding input validation), implement it as a sprout method. Write full tests for the new method.

5. **Strangler Fig Plan**: For a module with 500 lines of untested code, create a written plan for introducing tests over 4 sprints. Prioritize by risk and change frequency. Estimate effort for each phase.

---

**License**: CC BY-NC 4.0
