# 레슨 17: 레거시 코드 테스트

**이전**: [Database Testing](./16_Database_Testing.md) | **다음**: [Test Strategy and Planning](./18_Test_Strategy_and_Planning.md)

---

레거시 코드는 테스트가 없는 코드입니다. 이것은 도덕적 판단이 아니라 Michael Feathers의 *Working Effectively with Legacy Code*에서 나온 실용적인 정의입니다. 테스트 없이는 확신을 갖고 코드를 변경할 수 없습니다. 확신이 없으면 변경은 위험하고, 느리고, 오류가 발생하기 쉬워집니다. 아이러니하게도 변경이 가장 필요한 코드가 종종 변경하기 가장 위험한 코드입니다. 이 레슨은 핵심 과제를 다룹니다: 테스트 가능하도록 설계되지 않은 코드에 어떻게 테스트를 추가하는가?

**난이도**: ⭐⭐⭐⭐

**사전 요구사항**:
- 강력한 pytest 기술 (레슨 02-06)
- 테스트 대역에 대한 이해 (레슨 14)
- 리팩토링 개념에 대한 익숙함
- 기존 코드베이스 유지보수 경험

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 기존 동작을 포착하는 특성화 테스트(Golden Master 테스트)를 작성할 수 있다
2. 테스트할 수 없는 코드에 테스트 대역을 주입하기 위한 심(seam)을 식별하고 활용할 수 있다
3. 코드를 테스트 가능하게 만드는 의존성 분리 기법을 적용할 수 있다
4. 복잡한 출력을 고정하기 위해 승인 테스트(approval testing)를 사용할 수 있다
5. 레거시 코드베이스에 테스트를 도입하기 위한 점진적 전략을 계획할 수 있다

---

## 1. 레거시 코드 딜레마

근본적인 문제는 순환적입니다:

```
코드를 안전하게 변경하려면 → 테스트가 필요
테스트를 작성하려면 → 코드를 변경해야 함 (테스트 가능하게)
코드를 변경하려면 → 안전하게 하기 위해 테스트가 필요
```

이 순환을 끊으려면 코드의 동작을 변경하지 않으면서 테스트 가능하게 만드는 신중하고 최소한의 변경이 필요합니다. 이 레슨의 모든 기법은 초기 변경의 위험을 최소화하도록 설계되었습니다.

### 1.1 레거시 코드가 테스트에 저항하는 이유

테스트할 수 없는 코드의 일반적인 특징:

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

이 함수는 단위 테스트가 불가능합니다: 데이터베이스에 연결하고, HTTP 호출을 하고, 파일을 쓰고, 이메일을 보냅니다. 테스트하려면 이 모든 시스템이 실행 중이어야 합니다.

---

## 2. 특성화 테스트 (Golden Master)

리팩토링 전에 코드가 현재 무엇을 하는지 알아야 합니다 -- 무엇을 해야 하는지가 아니라 실제로 무엇을 하고 있는지. 특성화 테스트는 버그를 포함하여 현재 동작을 포착합니다.

### 2.1 프로세스

1. 알려진 입력으로 코드를 호출합니다
2. 출력을 기록합니다 (무엇이든)
3. 기록된 값과 출력이 일치하는지 assert하는 테스트를 작성합니다

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

### 2.2 복잡한 출력에 대한 특성화 테스트

복잡한 출력(리포트, HTML, 데이터 구조)을 가진 함수의 경우, 전체 출력을 포착합니다:

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

### 2.3 특성화 테스트 사용 시점

- **리팩토링 전**: 코드 구조를 변경하기 전에 현재 동작을 고정
- **기능 추가 전**: 기존 기능이 깨지지 않도록 보장
- **문서가 없을 때**: 테스트가 살아있는 문서가 됨
- **코드를 이해하지 못할 때**: 테스트가 코드의 동작을 드러냄

---

## 3. 심(Seam) 기반 테스트

**심(seam)**은 코드 자체를 편집하지 않고 동작을 변경할 수 있는 코드의 지점입니다. Michael Feathers는 세 가지 유형의 심을 식별합니다:

### 3.1 객체 심 (의존성 주입)

매개변수로 전달하여 의존성을 교체합니다:

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

### 3.2 링크 심 (모듈 수준 대체)

`unittest.mock.patch`를 사용하여 모듈 수준의 의존성을 교체합니다:

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

### 3.3 전처리기 심 (환경 기반)

환경 변수나 설정을 사용하여 동작을 변경합니다:

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

## 4. 의존성 분리 기법

코드에 테스트를 불가능하게 만드는 깊은 의존성이 있을 때, 심을 도입하기 위해 다음의 타겟팅된 기법을 적용합니다.

### 4.1 추출 후 오버라이드

문제가 되는 코드를 메서드로 추출한 다음, 테스트 서브클래스에서 오버라이드합니다:

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

### 4.2 래핑 후 위임

레거시 코드를 테스트 가능한 인터페이스를 가진 새 클래스로 래핑합니다:

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

### 4.3 분기 메서드 (Sprout Method)

레거시 함수에 새 기능을 추가해야 할 때, 수정하지 말고 별도의 테스트 가능한 메서드에 새 로직을 추가합니다:

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

## 5. 테스트 마이그레이션을 위한 Strangler Fig 패턴

Strangler Fig 패턴(원래 시스템 마이그레이션용)은 테스트 마이그레이션에도 적용됩니다: 테스트되지 않은 레거시 코드를 한 번에 하나씩 테스트된 새 코드로 점진적으로 교체합니다.

### 5.1 프로세스

```
Phase 1: 특성화 (코드 변경 없음)
├── 가장 중요한 경로에 대한 특성화 테스트 작성
├── 가장 위험한 영역부터 커버
└── 변경 전에 안전망 구축

Phase 2: 추출 (최소한의 변경)
├── 레거시 코드에서 심 식별
├── 의존성을 주입 가능한 매개변수로 추출
├── 추출된 부분에 대한 단위 테스트 작성
└── 특성화 테스트가 여전히 통과하는지 검증

Phase 3: 교체 (점진적)
├── 한 번에 하나의 모듈을 테스트와 함께 재작성
├── 이전 코드를 함께 실행 유지 (feature flag)
├── 이전과 새 코드의 출력 비교
└── 확신이 생길 때만 이전 코드 제거

Phase 4: 유지보수
├── 모든 새 코드에 테스트 필수 (코드 리뷰에서 강제)
├── 모든 버그 수정에 회귀 테스트 포함
└── 레거시 코드를 건드릴 때마다 점진적으로 커버리지 확장
```

### 5.2 구현 예제

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

## 6. 승인 테스트 (Approval Testing)

승인 테스트(스냅샷 테스트라고도 함)는 golden master 접근 방식을 자동화합니다. [approvaltests](https://github.com/approvals/ApprovalTests.Python) 라이브러리가 파일 비교와 승인 워크플로우를 처리합니다.

### 6.1 설치

```bash
pip install approvaltests
```

### 6.2 기본 승인 테스트

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

### 6.3 승인 테스트의 작동 방식

```
첫 실행:
  1. 테스트가 출력을 생성 → test_name.received.txt에 저장
  2. .approved.txt가 존재하지 않음 → 테스트 실패
  3. 개발자가 .received.txt를 검토
  4. 올바르면: .approved.txt로 이름 변경 (승인)

이후 실행:
  1. 테스트가 출력을 생성 → test_name.received.txt에 저장
  2. test_name.approved.txt와 비교
  3. 동일하면 → 테스트 통과
  4. 다르면 → 테스트 실패, diff 표시
```

### 6.4 레거시 리팩토링을 위한 승인 테스트

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

## 7. 점진적인 테스트 도입

### 7.1 테스트를 위한 보이스카우트 규칙

"코드를 발견했을 때보다 더 잘 테스트된 상태로 남기십시오." 레거시 코드를 건드릴 때마다:

1. 현재 동작에 대한 특성화 테스트를 작성
2. 변경 수행
3. 새 동작에 대한 테스트 작성
4. 특성화 테스트가 여전히 통과하는지 검증 (의도적으로 동작을 변경한 경우 제외)

### 7.2 먼저 테스트할 대상 우선순위

| 우선순위 | 기준 | 근거 |
|---|---|---|
| 1 | 자주 변경되는 코드 | 높은 변경률 = 높은 위험 |
| 2 | 알려진 버그가 있는 코드 | 버그는 취약한 로직을 나타냄 |
| 3 | 핵심 경로에 있는 코드 | 실패 시 최대 영향 |
| 4 | 곧 변경될 코드 | 투자가 즉시 보상받음 |
| 5 | 테스트하기 쉬운 코드 | 빠른 성과로 추진력 구축 |

### 7.3 진행 상황 추적

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

## 8. 레거시 코드 테스트 시 흔한 실수

1. **즉시 100% 커버리지를 달성하려는 시도**: 고위험 영역에 먼저 집중하십시오. 커버리지는 수개월에 걸쳐 점진적으로 증가합니다.

2. **리팩토링 대신 재작성**: 테스트라는 안전망 없이 대규모 재작성을 하는 것이 가장 위험한 접근 방식입니다. 작고 테스트된 단계로 리팩토링하십시오.

3. **특성화 테스트를 영구적으로 취급**: 버그를 포함한 현재 동작을 포착합니다. 버그를 수정하면서 특성화 테스트를 올바른 명세 테스트로 업데이트하거나 교체하십시오.

4. **한 번에 너무 많이 변경**: 각 변경은 테스트가 깨질 경우 원인이 명확할 만큼 작아야 합니다.

5. **팀과 소통하지 않음**: 레거시 코드 테스트는 팀 노력입니다. 전략을 공유하고, 진행 상황을 가시적으로 추적하며, 이정표를 축하하십시오.

---

## 연습 문제

1. **특성화 테스트**: 실제 프로젝트(또는 샘플 코드)의 함수를 가져와서 특성화 테스트를 작성하십시오. 최소 10개의 다른 입력으로 실행하고 출력을 기록하십시오. 동작이 잘못된 것으로 보이는 경우를 최소 하나 포함하십시오.

2. **심 식별**: 세 가지 외부 의존성(데이터베이스, HTTP API, 파일 시스템)을 가진 50줄짜리 레거시 함수가 주어졌을 때, 모든 심을 식별하고 테스트 가능하게 만드는 가장 침습적이지 않은 방법을 제안하십시오.

3. **추출 후 오버라이드**: `datetime.now()`에 하드코딩된 의존성을 가진 클래스를 가져와서 추출 후 오버라이드 기법을 적용하십시오. 시간 의존적 동작에 대한 테스트를 작성하십시오.

4. **분기 메서드**: 새 기능(예: 입력 검증 추가)이 필요한 레거시 함수가 주어졌을 때, 분기 메서드로 구현하십시오. 새 메서드에 대한 완전한 테스트를 작성하십시오.

5. **Strangler Fig 계획**: 500줄의 테스트되지 않은 코드가 있는 모듈에 대해 4개의 스프린트에 걸쳐 테스트를 도입하기 위한 서면 계획을 작성하십시오. 위험과 변경 빈도에 따라 우선순위를 정하십시오. 각 단계의 노력을 추정하십시오.

---

**License**: CC BY-NC 4.0
