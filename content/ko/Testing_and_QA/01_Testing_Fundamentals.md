# 테스팅 기초 (Testing Fundamentals)

**이전**: [개요](./00_Overview.md) | **다음**: [pytest를 이용한 단위 테스팅](./02_Unit_Testing_with_pytest.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단위 테스트, 통합 테스트, 엔드 투 엔드 테스트, 인수 테스트를 구분할 수 있다
2. 테스트 피라미드를 적용하여 테스트 스위트 구성의 균형을 잡을 수 있다
3. 테스팅 노력의 경제적 트레이드오프를 평가할 수 있다
4. 일반적인 테스팅 안티패턴을 식별하고 피할 수 있다
5. 단순한 통과/실패 지표를 넘어 테스트 품질을 평가할 수 있다

---

## 테스팅이 중요한 이유

소프트웨어 테스팅은 코드가 동작한다는 것을 증명하는 것이 아닙니다. 동작하지 않는 부분을 찾는 것입니다. 프로덕션에 도달하는 모든 버그는 비용을 수반합니다: 매출 손실, 평판 손상, 새 기능 개발에서 전환된 엔지니어링 시간, 때로는 인명 안전까지. 테스팅은 이 리스크를 줄이기 위한 체계적인 실천입니다.

흔한 오해 중 하나는 테스팅이 개발 속도를 늦춘다는 것입니다. 실제로 테스트되지 않은 코드는 *나중에* 개발을 늦춥니다. 프로덕션 인시던트 디버깅, 수동으로 변경 사항 확인, 리팩토링에 대한 두려움은 사전에 테스트를 작성하는 것보다 훨씬 더 비쌉니다.

```python
# Without tests: "I think this works"
def calculate_discount(price, percentage):
    return price * percentage / 100

# With tests: "I know this works — and I know when it breaks"
def test_calculate_discount():
    assert calculate_discount(100, 10) == 10.0
    assert calculate_discount(200, 50) == 100.0
    assert calculate_discount(0, 25) == 0.0
```

---

## 테스트 유형

### 단위 테스트 (Unit Tests)

단위 테스트는 단일 함수, 메서드 또는 클래스를 격리된 상태에서 검증합니다. 빠르고, 결정론적이며, 신뢰할 수 있는 테스트 스위트의 기초를 형성합니다.

**특성:**
- 밀리초 단위로 실행
- I/O 없음 (데이터베이스, 네트워크, 파일 시스템 없음)
- 테스트 함수당 하나의 동작을 테스트
- 실패 원인 파악이 용이

```python
# Function under test
def validate_email(email: str) -> bool:
    """Return True if email has a valid format."""
    import re
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# Unit tests
def test_valid_email():
    assert validate_email("user@example.com") is True

def test_email_without_at_sign():
    assert validate_email("userexample.com") is False

def test_email_without_domain():
    assert validate_email("user@") is False

def test_email_with_subdomain():
    assert validate_email("user@mail.example.com") is True
```

### 통합 테스트 (Integration Tests)

통합 테스트는 여러 컴포넌트가 함께 올바르게 동작하는지 검증합니다. 모듈 경계를 넘으며, 실제 외부 시스템(데이터베이스, 파일 시스템, API)을 포함하는 경우가 많습니다.

**특성:**
- 단위 테스트보다 느림 (초에서 분 단위)
- 외부 리소스의 설정/해제가 필요할 수 있음
- 컴포넌트 간의 *이음새(seam)*를 테스트
- 단위 테스트가 놓치는 연결 버그를 포착

```python
import sqlite3

def test_user_repository_stores_and_retrieves():
    """Integration test: repository + database working together."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    # Store
    conn.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    conn.commit()

    # Retrieve
    row = conn.execute("SELECT name FROM users WHERE id = 1").fetchone()
    assert row[0] == "Alice"
    conn.close()
```

### 엔드 투 엔드 (E2E) 테스트

E2E 테스트는 사용자의 관점에서 전체 애플리케이션을 실행합니다. 웹 애플리케이션의 경우, 브라우저를 실행하고, 페이지를 탐색하고, 버튼을 클릭하고, 결과를 검증하는 것을 의미합니다.

**특성:**
- 실행 속도가 가장 느림 (테스트당 수 초에서 수 분)
- 가장 현실적 — 실제 사용자 워크플로우를 시뮬레이션
- 가장 취약 — UI가 변경되면 깨짐
- 실행 중인 애플리케이션 스택이 필요

```python
# Conceptual E2E test (Playwright example)
def test_user_can_log_in(page):
    page.goto("http://localhost:8000/login")
    page.fill("#username", "alice")
    page.fill("#password", "secret123")
    page.click("button[type=submit]")
    assert page.text_content("h1") == "Welcome, Alice"
```

### 인수 테스트 (Acceptance Tests)

인수 테스트는 소프트웨어가 비즈니스 요구사항을 충족하는지 검증합니다. 이해관계자와 협업하여 작성되는 경우가 많으며, 시스템이 *어떻게* 동작하는지가 아니라 *무엇을* 해야 하는지를 기술합니다.

```python
# Acceptance test: business language, not implementation details
def test_premium_users_get_free_shipping():
    """Business rule: orders over $50 from premium users ship free."""
    user = create_user(membership="premium")
    order = create_order(user=user, total=75.00)
    assert order.shipping_cost == 0.0

def test_regular_users_pay_shipping():
    """Business rule: regular users always pay shipping."""
    user = create_user(membership="regular")
    order = create_order(user=user, total=75.00)
    assert order.shipping_cost > 0.0
```

---

## 테스트 피라미드

테스트 피라미드는 테스트 유형 간의 균형을 잡기 위한 모델로, Mike Cohn이 제안했습니다. 하단에 많은 단위 테스트를, 중간에 적은 수의 통합 테스트를, 상단에 더 적은 수의 E2E 테스트를 권장합니다.

```
        /  E2E  \          Slow, expensive, realistic
       /----------\
      / Integration \      Medium speed, cross-boundary
     /----------------\
    /    Unit Tests     \  Fast, cheap, isolated
   /____________________\
```

**왜 이런 형태인가?**

| 속성             | 단위   | 통합        | E2E     |
|-----------------|--------|-------------|---------|
| 속도            | ~1 ms  | ~100 ms     | ~5 sec  |
| 작성 비용       | 낮음   | 중간        | 높음    |
| 유지보수 비용   | 낮음   | 중간        | 높음    |
| 실패 명확성     | 높음   | 중간        | 낮음    |
| 신뢰도          | 낮음   | 중간        | 높음    |

피라미드는 가이드라인이지, 법칙이 아닙니다. 일부 애플리케이션(예: 얇은 로직의 CRUD 앱)은 통합 테스트를 더 많이 하는 것이 유리할 수 있습니다. 핵심 통찰은: **충분한 신뢰를 제공하는 가장 낮은 수준으로 테스트를 밀어 내리는 것**입니다.

### 아이스크림 콘 안티패턴

테스트 피라미드의 역전입니다. 대부분 수동 또는 E2E 테스트이고 단위 테스트가 적은 팀은 다음을 경험합니다:

- 느린 피드백 루프 (CI가 수 시간 소요)
- 신뢰를 잠식하는 불안정한 테스트 스위트
- UI 변경 시 높은 유지보수 비용
- 로컬에서 테스트 실행을 피하는 개발자

---

## 테스팅 경제학

### 시간에 따른 버그 비용

버그가 늦게 발견될수록 수정 비용이 더 높습니다:

| 단계               | 상대적 비용 |
|-------------------|-------------|
| 코딩 중            | 1x          |
| 코드 리뷰          | 2-5x        |
| QA / 테스팅        | 5-15x       |
| 프로덕션           | 30-100x     |

이것이 테스팅을 *왼쪽으로 이동*(개발 프로세스 초기에)시키는 것이 효과적인 이유입니다.

### 무엇을 테스트할 것인가

모든 것에 같은 수준의 테스팅이 필요한 것은 아닙니다. 우선순위를 정하세요:

1. **비즈니스 핵심 경로** — 결제 처리, 인증, 데이터 무결성
2. **복잡한 로직** — 알고리즘, 상태 머신, 계산
3. **엣지 케이스** — 빈 입력, 경계값, 오류 조건
4. **버그가 잦은 영역** — 이전에 깨졌던 코드는 다시 깨질 가능성이 높음
5. **공개 API** — 다른 코드나 사용자가 의존하는 인터페이스

### 과도하게 테스트하지 않아야 할 것

- 로직이 없는 단순한 getter/setter
- 프레임워크가 제공하는 기능 (Flask가 200을 반환하는지 테스트하지 마세요)
- 자주 변경되는 구현 세부사항
- 서드파티 라이브러리 내부

---

## 테스트 품질 vs 수량

1,000개의 테스트가 있어도 모두 동일한 해피 패스만 테스트한다면 의미가 없습니다. 품질 차원에는 다음이 포함됩니다:

### 동작 커버리지 (Behavior Coverage)

테스트 스위트가 코드 라인이 아니라 중요한 *동작*을 커버하고 있나요?

```python
# Poor: tests the happy path only
def test_divide():
    assert divide(10, 2) == 5.0

# Better: tests behavior including edge cases
def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_divide_negative_numbers():
    assert divide(-10, 2) == -5.0

def test_divide_returns_float():
    assert isinstance(divide(7, 2), float)
```

### 테스트 독립성

각 테스트는 격리되어 실행될 수 있어야 하며, 어떤 순서로든 실행 가능해야 합니다. 실행 순서에 의존하는 테스트는 유지보수의 악몽입니다.

```python
# BAD: test_b depends on test_a having run first
class TestBadDependency:
    result = None

    def test_a_create(self):
        TestBadDependency.result = create_item("widget")
        assert self.result is not None

    def test_b_read(self):
        # Fails if test_a did not run first!
        item = get_item(TestBadDependency.result.id)
        assert item.name == "widget"

# GOOD: each test sets up its own data
def test_create_item():
    item = create_item("widget")
    assert item is not None
    assert item.name == "widget"

def test_read_item():
    item = create_item("widget")  # Own setup
    retrieved = get_item(item.id)
    assert retrieved.name == "widget"
```

### 결정론성 (Determinism)

불안정한 테스트(flaky test)란 코드 변경 없이도 때때로 통과하고 때때로 실패하는 테스트로, 신뢰를 파괴합니다. 일반적인 원인:

- 시간 의존 로직 (`datetime.now()`)
- 시드 없는 랜덤 데이터
- 테스트 간 공유된 가변 상태
- 외부 서비스로의 네트워크 호출
- 비동기 코드에서의 경쟁 조건

---

## 테스팅 안티패턴

### 1. 거대한 테스트

단일 테스트 함수에서 20가지를 검증하는 경우. 실패하면 어떤 동작이 깨졌는지 알 수 없습니다.

### 2. 동작이 아닌 구현을 테스트

```python
# Anti-pattern: testing HOW, not WHAT
def test_sort_uses_quicksort():
    with patch("mymodule.quicksort") as mock_qs:
        sort_data([3, 1, 2])
        mock_qs.assert_called_once()

# Better: testing WHAT the function achieves
def test_sort_returns_sorted_list():
    assert sort_data([3, 1, 2]) == [1, 2, 3]
```

### 3. 거짓말하는 테스트 (The Liar Test)

어서션이 잘못되었거나 누락되어 정확성과 관계없이 항상 통과하는 테스트입니다.

```python
# Anti-pattern: no real assertion
def test_process_data():
    result = process_data([1, 2, 3])
    assert result is not None  # This tells you almost nothing
```

### 4. 과도한 Mocking

Mock을 너무 많이 사용하여 코드가 아닌 mock을 테스트하는 경우. 테스트에 2-3개 이상의 mock이 필요하다면, 테스트 대상 코드의 의존성이 너무 많은 것은 아닌지 고려하세요.

### 5. 복사-붙여넣기 테스트

하나의 값만 다른 거의 동일한 테스트가 수십 개인 경우. 대신 매개변수화된 테스트를 사용하세요 (레슨 03에서 다룸).

### 6. 테스트 유지보수 무시

테스트는 코드입니다. 프로덕션 코드와 마찬가지로 리팩토링, 문서화, 리뷰가 필요합니다. 방치된 테스트 스위트는 부채가 됩니다.

---

## 언제 테스트를 작성할 것인가

| 접근법                 | 적합한 경우                                  |
|-----------------------|----------------------------------------------|
| 테스트 우선 (TDD)      | 잘 정의된 요구사항, 알고리즘 코드             |
| 테스트 후 작성         | 탐색적/프로토타입 코드, UI 실험              |
| 테스트와 함께 작성     | 대부분의 프로덕션 코드 — 코드와 함께 테스트 작성 |
| 회귀 테스트            | 버그 발견 후 — 재발 방지                      |

테스트를 작성하기 가장 좋은 시점은 기대하는 동작을 명확하게 표현할 수 있을 때입니다. 테스트를 *작성하지 않기에* 가장 나쁜 시점은 배포 직전입니다.

---

## 테스트 프로젝트 설정

테스팅을 포함한 최소 Python 프로젝트:

```
my_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── calculator.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_calculator.py
├── pyproject.toml
└── README.md
```

```toml
# pyproject.toml
[project]
name = "my-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

```python
# src/my_project/calculator.py
def add(a: float, b: float) -> float:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

```python
# tests/test_calculator.py
import pytest
from my_project.calculator import add, divide

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -1) == -2

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        divide(10, 0)
```

다음 명령으로 실행합니다:

```bash
pip install pytest
pytest
```

---

## 연습 문제

1. **테스트 분류하기**: 테스트 설명 목록이 주어졌을 때, 각각을 단위 테스트, 통합 테스트, E2E 테스트, 인수 테스트로 분류하세요. 각각에 대한 근거를 설명하세요.

2. **안티패턴 찾기**: 다음 테스트 스위트를 검토하고 최소 3가지 안티패턴을 식별하세요. 이를 수정하여 테스트를 다시 작성하세요.

3. **테스트 계획 설계**: 간단한 도서관 관리 시스템(도서 추가, 대출, 반납, 제목 검색)에 대해, 테스트 피라미드의 각 수준에서 어떤 동작을 테스트할지 정리하세요. 각 테스트를 해당 위치에 배치한 이유를 설명하세요.

---

**License**: CC BY-NC 4.0
