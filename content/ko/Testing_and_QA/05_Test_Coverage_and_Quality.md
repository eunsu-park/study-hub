# 테스트 커버리지와 품질 (Test Coverage and Quality)

**이전**: [Mocking과 Patching](./04_Mocking_and_Patching.md) | **다음**: [테스트 주도 개발](./06_Test_Driven_Development.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `coverage.py`를 사용하여 라인 및 브랜치 커버리지를 측정할 수 있다
2. `.coveragerc`로 커버리지 임계값과 제외 항목을 설정할 수 있다
3. 커버리지 리포트를 해석하고 의미 있는 갭을 식별할 수 있다
4. `mutmut`으로 뮤테이션 테스팅을 적용하여 테스트 효과를 평가할 수 있다
5. 높은 커버리지와 고품질 테스트를 구분할 수 있다

---

## 테스트 커버리지란?

테스트 커버리지는 테스팅 중에 코드의 몇 퍼센트가 실행되는지를 측정합니다. "테스트가 실제로 어떤 라인(또는 브랜치)을 실행했는가?"라는 질문에 답합니다.

커버리지는 테스트 품질의 **필요하지만 충분하지 않은** 지표입니다. 실행된 코드가 반드시 *검증된* 코드는 아닙니다. 테스트가 함수를 실행하면서도 출력을 확인하지 않을 수 있습니다.

```python
# This test "covers" the function but verifies nothing meaningful
def test_misleading_coverage():
    result = complex_calculation(42)
    assert True  # 100% coverage, 0% confidence
```

---

## coverage.py 설정

`coverage.py`는 Python의 표준 커버리지 도구입니다. pytest는 `pytest-cov` 플러그인을 통해 이와 통합됩니다.

```bash
pip install coverage pytest-cov
```

### 커버리지 실행

```bash
# Using coverage directly
coverage run -m pytest
coverage report
coverage html  # Generate HTML report in htmlcov/

# Using pytest-cov (simpler)
pytest --cov=mypackage
pytest --cov=mypackage --cov-report=html
pytest --cov=mypackage --cov-report=term-missing
```

### 리포트 읽기

```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
mypackage/__init__.py       2      0   100%
mypackage/calculator.py    25      3    88%   42-44
mypackage/validator.py     40     12    70%   15-18, 33-40, 55
-----------------------------------------------------
TOTAL                      67     15    78%
```

- **Stmts**: 전체 실행 가능한 문장 수
- **Miss**: 테스트 중 실행되지 않은 문장 수
- **Cover**: 실행된 문장의 백분율
- **Missing**: 커버되지 않은 라인 번호

---

## 라인 커버리지 vs 브랜치 커버리지

### 라인 커버리지

라인 커버리지는 어떤 *라인*이 실행되었는지를 셉니다. 테스트되지 않은 조건부 브랜치를 놓칠 수 있습니다.

```python
def categorize_age(age: int) -> str:
    if age < 0:
        return "invalid"
    elif age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"


# This test covers only 2 of 4 branches
def test_adult():
    assert categorize_age(30) == "adult"

def test_minor():
    assert categorize_age(10) == "minor"

# Line coverage might show 70%+, but "invalid" and "senior" paths are untested
```

### 브랜치 커버리지

브랜치 커버리지는 각 조건의 두 결과(True와 False)를 모두 추적합니다. 활성화 방법:

```bash
pytest --cov=mypackage --cov-branch
```

또는 설정에서:

```ini
# .coveragerc or pyproject.toml
[tool.coverage.run]
branch = true
```

브랜치 커버리지는 위 테스트가 `age < 0`과 `age >= 65` 브랜치를 놓치고 있음을 드러냅니다.

```
Name                 Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------
mypackage/age.py        8      2      6      2    67%   3->4, 9->10
```

- **Branch**: 전체 브랜치 포인트
- **BrPart**: 부분적으로 커버된 브랜치 (일부 결과만 테스트됨)

---

## coverage.py 설정

### pyproject.toml 설정

```toml
[tool.coverage.run]
source = ["src/mypackage"]
branch = true
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__main__.py",
]

[tool.coverage.report]
# Fail if coverage drops below this threshold
fail_under = 85

# Lines matching these patterns are excluded from coverage
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.",
    "raise NotImplementedError",
    "pass",
    "\\.\\.\\.",
]

# Show missing lines in terminal report
show_missing = true

# Precision of coverage percentage
precision = 1

[tool.coverage.html]
directory = "htmlcov"
```

### .coveragerc (대안)

```ini
[run]
source = src/mypackage
branch = True
omit =
    */tests/*
    */migrations/*

[report]
fail_under = 85
exclude_lines =
    pragma: no cover
    def __repr__
    if __name__ == .__main__.
show_missing = True

[html]
directory = htmlcov
```

### 커버리지에서 코드 제외

테스트 커버리지에 포함하지 않아야 할 코드에 `# pragma: no cover` 주석을 사용합니다:

```python
def platform_specific_init():  # pragma: no cover
    """Only runs on Linux; tested in CI, not locally."""
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))


if __name__ == "__main__":  # pragma: no cover
    main()
```

제외 항목은 절제하여 사용하세요. 모든 제외 항목은 테스트하지 않기로 선택한 라인입니다.

---

## 커버리지 임계값 설정

### CI에서의 최소 커버리지

`fail_under`를 설정하여 커버리지를 CI 게이트로 만듭니다:

```toml
[tool.coverage.report]
fail_under = 85
```

```bash
# CI pipeline
pytest --cov=mypackage --cov-fail-under=85
```

### 임계값 선택

| 임계값    | 적합한 경우 |
|-----------|------------|
| 60-70%    | 초기 단계 프로젝트, 빠른 프로토타이핑 |
| 75-85%    | 대부분의 프로덕션 애플리케이션 |
| 85-95%    | 라이브러리, 금융/의료 소프트웨어 |
| 95%+      | 안전 필수 시스템, 핵심 알고리즘 |

**100%를 쫓지 마세요**. 마지막 5%를 커버하는 것은 종종 사소한 코드(문자열 표현, 방어적 오류 처리)를 테스트하는 것으로 수확 체감 효과가 있습니다. 목표는 의미 있는 커버리지이지, 허세를 위한 지표가 아닙니다.

### 래칫 방식 커버리지 (Ratcheting Coverage)

실용적인 접근법: 커버리지가 절대 *감소*하지 않도록 합니다. 현재 커버리지를 기록하고, 새 변경 사항이 이를 낮추면 CI를 실패시킵니다:

```bash
# Record baseline
pytest --cov=mypackage --cov-report=json
# Store .coverage or coverage.json as baseline

# In CI: compare against baseline
pytest --cov=mypackage --cov-fail-under=$(cat .coverage-baseline)
```

---

## 커버리지 함정

### 함정 1: 높은 커버리지, 낮은 신뢰도

```python
def calculate_discount(price, membership):
    if membership == "gold":
        return price * 0.20
    elif membership == "silver":
        return price * 0.10
    else:
        return 0


# 100% line coverage, but no assertions on values!
def test_all_paths():
    calculate_discount(100, "gold")
    calculate_discount(100, "silver")
    calculate_discount(100, "bronze")
    # No assert statements — tests always pass
```

### 함정 2: 동작이 아닌 구현 테스팅

관찰 가능한 동작이 아닌 내부 상태를 테스트하여 얻은 높은 커버리지. 동작이 보존되더라도 리팩토링 시 테스트가 깨집니다.

### 함정 3: 목표로서의 커버리지

굿하트의 법칙: "측정 지표가 목표가 되면, 좋은 측정 지표가 되기를 멈춘다." 엔지니어가 커버리지 수치에 대해 보상받으면, 품질을 개선하지 않고 지표를 올리기 위한 얕은 테스트를 작성합니다.

### 함정 4: 테스트되지 않은 엣지 케이스 무시

커버리지는 어떤 라인이 실행되었는지 보여주지, 어떤 *입력*이 테스트되었는지는 보여주지 않습니다. 함수가 하나의 테스트 케이스로 커버되더라도 중요한 엣지 케이스(null, 빈 값, 경계값, 오버플로우)가 테스트되지 않은 채 남을 수 있습니다.

---

## 뮤테이션 테스팅 (Mutation Testing)

뮤테이션 테스팅은 "버그를 도입하면 테스트가 이를 잡을 수 있는가?"라는 질문에 답합니다. 소스 코드를 체계적으로 수정(*뮤턴트* 생성)하여 테스트가 실패하는지 확인합니다. 뮤턴트가 살아남으면(테스트가 여전히 통과), 테스트에 갭이 있는 것입니다.

### 동작 방식

1. 도구가 소스 코드의 한 부분을 변경하여 *뮤턴트*를 생성 (예: `>`를 `>=`로, `+`를 `-`로, `True`를 `False`로)
2. 뮤턴트에 대해 전체 테스트 스위트를 실행
3. 테스트가 실패하면, 뮤턴트가 **killed** (좋음)
4. 테스트가 통과하면, 뮤턴트가 **survived** (테스트가 버그를 놓침)

### mutmut 사용

```bash
pip install mutmut

# Run mutation testing
mutmut run --paths-to-mutate=src/mypackage/

# View results
mutmut results

# Show a specific surviving mutant
mutmut show 42

# Generate HTML report
mutmut html
```

### 예제

```python
# discount.py
def apply_discount(price: float, percentage: float) -> float:
    if percentage < 0 or percentage > 100:
        raise ValueError("Invalid percentage")
    return price * (1 - percentage / 100)
```

```python
# test_discount.py
def test_apply_discount():
    assert apply_discount(100, 10) == 90.0
```

mutmut이 생성할 수 있는 뮤턴트:

| 뮤턴트 | 변경 | 생존 여부 |
|--------|------|-----------|
| 1 | `percentage < 0` -> `percentage <= 0` | 생존 (0%에 대한 테스트 없음) |
| 2 | `percentage > 100` -> `percentage >= 100` | 생존 (100%에 대한 테스트 없음) |
| 3 | `1 - percentage / 100` -> `1 + percentage / 100` | 사멸 (잡혔음!) |
| 4 | `percentage / 100` -> `percentage / 101` | 생존 (테스트 케이스가 하나뿐) |

살아남은 뮤턴트는 실제 갭을 드러냅니다. 더 많은 테스트를 추가하면 이를 사멸시킵니다:

```python
def test_zero_discount():
    assert apply_discount(100, 0) == 100.0

def test_full_discount():
    assert apply_discount(100, 100) == 0.0

def test_half_discount():
    assert apply_discount(200, 50) == 100.0

def test_negative_percentage_raises():
    with pytest.raises(ValueError):
        apply_discount(100, -1)

def test_over_100_percentage_raises():
    with pytest.raises(ValueError):
        apply_discount(100, 101)
```

### 뮤테이션 테스팅의 한계

- **느림**: 모든 뮤턴트에 대해 전체 테스트 스위트를 실행 (대규모 코드베이스에서는 수 시간 소요)
- **동등 뮤턴트**: 일부 뮤테이션은 동작을 변경하지 않음 (예: 독립적인 문장의 순서 변경)
- **핵심 코드에 적합**: 전체 코드베이스가 아닌 핵심 비즈니스 로직에 선택적으로 적용

---

## 커버리지를 넘어선 코드 품질

### 순환 복잡도 (Cyclomatic Complexity)

높은 복잡도는 테스트해야 할 경로가 더 많다는 의미입니다. `radon` 같은 도구로 측정합니다:

```bash
pip install radon
radon cc src/mypackage/ -s -a
```

```
src/mypackage/processor.py
    F 12:0 process_order - C (14)   # Complexity 14 = many branches
    F 45:0 validate_input - A (3)   # Complexity 3 = simple
```

함수당 복잡도 10 미만을 목표로 하세요. 높은 복잡도의 함수는 더 많은 테스트가 필요할 뿐만 아니라 리팩토링되어야 합니다.

### 정적 분석

타입 체커와 린터는 테스트가 놓칠 수 있는 버그를 잡습니다:

```bash
# Type checking
pip install mypy
mypy src/mypackage/

# Linting
pip install ruff
ruff check src/mypackage/
```

### 속성 기반 테스팅 (Property-Based Testing)

특정 테스트 케이스를 작성하는 대신, 항상 성립해야 하는 속성을 기술합니다. Hypothesis가 수백 개의 무작위 입력을 생성합니다:

```bash
pip install hypothesis
```

```python
from hypothesis import given
from hypothesis import strategies as st


def reverse_string(s: str) -> str:
    return s[::-1]


@given(st.text())
def test_reverse_is_involution(s):
    """Reversing twice returns the original string."""
    assert reverse_string(reverse_string(s)) == s

@given(st.text())
def test_reverse_preserves_length(s):
    assert len(reverse_string(s)) == len(s)

@given(st.lists(st.integers()))
def test_sort_is_idempotent(lst):
    """Sorting a sorted list returns the same list."""
    assert sorted(sorted(lst)) == sorted(lst)
```

---

## 균형 잡힌 테스트 품질 전략

1. **동작 중심 단위 테스트로 시작** — 해피 패스와 핵심 엣지 케이스를 커버
2. **브랜치 커버리지를 활성화**하고 합리적인 하한으로 80-85%를 목표
3. **핵심 비즈니스 로직에 뮤테이션 테스팅 사용** — 어서션 갭 발견
4. **순수 함수와 데이터 변환에 속성 기반 테스트 추가**
5. **CI에서 테스트와 함께 정적 분석** (mypy, ruff) 실행
6. **커버리지 추세 모니터링** — 감소를 방지하고, 최대화에 집착하지 않기

```toml
# A balanced pyproject.toml test configuration
[tool.pytest.ini_options]
addopts = "--cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80"
testpaths = ["tests"]

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "raise NotImplementedError",
]
```

---

## 연습 문제

1. **커버리지 갭 분석**: 작은 Python 모듈(50-100줄)을 가져와 `pytest --cov --cov-branch --cov-report=term-missing`을 실행하세요. 커버되지 않은 브랜치를 식별하세요. 95% 브랜치 커버리지에 도달하도록 테스트를 작성하고, 나머지 5%가 무엇을 커버하는지 설명하세요.

2. **뮤테이션 테스팅**: `mutmut`을 설치하고 테스트 스위트가 있는 모듈에 대해 실행하세요. 최소 3개의 살아남은 뮤턴트를 식별하세요. 각각에 대해 뮤턴트가 왜 살아남았는지 설명하고 이를 사멸시키는 테스트를 작성하세요.

3. **커버리지 vs 품질 논쟁**: 100% 라인 커버리지이지만 최소 3개의 살아남은 뮤턴트가 있는 모듈을 작성하세요. 그런 다음 80% 라인 커버리지이지만 살아남은 뮤턴트가 0개인 모듈을 작성하세요. 어떤 테스트 스위트가 더 나은지 그 이유를 설명하세요.

---

**License**: CC BY-NC 4.0
