# 레슨 10: 속성 기반 테스트 (Property-Based Testing)

**이전**: [엔드투엔드 테스트](./09_End_to_End_Testing.md) | **다음**: [성능 테스트](./11_Performance_Testing.md)

---

전통적인 예제 기반 테스트는 특정 입력이 특정 출력을 생성하는지 검증합니다. 그러나 선택한 예제가 실제로 중요한 엣지 케이스를 다루고 있는지 어떻게 알 수 있을까요? 속성 기반 테스트(property-based testing)는 접근 방식을 뒤집습니다: 개별 예제를 지정하는 대신, 항상 성립해야 하는 **속성**을 기술하고, 테스트 프레임워크가 수백 또는 수천 개의 무작위 입력을 생성하여 이를 깨뜨리려 시도합니다. 실패를 발견하면, 가능한 가장 작은 반례(counterexample)로 자동 축소합니다.

**난이도**: ⭐⭐⭐

**선수 조건**:
- pytest 테스트 작성에 익숙할 것 (레슨 02-03)
- 테스트 설계 원칙 이해 (레슨 05)
- Python 타입 힌트에 대한 기본 지식

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 속성 기반 테스트가 예제 기반 테스트로 놓치는 버그를 잡는 이유를 설명할 수 있다
2. Hypothesis 라이브러리를 사용하여 `@given`과 전략(strategy)으로 속성 기반 테스트를 작성할 수 있다
3. 함수, 데이터 구조, API에 대한 효과적인 속성을 설계할 수 있다
4. 복잡한 상태 전환이 있는 시스템을 검증하기 위해 상태 기반 테스트를 적용할 수 있다
5. Hypothesis 설정을 구성하고 CI 파이프라인에 통합할 수 있다

---

## 1. 예제 기반 테스트의 한계

리스트를 정렬하는 함수를 생각해 봅시다:

```python
def test_sort_basic():
    assert sort_list([3, 1, 2]) == [1, 2, 3]
    assert sort_list([]) == []
    assert sort_list([1]) == [1]
```

이 코드는 세 가지 케이스를 다룹니다. 하지만 음수는? 중복값은? 매우 큰 리스트는? `None` 값이 있는 리스트는? 상상력에 의해 제한되며 — 상상력은 작동하는 케이스 쪽으로 편향됩니다.

속성 기반 테스트는 다른 질문을 던집니다: **입력에 관계없이 `sort_list`의 출력에 대해 항상 참이어야 하는 속성은 무엇인가?**

- 출력의 길이가 입력과 같다
- 출력이 비감소(non-decreasing) 순서이다
- 출력이 입력과 정확히 같은 원소를 포함한다

이 속성들은 *모든* 유효한 입력에 대해 성립하며 — 속성 기반 테스트 프레임워크가 이들이 성립하지 않는 입력을 찾으려 시도합니다.

---

## 2. Hypothesis 시작하기

[Hypothesis](https://hypothesis.readthedocs.io/)는 Python의 표준 속성 기반 테스트 라이브러리입니다. pytest와 함께 설치합니다:

```bash
pip install hypothesis
```

### 2.1 첫 번째 속성 테스트

```python
from hypothesis import given
from hypothesis import strategies as st


def sort_list(xs):
    return sorted(xs)


@given(st.lists(st.integers()))
def test_sort_preserves_length(xs):
    result = sort_list(xs)
    assert len(result) == len(xs)


@given(st.lists(st.integers()))
def test_sort_produces_ordered_output(xs):
    result = sort_list(xs)
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


@given(st.lists(st.integers()))
def test_sort_preserves_elements(xs):
    result = sort_list(xs)
    assert sorted(result) == sorted(xs)  # Same multiset
```

`pytest`를 실행하면, Hypothesis가 기본적으로 100개의 무작위 정수 리스트를 생성하여 각 속성을 확인합니다. 하나라도 실패하면, 최소 실패 예제로 입력을 축소합니다.

### 2.2 Hypothesis의 내부 동작 방식

Hypothesis는 단순히 무작위 데이터를 생성하지 않습니다. 그 과정은 다음과 같습니다:

1. **생성**: 지정된 전략을 사용하여 무작위 예제를 생성합니다
2. **테스트**: 각 예제로 테스트 함수를 실행합니다
3. **축소**: 실패가 발견되면, 가장 작은 반례를 찾기 위해 체계적으로 입력을 줄입니다
4. **데이터베이스**: 실패한 예제를 `.hypothesis/`에 저장하여 이후 실행에서 다시 재생합니다

축소 단계가 핵심입니다. 47개의 무작위 정수로 이루어진 리스트에서 함수가 실패하면, 원시 실패는 디버깅하기 어렵습니다. Hypothesis는 이를 `[0, 1]`이나 `[-1]`과 같이 — 여전히 버그를 유발하는 가장 단순한 입력으로 축소합니다.

---

## 3. 전략 심화

전략(strategy)은 Hypothesis가 테스트 데이터의 형태를 기술하는 방법입니다. `hypothesis.strategies` 모듈(관례적으로 `st`로 임포트)은 조합 가능한 구성 요소를 제공합니다.

### 3.1 기본 전략

```python
from hypothesis import strategies as st

# Integers with optional bounds
st.integers()                        # Any integer
st.integers(min_value=0, max_value=100)  # Bounded

# Floating point
st.floats()                          # Includes NaN, inf, -inf
st.floats(allow_nan=False, allow_infinity=False)

# Text
st.text()                            # Any Unicode string
st.text(min_size=1, max_size=50)     # Bounded length
st.text(alphabet="abcdef")           # Restricted alphabet

# Booleans
st.booleans()                        # True or False

# Binary
st.binary(min_size=0, max_size=100)  # bytes objects

# None
st.none()                            # Always None
```

### 3.2 컬렉션 전략

```python
# Lists
st.lists(st.integers())                          # List[int]
st.lists(st.integers(), min_size=1, max_size=10) # Non-empty, bounded

# Sets (no duplicates)
st.sets(st.integers(), min_size=1)

# Dictionaries
st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.integers()
)

# Tuples (fixed structure)
st.tuples(st.integers(), st.text(), st.booleans())  # (int, str, bool)

# Frozensets
st.frozensets(st.integers())
```

### 3.3 전략 조합하기

전략의 진정한 힘은 조합에서 나옵니다:

```python
# Union: one of several types
st.one_of(st.integers(), st.text(), st.none())

# Mapping: transform generated values
st.integers(min_value=1, max_value=12).map(lambda m: f"2024-{m:02d}-01")

# Filtering: reject values (use sparingly — can slow generation)
st.integers().filter(lambda x: x % 2 == 0)  # Even integers

# Flatmap: dependent generation
def list_and_index(draw):
    """Generate a non-empty list and a valid index into it."""
    xs = draw(st.lists(st.integers(), min_size=1))
    i = draw(st.integers(min_value=0, max_value=len(xs) - 1))
    return (xs, i)

# Using @st.composite for complex strategies
@st.composite
def valid_email(draw):
    local = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=1, max_size=20
    ))
    domain = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=1, max_size=10
    ))
    tld = draw(st.sampled_from(["com", "org", "net", "io"]))
    return f"{local}@{domain}.{tld}"
```

### 3.4 `@st.composite` 데코레이터

`@st.composite`는 커스텀 전략을 구축하는 가장 유연한 방법입니다. 데코레이팅된 함수는 다른 전략에서 값을 가져오는 `draw` 호출 가능 객체를 받습니다:

```python
@st.composite
def sorted_pair(draw):
    """Generate two integers where a <= b."""
    a = draw(st.integers())
    b = draw(st.integers(min_value=a))
    return (a, b)


@given(sorted_pair())
def test_range_is_valid(pair):
    a, b = pair
    assert a <= b
```

---

## 4. `@example`로 명시적 예제 제공하기

때로는 생성된 예제에 더하여, 특정 엣지 케이스가 항상 테스트되도록 보장하고 싶을 때가 있습니다:

```python
from hypothesis import given, example
from hypothesis import strategies as st


@given(st.integers(), st.integers())
@example(0, 0)          # Always test zero/zero
@example(1, -1)         # Always test opposite signs
@example(2**31, 2**31)  # Always test large values
def test_addition_is_commutative(a, b):
    assert a + b == b + a
```

`@example`은 무작위 생성 단계 이전에 주어진 입력을 *매번* 실행합니다. 다음과 같은 경우에 유용합니다:
- 이전에 발견된 버그에 대한 회귀 테스트
- 알려진 엣지 케이스 (빈 문자열, 0, 경계 값)
- 무작위 생성으로 발견하는 데 오래 걸릴 수 있는 케이스

---

## 5. `settings`로 Hypothesis 설정하기

`settings` 데코레이터로 생성 동작을 제어합니다:

```python
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


@settings(
    max_examples=500,          # Default is 100; increase for thorough testing
    deadline=1000,             # Max ms per example (None to disable)
    suppress_health_check=[
        HealthCheck.too_slow,  # Allow slow tests
    ],
    database=None,             # Disable example database
)
@given(st.lists(st.integers()))
def test_with_custom_settings(xs):
    result = sorted(xs)
    assert len(result) == len(xs)
```

### 5.1 환경별 프로파일

```python
from hypothesis import settings, Phase

# CI profile: more thorough
settings.register_profile("ci", max_examples=1000)

# Development profile: fast feedback
settings.register_profile("dev", max_examples=50)

# Debug profile: minimal, with verbose output
settings.register_profile("debug", max_examples=10, verbosity=3)

# Load profile from environment variable
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

CI에서 프로파일을 설정합니다:

```bash
HYPOTHESIS_PROFILE=ci pytest tests/
```

---

## 6. 상태 기반 테스트

순수 함수에 대한 속성 테스트는 직관적입니다. 그러나 변경 가능한 상태를 가진 시스템 — 데이터베이스, 캐시, API — 은 어떨까요? Hypothesis는 규칙 기반 상태 머신을 사용한 **상태 기반 테스트(stateful testing)**를 제공합니다.

### 6.1 규칙 기반 상태 머신

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant
from hypothesis import strategies as st


class SetMachine(RuleBasedStateMachine):
    """Test that our CustomSet behaves like Python's built-in set."""

    def __init__(self):
        super().__init__()
        self.model = set()           # Reference (known-correct)
        self.actual = CustomSet()    # Implementation under test

    @rule(value=st.integers())
    def add_element(self, value):
        self.model.add(value)
        self.actual.add(value)

    @rule(value=st.integers())
    def remove_element(self, value):
        try:
            self.model.remove(value)
            self.actual.remove(value)
        except KeyError:
            pass  # Both should raise or neither

    @rule(value=st.integers())
    def check_contains(self, value):
        assert (value in self.model) == (value in self.actual)

    @invariant()
    def size_matches(self):
        assert len(self.model) == len(self.actual)


# Hypothesis will generate random sequences of add/remove/check operations
TestSetMachine = SetMachine.TestCase
```

### 6.2 상태 기반 테스트가 중요한 이유

상태 기반 테스트는 특히 다음과 같은 경우에 강력합니다:
- **데이터 구조**: 구현이 참조 구현과 일치하는지 검증
- **API**: 어떤 순서의 API 호출이든 시스템을 유효한 상태로 유지하는지 검증
- **프로토콜**: 상태 머신이 모든 유효한 전환을 처리하는지 검증
- **캐시**: 캐시된 결과가 항상 새로 계산한 결과와 일치하는지 검증

Hypothesis는 연산 시퀀스를 생성하고, 실패가 발견되면 연산과 인자 모두를 최소 실패 시퀀스로 축소합니다.

---

## 7. 속성 기반 테스트가 빛나는 순간

속성 기반 테스트는 예제 기반 테스트의 대체가 아닙니다. 특정 상황에서 뛰어난 성과를 발휘합니다:

### 7.1 왕복(Roundtrip) 속성

인코딩과 디코딩이 가능하다면, 왕복은 항등(identity)이어야 합니다:

```python
import json

@given(st.dictionaries(st.text(), st.integers()))
def test_json_roundtrip(data):
    assert json.loads(json.dumps(data)) == data
```

### 7.2 불변(Invariant) 속성

특정 특성을 보존해야 하는 연산:

```python
@given(st.lists(st.integers()))
def test_reverse_preserves_length(xs):
    assert len(list(reversed(xs))) == len(xs)

@given(st.lists(st.integers()))
def test_reverse_is_involution(xs):
    assert list(reversed(list(reversed(xs)))) == xs
```

### 7.3 오라클 테스트

구현을 알려진 정확한(하지만 느린) 참조와 비교합니다:

```python
@given(st.lists(st.integers()))
def test_my_sort_matches_builtin(xs):
    assert my_sort(xs) == sorted(xs)
```

### 7.4 대수적(Algebraic) 속성

코드가 만족해야 하는 수학적 속성:

```python
@given(st.integers(), st.integers(), st.integers())
def test_addition_is_associative(a, b, c):
    assert (a + b) + c == a + (b + c)
```

### 7.5 멱등성(Idempotence)

연산을 두 번 적용해도 한 번 적용한 것과 같은 결과를 내야 합니다:

```python
@given(st.text())
def test_normalize_is_idempotent(s):
    assert normalize(normalize(s)) == normalize(s)
```

---

## 8. 실용 예제: URL 파서 테스트

더 현실적인 예제를 구축해 봅시다 — URL 파서의 속성 테스트:

```python
from urllib.parse import urlparse, urlunparse
from hypothesis import given
from hypothesis import strategies as st


@st.composite
def urls(draw):
    scheme = draw(st.sampled_from(["http", "https"]))
    host = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
        min_size=1, max_size=20
    ))
    tld = draw(st.sampled_from(["com", "org", "net"]))
    path = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz/",
        min_size=0, max_size=30
    ))
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{host}.{tld}{path}"


@given(urls())
def test_parse_roundtrip(url):
    """Parsing and reconstructing a URL should produce the original."""
    parsed = urlparse(url)
    reconstructed = urlunparse(parsed)
    assert reconstructed == url


@given(urls())
def test_scheme_is_preserved(url):
    parsed = urlparse(url)
    assert parsed.scheme in ("http", "https")


@given(urls())
def test_netloc_is_non_empty(url):
    parsed = urlparse(url)
    assert len(parsed.netloc) > 0
```

---

## 9. 일반적인 함정과 모범 사례

### 9.1 과도하게 엄격한 필터 피하기

```python
# BAD: Hypothesis may give up if too many values are rejected
@given(st.integers().filter(lambda x: x % 1000 == 42))
def test_bad_filter(x):
    ...

# GOOD: Generate directly
@given(st.integers().map(lambda x: x * 1000 + 42))
def test_good_generation(x):
    ...
```

### 9.2 속성을 단순하고 독립적으로 유지하기

각 테스트는 하나의 속성을 검증해야 합니다. "정렬됨, 같은 길이, 같은 원소"를 모두 한 번에 확인하는 테스트는 세 개의 개별 테스트보다 디버깅하기 어렵습니다.

### 9.3 회귀 테스트에 `@example` 사용하기

Hypothesis가 버그를 발견하면, `@example(failing_input)`을 추가하여 영구적으로 고정합니다:

```python
@given(st.text())
@example("")           # Found this edge case in CI
@example("\x00\xff")   # Null and high bytes
def test_my_encoder(s):
    assert decode(encode(s)) == s
```

### 9.4 합리적인 데드라인 설정하기

함수가 본질적으로 느린 경우, 테스트가 타임아웃되지 않도록 데드라인을 조정합니다:

```python
@settings(deadline=5000)  # 5 seconds per example
@given(st.lists(st.integers(), max_size=10000))
def test_slow_function(xs):
    ...
```

### 9.5 전제 조건에 `assume()` 사용하기

전략으로 표현하기 어려운 전제 조건을 입력이 만족해야 하는 경우:

```python
from hypothesis import assume

@given(st.floats(), st.floats())
def test_division(a, b):
    assume(b != 0)
    assume(not (a == 0 and b == 0))
    result = a / b
    assert result * b == pytest.approx(a, rel=1e-6)
```

---

## 10. Hypothesis를 워크플로우에 통합하기

### 10.1 프로젝트 설정

`pyproject.toml`에 추가합니다:

```toml
[tool.hypothesis]
database_backend = "directory"

[tool.pytest.ini_options]
addopts = "--hypothesis-show-statistics"
```

### 10.2 CI 설정

```yaml
# .github/workflows/test.yml
- name: Run property tests
  env:
    HYPOTHESIS_PROFILE: ci
  run: pytest tests/ -x --hypothesis-show-statistics
```

### 10.3 속성 테스트 vs 예제 테스트 선택 기준

| 상황 | 권장 접근 방식 |
|---|---|
| 특정 규칙이 있는 비즈니스 로직 | 예제 기반 |
| 직렬화/역직렬화 | 속성 기반 (왕복) |
| 데이터 변환 | 속성 기반 (불변) |
| 알고리즘 정확성 | 속성 기반 (오라클) |
| UI 동작 | 예제 기반 |
| 엣지 케이스 회귀 | 속성 테스트의 `@example` |
| 수학적 연산 | 속성 기반 (대수적) |

---

## 연습 문제

1. **왕복 테스트**: `compress`/`decompress` 함수 쌍에 대해 임의의 바이트 문자열에 대한 왕복 속성을 검증하는 속성 테스트를 작성합니다.

2. **불변 속성 발견**: 순서를 유지하면서 중복을 제거하는 함수 `deduplicate(xs: list) -> list`가 주어졌을 때, 이 함수가 만족해야 하는 최소 세 가지 속성을 식별하고 테스트합니다.

3. **커스텀 전략**: 설정 가능한 깊이까지 유효한 JSON과 유사한 중첩 구조(dict가 list를 포함하고 list가 dict를 포함하는 등)를 생성하는 `@st.composite` 전략을 구축합니다. 이를 사용하여 속성 테스트를 작성합니다.

4. **상태 기반 테스트**: 간단한 키-값 저장소(dict 래퍼, `get`, `set`, `delete` 포함)를 Python 내장 `dict`와 비교하여 테스트하는 `RuleBasedStateMachine`을 구현합니다.

5. **버그 찾기**: 정렬 함수에 의도적으로 미묘한 버그를 도입하고(예: 병합 단계에서 off-by-one), Hypothesis가 반례를 찾아 축소하는 것을 시연합니다. 같은 버그를 잡기 위해 필요한 예제 기반 테스트의 수와 비교합니다.

---

**License**: CC BY-NC 4.0
