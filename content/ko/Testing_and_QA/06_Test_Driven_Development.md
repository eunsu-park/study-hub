# 테스트 주도 개발 (Test-Driven Development)

**이전**: [테스트 커버리지와 품질](./05_Test_Coverage_and_Quality.md) | **다음**: [통합 테스팅](./07_Integration_Testing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Red-Green-Refactor 사이클을 적용하여 코드를 점진적으로 개발할 수 있다
2. 베이비 스텝과 삼각측량법을 사용하여 구현을 점진적으로 구축할 수 있다
3. 변환 우선순위 전제(Transformation Priority Premise)를 적용하여 리팩토링 결정을 안내할 수 있다
4. 다양한 상황에서 TDD의 장점과 한계를 평가할 수 있다
5. 실제 모듈을 구축하며 완전한 TDD 워크스루를 수행할 수 있다

---

## TDD란 무엇인가?

테스트 주도 개발(Test-Driven Development)은 통과시킬 프로덕션 코드를 작성하기 *전에* 실패하는 테스트를 먼저 작성하는 소프트웨어 개발 실천입니다. Kent Beck이 익스트림 프로그래밍의 일부로 만들었으며, 전통적인 "코드 작성 후 테스트" 워크플로우를 뒤집습니다.

기본 사이클은 세 단계입니다:

```
    ┌──────────┐
    │   RED    │   Write a test that fails
    └────┬─────┘
         │
    ┌────▼─────┐
    │  GREEN   │   Write the simplest code to pass the test
    └────┬─────┘
         │
    ┌────▼─────┐
    │ REFACTOR │   Improve the code without changing behavior
    └────┬─────┘
         │
         └──────── Repeat
```

### TDD의 세 가지 법칙 (Robert C. Martin)

1. 실패하는 단위 테스트를 작성하기 전에는 프로덕션 코드를 작성할 수 없다
2. 실패하기에 충분한 이상의 단위 테스트를 작성할 수 없다 (컴파일 실패 포함)
3. 현재 실패하는 테스트를 통과시키기에 충분한 이상의 프로덕션 코드를 작성할 수 없다

---

## 실전에서의 Red-Green-Refactor

### RED: 실패하는 테스트 작성

하나의 동작을 기술하는 가장 작은 테스트를 작성합니다. 실행하여 실패하는지 확인합니다. 실패 메시지는 명확하고 구체적이어야 합니다.

```python
# test_stack.py
from stack import Stack

def test_new_stack_is_empty():
    s = Stack()
    assert s.is_empty() is True
```

```bash
$ pytest test_stack.py
E   ModuleNotFoundError: No module named 'stack'
```

`Stack`이 아직 존재하지 않기 때문에 테스트가 실패합니다. 이것이 RED 단계입니다.

### GREEN: 통과시키기

테스트를 통과시키기 위한 최소한의 코드를 작성합니다. 미래의 요구사항을 예측하지 마세요. 최적화하지 마세요. 그냥 초록색으로 만드세요.

```python
# stack.py
class Stack:
    def is_empty(self):
        return True
```

```bash
$ pytest test_stack.py
PASSED
```

네, `return True`는 속임수입니다. 이것은 의도적입니다. 현재 테스트는 `is_empty()`가 `True`를 반환하는 것만 요구합니다. 더 많은 테스트가 실제 구현을 강제할 것입니다.

### REFACTOR: 정리

테스트 코드와 프로덕션 코드를 모두 살펴봅니다. 중복이 있나요? 잘못된 이름이 있나요? 불필요한 복잡성이 있나요? 모든 테스트를 초록색으로 유지하면서 정리합니다.

이 경우, 코드가 충분히 단순하므로 아직 리팩토링이 필요하지 않습니다.

---

## 베이비 스텝과 삼각측량법

### 베이비 스텝 (Baby Steps)

각 사이클은 아주 작아야 합니다 — 최대 몇 분 정도. RED 단계에서 복잡한 테스트를 작성하며 20분을 보낸다면, 너무 큰 발걸음을 내딛고 있는 것입니다.

큰 기능을 가능한 한 가장 작은 단위로 분해하세요:

```
Bad:  "Test that the calculator handles all arithmetic operations"
Good: "Test that add(2, 3) returns 5"
      "Test that add(-1, 1) returns 0"
      "Test that subtract(5, 3) returns 2"
```

### 삼각측량법 (Triangulation)

일반화하는 방법이 확실하지 않을 때, 패턴이 일반적인 구현을 강제할 때까지 더 많은 구체적인 테스트 케이스를 추가합니다.

```python
# Test 1: forces add() to exist and return something
def test_add_two_and_three():
    assert add(2, 3) == 5

# stack.py — the "cheat" implementation
def add(a, b):
    return 5  # Passes, but obviously fake
```

```python
# Test 2: triangulates — forces a real implementation
def test_add_one_and_one():
    assert add(1, 1) == 2

# Now the fake implementation fails. We must generalize:
def add(a, b):
    return a + b
```

두 개의 테스트 케이스가 실제 구현을 "삼각측량"했습니다. 하나의 테스트 케이스는 속임수를 허용했지만, 두 개는 진실을 강제했습니다.

---

## 변환 우선순위 전제 (Transformation Priority Premise)

Kent Beck과 Robert Martin은 TDD 구현이 단순한 것에서 복잡한 것으로 예측 가능한 변환을 거쳐 발전한다고 관찰했습니다. 더 단순한 변환을 선호하세요:

| 우선순위 | 변환 | 예시 |
|----------|------|------|
| 1 | `{}` -> nil | 코드 없음 -> return None |
| 2 | nil -> 상수 | return None -> return 42 |
| 3 | 상수 -> 변수 | return 42 -> return x |
| 4 | 무조건 -> 조건 | 문장 -> if/else |
| 5 | 스칼라 -> 컬렉션 | 단일 값 -> 리스트 |
| 6 | 문장 -> 재귀 | 루프 -> 재귀 호출 |
| 7 | 값 -> 변경된 값 | 불변 -> 가변 상태 |

테스트를 통과시키는 방법을 선택할 때, 이 목록에서 *위쪽*(더 단순한)에 있는 변환을 선호하세요. 이는 더 깨끗하고 점진적인 설계로 이어집니다.

---

## TDD 워크스루: 로마 숫자 변환기 구축

엄격한 TDD를 사용하여 정수를 로마 숫자로 변환하는 함수를 구축해 봅시다.

### 사이클 1: 가장 단순한 경우

**RED:**

```python
# test_roman.py
from roman import to_roman

def test_1_is_I():
    assert to_roman(1) == "I"
```

**GREEN:**

```python
# roman.py
def to_roman(number: int) -> str:
    return "I"
```

### 사이클 2: 삼각측량

**RED:**

```python
def test_2_is_II():
    assert to_roman(2) == "II"
```

**GREEN:**

```python
def to_roman(number: int) -> str:
    return "I" * number
```

이 구현은 두 테스트를 모두 통과합니다. `"I" * number` 패턴이 두 테스트 케이스에서 일반화됩니다.

### 사이클 3: 새로운 기호 도입

**RED:**

```python
def test_5_is_V():
    assert to_roman(5) == "V"
```

`"I" * 5 = "IIIII"`이므로 올바르지 않습니다. 조건문이 필요합니다.

**GREEN:**

```python
def to_roman(number: int) -> str:
    if number == 5:
        return "V"
    return "I" * number
```

### 사이클 4: 조합

**RED:**

```python
def test_4_is_IV():
    assert to_roman(4) == "IV"

def test_6_is_VI():
    assert to_roman(6) == "VI"
```

일반적인 알고리즘으로 리팩토링할 시점입니다.

**GREEN + REFACTOR:**

```python
def to_roman(number: int) -> str:
    values = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = ""
    for value, numeral in values:
        while number >= value:
            result += numeral
            number -= value
    return result
```

### 사이클 5: 종합 테스트

이제 신뢰를 구축하기 위해 더 많은 테스트 케이스를 추가합니다:

```python
import pytest

@pytest.mark.parametrize("number, expected", [
    (1, "I"),
    (3, "III"),
    (4, "IV"),
    (5, "V"),
    (9, "IX"),
    (10, "X"),
    (14, "XIV"),
    (40, "XL"),
    (42, "XLII"),
    (90, "XC"),
    (99, "XCIX"),
    (100, "C"),
    (400, "CD"),
    (500, "D"),
    (900, "CM"),
    (1000, "M"),
    (1994, "MCMXCIV"),
    (3999, "MMMCMXCIX"),
])
def test_to_roman(number, expected):
    assert to_roman(number) == expected
```

모두 통과합니다. TDD가 어떻게 가짜 `return "I"`에서 완전한 알고리즘까지 작고 검증된 단계를 통해 우리를 안내했는지 주목하세요.

---

## TDD 워크스루: 스택 구축

데이터 구조를 사용한 좀 더 실질적인 예제로 TDD를 보여줍니다.

```python
# test_stack.py — tests written in TDD order
import pytest
from stack import Stack


def test_new_stack_is_empty():
    s = Stack()
    assert s.is_empty() is True


def test_stack_after_push_is_not_empty():
    s = Stack()
    s.push(42)
    assert s.is_empty() is False


def test_pop_returns_pushed_value():
    s = Stack()
    s.push(42)
    assert s.pop() == 42


def test_pop_removes_element():
    s = Stack()
    s.push(42)
    s.pop()
    assert s.is_empty() is True


def test_lifo_order():
    s = Stack()
    s.push("first")
    s.push("second")
    assert s.pop() == "second"
    assert s.pop() == "first"


def test_peek_returns_top_without_removing():
    s = Stack()
    s.push(99)
    assert s.peek() == 99
    assert s.is_empty() is False


def test_pop_empty_stack_raises():
    s = Stack()
    with pytest.raises(IndexError, match="empty"):
        s.pop()


def test_peek_empty_stack_raises():
    s = Stack()
    with pytest.raises(IndexError, match="empty"):
        s.peek()


def test_size():
    s = Stack()
    assert s.size() == 0
    s.push("a")
    s.push("b")
    assert s.size() == 2
    s.pop()
    assert s.size() == 1
```

```python
# stack.py — implementation driven by the tests above
class Stack:
    def __init__(self):
        self._items: list = []

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack")
        return self._items[-1]

    def size(self) -> int:
        return len(self._items)
```

---

## TDD의 장점

### 1. 설계 피드백

TDD는 구현 전에 *인터페이스*에 대해 생각하도록 강제합니다. 함수가 테스트하기 어렵다면, 보통 사용하기도 어렵습니다. TDD는 자연스럽게 다음을 향해 밀어줍니다:

- 명확한 책임을 가진 더 작은 함수
- 숨겨진 의존성 대신 의존성 주입
- 가능한 곳에서의 순수 함수

### 2. 내장된 회귀 테스트 스위트

모든 기능은 태어날 때부터 테스트를 갖습니다. 나중에 리팩토링하거나 기능을 추가할 때, 기존 테스트가 즉시 회귀를 잡습니다.

### 3. 리팩토링에 대한 자신감

테스트 없이는 리팩토링이 두렵습니다. TDD를 사용하면 안전망이 있습니다. 테스트가 실수를 잡아줄 것을 알기에 코드를 적극적으로 단순화할 수 있습니다.

### 4. 예제를 통한 문서화

TDD 스타일로 작성된 테스트는 코드가 무엇을 하는지 구체적이고 실행 가능한 예제로 기술합니다. 절대 오래되지 않는 살아있는 문서의 역할을 합니다.

---

## 비판과 한계

### 1. 탐색적 작업에 대한 오버헤드

무엇을 만들지 모를 때(프로토타이핑, 연구), 테스트를 먼저 작성하는 것은 시기상조입니다. 테스트와 코드를 모두 버릴 수 있습니다. TDD는 요구사항이 합리적으로 명확할 때 가장 잘 동작합니다.

### 2. 과도한 명세화

잘못 실천된 TDD는 구현에 갇히게 할 수 있습니다. 테스트가 내부 구조에 너무 결합되면, 테스트를 다시 작성하지 않고는 리팩토링이 불가능해집니다.

### 3. 거짓 자신감

TDD가 정확성을 보장하지는 않습니다. 100% 커버리지를 가지고도 여전히 엣지 케이스를 놓칠 수 있습니다. TDD 테스트는 개발자의 이해를 반영하며, 이는 불완전할 수 있습니다.

### 4. 학습 곡선

TDD는 훈련과 연습이 필요합니다. 초보자는 종종 너무 크거나 구현 세부사항에 너무 결합된 테스트를 작성합니다. 좋은 테스트 세분화에 대한 감각을 기르는 데 시간이 걸립니다.

### 언제 TDD를 사용할 것인가

| 상황 | TDD? |
|------|------|
| 잘 정의된 비즈니스 로직 | 예 |
| 알고리즘 구현 | 예 |
| 라이브러리/API 설계 | 예 |
| UI 레이아웃과 스타일링 | 보통 아니오 |
| 탐색적 프로토타이핑 | 아니오 |
| 일회성 스크립트 | 아니오 |
| 프로덕션 이슈 디버깅 | 예 (먼저 버그에 대한 실패하는 테스트 작성) |

---

## TDD와 전체 프로세스

TDD는 더 넓은 품질 도구 모음의 하나의 실천입니다:

```
Developer writes failing test (TDD)
  → Implements code to pass
  → Refactors
  → Pushes to CI
    → CI runs full test suite
    → CI runs linter and type checker
    → CI runs coverage check
    → Code review
    → Merge
```

TDD는 통합 테스트, 코드 리뷰, 모니터링의 대체물이 아닙니다. 잘 테스트되고 잘 설계된 코드 단위를 생산하는 개발 규율입니다.

---

## 연습 문제

1. **TDD로 FizzBuzz 구현**: 엄격한 TDD를 사용하여(하나의 테스트 작성, 통과시키기, 리팩토링) FizzBuzz를 구현하세요. 각 Red-Green-Refactor 사이클을 문서화하세요. 각 단계에서의 테스트와 구현을 보여주세요.

2. **TDD로 StringCalculator 구현**: `string_calculator(expression: str) -> int`를 구축하세요. 처리해야 할 내용:
   - 빈 문자열은 0을 반환
   - 단일 숫자는 해당 숫자를 반환
   - 쉼표로 구분된 두 숫자는 그 합을 반환
   - 줄바꿈을 구분자로 사용 (`"1\n2,3"`은 6을 반환)
   - 음수는 ValueError를 발생
   엄격하게 TDD를 적용하세요 — 한 번에 하나의 테스트 케이스.

3. **TDD 비평**: TDD 없이 작성한 기존 코드를 가져오세요. 사후에 테스트를 추가해 보세요. 테스팅을 어렵게 만드는 최소 2가지 설계 결정을 식별하세요. TDD가 어떻게 다른 설계로 이끌었을 수 있는지 설명하세요.

---

**License**: CC BY-NC 4.0
