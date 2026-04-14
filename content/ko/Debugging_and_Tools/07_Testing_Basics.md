# 테스트 기초

**이전**: [로깅](./06_Logging.md) | **다음**: [린터와 포매터](./08_Linters_and_Formatters.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 테스트가 단순한 품질 검사가 아니라 디버깅 도구인 이유 설명하기
2. 가정을 검증하는 효과적인 `assert` 문 작성하기
3. `pytest` 규칙에 따라 테스트 함수 작성하기
4. 테스트 파일, 테스트 클래스, 픽스처로 테스트 조직하기
5. `unittest.TestCase`와 어설션 메서드 사용하기
6. 테스트 주도 디버깅 적용: 먼저 실패하는 테스트를 작성한 후 버그 수정
7. 명령줄에서 테스트를 실행하고 출력 해석하기
8. 테스트 커버리지와 버그 감지의 관계 이해하기

---

테스트는 가용한 가장 강력한 디버깅 도구 중 하나입니다. 버그를 만났을 때 첫 번째 단계는 버그를 보여주는 테스트 -- 실패하는 테스트를 작성하는 것입니다. 그런 다음 테스트가 통과할 때까지 코드를 수정합니다. 이 접근법인 **테스트 주도 디버깅**은 버그를 진정으로 이해하고, 수정을 확인하며, 다시는 돌아오지 않도록 보장합니다.

> **핵심 통찰:** 테스트가 없는 버그는 돌아오기를 기다리는 버그입니다. 수정하는 모든 버그는 코드베이스를 영구적으로 강화하는 테스트를 추가할 기회입니다.

---

## 1. `assert` 문

### 1.1 기본 Assert

가장 간단한 형태의 테스트:

```python
def add(a, b):
    return a + b

# assert로 테스트
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0
print("모든 테스트 통과!")
```

### 1.2 메시지가 있는 Assert

항상 설명적인 메시지를 포함하세요:

```python
assert add(2, 3) == 5, f"5를 기대했으나 {add(2, 3)}을 받음"
assert len(result) > 0, "결과가 비어있으면 안 됨"
assert isinstance(value, int), f"int를 기대했으나 {type(value).__name__}을 받음"
```

**주의**: `assert` 문은 Python이 `-O` (최적화) 플래그로 실행될 때 제거됩니다. 프로덕션 코드에서 입력 검증에 `assert`를 절대 사용하지 마세요 -- 대신 `if`/`raise`를 사용하세요.

---

## 2. pytest: 표준 테스트 프레임워크

### 2.1 첫 번째 테스트 작성

```python
# file: test_calculator.py
def add(a, b):
    return a + b

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -2) == -3

def test_add_zero():
    assert add(5, 0) == 5
    assert add(0, 5) == 5
```

### 2.2 테스트 실행

```bash
pip install pytest
pytest                    # 현재 디렉토리의 모든 테스트 실행
pytest -v                 # 상세 출력
pytest test_calculator.py::test_add_positive_numbers  # 특정 테스트
pytest -x                 # 첫 실패에서 중지
```

### 2.3 예외 테스트

```python
import pytest

def test_invalid_input():
    with pytest.raises(ValueError):
        int("not_a_number")

def test_exception_message():
    with pytest.raises(ValueError, match="invalid literal"):
        int("abc")
```

---

## 3. 테스트 조직

### 3.1 파일 구조

```
project/
├── calculator.py          # 소스 코드
├── validator.py
├── tests/
│   ├── test_calculator.py # calculator.py 테스트
│   └── test_validator.py  # validator.py 테스트
```

### 3.2 테스트 클래스

```python
class TestCalculator:
    def test_add(self):
        assert add(2, 3) == 5

    def test_subtract(self):
        assert subtract(5, 3) == 2

    def test_divide_by_zero(self):
        import pytest
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)
```

---

## 4. pytest 픽스처

픽스처는 테스트를 위한 설정과 해체를 제공합니다:

```python
import pytest

@pytest.fixture
def sample_data():
    """테스트 데이터 제공."""
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

def test_length(sample_data):
    assert len(sample_data) == 5
```

### 임시 파일을 사용한 픽스처

```python
@pytest.fixture
def temp_file(tmp_path):
    """테스트용 임시 파일 생성."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    yield file_path

def test_read_file(temp_file):
    content = temp_file.read_text()
    assert content == "hello world"
```

---

## 5. 테스트 주도 디버깅

### 5.1 과정

```
버그 보고: "average()가 단일 요소 리스트에서 잘못된 값을 반환"
          │
          ▼
1단계: 실패하는 테스트 작성
          │    def test_average_single_element():
          │        assert average([42]) == 42.0  # 이것이 실패!
          ▼
2단계: 테스트를 실행하여 실패 확인
          │    FAILED: assert 0.0 == 42.0
          ▼
3단계: 코드를 디버그하고 수정
          │    발견: len(data) 대신 len(data) + 1로 나눔
          ▼
4단계: 테스트를 실행하여 통과 확인
          │    PASSED
          ▼
5단계: 모든 테스트를 실행하여 다른 것이 안 깨졌는지 확인
          │    5 passed, 0 failed
          ▼
6단계: 수정과 테스트를 함께 커밋
```

---

## 6. 매개변수화된 테스트

하나의 함수로 많은 입력을 테스트:

```python
import pytest

@pytest.mark.parametrize("input_val, expected", [
    ([1, 2, 3], 6),
    ([0], 0),
    ([-1, 1], 0),
    ([10, 20, 30, 40], 100),
])
def test_sum(input_val, expected):
    assert sum(input_val) == expected
```

---

## 7. 무엇을 테스트할 것인가

### 테스트 체크리스트

```
각 함수에 대해 다음을 고려:
□ 정상 케이스 (해피 패스)
□ 엣지 케이스 (빈 입력, 단일 요소, 0, 음수)
□ 경계 값 (첫/마지막 유효 값)
□ 에러 케이스 (잘못된 입력, 예외가 발생해야 함)
□ 반환 타입 (올바른 타입인지?)
□ 부작용 (수정하면 안 되는 것을 수정했는지?)
```

---

## 8. 테스트 커버리지

```bash
pip install pytest-cov
pytest --cov=mymodule --cov-report=term-missing
```

```
Name            Stmts   Miss  Cover   Missing
----------------------------------------------
calculator.py      20      4    80%   15-18
validator.py       35     10    71%   22-31
----------------------------------------------
TOTAL              55     14    75%
```

- **80%+**가 대부분의 프로젝트에 좋은 목표
- **100%** 커버리지는 버그가 없다는 뜻이 아님 (모든 줄을 커버해도 엣지 케이스를 놓칠 수 있음)
- 임의의 커버리지 숫자가 아닌 **핵심 경로**와 **엣지 케이스** 테스트에 집중

---

## 9. 테스트 팁

### 9.1 테스트를 단순하게

각 테스트는 **하나**만 테스트해야 합니다:

```python
# 좋은 예: 개념당 하나의 어설션
def test_process_preserves_length():
    assert len(process([1, 2, 3])) == 3

def test_process_returns_positive():
    assert all(x > 0 for x in process([1, 2, 3]))
```

### 9.2 테스트 이름은 설명적으로

```python
# 나쁜 예
def test_1():
    ...

# 좋은 예
def test_average_returns_zero_for_equal_positive_and_negative():
    ...
```

### 9.3 테스트는 독립적으로

테스트는 서로 의존하거나 실행 순서에 의존하면 안 됩니다.

---

## 요약

- 테스트는 디버깅 도구: 먼저 실패하는 테스트를 작성한 후 버그 수정
- `assert`는 인라인 검사 제공; `pytest`는 완전한 테스팅 프레임워크 제공
- `pytest.raises()`로 예외가 올바르게 발생하는지 테스트
- `test_` 파일에 `test_` 함수 이름으로 테스트 조직
- 픽스처가 재사용 가능한 설정/해체 제공
- 매개변수화된 테스트로 하나의 함수에서 많은 입력 테스트
- 테스트 커버리지는 미테스트 코드를 식별하지만 100%가 정확성을 보장하지 않음
- 모든 버그 수정은 회귀를 방지하는 테스트를 동반해야 함

---

## 연습문제

1. 정상 및 엣지 케이스를 포함하여 주어진 함수에 대한 pytest 테스트 작성하기
2. 테스트 주도 디버깅으로 버그 찾고 수정하기
3. 수학 함수에 대한 매개변수화된 테스트 작성하기
4. `pytest.raises()`로 에러 처리 테스트하기

**이전**: [로깅](./06_Logging.md) | **다음**: [린터와 포매터](./08_Linters_and_Formatters.md)
