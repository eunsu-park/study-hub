# 레슨 03: 생성자와 초기화

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 다양한 매개변수 패턴으로 `__init__` 메서드를 작성할 수 있다
2. 생성자 인자를 검증하고 적절한 예외를 발생시킬 수 있다
3. 기본값과 선택적 매개변수를 효과적으로 사용할 수 있다
4. 복잡한 객체 생성을 위한 빌더 패턴을 구현할 수 있다
5. `__new__`와 `__init__`의 차이를 이해할 수 있다
6. 신중한 초기화를 통해 불변 스타일 객체를 생성할 수 있다
7. 일반적인 초기화 패턴과 안티패턴을 적용할 수 있다

## `__init__` 메서드

`__init__` 메서드는 Python의 **초기화 메서드**입니다 (기술적으로 `__new__`가 진짜 생성자이지만, 흔히 생성자라고 불립니다). 객체가 생성된 후 자동으로 호출되어 초기 상태를 설정합니다.

```
    Car("Toyota", 2024)
          │
          ▼
   ┌──────────────┐
   │ Python이      │
   │ Car.__new__() │  ── 메모리 할당, 빈 객체 반환
   │ 호출          │
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │ Python이      │
   │ Car.__init__()│  ── 객체에 속성 설정
   │ 호출          │
   └──────┬───────┘
          │
          ▼
   객체 사용 준비 완료
```

### `self` 매개변수

`self`는 초기화 중인 **현재 인스턴스**에 대한 참조입니다. 항상 인스턴스 메서드의 첫 번째 매개변수이지만, 직접 전달하지 않아도 됩니다 — Python이 처리합니다.

## 기본값

기본값을 사용하면 호출자가 합리적인 기본값이 있는 인자를 생략할 수 있습니다:

```python
class Connection:
    """합리적인 기본값을 가진 데이터베이스 연결."""

    def __init__(self, host="localhost", port=5432, database="mydb",
                 timeout=30, max_retries=3):
        self.host = host
        self.port = port
        self.database = database
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_connected = False

# 모든 기본값 사용
c1 = Connection()                           # localhost:5432/mydb

# 일부만 오버라이드
c2 = Connection(host="prod-db", port=5433)  # prod-db:5433/mydb
```

### 가변 기본 인자 함정

Python에서 가장 흔한 실수 중 하나:

```python
# 나쁜 예: 가변 기본 인자
class BadCollector:
    def __init__(self, items=[]):  # 이렇게 하면 안 됩니다!
        self.items = items

a = BadCollector()
a.items.append("x")

b = BadCollector()
print(b.items)  # ['x']  -- 리스트가 공유됨!


# 좋은 예: None을 기본값으로 사용
class GoodCollector:
    def __init__(self, items=None):
        self.items = items if items is not None else []

a = GoodCollector()
a.items.append("x")

b = GoodCollector()
print(b.items)  # []  -- 각 인스턴스가 자체 리스트를 가짐
```

```
┌─────────────────────────────────────────────┐
│  규칙: 함수 시그니처에서 가변 기본값        │
│  (list, dict, set)을 절대 사용하지 마세요.  │
│  None을 사용하고 __init__ 내에서 생성하세요.│
└─────────────────────────────────────────────┘
```

## `__init__`에서의 검증

입력을 조기에 검증하세요 — 명확한 오류 메시지로 빠르게 실패:

```python
class Temperature:
    """검증이 있는 온도 값."""

    ABSOLUTE_ZERO_C = -273.15
    VALID_SCALES = ("C", "F", "K")

    def __init__(self, value, scale="C"):
        if scale not in self.VALID_SCALES:
            raise ValueError(
                f"잘못된 단위 '{scale}'. {self.VALID_SCALES} 중 하나여야 합니다"
            )
        if not isinstance(value, (int, float)):
            raise TypeError(f"온도 값은 숫자여야 합니다. {type(value).__name__} 받음")

        celsius = self._to_celsius(value, scale)
        if celsius < self.ABSOLUTE_ZERO_C:
            raise ValueError(f"온도 {value}{scale}는 절대 영도 아래입니다")

        self._value = value
        self._scale = scale

    @staticmethod
    def _to_celsius(value, scale):
        if scale == "C":
            return value
        elif scale == "F":
            return (value - 32) * 5 / 9
        elif scale == "K":
            return value - 273.15
```

## 초기화 패턴

### 패턴 1: 키워드 전용 인자

명확성을 위해 호출자가 키워드 인자를 사용하도록 강제:

```python
class Config:
    def __init__(self, *, debug=False, verbose=False, log_file=None,
                 max_workers=4):
        # *는 모든 인자가 키워드로 전달되어야 함을 의미
        self.debug = debug
        self.verbose = verbose
        self.log_file = log_file
        self.max_workers = max_workers

# 명확하고 자체 문서화됨
config = Config(debug=True, max_workers=8)
```

### 패턴 2: `@classmethod`를 이용한 대체 생성자

```python
class Vector:
    """여러 생성 방법을 가진 2D 벡터."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_polar(cls, r, theta):
        """극좌표에서 벡터 생성."""
        import math
        return cls(r * math.cos(theta), r * math.sin(theta))

    @classmethod
    def from_tuple(cls, coords):
        """(x, y) 튜플에서 벡터 생성."""
        return cls(*coords)

    @classmethod
    def zero(cls):
        """영벡터 생성."""
        return cls(0, 0)
```

### 패턴 3: 빌더 패턴

선택적 설정 매개변수가 많은 객체의 경우:

```python
class Pizza:
    def __init__(self, size):
        self.size = size
        self.cheese = False
        self.pepperoni = False
        self.mushrooms = False

    def add_cheese(self):
        self.cheese = True
        return self  # 체이닝을 위해 self 반환

    def add_pepperoni(self):
        self.pepperoni = True
        return self

    def add_mushrooms(self):
        self.mushrooms = True
        return self

# 유창한 빌더 인터페이스 (메서드 체이닝)
pizza = (Pizza("large")
         .add_cheese()
         .add_pepperoni()
         .add_mushrooms())
```

## `__new__` vs `__init__`

대부분의 경우 `__init__`만 필요하지만, `__new__`를 이해하는 것이 중요합니다:

```
┌─────────────────────────────────────────────┐
│  __new__(cls, ...)                          │
│  - 먼저 호출됨                               │
│  - 메모리 할당                               │
│  - 새 인스턴스 반환                           │
│  - 거의 오버라이드하지 않음 (싱글턴,          │
│    불변 타입 제외)                            │
├─────────────────────────────────────────────┤
│  __init__(self, ...)                        │
│  - 두 번째로 호출됨                           │
│  - 이미 생성된 인스턴스를 받음                │
│  - 속성 설정                                  │
│  - 항상 None 반환                             │
│  - 거의 모든 클래스에서 오버라이드             │
└─────────────────────────────────────────────┘
```

## 흔한 실수

### 실수 1: `self` 잊기

```python
class Broken:
    def __init__(self, name):
        name = name  # 아무 일도 안 함 — 로컬 변수를 자기 자신에게 할당

class Fixed:
    def __init__(self, name):
        self.name = name  # 인스턴스 속성 생성
```

### 실수 2: `__init__`에서 반환

```python
class Bad:
    def __init__(self, x):
        self.x = x
        return self  # TypeError! __init__은 None을 반환해야 합니다
```

### 실수 3: `__init__`에서 무거운 작업

```python
# 나쁜 예: __init__에서 너무 많은 일을 함
class BadReport:
    def __init__(self, data_path):
        self.data = self._load_data(data_path)        # I/O
        self.analysis = self._run_analysis(self.data)   # CPU 집약적
        self._send_notification()                       # 부작용

# 좋은 예: 지연 초기화
class GoodReport:
    def __init__(self, data_path):
        self.data_path = data_path
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._load_data(self.data_path)
        return self._data
```

## 요약

- `__init__`은 객체 생성 후 호출되는 초기화 메서드로 인스턴스 속성을 설정합니다
- `self`는 현재 인스턴스를 참조합니다 — 항상 `self.attr`로 인스턴스 속성을 생성하세요
- 가변 인자 (리스트, 딕셔너리, 세트)에는 `None`을 기본값으로 사용하세요
- `__init__`에서 입력을 검증하고 명확한 오류 메시지로 빠르게 실패하세요
- 대체 생성자에는 `@classmethod`를 사용하세요
- 많은 매개변수에는 키워드 전용 인자 (`*`)를 사용하세요
- `__init__`에서 무거운 I/O나 부작용을 피하세요 — 지연 초기화를 선호하세요

## 다음 단계

[레슨 04: 캡슐화](04_Encapsulation.md)에서 객체의 내부 상태를 보호하고 제어된 인터페이스를 노출하는 방법을 배웁니다.
