# 레슨 04: 캡슐화

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 캡슐화가 왜 OOP의 네 가지 기둥 중 하나인지 설명할 수 있다
2. Python의 접근 제어 명명 규칙 (`_`, `__`)을 적용할 수 있다
3. `@property` 데코레이터를 사용하여 getter/setter를 구현할 수 있다
4. 구현 세부사항을 숨기는 깨끗한 공개 인터페이스를 설계할 수 있다
5. 불변성을 강제하고 상태 변경을 검증하는 캡슐화를 사용할 수 있다
6. Python의 캡슐화 접근법을 다른 언어와 비교할 수 있다
7. 일반적인 캡슐화 실수를 피할 수 있다

## 캡슐화란?

캡슐화는 **데이터와 그 데이터를 조작하는 메서드를 함께 묶으면서** 객체의 일부 구성 요소에 대한 **직접 접근을 제한하는** 원칙입니다.

```
┌─────────────────────────────────────────────┐
│              캡슐화 없이                     │
│                                             │
│  외부 코드 ──── 직접 수정 ──────────────▶   │
│                 obj.balance = -999           │
│                 obj.status = "invalid"       │
│                                             │
│  결과: 깨진 불변성, 버그, 혼돈              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              캡슐화 사용                     │
│                                             │
│  외부 코드 ──── 메서드 사용 ────────────▶   │
│                 obj.withdraw(100)            │
│                 obj.set_status("active")     │
│                                             │
│  메서드가 검증, 규칙 강제, 로깅             │
│  결과: 일관된 상태, 적은 버그               │
└─────────────────────────────────────────────┘
```

## Python의 접근 제어 규칙

Java나 C++과 달리 `private`, `protected`, `public` 키워드가 있는 것이 아니라, Python은 **명명 규칙**에 의존합니다. "우리 모두 성숙한 성인입니다"라는 철학을 따릅니다.

```
┌──────────────┬─────────────┬──────────────────────────┐
│ 규칙         │ 접근 수준   │ 의미                     │
├──────────────┼─────────────┼──────────────────────────┤
│ name         │ 공개        │ 어디서든 자유롭게 사용    │
│ _name        │ 보호        │ "내부용" 힌트             │
│ __name       │ 비공개      │ Python이 이름을 맹글링   │
│ __name__     │ 던더/매직   │ Python에 의해 예약됨     │
└──────────────┴─────────────┴──────────────────────────┘
```

### 보호 속성 (단일 언더스코어 `_`)

단일 언더스코어는 **규칙**입니다 — 다른 개발자에게 "이것은 내부용이니 위험을 감수하고 사용하세요"라고 알려줍니다. Python은 이를 강제하지 않습니다.

### 비공개 속성 (이중 언더스코어 `__`)

이중 언더스코어는 **이름 맹글링**을 유발합니다: Python이 `__attr`을 `_ClassName__attr`로 이름을 변경하여 서브클래스에서의 우발적 접근을 방지합니다.

```python
class SecureAccount:
    def __init__(self, owner, pin):
        self.owner = owner
        self.__pin = pin       # _SecureAccount__pin으로 맹글링됨

    def verify_pin(self, pin):
        return pin == self.__pin

acct = SecureAccount("Bob", "1234")
# 직접 접근 실패
# print(acct.__pin)  -> AttributeError

# 하지만 맹글링된 이름으로는 여전히 접근 가능 (진정한 비공개가 아님!)
print(acct._SecureAccount__pin)  # "1234"
```

## `@property` 데코레이터

`@property` 데코레이터는 캡슐화를 위한 Python의 우아한 해결책입니다. 속성처럼 보이는 메서드를 정의할 수 있습니다:

```python
class Temperature:
    """자동 화씨 변환이 있는 온도."""

    def __init__(self, celsius=0):
        self.celsius = celsius  # setter를 트리거함!

    @property
    def celsius(self):
        """섭씨 온도를 가져옵니다."""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """검증과 함께 섭씨 온도를 설정합니다."""
        if value < -273.15:
            raise ValueError("절대 영도 아래의 온도는 불가능합니다")
        self._celsius = value

    @property
    def fahrenheit(self):
        """화씨 온도를 가져옵니다 (읽기 전용 계산 속성)."""
        return self._celsius * 9 / 5 + 32

t = Temperature(100)
print(t.celsius)      # 100 (속성 접근처럼 보이지만 getter 호출)
print(t.fahrenheit)   # 212.0 (즉석 계산)
t.celsius = 0         # 대입처럼 보이지만 검증 포함 setter 호출
```

### 읽기 전용 속성

```python
class Employee:
    def __init__(self, first_name, last_name, hourly_rate, hours_per_week):
        self.first_name = first_name
        self.last_name = last_name
        self.hourly_rate = hourly_rate
        self.hours_per_week = hours_per_week

    @property
    def full_name(self):
        """전체 이름은 읽기 전용이며 계산됩니다."""
        return f"{self.first_name} {self.last_name}"

    @property
    def weekly_pay(self):
        """주급은 읽기 전용이며 계산됩니다."""
        return self.hourly_rate * self.hours_per_week

emp = Employee("Alice", "Smith", 50, 40)
print(emp.full_name)       # Alice Smith
print(emp.weekly_pay)      # 2000
# emp.full_name = "Bob"    # AttributeError: 설정 불가
```

## 불변성 강제

캡슐화의 가장 큰 힘은 **불변성** 유지입니다 — 항상 참이어야 하는 조건:

```python
class DateRange:
    """시작이 끝보다 앞서야 하는 날짜 범위."""

    def __init__(self, start, end):
        if start > end:
            raise ValueError(f"시작({start})이 끝({end})보다 앞서야 합니다")
        self._start = start
        self._end = end

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if value > self._end:
            raise ValueError(f"시작({value})이 끝({self._end})보다 앞서야 합니다")
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if value < self._start:
            raise ValueError(f"끝({value})이 시작({self._start})보다 뒤여야 합니다")
        self._end = value
```

## 흔한 실수

### 실수 1: 과도한 캡슐화

```python
# 나쁜 예: 로직 없는 무의미한 getter/setter
class OverEngineered:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value  # 검증 없음, 로직 없음 — 의미없음!

# 좋은 예: 검증이 필요 없으면 공개 속성 사용
class Simple:
    def __init__(self, name):
        self.name = name
```

### 실수 2: 내부 가변 상태 노출

```python
# 나쁜 예: 내부 리스트 반환
class BadTeam:
    def __init__(self):
        self._members = []

    @property
    def members(self):
        return self._members  # 호출자가 내부 리스트를 변경할 수 있음!

# 좋은 예: 복사본 반환
class GoodTeam:
    def __init__(self):
        self._members = []

    @property
    def members(self):
        return list(self._members)  # 복사본 반환

    def add_member(self, name):
        if name in self._members:
            raise ValueError(f"{name}은(는) 이미 멤버입니다")
        self._members.append(name)
```

## 요약

- 캡슐화는 내부 상태에 대한 접근을 제어하면서 데이터와 메서드를 함께 묶습니다
- Python은 명명 규칙을 사용합니다: `public`, `_protected`, `__private` (이름 맹글링)
- `@property`는 getter/setter/deleter 로직으로 속성과 같은 구문을 제공합니다
- 속성을 사용하여 불변성을 강제하고 상태 변경을 검증하세요
- 검증이 필요 없을 때는 공개 속성을 선호하세요 — 사소한 getter/setter를 피하세요
- 내부 가변 상태를 직접 노출하지 마세요 — 복사본을 반환하세요

## 다음 단계

[레슨 05: 상속](05_Inheritance.md)에서 클래스가 부모 클래스로부터 속성과 메서드를 상속받는 방법을 탐구합니다.
