# 레슨 08: 추상화

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. OOP 기둥으로서의 추상화를 설명하고 캡슐화와 구분할 수 있다
2. Python의 `abc` 모듈을 사용하여 추상 기본 클래스(ABC)를 생성할 수 있다
3. 추상 메서드와 추상 프로퍼티를 정의할 수 있다
4. ABC를 사용하여 서브클래스에 계약을 강제할 수 있다
5. ABC와 Protocol을 사용하여 인터페이스를 구현할 수 있다
6. 구조적 호환성을 위해 가상 서브클래스를 등록할 수 있다
7. 실제 시스템을 위한 효과적인 추상화를 설계할 수 있다

## 추상화란?

추상화는 **복잡한 구현 세부사항을 숨기고** 단순한 인터페이스를 통해 **필수 기능만 노출하는** 과정입니다. "이 객체가 *무엇을* 하는가?"에 답하면서 "어떻게 하는가?"는 드러내지 않습니다.

```
┌─────────────────────────────────────────────────┐
│  추상화 vs 캡슐화                               │
│                                                 │
│  추상화:     객체가 무엇을 하는가               │
│              인터페이스를 정의함                  │
│                                                 │
│  캡슐화:     어떻게 하는가                       │
│              구현을 숨김                          │
└─────────────────────────────────────────────────┘
```

## 추상 기본 클래스 (ABC)

Python의 `abc` 모듈은 추상 클래스를 만드는 도구를 제공합니다 — 인스턴스화할 수 없고 서브클래스에 계약을 강제하는 클래스입니다.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """도형의 추상 기본 클래스.
    모든 구체 서브클래스는 반드시 area()와 perimeter()를 구현해야 합니다.
    """

    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    # 구체 메서드: 모든 서브클래스에서 사용 가능
    def describe(self) -> str:
        return f"{self.__class__.__name__}: 면적={self.area():.2f}, 둘레={self.perimeter():.2f}"

# ABC는 인스턴스화할 수 없음
# shape = Shape()  -> TypeError

# 모든 추상 메서드를 구현해야 함
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        from math import pi
        return pi * self.radius ** 2

    def perimeter(self):
        from math import pi
        return 2 * pi * self.radius
```

## 추상 프로퍼티

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def sound(self) -> str:
        pass

    @property
    @abstractmethod
    def legs(self) -> int:
        pass

    def describe(self):
        return f"{self.name} ({self.__class__.__name__}): '{self.sound}' 소리, 다리 {self.legs}개"

class Dog(Animal):
    @property
    def sound(self):
        return "멍멍"

    @property
    def legs(self):
        return 4
```

## ABC를 통한 인터페이스

**인터페이스**는 순수 추상 클래스입니다 — 모든 메서드가 추상이고 구현은 제공하지 않습니다:

```python
from abc import ABC, abstractmethod

class Printable(ABC):
    @abstractmethod
    def to_string(self) -> str:
        pass

class Saveable(ABC):
    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

class Report(Printable, Saveable):
    """여러 인터페이스를 구현하는 보고서."""

    def __init__(self, title, content):
        self.title = title
        self.content = content

    def to_string(self):
        return f"=== {self.title} ===\n{self.content}"

    def save(self, path):
        print(f"보고서를 {path}에 저장 중")

    def load(self, path):
        print(f"보고서를 {path}에서 불러오는 중")
```

## 템플릿 메서드 패턴

ABC는 **템플릿 메서드** 패턴을 자주 구현합니다: 추상 클래스가 알고리즘의 뼈대를 정의하고, 서브클래스가 단계를 채웁니다:

```python
from abc import ABC, abstractmethod

class DataPipeline(ABC):
    """템플릿 메서드 패턴: 파이프라인을 정의하고 서브클래스가 단계를 채움."""

    def run(self):
        """템플릿 메서드 — 알고리즘의 뼈대를 정의합니다."""
        data = self.extract()
        cleaned = self.transform(data)
        self.load(cleaned)
        print("파이프라인 완료!")

    @abstractmethod
    def extract(self) -> list:
        pass

    @abstractmethod
    def transform(self, data: list) -> list:
        pass

    @abstractmethod
    def load(self, data: list) -> None:
        pass

class CSVPipeline(DataPipeline):
    def extract(self):
        print("CSV 파일에서 추출 중...")
        return [{"name": "Alice", "age": "30"}]

    def transform(self, data):
        print("나이를 정수로 변환 중...")
        return [{**d, "age": int(d["age"])} for d in data]

    def load(self, data):
        print(f"{len(data)}개 레코드를 데이터베이스에 적재함")
```

## 표준 라이브러리의 ABC

Python의 `collections.abc`는 유용한 ABC들을 제공합니다:

```python
from collections.abc import Sequence, Mapping, MutableSequence

# 표준 인터페이스 구현 여부 확인
print(isinstance([1, 2, 3], Sequence))     # True
print(isinstance({"a": 1}, Mapping))       # True

# ABC를 상속받아 커스텀 컬렉션 생성
class ValidatedList(MutableSequence):
    """항목 추가 전에 검증하는 리스트."""

    def __init__(self, validator, initial=None):
        self._validator = validator
        self._data = []
        if initial:
            for item in initial:
                self.append(item)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._validator(value)
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __len__(self):
        return len(self._data)

    def insert(self, index, value):
        self._validator(value)
        self._data.insert(index, value)

def positive_only(x):
    if x <= 0:
        raise ValueError(f"값은 양수여야 합니다, {x} 받음")

nums = ValidatedList(positive_only, [1, 2, 3])
nums.append(4)      # OK
# nums.append(-1)   # ValueError!
```

## 요약

- 추상화는 단순한 인터페이스 뒤에 복잡성을 숨깁니다 — "무엇을"이지 "어떻게"가 아닙니다
- `ABC`와 `@abstractmethod`를 사용하여 계약을 강제하는 추상 클래스를 만드세요
- 추상 클래스는 인스턴스화할 수 없습니다 — 서브클래스가 모든 추상 메서드를 구현해야 합니다
- 추상 프로퍼티와 클래스 메서드도 지원됩니다
- 템플릿 메서드 패턴은 ABC를 사용하여 알고리즘의 뼈대를 정의합니다
- Python의 `collections.abc`는 일반적인 인터페이스를 위한 표준 ABC를 제공합니다

## 다음 단계

[레슨 09: 합성 vs 상속](09_Composition_vs_Inheritance.md)에서 상속("is-a") 대신 합성("has-a")을 사용할 때를 탐구합니다.
