# 레슨 06: 다중 상속

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 여러 부모로부터 상속받는 클래스를 정의할 수 있다
2. 메서드 해결 순서(MRO)와 C3 선형화를 설명할 수 있다
3. 다이아몬드 문제를 식별하고 해결할 수 있다
4. 깊은 계층 없이 재사용 가능한 동작을 추가하는 믹스인을 설계할 수 있다
5. 다중 상속 시나리오에서 `super()`를 올바르게 사용할 수 있다
6. 다중 상속에 대한 모범 사례를 적용할 수 있다

## 다중 상속이란?

다중 상속은 클래스가 **하나 이상의 부모 클래스**로부터 상속받아 모든 부모의 속성과 메서드를 결합하는 것입니다.

```
┌──────────┐     ┌──────────┐
│  Flyer   │     │ Swimmer  │
│ fly()    │     │ swim()   │
└────┬─────┘     └────┬─────┘
     │                │
     └────┬───────────┘
          │
     ┌────▼─────┐
     │  Duck    │
     │ fly()   │  Flyer에서 상속
     │ swim()  │  Swimmer에서 상속
     │ quack() │  자체 메서드
     └─────────┘
```

```python
class Flyer:
    def fly(self):
        return f"{self.__class__.__name__}이(가) 날고 있습니다!"

class Swimmer:
    def swim(self):
        return f"{self.__class__.__name__}이(가) 수영하고 있습니다!"

class Duck(Flyer, Swimmer):
    def quack(self):
        return "꽥! 꽥!"


donald = Duck()
print(donald.fly())    # Duck이(가) 날고 있습니다!
print(donald.swim())   # Duck이(가) 수영하고 있습니다!
print(donald.quack())  # 꽥! 꽥!
```

## 다이아몬드 문제

두 부모 클래스가 공통 조상을 공유할 때 발생합니다:

```
        ┌───────────┐
        │  Animal   │
        │  __init__ │
        └─────┬─────┘
         ┌────┴────┐
         │         │
    ┌────▼───┐ ┌───▼────┐
    │ Flyer  │ │Swimmer │
    └────┬───┘ └───┬────┘
         │         │
         └────┬────┘
         ┌────▼────┐
         │  Duck   │   <-- Animal.__init__이 한 번? 두 번?
         └─────────┘
```

### 협력적 `super()`로 해결

`**kwargs`를 사용하여 MRO 체인을 통해 인자를 전달합니다:

```python
class Animal:
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

class Flyer(Animal):
    def __init__(self, wingspan=0, **kwargs):
        super().__init__(**kwargs)
        self.wingspan = wingspan

class Swimmer(Animal):
    def __init__(self, swim_speed=0, **kwargs):
        super().__init__(**kwargs)
        self.swim_speed = swim_speed

class Duck(Flyer, Swimmer):
    def __init__(self, name, wingspan, swim_speed):
        super().__init__(name=name, wingspan=wingspan, swim_speed=swim_speed)

donald = Duck("Donald", wingspan=60, swim_speed=5)
print(donald.name)        # Donald
print(donald.wingspan)    # 60
print(donald.swim_speed)  # 5
```

## 메서드 해결 순서 (MRO)

객체에서 메서드를 호출할 때, Python은 **C3 알고리즘**을 사용한 **메서드 해결 순서**를 따릅니다.

```python
class A:
    def who(self):
        return "A"

class B(A):
    def who(self):
        return "B"

class C(A):
    def who(self):
        return "C"

class D(B, C):
    pass

print(D().who())  # "B"

# MRO가 이유를 설명합니다:
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

### MRO 규칙 (C3 선형화)

1. 클래스는 항상 부모보다 먼저 옵니다
2. 여러 부모로부터 상속하면 그 순서가 유지됩니다
3. 공통 부모는 모든 자식 뒤에 옵니다

## 믹스인

**믹스인**은 다중 상속을 통해 다른 클래스에 특정 동작을 제공하도록 설계된 클래스입니다. 독립적으로 사용되지 않으며, 집중된 기능을 추가합니다.

```
┌─────────────────────────────────────────────────┐
│  믹스인 규칙:                                   │
│  1. 믹스인을 직접 인스턴스화하지 않기           │
│  2. 믹스인은 단일 기능을 제공해야 함            │
│  3. 믹스인에 __init__이 없어야 함              │
│     (또는 협력적 **kwargs 사용)                 │
│  4. "Mixin" 접미사로 이름 붙이기               │
└─────────────────────────────────────────────────┘
```

```python
import json


class SerializableMixin:
    """JSON 직렬화 기능을 추가합니다."""

    def to_json(self):
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)


class LoggableMixin:
    """로깅 기능을 추가합니다."""

    def log(self, message, level="INFO"):
        class_name = self.__class__.__name__
        print(f"[{level}] {class_name}: {message}")


class ComparableMixin:
    """`_compare_key` 메서드를 기반으로 비교 연산자를 추가합니다."""

    def _compare_key(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._compare_key() == other._compare_key()

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._compare_key() < other._compare_key()


# 믹스인을 결합하여 기능이 풍부한 클래스 구성
class Product(SerializableMixin, LoggableMixin, ComparableMixin):
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def _compare_key(self):
        return self.price

laptop = Product("Laptop", 999)
phone = Product("Phone", 699)

laptop.log("새 제품 생성")          # [INFO] Product: 새 제품 생성
print(laptop.to_json())            # {"name": "Laptop", "price": 999}
print(laptop > phone)              # True (999 > 699)
```

## 모범 사례

1. **깊은 다중 상속보다 합성을 선호하세요** — 믹스인은 괜찮지만 복잡한 다이아몬드 계층은 아닙니다
2. **`super()`를 협력적으로 사용하세요** — 믹스인 `__init__`에서 항상 `**kwargs`를 전달하세요
3. **믹스인 클래스를 집중적으로 유지하세요** — 믹스인당 하나의 기능
4. **믹스인에 명확한 이름을 붙이세요** — `Mixin` 접미사 사용
5. **믹스인에서 상태를 피하세요** — 믹스인은 데이터가 아닌 동작을 추가해야 합니다
6. **MRO를 확인하세요** — `ClassName.__mro__`로 해결 순서를 검증하세요

## 요약

- 다중 상속은 클래스가 여러 부모의 동작을 결합할 수 있게 합니다
- **다이아몬드 문제**는 두 부모가 공통 조상을 공유할 때 발생합니다
- Python은 **MRO** (C3 선형화)를 사용하여 메서드 조회를 해결합니다
- **믹스인**은 다중 상속의 권장 패턴입니다: 집중적이고 상태 없는 동작 단위
- 적절한 초기화 체인을 위해 `**kwargs`와 함께 협력적 `super()`를 사용하세요
- 항상 `ClassName.__mro__`로 MRO를 검증하세요

## 다음 단계

[레슨 07: 다형성](07_Polymorphism.md)에서 같은 인터페이스가 객체 타입에 따라 다른 동작을 만들어내는 방법을 탐구합니다.
