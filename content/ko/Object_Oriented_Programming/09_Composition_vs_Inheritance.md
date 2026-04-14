# 레슨 09: 합성 vs 상속

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. "is-a" (상속)과 "has-a" (합성) 관계를 구분할 수 있다
2. 작은 컴포넌트로 유연한 시스템을 구축하는 합성을 적용할 수 있다
3. 합성된 객체에 메서드 호출을 전달하는 위임을 구현할 수 있다
4. 상속이 잘못 사용된 경우를 인식하고 합성으로 리팩토링할 수 있다
5. 상속 기반 대안으로 전략 패턴을 사용할 수 있다
6. 합성과 상속을 효과적으로 결합할 수 있다
7. "상속보다 합성을 선호하라" 원칙을 따를 수 있다

## Is-A vs Has-A

```
┌─────────────────────────────────────────────┐
│  상속 (Is-A)                                │
│  "Dog IS an Animal"                         │
│  class Dog(Animal): ...                     │
│  서브클래스는 부모의 특수화된 버전이다       │
├─────────────────────────────────────────────┤
│  합성 (Has-A)                               │
│  "Car HAS an Engine"                        │
│  class Car:                                 │
│      def __init__(self):                    │
│          self.engine = Engine()             │
│  클래스가 다른 객체를 부품으로 포함한다      │
└─────────────────────────────────────────────┘
```

## 합성의 실제 적용

```python
class Engine:
    def __init__(self, horsepower, fuel_type="gasoline"):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.is_running = False

    def start(self):
        self.is_running = True
        return f"엔진 ({self.horsepower}마력) 시동됨"

    def stop(self):
        self.is_running = False
        return "엔진 정지"

class Transmission:
    def __init__(self, type="automatic", gears=6):
        self.type = type
        self.gears = gears
        self.current_gear = 0

    def shift(self, gear):
        if 0 <= gear <= self.gears:
            self.current_gear = gear
            return f"기어 {gear}로 변속"
        raise ValueError(f"잘못된 기어: {gear}")

class Car:
    """컴포넌트로 합성된 자동차 (상속받지 않음)."""

    def __init__(self, make, model, horsepower):
        self.make = make
        self.model = model
        self.engine = Engine(horsepower)          # 합성
        self.transmission = Transmission()         # 합성

    def start(self):
        return self.engine.start()                 # 위임

    def describe(self):
        return (f"{self.make} {self.model}: "
                f"{self.engine.horsepower}마력 {self.engine.fuel_type}, "
                f"{self.transmission.type} {self.transmission.gears}단")
```

## 전략 패턴: 합성이 상속을 대체

전략 패턴은 상속 계층을 교체 가능한 컴포넌트로 대체합니다:

```
┌─────────────────────────────────────────────────┐
│  상속 접근법 (경직됨):                          │
│       Sorter                                    │
│      /    \                                     │
│  BubbleSorter  QuickSorter  MergeSorter        │
│  문제: 런타임에 알고리즘 변경 불가              │
├─────────────────────────────────────────────────┤
│  합성 접근법 (유연함):                          │
│  Sorter ──가짐──▶ SortStrategy                 │
│                   /    |    \                    │
│              Bubble  Quick  Merge               │
│  장점: 런타임에 전략 교체 가능!                 │
└─────────────────────────────────────────────────┘
```

```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list) -> list:
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        arr = list(data)
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return list(data)
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class Sorter:
    """교체 가능한 전략을 사용하는 정렬기 (합성)."""

    def __init__(self, strategy: SortStrategy = None):
        self._strategy = strategy or BubbleSort()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: SortStrategy):
        self._strategy = new_strategy

    def sort(self, data):
        return self._strategy.sort(data)

sorter = Sorter(BubbleSort())
print(sorter.sort([5, 2, 8, 1]))

sorter.strategy = QuickSort()  # 런타임에 전략 교체!
print(sorter.sort([5, 2, 8, 1]))
```

## 언제 무엇을 사용할까

```
┌──────────────────────────────────────────────────────────────┐
│  상속을 사용할 때:                                            │
│  - 진정한 "is-a" 관계가 있을 때                               │
│  - 서브클래스가 부모를 어디서든 대체할 수 있을 때 (LSP)        │
│  - 부모의 인터페이스 + 구현을 재사용하고 싶을 때               │
│  - 계층이 얕을 때 (최대 2-3 수준)                             │
│                                                              │
│  합성을 사용할 때:                                            │
│  - "has-a" 또는 "uses-a" 관계일 때                            │
│  - 여러 소스의 동작을 결합해야 할 때                           │
│  - 런타임에 동작을 변경하고 싶을 때                            │
│  - 컴포넌트 간 느슨한 결합을 원할 때                           │
│  - 상속 계층이 깊거나 복잡해질 때                              │
└──────────────────────────────────────────────────────────────┘
```

### 상속에서 합성으로 리팩토링

```python
# 전: 깊은 상속 계층
class FlyingAnimal(Animal):
    def fly(self): pass

class SwimmingAnimal(Animal):
    def swim(self): pass

# 문제: 오리처럼 날고 수영하는 동물은?

# 후: 동작 객체를 사용한 합성
class FlyBehavior:
    def fly(self):
        return "날고 있습니다!"

class SwimBehavior:
    def swim(self):
        return "수영하고 있습니다!"

class Animal:
    def __init__(self, name, fly_behavior=None, swim_behavior=None):
        self.name = name
        self._fly_behavior = fly_behavior
        self._swim_behavior = swim_behavior

    def fly(self):
        return self._fly_behavior.fly() if self._fly_behavior else "날 수 없음"

    def swim(self):
        return self._swim_behavior.swim() if self._swim_behavior else "수영할 수 없음"

duck = Animal("오리", FlyBehavior(), SwimBehavior())
penguin = Animal("펭귄", None, SwimBehavior())
```

## 요약

- **상속** = "is-a" 관계 (Dog is an Animal)
- **합성** = "has-a" 관계 (Car has an Engine)
- 합성이 더 유연합니다: 컴포넌트를 런타임에 교체 가능
- 위임은 합성된 객체에서 컴포넌트로 메서드 호출을 전달합니다
- 전략 패턴은 상속 계층을 교체 가능한 컴포넌트로 대체합니다
- "상속보다 합성을 선호하라" — 진정한 "is-a" 관계와 얕은 계층에만 상속을 사용하세요

## 다음 단계

[레슨 10: SOLID 원칙](10_SOLID_Principles.md)에서 유지보수 가능하고 확장 가능한 OOP 시스템을 만들기 위한 다섯 가지 SOLID 설계 원칙을 배웁니다.
