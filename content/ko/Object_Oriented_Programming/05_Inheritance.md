# 레슨 05: 상속

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 상속과 "is-a" 관계를 설명할 수 있다
2. 부모 클래스를 확장하는 서브클래스를 생성할 수 있다
3. `super()`를 사용하여 부모 클래스 메서드를 호출할 수 있다
4. 부모 동작을 보존하면서 메서드를 오버라이드할 수 있다
5. 상속 계층에서 Python이 메서드 호출을 해결하는 방법을 이해할 수 있다
6. 현실 세계의 계층 구조를 모델링하는 데 상속을 적용할 수 있다
7. 상속이 적절한 경우와 그렇지 않은 경우를 구분할 수 있다

## 상속이란?

상속은 기존 클래스(**부모/슈퍼클래스**)로부터 새 클래스(**자식/서브클래스**)를 만들어 속성과 메서드를 물려받는 메커니즘입니다.

```
┌──────────────────┐
│     Animal       │  <-- 부모 (슈퍼클래스 / 기본 클래스)
├──────────────────┤
│  name            │
│  age             │
├──────────────────┤
│  eat()           │
│  sleep()         │
│  describe()      │
└────────┬─────────┘
         │ 상속
    ┌────┴─────┐
    │          │
┌───▼──────┐ ┌▼──────────┐
│   Dog    │ │   Cat      │  <-- 자식 (서브클래스)
├──────────┤ ├────────────┤
│  breed   │ │  indoor    │  <-- 새 속성
├──────────┤ ├────────────┤
│  bark()  │ │  purr()    │  <-- 새 메서드
│  fetch() │ │  scratch() │
└──────────┘ └────────────┘
```

핵심 관계는 **"is-a"**입니다: Dog IS an Animal. Cat IS an Animal.

## 기본 상속

```python
class Animal:
    """모든 동물의 기본 클래스."""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self, food):
        return f"{self.name}이(가) {food}을(를) 먹고 있습니다"

    def sleep(self):
        return f"{self.name}이(가) 자고 있습니다... Zzz"

    def describe(self):
        return f"{self.name} (나이 {self.age})"


class Dog(Animal):
    """개는 추가적인 개 특유의 동작을 가진 동물입니다."""

    def __init__(self, name, age, breed):
        super().__init__(name, age)  # 부모의 __init__ 호출
        self.breed = breed

    def bark(self):
        return f"{self.name} says: Woof! Woof!"

    def fetch(self, item):
        return f"{self.name}이(가) {item}을(를) 가져옵니다!"

    def describe(self):
        """부모의 describe를 오버라이드하여 품종 포함."""
        return f"{self.name} ({self.breed}, 나이 {self.age})"


# 사용
rex = Dog("Rex", 3, "German Shepherd")
print(rex.eat("kibble"))      # Rex이(가) kibble을(를) 먹고 있습니다
print(rex.bark())             # Rex says: Woof! Woof!
print(rex.describe())         # Rex (German Shepherd, 나이 3)

# isinstance 검사
print(isinstance(rex, Dog))    # True
print(isinstance(rex, Animal)) # True  (Dog IS an Animal)
```

## `super()` 함수

`super()`는 부모 클래스로 메서드 호출을 위임하는 프록시 객체를 반환합니다:

```python
class Shape:
    def __init__(self, color="black"):
        self.color = color

class Rectangle(Shape):
    def __init__(self, width, height, color="black"):
        super().__init__(color)  # 부모의 속성 초기화
        self.width = width
        self.height = height

class Square(Rectangle):
    def __init__(self, side, color="black"):
        super().__init__(side, side, color)  # Rectangle.__init__
```

### 부모를 직접 호출하지 않는 이유

```python
# 나쁜 예: 부모 클래스 이름을 하드코딩
class Child(Parent):
    def __init__(self):
        Parent.__init__(self)  # 다중 상속에서 깨짐!

# 좋은 예: super() 사용
class Child(Parent):
    def __init__(self):
        super().__init__()  # MRO와 올바르게 작동
```

## 메서드 오버라이딩

서브클래스가 부모 클래스에 이미 정의된 메서드의 특정 구현을 제공합니다:

```python
class Vehicle:
    def __init__(self, make, model, fuel_capacity):
        self.make = make
        self.model = model
        self.fuel_capacity = fuel_capacity
        self.fuel_level = fuel_capacity

    def fuel_efficiency(self):
        return 25.0  # 기본 mpg

    def range(self):
        return self.fuel_level * self.fuel_efficiency()

class Sedan(Vehicle):
    def fuel_efficiency(self):
        return 35.0  # 세단은 더 효율적

class Truck(Vehicle):
    def __init__(self, make, model, fuel_capacity, payload_capacity):
        super().__init__(make, model, fuel_capacity)
        self.payload_capacity = payload_capacity

    def fuel_efficiency(self):
        return 18.0  # 트럭은 덜 효율적

class ElectricCar(Vehicle):
    def __init__(self, make, model, battery_kwh):
        super().__init__(make, model, fuel_capacity=0)
        self.battery_kwh = battery_kwh

    def fuel_efficiency(self):
        return 4.0  # kWh당 마일

    def range(self):
        return self.battery_kwh * self.fuel_efficiency()
```

## 실전 예제: 직원 계층

```python
class Employee:
    def __init__(self, name, employee_id, base_salary):
        self.name = name
        self.employee_id = employee_id
        self.base_salary = base_salary

    def calculate_pay(self):
        return self.base_salary

class SalariedEmployee(Employee):
    def calculate_pay(self):
        return self.base_salary / 12  # 연봉을 월급으로

class HourlyEmployee(Employee):
    def __init__(self, name, employee_id, hourly_rate, hours_per_week=40):
        super().__init__(name, employee_id, hourly_rate)
        self.hourly_rate = hourly_rate
        self.hours_per_week = hours_per_week

    def calculate_pay(self):
        weekly = self.hourly_rate * self.hours_per_week
        overtime = max(0, self.hours_per_week - 40) * self.hourly_rate * 0.5
        return (weekly + overtime) * 52 / 12

class Manager(SalariedEmployee):
    def __init__(self, name, employee_id, base_salary, bonus_pct=0.1):
        super().__init__(name, employee_id, base_salary)
        self.bonus_pct = bonus_pct

    def calculate_pay(self):
        base_monthly = super().calculate_pay()
        return base_monthly + base_monthly * self.bonus_pct
```

## 상속을 사용할 때

### 좋은 사용 (진정한 "Is-A" 관계)

- `Dog`는 `Animal`이다
- `Manager`는 `Employee`이다
- `ElectricCar`는 `Vehicle`이다

### 나쁜 사용 (진정한 "Is-A"가 아님)

- `Stack`은 `List`가 아니다 (스택은 리스트 연산을 제한함)
- `Engine`은 `Car`가 아니다 (엔진은 자동차의 일부)

```
┌─────────────────────────────────────────────────┐
│  경험 법칙:                                     │
│                                                 │
│  "X는 Y이다"라고 자연스럽게 말할 수 있고       │
│  서브클래스가 부모가 사용되는 모든 곳에서        │
│  사용될 수 있다면 상속이 적절합니다.             │
│                                                 │
│  "X는 Y를 가지고 있다" 또는 "X는 Y를           │
│  사용한다"라면 합성을 사용하세요 (레슨 09).     │
└─────────────────────────────────────────────────┘
```

## 요약

- 상속은 자식이 부모의 모든 속성과 메서드를 물려받는 부모-자식 관계를 만듭니다
- `super()`를 사용하여 부모 메서드를 호출하세요 — 부모 클래스 이름을 하드코딩하지 마세요
- 메서드 오버라이딩으로 서브클래스가 특수한 동작을 제공할 수 있습니다
- `isinstance()`와 `issubclass()`로 상속 관계를 확인합니다
- 진정한 "is-a" 관계에만 상속을 사용하세요, 코드 재사용만을 위해서가 아닙니다
- 상속은 강한 결합을 만듭니다 — 얕은 계층을 선호하세요

## 다음 단계

[레슨 06: 다중 상속](06_Multiple_Inheritance.md)에서 클래스가 하나 이상의 부모로부터 상속받을 때 어떤 일이 일어나는지 탐구합니다.
