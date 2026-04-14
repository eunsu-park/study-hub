# Lesson 05: Inheritance

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain inheritance and the "is-a" relationship
2. Create subclasses that extend parent classes
3. Use `super()` to call parent class methods
4. Override methods while preserving parent behavior
5. Understand how Python resolves method calls in an inheritance hierarchy
6. Apply inheritance to model real-world hierarchies
7. Recognize when inheritance is appropriate vs when it is not

## What Is Inheritance?

Inheritance is a mechanism where a new class (**child/subclass**) is created from an existing class (**parent/superclass**), inheriting its attributes and methods. The child class can then add new functionality or modify existing behavior.

```
┌──────────────────┐
│     Animal       │  <-- Parent (Superclass / Base class)
├──────────────────┤
│  name            │
│  age             │
├──────────────────┤
│  eat()           │
│  sleep()         │
│  describe()      │
└────────┬─────────┘
         │ inherits
    ┌────┴─────┐
    │          │
┌───▼──────┐ ┌▼──────────┐
│   Dog    │ │   Cat      │  <-- Children (Subclasses)
├──────────┤ ├────────────┤
│  breed   │ │  indoor    │  <-- New attributes
├──────────┤ ├────────────┤
│  bark()  │ │  purr()    │  <-- New methods
│  fetch() │ │  scratch() │
└──────────┘ └────────────┘
```

The key relationship is **"is-a"**: a Dog IS an Animal. A Cat IS an Animal. They share the common behavior of animals but also have their own unique behavior.

## Basic Inheritance

```python
class Animal:
    """Base class for all animals."""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self, food):
        return f"{self.name} is eating {food}"

    def sleep(self):
        return f"{self.name} is sleeping... Zzz"

    def describe(self):
        return f"{self.name} (age {self.age})"


class Dog(Animal):
    """A dog is an animal with additional dog-specific behavior."""

    def __init__(self, name, age, breed):
        super().__init__(name, age)  # Call parent's __init__
        self.breed = breed

    def bark(self):
        return f"{self.name} says: Woof! Woof!"

    def fetch(self, item):
        return f"{self.name} fetches the {item}!"

    def describe(self):
        """Override parent's describe to include breed."""
        return f"{self.name} ({self.breed}, age {self.age})"


class Cat(Animal):
    """A cat is an animal with additional cat-specific behavior."""

    def __init__(self, name, age, indoor=True):
        super().__init__(name, age)
        self.indoor = indoor

    def purr(self):
        return f"{self.name} purrs... Prrr"

    def scratch(self, surface):
        return f"{self.name} scratches the {surface}!"


# Usage
rex = Dog("Rex", 3, "German Shepherd")
whiskers = Cat("Whiskers", 5)

# Inherited methods work
print(rex.eat("kibble"))      # Rex is eating kibble
print(whiskers.sleep())       # Whiskers is sleeping... Zzz

# Subclass-specific methods
print(rex.bark())             # Rex says: Woof! Woof!
print(whiskers.purr())        # Whiskers purrs... Prrr

# Overridden method
print(rex.describe())         # Rex (German Shepherd, age 3)

# isinstance checks
print(isinstance(rex, Dog))    # True
print(isinstance(rex, Animal)) # True  (Dog IS an Animal)
print(isinstance(rex, Cat))    # False (Dog is NOT a Cat)
```

## The `super()` Function

`super()` returns a proxy object that delegates method calls to the parent class. It is essential for proper initialization and method extension.

```python
class Shape:
    def __init__(self, color="black"):
        self.color = color

    def describe(self):
        return f"A {self.color} shape"


class Rectangle(Shape):
    def __init__(self, width, height, color="black"):
        super().__init__(color)  # Initialize parent's attributes
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    def describe(self):
        # Extend parent behavior rather than replace it
        parent_desc = super().describe()
        return f"{parent_desc}: Rectangle {self.width}x{self.height}"


class Square(Rectangle):
    def __init__(self, side, color="black"):
        super().__init__(side, side, color)  # Rectangle.__init__

    def describe(self):
        parent_desc = super().describe()
        return f"{parent_desc} (square)"


sq = Square(5, "red")
print(sq.describe())  # A red shape: Rectangle 5x5 (square)
print(sq.area)        # 25
```

### Why Not Call the Parent Directly?

```python
# BAD: Hard-coding parent class name
class Child(Parent):
    def __init__(self):
        Parent.__init__(self)  # Breaks with multiple inheritance!

# GOOD: Using super()
class Child(Parent):
    def __init__(self):
        super().__init__()  # Works correctly with MRO
```

## Method Overriding

Method overriding lets a subclass provide a specific implementation for a method already defined in the parent class:

```python
class Vehicle:
    """Base class for vehicles."""

    def __init__(self, make, model, fuel_capacity):
        self.make = make
        self.model = model
        self.fuel_capacity = fuel_capacity
        self.fuel_level = fuel_capacity

    def fuel_efficiency(self):
        """Override in subclasses for specific efficiency."""
        return 25.0  # default mpg

    def range(self):
        """Calculate range based on fuel level and efficiency."""
        return self.fuel_level * self.fuel_efficiency()

    def describe(self):
        return f"{self.make} {self.model}"


class Sedan(Vehicle):
    def fuel_efficiency(self):
        return 35.0  # Sedans are more efficient


class Truck(Vehicle):
    def __init__(self, make, model, fuel_capacity, payload_capacity):
        super().__init__(make, model, fuel_capacity)
        self.payload_capacity = payload_capacity

    def fuel_efficiency(self):
        return 18.0  # Trucks are less efficient

    def describe(self):
        return f"{super().describe()} (payload: {self.payload_capacity} lbs)"


class ElectricCar(Vehicle):
    def __init__(self, make, model, battery_kwh):
        super().__init__(make, model, fuel_capacity=0)
        self.battery_kwh = battery_kwh

    def fuel_efficiency(self):
        return 4.0  # miles per kWh

    def range(self):
        """Override completely: electric cars use kWh, not gallons."""
        return self.battery_kwh * self.fuel_efficiency()

    def describe(self):
        return f"{super().describe()} (electric, {self.battery_kwh} kWh)"


# Polymorphism in action: same interface, different behavior
vehicles = [
    Sedan("Toyota", "Camry", 16),
    Truck("Ford", "F-150", 26, 2000),
    ElectricCar("Tesla", "Model 3", 75),
]

for v in vehicles:
    print(f"{v.describe()} — range: {v.range():.0f} miles")
# Toyota Camry — range: 560 miles
# Ford F-150 (payload: 2000 lbs) — range: 468 miles
# Tesla Model 3 (electric, 75 kWh) — range: 300 miles
```

## Inheritance Hierarchy and `isinstance`/`issubclass`

```python
class A:
    pass

class B(A):
    pass

class C(B):
    pass

c = C()

# isinstance checks the entire chain
print(isinstance(c, C))  # True
print(isinstance(c, B))  # True
print(isinstance(c, A))  # True

# issubclass checks class relationships
print(issubclass(C, B))  # True
print(issubclass(C, A))  # True
print(issubclass(B, A))  # True
print(issubclass(A, C))  # False
```

## Practical Example: Employee Hierarchy

```python
class Employee:
    """Base class for all employees."""

    def __init__(self, name, employee_id, base_salary):
        self.name = name
        self.employee_id = employee_id
        self.base_salary = base_salary

    def calculate_pay(self):
        """Calculate monthly pay. Override in subclasses."""
        return self.base_salary

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, ${self.calculate_pay():,.0f}/mo)"


class SalariedEmployee(Employee):
    """Employee paid a fixed salary."""

    def calculate_pay(self):
        return self.base_salary / 12  # Annual to monthly


class HourlyEmployee(Employee):
    """Employee paid by the hour."""

    def __init__(self, name, employee_id, hourly_rate, hours_per_week=40):
        super().__init__(name, employee_id, hourly_rate)
        self.hourly_rate = hourly_rate
        self.hours_per_week = hours_per_week

    def calculate_pay(self):
        weekly = self.hourly_rate * self.hours_per_week
        overtime = max(0, self.hours_per_week - 40) * self.hourly_rate * 0.5
        return (weekly + overtime) * 52 / 12  # Monthly


class Manager(SalariedEmployee):
    """Manager with a bonus structure."""

    def __init__(self, name, employee_id, base_salary, bonus_pct=0.1):
        super().__init__(name, employee_id, base_salary)
        self.bonus_pct = bonus_pct
        self.reports = []

    def add_report(self, employee):
        self.reports.append(employee)

    def calculate_pay(self):
        base_monthly = super().calculate_pay()
        bonus = base_monthly * self.bonus_pct
        return base_monthly + bonus


# Usage
team = [
    SalariedEmployee("Alice", "E001", 90000),
    HourlyEmployee("Bob", "E002", 35, 45),
    Manager("Carol", "E003", 120000, 0.15),
]

for emp in team:
    print(emp)
# SalariedEmployee(Alice, $7,500/mo)
# HourlyEmployee(Bob, $7,219/mo)
# Manager(Carol, $11,500/mo)
```

## When to Use Inheritance

### Good Uses (True "Is-A" Relationships)

- `Dog` is an `Animal`
- `Manager` is an `Employee`
- `ElectricCar` is a `Vehicle`
- `Square` is a `Rectangle` (but be careful! See Lesson 10: LSP)

### Bad Uses (Not True "Is-A")

- `Stack` is NOT a `List` (a stack restricts list operations)
- `Engine` is NOT a `Car` (an engine is PART OF a car)
- `Logger` is NOT a `FileWriter` (logging is a behavior, not an identity)

```
┌─────────────────────────────────────────────────┐
│  Rule of Thumb:                                 │
│                                                 │
│  If you can say "X IS A Y" naturally and the    │
│  subclass can be used EVERYWHERE the parent is  │
│  used, inheritance is appropriate.              │
│                                                 │
│  If you say "X HAS A Y" or "X USES A Y",       │
│  use composition instead (Lesson 09).           │
└─────────────────────────────────────────────────┘
```

## Summary

- Inheritance creates a parent-child relationship where the child inherits all attributes and methods
- Use `super()` to call parent methods — never hard-code parent class names
- Method overriding lets subclasses provide specialized behavior
- `isinstance()` and `issubclass()` check inheritance relationships
- Use inheritance for true "is-a" relationships, not for code reuse alone
- Inheritance creates tight coupling — prefer shallow hierarchies

## Next Steps

In [Lesson 06: Multiple Inheritance](06_Multiple_Inheritance.md), we will explore what happens when a class inherits from more than one parent.
