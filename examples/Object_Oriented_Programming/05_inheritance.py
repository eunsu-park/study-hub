"""
Example 05: Inheritance
Topic: Object-Oriented Programming

Demonstrates basic inheritance, super(), method overriding,
and an employee hierarchy with polymorphic pay calculation.
"""

# =============================================================================
# BASIC INHERITANCE
# =============================================================================

class Animal:
    """Base class for animals."""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self, food):
        return f"{self.name} is eating {food}"

    def sleep(self):
        return f"{self.name} is sleeping... Zzz"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, age={self.age})"


class Dog(Animal):
    """Dog extends Animal with breed and bark."""

    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def bark(self):
        return f"{self.name} says: Woof!"

    def __repr__(self):
        return f"Dog({self.name!r}, {self.breed!r}, age={self.age})"


class Cat(Animal):
    """Cat extends Animal with indoor/outdoor status."""

    def __init__(self, name, age, indoor=True):
        super().__init__(name, age)
        self.indoor = indoor

    def purr(self):
        return f"{self.name} purrs..."

    def __repr__(self):
        loc = "indoor" if self.indoor else "outdoor"
        return f"Cat({self.name!r}, {loc}, age={self.age})"


# =============================================================================
# EMPLOYEE HIERARCHY
# =============================================================================

class Employee:
    """Base employee class."""

    def __init__(self, name, emp_id, base_salary):
        self.name = name
        self.emp_id = emp_id
        self.base_salary = base_salary

    def monthly_pay(self):
        """Calculate monthly pay. Override in subclasses."""
        return self.base_salary / 12

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, ${self.monthly_pay():,.0f}/mo)"


class SalariedEmployee(Employee):
    """Paid a fixed annual salary."""

    def monthly_pay(self):
        return self.base_salary / 12


class HourlyEmployee(Employee):
    """Paid by the hour."""

    def __init__(self, name, emp_id, hourly_rate, hours_per_week=40):
        super().__init__(name, emp_id, hourly_rate * hours_per_week * 52)
        self.hourly_rate = hourly_rate
        self.hours_per_week = hours_per_week

    def monthly_pay(self):
        weekly = self.hourly_rate * self.hours_per_week
        overtime = max(0, self.hours_per_week - 40) * self.hourly_rate * 0.5
        return (weekly + overtime) * 52 / 12


class Manager(SalariedEmployee):
    """Manager with bonus and direct reports."""

    def __init__(self, name, emp_id, base_salary, bonus_pct=0.10):
        super().__init__(name, emp_id, base_salary)
        self.bonus_pct = bonus_pct
        self.reports = []

    def add_report(self, employee):
        self.reports.append(employee)

    def monthly_pay(self):
        base = super().monthly_pay()
        return base * (1 + self.bonus_pct)


if __name__ == "__main__":
    # Animal hierarchy
    print("=== Animal Hierarchy ===")
    rex = Dog("Rex", 3, "German Shepherd")
    whiskers = Cat("Whiskers", 5)

    print(rex.eat("kibble"))
    print(whiskers.sleep())
    print(rex.bark())
    print(whiskers.purr())

    print(f"\n{rex} isinstance Animal? {isinstance(rex, Animal)}")
    print(f"{whiskers} isinstance Dog? {isinstance(whiskers, Dog)}")

    # Employee hierarchy
    print("\n=== Employee Hierarchy ===")
    team = [
        SalariedEmployee("Alice", "E001", 90000),
        HourlyEmployee("Bob", "E002", 35, 45),
        Manager("Carol", "E003", 120000, 0.15),
    ]

    for emp in team:
        print(emp)

    # Manager with reports
    carol = team[2]
    carol.add_report(team[0])
    carol.add_report(team[1])
    print(f"\n{carol.name} manages {len(carol.reports)} people")

    # Total payroll
    total = sum(e.monthly_pay() for e in team)
    print(f"Total monthly payroll: ${total:,.0f}")
