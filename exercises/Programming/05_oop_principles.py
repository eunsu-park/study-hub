"""
Exercises for Lesson 05: OOP Principles
Topic: Programming

Solutions to practice problems from the lesson.
"""
from abc import ABC, abstractmethod


# === Exercise 1: Identify Violations ===
# Problem: Review the EmailService class and identify SOLID violations.

def exercise_1():
    """Solution: Identify SOLID principle violations in EmailService."""

    print("  Original class: EmailService with send_email, save_user,")
    print("  generate_invoice, and calculate_tax methods.")
    print()

    violations = {
        "SRP (Single Responsibility Principle)": [
            "EmailService has FOUR unrelated responsibilities:",
            "  1. Sending emails (email logic)",
            "  2. Saving users (database logic)",
            "  3. Generating invoices (PDF generation)",
            "  4. Calculating taxes (business/financial logic)",
            "Each should be a separate class with one reason to change.",
        ],
        "OCP (Open/Closed Principle)": [
            "calculate_tax uses if/elif chain for countries.",
            "Adding a new country requires MODIFYING the class.",
            "Should use a strategy pattern or lookup table instead,",
            "so new countries can be ADDED without changing existing code.",
        ],
        "DIP (Dependency Inversion Principle)": [
            "SMTP server details are hardcoded (smtp.gmail.com, port 587).",
            "The class depends on a concrete implementation, not an abstraction.",
            "Should inject an email transport interface instead.",
        ],
    }

    for principle, issues in violations.items():
        print(f"  {principle}:")
        for issue in issues:
            print(f"    {issue}")
        print()

    # Demonstrate the fix
    print("  Refactored design:")
    print("    - EmailService: only sends emails (via injected transport)")
    print("    - UserRepository: only handles user persistence")
    print("    - InvoiceGenerator: only generates invoices")
    print("    - TaxCalculator: uses strategy/dict for extensible tax rates")


# === Exercise 2: Refactor to Composition ===
# Problem: Solve the diamond problem using composition instead of inheritance.

def exercise_2():
    """Solution: Refactor Employee hierarchy to use composition."""

    # The inheritance problem:
    # class Manager extends Employee { managePeople() }
    # class Developer extends Employee { writeCode() }
    # class ManagerDeveloper extends ??? { needs both }  <- Diamond!

    # Composition solution: define behaviors as separate components,
    # then compose them into employee roles.

    class ManagementSkill:
        """Component for management capabilities."""
        def manage_people(self):
            return "Managing team members, running 1:1s, setting goals"

    class DevelopmentSkill:
        """Component for development capabilities."""
        def write_code(self):
            return "Writing, reviewing, and debugging code"

    class Employee:
        """Base employee with composable skills (has-a, not is-a)."""
        def __init__(self, name, salary, skills=None):
            self.name = name
            self.salary = salary
            self._skills = skills or []

        def add_skill(self, skill):
            self._skills.append(skill)

        def work(self):
            """Demonstrate all skills this employee has."""
            activities = []
            for skill in self._skills:
                # Use duck typing: call whatever methods the skill has
                for method_name in dir(skill):
                    if not method_name.startswith("_"):
                        method = getattr(skill, method_name)
                        if callable(method):
                            activities.append(method())
            return activities

        def __repr__(self):
            skill_names = [type(s).__name__ for s in self._skills]
            return f"Employee('{self.name}', skills={skill_names})"

    # Create different roles through composition
    manager = Employee("Alice", 120000, [ManagementSkill()])
    developer = Employee("Bob", 100000, [DevelopmentSkill()])
    # No diamond problem: just compose both skills!
    manager_developer = Employee("Charlie", 130000, [ManagementSkill(), DevelopmentSkill()])

    for emp in [manager, developer, manager_developer]:
        print(f"  {emp}")
        for activity in emp.work():
            print(f"    -> {activity}")
        print()

    print("  Key insight: Composition avoids the diamond problem entirely.")
    print("  Roles are defined by what skills they HAVE, not what they ARE.")


# === Exercise 3: Apply Liskov Substitution ===
# Problem: Fix the Bird/Ostrich LSP violation.

def exercise_3():
    """Solution: Fix LSP violation where Ostrich extends Bird but can't fly."""

    # Original violation:
    # class Bird: def fly() -> works
    # class Ostrich(Bird): def fly() -> throws exception!
    # A function expecting Bird.fly() will crash with Ostrich.

    # Fix: separate flying ability from bird identity.
    # Not all birds fly, so "fly" shouldn't be in the base Bird class.

    class Bird(ABC):
        """Base class for all birds. Only includes universal bird behavior."""
        def __init__(self, name):
            self.name = name

        @abstractmethod
        def move(self):
            """All birds can move, but the mechanism varies."""
            pass

        def eat(self):
            return f"{self.name} is eating"

    class FlyingBird(Bird):
        """Birds that can fly. Subclasses of this are guaranteed to fly."""
        def move(self):
            return f"{self.name} is flying through the air"

        def fly(self):
            return f"{self.name} is soaring high!"

    class FlightlessBird(Bird):
        """Birds that cannot fly. They move differently."""
        def move(self):
            return f"{self.name} is walking/running on the ground"

    class Sparrow(FlyingBird):
        pass

    class Eagle(FlyingBird):
        pass

    class Ostrich(FlightlessBird):
        def move(self):
            return f"{self.name} is running at 70 km/h!"

    class Penguin(FlightlessBird):
        def move(self):
            return f"{self.name} is waddling and swimming"

    # Now LSP is satisfied: any Bird can move(), any FlyingBird can fly()
    def make_bird_move(bird: Bird):
        """Works correctly with ANY bird - no exceptions."""
        return bird.move()

    def make_bird_fly(bird: FlyingBird):
        """Only accepts birds that can actually fly."""
        return bird.fly()

    birds = [Sparrow("Sparrow"), Eagle("Eagle"), Ostrich("Ostrich"), Penguin("Penguin")]
    print("  All birds can move (LSP satisfied):")
    for bird in birds:
        print(f"    {make_bird_move(bird)}")

    print("\n  Only flying birds can fly:")
    for bird in birds:
        if isinstance(bird, FlyingBird):
            print(f"    {make_bird_fly(bird)}")
        else:
            print(f"    {bird.name}: Not a FlyingBird (correctly excluded)")


# === Exercise 4: Implement SOLID Design ===
# Problem: Design a notification system following SOLID principles.

def exercise_4():
    """Solution: SOLID-compliant notification system."""

    # S: Each class has a single responsibility
    # O: New notification types can be added without modifying existing code
    # L: All notifiers are substitutable for the base Notifier interface
    # I: Separate interfaces for sending and filtering
    # D: High-level NotificationService depends on abstractions

    # --- Abstractions (interfaces) ---

    class Notifier(ABC):
        """Interface for sending notifications (ISP: single-purpose)."""
        @abstractmethod
        def send(self, recipient, message):
            pass

    class NotificationFilter(ABC):
        """Interface for filtering notifications by priority (ISP)."""
        @abstractmethod
        def should_send(self, priority):
            pass

    # --- Concrete notifiers (OCP: extend, don't modify) ---

    class EmailNotifier(Notifier):
        """Send notifications via email."""
        def send(self, recipient, message):
            return f"[Email to {recipient}]: {message}"

    class SMSNotifier(Notifier):
        """Send notifications via SMS."""
        def send(self, recipient, message):
            return f"[SMS to {recipient}]: {message}"

    class PushNotifier(Notifier):
        """Send push notifications to devices."""
        def send(self, recipient, message):
            return f"[Push to {recipient}]: {message}"

    # --- Filters (OCP: new filters without changing existing code) ---

    class PriorityFilter(NotificationFilter):
        """Only send notifications at or above a minimum priority level."""
        LEVELS = {"low": 1, "medium": 2, "high": 3, "critical": 4}

        def __init__(self, min_priority="medium"):
            self._min_level = self.LEVELS.get(min_priority, 2)

        def should_send(self, priority):
            return self.LEVELS.get(priority, 0) >= self._min_level

    class AllPassFilter(NotificationFilter):
        """No filtering - send all notifications."""
        def should_send(self, priority):
            return True

    # --- Service (DIP: depends on abstractions, not concretions) ---

    class NotificationService:
        """
        High-level notification orchestrator.
        Depends on Notifier and NotificationFilter abstractions (DIP).
        """
        def __init__(self, notifiers, notification_filter=None):
            self._notifiers = notifiers  # List of Notifier abstractions
            self._filter = notification_filter or AllPassFilter()

        def notify(self, recipient, message, priority="medium"):
            if not self._filter.should_send(priority):
                return [f"Filtered out ({priority} priority): {message}"]

            results = []
            for notifier in self._notifiers:
                results.append(notifier.send(recipient, message))
            return results

    # --- Demonstration ---

    # Configure with different notifiers and filters
    service = NotificationService(
        notifiers=[EmailNotifier(), SMSNotifier(), PushNotifier()],
        notification_filter=PriorityFilter(min_priority="high"),
    )

    print("  Notification system (min priority: high):")
    for priority in ["low", "medium", "high", "critical"]:
        results = service.notify("alice@example.com", f"Alert ({priority})", priority)
        for r in results:
            print(f"    {r}")
        print()

    # Easy to extend: add a new notifier without changing anything
    class SlackNotifier(Notifier):
        def send(self, recipient, message):
            return f"[Slack to #{recipient}]: {message}"

    service2 = NotificationService(
        notifiers=[SlackNotifier()],
        notification_filter=AllPassFilter(),
    )
    print("  Extended with Slack (no code changes to existing classes):")
    for r in service2.notify("general", "New feature deployed!", "low"):
        print(f"    {r}")


# === Exercise 5: Code Review ===
# Problem: Identify OOP violations in sample code patterns.

def exercise_5():
    """Solution: Identify common OOP violations with examples."""

    print("  1. Encapsulation Violation:")
    print("     class User:")
    print("         name = ''          # Public attributes allow direct modification")
    print("         permissions = []   # Mutable collection exposed directly")
    print("     Fix: Use private attributes with getters/setters that validate")
    print()

    print("  2. Abstraction Improvement:")
    print("     class DatabaseManager:")
    print("         def execute_query(self, sql): ...")
    print("     Problem: Exposes raw SQL to callers (leaky abstraction)")
    print("     Fix: Provide domain-specific methods like find_user_by_id()")
    print()

    print("  3. Inappropriate Inheritance:")
    print("     class Stack(list):  # Inherits insert(), sort(), etc.")
    print("     Problem: Stack 'is-a' list exposes operations that break the contract")
    print("     Fix: Use composition - Stack 'has-a' list, expose only push/pop")
    print()

    print("  4. Where Polymorphism Would Help:")
    print("     def export(data, format):")
    print("         if format == 'json': ...")
    print("         elif format == 'csv': ...")
    print("         elif format == 'xml': ...")
    print("     Problem: Adding new formats requires modifying this function")
    print("     Fix: Define an Exporter interface with JSON/CSV/XML implementations")
    print()

    print("  5. SOLID Violations:")
    print("     a) SRP: class UserController handles auth + CRUD + email + logging")
    print("     b) OCP: if/elif chains for different payment types")
    print("     c) DIP: class OrderService creates MySQLConnection internally")
    print("     Fix each by separating concerns, using interfaces, injecting deps")


if __name__ == "__main__":
    print("=== Exercise 1: Identify Violations ===")
    exercise_1()
    print("\n=== Exercise 2: Refactor to Composition ===")
    exercise_2()
    print("\n=== Exercise 3: Apply Liskov Substitution ===")
    exercise_3()
    print("\n=== Exercise 4: Implement SOLID Design ===")
    exercise_4()
    print("\n=== Exercise 5: Code Review ===")
    exercise_5()
    print("\nAll exercises completed!")
