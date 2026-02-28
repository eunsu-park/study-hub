"""
Exercises for Lesson 02: Programming Paradigms
Topic: Programming

Solutions to practice problems from the lesson.
"""
from functools import reduce


# === Exercise 1: Paradigm Recognition ===
# Problem: Identify the paradigm(s) used in each code snippet.

def exercise_1():
    """Solution: Identify paradigms in code snippets."""

    # Snippet A: numbers = [1,2,3,4,5]; result = sum(filter(lambda x: x%2==0, numbers))
    # -> FUNCTIONAL: uses higher-order functions (filter), lambda, no mutation

    # Snippet B: public class Car { private int speed; public void accelerate() { speed += 10; } }
    # -> OBJECT-ORIENTED: encapsulation (private field), class with methods, mutable state

    # Snippet C: document.getElementById('btn').addEventListener('click', function() { alert('Clicked!') })
    # -> EVENT-DRIVEN: responds to user events via callbacks/event handlers

    analyses = {
        "Snippet A (filter + lambda)": {
            "paradigm": "Functional Programming",
            "reasoning": [
                "Uses higher-order function 'filter' that takes a function as argument",
                "Uses lambda (anonymous function) for the predicate",
                "Declarative: describes WHAT to compute, not HOW",
                "No mutable state or side effects in the computation",
            ],
        },
        "Snippet B (Car class with accelerate)": {
            "paradigm": "Object-Oriented Programming",
            "reasoning": [
                "Uses a class to bundle data (speed) with behavior (accelerate)",
                "Encapsulation: 'private' access modifier hides internal state",
                "Mutable state: speed changes over time",
                "Models a real-world entity (Car) as an object",
            ],
        },
        "Snippet C (addEventListener)": {
            "paradigm": "Event-Driven Programming",
            "reasoning": [
                "Registers a callback function to respond to 'click' events",
                "Program flow determined by user actions, not sequential execution",
                "Asynchronous: handler runs when event occurs, not immediately",
                "Also uses functional concepts (function as callback argument)",
            ],
        },
    }

    for snippet, info in analyses.items():
        print(f"\n  {snippet}")
        print(f"    Paradigm: {info['paradigm']}")
        for reason in info["reasoning"]:
            print(f"      - {reason}")


# === Exercise 2: Imperative to Functional ===
# Problem: Rewrite imperative code in a functional style.

def exercise_2():
    """Solution: Convert imperative process_data to functional style."""

    # Original imperative version:
    # def process_data(numbers):
    #     result = []
    #     for num in numbers:
    #         if num > 0:
    #             result.append(num * 2)
    #     return result

    # Functional version using built-in higher-order functions.
    # Why functional: no mutable state (result list), no explicit loop,
    # each transformation is a separate, composable step.

    def process_data_functional(numbers):
        """Filter positive numbers and double them (functional style)."""
        return list(map(lambda x: x * 2, filter(lambda x: x > 0, numbers)))

    # More Pythonic functional version using list comprehension.
    # List comprehensions are considered idiomatic Python for this pattern.
    def process_data_pythonic(numbers):
        """Filter positive numbers and double them (Pythonic style)."""
        return [num * 2 for num in numbers if num > 0]

    test_data = [-3, -1, 0, 1, 2, 3, 4, 5]
    expected = [2, 4, 6, 8, 10]

    result_functional = process_data_functional(test_data)
    result_pythonic = process_data_pythonic(test_data)

    print(f"  Input: {test_data}")
    print(f"  Functional (map/filter): {result_functional}")
    print(f"  Pythonic (comprehension): {result_pythonic}")
    print(f"  Expected: {expected}")
    print(f"  Both correct: {result_functional == expected and result_pythonic == expected}")


# === Exercise 3: OOP Design ===
# Problem: Design classes for a simple library system.

def exercise_3():
    """Solution: Library system using OOP with encapsulation and inheritance."""

    class Book:
        """Represents a physical book in the library."""

        def __init__(self, title, author, isbn):
            self._title = title
            self._author = author
            self._isbn = isbn
            self._is_available = True

        @property
        def title(self):
            return self._title

        @property
        def author(self):
            return self._author

        @property
        def isbn(self):
            return self._isbn

        @property
        def is_available(self):
            return self._is_available

        @is_available.setter
        def is_available(self, value):
            self._is_available = value

        def __repr__(self):
            status = "Available" if self._is_available else "Borrowed"
            return f"Book('{self._title}' by {self._author}, {status})"

    class EBook(Book):
        """An electronic book that extends Book with a download URL."""

        def __init__(self, title, author, isbn, download_url):
            super().__init__(title, author, isbn)
            self._download_url = download_url
            # E-books are always "available" since they can be copied
            self._is_available = True

        @property
        def download_url(self):
            return self._download_url

        @Book.is_available.setter
        def is_available(self, value):
            # E-books are always available (override borrowing behavior)
            self._is_available = True

        def __repr__(self):
            return f"EBook('{self._title}' by {self._author}, URL: {self._download_url})"

    class Member:
        """Represents a library member who can borrow books."""

        def __init__(self, name, member_id):
            self._name = name
            self._member_id = member_id
            self._borrowed_books = []

        @property
        def name(self):
            return self._name

        @property
        def member_id(self):
            return self._member_id

        @property
        def borrowed_books(self):
            return list(self._borrowed_books)  # Return copy for encapsulation

        def borrow(self, book):
            self._borrowed_books.append(book)

        def return_book(self, book):
            self._borrowed_books.remove(book)

        def __repr__(self):
            return f"Member('{self._name}', borrowed: {len(self._borrowed_books)})"

    class Library:
        """Manages a collection of books and members with borrow/return operations."""

        def __init__(self, name):
            self._name = name
            self._books = []
            self._members = []

        def add_book(self, book):
            self._books.append(book)

        def register_member(self, member):
            self._members.append(member)

        def borrow_book(self, member, book):
            """Lend a book to a member if available."""
            if not book.is_available:
                return f"'{book.title}' is not available"
            book.is_available = False
            member.borrow(book)
            return f"'{book.title}' borrowed by {member.name}"

        def return_book(self, member, book):
            """Accept a returned book from a member."""
            book.is_available = True
            member.return_book(book)
            return f"'{book.title}' returned by {member.name}"

        def search_by_title(self, title):
            return [b for b in self._books if title.lower() in b.title.lower()]

        def search_by_author(self, author):
            return [b for b in self._books if author.lower() in b.author.lower()]

    # Demonstration
    library = Library("City Library")

    book1 = Book("Clean Code", "Robert C. Martin", "978-0132350884")
    book2 = Book("Design Patterns", "Gang of Four", "978-0201633610")
    ebook = EBook("Python Tricks", "Dan Bader", "978-1775093305", "https://example.com/python-tricks")

    library.add_book(book1)
    library.add_book(book2)
    library.add_book(ebook)

    alice = Member("Alice", "M001")
    library.register_member(alice)

    print(f"  {library.borrow_book(alice, book1)}")
    print(f"  {library.borrow_book(alice, book1)}")  # Should fail - already borrowed
    print(f"  Alice's books: {alice.borrowed_books}")
    print(f"  {library.return_book(alice, book1)}")
    print(f"  Search 'clean': {library.search_by_title('clean')}")
    print(f"  E-book always available: {ebook}")


# === Exercise 4: Functional vs OOP ===
# Problem: Calculate statistics using both OOP and functional approaches.

def exercise_4():
    """Solution: Statistics via OOP and functional approaches."""

    # --- OOP Approach ---
    # Why OOP: groups related operations with data, maintains state,
    # easy to extend with new statistics methods.
    class Statistics:
        """Compute basic statistics for a list of numbers."""

        def __init__(self, numbers):
            if not numbers:
                raise ValueError("Cannot compute statistics of empty list")
            self._numbers = list(numbers)

        @property
        def total(self):
            return sum(self._numbers)

        @property
        def average(self):
            return self.total / len(self._numbers)

        @property
        def maximum(self):
            return max(self._numbers)

        @property
        def minimum(self):
            return min(self._numbers)

        def summary(self):
            return {
                "sum": self.total,
                "avg": self.average,
                "max": self.maximum,
                "min": self.minimum,
            }

    # --- Functional Approach ---
    # Why functional: pure functions, composable, easy to test individually,
    # no hidden state.
    def stats_sum(numbers):
        return reduce(lambda a, b: a + b, numbers)

    def stats_avg(numbers):
        return stats_sum(numbers) / len(numbers)

    def stats_max(numbers):
        return reduce(lambda a, b: a if a > b else b, numbers)

    def stats_min(numbers):
        return reduce(lambda a, b: a if a < b else b, numbers)

    def stats_summary(numbers):
        """Pure function returning all statistics as a dict."""
        return {
            "sum": stats_sum(numbers),
            "avg": stats_avg(numbers),
            "max": stats_max(numbers),
            "min": stats_min(numbers),
        }

    data = [10, 20, 30, 40, 50]

    # OOP
    stats_oop = Statistics(data)
    print(f"  OOP approach: {stats_oop.summary()}")

    # Functional
    stats_fp = stats_summary(data)
    print(f"  Functional approach: {stats_fp}")

    # Both produce the same results
    print(f"  Results match: {stats_oop.summary() == stats_fp}")
    print(f"\n  Which feels more natural?")
    print(f"    - OOP: better when statistics are reused or extended (e.g., add median, std dev)")
    print(f"    - FP: better for one-shot computations and data pipelines")


# === Exercise 5: Multi-Paradigm ===
# Problem: Combine OOP, FP, and imperative styles in one program.

def exercise_5():
    """Solution: Multi-paradigm program combining OOP, FP, and imperative."""

    # Step 1: OOP - Define a Person class
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def __repr__(self):
            return f"Person('{self.name}', {self.age})"

    # Step 2: Create a list of Person objects
    people = [
        Person("Alice", 25),
        Person("Bob", 15),
        Person("Charlie", 30),
        Person("Diana", 17),
        Person("Eve", 22),
        Person("Frank", 12),
    ]

    # Step 3: Functional - filter adults (age >= 18)
    adults = list(filter(lambda p: p.age >= 18, people))

    # Step 4: Functional - sort by name using built-in sorted
    sorted_adults = sorted(adults, key=lambda p: p.name)

    # Step 5: Imperative - print results with a loop
    print(f"  All people: {people}")
    print(f"  Adults (sorted by name):")
    for person in sorted_adults:
        print(f"    {person.name}, age {person.age}")
    print(f"  Total adults: {len(sorted_adults)} out of {len(people)}")


# === Exercise 6: Event-Driven ===
# Problem: Write a simple event-driven program.
# Since this is a console exercise, we simulate an event system in Python.

def exercise_6():
    """Solution: Simple event-driven system (console-based simulation)."""

    # We simulate an event-driven architecture in pure Python.
    # In a real app, this would be a GUI or web framework with actual events.

    class EventEmitter:
        """Simple event system that mimics event-driven programming."""

        def __init__(self):
            self._listeners = {}

        def on(self, event, callback):
            """Register a callback for an event."""
            if event not in self._listeners:
                self._listeners[event] = []
            self._listeners[event].append(callback)

        def emit(self, event, *args):
            """Trigger all callbacks for an event."""
            if event in self._listeners:
                for callback in self._listeners[event]:
                    callback(*args)

    class SimpleApp:
        """Simulated event-driven app with a button and counter."""

        def __init__(self):
            self.emitter = EventEmitter()
            self.click_count = 0
            self.last_input = ""

            # Register event handlers (like addEventListener in JS)
            self.emitter.on("button_click", self._on_button_click)
            self.emitter.on("button_click", self._on_increment_counter)
            self.emitter.on("input_change", self._on_input_change)

        def _on_button_click(self, text):
            """Display the current input text (simulates alert/label update)."""
            print(f"    [Alert] You entered: '{text}'")

        def _on_increment_counter(self, text):
            """Increment click counter (bonus requirement)."""
            self.click_count += 1
            print(f"    [Counter] Button clicked {self.click_count} time(s)")

        def _on_input_change(self, text):
            """Handle input text changes."""
            self.last_input = text
            print(f"    [Input] Text changed to: '{text}'")

        def simulate_type(self, text):
            """Simulate user typing into an input field."""
            self.emitter.emit("input_change", text)

        def simulate_click(self):
            """Simulate a button click event."""
            self.emitter.emit("button_click", self.last_input)

    # Simulate user interactions
    app = SimpleApp()
    print("  Simulating event-driven app:")

    app.simulate_type("Hello, World!")
    app.simulate_click()

    app.simulate_type("Event-Driven Programming")
    app.simulate_click()

    app.simulate_click()  # Third click with same text


if __name__ == "__main__":
    print("=== Exercise 1: Paradigm Recognition ===")
    exercise_1()
    print("\n=== Exercise 2: Imperative to Functional ===")
    exercise_2()
    print("\n=== Exercise 3: OOP Design ===")
    exercise_3()
    print("\n=== Exercise 4: Functional vs OOP ===")
    exercise_4()
    print("\n=== Exercise 5: Multi-Paradigm ===")
    exercise_5()
    print("\n=== Exercise 6: Event-Driven ===")
    exercise_6()
    print("\nAll exercises completed!")
