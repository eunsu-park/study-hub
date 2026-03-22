"""
Exercises for Lesson 03: Data Types & Abstraction
Topic: Programming

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Type System Analysis ===
# Problem: Analyze JavaScript type coercion behavior.

def exercise_1():
    """Solution: Explain type coercion in JavaScript and compare with Python."""

    print("  JavaScript: let x = '10'; let y = 5;")
    print()

    # Q1: Why does + produce "105" but - produces 5?
    print("  Q1: Why does x + y produce '105' but x - y produces 5?")
    print("    The '+' operator is overloaded in JavaScript:")
    print("    - If either operand is a string, '+' performs string concatenation")
    print("    - So '10' + 5 converts 5 to '5', then concatenates: '105'")
    print("    The '-' operator only works with numbers:")
    print("    - JavaScript implicitly converts '10' to the number 10")
    print("    - Then computes 10 - 5 = 5")
    print()

    # Q2: Would this work in Python?
    print("  Q2: Would this work in Python?")
    print("    No. Python is strongly typed:")
    try:
        x = "10"
        y = 5
        result = x + y  # type: ignore
    except TypeError as e:
        print(f"    x + y raises TypeError: {e}")
    print("    Python requires explicit conversion: int('10') + 5 = 15")
    print(f"    Result: {int('10') + 5}")
    print()

    # Q3: Strong/weak and static/dynamic?
    print("  Q3: What kind of typing does JavaScript use?")
    print("    - Weak typing: allows implicit type coercion (string to number)")
    print("    - Dynamic typing: variable types determined at runtime")
    print("    Python, by contrast, is strongly typed (no implicit coercion)")
    print("    but also dynamically typed (types checked at runtime)")


# === Exercise 2: Implement a Stack ADT ===
# Problem: Implement a stack using an array, test with ints and strings.

def exercise_2():
    """Solution: Stack ADT using a Python list as underlying storage."""

    class Stack:
        """
        Stack ADT backed by a Python list.

        Supports any type (Python's dynamic typing gives us generics for free).
        LIFO: Last-In, First-Out ordering.
        """

        def __init__(self):
            self._data = []

        def push(self, item):
            """Add item to the top of the stack."""
            self._data.append(item)

        def pop(self):
            """Remove and return the top item. Raises IndexError if empty."""
            if self.is_empty():
                raise IndexError("Cannot pop from an empty stack")
            return self._data.pop()

        def peek(self):
            """Return the top item without removing it."""
            if self.is_empty():
                raise IndexError("Cannot peek at an empty stack")
            return self._data[-1]

        def is_empty(self):
            """Check if the stack has no elements."""
            return len(self._data) == 0

        def __len__(self):
            return len(self._data)

        def __repr__(self):
            return f"Stack({self._data})"

    # Test with integers
    print("  Testing with integers:")
    int_stack = Stack()
    for val in [10, 20, 30]:
        int_stack.push(val)
    print(f"    Stack after pushing 10, 20, 30: {int_stack}")
    print(f"    peek(): {int_stack.peek()}")
    print(f"    pop(): {int_stack.pop()}")
    print(f"    pop(): {int_stack.pop()}")
    print(f"    Stack now: {int_stack}")

    # Test with strings (same code works - no type changes needed)
    print("\n  Testing with strings:")
    str_stack = Stack()
    for word in ["hello", "world", "python"]:
        str_stack.push(word)
    print(f"    Stack after pushing words: {str_stack}")
    print(f"    pop(): {str_stack.pop()}")

    # Test empty stack error handling
    print("\n  Testing edge case (empty stack):")
    empty_stack = Stack()
    try:
        empty_stack.pop()
    except IndexError as e:
        print(f"    pop() on empty stack: {e}")


# === Exercise 3: Generics ===
# Problem: Implement a Pair<T, U> class that holds two values.

def exercise_3():
    """Solution: Generic Pair class holding two values of any type."""

    # In Python, generics are achieved through duck typing and type hints.
    # Python doesn't enforce type constraints at runtime like Java/C++,
    # but type hints document the intended types for tools like mypy.
    from typing import TypeVar, Generic

    T = TypeVar("T")
    U = TypeVar("U")

    class Pair(Generic[T, U]):
        """
        A generic pair holding two values of potentially different types.

        Equivalent to Java's Pair<T, U> or C++ std::pair<T, U>.
        """

        def __init__(self, first: T, second: U):
            self._first = first
            self._second = second

        def get_first(self) -> T:
            return self._first

        def get_second(self) -> U:
            return self._second

        def set_first(self, value: T):
            self._first = value

        def set_second(self, value: U):
            self._second = value

        def __repr__(self):
            return f"Pair({self._first!r}, {self._second!r})"

    # Test: Pair[str, int] for ("Alice", 30)
    person = Pair("Alice", 30)
    print(f"  Created: {person}")
    print(f"  First: {person.get_first()}")
    print(f"  Second: {person.get_second()}")

    person.set_first("Bob")
    person.set_second(25)
    print(f"  After update: {person}")

    # Test with different types
    coord = Pair(3.14, True)
    print(f"  Float-Bool pair: {coord}")

    nested = Pair("key", [1, 2, 3])
    print(f"  String-List pair: {nested}")


# === Exercise 4: Option Type ===
# Problem: Implement a simple Option<T> type similar to Rust/Java.

def exercise_4():
    """Solution: Option type for safe null handling."""

    class Option:
        """
        Option type that explicitly represents presence or absence of a value.

        Prevents null/None-related errors by forcing callers to handle
        both cases. Inspired by Rust's Option<T> and Java's Optional<T>.
        """

        def __init__(self, value=None, _is_some=False):
            # Private constructor pattern - use Some() or NONE instead
            self._value = value
            self._is_some = _is_some

        @staticmethod
        def some(value):
            """Create an Option containing a value."""
            return Option(value, _is_some=True)

        @staticmethod
        def none():
            """Create an empty Option."""
            return Option(None, _is_some=False)

        def is_some(self):
            """Returns True if this Option contains a value."""
            return self._is_some

        def is_none(self):
            """Returns True if this Option is empty."""
            return not self._is_some

        def unwrap(self):
            """Returns the contained value or raises an error if empty."""
            if self.is_none():
                raise ValueError("Called unwrap() on a None Option")
            return self._value

        def unwrap_or(self, default):
            """Returns the contained value, or the default if empty."""
            if self.is_some():
                return self._value
            return default

        def map(self, func):
            """Apply a function to the contained value, if present."""
            if self.is_some():
                return Option.some(func(self._value))
            return Option.none()

        def __repr__(self):
            if self.is_some():
                return f"Some({self._value!r})"
            return "None"

    # Usage demonstrations
    def safe_divide(a, b):
        """Division that returns Option instead of raising ZeroDivisionError."""
        if b == 0:
            return Option.none()
        return Option.some(a / b)

    # Test cases
    result1 = safe_divide(10, 2)
    result2 = safe_divide(10, 0)

    print(f"  10 / 2 = {result1}")
    print(f"  10 / 0 = {result2}")
    print(f"  unwrap 10/2: {result1.unwrap()}")
    print(f"  unwrap_or 10/0 with default 0: {result2.unwrap_or(0)}")

    # Chaining with map
    doubled = result1.map(lambda x: x * 2)
    print(f"  10/2 doubled: {doubled}")

    none_doubled = result2.map(lambda x: x * 2)
    print(f"  10/0 doubled: {none_doubled}")  # Still None, map is a no-op

    # Test unwrap on None
    try:
        result2.unwrap()
    except ValueError as e:
        print(f"  unwrap() on None: {e}")


# === Exercise 5: ADT Design ===
# Problem: Design an ADT interface for a Library System.

def exercise_5():
    """Solution: Design a Library System ADT interface."""

    from abc import ABC, abstractmethod

    # The ADT defines WHAT operations are available without specifying HOW.
    # Multiple implementations could use different backing stores
    # (dict, database, file system, etc.).

    class LibraryADT(ABC):
        """
        Abstract Data Type for a Library System.

        This interface defines the operations; implementations decide
        whether to use lists, databases, or other storage.
        """

        @abstractmethod
        def add_book(self, title, author, isbn):
            """Add a book to the collection. Returns book ID."""
            pass

        @abstractmethod
        def remove_book(self, isbn):
            """Remove a book by ISBN. Returns True if found and removed."""
            pass

        @abstractmethod
        def search_by_title(self, title):
            """Search books by title (partial match). Returns list of books."""
            pass

        @abstractmethod
        def search_by_author(self, author):
            """Search books by author (partial match). Returns list of books."""
            pass

        @abstractmethod
        def search_by_isbn(self, isbn):
            """Find a specific book by ISBN. Returns book or None."""
            pass

        @abstractmethod
        def borrow_book(self, isbn, member_id):
            """Borrow a book. Returns True if successful, False if unavailable."""
            pass

        @abstractmethod
        def return_book(self, isbn, member_id):
            """Return a borrowed book. Returns True if successful."""
            pass

    # Concrete implementation using dictionaries
    class DictLibrary(LibraryADT):
        """Library implementation using Python dictionaries."""

        def __init__(self):
            self._books = {}       # isbn -> {title, author, available}
            self._borrowed = {}    # isbn -> member_id

        def add_book(self, title, author, isbn):
            self._books[isbn] = {"title": title, "author": author, "available": True}
            return isbn

        def remove_book(self, isbn):
            if isbn in self._books:
                del self._books[isbn]
                return True
            return False

        def search_by_title(self, title):
            title_lower = title.lower()
            return [
                {"isbn": isbn, **info}
                for isbn, info in self._books.items()
                if title_lower in info["title"].lower()
            ]

        def search_by_author(self, author):
            author_lower = author.lower()
            return [
                {"isbn": isbn, **info}
                for isbn, info in self._books.items()
                if author_lower in info["author"].lower()
            ]

        def search_by_isbn(self, isbn):
            if isbn in self._books:
                return {"isbn": isbn, **self._books[isbn]}
            return None

        def borrow_book(self, isbn, member_id):
            book = self._books.get(isbn)
            if book and book["available"]:
                book["available"] = False
                self._borrowed[isbn] = member_id
                return True
            return False

        def return_book(self, isbn, member_id):
            if isbn in self._borrowed and self._borrowed[isbn] == member_id:
                self._books[isbn]["available"] = True
                del self._borrowed[isbn]
                return True
            return False

    # Demonstrate the ADT in action
    lib = DictLibrary()
    lib.add_book("Clean Code", "Robert Martin", "978-0132350884")
    lib.add_book("Clean Architecture", "Robert Martin", "978-0134494166")
    lib.add_book("Python Crash Course", "Eric Matthes", "978-1593279288")

    print("  Search 'clean':", lib.search_by_title("clean"))
    print("  Search author 'martin':", lib.search_by_author("martin"))
    print(f"  Borrow Clean Code: {lib.borrow_book('978-0132350884', 'M001')}")
    print(f"  Borrow again (unavailable): {lib.borrow_book('978-0132350884', 'M002')}")
    print(f"  Return: {lib.return_book('978-0132350884', 'M001')}")
    print(f"  Now available: {lib.search_by_isbn('978-0132350884')}")

    print("\n  Possible implementations for this ADT:")
    print("    - Dict/HashMap: O(1) ISBN lookup, good for small collections")
    print("    - SQL Database: scalable, persistent, supports complex queries")
    print("    - B-Tree index: efficient range queries and sorted access")
    print("    - Elasticsearch: full-text search, fuzzy matching")


# === Exercise 6: Null Safety ===
# Problem: Refactor Java-style code using Optional pattern in Python.

def exercise_6():
    """Solution: Null-safe email lookup using Optional-style pattern."""

    from typing import Optional

    # Original unsafe pattern (translated to Python):
    # def get_user_email(user_id):
    #     if user_id == 1: return "alice@example.com"
    #     return None  # Dangerous!
    # email = get_user_email(1)
    # print(email.upper())  # AttributeError if None!

    # Safe version using Optional type hint and explicit None handling
    def get_user_email(user_id: int) -> Optional[str]:
        """Look up user email by ID. Returns None if user not found."""
        users = {
            1: "alice@example.com",
            2: "bob@example.com",
        }
        return users.get(user_id)

    # Safe usage pattern 1: Explicit None check
    def print_email_safe(user_id: int):
        """Safely print a user's email in uppercase."""
        email = get_user_email(user_id)
        if email is not None:
            print(f"    User {user_id}: {email.upper()}")
        else:
            print(f"    User {user_id}: Not found")

    # Safe usage pattern 2: Default value
    def get_email_or_default(user_id: int, default: str = "unknown@example.com") -> str:
        """Get email with a fallback default."""
        email = get_user_email(user_id)
        return email if email is not None else default

    # Demonstrate
    print("  Pattern 1: Explicit None check")
    print_email_safe(1)  # Found
    print_email_safe(99)  # Not found

    print("\n  Pattern 2: Default value")
    print(f"    User 1: {get_email_or_default(1)}")
    print(f"    User 99: {get_email_or_default(99)}")

    print("\n  Key takeaway:")
    print("    Never call methods on a value that might be None.")
    print("    Use Optional type hints to signal that None is possible.")
    print("    Force callers to handle both cases explicitly.")


if __name__ == "__main__":
    print("=== Exercise 1: Type System Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Implement a Stack ADT ===")
    exercise_2()
    print("\n=== Exercise 3: Generics ===")
    exercise_3()
    print("\n=== Exercise 4: Option Type ===")
    exercise_4()
    print("\n=== Exercise 5: ADT Design ===")
    exercise_5()
    print("\n=== Exercise 6: Null Safety ===")
    exercise_6()
    print("\nAll exercises completed!")
