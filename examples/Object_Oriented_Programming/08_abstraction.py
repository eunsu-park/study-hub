"""
Example 08: Abstraction
Topic: Object-Oriented Programming

Demonstrates ABC, abstract methods, abstract properties,
template method pattern, and collections.abc usage.
"""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class DataStore(ABC):
    """Abstract interface for data storage."""

    @abstractmethod
    def save(self, key: str, value) -> None:
        """Save a value."""
        pass

    @abstractmethod
    def load(self, key: str):
        """Load a value. Return None if not found."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value. Return True if deleted."""
        pass

    @abstractmethod
    def keys(self) -> list:
        """Return all keys."""
        pass

    # Concrete method using abstract methods
    def exists(self, key: str) -> bool:
        return self.load(key) is not None

    def count(self) -> int:
        return len(self.keys())


class MemoryStore(DataStore):
    def __init__(self):
        self._data = {}

    def save(self, key, value):
        self._data[key] = value

    def load(self, key):
        return self._data.get(key)

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            return True
        return False

    def keys(self):
        return list(self._data.keys())


class FileStore(DataStore):
    """Simulated file-based store."""

    def __init__(self):
        self._files = {}

    def save(self, key, value):
        self._files[key] = str(value)
        print(f"  [FileStore] Wrote '{key}' to disk")

    def load(self, key):
        return self._files.get(key)

    def delete(self, key):
        if key in self._files:
            del self._files[key]
            print(f"  [FileStore] Deleted '{key}' from disk")
            return True
        return False

    def keys(self):
        return list(self._files.keys())


# =============================================================================
# TEMPLATE METHOD PATTERN
# =============================================================================

class ReportGenerator(ABC):
    """Template method: generate reports with customizable steps."""

    def generate(self, data):
        """Template method — defines the algorithm skeleton."""
        header = self.format_header()
        body = self.format_body(data)
        footer = self.format_footer(data)
        return f"{header}\n{body}\n{footer}"

    @abstractmethod
    def format_header(self) -> str:
        pass

    @abstractmethod
    def format_body(self, data) -> str:
        pass

    def format_footer(self, data) -> str:
        """Default footer — can be overridden."""
        return f"--- {len(data)} items ---"


class TextReport(ReportGenerator):
    def format_header(self):
        return "=" * 40 + "\n  TEXT REPORT\n" + "=" * 40

    def format_body(self, data):
        lines = [f"  {i+1}. {item}" for i, item in enumerate(data)]
        return "\n".join(lines)


class HTMLReport(ReportGenerator):
    def format_header(self):
        return "<html><body><h1>Report</h1>"

    def format_body(self, data):
        items = "\n".join(f"  <li>{item}</li>" for item in data)
        return f"<ul>\n{items}\n</ul>"

    def format_footer(self, data):
        return f"<p>Total: {len(data)} items</p></body></html>"


# =============================================================================
# CUSTOM COLLECTION WITH collections.abc
# =============================================================================

class TypedList(MutableSequence):
    """A list that only accepts items of a specific type."""

    def __init__(self, item_type, initial=None):
        self._type = item_type
        self._data = []
        if initial:
            for item in initial:
                self.append(item)

    def _validate(self, value):
        if not isinstance(value, self._type):
            raise TypeError(
                f"Expected {self._type.__name__}, got {type(value).__name__}"
            )

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._validate(value)
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __len__(self):
        return len(self._data)

    def insert(self, index, value):
        self._validate(value)
        self._data.insert(index, value)

    def __repr__(self):
        return f"TypedList[{self._type.__name__}]({self._data})"


if __name__ == "__main__":
    # DataStore abstraction
    print("=== DataStore Abstraction ===")
    for store_cls in [MemoryStore, FileStore]:
        store = store_cls()
        print(f"\n{store_cls.__name__}:")
        store.save("name", "Alice")
        store.save("age", 30)
        print(f"  Keys: {store.keys()}")
        print(f"  Count: {store.count()}")
        print(f"  name exists? {store.exists('name')}")
        print(f"  Load name: {store.load('name')}")
        store.delete("age")
        print(f"  After delete: {store.keys()}")

    # Template method
    print("\n=== Template Method Pattern ===")
    data = ["Python", "Java", "Rust", "Go"]

    text_report = TextReport()
    print(text_report.generate(data))

    print()
    html_report = HTMLReport()
    print(html_report.generate(data))

    # TypedList
    print("\n=== TypedList (collections.abc) ===")
    numbers = TypedList(int, [1, 2, 3])
    numbers.append(4)
    print(f"Numbers: {numbers}")
    print(f"Reversed: {list(reversed(numbers))}")  # Free from MutableSequence!

    try:
        numbers.append("not a number")
    except TypeError as e:
        print(f"Type error: {e}")
