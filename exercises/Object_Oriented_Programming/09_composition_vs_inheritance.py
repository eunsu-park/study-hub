"""
Exercise 09: Composition vs Inheritance
Topic: Object-Oriented Programming

Practice composition, delegation, and the strategy pattern.
"""

from abc import ABC, abstractmethod


class Logger:
    """A simple logger component. Already implemented."""

    def __init__(self, name):
        self.name = name
        self.entries = []

    def log(self, message):
        entry = f"[{self.name}] {message}"
        self.entries.append(entry)
        return entry


class TextFormatter(ABC):
    """Abstract text formatter (Strategy)."""

    @abstractmethod
    def format(self, text: str) -> str:
        pass


class UpperCaseFormatter(TextFormatter):
    """Formats text to uppercase.

    format("hello") -> "HELLO"
    """

    # TODO: Implement
    pass


class TitleCaseFormatter(TextFormatter):
    """Formats text to title case.

    format("hello world") -> "Hello World"
    """

    # TODO: Implement
    pass


class SlugFormatter(TextFormatter):
    """Formats text to URL slug.

    format("Hello World!") -> "hello-world"
    (lowercase, spaces to hyphens, remove non-alphanumeric except hyphens)
    """

    # TODO: Implement
    pass


class TextProcessor:
    """Text processor composed of a formatter and logger.

    Uses composition (has-a) not inheritance.

    Args:
        formatter (TextFormatter): The formatting strategy.
        logger (Logger, optional): Logger for tracking operations.

    Methods:
        process(text): Format the text, log it (if logger exists),
                       return formatted text.
        set_formatter(formatter): Change the formatting strategy.
        history(): Return list of processed texts (original -> formatted).
    """

    # TODO: Implement this class
    pass


class FileSystem:
    """A simple in-memory file system using composition.

    Components:
        - files dict: {path: content}

    Methods:
        write(path, content): Write content to path.
        read(path): Read content. Raise FileNotFoundError if not found.
        delete(path): Delete file. Raise FileNotFoundError if not found.
        exists(path): Return True if path exists.
        list_files(): Return sorted list of all paths.
        __len__(): Return number of files.
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test TextProcessor with Strategy
    proc = TextProcessor(UpperCaseFormatter(), Logger("TextProc"))

    result = proc.process("hello world")
    assert result == "HELLO WORLD"

    proc.set_formatter(TitleCaseFormatter())
    result = proc.process("hello world")
    assert result == "Hello World"

    proc.set_formatter(SlugFormatter())
    result = proc.process("Hello World!")
    assert result == "hello-world"

    history = proc.history()
    assert len(history) == 3
    print("Processing history:")
    for entry in history:
        print(f"  {entry}")

    # Test FileSystem
    fs = FileSystem()
    fs.write("/readme.md", "# Hello")
    fs.write("/src/main.py", "print('hi')")

    assert fs.exists("/readme.md") is True
    assert fs.read("/readme.md") == "# Hello"
    assert len(fs) == 2
    assert fs.list_files() == ["/readme.md", "/src/main.py"]

    fs.delete("/readme.md")
    assert fs.exists("/readme.md") is False

    try:
        fs.read("/nonexistent")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass

    print(f"\nFileSystem: {len(fs)} files")

    print("\nAll tests passed!")
