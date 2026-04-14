"""
Exercise 08: Abstraction
Topic: Object-Oriented Programming

Practice ABC, abstract methods, and the template method pattern.
"""

from abc import ABC, abstractmethod


class Validator(ABC):
    """Abstract validator interface.

    Abstract Methods:
        validate(value) -> bool: Return True if valid.
        error_message(value) -> str: Return error message for invalid values.

    Concrete Method:
        __call__(value): Raise ValueError(error_message) if not valid,
                         otherwise return True. This makes validators callable.
    """

    # TODO: Implement the abstract and concrete methods
    pass


class RangeValidator(Validator):
    """Validates that a number is within a range [min_val, max_val].

    Args:
        min_val (float): Minimum allowed value.
        max_val (float): Maximum allowed value.
    """

    # TODO: Implement this class
    pass


class PatternValidator(Validator):
    """Validates that a string matches a regex pattern.

    Args:
        pattern (str): Regex pattern string.
        description (str): Human-readable description of what's expected.

    Hint: Use re.fullmatch(pattern, value)
    """

    # TODO: Implement this class
    pass


class DataPipeline(ABC):
    """Abstract data pipeline using template method pattern.

    The run() method defines the algorithm skeleton:
        1. data = extract()
        2. cleaned = transform(data)
        3. result = load(cleaned)
        4. return result

    Subclasses implement extract(), transform(), and load().
    """

    def run(self):
        """Template method — do NOT override."""
        data = self.extract()
        cleaned = self.transform(data)
        result = self.load(cleaned)
        return result

    @abstractmethod
    def extract(self) -> list:
        pass

    @abstractmethod
    def transform(self, data: list) -> list:
        pass

    @abstractmethod
    def load(self, data: list) -> int:
        """Load data and return count of items loaded."""
        pass


class NumberPipeline(DataPipeline):
    """Pipeline that extracts numbers, filters positives, and loads them.

    extract(): Return the raw_data list passed to __init__.
    transform(data): Return only positive numbers, sorted ascending.
    load(data): Store in self.result and return count.
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test RangeValidator
    age_check = RangeValidator(0, 150)
    assert age_check.validate(25) is True
    assert age_check.validate(-1) is False
    assert age_check(25) is True

    try:
        age_check(200)
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"RangeValidator error: {e}")

    # Test PatternValidator
    email_check = PatternValidator(r"[^@]+@[^@]+\.[^@]+", "valid email address")
    assert email_check.validate("alice@mail.com") is True
    assert email_check.validate("invalid") is False

    try:
        email_check("bad-email")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"PatternValidator error: {e}")

    # Validators are callable (duck typing with __call__)
    assert callable(age_check)
    assert callable(email_check)

    # Test DataPipeline
    pipeline = NumberPipeline([5, -3, 8, -1, 2, 0, 7])
    count = pipeline.run()
    assert count == 4
    assert pipeline.result == [2, 5, 7, 8]
    print(f"\nPipeline result: {pipeline.result} ({count} items)")

    print("\nAll tests passed!")
