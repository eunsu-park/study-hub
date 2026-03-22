"""
Exercise 01: Getting Started with Python

Practice basic Python programs: arithmetic calculations,
temperature conversion, and formatted output.
"""


def calculate_rectangle(width, height):
    """Return a tuple (area, perimeter) of a rectangle.

    Args:
        width: Width of the rectangle (positive number).
        height: Height of the rectangle (positive number).

    Returns:
        A tuple (area, perimeter).
    """
    # TODO: Implement this
    pass


def calculate_circle(radius):
    """Return a tuple (area, circumference) of a circle.

    Use math.pi for the value of pi.

    Args:
        radius: Radius of the circle (positive number).

    Returns:
        A tuple (area, circumference), both rounded to 2 decimal places.
    """
    # TODO: Implement this
    pass


def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit.

    Formula: F = C * 9/5 + 32

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Temperature in Fahrenheit.
    """
    # TODO: Implement this
    pass


def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius.

    Formula: C = (F - 32) * 5/9

    Args:
        fahrenheit: Temperature in Fahrenheit.

    Returns:
        Temperature in Celsius, rounded to 2 decimal places.
    """
    # TODO: Implement this
    pass


def format_greeting(name, age, city):
    """Return a formatted greeting string.

    The format must be exactly:
        "Hello, my name is {name}. I am {age} years old and live in {city}."

    Args:
        name: Person's name.
        age: Person's age.
        city: Person's city.

    Returns:
        Formatted greeting string.
    """
    # TODO: Implement this using an f-string
    pass


def format_table_row(item, quantity, price):
    """Return a formatted table row string.

    The format must be:
        "{item:<20}{quantity:>5}{price:>10.2f}"

    Args:
        item: Item name.
        quantity: Quantity (integer).
        price: Price (float).

    Returns:
        Formatted table row string.
    """
    # TODO: Implement this
    pass


# === Tests ===

assert calculate_rectangle(5, 3) == (15, 16), "Rectangle 5x3"
assert calculate_rectangle(10, 10) == (100, 40), "Square 10x10"

assert calculate_circle(1) == (3.14, 6.28), "Unit circle"
assert calculate_circle(5) == (78.54, 31.42), "Circle r=5"

assert celsius_to_fahrenheit(0) == 32, "Freezing point"
assert celsius_to_fahrenheit(100) == 212, "Boiling point"

assert fahrenheit_to_celsius(32) == 0, "Freezing point inverse"
assert fahrenheit_to_celsius(212) == 100.0, "Boiling point inverse"

assert format_greeting("Alice", 30, "Seoul") == (
    "Hello, my name is Alice. I am 30 years old and live in Seoul."
), "Greeting format"

assert format_table_row("Apple", 10, 1.5) == "Apple                   10      1.50", "Table row"

print("All tests passed!")
