"""
01 Getting Started with Python
==============================
Demonstrates basic I/O, simple calculations, string formatting,
and fundamental Python syntax.
"""


def hello_world():
    """The classic first program."""
    print("Hello, World!")
    print("Welcome to Python programming!")


def basic_input_output():
    """Demonstrate print() options and input simulation."""
    # print() with various arguments
    print("Name:", "Alice", "Age:", 30)
    print("Items:", "apple", "banana", "cherry", sep=" | ")
    print("No newline here...", end=" ")
    print("continues on the same line.")

    # Simulating user input (in a real script, use input())
    name = "Alice"  # In interactive mode: name = input("Enter your name: ")
    print(f"Hello, {name}!")

    # Multiple values on one line
    print("x =", 10, "y =", 20, "sum =", 10 + 20)


def simple_calculations():
    """Basic arithmetic operations."""
    a, b = 17, 5

    print(f"a = {a}, b = {b}")
    print(f"Addition:       a + b = {a + b}")
    print(f"Subtraction:    a - b = {a - b}")
    print(f"Multiplication: a * b = {a * b}")
    print(f"Division:       a / b = {a / b}")
    print(f"Floor division: a // b = {a // b}")
    print(f"Modulus:        a % b = {a % b}")
    print(f"Exponentiation: a ** b = {a ** b}")

    # Order of operations (PEMDAS)
    result = 2 + 3 * 4 ** 2 - 1
    print(f"\n2 + 3 * 4 ** 2 - 1 = {result}")
    print("Evaluated as: 2 + 3 * 16 - 1 = 2 + 48 - 1 = 49")

    # Useful built-in math functions
    print(f"\nabs(-7)        = {abs(-7)}")
    print(f"round(3.14159) = {round(3.14159, 2)}")
    print(f"pow(2, 10)     = {pow(2, 10)}")
    print(f"divmod(17, 5)  = {divmod(17, 5)}")


def string_formatting():
    """Different ways to format strings in Python."""
    name = "Alice"
    age = 30
    height = 1.725

    # 1. f-strings (recommended, Python 3.6+)
    print(f"f-string:  {name} is {age} years old, {height:.1f}m tall.")

    # 2. str.format()
    print("format():  {} is {} years old, {:.1f}m tall.".format(name, age, height))

    # 3. %-formatting (older style)
    print("%%-format:  %s is %d years old, %.1fm tall." % (name, age, height))

    # f-string expressions and formatting
    price = 49.95
    quantity = 3
    print(f"\nPrice: ${price:>8.2f}")
    print(f"Qty:   {quantity:>8d}")
    print(f"Total: ${price * quantity:>8.2f}")

    # Alignment and padding
    for item in ["apple", "banana", "cherry"]:
        print(f"  {item:<10} | {'*' * len(item)}")

    # Debug with f-string = (Python 3.8+)
    x = 42
    print(f"\nDebug: {x = }, {x * 2 = }, {type(x) = }")


def multi_line_strings():
    """Triple-quoted strings and escape characters."""
    # Triple-quoted string
    poem = """Roses are red,
Violets are blue,
Python is great,
And so are you."""
    print(poem)

    # Common escape characters
    print("\nEscape characters:")
    print("Tab:\tindented")
    print("Newline: line1\nline2")
    print("Backslash: C:\\Users\\Alice")
    print("Quote: She said \"hello\"")

    # Raw strings (ignore escapes)
    raw = r"C:\new\test\path"
    print(f"\nRaw string: {raw}")


def comments_and_docs():
    """Show comment styles and docstring conventions."""
    # Single-line comment
    x = 42  # Inline comment

    # Multi-line comments use consecutive single-line comments
    # There is no dedicated multi-line comment syntax,
    # but triple-quoted strings serve as docstrings.

    def example_function(n):
        """
        Calculate the square of n.

        Args:
            n: A number to square.

        Returns:
            The square of n.
        """
        return n ** 2

    print(f"Square of 7: {example_function(7)}")
    print(f"Docstring: {example_function.__doc__.strip().splitlines()[0]}")


if __name__ == "__main__":
    sections = [
        ("Hello World", hello_world),
        ("Basic I/O", basic_input_output),
        ("Simple Calculations", simple_calculations),
        ("String Formatting", string_formatting),
        ("Multi-line Strings", multi_line_strings),
        ("Comments and Docs", comments_and_docs),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
