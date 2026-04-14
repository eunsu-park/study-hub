"""
Exercise 05: Inheritance
Topic: Object-Oriented Programming

Practice inheritance, super(), and method overriding.
"""


class Shape:
    """Base class for shapes. Already implemented for you."""

    def __init__(self, color="black"):
        self.color = color

    def area(self):
        raise NotImplementedError

    def perimeter(self):
        raise NotImplementedError

    def describe(self):
        return f"{self.color} {self.__class__.__name__}: area={self.area():.2f}"


class Circle(Shape):
    """Circle inheriting from Shape.

    Args:
        radius (float): Circle radius. Must be positive.
        color (str): Color (default "black").

    Properties:
        area: pi * r^2
        perimeter: 2 * pi * r

    Must call super().__init__.
    """

    # TODO: Implement this class
    pass


class Rectangle(Shape):
    """Rectangle inheriting from Shape.

    Args:
        width, height (float): Dimensions. Must be positive.
        color (str): Color (default "black").

    Properties:
        area: width * height
        perimeter: 2 * (width + height)
        is_square: True if width == height
    """

    # TODO: Implement this class
    pass


class Square(Rectangle):
    """Square inheriting from Rectangle.

    Args:
        side (float): Side length.
        color (str): Color.

    Should reuse Rectangle's implementation.
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test Circle
    c = Circle(5, "red")
    assert abs(c.area() - 78.54) < 0.01
    assert abs(c.perimeter() - 31.42) < 0.01
    assert c.color == "red"
    print(c.describe())

    # Test Rectangle
    r = Rectangle(4, 6, "blue")
    assert r.area() == 24
    assert r.perimeter() == 20
    assert r.is_square is False
    print(r.describe())

    # Test Square
    s = Square(5, "green")
    assert s.area() == 25
    assert s.perimeter() == 20
    assert s.is_square is True
    assert isinstance(s, Rectangle)
    assert isinstance(s, Shape)
    print(s.describe())

    # Polymorphism
    shapes = [Circle(3), Rectangle(4, 5), Square(6)]
    total = sum(s.area() for s in shapes)
    print(f"\nTotal area: {total:.2f}")

    print("\nAll tests passed!")
