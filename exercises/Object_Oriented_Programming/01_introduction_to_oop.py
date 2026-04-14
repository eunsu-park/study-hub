"""
Exercise 01: Introduction to OOP
Topic: Object-Oriented Programming

Implement basic classes to practice OOP fundamentals.
"""


class Counter:
    """A simple counter.

    Attributes:
        name (str): Counter name.
        count (int): Current count (starts at 0).

    Methods:
        increment(): Add 1 to count.
        decrement(): Subtract 1 from count. Don't go below 0.
        reset(): Reset count to 0.
        __repr__(): Return "Counter(name, count=N)"
    """

    # TODO: Implement __init__, increment, decrement, reset, __repr__
    pass


class LightSwitch:
    """A light switch that can be on or off.

    Attributes:
        location (str): Where the switch is.
        is_on (bool): Whether the light is on (starts False).
        toggle_count (int): Number of times toggled (starts 0).

    Methods:
        toggle(): Switch on/off and increment toggle_count.
        status(): Return "on" or "off".
        __repr__(): Return "LightSwitch(location, status)"
    """

    # TODO: Implement this class
    pass


class ShoppingItem:
    """An item in a shopping list.

    Attributes:
        name (str): Item name.
        quantity (int): How many to buy (default 1).
        purchased (bool): Whether it's been purchased (default False).

    Methods:
        buy(): Mark as purchased.
        __repr__(): Return "ShoppingItem(name, qty=N, purchased=True/False)"
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test Counter
    c = Counter("visitors")
    c.increment()
    c.increment()
    c.increment()
    c.decrement()
    assert c.count == 2, f"Expected 2, got {c.count}"
    c.reset()
    assert c.count == 0
    c.decrement()  # Should not go below 0
    assert c.count == 0
    print(f"Counter: {c}")

    # Test LightSwitch
    sw = LightSwitch("Kitchen")
    assert sw.status() == "off"
    sw.toggle()
    assert sw.status() == "on"
    sw.toggle()
    assert sw.status() == "off"
    assert sw.toggle_count == 2
    print(f"Switch: {sw}")

    # Test ShoppingItem
    milk = ShoppingItem("Milk", 2)
    assert not milk.purchased
    milk.buy()
    assert milk.purchased
    print(f"Item: {milk}")

    print("\nAll tests passed!")
