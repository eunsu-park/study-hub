"""
Exercise 06: Multiple Inheritance
Topic: Object-Oriented Programming

Practice mixins and MRO understanding.
"""

import json


class TimestampMixin:
    """Mixin that adds a created_at timestamp.

    When mixed in, the object should have a `created_at` attribute
    set to the current ISO format datetime string during initialization.

    Methods:
        age_seconds(): Return seconds elapsed since creation.

    Hint: Use datetime.now().isoformat() for created_at.
          Use (datetime.now() - parse(created_at)).total_seconds() for age.
    """

    # TODO: Implement this mixin
    pass


class SerializableMixin:
    """Mixin that adds JSON serialization.

    Methods:
        to_json(): Return JSON string of all public attributes (no _ prefix).
        to_dict(): Return dict of all public attributes.

    Use json.dumps with default=str for non-serializable types.
    """

    # TODO: Implement this mixin
    pass


class ComparableMixin:
    """Mixin that adds comparison based on a _compare_key() method.

    Subclasses must implement _compare_key() returning a comparable value.

    Provides: __eq__, __lt__, __le__, __gt__, __ge__
    """

    # TODO: Implement this mixin
    pass


class Product(SerializableMixin, ComparableMixin):
    """Product combining serialization and comparison mixins.

    Attributes:
        name (str): Product name.
        price (float): Product price.

    Comparison should be by price.
    """

    # TODO: Implement this class using the mixins
    pass


if __name__ == "__main__":
    # Test SerializableMixin + ComparableMixin via Product
    laptop = Product("Laptop", 999.99)
    phone = Product("Phone", 699.99)
    tablet = Product("Tablet", 999.99)

    # Serialization
    data = laptop.to_dict()
    assert data["name"] == "Laptop"
    assert data["price"] == 999.99
    json_str = laptop.to_json()
    assert "Laptop" in json_str
    print(f"JSON: {json_str}")

    # Comparison
    assert laptop > phone
    assert phone < laptop
    assert laptop == tablet  # Same price
    assert laptop >= tablet
    assert phone <= laptop

    # Sorting
    products = [laptop, phone, tablet]
    sorted_products = sorted(products)
    assert sorted_products[0].name == "Phone"
    print(f"Sorted: {[p.name for p in sorted_products]}")

    # MRO
    print(f"\nProduct MRO: {[c.__name__ for c in Product.__mro__]}")

    print("\nAll tests passed!")
