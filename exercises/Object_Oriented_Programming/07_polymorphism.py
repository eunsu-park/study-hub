"""
Exercise 07: Polymorphism
Topic: Object-Oriented Programming

Practice duck typing, operator overloading, and polymorphic functions.
"""


class Fraction:
    """A fraction (rational number) with operator overloading.

    Attributes:
        numerator (int): Top number.
        denominator (int): Bottom number (never 0, always positive).

    The fraction should be stored in reduced form (e.g., 4/6 -> 2/3).
    Denominator should always be positive (e.g., 1/-2 -> -1/2).

    Operators:
        + : Add two fractions -> Fraction
        - : Subtract two fractions -> Fraction
        * : Multiply two fractions -> Fraction
        == : True if same value
        < : Compare values
        float() : Return decimal value
        __repr__: "Fraction(num, den)"
        __str__: "num/den" (or just "num" if den is 1)

    Hint: Use math.gcd for reducing.
    """

    # TODO: Implement this class
    pass


class FileExporter:
    """Export data to different formats using duck typing.

    Implement three concrete exporters (no base class needed — duck typing!):
    - CSVExporter: export(data) returns comma-separated values
    - JSONExporter: export(data) returns JSON string
    - MarkdownExporter: export(data) returns markdown table

    Then implement export_data(exporter, data) that works with any of them.

    data is a list of dicts, e.g.:
    [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]
    """
    pass


class CSVExporter:
    """Export data as CSV.

    export(data) should return:
        "name,score\nAlice,95\nBob,87"
    """

    # TODO: Implement export(data)
    pass


class JSONExporter:
    """Export data as JSON string."""

    # TODO: Implement export(data)
    pass


class MarkdownExporter:
    """Export data as markdown table.

    export(data) should return:
        "| name | score |\n| --- | --- |\n| Alice | 95 |\n| Bob | 87 |"
    """

    # TODO: Implement export(data)
    pass


def export_data(exporter, data):
    """Polymorphic function — works with any exporter that has export()."""
    # TODO: Implement
    pass


if __name__ == "__main__":
    # Test Fraction
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 3)

    assert str(f1 + f2) == "5/6"
    assert str(f1 - f2) == "1/6"
    assert str(f1 * f2) == "1/6"
    assert Fraction(2, 4) == Fraction(1, 2)  # Auto-reduced
    assert f1 > f2
    assert float(f1) == 0.5
    assert str(Fraction(6, 3)) == "2"  # Whole number

    # Negative fractions
    assert str(Fraction(-1, 2)) == "-1/2"
    assert str(Fraction(1, -2)) == "-1/2"  # Denominator normalized

    print(f"1/2 + 1/3 = {f1 + f2}")
    print(f"1/2 - 1/3 = {f1 - f2}")
    print(f"1/2 * 1/3 = {f1 * f2}")
    print(f"float(1/2) = {float(f1)}")

    # Test Exporters (duck typing)
    data = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]

    csv_out = export_data(CSVExporter(), data)
    assert "Alice" in csv_out
    assert "," in csv_out
    print(f"\nCSV:\n{csv_out}")

    json_out = export_data(JSONExporter(), data)
    assert "Alice" in json_out
    print(f"\nJSON:\n{json_out}")

    md_out = export_data(MarkdownExporter(), data)
    assert "|" in md_out
    print(f"\nMarkdown:\n{md_out}")

    print("\nAll tests passed!")
