"""
Exercise 03: Constructors and Initialization
Topic: Object-Oriented Programming

Practice __init__ patterns: validation, defaults, alternative constructors.
"""


class Email:
    """An email message with validation.

    Attributes:
        sender (str): Sender email address.
        recipient (str): Recipient email address.
        subject (str): Email subject (default "No Subject").
        body (str): Email body (default "").

    Validation in __init__:
        - sender and recipient must contain "@"
        - subject must be <= 100 characters
        Raise ValueError with descriptive message on failure.

    Class Methods:
        from_dict(data): Create from dict with keys 'sender', 'recipient',
                         'subject' (optional), 'body' (optional).

    Methods:
        __repr__(): Return "Email(sender -> recipient: subject)"
    """

    # TODO: Implement this class
    pass


class Matrix:
    """A simple 2D matrix.

    Constructor:
        __init__(rows): Takes a list of lists. Validates that all rows
                        have the same length. Raise ValueError if not.

    Class Methods:
        zeros(rows, cols): Create a matrix of zeros.
        identity(n): Create an n x n identity matrix.

    Properties:
        shape: Return (num_rows, num_cols) tuple.

    Methods:
        get(row, col): Return element at position.
        set(row, col, value): Set element at position.
        __repr__(): Return "Matrix(rows x cols)"
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test Email
    e1 = Email("alice@mail.com", "bob@mail.com", "Hello", "Hi Bob!")
    print(e1)

    e2 = Email("alice@mail.com", "bob@mail.com")
    assert e2.subject == "No Subject"
    assert e2.body == ""

    e3 = Email.from_dict({
        "sender": "a@b.com",
        "recipient": "c@d.com",
        "subject": "Test"
    })
    assert e3.subject == "Test"

    try:
        Email("invalid", "bob@mail.com")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    try:
        Email("a@b.com", "c@d.com", "x" * 101)
        assert False, "Should raise ValueError for long subject"
    except ValueError:
        pass

    # Test Matrix
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape == (2, 3)
    assert m.get(0, 1) == 2
    m.set(1, 2, 99)
    assert m.get(1, 2) == 99

    z = Matrix.zeros(3, 4)
    assert z.shape == (3, 4)
    assert z.get(0, 0) == 0

    eye = Matrix.identity(3)
    assert eye.get(0, 0) == 1
    assert eye.get(0, 1) == 0
    assert eye.get(2, 2) == 1

    try:
        Matrix([[1, 2], [3]])
        assert False, "Should raise ValueError for unequal rows"
    except ValueError:
        pass

    print(f"Matrix: {m}")
    print(f"Zeros: {z}")
    print(f"Identity: {eye}")

    print("\nAll tests passed!")
