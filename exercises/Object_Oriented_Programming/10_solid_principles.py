"""
Exercise 10: SOLID Principles
Topic: Object-Oriented Programming

Practice applying SOLID principles through design challenges.
"""

from abc import ABC, abstractmethod


# =============================================================================
# Challenge 1: SRP + DIP
# =============================================================================

class Notifier(ABC):
    """Abstract notification interface (for DIP)."""

    @abstractmethod
    def send(self, recipient: str, message: str) -> str:
        pass


class ConsoleNotifier(Notifier):
    """Prints notifications to console.

    send("alice", "hello") -> "Console -> alice: hello"
    """

    # TODO: Implement
    pass


class FileNotifier(Notifier):
    """Stores notifications in an internal list.

    send("alice", "hello") -> "File -> alice: hello"

    Also stores all sent messages in self.messages list.
    """

    # TODO: Implement
    pass


class UserService:
    """User service following SRP and DIP.

    SRP: Only handles user registration logic.
    DIP: Depends on Notifier abstraction, not concrete class.

    Args:
        notifier (Notifier): Injected notification dependency.

    Attributes:
        users (dict): {username: email}

    Methods:
        register(username, email): Register user, send notification,
            return True. Raise ValueError if username already exists.
        get_user(username): Return email or None.
    """

    # TODO: Implement
    pass


# =============================================================================
# Challenge 2: OCP + ISP
# =============================================================================

class Readable(ABC):
    """Interface: read data."""

    @abstractmethod
    def read(self) -> str:
        pass


class Writable(ABC):
    """Interface: write data."""

    @abstractmethod
    def write(self, data: str) -> None:
        pass


class ReadWriteFile:
    """A file that supports both reading and writing.

    Implements both Readable and Writable.

    Methods:
        write(data): Append data to internal storage.
        read(): Return all stored data as single string.
    """

    # TODO: Implement (inherit from Readable and Writable)
    pass


class ReadOnlyFile:
    """A file that supports only reading.

    Implements only Readable (ISP: don't force write on read-only).

    Args:
        content (str): Fixed content.

    Methods:
        read(): Return the content.
    """

    # TODO: Implement (inherit from Readable only)
    pass


if __name__ == "__main__":
    # Test SRP + DIP
    print("=== SRP + DIP: UserService ===")
    file_notifier = FileNotifier()
    service = UserService(file_notifier)

    service.register("alice", "alice@mail.com")
    service.register("bob", "bob@mail.com")

    assert service.get_user("alice") == "alice@mail.com"
    assert service.get_user("unknown") is None
    assert len(file_notifier.messages) == 2

    try:
        service.register("alice", "alice2@mail.com")
        assert False, "Should raise ValueError for duplicate"
    except ValueError:
        pass

    print(f"Users registered: {list(service.users.keys())}")
    print(f"Notifications: {file_notifier.messages}")

    # Test with ConsoleNotifier (DIP: swap implementation)
    console_service = UserService(ConsoleNotifier())
    console_service.register("carol", "carol@mail.com")

    # Test OCP + ISP
    print("\n=== OCP + ISP: Files ===")
    rw = ReadWriteFile()
    rw.write("Hello ")
    rw.write("World")
    assert rw.read() == "Hello World"
    assert isinstance(rw, Readable)
    assert isinstance(rw, Writable)

    ro = ReadOnlyFile("Fixed content")
    assert ro.read() == "Fixed content"
    assert isinstance(ro, Readable)
    assert not isinstance(ro, Writable)  # ISP: no write!

    print(f"ReadWrite: {rw.read()}")
    print(f"ReadOnly: {ro.read()}")

    print("\nAll tests passed!")
