"""
Data Types & Abstraction

Demonstrates the progression from primitive types to Abstract Data Types (ADTs):
1. Primitive types and static vs. dynamic typing (via type hints)
2. Composite types (tuple, list, dict, dataclass)
3. Abstract Data Types: same Stack interface, two different implementations
4. Algebraic data types: Option/Maybe pattern to handle absence explicitly

The core idea: abstraction lets callers depend on WHAT an operation does,
not HOW it is implemented. Swapping the implementation should not break callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, TypeVar


# =============================================================================
# 1. PRIMITIVE TYPES — the building blocks
# =============================================================================

def demonstrate_primitives() -> None:
    """Python is dynamically typed, but type hints add a static-check layer."""

    # Primitives map fairly directly to hardware representations
    an_int: int = 42
    a_float: float = 3.14
    a_bool: bool = True
    a_str: str = "hello"
    nothing: None = None

    # Python allows rebinding across types (dynamic), but hints signal intent:
    # an_int = "not an int"  # mypy/pyright would flag this

    print(f"int:   {an_int} ({type(an_int).__name__})")
    print(f"float: {a_float} ({type(a_float).__name__})")
    print(f"bool:  {a_bool} ({type(a_bool).__name__})")
    print(f"str:   {a_str!r} ({type(a_str).__name__})")
    print(f"None:  {nothing} ({type(nothing).__name__})")


# =============================================================================
# 2. COMPOSITE TYPES — tuples, records (dataclass), collections
# =============================================================================

# A tuple is a product type: Point is both an x AND a y.
Point = tuple[float, float]


@dataclass(frozen=True)
class Rectangle:
    """
    A record (product type) with named fields. `frozen=True` makes it immutable —
    a common default for value-like data that should not change after creation.
    """
    top_left: Point
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height


def demonstrate_composites() -> None:
    p: Point = (1.0, 2.0)
    rect = Rectangle(top_left=p, width=3.0, height=4.0)

    # Tuples are indexable; dataclasses are attribute-accessed
    print(f"Point: x={p[0]}, y={p[1]}")
    print(f"Rectangle: {rect} — area={rect.area()}")

    # Immutability: mutating a frozen dataclass raises at runtime
    try:
        rect.width = 999  # type: ignore[misc]
    except Exception as e:
        print(f"Cannot mutate frozen dataclass: {type(e).__name__}")


# =============================================================================
# 3. ABSTRACT DATA TYPES — interface vs implementation
# =============================================================================
#
# A Stack is defined by its operations (push, pop, peek, len), not by how it
# stores data. Two implementations below satisfy the same Protocol; callers
# written against the Protocol work with either one.

T = TypeVar("T")


class Stack(Protocol[T]):
    """Abstract interface for a LIFO stack."""

    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...
    def peek(self) -> T: ...
    def __len__(self) -> int: ...


class ArrayStack(Generic[T]):
    """Implementation 1: Python list as a dynamic array."""

    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class _Node(Generic[T]):
    value: T
    next: Optional["_Node[T]"] = None


class LinkedStack(Generic[T]):
    """Implementation 2: singly linked list. Same Protocol as ArrayStack."""

    def __init__(self) -> None:
        self._head: Optional[_Node[T]] = None
        self._size: int = 0

    def push(self, item: T) -> None:
        self._head = _Node(item, self._head)
        self._size += 1

    def pop(self) -> T:
        if self._head is None:
            raise IndexError("pop from empty stack")
        value = self._head.value
        self._head = self._head.next
        self._size -= 1
        return value

    def peek(self) -> T:
        if self._head is None:
            raise IndexError("peek from empty stack")
        return self._head.value

    def __len__(self) -> int:
        return self._size


def use_any_stack(stack: Stack[int], values: List[int]) -> List[int]:
    """Client code depends on the Stack Protocol, not a concrete class."""
    for v in values:
        stack.push(v)
    return [stack.pop() for _ in range(len(stack))]


def demonstrate_adt() -> None:
    """Both implementations satisfy the same interface, so output is identical."""
    data = [1, 2, 3]

    array_result = use_any_stack(ArrayStack[int](), data)
    linked_result = use_any_stack(LinkedStack[int](), data)

    print(f"Input order:       {data}")
    print(f"ArrayStack order:  {array_result}")
    print(f"LinkedStack order: {linked_result}")
    assert array_result == linked_result == [3, 2, 1]
    print("Same interface, same behavior — implementation is hidden.")


# =============================================================================
# 4. ALGEBRAIC DATA TYPES — Option/Maybe for explicit absence
# =============================================================================
#
# Sum type: a value is EITHER Some(x) OR None.
# This replaces implicit null with an explicit tag, forcing callers to handle
# the missing case. In Python, `Optional[T]` plays this role.


def find_first_positive(numbers: List[int]) -> Optional[int]:
    """Returns the first positive number, or None if none exists."""
    for n in numbers:
        if n > 0:
            return n
    return None


def demonstrate_option() -> None:
    present = find_first_positive([-3, -1, 5, 2])
    absent = find_first_positive([-3, -1, 0])

    # `match` forces consideration of both cases; no silent null surprises
    for label, result in [("with positive", present), ("no positive", absent)]:
        match result:
            case None:
                print(f"{label}: absent")
            case x:
                print(f"{label}: found {x}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    for title, fn in [
        ("1. PRIMITIVE TYPES", demonstrate_primitives),
        ("2. COMPOSITE TYPES", demonstrate_composites),
        ("3. ABSTRACT DATA TYPES", demonstrate_adt),
        ("4. OPTION / MAYBE", demonstrate_option),
    ]:
        print("=" * 70)
        print(title)
        print("=" * 70)
        fn()
        print()


if __name__ == "__main__":
    main()
