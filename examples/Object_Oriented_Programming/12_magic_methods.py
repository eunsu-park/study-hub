"""
Example 12: Magic Methods
Topic: Object-Oriented Programming

Demonstrates __str__, __repr__, __eq__, __hash__, __iter__,
__getitem__, __len__, __call__, __enter__/__exit__.
"""

import time
from functools import total_ordering


# =============================================================================
# MONEY CLASS (repr, str, eq, hash, comparison, arithmetic)
# =============================================================================

@total_ordering
class Money:
    """Money with currency, full magic method support."""

    def __init__(self, amount, currency="USD"):
        self.amount = round(amount, 2)
        self.currency = currency

    def __repr__(self):
        return f"Money({self.amount}, {self.currency!r})"

    def __str__(self):
        symbols = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5"}
        sym = symbols.get(self.currency, self.currency + " ")
        return f"{sym}{self.amount:,.2f}"

    def _check_currency(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError(f"Cannot mix {self.currency} and {other.currency}")

    def __add__(self, other):
        self._check_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other):
        self._check_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return Money(self.amount * factor, self.currency)
        return NotImplemented

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __eq__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __lt__(self, other):
        self._check_currency(other)
        return self.amount < other.amount

    def __hash__(self):
        return hash((self.amount, self.currency))

    def __bool__(self):
        return self.amount != 0


# =============================================================================
# ITERABLE: Deck of Cards
# =============================================================================

class Card:
    SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit[0]}"


class Deck:
    """Iterable deck of cards."""

    def __init__(self):
        self.cards = [Card(r, s) for s in Card.SUITS for r in Card.RANKS]

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

    def __contains__(self, card):
        return any(c.rank == card.rank and c.suit == card.suit for c in self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __repr__(self):
        return f"Deck({len(self)} cards)"


# =============================================================================
# CALLABLE: Memoizer
# =============================================================================

class Memoize:
    """Callable that caches function results."""

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

    def __repr__(self):
        return f"Memoize({self.func.__name__}, {len(self.cache)} cached)"


# =============================================================================
# CONTEXT MANAGER: Timer
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label="Block"):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        print(f"  {self.label}: {self.elapsed:.4f}s")
        return False


if __name__ == "__main__":
    # Money
    print("=== Money ===")
    price = Money(29.99)
    tax = price * 0.08
    total = price + tax
    print(f"Price: {price}")
    print(f"Tax: {tax}")
    print(f"Total: {total}")
    print(f"repr: {repr(total)}")
    print(f"$10 == $10? {Money(10) == Money(10)}")
    print(f"$10 < $20? {Money(10) < Money(20)}")

    # Deck
    print("\n=== Deck of Cards ===")
    deck = Deck()
    print(f"{deck}")
    print(f"First 5: {deck[:5]}")
    print(f"Last card: {deck[-1]}")
    print(f"AH in deck? {Card('A', 'Hearts') in deck}")

    # Callable
    print("\n=== Memoize (__call__) ===")

    @Memoize
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    print(f"fib(10) = {fibonacci(10)}")
    print(f"fib(20) = {fibonacci(20)}")
    print(fibonacci)

    # Context manager
    print("\n=== Timer (Context Manager) ===")
    with Timer("Sum of range"):
        total = sum(range(1_000_000))
    print(f"  Result: {total:,}")
