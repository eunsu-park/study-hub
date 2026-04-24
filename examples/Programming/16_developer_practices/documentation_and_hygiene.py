"""
Developer Practices: Documentation, Type Hints, and Hygiene

Demonstrates the practices that separate a working script from a codebase
that survives a team and a year:
1. Comments that explain WHY, not WHAT
2. Docstrings — module, function, and class — in a consistent style
3. Type hints that serve as both documentation and static-check targets
4. Logging instead of scattered print statements
5. A minimal example of managing technical debt deliberately (TODO tags
   that follow a convention) vs. letting it accumulate silently

The exported `Account` class uses all five practices so you can see them
compose in a realistic small API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

# Module-level logger. Libraries should get a named logger and NOT configure
# handlers — that is the application's responsibility. This makes the library
# polite: the caller chooses where logs go (stdout, file, syslog, ...).
log = logging.getLogger(__name__)


# =============================================================================
# 1. COMMENTS — WHY, not WHAT
# =============================================================================

def bad_comment_example(balance: float, withdrawal: float) -> float:
    # Subtract withdrawal from balance
    return balance - withdrawal
    # ^ This comment restates the code. A reader can see the subtraction.


def good_comment_example(balance: float, withdrawal: float) -> float:
    # Accounts may briefly go negative during pending transactions; the
    # daily reconciliation job enforces the non-negative invariant.
    # Do NOT add a guard here or the reconciler's retry logic breaks.
    return balance - withdrawal
    # ^ This comment explains a non-obvious invariant and warns the next reader.


# =============================================================================
# 2. DOCSTRINGS — module, function, class
# =============================================================================
#
# The module docstring is at the top of this file. Function and class
# docstrings follow the Google style: a one-line summary, a blank line,
# then Args/Returns/Raises sections as needed.


def transfer(from_balance: float, to_balance: float, amount: float) -> tuple[float, float]:
    """Move `amount` from one balance to another.

    Both sides are updated atomically from the caller's perspective — this
    function is pure and returns the new balances. Persisting the result is
    the caller's job.

    Args:
        from_balance: Source balance in account currency.
        to_balance: Destination balance in account currency.
        amount: Positive amount to transfer.

    Returns:
        The new (from_balance, to_balance) after the transfer.

    Raises:
        ValueError: If `amount` is non-positive or exceeds `from_balance`.
    """
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")
    if amount > from_balance:
        raise ValueError(f"insufficient funds: {amount} > {from_balance}")
    return from_balance - amount, to_balance + amount


# =============================================================================
# 3. TYPE HINTS — signatures as documentation
# =============================================================================
#
# Type hints give three benefits: (a) they document intent at the signature,
# (b) they let static checkers (mypy, pyright) catch mistakes before runtime,
# (c) IDE completions improve. They are optional but almost always worth it
# in code meant to be read by someone other than its author.


@dataclass
class Account:
    """A bank account with a balance and an optional daily limit.

    This class exists primarily to demonstrate good documentation, type
    hints, and logging in combination — not as a realistic banking model.
    """

    owner: str
    balance: float = 0.0
    daily_limit: Optional[float] = None
    _withdrawn_today: float = field(default=0.0, repr=False)

    def deposit(self, amount: float) -> None:
        """Add `amount` to the balance.

        Args:
            amount: Positive amount to deposit.

        Raises:
            ValueError: If `amount` is non-positive.
        """
        if amount <= 0:
            raise ValueError(f"deposit amount must be positive, got {amount}")
        self.balance += amount
        log.info("deposit owner=%s amount=%.2f balance=%.2f", self.owner, amount, self.balance)

    def withdraw(self, amount: float) -> float:
        """Subtract `amount` from the balance, respecting the daily limit.

        Args:
            amount: Positive amount to withdraw.

        Returns:
            The new balance after the withdrawal.

        Raises:
            ValueError: If `amount` is non-positive or exceeds the balance
                or daily limit.
        """
        if amount <= 0:
            raise ValueError(f"withdraw amount must be positive, got {amount}")
        if amount > self.balance:
            raise ValueError(f"insufficient funds: {amount} > {self.balance}")
        if self.daily_limit is not None and self._withdrawn_today + amount > self.daily_limit:
            raise ValueError(
                f"daily limit {self.daily_limit} exceeded "
                f"(already withdrew {self._withdrawn_today})"
            )

        self.balance -= amount
        self._withdrawn_today += amount
        log.info("withdraw owner=%s amount=%.2f balance=%.2f", self.owner, amount, self.balance)
        return self.balance


# =============================================================================
# 4. LOGGING — structured, leveled, configurable
# =============================================================================
#
# Print statements are fine for one-off scripts, but in library / production
# code, prefer logging: it has levels (DEBUG/INFO/WARNING/ERROR), can be
# redirected, and callers can filter by module. This function configures a
# basic handler so the demo produces visible output.


def _configure_demo_logging() -> None:
    """Attach a simple stderr handler at INFO level for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="  [%(levelname)s %(name)s] %(message)s",
    )


# =============================================================================
# 5. TECHNICAL DEBT — make it visible
# =============================================================================
#
# `TODO` / `FIXME` / `XXX` tags should follow a convention so a grep reveals
# the whole backlog. A good convention:
#   TODO(owner, context): short description, link to ticket if one exists.
# Untagged "quick fix"s accumulate invisibly and are the worst kind of debt.


def summarize_account(account: Account) -> str:
    # TODO(eunsu, 2026-Q2): add multi-currency support once the FX rate
    # service is production-ready. Tracking: STUDY-147.
    return f"{account.owner}: balance={account.balance:.2f}"


# =============================================================================
# MAIN — run all demos
# =============================================================================

def main() -> None:
    _configure_demo_logging()

    print("=" * 70)
    print("1. COMMENTS (WHY, not WHAT)")
    print("=" * 70)
    print(f"  bad_comment_example(100, 30)  = {bad_comment_example(100, 30)}")
    print(f"  good_comment_example(100, 30) = {good_comment_example(100, 30)}")
    print("  (see source for the contrast in comment style)\n")

    print("=" * 70)
    print("2. DOCSTRINGS + pure functions")
    print("=" * 70)
    a, b = transfer(from_balance=100, to_balance=10, amount=25)
    print(f"  transfer(100, 10, 25) -> from={a}, to={b}")
    try:
        transfer(from_balance=100, to_balance=10, amount=200)
    except ValueError as e:
        print(f"  transfer(100, 10, 200) raised ValueError: {e}\n")

    print("=" * 70)
    print("3. TYPE HINTS + DATACLASS (Account)")
    print("=" * 70)
    account = Account(owner="alice", balance=200.0, daily_limit=150.0)
    account.deposit(50)
    account.withdraw(100)
    try:
        account.withdraw(80)  # would exceed daily_limit
    except ValueError as e:
        print(f"  second withdraw refused: {e}")
    print(f"  final: {summarize_account(account)}\n")

    print("=" * 70)
    print("4. LOGGING")
    print("=" * 70)
    print("  (deposit/withdraw above emitted log.info lines; see [INFO ...] output)\n")

    print("=" * 70)
    print("5. TECHNICAL DEBT TAGS (grep 'TODO(' to see them all)")
    print("=" * 70)
    print("  summarize_account has a TODO tagged with owner, quarter, and ticket.")
    print("  Convention: TODO(owner, context): description. Untagged fixes accumulate silently.")


if __name__ == "__main__":
    main()
