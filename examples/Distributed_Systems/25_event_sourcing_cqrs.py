"""
Event Sourcing and CQRS Pattern

Implements event sourcing with an append-only event store, aggregate
reconstruction, and CQRS (Command Query Responsibility Segregation)
with separate write and read models. Demonstrates snapshotting for
performance and projections for materialised views.

Key concepts:
- Event sourcing: store events, not state; derive state by replaying
- CQRS: separate models for writes (commands) and reads (queries)
- Event store: append-only, immutable log of domain events
- Projections: materialised views built from events
- Snapshots: periodic state checkpoints to speed up replay

Usage:
    python 25_event_sourcing_cqrs.py
"""

from __future__ import annotations

import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """An immutable domain event."""
    event_type: str
    aggregate_id: str
    data: dict[str, Any]
    version: int = 0
    timestamp: float = 0.0

    def __repr__(self) -> str:
        return f"Event({self.event_type}, agg={self.aggregate_id}, v={self.version})"


# ---------------------------------------------------------------------------
# Event Store
# ---------------------------------------------------------------------------

class EventStore:
    """Append-only event store."""

    def __init__(self):
        self._events: list[Event] = []
        self._by_aggregate: dict[str, list[Event]] = defaultdict(list)
        self._version_counter: dict[str, int] = defaultdict(int)

    def append(self, event: Event) -> Event:
        """Append an event. Assigns version and timestamp."""
        self._version_counter[event.aggregate_id] += 1
        event.version = self._version_counter[event.aggregate_id]
        event.timestamp = len(self._events)  # Simulated timestamp
        self._events.append(event)
        self._by_aggregate[event.aggregate_id].append(event)
        return event

    def get_events(self, aggregate_id: str,
                   after_version: int = 0) -> list[Event]:
        """Get events for an aggregate after a given version."""
        return [e for e in self._by_aggregate.get(aggregate_id, [])
                if e.version > after_version]

    def get_all_events(self, after: int = 0) -> list[Event]:
        """Get all events in the store after a position."""
        return self._events[after:]

    @property
    def count(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Aggregate: Bank Account
# ---------------------------------------------------------------------------

@dataclass
class BankAccountState:
    """Current state of a bank account (derived from events)."""
    account_id: str = ""
    owner: str = ""
    balance: float = 0.0
    is_open: bool = False
    version: int = 0


class BankAccountAggregate:
    """
    Bank account aggregate with event sourcing.
    All state changes produce events; state is reconstructed from events.
    """

    def __init__(self, account_id: str, store: EventStore):
        self.account_id = account_id
        self.store = store
        self.state = BankAccountState(account_id=account_id)
        self._pending_events: list[Event] = []

    def load(self, snapshot: BankAccountState | None = None) -> None:
        """Reconstruct state from events (optionally starting from snapshot)."""
        if snapshot:
            self.state = snapshot
            after_version = snapshot.version
        else:
            self.state = BankAccountState(account_id=self.account_id)
            after_version = 0

        events = self.store.get_events(self.account_id, after_version)
        for event in events:
            self._apply(event)

    def open_account(self, owner: str) -> Event:
        """Command: open a new account."""
        if self.state.is_open:
            raise ValueError("Account already open")
        event = Event("AccountOpened", self.account_id, {"owner": owner})
        return self._record(event)

    def deposit(self, amount: float) -> Event:
        if not self.state.is_open:
            raise ValueError("Account not open")
        if amount <= 0:
            raise ValueError("Amount must be positive")
        event = Event("MoneyDeposited", self.account_id, {"amount": amount})
        return self._record(event)

    def withdraw(self, amount: float) -> Event:
        if not self.state.is_open:
            raise ValueError("Account not open")
        if amount > self.state.balance:
            raise ValueError(f"Insufficient funds: {self.state.balance} < {amount}")
        event = Event("MoneyWithdrawn", self.account_id, {"amount": amount})
        return self._record(event)

    def close_account(self) -> Event:
        if not self.state.is_open:
            raise ValueError("Account not open")
        if self.state.balance != 0:
            raise ValueError("Balance must be zero to close")
        event = Event("AccountClosed", self.account_id, {})
        return self._record(event)

    def _record(self, event: Event) -> Event:
        """Record event to store and apply to state."""
        stored = self.store.append(event)
        self._apply(stored)
        return stored

    def _apply(self, event: Event) -> None:
        """Apply an event to the current state."""
        if event.event_type == "AccountOpened":
            self.state.owner = event.data["owner"]
            self.state.is_open = True
        elif event.event_type == "MoneyDeposited":
            self.state.balance += event.data["amount"]
        elif event.event_type == "MoneyWithdrawn":
            self.state.balance -= event.data["amount"]
        elif event.event_type == "AccountClosed":
            self.state.is_open = False
        self.state.version = event.version

    def take_snapshot(self) -> BankAccountState:
        """Create a snapshot of current state."""
        return BankAccountState(
            account_id=self.state.account_id,
            owner=self.state.owner,
            balance=self.state.balance,
            is_open=self.state.is_open,
            version=self.state.version,
        )


# ---------------------------------------------------------------------------
# CQRS Read Model (Projections)
# ---------------------------------------------------------------------------

class AccountSummaryProjection:
    """Read model: account summaries materialised from events."""

    def __init__(self):
        self.summaries: dict[str, dict] = {}
        self._processed_count = 0

    def process(self, event: Event) -> None:
        """Project an event into the read model."""
        aid = event.aggregate_id

        if event.event_type == "AccountOpened":
            self.summaries[aid] = {
                "account_id": aid,
                "owner": event.data["owner"],
                "balance": 0.0,
                "status": "open",
                "transaction_count": 0,
            }
        elif event.event_type == "MoneyDeposited":
            if aid in self.summaries:
                self.summaries[aid]["balance"] += event.data["amount"]
                self.summaries[aid]["transaction_count"] += 1
        elif event.event_type == "MoneyWithdrawn":
            if aid in self.summaries:
                self.summaries[aid]["balance"] -= event.data["amount"]
                self.summaries[aid]["transaction_count"] += 1
        elif event.event_type == "AccountClosed":
            if aid in self.summaries:
                self.summaries[aid]["status"] = "closed"

        self._processed_count += 1

    def catch_up(self, store: EventStore) -> int:
        """Process all new events from the store."""
        events = store.get_all_events(after=self._processed_count)
        for event in events:
            self.process(event)
        return len(events)


class TransactionLogProjection:
    """Read model: transaction log per account."""

    def __init__(self):
        self.logs: dict[str, list[dict]] = defaultdict(list)
        self._processed_count = 0

    def process(self, event: Event) -> None:
        aid = event.aggregate_id
        if event.event_type in ("MoneyDeposited", "MoneyWithdrawn"):
            self.logs[aid].append({
                "type": event.event_type,
                "amount": event.data["amount"],
                "version": event.version,
                "timestamp": event.timestamp,
            })
        self._processed_count += 1

    def catch_up(self, store: EventStore) -> int:
        events = store.get_all_events(after=self._processed_count)
        for event in events:
            self.process(event)
        return len(events)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_event_sourcing() -> None:
    """Demonstrate basic event sourcing."""
    print("=" * 70)
    print("Event Sourcing: Bank Account")
    print("=" * 70)

    store = EventStore()
    account = BankAccountAggregate("ACC-001", store)

    # Execute commands
    account.open_account("Alice")
    account.deposit(1000.0)
    account.deposit(500.0)
    account.withdraw(200.0)
    account.withdraw(50.0)

    print(f"\n  Current state: balance=${account.state.balance:.2f}, "
          f"owner={account.state.owner}")
    print(f"  Total events: {store.count}")

    # Show event log
    print(f"\n  Event log:")
    for event in store.get_events("ACC-001"):
        print(f"    v{event.version}: {event.event_type} {event.data}")

    # Reconstruct from scratch
    print(f"\n  Reconstructing state from events:")
    account2 = BankAccountAggregate("ACC-001", store)
    account2.load()
    print(f"    Replayed {account2.state.version} events")
    print(f"    Balance: ${account2.state.balance:.2f}")
    print(f"    Match: {account.state.balance == account2.state.balance}")


def demo_snapshots() -> None:
    """Demonstrate snapshotting for faster replay."""
    print("\n" + "=" * 70)
    print("Snapshots: Faster State Reconstruction")
    print("=" * 70)

    store = EventStore()
    account = BankAccountAggregate("ACC-002", store)

    account.open_account("Bob")
    for i in range(20):
        account.deposit(100.0)

    # Take snapshot at version 21
    snapshot = account.take_snapshot()
    print(f"\n  After 20 deposits: balance=${account.state.balance:.2f}")
    print(f"  Snapshot at version {snapshot.version}: "
          f"balance=${snapshot.balance:.2f}")

    # More events after snapshot
    for i in range(5):
        account.withdraw(50.0)

    print(f"  After 5 withdrawals: balance=${account.state.balance:.2f}")

    # Reconstruct from snapshot (only 5 events to replay, not 26)
    account3 = BankAccountAggregate("ACC-002", store)
    account3.load(snapshot=snapshot)
    events_replayed = account3.state.version - snapshot.version

    print(f"\n  Reconstruction from snapshot:")
    print(f"    Without snapshot: replay {store.count} events")
    print(f"    With snapshot: replay {events_replayed} events (from v{snapshot.version})")
    print(f"    Balance: ${account3.state.balance:.2f}")


def demo_cqrs() -> None:
    """Demonstrate CQRS with separate read/write models."""
    print("\n" + "=" * 70)
    print("CQRS: Separate Read and Write Models")
    print("=" * 70)

    store = EventStore()

    # Write model: execute commands
    acc1 = BankAccountAggregate("ACC-A", store)
    acc2 = BankAccountAggregate("ACC-B", store)

    acc1.open_account("Alice")
    acc1.deposit(5000.0)
    acc1.withdraw(1000.0)

    acc2.open_account("Bob")
    acc2.deposit(3000.0)
    acc2.deposit(500.0)

    # Read models: projections built from events
    summary_proj = AccountSummaryProjection()
    txn_proj = TransactionLogProjection()

    processed = summary_proj.catch_up(store)
    txn_proj.catch_up(store)

    print(f"\n  Write model processed {store.count} events")
    print(f"  Read model caught up with {processed} events")

    print(f"\n  Account Summary Projection (read model):")
    for aid, summary in summary_proj.summaries.items():
        print(f"    {aid}: owner={summary['owner']}, "
              f"balance=${summary['balance']:.2f}, "
              f"txns={summary['transaction_count']}, "
              f"status={summary['status']}")

    print(f"\n  Transaction Log Projection (read model):")
    for aid, txns in txn_proj.logs.items():
        print(f"    {aid}:")
        for txn in txns:
            print(f"      {txn['type']}: ${txn['amount']:.2f}")

    # Add more events — projections need to catch up
    acc1.withdraw(500.0)
    new = summary_proj.catch_up(store)
    print(f"\n  After new withdrawal: caught up {new} new event(s)")
    print(f"    ACC-A balance: ${summary_proj.summaries['ACC-A']['balance']:.2f}")


def demo_comparison() -> None:
    """Compare event sourcing patterns."""
    print("\n" + "=" * 70)
    print("Event Sourcing / CQRS Patterns Comparison")
    print("=" * 70)

    print("""
  ┌─────────────────────┬──────────────────────────────────────────┐
  │ Pattern             │ Description                              │
  ├─────────────────────┼──────────────────────────────────────────┤
  │ Event Store         │ Append-only log, immutable events        │
  │ Aggregate           │ Consistency boundary, apply events       │
  │ Projection          │ Materialised read view from events       │
  │ Snapshot            │ Checkpoint to avoid full replay          │
  │ CQRS                │ Separate write model and read model      │
  │ Event Replay        │ Rebuild state by replaying all events    │
  │ Temporal Query      │ "What was state at time T?" via replay   │
  └─────────────────────┴──────────────────────────────────────────┘

  Benefits:
  - Complete audit trail (every state change recorded)
  - Temporal queries (reconstruct state at any point in time)
  - Event replay for bug investigation and new projections
  - Decoupled read/write scaling (CQRS)

  Challenges:
  - Event schema evolution (versioning events)
  - Eventual consistency between write and read models
  - Complexity of managing projections
  - Storage growth (mitigated by snapshots)
""")


if __name__ == "__main__":
    demo_event_sourcing()
    demo_snapshots()
    demo_cqrs()
    demo_comparison()
    print("Done.")
