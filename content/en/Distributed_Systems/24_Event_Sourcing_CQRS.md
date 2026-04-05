# Lesson 24: Event Sourcing and CQRS

[Overview](./00_Overview.md) | [Previous: Distributed Rate Limiting](./23_Distributed_Rate_Limiting.md) | [Next: Vector Clocks](./25_Vector_Clocks.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design event-sourced systems where state is derived from an append-only log of domain events
2. Implement the CQRS pattern to separate read and write models for scalability
3. Build event stores with snapshotting and projection rebuilding
4. Handle eventual consistency between command and query sides
5. Analyze trade-offs of event sourcing including replay cost, schema evolution, and debugging

---

## Table of Contents

1. [Event Sourcing Fundamentals](#1-event-sourcing-fundamentals)
2. [Event Store Implementation](#2-event-store-implementation)
3. [Aggregates and Commands](#3-aggregates-and-commands)
4. [Projections and Read Models](#4-projections-and-read-models)
5. [CQRS Architecture](#5-cqrs-architecture)
6. [Snapshots and Performance](#6-snapshots-and-performance)
7. [Event Schema Evolution](#7-event-schema-evolution)
8. [Distributed Event Sourcing](#8-distributed-event-sourcing)
9. [Real-World Systems](#9-real-world-systems)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Event Sourcing Fundamentals

### 1.1 Traditional CRUD vs Event Sourcing

```
CRUD:                          Event Sourcing:
┌─────────────┐               ┌─────────────────────┐
│ Account     │               │ Event Log           │
│ balance: 150│               │ 1. AccountCreated   │
│ name: Alice │               │ 2. Deposited(200)   │
└─────────────┘               │ 3. Withdrawn(50)    │
                              │ 4. NameChanged(Alice)│
State = latest snapshot       └─────────────────────┘
Lost: how did we get here?    State = replay(events)
                              Full history preserved
```

### 1.2 Core Concepts

```python
import time
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod


@dataclass
class Event:
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    aggregate_id: str = ""
    aggregate_type: str = ""
    version: int = 0
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "version": self.version,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
        }
```

---

## 2. Event Store Implementation

### 2.1 Append-Only Event Store

```python
class EventStore:
    """
    Append-only event store with optimistic concurrency control.

    Events are stored in per-aggregate streams. Each append
    specifies the expected version to prevent concurrent
    modifications from creating conflicts.
    """

    def __init__(self):
        self.streams: Dict[str, list[Event]] = defaultdict(list)
        self.global_log: list[Event] = []
        self.global_position: int = 0
        self.subscribers: list[Callable] = []

    def append(self, aggregate_id: str, events: list[Event],
               expected_version: int) -> int:
        """
        Append events to an aggregate stream.

        Args:
            aggregate_id: The aggregate to append to
            events: List of events to append
            expected_version: Expected current version (optimistic concurrency)

        Returns:
            New version number after append

        Raises:
            ConcurrencyError if expected_version doesn't match
        """
        stream = self.streams[aggregate_id]
        current_version = len(stream)

        if current_version != expected_version:
            raise ConcurrencyError(
                f"Expected version {expected_version}, "
                f"but current is {current_version}"
            )

        for i, event in enumerate(events):
            event.version = current_version + i + 1
            event.aggregate_id = aggregate_id
            stream.append(event)
            self.global_log.append(event)
            self.global_position += 1

            # Notify subscribers
            for subscriber in self.subscribers:
                subscriber(event)

        return current_version + len(events)

    def read_stream(self, aggregate_id: str,
                    from_version: int = 0) -> list[Event]:
        """Read events from an aggregate stream."""
        stream = self.streams.get(aggregate_id, [])
        return [e for e in stream if e.version > from_version]

    def read_all(self, from_position: int = 0) -> list[Event]:
        """Read from the global event log."""
        return self.global_log[from_position:]

    def subscribe(self, callback: Callable):
        """Subscribe to new events."""
        self.subscribers.append(callback)

    def stream_version(self, aggregate_id: str) -> int:
        """Get the current version of a stream."""
        return len(self.streams.get(aggregate_id, []))


class ConcurrencyError(Exception):
    pass
```

---

## 3. Aggregates and Commands

### 3.1 Aggregate Pattern

```python
class Aggregate(ABC):
    """
    Base class for event-sourced aggregates.

    An aggregate:
    1. Receives commands
    2. Validates business rules
    3. Emits events (if valid)
    4. Applies events to update state
    """

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version: int = 0
        self.uncommitted_events: list[Event] = []

    def load(self, events: list[Event]):
        """Rebuild state from a list of events."""
        for event in events:
            self._apply(event)
            self.version = event.version

    def _emit(self, event_type: str, data: dict):
        """Emit a new event."""
        event = Event(
            event_type=event_type,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.__class__.__name__,
            data=data,
        )
        self._apply(event)
        self.uncommitted_events.append(event)

    @abstractmethod
    def _apply(self, event: Event):
        """Apply an event to update state. Must be pure."""
        pass

    def get_uncommitted_events(self) -> list[Event]:
        """Get and clear uncommitted events."""
        events = self.uncommitted_events
        self.uncommitted_events = []
        return events


class BankAccount(Aggregate):
    """Event-sourced bank account aggregate."""

    def __init__(self, account_id: str):
        super().__init__(account_id)
        self.balance: float = 0.0
        self.owner: str = ""
        self.is_open: bool = False
        self.transaction_count: int = 0

    def open(self, owner: str, initial_deposit: float = 0.0):
        """Command: Open a new account."""
        if self.is_open:
            raise ValueError("Account already open")
        if initial_deposit < 0:
            raise ValueError("Initial deposit must be non-negative")

        self._emit("AccountOpened", {
            "owner": owner,
            "initial_deposit": initial_deposit,
        })

    def deposit(self, amount: float, description: str = ""):
        """Command: Deposit money."""
        if not self.is_open:
            raise ValueError("Account is not open")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        self._emit("MoneyDeposited", {
            "amount": amount,
            "description": description,
        })

    def withdraw(self, amount: float, description: str = ""):
        """Command: Withdraw money."""
        if not self.is_open:
            raise ValueError("Account is not open")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError(f"Insufficient funds: {self.balance} < {amount}")

        self._emit("MoneyWithdrawn", {
            "amount": amount,
            "description": description,
        })

    def close(self):
        """Command: Close the account."""
        if not self.is_open:
            raise ValueError("Account already closed")
        if self.balance != 0:
            raise ValueError("Balance must be zero to close")

        self._emit("AccountClosed", {})

    def _apply(self, event: Event):
        """Apply event to state — must be a pure function."""
        if event.event_type == "AccountOpened":
            self.is_open = True
            self.owner = event.data["owner"]
            self.balance = event.data.get("initial_deposit", 0.0)
            if self.balance > 0:
                self.transaction_count += 1

        elif event.event_type == "MoneyDeposited":
            self.balance += event.data["amount"]
            self.transaction_count += 1

        elif event.event_type == "MoneyWithdrawn":
            self.balance -= event.data["amount"]
            self.transaction_count += 1

        elif event.event_type == "AccountClosed":
            self.is_open = False


def demonstrate_event_sourcing():
    """Demonstrate event sourcing with a bank account."""
    print("=== Event Sourcing: Bank Account ===\n")

    store = EventStore()

    # Create and operate on an account
    account = BankAccount("ACC-001")
    account.open("Alice", initial_deposit=1000.0)
    account.deposit(500.0, "Salary")
    account.withdraw(200.0, "Rent")
    account.deposit(100.0, "Refund")

    # Save events
    events = account.get_uncommitted_events()
    version = store.append("ACC-001", events, expected_version=0)

    print(f"Account ACC-001 after {len(events)} events:")
    print(f"  Balance: ${account.balance:.2f}")
    print(f"  Owner: {account.owner}")
    print(f"  Transactions: {account.transaction_count}")
    print(f"  Version: {version}")

    # Rebuild from events
    print(f"\nRebuilding from event log:")
    rebuilt = BankAccount("ACC-001")
    rebuilt.load(store.read_stream("ACC-001"))
    print(f"  Balance: ${rebuilt.balance:.2f} (matches: {rebuilt.balance == account.balance})")

    # Full event history
    print(f"\nEvent History:")
    for event in store.read_stream("ACC-001"):
        print(f"  v{event.version}: {event.event_type} — {event.data}")


demonstrate_event_sourcing()
```

---

## 4. Projections and Read Models

### 4.1 Building Read Models from Events

```python
class Projection(ABC):
    """
    Base class for projections (read models).

    A projection subscribes to events and builds an optimized
    read model for specific query patterns.
    """

    @abstractmethod
    def handle(self, event: Event):
        """Process an event and update the read model."""
        pass


class AccountBalanceProjection(Projection):
    """Projection: current balance for each account."""

    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.events_processed: int = 0

    def handle(self, event: Event):
        aid = event.aggregate_id
        if event.event_type == "AccountOpened":
            self.balances[aid] = event.data.get("initial_deposit", 0.0)
        elif event.event_type == "MoneyDeposited":
            self.balances[aid] = self.balances.get(aid, 0) + event.data["amount"]
        elif event.event_type == "MoneyWithdrawn":
            self.balances[aid] = self.balances.get(aid, 0) - event.data["amount"]
        elif event.event_type == "AccountClosed":
            self.balances.pop(aid, None)
        self.events_processed += 1

    def get_balance(self, account_id: str) -> Optional[float]:
        return self.balances.get(account_id)

    def get_total_deposits(self) -> float:
        return sum(self.balances.values())


class TransactionHistoryProjection(Projection):
    """Projection: transaction history with running balance."""

    def __init__(self):
        self.history: Dict[str, list[dict]] = defaultdict(list)

    def handle(self, event: Event):
        if event.event_type in ("MoneyDeposited", "MoneyWithdrawn"):
            self.history[event.aggregate_id].append({
                "type": "deposit" if event.event_type == "MoneyDeposited" else "withdrawal",
                "amount": event.data["amount"],
                "description": event.data.get("description", ""),
                "timestamp": event.timestamp,
            })

    def get_history(self, account_id: str) -> list[dict]:
        return self.history.get(account_id, [])


class TopAccountsProjection(Projection):
    """Projection: top accounts by balance."""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n
        self.balances: Dict[str, float] = {}

    def handle(self, event: Event):
        aid = event.aggregate_id
        if event.event_type == "AccountOpened":
            self.balances[aid] = event.data.get("initial_deposit", 0.0)
        elif event.event_type == "MoneyDeposited":
            self.balances[aid] = self.balances.get(aid, 0) + event.data["amount"]
        elif event.event_type == "MoneyWithdrawn":
            self.balances[aid] = self.balances.get(aid, 0) - event.data["amount"]

    def get_top(self) -> list[Tuple[str, float]]:
        sorted_accounts = sorted(
            self.balances.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_accounts[:self.top_n]


def demonstrate_projections():
    """Demonstrate multiple projections from the same event stream."""
    print("=== Projections (Read Models) ===\n")

    store = EventStore()
    balance_proj = AccountBalanceProjection()
    history_proj = TransactionHistoryProjection()
    top_proj = TopAccountsProjection(top_n=3)

    # Subscribe projections to the store
    store.subscribe(balance_proj.handle)
    store.subscribe(history_proj.handle)
    store.subscribe(top_proj.handle)

    # Create multiple accounts
    accounts_data = [
        ("ACC-001", "Alice", 1000, [(500, "Salary"), (-200, "Rent")]),
        ("ACC-002", "Bob", 500, [(300, "Freelance"), (-100, "Food")]),
        ("ACC-003", "Charlie", 2000, [(-500, "Investment")]),
    ]

    for acc_id, owner, initial, txns in accounts_data:
        account = BankAccount(acc_id)
        account.open(owner, initial)
        for amount, desc in txns:
            if amount > 0:
                account.deposit(amount, desc)
            else:
                account.withdraw(-amount, desc)
        store.append(acc_id, account.get_uncommitted_events(), 0)

    # Query projections
    print("Balance Projection:")
    for acc_id, _, _, _ in accounts_data:
        bal = balance_proj.get_balance(acc_id)
        print(f"  {acc_id}: ${bal:.2f}")
    print(f"  Total: ${balance_proj.get_total_deposits():.2f}")

    print(f"\nTransaction History (ACC-001):")
    for txn in history_proj.get_history("ACC-001"):
        print(f"  {txn['type']}: ${txn['amount']:.2f} — {txn['description']}")

    print(f"\nTop Accounts:")
    for acc_id, balance in top_proj.get_top():
        print(f"  {acc_id}: ${balance:.2f}")


demonstrate_projections()
```

---

## 5. CQRS Architecture

### 5.1 Command and Query Separation

```python
class CommandBus:
    """
    Command bus for CQRS command handling.

    Commands are validated, processed by the appropriate aggregate,
    and resulting events are stored. The read side is updated
    asynchronously via event subscriptions.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, command_type: str, handler: Callable):
        self.handlers[command_type] = handler

    def dispatch(self, command: dict) -> dict:
        """Dispatch a command to its handler."""
        cmd_type = command.get("type")
        handler = self.handlers.get(cmd_type)
        if not handler:
            return {"ok": False, "error": f"Unknown command: {cmd_type}"}

        try:
            result = handler(command, self.event_store)
            return {"ok": True, **result}
        except (ValueError, ConcurrencyError) as e:
            return {"ok": False, "error": str(e)}


class QueryService:
    """
    Query service for CQRS reads.

    Reads from projections (read models) which are optimized
    for specific query patterns.
    """

    def __init__(self):
        self.projections: Dict[str, Projection] = {}

    def register_projection(self, name: str, projection: Projection):
        self.projections[name] = projection

    def query(self, projection_name: str, query_params: dict) -> Any:
        """Execute a query against a projection."""
        proj = self.projections.get(projection_name)
        if not proj:
            return {"error": f"Unknown projection: {projection_name}"}

        if projection_name == "balance":
            return proj.get_balance(query_params.get("account_id"))
        elif projection_name == "history":
            return proj.get_history(query_params.get("account_id"))
        elif projection_name == "top_accounts":
            return proj.get_top()
        return None


def demonstrate_cqrs():
    """Demonstrate the full CQRS pattern."""
    print("=== CQRS Architecture ===\n")

    # Setup
    store = EventStore()
    balance_proj = AccountBalanceProjection()
    store.subscribe(balance_proj.handle)

    command_bus = CommandBus(store)
    query_service = QueryService()
    query_service.register_projection("balance", balance_proj)

    # Register command handlers
    def handle_open_account(cmd, es):
        account = BankAccount(cmd["account_id"])
        account.open(cmd["owner"], cmd.get("initial_deposit", 0))
        events = account.get_uncommitted_events()
        version = es.append(cmd["account_id"], events, 0)
        return {"version": version}

    def handle_deposit(cmd, es):
        account = BankAccount(cmd["account_id"])
        account.load(es.read_stream(cmd["account_id"]))
        account.deposit(cmd["amount"], cmd.get("description", ""))
        events = account.get_uncommitted_events()
        version = es.append(cmd["account_id"], events, account.version)
        return {"version": version}

    command_bus.register_handler("OpenAccount", handle_open_account)
    command_bus.register_handler("Deposit", handle_deposit)

    # Execute commands
    print("Commands:")
    result = command_bus.dispatch({
        "type": "OpenAccount",
        "account_id": "ACC-100",
        "owner": "Diana",
        "initial_deposit": 500,
    })
    print(f"  OpenAccount: {result}")

    result = command_bus.dispatch({
        "type": "Deposit",
        "account_id": "ACC-100",
        "amount": 250,
        "description": "Bonus",
    })
    print(f"  Deposit: {result}")

    # Query (reads from projection, not events)
    balance = query_service.query("balance", {"account_id": "ACC-100"})
    print(f"\nQuery: balance of ACC-100 = ${balance:.2f}")

    print(f"\nArchitecture:")
    print(f"  Write side: Command → Aggregate → Event Store")
    print(f"  Read side:  Event Store → Projection → Query")
    print(f"  Consistency: Eventually consistent between write and read")


demonstrate_cqrs()
```

---

## 6. Snapshots and Performance

### 6.1 Aggregate Snapshots

```python
class SnapshotStore:
    """
    Snapshot store for aggregate state.

    When an aggregate has many events, replaying all of them
    becomes slow. Snapshots capture the aggregate state at a
    point in time, allowing replay to start from the snapshot.
    """

    def __init__(self, snapshot_interval: int = 100):
        self.snapshots: Dict[str, dict] = {}
        self.snapshot_interval = snapshot_interval

    def should_snapshot(self, version: int) -> bool:
        return version > 0 and version % self.snapshot_interval == 0

    def save_snapshot(self, aggregate_id: str, version: int, state: dict):
        self.snapshots[aggregate_id] = {
            "version": version,
            "state": state,
            "timestamp": time.time(),
        }

    def load_snapshot(self, aggregate_id: str) -> Optional[dict]:
        return self.snapshots.get(aggregate_id)


def demonstrate_snapshots():
    """Demonstrate snapshot optimization for event replay."""
    print("=== Snapshots ===\n")

    store = EventStore()
    snapshot_store = SnapshotStore(snapshot_interval=50)

    # Create account with many transactions
    account = BankAccount("ACC-PERF")
    account.open("Performance Test", 10000)
    events_batch = account.get_uncommitted_events()
    store.append("ACC-PERF", events_batch, 0)

    for i in range(200):
        account_reload = BankAccount("ACC-PERF")
        account_reload.load(store.read_stream("ACC-PERF"))
        if i % 2 == 0:
            account_reload.deposit(10.0, f"Deposit {i}")
        else:
            account_reload.withdraw(5.0, f"Withdrawal {i}")
        events = account_reload.get_uncommitted_events()
        store.append("ACC-PERF", events, account_reload.version)

        # Periodic snapshots
        current_version = store.stream_version("ACC-PERF")
        if snapshot_store.should_snapshot(current_version):
            snap_account = BankAccount("ACC-PERF")
            snap_account.load(store.read_stream("ACC-PERF"))
            snapshot_store.save_snapshot("ACC-PERF", current_version, {
                "balance": snap_account.balance,
                "owner": snap_account.owner,
                "is_open": snap_account.is_open,
                "transaction_count": snap_account.transaction_count,
            })

    total_events = store.stream_version("ACC-PERF")
    snapshot = snapshot_store.load_snapshot("ACC-PERF")

    print(f"Total events: {total_events}")
    if snapshot:
        print(f"Snapshot at version: {snapshot['version']}")
        print(f"Events to replay with snapshot: {total_events - snapshot['version']}")
        print(f"Events saved: {snapshot['version']} ({snapshot['version']/total_events*100:.0f}%)")
        print(f"Snapshot state: balance=${snapshot['state']['balance']:.2f}")


demonstrate_snapshots()
```

---

## 7. Event Schema Evolution

### 7.1 Handling Schema Changes

```python
class EventUpcaster:
    """
    Event upcaster for handling schema evolution.

    When the event schema changes, old events must be upcast
    (transformed) to the new schema during replay.
    """

    def __init__(self):
        self.upcasters: Dict[Tuple[str, int], Callable] = {}

    def register(self, event_type: str, from_version: int, upcaster: Callable):
        self.upcasters[(event_type, from_version)] = upcaster

    def upcast(self, event: Event) -> Event:
        schema_version = event.metadata.get("schema_version", 1)
        key = (event.event_type, schema_version)

        while key in self.upcasters:
            event = self.upcasters[key](event)
            schema_version += 1
            event.metadata["schema_version"] = schema_version
            key = (event.event_type, schema_version)

        return event


def demonstrate_schema_evolution():
    """Demonstrate event schema evolution with upcasters."""
    print("=== Event Schema Evolution ===\n")

    upcaster = EventUpcaster()

    # V1 → V2: "amount" was in cents, now in dollars
    def upcast_deposit_v1_to_v2(event: Event) -> Event:
        new_data = dict(event.data)
        new_data["amount"] = new_data["amount"] / 100.0  # cents → dollars
        new_data["currency"] = "USD"  # New field
        return Event(
            event_id=event.event_id,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            version=event.version,
            timestamp=event.timestamp,
            data=new_data,
            metadata={**event.metadata, "schema_version": 2},
        )

    upcaster.register("MoneyDeposited", 1, upcast_deposit_v1_to_v2)

    # Old event (schema v1)
    old_event = Event(
        event_type="MoneyDeposited",
        data={"amount": 50000, "description": "Salary"},  # 500.00 in cents
        metadata={"schema_version": 1},
    )

    print(f"Original (v1): {old_event.data}")
    upgraded = upcaster.upcast(old_event)
    print(f"Upgraded (v2): {upgraded.data}")


demonstrate_schema_evolution()
```

---

## 8. Distributed Event Sourcing

### 8.1 Multi-Node Event Store

```python
class DistributedEventStore:
    """
    Distributed event store with partitioning and replication.

    Events are partitioned by aggregate_id and replicated
    across multiple nodes for fault tolerance.
    """

    def __init__(self, num_partitions: int = 4, replication_factor: int = 3):
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.partitions: Dict[int, EventStore] = {
            i: EventStore() for i in range(num_partitions)
        }

    def _partition_for(self, aggregate_id: str) -> int:
        h = int(hashlib.md5(aggregate_id.encode()).hexdigest(), 16)
        return h % self.num_partitions

    def append(self, aggregate_id: str, events: list[Event],
               expected_version: int) -> int:
        partition = self._partition_for(aggregate_id)
        return self.partitions[partition].append(
            aggregate_id, events, expected_version
        )

    def read_stream(self, aggregate_id: str,
                    from_version: int = 0) -> list[Event]:
        partition = self._partition_for(aggregate_id)
        return self.partitions[partition].read_stream(aggregate_id, from_version)

    def stats(self) -> dict:
        return {
            f"partition_{i}": len(store.global_log)
            for i, store in self.partitions.items()
        }


def demonstrate_distributed_es():
    """Demonstrate distributed event sourcing."""
    print("=== Distributed Event Sourcing ===\n")

    store = DistributedEventStore(num_partitions=4)

    # Create accounts (distributed across partitions)
    for i in range(20):
        acc_id = f"ACC-{i:03d}"
        account = BankAccount(acc_id)
        account.open(f"User-{i}", initial_deposit=100.0 * (i + 1))
        account.deposit(50.0, "Welcome bonus")
        store.append(acc_id, account.get_uncommitted_events(), 0)

    print("Partition distribution:")
    for name, count in store.stats().items():
        print(f"  {name}: {count} events")

    # Rebuild a specific account
    account = BankAccount("ACC-005")
    account.load(store.read_stream("ACC-005"))
    print(f"\nACC-005 balance: ${account.balance:.2f}")


demonstrate_distributed_es()
```

---

## 9. Real-World Systems

### 9.1 System Comparison

```python
def compare_event_sourcing_systems():
    """Compare event sourcing and CQRS implementations."""
    print("=== Event Sourcing Systems ===\n")

    systems = [
        {"name": "EventStoreDB", "type": "Purpose-built event store",
         "features": "Projections, subscriptions, competing consumers"},
        {"name": "Apache Kafka", "type": "Distributed log",
         "features": "High throughput, partitioning, exactly-once"},
        {"name": "Axon Framework", "type": "Java CQRS/ES framework",
         "features": "Aggregate, saga, projection, Axon Server"},
        {"name": "Marten", "type": ".NET event store (on PostgreSQL)",
         "features": "Document store + event store, projections"},
        {"name": "DynamoDB Streams", "type": "Change data capture",
         "features": "Event-driven, Lambda integration"},
    ]

    for sys in systems:
        print(f"  {sys['name']} ({sys['type']}):")
        print(f"    Features: {sys['features']}")


compare_event_sourcing_systems()
```

---

## 10. Summary and Key Takeaways

### Event Sourcing Decision Matrix

> **WHEN TO USE EVENT SOURCING**
>
> Use when: audit trail is required, complex domain logic, event-driven architecture
> Avoid when: simple CRUD, frequent ad-hoc queries, team unfamiliar with pattern
>
> **CQRS ADDS VALUE WHEN**
>
> Read and write models have very different structures
> Read and write loads need independent scaling
> Complex queries would burden the write model

### Key Principles

1. **Events are facts**: Immutable, append-only, describe what happened.
2. **State is derived**: Current state is always computable from the event log.
3. **Projections are disposable**: They can be rebuilt from the event log at any time.
4. **Snapshots are an optimization**: Not a requirement, but necessary for long-lived aggregates.
5. **Schema evolution is inevitable**: Plan for upcasting from day one.

---

## 11. Practice Problems

### Problem 1: Aggregate Design

Design an event-sourced Shopping Cart aggregate with events: CartCreated, ItemAdded, ItemRemoved, QuantityChanged, CartCheckedOut. Implement all validation rules.

### Problem 2: Projection Design

Create three projections from a stream of order events: (a) per-customer order count, (b) daily revenue summary, (c) product popularity ranking. Ensure they can be rebuilt from scratch.

### Problem 3: Concurrency

Two users simultaneously try to withdraw from the same account. The balance is $100, and each wants $80. Show how optimistic concurrency control prevents overdraft.

### Problem 4: Implementation Challenge

Build a complete CQRS system with: command bus, event store, 3 projections, snapshot support, and an HTTP API for commands and queries.

### Problem 5: Schema Evolution

An event "OrderPlaced" originally had {product_id, quantity, price}. Design upcasters for: (a) adding "currency" field, (b) splitting "price" into "unit_price" and "total_price", (c) renaming "product_id" to "sku".

---

## 12. References

1. Young, G. (2010). "CQRS Documents." https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf
2. Fowler, M. (2005). "Event Sourcing." martinfowler.com.
3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 11. O'Reilly Media.
4. Vernon, V. (2013). *Implementing Domain-Driven Design*. Addison-Wesley.
5. EventStoreDB documentation: https://www.eventstore.com/docs
6. Overeem, M. et al. (2017). "The Dark Side of Event Sourcing." *IEEE Software*.
7. Betts, D. et al. (2013). *Exploring CQRS and Event Sourcing*. Microsoft patterns & practices.

---

[Next: Lesson 25 — Vector Clocks](./25_Vector_Clocks.md)
