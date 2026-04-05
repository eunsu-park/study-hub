# 레슨 24: 이벤트 소싱과 CQRS (Event Sourcing and CQRS)

[개요](./00_Overview.md) | [이전: 분산 속도 제한](./23_Distributed_Rate_Limiting.md) | [다음: 벡터 클럭](./25_Vector_Clocks.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 추가 전용(append-only) 도메인 이벤트 로그에서 상태를 도출하는 이벤트 소싱(event sourcing) 시스템 설계
2. 확장성을 위해 읽기와 쓰기 모델을 분리하는 CQRS 패턴 구현
3. 스냅샷팅(snapshotting)과 프로젝션(projection) 재구축이 가능한 이벤트 스토어(event store) 구축
4. 커맨드(command)와 쿼리(query) 측 간의 최종 일관성(eventual consistency) 처리
5. 재생 비용, 스키마 진화(schema evolution), 디버깅을 포함한 이벤트 소싱 트레이드오프 분석

---

## 목차

1. [이벤트 소싱 기초](#1-이벤트-소싱-기초)
2. [이벤트 스토어 구현](#2-이벤트-스토어-구현)
3. [집계와 커맨드](#3-집계와-커맨드)
4. [프로젝션과 읽기 모델](#4-프로젝션과-읽기-모델)
5. [CQRS 아키텍처](#5-cqrs-아키텍처)
6. [스냅샷과 성능](#6-스냅샷과-성능)
7. [이벤트 스키마 진화](#7-이벤트-스키마-진화)
8. [분산 이벤트 소싱](#8-분산-이벤트-소싱)
9. [실제 시스템](#9-실제-시스템)
10. [요약 및 핵심 정리](#10-요약-및-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 이벤트 소싱 기초

### 1.1 전통적 CRUD vs 이벤트 소싱

```
CRUD:                          이벤트 소싱:
┌─────────────┐               ┌─────────────────────┐
│ Account     │               │ Event Log           │
│ balance: 150│               │ 1. AccountCreated   │
│ name: Alice │               │ 2. Deposited(200)   │
└─────────────┘               │ 3. Withdrawn(50)    │
                              │ 4. NameChanged(Alice)│
상태 = 최신 스냅샷            └─────────────────────┘
손실: 여기에 어떻게 왔는가?    상태 = replay(events)
                              전체 이력 보존
```

### 1.2 핵심 개념

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
    """모든 도메인 이벤트의 기본 클래스."""
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

## 2. 이벤트 스토어 구현

### 2.1 추가 전용 이벤트 스토어

```python
class EventStore:
    """
    낙관적 동시성 제어(optimistic concurrency control)가 포함된
    추가 전용(append-only) 이벤트 스토어.

    이벤트는 집계별(per-aggregate) 스트림에 저장된다. 각 추가 시
    예상 버전(expected version)을 지정하여 동시 수정이
    충돌을 생성하는 것을 방지한다.
    """

    def __init__(self):
        self.streams: Dict[str, list[Event]] = defaultdict(list)
        self.global_log: list[Event] = []
        self.global_position: int = 0
        self.subscribers: list[Callable] = []

    def append(self, aggregate_id: str, events: list[Event],
               expected_version: int) -> int:
        """
        집계 스트림에 이벤트를 추가한다.

        Args:
            aggregate_id: 추가할 집계
            events: 추가할 이벤트 목록
            expected_version: 예상 현재 버전 (낙관적 동시성)

        Returns:
            추가 후 새 버전 번호

        Raises:
            expected_version이 일치하지 않으면 ConcurrencyError
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

            # 구독자에게 통지
            for subscriber in self.subscribers:
                subscriber(event)

        return current_version + len(events)

    def read_stream(self, aggregate_id: str,
                    from_version: int = 0) -> list[Event]:
        """집계 스트림에서 이벤트를 읽는다."""
        stream = self.streams.get(aggregate_id, [])
        return [e for e in stream if e.version > from_version]

    def read_all(self, from_position: int = 0) -> list[Event]:
        """전역 이벤트 로그에서 읽는다."""
        return self.global_log[from_position:]

    def subscribe(self, callback: Callable):
        """새 이벤트를 구독한다."""
        self.subscribers.append(callback)

    def stream_version(self, aggregate_id: str) -> int:
        """스트림의 현재 버전을 가져온다."""
        return len(self.streams.get(aggregate_id, []))


class ConcurrencyError(Exception):
    pass
```

---

## 3. 집계와 커맨드

### 3.1 집계 패턴 (Aggregate Pattern)

```python
class Aggregate(ABC):
    """
    이벤트 소싱 집계(aggregate)의 기본 클래스.

    집계는:
    1. 커맨드(command)를 수신한다
    2. 비즈니스 규칙을 검증한다
    3. 이벤트를 발행한다 (유효한 경우)
    4. 이벤트를 적용하여 상태를 업데이트한다
    """

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version: int = 0
        self.uncommitted_events: list[Event] = []

    def load(self, events: list[Event]):
        """이벤트 목록에서 상태를 재구축한다."""
        for event in events:
            self._apply(event)
            self.version = event.version

    def _emit(self, event_type: str, data: dict):
        """새 이벤트를 발행한다."""
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
        """이벤트를 적용하여 상태를 업데이트한다. 순수 함수여야 한다."""
        pass

    def get_uncommitted_events(self) -> list[Event]:
        """커밋되지 않은 이벤트를 가져오고 지운다."""
        events = self.uncommitted_events
        self.uncommitted_events = []
        return events


class BankAccount(Aggregate):
    """이벤트 소싱 은행 계좌 집계."""

    def __init__(self, account_id: str):
        super().__init__(account_id)
        self.balance: float = 0.0
        self.owner: str = ""
        self.is_open: bool = False
        self.transaction_count: int = 0

    def open(self, owner: str, initial_deposit: float = 0.0):
        """커맨드: 새 계좌를 개설한다."""
        if self.is_open:
            raise ValueError("Account already open")
        if initial_deposit < 0:
            raise ValueError("Initial deposit must be non-negative")

        self._emit("AccountOpened", {
            "owner": owner,
            "initial_deposit": initial_deposit,
        })

    def deposit(self, amount: float, description: str = ""):
        """커맨드: 입금한다."""
        if not self.is_open:
            raise ValueError("Account is not open")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        self._emit("MoneyDeposited", {
            "amount": amount,
            "description": description,
        })

    def withdraw(self, amount: float, description: str = ""):
        """커맨드: 출금한다."""
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
        """커맨드: 계좌를 닫는다."""
        if not self.is_open:
            raise ValueError("Account already closed")
        if self.balance != 0:
            raise ValueError("Balance must be zero to close")

        self._emit("AccountClosed", {})

    def _apply(self, event: Event):
        """이벤트를 상태에 적용한다 — 순수 함수여야 한다."""
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
    """은행 계좌를 사용하여 이벤트 소싱을 시연한다."""
    print("=== Event Sourcing: Bank Account ===\n")

    store = EventStore()

    # 계좌 생성 및 운영
    account = BankAccount("ACC-001")
    account.open("Alice", initial_deposit=1000.0)
    account.deposit(500.0, "Salary")
    account.withdraw(200.0, "Rent")
    account.deposit(100.0, "Refund")

    # 이벤트 저장
    events = account.get_uncommitted_events()
    version = store.append("ACC-001", events, expected_version=0)

    print(f"Account ACC-001 after {len(events)} events:")
    print(f"  Balance: ${account.balance:.2f}")
    print(f"  Owner: {account.owner}")
    print(f"  Transactions: {account.transaction_count}")
    print(f"  Version: {version}")

    # 이벤트에서 재구축
    print(f"\nRebuilding from event log:")
    rebuilt = BankAccount("ACC-001")
    rebuilt.load(store.read_stream("ACC-001"))
    print(f"  Balance: ${rebuilt.balance:.2f} (matches: {rebuilt.balance == account.balance})")

    # 전체 이벤트 이력
    print(f"\nEvent History:")
    for event in store.read_stream("ACC-001"):
        print(f"  v{event.version}: {event.event_type} — {event.data}")


demonstrate_event_sourcing()
```

---

## 4. 프로젝션과 읽기 모델

### 4.1 이벤트에서 읽기 모델 구축

```python
class Projection(ABC):
    """
    프로젝션(읽기 모델)의 기본 클래스.

    프로젝션은 이벤트를 구독하고 특정 쿼리 패턴에
    최적화된 읽기 모델을 구축한다.
    """

    @abstractmethod
    def handle(self, event: Event):
        """이벤트를 처리하고 읽기 모델을 업데이트한다."""
        pass


class AccountBalanceProjection(Projection):
    """프로젝션: 각 계좌의 현재 잔액."""

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
    """프로젝션: 누적 잔액이 포함된 거래 이력."""

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
    """프로젝션: 잔액별 상위 계좌."""

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
    """동일한 이벤트 스트림에서의 다중 프로젝션을 시연한다."""
    print("=== Projections (Read Models) ===\n")

    store = EventStore()
    balance_proj = AccountBalanceProjection()
    history_proj = TransactionHistoryProjection()
    top_proj = TopAccountsProjection(top_n=3)

    # 프로젝션을 스토어에 구독
    store.subscribe(balance_proj.handle)
    store.subscribe(history_proj.handle)
    store.subscribe(top_proj.handle)

    # 여러 계좌 생성
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

    # 프로젝션 쿼리
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

## 5. CQRS 아키텍처

### 5.1 커맨드와 쿼리 분리

```python
class CommandBus:
    """
    CQRS 커맨드 처리를 위한 커맨드 버스(command bus).

    커맨드는 검증되고, 적절한 집계에 의해 처리되며,
    결과 이벤트가 저장된다. 읽기 측은 이벤트 구독을 통해
    비동기적으로 업데이트된다.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, command_type: str, handler: Callable):
        self.handlers[command_type] = handler

    def dispatch(self, command: dict) -> dict:
        """커맨드를 해당 핸들러에 디스패치한다."""
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
    CQRS 읽기를 위한 쿼리 서비스(query service).

    특정 쿼리 패턴에 최적화된 프로젝션(읽기 모델)에서 읽는다.
    """

    def __init__(self):
        self.projections: Dict[str, Projection] = {}

    def register_projection(self, name: str, projection: Projection):
        self.projections[name] = projection

    def query(self, projection_name: str, query_params: dict) -> Any:
        """프로젝션에 대해 쿼리를 실행한다."""
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
    """전체 CQRS 패턴을 시연한다."""
    print("=== CQRS Architecture ===\n")

    # 설정
    store = EventStore()
    balance_proj = AccountBalanceProjection()
    store.subscribe(balance_proj.handle)

    command_bus = CommandBus(store)
    query_service = QueryService()
    query_service.register_projection("balance", balance_proj)

    # 커맨드 핸들러 등록
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

    # 커맨드 실행
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

    # 쿼리 (이벤트가 아닌 프로젝션에서 읽기)
    balance = query_service.query("balance", {"account_id": "ACC-100"})
    print(f"\nQuery: balance of ACC-100 = ${balance:.2f}")

    print(f"\nArchitecture:")
    print(f"  Write side: Command → Aggregate → Event Store")
    print(f"  Read side:  Event Store → Projection → Query")
    print(f"  Consistency: Eventually consistent between write and read")


demonstrate_cqrs()
```

---

## 6. 스냅샷과 성능

### 6.1 집계 스냅샷 (Aggregate Snapshots)

```python
class SnapshotStore:
    """
    집계 상태를 위한 스냅샷 스토어.

    집계에 많은 이벤트가 있으면 모두 재생하는 것이
    느려진다. 스냅샷은 특정 시점의 집계 상태를 캡처하여
    스냅샷부터 재생을 시작할 수 있게 한다.
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
    """이벤트 재생을 위한 스냅샷 최적화를 시연한다."""
    print("=== Snapshots ===\n")

    store = EventStore()
    snapshot_store = SnapshotStore(snapshot_interval=50)

    # 많은 거래가 있는 계좌 생성
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

        # 주기적 스냅샷
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

## 7. 이벤트 스키마 진화

### 7.1 스키마 변경 처리

```python
class EventUpcaster:
    """
    스키마 진화(schema evolution)를 처리하기 위한 이벤트 업캐스터(upcaster).

    이벤트 스키마가 변경되면 재생 중에 이전 이벤트를
    새 스키마로 업캐스트(변환)해야 한다.
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
    """업캐스터를 사용한 이벤트 스키마 진화를 시연한다."""
    print("=== Event Schema Evolution ===\n")

    upcaster = EventUpcaster()

    # V1 → V2: "amount"가 센트 단위였으나 이제 달러 단위
    def upcast_deposit_v1_to_v2(event: Event) -> Event:
        new_data = dict(event.data)
        new_data["amount"] = new_data["amount"] / 100.0  # 센트 → 달러
        new_data["currency"] = "USD"  # 새 필드
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

    # 이전 이벤트 (스키마 v1)
    old_event = Event(
        event_type="MoneyDeposited",
        data={"amount": 50000, "description": "Salary"},  # 센트 단위 500.00
        metadata={"schema_version": 1},
    )

    print(f"Original (v1): {old_event.data}")
    upgraded = upcaster.upcast(old_event)
    print(f"Upgraded (v2): {upgraded.data}")


demonstrate_schema_evolution()
```

---

## 8. 분산 이벤트 소싱

### 8.1 다중 노드 이벤트 스토어

```python
class DistributedEventStore:
    """
    파티셔닝(partitioning)과 복제(replication)가 포함된
    분산 이벤트 스토어.

    이벤트는 aggregate_id로 파티셔닝되고 장애 허용을 위해
    여러 노드에 복제된다.
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
    """분산 이벤트 소싱을 시연한다."""
    print("=== Distributed Event Sourcing ===\n")

    store = DistributedEventStore(num_partitions=4)

    # 계좌 생성 (파티션에 분산)
    for i in range(20):
        acc_id = f"ACC-{i:03d}"
        account = BankAccount(acc_id)
        account.open(f"User-{i}", initial_deposit=100.0 * (i + 1))
        account.deposit(50.0, "Welcome bonus")
        store.append(acc_id, account.get_uncommitted_events(), 0)

    print("Partition distribution:")
    for name, count in store.stats().items():
        print(f"  {name}: {count} events")

    # 특정 계좌 재구축
    account = BankAccount("ACC-005")
    account.load(store.read_stream("ACC-005"))
    print(f"\nACC-005 balance: ${account.balance:.2f}")


demonstrate_distributed_es()
```

---

## 9. 실제 시스템

### 9.1 시스템 비교

```python
def compare_event_sourcing_systems():
    """이벤트 소싱과 CQRS 구현을 비교한다."""
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

## 10. 요약 및 핵심 정리

### 이벤트 소싱 결정 매트릭스

> **이벤트 소싱을 사용해야 할 때 (WHEN TO USE EVENT SOURCING)**
>
> 사용할 때: 감사 추적(audit trail)이 필요한 경우, 복잡한 도메인 로직, 이벤트 구동 아키텍처
> 피할 때: 단순 CRUD, 잦은 임시 쿼리(ad-hoc query), 팀이 패턴에 익숙하지 않은 경우
>
> **CQRS가 가치를 더하는 경우**
>
> 읽기와 쓰기 모델이 매우 다른 구조를 가질 때
> 읽기와 쓰기 부하가 독립적으로 확장되어야 할 때
> 복잡한 쿼리가 쓰기 모델에 부담을 줄 때

### 핵심 원칙

1. **이벤트는 사실(fact)이다**: 불변(immutable)이고, 추가 전용(append-only)이며, 발생한 일을 설명한다.
2. **상태는 도출된다**: 현재 상태는 항상 이벤트 로그에서 계산 가능하다.
3. **프로젝션은 일회용이다**: 이벤트 로그에서 언제든지 재구축할 수 있다.
4. **스냅샷은 최적화이다**: 요구사항이 아니지만 장수 집계에 필요하다.
5. **스키마 진화는 불가피하다**: 처음부터 업캐스팅(upcasting)을 계획한다.

---

## 11. 연습 문제

### 문제 1: 집계 설계

CartCreated, ItemAdded, ItemRemoved, QuantityChanged, CartCheckedOut 이벤트가 포함된 이벤트 소싱 장바구니(Shopping Cart) 집계를 설계한다. 모든 검증 규칙을 구현한다.

### 문제 2: 프로젝션 설계

주문 이벤트 스트림에서 세 가지 프로젝션을 생성한다: (a) 고객별 주문 수, (b) 일별 매출 요약, (c) 제품 인기 순위. 처음부터 재구축할 수 있도록 보장한다.

### 문제 3: 동시성

두 사용자가 동시에 같은 계좌에서 출금을 시도한다. 잔액은 $100이고 각각 $80을 원한다. 낙관적 동시성 제어(optimistic concurrency control)가 초과 인출(overdraft)을 방지하는 방법을 보여준다.

### 문제 4: 구현 도전

다음을 포함하는 완전한 CQRS 시스템을 구축한다: 커맨드 버스, 이벤트 스토어, 3개 프로젝션, 스냅샷 지원, 커맨드와 쿼리를 위한 HTTP API.

### 문제 5: 스키마 진화

"OrderPlaced" 이벤트가 원래 {product_id, quantity, price}를 가졌다. 다음에 대한 업캐스터를 설계한다: (a) "currency" 필드 추가, (b) "price"를 "unit_price"와 "total_price"로 분리, (c) "product_id"를 "sku"로 이름 변경.

---

## 12. 참고 문헌

1. Young, G. (2010). "CQRS Documents." https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf
2. Fowler, M. (2005). "Event Sourcing." martinfowler.com.
3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 11. O'Reilly Media.
4. Vernon, V. (2013). *Implementing Domain-Driven Design*. Addison-Wesley.
5. EventStoreDB documentation: https://www.eventstore.com/docs
6. Overeem, M. et al. (2017). "The Dark Side of Event Sourcing." *IEEE Software*.
7. Betts, D. et al. (2013). *Exploring CQRS and Event Sourcing*. Microsoft patterns & practices.

---

[다음: 레슨 25 — 벡터 클럭](./25_Vector_Clocks.md)
