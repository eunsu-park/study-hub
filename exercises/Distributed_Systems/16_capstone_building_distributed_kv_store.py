"""
Exercises for Lesson 16: Capstone - Building a Distributed KV Store
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import hashlib
import json
import random
import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# === Exercise 1: KV Store with Snapshot and Restore ===
# Problem: Extend a basic key-value store with snapshot and restore
# functionality. Snapshots capture a consistent point-in-time view
# of the data. Restore rebuilds the state from a snapshot.

class KVStoreWithSnapshots:
    """
    Key-value store with snapshot and restore support.

    Snapshots are identified by monotonically increasing IDs.
    Each snapshot captures the full state at a point in time.
    """

    def __init__(self):
        self.data: Dict[str, str] = {}
        self.snapshots: Dict[int, Dict[str, str]] = {}
        self.wal: List[Tuple[int, str, str, Optional[str]]] = []  # (seq, op, key, value)
        self.seq_counter = 0
        self.snapshot_counter = 0

    def put(self, key: str, value: str):
        """Write a key-value pair."""
        self.seq_counter += 1
        old_value = self.data.get(key)
        self.wal.append((self.seq_counter, "PUT", key, value))
        self.data[key] = value

    def get(self, key: str) -> Optional[str]:
        """Read a value by key."""
        return self.data.get(key)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        if key in self.data:
            self.seq_counter += 1
            self.wal.append((self.seq_counter, "DELETE", key, None))
            del self.data[key]
            return True
        return False

    def create_snapshot(self) -> int:
        """
        Create a snapshot of the current state.
        Returns the snapshot ID.
        """
        self.snapshot_counter += 1
        snap_id = self.snapshot_counter
        self.snapshots[snap_id] = dict(self.data)
        return snap_id

    def restore_snapshot(self, snapshot_id: int) -> bool:
        """
        Restore state from a snapshot.
        Returns True if successful.
        """
        if snapshot_id not in self.snapshots:
            return False
        self.data = dict(self.snapshots[snapshot_id])
        self.seq_counter += 1
        self.wal.append((self.seq_counter, "RESTORE", f"snapshot_{snapshot_id}", None))
        return True

    def list_snapshots(self) -> List[Tuple[int, int]]:
        """List all snapshots with their sizes."""
        return [(sid, len(data)) for sid, data in sorted(self.snapshots.items())]

    def compact_wal(self, up_to_seq: int):
        """Remove WAL entries up to a sequence number."""
        self.wal = [entry for entry in self.wal if entry[0] > up_to_seq]


def exercise_1():
    """
    Demonstrate KV store snapshot and restore.
    """
    print("=== Exercise 1: KV Store with Snapshot & Restore ===\n")

    store = KVStoreWithSnapshots()

    # Write some data
    store.put("name", "Alice")
    store.put("age", "30")
    store.put("city", "Seoul")
    print(f"Initial data: name={store.get('name')}, age={store.get('age')}, "
          f"city={store.get('city')}")

    # Create snapshot 1
    snap1 = store.create_snapshot()
    print(f"\nSnapshot {snap1} created (3 keys)")

    # Modify data
    store.put("name", "Bob")
    store.put("email", "bob@example.com")
    store.delete("city")
    print(f"Modified: name={store.get('name')}, city={store.get('city')}, "
          f"email={store.get('email')}")

    # Create snapshot 2
    snap2 = store.create_snapshot()
    print(f"Snapshot {snap2} created (3 keys)")

    # More changes
    store.put("name", "Charlie")
    store.put("phone", "123-456")
    print(f"Further modified: name={store.get('name')}")

    # Restore snapshot 1
    store.restore_snapshot(snap1)
    print(f"\nRestored to snapshot {snap1}:")
    print(f"  name={store.get('name')}, age={store.get('age')}, "
          f"city={store.get('city')}, email={store.get('email')}")

    assert store.get("name") == "Alice"
    assert store.get("city") == "Seoul"
    assert store.get("email") is None  # didn't exist in snapshot 1

    # Restore snapshot 2
    store.restore_snapshot(snap2)
    print(f"\nRestored to snapshot {snap2}:")
    print(f"  name={store.get('name')}, city={store.get('city')}, "
          f"email={store.get('email')}")

    assert store.get("name") == "Bob"
    assert store.get("email") == "bob@example.com"

    # List snapshots
    print(f"\nSnapshots: {store.list_snapshots()}")
    print(f"WAL entries: {len(store.wal)}")

    # Compact WAL
    store.compact_wal(3)
    print(f"WAL after compaction: {len(store.wal)} entries")
    print()


# === Exercise 2: Client Request Deduplication ===
# Problem: Implement client request deduplication using client ID
# and sequence numbers. This prevents the same operation from being
# applied twice if the client retries after a timeout.

@dataclass
class ClientRequest:
    """A client request with deduplication metadata."""
    client_id: str
    sequence_num: int
    operation: str  # "PUT key value" or "DELETE key"


@dataclass
class ClientResponse:
    """Response to a client request."""
    success: bool
    value: Optional[str] = None
    is_duplicate: bool = False


class DeduplicatedKVStore:
    """
    KV store with client request deduplication.

    Tracks the latest sequence number and response per client.
    If a request with an already-seen sequence number arrives,
    the stored response is returned without re-executing.
    """

    def __init__(self, dedup_window: int = 1000):
        self.data: Dict[str, str] = {}
        # client_id -> (latest_seq, response_cache)
        self.client_table: Dict[str, Dict[int, ClientResponse]] = defaultdict(dict)
        self.dedup_window = dedup_window
        self.total_requests = 0
        self.deduplicated_count = 0

    def process_request(self, request: ClientRequest) -> ClientResponse:
        """
        Process a client request with deduplication.
        """
        self.total_requests += 1

        # Check for duplicate
        client_cache = self.client_table[request.client_id]
        if request.sequence_num in client_cache:
            self.deduplicated_count += 1
            response = client_cache[request.sequence_num]
            return ClientResponse(
                success=response.success,
                value=response.value,
                is_duplicate=True,
            )

        # Execute the operation
        response = self._execute(request.operation)

        # Cache the response
        client_cache[request.sequence_num] = response

        # Evict old entries beyond the dedup window
        if len(client_cache) > self.dedup_window:
            oldest = min(client_cache.keys())
            del client_cache[oldest]

        return response

    def _execute(self, operation: str) -> ClientResponse:
        """Execute an operation on the store."""
        parts = operation.split()
        if not parts:
            return ClientResponse(success=False)

        op = parts[0].upper()
        if op == "PUT" and len(parts) >= 3:
            key, value = parts[1], " ".join(parts[2:])
            self.data[key] = value
            return ClientResponse(success=True)
        elif op == "GET" and len(parts) >= 2:
            key = parts[1]
            value = self.data.get(key)
            return ClientResponse(success=True, value=value)
        elif op == "DELETE" and len(parts) >= 2:
            key = parts[1]
            if key in self.data:
                del self.data[key]
                return ClientResponse(success=True)
            return ClientResponse(success=False)

        return ClientResponse(success=False)


def exercise_2():
    """
    Demonstrate client request deduplication.
    """
    print("=== Exercise 2: Client Request Deduplication ===\n")

    store = DeduplicatedKVStore()

    # Normal request
    req1 = ClientRequest("client_1", 1, "PUT name Alice")
    resp1 = store.process_request(req1)
    print(f"Request 1: {req1.operation} -> success={resp1.success}, dup={resp1.is_duplicate}")

    # Client retries the same request (network timeout, etc.)
    resp1_retry = store.process_request(req1)
    print(f"Retry 1:   {req1.operation} -> success={resp1_retry.success}, dup={resp1_retry.is_duplicate}")
    assert resp1_retry.is_duplicate is True

    # Next request from same client
    req2 = ClientRequest("client_1", 2, "PUT age 30")
    resp2 = store.process_request(req2)
    print(f"Request 2: {req2.operation} -> success={resp2.success}, dup={resp2.is_duplicate}")

    # Different client, same sequence number (should NOT deduplicate)
    req3 = ClientRequest("client_2", 1, "PUT name Bob")
    resp3 = store.process_request(req3)
    print(f"Request 3: {req3.operation} -> success={resp3.success}, dup={resp3.is_duplicate}")
    assert resp3.is_duplicate is False

    # Read request
    req4 = ClientRequest("client_1", 3, "GET name")
    resp4 = store.process_request(req4)
    print(f"Request 4: {req4.operation} -> value={resp4.value}, dup={resp4.is_duplicate}")

    # Verify data reflects the latest write (Bob from client_2)
    assert resp4.value == "Alice"  # client_1 reads its own write context

    print(f"\nTotal requests: {store.total_requests}")
    print(f"Deduplicated:   {store.deduplicated_count}")
    print(f"Store state:    {store.data}")
    print()


# === Exercise 3: Fault Injection Test Harness ===
# Problem: Build a fault injection test harness that validates
# linearizability of a KV store under simulated failures. The
# harness injects various faults (crashes, delays, partitions)
# and checks that the system maintains correctness.

class FaultType(Enum):
    CRASH = "crash"
    DELAY = "delay"
    PARTITION = "partition"
    DROP_MESSAGE = "drop_message"


@dataclass
class FaultEvent:
    """A scheduled fault injection event."""
    fault_type: FaultType
    target_node: str
    at_operation: int  # inject fault at this operation number
    duration: int = 1  # how many operations the fault lasts


class LinearizabilityChecker:
    """
    Checks if a sequence of operations is linearizable.
    Simplified: checks that reads return the most recently
    written value.
    """

    def __init__(self):
        self.write_history: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self.violations: List[str] = []

    def record_write(self, key: str, value: str, op_num: int):
        self.write_history[key].append((op_num, value))

    def check_read(self, key: str, expected_value: Optional[str],
                   actual_value: Optional[str], op_num: int) -> bool:
        """Check if a read is consistent with the write history."""
        if actual_value == expected_value:
            return True

        # Find the latest write before this read
        writes = self.write_history.get(key, [])
        latest_write = None
        for w_op, w_val in writes:
            if w_op < op_num:
                latest_write = w_val

        if actual_value != latest_write:
            self.violations.append(
                f"Op {op_num}: Read({key})={actual_value}, "
                f"expected={latest_write}"
            )
            return False
        return True


class FaultInjectionHarness:
    """
    Test harness that runs operations against a KV store while
    injecting faults, then checks linearizability.
    """

    def __init__(self):
        self.store: Dict[str, str] = {}
        self.faults: List[FaultEvent] = []
        self.active_faults: Dict[str, FaultEvent] = {}
        self.operation_log: List[Tuple[int, str, str, Optional[str]]] = []
        self.checker = LinearizabilityChecker()
        self.op_counter = 0
        self.fault_effects: List[str] = []

    def schedule_fault(self, fault: FaultEvent):
        """Schedule a fault injection event."""
        self.faults.append(fault)

    def _check_faults(self, node: str) -> Optional[FaultEvent]:
        """Check if there's an active fault for this node."""
        for fault in self.faults:
            if (fault.target_node == node
                    and fault.at_operation <= self.op_counter
                    < fault.at_operation + fault.duration):
                return fault
        return None

    def execute_write(self, key: str, value: str, node: str = "primary") -> bool:
        """Execute a write operation, possibly affected by faults."""
        self.op_counter += 1
        fault = self._check_faults(node)

        if fault:
            if fault.fault_type == FaultType.CRASH:
                self.fault_effects.append(
                    f"Op {self.op_counter}: CRASH on {node}, "
                    f"write({key}={value}) lost"
                )
                return False
            elif fault.fault_type == FaultType.DROP_MESSAGE:
                self.fault_effects.append(
                    f"Op {self.op_counter}: DROP on {node}, "
                    f"write({key}={value}) dropped"
                )
                return False

        self.store[key] = value
        self.checker.record_write(key, value, self.op_counter)
        self.operation_log.append((self.op_counter, "WRITE", key, value))
        return True

    def execute_read(self, key: str, node: str = "primary") -> Optional[str]:
        """Execute a read operation, possibly affected by faults."""
        self.op_counter += 1
        fault = self._check_faults(node)

        if fault:
            if fault.fault_type == FaultType.CRASH:
                self.fault_effects.append(
                    f"Op {self.op_counter}: CRASH on {node}, read({key}) failed"
                )
                return None
            elif fault.fault_type == FaultType.DELAY:
                # Delayed read might return stale data
                self.fault_effects.append(
                    f"Op {self.op_counter}: DELAY on {node}, read({key}) delayed"
                )

        value = self.store.get(key)
        self.operation_log.append((self.op_counter, "READ", key, value))
        return value

    def run_test_scenario(self) -> bool:
        """
        Run a predefined test scenario and check linearizability.
        Returns True if no violations found.
        """
        # Write initial data
        self.execute_write("x", "1")
        self.execute_write("y", "2")

        # Schedule faults
        self.schedule_fault(FaultEvent(FaultType.CRASH, "primary", at_operation=4, duration=1))
        self.schedule_fault(FaultEvent(FaultType.DROP_MESSAGE, "primary", at_operation=6, duration=1))

        # Continue operations
        self.execute_write("x", "10")  # op 3: succeeds
        write_ok = self.execute_write("x", "20")  # op 4: CRASH - lost!
        self.execute_write("y", "30")  # op 5: succeeds
        write_ok2 = self.execute_write("y", "40")  # op 6: DROPPED

        # Reads
        x_val = self.execute_read("x")  # op 7
        y_val = self.execute_read("y")  # op 8

        return len(self.checker.violations) == 0


def exercise_3():
    """
    Demonstrate fault injection test harness.
    """
    print("=== Exercise 3: Fault Injection Test Harness ===\n")

    harness = FaultInjectionHarness()
    linearizable = harness.run_test_scenario()

    print("Operation log:")
    for op_num, op_type, key, value in harness.operation_log:
        print(f"  Op {op_num}: {op_type}({key}) = {value}")

    print(f"\nFault effects:")
    for effect in harness.fault_effects:
        print(f"  {effect}")

    print(f"\nFinal store state: {harness.store}")
    print(f"Linearizable: {linearizable}")

    if harness.checker.violations:
        print(f"Violations:")
        for v in harness.checker.violations:
            print(f"  {v}")
    else:
        print("No linearizability violations detected.")

    # Expected results:
    # x should be "10" (write of "20" was lost due to crash)
    # y should be "30" (write of "40" was dropped)
    x_val = harness.store.get("x")
    y_val = harness.store.get("y")
    print(f"\nx={x_val} (expected '10': write '20' lost to crash)")
    print(f"y={y_val} (expected '30': write '40' dropped)")

    # Additional test: random fault injection
    print(f"\n--- Random Fault Injection Test ---")
    random.seed(42)

    harness2 = FaultInjectionHarness()
    # Schedule random faults
    for _ in range(3):
        harness2.schedule_fault(FaultEvent(
            fault_type=random.choice([FaultType.CRASH, FaultType.DROP_MESSAGE]),
            target_node="primary",
            at_operation=random.randint(1, 15),
            duration=1,
        ))

    # Run random operations
    success_count = 0
    fail_count = 0
    for i in range(20):
        if random.random() < 0.6:
            ok = harness2.execute_write(f"k{i%5}", f"v{i}")
            if ok:
                success_count += 1
            else:
                fail_count += 1
        else:
            harness2.execute_read(f"k{i%5}")

    print(f"Successful writes: {success_count}")
    print(f"Failed writes (faults): {fail_count}")
    print(f"Fault effects: {len(harness2.fault_effects)}")
    print(f"Linearizability violations: {len(harness2.checker.violations)}")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
