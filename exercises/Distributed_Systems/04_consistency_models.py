"""
Exercises for Lesson 04: Consistency Models
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

from typing import Dict, List, Optional, Set, Tuple
from itertools import permutations
from collections import defaultdict


# === Exercise 1: Linearizability Checker ===
# Problem: Given a set of read/write operations with invocation and
# response times, determine if the history is linearizable.
# A history is linearizable if operations can be ordered sequentially
# such that:
# 1. The order respects real-time ordering (non-overlapping ops)
# 2. Each read returns the value of the most recent write in that order

class Operation:
    """Represents a read or write operation with timing."""

    def __init__(
        self, op_id: int, op_type: str, key: str,
        value: Optional[int], ret_value: Optional[int],
        invoke_time: int, response_time: int,
    ):
        self.op_id = op_id
        self.op_type = op_type  # "write" or "read"
        self.key = key
        self.value = value          # value written (for writes)
        self.ret_value = ret_value  # value returned (for reads)
        self.invoke_time = invoke_time
        self.response_time = response_time

    def __repr__(self):
        if self.op_type == "write":
            return f"W({self.key}={self.value})[{self.invoke_time},{self.response_time}]"
        return f"R({self.key})={self.ret_value}[{self.invoke_time},{self.response_time}]"


def must_precede(op1: Operation, op2: Operation) -> bool:
    """op1 must come before op2 if op1 completes before op2 starts."""
    return op1.response_time < op2.invoke_time


def check_linearizable(operations: List[Operation]) -> bool:
    """
    Check if a history of operations on a single key is linearizable.
    Uses brute-force: try all valid orderings.

    Returns True if any linearization exists.
    """
    n = len(operations)
    if n == 0:
        return True

    for perm in permutations(range(n)):
        ordering = [operations[i] for i in perm]

        # Check real-time constraint
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if must_precede(ordering[j], ordering[i]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Check sequential semantics
        current_value = None
        seq_valid = True
        for op in ordering:
            if op.op_type == "write":
                current_value = op.value
            else:  # read
                if op.ret_value != current_value:
                    seq_valid = False
                    break

        if seq_valid:
            return True

    return False


def exercise_1():
    """
    Check linearizability of operation histories.
    """
    print("=== Exercise 1: Linearizability Checker ===\n")

    # History 1: Linearizable
    # W(x=1)[0,2], R(x)=1[1,3], R(x)=1[3,4]
    history1 = [
        Operation(0, "write", "x", 1, None, 0, 2),
        Operation(1, "read", "x", None, 1, 1, 3),
        Operation(2, "read", "x", None, 1, 3, 4),
    ]

    result1 = check_linearizable(history1)
    print(f"History 1: {[str(op) for op in history1]}")
    print(f"  Linearizable: {result1}")
    assert result1 is True

    # History 2: NOT linearizable
    # W(x=1)[0,1], W(x=2)[2,3], R(x)=1[4,5]
    # Read returns 1 after write of 2 completed -> not linearizable
    history2 = [
        Operation(0, "write", "x", 1, None, 0, 1),
        Operation(1, "write", "x", 2, None, 2, 3),
        Operation(2, "read", "x", None, 1, 4, 5),
    ]

    result2 = check_linearizable(history2)
    print(f"\nHistory 2: {[str(op) for op in history2]}")
    print(f"  Linearizable: {result2}")
    assert result2 is False

    # History 3: Linearizable with concurrent ops
    # W(x=1)[0,3], W(x=2)[1,4], R(x)=2[5,6]
    # Both writes overlap, read sees 2 -> linearize as W1, W2, R
    history3 = [
        Operation(0, "write", "x", 1, None, 0, 3),
        Operation(1, "write", "x", 2, None, 1, 4),
        Operation(2, "read", "x", None, 2, 5, 6),
    ]

    result3 = check_linearizable(history3)
    print(f"\nHistory 3: {[str(op) for op in history3]}")
    print(f"  Linearizable: {result3}")
    assert result3 is True

    print("\nAll linearizability checks passed.")
    print()


# === Exercise 2: Sequential Consistency Simulator ===
# Problem: Implement a sequentially consistent memory that allows
# operations from different processes to be interleaved, as long as
# each process's operations appear in program order.

class SequentialMemory:
    """
    Sequentially consistent memory simulator.

    Maintains a global memory and per-process operation queues.
    Processes submit operations that are executed in some interleaving
    that preserves per-process order.
    """

    def __init__(self):
        self.memory: Dict[str, int] = {}
        self.process_ops: Dict[str, List] = defaultdict(list)
        self.execution_log: List[str] = []

    def submit_write(self, process: str, key: str, value: int):
        """Queue a write operation from a process."""
        self.process_ops[process].append(("write", key, value))

    def submit_read(self, process: str, key: str):
        """Queue a read operation from a process."""
        self.process_ops[process].append(("read", key))

    def execute_sequential(self) -> List[Tuple[str, str, Optional[int]]]:
        """
        Execute all operations in a sequentially consistent order.
        Returns list of (process, operation_desc, read_result).
        """
        results = []
        # Track current index per process
        indices = {p: 0 for p in self.process_ops}

        while any(
            indices[p] < len(ops)
            for p, ops in self.process_ops.items()
        ):
            # Pick a process that has remaining operations (round-robin)
            for p in sorted(self.process_ops.keys()):
                if indices[p] < len(self.process_ops[p]):
                    op = self.process_ops[p][indices[p]]
                    indices[p] += 1

                    if op[0] == "write":
                        self.memory[op[1]] = op[2]
                        results.append((p, f"W({op[1]}={op[2]})", None))
                    else:  # read
                        val = self.memory.get(op[1])
                        results.append((p, f"R({op[1]})", val))
                    break

        return results

    def find_valid_interleaving(
        self, expected_reads: Dict[int, int]
    ) -> Optional[List[Tuple]]:
        """
        Find an interleaving that matches expected read results.
        Uses backtracking.
        """
        indices = {p: 0 for p in self.process_ops}
        result = []
        read_counter = [0]

        def backtrack():
            if all(
                indices[p] >= len(ops)
                for p, ops in self.process_ops.items()
            ):
                return True

            for p in sorted(self.process_ops.keys()):
                if indices[p] < len(self.process_ops[p]):
                    op = self.process_ops[p][indices[p]]
                    indices[p] += 1

                    if op[0] == "write":
                        old = self.memory.get(op[1])
                        self.memory[op[1]] = op[2]
                        result.append((p, f"W({op[1]}={op[2]})", None))
                        if backtrack():
                            return True
                        result.pop()
                        if old is None:
                            del self.memory[op[1]]
                        else:
                            self.memory[op[1]] = old
                    else:
                        val = self.memory.get(op[1])
                        ridx = read_counter[0]
                        if ridx in expected_reads and val != expected_reads[ridx]:
                            indices[p] -= 1
                            continue
                        read_counter[0] += 1
                        result.append((p, f"R({op[1]})", val))
                        if backtrack():
                            return True
                        result.pop()
                        read_counter[0] -= 1

                    indices[p] -= 1

            return False

        self.memory.clear()
        if backtrack():
            return list(result)
        return None


def exercise_2():
    """
    Demonstrate sequential consistency with different valid interleavings.
    """
    print("=== Exercise 2: Sequential Consistency Simulator ===\n")

    mem = SequentialMemory()

    # Process P1: W(x=1), W(x=3)
    # Process P2: W(x=2), R(x)
    mem.submit_write("P1", "x", 1)
    mem.submit_write("P1", "x", 3)
    mem.submit_write("P2", "x", 2)
    mem.submit_read("P2", "x")

    print("Process operations:")
    print("  P1: W(x=1), W(x=3)")
    print("  P2: W(x=2), R(x)")

    results = mem.execute_sequential()
    print("\nOne valid sequential execution (round-robin):")
    for proc, op, val in results:
        if val is not None:
            print(f"  {proc}: {op} -> {val}")
        else:
            print(f"  {proc}: {op}")

    print("\nNote: R(x) could return 1, 2, or 3 depending on interleaving.")
    print("All are sequentially consistent as long as per-process order is preserved.")
    print("  P1's W(x=1) must precede W(x=3)")
    print("  P2's W(x=2) must precede R(x)")
    print()


# === Exercise 3: Session Guarantees ===
# Problem: Implement session guarantees (read-your-writes, monotonic
# reads) over an eventually consistent store. Each client session
# tracks metadata to ensure these guarantees are met.

class EventuallyConsistentStore:
    """
    Simulates an eventually consistent store with multiple replicas.
    Each replica may have different versions of data.
    """

    def __init__(self, num_replicas: int):
        self.replicas: List[Dict[str, Tuple[int, int]]] = [
            {} for _ in range(num_replicas)
        ]  # key -> (value, write_timestamp)

    def write(self, replica_id: int, key: str, value: int, timestamp: int):
        """Write to a specific replica."""
        current = self.replicas[replica_id].get(key)
        if current is None or timestamp > current[1]:
            self.replicas[replica_id][key] = (value, timestamp)

    def read(self, replica_id: int, key: str) -> Optional[Tuple[int, int]]:
        """Read from a specific replica. Returns (value, timestamp)."""
        return self.replicas[replica_id].get(key)

    def propagate(self, from_replica: int, to_replica: int, key: str):
        """Propagate a key from one replica to another."""
        data = self.replicas[from_replica].get(key)
        if data:
            current = self.replicas[to_replica].get(key)
            if current is None or data[1] > current[1]:
                self.replicas[to_replica][key] = data


class SessionClient:
    """
    Client session that enforces read-your-writes and monotonic reads
    over an eventually consistent store.
    """

    def __init__(self, store: EventuallyConsistentStore):
        self.store = store
        self.num_replicas = len(store.replicas)
        # Track the minimum timestamp we need for each key
        self.write_timestamps: Dict[str, int] = {}  # read-your-writes
        self.read_timestamps: Dict[str, int] = {}   # monotonic reads
        self.current_ts = 0

    def session_write(self, key: str, value: int, replica_id: int):
        """Write with session tracking."""
        self.current_ts += 1
        self.store.write(replica_id, key, value, self.current_ts)
        self.write_timestamps[key] = self.current_ts
        return self.current_ts

    def session_read(self, key: str, replica_id: int) -> Optional[int]:
        """
        Read with session guarantees:
        - Read-your-writes: result must be at least as recent as
          our last write.
        - Monotonic reads: result must be at least as recent as
          our last read.
        """
        min_ts = max(
            self.write_timestamps.get(key, 0),
            self.read_timestamps.get(key, 0),
        )

        result = self.store.read(replica_id, key)
        if result is None or result[1] < min_ts:
            # This replica is stale. Try other replicas.
            for rid in range(self.num_replicas):
                candidate = self.store.read(rid, key)
                if candidate and candidate[1] >= min_ts:
                    self.read_timestamps[key] = candidate[1]
                    return candidate[0]
            # No replica has fresh enough data
            return None

        self.read_timestamps[key] = result[1]
        return result[0]


def exercise_3():
    """
    Demonstrate session guarantees preventing stale reads.
    """
    print("=== Exercise 3: Session Guarantees ===\n")

    store = EventuallyConsistentStore(num_replicas=3)
    client = SessionClient(store)

    # Write x=10 to replica 0
    ts = client.session_write("x", 10, replica_id=0)
    print(f"Client writes x=10 to replica 0 (ts={ts})")

    # Try to read from replica 1 (stale - hasn't propagated yet)
    print(f"\nReplica 1 state: {store.read(1, 'x')}")
    result = client.session_read("x", replica_id=1)
    print(f"Session read from replica 1: {result}")
    print("  (Session detected stale replica, found fresh data on replica 0)")
    assert result == 10, "Read-your-writes should see value 10"

    # Write x=20 to replica 0
    ts = client.session_write("x", 20, replica_id=0)
    print(f"\nClient writes x=20 to replica 0 (ts={ts})")

    # Propagate old value to replica 2
    store.write(2, "x", 10, 1)  # replica 2 has stale value
    print(f"Replica 2 has stale x=10 (ts=1)")

    # Monotonic read should not return the stale value
    result = client.session_read("x", replica_id=2)
    print(f"Session read from replica 2: {result}")
    print("  (Monotonic reads prevented reading stale ts=1)")
    assert result == 20, "Monotonic reads should see value 20"

    print("\nSession guarantees working correctly.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
