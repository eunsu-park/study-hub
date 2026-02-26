"""
Two-Phase Locking (2PL)

Demonstrates:
- Strict 2PL protocol
- Lock types (shared/exclusive)
- Deadlock detection via wait-for graph
- Lock escalation

Theory:
- 2PL ensures serializability: transactions acquire locks in a
  growing phase and release in a shrinking phase.
- Strict 2PL: all locks held until commit/abort. Prevents
  cascading aborts.
- Shared lock (S): multiple readers allowed.
- Exclusive lock (X): only one writer, no readers.
- Deadlock: cycle in the wait-for graph.

Adapted from Database Theory Lesson 11.
"""

from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field


class LockType(Enum):
    SHARED = "S"
    EXCLUSIVE = "X"


@dataclass
class Lock:
    resource: str
    lock_type: LockType
    txn_id: str


class LockManager:
    """Lock manager with strict 2PL and deadlock detection."""

    def __init__(self):
        self.locks: dict[str, list[Lock]] = defaultdict(list)
        self.waiting: dict[str, tuple[str, LockType]] = {}  # txn → (resource, type)
        self.txn_locks: dict[str, list[Lock]] = defaultdict(list)
        self.log: list[str] = []

    def acquire(self, txn_id: str, resource: str,
                lock_type: LockType) -> bool:
        """Try to acquire a lock. Returns True if granted."""
        current_locks = self.locks[resource]

        # Check compatibility
        if lock_type == LockType.SHARED:
            # S lock: compatible with other S locks
            if all(l.lock_type == LockType.SHARED or l.txn_id == txn_id
                   for l in current_locks):
                lock = Lock(resource, lock_type, txn_id)
                current_locks.append(lock)
                self.txn_locks[txn_id].append(lock)
                self.log.append(
                    f"  {txn_id}: GRANT S-lock on {resource}")
                return True

        elif lock_type == LockType.EXCLUSIVE:
            # X lock: only if no other locks
            other_locks = [l for l in current_locks if l.txn_id != txn_id]
            if not other_locks:
                # Check if upgrading S → X
                own_locks = [l for l in current_locks if l.txn_id == txn_id]
                for l in own_locks:
                    current_locks.remove(l)
                    self.txn_locks[txn_id].remove(l)

                lock = Lock(resource, lock_type, txn_id)
                current_locks.append(lock)
                self.txn_locks[txn_id].append(lock)
                self.log.append(
                    f"  {txn_id}: GRANT X-lock on {resource}")
                return True

        # Cannot grant — transaction must wait
        self.waiting[txn_id] = (resource, lock_type)
        self.log.append(
            f"  {txn_id}: WAIT for {lock_type.value}-lock on {resource}")
        return False

    def release_all(self, txn_id: str) -> list[str]:
        """Release all locks for a transaction (strict 2PL: at commit)."""
        released = []
        for lock in self.txn_locks.get(txn_id, []):
            if lock in self.locks[lock.resource]:
                self.locks[lock.resource].remove(lock)
                released.append(lock.resource)
        self.txn_locks[txn_id] = []
        self.waiting.pop(txn_id, None)
        self.log.append(f"  {txn_id}: RELEASE all locks")

        # Try to grant waiting transactions
        granted = []
        for wait_txn, (resource, lt) in list(self.waiting.items()):
            if self.acquire(wait_txn, resource, lt):
                del self.waiting[wait_txn]
                granted.append(wait_txn)

        return granted

    def detect_deadlock(self) -> list[list[str]] | None:
        """Detect deadlock cycles in wait-for graph."""
        # Build wait-for graph
        graph: dict[str, set[str]] = defaultdict(set)
        for wait_txn, (resource, _) in self.waiting.items():
            holders = [l.txn_id for l in self.locks[resource]
                       if l.txn_id != wait_txn]
            for h in holders:
                graph[wait_txn].add(h)

        # DFS cycle detection
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    start = path.index(neighbor)
                    cycles.append(path[start:] + [neighbor])

            path.pop()
            rec_stack.discard(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles if cycles else None

    def print_state(self) -> None:
        print(f"    Lock table:")
        for resource, locks in sorted(self.locks.items()):
            if locks:
                holders = [f"{l.txn_id}({l.lock_type.value})"
                           for l in locks]
                print(f"      {resource}: {', '.join(holders)}")
        if self.waiting:
            print(f"    Waiting:")
            for txn, (res, lt) in self.waiting.items():
                print(f"      {txn} → {lt.value}-lock on {res}")


# ── Demos ──────────────────────────────────────────────────────────────

def demo_basic_2pl():
    print("=" * 60)
    print("STRICT TWO-PHASE LOCKING")
    print("=" * 60)

    lm = LockManager()

    # T1: read A, write B
    print(f"\n  T1: READ(A), WRITE(B)")
    print(f"  T2: READ(B), WRITE(A)\n")

    # T1 reads A
    lm.acquire("T1", "A", LockType.SHARED)
    # T2 reads B
    lm.acquire("T2", "B", LockType.SHARED)
    # T1 writes B → needs X-lock, T2 holds S-lock
    lm.acquire("T1", "B", LockType.EXCLUSIVE)
    # T2 writes A → needs X-lock, T1 holds S-lock
    lm.acquire("T2", "A", LockType.EXCLUSIVE)

    print(f"\n  Lock state:")
    lm.print_state()

    # Deadlock detection
    cycles = lm.detect_deadlock()
    if cycles:
        print(f"\n  DEADLOCK detected! Cycles: {cycles}")
        # Abort T2 to break deadlock
        print(f"  → Aborting T2")
        lm.release_all("T2")
        print(f"\n  After T2 abort:")
        lm.print_state()

    # T1 can now commit
    lm.release_all("T1")
    print(f"\n  T1 committed. All locks released.")


def demo_no_deadlock():
    print("\n" + "=" * 60)
    print("SUCCESSFUL 2PL EXECUTION")
    print("=" * 60)

    lm = LockManager()

    print(f"\n  T1: READ(A), WRITE(A)")
    print(f"  T2: READ(B), WRITE(B)")
    print(f"  (No conflict — different resources)\n")

    lm.acquire("T1", "A", LockType.SHARED)
    lm.acquire("T2", "B", LockType.SHARED)
    lm.acquire("T1", "A", LockType.EXCLUSIVE)  # Upgrade S→X
    lm.acquire("T2", "B", LockType.EXCLUSIVE)

    lm.print_state()
    print(f"\n  No deadlock — both can proceed")

    lm.release_all("T1")
    lm.release_all("T2")


def demo_lock_upgrade():
    print("\n" + "=" * 60)
    print("LOCK UPGRADE (S → X)")
    print("=" * 60)

    lm = LockManager()

    print(f"\n  T1 reads A, then decides to update A\n")

    lm.acquire("T1", "A", LockType.SHARED)
    print(f"  State after S-lock:")
    lm.print_state()

    lm.acquire("T1", "A", LockType.EXCLUSIVE)
    print(f"\n  State after upgrade to X-lock:")
    lm.print_state()

    lm.release_all("T1")


if __name__ == "__main__":
    demo_basic_2pl()
    demo_no_deadlock()
    demo_lock_upgrade()
