"""
Distributed Coordination Primitives

Implements distributed locks, barriers, fencing tokens, and leader
election. Demonstrates why naive distributed locking is unsafe and
how fencing tokens prevent split-brain scenarios.

Key concepts:
- Distributed locks with TTL (lease-based)
- Fencing tokens to prevent stale lock holders from writing
- Distributed barrier for synchronising phases
- Leader election with heartbeats
- Redlock algorithm analysis

Usage:
    python 17_coordination_primitives.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Distributed Lock with Fencing Tokens
# ---------------------------------------------------------------------------

@dataclass
class LockGrant:
    """A granted lock with a fencing token."""
    holder: str
    token: int
    acquired_at: float
    ttl: float

    def is_expired(self, current_time: float) -> bool:
        return current_time > self.acquired_at + self.ttl

    def __repr__(self) -> str:
        return f"Lock(holder={self.holder}, token={self.token}, ttl={self.ttl})"


class DistributedLockService:
    """A lock service that issues fencing tokens."""

    def __init__(self):
        self._locks: dict[str, LockGrant] = {}
        self._next_token: int = 0

    def acquire(self, resource: str, requester: str,
                current_time: float, ttl: float = 5.0) -> LockGrant | None:
        """Try to acquire a lock. Returns LockGrant or None."""
        existing = self._locks.get(resource)

        if existing and not existing.is_expired(current_time):
            if existing.holder == requester:
                return existing  # Re-entrant
            return None  # Lock held by someone else

        # Grant new lock with monotonically increasing fencing token
        self._next_token += 1
        grant = LockGrant(requester, self._next_token, current_time, ttl)
        self._locks[resource] = grant
        return grant

    def release(self, resource: str, requester: str) -> bool:
        """Release a lock. Only the holder can release."""
        existing = self._locks.get(resource)
        if existing and existing.holder == requester:
            del self._locks[resource]
            return True
        return False


class FencedStorage:
    """Storage that checks fencing tokens to prevent stale writes."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._max_token: dict[str, int] = {}  # key -> highest token seen

    def write(self, key: str, value: str, fencing_token: int) -> tuple[bool, str]:
        """Write only if fencing token is >= highest seen."""
        max_t = self._max_token.get(key, 0)
        if fencing_token < max_t:
            return False, f"REJECTED: token {fencing_token} < max {max_t}"
        self._max_token[key] = fencing_token
        self._store[key] = value
        return True, f"OK: wrote {key}='{value}' with token {fencing_token}"

    def read(self, key: str) -> str | None:
        return self._store.get(key)


# ---------------------------------------------------------------------------
# Distributed Barrier
# ---------------------------------------------------------------------------

class DistributedBarrier:
    """
    A distributed barrier that blocks until N participants arrive.
    Used to synchronise phases in distributed computations.
    """

    def __init__(self, name: str, required: int):
        self.name = name
        self.required = required
        self._arrived: set[str] = set()
        self.released = False
        self.log: list[str] = []

    def arrive(self, participant: str) -> bool:
        """Register arrival. Returns True if barrier is now released."""
        self._arrived.add(participant)
        self.log.append(
            f"{participant} arrived ({len(self._arrived)}/{self.required})")

        if len(self._arrived) >= self.required:
            self.released = True
            self.log.append(f"BARRIER RELEASED — all {self.required} arrived!")
            return True
        return False

    @property
    def waiting_count(self) -> int:
        return len(self._arrived)


# ---------------------------------------------------------------------------
# Leader Election with Heartbeats
# ---------------------------------------------------------------------------

@dataclass
class ElectionNode:
    """A node participating in leader election."""
    node_id: str
    is_leader: bool = False
    leader_id: str | None = None
    last_heartbeat: float = 0.0
    term: int = 0


class LeaderElection:
    """Simple leader election with heartbeat-based failure detection."""

    def __init__(self, node_ids: list[str], heartbeat_timeout: float = 3.0):
        self.nodes = {nid: ElectionNode(nid) for nid in node_ids}
        self.heartbeat_timeout = heartbeat_timeout
        self.log: list[str] = []

    def elect(self, current_time: float) -> str | None:
        """Run an election. Highest node ID wins (simplified bully)."""
        candidates = [
            nid for nid, node in self.nodes.items()
            if not node.is_leader or self._is_leader_stale(current_time)
        ]

        if not candidates:
            return None

        winner = max(candidates)
        for nid, node in self.nodes.items():
            node.is_leader = (nid == winner)
            node.leader_id = winner
            if nid == winner:
                node.last_heartbeat = current_time
                node.term += 1

        self.log.append(f"t={current_time:.1f}: Elected {winner} (term {self.nodes[winner].term})")
        return winner

    def heartbeat(self, leader_id: str, current_time: float) -> None:
        """Leader sends heartbeat to all followers."""
        leader = self.nodes.get(leader_id)
        if leader and leader.is_leader:
            leader.last_heartbeat = current_time
            for nid, node in self.nodes.items():
                if nid != leader_id:
                    node.last_heartbeat = current_time

    def _is_leader_stale(self, current_time: float) -> bool:
        for node in self.nodes.values():
            if node.is_leader:
                return (current_time - node.last_heartbeat) > self.heartbeat_timeout
        return True


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_fencing_tokens() -> None:
    """Show why fencing tokens are necessary."""
    print("=" * 70)
    print("Distributed Lock with Fencing Tokens")
    print("=" * 70)

    lock_svc = DistributedLockService()
    storage = FencedStorage()

    # Client A acquires lock
    grant_a = lock_svc.acquire("resource-1", "client-A", current_time=0.0, ttl=5.0)
    print(f"\n  t=0: Client A acquires lock: {grant_a}")

    # Client A does some slow work...
    # Lock expires at t=5

    # Client B acquires lock after expiry
    grant_b = lock_svc.acquire("resource-1", "client-B", current_time=6.0, ttl=5.0)
    print(f"  t=6: Client B acquires lock (A's expired): {grant_b}")

    # Client A (stale!) tries to write with old token
    ok, msg = storage.write("data", "from-A", grant_a.token)
    print(f"\n  Client A writes with token {grant_a.token}: {msg}")

    # Client B writes with newer token
    ok, msg = storage.write("data", "from-B", grant_b.token)
    print(f"  Client B writes with token {grant_b.token}: {msg}")

    # Client A tries again — REJECTED
    ok, msg = storage.write("data", "from-A-retry", grant_a.token)
    print(f"  Client A retries with token {grant_a.token}: {msg}")

    print(f"\n  Final value: {storage.read('data')}")
    print(f"\n  Without fencing tokens, Client A's stale write would")
    print(f"  overwrite Client B's valid write => DATA CORRUPTION!")


def demo_unsafe_lock() -> None:
    """Show the problem without fencing tokens."""
    print("\n" + "=" * 70)
    print("UNSAFE: Lock Without Fencing Tokens")
    print("=" * 70)

    print("""
  Timeline of the problem:
    t=0: Client A acquires lock (TTL=5s)
    t=1: Client A starts long operation
    t=5: Lock expires (Client A doesn't know!)
    t=6: Client B acquires lock
    t=7: Client B writes data = "B"
    t=8: Client A finishes, writes data = "A"  <-- STALE WRITE!
    t=9: Client B reads data = "A"              <-- WRONG!

  The lock's TTL expired, but Client A still wrote.
  Fencing tokens prevent this: storage rejects token < max seen.
""")


def demo_barrier() -> None:
    """Demonstrate distributed barrier."""
    print("=" * 70)
    print("Distributed Barrier")
    print("=" * 70)

    barrier = DistributedBarrier("map-reduce-phase-1", required=4)

    participants = ["mapper-0", "mapper-1", "mapper-2", "mapper-3"]

    print(f"\n  Barrier requires {barrier.required} participants:")
    for p in participants:
        released = barrier.arrive(p)
        status = "RELEASED" if released else "waiting"
        print(f"    {p} arrived => {status}")

    print(f"\n  All participants synchronized — proceed to next phase!")


def demo_leader_election() -> None:
    """Demonstrate leader election with failure detection."""
    print("\n" + "=" * 70)
    print("Leader Election with Heartbeat Failure Detection")
    print("=" * 70)

    election = LeaderElection(
        ["node-A", "node-B", "node-C"],
        heartbeat_timeout=3.0,
    )

    # Initial election
    leader = election.elect(current_time=0.0)
    print(f"\n  Initial election: {leader}")

    # Leader sends heartbeats
    election.heartbeat(leader, current_time=1.0)
    election.heartbeat(leader, current_time=2.0)
    print(f"  Heartbeats at t=1, t=2: cluster stable")

    # Leader fails (no heartbeat at t=3, detected at t=6)
    print(f"  t=3-5: Leader stops sending heartbeats")
    new_leader = election.elect(current_time=6.0)
    print(f"  t=6: New election triggered: {new_leader}")

    for line in election.log:
        print(f"    {line}")


def demo_redlock_analysis() -> None:
    """Analyze the Redlock algorithm and its limitations."""
    print("\n" + "=" * 70)
    print("Redlock Algorithm Analysis")
    print("=" * 70)

    print("""
  Redlock (Redis distributed lock):
  1. Get current time T1
  2. Try to acquire lock on N Redis instances (with short TTL)
  3. Lock acquired if: majority succeeded AND elapsed time < TTL
  4. Effective TTL = initial_TTL - elapsed_time

  Problems identified by Martin Kleppmann:
  - Clock jumps can cause overlapping lock ownership
  - GC pauses can cause client to hold expired lock
  - No fencing tokens! (unsafe without them)

  Simulation: 5 Redis instances, lock TTL = 10s
""")

    rng = random.Random(42)
    n_instances = 5
    ttl = 10.0
    majority = n_instances // 2 + 1

    # Simulate Redlock acquisition
    start_time = 0.0
    successes = 0
    total_elapsed = 0.0

    for i in range(n_instances):
        delay = rng.uniform(0.01, 0.1)  # Network delay
        total_elapsed += delay
        if rng.random() > 0.1:  # 90% success rate per instance
            successes += 1

    effective_ttl = ttl - total_elapsed
    acquired = successes >= majority and effective_ttl > 0

    print(f"  Instances responding: {successes}/{n_instances}")
    print(f"  Majority needed: {majority}")
    print(f"  Elapsed time: {total_elapsed:.3f}s")
    print(f"  Effective TTL: {effective_ttl:.3f}s")
    print(f"  Lock acquired: {acquired}")

    print(f"""
  Recommendation:
  - Use Redlock for efficiency (avoid duplicate work), not correctness
  - For correctness, use a consensus-based lock (ZooKeeper, etcd)
  - ALWAYS use fencing tokens for storage writes
""")


if __name__ == "__main__":
    demo_fencing_tokens()
    demo_unsafe_lock()
    demo_barrier()
    demo_leader_election()
    demo_redlock_analysis()
    print("Done.")
