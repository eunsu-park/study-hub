"""
Exercises for Lesson 14: Distributed Coordination Primitives
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import random
import threading
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


# === Exercise 1: Fencing Token Service ===
# Problem: Implement a fencing token service that issues monotonically
# increasing tokens. When a client acquires a lock, it gets a fencing
# token. Storage servers reject operations with tokens older than
# the latest they have seen, preventing stale clients from corrupting
# data after a lock timeout.

class FencingTokenService:
    """
    Issues monotonically increasing fencing tokens.
    Each token is globally unique and ordered.
    """

    def __init__(self):
        self._counter = 0
        self._lock_holders: Dict[str, int] = {}  # resource -> token

    def acquire_lock(self, resource: str, client_id: str) -> int:
        """
        Acquire a lock on a resource and get a fencing token.
        Returns the fencing token.
        """
        self._counter += 1
        token = self._counter
        self._lock_holders[resource] = token
        return token

    def release_lock(self, resource: str, token: int) -> bool:
        """Release a lock if the token matches."""
        if self._lock_holders.get(resource) == token:
            del self._lock_holders[resource]
            return True
        return False


class FencedStorageServer:
    """
    Storage server that validates fencing tokens.
    Rejects writes with tokens older than the latest seen.
    """

    def __init__(self, server_id: str):
        self.server_id = server_id
        self.data: Dict[str, any] = {}
        self.latest_token: Dict[str, int] = {}  # resource -> latest token

    def write(self, resource: str, value: any, fencing_token: int) -> bool:
        """
        Write a value if the fencing token is valid (>= latest seen).
        """
        latest = self.latest_token.get(resource, 0)
        if fencing_token < latest:
            return False  # stale token, reject
        self.latest_token[resource] = fencing_token
        self.data[resource] = value
        return True

    def read(self, resource: str) -> Optional[any]:
        return self.data.get(resource)


def exercise_1():
    """
    Demonstrate fencing tokens preventing stale client writes.
    """
    print("=== Exercise 1: Fencing Token Service ===\n")

    token_service = FencingTokenService()
    storage = FencedStorageServer("S1")

    # Client A acquires lock (token 1)
    token_a = token_service.acquire_lock("file.txt", "ClientA")
    print(f"ClientA acquires lock: token={token_a}")

    # Client A writes successfully
    ok = storage.write("file.txt", "data_from_A", token_a)
    print(f"ClientA writes: success={ok}")

    # Client A's lock expires (GC pause, network delay, etc.)
    # Client B acquires the same lock (token 2)
    token_b = token_service.acquire_lock("file.txt", "ClientB")
    print(f"\nClientB acquires lock (A's expired): token={token_b}")

    # Client B writes
    ok = storage.write("file.txt", "data_from_B", token_b)
    print(f"ClientB writes: success={ok}")

    # Client A wakes up and tries to write with its old token
    ok = storage.write("file.txt", "stale_data_from_A", token_a)
    print(f"\nClientA (stale) tries to write: success={ok}")
    print(f"Storage value: {storage.read('file.txt')}")

    assert storage.read("file.txt") == "data_from_B", "Stale write should be rejected"
    assert ok is False, "Stale token should be rejected"
    print("\nFencing tokens successfully prevented data corruption.")
    print()


# === Exercise 2: Redlock Algorithm with GC Pause Safety Issue ===
# Problem: Implement the Redlock distributed lock algorithm and
# demonstrate the safety issue when a client experiences a long
# GC pause after acquiring the lock.

class RedisInstance:
    """Simulated Redis instance for Redlock."""

    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.locks: Dict[str, Tuple[str, float]] = {}  # key -> (owner, expiry)
        self.is_alive = True

    def set_lock(
        self, key: str, owner: str, ttl_ms: float, current_time: float
    ) -> bool:
        """SET key owner NX PX ttl."""
        if not self.is_alive:
            return False
        if key in self.locks:
            _, expiry = self.locks[key]
            if current_time < expiry:
                return False  # already locked
        self.locks[key] = (owner, current_time + ttl_ms)
        return True

    def release_lock(self, key: str, owner: str) -> bool:
        """Release lock only if we are the owner."""
        if key in self.locks and self.locks[key][0] == owner:
            del self.locks[key]
            return True
        return False


class Redlock:
    """
    Redlock distributed lock algorithm.
    Acquires lock on majority of Redis instances.
    """

    def __init__(self, instances: List[RedisInstance], ttl_ms: float = 10000):
        self.instances = instances
        self.ttl_ms = ttl_ms
        self.quorum = len(instances) // 2 + 1

    def acquire(
        self, resource: str, owner: str, current_time: float
    ) -> Tuple[bool, float]:
        """
        Try to acquire the lock on the majority.
        Returns (success, remaining_ttl_ms).
        """
        start_time = current_time
        acquired = 0

        for inst in self.instances:
            if inst.set_lock(resource, owner, self.ttl_ms, current_time):
                acquired += 1

        # Clock drift allowance
        elapsed = current_time - start_time  # in simulation, this is 0
        drift = self.ttl_ms * 0.01  # 1% drift factor
        validity_time = self.ttl_ms - elapsed - drift

        if acquired >= self.quorum and validity_time > 0:
            return (True, validity_time)

        # Failed: release all
        for inst in self.instances:
            inst.release_lock(resource, owner)
        return (False, 0)

    def release(self, resource: str, owner: str):
        for inst in self.instances:
            inst.release_lock(resource, owner)


def exercise_2():
    """
    Demonstrate Redlock and the GC pause safety issue.
    """
    print("=== Exercise 2: Redlock with GC Pause Issue ===\n")

    instances = [RedisInstance(f"Redis{i}") for i in range(5)]
    redlock = Redlock(instances, ttl_ms=10000)

    # Client A acquires lock
    success, ttl = redlock.acquire("resource:1", "ClientA", current_time=0)
    print(f"ClientA acquires lock: success={success}, remaining_ttl={ttl:.0f}ms")

    # Client A experiences a long GC pause (15 seconds)
    gc_pause_ms = 15000
    print(f"\nClientA enters GC pause for {gc_pause_ms}ms...")
    print(f"Lock TTL is 10000ms - lock will expire during GC pause!")

    # Time advances past lock expiry
    current_time = gc_pause_ms

    # Client B acquires the expired lock
    success_b, ttl_b = redlock.acquire("resource:1", "ClientB", current_time)
    print(f"\nClientB acquires lock at t={current_time}ms: success={success_b}")

    # Client A wakes up from GC, thinks it still holds the lock
    print(f"ClientA wakes up at t={current_time}ms, thinks it has the lock")
    print(f"  Client A's lock expired at t=10000ms")
    print(f"  Client B acquired lock at t={current_time}ms")
    print(f"  BOTH clients think they hold the lock!")

    print(f"\nSafety issue:")
    print(f"  Without fencing tokens, ClientA can corrupt data")
    print(f"  after its lock expired during the GC pause.")
    print(f"  Solution: Use fencing tokens (Exercise 1) together")
    print(f"  with distributed locks.")
    print()


# === Exercise 3: Snowflake ID Generator ===
# Problem: Implement Twitter's Snowflake ID generator that produces
# globally unique, roughly time-ordered 64-bit IDs. The ID consists
# of: timestamp (41 bits) | datacenter (5 bits) | machine (5 bits)
# | sequence (12 bits).

class SnowflakeIDGenerator:
    """
    Snowflake-style distributed ID generator.

    64-bit ID layout:
    - 1 bit:  sign (always 0)
    - 41 bits: timestamp in milliseconds (custom epoch)
    - 5 bits:  datacenter ID (0-31)
    - 5 bits:  machine ID (0-31)
    - 12 bits: sequence number (0-4095)
    """

    EPOCH = 1288834974657  # Twitter snowflake epoch (Nov 4, 2010)
    DATACENTER_BITS = 5
    MACHINE_BITS = 5
    SEQUENCE_BITS = 12

    MAX_DATACENTER = (1 << DATACENTER_BITS) - 1
    MAX_MACHINE = (1 << MACHINE_BITS) - 1
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1

    MACHINE_SHIFT = SEQUENCE_BITS
    DATACENTER_SHIFT = SEQUENCE_BITS + MACHINE_BITS
    TIMESTAMP_SHIFT = SEQUENCE_BITS + MACHINE_BITS + DATACENTER_BITS

    def __init__(self, datacenter_id: int, machine_id: int):
        if datacenter_id > self.MAX_DATACENTER or datacenter_id < 0:
            raise ValueError(f"Datacenter ID must be 0-{self.MAX_DATACENTER}")
        if machine_id > self.MAX_MACHINE or machine_id < 0:
            raise ValueError(f"Machine ID must be 0-{self.MAX_MACHINE}")

        self.datacenter_id = datacenter_id
        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = -1

    def _current_millis(self) -> int:
        return int(time.time() * 1000)

    def generate(self, timestamp_ms: Optional[int] = None) -> int:
        """
        Generate a unique Snowflake ID.

        Args:
            timestamp_ms: Optional override for testing.
        """
        ts = timestamp_ms if timestamp_ms is not None else self._current_millis()

        if ts < self.last_timestamp:
            raise RuntimeError(
                f"Clock moved backwards: {self.last_timestamp} -> {ts}"
            )

        if ts == self.last_timestamp:
            self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE
            if self.sequence == 0:
                # Sequence exhausted, wait for next millisecond
                while ts <= self.last_timestamp:
                    ts = self._current_millis() if timestamp_ms is None else ts + 1
        else:
            self.sequence = 0

        self.last_timestamp = ts

        snowflake_id = (
            ((ts - self.EPOCH) << self.TIMESTAMP_SHIFT)
            | (self.datacenter_id << self.DATACENTER_SHIFT)
            | (self.machine_id << self.MACHINE_SHIFT)
            | self.sequence
        )
        return snowflake_id

    @classmethod
    def parse(cls, snowflake_id: int) -> Dict[str, int]:
        """Parse a Snowflake ID into its components."""
        sequence = snowflake_id & cls.MAX_SEQUENCE
        machine = (snowflake_id >> cls.MACHINE_SHIFT) & cls.MAX_MACHINE
        datacenter = (snowflake_id >> cls.DATACENTER_SHIFT) & cls.MAX_DATACENTER
        timestamp = (snowflake_id >> cls.TIMESTAMP_SHIFT) + cls.EPOCH
        return {
            "timestamp_ms": timestamp,
            "datacenter_id": datacenter,
            "machine_id": machine,
            "sequence": sequence,
        }


def exercise_3():
    """
    Demonstrate Snowflake ID generation and parsing.
    """
    print("=== Exercise 3: Snowflake ID Generator ===\n")

    gen1 = SnowflakeIDGenerator(datacenter_id=1, machine_id=5)
    gen2 = SnowflakeIDGenerator(datacenter_id=2, machine_id=10)

    # Generate IDs with fixed timestamp for reproducibility
    base_ts = 1700000000000  # Nov 2023

    print("Generator 1 (DC=1, Machine=5):")
    ids1 = []
    for i in range(5):
        sid = gen1.generate(timestamp_ms=base_ts + i)
        parsed = SnowflakeIDGenerator.parse(sid)
        ids1.append(sid)
        print(f"  ID: {sid:>20d} | ts={parsed['timestamp_ms']} "
              f"dc={parsed['datacenter_id']} m={parsed['machine_id']} "
              f"seq={parsed['sequence']}")

    print(f"\nGenerator 2 (DC=2, Machine=10):")
    ids2 = []
    for i in range(5):
        sid = gen2.generate(timestamp_ms=base_ts + i)
        parsed = SnowflakeIDGenerator.parse(sid)
        ids2.append(sid)
        print(f"  ID: {sid:>20d} | ts={parsed['timestamp_ms']} "
              f"dc={parsed['datacenter_id']} m={parsed['machine_id']} "
              f"seq={parsed['sequence']}")

    # Verify uniqueness
    all_ids = ids1 + ids2
    assert len(all_ids) == len(set(all_ids)), "All IDs must be unique"
    print(f"\nAll {len(all_ids)} IDs are unique.")

    # Verify ordering within same generator
    assert all(ids1[i] < ids1[i + 1] for i in range(len(ids1) - 1))
    print("IDs from same generator are monotonically increasing.")

    # Same-millisecond IDs
    print(f"\nSame-millisecond IDs:")
    gen3 = SnowflakeIDGenerator(datacenter_id=0, machine_id=0)
    same_ms_ids = [gen3.generate(timestamp_ms=base_ts) for _ in range(5)]
    for sid in same_ms_ids:
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  ID: {sid:>20d} | seq={parsed['sequence']}")

    assert len(set(same_ms_ids)) == 5, "Same-ms IDs must be unique (via sequence)"
    print("Same-millisecond IDs differentiated by sequence number.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
