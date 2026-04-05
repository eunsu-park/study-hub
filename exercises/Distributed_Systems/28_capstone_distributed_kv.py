"""
Exercises for Lesson 28: Capstone — Building a Production Distributed KV Store
Topic: Distributed_Systems

Solutions to final challenges from the lesson.
"""

import random
import time
import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# === Challenge 1: Shard Split ===
def challenge_1():
    """Implement online shard splitting."""
    print("=== Challenge 1: Online Shard Split ===\n")

    class Shard:
        def __init__(self, shard_id, range_start, range_end):
            self.shard_id = shard_id
            self.range_start = range_start
            self.range_end = range_end
            self.data = {}

        def put(self, key, value):
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
            if self.range_start <= h < self.range_end:
                self.data[key] = value
                return True
            return False

    class ShardSplitter:
        def __init__(self):
            self.shards = {}

        def add_shard(self, shard):
            self.shards[shard.shard_id] = shard

        def split(self, shard_id):
            """Split a shard into two at the midpoint."""
            shard = self.shards[shard_id]
            mid = (shard.range_start + shard.range_end) // 2

            # Create two new shards
            left = Shard(f"{shard_id}_L", shard.range_start, mid)
            right = Shard(f"{shard_id}_R", mid, shard.range_end)

            # Redistribute data
            for key, value in shard.data.items():
                h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
                if h < mid:
                    left.data[key] = value
                else:
                    right.data[key] = value

            # Replace old shard
            del self.shards[shard_id]
            self.shards[left.shard_id] = left
            self.shards[right.shard_id] = right

            return left, right

    splitter = ShardSplitter()
    shard = Shard("S0", 0, 2**32)

    # Add data
    for i in range(1000):
        shard.put(f"key-{i}", f"val-{i}")

    splitter.add_shard(shard)
    print(f"  Before split: 1 shard, {len(shard.data)} keys")

    left, right = splitter.split("S0")
    print(f"  After split: 2 shards")
    print(f"    Left ({left.shard_id}): {len(left.data)} keys, "
          f"range=[{left.range_start}, {left.range_end})")
    print(f"    Right ({right.shard_id}): {len(right.data)} keys, "
          f"range=[{right.range_start}, {right.range_end})")
    print(f"    Total keys preserved: {len(left.data) + len(right.data)}")


challenge_1()


# === Challenge 2: Cross-Shard 2PC ===
def challenge_2():
    """Implement 2PC for cross-shard transactions."""
    print("\n=== Challenge 2: Cross-Shard 2PC ===\n")

    class TxCoordinator:
        def __init__(self):
            self.tx_log = []

        def execute_2pc(self, participants, operations):
            tx_id = f"tx-{random.randint(1000, 9999)}"
            print(f"  Transaction {tx_id}:")

            # Phase 1: Prepare
            votes = {}
            for pid in participants:
                # Simulate: each participant votes
                vote = random.random() > 0.1  # 90% success
                votes[pid] = vote
                print(f"    PREPARE {pid}: {'YES' if vote else 'NO'}")

            # Decision
            if all(votes.values()):
                # Phase 2: Commit
                self.tx_log.append((tx_id, "COMMIT"))
                for pid in participants:
                    print(f"    COMMIT {pid}: OK")
                return True
            else:
                # Phase 2: Abort
                self.tx_log.append((tx_id, "ABORT"))
                for pid in participants:
                    print(f"    ABORT {pid}: rolled back")
                return False

    coord = TxCoordinator()

    # Transaction spanning 3 shards
    result = coord.execute_2pc(
        ["shard-0", "shard-1", "shard-2"],
        {"shard-0": ("put", "x", "1"),
         "shard-1": ("put", "y", "2"),
         "shard-2": ("put", "z", "3")},
    )
    print(f"  Result: {'COMMITTED' if result else 'ABORTED'}")


challenge_2()


# === Challenge 3: Invariant Testing ===
def challenge_3():
    """Test invariants of a sharded KV store."""
    print("\n=== Challenge 3: Invariant Testing ===\n")

    class ShardedKV:
        def __init__(self, num_shards):
            self.shards = {i: {} for i in range(num_shards)}
            self.num_shards = num_shards

        def _shard_for(self, key):
            return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_shards

        def put(self, key, value):
            sid = self._shard_for(key)
            self.shards[sid][key] = value

        def get(self, key):
            sid = self._shard_for(key)
            return self.shards[sid].get(key)

        def check_invariants(self):
            """Verify system invariants."""
            errors = []

            # Invariant 1: Each key on exactly one shard
            key_locations = defaultdict(list)
            for sid, data in self.shards.items():
                for key in data:
                    key_locations[key].append(sid)
                    expected_shard = self._shard_for(key)
                    if sid != expected_shard:
                        errors.append(f"Key {key} on shard {sid}, expected {expected_shard}")

            # Invariant 2: No duplicate keys across shards
            for key, shards in key_locations.items():
                if len(shards) > 1:
                    errors.append(f"Key {key} on multiple shards: {shards}")

            return errors

    kv = ShardedKV(4)
    for i in range(500):
        kv.put(f"key-{i}", f"val-{i}")

    errors = kv.check_invariants()
    print(f"  Keys: {sum(len(s) for s in kv.shards.values())}")
    print(f"  Invariant violations: {len(errors)}")

    # Verify read-your-write
    ryw_violations = 0
    for i in range(500):
        key = f"key-{i}"
        if kv.get(key) != f"val-{i}":
            ryw_violations += 1
    print(f"  Read-your-write violations: {ryw_violations}")


challenge_3()


# === Challenge 4: Performance Benchmark ===
def challenge_4():
    """Benchmark the distributed KV store."""
    print("\n=== Challenge 4: Performance Benchmark ===\n")

    class BenchmarkKV:
        def __init__(self, num_shards):
            self.shards = {i: {} for i in range(num_shards)}
            self.num_shards = num_shards

        def _shard(self, key):
            return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_shards

        def put(self, key, value):
            self.shards[self._shard(key)][key] = value

        def get(self, key):
            return self.shards[self._shard(key)].get(key)

    for num_shards in [1, 4, 8, 16]:
        kv = BenchmarkKV(num_shards)

        # Write benchmark
        start = time.time()
        for i in range(100000):
            kv.put(f"key-{i}", f"value-{i}")
        write_time = time.time() - start

        # Read benchmark
        start = time.time()
        for i in range(100000):
            kv.get(f"key-{random.randint(0, 99999)}")
        read_time = time.time() - start

        write_ops = 100000 / write_time
        read_ops = 100000 / read_time

        # Check balance
        shard_sizes = [len(kv.shards[s]) for s in range(num_shards)]
        imbalance = max(shard_sizes) / max(min(shard_sizes), 1)

        print(f"  {num_shards:2d} shards: write={write_ops:,.0f} ops/s, "
              f"read={read_ops:,.0f} ops/s, imbalance={imbalance:.2f}x")


challenge_4()


# === Challenge 5: WAL Implementation ===
def challenge_5():
    """Implement write-ahead log for crash recovery."""
    print("\n=== Challenge 5: Write-Ahead Log ===\n")

    class WAL:
        def __init__(self):
            self.entries = []
            self.committed_index = 0

        def append(self, entry):
            self.entries.append(entry)
            return len(self.entries)

        def commit(self, index):
            self.committed_index = index

        def get_uncommitted(self):
            return self.entries[self.committed_index:]

        def replay(self, state):
            """Replay WAL to rebuild state."""
            for entry in self.entries[:self.committed_index]:
                op = entry.get("op")
                if op == "put":
                    state[entry["key"]] = entry["value"]
                elif op == "delete":
                    state.pop(entry.get("key"), None)
            return state

    wal = WAL()

    # Write operations
    for i in range(10):
        wal.append({"op": "put", "key": f"k{i}", "value": f"v{i}"})
    wal.commit(8)  # Committed up to index 8

    print(f"  WAL entries: {len(wal.entries)}")
    print(f"  Committed: {wal.committed_index}")
    print(f"  Uncommitted: {len(wal.get_uncommitted())}")

    # Simulate crash + recovery
    state = {}
    recovered = wal.replay(state)
    print(f"  Recovered state: {len(recovered)} keys")
    print(f"  Lost (uncommitted): {len(wal.get_uncommitted())} entries")


challenge_5()


if __name__ == "__main__":
    print("\nAll challenges completed.")
