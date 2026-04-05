"""
Exercises for Lesson 21: Gossip Protocols
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
import math
import time
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# === Exercise 1: Convergence Proof (Simulation) ===
def exercise_1():
    """Verify push gossip convergence in O(log_f N) rounds."""
    print("=== Exercise 1: Convergence Proof ===\n")

    for N in [10, 100, 1000]:
        for f in [1, 2, 3]:
            trials = 200
            rounds_list = []
            for _ in range(trials):
                informed = {0}
                nodes = set(range(N))
                rounds = 0
                while informed != nodes:
                    rounds += 1
                    new_informed = set(informed)
                    for node in informed:
                        targets = random.sample(
                            list(nodes - {node}), min(f, N - 1)
                        )
                        new_informed.update(targets)
                    informed = new_informed
                    if rounds > 100:
                        break
                rounds_list.append(rounds)

            avg = sum(rounds_list) / len(rounds_list)
            theoretical = math.ceil(math.log(N) / math.log(f + 1))
            print(f"  N={N:>5}, f={f}: avg={avg:.1f}, "
                  f"theoretical~{theoretical}, "
                  f"P(converge)={sum(1 for r in rounds_list if r <= 100)/trials:.2f}")


exercise_1()


# === Exercise 2: SWIM Analysis ===
def exercise_2():
    """SWIM cluster analysis with 1000 nodes."""
    print("\n=== Exercise 2: SWIM Analysis ===\n")

    N = 1000
    period = 1.0  # seconds
    k = 3

    # Expected time to detect single failure
    # Each round, a node has 1/N chance of being probed
    # Expected rounds to probe specific node: N
    # But with suspicion mechanism: typically 2-5 periods
    expected_detection = N * period / N  # ~1 period per probe attempt
    # On average, takes ~N/N = 1 round for some node to probe the failed one
    # But the failed node needs to be selected by SOME node
    print(f"  N={N}, period={period}s, k={k}")
    print(f"  Expected detection time: ~{period * 3:.1f}s (3 periods with suspicion)")
    print(f"  Messages per second: {N} direct + {N * k} indirect (worst) = {N * (1 + k)}")
    print(f"  Per-node message load: {1 + k} messages/period (O(1))")

    # False positive with 5x latency spike
    print(f"\n  With 5x latency spike:")
    print(f"    Direct ping timeout exceeded → indirect probes triggered")
    print(f"    If all k indirect probes also affected → false suspect")
    print(f"    Suspicion timeout provides grace period")
    print(f"    False positive rate depends on suspicion timeout setting")


exercise_2()


# === Exercise 3: Phi Threshold Tuning ===
def exercise_3():
    """Calculate phi values for given heartbeat distribution."""
    print("\n=== Exercise 3: Phi Threshold Tuning ===\n")

    mean = 100  # ms
    std_dev = 10  # ms

    for delay in [120, 150, 200, 500]:
        z = (delay - mean) / std_dev
        # P(X > delay) using normal approximation
        cdf = 1.0 / (1.0 + math.exp(-1.7 * z))
        p_late = 1.0 - cdf
        if p_late <= 0:
            phi = float('inf')
        else:
            phi = -math.log10(p_late)

        print(f"  delay={delay}ms: z={z:.1f}, P(late)={p_late:.6f}, phi={phi:.2f}")

    print(f"\n  For <0.1% false positives:")
    print(f"    Need P(false positive) < 0.001")
    print(f"    phi > -log10(0.001) = 3.0")
    print(f"    Recommended threshold: phi >= 4 (safe margin)")


exercise_3()


# === Exercise 4: Combined Protocol ===
def exercise_4():
    """Implement combined SWIM + push-pull gossip."""
    print("\n=== Exercise 4: Combined Protocol ===\n")

    class CombinedProtocol:
        def __init__(self, node_id, members, k=3):
            self.node_id = node_id
            self.members = set(members)
            self.state = {}  # shared state
            self.alive_members = set(members)
            self.k = k
            self.detection_rounds = 0

        def protocol_round(self, alive_ground_truth):
            """One round: SWIM probe + gossip state exchange."""
            # SWIM: probe random member
            target = random.choice(list(self.members - {self.node_id}))
            if target not in alive_ground_truth:
                # Try indirect
                proxies = random.sample(
                    list(self.members - {self.node_id, target}),
                    min(self.k, len(self.members) - 2)
                )
                reachable = any(p in alive_ground_truth for p in proxies)
                if not reachable:
                    self.alive_members.discard(target)
                    self.detection_rounds += 1

            # Gossip: exchange state with a random alive peer
            peer = random.choice(list(self.alive_members - {self.node_id}))
            # Push-pull: bidirectional state merge
            return target, peer

    proto = CombinedProtocol("n0", [f"n{i}" for i in range(20)])
    alive = {f"n{i}" for i in range(18)}  # n18, n19 are dead

    for r in range(30):
        target, peer = proto.protocol_round(alive)

    print(f"  Alive members detected: {len(proto.alive_members)}")
    print(f"  Detection rounds: {proto.detection_rounds}")
    print(f"  Dead nodes removed: {proto.members - proto.alive_members}")


exercise_4()


# === Exercise 5: Scale Comparison ===
def exercise_5():
    """Compare gossip vs heartbeat at different scales."""
    print("\n=== Exercise 5: Scale Comparison ===\n")

    for N in [10, 100, 1000, 10000]:
        # Heartbeat: each node pings all others
        heartbeat_msgs = N * (N - 1)

        # Gossip (SWIM): each node pings 1 + k indirect
        k = 3
        gossip_msgs = N * (1 + k)

        # Detection time
        heartbeat_detection = 3.0  # 3 missed heartbeats × 1s
        gossip_detection = 3.0     # ~3 protocol periods

        print(f"  N={N:>5}: heartbeat={heartbeat_msgs:>10} msg/s, "
              f"gossip={gossip_msgs:>6} msg/s, "
              f"ratio={heartbeat_msgs/gossip_msgs:.0f}x")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
