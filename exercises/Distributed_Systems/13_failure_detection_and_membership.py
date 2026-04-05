"""
Exercises for Lesson 13: Failure Detection and Membership
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import math
import random
import time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field


# === Exercise 1: SWIM Protocol with Suspicion ===
# Problem: Implement the SWIM failure detection protocol with the
# suspicion mechanism. A node is first suspected (not immediately
# declared dead) and only declared dead after a suspicion timeout.

class MemberState(Enum):
    ALIVE = "alive"
    SUSPECTED = "suspected"
    DEAD = "dead"


@dataclass
class Member:
    """A member in the SWIM protocol."""
    node_id: str
    state: MemberState = MemberState.ALIVE
    incarnation: int = 0
    suspected_at: Optional[float] = None


class SWIMProtocol:
    """
    SWIM (Scalable Weakly-consistent Infection-style Membership)
    protocol with suspicion mechanism.
    """

    def __init__(
        self,
        node_id: str,
        suspicion_timeout: float = 5.0,
        num_indirect_probes: int = 3,
    ):
        self.node_id = node_id
        self.members: Dict[str, Member] = {
            node_id: Member(node_id, MemberState.ALIVE)
        }
        self.suspicion_timeout = suspicion_timeout
        self.num_indirect_probes = num_indirect_probes
        self.protocol_log: List[str] = []

    def add_member(self, node_id: str):
        self.members[node_id] = Member(node_id, MemberState.ALIVE)

    def ping(self, target: str, responds: bool) -> str:
        """
        Direct ping to target.

        Returns: "ack" if target responds, "timeout" otherwise.
        """
        if target not in self.members:
            return "unknown"
        if responds:
            self.members[target].state = MemberState.ALIVE
            self.members[target].suspected_at = None
            return "ack"
        return "timeout"

    def indirect_ping(
        self, target: str, delegates: List[str],
        delegate_results: Dict[str, bool],
    ) -> bool:
        """
        Send indirect pings through delegates.
        Returns True if any delegate gets an ack from target.
        """
        for delegate in delegates:
            if delegate_results.get(delegate, False):
                self.members[target].state = MemberState.ALIVE
                self.members[target].suspected_at = None
                self.protocol_log.append(
                    f"Indirect ping: {target} alive (via {delegate})"
                )
                return True
        return False

    def suspect(self, target: str, current_time: float):
        """Mark a node as suspected."""
        if target in self.members:
            member = self.members[target]
            if member.state == MemberState.ALIVE:
                member.state = MemberState.SUSPECTED
                member.suspected_at = current_time
                self.protocol_log.append(
                    f"Suspected {target} at t={current_time:.1f}"
                )

    def check_suspicion_timeouts(self, current_time: float) -> List[str]:
        """
        Check for suspected members whose suspicion timeout has expired.
        Returns list of newly declared dead members.
        """
        newly_dead = []
        for mid, member in self.members.items():
            if member.state == MemberState.SUSPECTED and member.suspected_at:
                if current_time - member.suspected_at >= self.suspicion_timeout:
                    member.state = MemberState.DEAD
                    newly_dead.append(mid)
                    self.protocol_log.append(
                        f"Declared {mid} DEAD at t={current_time:.1f}"
                    )
        return newly_dead

    def refute_suspicion(self, node_id: str) -> bool:
        """
        A suspected node refutes suspicion by incrementing its
        incarnation number.
        """
        if node_id in self.members:
            member = self.members[node_id]
            if member.state == MemberState.SUSPECTED:
                member.incarnation += 1
                member.state = MemberState.ALIVE
                member.suspected_at = None
                self.protocol_log.append(
                    f"{node_id} refuted suspicion (incarnation={member.incarnation})"
                )
                return True
        return False

    def probe_round(
        self, target: str, direct_ok: bool,
        indirect_results: Dict[str, bool],
        current_time: float,
    ):
        """Run one probe round for a target."""
        result = self.ping(target, direct_ok)
        if result == "ack":
            self.protocol_log.append(f"Ping {target}: ACK")
            return

        # Direct ping failed, try indirect
        delegates = [
            m for m in self.members
            if m != self.node_id and m != target
            and self.members[m].state == MemberState.ALIVE
        ]
        selected = delegates[:self.num_indirect_probes]

        if self.indirect_ping(target, selected, indirect_results):
            return

        # Both failed: suspect the node
        self.suspect(target, current_time)


def exercise_1():
    """
    Demonstrate SWIM protocol with suspicion mechanism.
    """
    print("=== Exercise 1: SWIM Protocol with Suspicion ===\n")

    swim = SWIMProtocol("N0", suspicion_timeout=3.0, num_indirect_probes=2)
    for i in range(1, 6):
        swim.add_member(f"N{i}")

    # Round 1: Probe N3 - it responds
    swim.probe_round("N3", direct_ok=True, indirect_results={}, current_time=1.0)

    # Round 2: Probe N3 - no response (direct + indirect fail)
    swim.probe_round(
        "N3", direct_ok=False,
        indirect_results={"N1": False, "N2": False},
        current_time=2.0,
    )

    print("After probe rounds:")
    for mid, member in sorted(swim.members.items()):
        print(f"  {mid}: {member.state.value} (incarnation={member.incarnation})")

    # N3 refutes suspicion
    print()
    swim.refute_suspicion("N3")

    # Probe N5 - fails, and suspicion times out
    swim.probe_round(
        "N5", direct_ok=False,
        indirect_results={"N1": False, "N2": False},
        current_time=5.0,
    )

    # Time passes, check timeouts
    dead = swim.check_suspicion_timeouts(8.5)

    print("\nAfter suspicion timeout:")
    for mid, member in sorted(swim.members.items()):
        print(f"  {mid}: {member.state.value}")

    print(f"\nProtocol log:")
    for entry in swim.protocol_log:
        print(f"  {entry}")
    print()


# === Exercise 2: Push-Pull Gossip Protocol ===
# Problem: Implement a push-pull gossip protocol for membership
# dissemination. Nodes periodically exchange their full membership
# lists, merging based on incarnation numbers.

@dataclass
class GossipMember:
    """Member entry in gossip protocol."""
    node_id: str
    state: str  # "alive", "dead"
    incarnation: int
    heartbeat: int


class GossipProtocol:
    """Push-pull gossip for membership dissemination."""

    def __init__(self, node_id: str, fanout: int = 2):
        self.node_id = node_id
        self.fanout = fanout
        self.members: Dict[str, GossipMember] = {
            node_id: GossipMember(node_id, "alive", 0, 0)
        }
        self.round = 0

    def add_member(self, node_id: str):
        self.members[node_id] = GossipMember(node_id, "alive", 0, 0)

    def heartbeat(self):
        """Increment local heartbeat counter."""
        self.members[self.node_id].heartbeat += 1

    def push(self) -> Dict[str, GossipMember]:
        """Push: send our membership list."""
        return dict(self.members)

    def pull_merge(self, remote_members: Dict[str, GossipMember]) -> int:
        """
        Pull: merge remote membership list with ours.
        Use incarnation number for conflict resolution.
        Returns number of updates applied.
        """
        updates = 0
        for nid, remote in remote_members.items():
            local = self.members.get(nid)
            if local is None:
                self.members[nid] = GossipMember(
                    remote.node_id, remote.state,
                    remote.incarnation, remote.heartbeat,
                )
                updates += 1
            elif remote.incarnation > local.incarnation:
                self.members[nid] = GossipMember(
                    remote.node_id, remote.state,
                    remote.incarnation, remote.heartbeat,
                )
                updates += 1
            elif (
                remote.incarnation == local.incarnation
                and remote.heartbeat > local.heartbeat
            ):
                local.heartbeat = remote.heartbeat
                updates += 1

        return updates

    def gossip_round(self, peers: Dict[str, "GossipProtocol"]) -> int:
        """
        Run one gossip round: push-pull with random subset of peers.
        """
        self.round += 1
        self.heartbeat()

        available = [p for p in self.members if p != self.node_id and p in peers]
        targets = random.sample(available, min(self.fanout, len(available)))

        total_updates = 0
        for target_id in targets:
            target = peers[target_id]
            # Push: send our list
            remote_list = target.push()
            updates = self.pull_merge(remote_list)
            # Pull: target merges our list
            updates += target.pull_merge(self.push())
            total_updates += updates

        return total_updates


def exercise_2():
    """
    Demonstrate push-pull gossip protocol convergence.
    """
    print("=== Exercise 2: Push-Pull Gossip Protocol ===\n")

    random.seed(42)

    # Create 8 nodes, initially each only knows itself
    nodes = {}
    for i in range(8):
        nid = f"N{i}"
        nodes[nid] = GossipProtocol(nid, fanout=2)

    # N0 knows about N1 and N2 initially
    nodes["N0"].add_member("N1")
    nodes["N0"].add_member("N2")
    nodes["N1"].add_member("N0")
    nodes["N2"].add_member("N0")
    # N3 knows N4
    nodes["N3"].add_member("N4")
    nodes["N4"].add_member("N3")
    # Others isolated initially

    print("Initial membership knowledge:")
    for nid in sorted(nodes):
        known = sorted(nodes[nid].members.keys())
        print(f"  {nid} knows: {known}")

    # Run gossip rounds
    print("\nGossip rounds:")
    for round_num in range(10):
        total_updates = 0
        for nid in sorted(nodes):
            total_updates += nodes[nid].gossip_round(nodes)

        sizes = [len(n.members) for n in nodes.values()]
        avg_known = sum(sizes) / len(sizes)
        print(
            f"  Round {round_num+1}: avg_known={avg_known:.1f}, "
            f"updates={total_updates}, "
            f"fully_converged={all(s == len(nodes) for s in sizes)}"
        )
        if all(s == len(nodes) for s in sizes):
            print(f"  All nodes converged in {round_num+1} rounds!")
            break

    print("\nFinal membership:")
    for nid in sorted(nodes):
        known = sorted(nodes[nid].members.keys())
        print(f"  {nid} knows: {known}")
    print()


# === Exercise 3: Failure Detector Comparison ===
# Problem: Compare fixed timeout, adaptive timeout, and phi accrual
# failure detectors on the same set of heartbeat arrival data.

class FixedTimeoutDetector:
    """Simple fixed timeout failure detector."""

    def __init__(self, timeout_ms: float):
        self.timeout_ms = timeout_ms

    def is_alive(self, current_time: float, last_heartbeat: float) -> bool:
        return (current_time - last_heartbeat) < self.timeout_ms


class AdaptiveTimeoutDetector:
    """
    Adaptive timeout detector that adjusts based on observed
    heartbeat intervals.
    """

    def __init__(self, safety_margin: float = 2.0, window_size: int = 10):
        self.safety_margin = safety_margin
        self.window_size = window_size
        self.intervals: deque = deque(maxlen=window_size)
        self.last_arrival: Optional[float] = None

    def record_heartbeat(self, arrival_time: float):
        if self.last_arrival is not None:
            interval = arrival_time - self.last_arrival
            self.intervals.append(interval)
        self.last_arrival = arrival_time

    def get_timeout(self) -> float:
        if not self.intervals:
            return 5000.0  # default
        avg = sum(self.intervals) / len(self.intervals)
        return avg * self.safety_margin

    def is_alive(self, current_time: float) -> bool:
        if self.last_arrival is None:
            return True
        return (current_time - self.last_arrival) < self.get_timeout()


class PhiAccrualDetector:
    """
    Phi Accrual failure detector.
    Instead of a binary alive/dead, outputs a suspicion level (phi)
    on a continuous scale. Higher phi = more suspicious.
    """

    def __init__(self, threshold: float = 8.0, window_size: int = 100):
        self.threshold = threshold
        self.window_size = window_size
        self.intervals: deque = deque(maxlen=window_size)
        self.last_arrival: Optional[float] = None

    def record_heartbeat(self, arrival_time: float):
        if self.last_arrival is not None:
            self.intervals.append(arrival_time - self.last_arrival)
        self.last_arrival = arrival_time

    def phi(self, current_time: float) -> float:
        """
        Calculate the phi value (suspicion level).
        phi = -log10(P(interval >= t_now - t_last))
        Using normal distribution approximation.
        """
        if self.last_arrival is None or len(self.intervals) < 2:
            return 0.0

        t_diff = current_time - self.last_arrival
        mean = sum(self.intervals) / len(self.intervals)
        variance = sum(
            (x - mean) ** 2 for x in self.intervals
        ) / len(self.intervals)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        # P(X >= t_diff) using complementary CDF approximation
        y = (t_diff - mean) / std_dev
        # Approximate: P(X >= t_diff) ~ exp(-y^2/2) for large y
        if y <= 0:
            return 0.0
        p = math.exp(-0.5 * y * y) / (y * math.sqrt(2 * math.pi) + 1e-10)
        p = max(p, 1e-100)
        return -math.log10(p)

    def is_alive(self, current_time: float) -> bool:
        return self.phi(current_time) < self.threshold


def exercise_3():
    """
    Compare three failure detector approaches.
    """
    print("=== Exercise 3: Failure Detector Comparison ===\n")

    random.seed(42)

    # Generate heartbeat arrivals with some jitter
    base_interval = 1000.0  # 1 second
    heartbeats = [0.0]
    for i in range(20):
        jitter = random.gauss(0, 100)  # 100ms std dev jitter
        heartbeats.append(heartbeats[-1] + base_interval + jitter)

    # After 20 heartbeats, node "dies" (no more heartbeats)
    last_heartbeat_time = heartbeats[-1]

    # Set up detectors
    fixed = FixedTimeoutDetector(timeout_ms=3000.0)
    adaptive = AdaptiveTimeoutDetector(safety_margin=3.0, window_size=10)
    phi_det = PhiAccrualDetector(threshold=8.0, window_size=20)

    # Feed heartbeats
    for hb in heartbeats:
        adaptive.record_heartbeat(hb)
        phi_det.record_heartbeat(hb)

    # Check at various times after last heartbeat
    print(f"Last heartbeat at t={last_heartbeat_time:.0f}ms")
    print(f"Fixed timeout: {fixed.timeout_ms:.0f}ms")
    print(f"Adaptive timeout: {adaptive.get_timeout():.0f}ms")
    print(f"Phi threshold: {phi_det.threshold}\n")

    print(f"{'Time after last HB':>20s} {'Fixed':>8s} {'Adaptive':>10s} "
          f"{'Phi value':>10s} {'Phi alive':>10s}")
    print("-" * 65)

    for delta in [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
        check_time = last_heartbeat_time + delta
        f_alive = fixed.is_alive(check_time, last_heartbeat_time)
        a_alive = adaptive.is_alive(check_time)
        p_val = phi_det.phi(check_time)
        p_alive = phi_det.is_alive(check_time)

        print(
            f"{delta:>18d}ms {'alive' if f_alive else 'DEAD':>8s} "
            f"{'alive' if a_alive else 'DEAD':>10s} "
            f"{p_val:>10.1f} {'alive' if p_alive else 'DEAD':>10s}"
        )

    print(
        "\nFixed: simple but requires manual tuning."
    )
    print(
        "Adaptive: adjusts to network conditions but still binary."
    )
    print(
        "Phi accrual: continuous suspicion level, more nuanced detection."
    )
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
