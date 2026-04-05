# Lesson 21: Gossip Protocols

[Overview](./00_Overview.md) | [Previous: Distributed Hash Tables](./20_Distributed_Hash_Tables.md) | [Next: Service Discovery](./22_Service_Discovery.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement epidemic (gossip) protocols for reliable information dissemination
2. Build the SWIM failure detection protocol with suspicion and protocol period tuning
3. Design push, pull, and push-pull gossip variants and analyze their convergence properties
4. Implement phi-accrual failure detection for adaptive timeout management
5. Analyze gossip protocol performance in terms of convergence time, message overhead, and false positive rates

---

## Table of Contents

1. [Introduction to Gossip Protocols](#1-introduction-to-gossip-protocols)
2. [Epidemic Dissemination Models](#2-epidemic-dissemination-models)
3. [Push Gossip](#3-push-gossip)
4. [Pull and Push-Pull Gossip](#4-pull-and-push-pull-gossip)
5. [SWIM Failure Detection](#5-swim-failure-detection)
6. [Phi-Accrual Failure Detector](#6-phi-accrual-failure-detector)
7. [Gossip-Based Membership](#7-gossip-based-membership)
8. [Convergence Analysis](#8-convergence-analysis)
9. [Real-World Gossip Systems](#9-real-world-gossip-systems)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Introduction to Gossip Protocols

### 1.1 The Epidemic Metaphor

Gossip protocols spread information through a network the same way diseases spread through a population. Each node periodically contacts a random peer and exchanges information. Despite the randomness, information reaches all nodes in O(log N) rounds with high probability.

```
Round 0: [I] [ ] [ ] [ ] [ ] [ ] [ ] [ ]     1 infected
Round 1: [I] [ ] [ ] [I] [ ] [ ] [ ] [ ]     2 infected
Round 2: [I] [I] [ ] [I] [ ] [I] [ ] [ ]     4 infected
Round 3: [I] [I] [I] [I] [I] [I] [I] [ ]     7 infected
Round 4: [I] [I] [I] [I] [I] [I] [I] [I]     8 infected (all)
```

### 1.2 Why Gossip?

| Property | Gossip | Consensus (Raft/Paxos) |
|----------|--------|----------------------|
| Consistency | Eventual | Strong |
| Scalability | O(N log N) messages | O(N) per decision |
| Failure tolerance | Probabilistic, very robust | Requires majority |
| Latency | O(log N) rounds | O(1) rounds |
| Complexity | Simple | Complex |
| Use case | Membership, metrics, config | Writes, elections |

---

## 2. Epidemic Dissemination Models

### 2.1 Three Models

```python
import random
import time
import math
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class InfectionState(Enum):
    SUSCEPTIBLE = "S"    # Has not received the update
    INFECTED = "I"       # Has the update and is actively spreading
    REMOVED = "R"        # Has the update but stopped spreading


class EpidemicModel(Enum):
    SI = "SI"       # Susceptible → Infected (never stop spreading)
    SIR = "SIR"     # Susceptible → Infected → Removed (stop after k rounds)
    SIS = "SIS"     # Susceptible → Infected → Susceptible (can be re-infected)


@dataclass
class GossipNode:
    """A node in the gossip network."""
    node_id: str
    state: InfectionState = InfectionState.SUSCEPTIBLE
    data: dict = field(default_factory=dict)
    infection_round: int = -1
    spread_count: int = 0
    max_spreads: int = 3  # For SIR model: stop after this many rounds


class EpidemicSimulator:
    """
    Simulates epidemic information dissemination.

    Models how updates propagate through a gossip network
    under different epidemic models (SI, SIR, SIS).
    """

    def __init__(self, num_nodes: int, model: EpidemicModel = EpidemicModel.SIR,
                 fanout: int = 2):
        self.num_nodes = num_nodes
        self.model = model
        self.fanout = fanout  # Number of peers contacted per round
        self.nodes: Dict[str, GossipNode] = {}
        self.round_number: int = 0
        self.history: list[dict] = []

        # Create nodes
        for i in range(num_nodes):
            self.nodes[f"n{i}"] = GossipNode(node_id=f"n{i}")

    def infect(self, node_id: str, data: dict):
        """Start an epidemic by infecting a single node."""
        node = self.nodes[node_id]
        node.state = InfectionState.INFECTED
        node.data = data
        node.infection_round = 0

    def round(self):
        """Execute one gossip round."""
        self.round_number += 1
        all_ids = list(self.nodes.keys())

        # Each infected node contacts `fanout` random peers
        for node in list(self.nodes.values()):
            if node.state != InfectionState.INFECTED:
                continue

            # Select random peers (excluding self)
            peers = random.sample(
                [nid for nid in all_ids if nid != node.node_id],
                min(self.fanout, len(all_ids) - 1),
            )

            for peer_id in peers:
                peer = self.nodes[peer_id]
                if peer.state == InfectionState.SUSCEPTIBLE:
                    peer.state = InfectionState.INFECTED
                    peer.data = dict(node.data)
                    peer.infection_round = self.round_number

            node.spread_count += 1

            # SIR: become removed after max_spreads
            if self.model == EpidemicModel.SIR:
                if node.spread_count >= node.max_spreads:
                    node.state = InfectionState.REMOVED

        # Record state
        counts = self._count_states()
        self.history.append(counts)

    def _count_states(self) -> dict:
        """Count nodes in each state."""
        counts = {"S": 0, "I": 0, "R": 0}
        for node in self.nodes.values():
            counts[node.state.value] += 1
        return counts

    def run_until_complete(self, max_rounds: int = 100) -> int:
        """Run until all nodes have received the update."""
        for r in range(max_rounds):
            self.round()
            counts = self._count_states()
            if counts["S"] == 0:
                return self.round_number
        return max_rounds

    def convergence_report(self) -> dict:
        """Generate a convergence report."""
        total_infected = sum(
            1 for n in self.nodes.values()
            if n.state != InfectionState.SUSCEPTIBLE
        )
        return {
            "rounds": self.round_number,
            "infected": total_infected,
            "total": self.num_nodes,
            "coverage": total_infected / self.num_nodes * 100,
            "model": self.model.value,
            "fanout": self.fanout,
        }


def compare_epidemic_models():
    """Compare SI, SIR, and SIS epidemic models."""
    print("=== Epidemic Model Comparison ===\n")

    num_nodes = 100
    num_trials = 50

    for model in [EpidemicModel.SI, EpidemicModel.SIR]:
        rounds_list = []
        for _ in range(num_trials):
            sim = EpidemicSimulator(num_nodes, model=model, fanout=3)
            sim.infect("n0", {"key": "value"})
            rounds = sim.run_until_complete()
            rounds_list.append(rounds)

        avg_rounds = sum(rounds_list) / len(rounds_list)
        theoretical = math.log(num_nodes) / math.log(3 + 1)  # log_{f+1}(N)
        print(f"{model.value} model (N={num_nodes}, fanout=3):")
        print(f"  Average rounds to converge: {avg_rounds:.1f}")
        print(f"  Theoretical O(log N): ~{theoretical:.1f}")
        print(f"  Min/Max: {min(rounds_list)}/{max(rounds_list)}")
        print()


compare_epidemic_models()
```

---

## 3. Push Gossip

### 3.1 Push-Based Dissemination

In push gossip, infected nodes actively push updates to random peers:

```python
class PushGossipProtocol:
    """
    Push-based gossip protocol for state dissemination.

    Each node maintains a local state (e.g., membership list, metrics).
    Periodically, each node selects a random peer and sends its state.
    The receiver merges the received state with its own.
    """

    def __init__(self, node_id: str, all_nodes: list[str], fanout: int = 1):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.fanout = fanout
        self.state: Dict[str, dict] = {}  # key → {value, version, timestamp}
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.merges: int = 0

    def update_local(self, key: str, value: any):
        """Update a local state entry."""
        current = self.state.get(key, {})
        version = current.get("version", 0) + 1
        self.state[key] = {
            "value": value,
            "version": version,
            "timestamp": time.time(),
            "origin": self.node_id,
        }

    def prepare_push(self) -> list[dict]:
        """
        Prepare push messages to random peers.

        Returns a list of messages to send.
        """
        peers = [n for n in self.all_nodes if n != self.node_id]
        targets = random.sample(peers, min(self.fanout, len(peers)))

        messages = []
        for target in targets:
            messages.append({
                "type": "gossip_push",
                "from": self.node_id,
                "to": target,
                "state": dict(self.state),
            })
            self.messages_sent += 1

        return messages

    def receive_push(self, msg: dict):
        """
        Receive and merge a push gossip message.

        For each key, keep the entry with the highest version.
        """
        remote_state = msg.get("state", {})
        self.messages_received += 1

        for key, remote_entry in remote_state.items():
            local_entry = self.state.get(key)

            if local_entry is None or remote_entry["version"] > local_entry["version"]:
                self.state[key] = dict(remote_entry)
                self.merges += 1

    def stats(self) -> dict:
        return {
            "node": self.node_id,
            "keys": len(self.state),
            "sent": self.messages_sent,
            "received": self.messages_received,
            "merges": self.merges,
        }


def simulate_push_gossip():
    """Simulate push gossip for state dissemination."""
    print("=== Push Gossip Protocol ===\n")

    num_nodes = 20
    node_ids = [f"n{i}" for i in range(num_nodes)]
    nodes = {nid: PushGossipProtocol(nid, node_ids, fanout=2) for nid in node_ids}

    # Node 0 has an update
    nodes["n0"].update_local("config.version", "2.0.0")

    # Run gossip rounds
    for round_num in range(15):
        # Each node prepares push messages
        all_messages = []
        for node in nodes.values():
            all_messages.extend(node.prepare_push())

        # Deliver messages
        for msg in all_messages:
            target = msg["to"]
            if target in nodes:
                nodes[target].receive_push(msg)

        # Count how many nodes have the update
        informed = sum(
            1 for n in nodes.values()
            if "config.version" in n.state
        )

        if round_num < 5 or informed == num_nodes:
            print(f"  Round {round_num + 1}: {informed}/{num_nodes} nodes informed")

        if informed == num_nodes:
            total_msgs = sum(n.messages_sent for n in nodes.values())
            print(f"\n  Converged in {round_num + 1} rounds")
            print(f"  Total messages: {total_msgs}")
            print(f"  Messages per node: {total_msgs / num_nodes:.1f}")
            break


simulate_push_gossip()
```

---

## 4. Pull and Push-Pull Gossip

### 4.1 Pull Gossip

Pull gossip is more efficient when most nodes already have the update (the "tailing" phase of dissemination):

```python
class PushPullGossipProtocol:
    """
    Combined push-pull gossip protocol.

    Push phase: early dissemination when few nodes have the update.
    Pull phase: late dissemination when most nodes have the update.

    Push-pull combines both: each exchange involves sending AND
    requesting state, achieving faster convergence than either alone.
    """

    def __init__(self, node_id: str, all_nodes: list[str]):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.state: Dict[str, dict] = {}
        self.digest: Dict[str, int] = {}  # key → max_version seen
        self.messages_sent: int = 0

    def update_local(self, key: str, value: any):
        """Update local state."""
        version = self.digest.get(key, 0) + 1
        self.state[key] = {"value": value, "version": version}
        self.digest[key] = version

    def prepare_digest(self) -> dict:
        """Prepare a digest of our state versions for efficient sync."""
        return dict(self.digest)

    def exchange(self, peer_digest: dict, peer_state: dict) -> Tuple[dict, dict]:
        """
        Perform a push-pull exchange with a peer.

        1. Compare digests to find differences
        2. Send entries the peer is missing (push)
        3. Request entries we are missing (pull response)

        Returns (entries_to_send_to_peer, entries_we_need).
        """
        to_send = {}
        to_request = {}

        # Find keys we have but peer doesn't (or we have newer version)
        for key, version in self.digest.items():
            peer_version = peer_digest.get(key, 0)
            if version > peer_version:
                to_send[key] = self.state[key]

        # Find keys peer has but we don't (or peer has newer)
        for key, peer_version in peer_digest.items():
            our_version = self.digest.get(key, 0)
            if peer_version > our_version:
                if key in peer_state:
                    # Apply update from peer
                    self.state[key] = dict(peer_state[key])
                    self.digest[key] = peer_version

        self.messages_sent += 1
        return to_send, to_request

    def apply_updates(self, updates: dict):
        """Apply received updates from a peer."""
        for key, entry in updates.items():
            version = entry.get("version", 0)
            if version > self.digest.get(key, 0):
                self.state[key] = dict(entry)
                self.digest[key] = version


def simulate_push_pull():
    """Compare push-only, pull-only, and push-pull gossip."""
    print("=== Push vs Pull vs Push-Pull ===\n")

    num_nodes = 50
    num_trials = 30
    node_ids = [f"n{i}" for i in range(num_nodes)]

    for mode in ["push", "pull", "push-pull"]:
        rounds_to_converge = []

        for trial in range(num_trials):
            nodes = {
                nid: PushPullGossipProtocol(nid, node_ids)
                for nid in node_ids
            }

            # Node 0 has an update
            nodes["n0"].update_local("data", "update_v1")

            for round_num in range(50):
                # Each node picks a random peer
                for node in nodes.values():
                    peer_id = random.choice(
                        [n for n in node_ids if n != node.node_id]
                    )
                    peer = nodes[peer_id]

                    if mode == "push":
                        # Send our state to peer
                        peer.apply_updates(node.state)
                    elif mode == "pull":
                        # Request state from peer
                        node.apply_updates(peer.state)
                    else:
                        # Push-pull: bidirectional exchange
                        to_send, _ = node.exchange(
                            peer.prepare_digest(), peer.state
                        )
                        peer.apply_updates(to_send)

                informed = sum(1 for n in nodes.values() if "data" in n.state)
                if informed == num_nodes:
                    rounds_to_converge.append(round_num + 1)
                    break

        if rounds_to_converge:
            avg = sum(rounds_to_converge) / len(rounds_to_converge)
            print(f"{mode:12s}: avg={avg:.1f} rounds, "
                  f"min={min(rounds_to_converge)}, "
                  f"max={max(rounds_to_converge)}")
        else:
            print(f"{mode:12s}: did not converge in all trials")


simulate_push_pull()
```

---

## 5. SWIM Failure Detection

### 5.1 SWIM Protocol Overview

SWIM (Scalable Weakly-consistent Infection-style process group Membership) is a gossip-based failure detection protocol that achieves O(1) message load per member per protocol period:

```python
class SWIMNodeState(Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class SWIMMember:
    """A member in the SWIM group."""
    node_id: str
    state: SWIMNodeState = SWIMNodeState.ALIVE
    incarnation: int = 0
    last_updated: float = field(default_factory=time.time)


class SWIMProtocol:
    """
    Implementation of the SWIM failure detection protocol.

    Protocol period:
    1. Pick a random member M
    2. Send ping to M
    3. If M responds → M is alive
    4. If M doesn't respond within timeout:
       a. Pick k random members
       b. Ask them to ping M (indirect ping)
       c. If any indirect ping succeeds → M is alive
       d. If all fail → mark M as suspect

    The suspicion mechanism gives suspects a grace period
    before being declared dead.
    """

    def __init__(self, node_id: str, members: list[str],
                 k_indirect: int = 3, suspect_timeout: float = 5.0):
        self.node_id = node_id
        self.k_indirect = k_indirect
        self.suspect_timeout: float = suspect_timeout
        self.protocol_period: float = 1.0  # seconds

        self.members: Dict[str, SWIMMember] = {}
        for mid in members:
            self.members[mid] = SWIMMember(node_id=mid)

        # Statistics
        self.pings_sent: int = 0
        self.indirect_pings_sent: int = 0
        self.false_positives: int = 0
        self.true_positives: int = 0
        self.suspects: Dict[str, float] = {}  # node_id → suspect_since

    def protocol_round(self, alive_nodes: set[str]) -> list[dict]:
        """
        Execute one SWIM protocol round.

        Args:
            alive_nodes: Set of actually alive nodes (ground truth for simulation)

        Returns:
            List of events (membership changes)
        """
        events = []

        # Pick a random member to probe
        probe_candidates = [
            mid for mid in self.members
            if mid != self.node_id and self.members[mid].state != SWIMNodeState.DEAD
        ]

        if not probe_candidates:
            return events

        target_id = random.choice(probe_candidates)
        self.pings_sent += 1

        # Direct ping
        if target_id in alive_nodes:
            # Ping succeeded
            if target_id in self.suspects:
                del self.suspects[target_id]
                self.members[target_id].state = SWIMNodeState.ALIVE
                events.append({"type": "alive", "node": target_id})
        else:
            # Direct ping failed — try indirect probes
            indirect_targets = random.sample(
                [m for m in probe_candidates if m != target_id],
                min(self.k_indirect, len(probe_candidates) - 1),
            )

            indirect_success = False
            for proxy in indirect_targets:
                self.indirect_pings_sent += 1
                if proxy in alive_nodes and target_id in alive_nodes:
                    indirect_success = True
                    break

            if indirect_success:
                if target_id in self.suspects:
                    del self.suspects[target_id]
                    self.members[target_id].state = SWIMNodeState.ALIVE
            else:
                # Mark as suspect
                if target_id not in self.suspects:
                    self.suspects[target_id] = time.time()
                    self.members[target_id].state = SWIMNodeState.SUSPECT
                    events.append({"type": "suspect", "node": target_id})

        # Check suspect timeouts
        now = time.time()
        for suspect_id, suspect_since in list(self.suspects.items()):
            if now - suspect_since > self.suspect_timeout:
                self.members[suspect_id].state = SWIMNodeState.DEAD
                del self.suspects[suspect_id]
                events.append({"type": "dead", "node": suspect_id})

                # Track accuracy
                if suspect_id not in alive_nodes:
                    self.true_positives += 1
                else:
                    self.false_positives += 1

        return events

    def get_alive_members(self) -> list[str]:
        """Return list of members believed to be alive."""
        return [
            mid for mid, member in self.members.items()
            if member.state == SWIMNodeState.ALIVE
        ]

    def accuracy_report(self) -> dict:
        """Report detection accuracy."""
        return {
            "pings": self.pings_sent,
            "indirect_pings": self.indirect_pings_sent,
            "suspects": len(self.suspects),
            "dead": sum(1 for m in self.members.values() if m.state == SWIMNodeState.DEAD),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
        }


def simulate_swim():
    """Simulate the SWIM failure detection protocol."""
    print("=== SWIM Failure Detection ===\n")

    num_nodes = 20
    node_ids = [f"n{i}" for i in range(num_nodes)]
    swim = SWIMProtocol("n0", node_ids, k_indirect=3, suspect_timeout=0.5)

    # All nodes start alive
    alive = set(node_ids)

    # Simulate 50 protocol rounds
    for round_num in range(50):
        # At round 10, kill nodes 5 and 12
        if round_num == 10:
            alive.discard("n5")
            alive.discard("n12")
            print(f"  Round {round_num}: Killed n5 and n12")

        # At round 30, revive n5
        if round_num == 30:
            alive.add("n5")
            print(f"  Round {round_num}: Revived n5")

        events = swim.protocol_round(alive)
        for event in events:
            print(f"  Round {round_num}: {event['type'].upper()} → {event['node']}")

        time.sleep(0.01)  # Simulate time passing

    report = swim.accuracy_report()
    print(f"\nSWIM Accuracy Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")


simulate_swim()
```

---

## 6. Phi-Accrual Failure Detector

### 6.1 Adaptive Timeout

Instead of a fixed timeout, the phi-accrual detector outputs a continuous suspicion level (phi) based on the statistical distribution of heartbeat arrival times:

```python
class PhiAccrualDetector:
    """
    Phi-accrual failure detector (Hayashibara et al., 2004).

    Instead of a binary alive/dead output, this detector outputs
    a continuous suspicion level φ (phi). The higher φ, the more
    likely the node has failed.

    φ is calculated from the probability that a heartbeat would
    have arrived by now, given the historical arrival time distribution.

    φ = -log10(P(t_now - t_last > observed_interval))
    """

    def __init__(self, threshold: float = 8.0, window_size: int = 100,
                 min_std_dev_ms: float = 500.0):
        self.threshold = threshold  # φ above this → suspected
        self.window_size = window_size
        self.min_std_dev_ms = min_std_dev_ms

        # Heartbeat arrival intervals (ms)
        self.intervals: list[float] = []
        self.last_heartbeat: Optional[float] = None
        self.heartbeat_count: int = 0

    def heartbeat(self):
        """Record a heartbeat arrival."""
        now = time.time() * 1000  # ms

        if self.last_heartbeat is not None:
            interval = now - self.last_heartbeat
            self.intervals.append(interval)
            if len(self.intervals) > self.window_size:
                self.intervals.pop(0)

        self.last_heartbeat = now
        self.heartbeat_count += 1

    def phi(self) -> float:
        """
        Calculate the current phi value.

        phi = -log10(1 - CDF(t_now - t_last))

        where CDF is the cumulative distribution function of the
        heartbeat interval distribution (assumed normal).
        """
        if self.last_heartbeat is None or len(self.intervals) < 2:
            return 0.0

        now = time.time() * 1000
        elapsed = now - self.last_heartbeat

        # Calculate mean and std dev of intervals
        mean = sum(self.intervals) / len(self.intervals)
        variance = sum((x - mean) ** 2 for x in self.intervals) / len(self.intervals)
        std_dev = max(math.sqrt(variance), self.min_std_dev_ms)

        # Calculate P(X > elapsed) using normal distribution approximation
        # P(X > t) ≈ 1 - Φ((t - μ) / σ)
        z = (elapsed - mean) / std_dev

        # Approximate Φ(z) using logistic function
        cdf = 1.0 / (1.0 + math.exp(-1.7 * z))

        # φ = -log10(1 - CDF) = -log10(P(late))
        p_late = 1.0 - cdf
        if p_late <= 0:
            return float('inf')
        if p_late >= 1:
            return 0.0

        return -math.log10(p_late)

    def is_suspected(self) -> bool:
        """Check if the node is suspected of failure."""
        return self.phi() >= self.threshold

    def status(self) -> dict:
        """Get detector status."""
        intervals = self.intervals
        return {
            "phi": round(self.phi(), 2),
            "threshold": self.threshold,
            "suspected": self.is_suspected(),
            "heartbeats": self.heartbeat_count,
            "mean_interval_ms": round(sum(intervals) / len(intervals), 1) if intervals else 0,
            "std_dev_ms": round(
                (sum((x - sum(intervals)/len(intervals))**2
                     for x in intervals) / len(intervals)) ** 0.5, 1
            ) if len(intervals) > 1 else 0,
        }


def demonstrate_phi_detector():
    """Demonstrate the phi-accrual failure detector."""
    print("=== Phi-Accrual Failure Detector ===\n")

    detector = PhiAccrualDetector(threshold=8.0)

    # Phase 1: Regular heartbeats (every ~100ms with jitter)
    print("Phase 1: Regular heartbeats")
    for _ in range(20):
        detector.heartbeat()
        time.sleep(random.uniform(0.08, 0.12))  # 80-120ms

    print(f"  Status: {detector.status()}")

    # Phase 2: Delayed heartbeat (simulating GC pause)
    print("\nPhase 2: 500ms delay (GC pause)")
    time.sleep(0.5)
    phi_during_pause = detector.phi()
    print(f"  Phi during delay: {phi_during_pause:.2f}")
    print(f"  Suspected: {detector.is_suspected()}")

    # Heartbeat arrives
    detector.heartbeat()
    print(f"  After heartbeat: phi={detector.phi():.2f}, suspected={detector.is_suspected()}")

    # Phase 3: Node failure (no more heartbeats)
    print("\nPhase 3: Node failure (2 seconds silence)")
    time.sleep(0.3)
    for i in range(5):
        time.sleep(0.1)
        print(f"  t+{(i+1)*100 + 300}ms: phi={detector.phi():.2f}, "
              f"suspected={detector.is_suspected()}")


demonstrate_phi_detector()
```

---

## 7. Gossip-Based Membership

### 7.1 Membership with Piggyback Updates

SWIM piggybacks membership updates on ping/ack messages to avoid extra messages:

```python
class GossipMembershipProtocol:
    """
    Gossip-based membership protocol with piggybacked updates.

    Membership changes (join, leave, failure) are disseminated
    by piggybacking them on regular protocol messages.
    Each update has an incarnation number for ordering.
    """

    def __init__(self, node_id: str, seed_nodes: list[str]):
        self.node_id = node_id
        self.members: Dict[str, dict] = {
            node_id: {
                "state": "alive",
                "incarnation": 0,
                "address": f"{node_id}:7946",
            }
        }
        self.update_queue: list[dict] = []  # Piggybacked updates
        self.max_piggyback: int = 10  # Max updates per message
        self.update_retransmit: int = 3  # Retransmit each update N times

    def join(self, seed: str):
        """Join the cluster via a seed node."""
        self.members[seed] = {
            "state": "alive",
            "incarnation": 0,
            "address": f"{seed}:7946",
        }
        self._queue_update({
            "type": "join",
            "node": self.node_id,
            "incarnation": 0,
        })

    def leave(self):
        """Gracefully leave the cluster."""
        self.members[self.node_id]["state"] = "left"
        self._queue_update({
            "type": "leave",
            "node": self.node_id,
            "incarnation": self.members[self.node_id]["incarnation"],
        })

    def mark_suspect(self, node_id: str):
        """Mark a node as suspected."""
        if node_id in self.members and self.members[node_id]["state"] == "alive":
            self.members[node_id]["state"] = "suspect"
            self._queue_update({
                "type": "suspect",
                "node": node_id,
                "incarnation": self.members[node_id]["incarnation"],
            })

    def refute_suspect(self):
        """
        Refute our own suspicion by incrementing incarnation.

        If this node learns it is suspected, it increments its
        incarnation number and broadcasts an alive message.
        This overrides the suspect message.
        """
        self.members[self.node_id]["incarnation"] += 1
        self.members[self.node_id]["state"] = "alive"
        self._queue_update({
            "type": "alive",
            "node": self.node_id,
            "incarnation": self.members[self.node_id]["incarnation"],
        })

    def receive_update(self, update: dict) -> bool:
        """
        Process a received membership update.

        Returns True if the update was applied (new information).
        """
        node_id = update["node"]
        update_type = update["type"]
        incarnation = update["incarnation"]

        current = self.members.get(node_id)

        if current is None:
            # New member
            self.members[node_id] = {
                "state": "alive" if update_type in ("join", "alive") else update_type,
                "incarnation": incarnation,
                "address": f"{node_id}:7946",
            }
            return True

        # Incarnation-based ordering
        if incarnation < current["incarnation"]:
            return False  # Stale update

        if incarnation == current["incarnation"]:
            # Same incarnation: alive < suspect < dead
            priority = {"alive": 0, "suspect": 1, "dead": 2, "left": 3}
            if priority.get(update_type, 0) <= priority.get(current["state"], 0):
                return False  # Not newer

        # Apply update
        self.members[node_id]["state"] = update_type if update_type != "join" else "alive"
        self.members[node_id]["incarnation"] = incarnation
        self._queue_update(update)
        return True

    def _queue_update(self, update: dict):
        """Queue an update for piggybacking."""
        self.update_queue.append({
            **update,
            "retransmit_count": self.update_retransmit,
        })

    def get_piggyback_updates(self) -> list[dict]:
        """Get updates to piggyback on the next message."""
        updates = []
        remaining = []

        for entry in self.update_queue[:self.max_piggyback]:
            updates.append({k: v for k, v in entry.items() if k != "retransmit_count"})
            entry["retransmit_count"] -= 1
            if entry["retransmit_count"] > 0:
                remaining.append(entry)

        self.update_queue = remaining + self.update_queue[self.max_piggyback:]
        return updates

    def get_alive_members(self) -> list[str]:
        """Return alive members."""
        return [
            mid for mid, info in self.members.items()
            if info["state"] == "alive"
        ]


def demonstrate_gossip_membership():
    """Demonstrate gossip-based membership management."""
    print("=== Gossip Membership Protocol ===\n")

    # Create a 5-node cluster
    nodes = {}
    node_ids = [f"n{i}" for i in range(5)]

    for nid in node_ids:
        nodes[nid] = GossipMembershipProtocol(nid, [])

    # Bootstrap: all nodes know about each other
    for nid, node in nodes.items():
        for other in node_ids:
            if other != nid:
                node.join(other)

    print("Initial cluster: ", [n.get_alive_members() for n in nodes.values()][0])

    # Simulate: n3 is suspected by n0
    nodes["n0"].mark_suspect("n3")
    update = nodes["n0"].get_piggyback_updates()
    print(f"\nn0 suspects n3: {update}")

    # Gossip the suspicion to n1
    for u in update:
        nodes["n1"].receive_update(u)
    print(f"n1's view of n3: {nodes['n1'].members.get('n3', {}).get('state')}")

    # n3 refutes the suspicion
    nodes["n3"].refute_suspect()
    refute = nodes["n3"].get_piggyback_updates()
    print(f"\nn3 refutes with incarnation bump: {refute}")

    # Gossip refutation
    for u in refute:
        nodes["n0"].receive_update(u)
        nodes["n1"].receive_update(u)

    print(f"n0's view of n3 after refute: {nodes['n0'].members['n3']['state']}")
    print(f"n1's view of n3 after refute: {nodes['n1'].members['n3']['state']}")


demonstrate_gossip_membership()
```

---

## 8. Convergence Analysis

### 8.1 Mathematical Analysis

```python
def analyze_convergence():
    """Analyze gossip convergence properties mathematically."""
    print("=== Convergence Analysis ===\n")

    # Theoretical: with fanout f and N nodes
    # After r rounds, expected uninformed nodes: N * (1 - 1/N)^(f*r*...)
    # Simplified: converges in O(log N) rounds

    for n in [10, 100, 1000, 10000]:
        for fanout in [1, 2, 3]:
            # Simulation
            trials = 100
            rounds_list = []
            for _ in range(trials):
                sim = EpidemicSimulator(n, EpidemicModel.SIR, fanout=fanout)
                sim.infect("n0", {"data": True})
                rounds = sim.run_until_complete(max_rounds=50)
                rounds_list.append(rounds)

            avg = sum(rounds_list) / len(rounds_list)
            theoretical = math.ceil(math.log(n) / math.log(fanout + 1))
            print(f"  N={n:>5}, f={fanout}: avg_rounds={avg:.1f}, "
                  f"theoretical≈{theoretical}")

    # Message complexity
    print(f"\nMessage complexity per round:")
    print(f"  Push gossip: N × fanout messages")
    print(f"  Pull gossip: N × fanout messages")
    print(f"  SWIM: N × 1 direct + N × k indirect (worst case)")

    for n in [100, 1000, 10000]:
        for fanout in [2, 3]:
            total_messages = n * fanout * math.ceil(math.log(n) / math.log(fanout + 1))
            print(f"  N={n}, f={fanout}: ~{total_messages} total messages to converge")


analyze_convergence()
```

---

## 9. Real-World Gossip Systems

### 9.1 System Comparison

```python
def compare_gossip_systems():
    """Compare real-world systems that use gossip protocols."""
    print("=== Real-World Gossip Systems ===\n")

    systems = [
        {
            "name": "HashiCorp Serf/Consul",
            "protocol": "SWIM + Lifeguard extensions",
            "use_case": "Service discovery, membership",
            "gossip_interval": "200ms",
            "failure_detection": "~2-5 seconds",
        },
        {
            "name": "Apache Cassandra",
            "protocol": "Gossip (push-pull, phi-accrual)",
            "use_case": "Membership, schema propagation",
            "gossip_interval": "1 second",
            "failure_detection": "~10 seconds",
        },
        {
            "name": "Amazon S3",
            "protocol": "Custom gossip",
            "use_case": "Membership across thousands of nodes",
            "gossip_interval": "~1 second",
            "failure_detection": "Seconds",
        },
        {
            "name": "Redis Cluster",
            "protocol": "Gossip (custom)",
            "use_case": "Cluster state, slot mapping",
            "gossip_interval": "1 second",
            "failure_detection": "~15 seconds (configurable)",
        },
        {
            "name": "CockroachDB",
            "protocol": "Gossip overlay on Raft",
            "use_case": "Node liveness, range metadata",
            "gossip_interval": "~1 second",
            "failure_detection": "~9 seconds",
        },
    ]

    for sys in systems:
        print(f"{sys['name']}:")
        for key in ["protocol", "use_case", "gossip_interval", "failure_detection"]:
            print(f"  {key}: {sys[key]}")
        print()


compare_gossip_systems()
```

---

## 10. Summary and Key Takeaways

### Gossip Protocol Design Space

> **GOSSIP DESIGN DIMENSIONS**
>
> Dissemination:  Push │ Pull │ Push-Pull
> Model:          SI   │ SIR  │ SIS
> Detection:      Fixed timeout │ Phi-accrual │ SWIM
> Membership:     Centralized seed │ Gossip-based │ Hybrid
> Convergence:    O(log N) rounds with high probability

### Key Principles

1. **O(log N) convergence**: Gossip spreads exponentially — doubling informed nodes each round.
2. **Randomness provides robustness**: No single point of failure; tolerates arbitrary node failures.
3. **Push-pull is optimal**: Combines fast early spread (push) with efficient tailing (pull).
4. **SWIM scales better than heartbeats**: O(1) message load per member per period.
5. **Phi-accrual adapts to the network**: No fixed timeout to tune; adjusts to observed latency.

---

## 11. Practice Problems

### Problem 1: Convergence Proof

Prove that push gossip with fanout f converges in O(log_f N) rounds with probability at least 1 - 1/N.

### Problem 2: SWIM Analysis

In a 1000-node SWIM cluster with protocol period 1s and k=3:
- What is the expected time to detect a single node failure?
- What is the false positive rate if network latency spikes to 5x normal?
- How many protocol messages are sent per second cluster-wide?

### Problem 3: Phi Threshold Tuning

Given heartbeat intervals with mean=100ms and std_dev=10ms, calculate phi values at delays of 120ms, 150ms, 200ms, and 500ms. What threshold gives fewer than 0.1% false positives?

### Problem 4: Implementation Challenge

Implement a complete gossip protocol that combines SWIM failure detection with push-pull state dissemination. The protocol should:
- Detect failures within 3 protocol periods
- Disseminate membership updates within O(log N) periods
- Use piggyback to avoid extra messages

### Problem 5: Comparison Analysis

Design an experiment comparing gossip-based failure detection vs heartbeat-based detection at scales of 10, 100, 1000, and 10000 nodes. Measure: detection time, false positive rate, network bandwidth.

---

## 12. References

1. Demers, A. et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance." *PODC*.
2. Das, A., Gupta, I., & Motivala, A. (2002). "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol." *DSN*.
3. Hayashibara, N. et al. (2004). "The φ Accrual Failure Detector." *SRDS*.
4. Leitao, J., Pereira, J., & Rodrigues, L. (2007). "HyParView: A Membership Protocol for Reliable Gossip-Based Broadcast." *DSN*.
5. HashiCorp (2017). "Lifeguard: SWIM-ing with Situational Awareness."
6. Lakshman, A. & Malik, P. (2010). "Cassandra — A Decentralized Structured Storage System." *Operating Systems Review*.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 5. O'Reilly Media.

---

[Next: Lesson 22 — Service Discovery](./22_Service_Discovery.md)
