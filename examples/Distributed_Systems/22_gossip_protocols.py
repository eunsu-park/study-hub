"""
Gossip Protocol (Epidemic Broadcast) Simulator

Simulates epidemic-style gossip protocols for disseminating information
across a cluster. Demonstrates push, pull, and push-pull variants,
and the SWIM protocol for membership and failure detection.

Key concepts:
- Push gossip: infected nodes push updates to random peers
- Pull gossip: nodes pull updates from random peers
- Push-pull: combines both for faster convergence
- Convergence: O(log N) rounds for full dissemination
- SWIM: Scalable Weakly-consistent Infection-style Membership

Usage:
    python 22_gossip_protocols.py
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Gossip Protocol
# ---------------------------------------------------------------------------

@dataclass
class GossipNode:
    """A node in the gossip network."""
    node_id: int
    data: dict[str, int] = field(default_factory=dict)  # key -> version
    alive: bool = True
    infected: bool = False   # Has received the gossip

    def merge(self, other_data: dict[str, int]) -> int:
        """Merge data from another node. Returns count of updates applied."""
        updates = 0
        for key, version in other_data.items():
            if key not in self.data or version > self.data[key]:
                self.data[key] = version
                updates += 1
        return updates


class GossipSimulator:
    """Simulates gossip dissemination across a cluster."""

    def __init__(self, n: int, fanout: int = 3, seed: int = 42):
        """
        Args:
            n: Number of nodes.
            fanout: Number of peers each node gossips to per round.
        """
        self.nodes = {i: GossipNode(i) for i in range(n)}
        self.n = n
        self.fanout = fanout
        self._rng = random.Random(seed)

    def inject(self, node_id: int, key: str, version: int) -> None:
        """Inject a new piece of gossip at a specific node."""
        self.nodes[node_id].data[key] = version
        self.nodes[node_id].infected = True

    def push_round(self) -> int:
        """
        One round of push gossip: each infected node sends to fanout peers.
        Returns the number of newly infected nodes.
        """
        newly_infected = 0
        for node in list(self.nodes.values()):
            if not node.alive or not node.infected:
                continue
            peers = self._select_peers(node.node_id)
            for peer_id in peers:
                peer = self.nodes[peer_id]
                if peer.alive:
                    updates = peer.merge(node.data)
                    if updates > 0 and not peer.infected:
                        peer.infected = True
                        newly_infected += 1
        return newly_infected

    def pull_round(self) -> int:
        """
        One round of pull gossip: each node pulls from fanout random peers.
        """
        newly_infected = 0
        for node in self.nodes.values():
            if not node.alive:
                continue
            peers = self._select_peers(node.node_id)
            for peer_id in peers:
                peer = self.nodes[peer_id]
                if peer.alive:
                    updates = node.merge(peer.data)
                    if updates > 0 and not node.infected:
                        node.infected = True
                        newly_infected += 1
        return newly_infected

    def push_pull_round(self) -> int:
        """One round of push-pull gossip."""
        newly = 0
        for node in list(self.nodes.values()):
            if not node.alive:
                continue
            peers = self._select_peers(node.node_id)
            for peer_id in peers:
                peer = self.nodes[peer_id]
                if not peer.alive:
                    continue
                # Push
                u1 = peer.merge(node.data)
                if u1 > 0 and not peer.infected:
                    peer.infected = True
                    newly += 1
                # Pull
                u2 = node.merge(peer.data)
                if u2 > 0 and not node.infected:
                    node.infected = True
                    newly += 1

        return newly

    def infected_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.infected)

    def _select_peers(self, exclude: int) -> list[int]:
        candidates = [nid for nid in self.nodes if nid != exclude]
        return self._rng.sample(candidates, min(self.fanout, len(candidates)))


# ---------------------------------------------------------------------------
# SWIM Protocol
# ---------------------------------------------------------------------------

class SWIMState:
    """SWIM membership protocol simulator."""

    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"

    def __init__(self, n: int, k_indirect: int = 3, seed: int = 42):
        """
        Args:
            n: Number of initial members.
            k_indirect: Number of indirect probes for failure detection.
        """
        self.members: dict[int, str] = {i: self.ALIVE for i in range(n)}
        self.k_indirect = k_indirect
        self._rng = random.Random(seed)
        self.log: list[str] = []
        self._incarnation: dict[int, int] = {i: 0 for i in range(n)}

    def probe_round(self, prober: int, target: int,
                    target_responds: bool) -> str:
        """
        Run one SWIM probe round.
        Returns the final status of the target.
        """
        if target_responds:
            self.log.append(f"P{prober} -> P{target}: direct ping OK")
            self.members[target] = self.ALIVE
            return self.ALIVE

        # Direct ping failed, try indirect probes
        self.log.append(
            f"P{prober} -> P{target}: direct ping FAILED, "
            f"trying {self.k_indirect} indirect probes")

        others = [m for m in self.members if m != prober and m != target
                  and self.members[m] == self.ALIVE]
        delegates = self._rng.sample(others, min(self.k_indirect, len(others)))

        indirect_ok = False
        for delegate in delegates:
            # Simulate: delegate pings target
            can_reach = self._rng.random() > 0.5  # 50% chance via indirect
            if can_reach:
                self.log.append(
                    f"  P{delegate} -> P{target}: indirect ping OK")
                indirect_ok = True
                break
            else:
                self.log.append(
                    f"  P{delegate} -> P{target}: indirect ping FAILED")

        if indirect_ok:
            self.members[target] = self.ALIVE
            return self.ALIVE
        else:
            self.members[target] = self.SUSPECT
            self.log.append(f"  P{target} marked as SUSPECT")
            return self.SUSPECT

    def declare_dead(self, target: int) -> None:
        """Declare a suspected node as dead after timeout."""
        if self.members.get(target) == self.SUSPECT:
            self.members[target] = self.DEAD
            self.log.append(f"P{target} declared DEAD")

    def refute(self, node_id: int) -> None:
        """Node refutes suspicion by incrementing incarnation."""
        if self.members.get(node_id) == self.SUSPECT:
            self._incarnation[node_id] += 1
            self.members[node_id] = self.ALIVE
            self.log.append(
                f"P{node_id} REFUTES suspicion "
                f"(incarnation={self._incarnation[node_id]})")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_gossip_convergence() -> None:
    """Compare convergence of push, pull, and push-pull gossip."""
    print("=" * 70)
    print("Gossip Convergence: Push vs Pull vs Push-Pull")
    print("=" * 70)

    n = 100
    fanout = 3

    variants = [
        ("Push", lambda sim: sim.push_round()),
        ("Pull", lambda sim: sim.pull_round()),
        ("Push-Pull", lambda sim: sim.push_pull_round()),
    ]

    print(f"\n  {n} nodes, fanout={fanout}, 1 initial infected node")
    print(f"  Expected convergence: ~{math.log2(n):.0f} rounds (O(log N))\n")

    print(f"  {'Round':>6}", end="")
    for name, _ in variants:
        print(f"  {name:>10}", end="")
    print()
    print("  " + "-" * 40)

    results: dict[str, list[int]] = {name: [] for name, _ in variants}

    for round_num in range(1, 20):
        row = f"  {round_num:>6}"
        for name, round_fn in variants:
            if round_num == 1:
                sim = GossipSimulator(n, fanout, seed=42)
                sim.inject(0, "update-1", 1)
                # Store sim for subsequent rounds
                results[name] = [sim]
            sim = results[name][0]
            round_fn(sim)
            count = sim.infected_count()
            pct = 100.0 * count / n
            row += f"  {pct:>9.1f}%"

            if count == n and len(results[name]) == 1:
                results[name].append(round_num)

        print(row)

    print(f"\n  Full convergence:")
    for name, data in results.items():
        rounds = data[1] if len(data) > 1 else ">19"
        print(f"    {name:>10}: {rounds} rounds")


def demo_swim() -> None:
    """Demonstrate SWIM failure detection."""
    print("\n" + "=" * 70)
    print("SWIM Protocol: Failure Detection")
    print("=" * 70)

    swim = SWIMState(n=8, k_indirect=3, seed=42)

    print(f"\n  8 members, 3 indirect probes\n")

    # Healthy ping
    print("  Scenario 1: Healthy node responds")
    swim.probe_round(prober=0, target=1, target_responds=True)

    # Node fails direct ping but reachable indirectly
    print(f"\n  Scenario 2: Direct ping fails, indirect succeeds")
    swim.probe_round(prober=0, target=5, target_responds=False)

    # Node actually fails
    print(f"\n  Scenario 3: Node truly failed")
    random.seed(99)  # Make indirect probes fail
    swim2 = SWIMState(n=8, k_indirect=3, seed=99)
    swim2.probe_round(prober=0, target=7, target_responds=False)
    swim2.declare_dead(7)

    # Node refutes suspicion
    print(f"\n  Scenario 4: False suspicion refuted")
    swim3 = SWIMState(n=8, k_indirect=3, seed=99)
    swim3.probe_round(prober=0, target=3, target_responds=False)
    swim3.refute(3)

    # Print all logs
    print(f"\n  Event log:")
    for log in [swim, swim2, swim3]:
        for line in log.log:
            print(f"    {line}")
        if log.log:
            print()


def demo_gossip_with_failures() -> None:
    """Show gossip dissemination with node failures."""
    print("=" * 70)
    print("Gossip with Node Failures")
    print("=" * 70)

    n = 50
    sim = GossipSimulator(n, fanout=3, seed=42)

    # Kill 10 nodes (20%)
    for i in range(40, 50):
        sim.nodes[i].alive = False

    sim.inject(0, "critical-update", 1)

    print(f"\n  {n} nodes, 10 dead (20%), fanout=3")
    print(f"  Tracking gossip spread to {n - 10} alive nodes:\n")

    for r in range(1, 15):
        sim.push_pull_round()
        alive_infected = sum(1 for node in sim.nodes.values()
                             if node.alive and node.infected)
        alive_total = sum(1 for node in sim.nodes.values() if node.alive)
        pct = 100.0 * alive_infected / alive_total
        bar = "#" * int(pct / 2)
        print(f"    Round {r:>2}: {alive_infected:>3}/{alive_total} "
              f"alive infected ({pct:5.1f}%) {bar}")
        if alive_infected == alive_total:
            print(f"\n  All alive nodes infected in {r} rounds "
                  f"(despite 20% failures)")
            break


if __name__ == "__main__":
    demo_gossip_convergence()
    demo_swim()
    demo_gossip_with_failures()
    print("Done.")
