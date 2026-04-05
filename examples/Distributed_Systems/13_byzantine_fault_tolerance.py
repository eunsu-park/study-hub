"""
Practical Byzantine Fault Tolerance (PBFT) Simulator

Simulates the PBFT consensus protocol for tolerating Byzantine (arbitrary)
failures. A cluster of 3f+1 nodes can tolerate up to f Byzantine nodes.
The protocol proceeds through Pre-Prepare, Prepare, and Commit phases.

Key concepts:
- 3f+1 nodes required to tolerate f Byzantine faults
- Three-phase protocol: Pre-Prepare, Prepare, Commit
- View changes when primary is suspected faulty
- Byzantine nodes may send conflicting messages
- Compared with crash-fault protocols (Raft needs only 2f+1)

Usage:
    python 13_byzantine_fault_tolerance.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    IDLE = "idle"
    PRE_PREPARED = "pre-prepared"
    PREPARED = "prepared"
    COMMITTED = "committed"


class NodeBehavior(Enum):
    HONEST = "honest"
    BYZANTINE = "byzantine"


@dataclass
class PBFTMessage:
    """A message in the PBFT protocol."""
    phase: str          # "pre-prepare", "prepare", "commit"
    view: int
    sequence: int
    digest: str         # Hash of the request
    sender: int
    value: str          # The actual request value


@dataclass
class PBFTNode:
    """A node participating in PBFT consensus."""
    node_id: int
    behavior: NodeBehavior = NodeBehavior.HONEST
    phase: Phase = Phase.IDLE
    view: int = 0
    sequence: int = 0
    prepared_value: str | None = None
    committed_value: str | None = None
    prepare_count: int = 0
    commit_count: int = 0
    log: list[str] = field(default_factory=list)


class PBFTCluster:
    """
    Simulates a PBFT cluster.

    N = 3f + 1 nodes, tolerates f Byzantine faults.
    Node 0 is the primary (leader) for view 0.
    """

    def __init__(self, n: int, byzantine_ids: set[int] | None = None,
                 seed: int = 42):
        self.n = n
        self.f = (n - 1) // 3
        self.nodes = {i: PBFTNode(node_id=i) for i in range(n)}
        self.byzantine_ids = byzantine_ids or set()
        self._rng = random.Random(seed)

        for bid in self.byzantine_ids:
            if bid in self.nodes:
                self.nodes[bid].behavior = NodeBehavior.BYZANTINE

    @property
    def primary(self) -> int:
        """Current primary (leader) based on view."""
        return self.nodes[0].view % self.n

    def _quorum_size(self) -> int:
        """2f + 1 messages needed for quorum."""
        return 2 * self.f + 1

    def run_consensus(self, client_request: str) -> dict:
        """
        Run PBFT consensus on a client request.
        Returns result dict with decision and phase details.
        """
        result = {
            "request": client_request,
            "phases": [],
            "decided": False,
            "decided_value": None,
        }

        digest = f"H({client_request})"
        view = 0
        seq = 1

        # --- Phase 1: Pre-Prepare (primary broadcasts) ---
        primary_id = view % self.n
        primary = self.nodes[primary_id]

        phase_info = {"phase": "pre-prepare", "events": []}

        if primary.behavior == NodeBehavior.BYZANTINE:
            # Byzantine primary might send conflicting pre-prepares
            phase_info["events"].append(
                f"PRIMARY P{primary_id} is BYZANTINE — sends conflicting values!")
            for nid, node in self.nodes.items():
                if nid == primary_id:
                    continue
                fake_value = f"{client_request}_FAKE_{nid}" if nid % 2 == 0 else client_request
                node.log.append(f"Received pre-prepare: '{fake_value}'")
                node.prepared_value = fake_value if node.behavior == NodeBehavior.BYZANTINE else None
        else:
            phase_info["events"].append(
                f"PRIMARY P{primary_id} sends pre-prepare(v={view}, s={seq}, "
                f"d={digest})")
            for nid, node in self.nodes.items():
                if node.behavior == NodeBehavior.HONEST:
                    node.phase = Phase.PRE_PREPARED
                    node.prepared_value = client_request
                    node.log.append(f"Pre-prepared: '{client_request}'")

        result["phases"].append(phase_info)

        # --- Phase 2: Prepare (all replicas broadcast) ---
        phase_info = {"phase": "prepare", "events": []}

        prepare_votes: dict[str, int] = {}  # value -> count
        for nid, node in self.nodes.items():
            if node.behavior == NodeBehavior.BYZANTINE:
                # Byzantine node might vote for wrong value
                fake = f"{client_request}_EVIL"
                prepare_votes[fake] = prepare_votes.get(fake, 0) + 1
                phase_info["events"].append(
                    f"P{nid} (byzantine) sends prepare for '{fake}'")
            elif node.phase == Phase.PRE_PREPARED:
                val = node.prepared_value or client_request
                prepare_votes[val] = prepare_votes.get(val, 0) + 1
                phase_info["events"].append(
                    f"P{nid} sends prepare for '{val}'")

        # Each honest node counts matching prepares
        quorum = self._quorum_size()
        honest_prepared = False
        for nid, node in self.nodes.items():
            if node.behavior != NodeBehavior.HONEST:
                continue
            matching = prepare_votes.get(client_request, 0)
            node.prepare_count = matching
            if matching >= quorum:
                node.phase = Phase.PREPARED
                honest_prepared = True

        phase_info["events"].append(
            f"Prepare votes: {dict(prepare_votes)} "
            f"(need {quorum} for quorum)")
        phase_info["events"].append(
            f"Honest nodes prepared: {honest_prepared}")
        result["phases"].append(phase_info)

        # --- Phase 3: Commit (nodes that prepared broadcast commit) ---
        phase_info = {"phase": "commit", "events": []}

        commit_votes: dict[str, int] = {}
        for nid, node in self.nodes.items():
            if node.behavior == NodeBehavior.BYZANTINE:
                fake = f"{client_request}_EVIL"
                commit_votes[fake] = commit_votes.get(fake, 0) + 1
                phase_info["events"].append(
                    f"P{nid} (byzantine) sends commit for '{fake}'")
            elif node.phase == Phase.PREPARED:
                commit_votes[client_request] = commit_votes.get(client_request, 0) + 1
                phase_info["events"].append(
                    f"P{nid} sends commit for '{client_request}'")

        # Check for commit quorum
        decided = False
        for nid, node in self.nodes.items():
            if node.behavior != NodeBehavior.HONEST:
                continue
            matching = commit_votes.get(client_request, 0)
            node.commit_count = matching
            if matching >= quorum:
                node.phase = Phase.COMMITTED
                node.committed_value = client_request
                decided = True

        phase_info["events"].append(
            f"Commit votes: {dict(commit_votes)} "
            f"(need {quorum} for quorum)")
        result["phases"].append(phase_info)

        result["decided"] = decided
        result["decided_value"] = client_request if decided else None

        return result


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Pretty-print PBFT consensus result."""
    print(f"\n  Request: '{result['request']}'")
    for phase in result["phases"]:
        print(f"\n  [{phase['phase'].upper()}]")
        for event in phase["events"]:
            print(f"    {event}")
    status = "DECIDED" if result["decided"] else "FAILED"
    print(f"\n  Result: {status}", end="")
    if result["decided"]:
        print(f" on '{result['decided_value']}'")
    else:
        print()


def demo_no_faults() -> None:
    """PBFT with no Byzantine nodes."""
    print("=" * 70)
    print("PBFT: No Faults (4 nodes, f=1)")
    print("=" * 70)

    cluster = PBFTCluster(n=4)
    result = cluster.run_consensus("transfer $100")
    print_result(result)


def demo_one_byzantine() -> None:
    """PBFT with one Byzantine node (within tolerance)."""
    print("\n" + "=" * 70)
    print("PBFT: 1 Byzantine Node (4 nodes, f=1, within tolerance)")
    print("=" * 70)

    cluster = PBFTCluster(n=4, byzantine_ids={3})
    result = cluster.run_consensus("transfer $100")
    print_result(result)

    print(f"\n  With N=4, f=1: can tolerate 1 Byzantine node")
    print(f"  Quorum = 2f+1 = {2*1+1} honest nodes needed => OK")


def demo_too_many_byzantine() -> None:
    """PBFT with too many Byzantine nodes."""
    print("\n" + "=" * 70)
    print("PBFT: 2 Byzantine Nodes (4 nodes, f=1, EXCEEDS tolerance)")
    print("=" * 70)

    cluster = PBFTCluster(n=4, byzantine_ids={2, 3})
    result = cluster.run_consensus("transfer $100")
    print_result(result)

    print(f"\n  With N=4, f=1: can only tolerate 1 Byzantine node")
    print(f"  2 Byzantine nodes => safety may be violated!")


def demo_byzantine_primary() -> None:
    """PBFT with a Byzantine primary."""
    print("\n" + "=" * 70)
    print("PBFT: Byzantine Primary (4 nodes, primary P0 is Byzantine)")
    print("=" * 70)

    cluster = PBFTCluster(n=4, byzantine_ids={0})
    result = cluster.run_consensus("transfer $100")
    print_result(result)

    print(f"\n  Byzantine primary can send conflicting pre-prepares")
    print(f"  Honest nodes will not reach prepare quorum => VIEW CHANGE needed")


def demo_seven_nodes() -> None:
    """PBFT with 7 nodes (tolerates 2 Byzantine)."""
    print("\n" + "=" * 70)
    print("PBFT: 7 Nodes (f=2, tolerates 2 Byzantine)")
    print("=" * 70)

    cluster = PBFTCluster(n=7, byzantine_ids={4, 5})
    result = cluster.run_consensus("commit block #42")
    print_result(result)

    print(f"\n  N=7, f=2: quorum = 2*2+1 = 5 honest nodes needed")
    print(f"  5 honest nodes available => consensus succeeds")


def demo_comparison() -> None:
    """Compare BFT vs crash-fault tolerance requirements."""
    print("\n" + "=" * 70)
    print("BFT vs Crash-Fault Tolerance Comparison")
    print("=" * 70)

    print("""
  ┌─────────────────┬──────────────────┬──────────────────┐
  │ Faults (f)      │ Crash-Fault (Raft)│ Byzantine (PBFT) │
  │                 │ N = 2f+1         │ N = 3f+1         │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ f = 1           │ N = 3            │ N = 4            │
  │ f = 2           │ N = 5            │ N = 7            │
  │ f = 3           │ N = 7            │ N = 10           │
  │ f = 10          │ N = 21           │ N = 31           │
  └─────────────────┴──────────────────┴──────────────────┘

  Message complexity per consensus round:
  - Raft:  O(N)       — leader sends to all followers
  - PBFT:  O(N^2)     — all-to-all in Prepare and Commit phases
  - HotStuff: O(N)    — linear BFT using threshold signatures

  When to use BFT:
  - Blockchain / permissioned ledgers (untrusted participants)
  - Multi-party computation (no single trusted authority)
  - Safety-critical systems (defense against arbitrary failures)

  When crash-fault tolerance suffices:
  - Internal microservices (trusted network)
  - Database replication (nodes are under your control)
  - Most cloud infrastructure
""")


if __name__ == "__main__":
    demo_no_faults()
    demo_one_byzantine()
    demo_too_many_byzantine()
    demo_byzantine_primary()
    demo_seven_nodes()
    demo_comparison()
    print("Done.")
