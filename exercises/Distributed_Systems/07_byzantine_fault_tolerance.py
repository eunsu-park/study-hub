"""
Exercises for Lesson 07: Byzantine Fault Tolerance
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from dataclasses import dataclass, field


# === Exercise 1: Byzantine Generals OM(1) Algorithm ===
# Problem: Implement the Oral Messages algorithm OM(1) for Byzantine
# agreement. With n >= 3f+1 and m=1 round of message relay, honest
# generals can reach agreement even with 1 traitor.

def om1_algorithm(
    general_values: Dict[str, str],
    traitor_ids: Set[str],
    commander: str,
    commander_value: str,
) -> Dict[str, str]:
    """
    Oral Messages OM(1) algorithm.

    Args:
        general_values: map of general_id -> value (unused for honest)
        traitor_ids: set of traitor general IDs
        commander: the commanding general's ID
        commander_value: the value the commander sends

    Returns:
        Map of general_id -> decided_value for each lieutenant.
    """
    lieutenants = [g for g in general_values if g != commander]
    n = len(general_values)

    # Phase 1: Commander sends value to all lieutenants
    received_from_commander = {}
    for lt in lieutenants:
        if commander in traitor_ids:
            # Traitor commander may send different values
            received_from_commander[lt] = random.choice(["ATTACK", "RETREAT"])
        else:
            received_from_commander[lt] = commander_value

    # Phase 2: Each lieutenant relays what it received to all others
    # relay[receiver][sender] = value_that_sender_relayed
    relayed: Dict[str, Dict[str, str]] = {lt: {} for lt in lieutenants}
    for sender in lieutenants:
        for receiver in lieutenants:
            if sender == receiver:
                continue
            if sender in traitor_ids:
                # Traitor relays a possibly different value
                relayed[receiver][sender] = random.choice(["ATTACK", "RETREAT"])
            else:
                relayed[receiver][sender] = received_from_commander[sender]

    # Phase 3: Each honest lieutenant takes majority of:
    # - value received from commander
    # - values relayed by other lieutenants
    decisions = {}
    for lt in lieutenants:
        if lt in traitor_ids:
            decisions[lt] = "TRAITOR"
            continue

        values = [received_from_commander[lt]]
        for sender, val in relayed[lt].items():
            values.append(val)

        # Majority vote
        counts = Counter(values)
        decisions[lt] = counts.most_common(1)[0][0]

    return decisions


def exercise_1():
    """
    Demonstrate OM(1) with 4 generals (1 traitor).
    """
    print("=== Exercise 1: Byzantine Generals OM(1) ===\n")

    random.seed(42)

    # Case 1: Honest commander, 1 traitor lieutenant
    generals = {"G0": "", "G1": "", "G2": "", "G3": ""}
    traitors = {"G3"}
    commander = "G0"

    print("Case 1: Honest commander G0, traitor lieutenant G3")
    decisions = om1_algorithm(generals, traitors, commander, "ATTACK")
    for g, d in sorted(decisions.items()):
        role = "traitor" if g in traitors else "honest"
        print(f"  {g} ({role}): decided {d}")

    honest_decisions = {g: d for g, d in decisions.items() if g not in traitors}
    values = set(honest_decisions.values())
    print(f"  Agreement among honest: {len(values) == 1}")

    # Case 2: Traitor commander
    print("\nCase 2: Traitor commander G0, honest lieutenants G1-G3")
    traitors2 = {"G0"}
    decisions2 = om1_algorithm(generals, traitors2, "G0", "ATTACK")
    for g, d in sorted(decisions2.items()):
        role = "traitor" if g in traitors2 else "honest"
        print(f"  {g} ({role}): decided {d}")

    honest_decisions2 = {g: d for g, d in decisions2.items() if g not in traitors2}
    values2 = set(honest_decisions2.values())
    print(f"  Agreement among honest: {len(values2) == 1}")
    print()


# === Exercise 2: PBFT Pre-prepare/Prepare/Commit ===
# Problem: Simulate the three phases of PBFT with n=4 replicas
# (f=1 faulty). Show that honest replicas reach agreement on the
# request ordering.

@dataclass
class PBFTMessage:
    """A PBFT protocol message."""
    msg_type: str  # "pre-prepare", "prepare", "commit"
    view: int
    seq_num: int
    digest: str
    sender: str


class PBFTReplica:
    """A PBFT replica."""

    def __init__(self, replica_id: str, n: int, f: int, is_faulty: bool = False):
        self.replica_id = replica_id
        self.n = n
        self.f = f
        self.is_faulty = is_faulty
        self.pre_prepared: Dict[int, str] = {}  # seq -> digest
        self.prepare_count: Dict[int, Counter] = {}  # seq -> Counter(digest)
        self.commit_count: Dict[int, Counter] = {}
        self.committed: Dict[int, str] = {}
        self.prepared: Set[int] = set()

    def handle_pre_prepare(self, msg: PBFTMessage) -> Optional[PBFTMessage]:
        """Handle pre-prepare from primary. Returns prepare message."""
        if self.is_faulty:
            return None

        self.pre_prepared[msg.seq_num] = msg.digest
        return PBFTMessage(
            "prepare", msg.view, msg.seq_num, msg.digest, self.replica_id
        )

    def handle_prepare(self, msg: PBFTMessage):
        """Collect prepare messages. Mark as prepared if 2f received."""
        if msg.seq_num not in self.prepare_count:
            self.prepare_count[msg.seq_num] = Counter()
        self.prepare_count[msg.seq_num][msg.digest] += 1

        if (
            msg.seq_num in self.pre_prepared
            and self.prepare_count[msg.seq_num][msg.digest] >= 2 * self.f
        ):
            self.prepared.add(msg.seq_num)

    def handle_commit(self, msg: PBFTMessage):
        """Collect commit messages. Commit if 2f+1 received."""
        if msg.seq_num not in self.commit_count:
            self.commit_count[msg.seq_num] = Counter()
        self.commit_count[msg.seq_num][msg.digest] += 1

        if self.commit_count[msg.seq_num][msg.digest] >= 2 * self.f + 1:
            self.committed[msg.seq_num] = msg.digest


def exercise_2():
    """
    Simulate PBFT with 4 replicas (f=1).
    """
    print("=== Exercise 2: PBFT Protocol Simulation (f=1) ===\n")

    n, f = 4, 1
    replicas = [
        PBFTReplica(f"R{i}", n, f, is_faulty=(i == 3))
        for i in range(n)
    ]

    primary = replicas[0]
    view = 0
    seq_num = 1
    request_digest = "hash(SET x=42)"

    print(f"Primary: R0, Faulty: R3, View: {view}")
    print(f"Client request digest: {request_digest}\n")

    # Phase 1: Pre-prepare (primary broadcasts)
    print("Phase 1: PRE-PREPARE")
    pre_prepare = PBFTMessage("pre-prepare", view, seq_num, request_digest, "R0")
    prepare_msgs = []
    for r in replicas:
        prep = r.handle_pre_prepare(pre_prepare)
        if prep:
            prepare_msgs.append(prep)
            print(f"  {r.replica_id} accepted pre-prepare, sending prepare")
        elif r.is_faulty:
            print(f"  {r.replica_id} (FAULTY) dropped pre-prepare")

    # Phase 2: Prepare (replicas exchange prepare messages)
    print("\nPhase 2: PREPARE")
    for msg in prepare_msgs:
        for r in replicas:
            if not r.is_faulty:
                r.handle_prepare(msg)

    commit_msgs = []
    for r in replicas:
        if seq_num in r.prepared:
            commit = PBFTMessage("commit", view, seq_num, request_digest, r.replica_id)
            commit_msgs.append(commit)
            print(f"  {r.replica_id} prepared (received 2f={2*f} prepares)")

    # Phase 3: Commit
    print("\nPhase 3: COMMIT")
    for msg in commit_msgs:
        for r in replicas:
            if not r.is_faulty:
                r.handle_commit(msg)

    for r in replicas:
        if seq_num in r.committed:
            print(f"  {r.replica_id} COMMITTED seq={seq_num} digest={r.committed[seq_num]}")
        elif r.is_faulty:
            print(f"  {r.replica_id} (FAULTY) did not commit")
        else:
            print(f"  {r.replica_id} not yet committed")

    committed_replicas = [r for r in replicas if seq_num in r.committed]
    print(f"\n{len(committed_replicas)} replicas committed. Agreement reached.")
    print()


# === Exercise 3: PBFT View Change Detection ===
# Problem: Implement view change detection and triggering. When a
# replica suspects the primary is faulty (timeout), it broadcasts
# a VIEW-CHANGE message. When 2f+1 VIEW-CHANGE messages are received,
# the new primary sends a NEW-VIEW message.

class PBFTViewChange:
    """View change protocol for PBFT."""

    def __init__(self, n: int, f: int, replicas: List[str]):
        self.n = n
        self.f = f
        self.replicas = replicas
        self.current_view = 0
        self.view_change_msgs: Dict[int, Set[str]] = {}  # view -> set of senders
        self.new_view_sent: Set[int] = set()

    def primary_for_view(self, view: int) -> str:
        """Determine primary for a given view."""
        return self.replicas[view % self.n]

    def suspect_primary(self, replica_id: str) -> Optional[int]:
        """
        A replica suspects the current primary.
        Returns the target view for the VIEW-CHANGE, or None.
        """
        target_view = self.current_view + 1
        if target_view not in self.view_change_msgs:
            self.view_change_msgs[target_view] = set()
        self.view_change_msgs[target_view].add(replica_id)

        return target_view

    def check_view_change(self, target_view: int) -> Optional[str]:
        """
        Check if enough VIEW-CHANGE messages have been received.
        Returns the new primary if view change can proceed.
        """
        if target_view not in self.view_change_msgs:
            return None

        if len(self.view_change_msgs[target_view]) >= 2 * self.f + 1:
            if target_view not in self.new_view_sent:
                self.new_view_sent.add(target_view)
                self.current_view = target_view
                return self.primary_for_view(target_view)
        return None


def exercise_3():
    """
    Demonstrate PBFT view change protocol.
    """
    print("=== Exercise 3: PBFT View Change Detection ===\n")

    replicas = ["R0", "R1", "R2", "R3"]
    n, f = 4, 1
    vc = PBFTViewChange(n, f, replicas)

    print(f"View {vc.current_view}: Primary = {vc.primary_for_view(vc.current_view)}")
    print(f"R0 is suspected to be faulty...\n")

    # Replicas suspect the primary
    for rid in ["R1", "R2", "R3"]:
        target = vc.suspect_primary(rid)
        count = len(vc.view_change_msgs.get(target, set()))
        print(
            f"  {rid} sends VIEW-CHANGE for view {target} "
            f"(count: {count}/{2*f+1} needed)"
        )
        new_primary = vc.check_view_change(target)
        if new_primary:
            print(f"\n  VIEW CHANGE COMPLETE!")
            print(f"  New view: {vc.current_view}")
            print(f"  New primary: {new_primary}")
            break

    # Demonstrate consecutive view changes
    print(f"\nNew primary {vc.primary_for_view(vc.current_view)} also fails...")
    for rid in ["R0", "R2", "R3"]:
        target = vc.suspect_primary(rid)
        new_primary = vc.check_view_change(target)
        if new_primary:
            print(f"  View changed again to {vc.current_view}, primary: {new_primary}")
            break

    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
