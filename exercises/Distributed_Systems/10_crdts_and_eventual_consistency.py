"""
Exercises for Lesson 10: CRDTs and Eventual Consistency
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import uuid
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# === Exercise 1: OR-Set CRDT ===
# Problem: Implement an Observed-Remove Set (OR-Set) CRDT that supports
# add and remove operations. Each add assigns a unique tag; remove
# only removes the tags that were observed at the time of removal.
# This allows concurrent add and remove of the same element to
# converge correctly (add wins over concurrent remove).

class ORSet:
    """
    Observed-Remove Set CRDT.

    Each element is associated with a set of unique tags (one per add).
    Remove only removes the tags that the remover has observed.
    An element is in the set if it has at least one tag.
    """

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        # element -> set of (tag, replica_id) pairs
        self.elements: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.tombstones: Set[Tuple[str, str, str]] = set()  # (element, tag, replica)

    def add(self, element: str):
        """Add an element with a unique tag."""
        tag = str(uuid.uuid4())[:8]
        self.elements[element].add((tag, self.replica_id))

    def remove(self, element: str):
        """
        Remove an element by removing all currently observed tags.
        """
        if element in self.elements:
            for tag, rid in self.elements[element]:
                self.tombstones.add((element, tag, rid))
            self.elements[element].clear()

    def lookup(self, element: str) -> bool:
        """Check if an element is in the set."""
        return len(self.elements.get(element, set())) > 0

    def value(self) -> Set[str]:
        """Return all elements currently in the set."""
        return {e for e, tags in self.elements.items() if tags}

    def merge(self, other: "ORSet"):
        """
        Merge another OR-Set into this one.
        Add all tags from the other set, then apply all tombstones.
        """
        # Add all elements/tags from other
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for elem in all_elements:
            self.elements[elem] = self.elements[elem] | other.elements.get(elem, set())

        # Merge tombstones
        all_tombstones = self.tombstones | other.tombstones

        # Apply tombstones: remove tags that have been tombstoned
        for elem, tag, rid in all_tombstones:
            if elem in self.elements:
                self.elements[elem].discard((tag, rid))

        self.tombstones = all_tombstones


def exercise_1():
    """
    Demonstrate OR-Set with concurrent add/remove.
    """
    print("=== Exercise 1: OR-Set CRDT ===\n")

    r1 = ORSet("R1")
    r2 = ORSet("R2")

    # Both add "apple"
    r1.add("apple")
    r2.add("apple")
    print(f"R1 adds 'apple': {r1.value()}")
    print(f"R2 adds 'apple': {r2.value()}")

    # R1 removes "apple", R2 concurrently adds "apple" again
    # (Before merging, so R1 doesn't see R2's add)
    r1.remove("apple")
    r2.add("apple")  # concurrent add
    print(f"\nR1 removes 'apple': {r1.value()}")
    print(f"R2 adds 'apple' again: {r2.value()}")

    # Merge: add-wins semantics
    r1.merge(r2)
    r2.merge(r1)
    print(f"\nAfter merge:")
    print(f"  R1 sees: {r1.value()}")
    print(f"  R2 sees: {r2.value()}")
    print(f"  'apple' present: {r1.lookup('apple')}")
    print("  (Add wins over concurrent remove)")

    assert r1.lookup("apple"), "apple should be present (add wins)"
    print()


# === Exercise 2: LWW-Element-Set CRDT ===
# Problem: Implement a Last-Writer-Wins Element Set with configurable
# bias (add-bias or remove-bias) for tie-breaking when add and remove
# have the same timestamp.

class LWWElementSet:
    """
    Last-Writer-Wins Element Set CRDT.

    Maintains add-set and remove-set with timestamps.
    An element is in the set if its latest add timestamp is greater
    than its latest remove timestamp (or equal, depending on bias).
    """

    def __init__(self, bias: str = "add"):
        """
        Args:
            bias: "add" (add-bias) or "remove" (remove-bias).
                  Determines winner on timestamp tie.
        """
        assert bias in ("add", "remove"), "bias must be 'add' or 'remove'"
        self.bias = bias
        self.add_set: Dict[str, float] = {}
        self.remove_set: Dict[str, float] = {}

    def add(self, element: str, timestamp: float):
        """Add an element with a timestamp."""
        if element not in self.add_set or timestamp > self.add_set[element]:
            self.add_set[element] = timestamp

    def remove(self, element: str, timestamp: float):
        """Remove an element with a timestamp."""
        if element not in self.remove_set or timestamp > self.remove_set[element]:
            self.remove_set[element] = timestamp

    def lookup(self, element: str) -> bool:
        """Check if an element is in the set."""
        if element not in self.add_set:
            return False
        add_ts = self.add_set[element]
        remove_ts = self.remove_set.get(element, -1)

        if add_ts > remove_ts:
            return True
        elif add_ts < remove_ts:
            return False
        else:
            # Tie: use bias
            return self.bias == "add"

    def value(self) -> Set[str]:
        """Return all elements in the set."""
        return {e for e in self.add_set if self.lookup(e)}

    def merge(self, other: "LWWElementSet"):
        """Merge another LWW-Element-Set by taking max timestamps."""
        for elem, ts in other.add_set.items():
            self.add(elem, ts)
        for elem, ts in other.remove_set.items():
            self.remove(elem, ts)


def exercise_2():
    """
    Demonstrate LWW-Element-Set with add-bias and remove-bias.
    """
    print("=== Exercise 2: LWW-Element-Set CRDT ===\n")

    # Add-bias: tie goes to add
    add_biased = LWWElementSet(bias="add")
    add_biased.add("x", 10.0)
    add_biased.remove("x", 10.0)  # same timestamp
    print(f"Add-bias: add(x, t=10), remove(x, t=10)")
    print(f"  'x' present: {add_biased.lookup('x')} (tie -> add wins)")

    # Remove-bias: tie goes to remove
    rem_biased = LWWElementSet(bias="remove")
    rem_biased.add("x", 10.0)
    rem_biased.remove("x", 10.0)
    print(f"\nRemove-bias: add(x, t=10), remove(x, t=10)")
    print(f"  'x' present: {rem_biased.lookup('x')} (tie -> remove wins)")

    # Merge scenario
    r1 = LWWElementSet(bias="add")
    r2 = LWWElementSet(bias="add")

    r1.add("a", 1.0)
    r1.add("b", 2.0)
    r2.add("a", 3.0)
    r2.remove("b", 5.0)
    r1.remove("a", 2.0)

    print(f"\nBefore merge:")
    print(f"  R1: {r1.value()}")
    print(f"  R2: {r2.value()}")

    r1.merge(r2)
    print(f"After merge R1<-R2: {r1.value()}")
    # 'a': add@3 > remove@2 -> present
    # 'b': add@2 < remove@5 -> absent
    assert r1.lookup("a") is True
    assert r1.lookup("b") is False
    print("  'a' present (add@3 > remove@2)")
    print("  'b' absent  (add@2 < remove@5)")
    print()


# === Exercise 3: CRDT-Based Collaborative Text Editor ===
# Problem: Implement a simple CRDT-based collaborative text editor
# supporting insert and delete operations. Uses a list CRDT with
# unique position identifiers.

@dataclass
class TextChar:
    """A character in the collaborative text with a unique ID."""
    char: str
    position_id: Tuple[int, str]  # (sequence, replica_id)
    deleted: bool = False


class CRDTTextEditor:
    """
    A simple CRDT-based collaborative text editor.
    Uses a list of characters with unique position identifiers.
    Insertions create new position IDs between existing ones.
    Deletions are tombstoned (marked as deleted).
    """

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.chars: List[TextChar] = []
        self.seq_counter = 0

    def _next_id(self) -> Tuple[int, str]:
        self.seq_counter += 1
        return (self.seq_counter, self.replica_id)

    def insert(self, index: int, char: str):
        """Insert a character at the given index."""
        tc = TextChar(char, self._next_id())
        self.chars.insert(min(index, len(self.chars)), tc)

    def delete(self, index: int):
        """Delete the character at the given index (tombstone)."""
        visible_idx = 0
        for i, tc in enumerate(self.chars):
            if not tc.deleted:
                if visible_idx == index:
                    tc.deleted = True
                    return
                visible_idx += 1

    def text(self) -> str:
        """Return the visible text."""
        return "".join(tc.char for tc in self.chars if not tc.deleted)

    def merge(self, other: "CRDTTextEditor"):
        """
        Merge another editor's state. Use position IDs to determine
        ordering. Add characters not present locally.
        """
        local_ids = {tc.position_id for tc in self.chars}
        remote_ids = {tc.position_id for tc in other.chars}

        # Add missing characters from other
        for tc in other.chars:
            if tc.position_id not in local_ids:
                self.chars.append(TextChar(tc.char, tc.position_id, tc.deleted))
            else:
                # Update deletion status (tombstone wins)
                for local_tc in self.chars:
                    if local_tc.position_id == tc.position_id:
                        if tc.deleted:
                            local_tc.deleted = True
                        break

        # Sort by position ID for consistent ordering
        self.chars.sort(key=lambda tc: tc.position_id)


def exercise_3():
    """
    Demonstrate CRDT-based collaborative text editing.
    """
    print("=== Exercise 3: CRDT Collaborative Text Editor ===\n")

    editor1 = CRDTTextEditor("E1")
    editor2 = CRDTTextEditor("E2")

    # Editor 1 types "HELLO"
    for i, ch in enumerate("HELLO"):
        editor1.insert(i, ch)
    print(f"Editor 1 types: '{editor1.text()}'")

    # Sync to editor 2
    editor2.merge(editor1)
    print(f"Editor 2 after sync: '{editor2.text()}'")

    # Concurrent edits
    # Editor 1 deletes 'O' and types '!'
    editor1.delete(4)  # delete 'O'
    editor1.insert(4, "!")
    print(f"\nEditor 1 edits: '{editor1.text()}'")

    # Editor 2 inserts ' WORLD' at the end
    for i, ch in enumerate(" WORLD"):
        editor2.insert(5 + i, ch)
    print(f"Editor 2 edits: '{editor2.text()}'")

    # Merge both directions
    editor1.merge(editor2)
    editor2.merge(editor1)
    print(f"\nAfter merge:")
    print(f"  Editor 1: '{editor1.text()}'")
    print(f"  Editor 2: '{editor2.text()}'")
    print("  Both editors converge to the same text.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
