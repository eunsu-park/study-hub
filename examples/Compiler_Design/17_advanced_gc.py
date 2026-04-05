"""
17_advanced_gc.py - Advanced Garbage Collection Techniques

Builds on lesson 14's fundamentals with advanced GC topics:

  1. Tri-Color Marking
     Implements the tri-color invariant for incremental and concurrent
     collection. Objects are white (unvisited), gray (visited, children
     not scanned), or black (fully scanned).

  2. Write Barriers
     Demonstrates snapshot-at-the-beginning and incremental-update
     write barriers that maintain the tri-color invariant when the
     mutator modifies references during collection.

  3. Escape Analysis
     Determines which allocations can be placed on the stack instead
     of the heap, avoiding GC entirely for short-lived objects.

  4. Concurrent Collector Simulation
     Simulates a concurrent mark-sweep collector where the mutator
     and collector operate in interleaved phases.

Topics covered:
  - Tri-color abstraction and invariants
  - Write barrier designs (Dijkstra, Steele, Yuasa)
  - Incremental vs concurrent collection
  - Escape analysis for stack allocation
  - GC-safe points and handshakes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tri-Color Marking
# ---------------------------------------------------------------------------

class Color(Enum):
    WHITE = auto()  # Not yet visited
    GRAY  = auto()  # Visited, children not fully scanned
    BLACK = auto()  # Fully scanned


@dataclass
class GCObject:
    oid: int
    name: str
    refs: list[int] = field(default_factory=list)
    color: Color = Color.WHITE

    def __repr__(self):
        return f"{self.name}({self.color.name})"


class TriColorCollector:
    """
    Incremental tri-color mark-sweep collector.
    Collection proceeds in small increments, allowing the mutator
    to run between increments.
    """

    def __init__(self):
        self.heap: dict[int, GCObject] = {}
        self.roots: set[int] = set()
        self.gray_set: list[int] = []
        self.next_id = 0
        self.write_barrier_log: list[str] = []
        self.freed: list[str] = []

    def allocate(self, name: str) -> int:
        oid = self.next_id
        self.next_id += 1
        self.heap[oid] = GCObject(oid=oid, name=name)
        return oid

    def add_ref(self, from_oid: int, to_oid: int) -> None:
        """Add a reference with write barrier."""
        self.heap[from_oid].refs.append(to_oid)
        # Dijkstra-style write barrier: if source is black and target
        # is white, shade the target gray to maintain the invariant.
        src = self.heap[from_oid]
        tgt = self.heap.get(to_oid)
        if src.color == Color.BLACK and tgt and tgt.color == Color.WHITE:
            tgt.color = Color.GRAY
            self.gray_set.append(to_oid)
            self.write_barrier_log.append(
                f"  Barrier: {src.name}(BLACK) -> {tgt.name}(WHITE), "
                f"shaded {tgt.name} GRAY"
            )

    def init_marking(self) -> None:
        """Initialize marking: shade all objects white, roots gray."""
        for obj in self.heap.values():
            obj.color = Color.WHITE
        self.gray_set = []
        for rid in self.roots:
            obj = self.heap.get(rid)
            if obj:
                obj.color = Color.GRAY
                self.gray_set.append(rid)

    def mark_step(self, steps: int = 1) -> bool:
        """
        Process 'steps' gray objects.
        Returns True if marking is complete (no more gray objects).
        """
        for _ in range(steps):
            if not self.gray_set:
                return True
            oid = self.gray_set.pop(0)
            obj = self.heap.get(oid)
            if obj is None or obj.color == Color.BLACK:
                continue
            # Scan children
            for child_id in obj.refs:
                child = self.heap.get(child_id)
                if child and child.color == Color.WHITE:
                    child.color = Color.GRAY
                    self.gray_set.append(child_id)
            obj.color = Color.BLACK
        return len(self.gray_set) == 0

    def sweep(self) -> int:
        """Remove all white (unreachable) objects."""
        to_remove = [oid for oid, obj in self.heap.items()
                     if obj.color == Color.WHITE]
        for oid in to_remove:
            self.freed.append(self.heap[oid].name)
            del self.heap[oid]
        return len(to_remove)

    def full_collect(self) -> int:
        """Run a complete incremental collection."""
        self.init_marking()
        while not self.mark_step(steps=2):
            pass  # In practice, mutator runs between steps
        return self.sweep()

    def snapshot(self) -> dict[str, str]:
        return {obj.name: obj.color.name for obj in self.heap.values()}


# ---------------------------------------------------------------------------
# Escape Analysis
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    name: str
    func: str
    escapes: bool = False
    reason: str = ""


@dataclass
class FuncInfo:
    name: str
    allocations: list[str] = field(default_factory=list)
    returns: list[str] = field(default_factory=list)
    stores_to_global: list[str] = field(default_factory=list)
    calls_with: dict[str, list[str]] = field(default_factory=dict)


class EscapeAnalyzer:
    """
    Determines which allocations escape their defining function.
    An allocation escapes if:
      - It is returned from the function
      - It is stored to a global or heap location
      - It is passed as an argument to a function that may capture it
    Non-escaping allocations can be stack-allocated.
    """

    def __init__(self):
        self.allocations: dict[str, Allocation] = {}

    def analyze(self, func: FuncInfo) -> dict[str, Allocation]:
        for alloc_name in func.allocations:
            alloc = Allocation(name=alloc_name, func=func.name)

            if alloc_name in func.returns:
                alloc.escapes = True
                alloc.reason = "returned from function"
            elif alloc_name in func.stores_to_global:
                alloc.escapes = True
                alloc.reason = "stored to global/heap"
            else:
                for callee, args in func.calls_with.items():
                    if alloc_name in args:
                        alloc.escapes = True
                        alloc.reason = f"passed to {callee}()"
                        break

            if not alloc.escapes:
                alloc.reason = "does not escape -> stack allocate"

            self.allocations[alloc_name] = alloc

        return self.allocations


# ---------------------------------------------------------------------------
# Concurrent Collector Simulation
# ---------------------------------------------------------------------------

class ConcurrentCollector:
    """
    Simulates a concurrent mark-sweep collector with interleaved
    mutator and collector phases.
    """

    def __init__(self):
        self.heap: dict[int, GCObject] = {}
        self.roots: set[int] = set()
        self.next_id = 0
        self.phase = "idle"  # idle, marking, sweeping
        self.gray_set: list[int] = []
        self.log: list[str] = []
        self.freed: list[str] = []

    def allocate(self, name: str) -> int:
        oid = self.next_id
        self.next_id += 1
        obj = GCObject(oid=oid, name=name)
        # If marking is in progress, new objects are born black
        # (already considered reachable) to prevent premature collection
        if self.phase == "marking":
            obj.color = Color.BLACK
            self.log.append(f"  Alloc {name} as BLACK (marking in progress)")
        self.heap[oid] = obj
        return oid

    def start_marking(self) -> None:
        self.phase = "marking"
        for obj in self.heap.values():
            obj.color = Color.WHITE
        self.gray_set = []
        for rid in self.roots:
            obj = self.heap.get(rid)
            if obj:
                obj.color = Color.GRAY
                self.gray_set.append(rid)
        self.log.append("  Phase: MARKING started")

    def mark_increment(self, work: int = 2) -> None:
        """Do a bounded amount of marking work."""
        done = 0
        while self.gray_set and done < work:
            oid = self.gray_set.pop(0)
            obj = self.heap.get(oid)
            if obj is None or obj.color == Color.BLACK:
                continue
            for child_id in obj.refs:
                child = self.heap.get(child_id)
                if child and child.color == Color.WHITE:
                    child.color = Color.GRAY
                    self.gray_set.append(child_id)
            obj.color = Color.BLACK
            done += 1
        if not self.gray_set:
            self.log.append("  Marking complete, transitioning to SWEEP")
            self.phase = "sweeping"

    def sweep(self) -> int:
        to_remove = [oid for oid, obj in self.heap.items()
                     if obj.color == Color.WHITE]
        for oid in to_remove:
            self.freed.append(self.heap[oid].name)
            del self.heap[oid]
        self.phase = "idle"
        self.log.append(f"  Sweep complete: freed {len(to_remove)} objects")
        return len(to_remove)

    def run_collection(self) -> int:
        """Run a full concurrent collection with simulated mutator pauses."""
        self.start_marking()
        while self.phase == "marking":
            self.mark_increment(work=2)
            self.log.append(f"    (mutator runs here...)")
        return self.sweep()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_tricolor():
    print("=" * 60)
    print("1. Tri-Color Incremental Marking")
    print("=" * 60)

    gc = TriColorCollector()
    a = gc.allocate("A")
    b = gc.allocate("B")
    c = gc.allocate("C")
    d = gc.allocate("D")
    e = gc.allocate("E")

    gc.roots = {a}
    gc.add_ref(a, b)
    gc.add_ref(b, c)
    # D and E are unreachable

    gc.init_marking()
    print(f"\n  After init: {gc.snapshot()}")

    gc.mark_step(steps=1)
    print(f"  After step 1: {gc.snapshot()}")

    gc.mark_step(steps=1)
    print(f"  After step 2: {gc.snapshot()}")

    gc.mark_step(steps=1)
    print(f"  After step 3: {gc.snapshot()}")

    freed = gc.sweep()
    print(f"  After sweep: freed {freed} objects ({gc.freed})")
    print(f"  Alive: {[obj.name for obj in gc.heap.values()]}")


def demo_write_barrier():
    print("\n" + "=" * 60)
    print("2. Write Barrier (Dijkstra-style)")
    print("=" * 60)

    gc = TriColorCollector()
    a = gc.allocate("A")
    b = gc.allocate("B")
    c = gc.allocate("C")

    gc.roots = {a}
    gc.add_ref(a, b)

    # Start marking
    gc.init_marking()
    gc.mark_step(steps=1)  # A -> BLACK, B -> GRAY
    gc.mark_step(steps=1)  # B -> BLACK
    print(f"\n  During marking: {gc.snapshot()}")

    # Mutator adds reference from BLACK A to WHITE C
    print(f"  Mutator: A.ref = C")
    gc.add_ref(a, c)

    for entry in gc.write_barrier_log:
        print(entry)

    # Continue marking
    while not gc.mark_step(steps=1):
        pass
    freed = gc.sweep()
    print(f"  After collection: {gc.snapshot()}")
    print(f"  C survived due to write barrier: {'C' not in gc.freed}")


def demo_escape_analysis():
    print("\n" + "=" * 60)
    print("3. Escape Analysis")
    print("=" * 60)

    # Simulate analysis of a function
    func = FuncInfo(
        name="process",
        allocations=["point", "temp_buf", "result", "cache_entry"],
        returns=["result"],
        stores_to_global=["cache_entry"],
        calls_with={"send_network": ["point"]}
    )

    analyzer = EscapeAnalyzer()
    results = analyzer.analyze(func)

    print(f"\n  Function: {func.name}()")
    print(f"  {'Allocation':<15} {'Escapes':<10} {'Reason'}")
    print(f"  {'─'*15} {'─'*10} {'─'*35}")
    for name, alloc in results.items():
        esc = "YES" if alloc.escapes else "NO"
        print(f"  {name:<15} {esc:<10} {alloc.reason}")

    stack_allocs = [a.name for a in results.values() if not a.escapes]
    print(f"\n  Stack-allocatable: {stack_allocs}")


def demo_concurrent():
    print("\n" + "=" * 60)
    print("4. Concurrent Collector Simulation")
    print("=" * 60)

    gc = ConcurrentCollector()
    a = gc.allocate("Root")
    b = gc.allocate("Child1")
    c = gc.allocate("Child2")
    d = gc.allocate("Garbage1")
    e = gc.allocate("Garbage2")

    gc.roots = {a}
    gc.heap[a].refs = [b]
    gc.heap[b].refs = [c]

    freed = gc.run_collection()
    print()
    for entry in gc.log:
        print(entry)
    print(f"\n  Freed: {gc.freed}")
    print(f"  Alive: {[obj.name for obj in gc.heap.values()]}")


def main():
    demo_tricolor()
    demo_write_barrier()
    demo_escape_analysis()
    demo_concurrent()

    print("\n" + "=" * 60)
    print("Advanced GC Summary")
    print("=" * 60)
    print("""
  Technique           Purpose
  ─────────────────── ──────────────────────────────────────────
  Tri-color marking   Incremental/concurrent GC correctness
  Write barriers      Maintain invariant during concurrent mutation
  Escape analysis     Avoid heap allocation for non-escaping objects
  Concurrent GC       Reduce pause times by overlapping with mutator
    """)


if __name__ == "__main__":
    main()
