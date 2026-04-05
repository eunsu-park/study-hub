"""
14_garbage_collector.py - Garbage Collection Algorithms

Demonstrates fundamental garbage collection strategies for automatic
memory management in language runtimes:

  1. Reference Counting
     Track the number of references to each object. When the count
     drops to zero the object is immediately freed.

  2. Mark-Sweep Collector
     Starting from root references, mark all reachable objects, then
     sweep through the heap and free unmarked objects.

  3. Copying Collector (Cheney's Algorithm)
     Divide the heap into two semi-spaces. Copy live objects from
     the "from" space to the "to" space, compacting memory.

  4. Generational Collector
     Partition objects by age. Collect the young generation frequently
     (most objects die young) and promote survivors to an older
     generation that is collected less often.

Topics covered:
  - Root set identification
  - Tri-color marking abstraction
  - Reference counting with cycle detection
  - Semi-space copying and forwarding pointers
  - Generational hypothesis and write barriers
  - GC metrics: pause time, throughput, fragmentation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Heap Object representation
# ---------------------------------------------------------------------------

@dataclass
class HeapObject:
    """Represents an object on the managed heap."""
    oid: int
    name: str
    refs: list[int] = field(default_factory=list)  # ids of referenced objects
    ref_count: int = 0
    marked: bool = False
    generation: int = 0  # 0 = young, 1 = old

    def __repr__(self):
        return f"Obj({self.oid}:{self.name})"


# ---------------------------------------------------------------------------
# 1. Reference Counting Collector
# ---------------------------------------------------------------------------

class RefCountGC:
    """
    Simple reference counting garbage collector.
    Each object tracks how many references point to it.
    Objects are freed immediately when their count reaches zero.
    Limitation: cannot collect reference cycles.
    """

    def __init__(self):
        self.heap: dict[int, HeapObject] = {}
        self.roots: set[int] = set()
        self.freed: list[str] = []
        self.next_id = 0

    def allocate(self, name: str) -> int:
        oid = self.next_id
        self.next_id += 1
        obj = HeapObject(oid=oid, name=name, ref_count=0)
        self.heap[oid] = obj
        return oid

    def add_root(self, oid: int) -> None:
        self.roots.add(oid)
        self.heap[oid].ref_count += 1

    def remove_root(self, oid: int) -> None:
        self.roots.discard(oid)
        self._decrement(oid)

    def add_ref(self, from_oid: int, to_oid: int) -> None:
        self.heap[from_oid].refs.append(to_oid)
        self.heap[to_oid].ref_count += 1

    def remove_ref(self, from_oid: int, to_oid: int) -> None:
        self.heap[from_oid].refs.remove(to_oid)
        self._decrement(to_oid)

    def _decrement(self, oid: int) -> None:
        obj = self.heap.get(oid)
        if obj is None:
            return
        obj.ref_count -= 1
        if obj.ref_count <= 0:
            self.freed.append(obj.name)
            # Recursively decrement children
            for child_id in obj.refs:
                self._decrement(child_id)
            del self.heap[oid]

    def status(self) -> str:
        alive = [f"{o.name}(rc={o.ref_count})" for o in self.heap.values()]
        return f"  Alive: {alive}, Freed: {self.freed}"


# ---------------------------------------------------------------------------
# 2. Mark-Sweep Collector
# ---------------------------------------------------------------------------

class MarkSweepGC:
    """
    Tracing garbage collector using mark-sweep.
    Phase 1 (Mark): DFS from roots, marking reachable objects.
    Phase 2 (Sweep): Free all unmarked objects.
    """

    def __init__(self):
        self.heap: dict[int, HeapObject] = {}
        self.roots: set[int] = set()
        self.freed: list[str] = []
        self.next_id = 0

    def allocate(self, name: str) -> int:
        oid = self.next_id
        self.next_id += 1
        self.heap[oid] = HeapObject(oid=oid, name=name)
        return oid

    def add_ref(self, from_oid: int, to_oid: int) -> None:
        self.heap[from_oid].refs.append(to_oid)

    def collect(self) -> int:
        """Run a full mark-sweep collection. Returns number of objects freed."""
        # Reset marks
        for obj in self.heap.values():
            obj.marked = False

        # Mark phase: DFS from roots
        worklist = list(self.roots)
        while worklist:
            oid = worklist.pop()
            obj = self.heap.get(oid)
            if obj is None or obj.marked:
                continue
            obj.marked = True
            for child in obj.refs:
                if child in self.heap and not self.heap[child].marked:
                    worklist.append(child)

        # Sweep phase: free unmarked objects
        to_free = [oid for oid, obj in self.heap.items() if not obj.marked]
        for oid in to_free:
            self.freed.append(self.heap[oid].name)
            del self.heap[oid]

        return len(to_free)


# ---------------------------------------------------------------------------
# 3. Copying Collector (Cheney's Algorithm)
# ---------------------------------------------------------------------------

class CopyingGC:
    """
    Semi-space copying collector.
    Maintains two spaces: 'from_space' and 'to_space'.
    During collection, live objects are copied from from_space to to_space,
    then the spaces are swapped.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.from_space: dict[int, HeapObject] = {}
        self.to_space: dict[int, HeapObject] = {}
        self.roots: set[int] = set()
        self.next_id = 0
        self.collections = 0

    def allocate(self, name: str) -> int:
        if len(self.from_space) >= self.capacity:
            self.collect()
        oid = self.next_id
        self.next_id += 1
        self.from_space[oid] = HeapObject(oid=oid, name=name)
        return oid

    def add_ref(self, from_oid: int, to_oid: int) -> None:
        self.from_space[from_oid].refs.append(to_oid)

    def collect(self) -> int:
        """Copy live objects from from_space to to_space, then swap."""
        self.collections += 1
        self.to_space.clear()
        forwarding: dict[int, int] = {}  # old_id -> new_id (same id here)

        # BFS copy starting from roots (Cheney's scan pointer approach)
        scan_queue: list[int] = []

        def copy_obj(oid: int) -> int:
            if oid in forwarding:
                return forwarding[oid]
            obj = self.from_space.get(oid)
            if obj is None:
                return oid
            new_obj = HeapObject(oid=obj.oid, name=obj.name,
                                 refs=list(obj.refs))
            self.to_space[obj.oid] = new_obj
            forwarding[oid] = obj.oid
            scan_queue.append(obj.oid)
            return obj.oid

        # Copy root objects
        new_roots = set()
        for root_id in self.roots:
            new_roots.add(copy_obj(root_id))
        self.roots = new_roots

        # Scan copied objects and update their references
        scan_idx = 0
        while scan_idx < len(scan_queue):
            oid = scan_queue[scan_idx]
            scan_idx += 1
            obj = self.to_space[oid]
            obj.refs = [copy_obj(ref) for ref in obj.refs]

        freed_count = len(self.from_space) - len(self.to_space)

        # Swap spaces
        self.from_space = self.to_space
        self.to_space = {}

        return freed_count


# ---------------------------------------------------------------------------
# 4. Generational Collector
# ---------------------------------------------------------------------------

class GenerationalGC:
    """
    Two-generation garbage collector.
    Young objects are collected frequently (minor GC).
    Survivors are promoted to the old generation.
    Old generation is collected less often (major GC).
    """

    def __init__(self, young_limit: int = 8, promote_age: int = 2):
        self.young: dict[int, HeapObject] = {}
        self.old: dict[int, HeapObject] = {}
        self.roots: set[int] = set()
        self.ages: dict[int, int] = {}  # oid -> survival count
        self.next_id = 0
        self.young_limit = young_limit
        self.promote_age = promote_age
        self.minor_collections = 0
        self.major_collections = 0
        self.freed: list[str] = []

    def allocate(self, name: str) -> int:
        if len(self.young) >= self.young_limit:
            self.minor_collect()
        oid = self.next_id
        self.next_id += 1
        obj = HeapObject(oid=oid, name=name, generation=0)
        self.young[oid] = obj
        self.ages[oid] = 0
        return oid

    def add_ref(self, from_oid: int, to_oid: int) -> None:
        obj = self.young.get(from_oid) or self.old.get(from_oid)
        if obj:
            obj.refs.append(to_oid)

    def _all_objects(self) -> dict[int, HeapObject]:
        return {**self.young, **self.old}

    def _mark_reachable(self, heap: dict[int, HeapObject],
                        roots: set[int]) -> set[int]:
        """Mark and return the set of reachable object ids."""
        all_objs = self._all_objects()
        reachable: set[int] = set()
        worklist = [r for r in roots if r in heap or r in all_objs]
        while worklist:
            oid = worklist.pop()
            if oid in reachable:
                continue
            reachable.add(oid)
            obj = all_objs.get(oid)
            if obj:
                for child in obj.refs:
                    if child not in reachable:
                        worklist.append(child)
        return reachable

    def minor_collect(self) -> int:
        """Collect the young generation. Promote survivors."""
        self.minor_collections += 1
        reachable = self._mark_reachable(self.young, self.roots)
        freed = 0

        to_remove = []
        for oid, obj in self.young.items():
            if oid not in reachable:
                self.freed.append(obj.name)
                to_remove.append(oid)
                freed += 1
            else:
                self.ages[oid] = self.ages.get(oid, 0) + 1
                if self.ages[oid] >= self.promote_age:
                    obj.generation = 1
                    self.old[oid] = obj
                    to_remove.append(oid)

        for oid in to_remove:
            self.young.pop(oid, None)

        return freed

    def major_collect(self) -> int:
        """Full collection of both generations."""
        self.major_collections += 1
        freed = self.minor_collect()
        reachable = self._mark_reachable(self.old, self.roots)

        to_remove = []
        for oid, obj in self.old.items():
            if oid not in reachable:
                self.freed.append(obj.name)
                to_remove.append(oid)
                freed += 1

        for oid in to_remove:
            del self.old[oid]

        return freed


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_ref_counting():
    print("=" * 60)
    print("1. Reference Counting GC")
    print("=" * 60)

    gc = RefCountGC()
    a = gc.allocate("A")
    b = gc.allocate("B")
    c = gc.allocate("C")

    gc.add_root(a)
    gc.add_ref(a, b)
    gc.add_ref(b, c)
    print(f"\n  After setup (A -> B -> C):")
    print(gc.status())

    gc.remove_ref(a, b)
    print(f"\n  After removing A -> B:")
    print(gc.status())

    # Demonstrate cycle limitation
    gc2 = RefCountGC()
    x = gc2.allocate("X")
    y = gc2.allocate("Y")
    gc2.add_root(x)
    gc2.add_ref(x, y)
    gc2.add_ref(y, x)  # cycle
    gc2.remove_root(x)
    print(f"\n  Cycle demo (X <-> Y), after removing root:")
    print(gc2.status())
    print("  Note: cycle not collected (ref counting limitation)")


def demo_mark_sweep():
    print("\n" + "=" * 60)
    print("2. Mark-Sweep GC")
    print("=" * 60)

    gc = MarkSweepGC()
    a = gc.allocate("A")
    b = gc.allocate("B")
    c = gc.allocate("C")
    d = gc.allocate("D")  # unreachable

    gc.roots = {a}
    gc.add_ref(a, b)
    gc.add_ref(b, c)
    # D is not reachable from any root

    # Also create a cycle that IS reachable
    gc.add_ref(c, a)

    alive_before = len(gc.heap)
    freed = gc.collect()
    print(f"\n  Before GC: {alive_before} objects")
    print(f"  After GC:  {len(gc.heap)} objects (freed {freed})")
    print(f"  Freed: {gc.freed}")
    print(f"  Alive: {[o.name for o in gc.heap.values()]}")
    print(f"  Note: cycle A->B->C->A survived (all reachable from root)")


def demo_copying():
    print("\n" + "=" * 60)
    print("3. Copying Collector (Cheney's Algorithm)")
    print("=" * 60)

    gc = CopyingGC(capacity=5)
    a = gc.allocate("A")
    b = gc.allocate("B")
    c = gc.allocate("C")
    gc.roots = {a, b}
    gc.add_ref(a, c)

    print(f"\n  From-space: {[o.name for o in gc.from_space.values()]}")
    freed = gc.collect()
    print(f"  After collection (freed {freed}):")
    print(f"  From-space: {[o.name for o in gc.from_space.values()]}")
    print(f"  Collections performed: {gc.collections}")

    # Allocate more to trigger automatic collection
    for i in range(6):
        gc.allocate(f"T{i}")
    print(f"\n  After allocating 6 more objects:")
    print(f"  From-space size: {len(gc.from_space)}")
    print(f"  Total collections: {gc.collections}")


def demo_generational():
    print("\n" + "=" * 60)
    print("4. Generational GC")
    print("=" * 60)

    gc = GenerationalGC(young_limit=4, promote_age=2)

    # Allocate objects, some short-lived, some long-lived
    root = gc.allocate("Root")
    gc.roots = {root}

    long_lived = gc.allocate("LongLived")
    gc.add_ref(root, long_lived)

    # Allocate short-lived objects that trigger minor collections
    for i in range(10):
        temp = gc.allocate(f"Temp{i}")
        # temp is not referenced by anything reachable

    print(f"\n  Young generation: {[o.name for o in gc.young.values()]}")
    print(f"  Old generation:   {[o.name for o in gc.old.values()]}")
    print(f"  Minor collections: {gc.minor_collections}")
    print(f"  Freed: {gc.freed}")

    # Force a major collection
    freed = gc.major_collect()
    print(f"\n  After major collection (freed {freed}):")
    print(f"  Young: {[o.name for o in gc.young.values()]}")
    print(f"  Old:   {[o.name for o in gc.old.values()]}")
    print(f"  Total freed: {gc.freed}")


def main():
    demo_ref_counting()
    demo_mark_sweep()
    demo_copying()
    demo_generational()

    print("\n" + "=" * 60)
    print("GC Algorithm Comparison")
    print("=" * 60)
    print("""
  Algorithm          Pros                     Cons
  ────────────────── ──────────────────────── ────────────────────────
  Reference Counting Immediate reclamation,   Cannot collect cycles,
                     simple, predictable      overhead on every write
  Mark-Sweep         Collects cycles,         Stop-the-world pause,
                     no write barrier needed  fragmentation
  Copying            No fragmentation,        Half memory wasted,
                     fast allocation           copying cost
  Generational       Fast minor collections,  Write barrier overhead,
                     exploits object lifetime complex implementation
    """)


if __name__ == "__main__":
    main()
