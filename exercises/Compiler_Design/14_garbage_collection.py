"""
Exercises for Lesson 14: Garbage Collection (Basics)
Topic: Compiler_Design

Solutions to practice problems covering reference counting, mark-sweep,
mark-compact, copying collectors, the generational hypothesis, and
root set identification.

Note: This covers foundational GC algorithms. See exercise 17 for advanced
topics (cycle detection, tri-color marking, generational simulation,
escape analysis, GC benchmarking).
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional


# === Exercise 1: Reference Counting ===
# Problem: Implement a reference counting memory manager and trace its
# behavior, including the cascade release effect.

@dataclass
class RCObject:
    """An object managed by reference counting."""
    name: str
    ref_count: int = 0
    fields: Dict[str, Optional['RCObject']] = field(default_factory=dict)
    alive: bool = True

    def __repr__(self):
        return f"{self.name}(rc={self.ref_count})"


class RCMemoryManager:
    """Simple reference counting memory manager."""

    def __init__(self):
        self.objects: List[RCObject] = []
        self.log: List[str] = []

    def allocate(self, name: str) -> RCObject:
        obj = RCObject(name)
        self.objects.append(obj)
        self.log.append(f"ALLOC {name}")
        return obj

    def add_ref(self, obj: RCObject):
        """Increment reference count."""
        obj.ref_count += 1
        self.log.append(f"INCR {obj.name} -> rc={obj.ref_count}")

    def release(self, obj: RCObject):
        """Decrement reference count. Free if zero."""
        obj.ref_count -= 1
        self.log.append(f"DECR {obj.name} -> rc={obj.ref_count}")
        if obj.ref_count == 0:
            self._free(obj)

    def write_field(self, container: RCObject, field_name: str, new_val: Optional[RCObject]):
        """Write a field, adjusting reference counts."""
        old_val = container.fields.get(field_name)
        if old_val is not None:
            self.release(old_val)
        container.fields[field_name] = new_val
        if new_val is not None:
            self.add_ref(new_val)
        self.log.append(f"WRITE {container.name}.{field_name} = "
                        f"{new_val.name if new_val else 'null'}")

    def _free(self, obj: RCObject):
        """Free an object: decrement refs to children, mark dead."""
        self.log.append(f"FREE {obj.name}")
        obj.alive = False
        for field_name, child in list(obj.fields.items()):
            if child is not None and child.alive:
                self.release(child)
            obj.fields[field_name] = None

    def live_objects(self) -> List[str]:
        return [o.name for o in self.objects if o.alive]


def exercise_1():
    """Reference counting with cascade release."""
    print("Scenario: Build a linked list A -> B -> C -> D, then release A.")
    print()

    mm = RCMemoryManager()

    # Allocate objects
    a = mm.allocate("A")
    b = mm.allocate("B")
    c = mm.allocate("C")
    d = mm.allocate("D")

    # Build chain: A -> B -> C -> D
    # Root holds A
    mm.add_ref(a)  # root ref
    mm.write_field(a, "next", b)
    mm.write_field(b, "next", c)
    mm.write_field(c, "next", d)

    print("After building A -> B -> C -> D (root holds A):")
    for obj in [a, b, c, d]:
        print(f"  {obj.name}: rc={obj.ref_count}, alive={obj.alive}")
    print()

    # Release root reference to A
    print("Releasing root reference to A...")
    mm.release(a)
    print()

    print("After release -- cascade effect:")
    for obj in [a, b, c, d]:
        print(f"  {obj.name}: rc={obj.ref_count}, alive={obj.alive}")
    print()

    print(f"Live objects: {mm.live_objects()}")
    print()

    print("Operation log:")
    # Show the cascade portion
    cascade_start = next(i for i, l in enumerate(mm.log) if l == "DECR A -> rc=0")
    for entry in mm.log[cascade_start:]:
        print(f"  {entry}")
    print()
    print("Observation: Freeing A triggers a cascade: A frees B, B frees C, C frees D.")
    print("This cascade can cause pause spikes in real systems (deep object graphs).")


# === Exercise 2: Reference Counting Cycle Leak ===
# Problem: Show how reference counting fails to collect cyclic structures.

def exercise_2():
    """Demonstrate the reference counting cycle problem."""
    print("Scenario: Create a cycle A -> B -> A, then remove external references.")
    print()

    mm = RCMemoryManager()

    a = mm.allocate("A")
    b = mm.allocate("B")

    # External root refs
    mm.add_ref(a)  # root -> A
    mm.add_ref(b)  # root -> B

    # Create cycle
    mm.write_field(a, "ref", b)  # A -> B
    mm.write_field(b, "ref", a)  # B -> A (cycle!)

    print("After creating cycle A <-> B with root refs:")
    print(f"  A: rc={a.ref_count} (root + B.ref)")
    print(f"  B: rc={b.ref_count} (root + A.ref)")
    print()

    # Remove root references
    print("Removing root reference to A...")
    mm.release(a)
    print(f"  A: rc={a.ref_count}, alive={a.alive}")
    print()

    print("Removing root reference to B...")
    mm.release(b)
    print(f"  B: rc={b.ref_count}, alive={b.alive}")
    print()

    print(f"Live objects: {mm.live_objects()}")
    print()

    print("Problem: Both A and B are still 'alive' with rc=1 each.")
    print("They reference each other, but no external reference exists.")
    print("This is a MEMORY LEAK -- pure reference counting cannot collect cycles.")
    print()
    print("Solutions to the cycle problem:")
    print("  1. Weak references (programmer manually breaks cycles)")
    print("  2. Backup tracing collector (Python uses this)")
    print("  3. Trial deletion / Bacon-Rajan algorithm")
    print("  4. Use tracing GC instead (mark-sweep, etc.)")


# === Exercise 3: Mark-Sweep Collector ===
# Problem: Implement mark-sweep GC and trace its phases.

@dataclass
class HeapObject:
    """Object on the heap for mark-sweep GC."""
    obj_id: int
    name: str
    marked: bool = False
    refs: List['HeapObject'] = field(default_factory=list)

    def __repr__(self):
        return f"{self.name}(id={self.obj_id}, marked={self.marked})"


class MarkSweepCollector:
    """Mark-sweep garbage collector."""

    def __init__(self):
        self.heap: List[HeapObject] = []
        self._next_id = 0

    def allocate(self, name: str) -> HeapObject:
        self._next_id += 1
        obj = HeapObject(self._next_id, name)
        self.heap.append(obj)
        return obj

    def collect(self, roots: List[HeapObject]) -> Tuple[List[str], List[str]]:
        """Run mark-sweep. Return (marked_names, swept_names)."""
        # Phase 1: Mark
        mark_order = []
        stack = list(roots)
        while stack:
            obj = stack.pop()
            if not obj.marked:
                obj.marked = True
                mark_order.append(obj.name)
                for child in obj.refs:
                    if not child.marked:
                        stack.append(child)

        # Phase 2: Sweep
        swept = []
        new_heap = []
        for obj in self.heap:
            if obj.marked:
                obj.marked = False  # Reset for next collection
                new_heap.append(obj)
            else:
                swept.append(obj.name)
        self.heap = new_heap

        return mark_order, swept

    def heap_contents(self) -> List[str]:
        return [obj.name for obj in self.heap]


def exercise_3():
    """Mark-sweep GC step-by-step trace."""
    print("Object graph:")
    print("  Root1 -> A -> B -> C")
    print("                |")
    print("                v")
    print("                D")
    print("  Root2 -> E")
    print("  (unreachable: F -> G, H)")
    print()

    gc = MarkSweepCollector()

    a = gc.allocate("A")
    b = gc.allocate("B")
    c = gc.allocate("C")
    d = gc.allocate("D")
    e = gc.allocate("E")
    f = gc.allocate("F")
    g = gc.allocate("G")
    h = gc.allocate("H")

    # Build references
    a.refs = [b]
    b.refs = [c, d]
    f.refs = [g]  # F -> G, but F is unreachable

    roots = [a, e]

    print(f"Heap before GC: {gc.heap_contents()}")
    print(f"Roots: [A, E]")
    print()

    marked, swept = gc.collect(roots)

    print("Phase 1 -- Mark (DFS from roots):")
    print(f"  Mark order: {marked}")
    print()
    print("Phase 2 -- Sweep (scan heap, free unmarked):")
    print(f"  Swept (freed): {swept}")
    print()
    print(f"Heap after GC: {gc.heap_contents()}")
    print()

    print("Analysis:")
    print("  - A, B, C, D reachable from Root1 -> kept")
    print("  - E reachable from Root2 -> kept")
    print("  - F, G form a cycle but are unreachable -> correctly collected")
    print("  - H is isolated and unreachable -> collected")
    print()
    print("Mark-sweep properties:")
    print("  + Handles cycles (unlike reference counting)")
    print("  + No overhead during normal execution (no ref count updates)")
    print("  - Stop-the-world: must pause mutator during collection")
    print("  - Causes heap fragmentation (freed objects leave holes)")
    print("  - Cost proportional to heap size (must scan entire heap in sweep)")


# === Exercise 4: Mark-Compact Collector ===
# Problem: Simulate mark-compact collection showing how it eliminates
# fragmentation by sliding live objects.

def exercise_4():
    """Mark-compact collector simulation."""
    print("Heap layout (each object has an address and size):")
    print()

    # Simulate a heap with addresses
    @dataclass
    class CompactObj:
        name: str
        address: int
        size: int
        marked: bool = False
        forward_addr: Optional[int] = None

    heap = [
        CompactObj("A", 0, 4, marked=True),     # live
        CompactObj("X", 4, 2, marked=False),     # dead
        CompactObj("B", 6, 3, marked=True),      # live
        CompactObj("Y", 9, 5, marked=False),     # dead
        CompactObj("C", 14, 2, marked=True),     # live
        CompactObj("Z", 16, 4, marked=False),    # dead
        CompactObj("D", 20, 6, marked=True),     # live
    ]

    print("Before compaction:")
    print("  Addr  Size  Name  Status")
    print("  ----  ----  ----  ------")
    for obj in heap:
        status = "LIVE" if obj.marked else "DEAD"
        bar = "#" * obj.size if obj.marked else "." * obj.size
        print(f"  {obj.address:4d}  {obj.size:4d}  {obj.name:4s}  {status}  [{bar}]")

    total_size = sum(o.size for o in heap)
    live_size = sum(o.size for o in heap if o.marked)
    dead_size = total_size - live_size
    print(f"\n  Total: {total_size}, Live: {live_size}, Dead (fragmented): {dead_size}")
    print()

    # Phase 1: Compute forwarding addresses
    print("Phase 1: Compute forwarding addresses")
    next_free = 0
    for obj in heap:
        if obj.marked:
            obj.forward_addr = next_free
            print(f"  {obj.name}: {obj.address} -> {obj.forward_addr}")
            next_free += obj.size
    print()

    # Phase 2: Update references (conceptual -- we just show the addresses)
    print("Phase 2: Update all references to use forwarding addresses")
    print("  (All pointers to A now point to address 0)")
    print("  (All pointers to B now point to address 4)")
    print("  (All pointers to C now point to address 7)")
    print("  (All pointers to D now point to address 9)")
    print()

    # Phase 3: Slide objects
    print("Phase 3: Slide objects to forwarding addresses")
    compacted = [obj for obj in heap if obj.marked]
    for obj in compacted:
        obj.address = obj.forward_addr
        obj.forward_addr = None
        obj.marked = False

    print("\nAfter compaction:")
    print("  Addr  Size  Name")
    print("  ----  ----  ----")
    for obj in compacted:
        bar = "#" * obj.size
        print(f"  {obj.address:4d}  {obj.size:4d}  {obj.name:4s}  [{bar}]")
    free_start = sum(o.size for o in compacted)
    free_size = total_size - free_start
    print(f"  {free_start:4d}  {free_size:4d}  FREE  [{'_' * free_size}]")
    print()

    print("Mark-compact properties:")
    print("  + No fragmentation (contiguous free space)")
    print("  + Better cache locality (live objects are adjacent)")
    print("  + Handles cycles")
    print("  - Three passes over the heap (slower than mark-sweep)")
    print("  - Must update ALL pointers (expensive)")
    print("  - Objects move (invalidates raw pointers)")


# === Exercise 5: Copying Collector (Semi-Space) ===
# Problem: Simulate Cheney's semi-space copying collector.

def exercise_5():
    """Cheney's semi-space copying collector."""
    print("Semi-space copying collector (Cheney's algorithm):")
    print()

    @dataclass
    class CopyObj:
        name: str
        refs: List['CopyObj'] = field(default_factory=list)
        forwarded: bool = False
        forward_to: Optional['CopyObj'] = None

    # From-space objects
    a = CopyObj("A")
    b = CopyObj("B")
    c = CopyObj("C")
    d = CopyObj("D")
    garbage1 = CopyObj("GARB1")
    garbage2 = CopyObj("GARB2")

    a.refs = [b, c]
    b.refs = [d]
    c.refs = [d]  # Shared reference to D
    garbage1.refs = [garbage2]

    from_space = [a, garbage1, b, garbage2, c, d]
    roots = [a]

    print(f"From-space: {[o.name for o in from_space]}")
    print(f"Roots: [{', '.join(r.name for r in roots)}]")
    print(f"Graph: A -> [B, C], B -> D, C -> D, GARB1 -> GARB2")
    print()

    # Cheney's BFS copying algorithm
    to_space: List[CopyObj] = []
    scan_idx = 0

    def copy_if_needed(obj: CopyObj) -> CopyObj:
        if obj.forwarded:
            return obj.forward_to
        new_obj = CopyObj(obj.name + "'", list(obj.refs))
        to_space.append(new_obj)
        obj.forwarded = True
        obj.forward_to = new_obj
        return new_obj

    # Step 1: Copy roots
    print("Step 1: Copy root objects to to-space")
    new_roots = [copy_if_needed(r) for r in roots]
    print(f"  To-space: {[o.name for o in to_space]}")
    print(f"  scan=0, free={len(to_space)}")
    print()

    # BFS scan
    step = 2
    while scan_idx < len(to_space):
        obj = to_space[scan_idx]
        print(f"Step {step}: Scan {obj.name} at index {scan_idx}")

        new_refs = []
        for ref in obj.refs:
            # ref is an original from-space object
            if hasattr(ref, 'forwarded') and ref.forwarded:
                new_ref = ref.forward_to
                print(f"  {ref.name} already copied -> {new_ref.name}")
            else:
                new_ref = copy_if_needed(ref)
                print(f"  Copy {ref.name} -> {new_ref.name}")
            new_refs.append(new_ref)
        obj.refs = new_refs
        scan_idx += 1
        step += 1

        print(f"  To-space: {[o.name for o in to_space]}")
        print(f"  scan={scan_idx}, free={len(to_space)}")
        print()

    print(f"Collection complete!")
    print(f"  Live objects in to-space: {[o.name for o in to_space]}")
    print(f"  GARB1, GARB2 not copied (correctly collected)")
    print(f"  D was referenced by both B and C but copied only once")
    print()

    print("Semi-space properties:")
    print("  + No fragmentation (objects are compacted in to-space)")
    print("  + Allocation is trivial (bump pointer)")
    print("  + Cost proportional to live objects (not total heap)")
    print("  + Handles cycles naturally (forwarding pointer prevents revisit)")
    print("  - Wastes half the memory (two semi-spaces)")
    print("  - Must copy all live objects (expensive if survival rate is high)")
    print("  - Objects move (need to update all references)")


# === Exercise 6: Root Set Identification ===
# Problem: Given a program state, identify the root set and trace
# reachability.

def exercise_6():
    """Root set identification and reachability analysis."""
    print("Program state snapshot:")
    print()
    print("  Call stack:")
    print("    main():")
    print("      local x = -> Obj_1")
    print("      local y = -> Obj_2")
    print("    foo(a):")
    print("      param a = -> Obj_3")
    print("      local b = -> Obj_4")
    print("      local c = 42 (not a pointer)")
    print()
    print("  Global variables:")
    print("    G1 = -> Obj_5")
    print("    G2 = -> Obj_6")
    print()
    print("  Registers:")
    print("    R1 = -> Obj_7 (temporary)")
    print()
    print("  Heap objects:")
    print("    Obj_1.field = -> Obj_8")
    print("    Obj_3.field = -> Obj_9")
    print("    Obj_5.field = -> Obj_10")
    print("    Obj_8.field = -> Obj_3  (shared reference)")
    print("    Obj_11 (no incoming references)")
    print("    Obj_12.field = -> Obj_11")
    print()

    # Root set
    root_set = {"Obj_1", "Obj_2", "Obj_3", "Obj_4", "Obj_5", "Obj_6", "Obj_7"}
    print(f"Root set: {sorted(root_set)}")
    print("  Sources: stack locals (x, y, b), parameters (a),")
    print("           globals (G1, G2), registers (R1)")
    print()

    # Object graph
    graph = {
        "Obj_1":  ["Obj_8"],
        "Obj_2":  [],
        "Obj_3":  ["Obj_9"],
        "Obj_4":  [],
        "Obj_5":  ["Obj_10"],
        "Obj_6":  [],
        "Obj_7":  [],
        "Obj_8":  ["Obj_3"],
        "Obj_9":  [],
        "Obj_10": [],
        "Obj_11": [],
        "Obj_12": ["Obj_11"],
    }

    # Trace reachability from roots
    reachable = set()
    worklist = deque(root_set)
    trace_log = []

    while worklist:
        obj = worklist.popleft()
        if obj in reachable:
            continue
        reachable.add(obj)
        for child in graph.get(obj, []):
            if child not in reachable:
                trace_log.append(f"  {obj} -> {child}")
                worklist.append(child)

    all_objects = set(graph.keys())
    garbage = all_objects - reachable

    print("Tracing from roots:")
    for entry in trace_log:
        print(entry)
    print()
    print(f"Reachable objects: {sorted(reachable)}")
    print(f"Garbage (unreachable): {sorted(garbage)}")
    print()

    print("Analysis:")
    print("  Obj_8: reachable via Obj_1.field (root x -> Obj_1 -> Obj_8)")
    print("  Obj_9: reachable via Obj_3.field (root a -> Obj_3 -> Obj_9)")
    print("  Obj_10: reachable via Obj_5.field (global G1 -> Obj_5 -> Obj_10)")
    print("  Obj_3: reachable TWICE (directly as root, and via Obj_8.field)")
    print("  Obj_11: NOT reachable (only referenced by Obj_12, which is also garbage)")
    print("  Obj_12: NOT reachable (no root can reach it)")
    print()
    print("  Even though Obj_12 -> Obj_11, both are garbage because")
    print("  no path from any root leads to Obj_12.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Reference Counting ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Reference Counting Cycle Leak ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Mark-Sweep Collector ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Mark-Compact Collector ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Copying Collector (Semi-Space) ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Root Set Identification ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
