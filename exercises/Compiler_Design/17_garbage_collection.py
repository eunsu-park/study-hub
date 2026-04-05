"""
Exercises for Lesson 17: Garbage Collection and Memory Management
Topic: Compiler_Design

Solutions to practice problems covering cycle detection, tri-color marking,
generational GC, Cheney's algorithm, escape analysis, and GC benchmarking.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple, Optional
import time
import random


# === Exercise 1: Cycle Detection (Bacon-Rajan) ===

class Color(Enum):
    BLACK = "black"
    PURPLE = "purple"
    GREY = "grey"
    WHITE = "white"


@dataclass
class RCObject:
    name: str
    ref_count: int = 0
    color: Color = Color.BLACK
    buffered: bool = False
    children: List['RCObject'] = field(default_factory=list)

    def __repr__(self):
        return f"{self.name}(rc={self.ref_count}, {self.color.value})"


class CycleCollector:
    """Bacon-Rajan synchronous cycle collector."""

    def __init__(self):
        self.roots_buffer: List[RCObject] = []
        self.freed: List[str] = []

    def add_ref(self, obj: RCObject):
        obj.ref_count += 1
        obj.color = Color.BLACK

    def remove_ref(self, obj: RCObject):
        obj.ref_count -= 1
        if obj.ref_count == 0:
            self._release(obj)
        else:
            self._possible_root(obj)

    def _possible_root(self, obj: RCObject):
        if obj.color != Color.PURPLE:
            obj.color = Color.PURPLE
            if not obj.buffered:
                obj.buffered = True
                self.roots_buffer.append(obj)

    def _release(self, obj: RCObject):
        for child in obj.children:
            self.remove_ref(child)
        obj.color = Color.BLACK
        if not obj.buffered:
            self.freed.append(obj.name)

    def collect_cycles(self) -> List[str]:
        """Run 3-phase cycle collection. Return list of freed names."""
        # Phase 1: Mark grey (trial deletion)
        for root in self.roots_buffer:
            self._mark_grey(root)

        # Phase 2: Scan (identify garbage)
        for root in self.roots_buffer:
            self._scan(root)

        # Phase 3: Collect white
        collected = []
        for root in list(self.roots_buffer):
            root.buffered = False
            if root.color == Color.WHITE:
                self._collect_white(root, collected)

        self.roots_buffer.clear()
        return collected

    def _mark_grey(self, obj: RCObject):
        if obj.color != Color.GREY:
            obj.color = Color.GREY
            for child in obj.children:
                child.ref_count -= 1
                self._mark_grey(child)

    def _scan(self, obj: RCObject):
        if obj.color == Color.GREY:
            if obj.ref_count > 0:
                self._scan_black(obj)
            else:
                obj.color = Color.WHITE
                for child in obj.children:
                    self._scan(child)

    def _scan_black(self, obj: RCObject):
        obj.color = Color.BLACK
        for child in obj.children:
            child.ref_count += 1
            if child.color != Color.BLACK:
                self._scan_black(child)

    def _collect_white(self, obj: RCObject, collected: List[str]):
        if obj.color == Color.WHITE:
            obj.color = Color.BLACK
            collected.append(obj.name)
            for child in obj.children:
                self._collect_white(child, collected)


def exercise_1():
    """Demonstrate Bacon-Rajan cycle collection."""
    print("Exercise 1: Cycle Detection (Bacon-Rajan)")
    print("=" * 50)

    cc = CycleCollector()

    # Create objects
    a = RCObject("A")
    b = RCObject("B")
    c = RCObject("C")
    d = RCObject("D")
    e = RCObject("E")
    root = RCObject("Root")

    # Build graph: A -> B -> C -> A (cycle), Root -> D -> E
    a.children.append(b)
    cc.add_ref(b)  # A -> B
    b.children.append(c)
    cc.add_ref(c)  # B -> C
    c.children.append(a)
    cc.add_ref(a)  # C -> A (cycle!)

    d.children.append(e)
    cc.add_ref(e)  # D -> E
    root.children.append(d)
    cc.add_ref(d)  # Root -> D

    # External refs to A and Root
    cc.add_ref(a)   # "external" ref to A
    cc.add_ref(root) # "external" ref to Root

    print("Before removing external ref to A:")
    print(f"  A: rc={a.ref_count}, B: rc={b.ref_count}, "
          f"C: rc={c.ref_count}")
    print(f"  D: rc={d.ref_count}, E: rc={e.ref_count}")

    # Remove external reference to A
    print("\nRemoving external ref to A...")
    cc.remove_ref(a)

    print(f"\nAfter decrement (A now purple candidate):")
    print(f"  A: rc={a.ref_count}, color={a.color.value}")
    print(f"  B: rc={b.ref_count}, C: rc={c.ref_count}")

    # Run cycle collection
    print("\nRunning cycle collection...")
    collected = cc.collect_cycles()
    print(f"  Cycle-collected: {collected}")
    print(f"  Freed by refcount: {cc.freed}")

    # Verify D and E survive
    print(f"\n  D survives: rc={d.ref_count}, color={d.color.value}")
    print(f"  E survives: rc={e.ref_count}, color={e.color.value}")
    print()


# === Exercise 2: Tri-Color Marking Safety ===

class TriColor(Enum):
    WHITE = "white"
    GREY = "grey"
    BLACK = "black"


@dataclass
class TCObject:
    name: str
    color: TriColor = TriColor.WHITE
    refs: Dict[str, 'TCObject'] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.name}({self.color.value})"


def exercise_2():
    """Demonstrate tri-color marking safety and write barriers."""
    print("Exercise 2: Tri-Color Marking Safety")
    print("=" * 50)

    # (a) No mutator interference
    print("\n(a) Correct marking without mutator interference:")
    a = TCObject("A", TriColor.BLACK, {})
    b = TCObject("B", TriColor.GREY, {})
    c = TCObject("C", TriColor.WHITE, {})
    d = TCObject("D", TriColor.WHITE, {})
    a.refs["child"] = b
    b.refs["child"] = c
    c.refs["child"] = d

    worklist = deque([b])  # Grey objects
    step = 0
    while worklist:
        obj = worklist.popleft()
        step += 1
        for ref in obj.refs.values():
            if ref.color == TriColor.WHITE:
                ref.color = TriColor.GREY
                worklist.append(ref)
        obj.color = TriColor.BLACK
        print(f"  Step {step}: Scanned {obj.name} -> BLACK")
        print(f"    States: A={a.color.value}, B={b.color.value}, "
              f"C={c.color.value}, D={d.color.value}")

    print("  Result: All reachable objects are BLACK (correct)")

    # (b) Lost object problem
    print("\n(b) Lost object problem (no barrier):")
    a = TCObject("A", TriColor.BLACK, {})
    b = TCObject("B", TriColor.GREY, {})
    c = TCObject("C", TriColor.WHITE, {})
    d = TCObject("D", TriColor.WHITE, {})
    a.refs["child"] = b
    b.refs["child"] = c
    c.refs["child"] = d

    # Mutator: A.ref2 = D, B.child = null
    print("  Mutator: A.ref2 = D (BLACK -> WHITE violation!)")
    a.refs["ref2"] = d
    print("  Mutator: B.child = null")
    del b.refs["child"]

    # Now scanner processes B
    print("  Scanner processes B: no children to grey")
    b.color = TriColor.BLACK
    # C and D remain WHITE -- incorrectly garbage!
    print(f"  C is {c.color.value} -- LOST (will be freed incorrectly)")
    print(f"  D is {d.color.value} -- LOST (will be freed incorrectly)")

    # (c) Dijkstra's insertion barrier
    print("\n(c) Dijkstra's insertion barrier prevents the bug:")
    a = TCObject("A", TriColor.BLACK, {})
    b = TCObject("B", TriColor.GREY, {})
    c = TCObject("C", TriColor.WHITE, {})
    d = TCObject("D", TriColor.WHITE, {})
    a.refs["child"] = b
    b.refs["child"] = c
    c.refs["child"] = d
    worklist = deque([b])

    # Mutator with insertion barrier: A.ref2 = D
    print("  Mutator: A.ref2 = D (with Dijkstra barrier)")
    # Barrier: new target D is WHITE -> grey it
    d.color = TriColor.GREY
    worklist.append(d)
    a.refs["ref2"] = d
    print(f"  Barrier greys D: {d.color.value}")

    # Mutator: B.child = null
    del b.refs["child"]

    # Complete marking
    while worklist:
        obj = worklist.popleft()
        for ref in obj.refs.values():
            if ref.color == TriColor.WHITE:
                ref.color = TriColor.GREY
                worklist.append(ref)
        obj.color = TriColor.BLACK

    print(f"  After marking: A={a.color.value}, B={b.color.value}, "
          f"C={c.color.value}, D={d.color.value}")
    print(f"  D is {d.color.value} -- SAVED (correctly reachable)")
    print(f"  C is {c.color.value} -- floating garbage (retained until next cycle)")

    # (d) Yuasa's deletion barrier
    print("\n(d) Yuasa's deletion barrier prevents the bug:")
    a = TCObject("A", TriColor.BLACK, {})
    b = TCObject("B", TriColor.GREY, {})
    c = TCObject("C", TriColor.WHITE, {})
    d = TCObject("D", TriColor.WHITE, {})
    a.refs["child"] = b
    b.refs["child"] = c
    c.refs["child"] = d
    worklist = deque([b])

    # Mutator: A.ref2 = D (no barrier on insertion)
    a.refs["ref2"] = d

    # Mutator with deletion barrier: B.child = null
    print("  Mutator: B.child = null (with Yuasa barrier)")
    old_target = b.refs["child"]  # C
    # Barrier: old target C is WHITE -> grey it
    old_target.color = TriColor.GREY
    worklist.append(old_target)
    print(f"  Barrier greys old target C: {old_target.color.value}")
    del b.refs["child"]

    # Complete marking
    while worklist:
        obj = worklist.popleft()
        for ref in obj.refs.values():
            if ref.color == TriColor.WHITE:
                ref.color = TriColor.GREY
                worklist.append(ref)
        obj.color = TriColor.BLACK

    print(f"  After marking: A={a.color.value}, B={b.color.value}, "
          f"C={c.color.value}, D={d.color.value}")
    print(f"  C is {c.color.value} -- SAVED via snapshot")
    print(f"  D is {d.color.value} -- SAVED (reachable from C)")
    print()


# === Exercise 3: Generational GC Simulation ===

@dataclass(eq=False)
class GenObject:
    obj_id: int
    size: int = 1
    age: int = 0
    refs: List['GenObject'] = field(default_factory=list)
    generation: int = 0  # 0=young, 1=old

    def __hash__(self):
        return id(self)


class GenerationalGCSim:
    """Two-generation GC simulator with statistics."""

    _next_id = 0

    def __init__(self, young_cap=200, old_cap=2000, promo_age=3):
        self.young: List[GenObject] = []
        self.old: List[GenObject] = []
        self.young_cap = young_cap
        self.old_cap = old_cap
        self.promo_age = promo_age
        self.roots: List[GenObject] = []
        self.minor_collections = 0
        self.major_collections = 0
        self.total_freed_minor = 0
        self.total_promoted = 0

    def allocate(self, size=1) -> GenObject:
        used = sum(o.size for o in self.young)
        if used + size > self.young_cap:
            self.minor_gc()

        GenerationalGCSim._next_id += 1
        obj = GenObject(GenerationalGCSim._next_id, size)
        self.young.append(obj)
        return obj

    def minor_gc(self):
        self.minor_collections += 1
        reachable = self._trace(self.roots, young_only=True)

        survivors = []
        promoted = []
        for obj in self.young:
            if obj in reachable:
                obj.age += 1
                if obj.age >= self.promo_age:
                    obj.generation = 1
                    self.old.append(obj)
                    promoted.append(obj)
                else:
                    survivors.append(obj)

        freed = len(self.young) - len(reachable)
        self.total_freed_minor += freed
        self.total_promoted += len(promoted)
        self.young = survivors

    def major_gc(self):
        self.major_collections += 1
        reachable = self._trace(self.roots, young_only=False)
        self.old = [o for o in self.old if o in reachable]
        self.young = [o for o in self.young if o in reachable]

    def _trace(self, roots, young_only=False) -> Set[GenObject]:
        reachable = set()
        stack = list(roots)
        while stack:
            obj = stack.pop()
            if obj not in reachable:
                reachable.add(obj)
                for child in obj.refs:
                    if not young_only or child.generation == 0:
                        stack.append(child)
                    else:
                        reachable.add(child)  # Old gen objects are assumed live
        return reachable

    def stats(self) -> str:
        return (f"Minor GCs: {self.minor_collections}, "
                f"Major GCs: {self.major_collections}, "
                f"Freed(minor): {self.total_freed_minor}, "
                f"Promoted: {self.total_promoted}, "
                f"Young: {len(self.young)}, Old: {len(self.old)}")


def exercise_3():
    """Generational GC simulation with different promotion ages."""
    print("Exercise 3: Generational GC Simulation")
    print("=" * 50)

    # (a) 90% short-lived workload
    print("\n(a) 90% short-lived objects workload:")
    random.seed(42)

    gc = GenerationalGCSim(young_cap=100, old_cap=1000, promo_age=3)
    long_lived = []

    for i in range(500):
        obj = gc.allocate()
        if random.random() < 0.10:  # 10% long-lived
            long_lived.append(obj)
            gc.roots.append(obj)
        # Short-lived objects are not rooted -> will be collected

    print(f"  {gc.stats()}")

    # (b) Different promotion ages
    print("\n(b) Promotion age comparison:")
    for promo_age in [1, 3, 6, 10]:
        GenerationalGCSim._next_id = 0
        random.seed(42)
        gc = GenerationalGCSim(young_cap=100, old_cap=1000,
                               promo_age=promo_age)
        roots = []
        for i in range(500):
            obj = gc.allocate()
            if random.random() < 0.10:
                roots.append(obj)
                gc.roots.append(obj)

        print(f"  promo_age={promo_age:2d}: {gc.stats()}")
    print()


# === Exercise 4: Cheney's Algorithm Step-by-Step ===

@dataclass
class HeapObj:
    name: str
    refs: List['HeapObj'] = field(default_factory=list)
    forwarded: bool = False
    forward_to: Optional['HeapObj'] = None

    def __repr__(self):
        return self.name


def exercise_4():
    """Cheney's algorithm step-by-step trace."""
    print("Exercise 4: Cheney's Algorithm")
    print("=" * 50)

    # Build graph: Root -> A -> B -> C, B -> D, A -> E
    a = HeapObj("A")
    b = HeapObj("B")
    c = HeapObj("C")
    d = HeapObj("D")
    e = HeapObj("E")
    garbage = HeapObj("GARBAGE")  # Unreachable

    a.refs = [b, e]
    b.refs = [c, d]
    root_refs = [a]

    # From-space
    from_space = [a, b, c, d, e, garbage]
    to_space: List[HeapObj] = []
    scan = 0

    print(f"\nInitial from-space: {[o.name for o in from_space]}")
    print(f"Root references: A")
    print()

    def copy_obj(obj: HeapObj) -> HeapObj:
        if obj.forwarded:
            return obj.forward_to
        new_obj = HeapObj(obj.name + "'", list(obj.refs))
        to_space.append(new_obj)
        obj.forwarded = True
        obj.forward_to = new_obj
        return new_obj

    # Step 1: Copy roots
    print("Step 1: Copy root objects")
    new_roots = [copy_obj(r) for r in root_refs]
    print(f"  To-space: {[o.name for o in to_space]}")
    print(f"  scan=0, alloc={len(to_space)}")
    print()

    # BFS scan loop
    step = 2
    while scan < len(to_space):
        obj = to_space[scan]
        print(f"Step {step}: Scan {obj.name} (index {scan})")

        new_refs = []
        for ref in obj.refs:
            # ref might be an original from-space object
            orig = ref
            if hasattr(ref, 'forwarded') and not ref.forwarded:
                new_ref = copy_obj(ref)
            elif hasattr(ref, 'forward_to') and ref.forward_to:
                new_ref = ref.forward_to
            else:
                new_ref = ref
            new_refs.append(new_ref)
            if orig != new_ref or not orig.forwarded:
                print(f"  Copy/forward {orig.name} -> {new_ref.name}")

        obj.refs = new_refs
        scan += 1
        step += 1

        to_names = [o.name for o in to_space]
        print(f"  To-space: {to_names}")
        print(f"  scan={scan}, alloc={len(to_space)}")
        print()

    print(f"Collection complete!")
    print(f"  Live objects: {[o.name for o in to_space]}")
    print(f"  GARBAGE was not copied (correctly collected)")
    print()


# === Exercise 5: Escape Analysis ===

class EscapeState(Enum):
    NO_ESCAPE = 0
    ARG_ESCAPE = 1
    GLOBAL_ESCAPE = 2


class EscapeAnalyzer:
    """Flow-insensitive escape analysis."""

    def __init__(self):
        self.objects: Dict[str, EscapeState] = {}
        self.fields: Dict[str, Set[str]] = defaultdict(set)

    def new_object(self, name: str):
        self.objects[name] = EscapeState.NO_ESCAPE

    def assign_field(self, container: str, contained: str):
        self.fields[container].add(contained)
        self._propagate(container, contained)

    def return_obj(self, name: str):
        self._set_state(name, EscapeState.GLOBAL_ESCAPE)

    def pass_to_callee(self, name: str, callee_stores: bool = False):
        if callee_stores:
            self._set_state(name, EscapeState.GLOBAL_ESCAPE)
        elif self.objects[name] == EscapeState.NO_ESCAPE:
            self._set_state(name, EscapeState.ARG_ESCAPE)

    def _set_state(self, name: str, state: EscapeState):
        if state.value > self.objects.get(name, EscapeState.NO_ESCAPE).value:
            self.objects[name] = state
            for child in self.fields.get(name, set()):
                self._set_state(child, state)

    def _propagate(self, container: str, contained: str):
        cs = self.objects.get(container, EscapeState.NO_ESCAPE)
        if cs.value > self.objects.get(contained, EscapeState.NO_ESCAPE).value:
            self._set_state(contained, cs)

    def can_stack_allocate(self, name: str) -> bool:
        return self.objects.get(name) == EscapeState.NO_ESCAPE

    def report(self):
        for name in sorted(self.objects):
            state = self.objects[name]
            alloc = "STACK" if state == EscapeState.NO_ESCAPE else "HEAP"
            print(f"  {name}: {state.name} -> {alloc}")


def exercise_5():
    """Escape analysis for sample code."""
    print("Exercise 5: Escape Analysis")
    print("=" * 50)

    print("\nAnalyzing:")
    print("  def process():")
    print("      config = Config(timeout=30)     # Object 1")
    print("      result = compute(config)         # compute() only reads config")
    print("      pair = Pair(result, result*2)     # Object 2")
    print("      return pair.first + pair.second   # Returns primitive")
    print()

    ea = EscapeAnalyzer()

    # config = Config(timeout=30)
    ea.new_object("config")

    # result = compute(config) -- compute only reads, does not store
    ea.pass_to_callee("config", callee_stores=False)

    # pair = Pair(result, result*2) -- result is a primitive, not tracked
    ea.new_object("pair")

    # return pair.first + pair.second -- returns primitive, not pair itself
    # pair does NOT escape (only its fields are read)

    print("Escape analysis results:")
    ea.report()

    print()
    print("  config: ARG_ESCAPE (passed to compute, but compute doesn't store it)")
    print("    -> Could be stack-allocated if compute is inlined")
    print("  pair: NO_ESCAPE (only fields read, not returned)")
    print("    -> Can be stack-allocated or scalar-replaced")

    # Now show a version where pair escapes
    print("\n--- If we change to: return pair ---")
    ea2 = EscapeAnalyzer()
    ea2.new_object("config2")
    ea2.pass_to_callee("config2", callee_stores=False)
    ea2.new_object("pair2")
    ea2.return_obj("pair2")

    print("Escape analysis results:")
    ea2.report()
    print()


# === Exercise 6: GC Comparison Benchmark ===

class SimpleRefCountGC:
    """Reference counting GC simulator."""

    def __init__(self):
        self.objects = []
        self.ops = 0

    def allocate(self) -> dict:
        obj = {'rc': 1, 'refs': [], 'alive': True}
        self.objects.append(obj)
        self.ops += 1
        return obj

    def add_ref(self, obj):
        obj['rc'] += 1
        self.ops += 1

    def release(self, obj):
        obj['rc'] -= 1
        self.ops += 1
        if obj['rc'] == 0:
            obj['alive'] = False
            for child in obj['refs']:
                self.release(child)

    def live_count(self):
        return sum(1 for o in self.objects if o['alive'])


class SimpleMarkSweepGC:
    """Mark-sweep GC simulator."""

    def __init__(self):
        self.objects = []
        self.roots: List[dict] = []
        self.ops = 0
        self.gc_count = 0

    def allocate(self) -> dict:
        obj = {'marked': False, 'refs': [], 'alive': True}
        self.objects.append(obj)
        if len(self.objects) % 500 == 0:
            self.collect()
        return obj

    def collect(self):
        self.gc_count += 1
        # Mark
        stack = list(self.roots)
        while stack:
            obj = stack.pop()
            if not obj['marked'] and obj['alive']:
                obj['marked'] = True
                self.ops += 1
                for child in obj['refs']:
                    if not child['marked']:
                        stack.append(child)
        # Sweep
        new_objects = []
        for obj in self.objects:
            if obj['marked']:
                obj['marked'] = False
                new_objects.append(obj)
                self.ops += 1
            else:
                obj['alive'] = False
        self.objects = new_objects

    def live_count(self):
        return sum(1 for o in self.objects if o['alive'])


class SimpleGenGC:
    """Generational copying GC simulator."""

    def __init__(self):
        self.young = []
        self.old = []
        self.roots: List[dict] = []
        self.ops = 0
        self.gc_count = 0

    def allocate(self) -> dict:
        obj = {'age': 0, 'refs': [], 'alive': True}
        self.young.append(obj)
        if len(self.young) > 200:
            self.minor_gc()
        return obj

    def minor_gc(self):
        self.gc_count += 1
        reachable = set()
        stack = [r for r in self.roots if r in self.young]
        for old_obj in self.old:
            for ref in old_obj['refs']:
                if ref in self.young:
                    stack.append(ref)

        while stack:
            obj = stack.pop()
            if id(obj) not in reachable:
                reachable.add(id(obj))
                self.ops += 1
                for child in obj['refs']:
                    if child in self.young:
                        stack.append(child)

        survivors = []
        for obj in self.young:
            if id(obj) in reachable:
                obj['age'] += 1
                if obj['age'] >= 3:
                    self.old.append(obj)
                else:
                    survivors.append(obj)
            else:
                obj['alive'] = False
        self.young = survivors

    def live_count(self):
        return (sum(1 for o in self.young if o['alive']) +
                sum(1 for o in self.old if o['alive']))


def exercise_6():
    """GC comparison benchmark."""
    print("Exercise 6: GC Comparison Benchmark")
    print("=" * 50)
    N = 10000  # Reduced for demo purposes

    # (a) Burst: allocate N objects, discard all
    print(f"\n(a) Burst: allocate {N} objects, discard all")

    # Reference counting
    t0 = time.perf_counter()
    rc = SimpleRefCountGC()
    for _ in range(N):
        obj = rc.allocate()
        rc.release(obj)  # Immediately discard
    t_rc = time.perf_counter() - t0
    print(f"  RefCount:  {t_rc*1000:.1f}ms, ops={rc.ops}, "
          f"live={rc.live_count()}")

    # Mark-sweep
    t0 = time.perf_counter()
    ms = SimpleMarkSweepGC()
    for _ in range(N):
        ms.allocate()
    ms.collect()
    t_ms = time.perf_counter() - t0
    print(f"  MarkSweep: {t_ms*1000:.1f}ms, ops={ms.ops}, "
          f"gc_runs={ms.gc_count}, live={ms.live_count()}")

    # Generational
    t0 = time.perf_counter()
    gen = SimpleGenGC()
    for _ in range(N):
        gen.allocate()
    t_gen = time.perf_counter() - t0
    print(f"  GenGC:     {t_gen*1000:.1f}ms, ops={gen.ops}, "
          f"gc_runs={gen.gc_count}, live={gen.live_count()}")

    # (b) Steady-state: maintain 100 objects, churn 10000
    print(f"\n(b) Steady-state: maintain 100, churn {N}")

    t0 = time.perf_counter()
    rc = SimpleRefCountGC()
    working_set = []
    for i in range(100):
        obj = rc.allocate()
        working_set.append(obj)
    for _ in range(N):
        obj = rc.allocate()
        rc.release(obj)
    t_rc = time.perf_counter() - t0
    print(f"  RefCount:  {t_rc*1000:.1f}ms, ops={rc.ops}, "
          f"live={rc.live_count()}")

    t0 = time.perf_counter()
    ms = SimpleMarkSweepGC()
    working_set = []
    for i in range(100):
        obj = ms.allocate()
        working_set.append(obj)
        ms.roots.append(obj)
    for _ in range(N):
        ms.allocate()
    ms.collect()
    t_ms = time.perf_counter() - t0
    print(f"  MarkSweep: {t_ms*1000:.1f}ms, ops={ms.ops}, "
          f"gc_runs={ms.gc_count}, live={ms.live_count()}")

    t0 = time.perf_counter()
    gen = SimpleGenGC()
    working_set = []
    for i in range(100):
        obj = gen.allocate()
        working_set.append(obj)
        gen.roots.append(obj)
    for _ in range(N):
        gen.allocate()
    t_gen = time.perf_counter() - t0
    print(f"  GenGC:     {t_gen*1000:.1f}ms, ops={gen.ops}, "
          f"gc_runs={gen.gc_count}, live={gen.live_count()}")

    # (c) Cyclic: create 1000 linked-list cycles of length 5
    num_cycles = 1000
    print(f"\n(c) Cyclic: {num_cycles} linked-list cycles of length 5")

    t0 = time.perf_counter()
    rc = SimpleRefCountGC()
    for _ in range(num_cycles):
        nodes = [rc.allocate() for _ in range(5)]
        for i in range(5):
            nodes[i]['refs'].append(nodes[(i + 1) % 5])
            rc.add_ref(nodes[(i + 1) % 5])
        # Release external refs
        for n in nodes:
            rc.release(n)
    t_rc = time.perf_counter() - t0
    print(f"  RefCount:  {t_rc*1000:.1f}ms, "
          f"live={rc.live_count()} (LEAKED -- cycles not collected!)")

    t0 = time.perf_counter()
    ms = SimpleMarkSweepGC()
    for _ in range(num_cycles):
        nodes = [ms.allocate() for _ in range(5)]
        for i in range(5):
            nodes[i]['refs'].append(nodes[(i + 1) % 5])
    ms.collect()
    t_ms = time.perf_counter() - t0
    print(f"  MarkSweep: {t_ms*1000:.1f}ms, "
          f"gc_runs={ms.gc_count}, live={ms.live_count()} (cycles collected)")

    t0 = time.perf_counter()
    gen = SimpleGenGC()
    for _ in range(num_cycles):
        nodes = [gen.allocate() for _ in range(5)]
        for i in range(5):
            nodes[i]['refs'].append(nodes[(i + 1) % 5])
    t_gen = time.perf_counter() - t0
    print(f"  GenGC:     {t_gen*1000:.1f}ms, "
          f"gc_runs={gen.gc_count}, live={gen.live_count()}")
    print()


# === Main ===

def main():
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()


if __name__ == "__main__":
    main()
