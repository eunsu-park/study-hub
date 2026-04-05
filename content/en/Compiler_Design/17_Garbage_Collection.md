# 17. Garbage Collection and Memory Management -- Advanced Topics

**Previous**: [16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md) | **Next**: [18. SSA Form](./18_SSA_Form.md)

---

Lesson 14 introduced the fundamentals of garbage collection: reference counting, mark-sweep, copying, and generational collection. This lesson goes deeper. We examine the engineering details that separate textbook algorithms from production-quality collectors -- cycle detection strategies, tri-color invariants and their proofs, write barrier designs, concurrent collector architectures (G1, ZGC, Shenandoah), escape analysis for stack allocation, and a detailed comparison of GC strategies across the JVM, Go, Python, and Rust.

Understanding these advanced topics is essential for anyone building language runtimes, tuning GC-heavy applications, or reasoning about latency-sensitive systems where GC pauses matter.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [14. Garbage Collection](./14_Garbage_Collection.md), [10. Runtime Environments](./10_Runtime_Environments.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design reference counting systems with cycle detection and weak references
2. Implement tri-color marking with incremental and concurrent invariants
3. Build generational collectors with write barriers and promotion policies
4. Explain Cheney's algorithm and semi-space copying in detail
5. Describe the architectures of G1, ZGC, and Shenandoah concurrent collectors
6. Apply escape analysis to enable stack allocation and scalar replacement
7. Compare GC strategies across JVM, Go, Python, and Rust runtimes

---

## Table of Contents

1. [Reference Counting in Depth](#1-reference-counting-in-depth)
2. [Mark-and-Sweep with Tri-Color Marking](#2-mark-and-sweep-with-tri-color-marking)
3. [Generational Garbage Collection](#3-generational-garbage-collection)
4. [Copying Collectors](#4-copying-collectors)
5. [Concurrent Garbage Collectors](#5-concurrent-garbage-collectors)
6. [Escape Analysis and Stack Allocation](#6-escape-analysis-and-stack-allocation)
7. [GC in Practice: Runtime Comparison](#7-gc-in-practice-runtime-comparison)
8. [Summary](#8-summary)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. Reference Counting in Depth

### 1.1 The Basic Mechanism Revisited

Reference counting assigns each heap object a counter tracking how many references point to it. When the counter reaches zero, the object is immediately reclaimed. The appeal is simplicity and deterministic destruction, but the engineering details are subtle.

```
Object header:
┌──────────┬──────────┬──────────────────┐
│ ref_count│  type_id │    payload ...    │
│  (int)   │  (int)   │                  │
└──────────┴──────────┴──────────────────┘
```

Every pointer assignment requires updating two counters:

```python
# Pseudocode for pointer write: p = q
def write_pointer(target, field_name, new_value):
    old_value = getattr(target, field_name)
    if old_value is not None:
        old_value.ref_count -= 1
        if old_value.ref_count == 0:
            release(old_value)
    if new_value is not None:
        new_value.ref_count += 1
    setattr(target, field_name, new_value)

def release(obj):
    """Recursively decrement children, then free."""
    for child in obj.get_references():
        child.ref_count -= 1
        if child.ref_count == 0:
            release(child)
    free(obj)
```

### 1.2 The Cycle Problem

The fundamental weakness of pure reference counting is that **cyclic** structures are never collected. If A references B and B references A, both have ref_count >= 1 even when no external reference exists.

```
Root set: {R}
R -> A -> B -> A   (cycle)

After R = null:
  A.ref_count = 1  (from B)
  B.ref_count = 1  (from A)
  Neither reaches zero => memory leak
```

### 1.3 Trial Deletion (Cycle Detection)

The standard approach to cycle detection in reference counting systems is **trial deletion**, introduced by Lins (1992) and refined by Bacon and Rajan (2001). The key insight: when a reference count is decremented but does not reach zero, the object *might* be part of a garbage cycle.

The algorithm uses a color scheme:

| Color  | Meaning                                       |
|--------|-----------------------------------------------|
| Black  | In use, not a candidate for cycle collection  |
| Purple | Possible root of a garbage cycle              |
| Grey   | Being traced (trial deletion in progress)     |
| White  | Confirmed garbage                             |

```python
from enum import Enum
from collections import deque

class Color(Enum):
    BLACK = "black"
    PURPLE = "purple"
    GREY = "grey"
    WHITE = "white"

class RCObject:
    def __init__(self, name):
        self.name = name
        self.ref_count = 0
        self.color = Color.BLACK
        self.buffered = False
        self.children = []

    def __repr__(self):
        return f"{self.name}(rc={self.ref_count}, {self.color.value})"


class CycleDetector:
    """
    Bacon-Rajan concurrent cycle collector (synchronous version).
    """

    def __init__(self):
        self.roots = []  # Purple candidates

    def increment(self, obj):
        obj.ref_count += 1
        obj.color = Color.BLACK

    def decrement(self, obj):
        obj.ref_count -= 1
        if obj.ref_count == 0:
            self._release(obj)
        else:
            self._possible_root(obj)

    def _possible_root(self, obj):
        if obj.color != Color.PURPLE:
            obj.color = Color.PURPLE
            if not obj.buffered:
                obj.buffered = True
                self.roots.append(obj)

    def _release(self, obj):
        for child in obj.children:
            self.decrement(child)
        obj.color = Color.BLACK
        if not obj.buffered:
            print(f"  Freed: {obj.name}")

    def collect_cycles(self):
        """Three-phase cycle collection."""
        print("Phase 1: Mark candidates (trial deletion)")
        for root in self.roots:
            self._mark_grey(root)

        print("Phase 2: Scan -- identify garbage")
        for root in self.roots:
            self._scan(root)

        print("Phase 3: Collect white objects")
        collected = []
        for root in list(self.roots):
            root.buffered = False
            if root.color == Color.WHITE:
                collected.append(root)
                self._collect_white(root, collected)
        self.roots.clear()

        for obj in collected:
            print(f"  Cycle-collected: {obj.name}")
        return collected

    def _mark_grey(self, obj):
        if obj.color != Color.GREY:
            obj.color = Color.GREY
            for child in obj.children:
                child.ref_count -= 1  # Trial deletion
                self._mark_grey(child)

    def _scan(self, obj):
        if obj.color == Color.GREY:
            if obj.ref_count > 0:
                self._scan_black(obj)  # Externally referenced
            else:
                obj.color = Color.WHITE  # Garbage
                for child in obj.children:
                    self._scan(child)

    def _scan_black(self, obj):
        """Restore ref counts for reachable objects."""
        obj.color = Color.BLACK
        for child in obj.children:
            child.ref_count += 1
            if child.color != Color.BLACK:
                self._scan_black(child)

    def _collect_white(self, obj, collected):
        if obj.color == Color.WHITE:
            obj.color = Color.BLACK
            for child in obj.children:
                if child not in collected:
                    collected.append(child)
                self._collect_white(child, collected)
```

### 1.4 Weak References

Weak references solve a complementary problem: they allow observation of an object without preventing its collection. A weak reference does **not** contribute to the reference count.

```python
class WeakRef:
    """
    A weak reference that does not prevent collection.
    The runtime nullifies weak refs when their target is collected.
    """

    _all_weak_refs = []  # Global registry for nullification

    def __init__(self, target):
        self._target = target
        self._alive = True
        WeakRef._all_weak_refs.append(self)

    def get(self):
        if self._alive:
            return self._target
        return None

    @classmethod
    def nullify_refs_to(cls, obj):
        """Called by GC when obj is being freed."""
        for wr in cls._all_weak_refs:
            if wr._target is obj:
                wr._alive = False
                wr._target = None
```

Use cases for weak references:

- **Caches**: Cache entries should not prevent GC of cached objects
- **Observer patterns**: Observers should not keep subjects alive
- **Interning tables**: String/symbol intern tables use weak refs to allow de-duplication without leaks
- **Parent pointers**: In tree structures, child-to-parent pointers can be weak to avoid cycles

### 1.5 Deferred Reference Counting

A major overhead of reference counting is the cost of updating counts on every pointer write, especially for stack variables that are frequently assigned. **Deferred reference counting** (Deutsch & Bobrow, 1976) skips counting for stack references entirely:

```
Strategy:
  - Only count heap-to-heap references
  - Maintain a Zero Count Table (ZCT) of objects with heap-rc == 0
  - At collection time, scan the stack to find which ZCT entries are still reachable
  - Free ZCT entries not found on the stack

Trade-off:
  + Much less overhead on stack pointer operations
  - Collection is no longer fully incremental (requires stack scan)
  - Loss of deterministic deallocation for stack-referenced objects
```

---

## 2. Mark-and-Sweep with Tri-Color Marking

### 2.1 Tri-Color Abstraction

The **tri-color abstraction** (Dijkstra et al., 1978) provides a unified framework for understanding all tracing collectors. Every object is assigned one of three colors:

| Color | Meaning |
|-------|---------|
| White | Not yet visited; potentially garbage |
| Grey  | Visited but children not yet scanned |
| Black | Visited and all children scanned |

The invariant that ensures correctness:

> **Tri-color invariant**: No black object may point directly to a white object.

If this invariant holds when no grey objects remain, then every white object is unreachable garbage.

```
Initial state:           After marking:
┌─────┐                  ┌─────┐
│White│ ← all objects     │Black│ ← reachable
└─────┘                  └─────┘
                         ┌─────┐
                         │White│ ← garbage
                         └─────┘
```

### 2.2 Basic Mark-and-Sweep with Explicit Worklist

```python
from enum import Enum
from collections import deque

class TriColor(Enum):
    WHITE = 0
    GREY = 1
    BLACK = 2

class GCObject:
    _all_objects = []

    def __init__(self, name):
        self.name = name
        self.color = TriColor.WHITE
        self.references = []
        GCObject._all_objects.append(self)

    def __repr__(self):
        return f"{self.name}({self.color.name})"


def mark_and_sweep(roots):
    """
    Mark-and-sweep using explicit tri-color worklist.
    """
    # Phase 1: Initialize -- all objects are white
    for obj in GCObject._all_objects:
        obj.color = TriColor.WHITE

    # Phase 2: Mark -- BFS with grey worklist
    worklist = deque()
    for root in roots:
        root.color = TriColor.GREY
        worklist.append(root)

    while worklist:
        obj = worklist.popleft()
        for child in obj.references:
            if child.color == TriColor.WHITE:
                child.color = TriColor.GREY
                worklist.append(child)
        obj.color = TriColor.BLACK

    # Phase 3: Sweep -- free all white objects
    garbage = [o for o in GCObject._all_objects if o.color == TriColor.WHITE]
    for obj in garbage:
        GCObject._all_objects.remove(obj)
        print(f"  Swept: {obj.name}")

    return garbage
```

### 2.3 Incremental Mark-and-Sweep

Stop-the-world pauses are unacceptable in interactive or real-time systems. **Incremental GC** interleaves marking work with mutator execution. The challenge: the mutator can modify the object graph while the collector is marking.

The problem scenario (the **lost object problem**):

```
1. Collector has colored A black, B grey
2. Mutator executes: A.child = C  (C is white)
3. Mutator executes: B.child = null (was B -> C)
4. Collector scans B, finds no reference to C
5. C is never marked => incorrectly freed!

Timeline:
  Collector marks A black
  Mutator: A.ref = C (white)     ← black -> white violation!
  Mutator: B.ref = null          ← grey no longer reaches C
  Collector scans B              ← C is lost
```

Two solutions enforce the tri-color invariant:

### 2.4 Write Barriers

**Dijkstra's Insertion Barrier** (strong tri-color invariant): when a pointer is stored, if the target is white, mark it grey.

```python
def dijkstra_write_barrier(source, field, new_target):
    """Grey the new target to prevent black -> white."""
    if new_target is not None and new_target.color == TriColor.WHITE:
        new_target.color = TriColor.GREY
        worklist.append(new_target)
    setattr(source, field, new_target)
```

> Maintains: **No black object points to a white object** (strong invariant).

**Yuasa's Deletion Barrier** (weak tri-color invariant): when a pointer is overwritten, if the old target is white, mark it grey (snapshot-at-the-beginning).

```python
def yuasa_write_barrier(source, field, new_target):
    """Grey the old target to preserve snapshot."""
    old_target = getattr(source, field)
    if old_target is not None and old_target.color == TriColor.WHITE:
        old_target.color = TriColor.GREY
        worklist.append(old_target)
    setattr(source, field, new_target)
```

> Maintains: **Every white object reachable from a grey object (at the start of marking) remains reachable** (weak invariant). May retain some floating garbage until the next cycle.

### 2.5 Comparison of Barrier Approaches

| Property | Dijkstra (Insertion) | Yuasa (Deletion) |
|----------|---------------------|------------------|
| Invariant | Strong tri-color | Weak tri-color |
| What is greyed | New pointer target | Old pointer target |
| Floating garbage | Minimal | More (snapshot retains start-of-cycle graph) |
| Cost | On every pointer store | On every pointer overwrite |
| Used by | Go runtime | Java (CMS), Haskell |

### 2.6 Incremental Update Scheduling

An incremental collector must decide **how much work** to do per allocation or per time slice:

```
Allocation-driven pacing:
  - Do K marking steps per allocation
  - K chosen so marking completes before heap is full
  - Formula: K = (live_bytes * mark_cost) / (heap_size - live_bytes)

Time-driven pacing:
  - Dedicate a fixed time budget (e.g., 1ms) per mutator time slice
  - More responsive but harder to guarantee completion
```

---

## 3. Generational Garbage Collection

### 3.1 The Generational Hypothesis

The **generational hypothesis** states that most objects die young. Empirical measurement across diverse programs confirms this consistently:

```
Object survival rate by age:

100% |*
     | *
     |  *
     |   **
     |     ***
     |        *****
     |             **********
     |                       ***********************
  0% +──────────────────────────────────────────────
     0   1   2   3   4   5   6   7   8   9  10  ...
                    Age (# of GC cycles survived)
```

This suggests concentrating collection effort on young objects, which are most likely to be garbage.

### 3.2 Two-Generation Architecture

```
┌─────────────────────────────────────────────────┐
│                    Young Generation              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Eden    │  │ Survivor │  │ Survivor │      │
│  │  (alloc) │  │   From   │  │    To    │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│                                                  │
│  Minor GC: collect Eden + From, copy live to To  │
│  Promotion: after N survivals, move to Old Gen   │
├─────────────────────────────────────────────────┤
│                    Old Generation                │
│  ┌──────────────────────────────────────────┐   │
│  │  Tenured Space                           │   │
│  │  (mark-sweep or mark-compact)            │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  Major GC: collect entire old generation         │
│  (much less frequent)                            │
└─────────────────────────────────────────────────┘
```

### 3.3 Write Barriers for Generational GC

The critical challenge: old generation objects may reference young generation objects. Without tracking these **inter-generational pointers**, a minor GC would need to scan the entire old generation to find roots into the young generation.

**Card Table**: Divide old generation into fixed-size cards (e.g., 512 bytes). A card table has one byte per card -- set to "dirty" when a store modifies a pointer in that card.

```python
class CardTable:
    """
    Card table for tracking inter-generational pointers.
    """
    CARD_SIZE = 512  # bytes per card

    def __init__(self, heap_start, heap_size):
        self.heap_start = heap_start
        self.num_cards = (heap_size + self.CARD_SIZE - 1) // self.CARD_SIZE
        self.table = bytearray(self.num_cards)  # 0 = clean, 1 = dirty

    def card_index(self, addr):
        return (addr - self.heap_start) // self.CARD_SIZE

    def mark_dirty(self, addr):
        """Called by write barrier on every pointer store in old gen."""
        idx = self.card_index(addr)
        self.table[idx] = 1

    def dirty_cards(self):
        """Return addresses of dirty card regions for scanning."""
        result = []
        for i, dirty in enumerate(self.table):
            if dirty:
                start = self.heap_start + i * self.CARD_SIZE
                result.append((start, start + self.CARD_SIZE))
        return result

    def clear(self):
        for i in range(self.num_cards):
            self.table[i] = 0
```

**Remembered Sets**: An alternative to card tables. Each region maintains a set of references pointing into it from other regions. More precise but more expensive to maintain.

```
Card Table vs Remembered Set:

Card Table:
  + Fixed overhead: 1 byte per 512 bytes of heap (0.2%)
  + Simple write barrier: table[addr >> 9] = 1
  - Imprecise: must scan entire dirty card to find pointers
  - False positives from non-pointer writes

Remembered Set:
  + Precise: knows exactly which references cross regions
  + No scanning of dirty regions needed
  - Higher write barrier cost (set insertion)
  - Variable memory overhead
```

### 3.4 Promotion Policies

When should objects be promoted from young to old generation?

```python
class PromotionPolicy:
    """Strategies for promoting objects to the old generation."""

    @staticmethod
    def age_threshold(obj, threshold=6):
        """Promote after surviving `threshold` minor GCs."""
        return obj.age >= threshold

    @staticmethod
    def size_threshold(obj, max_young_size=8192):
        """Large objects go directly to old generation."""
        return obj.size > max_young_size

    @staticmethod
    def dynamic_threshold(survivor_occupancy, target=0.5):
        """
        JVM's dynamic tenuring: adjust threshold so survivor space
        stays below target occupancy.
        """
        # If survivor space is too full, lower the threshold
        # to promote objects sooner
        if survivor_occupancy > target:
            return max(1, current_threshold - 1)
        return min(15, current_threshold + 1)
```

### 3.5 Generational GC Simulator

```python
import random

class Object:
    _next_id = 0

    def __init__(self, size=1):
        self.id = Object._next_id
        Object._next_id += 1
        self.size = size
        self.age = 0
        self.references = []
        self.alive = True

    def __repr__(self):
        return f"Obj{self.id}(age={self.age}, size={self.size})"


class GenerationalGC:
    """
    Two-generation garbage collector simulator.
    """

    def __init__(self, young_size=100, old_size=500, promotion_age=3):
        self.young_gen = []
        self.old_gen = []
        self.young_size = young_size
        self.old_size = old_size
        self.promotion_age = promotion_age
        self.roots = []
        self.minor_count = 0
        self.major_count = 0
        self.card_table_dirty = set()  # Indices of old-gen objects with young refs

    def allocate(self, size=1):
        """Allocate in young generation; trigger minor GC if full."""
        used = sum(o.size for o in self.young_gen)
        if used + size > self.young_size:
            self.minor_gc()

        obj = Object(size)
        self.young_gen.append(obj)
        return obj

    def write_barrier(self, source, target):
        """Track old -> young references."""
        source.references.append(target)
        if source in self.old_gen and target in self.young_gen:
            idx = self.old_gen.index(source)
            self.card_table_dirty.add(idx)

    def minor_gc(self):
        """Collect young generation only."""
        self.minor_count += 1

        # Roots into young gen: global roots + dirty card table entries
        young_roots = set()
        for r in self.roots:
            if r in self.young_gen:
                young_roots.add(r)
        for idx in self.card_table_dirty:
            if idx < len(self.old_gen):
                for ref in self.old_gen[idx].references:
                    if ref in self.young_gen:
                        young_roots.add(ref)

        # Trace from young roots
        reachable = set()
        stack = list(young_roots)
        while stack:
            obj = stack.pop()
            if obj not in reachable and obj in self.young_gen:
                reachable.add(obj)
                for child in obj.references:
                    if child in self.young_gen:
                        stack.append(child)

        # Promote or retain survivors
        survivors = []
        promoted = []
        for obj in reachable:
            obj.age += 1
            if obj.age >= self.promotion_age:
                self.old_gen.append(obj)
                promoted.append(obj)
            else:
                survivors.append(obj)

        freed = len(self.young_gen) - len(reachable)
        self.young_gen = survivors
        self.card_table_dirty.clear()

        print(f"  Minor GC #{self.minor_count}: freed={freed}, "
              f"survived={len(survivors)}, promoted={len(promoted)}")

    def major_gc(self):
        """Full heap collection (mark-sweep on old generation)."""
        self.major_count += 1

        # Mark phase: trace from all roots
        reachable = set()
        stack = list(self.roots)
        while stack:
            obj = stack.pop()
            if obj not in reachable:
                reachable.add(obj)
                for child in obj.references:
                    stack.append(child)

        old_size = len(self.old_gen)
        self.old_gen = [o for o in self.old_gen if o in reachable]
        self.young_gen = [o for o in self.young_gen if o in reachable]

        print(f"  Major GC #{self.major_count}: freed={old_size - len(self.old_gen)} "
              f"old objects, {len(self.old_gen)} remain")
```

---

## 4. Copying Collectors

### 4.1 Semi-Space Design

A **copying collector** divides memory into two equal halves (**semi-spaces**). Allocation happens in one half (the **from-space**). When it fills up, live objects are copied to the other half (the **to-space**), and the roles are swapped.

```
Before collection:
┌──────────────────────┬──────────────────────┐
│     From-Space       │     To-Space         │
│ [A][B][C][ ][D][ ]  │  (empty)             │
│  ^live ^dead ^live   │                      │
└──────────────────────┴──────────────────────┘

After collection:
┌──────────────────────┬──────────────────────┐
│  (now To-Space)      │  (now From-Space)    │
│  (empty)             │ [A][C][D]            │
│                      │  ^compacted          │
└──────────────────────┴──────────────────────┘
```

Advantages:
- **No fragmentation**: live objects are compacted by copying
- **Allocation is O(1)**: just bump a pointer
- **Collection cost proportional to live data**, not heap size

Disadvantage:
- **50% memory overhead**: only half the heap is usable at any time

### 4.2 Cheney's Algorithm

Cheney's algorithm (1970) is an elegant BFS copying collector that requires **no auxiliary stack or recursion**. It uses the to-space itself as a queue.

```
To-Space layout during copying:
┌─────────────────────────────────────────────┐
│  [copied objects...]  [grey area]  [free]   │
│  ^                    ^            ^        │
│  start                scan         alloc    │
│                                             │
│  Objects between start..scan are BLACK      │
│  Objects between scan..alloc are GREY       │
│  (their children haven't been processed)    │
└─────────────────────────────────────────────┘
```

```python
class CheneyCollector:
    """
    Cheney's semi-space copying collector.
    Uses to-space as an implicit BFS queue.
    """

    def __init__(self, space_size=20):
        self.space_size = space_size
        # Represent objects as dicts with forwarding pointers
        self.from_space = []
        self.to_space = []
        self.scan = 0       # Next grey object to process
        self.alloc = 0      # Next free slot in to-space
        self.roots = []

    def allocate(self, name, refs=None):
        """Allocate in from-space."""
        obj = {
            'name': name,
            'refs': refs or [],
            'forwarded': False,
            'forward_addr': None,
            'space': 'from',
        }
        self.from_space.append(obj)
        return obj

    def collect(self):
        """Cheney's BFS copying collection."""
        print("Cheney collection starting...")
        self.to_space = []
        self.scan = 0
        self.alloc = 0

        # Copy roots
        new_roots = []
        for root in self.roots:
            new_roots.append(self._copy(root))
        self.roots = new_roots

        # BFS: scan grey objects in to-space
        while self.scan < self.alloc:
            obj = self.to_space[self.scan]
            # Process each reference
            new_refs = []
            for ref in obj['refs']:
                new_refs.append(self._copy(ref))
            obj['refs'] = new_refs
            self.scan += 1

        # Swap spaces
        freed_count = len(self.from_space)
        self.from_space = self.to_space
        for obj in self.from_space:
            obj['space'] = 'from'
            obj['forwarded'] = False
            obj['forward_addr'] = None
        self.to_space = []

        print(f"  Copied {len(self.from_space)} live objects, "
              f"freed {freed_count - len(self.from_space)}")
        return self.from_space

    def _copy(self, obj):
        """Copy object to to-space (or return forwarding pointer)."""
        if obj['forwarded']:
            return obj['forward_addr']

        # Copy to to-space
        new_obj = {
            'name': obj['name'],
            'refs': list(obj['refs']),  # Will be updated during scan
            'forwarded': False,
            'forward_addr': None,
            'space': 'to',
        }
        self.to_space.append(new_obj)
        self.alloc += 1

        # Leave forwarding pointer
        obj['forwarded'] = True
        obj['forward_addr'] = new_obj

        return new_obj

    def dump(self):
        """Display current heap state."""
        print(f"  Heap ({len(self.from_space)} objects):")
        for obj in self.from_space:
            refs = [r['name'] for r in obj['refs']]
            print(f"    {obj['name']} -> {refs}")
```

### 4.3 Forwarding Pointers

When an object is copied, a **forwarding pointer** is left in the old location. If another reference to the same object is encountered, the forwarding pointer redirects to the new copy. This ensures each object is copied exactly once.

```
From-space after copying object A:
┌──────────────────────┐
│  [FWD -> to:A]       │  ← forwarding pointer replaces A
│  [B]                 │  ← B not yet copied
│  [C]                 │
└──────────────────────┘

To-space:
┌──────────────────────┐
│  [A'] (copy of A)    │
└──────────────────────┘
```

### 4.4 Copying Collector Analysis

```
Time complexity:
  - Mark phase: O(live) -- only visit and copy live objects
  - Sweep phase: none! Dead objects are simply abandoned
  - Total: O(live), not O(heap)

Space complexity:
  - Requires 2× address space (semi-spaces)
  - Active working set uses at most 50% of available memory

Cache behavior:
  - Excellent spatial locality after compaction
  - BFS order tends to keep parent/child objects close
  - Allocation is pointer-bump: cache-friendly sequential writes

Compared to mark-sweep:
  + No fragmentation
  + O(live) instead of O(heap) sweep
  + Bump-pointer allocation (fast)
  - 50% memory overhead
  - Copying cost per live object (memcpy)
```

---

## 5. Concurrent Garbage Collectors

### 5.1 Why Concurrent GC?

As heap sizes grow into tens of gigabytes, stop-the-world pauses become unacceptable. A 10 GB heap with mark-sweep might pause for 100ms or more. **Concurrent collectors** perform most GC work while the application (mutator) continues to run.

```
Stop-the-world:
  Mutator: ████████████░░░░░░░░░░░░████████████████
  GC:                  ████████████
                       ^-- pause --^

Concurrent:
  Mutator: ███████████████████████████████████████
  GC:           ░░░░░░░░░░░░░░░░░░░░░░
               ^-- concurrent work, brief STW pauses only
```

### 5.2 G1 (Garbage-First) Collector

G1, introduced in JDK 7 and default since JDK 9, divides the heap into equal-sized **regions** (typically 1-32 MB) rather than contiguous generations.

```
Heap layout:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  E  │  S  │  O  │  O  │  E  │  H  │  O  │  E  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  E = Eden    S = Survivor    O = Old    H = Humongous

Key ideas:
  1. Region-based: any region can serve any role
  2. Garbage-First: prioritize collecting regions with most garbage
  3. Concurrent marking with SATB (snapshot-at-the-beginning) barrier
  4. Mixed collections: collect young + selected old regions together
  5. Pause-time target: user specifies desired max pause (e.g., 200ms)
```

G1 collection phases:

```
1. Young GC (STW):
   - Evacuate live objects from Eden/Survivor regions
   - Uses parallel worker threads
   - Typically < 10ms

2. Concurrent Marking:
   a. Initial Mark (STW, piggybacked on Young GC)
   b. Concurrent Mark (concurrent with mutator)
   c. Remark (STW, finalize marking)
   d. Cleanup (STW, identify empty regions)

3. Mixed GC (STW):
   - Collect young regions + selected old regions
   - Regions with most garbage collected first ("garbage-first")
   - Controlled by pause-time target
```

### 5.3 ZGC (Z Garbage Collector)

ZGC (JDK 11+) is designed for **sub-millisecond pause times** regardless of heap size. It can handle heaps from 8 MB to 16 TB.

```
Key innovations:

1. Colored pointers: GC metadata stored IN the pointer itself
   ┌────────┬───┬───┬───┬───┬────────────────────────────┐
   │ unused │ F │ R │ M1│ M0│      object address (42 bits)│
   │(16 bit)│(1)│(1)│(1)│(1)│        = 4 TB address space  │
   └────────┴───┴───┴───┴───┴────────────────────────────┘
   F  = Finalizable    R  = Remapped
   M1 = Marked1        M0 = Marked0

2. Load barrier (not store barrier):
   - Checked on every pointer LOAD
   - If pointer has wrong color, fix it up (remap/mark)
   - Self-healing: once fixed, subsequent loads are free

3. Concurrent relocation:
   - Objects can be relocated while mutator runs
   - Load barrier handles references to relocated objects
   - No STW compaction phase
```

ZGC phases:

```
Phase 1: Pause Mark Start (STW, ~1ms)
  - Scan thread stacks for root references
  - Mark root references

Phase 2: Concurrent Mark/Remap
  - Trace object graph concurrently
  - Remap references from previous relocation
  - Load barriers handle concurrent mutations

Phase 3: Pause Mark End (STW, ~1ms)
  - Handle remaining SATB references
  - Process weak references

Phase 4: Concurrent Prepare for Relocate
  - Select relocation set (fragmented regions)
  - Build forwarding tables

Phase 5: Pause Relocate Start (STW, ~1ms)
  - Relocate root-referenced objects

Phase 6: Concurrent Relocate
  - Relocate remaining objects
  - Load barriers remap stale references on-the-fly
```

### 5.4 Shenandoah

Shenandoah (developed by Red Hat, in OpenJDK) shares ZGC's goal of low pause times but uses a different mechanism: **Brooks forwarding pointers**.

```
Every object has an indirection pointer:
┌───────────┬──────────────────────────┐
│ fwd_ptr   │     object data          │
│  (self)   │                          │
└───────────┴──────────────────────────┘
     │
     └── Points to self normally.
         During relocation, points to new copy.

Access pattern:
  obj.field  =>  obj.fwd_ptr.field  (always indirect)

Cost: one extra indirection on every field access
Benefit: concurrent compaction without colored pointers
```

Shenandoah phases:

```
1. Init Mark (STW, brief)
2. Concurrent Mark
3. Final Mark (STW, brief)
4. Concurrent Cleanup
5. Concurrent Evacuation  ← key: copies objects while mutator runs
6. Init Update Refs (STW, brief)
7. Concurrent Update Refs ← rewrite old references to new locations
8. Final Update Refs (STW, brief)
9. Concurrent Cleanup
```

### 5.5 Concurrent Collector Comparison

| Feature | G1 | ZGC | Shenandoah |
|---------|-----|------|------------|
| Pause target | 200ms (configurable) | < 1ms | < 10ms |
| Heap size | Up to ~64 GB practical | 8 MB - 16 TB | Up to ~several TB |
| Barrier type | SATB write barrier | Load barrier (colored ptr) | Load barrier (Brooks ptr) |
| Compaction | STW evacuation | Concurrent relocation | Concurrent evacuation |
| Overhead | Low-moderate | Moderate (colored pointers) | Moderate (forwarding ptrs) |
| JDK version | 7+ (default 9+) | 11+ (production 15+) | 12+ |
| Throughput | Highest of the three | Slightly lower | Slightly lower |
| Latency | Moderate | Lowest | Low |

---

## 6. Escape Analysis and Stack Allocation

### 6.1 What is Escape Analysis?

**Escape analysis** determines whether an object's lifetime is confined to a single method or thread. If an object does not "escape" its creating method, it can be allocated on the stack instead of the heap, eliminating GC overhead entirely.

An object escapes if:
1. It is returned from the method
2. It is assigned to a static field or instance field of an escaping object
3. It is passed to another method that causes it to escape
4. It is thrown as an exception

```python
def escape_analysis_examples():
    """Illustrating escape vs. non-escape cases."""

    # Case 1: Does NOT escape -- can be stack-allocated
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def distance_from_origin():
        p = Point(3, 4)  # p does not escape this method
        return (p.x ** 2 + p.y ** 2) ** 0.5  # Returns a primitive

    # Case 2: ESCAPES via return
    def create_point():
        p = Point(1, 2)
        return p  # p escapes: caller gets a reference

    # Case 3: ESCAPES via field assignment
    results = []
    def collect_point():
        p = Point(5, 6)
        results.append(p)  # p escapes into the list

    # Case 4: Does NOT escape -- argument does not escape callee
    def use_point(p):
        return p.x + p.y  # p is only read, not stored

    def caller():
        p = Point(7, 8)
        return use_point(p)  # p does not escape if use_point is inlined
```

### 6.2 Escape Analysis Algorithm

A simplified flow-insensitive escape analysis using a connection graph:

```python
from enum import Enum
from collections import defaultdict

class EscapeState(Enum):
    NO_ESCAPE = 0       # Confined to creating method
    ARG_ESCAPE = 1      # Passed as argument but not stored globally
    GLOBAL_ESCAPE = 2   # Stored in global/heap, fully escapes

class EscapeAnalyzer:
    """
    Connection-graph-based escape analysis (simplified).
    Tracks how objects flow through a method.
    """

    def __init__(self):
        self.objects = {}       # name -> EscapeState
        self.edges = defaultdict(set)  # containment edges

    def new_object(self, name):
        self.objects[name] = EscapeState.NO_ESCAPE

    def assign_field(self, container, field_obj):
        """container.f = field_obj"""
        self.edges[container].add(field_obj)
        # If container escapes, so does field_obj
        self._propagate(container, field_obj)

    def return_value(self, name):
        """Object is returned from method."""
        self.objects[name] = EscapeState.GLOBAL_ESCAPE
        self._propagate_down(name)

    def pass_to_method(self, name, callee_escapes=False):
        """Object passed as argument to another method."""
        if callee_escapes:
            self.objects[name] = EscapeState.GLOBAL_ESCAPE
        elif self.objects[name] == EscapeState.NO_ESCAPE:
            self.objects[name] = EscapeState.ARG_ESCAPE
        self._propagate_down(name)

    def _propagate(self, container, contained):
        container_state = self.objects.get(container, EscapeState.NO_ESCAPE)
        if container_state.value > self.objects.get(contained, EscapeState.NO_ESCAPE).value:
            self.objects[contained] = container_state
            self._propagate_down(contained)

    def _propagate_down(self, name):
        for child in self.edges.get(name, set()):
            if self.objects[child].value < self.objects[name].value:
                self.objects[child] = self.objects[name]
                self._propagate_down(child)

    def can_stack_allocate(self, name):
        return self.objects.get(name) == EscapeState.NO_ESCAPE

    def report(self):
        for name, state in sorted(self.objects.items()):
            action = "STACK" if state == EscapeState.NO_ESCAPE else "HEAP"
            print(f"  {name}: {state.name} => {action}")
```

### 6.3 Scalar Replacement

When escape analysis proves an object does not escape, the compiler can go further than stack allocation: it can **decompose the object into its individual fields** (scalar replacement), eliminating the object entirely.

```
Before scalar replacement:
  Point p = new Point(3, 4);
  double d = Math.sqrt(p.x * p.x + p.y * p.y);

After escape analysis + scalar replacement:
  int p_x = 3;       // Object eliminated!
  int p_y = 4;       // Fields become local variables
  double d = Math.sqrt(p_x * p_x + p_y * p_y);

Benefits:
  - No allocation at all (not even stack)
  - Fields may be register-allocated
  - Enables further optimizations (constant folding, etc.)
```

### 6.4 Lock Elision

If an object does not escape the creating thread, any synchronization on it is unnecessary:

```
Before:
  synchronized(new Object()) {  // lock on non-escaping object
      counter++;
  }

After escape analysis + lock elision:
  counter++;  // Lock removed: object never shared between threads
```

### 6.5 Escape Analysis in Practice

| Runtime | Escape Analysis | Stack Allocation | Scalar Replacement |
|---------|----------------|------------------|--------------------|
| JVM (HotSpot) | Since JDK 6 | No (scalar replacement instead) | Yes |
| Go | Since 1.0 | Yes (primary optimization) | Limited |
| Graal/GraalVM | Advanced (partial) | Via scalar replacement | Yes |
| V8 (JavaScript) | Limited | No | Allocation folding |

Go's escape analysis is particularly visible:

```go
// Go: escape analysis is reported with -gcflags="-m"
func noEscape() int {
    p := &Point{3, 4}  // "does not escape" -- stack allocated
    return p.X + p.Y
}

func escapes() *Point {
    p := &Point{3, 4}  // "escapes to heap" -- must heap allocate
    return p
}
```

---

## 7. GC in Practice: Runtime Comparison

### 7.1 JVM Garbage Collectors

The JVM offers the most diverse set of production GCs:

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│   Collector  │  Serial  │ Parallel │   G1     │   ZGC    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Algorithm    │ Mark-    │ Mark-    │ Region-  │ Concurrent│
│              │ Compact  │ Compact  │ based    │ Relocating│
│              │ (STW)    │ (STW)    │ Mixed    │           │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Threads      │ Single   │ Multi    │ Multi    │ Multi    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Generations  │ Young+Old│ Young+Old│ Logical  │ Single   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Pause Target │ N/A      │ N/A      │ 200ms    │ < 1ms    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Best for     │ Small    │ Batch/   │ General  │ Latency- │
│              │ heaps    │ throughput│ purpose  │ critical │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Flag         │ -XX:+Use │ -XX:+Use │ -XX:+Use │ -XX:+Use │
│              │ SerialGC │ParallelGC│   G1GC   │    ZGC   │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

### 7.2 Go Garbage Collector

Go uses a **concurrent, tri-color, mark-sweep** collector with no generational component (as of Go 1.22):

```
Design principles:
  1. Low latency over throughput
  2. No compaction (relies on TCMalloc-style allocator)
  3. Write barrier (Dijkstra + Yuasa hybrid)
  4. Concurrent mark with brief STW pauses (~0.5ms)

GOGC parameter (default: 100):
  - Controls GC pacing
  - GOGC=100 means GC triggers when heap doubles
  - GOGC=200 means GC triggers when heap triples
  - GOMEMLIMIT: absolute memory ceiling (Go 1.19+)

Phases:
  1. Sweep Termination (STW, ~10μs): finish previous sweep
  2. Mark Phase (concurrent): trace reachable objects
  3. Mark Termination (STW, ~0.5ms): drain remaining work
  4. Sweep Phase (concurrent): reclaim unmarked objects
```

Why no generations in Go?

```
Go's rationale:
  1. Value types (structs) reduce heap allocations
  2. Escape analysis moves many objects to stack
  3. Goroutine stacks are small (2KB initial) and on heap
  4. Write barriers for generational GC would hurt goroutine performance
  5. Simple concurrent mark-sweep achieves < 1ms pauses without generations
```

### 7.3 Python Garbage Collection

Python (CPython) uses a **hybrid** approach: primary reference counting with a cyclic garbage collector as backup.

```
Layer 1: Reference Counting
  - Every object has ob_refcnt
  - Immediate deallocation when count reaches zero
  - Deterministic finalization (__del__ called immediately)
  - Cannot handle cycles

Layer 2: Cyclic GC (gc module)
  - Generational: 3 generations (gen0, gen1, gen2)
  - Only tracks container objects (list, dict, set, class instances)
  - Triggered by allocation count thresholds
  - Uses trial deletion algorithm

┌──────────┬────────────┬──────────────┐
│  Gen 0   │   Gen 1    │    Gen 2     │
│ (newest) │ (middle)   │  (oldest)    │
│ threshold│ threshold  │  threshold   │
│   = 700  │   = 10     │   = 10       │
│          │ (gen0 runs)│ (gen1 runs)  │
└──────────┴────────────┴──────────────┘

gc.get_threshold()  => (700, 10, 10)
  700: gen0 collected after 700 allocations minus deallocations
  10: gen1 collected after gen0 runs 10 times
  10: gen2 collected after gen1 runs 10 times
```

The GIL (Global Interpreter Lock) simplifies Python's GC but limits concurrency:

```
Reference counting thread safety:
  - GIL makes ob_refcnt updates atomic (no need for atomic ops)
  - But GIL prevents true parallel execution
  - Python 3.13+ "free-threaded" mode removes GIL:
    * Uses atomic reference counting
    * Deferred reference counting for some objects
    * Biased reference counting optimization
```

### 7.4 Rust: Ownership Instead of GC

Rust takes a radically different approach: **compile-time memory management** through the ownership system.

```
Rust's three rules:
  1. Each value has exactly one owner
  2. When the owner goes out of scope, the value is dropped
  3. References must not outlive the referent (lifetimes)

fn example() {
    let s = String::from("hello");  // s owns the String
    let r = &s;                     // r borrows s (immutable)
    println!("{}", r);              // OK: s is still alive
}                                   // s dropped here, memory freed
// No GC needed! Compiler inserts drop() calls at scope exit.
```

Rust does provide optional GC-like types for when ownership is insufficient:

```
┌──────────────┬──────────────────────────────────────────┐
│ Type         │ Purpose                                  │
├──────────────┼──────────────────────────────────────────┤
│ Box<T>       │ Heap allocation with single owner        │
│ Rc<T>        │ Reference counting (single-thread)       │
│ Arc<T>       │ Atomic reference counting (multi-thread) │
│ Weak<T>      │ Weak reference (for Rc/Arc cycles)       │
│ RefCell<T>   │ Interior mutability (runtime borrow check)│
└──────────────┴──────────────────────────────────────────┘

// Breaking cycles with Weak:
use std::rc::{Rc, Weak};
struct Node {
    parent: Weak<Node>,    // Weak: does not prevent collection
    children: Vec<Rc<Node>>, // Rc: shared ownership
}
```

### 7.5 Cross-Runtime Comparison

```
┌──────────────┬──────────────┬─────────┬──────────┬──────────┐
│              │     JVM      │   Go    │  Python  │   Rust   │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Primary      │ Tracing (G1, │ Tri-color│ Ref count│ Ownership│
│ strategy     │ ZGC, etc.)   │ M&S     │ + cyclic │ (compile)│
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Generational │ Yes          │ No      │ Yes (3)  │ N/A      │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Concurrent   │ Yes (G1/ZGC) │ Yes     │ No (GIL) │ N/A      │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Compaction   │ Yes          │ No      │ No       │ N/A      │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Pause times  │ < 1ms (ZGC)  │ < 1ms   │ ~10ms    │ 0 (no GC)│
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Throughput   │ Excellent    │ Good    │ Moderate │ Excellent│
│ overhead     │ (2-5%)       │ (~5%)   │ (~10-15%)│ (~0%)    │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Deterministic│ No           │ No      │ Partial  │ Yes      │
│ destruction  │ (finalizers) │         │ (refcount│ (Drop)   │
│              │              │         │  only)   │          │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ Escape       │ Yes (scalar  │ Yes     │ No       │ N/A      │
│ analysis     │ replacement) │ (stack) │          │ (all     │
│              │              │         │          │ explicit)│
└──────────────┴──────────────┴─────────┴──────────┴──────────┘
```

### 7.6 Choosing a GC Strategy

Decision framework for language implementers:

```
                    Need deterministic destruction?
                    ├── Yes: Reference counting (Swift, Python, Rust-Rc)
                    │        or ownership (Rust)
                    └── No:  Tracing collector
                             ├── Latency-critical?
                             │   ├── Yes: Concurrent (ZGC, Shenandoah, Go)
                             │   └── No:  Generational mark-sweep (G1, .NET)
                             └── Memory-constrained?
                                 ├── Yes: Copying (semi-space) or mark-compact
                                 └── No:  Mark-sweep with free lists
```

---

## 8. Summary

Key takeaways from this lesson:

1. **Reference counting** provides deterministic destruction but requires cycle detection (trial deletion) and benefits from weak references and deferred counting to reduce overhead.

2. **Tri-color marking** unifies all tracing collectors. The invariant (no black-to-white edges) can be maintained by insertion barriers (Dijkstra) or deletion barriers (Yuasa), each with different precision/cost trade-offs.

3. **Generational GC** exploits the generational hypothesis. Write barriers (card tables or remembered sets) track inter-generational pointers. Promotion policies control when objects graduate to older generations.

4. **Copying collectors** (Cheney's algorithm) eliminate fragmentation and achieve O(live) collection time at the cost of 50% memory overhead. The to-space doubles as a BFS queue.

5. **Concurrent collectors** (G1, ZGC, Shenandoah) achieve sub-millisecond pauses on multi-gigabyte heaps through techniques like colored pointers, load barriers, and region-based collection.

6. **Escape analysis** enables the compiler to stack-allocate or scalar-replace objects that do not escape their creating method, eliminating GC overhead entirely for short-lived objects.

7. **Runtime comparison**: JVM offers the most sophisticated GC options; Go prioritizes simplicity and low latency; Python relies on reference counting with backup cycle collection; Rust avoids GC entirely through compile-time ownership.

---

## 9. Exercises

### Exercise 1: Cycle Detection

Implement the Bacon-Rajan cycle collector and demonstrate it on the following graph:

```
A -> B -> C -> A    (cycle)
D -> E              (not a cycle)
Root -> D

Expected: A, B, C are cycle-collected; D, E survive
```

### Exercise 2: Tri-Color Marking Safety

Given the object graph and GC state:

```
Objects: A(BLACK), B(GREY), C(WHITE), D(WHITE)
References: A->{B}, B->{C}, C->{D}
Roots: {A}
```

(a) Show that marking completes correctly without mutator interference.
(b) The mutator executes `A.ref2 = D; B.child = null`. Show the lost object problem.
(c) Show how Dijkstra's insertion barrier prevents the bug.
(d) Show how Yuasa's deletion barrier prevents the bug.

### Exercise 3: Generational GC Simulation

Using the `GenerationalGC` class from Section 3.5:
(a) Create a workload where 90% of objects die young. Measure minor vs. major GC frequency.
(b) Experiment with different promotion ages (1, 3, 6, 10). How does this affect old generation growth?
(c) Add a third generation (middle) and compare collection behavior.

### Exercise 4: Cheney's Algorithm

Implement Cheney's algorithm for the following object graph:

```
Root -> A -> B -> C
             B -> D
        A -> E
```

Trace the algorithm step by step: show from-space, to-space, scan pointer, and alloc pointer at each step.

### Exercise 5: Escape Analysis

Write an escape analyzer for the following code and determine which objects can be stack-allocated:

```python
def process():
    config = Config(timeout=30)     # Object 1
    result = compute(config)        # compute() only reads config
    pair = Pair(result, result*2)   # Object 2
    return pair.first + pair.second # Returns primitive
```

### Exercise 6: GC Comparison Benchmark

Write a benchmark that allocates objects in three patterns and compares GC strategies:
(a) **Burst**: Allocate 100,000 small objects, discard all.
(b) **Steady-state**: Maintain a working set of 1,000 objects while allocating/discarding 100,000 temporaries.
(c) **Cyclic**: Create 10,000 linked-list cycles of length 5.

Compare reference counting, mark-sweep, and generational copying collectors.

---

## 10. References

1. Bacon, D. F., & Rajan, V. T. (2001). "Concurrent Cycle Collection in Reference Counted Systems." *ECOOP*.
2. Cheney, C. J. (1970). "A Nonrecursive List Compacting Algorithm." *Communications of the ACM*, 13(11).
3. Dijkstra, E. W., et al. (1978). "On-the-fly Garbage Collection: An Exercise in Cooperation." *Communications of the ACM*, 21(11).
4. Detlefs, D., Flood, C., Heller, S., & Printezis, T. (2004). "Garbage-First Garbage Collection." *ISMM*.
5. Yang, A. Y., et al. (2022). "The Design and Implementation of the Z Garbage Collector." *PLDI*.
6. Flood, C. H., Kennke, R., Dinn, A., Haley, A., & Westrelin, R. (2016). "Shenandoah: An open-source concurrent compacting garbage collector for OpenJDK." *PPPJ*.
7. Choi, J.-D., Gupta, M., Serrano, M. J., Sreedhar, V. C., & Midkiff, S. P. (1999). "Escape Analysis for Java." *OOPSLA*.
8. Jones, R., Hosking, A., & Moss, E. (2012). *The Garbage Collection Handbook*. CRC Press.
9. Lins, R. D. (1992). "Cyclic Reference Counting with Lazy Mark-Scan." *Information Processing Letters*, 44(4).

---

[Previous: 16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md) | [Next: 18. SSA Form](./18_SSA_Form.md) | [Overview](./00_Overview.md)
