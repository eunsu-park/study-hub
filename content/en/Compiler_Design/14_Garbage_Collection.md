# Garbage Collection

**Previous**: [13. Loop Optimization](./13_Loop_Optimization.md) | **Next**: [15. Interpreters and Virtual Machines](./15_Interpreters_and_Virtual_Machines.md)

---

Memory management is one of the most fundamental challenges in programming language implementation. Every program allocates memory, and that memory must eventually be reclaimed. Manual memory management (as in C and C++) gives the programmer full control but is notoriously error-prone: dangling pointers, double frees, and memory leaks are among the most common and dangerous bugs in software history. Garbage collection (GC) automates this process, freeing the programmer from tracking object lifetimes -- at the cost of runtime overhead and reduced control.

This lesson covers the major garbage collection algorithms and their trade-offs, from the simplest reference counting scheme to sophisticated generational and concurrent collectors used in production systems.

**Difficulty**: ⭐⭐⭐

**Prerequisites**: [10. Runtime Environments](./10_Runtime_Environments.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare manual and automatic memory management approaches
2. Implement reference counting with cycle detection
3. Understand tracing collectors: mark-sweep, mark-compact, and copying
4. Explain generational GC and why it works (the generational hypothesis)
5. Describe incremental and concurrent GC algorithms using tri-color marking
6. Analyze GC strategies in real systems (JVM, Go, Python, Rust)
7. Reason about GC tuning and performance trade-offs

---

## Table of Contents

1. [Memory Management Overview](#1-memory-management-overview)
2. [Reference Counting](#2-reference-counting)
3. [Cycle Detection](#3-cycle-detection)
4. [Mark-Sweep Collection](#4-mark-sweep-collection)
5. [Mark-Compact Collection](#5-mark-compact-collection)
6. [Copying Collectors](#6-copying-collectors)
7. [Generational Collection](#7-generational-collection)
8. [Incremental and Concurrent GC](#8-incremental-and-concurrent-gc)
9. [Tri-Color Marking](#9-tri-color-marking)
10. [GC in Real Systems](#10-gc-in-real-systems)
11. [GC Tuning and Metrics](#11-gc-tuning-and-metrics)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Memory Management Overview

### 1.1 The Heap

Dynamic memory allocation uses the **heap** -- a region of memory managed at runtime. Objects are allocated on the heap when their size or lifetime is not known at compile time.

```
┌─────────────────────────────────────────────┐
│                  Stack                       │
│  (automatic: local vars, return addresses)   │
├─────────────────────────────────────────────┤
│                    ↓                         │
│              (grows down)                    │
│                                             │
│              (grows up)                      │
│                    ↑                         │
├─────────────────────────────────────────────┤
│                   Heap                       │
│  (dynamic: malloc/new, managed by GC)        │
├─────────────────────────────────────────────┤
│              Static/Global Data              │
├─────────────────────────────────────────────┤
│                  Code/Text                   │
└─────────────────────────────────────────────┘
```

### 1.2 Manual Memory Management

In languages like C, the programmer explicitly manages memory:

```c
// C manual memory management
int *arr = (int *)malloc(n * sizeof(int));  // Allocate
// ... use arr ...
free(arr);  // Deallocate -- programmer's responsibility
arr = NULL; // Avoid dangling pointer
```

Common problems with manual memory management:

| Problem | Description | Consequence |
|---------|-------------|-------------|
| **Memory leak** | Forgetting to call `free` | Unbounded memory growth |
| **Dangling pointer** | Using memory after `free` | Undefined behavior, crashes |
| **Double free** | Calling `free` twice | Heap corruption |
| **Buffer overflow** | Writing past allocation bounds | Security vulnerabilities |
| **Use-after-free** | Accessing freed memory | Data corruption |

### 1.3 Automatic Memory Management

Garbage collection automatically identifies and reclaims objects that are no longer reachable by the program. The key insight: **an object is garbage if no sequence of pointer dereferences starting from the root set can reach it**.

The **root set** consists of:
- Global variables
- Local variables on the stack
- CPU registers containing pointers
- Any other runtime-managed references

```python
class GCObject:
    """Base class for garbage-collected objects."""

    def __init__(self, name):
        self.name = name
        self.references = []  # Pointers to other objects

    def add_reference(self, other):
        self.references.append(other)

    def __repr__(self):
        refs = [r.name for r in self.references]
        return f"Object({self.name}, refs={refs})"


# Example: reachability
root = GCObject("root")
a = GCObject("A")
b = GCObject("B")
c = GCObject("C")    # Unreachable -- garbage!
d = GCObject("D")

root.add_reference(a)
a.add_reference(b)
b.add_reference(d)

# Reachable from root: root -> A -> B -> D
# Unreachable: C (garbage)
```

### 1.4 Trade-offs: Manual vs Automatic

| Aspect | Manual | Reference Counting | Tracing GC |
|--------|--------|-------------------|------------|
| **Throughput** | Best (no GC overhead) | Good (incremental cost) | Good (amortized) |
| **Latency** | Predictable | Predictable | Pauses (unless concurrent) |
| **Memory overhead** | None | Per-object count | Metadata + copy space |
| **Safety** | Unsafe | Safe (except cycles) | Safe |
| **Programmer effort** | High | Low | Low |
| **Fragmentation** | Depends on allocator | Yes | Compact/copy fixes it |

### 1.5 Conservative vs Precise GC

A **precise** (or **exact**) GC knows exactly which values on the stack and in objects are pointers. This requires cooperation from the compiler (stack maps, type information).

A **conservative** GC treats any word on the stack that looks like a valid heap pointer as a potential pointer. This is simpler but may retain garbage (false pointers) and cannot move objects (since it might "fix up" a non-pointer value).

```python
def demonstrate_conservative_scan(stack_words, heap_start, heap_end):
    """
    Simulate conservative stack scanning.

    Any value that looks like a heap address is treated as a pointer.
    """
    potential_roots = []

    for i, word in enumerate(stack_words):
        if heap_start <= word < heap_end:
            potential_roots.append((i, word))
            print(f"  Stack[{i}] = 0x{word:08x} -- potential pointer (in heap range)")
        else:
            print(f"  Stack[{i}] = 0x{word:08x} -- not a pointer")

    return potential_roots


# Example
print("=== Conservative Stack Scan ===")
stack = [
    0x00000042,   # Integer 66 -- not a pointer
    0x08001000,   # In heap range -- potential pointer!
    0x00000000,   # NULL -- not in heap
    0x08003000,   # In heap range -- potential pointer!
    0x0800CAFE,   # In heap range but actually an integer -- false positive!
]

roots = demonstrate_conservative_scan(stack, 0x08000000, 0x08100000)
print(f"\nPotential roots found: {len(roots)}")
print("Note: 0x0800CAFE might be a false positive (an integer that happens")
print("to look like a heap address). Conservative GC cannot tell the difference.")
```

---

## 2. Reference Counting

### 2.1 Basic Reference Counting

The simplest form of automatic memory management: each object maintains a count of how many references point to it. When the count drops to zero, the object is immediately freed.

```python
class RefCountedObject:
    """Object with reference counting."""

    _all_objects = []  # Track all allocated objects for debugging

    def __init__(self, name, size=1):
        self.name = name
        self.size = size
        self.ref_count = 0
        self.references = {}  # field_name -> RefCountedObject
        RefCountedObject._all_objects.append(self)

    def inc_ref(self):
        self.ref_count += 1

    def dec_ref(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            self._free()

    def _free(self):
        """Free this object and decrement refs to children."""
        print(f"  Freeing {self.name} (size={self.size})")
        # Decrement reference counts of all referenced objects
        for field, obj in list(self.references.items()):
            obj.dec_ref()
        self.references.clear()
        RefCountedObject._all_objects.remove(self)

    def set_field(self, field_name, new_obj):
        """Set a field to point to new_obj, updating ref counts."""
        # Decrement old reference
        old_obj = self.references.get(field_name)
        if old_obj is not None:
            old_obj.dec_ref()

        # Set new reference
        if new_obj is not None:
            new_obj.inc_ref()
            self.references[field_name] = new_obj
        elif field_name in self.references:
            del self.references[field_name]

    def __repr__(self):
        return f"RC({self.name}, rc={self.ref_count})"


def demonstrate_ref_counting():
    """Demonstrate reference counting in action."""
    RefCountedObject._all_objects = []

    print("=== Reference Counting Demo ===\n")

    # Create root (simulate stack reference)
    root = RefCountedObject("root")
    root.inc_ref()  # Stack reference

    # Allocate objects
    a = RefCountedObject("A")
    b = RefCountedObject("B")
    c = RefCountedObject("C")

    # Build object graph: root -> A -> B, root -> C
    root.set_field("x", a)
    a.set_field("y", b)
    root.set_field("z", c)

    print("Object graph: root -> A -> B, root -> C")
    print(f"Objects: {RefCountedObject._all_objects}")
    print(f"Ref counts: root={root.ref_count}, A={a.ref_count}, "
          f"B={b.ref_count}, C={c.ref_count}")

    # Remove reference root -> A
    print("\nRemoving root -> A...")
    root.set_field("x", None)

    print(f"Objects after removal: {RefCountedObject._all_objects}")
    # A's ref count drops to 0, A is freed, which drops B's ref count to 0, B is freed

    # Remove reference root -> C
    print("\nRemoving root -> C...")
    root.set_field("z", None)
    print(f"Objects after removal: {RefCountedObject._all_objects}")

demonstrate_ref_counting()
```

Output:

```
=== Reference Counting Demo ===

Object graph: root -> A -> B, root -> C
Objects: [RC(root, rc=1), RC(A, rc=1), RC(B, rc=1), RC(C, rc=1)]
Ref counts: root=1, A=1, B=1, C=1

Removing root -> A...
  Freeing A (size=1)
  Freeing B (size=1)
Objects after removal: [RC(root, rc=1), RC(C, rc=1)]

Removing root -> C...
  Freeing C (size=1)
Objects after removal: [RC(root, rc=1)]
```

### 2.2 Advantages and Disadvantages

**Advantages**:
- **Immediate reclamation**: Objects are freed as soon as they become garbage
- **Incremental**: Cost is spread across program execution (no pauses)
- **Locality**: Freed memory was recently used (likely in cache)
- **Simplicity**: Easy to understand and implement

**Disadvantages**:
- **Cycles**: Cannot collect cyclic garbage (see Section 3)
- **Space overhead**: Every object needs a reference count field
- **Time overhead**: Every pointer assignment requires inc/dec operations
- **Thread safety**: Ref count updates need atomic operations in concurrent programs
- **Cache unfriendly**: Ref count updates dirty cache lines of unrelated objects

### 2.3 Deferred Reference Counting

To reduce the overhead of tracking stack references (which change very frequently), **deferred reference counting** only tracks heap-to-heap references. Stack references are handled by periodically scanning the stack.

```python
class DeferredRefCounting:
    """
    Deferred reference counting: only count heap-to-heap references.
    Stack references are ignored in ref counts.
    """

    def __init__(self):
        self.heap_objects = []
        self.zero_count_table = set()  # Objects with rc=0 (potential garbage)
        self.stack_roots = set()       # Current stack roots

    def allocate(self, name, size=1):
        obj = RefCountedObject(name, size)
        self.heap_objects.append(obj)
        # Don't increment for stack reference
        return obj

    def heap_write(self, source, field, target):
        """Record heap-to-heap pointer write."""
        source.set_field(field, target)

    def stack_root_change(self, added=None, removed=None):
        """Track stack root changes (deferred -- no ref count update)."""
        if added:
            self.stack_roots.add(added)
        if removed:
            self.stack_roots.discard(removed)

    def collect(self):
        """
        Periodic collection: scan stack roots to find true garbage.
        Objects with rc=0 that are NOT stack roots are garbage.
        """
        print("\n--- Deferred RC Collection ---")
        garbage = []
        for obj in self.heap_objects:
            if obj.ref_count == 0 and obj not in self.stack_roots:
                garbage.append(obj)
                print(f"  Garbage: {obj.name}")

        for obj in garbage:
            self.heap_objects.remove(obj)
            print(f"  Freed: {obj.name}")

        return len(garbage)
```

### 2.4 Coalesced Reference Counting

**Coalesced reference counting** buffers reference count updates and coalesces them before applying. If a pointer field is written multiple times between collections, only the net effect is recorded.

```python
class CoalescedRefCounting:
    """
    Coalesced reference counting: buffer ref count updates.

    Instead of updating counts on every pointer write, log
    the old and new values. At collection time, compute net changes.
    """

    def __init__(self):
        self.objects = {}      # name -> object
        self.update_log = []   # list of (old_target, new_target)

    def write_pointer(self, source_field, old_target, new_target):
        """Log a pointer update instead of immediately changing ref counts."""
        self.update_log.append((old_target, new_target))

    def process_updates(self):
        """Process buffered updates, computing net ref count changes."""
        # Net changes per object
        changes = {}
        for old_target, new_target in self.update_log:
            if old_target is not None:
                changes[old_target] = changes.get(old_target, 0) - 1
            if new_target is not None:
                changes[new_target] = changes.get(new_target, 0) + 1

        print("Coalesced updates:")
        for obj, delta in changes.items():
            if delta != 0:
                print(f"  {obj}: {'+' if delta > 0 else ''}{delta}")

        self.update_log.clear()
        return changes
```

---

## 3. Cycle Detection

### 3.1 The Cycle Problem

Reference counting cannot collect cycles: objects that reference each other but are not reachable from any root.

```python
def demonstrate_cycle_leak():
    """Show how reference counting fails with cycles."""
    print("=== Cycle Leak Demo ===\n")

    a = RefCountedObject("A")
    b = RefCountedObject("B")

    # Both referenced from stack
    a.inc_ref()  # stack ref
    b.inc_ref()  # stack ref

    # Create cycle: A -> B -> A
    a.set_field("next", b)   # A.ref=1(stack), B.ref=1(stack)+1(A)=2
    b.set_field("next", a)   # A.ref=1(stack)+1(B)=2, B.ref=2

    print(f"Before removing stack refs: A.rc={a.ref_count}, B.rc={b.ref_count}")

    # Remove stack references
    a.dec_ref()  # A.rc = 2-1 = 1
    b.dec_ref()  # B.rc = 2-1 = 1

    print(f"After removing stack refs: A.rc={a.ref_count}, B.rc={b.ref_count}")
    print("Neither is freed! Both have rc=1 due to the cycle.")
    print("This is a memory leak.\n")

# demonstrate_cycle_leak()
```

### 3.2 Trial Deletion (Recycler Algorithm)

The trial deletion algorithm (used in CPython) identifies cycles by tentatively removing internal references:

1. For each object in the candidate set, tentatively decrement the ref counts of all objects it references.
2. Objects whose ref count drops to zero (tentatively) are garbage.
3. Objects still reachable from external references survive.

```python
class CycleDetector:
    """
    Cycle detector using trial deletion algorithm.
    Based on the approach used in CPython's gc module.
    """

    def __init__(self):
        self.tracked_objects = []  # Objects that might be in cycles

    def track(self, obj):
        """Add an object to cycle detection tracking."""
        self.tracked_objects.append(obj)

    def collect_cycles(self, root_set):
        """
        Detect and collect cyclic garbage.

        Phase 1: Trial deletion -- tentatively decrement ref counts
                 for all internal references
        Phase 2: Scan -- objects with tentative rc > 0 are reachable
                 (mark them and their transitive closure as alive)
        Phase 3: Collect -- remaining objects with tentative rc = 0 are garbage
        """
        print("=== Cycle Collection ===")

        if not self.tracked_objects:
            print("No tracked objects.")
            return []

        # Phase 1: Compute tentative ref counts
        tentative_rc = {}
        for obj in self.tracked_objects:
            tentative_rc[id(obj)] = obj.ref_count

        # Subtract internal references
        for obj in self.tracked_objects:
            for field, ref in obj.references.items():
                if id(ref) in tentative_rc:
                    tentative_rc[id(ref)] -= 1

        print("\nPhase 1 - Trial deletion (tentative ref counts):")
        for obj in self.tracked_objects:
            orig = obj.ref_count
            tent = tentative_rc[id(obj)]
            print(f"  {obj.name}: original_rc={orig}, tentative_rc={tent}")

        # Phase 2: Scan -- find objects reachable from roots or external refs
        alive = set()

        # Objects with tentative_rc > 0 have external references
        worklist = []
        for obj in self.tracked_objects:
            if tentative_rc[id(obj)] > 0:
                worklist.append(obj)
                alive.add(id(obj))

        # Also add objects reachable from root set
        for root in root_set:
            if id(root) not in alive:
                worklist.append(root)
                alive.add(id(root))

        # Propagate reachability
        while worklist:
            obj = worklist.pop()
            for field, ref in obj.references.items():
                if id(ref) not in alive and id(ref) in tentative_rc:
                    alive.add(id(ref))
                    worklist.append(ref)

        print("\nPhase 2 - Reachability scan:")
        for obj in self.tracked_objects:
            status = "ALIVE" if id(obj) in alive else "GARBAGE"
            print(f"  {obj.name}: {status}")

        # Phase 3: Collect garbage
        garbage = [obj for obj in self.tracked_objects
                   if id(obj) not in alive]

        print("\nPhase 3 - Collection:")
        for obj in garbage:
            print(f"  Collecting {obj.name}")
            self.tracked_objects.remove(obj)

        return garbage


def demonstrate_cycle_detection():
    """Demonstrate cycle detection with trial deletion."""
    RefCountedObject._all_objects = []

    print("=== Cycle Detection Demo ===\n")

    detector = CycleDetector()

    # Create objects
    root = RefCountedObject("root")
    root.inc_ref()  # stack reference

    a = RefCountedObject("A")
    b = RefCountedObject("B")
    c = RefCountedObject("C")  # Not in cycle, reachable from root
    d = RefCountedObject("D")  # In cycle but unreachable
    e = RefCountedObject("E")  # In cycle but unreachable

    # Build graph
    root.set_field("child", c)
    c.set_field("next", a)

    # Create unreachable cycle: D <-> E
    d.set_field("partner", e)
    e.set_field("partner", d)

    # Track potentially cyclic objects
    for obj in [a, b, c, d, e]:
        detector.track(obj)

    print("Object graph:")
    print("  root -> C -> A")
    print("  D <-> E  (unreachable cycle)")
    print()

    garbage = detector.collect_cycles(root_set=[root])
    print(f"\nGarbage collected: {[g.name for g in garbage]}")

demonstrate_cycle_detection()
```

---

## 4. Mark-Sweep Collection

### 4.1 Algorithm

Mark-sweep is the most fundamental tracing collector. It works in two phases:

1. **Mark phase**: Starting from the root set, traverse all reachable objects and mark them.
2. **Sweep phase**: Scan the entire heap; any unmarked object is garbage and is freed.

```python
class MarkSweepGC:
    """
    Mark-Sweep garbage collector implementation.
    """

    def __init__(self, heap_size=1024):
        self.heap_size = heap_size
        self.used = 0
        self.objects = []       # All allocated objects
        self.root_set = []      # Root references
        self.collections = 0
        self.total_freed = 0

    def allocate(self, name, size=1):
        """Allocate a new object on the heap."""
        if self.used + size > self.heap_size:
            # Trigger GC
            freed = self.collect()
            if self.used + size > self.heap_size:
                raise MemoryError(f"Out of memory: need {size}, "
                                  f"have {self.heap_size - self.used}")

        obj = {
            'name': name,
            'size': size,
            'marked': False,
            'references': [],  # List of referenced objects
        }
        self.objects.append(obj)
        self.used += size
        return obj

    def add_root(self, obj):
        """Add an object to the root set."""
        self.root_set.append(obj)

    def remove_root(self, obj):
        """Remove an object from the root set."""
        if obj in self.root_set:
            self.root_set.remove(obj)

    def add_reference(self, source, target):
        """Add a reference from source to target."""
        source['references'].append(target)

    def collect(self):
        """Run mark-sweep collection."""
        self.collections += 1
        print(f"\n--- GC #{self.collections} ---")
        print(f"Before: {len(self.objects)} objects, {self.used} bytes used")

        # Mark phase
        self._mark()

        # Sweep phase
        freed = self._sweep()

        print(f"After: {len(self.objects)} objects, {self.used} bytes used")
        print(f"Freed: {freed} bytes")
        self.total_freed += freed
        return freed

    def _mark(self):
        """Mark all reachable objects starting from the root set."""
        # Clear all marks
        for obj in self.objects:
            obj['marked'] = False

        # DFS from roots
        worklist = list(self.root_set)
        while worklist:
            obj = worklist.pop()
            if obj['marked']:
                continue
            obj['marked'] = True
            for ref in obj['references']:
                if not ref['marked']:
                    worklist.append(ref)

    def _sweep(self):
        """Sweep the heap, freeing unmarked objects."""
        freed = 0
        surviving = []

        for obj in self.objects:
            if obj['marked']:
                surviving.append(obj)
            else:
                freed += obj['size']
                self.used -= obj['size']
                print(f"  Swept: {obj['name']} ({obj['size']} bytes)")

        self.objects = surviving
        return freed

    def status(self):
        """Print heap status."""
        print(f"\nHeap: {self.used}/{self.heap_size} bytes used "
              f"({100*self.used/self.heap_size:.1f}%)")
        print(f"Objects: {len(self.objects)}")
        print(f"Roots: {[r['name'] for r in self.root_set]}")
        reachable = {r['name'] for r in self.root_set}
        for obj in self.objects:
            if obj['marked']:
                reachable.add(obj['name'])
        print(f"Reachable: {reachable}")


def demonstrate_mark_sweep():
    """Full demonstration of mark-sweep GC."""
    gc = MarkSweepGC(heap_size=100)

    print("=== Mark-Sweep GC Demo ===\n")

    # Allocate objects
    root = gc.allocate("root", 8)
    gc.add_root(root)

    a = gc.allocate("A", 16)
    b = gc.allocate("B", 16)
    c = gc.allocate("C", 16)
    d = gc.allocate("D", 16)
    e = gc.allocate("E", 16)

    # Build object graph
    gc.add_reference(root, a)
    gc.add_reference(root, b)
    gc.add_reference(a, c)
    # D and E are unreachable

    gc.status()

    # Run collection
    gc.collect()
    gc.status()

    # Allocate more (into freed space)
    f = gc.allocate("F", 16)
    gc.add_reference(b, f)

    # Remove root -> B (makes B, F unreachable)
    root['references'] = [a]

    gc.collect()
    gc.status()

demonstrate_mark_sweep()
```

### 4.2 Free List Management

After sweeping, freed memory blocks are placed on a **free list**. Future allocations search this list for a suitable block.

```python
class FreeListAllocator:
    """
    Free list memory allocator used with mark-sweep GC.
    """

    def __init__(self, heap_size):
        self.heap_size = heap_size
        self.heap = bytearray(heap_size)
        # Free list: list of (start, size) sorted by address
        self.free_list = [(0, heap_size)]
        self.allocated = {}  # start_addr -> (size, name)

    def allocate(self, name, size):
        """Allocate using first-fit strategy."""
        for i, (start, block_size) in enumerate(self.free_list):
            if block_size >= size:
                # Allocate from this block
                self.allocated[start] = (size, name)

                if block_size > size:
                    # Shrink the free block
                    self.free_list[i] = (start + size, block_size - size)
                else:
                    # Remove the free block entirely
                    self.free_list.pop(i)

                return start

        return None  # Out of memory

    def free(self, addr):
        """Free a block and coalesce adjacent free blocks."""
        if addr not in self.allocated:
            raise ValueError(f"Address {addr} not allocated")

        size, name = self.allocated.pop(addr)

        # Insert into free list (maintaining sorted order)
        new_block = (addr, size)
        inserted = False
        for i, (start, _) in enumerate(self.free_list):
            if addr < start:
                self.free_list.insert(i, new_block)
                inserted = True
                break
        if not inserted:
            self.free_list.append(new_block)

        # Coalesce adjacent free blocks
        self._coalesce()

    def _coalesce(self):
        """Merge adjacent free blocks."""
        if len(self.free_list) <= 1:
            return

        merged = [self.free_list[0]]
        for start, size in self.free_list[1:]:
            prev_start, prev_size = merged[-1]
            if prev_start + prev_size == start:
                # Adjacent -- merge
                merged[-1] = (prev_start, prev_size + size)
            else:
                merged.append((start, size))

        self.free_list = merged

    def dump(self):
        """Print heap layout."""
        print("\nHeap layout:")
        # Combine allocated and free blocks, sort by address
        blocks = []
        for addr, (size, name) in self.allocated.items():
            blocks.append((addr, size, f"[{name}]"))
        for addr, size in self.free_list:
            blocks.append((addr, size, "[FREE]"))
        blocks.sort()

        for addr, size, label in blocks:
            bar = "#" * min(size, 40)
            print(f"  {addr:4d}-{addr+size-1:4d}: {label:10s} {bar} ({size} bytes)")


def demonstrate_free_list():
    """Show free list management and fragmentation."""
    print("=== Free List Allocator ===\n")

    alloc = FreeListAllocator(100)

    # Allocate several blocks
    a = alloc.allocate("A", 20)
    b = alloc.allocate("B", 15)
    c = alloc.allocate("C", 25)
    d = alloc.allocate("D", 10)
    e = alloc.allocate("E", 20)

    alloc.dump()

    # Free B and D (creates fragmentation)
    print("\nFreeing B and D...")
    alloc.free(b)
    alloc.free(d)

    alloc.dump()

    # Try to allocate 25 bytes -- no single free block is large enough!
    print("\nTrying to allocate 25 bytes...")
    result = alloc.allocate("F", 25)
    if result is None:
        print("  FAILED: External fragmentation!")
        total_free = sum(size for _, size in alloc.free_list)
        print(f"  Total free: {total_free} bytes, but no contiguous block >= 25")

demonstrate_free_list()
```

### 4.3 Mark-Sweep Complexity

- **Time**: $O(L + H)$ where $L$ is the number of live objects (mark phase) and $H$ is the heap size (sweep phase). Note the sweep must scan the entire heap.
- **Space**: One mark bit per object (can be stored in the object header).
- **Pause time**: Proportional to heap size (both mark and sweep phases must complete before the program resumes).

---

## 5. Mark-Compact Collection

### 5.1 Motivation

Mark-sweep leaves the heap fragmented: live objects are scattered among free blocks. **Mark-compact** eliminates fragmentation by sliding all live objects to one end of the heap.

```
Before compaction:
  [A][FREE][B][FREE][FREE][C][D][FREE][E][FREE]

After compaction:
  [A][B][C][D][E][FREE FREE FREE FREE FREE...]
```

### 5.2 Algorithm

Mark-compact works in three (or four) passes:

1. **Mark**: Same as mark-sweep -- mark all reachable objects.
2. **Compute forwarding addresses**: Scan the heap, assigning each live object its new address (where it will move to).
3. **Update references**: Scan all objects, replacing each pointer with its forwarding address.
4. **Compact**: Move each object to its forwarding address.

```python
class MarkCompactGC:
    """
    Mark-Compact garbage collector.

    Eliminates fragmentation by compacting live objects.
    """

    def __init__(self, heap_size=256):
        self.heap_size = heap_size
        self.objects = []   # list of dicts with 'name', 'size', 'addr', 'references'
        self.next_addr = 0
        self.root_set = []

    def allocate(self, name, size=8):
        if self.next_addr + size > self.heap_size:
            self.collect()
            if self.next_addr + size > self.heap_size:
                raise MemoryError("Out of memory after compaction")

        obj = {
            'name': name,
            'size': size,
            'addr': self.next_addr,
            'marked': False,
            'forward': None,      # Forwarding address
            'references': [],
        }
        self.objects.append(obj)
        self.next_addr += size
        return obj

    def collect(self):
        print("\n--- Mark-Compact GC ---")
        self._print_heap("Before")

        # Phase 1: Mark
        for obj in self.objects:
            obj['marked'] = False
        worklist = list(self.root_set)
        while worklist:
            obj = worklist.pop()
            if obj['marked']:
                continue
            obj['marked'] = True
            for ref in obj['references']:
                if not ref['marked']:
                    worklist.append(ref)

        # Phase 2: Compute forwarding addresses
        compact_addr = 0
        for obj in self.objects:
            if obj['marked']:
                obj['forward'] = compact_addr
                compact_addr += obj['size']
            else:
                obj['forward'] = None
                print(f"  Garbage: {obj['name']} at addr {obj['addr']}")

        # Phase 3: Update references
        for obj in self.objects:
            if obj['marked']:
                obj['references'] = [
                    ref for ref in obj['references'] if ref['marked']
                ]

        # Phase 4: Compact (move objects)
        live_objects = []
        for obj in self.objects:
            if obj['marked']:
                old_addr = obj['addr']
                obj['addr'] = obj['forward']
                obj['forward'] = None
                obj['marked'] = False
                live_objects.append(obj)
                if old_addr != obj['addr']:
                    print(f"  Moved {obj['name']}: {old_addr} -> {obj['addr']}")

        self.objects = live_objects
        self.next_addr = compact_addr

        self._print_heap("After")

    def _print_heap(self, label):
        print(f"\n{label}:")
        for obj in self.objects:
            bar = "#" * (obj['size'] // 2)
            refs = [r['name'] for r in obj['references']]
            print(f"  [{obj['addr']:3d}] {obj['name']:5s} {bar} "
                  f"(size={obj['size']}, refs={refs})")
        print(f"  Next free address: {self.next_addr}")
        print(f"  Free space: {self.heap_size - self.next_addr}")


def demonstrate_mark_compact():
    """Demonstrate mark-compact GC."""
    gc = MarkCompactGC(heap_size=200)

    print("=== Mark-Compact GC Demo ===")

    # Allocate objects
    root = gc.allocate("root", 8)
    gc.root_set.append(root)

    a = gc.allocate("A", 24)
    b = gc.allocate("B", 16)
    c = gc.allocate("C", 32)
    d = gc.allocate("D", 16)
    e = gc.allocate("E", 24)

    # root -> A -> C, root -> E
    root['references'] = [a, e]
    a['references'] = [c]
    # B and D are unreachable

    gc.collect()

demonstrate_mark_compact()
```

### 5.3 Trade-offs

| Aspect | Mark-Sweep | Mark-Compact |
|--------|------------|--------------|
| Fragmentation | Yes | No |
| Allocation speed | Slow (free list search) | Fast (bump pointer) |
| Collection passes | 2 | 3-4 |
| Object movement | No | Yes (need to update all pointers) |
| Cache locality | Poor (scattered) | Good (compacted) |

---

## 6. Copying Collectors

### 6.1 Semi-Space Collector

The **semi-space** copying collector divides the heap into two halves: **from-space** and **to-space**. Allocation happens in from-space using a bump pointer. During collection, live objects are copied to to-space, and the spaces are swapped.

```
Before collection:
  From-space: [A][B][C][D][E]  (A, C, E are live)
  To-space:   [empty..........]

After collection:
  To-space:   [A][C][E][free..]   (now becomes from-space)
  From-space: [abandoned.........]  (now becomes to-space)
```

```python
class SemiSpaceGC:
    """
    Semi-space copying garbage collector.

    Key idea: Copy live objects from 'from-space' to 'to-space',
    then swap the spaces.
    """

    def __init__(self, heap_size=256):
        self.space_size = heap_size // 2
        # Each space is a list of objects
        self.from_space = []
        self.to_space = []
        self.from_used = 0
        self.root_set = []
        self.collections = 0

    def allocate(self, name, size=8):
        """Allocate in from-space using bump pointer."""
        if self.from_used + size > self.space_size:
            self.collect()
            if self.from_used + size > self.space_size:
                raise MemoryError("Out of memory")

        obj = {
            'name': name,
            'size': size,
            'references': [],
            'forwarding': None,  # Points to copy in to-space
        }
        self.from_space.append(obj)
        self.from_used += size
        return obj

    def collect(self):
        """Cheney's breadth-first copying collection."""
        self.collections += 1
        print(f"\n--- Copying GC #{self.collections} ---")
        print(f"From-space: {[o['name'] for o in self.from_space]}")

        self.to_space = []
        to_used = 0
        scan_index = 0

        # Copy roots
        new_roots = []
        for root in self.root_set:
            copied = self._copy(root)
            new_roots.append(copied)
            to_used += copied['size']

        # Cheney's scanning: breadth-first traversal
        while scan_index < len(self.to_space):
            obj = self.to_space[scan_index]
            new_refs = []
            for ref in obj['references']:
                copied_ref = self._copy(ref)
                new_refs.append(copied_ref)
                if copied_ref not in self.to_space:
                    self.to_space.append(copied_ref)
                    to_used += copied_ref['size']
            obj['references'] = new_refs
            scan_index += 1

        # Swap spaces
        self.from_space = self.to_space
        self.to_space = []
        self.from_used = to_used
        self.root_set = new_roots

        print(f"To-space (new from): {[o['name'] for o in self.from_space]}")
        print(f"Copied {len(self.from_space)} objects, {self.from_used} bytes")

    def _copy(self, obj):
        """
        Copy an object to to-space (if not already copied).
        Uses forwarding pointer to avoid duplicate copies.
        """
        if obj.get('forwarding') is not None:
            return obj['forwarding']

        # Create copy in to-space
        copy = {
            'name': obj['name'],
            'size': obj['size'],
            'references': list(obj['references']),  # Will be updated
            'forwarding': None,
        }

        # Set forwarding pointer in original
        obj['forwarding'] = copy

        # Add to to-space
        if copy not in self.to_space:
            self.to_space.append(copy)

        print(f"  Copied: {obj['name']}")
        return copy


def demonstrate_copying_gc():
    """Demonstrate semi-space copying GC."""
    gc = SemiSpaceGC(heap_size=400)

    print("=== Copying GC Demo ===\n")

    root = gc.allocate("root", 8)
    gc.root_set.append(root)

    a = gc.allocate("A", 24)
    b = gc.allocate("B", 16)   # Will be garbage
    c = gc.allocate("C", 32)
    d = gc.allocate("D", 16)   # Will be garbage
    e = gc.allocate("E", 24)

    # root -> A -> C, root -> E
    root['references'] = [a, e]
    a['references'] = [c]

    print(f"Allocated {len(gc.from_space)} objects, {gc.from_used} bytes")
    print(f"Live: root, A, C, E  |  Garbage: B, D")

    gc.collect()

    print(f"\nAfter GC: {gc.from_used} bytes in use "
          f"(freed {120 - gc.from_used} bytes from B and D)")

demonstrate_copying_gc()
```

### 6.2 Cheney's Algorithm

The key insight of **Cheney's algorithm** (1970) is that breadth-first copying can be done using the to-space itself as the worklist queue, requiring **no additional memory**:

```
To-space:
  [root][A]...[E]...[C]...
   ^scan      ^free

scan pointer: next object to process (scan its references)
free pointer: next free location (where to copy the next object)

When scan catches up to free, collection is complete.
```

```python
def cheneys_algorithm_detailed():
    """
    Detailed simulation of Cheney's algorithm showing scan/free pointers.
    """
    print("=== Cheney's Algorithm Trace ===\n")

    # Simulate objects as (name, size, references)
    objects = {
        'root': {'name': 'root', 'size': 8, 'refs': ['A', 'E']},
        'A':    {'name': 'A', 'size': 24, 'refs': ['C']},
        'B':    {'name': 'B', 'size': 16, 'refs': []},          # Garbage
        'C':    {'name': 'C', 'size': 32, 'refs': []},
        'D':    {'name': 'D', 'size': 16, 'refs': []},          # Garbage
        'E':    {'name': 'E', 'size': 24, 'refs': ['A']},       # Note: E -> A
    }

    # To-space: flat array with scan and free pointers
    to_space = []
    copied = {}   # old name -> to-space index
    scan = 0
    free = 0

    def copy_object(name):
        nonlocal free
        if name in copied:
            return copied[name]
        obj = objects[name]
        idx = len(to_space)
        to_space.append({'name': name, 'size': obj['size'], 'refs': list(obj['refs'])})
        copied[name] = idx
        free += obj['size']
        print(f"  Copy {name} to to-space[{idx}] (free={free})")
        return idx

    # Step 1: Copy root
    print("Step 1: Copy root to to-space")
    copy_object('root')

    # Step 2: Scan loop
    step = 2
    scan_idx = 0
    while scan_idx < len(to_space):
        obj = to_space[scan_idx]
        print(f"\nStep {step}: Scan {obj['name']} (scan_idx={scan_idx})")

        new_refs = []
        for ref_name in obj['refs']:
            to_idx = copy_object(ref_name)
            new_refs.append(to_space[to_idx]['name'])

        obj['refs'] = new_refs
        scan_idx += 1
        step += 1

    print(f"\nDone! scan={scan_idx}, to-space has {len(to_space)} objects")
    print("To-space contents:")
    for i, obj in enumerate(to_space):
        print(f"  [{i}] {obj['name']} (size={obj['size']}, refs={obj['refs']})")
    print(f"\nGarbage not copied: B, D")

cheneys_algorithm_detailed()
```

### 6.3 Trade-offs of Copying Collection

**Advantages**:
- Fast allocation (bump pointer -- just increment a counter)
- No fragmentation (objects are compacted during copy)
- Collection time proportional to live data only (dead objects are simply abandoned)
- Improves cache locality (objects copied in traversal order)

**Disadvantages**:
- Wastes half the heap (only one semi-space is usable at a time)
- Must copy every live object every collection (expensive if most data is long-lived)
- Must update all pointers to moved objects

**Key complexity**: Collection time is $O(L)$ where $L$ is the size of live data, compared to $O(L + H)$ for mark-sweep ($H$ = total heap size).

---

## 7. Generational Collection

### 7.1 The Generational Hypothesis

The **generational hypothesis** (also called the "infant mortality" or "weak generational hypothesis") states:

> Most objects die young.

Empirical studies across many languages consistently show that the majority of allocations become garbage very quickly. This distribution typically follows a "bathtub curve":

```
Object Survival Rate
     ^
100% |*
     |  *
     |    *
     |      **
     |         ****
     |              ********************  (long-lived objects)
     +------------------------------------> Object Age
       Young                    Old
```

### 7.2 Generational GC Design

Based on this hypothesis, generational GC divides the heap into generations:

```
┌─────────────────────────────────────────────────────────────┐
│                        Heap                                  │
├──────────────┬──────────────────────────────────────────────┤
│   Young Gen  │                Old Generation                 │
│   (Nursery)  │               (Tenured)                       │
│              │                                               │
│  Small, fast │  Large, collected infrequently                │
│  collected   │                                               │
│  frequently  │                                               │
├──────────────┼──────────────────────────────────────────────┤
│    ~10%      │                    ~90%                        │
└──────────────┴──────────────────────────────────────────────┘
```

- New objects are allocated in the **young generation** (nursery).
- The nursery is collected frequently (minor GC) -- most objects there are dead.
- Objects that survive several nursery collections are **promoted** (tenured) to the old generation.
- The old generation is collected infrequently (major GC).

```python
class GenerationalGC:
    """
    Two-generation garbage collector.
    """

    def __init__(self, nursery_size=64, old_size=256, promotion_threshold=2):
        self.nursery_size = nursery_size
        self.old_size = old_size
        self.promotion_threshold = promotion_threshold

        self.nursery = []
        self.old_gen = []
        self.nursery_used = 0
        self.old_used = 0
        self.root_set = []

        # Write barrier: track old-to-young pointers
        self.remembered_set = set()

        self.minor_collections = 0
        self.major_collections = 0

    def allocate(self, name, size=4):
        """Allocate in nursery (young generation)."""
        if self.nursery_used + size > self.nursery_size:
            self.minor_gc()
            if self.nursery_used + size > self.nursery_size:
                self.major_gc()
                if self.nursery_used + size > self.nursery_size:
                    raise MemoryError("Out of memory")

        obj = {
            'name': name,
            'size': size,
            'generation': 'young',
            'age': 0,          # Number of GCs survived
            'marked': False,
            'references': [],
        }
        self.nursery.append(obj)
        self.nursery_used += size
        return obj

    def write_barrier(self, source, target):
        """
        Write barrier: called when source.field = target.

        If source is in old gen and target is in nursery,
        record source in the remembered set.
        """
        source['references'].append(target)

        if source['generation'] == 'old' and target['generation'] == 'young':
            self.remembered_set.add(id(source))
            # In practice, we'd store the actual object reference

    def minor_gc(self):
        """Collect the nursery (young generation)."""
        self.minor_collections += 1
        print(f"\n--- Minor GC #{self.minor_collections} ---")
        print(f"Nursery: {[o['name'] for o in self.nursery]}")

        # Roots for nursery collection:
        # 1. Program roots that point into nursery
        # 2. Old-to-young references (remembered set)
        nursery_roots = set()

        for root in self.root_set:
            if root['generation'] == 'young':
                nursery_roots.add(id(root))
            # Also trace from roots into nursery
            for ref in root['references']:
                if ref['generation'] == 'young':
                    nursery_roots.add(id(ref))

        # Trace from old gen (via remembered set)
        for old_obj in self.old_gen:
            for ref in old_obj['references']:
                if ref['generation'] == 'young':
                    nursery_roots.add(id(ref))

        # Mark reachable nursery objects
        marked = set()
        worklist = [obj for obj in self.nursery if id(obj) in nursery_roots]

        while worklist:
            obj = worklist.pop()
            if id(obj) in marked:
                continue
            marked.add(id(obj))
            for ref in obj['references']:
                if ref['generation'] == 'young' and id(ref) not in marked:
                    worklist.append(ref)

        # Process nursery objects
        survivors = []
        promoted = []
        freed = 0

        for obj in self.nursery:
            if id(obj) in marked:
                obj['age'] += 1
                if obj['age'] >= self.promotion_threshold:
                    # Promote to old generation
                    obj['generation'] = 'old'
                    self.old_gen.append(obj)
                    self.old_used += obj['size']
                    promoted.append(obj['name'])
                else:
                    survivors.append(obj)
            else:
                freed += obj['size']
                print(f"  Freed: {obj['name']}")

        self.nursery = survivors
        self.nursery_used = sum(o['size'] for o in self.nursery)

        if promoted:
            print(f"  Promoted to old gen: {promoted}")
        print(f"  Freed {freed} bytes, {len(self.nursery)} nursery objects remain")

    def major_gc(self):
        """Full heap collection (both generations)."""
        self.major_collections += 1
        print(f"\n=== Major GC #{self.major_collections} ===")

        all_objects = self.nursery + self.old_gen

        # Mark from roots
        marked = set()
        worklist = list(self.root_set)
        while worklist:
            obj = worklist.pop()
            if id(obj) in marked:
                continue
            marked.add(id(obj))
            for ref in obj['references']:
                if id(ref) not in marked:
                    worklist.append(ref)

        # Sweep both generations
        self.nursery = [o for o in self.nursery if id(o) in marked]
        self.old_gen = [o for o in self.old_gen if id(o) in marked]
        self.nursery_used = sum(o['size'] for o in self.nursery)
        self.old_used = sum(o['size'] for o in self.old_gen)

        freed_count = len(all_objects) - len(self.nursery) - len(self.old_gen)
        print(f"  Freed {freed_count} objects")

    def status(self):
        print(f"\n  Nursery: {self.nursery_used}/{self.nursery_size} "
              f"({len(self.nursery)} objects)")
        print(f"  Old gen: {self.old_used}/{self.old_size} "
              f"({len(self.old_gen)} objects)")


def demonstrate_generational_gc():
    """Demonstrate generational GC with promotions."""
    gc = GenerationalGC(nursery_size=40, old_size=200, promotion_threshold=2)

    print("=== Generational GC Demo ===")

    # Create a long-lived root
    root = gc.allocate("root", 4)
    gc.root_set.append(root)

    # Simulate workload: allocate temporary objects with some long-lived
    for wave in range(3):
        print(f"\n--- Allocation wave {wave + 1} ---")

        # Long-lived object (will eventually be promoted)
        long_lived = gc.allocate(f"L{wave}", 4)
        root['references'].append(long_lived)

        # Short-lived temporaries
        for i in range(3):
            temp = gc.allocate(f"T{wave}_{i}", 4)
            # temp is not referenced by anything persistent -> garbage

        gc.status()

    gc.status()
    print(f"\nTotal minor GCs: {gc.minor_collections}")
    print(f"Total major GCs: {gc.major_collections}")

demonstrate_generational_gc()
```

### 7.3 Write Barriers

The key challenge with generational GC: what if an old-generation object points to a nursery object? During minor GC, we only scan the nursery -- we would miss this reference!

**Write barriers** solve this by intercepting every pointer store. If an old object is modified to point to a young object, the old object is added to the **remembered set** (also called the **card table**).

```python
class CardTable:
    """
    Card table write barrier implementation.

    The heap is divided into 'cards' (e.g., 512-byte blocks).
    When a pointer in a card is modified, the card is marked dirty.
    During minor GC, only dirty cards are scanned.
    """

    def __init__(self, heap_size, card_size=512):
        self.card_size = card_size
        self.num_cards = (heap_size + card_size - 1) // card_size
        self.cards = [False] * self.num_cards  # False = clean, True = dirty

    def mark_dirty(self, address):
        """Mark the card containing this address as dirty."""
        card_index = address // self.card_size
        if card_index < self.num_cards:
            self.cards[card_index] = True

    def get_dirty_cards(self):
        """Return indices of all dirty cards."""
        return [i for i, dirty in enumerate(self.cards) if dirty]

    def clear(self):
        """Clear all dirty bits after scanning."""
        self.cards = [False] * self.num_cards

    def __repr__(self):
        return ''.join('D' if d else '.' for d in self.cards)


# Example
print("=== Card Table Write Barrier ===")
ct = CardTable(heap_size=4096, card_size=512)
print(f"Card table ({ct.num_cards} cards): {ct}")

# Simulate pointer writes at various addresses
writes = [100, 600, 1200, 3500]
for addr in writes:
    ct.mark_dirty(addr)
    print(f"Write at addr {addr} -> card {addr // 512} dirty")

print(f"Card table: {ct}")
print(f"Dirty cards to scan: {ct.get_dirty_cards()}")
```

### 7.4 Multi-Generation Schemes

Many production collectors use more than two generations:

```
JVM Generational Layout:
┌────────────────┬─────────────────────────────────────┐
│   Young Gen    │          Old Generation               │
├────┬─────┬─────┤                                       │
│Eden│ S0  │ S1  │                                       │
│    │(from│(to) │                                       │
│    │surv)│surv)│                                       │
├────┴─────┴─────┴───────────────────────────────────────┤
│ New objects     │ Objects surviving multiple            │
│ allocated here  │ minor GCs are promoted here           │
└────────────────┴───────────────────────────────────────┘

Eden: New allocations
S0/S1: Survivor spaces (semi-space copying for young gen)
Old: Long-lived objects
```

---

## 8. Incremental and Concurrent GC

### 8.1 The Pause Problem

Stop-the-world GC pauses all application threads (called **mutator** threads) during collection. For interactive or real-time applications, these pauses are problematic.

| Application | Acceptable Pause |
|-------------|-----------------|
| Batch processing | Seconds |
| Web server | 100ms |
| Interactive UI | 16ms (60 FPS) |
| Trading system | < 1ms |
| Hard real-time | Never |

### 8.2 Incremental Collection

**Incremental GC** breaks collection into small steps, interleaved with mutator execution:

```
Stop-the-world:
  Mutator ████████████░░░░░░░░████████████
  GC                  ████████

Incremental:
  Mutator ██████░██████░██████░██████░█████
  GC            ░      ░      ░      ░
              (small increments)
```

The challenge: the mutator can modify the object graph while the collector is working.

### 8.3 Concurrent Collection

**Concurrent GC** runs the collector on separate threads simultaneously with the mutator:

```
Concurrent:
  Mutator ████████████████████████████████████
  GC      ░░░░░░░░░░░░░░░░░░░░░░░░
          (runs on separate thread)
```

This requires careful synchronization to handle concurrent modifications.

---

## 9. Tri-Color Marking

### 9.1 The Tri-Color Abstraction

Tri-color marking provides a framework for reasoning about incremental and concurrent GC. Objects are classified into three colors:

- **White**: Not yet visited (potentially garbage).
- **Grey**: Visited, but references not yet scanned.
- **Black**: Visited and all references scanned.

```
Invariant: No black object points directly to a white object.
           (All references from black objects go to black or grey objects.)
```

```python
from enum import Enum
from collections import deque

class Color(Enum):
    WHITE = "white"
    GREY = "grey"
    BLACK = "black"


class TriColorGC:
    """
    Tri-color marking garbage collector.

    Demonstrates incremental collection using the tri-color abstraction.
    """

    def __init__(self):
        self.objects = {}  # name -> {'color': Color, 'refs': [names]}
        self.root_set = set()
        self.grey_set = deque()  # Worklist of grey objects

    def add_object(self, name, refs=None):
        self.objects[name] = {
            'color': Color.WHITE,
            'refs': refs or [],
        }

    def add_root(self, name):
        self.root_set.add(name)

    def init_marking(self):
        """Initialize marking: color all roots grey."""
        print("=== Initialize Marking ===")
        for name in self.root_set:
            if name in self.objects:
                self.objects[name]['color'] = Color.GREY
                self.grey_set.append(name)
                print(f"  Root {name} -> GREY")

        self._print_state()

    def mark_step(self):
        """
        One incremental marking step: process one grey object.

        Returns True if there is more work to do.
        """
        if not self.grey_set:
            return False

        # Pick a grey object
        name = self.grey_set.popleft()
        obj = self.objects[name]

        print(f"\nMark step: Processing {name}")

        # Scan its references
        for ref_name in obj['refs']:
            if ref_name in self.objects:
                ref_obj = self.objects[ref_name]
                if ref_obj['color'] == Color.WHITE:
                    ref_obj['color'] = Color.GREY
                    self.grey_set.append(ref_name)
                    print(f"  {ref_name}: WHITE -> GREY")

        # Mark this object black
        obj['color'] = Color.BLACK
        print(f"  {name}: GREY -> BLACK")

        self._print_state()
        return len(self.grey_set) > 0

    def sweep(self):
        """Sweep: collect all white objects."""
        print("\n=== Sweep ===")
        garbage = [name for name, obj in self.objects.items()
                   if obj['color'] == Color.WHITE]

        for name in garbage:
            print(f"  Collected: {name}")
            del self.objects[name]

        # Reset colors for next cycle
        for obj in self.objects.values():
            obj['color'] = Color.WHITE

        return garbage

    def _print_state(self):
        white = [n for n, o in self.objects.items() if o['color'] == Color.WHITE]
        grey = [n for n, o in self.objects.items() if o['color'] == Color.GREY]
        black = [n for n, o in self.objects.items() if o['color'] == Color.BLACK]
        print(f"  State: WHITE={white}, GREY={grey}, BLACK={black}")


def demonstrate_tricolor():
    """Demonstrate incremental tri-color marking."""
    gc = TriColorGC()

    # Create object graph
    #   root -> A -> C
    #        -> B -> D
    #   E (unreachable)
    #   F -> G (both unreachable)
    gc.add_object('root', ['A', 'B'])
    gc.add_object('A', ['C'])
    gc.add_object('B', ['D'])
    gc.add_object('C', [])
    gc.add_object('D', [])
    gc.add_object('E', [])           # Unreachable
    gc.add_object('F', ['G'])        # Unreachable
    gc.add_object('G', [])           # Unreachable
    gc.add_root('root')

    print("=== Tri-Color Marking Demo ===\n")
    print("Object graph: root -> {A -> C, B -> D}, E, F -> G")

    # Initialize
    gc.init_marking()

    # Incremental marking steps
    step = 1
    while gc.mark_step():
        step += 1

    # Final step (returns False)
    print(f"\nMarking complete after {step} steps")

    # Sweep
    garbage = gc.sweep()
    print(f"\nGarbage collected: {garbage}")
    print(f"Surviving objects: {list(gc.objects.keys())}")

demonstrate_tricolor()
```

### 9.2 The Lost Object Problem

If the mutator modifies the object graph during incremental marking, it can cause a live object to be missed (collected as garbage). This happens when:

1. The mutator stores a reference from a black object to a white object.
2. The mutator deletes all grey-to-white paths to that white object.

Now the white object is reachable but has no grey predecessor -- it will be collected!

```
Before mutator action:
  [BLACK: A] -> [GREY: B] -> [WHITE: C]

Mutator does:
  A.ref = C     (black -> white!)
  B.ref = null  (removes grey -> white path)

After:
  [BLACK: A] -> [WHITE: C]    C is reachable but will be missed!
  [GREY: B]
```

### 9.3 Write Barrier Solutions

Two approaches to prevent the lost object problem:

**Dijkstra's write barrier** (snapshot-at-the-beginning): When the mutator stores a reference to a white object, grey the target.

```python
def dijkstra_write_barrier(source, field, new_target, gc):
    """
    Dijkstra's insertion barrier.
    When writing a reference, if the target is white, mark it grey.
    """
    if new_target and gc.objects[new_target]['color'] == Color.WHITE:
        gc.objects[new_target]['color'] = Color.GREY
        gc.grey_set.append(new_target)
        print(f"  Barrier: {new_target} WHITE -> GREY (insertion barrier)")

    # Perform the actual write
    source_obj = gc.objects[source]
    if field < len(source_obj['refs']):
        source_obj['refs'][field] = new_target
```

**Steele's write barrier** (incremental update): When a black object is modified, grey the source (re-scan it).

```python
def steele_write_barrier(source, field, new_target, gc):
    """
    Steele's write barrier.
    When a black object gets a new reference, mark it grey (rescan it).
    """
    source_obj = gc.objects[source]

    if source_obj['color'] == Color.BLACK:
        source_obj['color'] = Color.GREY
        gc.grey_set.append(source)
        print(f"  Barrier: {source} BLACK -> GREY (rescan)")

    # Perform the actual write
    if field < len(source_obj['refs']):
        source_obj['refs'][field] = new_target
```

### 9.4 Yuasa's Snapshot-at-the-Beginning

Yuasa's deletion barrier ensures that no white object loses its last grey predecessor. When deleting a reference, if the old target is white, mark it grey:

```python
def yuasa_write_barrier(source, field, new_target, gc):
    """
    Yuasa's deletion barrier (snapshot-at-the-beginning).
    When overwriting a reference, grey the OLD target if it's white.
    """
    source_obj = gc.objects[source]
    old_target = source_obj['refs'][field] if field < len(source_obj['refs']) else None

    if old_target and gc.objects[old_target]['color'] == Color.WHITE:
        gc.objects[old_target]['color'] = Color.GREY
        gc.grey_set.append(old_target)
        print(f"  Barrier: {old_target} WHITE -> GREY (deletion barrier)")

    # Perform the actual write
    if field < len(source_obj['refs']):
        source_obj['refs'][field] = new_target
```

---

## 10. GC in Real Systems

### 10.1 JVM Garbage Collectors

The JVM offers multiple GC implementations:

| Collector | Type | Generations | Pause Goal | Best For |
|-----------|------|-------------|------------|----------|
| **Serial GC** | Stop-the-world | Young + Old | None | Small heaps, single core |
| **Parallel GC** | Stop-the-world, parallel | Young + Old | Throughput | Batch, background |
| **CMS** (deprecated) | Concurrent mark-sweep | Young + Old | Low latency | Deprecated in JDK 14 |
| **G1** | Region-based, concurrent | Young + Old | Configurable | General purpose (default) |
| **ZGC** | Concurrent, region-based | No generations* | < 10ms | Large heaps (TB-scale) |
| **Shenandoah** | Concurrent, compact | No generations* | < 10ms | Low latency |

**G1 (Garbage First) Collector**:

```
G1 Heap Layout (region-based):

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│E │E │S │  │O │O │  │O │H ├──┤  │E │O │  │S │O │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

E = Eden (young)       S = Survivor (young)
O = Old                H = Humongous (large objects, spans regions)
(empty) = Free region

Key idea: Collect regions with the most garbage first ("garbage first").
          Pause time is controlled by collecting a subset of regions.
```

**ZGC** key techniques:
- **Colored pointers**: Uses unused bits in 64-bit pointers for GC metadata
- **Load barriers**: Check and fix pointers on every load (not store)
- **Concurrent relocation**: Moves objects while mutators are running
- Pause times under 10ms regardless of heap size

### 10.2 Go Garbage Collector

Go uses a **concurrent, tri-color mark-sweep** collector:

- Non-generational (all objects are in one space)
- Concurrent marking using Dijkstra's write barrier
- STW pauses only for root scanning (typically < 1ms)
- Target: minimize latency (not maximize throughput)
- Pacing: GC starts when heap grows to 2x the live set

```
Go GC Phases:
1. STW: Scan stacks, enable write barrier  (~100μs)
2. Concurrent: Mark phase (mutators run, write barrier active)
3. STW: Rescan, disable write barrier      (~100μs)
4. Concurrent: Sweep phase
```

### 10.3 CPython Garbage Collector

CPython uses a hybrid approach:

- **Primary**: Reference counting (immediate reclamation)
- **Secondary**: Generational cycle detector (three generations)

```python
import gc
import sys

def cpython_gc_info():
    """Show CPython GC configuration and statistics."""
    print("=== CPython GC Info ===\n")

    # GC thresholds (generation 0, 1, 2)
    thresholds = gc.get_threshold()
    print(f"Thresholds: Gen0={thresholds[0]}, Gen1={thresholds[1]}, Gen2={thresholds[2]}")
    print("  Gen0: collected after {0} new allocations".format(thresholds[0]))
    print("  Gen1: collected after {0} Gen0 collections".format(thresholds[1]))
    print("  Gen2: collected after {0} Gen1 collections".format(thresholds[2]))

    # Current counts
    counts = gc.get_count()
    print(f"\nCurrent counts: Gen0={counts[0]}, Gen1={counts[1]}, Gen2={counts[2]}")

    # Reference count example
    a = [1, 2, 3]
    print(f"\nReference count of a list: {sys.getrefcount(a)}")
    # Note: getrefcount itself creates a temporary reference, so it's always +1

    # Demonstrate cycle detection
    print("\nCycle detection demo:")
    gc.collect()  # Clean slate

    # Create a cycle
    class Node:
        def __init__(self, name):
            self.name = name
            self.next = None
        def __repr__(self):
            return f"Node({self.name})"

    x = Node("X")
    y = Node("Y")
    x.next = y
    y.next = x  # Cycle!

    print(f"  Created cycle: X -> Y -> X")
    print(f"  Ref count of X: {sys.getrefcount(x) - 1}")  # Subtract our ref

    # Delete local references
    del x, y

    # Objects are NOT freed (cycle keeps them alive)
    # Run cycle detector
    collected = gc.collect()
    print(f"  After gc.collect(): {collected} objects collected")

cpython_gc_info()
```

### 10.4 Rust: Ownership Instead of GC

Rust takes a radically different approach: **no garbage collector at all**. Instead, it uses a compile-time ownership system:

```
Rust Ownership Rules:
1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped (freed)
3. References can borrow values (immutably or mutably, but not both)
4. The borrow checker enforces these rules at compile time
```

This eliminates GC pauses entirely but requires the programmer to think about ownership. For cases where ownership is unclear, Rust provides:

- `Rc<T>`: Reference counting (single-threaded)
- `Arc<T>`: Atomic reference counting (multi-threaded)
- These can create cycles, handled by `Weak<T>` references

### 10.5 Comparison

```python
def gc_comparison():
    """Compare GC characteristics across languages/runtimes."""
    collectors = [
        {
            'system': 'JVM G1',
            'type': 'Generational, region-based',
            'concurrent': 'Mostly concurrent',
            'pause': '~10-200ms',
            'throughput': 'High',
            'heap_overhead': '~10-20%',
        },
        {
            'system': 'JVM ZGC',
            'type': 'Region-based, load barrier',
            'concurrent': 'Fully concurrent',
            'pause': '<10ms',
            'throughput': 'Good',
            'heap_overhead': '~15%',
        },
        {
            'system': 'Go',
            'type': 'Tri-color mark-sweep',
            'concurrent': 'Mostly concurrent',
            'pause': '<1ms',
            'throughput': 'Moderate',
            'heap_overhead': '~25-50%',
        },
        {
            'system': 'CPython',
            'type': 'Ref counting + generational',
            'concurrent': 'No (GIL)',
            'pause': '~1-50ms',
            'throughput': 'Low',
            'heap_overhead': 'Per-object refcount',
        },
        {
            'system': 'Rust',
            'type': 'Ownership (no GC)',
            'concurrent': 'N/A',
            'pause': 'None',
            'throughput': 'Best',
            'heap_overhead': 'None',
        },
    ]

    print("=== GC Comparison ===\n")
    header = f"{'System':<12} {'Type':<30} {'Pause':<12} {'Throughput':<12}"
    print(header)
    print("-" * len(header))
    for c in collectors:
        print(f"{c['system']:<12} {c['type']:<30} {c['pause']:<12} {c['throughput']:<12}")

gc_comparison()
```

---

## 11. GC Tuning and Metrics

### 11.1 Key Metrics

| Metric | Definition | Goal |
|--------|-----------|------|
| **Throughput** | Fraction of time spent in application code (not GC) | Maximize (>95%) |
| **Pause time** | Duration of GC-induced application pauses | Minimize |
| **Footprint** | Total memory used (heap + GC metadata) | Minimize |
| **Promptness** | Time between object death and memory reclamation | Minimize |
| **Allocation rate** | Bytes allocated per second | Not directly controllable |
| **Promotion rate** | Bytes promoted from young to old gen per second | Minimize |

### 11.2 The GC Tuning Triangle

You cannot optimize all metrics simultaneously:

```
                Throughput
                    /\
                   /  \
                  /    \
                 /      \
                /  Pick  \
               /   Two    \
              /            \
             /              \
            /______________\
        Pause Time      Footprint
```

- **High throughput + low pause** = large heap (high footprint)
- **Low pause + small footprint** = more frequent collections (lower throughput)
- **High throughput + small footprint** = longer pauses (batched collection)

### 11.3 GC Tuning Example

```python
class GCSimulator:
    """
    Simulate GC behavior under different configurations.
    """

    def __init__(self, heap_size, nursery_ratio=0.1, gc_overhead_factor=0.1):
        self.heap_size = heap_size
        self.nursery_size = int(heap_size * nursery_ratio)
        self.gc_overhead_factor = gc_overhead_factor

        self.total_allocated = 0
        self.total_gc_time = 0
        self.total_app_time = 0
        self.gc_events = []

    def simulate_workload(self, alloc_rate_mb_s, live_set_mb, duration_s,
                          survival_rate=0.05):
        """
        Simulate a workload and report GC metrics.

        alloc_rate_mb_s: MB allocated per second
        live_set_mb: MB of long-lived data
        duration_s: simulation duration in seconds
        survival_rate: fraction of nursery objects that survive
        """
        time_elapsed = 0
        nursery_used = 0

        while time_elapsed < duration_s:
            # Simulate allocation
            alloc_per_tick = alloc_rate_mb_s * 0.001  # 1ms ticks
            nursery_used += alloc_per_tick
            self.total_allocated += alloc_per_tick
            self.total_app_time += 0.001

            if nursery_used >= self.nursery_size:
                # Minor GC
                survivors = nursery_used * survival_rate
                gc_time = nursery_used * self.gc_overhead_factor * 0.001

                self.gc_events.append({
                    'time': time_elapsed,
                    'type': 'minor',
                    'pause_ms': gc_time * 1000,
                    'freed_mb': nursery_used - survivors,
                })

                self.total_gc_time += gc_time
                nursery_used = survivors

            time_elapsed += 0.001

    def report(self):
        """Print GC performance report."""
        total_time = self.total_app_time + self.total_gc_time
        throughput = self.total_app_time / total_time * 100

        minor_gcs = [e for e in self.gc_events if e['type'] == 'minor']
        if minor_gcs:
            pauses = [e['pause_ms'] for e in minor_gcs]
            avg_pause = sum(pauses) / len(pauses)
            max_pause = max(pauses)
            p99_pause = sorted(pauses)[int(len(pauses) * 0.99)]
        else:
            avg_pause = max_pause = p99_pause = 0

        print(f"\n=== GC Performance Report ===")
        print(f"Total allocated: {self.total_allocated:.1f} MB")
        print(f"GC collections: {len(self.gc_events)}")
        print(f"Throughput: {throughput:.1f}%")
        print(f"Avg pause: {avg_pause:.2f} ms")
        print(f"Max pause: {max_pause:.2f} ms")
        print(f"P99 pause: {p99_pause:.2f} ms")
        print(f"Total GC time: {self.total_gc_time*1000:.1f} ms")


# Compare different heap configurations
for heap_mb, nursery_pct in [(256, 0.1), (512, 0.1), (256, 0.25)]:
    print(f"\n{'='*50}")
    print(f"Config: heap={heap_mb}MB, nursery={nursery_pct*100:.0f}%")
    sim = GCSimulator(heap_mb, nursery_ratio=nursery_pct)
    sim.simulate_workload(
        alloc_rate_mb_s=500,   # 500 MB/s allocation rate
        live_set_mb=100,       # 100 MB long-lived data
        duration_s=10,         # 10 second simulation
        survival_rate=0.05,    # 5% of nursery objects survive
    )
    sim.report()
```

### 11.4 Common GC Tuning Strategies

1. **Increase heap size**: Reduces GC frequency but increases pause times.
2. **Tune nursery size**: Larger nursery = fewer minor GCs but longer pauses.
3. **Adjust promotion threshold**: Higher threshold keeps objects in nursery longer (good if they die soon after).
4. **Choose the right collector**: Match the collector to your latency/throughput requirements.
5. **Reduce allocation rate**: The most effective strategy -- allocate fewer short-lived objects.

```python
def tuning_checklist():
    """GC tuning decision checklist."""
    print("=== GC Tuning Checklist ===\n")
    checks = [
        ("High GC frequency?", "Increase heap/nursery size"),
        ("Long GC pauses?", "Use concurrent collector (G1/ZGC/Shenandoah)"),
        ("High promotion rate?", "Increase nursery size or promotion threshold"),
        ("Full GC happening?", "Old gen too small or memory leak"),
        ("Low throughput (<95%)?", "Reduce allocation rate or increase heap"),
        ("OOM errors?", "Memory leak, or genuinely need more heap"),
    ]

    for symptom, remedy in checks:
        print(f"  Symptom: {symptom}")
        print(f"  Remedy:  {remedy}\n")

tuning_checklist()
```

---

## 12. Summary

Garbage collection is a fundamental aspect of modern programming language runtimes. We covered the following approaches:

| Algorithm | Reclaim Timing | Handles Cycles | Fragmentation | Copy Cost | Pause |
|-----------|---------------|----------------|---------------|-----------|-------|
| **Ref Counting** | Immediate | No | Yes | No | Incremental |
| **Mark-Sweep** | Batch | Yes | Yes | No | STW |
| **Mark-Compact** | Batch | Yes | No | Slide | STW |
| **Copying** | Batch | Yes | No | Copy all live | STW |
| **Generational** | Mixed | Yes | Varies | Young only | Short minor |
| **Concurrent** | Batch | Yes | Varies | Varies | Minimal |

Key principles:

1. **The generational hypothesis** drives the design of most modern collectors: collect young objects frequently and old objects rarely.
2. **Write barriers** are essential for both generational and concurrent GC, tracking cross-generation or concurrent mutations.
3. **Tri-color marking** provides a clean framework for reasoning about concurrent and incremental collection.
4. **No single GC is best for all workloads** -- the right choice depends on your latency, throughput, and footprint requirements.
5. **Rust's ownership model** shows that GC can be avoided entirely with the right type system, at the cost of programming complexity.

---

## 13. Exercises

### Exercise 1: Reference Counting Simulation

Implement a reference counting system that supports:
- Object allocation and deallocation
- Pointer assignment with automatic ref count updates
- Detection of when an object becomes garbage

Test with the following scenario:
```
A -> B -> C
A -> D
Remove A -> B  (should free B and C)
Remove A -> D  (should free D)
```

### Exercise 2: Mark-Sweep Implementation

Extend the mark-sweep implementation to handle:
(a) Multiple root sets (stack roots, global roots, register roots).
(b) A free list allocator with first-fit, best-fit, and worst-fit strategies. Compare fragmentation under each strategy.
(c) Measure and report: total pause time, objects freed, fragmentation ratio.

### Exercise 3: Copying Collector

Implement a full semi-space copying collector with Cheney's algorithm that:
(a) Uses a flat byte array as the heap (not Python objects).
(b) Implements forwarding pointers for moved objects.
(c) Handles the case where the live set exceeds half the heap (triggering an out-of-memory error).
(d) Measures allocation speed (bump pointer) vs. free-list allocation.

### Exercise 4: Generational GC

Extend the generational GC to:
(a) Three generations (nursery, intermediate, old).
(b) Implement a card table write barrier.
(c) Track and report promotion rate over time.
(d) Experiment: what happens if you set the promotion threshold too high? Too low?

### Exercise 5: Tri-Color Marking Safety

Given the following object graph and GC state:

```
Objects: A(BLACK), B(GREY), C(WHITE), D(WHITE), E(BLACK)
References: A->{B}, B->{C,D}, E->{}
Roots: A, E
```

(a) Show that marking completes correctly without mutator interference.
(b) Now suppose the mutator performs: `A.ref = C; B.ref = null` (removing B->C). Show that C becomes lost without a write barrier.
(c) Show how Dijkstra's barrier prevents the lost object.
(d) Show how Yuasa's barrier prevents the lost object.

### Exercise 6: GC Comparison

Write a benchmark that allocates objects in various patterns and measures performance under different simulated GC strategies:
(a) Short-lived burst: allocate 1 million small objects, discard all.
(b) Long-lived: allocate 10,000 objects, keep all, then allocate 1 million temporary objects.
(c) Cyclic: create linked lists that form cycles.

Compare: reference counting, mark-sweep, copying, and generational collectors in terms of total time and peak memory usage.

---

## 14. References

1. Jones, R., Hosking, A., & Moss, E. (2012). *The Garbage Collection Handbook: The Art of Automatic Memory Management*. CRC Press.
2. Wilson, P. R. (1992). "Uniprocessor Garbage Collection Techniques." *International Workshop on Memory Management*, Springer.
3. Cheney, C. J. (1970). "A Nonrecursive List Compacting Algorithm." *Communications of the ACM*, 13(11).
4. Dijkstra, E. W., Lamport, L., Martin, A. J., Scholten, C. S., & Steffens, E. F. M. (1978). "On-the-fly Garbage Collection: An Exercise in Cooperation." *Communications of the ACM*, 21(11).
5. Lieberman, H., & Hewitt, C. (1983). "A Real-Time Garbage Collector Based on the Lifetimes of Objects." *Communications of the ACM*, 26(6).
6. Bacon, D. F., & Rajan, V. T. (2001). "Concurrent Cycle Collection in Reference Counted Systems." *ECOOP*.
7. Tene, G., Iyengar, B., & Wolf, M. (2011). "C4: The Continuously Concurrent Compacting Collector." *ISMM*.

---

[Previous: 13. Loop Optimization](./13_Loop_Optimization.md) | [Next: 15. Interpreters and Virtual Machines](./15_Interpreters_and_Virtual_Machines.md) | [Overview](./00_Overview.md)
