# 가비지 컬렉션(Garbage Collection)

**이전**: [13. 루프 최적화](./13_Loop_Optimization.md) | **다음**: [15. 인터프리터와 가상 머신](./15_Interpreters_and_Virtual_Machines.md)

---

메모리 관리는 프로그래밍 언어 구현에서 가장 근본적인 과제 중 하나입니다. 모든 프로그램은 메모리를 할당하고, 그 메모리는 결국 회수되어야 합니다. 수동 메모리 관리(C나 C++에서처럼)는 프로그래머에게 완전한 제어권을 주지만 오류가 발생하기 쉽기로 악명 높습니다. 허상 포인터(dangling pointer), 이중 해제(double free), 메모리 누수(memory leak)는 소프트웨어 역사에서 가장 흔하고 위험한 버그들입니다. 가비지 컬렉션(GC)은 이 과정을 자동화하여 프로그래머를 객체 수명 추적으로부터 해방시켜 줍니다 -- 단, 런타임 오버헤드와 제어권 감소라는 비용을 치릅니다.

이 레슨은 주요 가비지 컬렉션 알고리즘과 그 트레이드오프를 다룹니다. 가장 단순한 참조 계수(reference counting) 방식부터 프로덕션 시스템에서 사용되는 정교한 세대별(generational) 및 동시(concurrent) 컬렉터까지 살펴봅니다.

**난이도**: ⭐⭐⭐

**선수 지식**: [10. 런타임 환경](./10_Runtime_Environments.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수동 및 자동 메모리 관리 방식을 비교한다
2. 사이클 감지가 포함된 참조 계수를 구현한다
3. 추적(tracing) 컬렉터를 이해한다: 마크-스윕, 마크-컴팩트, 복사(copying)
4. 세대별 GC와 그것이 작동하는 이유(세대별 가설)를 설명한다
5. 삼색 마킹(tri-color marking)을 이용한 증분(incremental) 및 동시(concurrent) GC 알고리즘을 기술한다
6. 실제 시스템(JVM, Go, Python, Rust)의 GC 전략을 분석한다
7. GC 튜닝과 성능 트레이드오프에 대해 추론한다

---

## 목차

1. [메모리 관리 개요](#1-메모리-관리-개요)
2. [참조 계수](#2-참조-계수)
3. [사이클 감지](#3-사이클-감지)
4. [마크-스윕 컬렉션](#4-마크-스윕-컬렉션)
5. [마크-컴팩트 컬렉션](#5-마크-컴팩트-컬렉션)
6. [복사 컬렉터](#6-복사-컬렉터)
7. [세대별 컬렉션](#7-세대별-컬렉션)
8. [증분 및 동시 GC](#8-증분-및-동시-gc)
9. [삼색 마킹](#9-삼색-마킹)
10. [실제 시스템의 GC](#10-실제-시스템의-gc)
11. [GC 튜닝과 메트릭](#11-gc-튜닝과-메트릭)
12. [요약](#12-요약)
13. [연습 문제](#13-연습-문제)
14. [참고 자료](#14-참고-자료)

---

## 1. 메모리 관리 개요

### 1.1 힙(Heap)

동적 메모리 할당은 **힙(heap)**을 사용합니다 -- 런타임에 관리되는 메모리 영역입니다. 객체의 크기나 수명을 컴파일 시점에 알 수 없을 때 힙에 할당됩니다.

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

### 1.2 수동 메모리 관리

C 같은 언어에서는 프로그래머가 메모리를 명시적으로 관리합니다:

```c
// C manual memory management
int *arr = (int *)malloc(n * sizeof(int));  // Allocate
// ... use arr ...
free(arr);  // Deallocate -- programmer's responsibility
arr = NULL; // Avoid dangling pointer
```

수동 메모리 관리의 흔한 문제점:

| 문제 | 설명 | 결과 |
|------|------|------|
| **메모리 누수** | `free` 호출을 잊어버림 | 메모리 무한 증가 |
| **허상 포인터** | `free` 후 메모리 사용 | 정의되지 않은 동작, 충돌 |
| **이중 해제** | `free`를 두 번 호출 | 힙 손상 |
| **버퍼 오버플로우** | 할당 경계를 넘어 쓰기 | 보안 취약점 |
| **해제 후 사용** | 해제된 메모리 접근 | 데이터 손상 |

### 1.3 자동 메모리 관리

가비지 컬렉션은 프로그램이 더 이상 도달할 수 없는 객체를 자동으로 식별하고 회수합니다. 핵심 통찰: **루트 집합(root set)에서 시작하는 포인터 역참조 시퀀스로 도달할 수 없는 객체는 가비지입니다**.

**루트 집합(root set)**은 다음으로 구성됩니다:
- 전역 변수
- 스택의 지역 변수
- 포인터를 포함하는 CPU 레지스터
- 런타임이 관리하는 기타 참조

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

### 1.4 트레이드오프: 수동 vs 자동

| 측면 | 수동 | 참조 계수 | 추적 GC |
|------|------|-----------|---------|
| **처리량** | 최선 (GC 오버헤드 없음) | 양호 (증분 비용) | 양호 (상각) |
| **지연 시간** | 예측 가능 | 예측 가능 | 일시 정지 (동시 방식 아니면) |
| **메모리 오버헤드** | 없음 | 객체당 카운트 | 메타데이터 + 복사 공간 |
| **안전성** | 불안전 | 안전 (사이클 제외) | 안전 |
| **프로그래머 노력** | 높음 | 낮음 | 낮음 |
| **단편화** | 할당기에 따라 다름 | 있음 | 컴팩트/복사로 해결 |

### 1.5 보수적(Conservative) vs 정밀(Precise) GC

**정밀(exact) GC**는 스택과 객체에서 어떤 값이 포인터인지 정확히 알고 있습니다. 이는 컴파일러의 협력(스택 맵, 타입 정보)이 필요합니다.

**보수적 GC**는 스택의 유효한 힙 포인터처럼 보이는 모든 워드(word)를 잠재적 포인터로 취급합니다. 구현이 더 단순하지만 가비지를 보유할 수 있고(거짓 포인터) 객체를 이동할 수 없습니다(포인터가 아닌 값을 "수정"할 수 있으므로).

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

## 2. 참조 계수

### 2.1 기본 참조 계수

자동 메모리 관리의 가장 단순한 형태: 각 객체는 몇 개의 참조가 자신을 가리키는지 카운트를 유지합니다. 카운트가 0이 되면 객체는 즉시 해제됩니다.

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

출력:

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

### 2.2 장점과 단점

**장점**:
- **즉시 회수**: 객체가 가비지가 되는 즉시 해제됨
- **증분**: 비용이 프로그램 실행에 걸쳐 분산됨 (일시 정지 없음)
- **지역성**: 해제된 메모리가 최근에 사용됨 (캐시에 있을 가능성)
- **단순성**: 이해하고 구현하기 쉬움

**단점**:
- **사이클**: 순환 가비지를 수집할 수 없음 (3절 참조)
- **공간 오버헤드**: 모든 객체에 참조 카운트 필드 필요
- **시간 오버헤드**: 모든 포인터 할당에 inc/dec 연산 필요
- **스레드 안전성**: 동시 프로그램에서 참조 카운트 업데이트에 원자 연산 필요
- **캐시 비친화적**: 참조 카운트 업데이트가 관련 없는 객체의 캐시 라인을 더럽힘

### 2.3 지연 참조 계수(Deferred Reference Counting)

매우 자주 변경되는 스택 참조 추적의 오버헤드를 줄이기 위해, **지연 참조 계수**는 힙-힙 참조만 추적합니다. 스택 참조는 주기적으로 스택을 스캔하여 처리합니다.

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

### 2.4 병합 참조 계수(Coalesced Reference Counting)

**병합 참조 계수**는 참조 카운트 업데이트를 버퍼에 담아 적용 전에 합칩니다. 컬렉션 사이에 포인터 필드가 여러 번 쓰이면 순 효과만 기록됩니다.

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

## 3. 사이클 감지

### 3.1 사이클 문제

참조 계수는 사이클을 수집할 수 없습니다: 서로를 참조하지만 루트에서 도달할 수 없는 객체들입니다.

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

### 3.2 시험적 삭제(Trial Deletion) - 재활용기 알고리즘

시험적 삭제 알고리즘(CPython에서 사용)은 내부 참조를 임시로 제거하여 사이클을 식별합니다:

1. 후보 집합의 각 객체에 대해, 해당 객체가 참조하는 모든 객체의 참조 카운트를 임시로 감소시킵니다.
2. 임시 카운트가 0으로 떨어지는 객체는 가비지입니다.
3. 외부 참조에서 여전히 도달 가능한 객체는 살아남습니다.

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

## 4. 마크-스윕 컬렉션

### 4.1 알고리즘

마크-스윕은 가장 기본적인 추적 컬렉터입니다. 두 단계로 작동합니다:

1. **마크 단계**: 루트 집합에서 시작하여 도달 가능한 모든 객체를 순회하고 표시합니다.
2. **스윕 단계**: 전체 힙을 스캔하고, 표시되지 않은 객체는 가비지이므로 해제합니다.

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

### 4.2 자유 리스트 관리(Free List Management)

스윕 후, 해제된 메모리 블록은 **자유 리스트(free list)**에 배치됩니다. 미래의 할당은 이 리스트에서 적합한 블록을 검색합니다.

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

### 4.3 마크-스윕 복잡도

- **시간**: $O(L + H)$ (마크 단계의 라이브 객체 $L$, 스윕 단계의 힙 크기 $H$). 스윕은 전체 힙을 스캔해야 합니다.
- **공간**: 객체당 마크 비트 1개 (객체 헤더에 저장 가능).
- **일시 정지 시간**: 힙 크기에 비례 (마크와 스윕 단계 모두 프로그램 재개 전에 완료되어야 함).

---

## 5. 마크-컴팩트 컬렉션

### 5.1 동기

마크-스윕은 힙을 단편화된 상태로 남깁니다: 라이브 객체들이 자유 블록들 사이에 흩어져 있습니다. **마크-컴팩트**는 라이브 객체들을 모두 힙의 한쪽 끝으로 밀어넣어 단편화를 제거합니다.

```
Before compaction:
  [A][FREE][B][FREE][FREE][C][D][FREE][E][FREE]

After compaction:
  [A][B][C][D][E][FREE FREE FREE FREE FREE...]
```

### 5.2 알고리즘

마크-컴팩트는 세 번(또는 네 번)의 패스로 작동합니다:

1. **마크**: 마크-스윕과 동일 -- 도달 가능한 모든 객체를 표시합니다.
2. **전달 주소 계산**: 힙을 스캔하여 각 라이브 객체의 새 주소(이동할 위치)를 할당합니다.
3. **참조 업데이트**: 모든 객체를 스캔하여 각 포인터를 전달 주소로 교체합니다.
4. **컴팩트**: 각 객체를 전달 주소로 이동합니다.

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

### 5.3 트레이드오프

| 측면 | 마크-스윕 | 마크-컴팩트 |
|------|-----------|------------|
| 단편화 | 있음 | 없음 |
| 할당 속도 | 느림 (자유 리스트 검색) | 빠름 (범프 포인터) |
| 컬렉션 패스 | 2 | 3-4 |
| 객체 이동 | 없음 | 있음 (모든 포인터 업데이트 필요) |
| 캐시 지역성 | 나쁨 (분산) | 좋음 (컴팩트) |

---

## 6. 복사 컬렉터

### 6.1 세미-스페이스(Semi-Space) 컬렉터

**세미-스페이스** 복사 컬렉터는 힙을 두 절반으로 나눕니다: **from-space**와 **to-space**. 할당은 범프 포인터를 사용하여 from-space에서 일어납니다. 컬렉션 중에 라이브 객체는 to-space로 복사되고, 공간이 교환됩니다.

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

### 6.2 체니(Cheney)의 알고리즘

**체니 알고리즘**(1970)의 핵심 통찰은 너비 우선 복사가 to-space 자체를 작업 큐로 사용하여 **추가 메모리 없이** 수행될 수 있다는 것입니다:

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

### 6.3 복사 컬렉션의 트레이드오프

**장점**:
- 빠른 할당 (범프 포인터 -- 카운터만 증가)
- 단편화 없음 (복사 중에 객체가 컴팩트됨)
- 컬렉션 시간이 라이브 데이터에만 비례 (죽은 객체는 단순히 무시됨)
- 캐시 지역성 향상 (순회 순서로 복사됨)

**단점**:
- 힙의 절반 낭비 (한 번에 한 세미-스페이스만 사용 가능)
- 모든 라이브 객체를 매 컬렉션마다 복사해야 함 (장기 데이터가 많으면 비쌈)
- 이동된 객체로의 모든 포인터를 업데이트해야 함

**핵심 복잡도**: 컬렉션 시간은 $O(L)$ ($L$은 라이브 데이터 크기), 마크-스윕의 $O(L + H)$($H$ = 전체 힙 크기)와 비교됩니다.

---

## 7. 세대별 컬렉션

### 7.1 세대별 가설

**세대별 가설**(유아 사망률 또는 약한 세대별 가설이라고도 함)은 다음을 주장합니다:

> 대부분의 객체는 젊어서 죽는다.

많은 언어에 걸친 실증적 연구는 할당의 대부분이 매우 빨리 가비지가 된다는 것을 일관되게 보여줍니다. 이 분포는 일반적으로 "욕조 곡선"을 따릅니다:

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

### 7.2 세대별 GC 설계

이 가설을 기반으로, 세대별 GC는 힙을 세대로 나눕니다:

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

- 새 객체는 **Young 세대(nursery)**에 할당됩니다.
- 너서리는 자주 수집됩니다 (마이너 GC) -- 대부분의 객체가 죽어있음.
- 몇 번의 너서리 컬렉션에서 살아남은 객체는 **승진(tenured)**되어 Old 세대로 이동합니다.
- Old 세대는 드물게 수집됩니다 (메이저 GC).

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

### 7.3 쓰기 배리어(Write Barriers)

세대별 GC의 핵심 과제: Old 세대 객체가 너서리 객체를 가리키면 어떻게 될까요? 마이너 GC 중에는 너서리만 스캔합니다 -- 이 참조를 놓칠 수 있습니다!

**쓰기 배리어(write barrier)**는 모든 포인터 저장을 가로채어 이 문제를 해결합니다. Old 객체가 Young 객체를 가리키도록 수정되면, Old 객체를 **기억 집합(remembered set)**(카드 테이블이라고도 함)에 추가합니다.

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

### 7.4 다중 세대 방식

많은 프로덕션 컬렉터는 두 세대 이상을 사용합니다:

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

## 8. 증분 및 동시 GC

### 8.1 일시 정지 문제

STW(Stop-the-World) GC는 컬렉션 중에 모든 애플리케이션 스레드(**뮤테이터** 스레드)를 멈춥니다. 인터랙티브 또는 실시간 애플리케이션에서 이러한 일시 정지는 문제가 됩니다.

| 애플리케이션 | 허용 가능한 일시 정지 |
|-------------|---------------------|
| 배치 처리 | 수 초 |
| 웹 서버 | 100ms |
| 인터랙티브 UI | 16ms (60 FPS) |
| 트레이딩 시스템 | < 1ms |
| 하드 실시간 | 절대 불가 |

### 8.2 증분 컬렉션(Incremental Collection)

**증분 GC**는 컬렉션을 작은 단계로 나누어 뮤테이터 실행과 교대로 수행합니다:

```
Stop-the-world:
  Mutator ████████████░░░░░░░░████████████
  GC                  ████████

Incremental:
  Mutator ██████░██████░██████░██████░█████
  GC            ░      ░      ░      ░
              (small increments)
```

과제: 컬렉터가 작동하는 동안 뮤테이터가 객체 그래프를 수정할 수 있습니다.

### 8.3 동시 컬렉션(Concurrent Collection)

**동시 GC**는 뮤테이터와 동시에 별도 스레드에서 컬렉터를 실행합니다:

```
Concurrent:
  Mutator ████████████████████████████████████
  GC      ░░░░░░░░░░░░░░░░░░░░░░░░
          (runs on separate thread)
```

이는 동시 수정을 처리하기 위한 신중한 동기화가 필요합니다.

---

## 9. 삼색 마킹

### 9.1 삼색 추상화

삼색 마킹(tri-color marking)은 증분 및 동시 GC에 대한 추론을 위한 프레임워크를 제공합니다. 객체는 세 가지 색으로 분류됩니다:

- **흰색(White)**: 아직 방문하지 않음 (잠재적 가비지).
- **회색(Grey)**: 방문했지만 참조가 아직 스캔되지 않음.
- **검은색(Black)**: 방문했고 모든 참조가 스캔됨.

```
불변 조건: 검은색 객체가 흰색 객체를 직접 가리키지 않음.
           (검은색 객체의 모든 참조는 검은색 또는 회색 객체로 향함.)
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

### 9.2 잃어버린 객체 문제

뮤테이터가 증분 마킹 중에 객체 그래프를 수정하면, 라이브 객체가 누락되어 가비지로 수집될 수 있습니다. 이는 다음 경우에 발생합니다:

1. 뮤테이터가 검은색 객체에서 흰색 객체로의 참조를 저장합니다.
2. 뮤테이터가 그 흰색 객체로의 모든 회색-흰색 경로를 삭제합니다.

이제 흰색 객체는 도달 가능하지만 회색 선행자가 없어 수집될 것입니다!

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

### 9.3 쓰기 배리어 해결책

잃어버린 객체 문제를 방지하는 두 가지 접근 방식:

**Dijkstra의 쓰기 배리어** (처음부터 스냅샷): 뮤테이터가 흰색 객체에 대한 참조를 저장할 때, 대상을 회색으로 만듭니다.

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

**Steele의 쓰기 배리어** (증분 업데이트): 검은색 객체가 수정될 때, 소스를 회색으로 만듭니다 (재스캔).

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

### 9.4 Yuasa의 처음부터 스냅샷(Snapshot-at-the-Beginning)

Yuasa의 삭제 배리어는 흰색 객체가 마지막 회색 선행자를 잃지 않도록 보장합니다. 참조를 삭제할 때, 이전 대상이 흰색이면 회색으로 만듭니다:

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

## 10. 실제 시스템의 GC

### 10.1 JVM 가비지 컬렉터

JVM은 여러 GC 구현을 제공합니다:

| 컬렉터 | 유형 | 세대 | 일시 정지 목표 | 최적 사용처 |
|--------|------|------|--------------|------------|
| **Serial GC** | STW | Young + Old | 없음 | 소형 힙, 단일 코어 |
| **Parallel GC** | STW, 병렬 | Young + Old | 처리량 | 배치, 백그라운드 |
| **CMS** (deprecated) | 동시 마크-스윕 | Young + Old | 낮은 지연 | JDK 14에서 deprecated |
| **G1** | 리전 기반, 동시 | Young + Old | 설정 가능 | 범용 (기본값) |
| **ZGC** | 동시, 리전 기반 | 세대 없음* | < 10ms | 대형 힙 (TB 규모) |
| **Shenandoah** | 동시, 컴팩트 | 세대 없음* | < 10ms | 저지연 |

**G1(Garbage First) 컬렉터**:

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

**ZGC** 핵심 기법:
- **컬러 포인터(Colored pointers)**: 64비트 포인터의 사용하지 않는 비트를 GC 메타데이터로 사용
- **로드 배리어(Load barriers)**: 모든 로드 시 포인터를 확인하고 수정 (저장이 아닌)
- **동시 재배치(Concurrent relocation)**: 뮤테이터 실행 중에 객체 이동
- 힙 크기에 관계없이 10ms 미만의 일시 정지 시간

### 10.2 Go 가비지 컬렉터

Go는 **동시, 삼색 마크-스윕** 컬렉터를 사용합니다:

- 비세대적 (모든 객체가 하나의 공간에)
- Dijkstra의 쓰기 배리어를 사용한 동시 마킹
- 루트 스캔에만 STW 일시 정지 (일반적으로 < 1ms)
- 목표: 처리량 최대화가 아닌 지연 최소화
- 페이싱: 힙이 라이브 세트의 2배로 커질 때 GC 시작

```
Go GC Phases:
1. STW: Scan stacks, enable write barrier  (~100μs)
2. Concurrent: Mark phase (mutators run, write barrier active)
3. STW: Rescan, disable write barrier      (~100μs)
4. Concurrent: Sweep phase
```

### 10.3 CPython 가비지 컬렉터

CPython은 하이브리드 접근 방식을 사용합니다:

- **기본**: 참조 계수 (즉시 회수)
- **보조**: 세대별 사이클 감지기 (세 세대)

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

### 10.4 Rust: GC 대신 소유권(Ownership)

Rust는 근본적으로 다른 접근 방식을 취합니다: **가비지 컬렉터가 전혀 없습니다**. 대신 컴파일 타임 소유권 시스템을 사용합니다:

```
Rust Ownership Rules:
1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped (freed)
3. References can borrow values (immutably or mutably, but not both)
4. The borrow checker enforces these rules at compile time
```

이는 GC 일시 정지를 완전히 제거하지만 프로그래머가 소유권에 대해 생각해야 합니다. 소유권이 불분명한 경우를 위해 Rust는 다음을 제공합니다:

- `Rc<T>`: 참조 계수 (단일 스레드)
- `Arc<T>`: 원자적 참조 계수 (다중 스레드)
- 이들은 사이클을 만들 수 있으며, `Weak<T>` 참조로 처리

### 10.5 비교

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

## 11. GC 튜닝과 메트릭

### 11.1 핵심 메트릭

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| **처리량** | 애플리케이션 코드에 소비되는 시간 비율 (GC 제외) | 최대화 (>95%) |
| **일시 정지 시간** | GC로 인한 애플리케이션 일시 정지 지속 시간 | 최소화 |
| **풋프린트** | 사용되는 총 메모리 (힙 + GC 메타데이터) | 최소화 |
| **즉시성** | 객체 사망과 메모리 회수 사이의 시간 | 최소화 |
| **할당 속도** | 초당 할당되는 바이트 | 직접 제어 불가 |
| **승진 속도** | Young 세대에서 Old 세대로 초당 승진되는 바이트 | 최소화 |

### 11.2 GC 튜닝 삼각형

모든 메트릭을 동시에 최적화할 수 없습니다:

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

- **높은 처리량 + 낮은 일시 정지** = 대형 힙 (높은 풋프린트)
- **낮은 일시 정지 + 작은 풋프린트** = 더 잦은 컬렉션 (낮은 처리량)
- **높은 처리량 + 작은 풋프린트** = 더 긴 일시 정지 (일괄 컬렉션)

### 11.3 GC 튜닝 예제

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

### 11.4 공통 GC 튜닝 전략

1. **힙 크기 증가**: GC 빈도를 줄이지만 일시 정지 시간이 늘어남.
2. **너서리 크기 조정**: 더 큰 너서리 = 더 적은 마이너 GC, 더 긴 일시 정지.
3. **승진 임계값 조정**: 더 높은 임계값은 객체를 너서리에 더 오래 유지 (곧 죽을 경우 좋음).
4. **올바른 컬렉터 선택**: 컬렉터를 지연/처리량 요구사항에 맞춤.
5. **할당 속도 줄이기**: 가장 효과적인 전략 -- 단기 객체를 더 적게 할당.

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

## 12. 요약

가비지 컬렉션은 현대 프로그래밍 언어 런타임의 근본적인 측면입니다. 다음 접근 방식들을 다루었습니다:

| 알고리즘 | 회수 타이밍 | 사이클 처리 | 단편화 | 복사 비용 | 일시 정지 |
|---------|-----------|-----------|-------|---------|---------|
| **참조 계수** | 즉시 | 아니오 | 있음 | 없음 | 증분 |
| **마크-스윕** | 일괄 | 예 | 있음 | 없음 | STW |
| **마크-컴팩트** | 일괄 | 예 | 없음 | 슬라이드 | STW |
| **복사** | 일괄 | 예 | 없음 | 라이브 모두 복사 | STW |
| **세대별** | 혼합 | 예 | 다양 | Young만 | 짧은 마이너 |
| **동시** | 일괄 | 예 | 다양 | 다양 | 최소 |

핵심 원칙:

1. **세대별 가설**은 대부분의 현대 컬렉터 설계를 이끕니다: Young 객체는 자주, Old 객체는 드물게 수집.
2. **쓰기 배리어**는 세대별 및 동시 GC 모두에 필수적이며, 세대 간 또는 동시 수정을 추적합니다.
3. **삼색 마킹**은 동시 및 증분 컬렉션에 대한 추론을 위한 깔끔한 프레임워크를 제공합니다.
4. **모든 워크로드에 최선인 GC는 없습니다** -- 올바른 선택은 지연, 처리량, 풋프린트 요구사항에 따라 다릅니다.
5. **Rust의 소유권 모델**은 올바른 타입 시스템으로 GC를 완전히 피할 수 있음을 보여주지만, 프로그래밍 복잡성이 증가합니다.

---

## 13. 연습 문제

### 연습 1: 참조 계수 시뮬레이션

다음을 지원하는 참조 계수 시스템을 구현하세요:
- 객체 할당 및 해제
- 자동 참조 카운트 업데이트가 있는 포인터 할당
- 객체가 가비지가 될 때 감지

다음 시나리오로 테스트하세요:
```
A -> B -> C
A -> D
Remove A -> B  (should free B and C)
Remove A -> D  (should free D)
```

### 연습 2: 마크-스윕 구현

마크-스윕 구현을 다음을 처리하도록 확장하세요:
(a) 다중 루트 집합 (스택 루트, 글로벌 루트, 레지스터 루트).
(b) 첫 번째 적합(first-fit), 최선 적합(best-fit), 최악 적합(worst-fit) 전략을 가진 자유 리스트 할당기. 각 전략에서 단편화를 비교.
(c) 총 일시 정지 시간, 해제된 객체, 단편화 비율을 측정하고 보고.

### 연습 3: 복사 컬렉터

체니 알고리즘을 사용한 완전한 세미-스페이스 복사 컬렉터를 구현하세요:
(a) 힙으로 평탄 바이트 배열 사용 (Python 객체 아님).
(b) 이동된 객체의 전달 포인터 구현.
(c) 라이브 세트가 힙 절반을 초과하는 경우 처리 (메모리 부족 에러 발생).
(d) 할당 속도(범프 포인터) vs 자유 리스트 할당을 측정.

### 연습 4: 세대별 GC

세대별 GC를 다음으로 확장하세요:
(a) 세 세대 (너서리, 중간, 노인).
(b) 카드 테이블 쓰기 배리어 구현.
(c) 시간 경과에 따른 승진 속도를 추적하고 보고.
(d) 실험: 승진 임계값을 너무 높게 설정하면 어떻게 될까요? 너무 낮으면?

### 연습 5: 삼색 마킹 안전성

다음 객체 그래프와 GC 상태를 고려하세요:

```
Objects: A(BLACK), B(GREY), C(WHITE), D(WHITE), E(BLACK)
References: A->{B}, B->{C,D}, E->{}
Roots: A, E
```

(a) 뮤테이터 간섭 없이 마킹이 올바르게 완료됨을 보이세요.
(b) 뮤테이터가 `A.ref = C; B.ref = null` (B->C 제거)을 수행한다고 가정하세요. 쓰기 배리어 없이 C가 잃어버려짐을 보이세요.
(c) Dijkstra의 배리어가 잃어버린 객체를 방지하는 방법을 보이세요.
(d) Yuasa의 배리어가 잃어버린 객체를 방지하는 방법을 보이세요.

### 연습 6: GC 비교

다양한 패턴으로 객체를 할당하고 시뮬레이션된 GC 전략에서 성능을 측정하는 벤치마크를 작성하세요:
(a) 단기 버스트: 백만 개의 작은 객체를 할당하고 모두 버리기.
(b) 장기: 10,000개의 객체를 할당하고 유지한 다음 임시 객체 백만 개 할당.
(c) 순환: 사이클을 형성하는 연결 리스트 생성.

총 시간과 최대 메모리 사용량 측면에서 참조 계수, 마크-스윕, 복사, 세대별 컬렉터를 비교하세요.

---

## 14. 참고 자료

1. Jones, R., Hosking, A., & Moss, E. (2012). *The Garbage Collection Handbook: The Art of Automatic Memory Management*. CRC Press.
2. Wilson, P. R. (1992). "Uniprocessor Garbage Collection Techniques." *International Workshop on Memory Management*, Springer.
3. Cheney, C. J. (1970). "A Nonrecursive List Compacting Algorithm." *Communications of the ACM*, 13(11).
4. Dijkstra, E. W., Lamport, L., Martin, A. J., Scholten, C. S., & Steffens, E. F. M. (1978). "On-the-fly Garbage Collection: An Exercise in Cooperation." *Communications of the ACM*, 21(11).
5. Lieberman, H., & Hewitt, C. (1983). "A Real-Time Garbage Collector Based on the Lifetimes of Objects." *Communications of the ACM*, 26(6).
6. Bacon, D. F., & Rajan, V. T. (2001). "Concurrent Cycle Collection in Reference Counted Systems." *ECOOP*.
7. Tene, G., Iyengar, B., & Wolf, M. (2011). "C4: The Continuously Concurrent Compacting Collector." *ISMM*.

---

[이전: 13. 루프 최적화](./13_Loop_Optimization.md) | [다음: 15. 인터프리터와 가상 머신](./15_Interpreters_and_Virtual_Machines.md) | [개요](./00_Overview.md)
