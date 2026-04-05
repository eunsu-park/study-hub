# 레지스터 할당(Register Allocation)

**이전**: [18. SSA 형식](./18_SSA_Form.md) | **다음**: [20. LLVM IR 입문](./20_LLVM_IR_Introduction.md)

---

레지스터 할당(register allocation)은 무한한 수의 가상 레지스터(또는 프로그램 변수)를 유한한 물리적 기계 레지스터 집합에 매핑하는 과정입니다. 이것은 컴파일의 가장 중요한 단계 중 하나입니다 -- 잘못된 레지스터 할당은 메모리 스필(spill)로 인해 실행 시간을 수 배로 증가시킬 수 있습니다.

이 레슨에서는 두 가지 주요 접근법인 그래프 색칠(graph coloring)과 선형 스캔(linear scan), 그리고 생존 범위 분할(live range splitting), 스필 전략, 레지스터 합병(register coalescing) 같은 실용적인 관심사를 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [11. 코드 생성](./11_Code_Generation.md), [18. SSA 형식](./18_SSA_Form.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 레지스터 할당이 일반적으로 NP-완전(NP-complete)인 이유를 설명한다
2. 생존 범위(live range)에서 간섭 그래프(interference graph)를 구축한다
3. 그래프 색칠 레지스터 할당(Chaitin-Briggs)을 구현한다
4. 선형 스캔 레지스터 할당을 구현한다
5. 생존 범위 분할과 스필 전략을 적용한다
6. 레지스터 합병과 색칠의 상호작용을 이해한다

---

## 목차

1. [레지스터 할당 문제](#1-레지스터-할당-문제)
2. [생존 분석](#2-생존-분석)
3. [간섭 그래프](#3-간섭-그래프)
4. [그래프 색칠 할당](#4-그래프-색칠-할당)
5. [스필링](#5-스필링)
6. [합병](#6-합병)
7. [선형 스캔 할당](#7-선형-스캔-할당)
8. [SSA 기반 레지스터 할당](#8-ssa-기반-레지스터-할당)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 레지스터 할당 문제

### 1.1 레지스터 할당이 중요한 이유

현대 프로세서는 16-32개의 범용 레지스터(general-purpose register)를 가지고 있습니다(x86-64는 16개, ARM64는 31개). 일반적인 함수는 수백 개의 가상 레지스터를 사용할 수 있습니다. 할당기는 다음을 수행해야 합니다:

- 가상 레지스터를 물리적 레지스터에 매핑
- 동시에 살아있는 두 값이 레지스터를 공유하지 않도록 보장
- 메모리 접근(스필과 리로드)을 최소화
- 아키텍처 제약 존중(예: 특정 명령어가 특정 레지스터를 요구)

```
Virtual registers: v1, v2, v3, v4, v5, v6, ...
Physical registers: r0, r1, r2 (only 3 available)

Challenge: Map many virtual registers to few physical registers
           without breaking the program semantics.
```

### 1.2 복잡도

레지스터 할당은 그래프 색칠(graph coloring)과 동치이며, 일반 그래프에 대해 NP-완전(NP-complete)입니다. 그러나 실제 프로그램은 특수한 구조를 가진 간섭 그래프를 생성하며, 실무에서는 휴리스틱(heuristic)이 잘 동작합니다.

### 1.3 생존 범위(Live Ranges)

변수의 **생존 범위(live range)**는 변수가 나중에 사용될 수 있는 값을 보유하고 있는 프로그램 지점(program point)의 집합입니다:

```
1: v1 = 10        # v1 live range starts
2: v2 = 20        # v2 live range starts
3: v3 = v1 + v2   # v1, v2 used; v3 starts
4: v4 = v3 * 2    # v3 used; v4 starts
5: print(v4)      # v4 used
```

생존 범위:
- v1: [1, 3]
- v2: [2, 3]
- v3: [3, 4]
- v4: [4, 5]

최대 동시 생존 = 2(지점 3에서: v1과 v2), 따라서 2개의 레지스터면 충분합니다.

---

## 2. 생존 분석(Liveness Analysis)

### 2.1 데이터 흐름 방정식

생존(liveness)은 역방향 데이터 흐름 분석(backward data-flow analysis)입니다:

```
LiveOut(B) = Union of LiveIn(S) for all successors S of B
LiveIn(B)  = Use(B) ∪ (LiveOut(B) - Def(B))
```

여기서:
- `Use(B)` = B에서 정의되기 전에 사용된 변수
- `Def(B)` = B에서 정의된 변수

### 2.2 구현

```python
def liveness_analysis(cfg):
    """
    Compute live-in and live-out sets for each block.
    Returns: (live_in, live_out) dicts mapping block -> set of variables
    """
    live_in = {b: set() for b in cfg.blocks}
    live_out = {b: set() for b in cfg.blocks}

    changed = True
    while changed:
        changed = False
        for block in reversed(cfg.blocks):  # reverse order for efficiency
            # LiveOut = union of LiveIn of successors
            new_out = set()
            for succ in block.successors:
                new_out |= live_in[succ]

            # LiveIn = Use ∪ (LiveOut - Def)
            new_in = block.use_set | (new_out - block.def_set)

            if new_in != live_in[block] or new_out != live_out[block]:
                live_in[block] = new_in
                live_out[block] = new_out
                changed = True

    return live_in, live_out


def compute_live_intervals(cfg, live_in, live_out):
    """
    Compute live intervals (start, end) for each variable.
    Uses instruction numbering across all blocks.
    """
    intervals = {}  # var -> (start, end)

    for block in cfg.blocks:
        live = set(live_out[block])

        # Walk backward through instructions
        for instr in reversed(block.instructions):
            pos = instr.position

            # Definition: end of live range (or start if first seen)
            if instr.dest:
                if instr.dest in live:
                    live.discard(instr.dest)
                if instr.dest not in intervals:
                    intervals[instr.dest] = [pos, pos]
                else:
                    intervals[instr.dest][0] = min(intervals[instr.dest][0], pos)

            # Uses: extend live range
            for use in instr.uses:
                live.add(use)
                if use not in intervals:
                    intervals[use] = [pos, pos]
                else:
                    intervals[use][0] = min(intervals[use][0], pos)
                    intervals[use][1] = max(intervals[use][1], pos)

    return intervals
```

---

## 3. 간섭 그래프(Interference Graphs)

### 3.1 정의

**간섭 그래프(interference graph)** G = (V, E)에서:
- 각 노드는 가상 레지스터(또는 생존 범위)를 나타냄
- 두 변수가 어떤 프로그램 지점에서 동시에 살아있으면 간선 (u, v)가 존재

간섭하는 두 변수는 레지스터를 공유할 수 없습니다.

### 3.2 간섭 그래프 구축

```python
def build_interference_graph(cfg, live_out):
    """
    Build interference graph by walking each block backward.
    """
    graph = InterferenceGraph()

    for block in cfg.blocks:
        live = set(live_out[block])

        for instr in reversed(block.instructions):
            if instr.dest:
                # The defined variable interferes with everything live
                # (except itself and, for copy instructions, the source)
                for var in live:
                    if var != instr.dest:
                        if instr.is_copy and var == instr.source:
                            continue  # Don't add interference for copies
                        graph.add_edge(instr.dest, var)
                live.discard(instr.dest)

            for use in instr.uses:
                live.add(use)

    return graph


class InterferenceGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.adj = {}  # node -> set of neighbors

    def add_edge(self, u, v):
        self.nodes.add(u)
        self.nodes.add(v)
        if u != v:
            self.edges.add((min(u, v), max(u, v)))
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)

    def degree(self, node):
        return len(self.adj.get(node, set()))

    def remove_node(self, node):
        for neighbor in list(self.adj.get(node, set())):
            self.adj[neighbor].discard(node)
        del self.adj[node]
        self.nodes.discard(node)
```

### 3.3 예제

```
Instructions:        Live sets:
1: a = ...           {a}
2: b = ...           {a, b}
3: c = a + b         {c}         (a,b dead after use)
4: d = ...           {c, d}
5: e = c + d         {e}
6: return e          {}

Interference graph:
  a --- b     (both live at point 2)
  c --- d     (both live at point 4)

  (a, c, d, e have no mutual interference except c-d)

Chromatic number = 2 (two registers suffice)
```

---

## 4. 그래프 색칠 할당(Graph Coloring Allocation)

### 4.1 Chaitin 알고리즘(1981)

기본 아이디어: 간섭 그래프를 k-색칠(k-color)합니다. 여기서 k는 물리적 레지스터의 수입니다.

```python
def chaitin_allocate(graph, k):
    """
    Chaitin's register allocation via graph coloring.
    k: number of available physical registers
    """
    stack = []

    # Simplify: iteratively remove nodes with degree < k
    work_graph = graph.copy()
    while work_graph.nodes:
        # Find a node with degree < k
        low_degree = None
        for node in work_graph.nodes:
            if work_graph.degree(node) < k:
                low_degree = node
                break

        if low_degree:
            stack.append(low_degree)
            work_graph.remove_node(low_degree)
        else:
            # No low-degree node: must spill
            victim = select_spill(work_graph)
            stack.append(('spill', victim))
            work_graph.remove_node(victim)

    # Select: pop nodes and assign colors
    coloring = {}
    for item in reversed(stack):
        if isinstance(item, tuple) and item[0] == 'spill':
            node = item[1]
            # Try to color; if impossible, actually spill
            used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
            available = set(range(k)) - used
            if available:
                coloring[node] = min(available)  # Optimistic: might color after all
            else:
                coloring[node] = 'SPILL'
        else:
            node = item
            used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
            available = set(range(k)) - used
            coloring[node] = min(available)

    return coloring
```

### 4.2 Briggs의 개선: 낙관적 색칠(Optimistic Coloring)

Briggs(1994)는 스필 후보에 대해 **낙관적**으로 접근하여 Chaitin 알고리즘을 개선했습니다:

```python
def briggs_allocate(graph, k):
    """
    Briggs's optimistic coloring.
    Difference from Chaitin: push potential spills onto stack
    instead of immediately spilling. They might be colorable
    when popped (neighbors may have been removed).
    """
    stack = []
    work_graph = graph.copy()

    while work_graph.nodes:
        # Phase 1: Remove nodes with degree < k
        found = True
        while found:
            found = False
            for node in list(work_graph.nodes):
                if work_graph.degree(node) < k:
                    stack.append(node)
                    work_graph.remove_node(node)
                    found = True

        if work_graph.nodes:
            # Phase 2: Optimistically push a potential spill
            victim = select_spill(work_graph)
            stack.append(victim)  # Not marked as spill yet!
            work_graph.remove_node(victim)

    # Select phase: try to color everything
    coloring = {}
    for node in reversed(stack):
        used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
        available = set(range(k)) - used
        if available:
            coloring[node] = min(available)
        else:
            coloring[node] = 'SPILL'  # Only spill if truly uncolorable

    return coloring
```

### 4.3 스필 비용 휴리스틱(Spill Cost Heuristics)

```python
def select_spill(graph):
    """
    Choose which variable to spill.
    Heuristic: minimize (use_count / degree) -- spill the variable
    with the most interference but fewest uses.
    """
    best = None
    best_score = float('inf')

    for node in graph.nodes:
        if node.is_precolored:
            continue  # Never spill physical registers
        score = node.use_count / max(graph.degree(node), 1)
        if score < best_score:
            best_score = score
            best = node

    return best
```

---

## 5. 스필링(Spilling)

### 5.1 스필링이란?

변수에 레지스터를 할당할 수 없을 때, 메모리의 스택 슬롯(stack slot)으로 **스필(spill)**됩니다. 할당기는 다음을 삽입합니다:
- 각 정의 후에 **저장(store)** 명령어(스택에 기록)
- 각 사용 전에 **로드(load)** 명령어(스택에서 읽기)

```
# Before spilling (v3 spilled):
v3 = v1 + v2
v4 = v3 * 2

# After spilling:
v3 = v1 + v2
store v3, [sp + offset]     # spill store
v3_reload = load [sp + offset]  # spill load
v4 = v3_reload * 2
```

### 5.2 전체 스필 vs 부분 스필(Spill Everywhere vs. Spill Around)

**전체 스필(Spill Everywhere)**: 모든 사용 전에 로드를, 모든 정의 후에 저장을 삽입합니다. 단순하지만 많은 메모리 연산을 생성합니다.

**부분 스필(Spill Around)**: 레지스터 압력(register pressure)이 높은 영역에서만 스필합니다. 가능한 곳에서는 레지스터에 값을 유지합니다.

### 5.3 재구체화(Rematerialization)

스필된 값을 메모리에서 로드하는 대신, 비용이 더 적은 경우 **재구체화(rematerialization)**로 다시 계산합니다:

```python
def can_rematerialize(instr):
    """
    A value can be rematerialized if:
    - It's a constant load
    - It's a simple computation with available operands
    - Recomputing is cheaper than a memory load
    """
    if instr.is_constant_load:
        return True
    if instr.is_address_computation and all_operands_available(instr):
        return True
    return False
```

---

## 6. 합병(Coalescing)

### 6.1 복사 합병(Copy Coalescing)

많은 프로그램은 복사 명령어(`a = b`)를 포함합니다. `a`와 `b`가 간섭하지 않으면, 같은 레지스터를 할당하고 복사를 제거할 수 있습니다:

```python
def coalesce(graph, copies):
    """
    Aggressive coalescing: merge copy-related variables
    that don't interfere.
    """
    for src, dst in copies:
        if not graph.has_edge(src, dst):
            # Safe to coalesce: merge src into dst
            graph.merge_nodes(src, dst)
            # All references to src become dst
```

### 6.2 보수적 합병: Briggs 기준(Briggs's Criterion)

공격적 합병(aggressive coalescing)은 병합된 노드의 차수(degree)를 증가시켜 색칠 불가능하게 만들 수 있습니다. Briggs 기준: 병합된 노드의 차수 >= k인 이웃 수가 k 미만일 때만 합병합니다.

```python
def briggs_coalesce(graph, src, dst, k):
    """
    Coalesce src and dst only if the result has fewer than k
    high-degree neighbors.
    """
    # Compute neighbors of merged node
    merged_neighbors = (graph.adj.get(src, set()) | graph.adj.get(dst, set())) - {src, dst}

    high_degree_count = sum(1 for n in merged_neighbors if graph.degree(n) >= k)

    if high_degree_count < k:
        graph.merge_nodes(src, dst)
        return True
    return False
```

### 6.3 George 기준(George's Criterion)

대안적 방법: a의 모든 이웃 t가 b와 간섭하거나 차수 < k이면 a와 b를 합병합니다.

```python
def george_coalesce(graph, a, b, k):
    """George's coalescing criterion."""
    for t in graph.adj.get(a, set()):
        if t == b:
            continue
        if not graph.has_edge(t, b) and graph.degree(t) >= k:
            return False  # Unsafe
    graph.merge_nodes(a, b)
    return True
```

---

## 7. 선형 스캔 할당(Linear Scan Allocation)

### 7.1 동기

그래프 색칠은 효과적이지만 빠른 컴파일이 필요한 JIT 컴파일러에는 느립니다. **선형 스캔(linear scan)**(Poletto와 Sarkar, 1999)은 O(n log n) 시간에 실행됩니다.

### 7.2 알고리즘

```python
def linear_scan_allocate(intervals, k):
    """
    Linear scan register allocation.
    intervals: list of (var, start, end) sorted by start
    k: number of physical registers
    """
    active = []       # currently active intervals, sorted by end
    free_regs = list(range(k))  # available registers
    allocation = {}   # var -> register or 'SPILL'
    spill_slots = {}

    for var, start, end in sorted(intervals, key=lambda x: x[1]):
        # Expire old intervals
        active_new = []
        for a_var, a_start, a_end, a_reg in active:
            if a_end <= start:
                free_regs.append(a_reg)  # Return register
            else:
                active_new.append((a_var, a_start, a_end, a_reg))
        active = active_new

        if free_regs:
            # Allocate a register
            reg = free_regs.pop(0)
            allocation[var] = reg
            active.append((var, start, end, reg))
            active.sort(key=lambda x: x[2])  # Sort by end point
        else:
            # Spill: evict the interval with the farthest end point
            if active and active[-1][2] > end:
                # Spill the active interval that ends latest
                spill = active.pop()
                spill_slots[spill[0]] = allocate_stack_slot()
                allocation[spill[0]] = 'SPILL'
                allocation[var] = spill[3]  # Take its register
                active.append((var, start, end, spill[3]))
                active.sort(key=lambda x: x[2])
            else:
                # Spill current interval
                spill_slots[var] = allocate_stack_slot()
                allocation[var] = 'SPILL'

    return allocation, spill_slots
```

### 7.3 세컨드 찬스 할당(Second-Chance Allocation)

스필된 값이 생존 범위의 후반부에서 레지스터를 재할당받을 수 있는 확장으로, 간격(interval)을 분할합니다:

```python
def linear_scan_with_splitting(intervals, k):
    """
    Linear scan with live range splitting.
    When spilling, split the interval and try to allocate
    the remaining portion later.
    """
    # Sort by start point
    queue = sorted(intervals, key=lambda x: x[1])
    active = []
    allocation = {}

    while queue:
        var, start, end = queue.pop(0)
        expire_old(active, start)

        if len(active) < k:
            reg = assign_register(active)
            allocation[(var, start, end)] = reg
            active.append((var, start, end, reg))
        else:
            # Split: spill current interval at this point,
            # but re-enqueue the tail for later allocation
            split_point = find_optimal_split(start, end, active)
            if split_point < end:
                allocation[(var, start, split_point)] = 'SPILL'
                queue.append((var, split_point, end))
                queue.sort(key=lambda x: x[1])
            else:
                allocation[(var, start, end)] = 'SPILL'

    return allocation
```

---

## 8. SSA 기반 레지스터 할당

### 8.1 SSA가 할당을 단순화하는 이유

SSA 형식에서는 같은 변수에 대해 생존 범위가 겹치지 않습니다(각 변수가 한 번만 정의되므로). SSA 프로그램의 간섭 그래프는 항상 **현 그래프(chordal graph)**이며, 이는 다항 시간에 최적으로 색칠할 수 있음을 의미합니다.

### 8.2 할당 중 SSA 해체

현대 할당기는 SSA 형식에서 직접 동작합니다:

1. SSA 형식에서 간섭 그래프 구축(현 그래프)
2. 완전 제거 순서(perfect elimination ordering)를 사용하여 최적 색칠
3. 파이 함수에 대한 복사 삽입(SSA 해체)
4. 가능한 곳에서 복사 합병

```python
def ssa_register_allocate(ssa_program, k):
    """
    Register allocation on SSA form.
    The interference graph is chordal -> optimal coloring in O(V+E).
    """
    graph = build_ssa_interference_graph(ssa_program)

    # Chordal graph: compute perfect elimination ordering
    peo = maximum_cardinality_search(graph)

    # Color greedily in reverse PEO order
    coloring = {}
    for node in reversed(peo):
        used = {coloring[n] for n in graph.adj.get(node, set()) if n in coloring}
        for color in range(k):
            if color not in used:
                coloring[node] = color
                break
        else:
            coloring[node] = 'SPILL'

    return coloring


def maximum_cardinality_search(graph):
    """
    Compute perfect elimination ordering for chordal graph.
    Greedy: always pick the unvisited node with the most visited neighbors.
    """
    visited = set()
    order = []
    weight = {n: 0 for n in graph.nodes}

    for _ in range(len(graph.nodes)):
        # Pick node with maximum weight
        best = max((n for n in graph.nodes if n not in visited),
                   key=lambda n: weight[n])
        order.append(best)
        visited.add(best)
        for neighbor in graph.adj.get(best, set()):
            if neighbor not in visited:
                weight[neighbor] += 1

    return order
```

---

## 9. 요약

- **레지스터 할당**은 스필을 최소화하면서 가상 레지스터를 물리적 레지스터에 매핑합니다
- **생존 분석**은 각 프로그램 지점에서 어떤 변수가 살아있는지 결정합니다
- **간섭 그래프**는 레지스터를 공유할 수 없는 변수를 포착합니다
- **그래프 색칠**(Chaitin-Briggs)은 최적화 컴파일러의 표준 접근법입니다
- **선형 스캔**은 JIT 컴파일러를 위한 빠른 할당을 제공합니다
- **스필링**은 레지스터가 소진되었을 때 값을 메모리에 저장합니다
- **합병**은 간섭하지 않는 변수를 병합하여 복사 명령어를 제거합니다
- **SSA 기반 할당**은 현 그래프 속성을 활용하여 최적 색칠합니다

---

## 10. 연습 문제

1. **수동 생존 분석**: 5개 블록 CFG의 live-in과 live-out 집합을 계산하세요.

2. **간섭 그래프**: 주어진 생존 범위에서 간섭 그래프를 구축하고 색수(chromatic number)를 결정하세요.

3. **Chaitin-Briggs**: 낙관적 색칠 알고리즘을 구현하고 k=3인 작은 간섭 그래프에서 테스트하세요.

4. **선형 스캔**: 생존 간격 집합에 대해 선형 스캔 할당을 구현하고 그래프 색칠과 스필 수를 비교하세요.

5. **합병**: 복사 명령어가 있는 프로그램에 Briggs의 보수적 합병 기준을 적용하세요.

---

## 11. 참고 자료

1. Chaitin, G. J. (1982). "Register Allocation & Spilling via Graph Coloring." *ACM SIGPLAN Notices*, 17(6), 98-105.
2. Briggs, P., Cooper, K. D., Torczon, L. (1994). "Improvements to Graph Coloring Register Allocation." *ACM Transactions on Programming Languages and Systems*, 16(3), 428-455.
3. Poletto, M., Sarkar, V. (1999). "Linear Scan Register Allocation." *ACM Transactions on Programming Languages and Systems*, 21(5), 895-913.
4. Hack, S., Grund, D., Goos, G. (2006). "Register Allocation for Programs in SSA Form." *Compiler Construction (CC)*.
5. Wimmer, C., Franz, M. (2010). "Linear Scan Register Allocation on SSA Form." *CGO*.

---

**이전**: [18. SSA 형식](./18_SSA_Form.md) | **다음**: [20. LLVM IR 입문](./20_LLVM_IR_Introduction.md)
