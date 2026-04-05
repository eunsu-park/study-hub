# 루프 최적화

**이전**: [12. 최적화 -- 지역 최적화와 전역 최적화](./12_Optimization_Local_and_Global.md) | **다음**: [14. 가비지 컬렉션](./14_Garbage_Collection.md)

---

프로그램은 실행 시간의 대부분을 루프 내부에서 보냅니다. "90/10 규칙"이라고도 불리는 경험적 법칙에 따르면 실행 시간의 90%는 코드의 10%에서 소비되며 -- 그 10%는 거의 항상 루프 안입니다. 이 때문에 루프 최적화는 컴파일러 최적화 범주 중 가장 큰 영향을 미칩니다. 백만 번 실행되는 루프는 모든 개선을 백만 배로 증폭합니다.

이 레슨은 루프 최적화의 전체 스펙트럼을 다룹니다: 제어 흐름 그래프에서 루프를 감지하는 것부터, 루프 불변 코드 이동(loop-invariant code motion)과 강도 감소(strength reduction)와 같은 고전적인 변환, 벡터화(vectorization)와 다면체 모델(polyhedral model)과 같은 현대적인 기법까지.

**난이도**: ⭐⭐⭐⭐

**전제 조건**: [09. 중간 표현](./09_Intermediate_Representations.md), [12. 최적화 -- 지역 최적화와 전역 최적화](./12_Optimization_Local_and_Global.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 지배 분석(dominance analysis)과 후방 엣지(back edge)를 사용하여 자연 루프(natural loop) 식별
2. 지배자 트리(dominator tree) 구성 및 지배 경계(dominance frontier) 계산
3. 루프 불변 코드 이동(LICM)을 적용하여 계산을 루프 밖으로 끌어올리기
4. 강도 감소를 통한 귀납 변수(induction variable) 인식 및 최적화
5. 루프 언롤링(loop unrolling)과 그 트레이드오프 이해
6. 루프 변환 적용: 결합(fusion), 분리(fission), 교환(interchange), 타일링(tiling)
7. 벡터화의 기초와 다면체 모델 설명
8. 루프 병렬화 기회 추론

---

## 목차

1. [루프 최적화가 중요한 이유](#1-루프-최적화가-중요한-이유)
2. [지배자와 지배자 트리](#2-지배자와-지배자-트리)
3. [자연 루프 감지](#3-자연-루프-감지)
4. [루프 불변 코드 이동 (LICM)](#4-루프-불변-코드-이동-licm)
5. [귀납 변수 분석](#5-귀납-변수-분석)
6. [강도 감소](#6-강도-감소)
7. [루프 언롤링](#7-루프-언롤링)
8. [루프 결합과 분리](#8-루프-결합과-분리)
9. [루프 타일링 (블로킹)](#9-루프-타일링-블로킹)
10. [루프 교환](#10-루프-교환)
11. [벡터화](#11-벡터화)
12. [다면체 모델](#12-다면체-모델)
13. [루프 병렬화](#13-루프-병렬화)
14. [요약](#14-요약)
15. [연습 문제](#15-연습-문제)
16. [참고 문헌](#16-참고-문헌)

---

## 1. 루프 최적화가 중요한 이유

간단한 루프를 생각해봅시다:

```python
# Before optimization
total = 0
for i in range(1_000_000):
    x = a * b + c        # Invariant: same result every iteration
    total += arr[i] + x
```

`a * b + c` 표현식은 절대 변하지 않는데도 백만 번 재계산됩니다. 이를 루프 밖으로 이동하면:

```python
# After optimization
x = a * b + c            # Computed once
total = 0
for i in range(1_000_000):
    total += arr[i] + x
```

이로써 999,999번의 중복 곱셈과 덧셈이 제거됩니다. 이런 절감이 중첩 루프에 걸쳐 곱해지면 영향은 엄청납니다.

### 최적화 지형

루프 최적화는 여러 범주로 분류됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Loop Optimizations                            │
├─────────────────────┬──────────────────────┬────────────────────┤
│  Reduce Work        │  Improve Locality    │  Exploit Hardware  │
│                     │                      │                    │
│  - LICM             │  - Loop tiling       │  - Vectorization   │
│  - Strength         │  - Loop interchange  │  - Parallelization │
│    reduction        │  - Loop fusion       │  - Software        │
│  - Dead code in     │  - Data prefetching  │    pipelining      │
│    loops            │                      │                    │
│  - Loop unrolling   │                      │                    │
│  - Loop unswitching │                      │                    │
└─────────────────────┴──────────────────────┴────────────────────┘
```

루프를 최적화하기 전에 찾아야 합니다. 그러려면 지배자를 이해해야 합니다.

---

## 2. 지배자와 지배자 트리

### 2.1 지배(Dominance)

진입 노드 $\text{entry}$가 있는 제어 흐름 그래프(CFG)에서, $\text{entry}$에서 $n$까지의 모든 경로가 $d$를 통과해야 하면, 노드 $d$가 노드 $n$을 **지배(dominates)**한다고 합니다 ($d \;\text{dom}\; n$으로 표기).

주요 성질:

- **반사성(Reflexivity)**: 모든 노드는 자기 자신을 지배합니다: $n \;\text{dom}\; n$.
- **추이성(Transitivity)**: $a \;\text{dom}\; b$이고 $b \;\text{dom}\; c$이면 $a \;\text{dom}\; c$.
- **반대칭성(Antisymmetry)**: $a \;\text{dom}\; b$이고 $b \;\text{dom}\; a$이면 $a = b$.

$n$의 **직접 지배자(immediate dominator)** ($\text{idom}(n)$으로 표기)는 $n$의 가장 가까운 진성 지배자(strict dominator)입니다: $d \neq n$이면서 $n$의 다른 모든 지배자가 $d$를 지배하는 지배자 $d$.

### 2.2 지배자 계산

고전적인 반복 알고리즘은 모든 노드의 지배자 집합을 모든 노드를 포함하도록 초기화한 후, 교집합으로 반복적으로 정제합니다:

```python
def compute_dominators(cfg, entry):
    """
    Compute dominator sets for each node in a CFG.

    cfg: dict mapping node -> list of predecessor nodes
    entry: the entry node of the CFG
    Returns: dict mapping node -> set of dominator nodes
    """
    all_nodes = set(cfg.keys())
    dom = {}

    # Initialize
    dom[entry] = {entry}
    for node in all_nodes:
        if node != entry:
            dom[node] = set(all_nodes)  # Start with all nodes

    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        for node in all_nodes:
            if node == entry:
                continue

            predecessors = cfg[node]
            if not predecessors:
                continue

            # New dom set = intersection of all predecessors' dom sets, plus self
            new_dom = set.intersection(*(dom[p] for p in predecessors))
            new_dom.add(node)

            if new_dom != dom[node]:
                dom[node] = new_dom
                changed = True

    return dom
```

**예시**:

```
CFG:
    entry -> A -> B -> C -> D
                  ^         |
                  |         |
                  +----E----+

Predecessors:
  entry: []
  A: [entry]
  B: [A, E]
  C: [B]
  D: [C]
  E: [D]
```

```python
cfg = {
    'entry': [],
    'A': ['entry'],
    'B': ['A', 'E'],
    'C': ['B'],
    'D': ['C'],
    'E': ['D'],
}

dom = compute_dominators(cfg, 'entry')

# Result:
# dom['entry'] = {'entry'}
# dom['A']     = {'entry', 'A'}
# dom['B']     = {'entry', 'A', 'B'}
# dom['C']     = {'entry', 'A', 'B', 'C'}
# dom['D']     = {'entry', 'A', 'B', 'C', 'D'}
# dom['E']     = {'entry', 'A', 'B', 'C', 'D', 'E'}
```

### 2.3 지배자 트리

지배자 트리는 직접 지배 관계를 나타냅니다. 트리에서 각 노드의 부모는 직접 지배자입니다.

```python
def build_dominator_tree(dom, entry):
    """
    Build the dominator tree from dominator sets.

    Returns: dict mapping node -> immediate dominator
    """
    idom = {}
    for node in dom:
        if node == entry:
            continue
        # Immediate dominator is the closest strict dominator
        strict_doms = dom[node] - {node}
        # idom is the one that is dominated by all others
        # (i.e., has the largest dominator set among strict dominators)
        idom_node = max(strict_doms, key=lambda d: len(dom[d]))
        idom[node] = idom_node
    return idom


def print_dominator_tree(idom, entry, indent=0):
    """Print the dominator tree."""
    print("  " * indent + entry)
    children = [n for n, d in idom.items() if d == entry]
    for child in sorted(children):
        print_dominator_tree(idom, child, indent + 1)
```

위 예시에 대해:

```
Dominator Tree:
entry
  A
    B
      C
        D
          E
```

### 2.4 Cooper-Harvey-Kennedy 알고리즘

실제로 가장 널리 사용되는 알고리즘은 역후위 순서 순회를 사용하여 거의 선형 시간에 지배자를 계산하는 Cooper-Harvey-Kennedy(CHK) 알고리즘입니다:

```python
def compute_rpo(cfg_successors, entry):
    """Compute reverse postorder of CFG nodes."""
    visited = set()
    postorder = []

    def dfs(node):
        visited.add(node)
        for succ in cfg_successors.get(node, []):
            if succ not in visited:
                dfs(succ)
        postorder.append(node)

    dfs(entry)
    postorder.reverse()
    return postorder


def intersect(idom, rpo_number, b1, b2):
    """Find common dominator using the finger-moving technique."""
    finger1 = b1
    finger2 = b2
    while finger1 != finger2:
        while rpo_number[finger1] > rpo_number[finger2]:
            finger1 = idom[finger1]
        while rpo_number[finger2] > rpo_number[finger1]:
            finger2 = idom[finger2]
    return finger1


def compute_idom_chk(cfg_preds, cfg_succs, entry):
    """
    Cooper-Harvey-Kennedy dominator algorithm.

    cfg_preds: dict node -> list of predecessors
    cfg_succs: dict node -> list of successors
    entry: entry node
    Returns: dict node -> immediate dominator
    """
    rpo = compute_rpo(cfg_succs, entry)
    rpo_number = {node: i for i, node in enumerate(rpo)}

    idom = {entry: entry}

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry:
                continue
            preds = [p for p in cfg_preds[b] if p in idom]
            if not preds:
                continue

            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(idom, rpo_number, p, new_idom)

            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True

    return idom
```

### 2.5 지배 경계(Dominance Frontiers)

노드 $n$의 **지배 경계(dominance frontier)**는 $n$의 지배가 막 "멈추는" 노드들의 집합입니다 -- $n$이 엄격하게 지배하지 않지만 $n$이 지배하는 선행 블록을 가지는 첫 번째 노드들.

$$DF(n) = \{ y \mid \exists\, x \in \text{pred}(y) : n \;\text{dom}\; x \land n \not\text{sdom}\; y \}$$

여기서 $\text{sdom}$은 "엄격하게 지배" (지배하고 같지 않음)를 의미합니다.

지배 경계는 SSA 형식 구성 시 $\phi$-함수를 배치하는 데 중요합니다 (레슨 9 참조).

```python
def compute_dominance_frontiers(cfg_preds, idom, all_nodes):
    """
    Compute dominance frontiers for all nodes.

    Returns: dict mapping node -> set of nodes in its dominance frontier
    """
    df = {n: set() for n in all_nodes}

    for node in all_nodes:
        preds = cfg_preds.get(node, [])
        if len(preds) >= 2:  # Join point
            for pred in preds:
                runner = pred
                while runner != idom.get(node):
                    df[runner].add(node)
                    runner = idom.get(runner)

    return df
```

---

## 3. 자연 루프 감지

### 3.1 후방 엣지(Back Edges)

CFG의 **후방 엣지(back edge)**는 $h$가 $n$을 지배하는 엣지 $n \to h$입니다. 후방 엣지는 루프의 존재를 나타냅니다.

```python
def find_back_edges(cfg_succs, idom_sets):
    """
    Find all back edges in the CFG.

    cfg_succs: dict node -> list of successors
    idom_sets: dict node -> set of all dominators of that node
    Returns: list of (tail, head) tuples
    """
    back_edges = []
    for node in cfg_succs:
        for succ in cfg_succs[node]:
            if succ in idom_sets.get(node, set()):
                # succ dominates node, so node -> succ is a back edge
                back_edges.append((node, succ))
    return back_edges
```

### 3.2 자연 루프(Natural Loops)

후방 엣지 $n \to h$가 주어지면, **자연 루프(natural loop)**는 **헤더(header)** $h$와 $h$를 통하지 않고 $n$에 도달할 수 있는 모든 노드로 구성됩니다.

```python
def find_natural_loop(back_edge_tail, header, cfg_preds):
    """
    Find the natural loop for a back edge tail -> header.

    Returns: set of nodes in the loop (including header)
    """
    loop = {header}
    worklist = []

    if back_edge_tail != header:
        loop.add(back_edge_tail)
        worklist.append(back_edge_tail)

    while worklist:
        node = worklist.pop()
        for pred in cfg_preds.get(node, []):
            if pred not in loop:
                loop.add(pred)
                worklist.append(pred)

    return loop
```

**예시 -- CFG에서 루프 감지**:

```python
# Example CFG with two loops
#
#  entry -> A -> B -> C -> D -> exit
#                ^    |
#                |    v
#                +--- E
#           and D -> B (outer loop)

cfg_succs = {
    'entry': ['A'],
    'A': ['B'],
    'B': ['C'],
    'C': ['D', 'E'],
    'D': ['exit', 'B'],  # D->B is a back edge (B dom D)
    'E': ['B'],           # E->B is a back edge (B dom E)
    'exit': [],
}

cfg_preds = {
    'entry': [],
    'A': ['entry'],
    'B': ['A', 'D', 'E'],
    'C': ['B'],
    'D': ['C'],
    'E': ['C'],
    'exit': ['D'],
}

# After computing dominators, we find:
# Back edges: (D, B) and (E, B)
# Natural loop for D->B: {B, C, D, E}
# Natural loop for E->B: {B, C, E}
```

### 3.3 루프 중첩과 루프 트리

루프는 중첩될 수 있습니다. **루프 트리(loop tree)** (또는 **루프 포레스트**)는 루프를 중첩으로 구성합니다:

```
Loop Tree:
  Function
  ├── Loop L1 (header: B, body: {B, C, D, E})
  │   └── Loop L2 (header: B, body: {B, C, E})  [inner loop]
  └── (no more top-level loops)
```

두 루프가 헤더를 공유하면, 본문이 겹치는 경우 하나의 루프로 병합될 수 있습니다.

```python
class Loop:
    """Represents a natural loop in the CFG."""

    def __init__(self, header, body, back_edges):
        self.header = header
        self.body = body           # Set of nodes
        self.back_edges = back_edges
        self.parent = None         # Enclosing loop
        self.children = []         # Nested loops
        self.depth = 0             # Nesting depth (0 = outermost)

    def __repr__(self):
        return f"Loop(header={self.header}, body={self.body}, depth={self.depth})"


def build_loop_tree(loops):
    """
    Build a loop nesting tree from a list of loops.

    Loops are sorted by body size; smaller loops are children of larger ones.
    """
    sorted_loops = sorted(loops, key=lambda l: len(l.body))

    for i, inner in enumerate(sorted_loops):
        for outer in sorted_loops[i+1:]:
            if inner.body.issubset(outer.body):
                inner.parent = outer
                outer.children.append(inner)
                break

    # Compute depths
    for loop in sorted_loops:
        depth = 0
        current = loop
        while current.parent is not None:
            depth += 1
            current = current.parent
        loop.depth = depth

    # Return top-level loops (those with no parent)
    return [l for l in sorted_loops if l.parent is None]
```

### 3.4 프리헤더(Preheader) 삽입

많은 루프 최적화는 루프 직전에 정확히 한 번 실행되어야 하는 코드를 넣을 위치가 필요합니다. **프리헤더(preheader)**는 루프의 비후방 엣지(non-back-edge) 선행 블록과 헤더 사이에 삽입되는 전용 블록입니다:

```
Before:                       After:

  A ──┐                        A ──┐
       ├──▶ Header                  ├──▶ Preheader ──▶ Header
  B ──┘     ▲                  B ──┘                   ▲
             │                                          │
         back edge                                  back edge
             │                                          │
           Latch                                      Latch
```

```python
def insert_preheader(cfg_preds, cfg_succs, header, loop_body):
    """
    Insert a preheader block for a loop.

    Modifies cfg_preds and cfg_succs in place.
    Returns the name of the new preheader node.
    """
    preheader = f"pre_{header}"

    # Find non-back-edge predecessors of header
    external_preds = [p for p in cfg_preds[header] if p not in loop_body]

    # Redirect external predecessors to preheader
    cfg_preds[preheader] = external_preds
    cfg_succs[preheader] = [header]

    # Update header's predecessors
    cfg_preds[header] = [p for p in cfg_preds[header] if p in loop_body]
    cfg_preds[header].append(preheader)

    # Update external predecessors' successors
    for pred in external_preds:
        cfg_succs[pred] = [preheader if s == header else s
                           for s in cfg_succs[pred]]

    return preheader
```

---

## 4. 루프 불변 코드 이동 (LICM)

### 4.1 루프 불변 계산 식별

루프 내부의 계산이 반복 간에 결과가 변하지 않으면 **루프 불변(loop-invariant)**이라고 합니다. 형식적으로, 명령어 `x = op(a, b)`는 다음 경우 루프 불변입니다:

1. 모든 피연산자가 상수인 경우, 또는
2. 각 피연산자의 이 명령어에 도달하는 모든 정의가 루프 외부에 있는 경우, 또는
3. 각 피연산자에 대해 정확히 하나의 도달 정의가 있고, 그 정의 자체가 루프 불변인 경우

```python
def find_loop_invariant_instructions(loop_body, instructions, reaching_defs):
    """
    Find all loop-invariant instructions.

    loop_body: set of basic blocks in the loop
    instructions: dict block -> list of (target, op, operands)
    reaching_defs: dict (block, var) -> set of defining blocks

    Returns: set of loop-invariant instructions (block, index)
    """
    # Collect all variables defined inside the loop
    loop_defs = set()
    for block in loop_body:
        for idx, (target, op, operands) in enumerate(instructions.get(block, [])):
            loop_defs.add(target)

    invariant = set()
    changed = True

    while changed:
        changed = False
        for block in loop_body:
            for idx, (target, op, operands) in enumerate(instructions.get(block, [])):
                if (block, idx) in invariant:
                    continue

                # Check if all operands are loop-invariant
                all_invariant = True
                for operand in operands:
                    if operand not in loop_defs:
                        # Defined outside the loop -- invariant
                        continue

                    # Check if operand has a single reaching def that is invariant
                    defs = reaching_defs.get((block, operand), set())
                    loop_local_defs = defs & loop_body

                    if not loop_local_defs:
                        continue  # All defs outside loop

                    # Must have exactly one loop-local def, and it must be invariant
                    if len(loop_local_defs) != 1:
                        all_invariant = False
                        break

                    def_block = loop_local_defs.pop()
                    # Find the instruction index in def_block
                    found_invariant = False
                    for di, (dt, _, _) in enumerate(instructions.get(def_block, [])):
                        if dt == operand and (def_block, di) in invariant:
                            found_invariant = True
                            break

                    if not found_invariant:
                        all_invariant = False
                        break

                if all_invariant:
                    invariant.add((block, idx))
                    changed = True

    return invariant
```

### 4.2 안전한 코드 이동 조건

모든 루프 불변 명령어를 프리헤더로 안전하게 이동할 수 있는 것은 아닙니다. 명령어 `s: x = op(a, b)`는 다음 경우 끌어올릴 수 있습니다:

1. **지배 조건**: $s$를 포함하는 블록이 루프의 모든 출구를 지배합니다 (명령어가 출구에 도달하는 모든 반복에서 실행되었을 것입니다).
2. **유일성 조건**: 루프 내에 $x$의 다른 정의가 없습니다.
3. **활성성 조건**: 변수 $x$가 루프 내 $x$의 어떤 정의에서도 활성이 아닙니다 (다른 정의 $x$의 다른 사용 없음).

대안으로, 명령어에 부작용이 없고 그 결과가 루프 내에서만 사용된다면 항상 투기적으로 끌어올릴 수 있습니다 (최악의 경우 미사용 계산을 합니다).

```python
def can_hoist(instr_block, instr_target, loop_body, loop_exits, dom_sets,
              all_defs_in_loop, live_at_defs):
    """
    Check if a loop-invariant instruction can be safely hoisted.

    instr_block: the block containing the instruction
    instr_target: the variable being defined
    loop_body: set of blocks in the loop
    loop_exits: set of blocks that are loop exits
    dom_sets: dict block -> set of blocks it dominates
    all_defs_in_loop: dict variable -> set of defining blocks in loop
    live_at_defs: dict (block, variable) -> bool
    """
    # Condition 1: Block dominates all loop exits
    for exit_block in loop_exits:
        if exit_block not in dom_sets.get(instr_block, set()):
            return False

    # Condition 2: Unique definition in loop
    if len(all_defs_in_loop.get(instr_target, set())) > 1:
        return False

    # Condition 3: No interference with other defs
    for def_block in all_defs_in_loop.get(instr_target, set()):
        if def_block != instr_block:
            if live_at_defs.get((def_block, instr_target), False):
                return False

    return True
```

### 4.3 실제 LICM

다음은 간단한 루프에서 LICM을 시연하는 완전한 예시입니다:

```python
class SimpleInstruction:
    """Represents a three-address instruction."""
    def __init__(self, target, op, operands, is_side_effect=False):
        self.target = target
        self.op = op
        self.operands = operands
        self.is_side_effect = is_side_effect

    def __repr__(self):
        if self.op == 'copy':
            return f"{self.target} = {self.operands[0]}"
        return f"{self.target} = {self.operands[0]} {self.op} {self.operands[1]}"


def demonstrate_licm():
    """
    Demonstrate LICM on a simple loop.

    Original code:
        i = 0
        while i < n:
            x = a * b        # invariant
            y = x + c        # invariant (depends on x which is invariant)
            arr[i] = y + i
            i = i + 1

    After LICM:
        x = a * b            # hoisted
        y = x + c            # hoisted
        i = 0
        while i < n:
            arr[i] = y + i
            i = i + 1
    """
    print("=== Before LICM ===")
    preheader = [
        SimpleInstruction('i', 'copy', ['0']),
    ]
    loop_body = [
        SimpleInstruction('x', '*', ['a', 'b']),
        SimpleInstruction('y', '+', ['x', 'c']),
        SimpleInstruction('t1', '+', ['y', 'i']),
        SimpleInstruction('arr[i]', 'store', ['t1']),
        SimpleInstruction('i', '+', ['i', '1']),
    ]

    print("Preheader:")
    for instr in preheader:
        print(f"  {instr}")
    print("Loop body:")
    for instr in loop_body:
        print(f"  {instr}")

    # Identify invariant instructions
    # x = a * b  -- a, b not defined in loop -> invariant
    # y = x + c  -- x is invariant, c not defined in loop -> invariant
    # t1, arr[i], i are NOT invariant (depend on i which changes)

    hoisted = [loop_body[0], loop_body[1]]
    remaining = loop_body[2:]

    print("\n=== After LICM ===")
    print("Preheader:")
    for instr in preheader + hoisted:
        print(f"  {instr}")
    print("Loop body:")
    for instr in remaining:
        print(f"  {instr}")

demonstrate_licm()
```

출력:

```
=== Before LICM ===
Preheader:
  i = 0
Loop body:
  x = a * b
  y = x + c
  t1 = y + i
  arr[i] = store
  i = i + 1

=== After LICM ===
Preheader:
  i = 0
  x = a * b
  y = x + c
Loop body:
  t1 = y + i
  arr[i] = store
  i = i + 1
```

---

## 5. 귀납 변수 분석

### 5.1 기본 귀납 변수(Basic Induction Variables)

**기본 귀납 변수(basic induction variable, BIV)**는 루프 내부에서의 유일한 정의가 다음 형식인 변수 $i$입니다:

$$i = i + c \quad \text{or} \quad i = i - c$$

여기서 $c$는 루프 불변 양입니다. 전형적인 예는 루프 카운터입니다.

### 5.2 파생 귀납 변수(Derived Induction Variables)

**파생 귀납 변수(derived induction variable, DIV)** $j$는 기본 귀납 변수의 선형 함수로 정의되는 변수입니다:

$$j = a \cdot i + b$$

여기서 $a$와 $b$는 루프 불변입니다. **귀납 삼중쌍(induction triple)**을 $(i, a, b)$로 표기하며, 이는 $j = a \cdot i + b$를 의미합니다.

```python
class InductionVariable:
    """Represents an induction variable as a triple (base_iv, multiplier, offset)."""

    def __init__(self, name, base_iv=None, multiplier=1, offset=0):
        self.name = name
        self.base_iv = base_iv or name  # Basic IV it derives from
        self.multiplier = multiplier     # a in j = a*i + b
        self.offset = offset             # b in j = a*i + b
        self.is_basic = (base_iv is None or base_iv == name)

    def value_at(self, i_value):
        """Compute the value of this IV when the base IV has value i_value."""
        return self.multiplier * i_value + self.offset

    def __repr__(self):
        if self.is_basic:
            return f"BIV({self.name})"
        return f"DIV({self.name} = {self.multiplier}*{self.base_iv} + {self.offset})"


def detect_induction_variables(loop_instructions, loop_invariants):
    """
    Detect basic and derived induction variables in a loop.

    loop_instructions: list of (target, op, operands)
    loop_invariants: set of variable names that are loop-invariant

    Returns: dict name -> InductionVariable
    """
    # Phase 1: Find basic induction variables
    # A variable i is a BIV if its only definition is i = i +/- c (c invariant)
    definitions = {}  # var -> list of (op, operands)
    for target, op, operands in loop_instructions:
        if target not in definitions:
            definitions[target] = []
        definitions[target].append((op, operands))

    bivs = {}
    for var, defs in definitions.items():
        if len(defs) == 1:
            op, operands = defs[0]
            if op in ('+', '-') and var in operands:
                other = [o for o in operands if o != var]
                if len(other) == 1 and (other[0] in loop_invariants or
                                         other[0].lstrip('-').isdigit()):
                    step = int(other[0]) if op == '+' else -int(other[0])
                    bivs[var] = InductionVariable(var, multiplier=1, offset=0)
                    bivs[var].step = step

    # Phase 2: Find derived induction variables
    # j = a * i + b where i is BIV and a, b are loop-invariant
    divs = {}
    for target, op, operands in loop_instructions:
        if target in bivs:
            continue

        if op == '*':
            # Check if one operand is a BIV and the other is invariant
            for i, o in enumerate(operands):
                other_idx = 1 - i
                if o in bivs and (operands[other_idx] in loop_invariants or
                                   operands[other_idx].lstrip('-').isdigit()):
                    mult = operands[other_idx]
                    divs[target] = InductionVariable(
                        target, base_iv=o,
                        multiplier=int(mult), offset=0
                    )

        elif op == '+':
            # Check if one operand is a DIV/BIV and the other is invariant
            for i, o in enumerate(operands):
                other_idx = 1 - i
                if o in bivs and (operands[other_idx] in loop_invariants or
                                   operands[other_idx].lstrip('-').isdigit()):
                    off = int(operands[other_idx])
                    divs[target] = InductionVariable(
                        target, base_iv=o,
                        multiplier=1, offset=off
                    )

    result = {}
    result.update(bivs)
    result.update(divs)
    return result


# Example
print("=== Induction Variable Detection ===")
instructions = [
    ('i', '+', ['i', '1']),        # i = i + 1  (BIV)
    ('t1', '*', ['i', '4']),       # t1 = 4*i   (DIV: base=i, mult=4, off=0)
    ('t2', '+', ['t1', 'base']),   # t2 = t1 + base (not simple IV -- base is invariant
                                   #   but t1 is a DIV, so t2 = 4*i + base)
    ('arr_t2', 'load', ['t2']),    # memory load
]

invariants = {'base', '4', '1'}

ivs = detect_induction_variables(instructions, invariants)
for name, iv in ivs.items():
    print(f"  {iv}")
```

---

## 6. 강도 감소

### 6.1 아이디어

강도 감소(strength reduction)는 귀납 변수의 규칙적인 진행을 이용하여 비용이 많이 드는 연산 (곱셈 등)을 더 저렴한 연산 (덧셈 등)으로 교체합니다.

$j = a \cdot i + b$이고 $i$가 각 반복에서 $s$씩 증가하면:

$$j_{\text{new}} = j_{\text{old}} + a \cdot s$$

매 반복마다 $a \cdot i$를 계산하는 대신, $j = j + a \cdot s$를 계산합니다 -- 곱셈을 덧셈으로 교체.

### 6.2 Allen-Cocke-Kennedy 알고리즘

고전적인 강도 감소 알고리즘은 다음과 같이 동작합니다:

1. 모든 기본 귀납 변수와 그 단계(step) 찾기
2. 모든 파생 귀납 변수 찾기
3. 각 DIV $j = a \cdot i + b$에 대해:
   - 새 변수 $j'$ 생성
   - 프리헤더에서 $j' = a \cdot i_0 + b$ 초기화
   - $i$가 $s$씩 증가할 때마다 $j' = j' + a \cdot s$ 추가
   - $j$의 사용을 $j'$로 교체

```python
def strength_reduce(loop_code, preheader_code, bivs, divs):
    """
    Apply strength reduction to derived induction variables.

    Demonstrates the transformation on a simple example.
    """
    print("=== Before Strength Reduction ===")
    print("Preheader:")
    for line in preheader_code:
        print(f"  {line}")
    print("Loop body:")
    for line in loop_code:
        print(f"  {line}")

    new_preheader = list(preheader_code)
    new_loop = []

    # For each derived IV, create strength-reduced version
    reduced = {}  # original var -> new var

    for div_name, div_info in divs.items():
        base = div_info['base_iv']
        mult = div_info['multiplier']
        offset = div_info['offset']
        step = bivs[base]['step']

        new_var = f"{div_name}_sr"
        reduced[div_name] = new_var

        # Initialize in preheader: new_var = mult * base_init + offset
        init_val = mult * bivs[base]['init'] + offset
        new_preheader.append(f"{new_var} = {init_val}")

        # Increment in loop: new_var = new_var + mult * step
        increment = mult * step
        new_loop.append(f"{new_var} = {new_var} + {increment}")

    # Copy non-reduced instructions
    for line in loop_code:
        # Replace references to reduced variables
        replaced = False
        for orig, new in reduced.items():
            if line.startswith(f"{orig} ="):
                replaced = True  # Skip original computation
                break
            if orig in line:
                line = line.replace(orig, new)
        if not replaced:
            new_loop.append(line)

    print("\n=== After Strength Reduction ===")
    print("Preheader:")
    for line in new_preheader:
        print(f"  {line}")
    print("Loop body:")
    for line in new_loop:
        print(f"  {line}")


# Example: Array indexing
#   for i in range(n):
#       addr = base + i * 4    # addr is a DIV of i
#       arr[addr] = 0

strength_reduce(
    loop_code=[
        "t1 = i * 4",          # DIV: t1 = 4*i
        "addr = t1 + base",    # addr = 4*i + base (also a DIV)
        "store 0, addr",
        "i = i + 1",
    ],
    preheader_code=[
        "i = 0",
    ],
    bivs={
        'i': {'init': 0, 'step': 1}
    },
    divs={
        't1': {'base_iv': 'i', 'multiplier': 4, 'offset': 0},
    }
)
```

출력:

```
=== Before Strength Reduction ===
Preheader:
  i = 0
Loop body:
  t1 = i * 4
  addr = t1 + base
  store 0, addr
  i = i + 1

=== After Strength Reduction ===
Preheader:
  i = 0
  t1_sr = 0
Loop body:
  t1_sr = t1_sr + 4
  addr = t1_sr + base
  store 0, addr
  i = i + 1
```

곱셈 `i * 4`가 덧셈 `t1_sr + 4`로 교체되었습니다.

### 6.3 선형 테스트 교체

강도 감소 후, 원래 기본 귀납 변수가 루프의 출구 테스트에서만 사용될 수 있습니다. 강도 감소된 변수에 기반한 테스트로 교체할 수 있습니다:

```
이전:  if i < n goto loop     이후:  if t1_sr < 4*n goto loop
```

이를 통해 죽은 코드 제거로 원래 BIV를 완전히 제거할 수 있습니다.

---

## 7. 루프 언롤링

### 7.1 완전 언롤링

트립 카운트(반복 횟수)가 컴파일 시간에 알려져 있고 작으면, 루프를 순차적 코드로 완전히 교체할 수 있습니다:

```python
def demonstrate_full_unrolling():
    """Full unrolling: replace loop with sequential statements."""

    print("=== Before Full Unrolling ===")
    print("""
    sum = 0
    for i in range(4):
        sum += arr[i]
    """)

    print("=== After Full Unrolling ===")
    print("""
    sum = 0
    sum += arr[0]
    sum += arr[1]
    sum += arr[2]
    sum += arr[3]
    """)

demonstrate_full_unrolling()
```

이점:
- 루프 오버헤드 제거 (분기, 카운터 증가, 비교)
- 추가 최적화 가능 (인덱스에 대한 상수 폴딩, 명령어 스케줄링)

비용:
- 코드 크기 증가
- 작은 알려진 트립 카운트에만 실용적

### 7.2 부분 언롤링

더 일반적으로, 루프는 인수 $k$로 **부분적으로 언롤링**됩니다: 새 루프의 각 반복이 $k$번 반복 분량의 작업을 수행합니다.

트립 카운트 $n$과 언롤링 인수 $k$에 대한 루프:

```python
def demonstrate_partial_unrolling(n=100, k=4):
    """
    Partial unrolling by factor k.

    Original: for i in range(n): body(i)
    Unrolled:
        for i in range(0, n - n%k, k):
            body(i)
            body(i+1)
            body(i+2)
            body(i+3)
        # Cleanup loop for remainder
        for i in range(n - n%k, n):
            body(i)
    """
    print(f"=== Partial Unrolling (factor {k}) ===")

    # Original loop
    total_orig = 0
    iterations_orig = 0
    for i in range(n):
        total_orig += i * i
        iterations_orig += 1

    # Unrolled loop
    total_unrolled = 0
    iterations_unrolled = 0
    main_limit = n - (n % k)

    for i in range(0, main_limit, k):
        total_unrolled += i * i
        total_unrolled += (i + 1) * (i + 1)
        total_unrolled += (i + 2) * (i + 2)
        total_unrolled += (i + 3) * (i + 3)
        iterations_unrolled += 1

    # Cleanup
    for i in range(main_limit, n):
        total_unrolled += i * i
        iterations_unrolled += 1

    print(f"Original:  {iterations_orig} loop iterations, sum = {total_orig}")
    print(f"Unrolled:  {iterations_unrolled} loop iterations, sum = {total_unrolled}")
    print(f"Results match: {total_orig == total_unrolled}")
    print(f"Branch reduction: {iterations_orig - iterations_unrolled} fewer branches")

demonstrate_partial_unrolling()
```

### 7.3 언롤링과 소프트웨어 파이프라이닝

언롤링은 **소프트웨어 파이프라이닝(software pipelining)**을 가능하게 합니다 -- 서로 다른 반복의 연산을 겹쳐 지연을 숨깁니다. 3사이클 곱셈과 1사이클 덧셈을 가진 프로세서에서:

```
Iteration:   Body 1      Body 2      Body 3      Body 4
Cycle 1:     MUL a1      ---         ---         ---
Cycle 2:     ---          MUL a2     ---         ---
Cycle 3:     ---          ---         MUL a3     ---
Cycle 4:     ADD r1       ---         ---         MUL a4
Cycle 5:     ---          ADD r2     ---         ---
...
```

언롤링 없이는 곱셈이 완료될 때까지 기다리는 버블 (스톨)이 발생합니다.

### 7.4 구현 고려 사항

```python
def unroll_loop(original_body, trip_count, unroll_factor):
    """
    Generate an unrolled loop.

    original_body: function(i) -> list of instructions
    trip_count: number of iterations (may be symbolic)
    unroll_factor: k

    Returns: (main_loop, cleanup_loop)
    """
    if isinstance(trip_count, int) and trip_count <= unroll_factor:
        # Full unroll
        full_body = []
        for i in range(trip_count):
            full_body.extend(original_body(i))
        return full_body, []

    # Partial unroll
    main_body = []
    for offset in range(unroll_factor):
        main_body.extend(original_body(f"i+{offset}" if offset > 0 else "i"))

    cleanup_body = original_body("i")

    return main_body, cleanup_body


# Decision heuristics
def should_unroll(loop, unroll_factor=4):
    """
    Heuristic: decide whether to unroll a loop.
    """
    # Don't unroll loops with function calls (code bloat)
    if loop.get('has_calls', False):
        return False

    # Don't unroll loops with many instructions (code bloat)
    if loop.get('body_size', 0) > 20:
        return False

    # Don't unroll deeply nested inner loops (already many iterations)
    if loop.get('nesting_depth', 0) > 3:
        return False

    # Unroll small inner loops
    if loop.get('is_innermost', False) and loop.get('body_size', 0) <= 10:
        return True

    return False
```

---

## 8. 루프 결합과 분리

### 8.1 루프 결합(Loop Fusion, Jamming)

루프 결합은 동일한 반복 공간을 가진 두 인접 루프를 하나로 결합합니다:

```python
def demonstrate_loop_fusion():
    """Show loop fusion combining two loops into one."""
    import time

    n = 1_000_000
    a = list(range(n))
    b = [0] * n
    c = [0] * n

    # Before fusion: two separate loops
    print("=== Before Fusion (Two Loops) ===")
    start = time.perf_counter()

    for i in range(n):
        b[i] = a[i] * 2

    for i in range(n):
        c[i] = a[i] + 1

    time_separate = time.perf_counter() - start
    print(f"Time: {time_separate:.4f}s")

    # After fusion: one loop
    print("\n=== After Fusion (Single Loop) ===")
    b2 = [0] * n
    c2 = [0] * n
    start = time.perf_counter()

    for i in range(n):
        b2[i] = a[i] * 2
        c2[i] = a[i] + 1

    time_fused = time.perf_counter() - start
    print(f"Time: {time_fused:.4f}s")
    print(f"Speedup: {time_separate / time_fused:.2f}x")
    print(f"Results correct: {b == b2 and c == c2}")

# demonstrate_loop_fusion()  # Uncomment to run
```

**결합의 이점**:
- 루프 오버헤드 감소 (하나의 루프 대신 두 개)
- 데이터 지역성(data locality) 향상 (배열 `a`에 두 번 대신 한 번 접근)
- 추가 최적화 가능 (병합된 본문에 걸쳐 공통 부분식 제거)

**합법성(Legality)**: 두 루프 간에 결합을 방해하는 데이터 의존성이 없을 때 결합은 합법적입니다. 구체적으로, **결합 방지 의존성(fusion-preventing dependency)** -- 루프 2의 반복 $i$에서 생성된 원소를 루프 1의 반복 $j > i$에서 소비하는 의존성 -- 이 없어야 합니다.

### 8.2 루프 분리(Loop Fission, Distribution)

루프 분리는 그 반대입니다: 하나의 루프를 여러 루프로 분리합니다.

```python
def demonstrate_loop_fission():
    """Show loop fission splitting one loop into two."""

    print("=== Before Fission (Single Loop) ===")
    print("""
    for i in range(n):
        b[i] = a[i] * 2        # Vectorizable
        c[i] = c[i-1] + b[i]   # Sequential dependency
    """)

    print("=== After Fission (Two Loops) ===")
    print("""
    # Loop 1: vectorizable
    for i in range(n):
        b[i] = a[i] * 2

    # Loop 2: sequential
    for i in range(n):
        c[i] = c[i-1] + b[i]
    """)
    print("Loop 1 can now be vectorized independently!")
    print("Loop 2 carries a dependency that prevents vectorization.")

demonstrate_loop_fission()
```

**분리의 이점**:
- 다른 부분에 의존성이 있어도 한 부분의 벡터화 가능
- 각 루프가 서로 다른 데이터에 접근할 때 캐시 동작 개선
- 추가 최적화를 위한 분석 단순화

---

## 9. 루프 타일링 (블로킹)

### 9.1 캐시 문제

큰 2D 배열을 열 단위로 처리할 때, 메모리에서 인접 접근이 멀리 떨어져 있습니다 (스트라이드 $= n$, 행 크기). 이는 **캐시 스래싱(cache thrashing)**을 유발합니다.

### 9.2 타일링 변환

루프 타일링은 반복을 캐시에 맞는 작은 블록 ("타일")으로 분할합니다:

```python
import time

def demonstrate_loop_tiling(n=1000):
    """
    Demonstrate the effect of loop tiling on matrix multiplication.

    Standard: O(n^3) with poor cache behavior
    Tiled: O(n^3) same work, but much better cache locality
    """
    import random
    random.seed(42)

    # Create matrices
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]

    # Standard matrix multiply (ijk order)
    print(f"=== Matrix Multiply {n}x{n} ===")
    C1 = [[0.0] * n for _ in range(n)]

    start = time.perf_counter()
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += A[i][k] * B[k][j]
            C1[i][j] = total
    time_standard = time.perf_counter() - start
    print(f"Standard (ijk): {time_standard:.3f}s")

    # Tiled matrix multiply
    TILE = 32  # Tile size -- should fit in L1 cache
    C2 = [[0.0] * n for _ in range(n)]

    start = time.perf_counter()
    for ii in range(0, n, TILE):
        for jj in range(0, n, TILE):
            for kk in range(0, n, TILE):
                # Multiply tile
                for i in range(ii, min(ii + TILE, n)):
                    for j in range(jj, min(jj + TILE, n)):
                        total = C2[i][j]
                        for k in range(kk, min(kk + TILE, n)):
                            total += A[i][k] * B[k][j]
                        C2[i][j] = total
    time_tiled = time.perf_counter() - start
    print(f"Tiled (tile={TILE}): {time_tiled:.3f}s")
    print(f"Speedup: {time_standard / time_tiled:.2f}x")

    # Verify correctness
    max_diff = max(abs(C1[i][j] - C2[i][j]) for i in range(n) for j in range(n))
    print(f"Max difference: {max_diff:.2e}")

# demonstrate_loop_tiling(500)  # Uncomment to run (may take a while)
```

### 9.3 타일 크기 선택

타일 크기 $T$는 작업 집합(working set)이 캐시에 맞도록 선택해야 합니다. 행렬 곱셈에서 접근되는 $A$, $B$, $C$의 타일은 각각 $T \times T$이므로:

$$3 \cdot T^2 \cdot \text{sizeof(element)} \leq \text{L1 cache size}$$

8바이트 배정밀도 실수와 32 KB의 L1 캐시에서:

$$T \leq \sqrt{\frac{32768}{3 \times 8}} = \sqrt{1365} \approx 36$$

일반적인 선택은 $T = 32$입니다.

### 9.4 다중 수준 타일링

여러 캐시 수준 (L1, L2, L3)이 있는 머신에서는 여러 수준에서 타일링을 적용할 수 있습니다:

```
Original:
  for i in range(n):
      for j in range(n):
          C[i][j] += A[i][k] * B[k][j]

Two-level tiled:
  for ii in range(0, n, T2):          # L2 tile
      for jj in range(0, n, T2):
          for kk in range(0, n, T2):
              for iii in range(ii, ii+T2, T1):    # L1 tile
                  for jjj in range(jj, jj+T2, T1):
                      for kkk in range(kk, kk+T2, T1):
                          # Micro-kernel on T1 x T1 tile
                          ...
```

---

## 10. 루프 교환

### 10.1 동기

루프 교환(loop interchange)은 중첩 루프의 순서를 바꾸어 메모리 접근 패턴을 개선합니다.

```python
def demonstrate_loop_interchange():
    """
    Show how loop interchange improves cache behavior.

    Row-major arrays: A[i][j] is stored next to A[i][j+1] in memory.
    Accessing A[i][j] with j as the inner loop gives stride-1 access (good).
    Accessing with i as the inner loop gives stride-n access (bad).
    """
    import time
    n = 2000

    # Create a 2D array (list of lists in Python, simulating row-major)
    A = [[0.0] * n for _ in range(n)]

    # Bad order: column-major traversal (i inner, j outer)
    print(f"=== Loop Interchange Demo ({n}x{n}) ===")
    start = time.perf_counter()
    for j in range(n):          # Outer
        for i in range(n):      # Inner -- stride-n access to A[i][j]
            A[i][j] = i + j
    time_bad = time.perf_counter() - start
    print(f"Column-major (j outer, i inner): {time_bad:.3f}s")

    # Good order: row-major traversal (j inner, i outer)
    A = [[0.0] * n for _ in range(n)]
    start = time.perf_counter()
    for i in range(n):          # Outer
        for j in range(n):      # Inner -- stride-1 access to A[i][j]
            A[i][j] = i + j
    time_good = time.perf_counter() - start
    print(f"Row-major (i outer, j inner): {time_good:.3f}s")
    print(f"Speedup: {time_bad / time_good:.2f}x")

# demonstrate_loop_interchange()  # Uncomment to run
```

### 10.2 루프 교환의 합법성

루프 교환은 데이터 의존성을 위반하지 않을 때만 합법적입니다. 루프 $i$와 $j$에 대한 거리 벡터 $(d_1, d_2)$를 가진 의존성이 있을 때:

- 교환은 결과 거리 벡터 $(d_2, d_1)$이 **사전 순서상 양수(lexicographically positive)** (가장 왼쪽의 0이 아닌 성분이 양수)이면 합법적입니다.

```python
def can_interchange(distance_vectors):
    """
    Check if loop interchange is legal given dependency distance vectors.

    A distance vector (d1, d2) becomes (d2, d1) after interchange.
    All resulting vectors must be lexicographically positive.
    """
    for d1, d2 in distance_vectors:
        # After interchange: (d2, d1)
        new_vec = (d2, d1)

        # Check lexicographic positivity
        if new_vec[0] < 0:
            return False
        if new_vec[0] == 0 and new_vec[1] < 0:
            return False

    return True


# Examples
print("Dependency (1, 0):", can_interchange([(1, 0)]))   # True: becomes (0, 1)
print("Dependency (0, 1):", can_interchange([(0, 1)]))   # True: becomes (1, 0)
print("Dependency (1, -1):", can_interchange([(1, -1)])) # False: becomes (-1, 1)
print("Dependency (1, 1):", can_interchange([(1, 1)]))   # True: becomes (1, 1)
```

---

## 11. 벡터화

### 11.1 SIMD 개요

**SIMD**(Single Instruction, Multiple Data, 단일 명령어 다중 데이터)는 여러 데이터 원소를 동시에 처리하는 명령어입니다. 현대 CPU에는 한 번에 128, 256, 또는 512비트를 처리할 수 있는 SIMD 유닛 (x86의 SSE, AVX, AVX-512; ARM의 NEON)이 있습니다.

```
Scalar:                    SIMD (4-wide):
  a[0] + b[0] = c[0]        a[0:4] + b[0:4] = c[0:4]
  a[1] + b[1] = c[1]        (single instruction)
  a[2] + b[2] = c[2]
  a[3] + b[3] = c[3]
  (4 instructions)
```

256비트 AVX와 32비트 부동 소수점에서 하나의 명령어가 $256 / 32 = 8$개의 원소를 처리합니다.

### 11.2 자동 벡터화

컴파일러는 루프를 자동으로 벡터화하려고 시도합니다. 핵심 요구 사항은 **루프 전이 의존성(loop-carried dependencies)이 없음** -- 각 반복이 독립적이어야 합니다.

```python
def vectorization_analysis():
    """Analyze loops for vectorizability."""

    examples = [
        {
            'name': 'Simple element-wise',
            'code': 'c[i] = a[i] + b[i]',
            'vectorizable': True,
            'reason': 'No loop-carried dependency'
        },
        {
            'name': 'Reduction',
            'code': 'sum += a[i]',
            'vectorizable': True,
            'reason': 'Reduction pattern -- compiler uses horizontal add'
        },
        {
            'name': 'Prefix sum',
            'code': 'a[i] = a[i-1] + b[i]',
            'vectorizable': False,
            'reason': 'Loop-carried dependency: a[i] depends on a[i-1]'
        },
        {
            'name': 'Conditional',
            'code': 'if a[i] > 0: b[i] = a[i]',
            'vectorizable': True,
            'reason': 'Predicated execution with masking'
        },
        {
            'name': 'Indirect access',
            'code': 'b[idx[i]] = a[i]',
            'vectorizable': False,  # Usually
            'reason': 'Scatter -- possible with AVX-512 but slow'
        },
        {
            'name': 'Function call',
            'code': 'b[i] = expensive_func(a[i])',
            'vectorizable': False,
            'reason': 'Function calls prevent vectorization (unless intrinsic)'
        },
    ]

    print("=== Vectorization Analysis ===")
    for ex in examples:
        status = "YES" if ex['vectorizable'] else "NO"
        print(f"\n  {ex['name']}: {ex['code']}")
        print(f"    Vectorizable: {status}")
        print(f"    Reason: {ex['reason']}")

vectorization_analysis()
```

### 11.3 NumPy를 사용한 벡터화

Python에서 NumPy는 내부적으로 SIMD를 활용하는 벡터화 연산을 제공합니다:

```python
import numpy as np
import time

def numpy_vectorization_demo():
    """Demonstrate NumPy vectorization vs pure Python loops."""
    n = 10_000_000

    # Pure Python
    a = list(range(n))
    b = list(range(n))

    start = time.perf_counter()
    c_python = [a[i] + b[i] for i in range(n)]
    time_python = time.perf_counter() - start

    # NumPy vectorized
    a_np = np.arange(n, dtype=np.float64)
    b_np = np.arange(n, dtype=np.float64)

    start = time.perf_counter()
    c_np = a_np + b_np  # Single vectorized operation
    time_numpy = time.perf_counter() - start

    print(f"Python loop: {time_python:.3f}s")
    print(f"NumPy vectorized: {time_numpy:.3f}s")
    print(f"Speedup: {time_python / time_numpy:.1f}x")

# numpy_vectorization_demo()  # Uncomment to run
```

### 11.4 스트립 마이닝(Strip Mining)

스트립 마이닝(strip mining)은 SIMD를 위해 루프를 변환하여 벡터 길이를 명시적으로 만듭니다:

```python
def strip_mine(n, vector_length=4):
    """
    Strip mine a loop for vectorization.

    Before: for i in range(n): body(i)
    After:  for i in range(0, n, VL):
                for ii in range(i, min(i+VL, n)):
                    body(ii)
    """
    print(f"=== Strip Mining (VL={vector_length}) ===")
    print(f"Original: for i in range({n}): body(i)")
    print(f"Strip-mined:")

    iteration_count = 0
    for i in range(0, n, vector_length):
        end = min(i + vector_length, n)
        vec_indices = list(range(i, end))
        print(f"  Vector op: body({vec_indices})")
        iteration_count += 1

    print(f"Outer iterations: {iteration_count} (was {n})")

strip_mine(14, 4)
```

출력:

```
=== Strip Mining (VL=4) ===
Original: for i in range(14): body(i)
Strip-mined:
  Vector op: body([0, 1, 2, 3])
  Vector op: body([4, 5, 6, 7])
  Vector op: body([8, 9, 10, 11])
  Vector op: body([12, 13])
Outer iterations: 4 (was 14)
```

### 11.5 정렬과 필링(Peeling)

SIMD 명령어는 종종 특정 경계에 데이터가 정렬되어야 합니다 (SSE는 16바이트, AVX는 32바이트). **루프 필링(loop peeling)**은 첫 몇 번의 반복 (정렬에 도달할 때까지)을 스칼라 연산으로 처리합니다:

```
필링된 반복:       i = 0, 1  (스칼라, 정렬될 때까지)
주요 벡터화 루프:  i = 2, 3, 4, 5 | 6, 7, 8, 9 | ...  (SIMD)
정리 반복:         i = n-2, n-1  (스칼라, 나머지)
```

---

## 12. 다면체 모델

### 12.1 개요

**다면체 모델(polyhedral model)** (또는 **폴리토프 모델(polytope model)**)은 루프 중첩 최적화를 위한 강력한 수학적 프레임워크입니다. 루프 반복을 다면체 내의 정수 점으로 표현하고, 루프 변환을 이 점들에 대한 아핀(affine) 함수로 표현합니다.

### 12.2 반복 도메인

이중 중첩 루프를 생각해봅시다:

```python
for i in range(0, N):
    for j in range(0, M):
        A[i][j] = B[i][j-1] + B[i-1][j]
```

**반복 도메인(iteration domain)**은 실행되는 모든 $(i, j)$ 쌍의 집합입니다:

$$\mathcal{D} = \{ (i, j) \in \mathbb{Z}^2 \mid 0 \leq i < N, \; 0 \leq j < M \}$$

이것은 2D 직사각형 다면체(폴리토프)입니다.

### 12.3 접근 함수

각 메모리 접근은 **접근 함수(access function)** -- 반복 좌표에서 배열 인덱스로의 아핀 매핑 -- 으로 설명됩니다:

- `A[i][j]`: 접근 함수 $f_A(i, j) = (i, j)$
- `B[i][j-1]`: 접근 함수 $f_{B1}(i, j) = (i, j-1)$
- `B[i-1][j]`: 접근 함수 $f_{B2}(i, j) = (i-1, j)$

### 12.4 의존성 다면체

문장 인스턴스 간의 의존성은 **의존성 다면체(dependence polyhedra)**로 포착됩니다. 반복 $(i_1, j_1)$에서 $(i_2, j_2)$로의 의존성은 동일한 메모리 위치에 접근하고 소스가 싱크보다 먼저 실행될 때 존재합니다.

읽기 `B[i][j-1]`과 그것을 생성한 쓰기에 대해:

$$i_2 = i_1, \quad j_2 - 1 = j_1 \implies j_2 = j_1 + 1$$

의존성은 $(i, j) \to (i, j+1)$이며 거리 벡터는 $(0, 1)$입니다.

### 12.5 스케줄과 변환

**스케줄(schedule)**은 각 반복 점을 타임 스텝에 매핑합니다. 유효한 스케줄은 모든 의존성을 존중해야 합니다. 스케줄은 종종 아핀 함수로 표현됩니다:

$$\theta(i, j) = \alpha_1 i + \alpha_2 j + \alpha_0$$

다면체 모델은 정수 선형 프로그래밍(ILP)을 풀어 최적 스케줄을 찾을 수 있습니다.

```python
def polyhedral_example():
    """
    Simple demonstration of polyhedral iteration domain and scheduling.
    """
    N, M = 4, 4

    print("=== Polyhedral Model Example ===")
    print(f"Iteration domain: 0 <= i < {N}, 0 <= j < {M}")
    print(f"Statement: A[i][j] = B[i][j-1] + B[i-1][j]")
    print(f"Dependencies: (0,1) and (1,0)")

    # Original schedule: theta(i,j) = (i, j) -- lexicographic order
    print("\nOriginal schedule (row by row):")
    for i in range(N):
        for j in range(M):
            print(f"  t={i*M + j}: S({i},{j})")

    # Skewed schedule: theta(i,j) = (i+j, j) -- enables wavefront parallelism
    print("\nSkewed schedule (wavefront):")
    schedule = []
    for i in range(N):
        for j in range(M):
            time = (i + j, j)
            schedule.append((time, i, j))

    schedule.sort()

    for time, i, j in schedule:
        print(f"  t={time}: S({i},{j})")

    print("\nWavefront parallelism:")
    print("Iterations on the same diagonal (same i+j) can execute in parallel!")
    for wave in range(N + M - 1):
        parallel = [(i, wave - i) for i in range(N) if 0 <= wave - i < M]
        print(f"  Wave {wave}: {parallel}")

polyhedral_example()
```

### 12.6 도구

실제 다면체 최적화기에는 다음이 포함됩니다:
- **ISL** (Integer Set Library): 수학적 기반
- **Pluto**: 자동 병렬화기 및 지역성 최적화기
- **Polly**: LLVM의 다면체 최적화 패스
- **PPCG**: 다면체 병렬 코드 생성기 (GPU용)

---

## 13. 루프 병렬화

### 13.1 의존성이 없는 루프

루프 전이 의존성이 없는 루프는 **자명하게 병렬화 가능(trivially parallelizable)** (또는 "당혹스럽게 병렬(embarrassingly parallel)")합니다:

```python
# Parallelizable: each iteration is independent
for i in range(n):
    c[i] = a[i] + b[i]

# NOT parallelizable: iteration i depends on i-1
for i in range(1, n):
    a[i] = a[i-1] + b[i]
```

### 13.2 의존성 유형

루프 전이 의존성의 세 가지 유형:

| 유형 | 이름 | 예시 | 병렬화 방해? |
|------|------|---------|------------------------|
| RAW | 참/흐름(True/Flow) | `a[i] = ...; ... = a[i-1]` | 예 |
| WAR | 역(Anti) | `... = a[i-1]; a[i] = ...` | 이름 변경 가능 |
| WAW | 출력(Output) | `a[i] = ...; a[i] = ...` | 이름 변경 가능 |

WAR와 WAW 의존성은 **프라이빗화(privatization)** (각 스레드에 자체 복사본 제공) 또는 이름 변경으로 제거할 수 있습니다.

### 13.3 리덕션 병렬화

리덕션(합, 곱, 최솟값, 최댓값)은 병렬 실행을 허용하는 특별한 패턴을 가집니다:

```python
# Sequential reduction
total = 0
for i in range(n):
    total += a[i]  # RAW dependency on total

# Parallel reduction
# Each thread computes a partial sum, then combine
from concurrent.futures import ThreadPoolExecutor

def parallel_sum(arr, num_threads=4):
    """Parallel reduction using thread pool."""
    n = len(arr)
    chunk_size = (n + num_threads - 1) // num_threads

    def partial_sum(start, end):
        return sum(arr[start:end])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for t in range(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            futures.append(executor.submit(partial_sum, start, end))

        total = sum(f.result() for f in futures)

    return total
```

### 13.4 DOALL, DOACROSS, DOPIPE

세 가지 병렬화 전략:

**DOALL**: 모든 반복이 독립적 -- 모두 병렬로 실행.

```python
# DOALL parallelism
# for i in range(n):
#     c[i] = f(a[i])
# All iterations can execute simultaneously
```

**DOACROSS**: 반복들이 의존성을 가지지만, 동기화를 통해 실행을 겹칠 수 있음.

```python
# DOACROSS parallelism
# for i in range(n):
#     a[i] = a[i-1] + b[i]    # depends on previous iteration
#     c[i] = expensive(a[i])   # expensive independent computation
#
# Thread 1 computes a[0], starts expensive(a[0])
# Thread 2 waits for a[0], then computes a[1], starts expensive(a[1])
# Overlap the expensive computation
```

**DOPIPE**: 파이프라인 병렬성 -- 계산의 각 단계가 다른 스레드에서 실행됨.

```python
# DOPIPE parallelism
# Stage 1 (Thread 1): for i: a[i] = input[i] * 2
# Stage 2 (Thread 2): for i: b[i] = a[i] + offset  (wait for a[i] from Stage 1)
# Stage 3 (Thread 3): for i: output[i] = transform(b[i])
```

### 13.5 OpenMP 스타일 병렬화

OpenMP를 지원하는 컴파일러는 프라그마(pragma)로 루프를 병렬화할 수 있습니다. Python에서 동등한 코드:

```python
from multiprocessing import Pool

def openmp_style_parallel():
    """Demonstrate parallel loop execution (Python equivalent of OpenMP)."""
    import os

    n = 1_000_000
    a = list(range(n))
    num_workers = os.cpu_count() or 4

    def process_chunk(args):
        start, end, data = args
        return [x * x + 2 * x + 1 for x in data[start:end]]

    chunk_size = (n + num_workers - 1) // num_workers
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, n), a)
        for i in range(num_workers)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Flatten results
    c = []
    for chunk_result in results:
        c.extend(chunk_result)

    print(f"Processed {len(c)} elements using {num_workers} workers")
    print(f"First 5: {c[:5]}")
    print(f"Last 5: {c[-5:]}")

# openmp_style_parallel()  # Uncomment to run
```

### 13.6 병렬화를 위한 컴파일러 분석

```python
def analyze_parallelizability(loop_deps):
    """
    Analyze whether a loop can be parallelized and suggest strategy.

    loop_deps: list of dependency dicts with 'type', 'distance', 'variable'
    """
    if not loop_deps:
        return "DOALL", "No dependencies -- fully parallel"

    has_true_dep = any(d['type'] == 'RAW' for d in loop_deps)
    has_reduction = any(d.get('is_reduction', False) for d in loop_deps)
    only_renamable = all(d['type'] in ('WAR', 'WAW') for d in loop_deps)

    if has_reduction and not has_true_dep:
        return "DOALL + Reduction", "Reduction pattern detected -- use parallel reduction"

    if only_renamable:
        return "DOALL (after privatization)", "Only anti/output deps -- privatize variables"

    if has_true_dep:
        min_distance = min(d['distance'] for d in loop_deps if d['type'] == 'RAW')
        if min_distance > 1:
            return "DOACROSS", f"True dep with distance {min_distance} -- overlap possible"
        return "Sequential", "True dependency with distance 1 -- cannot parallelize"

    return "DOALL", "Safe to parallelize"


# Examples
examples = [
    ("c[i] = a[i] + b[i]", []),
    ("sum += a[i]", [{'type': 'RAW', 'distance': 1, 'variable': 'sum', 'is_reduction': True}]),
    ("a[i] = a[i-1] + 1", [{'type': 'RAW', 'distance': 1, 'variable': 'a'}]),
    ("a[i] = a[i-4] + 1", [{'type': 'RAW', 'distance': 4, 'variable': 'a'}]),
]

print("=== Parallelizability Analysis ===")
for code, deps in examples:
    strategy, reason = analyze_parallelizability(deps)
    print(f"\n  Code: {code}")
    print(f"  Strategy: {strategy}")
    print(f"  Reason: {reason}")
```

---

## 14. 요약

루프 최적화는 프로그램이 실행 시간의 압도적인 대부분을 루프에서 보내기 때문에 컴파일러 최적화 중 가장 큰 영향을 미치는 범주입니다. 다음을 다루었습니다:

| 최적화 | 효과 | 핵심 요구 사항 |
|-------------|--------|-----------------|
| **지배자 분석(Dominator analysis)** | 루프 감지의 기반 | CFG 사용 가능 |
| **루프 감지(Loop detection)** | 후방 엣지로 자연 루프 식별 | 지배자 계산됨 |
| **LICM** | 불변 코드를 루프 밖으로 끌어올림 | 지배 + 활성성 |
| **강도 감소(Strength reduction)** | IV에서 곱셈을 덧셈으로 교체 | 귀납 변수 분석 |
| **루프 언롤링(Loop unrolling)** | 분기 오버헤드 감소, ILP 가능 | 트립 카운트 지식 |
| **루프 결합(Loop fusion)** | 지역성 향상, 오버헤드 감소 | 결합 방지 의존성 없음 |
| **루프 분리(Loop fission)** | 서브루프의 벡터화 가능 | 의존성 분할 |
| **루프 타일링(Loop tiling)** | 작업 집합을 캐시에 맞춤 | 알려진 경계 |
| **루프 교환(Loop interchange)** | 접근 패턴을 메모리 레이아웃에 맞춤 | 합법적 거리 벡터 |
| **벡터화(Vectorization)** | SIMD 하드웨어 활용 | 루프 전이 의존성 없음 |
| **다면체 모델(Polyhedral model)** | 모든 변환을 위한 통합 프레임워크 | 아핀 루프 |
| **병렬화(Parallelization)** | 다중 코어 활용 | 의존성 분석 |

이 모든 최적화를 통합하는 핵심 통찰은 **의존성 분석(dependency analysis)**입니다: 모든 변환은 원래 프로그램의 데이터 의존성을 보존해야 합니다. 어떤 계산이 무엇에 의존하는지 -- 그리고 무엇을 안전하게 재정렬, 끌어올리기, 또는 병렬화할 수 있는지 -- 이해하는 것이 모든 루프 최적화의 기반입니다.

---

## 15. 연습 문제

### 연습 1: 지배자 계산

다음 CFG에 대해 각 노드의 지배자 집합을 계산하세요:

```
Entry -> A -> B -> C
         |         |
         v         v
         D -> E -> F -> Exit
              ^
              |
              G <-- F
```

엣지: Entry->A, A->B, A->D, B->C, C->F, D->E, E->F, F->Exit, F->G, G->E

(a) 각 노드의 지배자 집합을 계산하세요.
(b) 지배자 트리를 그리세요.
(c) 모든 후방 엣지와 자연 루프를 식별하세요.

### 연습 2: 루프 불변 코드 이동

다음 루프 (의사 코드)를 고려하세요:

```
i = 0
while i < n:
    t1 = a + b
    t2 = t1 * c
    t3 = d[i] + t2
    t4 = i * t1
    e[i] = t3 + t4
    i = i + 1
```

(a) 모든 루프 불변 계산을 식별하세요.
(b) 어떤 불변 계산을 안전하게 끌어올릴 수 있나요? 이유는 무엇인가요?
(c) LICM 적용 후 최적화된 코드를 작성하세요.

### 연습 3: 강도 감소

다음 루프를 고려하세요:

```python
for i in range(0, N):
    addr1 = base1 + i * 8
    addr2 = base2 + i * 12
    mem[addr1] = mem[addr2]
```

(a) 모든 기본 및 파생 귀납 변수를 식별하세요.
(b) 강도 감소를 적용하세요. 결과 코드를 보여주세요.
(c) 가능하다면 원래 카운터를 제거하기 위해 선형 테스트 교체를 적용하세요.

### 연습 4: 루프 타일링

행렬 전치(matrix transposition)를 고려하세요:

```python
for i in range(N):
    for j in range(N):
        B[j][i] = A[i][j]
```

(a) 이 코드의 캐시 동작을 분석하세요 (행 우선 저장 가정).
(b) 타일 크기 $T = 32$로 루프 타일링을 적용하세요. 타일링된 코드를 작성하세요.
(c) 32 KB의 L1 캐시와 8바이트 원소에 대해 최적 타일 크기는 무엇인가요?

### 연습 5: 벡터화 분석

아래의 각 루프에 대해 벡터화 가능 여부를 결정하세요. 불가능하다면 이유를 설명하고 도움이 될 수 있는 변환을 제안하세요:

```python
# (a)
for i in range(n):
    a[i] = b[i] * c[i] + d[i]

# (b)
for i in range(1, n):
    a[i] = a[i-1] + b[i]

# (c)
for i in range(n):
    if a[i] > 0:
        b[i] = a[i] * 2
    else:
        b[i] = 0

# (d)
for i in range(0, n, 2):
    a[i] = b[i] + c[i]
    a[i+1] = b[i+1] - c[i+1]
```

### 연습 6: 다면체 변환

다음 루프 중첩을 고려하세요:

```python
for i in range(1, N):
    for j in range(1, M):
        A[i][j] = A[i-1][j] + A[i][j-1]
```

(a) 반복 도메인을 2D 격자로 그리세요.
(b) 모든 데이터 의존성과 거리 벡터를 식별하세요.
(c) 루프 교환이 합법적인가요? 근거를 제시하세요.
(d) 기울임(skewing) 변환 $\theta(i, j) = (i + j, j)$를 제안하고 의존성을 존중하는지 검증하세요.
(e) 기울인 스케줄이 어떻게 파면(wavefront) 병렬성을 가능하게 하는가?

---

## 16. 참고 문헌

1. Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.), Chapters 9-10.
2. Cooper, K. D., & Torczon, L. (2011). *Engineering a Compiler* (2nd ed.), Chapters 8-10.
3. Muchnick, S. S. (1997). *Advanced Compiler Design and Implementation*, Chapters 14-18.
4. Allen, R., & Kennedy, K. (2001). *Optimizing Compilers for Modern Architectures*.
5. Wolfe, M. (1996). *High Performance Compilers for Parallel Computing*.
6. Bondhugula, U., et al. (2008). "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer." *PLDI*.
7. Cooper, K. D., Harvey, T. J., & Kennedy, K. (2001). "A Simple, Fast Dominance Algorithm." *Software Practice and Experience*.

---

[Previous: 12. Optimization -- Local and Global](./12_Optimization_Local_and_Global.md) | [Next: 14. Garbage Collection](./14_Garbage_Collection.md) | [Overview](./00_Overview.md)
