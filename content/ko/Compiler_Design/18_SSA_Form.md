# 정적 단일 할당 형식(Static Single Assignment Form)

**이전**: [16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md) | **다음**: [19. 레지스터 할당](./19_Register_Allocation.md)

---

정적 단일 할당(SSA, Static Single Assignment) 형식은 현대 최적화 컴파일러에서 가장 중요한 중간 표현입니다. SSA 형식에서는 모든 변수가 정확히 한 번만 할당되며, 변수의 모든 사용은 정확히 하나의 정의를 참조합니다. 이 속성은 많은 컴파일러 분석과 최적화를 극적으로 단순화합니다.

LLVM IR, GCC의 GIMPLE, HotSpot의 Sea-of-Nodes, 그리고 사실상 모든 현대 컴파일러가 SSA를 중심 IR로 사용합니다. 이 레슨에서는 SSA 구성의 이론, 파이 함수(phi function)의 역할, 지배 관계(dominance relationship), 그리고 SSA가 어떻게 강력한 최적화를 가능하게 하는지 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [09. 중간 표현](./09_Intermediate_Representations.md), [12. 최적화 -- 지역 및 전역](./12_Optimization_Local_and_Global.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일 할당 속성과 그것이 최적화를 단순화하는 이유를 설명한다
2. 파이 함수를 사용하여 일반 코드를 SSA 형식으로 변환한다
3. 지배 트리(dominator tree)와 지배 프론티어(dominance frontier)를 계산한다
4. Cytron 등의 SSA 구성 알고리즘을 구현한다
5. SSA 형식에서 최적화를 수행한다(상수 전파, 죽은 코드 제거)
6. 코드 생성을 위해 SSA 형식을 해체한다

---

## 목차

1. [SSA 형식이란?](#1-ssa-형식이란)
2. [파이 함수](#2-파이-함수)
3. [지배](#3-지배)
4. [지배 프론티어](#4-지배-프론티어)
5. [SSA 구성 알고리즘](#5-ssa-구성-알고리즘)
6. [SSA에서의 최적화](#6-ssa에서의-최적화)
7. [SSA 해체](#7-ssa-해체)
8. [가지치기된 SSA와 반-가지치기된 SSA](#8-가지치기된-ssa와-반-가지치기된-ssa)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. SSA 형식이란?

### 1.1 단일 할당 속성(The Single-Assignment Property)

일반적인 3-주소 코드(three-address code)에서 변수는 여러 번 할당될 수 있습니다:

```
x = 1
x = x + 1
y = x * 2
x = y + x
```

SSA 형식에서는 각 할당이 변수의 새로운 버전을 생성합니다:

```
x1 = 1
x2 = x1 + 1
y1 = x2 * 2
x3 = y1 + x2
```

모든 변수 이름은 왼쪽(정의 위치)에 정확히 한 번만 나타납니다. 이는 def-use 체인이 자명하게 사용 가능하다는 것을 의미합니다: `x2`의 어떤 사용이든, 그것이 `x2 = x1 + 1`로 정의되었음을 즉시 알 수 있습니다.

### 1.2 SSA가 중요한 이유

SSA 없이는 도달 정의(reaching definitions) 분석에 반복적인 데이터 흐름 방정식이 필요합니다. SSA를 사용하면 핵심 정보가 IR에 직접 인코딩됩니다:

| 속성 | SSA 없이 | SSA 사용 |
|------|---------|---------|
| 어떤 정의가 사용에 도달하는가? | 도달 정의 분석 필요 | 변수 이름에서 즉시 알 수 있음 |
| 변수가 살아있는가? | 생존 분석 필요 | 사용이 존재하는지 확인 |
| 상수 전파 | CFG에서 반복적 워크리스트 | SSA 그래프에서 단순 워크리스트 |
| 죽은 코드 제거 | 생존 + 도달 정의 필요 | use-count = 0 확인 |

### 1.3 직선 프로그램에서의 SSA

직선 코드(straight-line code) 변환은 단순합니다 -- 그냥 이름을 바꾸면 됩니다:

```python
# Original
a = 5
b = a + 3
a = b * 2
c = a + b

# SSA
a1 = 5
b1 = a1 + 3
a2 = b1 * 2
c1 = a2 + b1
```

문제는 **제어 흐름 합류점(control-flow merge point)**에서 발생합니다. 여기서 변수가 서로 다른 경로의 서로 다른 정의에서 올 수 있기 때문입니다.

---

## 2. 파이 함수

### 2.1 합류 문제(The Merge Problem)

다음 제어 흐름 그래프(control flow graph)를 고려해 봅시다:

```
        ┌──── B1 ────┐
        │  x = 1      │
        │  if cond     │
        └──┬───────┬──┘
           │       │
     ┌──── B2 ──┐  ┌──── B3 ──┐
     │  x = 2   │  │  x = 3   │
     └────┬─────┘  └────┬─────┘
           │              │
        ┌──┴──── B4 ──────┴──┐
        │  y = x + 1         │
        └────────────────────┘
```

블록 B4에서 어떤 `x`의 정의가 사용에 도달하는가? 런타임 경로에 따라 달라집니다. SSA는 이를 처리하기 위해 **파이 함수(phi function)**(그리스 문자 파이로 표기)를 도입합니다:

```
B4:
  x3 = phi(x1:B2, x2:B3)    // x3 gets x1 if from B2, x2 if from B3
  y1 = x3 + 1
```

### 2.2 파이 함수 의미론(Phi Function Semantics)

파이 함수 `x = phi(x1:B1, x2:B2, ..., xn:Bn)`의 의미:

- 제어가 블록 `Bi`에서 흘러온 경우 `x`의 값은 `xi`
- 파이 함수는 블록의 시작에서 "동시에" 실행됨
- 실제 명령어가 아님 -- 레지스터 할당을 안내하며 코드 생성 전에 제거됨

```python
# Phi function as a conceptual select
def phi_semantics(predecessors, values):
    """
    phi(x1:B1, x2:B2) means:
    - if we came from B1, the value is x1
    - if we came from B2, the value is x2
    """
    for pred, val in zip(predecessors, values):
        if came_from(pred):
            return val
```

### 2.3 복수 파이 함수(Multiple Phi Functions)

블록은 여러 파이 함수를 가질 수 있습니다:

```
B4:
  x3 = phi(x1:B2, x2:B3)
  y3 = phi(y1:B2, y2:B3)
  z1 = x3 + y3
```

블록 내의 모든 파이 함수는 블록의 다른 어떤 명령어보다 먼저 개념적으로 동시에 실행됩니다.

### 2.4 루프에서의 파이 함수

루프는 루프 헤더(loop header)에서 파이 함수를 생성합니다:

```
// Original:
// x = 0
// while x < 10:
//     x = x + 1

B1: x1 = 0
    goto B2

B2: x2 = phi(x1:B1, x3:B3)    // merge initial and loop-back
    if x2 < 10 goto B3 else B4

B3: x3 = x2 + 1
    goto B2

B4: // x2 is the final value
```

---

## 3. 지배(Dominance)

### 3.1 정의

노드 **d가 노드 n을 지배한다**(d dom n)는 진입 노드(entry node)에서 n까지의 모든 경로가 d를 통과해야 함을 의미합니다. 모든 노드는 자기 자신을 지배합니다.

노드 **d가 n을 엄격하게 지배한다**(d sdom n)는 d dom n이고 d ≠ n인 경우입니다.

**직접 지배자(immediate dominator, idom)**는 가장 가까운 엄격한 지배자입니다 -- d sdom n이고, n을 엄격하게 지배하는 다른 모든 d'에 대해 d' dom d가 성립하는 유일한 노드 d입니다.

### 3.2 지배 트리(Dominator Trees)

직접 지배자 관계는 진입 노드를 루트로 하는 트리를 형성합니다:

```
CFG:                    Dominator Tree:
    ┌─ B1 ─┐                B1
    ▼      ▼               / \
   B2     B3              B2  B3
    │      │              |
    ▼      │              B4
   B4 ◀───┘
    │
    ▼
   B5                     B5
```

이 트리에서:
- B1이 루트(진입)
- B1이 B2와 B3을 직접 지배
- B2가 B4를 직접 지배(모든 경로에서 B4에 도달하려면 B2를 거쳐야 한다고 가정)
- B4가 B5를 직접 지배

### 3.3 지배자 계산(Computing Dominators)

Cooper, Harvey, Kennedy(2001)의 고전적 알고리즘은 단순하고 효율적입니다:

```python
def compute_dominators(cfg, entry):
    """
    Compute immediate dominators using the iterative algorithm.
    cfg: dict mapping block -> list of predecessors
    entry: entry block name
    Returns: dict mapping block -> immediate dominator
    """
    blocks = list(cfg.keys())
    idom = {b: None for b in blocks}
    idom[entry] = entry

    # Assign reverse postorder numbers
    rpo = reverse_postorder(cfg, entry)
    rpo_number = {b: i for i, b in enumerate(rpo)}

    def intersect(b1, b2):
        """Find common dominator of b1 and b2."""
        finger1, finger2 = b1, b2
        while finger1 != finger2:
            while rpo_number[finger1] > rpo_number[finger2]:
                finger1 = idom[finger1]
            while rpo_number[finger2] > rpo_number[finger1]:
                finger2 = idom[finger2]
        return finger1

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry:
                continue
            # Pick first processed predecessor as initial idom
            preds = [p for p in cfg[b] if idom[p] is not None]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(new_idom, p)
            if idom[b] != new_idom:
                idom[b] = new_idom
                changed = True

    return idom


def reverse_postorder(cfg, entry):
    """Compute reverse postorder traversal of CFG."""
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        for succ in get_successors(cfg, node):
            if succ not in visited:
                dfs(succ)
        order.append(node)

    dfs(entry)
    return list(reversed(order))
```

### 3.4 지배 트리 속성

SSA 구성에 사용되는 핵심 속성:

1. **d dom n이면, d는 진입에서 n까지의 모든 경로에 나타남**
2. **지배 관계는 추이적(transitive)**: a dom b이고 b dom c이면, a dom c
3. **지배 트리는 주어진 CFG에 대해 유일함**
4. **루프에서, 루프 헤더는 루프 본체의 모든 블록을 지배함**

---

## 4. 지배 프론티어(Dominance Frontiers)

### 4.1 정의

노드 d의 **지배 프론티어(dominance frontier)** DF(d)는 다음과 같은 노드 n의 집합입니다:
- d가 n의 선행자(predecessor)를 지배하지만
- d가 n을 엄격하게 지배하지 않음

직관적으로, d의 지배 프론티어는 d의 "지배가 끝나는" 지점입니다 -- d에서 도달 가능하지만 d가 지배하지 않는 첫 번째 노드들입니다.

```
        B1 (entry)
       / \
      B2   B3
       \  /
        B4
```

- DF(B2) = {B4} -- B2가 B4의 선행자(즉, B2 자체)를 지배하지만, B2는 B4를 엄격하게 지배하지 않음
- DF(B3) = {B4} -- 마찬가지
- DF(B1) = {} -- B1은 모든 것을 지배

### 4.2 지배 프론티어가 SSA에 중요한 이유

**핵심 통찰**: 파이 함수는 정확히 지배 프론티어 노드에서 필요합니다.

블록 B가 변수 x를 정의하면, DF(B)의 모든 노드에서 x에 대한 파이 함수가 필요합니다. 이는 DF(B)가 x의 대안적 정의가 B에서의 정의와 "합류"할 수 있는 정확한 합류점을 포함하기 때문입니다.

### 4.3 지배 프론티어 계산

```python
def compute_dominance_frontiers(cfg, idom):
    """
    Compute dominance frontier for each block.
    Uses the algorithm from Cytron et al. (1991).
    """
    df = {b: set() for b in cfg}

    for b in cfg:
        preds = cfg[b]  # predecessors of b
        if len(preds) >= 2:  # b is a merge point
            for p in preds:
                runner = p
                while runner != idom[b]:
                    df[runner].add(b)
                    runner = idom[runner]

    return df
```

### 4.4 반복 지배 프론티어(Iterated Dominance Frontier)

파이 함수를 배치할 때, **반복 지배 프론티어(IDF, Iterated Dominance Frontier)**가 필요합니다. 파이 함수 자체가 새로운 정의이므로, 자신의 지배 프론티어에서 추가적인 파이 함수가 필요할 수 있습니다:

```python
def iterated_dominance_frontier(df, defs):
    """
    Compute IDF for a set of defining blocks.
    defs: set of blocks that define a variable
    """
    worklist = list(defs)
    idf = set()

    while worklist:
        b = worklist.pop()
        for frontier_node in df[b]:
            if frontier_node not in idf:
                idf.add(frontier_node)
                worklist.append(frontier_node)

    return idf
```

---

## 5. SSA 구성 알고리즘

### 5.1 Cytron 등의 알고리즘

표준 SSA 구성 알고리즘(Cytron, Ferrante, Rosen, Wegman, Zadeck, 1991)은 두 단계로 구성됩니다:

**1단계: 파이 함수 배치** -- 지배 프론티어를 사용합니다.
**2단계: 변수 이름 변경** -- 지배 트리 순회를 사용합니다.

### 5.2 1단계: 파이 함수 배치

```python
def place_phi_functions(cfg, idom, df, var_defs):
    """
    Place phi functions for each variable.
    var_defs: dict mapping variable -> set of blocks that define it
    Returns: dict mapping (block, variable) -> phi function
    """
    phi_functions = {}

    for var, def_blocks in var_defs.items():
        # Compute iterated dominance frontier
        idf = iterated_dominance_frontier(df, def_blocks)

        for block in idf:
            preds = cfg[block]
            phi_functions[(block, var)] = {
                'var': var,
                'args': {pred: None for pred in preds}  # filled in Phase 2
            }

    return phi_functions
```

### 5.3 2단계: 변수 이름 변경

```python
def rename_variables(cfg, idom, phi_functions, blocks_instructions):
    """
    Rename variables to create SSA form.
    Walk the dominator tree, maintaining a stack of names for each variable.
    """
    counter = {}       # variable -> next version number
    stack = {}         # variable -> stack of current SSA names

    def new_name(var):
        """Generate a fresh SSA name for var."""
        if var not in counter:
            counter[var] = 0
            stack[var] = []
        i = counter[var]
        counter[var] = i + 1
        name = f"{var}_{i}"
        stack[var].append(name)
        return name

    def current_name(var):
        """Get the current SSA name for var."""
        if var not in stack or not stack[var]:
            return f"{var}_undef"
        return stack[var][-1]

    def rename_block(block):
        # Track how many names we push (to pop later)
        push_count = {var: 0 for var in stack}

        # 1. Rename phi function destinations
        for var in get_phi_vars(phi_functions, block):
            name = new_name(var)
            phi_functions[(block, var)]['dest'] = name
            push_count.setdefault(var, 0)
            push_count[var] = push_count.get(var, 0) + 1

        # 2. Rename instructions
        for instr in blocks_instructions[block]:
            # Rename uses (right-hand side)
            for use_var in instr.uses:
                instr.replace_use(use_var, current_name(use_var))
            # Rename definition (left-hand side)
            if instr.defines:
                old_def = instr.defines
                new_def = new_name(old_def)
                instr.defines = new_def
                push_count.setdefault(old_def, 0)
                push_count[old_def] = push_count.get(old_def, 0) + 1

        # 3. Fill phi function arguments in successors
        for succ in get_successors(cfg, block):
            for var in get_phi_vars(phi_functions, succ):
                phi_functions[(succ, var)]['args'][block] = current_name(var)

        # 4. Recurse into dominator tree children
        for child in dominator_tree_children(idom, block):
            rename_block(child)

        # 5. Pop names
        for var, count in push_count.items():
            for _ in range(count):
                if stack.get(var):
                    stack[var].pop()

    rename_block(entry_block(cfg))
```

### 5.4 완전한 예제

```python
# Original program:
#   a = read()
#   b = read()
#   if a > b:
#       c = a - b
#   else:
#       c = b - a
#   print(c)

# CFG:
# B1: a = read(); b = read(); if a > b goto B2 else B3
# B2: c = a - b; goto B4
# B3: c = b - a; goto B4
# B4: print(c)

# Dominance: idom(B2)=B1, idom(B3)=B1, idom(B4)=B1
# DF(B2) = {B4}, DF(B3) = {B4}
# Variable c defined in B2, B3 -> phi for c at B4

# SSA form:
# B1: a1 = read(); b1 = read(); if a1 > b1 goto B2 else B3
# B2: c1 = a1 - b1; goto B4
# B3: c2 = b1 - a1; goto B4
# B4: c3 = phi(c1:B2, c2:B3); print(c3)
```

---

## 6. SSA에서의 최적화

### 6.1 희소 조건부 상수 전파(Sparse Conditional Constant Propagation, SCCP)

SSA는 상수 전파(constant propagation)와 도달 불가능한 코드 감지의 강력한 조합을 가능하게 합니다:

```python
class SCCPSolver:
    """
    Sparse Conditional Constant Propagation on SSA form.
    Uses a lattice: TOP -> constant -> BOTTOM
    """
    TOP = "TOP"         # undefined / not yet analyzed
    BOTTOM = "BOTTOM"   # multiple possible values

    def __init__(self, ssa_program):
        self.program = ssa_program
        self.lattice = {}       # variable -> lattice value
        self.cfg_worklist = []  # edges to process
        self.ssa_worklist = []  # SSA edges (def->use) to process
        self.executable = set() # executable CFG edges

    def meet(self, a, b):
        """Lattice meet operation."""
        if a == self.TOP:
            return b
        if b == self.TOP:
            return a
        if a == b:
            return a
        return self.BOTTOM

    def evaluate(self, instr):
        """Evaluate instruction given current lattice values."""
        if instr.is_phi:
            result = self.TOP
            for pred, var in instr.phi_args.items():
                edge = (pred, instr.block)
                if edge in self.executable:
                    result = self.meet(result, self.lattice.get(var, self.TOP))
            return result
        # ... evaluate arithmetic, comparisons, etc.

    def solve(self):
        """Run SCCP to fixpoint."""
        # Initialize: mark entry edge executable
        self.cfg_worklist.append(("entry", self.program.entry))

        while self.cfg_worklist or self.ssa_worklist:
            while self.cfg_worklist:
                edge = self.cfg_worklist.pop()
                if edge not in self.executable:
                    self.executable.add(edge)
                    target = edge[1]
                    # Process phi functions in target
                    for phi in self.program.get_phis(target):
                        new_val = self.evaluate(phi)
                        if new_val != self.lattice.get(phi.dest, self.TOP):
                            self.lattice[phi.dest] = new_val
                            self.ssa_worklist.extend(phi.dest_uses())

            while self.ssa_worklist:
                use_instr = self.ssa_worklist.pop()
                new_val = self.evaluate(use_instr)
                if new_val != self.lattice.get(use_instr.dest, self.TOP):
                    self.lattice[use_instr.dest] = new_val
                    self.ssa_worklist.extend(use_instr.dest_uses())
```

### 6.2 전역 값 번호 매기기(Global Value Numbering, GVN)

SSA는 중복 계산(redundant computation)을 쉽게 감지할 수 있게 합니다:

```python
def global_value_numbering(ssa_block):
    """
    Assign value numbers to SSA variables.
    If two expressions have the same value number, they compute the same value.
    """
    value_table = {}  # (op, vn1, vn2) -> value number
    var_to_vn = {}    # variable -> value number
    next_vn = [0]

    def get_vn(var):
        if var in var_to_vn:
            return var_to_vn[var]
        # Constants get unique value numbers
        vn = next_vn[0]
        next_vn[0] += 1
        var_to_vn[var] = vn
        return vn

    for instr in ssa_block:
        if instr.is_binary_op:
            vn1 = get_vn(instr.left)
            vn2 = get_vn(instr.right)
            key = (instr.op, vn1, vn2)

            # Check for commutative operations
            if instr.op in ('+', '*'):
                key = (instr.op, min(vn1, vn2), max(vn1, vn2))

            if key in value_table:
                # Redundant! Reuse existing value
                var_to_vn[instr.dest] = value_table[key]
                instr.replace_with_copy(value_table[key])
            else:
                vn = next_vn[0]
                next_vn[0] += 1
                value_table[key] = vn
                var_to_vn[instr.dest] = vn

    return var_to_vn
```

### 6.3 SSA에서의 죽은 코드 제거(Dead Code Elimination)

```python
def aggressive_dead_code_elimination(ssa_program):
    """
    Mark-sweep style DCE on SSA form.
    Start from essential instructions (I/O, stores, branches),
    mark everything they depend on as live.
    """
    live = set()
    worklist = []

    # Mark essential instructions
    for block in ssa_program.blocks:
        for instr in block.instructions:
            if instr.has_side_effect or instr.is_branch:
                live.add(instr)
                worklist.append(instr)

    # Propagate liveness along SSA def-use chains
    while worklist:
        instr = worklist.pop()
        for use_var in instr.uses:
            def_instr = ssa_program.get_definition(use_var)
            if def_instr not in live:
                live.add(def_instr)
                worklist.append(def_instr)

    # Remove dead instructions
    for block in ssa_program.blocks:
        block.instructions = [i for i in block.instructions if i in live]
```

---

## 7. SSA 해체(SSA Destruction)

### 7.1 왜 SSA를 해체하는가?

기계 코드를 생성하기 전에 파이 함수를 제거해야 합니다. 파이 함수는 직접적인 하드웨어 대응물이 없기 때문입니다. SSA 해체는 파이 함수를 CFG 간선(edge)을 따라 배치된 복사 명령어(copy instruction)로 변환합니다.

### 7.2 나이브 접근법: 병렬 복사(Parallel Copies)

```
# SSA:
B3: x3 = phi(x1:B1, x2:B2)
    y3 = phi(y1:B1, y2:B2)

# After destruction (naive):
B1: ...
    x3 = x1    # copy at end of B1
    y3 = y1
    goto B3

B2: ...
    x3 = x2    # copy at end of B2
    y3 = y2
    goto B3

B3: // no phi functions, x3 and y3 are ready
```

### 7.3 분실 복사 문제(The Lost-Copy Problem)

나이브한 삽입은 파이 소스가 복사 전에 수정되는 경우 잘못된 코드를 생성할 수 있습니다:

```
# SSA:
B2: x2 = phi(x1:B1, x3:B2)
    x3 = x2 + 1
    if x3 < 10 goto B2 else B3

# Naive destruction (INCORRECT):
B2: x2 = x3     # but x3 hasn't been computed yet on first iteration!
    x3 = x2 + 1
    if x3 < 10 goto B2 else B3
```

### 7.4 정확한 SSA 해체

해결책은 **임계 간선 분할(critical edge splitting)**과 신중한 복사 순서 지정을 사용합니다:

```python
def destroy_ssa(ssa_program):
    """
    Convert out of SSA by replacing phi functions with copies.
    Handles the lost-copy and swap problems correctly.
    """
    for block in ssa_program.blocks:
        phis = [i for i in block.instructions if i.is_phi]
        if not phis:
            continue

        for phi in phis:
            for pred, src_var in phi.args.items():
                dest_var = phi.dest
                if src_var != dest_var:
                    # Insert copy at end of predecessor (before branch)
                    # May need to split critical edges first
                    if is_critical_edge(pred, block):
                        new_block = split_edge(pred, block)
                        new_block.append_copy(dest_var, src_var)
                    else:
                        pred.insert_copy_before_branch(dest_var, src_var)

        # Remove phi functions
        block.instructions = [i for i in block.instructions if not i.is_phi]


def is_critical_edge(pred, succ):
    """An edge is critical if pred has multiple successors and succ has multiple predecessors."""
    return len(pred.successors) > 1 and len(succ.predecessors) > 1


def split_edge(pred, succ):
    """Insert a new empty block on the edge from pred to succ."""
    new_block = BasicBlock(f"split_{pred.name}_{succ.name}")
    new_block.add_successor(succ)
    pred.replace_successor(succ, new_block)
    succ.replace_predecessor(pred, new_block)
    return new_block
```

---

## 8. 가지치기된 SSA와 반-가지치기된 SSA

### 8.1 최소 SSA(Minimal SSA)

Cytron 알고리즘은 **최소 SSA(minimal SSA)** -- 정확성에 필요한 최소한의 파이 함수를 생성합니다. 하지만 이 파이 함수 중 일부는 죽어있을 수 있습니다(값이 사용되지 않음).

### 8.2 가지치기된 SSA(Pruned SSA)

**가지치기된 SSA(Pruned SSA)**는 SSA 구성과 생존 분석(liveness analysis)을 결합하여 죽은 파이 함수를 제거합니다:

```python
def build_pruned_ssa(cfg, idom, df, var_defs, live_in):
    """
    Build pruned SSA: only place phi where variable is live-in.
    live_in: dict mapping block -> set of variables live at block entry
    """
    phi_functions = {}

    for var, def_blocks in var_defs.items():
        idf = iterated_dominance_frontier(df, def_blocks)
        for block in idf:
            if var in live_in[block]:  # Only if live!
                preds = cfg[block]
                phi_functions[(block, var)] = {
                    'var': var,
                    'args': {pred: None for pred in preds}
                }

    return phi_functions
```

### 8.3 반-가지치기된 SSA(Semi-Pruned SSA)

반-가지치기된 SSA는 실용적인 절충안입니다: 전체 생존 분석 없이, 단일 블록 내에서만 사용되는 변수(블록 경계를 넘어 살아있지 않는 변수)에 대한 파이 함수를 제거합니다.

```python
def build_semi_pruned_ssa(cfg, idom, df, var_defs, block_local_vars):
    """
    Semi-pruned SSA: skip phi placement for block-local variables.
    block_local_vars: set of variables only used in their defining block
    """
    phi_functions = {}

    for var, def_blocks in var_defs.items():
        if var in block_local_vars:
            continue  # No phi needed for block-local vars

        idf = iterated_dominance_frontier(df, def_blocks)
        for block in idf:
            preds = cfg[block]
            phi_functions[(block, var)] = {
                'var': var,
                'args': {pred: None for pred in preds}
            }

    return phi_functions
```

### 8.4 SSA 변형 비교

| 변형 | 파이 수 | 추가 분석 필요 | 사용 사례 |
|------|---------|--------------|----------|
| 최대(Maximal) | 많음 (모든 합류점의 모든 변수) | 없음 | 이론적으로만 |
| 최소(Minimal) | 정확성을 위한 최소 | 지배 프론티어 | 표준 |
| 반-가지치기(Semi-Pruned) | 더 적음 | 블록-로컬 식별 | 빠른 컴파일 |
| 가지치기(Pruned) | 유용한 최소 | 전체 생존 분석 | 공격적 최적화 |

---

## 9. 요약

- **SSA 형식**은 각 변수를 정확히 한 번만 할당하여 def-use 체인을 명시적으로 만듭니다
- **파이 함수**는 제어 흐름 합류점에서 값을 병합합니다
- **지배**와 **지배 프론티어**가 파이 함수가 필요한 위치를 결정합니다
- **Cytron 등의 알고리즘**은 파이 배치와 이름 변경의 두 단계로 SSA를 구성합니다
- SSA는 SCCP, GVN, 공격적 DCE 등 강력한 최적화를 가능하게 합니다
- **SSA 해체**는 임계 간선을 올바르게 처리하면서 파이 함수를 복사로 대체합니다
- **가지치기된 SSA**는 생존 정보를 사용하여 죽은 파이 함수를 방지합니다

---

## 10. 연습 문제

1. **수동 SSA 변환**: 다음 코드를 수동으로 SSA 형식으로 변환하고, 모든 파이 함수를 보여주세요:
   ```
   x = 0
   y = 1
   while x < 10:
       if x % 2 == 0:
           y = y + x
       else:
           y = y * 2
       x = x + 1
   print(y)
   ```

2. **지배 트리**: 블록 B1-B7이 있는 CFG가 주어지면, 지배 트리와 지배 프론티어를 계산하세요.

3. **SSA 구성 구현**: 간단한 3-주소 코드 프로그램을 받아 SSA 형식으로 변환하는 Python 프로그램을 작성하세요.

4. **SSA에서의 SCCP**: 알려진 상수 입력이 있는 SSA 프로그램에 희소 조건부 상수 전파를 적용하세요.

5. **SSA 해체**: 파이 함수가 있는 SSA 프로그램이 주어지면, 임계 간선을 식별하고 분할하여 올바른 복사 명령어를 생성하세요.

---

## 11. 참고 자료

1. Cytron, R., Ferrante, J., Rosen, B., Wegman, M., Zadeck, F. K. (1991). "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph." *ACM Transactions on Programming Languages and Systems*, 13(4), 451-490.
2. Cooper, K. D., Harvey, T. J., Kennedy, K. (2001). "A Simple, Fast Dominance Algorithm." *Software Practice and Experience*.
3. Appel, A. W. (1998). "SSA is Functional Programming." *ACM SIGPLAN Notices*, 33(4), 17-20.
4. Briggs, P., Cooper, K., Harvey, T., Simpson, T. (1998). "Practical Improvements to the Construction and Destruction of Static Single Assignment Form." *Software Practice and Experience*, 28(8), 859-881.
5. Wegman, M. N., Zadeck, F. K. (1991). "Constant Propagation with Conditional Branches." *ACM Transactions on Programming Languages and Systems*, 13(2), 181-210.

---

**이전**: [16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md) | **다음**: [19. 레지스터 할당](./19_Register_Allocation.md)
