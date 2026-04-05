# 레슨 11: 코드 생성(Code Generation)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. **설명하기**: 코드 생성에 사용되는 목표 기계 모델(target machine model)
2. **설명하기**: 트리 패턴 매칭(tree pattern matching) 및 타일링(tiling)을 통한 명령어 선택(instruction selection)
3. **구현하기**: 명령어 선택을 위한 최대 한입 알고리즘(Maximal Munch algorithm)
4. **적용하기**: 레지스터 할당(register allocation) 기법: 그래프 색칠(graph coloring) 및 선형 스캔(linear scan)
5. **이해하기**: 명령어 스케줄링(instruction scheduling): 리스트 스케줄링(list scheduling), 소프트웨어 파이프라이닝(software pipelining) 개요
6. **수행하기**: 생성된 코드에 대한 핍홀 최적화(peephole optimization)
7. **생성하기**: 식, 제어 흐름, 함수 호출에 대한 코드
8. **구현하기**: Python으로 스택 기계(stack machine)용 완전한 코드 생성기

---

## 1. 목표 기계 모델(Target Machine Model)

### 1.1 기계 모델이 필요한 이유

코드 생성은 중간 표현(IR, Intermediate Representation)을 특정 목표 기계(target machine)의 명령어로 변환합니다. 기계에 독립적인 방식으로 코드 생성 기법을 연구하기 위해, 실제 아키텍처의 핵심 특징을 담은 추상적인 목표 기계 모델을 정의합니다.

### 1.2 간단한 목표 기계

우리의 모델 기계는 다음과 같은 특성을 가집니다:

| 특성 | 설명 |
|------|------|
| 레지스터(Registers) | $R_0, R_1, \ldots, R_{k-1}$ (범용) |
| 메모리(Memory) | 바이트 주소 지정, 워드 크기 = 4바이트 |
| 명령어(Instructions) | 3주소 형식: `op dst, src1, src2` |
| 주소 지정 모드(Addressing modes) | 레지스터, 즉시값, 레지스터 간접, 인덱스 |
| 스택(Stack) | 아래 방향으로 증가, SP 레지스터 |

### 1.3 주소 지정 모드(Addressing Modes)

| 모드 | 구문 | 의미 | 사용 사례 |
|------|------|------|-----------|
| 레지스터(Register) | `R0` | R0 레지스터의 값 | 레지스터의 지역 변수 |
| 즉시값(Immediate) | `#42` | 상수 값 42 | 상수, 오프셋 |
| 레지스터 간접(Register-indirect) | `[R0]` | R0에 저장된 주소의 메모리 | 포인터 역참조 |
| 인덱스(Indexed) | `[R0 + #8]` | R0 + 8 주소의 메모리 | 배열 접근, 구조체 필드 |
| 직접(Direct) | `[addr]` | 절대 주소의 메모리 | 전역 변수 |

### 1.4 명령어 집합(Instruction Set)

```
Arithmetic:
    ADD  Rd, Rs1, Rs2    ; Rd = Rs1 + Rs2
    SUB  Rd, Rs1, Rs2    ; Rd = Rs1 - Rs2
    MUL  Rd, Rs1, Rs2    ; Rd = Rs1 * Rs2
    DIV  Rd, Rs1, Rs2    ; Rd = Rs1 / Rs2
    ADDI Rd, Rs, #imm    ; Rd = Rs + imm

Data movement:
    MOV  Rd, Rs          ; Rd = Rs
    MOVI Rd, #imm        ; Rd = imm (load immediate)
    LOAD Rd, [Rs + #off] ; Rd = Mem[Rs + off]
    STORE Rs, [Rd + #off]; Mem[Rd + off] = Rs

Control flow:
    CMP  Rs1, Rs2        ; Set condition flags
    BEQ  label           ; Branch if equal
    BNE  label           ; Branch if not equal
    BLT  label           ; Branch if less than
    BGT  label           ; Branch if greater than
    BLE  label           ; Branch if less or equal
    BGE  label           ; Branch if greater or equal
    JMP  label           ; Unconditional jump
    CALL label           ; Function call (push return address, jump)
    RET                  ; Return (pop return address, jump)

Stack:
    PUSH Rs              ; Push register onto stack
    POP  Rd              ; Pop from stack into register
```

### 1.5 명령어 비용(Instruction Costs)

명령어마다 비용(사이클 수)이 다릅니다:

| 명령어 | 비용 | 참고 |
|--------|------|------|
| ADD, SUB, MOV | 1 | 레지스터-레지스터 연산 |
| MUL | 3 | 곱셈은 더 느림 |
| DIV | 10-40 | 나눗셈은 매우 비쌈 |
| LOAD | 4 (L1 히트) | 메모리 접근 (캐시 의존) |
| STORE | 1 (버퍼링) | 쓰기 버퍼가 지연을 숨김 |
| Branch (taken) | 2 | 파이프라인 플러시 패널티 |
| Branch (not taken) | 1 | 올바르게 예측됨 |

코드 생성기는 가능한 경우 더 저렴한 명령어를 선호해야 합니다 (예: 2의 거듭제곱 곱셈 대신 시프트 사용).

---

## 2. 명령어 선택(Instruction Selection)

### 2.1 문제 정의

명령어 선택은 IR 연산을 목표 기계 명령어로 매핑합니다. 다음과 같은 도전이 있습니다:

1. 단일 IR 연산이 여러 기계 명령어에 대응될 수 있음
2. 단일 기계 명령어가 여러 IR 연산을 커버할 수 있음
3. 서로 다른 명령어 시퀀스가 서로 다른 비용으로 동일한 결과를 계산할 수 있음
4. 최적의 선택은 문맥(레지스터 가용성, 주변 명령어)에 따라 다름

### 2.2 트리 패턴 매칭(Tree Pattern Matching)

많은 아키텍처에는 복합 연산을 수행할 수 있는 명령어가 있습니다 (예: `LOAD R0, [R1 + R2*4 + #8]`은 덧셈, 곱셈, 메모리 로드를 하나의 명령어로 수행).

**트리 패턴 매칭(tree pattern matching)**의 아이디어는:

1. IR을 식 트리(expression tree)로 표현
2. 단일 기계 명령어에 대응하는 트리 패턴인 **타일(tile)** 정의
3. 총 비용을 최소화하는 비겹침 타일로 IR 트리를 **커버(cover)**

### 2.3 타일(Tiles)

**타일(tile)**은 기계 명령어와 비용이 쌍을 이루는 트리 패턴입니다.

우리의 목표 기계에 대한 예시 타일:

```
Tile 1: register                    Tile 2: immediate
Pattern:   reg                      Pattern:   #imm
Instr:     (none, already in reg)   Instr:     MOVI Rd, #imm
Cost:      0                        Cost:      1

Tile 3: add                         Tile 4: add immediate
Pattern:     +                      Pattern:     +
            / \                                 / \
          reg  reg                            reg  #imm
Instr:   ADD Rd, Rs1, Rs2          Instr:   ADDI Rd, Rs, #imm
Cost:    1                          Cost:    1

Tile 5: load                        Tile 6: indexed load
Pattern:   MEM                      Pattern:   MEM
            |                                   |
           reg                                  +
                                               / \
Instr:   LOAD Rd, [Rs]                       reg  #imm
Cost:    4                          Instr:   LOAD Rd, [Rs + #imm]
                                    Cost:    4

Tile 7: store                       Tile 8: indexed store
Pattern:  STORE                     Pattern:  STORE
          /   \                               /    \
        addr   reg                           +      reg
                                            / \
Instr:  STORE Rs, [Rd]                    reg  #imm
Cost:   1                           Instr:  STORE Rs, [Rd + #imm]
                                    Cost:   1
```

### 2.4 최적 타일링(Optimal Tiling) 대 최대 한입(Maximal Munch)

트리를 타일로 커버하는 두 가지 주요 전략이 있습니다:

**최적 타일링(Optimal Tiling)**: 최소 총 비용으로 전체 트리를 커버하는 타일 집합을 찾습니다. 트리에서 하향식 동적 프로그래밍으로 해결할 수 있습니다.

**최대 한입(Maximal Munch)** (탐욕적): 각 노드에서 루트부터 시작하여 가장 큰(가장 많이 커버하는) 타일을 선택합니다. 더 단순하지만 전역적으로 최적인 해를 찾지 못할 수 있습니다.

실제로는 최대 한입이 최적에 가까운 결과를 생성하며 널리 사용됩니다.

### 2.5 최대 한입 알고리즘(Maximal Munch Algorithm)

```
function maximal_munch(node):
    Find the largest tile that matches at this node
    (matching means the tile pattern matches the subtree rooted here)

    For each leaf of the selected tile that corresponds to a
    subtree (not yet covered):
        Recursively apply maximal_munch to that subtree

    Emit the instruction associated with the selected tile
```

이 알고리즘은 **하향식(top-down)**으로 동작합니다: 루트에서 가장 큰 패턴을 매칭한 후, 나머지 하위 트리를 재귀적으로 처리합니다.

**예시**: `a[i] = b + 1`에 대한 코드 생성

IR 트리:
```
        STORE
       /     \
      +       +
     / \     / \
    a   *   b   1
       / \
      i   4
```

최대 한입이 매칭할 수 있는 패턴:
1. 루트에서: `STORE(addr, value)`를 커버하는 STORE 타일 -- STORE 명령어 방출
2. 왼쪽 자식 `+`: `a + (i * 4)` 또는 인덱스 주소 지정을 위한 ADD 타일
3. 오른쪽 자식 `+`: `b + 1`을 위한 ADDI 타일

### 2.6 Python 구현: 최대 한입(Maximal Munch)

```python
"""Instruction selection via Maximal Munch algorithm."""

from dataclasses import dataclass, field
from typing import Optional


# ---------- IR Tree Nodes ----------

@dataclass
class IRNode:
    """Base class for IR tree nodes."""
    pass

@dataclass
class Const(IRNode):
    """Integer constant."""
    value: int

@dataclass
class Reg(IRNode):
    """Virtual register reference."""
    name: str

@dataclass
class BinOp(IRNode):
    """Binary operation."""
    op: str        # '+', '-', '*', '/'
    left: IRNode
    right: IRNode

@dataclass
class Mem(IRNode):
    """Memory access (load)."""
    address: IRNode

@dataclass
class Store(IRNode):
    """Memory store."""
    address: IRNode
    value: IRNode

@dataclass
class CJump(IRNode):
    """Conditional jump."""
    op: str           # '<', '>', '==', '!=', '<=', '>='
    left: IRNode
    right: IRNode
    true_label: str
    false_label: str


# ---------- Machine Instructions ----------

@dataclass
class MachineInstr:
    """A target machine instruction."""
    opcode: str
    operands: list = field(default_factory=list)
    comment: str = ""

    def __str__(self):
        ops = ", ".join(str(o) for o in self.operands)
        comment = f"  ; {self.comment}" if self.comment else ""
        return f"    {self.opcode:6s} {ops}{comment}"


# ---------- Maximal Munch Code Generator ----------

class MaximalMunchCodeGen:
    """
    Instruction selection using the Maximal Munch (greedy) algorithm.
    Generates code for a simple RISC-like target.
    """

    def __init__(self):
        self.instructions: list[MachineInstr] = []
        self._reg_counter = 0

    def new_reg(self) -> str:
        """Allocate a new virtual register."""
        self._reg_counter += 1
        return f"v{self._reg_counter}"

    def emit(self, opcode: str, operands: list = None,
             comment: str = "") -> MachineInstr:
        """Emit a machine instruction."""
        instr = MachineInstr(opcode, operands or [], comment)
        self.instructions.append(instr)
        return instr

    def munch_expr(self, node: IRNode) -> str:
        """
        Generate code for an expression node using Maximal Munch.
        Returns the register holding the result.
        """

        # --- Tile: Constant ---
        if isinstance(node, Const):
            rd = self.new_reg()
            self.emit("MOVI", [rd, f"#{node.value}"],
                      f"load constant {node.value}")
            return rd

        # --- Tile: Register ---
        if isinstance(node, Reg):
            return node.name

        # --- Tile: Memory load with indexed addressing ---
        # Pattern: MEM(BinOp(+, e1, Const(n)))
        if isinstance(node, Mem) and isinstance(node.address, BinOp) \
           and node.address.op == '+' and isinstance(node.address.right, Const):
            base_reg = self.munch_expr(node.address.left)
            offset = node.address.right.value
            rd = self.new_reg()
            self.emit("LOAD", [rd, f"[{base_reg} + #{offset}]"],
                      f"indexed load")
            return rd

        # --- Tile: Memory load with register addressing ---
        # Pattern: MEM(e)
        if isinstance(node, Mem):
            addr_reg = self.munch_expr(node.address)
            rd = self.new_reg()
            self.emit("LOAD", [rd, f"[{addr_reg}]"],
                      "register-indirect load")
            return rd

        # --- Tile: Add immediate ---
        # Pattern: BinOp(+, e, Const(n))
        if isinstance(node, BinOp) and node.op == '+' \
           and isinstance(node.right, Const):
            rs = self.munch_expr(node.left)
            rd = self.new_reg()
            self.emit("ADDI", [rd, rs, f"#{node.right.value}"],
                      f"add immediate {node.right.value}")
            return rd

        # --- Tile: Add immediate (commuted) ---
        # Pattern: BinOp(+, Const(n), e)
        if isinstance(node, BinOp) and node.op == '+' \
           and isinstance(node.left, Const):
            rs = self.munch_expr(node.right)
            rd = self.new_reg()
            self.emit("ADDI", [rd, rs, f"#{node.left.value}"],
                      f"add immediate {node.left.value}")
            return rd

        # --- Tile: Multiply by power of 2 → shift ---
        if isinstance(node, BinOp) and node.op == '*' \
           and isinstance(node.right, Const) \
           and node.right.value > 0 \
           and (node.right.value & (node.right.value - 1)) == 0:
            rs = self.munch_expr(node.left)
            shift = node.right.value.bit_length() - 1
            rd = self.new_reg()
            self.emit("SHL", [rd, rs, f"#{shift}"],
                      f"multiply by {node.right.value} via shift")
            return rd

        # --- Tile: General binary operation ---
        if isinstance(node, BinOp):
            rs1 = self.munch_expr(node.left)
            rs2 = self.munch_expr(node.right)
            rd = self.new_reg()
            op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}
            opcode = op_map.get(node.op, 'BINOP')
            self.emit(opcode, [rd, rs1, rs2],
                      f"{node.op} operation")
            return rd

        raise ValueError(f"Cannot generate code for: {type(node)}")

    def munch_stmt(self, node: IRNode):
        """Generate code for a statement node."""

        # --- Tile: Indexed store ---
        # Pattern: Store(BinOp(+, e1, Const(n)), e2)
        if isinstance(node, Store) and isinstance(node.address, BinOp) \
           and node.address.op == '+' \
           and isinstance(node.address.right, Const):
            base_reg = self.munch_expr(node.address.left)
            val_reg = self.munch_expr(node.value)
            offset = node.address.right.value
            self.emit("STORE", [val_reg, f"[{base_reg} + #{offset}]"],
                      "indexed store")
            return

        # --- Tile: General store ---
        if isinstance(node, Store):
            addr_reg = self.munch_expr(node.address)
            val_reg = self.munch_expr(node.value)
            self.emit("STORE", [val_reg, f"[{addr_reg}]"],
                      "register-indirect store")
            return

        # --- Tile: Conditional jump ---
        if isinstance(node, CJump):
            rs1 = self.munch_expr(node.left)
            rs2 = self.munch_expr(node.right)
            self.emit("CMP", [rs1, rs2],
                      f"compare for {node.op}")
            branch_map = {
                '<': 'BLT', '>': 'BGT', '==': 'BEQ',
                '!=': 'BNE', '<=': 'BLE', '>=': 'BGE'
            }
            opcode = branch_map.get(node.op, 'BR')
            self.emit(opcode, [node.true_label],
                      f"branch to {node.true_label}")
            self.emit("JMP", [node.false_label],
                      f"fall-through to {node.false_label}")
            return

        # Expression statement: just evaluate for side effects
        self.munch_expr(node)

    def print_code(self):
        """Print all generated instructions."""
        for instr in self.instructions:
            print(instr)


# ---------- Examples ----------

def demo_expression():
    """Generate code for: result = a[i*4 + 8] + 1"""
    print("=== Expression: a[i*4 + 8] + 1 ===\n")

    # IR tree: BinOp(+, Mem(BinOp(+, BinOp(*, i, 4), 8)), 1)
    # Simplified: Mem(BinOp(+, BinOp(+, a, BinOp(*, i, 4)), 8))
    ir = BinOp("+",
        Mem(
            BinOp("+",
                BinOp("+",
                    Reg("a"),
                    BinOp("*", Reg("i"), Const(4))
                ),
                Const(8)
            )
        ),
        Const(1)
    )

    gen = MaximalMunchCodeGen()
    result_reg = gen.munch_expr(ir)
    print(f"Result in register: {result_reg}\n")
    gen.print_code()


def demo_array_store():
    """Generate code for: a[i] = b + c"""
    print("\n=== Statement: a[i] = b + c ===\n")

    # Store(BinOp(+, a, BinOp(*, i, 4)), BinOp(+, b, c))
    ir = Store(
        address=BinOp("+", Reg("a"), BinOp("*", Reg("i"), Const(4))),
        value=BinOp("+", Reg("b"), Reg("c"))
    )

    gen = MaximalMunchCodeGen()
    gen.munch_stmt(ir)
    gen.print_code()


def demo_conditional():
    """Generate code for: if (x < y) goto L1 else goto L2"""
    print("\n=== Conditional: if (x < y) goto L1 else goto L2 ===\n")

    ir = CJump("<", Reg("x"), Reg("y"), "L1", "L2")

    gen = MaximalMunchCodeGen()
    gen.munch_stmt(ir)
    gen.print_code()


if __name__ == "__main__":
    demo_expression()
    demo_array_store()
    demo_conditional()
```

---

## 3. 레지스터 할당(Register Allocation)

### 3.1 문제 정의

IR은 무한한 수의 가상 레지스터(virtual register) 또는 임시 변수(temporary)를 사용하지만, 실제 기계는 소수의 물리 레지스터($k$개)만 가집니다. 레지스터 할당은 가상 레지스터를 물리 레지스터에 배정하며, $k$개의 레지스터가 부족하면 일부를 메모리에 **스필(spill)**합니다.

### 3.2 레지스터 할당이 중요한 이유

레지스터 접근은 메모리 접근보다 수십~수백 배 빠릅니다:

| 접근 | 지연 시간(사이클) |
|------|-----------------|
| 레지스터(Register) | 0-1 |
| L1 캐시(L1 cache) | 3-5 |
| L2 캐시(L2 cache) | 10-15 |
| L3 캐시(L3 cache) | 30-50 |
| 주 메모리(Main memory) | 100-300 |

좋은 레지스터 할당은 프로그램 성능을 크게 향상시킬 수 있습니다.

### 3.3 활성 변수 분석(Liveness Analysis)

레지스터를 할당하기 전에 각 프로그램 지점에서 어떤 변수가 **살아있는지(live)** 파악해야 합니다. 변수가 어떤 지점에서 **살아있다**는 것은 해당 변수가 미래에 사용될 값을 보유하고 있다는 의미입니다.

**정의**:
- 지점 $p$에서 변수 $v$가 **정의된다(defined)**: $p$가 $v$에 값을 할당하는 경우
- 지점 $p$에서 변수 $v$가 **사용된다(used)**: $p$가 $v$를 읽는 경우
- 지점 $p$에서 변수 $v$가 **살아있다(live)**: $p$에서 $v$의 재정의 없이 $v$의 사용으로 가는 경로가 존재하는 경우

**데이터 흐름 방정식** (역방향으로 계산):

$$\text{LiveIn}(B) = \text{Use}(B) \cup (\text{LiveOut}(B) - \text{Def}(B))$$

$$\text{LiveOut}(B) = \bigcup_{S \in \text{succ}(B)} \text{LiveIn}(S)$$

### 3.4 간섭 그래프(Interference Graph)

두 변수가 어떤 프로그램 지점에서 동시에 살아있으면 **간섭(interfere)**합니다. **간섭 그래프** $G = (V, E)$는 다음을 가집니다:

- 꼭짓점 $V$: 변수(가상 레지스터)마다 하나
- 간선 $E$: $(u, v) \in E$ if $u$와 $v$가 간섭하는 경우

레지스터 할당은 **그래프 색칠(graph coloring)** 문제가 됩니다: 인접한 두 꼭짓점이 같은 색을 갖지 않도록 꼭짓점에 $k$개의 색(물리 레지스터)을 배정합니다.

### 3.5 그래프 색칠 레지스터 할당(Graph Coloring Register Allocation)

그래프 색칠 방식의 레지스터 할당은 Chaitin(1981)이 도입했습니다.

#### Chaitin 알고리즘

```
repeat:
    1. Build the interference graph
    2. Simplify: While there exists a node with degree < k:
       - Remove it from the graph and push it on a stack
       - (It can always be colored since it has fewer than k neighbors)
    3. Spill: If no such node exists:
       - Select a node to spill (heuristic: high degree, low use frequency)
       - Insert load/store instructions for the spilled variable
       - Go back to step 1
    4. Select: Pop nodes from the stack and assign colors
       - Each node has fewer than k colored neighbors, so a color is available
```

#### 예시

4개의 가상 레지스터 $\{a, b, c, d\}$와 간섭 그래프를 고려합니다:

```
a --- b
|  X  |     (a-b, a-c, b-c, b-d interfere)
c --- d
   |
   b
```

정확히 표현하면:
- $a$는 $b$, $c$와 간섭
- $b$는 $a$, $c$, $d$와 간섭
- $c$는 $a$, $b$와 간섭
- $d$는 $b$와 간섭

$k = 3$개 레지스터일 때:

```
Step 1 (Simplify):
  d has degree 1 < 3 → push d, remove from graph
  a has degree 2 < 3 → push a, remove
  c has degree 1 < 3 → push c, remove
  b has degree 0 < 3 → push b, remove

Stack (top to bottom): [b, c, a, d]

Step 2 (Select):
  Pop b → assign R0 (no neighbors colored yet)
  Pop c → neighbors: {b=R0} → assign R1
  Pop a → neighbors: {b=R0, c=R1} → assign R2
  Pop d → neighbors: {b=R0} → assign R1 (or R2)

Result: a=R2, b=R0, c=R1, d=R1
```

### 3.6 스필링(Spilling)

그래프가 $k$-색칠 불가능한 경우($k$보다 낮은 차수를 가진 노드가 없을 때), 변수를 메모리에 **스필(spill)**해야 합니다. 이는 다음을 의미합니다:

1. 스필된 변수의 각 사용 전에 메모리에서 `LOAD` 명령어 삽입
2. 각 정의 후에 메모리에 `STORE` 명령어 삽입
3. 스필된 변수는 더 이상 레지스터가 필요 없음 (또는 잠깐만 필요)
4. 간섭 그래프를 다시 빌드하고 재시도

**스필 휴리스틱(Spill heuristics)**:
- 차수가 가장 높은 변수 스필 (간섭이 가장 많음)
- 사용 빈도가 가장 낮은 변수 스필 (사용 횟수가 최소)
- 차수/사용 비율이 가장 높은 변수 스필
- 루프 내부의 변수는 스필 방지

### 3.7 선형 스캔 레지스터 할당(Linear Scan Register Allocation)

그래프 색칠은 효과적이지만 대형 프로그램에는 비용이 큽니다. **선형 스캔(linear scan)**은 JIT 컴파일러(예: HotSpot JVM, V8)에서 사용되는 더 빠른 대안입니다.

**아이디어**: 변수의 **활성 구간(live interval)** (첫 정의부터 마지막 사용까지의 범위) 순서대로 변수를 처리합니다. 레지스터를 탐욕적으로 할당합니다:

```
Algorithm: Linear Scan Register Allocation

Input:  Live intervals sorted by start point
        k available registers

1. active = []  (intervals currently occupying a register)
2. free_regs = {R0, R1, ..., R(k-1)}

3. for each interval i in order of increasing start:
    a. Expire old intervals:
       for each j in active (sorted by end point):
           if j.end < i.start:
               remove j from active
               return j.register to free_regs

    b. if free_regs is empty:
       Spill: find the interval in active with the latest end point
              if that end point > i.end:
                  spill that interval, give its register to i
              else:
                  spill i (assign memory location)

    c. else:
       Allocate: assign a register from free_regs to i
       add i to active
```

**시간 복잡도**: $O(n \log n)$ ($n$은 구간 수이며, 정렬이 지배적).

**비교**:

| 측면 | 그래프 색칠(Graph Coloring) | 선형 스캔(Linear Scan) |
|------|---------------------------|----------------------|
| 품질 | 더 좋음 (전역적 관점) | 좋음 (최적이 아님) |
| 컴파일 시간 | $O(n^2)$ 이상 | $O(n \log n)$ |
| 사용 사례 | 사전 컴파일러(AOT) | JIT 컴파일러 |
| 통합(coalescing) 처리 | 자연스럽게 | 추가 패스 필요 |

### 3.8 Python 구현: 선형 스캔(Linear Scan)

```python
"""Linear Scan Register Allocation."""

from dataclasses import dataclass, field


@dataclass
class LiveInterval:
    """A live interval for a virtual register."""
    vreg: str      # Virtual register name
    start: int     # First definition point
    end: int       # Last use point
    preg: str = "" # Assigned physical register (or "spill")

    def __str__(self):
        alloc = self.preg if self.preg else "unallocated"
        return f"{self.vreg}: [{self.start}, {self.end}] -> {alloc}"


class LinearScanAllocator:
    """
    Linear scan register allocator.
    """

    def __init__(self, num_physical_regs: int):
        self.k = num_physical_regs
        self.physical_regs = [f"R{i}" for i in range(num_physical_regs)]
        self.free_regs: list[str] = list(self.physical_regs)
        self.active: list[LiveInterval] = []  # Sorted by end point
        self.spilled: list[LiveInterval] = []

    def allocate(self, intervals: list[LiveInterval]):
        """
        Perform linear scan allocation.
        Modifies interval.preg in place.
        """
        # Sort by start point
        intervals.sort(key=lambda iv: iv.start)

        for iv in intervals:
            # Expire old intervals
            self._expire_old(iv)

            if not self.free_regs:
                # Must spill
                self._spill_at(iv)
            else:
                # Allocate a register
                reg = self.free_regs.pop(0)
                iv.preg = reg
                self.active.append(iv)
                self.active.sort(key=lambda x: x.end)

    def _expire_old(self, current: LiveInterval):
        """Remove intervals that have ended before the current one starts."""
        still_active = []
        for iv in self.active:
            if iv.end < current.start:
                # This interval has expired
                self.free_regs.append(iv.preg)
            else:
                still_active.append(iv)
        self.active = still_active
        self.active.sort(key=lambda x: x.end)

    def _spill_at(self, current: LiveInterval):
        """Spill either the current interval or the one ending latest."""
        if self.active and self.active[-1].end > current.end:
            # Spill the active interval ending latest
            spill = self.active.pop()
            current.preg = spill.preg
            spill.preg = "SPILL"
            self.spilled.append(spill)
            self.active.append(current)
            self.active.sort(key=lambda x: x.end)
        else:
            # Spill the current interval
            current.preg = "SPILL"
            self.spilled.append(current)


def demo_linear_scan():
    """Demonstrate linear scan register allocation."""
    print("=== Linear Scan Register Allocation ===\n")

    intervals = [
        LiveInterval("t1", start=0,  end=10),
        LiveInterval("t2", start=2,  end=8),
        LiveInterval("t3", start=4,  end=12),
        LiveInterval("t4", start=6,  end=14),
        LiveInterval("t5", start=9,  end=16),
        LiveInterval("t6", start=11, end=18),
        LiveInterval("t7", start=15, end=20),
    ]

    print("Live intervals:")
    for iv in intervals:
        print(f"  {iv.vreg}: [{iv.start}, {iv.end}]")

    # Allocate with 3 physical registers
    allocator = LinearScanAllocator(num_physical_regs=3)
    allocator.allocate(intervals)

    print(f"\nAllocation with {allocator.k} registers:")
    for iv in intervals:
        status = f"  {iv}"
        if iv.preg == "SPILL":
            status += " (SPILLED)"
        print(status)

    if allocator.spilled:
        print(f"\nSpilled variables: "
              f"{[s.vreg for s in allocator.spilled]}")

    # Visualize
    print("\nTimeline visualization:")
    max_time = max(iv.end for iv in intervals)
    print("  Time: " + "".join(f"{t:2d}" for t in range(max_time + 1)))
    for iv in intervals:
        line = "  " + f"{iv.vreg:4s}: "
        for t in range(max_time + 1):
            if iv.start <= t <= iv.end:
                if iv.preg == "SPILL":
                    line += " S"
                else:
                    line += f" {iv.preg[1]}"  # Just the number
            else:
                line += " ."
        line += f"  ({iv.preg})"
        print(line)


if __name__ == "__main__":
    demo_linear_scan()
```

---

## 4. 명령어 스케줄링(Instruction Scheduling)

### 4.1 문제 정의

현대 프로세서는 **파이프라인(pipelined)** 구조입니다: 명령어 실행의 서로 다른 단계가 겹쳐서 실행됩니다. 그러나 명령어가 아직 완료되지 않은 이전 명령어의 결과에 의존할 때 **데이터 해저드(data hazard)**가 발생하여 **파이프라인 스톨(pipeline stall)**을 일으킬 수 있습니다.

**스톨 예시**:
```
LOAD R1, [R2]     ; takes 4 cycles to complete
ADD  R3, R1, R4   ; must wait for R1 → pipeline stall (3 wasted cycles)
```

**스케줄링 후**:
```
LOAD R1, [R2]     ; issue load
ADD  R5, R6, R7   ; independent instruction fills the gap
SUB  R8, R9, R10  ; another independent instruction
MOV  R11, R12     ; yet another
ADD  R3, R1, R4   ; R1 is now ready, no stall
```

### 4.2 리스트 스케줄링(List Scheduling)

**리스트 스케줄링(list scheduling)**은 가장 일반적인 명령어 스케줄링 알고리즘입니다. 단일 기본 블록(basic block)에서 동작합니다.

**입력**: **의존성 DAG(dependency DAG)**:
- 노드는 명령어
- 간선은 데이터 의존성 표현 (읽기 후 쓰기, 쓰기 후 읽기, 쓰기 후 쓰기)
- 간선 가중치는 소스 명령어의 지연 시간(latency)

**알고리즘**:

```
function list_schedule(DAG, num_functional_units):
    ready = set of instructions with no predecessors
    schedule = []
    cycle = 0

    while ready is not empty or pending instructions exist:
        # Select from ready instructions (priority-based)
        available = instructions in ready whose operands are available
        for each functional unit:
            if available is not empty:
                instr = select highest-priority instruction from available
                issue instr at current cycle
                schedule[cycle] = instr
                remove instr from ready

        cycle += 1

        # Check if any pending instructions have completed
        # Add their successors to ready if all predecessors are done

    return schedule
```

**우선순위 휴리스틱** (어떤 명령어를 먼저 스케줄할지):
- **임계 경로 길이(Critical path length)**: 임의의 싱크까지의 가장 긴 경로에 있는 명령어 선호
- **후계자 수(Number of successors)**: 의존하는 후계자가 많은 명령어 선호
- **지연 시간(Latency)**: 지연 시간이 높은 명령어 선호 (더 일찍 시작)

### 4.3 예시: 리스트 스케줄링

```
Instructions:
  I1: LOAD R1, [addr1]   ; latency 4
  I2: LOAD R2, [addr2]   ; latency 4
  I3: ADD  R3, R1, R2    ; latency 1, depends on I1, I2
  I4: MUL  R4, R3, R1    ; latency 3, depends on I3, I1
  I5: STORE R4, [addr3]  ; latency 1, depends on I4

Dependency DAG:
  I1 ──(4)──▶ I3
  I2 ──(4)──▶ I3
  I3 ──(1)──▶ I4
  I1 ──(4)──▶ I4
  I4 ──(3)──▶ I5

Unscheduled (all instructions sequentially):
  Cycle 0: I1 (LOAD)
  Cycle 1-3: stall (waiting for I1)
  Cycle 4: I2 (LOAD)
  Cycle 5-7: stall (waiting for I2)
  Cycle 8: I3 (ADD)
  Cycle 9: I4 (MUL)
  Cycle 10-11: stall
  Cycle 12: I5 (STORE)
  Total: 13 cycles

Scheduled (list scheduling):
  Cycle 0: I1 (LOAD R1)
  Cycle 1: I2 (LOAD R2)    ← issued during I1's latency
  Cycle 4: I3 (ADD R3)     ← both I1, I2 ready
  Cycle 5: I4 (MUL R4)     ← I3 ready
  Cycle 8: I5 (STORE R4)   ← I4 ready
  Total: 9 cycles (saved 4 cycles)
```

### 4.4 소프트웨어 파이프라이닝(Software Pipelining) 개요

**소프트웨어 파이프라이닝(software pipelining)**은 하드웨어 파이프라이닝이 명령어 단계를 겹치는 것처럼 루프의 반복을 겹치는 루프 최적화 기법입니다.

**아이디어**: 반복 $i$가 완료되기 전에 반복 $i+1$을 시작합니다:

```
Without software pipelining:     With software pipelining:
  Iter 1: LOAD-ADD-STORE         Cycle 0: LOAD[1]
  Iter 2: LOAD-ADD-STORE         Cycle 1: LOAD[2], ADD[1]
  Iter 3: LOAD-ADD-STORE         Cycle 2: LOAD[3], ADD[2], STORE[1]
                                  Cycle 3: LOAD[4], ADD[3], STORE[2]
                                  ...
```

소프트웨어 파이프라인의 정상 상태(steady-state)는 여러 반복의 일부를 동시에 실행하여 모든 기능 유닛을 바쁘게 유지합니다.

소프트웨어 파이프라이닝은 주로 **모듈로 스케줄링(modulo scheduling)**으로 구현되며, 고정된 **시작 간격(initiation interval, II)**으로 반복할 때 유효한 겹침 스케줄이 생성되는 하나의 반복에 대한 스케줄을 찾습니다.

---

## 5. 핍홀 최적화(Peephole Optimization)

### 5.1 핍홀 최적화란?

**핍홀 최적화(peephole optimization)**는 연속된 명령어들의 작은 윈도우("핍홀")를 검사하고 더 빠르거나 더 짧은 동등한 명령어로 교체합니다. 코드 생성 후 최종 정리 패스로 적용됩니다.

### 5.2 일반적인 핍홀 최적화

#### 중복 로드/스토어 제거(Redundant Load/Store Elimination)

```
Before:                  After:
  STORE R1, [addr]       STORE R1, [addr]
  LOAD  R1, [addr]       (load eliminated -- R1 already has the value)
```

#### 중복 이동 제거(Redundant Moves)

```
Before:                  After:
  MOV R1, R2             (eliminated if R1 == R2, or
  MOV R2, R1              second move eliminated if first is sufficient)
```

#### 강도 감소(Strength Reduction)

```
Before:                  After:
  MUL R1, R2, #2         SHL R1, R2, #1    (shift is cheaper)
  MUL R1, R2, #8         SHL R1, R2, #3
  DIV R1, R2, #4         SHR R1, R2, #2    (for unsigned)
```

#### 대수적 단순화(Algebraic Simplification)

```
Before:                  After:
  ADD R1, R2, #0         MOV R1, R2         (or eliminated if R1==R2)
  MUL R1, R2, #1         MOV R1, R2
  MUL R1, R2, #0         MOVI R1, #0
  SUB R1, R2, #0         MOV R1, R2
```

#### 분기 최적화(Branch Optimization)

```
Before:                  After:
  JMP L1                 JMP L2             (jump-to-jump elimination)
  ...
L1: JMP L2

Before:                  After:
  BEQ L1                 BNE L2             (branch-over-jump elimination)
  JMP L2                 ...
L1: ...                  L2: ...
```

#### 도달 불가 코드 제거(Unreachable Code Elimination)

```
Before:                  After:
  JMP L1                 JMP L1
  ADD R1, R2, R3         (unreachable, eliminated)
  MOV R4, R5             (unreachable, eliminated)
L1: ...                  L1: ...
```

### 5.3 Python 구현: 핍홀 최적화기(Peephole Optimizer)

```python
"""Peephole optimizer for a simple instruction set."""

from dataclasses import dataclass
import re


@dataclass
class Instruction:
    """A machine instruction for peephole optimization."""
    text: str       # Full instruction text
    opcode: str = ""
    operands: list = None
    label: str = ""  # If this is a label

    def __post_init__(self):
        if self.operands is None:
            self.operands = []
        self._parse()

    def _parse(self):
        text = self.text.strip()
        if text.endswith(":"):
            self.label = text[:-1]
            self.opcode = "LABEL"
            return
        parts = text.split(None, 1)
        if parts:
            self.opcode = parts[0].upper()
            if len(parts) > 1:
                self.operands = [o.strip() for o in parts[1].split(",")]

    def __str__(self):
        if self.label:
            return f"{self.label}:"
        return self.text


class PeepholeOptimizer:
    """
    Apply peephole optimizations to a list of instructions.
    """

    def __init__(self, instructions: list[str]):
        self.instructions = [Instruction(text=instr) for instr in instructions]
        self.changed = True  # Track if any optimization was applied

    def optimize(self, max_passes: int = 10) -> list[str]:
        """Run peephole optimizations until no more changes occur."""
        pass_num = 0
        while self.changed and pass_num < max_passes:
            self.changed = False
            pass_num += 1
            self._redundant_moves()
            self._redundant_load_after_store()
            self._strength_reduction()
            self._algebraic_simplification()
            self._jump_to_jump()
            self._unreachable_code()
            # Remove None entries
            self.instructions = [i for i in self.instructions if i is not None]

        return [str(i) for i in self.instructions]

    def _redundant_moves(self):
        """Remove MOV Rx, Rx (move to self)."""
        for i, instr in enumerate(self.instructions):
            if instr and instr.opcode == "MOV" and len(instr.operands) == 2:
                if instr.operands[0] == instr.operands[1]:
                    self.instructions[i] = None
                    self.changed = True

    def _redundant_load_after_store(self):
        """Remove LOAD Rx, [addr] immediately after STORE Rx, [addr]."""
        for i in range(len(self.instructions) - 1):
            curr = self.instructions[i]
            nxt = self.instructions[i + 1]
            if curr is None or nxt is None:
                continue
            if curr.opcode == "STORE" and nxt.opcode == "LOAD":
                # Check if same register and same address
                if (len(curr.operands) >= 2 and len(nxt.operands) >= 2
                    and curr.operands[0] == nxt.operands[0]
                    and curr.operands[1] == nxt.operands[1]):
                    self.instructions[i + 1] = None
                    self.changed = True

    def _strength_reduction(self):
        """Replace MUL/DIV by power of 2 with shifts."""
        for i, instr in enumerate(self.instructions):
            if instr is None:
                continue
            if instr.opcode == "MUL" and len(instr.operands) == 3:
                imm = instr.operands[2]
                if imm.startswith("#"):
                    val = int(imm[1:])
                    if val > 0 and (val & (val - 1)) == 0:
                        shift = val.bit_length() - 1
                        new_text = (f"  SHL {instr.operands[0]}, "
                                    f"{instr.operands[1]}, #{shift}")
                        self.instructions[i] = Instruction(text=new_text)
                        self.changed = True

            if instr.opcode == "DIV" and len(instr.operands) == 3:
                imm = instr.operands[2]
                if imm.startswith("#"):
                    val = int(imm[1:])
                    if val > 0 and (val & (val - 1)) == 0:
                        shift = val.bit_length() - 1
                        new_text = (f"  SHR {instr.operands[0]}, "
                                    f"{instr.operands[1]}, #{shift}")
                        self.instructions[i] = Instruction(text=new_text)
                        self.changed = True

    def _algebraic_simplification(self):
        """Simplify ADD x, y, #0 → MOV x, y, etc."""
        for i, instr in enumerate(self.instructions):
            if instr is None:
                continue

            if instr.opcode == "ADD" and len(instr.operands) == 3:
                if instr.operands[2] == "#0":
                    new_text = f"  MOV {instr.operands[0]}, {instr.operands[1]}"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True

            if instr.opcode == "MUL" and len(instr.operands) == 3:
                if instr.operands[2] == "#1":
                    new_text = f"  MOV {instr.operands[0]}, {instr.operands[1]}"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True
                elif instr.operands[2] == "#0":
                    new_text = f"  MOVI {instr.operands[0]}, #0"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True

            if instr.opcode == "SUB" and len(instr.operands) == 3:
                if instr.operands[2] == "#0":
                    new_text = f"  MOV {instr.operands[0]}, {instr.operands[1]}"
                    self.instructions[i] = Instruction(text=new_text)
                    self.changed = True

    def _jump_to_jump(self):
        """Replace JMP L1 where L1: JMP L2 with JMP L2."""
        # Build label -> index map
        label_map = {}
        for i, instr in enumerate(self.instructions):
            if instr and instr.opcode == "LABEL":
                label_map[instr.label] = i

        for i, instr in enumerate(self.instructions):
            if instr is None:
                continue
            if instr.opcode == "JMP" and len(instr.operands) == 1:
                target = instr.operands[0]
                if target in label_map:
                    target_idx = label_map[target]
                    # Find the next non-label instruction after the target
                    j = target_idx + 1
                    while (j < len(self.instructions)
                           and self.instructions[j] is not None
                           and self.instructions[j].opcode == "LABEL"):
                        j += 1
                    if (j < len(self.instructions)
                        and self.instructions[j] is not None
                        and self.instructions[j].opcode == "JMP"):
                        new_target = self.instructions[j].operands[0]
                        if new_target != target:
                            new_text = f"  JMP {new_target}"
                            self.instructions[i] = Instruction(text=new_text)
                            self.changed = True

    def _unreachable_code(self):
        """Remove instructions between an unconditional jump and the next label."""
        i = 0
        while i < len(self.instructions):
            instr = self.instructions[i]
            if instr and instr.opcode in ("JMP", "RET"):
                j = i + 1
                while (j < len(self.instructions)
                       and self.instructions[j] is not None
                       and self.instructions[j].opcode != "LABEL"):
                    self.instructions[j] = None
                    self.changed = True
                    j += 1
            i += 1


def demo_peephole():
    """Demonstrate peephole optimization."""
    instructions = [
        "  MOVI R1, #5",
        "  MOVI R2, #10",
        "  ADD  R3, R1, R2",
        "  MOV  R3, R3",           # Redundant self-move
        "  MUL  R4, R3, #8",       # Strength reduction: * 8 → << 3
        "  ADD  R5, R4, #0",       # Algebraic: + 0 → move
        "  STORE R5, [R6]",
        "  LOAD  R5, [R6]",        # Redundant load after store
        "  MUL  R7, R5, #1",       # Algebraic: * 1 → move
        "  MUL  R8, R5, #0",       # Algebraic: * 0 → 0
        "  DIV  R9, R3, #4",       # Strength reduction: / 4 → >> 2
        "  JMP  L1",
        "  ADD  R10, R1, R2",      # Unreachable code
        "  SUB  R11, R3, R4",      # Unreachable code
        "L1:",
        "  JMP  L2",               # Jump-to-jump target
        "L2:",
        "  SUB  R12, R1, #0",      # Algebraic: - 0 → move
        "  RET",
    ]

    print("=== Before Peephole Optimization ===")
    for instr in instructions:
        print(instr)

    optimizer = PeepholeOptimizer(instructions)
    optimized = optimizer.optimize()

    print("\n=== After Peephole Optimization ===")
    for instr in optimized:
        print(instr)


if __name__ == "__main__":
    demo_peephole()
```

---

## 6. 언어 구성요소를 위한 코드 생성(Code Generation for Language Constructs)

### 6.1 식(Expressions)

산술 식에 대한 코드 생성은 식 트리의 구조를 따릅니다. 이항 연산 $a \;\text{op}\; b$의 경우:

```
generate(a)         → result in R1
generate(b)         → result in R2
 OP R3, R1, R2       → R3 = R1 op R2
```

깊게 중첩된 식의 경우 레지스터를 신중하게 관리해야 합니다. **Sethi-Ullman 번호 매기기(Sethi-Ullman numbering)** 알고리즘은 식 트리를 평가하는 데 필요한 최소 레지스터 수를 계산합니다.

**Sethi-Ullman 번호 매기기**:
- 왼쪽 자식인 리프: 레이블 = 1
- 오른쪽 자식인 리프: 레이블 = 0
- 레이블 $l_1$과 $l_2$를 가진 자식들을 가진 내부 노드:

$$\text{label}(n) = \begin{cases} \max(l_1, l_2) & \text{if } l_1 \neq l_2 \\ l_1 + 1 & \text{if } l_1 = l_2 \end{cases}$$

루트의 레이블이 필요한 최소 레지스터 수를 나타냅니다.

### 6.2 제어 흐름(Control Flow)

#### If-Else

```c
if (condition) {
    then_body;
} else {
    else_body;
}
```

생성된 코드:
```
    <evaluate condition into R1>
    CMP R1, #0
    BEQ else_label
    <then_body code>
    JMP end_label
else_label:
    <else_body code>
end_label:
```

#### While 루프

```c
while (condition) {
    body;
}
```

생성된 코드:
```
loop_start:
    <evaluate condition into R1>
    CMP R1, #0
    BEQ loop_end
    <body code>
    JMP loop_start
loop_end:
```

#### For 루프

for 루프는 일반적으로 while 루프로 변환됩니다:

```c
for (init; condition; step) { body; }
```

다음과 같이 됩니다:
```
    <init code>
loop_start:
    <evaluate condition>
    BEQ loop_end
    <body code>
    <step code>
    JMP loop_start
loop_end:
```

#### 단락(Short-Circuit) 불리언 평가

`a && b`의 경우:
```
    <evaluate a>
    CMP R1, #0
    BEQ false_label     ; if a is false, skip b
    <evaluate b>
    CMP R2, #0
    BEQ false_label     ; if b is false
    MOVI R3, #1          ; result is true
    JMP end_label
false_label:
    MOVI R3, #0          ; result is false
end_label:
```

`a || b`의 경우:
```
    <evaluate a>
    CMP R1, #0
    BNE true_label       ; if a is true, skip b
    <evaluate b>
    CMP R2, #0
    BNE true_label       ; if b is true
    MOVI R3, #0           ; result is false
    JMP end_label
true_label:
    MOVI R3, #1           ; result is true
end_label:
```

### 6.3 함수 호출(Function Calls)

함수 호출을 위한 코드 생성은 다음을 포함합니다:

```
; Caller-saved registers
PUSH R_caller_saved1
PUSH R_caller_saved2

; Arguments (System V AMD64: first 6 in registers)
MOV RDI, arg1
MOV RSI, arg2
MOV RDX, arg3
; ... or push on stack if > 6 args

; Call
CALL function_label

; Return value is in RAX
MOV result, RAX

; Restore caller-saved registers
POP R_caller_saved2
POP R_caller_saved1
```

---

## 7. 스택 기계 코드 생성기(Stack Machine Code Generator)

### 7.1 스택 기계란?

**스택 기계(stack machine)**는 레지스터 대신 피연산자 스택(operand stack)을 사용합니다. 명령어는 암묵적으로 스택 상단에서 동작합니다:

```
PUSH 5        ; stack: [5]
PUSH 3        ; stack: [5, 3]
ADD           ; stack: [8]       (pop two, push sum)
PUSH 2        ; stack: [8, 2]
MUL           ; stack: [16]      (pop two, push product)
```

스택 기계는 레지스터 할당 문제가 없기 때문에 코드 생성이 더 단순합니다. 예시: JVM 바이트코드, .NET CIL, Python 바이트코드, WebAssembly (일부).

### 7.2 스택 기계 명령어 집합

```
Stack operations:
    PUSH <value>     ; Push constant onto stack
    LOAD <var>       ; Push variable's value onto stack
    STORE <var>      ; Pop value and store in variable
    POP              ; Discard top of stack
    DUP              ; Duplicate top of stack

Arithmetic (pop operands, push result):
    ADD, SUB, MUL, DIV, MOD, NEG

Comparison (pop two, push boolean):
    EQ, NE, LT, GT, LE, GE

Logic:
    AND, OR, NOT

Control flow:
    LABEL <name>     ; Define a label
    JMP <label>      ; Unconditional jump
    JMPF <label>     ; Jump if top of stack is false (0)
    JMPT <label>     ; Jump if top of stack is true (non-0)

Functions:
    CALL <func> <nargs>  ; Call function with n arguments
    RET                  ; Return (top of stack is return value)

I/O:
    PRINT            ; Pop and print top of stack
    READ             ; Read input and push onto stack
```

### 7.3 완전한 스택 기계 구현

```python
"""
Complete code generator and virtual machine for a stack-based target.

Compiles a simple language to stack machine bytecode and executes it.
"""

from dataclasses import dataclass
from typing import Union
from enum import Enum, auto


# ============================================================
# Part 1: Source Language AST
# ============================================================

@dataclass
class NumLit:
    value: int

@dataclass
class BoolLit:
    value: bool

@dataclass
class VarRef:
    name: str

@dataclass
class BinaryExpr:
    op: str
    left: 'Expression'
    right: 'Expression'

@dataclass
class UnaryExpr:
    op: str
    operand: 'Expression'

@dataclass
class CallExpr:
    func: str
    args: list

Expression = Union[NumLit, BoolLit, VarRef, BinaryExpr, UnaryExpr, CallExpr]


@dataclass
class AssignStmt:
    target: str
    value: Expression

@dataclass
class PrintStmt:
    value: Expression

@dataclass
class IfStmt:
    condition: Expression
    then_body: list
    else_body: list

@dataclass
class WhileStmt:
    condition: Expression
    body: list

@dataclass
class ReturnStmt:
    value: Expression

@dataclass
class FuncDef:
    name: str
    params: list  # list of parameter names
    body: list    # list of statements

Statement = Union[AssignStmt, PrintStmt, IfStmt, WhileStmt, ReturnStmt]


# ============================================================
# Part 2: Stack Machine Bytecode
# ============================================================

class Opcode(Enum):
    PUSH = auto()
    LOAD = auto()
    STORE = auto()
    POP = auto()
    DUP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    JMP = auto()
    JMPF = auto()
    JMPT = auto()
    CALL = auto()
    RET = auto()
    PRINT = auto()
    LABEL = auto()
    HALT = auto()


@dataclass
class BytecodeInstr:
    opcode: Opcode
    operand: Union[int, str, None] = None
    operand2: Union[int, str, None] = None

    def __str__(self):
        parts = [f"{self.opcode.name:8s}"]
        if self.operand is not None:
            parts.append(str(self.operand))
        if self.operand2 is not None:
            parts.append(str(self.operand2))
        return " ".join(parts)


# ============================================================
# Part 3: Code Generator
# ============================================================

class CodeGenerator:
    """
    Generate stack machine bytecode from AST.
    """

    def __init__(self):
        self.code: list[BytecodeInstr] = []
        self._label_counter = 0
        self.functions: dict[str, int] = {}  # func name -> code address

    def new_label(self) -> str:
        self._label_counter += 1
        return f"L{self._label_counter}"

    def emit(self, opcode: Opcode, operand=None, operand2=None):
        self.code.append(BytecodeInstr(opcode, operand, operand2))

    def generate_program(self, program: list):
        """
        Generate code for a complete program.
        Program is a list of function definitions and statements.
        """
        # Separate functions from top-level statements
        functions = [node for node in program if isinstance(node, FuncDef)]
        statements = [node for node in program
                      if not isinstance(node, FuncDef)]

        # Jump over function definitions to main code
        main_label = self.new_label()
        self.emit(Opcode.JMP, main_label)

        # Generate code for functions
        for func in functions:
            self._generate_function(func)

        # Generate main code
        self.emit(Opcode.LABEL, main_label)
        for stmt in statements:
            self._generate_stmt(stmt)
        self.emit(Opcode.HALT)

    def _generate_function(self, func: FuncDef):
        """Generate code for a function definition."""
        self.functions[func.name] = len(self.code)
        self.emit(Opcode.LABEL, f"func_{func.name}")

        # Function body
        for stmt in func.body:
            self._generate_stmt(stmt)

        # Implicit return 0 if no explicit return
        self.emit(Opcode.PUSH, 0)
        self.emit(Opcode.RET)

    def _generate_stmt(self, stmt):
        """Generate code for a statement."""
        if isinstance(stmt, AssignStmt):
            self._generate_expr(stmt.value)
            self.emit(Opcode.STORE, stmt.target)

        elif isinstance(stmt, PrintStmt):
            self._generate_expr(stmt.value)
            self.emit(Opcode.PRINT)

        elif isinstance(stmt, IfStmt):
            self._generate_if(stmt)

        elif isinstance(stmt, WhileStmt):
            self._generate_while(stmt)

        elif isinstance(stmt, ReturnStmt):
            self._generate_expr(stmt.value)
            self.emit(Opcode.RET)

        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    def _generate_expr(self, expr):
        """Generate code for an expression (result pushed on stack)."""
        if isinstance(expr, NumLit):
            self.emit(Opcode.PUSH, expr.value)

        elif isinstance(expr, BoolLit):
            self.emit(Opcode.PUSH, 1 if expr.value else 0)

        elif isinstance(expr, VarRef):
            self.emit(Opcode.LOAD, expr.name)

        elif isinstance(expr, BinaryExpr):
            self._generate_expr(expr.left)
            self._generate_expr(expr.right)
            op_map = {
                '+': Opcode.ADD, '-': Opcode.SUB,
                '*': Opcode.MUL, '/': Opcode.DIV,
                '%': Opcode.MOD,
                '==': Opcode.EQ, '!=': Opcode.NE,
                '<': Opcode.LT, '>': Opcode.GT,
                '<=': Opcode.LE, '>=': Opcode.GE,
                'and': Opcode.AND, 'or': Opcode.OR,
            }
            if expr.op in op_map:
                self.emit(op_map[expr.op])
            else:
                raise ValueError(f"Unknown operator: {expr.op}")

        elif isinstance(expr, UnaryExpr):
            self._generate_expr(expr.operand)
            if expr.op == '-':
                self.emit(Opcode.NEG)
            elif expr.op == 'not':
                self.emit(Opcode.NOT)

        elif isinstance(expr, CallExpr):
            # Push arguments in order
            for arg in expr.args:
                self._generate_expr(arg)
            self.emit(Opcode.CALL, f"func_{expr.func}", len(expr.args))

    def _generate_if(self, stmt: IfStmt):
        else_label = self.new_label()
        end_label = self.new_label()

        self._generate_expr(stmt.condition)
        self.emit(Opcode.JMPF, else_label)

        # Then body
        for s in stmt.then_body:
            self._generate_stmt(s)
        self.emit(Opcode.JMP, end_label)

        # Else body
        self.emit(Opcode.LABEL, else_label)
        for s in stmt.else_body:
            self._generate_stmt(s)

        self.emit(Opcode.LABEL, end_label)

    def _generate_while(self, stmt: WhileStmt):
        loop_label = self.new_label()
        end_label = self.new_label()

        self.emit(Opcode.LABEL, loop_label)
        self._generate_expr(stmt.condition)
        self.emit(Opcode.JMPF, end_label)

        # Body
        for s in stmt.body:
            self._generate_stmt(s)
        self.emit(Opcode.JMP, loop_label)

        self.emit(Opcode.LABEL, end_label)

    def print_code(self):
        """Pretty-print the generated bytecode."""
        for i, instr in enumerate(self.code):
            label_marker = ""
            if instr.opcode == Opcode.LABEL:
                label_marker = f"\n{instr.operand}:"
                print(label_marker)
                continue
            print(f"  [{i:3d}] {instr}")


# ============================================================
# Part 4: Stack Machine Virtual Machine
# ============================================================

class StackMachineVM:
    """
    Execute stack machine bytecode.
    """

    def __init__(self, code: list[BytecodeInstr],
                 functions: dict[str, int]):
        self.code = code
        self.functions = functions
        self.stack: list = []          # Operand stack
        self.call_stack: list = []     # Return addresses + saved state
        self.variables: dict[str, int] = {}
        self.pc = 0                    # Program counter
        self.output: list[str] = []    # Captured output

        # Resolve labels to addresses
        self.label_map: dict[str, int] = {}
        for i, instr in enumerate(self.code):
            if instr.opcode == Opcode.LABEL:
                self.label_map[instr.operand] = i

    def run(self, trace: bool = False):
        """Execute the bytecode program."""
        while self.pc < len(self.code):
            instr = self.code[self.pc]

            if trace and instr.opcode != Opcode.LABEL:
                print(f"  PC={self.pc:3d} {instr}  "
                      f"stack={self.stack[-5:]}")

            if instr.opcode == Opcode.HALT:
                break

            elif instr.opcode == Opcode.LABEL:
                self.pc += 1
                continue

            elif instr.opcode == Opcode.PUSH:
                self.stack.append(instr.operand)

            elif instr.opcode == Opcode.LOAD:
                name = instr.operand
                value = self.variables.get(name, 0)
                self.stack.append(value)

            elif instr.opcode == Opcode.STORE:
                name = instr.operand
                value = self.stack.pop()
                self.variables[name] = value

            elif instr.opcode == Opcode.POP:
                self.stack.pop()

            elif instr.opcode == Opcode.DUP:
                self.stack.append(self.stack[-1])

            # Arithmetic
            elif instr.opcode == Opcode.ADD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)

            elif instr.opcode == Opcode.SUB:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)

            elif instr.opcode == Opcode.MUL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)

            elif instr.opcode == Opcode.DIV:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a // b if b != 0 else 0)

            elif instr.opcode == Opcode.MOD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a % b if b != 0 else 0)

            elif instr.opcode == Opcode.NEG:
                self.stack.append(-self.stack.pop())

            # Comparison
            elif instr.opcode == Opcode.EQ:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a == b else 0)

            elif instr.opcode == Opcode.NE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a != b else 0)

            elif instr.opcode == Opcode.LT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a < b else 0)

            elif instr.opcode == Opcode.GT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a > b else 0)

            elif instr.opcode == Opcode.LE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a <= b else 0)

            elif instr.opcode == Opcode.GE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a >= b else 0)

            # Logic
            elif instr.opcode == Opcode.AND:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if (a and b) else 0)

            elif instr.opcode == Opcode.OR:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if (a or b) else 0)

            elif instr.opcode == Opcode.NOT:
                self.stack.append(1 if not self.stack.pop() else 0)

            # Control flow
            elif instr.opcode == Opcode.JMP:
                self.pc = self.label_map[instr.operand]
                continue

            elif instr.opcode == Opcode.JMPF:
                cond = self.stack.pop()
                if not cond:
                    self.pc = self.label_map[instr.operand]
                    continue

            elif instr.opcode == Opcode.JMPT:
                cond = self.stack.pop()
                if cond:
                    self.pc = self.label_map[instr.operand]
                    continue

            # Functions
            elif instr.opcode == Opcode.CALL:
                func_label = instr.operand
                nargs = instr.operand2

                # Save state
                self.call_stack.append({
                    'return_pc': self.pc + 1,
                    'saved_vars': dict(self.variables),
                })

                # Pop arguments and bind to params
                args = []
                for _ in range(nargs):
                    args.insert(0, self.stack.pop())

                # Jump to function
                self.pc = self.label_map[func_label]

                # Bind arguments (simplified: use positional names)
                for idx, arg_val in enumerate(args):
                    self.variables[f"__arg{idx}"] = arg_val

                continue

            elif instr.opcode == Opcode.RET:
                ret_val = self.stack.pop() if self.stack else 0
                if self.call_stack:
                    frame = self.call_stack.pop()
                    self.variables = frame['saved_vars']
                    self.pc = frame['return_pc']
                    self.stack.append(ret_val)
                    continue
                else:
                    self.stack.append(ret_val)
                    break

            elif instr.opcode == Opcode.PRINT:
                val = self.stack.pop()
                self.output.append(str(val))
                print(f"  OUTPUT: {val}")

            self.pc += 1


# ============================================================
# Part 5: Examples
# ============================================================

def demo_fibonacci():
    """
    Compile and run:
        n = 10
        a = 0
        b = 1
        i = 0
        while (i < n) {
            print a
            temp = a + b
            a = b
            b = temp
            i = i + 1
        }
    """
    print("=" * 60)
    print("Stack Machine: Fibonacci Sequence (first 10)")
    print("=" * 60)

    program = [
        AssignStmt("n", NumLit(10)),
        AssignStmt("a", NumLit(0)),
        AssignStmt("b", NumLit(1)),
        AssignStmt("i", NumLit(0)),
        WhileStmt(
            condition=BinaryExpr("<", VarRef("i"), VarRef("n")),
            body=[
                PrintStmt(VarRef("a")),
                AssignStmt("temp", BinaryExpr("+", VarRef("a"), VarRef("b"))),
                AssignStmt("a", VarRef("b")),
                AssignStmt("b", VarRef("temp")),
                AssignStmt("i", BinaryExpr("+", VarRef("i"), NumLit(1))),
            ]
        ),
    ]

    # Generate code
    gen = CodeGenerator()
    gen.generate_program(program)
    print("\n--- Generated Bytecode ---")
    gen.print_code()

    # Execute
    print("\n--- Execution ---")
    vm = StackMachineVM(gen.code, gen.functions)
    vm.run()
    print(f"\nFibonacci output: {vm.output}")


def demo_factorial():
    """
    Compile and run:
        result = 1
        n = 7
        while (n > 0) {
            result = result * n
            n = n - 1
        }
        print result
    """
    print("\n" + "=" * 60)
    print("Stack Machine: Factorial of 7")
    print("=" * 60)

    program = [
        AssignStmt("result", NumLit(1)),
        AssignStmt("n", NumLit(7)),
        WhileStmt(
            condition=BinaryExpr(">", VarRef("n"), NumLit(0)),
            body=[
                AssignStmt("result",
                    BinaryExpr("*", VarRef("result"), VarRef("n"))),
                AssignStmt("n",
                    BinaryExpr("-", VarRef("n"), NumLit(1))),
            ]
        ),
        PrintStmt(VarRef("result")),
    ]

    gen = CodeGenerator()
    gen.generate_program(program)
    print("\n--- Generated Bytecode ---")
    gen.print_code()

    print("\n--- Execution ---")
    vm = StackMachineVM(gen.code, gen.functions)
    vm.run()
    print(f"\n7! = {vm.output[0]}")


def demo_conditionals():
    """
    Compile and run:
        x = 15
        if (x > 10) {
            if (x > 20) {
                print 3
            } else {
                print 2
            }
        } else {
            print 1
        }
    """
    print("\n" + "=" * 60)
    print("Stack Machine: Nested If-Else")
    print("=" * 60)

    program = [
        AssignStmt("x", NumLit(15)),
        IfStmt(
            condition=BinaryExpr(">", VarRef("x"), NumLit(10)),
            then_body=[
                IfStmt(
                    condition=BinaryExpr(">", VarRef("x"), NumLit(20)),
                    then_body=[PrintStmt(NumLit(3))],
                    else_body=[PrintStmt(NumLit(2))],
                )
            ],
            else_body=[PrintStmt(NumLit(1))],
        ),
    ]

    gen = CodeGenerator()
    gen.generate_program(program)
    print("\n--- Generated Bytecode ---")
    gen.print_code()

    print("\n--- Execution ---")
    vm = StackMachineVM(gen.code, gen.functions)
    vm.run()
    print(f"\nExpected: 2, Got: {vm.output[0]}")


if __name__ == "__main__":
    demo_fibonacci()
    demo_factorial()
    demo_conditionals()
```

---

## 8. 기계 의존 최적화(Machine-Dependent Optimization)

### 8.1 특수 명령어 활용

많은 프로세서에는 코드 생성기가 활용할 수 있는 특수 명령어가 있습니다:

| 최적화 | 예시 |
|--------|------|
| 곱셈-누산(Multiply-accumulate) | `MADD Rd, Rs1, Rs2, Rs3` ($Rd = Rs1 + Rs2 \times Rs3$) |
| 선행 0 개수(Count leading zeros) | `CLZ Rd, Rs` (log2에 유용) |
| 바이트 스왑(Byte swap) | `REV Rd, Rs` (엔디언 변환) |
| 조건부 이동(Conditional move) | `CMOV Rd, Rs, cond` (분기 방지) |
| SIMD | `VADD.4S V0, V1, V2` (4개 덧셈 병렬 처리) |

### 8.2 주소 지정 모드 선택(Addressing Mode Selection)

복잡한 주소 지정 모드는 명령어 수를 줄일 수 있습니다:

```
; Without: 3 instructions
MOV  R1, R_base
ADD  R1, R1, R_index, LSL #2    ; index * 4
LOAD R2, [R1]

; With indexed addressing: 1 instruction
LOAD R2, [R_base, R_index, LSL #2]
```

### 8.3 조건부 실행(ARM) (Conditional Execution)

ARM 아키텍처는 조건 플래그가 설정된 경우에만 명령어가 실행되는 조건부 실행(predicated execution)을 지원합니다:

```
; Standard if-else:
CMP R0, #0
BEQ else
ADD R1, R2, R3      ; then branch
B   endif
else:
SUB R1, R2, R3      ; else branch
endif:

; Predicated (no branches, no pipeline stalls):
CMP R0, #0
ADDNE R1, R2, R3    ; execute if NE (not equal to 0)
SUBEQ R1, R2, R3    ; execute if EQ (equal to 0)
```

### 8.4 분기 예측 힌트(Branch Prediction Hints)

일부 아키텍처는 컴파일러가 예상되는 분기 방향에 대한 힌트를 제공할 수 있습니다:

```
; x86 branch hints (via instruction prefix)
; Prefix 0x3E: branch likely taken
; Prefix 0x2E: branch likely not taken

; GCC built-in:
if (__builtin_expect(error_condition, 0)) {
    // unlikely path
    handle_error();
}
```

---

## 9. 요약

이 레슨에서 코드 생성의 주요 단계를 다루었습니다:

1. **목표 기계 모델(Target machine model)**: 코드 생성 결정을 안내하는 여러 주소 지정 모드와 명령어 비용을 가진 간단한 RISC형 명령어 집합을 정의했습니다.

2. **명령어 선택(Instruction selection)**은 IR 트리를 기계 명령어로 매핑합니다. 타일을 이용한 **트리 패턴 매칭(tree pattern matching)**은 체계적인 접근 방식을 제공합니다. **최대 한입(Maximal Munch)** 알고리즘은 각 노드에서 탐욕적으로 가장 큰 매칭 타일을 선택합니다.

3. **레지스터 할당(Register allocation)**은 물리 레지스터를 가상 레지스터에 배정합니다. **그래프 색칠(graph coloring)**은 최적 할당을 제공하지만 비용이 큽니다. **선형 스캔(linear scan)**은 JIT 컴파일러에 적합한 빠른 대안을 제공합니다. 스필링(spilling)은 넘치는 변수를 메모리로 이동합니다.

4. **명령어 스케줄링(Instruction scheduling)**은 파이프라인 스톨을 최소화하기 위해 명령어를 재정렬합니다. **리스트 스케줄링(list scheduling)**은 의존성 DAG와 우선순위 휴리스틱을 사용합니다. **소프트웨어 파이프라이닝(software pipelining)**은 루프 반복을 겹칩니다.

5. **핍홀 최적화(Peephole optimization)**는 작은 명령어 윈도우에 로컬 변환을 적용합니다: 강도 감소, 중복 코드 제거, 대수적 단순화, 분기 최적화.

6. **언어 구성요소를 위한 코드 생성(Code generation for language constructs)**은 예측 가능한 패턴을 따릅니다: 식은 후위 순회(post-order traversal)를 사용하고, 제어 흐름은 조건부 분기와 레이블을 사용하며, 함수 호출은 호출 규약(calling convention)을 따릅니다.

7. **스택 기계(stack machine)**는 모든 연산이 암묵적 스택을 사용하는 간단한 코드 생성 목표를 제공하여 레지스터 할당의 필요성을 없앱니다.

---

## 연습 문제

### 연습 1: 최대 한입(Maximal Munch)

다음 IR 트리에 최대 한입 알고리즘을 적용하여 명령어를 선택하세요. 각 단계와 최종 명령어 시퀀스를 보여주세요.

```
        STORE
       /     \
      +       MEM
     / \       |
    FP  #-8   +
             / \
            *   #4
           / \
          i   #4
```

사용 가능한 타일: register, constant, ADD, ADDI, MUL, SHL (2의 거듭제곱 곱셈을 위한 왼쪽 시프트), LOAD with indexed addressing, STORE with indexed addressing.

### 연습 2: 레지스터 할당(Register Allocation)

다음 활성 구간과 3개의 물리 레지스터가 주어졌을 때, 선형 스캔 레지스터 할당을 수행하세요. 어떤 변수가 스필되나요?

```
a: [1, 15]
b: [2, 10]
c: [3, 12]
d: [5, 8]
e: [7, 20]
f: [13, 18]
```

### 연습 3: 명령어 스케줄링(Instruction Scheduling)

1개의 ALU 유닛과 1개의 로드/스토어 유닛을 가진 기계에서 다음 명령어들을 스케줄하세요. 로드 지연 시간은 3사이클, ALU 지연 시간은 1사이클입니다.

```
I1: LOAD R1, [addr1]      ; uses load unit, latency 3
I2: LOAD R2, [addr2]      ; uses load unit, latency 3
I3: ADD  R3, R1, R2       ; uses ALU, latency 1, depends on I1, I2
I4: LOAD R4, [addr3]      ; uses load unit, latency 3
I5: MUL  R5, R3, R4       ; uses ALU, latency 3, depends on I3, I4
I6: ADD  R6, R5, #1       ; uses ALU, latency 1, depends on I5
```

최소 사이클 수는 얼마인가요? 스케줄을 그려보세요.

### 연습 4: 핍홀 최적화(Peephole Optimization)

다음 코드에 핍홀 최적화를 적용하세요. 각 패스 후의 결과를 보여주세요.

```
  MOVI R1, #10
  ADD  R2, R1, #0
  MUL  R3, R2, #16
  MOV  R4, R4
  STORE R3, [R5]
  LOAD  R3, [R5]
  MUL  R6, R3, #1
  DIV  R7, R6, #8
  JMP  L1
  ADD  R8, R1, R2
L1:
  JMP  L2
L2:
  RET
```

### 연습 5: 스택 기계(Stack Machine)

다음 식을 스택 기계 바이트코드로 직접 컴파일하고 각 명령어 이후의 스택 내용을 추적하세요:

```
result = (3 + 4) * (10 - 2) / (1 + 1)
```

### 연습 6: 구현 도전

스택 기계 코드 생성기와 VM을 다음을 지원하도록 확장하세요:
1. **배열(Arrays)**: `ALLOC n` (크기 n의 배열 할당), `ALOAD` (배열에서 로드), `ASTORE` (배열에 저장)
2. **For 루프(For loops)**: `for i = start to end { body }`를 소스 레벨 구성요소로 구현

배열을 할당하고 제곱수로 채운 다음 내용을 출력하는 프로그램으로 테스트하세요.

---

[이전: 런타임 환경](./10_Runtime_Environments.md) | [다음: 최적화 — 지역 최적화와 전역 최적화](./12_Optimization_Local_and_Global.md) | [개요](./00_Overview.md)
