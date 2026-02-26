# 레슨 12: 최적화 -- 지역 최적화와 전역 최적화

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. **설명**: 컴파일러 최적화의 원칙 -- 안전성(safety), 수익성(profitability), 기회(opportunity)
2. **적용**: 단일 기본 블록 내 지역 최적화(상수 폴딩, CSE, 복사 전파, 죽은 코드 제거, 강도 감소)
3. **공식화**: 전역 데이터 흐름 분석(도달 정의, 가용 표현식, 활성 변수, 매우 바쁜 표현식)
4. **설명**: 격자(lattice)와 고정점 반복을 사용한 데이터 흐름 분석의 수학적 프레임워크
5. **구현**: 반복적 데이터 흐름 분석을 위한 워크리스트(worklist) 알고리즘
6. **구축**: Python으로 완전한 데이터 흐름 분석 엔진 구현

---

## 1. 최적화 개요

### 1.1 최적화란 무엇인가?

컴파일러 용어에서 **최적화(optimization)**란 프로그램의 관찰 가능한 동작을 보존하면서 어떤 지표 -- 일반적으로 실행 속도, 코드 크기, 또는 에너지 소비 -- 를 개선하는 프로그램 변환입니다.

"최적화"라는 용어는 다소 오해의 소지가 있습니다: 컴파일러가 진정으로 *최적의* 코드를 찾는 경우는 거의 없습니다. 더 정확하게는 컴파일러가 **코드 개선(code improvement)**을 수행한다고 할 수 있습니다.

### 1.2 세 가지 요구 사항

모든 최적화는 세 가지 기준을 충족해야 합니다:

1. **안전성(Safety/Correctness)**: 변환이 프로그램의 관찰 가능한 동작을 변경해서는 안 됩니다. 안전한 변환은 모든 가능한 입력에 대해 동일한 출력을 생성합니다.

2. **수익성(Profitability)**: 변환이 실제로 코드를 개선해야 합니다. 레지스터 압박(register pressure)을 높이고 더 많은 스필(spill)을 유발하는 변환은 명령어 수를 줄임에도 불구하고 코드를 더 느리게 만들 수 있습니다.

3. **기회(Opportunity)**: 최적화가 목표로 하는 패턴이 실제로 코드에 존재해야 합니다. 컴파일러는 변환을 적용하기 전에 기회를 **감지**해야 합니다.

### 1.3 최적화 분류

| 범위 | 설명 | 예시 |
|-------|-------------|---------|
| **지역(Local)** | 단일 기본 블록 내 | 상수 폴딩, CSE, 복사 전파 |
| **전역(Global)** (프로시저 내) | 하나의 함수 내 기본 블록 간 | 루프 불변 코드 이동, 전역 CSE |
| **프로시저 간(Interprocedural)** | 함수 경계 간 | 인라이닝, 프로시저 간 상수 전파 |
| **기계 종속(Machine-dependent)** | 대상 기계 특성 활용 | 명령어 스케줄링, 핍홀(peephole) 최적화 |
| **기계 독립(Machine-independent)** | 모든 대상에 적용 | 대부분의 IR 수준 최적화 |

### 1.4 최적화 적용 시점

```
Source → [Front End] → IR
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Machine-Independent Optimizations │
         │  (operate on IR)                   │
         │  - Constant propagation            │
         │  - Dead code elimination           │
         │  - Loop optimizations              │
         │  - Inlining                        │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Code Generation                   │
         │  (instruction selection, reg alloc) │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Machine-Dependent Optimizations   │
         │  (operate on machine code)         │
         │  - Peephole optimization           │
         │  - Instruction scheduling          │
         └──────────────────────────────────┘
                        │
                        ▼
                    Machine Code
```

### 1.5 단계 순서 문제(Phase-Ordering Problem)

최적화들은 서로 상호 작용합니다: 하나의 최적화가 다른 최적화의 기회를 만들거나, 이전 최적화의 작업을 되돌릴 수 있습니다. 최적화가 적용되는 순서는 중요하며, 최선의 순서를 찾는 것은 일반적으로 결정 불가능합니다.

실제 컴파일러는 신중하게 조정된 순서를 사용합니다 (예: GCC의 `-O2`는 특정 순서로 약 60개의 최적화 패스를 활성화합니다).

---

## 2. 지역 최적화

지역 최적화는 **단일 기본 블록** -- 시작과 끝을 제외하고 분기가 없는 직선형 명령어 시퀀스 -- 내에서 동작합니다. 기본 블록에는 하나의 실행 경로만 있으므로, 이러한 최적화에는 데이터 흐름 분석이 필요하지 않습니다.

### 2.1 상수 폴딩(Constant Folding)

**상수 폴딩(constant folding)**은 런타임이 아닌 컴파일 시간에 상수 표현식을 평가합니다.

**이전**:
```
t1 = 3 + 4
t2 = t1 * 2
```

**이후**:
```
t1 = 7
t2 = 14
```

**규칙**: 컴파일러는 $e_1$과 $e_2$ 모두 상수인 경우 $e_1 \;\text{op}\; e_2$ 표현식을 폴딩할 수 있습니다. 이는 재귀적으로 적용됩니다.

**주의 사항**: 컴파일러는 대상 기계의 산술 의미론을 존중해야 합니다:
- 정수 오버플로우 동작 (부호 있는 vs 부호 없는)
- 부동 소수점 반올림 모드
- 0으로 나누기 처리

### 2.2 상수 전파(Constant Propagation)

**상수 전파(constant propagation)**는 변수가 상수인 것으로 알려진 경우 변수의 사용을 해당 상수 값으로 교체합니다.

**이전**:
```
x = 5
y = x + 3
z = x * y
```

**이후** (전파 + 폴딩):
```
x = 5
y = 8       // 5 + 3
z = 40      // 5 * 8
```

상수 전파와 상수 폴딩의 조합은 매우 강력합니다 -- 각각이 연쇄적으로 다른 것을 가능하게 합니다.

### 2.3 대수적 단순화(Algebraic Simplification)

**대수적 단순화(algebraic simplification)**는 수학적 항등식을 적용하여 표현식을 단순화합니다:

| 원본 | 단순화 | 항등식 |
|----------|-----------|----------|
| $x + 0$ | $x$ | 덧셈 항등원 |
| $x - 0$ | $x$ | 덧셈 항등원 |
| $x \times 1$ | $x$ | 곱셈 항등원 |
| $x \times 0$ | $0$ | 영(zero) 성질 |
| $x / 1$ | $x$ | 나눗셈 항등원 |
| $x - x$ | $0$ | 자기 뺄셈 |
| $x / x$ | $1$ | 자기 나눗셈 ($x \neq 0$일 때) |
| $-(-x)$ | $x$ | 이중 부정 |
| $x + x$ | $2 \times x$ | (또는 1비트 왼쪽 시프트) |

**불리언의 경우**:

| 원본 | 단순화 |
|----------|-----------|
| $x \land \text{true}$ | $x$ |
| $x \land \text{false}$ | $\text{false}$ |
| $x \lor \text{true}$ | $\text{true}$ |
| $x \lor \text{false}$ | $x$ |
| $\lnot(\lnot x)$ | $x$ |

### 2.4 강도 감소(Strength Reduction) -- 지역

**강도 감소(strength reduction)**는 비용이 많이 드는 연산을 더 저렴한 연산으로 교체합니다:

| 비용이 많이 드는 연산 | 저렴한 연산 | 조건 |
|-----------|-------|-----------|
| $x \times 2^n$ | $x \ll n$ | 정수에 대해 항상 유효 |
| $x / 2^n$ | $x \gg n$ | 부호 없는 정수 |
| $x \% 2^n$ | $x \;\&\; (2^n - 1)$ | 부호 없는 정수 |
| $x \times 3$ | $(x \ll 1) + x$ | 시프트+덧셈이 더 저렴할 때 |
| $x \times 5$ | $(x \ll 2) + x$ | 시프트+덧셈이 더 저렴할 때 |
| $x \times 15$ | $(x \ll 4) - x$ | 시프트-뺄셈이 더 저렴할 때 |

**예시**:
```
이전: t1 = i * 4
이후:  t1 = i << 2
```

### 2.5 죽은 코드 제거(Dead Code Elimination) -- 지역

**죽은 코드(dead code)**는 결과가 절대 사용되지 않는 코드입니다. 이를 제거하면 코드 크기가 줄어들고 캐시 동작이 개선될 수 있습니다.

**이전**:
```
t1 = a + b      // t1 사용됨
t2 = c * d      // t2는 이후에 절대 사용되지 않음
t3 = t1 - 5     // t3 사용됨
```

**이후**:
```
t1 = a + b
t3 = t1 - 5
```

기본 블록에서, 변수가 재정의되기 전이나 블록 끝 전에 사용되지 않으면 (그리고 블록에서 나갈 때 활성 상태가 아니면) 정의는 죽어 있습니다.

### 2.6 공통 부분식 제거(Common Subexpression Elimination, CSE)

**CSE**는 두 번 이상 계산되는 표현식을 식별하고 중복 계산을 첫 번째 계산에 대한 참조로 교체합니다.

**이전**:
```
t1 = a + b
t2 = a + b      // t1과 동일한 표현식
t3 = t1 * t2
```

**이후**:
```
t1 = a + b
t2 = t1          // t1의 값을 재사용
t3 = t1 * t1
```

**안전 조건**: `a`도 `b`도 첫 번째 계산과 중복 계산 사이에 재정의되어서는 안 됩니다.

기본 블록 내 CSE는 **값 번호 매기기(value numbering)** (아래 참조) 또는 DAG 구성 (레슨 9 참조)을 사용하여 효율적으로 구현할 수 있습니다.

### 2.7 복사 전파(Copy Propagation)

**복사 전파(copy propagation)**는 복사 (`x = y`)로 할당된 변수의 사용을 원래 변수로 교체합니다.

**이전**:
```
t1 = a + b
t2 = t1          // 복사
t3 = t2 * c      // t2를 사용 (실제로는 t1)
```

**이후**:
```
t1 = a + b
t2 = t1           // 복사 (죽은 코드가 될 수 있음)
t3 = t1 * c       // t2를 t1로 교체
```

복사 전파 후, 복사 `t2 = t1`은 죽은 코드가 되어 제거될 수 있습니다.

### 2.8 지역 값 번호 매기기(Local Value Numbering)

**지역 값 번호 매기기(local value numbering)**는 기본 블록을 한 번 통과하면서 CSE, 상수 폴딩, 대수적 단순화를 수행하는 체계적인 방법입니다.

**아이디어**: 각 계산된 값에 **값 번호(value number)**를 할당합니다. 두 표현식이 같은 값을 계산하면 같은 값 번호를 얻습니다.

**알고리즘**:

```
hash_table = {}  # Maps (op, vn1, vn2) → value number
var_to_vn = {}   # Maps variable name → value number
next_vn = 0

for each instruction "x = y op z":
    vn_y = var_to_vn[y]
    vn_z = var_to_vn[z]

    key = (op, vn_y, vn_z)
    if key in hash_table:
        vn_x = hash_table[key]   # Same value already computed
        # Replace instruction with copy from the representative
    else:
        vn_x = next_vn++
        hash_table[key] = vn_x

    var_to_vn[x] = vn_x
```

### 2.9 Python 구현: 지역 최적화기

```python
"""
Local optimizations within a single basic block.
Implements constant folding, constant propagation, algebraic
simplification, strength reduction, CSE, copy propagation,
and dead code elimination.
"""

from dataclasses import dataclass
from typing import Optional, Union
import operator


@dataclass
class TAC:
    """Three-address code instruction."""
    result: str
    op: str           # '+', '-', '*', '/', 'copy', 'nop'
    arg1: str = ""
    arg2: str = ""

    def __str__(self):
        if self.op == "copy":
            return f"{self.result} = {self.arg1}"
        elif self.op == "nop":
            return f"// NOP (eliminated)"
        elif self.arg2:
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.arg1:
            return f"{self.result} = {self.op} {self.arg1}"
        else:
            return f"{self.result} = {self.op}"


def is_constant(s: str) -> bool:
    """Check if a string represents an integer constant."""
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def const_val(s: str) -> int:
    """Get the integer value of a constant string."""
    return int(s)


class LocalOptimizer:
    """
    Performs local optimizations on a basic block (list of TAC instructions).
    """

    def __init__(self, instructions: list[TAC], live_out: set = None):
        self.instructions = list(instructions)
        self.live_out = live_out or set()  # Variables live at exit
        self.changed = False

    def optimize(self, max_passes: int = 10) -> list[TAC]:
        """Run all local optimizations until convergence."""
        for _ in range(max_passes):
            self.changed = False
            self.constant_folding()
            self.constant_propagation()
            self.algebraic_simplification()
            self.strength_reduction()
            self.copy_propagation()
            self.common_subexpression_elimination()
            self.dead_code_elimination()
            if not self.changed:
                break

        # Remove NOP instructions
        return [i for i in self.instructions if i.op != "nop"]

    def constant_folding(self):
        """Evaluate constant expressions at compile time."""
        ops = {
            '+': operator.add, '-': operator.sub,
            '*': operator.mul, '/': operator.floordiv,
            '%': operator.mod,
        }

        for i, instr in enumerate(self.instructions):
            if instr.op in ops and is_constant(instr.arg1) \
               and is_constant(instr.arg2):
                a = const_val(instr.arg1)
                b = const_val(instr.arg2)
                if instr.op in ('/', '%') and b == 0:
                    continue  # Cannot fold division by zero
                result = ops[instr.op](a, b)
                self.instructions[i] = TAC(instr.result, "copy", str(result))
                self.changed = True

    def constant_propagation(self):
        """Replace variable uses with constant values when known."""
        constants = {}  # var -> constant value

        for i, instr in enumerate(self.instructions):
            # Check if this defines a constant
            if instr.op == "copy" and is_constant(instr.arg1):
                constants[instr.result] = instr.arg1

            # Replace uses with constants
            new_arg1 = instr.arg1
            new_arg2 = instr.arg2

            if instr.arg1 in constants and instr.op != "copy":
                new_arg1 = constants[instr.arg1]
            elif instr.arg1 in constants and instr.op == "copy":
                new_arg1 = constants[instr.arg1]

            if instr.arg2 in constants:
                new_arg2 = constants[instr.arg2]

            if new_arg1 != instr.arg1 or new_arg2 != instr.arg2:
                self.instructions[i] = TAC(instr.result, instr.op,
                                           new_arg1, new_arg2)
                self.changed = True

            # If this instruction redefines a variable, invalidate it
            if instr.result in constants and instr.op != "copy":
                del constants[instr.result]
            elif (instr.result in constants and instr.op == "copy"
                  and not is_constant(instr.arg1)):
                del constants[instr.result]

    def algebraic_simplification(self):
        """Apply algebraic identities."""
        for i, instr in enumerate(self.instructions):
            result = instr.result
            new_instr = None

            # x + 0 = x, 0 + x = x
            if instr.op == '+':
                if instr.arg2 == '0':
                    new_instr = TAC(result, "copy", instr.arg1)
                elif instr.arg1 == '0':
                    new_instr = TAC(result, "copy", instr.arg2)

            # x - 0 = x
            elif instr.op == '-':
                if instr.arg2 == '0':
                    new_instr = TAC(result, "copy", instr.arg1)
                # x - x = 0
                elif instr.arg1 == instr.arg2:
                    new_instr = TAC(result, "copy", "0")

            # x * 1 = x, 1 * x = x, x * 0 = 0
            elif instr.op == '*':
                if instr.arg2 == '1':
                    new_instr = TAC(result, "copy", instr.arg1)
                elif instr.arg1 == '1':
                    new_instr = TAC(result, "copy", instr.arg2)
                elif instr.arg2 == '0' or instr.arg1 == '0':
                    new_instr = TAC(result, "copy", "0")

            # x / 1 = x
            elif instr.op == '/':
                if instr.arg2 == '1':
                    new_instr = TAC(result, "copy", instr.arg1)
                # x / x = 1 (assume x != 0)
                elif instr.arg1 == instr.arg2:
                    new_instr = TAC(result, "copy", "1")

            if new_instr:
                self.instructions[i] = new_instr
                self.changed = True

    def strength_reduction(self):
        """Replace expensive operations with cheaper ones."""
        for i, instr in enumerate(self.instructions):
            if instr.op == '*' and is_constant(instr.arg2):
                val = const_val(instr.arg2)
                if val > 0 and (val & (val - 1)) == 0:
                    shift = val.bit_length() - 1
                    self.instructions[i] = TAC(
                        instr.result, "<<", instr.arg1, str(shift)
                    )
                    self.changed = True

            elif instr.op == '/' and is_constant(instr.arg2):
                val = const_val(instr.arg2)
                if val > 0 and (val & (val - 1)) == 0:
                    shift = val.bit_length() - 1
                    self.instructions[i] = TAC(
                        instr.result, ">>", instr.arg1, str(shift)
                    )
                    self.changed = True

    def copy_propagation(self):
        """Replace uses of copied variables with the source."""
        copies = {}  # target -> source

        for i, instr in enumerate(self.instructions):
            # Record copies
            if instr.op == "copy" and not is_constant(instr.arg1):
                copies[instr.result] = instr.arg1

            # Propagate copies in uses
            new_arg1 = copies.get(instr.arg1, instr.arg1)
            new_arg2 = copies.get(instr.arg2, instr.arg2)

            if new_arg1 != instr.arg1 or new_arg2 != instr.arg2:
                self.instructions[i] = TAC(instr.result, instr.op,
                                           new_arg1, new_arg2)
                self.changed = True

            # Invalidate if source is redefined
            to_remove = []
            for target, source in copies.items():
                if source == instr.result or target == instr.result:
                    to_remove.append(target)
            for t in to_remove:
                if t != instr.result or instr.op != "copy":
                    copies.pop(t, None)

    def common_subexpression_elimination(self):
        """Eliminate redundant computations using value numbering."""
        # (op, arg1, arg2) -> result variable
        computed = {}

        for i, instr in enumerate(self.instructions):
            if instr.op in ('+', '-', '*', '/', '%', '<<', '>>'):
                key = (instr.op, instr.arg1, instr.arg2)

                # Check commutative operators
                if instr.op in ('+', '*'):
                    comm_key = (instr.op, instr.arg2, instr.arg1)
                    if comm_key in computed:
                        key = comm_key

                if key in computed:
                    # Replace with copy from previous computation
                    prev_result = computed[key]
                    self.instructions[i] = TAC(
                        instr.result, "copy", prev_result
                    )
                    self.changed = True
                else:
                    computed[key] = instr.result

                # Invalidate entries if arg is redefined
                to_remove = [k for k, v in computed.items()
                             if instr.result in k[1:]]
                for k in to_remove:
                    del computed[k]

    def dead_code_elimination(self):
        """Remove instructions whose results are never used."""
        # Find all used variables (scan backward)
        used = set(self.live_out)

        # Scan backward to find used variables
        for instr in reversed(self.instructions):
            if instr.op == "nop":
                continue
            if instr.arg1 and not is_constant(instr.arg1):
                used.add(instr.arg1)
            if instr.arg2 and not is_constant(instr.arg2):
                used.add(instr.arg2)

        # Mark instructions as dead if result is not used
        for i, instr in enumerate(self.instructions):
            if instr.op == "nop":
                continue
            if instr.result not in used and instr.result not in self.live_out:
                self.instructions[i] = TAC("", "nop")
                self.changed = True


def demo_local_optimizations():
    """Demonstrate local optimizations on a basic block."""

    instructions = [
        TAC("t1", "copy", "5"),         # t1 = 5
        TAC("t2", "copy", "3"),         # t2 = 3
        TAC("t3", "+", "t1", "t2"),     # t3 = t1 + t2 → fold to 8
        TAC("t4", "*", "t3", "0"),      # t4 = t3 * 0 → simplify to 0
        TAC("t5", "+", "a", "b"),       # t5 = a + b
        TAC("t6", "+", "a", "b"),       # t6 = a + b  (CSE)
        TAC("t7", "*", "t5", "t6"),     # t7 = t5 * t6 → t5 * t5
        TAC("t8", "*", "t7", "8"),      # t8 = t7 * 8 → strength reduce
        TAC("t9", "copy", "t8"),        # t9 = t8 (copy)
        TAC("t10", "+", "t9", "0"),     # t10 = t9 + 0 → simplify
        TAC("unused", "+", "t1", "t2"), # dead code (unused not live-out)
    ]

    live_out = {"t8", "t10"}

    print("=== Before Optimization ===")
    for instr in instructions:
        print(f"  {instr}")

    optimizer = LocalOptimizer(instructions, live_out)
    optimized = optimizer.optimize()

    print("\n=== After Optimization ===")
    for instr in optimized:
        print(f"  {instr}")


if __name__ == "__main__":
    demo_local_optimizations()
```

---

## 3. 전역 데이터 흐름 분석

### 3.1 전역 분석이 필요한 이유

지역 최적화는 단일 기본 블록으로 제한됩니다. 블록 경계를 넘어 최적화하려면 전체 제어 흐름 그래프(CFG)를 통한 데이터 흐름을 추론하는 **전역**(프로시저 내) 분석이 필요합니다.

**예시**: 특정 지점에서 변수가 상수 값을 가지는지 결정하려면, 해당 지점에 도달할 수 있는 모든 경로를 고려해야 합니다 -- 이는 전체 CFG 분석을 필요로 합니다.

### 3.2 네 가지 고전적 분석

네 가지 기반이 되는 데이터 흐름 분석은 추적하는 것과 정보가 흐르는 방향이 다릅니다:

| 분석 | 방향 | 합류 연산자 | 질문 |
|----------|-----------|---------------|----------|
| **도달 정의(Reaching Definitions)** | 순방향 | 합류점 ($\cup$) | $x$의 어떤 정의가 이 지점에 도달할 수 있는가? |
| **가용 표현식(Available Expressions)** | 순방향 | 합류점 ($\cap$) | 어떤 표현식이 계산되었다고 보장되는가? |
| **활성 변수(Live Variables)** | 역방향 | 합류점 ($\cup$) | 어떤 변수가 재정의 전에 사용될 수 있는가? |
| **매우 바쁜 표현식(Very Busy Expressions)** | 역방향 | 합류점 ($\cap$) | 어떤 표현식이 모든 경로에서 반드시 평가되는가? |

### 3.3 도달 정의(Reaching Definitions)

**정의(definition)**는 변수에 값을 할당하는 명령어입니다 (예: `d: x = ...`).

변수 $x$의 정의 $d$가 지점 $p$에 **도달(reaches)**하는 경우:
- $d$에서 $p$까지의 경로가 존재하고,
- 그 경로를 따라 $x$가 재정의되지 않은 경우

**사용처**: 상수 전파 (도달 정의가 하나뿐이고 상수를 할당하는 경우 전파 가능).

#### 데이터 흐름 방정식

각 기본 블록 $B$에 대해:

$$\text{Out}(B) = \text{Gen}(B) \cup (\text{In}(B) - \text{Kill}(B))$$

$$\text{In}(B) = \bigcup_{P \in \text{pred}(B)} \text{Out}(P)$$

여기서:
- $\text{Gen}(B)$ = $B$에서 생성되어 $B$의 끝까지 살아남는 정의
- $\text{Kill}(B)$ = $B$가 제거하는 정의 ($B$가 재정의하는 변수의 정의)

**방향**: 순방향 (정보가 선행 블록에서 후속 블록으로 흐름)

**합류 연산자**: 합집합($\cup$) -- 정의가 *임의의* 경로를 통해 도달하면 도달함

**초기화**: 모든 블록에 대해 $\text{Out}(B) = \text{Gen}(B)$; $\text{In}(\text{entry}) = \emptyset$

#### 예시

```
B1: d1: x = 5
    d2: y = 1
    → B2

B2: d3: z = x + y
    if z < 10 → B3 else → B4

B3: d4: x = x + 1
    d5: y = y * 2
    → B2

B4: print z
```

**Gen과 Kill 집합**:

| 블록 | Gen | Kill |
|-------|-----|------|
| B1 | {d1, d2} | {d4, d5} (x, y의 다른 정의) |
| B2 | {d3} | {} (z의 다른 정의 없음) |
| B3 | {d4, d5} | {d1, d2} (x, y의 다른 정의) |
| B4 | {} | {} |

**반복**:

| 반복 | In(B1) | Out(B1) | In(B2) | Out(B2) | In(B3) | Out(B3) | In(B4) | Out(B4) |
|-----------|--------|---------|--------|---------|--------|---------|--------|---------|
| Init | {} | {d1,d2} | {} | {d3} | {} | {d4,d5} | {} | {} |
| 1 | {} | {d1,d2} | {d1,d2,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} | {d3,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} |
| 2 | {} | {d1,d2} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} | {d3,d4,d5} | {d1,d2,d3,d4,d5} | {d1,d2,d3,d4,d5} |

B2에서 $d_1$과 $d_4$ 모두 $x$를 정의하므로, $x$에 대한 단일 상수를 전파할 수 없습니다.

### 3.4 가용 표현식(Available Expressions)

표현식 $e$가 지점 $p$에서 **가용(available)**한 경우:
- 진입점에서 $p$까지의 **모든** 경로에서 $e$가 계산되고,
- $e$의 마지막 계산 이후에 $e$의 피연산자가 재정의되지 않은 경우

**사용처**: 전역 공통 부분식 제거. $e$가 $p$에서 가용하고 $p$가 $e$를 계산하면, $p$의 계산을 이전에 계산된 값으로 교체할 수 있습니다.

#### 데이터 흐름 방정식

$$\text{Out}(B) = \text{Gen}(B) \cup (\text{In}(B) - \text{Kill}(B))$$

$$\text{In}(B) = \bigcap_{P \in \text{pred}(B)} \text{Out}(P)$$

**방향**: 순방향

**합류 연산자**: 교집합($\cap$) -- 표현식은 *모든* 경로에서 가용할 때만 가용함

**초기화**: $\text{Out}(\text{entry}) = \emptyset$; $B \neq \text{entry}$인 경우 $\text{Out}(B) = U$ (모든 표현식)

교집합 사용이 핵심입니다: 표현식은 *모든* 진입 경로를 따라 계산된 경우에만 가용합니다. 이는 "must" 분석 (모든 경로에서 참이어야 함)으로, 도달 정의의 "may" 분석 (임의의 경로를 통해 도달할 수 있음)과 대조됩니다.

### 3.5 활성 변수(Live Variables)

변수 $v$가 지점 $p$에서 **활성(live)**인 경우:
- $p$에서 $v$의 사용까지 경로가 존재하고,
- 그 경로를 따라 $v$가 재정의되지 않은 경우

**사용처**: 레지스터 할당 (활성 변수는 레지스터가 필요), 죽은 코드 제거 (정의 이후 변수가 활성이 아니면 정의는 죽은 코드).

#### 데이터 흐름 방정식

$$\text{In}(B) = \text{Use}(B) \cup (\text{Out}(B) - \text{Def}(B))$$

$$\text{Out}(B) = \bigcup_{S \in \text{succ}(B)} \text{In}(S)$$

여기서:
- $\text{Use}(B)$ = $B$에서 정의되기 전에 사용되는 변수
- $\text{Def}(B)$ = $B$에서 정의되는 변수

**방향**: 역방향 (정보가 후속 블록에서 선행 블록으로 흐름)

**합류 연산자**: 합집합($\cup$) -- 변수가 *임의의* 앞으로의 경로에서 필요하면 활성

**초기화**: 모든 블록에 대해 $\text{In}(B) = \emptyset$; $\text{Out}(\text{exit}) = \emptyset$ (또는 함수가 반환된 후 필요한 변수 집합)

### 3.6 매우 바쁜 표현식(Very Busy Expressions)

표현식 $e$가 지점 $p$에서 **매우 바쁜(very busy)** 경우:
- $p$에서 출구까지의 **모든** 경로에서, 피연산자가 재정의되기 전에 $e$가 평가되는 경우

**사용처**: 코드 끌어올리기(code hoisting) -- $e$가 $p$에서 매우 바쁘다면, $e$의 계산을 $p$로 (더 이르게) 안전하게 이동할 수 있어 코드 크기를 줄일 수 있습니다.

#### 데이터 흐름 방정식

$$\text{In}(B) = \text{Gen}(B) \cup (\text{Out}(B) - \text{Kill}(B))$$

$$\text{Out}(B) = \bigcap_{S \in \text{succ}(B)} \text{In}(S)$$

**방향**: 역방향

**합류 연산자**: 교집합($\cap$) -- 표현식은 *모든* 경로에서 평가될 때만 매우 바쁨

### 3.7 네 가지 분석 요약

| | 도달 정의 | 가용 표현식 | 활성 변수 | 매우 바쁜 표현식 |
|--|---------------|-----------------|-----------|-----------------|
| **도메인** | 정의 집합 | 표현식 집합 | 변수 집합 | 표현식 집합 |
| **방향** | 순방향 | 순방향 | 역방향 | 역방향 |
| **합류** | $\cup$ (may) | $\cap$ (must) | $\cup$ (may) | $\cap$ (must) |
| **전달 함수** | $\text{Gen} \cup (\text{In} - \text{Kill})$ | $\text{Gen} \cup (\text{In} - \text{Kill})$ | $\text{Use} \cup (\text{Out} - \text{Def})$ | $\text{Gen} \cup (\text{Out} - \text{Kill})$ |
| **초기값 (경계)** | $\text{Out}(\text{entry}) = \emptyset$ | $\text{Out}(\text{entry}) = \emptyset$ | $\text{In}(\text{exit}) = \emptyset$ | $\text{In}(\text{exit}) = \emptyset$ |
| **초기값 (기타)** | $\emptyset$ | $U$ (모든 표현식) | $\emptyset$ | $U$ (모든 표현식) |
| **사용** | 상수 전파 | 전역 CSE | 레지스터 할당, DCE | 코드 끌어올리기 |

---

## 4. 데이터 흐름 분석 프레임워크

### 4.1 격자 이론 기초

데이터 흐름 분석은 **격자 이론(lattice theory)**에 기반한 수학적 프레임워크로 통합될 수 있습니다.

**정의**: **격자(lattice)** $(L, \sqsubseteq)$는 모든 원소 쌍이 **만남(meet)** ($\sqcap$, 최대 하한) 과 **결합(join)** ($\sqcup$, 최소 상한)을 가지는 부분 순서 집합입니다.

데이터 흐름 분석에서는 일반적으로 모든 부분집합이 만남과 결합을 가지는 **완전 격자(complete lattice)**를 사용합니다. 주요 원소:

- $\top$ (top): 최대 원소 -- "아직 정보 없음" 또는 "모든 가능성"을 나타냄
- $\bot$ (bottom): 최소 원소 -- "도달 불가" 또는 "모순"을 나타냄

#### 도달 정의를 위한 격자

격자는 모든 정의의 **멱집합(powerset)**으로, 부분집합 포함 관계로 순서화됩니다:

$$L = 2^{\text{Defs}}, \quad A \sqsubseteq B \iff A \subseteq B$$

- $\top = \text{Defs}$ (모든 정의 -- 전체 집합)
- $\bot = \emptyset$ (정의 없음)
- 만남($\sqcap$) = 합집합($\cup$): may 분석 사용 (임의의 경로)

잠깐 -- 표준 공식화에서 규약은 격자 방향에 따라 달라집니다. 주의가 필요합니다.

**may 분석** (도달 정의 등)에서 격자는 $\bot = \emptyset$ (정보 없음)에서 위로 올라가고, 만남은 $\cup$ (합집합)입니다. 가장 낙관적인 값($\bot = \emptyset$: "여기에 아무것도 도달하지 않음")에서 시작하여 고정점에 도달할 때까지 정보를 추가합니다.

**must 분석** (가용 표현식 등)에서 격자는 $\top = U$ (모든 표현식이 가용)에서 아래로 내려가고, 만남은 $\cap$ (교집합)입니다. 가장 낙관적인 값($\top = U$: "모든 것이 가용함")에서 시작하여 고정점에 도달할 때까지 정보를 제거합니다.

#### 일반 프레임워크

데이터 흐름 분석은 다음으로 정의됩니다:

1. 만남 $\sqcap$을 가진 **격자** $(L, \sqsubseteq)$
2. 각 블록 $B$에 대한 **전달 함수(transfer function)** $f_B : L \to L$
3. **방향** (순방향 또는 역방향)
4. **경계 조건** (진입/출구 블록의 초기값)
5. 다른 모든 블록의 **초기값**

해는 방정식 시스템의 **최대 고정점(greatest fixed point)**입니다.

### 4.2 전달 함수

전달 함수 $f_B$는 블록 $B$를 통과할 때 정보가 어떻게 변하는지 설명합니다. 대부분의 데이터 흐름 분석은 다음 형식의 전달 함수를 사용합니다:

$$f_B(X) = \text{Gen}(B) \cup (X - \text{Kill}(B))$$

이것은 **단조(monotone)** 함수입니다: $X \sqsubseteq Y$이면 $f_B(X) \sqsubseteq f_B(Y)$.

**단조성은 필수적입니다**: 반복 알고리즘의 수렴을 보장합니다.

### 4.3 경로 위의 만남(Meet-Over-Paths, MOP) 해

**이상적인** 해는 **모든 경로 위의 만남(Meet-Over-All-Paths, MOP)** 해입니다. 순방향 분석에서 블록 $B$의 진입점의 MOP 해는:

$$\text{MOP}(B) = \bigsqcap_{\text{path } p : \text{entry} \to B} f_p(\text{boundary\_value})$$

여기서 $f_p$는 경로 $p$를 따른 전달 함수의 합성입니다.

MOP를 직접 계산하는 것은 일반적으로 결정 불가능합니다 (루프 때문에 무한히 많은 경로가 있을 수 있습니다). 대신 **최대 고정점(Maximum Fixed Point, MFP)**을 계산하는데, 이는 MOP보다 보수적임이 보장됩니다 (즉, 적절한 순서에서 $\text{MFP} \sqsubseteq \text{MOP}$).

$f(X) = \text{Gen} \cup (X - \text{Kill})$ 형식의 전달 함수 (**분배적(distributive)**)의 경우, MFP = MOP가 정확히 성립합니다.

### 4.4 고정점 반복

**정리 (타르스키, Tarski)**: $f$가 완전 격자 위의 단조 함수이면, $f$는 최소 고정점을 가지며, 이는 $\bot$에서 반복하여 얻을 수 있습니다:

$$\bot, f(\bot), f^2(\bot), \ldots$$

데이터 흐름 분석에서는 어떤 값도 변하지 않을 때까지 모든 블록에 대해 동시에 반복합니다:

```
Algorithm: Iterative Data Flow Analysis (Forward)

Initialize:
    Out(entry) = boundary_value
    Out(B) = initial_value for all B ≠ entry

Repeat:
    for each block B (in some order):
        In(B) = Meet over all predecessors P of Out(P)
        Out(B) = f_B(In(B))
until no Out(B) changed
```

**수렴**: 다음으로 보장됩니다:
1. 격자의 높이가 유한합니다 (정의/표현식의 유한한 수)
2. 전달 함수가 단조입니다
3. 각 반복에서 값이 한 방향으로 (격자에서 위 또는 아래로) 이동합니다

**최악의 경우 반복 횟수**: 최대 $h \times |V|$ (여기서 $h$는 격자 높이, $|V|$는 CFG 노드 수). 실제로는 보통 2--3번의 반복으로 충분합니다.

### 4.5 워크리스트 알고리즘

단순한 반복 알고리즘은 대부분이 변하지 않았어도 모든 블록을 반복적으로 스캔합니다. **워크리스트 알고리즘(worklist algorithm)**은 입력이 변한 블록만 처리합니다:

```
Algorithm: Worklist Data Flow Analysis (Forward)

Initialize:
    Out(entry) = boundary_value
    Out(B) = initial_value for all B ≠ entry
    worklist = all blocks except entry

While worklist is not empty:
    Remove a block B from the worklist
    In(B) = Meet over predecessors P of Out(P)
    new_out = f_B(In(B))
    if new_out ≠ Out(B):
        Out(B) = new_out
        Add all successors of B to the worklist
```

**효율성**: 워크리스트 알고리즘은 중복 작업을 피합니다. 순방향 분석에 역후위 순서(reverse postorder)를 (또는 역방향 분석에 후위 순서를) 초기 워크리스트 순서로 사용하면 수렴이 향상됩니다.

---

## 5. Python 구현: 데이터 흐름 분석

```python
"""
Generic Data Flow Analysis Framework.

Implements the worklist algorithm for forward and backward analyses.
Includes implementations of all four classic analyses.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum, auto
from abc import ABC, abstractmethod


# ============================================================
# CFG Data Structures
# ============================================================

@dataclass
class Instruction:
    """A three-address code instruction."""
    index: int
    result: str = ""
    op: str = ""
    arg1: str = ""
    arg2: str = ""
    label: str = ""

    @property
    def defined_var(self) -> str:
        """Variable defined by this instruction."""
        if self.op and self.result and self.op not in ("goto", "iffalse",
                                                        "iftrue", "label",
                                                        "return"):
            return self.result
        return ""

    @property
    def used_vars(self) -> set:
        """Variables used by this instruction."""
        used = set()
        for arg in (self.arg1, self.arg2):
            if arg and not arg.isdigit() and not arg.startswith("L") \
               and not arg.startswith("#"):
                used.add(arg)
        return used

    def __str__(self):
        if self.op == "label":
            return f"{self.result}:"
        elif self.op == "copy":
            return f"{self.result} = {self.arg1}"
        elif self.arg2:
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.op in ("goto", "iffalse", "iftrue"):
            return f"{self.op} {self.arg1} {self.result}"
        elif self.op == "return":
            return f"return {self.arg1}"
        else:
            return f"{self.result} = {self.op} {self.arg1}"


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    id: int
    instructions: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)

    def __str__(self):
        lines = [f"B{self.id}:"]
        for instr in self.instructions:
            lines.append(f"  {instr}")
        return "\n".join(lines)


class CFG:
    """Control flow graph."""

    def __init__(self):
        self.blocks: dict[int, BasicBlock] = {}

    def add_block(self, block: BasicBlock):
        self.blocks[block.id] = block

    def add_edge(self, from_id: int, to_id: int):
        if to_id not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(to_id)
        if from_id not in self.blocks[to_id].predecessors:
            self.blocks[to_id].predecessors.append(from_id)


# ============================================================
# Data Flow Analysis Framework
# ============================================================

class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class MeetOp(Enum):
    UNION = auto()        # May analysis
    INTERSECTION = auto()  # Must analysis


class DataFlowAnalysis(ABC):
    """
    Abstract base class for data flow analyses.
    """

    def __init__(self, cfg: CFG, direction: Direction, meet_op: MeetOp):
        self.cfg = cfg
        self.direction = direction
        self.meet_op = meet_op
        self.in_sets: dict[int, set] = {}
        self.out_sets: dict[int, set] = {}

    @abstractmethod
    def gen(self, block: BasicBlock) -> set:
        """Compute the Gen set for a block."""
        pass

    @abstractmethod
    def kill(self, block: BasicBlock) -> set:
        """Compute the Kill set for a block."""
        pass

    @abstractmethod
    def universe(self) -> set:
        """Return the universal set for must-analyses."""
        pass

    @abstractmethod
    def boundary_value(self) -> set:
        """Return the boundary value (for entry/exit block)."""
        pass

    def meet(self, sets: list[set]) -> set:
        """Apply the meet operator to a list of sets."""
        if not sets:
            if self.meet_op == MeetOp.UNION:
                return set()
            else:
                return self.universe()

        result = sets[0].copy()
        for s in sets[1:]:
            if self.meet_op == MeetOp.UNION:
                result = result | s
            else:
                result = result & s
        return result

    def transfer(self, block: BasicBlock, input_set: set) -> set:
        """Apply the transfer function: Gen ∪ (input - Kill)."""
        return self.gen(block) | (input_set - self.kill(block))

    def initial_value(self) -> set:
        """Initial value for non-boundary blocks."""
        if self.meet_op == MeetOp.UNION:
            return set()       # Most optimistic for may-analyses
        else:
            return self.universe()  # Most optimistic for must-analyses

    def solve(self):
        """
        Solve the data flow equations using the worklist algorithm.
        """
        block_ids = sorted(self.cfg.blocks.keys())

        if self.direction == Direction.FORWARD:
            self._solve_forward(block_ids)
        else:
            self._solve_backward(block_ids)

    def _solve_forward(self, block_ids):
        """Forward worklist algorithm."""
        entry_id = block_ids[0]

        # Initialize
        for bid in block_ids:
            self.in_sets[bid] = set()
            if bid == entry_id:
                self.out_sets[bid] = self.boundary_value()
            else:
                self.out_sets[bid] = self.initial_value()

        # Worklist (use reverse postorder for efficiency)
        worklist = deque(block_ids[1:])  # Exclude entry initially
        in_worklist = set(worklist)

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)
            block = self.cfg.blocks[bid]

            # Compute In
            pred_outs = [self.out_sets[p] for p in block.predecessors]
            new_in = self.meet(pred_outs)
            self.in_sets[bid] = new_in

            # Compute Out via transfer function
            new_out = self.transfer(block, new_in)

            if new_out != self.out_sets[bid]:
                self.out_sets[bid] = new_out
                for succ in block.successors:
                    if succ not in in_worklist:
                        worklist.append(succ)
                        in_worklist.add(succ)

    def _solve_backward(self, block_ids):
        """Backward worklist algorithm."""
        # Find exit blocks (no successors)
        exit_ids = [bid for bid in block_ids
                    if not self.cfg.blocks[bid].successors]

        # Initialize
        for bid in block_ids:
            self.out_sets[bid] = set()
            if bid in exit_ids:
                self.in_sets[bid] = self.boundary_value()
            else:
                self.in_sets[bid] = self.initial_value()

        # Worklist
        worklist = deque(bid for bid in block_ids if bid not in exit_ids)
        in_worklist = set(worklist)

        # Also add exit blocks to process their predecessors
        for eid in exit_ids:
            block = self.cfg.blocks[eid]
            self.out_sets[eid] = set()
            self.in_sets[eid] = self.transfer(block, self.out_sets[eid])

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)
            block = self.cfg.blocks[bid]

            # Compute Out from successors
            succ_ins = [self.in_sets[s] for s in block.successors]
            new_out = self.meet(succ_ins)
            self.out_sets[bid] = new_out

            # Compute In via transfer function
            new_in = self.transfer(block, new_out)

            if new_in != self.in_sets[bid]:
                self.in_sets[bid] = new_in
                for pred in block.predecessors:
                    if pred not in in_worklist:
                        worklist.append(pred)
                        in_worklist.add(pred)

    def print_results(self, analysis_name: str):
        """Pretty-print the analysis results."""
        print(f"\n=== {analysis_name} ===")
        for bid in sorted(self.cfg.blocks.keys()):
            in_s = sorted(self.in_sets.get(bid, set()))
            out_s = sorted(self.out_sets.get(bid, set()))
            print(f"  B{bid}: In = {{{', '.join(str(x) for x in in_s)}}}")
            print(f"       Out = {{{', '.join(str(x) for x in out_s)}}}")


# ============================================================
# Concrete Analyses
# ============================================================

class ReachingDefinitions(DataFlowAnalysis):
    """Reaching Definitions analysis."""

    def __init__(self, cfg: CFG):
        super().__init__(cfg, Direction.FORWARD, MeetOp.UNION)
        # Pre-compute all definitions
        self.all_defs: dict[str, set] = defaultdict(set)  # var -> {def labels}
        for bid, block in cfg.blocks.items():
            for instr in block.instructions:
                dvar = instr.defined_var
                if dvar:
                    def_label = f"d{instr.index}:{dvar}"
                    self.all_defs[dvar].add(def_label)

    def gen(self, block: BasicBlock) -> set:
        gen_set = set()
        killed_in_block = set()  # Track vars defined in this block
        # Process in order; last def of each var survives
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                # Kill previous defs of this var in this block
                gen_set = {d for d in gen_set
                           if not d.endswith(f":{dvar}")}
                gen_set.add(f"d{instr.index}:{dvar}")
        return gen_set

    def kill(self, block: BasicBlock) -> set:
        kill_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                # Kill all other defs of this var
                for d in self.all_defs[dvar]:
                    if not d.startswith(f"d{instr.index}:"):
                        kill_set.add(d)
        return kill_set

    def universe(self) -> set:
        return set().union(*self.all_defs.values()) if self.all_defs else set()

    def boundary_value(self) -> set:
        return set()


class LiveVariables(DataFlowAnalysis):
    """Live Variables analysis."""

    def __init__(self, cfg: CFG, exit_live: set = None):
        super().__init__(cfg, Direction.BACKWARD, MeetOp.UNION)
        self.exit_live = exit_live or set()

    def gen(self, block: BasicBlock) -> set:
        """Use set: variables used before being defined in this block."""
        use_set = set()
        defined = set()
        for instr in block.instructions:
            # Uses that haven't been defined yet in this block
            for v in instr.used_vars:
                if v not in defined:
                    use_set.add(v)
            # Definition
            dvar = instr.defined_var
            if dvar:
                defined.add(dvar)
        return use_set

    def kill(self, block: BasicBlock) -> set:
        """Def set: variables defined in this block."""
        def_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                def_set.add(dvar)
        return def_set

    def universe(self) -> set:
        all_vars = set()
        for block in self.cfg.blocks.values():
            for instr in block.instructions:
                all_vars |= instr.used_vars
                if instr.defined_var:
                    all_vars.add(instr.defined_var)
        return all_vars

    def boundary_value(self) -> set:
        return self.exit_live


class AvailableExpressions(DataFlowAnalysis):
    """Available Expressions analysis."""

    def __init__(self, cfg: CFG):
        super().__init__(cfg, Direction.FORWARD, MeetOp.INTERSECTION)
        # Collect all expressions
        self.all_exprs = set()
        for block in cfg.blocks.values():
            for instr in block.instructions:
                if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                    self.all_exprs.add(f"{instr.arg1} {instr.op} {instr.arg2}")

    def gen(self, block: BasicBlock) -> set:
        gen_set = set()
        killed_vars = set()  # Variables defined so far
        for instr in block.instructions:
            # Add expression if operands haven't been killed
            if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                expr = f"{instr.arg1} {instr.op} {instr.arg2}"
                if instr.arg1 not in killed_vars and instr.arg2 not in killed_vars:
                    gen_set.add(expr)
            # Kill expressions involving the defined variable
            dvar = instr.defined_var
            if dvar:
                gen_set = {e for e in gen_set
                           if dvar not in e.split()}
                killed_vars.add(dvar)
        return gen_set

    def kill(self, block: BasicBlock) -> set:
        kill_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                for expr in self.all_exprs:
                    if dvar in expr.split():
                        kill_set.add(expr)
        return kill_set

    def universe(self) -> set:
        return set(self.all_exprs)

    def boundary_value(self) -> set:
        return set()


class VeryBusyExpressions(DataFlowAnalysis):
    """Very Busy Expressions analysis."""

    def __init__(self, cfg: CFG):
        super().__init__(cfg, Direction.BACKWARD, MeetOp.INTERSECTION)
        self.all_exprs = set()
        for block in cfg.blocks.values():
            for instr in block.instructions:
                if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                    self.all_exprs.add(f"{instr.arg1} {instr.op} {instr.arg2}")

    def gen(self, block: BasicBlock) -> set:
        """Expressions computed in this block before any operand is redefined."""
        gen_set = set()
        killed_vars = set()
        for instr in block.instructions:
            # Check if this computes an expression with unkilled operands
            if instr.op in ('+', '-', '*', '/') and instr.arg1 and instr.arg2:
                expr = f"{instr.arg1} {instr.op} {instr.arg2}"
                if instr.arg1 not in killed_vars and instr.arg2 not in killed_vars:
                    gen_set.add(expr)
            # Track defined variables
            dvar = instr.defined_var
            if dvar:
                killed_vars.add(dvar)
        return gen_set

    def kill(self, block: BasicBlock) -> set:
        kill_set = set()
        for instr in block.instructions:
            dvar = instr.defined_var
            if dvar:
                for expr in self.all_exprs:
                    if dvar in expr.split():
                        kill_set.add(expr)
        return kill_set

    def universe(self) -> set:
        return set(self.all_exprs)

    def boundary_value(self) -> set:
        return set()


# ============================================================
# Example
# ============================================================

def build_example_cfg() -> CFG:
    """
    Build example CFG:

    B0: d0: x = 5
        d1: y = 1

    B1: d2: z = x + y
        if z < 10 goto B2

    B2: d3: x = x + 1
        d4: y = y * 2
        goto B1

    B3: d5: w = x + y
        return w
    """
    cfg = CFG()

    b0 = BasicBlock(0, instructions=[
        Instruction(0, "x", "copy", "5"),
        Instruction(1, "y", "copy", "1"),
    ])

    b1 = BasicBlock(1, instructions=[
        Instruction(2, "z", "+", "x", "y"),
        Instruction(3, "", "iffalse", "z < 10", result="B3"),
    ])

    b2 = BasicBlock(2, instructions=[
        Instruction(4, "x", "+", "x", "1"),
        Instruction(5, "y", "*", "y", "2"),
    ])

    b3 = BasicBlock(3, instructions=[
        Instruction(6, "w", "+", "x", "y"),
        Instruction(7, "", "return", "w"),
    ])

    cfg.add_block(b0)
    cfg.add_block(b1)
    cfg.add_block(b2)
    cfg.add_block(b3)

    cfg.add_edge(0, 1)
    cfg.add_edge(1, 2)  # z < 10
    cfg.add_edge(1, 3)  # z >= 10
    cfg.add_edge(2, 1)  # loop back

    return cfg


def demo_all_analyses():
    """Run all four data flow analyses on the example CFG."""
    cfg = build_example_cfg()

    print("=== Control Flow Graph ===")
    for bid in sorted(cfg.blocks.keys()):
        block = cfg.blocks[bid]
        print(f"\nB{bid}: (pred={block.predecessors}, succ={block.successors})")
        for instr in block.instructions:
            print(f"  d{instr.index}: {instr}")

    # 1. Reaching Definitions
    rd = ReachingDefinitions(cfg)
    rd.solve()
    rd.print_results("Reaching Definitions")

    # 2. Live Variables
    lv = LiveVariables(cfg, exit_live={"w"})
    lv.solve()
    lv.print_results("Live Variables")

    # 3. Available Expressions
    ae = AvailableExpressions(cfg)
    ae.solve()
    ae.print_results("Available Expressions")

    # 4. Very Busy Expressions
    vb = VeryBusyExpressions(cfg)
    vb.solve()
    vb.print_results("Very Busy Expressions")

    # Print Gen/Kill sets for each analysis
    print("\n=== Gen/Kill Sets ===")
    for bid in sorted(cfg.blocks.keys()):
        block = cfg.blocks[bid]
        print(f"\nB{bid}:")
        print(f"  RD:  Gen={sorted(rd.gen(block))}, Kill={sorted(rd.kill(block))}")
        print(f"  LV:  Use={sorted(lv.gen(block))}, Def={sorted(lv.kill(block))}")
        print(f"  AE:  Gen={sorted(ae.gen(block))}, Kill={sorted(ae.kill(block))}")
        print(f"  VBE: Gen={sorted(vb.gen(block))}, Kill={sorted(vb.kill(block))}")


if __name__ == "__main__":
    demo_all_analyses()
```

---

## 6. 분석 결과를 최적화에 적용

### 6.1 전역 상수 전파

**도달 정의**를 사용하여 전역 상수 전파를 수행할 수 있습니다:

```
For each use of variable v at point p:
    Let D = set of reaching definitions of v at p
    If |D| == 1 and that definition assigns a constant c:
        Replace v with c at p
```

**예시**:
```
B1: x = 5         // d1

B2: y = x + 1     // x의 도달 정의: {d1}
                   // d1은 상수 5를 할당
                   // → y = 5 + 1 → y = 6
```

### 6.2 전역 죽은 코드 제거

**활성 변수**를 사용하여 정의 이후 활성 상태가 아닌 변수의 정의를 제거합니다:

```
For each definition d: v = expr at point p:
    If v is NOT in LiveOut at point p:
        Remove d (it's dead code)
```

### 6.3 전역 공통 부분식 제거

**가용 표현식**을 사용하여 중복 계산을 제거합니다:

```
For each computation of expression e at point p:
    If e is in the Available set at p:
        Replace the computation with a reference to
        the previously computed value

Note: We need to insert a temporary at the original computation
      to store the value for later use.
```

**이전**:
```
B1: t1 = a + b      // a + b 계산
    ...

B2: ...              // a 또는 b의 재정의 없음

B3: t2 = a + b      // a + b는 여기서 가용 (B1에서)
```

**이후**:
```
B1: t1 = a + b      // a + b가 t1에 저장
    ...

B3: t2 = t1         // 이전에 계산된 값 재사용
```

### 6.4 코드 끌어올리기(Code Hoisting)

**매우 바쁜 표현식**을 사용하여 계산을 더 이른 지점으로 이동합니다:

```
If expression e is very busy at the entry of block B:
    Compute e at the entry of B (or at the end of B's predecessors)
    Store the result in a temporary
    Replace all subsequent computations of e with the temporary
```

**이전**:
```
B1:
    if (cond) goto B2 else goto B3

B2: t1 = a + b      // 이 경로에서 a + b 계산
    ...

B3: t2 = a + b      // 이 경로에서도 a + b 계산
    ...
```

`a + b`가 B1의 출구에서 매우 바쁘다면 (모든 경로에서 평가됨), 끌어올릴 수 있습니다:

**이후**:
```
B1: t0 = a + b      // 끌어올림
    if (cond) goto B2 else goto B3

B2: t1 = t0         // 재사용
    ...

B3: t2 = t0         // 재사용
    ...
```

---

## 7. 고급 주제

### 7.1 반복 분석 순서

블록이 처리되는 순서는 반복 횟수에 영향을 미칩니다:

| 방향 | 최선의 순서 | 이유 |
|-----------|--------------|-----|
| 순방향 | 역후위 순서 | 후방 엣지를 제외한 후속 블록보다 선행 블록이 먼저 처리됨 |
| 역방향 | 후위 순서 | 후방 엣지를 제외한 선행 블록보다 후속 블록이 먼저 처리됨 |

최적 순서를 사용하면 대부분의 분석이 2--3번의 패스로 수렴합니다 (한 패스가 비순환 부분을 처리하고, 추가 패스가 루프를 처리합니다).

### 7.2 SSA 기반 분석

SSA 형식에서는 많은 데이터 흐름 분석이 단순해집니다:

- **도달 정의**: 자명함 -- 각 변수는 정확히 하나의 정의를 가집니다
- **정의-사용 체인(Def-use chains)**: SSA 구조에서 즉시 사용 가능
- **상수 전파**: 희소 조건부 상수 전파(Sparse Conditional Constant Propagation, SCCP)가 SSA에서 직접 동작

### 7.3 프로시저 간 분석

함수 경계를 넘어 분석을 확장하려면 다음이 필요합니다:

1. **호출 그래프(call graph) 구성**: 어떤 함수가 어떤 함수를 호출하는가?
2. **문맥 감수성(Context sensitivity)**: 서로 다른 호출 문맥 구별
3. **요약 함수(Summary functions)**: 매번 함수 본문을 분석하지 않고 함수 호출의 효과를 계산

프로시저 간 분석이 가능하게 하는 것:
- 함수 호출 간 전역 상수 전파
- 별칭 분석(Alias analysis) (어떤 포인터가 같은 메모리를 가리킬 수 있는가?)
- 이탈 분석(Escape analysis) (객체가 생성 함수를 벗어나는가?)

### 7.4 확장과 수축(Widening and Narrowing)

무한 격자(예: 수치 범위)에 대한 분석에서는 반복이 수렴하지 않을 수 있습니다. **확장(widening)**은 더 높은 격자 원소로 점프하여 수렴을 가속합니다:

$$X_{n+1} = X_n \nabla f(X_n)$$

여기서 $\nabla$는 확장 연산자입니다. 수렴 후 **수축(narrowing)**이 결과를 정제합니다:

$$X_{n+1} = X_n \Delta f(X_n)$$

이는 Astrée 정적 분석기 뒤의 추상 해석(abstract interpretation) 프레임워크에서 사용됩니다.

---

## 8. 최적화 조합: 완전한 패스

다음은 컴파일러가 지역 최적화와 전역 최적화를 어떻게 연결할 수 있는지 보여줍니다:

```python
"""
Example optimization pipeline combining local and global passes.
"""


def optimization_pipeline(cfg):
    """
    Run a sequence of optimization passes on a CFG.

    Pipeline:
    1. Global constant propagation (using reaching definitions)
    2. Local constant folding (within each block)
    3. Global CSE (using available expressions)
    4. Local copy propagation (within each block)
    5. Global DCE (using live variables)
    6. Local algebraic simplification
    7. Repeat if anything changed
    """
    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1
        print(f"\n--- Optimization Pass {iteration} ---")

        # 1. Global: Reaching definitions → constant propagation
        rd = ReachingDefinitions(cfg)
        rd.solve()
        if apply_global_constant_propagation(cfg, rd):
            changed = True
            print("  Applied global constant propagation")

        # 2. Local: Constant folding in each block
        for block in cfg.blocks.values():
            if apply_local_constant_folding(block):
                changed = True
                print(f"  Applied constant folding in B{block.id}")

        # 3. Global: Available expressions → CSE
        ae = AvailableExpressions(cfg)
        ae.solve()
        if apply_global_cse(cfg, ae):
            changed = True
            print("  Applied global CSE")

        # 4. Local: Copy propagation in each block
        for block in cfg.blocks.values():
            if apply_local_copy_propagation(block):
                changed = True
                print(f"  Applied copy propagation in B{block.id}")

        # 5. Global: Live variables → DCE
        lv = LiveVariables(cfg)
        lv.solve()
        if apply_global_dce(cfg, lv):
            changed = True
            print("  Applied global DCE")

        # 6. Local: Algebraic simplification
        for block in cfg.blocks.values():
            if apply_local_algebraic_simplification(block):
                changed = True
                print(f"  Applied algebraic simplification in B{block.id}")

    print(f"\nOptimization converged after {iteration} iterations")
```

참고: `apply_global_constant_propagation` 등의 함수는 섹션 6에서 설명된 변환을 구현하여 분석 결과를 사용해 어디에 변경을 적용할지 결정합니다. 위 코드는 최적화 파이프라인의 전반적인 구조를 보여줍니다.

---

## 9. 요약

이 레슨에서는 컴파일러 최적화의 이론과 실제를 다루었습니다:

1. **최적화 원칙**: 모든 최적화는 안전해야 하고 (의미론 보존), 수익성이 있어야 하며 (코드 개선), 적용 기회를 감지해야 합니다.

2. **지역 최적화**는 단일 기본 블록 내에서 동작하며 다음을 포함합니다:
   - **상수 폴딩(Constant folding)**: 컴파일 시간에 상수 표현식 평가
   - **상수 전파(Constant propagation)**: 변수를 상수 값으로 교체
   - **대수적 단순화(Algebraic simplification)**: 수학적 항등식 적용
   - **강도 감소(Strength reduction)**: 비용이 많이 드는 연산을 더 저렴한 것으로 교체
   - **CSE**: 중복 계산 제거
   - **복사 전파(Copy propagation)**: 복사를 원래 값으로 교체
   - **죽은 코드 제거(Dead code elimination)**: 미사용 계산 제거

3. **전역 데이터 흐름 분석**은 전체 CFG를 통한 정보 흐름을 추론합니다:
   - **도달 정의** (순방향, 합집합): 어떤 정의가 한 지점에 도달하는가?
   - **가용 표현식** (순방향, 교집합): 어떤 표현식이 모든 경로에서 계산되는가?
   - **활성 변수** (역방향, 합집합): 어떤 변수가 나중에 필요한가?
   - **매우 바쁜 표현식** (역방향, 교집합): 어떤 표현식이 모든 경로에서 평가되는가?

4. **데이터 흐름 프레임워크**는 격자 이론, 단조 전달 함수, 고정점 반복을 사용하여 모든 분석을 통합합니다. **워크리스트 알고리즘**은 효율적인 구현을 제공합니다.

5. 분석 결과는 전역 최적화를 가능하게 합니다: 전역 상수 전파, 전역 CSE, 전역 DCE, 코드 끌어올리기.

이러한 기술들은 컴파일러의 "미들 엔드" 핵심을 형성하여, 백엔드가 기계 명령어를 생성하기 전에 IR을 변환하여 더 빠르고 작은 코드를 생성합니다.

---

## 연습 문제

### 연습 1: 지역 최적화

다음 기본 블록에 모든 지역 최적화를 (순서: 상수 전파, 상수 폴딩, 대수적 단순화, 복사 전파, 죽은 코드 제거) 적용하세요. 각 패스 이후의 결과를 보여주세요.

```
t1 = 4
t2 = 8
t3 = t1 * t2
t4 = t3 + 0
t5 = t4
t6 = t5 * 1
t7 = t6 / 2
t8 = t7 + t7
unused = t1 + t2
result = t8
```

`result`가 live out이라고 가정합니다.

### 연습 2: 도달 정의

다음 CFG의 각 블록에 대해 도달 정의를 계산하세요. Gen, Kill, In, Out 집합을 보여주세요.

```
B0: d0: a = 1
    d1: b = 2

B1: d2: c = a + b
    if c > 10 goto B3

B2: d3: a = a + 1
    d4: b = b - 1
    goto B1

B3: d5: d = a * b
    return d
```

### 연습 3: 활성 변수

연습 2의 CFG에 대해 각 블록의 진입점과 출구점에서의 활성 변수를 계산하세요. B3의 출구에서 `d`가 활성이라고 가정합니다.

### 연습 4: 가용 표현식

다음 CFG에 대해 각 블록 진입점에서의 가용 표현식을 계산하세요:

```
B0: t1 = a + b
    t2 = c + d

B1: t3 = a + b      (a+b가 여기서 가용한가?)
    if (t3 > 0) goto B2 else goto B3

B2: a = a + 1        (a+b를 제거)
    t4 = c + d
    goto B4

B3: t5 = a + b
    t6 = c + d
    goto B4

B4: t7 = a + b       (a+b가 여기서 가용한가?)
    t8 = c + d       (c+d가 여기서 가용한가?)
```

### 연습 5: 워크리스트 알고리즘 추적

다음 CFG에 대해 활성 변수 분석의 워크리스트 알고리즘 실행을 추적하세요. 각 단계에서 워크리스트 내용과 In/Out 집합을 보여주세요.

```
B0: x = read()
    y = read()

B1: if (x > 0) goto B2 else goto B3

B2: z = x + y
    x = x - 1
    goto B1

B3: print z
```

### 연습 6: 구현 도전

데이터 흐름 분석 프레임워크를 확장하여 다음을 구현하세요:

1. **상수 전파 분석(Constant propagation analysis)**: 각 변수의 격자 원소가 {$\top$, 상수 $c$, $\bot$} 중 하나인 순방향 분석. 두 개의 다른 상수의 만남은 $\bot$입니다.

2. **복사 전파 분석(Copy propagation analysis)**: 어떤 복사 `x = y`가 각 프로그램 지점에서 아직 유효한지 (제거되지 않았는지) 결정합니다.

두 분석을 연습 2의 CFG에서 테스트하고 어떻게 추가 최적화를 가능하게 하는지 보여주세요.

---

[이전: 코드 생성](./11_Code_Generation.md) | [다음: 루프 최적화](./13_Loop_Optimization.md) | [개요](./00_Overview.md)
