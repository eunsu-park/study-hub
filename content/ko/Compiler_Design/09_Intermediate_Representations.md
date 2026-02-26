# 레슨 9: 중간 표현(Intermediate Representations)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 현대 컴파일러 설계에서 중간 표현(intermediate representations, IRs)이 왜 필수적인지 **설명**할 수 있다
2. 고수준, 중간 수준, 저수준 IR의 차이와 그 트레이드오프를 **구별**할 수 있다
3. 추상 구문 트리(abstract syntax tree)로부터 3-주소 코드(three-address code, TAC)를 **생성**할 수 있다
4. 제어 흐름 그래프(control flow graphs, CFGs)를 **구성**하고 기본 블록(basic blocks)을 식별할 수 있다
5. 코드를 정적 단일 대입(Static Single Assignment, SSA) 형식으로 **변환**하고 파이 함수(phi functions)를 설명할 수 있다
6. Python으로 TAC 생성기와 CFG 빌더를 **구현**할 수 있다
7. DAG 표현과 최적화에서의 역할을 **설명**할 수 있다

---

## 1. 중간 표현이 중요한 이유

### 1.1 컴파일러의 다리

컴파일러는 고수준 언어로 작성된 소스 코드를 특정 대상 아키텍처의 기계어 코드로 변환합니다. 중간 표현 없이는, (소스 언어, 대상 기계) 쌍마다 별도의 컴파일러 프론트 엔드가 필요합니다. $m$개의 소스 언어와 $n$개의 대상 기계가 있다면, 단순한 접근 방식은 $m \times n$개의 번역기가 필요합니다.

IR은 프론트 엔드와 백 엔드를 분리합니다:

```
          Front Ends                Back Ends
Source 1 ──┐                    ┌── Target A
Source 2 ──┼──▶  IR  ──▶──┼── Target B
Source 3 ──┘                    └── Target C
```

IR을 사용하면 $m$개의 프론트 엔드와 $n$개의 백 엔드만 필요하여, 총 작업이 $m + n$으로 줄어듭니다.

### 1.2 최적화 활성화

IR은 최적화(optimization)가 수행되는 기반 역할을 합니다. 잘 설계된 IR은 변환을 표현하고, 검증하며, 구성하기 쉽게 만듭니다. 서로 다른 IR은 서로 다른 최적화 기회를 노출합니다:

- **고수준 IR**: 루프 변환(loop transformations), 인라이닝(inlining), 가상 함수 해제(devirtualization)
- **중간 수준 IR**: 상수 전파(constant propagation), 죽은 코드 제거(dead code elimination), 레지스터 승격(register promotion)
- **저수준 IR**: 명령어 선택(instruction selection), 레지스터 할당(register allocation), 스케줄링(scheduling)

### 1.3 이식성과 재사용성

실제 사례들이 좋은 IR의 강력함을 보여줍니다:

| 컴파일러 | IR | 목적 |
|----------|----|------|
| GCC | GIMPLE, RTL | 다중 언어, 다중 대상 |
| LLVM | LLVM IR | 언어 독립적 최적화기 및 코드 생성기 |
| JVM | Java 바이트코드 | 플랫폼 독립적 실행 |
| .NET | CIL (Common Intermediate Language) | 다중 언어 런타임 |
| WebAssembly | Wasm 바이너리 형식 | 이식 가능한 웹 실행 |

### 1.4 IR의 설계 목표

좋은 IR은 다음과 같아야 합니다:

1. 소스 언어 AST로부터 **생성하기 쉬움**
2. 대상 기계어 코드로 **번역하기 쉬움**
3. **최적화하기 적합함** -- 기회를 명확하게 노출
4. **간결함** -- 대형 프로그램에 대해 메모리 낭비 없음
5. **명확하게 정의됨** -- 모든 구조에 대한 명확한 의미론

---

## 2. 중간 표현의 스펙트럼

### 2.1 고수준 IR

고수준 IR은 소스 언어의 구조를 많이 유지합니다: 루프, 조건문, 인덱스가 있는 배열 접근, 구조화된 타입. 추상 구문 트리(AST)나 그것의 주석된 버전에 가깝습니다.

**예시**: 고수준 IR은 `for` 루프를 단일 노드로 표현할 수 있습니다:

```
ForLoop(
    init = Assign(i, 0),
    cond = LessThan(i, n),
    step = Assign(i, Add(i, 1)),
    body = [
        Assign(ArrayAccess(a, i), Multiply(i, i))
    ]
)
```

**장점**:
- 고수준 최적화를 위한 소스 수준 의미론 보존 (루프 교환, 루프 융합)
- 타입 검사 및 의미 분석에 자연스러움

**단점**:
- 저수준 최적화에는 너무 추상적 (레지스터 할당, 명령어 스케줄링)

### 2.2 중간 수준 IR

중간 수준 IR (때로 "3-주소 코드" 수준이라 불림)은 고수준 제어 구조를 제거하고 레이블과 점프로 대체합니다. 변수는 명시적이지만 기계 세부 사항(레지스터, 주소 지정 모드)은 추상화됩니다.

**예시**: 중간 수준 IR에서의 동일한 루프:

```
    i = 0
L1: if i >= n goto L2
    t1 = i * i
    a[i] = t1
    i = i + 1
    goto L1
L2:
```

**장점**:
- 대부분의 고전적 최적화에 이상적 (CSE, 상수 전파, DCE)
- 직관적 분석에 충분히 단순

**단점**:
- 소스 수준 구조를 잃음 (루프 변환이 더 어려움)

### 2.3 저수준 IR

저수준 IR은 기계어 코드에 가깝지만 여전히 다소 추상화되어 있습니다. 가상 레지스터(무제한 공급)를 사용하고 주소 지정 모드, 조건 코드, 특정 명령어 형식과 같은 기계별 연산을 노출합니다.

**예시**: x86 유사 대상에 대한 저수준 IR에서의 동일한 루프:

```
    mov v1, 0          ; i = 0
L1: cmp v1, v2         ; compare i with n
    jge L2             ; if i >= n goto L2
    mov v3, v1
    imul v3, v1        ; t1 = i * i
    mov [v4 + v1*4], v3 ; a[i] = t1
    add v1, 1          ; i = i + 1
    jmp L1
L2:
```

**장점**:
- 최종 출력에 가까움 -- 명령어 선택 및 레지스터 할당 용이
- 기계별 최적화 표현 가능

**단점**:
- 기계 의존적이어서 이식성 감소

### 2.4 다중 수준 IR 전략

현대 컴파일러는 종종 여러 IR 수준을 사용합니다. 예를 들어:

```
Source ──▶ AST ──▶ High IR ──▶ Medium IR ──▶ Low IR ──▶ Machine Code
             (type check)  (loop opts)  (scalar opts)  (reg alloc)
```

LLVM은 단일 IR (LLVM IR)을 사용하지만 점진적인 저수준화 패스를 통해 여러 추상화 수준에서 작동합니다. GCC는 GIMPLE (중간 수준)과 RTL (저수준)을 사용합니다.

---

## 3. 3-주소 코드(Three-Address Code, TAC)

### 3.1 정의

3-주소 코드(three-address code)는 가장 널리 사용되는 중간 표현 중 하나입니다. 각 명령어는 최대 세 개의 피연산자(주소)를 가집니다: 일반적으로 하나의 결과와 하나 또는 두 개의 인자.

일반적인 형태는:

$$x = y \;\text{op}\; z$$

여기서 $x$, $y$, $z$는 이름(변수, 임시 변수, 또는 상수)이고 $\text{op}$는 연산자입니다.

### 3.2 명령어 유형

TAC는 여러 명령어 형태를 포함합니다:

| 범주 | 형태 | 예시 |
|------|------|------|
| 할당 (이진) | `x = y op z` | `t1 = a + b` |
| 할당 (단항) | `x = op y` | `t2 = -t1` |
| 복사 | `x = y` | `t3 = t1` |
| 무조건 점프 | `goto L` | `goto L2` |
| 조건부 점프 | `if x relop y goto L` | `if t1 < n goto L1` |
| 인덱스 할당 (저장) | `x[i] = y` | `a[t2] = t3` |
| 인덱스 할당 (로드) | `x = y[i]` | `t4 = a[t2]` |
| 주소/포인터 | `x = &y`, `x = *y`, `*x = y` | `t5 = &a` |
| 프로시저 호출 | `param x`, `call p, n`, `x = call p, n` | `param t1` |
| 반환 | `return x` | `return t1` |

### 3.3 임시 변수(Temporaries)

TAC는 중간 결과를 담기 위해 임시 변수($t_1, t_2, \ldots$)를 도입합니다. 이러한 분해는 모든 복잡한 표현식이 간단한 단계로 분해되도록 보장합니다.

**소스 표현식**: `a + b * c - d / e`

**TAC**:
```
t1 = b * c
t2 = a + t1
t3 = d / e
t4 = t2 - t3
```

### 3.4 TAC를 위한 자료구조

TAC 명령어를 저장하는 세 가지 일반적인 방법이 있습니다:

#### 쿼드러플(Quadruples)

쿼드러플은 네 개의 필드를 가집니다: `(op, arg1, arg2, result)`.

| 인덱스 | op | arg1 | arg2 | result |
|-------|----|------|------|--------|
| 0 | `*` | `b` | `c` | `t1` |
| 1 | `+` | `a` | `t1` | `t2` |
| 2 | `/` | `d` | `e` | `t3` |
| 3 | `-` | `t2` | `t3` | `t4` |

**장점**: 단순하고, 직접적이며, 명령어 재정렬이 쉬움.

**단점**: 모든 임시 변수에 대한 명시적 이름 필요.

#### 트리플(Triples)

트리플은 세 개의 필드를 가집니다: `(op, arg1, arg2)`. 결과는 명명된 임시 변수가 아닌 트리플의 인덱스로 식별됩니다.

| 인덱스 | op | arg1 | arg2 |
|-------|----|------|------|
| (0) | `*` | `b` | `c` |
| (1) | `+` | `a` | (0) |
| (2) | `/` | `d` | `e` |
| (3) | `-` | (1) | (2) |

**장점**: 명시적 임시 변수 불필요 -- 더 간결함.

**단점**: 참조가 인덱스를 사용하기 때문에 명령어 재정렬이 어려움.

#### 간접 트리플(Indirect Triples)

간접 트리플은 명령어 위치를 트리플 인덱스에 매핑하는 간접 배열을 추가합니다. 트리플 자체를 수정하지 않고 간접 배열을 순열하여 재정렬을 허용합니다.

```
Instruction list: [0, 1, 2, 3]   <-- can be reordered
Triple table:     same as above   <-- unchanged
```

### 3.5 Python 구현: TAC 생성기

간단한 표현식 AST를 3-주소 코드로 변환하는 TAC 생성기를 만들어 보겠습니다.

```python
"""Three-Address Code (TAC) Generator from AST."""

from dataclasses import dataclass, field
from typing import Union


# ---------- AST Nodes ----------

@dataclass
class Num:
    """Numeric literal."""
    value: int

@dataclass
class Var:
    """Variable reference."""
    name: str

@dataclass
class BinOp:
    """Binary operation."""
    op: str          # '+', '-', '*', '/'
    left: 'Expr'
    right: 'Expr'

@dataclass
class UnaryOp:
    """Unary operation."""
    op: str          # '-', 'not'
    operand: 'Expr'

@dataclass
class Assign:
    """Assignment statement."""
    target: str
    value: 'Expr'

@dataclass
class IfElse:
    """If-else statement."""
    condition: 'Expr'
    then_body: list
    else_body: list

@dataclass
class While:
    """While loop."""
    condition: 'Expr'
    body: list

@dataclass
class RelOp:
    """Relational comparison."""
    op: str          # '<', '>', '<=', '>=', '==', '!='
    left: 'Expr'
    right: 'Expr'

Expr = Union[Num, Var, BinOp, UnaryOp, RelOp]
Stmt = Union[Assign, IfElse, While]


# ---------- TAC Instruction ----------

@dataclass
class TACInstruction:
    """A single TAC instruction."""
    op: str
    arg1: str = ""
    arg2: str = ""
    result: str = ""

    def __str__(self):
        if self.op == "label":
            return f"{self.result}:"
        elif self.op == "goto":
            return f"    goto {self.result}"
        elif self.op == "iffalse":
            return f"    iffalse {self.arg1} goto {self.result}"
        elif self.op == "iftrue":
            return f"    iftrue {self.arg1} goto {self.result}"
        elif self.op == "copy":
            return f"    {self.result} = {self.arg1}"
        elif self.op == "param":
            return f"    param {self.arg1}"
        elif self.op == "call":
            return f"    {self.result} = call {self.arg1}, {self.arg2}"
        elif self.op == "return":
            return f"    return {self.arg1}"
        elif self.arg2:
            return f"    {self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.arg1:
            return f"    {self.result} = {self.op} {self.arg1}"
        else:
            return f"    {self.op}"


# ---------- TAC Generator ----------

class TACGenerator:
    """Generates three-address code from an AST."""

    def __init__(self):
        self.instructions: list[TACInstruction] = []
        self._temp_counter = 0
        self._label_counter = 0

    def new_temp(self) -> str:
        """Generate a fresh temporary variable name."""
        self._temp_counter += 1
        return f"t{self._temp_counter}"

    def new_label(self) -> str:
        """Generate a fresh label name."""
        self._label_counter += 1
        return f"L{self._label_counter}"

    def emit(self, op: str, arg1: str = "", arg2: str = "",
             result: str = "") -> TACInstruction:
        """Emit a single TAC instruction."""
        instr = TACInstruction(op, arg1, arg2, result)
        self.instructions.append(instr)
        return instr

    def generate(self, node) -> str:
        """Generate TAC for a node. Returns the name holding the result."""
        if isinstance(node, Num):
            return str(node.value)

        elif isinstance(node, Var):
            return node.name

        elif isinstance(node, BinOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            temp = self.new_temp()
            self.emit(node.op, left, right, temp)
            return temp

        elif isinstance(node, UnaryOp):
            operand = self.generate(node.operand)
            temp = self.new_temp()
            self.emit(node.op, operand, result=temp)
            return temp

        elif isinstance(node, RelOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            temp = self.new_temp()
            self.emit(node.op, left, right, temp)
            return temp

        elif isinstance(node, Assign):
            value = self.generate(node.value)
            self.emit("copy", value, result=node.target)
            return node.target

        elif isinstance(node, IfElse):
            return self._generate_if_else(node)

        elif isinstance(node, While):
            return self._generate_while(node)

        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def _generate_if_else(self, node: IfElse) -> str:
        """Generate TAC for an if-else statement."""
        cond = self.generate(node.condition)
        else_label = self.new_label()
        end_label = self.new_label()

        self.emit("iffalse", cond, result=else_label)

        # Then body
        for stmt in node.then_body:
            self.generate(stmt)

        self.emit("goto", result=end_label)

        # Else label
        self.emit("label", result=else_label)

        # Else body
        for stmt in node.else_body:
            self.generate(stmt)

        # End label
        self.emit("label", result=end_label)
        return ""

    def _generate_while(self, node: While) -> str:
        """Generate TAC for a while loop."""
        begin_label = self.new_label()
        end_label = self.new_label()

        # Loop start
        self.emit("label", result=begin_label)

        # Condition
        cond = self.generate(node.condition)
        self.emit("iffalse", cond, result=end_label)

        # Body
        for stmt in node.body:
            self.generate(stmt)

        self.emit("goto", result=begin_label)

        # Loop end
        self.emit("label", result=end_label)
        return ""

    def print_tac(self):
        """Pretty-print all generated TAC instructions."""
        for instr in self.instructions:
            print(instr)


# ---------- Example Usage ----------

def demo_expression():
    """Generate TAC for: result = (a + b) * (c - d)"""
    print("=== Expression: result = (a + b) * (c - d) ===")
    ast = Assign("result",
        BinOp("*",
            BinOp("+", Var("a"), Var("b")),
            BinOp("-", Var("c"), Var("d"))
        )
    )
    gen = TACGenerator()
    gen.generate(ast)
    gen.print_tac()
    print()


def demo_if_else():
    """Generate TAC for: if (x < y) max = y; else max = x;"""
    print("=== If-Else: if (x < y) max = y; else max = x; ===")
    ast = IfElse(
        condition=RelOp("<", Var("x"), Var("y")),
        then_body=[Assign("max", Var("y"))],
        else_body=[Assign("max", Var("x"))]
    )
    gen = TACGenerator()
    gen.generate(ast)
    gen.print_tac()
    print()


def demo_while():
    """Generate TAC for: while (i < n) { s = s + a[i]; i = i + 1; }"""
    print("=== While: while (i < n) { s = s + i; i = i + 1; } ===")
    ast = While(
        condition=RelOp("<", Var("i"), Var("n")),
        body=[
            Assign("s", BinOp("+", Var("s"), Var("i"))),
            Assign("i", BinOp("+", Var("i"), Num(1)))
        ]
    )
    gen = TACGenerator()
    gen.generate(ast)
    gen.print_tac()
    print()


if __name__ == "__main__":
    demo_expression()
    demo_if_else()
    demo_while()
```

**`demo_expression()`의 예상 출력**:
```
=== Expression: result = (a + b) * (c - d) ===
    t1 = a + b
    t2 = c - d
    t3 = t1 * t2
    result = t3
```

**`demo_while()`의 예상 출력**:
```
=== While: while (i < n) { s = s + i; i = i + 1; } ===
L1:
    t1 = i < n
    iffalse t1 goto L2
    t2 = s + i
    s = t2
    t3 = i + 1
    i = t3
    goto L1
L2:
```

---

## 4. 제어 흐름 그래프(Control Flow Graphs, CFGs)

### 4.1 동기

3-주소 코드를 가지면, 명령어의 선형 순서는 프로그램의 실행 흐름 구조를 즉시 드러내지 않습니다. 제어 흐름 그래프(control flow graph, CFG)는 명령어를 **기본 블록(basic blocks)**으로 조직하고 가능한 제어 이동을 나타내는 **간선(edges)**으로 연결하여 이 구조를 명시적으로 만듭니다.

CFG는 방향 그래프 $G = (V, E)$입니다:
- 각 노드 $v \in V$는 **기본 블록(basic block)**
- 각 간선 $(u, v) \in E$는 블록 $u$에서 블록 $v$로의 가능한 제어 흐름을 나타냄
- 구별되는 **진입(entry)** 노드와 하나 이상의 **출구(exit)** 노드가 있음

### 4.2 기본 블록(Basic Blocks)

**기본 블록(basic block)**은 다음과 같은 최대 연속 명령어 시퀀스입니다:

1. **제어는 첫 번째 명령어로만 진입** (리더)
2. **제어는 마지막 명령어에서만 이탈** (중간에서 점프 없음)
3. **블록이 진입되면 모든 명령어가 순차적으로 실행**

이는 기본 블록 내에서, 첫 번째 명령어가 실행되면 모든 명령어가 실행된다는 것을 의미하며, 블록이 분석의 원자 단위가 됩니다.

### 4.3 리더 식별(Identifying Leaders)

TAC를 기본 블록으로 분할하기 위해, 새로운 블록을 시작하는 명령어인 **리더(leaders)**를 식별합니다:

1. 프로그램의 **첫 번째 명령어**는 리더
2. 점프의 **대상인 명령어** (조건부 또는 무조건)는 리더
3. 점프 **바로 다음에 오는 명령어**는 리더

**알고리즘: 기본 블록 식별**

```
입력:  TAC 명령어 목록
출력: 기본 블록 목록

1. 리더 집합 결정:
   a. 첫 번째 명령어는 리더
   b. goto/branch의 대상은 리더
   c. goto/branch 다음 명령어는 리더

2. 각 리더에 대해, 기본 블록은 리더와
   다음 리더(포함하지 않음) 또는 프로그램 끝까지의
   모든 명령어로 구성됨.
```

### 4.4 CFG 구성(Building the CFG)

기본 블록을 식별한 후, 간선으로 연결합니다:

1. 블록 $B_i$가 조건부 분기 `if ... goto L`로 끝나는 경우:
   - $B_i$에서 레이블 $L$로 시작하는 블록으로 간선 추가 (참 분기)
   - $B_i$에서 $B_{i+1}$로 **폴-스루(fall-through)** 간선 추가 (거짓 분기)

2. 블록 $B_i$가 무조건 `goto L`로 끝나는 경우:
   - $B_i$에서 레이블 $L$로 시작하는 블록으로 간선 추가
   - 폴-스루 간선 없음

3. 블록 $B_i$가 분기나 goto 없이 끝나는 경우:
   - $B_i$에서 $B_{i+1}$로 폴-스루 간선 추가

### 4.5 예시: CFG 구성

while 루프 예시의 TAC를 고려해 보겠습니다:

```
(0)  L1:                  <-- Block B1 leader (label target)
(1)  t1 = i < n
(2)  iffalse t1 goto L2
(3)  t2 = s + i           <-- Block B2 leader (follows branch)
(4)  s = t2
(5)  t3 = i + 1
(6)  i = t3
(7)  goto L1
(8)  L2:                  <-- Block B3 leader (label target)
```

**리더**: 명령어 0 (첫 번째 + 레이블 대상), 명령어 3 (분기 다음), 명령어 8 (레이블 대상)

**기본 블록**:
- $B_1$: 명령어 0-2 (조건 테스트)
- $B_2$: 명령어 3-7 (루프 본문)
- $B_3$: 명령어 8 (루프 종료)

**CFG**:
```
    ┌─────────┐
    │   B1    │ ◀────────┐
    │ L1:     │          │
    │ t1=i<n  │          │
    │ if !t1  │          │
    │ goto L2 │          │
    └────┬────┘          │
    true │  \false       │
         │   \           │
         │  ┌──▼──────┐  │
         │  │   B2    │  │
         │  │ t2=s+i  │  │
         │  │ s=t2    │  │
         │  │ t3=i+1  │  │
         │  │ i=t3    │  │
         │  │ goto L1 │──┘
         │  └─────────┘
         │
    ┌────▼────┐
    │   B3    │
    │ L2:     │
    └─────────┘
```

참고: 간선 레이블은 `iffalse`의 의미론에 따라 다릅니다. 조건이 거짓이면 L2 (B3)로 이동합니다. 조건이 참이면 (즉, `iffalse`가 점프하지 않으면) B2로 폴스루합니다.

### 4.6 Python 구현: CFG 빌더

```python
"""Control Flow Graph (CFG) Builder from TAC instructions."""

from dataclasses import dataclass, field


@dataclass
class TACInstr:
    """A TAC instruction for CFG construction."""
    index: int
    op: str
    arg1: str = ""
    arg2: str = ""
    result: str = ""

    @property
    def is_label(self) -> bool:
        return self.op == "label"

    @property
    def is_jump(self) -> bool:
        return self.op in ("goto", "iffalse", "iftrue")

    @property
    def is_unconditional_jump(self) -> bool:
        return self.op == "goto"

    @property
    def is_conditional_jump(self) -> bool:
        return self.op in ("iffalse", "iftrue")

    @property
    def jump_target(self) -> str:
        """Return the label this instruction jumps to, if any."""
        if self.is_jump:
            return self.result
        return ""

    def __str__(self):
        if self.op == "label":
            return f"{self.result}:"
        elif self.op == "goto":
            return f"  goto {self.result}"
        elif self.op in ("iffalse", "iftrue"):
            return f"  {self.op} {self.arg1} goto {self.result}"
        elif self.op == "copy":
            return f"  {self.result} = {self.arg1}"
        elif self.arg2:
            return f"  {self.result} = {self.arg1} {self.op} {self.arg2}"
        else:
            return f"  {self.result} = {self.op} {self.arg1}"


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    id: int
    instructions: list = field(default_factory=list)
    successors: list = field(default_factory=list)   # Block IDs
    predecessors: list = field(default_factory=list)  # Block IDs

    @property
    def leader(self):
        """The first instruction in the block."""
        return self.instructions[0] if self.instructions else None

    @property
    def terminator(self):
        """The last instruction in the block."""
        return self.instructions[-1] if self.instructions else None

    @property
    def label(self) -> str:
        """Return the label of this block, if it starts with one."""
        if self.instructions and self.instructions[0].is_label:
            return self.instructions[0].result
        return ""

    def __str__(self):
        lines = [f"Block B{self.id}:"]
        lines.append(f"  Predecessors: {[f'B{p}' for p in self.predecessors]}")
        lines.append(f"  Successors:   {[f'B{s}' for s in self.successors]}")
        lines.append("  Instructions:")
        for instr in self.instructions:
            lines.append(f"    {instr}")
        return "\n".join(lines)


class CFG:
    """Control Flow Graph."""

    def __init__(self):
        self.blocks: dict[int, BasicBlock] = {}
        self.entry_id: int = -1
        self.exit_ids: list[int] = []

    def add_block(self, block: BasicBlock):
        """Add a basic block to the CFG."""
        self.blocks[block.id] = block

    def add_edge(self, from_id: int, to_id: int):
        """Add a directed edge between two blocks."""
        if to_id not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(to_id)
        if from_id not in self.blocks[to_id].predecessors:
            self.blocks[to_id].predecessors.append(from_id)

    def __str__(self):
        lines = [f"=== Control Flow Graph ({len(self.blocks)} blocks) ==="]
        lines.append(f"Entry: B{self.entry_id}")
        lines.append(f"Exit:  {[f'B{e}' for e in self.exit_ids]}")
        lines.append("")
        for block_id in sorted(self.blocks.keys()):
            lines.append(str(self.blocks[block_id]))
            lines.append("")
        return "\n".join(lines)


def build_cfg(instructions: list[TACInstr]) -> CFG:
    """
    Build a Control Flow Graph from a list of TAC instructions.

    Algorithm:
    1. Identify leaders (first instruction, jump targets, post-jump instructions)
    2. Partition instructions into basic blocks
    3. Connect blocks with edges based on control flow
    """
    if not instructions:
        return CFG()

    # Step 1: Identify leader indices
    leaders = set()
    leaders.add(0)  # First instruction is always a leader

    # Map labels to instruction indices
    label_to_index = {}
    for instr in instructions:
        if instr.is_label:
            label_to_index[instr.result] = instr.index

    for instr in instructions:
        if instr.is_jump:
            # Target of jump is a leader
            target_label = instr.jump_target
            if target_label in label_to_index:
                leaders.add(label_to_index[target_label])

            # Instruction after jump is a leader (if it exists)
            next_idx = instr.index + 1
            if next_idx < len(instructions):
                leaders.add(next_idx)

    sorted_leaders = sorted(leaders)

    # Step 2: Partition into basic blocks
    cfg = CFG()
    block_id = 0
    leader_to_block_id = {}

    for i, leader_idx in enumerate(sorted_leaders):
        # Determine the end of this block
        if i + 1 < len(sorted_leaders):
            end_idx = sorted_leaders[i + 1]
        else:
            end_idx = len(instructions)

        block = BasicBlock(id=block_id)
        block.instructions = instructions[leader_idx:end_idx]
        cfg.add_block(block)
        leader_to_block_id[leader_idx] = block_id
        block_id += 1

    # Build label -> block_id mapping
    label_to_block_id = {}
    for bid, block in cfg.blocks.items():
        label = block.label
        if label:
            label_to_block_id[label] = bid

    # Step 3: Add edges
    block_ids = sorted(cfg.blocks.keys())
    for i, bid in enumerate(block_ids):
        block = cfg.blocks[bid]
        term = block.terminator

        if term is None:
            continue

        if term.is_unconditional_jump:
            # Edge to jump target only
            target_label = term.jump_target
            if target_label in label_to_block_id:
                cfg.add_edge(bid, label_to_block_id[target_label])

        elif term.is_conditional_jump:
            # Edge to jump target
            target_label = term.jump_target
            if target_label in label_to_block_id:
                cfg.add_edge(bid, label_to_block_id[target_label])
            # Fall-through edge to next block
            if i + 1 < len(block_ids):
                cfg.add_edge(bid, block_ids[i + 1])

        else:
            # No jump: fall through to next block
            if i + 1 < len(block_ids):
                cfg.add_edge(bid, block_ids[i + 1])

    # Set entry and exit
    cfg.entry_id = block_ids[0] if block_ids else -1
    for bid in block_ids:
        block = cfg.blocks[bid]
        if not block.successors:
            cfg.exit_ids.append(bid)

    return cfg


# ---------- Example: Build CFG for a While Loop ----------

def demo_while_loop_cfg():
    """Build CFG for: while (i < n) { s = s + i; i = i + 1; }"""
    instructions = [
        TACInstr(0, "label", result="L1"),
        TACInstr(1, "<", "i", "n", "t1"),
        TACInstr(2, "iffalse", "t1", result="L2"),
        TACInstr(3, "+", "s", "i", "t2"),
        TACInstr(4, "copy", "t2", result="s"),
        TACInstr(5, "+", "i", "1", "t3"),
        TACInstr(6, "copy", "t3", result="i"),
        TACInstr(7, "goto", result="L1"),
        TACInstr(8, "label", result="L2"),
    ]

    cfg = build_cfg(instructions)
    print(cfg)


def demo_if_else_cfg():
    """Build CFG for: if (x < y) max = y; else max = x;"""
    instructions = [
        TACInstr(0, "<", "x", "y", "t1"),
        TACInstr(1, "iffalse", "t1", result="L1"),
        TACInstr(2, "copy", "y", result="max"),
        TACInstr(3, "goto", result="L2"),
        TACInstr(4, "label", result="L1"),
        TACInstr(5, "copy", "x", result="max"),
        TACInstr(6, "label", result="L2"),
    ]

    cfg = build_cfg(instructions)
    print(cfg)


if __name__ == "__main__":
    print("--- While Loop CFG ---")
    demo_while_loop_cfg()
    print()
    print("--- If-Else CFG ---")
    demo_if_else_cfg()
```

---

## 5. 정적 단일 대입(Static Single Assignment, SSA) 형식

### 5.1 SSA란?

정적 단일 대입(Static Single Assignment, SSA) 형식은 모든 변수가 정확히 한 번 할당되는 IR 속성입니다. 원래 프로그램에서 변수가 여러 곳에서 할당된다면, SSA는 각각 고유한 이름을 가진 그 변수의 별개 **버전(versions)**을 만듭니다.

**원래 코드**:
```
x = 1
x = x + 1
y = x * 2
```

**SSA 형식**:
```
x1 = 1
x2 = x1 + 1
y1 = x2 * 2
```

각 정의는 새로운 버전을 만듭니다. 이 속성은 모든 변수의 정의가 고유하고 찾기 쉽기 때문에 많은 최적화를 더 단순하고 효율적으로 만듭니다.

### 5.2 파이 함수의 필요성(The Need for Phi Functions)

복잡한 상황은 제어 흐름 그래프의 **합류 지점(join points)**에서 발생합니다 -- 두 개 이상의 경로가 수렴하는 곳. 다음을 고려해 보세요:

```
if (cond)
    x = 1       // Path A
else
    x = 2       // Path B
y = x + 3       // Which x?
```

if-else 이후, 어떤 버전의 `x`를 사용해야 할까요? 런타임에 어떤 경로가 취해졌는지 정적으로 결정할 수 없습니다. SSA는 **파이 함수(phi function)** ($\phi$-함수)로 이를 해결합니다:

```
if (cond)
    x1 = 1       // Path A
else
    x2 = 2       // Path B
x3 = phi(x1, x2) // Merge point
y1 = x3 + 3
```

파이 함수 $x_3 = \phi(x_1, x_2)$는 "제어가 경로 A에서 왔으면 $x_1$을 사용하고, 경로 B에서 왔으면 $x_2$를 사용한다"를 의미합니다.

**형식적으로**: $k$개의 선행자 $B_1, B_2, \ldots, B_k$를 가진 블록 $B$의 진입부에서 $\phi$-함수는 다음과 같은 형태를 가집니다:

$$x_i = \phi(x_{j_1}, x_{j_2}, \ldots, x_{j_k})$$

여기서 $x_{j_m}$은 선행자 $B_m$에서 도달하는 $x$의 버전입니다.

### 5.3 SSA의 속성

1. **고유한 정의**: 모든 변수는 정확히 하나의 정의 지점을 가짐
2. **사용-정의 체인이 자명**: 사용 지점에서 변수의 정의를 찾는 것이 즉각적
3. **정의-사용 체인이 간결**: 각 정의는 명확한 사용 집합을 가짐
4. **희소 표현**: 분석이 프로그램 지점보다 변수에 대해 작동

### 5.4 지배와 지배 경계(Dominance and Dominance Frontiers)

$\phi$-함수를 효율적으로 배치하기 위해 **지배(dominance)** 개념이 필요합니다.

**정의**: CFG에서 노드 $d$가 노드 $n$을 **지배(dominates)**한다($d \;\text{dom}\; n$으로 표기)는 것은 진입 노드에서 $n$으로의 모든 경로가 반드시 $d$를 통과해야 함을 의미합니다.

**즉각 지배자(Immediate dominator)**: 노드 $d$가 $n$의 **즉각 지배자(immediate dominator)**($\text{idom}(n) = d$로 표기)라는 것은 $d$가 $n$을 지배하고, $d \neq n$이며, $n$의 다른 모든 지배자도 $d$를 지배함을 의미합니다.

**지배 트리(Dominator tree)**: 즉각 지배자 관계는 진입 노드를 루트로 하는 트리를 형성합니다.

**지배 경계(Dominance frontier)**: 노드 $d$의 **지배 경계(dominance frontier)** $DF(d)$는 다음을 만족하는 노드 $n$의 집합입니다:
- $d$가 $n$의 선행자를 지배하지만,
- $d$가 $n$을 엄격하게 지배하지는 않음

직관적으로, $d$의 지배 경계는 $d$의 지배가 "끝나는" 노드들의 집합입니다 -- 이것들이 정확히 $d$에서 정의된 변수에 대한 $\phi$-함수가 필요할 수 있는 합류 지점입니다.

### 5.5 알고리즘: 지배 경계 계산

다음 Cooper, Harvey, Kennedy (2001)의 알고리즘은 지배 경계를 효율적으로 계산합니다:

```
for each node b:
    if b has multiple predecessors:
        for each predecessor p of b:
            runner = p
            while runner != idom(b):
                DF(runner) = DF(runner) ∪ {b}
                runner = idom(runner)
```

### 5.6 알고리즘: 파이 함수 배치

$\phi$-함수를 배치하는 고전 알고리즘은 지배 경계를 사용합니다:

```
입력:  지배 경계가 있는 CFG, 변수 집합
출력: 파이 함수 배치

각 변수 v에 대해:
    worklist = v를 정의하는 블록 집합
    ever_on_worklist = worklist의 복사본
    has_phi = 빈 집합

    while worklist가 비어있지 않을 때:
        worklist에서 블록 b를 제거
        DF(b)의 각 블록 d에 대해:
            if d가 has_phi에 없으면:
                d의 시작 부분에 v에 대한 phi-함수 삽입
                has_phi = has_phi ∪ {d}
                if d가 ever_on_worklist에 없으면:
                    ever_on_worklist = ever_on_worklist ∪ {d}
                    d를 worklist에 추가
```

### 5.7 알고리즘: 변수 이름 바꾸기

$\phi$-함수를 배치한 후, 지배 트리를 순회하며 변수를 이름 바꿉니다:

```
counter[v] = 0 (모든 변수 v에 대해)
stack[v] = 비어있음 (모든 변수 v에 대해)

function rename(block b):
    b에서 각 phi-함수 "v = phi(...)"에 대해:
        i = counter[v]++
        stack[v]에 i를 push
        phi에서 v를 v_i로 교체

    b의 각 명령어에 대해:
        변수 v의 각 사용에 대해:
            v를 v_{top(stack[v])}으로 교체
        변수 v의 각 정의에 대해:
            i = counter[v]++
            stack[v]에 i를 push
            v를 v_i로 교체

    b의 각 후계자 s에 대해:
        s의 각 phi-함수에 대해:
            j를 s의 선행자 목록에서 b의 인덱스라 하자
            j번째 피연산자 v를 v_{top(stack[v])}으로 교체

    지배 트리에서 b의 각 자식 c에 대해:
        rename(c)

    b에서 v_i를 정의한 각 phi 또는 명령어에 대해:
        stack[v]에서 pop
```

### 5.8 완전한 SSA 예시

다음 프로그램을 고려해 보세요:

```
    a = 0
    b = 1
L1: if a >= n goto L2
    c = a + b
    a = b
    b = c
    goto L1
L2: return b
```

**CFG**:
```
B0: a=0, b=1
 │
 ▼
B1: if a>=n goto L2  ◀──┐
 │        │              │
 │ false  │ true         │
 ▼        │              │
B2: c=a+b │              │
    a=b   │              │
    b=c   │              │
    goto L1 ─────────────┘
          │
          ▼
B3: return b
```

**SSA 형식**:
```
B0: a0 = 0
    b0 = 1
    goto B1

B1: a2 = phi(a0, a3)    // From B0 or B2
    b2 = phi(b0, b3)    // From B0 or B2
    if a2 >= n goto B3

B2: c1 = a2 + b2
    a3 = b2
    b3 = c1
    goto B1

B3: b4 = phi(b2)         // (trivial, single predecessor)
    return b4
```

참고: B1이 합류 지점이기 때문에 (B0과 B2 모두에서 도달) $\phi$-함수가 배치됩니다.

### 5.9 Python 구현: SSA 구성

```python
"""Simplified SSA construction for a basic CFG."""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SSABlock:
    """A basic block in SSA form."""
    id: int
    phis: dict = field(default_factory=dict)     # var -> phi operands
    instructions: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    idom: int = -1  # Immediate dominator block id
    dom_frontier: set = field(default_factory=set)
    dom_children: list = field(default_factory=list)


class SSABuilder:
    """
    Build SSA form from a simple CFG.

    This is a simplified demonstration that illustrates the key ideas:
    1. Compute dominance (simplified)
    2. Compute dominance frontiers
    3. Place phi functions
    4. Rename variables
    """

    def __init__(self):
        self.blocks: dict[int, SSABlock] = {}
        self.var_defs: dict[str, set] = defaultdict(set)  # var -> blocks defining it
        self.all_vars: set = set()

    def add_block(self, block_id: int, instructions: list,
                  predecessors: list, successors: list):
        """Add a block with its instructions and connectivity."""
        block = SSABlock(
            id=block_id,
            instructions=instructions,
            predecessors=predecessors,
            successors=successors,
        )
        self.blocks[block_id] = block

        # Track which blocks define which variables
        for instr in instructions:
            if "=" in instr and not instr.strip().startswith("if") \
               and not instr.strip().startswith("goto") \
               and not instr.strip().startswith("return"):
                var = instr.split("=")[0].strip()
                self.var_defs[var].add(block_id)
                self.all_vars.add(var)

    def compute_dominators(self, entry: int):
        """
        Compute immediate dominators using a simplified iterative algorithm.
        """
        all_ids = sorted(self.blocks.keys())

        # Initialize: entry dominates itself; others dominated by all
        dom = {b: set(all_ids) for b in all_ids}
        dom[entry] = {entry}

        changed = True
        while changed:
            changed = False
            for b in all_ids:
                if b == entry:
                    continue
                preds = self.blocks[b].predecessors
                if not preds:
                    continue
                new_dom = set.intersection(*(dom[p] for p in preds))
                new_dom.add(b)
                if new_dom != dom[b]:
                    dom[b] = new_dom
                    changed = True

        # Compute immediate dominators from dominator sets
        for b in all_ids:
            if b == entry:
                self.blocks[b].idom = -1
                continue
            strict_doms = dom[b] - {b}
            # idom is the dominator closest to b (dominated by all others)
            idom = None
            for d in strict_doms:
                if all(d in dom[other] or other == d
                       for other in strict_doms):
                    if idom is None or d not in dom.get(idom, set()) or d == idom:
                        # d should be the one dominated by fewest
                        if idom is None or idom in dom[d]:
                            idom = d
            if idom is not None:
                self.blocks[b].idom = idom
                self.blocks[idom].dom_children.append(b)

    def compute_dominance_frontiers(self):
        """Compute dominance frontiers using the Cooper-Harvey-Kennedy algorithm."""
        for b_id, block in self.blocks.items():
            if len(block.predecessors) >= 2:
                for p in block.predecessors:
                    runner = p
                    while runner != block.idom and runner != -1:
                        self.blocks[runner].dom_frontier.add(b_id)
                        runner = self.blocks[runner].idom

    def place_phi_functions(self):
        """Insert phi functions at dominance frontier nodes."""
        for var in self.all_vars:
            worklist = list(self.var_defs[var])
            ever_on_worklist = set(worklist)
            has_phi = set()

            while worklist:
                block_id = worklist.pop()
                for df_id in self.blocks[block_id].dom_frontier:
                    if df_id not in has_phi:
                        # Insert phi: var = phi(var, var, ...)
                        n_preds = len(self.blocks[df_id].predecessors)
                        self.blocks[df_id].phis[var] = [var] * n_preds
                        has_phi.add(df_id)
                        if df_id not in ever_on_worklist:
                            ever_on_worklist.add(df_id)
                            worklist.append(df_id)

    def rename_variables(self, entry: int):
        """Rename variables to achieve SSA form."""
        counter = defaultdict(int)
        stack = defaultdict(list)

        def new_name(var: str) -> str:
            i = counter[var]
            counter[var] += 1
            name = f"{var}{i}"
            stack[var].append(name)
            return name

        def current_name(var: str) -> str:
            if stack[var]:
                return stack[var][-1]
            return var + "0"

        def rename(block_id: int):
            block = self.blocks[block_id]
            push_counts = defaultdict(int)

            # Rename phi definitions
            for var in list(block.phis.keys()):
                new = new_name(var)
                push_counts[var] += 1
                # Store the SSA name for this phi
                block.phis[var] = {
                    "result": new,
                    "operands": block.phis[var]  # Will rename operands from preds
                }

            # Rename instructions
            new_instructions = []
            for instr in block.instructions:
                new_instr = instr

                # Handle assignments: replace uses then define
                if "=" in instr and not instr.strip().startswith("if") \
                   and not instr.strip().startswith("goto") \
                   and not instr.strip().startswith("return"):
                    parts = instr.split("=", 1)
                    lhs_var = parts[0].strip()
                    rhs = parts[1].strip()

                    # Rename uses in RHS
                    for v in self.all_vars:
                        if v in rhs:
                            rhs = rhs.replace(v, current_name(v))

                    # Rename definition in LHS
                    new_lhs = new_name(lhs_var)
                    push_counts[lhs_var] += 1
                    new_instr = f"{new_lhs} = {rhs}"

                elif instr.strip().startswith("return"):
                    parts = instr.strip().split()
                    if len(parts) > 1:
                        v = parts[1]
                        if v in self.all_vars:
                            new_instr = f"return {current_name(v)}"

                elif instr.strip().startswith("if"):
                    new_instr = instr
                    for v in self.all_vars:
                        if v in new_instr:
                            new_instr = new_instr.replace(v, current_name(v))

                new_instructions.append(new_instr)

            block.instructions = new_instructions

            # Fill in phi operands in successors
            for succ_id in block.successors:
                succ = self.blocks[succ_id]
                pred_index = succ.predecessors.index(block_id)
                for var, phi_info in succ.phis.items():
                    if isinstance(phi_info, dict):
                        phi_info["operands"][pred_index] = current_name(var)

            # Recurse on dominator tree children
            for child_id in block.dom_children:
                rename(child_id)

            # Pop stacks
            for var, count in push_counts.items():
                for _ in range(count):
                    if stack[var]:
                        stack[var].pop()

        rename(entry)

    def build(self, entry: int):
        """Run the full SSA construction pipeline."""
        self.compute_dominators(entry)
        self.compute_dominance_frontiers()
        self.place_phi_functions()
        self.rename_variables(entry)

    def print_ssa(self):
        """Pretty-print the SSA form."""
        for bid in sorted(self.blocks.keys()):
            block = self.blocks[bid]
            print(f"B{bid}:")

            # Print phi functions
            for var, phi_info in block.phis.items():
                if isinstance(phi_info, dict):
                    operands = ", ".join(phi_info["operands"])
                    print(f"  {phi_info['result']} = phi({operands})")

            # Print instructions
            for instr in block.instructions:
                print(f"  {instr}")
            print()


# ---------- Example ----------

def demo_fibonacci_ssa():
    """
    Convert a Fibonacci-like loop to SSA.

    Original:
        a = 0
        b = 1
    L1: if a >= n goto L2
        c = a + b
        a = b
        b = c
        goto L1
    L2: return b
    """
    print("=== Fibonacci Loop in SSA Form ===\n")

    builder = SSABuilder()

    builder.add_block(0,
        instructions=["a = 0", "b = 1"],
        predecessors=[],
        successors=[1])

    builder.add_block(1,
        instructions=["if a >= n goto L2"],
        predecessors=[0, 2],
        successors=[2, 3])

    builder.add_block(2,
        instructions=["c = a + b", "a = b", "b = c", "goto L1"],
        predecessors=[1],
        successors=[1])

    builder.add_block(3,
        instructions=["return b"],
        predecessors=[1],
        successors=[])

    builder.build(entry=0)
    builder.print_ssa()


if __name__ == "__main__":
    demo_fibonacci_ssa()
```

---

## 6. 방향 비순환 그래프(Directed Acyclic Graphs, DAGs)

### 6.1 표현식을 위한 DAG

**방향 비순환 그래프(Directed Acyclic Graph, DAG)**는 중복 계산을 제거하는 표현식의 간결한 표현입니다. 트리와 달리, DAG는 공통 부분 표현식(common subexpressions)의 공유를 허용합니다.

**표현식**: `(a + b) * (a + b) - c`

**트리 표현** (중복):
```
        -
       / \
      *   c
     / \
    +   +
   / \ / \
  a  b a  b
```

**DAG 표현** (공유):
```
        -
       / \
      *   c
     /|
    +
   / \
  a   b
```

DAG에서 공통 부분 표현식 `a + b`는 한 번 계산되고 그 결과가 공유됩니다.

### 6.2 DAG 구성

표현식으로부터 DAG를 구성하는 알고리즘은 표현식을 아래에서 위로 처리합니다:

```
function build_dag(expr):
    if expr is a leaf (variable or constant):
        if node for expr already exists:
            return existing node
        else:
            create new leaf node
            return it

    if expr is "left op right":
        left_node = build_dag(left)
        right_node = build_dag(right)

        if interior node (op, left_node, right_node) already exists:
            return existing node
        else:
            create new interior node
            return it
```

**핵심**: 같은 (op, left, right) 트리플을 가진 기존 노드를 확인하기 위해 해시 테이블을 사용합니다.

### 6.3 DAG의 활용

1. **공통 부분 표현식 제거(Common subexpression elimination)**: 공유 노드는 한 번 수행되는 계산을 나타냄
2. **명령어 정렬(Instruction ordering)**: DAG 간선은 부분 순서를 정의하며, 위상 정렬은 유효한 실행 순서를 제공
3. **죽은 코드 탐지(Dead code detection)**: 나가는 사용이 없는 노드(최종 결과 제외)는 죽은 코드일 수 있음
4. **대수적 최적화(Algebraic optimization)**: DAG 노드에 단순화 규칙 적용 가능

### 6.4 Python 구현: 표현식 DAG

```python
"""DAG construction for common subexpression elimination."""

from dataclasses import dataclass, field


@dataclass
class DAGNode:
    """A node in the expression DAG."""
    id: int
    op: str           # Operator or "leaf"
    value: str = ""   # For leaf nodes: variable name or constant
    left: int = -1    # Left child node id
    right: int = -1   # Right child node id
    labels: list = field(default_factory=list)  # Variable names assigned to this node

    def __str__(self):
        if self.op == "leaf":
            label_str = f" [{', '.join(self.labels)}]" if self.labels else ""
            return f"n{self.id}: {self.value}{label_str}"
        else:
            label_str = f" [{', '.join(self.labels)}]" if self.labels else ""
            return f"n{self.id}: n{self.left} {self.op} n{self.right}{label_str}"


class ExpressionDAG:
    """Build and manage an expression DAG."""

    def __init__(self):
        self.nodes: dict[int, DAGNode] = {}
        self._next_id = 0
        self._leaf_map: dict[str, int] = {}         # value -> node_id
        self._interior_map: dict[tuple, int] = {}    # (op, left, right) -> node_id

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def find_or_create_leaf(self, value: str) -> int:
        """Find or create a leaf node for a variable/constant."""
        if value in self._leaf_map:
            return self._leaf_map[value]
        nid = self._new_id()
        node = DAGNode(id=nid, op="leaf", value=value)
        node.labels.append(value)
        self.nodes[nid] = node
        self._leaf_map[value] = nid
        return nid

    def find_or_create_op(self, op: str, left_id: int, right_id: int) -> int:
        """Find or create an interior node for an operation."""
        key = (op, left_id, right_id)

        # Check commutativity for + and *
        if op in ("+", "*"):
            alt_key = (op, right_id, left_id)
            if alt_key in self._interior_map:
                return self._interior_map[alt_key]

        if key in self._interior_map:
            return self._interior_map[key]

        nid = self._new_id()
        node = DAGNode(id=nid, op=op, left=left_id, right=right_id)
        self.nodes[nid] = node
        self._interior_map[key] = nid
        return nid

    def assign_label(self, node_id: int, var_name: str):
        """Assign a variable name to a DAG node (for variable tracking)."""
        node = self.nodes[node_id]
        if var_name not in node.labels:
            node.labels.append(var_name)
        # Update leaf map so future references to this variable use this node
        self._leaf_map[var_name] = node_id

    def process_tac(self, instructions: list[str]):
        """
        Process a sequence of TAC instructions and build the DAG.

        Each instruction should be of the form:
            'x = y op z'  or  'x = y'
        """
        for instr in instructions:
            parts = instr.strip().split()

            if len(parts) == 3:
                # Copy: x = y
                target, _, source = parts
                source_id = self.find_or_create_leaf(source)
                self.assign_label(source_id, target)

            elif len(parts) == 5:
                # Binary: x = y op z
                target, _, left, op, right = parts
                left_id = self.find_or_create_leaf(left)
                right_id = self.find_or_create_leaf(right)
                result_id = self.find_or_create_op(op, left_id, right_id)
                self.assign_label(result_id, target)

    def print_dag(self):
        """Pretty-print the DAG."""
        print("=== Expression DAG ===")
        for nid in sorted(self.nodes.keys()):
            print(f"  {self.nodes[nid]}")

    def generate_optimized_tac(self) -> list[str]:
        """
        Generate optimized TAC from the DAG using topological order.
        Common subexpressions are computed only once.
        """
        # Topological sort (using DFS post-order)
        visited = set()
        order = []

        def dfs(nid):
            if nid in visited or nid == -1:
                return
            visited.add(nid)
            node = self.nodes[nid]
            if node.left != -1:
                dfs(node.left)
            if node.right != -1:
                dfs(node.right)
            order.append(nid)

        # Start DFS from all root-like nodes (those not used as children)
        children = set()
        for n in self.nodes.values():
            if n.left != -1:
                children.add(n.left)
            if n.right != -1:
                children.add(n.right)
        roots = [nid for nid in self.nodes if nid not in children]

        for root in roots:
            dfs(root)

        # Generate TAC
        tac = []
        node_result_name = {}

        for nid in order:
            node = self.nodes[nid]
            if node.op == "leaf":
                # Leaf nodes: the variable/constant itself is the name
                node_result_name[nid] = node.value
            else:
                left_name = node_result_name.get(node.left, "?")
                right_name = node_result_name.get(node.right, "?")
                result_name = node.labels[0] if node.labels else f"t{nid}"
                tac.append(f"{result_name} = {left_name} {node.op} {right_name}")
                node_result_name[nid] = result_name

        return tac


# ---------- Example ----------

def demo_cse_dag():
    """
    Demonstrate CSE via DAG for:
        t1 = a + b
        t2 = a + b    <-- common subexpression
        t3 = t1 * t2
        t4 = t3 - c
    """
    print("Original TAC:")
    instructions = [
        "t1 = a + b",
        "t2 = a + b",
        "t3 = t1 * t2",
        "t4 = t3 - c",
    ]
    for instr in instructions:
        print(f"  {instr}")
    print()

    dag = ExpressionDAG()
    dag.process_tac(instructions)
    dag.print_dag()
    print()

    optimized = dag.generate_optimized_tac()
    print("Optimized TAC (CSE eliminated):")
    for instr in optimized:
        print(f"  {instr}")


if __name__ == "__main__":
    demo_cse_dag()
```

**예상 출력**:
```
Original TAC:
  t1 = a + b
  t2 = a + b
  t3 = t1 * t2
  t4 = t3 - c

=== Expression DAG ===
  n0: a [a]
  n1: b [b]
  n2: n0 + n1 [t1, t2]
  n3: n2 * n2 [t3]
  n4: c [c]
  n5: n3 - n4 [t4]

Optimized TAC (CSE eliminated):
  t1 = a + b
  t3 = t1 * t1
  t4 = t3 - c
```

`t2 = a + b`가 제거된 것에 주목하세요. DAG가 `a + b`가 이미 `t1`으로 계산되었음을 인식했기 때문입니다.

---

## 7. 선형화: CFG에서 선형 코드로(Linearization: From CFG Back to Linear Code)

### 7.1 문제

CFG(또는 그것의 SSA 형식)에 대한 최적화를 수행한 후, 코드 생성기를 위해 다시 선형 명령어 시퀀스로 변환해야 합니다. 이 과정을 **선형화(linearization)** 또는 **코드 레이아웃(code layout)**이라고 합니다.

### 7.2 블록 정렬 전략(Block Ordering Strategies)

기본 블록이 최종 출력에 나타나는 순서는 다음에 영향을 미칩니다:
- **폴-스루 효율성**: 무조건 점프 최소화
- **캐시 동작**: 자주 실행되는 경로를 연속으로 유지
- **분기 예측**: 가능성 높은 경로를 순차적으로 배치

일반적인 전략:

1. **역 후위 순서(Reverse postorder, RPO)**: CFG에 대한 DFS를 수행하고 역 후위 순서로 블록을 방문합니다. 이는 블록의 지배자가 그 앞에 나타나도록 보장합니다.

2. **추적 기반 레이아웃(Trace-based layout)**: 자주 실행되는 경로(추적)를 식별하고 연속으로 배치합니다.

3. **하향식 레이아웃(Bottom-up layout)**: 출구 블록에서 시작하여 역방향으로 작업합니다.

### 7.3 역 후위 순서 알고리즘

```python
def reverse_postorder(cfg, entry_id):
    """
    Compute reverse postorder of CFG blocks.
    Returns a list of block IDs in RPO.
    """
    visited = set()
    postorder = []

    def dfs(block_id):
        if block_id in visited:
            return
        visited.add(block_id)
        block = cfg.blocks[block_id]
        for succ in block.successors:
            dfs(succ)
        postorder.append(block_id)

    dfs(entry_id)
    return list(reversed(postorder))
```

### 7.4 중복 점프 제거(Eliminating Redundant Jumps)

선형화 후, 대상 블록이 점프하는 블록 바로 다음에 오기 때문에 (폴-스루) 일부 점프가 불필요해집니다. 간단한 패스가 이를 제거합니다:

```python
def eliminate_redundant_jumps(block_order, cfg):
    """
    Remove goto instructions when the target is the next block
    in the linear layout.
    """
    for i, block_id in enumerate(block_order):
        block = cfg.blocks[block_id]
        if not block.instructions:
            continue

        last = block.instructions[-1]
        if last.is_unconditional_jump:
            # Check if the next block in layout is the jump target
            if i + 1 < len(block_order):
                next_block = cfg.blocks[block_order[i + 1]]
                if next_block.label == last.jump_target:
                    # Remove redundant goto
                    block.instructions.pop()
```

### 7.5 SSA 해체(Deconstructing SSA)

코드 생성 전에, $\phi$-함수는 하드웨어 대응이 없기 때문에 제거되어야 합니다. 표준 접근 방식은 각 $\phi$-함수를 선행자 블록의 **복사 명령어(copy instructions)**로 교체합니다.

주어진:
```
B1: ...
    goto B3

B2: ...
    goto B3

B3: x3 = phi(x1, x2)    // x1 from B1, x2 from B2
```

해체 후:
```
B1: ...
    x3 = x1              // copy added
    goto B3

B2: ...
    x3 = x2              // copy added
    goto B3

B3: // phi removed
```

이렇게 하면 나중에 **복사 전파(copy propagation)** 또는 **레지스터 합치기(register coalescing)** 패스에 의해 종종 제거될 수 있는 복사 명령어가 도입됩니다.

**복잡한 경우**: 여러 $\phi$-함수가 서로의 피연산자를 참조할 때 ("교환 문제"), 단순한 접근 방식은 잘못된 결과를 생성할 수 있습니다. 해결책은 신중한 순서로 복사를 삽입하거나 추가 임시 변수를 도입하는 것입니다.

---

## 8. 요약

이 레슨에서 우리는 컴파일러 설계에서 중간 표현의 역할을 탐구했습니다:

1. **IR은 프론트 엔드와 백 엔드를 분리**하여 $m \times n$ 문제를 $m + n$으로 줄입니다.

2. **3-주소 코드(TAC)**는 각 명령어가 최대 세 개의 피연산자를 가지는 중간 수준 IR입니다. 쿼드러플, 트리플, 또는 간접 트리플로 저장할 수 있습니다.

3. **제어 흐름 그래프(CFG)**는 TAC를 제어 흐름 간선으로 연결된 기본 블록으로 조직합니다. 리더는 점프 대상 및 점프 후 명령어로 식별됩니다.

4. **정적 단일 대입(SSA) 형식**은 각 변수를 정확히 한 번 할당합니다. 파이 함수는 합류 지점을 처리합니다. 구성에는 지배 경계, 파이 배치, 변수 이름 바꾸기가 필요합니다.

5. **DAG**는 공통 부분 표현식을 자연스럽게 노출하는 간결한 표현식 표현을 제공합니다.

6. **선형화**는 CFG를 순차적 코드로 다시 변환하고, SSA 해체는 파이 함수를 복사 명령어로 교체합니다.

이러한 표현들은 현대 컴파일러 인프라의 근간을 형성하며, 이후 레슨에서 공부할 강력한 최적화 패스를 가능하게 합니다.

---

## 연습 문제

### 연습 1: TAC 생성

다음 코드를 3-주소 코드로 번역하세요:

```
x = 2 * a + b
y = a * a - b * b
if (x > y)
    z = x - y
else
    z = y - x
result = z * z
```

임시 변수, 레이블, 점프를 포함한 완전한 TAC를 보여주세요.

### 연습 2: 기본 블록 식별

다음 TAC에서 리더를 식별하고 코드를 기본 블록으로 분할하세요:

```
(1)  i = 0
(2)  j = 0
(3)  t1 = i < 10
(4)  iffalse t1 goto L3
(5)  j = 0
(6)  t2 = j < 10
(7)  iffalse t2 goto L2
(8)  t3 = i * 10
(9)  t4 = t3 + j
(10) a[t4] = 0
(11) j = j + 1
(12) goto L1
(13) i = i + 1
(14) goto L0
(15) return
```

L0은 명령어 3, L1은 명령어 6, L2는 명령어 13, L3는 명령어 15입니다.

### 연습 3: SSA 변환

다음 코드를 SSA 형식으로 변환하세요. 모든 파이 함수와 그 피연산자를 명확하게 보여주세요.

```
B0: x = 1
    y = 2
    goto B1

B1: if (x < 10) goto B2 else goto B3

B2: x = x + 1
    y = y * 2
    goto B1

B3: z = x + y
    return z
```

### 연습 4: DAG 구성

다음 기본 블록에 대한 DAG를 구성하세요:

```
a = b + c
d = b + c
e = a - d
f = a * e
g = f + e
```

공유되는 부분 표현식을 식별하고 최적화된 TAC를 작성하세요.

### 연습 5: 지배 경계

다음 CFG에 대해 다음을 계산하세요:
1. 지배 트리
2. 각 노드의 지배 경계

```
Entry → B1
B1 → B2, B3
B2 → B4
B3 → B4
B4 → B5, B6
B5 → B1
B6 → Exit
```

### 연습 6: 구현 도전

다음을 처리하도록 Python TAC 생성기를 확장하세요:
1. **함수 호출**: `param x`, `call f, n`, 그리고 `result = call f, n`
2. **배열 접근**: `x = a[i]`와 `a[i] = x`

배열의 합을 계산하는 함수에 대한 TAC를 생성하는 테스트 케이스를 작성하세요.

---

[이전: 의미 분석](./08_Semantic_Analysis.md) | [다음: 런타임 환경](./10_Runtime_Environments.md) | [개요](./00_Overview.md)
