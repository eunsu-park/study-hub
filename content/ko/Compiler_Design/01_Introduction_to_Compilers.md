# 레슨 1: 컴파일러 입문(Introduction to Compilers)

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. 컴파일러의 정의 및 인터프리터와의 차이점 설명
2. FORTRAN부터 현대 시스템까지 컴파일러의 역사적 발전 추적
3. 컴파일 과정의 각 단계 식별 및 설명
4. 프론트엔드(front-end)와 백엔드(back-end)의 구분 설명
5. 단일 패스(single-pass) vs. 다중 패스(multi-pass) 컴파일 전략 이해
6. 부트스트래핑(bootstrapping)과 크로스 컴파일(cross-compilation) 설명
7. T-다이어그램(tombstone diagram) 읽기 및 작성
8. 간단한 프로그램의 모든 컴파일 단계 추적

---

## 1. 컴파일러란 무엇인가?

**컴파일러(compiler)**는 한 언어(**소스 언어(source language)**)로 작성된 소스 코드를 다른 언어(**목적 언어(target language)**)로 변환하면서, 변환 과정에서 감지된 오류를 보고하는 프로그램입니다.

$$\text{Compiler}: \text{Source Language} \longrightarrow \text{Target Language}$$

보다 정확히 말하면, 컴파일러는 프로그램을 매핑하는 함수입니다:

$$C: \mathcal{P}_S \rightarrow \mathcal{P}_T$$

여기서 $\mathcal{P}_S$는 소스 언어의 유효한 프로그램 집합이고 $\mathcal{P}_T$는 목적 언어의 프로그램 집합이며, 모든 입력 $x$에 대해 다음을 만족합니다:

$$\text{meaning}(P_S, x) = \text{meaning}(C(P_S), x)$$

**의미론적 동등성(semantic equivalence)** 요건은 매우 중요합니다: 컴파일된 프로그램은 모든 유효한 입력에 대해 소스 프로그램과 동일한 관찰 가능한 동작을 생성해야 합니다.

### 일반적인 소스-목적 언어 쌍

| 소스 언어 | 목적 언어 | 예시 |
|-----------------|-----------------|---------|
| C | x86 기계 코드 | GCC, Clang |
| Java | JVM 바이트코드 | javac |
| TypeScript | JavaScript | tsc |
| Python | Python 바이트코드 (.pyc) | CPython |
| Rust | LLVM IR, 이후 기계 코드 | rustc |
| SASS | CSS | sass 컴파일러 |
| LaTeX | DVI/PDF | pdflatex |

### 컴파일러 vs. 트랜스파일러

**트랜스파일러(transpiler)**(소스 간 컴파일러)는 비슷한 추상화 수준의 언어 간에 변환합니다. TypeScript에서 JavaScript로, CoffeeScript에서 JavaScript로의 변환이 트랜스파일이에 해당합니다. 컴파일과 트랜스파일의 경계는 모호합니다; 중요한 것은 둘 다 의미론(semantics)을 보존한다는 점입니다.

---

## 2. 컴파일러 vs. 인터프리터

**인터프리터(interpreter)**는 별도의 목적 프로그램을 생성하지 않고 소스 프로그램을 직접 실행합니다.

```
Compiler workflow:
  Source Code  --->  [Compiler]  --->  Target Code  --->  [CPU/VM]  --->  Output
                                          (stored)

Interpreter workflow:
  Source Code  --->  [Interpreter + Input]  --->  Output
                     (no separate target)
```

### 주요 차이점

| 측면 | 컴파일러 | 인터프리터 |
|--------|----------|-------------|
| 출력 | 목적 프로그램 생성 | 목적 프로그램 미생성 |
| 실행 | 목적 코드 독립 실행 | 인터프리터 필요 |
| 속도 (시작) | 느림 (컴파일 오버헤드) | 빠름 (즉시 실행) |
| 속도 (런타임) | 빠름 (최적화된 목적 코드) | 느림 (반복 분석) |
| 오류 보고 | 실행 전 모든 오류 보고 | 런타임에 오류 발생 |
| 메모리 | 목적 프로그램 저장 | 소스 매번 재분석 |

### 하이브리드 접근법

대부분의 현대 언어 구현은 **하이브리드(hybrid)** 접근법을 사용합니다:

1. **Java**: 바이트코드로 컴파일(javac), 이후 JVM에서 인터프리트/JIT 컴파일
2. **Python**: 바이트코드(.pyc)로 컴파일, 이후 CPython VM에서 인터프리트
3. **JavaScript**: 파싱 후 핫 경로를 JIT 컴파일 (V8, SpiderMonkey)
4. **C#/.NET**: CIL로 컴파일, 이후 네이티브 코드로 JIT 컴파일

```python
# Python's hybrid approach: you can see the bytecode
import dis

def add(a, b):
    return a + b

# Show the compiled bytecode instructions
dis.dis(add)
# Output:
#   2           0 LOAD_FAST                0 (a)
#               2 LOAD_FAST                1 (b)
#               4 BINARY_ADD
#               6 RETURN_VALUE
```

### 컴파일 스펙트럼

이진적인 구분보다는 스펙트럼이 존재합니다:

```
Pure Interpreter  <----------------------------->  Ahead-of-Time Compiler
     |                    |              |                |
   Shell scripts    CPython (bytecode)  JIT (V8, JVM)   GCC/Clang
```

---

## 3. 역사적 발전

### 초기 시대 (1950년대)

컴파일러가 등장하기 전, 프로그래머들은 기계 코드나 어셈블리 언어를 직접 작성했습니다. 기계가 고수준 표기법을 효율적인 기계 코드로 번역할 수 있다는 아이디어는 회의적인 시선을 받았습니다.

**1957년 -- FORTRAN 컴파일러** (John Backus, IBM)
- 최초의 최적화 컴파일러
- 개발에 18 인년(person-year) 소요
- 목표: 수작업으로 작성한 어셈블리만큼 효율적인 코드 생성
- 성공을 통해 고수준 언어의 실용성 증명

**1960년 -- COBOL 컴파일러**
- Grace Hopper의 A-0 (1952) 작업이 기반 마련
- 기계 간 이식성을 위해 설계된 최초의 언어

**1962년 -- LISP 컴파일러**
- 최초의 자기 호스팅(self-hosting) 컴파일러 (컴파일러 자신이 컴파일하는 언어로 작성)
- 가비지 컬렉션 도입

### 형식적 기초 (1960-1970년대)

**1960년대 -- 촘스키 계층(Chomsky Hierarchy)과 파싱 이론**
- 노암 촘스키(Noam Chomsky)의 형식 언어 분류
- 컴파일러 설계에 직접 응용
- 구문을 위한 문맥 자유 문법(context-free grammar) 발전

**1965년 -- Knuth의 LR 파싱**
- 도널드 크누스(Donald Knuth)가 LR 파싱 도입
- 결정론적 문맥 자유 언어를 위한 파서를 체계적으로 구축하는 방법 제공

**1970년 -- Yacc (Yet Another Compiler-Compiler)**
- Bell Labs의 Stephen C. Johnson
- LALR(1) 파서 생성기
- 파서 구성을 실용적으로 만듦

**1975년 -- Lex**
- Bell Labs의 Mike Lesk와 Eric Schmidt
- 어휘 분석기 생성기
- Yacc와 함께 동작: Lex가 토큰 처리, Yacc가 문법 처리

### 현대 시대 (1980년대~현재)

**1987년 -- GCC (GNU Compiler Collection)**
- Richard Stallman
- 최초의 주요 오픈 소스 컴파일러
- 다수의 언어 및 목적 플랫폼 지원

**2003년 -- LLVM**
- UIUC의 Chris Lattner
- 모듈형 컴파일러 인프라
- 재사용 가능한 컴포넌트로 컴파일러 구성에 혁명을 일으킴

**2010년대 -- Rust, Go, Swift**
- 현대 컴파일러 기술로 설계된 언어들
- Rust: 소유권 기반 메모리 안전성 (GC 없음)
- Go: 빠른 컴파일을 설계 목표로 설정
- Swift: LLVM 기반으로 구축

**2020년대 -- AI 보조 컴파일(AI-Assisted Compilation)**
- 최적화 휴리스틱을 위한 머신 러닝
- 신경망 프로그램 합성(neural program synthesis)
- LLM 기반 코드 생성 (전통적인 컴파일과는 다른 패러다임)

---

## 4. 컴파일의 단계

컴파일러는 **단계(phase)**의 시퀀스로 구성되며, 각 단계는 프로그램을 하나의 표현에서 다른 표현으로 변환합니다.

```
Source Code (characters)
    |
    v
+-------------------+
| Lexical Analysis  |  (Scanner / Lexer)
+-------------------+
    |  tokens
    v
+-------------------+
| Syntax Analysis   |  (Parser)
+-------------------+
    |  parse tree / AST
    v
+-------------------+
| Semantic Analysis |  (Type checker)
+-------------------+
    |  annotated AST
    v
+-------------------+
| IR Generation     |  (Intermediate Representation)
+-------------------+
    |  IR (three-address code, SSA, ...)
    v
+-------------------+
| Optimization      |  (Machine-independent)
+-------------------+
    |  optimized IR
    v
+-------------------+
| Code Generation   |  (Machine-dependent)
+-------------------+
    |  target code (assembly / machine code)
    v
+-------------------+
| Target Optimization |  (Peephole, scheduling)
+-------------------+
    |
    v
Target Code (machine instructions)
```

두 가지 추가 컴포넌트가 모든 단계에서 실행됩니다:

- **심볼 테이블 관리자(Symbol Table Manager)**: 식별자(이름, 타입, 범위)에 대한 정보 유지
- **오류 처리기(Error Handler)**: 오류 감지, 보고 및 (가능한 경우) 복구

### 1단계: 어휘 분석(Lexical Analysis, 스캐닝)

**렉서(lexer)**(또는 스캐너)는 소스 프로그램을 문자 스트림으로 읽어 **토큰(token)** — 언어의 가장 작은 의미 있는 단위 — 으로 그룹화합니다.

**입력**: 문자 스트림
**출력**: 토큰 스트림

```
Source:  position = initial + rate * 60

Tokens:  [ID:"position"] [ASSIGN:"="] [ID:"initial"] [PLUS:"+"]
         [ID:"rate"] [STAR:"*"] [INT:"60"]
```

각 토큰은 다음을 가집니다:
- **토큰 타입(token type)** (또는 토큰 클래스): `ID`, `INT`, `ASSIGN`, `PLUS` 등
- 선택적 **속성 값(attribute value)**: 실제 어휘소(lexeme) 또는 그 값

렉서는 또한 다음을 수행합니다:
- 공백 및 주석 제거
- 전처리기 지시문 처리 (C의 경우)
- 어휘 오류 보고 (잘못된 문자, 종료되지 않은 문자열)

### 2단계: 구문 분석(Syntax Analysis, 파싱)

**파서(parser)**는 토큰 스트림을 받아 언어의 문법 규칙에 따라 **파스 트리(parse tree)** (또는 직접 AST)를 구성합니다.

**입력**: 토큰 스트림
**출력**: 파스 트리 또는 AST

```
Parse tree for: position = initial + rate * 60

        assignment_stmt
       /       |       \
     ID       '='      expr
  "position"          /  |  \
                   expr  '+'  term
                    |        / | \
                   ID    term '*' factor
                "initial"  |       |
                          ID     INT
                        "rate"   60
```

파서는 다음을 수행합니다:
- 토큰 시퀀스가 언어 문법에 부합하는지 검증
- 구문 오류 보고 (`}`앞에 `;` 예상")
- 오류 후에도 파싱을 계속하기 위한 오류 복구 수행 가능

### 3단계: 의미 분석(Semantic Analysis)

**의미 분석(semantic analysis)**은 문맥 자유 문법으로 포착할 수 없는 의미론적 일관성을 검사합니다.

**입력**: AST
**출력**: 주석 달린/장식된 AST

주요 작업:
- **타입 검사(Type checking)**: `rate * 60`이 유효한가? (float에 int를 곱할 수 있는가?)
- **타입 강제 변환(Type coercion)**: 암묵적 변환 삽입 (int 60 -> float 60.0)
- **이름 해석(Name resolution)**: `rate`의 각 사용이 어떤 선언을 참조하는가?
- **범위 검사(Scope checking)**: `position`이 이 범위에서 가시적인가?
- **확정 할당(Definite assignment)**: `initial`이 사용 전에 초기화되었는가?

```
Annotated AST:

        assignment_stmt
       /       |       \
     ID       '='      expr : float
  "position"          /  |  \
  (type:float)     expr  '+'  term : float
                    |        / | \
                   ID    term '*' factor : float
                "initial"  |       |
               (float)    ID    intToFloat(INT)
                        "rate"     60 -> 60.0
                       (float)
```

### 4단계: 중간 표현(IR, Intermediate Representation) 생성

컴파일러는 프로그램의 기계 독립적(machine-independent) **중간 표현(intermediate representation)**을 생성합니다.

**입력**: 주석 달린 AST
**출력**: IR (예: 3-주소 코드)

```
Three-address code for: position = initial + rate * 60

    t1 = intToFloat(60)
    t2 = rate * t1
    t3 = initial + t2
    position = t3
```

각 명령어는 최대 세 개의 피연산자를 가집니다 (따라서 "3-주소 코드"). 이 평탄한 표현은 트리보다 최적화하기 훨씬 쉽습니다.

### 5단계: 최적화(Optimization)

**최적화기(optimizer)**는 프로그램의 의미론을 변경하지 않고 성능을 개선하기 위해 IR을 변환합니다.

**입력**: IR
**출력**: 최적화된 IR

```
Before optimization:           After optimization:
    t1 = intToFloat(60)           t1 = rate * 60.0
    t2 = rate * t1                position = initial + t1
    t3 = initial + t2
    position = t3
```

적용된 최적화:
- **상수 폴딩(Constant folding)**: `intToFloat(60)` -> `60.0` (컴파일 시점에 계산)
- **죽은 코드 제거(Dead code elimination)**: 불필요한 임시 변수 제거
- **복사 전파(Copy propagation)**: t3의 사용을 정의로 대체

### 6단계: 코드 생성(Code Generation)

**코드 생성기(code generator)**는 최적화된 IR을 목적 기계의 명령어 세트로 매핑합니다.

**입력**: 최적화된 IR
**출력**: 목적 코드 (어셈블리 또는 기계 코드)

```asm
; x86-64 assembly for: position = initial + rate * 60.0
    movsd   xmm0, QWORD PTR [rate]       ; load rate into xmm0
    mulsd   xmm0, QWORD PTR [.LC0]       ; xmm0 = rate * 60.0
    addsd   xmm0, QWORD PTR [initial]    ; xmm0 = initial + rate * 60.0
    movsd   QWORD PTR [position], xmm0   ; store result in position
.LC0:
    .double 60.0
```

코드 생성기는 다음을 수행해야 합니다:
- 적절한 명령어 선택 (**명령어 선택(instruction selection)**)
- 변수를 레지스터에 할당 (**레지스터 할당(register allocation)**)
- 명령어 실행 순서 결정 (**명령어 스케줄링(instruction scheduling)**)

---

## 5. 완전한 예제: 소스에서 목적 코드까지

`a = b + c * 2` 표현식을 모든 단계에 걸쳐 추적해 보겠습니다.

### 소스 코드

```c
int a, b, c;
b = 3;
c = 5;
a = b + c * 2;
```

### 1단계: 어휘 분석

```
Token Stream:
  [KW_INT:"int"] [ID:"a"] [COMMA:","] [ID:"b"] [COMMA:","] [ID:"c"] [SEMI:";"]
  [ID:"b"] [ASSIGN:"="] [INT:"3"] [SEMI:";"]
  [ID:"c"] [ASSIGN:"="] [INT:"5"] [SEMI:";"]
  [ID:"a"] [ASSIGN:"="] [ID:"b"] [PLUS:"+"] [ID:"c"] [STAR:"*"] [INT:"2"] [SEMI:";"]
```

### 2단계: 구문 분석 (AST)

```
Program
 |- VarDecl(type=int, names=[a, b, c])
 |- Assign(target=b, value=IntLit(3))
 |- Assign(target=c, value=IntLit(5))
 |- Assign(target=a,
           value=BinOp(+,
                        left=Var(b),
                        right=BinOp(*,
                                     left=Var(c),
                                     right=IntLit(2))))
```

### 3단계: 의미 분석

```
Symbol Table:
  a : int (declared, scope=global)
  b : int (declared, scope=global)
  c : int (declared, scope=global)

Type checking:
  b + c * 2  =>  int + (int * int)  =>  int + int  =>  int  ✓
  a = (int)  =>  int = int  ✓
```

### 4단계: IR 생성 (3-주소 코드)

```
    b = 3
    c = 5
    t1 = c * 2
    t2 = b + t1
    a = t2
```

### 5단계: 최적화

```
Constant propagation: b=3, c=5 are known constants

    t1 = 5 * 2     (c replaced with 5)
    t2 = 3 + t1    (b replaced with 3)
    a = t2

Constant folding:

    t1 = 10         (5 * 2 computed at compile time)
    t2 = 13         (3 + 10 computed at compile time)
    a = t2

Copy propagation + dead code elimination:

    a = 13
```

전체 계산이 컴파일 시점에 해결되었습니다.

### 6단계: 코드 생성

```asm
; x86-64
    mov DWORD PTR [a], 13
```

### 모든 단계의 Python 시뮬레이션

```python
"""
A simplified demonstration of compiler phases.
This processes the expression: a = b + c * 2
"""

# ============================================================
# Phase 1: Lexical Analysis
# ============================================================

import re
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class Token:
    type: str
    value: str
    line: int = 0
    col: int = 0

def tokenize(source: str) -> List[Token]:
    """Simple lexer for a tiny language."""
    token_spec = [
        ('INT',     r'\d+'),
        ('ID',      r'[a-zA-Z_]\w*'),
        ('ASSIGN',  r'='),
        ('PLUS',    r'\+'),
        ('STAR',    r'\*'),
        ('SEMI',    r';'),
        ('COMMA',   r','),
        ('SKIP',    r'[ \t]+'),
        ('NEWLINE', r'\n'),
        ('MISMATCH', r'.'),
    ]
    keywords = {'int'}
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    tokens = []
    for mo in re.finditer(tok_regex, source):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'SKIP' or kind == 'NEWLINE':
            continue
        if kind == 'MISMATCH':
            raise SyntaxError(f'Unexpected character: {value!r}')
        if kind == 'ID' and value in keywords:
            kind = f'KW_{value.upper()}'
        tokens.append(Token(kind, value))
    return tokens

source = """int a, b, c;
b = 3;
c = 5;
a = b + c * 2;"""

tokens = tokenize(source)
print("=== Phase 1: Lexical Analysis ===")
for tok in tokens:
    print(f"  {tok.type:10s} {tok.value!r}")


# ============================================================
# Phase 2: Syntax Analysis (Build AST)
# ============================================================

@dataclass
class ASTNode:
    """Base class for AST nodes."""
    pass

@dataclass
class IntLiteral(ASTNode):
    value: int

@dataclass
class Identifier(ASTNode):
    name: str
    resolved_type: Optional[str] = None

@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    resolved_type: Optional[str] = None

@dataclass
class Assignment(ASTNode):
    target: str
    value: ASTNode

@dataclass
class VarDecl(ASTNode):
    type_name: str
    names: List[str]

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

class Parser:
    """Recursive descent parser for our tiny language."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str = None) -> Token:
        tok = self.tokens[self.pos]
        if expected_type and tok.type != expected_type:
            raise SyntaxError(
                f"Expected {expected_type}, got {tok.type} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def parse_program(self) -> Program:
        stmts = []
        while self.pos < len(self.tokens):
            stmts.append(self.parse_statement())
        return Program(stmts)

    def parse_statement(self):
        tok = self.peek()
        if tok.type == 'KW_INT':
            return self.parse_var_decl()
        else:
            return self.parse_assignment()

    def parse_var_decl(self) -> VarDecl:
        self.consume('KW_INT')
        names = [self.consume('ID').value]
        while self.peek() and self.peek().type == 'COMMA':
            self.consume('COMMA')
            names.append(self.consume('ID').value)
        self.consume('SEMI')
        return VarDecl('int', names)

    def parse_assignment(self) -> Assignment:
        name = self.consume('ID').value
        self.consume('ASSIGN')
        expr = self.parse_expr()
        self.consume('SEMI')
        return Assignment(name, expr)

    def parse_expr(self) -> ASTNode:
        """expr = term (('+') term)*"""
        left = self.parse_term()
        while self.peek() and self.peek().type == 'PLUS':
            self.consume('PLUS')
            right = self.parse_term()
            left = BinOp('+', left, right)
        return left

    def parse_term(self) -> ASTNode:
        """term = factor (('*') factor)*"""
        left = self.parse_factor()
        while self.peek() and self.peek().type == 'STAR':
            self.consume('STAR')
            right = self.parse_factor()
            left = BinOp('*', left, right)
        return left

    def parse_factor(self) -> ASTNode:
        """factor = INT | ID"""
        tok = self.peek()
        if tok.type == 'INT':
            self.consume()
            return IntLiteral(int(tok.value))
        elif tok.type == 'ID':
            self.consume()
            return Identifier(tok.value)
        else:
            raise SyntaxError(f"Unexpected token: {tok}")

parser = Parser(tokens)
ast = parser.parse_program()

def print_ast(node, indent=0):
    prefix = "  " * indent
    if isinstance(node, Program):
        print(f"{prefix}Program")
        for s in node.statements:
            print_ast(s, indent + 1)
    elif isinstance(node, VarDecl):
        print(f"{prefix}VarDecl(type={node.type_name}, names={node.names})")
    elif isinstance(node, Assignment):
        print(f"{prefix}Assign(target={node.target})")
        print_ast(node.value, indent + 1)
    elif isinstance(node, BinOp):
        print(f"{prefix}BinOp({node.op})")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)
    elif isinstance(node, IntLiteral):
        print(f"{prefix}IntLit({node.value})")
    elif isinstance(node, Identifier):
        print(f"{prefix}Id({node.name})")

print("\n=== Phase 2: Syntax Analysis (AST) ===")
print_ast(ast)


# ============================================================
# Phase 3: Semantic Analysis
# ============================================================

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}  # name -> type

    def analyze(self, node: ASTNode):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.analyze(stmt)
        elif isinstance(node, VarDecl):
            for name in node.names:
                if name in self.symbol_table:
                    raise NameError(f"Variable '{name}' already declared")
                self.symbol_table[name] = node.type_name
        elif isinstance(node, Assignment):
            if node.target not in self.symbol_table:
                raise NameError(f"Variable '{node.target}' not declared")
            expr_type = self.check_type(node.value)
            target_type = self.symbol_table[node.target]
            if expr_type != target_type:
                raise TypeError(
                    f"Cannot assign {expr_type} to {target_type} variable '{node.target}'"
                )
        return node

    def check_type(self, node: ASTNode) -> str:
        if isinstance(node, IntLiteral):
            return 'int'
        elif isinstance(node, Identifier):
            if node.name not in self.symbol_table:
                raise NameError(f"Undeclared variable '{node.name}'")
            node.resolved_type = self.symbol_table[node.name]
            return node.resolved_type
        elif isinstance(node, BinOp):
            left_type = self.check_type(node.left)
            right_type = self.check_type(node.right)
            if left_type != right_type:
                raise TypeError(
                    f"Type mismatch: {left_type} {node.op} {right_type}"
                )
            node.resolved_type = left_type
            return left_type
        raise TypeError(f"Unknown node type: {type(node)}")

analyzer = SemanticAnalyzer()
analyzer.analyze(ast)

print("\n=== Phase 3: Semantic Analysis ===")
print("Symbol table:", analyzer.symbol_table)
print("Type checking passed!")


# ============================================================
# Phase 4: IR Generation (Three-Address Code)
# ============================================================

class IRGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_count = 0

    def new_temp(self) -> str:
        self.temp_count += 1
        return f"t{self.temp_count}"

    def generate(self, node: ASTNode):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.generate(stmt)
        elif isinstance(node, VarDecl):
            pass  # Declarations don't generate code
        elif isinstance(node, Assignment):
            result = self.gen_expr(node.value)
            self.instructions.append(f"{node.target} = {result}")

    def gen_expr(self, node: ASTNode) -> str:
        if isinstance(node, IntLiteral):
            return str(node.value)
        elif isinstance(node, Identifier):
            return node.name
        elif isinstance(node, BinOp):
            left = self.gen_expr(node.left)
            right = self.gen_expr(node.right)
            temp = self.new_temp()
            self.instructions.append(f"{temp} = {left} {node.op} {right}")
            return temp
        raise ValueError(f"Unknown node: {type(node)}")

ir_gen = IRGenerator()
ir_gen.generate(ast)

print("\n=== Phase 4: IR Generation ===")
for instr in ir_gen.instructions:
    print(f"  {instr}")


# ============================================================
# Phase 5: Optimization (Constant Folding + Propagation)
# ============================================================

def optimize(instructions: List[str]) -> List[str]:
    """Simple constant folding and propagation optimizer."""
    constants = {}  # variable -> known constant value
    optimized = []

    for instr in instructions:
        parts = instr.split(' = ', 1)
        target = parts[0].strip()
        expr = parts[1].strip()

        # Substitute known constants
        for var, val in constants.items():
            # Replace variable with its constant value (whole word only)
            import re as _re
            expr = _re.sub(r'\b' + _re.escape(var) + r'\b', str(val), expr)

        # Try to evaluate the expression (constant folding)
        try:
            # Only evaluate if it looks like a pure arithmetic expression
            result = eval(expr)
            if isinstance(result, (int, float)):
                constants[target] = result
                optimized.append(f"{target} = {result}")
                continue
        except:
            pass

        optimized.append(f"{target} = {expr}")

    # Remove assignments to temporaries that are only used once
    # (simplified dead code elimination)
    final = []
    used_vars = set()
    for instr in reversed(optimized):
        parts = instr.split(' = ', 1)
        target = parts[0].strip()
        if target.startswith('t') and target not in used_vars:
            continue  # Dead temporary
        final.append(instr)
        # Track variable uses in the expression
        for word in re.findall(r'\b[a-zA-Z_]\w*\b', parts[1]):
            used_vars.add(word)
    final.reverse()

    return final

optimized_ir = optimize(ir_gen.instructions)

print("\n=== Phase 5: Optimization ===")
for instr in optimized_ir:
    print(f"  {instr}")


# ============================================================
# Phase 6: Code Generation (Simple Stack Machine)
# ============================================================

def generate_code(instructions: List[str]) -> List[str]:
    """Generate simple stack-machine instructions."""
    code = []
    for instr in instructions:
        parts = instr.split(' = ', 1)
        target = parts[0].strip()
        expr = parts[1].strip()

        # Try to parse as "left op right"
        match = re.match(r'(\w+)\s*([+\-*/])\s*(\w+)', expr)
        if match:
            left, op, right = match.groups()
            code.append(f"  LOAD  {left}")
            code.append(f"  LOAD  {right}")
            op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}
            code.append(f"  {op_map[op]}")
            code.append(f"  STORE {target}")
        else:
            # Simple assignment: target = value
            code.append(f"  LOAD  {expr}")
            code.append(f"  STORE {target}")
    return code

target_code = generate_code(optimized_ir)

print("\n=== Phase 6: Code Generation (Stack Machine) ===")
for line in target_code:
    print(line)
```

실행하면 다음이 출력됩니다:

```
=== Phase 1: Lexical Analysis ===
  KW_INT     'int'
  ID         'a'
  COMMA      ','
  ID         'b'
  ...

=== Phase 2: Syntax Analysis (AST) ===
Program
  VarDecl(type=int, names=['a', 'b', 'c'])
  Assign(target=b)
    IntLit(3)
  Assign(target=c)
    IntLit(5)
  Assign(target=a)
    BinOp(+)
      Id(b)
      BinOp(*)
        Id(c)
        IntLit(2)

=== Phase 3: Semantic Analysis ===
Symbol table: {'a': 'int', 'b': 'int', 'c': 'int'}
Type checking passed!

=== Phase 4: IR Generation ===
  b = 3
  c = 5
  t1 = c * 2
  t2 = b + t1
  a = t2

=== Phase 5: Optimization ===
  a = 13

=== Phase 6: Code Generation (Stack Machine) ===
  LOAD  13
  STORE a
```

최적화기가 전체 계산을 단일 상수 할당으로 줄였습니다.

---

## 6. 프론트엔드(Front-End) vs. 백엔드(Back-End)

컴파일러는 논리적으로 두 부분으로 나뉩니다:

```
                    Front-End                    Back-End
              (language-dependent)          (machine-dependent)
         +---------------------------+  +---------------------------+
Source -->| Lexer | Parser | Semantic |->| Optimizer | Code Gen     |--> Target
         | Analysis       | Analysis |  |           | Register Alloc|
         +---------------------------+  +---------------------------+
                     |                              |
                     v                              v
               Intermediate                    Target Code
              Representation (IR)
```

### 프론트엔드(Front-End)

프론트엔드는 **소스 언어 의존적**이고 **목적 기계 독립적**입니다:

- 어휘 분석
- 구문 분석
- 의미 분석
- IR 생성

프론트엔드는 특정 목적 기계에 구속되지 않고 프로그램의 의미를 포착하는 중간 표현을 생성합니다.

### 백엔드(Back-End)

백엔드는 **소스 언어 독립적**이고 **목적 기계 의존적**입니다:

- 기계 독립적 최적화
- 기계 의존적 최적화
- 명령어 선택
- 레지스터 할당
- 명령어 스케줄링

### $M \times N$ 문제

IR 없이 $M$개의 소스 언어와 $N$개의 목적 기계를 지원하려면 $M \times N$개의 컴파일러가 필요합니다. 공유 IR을 사용하면 $M$개의 프론트엔드와 $N$개의 백엔드만 필요합니다:

```
Without IR:  M x N compilers needed

  C    ---->  x86
  C    ---->  ARM
  C    ---->  MIPS
  Java ---->  x86
  Java ---->  ARM
  Java ---->  MIPS
  Rust ---->  x86
  Rust ---->  ARM
  Rust ---->  MIPS

  3 languages x 3 targets = 9 compilers


With IR:  M + N components needed

  C    ---->  \
  Java ---->   ----> IR ----> x86
  Rust ---->  /          ---> ARM
                         ---> MIPS

  3 front-ends + 3 back-ends = 6 components
```

이것이 바로 **LLVM**의 설계 철학입니다: 많은 프론트엔드(C/C++용 Clang, Rust용 rustc, Swift용 swiftc)가 공통 IR(LLVM IR)과 공통 백엔드를 공유합니다.

---

## 7. 단일 패스(Single-Pass) vs. 다중 패스(Multi-Pass) 컴파일

### 단일 패스 컴파일러

**단일 패스 컴파일러(single-pass compiler)**는 소스 코드를 정확히 한 번만 처리하며, 왼쪽에서 오른쪽으로 읽으면서 모든 단계를 동시에 수행합니다.

**장점**:
- 빠른 컴파일 (소스에 대해 단 한 번의 패스)
- 낮은 메모리 사용량
- 단순한 구현

**단점**:
- 제한된 최적화 기회
- 언어가 단일 패스 컴파일을 위해 설계되어야 함 (예: C/Pascal의 전방 선언)
- 전역 분석 불가능

**예시**: 초기 Pascal 컴파일러는 단일 패스였습니다. 이것이 Pascal에서 전방 선언(forward declaration)이 필요한 이유입니다:

```pascal
{ Pascal: must declare before use }
procedure B; forward;  { forward declaration needed }

procedure A;
begin
    B;  { calls B, which hasn't been fully defined yet }
end;

procedure B;
begin
    A;
end;
```

### 다중 패스 컴파일러

**다중 패스 컴파일러(multi-pass compiler)**는 소스 코드를 여러 번 처리하며, 각 패스가 특정 변환을 수행합니다.

**장점**:
- 더 나은 최적화 (프로그램의 전역적 관점)
- 더 깔끔한 관심사 분리
- 전방 선언 불필요

**단점**:
- 느린 컴파일
- 높은 메모리 사용량 (중간 표현을 저장해야 함)

**예시**: GCC와 Clang 같은 현대 컴파일러는 많은 패스를 사용합니다:

```
GCC compilation passes (simplified):
  Pass 1: Preprocessing (cpp)
  Pass 2: Parsing -> GENERIC IR
  Pass 3: GENERIC -> GIMPLE (lowering)
  Pass 4: GIMPLE optimizations (SSA, constant prop, DCE, ...)
  Pass 5: GIMPLE -> RTL (Register Transfer Language)
  Pass 6: RTL optimizations (CSE, loop opt, register alloc)
  Pass 7: RTL -> Assembly
```

---

## 8. 부트스트래핑(Bootstrapping)

**부트스트래핑(bootstrapping)**은 언어 자체를 사용하여 그 언어용 컴파일러를 작성하는 과정입니다. 이는 닭이 먼저냐 달걀이 먼저냐 문제를 만들어냅니다: 컴파일러가 없는데 어떻게 컴파일러를 컴파일할 수 있을까요?

### 부트스트랩 과정

```
Step 1: Write a simple compiler C0 for language L in an existing language E
        (e.g., write a C compiler in assembly)

        C0 written in E,  compiles subset of L

Step 2: Use C0 to compile a better compiler C1 written in L
        (e.g., write a C compiler in C, compile it with C0)

        C1 written in L,  compiled by C0

Step 3: Use C1 to recompile itself
        (the compiler compiles its own source code)

        C1 written in L,  compiled by C1

Step 4: Iterate to improve the compiler
        C2, C3, ... each version compiled by the previous
```

### 역사적 예시: 최초의 C 컴파일러

Ken Thompson과 Dennis Ritchie는 다음과 같이 C를 부트스트래핑했습니다:

1. PDP-11 어셈블리로 최소한의 C 컴파일러 작성 (실제로 B 언어가 C보다 먼저였습니다)
2. C로 더 나은 C 컴파일러 작성
3. 어셈블리 버전으로 C 컴파일러 컴파일
4. C 컴파일러 자체로 재컴파일

### 신뢰 문제 (Thompson의 해킹)

Ken Thompson은 1984년 튜링상 강연 "Reflections on Trusting Trust"에서 부트스트래핑된 컴파일러가 숨겨진 백도어를 포함할 수 있음을 증명했습니다:

1. 특정 프로그램(예: `login`)을 컴파일할 때 백도어를 삽입하도록 컴파일러 수정
2. *자기 자신*을 컴파일할 때 수정 (1)을 삽입하도록 컴파일러 수정
3. 소스 코드에서 수정 사항 제거
4. 컴파일된 컴파일러 바이너리는 소스가 깨끗하더라도 두 수정 사항을 모두 포함

이것이 **재현 가능한 빌드(reproducible builds)**와 **다중 컴파일러를 이용한 이중 컴파일(diverse double-compilation)**이 보안에 중요한 이유입니다.

---

## 9. 크로스 컴파일(Cross-Compilation)

**크로스 컴파일러(cross-compiler)**는 한 플랫폼(**호스트(host)**)에서 실행되지만 다른 플랫폼(**목적(target)**)을 위한 코드를 생성합니다.

$$\text{Cross-compiler}: \text{Source} \xrightarrow{\text{runs on Host}} \text{Target code for Target platform}$$

### 용어

세 가지 용어를 사용합니다:
- **빌드(Build)**: 컴파일러가 빌드되는 플랫폼
- **호스트(Host)**: 컴파일러가 실행되는 플랫폼
- **목적(Target)**: 컴파일러가 코드를 생성하는 플랫폼

| Build = Host = Target | 명칭 |
|-----------------------|------|
| 모두 동일 | 네이티브 컴파일러(Native compiler) |
| Build = Host, 다른 Target | 크로스 컴파일러(Cross-compiler) |
| 모두 다름 | 캐나다 크로스(Canadian cross) |

### 사용 사례

- **임베디드 시스템**: x86 PC에서 개발, ARM 마이크로컨트롤러에 배포
- **모바일 개발**: macOS에서 컴파일, iOS/Android에서 실행
- **운영 체제**: 컴파일러가 없는 플랫폼을 위한 커널 컴파일

```bash
# Example: Cross-compiling C for ARM on an x86 Linux host
arm-linux-gnueabihf-gcc -o hello hello.c

# The resulting 'hello' binary runs on ARM, not on the x86 host
file hello
# hello: ELF 32-bit LSB executable, ARM, EABI5, ...
```

---

## 10. T-다이어그램(T-Diagrams, 묘비 다이어그램)

**T-다이어그램(T-diagrams)**은 컴파일러, 인터프리터, 프로그램을 소스 언어, 목적 언어, 구현 언어의 관점에서 기술하는 시각적 표기법입니다.

### 표기법

**컴파일러**는 T 모양으로 그립니다:

```
+-------------------+
|  Source  | Target  |
+----+-----+----+---+
     | Impl     |
     +----------+
```

- **Source**: 컴파일러가 받아들이는 언어
- **Target**: 컴파일러가 생성하는 언어
- **Impl**: 컴파일러가 작성된 언어

**프로그램**은 역사다리꼴로 그립니다:

```
+---------+
| Program |
+----+----+
     | Lang
     +----+
```

**인터프리터**는 둥근 T 모양으로 그립니다:

```
+-------------------+
|  Language          |
+----+---------+----+
     | Impl    |
     +---------+
```

### 예시: T-다이어그램으로 부트스트래핑

**1단계**: x86을 목적으로 하는 C 컴파일러를 어셈블리로 작성합니다.

```
+-------------------+
|    C     |  x86   |
+----+-----+----+---+
     |   ASM    |
     +----------+
```

**2단계**: x86을 목적으로 하는 C 컴파일러를 C로 작성합니다.

```
+-------------------+
|    C     |  x86   |
+----+-----+----+---+
     |    C      |
     +----------+
```

**3단계**: 어셈블리 컴파일러(1단계)를 사용하여 C 컴파일러(2단계)를 컴파일합니다. 결과는 x86 기계 코드로 작성된 C 컴파일러입니다.

```
+-------------------+       +-------------------+       +-------------------+
|    C     |  x86   |       |    C     |  x86   |       |    C     |  x86   |
+----+-----+----+---+  ==>  +----+-----+----+---+  ==>  +----+-----+----+---+
     |    C      |                |   ASM    |                |   x86    |
     +----+------+                +----------+                +----------+
          |
    compiled by
          |
     +----+-----+----+
     |    C     | x86 |
     +----+-----+----+
          | ASM  |
          +------+
```

### T-다이어그램의 Python 표현

```python
from dataclasses import dataclass

@dataclass
class Compiler:
    """Represents a compiler as a T-diagram."""
    source: str      # Source language
    target: str      # Target language
    impl: str        # Implementation language

    def __repr__(self):
        return f"Compiler({self.source} -> {self.target}, written in {self.impl})"

@dataclass
class Program:
    """Represents a program."""
    name: str
    language: str

    def __repr__(self):
        return f"Program({self.name}, in {self.language})"

def compile_with(compiler: Compiler, artifact):
    """
    Simulate compilation:
    - If artifact is a Program in compiler.source, produce Program in compiler.target
    - If artifact is a Compiler in compiler.source, produce Compiler with impl=compiler.target
    """
    if isinstance(artifact, Program):
        if artifact.language != compiler.source:
            raise ValueError(
                f"Cannot compile {artifact.language} program "
                f"with {compiler.source} compiler"
            )
        return Program(artifact.name, compiler.target)

    elif isinstance(artifact, Compiler):
        if artifact.impl != compiler.source:
            raise ValueError(
                f"Cannot compile compiler written in {artifact.impl} "
                f"with {compiler.source} compiler"
            )
        return Compiler(artifact.source, artifact.target, compiler.target)

# Bootstrapping example
print("=== Bootstrapping a C Compiler ===\n")

# Step 1: We have a C compiler written in ASM that targets x86
c_compiler_asm = Compiler("C", "x86", "ASM")
print(f"Step 1 (existing):  {c_compiler_asm}")

# Step 2: We write a C compiler in C that targets x86
c_compiler_in_c = Compiler("C", "x86", "C")
print(f"Step 2 (source):    {c_compiler_in_c}")

# Step 3: Compile the C compiler with the ASM compiler
c_compiler_native = compile_with(c_compiler_asm, c_compiler_in_c)
print(f"Step 3 (compiled):  {c_compiler_native}")

# Step 4: The native compiler can now compile itself
c_compiler_self = compile_with(c_compiler_native, c_compiler_in_c)
print(f"Step 4 (self-comp): {c_compiler_self}")
```

출력:

```
=== Bootstrapping a C Compiler ===

Step 1 (existing):  Compiler(C -> x86, written in ASM)
Step 2 (source):    Compiler(C -> x86, written in C)
Step 3 (compiled):  Compiler(C -> x86, written in x86)
Step 4 (self-comp): Compiler(C -> x86, written in x86)
```

---

## 11. 컴파일러 구성 도구

수십 년에 걸쳐 컴파일러 구성의 일부를 자동화하는 많은 도구들이 개발되었습니다:

### 렉서 생성기(Lexer Generators)

| 도구 | 입력 | 출력 | 비고 |
|------|-------|--------|-------|
| Lex | 정규 표현식 | C 스캐너 | 원래 Unix 도구 (1975) |
| Flex | 정규 표현식 | C 스캐너 | Fast Lex (GNU 대체품) |
| re2c | 정규 표현식 | C/C++ 스캐너 | 직접 코드 생성, 테이블 없음 |
| ANTLR | 문법 | Java/Python/... 스캐너+파서 | 렉서+파서 통합 |
| PLY | Python 정규식 | Python 스캐너 | Python Lex-Yacc |

### 파서 생성기(Parser Generators)

| 도구 | 문법 유형 | 출력 | 비고 |
|------|-------------|--------|-------|
| Yacc | LALR(1) | C 파서 | 원래 Unix 도구 (1975) |
| Bison | LALR(1)/GLR | C/C++/Java 파서 | Yacc의 GNU 대체품 |
| ANTLR | LL(*) | 다중 언어 파서 | 인기 많고 사용자 친화적 |
| Lark | Earley/LALR | Python 파서 | 현대 Python 도구 |
| tree-sitter | LR(1)/GLR | C 파서 + 바인딩 | 편집기용 증분 파싱 |

### 컴파일러 프레임워크(Compiler Frameworks)

| 프레임워크 | 유형 | 비고 |
|-----------|------|-------|
| LLVM | 컴파일러 인프라 | 모듈형, 광범위하게 사용됨 |
| GCC | 컴파일러 컬렉션 | 성숙하고 많은 목적 플랫폼 지원 |
| Cranelift | 코드 생성기 | Wasmtime에서 사용, 빠른 컴파일 |
| QBE | 컴파일러 백엔드 | LLVM의 경량 대안 |
| MLIR | IR 프레임워크 | 다중 수준 IR (LLVM 프로젝트의 일부) |

---

## 12. 컴파일러를 공부하는 이유

프로덕션 컴파일러를 만들지 않더라도 컴파일러 기법은 소프트웨어 공학 전반에 등장합니다:

1. **설정 파일 파서**: YAML, TOML, JSON, INI 파서는 렉서+파서 기법 사용
2. **쿼리 언어**: SQL, GraphQL, Elasticsearch 쿼리는 컴파일/인터프리트됨
3. **템플릿 엔진**: Jinja2, Handlebars, JSX 모두 파싱과 코드 생성 포함
4. **빌드 시스템**: Make, CMake, Bazel은 자체 DSL을 파싱
5. **정규 표현식**: 모든 정규식 엔진은 유한 오토마타를 구현
6. **IDE**: 구문 강조, 코드 완성, 리팩터링은 컴파일러 프론트엔드를 사용
7. **정적 분석**: 린터와 보안 스캐너는 의미 분석을 수행
8. **코드 생성**: ORM, 프로토콜 버퍼, API 생성기는 명세에서 코드를 생성
9. **최적화**: 컴파일러 최적화를 이해하면 더 빠른 코드를 작성하는 데 도움
10. **형식 검증**: 타입 시스템과 프로그램 분석은 컴파일러 기법을 기반으로 함

---

## 요약

- **컴파일러(compiler)**는 의미론을 보존하면서 소스 코드를 한 언어에서 다른 언어로 변환합니다
- 컴파일은 잘 정의된 **단계(phase)**를 통해 진행됩니다: 어휘 분석, 파싱, 의미 분석, IR 생성, 최적화, 코드 생성
- **프론트엔드**(언어 의존적)와 **백엔드**(기계 의존적)는 **중간 표현(intermediate representation)**으로 분리됩니다
- **부트스트래핑(bootstrapping)**은 컴파일러가 자기 자신을 컴파일할 수 있게 하지만, 신뢰 문제를 야기합니다
- **크로스 컴파일(cross-compilation)**은 컴파일러가 실행되는 플랫폼과 다른 플랫폼을 위한 코드를 생성합니다
- **T-다이어그램(T-diagrams)**은 컴파일러, 프로그램, 그 관계에 대해 추론하는 시각적 표기법을 제공합니다
- GCC와 LLVM/Clang 같은 현대 컴파일러는 여러 중간 표현에 걸쳐 수십 개의 패스를 사용합니다
- 컴파일러 기법은 컴파일러 구성을 넘어 광범위하게 적용됩니다

---

## 연습 문제

### 연습 1: 단계 식별

다음 프로그램에 대해 각 컴파일러 단계가 수행할 작업을 식별하세요. 각 단계의 출력(토큰, AST 구조, 타입 검사 결과, 3-주소 코드, 최적화된 코드)을 작성하세요.

```c
float area;
float radius = 5.0;
area = 3.14159 * radius * radius;
```

### 연습 2: 컴파일러 vs. 인터프리터

다음 각 시나리오에 대해 컴파일러, 인터프리터, 또는 하이브리드 접근법이 가장 적합한지 설명하고 그 이유를 기술하세요:

1. 시스템 관리 작업 자동화를 위한 스크립팅 언어
2. 고성능 게임 엔진 작성을 위한 언어
3. 빌드 규칙 명세를 위한 설정 언어
4. 웹 브라우저에서 실행되는 언어 (샌드박스 환경)

### 연습 3: T-다이어그램

다음 시나리오에 대해 T-다이어그램을 그리세요:

주어진 것:
- x86에서 실행되는 C로 작성된 Python 인터프리터
- x86을 목적으로 하는 C로 작성된 C 컴파일러

다음을 보여주세요:
1. x86 머신에서 Python 프로그램 실행
2. C 컴파일러 자체를 이용한 컴파일
3. x86에서 실행되지만 ARM을 목적으로 하는 크로스 컴파일러 생성

### 연습 4: 부트스트래핑 시퀀스

"Nova"라는 새로운 언어를 만들고 Nova 컴파일러를 Nova 자체로 작성하고 싶다고 가정합니다. 최소 세 단계의 구체적인 부트스트래핑 전략을 기술하세요. 기존 언어로 구현해야 할 최소한의 것은 무엇인가요?

### 연습 5: 프론트엔드 / 백엔드 분리

한 회사가 4개의 소스 언어를 3개의 아키텍처를 목적으로 하는 컴파일러를 가지고 있습니다. 필요한 컴파일러 컴포넌트의 수는:
1. 공유 IR 없이?
2. 공유 IR을 사용하면?

언어를 2개 더, 아키텍처를 2개 더 추가하면 각 경우에 수치가 어떻게 변하나요?

### 연습 6: 실제 컴파일 단계

Python의 `dis` 모듈과 `ast` 모듈을 사용하여 CPython이 Python 코드를 컴파일하는 방법을 살펴보세요:

```python
import ast
import dis

source = "x = [i**2 for i in range(10)]"

# 1. Parse to AST
tree = ast.parse(source)
print(ast.dump(tree, indent=2))

# 2. Compile to bytecode
code = compile(source, "<string>", "exec")

# 3. Disassemble
dis.dis(code)
```

출력을 살펴보고 어떤 컴파일러 단계가 보이는지 식별하세요. CPython이 수행한 최적화가 있다면 무엇인가요?

---

[다음: 어휘 분석(Lexical Analysis)](./02_Lexical_Analysis.md) | [개요](./00_Overview.md)
