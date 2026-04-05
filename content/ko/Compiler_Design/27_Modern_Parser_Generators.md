# 현대 파서 생성기(Modern Parser Generators)

**이전**: [26. 디버그 정보](./26_Debug_Information.md) | **다음**: [28. 캡스톤 컴파일러 프로젝트](./28_Capstone_Compiler_Project.md)

---

Yacc와 Bison 같은 고전적 파서 생성기(parser generator)는 전체 파일을 한 번에 처리하는 배치 파서를 생성합니다. 현대 개발에는 더 많은 것이 요구됩니다: 실시간 편집기 지원을 위한 증분 파싱(incremental parsing), 불완전한 코드에 대한 오류 복구(error recovery), 그리고 리팩토링 도구를 위해 모든 토큰을 보존하는 구체 구문 트리(concrete syntax tree)가 필요합니다. 이 레슨에서는 tree-sitter, PEG 파서, Packrat 파싱, 증분 파싱, 그리고 이를 개발 도구에 연결하는 Language Server Protocol(LSP)을 포함한 현대 파싱 기술을 탐구합니다.

**난이도**: ⭐⭐⭐

**선수 지식**: [05. 하향식 파싱](./05_Top_Down_Parsing.md), [06. 상향식 파싱](./06_Bottom_Up_Parsing.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 파싱 표현식 문법(PEG)과 문맥 자유 문법(CFG)과의 차이를 설명한다
2. 메모이제이션(memoization)이 있는 Packrat 파서를 구현한다
3. tree-sitter의 아키텍처와 증분 파싱 접근 방식을 기술한다
4. 구체 구문 트리(CST)와 추상 구문 트리(AST)를 이해한다
5. Language Server Protocol과 파서가 편집기와 통합되는 방법을 설명한다
6. 다양한 사용 사례에 적합한 파싱 기술을 선택한다

---

## 목차

1. [Yacc를 넘어서: 현대적 요구사항](#1-yacc를-넘어서-현대적-요구사항)
2. [파싱 표현식 문법](#2-파싱-표현식-문법)
3. [Packrat 파싱](#3-packrat-파싱)
4. [Tree-sitter](#4-tree-sitter)
5. [증분 파싱](#5-증분-파싱)
6. [오류 복구](#6-오류-복구)
7. [Language Server Protocol](#7-language-server-protocol)
8. [파싱 기술 선택](#8-파싱-기술-선택)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. Yacc를 넘어서: 현대적 요구사항

### 1.1 편집기가 필요로 하는 것

현대 코드 편집기는 고전적 도구에 없는 파서 기능을 요구합니다:

| 요구사항 | 고전적 (Yacc/Bison) | 현대적 (tree-sitter 등) |
|---------|-------------------|----------------------|
| 재편집 속도 | 전체 파일 재파싱 | 증분 (변경 영역만 재파싱) |
| 오류 복구 | 중단 또는 기본 복구 | 강건: 불완전/깨진 코드 파싱 |
| 출력 형식 | AST (손실) | CST (무손실: 모든 토큰 보존) |
| 동시성 | 단일 스레드 | 재진입 가능, 스레드 안전 |
| 언어 지원 | 파서당 하나의 언어 | 다중 언어, 내장 문법 |

### 1.2 CST vs. AST

```
Source: x = 1 + 2 * 3;

CST (Concrete Syntax Tree - preserves everything):
  assignment_statement
  ├── identifier "x"
  ├── "="
  ├── binary_expression
  │   ├── number "1"
  │   ├── "+"
  │   └── binary_expression
  │       ├── number "2"
  │       ├── "*"
  │       └── number "3"
  └── ";"

AST (Abstract Syntax Tree - semantic only):
  Assign
  ├── Var("x")
  └── Add
      ├── Num(1)
      └── Mul
          ├── Num(2)
          └── Num(3)
```

CST가 필요한 이유:
- 구문 강조(syntax highlighting) (모든 토큰의 위치를 알아야 함)
- 코드 포매팅(code formatting) (공백을 보존하거나 재구성해야 함)
- 리팩토링(refactoring) (코드 구조를 유지해야 함)

---

## 2. 파싱 표현식 문법

### 2.1 PEG vs. CFG

**파싱 표현식 문법(Parsing Expression Grammar)** (Ford, 2004)은 CFG와 비슷해 보이지만, 모호한 대안 대신 순서가 있는 선택지(ordered choice)를 사용합니다:

```
CFG:  Expr → Expr '+' Term | Term        (ambiguous: which alternative?)
PEG:  Expr ← Expr '+' Term / Term        (ordered: try first, backtrack if fails)
```

### 2.2 PEG 연산자

| 연산자 | 의미 | 예제 |
|--------|------|------|
| `'...'` | 리터럴 문자열 | `'if'` |
| `[...]` | 문자 클래스 | `[a-zA-Z]` |
| `.` | 임의의 문자 | `.` |
| `e1 e2` | 시퀀스 | `Expr '+' Term` |
| `e1 / e2` | 순서 선택지 | `IfStmt / WhileStmt` |
| `e*` | 0회 이상 반복 | `Digit*` |
| `e+` | 1회 이상 반복 | `Digit+` |
| `e?` | 선택적 | `'-'?` |
| `&e` | 긍정 전방탐색 | `&'{'` |
| `!e` | 부정 전방탐색 | `!'\n'` |

### 2.3 PEG 예제: 간단한 표현식 언어

```
# PEG grammar for arithmetic expressions
Expr    ← Term (('+' / '-') Term)*
Term    ← Factor (('*' / '/') Factor)*
Factor  ← Number / '(' Expr ')'
Number  ← [0-9]+
Spacing ← [ \t\n]*
```

### 2.4 PEG 구현

```python
class PEGParser:
    """Simple PEG parser with backtracking."""

    def __init__(self, text):
        self.text = text
        self.pos = 0

    def literal(self, expected):
        """Match a literal string."""
        if self.text[self.pos:self.pos + len(expected)] == expected:
            self.pos += len(expected)
            return expected
        return None

    def char_class(self, chars):
        """Match one character from a set."""
        if self.pos < len(self.text) and self.text[self.pos] in chars:
            c = self.text[self.pos]
            self.pos += 1
            return c
        return None

    def ordered_choice(self, *alternatives):
        """Try alternatives in order; backtrack on failure."""
        save = self.pos
        for alt in alternatives:
            result = alt()
            if result is not None:
                return result
            self.pos = save  # Backtrack
        return None

    def zero_or_more(self, parser_func):
        """Match zero or more repetitions."""
        results = []
        while True:
            save = self.pos
            result = parser_func()
            if result is None:
                self.pos = save
                break
            results.append(result)
        return results

    def sequence(self, *parsers):
        """Match a sequence of parsers."""
        save = self.pos
        results = []
        for p in parsers:
            result = p()
            if result is None:
                self.pos = save
                return None
            results.append(result)
        return results
```

### 2.5 PEG의 특성

- **모호하지 않음(Unambiguous)**: 순서 선택지는 정확히 하나의 파스를 의미합니다
- **무한 전방탐색(Unlimited lookahead)**: 임의로 먼 곳까지 전방탐색 가능
- **직접 좌재귀 불가(No left recursion)**: 재작성이 필요합니다
- **선형 시간**: Packrat 파싱(메모이제이션)으로 가능

---

## 3. Packrat 파싱

### 3.1 백트래킹의 문제

순수 PEG 파싱은 반복적인 백트래킹(backtracking)으로 인해 지수 시간 복잡도를 가질 수 있습니다:

```
# Pathological case:
A ← a A / a
# Parsing "aaaa" tries A → a A, A → a A, A → a A, ... then backtracks
```

### 3.2 Packrat 해결책: 메모이제이션

**Packrat 파싱** (Ford, 2002)은 모든 파싱 함수를 모든 입력 위치에서 메모이제이션합니다:

```python
class PackratParser:
    """PEG parser with memoization for O(n) parsing."""

    def __init__(self, text):
        self.text = text
        self.memo = {}  # (rule_name, position) -> (result, new_position)

    def memoize(self, rule_name, parser_func):
        """Wrap a parser function with memoization."""
        def memoized():
            key = (rule_name, self.pos)
            if key in self.memo:
                result, new_pos = self.memo[key]
                self.pos = new_pos
                return result

            result = parser_func()
            self.memo[key] = (result, self.pos)
            return result

        return memoized

    def parse_expr(self):
        """Expr ← Term (('+' / '-') Term)*"""
        key = ("expr", self.pos)
        if key in self.memo:
            result, self.pos = self.memo[key]
            return result

        left = self.parse_term()
        if left is None:
            self.memo[key] = (None, self.pos)
            return None

        while True:
            save = self.pos
            op_result = self.ordered_choice(
                lambda: self.literal('+'),
                lambda: self.literal('-')
            )
            if op_result is None:
                self.pos = save
                break
            right = self.parse_term()
            if right is None:
                self.pos = save
                break
            left = BinOp(op_result, left, right)

        self.memo[key] = (left, self.pos)
        return left
```

### 3.3 공간-시간 트레이드오프

| 파서 타입 | 시간 | 공간 |
|----------|------|------|
| 재귀 하강(Recursive descent) | 최악 O(2^n) | O(n) 스택 |
| Packrat (메모이제이션된 PEG) | O(n) 보장 | O(n * G), G = 문법 규칙 수 |
| LALR (Yacc/Bison) | O(n) | O(n) 스택 |

---

## 4. Tree-sitter

### 4.1 Tree-sitter란?

**Tree-sitter**는 코드 편집기를 위해 설계된 파서 생성기이자 증분 파싱 라이브러리입니다. 구체 구문 트리(CST)를 생성하며 다음을 지원합니다:

- 증분 파싱 (변경된 영역만 재파싱)
- 오류 복구 (항상 트리를 생성)
- 다중 언어 지원 (문법이 분리됨)
- 스레드 안전성

### 4.2 문법 정의

Tree-sitter 문법은 JavaScript로 작성됩니다:

```javascript
// grammar.js for a simple language
module.exports = grammar({
  name: 'my_language',

  rules: {
    source_file: $ => repeat($._statement),

    _statement: $ => choice(
      $.assignment,
      $.if_statement,
      $.expression_statement,
    ),

    assignment: $ => seq(
      field('left', $.identifier),
      '=',
      field('right', $.expression),
      ';',
    ),

    if_statement: $ => seq(
      'if',
      '(',
      field('condition', $.expression),
      ')',
      field('body', $.block),
      optional(seq('else', field('else', $.block))),
    ),

    block: $ => seq('{', repeat($._statement), '}'),

    expression: $ => choice(
      $.binary_expression,
      $.number,
      $.identifier,
      seq('(', $.expression, ')'),
    ),

    binary_expression: $ => prec.left(1, seq(
      field('left', $.expression),
      field('operator', choice('+', '-', '*', '/', '==', '<')),
      field('right', $.expression),
    )),

    number: $ => /\d+/,
    identifier: $ => /[a-zA-Z_]\w*/,
    expression_statement: $ => seq($.expression, ';'),
  },
});
```

### 4.3 Python에서 Tree-sitter 사용하기

```python
from tree_sitter import Language, Parser

# Load the language
PY_LANGUAGE = Language('build/my-languages.so', 'python')

parser = Parser()
parser.set_language(PY_LANGUAGE)

# Parse source code
source = b"""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

tree = parser.parse(source)
root = tree.root_node

# Walk the tree
def print_tree(node, indent=0):
    print("  " * indent + f"{node.type} [{node.start_point}..{node.end_point}]")
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(root)
```

### 4.4 Tree-sitter 쿼리 언어

```scheme
;; Find all function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)

;; Find all string literals
(string) @string

;; Find if statements with else
(if_statement
  condition: (_) @condition
  consequence: (_) @then
  alternative: (_) @else)
```

---

## 5. 증분 파싱

### 5.1 아이디어

사용자가 코드를 편집할 때, 구문 트리의 대부분은 변경되지 않습니다. 증분 파싱은 변경되지 않은 부분을 재사용합니다:

```
Before edit:                After edit (insert "x + "):
def foo():                  def foo():
    return 42                   return x + 42
           ^^                          ^^^^^
           changed                     changed

Only re-parse the modified region and its ancestors in the tree.
```

### 5.2 Tree-sitter의 구현 방식

```python
# Initial parse
tree = parser.parse(source)

# Edit the source
new_source = source[:offset] + insertion + source[offset:]

# Tell tree-sitter about the edit
tree.edit(
    start_byte=offset,
    old_end_byte=offset,
    new_end_byte=offset + len(insertion),
    start_point=(line, col),
    old_end_point=(line, col),
    new_end_point=(line, col + len(insertion)),
)

# Incremental re-parse (reuses unchanged subtrees)
new_tree = parser.parse(new_source, tree)

# Find what changed
for range_ in tree.changed_ranges(new_tree):
    print(f"Changed: bytes {range_.start_byte}-{range_.end_byte}")
```

### 5.3 성능

Tree-sitter는 수천 줄의 파일에서 일반적인 편집에 대해 밀리초 이하의 재파싱 시간을 달성합니다. 이는 다음과 같은 이유 때문입니다:

- 재사용된 서브트리는 재파싱되지 않음
- GLR과 유사한 알고리즘이 모호성을 효율적으로 처리
- CST는 불변(immutable)이며 (이전 트리와 새 트리가 구조를 공유)

---

## 6. 오류 복구

### 6.1 오류 복구가 중요한 이유

편집기에서 코드는 거의 항상 구문적으로 불완전하거나 깨져 있습니다:

```python
# User is typing:
def foo():
    if x >        # <-- incomplete! But the editor still needs a parse tree
```

### 6.2 Tree-sitter의 오류 복구

Tree-sitter는 여러 전략을 조합하여 사용합니다:
- **ERROR 노드**: 파싱할 수 없는 토큰을 ERROR 노드로 감쌈
- **MISSING 노드**: 누락된 예상 토큰을 삽입
- **복구 휴리스틱**: 토큰을 건너뛰어 유효한 계속 지점을 찾음

```
Parse of "if (x > ) { y = 1; }":

if_statement
├── "if"
├── "("
├── binary_expression
│   ├── identifier "x"
│   ├── ">"
│   └── MISSING expression    ← inserted by error recovery
├── ")"
├── block
│   ├── "{"
│   ├── assignment ...
│   └── "}"
```

### 6.3 재귀 하강에서의 오류 복구

```python
def parse_statement(self):
    """Parse a statement with error recovery."""
    try:
        if self.current_token.type == 'IF':
            return self.parse_if_statement()
        elif self.current_token.type == 'WHILE':
            return self.parse_while_statement()
        elif self.current_token.type == 'IDENTIFIER':
            return self.parse_assignment()
        else:
            raise ParseError(f"Unexpected token: {self.current_token}")
    except ParseError as e:
        # Error recovery: skip to next synchronization point
        self.report_error(e)
        self.synchronize(follow_set={'SEMICOLON', 'RBRACE', 'EOF'})
        return ErrorNode(e.message)

def synchronize(self, follow_set):
    """Skip tokens until we find one in the follow set."""
    while self.current_token.type not in follow_set:
        self.advance()
    if self.current_token.type == 'SEMICOLON':
        self.advance()  # consume the semicolon
```

---

## 7. Language Server Protocol

### 7.1 LSP란?

**Language Server Protocol** (Microsoft, 2016)은 코드 편집기와 언어별 도구 간의 통신을 표준화합니다:

```
Editor (VS Code, Neovim, Emacs)  <--LSP (JSON-RPC)-->  Language Server

Editor sends:                    Server responds:
  textDocument/didOpen             (acknowledges)
  textDocument/completion          completionItems: [...]
  textDocument/hover               hover: {contents: "..."}
  textDocument/definition          location: {file, line, col}
  textDocument/diagnostic          diagnostics: [{message, range}]
```

### 7.2 LSP 아키텍처

```
┌─────────────┐     JSON-RPC      ┌──────────────────┐
│    Editor    │ ◄──(stdio/TCP)──► │  Language Server  │
│             │                    │                  │
│ - UI        │                    │ - Parser         │
│ - Buffer    │                    │ - Type Checker   │
│ - Renderer  │                    │ - Completer      │
│             │                    │ - Diagnostics    │
└─────────────┘                    └──────────────────┘
```

### 7.3 주요 LSP 기능

| 기능 | LSP 메서드 | 파서 요구사항 |
|------|-----------|-------------|
| 구문 강조 | `textDocument/semanticTokens` | 토큰 분류 |
| 오류 보고 | `textDocument/publishDiagnostics` | 전체 파싱 + 타입 검사 |
| 정의로 이동 | `textDocument/definition` | 심볼 해석 |
| 자동 완성 | `textDocument/completion` | 부분 파싱 + 스코프 분석 |
| 호버 정보 | `textDocument/hover` | 타입 추론 |
| 심볼 이름 변경 | `textDocument/rename` | 전체 AST + 참조 |
| 코드 접기 | `textDocument/foldingRange` | 블록 구조 |

### 7.4 간단한 LSP 서버 스케치

```python
import json
import sys

def handle_request(request):
    method = request["method"]

    if method == "initialize":
        return {
            "capabilities": {
                "completionProvider": {},
                "hoverProvider": True,
                "definitionProvider": True,
            }
        }
    elif method == "textDocument/completion":
        doc = request["params"]["textDocument"]["uri"]
        pos = request["params"]["position"]
        return {"items": get_completions(doc, pos)}

    elif method == "textDocument/hover":
        doc = request["params"]["textDocument"]["uri"]
        pos = request["params"]["position"]
        info = get_hover_info(doc, pos)
        return {"contents": info}

def main():
    """LSP server main loop (stdio transport)."""
    while True:
        header = read_header(sys.stdin)
        content_length = parse_content_length(header)
        body = sys.stdin.read(content_length)
        request = json.loads(body)

        result = handle_request(request)

        response = json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result})
        sys.stdout.write(f"Content-Length: {len(response)}\r\n\r\n{response}")
        sys.stdout.flush()
```

---

## 8. 파싱 기술 선택

| 기술 | 최적 용도 | 속도 | 오류 복구 | 증분 |
|------|---------|------|---------|------|
| Yacc/Bison | 배치 컴파일러 | 빠름 (LALR) | 기본 | 아니오 |
| ANTLR | IDE 지원, 도구 개발 | 좋음 (LL(*)) | 좋음 | 아니오 |
| Tree-sitter | 편집기, 구문 강조 | 우수 | 우수 | 예 |
| PEG/Packrat | 간단한 DSL, 프로토타이핑 | 좋음 | 수동 | 아니오 |
| 수동 재귀 하강 | 완전한 제어, 최상의 오류 메시지 | 다양 | 커스텀 | 수동 |
| Nom/Combine | 바이너리 형식, Rust 파서 | 우수 | 좋음 | 아니오 |

---

## 9. 요약

- **PEG**는 순서 선택지로 모호하지 않은 파싱을 제공하며 무한 전방탐색이 가능합니다
- **Packrat 파싱**은 PEG에 메모이제이션을 추가하여 선형 시간을 보장합니다
- **Tree-sitter**는 편집기급 파싱의 현대 표준으로, 증분적이고 오류에 강하며 빠릅니다
- **증분 파싱**은 변경된 영역만 재파싱하여 실시간 편집기 지원을 가능하게 합니다
- **오류 복구**는 깨진 코드에서 부분적 파스 트리를 생성합니다
- **Language Server Protocol**은 파서와 언어 도구를 어떤 편집기에든 연결합니다
- 요구사항에 따라 파싱 기술을 선택하세요: 배치 컴파일, IDE 지원, 또는 편집기 통합

---

## 10. 연습 문제

1. **PEG 파서**: Packrat 메모이제이션이 있는 산술 표현식 PEG 파서를 구현하세요.

2. **Tree-sitter 문법**: 변수, if/else, 루프가 있는 간단한 언어의 tree-sitter 문법을 작성하세요.

3. **증분 편집**: tree-sitter의 Python 바인딩을 사용하여 파일을 파싱하고, 편집을 시뮬레이션하며, 어떤 트리 노드가 변경되었는지 관찰하세요.

4. **오류 복구**: 동기화 기반 오류 복구가 있는 재귀 하강 파서를 구현하세요.

5. **미니 LSP 서버**: 간단한 언어에 대해 호버 정보와 정의로 이동을 제공하는 최소 LSP 서버를 작성하세요.

---

## 11. 참고 자료

1. Ford, B. (2004). "Parsing Expression Grammars: A Recognition-Based Syntactic Foundation." *POPL*.
2. Ford, B. (2002). "Packrat Parsing: Simple, Powerful, Lazy, Linear Time." *ICFP*.
3. Tree-sitter documentation: https://tree-sitter.github.io/tree-sitter/
4. Language Server Protocol specification: https://microsoft.github.io/language-server-protocol/
5. Medeiros, S., Ierusalimschy, R. (2008). "A Parsing Machine for PEGs." *DLS*.

---

**이전**: [26. 디버그 정보](./26_Debug_Information.md) | **다음**: [28. 캡스톤 컴파일러 프로젝트](./28_Capstone_Compiler_Project.md)
