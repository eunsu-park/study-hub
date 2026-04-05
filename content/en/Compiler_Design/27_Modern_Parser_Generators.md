# Modern Parser Generators

**Previous**: [26. Debug Information](./26_Debug_Information.md) | **Next**: [28. Capstone Compiler Project](./28_Capstone_Compiler_Project.md)

---

Classical parser generators like Yacc and Bison produce batch parsers that process entire files at once. Modern development requires more: incremental parsing for real-time editor support, error recovery for incomplete code, and concrete syntax trees that preserve every token for refactoring tools. This lesson explores modern parsing technologies including tree-sitter, PEG parsers, Packrat parsing, incremental parsing, and the Language Server Protocol (LSP) that ties them to developer tools.

**Difficulty**: ⭐⭐⭐

**Prerequisites**: [05. Top-Down Parsing](./05_Top_Down_Parsing.md), [06. Bottom-Up Parsing](./06_Bottom_Up_Parsing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Parsing Expression Grammars (PEGs) and how they differ from CFGs
2. Implement a Packrat parser with memoization
3. Describe tree-sitter's architecture and incremental parsing approach
4. Understand concrete syntax trees (CSTs) vs. abstract syntax trees (ASTs)
5. Explain the Language Server Protocol and how parsers integrate with editors
6. Choose the right parsing technology for different use cases

---

## Table of Contents

1. [Beyond Yacc: Modern Requirements](#1-beyond-yacc-modern-requirements)
2. [Parsing Expression Grammars](#2-parsing-expression-grammars)
3. [Packrat Parsing](#3-packrat-parsing)
4. [Tree-sitter](#4-tree-sitter)
5. [Incremental Parsing](#5-incremental-parsing)
6. [Error Recovery](#6-error-recovery)
7. [Language Server Protocol](#7-language-server-protocol)
8. [Choosing a Parser Technology](#8-choosing-a-parser-technology)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Beyond Yacc: Modern Requirements

### 1.1 What Editors Need

Modern code editors require parsers with capabilities that classical tools lack:

| Requirement | Classical (Yacc/Bison) | Modern (tree-sitter, etc.) |
|-------------|----------------------|---------------------------|
| Speed on re-edit | Reparse entire file | Incremental (reparse changed region) |
| Error recovery | Abort or basic recovery | Robust: parse incomplete/broken code |
| Output format | AST (lossy) | CST (lossless: preserves all tokens) |
| Concurrency | Single-threaded | Reentrant, thread-safe |
| Language support | One language per parser | Multiple languages, embedded grammars |

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

CSTs are needed for:
- Syntax highlighting (need to know where every token is)
- Code formatting (must preserve or reconstruct whitespace)
- Refactoring (must maintain code structure)

---

## 2. Parsing Expression Grammars

### 2.1 PEG vs. CFG

A **Parsing Expression Grammar** (Ford, 2004) looks similar to a CFG but has ordered choice instead of ambiguous alternatives:

```
CFG:  Expr → Expr '+' Term | Term        (ambiguous: which alternative?)
PEG:  Expr ← Expr '+' Term / Term        (ordered: try first, backtrack if fails)
```

### 2.2 PEG Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `'...'` | Literal string | `'if'` |
| `[...]` | Character class | `[a-zA-Z]` |
| `.` | Any character | `.` |
| `e1 e2` | Sequence | `Expr '+' Term` |
| `e1 / e2` | Ordered choice | `IfStmt / WhileStmt` |
| `e*` | Zero or more | `Digit*` |
| `e+` | One or more | `Digit+` |
| `e?` | Optional | `'-'?` |
| `&e` | Positive lookahead | `&'{'` |
| `!e` | Negative lookahead | `!'\n'` |

### 2.3 PEG Example: Simple Expression Language

```
# PEG grammar for arithmetic expressions
Expr    ← Term (('+' / '-') Term)*
Term    ← Factor (('*' / '/') Factor)*
Factor  ← Number / '(' Expr ')'
Number  ← [0-9]+
Spacing ← [ \t\n]*
```

### 2.4 PEG Implementation

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

### 2.5 PEG Properties

- **Unambiguous**: Ordered choice means exactly one parse
- **Unlimited lookahead**: Can look ahead arbitrarily far
- **No left recursion** (directly): must be rewritten
- **Linear time** with Packrat parsing (memoization)

---

## 3. Packrat Parsing

### 3.1 The Problem with Backtracking

Plain PEG parsing can have exponential time complexity due to repeated backtracking:

```
# Pathological case:
A ← a A / a
# Parsing "aaaa" tries A → a A, A → a A, A → a A, ... then backtracks
```

### 3.2 Packrat Solution: Memoization

**Packrat parsing** (Ford, 2002) memoizes every parsing function at every input position:

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

### 3.3 Space-Time Tradeoff

| Parser Type | Time | Space |
|-------------|------|-------|
| Recursive descent | O(2^n) worst case | O(n) stack |
| Packrat (memoized PEG) | O(n) guaranteed | O(n * G) where G = grammar rules |
| LALR (Yacc/Bison) | O(n) | O(n) stack |

---

## 4. Tree-sitter

### 4.1 What is Tree-sitter?

**Tree-sitter** is a parser generator and incremental parsing library designed for code editors. It produces concrete syntax trees (CSTs) and supports:

- Incremental parsing (re-parse only changed regions)
- Error recovery (always produces a tree)
- Multiple language support (grammars are separate)
- Thread safety

### 4.2 Grammar Definition

Tree-sitter grammars are written in JavaScript:

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

### 4.3 Using Tree-sitter from Python

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

### 4.4 Tree-sitter Query Language

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

## 5. Incremental Parsing

### 5.1 The Idea

When a user edits code, most of the syntax tree remains unchanged. Incremental parsing reuses the unchanged parts:

```
Before edit:                After edit (insert "x + "):
def foo():                  def foo():
    return 42                   return x + 42
           ^^                          ^^^^^
           changed                     changed

Only re-parse the modified region and its ancestors in the tree.
```

### 5.2 How Tree-sitter Does It

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

### 5.3 Performance

Tree-sitter achieves sub-millisecond re-parse times for typical edits in files with thousands of lines, because:

- Reused subtrees are not re-parsed
- The GLR-like algorithm handles ambiguity efficiently
- The CST is immutable (old and new trees share structure)

---

## 6. Error Recovery

### 6.1 Why Error Recovery Matters

In an editor, code is almost always syntactically incomplete or broken:

```python
# User is typing:
def foo():
    if x >        # <-- incomplete! But the editor still needs a parse tree
```

### 6.2 Tree-sitter's Error Recovery

Tree-sitter uses a combination of strategies:
- **ERROR nodes**: Wrap unparseable tokens in ERROR nodes
- **MISSING nodes**: Insert expected tokens that are missing
- **Repair heuristics**: Skip tokens to find a valid continuation

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

### 6.3 Error Recovery in Recursive Descent

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

### 7.1 What is LSP?

The **Language Server Protocol** (Microsoft, 2016) standardizes communication between code editors and language-specific tools:

```
Editor (VS Code, Neovim, Emacs)  <--LSP (JSON-RPC)-->  Language Server

Editor sends:                    Server responds:
  textDocument/didOpen             (acknowledges)
  textDocument/completion          completionItems: [...]
  textDocument/hover               hover: {contents: "..."}
  textDocument/definition          location: {file, line, col}
  textDocument/diagnostic          diagnostics: [{message, range}]
```

### 7.2 LSP Architecture

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

### 7.3 Key LSP Features

| Feature | LSP Method | Parser Requirement |
|---------|-----------|-------------------|
| Syntax highlighting | `textDocument/semanticTokens` | Token classification |
| Error reporting | `textDocument/publishDiagnostics` | Full parse + type check |
| Go to definition | `textDocument/definition` | Symbol resolution |
| Auto-completion | `textDocument/completion` | Partial parse + scope analysis |
| Hover info | `textDocument/hover` | Type inference |
| Rename symbol | `textDocument/rename` | Full AST + references |
| Code folding | `textDocument/foldingRange` | Block structure |

### 7.4 Simple LSP Server Sketch

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

## 8. Choosing a Parser Technology

| Technology | Best For | Speed | Error Recovery | Incremental |
|-----------|---------|-------|---------------|-------------|
| Yacc/Bison | Batch compilers | Fast (LALR) | Basic | No |
| ANTLR | IDE support, tool building | Good (LL(*)) | Good | No |
| Tree-sitter | Editors, syntax highlighting | Excellent | Excellent | Yes |
| PEG/Packrat | Simple DSLs, prototyping | Good | Manual | No |
| Hand-written recursive descent | Full control, best errors | Varies | Custom | Manual |
| Nom/Combine | Binary formats, Rust parsers | Excellent | Good | No |

---

## 9. Summary

- **PEGs** provide unambiguous, ordered-choice parsing with unlimited lookahead
- **Packrat parsing** adds memoization to PEGs for guaranteed linear time
- **Tree-sitter** is the modern standard for editor-grade parsing: incremental, error-tolerant, and fast
- **Incremental parsing** re-parses only changed regions, enabling real-time editor support
- **Error recovery** produces partial parse trees from broken code
- The **Language Server Protocol** connects parsers and language tools to any editor
- Choose your parser technology based on requirements: batch compilation, IDE support, or editor integration

---

## 10. Exercises

1. **PEG parser**: Implement a PEG parser for arithmetic expressions with Packrat memoization.

2. **Tree-sitter grammar**: Write a tree-sitter grammar for a simple language with variables, if/else, and loops.

3. **Incremental edit**: Use tree-sitter's Python bindings to parse a file, simulate an edit, and observe which tree nodes changed.

4. **Error recovery**: Implement a recursive descent parser with synchronization-based error recovery.

5. **Mini LSP server**: Write a minimal LSP server that provides hover information and go-to-definition for a simple language.

---

## 11. References

1. Ford, B. (2004). "Parsing Expression Grammars: A Recognition-Based Syntactic Foundation." *POPL*.
2. Ford, B. (2002). "Packrat Parsing: Simple, Powerful, Lazy, Linear Time." *ICFP*.
3. Tree-sitter documentation: https://tree-sitter.github.io/tree-sitter/
4. Language Server Protocol specification: https://microsoft.github.io/language-server-protocol/
5. Medeiros, S., Ierusalimschy, R. (2008). "A Parsing Machine for PEGs." *DLS*.

---

**Previous**: [26. Debug Information](./26_Debug_Information.md) | **Next**: [28. Capstone Compiler Project](./28_Capstone_Compiler_Project.md)
