"""
Exercises for Lesson 02: Lexical Analysis
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

import re
import time
import random
import string
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


# === Exercise 1: Regular Expressions ===
# Problem: Write regular expressions for:
# 1. Binary strings with an even number of 0's
# 2. C-style identifiers that start with an underscore
# 3. Email addresses (simplified)
# 4. Floating-point numbers in scientific notation
# 5. C-style comments: /* ... */ (non-nested)

def exercise_1():
    """Write and test regular expressions for various patterns."""

    # 1. Binary strings with an even number of 0's
    # Strategy: 1*(01*01*)* -- any number of 1's, then pairs of 0's separated by 1's
    pattern1 = r'^1*(01*01*)*$'
    test1 = [
        ("", True),       # zero 0's (even)
        ("1", True),      # zero 0's
        ("11", True),     # zero 0's
        ("00", True),     # two 0's
        ("010", False),   # one 0... wait: 010 has two 0's -> True
        ("0", False),     # one 0 (odd)
        ("100", True),    # two 0's
        ("1001", True),   # two 0's
        ("000", False),   # three 0's
    ]
    # Actually let's be more precise: even number of 0s in {0,1}*
    # The DFA has 2 states: even-0s (accept) and odd-0s.
    # Regex: (1*01*01*)*1*  or equivalently  (1|01*0)*
    pattern1 = r'^(1|01*0)*$'
    test1 = [
        ("", True),
        ("1", True),
        ("11", True),
        ("00", True),
        ("010", False),   # "010" -> one pair: 0_0 with 1 in between? 0,1,0 = two 0's -> True
    ]
    # Let me re-examine: "010" matches (01*0) -> 0, then 1*, then 0
    # (0)(1)(0) -- that's 01*0 with one '1', so it matches. Two 0's is even.
    test1 = [
        ("", True),        # 0 zeros = even
        ("1", True),       # 0 zeros = even
        ("0", False),      # 1 zero = odd
        ("00", True),      # 2 zeros = even
        ("010", True),     # 2 zeros = even
        ("101", True),     # 0 zeros = even
        ("000", False),    # 3 zeros = odd
        ("0000", True),    # 4 zeros = even
        ("10001", False),  # 3 zeros = odd
    ]
    print("1. Binary strings with even number of 0's")
    print(f"   Pattern: {pattern1}")
    for s, expected in test1:
        result = bool(re.match(pattern1, s))
        status = "PASS" if result == expected else "FAIL"
        print(f"   [{status}] '{s}' -> {result} (expected {expected})")
    print()

    # 2. C-style identifiers starting with underscore
    pattern2 = r'^_[a-zA-Z0-9_]*$'
    test2 = [
        ("_foo", True),
        ("_x1", True),
        ("__init__", True),
        ("_", True),
        ("foo", False),
        ("1_x", False),
        ("_hello world", False),
    ]
    print("2. C-style identifiers starting with underscore")
    print(f"   Pattern: {pattern2}")
    for s, expected in test2:
        result = bool(re.match(pattern2, s))
        status = "PASS" if result == expected else "FAIL"
        print(f"   [{status}] '{s}' -> {result} (expected {expected})")
    print()

    # 3. Email addresses (simplified: name@domain.tld)
    pattern3 = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    test3 = [
        ("user@example.com", True),
        ("john.doe@company.org", True),
        ("test+tag@sub.domain.co.uk", True),
        ("@example.com", False),
        ("user@", False),
        ("user@.com", False),
        ("user@domain", False),
    ]
    print("3. Email addresses (simplified)")
    print(f"   Pattern: {pattern3}")
    for s, expected in test3:
        result = bool(re.match(pattern3, s))
        status = "PASS" if result == expected else "FAIL"
        print(f"   [{status}] '{s}' -> {result} (expected {expected})")
    print()

    # 4. Floating-point in scientific notation (e.g., 1.5e-3, -2.0E+10)
    pattern4 = r'^-?[0-9]+\.[0-9]+[eE][+-]?[0-9]+$'
    test4 = [
        ("1.5e-3", True),
        ("-2.0E+10", True),
        ("3.14e0", True),
        ("0.5E-100", True),
        ("1e5", False),     # no decimal point
        ("1.5", False),     # no exponent
        (".5e3", False),    # no leading digit
    ]
    print("4. Floating-point in scientific notation")
    print(f"   Pattern: {pattern4}")
    for s, expected in test4:
        result = bool(re.match(pattern4, s))
        status = "PASS" if result == expected else "FAIL"
        print(f"   [{status}] '{s}' -> {result} (expected {expected})")
    print()

    # 5. C-style comments /* ... */ (non-nested)
    # Must match /* followed by anything except */ followed by */
    pattern5 = r'^/\*([^*]|\*+[^*/])*\*+/$'
    test5 = [
        ("/* hello */", True),
        ("/* multi\nline */", True),
        ("/* ** stars ** */", True),
        ("/**/", True),
        ("/* unclosed", False),
        ("/ * space */", False),
    ]
    print("5. C-style comments /* ... */")
    print(f"   Pattern: {pattern5}")
    for s, expected in test5:
        result = bool(re.match(pattern5, s, re.DOTALL))
        status = "PASS" if result == expected else "FAIL"
        print(f"   [{status}] {s!r} -> {result} (expected {expected})")


# === Exercise 2: Thompson's Construction ===
# Problem: Apply Thompson's construction to a(b|c)*d, simulate on test strings.

class NFAState:
    """State in an NFA."""
    _id_counter = 0

    def __init__(self):
        self.id = NFAState._id_counter
        NFAState._id_counter += 1
        self.transitions = {}  # char -> set of states
        self.epsilon = set()   # epsilon transitions

    def __repr__(self):
        return f"q{self.id}"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, NFAState) and self.id == other.id


class NFA:
    """NFA with a single start state and single accept state."""
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept


def thompson_char(c):
    """Thompson's construction for a single character."""
    start = NFAState()
    accept = NFAState()
    start.transitions[c] = {accept}
    return NFA(start, accept)


def thompson_concat(nfa1, nfa2):
    """Thompson's construction for concatenation."""
    nfa1.accept.epsilon.add(nfa2.start)
    return NFA(nfa1.start, nfa2.accept)


def thompson_union(nfa1, nfa2):
    """Thompson's construction for union (|)."""
    start = NFAState()
    accept = NFAState()
    start.epsilon.add(nfa1.start)
    start.epsilon.add(nfa2.start)
    nfa1.accept.epsilon.add(accept)
    nfa2.accept.epsilon.add(accept)
    return NFA(start, accept)


def thompson_star(nfa):
    """Thompson's construction for Kleene star (*)."""
    start = NFAState()
    accept = NFAState()
    start.epsilon.add(nfa.start)
    start.epsilon.add(accept)
    nfa.accept.epsilon.add(nfa.start)
    nfa.accept.epsilon.add(accept)
    return NFA(start, accept)


def epsilon_closure(states):
    """Compute epsilon-closure of a set of states."""
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for next_state in state.epsilon:
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)
    return frozenset(closure)


def nfa_simulate(nfa, input_str):
    """Simulate an NFA on an input string."""
    current = epsilon_closure({nfa.start})
    for char in input_str:
        next_states = set()
        for state in current:
            if char in state.transitions:
                next_states.update(state.transitions[char])
        current = epsilon_closure(next_states)
    return nfa.accept in current


def exercise_2():
    """Thompson's construction for a(b|c)*d and NFA simulation."""
    NFAState._id_counter = 0

    # Build NFA for a(b|c)*d using Thompson's construction
    nfa_a = thompson_char('a')
    nfa_b = thompson_char('b')
    nfa_c = thompson_char('c')
    nfa_bc = thompson_union(nfa_b, nfa_c)
    nfa_bc_star = thompson_star(nfa_bc)
    nfa_d = thompson_char('d')
    nfa_abc = thompson_concat(nfa_a, nfa_bc_star)
    nfa_full = thompson_concat(nfa_abc, nfa_d)

    print("NFA for a(b|c)*d built using Thompson's construction")
    print(f"  Start state: {nfa_full.start}")
    print(f"  Accept state: {nfa_full.accept}")
    print()

    # Print all states and transitions
    visited = set()
    queue = [nfa_full.start]
    visited.add(nfa_full.start)
    print("  States and transitions:")
    while queue:
        state = queue.pop(0)
        for char, targets in state.transitions.items():
            for t in targets:
                print(f"    {state} --{char}--> {t}")
                if t not in visited:
                    visited.add(t)
                    queue.append(t)
        for t in state.epsilon:
            print(f"    {state} --eps--> {t}")
            if t not in visited:
                visited.add(t)
                queue.append(t)
    print()

    # Simulate on test inputs
    test_cases = [
        ("ad", True),    # a followed by zero (b|c) then d
        ("abcd", True),  # a, then b then c, then d
        ("abbd", True),  # a, then b then b, then d
        ("abc", False),  # no trailing d
    ]

    print("  Simulation results:")
    for input_str, expected in test_cases:
        result = nfa_simulate(nfa_full, input_str)
        status = "PASS" if result == expected else "FAIL"
        print(f"    [{status}] '{input_str}' -> {'Accept' if result else 'Reject'} "
              f"(expected {'Accept' if expected else 'Reject'})")


# === Exercise 3: Subset Construction ===
# Problem: Convert the NFA from Exercise 2 to a DFA using subset construction.

def exercise_3():
    """Subset construction: convert NFA for a(b|c)*d to DFA."""
    NFAState._id_counter = 0

    # Rebuild NFA for a(b|c)*d
    nfa_a = thompson_char('a')
    nfa_b = thompson_char('b')
    nfa_c = thompson_char('c')
    nfa_bc = thompson_union(nfa_b, nfa_c)
    nfa_bc_star = thompson_star(nfa_bc)
    nfa_d = thompson_char('d')
    nfa_abc = thompson_concat(nfa_a, nfa_bc_star)
    nfa_full = thompson_concat(nfa_abc, nfa_d)

    # Collect all input symbols (excluding epsilon)
    alphabet = {'a', 'b', 'c', 'd'}

    # Subset construction
    start_closure = epsilon_closure({nfa_full.start})
    dfa_states = {start_closure: 0}
    dfa_transitions = {}
    dfa_accept = set()
    worklist = [start_closure]
    state_counter = 1

    while worklist:
        current = worklist.pop(0)
        current_id = dfa_states[current]

        if nfa_full.accept in current:
            dfa_accept.add(current_id)

        for symbol in sorted(alphabet):
            next_nfa_states = set()
            for nfa_state in current:
                if symbol in nfa_state.transitions:
                    next_nfa_states.update(nfa_state.transitions[symbol])
            if not next_nfa_states:
                continue
            next_closure = epsilon_closure(next_nfa_states)
            if next_closure not in dfa_states:
                dfa_states[next_closure] = state_counter
                state_counter += 1
                worklist.append(next_closure)
            dfa_transitions[(current_id, symbol)] = dfa_states[next_closure]

    print("Subset Construction: NFA for a(b|c)*d -> DFA")
    print(f"  Number of DFA states: {len(dfa_states)}")
    print(f"  Start state: 0")
    print(f"  Accept states: {dfa_accept}")
    print()

    print("  DFA Transition Table:")
    print(f"  {'State':>6} | {'a':>4} | {'b':>4} | {'c':>4} | {'d':>4}")
    print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}")
    for state_id in range(state_counter):
        row = f"  {'*' if state_id in dfa_accept else ' '}{state_id:>4}  |"
        for symbol in ['a', 'b', 'c', 'd']:
            target = dfa_transitions.get((state_id, symbol), '-')
            row += f" {target:>4} |"
        print(row)
    print()

    # Verify DFA by simulating
    def dfa_simulate(input_str):
        state = 0
        for ch in input_str:
            key = (state, ch)
            if key not in dfa_transitions:
                return False
            state = dfa_transitions[key]
        return state in dfa_accept

    test_cases = [("ad", True), ("abcd", True), ("abbd", True), ("abc", False),
                  ("d", False), ("abd", True), ("acd", True), ("abcbcd", True)]
    print("  DFA Simulation verification:")
    for s, expected in test_cases:
        result = dfa_simulate(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"    [{status}] '{s}' -> {'Accept' if result else 'Reject'}")


# === Exercise 4: Lexer Extension ===
# Problem: Extend a lexer to handle hex, octal, binary literals and numeric separators.

class TokenType(Enum):
    INT = auto()
    HEX_INT = auto()
    OCT_INT = auto()
    BIN_INT = auto()
    FLOAT = auto()
    ID = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    ASSIGN = auto()
    LPAREN = auto()
    RPAREN = auto()
    SEMI = auto()
    EOF = auto()
    ERROR = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int


class ExtendedLexer:
    """Lexer with support for hex, octal, binary literals and numeric separators."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self) -> Optional[str]:
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\n\r':
            self._advance()

    def _read_number(self) -> Token:
        start_col = self.col
        start_pos = self.pos
        raw = ""

        # Check for 0x, 0o, 0b prefixes
        if self._peek() == '0' and self.pos + 1 < len(self.source):
            next_ch = self.source[self.pos + 1]
            if next_ch in 'xX':
                raw += self._advance()  # '0'
                raw += self._advance()  # 'x'
                while self._peek() and (self._peek() in '0123456789abcdefABCDEF_'):
                    raw += self._advance()
                clean = raw.replace('_', '')
                return Token(TokenType.HEX_INT, clean, self.line, start_col)
            elif next_ch in 'oO':
                raw += self._advance()  # '0'
                raw += self._advance()  # 'o'
                while self._peek() and (self._peek() in '01234567_'):
                    raw += self._advance()
                clean = raw.replace('_', '')
                return Token(TokenType.OCT_INT, clean, self.line, start_col)
            elif next_ch in 'bB':
                raw += self._advance()  # '0'
                raw += self._advance()  # 'b'
                while self._peek() and (self._peek() in '01_'):
                    raw += self._advance()
                clean = raw.replace('_', '')
                return Token(TokenType.BIN_INT, clean, self.line, start_col)

        # Regular decimal number (possibly with separators)
        is_float = False
        while self._peek() and (self._peek().isdigit() or self._peek() == '_'):
            raw += self._advance()
        if self._peek() == '.':
            is_float = True
            raw += self._advance()
            while self._peek() and (self._peek().isdigit() or self._peek() == '_'):
                raw += self._advance()

        clean = raw.replace('_', '')
        if is_float:
            return Token(TokenType.FLOAT, clean, self.line, start_col)
        return Token(TokenType.INT, clean, self.line, start_col)

    def tokenize(self):
        tokens = []
        while self.pos < len(self.source):
            self._skip_whitespace()
            if self.pos >= len(self.source):
                break

            ch = self._peek()
            start_col = self.col

            if ch.isdigit():
                tokens.append(self._read_number())
            elif ch.isalpha() or ch == '_':
                raw = ""
                while self._peek() and (self._peek().isalnum() or self._peek() == '_'):
                    raw += self._advance()
                tokens.append(Token(TokenType.ID, raw, self.line, start_col))
            elif ch == '+':
                self._advance()
                tokens.append(Token(TokenType.PLUS, '+', self.line, start_col))
            elif ch == '-':
                self._advance()
                tokens.append(Token(TokenType.MINUS, '-', self.line, start_col))
            elif ch == '*':
                self._advance()
                tokens.append(Token(TokenType.STAR, '*', self.line, start_col))
            elif ch == '/':
                self._advance()
                tokens.append(Token(TokenType.SLASH, '/', self.line, start_col))
            elif ch == '=':
                self._advance()
                tokens.append(Token(TokenType.ASSIGN, '=', self.line, start_col))
            elif ch == '(':
                self._advance()
                tokens.append(Token(TokenType.LPAREN, '(', self.line, start_col))
            elif ch == ')':
                self._advance()
                tokens.append(Token(TokenType.RPAREN, ')', self.line, start_col))
            elif ch == ';':
                self._advance()
                tokens.append(Token(TokenType.SEMI, ';', self.line, start_col))
            else:
                self._advance()
                tokens.append(Token(TokenType.ERROR, ch, self.line, start_col))

        tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return tokens


def exercise_4():
    """Extended lexer with hex, octal, binary literals and numeric separators."""
    test_cases = [
        "0xFF",
        "0x1A3",
        "0xFF_FF",
        "0o77",
        "0o12",
        "0b1010",
        "0b11001",
        "1_000_000",
        "3.14_15_92",
        "x = 0xFF + 0b1010 * 0o77;",
        "total = 1_000_000 + 0xFF_FF;",
    ]

    for source in test_cases:
        lexer = ExtendedLexer(source)
        tokens = lexer.tokenize()
        print(f"  Source: {source!r}")
        for tok in tokens:
            if tok.type == TokenType.EOF:
                continue
            print(f"    {tok.type.name:>10}: {tok.value!r}")

        # Verify values can be parsed
        for tok in tokens:
            if tok.type == TokenType.HEX_INT:
                print(f"    -> int value: {int(tok.value, 16)}")
            elif tok.type == TokenType.OCT_INT:
                print(f"    -> int value: {int(tok.value, 8)}")
            elif tok.type == TokenType.BIN_INT:
                print(f"    -> int value: {int(tok.value, 2)}")
            elif tok.type == TokenType.INT:
                print(f"    -> int value: {int(tok.value)}")
        print()


# === Exercise 5: Error Recovery ===
# Problem: Design error recovery for unterminated strings, typo suggestions,
# and grouping consecutive illegal characters.

class ErrorRecoveryLexer:
    """Lexer with sophisticated error recovery."""

    KEYWORDS = {'if', 'else', 'while', 'for', 'return', 'int', 'float', 'void',
                'break', 'continue', 'struct', 'switch', 'case', 'default'}

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.errors = []

    def _peek(self) -> Optional[str]:
        return self.source[self.pos] if self.pos < len(self.source) else None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\n\r':
            self._advance()

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Levenshtein distance for typo detection."""
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def _suggest_keyword(self, word: str) -> Optional[str]:
        """Suggest a keyword if the word is a close typo."""
        best_kw = None
        best_dist = float('inf')
        for kw in self.KEYWORDS:
            dist = self._edit_distance(word, kw)
            if dist < best_dist and dist <= 2:  # max edit distance of 2
                best_dist = dist
                best_kw = kw
        return best_kw

    def _read_string(self) -> Token:
        """Read a string literal with unterminated string detection."""
        start_line = self.line
        start_col = self.col
        quote = self._advance()  # consume opening quote
        raw = quote
        while self.pos < len(self.source):
            ch = self._peek()
            if ch == '\n':
                # Unterminated string at end of line
                self.errors.append(
                    f"Line {start_line}, Col {start_col}: Unterminated string literal "
                    f"(started at line {start_line})"
                )
                return Token(TokenType.ERROR, raw, start_line, start_col)
            if ch == '\\':
                raw += self._advance()
                if self.pos < len(self.source):
                    raw += self._advance()
                continue
            if ch == quote:
                raw += self._advance()
                return Token(TokenType.ID, raw, start_line, start_col)  # simplified
            raw += self._advance()

        # Reached end of file without closing quote
        self.errors.append(
            f"Line {start_line}, Col {start_col}: Unterminated string literal "
            f"(started at line {start_line}, reached end of file)"
        )
        return Token(TokenType.ERROR, raw, start_line, start_col)

    def tokenize(self):
        """Tokenize with error recovery."""
        tokens = []
        while self.pos < len(self.source):
            self._skip_whitespace()
            if self.pos >= len(self.source):
                break

            ch = self._peek()
            start_line = self.line
            start_col = self.col

            if ch == '"' or ch == "'":
                tokens.append(self._read_string())
            elif ch.isalpha() or ch == '_':
                raw = ""
                while self._peek() and (self._peek().isalnum() or self._peek() == '_'):
                    raw += self._advance()
                # Check for keyword typos
                if raw not in self.KEYWORDS and raw.isalpha():
                    suggestion = self._suggest_keyword(raw)
                    if suggestion:
                        self.errors.append(
                            f"Line {start_line}, Col {start_col}: Unknown identifier '{raw}' "
                            f"-- did you mean '{suggestion}'?"
                        )
                tokens.append(Token(TokenType.ID, raw, start_line, start_col))
            elif ch.isdigit():
                raw = ""
                while self._peek() and self._peek().isdigit():
                    raw += self._advance()
                tokens.append(Token(TokenType.INT, raw, start_line, start_col))
            elif ch in '+-*/=();{}':
                self._advance()
                tokens.append(Token(TokenType.ID, ch, start_line, start_col))
            else:
                # Group consecutive illegal characters
                illegal = ""
                while self._peek() and not (
                    self._peek().isalnum() or self._peek() in ' \t\n\r+-*/=();{}_"\''):
                    illegal += self._advance()
                self.errors.append(
                    f"Line {start_line}, Col {start_col}: Illegal characters: {illegal!r}"
                )
                tokens.append(Token(TokenType.ERROR, illegal, start_line, start_col))

        tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return tokens


def exercise_5():
    """Lexer error recovery: unterminated strings, typo suggestions, grouped errors."""
    test_cases = [
        ('x = "hello world', "Unterminated string"),
        ('retrun x;', "Keyword typo"),
        ('whlie (x) { breek; }', "Multiple keyword typos"),
        ('x = 5 @#$ y;', "Consecutive illegal characters"),
    ]

    for source, description in test_cases:
        print(f"  Test: {description}")
        print(f"  Source: {source!r}")
        lexer = ErrorRecoveryLexer(source)
        tokens = lexer.tokenize()
        for tok in tokens:
            if tok.type == TokenType.EOF:
                continue
            print(f"    {tok.type.name:>10}: {tok.value!r}")
        if lexer.errors:
            print(f"  Errors:")
            for err in lexer.errors:
                print(f"    {err}")
        print()


# === Exercise 6: Performance Comparison ===
# Problem: Benchmark hand-written lexer vs regex-based lexer on a large input.

def exercise_6():
    """Performance comparison: hand-written lexer vs regex-based tokenizer."""
    # Generate a large source file with random tokens
    random.seed(42)
    keywords = ['if', 'else', 'while', 'for', 'return', 'int', 'float']
    operators = ['+', '-', '*', '/', '=', '(', ')', '{', '}', ';']

    lines = []
    for _ in range(5000):
        tokens_per_line = random.randint(3, 10)
        line_tokens = []
        for _ in range(tokens_per_line):
            choice = random.random()
            if choice < 0.2:
                line_tokens.append(random.choice(keywords))
            elif choice < 0.4:
                line_tokens.append(str(random.randint(0, 99999)))
            elif choice < 0.6:
                name_len = random.randint(1, 10)
                name = random.choice(string.ascii_lowercase) + ''.join(
                    random.choices(string.ascii_lowercase + string.digits, k=name_len-1))
                line_tokens.append(name)
            else:
                line_tokens.append(random.choice(operators))
        lines.append(' '.join(line_tokens))
    source = '\n'.join(lines)
    print(f"  Generated source: {len(source)} bytes, {len(lines)} lines")

    # Method 1: Hand-written lexer (our ExtendedLexer)
    start = time.perf_counter()
    lexer = ExtendedLexer(source)
    tokens1 = lexer.tokenize()
    time1 = time.perf_counter() - start
    count1 = len([t for t in tokens1 if t.type != TokenType.EOF])
    print(f"  Hand-written lexer: {time1:.4f}s, {count1} tokens")

    # Method 2: Regex-based lexer using Python's re module
    TOKEN_PATTERN = re.compile(r"""
        (?P<ID>[a-zA-Z_]\w*)
        | (?P<INT>\d+)
        | (?P<OP>[+\-*/=(){};])
        | (?P<WS>\s+)
        | (?P<ERR>.)
    """, re.VERBOSE)

    start = time.perf_counter()
    tokens2 = []
    for m in TOKEN_PATTERN.finditer(source):
        kind = m.lastgroup
        if kind == 'WS':
            continue
        tokens2.append((kind, m.group()))
    time2 = time.perf_counter() - start
    count2 = len(tokens2)
    print(f"  Regex-based lexer:  {time2:.4f}s, {count2} tokens")

    # Compare
    print(f"  Token count match: {count1 == count2}")
    if time1 > 0 and time2 > 0:
        ratio = time1 / time2
        faster = "regex" if ratio > 1 else "hand-written"
        print(f"  Speed ratio: {ratio:.2f}x ({faster} is faster)")
    print(f"  Note: Python's re module is implemented in C, so the regex approach")
    print(f"  is typically faster for pure Python hand-written lexers.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Regular Expressions ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Thompson's Construction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Subset Construction ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Lexer Extension ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Error Recovery ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Performance Comparison ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
