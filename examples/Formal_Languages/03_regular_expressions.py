"""
Regular Expression Engine (Thompson's Construction + NFA Simulation)
=====================================================================

Demonstrates:
- Parsing regex syntax into an AST
- Thompson's construction: regex → NFA
- NFA simulation for pattern matching
- Equivalence: regex ↔ NFA ↔ DFA

Supported operators: | (union), * (Kleene star), + (one or more),
                     ? (optional), . (concatenation implicit)

Reference: Formal_Languages Lesson 4 — Regular Expressions
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


# ─────────────── AST ───────────────

class NodeType(Enum):
    LITERAL = auto()
    UNION = auto()
    CONCAT = auto()
    STAR = auto()
    PLUS = auto()
    OPTIONAL = auto()


@dataclass
class RegexNode:
    type: NodeType
    value: Optional[str] = None  # for LITERAL
    left: Optional[RegexNode] = None
    right: Optional[RegexNode] = None  # for UNION, CONCAT


# ─────────────── Parser ───────────────

# Why: A recursive-descent parser mirrors the grammar rules one-to-one,
# making it easy to verify correctness. Operator precedence is encoded
# structurally: * binds tightest (factor), then concat (term), then | (expr).
class RegexParser:
    """
    Parse regex string into AST.
    Grammar:
        expr   → term ('|' term)*
        term   → factor factor*
        factor → base ('*' | '+' | '?')*
        base   → CHAR | '(' expr ')'
    """
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0

    def parse(self) -> RegexNode:
        node = self._expr()
        if self.pos < len(self.pattern):
            raise ValueError(f"Unexpected '{self.pattern[self.pos]}' at position {self.pos}")
        return node

    def _expr(self) -> RegexNode:
        node = self._term()
        while self.pos < len(self.pattern) and self._peek() == '|':
            self._advance()  # consume '|'
            right = self._term()
            node = RegexNode(NodeType.UNION, left=node, right=right)
        return node

    def _term(self) -> RegexNode:
        node = self._factor()
        while self.pos < len(self.pattern) and self._peek() not in ('|', ')'):
            right = self._factor()
            node = RegexNode(NodeType.CONCAT, left=node, right=right)
        return node

    def _factor(self) -> RegexNode:
        node = self._base()
        while self.pos < len(self.pattern) and self._peek() in ('*', '+', '?'):
            op = self._advance()
            if op == '*':
                node = RegexNode(NodeType.STAR, left=node)
            elif op == '+':
                node = RegexNode(NodeType.PLUS, left=node)
            elif op == '?':
                node = RegexNode(NodeType.OPTIONAL, left=node)
        return node

    def _base(self) -> RegexNode:
        if self._peek() == '(':
            self._advance()  # consume '('
            node = self._expr()
            if self.pos >= len(self.pattern) or self._peek() != ')':
                raise ValueError("Missing closing parenthesis")
            self._advance()  # consume ')'
            return node
        else:
            ch = self._advance()
            return RegexNode(NodeType.LITERAL, value=ch)

    def _peek(self) -> str:
        return self.pattern[self.pos]

    def _advance(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch


# ─────────────── NFA (Thompson's Construction) ───────────────

@dataclass
class NFAState:
    id: int
    transitions: Dict[str, List[NFAState]]  # symbol -> [states]

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, NFAState) and self.id == other.id


class ThompsonNFA:
    """NFA fragment with single start and single accept state."""
    def __init__(self, start: NFAState, accept: NFAState):
        self.start = start
        self.accept = accept


# Why: Thompson's construction guarantees exactly 2 states per literal and
# O(n) total states for a regex of length n. Each fragment has one start and
# one accept, making composition (concat, union, star) clean and modular.
class NFABuilder:
    """Build NFA from regex AST using Thompson's construction."""
    def __init__(self):
        self._next_id = 0

    def _new_state(self) -> NFAState:
        state = NFAState(self._next_id, {})
        self._next_id += 1
        return state

    def _add_transition(self, src: NFAState, symbol: str, dst: NFAState):
        src.transitions.setdefault(symbol, []).append(dst)

    def build(self, node: RegexNode) -> ThompsonNFA:
        if node.type == NodeType.LITERAL:
            return self._literal(node.value)
        elif node.type == NodeType.CONCAT:
            return self._concat(self.build(node.left), self.build(node.right))
        elif node.type == NodeType.UNION:
            return self._union(self.build(node.left), self.build(node.right))
        elif node.type == NodeType.STAR:
            return self._star(self.build(node.left))
        elif node.type == NodeType.PLUS:
            return self._plus(self.build(node.left))
        elif node.type == NodeType.OPTIONAL:
            return self._optional(self.build(node.left))
        raise ValueError(f"Unknown node type: {node.type}")

    def _literal(self, ch: str) -> ThompsonNFA:
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, ch, accept)
        return ThompsonNFA(start, accept)

    def _concat(self, a: ThompsonNFA, b: ThompsonNFA) -> ThompsonNFA:
        self._add_transition(a.accept, "ε", b.start)
        return ThompsonNFA(a.start, b.accept)

    def _union(self, a: ThompsonNFA, b: ThompsonNFA) -> ThompsonNFA:
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, "ε", a.start)
        self._add_transition(start, "ε", b.start)
        self._add_transition(a.accept, "ε", accept)
        self._add_transition(b.accept, "ε", accept)
        return ThompsonNFA(start, accept)

    def _star(self, a: ThompsonNFA) -> ThompsonNFA:
        # Why: Kleene star needs four epsilon edges: start→inner (enter loop),
        # start→accept (match empty), inner_accept→inner_start (repeat),
        # inner_accept→accept (exit). This covers zero or more repetitions.
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, "ε", a.start)
        self._add_transition(start, "ε", accept)
        self._add_transition(a.accept, "ε", a.start)
        self._add_transition(a.accept, "ε", accept)
        return ThompsonNFA(start, accept)

    def _plus(self, a: ThompsonNFA) -> ThompsonNFA:
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, "ε", a.start)
        self._add_transition(a.accept, "ε", a.start)
        self._add_transition(a.accept, "ε", accept)
        return ThompsonNFA(start, accept)

    def _optional(self, a: ThompsonNFA) -> ThompsonNFA:
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, "ε", a.start)
        self._add_transition(start, "ε", accept)
        self._add_transition(a.accept, "ε", accept)
        return ThompsonNFA(start, accept)


# ─────────────── NFA Simulation ───────────────

def epsilon_closure(states: Set[NFAState]) -> Set[NFAState]:
    """Compute ε-closure of a set of NFA states."""
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for next_state in state.transitions.get("ε", []):
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)
    return closure


# Why: On-the-fly simulation avoids building the full DFA. We track the set
# of active NFA states and advance them symbol-by-symbol — O(n*m) time for
# input length n and NFA size m, which is fast enough for most patterns.
def nfa_simulate(nfa: ThompsonNFA, input_str: str) -> bool:
    """Simulate NFA on input string using on-the-fly subset construction."""
    current = epsilon_closure({nfa.start})
    for ch in input_str:
        next_states: Set[NFAState] = set()
        for state in current:
            for target in state.transitions.get(ch, []):
                next_states.add(target)
        current = epsilon_closure(next_states)
    return nfa.accept in current


# ─────────────── High-Level API ───────────────

def regex_match(pattern: str, text: str) -> bool:
    """Check if the entire text matches the regex pattern."""
    ast = RegexParser(pattern).parse()
    nfa = NFABuilder().build(ast)
    return nfa_simulate(nfa, text)


def count_nfa_states(nfa: ThompsonNFA) -> int:
    """Count total NFA states by BFS."""
    visited: Set[int] = set()
    queue = [nfa.start]
    visited.add(nfa.start.id)
    while queue:
        state = queue.pop(0)
        for targets in state.transitions.values():
            for t in targets:
                if t.id not in visited:
                    visited.add(t.id)
                    queue.append(t)
    return len(visited)


# ─────────────── Demos ───────────────

def demo_basic_matching():
    """Basic regex matching examples."""
    print("=" * 60)
    print("Demo 1: Basic Regex Matching")
    print("=" * 60)

    tests = [
        ("a", "a", True),
        ("a", "b", False),
        ("ab", "ab", True),
        ("a|b", "a", True),
        ("a|b", "b", True),
        ("a|b", "c", False),
        ("a*", "", True),
        ("a*", "aaa", True),
        ("a*", "b", False),
        ("a+", "", False),
        ("a+", "a", True),
        ("a+", "aaa", True),
        ("a?", "", True),
        ("a?", "a", True),
        ("a?", "aa", False),
    ]

    for pattern, text, expected in tests:
        result = regex_match(pattern, text)
        display = text if text else "ε"
        status = "OK" if result == expected else "FAIL"
        print(f"  /{pattern}/ on '{display}': {result} (expected {expected}) {status}")


def demo_complex_patterns():
    """More complex pattern matching."""
    print("\n" + "=" * 60)
    print("Demo 2: Complex Patterns")
    print("=" * 60)

    patterns = [
        ("(a|b)*", "Strings over {a,b}"),
        ("(ab)*", "Repetitions of 'ab'"),
        ("a(b|c)*d", "a, then b's and c's, then d"),
        ("(0|1)*01", "Binary strings ending in 01"),
        ("(a|b)*aba(a|b)*", "Strings containing 'aba'"),
    ]

    for pattern, desc in patterns:
        print(f"\n  Pattern: /{pattern}/  ({desc})")
        ast = RegexParser(pattern).parse()
        nfa = NFABuilder().build(ast)
        n_states = count_nfa_states(nfa)
        print(f"  NFA states: {n_states}")

        if pattern == "(a|b)*":
            tests = ["", "a", "b", "ab", "ba", "abba", "c"]
        elif pattern == "(ab)*":
            tests = ["", "ab", "abab", "ababab", "a", "ba", "aba"]
        elif pattern == "a(b|c)*d":
            tests = ["ad", "abd", "acd", "abcd", "abbd", "a", "d", "abcde"]
        elif pattern == "(0|1)*01":
            tests = ["01", "001", "101", "0101", "10", "1", ""]
        else:
            tests = ["aba", "aaba", "abab", "baba", "bb", ""]

        for text in tests:
            result = nfa_simulate(nfa, text)
            display = text if text else "ε"
            print(f"    '{display}': {'MATCH' if result else 'no match'}")


def demo_thompson_sizes():
    """Show how Thompson's construction produces O(n) states."""
    print("\n" + "=" * 60)
    print("Demo 3: Thompson's NFA Size (O(n) States)")
    print("=" * 60)

    patterns = ["a", "ab", "a|b", "a*", "(a|b)*", "(ab|cd)*ef",
                "a(b|c)*d(e|f)*g", "(a|b|c|d)*"]

    print(f"  {'Pattern':<25} {'Length':>6} {'NFA States':>10}")
    print("  " + "-" * 45)
    for pattern in patterns:
        ast = RegexParser(pattern).parse()
        nfa = NFABuilder().build(ast)
        n_states = count_nfa_states(nfa)
        print(f"  {pattern:<25} {len(pattern):>6} {n_states:>10}")


def demo_algebraic_identities():
    """Verify regex algebraic identities by testing equivalence."""
    print("\n" + "=" * 60)
    print("Demo 4: Algebraic Identities Verification")
    print("=" * 60)

    # Generate all strings up to length 4 over {a, b}
    # Why: Exhaustive testing over all strings up to length 4 is a practical
    # way to verify regex algebraic identities — a finite but thorough check.
    def gen_strings(alphabet: str, max_len: int) -> list[str]:
        strings = [""]
        for length in range(1, max_len + 1):
            for s in gen_strings_of_len(alphabet, length):
                strings.append(s)
        return strings

    def gen_strings_of_len(alphabet: str, length: int) -> list[str]:
        if length == 0:
            return [""]
        return [c + s for c in alphabet
                for s in gen_strings_of_len(alphabet, length - 1)]

    test_strings = gen_strings("ab", 4)

    identities = [
        ("a|a", "a", "R ∪ R = R (idempotence)"),
        ("a(b|c)", "ab|ac", "R(S∪T) = RS ∪ RT (distribution)"),
        ("(a|b)(a|b)*", "(a|b)+", "(R)(R*) = R+"),
    ]

    for left, right, name in identities:
        all_match = True
        for s in test_strings:
            l_result = regex_match(left, s)
            r_result = regex_match(right, s)
            if l_result != r_result:
                all_match = False
                print(f"  MISMATCH on '{s}': /{left}/={l_result}, /{right}/={r_result}")
                break
        status = "VERIFIED" if all_match else "FAILED"
        print(f"  {name}")
        print(f"    /{left}/ ≡ /{right}/ : {status} (tested {len(test_strings)} strings)")


if __name__ == "__main__":
    demo_basic_matching()
    demo_complex_patterns()
    demo_thompson_sizes()
    demo_algebraic_identities()
