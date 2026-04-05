"""
Context-Free Grammar Tools: CFG, CNF Conversion, and CYK Parser
=================================================================

Demonstrates:
- CFG representation and derivation
- Chomsky Normal Form (CNF) conversion
- CYK (Cocke-Younger-Kasami) parsing algorithm
- Parse tree reconstruction

Reference: Formal_Languages Lesson 6 — Context-Free Grammars
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class CFG:
    """Context-Free Grammar."""
    variables: Set[str]
    terminals: Set[str]
    rules: Dict[str, List[List[str]]]  # variable -> list of productions
    start: str

    def derives(self, variable: str, s: str, depth: int = 20) -> bool:
        """Check if variable derives string s (brute force, limited depth)."""
        # Why: Brute-force derivation with a depth limit avoids infinite recursion
        # on left-recursive grammars. For reliable membership testing, use CYK instead.
        if depth <= 0:
            return False
        if variable in self.terminals:
            return variable == s
        for production in self.rules.get(variable, []):
            if self._try_production(production, s, depth - 1):
                return True
        return False

    def _try_production(self, production: List[str], s: str, depth: int) -> bool:
        """Try to match a production against string s."""
        if not production:
            return s == ""
        if len(production) == 1:
            symbol = production[0]
            if symbol in self.terminals:
                return symbol == s
            return self.derives(symbol, s, depth)
        # Split string for each possible position
        first = production[0]
        rest = production[1:]
        for i in range(len(s) + 1):
            left_str = s[:i]
            right_str = s[i:]
            if first in self.terminals:
                if left_str == first and self._try_production(rest, right_str, depth):
                    return True
            else:
                if self.derives(first, left_str, depth):
                    if self._try_production(rest, right_str, depth):
                        return True
        return False

    def display(self):
        """Pretty-print the grammar."""
        for var in sorted(self.rules.keys()):
            prods = [" ".join(p) if p else "ε" for p in self.rules[var]]
            print(f"  {var} → {' | '.join(prods)}")


# Why: CNF is required by the CYK algorithm. Every rule has exactly 2 variables
# or 1 terminal, enabling the O(n^3) dynamic programming approach. The 4-step
# conversion (eliminate ε, unit rules, mixed terminals, long rules) preserves the language.
def to_cnf(cfg: CFG) -> CFG:
    """
    Convert a CFG to Chomsky Normal Form.

    CNF rules are either:
    - A → BC (two variables)
    - A → a (single terminal)
    - S → ε (if ε ∈ L(G), and S doesn't appear on RHS)
    """
    variables = set(cfg.variables)
    terminals = set(cfg.terminals)
    rules: Dict[str, List[List[str]]] = {v: list(prods) for v, prods in cfg.rules.items()}
    start = cfg.start

    # Step 1: Eliminate ε-productions (except S → ε)
    nullable = _find_nullable(variables, rules)

    new_rules: Dict[str, List[List[str]]] = {v: [] for v in variables}
    for var in variables:
        for prod in rules.get(var, []):
            if prod == []:  # ε-production
                continue
            # Generate all combinations with nullable symbols omitted
            for combo in _nullable_combos(prod, nullable):
                if combo and combo not in new_rules[var]:
                    new_rules[var].append(combo)

    if start in nullable:
        new_rules[start].append([])  # allow S → ε

    rules = new_rules

    # Step 2: Eliminate unit productions (A → B)
    changed = True
    while changed:
        changed = False
        for var in variables:
            new_prods = []
            for prod in rules[var]:
                if len(prod) == 1 and prod[0] in variables:
                    # Unit production: replace with target's productions
                    for target_prod in rules.get(prod[0], []):
                        if target_prod not in rules[var] and target_prod not in new_prods:
                            new_prods.append(target_prod)
                            changed = True
                else:
                    if prod not in new_prods:
                        new_prods.append(prod)
            rules[var] = new_prods

    # Step 3: Replace terminals in long rules
    term_vars: Dict[str, str] = {}
    extra_rules: Dict[str, List[List[str]]] = {}

    for var in list(variables):
        new_prods = []
        for prod in rules[var]:
            if len(prod) <= 1:
                new_prods.append(prod)
                continue
            # Replace terminals with new variables
            new_prod = []
            for symbol in prod:
                if symbol in terminals:
                    if symbol not in term_vars:
                        tv = f"T_{symbol}"
                        term_vars[symbol] = tv
                        variables.add(tv)
                        extra_rules[tv] = [[symbol]]
                    new_prod.append(term_vars[symbol])
                else:
                    new_prod.append(symbol)
            new_prods.append(new_prod)
        rules[var] = new_prods

    rules.update(extra_rules)

    # Step 4: Break long rules (A → BCD becomes A → BX, X → CD)
    counter = [0]
    final_rules: Dict[str, List[List[str]]] = {v: [] for v in variables}
    for var in list(variables):
        for prod in rules.get(var, []):
            if len(prod) <= 2:
                final_rules.setdefault(var, []).append(prod)
            else:
                # Break into binary rules
                current = var
                for i in range(len(prod) - 2):
                    new_var = f"X{counter[0]}"
                    counter[0] += 1
                    variables.add(new_var)
                    final_rules.setdefault(current, []).append([prod[i], new_var])
                    current = new_var
                final_rules.setdefault(current, []).append(prod[-2:])

    return CFG(variables=variables, terminals=terminals, rules=final_rules, start=start)


def _find_nullable(variables: Set[str], rules: Dict[str, List[List[str]]]) -> Set[str]:
    """Find all nullable variables (those that can derive ε)."""
    nullable: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for var in variables:
            if var in nullable:
                continue
            for prod in rules.get(var, []):
                if not prod or all(s in nullable for s in prod):
                    nullable.add(var)
                    changed = True
                    break
    return nullable


def _nullable_combos(prod: List[str], nullable: Set[str]) -> List[List[str]]:
    """Generate all combinations of a production with nullable symbols omitted."""
    if not prod:
        return [[]]
    first = prod[0]
    rest_combos = _nullable_combos(prod[1:], nullable)
    result = []
    for combo in rest_combos:
        result.append([first] + combo)
        if first in nullable:
            result.append(combo)
    return result


# ─────────────── CYK Parser ───────────────

@dataclass
class ParseNode:
    """Node in a parse tree."""
    variable: str
    children: List[ParseNode | str] = field(default_factory=list)

    def display(self, indent: int = 0):
        prefix = "  " * indent
        if not self.children:
            print(f"{prefix}{self.variable}")
        elif len(self.children) == 1 and isinstance(self.children[0], str):
            print(f"{prefix}{self.variable} → {self.children[0]}")
        else:
            print(f"{prefix}{self.variable}")
            for child in self.children:
                if isinstance(child, ParseNode):
                    child.display(indent + 1)
                else:
                    print(f"{prefix}  {child}")


# Why: CYK is the standard O(n^3) parsing algorithm for CFGs. It fills a
# triangular table bottom-up: T[i][j] holds variables that derive substring
# input[i..j]. Backpointers enable parse tree reconstruction.
def cyk_parse(cfg: CFG, input_str: str) -> Tuple[bool, Optional[ParseNode]]:
    """
    CYK algorithm for membership testing and parse tree construction.

    Requires grammar in CNF.
    Returns (accepted, parse_tree).
    """
    n = len(input_str)
    if n == 0:
        # Check if start variable has ε-production
        for prod in cfg.rules.get(cfg.start, []):
            if not prod:
                return True, ParseNode(cfg.start, ["ε"])
        return False, None

    # T[i][j] = set of (variable, backpointer) for substring input_str[i:j+1]
    # backpointer: None for terminals, (k, B, C) for A → BC split at k
    T: List[List[Dict[str, Optional[Tuple]]]] = [
        [{} for _ in range(n)] for _ in range(n)
    ]

    # Base case: single characters
    for i in range(n):
        ch = input_str[i]
        for var in cfg.variables:
            for prod in cfg.rules.get(var, []):
                if len(prod) == 1 and prod[0] == ch:
                    T[i][i][var] = None  # terminal, no backpointer

    # Why: We try every split point k for substring [i..j]. If B derives [i..k]
    # and C derives [k+1..j], then A → BC means A derives [i..j]. This is the
    # core DP recurrence that gives CYK its O(n^3 * |G|) complexity.
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for var in cfg.variables:
                    for prod in cfg.rules.get(var, []):
                        if len(prod) == 2:
                            B, C = prod
                            if B in T[i][k] and C in T[k+1][j]:
                                if var not in T[i][j]:
                                    T[i][j][var] = (k, B, C)

    accepted = cfg.start in T[0][n-1]
    tree = None
    if accepted:
        tree = _build_tree(T, input_str, 0, n - 1, cfg.start)

    return accepted, tree


def _build_tree(T, input_str, i, j, var) -> ParseNode:
    """Reconstruct parse tree from CYK table."""
    node = ParseNode(var)
    if i == j:
        node.children = [input_str[i]]
    else:
        bp = T[i][j][var]
        if bp is not None:
            k, B, C = bp
            node.children = [
                _build_tree(T, input_str, i, k, B),
                _build_tree(T, input_str, k + 1, j, C),
            ]
    return node


def cyk_table_display(cfg: CFG, input_str: str):
    """Display the CYK parsing table."""
    n = len(input_str)
    T: List[List[Set[str]]] = [[set() for _ in range(n)] for _ in range(n)]

    # Base case
    for i in range(n):
        ch = input_str[i]
        for var in cfg.variables:
            for prod in cfg.rules.get(var, []):
                if len(prod) == 1 and prod[0] == ch:
                    T[i][i].add(var)

    # Fill
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for var in cfg.variables:
                    for prod in cfg.rules.get(var, []):
                        if len(prod) == 2:
                            B, C = prod
                            if B in T[i][k] and C in T[k+1][j]:
                                T[i][j].add(var)

    # Display
    print(f"\n  CYK Table for '{input_str}':")
    header = "  " + "".join(f"  {input_str[i]:^8}" for i in range(n))
    print(header)
    for length in range(1, n + 1):
        row = f"  {length}: "
        for i in range(n - length + 1):
            j = i + length - 1
            cell = ",".join(sorted(T[i][j])) if T[i][j] else "∅"
            row += f"{cell:^10}"
        print(row)


# ─────────────── Demos ───────────────

def demo_cfg_derivation():
    """Basic CFG derivation."""
    print("=" * 60)
    print("Demo 1: CFG for {a^n b^n | n >= 0}")
    print("=" * 60)

    cfg = CFG(
        variables={"S"},
        terminals={"a", "b"},
        rules={"S": [["a", "S", "b"], []]},
        start="S",
    )

    print("  Grammar:")
    cfg.display()

    tests = ["", "ab", "aabb", "aaabbb", "a", "abb", "aab"]
    print("\n  Testing strings:")
    for s in tests:
        result = cfg.derives("S", s)
        display = s if s else "ε"
        print(f"    '{display}' ∈ L(G)? {result}")


def demo_cnf_conversion():
    """Convert a grammar to Chomsky Normal Form."""
    print("\n" + "=" * 60)
    print("Demo 2: CNF Conversion")
    print("=" * 60)

    # Grammar for simple arithmetic: E → E+T | T, T → T*F | F, F → (E) | a
    cfg = CFG(
        variables={"E", "T", "F"},
        terminals={"a", "+", "*", "(", ")"},
        rules={
            "E": [["E", "+", "T"], ["T"]],
            "T": [["T", "*", "F"], ["F"]],
            "F": [["(", "E", ")"], ["a"]],
        },
        start="E",
    )

    print("  Original grammar:")
    cfg.display()

    cnf = to_cnf(cfg)
    print("\n  CNF grammar:")
    cnf.display()
    print(f"\n  Variables: {len(cnf.variables)} (was {len(cfg.variables)})")

    # Verify same language
    tests = ["a", "a+a", "a*a", "a+a*a", "a*a+a"]
    print("\n  Verification (both should agree):")
    for s in tests:
        orig = cfg.derives("E", s, depth=10)
        # CNF check via CYK
        accepted, _ = cyk_parse(cnf, s)
        print(f"    '{s}': original={orig}, CNF/CYK={accepted}")


def demo_cyk_parser():
    """CYK parsing with table display."""
    print("\n" + "=" * 60)
    print("Demo 3: CYK Parsing Algorithm")
    print("=" * 60)

    # Grammar in CNF for palindromes over {a, b}
    # S → AA | BB | AB₁ | BA₁ | a | b
    # A₁ → SA, B₁ → SB, A → a, B → b
    cnf = CFG(
        variables={"S", "A", "B", "A1", "B1"},
        terminals={"a", "b"},
        rules={
            "S": [["A", "A1"], ["B", "B1"], ["A", "A"], ["B", "B"], ["a"], ["b"]],
            "A1": [["S", "A"]],
            "B1": [["S", "B"]],
            "A": [["a"]],
            "B": [["b"]],
        },
        start="S",
    )

    print("  Grammar (CNF) for palindromes over {a, b}:")
    cnf.display()

    tests = ["a", "aba", "abba", "aabaa", "ab", "abc"]
    for s in tests:
        accepted, tree = cyk_parse(cnf, s)
        cyk_table_display(cnf, s)
        print(f"  Result: '{s}' → {'ACCEPTED' if accepted else 'REJECTED'}")
        if tree:
            print("  Parse tree:")
            tree.display(2)
        print()


def demo_ambiguity():
    """Demonstrate grammar ambiguity."""
    print("=" * 60)
    print("Demo 4: Ambiguous vs Unambiguous Grammar")
    print("=" * 60)

    # Ambiguous grammar for arithmetic
    print("  Ambiguous grammar:")
    print("    E → E + E | E * E | a")
    print("  String 'a+a*a' has TWO parse trees:")
    print("    Tree 1: (a + a) * a  — addition first")
    print("    Tree 2: a + (a * a)  — multiplication first")

    # Unambiguous grammar (with precedence)
    print("\n  Unambiguous grammar (with precedence):")
    print("    E → E + T | T")
    print("    T → T * F | F")
    print("    F → a")
    print("  String 'a+a*a' has ONE parse tree:")
    print("    a + (a * a)  — multiplication binds tighter")

    # Demonstrate with CYK on the ambiguous grammar in CNF
    # E → EX | EM | a, X → PE, M → SE
    # P → +, S → *
    cfg_ambig = CFG(
        variables={"E", "X", "M", "P", "S"},
        terminals={"a", "+", "*"},
        rules={
            "E": [["E", "X"], ["E", "M"], ["a"]],
            "X": [["P", "E"]],
            "M": [["S", "E"]],
            "P": [["+"]],
            "S": [["*"]],
        },
        start="E",
    )

    test = "a+a*a"
    accepted, tree = cyk_parse(cfg_ambig, test)
    print(f"\n  CYK result for '{test}': {'ACCEPTED' if accepted else 'REJECTED'}")
    if tree:
        print("  One parse tree found by CYK:")
        tree.display(2)


if __name__ == "__main__":
    demo_cfg_derivation()
    demo_cnf_conversion()
    demo_cyk_parser()
    demo_ambiguity()
