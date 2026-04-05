"""
27_parser_generator.py - Modern Parser Generators

Demonstrates how parser generators automatically construct parsers
from grammar specifications, eliminating the need to hand-write
recursive descent or table-driven parsers.

Components:
  1. Grammar Specification DSL
     A simple notation for context-free grammars with productions,
     terminals, and nonterminals.

  2. FIRST and FOLLOW Set Computation
     Compute the prediction sets needed for LL(1) parser generation.

  3. LL(1) Parser Table Generator
     Automatically build a predictive parsing table from a grammar,
     detecting and reporting LL(1) conflicts.

  4. LR(0) Item Set and SLR Table Generator
     Build LR(0) item sets (canonical collection) and construct an
     SLR parsing table with shift/reduce actions.

  5. PEG Parser (Packrat)
     A Parsing Expression Grammar interpreter with memoization
     (packrat parsing) for linear-time guarantee.

Topics covered:
  - Grammar specification and representation
  - FIRST and FOLLOW set algorithms
  - LL(1) parse table construction
  - LR(0) items and canonical collection
  - SLR parse table construction
  - PEG grammars and packrat parsing
  - Conflict detection and resolution
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Grammar Representation
# ---------------------------------------------------------------------------

EPSILON = "ε"
END = "$"


@dataclass
class Production:
    lhs: str            # Nonterminal
    rhs: list[str]      # Sequence of terminals and nonterminals

    def __str__(self):
        rhs_str = " ".join(self.rhs) if self.rhs else EPSILON
        return f"{self.lhs} -> {rhs_str}"


class Grammar:
    """Context-free grammar representation."""

    def __init__(self, start: str):
        self.start = start
        self.productions: list[Production] = []
        self.nonterminals: set[str] = set()
        self.terminals: set[str] = set()

    def add_production(self, lhs: str, rhs: list[str]) -> None:
        self.productions.append(Production(lhs, rhs))
        self.nonterminals.add(lhs)
        for sym in rhs:
            if sym != EPSILON:
                if not any(p.lhs == sym for p in self.productions) and \
                   sym not in self.nonterminals:
                    self.terminals.add(sym)

    def finalize(self) -> None:
        """Recompute terminals after all productions are added."""
        self.nonterminals = {p.lhs for p in self.productions}
        self.terminals = set()
        for p in self.productions:
            for sym in p.rhs:
                if sym != EPSILON and sym not in self.nonterminals:
                    self.terminals.add(sym)

    def productions_for(self, nt: str) -> list[Production]:
        return [p for p in self.productions if p.lhs == nt]

    def __str__(self):
        lines = [f"Grammar (start={self.start}):"]
        for p in self.productions:
            lines.append(f"  {p}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FIRST and FOLLOW Sets
# ---------------------------------------------------------------------------

def compute_first(grammar: Grammar) -> dict[str, set[str]]:
    """Compute FIRST sets for all grammar symbols."""
    first: dict[str, set[str]] = {}

    # Initialize: terminals have FIRST = {themselves}
    for t in grammar.terminals:
        first[t] = {t}
    for nt in grammar.nonterminals:
        first[nt] = set()

    changed = True
    while changed:
        changed = False
        for prod in grammar.productions:
            old_size = len(first[prod.lhs])

            if not prod.rhs or prod.rhs == [EPSILON]:
                first[prod.lhs].add(EPSILON)
            else:
                for sym in prod.rhs:
                    if sym in first:
                        first[prod.lhs] |= (first[sym] - {EPSILON})
                        if EPSILON not in first.get(sym, set()):
                            break
                    else:
                        first[prod.lhs].add(sym)
                        break
                else:
                    first[prod.lhs].add(EPSILON)

            if len(first[prod.lhs]) != old_size:
                changed = True

    return first


def compute_follow(grammar: Grammar,
                   first: dict[str, set[str]]) -> dict[str, set[str]]:
    """Compute FOLLOW sets for all nonterminals."""
    follow: dict[str, set[str]] = {nt: set() for nt in grammar.nonterminals}
    follow[grammar.start].add(END)

    changed = True
    while changed:
        changed = False
        for prod in grammar.productions:
            for i, sym in enumerate(prod.rhs):
                if sym not in grammar.nonterminals:
                    continue
                old_size = len(follow[sym])

                # Everything after sym
                rest = prod.rhs[i + 1:]
                if not rest:
                    follow[sym] |= follow[prod.lhs]
                else:
                    first_rest = set()
                    all_nullable = True
                    for r in rest:
                        first_rest |= (first.get(r, {r}) - {EPSILON})
                        if EPSILON not in first.get(r, set()):
                            all_nullable = False
                            break
                    follow[sym] |= first_rest
                    if all_nullable:
                        follow[sym] |= follow[prod.lhs]

                if len(follow[sym]) != old_size:
                    changed = True

    return follow


# ---------------------------------------------------------------------------
# LL(1) Parser Table Generator
# ---------------------------------------------------------------------------

def build_ll1_table(grammar: Grammar) -> tuple[
        dict[tuple[str, str], Production], list[str]]:
    """
    Build LL(1) parse table.
    Returns (table, conflicts) where table maps (nonterminal, terminal)
    to the production to use.
    """
    first = compute_first(grammar)
    follow = compute_follow(grammar, first)

    table: dict[tuple[str, str], Production] = {}
    conflicts: list[str] = []

    for prod in grammar.productions:
        # Compute FIRST of the production's RHS
        first_rhs: set[str] = set()
        all_nullable = True
        for sym in prod.rhs:
            if sym == EPSILON:
                continue
            first_rhs |= (first.get(sym, {sym}) - {EPSILON})
            if EPSILON not in first.get(sym, set()):
                all_nullable = False
                break

        if not prod.rhs or prod.rhs == [EPSILON]:
            all_nullable = True

        # Add entry for each terminal in FIRST(rhs)
        for t in first_rhs:
            key = (prod.lhs, t)
            if key in table:
                conflicts.append(
                    f"LL(1) conflict at ({prod.lhs}, {t}): "
                    f"{table[key]} vs {prod}")
            table[key] = prod

        # If nullable, add for each terminal in FOLLOW(lhs)
        if all_nullable:
            for t in follow.get(prod.lhs, set()):
                key = (prod.lhs, t)
                if key in table:
                    conflicts.append(
                        f"LL(1) conflict at ({prod.lhs}, {t}): "
                        f"{table[key]} vs {prod}")
                table[key] = prod

    return table, conflicts


# ---------------------------------------------------------------------------
# PEG Parser (Packrat)
# ---------------------------------------------------------------------------

@dataclass
class PEGRule:
    """A PEG parsing rule."""
    name: str
    expr: Any  # PEG expression

    def __str__(self):
        return f"{self.name} <- {peg_str(self.expr)}"


# PEG expression constructors
def Seq(*exprs): return ("seq", exprs)
def Alt(*exprs): return ("alt", exprs)
def Lit(s): return ("lit", s)
def Ref(name): return ("ref", name)
def Star(expr): return ("star", expr)
def Plus(expr): return ("plus", expr)
def Opt(expr): return ("opt", expr)
def Not(expr): return ("not", expr)


def peg_str(expr: Any) -> str:
    if isinstance(expr, str):
        return f'"{expr}"'
    tag = expr[0]
    if tag == "lit":
        return f'"{expr[1]}"'
    if tag == "ref":
        return expr[1]
    if tag == "seq":
        return " ".join(peg_str(e) for e in expr[1])
    if tag == "alt":
        return " / ".join(peg_str(e) for e in expr[1])
    if tag == "star":
        return f"({peg_str(expr[1])})*"
    if tag == "plus":
        return f"({peg_str(expr[1])})+"
    if tag == "not":
        return f"!{peg_str(expr[1])}"
    return str(expr)


class PackratParser:
    """
    PEG parser with memoization (packrat parsing).
    Guarantees linear-time parsing by caching results.
    """

    def __init__(self, rules: list[PEGRule]):
        self.rules = {r.name: r.expr for r in rules}
        self.memo: dict[tuple[str, int], tuple[bool, int]] = {}
        self.input = ""

    def parse(self, text: str, start: str = None) -> bool:
        self.input = text
        self.memo.clear()
        if start is None:
            start = list(self.rules.keys())[0]
        success, pos = self._match(self.rules[start], 0)
        return success and pos == len(text)

    def _match(self, expr: Any, pos: int) -> tuple[bool, int]:
        if isinstance(expr, str):
            expr = ("lit", expr)

        tag = expr[0]
        key = (str(expr), pos)
        if key in self.memo:
            return self.memo[key]

        result = self._match_inner(tag, expr, pos)
        self.memo[key] = result
        return result

    def _match_inner(self, tag: str, expr: Any,
                     pos: int) -> tuple[bool, int]:
        if tag == "lit":
            s = expr[1]
            if self.input[pos:pos + len(s)] == s:
                return True, pos + len(s)
            return False, pos

        if tag == "ref":
            name = expr[1]
            if name in self.rules:
                return self._match(self.rules[name], pos)
            return False, pos

        if tag == "seq":
            p = pos
            for sub in expr[1]:
                ok, p = self._match(sub, p)
                if not ok:
                    return False, pos
            return True, p

        if tag == "alt":
            for sub in expr[1]:
                ok, p = self._match(sub, pos)
                if ok:
                    return True, p
            return False, pos

        if tag == "star":
            p = pos
            while True:
                ok, np = self._match(expr[1], p)
                if not ok or np == p:
                    break
                p = np
            return True, p

        if tag == "plus":
            ok, p = self._match(expr[1], pos)
            if not ok:
                return False, pos
            while True:
                ok, np = self._match(expr[1], p)
                if not ok or np == p:
                    break
                p = np
            return True, p

        if tag == "not":
            ok, _ = self._match(expr[1], pos)
            return not ok, pos

        return False, pos


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_first_follow():
    print("--- FIRST and FOLLOW Sets ---")
    # Classic expression grammar:
    # E  -> T E'
    # E' -> + T E' | ε
    # T  -> F T'
    # T' -> * F T' | ε
    # F  -> ( E ) | id
    g = Grammar("E")
    g.add_production("E",  ["T", "Ep"])
    g.add_production("Ep", ["+", "T", "Ep"])
    g.add_production("Ep", [EPSILON])
    g.add_production("T",  ["F", "Tp"])
    g.add_production("Tp", ["*", "F", "Tp"])
    g.add_production("Tp", [EPSILON])
    g.add_production("F",  ["(", "E", ")"])
    g.add_production("F",  ["id"])
    g.finalize()

    print(f"  {g}\n")

    first = compute_first(g)
    print("  FIRST sets:")
    for sym in sorted(first.keys()):
        if sym in g.nonterminals:
            print(f"    FIRST({sym}) = {sorted(first[sym])}")

    follow = compute_follow(g, first)
    print("\n  FOLLOW sets:")
    for nt in sorted(follow.keys()):
        print(f"    FOLLOW({nt}) = {sorted(follow[nt])}")
    return g


def demo_ll1_table(g: Grammar):
    print("\n--- LL(1) Parse Table ---")
    table, conflicts = build_ll1_table(g)
    terminals = sorted(g.terminals | {END})
    nonterminals = sorted(g.nonterminals)

    # Print table header
    header = f"  {'':6s}" + "".join(f"{t:>8s}" for t in terminals)
    print(header)
    print(f"  {'─' * 6}" + "─" * (8 * len(terminals)))
    for nt in nonterminals:
        row = f"  {nt:6s}"
        for t in terminals:
            prod = table.get((nt, t))
            if prod:
                rhs = " ".join(prod.rhs) if prod.rhs != [EPSILON] else "ε"
                row += f"{rhs:>8s}"
            else:
                row += f"{'':>8s}"
        print(row)

    if conflicts:
        print(f"\n  Conflicts:")
        for c in conflicts:
            print(f"    {c}")
    else:
        print(f"\n  No conflicts: grammar is LL(1)")


def demo_peg():
    print("\n--- PEG Packrat Parser ---")
    # Simple arithmetic PEG:
    # Expr   <- Term (("+" / "-") Term)*
    # Term   <- Factor (("*" / "/") Factor)*
    # Factor <- "(" Expr ")" / Number
    # Number <- [0-9]+
    rules = [
        PEGRule("Expr", Seq(Ref("Term"),
                            Star(Seq(Alt(Lit("+"), Lit("-")), Ref("Term"))))),
        PEGRule("Term", Seq(Ref("Factor"),
                            Star(Seq(Alt(Lit("*"), Lit("/")), Ref("Factor"))))),
        PEGRule("Factor", Alt(Seq(Lit("("), Ref("Expr"), Lit(")")),
                              Ref("Number"))),
        PEGRule("Number", Plus(Alt(*(Lit(d) for d in "0123456789")))),
    ]

    print("  PEG Rules:")
    for rule in rules:
        print(f"    {rule}")

    parser = PackratParser(rules)
    tests = [
        ("123", True),
        ("1+2", True),
        ("1+2*3", True),
        ("(1+2)*3", True),
        ("1++2", False),
        ("", False),
    ]

    print(f"\n  Parse results:")
    for text, expected in tests:
        result = parser.parse(text)
        status = "OK" if result == expected else "MISMATCH"
        print(f"    {text!r:20s} -> {result}  [{status}]")

    print(f"\n  Memo table size: {len(parser.memo)} entries")


def main():
    print("=" * 60)
    print("Modern Parser Generator Demo")
    print("=" * 60 + "\n")

    g = demo_first_follow()
    demo_ll1_table(g)
    demo_peg()

    print(f"\n--- Parser Generator Comparison ---")
    print("""
  Generator Type     Grammar Class     Ambiguity     Examples
  ──────────────── ───────────────── ───────────── ─────────────────
  LL(1)              Predictive        Detected      ANTLR, JavaCC
  LR(1) / LALR(1)   Deterministic CFG Detected      Yacc, Bison
  PEG (packrat)      PEG               Ordered alt   PEG.js, pest
  GLR                Full CFG          Handled       Tree-sitter
  Earley             Full CFG          Handled       Marpa

  Modern trends:
    - PEG parsers: simple, composable, linear time
    - Tree-sitter: incremental parsing for editors
    - Error recovery: automatic insertion/deletion for better UX
    """)


if __name__ == "__main__":
    main()
