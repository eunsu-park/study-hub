"""
Exercises for Lesson 09: Intermediate Representations
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple


# === Exercise 1: TAC Generation ===
# Problem: Translate code to three-address code.

def exercise_1():
    """Generate three-address code (TAC) for a program fragment."""
    print("Source code:")
    print("  x = 2 * a + b")
    print("  y = a * a - b * b")
    print("  if (x > y)")
    print("      z = x - y")
    print("  else")
    print("      z = y - x")
    print("  result = z * z")
    print()

    tac = [
        "// x = 2 * a + b",
        "t1 = 2 * a",
        "x  = t1 + b",
        "",
        "// y = a * a - b * b",
        "t2 = a * a",
        "t3 = b * b",
        "y  = t2 - t3",
        "",
        "// if (x > y)",
        "t4 = x > y",
        "iffalse t4 goto L1",
        "",
        "// then: z = x - y",
        "z  = x - y",
        "goto L2",
        "",
        "L1:  // else: z = y - x",
        "z  = y - x",
        "",
        "L2:  // result = z * z",
        "result = z * z",
    ]

    print("Three-Address Code:")
    for line in tac:
        print(f"  {line}")
    print()

    # Also implement a TAC generator
    class TACGenerator:
        def __init__(self):
            self.temp_count = 0
            self.label_count = 0
            self.instructions = []

        def new_temp(self):
            self.temp_count += 1
            return f"t{self.temp_count}"

        def new_label(self):
            self.label_count += 1
            return f"L{self.label_count}"

        def emit(self, instr):
            self.instructions.append(instr)

        def gen_expr(self, expr):
            """Generate TAC for an expression, return the result temp/var."""
            if isinstance(expr, (int, float)):
                return str(expr)
            if isinstance(expr, str):
                return expr
            if isinstance(expr, tuple):
                op, *args = expr
                if op in ('+', '-', '*', '/', '>', '<', '>=', '<=', '==', '!='):
                    left = self.gen_expr(args[0])
                    right = self.gen_expr(args[1])
                    t = self.new_temp()
                    self.emit(f"{t} = {left} {op} {right}")
                    return t
            return str(expr)

        def gen_stmt(self, stmt):
            """Generate TAC for a statement."""
            if stmt[0] == 'assign':
                _, var, expr = stmt
                result = self.gen_expr(expr)
                self.emit(f"{var} = {result}")
            elif stmt[0] == 'if':
                _, cond, then_stmts, else_stmts = stmt
                cond_result = self.gen_expr(cond)
                else_label = self.new_label()
                end_label = self.new_label()
                self.emit(f"iffalse {cond_result} goto {else_label}")
                for s in then_stmts:
                    self.gen_stmt(s)
                self.emit(f"goto {end_label}")
                self.emit(f"{else_label}:")
                for s in else_stmts:
                    self.gen_stmt(s)
                self.emit(f"{end_label}:")

    gen = TACGenerator()
    # Represent the program as AST tuples
    program = [
        ('assign', 'x', ('+', ('*', 2, 'a'), 'b')),
        ('assign', 'y', ('-', ('*', 'a', 'a'), ('*', 'b', 'b'))),
        ('if', ('>', 'x', 'y'),
         [('assign', 'z', ('-', 'x', 'y'))],
         [('assign', 'z', ('-', 'y', 'x'))]),
        ('assign', 'result', ('*', 'z', 'z')),
    ]

    for stmt in program:
        gen.gen_stmt(stmt)

    print("Generated TAC (via TACGenerator):")
    for instr in gen.instructions:
        print(f"  {instr}")


# === Exercise 2: Basic Block Identification ===
# Problem: Identify leaders and partition TAC into basic blocks.

def exercise_2():
    """Identify basic blocks from TAC."""
    tac = [
        "(1)  i = 0",
        "(2)  j = 0",
        "(3)  t1 = i < 10",        # L0: target of goto at (14)
        "(4)  iffalse t1 goto L3",
        "(5)  j = 0",
        "(6)  t2 = j < 10",        # L1: target of goto at (12)
        "(7)  iffalse t2 goto L2",
        "(8)  t3 = i * 10",
        "(9)  t4 = t3 + j",
        "(10) a[t4] = 0",
        "(11) j = j + 1",
        "(12) goto L1",
        "(13) i = i + 1",          # L2: target of iffalse at (7)
        "(14) goto L0",
        "(15) return",              # L3: target of iffalse at (4)
    ]

    print("TAC instructions:")
    for line in tac:
        print(f"  {line}")
    print()

    # Leaders:
    # 1. First instruction (1)
    # 2. Targets of jumps: L0=(3), L1=(6), L2=(13), L3=(15)
    # 3. Instructions after conditional jumps: (5) after (4), (8) after (7)
    # 4. Instructions after unconditional jumps: (13) after (12), (15) after (14)

    leaders = {1, 3, 5, 6, 8, 13, 15}
    print(f"Leaders: {sorted(leaders)}")
    print()

    # Partition into basic blocks
    blocks = {
        'B0': {'range': '(1)-(2)', 'instructions': ['i = 0', 'j = 0'],
               'note': 'Entry block'},
        'B1': {'range': '(3)-(4)', 'instructions': ['t1 = i < 10', 'iffalse t1 goto L3'],
               'note': 'Outer loop header (L0)'},
        'B2': {'range': '(5)', 'instructions': ['j = 0'],
               'note': 'Inner loop init'},
        'B3': {'range': '(6)-(7)', 'instructions': ['t2 = j < 10', 'iffalse t2 goto L2'],
               'note': 'Inner loop header (L1)'},
        'B4': {'range': '(8)-(12)', 'instructions': [
                   't3 = i * 10', 't4 = t3 + j', 'a[t4] = 0',
                   'j = j + 1', 'goto L1'],
               'note': 'Inner loop body'},
        'B5': {'range': '(13)-(14)', 'instructions': ['i = i + 1', 'goto L0'],
               'note': 'Outer loop increment (L2)'},
        'B6': {'range': '(15)', 'instructions': ['return'],
               'note': 'Exit (L3)'},
    }

    print("Basic Blocks:")
    for name, block in blocks.items():
        print(f"  {name} [{block['range']}] -- {block['note']}")
        for instr in block['instructions']:
            print(f"    {instr}")
    print()

    print("Control Flow Graph edges:")
    edges = [
        ("B0", "B1"),    # B0 falls through to B1
        ("B1", "B2"),    # B1 falls through (condition true)
        ("B1", "B6"),    # B1 jumps to L3=B6 (condition false)
        ("B2", "B3"),    # B2 falls through
        ("B3", "B4"),    # B3 falls through (condition true)
        ("B3", "B5"),    # B3 jumps to L2=B5 (condition false)
        ("B4", "B3"),    # B4 jumps to L1=B3
        ("B5", "B1"),    # B5 jumps to L0=B1
    ]
    for src, dst in edges:
        print(f"  {src} -> {dst}")


# === Exercise 3: SSA Conversion ===
# Problem: Convert code to SSA form with phi functions.

def exercise_3():
    """Convert code to SSA form."""
    print("Original code:")
    print("  B0: x = 1")
    print("      y = 2")
    print("      goto B1")
    print()
    print("  B1: if (x < 10) goto B2 else goto B3")
    print()
    print("  B2: x = x + 1")
    print("      y = y * 2")
    print("      goto B1")
    print()
    print("  B3: z = x + y")
    print("      return z")
    print()

    print("SSA Form:")
    print("=" * 50)
    print("  B0: x_1 = 1")
    print("      y_1 = 2")
    print("      goto B1")
    print()
    print("  B1: x_2 = phi(x_1 [from B0], x_3 [from B2])")
    print("      y_2 = phi(y_1 [from B0], y_3 [from B2])")
    print("      t_1 = x_2 < 10")
    print("      if t_1 goto B2 else goto B3")
    print()
    print("  B2: x_3 = x_2 + 1")
    print("      y_3 = y_2 * 2")
    print("      goto B1")
    print()
    print("  B3: z_1 = x_2 + y_2")
    print("      return z_1")
    print()

    print("Explanation:")
    print("  - B1 is the merge point for B0 (loop entry) and B2 (back edge).")
    print("  - x is defined in B0 (x_1) and B2 (x_3), so phi needed at B1 -> x_2.")
    print("  - y is defined in B0 (y_1) and B2 (y_3), so phi needed at B1 -> y_2.")
    print("  - z is only defined once, no phi needed.")
    print("  - Each definition gets a unique subscript (SSA property).")


# === Exercise 4: DAG Construction ===
# Problem: Build DAG for a basic block, identify shared subexpressions.

def exercise_4():
    """DAG construction for common subexpression elimination."""
    print("Basic block:")
    print("  a = b + c")
    print("  d = b + c")
    print("  e = a - d")
    print("  f = a * e")
    print("  g = f + e")
    print()

    # Build the DAG
    # Nodes are identified by their operation and operands
    print("DAG Construction:")
    print()
    print("  Step 1: a = b + c")
    print("    Create node '+' with children b, c")
    print("    a -> [+ : b, c]")
    print()
    print("  Step 2: d = b + c")
    print("    b + c already exists! Reuse the same node.")
    print("    d -> [+ : b, c]  (same node as a)")
    print()
    print("  Step 3: e = a - d")
    print("    Since a and d point to the same node (b+c),")
    print("    e = (b+c) - (b+c) = 0")
    print("    Create node '-' with children [+:b,c], [+:b,c]")
    print("    e -> [- : (b+c), (b+c)]")
    print()
    print("  Step 4: f = a * e")
    print("    Create node '*' with children [+:b,c], [-:(b+c),(b+c)]")
    print("    f -> [* : (b+c), e]")
    print()
    print("  Step 5: g = f + e")
    print("    Create node '+' (different from b+c!) with children f, e")
    print("    g -> [+ : f, e]")
    print()

    print("DAG (text representation):")
    print("       [g: +]")
    print("      /      \\")
    print("  [f: *]    [e: -]")
    print("    / \\      / \\")
    print("  [+]  [e] [+]  [+]   <- [+] is shared (b+c)")
    print("  / \\       ^    ^")
    print(" b   c      |    |")
    print("            +----+  (same node)")
    print()

    print("Shared subexpressions: b + c (used by a, d, and indirectly by e)")
    print()

    print("Optimized TAC:")
    optimized = [
        "t1 = b + c       // shared: a and d both use this",
        "a = t1",
        "d = t1            // CSE: reuse t1 instead of recomputing",
        "e = t1 - t1       // could be further optimized to e = 0",
        "f = t1 * e",
        "g = f + e",
    ]
    for line in optimized:
        print(f"  {line}")
    print()

    print("Further optimization (constant folding on e = t1 - t1):")
    further = [
        "t1 = b + c",
        "a = t1",
        "d = t1",
        "e = 0             // algebraic simplification: x - x = 0",
        "f = 0             // algebraic: x * 0 = 0",
        "g = 0             // algebraic: 0 + 0 = 0",
    ]
    for line in further:
        print(f"  {line}")
    print("  (If only g and result are live-out, much of this is dead code.)")


# === Exercise 5: Dominance Frontiers ===
# Problem: Compute dominator tree and dominance frontiers for a CFG.

def exercise_5():
    """Compute dominator tree and dominance frontiers."""
    # CFG: Entry -> B1, B1 -> B2,B3, B2 -> B4, B3 -> B4, B4 -> B5,B6, B5 -> B1, B6 -> Exit
    cfg = {
        'Entry': ['B1'],
        'B1': ['B2', 'B3'],
        'B2': ['B4'],
        'B3': ['B4'],
        'B4': ['B5', 'B6'],
        'B5': ['B1'],
        'B6': ['Exit'],
        'Exit': [],
    }
    all_nodes = list(cfg.keys())

    print("CFG:")
    for node, succs in cfg.items():
        if succs:
            print(f"  {node} -> {', '.join(succs)}")
    print()

    # Compute dominators using iterative algorithm
    dom = {n: set(all_nodes) for n in all_nodes}
    dom['Entry'] = {'Entry'}

    # Compute predecessors
    preds = defaultdict(list)
    for n, succs in cfg.items():
        for s in succs:
            preds[s].append(n)

    changed = True
    while changed:
        changed = False
        for n in all_nodes:
            if n == 'Entry':
                continue
            if not preds[n]:
                continue
            new_dom = set.intersection(*[dom[p] for p in preds[n]]) | {n}
            if new_dom != dom[n]:
                dom[n] = new_dom
                changed = True

    print("1. Dominators:")
    for n in all_nodes:
        print(f"  dom({n}) = {{{', '.join(sorted(dom[n]))}}}")
    print()

    # Compute immediate dominators
    idom = {}
    for n in all_nodes:
        if n == 'Entry':
            continue
        doms = dom[n] - {n}
        # idom(n) is the dominator of n that is dominated by all other dominators of n
        for d in doms:
            if all(d in dom[other] or d == other for other in doms):
                idom[n] = d
                break

    print("  Dominator tree (immediate dominators):")
    for n in sorted(idom.keys()):
        print(f"    idom({n}) = {idom[n]}")
    print()

    # Draw dominator tree
    print("  Dominator tree structure:")
    print("        Entry")
    print("          |")
    print("         B1")
    print("       / | \\")
    print("      B2 B3 B5")
    print("       \\ /")
    print("        B4")
    print("        |")
    print("        B6")
    print("        |")
    print("       Exit")
    print()

    # Compute dominance frontiers
    df = defaultdict(set)
    for n in all_nodes:
        if len(preds[n]) >= 2:
            for p in preds[n]:
                runner = p
                while runner != idom.get(n):
                    df[runner].add(n)
                    runner = idom.get(runner)
                    if runner is None:
                        break

    print("2. Dominance frontiers:")
    for n in all_nodes:
        frontier = df.get(n, set())
        print(f"  DF({n}) = {{{', '.join(sorted(frontier))}}}")
    print()

    print("Interpretation:")
    print("  DF(B2) = {B4}: B2 dominates the path to B4 but B4 has another predecessor (B3)")
    print("  DF(B3) = {B4}: similarly")
    print("  DF(B5) = {B1}: B5 goes back to B1, which has another predecessor (Entry)")
    print("  DF tells us where phi functions are needed when placing SSA phi nodes.")


# === Exercise 6: Implementation Challenge ===
# Problem: Extend TAC generator for function calls and array accesses.

class ExtendedTACGenerator:
    """TAC generator with function calls and array accesses."""

    def __init__(self):
        self.temp_count = 0
        self.label_count = 0
        self.instructions = []

    def new_temp(self):
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self):
        self.label_count += 1
        return f"L{self.label_count}"

    def emit(self, instr):
        self.instructions.append(instr)

    def gen_expr(self, expr):
        if isinstance(expr, (int, float)):
            return str(expr)
        if isinstance(expr, str):
            return expr
        if isinstance(expr, tuple):
            op = expr[0]
            if op in ('+', '-', '*', '/', '<', '>', '==', '!='):
                left = self.gen_expr(expr[1])
                right = self.gen_expr(expr[2])
                t = self.new_temp()
                self.emit(f"{t} = {left} {op} {right}")
                return t
            elif op == 'call':
                # ('call', func_name, [arg1, arg2, ...])
                func_name = expr[1]
                args = expr[2]
                # Push parameters
                for arg in args:
                    arg_val = self.gen_expr(arg)
                    self.emit(f"param {arg_val}")
                t = self.new_temp()
                self.emit(f"{t} = call {func_name}, {len(args)}")
                return t
            elif op == 'array_access':
                # ('array_access', array_name, index_expr)
                arr = expr[1]
                idx = self.gen_expr(expr[2])
                t = self.new_temp()
                self.emit(f"{t} = {arr}[{idx}]")
                return t
        return str(expr)

    def gen_stmt(self, stmt):
        if stmt[0] == 'assign':
            _, var, expr = stmt
            result = self.gen_expr(expr)
            self.emit(f"{var} = {result}")
        elif stmt[0] == 'array_store':
            # ('array_store', array_name, index_expr, value_expr)
            _, arr, idx_expr, val_expr = stmt
            idx = self.gen_expr(idx_expr)
            val = self.gen_expr(val_expr)
            self.emit(f"{arr}[{idx}] = {val}")
        elif stmt[0] == 'if':
            _, cond, then_stmts, else_stmts = stmt
            cond_result = self.gen_expr(cond)
            else_label = self.new_label()
            end_label = self.new_label()
            self.emit(f"iffalse {cond_result} goto {else_label}")
            for s in then_stmts:
                self.gen_stmt(s)
            self.emit(f"goto {end_label}")
            self.emit(f"{else_label}:")
            for s in else_stmts:
                self.gen_stmt(s)
            self.emit(f"{end_label}:")
        elif stmt[0] == 'while':
            _, cond, body = stmt
            loop_label = self.new_label()
            end_label = self.new_label()
            self.emit(f"{loop_label}:")
            cond_result = self.gen_expr(cond)
            self.emit(f"iffalse {cond_result} goto {end_label}")
            for s in body:
                self.gen_stmt(s)
            self.emit(f"goto {loop_label}")
            self.emit(f"{end_label}:")
        elif stmt[0] == 'return':
            result = self.gen_expr(stmt[1])
            self.emit(f"return {result}")
        elif stmt[0] == 'call_stmt':
            # Procedure call (discard result)
            func_name = stmt[1]
            args = stmt[2]
            for arg in args:
                arg_val = self.gen_expr(arg)
                self.emit(f"param {arg_val}")
            self.emit(f"call {func_name}, {len(args)}")


def exercise_6():
    """Extended TAC generator with function calls and array accesses."""
    gen = ExtendedTACGenerator()

    # Program: sum of an array
    # function sum_array(a, n):
    #     s = 0
    #     i = 0
    #     while (i < n):
    #         s = s + a[i]
    #         i = i + 1
    #     return s

    print("Source: function to compute sum of an array")
    print("  function sum_array(a, n):")
    print("      s = 0")
    print("      i = 0")
    print("      while (i < n):")
    print("          s = s + a[i]")
    print("          i = i + 1")
    print("      return s")
    print()

    program = [
        ('assign', 's', 0),
        ('assign', 'i', 0),
        ('while', ('<', 'i', 'n'), [
            ('assign', 's', ('+', 's', ('array_access', 'a', 'i'))),
            ('assign', 'i', ('+', 'i', 1)),
        ]),
        ('return', 's'),
    ]

    gen.emit("// function sum_array(a, n)")
    for stmt in program:
        gen.gen_stmt(stmt)

    print("Generated TAC:")
    for instr in gen.instructions:
        print(f"  {instr}")
    print()

    # Test function call: result = sum_array(my_array, 10)
    gen2 = ExtendedTACGenerator()
    gen2.emit("// result = sum_array(my_array, 10)")
    gen2.gen_stmt(('assign', 'result', ('call', 'sum_array', ['my_array', 10])))

    print("Function call TAC:")
    for instr in gen2.instructions:
        print(f"  {instr}")
    print()

    # Test array store: a[i] = value
    gen3 = ExtendedTACGenerator()
    gen3.emit("// Fill array with squares: a[i] = i * i")
    fill_program = [
        ('assign', 'i', 0),
        ('while', ('<', 'i', 'n'), [
            ('array_store', 'a', 'i', ('*', 'i', 'i')),
            ('assign', 'i', ('+', 'i', 1)),
        ]),
    ]
    for stmt in fill_program:
        gen3.gen_stmt(stmt)

    print("Array store TAC:")
    for instr in gen3.instructions:
        print(f"  {instr}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: TAC Generation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Basic Block Identification ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: SSA Conversion ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: DAG Construction ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Dominance Frontiers ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Implementation Challenge ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
