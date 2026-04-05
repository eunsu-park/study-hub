"""
Exercises for Lesson 22: JIT Compilation
Topic: Compiler_Design

Solutions demonstrating JIT concepts.
"""

import time
from typing import Callable, Dict, List, Any


# === Exercise 1: JIT Threshold Experiment ===

def exercise_1():
    """Experiment with JIT compilation thresholds."""
    print("Exercise 1: JIT Threshold Analysis")
    print()

    def interpreted_add(a, b):
        return a + b

    compile_cost_ms = 5.0
    interpret_cost_per_call_ms = 0.01
    compiled_cost_per_call_ms = 0.001

    print("Analysis: At what invocation count does JIT pay off?")
    print(f"  Compile cost: {compile_cost_ms} ms")
    print(f"  Interpret per call: {interpret_cost_per_call_ms} ms")
    print(f"  Compiled per call: {compiled_cost_per_call_ms} ms")
    print()

    savings_per_call = interpret_cost_per_call_ms - compiled_cost_per_call_ms
    breakeven = compile_cost_ms / savings_per_call
    print(f"  Savings per call: {savings_per_call} ms")
    print(f"  Break-even after: {breakeven:.0f} calls after compilation")
    print()

    for threshold in [10, 100, 500, 1000, 5000]:
        total_calls = 10000
        interpret_calls = min(threshold, total_calls)
        compiled_calls = max(0, total_calls - threshold)
        total_time = (interpret_calls * interpret_cost_per_call_ms
                      + compile_cost_ms + compiled_calls * compiled_cost_per_call_ms)
        pure_interpret = total_calls * interpret_cost_per_call_ms
        print(f"  Threshold={threshold:5d}: "
              f"JIT={total_time:.1f}ms, Interpret={pure_interpret:.1f}ms, "
              f"Speedup={pure_interpret/total_time:.2f}x")
    print()


# === Exercise 2: Simple Expression JIT ===

def exercise_2():
    """Build a JIT that compiles arithmetic expressions."""
    print("Exercise 2: Expression JIT Compiler")
    print()

    def compile_expr(expr_str):
        """Compile an arithmetic expression to a Python function."""
        code = f"def _expr(x): return {expr_str}"
        namespace = {}
        exec(code, namespace)
        return namespace['_expr']

    expressions = [
        ("x * x + 2 * x + 1", 5),
        ("x * x * x", 3),
        ("(x + 1) * (x - 1)", 10),
    ]

    for expr, test_val in expressions:
        compiled = compile_expr(expr)
        result = compiled(test_val)
        print(f"  Expression: {expr}")
        print(f"  f({test_val}) = {result}")
        print()


# === Exercise 3: Simple Trace Recorder ===

class BytecodeVM:
    """Simple bytecode interpreter with trace recording."""

    CONST = 'CONST'
    LOAD = 'LOAD'
    STORE = 'STORE'
    ADD = 'ADD'
    MUL = 'MUL'
    CMP_LT = 'CMP_LT'
    JUMP_IF = 'JUMP_IF'
    JUMP = 'JUMP'
    PRINT = 'PRINT'
    HALT = 'HALT'

    def __init__(self, program):
        self.program = program
        self.pc = 0
        self.stack = []
        self.vars = {}
        self.trace = None
        self.recording = False

    def record_trace(self, start_pc, max_iters=2):
        """Record a trace starting at start_pc."""
        self.pc = start_pc
        self.recording = True
        self.trace = []
        iterations = 0

        while iterations < max_iters:
            instr = self.program[self.pc]
            self.trace.append((self.pc, instr))

            if instr[0] == self.JUMP and instr[1] == start_pc:
                iterations += 1
                if iterations >= max_iters:
                    break

            self.pc += 1

        self.recording = False
        return self.trace


def exercise_3():
    """Record a trace through a loop."""
    print("Exercise 3: Trace Recording")
    print()

    vm = BytecodeVM([
        ('CONST', 0),       # 0: push 0
        ('STORE', 'sum'),   # 1: sum = 0
        ('CONST', 0),       # 2: push 0
        ('STORE', 'i'),     # 3: i = 0
        # Loop header (pc=4):
        ('LOAD', 'i'),      # 4: push i
        ('CONST', 10),      # 5: push 10
        ('CMP_LT',),        # 6: i < 10
        ('JUMP_IF', 12),    # 7: if false, jump to 12
        ('LOAD', 'sum'),    # 8: push sum
        ('LOAD', 'i'),      # 9: push i
        ('ADD',),           # 10: sum + i
        ('STORE', 'sum'),   # 11: sum = sum + i
        ('LOAD', 'i'),      # 12: push i  (correction: this should be i increment)
        ('CONST', 1),       # 13: push 1
        ('ADD',),           # 14: i + 1
        ('STORE', 'i'),     # 15: i = i + 1
        ('JUMP', 4),        # 16: goto loop header
        ('HALT',),          # 17
    ])

    trace = vm.record_trace(4, max_iters=1)
    print("Recorded trace (1 iteration from pc=4):")
    for pc, instr in trace:
        print(f"  [{pc:2d}] {instr}")
    print()
    print(f"Trace length: {len(trace)} instructions")
    print("A JIT would compile this linear trace to native code,")
    print("inserting guards at branch points.")
    print()


# === Exercise 4: OSR Simulation ===

def exercise_4():
    """Simulate on-stack replacement."""
    print("Exercise 4: OSR Simulation")
    print()

    def interpreted_loop(n):
        total = 0
        for i in range(n):
            total += i
            if i == 100:
                print(f"  OSR trigger at i={i}, total={total}")
                return compiled_continuation(i, total, n)
        return total

    def compiled_continuation(start_i, start_total, n):
        """Simulates compiled code picking up from OSR point."""
        print(f"  Compiled code: resume from i={start_i}, total={start_total}")
        total = start_total
        for i in range(start_i + 1, n):
            total += i
        return total

    result = interpreted_loop(1000)
    print(f"  Final result: {result}")
    print(f"  Expected: {sum(range(1000))}")
    print()


# === Exercise 5: Inline Cache ===

class InlineCache:
    """Monomorphic inline cache for dynamic dispatch."""

    def __init__(self):
        self.cached_type = None
        self.cached_method = None
        self.hits = 0
        self.misses = 0

    def call(self, obj, method_name):
        obj_type = type(obj)
        if obj_type == self.cached_type:
            self.hits += 1
            return self.cached_method(obj)
        else:
            self.misses += 1
            method = getattr(obj, method_name)
            self.cached_type = obj_type
            self.cached_method = lambda o: getattr(o, method_name)()
            return method()


def exercise_5():
    """Implement monomorphic inline cache."""
    print("Exercise 5: Inline Cache")
    print()

    class Dog:
        def speak(self): return "Woof"

    class Cat:
        def speak(self): return "Meow"

    cache = InlineCache()

    # Monomorphic: same type
    dogs = [Dog() for _ in range(5)]
    for d in dogs:
        result = cache.call(d, 'speak')

    print(f"After 5 Dog.speak() calls:")
    print(f"  Hits: {cache.hits}, Misses: {cache.misses}")

    # Polymorphic: different type invalidates cache
    cat = Cat()
    cache.call(cat, 'speak')
    print(f"\nAfter calling Cat.speak():")
    print(f"  Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"  Cache invalidation: Dog -> Cat")
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
