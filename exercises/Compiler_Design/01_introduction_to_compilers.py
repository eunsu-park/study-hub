"""
Exercises for Lesson 01: Introduction to Compilers
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

import ast
import dis


# === Exercise 1: Phase Identification ===
# Problem: For the program:
#   float area;
#   float radius = 5.0;
#   area = 3.14159 * radius * radius;
# Identify what each compiler phase would do. Write down the output of each phase.

def exercise_1():
    """Trace a simple program through every compilation phase."""
    print("Source program:")
    print("  float area;")
    print("  float radius = 5.0;")
    print("  area = 3.14159 * radius * radius;")
    print()

    # Phase 1: Lexical Analysis (Tokenization)
    tokens = [
        ("KW_FLOAT", "float"), ("ID", "area"), ("SEMI", ";"),
        ("KW_FLOAT", "float"), ("ID", "radius"), ("ASSIGN", "="),
        ("FLOAT_LIT", "5.0"), ("SEMI", ";"),
        ("ID", "area"), ("ASSIGN", "="),
        ("FLOAT_LIT", "3.14159"), ("MUL", "*"), ("ID", "radius"),
        ("MUL", "*"), ("ID", "radius"), ("SEMI", ";"),
    ]
    print("Phase 1 - Lexical Analysis (Tokens):")
    for tok_type, lexeme in tokens:
        print(f"  <{tok_type}, {lexeme!r}>")
    print()

    # Phase 2: Syntax Analysis (Parse Tree / AST)
    print("Phase 2 - Syntax Analysis (AST structure):")
    print("  Program")
    print("    VarDecl(type=float, name='area')")
    print("    VarDecl(type=float, name='radius', init=FloatLit(5.0))")
    print("    Assign(")
    print("      target=Id('area'),")
    print("      value=BinOp('*',")
    print("        BinOp('*',")
    print("          FloatLit(3.14159),")
    print("          Id('radius')),")
    print("        Id('radius')))")
    print()

    # Phase 3: Semantic Analysis (Type Checking)
    print("Phase 3 - Semantic Analysis (Type Checking):")
    print("  'area' declared as float -> OK")
    print("  'radius' declared as float, init 5.0 (float) -> OK")
    print("  3.14159 (float) * radius (float) = float -> OK")
    print("  (float) * radius (float) = float -> OK")
    print("  area (float) = (float) -> OK")
    print("  All types consistent.")
    print()

    # Phase 4: Intermediate Code Generation (Three-Address Code)
    print("Phase 4 - Intermediate Code (Three-Address Code):")
    tac = [
        "radius = 5.0",
        "t1 = 3.14159 * radius",
        "t2 = t1 * radius",
        "area = t2",
    ]
    for line in tac:
        print(f"  {line}")
    print()

    # Phase 5: Optimization
    print("Phase 5 - Optimized Code:")
    optimized = [
        "radius = 5.0",
        "t1 = 3.14159 * 5.0     // constant propagation",
        "t2 = 15.70795 * 5.0    // constant folding",
        "area = 78.53975        // constant folding",
    ]
    for line in optimized:
        print(f"  {line}")
    print()

    # Phase 6: Code Generation (pseudo-assembly)
    print("Phase 6 - Code Generation (x86-like pseudo-assembly):")
    asm = [
        "MOVSS  xmm0, [const_5.0]",
        "MOVSS  [rbp-8], xmm0          ; radius = 5.0",
        "MOVSS  xmm0, [const_78.53975]",
        "MOVSS  [rbp-4], xmm0          ; area = 78.53975",
    ]
    for line in asm:
        print(f"  {line}")


# === Exercise 2: Compiler vs. Interpreter ===
# Problem: For each scenario, explain whether a compiler, interpreter, or hybrid approach
# would be most appropriate and why.

def exercise_2():
    """Analyze when to use compiler, interpreter, or hybrid."""
    scenarios = {
        "1. Scripting language for system admin tasks": {
            "recommendation": "Interpreter (or hybrid with bytecode compilation)",
            "reasoning": (
                "System admin scripts need rapid iteration and quick startup. "
                "An interpreter allows running scripts immediately without a separate "
                "compile step. Dynamic typing is common in admin scripts. "
                "Example: Python, Bash, Perl -- all interpreted or hybrid."
            ),
        },
        "2. Language for high-performance game engines": {
            "recommendation": "Compiler (ahead-of-time, native code)",
            "reasoning": (
                "Game engines require maximum performance: tight loops, low latency, "
                "predictable memory usage. Ahead-of-time compilation to native code "
                "(with aggressive optimization) is essential. "
                "Example: C++, Rust -- compiled to native machine code."
            ),
        },
        "3. Configuration language for build rules": {
            "recommendation": "Interpreter",
            "reasoning": (
                "Build configuration files are declarative and read once per build. "
                "They are typically small, and execution speed is irrelevant compared "
                "to the actual build steps. An interpreter is simplest. "
                "Example: Makefile, CMake, Meson -- interpreted."
            ),
        },
        "4. Language running in a web browser (sandboxed)": {
            "recommendation": "Hybrid (bytecode + JIT compilation)",
            "reasoning": (
                "Web code must be portable (runs on any platform), sandboxed for "
                "security, and still performant for interactive apps. A bytecode VM "
                "with JIT compilation provides portability (bytecode is platform-"
                "independent) with near-native speed for hot paths. "
                "Example: JavaScript with V8 (interpreter + TurboFan JIT), "
                "WebAssembly (compiled to bytecode, then JIT'd)."
            ),
        },
    }

    for scenario, info in scenarios.items():
        print(f"{scenario}")
        print(f"  Recommendation: {info['recommendation']}")
        print(f"  Reasoning: {info['reasoning']}")
        print()


# === Exercise 3: T-Diagrams ===
# Problem: Draw T-diagrams for given scenarios with a Python interpreter in C on x86
# and a C compiler in C targeting x86.

def exercise_3():
    """T-diagrams for compiler composition scenarios."""
    print("Given:")
    print("  - Python interpreter written in C, running on x86")
    print("  - C compiler written in C, targeting x86")
    print()

    print("Scenario 1: Run a Python program on x86")
    print("=" * 50)
    print("  T-diagram composition:")
    print()
    print("  ┌─────────────────┐")
    print("  │   Python prog   │   (source: Python)")
    print("  │  Python → [out] │")
    print("  └───────┬─────────┘")
    print("          │ runs on")
    print("  ┌───────┴─────────┐")
    print("  │  Python interp  │   (interpreter)")
    print("  │   C → x86       │")
    print("  └───────┬─────────┘")
    print("          │ compiled by")
    print("  ┌───────┴─────────┐")
    print("  │    C compiler   │")
    print("  │  C → x86 in C   │")
    print("  └───────┬─────────┘")
    print("          │ runs on")
    print("      [ x86 machine ]")
    print()

    print("Scenario 2: Compile the C compiler using itself (self-compilation)")
    print("=" * 50)
    print("  ┌─────────────────┐")
    print("  │  C compiler src │   (source: C)")
    print("  │    C → x86      │")
    print("  └───────┬─────────┘")
    print("          │ compiled by")
    print("  ┌───────┴─────────┐")
    print("  │  C compiler     │   (existing binary on x86)")
    print("  │  C → x86  [x86] │")
    print("  └───────┬─────────┘")
    print("          │ produces")
    print("  ┌───────┴─────────┐")
    print("  │  C compiler     │   (new binary on x86)")
    print("  │  C → x86  [x86] │")
    print("  └─────────────────┘")
    print()

    print("Scenario 3: Create a cross-compiler (runs on x86, targets ARM)")
    print("=" * 50)
    print("  Step 1: Write a C→ARM backend in C")
    print("  Step 2: Compile it with the existing C→x86 compiler on x86")
    print()
    print("  ┌──────────────────┐")
    print("  │  C cross-compiler│   (source in C, generates ARM code)")
    print("  │    C → ARM       │")
    print("  └───────┬──────────┘")
    print("          │ compiled by")
    print("  ┌───────┴──────────┐")
    print("  │   C compiler     │   (runs on x86, targets x86)")
    print("  │  C → x86  [x86]  │")
    print("  └───────┬──────────┘")
    print("          │ produces")
    print("  ┌───────┴──────────┐")
    print("  │  Cross compiler  │   (runs on x86, targets ARM)")
    print("  │  C → ARM  [x86]  │")
    print("  └──────────────────┘")


# === Exercise 4: Bootstrapping Sequence ===
# Problem: Describe a concrete bootstrapping strategy for a new language "Nova"
# where you want to write the Nova compiler in Nova itself.

def exercise_4():
    """Bootstrapping strategy for a self-hosting compiler."""
    print("Bootstrapping Strategy for 'Nova' Language")
    print("=" * 50)
    print()

    steps = [
        (
            "Step 1: Write Nova-subset compiler in Python (or C)",
            "Implement a minimal 'Nova0' compiler in an existing language (e.g., Python).\n"
            "  Nova0 supports only the subset of Nova needed to write a compiler:\n"
            "  - Basic types (int, string, bool)\n"
            "  - Functions, if/else, while loops\n"
            "  - Structs/records, arrays\n"
            "  - File I/O (to read source and write output)\n"
            "  Target: generate C code (or assembly) from Nova0 source.\n"
            "  This is the 'seed compiler'."
        ),
        (
            "Step 2: Rewrite the compiler in Nova0",
            "Using only Nova0 features, rewrite the compiler in Nova itself.\n"
            "  Call this source 'nova_compiler.nova'.\n"
            "  Compile it using the Python-based seed compiler:\n"
            "    python nova0_compiler.py nova_compiler.nova -> nova_compiler_v1 (binary)\n"
            "  Now nova_compiler_v1 is a Nova compiler written in Nova, compiled by Python."
        ),
        (
            "Step 3: Self-compile (first bootstrap)",
            "Use nova_compiler_v1 to compile its own source:\n"
            "    ./nova_compiler_v1 nova_compiler.nova -> nova_compiler_v2\n"
            "  nova_compiler_v2 is now a Nova compiler that was compiled by a Nova compiler.\n"
            "  Verify: nova_compiler_v1 and nova_compiler_v2 should produce identical output\n"
            "  when compiling any test program (this is the 'bootstrap comparison test')."
        ),
        (
            "Step 4: Extend the language",
            "Now add more Nova features to the compiler source.\n"
            "  Each new feature is first compiled by the previous version:\n"
            "    ./nova_compiler_v2 nova_compiler_extended.nova -> nova_compiler_v3\n"
            "  The Python seed compiler is no longer needed.\n"
            "  The compiler is now fully 'self-hosting'."
        ),
    ]

    for title, description in steps:
        print(f"{title}")
        print(f"  {description}")
        print()

    print("Minimum needed in Step 1 (seed compiler):")
    print("  - Lexer and parser for Nova0 subset")
    print("  - Type checker for basic types")
    print("  - Code generator targeting C (easiest) or native code")
    print("  - Enough language features to express a compiler (recursion, data structures, I/O)")


# === Exercise 5: Front-End / Back-End Separation ===
# Problem: Calculate compiler components needed with and without a shared IR.

def exercise_5():
    """Calculate compiler components with/without shared IR."""
    print("Front-End / Back-End Separation Analysis")
    print("=" * 50)
    print()

    # Initial: 4 languages, 3 architectures
    langs_initial = 4
    archs_initial = 3

    without_ir_initial = langs_initial * archs_initial
    with_ir_initial = langs_initial + archs_initial

    print(f"Initial: {langs_initial} languages, {archs_initial} architectures")
    print(f"  Without shared IR: {langs_initial} x {archs_initial} = {without_ir_initial} compiler components")
    print(f"  With shared IR:    {langs_initial} + {archs_initial} = {with_ir_initial} components "
          f"({langs_initial} front-ends + {archs_initial} back-ends)")
    print()

    # After adding 2 languages and 2 architectures
    langs_new = langs_initial + 2  # 6
    archs_new = archs_initial + 2  # 5

    without_ir_new = langs_new * archs_new
    with_ir_new = langs_new + archs_new

    print(f"After adding 2 languages + 2 architectures: {langs_new} languages, {archs_new} architectures")
    print(f"  Without shared IR: {langs_new} x {archs_new} = {without_ir_new} compiler components")
    print(f"  With shared IR:    {langs_new} + {archs_new} = {with_ir_new} components "
          f"({langs_new} front-ends + {archs_new} back-ends)")
    print()

    # Growth analysis
    without_growth = without_ir_new - without_ir_initial
    with_growth = with_ir_new - with_ir_initial

    print("Growth analysis:")
    print(f"  Without IR: +{without_growth} components needed (from {without_ir_initial} to {without_ir_new})")
    print(f"  With IR:    +{with_growth} components needed (from {with_ir_initial} to {with_ir_new})")
    print(f"  The shared IR approach scales linearly O(m+n) vs quadratically O(m*n).")


# === Exercise 6: Compilation Phases in Practice ===
# Problem: Use Python's ast and dis modules to explore CPython's compilation phases.

def exercise_6():
    """Explore CPython compilation phases using ast and dis modules."""
    source = "x = [i**2 for i in range(10)]"
    print(f"Source: {source!r}")
    print()

    # Phase 1: Parse to AST
    print("Phase 1: Parse to AST")
    print("-" * 40)
    tree = ast.parse(source)
    print(ast.dump(tree, indent=2))
    print()

    # Phase 2: Compile to bytecode
    print("Phase 2: Compile to bytecode object")
    print("-" * 40)
    code = compile(source, "<string>", "exec")
    print(f"  Code object: {code}")
    print(f"  Constants: {code.co_consts}")
    print(f"  Names: {code.co_names}")
    print(f"  Stack size: {code.co_stacksize}")
    print()

    # Phase 3: Disassemble
    print("Phase 3: Disassemble bytecode")
    print("-" * 40)
    dis.dis(code)
    print()

    # Analysis
    print("Analysis of visible compiler phases:")
    print("-" * 40)
    print("  1. LEXICAL ANALYSIS: Not directly visible, but ast.parse() calls the")
    print("     tokenizer internally (see tokenize module for details).")
    print()
    print("  2. SYNTAX ANALYSIS: ast.parse() produces the AST. The dump shows")
    print("     the tree structure with nodes like Module, Assign, ListComp, etc.")
    print()
    print("  3. SEMANTIC ANALYSIS: Limited in CPython (dynamic typing), but")
    print("     compile() checks for syntax errors and some semantic issues")
    print("     (e.g., 'break' outside loop, 'return' outside function).")
    print()
    print("  4. CODE GENERATION: compile() generates CPython bytecode.")
    print("     The bytecode uses a stack-based virtual machine model.")
    print()
    print("  5. OPTIMIZATIONS observed:")
    print("     - Constant folding: Small constant expressions may be precomputed")
    print("     - The list comprehension is compiled to a separate code object")
    print("     - Peephole optimizer simplifies some bytecode patterns")
    print("     Note: CPython does minimal optimization -- it prioritizes fast compilation.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Phase Identification ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Compiler vs. Interpreter ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: T-Diagrams ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Bootstrapping Sequence ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Front-End / Back-End Separation ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Compilation Phases in Practice ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
