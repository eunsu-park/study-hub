# Compiler Design

## Topic Overview

Compiler design is one of the foundational areas of computer science, bridging the gap between human-readable programming languages and machine-executable code. A compiler translates source code written in a high-level language into a lower-level representation — typically machine code, bytecode, or another programming language — through a series of well-defined phases.

This topic covers the complete pipeline of compilation: from scanning raw characters into tokens, parsing tokens into structured representations, analyzing meaning, generating intermediate code, optimizing it, and finally producing target code. Along the way, you will encounter deep connections to formal language theory, algorithm design, graph theory, and computer architecture.

Understanding compiler design is valuable not only for building compilers but also for:

- **Language design**: Creating domain-specific languages (DSLs) and configuration languages
- **Tool building**: Writing linters, formatters, transpilers, and static analyzers
- **Performance understanding**: Knowing what optimizations your compiler performs (and what it cannot do)
- **Software engineering**: Applying patterns like visitors, interpreters, and intermediate representations to general software design
- **Security**: Understanding how code injection, sandboxing, and code analysis tools work

## Learning Objectives

By completing this topic, you will be able to:

1. Explain the phases of compilation and how they interact
2. Implement lexical analyzers using finite automata theory
3. Construct parsers using both top-down and bottom-up techniques
4. Design abstract syntax trees and perform semantic analysis
5. Generate intermediate representations and perform optimizations
6. Understand runtime environments, garbage collection, and virtual machines
7. Use modern compiler infrastructure (LLVM) for practical compilation tasks
8. Build a simple end-to-end compiler or interpreter for a small language

## Prerequisites

Before studying Compiler Design, you should be comfortable with:

- **Algorithm** (this project): Particularly graph algorithms, trees, and dynamic programming
- **Programming** (this project): Strong grasp of data structures, recursion, and OOP
- **Python** (this project): Most examples use Python for clarity
- **C_Basics** (this project): Helpful for understanding low-level code generation
- **Discrete mathematics**: Sets, relations, functions, proof techniques (induction, contradiction)
- **Basic computer architecture**: Registers, memory, instruction sets (covered in Computer_Architecture)

## Lesson Table

| # | Title | Key Topics | Estimated Time |
|---|-------|------------|----------------|
| [01](./01_Introduction_to_Compilers.md) | Introduction to Compilers | Compiler structure, phases, bootstrapping, T-diagrams | 2-3 hours |
| [02](./02_Lexical_Analysis.md) | Lexical Analysis | Tokens, regular expressions, DFA/NFA, lexer implementation | 3-4 hours |
| [03](./03_Finite_Automata.md) | Finite Automata | NFA-to-DFA conversion, minimization, Myhill-Nerode, Lex/Flex | 3-4 hours |
| [04](./04_Context_Free_Grammars.md) | Context-Free Grammars | BNF, derivations, parse trees, ambiguity, CNF, CYK | 3-4 hours |
| [05](./05_Top_Down_Parsing.md) | Top-Down Parsing | Recursive descent, LL(1), FIRST/FOLLOW sets | 3-4 hours |
| [06](./06_Bottom_Up_Parsing.md) | Bottom-Up Parsing | LR(0), SLR, LALR, parser generators (Yacc/Bison) | 4-5 hours |
| [07](./07_Abstract_Syntax_Trees.md) | Abstract Syntax Trees | AST design, visitor pattern, tree traversal strategies | 2-3 hours |
| [08](./08_Semantic_Analysis.md) | Semantic Analysis | Type checking, symbol tables, scoping rules | 3-4 hours |
| [09](./09_Intermediate_Representations.md) | Intermediate Representations | Three-address code, SSA form, control flow graphs | 3-4 hours |
| [10](./10_Runtime_Environments.md) | Runtime Environments | Activation records, calling conventions, stack frames | 3-4 hours |
| [11](./11_Code_Generation.md) | Code Generation | Instruction selection, register allocation, tiling | 3-4 hours |
| [12](./12_Optimization_Local_and_Global.md) | Optimization -- Local and Global | Data flow analysis, constant propagation, dead code elimination | 4-5 hours |
| [13](./13_Loop_Optimization.md) | Loop Optimization | Loop-invariant code motion, strength reduction, vectorization | 3-4 hours |
| [14](./14_Garbage_Collection.md) | Garbage Collection | Mark-Sweep, Copying, Generational, Reference Counting | 3-4 hours |
| [15](./15_Interpreters_and_Virtual_Machines.md) | Interpreters and Virtual Machines | Bytecode, stack-based VM, JIT compilation basics | 3-4 hours |
| [16](./16_Modern_Compiler_Infrastructure.md) | Modern Compiler Infrastructure | LLVM IR, pass structure, DSL design | 3-4 hours |
| [17](./17_Garbage_Collection.md) | Garbage Collection -- Advanced Topics | Concurrent collectors, tri-color marking, write barriers, G1/ZGC/Shenandoah, escape analysis | 3-4 hours |
| [18](./18_SSA_Form.md) | SSA Form | Static Single Assignment, phi functions, dominance frontiers, SSA construction | 4-5 hours |
| [19](./19_Register_Allocation.md) | Register Allocation | Graph coloring, linear scan, live range splitting, spilling strategies | 4-5 hours |
| [20](./20_LLVM_IR_Introduction.md) | LLVM IR Introduction | LLVM IR syntax, types, instructions, basic blocks, functions | 3-4 hours |
| [21](./21_LLVM_Backend.md) | LLVM Backend | LLVM pass pipeline, custom passes, SelectionDAG, instruction selection | 4-5 hours |
| [22](./22_JIT_Compilation.md) | JIT Compilation | JIT fundamentals, tracing JIT, method JIT, LLVM ORC, libgccjit | 3-4 hours |
| [23](./23_Type_Inference.md) | Type Inference | Hindley-Milner, Algorithm W, unification, constraint-based typing | 4-5 hours |
| [24](./24_Closure_Conversion.md) | Closure Conversion | Lambda lifting, closure conversion, defunctionalization, CPS transform | 3-4 hours |
| [25](./25_Linking_and_Loading.md) | Linking and Loading | Object files, ELF format, static/dynamic linking, symbol resolution | 3-4 hours |
| [26](./26_Debug_Information.md) | Debug Information | DWARF format, source maps, debug info generation, debugger integration | 3-4 hours |
| [27](./27_Modern_Parser_Generators.md) | Modern Parser Generators | tree-sitter, PEG, Packrat parsing, incremental parsing, LSP integration | 3-4 hours |
| [28](./28_Capstone_Compiler_Project.md) | Capstone Compiler Project | Build a complete compiler: toy language to LLVM codegen | 8-10 hours |

**Total estimated time: 98-129 hours**

## Learning Path Recommendations

### Path 1: Foundations First (Recommended)

Follow the lessons in order. This path builds each concept on the previous ones:

```
01 Introduction
    |
02 Lexical Analysis  --->  03 Finite Automata
    |
04 Context-Free Grammars
    |
    +---> 05 Top-Down Parsing
    +---> 06 Bottom-Up Parsing
    |
07 Abstract Syntax Trees
    |
08 Semantic Analysis
    |
09 Intermediate Representations
    |
    +---> 10 Runtime Environments
    +---> 11 Code Generation
    +---> 12 Local/Global Optimization
    +---> 13 Loop Optimization
    |
14 Garbage Collection
    |
15 Interpreters and VMs
    |
16 Modern Compiler Infrastructure
    |
17 Garbage Collection (Advanced)
    |
18 SSA Form
    |
    +---> 19 Register Allocation
    +---> 20 LLVM IR Introduction
    |
21 LLVM Backend
    |
22 JIT Compilation
    |
    +---> 23 Type Inference
    +---> 24 Closure Conversion
    |
25 Linking and Loading
    |
26 Debug Information
    |
27 Modern Parser Generators
    |
28 Capstone Compiler Project
```

### Path 2: Quick Practical Compiler

If you want to build a working compiler or interpreter quickly:

```
01 Introduction  -->  02 Lexical Analysis  -->  04 CFGs  -->  05 Top-Down Parsing
    -->  07 ASTs  -->  08 Semantic Analysis  -->  15 Interpreters and VMs
```

### Path 3: Optimization Focus

If you are interested in compiler optimizations and performance:

```
01 Introduction  -->  09 Intermediate Representations  -->  12 Local/Global Optimization
    -->  13 Loop Optimization  -->  11 Code Generation  -->  16 LLVM
```

### Path 4: Language Theory

If you are interested in formal language theory and parsing:

```
02 Lexical Analysis  -->  03 Finite Automata  -->  04 CFGs
    -->  05 Top-Down Parsing  -->  06 Bottom-Up Parsing
```

## Related Topics in This Project

| Topic | Relevance |
|-------|-----------|
| **Algorithm** | Graph algorithms (DFS, topological sort) used in data flow analysis and optimization |
| **OS_Theory** | Process memory layout, linking, loading — directly related to runtime environments |
| **Computer_Architecture** | Instruction sets, registers, caches — essential for code generation |
| **Programming** | Design patterns (Visitor, Interpreter) used extensively in compiler construction |
| **Python** | Implementation language for most examples in this topic |
| **C_Basics** | Target language understanding; many compilers are written in C |
| **Math_for_AI** | Linear algebra and graph theory concepts overlap with optimization algorithms |

## Recommended Textbooks and References

1. **Aho, Lam, Sethi, Ullman** — *Compilers: Principles, Techniques, and Tools* (2nd ed., "The Dragon Book")
2. **Cooper, Torczon** — *Engineering a Compiler* (2nd ed.)
3. **Appel** — *Modern Compiler Implementation in ML/Java/C*
4. **Muchnick** — *Advanced Compiler Design and Implementation*
5. **Grune, Bal, Jacobs, Langendoen** — *Modern Compiler Design* (2nd ed.)

## Example Code

Example code for this topic can be found in [`examples/Compiler_Design/`](../../../examples/Compiler_Design/).

Examples include:
- Lexer implementations (Python)
- Parser implementations (recursive descent, operator precedence)
- AST construction and traversal
- Simple interpreter and bytecode VM
- Optimization passes

---

*This topic is part of the [Study Materials](../../README.md) collection.*
