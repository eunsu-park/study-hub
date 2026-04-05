# Debug Information

**Previous**: [25. Linking and Loading](./25_Linking_and_Loading.md) | **Next**: [27. Modern Parser Generators](./27_Modern_Parser_Generators.md)

---

When a program crashes or behaves unexpectedly, developers rely on debuggers to inspect variables, set breakpoints, and step through source code. But the machine executes binary instructions, not source code. Debug information bridges this gap by mapping machine code back to source locations, variable names, types, and scoping rules. Compilers must generate this metadata carefully, and understanding its format is essential for building debugging tools, profilers, and crash analysis systems.

This lesson covers the DWARF debug format, source maps for web languages, how compilers generate debug information, and how debuggers consume it.

**Difficulty**: ⭐⭐⭐

**Prerequisites**: [11. Code Generation](./11_Code_Generation.md), [25. Linking and Loading](./25_Linking_and_Loading.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why debug information is needed and how it relates to the source
2. Describe the DWARF debug information format and its key structures
3. Understand line number tables that map addresses to source locations
4. Explain how variable location descriptions work
5. Generate debug metadata in LLVM IR
6. Understand source maps for JavaScript and web languages
7. Describe how debuggers use debug information for breakpoints and stepping

---

## Table of Contents

1. [Why Debug Information?](#1-why-debug-information)
2. [DWARF Format Overview](#2-dwarf-format-overview)
3. [Compilation Units and DIEs](#3-compilation-units-and-dies)
4. [Line Number Tables](#4-line-number-tables)
5. [Variable Locations](#5-variable-locations)
6. [Debug Info in LLVM](#6-debug-info-in-llvm)
7. [Source Maps](#7-source-maps)
8. [Debugger Integration](#8-debugger-integration)
9. [Debug Info and Optimization](#9-debug-info-and-optimization)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Why Debug Information?

### 1.1 The Problem

Machine code has no concept of variable names, source lines, or types:

```c
// Source code (what the developer sees):
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Machine code (what executes):
// 0x4005a0: push rbp
// 0x4005a1: mov rbp, rsp
// 0x4005a4: mov [rbp-4], edi
// 0x4005a7: cmp dword [rbp-4], 1
// ...
```

Debug information answers:
- Which source file and line corresponds to address `0x4005a7`?
- What is the value of variable `n` at this point?
- What is the type of `n`?

### 1.2 Generating Debug Info

```bash
# Compile with debug info
gcc -g program.c -o program           # Full debug info (DWARF)
gcc -g -O2 program.c -o program       # Debug info + optimization
gcc -g1 program.c -o program          # Minimal (line tables only)
gcc -g3 program.c -o program          # Maximum (includes macros)

# Strip debug info
strip --strip-debug program
# or separate it:
objcopy --only-keep-debug program program.debug
objcopy --add-gnu-debuglink=program.debug program
```

### 1.3 Debug Info Formats

| Format | Platforms | Used By |
|--------|-----------|---------|
| **DWARF** | Linux, macOS, BSD | GCC, Clang, GDB, LLDB |
| **PDB** | Windows | MSVC, Visual Studio |
| **CodeView** | Windows | MSVC (legacy) |
| **Stabs** | Legacy Unix | Older GCC (deprecated) |
| **Source Maps** | Web | JavaScript, TypeScript |

---

## 2. DWARF Format Overview

### 2.1 What is DWARF?

DWARF (Debugging With Attributed Record Formats) is the standard debug information format for ELF binaries. It is stored in special sections of the object file:

| Section | Content |
|---------|---------|
| `.debug_info` | Type, variable, and function descriptions (DIEs) |
| `.debug_abbrev` | Abbreviation tables (templates for DIEs) |
| `.debug_line` | Line number table (address → source line) |
| `.debug_str` | String table for debug info |
| `.debug_loc` | Variable location lists |
| `.debug_ranges` | Address ranges for scopes |
| `.debug_frame` | Call frame information (for unwinding) |

### 2.2 Inspecting DWARF

```bash
# View all debug info
readelf --debug-dump=info program

# View line number table
readelf --debug-dump=line program

# Using dwarfdump (more detailed)
dwarfdump program

# Using llvm-dwarfdump
llvm-dwarfdump program

# View only specific DIE types
llvm-dwarfdump --name=factorial program
```

---

## 3. Compilation Units and DIEs

### 3.1 Debug Information Entries (DIEs)

DWARF represents debug information as a tree of **DIEs** (Debug Information Entries). Each DIE has a **tag** (type of entity) and **attributes** (properties):

```
DW_TAG_compile_unit (root)
  DW_AT_name: "main.c"
  DW_AT_language: DW_LANG_C11
  DW_AT_producer: "clang 17.0"
  │
  ├── DW_TAG_subprogram          (function)
  │     DW_AT_name: "factorial"
  │     DW_AT_type: <ref to int>
  │     DW_AT_low_pc: 0x4005a0
  │     DW_AT_high_pc: 0x4005d0
  │     │
  │     ├── DW_TAG_formal_parameter
  │     │     DW_AT_name: "n"
  │     │     DW_AT_type: <ref to int>
  │     │     DW_AT_location: DW_OP_fbreg -20
  │     │
  │     └── DW_TAG_lexical_block   (scope)
  │           DW_AT_low_pc: 0x4005b0
  │           DW_AT_high_pc: 0x4005c0
  │
  ├── DW_TAG_base_type            (type)
  │     DW_AT_name: "int"
  │     DW_AT_encoding: DW_ATE_signed
  │     DW_AT_byte_size: 4
  │
  └── DW_TAG_variable             (global variable)
        DW_AT_name: "global_count"
        DW_AT_type: <ref to int>
        DW_AT_location: DW_OP_addr 0x601000
```

### 3.2 Common DIE Tags

| Tag | Describes |
|-----|-----------|
| `DW_TAG_compile_unit` | A source file |
| `DW_TAG_subprogram` | A function |
| `DW_TAG_variable` | A variable |
| `DW_TAG_formal_parameter` | A function parameter |
| `DW_TAG_base_type` | Primitive type (int, float, etc.) |
| `DW_TAG_pointer_type` | Pointer type |
| `DW_TAG_structure_type` | Struct/class |
| `DW_TAG_array_type` | Array type |
| `DW_TAG_lexical_block` | Scope block |
| `DW_TAG_inlined_subroutine` | Inlined function body |

---

## 4. Line Number Tables

### 4.1 Purpose

The line number table maps machine code addresses to source file locations:

```
Address         File        Line    Column
0x4005a0        main.c      5       1
0x4005a4        main.c      6       5
0x4005a7        main.c      6       8
0x4005b0        main.c      7       5
```

### 4.2 State Machine Encoding

DWARF encodes line tables as a sequence of opcodes for a virtual state machine:

```python
class LineNumberStateMachine:
    """DWARF line number state machine."""

    def __init__(self):
        self.address = 0
        self.file = 1
        self.line = 1
        self.column = 0
        self.is_stmt = True
        self.basic_block = False
        self.end_sequence = False

    def execute_opcode(self, opcode, operand=None):
        if opcode == DW_LNS_advance_pc:
            self.address += operand
        elif opcode == DW_LNS_advance_line:
            self.line += operand
        elif opcode == DW_LNS_set_file:
            self.file = operand
        elif opcode == DW_LNS_set_column:
            self.column = operand
        elif opcode == DW_LNS_copy:
            self.emit_row()
        elif opcode == DW_LNE_end_sequence:
            self.end_sequence = True
            self.emit_row()
            self.reset()

    def emit_row(self):
        """Record current state as a line table entry."""
        print(f"0x{self.address:08x}  file={self.file}  "
              f"line={self.line}  col={self.column}")
```

### 4.3 Special Opcodes

For compact encoding, DWARF uses "special opcodes" that advance both address and line in a single byte:

```python
def decode_special_opcode(opcode, opcode_base, line_base, line_range, min_insn_length):
    """
    Special opcodes encode address+line advance in one byte.
    """
    adjusted = opcode - opcode_base
    line_advance = line_base + (adjusted % line_range)
    addr_advance = (adjusted // line_range) * min_insn_length
    return addr_advance, line_advance
```

---

## 5. Variable Locations

### 5.1 The Location Problem

Due to optimization, a variable's storage changes throughout execution:

```c
int x = compute();    // x in register rax
use(x);
// ... more code ...  // x spilled to stack [rbp-8]
use(x);
// ... more code ...  // x optimized away (dead)
```

### 5.2 DWARF Location Expressions

DWARF uses a stack-based expression language to describe locations:

```
DW_OP_reg0          -- value is in register 0 (rax on x86-64)
DW_OP_breg6 -20     -- value is at [rbp - 20] (rbp is reg 6)
DW_OP_fbreg -8      -- value is at [frame_base - 8]
DW_OP_addr 0x601000 -- value is at absolute address
DW_OP_lit5          -- value is the constant 5

# Complex expression: *(rbp - 16) + 4
DW_OP_breg6 -16     -- push [rbp - 16]
DW_OP_deref          -- dereference
DW_OP_lit4           -- push 4
DW_OP_plus           -- add
```

### 5.3 Location Lists

When a variable moves between locations, DWARF uses a **location list**:

```
Variable "x":
  [0x4005a0, 0x4005b0): DW_OP_reg0        -- x in rax
  [0x4005b0, 0x4005d0): DW_OP_fbreg -8    -- x on stack
  [0x4005d0, 0x4005e0): <empty>            -- x optimized out
```

---

## 6. Debug Info in LLVM

### 6.1 LLVM Debug Metadata

LLVM IR represents debug information using metadata:

```llvm
define i32 @factorial(i32 %n) !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !10, metadata !DIExpression()), !dbg !11
  %cmp = icmp sle i32 %n, 1, !dbg !12
  ; ...
}

!5 = distinct !DISubprogram(name: "factorial", file: !6, line: 3,
       type: !7, scopeLine: 3, unit: !0)
!6 = !DIFile(filename: "main.c", directory: "/home/user")
!10 = !DILocalVariable(name: "n", arg: 1, scope: !5, file: !6,
        line: 3, type: !9)
!11 = !DILocation(line: 3, column: 20, scope: !5)
!12 = !DILocation(line: 4, column: 8, scope: !5)
```

### 6.2 Generating Debug Info with llvmlite

```python
from llvmlite import ir

def build_with_debug_info():
    module = ir.Module(name="debug_example")

    # Create debug info
    di_file = module.add_debug_info("DIFile", {
        "filename": "example.c",
        "directory": "/home/user",
    })

    di_compile_unit = module.add_debug_info("DICompileUnit", {
        "language": ir.DIToken("DW_LANG_C"),
        "file": di_file,
        "producer": "my_compiler 1.0",
        "isOptimized": False,
    }, is_distinct=True)

    di_int = module.add_debug_info("DIBasicType", {
        "name": "int",
        "size": 32,
        "encoding": ir.DIToken("DW_ATE_signed"),
    })

    di_func_type = module.add_debug_info("DISubroutineType", {
        "types": module.add_metadata([di_int, di_int]),
    })

    # Create function with debug info
    i32 = ir.IntType(32)
    func = ir.Function(module, ir.FunctionType(i32, [i32]), name="square")

    di_func = module.add_debug_info("DISubprogram", {
        "name": "square",
        "file": di_file,
        "line": 1,
        "type": di_func_type,
        "scopeLine": 1,
        "unit": di_compile_unit,
    }, is_distinct=True)
    func.set_metadata("dbg", di_func)

    block = func.append_basic_block("entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = module.add_debug_info("DILocation", {
        "line": 2, "column": 5, "scope": di_func,
    })

    result = builder.mul(func.args[0], func.args[0])
    builder.ret(result)

    return module
```

---

## 7. Source Maps

### 7.1 What are Source Maps?

Source maps serve the same purpose as DWARF but for web languages. They map minified/transpiled JavaScript back to original TypeScript, JSX, or other source:

```
Original (TypeScript):        Compiled (JavaScript):
function greet(name) {        function a(b){return"Hello "+b}
  return "Hello " + name;
}
```

### 7.2 Source Map Format

```json
{
  "version": 3,
  "file": "output.js",
  "sourceRoot": "",
  "sources": ["input.ts"],
  "names": ["greet", "name"],
  "mappings": "AAAA,SAASA,MAAKC,GAAQ,MAAO,SAAWA"
}
```

### 7.3 VLQ Encoding

Source maps use Variable-Length Quantity (VLQ) encoding for compact mappings:

```python
def encode_vlq(value):
    """Encode an integer as a VLQ base64 string."""
    VLQ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

    # Handle sign
    if value < 0:
        vlq = ((-value) << 1) | 1
    else:
        vlq = value << 1

    result = ""
    while True:
        digit = vlq & 0x1F  # 5 bits
        vlq >>= 5
        if vlq > 0:
            digit |= 0x20   # continuation bit
        result += VLQ_CHARS[digit]
        if vlq == 0:
            break

    return result
```

---

## 8. Debugger Integration

### 8.1 How Breakpoints Work

```
1. User sets breakpoint at line 10
2. Debugger looks up line 10 in line table → address 0x4005b0
3. Debugger replaces instruction at 0x4005b0 with INT 3 (0xCC)
4. When execution hits 0xCC, CPU raises SIGTRAP
5. Debugger catches SIGTRAP
6. Debugger restores original instruction
7. User inspects variables using DWARF location info
8. Debugger single-steps past the breakpoint, reinserts INT 3
```

### 8.2 Single Stepping

Line-level stepping uses the line table to determine where each source line begins:

```python
def step_over_line(debugger, current_addr):
    """Step to the next source line."""
    current_line = debugger.line_table.addr_to_line(current_addr)

    while True:
        debugger.single_step_instruction()
        new_addr = debugger.get_pc()
        new_line = debugger.line_table.addr_to_line(new_addr)
        if new_line != current_line:
            break
```

### 8.3 Stack Unwinding

The `.debug_frame` (or `.eh_frame`) section contains call frame information for unwinding the stack:

```
CIE (Common Information Entry):
  Code alignment: 1
  Data alignment: -8
  Return register: 16 (rip)

FDE (Frame Description Entry) for factorial:
  Address range: [0x4005a0, 0x4005d0)
  Instructions:
    DW_CFA_def_cfa: r7 (rsp) offset 8
    DW_CFA_offset: r16 (rip) at CFA-8
    advance 1 to 0x4005a1
    DW_CFA_def_cfa_offset: 16
    DW_CFA_offset: r6 (rbp) at CFA-16
```

---

## 9. Debug Info and Optimization

### 9.1 The Challenge

Optimizations rearrange, eliminate, and transform code, making it hard to maintain accurate debug info:

| Optimization | Debug Impact |
|-------------|--------------|
| Dead code elimination | Variable "optimized out" |
| Register allocation | Variable location changes |
| Inlining | Function "disappears," appears in caller |
| Loop unrolling | One source line maps to many copies |
| Instruction reordering | Non-linear stepping |

### 9.2 Strategies

- **Maintain metadata through transforms**: LLVM propagates `!dbg` metadata on instructions
- **Location lists**: Track variable movement across registers and stack slots
- **Inlined subroutine DIEs**: Record inlining decisions in DWARF
- **Is_stmt flag**: Mark which addresses are good breakpoint candidates

### 9.3 Debug Level Tradeoffs

```bash
-g0  # No debug info (smallest binary, no debugging)
-g1  # Line tables only (backtraces work, no variable inspection)
-g   # Full debug info (default: types, variables, scopes)
-g3  # Full + macros (largest, can expand macros in debugger)
```

---

## 10. Summary

- **Debug information** maps machine code back to source code for debugging
- **DWARF** is the standard format, stored in `.debug_*` ELF sections
- **DIEs** form a tree describing types, functions, variables, and scopes
- **Line number tables** map code addresses to source file/line/column
- **Location expressions** describe where variables are stored (register, stack, memory)
- **LLVM** generates debug info through IR metadata (`!dbg`, `!DILocation`, etc.)
- **Source maps** provide equivalent functionality for JavaScript/TypeScript
- Optimizations complicate debug info; location lists and inlined subroutine DIEs help

---

## 11. Exercises

1. **Inspect DWARF**: Compile a C program with `-g` and explore its DWARF information using `readelf --debug-dump` and `llvm-dwarfdump`.

2. **Line table walkthrough**: Trace the line number state machine for a small function and verify the address-to-line mapping.

3. **Variable locations**: Compile with `-g -O2` and examine how variable locations change across the function using `llvm-dwarfdump --name=<variable>`.

4. **Source map**: Write a simple transpiler (e.g., CoffeeScript-like to JS) that generates a source map.

5. **Debug info generation**: Extend a simple compiler to emit LLVM debug metadata for functions and variables.

---

## 12. References

1. DWARF Debugging Standard Committee. *DWARF Debugging Information Format Version 5*. https://dwarfstd.org/
2. Eager, M. J. (2012). "Introduction to the DWARF Debugging Format." https://dwarfstd.org/doc/Debugging%20using%20DWARF-2012.pdf
3. Source Map Specification v3: https://sourcemaps.info/spec.html
4. LLVM Source Level Debugging: https://llvm.org/docs/SourceLevelDebugging.html
5. Eli Bendersky. "How debuggers work." https://eli.thegreenplace.net/tag/debuggers

---

**Previous**: [25. Linking and Loading](./25_Linking_and_Loading.md) | **Next**: [27. Modern Parser Generators](./27_Modern_Parser_Generators.md)
