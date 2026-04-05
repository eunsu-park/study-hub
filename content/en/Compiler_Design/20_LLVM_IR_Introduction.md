# LLVM IR Introduction

**Previous**: [19. Register Allocation](./19_Register_Allocation.md) | **Next**: [21. LLVM Backend](./21_LLVM_Backend.md)

---

LLVM IR (Intermediate Representation) is the lingua franca of the LLVM compiler infrastructure. It is a typed, SSA-based representation that sits between high-level source languages and low-level machine code. Learning to read and write LLVM IR is essential for anyone building language frontends, writing optimization passes, or understanding how modern compilers work at the IR level.

This lesson covers LLVM IR syntax, its type system, instructions, control flow, functions, and how to generate IR from a simple language.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [16. Modern Compiler Infrastructure](./16_Modern_Compiler_Infrastructure.md), [18. SSA Form](./18_SSA_Form.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and understand LLVM IR textual format
2. Explain the LLVM type system including integers, floats, pointers, arrays, and structs
3. Write arithmetic, comparison, and conversion instructions
4. Construct control flow with branches, phi nodes, and function calls
5. Generate LLVM IR programmatically using the `llvmlite` Python library
6. Compile and run LLVM IR using `lli` and `llc`

---

## Table of Contents

1. [LLVM IR Overview](#1-llvm-ir-overview)
2. [Module Structure](#2-module-structure)
3. [Type System](#3-type-system)
4. [Instructions](#4-instructions)
5. [Control Flow](#5-control-flow)
6. [Functions](#6-functions)
7. [Memory Operations](#7-memory-operations)
8. [Generating IR with llvmlite](#8-generating-ir-with-llvmlite)
9. [Compiling and Running IR](#9-compiling-and-running-ir)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. LLVM IR Overview

### 1.1 Three Equivalent Forms

LLVM IR exists in three equivalent forms:

| Form | Extension | Purpose |
|------|-----------|---------|
| **Textual** (human-readable) | `.ll` | Reading, debugging, learning |
| **Bitcode** (binary) | `.bc` | Storage, fast loading |
| **In-memory** (C++ objects) | N/A | Manipulation by passes |

You can convert between them:

```bash
# C source -> LLVM IR (textual)
clang -S -emit-llvm -O0 hello.c -o hello.ll

# LLVM IR -> bitcode
llvm-as hello.ll -o hello.bc

# Bitcode -> LLVM IR
llvm-dis hello.bc -o hello.ll

# LLVM IR -> native object
llc hello.ll -o hello.o

# Run LLVM IR directly
lli hello.ll
```

### 1.2 Key Properties

- **SSA form**: Every virtual register is assigned exactly once
- **Typed**: Every value has an explicit type
- **Target-independent**: IR is portable (target triple specifies the backend)
- **Infinite registers**: Uses `%name` or `%number` for virtual registers
- **Three-address code**: Most instructions take two operands and produce one result

### 1.3 A Minimal Example

```llvm
; hello.ll - A minimal LLVM IR program
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"Hello, LLVM!\0A\00"

declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  ret i32 0
}
```

---

## 2. Module Structure

### 2.1 Top-Level Entities

An LLVM module contains:

```llvm
; Target specification
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Global variables
@global_var = global i32 42
@const_str = constant [6 x i8] c"hello\00"

; Type definitions (named struct types)
%struct.Point = type { i32, i32 }

; Function declarations (external)
declare i32 @puts(ptr)

; Function definitions
define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
```

### 2.2 Identifiers

```llvm
; Global identifiers start with @
@global_var = global i32 0
@function_name = ...

; Local identifiers start with %
%local_var = add i32 %a, %b
%struct.MyType = type { i32, float }

; Unnamed temporaries use numbers
%0 = add i32 %a, %b
%1 = mul i32 %0, 2
```

### 2.3 Comments and Metadata

```llvm
; This is a comment (semicolon to end of line)

; Metadata nodes
!0 = !{!"my_metadata", i32 42}
!dbg = !{!0}

; Instructions can reference metadata
%x = add i32 %a, %b, !dbg !0
```

---

## 3. Type System

### 3.1 Integer Types

```llvm
i1     ; boolean (1 bit)
i8     ; byte
i16    ; short
i32    ; int
i64    ; long
i128   ; 128-bit integer
iN     ; arbitrary width integer (N bits)
```

### 3.2 Floating-Point Types

```llvm
half      ; 16-bit float (IEEE 754)
float     ; 32-bit float
double    ; 64-bit double
fp128     ; 128-bit float (IEEE 754)
x86_fp80  ; 80-bit float (x86 extended)
```

### 3.3 Pointer Type

```llvm
ptr       ; opaque pointer (LLVM 15+)

; Old-style typed pointers (deprecated):
; i32*     ; pointer to i32
; [10 x i8]*  ; pointer to array
```

### 3.4 Aggregate Types

```llvm
; Arrays: [count x element_type]
[10 x i32]          ; array of 10 i32s
[4 x [3 x float]]   ; 4x3 matrix of floats

; Structs: { type1, type2, ... }
{ i32, float, ptr }              ; struct with 3 fields
{ i32, { i32, i32 } }           ; nested struct

; Packed structs (no padding)
<{ i8, i32 }>

; Named structs
%struct.Point = type { i32, i32 }
```

### 3.5 Vector Types

```llvm
<4 x i32>       ; vector of 4 i32s (for SIMD)
<8 x float>     ; vector of 8 floats
<2 x double>    ; vector of 2 doubles
```

### 3.6 Other Types

```llvm
void       ; no value (for function return types)
label      ; basic block label
token      ; used for exception handling
```

---

## 4. Instructions

### 4.1 Arithmetic

```llvm
; Integer arithmetic
%sum  = add i32 %a, %b          ; addition
%diff = sub i32 %a, %b          ; subtraction
%prod = mul i32 %a, %b          ; multiplication
%quot = sdiv i32 %a, %b         ; signed division
%rem  = srem i32 %a, %b         ; signed remainder
%uqot = udiv i32 %a, %b        ; unsigned division
%urem = urem i32 %a, %b        ; unsigned remainder

; Flags for undefined behavior optimization
%sum_nsw = add nsw i32 %a, %b   ; no signed wrap
%sum_nuw = add nuw i32 %a, %b   ; no unsigned wrap

; Floating-point arithmetic
%fsum  = fadd float %x, %y
%fdiff = fsub float %x, %y
%fprod = fmul float %x, %y
%fquot = fdiv float %x, %y
%frem  = frem float %x, %y      ; floating-point remainder
```

### 4.2 Bitwise Operations

```llvm
%and = and i32 %a, %b
%or  = or  i32 %a, %b
%xor = xor i32 %a, %b
%shl = shl i32 %a, 2            ; shift left
%lshr = lshr i32 %a, 2          ; logical shift right
%ashr = ashr i32 %a, 2          ; arithmetic shift right
```

### 4.3 Comparison

```llvm
; Integer comparison (returns i1)
%eq  = icmp eq  i32 %a, %b      ; equal
%ne  = icmp ne  i32 %a, %b      ; not equal
%slt = icmp slt i32 %a, %b      ; signed less than
%sle = icmp sle i32 %a, %b      ; signed less or equal
%sgt = icmp sgt i32 %a, %b      ; signed greater than
%sge = icmp sge i32 %a, %b      ; signed greater or equal
%ult = icmp ult i32 %a, %b      ; unsigned less than

; Floating-point comparison
%feq = fcmp oeq float %x, %y    ; ordered and equal
%flt = fcmp olt float %x, %y    ; ordered and less than
%fun = fcmp uno float %x, %y    ; unordered (either is NaN)
```

### 4.4 Conversion

```llvm
; Integer conversions
%ext   = sext i8 %val to i32       ; sign extend
%zext  = zext i8 %val to i32       ; zero extend
%trunc = trunc i32 %val to i8      ; truncate

; Float conversions
%fpext  = fpext float %f to double   ; widen
%fptrunc = fptrunc double %d to float ; narrow

; Integer <-> Float
%sitofp = sitofp i32 %i to float    ; signed int to float
%fptosi = fptosi float %f to i32    ; float to signed int

; Pointer conversions
%ptoi = ptrtoint ptr %p to i64      ; pointer to integer
%itop = inttoptr i64 %i to ptr      ; integer to pointer
```

---

## 5. Control Flow

### 5.1 Basic Blocks and Labels

```llvm
define i32 @example(i32 %n) {
entry:                              ; label for the entry block
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %positive, label %negative

positive:                           ; label for the positive block
  %result1 = mul i32 %n, 2
  br label %merge

negative:                           ; label for the negative block
  %result2 = sub i32 0, %n
  br label %merge

merge:                              ; label for the merge block
  %result = phi i32 [%result1, %positive], [%result2, %negative]
  ret i32 %result
}
```

### 5.2 Branch Instructions

```llvm
; Unconditional branch
br label %target

; Conditional branch
br i1 %condition, label %true_block, label %false_block

; Switch (multi-way branch)
switch i32 %val, label %default [
  i32 0, label %case0
  i32 1, label %case1
  i32 2, label %case2
]
```

### 5.3 Phi Nodes

```llvm
; Phi selects value based on which predecessor block we came from
%x = phi i32 [%val1, %block1], [%val2, %block2]

; Loop example
define i32 @sum_to_n(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next_i, %loop]
  %sum = phi i32 [0, %entry], [%next_sum, %loop]
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  %cond = icmp slt i32 %next_i, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %next_sum
}
```

### 5.4 Select Instruction

```llvm
; Ternary operator (no branch needed)
%max = select i1 %cmp, i32 %a, i32 %b
; Equivalent to: max = cmp ? a : b
```

---

## 6. Functions

### 6.1 Function Definitions

```llvm
; Basic function
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}

; Function with attributes
define i32 @factorial(i32 %n) nounwind readnone {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret i32 1

recurse:
  %n_minus_1 = sub i32 %n, 1
  %sub_result = call i32 @factorial(i32 %n_minus_1)
  %result = mul i32 %n, %sub_result
  ret i32 %result
}
```

### 6.2 Calling Conventions

```llvm
; Default C calling convention
define i32 @func(i32 %a) {
  ret i32 %a
}

; Fast calling convention
define fastcc i32 @fast_func(i32 %a) {
  ret i32 %a
}

; Tail call optimization
define fastcc i32 @tail_recursive(i32 %n, i32 %acc) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret i32 %acc

recurse:
  %n1 = sub i32 %n, 1
  %acc1 = mul i32 %acc, %n
  %result = tail call fastcc i32 @tail_recursive(i32 %n1, i32 %acc1)
  ret i32 %result
}
```

### 6.3 Function Declarations

```llvm
; External function (defined elsewhere)
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)
```

---

## 7. Memory Operations

### 7.1 Stack Allocation

```llvm
; alloca allocates on the stack
%ptr = alloca i32                ; allocate space for one i32
%arr = alloca [10 x i32]        ; allocate array of 10 i32s
%aligned = alloca i32, align 16  ; with alignment
```

### 7.2 Load and Store

```llvm
; Store a value to memory
store i32 42, ptr %ptr

; Load a value from memory
%val = load i32, ptr %ptr

; Volatile (cannot be optimized away)
store volatile i32 42, ptr %ptr
%val2 = load volatile i32, ptr %ptr
```

### 7.3 GetElementPtr (GEP)

GEP computes addresses without accessing memory:

```llvm
; Array element access: arr[5]
%arr_ptr = alloca [10 x i32]
%elem_ptr = getelementptr [10 x i32], ptr %arr_ptr, i64 0, i64 5
%elem = load i32, ptr %elem_ptr

; Struct field access: point.y (second field)
%struct.Point = type { i32, i32 }
%point = alloca %struct.Point
%y_ptr = getelementptr %struct.Point, ptr %point, i64 0, i32 1
store i32 42, ptr %y_ptr

; Pointer arithmetic: ptr + offset
%next = getelementptr i32, ptr %base, i64 1
```

---

## 8. Generating IR with llvmlite

### 8.1 Setup

```python
# pip install llvmlite
from llvmlite import ir

# Create a module
module = ir.Module(name="my_module")
module.triple = "x86_64-unknown-linux-gnu"
```

### 8.2 Building a Function

```python
from llvmlite import ir

def build_add_function():
    """Build: int add(int a, int b) { return a + b; }"""
    module = ir.Module(name="example")

    # Define function type: i32 (i32, i32)
    func_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)])

    # Create the function
    func = ir.Function(module, func_type, name="add")
    func.args[0].name = "a"
    func.args[1].name = "b"

    # Create entry basic block
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Build instructions
    result = builder.add(func.args[0], func.args[1], name="result")
    builder.ret(result)

    print(module)
    return module

# Output:
# define i32 @add(i32 %a, i32 %b) {
# entry:
#   %result = add i32 %a, %b
#   ret i32 %result
# }
```

### 8.3 Control Flow with llvmlite

```python
def build_max_function():
    """Build: int max(int a, int b) { return a > b ? a : b; }"""
    module = ir.Module(name="example")
    i32 = ir.IntType(32)

    func_type = ir.FunctionType(i32, [i32, i32])
    func = ir.Function(module, func_type, name="max")
    a, b = func.args

    entry = func.append_basic_block("entry")
    then_bb = func.append_basic_block("then")
    else_bb = func.append_basic_block("else")
    merge_bb = func.append_basic_block("merge")

    # Entry block
    builder = ir.IRBuilder(entry)
    cmp = builder.icmp_signed(">", a, b, name="cmp")
    builder.cbranch(cmp, then_bb, else_bb)

    # Then block
    builder = ir.IRBuilder(then_bb)
    builder.branch(merge_bb)

    # Else block
    builder = ir.IRBuilder(else_bb)
    builder.branch(merge_bb)

    # Merge block with phi
    builder = ir.IRBuilder(merge_bb)
    phi = builder.phi(i32, name="result")
    phi.add_incoming(a, then_bb)
    phi.add_incoming(b, else_bb)
    builder.ret(phi)

    print(module)
    return module
```

### 8.4 Loops with llvmlite

```python
def build_factorial():
    """Build iterative factorial."""
    module = ir.Module(name="factorial")
    i32 = ir.IntType(32)

    func_type = ir.FunctionType(i32, [i32])
    func = ir.Function(module, func_type, name="factorial")
    n = func.args[0]
    n.name = "n"

    entry = func.append_basic_block("entry")
    loop = func.append_basic_block("loop")
    exit_bb = func.append_basic_block("exit")

    # Entry
    builder = ir.IRBuilder(entry)
    builder.branch(loop)

    # Loop
    builder = ir.IRBuilder(loop)
    i = builder.phi(i32, name="i")
    acc = builder.phi(i32, name="acc")

    i.add_incoming(ir.Constant(i32, 1), entry)
    acc.add_incoming(ir.Constant(i32, 1), entry)

    new_acc = builder.mul(acc, i, name="new_acc")
    new_i = builder.add(i, ir.Constant(i32, 1), name="new_i")
    cond = builder.icmp_signed("<=", new_i, n, name="cond")

    i.add_incoming(new_i, loop)
    acc.add_incoming(new_acc, loop)

    builder.cbranch(cond, loop, exit_bb)

    # Exit
    builder = ir.IRBuilder(exit_bb)
    builder.ret(new_acc)

    print(module)
    return module
```

---

## 9. Compiling and Running IR

### 9.1 Using Command-Line Tools

```bash
# Write IR to a file
cat > example.ll << 'EOF'
define i32 @main() {
  ret i32 42
}
EOF

# Interpret directly
lli example.ll
echo $?  # prints 42

# Compile to object file
llc -filetype=obj example.ll -o example.o

# Link and run
gcc example.o -o example
./example
echo $?  # prints 42

# Optimize
opt -O2 example.ll -S -o example_opt.ll
```

### 9.2 JIT Execution with llvmlite

```python
from llvmlite import ir, binding
import ctypes

def jit_execute():
    """Build a function and execute it via JIT."""
    # Initialize LLVM
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    # Build the module (reuse build_add_function)
    module = build_add_function()

    # Parse the IR
    llvm_ir = str(module)
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    # Create execution engine
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = binding.create_mcjit_compiler(mod, target_machine)

    # Get function pointer
    func_ptr = engine.get_function_address("add")

    # Call via ctypes
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(func_ptr)
    result = cfunc(3, 4)
    print(f"add(3, 4) = {result}")  # prints: add(3, 4) = 7
```

---

## 10. Summary

- LLVM IR is a **typed, SSA-based** intermediate representation
- It exists in three forms: textual (`.ll`), bitcode (`.bc`), and in-memory
- The **type system** includes integers, floats, pointers, arrays, structs, and vectors
- **Instructions** cover arithmetic, bitwise, comparison, conversion, and memory operations
- **Control flow** uses basic blocks, branches, phi nodes, and switch statements
- **GEP** (getelementptr) computes addresses for arrays and structs
- **llvmlite** enables generating and JIT-executing LLVM IR from Python
- Command-line tools (`llc`, `lli`, `opt`) compile, run, and optimize IR

---

## 11. Exercises

1. **Read IR**: Given a C function, compile it to LLVM IR with `clang -S -emit-llvm` and annotate each instruction.

2. **Write IR by hand**: Write LLVM IR for a function that computes the GCD of two integers using a loop.

3. **Fibonacci with llvmlite**: Use `llvmlite` to generate IR for an iterative Fibonacci function and JIT-execute it.

4. **Struct access**: Write LLVM IR that creates a struct with three fields, stores values, and reads them back.

5. **Array sum**: Write LLVM IR for a function that sums all elements of an i32 array given a pointer and length.

---

## 12. References

1. LLVM Language Reference Manual: https://llvm.org/docs/LangRef.html
2. LLVM Tutorial: Implementing a Language with LLVM: https://llvm.org/docs/tutorial/
3. llvmlite documentation: https://llvmlite.readthedocs.io/
4. Lattner, C., Adve, V. (2004). "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation." *CGO*.
5. LLVM IR Tutorial (Mapping High-Level Constructs to LLVM IR): https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/

---

**Previous**: [19. Register Allocation](./19_Register_Allocation.md) | **Next**: [21. LLVM Backend](./21_LLVM_Backend.md)
