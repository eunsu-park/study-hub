"""
Exercises for Lesson 26: Debug Information
Topic: Compiler_Design

Demonstrates DWARF inspection, line tables, and source maps.
"""

import json
import base64


def exercise_1():
    """DWARF inspection commands."""
    print("Exercise 1: Inspect DWARF Debug Info")
    commands = [
        "gcc -g -O0 test.c -o test",
        "readelf --debug-dump=info test | head -60",
        "readelf --debug-dump=line test",
        "llvm-dwarfdump test",
        "llvm-dwarfdump --name=main test",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    print()


def exercise_2():
    """Trace line number state machine."""
    print("Exercise 2: Line Number State Machine Trace")
    print()
    opcodes = [
        ("set_file", 1, "main.c"),
        ("advance_line", 5, None),
        ("advance_pc", 0x4005a0, None),
        ("copy", None, "emit row: 0x4005a0 -> main.c:5"),
        ("advance_line", 1, None),
        ("advance_pc", 0x4005a8, None),
        ("copy", None, "emit row: 0x4005a8 -> main.c:6"),
        ("advance_line", 2, None),
        ("advance_pc", 0x4005b4, None),
        ("copy", None, "emit row: 0x4005b4 -> main.c:8"),
    ]
    state = {"address": 0, "line": 1, "file": 0}
    print("  State machine execution:")
    for op, arg, note in opcodes:
        if op == "set_file":
            state["file"] = arg
        elif op == "advance_line":
            state["line"] = arg if state["line"] == 1 else state["line"] + (arg - state["line"]) if isinstance(arg, int) and arg > 10 else state["line"] + 1
            state["line"] = arg
        elif op == "advance_pc":
            state["address"] = arg
        if note:
            print(f"    {note}")
    print()


def exercise_3():
    """Variable location changes with optimization."""
    print("Exercise 3: Variable Locations Under Optimization")
    print()
    locations = [
        ("0x4005a0", "0x4005b0", "DW_OP_reg0 (rax)", "x is in register rax"),
        ("0x4005b0", "0x4005d0", "DW_OP_fbreg -8", "x spilled to stack [rbp-8]"),
        ("0x4005d0", "0x4005e0", "<optimized out>", "x is dead"),
    ]
    print("  Variable 'x' location list:")
    for start, end, loc, desc in locations:
        print(f"    [{start}, {end}): {loc}  -- {desc}")
    print()
    print("  Commands to inspect:")
    print("    gcc -g -O2 test.c -o test")
    print('    llvm-dwarfdump --name=x test')
    print()


def exercise_4():
    """Generate a simple source map."""
    print("Exercise 4: Source Map Generation")
    print()
    original = 'function greet(name) {\n  return "Hello " + name;\n}'
    compiled = 'function a(b){return"Hello "+b}'

    source_map = {
        "version": 3,
        "file": "output.js",
        "sources": ["input.js"],
        "names": ["greet", "name"],
        "mappings": "AAAA,SAASA,CAAQC,GAAM,MAAO,SAAWA",
    }

    print("  Original:")
    for line in original.split('\n'):
        print(f"    {line}")
    print()
    print("  Compiled:")
    print(f"    {compiled}")
    print()
    print("  Source map:")
    print(f"    {json.dumps(source_map, indent=4)}")
    print()


def exercise_5():
    """Debug info generation in LLVM IR."""
    print("Exercise 5: LLVM Debug Metadata")
    print()
    ir = '''
define i32 @square(i32 %x) !dbg !5 {
entry:
  call void @llvm.dbg.value(
    metadata i32 %x, metadata !10, metadata !DIExpression()), !dbg !11
  %result = mul i32 %x, %x, !dbg !12
  ret i32 %result, !dbg !13
}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
       producer: "my_compiler", isOptimized: false)
!1 = !DIFile(filename: "test.c", directory: "/home")
!5 = distinct !DISubprogram(name: "square", file: !1, line: 1,
       type: !6, scopeLine: 1, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1,
        line: 1, type: !8)
!11 = !DILocation(line: 1, column: 18, scope: !5)
!12 = !DILocation(line: 2, column: 10, scope: !5)
!13 = !DILocation(line: 2, column: 3, scope: !5)
'''.strip()
    print(ir)
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
