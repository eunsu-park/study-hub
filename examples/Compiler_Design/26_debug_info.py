"""
26_debug_info.py - Debug Information Generation

Demonstrates how compilers generate debug information that enables
source-level debugging (breakpoints, variable inspection, stack traces)
even after optimization and code generation.

Components:
  1. Source Location Tracking
     Map each IR instruction and machine instruction back to its
     source file, line, and column.

  2. Variable Location Tracking
     Track where each source variable lives at every point in the
     program: in a register, on the stack, or optimized away.

  3. DWARF-like Debug Info Encoding
     Generate simplified DWARF debug information entries (DIEs)
     describing compilation units, functions, variables, and types.

  4. Line Number Table
     Build a line number program that maps addresses to source lines,
     enabling the debugger to set breakpoints and single-step.

  5. Call Frame Information
     Describe how to unwind the call stack at any point, enabling
     stack traces and exception handling.

Topics covered:
  - Source mapping (line tables)
  - DWARF debug information format
  - Variable location descriptions
  - Debug info through optimization passes
  - Call frame information (CFI) for stack unwinding
  - Source map concepts (applicable to JS, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Source Location
# ---------------------------------------------------------------------------

@dataclass
class SourceLoc:
    file: str
    line: int
    column: int = 0

    def __str__(self):
        if self.column:
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.file}:{self.line}"


# ---------------------------------------------------------------------------
# Line Number Table
# ---------------------------------------------------------------------------

@dataclass
class LineEntry:
    """Maps a code address to a source location."""
    address: int
    loc: SourceLoc
    is_stmt: bool = True  # Is this a statement boundary?
    prologue_end: bool = False
    epilogue_begin: bool = False

    def __str__(self):
        flags = []
        if self.is_stmt:
            flags.append("stmt")
        if self.prologue_end:
            flags.append("prologue_end")
        if self.epilogue_begin:
            flags.append("epilogue_begin")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"  0x{self.address:08x}  {self.loc}{flag_str}"


class LineTable:
    """
    Line number program that maps instruction addresses to source lines.
    Supports the state machine model used by DWARF.
    """

    def __init__(self, comp_dir: str = ".", file_name: str = ""):
        self.entries: list[LineEntry] = []
        self.comp_dir = comp_dir
        self.file_name = file_name

    def add_entry(self, address: int, loc: SourceLoc,
                  is_stmt: bool = True, prologue_end: bool = False,
                  epilogue_begin: bool = False) -> None:
        self.entries.append(
            LineEntry(address, loc, is_stmt, prologue_end, epilogue_begin))

    def lookup(self, address: int) -> Optional[SourceLoc]:
        """Find the source location for a given address."""
        best = None
        for entry in self.entries:
            if entry.address <= address:
                best = entry.loc
            else:
                break
        return best

    def __str__(self):
        lines = ["Line Number Table:", f"  File: {self.file_name}"]
        lines.append(f"  {'Address':<14} {'Location':<25} {'Flags'}")
        lines.append(f"  {'─' * 14} {'─' * 25} {'─' * 20}")
        for entry in self.entries:
            lines.append(str(entry))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variable Location
# ---------------------------------------------------------------------------

class LocKind(Enum):
    REGISTER = auto()
    STACK    = auto()
    CONSTANT = auto()
    OPTIMIZED_OUT = auto()


@dataclass
class VarLocation:
    """Describes where a variable's value can be found."""
    kind: LocKind
    detail: Any = None  # register name, stack offset, constant value

    def __str__(self):
        if self.kind == LocKind.REGISTER:
            return f"reg({self.detail})"
        if self.kind == LocKind.STACK:
            return f"[rbp{self.detail:+d}]"
        if self.kind == LocKind.CONSTANT:
            return f"const({self.detail})"
        return "<optimized out>"


@dataclass
class VarLocationRange:
    """Variable location valid for an address range."""
    var_name: str
    start_addr: int
    end_addr: int
    location: VarLocation

    def __str__(self):
        return (f"  {self.var_name}: 0x{self.start_addr:08x}-"
                f"0x{self.end_addr:08x} -> {self.location}")


class VariableTracker:
    """Track variable locations throughout a function."""

    def __init__(self):
        self.ranges: list[VarLocationRange] = []

    def add_range(self, var: str, start: int, end: int,
                  loc: VarLocation) -> None:
        self.ranges.append(VarLocationRange(var, start, end, loc))

    def lookup(self, var: str, address: int) -> Optional[VarLocation]:
        for r in self.ranges:
            if r.var_name == var and r.start_addr <= address < r.end_addr:
                return r.location
        return None


# ---------------------------------------------------------------------------
# DWARF-like Debug Information Entries (DIEs)
# ---------------------------------------------------------------------------

class DWTag(Enum):
    COMPILE_UNIT   = auto()
    SUBPROGRAM     = auto()
    VARIABLE       = auto()
    FORMAL_PARAM   = auto()
    BASE_TYPE      = auto()
    POINTER_TYPE   = auto()
    LEXICAL_BLOCK  = auto()


@dataclass
class DIE:
    """Debug Information Entry (simplified DWARF DIE)."""
    tag: DWTag
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[DIE] = field(default_factory=list)

    def dump(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [f"{pad}<{self.tag.name}>"]
        for key, val in self.attributes.items():
            lines.append(f"{pad}  {key}: {val}")
        for child in self.children:
            lines.append(child.dump(indent + 1))
        return "\n".join(lines)


class DebugInfoBuilder:
    """Builder for DWARF-like debug information."""

    def __init__(self):
        self.compile_unit: Optional[DIE] = None

    def create_compile_unit(self, name: str, comp_dir: str,
                            language: str = "C") -> DIE:
        self.compile_unit = DIE(DWTag.COMPILE_UNIT, {
            "name": name,
            "comp_dir": comp_dir,
            "language": language,
            "producer": "example_compiler 1.0",
        })
        return self.compile_unit

    def create_base_type(self, name: str, size: int,
                         encoding: str = "signed") -> DIE:
        die = DIE(DWTag.BASE_TYPE, {
            "name": name,
            "byte_size": size,
            "encoding": encoding,
        })
        if self.compile_unit:
            self.compile_unit.children.append(die)
        return die

    def create_function(self, name: str, low_pc: int, high_pc: int,
                        return_type: str) -> DIE:
        die = DIE(DWTag.SUBPROGRAM, {
            "name": name,
            "low_pc": f"0x{low_pc:08x}",
            "high_pc": f"0x{high_pc:08x}",
            "return_type": return_type,
            "frame_base": "rbp",
        })
        if self.compile_unit:
            self.compile_unit.children.append(die)
        return die

    def create_variable(self, parent: DIE, name: str,
                        type_name: str, location: str) -> DIE:
        die = DIE(DWTag.VARIABLE, {
            "name": name,
            "type": type_name,
            "location": location,
        })
        parent.children.append(die)
        return die

    def create_parameter(self, parent: DIE, name: str,
                         type_name: str, location: str) -> DIE:
        die = DIE(DWTag.FORMAL_PARAM, {
            "name": name,
            "type": type_name,
            "location": location,
        })
        parent.children.append(die)
        return die


# ---------------------------------------------------------------------------
# Call Frame Information
# ---------------------------------------------------------------------------

@dataclass
class CFIEntry:
    """Call Frame Information entry for stack unwinding."""
    address: int
    cfa_register: str  # Canonical Frame Address register
    cfa_offset: int    # CFA = register + offset
    return_addr_loc: str  # Where return address is stored

    def __str__(self):
        return (f"  0x{self.address:08x}: CFA = {self.cfa_register}"
                f"{self.cfa_offset:+d}, "
                f"return_addr = {self.return_addr_loc}")


class CFITable:
    """Call Frame Information table for a function."""

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.entries: list[CFIEntry] = []

    def add_entry(self, address: int, cfa_reg: str,
                  cfa_off: int, ret_loc: str) -> None:
        self.entries.append(CFIEntry(address, cfa_reg, cfa_off, ret_loc))

    def __str__(self):
        lines = [f"CFI Table for {self.func_name}:"]
        for entry in self.entries:
            lines.append(str(entry))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Debug Information Generation Demo")
    print("=" * 60)

    # Simulated source code:
    # example.c:
    #   1: int add(int a, int b) {
    #   2:     int result = a + b;
    #   3:     return result;
    #   4: }
    #   5:
    #   6: int main() {
    #   7:     int x = 10;
    #   8:     int y = 20;
    #   9:     int sum = add(x, y);
    #  10:     return sum;
    #  11: }

    # 1. Line Number Table
    print("\n--- 1. Line Number Table ---")
    lt = LineTable(".", "example.c")
    # add() at 0x400000
    lt.add_entry(0x400000, SourceLoc("example.c", 1), prologue_end=True)
    lt.add_entry(0x400004, SourceLoc("example.c", 2))
    lt.add_entry(0x400008, SourceLoc("example.c", 3))
    lt.add_entry(0x40000c, SourceLoc("example.c", 3), epilogue_begin=True)
    # main() at 0x400010
    lt.add_entry(0x400010, SourceLoc("example.c", 6), prologue_end=True)
    lt.add_entry(0x400018, SourceLoc("example.c", 7))
    lt.add_entry(0x40001c, SourceLoc("example.c", 8))
    lt.add_entry(0x400020, SourceLoc("example.c", 9))
    lt.add_entry(0x400028, SourceLoc("example.c", 10))
    lt.add_entry(0x40002c, SourceLoc("example.c", 10), epilogue_begin=True)
    print(lt)

    # Lookup demo
    for addr in [0x400004, 0x400020, 0x400028]:
        loc = lt.lookup(addr)
        print(f"\n  Lookup 0x{addr:08x} -> {loc}")

    # 2. Variable Locations
    print("\n\n--- 2. Variable Location Tracking ---")
    tracker = VariableTracker()
    # add(): parameters in registers, result on stack
    tracker.add_range("a", 0x400000, 0x40000c,
                      VarLocation(LocKind.REGISTER, "edi"))
    tracker.add_range("b", 0x400000, 0x40000c,
                      VarLocation(LocKind.REGISTER, "esi"))
    tracker.add_range("result", 0x400004, 0x40000c,
                      VarLocation(LocKind.REGISTER, "eax"))
    # main(): variables on stack
    tracker.add_range("x", 0x400018, 0x40002c,
                      VarLocation(LocKind.STACK, -4))
    tracker.add_range("y", 0x40001c, 0x40002c,
                      VarLocation(LocKind.STACK, -8))
    tracker.add_range("sum", 0x400020, 0x40002c,
                      VarLocation(LocKind.REGISTER, "eax"))

    for r in tracker.ranges:
        print(r)

    # Lookup demo
    print()
    for var, addr in [("a", 0x400004), ("x", 0x400020), ("sum", 0x400028)]:
        loc = tracker.lookup(var, addr)
        print(f"  {var} at 0x{addr:08x} -> {loc}")

    # 3. DWARF-like DIEs
    print("\n\n--- 3. DWARF Debug Information Entries ---")
    builder = DebugInfoBuilder()
    cu = builder.create_compile_unit("example.c", "/home/user/src")

    int_type = builder.create_base_type("int", 4, "signed")

    add_func = builder.create_function("add", 0x400000, 0x40000c, "int")
    builder.create_parameter(add_func, "a", "int", "reg(edi)")
    builder.create_parameter(add_func, "b", "int", "reg(esi)")
    builder.create_variable(add_func, "result", "int", "reg(eax)")

    main_func = builder.create_function("main", 0x400010, 0x40002c, "int")
    builder.create_variable(main_func, "x", "int", "[rbp-4]")
    builder.create_variable(main_func, "y", "int", "[rbp-8]")
    builder.create_variable(main_func, "sum", "int", "reg(eax)")

    print(cu.dump())

    # 4. Call Frame Information
    print("\n\n--- 4. Call Frame Information ---")
    cfi = CFITable("main")
    cfi.add_entry(0x400010, "rsp", 8, "[rsp]")       # after call
    cfi.add_entry(0x400011, "rsp", 16, "[rsp+8]")    # after push rbp
    cfi.add_entry(0x400014, "rbp", 16, "[rbp+8]")    # after mov rbp,rsp
    cfi.add_entry(0x40002b, "rsp", 8, "[rsp]")       # after leave
    print(cfi)

    print(f"\n--- Debug Info Through Optimization ---")
    print("""
  Challenges when optimizing code with debug info:
    - Inlined functions: debugger must reconstruct virtual call stack
    - Dead code elimination: variable may be "optimized out"
    - Register allocation: variable moves between registers and stack
    - Instruction reordering: source line mappings become non-monotonic
    - Constant propagation: variable has known value, no storage location

  Solutions:
    - Location lists: variable location changes throughout execution
    - Inlined subroutine DIEs: mark inlined call sites
    - DW_AT_const_value: encode compile-time known values
    - is_stmt flag: mark "recommended breakpoint" positions
    """)


if __name__ == "__main__":
    main()
