"""
25_linker.py - Linking and Loading

Demonstrates how a linker combines separately compiled object files
into an executable, resolving symbol references across modules.

Components:
  1. Object File Representation
     Simulates the structure of object files: sections (text, data, bss),
     symbol tables, and relocation entries.

  2. Symbol Resolution
     Match undefined symbol references in one module with definitions
     in another, detecting duplicates and missing symbols.

  3. Section Merging and Layout
     Combine .text, .data, and .bss sections from multiple object files
     into a single address space with proper alignment.

  4. Relocation
     Patch code and data references to reflect the final addresses of
     symbols after section layout.

  5. Static vs Dynamic Linking
     Compare static linking (all code in executable) with dynamic
     linking (shared libraries resolved at load time).

Topics covered:
  - Object file format (simplified ELF-like)
  - Symbol tables: global, local, external
  - Relocation types: absolute, PC-relative
  - Section merging and address assignment
  - Static vs dynamic linking trade-offs
  - Position-Independent Code (PIC)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Symbol and Relocation types
# ---------------------------------------------------------------------------

class SymbolBind(Enum):
    LOCAL  = auto()  # Visible only within its object file
    GLOBAL = auto()  # Visible to other object files
    EXTERN = auto()  # Defined in another object file


class RelocType(Enum):
    ABS32   = auto()  # 32-bit absolute address
    PC_REL  = auto()  # PC-relative (for calls/branches)


@dataclass
class Symbol:
    name: str
    bind: SymbolBind
    section: str = ""      # ".text", ".data", etc.
    offset: int = 0        # Offset within section
    size: int = 0
    address: int = 0       # Final resolved address

    def __repr__(self):
        return (f"Symbol({self.name}, {self.bind.name}, "
                f"sec={self.section}, off={self.offset})")


@dataclass
class Relocation:
    offset: int            # Offset in section where fixup is needed
    symbol: str            # Symbol being referenced
    type: RelocType
    addend: int = 0        # Constant to add to symbol address

    def __repr__(self):
        return (f"Reloc(off={self.offset}, sym={self.symbol}, "
                f"type={self.type.name}, add={self.addend})")


# ---------------------------------------------------------------------------
# Section and Object File
# ---------------------------------------------------------------------------

@dataclass
class Section:
    name: str              # ".text", ".data", ".bss"
    data: bytearray = field(default_factory=bytearray)
    align: int = 4
    relocations: list[Relocation] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f"Section({self.name}, size={self.size}, relocs={len(self.relocations)})"


@dataclass
class ObjectFile:
    name: str
    sections: dict[str, Section] = field(default_factory=dict)
    symbols: list[Symbol] = field(default_factory=list)

    def add_section(self, name: str, data: bytes, align: int = 4) -> Section:
        sec = Section(name, bytearray(data), align)
        self.sections[name] = sec
        return sec

    def add_symbol(self, name: str, bind: SymbolBind,
                   section: str = "", offset: int = 0,
                   size: int = 0) -> Symbol:
        sym = Symbol(name, bind, section, offset, size)
        self.symbols.append(sym)
        return sym

    def add_relocation(self, section: str, offset: int,
                       symbol: str, rtype: RelocType,
                       addend: int = 0) -> None:
        reloc = Relocation(offset, symbol, rtype, addend)
        self.sections[section].relocations.append(reloc)

    def __repr__(self):
        syms = [s.name for s in self.symbols]
        return f"ObjectFile({self.name}, sections={list(self.sections.keys())}, symbols={syms})"


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------

@dataclass
class LinkedSection:
    name: str
    base_address: int
    data: bytearray
    size: int


class Linker:
    """
    Combines multiple object files into a single executable.
    Performs symbol resolution, section merging, and relocation.
    """

    def __init__(self, base_address: int = 0x400000):
        self.base_address = base_address
        self.global_symbols: dict[str, Symbol] = {}
        self.sections: dict[str, LinkedSection] = {}
        self.log: list[str] = []
        self.errors: list[str] = []

    def link(self, objects: list[ObjectFile]) -> bool:
        """Link object files into an executable."""
        self.log.append("=== Linking Phase 1: Symbol Resolution ===")
        if not self._resolve_symbols(objects):
            return False

        self.log.append("\n=== Linking Phase 2: Section Merging ===")
        self._merge_sections(objects)

        self.log.append("\n=== Linking Phase 3: Relocation ===")
        self._apply_relocations(objects)

        return len(self.errors) == 0

    def _resolve_symbols(self, objects: list[ObjectFile]) -> bool:
        """Resolve symbol references across all object files."""
        # Pass 1: Collect all global definitions
        for obj in objects:
            for sym in obj.symbols:
                if sym.bind == SymbolBind.GLOBAL:
                    if sym.name in self.global_symbols:
                        self.errors.append(
                            f"Duplicate symbol: {sym.name} "
                            f"(in {obj.name} and previous)")
                        return False
                    self.global_symbols[sym.name] = sym
                    self.log.append(
                        f"  Global: {sym.name} from {obj.name} "
                        f"(sec={sym.section}, off={sym.offset})")

        # Pass 2: Check all extern references are satisfied
        for obj in objects:
            for sym in obj.symbols:
                if sym.bind == SymbolBind.EXTERN:
                    if sym.name not in self.global_symbols:
                        self.errors.append(
                            f"Undefined symbol: {sym.name} "
                            f"(referenced in {obj.name})")
                    else:
                        self.log.append(
                            f"  Resolved: {sym.name} in {obj.name} "
                            f"-> defined in global table")

        return len(self.errors) == 0

    def _merge_sections(self, objects: list[ObjectFile]) -> None:
        """Merge sections from all object files and assign addresses."""
        section_names = [".text", ".data", ".bss"]
        current_addr = self.base_address

        for sec_name in section_names:
            merged_data = bytearray()
            section_offsets: dict[str, int] = {}  # obj_name -> offset in merged

            for obj in objects:
                sec = obj.sections.get(sec_name)
                if sec is None:
                    continue

                # Align
                padding = (sec.align - len(merged_data) % sec.align) % sec.align
                merged_data.extend(b'\x00' * padding)

                offset = len(merged_data)
                section_offsets[obj.name] = offset

                # Update symbol addresses for this object's symbols
                for sym in obj.symbols:
                    if sym.section == sec_name and sym.bind != SymbolBind.EXTERN:
                        sym.address = current_addr + offset + sym.offset
                        if sym.bind == SymbolBind.GLOBAL:
                            self.global_symbols[sym.name].address = sym.address
                        self.log.append(
                            f"  {sym.name}: {sec_name} -> addr=0x{sym.address:08x}")

                merged_data.extend(sec.data)

            if merged_data:
                self.sections[sec_name] = LinkedSection(
                    sec_name, current_addr, merged_data, len(merged_data))
                self.log.append(
                    f"  Section {sec_name}: base=0x{current_addr:08x}, "
                    f"size={len(merged_data)}")
                current_addr += len(merged_data)
                # Align to page
                current_addr = (current_addr + 0xFFF) & ~0xFFF

    def _apply_relocations(self, objects: list[ObjectFile]) -> None:
        """Patch all relocation entries with final addresses."""
        for obj in objects:
            for sec_name, sec in obj.sections.items():
                linked_sec = self.sections.get(sec_name)
                if not linked_sec:
                    continue

                for reloc in sec.relocations:
                    sym = self.global_symbols.get(reloc.symbol)
                    if sym is None:
                        self.errors.append(
                            f"Relocation error: undefined {reloc.symbol}")
                        continue

                    target_addr = sym.address + reloc.addend

                    if reloc.type == RelocType.ABS32:
                        self.log.append(
                            f"  Reloc ABS32: {reloc.symbol} at "
                            f"off={reloc.offset} -> 0x{target_addr:08x}")

                    elif reloc.type == RelocType.PC_REL:
                        # PC-relative: target - (current instruction address)
                        pc = linked_sec.base_address + reloc.offset
                        rel_addr = target_addr - pc
                        self.log.append(
                            f"  Reloc PC_REL: {reloc.symbol} at "
                            f"off={reloc.offset} -> relative={rel_addr}")

    def memory_map(self) -> str:
        """Display the memory map of the linked executable."""
        lines = ["Memory Map:"]
        for name, sec in sorted(self.sections.items(),
                                key=lambda x: x[1].base_address):
            end_addr = sec.base_address + sec.size
            lines.append(
                f"  0x{sec.base_address:08x} - 0x{end_addr:08x}  "
                f"{name:8s} ({sec.size} bytes)")
        lines.append("\nSymbol Table:")
        for name, sym in sorted(self.global_symbols.items(),
                                key=lambda x: x[1].address):
            lines.append(f"  0x{sym.address:08x}  {name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def create_main_obj() -> ObjectFile:
    """
    Simulate main.o:
      .text: main() calls add() and printf()
      .data: format string
    """
    obj = ObjectFile("main.o")

    # .text section (simulated machine code bytes)
    text_data = bytes(32)  # 32 bytes of "code"
    obj.add_section(".text", text_data)

    # .data section
    data = b"Result: %d\n\0"
    obj.add_section(".data", data)

    # Symbols
    obj.add_symbol("main", SymbolBind.GLOBAL, ".text", 0, 32)
    obj.add_symbol("add", SymbolBind.EXTERN)
    obj.add_symbol("printf", SymbolBind.EXTERN)
    obj.add_symbol("fmt_str", SymbolBind.GLOBAL, ".data", 0, len(data))

    # Relocations: main calls add at offset 8, printf at offset 20
    obj.add_relocation(".text", 8, "add", RelocType.PC_REL)
    obj.add_relocation(".text", 20, "printf", RelocType.PC_REL)

    return obj


def create_math_obj() -> ObjectFile:
    """
    Simulate math.o:
      .text: add(a, b) { return a + b; }
    """
    obj = ObjectFile("math.o")

    text_data = bytes(16)  # 16 bytes of "code"
    obj.add_section(".text", text_data)

    obj.add_symbol("add", SymbolBind.GLOBAL, ".text", 0, 16)

    return obj


def create_libc_obj() -> ObjectFile:
    """Simulate libc.o with printf."""
    obj = ObjectFile("libc.o")

    text_data = bytes(48)
    obj.add_section(".text", text_data)

    obj.add_symbol("printf", SymbolBind.GLOBAL, ".text", 0, 48)

    return obj


def main():
    print("=" * 60)
    print("Linking and Loading Demo")
    print("=" * 60)

    # Create object files
    main_o = create_main_obj()
    math_o = create_math_obj()
    libc_o = create_libc_obj()

    print(f"\n--- Object Files ---")
    for obj in [main_o, math_o, libc_o]:
        print(f"  {obj}")
        for sym in obj.symbols:
            print(f"    {sym}")

    # Link
    linker = Linker(base_address=0x400000)
    success = linker.link([main_o, math_o, libc_o])

    print(f"\n--- Linker Log ---")
    for entry in linker.log:
        print(f"  {entry}")

    if linker.errors:
        print(f"\n--- Errors ---")
        for err in linker.errors:
            print(f"  ERROR: {err}")
    else:
        print(f"\n--- {linker.memory_map()} ---")

    # Demo: undefined symbol error
    print(f"\n--- Undefined Symbol Demo ---")
    bad_obj = ObjectFile("bad.o")
    bad_obj.add_section(".text", bytes(8))
    bad_obj.add_symbol("bar", SymbolBind.EXTERN)
    bad_obj.add_symbol("foo", SymbolBind.GLOBAL, ".text", 0, 8)

    linker2 = Linker()
    linker2.link([bad_obj])
    for err in linker2.errors:
        print(f"  {err}")

    # Demo: duplicate symbol error
    print(f"\n--- Duplicate Symbol Demo ---")
    dup1 = ObjectFile("dup1.o")
    dup1.add_section(".text", bytes(8))
    dup1.add_symbol("func", SymbolBind.GLOBAL, ".text", 0, 8)

    dup2 = ObjectFile("dup2.o")
    dup2.add_section(".text", bytes(8))
    dup2.add_symbol("func", SymbolBind.GLOBAL, ".text", 0, 8)

    linker3 = Linker()
    linker3.link([dup1, dup2])
    for err in linker3.errors:
        print(f"  {err}")

    print(f"\n--- Static vs Dynamic Linking ---")
    print("""
  Static Linking:
    + Self-contained executable (no runtime dependencies)
    + Faster startup (no dynamic resolution)
    - Larger executable size
    - No shared memory between processes
    - Must recompile to update library code

  Dynamic Linking:
    + Smaller executables (shared library code)
    + Shared memory for library code across processes
    + Update library without recompiling
    - Slower startup (symbol resolution at load time)
    - DLL hell / version conflicts
    - Requires Position-Independent Code (PIC)
    """)


if __name__ == "__main__":
    main()
