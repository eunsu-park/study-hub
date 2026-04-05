"""
Example: Reverse Engineering
==============================
Control flow graph builder, string deobfuscation, disassembly patterns,
and function signature heuristics.

IMPORTANT: For authorized security testing and CTF only.
"""

import base64
from dataclasses import dataclass, field
from collections import defaultdict


# ---------------------------------------------------------------------------
# Control Flow Graph
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    address: int
    size: int
    instructions: list[str]
    successors: list[int] = field(default_factory=list)

    @property
    def end_address(self) -> int:
        return self.address + self.size


class ControlFlowGraph:
    """Simple control flow graph representation."""

    def __init__(self):
        self.blocks: dict[int, BasicBlock] = {}
        self.entry: int = 0

    def add_block(self, block: BasicBlock):
        self.blocks[block.address] = block

    def predecessors(self, addr: int) -> list[int]:
        return [a for a, b in self.blocks.items() if addr in b.successors]

    def find_loops(self) -> list[tuple[int, int]]:
        """Find back edges (loops) using DFS."""
        visited = set()
        in_stack = set()
        back_edges = []

        def dfs(node):
            visited.add(node)
            in_stack.add(node)
            block = self.blocks.get(node)
            if block:
                for succ in block.successors:
                    if succ in in_stack:
                        back_edges.append((node, succ))
                    elif succ not in visited:
                        dfs(succ)
            in_stack.discard(node)

        if self.entry in self.blocks:
            dfs(self.entry)
        return back_edges

    def to_adjacency(self) -> dict[int, list[int]]:
        return {addr: block.successors for addr, block in self.blocks.items()}

    def display(self) -> str:
        lines = [f"CFG (entry=0x{self.entry:x}, {len(self.blocks)} blocks)"]
        for addr in sorted(self.blocks):
            block = self.blocks[addr]
            succs = ", ".join(f"0x{s:x}" for s in block.successors)
            lines.append(f"  0x{addr:x} ({block.size}B) -> [{succs}]")
        loops = self.find_loops()
        if loops:
            lines.append(f"  Loops: {[(f'0x{a:x}', f'0x{b:x}') for a, b in loops]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# String Deobfuscation Techniques
# ---------------------------------------------------------------------------

def xor_deobfuscate(data: list[int], key: int) -> str:
    """Deobfuscate XOR-encoded string."""
    return "".join(chr(b ^ key) for b in data)


def reverse_base64(encoded: str) -> str:
    """Decode reversed base64 string."""
    return base64.b64decode(encoded[::-1]).decode()


def stack_string(chars: list[int]) -> str:
    """Reconstruct stack string from character codes."""
    return "".join(chr(c) for c in chars)


def rolling_xor(data: list[int]) -> str:
    """Deobfuscate with rolling XOR (each byte XORed with previous)."""
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(data[i] ^ data[i - 1])
    return "".join(chr(b) for b in result)


# ---------------------------------------------------------------------------
# Common Assembly Patterns
# ---------------------------------------------------------------------------

ASM_PATTERNS = {
    "function_prologue": [
        "push rbp",
        "mov rbp, rsp",
        "sub rsp, N",
    ],
    "function_epilogue": [
        "leave",  # = mov rsp, rbp; pop rbp
        "ret",
    ],
    "loop": [
        "cmp ecx, 0",
        "je end",
        "...",
        "dec ecx",
        "jmp loop",
    ],
    "switch_table": [
        "cmp eax, MAX_CASE",
        "ja default",
        "jmp [table + eax*8]",
    ],
    "stack_canary_check": [
        "mov rax, [rbp-8]",
        "xor rax, fs:[0x28]",
        "je .ok",
        "call __stack_chk_fail",
    ],
}


# ---------------------------------------------------------------------------
# Function Signature Heuristics
# ---------------------------------------------------------------------------

KNOWN_SIGNATURES = {
    "strlen": {"args": 1, "returns": "int",
               "pattern": "loops reading bytes until null"},
    "memcpy": {"args": 3, "returns": "void*",
               "pattern": "copies N bytes from src to dst"},
    "strcmp": {"args": 2, "returns": "int",
               "pattern": "compares two strings byte by byte"},
    "malloc": {"args": 1, "returns": "void*",
               "pattern": "calls brk/mmap, returns pointer"},
    "free": {"args": 1, "returns": "void",
             "pattern": "marks heap chunk as free"},
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Reverse Engineering Examples")
    print("=" * 50)

    # CFG
    cfg = ControlFlowGraph()
    cfg.entry = 0x401000
    cfg.add_block(BasicBlock(0x401000, 10, ["cmp", "jle"],
                             [0x401020, 0x401040]))
    cfg.add_block(BasicBlock(0x401020, 8, ["add", "jmp"], [0x401000]))
    cfg.add_block(BasicBlock(0x401040, 6, ["mov", "ret"], []))
    print(f"\n{cfg.display()}")

    # String deobfuscation
    print("\nString Deobfuscation:")
    xor_data = [0x2a, 0x27, 0x2e, 0x2e, 0x21]
    print(f"  XOR(0x42): {xor_deobfuscate(xor_data, 0x42)}")

    stack_chars = [0x48, 0x65, 0x6C, 0x6C, 0x6F]
    print(f"  Stack string: {stack_string(stack_chars)}")

    print("\nCommon ASM Patterns:")
    for name, instrs in ASM_PATTERNS.items():
        print(f"  {name}: {' ; '.join(instrs[:3])}")

    print("\nKnown Function Signatures:")
    for name, sig in KNOWN_SIGNATURES.items():
        print(f"  {name}({sig['args']} args) -> {sig['returns']}")


if __name__ == "__main__":
    demo()
