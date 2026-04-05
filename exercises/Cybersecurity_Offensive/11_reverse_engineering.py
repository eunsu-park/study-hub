"""
Exercises for Lesson 11: Reverse Engineering
Topic: Cybersecurity_Offensive

Practice problems covering disassembly, control flow reconstruction,
string analysis, and anti-reversing techniques.
"""


# === Exercise 1: Assembly to Python Translation ===
# Problem: Translate x86-64 assembly snippets to equivalent Python code.

def exercise_1():
    """
    Translate each assembly snippet to Python:

    Snippet A:
        mov eax, [rbp-4]    ; load local var
        imul eax, eax       ; multiply by itself
        add eax, [rbp-8]    ; add another local var
        cmp eax, 100
        jle .L1

    Snippet B:
        xor eax, eax        ; eax = 0
        .loop:
        add eax, ecx
        shr ecx, 1          ; shift right by 1
        test ecx, ecx
        jnz .loop

    For each, implement the equivalent Python function and describe
    what it computes.
    Return dict {"A": {"code": callable, "description": str},
                 "B": {"code": callable, "description": str}}
    """
    # TODO: Translate assembly to Python
    pass


# === Exercise 2: Control Flow Graph Builder ===
# Problem: Given a list of basic blocks with their branch targets,
# reconstruct the control flow graph as an adjacency list.
# Identify loops and conditional branches.

def exercise_2():
    """
    basic_blocks = [
        {"addr": 0x401000, "instructions": 5, "branch": "conditional",
         "true_target": 0x401020, "false_target": 0x401040},
        {"addr": 0x401020, "instructions": 3, "branch": "unconditional",
         "target": 0x401000},  # back edge = loop
        {"addr": 0x401040, "instructions": 4, "branch": "conditional",
         "true_target": 0x401060, "false_target": 0x401080},
        {"addr": 0x401060, "instructions": 2, "branch": "return"},
        {"addr": 0x401080, "instructions": 2, "branch": "return"},
    ]
    Return:
      {"cfg": dict (adjacency list), "loops": list of back edges,
       "entry": int, "exits": list[int]}
    """
    # TODO: Build control flow graph and identify loops
    pass


# === Exercise 3: String Deobfuscation ===
# Problem: Reverse common string obfuscation techniques.

def exercise_3():
    """
    Deobfuscate strings hidden with these techniques:

    A. XOR with key 0x42:
       obfuscated = [0x2a, 0x27, 0x2e, 0x2e, 0x21, 0x64, 0x15, 0x21, 0x30, 0x2e, 0x26]

    B. Reverse + Base64:
       obfuscated = "=QXZpRXa"  # reversed base64

    C. Stack string (character-by-character push):
       stack_chars = [0x48, 0x65, 0x6C, 0x6C, 0x6F]

    Return dict {"A": str, "B": str, "C": str} with decoded strings.
    """
    # TODO: Deobfuscate strings
    pass


# === Exercise 4: Function Signature Recovery ===
# Problem: Given function behavior observations, recover the likely
# function signature and purpose.

def exercise_4():
    """
    Observations from dynamic analysis:
    function_at_0x401200:
      - Takes 2 arguments (rdi = pointer, rsi = integer)
      - Reads bytes from memory at rdi
      - Loops rsi times
      - Compares each byte against 0x00
      - Returns count in eax
      - Behavior: f(b"hello\\x00world", 11) returns 5

    function_at_0x401300:
      - Takes 3 arguments (rdi = ptr, rsi = ptr, rdx = integer)
      - Copies rdx bytes from rsi to rdi
      - Returns rdi
      - Behavior matches memcpy

    Identify each function:
    Return list of {"address": str, "likely_function": str,
                    "signature": str, "confidence": str}
    """
    # TODO: Recover function signatures
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Assembly to Python ===")
    print(exercise_1())
    print("\n=== Exercise 2: Control Flow Graph ===")
    print(exercise_2())
    print("\n=== Exercise 3: String Deobfuscation ===")
    print(exercise_3())
    print("\n=== Exercise 4: Function Signature Recovery ===")
    print(exercise_4())
