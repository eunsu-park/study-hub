"""
Exercises for Lesson 11: Code Generation
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# === Exercise 1: Maximal Munch ===
# Problem: Apply Maximal Munch to an IR tree for instruction selection.

def exercise_1():
    """Maximal Munch algorithm for instruction selection."""
    print("IR Tree:")
    print("        STORE")
    print("       /     \\")
    print("      +       MEM")
    print("     / \\       |")
    print("    FP  #-8   +")
    print("             / \\")
    print("            *   #4")
    print("           / \\")
    print("          i   #4")
    print()

    print("Available tiles:")
    print("  ADDI Rd, Rs, #imm   -- add immediate")
    print("  SHL  Rd, Rs, #imm   -- shift left (for power-of-2 multiply)")
    print("  LOAD Rd, [Rs+Ri]    -- indexed load")
    print("  STORE [Rs+#off], Rd -- store with offset")
    print()

    print("Maximal Munch (top-down, greedily match largest tile):")
    print()

    steps = [
        (
            "Step 1: Match root STORE",
            "STORE pattern: STORE(MEM(addr), value)",
            "Check if we can tile STORE with offset addressing: STORE [FP + #-8], Rd",
            "Left child is FP + #-8 -> matches STORE [Rs + #off], Rd pattern",
            "Emit: STORE [FP + #-8], Rd  (need to compute Rd = MEM(...))"
        ),
        (
            "Step 2: Compute the source value (right subtree: MEM(+ (* i #4) #4))",
            "MEM(addr) -> LOAD instruction",
            "addr = (i * 4) + 4 -> can use indexed addressing?",
            "Recognize i * 4 as SHL: i << 2 (power-of-2 multiply)",
            "Emit: SHL R1, Ri, #2       ; R1 = i * 4"
        ),
        (
            "Step 3: Add offset",
            "addr = R1 + 4",
            "Emit: ADDI R2, R1, #4      ; R2 = i*4 + 4"
        ),
        (
            "Step 4: Load from computed address",
            "Emit: LOAD R3, [R2]        ; R3 = MEM[i*4 + 4]"
        ),
        (
            "Step 5: Store to destination",
            "Emit: STORE [FP + #-8], R3 ; store result"
        ),
    ]

    for title, *details in steps:
        print(f"  {title}")
        for d in details:
            print(f"    {d}")
        print()

    print("Final instruction sequence:")
    instructions = [
        "SHL   R1, Ri, #2",
        "ADDI  R2, R1, #4",
        "LOAD  R3, [R2]",
        "STORE [FP + #-8], R3",
    ]
    for instr in instructions:
        print(f"  {instr}")
    print()
    print("  Alternative (if indexed LOAD is available):")
    print("  SHL   R1, Ri, #2")
    print("  LOAD  R3, [R1 + #4]       ; indexed with immediate offset")
    print("  STORE [FP + #-8], R3")


# === Exercise 2: Register Allocation (Linear Scan) ===
# Problem: Linear scan allocation with 3 registers.

def exercise_2():
    """Linear scan register allocation with spilling."""
    intervals = [
        ('a', 1, 15),
        ('b', 2, 10),
        ('c', 3, 12),
        ('d', 5, 8),
        ('e', 7, 20),
        ('f', 13, 18),
    ]
    num_registers = 3

    print(f"Live intervals (sorted by start):")
    for name, start, end in intervals:
        bar = ' ' * (start - 1) + '-' * (end - start + 1)
        print(f"  {name}: [{start:>2}, {end:>2}]  {bar}")
    print(f"  Available registers: {num_registers}")
    print()

    # Linear scan algorithm
    active = []  # list of (end, name, register)
    free_regs = list(range(num_registers))  # available registers
    allocation = {}
    spilled = []

    print("Linear Scan Allocation:")
    for name, start, end in sorted(intervals, key=lambda x: x[1]):
        # Expire old intervals
        expired = []
        for i, (act_end, act_name, reg) in enumerate(active):
            if act_end < start:
                expired.append(i)
        for i in sorted(expired, reverse=True):
            _, _, reg = active.pop(i)
            free_regs.append(reg)
            free_regs.sort()

        if free_regs:
            reg = free_regs.pop(0)
            allocation[name] = f"R{reg}"
            active.append((end, name, reg))
            active.sort()
            print(f"  {name} [{start},{end}]: assigned R{reg}")
        else:
            # Spill: find the interval in active with the farthest end point
            # If that end > current end, spill that one; otherwise spill current
            if active and active[-1][0] > end:
                spill_end, spill_name, reg = active.pop()
                spilled.append(spill_name)
                allocation[spill_name] = "SPILLED"
                allocation[name] = f"R{reg}"
                active.append((end, name, reg))
                active.sort()
                print(f"  {name} [{start},{end}]: spill {spill_name}, assign R{reg} to {name}")
            else:
                spilled.append(name)
                allocation[name] = "SPILLED"
                print(f"  {name} [{start},{end}]: SPILLED (no register available)")

    print()
    print("Final allocation:")
    for name, reg in allocation.items():
        print(f"  {name} -> {reg}")
    print(f"  Spilled: {spilled}")


# === Exercise 3: Instruction Scheduling ===
# Problem: Schedule instructions with 1 ALU and 1 load/store unit.

def exercise_3():
    """List scheduling for instruction-level parallelism."""
    print("Instructions:")
    print("  I1: LOAD R1, [addr1]   ; load, latency 3")
    print("  I2: LOAD R2, [addr2]   ; load, latency 3")
    print("  I3: ADD  R3, R1, R2    ; ALU, latency 1, depends I1,I2")
    print("  I4: LOAD R4, [addr3]   ; load, latency 3")
    print("  I5: MUL  R5, R3, R4    ; ALU, latency 3, depends I3,I4")
    print("  I6: ADD  R6, R5, #1    ; ALU, latency 1, depends I5")
    print()

    # Dependencies: I3 depends on I1,I2; I5 depends on I3,I4; I6 depends on I5
    # Resources: 1 ALU, 1 Load/Store
    # Latencies: LOAD=3, ADD=1, MUL=3

    print("Dependency graph:")
    print("  I1 --3--> I3")
    print("  I2 --3--> I3")
    print("  I3 --1--> I5")
    print("  I4 --3--> I5")
    print("  I5 --3--> I6")
    print()

    schedule = [
        (1, "I1: LOAD R1, [addr1]", "Load unit",  "ALU idle"),
        (2, "I2: LOAD R2, [addr2]", "Load unit",  "ALU idle"),
        (3, "I4: LOAD R4, [addr3]", "Load unit",  "ALU idle (I1 not ready yet)"),
        (4, "I3: ADD  R3, R1, R2",  "ALU unit",   "Load idle (I1,I2 ready at cycle 4)"),
        (5, "--- stall ---",        "waiting",     "I3 result ready, I4 not ready yet"),
        (6, "I5: MUL  R5, R3, R4",  "ALU unit",   "Load idle (I4 ready at cycle 6)"),
        (7, "--- stall ---",        "waiting",     "MUL latency"),
        (8, "--- stall ---",        "waiting",     "MUL latency"),
        (9, "I6: ADD  R6, R5, #1",  "ALU unit",   "I5 ready at cycle 9"),
    ]

    print("Schedule (1 ALU + 1 Load/Store unit):")
    print(f"  {'Cycle':<6} {'Instruction':<30} {'Unit':<12} {'Notes'}")
    print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*35}")
    for cycle, instr, unit, notes in schedule:
        print(f"  {cycle:<6} {instr:<30} {unit:<12} {notes}")
    print()
    print(f"  Total cycles: {schedule[-1][0]}")
    print(f"  Minimum possible: 9 cycles (critical path: I1/I2(3) + I3(1) + I5(3) + I6(1) + stalls)")
    print(f"  Key optimization: I4 is scheduled in cycle 3 to overlap with I1/I2 latency.")


# === Exercise 4: Peephole Optimization ===
# Problem: Apply peephole optimizations to given code.

def exercise_4():
    """Apply peephole optimizations to instruction sequence."""
    original = [
        "MOVI R1, #10",
        "ADD  R2, R1, #0",     # identity: x + 0 = x
        "MUL  R3, R2, #16",    # strength reduction: x * 16 = x << 4
        "MOV  R4, R4",         # redundant self-move
        "STORE R3, [R5]",
        "LOAD  R3, [R5]",      # redundant load after store to same location
        "MUL  R6, R3, #1",     # identity: x * 1 = x
        "DIV  R7, R6, #8",     # strength reduction: x / 8 = x >> 3
        "JMP  L1",
        "ADD  R8, R1, R2",     # dead code (unreachable after JMP)
        "L1:",
        "JMP  L2",             # jump chain: JMP L1 -> JMP L2 = JMP L2
        "L2:",
        "RET",
    ]

    print("Original code:")
    for line in original:
        print(f"  {line}")
    print()

    # Pass 1: Algebraic simplification and strength reduction
    pass1 = [
        "MOVI R1, #10",
        "MOV  R2, R1",          # ADD R2, R1, #0 -> MOV R2, R1 (identity)
        "SHL  R3, R2, #4",     # MUL R3, R2, #16 -> SHL (strength reduction)
        "MOV  R4, R4",         # redundant (will be removed next pass)
        "STORE R3, [R5]",
        "LOAD  R3, [R5]",      # redundant (will be removed)
        "MOV  R6, R3",         # MUL R6, R3, #1 -> MOV (identity)
        "SHR  R7, R6, #3",    # DIV R7, R6, #8 -> SHR (strength reduction)
        "JMP  L1",
        "ADD  R8, R1, R2",     # dead code
        "L1:",
        "JMP  L2",
        "L2:",
        "RET",
    ]
    print("Pass 1 (algebraic simplification + strength reduction):")
    for line in pass1:
        print(f"  {line}")
    print()

    # Pass 2: Redundant instruction elimination
    pass2 = [
        "MOVI R1, #10",
        "MOV  R2, R1",
        "SHL  R3, R2, #4",
        # MOV R4, R4 removed (self-move)
        "STORE R3, [R5]",
        # LOAD R3, [R5] removed (redundant: R3 already has the value)
        "MOV  R6, R3",
        "SHR  R7, R6, #3",
        "JMP  L1",
        # ADD R8, R1, R2 removed (unreachable dead code)
        "L1:",
        "JMP  L2",
        "L2:",
        "RET",
    ]
    print("Pass 2 (redundant elimination + dead code removal):")
    for line in pass2:
        print(f"  {line}")
    print()

    # Pass 3: Jump chain elimination
    pass3 = [
        "MOVI R1, #10",
        "MOV  R2, R1",
        "SHL  R3, R2, #4",
        "STORE R3, [R5]",
        "MOV  R6, R3",
        "SHR  R7, R6, #3",
        "JMP  L2",              # L1: JMP L2 -> direct JMP L2 (chain elimination)
        # L1: removed (only had JMP L2)
        "L2:",
        "RET",
    ]
    print("Pass 3 (jump chain elimination):")
    for line in pass3:
        print(f"  {line}")
    print()

    # Pass 4: Further copy propagation
    pass4 = [
        "MOVI R1, #10",
        # MOV R2, R1 -- propagate R2 -> R1
        "SHL  R3, R1, #4",     # R2 replaced with R1
        "STORE R3, [R5]",
        # MOV R6, R3 -- propagate R6 -> R3
        "SHR  R7, R3, #3",    # R6 replaced with R3
        "RET",                  # JMP to next instruction = RET directly
    ]
    print("Pass 4 (copy propagation + final cleanup):")
    for line in pass4:
        print(f"  {line}")
    print()
    print(f"  Reduced from {len(original)} instructions to {len(pass4)} instructions")


# === Exercise 5: Stack Machine ===
# Problem: Compile (3+4)*(10-2)/(1+1) to stack bytecode and trace.

def exercise_5():
    """Stack machine bytecode compilation and execution trace."""
    print("Expression: result = (3 + 4) * (10 - 2) / (1 + 1)")
    print()

    bytecode = [
        ("PUSH 3",     "push 3"),
        ("PUSH 4",     "push 4"),
        ("ADD",        "3 + 4 = 7"),
        ("PUSH 10",    "push 10"),
        ("PUSH 2",     "push 2"),
        ("SUB",        "10 - 2 = 8"),
        ("MUL",        "7 * 8 = 56"),
        ("PUSH 1",     "push 1"),
        ("PUSH 1",     "push 1"),
        ("ADD",        "1 + 1 = 2"),
        ("DIV",        "56 / 2 = 28"),
        ("STORE result", "store top to 'result'"),
    ]

    print("Bytecode and stack trace:")
    print(f"  {'Instruction':<18} {'Stack After':<25} {'Comment'}")
    print(f"  {'-'*18} {'-'*25} {'-'*25}")

    stack = []
    variables = {}

    for instr, comment in bytecode:
        parts = instr.split()
        op = parts[0]

        if op == 'PUSH':
            stack.append(int(parts[1]))
        elif op == 'ADD':
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == 'SUB':
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif op == 'MUL':
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif op == 'DIV':
            b, a = stack.pop(), stack.pop()
            stack.append(a // b)
        elif op == 'STORE':
            variables[parts[1]] = stack.pop()

        stack_str = str(stack)
        print(f"  {instr:<18} {stack_str:<25} {comment}")

    print()
    print(f"  result = {variables.get('result', 'undefined')}")


# === Exercise 6: Implementation Challenge ===
# Problem: Stack machine VM with arrays and for loops.

class StackMachineVM:
    """Stack-based virtual machine with arrays and for loops."""

    # Opcodes
    PUSH = 'PUSH'
    POP = 'POP'
    ADD = 'ADD'
    SUB = 'SUB'
    MUL = 'MUL'
    DIV = 'DIV'
    LOAD = 'LOAD'      # load variable
    STORE = 'STORE'     # store variable
    ALLOC = 'ALLOC'     # allocate array of size n
    ALOAD = 'ALOAD'     # load from array: stack=[..., arr, idx] -> [..., arr[idx]]
    ASTORE = 'ASTORE'   # store to array: stack=[..., val, arr, idx] -> store val at arr[idx]
    JMP = 'JMP'
    JMPF = 'JMPF'      # jump if false (top of stack is 0)
    LT = 'LT'          # less than comparison
    PRINT = 'PRINT'
    HALT = 'HALT'

    def __init__(self):
        self.stack = []
        self.variables = {}
        self.arrays = {}
        self.output = []
        self.array_counter = 0

    def run(self, program):
        """Execute a program (list of instructions)."""
        pc = 0
        while pc < len(program):
            instr = program[pc]
            op = instr[0]

            if op == self.PUSH:
                self.stack.append(instr[1])
            elif op == self.POP:
                self.stack.pop()
            elif op == self.ADD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)
            elif op == self.SUB:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)
            elif op == self.MUL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)
            elif op == self.DIV:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a // b)
            elif op == self.LT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a < b else 0)
            elif op == self.LOAD:
                name = instr[1]
                self.stack.append(self.variables[name])
            elif op == self.STORE:
                name = instr[1]
                self.variables[name] = self.stack.pop()
            elif op == self.ALLOC:
                size = self.stack.pop()
                arr_id = f"arr_{self.array_counter}"
                self.array_counter += 1
                self.arrays[arr_id] = [0] * size
                self.stack.append(arr_id)
            elif op == self.ALOAD:
                idx = self.stack.pop()
                arr_id = self.stack.pop()
                self.stack.append(self.arrays[arr_id][idx])
            elif op == self.ASTORE:
                idx = self.stack.pop()
                arr_id = self.stack.pop()
                val = self.stack.pop()
                self.arrays[arr_id][idx] = val
            elif op == self.JMP:
                pc = instr[1]
                continue
            elif op == self.JMPF:
                cond = self.stack.pop()
                if cond == 0:
                    pc = instr[1]
                    continue
            elif op == self.PRINT:
                val = self.stack.pop()
                self.output.append(str(val))
            elif op == self.HALT:
                break

            pc += 1

        return self.output


def exercise_6():
    """Stack machine VM with arrays and for loops."""
    vm = StackMachineVM()

    # Program: allocate array, fill with squares, print contents
    # for i = 0 to 9:
    #     arr[i] = i * i
    # for i = 0 to 9:
    #     print(arr[i])
    n = 10
    program = [
        # Allocate array of size 10
        (StackMachineVM.PUSH, n),      # 0: push size
        (StackMachineVM.ALLOC,),       # 1: allocate
        (StackMachineVM.STORE, 'arr'), # 2: store array ref

        # i = 0
        (StackMachineVM.PUSH, 0),      # 3
        (StackMachineVM.STORE, 'i'),   # 4

        # Loop 1: fill with squares
        # 5: loop start
        (StackMachineVM.LOAD, 'i'),    # 5: load i
        (StackMachineVM.PUSH, n),      # 6: push n
        (StackMachineVM.LT,),          # 7: i < n?
        (StackMachineVM.JMPF, 16),     # 8: if false, jump past loop

        # arr[i] = i * i
        (StackMachineVM.LOAD, 'i'),    # 9: push i (value)
        (StackMachineVM.LOAD, 'i'),    # 10: push i
        (StackMachineVM.MUL,),         # 11: i * i
        (StackMachineVM.LOAD, 'arr'),  # 12: push arr ref
        (StackMachineVM.LOAD, 'i'),    # 13: push index
        (StackMachineVM.ASTORE,),      # 14: arr[i] = i*i

        # Note: ASTORE pops idx, arr, val from stack. We need stack order: val, arr, idx
        # Let me fix the order: stack should be [..., val, arr, idx] for ASTORE

        # i = i + 1
        (StackMachineVM.LOAD, 'i'),    # 15
        (StackMachineVM.PUSH, 1),      # 16
        (StackMachineVM.ADD,),         # 17
        (StackMachineVM.STORE, 'i'),   # 18
        (StackMachineVM.JMP, 5),       # 19: back to loop start

        # 20: After loop 1
        # i = 0
        (StackMachineVM.PUSH, 0),      # 20
        (StackMachineVM.STORE, 'i'),   # 21

        # Loop 2: print array contents
        # 22: loop start
        (StackMachineVM.LOAD, 'i'),    # 22
        (StackMachineVM.PUSH, n),      # 23
        (StackMachineVM.LT,),          # 24
        (StackMachineVM.JMPF, 31),     # 25

        # print(arr[i])
        (StackMachineVM.LOAD, 'arr'),  # 26
        (StackMachineVM.LOAD, 'i'),    # 27
        (StackMachineVM.ALOAD,),       # 28
        (StackMachineVM.PRINT,),       # 29

        # i = i + 1
        (StackMachineVM.LOAD, 'i'),    # 30
        (StackMachineVM.PUSH, 1),      # 31
        (StackMachineVM.ADD,),         # 32
        (StackMachineVM.STORE, 'i'),   # 33
        (StackMachineVM.JMP, 22),      # 34

        # 35: end
        (StackMachineVM.HALT,),        # 35
    ]

    # Fix: The JMPF targets need adjustment after fixing instruction numbers.
    # Let me rebuild with correct addresses.
    program = []
    # Allocate array
    program.append((StackMachineVM.PUSH, n))       # 0
    program.append((StackMachineVM.ALLOC,))        # 1
    program.append((StackMachineVM.STORE, 'arr'))  # 2
    # i = 0
    program.append((StackMachineVM.PUSH, 0))       # 3
    program.append((StackMachineVM.STORE, 'i'))    # 4
    # Loop 1 header (address 5)
    program.append((StackMachineVM.LOAD, 'i'))     # 5
    program.append((StackMachineVM.PUSH, n))       # 6
    program.append((StackMachineVM.LT,))           # 7
    program.append((StackMachineVM.JMPF, 18))      # 8 -> jump to after loop1
    # arr[i] = i * i : stack needs [val, arr, idx]
    program.append((StackMachineVM.LOAD, 'i'))     # 9
    program.append((StackMachineVM.LOAD, 'i'))     # 10
    program.append((StackMachineVM.MUL,))          # 11: val = i*i on stack
    program.append((StackMachineVM.LOAD, 'arr'))   # 12: arr on stack
    program.append((StackMachineVM.LOAD, 'i'))     # 13: idx on stack
    program.append((StackMachineVM.ASTORE,))       # 14: store
    # i++
    program.append((StackMachineVM.LOAD, 'i'))     # 15
    program.append((StackMachineVM.PUSH, 1))       # 16
    program.append((StackMachineVM.ADD,))          # 17
    program.append((StackMachineVM.STORE, 'i'))    # 18
    program.append((StackMachineVM.JMP, 5))        # 19

    # After loop 1 -- fix JMPF target
    program[8] = (StackMachineVM.JMPF, 20)

    # i = 0
    program.append((StackMachineVM.PUSH, 0))       # 20
    program.append((StackMachineVM.STORE, 'i'))    # 21
    # Loop 2 header (address 22)
    program.append((StackMachineVM.LOAD, 'i'))     # 22
    program.append((StackMachineVM.PUSH, n))       # 23
    program.append((StackMachineVM.LT,))           # 24
    program.append((StackMachineVM.JMPF, 33))      # 25
    # print(arr[i])
    program.append((StackMachineVM.LOAD, 'arr'))   # 26
    program.append((StackMachineVM.LOAD, 'i'))     # 27
    program.append((StackMachineVM.ALOAD,))        # 28
    program.append((StackMachineVM.PRINT,))        # 29
    # i++
    program.append((StackMachineVM.LOAD, 'i'))     # 30
    program.append((StackMachineVM.PUSH, 1))       # 31
    program.append((StackMachineVM.ADD,))          # 32
    program.append((StackMachineVM.STORE, 'i'))    # 33
    program.append((StackMachineVM.JMP, 22))       # 34

    # Fix JMPF target for loop 2
    program[25] = (StackMachineVM.JMPF, 35)

    program.append((StackMachineVM.HALT,))         # 35

    print("Program: Allocate array of 10, fill with squares, print")
    print()
    print("Bytecode listing:")
    for i, instr in enumerate(program):
        print(f"  {i:>3}: {instr}")
    print()

    output = vm.run(program)
    print("Output:")
    expected = [str(i * i) for i in range(n)]
    for i, val in enumerate(output):
        status = "OK" if val == expected[i] else "FAIL"
        print(f"  [{status}] arr[{i}] = {val} (expected {i}^2 = {expected[i]})")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Maximal Munch ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Register Allocation (Linear Scan) ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Instruction Scheduling ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Peephole Optimization ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Stack Machine ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Implementation Challenge ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
