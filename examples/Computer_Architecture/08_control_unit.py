"""
Control Unit Simulator

Demonstrates:
- Hardwired vs microprogrammed control
- Control signal generation
- Finite state machine for instruction sequencing
- Microinstruction format and execution

Theory:
- The control unit orchestrates datapath operations by asserting
  control signals at the right time in the right sequence.
- Hardwired control uses combinational logic (fast but inflexible).
- Microprogrammed control stores micro-operations in a control
  store ROM (flexible but slower due to extra memory access).
- Each machine instruction maps to a sequence of micro-operations
  that drive the ALU, register file, memory, and multiplexers.

Adapted from Computer Architecture Lesson 08.
"""

from dataclasses import dataclass
from enum import IntFlag, auto


# ── Control Signals ───────────────────────────────────────────────────

class Signal(IntFlag):
    """Control signals asserted by the control unit each cycle."""
    NONE        = 0
    PC_WRITE    = auto()  # Write to program counter
    PC_SRC      = auto()  # PC source: 0=PC+1, 1=branch target
    IR_WRITE    = auto()  # Latch instruction register
    MEM_READ    = auto()  # Read from memory
    MEM_WRITE   = auto()  # Write to memory
    MEM_TO_REG  = auto()  # Register write data from memory (vs ALU)
    REG_WRITE   = auto()  # Write to register file
    ALU_SRC     = auto()  # ALU input B: 0=register, 1=immediate
    ALU_OP_ADD  = auto()  # ALU operation: add
    ALU_OP_SUB  = auto()  # ALU operation: subtract
    ALU_OP_AND  = auto()  # ALU operation: bitwise AND


def signal_names(sig: Signal) -> list[str]:
    """Return list of active signal names."""
    return [s.name for s in Signal if s in sig and s != Signal.NONE]


# ── Microinstruction ──────────────────────────────────────────────────

@dataclass
class MicroInstr:
    """One microinstruction in the control store.

    Each microinstruction specifies which control signals to assert
    and which micro-address to go to next.
    """
    label: str
    signals: Signal
    next_addr: int = -1  # -1 = sequential, else jump to address
    dispatch: bool = False  # if True, next address depends on opcode

    def __repr__(self) -> str:
        sigs = signal_names(self.signals) or ["NONE"]
        return f"{self.label}: {', '.join(sigs)}"


# ── Microprogrammed Control Unit ──────────────────────────────────────

class MicroprogrammedControlUnit:
    """Control unit driven by a control store (microprogram).

    The control store is a small ROM indexed by micro-PC.  Each
    microinstruction specifies control signals for one cycle and
    the address of the next microinstruction.  A dispatch table
    maps opcodes to the first microinstruction of each routine.
    """

    def __init__(self):
        # Control store: list of microinstructions
        self.control_store: list[MicroInstr] = []
        # Dispatch table: opcode → control store address
        self.dispatch_table: dict[str, int] = {}
        self.micro_pc: int = 0
        self.trace: list[str] = []

    def load_microprogram(self) -> None:
        """Load a microprogram for a simple CPU.

        Instruction classes:
        - R-type (ADD, SUB, AND): register-register ALU
        - LOAD: memory read to register
        - STORE: register to memory write
        """
        self.control_store = [
            # 0: Common fetch — read instruction from memory
            MicroInstr("FETCH",
                       Signal.MEM_READ | Signal.IR_WRITE,
                       next_addr=1),
            # 1: Increment PC, decode opcode, dispatch
            MicroInstr("DECODE",
                       Signal.PC_WRITE,
                       dispatch=True),

            # 2-3: R-type (ADD)
            MicroInstr("ADD_EX",
                       Signal.ALU_OP_ADD,
                       next_addr=3),
            MicroInstr("ADD_WB",
                       Signal.REG_WRITE,
                       next_addr=0),

            # 4-5: R-type (SUB)
            MicroInstr("SUB_EX",
                       Signal.ALU_OP_SUB,
                       next_addr=5),
            MicroInstr("SUB_WB",
                       Signal.REG_WRITE,
                       next_addr=0),

            # 6-7: R-type (AND)
            MicroInstr("AND_EX",
                       Signal.ALU_OP_AND,
                       next_addr=7),
            MicroInstr("AND_WB",
                       Signal.REG_WRITE,
                       next_addr=0),

            # 8-10: LOAD — compute address, read memory, write register
            MicroInstr("LOAD_ADDR",
                       Signal.ALU_SRC | Signal.ALU_OP_ADD,
                       next_addr=9),
            MicroInstr("LOAD_MEM",
                       Signal.MEM_READ,
                       next_addr=10),
            MicroInstr("LOAD_WB",
                       Signal.MEM_TO_REG | Signal.REG_WRITE,
                       next_addr=0),

            # 11-12: STORE — compute address, write memory
            MicroInstr("STORE_ADDR",
                       Signal.ALU_SRC | Signal.ALU_OP_ADD,
                       next_addr=12),
            MicroInstr("STORE_MEM",
                       Signal.MEM_WRITE,
                       next_addr=0),
        ]

        self.dispatch_table = {
            "ADD":   2,
            "SUB":   4,
            "AND":   6,
            "LOAD":  8,
            "STORE": 11,
        }

    def execute_instruction(self, opcode: str) -> list[tuple[str, list[str]]]:
        """Execute one machine instruction through the microprogram.

        Returns list of (micro_label, [signal_names]) per micro-step.
        """
        self.micro_pc = 0  # start at FETCH
        steps = []

        for _ in range(20):  # safety limit
            uinstr = self.control_store[self.micro_pc]
            sigs = signal_names(uinstr.signals)
            steps.append((uinstr.label, sigs))
            self.trace.append(
                f"  uPC={self.micro_pc:>2}  {uinstr.label:<12}  "
                f"signals: {', '.join(sigs) or 'NONE'}")

            # Determine next micro-address
            if uinstr.dispatch:
                # Use dispatch table to jump based on opcode
                if opcode in self.dispatch_table:
                    self.micro_pc = self.dispatch_table[opcode]
                else:
                    raise ValueError(f"Unknown opcode: {opcode}")
            elif uinstr.next_addr == 0 and self.micro_pc != 0:
                # Return to FETCH = instruction complete
                break
            elif uinstr.next_addr >= 0:
                self.micro_pc = uinstr.next_addr
            else:
                self.micro_pc += 1

        return steps


# ── Hardwired Control (FSM) ──────────────────────────────────────────

class HardwiredControlFSM:
    """Hardwired control unit modeled as a finite state machine.

    States correspond to pipeline phases.  Transitions depend on
    the opcode being executed.  In real hardware this is built from
    combinational logic and flip-flops — fast but hard to modify.
    """

    STATES = {
        "FETCH":     {"next": "DECODE",  "signals": ["MEM_READ", "IR_WRITE"]},
        "DECODE":    {"next": None,      "signals": ["PC_WRITE"]},
        "ALU_EX":    {"next": "ALU_WB",  "signals": []},  # filled per opcode
        "ALU_WB":    {"next": "FETCH",   "signals": ["REG_WRITE"]},
        "MEM_ADDR":  {"next": None,      "signals": ["ALU_SRC", "ALU_OP_ADD"]},
        "MEM_READ":  {"next": "MEM_WB",  "signals": ["MEM_READ"]},
        "MEM_WB":    {"next": "FETCH",   "signals": ["MEM_TO_REG", "REG_WRITE"]},
        "MEM_WRITE": {"next": "FETCH",   "signals": ["MEM_WRITE"]},
    }

    # Opcode → sequence of states after DECODE
    OPCODE_PATHS = {
        "ADD":   ["ALU_EX", "ALU_WB"],
        "SUB":   ["ALU_EX", "ALU_WB"],
        "AND":   ["ALU_EX", "ALU_WB"],
        "LOAD":  ["MEM_ADDR", "MEM_READ", "MEM_WB"],
        "STORE": ["MEM_ADDR", "MEM_WRITE"],
    }

    ALU_SIGNALS = {
        "ADD": ["ALU_OP_ADD"],
        "SUB": ["ALU_OP_SUB"],
        "AND": ["ALU_OP_AND"],
    }

    def execute(self, opcode: str) -> list[tuple[str, list[str]]]:
        """Execute instruction, returning (state, signals) per cycle."""
        steps = []

        # FETCH
        steps.append(("FETCH", self.STATES["FETCH"]["signals"]))
        # DECODE
        steps.append(("DECODE", self.STATES["DECODE"]["signals"]))

        path = self.OPCODE_PATHS.get(opcode, [])
        for state in path:
            sigs = list(self.STATES[state]["signals"])
            if state == "ALU_EX" and opcode in self.ALU_SIGNALS:
                sigs = self.ALU_SIGNALS[opcode]
            steps.append((state, sigs))

        return steps


# ── Demos ─────────────────────────────────────────────────────────────

def demo_microprogrammed():
    """Microprogrammed control unit execution."""
    print("=" * 60)
    print("MICROPROGRAMMED CONTROL UNIT")
    print("=" * 60)

    cu = MicroprogrammedControlUnit()
    cu.load_microprogram()

    print(f"\n  Control store size: {len(cu.control_store)} microinstructions")
    print(f"  Dispatch table: {cu.dispatch_table}")

    for opcode in ["ADD", "LOAD", "STORE"]:
        print(f"\n  Instruction: {opcode}")
        print(f"  {'Step':<4} {'uLabel':<14} {'Signals'}")
        print(f"  {'-'*4} {'-'*14} {'-'*30}")

        cu.trace.clear()
        steps = cu.execute_instruction(opcode)
        for i, (label, sigs) in enumerate(steps):
            print(f"  {i:<4} {label:<14} {', '.join(sigs) or 'NONE'}")
        print(f"  Micro-steps: {len(steps)}")


def demo_hardwired():
    """Hardwired control (FSM) execution."""
    print("\n" + "=" * 60)
    print("HARDWIRED CONTROL (FSM)")
    print("=" * 60)

    fsm = HardwiredControlFSM()

    for opcode in ["ADD", "SUB", "LOAD", "STORE"]:
        steps = fsm.execute(opcode)
        print(f"\n  Instruction: {opcode}")
        print(f"  {'Cycle':<6} {'State':<12} {'Signals'}")
        print(f"  {'-'*6} {'-'*12} {'-'*30}")
        for i, (state, sigs) in enumerate(steps):
            print(f"  {i:<6} {state:<12} {', '.join(sigs) or 'NONE'}")
        print(f"  Total cycles: {len(steps)}")


def demo_comparison():
    """Compare microprogrammed vs hardwired cycle counts."""
    print("\n" + "=" * 60)
    print("CONTROL UNIT COMPARISON")
    print("=" * 60)

    micro = MicroprogrammedControlUnit()
    micro.load_microprogram()
    hardwired = HardwiredControlFSM()

    print(f"\n  {'Opcode':<10} {'Microprogrammed':>16} {'Hardwired (FSM)':>16}")
    print(f"  {'-'*10} {'-'*16} {'-'*16}")

    for opcode in ["ADD", "SUB", "AND", "LOAD", "STORE"]:
        micro.trace.clear()
        m_steps = micro.execute_instruction(opcode)
        h_steps = hardwired.execute(opcode)
        print(f"  {opcode:<10} {len(m_steps):>12} cyc {len(h_steps):>12} cyc")

    print("""
  Microprogrammed advantages:
  - Easy to modify (update ROM, not logic gates)
  - Supports complex instructions (CISC)
  - Simpler verification

  Hardwired advantages:
  - Faster (no control store access delay)
  - Less hardware for simple ISAs (RISC)
  - Lower power consumption
""")


if __name__ == "__main__":
    demo_microprogrammed()
    demo_hardwired()
    demo_comparison()
