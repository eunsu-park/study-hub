"""
Superscalar and Out-of-Order Execution

Demonstrates:
- Instruction-level parallelism (ILP) detection
- Tomasulo's algorithm concepts (reservation stations, CDB)
- Reorder Buffer (ROB) for in-order commit
- Dependency analysis (RAW, WAR, WAW hazards)

Theory:
- Superscalar processors issue multiple instructions per cycle
  by exploiting ILP — independent instructions that can execute
  simultaneously.
- Tomasulo's algorithm enables out-of-order execution via:
  - Reservation stations: buffer instructions waiting for operands
  - Common Data Bus (CDB): broadcast results to waiting stations
  - Register renaming: eliminates WAR and WAW hazards
- The Reorder Buffer (ROB) restores in-order commit semantics
  so that exceptions and interrupts see a consistent state.

Adapted from Computer Architecture Lesson 13.
"""

from dataclasses import dataclass, field
from collections import deque


# ── Dependency Analysis ───────────────────────────────────────────────

@dataclass
class Instruction:
    """Simple instruction with register operands."""
    name: str
    dest: str = ""       # destination register
    src1: str = ""       # source register 1
    src2: str = ""       # source register 2
    latency: int = 1     # execution latency in cycles
    idx: int = 0         # program order

    def __repr__(self) -> str:
        parts = [self.name]
        if self.dest:
            parts.append(self.dest)
        if self.src1:
            parts.append(self.src1)
        if self.src2:
            parts.append(self.src2)
        return " ".join(parts)


def analyze_dependencies(instructions: list[Instruction]) -> dict:
    """Detect data dependencies between instructions.

    Returns dict with lists of (from_idx, to_idx, register) tuples
    for each hazard type.
    """
    raw = []  # Read After Write — true dependency
    war = []  # Write After Read — anti-dependency
    waw = []  # Write After Write — output dependency

    for i, instr_i in enumerate(instructions):
        for j, instr_j in enumerate(instructions):
            if j <= i:
                continue

            # RAW: j reads what i writes
            if instr_i.dest:
                if instr_i.dest in (instr_j.src1, instr_j.src2):
                    raw.append((i, j, instr_i.dest))

            # WAR: j writes what i reads
            if instr_j.dest:
                if instr_j.dest in (instr_i.src1, instr_i.src2):
                    war.append((i, j, instr_j.dest))

            # WAW: j writes same register as i
            if instr_i.dest and instr_j.dest == instr_i.dest:
                waw.append((i, j, instr_i.dest))

    return {"RAW": raw, "WAR": war, "WAW": waw}


# ── Tomasulo Simulation ──────────────────────────────────────────────

@dataclass
class ReservationStation:
    """One reservation station entry."""
    name: str
    busy: bool = False
    op: str = ""
    vj: int = 0          # value of source 1
    vk: int = 0          # value of source 2
    qj: str = ""         # station producing source 1 (empty = ready)
    qk: str = ""         # station producing source 2 (empty = ready)
    dest: str = ""       # destination register
    rob_idx: int = -1    # ROB entry index
    cycles_left: int = 0
    result: int = 0


@dataclass
class ROBEntry:
    """Reorder Buffer entry for in-order commit."""
    idx: int
    instr: str
    dest: str = ""
    value: int = 0
    ready: bool = False
    committed: bool = False


class TomasuloSimulator:
    """Simplified Tomasulo algorithm with ROB.

    Models the key concepts: issue, execute, write-back, and commit
    stages.  Values are symbolic (not computed) — the focus is on
    scheduling and dependency resolution.
    """

    def __init__(self, n_add_stations: int = 3, n_mul_stations: int = 2):
        self.add_stations = [
            ReservationStation(name=f"Add{i}")
            for i in range(n_add_stations)
        ]
        self.mul_stations = [
            ReservationStation(name=f"Mul{i}")
            for i in range(n_mul_stations)
        ]
        self.rob: list[ROBEntry] = []
        self.register_status: dict[str, str] = {}  # reg → producing station
        self.cycle = 0
        self.log: list[str] = []
        self.timeline: dict[int, dict[str, int]] = {}  # instr_idx → stage → cycle

    def _get_stations(self, op: str) -> list[ReservationStation]:
        if op in ("ADD", "SUB"):
            return self.add_stations
        return self.mul_stations

    def _find_free_station(self, op: str) -> ReservationStation | None:
        for rs in self._get_stations(op):
            if not rs.busy:
                return rs
        return None

    def issue(self, instr: Instruction) -> bool:
        """Issue instruction to a reservation station if available."""
        rs = self._find_free_station(instr.name)
        if rs is None:
            return False

        # Create ROB entry
        rob_entry = ROBEntry(
            idx=len(self.rob),
            instr=repr(instr),
            dest=instr.dest,
        )
        self.rob.append(rob_entry)

        # Fill reservation station
        rs.busy = True
        rs.op = instr.name
        rs.dest = instr.dest
        rs.rob_idx = rob_entry.idx
        rs.cycles_left = instr.latency

        # Check source operand availability (register renaming)
        # If a register is being produced by another station, record
        # the dependency (qj/qk) instead of the value — this is the
        # essence of register renaming that eliminates WAR/WAW hazards.
        if instr.src1 and instr.src1 in self.register_status:
            rs.qj = self.register_status[instr.src1]
        else:
            rs.qj = ""
            rs.vj = hash(instr.src1) & 0xFF if instr.src1 else 0

        if instr.src2 and instr.src2 in self.register_status:
            rs.qk = self.register_status[instr.src2]
        else:
            rs.qk = ""
            rs.vk = hash(instr.src2) & 0xFF if instr.src2 else 0

        # Mark destination register as being produced by this station
        if instr.dest:
            self.register_status[instr.dest] = rs.name

        self.timeline.setdefault(instr.idx, {})["issue"] = self.cycle
        self.log.append(
            f"  Cycle {self.cycle:>2}: ISSUE  {repr(instr):<25} → {rs.name}")
        return True

    def execute_step(self) -> None:
        """Advance execution in all busy stations."""
        for rs in self.add_stations + self.mul_stations:
            if not rs.busy:
                continue
            # Can only execute when both operands are ready
            if rs.qj or rs.qk:
                continue
            rs.cycles_left -= 1
            if rs.cycles_left == 0:
                rs.result = rs.vj + rs.vk  # simplified
                # Find the instruction index for timeline
                for idx, entry in self.timeline.items():
                    if "issue" in entry and "exec_done" not in entry:
                        if self.rob[rs.rob_idx].instr:
                            entry["exec_done"] = self.cycle
                            break

    def write_back(self) -> None:
        """Write results via CDB, waking dependent stations."""
        for rs in self.add_stations + self.mul_stations:
            if not rs.busy or rs.cycles_left > 0:
                continue
            if rs.qj or rs.qk:
                continue

            # Broadcast on CDB — any station waiting on this result
            # can now proceed.  This is the key mechanism that enables
            # out-of-order execution without stalling the pipeline.
            for other in self.add_stations + self.mul_stations:
                if other.qj == rs.name:
                    other.qj = ""
                    other.vj = rs.result
                if other.qk == rs.name:
                    other.qk = ""
                    other.vk = rs.result

            # Mark ROB entry ready
            self.rob[rs.rob_idx].value = rs.result
            self.rob[rs.rob_idx].ready = True

            # Clear register status if still pointing to this station
            if rs.dest in self.register_status:
                if self.register_status[rs.dest] == rs.name:
                    del self.register_status[rs.dest]

            self.log.append(
                f"  Cycle {self.cycle:>2}: WRITE  {rs.name} → CDB "
                f"(ROB#{rs.rob_idx})")

            # Update timeline
            for idx, entry in self.timeline.items():
                if "exec_done" in entry and "writeback" not in entry:
                    entry["writeback"] = self.cycle
                    break

            rs.busy = False

    def commit(self) -> None:
        """Commit instructions in order from ROB head."""
        for entry in self.rob:
            if entry.committed:
                continue
            if not entry.ready:
                break  # must commit in order
            entry.committed = True
            self.log.append(
                f"  Cycle {self.cycle:>2}: COMMIT ROB#{entry.idx} "
                f"{entry.instr}")
            # Update timeline
            for idx, tl in self.timeline.items():
                if "writeback" in tl and "commit" not in tl:
                    tl["commit"] = self.cycle
                    break

    def run(self, instructions: list[Instruction],
            max_cycles: int = 30) -> None:
        """Run Tomasulo simulation."""
        issue_queue = deque(instructions)

        for c in range(1, max_cycles + 1):
            self.cycle = c

            # Commit (in-order)
            self.commit()

            # Write back (CDB broadcast)
            self.write_back()

            # Execute
            self.execute_step()

            # Issue (in-order from queue)
            if issue_queue:
                instr = issue_queue[0]
                if self.issue(instr):
                    issue_queue.popleft()

            # Check if all committed
            if all(e.committed for e in self.rob) and not issue_queue:
                break


# ── Demos ─────────────────────────────────────────────────────────────

def demo_dependency_analysis():
    """Detect data dependencies in an instruction sequence."""
    print("=" * 60)
    print("DEPENDENCY ANALYSIS")
    print("=" * 60)

    instructions = [
        Instruction("MUL", "R1", "R2", "R3", idx=0),
        Instruction("ADD", "R4", "R1", "R5", idx=1),   # RAW on R1
        Instruction("SUB", "R5", "R6", "R7", idx=2),   # WAR on R5 (i=1)
        Instruction("ADD", "R1", "R8", "R9", idx=3),   # WAW on R1 (i=0)
        Instruction("MUL", "R6", "R4", "R1", idx=4),   # RAW on R4, R1
    ]

    print("\n  Program:")
    for i, instr in enumerate(instructions):
        print(f"    I{i}: {instr}")

    deps = analyze_dependencies(instructions)

    for hazard_type in ("RAW", "WAR", "WAW"):
        print(f"\n  {hazard_type} dependencies:")
        if not deps[hazard_type]:
            print("    (none)")
        for src, dst, reg in deps[hazard_type]:
            print(f"    I{src} → I{dst} on {reg}")

    # Count ILP
    raw_pairs = {(s, d) for s, d, _ in deps["RAW"]}
    dependent = set()
    for s, d in raw_pairs:
        dependent.add(d)
    independent = len(instructions) - len(dependent)
    print(f"\n  Independent instructions: {independent}/{len(instructions)}")
    print(f"  → Potential ILP ≈ {independent} instructions can start in cycle 1")


def demo_register_renaming():
    """Show how register renaming eliminates false dependencies."""
    print("\n" + "=" * 60)
    print("REGISTER RENAMING (WAR/WAW ELIMINATION)")
    print("=" * 60)

    original = [
        ("I0", "ADD R1, R2, R3"),
        ("I1", "SUB R4, R1, R5"),    # RAW on R1 — true dependency
        ("I2", "MUL R1, R6, R7"),    # WAW on R1 — false dependency
        ("I3", "ADD R4, R1, R8"),    # WAW on R4, RAW on renamed R1
    ]

    renamed = [
        ("I0", "ADD P1, P2, P3"),     # R1→P1
        ("I1", "SUB P4, P1, P5"),     # R4→P4, still depends on P1 (true)
        ("I2", "MUL P6, P7, P8"),     # R1→P6 (new physical register!)
        ("I3", "ADD P9, P6, P10"),    # R4→P9 (new!), uses P6 not P1
    ]

    print("\n  Before renaming:")
    for label, asm in original:
        print(f"    {label}: {asm}")
    print("    WAW: I0→I2 (R1), I1→I3 (R4)")
    print("    WAR: I1 reads R1 before I2 writes R1")

    print("\n  After renaming (physical registers):")
    for label, asm in renamed:
        print(f"    {label}: {asm}")
    print("    WAW: eliminated (different physical registers)")
    print("    WAR: eliminated (I1 reads P1, I2 writes P6)")
    print("    RAW: I0→I1 (P1), I2→I3 (P6) — true deps remain")
    print("\n    → I0 and I2 can now execute in parallel!")


def demo_tomasulo():
    """Run Tomasulo algorithm simulation."""
    print("\n" + "=" * 60)
    print("TOMASULO ALGORITHM WITH ROB")
    print("=" * 60)

    instructions = [
        Instruction("MUL", "R1", "R2", "R3", latency=3, idx=0),
        Instruction("ADD", "R4", "R1", "R5", latency=1, idx=1),  # waits for R1
        Instruction("SUB", "R6", "R7", "R8", latency=1, idx=2),  # independent
        Instruction("ADD", "R9", "R6", "R4", latency=1, idx=3),  # waits for R6, R4
    ]

    print("\n  Instructions:")
    for i, instr in enumerate(instructions):
        print(f"    I{i}: {instr}  (latency={instr.latency})")

    sim = TomasuloSimulator(n_add_stations=3, n_mul_stations=2)
    sim.run(instructions)

    print("\n  Execution log:")
    for line in sim.log:
        print(line)

    print(f"\n  Total cycles: {sim.cycle}")
    print(f"  IPC: {len(instructions) / sim.cycle:.2f}")


def demo_ilp_limits():
    """Illustrate limits of instruction-level parallelism."""
    print("\n" + "=" * 60)
    print("ILP LIMITS")
    print("=" * 60)

    print("""
  Factors limiting ILP in superscalar processors:

  1. True dependencies (RAW)
     - Cannot be eliminated — must wait for data
     - Limits the critical path length

  2. Resource conflicts
     - Limited functional units, issue width
     - 4-wide superscalar ≠ 4x speedup

  3. Branch prediction accuracy
     - Misprediction flushes pipeline
     - ~5-10% misprediction rate in practice

  4. Memory latency
     - Cache misses stall execution
     - Out-of-order helps hide some latency

  Typical ILP in practice:
""")

    configs = [
        ("In-order scalar",    1, 1.0),
        ("In-order 2-wide",    2, 1.4),
        ("OoO 4-wide",         4, 2.5),
        ("OoO 6-wide",         6, 3.0),
        ("OoO 8-wide",         8, 3.2),
    ]

    print(f"  {'Processor':<22} {'Issue Width':>12} {'Typical IPC':>12} "
          f"{'Efficiency':>11}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*11}")
    for name, width, ipc in configs:
        eff = ipc / width
        print(f"  {name:<22} {width:>12} {ipc:>12.1f} {eff:>10.0%}")


if __name__ == "__main__":
    demo_dependency_analysis()
    demo_register_renaming()
    demo_tomasulo()
    demo_ilp_limits()
