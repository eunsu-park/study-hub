"""
5-Stage Pipeline Simulator

Demonstrates:
- Classic 5-stage RISC pipeline: IF, ID, EX, MEM, WB
- Data hazards and forwarding
- Control hazards (branch prediction)
- Pipeline stalls (bubbles)

Theory:
- Pipelining overlaps instruction execution for throughput gain.
- Ideal speedup = number of stages (5x for 5-stage).
- Hazards reduce effective throughput:
  - Data hazard: instruction depends on result not yet written back
  - Control hazard: branch target unknown until later stage
  - Structural hazard: two instructions need same resource (not modeled here)

Adapted from Computer Architecture Lesson 11.
"""

from dataclasses import dataclass


@dataclass
class PipelineInstr:
    """Instruction for the pipeline simulator."""
    name: str
    rd: str = ""
    rs: str = ""
    rt: str = ""
    is_branch: bool = False
    is_load: bool = False
    is_nop: bool = False

    def __repr__(self) -> str:
        if self.is_nop:
            return "NOP (bubble)"
        return self.name


NOP = PipelineInstr(name="NOP", is_nop=True)


class PipelineSimulator:
    """5-stage pipeline simulator with hazard detection."""

    STAGES = ["IF", "ID", "EX", "MEM", "WB"]

    def __init__(self, instructions: list[PipelineInstr], forwarding: bool = True):
        self.instructions = instructions
        self.forwarding = forwarding
        self.cycle = 0
        self.stalls = 0
        self.forwards = 0

        # Pipeline registers sit between stages and hold in-flight
        # instruction state.  NOPs represent empty slots (bubbles)
        # that flow through without side effects.
        self.pipeline: list[PipelineInstr] = [NOP] * 5
        self.fetch_idx = 0
        self.history: list[list[str]] = []  # cycle × instruction mapping

    def _detect_data_hazard(self, new_instr: PipelineInstr) -> tuple[bool, str]:
        """Check for data hazard between new_instr (in ID) and later stages."""
        sources = [s for s in [new_instr.rs, new_instr.rt] if s]

        for stage_idx, stage_name in [(2, "EX"), (3, "MEM")]:
            stage_instr = self.pipeline[stage_idx]
            if stage_instr.is_nop or not stage_instr.rd:
                continue

            if stage_instr.rd in sources:
                if self.forwarding:
                    if stage_instr.is_load and stage_idx == 2:
                        # Load-use hazard: the loaded value is not available
                        # until the end of MEM, but the dependent instruction
                        # needs it at the start of EX.  Forwarding can bridge
                        # one stage, not two — so a 1-cycle stall is mandatory.
                        return True, f"load-use ({stage_instr.rd})"
                    else:
                        # Forwarding from EX or MEM: the ALU result is
                        # available at the end of the EX stage, so we can
                        # bypass the register file and feed it directly to
                        # the next instruction's ALU input.
                        self.forwards += 1
                        return False, f"forward from {stage_name} ({stage_instr.rd})"
                else:
                    return True, f"RAW on {stage_instr.rd} (no forwarding)"

        return False, ""

    def step(self) -> bool:
        """Advance pipeline by one cycle."""
        self.cycle += 1

        # Try to fetch next instruction
        if self.fetch_idx < len(self.instructions):
            fetched = self.instructions[self.fetch_idx]
        else:
            fetched = NOP

        # Check for data hazard (instruction in ID stage)
        stall = False
        hazard_msg = ""
        if not fetched.is_nop and self.fetch_idx < len(self.instructions):
            stall, hazard_msg = self._detect_data_hazard(fetched)

        # Record pipeline state
        row = []
        for i, instr in enumerate(self.pipeline):
            row.append(str(instr) if not instr.is_nop else "---")
        self.history.append(row)

        # Advance pipeline
        # WB stage completes (shift out)
        new_pipeline = [NOP] * 5

        if stall:
            self.stalls += 1
            # Insert bubble: freeze IF/ID in place while later stages
            # drain forward.  The NOP in EX prevents the stalled
            # instruction's incorrect operands from reaching the ALU.
            # IF/ID hold their values so the stalled instruction can
            # retry on the next cycle with forwarded data available.
            new_pipeline[4] = self.pipeline[3]  # MEM → WB
            new_pipeline[3] = self.pipeline[2]  # EX → MEM
            new_pipeline[2] = NOP               # bubble in EX
            new_pipeline[1] = self.pipeline[1]  # ID stays
            new_pipeline[0] = self.pipeline[0]  # IF stays
        else:
            new_pipeline[4] = self.pipeline[3]  # MEM → WB
            new_pipeline[3] = self.pipeline[2]  # EX → MEM
            new_pipeline[2] = self.pipeline[1]  # ID → EX
            new_pipeline[1] = self.pipeline[0]  # IF → ID
            if self.fetch_idx < len(self.instructions):
                new_pipeline[0] = self.instructions[self.fetch_idx]
                self.fetch_idx += 1
            else:
                new_pipeline[0] = NOP

        self.pipeline = new_pipeline

        # Check if pipeline is empty
        return not all(p.is_nop for p in self.pipeline)

    def run(self, max_cycles: int = 50) -> None:
        """Run pipeline until empty."""
        # Initialize: fetch first instruction
        if self.instructions:
            self.pipeline[0] = self.instructions[0]
            self.fetch_idx = 1

        while self.step() and self.cycle < max_cycles:
            pass

    def display(self) -> None:
        """Display pipeline execution diagram."""
        n_instr = len(self.instructions)
        # Build per-instruction timeline
        print(f"\n  Pipeline Diagram (forwarding={'ON' if self.forwarding else 'OFF'}):")
        print(f"  {'Instruction':<30}", end="")
        for c in range(1, self.cycle + 1):
            print(f" {c:>3}", end="")
        print()
        print(f"  {'-'*30}", end="")
        print(f" {'---' * self.cycle}")

        # Track each instruction through stages
        instr_stages: dict[int, list[str]] = {i: [] for i in range(n_instr)}
        stage_map = {}  # (cycle, stage_idx) → instr_idx

        # Replay simulation
        pipeline = [None] * 5
        fetch_idx = 0
        cycle = 0

        if self.instructions:
            pipeline[0] = 0
            fetch_idx = 1

        for _ in range(self.cycle):
            cycle += 1

            # Record current state
            for s in range(5):
                if pipeline[s] is not None:
                    stage_map[(cycle, pipeline[s])] = self.STAGES[s]

            # Check stall
            stall = False
            if fetch_idx < n_instr:
                new_instr = self.instructions[fetch_idx]
                sources = [s for s in [new_instr.rs, new_instr.rt] if s]
                for si in [2, 3]:
                    if pipeline[si] is not None:
                        pi = self.instructions[pipeline[si]]
                        if pi.rd and pi.rd in sources:
                            if self.forwarding:
                                if pi.is_load and si == 2:
                                    stall = True
                            else:
                                stall = True

            new_pipeline = [None] * 5
            if stall:
                new_pipeline[4] = pipeline[3]
                new_pipeline[3] = pipeline[2]
                new_pipeline[2] = None  # bubble
                new_pipeline[1] = pipeline[1]
                new_pipeline[0] = pipeline[0]
            else:
                new_pipeline[4] = pipeline[3]
                new_pipeline[3] = pipeline[2]
                new_pipeline[2] = pipeline[1]
                new_pipeline[1] = pipeline[0]
                if fetch_idx < n_instr:
                    new_pipeline[0] = fetch_idx
                    fetch_idx += 1
                else:
                    new_pipeline[0] = None

            pipeline = new_pipeline

        # Print each instruction's timeline
        for i in range(n_instr):
            instr = self.instructions[i]
            label = str(instr)[:28]
            print(f"  {label:<30}", end="")
            for c in range(1, self.cycle + 1):
                stage = stage_map.get((c, i), "")
                if stage:
                    print(f" {stage:>3}", end="")
                else:
                    print(f"    ", end="")
            print()

        print(f"\n  Total cycles: {self.cycle}")
        print(f"  Instructions: {n_instr}")
        print(f"  CPI: {self.cycle / n_instr:.2f}")
        print(f"  Stalls: {self.stalls}")
        if self.forwarding:
            print(f"  Forwards: {self.forwards}")


# ── Demos ───────────────────────────────────────────────────────────────

def demo_no_hazard():
    """Pipeline with no hazards."""
    print("=" * 60)
    print("PIPELINE: NO HAZARDS")
    print("=" * 60)

    instructions = [
        PipelineInstr("add R1, R2, R3", rd="R1", rs="R2", rt="R3"),
        PipelineInstr("sub R4, R5, R6", rd="R4", rs="R5", rt="R6"),
        PipelineInstr("and R7, R8, R9", rd="R7", rs="R8", rt="R9"),
        PipelineInstr("or  R10, R11, R12", rd="R10", rs="R11", rt="R12"),
    ]

    sim = PipelineSimulator(instructions, forwarding=True)
    sim.run()
    sim.display()


def demo_data_hazard():
    """Pipeline with RAW data hazard."""
    print("\n" + "=" * 60)
    print("PIPELINE: DATA HAZARD (WITH FORWARDING)")
    print("=" * 60)

    instructions = [
        PipelineInstr("add R1, R2, R3", rd="R1", rs="R2", rt="R3"),
        PipelineInstr("sub R4, R1, R5", rd="R4", rs="R1", rt="R5"),  # depends on R1
        PipelineInstr("and R6, R1, R7", rd="R6", rs="R1", rt="R7"),  # depends on R1
        PipelineInstr("or  R8, R4, R9", rd="R8", rs="R4", rt="R9"),  # depends on R4
    ]

    sim = PipelineSimulator(instructions, forwarding=True)
    sim.run()
    sim.display()


def demo_load_use():
    """Pipeline with load-use hazard (must stall)."""
    print("\n" + "=" * 60)
    print("PIPELINE: LOAD-USE HAZARD (STALL)")
    print("=" * 60)

    instructions = [
        PipelineInstr("lw  R1, 0(R2)", rd="R1", rs="R2", is_load=True),
        PipelineInstr("add R3, R1, R4", rd="R3", rs="R1", rt="R4"),  # load-use!
        PipelineInstr("sub R5, R3, R6", rd="R5", rs="R3", rt="R6"),
    ]

    sim = PipelineSimulator(instructions, forwarding=True)
    sim.run()
    sim.display()


def demo_no_forwarding():
    """Same hazard without forwarding — more stalls."""
    print("\n" + "=" * 60)
    print("PIPELINE: NO FORWARDING (MORE STALLS)")
    print("=" * 60)

    instructions = [
        PipelineInstr("add R1, R2, R3", rd="R1", rs="R2", rt="R3"),
        PipelineInstr("sub R4, R1, R5", rd="R4", rs="R1", rt="R5"),
        PipelineInstr("and R6, R4, R7", rd="R6", rs="R4", rt="R7"),
    ]

    sim = PipelineSimulator(instructions, forwarding=False)
    sim.run()
    sim.display()


if __name__ == "__main__":
    demo_no_hazard()
    demo_data_hazard()
    demo_load_use()
    demo_no_forwarding()
