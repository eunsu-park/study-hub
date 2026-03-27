"""
Process Concepts Demonstration

Simulates operating system process management:
- Process Control Block (PCB) structure
- Process state transitions (New -> Ready -> Running -> Waiting -> Terminated)
- Process creation and context switching
- Process table management

Theory:
- A process is a program in execution with its own address space
- The PCB stores all information needed to manage a process
- State transitions are triggered by system calls, interrupts, and scheduling
- Context switching saves/restores register state between processes

Adapted from OS Theory Lesson 02.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ProcessState(Enum):
    NEW = auto()
    READY = auto()
    RUNNING = auto()
    WAITING = auto()
    TERMINATED = auto()


@dataclass
class PCB:
    """Process Control Block — stores all process metadata."""
    pid: int
    name: str
    state: ProcessState = ProcessState.NEW
    program_counter: int = 0
    registers: dict = field(default_factory=dict)
    priority: int = 0
    memory_base: int = 0
    memory_limit: int = 0
    parent_pid: Optional[int] = None
    cpu_time_used: int = 0
    io_wait_time: int = 0

    def __str__(self) -> str:
        return (
            f"PID={self.pid:3d} | {self.name:12s} | "
            f"{self.state.name:11s} | PC={self.program_counter:4d} | "
            f"CPU={self.cpu_time_used:3d} | Priority={self.priority}"
        )


class ProcessTable:
    """Maintains a table of all processes in the system."""

    def __init__(self):
        self._processes: dict[int, PCB] = {}
        # PIDs start at 1, not 0 — PID 0 is conventionally reserved for
        # the kernel's idle/swapper process in real OSes
        self._next_pid = 1

    def create_process(
        self, name: str, priority: int = 0, parent_pid: Optional[int] = None
    ) -> PCB:
        """Create a new process and add it to the table."""
        pcb = PCB(
            pid=self._next_pid,
            name=name,
            priority=priority,
            parent_pid=parent_pid,
            # Each process gets a non-overlapping memory region; multiplying by PID
            # simulates base-and-limit register protection in a contiguous allocation scheme
            memory_base=self._next_pid * 1000,
            memory_limit=1000,
        )
        self._processes[pcb.pid] = pcb
        self._next_pid += 1
        print(f"  [CREATE] {pcb}")
        return pcb

    def transition(self, pid: int, new_state: ProcessState) -> None:
        """Transition a process to a new state with validation."""
        pcb = self._processes[pid]
        old_state = pcb.state

        # Validate transitions — enforcing the standard 5-state model prevents
        # illegal jumps (e.g., NEW->RUNNING) that would bypass resource allocation
        # the OS performs during the NEW->READY transition
        valid_transitions = {
            ProcessState.NEW: {ProcessState.READY},
            ProcessState.READY: {ProcessState.RUNNING},
            ProcessState.RUNNING: {ProcessState.READY, ProcessState.WAITING, ProcessState.TERMINATED},
            ProcessState.WAITING: {ProcessState.READY},
        }
        allowed = valid_transitions.get(old_state, set())
        if new_state not in allowed:
            print(f"  [ERROR] Invalid transition: {old_state.name} -> {new_state.name}")
            return

        pcb.state = new_state
        print(f"  [{old_state.name:>10s} -> {new_state.name:<11s}] PID={pid} ({pcb.name})")

    def context_switch(self, old_pid: int, new_pid: int) -> None:
        """Simulate a context switch between two processes."""
        old_pcb = self._processes[old_pid]
        new_pcb = self._processes[new_pid]

        # Save old process state — the program counter must be persisted so this
        # process resumes at the correct instruction when rescheduled; a real OS
        # also saves all general-purpose registers, stack pointer, and flags
        old_pcb.registers["saved_pc"] = old_pcb.program_counter

        # Restore new process state — only restore if previously saved; a brand-new
        # process that has never run won't have a saved_pc yet
        if "saved_pc" in new_pcb.registers:
            new_pcb.program_counter = new_pcb.registers["saved_pc"]

        self.transition(old_pid, ProcessState.READY)
        self.transition(new_pid, ProcessState.RUNNING)
        print(f"  [SWITCH] PID {old_pid} -> PID {new_pid}")

    def display(self) -> None:
        """Print the full process table."""
        print("\n" + "=" * 70)
        print("PROCESS TABLE")
        print("-" * 70)
        for pcb in self._processes.values():
            print(f"  {pcb}")
        print("=" * 70)


def demo_process_lifecycle():
    """Demonstrate process creation, state transitions, and context switching."""
    print("=" * 60)
    print("PROCESS LIFECYCLE DEMONSTRATION")
    print("=" * 60)

    table = ProcessTable()

    # Phase 1: Create processes
    print("\n--- Phase 1: Process Creation ---")
    init = table.create_process("init", priority=0)
    shell = table.create_process("bash", priority=5, parent_pid=init.pid)
    editor = table.create_process("vim", priority=3, parent_pid=shell.pid)
    compiler = table.create_process("gcc", priority=7, parent_pid=shell.pid)

    # Phase 2: Admit to ready queue
    print("\n--- Phase 2: Admit to Ready Queue ---")
    for pid in [init.pid, shell.pid, editor.pid, compiler.pid]:
        table.transition(pid, ProcessState.READY)

    # Phase 3: Schedule and run
    print("\n--- Phase 3: Scheduling ---")
    table.transition(init.pid, ProcessState.RUNNING)
    init.program_counter = 100
    init.cpu_time_used = 10

    # Phase 4: Context switch
    print("\n--- Phase 4: Context Switch (init -> shell) ---")
    table.context_switch(init.pid, shell.pid)
    shell.program_counter = 200
    shell.cpu_time_used = 5

    # Phase 5: I/O wait
    print("\n--- Phase 5: I/O Wait (shell waits for disk) ---")
    table.transition(shell.pid, ProcessState.WAITING)
    table.transition(editor.pid, ProcessState.RUNNING)

    # Phase 6: I/O complete, shell returns to ready
    print("\n--- Phase 6: I/O Complete ---")
    table.transition(shell.pid, ProcessState.READY)

    # Phase 7: Process termination
    print("\n--- Phase 7: Termination ---")
    table.transition(editor.pid, ProcessState.TERMINATED)

    table.display()


def demo_fork_tree():
    """Demonstrate process tree creation (fork-like behavior)."""
    print("\n" + "=" * 60)
    print("PROCESS TREE DEMONSTRATION")
    print("=" * 60)

    table = ProcessTable()
    init = table.create_process("init")
    table.transition(init.pid, ProcessState.READY)
    table.transition(init.pid, ProcessState.RUNNING)

    # init forks login
    login = table.create_process("login", parent_pid=init.pid)
    table.transition(login.pid, ProcessState.READY)

    # login forks shell
    shell = table.create_process("bash", parent_pid=login.pid)
    table.transition(shell.pid, ProcessState.READY)

    # shell forks two children
    ls = table.create_process("ls", parent_pid=shell.pid)
    grep = table.create_process("grep", parent_pid=shell.pid)
    table.transition(ls.pid, ProcessState.READY)
    table.transition(grep.pid, ProcessState.READY)

    # Display tree
    print("\nProcess Tree:")
    for pcb in table._processes.values():
        parent = f"(parent={pcb.parent_pid})" if pcb.parent_pid else "(root)"
        indent = "  " * (pcb.pid - 1)
        print(f"  {indent}PID {pcb.pid}: {pcb.name} {parent}")

    table.display()


if __name__ == "__main__":
    demo_process_lifecycle()
    demo_fork_tree()
