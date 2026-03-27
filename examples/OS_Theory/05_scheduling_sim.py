"""
CPU Scheduling Algorithm Simulator

Implements and compares classic scheduling algorithms:
- FCFS (First Come First Served)
- SJF (Shortest Job First) — non-preemptive
- SRTF (Shortest Remaining Time First) — preemptive SJF
- Round Robin (with configurable quantum)

Outputs Gantt chart, average waiting time, average turnaround time.

Theory:
- Scheduling decides which process gets the CPU next
- Key metrics: waiting time, turnaround time, response time, throughput
- Non-preemptive: process runs until completion or I/O
- Preemptive: process can be interrupted by higher-priority arrival
- RR provides fair time-sharing but increases context switches

Adapted from OS Theory Lesson 05.
"""

from dataclasses import dataclass


@dataclass
class Process:
    name: str
    arrival: int
    burst: int
    remaining: int = 0
    start_time: int = -1
    finish_time: int = -1
    wait_time: int = 0

    def __post_init__(self):
        self.remaining = self.burst

    @property
    def turnaround(self) -> int:
        return self.finish_time - self.arrival

    @property
    def response(self) -> int:
        return self.start_time - self.arrival


def fcfs(processes: list[Process]) -> list[tuple[str, int, int]]:
    """First Come First Served scheduling."""
    procs = sorted(processes, key=lambda p: (p.arrival, p.name))
    gantt: list[tuple[str, int, int]] = []
    time = 0

    for p in procs:
        if time < p.arrival:
            gantt.append(("idle", time, p.arrival))
            time = p.arrival
        p.start_time = time
        p.wait_time = time - p.arrival
        gantt.append((p.name, time, time + p.burst))
        time += p.burst
        p.finish_time = time
        p.remaining = 0

    return gantt


def sjf(processes: list[Process]) -> list[tuple[str, int, int]]:
    """Shortest Job First (non-preemptive)."""
    procs = [p for p in processes]
    ready: list[Process] = []
    gantt: list[tuple[str, int, int]] = []
    time = 0
    done = 0

    while done < len(procs):
        # Add newly arrived processes
        for p in procs:
            if p.arrival <= time and p.remaining > 0 and p not in ready:
                ready.append(p)

        if not ready:
            # Jump to next arrival
            next_arrival = min(p.arrival for p in procs if p.remaining > 0)
            gantt.append(("idle", time, next_arrival))
            time = next_arrival
            continue

        # Pick shortest burst — tie-break by arrival time to ensure FCFS ordering
        # among equal-burst processes, which provides deterministic behavior
        ready.sort(key=lambda p: (p.burst, p.arrival))
        p = ready.pop(0)
        p.start_time = time
        p.wait_time = time - p.arrival
        gantt.append((p.name, time, time + p.burst))
        time += p.burst
        p.finish_time = time
        p.remaining = 0
        done += 1

    return gantt


def srtf(processes: list[Process]) -> list[tuple[str, int, int]]:
    """Shortest Remaining Time First (preemptive SJF)."""
    procs = [p for p in processes]
    gantt: list[tuple[str, int, int]] = []
    time = 0
    done = 0
    current_name = ""
    segment_start = 0

    # Upper bound prevents infinite loop — worst case is all processes arrive
    # at the latest time and run sequentially; +1 accounts for time-step granularity
    max_time = max(p.arrival for p in procs) + sum(p.burst for p in procs) + 1

    while done < len(procs) and time < max_time:
        # Find ready processes
        ready = [p for p in procs if p.arrival <= time and p.remaining > 0]

        if not ready:
            if current_name:
                gantt.append((current_name, segment_start, time))
                current_name = ""
            next_arrival = min(
                (p.arrival for p in procs if p.remaining > 0), default=time + 1
            )
            gantt.append(("idle", time, next_arrival))
            time = next_arrival
            segment_start = time
            continue

        # Pick shortest remaining — SRTF re-evaluates at every time unit, so a
        # newly arrived short job can preempt the current one mid-execution
        ready.sort(key=lambda p: (p.remaining, p.arrival))
        chosen = ready[0]

        if chosen.name != current_name:
            if current_name:
                gantt.append((current_name, segment_start, time))
            current_name = chosen.name
            segment_start = time

        if chosen.start_time == -1:
            chosen.start_time = time

        chosen.remaining -= 1
        time += 1

        if chosen.remaining == 0:
            chosen.finish_time = time
            done += 1
            gantt.append((current_name, segment_start, time))
            current_name = ""
            segment_start = time

    # Calculate wait times
    for p in procs:
        p.wait_time = p.turnaround - p.burst

    return gantt


def round_robin(processes: list[Process], quantum: int = 2) -> list[tuple[str, int, int]]:
    """Round Robin scheduling with configurable time quantum."""
    from collections import deque

    procs = sorted(processes, key=lambda p: p.arrival)
    queue: deque[Process] = deque()
    gantt: list[tuple[str, int, int]] = []
    time = 0
    idx = 0  # next process to admit

    while idx < len(procs) or queue:
        # Admit new arrivals
        while idx < len(procs) and procs[idx].arrival <= time:
            queue.append(procs[idx])
            idx += 1

        if not queue:
            next_arrival = procs[idx].arrival
            gantt.append(("idle", time, next_arrival))
            time = next_arrival
            continue

        p = queue.popleft()
        if p.start_time == -1:
            p.start_time = time

        run_time = min(quantum, p.remaining)
        gantt.append((p.name, time, time + run_time))
        p.remaining -= run_time
        time += run_time

        # Admit processes that arrived during this quantum BEFORE re-enqueuing
        # the current process — this ensures newly arrived jobs go ahead of the
        # preempted process, matching real RR scheduler behavior
        while idx < len(procs) and procs[idx].arrival <= time:
            queue.append(procs[idx])
            idx += 1

        if p.remaining > 0:
            queue.append(p)
        else:
            p.finish_time = time

    # Calculate wait times
    for p in procs:
        p.wait_time = p.turnaround - p.burst

    return gantt


# ── Output formatting ────────────────────────────────────────────────────

def print_gantt(gantt: list[tuple[str, int, int]], title: str) -> None:
    """Print a Gantt chart."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")

    # Merge consecutive identical segments
    merged: list[tuple[str, int, int]] = []
    for name, start, end in gantt:
        if merged and merged[-1][0] == name and merged[-1][2] == start:
            merged[-1] = (name, merged[-1][1], end)
        else:
            merged.append((name, start, end))

    # Top border
    line1 = "  |"
    line2 = "  "
    for name, start, end in merged:
        width = max(3, (end - start) * 3)
        line1 += f" {name:^{width-1}}|"
        line2 += f"{start:<{width+1}}"
    line2 += str(merged[-1][2])

    print(line1)
    print(line2)


def print_metrics(processes: list[Process]) -> None:
    """Print scheduling metrics for all processes."""
    print(f"\n  {'Process':>8} {'Arrival':>8} {'Burst':>6} {'Start':>6} "
          f"{'Finish':>7} {'Wait':>5} {'Turn':>5} {'Resp':>5}")
    print(f"  {'─' * 56}")
    for p in sorted(processes, key=lambda x: x.name):
        print(f"  {p.name:>8} {p.arrival:>8} {p.burst:>6} {p.start_time:>6} "
              f"{p.finish_time:>7} {p.wait_time:>5} {p.turnaround:>5} {p.response:>5}")

    avg_wait = sum(p.wait_time for p in processes) / len(processes)
    avg_turn = sum(p.turnaround for p in processes) / len(processes)
    avg_resp = sum(p.response for p in processes) / len(processes)
    print(f"\n  Avg Wait: {avg_wait:.2f}  |  Avg Turnaround: {avg_turn:.2f}  |  Avg Response: {avg_resp:.2f}")


def make_processes() -> list[Process]:
    """Create a standard test set of processes."""
    return [
        Process("P1", arrival=0, burst=6),
        Process("P2", arrival=1, burst=4),
        Process("P3", arrival=2, burst=2),
        Process("P4", arrival=3, burst=3),
    ]


def run_algorithm(name: str, algo, **kwargs) -> None:
    """Run a scheduling algorithm and display results."""
    procs = make_processes()
    gantt = algo(procs, **kwargs)
    print_gantt(gantt, name)
    print_metrics(procs)


if __name__ == "__main__":
    print("=" * 60)
    print("CPU SCHEDULING ALGORITHM COMPARISON")
    print("=" * 60)
    print("\nProcesses: P1(0,6) P2(1,4) P3(2,2) P4(3,3)")

    run_algorithm("FCFS", fcfs)
    run_algorithm("SJF (Non-preemptive)", sjf)
    run_algorithm("SRTF (Preemptive SJF)", srtf)
    run_algorithm("Round Robin (quantum=2)", round_robin, quantum=2)
