"""
Multi-Level Feedback Queue (MLFQ) Simulator

Implements MLFQ scheduling with:
- Multiple priority queues (highest = 0)
- Round-robin within each queue (increasing quantum)
- Priority demotion on quantum expiry
- Priority boost to prevent starvation

Theory:
- MLFQ combines benefits of SJF (short jobs first) and RR (fairness)
- New processes start at highest priority
- CPU-bound processes get demoted to lower priority queues
- I/O-bound processes stay at high priority (they yield before quantum)
- Periodic boost prevents indefinite starvation of long-running jobs

Adapted from OS Theory Lesson 06.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Job:
    name: str
    arrival: int
    bursts: list[int]  # list of CPU burst durations
    remaining: int = 0
    current_burst_idx: int = 0
    priority: int = 0
    finish_time: int = -1
    total_wait: int = 0
    last_ready_time: int = 0

    def __post_init__(self):
        self.remaining = self.bursts[0] if self.bursts else 0

    @property
    def done(self) -> bool:
        return self.current_burst_idx >= len(self.bursts) and self.remaining == 0


class MLFQ:
    """Multi-Level Feedback Queue scheduler."""

    def __init__(self, num_levels: int = 3, base_quantum: int = 2, boost_interval: int = 20):
        self.num_levels = num_levels
        # Exponentially increasing quanta: lower-priority queues get longer
        # time slices because their jobs are likely CPU-bound, reducing the
        # context-switch overhead for processes that won't yield voluntarily
        self.quanta = [base_quantum * (2 ** i) for i in range(num_levels)]
        self.boost_interval = boost_interval
        self.queues: list[deque[Job]] = [deque() for _ in range(num_levels)]
        self.gantt: list[tuple[str, int, int]] = []

    def add_job(self, job: Job, time: int) -> None:
        """Add a job at the highest priority."""
        job.priority = 0
        job.last_ready_time = time
        self.queues[0].append(job)

    def boost(self, time: int) -> None:
        """Move all jobs to highest priority queue."""
        for level in range(1, self.num_levels):
            while self.queues[level]:
                job = self.queues[level].popleft()
                job.priority = 0
                job.last_ready_time = time
                self.queues[0].append(job)

    def pick_next(self) -> Job | None:
        """Pick the next job from the highest non-empty queue."""
        for level in range(self.num_levels):
            if self.queues[level]:
                return self.queues[level].popleft()
        return None

    def run(self, jobs: list[Job]) -> list[tuple[str, int, int]]:
        """Run the MLFQ simulation."""
        time = 0
        pending = sorted(jobs, key=lambda j: j.arrival)
        admitted = 0
        completed = 0
        total = len(jobs)
        last_boost = 0

        while completed < total:
            # Admit new arrivals
            while admitted < total and pending[admitted].arrival <= time:
                self.add_job(pending[admitted], time)
                admitted += 1

            # Priority boost check — without periodic boosting, CPU-bound jobs
            # demoted to the lowest queue could starve indefinitely when a steady
            # stream of I/O-bound jobs keeps the top queue busy
            if time - last_boost >= self.boost_interval:
                self.boost(time)
                last_boost = time

            job = self.pick_next()
            if job is None:
                # No ready jobs — advance to next arrival
                if admitted < total:
                    next_arr = pending[admitted].arrival
                    self.gantt.append(("idle", time, next_arr))
                    time = next_arr
                else:
                    break
                continue

            # Account waiting time
            job.total_wait += time - job.last_ready_time

            # Run for up to the quantum of this level
            quantum = self.quanta[job.priority]
            run_time = min(quantum, job.remaining)
            self.gantt.append((job.name, time, time + run_time))
            job.remaining -= run_time
            time += run_time

            # Admit jobs that arrived during execution
            while admitted < total and pending[admitted].arrival <= time:
                self.add_job(pending[admitted], time)
                admitted += 1

            if job.remaining == 0:
                # Current burst complete
                job.current_burst_idx += 1
                if job.done:
                    job.finish_time = time
                    completed += 1
                else:
                    # Start next burst — reset to highest priority because a job
                    # returning from I/O exhibits interactive behavior; penalizing
                    # it for past CPU use would hurt I/O-bound responsiveness
                    job.remaining = job.bursts[job.current_burst_idx]
                    job.priority = 0
                    job.last_ready_time = time
                    self.queues[0].append(job)
            else:
                # Quantum expired — demote to penalize CPU-bound behavior.
                # Clamping to num_levels-1 ensures the job stays in the lowest
                # queue rather than going out of bounds.
                new_level = min(job.priority + 1, self.num_levels - 1)
                job.priority = new_level
                job.last_ready_time = time
                self.queues[new_level].append(job)

        return self.gantt


def print_gantt(gantt: list[tuple[str, int, int]]) -> None:
    """Print Gantt chart."""
    # Merge consecutive identical segments
    merged: list[tuple[str, int, int]] = []
    for name, start, end in gantt:
        if merged and merged[-1][0] == name and merged[-1][2] == start:
            merged[-1] = (name, merged[-1][1], end)
        else:
            merged.append((name, start, end))

    line = "  |"
    times = "  "
    for name, start, end in merged:
        w = max(3, (end - start) * 2)
        line += f" {name:^{w-1}}|"
        times += f"{start:<{w+1}}"
    times += str(merged[-1][2])
    print(line)
    print(times)


if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-LEVEL FEEDBACK QUEUE (MLFQ) SIMULATION")
    print("=" * 60)

    # Mix of CPU-bound and I/O-bound jobs
    jobs = [
        Job("CPU1", arrival=0, bursts=[12]),       # Long CPU-bound
        Job("IO1",  arrival=0, bursts=[2, 2, 2]),  # Short I/O-bound bursts
        Job("CPU2", arrival=2, bursts=[8]),         # Medium CPU-bound
        Job("IO2",  arrival=3, bursts=[1, 1]),      # Very short I/O
    ]

    scheduler = MLFQ(num_levels=3, base_quantum=2, boost_interval=15)

    print(f"\nQueue levels: {scheduler.num_levels}")
    print(f"Quanta: {scheduler.quanta}")
    print(f"Boost interval: {scheduler.boost_interval}")
    print(f"\nJobs:")
    for j in jobs:
        print(f"  {j.name}: arrival={j.arrival}, bursts={j.bursts}")

    gantt = scheduler.run(jobs)

    print(f"\nGantt Chart:")
    print_gantt(gantt)

    print(f"\nResults:")
    print(f"  {'Job':>5} {'Finish':>7} {'Wait':>5}")
    print(f"  {'─' * 20}")
    for j in sorted(jobs, key=lambda x: x.name):
        print(f"  {j.name:>5} {j.finish_time:>7} {j.total_wait:>5}")

    print(f"\n  Avg Wait: {sum(j.total_wait for j in jobs) / len(jobs):.1f}")
    print(f"\n  Observation: I/O-bound jobs (IO1, IO2) maintain high priority")
    print(f"  while CPU-bound jobs (CPU1, CPU2) get demoted to lower queues.")
