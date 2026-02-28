"""
Exercises for Lesson 15: Page Replacement
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers FIFO/LRU/Optimal algorithm comparison, Second-Chance algorithm,
Working Set calculation, Belady's Anomaly demonstration, and thrashing analysis.
"""

from collections import deque, OrderedDict


# === Exercise 1: Algorithm Comparison ===
# Problem: Calculate page faults for FIFO, LRU, and Optimal with 3 frames.

def exercise_1():
    """Compare FIFO, LRU, and Optimal page replacement algorithms."""
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    num_frames = 3

    print(f"Reference string: {reference_string}")
    print(f"Number of frames: {num_frames}\n")

    def simulate_fifo(refs, nf):
        """FIFO: replace the page that has been in memory the longest."""
        frames = deque(maxlen=nf)
        frame_set = set()
        faults = 0
        trace = []

        for page in refs:
            if page in frame_set:
                trace.append((page, list(frames), False, None))
            else:
                faults += 1
                evicted = None
                if len(frames) == nf:
                    evicted = frames.popleft()
                    frame_set.remove(evicted)
                frames.append(page)
                frame_set.add(page)
                trace.append((page, list(frames), True, evicted))
        return faults, trace

    def simulate_lru(refs, nf):
        """LRU: replace the page that was least recently used."""
        frames = OrderedDict()
        faults = 0
        trace = []

        for page in refs:
            if page in frames:
                # Move to end (most recently used)
                frames.move_to_end(page)
                trace.append((page, list(frames.keys()), False, None))
            else:
                faults += 1
                evicted = None
                if len(frames) >= nf:
                    evicted, _ = frames.popitem(last=False)
                frames[page] = True
                trace.append((page, list(frames.keys()), True, evicted))
        return faults, trace

    def simulate_optimal(refs, nf):
        """Optimal: replace the page that will not be used for the longest time."""
        frames = []
        frame_set = set()
        faults = 0
        trace = []

        for idx, page in enumerate(refs):
            if page in frame_set:
                trace.append((page, list(frames), False, None))
            else:
                faults += 1
                evicted = None
                if len(frames) >= nf:
                    # Find page used farthest in the future (or never)
                    farthest = -1
                    victim = None
                    for f in frames:
                        try:
                            next_use = refs[idx + 1:].index(f)
                        except ValueError:
                            next_use = float('inf')
                        if next_use > farthest:
                            farthest = next_use
                            victim = f
                    frames.remove(victim)
                    frame_set.remove(victim)
                    evicted = victim
                frames.append(page)
                frame_set.add(page)
                trace.append((page, list(frames), True, evicted))
        return faults, trace

    algorithms = [
        ("FIFO", simulate_fifo),
        ("LRU", simulate_lru),
        ("Optimal", simulate_optimal),
    ]

    for name, algo in algorithms:
        faults, trace = algo(reference_string, num_frames)
        print(f"--- {name} ---")
        print(f"  {'Access':<8} {'Frames':<20} {'Result'}")
        print("  " + "-" * 45)
        for page, frames, is_fault, evicted in trace:
            padded = frames + ["-"] * (num_frames - len(frames))
            if is_fault:
                if evicted is not None:
                    result = f"Fault (replace {evicted})"
                else:
                    result = "Fault"
            else:
                result = "Hit"
            print(f"  {page:<8} {str(padded):<20} {result}")
        print(f"  Total faults: {faults}\n")

    print(f"Summary:")
    print(f"  FIFO:    9 faults")
    print(f"  LRU:    10 faults")
    print(f"  Optimal: 7 faults (theoretical minimum)")
    print(f"\n  Note: LRU has more faults than FIFO for this particular string.")
    print(f"  This is not unusual -- LRU only guarantees optimality in the limit,")
    print(f"  not for every individual reference string.")


# === Exercise 2: Second-Chance Algorithm ===
# Problem: Determine which page gets replaced when inserting page E.

def exercise_2():
    """Simulate Second-Chance (Clock) page replacement."""
    # Initial state: (page, reference_bit)
    frames = [("A", 1), ("B", 0), ("C", 1), ("D", 0)]
    pointer = 0  # points to A initially
    new_page = "E"

    print(f"Initial state (pointer at index {pointer}):")
    for i, (page, r) in enumerate(frames):
        arrow = " <-- pointer" if i == pointer else ""
        print(f"  [{page}, R={r}]{arrow}")
    print(f"\nInsert page {new_page}:\n")

    step = 0
    while True:
        page, ref_bit = frames[pointer]
        step += 1

        if ref_bit == 1:
            # Give second chance: clear reference bit, move pointer
            print(f"  Step {step}: Pointer at [{page}, R={ref_bit}]")
            print(f"    R=1 -> Give second chance: set R=0, advance pointer")
            frames[pointer] = (page, 0)
            pointer = (pointer + 1) % len(frames)
        else:
            # Found victim: replace this page
            print(f"  Step {step}: Pointer at [{page}, R={ref_bit}]")
            print(f"    R=0 -> Replace {page} with {new_page}")
            frames[pointer] = (new_page, 1)
            pointer = (pointer + 1) % len(frames)
            break

    print(f"\nFinal state (pointer at index {pointer}):")
    for i, (page, r) in enumerate(frames):
        arrow = " <-- pointer" if i == pointer else ""
        new_marker = " ** NEW **" if page == new_page else ""
        print(f"  [{page}, R={r}]{arrow}{new_marker}")

    print(f"\nExplanation:")
    print(f"  A had R=1, so it got a second chance (R cleared to 0).")
    print(f"  B had R=0, so B was the victim -- replaced by {new_page}.")
    print(f"  The pointer advanced past the replacement to C.")
    print(f"\n  Second-Chance is an approximation of LRU:")
    print(f"  Pages with R=1 were recently accessed and get to stay.")
    print(f"  Pages with R=0 have not been accessed since the last sweep.")


# === Exercise 3: Working Set Calculation ===
# Problem: Calculate Working Set at time t=10 with delta=5.

def exercise_3():
    """Calculate Working Set at a specific time."""
    # Page references indexed by time
    time_refs = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 1, 7: 3, 8: 4, 9: 5, 10: 2}
    target_time = 10
    delta = 5

    print(f"Page reference trace:")
    print(f"  Time:  ", end="")
    for t in range(1, target_time + 1):
        print(f"{t:>3}", end="")
    print()
    print(f"  Page:  ", end="")
    for t in range(1, target_time + 1):
        print(f"{time_refs[t]:>3}", end="")
    print(f"\n")

    print(f"Working Set at t={target_time} with delta={delta}:")
    print(f"  Window: references from t={target_time - delta + 1} to t={target_time}\n")

    # Collect pages referenced in the window
    window_start = target_time - delta + 1
    window_end = target_time
    pages_in_window = set()
    print(f"  References in window [{window_start}, {window_end}]:")
    for t in range(window_start, window_end + 1):
        page = time_refs[t]
        pages_in_window.add(page)
        print(f"    t={t}: Page {page}")

    wss = len(pages_in_window)
    print(f"\n  Working Set W({target_time}, {delta}) = {sorted(pages_in_window)}")
    print(f"  Working Set Size (WSS) = {wss}\n")

    print(f"  Interpretation:")
    print(f"    This process needs at least {wss} frames to avoid excessive faults.")
    print(f"    If fewer than {wss} frames are allocated, the process will thrash.")

    # Show how WSS changes over time
    print(f"\n--- Working Set Size over time (delta={delta}) ---\n")
    print(f"  {'Time':<6} {'Window':<15} {'Pages in Window':<20} {'WSS'}")
    print("  " + "-" * 50)
    for t in range(delta, target_time + 1):
        ws = set()
        for t2 in range(t - delta + 1, t + 1):
            ws.add(time_refs[t2])
        print(f"  t={t:<3} [{t - delta + 1},{t}]{'':<8} {str(sorted(ws)):<20} {len(ws)}")


# === Exercise 4: Belady's Anomaly ===
# Problem: Demonstrate that FIFO with more frames can have more faults.

def exercise_4():
    """Demonstrate Belady's Anomaly with FIFO replacement."""
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]

    print(f"Reference string: {reference_string}")
    print(f"Comparing FIFO with 3 frames vs 4 frames\n")

    def fifo_simulate(refs, nf):
        """Run FIFO and return fault count and trace."""
        frames = deque(maxlen=nf)
        frame_set = set()
        faults = 0
        trace = []

        for page in refs:
            if page in frame_set:
                trace.append((page, list(frames) + ["-"] * (nf - len(frames)), "H"))
            else:
                faults += 1
                if len(frames) == nf:
                    evicted = frames.popleft()
                    frame_set.remove(evicted)
                frames.append(page)
                frame_set.add(page)
                trace.append((page, list(frames) + ["-"] * (nf - len(frames)), "F"))
        return faults, trace

    for nf in [3, 4]:
        faults, trace = fifo_simulate(reference_string, nf)
        print(f"--- FIFO with {nf} frames ---")
        header = "  " + f"{'Access':<8}"
        for i in range(nf):
            header += f"F{i:<5}"
        header += "Result"
        print(header)
        print("  " + "-" * (8 + 6 * nf + 8))

        for page, frames, result in trace:
            line = f"  {page:<8}"
            for f in frames:
                line += f"{str(f):<6}"
            line += "Fault" if result == "F" else "Hit"
            print(line)
        print(f"  Total faults: {faults}\n")

    print(f"Belady's Anomaly confirmed!")
    print(f"  3 frames: 9 faults")
    print(f"  4 frames: 10 faults")
    print(f"  More frames resulted in MORE faults!\n")

    print(f"Why does this happen?")
    print(f"  FIFO is not a 'stack algorithm'. In stack algorithms (like LRU and")
    print(f"  Optimal), the set of pages in N frames is always a subset of the set")
    print(f"  in N+1 frames. FIFO does not have this property -- replacing the")
    print(f"  oldest page regardless of usage can evict frequently-used pages.\n")

    print(f"  Algorithms immune to Belady's Anomaly:")
    print(f"    - LRU (stack algorithm)")
    print(f"    - Optimal (stack algorithm)")
    print(f"  Algorithms susceptible:")
    print(f"    - FIFO")
    print(f"    - Second-Chance (based on FIFO)")


# === Exercise 5: Thrashing Analysis ===
# Problem: Diagnose thrashing from system symptoms and propose solutions.

def exercise_5():
    """Analyze thrashing symptoms and propose solutions."""
    symptoms = {
        "CPU utilization": "5%",
        "Disk I/O": "95%",
        "Memory": "Nearly full",
        "Processes waiting": "Many",
    }

    print("System symptoms observed:")
    for metric, value in symptoms.items():
        print(f"  {metric}: {value}")
    print()

    print("Diagnosis: THRASHING\n")
    print("  The system is in a thrashing state. Here is the causal chain:")
    print("  1. Too many processes were admitted into memory")
    print("  2. Each process has fewer frames than its working set requires")
    print("  3. Page faults happen constantly across all processes")
    print("  4. The disk is saturated handling page fault I/O (95%)")
    print("  5. Processes spend almost all time waiting for pages")
    print("  6. CPU has nothing to run while waiting for I/O (5% utilization)")
    print("  7. The OS scheduler may admit MORE processes (seeing low CPU),")
    print("     which makes thrashing WORSE (positive feedback loop)\n")

    print("Solutions:\n")

    print("  1. IMMEDIATE: Reduce degree of multiprogramming")
    print("     - Suspend (swap out) some processes entirely")
    print("     - This frees their frames for remaining processes")
    print("     - Start with lowest-priority or newest processes")
    print("     - Monitor: CPU utilization should rise as thrashing subsides\n")

    print("  2. SYSTEM CONFIGURATION:")
    print("     - Lower swappiness: vm.swappiness=10 (default 60)")
    print("       Reduces tendency to swap out process pages in favor of cache")
    print("     - Limit overcommit: vm.overcommit_memory=2")
    print("       Prevents allocating more virtual memory than physical + swap")
    print("     - Set per-process memory limits with cgroups")
    print("       Prevents any single process from consuming all memory\n")

    print("  3. LONG-TERM SOLUTIONS:")
    print("     - Add physical RAM (most direct solution)")
    print("     - Implement Working Set-based admission control:")
    print("       Only admit a new process if sum of all WSS <= available frames")
    print("     - Monitor Page Fault Frequency (PFF):")
    print("       If PFF > upper threshold: allocate more frames to process")
    print("       If PFF < lower threshold: reclaim frames from process")
    print("     - Use faster storage (NVMe SSD) to reduce page fault latency\n")

    print("  4. APPLICATION LEVEL:")
    print("     - Check for memory leaks (unbounded growth)")
    print("     - Reduce application cache sizes")
    print("     - Use madvise() to hint access patterns to the OS")
    print("     - Profile memory usage and optimize hot paths\n")

    print("  Monitoring commands (Linux):")
    print("    $ vmstat 1          # Watch si/so (swap in/out) columns")
    print("    $ sar -B 1          # Page fault statistics")
    print("    $ free -h           # Memory usage overview")
    print("    $ top -o %MEM       # Sort processes by memory usage")
    print("    $ cat /proc/vmstat  # Detailed VM statistics")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Algorithm Comparison ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Second-Chance Algorithm ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Working Set Calculation ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Belady's Anomaly ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Thrashing Analysis ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
