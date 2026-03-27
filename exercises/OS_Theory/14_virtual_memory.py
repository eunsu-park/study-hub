"""
Exercises for Lesson 14: Virtual Memory
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers page fault counting (FIFO), EAT calculation with page faults,
copy-on-write analysis, mmap() behavior, and demand paging design.
"""

from collections import deque


# === Exercise 1: Page Fault Scenario ===
# Problem: Count page faults for a reference string with 3 frames using FIFO.

def exercise_1():
    """Count page faults using FIFO replacement with 3 frames."""
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    num_frames = 3

    print(f"Reference string: {reference_string}")
    print(f"Number of frames: {num_frames}")
    print(f"Replacement algorithm: FIFO\n")

    # FIFO simulation using a deque
    frames = deque(maxlen=num_frames)
    frame_set = set()
    faults = 0

    print(f"{'Access':<8} {'Frames':<20} {'Result'}")
    print("-" * 48)

    for page in reference_string:
        if page in frame_set:
            # Hit
            frame_list = list(frames)
            # Pad to show all frame slots
            display = frame_list + ["-"] * (num_frames - len(frame_list))
            print(f"  {page:<6} {str(display):<20} Hit")
        else:
            # Fault
            faults += 1
            if len(frames) == num_frames:
                evicted = frames.popleft()
                frame_set.remove(evicted)
                replaced_msg = f"Fault (replace {evicted})"
            else:
                replaced_msg = "Fault"

            frames.append(page)
            frame_set.add(page)

            frame_list = list(frames)
            display = frame_list + ["-"] * (num_frames - len(frame_list))
            print(f"  {page:<6} {str(display):<20} {replaced_msg}")

    print(f"\nTotal page faults: {faults}")
    print(f"Page fault rate: {faults}/{len(reference_string)} = {faults / len(reference_string):.1%}")


# === Exercise 2: EAT Calculation with Page Faults ===
# Problem: Calculate required page fault probability for < 5% performance degradation.

def exercise_2():
    """Calculate maximum page fault probability for acceptable performance."""
    mem_access_time = 50          # ns
    page_fault_time = 10_000_000  # 10ms in ns
    max_degradation = 0.05        # 5%

    acceptable_eat = mem_access_time * (1 + max_degradation)

    print(f"Given:")
    print(f"  Memory access time (ma): {mem_access_time}ns")
    print(f"  Page fault service time (pft): {page_fault_time / 1_000_000:.0f}ms = {page_fault_time:,}ns")
    print(f"  Maximum performance degradation: {max_degradation * 100:.0f}%")
    print(f"  Acceptable EAT: {mem_access_time} * {1 + max_degradation} = {acceptable_eat}ns\n")

    # EAT = (1-p) * ma + p * pft
    # acceptable_eat = ma - p*ma + p*pft
    # acceptable_eat - ma = p * (pft - ma)
    # p = (acceptable_eat - ma) / (pft - ma)

    p = (acceptable_eat - mem_access_time) / (page_fault_time - mem_access_time)

    print(f"Derivation:")
    print(f"  EAT = (1-p) * ma + p * pft")
    print(f"  {acceptable_eat} = (1-p) * {mem_access_time} + p * {page_fault_time}")
    print(f"  {acceptable_eat} = {mem_access_time} - {mem_access_time}p + {page_fault_time}p")
    print(f"  {acceptable_eat - mem_access_time} = {page_fault_time - mem_access_time}p")
    print(f"  p = {acceptable_eat - mem_access_time} / {page_fault_time - mem_access_time}")
    print(f"  p = {p:.2e}")
    print(f"  p ~ 1 / {int(1/p):,}\n")

    print(f"Conclusion: At most 1 page fault per {int(1/p):,} memory accesses")
    print(f"to keep performance within {max_degradation * 100:.0f}% of ideal.\n")

    # Verify
    eat_check = (1 - p) * mem_access_time + p * page_fault_time
    print(f"Verification: EAT = (1-{p:.2e}) * {mem_access_time} + {p:.2e} * {page_fault_time}")
    print(f"            = {eat_check:.2f}ns (target: {acceptable_eat}ns)")


# === Exercise 3: Copy-on-Write Analysis ===
# Problem: Track physical page count through fork(), modification, and termination.

def exercise_3():
    """Analyze Copy-on-Write behavior through process lifecycle."""
    parent_pages = 100
    child_modified = 10

    print(f"Parent process has {parent_pages} pages.\n")

    # Phase 1: After fork()
    print(f"Phase 1: After fork()")
    print(f"  - All {parent_pages} pages marked read-only and shared")
    print(f"  - Both parent and child point to the same physical frames")
    print(f"  - Reference count for each page: 2 (parent + child)")
    phase1_physical = parent_pages
    print(f"  Physical pages in use: {phase1_physical}\n")

    # Phase 2: Child modifies 10 pages
    print(f"Phase 2: Child modifies {child_modified} pages")
    print(f"  - Each write triggers a COW fault:")
    print(f"    1. OS allocates a new frame")
    print(f"    2. Copies the original page content to the new frame")
    print(f"    3. Updates child's page table to point to the new frame")
    print(f"    4. Makes the new frame writable for the child")
    print(f"    5. Decrements reference count on the original page")
    shared_pages = parent_pages - child_modified
    phase2_physical = parent_pages + child_modified
    print(f"  Pages still shared: {shared_pages} (ref count = 2)")
    print(f"  Parent-only pages: 0 (parent didn't write)")
    print(f"  Child-only pages: {child_modified} (newly allocated)")
    print(f"  Physical pages in use: {parent_pages} + {child_modified} = {phase2_physical}\n")

    # Phase 3: Child terminates
    print(f"Phase 3: Child terminates")
    print(f"  - Child's {child_modified} private pages: freed immediately")
    print(f"  - {shared_pages} shared pages: reference count decremented (2 -> 1)")
    print(f"  - {child_modified} original pages child stopped sharing: already ref=1")
    phase3_physical = parent_pages
    print(f"  Physical pages in use: {phase3_physical} (only parent remains)\n")

    print(f"Summary of physical page counts:")
    print(f"  {'Event':<35} {'Physical Pages':<15} {'Change'}")
    print("  " + "-" * 55)
    print(f"  {'Before fork()':<35} {parent_pages:<15} {'(baseline)'}")
    print(f"  {'After fork()':<35} {phase1_physical:<15} {'+0 (shared)'}")
    print(f"  {'After child modifies 10 pages':<35} {phase2_physical:<15} {f'+{child_modified} (COW copies)'}")
    print(f"  {'After child terminates':<35} {phase3_physical:<15} {f'-{child_modified} (freed)'}")

    print(f"\nCOW savings: Without COW, fork() would immediately copy all")
    print(f"{parent_pages} pages, costing {parent_pages * 4}KB (assuming 4KB pages).")
    print(f"With COW, only {child_modified} pages were actually copied ({child_modified * 4}KB),")
    print(f"saving {(parent_pages - child_modified) * 4}KB of unnecessary copying.")


# === Exercise 4: mmap() Analysis ===
# Problem: Predict behavior of MAP_SHARED vs MAP_PRIVATE mappings.

def exercise_4():
    """Analyze mmap() MAP_SHARED vs MAP_PRIVATE behavior."""
    print("Code analysis:\n")
    print("```c")
    print('int fd = open("test.txt", O_RDWR);  // Contents: "AAAAAAAAAA"')
    print("char* p1 = mmap(NULL, 10, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);")
    print("char* p2 = mmap(NULL, 10, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);")
    print("")
    print("p1[0] = 'B';")
    print("p2[1] = 'C';")
    print("")
    print('printf("p1: %.10s\\n", p1);')
    print('printf("p2: %.10s\\n", p2);')
    print("```\n")

    # Simulate the behavior
    file_content = list("AAAAAAAAAA")

    print(f"Initial file:  {''.join(file_content)}\n")

    # p1 is MAP_SHARED -- changes are reflected to the file and visible to other mappings
    print("Step 1: p1[0] = 'B' (MAP_SHARED)")
    file_content[0] = 'B'
    print(f"  MAP_SHARED: write goes through to the file page cache")
    print(f"  File content now: {''.join(file_content)}")
    print(f"  Change is visible to all shared mappings and the file itself\n")

    # p2 is MAP_PRIVATE -- COW semantics
    p2_content = list(file_content)  # p2 sees the current file state
    print("Step 2: p2[1] = 'C' (MAP_PRIVATE)")
    print(f"  MAP_PRIVATE: triggers Copy-on-Write")
    print(f"  The page is copied to a private page for p2")
    p2_content[1] = 'C'
    print(f"  p2's private copy: {''.join(p2_content)}")
    print(f"  File content unchanged by this write: {''.join(file_content)}\n")

    print(f"Output:")
    print(f"  p1: {''.join(file_content)}")
    print(f"  p2: {''.join(p2_content)}\n")

    print(f"Explanation:")
    print(f"  p1 = BAAAAAAAAA")
    print(f"    p1 is MAP_SHARED, so p1[0]='B' modifies the file.")
    print(f"    p1 sees the current file content.")
    print(f"")
    print(f"  p2 = BCAAAAAAAA")
    print(f"    p2 initially shares the same page as the file.")
    print(f"    p2 sees p1's 'B' change (both map the same file page before COW).")
    print(f"    When p2[1]='C' is written, COW triggers: p2 gets a private copy")
    print(f"    with both 'B' (from p1's earlier write) and 'C' (p2's write).\n")

    print(f"If we re-read the file from disk:")
    print(f"  Content: {''.join(file_content)}")
    print(f"  Only p1's MAP_SHARED change ('B') is reflected in the file.")
    print(f"  p2's MAP_PRIVATE change ('C') is visible ONLY in p2's address space.")


# === Exercise 5: Demand Paging Design ===
# Problem: Compare Pure Demand Paging vs Prefetching approaches.

def exercise_5():
    """Evaluate demand paging vs prefetching for OS design."""
    print("OS Design Decision: Demand Paging vs Prefetching\n")

    print("=== Pure Demand Paging ===\n")
    print("  Mechanism: Load a page into memory ONLY when a page fault occurs.")
    print("  No pages are loaded until the process actually accesses them.\n")

    print("  Advantages:")
    print("    + No unnecessary pages loaded (minimal memory waste)")
    print("    + Memory usage is exactly what the process needs right now")
    print("    + Simple to implement: just handle page faults as they come")
    print("    + Good for processes that use only a small fraction of their code\n")

    print("  Disadvantages:")
    print("    - Many page faults at program start (cold start penalty)")
    print("    - Each fault incurs ~10ms disk I/O latency")
    print("    - Random access patterns cause continuous faults")
    print("    - Poor performance for sequential scans (e.g., reading a large file)\n")

    # Quantitative example
    startup_pages = 50
    fault_time_ms = 10
    startup_time = startup_pages * fault_time_ms
    print(f"  Example: Program touches {startup_pages} pages at startup")
    print(f"    Pure demand paging: {startup_pages} faults * {fault_time_ms}ms = {startup_time}ms = {startup_time / 1000:.1f}s")
    print(f"    This is noticeable to users!\n")

    print("=== Prefetching (Read-Ahead) ===\n")
    print("  Mechanism: When a page fault occurs, load the faulting page PLUS")
    print("  additional pages that are likely to be accessed soon.\n")

    print("  Advantages:")
    print("    + Dramatically fewer page faults for sequential access")
    print("    + Disk reads are more efficient in larger chunks (amortize seek time)")
    print("    + Exploits spatial locality: nearby pages are likely needed soon")
    print("    + Faster program startup (load code pages ahead of execution)\n")

    print("  Disadvantages:")
    print("    - Wasted memory if predictions are wrong")
    print("    - Unnecessary disk I/O for pages never accessed")
    print("    - Complex implementation (need prediction algorithm)")
    print("    - May evict useful pages to make room for prefetched ones\n")

    prefetch_pages = 8  # typical readahead
    num_faults = startup_pages // prefetch_pages
    prefetch_time = num_faults * fault_time_ms
    print(f"  Example: Same {startup_pages} pages, prefetch {prefetch_pages} pages per fault")
    print(f"    Prefetching: ~{num_faults} faults * {fault_time_ms}ms = {prefetch_time}ms")
    print(f"    Speedup: {startup_time / prefetch_time:.1f}x faster startup\n")

    print("=== Real-World Approach (Linux) ===\n")
    print("  Linux uses ADAPTIVE demand paging with intelligent prefetching:")
    print("  1. Default: Pure demand paging")
    print("  2. Sequential detection: When the OS detects sequential page")
    print("     faults (pages N, N+1, N+2...), it enables readahead:")
    print("     - Initial readahead window: 4 pages")
    print("     - Grows exponentially up to 128 pages (512KB)")
    print("  3. madvise() hints: Applications can guide prefetching:")
    print("     - MADV_SEQUENTIAL: aggressive readahead")
    print("     - MADV_RANDOM: disable readahead")
    print("     - MADV_WILLNEED: prefetch specific range")
    print("     - MADV_DONTNEED: release pages immediately\n")

    print("  Recommendation for a new OS:")
    print("    Start with demand paging as default.")
    print("    Add adaptive prefetching triggered by access pattern detection.")
    print("    Provide user-space hints (like madvise) for applications")
    print("    that know their own access patterns better than the OS can guess.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Page Fault Scenario ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: EAT Calculation with Page Faults ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Copy-on-Write Analysis ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: mmap() Analysis ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Demand Paging Design ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
