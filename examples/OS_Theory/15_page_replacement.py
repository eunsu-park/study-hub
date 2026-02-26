"""
Page Replacement Algorithms

Implements and compares:
- FIFO (First-In, First-Out)
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- Clock (Second-Chance)
- Optimal (Bélády's algorithm — theoretical best)

Theory:
- When a page fault occurs and all frames are occupied, the OS
  must evict a page. The choice of victim determines performance.
- Optimal: evict page not used for longest future time (not implementable)
- FIFO: evict oldest page (simple but suffers Bélády's anomaly)
- LRU: evict least recently used page (good but expensive)
- Clock: approximation of LRU using reference bits
- LFU: evict least frequently used page

Adapted from OS Theory Lesson 15.
"""


def fifo(pages: list[int], n_frames: int) -> dict:
    """FIFO page replacement. Evicts oldest page in memory."""
    frames: list[int] = []
    faults = 0
    history: list[dict] = []

    for page in pages:
        hit = page in frames
        evicted = None
        if not hit:
            faults += 1
            if len(frames) >= n_frames:
                # Evict from front — frames list acts as a queue (FIFO order).
                # Simple but can exhibit Belady's anomaly because eviction
                # ignores actual usage patterns.
                evicted = frames.pop(0)
            frames.append(page)

        history.append({
            "page": page,
            "frames": frames[:],
            "fault": not hit,
            "evicted": evicted,
        })

    return {"name": "FIFO", "faults": faults, "history": history}


def lru(pages: list[int], n_frames: int) -> dict:
    """LRU page replacement. Evicts least recently used page."""
    frames: list[int] = []
    last_used: dict[int, int] = {}
    faults = 0
    history: list[dict] = []

    for t, page in enumerate(pages):
        hit = page in frames
        evicted = None
        if not hit:
            faults += 1
            if len(frames) >= n_frames:
                # Find LRU page — uses timestamps (last access time) to identify
                # the victim. A real OS cannot afford O(n) scans, which is why
                # Clock approximation exists. LRU is a stack algorithm, so it
                # never exhibits Belady's anomaly.
                lru_page = min(frames, key=lambda p: last_used[p])
                frames.remove(lru_page)
                evicted = lru_page
            frames.append(page)
        last_used[page] = t

        history.append({
            "page": page,
            "frames": frames[:],
            "fault": not hit,
            "evicted": evicted,
        })

    return {"name": "LRU", "faults": faults, "history": history}


def lfu(pages: list[int], n_frames: int) -> dict:
    """LFU page replacement. Evicts least frequently used page."""
    frames: list[int] = []
    freq: dict[int, int] = {}
    arrival: dict[int, int] = {}
    faults = 0
    history: list[dict] = []

    for t, page in enumerate(pages):
        hit = page in frames
        evicted = None
        if not hit:
            faults += 1
            if len(frames) >= n_frames:
                # Find LFU — tie-break by oldest arrival (FIFO among same-
                # frequency pages) to prevent recently loaded pages from being
                # evicted before they've had a chance to accumulate references
                lfu_page = min(
                    frames,
                    key=lambda p: (freq.get(p, 0), arrival.get(p, 0))
                )
                frames.remove(lfu_page)
                evicted = lfu_page
            frames.append(page)
            arrival[page] = t

        freq[page] = freq.get(page, 0) + 1

        history.append({
            "page": page,
            "frames": frames[:],
            "fault": not hit,
            "evicted": evicted,
        })

    return {"name": "LFU", "faults": faults, "history": history}


def clock(pages: list[int], n_frames: int) -> dict:
    """Clock (second-chance) page replacement.

    Uses a circular buffer with reference bits.
    On fault: advance clock hand, clear ref bits until finding
    a page with ref=0, then evict it.
    """
    frames: list[int] = [-1] * n_frames
    ref_bits: list[bool] = [False] * n_frames
    hand = 0
    count = 0  # pages loaded so far
    faults = 0
    history: list[dict] = []

    for page in pages:
        # Check if page is in frames
        hit = False
        for i in range(n_frames):
            if frames[i] == page:
                ref_bits[i] = True
                hit = True
                break

        evicted = None
        if not hit:
            faults += 1
            if count < n_frames:
                # Still have empty frames
                frames[count] = page
                ref_bits[count] = True
                count += 1
            else:
                # Clock sweep: pages with ref_bit=1 get a "second chance" —
                # clear the bit and advance. This approximates LRU with only
                # 1 bit per frame (vs. full timestamps), making it practical
                # for real hardware where the MMU sets the reference bit.
                while ref_bits[hand]:
                    ref_bits[hand] = False  # second chance
                    hand = (hand + 1) % n_frames
                evicted = frames[hand]
                frames[hand] = page
                ref_bits[hand] = True
                hand = (hand + 1) % n_frames

        history.append({
            "page": page,
            "frames": [f for f in frames if f != -1],
            "fault": not hit,
            "evicted": evicted,
        })

    return {"name": "Clock", "faults": faults, "history": history}


def optimal(pages: list[int], n_frames: int) -> dict:
    """Optimal (Bélády's) page replacement.

    Evicts the page not used for the longest future time.
    Requires knowledge of future accesses (theoretical bound).
    """
    frames: list[int] = []
    faults = 0
    history: list[dict] = []

    for t, page in enumerate(pages):
        hit = page in frames
        evicted = None
        if not hit:
            faults += 1
            if len(frames) >= n_frames:
                # Find page used farthest in the future
                future_use = {}
                for f in frames:
                    try:
                        future_use[f] = pages[t + 1:].index(f)
                    except ValueError:
                        # Page never used again — evicting it costs nothing;
                        # this is the ideal victim and gives Optimal its
                        # theoretical minimum fault rate
                        future_use[f] = float("inf")
                victim = max(frames, key=lambda p: future_use[p])
                frames.remove(victim)
                evicted = victim
            frames.append(page)

        history.append({
            "page": page,
            "frames": frames[:],
            "fault": not hit,
            "evicted": evicted,
        })

    return {"name": "Optimal", "faults": faults, "history": history}


# ── Visualization ───────────────────────────────────────────────────────

def print_trace(result: dict, n_frames: int) -> None:
    """Print page replacement trace table."""
    history = result["history"]
    name = result["name"]

    # Header
    pages_str = "  ".join(f"{h['page']:>2}" for h in history)
    print(f"\n  {name} (frames={n_frames}):")
    print(f"  Pages: {pages_str}")

    # Frame rows
    for f in range(n_frames):
        row = []
        for h in history:
            frames = h["frames"]
            if f < len(frames):
                row.append(f"{frames[f]:>2}")
            else:
                row.append(" .")
        print(f"  F{f}:    {'  '.join(row)}")

    # Fault indicators
    faults_str = "  ".join(
        " *" if h["fault"] else "  " for h in history
    )
    print(f"  Fault: {faults_str}")
    print(f"  Total faults: {result['faults']}/{len(history)} "
          f"({result['faults']/len(history):.1%})")


def demo_comparison():
    """Compare all page replacement algorithms."""
    print("=" * 60)
    print("PAGE REPLACEMENT ALGORITHM COMPARISON")
    print("=" * 60)

    # Classic reference string
    pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    n_frames = 3

    print(f"\n  Reference string: {pages}")
    print(f"  Frames: {n_frames}")

    algorithms = [optimal, fifo, lru, lfu, clock]
    results = [algo(pages, n_frames) for algo in algorithms]

    for result in results:
        print_trace(result, n_frames)

    # Summary
    print("\n  " + "-" * 40)
    print(f"  {'Algorithm':<12} {'Faults':>8} {'Hit Rate':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    for r in results:
        hit_rate = 1 - r["faults"] / len(pages)
        print(f"  {r['name']:<12} {r['faults']:>8} {hit_rate:>9.1%}")
    print(f"\n  Optimal is the theoretical lower bound.")


def demo_belady_anomaly():
    """Demonstrate Bélády's anomaly: more frames → more faults with FIFO."""
    print("\n" + "=" * 60)
    print("BÉLÁDY'S ANOMALY (FIFO)")
    print("=" * 60)

    # Classic anomaly sequence
    pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]

    print(f"\n  Reference string: {pages}")
    print(f"\n  {'Frames':>8} {'FIFO Faults':>13} {'LRU Faults':>12}")
    print(f"  {'-'*8} {'-'*13} {'-'*12}")

    for n in range(3, 6):
        fifo_r = fifo(pages, n)
        lru_r = lru(pages, n)
        anomaly = ""
        if n > 3 and fifo_r["faults"] > fifo(pages, n - 1)["faults"]:
            anomaly = " ← ANOMALY"
        print(f"  {n:>8} {fifo_r['faults']:>13} {lru_r['faults']:>12}{anomaly}")

    print(f"\n  FIFO can have MORE faults with MORE frames!")
    print(f"  LRU never exhibits this anomaly (it's a stack algorithm).")


if __name__ == "__main__":
    demo_comparison()
    demo_belady_anomaly()
