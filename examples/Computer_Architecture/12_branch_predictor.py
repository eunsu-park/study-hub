"""
Branch Predictor Simulator

Demonstrates:
- Static prediction (always taken / always not-taken)
- 1-bit dynamic predictor
- 2-bit saturating counter predictor
- Branch Target Buffer (BTB)
- Correlating predictor (1,1)

Theory:
- Branches are ~15-25% of instructions. Misprediction flushes the
  pipeline (penalty = pipeline depth cycles).
- 1-bit predictor: remembers last outcome. Fails on alternating branches.
- 2-bit saturating counter: requires two consecutive mispredictions to
  change prediction. Better for loop branches.
- BTB: cache mapping branch PC to target address.

Adapted from Computer Architecture Lesson 12.
"""

from dataclasses import dataclass


@dataclass
class BranchEvent:
    """A single branch event."""
    pc: int
    taken: bool
    target: int = 0


# ── Static Predictor ────────────────────────────────────────────────────

def predict_always_taken(events: list[BranchEvent]) -> dict:
    correct = sum(1 for e in events if e.taken)
    return {"name": "Always Taken", "correct": correct, "total": len(events)}


def predict_always_not_taken(events: list[BranchEvent]) -> dict:
    correct = sum(1 for e in events if not e.taken)
    return {"name": "Always Not-Taken", "correct": correct, "total": len(events)}


# ── 1-Bit Predictor ────────────────────────────────────────────────────

class OneBitPredictor:
    """1-bit branch predictor. Remembers last outcome."""

    def __init__(self):
        self.table: dict[int, bool] = {}  # PC → last prediction

    def predict(self, pc: int) -> bool:
        # Default to "taken" because most branches are loop-back edges,
        # which are taken on all iterations except the last one.
        return self.table.get(pc, True)

    def update(self, pc: int, actual: bool) -> None:
        # 1-bit predictor changes prediction after every misprediction.
        # This makes it vulnerable to regular patterns like loops: it
        # mispredicts both the last (exit) and the first (re-entry)
        # iteration of every loop execution.
        self.table[pc] = actual


# ── 2-Bit Saturating Counter ───────────────────────────────────────────

class TwoBitPredictor:
    """2-bit saturating counter predictor.

    States: 0=Strongly Not-Taken, 1=Weakly Not-Taken,
            2=Weakly Taken, 3=Strongly Taken
    """

    def __init__(self):
        self.table: dict[int, int] = {}  # PC → counter (0-3)

    def predict(self, pc: int) -> bool:
        counter = self.table.get(pc, 2)  # default: weakly taken
        return counter >= 2

    def update(self, pc: int, actual: bool) -> None:
        counter = self.table.get(pc, 2)
        # Saturating increment/decrement: the counter clamps at 0 and 3
        # instead of wrapping.  This means a single anomalous outcome
        # (e.g., the loop-exit branch) only moves the counter one step —
        # it takes two consecutive mispredictions to actually flip the
        # prediction direction, which is why 2-bit beats 1-bit on loops.
        if actual:
            counter = min(3, counter + 1)
        else:
            counter = max(0, counter - 1)
        self.table[pc] = counter

    def state_name(self, counter: int) -> str:
        names = ["SN", "WN", "WT", "ST"]
        return names[counter]


# ── Branch Target Buffer ───────────────────────────────────────────────

class BTB:
    """Branch Target Buffer — caches branch targets."""

    def __init__(self, size: int = 16):
        self.size = size
        self.entries: dict[int, int] = {}  # PC → target
        self.hits = 0
        self.misses = 0

    def lookup(self, pc: int) -> int | None:
        if pc in self.entries:
            self.hits += 1
            return self.entries[pc]
        self.misses += 1
        return None

    def update(self, pc: int, target: int) -> None:
        if len(self.entries) >= self.size and pc not in self.entries:
            # FIFO eviction is used here for simplicity.  Real BTBs
            # typically use LRU or pseudo-LRU, but the key insight is
            # that the BTB is a cache of *targets*, separate from the
            # direction predictor — even a BTB hit is useless if the
            # branch is predicted not-taken.
            oldest = next(iter(self.entries))
            del self.entries[oldest]
        self.entries[pc] = target


# ── Correlating Predictor ──────────────────────────────────────────────

class CorrelatingPredictor:
    """(1,1) correlating predictor.

    Uses global branch history (1 bit) to index into
    two sets of 2-bit counters per branch.
    """

    def __init__(self):
        # table[pc][global_history] → 2-bit counter
        self.table: dict[int, list[int]] = {}
        self.global_history = 0  # 1-bit history

    def predict(self, pc: int) -> bool:
        if pc not in self.table:
            self.table[pc] = [2, 2]  # default: weakly taken
        counter = self.table[pc][self.global_history]
        return counter >= 2

    def update(self, pc: int, actual: bool) -> None:
        if pc not in self.table:
            self.table[pc] = [2, 2]
        counter = self.table[pc][self.global_history]
        if actual:
            counter = min(3, counter + 1)
        else:
            counter = max(0, counter - 1)
        self.table[pc][self.global_history] = counter
        # Shift the outcome into global history AFTER updating the counter,
        # so the counter update uses the history context that was active
        # when the prediction was made (not the newly observed outcome).
        self.global_history = int(actual)


# ── Simulation ──────────────────────────────────────────────────────────

def simulate(name: str, events: list[BranchEvent], predictor) -> dict:
    """Run predictor over events, return accuracy."""
    correct = 0
    for event in events:
        prediction = predictor.predict(event.pc)
        if prediction == event.taken:
            correct += 1
        predictor.update(event.pc, event.taken)

    return {"name": name, "correct": correct, "total": len(events)}


def print_results(results: list[dict]) -> None:
    print(f"\n  {'Predictor':<25} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*10}")
    for r in results:
        acc = r["correct"] / r["total"] * 100
        print(f"  {r['name']:<25} {r['correct']:>8} {r['total']:>7} {acc:>9.1f}%")


# ── Demo Workloads ──────────────────────────────────────────────────────

def demo_loop_branch():
    """Simulate a typical loop branch (taken N-1 times, not taken once)."""
    print("=" * 60)
    print("BRANCH PREDICTION: LOOP PATTERN")
    print("=" * 60)

    loop_iterations = 10
    n_loops = 20
    events = []
    for _ in range(n_loops):
        for i in range(loop_iterations):
            taken = i < loop_iterations - 1  # taken all but last
            events.append(BranchEvent(pc=100, taken=taken, target=100))

    print(f"\n  Pattern: loop of {loop_iterations} iterations, repeated {n_loops} times")
    print(f"  Total branches: {len(events)}")

    results = [
        predict_always_taken(events),
        predict_always_not_taken(events),
        simulate("1-bit", events, OneBitPredictor()),
        simulate("2-bit saturating", events, TwoBitPredictor()),
        simulate("Correlating (1,1)", events, CorrelatingPredictor()),
    ]
    print_results(results)

    # Show 2-bit state transitions for first loop
    print(f"\n  2-bit counter state for first loop iteration:")
    pred = TwoBitPredictor()
    print(f"    {'Iter':>5}  {'Actual':>7}  {'Predict':>8}  {'Correct':>8}  {'State':>6}")
    for i in range(loop_iterations):
        taken = i < loop_iterations - 1
        prediction = pred.predict(100)
        correct = prediction == taken
        counter = pred.table.get(100, 2)
        state = pred.state_name(counter)
        pred.update(100, taken)
        print(f"    {i:>5}  {'T' if taken else 'NT':>7}  "
              f"{'T' if prediction else 'NT':>8}  "
              f"{'✓' if correct else '✗':>8}  {state:>6}")


def demo_alternating():
    """Alternating branch pattern — worst case for 1-bit."""
    print("\n" + "=" * 60)
    print("BRANCH PREDICTION: ALTERNATING PATTERN")
    print("=" * 60)

    events = [BranchEvent(pc=200, taken=(i % 2 == 0)) for i in range(40)]

    print(f"\n  Pattern: T, NT, T, NT, T, NT, ...")
    print(f"  Total branches: {len(events)}")

    results = [
        predict_always_taken(events),
        predict_always_not_taken(events),
        simulate("1-bit", events, OneBitPredictor()),
        simulate("2-bit saturating", events, TwoBitPredictor()),
        simulate("Correlating (1,1)", events, CorrelatingPredictor()),
    ]
    print_results(results)


def demo_btb():
    """Demonstrate Branch Target Buffer."""
    print("\n" + "=" * 60)
    print("BRANCH TARGET BUFFER (BTB)")
    print("=" * 60)

    btb = BTB(size=4)
    events = [
        (100, 200), (200, 300), (100, 200), (300, 400),
        (400, 500), (100, 200), (500, 100), (200, 300),
    ]

    print(f"\n  BTB size: 4 entries")
    print(f"\n  {'PC':>5}  {'Target':>7}  {'BTB Hit':>8}  {'BTB Pred':>9}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}")

    for pc, target in events:
        predicted_target = btb.lookup(pc)
        hit = predicted_target is not None
        correct_target = predicted_target == target if hit else False
        btb.update(pc, target)
        print(f"  {pc:>5}  {target:>7}  {'HIT' if hit else 'MISS':>8}  "
              f"{'correct' if correct_target else 'wrong/miss':>9}")

    print(f"\n  BTB hits: {btb.hits}, misses: {btb.misses}")
    print(f"  Hit rate: {btb.hits/(btb.hits+btb.misses)*100:.1f}%")


if __name__ == "__main__":
    demo_loop_branch()
    demo_alternating()
    demo_btb()
