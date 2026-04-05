"""
Exercises for Lesson 12: Branch Prediction
Topic: Computer_Architecture

Solutions to practice problems covering static vs dynamic prediction,
1-bit and 2-bit saturating counters, Branch Target Buffers (BTB),
branch prediction performance impact, and advanced prediction techniques.
"""


def exercise_1():
    """
    Compare static branch prediction strategies:
    always-taken, always-not-taken, backward-taken-forward-not-taken (BTFNT).
    """
    print("Static Branch Prediction Strategies:")
    print()

    strategies = {
        "Always Not Taken": {
            "prediction": "Assume branch is NOT taken (fall through)",
            "advantage": "Simple; no hardware needed. Correct for forward if-else exits.",
            "disadvantage": "Poor for loops (backward branches are usually taken).",
            "typical_accuracy": "~40-50%",
        },
        "Always Taken": {
            "prediction": "Assume branch IS taken (go to target)",
            "advantage": "Good for loops (most backward branches are taken).",
            "disadvantage": "Poor for forward conditional branches.",
            "typical_accuracy": "~60-70%",
        },
        "BTFNT (Backward Taken, Forward Not Taken)": {
            "prediction": "Backward branches: taken (loops). Forward: not taken (if-else).",
            "advantage": "Matches common code patterns well.",
            "disadvantage": "Requires knowing branch direction at decode.",
            "typical_accuracy": "~65-75%",
        },
    }

    for name, info in strategies.items():
        print(f"  {name}:")
        for key, val in info.items():
            print(f"    {key:<22s}: {val}")
        print()

    # Simulate on a loop pattern
    print("  Simulation: for (i=0; i<10; i++) loop body")
    print("  Branch at end of loop: taken 9 times, not-taken 1 time")
    branch_outcomes = [True] * 9 + [False]  # T T T T T T T T T NT

    for name, predict_func in [
        ("Always Not Taken", lambda i, outcome: False),
        ("Always Taken", lambda i, outcome: True),
        ("BTFNT (backward)", lambda i, outcome: True),  # Loop = backward
    ]:
        correct = sum(1 for i, o in enumerate(branch_outcomes) if predict_func(i, o) == o)
        accuracy = correct / len(branch_outcomes)
        print(f"    {name:<30s}: {correct}/10 correct = {accuracy:.0%}")


def exercise_2():
    """
    Simulate 1-bit branch predictor for a loop that runs 4 times.
    Pattern: T T T NT T T T NT ...
    """
    print("1-bit Branch Predictor Simulation:")
    print("  Loop pattern: T T T NT T T T NT (repeating)")
    print()

    # 1-bit predictor: predict based on last outcome
    pattern = [True, True, True, False] * 3  # 3 iterations of the loop
    state = False  # Initial prediction: Not Taken
    correct = 0
    total = 0

    print(f"  {'#':>3s} {'Actual':>7s} {'Predict':>8s} {'Correct':>8s} {'New State':>10s}")
    print(f"  {'-'*3} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")

    for i, actual in enumerate(pattern):
        prediction = state
        is_correct = prediction == actual
        if is_correct:
            correct += 1
        total += 1
        state = actual  # 1-bit: update to last outcome

        actual_str = "T" if actual else "NT"
        pred_str = "T" if prediction else "NT"
        corr_str = "HIT" if is_correct else "MISS"
        state_str = "T" if state else "NT"
        print(f"  {i+1:>3d} {actual_str:>7s} {pred_str:>8s} {corr_str:>8s} {state_str:>10s}")

    accuracy = correct / total
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"\n  Problem: 1-bit predictor mispredicts TWICE per loop iteration:")
    print(f"    1. On the last iteration (NT after predicting T)")
    print(f"    2. On the first iteration of next loop (T after predicting NT)")


def exercise_3():
    """
    Simulate 2-bit saturating counter predictor for the same loop pattern.
    States: 00=Strongly NT, 01=Weakly NT, 10=Weakly T, 11=Strongly T
    """
    print("2-bit Saturating Counter Predictor Simulation:")
    print("  States: 00=Strongly NT, 01=Weakly NT, 10=Weakly T, 11=Strongly T")
    print("  Loop pattern: T T T NT (repeating)")
    print()

    pattern = [True, True, True, False] * 3
    state = 0b00  # Start at Strongly Not Taken

    state_names = {0b00: "SN", 0b01: "WN", 0b10: "WT", 0b11: "ST"}
    correct = 0
    total = 0

    print(f"  {'#':>3s} {'Actual':>7s} {'State':>6s} {'Predict':>8s} {'Correct':>8s} {'NewState':>9s}")
    print(f"  {'-'*3} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")

    for i, actual in enumerate(pattern):
        prediction = state >= 0b10  # Predict Taken if state is 10 or 11
        is_correct = prediction == actual
        if is_correct:
            correct += 1
        total += 1

        old_state = state
        # Update state
        if actual:  # Branch was taken
            state = min(state + 1, 0b11)
        else:  # Branch was not taken
            state = max(state - 1, 0b00)

        actual_str = "T" if actual else "NT"
        pred_str = "T" if prediction else "NT"
        corr_str = "HIT" if is_correct else "MISS"
        print(f"  {i+1:>3d} {actual_str:>7s} {state_names[old_state]:>6s} {pred_str:>8s} "
              f"{corr_str:>8s} {state_names[state]:>9s}")

    accuracy = correct / total
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"\n  Advantage over 1-bit: After warmup, 2-bit predictor only mispredicts")
    print(f"  ONCE per loop iteration (on the exit). The 'Weakly Taken' state absorbs")
    print(f"  the loop-exit misprediction without flipping the base prediction.")


def exercise_4():
    """
    Explain Branch Target Buffer (BTB) operation.
    """
    print("Branch Target Buffer (BTB):")
    print()
    print("  Purpose: Cache branch instruction addresses and their targets")
    print("  to predict BOTH direction (taken/not-taken) AND target address.")
    print()

    # Simulate BTB
    class BTB:
        def __init__(self, size=4):
            self.entries = {}  # PC -> (target, 2-bit counter)
            self.size = size

        def lookup(self, pc):
            """Returns (hit, predicted_target, predicted_taken)."""
            if pc in self.entries:
                target, counter = self.entries[pc]
                return True, target, counter >= 2
            return False, None, False

        def update(self, pc, target, taken):
            """Update BTB after branch resolution."""
            if pc in self.entries:
                _, counter = self.entries[pc]
                if taken:
                    counter = min(counter + 1, 3)
                else:
                    counter = max(counter - 1, 0)
                self.entries[pc] = (target, counter)
            elif taken:
                # Only allocate on taken branches
                if len(self.entries) >= self.size:
                    # Evict LRU (simplified: evict first entry)
                    oldest = next(iter(self.entries))
                    del self.entries[oldest]
                self.entries[pc] = (target, 2)  # Start at Weakly Taken

    btb = BTB(size=4)

    # Simulate branch sequence
    branches = [
        (0x100, 0x200, True,  "Loop branch (taken)"),
        (0x100, 0x200, True,  "Loop branch (taken)"),
        (0x100, 0x200, True,  "Loop branch (taken)"),
        (0x100, 0x200, False, "Loop exit (not taken)"),
        (0x300, 0x400, True,  "If-else branch (taken)"),
        (0x100, 0x200, True,  "Loop restart (taken)"),
    ]

    print(f"  {'PC':>6s} {'Target':>8s} {'Actual':>7s} {'BTB Hit':>8s} {'Predict':>8s} {'Correct':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

    for pc, target, taken, desc in branches:
        hit, pred_target, pred_taken = btb.lookup(pc)
        correct = hit and (pred_taken == taken) and (pred_target == target if taken else True)
        btb.update(pc, target, taken)

        print(f"  0x{pc:03X} 0x{target:03X}  {'T':>5s if taken else 'NT':>5s}  "
              f"{'HIT' if hit else 'MISS':>7s}  {'T' if pred_taken and hit else 'NT':>7s}  "
              f"{'HIT' if correct else 'MISS':>7s}  {desc}")

    print()
    print("  BTB structure: [PC tag | Target Address | 2-bit Counter | Valid]")
    print("  Typical size: 256-4096 entries")
    print("  Lookup: During IF stage, check if PC matches a BTB entry")
    print("  If hit + predicted taken → redirect fetch to target immediately")


def exercise_5():
    """
    Calculate performance impact of branch prediction accuracy.
    Compare 85% vs 95% vs 99% accuracy with a 15-cycle pipeline.
    """
    pipeline_depth = 15  # Modern deep pipeline
    branch_freq = 0.20   # 20% of instructions are branches
    base_cpi = 1.0

    print("Branch Prediction Performance Impact:")
    print(f"  Pipeline depth: {pipeline_depth} stages")
    print(f"  Branch frequency: {branch_freq:.0%}")
    print(f"  Misprediction penalty: {pipeline_depth} cycles (full flush)")
    print()

    accuracies = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    print(f"  {'Accuracy':>9s} {'Miss Rate':>10s} {'Penalty/Instr':>14s} {'CPI':>6s} {'Slowdown':>10s}")
    print(f"  {'-'*9} {'-'*10} {'-'*14} {'-'*6} {'-'*10}")

    for acc in accuracies:
        miss_rate = 1 - acc
        penalty_per_instr = branch_freq * miss_rate * pipeline_depth
        cpi = base_cpi + penalty_per_instr
        slowdown = (cpi - base_cpi) / base_cpi

        print(f"  {acc:>9.0%} {miss_rate:>10.0%} {penalty_per_instr:>14.2f} {cpi:>6.2f} {slowdown:>10.0%}")

    print()
    print("  Key insight: Even 1% improvement in accuracy matters significantly")
    print("  in deep pipelines. Going from 95% → 99% reduces CPI from 1.15 → 1.03.")
    print()
    print("  This is why modern CPUs invest heavily in branch prediction:")
    print("  - TAGE predictor (tagged geometric history)")
    print("  - Perceptron predictor")
    print("  - 64KB+ prediction tables")
    print("  - Achieving >97% accuracy on real workloads")


def exercise_6():
    """
    Simulate a global history-based 2-bit predictor (correlating predictor).
    Use a 2-bit global history register (GHR) to index into a pattern history table (PHT).
    """
    print("Global History Branch Predictor Simulation:")
    print("  Uses recent branch history to correlate predictions.")
    print("  GHR: 2-bit shift register of recent branch outcomes")
    print("  PHT: 4 entries (indexed by GHR), each a 2-bit saturating counter")
    print()

    # Pattern: alternating branches with correlation
    # Branch A: T, NT, T, NT (alternating)
    # A correlating predictor should learn this pattern
    pattern = [True, False, True, False, True, False, True, False]

    ghr = 0b00  # 2-bit global history register
    pht = [0b01, 0b01, 0b01, 0b01]  # 4 entries, start at Weakly NT

    state_names = {0: "SN", 1: "WN", 2: "WT", 3: "ST"}
    correct = 0

    print(f"  {'#':>3s} {'GHR':>4s} {'PHT[GHR]':>9s} {'Predict':>8s} {'Actual':>7s} "
          f"{'Correct':>8s} {'NewGHR':>7s}")
    print(f"  {'-'*3} {'-'*4} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

    for i, actual in enumerate(pattern):
        # Predict using PHT[GHR]
        counter = pht[ghr]
        prediction = counter >= 2

        is_correct = prediction == actual
        if is_correct:
            correct += 1

        # Update PHT
        if actual:
            pht[ghr] = min(pht[ghr] + 1, 3)
        else:
            pht[ghr] = max(pht[ghr] - 1, 0)

        old_ghr = ghr
        # Update GHR
        ghr = ((ghr << 1) | (1 if actual else 0)) & 0b11

        actual_str = "T" if actual else "NT"
        pred_str = "T" if prediction else "NT"
        corr_str = "HIT" if is_correct else "MISS"
        print(f"  {i+1:>3d} {format(old_ghr, '02b'):>4s} {state_names[counter]:>9s} "
              f"{pred_str:>8s} {actual_str:>7s} {corr_str:>8s} {format(ghr, '02b'):>7s}")

    accuracy = correct / len(pattern)
    print(f"\n  Accuracy: {correct}/{len(pattern)} = {accuracy:.1%}")
    print(f"\n  After warmup, the correlating predictor learns that:")
    print(f"    After TN (GHR=10): next is T")
    print(f"    After NT (GHR=01): next is N")
    print(f"  This pattern is impossible for a simple 1-bit/2-bit local predictor.")


def exercise_7():
    """
    Calculate BTB hit rate impact on prediction effectiveness.
    """
    print("BTB Miss Impact on Branch Prediction:")
    print()
    print("  Even with a perfect direction predictor, a BTB miss means")
    print("  we cannot redirect fetch — effectively a misprediction.")
    print()

    btb_sizes = [64, 256, 1024, 4096]
    branch_freq = 0.20
    unique_branches = 500  # Number of unique branch PCs in workload
    pipeline_penalty = 12

    print(f"  Workload: {unique_branches} unique branch PCs, {branch_freq:.0%} branch freq")
    print()
    print(f"  {'BTB Size':>9s} {'Hit Rate':>9s} {'Eff Accuracy':>14s} {'CPI':>6s}")
    print(f"  {'-'*9} {'-'*9} {'-'*14} {'-'*6}")

    direction_accuracy = 0.95  # 95% direction prediction accuracy

    for size in btb_sizes:
        # Simplified: if BTB size >= unique branches, ~100% hit rate
        # Otherwise, some branches miss (conflict misses)
        hit_rate = min(size / unique_branches, 1.0)
        # Effective accuracy = BTB hit rate * direction accuracy
        # BTB miss → treated as misprediction
        eff_accuracy = hit_rate * direction_accuracy
        miss_rate = 1 - eff_accuracy
        cpi = 1.0 + branch_freq * miss_rate * pipeline_penalty

        print(f"  {size:>9d} {hit_rate:>9.1%} {eff_accuracy:>14.1%} {cpi:>6.2f}")

    print()
    print("  Lesson: BTB size must be large enough to capture the working set")
    print("  of branch instructions, or even perfect direction prediction is wasted.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Static Prediction Strategies", exercise_1),
        ("Exercise 2: 1-bit Predictor Simulation", exercise_2),
        ("Exercise 3: 2-bit Saturating Counter", exercise_3),
        ("Exercise 4: Branch Target Buffer", exercise_4),
        ("Exercise 5: Prediction Accuracy Impact", exercise_5),
        ("Exercise 6: Correlating Predictor", exercise_6),
        ("Exercise 7: BTB Size Impact", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")
