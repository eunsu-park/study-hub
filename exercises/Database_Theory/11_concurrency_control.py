"""
Exercises for Lesson 11: Concurrency Control
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers lock-based protocols, timestamp ordering, MVCC, OCC validation,
deadlock detection, and concurrency control scheme selection.
"""


# === Exercise 1: Shared vs Exclusive Locks ===
# Problem: Explain lock compatibility.

def exercise_1():
    """Explain S and X lock compatibility."""
    print("Lock Compatibility Matrix:")
    print()
    print("           |  None  |  S-lock  |  X-lock  |")
    print("  ---------+--------+----------+----------+")
    print("  Request S|  Grant |  Grant   |  Wait    |")
    print("  Request X|  Grant |  Wait    |  Wait    |")
    print()

    print("Why multiple S-locks can coexist:")
    print("  S-locks (shared/read locks) are for reading. Multiple transactions")
    print("  reading the same data simultaneously causes no conflict because")
    print("  reads don't modify data. No transaction sees incorrect values.")
    print()

    print("Why multiple X-locks cannot coexist:")
    print("  X-locks (exclusive/write locks) are for writing. If two transactions")
    print("  could write the same item concurrently, we'd get lost updates,")
    print("  inconsistent reads, and other anomalies. Only one writer at a time")
    print("  ensures the data item has a well-defined value at each moment.")


# === Exercise 2: 2PL Variants ===
# Problem: Classify a transaction's locking behavior.

def exercise_2():
    """Classify 2PL variants."""
    print("Transaction T: acquires ALL locks at the beginning, releases ALL at the end.")
    print()
    print("Is T following strict 2PL?")
    print("  YES. Strict 2PL requires all EXCLUSIVE locks held until commit/abort.")
    print("  Since T releases all locks at the end (commit/abort), exclusive locks")
    print("  are held until that point.")
    print()
    print("Is T following rigorous 2PL?")
    print("  YES. Rigorous 2PL requires ALL locks (both S and X) held until commit/abort.")
    print("  Since T releases ALL locks at the end, this is satisfied.")
    print()
    print("Both? YES.")
    print("  T follows rigorous 2PL, which is a stricter version of strict 2PL.")
    print("  Rigorous 2PL => Strict 2PL => Basic 2PL.")
    print()
    print("Note: This is also called 'conservative 2PL' or 'static locking'")
    print("because all locks are acquired upfront. It prevents deadlocks but")
    print("requires knowing the entire read/write set in advance.")


# === Exercise 3: Wait-Die vs Wound-Wait ===
# Problem: Compare deadlock prevention schemes.

def exercise_3():
    """Compare Wait-Die and Wound-Wait schemes."""
    print("Wait-Die (non-preemptive):")
    print("  - Older transaction (lower timestamp) WAITS for younger one")
    print("  - Younger transaction requesting lock held by older: DIES (aborts)")
    print("  - Rule: 'Only old waits for young; young always dies'")
    print("  - No preemption: locks are never forcibly taken")
    print()

    print("Wound-Wait (preemptive):")
    print("  - Older transaction WOUNDS (forces abort of) younger lock holder")
    print("  - Younger transaction requesting lock held by older: WAITS")
    print("  - Rule: 'Old wounds young; young waits for old'")
    print("  - Preemptive: older transaction can force younger to abort")
    print()

    # Simulate scenarios
    print("Example: T1 (ts=10, older) and T2 (ts=20, younger)")
    print()

    scenarios = [
        ("T2 requests lock held by T1", "Wait-Die", "T2 DIES (younger can't wait for older)"),
        ("T2 requests lock held by T1", "Wound-Wait", "T2 WAITS (younger waits for older)"),
        ("T1 requests lock held by T2", "Wait-Die", "T1 WAITS (older can wait for younger)"),
        ("T1 requests lock held by T2", "Wound-Wait", "T1 WOUNDS T2 (older preempts younger)"),
    ]

    print(f"  {'Scenario':<35} {'Scheme':<15} {'Result'}")
    print(f"  {'-'*35} {'-'*15} {'-'*45}")
    for scenario, scheme, result in scenarios:
        print(f"  {scenario:<35} {scheme:<15} {result}")

    print()
    print("Starvation prevention:")
    print("  Both schemes: restarted transactions keep their ORIGINAL timestamp.")
    print("  Over time, a restarted transaction becomes the 'oldest' and gets priority.")
    print("  This guarantees eventual completion (no starvation).")


# === Exercise 5: Lock Request Sequence with Deadlock ===
# Problem: Trace lock requests and detect deadlock.

def exercise_5():
    """Trace lock requests and detect deadlock."""
    print("Lock request sequence:")
    print()

    steps = [
        {
            "request": "T1: lock-S(A)",
            "action": "GRANTED. T1 holds S(A).",
            "state": {"A": [("T1", "S")], "B": []}
        },
        {
            "request": "T2: lock-X(B)",
            "action": "GRANTED. T2 holds X(B).",
            "state": {"A": [("T1", "S")], "B": [("T2", "X")]}
        },
        {
            "request": "T3: lock-S(A)",
            "action": "GRANTED. S-lock compatible with existing S-lock. T1,T3 hold S(A).",
            "state": {"A": [("T1", "S"), ("T3", "S")], "B": [("T2", "X")]}
        },
        {
            "request": "T1: lock-X(B)",
            "action": "WAIT. T2 holds X(B). T1 waits for T2.",
            "state": {"A": [("T1", "S"), ("T3", "S")], "B": [("T2", "X")]}
        },
        {
            "request": "T2: lock-S(A)",
            "action": "GRANTED. S(A) compatible with existing S-locks. T1,T2,T3 hold S(A).",
            "state": {"A": [("T1", "S"), ("T3", "S"), ("T2", "S")], "B": [("T2", "X")]}
        },
        {
            "request": "T3: lock-X(A)",
            "action": "WAIT. T1 and T2 hold S(A). T3 must wait for both to release.",
            "state": {"A": [("T1", "S"), ("T3", "S"), ("T2", "S")], "B": [("T2", "X")]}
        },
    ]

    for i, step in enumerate(steps, 1):
        print(f"  Step {i}: {step['request']}")
        print(f"    Action: {step['action']}")
        print()

    # (b) Deadlock detection
    print("(b) Wait-for graph:")
    print("    T1 -> T2  (T1 waiting for X(B), held by T2)")
    print("    T3 -> T1  (T3 waiting for X(A), T1 holds S(A))")
    print("    T3 -> T2  (T3 waiting for X(A), T2 holds S(A))")
    print()
    print("    No cycle! T1 waits for T2, T3 waits for T1 and T2.")
    print("    If T2 releases S(A): T3 still waits for T1.")
    print("    If T2 releases X(B): T1 gets X(B), completes, releases S(A).")
    print("    Then T3 can get X(A). No deadlock (but T3 may wait a long time).")
    print()

    # (c) Wait-Die resolution
    print("(c) Wait-Die resolution (TS(T1)=100 < TS(T2)=200 < TS(T3)=300):")
    print("    T1 requests X(B) held by T2: T1 is older -> T1 WAITS (ok)")
    print("    T2 requests S(A): compatible with existing S-locks -> GRANTED")
    print("    T3 requests X(A) held by T1: T3 is younger -> T3 DIES (abort)")
    print("    T3 restarts with same timestamp (300), will eventually succeed")


# === Exercise 7: Timestamp Ordering ===
# Problem: Execute operations using basic timestamp ordering and Thomas's Write Rule.

def exercise_7():
    """Timestamp ordering protocol simulation."""
    print("TS(T1)=100, TS(T2)=150, TS(T3)=200")
    print("Initial: W-ts(A)=0, R-ts(A)=0, W-ts(B)=0, R-ts(B)=0")
    print()

    # State tracking
    w_ts = {"A": 0, "B": 0}
    r_ts = {"A": 0, "B": 0}
    ts = {"T1": 100, "T2": 150, "T3": 200}

    operations = [
        ("T2", "read", "A"),
        ("T3", "read", "A"),
        ("T1", "write", "A"),
        ("T2", "write", "A"),
        ("T3", "read", "B"),
        ("T2", "write", "B"),
        ("T1", "read", "B"),
    ]

    print("=== Basic Timestamp Ordering ===")
    print()

    # Reset state
    w_ts = {"A": 0, "B": 0}
    r_ts = {"A": 0, "B": 0}

    for txn, op, item in operations:
        txn_ts = ts[txn]
        print(f"  {txn} {op}({item}): TS({txn})={txn_ts}, W-ts({item})={w_ts[item]}, R-ts({item})={r_ts[item]}")

        if op == "read":
            if txn_ts < w_ts[item]:
                print(f"    REJECTED: TS({txn})={txn_ts} < W-ts({item})={w_ts[item]}. {txn} must abort.")
            else:
                r_ts[item] = max(r_ts[item], txn_ts)
                print(f"    ALLOWED. R-ts({item}) updated to {r_ts[item]}")
        else:  # write
            if txn_ts < r_ts[item]:
                print(f"    REJECTED: TS({txn})={txn_ts} < R-ts({item})={r_ts[item]}. {txn} must abort.")
            elif txn_ts < w_ts[item]:
                print(f"    REJECTED: TS({txn})={txn_ts} < W-ts({item})={w_ts[item]}. {txn} must abort.")
            else:
                w_ts[item] = txn_ts
                print(f"    ALLOWED. W-ts({item}) updated to {w_ts[item]}")
        print()

    # Thomas's Write Rule
    print("=== Thomas's Write Rule ===")
    print()

    w_ts = {"A": 0, "B": 0}
    r_ts = {"A": 0, "B": 0}

    for txn, op, item in operations:
        txn_ts = ts[txn]
        print(f"  {txn} {op}({item}): TS({txn})={txn_ts}, W-ts({item})={w_ts[item]}, R-ts({item})={r_ts[item]}")

        if op == "read":
            if txn_ts < w_ts[item]:
                print(f"    REJECTED: TS({txn})={txn_ts} < W-ts({item})={w_ts[item]}. {txn} must abort.")
            else:
                r_ts[item] = max(r_ts[item], txn_ts)
                print(f"    ALLOWED. R-ts({item}) updated to {r_ts[item]}")
        else:  # write
            if txn_ts < r_ts[item]:
                print(f"    REJECTED: TS({txn})={txn_ts} < R-ts({item})={r_ts[item]}. {txn} must abort.")
            elif txn_ts < w_ts[item]:
                # Thomas's Write Rule: IGNORE the write (it's obsolete)
                print(f"    IGNORED (Thomas's Write Rule): TS({txn})={txn_ts} < W-ts({item})={w_ts[item]}.")
                print(f"    A newer write already exists. This write is obsolete; skip it.")
            else:
                w_ts[item] = txn_ts
                print(f"    ALLOWED. W-ts({item}) updated to {w_ts[item]}")
        print()


# === Exercise 9: OCC Validation ===
# Problem: Validate transactions under Optimistic Concurrency Control.

def exercise_9():
    """OCC validation of three transactions."""
    print("Optimistic Concurrency Control Validation")
    print()

    txns = {
        "T1": {"read_set": {"A", "B"}, "write_set": {"A"}, "start": 0, "validate": 5},
        "T2": {"read_set": {"B", "C"}, "write_set": {"B"}, "start": 1, "validate": 6},
        "T3": {"read_set": {"A", "C"}, "write_set": {"C"}, "start": 2, "validate": 7},
    }

    for name, info in txns.items():
        print(f"  {name}: ReadSet={info['read_set']}, WriteSet={info['write_set']}, "
              f"Start=t={info['start']}, Validate=t={info['validate']}")
    print()

    def validate(ti_name, tj_name, txns):
        """Check if ti can validate against tj (tj validated before ti)."""
        ti = txns[ti_name]
        tj = txns[tj_name]

        # Condition: tj's WriteSet must not overlap with ti's ReadSet
        overlap = tj["write_set"] & ti["read_set"]

        print(f"  Validate {ti_name} against {tj_name}:")
        print(f"    {tj_name}.WriteSet = {tj['write_set']}")
        print(f"    {ti_name}.ReadSet = {ti['read_set']}")
        print(f"    Overlap: {overlap if overlap else 'empty set'}")

        if not overlap:
            print(f"    PASS: No conflict. {ti_name} can proceed.")
        else:
            print(f"    FAIL: {ti_name} may have read stale data. Must abort and restart.")
        return len(overlap) == 0

    # (a) Validate T2 against T1
    print("(a) Validate T2 against T1 (T1 validated at t=5, before T2 at t=6):")
    result_a = validate("T2", "T1", txns)
    print()

    # (b) Validate T3 against T1 and T2
    print("(b) Validate T3 against T1:")
    result_b1 = validate("T3", "T1", txns)
    print()
    print("    Validate T3 against T2:")
    result_b2 = validate("T3", "T2", txns)
    print()
    result_b = result_b1 and result_b2
    print(f"    T3 overall: {'PASS' if result_b else 'FAIL'}")
    print()

    # (c) Modified scenario
    print("(c) If T1.WriteSet = {B} instead of {A}:")
    txns_modified = {
        "T1": {"read_set": {"A", "B"}, "write_set": {"B"}, "start": 0, "validate": 5},
        "T2": {"read_set": {"B", "C"}, "write_set": {"B"}, "start": 1, "validate": 6},
        "T3": {"read_set": {"A", "C"}, "write_set": {"C"}, "start": 2, "validate": 7},
    }

    print("    Validate T2 against T1:")
    validate("T2", "T1", txns_modified)
    print()
    print("    Validate T3 against T1:")
    validate("T3", "T1", txns_modified)
    print()
    print("    Now T2 FAILS because T1 wrote B, which T2 reads.")
    print("    T3 PASSES against T1 because T3 doesn't read B.")


# === Exercise 10: Concurrency Control Scheme Selection ===
# Problem: Recommend a CC scheme for a banking application.

def exercise_10():
    """Recommend concurrency control scheme for banking workload."""
    print("Banking Application Workload:")
    print("  - Balance inquiry (read one account): 60%")
    print("  - Transfer (write two accounts): 30%")
    print("  - Month-end report (read all accounts): 10%")
    print()

    schemes = [
        {
            "name": "Strict 2PL",
            "pros": [
                "Strong consistency guarantees (serializability)",
                "Well-suited for write-heavy workloads with short transactions",
                "Good for transfers: lock both accounts, update, release"
            ],
            "cons": [
                "Month-end reports would block all transfers (and vice versa)",
                "Reader-writer contention: reports lock out writes for minutes",
                "Deadlock possible between concurrent transfers"
            ]
        },
        {
            "name": "MVCC (recommended)",
            "pros": [
                "Readers never block writers, writers never block readers",
                "Balance inquiries (60%) run without any locking",
                "Month-end reports see a consistent snapshot without blocking transfers",
                "Transfers proceed with row-level locking (minimal contention)"
            ],
            "cons": [
                "Storage overhead for version chain",
                "VACUUM needed (PostgreSQL) to clean old versions",
                "Write skew possible under Snapshot Isolation (need SSI for full serializability)"
            ]
        },
        {
            "name": "OCC",
            "pros": [
                "No lock overhead during execution",
                "Good if conflicts are rare"
            ],
            "cons": [
                "30% write transactions = high conflict rate -> many aborts",
                "Wasted work: transactions execute fully before validation",
                "Month-end reports would frequently abort (conflict with transfers)",
                "Poor fit for this workload"
            ]
        }
    ]

    for s in schemes:
        print(f"{s['name']}:")
        print("  Pros:")
        for p in s['pros']:
            print(f"    + {p}")
        print("  Cons:")
        for c in s['cons']:
            print(f"    - {c}")
        print()

    print("RECOMMENDATION: MVCC")
    print("  - 60% reads benefit from non-blocking snapshots")
    print("  - 30% transfers use row-level locks (low contention)")
    print("  - 10% reports get consistent snapshots without blocking anything")
    print("  - This is why PostgreSQL (MVCC) is the #1 choice for banking systems")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Shared vs Exclusive Locks ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: 2PL Variants ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: Wait-Die vs Wound-Wait ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 5: Lock Sequence with Deadlock Detection ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 7: Timestamp Ordering ===")
    print("=" * 70)
    exercise_7()

    print("=" * 70)
    print("=== Exercise 9: OCC Validation ===")
    print("=" * 70)
    exercise_9()

    print("=" * 70)
    print("=== Exercise 10: CC Scheme Selection ===")
    print("=" * 70)
    exercise_10()

    print("\nAll exercises completed!")
