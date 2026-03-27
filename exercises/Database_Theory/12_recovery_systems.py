"""
Exercises for Lesson 12: Recovery Systems
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers failure classification, log-based recovery, ARIES algorithm,
steal/force policies, and backup/PITR.
"""


# === Exercise 1: Failure Classification ===
# Problem: Classify failures and describe recovery actions.

def exercise_1():
    """Classify failures and describe recovery actions."""
    failures = [
        {
            "failure": "(a) A transaction divides by zero",
            "type": "Transaction failure (logical error)",
            "recovery": "UNDO the transaction: rollback all changes using the log. "
                       "The error is local to one transaction. Other transactions are unaffected."
        },
        {
            "failure": "(b) A power outage occurs",
            "type": "System failure (system crash)",
            "recovery": "On restart: REDO committed transactions whose changes may not have reached disk. "
                       "UNDO uncommitted transactions. Use WAL log for both."
        },
        {
            "failure": "(c) A disk head crashes, destroying the data disk",
            "type": "Media failure (disk failure)",
            "recovery": "Restore from latest backup + replay WAL/archived logs up to the point of failure. "
                       "Requires backup infrastructure (base backup + continuous archiving)."
        },
        {
            "failure": "(d) A deadlock is detected",
            "type": "Transaction failure (system-initiated abort)",
            "recovery": "UNDO the chosen victim transaction (rollback using log). "
                       "The victim is restarted automatically. Other transactions in the deadlock continue."
        },
        {
            "failure": "(e) The operating system kernel panics",
            "type": "System failure (OS crash)",
            "recovery": "Same as power outage: on restart, perform crash recovery (REDO + UNDO). "
                       "Contents of buffer pool are lost; stable storage (disk + logs) is preserved."
        }
    ]

    for f in failures:
        print(f"{f['failure']}")
        print(f"  Type: {f['type']}")
        print(f"  Recovery: {f['recovery']}")
        print()


# === Exercise 4: Log-Based Recovery ===
# Problem: Perform recovery from a log without checkpoints.

def exercise_4():
    """Log-based recovery: redo/undo analysis."""
    log = [
        (1,  "T1", "start", None, None),
        (2,  "T1", "write", "A", (10, 20)),
        (3,  "T2", "start", None, None),
        (4,  "T2", "write", "B", (30, 40)),
        (5,  "T1", "write", "C", (50, 60)),
        (6,  "T1", "commit", None, None),
        (7,  "T2", "write", "D", (70, 80)),
        (8,  "T3", "start", None, None),
        (9,  "T3", "write", "A", (20, 30)),
        (10, "T3", "commit", None, None),
        # CRASH
    ]

    print("Log (no checkpoints):")
    print(f"  {'LSN':<4} {'Record':<40}")
    print(f"  {'-'*4} {'-'*40}")
    for lsn, txn, op, item, vals in log:
        if op == "start":
            print(f"  {lsn:<4} <{txn}, start>")
        elif op == "commit":
            print(f"  {lsn:<4} <{txn}, commit>")
        elif op == "write":
            old_val, new_val = vals
            print(f"  {lsn:<4} <{txn}, {item}, {old_val}, {new_val}>")
    print(f"  {'':4} <- CRASH")
    print()

    # (a) Identify redo and undo sets
    committed = set()
    active = set()
    for lsn, txn, op, item, vals in log:
        if op == "start":
            active.add(txn)
        elif op == "commit":
            active.discard(txn)
            committed.add(txn)

    redo_set = committed  # Must redo committed transactions
    undo_set = active     # Must undo active (uncommitted) transactions

    print("(a) Redo and Undo sets:")
    print(f"    REDO (committed): {sorted(redo_set)}")
    print(f"    UNDO (active at crash): {sorted(undo_set)}")
    print()

    # (b) Recovery process
    print("(b) Recovery process:")
    print()

    # Redo phase (forward scan)
    values = {}  # Track final values
    print("  REDO phase (forward scan, LSN 1 to 10):")
    for lsn, txn, op, item, vals in log:
        if op == "write":
            old_val, new_val = vals
            if txn in redo_set:
                values[item] = new_val
                print(f"    LSN {lsn}: REDO {txn} write({item}): {old_val} -> {new_val}")
            else:
                # Also redo uncommitted for ARIES "repeat history"
                values[item] = new_val
                print(f"    LSN {lsn}: REDO (repeat history) {txn} write({item}): {old_val} -> {new_val}")
    print()

    # Undo phase (backward scan)
    print("  UNDO phase (backward scan, undo uncommitted):")
    for lsn, txn, op, item, vals in reversed(log):
        if op == "write" and txn in undo_set:
            old_val, new_val = vals
            values[item] = old_val
            print(f"    LSN {lsn}: UNDO {txn} write({item}): {new_val} -> {old_val}")
    print()

    # (c) Final values
    print("(c) Final values after recovery:")
    # A: T1 wrote 10->20, T3 wrote 20->30 (both committed). Final = 30
    # B: T2 wrote 30->40 (uncommitted). Undo -> 30
    # C: T1 wrote 50->60 (committed). Final = 60
    # D: T2 wrote 70->80 (uncommitted). Undo -> 70
    expected = {"A": 30, "B": 30, "C": 60, "D": 70}
    for item in sorted(expected.keys()):
        print(f"    {item} = {expected[item]}")


# === Exercise 5: Recovery with Checkpoint ===
# Problem: Repeat Exercise 4 with a checkpoint.

def exercise_5():
    """Recovery with checkpoint."""
    print("Same log as Exercise 4, but with checkpoint between LSN 6 and 7:")
    print("  <checkpoint {T2}>  (T2 is active at checkpoint time)")
    print()

    print("Impact of checkpoint on recovery:")
    print("  The checkpoint says: at this point, T2 was active, and all data pages")
    print("  modified by T1 before the checkpoint have been flushed to disk.")
    print()
    print("  REDO phase starts from the checkpoint (not from LSN 1):")
    print("    - Skip LSN 1-6 (T1's writes are already on disk; T1 committed before checkpoint)")
    print("    - Start REDO from LSN 7")
    print()
    print("  UNDO phase: T2 was active at checkpoint. Scan backward from crash.")
    print("    - Undo LSN 7: T2 write(D): 80 -> 70")
    print("    - Undo LSN 4: T2 write(B): 40 -> 30")
    print()
    print("  Final values: same as Exercise 4 (A=30, B=30, C=60, D=70)")
    print("  But recovery is FASTER because we skip LSN 1-6 in the redo phase.")


# === Exercise 6: ARIES Recovery ===
# Problem: Perform full ARIES recovery (Analysis, Redo, Undo).

def exercise_6():
    """ARIES recovery algorithm simulation."""
    print("ARIES Recovery Simulation")
    print()

    # Log
    log = [
        (10, "T1", "start",      None, None,       None),
        (20, "T1", "write",      "P1", ("X", 5, 10), 10),
        (30, "T2", "start",      None, None,       None),
        (40, "T2", "write",      "P2", ("Y", 15, 25), 30),
        (50, None, "begin_chkpt", None, None,       None),
        (55, None, "end_chkpt",  None, None,       None),  # ATT={T1(20), T2(40)}, DPT={P1:20, P2:40}
        (60, "T1", "write",      "P3", ("Z", 35, 45), 20),
        (70, "T2", "commit",     None, None,       None),
        (80, "T3", "start",      None, None,       None),
        (90, "T3", "write",      "P1", ("X", 10, 20), 80),
        (100,"T1", "write",      "P2", ("W", 50, 60), 60),
        # CRASH
    ]

    print("Log:")
    for lsn, txn, op, page, vals, prev_lsn in log:
        if op == "start":
            print(f"  LSN {lsn:>3}: <{txn}, start>")
        elif op == "commit":
            print(f"  LSN {lsn:>3}: <{txn}, commit>")
        elif op == "write":
            var, old, new = vals
            print(f"  LSN {lsn:>3}: <{txn}, {page}, {var}: {old}->{new}>  (prevLSN={prev_lsn})")
        elif op == "begin_chkpt":
            print(f"  LSN {lsn:>3}: <begin_checkpoint>")
        elif op == "end_chkpt":
            print(f"  LSN {lsn:>3}: <end_checkpoint ATT={{T1(20),T2(40)}}, DPT={{P1:20, P2:40}}>")
    print(f"  {'':>7} <- CRASH")
    print()

    # (a) Analysis Phase
    print("(a) ANALYSIS PHASE (scan forward from end_checkpoint at LSN 55)")
    print()

    # Initialize from checkpoint
    att = {"T1": 20, "T2": 40}  # txn -> lastLSN
    dpt = {"P1": 20, "P2": 40}  # page -> recLSN

    # Scan from LSN 60 onwards
    analysis_log = [(60, "T1", "write", "P3"), (70, "T2", "commit", None),
                    (80, "T3", "start", None), (90, "T3", "write", "P1"),
                    (100, "T1", "write", "P2")]

    for lsn, txn, op, page in analysis_log:
        if op == "start":
            att[txn] = lsn
            print(f"  LSN {lsn}: {txn} start -> add to ATT")
        elif op == "commit":
            del att[txn]
            print(f"  LSN {lsn}: {txn} commit -> remove from ATT")
        elif op == "write":
            att[txn] = lsn
            if page not in dpt:
                dpt[page] = lsn
            print(f"  LSN {lsn}: {txn} write {page} -> update ATT[{txn}]={lsn}" +
                  (f", add DPT[{page}]={lsn}" if dpt[page] == lsn else ""))
    print()

    print(f"  Final ATT (Active Transaction Table): {att}")
    print(f"  Final DPT (Dirty Page Table): {dpt}")
    print()

    # (b) Redo starting LSN
    redo_start = min(dpt.values())
    print(f"(b) Redo phase starts from: min(DPT recLSN) = LSN {redo_start}")
    print()

    # (c) Redo Phase
    print("(c) REDO PHASE (scan forward from LSN 20)")
    print("    Redo all log records where page is in DPT and LSN >= recLSN:")
    redo_records = [
        (20, "T1", "P1", "X: 5->10", True),
        (40, "T2", "P2", "Y: 15->25", True),
        (60, "T1", "P3", "Z: 35->45", True),
        (90, "T3", "P1", "X: 10->20", True),
        (100,"T1", "P2", "W: 50->60", True),
    ]
    for lsn, txn, page, change, redo in redo_records:
        status = "REDO" if redo else "skip (page not dirty or LSN < recLSN)"
        print(f"    LSN {lsn:>3}: {txn} write {page} ({change}) -> {status}")
    print()

    # (d) Undo Phase
    print("(d) UNDO PHASE (scan backward, undo uncommitted: T1 and T3)")
    print()
    print("    ToUndo = {LSN 100 (T1), LSN 90 (T3)} (lastLSN of each loser)")
    print()

    undo_steps = [
        {
            "lsn": 100, "txn": "T1", "action": "Undo T1 write P2 (W: 60->50)",
            "clr": "Write CLR: <T1, P2, W: 60->50, undoNextLSN=60>",
            "to_undo": "{LSN 90 (T3), LSN 60 (T1)}"
        },
        {
            "lsn": 90, "txn": "T3", "action": "Undo T3 write P1 (X: 20->10)",
            "clr": "Write CLR: <T3, P1, X: 20->10, undoNextLSN=80>",
            "to_undo": "{LSN 80 (T3), LSN 60 (T1)}"
        },
        {
            "lsn": 80, "txn": "T3", "action": "T3 start record -> T3 fully undone",
            "clr": "Write <T3, abort>",
            "to_undo": "{LSN 60 (T1)}"
        },
        {
            "lsn": 60, "txn": "T1", "action": "Undo T1 write P3 (Z: 45->35)",
            "clr": "Write CLR: <T1, P3, Z: 45->35, undoNextLSN=20>",
            "to_undo": "{LSN 20 (T1)}"
        },
        {
            "lsn": 20, "txn": "T1", "action": "Undo T1 write P1 (X: 10->5)",
            "clr": "Write CLR: <T1, P1, X: 10->5, undoNextLSN=10>",
            "to_undo": "{LSN 10 (T1)}"
        },
        {
            "lsn": 10, "txn": "T1", "action": "T1 start record -> T1 fully undone",
            "clr": "Write <T1, abort>",
            "to_undo": "{} (empty -> DONE)"
        },
    ]

    for step in undo_steps:
        print(f"    Process LSN {step['lsn']}: {step['action']}")
        print(f"      {step['clr']}")
        print(f"      ToUndo = {step['to_undo']}")
        print()

    # (e) Final values
    print("(e) Final values after recovery:")
    # X: T1 wrote 5->10, T3 wrote 10->20. Both undone. X = 5.
    # Wait: T2 committed and wrote Y:15->25. Y = 25.
    # Z: T1 wrote 35->45. Undone. Z = 35.
    # W: T1 wrote 50->60. Undone. W = 50.
    final_values = {
        "X": (5, "T1 (5->10) and T3 (10->20) both undone"),
        "Y": (25, "T2 committed (15->25)"),
        "Z": (35, "T1 undone (45->35)"),
        "W": (50, "T1 undone (60->50)")
    }
    for var in sorted(final_values.keys()):
        val, reason = final_values[var]
        print(f"    {var} = {val}  ({reason})")


# === Exercise 9: Steal/Force Policy Analysis ===
# Problem: Analyze steal/no-steal and force/no-force combinations.

def exercise_9():
    """Analyze steal/force policy combinations."""
    policies = [
        {
            "policy": "No-Steal / Force",
            "undo_needed": False,
            "redo_needed": False,
            "undo_reason": "No dirty pages from uncommitted txns on disk (no-steal) -> nothing to undo",
            "redo_reason": "All committed changes forced to disk at commit (force) -> nothing to redo",
            "performance": "Poor commit latency (force all pages), excellent buffer utilization is limited",
            "example": "Simple embedded databases, early systems"
        },
        {
            "policy": "No-Steal / No-Force",
            "undo_needed": False,
            "redo_needed": True,
            "undo_reason": "No dirty pages from uncommitted txns on disk (no-steal)",
            "redo_reason": "Committed changes may only be in buffer pool, not disk (no-force) -> must redo",
            "performance": "Fast commits (only log flush), but buffer pool pressure (no-steal means pinning pages)",
            "example": "Some in-memory databases"
        },
        {
            "policy": "Steal / Force",
            "undo_needed": True,
            "redo_needed": False,
            "undo_reason": "Dirty pages from uncommitted txns may be on disk (steal) -> must undo",
            "redo_reason": "All committed changes forced to disk (force) -> nothing to redo",
            "performance": "Poor commit latency, good buffer flexibility",
            "example": "Rarely used (worst of both worlds for performance)"
        },
        {
            "policy": "Steal / No-Force (ARIES)",
            "undo_needed": True,
            "redo_needed": True,
            "undo_reason": "Dirty pages from uncommitted txns may be on disk (steal)",
            "redo_reason": "Committed changes may not be on disk (no-force)",
            "performance": "Best runtime: fast commits (log flush only) + full buffer flexibility",
            "example": "PostgreSQL, MySQL InnoDB, Oracle, SQL Server -- virtually all modern DBMS"
        }
    ]

    print(f"  {'Policy':<25} {'UNDO?':<8} {'REDO?':<8} {'Performance':<50}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*50}")
    for p in policies:
        undo = "YES" if p["undo_needed"] else "NO"
        redo = "YES" if p["redo_needed"] else "NO"
        print(f"  {p['policy']:<25} {undo:<8} {redo:<8} {p['performance'][:50]}")
    print()

    for p in policies:
        print(f"  {p['policy']}:")
        print(f"    UNDO: {p['undo_reason']}")
        print(f"    REDO: {p['redo_reason']}")
        print(f"    Example: {p['example']}")
        print()


# === Exercise 10: Force vs No-Force Cost Analysis ===
# Problem: Calculate commit latency and I/O bandwidth.

def exercise_10():
    """Force vs no-force cost comparison."""
    n_pages = 1000
    page_flush_ms = 5     # random I/O
    log_flush_ms = 2      # sequential I/O
    txns_per_sec = 100

    print(f"Given: Transaction modifies {n_pages} pages")
    print(f"  Page flush: {page_flush_ms}ms (random I/O)")
    print(f"  Log flush: {log_flush_ms}ms (sequential I/O, ~10 KB)")
    print(f"  Throughput: {txns_per_sec} transactions/second")
    print()

    # (a) Commit latency
    force_latency = n_pages * page_flush_ms
    no_force_latency = log_flush_ms

    print(f"(a) Commit latency:")
    print(f"    Force: {n_pages} pages x {page_flush_ms}ms = {force_latency:,}ms ({force_latency/1000:.1f}s)")
    print(f"    No-force: 1 log flush = {no_force_latency}ms")
    print(f"    Speedup: {force_latency / no_force_latency:.0f}x")
    print()

    # (b) I/O bandwidth
    page_size_kb = 8  # typical page size
    force_io_per_sec = txns_per_sec * n_pages * page_size_kb / 1024  # MB/s
    no_force_io_per_sec = txns_per_sec * 10 / 1024  # 10 KB log per txn, MB/s

    print(f"(b) I/O bandwidth at {txns_per_sec} txns/sec:")
    print(f"    Force: {txns_per_sec} x {n_pages} x {page_size_kb}KB = {force_io_per_sec:.0f} MB/s")
    print(f"    No-force: {txns_per_sec} x 10KB = {no_force_io_per_sec:.2f} MB/s")
    print(f"    Ratio: {force_io_per_sec / no_force_io_per_sec:.0f}x more I/O with force")
    print()

    # (c) Why no-force?
    print("(c) Why virtually all modern DBMS use no-force:")
    print("    1. Commit latency: 2ms vs 5,000ms -- unacceptable for interactive workloads")
    print("    2. I/O bandwidth: force requires ~800 MB/s vs ~1 MB/s")
    print("    3. Sequential log writes are 100x faster than random page writes")
    print("    4. Dirty pages are flushed lazily by background checkpoint process")
    print("    5. The cost of redo during recovery is rare and amortized over long uptime")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Failure Classification ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 4: Log-Based Recovery ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Recovery with Checkpoint ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: ARIES Recovery ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 9: Steal/Force Policy Analysis ===")
    print("=" * 70)
    exercise_9()

    print("=" * 70)
    print("=== Exercise 10: Force vs No-Force Cost ===")
    print("=" * 70)
    exercise_10()

    print("\nAll exercises completed!")
