"""
ARIES Recovery Algorithm (Simplified)

Demonstrates:
- Write-Ahead Logging (WAL)
- Log record types (UPDATE, COMMIT, CHECKPOINT)
- ARIES recovery: Analysis → Redo → Undo
- Dirty page tracking

Theory:
- ARIES (Algorithms for Recovery and Isolation Exploiting Semantics)
  is the standard crash recovery algorithm for databases.
- WAL rule: log record must be written to disk before the
  corresponding data page is written.
- Recovery phases:
  1. Analysis: scan log from last checkpoint to find dirty pages
     and active transactions.
  2. Redo: replay all logged actions from the earliest dirty page LSN.
  3. Undo: rollback all uncommitted transactions.

Adapted from Database Theory Lesson 12.
"""

from dataclasses import dataclass, field
from enum import Enum


class LogType(Enum):
    UPDATE = "UPDATE"
    COMMIT = "COMMIT"
    ABORT = "ABORT"
    CHECKPOINT = "CHECKPOINT"
    CLR = "CLR"  # Compensation Log Record (undo)
    END = "END"


@dataclass
class LogRecord:
    lsn: int             # Log Sequence Number
    txn_id: str
    log_type: LogType
    page_id: str = ""
    old_value: str = ""
    new_value: str = ""
    prev_lsn: int = -1   # Previous LSN for this transaction
    undo_next: int = -1   # For CLR records

    def __str__(self) -> str:
        if self.log_type == LogType.UPDATE:
            return (f"LSN={self.lsn} [{self.txn_id}] UPDATE "
                    f"{self.page_id}: '{self.old_value}'→'{self.new_value}'")
        if self.log_type == LogType.CLR:
            return (f"LSN={self.lsn} [{self.txn_id}] CLR "
                    f"{self.page_id}: undo→'{self.old_value}'")
        if self.log_type == LogType.CHECKPOINT:
            return f"LSN={self.lsn} CHECKPOINT"
        return f"LSN={self.lsn} [{self.txn_id}] {self.log_type.value}"


class SimpleDB:
    """Simplified database with WAL-based recovery."""

    def __init__(self):
        self.log: list[LogRecord] = []
        self.next_lsn = 1
        self.pages: dict[str, str] = {}         # page_id → value
        self.flushed_pages: dict[str, str] = {}  # disk (survives crash)
        self.dirty_pages: dict[str, int] = {}    # page_id → recovery LSN
        self.txn_last_lsn: dict[str, int] = {}
        self.active_txns: set[str] = set()
        self.committed_txns: set[str] = set()

    def _write_log(self, txn_id: str, log_type: LogType,
                   page_id: str = "", old_val: str = "",
                   new_val: str = "", undo_next: int = -1) -> LogRecord:
        prev = self.txn_last_lsn.get(txn_id, -1)
        rec = LogRecord(self.next_lsn, txn_id, log_type, page_id,
                        old_val, new_val, prev, undo_next)
        self.log.append(rec)
        self.txn_last_lsn[txn_id] = self.next_lsn
        self.next_lsn += 1
        return rec

    def begin(self, txn_id: str) -> None:
        self.active_txns.add(txn_id)

    def update(self, txn_id: str, page_id: str, new_value: str) -> None:
        """Write a value (with WAL)."""
        old_value = self.pages.get(page_id, "")
        # WAL: write log FIRST
        rec = self._write_log(txn_id, LogType.UPDATE, page_id,
                              old_value, new_value)
        # Then update in-memory page
        self.pages[page_id] = new_value
        if page_id not in self.dirty_pages:
            self.dirty_pages[page_id] = rec.lsn

    def commit(self, txn_id: str) -> None:
        self._write_log(txn_id, LogType.COMMIT)
        # Flush dirty pages to disk
        for page_id in list(self.dirty_pages):
            self.flushed_pages[page_id] = self.pages[page_id]
        self.dirty_pages.clear()
        self.active_txns.discard(txn_id)
        self.committed_txns.add(txn_id)
        self._write_log(txn_id, LogType.END)

    def checkpoint(self) -> None:
        self._write_log("SYS", LogType.CHECKPOINT)

    def crash(self) -> None:
        """Simulate crash: lose in-memory state, keep log and flushed pages."""
        self.pages = dict(self.flushed_pages)  # Only flushed pages survive
        self.dirty_pages.clear()
        self.active_txns.clear()
        self.committed_txns.clear()
        self.txn_last_lsn.clear()

    def recover(self) -> dict:
        """ARIES recovery: Analysis → Redo → Undo."""
        result = {"analysis": {}, "redo": [], "undo": []}

        # ── Phase 1: Analysis ──────────────────────────────────
        # Find checkpoint, determine dirty pages and active transactions
        checkpoint_lsn = 0
        for rec in reversed(self.log):
            if rec.log_type == LogType.CHECKPOINT:
                checkpoint_lsn = rec.lsn
                break

        active = set()
        committed = set()
        dirty = {}

        for rec in self.log:
            if rec.lsn < checkpoint_lsn:
                continue
            if rec.txn_id == "SYS":
                continue

            active.add(rec.txn_id)

            if rec.log_type == LogType.UPDATE:
                if rec.page_id not in dirty:
                    dirty[rec.page_id] = rec.lsn
            elif rec.log_type == LogType.COMMIT:
                committed.add(rec.txn_id)
            elif rec.log_type == LogType.END:
                active.discard(rec.txn_id)

        losers = active - committed
        result["analysis"] = {
            "dirty_pages": dirty,
            "active_txns": active,
            "committed": committed,
            "losers": losers,
        }

        # ── Phase 2: Redo ──────────────────────────────────────
        # Replay all updates from earliest dirty page LSN
        min_lsn = min(dirty.values()) if dirty else 0

        for rec in self.log:
            if rec.lsn < min_lsn:
                continue
            if rec.log_type == LogType.UPDATE:
                self.pages[rec.page_id] = rec.new_value
                result["redo"].append(rec)

        # ── Phase 3: Undo ──────────────────────────────────────
        # Rollback loser transactions in reverse order
        for rec in reversed(self.log):
            if rec.txn_id in losers and rec.log_type == LogType.UPDATE:
                self.pages[rec.page_id] = rec.old_value
                # Write CLR
                self._write_log(rec.txn_id, LogType.CLR, rec.page_id,
                                rec.old_value, undo_next=rec.prev_lsn)
                result["undo"].append(rec)

        # End loser transactions
        for txn in losers:
            self._write_log(txn, LogType.ABORT)
            self._write_log(txn, LogType.END)

        return result


# ── Demos ──────────────────────────────────────────────────────────────

def demo_normal_recovery():
    print("=" * 60)
    print("ARIES RECOVERY: BASIC SCENARIO")
    print("=" * 60)

    db = SimpleDB()

    # Normal operations
    db.begin("T1")
    db.begin("T2")
    db.update("T1", "P1", "A")
    db.update("T2", "P2", "B")
    db.checkpoint()
    db.update("T1", "P1", "C")
    db.commit("T1")
    db.update("T2", "P2", "D")
    # T2 does NOT commit → crash

    print(f"\n  Log before crash:")
    for rec in db.log:
        print(f"    {rec}")

    print(f"\n  Pages before crash: {db.pages}")

    # Crash!
    db.crash()
    print(f"\n  === CRASH ===")
    print(f"  Pages after crash (disk only): {db.pages}")

    # Recovery
    result = db.recover()

    print(f"\n  --- Analysis Phase ---")
    a = result["analysis"]
    print(f"    Dirty pages: {a['dirty_pages']}")
    print(f"    Active txns: {a['active_txns']}")
    print(f"    Committed: {a['committed']}")
    print(f"    Losers (to undo): {a['losers']}")

    print(f"\n  --- Redo Phase ---")
    for rec in result["redo"]:
        print(f"    Redo: {rec}")

    print(f"\n  --- Undo Phase ---")
    for rec in result["undo"]:
        print(f"    Undo: {rec}")

    print(f"\n  Pages after recovery: {db.pages}")
    print(f"  T1 committed → P1='C' (preserved)")
    print(f"  T2 aborted   → P2='{db.pages.get('P2', '')}' (rolled back)")


def demo_wal_importance():
    print("\n" + "=" * 60)
    print("WHY WRITE-AHEAD LOGGING MATTERS")
    print("=" * 60)

    print(f"""
  Without WAL:
    1. Update page P1 in memory
    2. Crash before log write
    → Cannot redo: update is lost
    → Cannot undo: don't know what changed

  With WAL:
    1. Write log record (UPDATE P1: old→new)
    2. Update page P1 in memory
    3. Eventually flush P1 to disk
    → On crash after step 1: can redo from log
    → On crash, uncommitted: can undo from log

  WAL guarantees:
    - Durability: committed data survives crashes
    - Atomicity: uncommitted changes are rolled back""")


if __name__ == "__main__":
    demo_normal_recovery()
    demo_wal_importance()
