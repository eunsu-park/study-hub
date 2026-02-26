"""
Turing Machine Simulator
==========================

Demonstrates:
- Standard single-tape Turing machine
- TM computation with tape visualization
- TM design for classic languages
- Multi-tape TM simulation on single tape
- Universal TM concept

Reference: Formal_Languages Lesson 9 — Turing Machines
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class Direction(Enum):
    LEFT = "L"
    RIGHT = "R"
    STAY = "S"


# Why: A distinct blank symbol separates "no data" from valid input characters.
# The tape is conceptually infinite in both directions, filled with blanks.
BLANK = "□"


@dataclass
class TuringMachine:
    """Single-tape deterministic Turing Machine."""
    states: Set[str]
    input_alphabet: Set[str]
    tape_alphabet: Set[str]
    transitions: Dict[Tuple[str, str], Tuple[str, str, Direction]]
    start: str
    accept: str
    reject: str

    def run(self, input_str: str, max_steps: int = 10000,
            trace: bool = False) -> Tuple[str, List[str], int]:
        """
        Run TM on input.

        Returns: (result, tape_contents, steps)
            result: "accept", "reject", or "timeout"
        """
        # Why: Using a dict instead of a list allows the tape to extend infinitely
        # in both directions — negative indices work naturally. Missing keys
        # default to BLANK, simulating the infinite blank tape.
        tape: Dict[int, str] = {}
        for i, ch in enumerate(input_str):
            tape[i] = ch

        head = 0
        state = self.start
        steps = 0
        trace_log: List[str] = []

        while steps < max_steps:
            if trace:
                trace_log.append(self._format_tape(tape, head, state))

            if state == self.accept:
                tape_result = self._read_tape(tape)
                return "accept", tape_result, steps

            if state == self.reject:
                tape_result = self._read_tape(tape)
                return "reject", tape_result, steps

            symbol = tape.get(head, BLANK)
            key = (state, symbol)

            if key not in self.transitions:
                # No transition = implicit reject
                tape_result = self._read_tape(tape)
                return "reject", tape_result, steps

            new_state, write_symbol, direction = self.transitions[key]
            tape[head] = write_symbol
            state = new_state

            if direction == Direction.LEFT:
                head -= 1
            elif direction == Direction.RIGHT:
                head += 1
            # STAY: don't move

            steps += 1

        return "timeout", self._read_tape(tape), steps

    def _format_tape(self, tape: Dict[int, str], head: int, state: str) -> str:
        """Format tape for display with head position marked."""
        if not tape:
            return f"  [{state}] □"

        min_pos = min(min(tape.keys()), head)
        max_pos = max(max(tape.keys()), head)

        cells = []
        for i in range(min_pos, max_pos + 1):
            ch = tape.get(i, BLANK)
            if i == head:
                cells.append(f"[{state}]{ch}")
            else:
                cells.append(ch)
        return "  " + " ".join(cells)

    def _read_tape(self, tape: Dict[int, str]) -> List[str]:
        """Read non-blank tape contents."""
        if not tape:
            return []
        min_pos = min(tape.keys())
        max_pos = max(tape.keys())
        result = []
        for i in range(min_pos, max_pos + 1):
            ch = tape.get(i, BLANK)
            result.append(ch)
        # Trim trailing blanks
        while result and result[-1] == BLANK:
            result.pop()
        while result and result[0] == BLANK:
            result.pop(0)
        return result


# ─────────────── TM Builders ───────────────

# Why: The "mark and sweep" strategy pairs each 'a' with a 'b' by replacing
# them with markers (X, Y). Each pass marks one pair and rewinds, giving O(n^2)
# time — a classic TM design pattern showing that TMs can count with markers.
def tm_anbn() -> TuringMachine:
    """TM for {a^n b^n | n >= 0}. Marks pairs of a's and b's."""
    return TuringMachine(
        states={"q0", "q1", "q2", "q3", "q4", "q_acc", "q_rej"},
        input_alphabet={"a", "b"},
        tape_alphabet={"a", "b", "X", "Y", BLANK},
        transitions={
            # q0: scan right for 'a' to mark
            ("q0", "a"): ("q1", "X", Direction.RIGHT),  # mark a as X
            ("q0", "Y"): ("q3", "Y", Direction.RIGHT),  # skip Y's
            ("q0", BLANK): ("q_acc", BLANK, Direction.STAY),  # no a's left = accept

            # q1: scan right past a's and Y's to find matching b
            ("q1", "a"): ("q1", "a", Direction.RIGHT),
            ("q1", "Y"): ("q1", "Y", Direction.RIGHT),
            ("q1", "b"): ("q2", "Y", Direction.LEFT),  # mark b as Y, go left

            # q2: scan left back to start
            ("q2", "a"): ("q2", "a", Direction.LEFT),
            ("q2", "Y"): ("q2", "Y", Direction.LEFT),
            ("q2", "X"): ("q0", "X", Direction.RIGHT),  # back to start of unmarked

            # q3: verify no unmarked b's remain
            ("q3", "Y"): ("q3", "Y", Direction.RIGHT),
            ("q3", BLANK): ("q_acc", BLANK, Direction.STAY),
        },
        start="q0",
        accept="q_acc",
        reject="q_rej",
    )


# Why: {a^n b^n c^n} is not context-free, proving TMs are strictly more powerful
# than PDAs. The strategy extends mark-and-sweep to three symbols (X, Y, Z),
# marking one triple per pass.
def tm_anbncn() -> TuringMachine:
    """TM for {a^n b^n c^n | n >= 1}. Marks triples."""
    return TuringMachine(
        states={"q0", "q1", "q2", "q3", "q4", "q5", "q_acc", "q_rej"},
        input_alphabet={"a", "b", "c"},
        tape_alphabet={"a", "b", "c", "X", "Y", "Z", BLANK},
        transitions={
            # q0: find leftmost 'a' and mark it
            ("q0", "a"): ("q1", "X", Direction.RIGHT),
            ("q0", "Y"): ("q4", "Y", Direction.RIGHT),  # all a's marked; check rest
            ("q0", BLANK): ("q_rej", BLANK, Direction.STAY),

            # q1: scan right past a's and Y's to find 'b'
            ("q1", "a"): ("q1", "a", Direction.RIGHT),
            ("q1", "Y"): ("q1", "Y", Direction.RIGHT),
            ("q1", "b"): ("q2", "Y", Direction.RIGHT),  # mark b as Y

            # q2: scan right past b's and Z's to find 'c'
            ("q2", "b"): ("q2", "b", Direction.RIGHT),
            ("q2", "Z"): ("q2", "Z", Direction.RIGHT),
            ("q2", "c"): ("q3", "Z", Direction.LEFT),   # mark c as Z

            # q3: scan left all the way back to the marked X
            ("q3", "a"): ("q3", "a", Direction.LEFT),
            ("q3", "b"): ("q3", "b", Direction.LEFT),
            ("q3", "Y"): ("q3", "Y", Direction.LEFT),
            ("q3", "Z"): ("q3", "Z", Direction.LEFT),
            ("q3", "X"): ("q0", "X", Direction.RIGHT),  # back to q0

            # q4: verify all b's and c's are marked
            ("q4", "Y"): ("q4", "Y", Direction.RIGHT),
            ("q4", "Z"): ("q5", "Z", Direction.RIGHT),
            ("q4", BLANK): ("q_rej", BLANK, Direction.STAY),

            ("q5", "Z"): ("q5", "Z", Direction.RIGHT),
            ("q5", BLANK): ("q_acc", BLANK, Direction.STAY),
        },
        start="q0",
        accept="q_acc",
        reject="q_rej",
    )


# Why: The "shrink from both ends" strategy erases the first symbol, scans right
# to check the last, then erases it and returns. Each pass reduces the string
# by 2, giving O(n^2) time — demonstrating TM tape head movement patterns.
def tm_palindrome() -> TuringMachine:
    """TM for binary palindromes. Compares first and last symbols."""
    return TuringMachine(
        states={"q0", "q0a", "q0b", "q1a", "q1b", "q2", "q_acc", "q_rej"},
        input_alphabet={"a", "b"},
        tape_alphabet={"a", "b", BLANK},
        transitions={
            # q0: read and erase leftmost symbol
            ("q0", "a"): ("q1a", BLANK, Direction.RIGHT),
            ("q0", "b"): ("q1b", BLANK, Direction.RIGHT),
            ("q0", BLANK): ("q_acc", BLANK, Direction.STAY),  # empty = palindrome

            # q1a: scan right to find rightmost symbol (should be 'a')
            ("q1a", "a"): ("q1a", "a", Direction.RIGHT),
            ("q1a", "b"): ("q1a", "b", Direction.RIGHT),
            ("q1a", BLANK): ("q0a", BLANK, Direction.LEFT),

            # q0a: check rightmost is 'a'
            ("q0a", "a"): ("q2", BLANK, Direction.LEFT),  # match!
            ("q0a", "b"): ("q_rej", "b", Direction.STAY),  # mismatch
            ("q0a", BLANK): ("q_acc", BLANK, Direction.STAY),  # single char

            # q1b: scan right to find rightmost symbol (should be 'b')
            ("q1b", "a"): ("q1b", "a", Direction.RIGHT),
            ("q1b", "b"): ("q1b", "b", Direction.RIGHT),
            ("q1b", BLANK): ("q0b", BLANK, Direction.LEFT),

            # q0b: check rightmost is 'b'
            ("q0b", "b"): ("q2", BLANK, Direction.LEFT),
            ("q0b", "a"): ("q_rej", "a", Direction.STAY),
            ("q0b", BLANK): ("q_acc", BLANK, Direction.STAY),

            # q2: scan left back to start
            ("q2", "a"): ("q2", "a", Direction.LEFT),
            ("q2", "b"): ("q2", "b", Direction.LEFT),
            ("q2", BLANK): ("q0", BLANK, Direction.RIGHT),
        },
        start="q0",
        accept="q_acc",
        reject="q_rej",
    )


# Why: Repeatedly halving tests whether the count is a power of 2. Each pass
# marks every other 'a', effectively dividing by 2. If the count is ever odd
# (except 1), we reject. This shows TMs can perform division-like arithmetic.
def tm_power_of_2() -> TuringMachine:
    """
    TM for {a^(2^n) | n >= 0}.
    Repeatedly halves the number of a's. If it reaches 1, accept.
    """
    return TuringMachine(
        states={"q0", "q1", "q2", "q3", "q4", "q_acc", "q_rej"},
        input_alphabet={"a"},
        tape_alphabet={"a", "X", BLANK},
        transitions={
            # q0: move right past X's; if single 'a', accept
            ("q0", "X"): ("q0", "X", Direction.RIGHT),
            ("q0", "a"): ("q1", "X", Direction.RIGHT),  # mark first a
            ("q0", BLANK): ("q_rej", BLANK, Direction.STAY),

            # q1: check if only one 'a' was there
            ("q1", BLANK): ("q_acc", BLANK, Direction.STAY),  # single a = 2^0
            ("q1", "a"): ("q2", "a", Direction.RIGHT),  # more to process
            ("q1", "X"): ("q1", "X", Direction.RIGHT),

            # q2: mark every other 'a' as 'X' (halving)
            ("q2", "a"): ("q3", "X", Direction.RIGHT),
            ("q2", "X"): ("q2", "X", Direction.RIGHT),
            ("q2", BLANK): ("q4", BLANK, Direction.LEFT),  # done with pass

            # q3: skip one 'a' (keep it)
            ("q3", "a"): ("q2", "a", Direction.RIGHT),
            ("q3", "X"): ("q3", "X", Direction.RIGHT),
            ("q3", BLANK): ("q_rej", BLANK, Direction.STAY),  # odd count: reject

            # q4: rewind to start
            ("q4", "a"): ("q4", "a", Direction.LEFT),
            ("q4", "X"): ("q4", "X", Direction.LEFT),
            ("q4", BLANK): ("q0", BLANK, Direction.RIGHT),
        },
        start="q0",
        accept="q_acc",
        reject="q_rej",
    )


# ─────────────── Demos ───────────────

def demo_anbn():
    """TM for {a^n b^n}."""
    print("=" * 60)
    print("Demo 1: TM for {a^n b^n | n >= 0}")
    print("=" * 60)

    tm = tm_anbn()
    tests = ["", "ab", "aabb", "aaabbb", "a", "b", "aab", "abb", "ba"]
    for s in tests:
        result, tape, steps = tm.run(s)
        display = s if s else "ε"
        print(f"  '{display}': {result} ({steps} steps)")

    # Show trace for "aabb"
    print("\n  Trace for 'aabb':")
    result, tape, steps = tm.run("aabb", trace=True)
    result2, tape2, steps2 = tm.run("aabb", trace=True)
    # Re-run with trace to show
    tm2 = tm_anbn()
    tape_dict: Dict[int, str] = {i: c for i, c in enumerate("aabb")}
    head = 0
    state = "q0"
    for step in range(steps + 1):
        line = tm2._format_tape(tape_dict, head, state)
        print(f"    Step {step}: {line}")
        if state in (tm2.accept, tm2.reject):
            break
        symbol = tape_dict.get(head, BLANK)
        key = (state, symbol)
        if key not in tm2.transitions:
            break
        new_state, write, direction = tm2.transitions[key]
        tape_dict[head] = write
        state = new_state
        if direction == Direction.LEFT:
            head -= 1
        elif direction == Direction.RIGHT:
            head += 1


def demo_anbncn():
    """TM for {a^n b^n c^n}."""
    print("\n" + "=" * 60)
    print("Demo 2: TM for {a^n b^n c^n | n >= 1}")
    print("=" * 60)

    tm = tm_anbncn()
    tests = ["abc", "aabbcc", "aaabbbccc", "ab", "abcc", "aabbc", "abcabc"]
    for s in tests:
        result, tape, steps = tm.run(s)
        # Expected
        i = 0
        while i < len(s) and s[i] == 'a': i += 1
        na = i
        while i < len(s) and s[i] == 'b': i += 1
        nb = i - na
        while i < len(s) and s[i] == 'c': i += 1
        nc = i - na - nb
        expected = i == len(s) and na == nb == nc and na >= 1
        status = "OK" if (result == "accept") == expected else "ERROR"
        print(f"  '{s}': {result} ({steps} steps) {status}")


def demo_palindrome():
    """TM for palindromes."""
    print("\n" + "=" * 60)
    print("Demo 3: TM for Binary Palindromes")
    print("=" * 60)

    tm = tm_palindrome()
    tests = ["a", "b", "aa", "ab", "aba", "abb", "abba", "abab", "aabaa", "abcba"]
    for s in tests:
        result, tape, steps = tm.run(s)
        expected = all(c in "ab" for c in s) and s == s[::-1]
        status = "OK" if (result == "accept") == expected else "ERROR"
        print(f"  '{s}': {result} ({steps} steps) {status}")


def demo_power_of_2():
    """TM for {a^(2^n) | n >= 0}."""
    print("\n" + "=" * 60)
    print("Demo 4: TM for {a^(2^n) | n >= 0}")
    print("=" * 60)

    tm = tm_power_of_2()
    for length in range(1, 20):
        s = "a" * length
        result, tape, steps = tm.run(s)
        is_power = (length & (length - 1)) == 0 and length > 0
        status = "OK" if (result == "accept") == is_power else "ERROR"
        marker = "← 2^" + str(length.bit_length() - 1) if is_power else ""
        print(f"  a^{length:2d}: {result:7s} ({steps:4d} steps) {status} {marker}")


def demo_tm_complexity():
    """Show time complexity of TMs on different inputs."""
    print("\n" + "=" * 60)
    print("Demo 5: TM Time Complexity")
    print("=" * 60)

    tm = tm_anbn()
    print("  {a^n b^n} TM step count vs n:")
    print(f"  {'n':>4} {'steps':>8} {'~n^2':>8}")
    for n in [1, 2, 4, 8, 16, 32]:
        s = "a" * n + "b" * n
        _, _, steps = tm.run(s)
        print(f"  {n:4d} {steps:8d} {n*n:8d}")


if __name__ == "__main__":
    demo_anbn()
    demo_anbncn()
    demo_palindrome()
    demo_power_of_2()
    demo_tm_complexity()
