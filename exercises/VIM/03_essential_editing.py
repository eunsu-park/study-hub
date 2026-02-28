"""
Exercises for Lesson 03: Essential Editing
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Insert Mode Entry Points ===
# Problem: Given "function greet(name) {" with cursor on 'g' of 'greet',
# write the single key for each task.

def exercise_1():
    """Solution: Choose the correct Insert mode entry command."""
    print("  Line: function greet(name) {")
    print("  Cursor on 'g' of 'greet'\n")

    tasks = [
        ("1. Add '// helper' on the line above",
         "O",
         "Opens a new line ABOVE and enters Insert mode. Type '// helper'."),
        ("2. Add 'return ' at the very start of the line",
         "I",
         "Moves cursor to the beginning of the line and enters Insert mode."),
        ("3. Add '  // end' on a new blank line below",
         "o",
         "Opens a new line BELOW and enters Insert mode."),
        ("4. Add ' async' after the word 'function'",
         "a (after navigating with b then e)",
         "b moves back to 'function', e goes to end of word, a appends after cursor."),
    ]
    # Why: Each Insert entry command (i, a, I, A, o, O) positions the cursor
    # differently, eliminating the need for separate navigate-then-insert steps.
    for task, key, explanation in tasks:
        print(f"  {task}")
        print(f"    Key: {key}")
        print(f"    Why: {explanation}\n")


# === Exercise 2: Delete Operations ===
# Problem: Perform deletions on import lines (cursor on first char of line 2).

def exercise_2():
    """Solution: Delete operations on import statements."""
    print("  Text:")
    print("    Line 1: import os")
    print("    Line 2: import sys    <- cursor here")
    print("    Line 3: import json")
    print("    Line 4: import re\n")

    operations = [
        ("1. Delete the entire line 2",
         "dd",
         "Deletes the entire current line. The deleted text goes to a register."),
        ("2. Delete just the word 'sys' (cursor on 's' of 'sys')",
         "de (or dw)",
         "de deletes to end of current word. dw deletes to start of next word (includes trailing space)."),
        ("3. Delete lines 2, 3, and 4 in one command",
         "3dd",
         "Count (3) + dd: deletes 3 lines starting from the current line."),
    ]
    # Why: Vim's delete commands work at different granularities (char, word, line)
    # and all accept counts for efficiency.
    for task, cmd, explanation in operations:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: Yank, Put, and the Dot Command ===
# Problem: Duplicate 'console.log("debug");' five times below itself.

def exercise_3():
    """Solution: Most efficient way to duplicate a line 5 times."""
    print("  Line: console.log(\"debug\");\n")

    print("  Method 1 -- yank + paste + dot:")
    steps_1 = [
        ("yy", "Yank (copy) the current line"),
        ("p", "Paste below -- now 2 copies"),
        (".", "Repeat paste -- 3 copies"),
        (".", "Repeat paste -- 4 copies"),
        (".", "Repeat paste -- 5 copies"),
    ]
    # Why: The dot command (.) repeats the last change, making p repeatable.
    # Total: 6 keystrokes for 5 duplicates.
    for cmd, explanation in steps_1:
        print(f"    {cmd:5s} -- {explanation}")

    print()
    print("  Method 2 -- yank + counted paste:")
    steps_2 = [
        ("yy", "Yank the current line"),
        ("4p", "Paste 4 copies below (original + 4 = 5 total lines)"),
    ]
    for cmd, explanation in steps_2:
        print(f"    {cmd:5s} -- {explanation}")

    print()
    print("  Both methods produce 6 identical lines (1 original + 5 duplicates).")


# === Exercise 4: Change vs. Delete ===
# Problem: Compare dw+i+type vs cw+type on the word "old" in "The old house".

def exercise_4():
    """Solution: Why cw is preferred over dw + i."""
    print("  Line: The old house")
    print("  Cursor on 'o' of 'old'\n")

    print("  Sequence A: dw -> i -> type 'new' -> Esc")
    print("    Three separate actions. The dot command would only repeat")
    print("    the 'i + type new + Esc' part, NOT the delete.\n")

    print("  Sequence B: cw -> type 'new' -> Esc")
    print("    A single atomic change. The entire operation (delete word +")
    print("    insert 'new') is one undo unit and one dot-repeatable action.\n")

    # Why: cw creates a single repeatable, undoable operation.
    # If you need to change another 'old' to 'new', just press . after Sequence B.
    print("  Verdict: Sequence B (cw) is preferred because:")
    print("    1. Fewer keystrokes (one command vs three)")
    print("    2. Single undo unit (one 'u' undoes the whole change)")
    print("    3. Dot-repeatable (. replays the full change-word operation)")


# === Exercise 5: Undo Tree Exploration ===
# Problem: Trace the state through undo/redo operations.

def exercise_5():
    """Solution: Undo tree step-by-step trace."""
    steps = [
        ("1. Type 'first' in Insert, press Esc",
         "Buffer: 'first'",
         "First Insert session = one undo unit."),
        ("2. Press A, add ' second', press Esc",
         "Buffer: 'first second'",
         "Second Insert session = another undo unit."),
        ("3. Press u twice",
         "First u: 'first' (undoes ' second')\n"
         "                                        Second u: '' (empty -- undoes 'first')",
         "Each u undoes one Insert mode session."),
        ("4. Press Ctrl-r once",
         "Buffer: 'first'",
         "Redo restores the last undone change (the 'first' insert)."),
    ]
    # Why: Each enter-Insert -> type -> Esc cycle creates one undo unit.
    # This is why making smaller, more frequent trips to Normal mode gives
    # finer-grained undo points.
    for step, state, note in steps:
        print(f"  {step}")
        print(f"    State: {state}")
        print(f"    Note:  {note}\n")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Insert Mode Entry Points", exercise_1),
        ("Exercise 2: Delete Operations", exercise_2),
        ("Exercise 3: Yank, Put, and the Dot Command", exercise_3),
        ("Exercise 4: Change vs. Delete", exercise_4),
        ("Exercise 5: Undo Tree Exploration", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
