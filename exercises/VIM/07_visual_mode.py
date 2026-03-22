"""
Exercises for Lesson 07: Visual Mode
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Choose the Right Visual Mode ===
# Problem: For each scenario, state which Visual mode is most appropriate.

def exercise_1():
    """Solution: Choose v, V, or Ctrl-v for each scenario."""
    scenarios = [
        ("1. Add // to the beginning of 5 lines to comment them out",
         "Ctrl-v (Block Visual)",
         "Column insert: select column 1 across 5 lines, then I//Esc to prepend to all."),
        ("2. Select just the word 'error' inside a longer string",
         "v (Character-wise)",
         "Need an arbitrary character range within a line, not whole lines."),
        ("3. Delete 3 complete lines of a function",
         "V (Line-wise)",
         "Whole line operations are cleanest in line-wise mode: V2jd."),
        ("4. Remove 4 spaces of indentation from 6 lines simultaneously",
         "Ctrl-v (Block Visual)",
         "Select a 4-column x 6-row block at the start of lines, then press d."),
    ]
    # Why: Choosing the right Visual mode reduces keystrokes and avoids errors.
    # v = arbitrary ranges, V = whole lines, Ctrl-v = rectangular columns.
    for scenario, mode, explanation in scenarios:
        print(f"  {scenario}")
        print(f"    Mode: {mode}")
        print(f"    Why:  {explanation}\n")


# === Exercise 2: Block Visual Column Insert ===
# Problem: Transform a plain list into a markdown list using Ctrl-v.

def exercise_2():
    """Solution: Visual Block column insert to prepend '- ' to each line."""
    print("  Before:        After:")
    print("    apple          - apple")
    print("    banana         - banana")
    print("    cherry         - cherry")
    print("    date           - date\n")

    steps = [
        ("Place cursor on 'a' of 'apple' (column 1)", "Starting position"),
        ("Ctrl-v", "Enter Visual Block mode"),
        ("3j", "Extend selection down 3 lines (covers all 4 lines)"),
        ("I", "Capital I: enter Insert mode at the start of the block"),
        ("- ", "Type dash-space (the list marker)"),
        ("Esc", "Apply the insertion to ALL selected lines"),
    ]
    # Why: Visual Block mode applies the same column edit to every selected line.
    # This is unique to Vim and extremely powerful for structured data.
    print("  Key sequence:")
    for i, (cmd, explanation) in enumerate(steps, 1):
        print(f"    {i}. {cmd:50s} -- {explanation}")


# === Exercise 3: gv and the o Key ===
# Problem: Re-indent lines and adjust selection boundaries.

def exercise_3():
    """Solution: Using gv to reselect and o to switch selection ends."""
    print("  Main question: Already indented 4 lines with V3j>")
    print("  Need to indent one more level.\n")

    main = [
        ("gv", "Reselect the exact same 4-line visual selection"),
        (">", "Indent the reselected lines one more level"),
    ]
    # Why: gv saves time -- no need to re-navigate and re-select.
    # Much faster than repeating V3j> from scratch.
    print("  Solution: gv>")
    for cmd, explanation in main:
        print(f"    {cmd:5s} -- {explanation}")

    print()
    print("  Bonus: Selected too far in character visual mode?")
    bonus = [
        ("o", "Jump cursor to the OTHER end of the selection"),
        ("(then use b, h, etc.)", "Move that end inward to shrink the selection"),
    ]
    for cmd, explanation in bonus:
        print(f"    {cmd:30s} -- {explanation}")
    print()
    print("  The 'o' key toggles between selection endpoints without restarting.")


# === Exercise 4: Visual vs. Operator Efficiency ===
# Problem: Rewrite Visual mode operations as direct operator + text object.

def exercise_4():
    """Solution: Direct operators are shorter and dot-repeatable."""
    conversions = [
        ("viwd (select inner word, delete)",
         "diw",
         "3 keys vs 4"),
        ('vi"c (select inside quotes, change)',
         'ci"',
         "3 keys vs 4"),
        ("Vd (select current line, delete)",
         "dd",
         "2 keys vs 2 (same!)"),
        ("vi(y (select inside parens, yank)",
         "yi(",
         "3 keys vs 4"),
    ]
    # Why: Operator+text object is always at least as efficient (usually fewer keystrokes)
    # AND it is dot-repeatable, while Visual mode version is NOT.
    print(f"  {'Visual Mode':<25s} {'Operator Form':<15s} {'Keys Saved'}")
    print(f"  {'-' * 25} {'-' * 15} {'-' * 10}")
    for visual, operator, savings in conversions:
        print(f"  {visual:<25s} {operator:<15s} {savings}")
    print()
    print("  Key advantage: Operator form is dot-repeatable (.) but Visual is not.")


# === Exercise 5: Real-World Block Editing ===
# Problem: Remove the second column from a CSV using Visual Block mode.

def exercise_5():
    """Solution: Visual Block delete to remove a CSV column."""
    print("  Before:")
    print("    John,Smith,30,Engineer")
    print("    Jane,Doe,25,Designer")
    print("    Bob,Jones,35,Manager\n")

    steps = [
        ("Place cursor on ',' after 'John'", "The comma before 'Smith'"),
        ("Ctrl-v", "Enter Visual Block mode"),
        ("2j", "Extend block down to cover all 3 rows"),
        ("f,", "Extend to the next comma (selects ,Smith  ,Doe  ,Jones)"),
        ("d", "Delete the selected block"),
    ]
    # Why: Visual Block mode applies the same column range to all rows,
    # even when the text in each row has different content/lengths.
    print("  Key sequence:")
    for i, (cmd, explanation) in enumerate(steps, 1):
        print(f"    {i}. {cmd:45s} -- {explanation}")

    print()
    print("  After:")
    print("    John,30,Engineer")
    print("    Jane,25,Designer")
    print("    Bob,35,Manager")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Choose the Right Visual Mode", exercise_1),
        ("Exercise 2: Block Visual Column Insert", exercise_2),
        ("Exercise 3: gv and the o Key", exercise_3),
        ("Exercise 4: Visual vs. Operator Efficiency", exercise_4),
        ("Exercise 5: Real-World Block Editing", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
