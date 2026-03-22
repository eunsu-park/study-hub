"""
Exercises for Lesson 05: Operators and Composability
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Parse the Command ===
# Problem: For each command, identify operator, motion, count, and describe in English.

def exercise_1():
    """Solution: Parse Vim commands into operator + motion."""
    commands = [
        ("d3w",
         "operator=d (delete), motion=3w (3 words forward)",
         "Delete the next 3 words."),
        ("gU$",
         "operator=gU (uppercase), motion=$ (end of line)",
         "Uppercase everything from cursor to end of line."),
        (">G",
         "operator=> (indent right), motion=G (last line)",
         "Indent all lines from cursor to end of file."),
        ("ct;",
         "operator=c (change), motion=t; (till just before ;)",
         "Delete from cursor to just before the next ';', then enter Insert mode."),
        ("ygg",
         "operator=y (yank), motion=gg (first line)",
         "Copy everything from cursor to the beginning of the file."),
    ]
    # Why: Vim's grammar is [count] operator [count] motion.
    # Understanding this decomposition is the key to Vim fluency.
    for cmd, breakdown, english in commands:
        print(f"  {cmd:6s} -> {breakdown}")
        print(f"         English: {english}\n")


# === Exercise 2: Choose the Right Operator + Motion ===
# Problem: Given "var userName = getUserInput();" with cursor on 'v' of 'var'.

def exercise_2():
    """Solution: Choose the correct operator + motion for each task."""
    print("  Line: var userName = getUserInput();")
    print("  Cursor on 'v' of 'var'\n")

    tasks = [
        ("1. Delete 'var' and the space after it",
         "dw (or dW)",
         "dw deletes to next word start, including trailing space."),
        ("2. Change 'getUserInput' to something else",
         "ct( (with cursor on 'g' of getUserInput)",
         "Change till '(' -- deletes getUserInput, enters Insert mode, leaves () intact."),
        ("3. Copy from cursor to end of line",
         "y$",
         "Yank from cursor to end of line ($ motion)."),
        ("4. Make 'userName' uppercase (cursor on 'u')",
         "gUe (or gUiw for text object precision)",
         "gUe uppercases to end of word. gUiw uppercases the entire inner word."),
    ]
    # Why: The operator+motion grammar lets you express any editing intent concisely.
    # Each new motion you learn multiplies your available commands.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: Design for Dot-Repeatability ===
# Problem: Add a semicolon to the end of each line.

def exercise_3():
    """Solution: Dot-repeatable approach to append semicolons."""
    print("  Before:")
    for line in ["const a = 1", "const b = 2", "const c = 3", "const d = 4"]:
        print(f"    {line}")
    print()

    steps = [
        ("A", "Jump to end of line and enter Insert mode"),
        (";", "Type the semicolon"),
        ("Esc", "Return to Normal mode (records 'A;Esc' as the change)"),
        ("j", "Move to the next line"),
        (".", "Repeat the change (appends ; at end of line)"),
        ("j.", "Move down and repeat again"),
        ("j.", "Move down and repeat once more"),
    ]
    # Why: The dot command replays the entire "A ; Esc" operation.
    # Planning your edits around . is the heart of efficient Vim editing.
    print("  Step-by-step:")
    for cmd, explanation in steps:
        print(f"    {cmd:5s} -- {explanation}")
    print()
    print("  Alternative: :%s/$/;/  (substitution adds ; to every line at once)")
    print("  But the A;Esc + j. approach demonstrates the dot-repeat principle.")


# === Exercise 4: The Multiplication Effect ===
# Problem: Calculate operator x motion combinations.

def exercise_4():
    """Solution: The multiplicative power of operators and motions."""
    operators = ["d", "c", "y", ">", "gU"]
    motions = ["w", "e", "b", "$", "0", "gg", "G", "}", "f{char}"]

    n_ops = len(operators)
    n_motions = len(motions)
    total = n_ops * n_motions

    print(f"  Operators ({n_ops}): {', '.join(operators)}")
    print(f"  Motions ({n_motions}):   {', '.join(motions)}\n")

    # Why: Learning is multiplicative, not additive.
    # Each new operator gives M new commands; each new motion gives N new commands.
    print(f"  1. Total combinations: {n_ops} x {n_motions} = {total}")
    print()

    motions_new = motions + ["t{char}"]
    print(f"  2. After learning t{{char}} (new motion):")
    print(f"     {n_ops} operators x {len(motions_new)} motions = {n_ops * len(motions_new)}")
    print(f"     Increase: {n_ops} new commands (one per operator: dt{{c}}, ct{{c}}, yt{{c}}, >t{{c}}, gUt{{c}})")
    print()

    operators_new = operators + ["="]
    print(f"  3. After learning = (auto-indent, new operator):")
    print(f"     {len(operators_new)} operators x {len(motions_new)} motions = {len(operators_new) * len(motions_new)}")
    print(f"     Increase: {len(motions_new)} new commands (one per existing motion)")


# === Exercise 5: Complete Editing Task ===
# Problem: Lowercase all-caps words in three lines.

def exercise_5():
    """Solution: Transform ALL-CAPS words to lowercase."""
    print("  Before:")
    print("    hello WORLD")
    print("    This is a TEST line.")
    print("    foo BAR baz\n")

    print("  After:")
    print("    hello world")
    print("    This is a test line.")
    print("    foo bar baz\n")

    steps = [
        ("W", "Move to WORLD (next WORD)"),
        ("gue", "Lowercase to end of word -> 'world'"),
        ("j", "Move down to line 2"),
        ("fT", "Find 'T' of TEST on this line"),
        (".", "Repeat gue -> lowercase 'TEST' to 'test'"),
        ("j", "Move down to line 3"),
        ("W", "Move to BAR"),
        (".", "Repeat gue -> lowercase 'BAR' to 'bar'"),
    ]
    # Why: The dot command remembers 'gue' (lowercase to end of word) and
    # replays it at each new position. This is the change-move-repeat pattern.
    print("  Streamlined key sequence: W gue j fT . j W .\n")
    for cmd, explanation in steps:
        print(f"    {cmd:5s} -- {explanation}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Parse the Command", exercise_1),
        ("Exercise 2: Choose the Right Operator + Motion", exercise_2),
        ("Exercise 3: Design for Dot-Repeatability", exercise_3),
        ("Exercise 4: The Multiplication Effect", exercise_4),
        ("Exercise 5: Complete Editing Task", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
