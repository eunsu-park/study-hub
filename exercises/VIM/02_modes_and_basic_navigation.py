"""
Exercises for Lesson 02: Modes and Basic Navigation
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Mode Identification ===
# Problem: For each scenario, identify which key(s) to press.

def exercise_1():
    """Solution: Identify the correct key(s) for each scenario."""
    scenarios = [
        ("1. Start typing at the current cursor position",
         "i",
         "Enters Insert mode with cursor before the current character."),
        ("2. Typed some text, want to go back to Normal mode",
         "Esc (or Ctrl-[)",
         "Returns to Normal mode from any other mode. Ctrl-[ is easier to reach."),
        ("3. Add a new line below the current line and start typing",
         "o",
         "Opens a new blank line below and enters Insert mode."),
        ("4. In Normal mode, save the file",
         ": w <Enter>",
         "The : enters Command-line mode, w writes the file, Enter submits."),
    ]
    # Why: Each command transitions between modes in a specific way.
    # Understanding mode transitions is the foundation of Vim fluency.
    for scenario, key, explanation in scenarios:
        print(f"  {scenario}")
        print(f"    Key: {key}")
        print(f"    Why: {explanation}\n")


# === Exercise 2: hjkl Navigation ===
# Problem: Write the key sequence for each movement using hjkl only.

def exercise_2():
    """Solution: hjkl navigation sequences."""
    movements = [
        ("1. Move down 7 lines",
         "7j",
         "Count (7) + direction (j = down). Counts prefix any motion."),
        ("2. Move left 3 characters",
         "3h",
         "Count (3) + direction (h = left)."),
        ("3. Move to the last line of the file",
         "G",
         "Capital G jumps to the last line. This is a file motion, not hjkl."),
        ("4. Move to the first line of the file",
         "gg",
         "Double lowercase g jumps to the first line."),
    ]
    # Why: The count+motion pattern is Vim's composable command language.
    # Every motion can be prefixed with a number to repeat it.
    for task, keys, explanation in movements:
        print(f"  {task}")
        print(f"    Keys: {keys}")
        print(f"    Note: {explanation}\n")


# === Exercise 3: Replace Mode vs. Delete and Insert ===
# Problem: Change "cat" to "dog" using two different methods.
# Line: The cat sat on the mat  (cursor on 'c' of 'cat')

def exercise_3():
    """Solution: Two ways to change 'cat' to 'dog'."""
    print("  Text: The cat sat on the mat")
    print("  Cursor is on 'c' of 'cat'\n")

    print("  Method A -- Replace mode (R):")
    method_a = [
        ("R", "Enter Replace mode (overwrite characters as you type)"),
        ("d o g", "Type 'dog' -- overwrites 'c', 'a', 't' one by one"),
        ("Esc", "Return to Normal mode"),
    ]
    for cmd, explanation in method_a:
        print(f"    {cmd:10s} -- {explanation}")

    print()
    print("  Method B -- Delete and Insert:")
    method_b = [
        ("3x (or xxx)", "Delete three characters ('cat')"),
        ("i", "Enter Insert mode"),
        ("d o g", "Type 'dog'"),
        ("Esc", "Return to Normal mode"),
    ]
    for cmd, explanation in method_b:
        print(f"    {cmd:15s} -- {explanation}")

    # Why: Replace mode is efficient when the replacement has the same length.
    # For different lengths, use the change command (cw) which you learn in Lesson 3.
    print()
    print("  Note: Method A is more concise when replacement length equals original.")
    print("  Even better: use 'cw' (change word) from Lesson 3 for maximum efficiency.")


# === Exercise 4: Save and Quit Variants ===
# Problem: Match each scenario to the correct command.

def exercise_4():
    """Solution: Save and quit variants."""
    scenarios = [
        ("1. Save changes and exit",
         ":wq (or ZZ in Normal mode)",
         "Write the file then quit. ZZ is the keyboard shortcut."),
        ("2. Discard changes and exit",
         ":q! (or ZQ in Normal mode)",
         "Force quit, discarding all unsaved changes."),
        ("3. Save but continue editing",
         ":w",
         "Write (save) without quitting. You stay in the editor."),
        ("4. No changes made, just close",
         ":q",
         "Quit cleanly. Works when there are no unsaved changes."),
    ]
    # Why: Vim distinguishes between write and quit as separate operations,
    # allowing flexible combinations (:w, :q, :wq, :q!).
    for scenario, cmd, explanation in scenarios:
        print(f"  {scenario}")
        print(f"    Command: {cmd}")
        print(f"    Reason:  {explanation}\n")


# === Exercise 5: The Mode Indicator Challenge ===
# Problem: Practice switching modes and observe the indicator.

def exercise_5():
    """Solution: Mode indicator observations at each step."""
    steps = [
        ("1. Open Vim",
         "(nothing / blank)",
         "Normal mode has no indicator by default."),
        ("2. Press i",
         "-- INSERT --",
         "The Insert mode indicator appears at the bottom left."),
        ("3. Press Esc",
         "(nothing / blank)",
         "Back to Normal mode -- the indicator disappears."),
        ("4. Press v",
         "-- VISUAL --",
         "Character-wise Visual mode is now active."),
        ("5. Press Esc then :",
         ": prompt at the bottom",
         "Esc returns to Normal, then : opens Command-line mode."),
    ]
    # Why: Reading the mode indicator is essential for mode awareness.
    # Beginners should always glance at the bottom-left to confirm their mode.
    for step, indicator, explanation in steps:
        print(f"  {step}")
        print(f"    Indicator: {indicator}")
        print(f"    Explanation: {explanation}\n")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Mode Identification", exercise_1),
        ("Exercise 2: hjkl Navigation", exercise_2),
        ("Exercise 3: Replace Mode vs. Delete and Insert", exercise_3),
        ("Exercise 4: Save and Quit Variants", exercise_4),
        ("Exercise 5: The Mode Indicator Challenge", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
