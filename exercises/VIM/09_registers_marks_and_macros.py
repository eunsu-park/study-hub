"""
Exercises for Lesson 09: Registers, Marks, and Macros
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: The Unnamed Register Problem ===
# Problem: What gets pasted after yy, then dd, then p?

def exercise_1():
    """Solution: Understanding the unnamed vs yank register."""
    steps = [
        ("yy", 'Yank a line -> stored in "" (unnamed) and "0 (yank register)'),
        ("(move down)", "Navigate to another line"),
        ("dd", 'Delete a line -> OVERWRITES "" with deleted content, but "0 is unchanged'),
        ("p", "Pastes from unnamed register -> THE DELETED LINE (not the yanked one!)"),
    ]
    # Why: dd writes to "" (unnamed), overwriting the yanked content.
    # But "0 (yank register) is only updated by yank commands, never by deletes.
    print("  Scenario trace:")
    for cmd, explanation in steps:
        print(f"    {cmd:15s} -- {explanation}")

    print()
    print('  What gets pasted? The DELETED line (from "").')
    print('  How to paste the yanked line? Use "0p')
    print()
    print('  Key insight: "0 (yank register) preserves your last yank even after deletes.')
    print("  This is one of the most common Vim surprises for new users.")


# === Exercise 2: Multi-Register Copy/Paste ===
# Problem: Copy three separate snippets and paste them later.

def exercise_2():
    """Solution: Named registers for multi-clipboard workflow."""
    print("  Copying to named registers:\n")

    copy_steps = [
        ('"ayy', "Navigate to snippet 1, yank into register a"),
        ('"byy', "Navigate to snippet 2, yank into register b"),
        ('"cyy', "Navigate to snippet 3, yank into register c"),
    ]
    for cmd, explanation in copy_steps:
        print(f"    {cmd:8s} -- {explanation}")

    print()
    print("  Verify contents:")
    print("    :reg a b c  -- Inspect all three registers")

    print()
    print("  Paste them later (in any order, at any location):")
    paste_steps = [
        ('"ap', "Paste the function signature from register a"),
        ('"bp', "Paste the import statement from register b"),
        ('"cp', "Paste the configuration constant from register c"),
    ]
    # Why: Named registers (a-z) persist for the entire Vim session.
    # They act as 26 independent clipboards, completely avoiding the
    # unnamed register overwrite problem.
    for cmd, explanation in paste_steps:
        print(f"    {cmd:8s} -- {explanation}")
    print()
    print("  Named registers persist for the entire session -- paste anywhere, anytime.")


# === Exercise 3: Record and Apply a Macro ===
# Problem: Transform each line from "apple" to '- "apple",' using a macro.

def exercise_3():
    """Solution: Record a formatting macro and apply it."""
    print("  Before:           After:")
    before = ["apple", "banana", "cherry", "date", "elderberry"]
    after = ['- "apple",', '- "banana",', '- "cherry",', '- "date",', '- "elderberry",']
    for b, a in zip(before, after):
        print(f"    {b:15s}     {a}")
    print()

    print("  Recording the macro in register q:\n")
    macro_steps = [
        ("qq", "Start recording into register q"),
        ("0", "Go to start of line (predictable position)"),
        ('I- "', 'Insert at line start: - "'),
        ("Esc", "Return to Normal mode"),
        ('A",', 'Append at line end: ",'),
        ("Esc", "Return to Normal mode"),
        ("j", "Move to next line (advance for next iteration)"),
        ("q", "Stop recording"),
    ]
    # Why: A reliable macro starts at a predictable position (0 or ^),
    # performs the edit, and advances to the next target (j).
    for cmd, explanation in macro_steps:
        print(f"    {cmd:8s} -- {explanation}")

    print()
    print("  Applying the macro:")
    print("    @q    -- Apply to current line (cursor auto-advances to next)")
    print("    4@q   -- Apply to remaining 4 lines")
    print()
    print("  Or position on line 1 first and run the macro, then 4@q for the rest.")


# === Exercise 4: Black Hole and Yank Register ===
# Problem: Copy a line, delete unwanted lines without losing the copy, then paste.

def exercise_4():
    """Solution: Using the black hole register to preserve yanked content."""
    steps = [
        ("Step 1: Copy the important line",
         "yy",
         'Yanks into "" (unnamed) and "0 (yank register).'),
        ("Step 2: Delete unwanted lines WITHOUT overwriting the copy",
         '"_dd',
         'Sends deleted text to "_ (black hole) instead of "". Your yank is safe.'),
        ("Step 3: Paste the originally copied line",
         'p (or "0p)',
         'Since "_dd did not overwrite "", plain p still pastes the yanked content.'),
    ]
    # Why: The black hole register ("_) is a write-only sink.
    # Anything sent there is discarded permanently, leaving all other registers untouched.
    for step, cmd, explanation in steps:
        print(f"  {step}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")

    print('  Key point: "_dd ensures deletes do NOT overwrite the unnamed register.')
    print("  This lets you freely delete garbage while preserving your yank for pasting.")


# === Exercise 5: Edit a Macro ===
# Problem: Modify a recorded macro without re-recording from scratch.

def exercise_5():
    """Solution: Two methods to edit an existing macro."""
    print("  Original macro in register a: adds console.log inline.")
    print("  Goal: modify it to add console.log on a NEW line above instead.\n")

    print("  Method 1: Paste, edit, yank back")
    method1 = [
        ("Open a blank line", "Create space to work"),
        ('"ap', "Paste register a contents (raw keystrokes as text)"),
        ("(edit the text)", "Change the I to O or modify as needed"),
        ("V", "Select the modified line"),
        ('"ay', "Yank the modified text back into register a"),
        ("@a", "Run the modified macro"),
    ]
    for cmd, explanation in method1:
        print(f"    {cmd:20s} -- {explanation}")

    print()
    print("  Method 2: Set register directly with :let")
    print('    :let @a = "Oconsole.log(\\"debug\\");\\<Esc>j"')
    print()
    # Why: Macros are just text stored in registers. You can edit them like
    # any other text. :let @a = "..." sets the register directly.
    # Use \\<Esc> for the Escape key and \\" for literal quotes.
    print("  Notes:")
    print("    - Macros are just keystrokes stored in registers -- editable as text")
    print('    - In :let, use \\<Esc> for Escape key, \\" for literal quotes')
    print("    - You can also append to macros: qA...q (uppercase A appends)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: The Unnamed Register Problem", exercise_1),
        ("Exercise 2: Multi-Register Copy/Paste", exercise_2),
        ("Exercise 3: Record and Apply a Macro", exercise_3),
        ("Exercise 4: Black Hole and Yank Register", exercise_4),
        ("Exercise 5: Edit a Macro", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
