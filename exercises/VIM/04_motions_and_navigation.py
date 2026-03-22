"""
Exercises for Lesson 04: Motions and Navigation
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: word vs. WORD ===
# Problem: Given "http://api.example.com/v2/users?limit=10" with cursor at 'h',
# compare w vs W to reach 'limit'.

def exercise_1():
    """Solution: word vs WORD navigation on a URL."""
    print("  Text: http://api.example.com/v2/users?limit=10")
    print("  Cursor at 'h' (start of URL)\n")

    answers = [
        ("1. How many w presses to reach 'limit'?",
         "~12-14 presses",
         "w stops at every punctuation boundary (:, //, ., ?, =), creating many word boundaries."),
        ("2. How many W presses to reach 'limit'?",
         "Cannot reach 'limit' with W",
         "The entire URL is one WORD (no spaces). W jumps past the whole URL in 1 press."),
        ("3. Why do the counts differ?",
         "word boundaries include punctuation; WORD boundaries are whitespace only",
         "Use w for fine-grained navigation within symbols; use W to skip entire tokens."),
    ]
    # Why: Understanding word vs WORD helps you choose the right granularity.
    # URLs, file paths, and complex expressions are single WORDs but many words.
    for question, answer, note in answers:
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"     {note}\n")


# === Exercise 2: Line Navigation ===
# Problem: Given "    return user.profile.settings['theme']" (cursor at column 0),
# find the single command to reach each target.

def exercise_2():
    """Solution: Line navigation commands."""
    print('  Text:     return user.profile.settings["theme"]')
    print("  Cursor at column 0 (a space character)\n")

    targets = [
        ("1. The 'r' in 'return'",
         "^",
         "Moves to the first non-blank character. Unlike 0 (which stays at the space)."),
        ("2. The '[' before 'theme'",
         "f[",
         "Finds the next '[' forward on the current line."),
        ("3. The ']' at the end",
         "$",
         "Moves to the end of the line. Alternatively, f] also works."),
    ]
    # Why: Line motions (0, ^, $, f, t) let you jump precisely within a line.
    # ^ is generally more useful than 0 because you want the first real character.
    for target, cmd, explanation in targets:
        print(f"  {target}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: Marks for Multi-Location Editing ===
# Problem: Edit sections at lines 50 and 300, then return to line 50.

def exercise_3():
    """Solution: Complete marks workflow for multi-location editing."""
    workflow = [
        (":50 (or 50G)", "Navigate to line 50"),
        ("ma", "Set mark 'a' at line 50"),
        ("(edit)", "Make your edits at line 50"),
        (":300 (or 300G)", "Navigate to line 300"),
        ("mb", "Set mark 'b' at line 300 (good practice)"),
        ("(edit)", "Make your edits at line 300"),
        ("'a", "Jump back to beginning of line 50"),
        ("(or `a)", "Jump back to EXACT cursor position at line 50"),
    ]
    # Why: Marks let you bookmark positions and return instantly.
    # Lowercase marks (a-z) are file-local; backtick returns to exact column,
    # apostrophe returns to first non-blank of the marked line.
    print("  Workflow:\n")
    for i, (cmd, action) in enumerate(workflow, 1):
        print(f"    {i}. {cmd:20s} -- {action}")
    print()
    print("  Tip: Use `a (backtick) for exact position, 'a (apostrophe) for line start.")


# === Exercise 4: The % Jump ===
# Problem: Diagnose a bracket mismatch using % in:
# result = [x for x in data if (x > 0 and x < 100]

def exercise_4():
    """Solution: Using % to diagnose bracket mismatches."""
    print("  Code: result = [x for x in data if (x > 0 and x < 100]")
    print()

    steps = [
        ("1. Place cursor on [", "The opening bracket after 'result = '"),
        ("2. Press %", "Vim jumps to the matching ] at the end"),
        ("3. Observe", "The ] matches [ -- but the ( inside has no matching )"),
        ("4. Move to ( after 'if'", "Position cursor on the opening parenthesis"),
        ("5. Press %", "Vim cannot find matching ) -- the jump fails or stays put"),
    ]
    for step, explanation in steps:
        print(f"  {step:25s} -- {explanation}")

    # Why: % jumps between matching pairs: (), [], {}.
    # When % fails to jump, it reveals mismatched brackets -- a powerful debugging tool.
    print()
    print("  Diagnosis: The ( has no closing ).")
    print("  Fix: result = [x for x in data if (x > 0 and x < 100)]")
    print("                                                       ^  added )")


# === Exercise 5: Jump List Navigation ===
# Problem: Trace the jump list after: line 1 -> G -> 50% -> gg.
# How many Ctrl-o presses to return to line 200?

def exercise_5():
    """Solution: Jump list trace and Ctrl-o navigation."""
    print("  Actions performed:")
    print("    1. Start at line 1")
    print("    2. G   -> go to line 200 (last line)")
    print("    3. 50% -> go to line 100 (middle)")
    print("    4. gg  -> go to line 1 (first line)\n")

    print("  Jump list (most recent first):")
    jump_list = [
        ("Current", "line 1", "gg brought us here"),
        ("1 step back", "line 100", "50% brought us here"),
        ("2 steps back", "line 200", "G brought us here"),
        ("3 steps back", "line 1", "starting position"),
    ]
    for position, line, how in jump_list:
        print(f"    {position:15s} -> {line:10s} ({how})")

    # Why: Ctrl-o navigates backward through the jump list (like a browser Back button).
    # Ctrl-i navigates forward. Only "jumps" (large motions like G, gg, /, marks) are recorded.
    print()
    print("  To reach line 200:")
    print("    Ctrl-o once  -> line 100")
    print("    Ctrl-o twice -> line 200")
    print()
    print("  Answer: 2 Ctrl-o presses.")
    print("  To go forward again: use Ctrl-i.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: word vs. WORD", exercise_1),
        ("Exercise 2: Line Navigation", exercise_2),
        ("Exercise 3: Marks for Multi-Location Editing", exercise_3),
        ("Exercise 4: The % Jump", exercise_4),
        ("Exercise 5: Jump List Navigation", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
