"""
Exercises for Lesson 10: Buffers, Windows, and Tabs
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Buffer vs Window vs Tab ===
# Problem: Classify each scenario as correct or incorrect usage.

def exercise_1():
    """Solution: Evaluate buffer/window/tab usage patterns."""
    scenarios = [
        ("1. One tab per file (app.py in Tab 1, models.py in Tab 2, views.py in Tab 3)",
         "INCORRECT",
         "Classic anti-pattern. Vim tabs are window layouts, not file holders.\n"
         "         Use :e to open files as buffers and navigate with :bn/:bp or :b {name}."),
        ("2. Open all files with :e, navigate with :bn/:bp and Ctrl-^",
         "CORRECT",
         "The intended buffer workflow. Ctrl-^ quickly toggles between the two\n"
         "         most recently used buffers."),
        ("3. Use :vsp to show app.py and test_app.py side by side",
         "CORRECT",
         "Windows (splits) are exactly for viewing multiple buffers simultaneously."),
        ("4. Tab 1: source split with tests, Tab 2: documentation",
         "CORRECT",
         "Using tabs for distinct 'workspaces' or concerns is a legitimate use."),
    ]
    # Why: Buffers are the primary unit in Vim. Windows are viewports.
    # Tabs are collections of windows (layouts), NOT file holders.
    for scenario, verdict, explanation in scenarios:
        print(f"  {scenario}")
        print(f"    Verdict: {verdict}")
        print(f"    Reason:  {explanation}\n")


# === Exercise 2: Reading the Buffer List ===
# Problem: Interpret :ls output.

def exercise_2():
    """Solution: Interpret buffer list symbols."""
    print("  :ls output:")
    print('    1 %a   "main.py"        line 10')
    print('    2 #    "utils.py"       line 1')
    print('    3  h   "config.py"      line 5')
    print('    4  h+  "README.md"      line 1')
    print()

    answers = [
        ("1. Which file is currently displayed?",
         "main.py",
         "% marks the current buffer; 'a' means active (visible in a window)."),
        ("2. What does Ctrl-^ do right now?",
         "Switches to utils.py",
         "Ctrl-^ (or :b#) toggles to the alternate buffer marked with #."),
        ("3. Which buffer has unsaved changes?",
         "README.md (buffer 4)",
         "The + flag means modified (unsaved changes). Buffer 4 shows h+ (hidden + modified)."),
        ("4. How many buffers are loaded but NOT visible?",
         "Two: config.py and README.md",
         "Both marked 'h' (hidden -- loaded in memory but not displayed in any window)."),
    ]
    # Why: Reading the buffer list is essential for multi-file workflows.
    # The symbols %, #, a, h, + give you instant situational awareness.
    for question, answer, explanation in answers:
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"     {explanation}\n")


# === Exercise 3: Window Navigation Sequence ===
# Problem: Navigate a 3-window layout (A|B over C).

def exercise_3():
    """Solution: Window navigation key sequences."""
    print("  Layout:")
    print("    +--------+--------+")
    print("    |  A     |  B     |")
    print("    | main.py| test.py|")
    print("    +--------+--------+")
    print("    |       C         |")
    print("    |    utils.py     |")
    print("    +-----------------+")
    print("  Cursor starts in Window A\n")

    tasks = [
        ("1. Move from A to B",
         "Ctrl-w l",
         "Move right from A to B."),
        ("2. Move from B to C",
         "Ctrl-w j",
         "Move down from B to C."),
        ("3. From C, move to A",
         "Ctrl-w k (or Ctrl-w h then Ctrl-w k)",
         "Move up from C. Goes to window directly above."),
        ("4. Maximize Window C's height",
         "Ctrl-w _",
         "Maximizes the current window's height (others shrink to minimum)."),
        ("5. Restore all windows to equal size",
         "Ctrl-w =",
         "Equalizes all window sizes."),
    ]
    # Why: All window commands use the Ctrl-w prefix, followed by a direction
    # (h/j/k/l) or a size command (=, _, |, +, -).
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Keys: {cmd}")
        print(f"    Note: {explanation}\n")


# === Exercise 4: Multi-File Rename with :bufdo ===
# Problem: Rename a variable across all open buffers.

def exercise_4():
    """Solution: Using :bufdo for cross-file operations."""
    print("  Task: Rename 'user_id' to 'account_id' across 5 open buffers.\n")

    answers = [
        ("1. Single command to rename across all buffers and save:",
         ":bufdo %s/user_id/account_id/ge | update",
         "  :bufdo         -- execute in every loaded buffer\n"
         "  %s/.../...ge   -- replace all (g), suppress 'not found' errors (e)\n"
         "  | update       -- save each buffer only if modified"),
        ("2. Why is 'set hidden' needed?",
         "Without it, Vim refuses to switch from a modified buffer.",
         "  :bufdo must move through every buffer. If any has pending changes\n"
         "  and hidden is not set, the command aborts with an error."),
        ("3. What does the 'e' flag do?",
         "Suppresses 'Pattern not found' errors.",
         "  Without e, :bufdo aborts on the first buffer where user_id\n"
         "  doesn't exist, leaving remaining buffers unprocessed."),
    ]
    # Why: :bufdo is the batch operation tool for multi-file editing.
    # Combined with the 'e' flag and 'set hidden', it processes all files reliably.
    for question, answer, detail in answers:
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"    {detail}\n")


# === Exercise 5: Tab Workspace Design ===
# Problem: Design a tab + window layout for a web project.

def exercise_5():
    """Solution: Complete Vim commands to set up a 3-tab workspace."""
    print("  Requirements:")
    print("    Tab 1: app.py and models.py side by side")
    print("    Tab 2: test_app.py and test_models.py side by side")
    print("    Tab 3: README.md alone\n")

    commands = [
        ("\" Tab 1: Source code side by side", ""),
        (":e app.py", "Open source file"),
        (":vsp models.py", "Vertical split with models"),
        ("", ""),
        ("\" Tab 2: Tests side by side", ""),
        (":tabnew test_app.py", "New tab with test file"),
        (":vsp test_models.py", "Vertical split with models test"),
        ("", ""),
        ("\" Tab 3: Documentation", ""),
        (":tabnew README.md", "New tab with docs"),
        ("", ""),
        ("\" Navigate back to Tab 1", ""),
        (":tabfirst (or 1gt)", "Go to first tab"),
    ]
    # Why: Tabs serve as workspaces (code, tests, docs) -- not as file holders.
    # All files are also accessible as buffers from any tab.
    print("  Setup commands:")
    for cmd, explanation in commands:
        if not cmd:
            print()
        elif explanation:
            print(f"    {cmd:30s} \" {explanation}")
        else:
            print(f"    {cmd}")

    print()
    print("  Navigation after setup:")
    print("    gt / gT              -- Switch between tabs")
    print("    Ctrl-w h / Ctrl-w l  -- Switch windows within a tab")
    print()
    print("  Note: Files are accessible as buffers from ANY tab -- no re-opening needed.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Buffer vs Window vs Tab", exercise_1),
        ("Exercise 2: Reading the Buffer List", exercise_2),
        ("Exercise 3: Window Navigation Sequence", exercise_3),
        ("Exercise 4: Multi-File Rename with :bufdo", exercise_4),
        ("Exercise 5: Tab Workspace Design", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
