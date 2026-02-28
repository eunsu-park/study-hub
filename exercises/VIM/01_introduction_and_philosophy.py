"""
Exercises for Lesson 01: Introduction and Philosophy
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Exit Vim ===
# Problem: Open Vim with `vim` and exit without saving.

def exercise_1():
    """Solution: Exit Vim from a blank buffer."""
    commands = [
        (":q <Enter>", "Quit Vim. Works immediately because the buffer is empty (no changes)."),
        (":q! <Enter>", "Force quit without saving. Use this if you accidentally typed something."),
    ]
    # Why: The :q command is an Ex command inherited from the ed/ex line editor lineage.
    # Vim refuses :q when there are unsaved changes, so :q! overrides that safety check.
    print("  Open Vim with: vim")
    print("  Then use one of these commands:\n")
    for cmd, explanation in commands:
        print(f"  {cmd:20s} -- {explanation}")


# === Exercise 2: Identify Vim's Modes ===
# Problem: Match each action with the correct Vim mode.

def exercise_2():
    """Solution: Match actions to modes."""
    answers = [
        ("1. Typing new code into a function body", "Insert mode",
         "You are inputting new text character by character."),
        ("2. Jumping between lines searching for a bug", "Normal mode",
         "Normal mode is the home base for all navigation."),
        ("3. Running :w to save the file", "Command-line mode",
         ":w is an Ex command entered at the : prompt."),
        ("4. Highlighting three lines to copy them", "Visual mode",
         "Text selection uses Visual mode (V for linewise)."),
    ]
    # Why: Vim's modal design separates concerns -- each mode is optimized
    # for a different task (navigate, type, select, execute commands).
    for action, mode, reason in answers:
        print(f"  {action}")
        print(f"    -> {mode}: {reason}\n")


# === Exercise 3: The Editing Time Distribution ===
# Problem: Recall the 70/20/10 split and explain why it justifies modal design.

def exercise_3():
    """Solution: Editing time distribution and modal design justification."""
    distribution = [
        ("70%", "Reading and navigating code"),
        ("20%", "Modifying existing text"),
        ("10%", "Inserting new text from scratch"),
    ]
    # Why: A conventional single-mode editor optimizes for the 10% (typing).
    # Vim's Normal mode optimizes for the 90% (reading + modifying), making
    # the modal trade-off worthwhile for the most common editing activities.
    print("  Editing time distribution:")
    for pct, activity in distribution:
        print(f"    {pct:5s} -- {activity}")
    print()
    print("  Justification for modal design:")
    print("    Single-mode editors optimize for 10% of editing time (typing).")
    print("    Vim optimizes for the full 90% by giving navigation/manipulation")
    print("    their own dedicated mode (Normal mode).")


# === Exercise 4: vi vs Vim vs Neovim Feature Comparison ===
# Problem: Answer feature questions about the three editors.

def exercise_4():
    """Solution: vi vs Vim vs Neovim features."""
    answers = [
        ("1. Which editor first introduced multi-level undo?",
         "Vim (1991)",
         "vi only had single-level undo -- you could undo once, then undo-the-undo."),
        ("2. Which editor has LSP support built-in?",
         "Neovim",
         "Vim requires a plugin (e.g., coc.nvim); Neovim has a native LSP client."),
        ("3. What configuration file does each editor use?",
         "vi: .exrc  |  Vim: .vimrc  |  Neovim: init.lua or init.vim",
         "The config file name reflects the editor's lineage and scripting language."),
    ]
    for question, answer, note in answers:
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"     Note: {note}\n")


# === Exercise 5: Run vimtutor and Reflect ===
# Problem: After completing vimtutor, answer three questions.

def exercise_5():
    """Solution: vimtutor reflection questions."""
    answers = [
        ("1. What keystroke moves the cursor down one line?",
         "j",
         "Mnemonic: 'j' hangs below the baseline, pointing downward."),
        ("2. What command deletes the character under the cursor?",
         "x",
         "Think of 'x' as crossing out a character on paper."),
        ("3. What command saves and quits in one step?",
         ":wq (or ZZ in Normal mode)",
         ":w writes the file, :q quits. ZZ is the Normal-mode shortcut."),
    ]
    print("  After completing vimtutor:\n")
    for question, answer, note in answers:
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"     {note}\n")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Exit Vim", exercise_1),
        ("Exercise 2: Identify Vim's Modes", exercise_2),
        ("Exercise 3: The Editing Time Distribution", exercise_3),
        ("Exercise 4: vi vs Vim vs Neovim Feature Comparison", exercise_4),
        ("Exercise 5: Run vimtutor and Reflect", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
