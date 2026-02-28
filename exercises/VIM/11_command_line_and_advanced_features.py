"""
Exercises for Lesson 11: Command Line and Advanced Features
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Ex Range Syntax ===
# Problem: Write Ex commands for line operations (no search/replace).

def exercise_1():
    """Solution: Ex range commands for line manipulation."""
    tasks = [
        ("1. Delete lines 5 through 12",
         ":5,12d",
         "Range 5,12 targets those lines; d deletes them."),
        ("2. Move lines 20-25 to the end of the file",
         ":20,25m$",
         "$ is the last line address. m moves lines to after the target."),
        ("3. Copy line 10 to just after line 50",
         ":10t50",
         ":t (same as :co) copies. The destination is after line 50."),
        ("4. Delete the current line and the next 3 lines below",
         ":.,+3d",
         ". is current line, +3 is 3 lines below. Range covers 4 lines total."),
        ("5. Move all lines between marks a and b to the top",
         ":'a,'bm0",
         "'a,'b is the range between marks. m0 moves to before line 1 (top of file)."),
    ]
    # Why: Ex commands inherit from the ed/ex line editor.
    # They operate on ranges of lines and are ideal for batch line manipulation.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 2: Shell Filtering ===
# Problem: Use shell commands to format, sort, and insert content.

def exercise_2():
    """Solution: Shell integration commands."""
    tasks = [
        ("1. Format entire file as pretty-printed JSON",
         ":%!python -m json.tool",
         "Pipes the entire file (%) through Python's JSON formatter.\n"
         "         The buffer content is REPLACED with the formatted output."),
        ("2. Sort only lines 30-50 in place",
         ":30,50!sort",
         "Pipes lines 30-50 through system sort; replaces those lines with sorted output."),
        ("3. Insert the current date and time at cursor position",
         ":r !date",
         ":r reads command output into buffer. !date runs the shell date command.\n"
         "         The result is inserted below the cursor."),
        ("4. Difference between :!sort and :%!sort",
         ":!sort shows output; :%!sort replaces buffer",
         ":!sort runs sort and DISPLAYS the output (buffer unchanged).\n"
         "         :%!sort PIPES the buffer through sort and REPLACES the content."),
    ]
    # Why: Shell integration lets you leverage Unix tools directly from Vim.
    # The ! operator is a bridge between Vim's text and the power of the shell.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: Working with Folds ===
# Problem: Configure and use folds for Python code navigation.

def exercise_3():
    """Solution: Fold configuration and usage."""
    tasks = [
        ("1. Which foldmethod auto-folds based on indentation?",
         "set foldmethod=indent",
         "Folds are determined by indentation level -- aligns well with Python's structure."),
        ("2. After zM (all closed), open just one level of folds",
         "zr",
         "'Fold less': increases fold level by 1, revealing class bodies.\n"
         "         Also: :set foldlevel=1"),
        ("3. Create a manual fold over visually selected lines",
         "V (select lines) then zf",
         "V enters line-wise visual mode, select the lines, then zf creates the fold."),
        ("4. Enable fold markers ({{{ and }}})",
         "set foldmethod=marker",
         "Vim looks for {{{ to open folds and }}} to close them.\n"
         "         Embed markers in comments: # Section {{{ ... # }}}"),
    ]
    # Why: Folds let you collapse code sections to see the big picture.
    # indent foldmethod is ideal for Python; marker is useful for manual sections.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Answer: {cmd}")
        print(f"    Note:   {explanation}\n")


# === Exercise 4: The :normal Command ===
# Problem: Explain :normal commands and write one for a specific task.

def exercise_4():
    """Solution: The :normal command for batch Normal-mode operations."""
    tasks = [
        ("1. :%normal A;",
         "Appends ';' to the END of EVERY line in the file.",
         "% = entire file. A enters Insert mode at line end. ; is typed. Esc is implicit."),
        ("2. :'<,'>normal I# ",
         "Inserts '# ' at the BEGINNING of every line in the visual selection.",
         "I enters Insert mode at line start. Comments out selected lines with # prefix."),
        ("3. Write: append ' // TODO' to every line containing 'function'",
         ":g/function/normal A // TODO",
         ":g/function/ finds matching lines. normal A // TODO appends to each.\n"
         "         Combines :g (global pattern match) with :normal (Normal-mode execution)."),
    ]
    # Why: :normal executes Normal-mode keystrokes from the command line.
    # Combined with ranges or :g, it applies Normal-mode edits to many lines at once.
    for task, answer, explanation in tasks:
        print(f"  {task}")
        print(f"    Answer: {answer}")
        print(f"    Note:   {explanation}\n")


# === Exercise 5: Abbreviations and Productivity Features ===
# Problem: Create abbreviations and use spell checking.

def exercise_5():
    """Solution: Abbreviations and spell checking commands."""
    tasks = [
        ("1. Create an abbreviation: 'addr' -> '0x00000000' (Insert mode only)",
         ":iab addr 0x00000000",
         ":iab creates Insert-mode-only abbreviation.\n"
         "         Typing 'addr' + space/punctuation expands it automatically."),
        ("2. Auto-correct 'pritn' to 'print'",
         ":ab pritn print",
         ":ab creates a general abbreviation (works in Insert mode).\n"
         "         Triggers when 'pritn' is followed by a non-word character."),
        ("3. Enable spell checking and find misspelled words",
         ":set spell  then  ]s",
         ":set spell enables spell checking (misspelled words are highlighted).\n"
         "         ]s jumps forward to the next misspelled word."),
        ("4. See spelling suggestions and accept the first one",
         "z= then type 1 and Enter (or 1z= for shortcut)",
         "z= shows the list of suggestions, each numbered.\n"
         "         1z= directly applies the first suggestion without the menu."),
    ]
    # Why: Abbreviations auto-correct typos and expand shorthand.
    # Spell checking is built-in -- no plugin needed for writing documentation.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Ex Range Syntax", exercise_1),
        ("Exercise 2: Shell Filtering", exercise_2),
        ("Exercise 3: Working with Folds", exercise_3),
        ("Exercise 4: The :normal Command", exercise_4),
        ("Exercise 5: Abbreviations and Productivity Features", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
