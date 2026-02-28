"""
Exercises for Lesson 12: Configuration and Vimrc
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Interpreting Settings ===
# Problem: Explain what each setting does and why you might want it.

def exercise_1():
    """Solution: Interpret common .vimrc settings."""
    settings = [
        ("set relativenumber",
         "Shows line numbers relative to the cursor position.",
         "The current line shows its absolute number; lines above and below\n"
         "    show their distance (1, 2, 3...). Makes jump commands like 5j, 12k,\n"
         "    or d8j much easier because you can READ the distance from the numbers."),
        ("set scrolloff=8",
         "Keeps at least 8 lines visible above and below the cursor.",
         "Provides context -- you always see surrounding code rather than the\n"
         "    cursor jumping to the very edge of the screen. Prevents disorientation."),
        ("set undofile",
         "Saves undo history to disk (persistent undo).",
         "Without this, undo history is lost when you close a file. With undofile,\n"
         "    you can undo changes from PREVIOUS sessions even after restarting Vim."),
        ("set splitright",
         "New vertical splits open to the RIGHT instead of the default left.",
         "Matches the natural left-to-right reading direction. When you :vsp,\n"
         "    the new window appears on the right, which feels more intuitive."),
    ]
    # Why: Each setting solves a specific usability problem. The key principle
    # is to add settings when you encounter a real friction point, not preemptively.
    for setting, what, why in settings:
        print(f"  {setting}")
        print(f"    What: {what}")
        print(f"    Why:  {why}\n")


# === Exercise 2: nmap vs nnoremap ===
# Problem: Explain the danger of recursive mappings.

def exercise_2():
    """Solution: The critical difference between nmap and nnoremap."""
    print("  1. Key difference:")
    print("    nmap    = RECURSIVE mapping (right-hand side follows other mappings)")
    print("    nnoremap = NON-RECURSIVE mapping (right-hand side is literal)\n")

    print("  2. The danger with recursive nmap:")
    print("    nmap j gj")
    print("    nmap gj j")
    print()
    print("    Pressing j triggers gj, but gj is mapped to j,")
    print("    which triggers gj again... INFINITE LOOP that hangs Vim.\n")

    # Why: nnoremap ignores other mappings on the right-hand side, so you
    # always know exactly what will happen. No accidental chains or loops.
    print("  3. Why nnoremap is almost always correct:")
    print("    - Ignores other mappings on the right-hand side")
    print("    - Prevents accidental chains and infinite loops")
    print("    - You always know exactly what the mapping does")
    print("    - Only use nmap when you INTENTIONALLY want one mapping")
    print("      to trigger another (which is rarely needed)")


# === Exercise 3: Writing Autocommands ===
# Problem: Write three autocommands wrapped in a proper augroup.

def exercise_3():
    """Solution: Autocommand group with three practical autocommands."""
    print("  Complete augroup block:\n")

    vimrc_block = [
        'augroup MyProductivity',
        '    autocmd!',
        '    " 1. Remove trailing whitespace on save (all files)',
        '    autocmd BufWritePre * :%s/\\s\\+$//e',
        '',
        '    " 2. Markdown: word wrap and spell check',
        '    autocmd FileType markdown setlocal wrap linebreak spell',
        '',
        '    " 3. Auto-run Python file after saving',
        '    autocmd BufWritePost *.py :!python %',
        'augroup END',
    ]
    for line in vimrc_block:
        print(f"    {line}")

    # Why: augroup + autocmd! prevents duplicate autocommands when re-sourcing .vimrc.
    # Without the group, each :source adds another copy of each autocommand.
    print()
    print("  Key points:")
    print("    - autocmd! at the start clears the group (prevents duplicates on reload)")
    print("    - Task 3 uses BufWritePost (AFTER saving) so Python runs the saved version")
    print("    - Using BufWritePre (BEFORE saving) would run the old version of the file")


# === Exercise 4: Leader Key Design ===
# Problem: Write four nnoremap commands using Space as leader.

def exercise_4():
    """Solution: Leader key mappings for common operations."""
    print('  let mapleader = " "    " Space as leader\n')

    mappings = [
        ('<leader>w  :w<CR>',
         "Save the current file",
         "<CR> is needed to submit the Ex command (like pressing Enter)."),
        ('<leader>q  :bd<CR>',
         "Close the current BUFFER (not just the window)",
         ":bd deletes the buffer. Different from :q which closes the window."),
        ('<leader>ev :edit $MYVIMRC<CR>',
         "Open .vimrc for editing",
         "$MYVIMRC is a special Vim variable holding the path to the active vimrc."),
        ('<leader>sv :source $MYVIMRC<CR>',
         "Reload .vimrc without restarting Vim",
         ":source re-reads and executes the file. Instant config changes."),
    ]
    # Why: Space as leader is popular because it is a large key that does nothing
    # useful in Normal mode by default. Leader mappings create a personal namespace
    # that avoids conflicts with built-in Vim commands.
    print("  Mappings (all using nnoremap):\n")
    for mapping, purpose, note in mappings:
        print(f"    nnoremap {mapping}")
        print(f"      Purpose: {purpose}")
        print(f"      Note:    {note}\n")


# === Exercise 5: Filetype Configuration ===
# Problem: Write augroup for Python (4-space), JavaScript (2-space), Go (real tabs).

def exercise_5():
    """Solution: Filetype-specific settings for three languages."""
    print("  Complete augroup block:\n")

    vimrc_block = [
        'augroup LanguageSettings',
        '    autocmd!',
        '',
        '    " Python: PEP 8 style',
        '    autocmd FileType python setlocal',
        '        \\ tabstop=4',
        '        \\ shiftwidth=4',
        '        \\ softtabstop=4',
        '        \\ expandtab',
        '        \\ colorcolumn=79',
        '',
        '    " JavaScript: common JS convention',
        '    autocmd FileType javascript,typescript setlocal',
        '        \\ tabstop=2',
        '        \\ shiftwidth=2',
        '        \\ softtabstop=2',
        '        \\ expandtab',
        '',
        '    " Go: gofmt uses real tabs',
        '    autocmd FileType go setlocal',
        '        \\ tabstop=4',
        '        \\ shiftwidth=4',
        '        \\ noexpandtab',
        '',
        'augroup END',
    ]
    for line in vimrc_block:
        print(f"    {line}")

    # Why: setlocal applies settings only to the current buffer (not globally).
    # Setting tabstop, shiftwidth, AND softtabstop ensures consistent indentation
    # regardless of how indentation is triggered (Tab key, auto-indent, >>).
    print()
    print("  Key points:")
    print("    - setlocal (not set) applies only to the current buffer")
    print("    - \\ at line start enables multi-line arguments in Vimscript")
    print("    - Set tabstop + shiftwidth + softtabstop for consistency")
    print("    - Go uses noexpandtab (real tabs) per gofmt convention")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Interpreting Settings", exercise_1),
        ("Exercise 2: nmap vs nnoremap", exercise_2),
        ("Exercise 3: Writing Autocommands", exercise_3),
        ("Exercise 4: Leader Key Design", exercise_4),
        ("Exercise 5: Filetype Configuration", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
