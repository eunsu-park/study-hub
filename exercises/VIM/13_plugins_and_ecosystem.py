"""
Exercises for Lesson 13: Plugins and Ecosystem
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: vim-plug Setup ===
# Problem: Write the plugin declaration, install command, and cleanup command.

def exercise_1():
    """Solution: vim-plug configuration and management."""
    print("  1. Complete .vimrc plugin section:\n")

    vimrc = [
        "call plug#begin('~/.vim/plugged')",
        "",
        "Plug 'tpope/vim-surround'",
        "Plug 'tpope/vim-commentary'",
        "Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }",
        "Plug 'junegunn/fzf.vim'",
        "",
        "call plug#end()",
    ]
    for line in vimrc:
        print(f"    {line}")

    # Why: junegunn/fzf must be declared before junegunn/fzf.vim because
    # fzf.vim depends on the fzf binary that the first plugin installs via its 'do' hook.
    print()
    print("  Note: fzf must be declared before fzf.vim (dependency order).\n")

    print("  2. Install all plugins:")
    print("    :PlugInstall")
    print("    vim-plug downloads all declared plugins to ~/.vim/plugged/\n")

    print("  3. Clean up removed plugins:")
    print("    (After removing plugin lines from .vimrc and reloading)")
    print("    :PlugClean")
    print("    Detects undeclared plugins in the plugged directory and offers to delete them.")


# === Exercise 2: vim-surround Operations ===
# Problem: Write vim-surround commands for each transformation.

def exercise_2():
    """Solution: vim-surround command for each text transformation."""
    transforms = [
        ('hello -> "hello"',
         'ysiw"',
         "ys (you surround) + iw (inner word) + \" (with double quotes)."),
        ('"hello" -> \'hello\'',
         "cs\"'",
         "cs (change surrounding) + \" (old) + ' (new)."),
        ("'hello' -> hello",
         "ds'",
         "ds (delete surrounding) + ' (the character to remove)."),
        ("hello world -> (hello world)",
         "yss)",
         "yss (surround entire line) + ) (closing paren, no spaces).\n"
         "         Use ( instead of ) to get spaces: ( hello world )."),
    ]
    # Why: vim-surround extends the text object grammar with surrounding operations.
    # cs (change), ds (delete), ys (add) follow the same composable pattern.
    for transform, cmd, explanation in transforms:
        print(f"  {transform}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: vim-commentary ===
# Problem: Comment/uncomment operations in a Python function.

def exercise_3():
    """Solution: vim-commentary commands for toggling comments."""
    print("  Code:")
    print("    def process(data):")
    print("        result = transform(data)")
    print("        debug_log(result)")
    print("        return result\n")

    tasks = [
        ("1. Comment out just the debug_log line (cursor on it)",
         "gcc",
         "Toggle comment on current line. Adds '# ' at the beginning."),
        ("2. Comment out the entire function body (lines 2-4)",
         "gc2j (with cursor on line 2, or V2j then gc)",
         "gc + 2j motion covers 3 lines. Visual: select lines first, then gc."),
        ("3. Uncomment the same lines",
         "gc2j (or gcc per line)",
         "gcc/gc toggles: if already commented, it removes the comment."),
        ("4. What does vim-repeat enable?",
         ". repeats the last gcc/gc action",
         "Without vim-repeat, . only repeats built-in Vim actions.\n"
         "         With it, you can comment a line, move, and press . to comment another."),
    ]
    # Why: vim-commentary follows Vim's composable grammar (gc is an operator).
    # vim-repeat makes it dot-repeatable, which is essential for efficient editing.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 4: Evaluating a Plugin ===
# Problem: Apply plugin audit questions to an indent guide plugin.

def exercise_4():
    """Solution: Plugin evaluation using audit criteria."""
    questions = [
        ("1. Can you do this with built-in Vim features?",
         "Partially. :set list shows whitespace characters, and foldmethod=indent\n"
         "    gives indentation structure. But there is no built-in visual indent\n"
         "    guide like VS Code's. The plugin provides genuinely new functionality."),
        ("2. What quality signals to check?",
         "- GitHub stars (1000+ is generally reliable)\n"
         "    - Last commit date (within 12 months is healthy)\n"
         "    - Open issues and maintainer responsiveness\n"
         "    - Number of active contributors"),
        ("3. Main cost/risk of adding this plugin?",
         "- Small startup time increase (usually negligible)\n"
         "    - May conflict with some color schemes or terminal emulators\n"
         "    - Another dependency to maintain and update"),
        ("4. Would you install it?",
         "Install IF you regularly work with deeply-nested code (Python, JS)\n"
         "    where tracking indentation is error-prone. Skip if you already use\n"
         "    set number + set relativenumber and navigate by line number.\n"
         "    Principle: install only if it solves a problem you actually experience."),
    ]
    # Why: The plugin audit questions prevent bloat.
    # Every plugin adds startup time, potential bugs, and maintenance burden.
    for question, answer in questions:
        print(f"  {question}")
        print(f"    {answer}\n")


# === Exercise 5: Writing a Custom Command ===
# Problem: Write a :Header command that inserts a comment header block.

def exercise_5():
    """Solution: Custom Vimscript command to insert a header block."""
    print("  Usage: :Header Utilities\n")
    print("  Output:")
    print("    # ============================================================")
    print("    # Section: Utilities")
    print("    # ============================================================\n")

    print("  Vimscript implementation:\n")
    vimscript = [
        'command! -nargs=1 Header call InsertHeader(<q-args>)',
        '',
        'function! InsertHeader(title)',
        '    let separator = "# ============================================================"',
        '    let titleline = "# Section: " . a:title',
        "    call append(line('.') - 1, separator)",
        "    call append(line('.'), titleline)",
        "    call append(line('.') + 1, separator)",
        'endfunction',
    ]
    for line in vimscript:
        print(f"    {line}")

    # Why: Custom commands let you automate repetitive formatting tasks.
    # command! -nargs=1 defines a command taking exactly 1 argument.
    # <q-args> passes the argument as a quoted string.
    # append() inserts lines at specific positions.
    print()
    print("  Key concepts:")
    print("    command! -nargs=1  -- defines a command taking 1 argument")
    print("    <q-args>           -- passes the argument as a quoted string")
    print("    a:title            -- accesses the function argument 'title'")
    print("    append(line, text) -- inserts text at the given line number")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: vim-plug Setup", exercise_1),
        ("Exercise 2: vim-surround Operations", exercise_2),
        ("Exercise 3: vim-commentary", exercise_3),
        ("Exercise 4: Evaluating a Plugin", exercise_4),
        ("Exercise 5: Writing a Custom Command", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
