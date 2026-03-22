"""
Exercises for Lesson 08: Search and Replace
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Search Navigation ===
# Problem: Navigate through search results in a Python file.

def exercise_1():
    """Solution: Search navigation commands."""
    tasks = [
        ("1. Search forward for the word 'return'",
         "/return <Enter>",
         "/ enters forward search mode, type the pattern, Enter confirms."),
        ("2. Jump to the next occurrence",
         "n",
         "Jumps to the next match in the same direction (forward)."),
        ("3. Jump back to the previous occurrence",
         "N",
         "Reverses the search direction -- jumps to the previous match."),
        ("4. Search for the exact word under cursor (e.g., 'calculate')",
         "*",
         "Searches forward for the whole word under cursor. Equivalent to /\\<calculate\\> but zero typing."),
    ]
    # Why: * is one of Vim's most underused features. It instantly searches for
    # the word under the cursor as a whole-word match -- no typing needed.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 2: Substitution Syntax ===
# Problem: Write the Vim command for each substitution task.

def exercise_2():
    """Solution: Substitution commands for various tasks."""
    tasks = [
        ("1. Replace all 'foo' with 'bar' in entire file",
         ":%s/foo/bar/g",
         "% = entire file, g = all occurrences per line."),
        ("2. Replace 'http://' with 'https://' (note slashes in pattern)",
         ":%s#http://#https://#g",
         "Use # as delimiter to avoid escaping /. Also works: :%s|http://|https://|g"),
        ("3. Replace 'old_var' with 'new_var' on lines 15-30 only",
         ":15,30s/old_var/new_var/g",
         "Range 15,30 limits substitution to those lines."),
        ("4. Replace 'DEBUG' with 'INFO' on current line with confirmation",
         ":s/DEBUG/INFO/gc",
         "No % = current line only. c flag asks for confirmation at each match."),
    ]
    # Why: The substitution command is extremely flexible with its range, delimiter,
    # and flag options. Learning these combinations handles 90% of bulk editing tasks.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: The Global Command ===
# Problem: Write :g or :v commands for line-based operations.

def exercise_3():
    """Solution: Global and inverse-global commands."""
    tasks = [
        ("1. Delete all lines containing 'TODO'",
         ":g/TODO/d",
         ":g runs the command (d = delete) on every line matching the pattern."),
        ("2. Delete all blank lines",
         ":g/^\\s*$/d",
         "Pattern matches lines with only whitespace from start (^) to end ($)."),
        ("3. Show (print) all lines containing 'import'",
         ":g/import/p",
         "p prints each matching line. Use :g/import/# to also show line numbers."),
        ("4. Delete all lines NOT containing 'def '",
         ":v/def /d",
         ":v is the inverse of :g -- acts on lines that do NOT match."),
    ]
    # Why: :g (global) and :v (inverse global) are the power tools for
    # line-based bulk operations. They combine pattern matching with any Ex command.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 4: Regex Pattern Writing ===
# Problem: Write substitution commands using regular expressions.

def exercise_4():
    """Solution: Regex-based substitutions."""
    tasks = [
        ("1. Remove all trailing whitespace",
         ":%s/\\s\\+$//g",
         "\\s\\+ matches one or more whitespace chars, $ anchors to end of line."),
        ("2. Swap 'Smith, John' to 'John Smith' (capture groups)",
         ":%s/\\(\\w\\+\\), \\(\\w\\+\\)/\\2 \\1/g",
         "\\(\\w\\+\\) captures words. \\1 and \\2 reference them in replacement.\n"
         "         Very magic mode: :%s/\\v(\\w+), (\\w+)/\\2 \\1/g"),
        ("3. Replace standalone 'count' with 'total' (whole word only)",
         ":%s/\\<count\\>/total/g",
         "\\< and \\> are word boundary anchors. Won't match 'counter' or 'discount'."),
    ]
    # Why: Vim regex differs slightly from PCRE (Perl). Key differences:
    # - Need to escape +, |, (, ) in default mode (or use \v for very magic).
    # - Word boundaries use \< and \> instead of \b.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 5: Combined Search and Replace Workflow ===
# Problem: Safely rename 'processData' to 'handleRequest' with confirmation.

def exercise_5():
    """Solution: Interactive rename with word boundaries and confirmation."""
    print("  Command: :%s/\\<processData\\>/handleRequest/gc\n")

    print("  Breakdown:")
    parts = [
        ("%", "Entire file"),
        ("s/", "Substitute"),
        ("\\<processData\\>", "Match exact whole word 'processData' (won't match processDataHelper)"),
        ("/handleRequest/", "Replacement text"),
        ("g", "Global -- all occurrences per line"),
        ("c", "Confirm each replacement"),
    ]
    for part, explanation in parts:
        print(f"    {part:25s} -- {explanation}")

    print()
    print("  Confirmation responses:")
    responses = [
        ("y", "Replace this occurrence"),
        ("n", "Skip this occurrence"),
        ("a", "Replace ALL remaining occurrences at once"),
        ("q", "Quit without replacing any more"),
        ("l", "Replace this one and quit (last)"),
    ]
    # Why: Using \<...\> word boundaries prevents accidental renames of
    # similarly-named identifiers. The c flag gives you manual control.
    for key, action in responses:
        print(f"    {key} -- {action}")
    print()
    print("  The \\<...\\> word boundaries make this safe -- no accidental partial matches.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Search Navigation", exercise_1),
        ("Exercise 2: Substitution Syntax", exercise_2),
        ("Exercise 3: The Global Command", exercise_3),
        ("Exercise 4: Regex Pattern Writing", exercise_4),
        ("Exercise 5: Combined Search and Replace Workflow", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
