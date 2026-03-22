"""
Exercises for Lesson 06: Text Objects
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Inner vs. Around ===
# Problem: Given "result = calculate(x + y, z * 2)" with cursor inside parentheses,
# describe the result of each command.

def exercise_1():
    """Solution: Inner vs Around text object commands."""
    print("  Text: result = calculate(x + y, z * 2)")
    print("  Cursor somewhere inside 'x + y, z * 2'\n")

    commands = [
        ("di(",
         "result = calculate()",
         "Deletes everything INSIDE the parentheses, keeping the parens themselves."),
        ("da(",
         "result = calculate",
         "Deletes the parentheses AND their content."),
        ("ci(",
         "result = calculate(|)",
         "Deletes content inside parens and enters Insert mode between ( and )."),
        ("yi(",
         "result = calculate(x + y, z * 2)  [unchanged]",
         "Copies 'x + y, z * 2' into the register. Text is not modified."),
    ]
    # Why: 'i' (inner) keeps delimiters, 'a' (around) removes them.
    # Rule of thumb: use 'i' to replace content (ci"), use 'a' to remove entirely (da").
    for cmd, result, explanation in commands:
        print(f"  {cmd:5s} -> {result}")
        print(f"         {explanation}\n")


# === Exercise 2: Choose the Right Text Object ===
# Problem: Write the single command for each editing task.

def exercise_2():
    """Solution: Choose the correct operator + text object."""
    tasks = [
        ('1. Inside "active", replace the word with a different value',
         'ci"',
         'Change inner quotes: deletes "active" content, enters Insert mode between "".'),
        ("2. Inside def process(data, config):, copy all arguments",
         "yi(",
         "Yank inner parentheses: copies 'data, config' without the parens."),
        ("3. Inside <p>Old content</p>, clear content and type new text",
         "cit",
         "Change inner tag: deletes 'Old content', enters Insert mode inside <p></p>."),
        ('4. Inside {"key": "value"}, delete everything including braces',
         "da{",
         'Delete around braces: removes {"key": "value"} entirely.'),
    ]
    # Why: Text objects are position-independent -- your cursor can be anywhere
    # inside the object. No need to navigate to the boundary first.
    for task, cmd, explanation in tasks:
        print(f"  {task}")
        print(f"    Command: {cmd}")
        print(f"    Note:    {explanation}\n")


# === Exercise 3: Paragraph Object for Code ===
# Problem: Delete the main() function using paragraph objects.

def exercise_3():
    """Solution: Using paragraph objects on Python functions."""
    print("  Code:")
    code = [
        "def setup():",
        "    initialize_db()",
        "    load_config()",
        "",
        "def main():       <- cursor on 'run_app()' line",
        "    setup()",
        "    run_app()",
        "",
        "def cleanup():",
        "    close_db()",
        "    save_state()",
    ]
    for line in code:
        print(f"    {line}")
    print()

    commands = [
        ("dip",
         "Delete inner paragraph",
         "Removes the lines between blank lines (def main, setup, run_app).\n"
         "         The surrounding blank lines remain."),
        ("dap",
         "Delete around paragraph",
         "Removes the function block AND the trailing blank line."),
    ]
    # Why: In code separated by blank lines, paragraphs naturally correspond
    # to function/class blocks. dip/dap are a fast way to operate on them.
    for cmd, name, explanation in commands:
        print(f"  {cmd:5s} -- {name}")
        print(f"         {explanation}\n")

    print("  Note: Cursor must be within the main() block, not on a blank line.")


# === Exercise 4: Nested Text Objects ===
# Problem: Given 'outer("inner value", more)' with cursor on 'v' of 'value'.

def exercise_4():
    """Solution: Text objects with nested delimiters."""
    print('  Text: outer("inner value", more)')
    print("  Cursor on 'v' of 'value'\n")

    steps = [
        ('1. di"',
         'outer("", more)',
         'Deletes content of nearest enclosing quotes: removes "inner value".'),
        ('2. da"',
         'outer(, more)',
         'Deletes the quotes AND their content: removes "inner value" entirely.'),
        ('3. After da", what does text look like?',
         'outer(, more)',
         'The leading comma and space remain (text objects only handle the quoted part).'),
        ("4. How to delete everything inside outer parens?",
         "di( (with cursor inside the outer parentheses)",
         "Deletes everything inside outer (): produces outer()."),
    ]
    # Why: Text objects respect nesting. di" targets the innermost enclosing "".
    # To reach outer delimiters, you may need to position outside the inner ones first.
    for step, result, explanation in steps:
        print(f"  {step}")
        print(f"    Result: {result}")
        print(f"    Note:   {explanation}\n")


# === Exercise 5: Real-World Editing Scenario ===
# Problem: Change a URL and change an entire object in a JS config.

def exercise_5():
    """Solution: Text objects in a real JavaScript configuration."""
    print("  Code:")
    code = [
        "const config = {",
        '    apiUrl: "https://old-api.example.com/v1",',
        "    timeout: 5000,",
        "    retries: 3,",
        "};",
    ]
    for line in code:
        print(f"    {line}")
    print()

    print("  Task 1: Change the URL to a new address")
    task1_steps = [
        ("(position cursor anywhere inside the URL string)", ""),
        ('ci"', "Deletes URL content, enters Insert mode between quotes"),
        ("(type new URL)", ""),
        ("Esc", "Return to Normal mode"),
    ]
    # Why: ci" works regardless of cursor position within the quoted string.
    # No need to navigate to the opening or closing quote.
    for cmd, explanation in task1_steps:
        if explanation:
            print(f"    {cmd:55s} -- {explanation}")
        else:
            print(f"    {cmd}")

    print()
    print("  Task 2: Change the entire object content")
    task2_steps = [
        ("(position cursor anywhere inside the { } block)", ""),
        ("ci{ (or ciB)", "Deletes all content between { and }, enters Insert mode"),
        ("(type new settings)", ""),
        ("Esc", "Return to Normal mode"),
    ]
    for cmd, explanation in task2_steps:
        if explanation:
            print(f"    {cmd:55s} -- {explanation}")
        else:
            print(f"    {cmd}")

    print()
    print("  Both tasks: 3 actions regardless of content size -- the power of text objects.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Inner vs. Around", exercise_1),
        ("Exercise 2: Choose the Right Text Object", exercise_2),
        ("Exercise 3: Paragraph Object for Code", exercise_3),
        ("Exercise 4: Nested Text Objects", exercise_4),
        ("Exercise 5: Real-World Editing Scenario", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
