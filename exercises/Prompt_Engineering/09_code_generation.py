# Exercise: Lesson 09 — Code Generation
# Complete the TODO items below.
#
# Run: python 09_code_generation.py

import anthropic
import json
import re
import textwrap

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Test-Driven Prompt ===
# Write tests first, then ask Claude to generate code that passes them.
# Hint: Include the test cases in the prompt so Claude targets them.

TEST_CASES_FIZZBUZZ = [
    (1, "1"), (3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz"),
    (9, "Fizz"), (10, "Buzz"), (30, "FizzBuzz"), (7, "7"),
]

def generate_code_from_tests(func_name: str, tests: list[tuple],
                             description: str) -> str:
    """Ask Claude to generate a Python function that passes the given tests.
    Args:
        func_name: name of the function to generate
        tests: list of (input, expected_output) tuples
        description: brief description of what the function should do
    Returns: the generated Python code as a string
    """
    # TODO: Format the test cases into the prompt
    # TODO: Ask Claude to write ONLY the function (no tests, no explanation)
    # TODO: Extract the code from the response (strip markdown fences if present)
    # Hint: Tell Claude the function signature: def {func_name}(n: int) -> str
    pass


def exercise_1():
    code = generate_code_from_tests(
        "fizzbuzz", TEST_CASES_FIZZBUZZ,
        "Return 'Fizz' for multiples of 3, 'Buzz' for 5, "
        "'FizzBuzz' for both, otherwise the number as a string.",
    )
    assert "def fizzbuzz" in code, "Must contain function definition"
    # Execute and test
    exec_globals = {}
    exec(code, exec_globals)
    func = exec_globals["fizzbuzz"]
    passed = 0
    for inp, expected in TEST_CASES_FIZZBUZZ:
        result = func(inp)
        ok = result == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[Ex1] {status} | fizzbuzz({inp}) = {result!r} (expected {expected!r})")
    print(f"[Ex1] {passed}/{len(TEST_CASES_FIZZBUZZ)} tests passed")


# === Exercise 2: Code Review Prompt ===
# Ask Claude to review code and suggest improvements.
# Hint: Ask for structured feedback (bugs, style, performance).

BUGGY_CODE = textwrap.dedent("""\
    def find_duplicates(lst):
        duplicates = []
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i != j and lst[i] == lst[j]:
                    duplicates.append(lst[i])
        return duplicates
""")

def review_code(code: str) -> dict:
    """Ask Claude to review code and return structured feedback.
    Return: {
        "bugs": list[str],
        "improvements": list[str],
        "revised_code": str
    }
    """
    # TODO: Build a prompt that asks for bugs, improvements, and revised code
    # TODO: Request JSON output for structured parsing
    # TODO: Parse and return the result
    pass


def exercise_2():
    result = review_code(BUGGY_CODE)
    assert "bugs" in result or "improvements" in result
    assert "revised_code" in result
    print("[Ex2] Code Review Results:")
    for bug in result.get("bugs", []):
        print(f"  BUG: {bug[:80]}")
    for imp in result.get("improvements", []):
        print(f"  IMP: {imp[:80]}")
    print(f"  Revised code preview: {result['revised_code'][:80]}...")


# === Exercise 3: Iterative Code Refinement ===
# Generate code, run it, feed errors back, and retry.
# Hint: This is a generate-test-fix loop.

def extract_code_block(text: str) -> str:
    """Extract Python code from a response (handle markdown fences)."""
    # TODO: Use regex to find ```python ... ``` blocks
    # TODO: If no fenced block, return the text stripped
    pass


def iterative_generate(description: str, test_code: str,
                       max_attempts: int = 3) -> dict:
    """Generate code, test it, and iterate on errors.
    Args:
        description: what the function should do
        test_code: Python code that tests the generated function
        max_attempts: maximum generation attempts
    Returns: {"code": str, "attempts": int, "success": bool}
    """
    # TODO: Loop up to max_attempts times:
    #   1. Ask Claude to generate (or fix) the code
    #   2. Try exec(generated_code + test_code)
    #   3. If no exception, return success
    #   4. If exception, include the error in the next prompt
    # Hint: Catch all exceptions from exec() and feed the traceback back
    pass


def exercise_3():
    result = iterative_generate(
        description=(
            "Write a function called 'flatten' that takes a nested list "
            "and returns a flat list. E.g., flatten([1, [2, [3]], 4]) -> [1, 2, 3, 4]"
        ),
        test_code=textwrap.dedent("""\
            assert flatten([1, [2, [3]], 4]) == [1, 2, 3, 4]
            assert flatten([]) == []
            assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]
            assert flatten([1, 2, 3]) == [1, 2, 3]
        """),
    )
    print(f"[Ex3] Success: {result['success']} in {result['attempts']} attempt(s)")
    if result["success"]:
        print(f"[Ex3] Code: {result['code'][:100]}...")


# === Exercise 4: Docstring Generator ===
# Given a function, generate a comprehensive docstring.
# Hint: Ask Claude to analyze the code and write Google-style docstring.

SAMPLE_FUNCTION = textwrap.dedent("""\
    def merge_sorted(a, b):
        result = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result
""")

def generate_docstring(code: str) -> str:
    """Ask Claude to generate a Google-style docstring for the given function.
    Return only the docstring text (with triple quotes).
    """
    # TODO: Prompt Claude to analyze the function and write a docstring
    # TODO: The docstring should include: summary, Args, Returns, Example
    # TODO: Return just the docstring (including triple-quote delimiters)
    pass


def exercise_4():
    docstring = generate_docstring(SAMPLE_FUNCTION)
    assert '"""' in docstring or "'''" in docstring, "Must contain triple quotes"
    assert "Args" in docstring or "arg" in docstring.lower()
    print(f"[Ex4] Generated docstring:\n{docstring}")


# === Exercise 5: Test-Driven Pipeline (End-to-End) ===
# Full pipeline: spec -> tests -> code -> review -> refined code.

def tdd_pipeline(spec: str) -> dict:
    """Full test-driven code generation pipeline.
    Steps:
      1. Ask Claude to write test cases from the spec
      2. Generate code to pass those tests (iterative_generate)
      3. Review the generated code
      4. Generate a docstring for the final code
    Return: {"tests": str, "code": str, "review": dict, "docstring": str}
    """
    # TODO: Step 1 — Generate test assertions from the spec
    #   Ask Claude to write 4-6 assert statements
    # TODO: Step 2 — Use iterative_generate with those tests
    # TODO: Step 3 — Review the generated code with review_code
    # TODO: Step 4 — Generate a docstring with generate_docstring
    # TODO: Return the combined results
    pass


def exercise_5():
    result = tdd_pipeline(
        "Write a function called 'caesar_cipher' that takes a string and "
        "an integer shift, and returns the string with each letter shifted "
        "by that amount. Non-letter characters stay unchanged. "
        "Handle both uppercase and lowercase."
    )
    assert "code" in result and "tests" in result
    print("[Ex5] TDD Pipeline Results:")
    print(f"  Tests generated: {len(result['tests'].splitlines())} lines")
    print(f"  Code generated:  {len(result['code'].splitlines())} lines")
    print(f"  Review bugs:     {len(result.get('review', {}).get('bugs', []))}")
    print(f"  Docstring:       {'yes' if result.get('docstring') else 'no'}")


if __name__ == "__main__":
    print("=== Exercise 1: Test-Driven Prompt ===")
    exercise_1()

    print("\n=== Exercise 2: Code Review ===")
    exercise_2()

    print("\n=== Exercise 3: Iterative Refinement ===")
    exercise_3()

    print("\n=== Exercise 4: Docstring Generator ===")
    exercise_4()

    print("\n=== Exercise 5: TDD Pipeline (End-to-End) ===")
    exercise_5()

    print("\nAll exercises completed!")
