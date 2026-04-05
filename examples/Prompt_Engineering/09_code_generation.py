# 09_code_generation.py — Test-driven prompting and debugging prompts
#
# Run: python 09_code_generation.py

"""
Demonstrates:
  1. Test-driven prompting  — provide tests first, let the model implement
  2. Debugging prompts      — feed buggy code + error, get a fix
  3. Code review prompting  — systematic review with categories
  4. Iterative refinement   — generate → test locally → fix loop
"""

import os
import textwrap
import subprocess
import sys
import tempfile

import anthropic

client: anthropic.Anthropic


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude(prompt: str, system: str = "") -> str:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def extract_python_code(text: str) -> str:
    """Extract the first Python code block from a response."""
    import re
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute Python code in a subprocess, return (success, output)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: Code took too long to execute."
        finally:
            os.unlink(f.name)


# ---------------------------------------------------------------------------
# 1. Test-Driven Prompting
# ---------------------------------------------------------------------------

TEST_SUITE = '''\
def test_merge_intervals():
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 3]]) == [[1, 3]]
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]
    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]
    print("All tests passed!")

test_merge_intervals()
'''


def demo_test_driven():
    """Give the model tests first, ask it to write the implementation."""

    system = (
        "You are an expert Python developer. Write clean, efficient code. "
        "Return ONLY the implementation in a Python code block."
    )

    prompt = (
        "Write a function `merge_intervals(intervals)` that merges "
        "overlapping intervals. Each interval is [start, end].\n\n"
        "The function must pass these tests:\n"
        f"```python\n{TEST_SUITE}```\n\n"
        "Return only the function implementation in a ```python``` block."
    )

    print("=" * 60)
    print("SECTION 1 — Test-Driven Prompting")
    print("=" * 60)

    response = call_claude(prompt, system=system)
    code = extract_python_code(response)
    print(f"\n[Generated Code]\n{textwrap.indent(code, '  ')}")

    # Run the generated code with the test suite
    full_code = code + "\n\n" + TEST_SUITE
    success, output = run_code(full_code)
    print(f"\n[Test Result] {'PASS' if success else 'FAIL'}")
    print(f"  {output}")


# ---------------------------------------------------------------------------
# 2. Debugging Prompts
# ---------------------------------------------------------------------------

BUGGY_CODE = '''\
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1

# Bug: binary_search([1, 2, 3, 4, 5], 2) enters infinite loop
'''


def demo_debugging():
    """Feed buggy code and error description to get a fix."""

    prompt = (
        "This Python function has a bug that causes an infinite loop:\n\n"
        f"```python\n{BUGGY_CODE}```\n\n"
        "Diagnose the bug step by step:\n"
        "1. Identify the exact line(s) causing the issue\n"
        "2. Explain WHY it causes an infinite loop\n"
        "3. Provide the corrected code in a ```python``` block\n"
        "4. Add a comment on the fixed line(s) explaining the fix"
    )

    print("\n" + "=" * 60)
    print("SECTION 2 — Debugging Prompts")
    print("=" * 60)

    response = call_claude(prompt)
    print(f"\n{response}")

    # Verify the fix
    fixed_code = extract_python_code(response)
    test_code = fixed_code + "\n" + (
        "assert binary_search([1, 2, 3, 4, 5], 2) == 1\n"
        "assert binary_search([1, 2, 3, 4, 5], 5) == 4\n"
        "assert binary_search([1, 2, 3, 4, 5], 6) == -1\n"
        "print('All debug tests passed!')\n"
    )
    success, output = run_code(test_code)
    print(f"\n[Verification] {'PASS' if success else 'FAIL'}: {output}")


# ---------------------------------------------------------------------------
# 3. Code Review Prompting
# ---------------------------------------------------------------------------

CODE_TO_REVIEW = '''\
import os, sys, json

def process(d):
    r = []
    for k in d:
        v = d[k]
        if type(v) == dict:
            for k2 in v:
                r.append(k + "." + k2 + "=" + str(v[k2]))
        else:
            r.append(k + "=" + str(v))
    return r

data = json.loads(open(sys.argv[1]).read())
print("\\n".join(process(data)))
'''


def demo_code_review():
    """Structured code review with categorized feedback."""

    system = "You are a senior Python developer conducting a code review."
    prompt = (
        f"Review this code:\n```python\n{CODE_TO_REVIEW}```\n\n"
        "Provide feedback in these categories:\n"
        "1. **Bugs** — actual errors or incorrect behavior\n"
        "2. **Security** — vulnerabilities or unsafe patterns\n"
        "3. **Style** — PEP 8 violations, naming, readability\n"
        "4. **Performance** — inefficiencies\n"
        "5. **Best Practices** — missing error handling, type hints, etc.\n\n"
        "Then provide a refactored version in a ```python``` block."
    )

    print("\n" + "=" * 60)
    print("SECTION 3 — Code Review Prompting")
    print("=" * 60)
    print(call_claude(prompt, system=system))


# ---------------------------------------------------------------------------
# 4. Iterative Generate-Test-Fix Loop
# ---------------------------------------------------------------------------

def demo_iterative_refinement(max_attempts: int = 3):
    """Generate code, test it locally, send errors back for fixing."""

    system = "You are an expert Python developer. Return ONLY code in a ```python``` block."
    task = (
        "Write a function `roman_to_int(s)` that converts a Roman numeral "
        "string to an integer. Handle subtractive notation (IV=4, IX=9, etc.)."
    )
    tests = (
        "assert roman_to_int('III') == 3\n"
        "assert roman_to_int('IV') == 4\n"
        "assert roman_to_int('IX') == 9\n"
        "assert roman_to_int('XLII') == 42\n"
        "assert roman_to_int('MCMXCIV') == 1994\n"
        "print('All roman numeral tests passed!')\n"
    )

    print("\n" + "=" * 60)
    print(f"SECTION 4 — Iterative Refinement (max {max_attempts} attempts)")
    print("=" * 60)

    prompt = f"{task}\n\nMust pass:\n```python\n{tests}```"

    for attempt in range(1, max_attempts + 1):
        response = call_claude(prompt, system=system)
        code = extract_python_code(response)
        full_code = code + "\n\n" + tests

        success, output = run_code(full_code)
        print(f"\n  Attempt {attempt}: {'PASS' if success else 'FAIL'}")
        print(f"  Output: {output[:200]}")

        if success:
            print(f"\n[Final Code]\n{textwrap.indent(code, '  ')}")
            break

        # Feed the error back
        prompt = (
            f"Your code:\n```python\n{code}```\n\n"
            f"Error:\n{output}\n\n"
            f"Fix the code. Return ONLY the corrected ```python``` block."
        )
    else:
        print(f"\n  Failed after {max_attempts} attempts.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_test_driven()
        demo_debugging()
        demo_code_review()
        demo_iterative_refinement()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
