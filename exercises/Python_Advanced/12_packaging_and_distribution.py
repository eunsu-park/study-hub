"""
Exercises for Lesson 12: Packaging and Distribution
Topic: Python

Solutions to practice problems from the lesson.

Note: Exercises 1 and 2 involve project configuration files (pyproject.toml, Poetry).
Those are demonstrated here as string templates and validation logic.
Exercise 3 implements a working CLI tool using argparse.
"""

import argparse
import sys
import json


# === Exercise 1: Write pyproject.toml ===
# Problem: Write pyproject.toml for a simple utility package.

PYPROJECT_TOML = """\
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "myutils"
version = "0.1.0"
description = "My utility functions"
readme = "README.md"
requires-python = ">=3.9"
authors = [{name = "Your Name", email = "you@example.com"}]
license = {text = "MIT"}
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=7.0", "black", "mypy"]

[tool.setuptools.packages.find]
where = ["src"]
"""


def validate_pyproject(content: str) -> list[str]:
    """Validate that a pyproject.toml string contains required sections.

    Returns a list of issues found. An empty list means the file is valid.
    This checks for structural requirements, not TOML syntax.
    """
    issues = []
    required_sections = [
        "[build-system]",
        "[project]",
    ]
    required_fields = [
        'name =',
        'version =',
        'requires-python',
        'build-backend',
    ]

    for section in required_sections:
        if section not in content:
            issues.append(f"Missing section: {section}")

    for field in required_fields:
        if field not in content:
            issues.append(f"Missing field: {field}")

    return issues


def exercise_1():
    """Demonstrate pyproject.toml creation and validation."""
    print("Generated pyproject.toml:")
    print("-" * 40)
    print(PYPROJECT_TOML)
    print("-" * 40)

    issues = validate_pyproject(PYPROJECT_TOML)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Validation: All required sections and fields present.")

    # Test with a bad pyproject.toml
    bad_toml = "[project]\nname = 'incomplete'"
    issues = validate_pyproject(bad_toml)
    print(f"\nBad pyproject.toml has {len(issues)} issues:")
    for issue in issues:
        print(f"  - {issue}")


# === Exercise 2: Poetry Project Setup ===
# Problem: Create a new project with Poetry and manage dependencies.

POETRY_COMMANDS = """
# Step 1: Create a new project with src layout
poetry new myproject --src
cd myproject

# Step 2: Add production dependencies
poetry add requests pydantic

# Step 3: Add development dependencies
poetry add --group dev pytest black mypy

# Step 4: Lock dependencies and install
poetry lock
poetry install

# Step 5: Build distributable packages
poetry build
# Creates dist/myproject-0.1.0.tar.gz and dist/myproject-0.1.0-py3-none-any.whl
"""


def simulate_poetry_project() -> dict:
    """Simulate the structure of a Poetry-managed project.

    Returns a dict representing the project layout, which is what
    `poetry new myproject --src` would create.
    """
    return {
        "myproject/": {
            "src/": {
                "myproject/": {
                    "__init__.py": '"""My project."""\n__version__ = "0.1.0"',
                },
            },
            "tests/": {
                "__init__.py": "",
                "test_myproject.py": (
                    "from myproject import __version__\n\n"
                    "def test_version():\n"
                    '    assert __version__ == "0.1.0"\n'
                ),
            },
            "pyproject.toml": "# Poetry-generated pyproject.toml",
            "README.md": "# myproject\n",
        }
    }


def exercise_2():
    """Demonstrate Poetry project setup commands and structure."""
    print("Poetry setup commands:")
    print(POETRY_COMMANDS)

    project = simulate_poetry_project()
    print("Project structure after `poetry new myproject --src`:")

    def print_tree(tree, indent=0):
        for name, content in sorted(tree.items()):
            prefix = "  " * indent
            if isinstance(content, dict):
                print(f"{prefix}{name}")
                print_tree(content, indent + 1)
            else:
                print(f"{prefix}{name}")

    print_tree(project)


# === Exercise 3: CLI Entry Point ===
# Problem: Create a CLI tool and register it in pyproject.toml.

def create_greeting_cli():
    """Create an argparse-based CLI tool.

    This function returns the parser so it can be tested without
    actually calling parse_args() on sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="A friendly greeting CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  greet Alice
  greet Alice -g "Good morning"
  greet Alice -g "Hola" --repeat 3
  greet Alice --json
""",
    )
    parser.add_argument("name", help="Name of the person to greet")
    parser.add_argument(
        "-g", "--greeting",
        default="Hello",
        help="Greeting phrase (default: Hello)",
    )
    parser.add_argument(
        "-r", "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the greeting (default: 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    return parser


def greet_main(args=None):
    """Entry point for the CLI tool.

    This would be registered in pyproject.toml as:
      [project.scripts]
      greet = "mypackage.cli:greet_main"
    """
    parser = create_greeting_cli()
    parsed = parser.parse_args(args)

    message = f"{parsed.greeting}, {parsed.name}!"

    if parsed.json:
        output = json.dumps({
            "greeting": parsed.greeting,
            "name": parsed.name,
            "message": message,
            "repeat": parsed.repeat,
        }, indent=2)
        print(output)
    else:
        for _ in range(parsed.repeat):
            print(message)


ENTRY_POINT_TOML = """\
# In pyproject.toml, register the CLI entry point:
[project.scripts]
greet = "mypackage.cli:greet_main"

# After installation (pip install -e .), the 'greet' command is available:
#   greet Alice           -> "Hello, Alice!"
#   greet Bob -g "Hi"     -> "Hi, Bob!"
"""


def exercise_3():
    """Demonstrate CLI tool with argparse."""
    print("Entry point configuration:")
    print(ENTRY_POINT_TOML)

    print("Running CLI with different arguments:\n")

    print('  $ greet Alice')
    greet_main(["Alice"])

    print('\n  $ greet Bob -g "Good morning"')
    greet_main(["Bob", "-g", "Good morning"])

    print('\n  $ greet Charlie --repeat 3')
    greet_main(["Charlie", "--repeat", "3"])

    print('\n  $ greet Dana --json')
    greet_main(["Dana", "--json"])


if __name__ == "__main__":
    print("=== Exercise 1: Write pyproject.toml ===")
    exercise_1()

    print("\n=== Exercise 2: Poetry Project Setup ===")
    exercise_2()

    print("\n=== Exercise 3: CLI Entry Point ===")
    exercise_3()

    print("\nAll exercises completed!")
