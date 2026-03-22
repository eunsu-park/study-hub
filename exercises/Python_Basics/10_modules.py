"""
Exercise 10: Modules and Packages

Practice module creation, import patterns, and package structure.
All exercises are self-contained in this file for simplicity.
"""


def create_calculator():
    """Create and return a calculator 'module' as a dict.

    The dict should have these keys mapping to functions:
        "add": (a, b) -> a + b
        "subtract": (a, b) -> a - b
        "multiply": (a, b) -> a * b
        "divide": (a, b) -> a / b (raise ZeroDivisionError if b == 0)
        "history": () -> list of (operation_name, a, b, result) tuples

    Each operation should record itself in the history.

    Returns:
        Dict of calculator functions.
    """
    # TODO: Implement with a shared history list via closure
    pass


def describe_module(module):
    """Return a dict describing a Python module's public interface.

    Keys:
        "name": module.__name__
        "functions": sorted list of callable attribute names (no _ prefix)
        "constants": sorted list of UPPER_CASE attribute names

    Args:
        module: A Python module object.

    Returns:
        Dict with module description.
    """
    # TODO: Implement this using dir() and attribute inspection
    pass


def lazy_import(module_name):
    """Import a module by name string and return it.

    Use importlib.import_module for dynamic import.

    Args:
        module_name: String name of module (e.g., "math", "json").

    Returns:
        The imported module.

    Raises:
        ImportError: If module cannot be found.
    """
    # TODO: Implement using importlib
    pass


def get_module_version(module_name):
    """Return the version string of an installed module.

    Try module.__version__ first, fall back to
    importlib.metadata.version(). Return "unknown" if neither works.

    Args:
        module_name: String name of module.

    Returns:
        Version string or "unknown".
    """
    # TODO: Implement this
    pass


def simulate_package_structure():
    """Return a dict representing a package directory structure.

    Simulate this package layout:
        mypackage/
            __init__.py  (exports: version = "1.0.0")
            utils.py     (exports: helper function)
            models/
                __init__.py
                user.py  (exports: User class name)

    Returns:
        A nested dict like:
        {
            "name": "mypackage",
            "version": "1.0.0",
            "modules": ["utils", "models"],
            "models_submodules": ["user"],
        }
    """
    # TODO: Return the dict described above
    pass


# === Tests ===

# Calculator module
calc = create_calculator()
assert calc["add"](2, 3) == 5, "Add"
assert calc["subtract"](10, 4) == 6, "Subtract"
assert calc["multiply"](3, 7) == 21, "Multiply"
assert calc["divide"](15, 3) == 5.0, "Divide"
try:
    calc["divide"](1, 0)
    assert False, "Should raise ZeroDivisionError"
except ZeroDivisionError:
    pass
hist = calc["history"]()
assert len(hist) == 4, "History has 4 entries"
assert hist[0] == ("add", 2, 3, 5), "First history entry"

# Module description
import math
desc = describe_module(math)
assert desc["name"] == "math", "Math module name"
assert "sqrt" in desc["functions"], "sqrt in functions"
assert "pi" not in desc["functions"], "pi is not a function"

# Lazy import
m = lazy_import("json")
assert hasattr(m, "dumps"), "json has dumps"
try:
    lazy_import("nonexistent_module_xyz")
    assert False, "Should raise ImportError"
except (ImportError, ModuleNotFoundError):
    pass

# Package structure
pkg = simulate_package_structure()
assert pkg["name"] == "mypackage", "Package name"
assert pkg["version"] == "1.0.0", "Package version"
assert "utils" in pkg["modules"], "Has utils module"
assert "models" in pkg["modules"], "Has models subpackage"
assert "user" in pkg["models_submodules"], "Has user submodule"

print("All tests passed!")
