"""
10 Modules and Packages
=======================
Demonstrates module creation, the __name__ == '__main__' pattern,
import mechanics, and package organization concepts.
"""

import sys
import os
import importlib
import math


def module_basics():
    """How Python modules work."""
    # Every .py file is a module
    print(f"This file's module name: {__name__}")
    print(f"This file's path: {__file__}")

    # Importing a module gives access to its attributes
    print(f"\nmath.pi = {math.pi}")
    print(f"math.sqrt(2) = {math.sqrt(2):.6f}")

    # Module attributes
    print(f"\nmath module attributes (first 10):")
    attrs = [a for a in dir(math) if not a.startswith("_")]
    for i in range(0, min(10, len(attrs)), 2):
        pair = f"  {attrs[i]:<20}"
        if i + 1 < len(attrs):
            pair += f" {attrs[i+1]:<20}"
        print(pair)


def import_patterns():
    """Different ways to import."""
    # 1. import module
    import json
    data = json.dumps({"key": "value"})
    print(f"json.dumps: {data}")

    # 2. from module import names
    from collections import OrderedDict, defaultdict
    od = OrderedDict(a=1, b=2)
    print(f"OrderedDict: {od}")

    # 3. import with alias
    from datetime import datetime as dt
    now = dt(2025, 3, 15, 14, 30)
    print(f"datetime: {now}")

    # 4. Conditional import (try alternative implementations)
    try:
        import ujson as json_impl
    except ImportError:
        import json as json_impl
    print(f"JSON implementation: {json_impl.__name__}")

    # 5. Lazy import inside function (defers loading)
    def process_csv(text):
        import csv
        from io import StringIO
        reader = csv.reader(StringIO(text))
        return list(reader)

    rows = process_csv("a,b,c\n1,2,3")
    print(f"CSV parsed: {rows}")


def name_guard_pattern():
    """The __name__ == '__main__' pattern explained."""
    print("When a file is run directly:")
    print(f"  __name__ = '__main__'")
    print()
    print("When a file is imported:")
    print(f"  __name__ = '<module_name>'")
    print()
    print("This pattern prevents code from running on import:")
    print()
    print('  def main():')
    print('      print("This runs only when executed directly")')
    print()
    print('  if __name__ == "__main__":')
    print('      main()')

    # Demonstrate with a simulated module
    print(f"\nCurrent __name__: {__name__!r}")


def module_search_path():
    """How Python finds modules to import."""
    print("Module search path (sys.path):")
    for i, path in enumerate(sys.path[:8]):
        marker = " <-- current dir" if path == "" or path == os.getcwd() else ""
        print(f"  [{i}] {path or '(empty = cwd)'}{marker}")
    if len(sys.path) > 8:
        print(f"  ... and {len(sys.path) - 8} more entries")

    # Where is a module located?
    print(f"\nos module:   {os.__file__}")
    print(f"json module: {importlib.import_module('json').__file__}")

    # sys.modules cache
    loaded = [m for m in sorted(sys.modules.keys()) if not m.startswith("_")]
    print(f"\nLoaded modules (first 15): {loaded[:15]}")


def creating_a_module():
    """Demonstrate creating and using a simple module dynamically."""
    import types

    # Create a module object programmatically
    mymodule = types.ModuleType("mymodule")
    mymodule.__doc__ = "A dynamically created module."
    mymodule.VERSION = "1.0.0"
    mymodule.greet = lambda name: f"Hello, {name}!"
    mymodule.PI = 3.14159

    print(f"Module: {mymodule}")
    print(f"Doc:    {mymodule.__doc__}")
    print(f"Version: {mymodule.VERSION}")
    print(f"greet:  {mymodule.greet('World')}")

    # Register in sys.modules (makes it importable)
    sys.modules["mymodule"] = mymodule
    import mymodule as mm  # noqa: F811
    print(f"Imported: {mm.PI}")

    # Clean up
    del sys.modules["mymodule"]


def package_structure():
    """Explain Python package organization."""
    structure = """
    Typical package layout:

    mypackage/
    ├── __init__.py          # Makes it a package; runs on import
    ├── module_a.py          # import mypackage.module_a
    ├── module_b.py          # from mypackage import module_b
    └── subpackage/
        ├── __init__.py
        └── module_c.py      # from mypackage.subpackage import module_c

    __init__.py can:
    - Be empty (just marks directory as package)
    - Define __all__ for 'from package import *'
    - Import submodules for convenience
    - Set package-level variables
    """
    print(structure)

    # Demonstrate __all__ concept
    print("  __all__ controls 'from module import *':")
    print("    __all__ = ['public_func', 'PublicClass']")
    print("    Excludes: _private_func, _InternalClass")

    # Namespace packages (Python 3.3+)
    print("\n  Namespace packages (no __init__.py):")
    print("    Allow splitting a package across directories")
    print("    Used by large frameworks for plugin systems")


def reload_and_introspection():
    """Module reloading and introspection tools."""
    import json

    # Module introspection
    print(f"Module name: {json.__name__}")
    print(f"Module file: {json.__file__}")
    print(f"Module doc:  {json.__doc__[:60]}...")

    # Checking if something is importable
    def is_importable(module_name):
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    modules_to_check = ["os", "sys", "numpy", "nonexistent_module"]
    print("\nImportability check:")
    for mod in modules_to_check:
        print(f"  {mod:<25} {'available' if is_importable(mod) else 'not found'}")

    # Reloading a module (useful during development)
    print("\n  importlib.reload(module) — reloads a previously imported module")
    print("  Useful in interactive sessions, but avoid in production")


if __name__ == "__main__":
    sections = [
        ("Module Basics", module_basics),
        ("Import Patterns", import_patterns),
        ("__name__ Guard Pattern", name_guard_pattern),
        ("Module Search Path", module_search_path),
        ("Creating a Module", creating_a_module),
        ("Package Structure", package_structure),
        ("Reload & Introspection", reload_and_introspection),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
