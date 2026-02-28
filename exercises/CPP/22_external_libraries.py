"""
Exercises for Lesson 22: External Libraries
Topic: CPP

Solutions to practice problems covering JSON configuration,
formatting patterns, logging concepts, async I/O patterns,
and CMake integration — demonstrated in Python equivalents.
"""

import json
import time
import logging
from collections import defaultdict


# ============================================================
# Exercise 1: JSON Configuration System
# ============================================================
def exercise_1():
    """
    Build a configuration system similar to nlohmann/json:
    load, validate, merge, serialize.
    """
    print("=== Exercise 1: JSON Configuration System ===\n")

    class Config:
        """Configuration system with validation and merging."""

        def __init__(self, data=None):
            self._data = data or {}

        @classmethod
        def from_json_string(cls, json_str):
            try:
                data = json.loads(json_str)
                return cls(data), None
            except json.JSONDecodeError as e:
                return None, f"ParseError: {e}"

        def get(self, path, default=None):
            """Access nested value with dot-separated path."""
            keys = path.split(".")
            current = self._data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current

        def set(self, path, value):
            """Set nested value with dot-separated path."""
            keys = path.split(".")
            current = self._data
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

        def validate(self, schema):
            """Validate against a schema (dict of required fields + types)."""
            errors = []
            for field, expected_type in schema.items():
                value = self.get(field)
                if value is None:
                    errors.append(f"Missing: {field}")
                elif not isinstance(value, expected_type):
                    errors.append(f"Type error: {field} expected "
                                  f"{expected_type.__name__}, got "
                                  f"{type(value).__name__}")
            return errors

        def merge(self, overrides):
            """Merge overrides (higher priority) into config."""
            def _merge(base, over):
                result = dict(base)
                for k, v in over.items():
                    if (k in result and isinstance(result[k], dict)
                            and isinstance(v, dict)):
                        result[k] = _merge(result[k], v)
                    else:
                        result[k] = v
                return result

            if isinstance(overrides, Config):
                self._data = _merge(self._data, overrides._data)
            else:
                self._data = _merge(self._data, overrides)

        def dump(self, indent=2):
            return json.dumps(self._data, indent=indent)

    # Defaults
    defaults = Config({
        "server": {"host": "localhost", "port": 8080, "tls": False},
        "database": {"url": "sqlite:///app.db", "pool_size": 5},
        "logging": {"level": "INFO", "file": "app.log"},
    })

    # User overrides
    overrides = Config({
        "server": {"port": 443, "tls": True},
        "database": {"url": "postgres://prod:5432/app"},
    })

    # Merge
    defaults.merge(overrides)

    print(f"  Merged config:")
    print(f"    server.host = {defaults.get('server.host')}")
    print(f"    server.port = {defaults.get('server.port')} (overridden)")
    print(f"    server.tls = {defaults.get('server.tls')} (overridden)")
    print(f"    database.url = {defaults.get('database.url')} (overridden)")
    print(f"    database.pool_size = {defaults.get('database.pool_size')} (default)")
    print(f"    logging.level = {defaults.get('logging.level')} (default)")

    # Validate
    schema = {
        "server.host": str,
        "server.port": int,
        "database.url": str,
    }
    errors = defaults.validate(schema)
    print(f"\n  Validation: {'PASS' if not errors else errors}")

    # Parse invalid JSON
    _, err = Config.from_json_string("{bad json")
    print(f"  Invalid JSON: {err}")
    print()


# ============================================================
# Exercise 2: Custom Formatter (fmt pattern)
# ============================================================
def exercise_2():
    """
    Build a custom formatting system similar to fmt library.
    """
    print("=== Exercise 2: Custom Formatter (fmt) ===\n")

    class Formatter:
        """Simplified fmt-style formatter with custom type support."""

        def __init__(self):
            self._formatters = {}

        def register(self, type_cls, format_fn):
            """Register a custom formatter for a type."""
            self._formatters[type_cls] = format_fn

        def format(self, template, *args, **kwargs):
            """Format string with positional and named arguments."""
            # Handle named arguments
            result = template
            for key, val in kwargs.items():
                placeholder = "{" + key + "}"
                result = result.replace(placeholder, self._format_value(val))

            # Handle positional arguments
            for i, arg in enumerate(args):
                result = result.replace("{}", self._format_value(arg), 1)

            return result

        def _format_value(self, val):
            if type(val) in self._formatters:
                return self._formatters[type(val)](val)
            return str(val)

    # Custom types
    class Matrix:
        def __init__(self, data):
            self.data = data

    class Duration:
        def __init__(self, seconds):
            self.seconds = seconds

    class HexColor:
        def __init__(self, r, g, b):
            self.r, self.g, self.b = r, g, b

    fmt = Formatter()

    # Register custom formatters
    fmt.register(Matrix, lambda m:
        "\n    " + "\n    ".join(
            "  ".join(f"{v:6.1f}" for v in row) for row in m.data))

    fmt.register(Duration, lambda d:
        f"{d.seconds // 3600}h {(d.seconds % 3600) // 60}m {d.seconds % 60}s")

    fmt.register(HexColor, lambda c:
        f"#{c.r:02X}{c.g:02X}{c.b:02X}")

    # Test formatting
    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    d = Duration(9015)
    c = HexColor(255, 128, 0)

    print(f"  Matrix: {fmt._format_value(m)}")
    print(f"\n  Duration: {fmt._format_value(d)}")
    print(f"  Color: {fmt._format_value(c)}")

    # Template formatting
    result = fmt.format("User {} scored {} points", "Alice", 95)
    print(f"\n  Template: {result}")

    result = fmt.format("Name: {name}, Score: {score}",
                        name="Bob", score=87)
    print(f"  Named: {result}")
    print()


# ============================================================
# Exercise 3: Structured Logging (spdlog pattern)
# ============================================================
def exercise_3():
    """
    Implement structured logging similar to spdlog patterns.
    """
    print("=== Exercise 3: Structured Logging ===\n")

    class StructuredLogger:
        """Structured logger with level filtering and formatting."""

        LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}

        def __init__(self, name, level="INFO"):
            self.name = name
            self.min_level = self.LEVELS[level]
            self.entries = []

        def _log(self, level, msg, **kwargs):
            if self.LEVELS[level] < self.min_level:
                return
            entry = {
                "time": time.strftime("%H:%M:%S"),
                "level": level,
                "logger": self.name,
                "message": msg,
                **kwargs,
            }
            self.entries.append(entry)

            # Format output
            kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
            level_str = f"[{level:>5}]"
            print(f"    {entry['time']} {level_str} [{self.name}] {msg} {kv}")

        def debug(self, msg, **kw): self._log("DEBUG", msg, **kw)
        def info(self, msg, **kw): self._log("INFO", msg, **kw)
        def warn(self, msg, **kw): self._log("WARN", msg, **kw)
        def error(self, msg, **kw): self._log("ERROR", msg, **kw)

    # Create loggers
    app_log = StructuredLogger("app", level="DEBUG")
    db_log = StructuredLogger("db", level="INFO")

    print("  Application log (level=DEBUG):")
    app_log.debug("Initializing components")
    app_log.info("Server started", port=8080, tls=True)
    app_log.warn("High memory usage", pct=87.5)
    app_log.error("Connection failed", host="db.example.com", retry=3)

    print(f"\n  Database log (level=INFO, DEBUG filtered):")
    db_log.debug("This won't appear")
    db_log.info("Pool initialized", size=10, driver="postgres")
    db_log.info("Query executed", table="users", rows=42, ms=3.2)

    print(f"\n  Total entries: app={len(app_log.entries)}, db={len(db_log.entries)}")
    print()


# ============================================================
# Exercise 4: Async I/O Pattern (Boost.Asio concept)
# ============================================================
def exercise_4():
    """
    Demonstrate async I/O patterns similar to Boost.Asio
    using Python's event loop simulation.
    """
    print("=== Exercise 4: Async I/O Pattern ===\n")

    class EventLoop:
        """Simplified event loop (Boost.Asio io_context equivalent)."""

        def __init__(self):
            self.handlers = []
            self.time = 0

        def post(self, delay, callback, *args):
            """Schedule a callback after delay."""
            self.handlers.append((self.time + delay, callback, args))
            self.handlers.sort(key=lambda x: x[0])

        def run(self):
            """Process all scheduled handlers."""
            while self.handlers:
                event_time, callback, args = self.handlers.pop(0)
                self.time = event_time
                callback(*args)

    class AsyncSocket:
        """Simulated async socket (Boost.Asio tcp::socket equivalent)."""

        def __init__(self, io, name):
            self.io = io
            self.name = name
            self.buffer = []

        def async_read(self, callback):
            """Schedule an async read (simulated with delay)."""
            data = f"data_from_{self.name}"
            self.io.post(1, callback, data, None)  # 1ms simulated latency

        def async_write(self, data, callback):
            """Schedule an async write."""
            self.io.post(1, callback, len(data), None)

    # Simulate an echo server handling 3 clients
    io = EventLoop()

    def handle_read(socket, data, error):
        if error:
            print(f"    [{io.time}ms] {socket.name}: error {error}")
            return
        print(f"    [{io.time}ms] {socket.name}: read '{data}'")
        # Echo back
        socket.async_write(data, lambda n, e:
            print(f"    [{io.time}ms] {socket.name}: wrote {n} bytes"))

    # Accept 3 clients
    clients = [AsyncSocket(io, f"client_{i}") for i in range(3)]

    print("  Simulated async echo server:")
    for client in clients:
        client.async_read(lambda data, err, s=client: handle_read(s, data, err))

    io.run()

    print(f"\n  Boost.Asio equivalent:")
    print(f"    asio::awaitable<void> echo(tcp::socket sock) {{")
    print(f"        auto n = co_await sock.async_read_some(buf, use_awaitable);")
    print(f"        co_await asio::async_write(sock, buf, use_awaitable);")
    print(f"    }}")
    print()


# ============================================================
# Exercise 5: CMake Integration Analysis
# ============================================================
def exercise_5():
    """
    Analyze CMake patterns for integrating external libraries.
    Compare find_package, FetchContent, and package managers.
    """
    print("=== Exercise 5: CMake Integration Patterns ===\n")

    patterns = {
        "find_package": {
            "cmake": 'find_package(fmt REQUIRED)\ntarget_link_libraries(app PRIVATE fmt::fmt)',
            "install": "System package manager or vcpkg",
            "pros": ["Fast configure", "System-wide caching", "Version flexibility"],
            "cons": ["Must pre-install", "Platform differences"],
            "best_for": "CI/CD, large projects",
        },
        "FetchContent": {
            "cmake": 'FetchContent_Declare(fmt GIT_REPOSITORY ... GIT_TAG 10.2.1)\nFetchContent_MakeAvailable(fmt)',
            "install": "Automatic at configure time",
            "pros": ["No pre-install needed", "Reproducible", "Simple"],
            "cons": ["Slower first configure", "Downloads each time"],
            "best_for": "Quick prototypes, small projects",
        },
        "vcpkg manifest": {
            "cmake": 'cmake -DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake ..',
            "install": 'vcpkg.json: {"dependencies": ["fmt"]}',
            "pros": ["Binary caching", "2500+ packages", "Declarative"],
            "cons": ["vcpkg setup required", "Limited custom builds"],
            "best_for": "Production C++ projects",
        },
        "Conan": {
            "cmake": 'conanfile.txt + CMakeDeps generator',
            "install": "pip install conan && conan install .",
            "pros": ["Cross-platform", "Profile system", "Remote hosting"],
            "cons": ["Python dependency", "Learning curve"],
            "best_for": "Cross-platform libraries, embedded",
        },
    }

    for name, info in patterns.items():
        print(f"  {name}:")
        print(f"    CMake: {info['cmake']}")
        print(f"    Install: {info['install']}")
        print(f"    Pros: {', '.join(info['pros'])}")
        print(f"    Cons: {', '.join(info['cons'])}")
        print(f"    Best for: {info['best_for']}")
        print()

    # Decision matrix
    print("  Decision Matrix:")
    print(f"  {'Scenario':>30} | {'Recommended':>15}")
    print(f"  {'-'*48}")
    scenarios = [
        ("CI/CD pipeline", "vcpkg manifest"),
        ("Quick prototype", "FetchContent"),
        ("Cross-platform lib", "Conan"),
        ("Header-only lib", "FetchContent"),
        ("System dependency (OpenSSL)", "find_package"),
        ("Large monorepo", "vcpkg manifest"),
    ]
    for scenario, rec in scenarios:
        print(f"  {scenario:>30} | {rec:>15}")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
