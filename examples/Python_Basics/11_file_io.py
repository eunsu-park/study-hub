"""
11 File I/O
===========
Demonstrates file reading/writing, pathlib, CSV handling,
JSON handling, and common file operation patterns.
"""

import csv
import json
import tempfile
from pathlib import Path
from io import StringIO


def basic_file_operations():
    """Read and write text files using context managers."""
    # Create a temporary directory for our demos
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "sample.txt"

        # Writing a file
        with open(filepath, "w") as f:
            f.write("Line 1: Hello, World!\n")
            f.write("Line 2: Python file I/O\n")
            f.write("Line 3: Context managers are great\n")
        print(f"Wrote to: {filepath.name}")

        # Reading entire file
        with open(filepath, "r") as f:
            content = f.read()
        print(f"\nFull content:\n{content}")

        # Reading line by line (memory efficient for large files)
        print("Line by line:")
        with open(filepath, "r") as f:
            for i, line in enumerate(f, 1):
                print(f"  {i}: {line.rstrip()}")

        # Reading all lines into a list
        with open(filepath, "r") as f:
            lines = f.readlines()
        print(f"\nreadlines(): {lines}")

        # Appending to a file
        with open(filepath, "a") as f:
            f.write("Line 4: Appended later\n")

        # writelines()
        extra_lines = ["Line 5: From writelines\n", "Line 6: Last line\n"]
        with open(filepath, "a") as f:
            f.writelines(extra_lines)

        with open(filepath, "r") as f:
            print(f"Final line count: {len(f.readlines())}")


def pathlib_demo():
    """Modern path handling with pathlib."""
    # Current directory
    cwd = Path.cwd()
    print(f"CWD: {cwd}")

    # Path construction
    p = Path("/usr") / "local" / "bin" / "python3"
    print(f"\nPath: {p}")
    print(f"  parent:  {p.parent}")
    print(f"  name:    {p.name}")
    print(f"  stem:    {p.stem}")
    print(f"  suffix:  {p.suffix}")
    print(f"  parts:   {p.parts}")

    # Path with file extension
    src = Path("project/src/main.py")
    print(f"\nPath: {src}")
    print(f"  with_suffix('.pyx'): {src.with_suffix('.pyx')}")
    print(f"  with_name('app.py'): {src.with_name('app.py')}")

    # Practical operations in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create directories
        subdir = base / "project" / "src"
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated: {subdir}")

        # Write and read with pathlib
        config = subdir / "config.txt"
        config.write_text("debug=true\nport=8080\n")
        print(f"Content: {config.read_text()!r}")

        # Check existence and type
        print(f"  exists:  {config.exists()}")
        print(f"  is_file: {config.is_file()}")
        print(f"  is_dir:  {subdir.is_dir()}")
        print(f"  size:    {config.stat().st_size} bytes")

        # Glob pattern matching
        for ext in [".py", ".txt"]:
            (subdir / f"file1{ext}").write_text("")
            (subdir / f"file2{ext}").write_text("")

        txt_files = sorted(subdir.glob("*.txt"))
        print(f"\n  *.txt files: {[f.name for f in txt_files]}")

        all_files = sorted(base.rglob("*.*"))
        print(f"  All files (recursive): {[f.name for f in all_files]}")


def csv_handling():
    """Reading and writing CSV files."""
    # Writing CSV
    data = [
        ["Name", "Age", "City"],
        ["Alice", 30, "New York"],
        ["Bob", 25, "San Francisco"],
        ["Charlie", 35, "Chicago"],
    ]

    # Use StringIO to demonstrate without actual files
    output = StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    csv_text = output.getvalue()
    print(f"CSV output:\n{csv_text}")

    # Reading CSV
    reader = csv.reader(StringIO(csv_text))
    header = next(reader)
    print(f"Header: {header}")
    for row in reader:
        print(f"  {row[0]}: age={row[1]}, city={row[2]}")

    # DictReader/DictWriter — header-based access
    print("\nDictReader:")
    dict_reader = csv.DictReader(StringIO(csv_text))
    for row in dict_reader:
        print(f"  {row}")

    # DictWriter
    output2 = StringIO()
    fieldnames = ["name", "score", "grade"]
    writer = csv.DictWriter(output2, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"name": "Alice", "score": 95, "grade": "A"})
    writer.writerow({"name": "Bob", "score": 82, "grade": "B"})
    print(f"\nDictWriter output:\n{output2.getvalue()}")


def json_handling():
    """Reading and writing JSON data."""
    # Python dict to JSON string
    data = {
        "name": "Alice",
        "age": 30,
        "scores": [95, 88, 92],
        "address": {
            "city": "New York",
            "zip": "10001",
        },
        "active": True,
        "notes": None,
    }

    # Serialize to JSON string
    json_str = json.dumps(data, indent=2)
    print(f"JSON string:\n{json_str}")

    # Deserialize from JSON string
    parsed = json.loads(json_str)
    print(f"\nParsed back: {parsed['name']}, age {parsed['age']}")
    print(f"Scores: {parsed['scores']}")

    # Compact JSON (for APIs/storage)
    compact = json.dumps(data, separators=(",", ":"))
    print(f"\nCompact: {compact}")

    # Custom serialization
    from datetime import datetime, date

    class DateEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            return super().default(obj)

    event = {"title": "Meeting", "date": date(2025, 3, 15)}
    print(f"\nCustom encoder: {json.dumps(event, cls=DateEncoder)}")

    # File read/write with json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2)
        tmp_path = f.name

    with open(tmp_path, "r") as f:
        loaded = json.load(f)
    print(f"\nLoaded from file: {loaded['name']}")

    Path(tmp_path).unlink()  # Clean up


def binary_files():
    """Working with binary data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        binpath = Path(tmpdir) / "data.bin"

        # Write binary data
        data = bytes(range(256))
        with open(binpath, "wb") as f:
            f.write(data)
        print(f"Wrote {len(data)} bytes")

        # Read binary data
        with open(binpath, "rb") as f:
            content = f.read()
        print(f"Read {len(content)} bytes")
        print(f"First 16 bytes: {content[:16].hex()}")

        # Struct for structured binary data
        import struct
        binpath2 = Path(tmpdir) / "record.bin"

        # Pack: int(4), float(4), char[10]
        record = struct.pack("if10s", 42, 3.14, b"hello")
        with open(binpath2, "wb") as f:
            f.write(record)

        with open(binpath2, "rb") as f:
            raw = f.read()
        unpacked = struct.unpack("if10s", raw)
        print(f"\nStruct: int={unpacked[0]}, float={unpacked[1]:.2f}, "
              f"str={unpacked[2].rstrip(b'\\x00').decode()}")


def file_best_practices():
    """Common patterns and best practices."""
    # 1. Always use context managers
    print("1. Context managers ensure files are closed:")
    print("   with open('file.txt') as f:")
    print("       data = f.read()")

    # 2. Specify encoding explicitly
    print("\n2. Always specify encoding:")
    print("   open('file.txt', encoding='utf-8')")

    # 3. Use pathlib over os.path
    print("\n3. Prefer pathlib:")
    print("   Path('file.txt').read_text()")
    print("   vs os.path.join('dir', 'file.txt')")

    # 4. Handle missing files gracefully
    print("\n4. Handle FileNotFoundError:")
    try:
        Path("/nonexistent/file.txt").read_text()
    except FileNotFoundError as e:
        print(f"   Caught: {e}")

    # 5. Atomic writes (write to temp, then rename)
    print("\n5. Atomic writes prevent corruption:")
    print("   Write to temp file, then os.replace() to target")


if __name__ == "__main__":
    sections = [
        ("Basic File Operations", basic_file_operations),
        ("Pathlib Demo", pathlib_demo),
        ("CSV Handling", csv_handling),
        ("JSON Handling", json_handling),
        ("Binary Files", binary_files),
        ("Best Practices", file_best_practices),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
