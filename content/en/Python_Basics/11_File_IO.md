# File I/O

**Previous**: [Modules and Packages](./10_Modules_and_Packages.md) | **Next**: [Exception Handling](./12_Exception_Handling.md)

> **Topic**: Python Basics
> **Lesson**: 11 of 14
> **Prerequisites**: Functions, Strings, Modules and Packages

## Learning Objectives

After completing this lesson, you will be able to:

1. Open, read, and write text files using `open()` with various modes (`r`, `w`, `a`, `rb`, `wb`)
2. Use the `with` statement to ensure proper file resource management
3. Read files line by line, in chunks, and all at once, choosing the right approach for each situation
4. Work with file paths using the `pathlib.Path` class for cross-platform compatibility
5. Parse and write CSV files using the `csv` module
6. Serialize and deserialize data with the `json` module (`dumps`, `loads`, `dump`, `load`)
7. Perform file system operations such as checking existence, listing directories, and copying files
8. Apply common file I/O patterns including config file reading and log file writing

---

## Introduction

Almost every real-world program needs to interact with files — reading configuration, processing data, writing logs, saving user data, or exchanging information with other systems. Python makes file I/O straightforward with built-in functions and a rich standard library.

This lesson covers everything from basic file reading and writing to working with structured formats like CSV and JSON, and navigating the file system with `pathlib`.

---

## Opening Files with `open()`

### Basic Syntax

The built-in `open()` function is your gateway to file operations:

```python
# Basic syntax
file = open("filename.txt", mode="r", encoding="utf-8")
# ... do something with the file ...
file.close()
```

### File Modes

| Mode | Description | Creates file? | Truncates? |
|------|-------------|---------------|------------|
| `"r"` | Read (default) | No | No |
| `"w"` | Write | Yes | Yes |
| `"a"` | Append | Yes | No |
| `"x"` | Exclusive create | Yes (fails if exists) | N/A |
| `"r+"` | Read and write | No | No |
| `"w+"` | Write and read | Yes | Yes |
| `"a+"` | Append and read | Yes | No |
| `"rb"` | Read binary | No | No |
| `"wb"` | Write binary | Yes | Yes |
| `"ab"` | Append binary | Yes | No |

```python
# Read mode (default)
f = open("data.txt", "r")

# Write mode - creates new file or OVERWRITES existing
f = open("output.txt", "w")

# Append mode - adds to end of file
f = open("log.txt", "a")

# Binary read mode - for images, PDFs, etc.
f = open("image.png", "rb")

# Exclusive creation - fails if file already exists
f = open("new_file.txt", "x")
```

### Encoding

Always specify encoding for text files:

```python
# Explicitly specify encoding (recommended)
f = open("data.txt", "r", encoding="utf-8")

# Other common encodings
f = open("legacy.txt", "r", encoding="latin-1")
f = open("windows_file.txt", "r", encoding="cp1252")
```

If you omit `encoding`, Python uses the platform default, which varies across operating systems and can lead to bugs.

---

## The `with` Statement (Context Manager)

### Why You Must Use `with`

Files are system resources. If you forget to close them, you risk:
- Memory leaks
- Data loss (buffered writes may not be flushed)
- Running out of file descriptors

```python
# BAD: Manual open/close - error-prone
f = open("data.txt", "r")
content = f.read()
f.close()  # What if an exception occurs before this line?

# BAD: Even with try/finally, it is verbose
f = open("data.txt", "r")
try:
    content = f.read()
finally:
    f.close()

# GOOD: The with statement handles closing automatically
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
# File is automatically closed here, even if an exception occurred
```

### Multiple Files

```python
# Open multiple files simultaneously
with open("input.txt", "r", encoding="utf-8") as infile, \
     open("output.txt", "w", encoding="utf-8") as outfile:
    for line in infile:
        outfile.write(line.upper())

# Python 3.10+ parenthesized context managers
with (
    open("input.txt", "r", encoding="utf-8") as infile,
    open("output.txt", "w", encoding="utf-8") as outfile,
):
    for line in infile:
        outfile.write(line.upper())
```

From this point on, all examples use the `with` statement.

---

## Reading Files

### `read()` — Read Entire File

```python
with open("story.txt", "r", encoding="utf-8") as f:
    content = f.read()  # Returns entire file as one string
    print(content)
    print(f"Total characters: {len(content)}")
```

**Warning:** Do not use `read()` on very large files — it loads everything into memory.

### `read(size)` — Read N Characters

```python
with open("large_file.txt", "r", encoding="utf-8") as f:
    chunk = f.read(100)  # Read first 100 characters
    print(chunk)

    next_chunk = f.read(100)  # Read next 100 characters
    print(next_chunk)
```

### `readline()` — Read One Line

```python
with open("data.txt", "r", encoding="utf-8") as f:
    first_line = f.readline()   # Includes trailing newline
    second_line = f.readline()
    print(f"Line 1: {first_line!r}")
    print(f"Line 2: {second_line!r}")
```

### `readlines()` — Read All Lines into a List

```python
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()  # List of strings, each ending with '\n'
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines[:5], 1):
        print(f"  {i}: {line.rstrip()}")
```

### Iterating Over Lines (Best Practice)

The most Pythonic and memory-efficient way to read lines:

```python
with open("data.txt", "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        line = line.rstrip("\n")  # Remove trailing newline
        print(f"{line_number:4d}: {line}")
```

This reads one line at a time, keeping memory usage constant regardless of file size.

### Reading Patterns Comparison

```python
# Pattern 1: Read everything at once (small files)
with open("config.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Pattern 2: Read all lines at once (small files, need line list)
with open("names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f.readlines()]

# Pattern 3: Iterate line by line (large files, streaming)
with open("big_log.txt", "r", encoding="utf-8") as f:
    error_count = 0
    for line in f:
        if "ERROR" in line:
            error_count += 1
    print(f"Found {error_count} errors")

# Pattern 4: Read in chunks (very large binary files)
with open("huge_file.bin", "rb") as f:
    while True:
        chunk = f.read(8192)  # 8 KB chunks
        if not chunk:
            break
        process(chunk)
```

---

## Writing Files

### `write()` — Write a String

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("This is the second line.\n")
    f.write("And the third.\n")
    # write() returns the number of characters written
    count = f.write("Fourth line.\n")
    print(f"Wrote {count} characters")  # 13
```

**Important:** `write()` does not add a newline automatically. You must include `\n` yourself.

### `writelines()` — Write Multiple Strings

```python
lines = [
    "Alice,30,Engineer\n",
    "Bob,25,Designer\n",
    "Charlie,35,Manager\n",
]

with open("people.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)

# writelines() does NOT add newlines between items
# You must include '\n' in each string
```

### `print()` to a File

```python
with open("report.txt", "w", encoding="utf-8") as f:
    print("Monthly Report", file=f)
    print("=" * 40, file=f)
    print(f"Total sales: ${12345.67:.2f}", file=f)
    print(f"Items sold: {432}", file=f)
    # print() adds newline automatically
```

### Appending to Files

```python
import datetime

def log_event(message, log_file="app.log"):
    """Append a timestamped message to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

log_event("Application started")
log_event("User logged in: alice")
log_event("Processing complete")
```

### Write Mode Comparison

```python
# 'w' mode: Creates new or OVERWRITES existing
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("This replaces everything!\n")

# 'a' mode: Creates new or APPENDS to existing
with open("data.txt", "a", encoding="utf-8") as f:
    f.write("This is added to the end.\n")

# 'x' mode: Creates new or FAILS if exists
try:
    with open("data.txt", "x", encoding="utf-8") as f:
        f.write("Only if file does not exist.\n")
except FileExistsError:
    print("File already exists!")
```

---

## Working with Paths: `pathlib`

### Why `pathlib`?

The `pathlib` module provides an object-oriented interface to file system paths that works across operating systems:

```python
from pathlib import Path

# Old way (fragile, OS-specific)
import os
path = os.path.join("data", "output", "results.csv")

# New way (clean, cross-platform)
path = Path("data") / "output" / "results.csv"
print(path)  # data/output/results.csv (or data\output\results.csv on Windows)
```

### Creating Path Objects

```python
from pathlib import Path

# From a string
p = Path("data/output/results.csv")

# Current directory
current = Path.cwd()
print(current)  # /home/user/project

# Home directory
home = Path.home()
print(home)  # /home/user

# Joining paths with /
data_dir = Path("data")
output_file = data_dir / "output" / "results.csv"
print(output_file)  # data/output/results.csv
```

### Path Properties

```python
from pathlib import Path

p = Path("/home/user/projects/data/output/results.csv")

print(p.name)      # results.csv
print(p.stem)      # results
print(p.suffix)    # .csv
print(p.suffixes)  # ['.csv']
print(p.parent)    # /home/user/projects/data/output
print(p.parents[0])  # /home/user/projects/data/output
print(p.parents[1])  # /home/user/projects/data
print(p.parts)     # ('/', 'home', 'user', 'projects', 'data', 'output', 'results.csv')

# Changing components
new_path = p.with_name("summary.json")
print(new_path)  # /home/user/projects/data/output/summary.json

new_path = p.with_suffix(".json")
print(new_path)  # /home/user/projects/data/output/results.json

new_path = p.with_stem("summary")
print(new_path)  # /home/user/projects/data/output/summary.csv
```

### Path Operations

```python
from pathlib import Path

p = Path("data/output")

# Check existence
print(p.exists())       # True/False
print(p.is_file())      # True/False
print(p.is_dir())       # True/False

# Create directories
p.mkdir(parents=True, exist_ok=True)
# parents=True: create intermediate directories
# exist_ok=True: don't error if directory already exists

# List directory contents
for item in Path(".").iterdir():
    prefix = "D" if item.is_dir() else "F"
    print(f"  [{prefix}] {item.name}")

# Glob: find files matching a pattern
for py_file in Path(".").glob("*.py"):
    print(py_file)

# Recursive glob
for py_file in Path(".").rglob("*.py"):
    print(py_file)

# Resolve to absolute path
absolute = Path("relative/path").resolve()
print(absolute)
```

### Reading and Writing with `pathlib`

```python
from pathlib import Path

p = Path("example.txt")

# Write text
p.write_text("Hello, pathlib!\nSecond line.", encoding="utf-8")

# Read text
content = p.read_text(encoding="utf-8")
print(content)

# Write bytes
p_bin = Path("data.bin")
p_bin.write_bytes(b"\x00\x01\x02\x03")

# Read bytes
data = p_bin.read_bytes()
print(data)  # b'\x00\x01\x02\x03'
```

### Practical Path Examples

```python
from pathlib import Path

def find_project_root(marker=".git"):
    """Walk up directories to find the project root."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    return None

def organize_files_by_extension(source_dir):
    """Sort files into subdirectories by their extension."""
    source = Path(source_dir)
    for file_path in source.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lstrip(".") or "no_extension"
            dest_dir = source / ext
            dest_dir.mkdir(exist_ok=True)
            file_path.rename(dest_dir / file_path.name)

def get_total_size(directory):
    """Calculate total size of all files in a directory tree."""
    total = sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"
```

---

## CSV Files

### Reading CSV Files

```python
import csv

# Basic CSV reading
with open("employees.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # Read header row
    print(f"Columns: {header}")

    for row in reader:
        name, department, salary = row
        print(f"  {name} - {department} - ${salary}")
```

### Reading CSV as Dictionaries

```python
import csv

# DictReader maps each row to a dictionary using the header
with open("employees.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['name']} works in {row['department']}")
        print(f"    Salary: ${row['salary']}")
```

### Writing CSV Files

```python
import csv

# Writing with csv.writer
employees = [
    ["Alice", "Engineering", 95000],
    ["Bob", "Design", 82000],
    ["Charlie", "Management", 105000],
]

with open("employees.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "department", "salary"])  # Header
    writer.writerows(employees)  # Data rows

# Writing with csv.DictWriter
employees_dict = [
    {"name": "Alice", "department": "Engineering", "salary": 95000},
    {"name": "Bob", "department": "Design", "salary": 82000},
    {"name": "Charlie", "department": "Management", "salary": 105000},
]

with open("employees.csv", "w", encoding="utf-8", newline="") as f:
    fieldnames = ["name", "department", "salary"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(employees_dict)
```

**Note:** Always use `newline=""` when opening CSV files for writing. This prevents double newlines on Windows.

### CSV with Custom Delimiters

```python
import csv

# Tab-separated values (TSV)
with open("data.tsv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        print(row)

# Pipe-separated values
with open("data.psv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        print(row)

# Handling quoted fields with commas inside
with open("tricky.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["Name", "Address", "Notes"])
    writer.writerow(["Alice", "123 Main St, Apt 4", "Has a comma, in notes"])
```

### Practical CSV Example

```python
import csv
from pathlib import Path

def csv_summary(filepath):
    """Print summary statistics for a CSV file."""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Empty CSV file")
        return

    print(f"File: {path.name}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {', '.join(rows[0].keys())}")
    print(f"Sample (first 3 rows):")
    for row in rows[:3]:
        print(f"  {dict(row)}")

def filter_csv(input_path, output_path, column, value):
    """Filter CSV rows where column matches value."""
    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        with open(output_path, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            count = 0
            for row in reader:
                if row[column] == value:
                    writer.writerow(row)
                    count += 1

    print(f"Wrote {count} matching rows to {output_path}")
```

---

## JSON Files

### JSON Basics

JSON (JavaScript Object Notation) is the most common data exchange format:

```python
import json

# Python dict → JSON string (serialization)
data = {
    "name": "Alice",
    "age": 30,
    "languages": ["Python", "JavaScript", "Go"],
    "active": True,
    "address": None
}

json_string = json.dumps(data)
print(json_string)
# {"name": "Alice", "age": 30, "languages": ["Python", "JavaScript", "Go"], "active": true, "address": null}

# Pretty-printed
json_pretty = json.dumps(data, indent=2)
print(json_pretty)
```

### Type Mapping Between Python and JSON

| Python | JSON |
|--------|------|
| `dict` | `object {}` |
| `list`, `tuple` | `array []` |
| `str` | `string ""` |
| `int`, `float` | `number` |
| `True` | `true` |
| `False` | `false` |
| `None` | `null` |

### `dumps` and `loads` — String Operations

```python
import json

# dumps: Python object → JSON string
data = {"name": "Alice", "scores": [95, 87, 91]}
json_str = json.dumps(data)
print(type(json_str))  # <class 'str'>

# loads: JSON string → Python object
parsed = json.loads(json_str)
print(type(parsed))       # <class 'dict'>
print(parsed["name"])     # Alice
print(parsed["scores"])   # [95, 87, 91]

# dumps options
print(json.dumps(data, indent=4))              # Pretty-print with 4 spaces
print(json.dumps(data, sort_keys=True))        # Sort keys alphabetically
print(json.dumps(data, separators=(",", ":"))) # Compact (no spaces)
```

### `dump` and `load` — File Operations

```python
import json

# dump: Write Python object to JSON file
data = {
    "users": [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "editor"},
    ],
    "version": "1.0"
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# load: Read JSON file to Python object
with open("data.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)

print(loaded["users"][0]["name"])  # Alice
print(loaded["version"])           # 1.0
```

### Handling Non-Serializable Types

```python
import json
from datetime import datetime, date
from pathlib import Path

# This fails: datetime is not JSON-serializable
data = {"timestamp": datetime.now(), "path": Path("/tmp")}
# json.dumps(data)  # TypeError!

# Solution 1: Custom default function
def json_serializer(obj):
    """Handle non-serializable types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

data = {
    "timestamp": datetime.now(),
    "path": Path("/home/user/data"),
    "tags": {"python", "json", "tutorial"},
}

json_str = json.dumps(data, default=json_serializer, indent=2)
print(json_str)

# Solution 2: Custom JSONEncoder class
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

json_str = json.dumps(data, cls=CustomEncoder, indent=2)
```

### Practical JSON Examples

```python
import json
from pathlib import Path

def load_config(config_path="config.json"):
    """Load application configuration from a JSON file."""
    path = Path(config_path)
    if not path.exists():
        # Return default configuration
        return {
            "database": {"host": "localhost", "port": 5432},
            "debug": False,
            "log_level": "INFO",
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config, config_path="config.json"):
    """Save application configuration to a JSON file."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def merge_json_files(file_paths, output_path):
    """Merge multiple JSON files into one."""
    merged = {}
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged.update(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    return merged

# Usage
config = load_config()
config["debug"] = True
config["new_setting"] = "value"
save_config(config)
```

---

## Binary Files

### Reading and Writing Bytes

```python
# Writing binary data
with open("data.bin", "wb") as f:
    f.write(b"\x89PNG\r\n")  # PNG file header bytes
    f.write(bytes([0, 1, 2, 3, 4, 5]))
    f.write(bytearray(range(10)))

# Reading binary data
with open("data.bin", "rb") as f:
    header = f.read(6)
    print(header)  # b'\x89PNG\r\n'
    rest = f.read()
    print(list(rest))  # [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Copying Binary Files

```python
def copy_file(source, destination, chunk_size=8192):
    """Copy a file in chunks (works for any file type)."""
    with open(source, "rb") as src, open(destination, "wb") as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)

copy_file("photo.jpg", "photo_backup.jpg")
```

### Working with Struct for Binary Data

```python
import struct

# Pack Python values into binary format
# 'i' = int (4 bytes), 'f' = float (4 bytes), '10s' = 10-byte string
packed = struct.pack("if10s", 42, 3.14, b"Hello     ")
print(len(packed))  # 18 bytes
print(packed)       # b'*\x00\x00\x00\xc3\xf5H@Hello     '

# Unpack binary data back to Python values
number, pi, text = struct.unpack("if10s", packed)
print(number)           # 42
print(pi)               # 3.140000104904175 (float precision)
print(text.strip())     # b'Hello'

# Writing structured binary data
records = [
    (1, "Alice", 95.5),
    (2, "Bob", 87.3),
    (3, "Charlie", 91.8),
]

with open("records.bin", "wb") as f:
    for id_, name, score in records:
        packed = struct.pack("i20sf", id_, name.encode("utf-8"), score)
        f.write(packed)

# Reading structured binary data
record_format = "i20sf"
record_size = struct.calcsize(record_format)

with open("records.bin", "rb") as f:
    while True:
        data = f.read(record_size)
        if not data:
            break
        id_, name_bytes, score = struct.unpack(record_format, data)
        name = name_bytes.decode("utf-8").rstrip("\x00")
        print(f"  ID: {id_}, Name: {name}, Score: {score:.1f}")
```

---

## Temporary Files

### Using `tempfile`

```python
import tempfile
from pathlib import Path

# Temporary file (automatically deleted when closed)
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("Temporary data\n")
    temp_path = f.name
    print(f"Temp file: {temp_path}")

# Read it back
with open(temp_path, "r") as f:
    print(f.read())

# Clean up manually since we used delete=False
Path(temp_path).unlink()

# Temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Temp dir: {tmpdir}")
    # Create files in the temp directory
    temp_file = Path(tmpdir) / "output.txt"
    temp_file.write_text("Hello from temp dir!", encoding="utf-8")
    print(temp_file.read_text(encoding="utf-8"))
# Directory and all contents are automatically deleted here

# Get the default temp directory
print(tempfile.gettempdir())  # /tmp on Linux, varies on other OS
```

### Practical Use: Safe File Writing

```python
import tempfile
import os
from pathlib import Path

def safe_write(filepath, content, encoding="utf-8"):
    """Write to a file atomically using a temporary file.

    This prevents data corruption if the process is interrupted.
    """
    filepath = Path(filepath)
    # Write to a temp file in the same directory
    fd, tmp_path = tempfile.mkstemp(
        dir=filepath.parent,
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        # Atomic rename (on most file systems)
        os.replace(tmp_path, filepath)
    except Exception:
        # Clean up temp file on failure
        os.unlink(tmp_path)
        raise

safe_write("config.json", '{"key": "value"}')
```

---

## File System Operations

### Using `os` and `os.path`

```python
import os

# Current working directory
print(os.getcwd())

# List directory contents
print(os.listdir("."))

# Check existence
print(os.path.exists("data.txt"))
print(os.path.isfile("data.txt"))
print(os.path.isdir("data"))

# File size
print(os.path.getsize("data.txt"))  # Size in bytes

# Path manipulation (prefer pathlib for new code)
print(os.path.join("data", "output", "file.txt"))
print(os.path.basename("/home/user/file.txt"))  # file.txt
print(os.path.dirname("/home/user/file.txt"))   # /home/user
print(os.path.splitext("report.csv"))           # ('report', '.csv')
print(os.path.abspath("relative/path"))

# Create directory
os.makedirs("data/output/reports", exist_ok=True)

# Remove files and directories
os.remove("temp.txt")            # Delete a file
os.rmdir("empty_dir")            # Delete an empty directory
```

### Using `shutil` for High-Level Operations

```python
import shutil

# Copy a file
shutil.copy("source.txt", "destination.txt")       # Copy file
shutil.copy2("source.txt", "destination.txt")      # Copy file + metadata

# Copy a directory tree
shutil.copytree("source_dir", "destination_dir")

# Move/rename
shutil.move("old_name.txt", "new_name.txt")
shutil.move("file.txt", "archive/file.txt")

# Remove a directory tree (be careful!)
shutil.rmtree("directory_to_delete")

# Disk usage
usage = shutil.disk_usage("/")
print(f"Total: {usage.total / (1024**3):.1f} GB")
print(f"Used:  {usage.used / (1024**3):.1f} GB")
print(f"Free:  {usage.free / (1024**3):.1f} GB")
```

### `pathlib` Equivalents

```python
from pathlib import Path

p = Path("data/output")

# Create directory
p.mkdir(parents=True, exist_ok=True)

# Check existence
p.exists()
p.is_file()
p.is_dir()

# List contents
list(p.iterdir())

# Glob
list(p.glob("*.csv"))
list(p.rglob("*.py"))  # Recursive

# File info
stat = Path("data.txt").stat()
print(f"Size: {stat.st_size} bytes")
print(f"Modified: {stat.st_mtime}")

# Delete
Path("temp.txt").unlink(missing_ok=True)     # Delete file
Path("empty_dir").rmdir()                      # Delete empty directory
```

---

## Common File I/O Patterns

### Reading a Configuration File

```python
import json
from pathlib import Path

def load_app_config(config_path="config.json", defaults=None):
    """Load configuration with defaults and validation."""
    default_config = defaults or {
        "host": "localhost",
        "port": 8080,
        "debug": False,
        "database": "app.db",
        "log_file": "app.log",
    }

    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        # Merge: user config overrides defaults
        config = {**default_config, **user_config}
    else:
        config = default_config
        # Save defaults for future reference
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config at {config_path}")

    return config
```

### Simple Logging to a File

```python
import datetime
from pathlib import Path

class SimpleLogger:
    """A simple file-based logger."""

    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def __init__(self, log_file="app.log", level="INFO"):
        self.log_file = Path(log_file)
        self.level = self.LEVELS.get(level, 20)

    def _write(self, level_name, message):
        if self.LEVELS[level_name] >= self.level:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] [{level_name:7s}] {message}\n"
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry)

    def debug(self, message):
        self._write("DEBUG", message)

    def info(self, message):
        self._write("INFO", message)

    def warning(self, message):
        self._write("WARNING", message)

    def error(self, message):
        self._write("ERROR", message)

# Usage
logger = SimpleLogger("app.log", level="INFO")
logger.info("Application started")
logger.debug("This won't appear (below INFO level)")
logger.warning("Low disk space")
logger.error("Connection failed")
```

### Processing Log Files

```python
from pathlib import Path
from collections import Counter

def analyze_log(log_path):
    """Analyze a log file and report statistics."""
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {log_path}")
        return

    level_counts = Counter()
    error_messages = []
    total_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            # Parse log level from lines like: [2024-01-15 10:30:00] [INFO   ] message
            if "] [" in line:
                parts = line.split("] [")
                if len(parts) >= 2:
                    level = parts[1].split("]")[0].strip()
                    level_counts[level] += 1
                    if level == "ERROR":
                        error_messages.append(line)

    print(f"Log Analysis: {path.name}")
    print(f"  Total lines: {total_lines}")
    print(f"  Level breakdown:")
    for level, count in level_counts.most_common():
        print(f"    {level}: {count}")
    if error_messages:
        print(f"  Recent errors:")
        for msg in error_messages[-5:]:
            print(f"    {msg}")
```

### Batch File Processing

```python
from pathlib import Path
import csv
import json

def convert_csv_to_json(csv_path, json_path=None):
    """Convert a CSV file to JSON format."""
    csv_path = Path(csv_path)
    if json_path is None:
        json_path = csv_path.with_suffix(".json")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Converted {csv_path.name} -> {json_path.name} ({len(data)} records)")
    return json_path

def batch_convert(directory, pattern="*.csv"):
    """Convert all CSV files in a directory to JSON."""
    source = Path(directory)
    for csv_file in source.glob(pattern):
        convert_csv_to_json(csv_file)
```

### Line-by-Line File Transformation

```python
from pathlib import Path

def transform_file(input_path, output_path, transform_func):
    """Apply a transformation function to each line of a file."""
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line_number, line in enumerate(infile, 1):
            transformed = transform_func(line, line_number)
            if transformed is not None:
                outfile.write(transformed)

# Example: Add line numbers
def add_line_numbers(line, num):
    return f"{num:4d} | {line}"

transform_file("source.py", "numbered_source.txt", add_line_numbers)

# Example: Filter lines containing a keyword
def filter_errors(line, num):
    if "ERROR" in line or "CRITICAL" in line:
        return line
    return None

transform_file("app.log", "errors_only.log", filter_errors)

# Example: Strip comments from a Python file
def strip_comments(line, num):
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return None
    return line

transform_file("script.py", "script_no_comments.py", strip_comments)
```

---

## File Encoding

### Understanding Encodings

```python
# UTF-8 is the default and most common encoding
text = "Hello, World! Cafe\u0301 \u00e9"

# Encode string to bytes
utf8_bytes = text.encode("utf-8")
print(utf8_bytes)       # b'Hello, World! Caf\xc3\xa9 \xc3\xa9'
print(len(utf8_bytes))  # More bytes than characters for non-ASCII

# Decode bytes to string
decoded = utf8_bytes.decode("utf-8")
print(decoded)  # Hello, World! Cafe e

# Common encoding issues
latin1_bytes = "caf\u00e9".encode("latin-1")
print(latin1_bytes)  # b'caf\xe9'

# Decoding with wrong encoding produces garbage
wrong = latin1_bytes.decode("utf-8", errors="replace")
print(wrong)  # caf\ufffd (replacement character)
```

### Handling Encoding Errors

```python
# Error handling strategies
text = "Hello \u2603 World"  # Snowman character

# 'strict' (default) - raises UnicodeEncodeError
try:
    text.encode("ascii")
except UnicodeEncodeError as e:
    print(f"Error: {e}")

# 'ignore' - skip unrepresentable characters
print(text.encode("ascii", errors="ignore"))  # b'Hello  World'

# 'replace' - use replacement character
print(text.encode("ascii", errors="replace"))  # b'Hello ? World'

# 'xmlcharrefreplace' - use XML character references
print(text.encode("ascii", errors="xmlcharrefreplace"))  # b'Hello &#9731; World'

# Reading files with encoding errors
with open("messy.txt", "r", encoding="utf-8", errors="replace") as f:
    content = f.read()  # Replaces bad bytes with U+FFFD
```

### Detecting File Encoding

```python
def detect_encoding(filepath, num_bytes=10000):
    """Simple heuristic to detect file encoding."""
    with open(filepath, "rb") as f:
        raw = f.read(num_bytes)

    # Check for BOM (Byte Order Mark)
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"

    # Try UTF-8
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # Fall back to latin-1 (always succeeds, may not be correct)
    return "latin-1"
```

---

## Summary

| Operation | Method | Best For |
|-----------|--------|----------|
| Read entire file | `f.read()` | Small files |
| Read line by line | `for line in f:` | Large files |
| Read all lines | `f.readlines()` | Need list of lines |
| Write string | `f.write(str)` | Single writes |
| Write lines | `f.writelines(list)` | Multiple lines |
| Print to file | `print(..., file=f)` | Formatted output |
| Path handling | `pathlib.Path` | Cross-platform paths |
| CSV | `csv.reader/writer` | Tabular data |
| JSON | `json.dump/load` | Structured data |
| Binary | `open(..., "rb/wb")` | Images, archives |

Key takeaways:
- **Always** use the `with` statement for file operations
- **Always** specify `encoding="utf-8"` for text files
- Use **`pathlib.Path`** for path manipulation instead of string operations
- **Iterate line by line** for large files instead of reading everything at once
- Use the **`csv` module** for CSV files (do not split on commas manually)
- Use the **`json` module** for JSON (do not construct JSON strings manually)
- Use `newline=""` when opening CSV files for writing

---

## Further Reading

- [Python `open()` Documentation](https://docs.python.org/3/library/functions.html#open)
- [pathlib — Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)
- [csv — CSV File Reading and Writing](https://docs.python.org/3/library/csv.html)
- [json — JSON encoder and decoder](https://docs.python.org/3/library/json.html)
- [Real Python: Working With Files in Python](https://realpython.com/working-with-files-in-python/)

---

**Previous**: [Modules and Packages](./10_Modules_and_Packages.md) | **Next**: [Exception Handling](./12_Exception_Handling.md)
