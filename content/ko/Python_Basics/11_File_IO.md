# 파일 입출력

**이전**: [모듈과 패키지](./10_Modules_and_Packages.md) | **다음**: [예외 처리](./12_Exception_Handling.md)

> **주제**: Python 기초
> **수업**: 14개 중 11번째
> **선수 지식**: 함수, 문자열, 모듈과 패키지

## 학습 목표

이 수업을 완료하면 다음을 할 수 있습니다:

1. 다양한 모드(`r`, `w`, `a`, `rb`, `wb`)로 `open()`을 사용하여 텍스트 파일을 열고, 읽고, 쓰기
2. `with` 문을 사용하여 적절한 파일 리소스 관리 보장하기
3. 파일을 줄 단위, 청크 단위, 전체를 한 번에 읽으며 각 상황에 맞는 접근법 선택하기
4. 크로스 플랫폼 호환성을 위해 `pathlib.Path` 클래스로 파일 경로 작업하기
5. `csv` 모듈을 사용하여 CSV 파일을 파싱하고 작성하기
6. `json` 모듈(`dumps`, `loads`, `dump`, `load`)로 데이터를 직렬화 (Serialization) 및 역직렬화하기
7. 존재 여부 확인, 디렉토리 나열, 파일 복사와 같은 파일 시스템 작업 수행하기
8. 설정 파일 읽기 및 로그 파일 쓰기를 포함한 일반적인 파일 I/O 패턴 적용하기

---

## 소개

거의 모든 실제 프로그램은 파일과 상호 작용해야 합니다 — 설정 읽기, 데이터 처리, 로그 쓰기, 사용자 데이터 저장, 또는 다른 시스템과 정보 교환 등. Python은 내장 함수와 풍부한 표준 라이브러리로 파일 I/O를 간단하게 만듭니다.

이 수업은 기본적인 파일 읽기 및 쓰기부터 CSV와 JSON 같은 구조화된 형식 작업, 그리고 `pathlib`을 사용한 파일 시스템 탐색까지 모든 것을 다룹니다.

---

## `open()`으로 파일 열기

### 기본 구문

내장 `open()` 함수는 파일 작업의 관문입니다:

```python
# Basic syntax
file = open("filename.txt", mode="r", encoding="utf-8")
# ... do something with the file ...
file.close()
```

### 파일 모드

| 모드 | 설명 | 파일 생성? | 잘라내기? |
|------|------|-----------|----------|
| `"r"` | 읽기 (기본값) | 아니오 | 아니오 |
| `"w"` | 쓰기 | 예 | 예 |
| `"a"` | 추가 | 예 | 아니오 |
| `"x"` | 배타적 생성 | 예 (존재하면 실패) | 해당 없음 |
| `"r+"` | 읽기 및 쓰기 | 아니오 | 아니오 |
| `"w+"` | 쓰기 및 읽기 | 예 | 예 |
| `"a+"` | 추가 및 읽기 | 예 | 아니오 |
| `"rb"` | 바이너리 읽기 | 아니오 | 아니오 |
| `"wb"` | 바이너리 쓰기 | 예 | 예 |
| `"ab"` | 바이너리 추가 | 예 | 아니오 |

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

### 인코딩

텍스트 파일에 대해서는 항상 인코딩을 지정하세요:

```python
# Explicitly specify encoding (recommended)
f = open("data.txt", "r", encoding="utf-8")

# Other common encodings
f = open("legacy.txt", "r", encoding="latin-1")
f = open("windows_file.txt", "r", encoding="cp1252")
```

`encoding`을 생략하면 Python은 플랫폼 기본값을 사용하며, 이는 운영 체제마다 다르고 버그로 이어질 수 있습니다.

---

## `with` 문 (컨텍스트 관리자)

### `with`를 반드시 사용해야 하는 이유

파일은 시스템 리소스입니다. 닫는 것을 잊으면 다음과 같은 위험이 있습니다:
- 메모리 누수
- 데이터 손실 (버퍼된 쓰기가 플러시되지 않을 수 있음)
- 파일 디스크립터 부족

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

### 여러 파일

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

이 시점부터 모든 예제는 `with` 문을 사용합니다.

---

## 파일 읽기

### `read()` — 전체 파일 읽기

```python
with open("story.txt", "r", encoding="utf-8") as f:
    content = f.read()  # Returns entire file as one string
    print(content)
    print(f"Total characters: {len(content)}")
```

**경고:** 매우 큰 파일에 `read()`를 사용하지 마세요 — 모든 것을 메모리에 로드합니다.

### `read(size)` — N개 문자 읽기

```python
with open("large_file.txt", "r", encoding="utf-8") as f:
    chunk = f.read(100)  # Read first 100 characters
    print(chunk)

    next_chunk = f.read(100)  # Read next 100 characters
    print(next_chunk)
```

### `readline()` — 한 줄 읽기

```python
with open("data.txt", "r", encoding="utf-8") as f:
    first_line = f.readline()   # Includes trailing newline
    second_line = f.readline()
    print(f"Line 1: {first_line!r}")
    print(f"Line 2: {second_line!r}")
```

### `readlines()` — 모든 줄을 리스트로 읽기

```python
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()  # List of strings, each ending with '\n'
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines[:5], 1):
        print(f"  {i}: {line.rstrip()}")
```

### 줄 단위 반복 (모범 사례)

줄을 읽는 가장 파이썬다운 방법이자 메모리 효율적인 방법:

```python
with open("data.txt", "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        line = line.rstrip("\n")  # Remove trailing newline
        print(f"{line_number:4d}: {line}")
```

이것은 한 번에 한 줄씩 읽어 파일 크기에 관계없이 메모리 사용량을 일정하게 유지합니다.

### 읽기 패턴 비교

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

## 파일 쓰기

### `write()` — 문자열 쓰기

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("This is the second line.\n")
    f.write("And the third.\n")
    # write() returns the number of characters written
    count = f.write("Fourth line.\n")
    print(f"Wrote {count} characters")  # 13
```

**중요:** `write()`는 자동으로 줄바꿈을 추가하지 않습니다. `\n`을 직접 포함해야 합니다.

### `writelines()` — 여러 문자열 쓰기

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

### `print()`로 파일에 쓰기

```python
with open("report.txt", "w", encoding="utf-8") as f:
    print("Monthly Report", file=f)
    print("=" * 40, file=f)
    print(f"Total sales: ${12345.67:.2f}", file=f)
    print(f"Items sold: {432}", file=f)
    # print() adds newline automatically
```

### 파일에 추가하기

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

### 쓰기 모드 비교

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

## 경로 작업: `pathlib`

### 왜 `pathlib`인가?

`pathlib` 모듈은 운영 체제 간에 작동하는 객체 지향적인 파일 시스템 경로 인터페이스를 제공합니다:

```python
from pathlib import Path

# Old way (fragile, OS-specific)
import os
path = os.path.join("data", "output", "results.csv")

# New way (clean, cross-platform)
path = Path("data") / "output" / "results.csv"
print(path)  # data/output/results.csv (or data\output\results.csv on Windows)
```

### Path 객체 생성

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

### 경로 속성

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

### 경로 작업

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

### `pathlib`으로 읽기 및 쓰기

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

### 실용적인 경로 예제

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

## CSV 파일

### CSV 파일 읽기

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

### 딕셔너리로 CSV 읽기

```python
import csv

# DictReader maps each row to a dictionary using the header
with open("employees.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['name']} works in {row['department']}")
        print(f"    Salary: ${row['salary']}")
```

### CSV 파일 쓰기

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

**참고:** CSV 파일을 쓸 때 항상 `newline=""`을 사용하세요. Windows에서 이중 줄바꿈을 방지합니다.

### 사용자 정의 구분자가 있는 CSV

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

### 실용적인 CSV 예제

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

## JSON 파일

### JSON 기초

JSON (JavaScript Object Notation)은 가장 일반적인 데이터 교환 형식입니다:

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

### Python과 JSON 간의 타입 매핑

| Python | JSON |
|--------|------|
| `dict` | `object {}` |
| `list`, `tuple` | `array []` |
| `str` | `string ""` |
| `int`, `float` | `number` |
| `True` | `true` |
| `False` | `false` |
| `None` | `null` |

### `dumps`와 `loads` — 문자열 작업

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

### `dump`와 `load` — 파일 작업

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

### 직렬화 불가능한 타입 처리

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

### 실용적인 JSON 예제

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

## 바이너리 파일

### 바이트 읽기 및 쓰기

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

### 바이너리 파일 복사

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

### 바이너리 데이터를 위한 Struct 사용

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

## 임시 파일

### `tempfile` 사용

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

### 실용적인 활용: 안전한 파일 쓰기

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

## 파일 시스템 작업

### `os`와 `os.path` 사용

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

### `shutil`을 사용한 고수준 작업

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

### `pathlib` 동등 기능

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

## 일반적인 파일 I/O 패턴

### 설정 파일 읽기

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

### 간단한 파일 로깅

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

### 로그 파일 처리

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

### 일괄 파일 처리

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

### 줄 단위 파일 변환

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

## 파일 인코딩

### 인코딩 이해하기

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

### 인코딩 오류 처리

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

### 파일 인코딩 감지

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

## 요약

| 작업 | 메서드 | 적합한 용도 |
|------|--------|------------|
| 전체 파일 읽기 | `f.read()` | 작은 파일 |
| 줄 단위 읽기 | `for line in f:` | 큰 파일 |
| 모든 줄 읽기 | `f.readlines()` | 줄 리스트가 필요할 때 |
| 문자열 쓰기 | `f.write(str)` | 단일 쓰기 |
| 줄 쓰기 | `f.writelines(list)` | 여러 줄 |
| 파일로 출력 | `print(..., file=f)` | 형식화된 출력 |
| 경로 처리 | `pathlib.Path` | 크로스 플랫폼 경로 |
| CSV | `csv.reader/writer` | 표 형식 데이터 |
| JSON | `json.dump/load` | 구조화된 데이터 |
| 바이너리 | `open(..., "rb/wb")` | 이미지, 아카이브 |

핵심 내용:
- 파일 작업에는 **항상** `with` 문을 사용하세요
- 텍스트 파일에는 **항상** `encoding="utf-8"`을 지정하세요
- 문자열 작업 대신 **`pathlib.Path`**를 경로 조작에 사용하세요
- 큰 파일은 한 번에 모든 것을 읽는 대신 **줄 단위로 반복**하세요
- CSV 파일에는 **`csv` 모듈**을 사용하세요 (쉼표로 직접 분할하지 마세요)
- JSON에는 **`json` 모듈**을 사용하세요 (JSON 문자열을 직접 구성하지 마세요)
- CSV 파일을 쓸 때 `newline=""`을 사용하세요

---

## 추가 자료

- [Python `open()` 문서](https://docs.python.org/3/library/functions.html#open)
- [pathlib — 객체 지향 파일시스템 경로](https://docs.python.org/3/library/pathlib.html)
- [csv — CSV 파일 읽기 및 쓰기](https://docs.python.org/3/library/csv.html)
- [json — JSON 인코더 및 디코더](https://docs.python.org/3/library/json.html)
- [Real Python: Python에서 파일 작업하기](https://realpython.com/working-with-files-in-python/)

---

**이전**: [모듈과 패키지](./10_Modules_and_Packages.md) | **다음**: [예외 처리](./12_Exception_Handling.md)
