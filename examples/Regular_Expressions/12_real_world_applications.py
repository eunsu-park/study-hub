"""
12 Real-World Applications
============================
Demonstrates log parsing, data cleaning, code refactoring,
INI parsing, tokenization, and batch file renaming.
"""

import re
from collections import Counter, defaultdict


def log_parsing():
    """Parse structured log files."""
    log = """[2024-01-15 08:30:45] INFO  server.py:42 - Server started
[2024-01-15 08:31:15] ERROR api.py:203 - Request timeout
[2024-01-15 08:31:45] ERROR db.py:67 - Connection refused
[2024-01-15 08:32:00] INFO  server.py:100 - Shutdown"""

    pattern = re.compile(r"""
        ^\[(?P<dt>[^\]]+)\]\s+
        (?P<level>\w+)\s+
        (?P<file>\S+):(?P<line>\d+)\s+-\s+
        (?P<msg>.+)$
    """, re.X | re.M)

    entries = [m.groupdict() for m in pattern.finditer(log)]
    counts = Counter(e['level'] for e in entries)
    print(f"Level counts: {dict(counts)}")
    for e in entries:
        if e['level'] == 'ERROR':
            print(f"  ERROR: {e['dt']} - {e['msg']}")


def data_cleaning():
    """Clean and normalize messy text data."""
    def clean(text):
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.M)
        return text.strip()

    dirty = '  <p>Hello   World</p>\n\n\n  Too   many   spaces  '
    print(f"Cleaned: '{clean(dirty)}'")

    # Phone normalization
    def normalize_phone(p):
        d = re.sub(r'\D', '', p)
        if len(d) == 11 and d[0] == '1':
            d = d[1:]
        return f"({d[:3]}) {d[3:6]}-{d[6:]}" if len(d) == 10 else None

    for phone in ["555-867-5309", "(555) 867.5309", "+1 555 867 5309"]:
        print(f"  {phone:25s} -> {normalize_phone(phone)}")


def code_refactoring():
    """Refactor code using regex substitution."""
    code = """def calc(item_price, item_count):
    item_total = item_price * item_count
    item_tax = item_total * 0.08
    return item_total + item_tax"""

    refactored = re.sub(r'\bitem_(\w+)', r'product_\1', code)
    print("Refactored:")
    print(refactored)


def ini_parser():
    """Parse INI-style configuration files."""
    ini = """[database]
host = localhost
port = 5432
[server]
host = 0.0.0.0
port = 8080
debug = true"""

    config = {}
    section = None
    for line in ini.split('\n'):
        line = line.strip()
        sec = re.match(r'^\[(\w+)\]$', line)
        if sec:
            section = sec.group(1)
            config[section] = {}
        else:
            kv = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if kv and section:
                config[section][kv.group(1)] = kv.group(2).strip()

    for sec, vals in config.items():
        print(f"[{sec}]")
        for k, v in vals.items():
            print(f"  {k} = {v}")


def tokenizer():
    """Tokenize a mathematical expression."""
    specs = [
        ('NUM', r'\d+\.?\d*'), ('OP', r'[+\-*/]'),
        ('LPAR', r'\('), ('RPAR', r'\)'), ('WS', r'\s+'),
    ]
    combined = '|'.join(f'(?P<{n}>{p})' for n, p in specs)
    expr = "3.14 * (2 + 5) - 10 / 3"
    tokens = [(m.lastgroup, m.group()) for m in re.finditer(combined, expr)
              if m.lastgroup != 'WS']
    for kind, val in tokens:
        print(f"  {kind:5s}: {val}")


def batch_rename():
    """Simulate batch file renaming with regex."""
    files = ["IMG_20240115_083045.jpg", "IMG_20240116_142030.jpg"]
    for f in files:
        new = re.sub(
            r'IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
            r'\1-\2-\3_\4-\5-\6', f
        )
        print(f"  {f} -> {new}")


if __name__ == "__main__":
    sections = [
        ("Log Parsing", log_parsing),
        ("Data Cleaning", data_cleaning),
        ("Code Refactoring", code_refactoring),
        ("INI Parser", ini_parser),
        ("Tokenizer", tokenizer),
        ("Batch Rename", batch_rename),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
