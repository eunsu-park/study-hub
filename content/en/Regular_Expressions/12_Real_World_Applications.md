# Real-World Applications

## Learning Objectives

After completing this lesson, you will be able to:

1. Parse and analyze log files using regex
2. Clean and normalize messy data with substitution patterns
3. Extract structured data from unstructured text
4. Refactor code using regex-based find and replace
5. Validate and transform configuration files
6. Build a simple tokenizer/lexer with regex
7. Process CSV and TSV data with regex-aware splitting
8. Combine multiple regex techniques into complete solutions

---

## 1. Log File Parsing

### Standard Log Format Parsing

```python
import re
from collections import Counter

log_text = """
[2024-01-15 08:30:45] INFO  server.py:42 - Server started on port 8080
[2024-01-15 08:30:46] DEBUG db.py:15 - Connection pool initialized (size=10)
[2024-01-15 08:31:02] WARN  auth.py:88 - Failed login attempt from 192.168.1.50
[2024-01-15 08:31:15] ERROR api.py:203 - Request timeout after 30s: GET /api/users
[2024-01-15 08:31:16] ERROR api.py:203 - Request timeout after 30s: GET /api/orders
[2024-01-15 08:31:20] INFO  api.py:55 - Health check OK
[2024-01-15 08:31:45] ERROR db.py:67 - Connection refused: max pool exhausted
[2024-01-15 08:32:00] INFO  server.py:100 - Graceful shutdown initiated
""".strip()

# Comprehensive log parser
log_pattern = re.compile(r"""
    ^\[(?P<datetime>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+
    (?P<level>\w+)\s+
    (?P<file>\S+):(?P<line>\d+)\s+-\s+
    (?P<message>.+)$
""", re.VERBOSE | re.MULTILINE)

# Parse all entries
entries = []
for match in log_pattern.finditer(log_text):
    entries.append(match.groupdict())

# Analysis: count by level
level_counts = Counter(e['level'] for e in entries)
print("Log Level Distribution:")
for level, count in level_counts.most_common():
    bar = "#" * (count * 5)
    print(f"  {level:5s}: {count} {bar}")

# Find errors
print("\nErrors:")
for e in entries:
    if e['level'] == 'ERROR':
        print(f"  [{e['datetime']}] {e['file']}:{e['line']} - {e['message']}")
```

### Apache/Nginx Access Log Parsing

```python
import re

access_log = """
192.168.1.10 - - [15/Jan/2024:08:30:45 +0000] "GET /index.html HTTP/1.1" 200 5432
192.168.1.20 - user1 [15/Jan/2024:08:30:46 +0000] "POST /api/login HTTP/1.1" 401 128
10.0.0.5 - - [15/Jan/2024:08:31:02 +0000] "GET /api/users?page=2 HTTP/1.1" 200 8192
192.168.1.10 - - [15/Jan/2024:08:31:15 +0000] "GET /static/style.css HTTP/1.1" 304 0
""".strip()

apache_pattern = re.compile(r"""
    (?P<ip>\S+)\s+                   # Client IP
    \S+\s+                            # Identity (usually -)
    (?P<user>\S+)\s+                  # User
    \[(?P<time>[^\]]+)\]\s+           # Timestamp
    "(?P<method>\w+)\s+               # HTTP method
    (?P<path>\S+)\s+                  # Request path
    (?P<protocol>[^"]+)"\s+           # Protocol
    (?P<status>\d{3})\s+              # Status code
    (?P<size>\d+)                     # Response size
""", re.VERBOSE)

print("Access Log Analysis:")
print(f"{'IP':20s} {'Method':6s} {'Status':6s} {'Path'}")
print("-" * 60)
for match in apache_pattern.finditer(access_log):
    d = match.groupdict()
    print(f"{d['ip']:20s} {d['method']:6s} {d['status']:6s} {d['path']}")
```

---

## 2. Data Cleaning and Normalization

### Cleaning User Input

```python
import re

def clean_text(text):
    """Clean and normalize user-submitted text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize unicode quotation marks to ASCII
    text = re.sub(r'[\u201c\u201d]', '"', text)  # smart quotes
    text = re.sub(r'[\u2018\u2019]', "'", text)   # smart apostrophes
    text = re.sub(r'\u2014', '--', text)           # em dash
    text = re.sub(r'\u2013', '-', text)            # en dash

    # Collapse multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize line endings
    text = re.sub(r'\r\n|\r', '\n', text)

    # Remove more than 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace per line
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)

    return text.strip()

dirty = '''  <p>Hello   &nbsp;  World</p>

    This   has    too    much   space.


And   too   many   blank   lines.   '''

print(clean_text(dirty))
```

### Phone Number Normalization

```python
import re

def normalize_phone(phone):
    """Normalize various phone formats to (XXX) XXX-XXXX."""
    # Extract just the digits
    digits = re.sub(r'\D', '', phone)

    # Remove leading country code
    if len(digits) == 11 and digits[0] == '1':
        digits = digits[1:]

    if len(digits) != 10:
        return None

    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

phones = [
    "555-867-5309",
    "(555) 867.5309",
    "1-555-867-5309",
    "+1 555 867 5309",
    "5558675309",
    "555.867.5309",
]

for phone in phones:
    normalized = normalize_phone(phone)
    print(f"{phone:25s} -> {normalized}")
```

### CSV Data Cleaning

```python
import re

def clean_csv_field(field):
    """Clean a CSV field value."""
    # Remove leading/trailing whitespace and quotes
    field = field.strip().strip('"').strip("'")

    # Normalize internal whitespace
    field = re.sub(r'\s+', ' ', field)

    # Remove non-printable characters
    field = re.sub(r'[\x00-\x1f\x7f]', '', field)

    return field

raw_fields = [
    '  "  Hello   World  "  ',
    "  John's   Place  ",
    '  Line1\nLine2  ',
    '  \t Tab \t Separated \t ',
]

for field in raw_fields:
    print(f"'{field.strip()}' -> '{clean_csv_field(field)}'")
```

---

## 3. Data Extraction from Unstructured Text

### Extract Structured Data from a Report

```python
import re

report = """
Quarterly Sales Report - Q4 2024
=================================

Region: North America
  Revenue: $1,234,567.89
  Units Sold: 45,678
  Growth: +12.5%

Region: Europe
  Revenue: $987,654.32
  Units Sold: 32,100
  Growth: -3.2%

Region: Asia Pacific
  Revenue: $2,345,678.90
  Units Sold: 78,900
  Growth: +28.7%
"""

# Extract region data
region_pattern = re.compile(r"""
    Region:\s+(?P<region>.+?)\n
    \s+Revenue:\s+\$(?P<revenue>[\d,]+\.\d{2})\n
    \s+Units\s+Sold:\s+(?P<units>[\d,]+)\n
    \s+Growth:\s+(?P<growth>[+-]?\d+\.?\d*%)
""", re.VERBOSE)

print(f"{'Region':20s} {'Revenue':>15s} {'Units':>10s} {'Growth':>8s}")
print("-" * 55)
for match in region_pattern.finditer(report):
    d = match.groupdict()
    print(f"{d['region']:20s} ${d['revenue']:>14s} {d['units']:>10s} {d['growth']:>8s}")
```

### Extract Data from Email Headers

```python
import re

email_headers = """
From: John Smith <john.smith@example.com>
To: Jane Doe <jane.doe@company.org>, Bob <bob@test.com>
Cc: admin@example.com
Date: Mon, 15 Jan 2024 08:30:45 -0500
Subject: Re: Meeting Tomorrow
Message-ID: <abc123@mail.example.com>
"""

# Extract individual fields
from_match = re.search(r'^From:\s+(.+)$', email_headers, re.M)
to_match = re.search(r'^To:\s+(.+)$', email_headers, re.M)
subject_match = re.search(r'^Subject:\s+(.+)$', email_headers, re.M)

print(f"From:    {from_match.group(1)}")
print(f"To:      {to_match.group(1)}")
print(f"Subject: {subject_match.group(1)}")

# Extract all email addresses
all_emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', email_headers)
print(f"All emails: {all_emails}")
```

---

## 4. Code Refactoring with Regex

### Rename Variables

```python
import re

code = """
def calculate_total(item_price, item_count):
    item_total = item_price * item_count
    item_tax = item_total * 0.08
    return item_total + item_tax
"""

# Rename "item_" prefix to "product_"
refactored = re.sub(r'\bitem_(\w+)', r'product_\1', code)
print(refactored)
```

### Convert Print Statements to Logging

```python
import re

code = '''
print("Starting process")
print(f"Processing {filename}")
print("Error: " + str(e))
print(f"Completed in {elapsed:.2f}s")
'''

# Convert print() to logger calls
def convert_print_to_log(match):
    content = match.group(1)
    # Determine log level based on content
    if re.search(r'(?i)error|fail|exception', content):
        return f'logger.error({content})'
    elif re.search(r'(?i)warn', content):
        return f'logger.warning({content})'
    else:
        return f'logger.info({content})'

result = re.sub(r'print\((.+)\)', convert_print_to_log, code)
print(result)
```

### Modernize String Formatting

```python
import re

code = '''
msg = "Hello %s, you are %d years old" % (name, age)
info = "Count: %d, Price: %.2f" % (count, price)
log = "Status: %s" % status
'''

# Convert % formatting to f-strings (simplified)
def percent_to_fstring(match):
    template = match.group(1)
    args_str = match.group(2)

    # Parse format specifiers
    specs = re.findall(r'%(?:\.\d+)?[sdf]', template)

    # Parse arguments (simplified - single variables only)
    args = [a.strip() for a in args_str.split(',')]

    # Replace format specifiers with f-string expressions
    result = template
    for spec, arg in zip(specs, args):
        if '.' in spec:
            # Preserve format spec: %.2f -> {price:.2f}
            fmt = spec[1:]  # Remove %
            result = result.replace(spec, '{' + arg + ':' + fmt + '}', 1)
        else:
            result = result.replace(spec, '{' + arg + '}', 1)

    return f'f{result}'

result = re.sub(r'"([^"]*%[^"]*)" % \(([^)]+)\)', percent_to_fstring, code)
# Handle single-arg case
result = re.sub(r'"([^"]*%s[^"]*)" % (\w+)', percent_to_fstring, result)
print(result)
```

---

## 5. Configuration File Processing

### INI File Parser

```python
import re

ini_text = """
[database]
host = localhost
port = 5432
name = myapp
password = s3cret

[server]
host = 0.0.0.0
port = 8080
debug = true
workers = 4

# This is a comment
[logging]
level = INFO
file = /var/log/app.log
"""

def parse_ini(text):
    """Parse an INI file into a nested dictionary."""
    config = {}
    current_section = None

    section_pattern = re.compile(r'^\[(\w+)\]$')
    kv_pattern = re.compile(r'^(\w+)\s*=\s*(.+?)$')
    comment_pattern = re.compile(r'^\s*[#;]')

    for line in text.strip().split('\n'):
        line = line.strip()

        if not line or comment_pattern.match(line):
            continue

        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            config[current_section] = {}
            continue

        kv_match = kv_pattern.match(line)
        if kv_match and current_section:
            key, value = kv_match.groups()
            # Auto-convert types
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            config[current_section][key.strip()] = value

    return config

config = parse_ini(ini_text)
for section, values in config.items():
    print(f"[{section}]")
    for key, value in values.items():
        print(f"  {key} = {value} ({type(value).__name__})")
```

---

## 6. Building a Simple Tokenizer

### Arithmetic Expression Tokenizer

```python
import re

def tokenize(expression):
    """Tokenize a mathematical expression."""
    token_spec = [
        ('NUMBER',   r'\d+\.?\d*'),      # Integer or float
        ('PLUS',     r'\+'),              # Addition
        ('MINUS',    r'-'),               # Subtraction
        ('TIMES',    r'\*'),              # Multiplication
        ('DIVIDE',   r'/'),               # Division
        ('LPAREN',   r'\('),              # Left parenthesis
        ('RPAREN',   r'\)'),              # Right parenthesis
        ('SKIP',     r'[ \t]+'),          # Whitespace (skip)
        ('MISMATCH', r'.'),               # Any other character (error)
    ]

    # Build combined pattern
    combined = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    master_pattern = re.compile(combined)

    tokens = []
    for match in master_pattern.finditer(expression):
        kind = match.lastgroup
        value = match.group()
        if kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise ValueError(f"Unexpected character: {value!r}")
        elif kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        tokens.append((kind, value))

    return tokens

# Test
expr = "3.14 * (2 + 5) - 10 / 3"
tokens = tokenize(expr)
for token in tokens:
    print(f"  {token[0]:10s} : {token[1]}")
```

### Simple Markdown to HTML Converter

```python
import re

def markdown_to_html(text):
    """Convert basic Markdown to HTML."""
    # Headers: # Title -> <h1>Title</h1>
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.M)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.M)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.M)

    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Italic: *text* -> <em>text</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Code: `code` -> <code>code</code>
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)

    # Links: [text](url) -> <a href="url">text</a>
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)

    # Paragraphs: blank line separates paragraphs
    text = re.sub(r'\n\n+', '</p>\n<p>', text)
    text = f'<p>{text}</p>'

    return text

md = """# Hello World

This is **bold** and *italic* text.

Check `code` and [link](https://example.com).

## Section Two

Another paragraph here."""

print(markdown_to_html(md))
```

---

## 7. Batch File Renaming (Simulation)

```python
import re

def simulate_rename(filenames, pattern, replacement):
    """Simulate batch file renaming with regex."""
    results = []
    for name in filenames:
        new_name = re.sub(pattern, replacement, name)
        if new_name != name:
            results.append((name, new_name))
    return results

# Example: normalize photo filenames
files = [
    "IMG_20240115_083045.jpg",
    "IMG_20240116_142030.jpg",
    "Screenshot 2024-01-17 at 10.30.00.png",
    "photo (1).jpg",
    "photo (2).jpg",
]

# Convert IMG_YYYYMMDD_HHMMSS to YYYY-MM-DD_HH-MM-SS
renames = simulate_rename(
    files,
    r'IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
    r'\1-\2-\3_\4-\5-\6'
)

print("Proposed renames:")
for old, new in renames:
    print(f"  {old:45s} -> {new}")
```

---

## 8. Web Scraping Data Extraction

```python
import re

# Simulated HTML content (in practice, use BeautifulSoup)
html = """
<table class="products">
  <tr><th>Name</th><th>Price</th><th>Stock</th></tr>
  <tr><td>Widget A</td><td>$19.99</td><td>In Stock</td></tr>
  <tr><td>Widget B</td><td>$29.99</td><td>Out of Stock</td></tr>
  <tr><td>Widget C</td><td>$9.99</td><td>In Stock</td></tr>
  <tr><td>Widget D</td><td>$49.99</td><td>In Stock</td></tr>
</table>
"""

# Extract product data
row_pattern = re.compile(
    r'<tr><td>(.+?)</td><td>\$(.+?)</td><td>(.+?)</td></tr>'
)

products = []
for match in row_pattern.finditer(html):
    name, price, stock = match.groups()
    products.append({
        'name': name,
        'price': float(price),
        'in_stock': stock == 'In Stock'
    })

# Display results
print(f"{'Product':15s} {'Price':>8s} {'Available'}")
print("-" * 35)
for p in products:
    status = "Yes" if p['in_stock'] else "No"
    print(f"{p['name']:15s} ${p['price']:>7.2f} {status}")

# Calculate total value of in-stock items
in_stock_total = sum(p['price'] for p in products if p['in_stock'])
print(f"\nTotal in-stock value: ${in_stock_total:.2f}")
```

> **Important**: For real HTML parsing, always use a proper HTML parser like BeautifulSoup or lxml. Regex for HTML is shown here for educational purposes only.

---

## 9. Log Analysis Pipeline

```python
import re
from collections import defaultdict
from datetime import datetime

# Simulated multi-format logs
logs = """
2024-01-15T08:30:45Z [api-server] INFO  GET /api/users 200 45ms
2024-01-15T08:30:46Z [api-server] INFO  POST /api/login 401 12ms
2024-01-15T08:30:47Z [api-server] ERROR GET /api/orders 500 3002ms
2024-01-15T08:31:00Z [db-server]  WARN  Connection pool 80% utilized
2024-01-15T08:31:02Z [api-server] INFO  GET /api/users 200 38ms
2024-01-15T08:31:05Z [api-server] INFO  GET /api/products 200 52ms
2024-01-15T08:31:10Z [api-server] ERROR POST /api/orders 500 5001ms
2024-01-15T08:31:15Z [api-server] INFO  GET /health 200 2ms
""".strip()

# Parse HTTP request logs
http_pattern = re.compile(r"""
    (?P<timestamp>\S+)\s+
    \[(?P<server>[^\]]+)\]\s+
    (?P<level>\w+)\s+
    (?P<method>GET|POST|PUT|DELETE|PATCH)\s+
    (?P<path>\S+)\s+
    (?P<status>\d{3})\s+
    (?P<duration>\d+)ms
""", re.VERBOSE)

# Collect metrics
endpoint_stats = defaultdict(lambda: {'count': 0, 'errors': 0, 'total_ms': 0})
slow_requests = []

for match in http_pattern.finditer(logs):
    d = match.groupdict()
    key = f"{d['method']} {d['path']}"
    duration = int(d['duration'])
    status = int(d['status'])

    endpoint_stats[key]['count'] += 1
    endpoint_stats[key]['total_ms'] += duration
    if status >= 400:
        endpoint_stats[key]['errors'] += 1
    if duration > 1000:
        slow_requests.append(d)

# Report
print("Endpoint Statistics:")
print(f"{'Endpoint':30s} {'Requests':>8s} {'Errors':>6s} {'Avg ms':>8s}")
print("-" * 55)
for endpoint, stats in sorted(endpoint_stats.items()):
    avg = stats['total_ms'] / stats['count']
    print(f"{endpoint:30s} {stats['count']:>8d} {stats['errors']:>6d} {avg:>8.0f}")

if slow_requests:
    print(f"\nSlow Requests (>1s): {len(slow_requests)}")
    for r in slow_requests:
        print(f"  {r['timestamp']} {r['method']} {r['path']} -> {r['duration']}ms")
```

---

## 10. Building a Complete Text Processor

```python
import re

class TextProcessor:
    """A configurable text processor using regex rules."""

    def __init__(self):
        self.rules = []

    def add_rule(self, name, pattern, replacement, flags=0):
        """Add a processing rule."""
        compiled = re.compile(pattern, flags)
        self.rules.append((name, compiled, replacement))

    def process(self, text, verbose=False):
        """Apply all rules to the text."""
        for name, pattern, replacement in self.rules:
            new_text = pattern.sub(replacement, text)
            if verbose and new_text != text:
                print(f"  Applied: {name}")
            text = new_text
        return text

# Configure processor
proc = TextProcessor()
proc.add_rule("strip_html", r'<[^>]+>', '')
proc.add_rule("normalize_whitespace", r'\s+', ' ')
proc.add_rule("fix_punctuation_space", r'\s+([.,!?;:])', r'\1')
proc.add_rule("capitalize_sentences", r'(?:^|[.!?]\s+)([a-z])',
              lambda m: m.group().upper())
proc.add_rule("trim", r'^\s+|\s+$', '')

# Process some messy text
messy = """
  <p>hello world .  this is a   test   !</p>
  <br>  multiple    spaces   and  <b>tags</b>  everywhere .
"""

clean = proc.process(messy, verbose=True)
print(f"\nResult: '{clean}'")
```

---

## Summary

| Application | Key Techniques Used |
|-------------|-------------------|
| Log parsing | Named groups, VERBOSE, MULTILINE, finditer |
| Data cleaning | sub with callbacks, character classes, anchors |
| Data extraction | Capture groups, findall, named groups |
| Code refactoring | Backreferences, word boundaries, sub |
| Config parsing | MULTILINE, line-by-line matching |
| Tokenization | Named groups, alternation, finditer |
| Batch rename | sub with group references |

Final advice:
- Always start with the simplest pattern that works
- Use `re.VERBOSE` to document complex patterns
- Test with edge cases and malformed input
- Consider dedicated parsers for structured formats (HTML, JSON, CSV)
- Combine regex with Python's string methods for best results
- Profile performance when processing large datasets

---

## Course Complete!

Congratulations on completing the Regular Expressions course! You now have a solid foundation in pattern matching and text processing. Continue practicing with real-world data to build fluency.

For further learning:
- [Python `re` module documentation](https://docs.python.org/3/library/re.html)
- [Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html)
- Practice at [regex101.com](https://regex101.com/) with Python flavor
