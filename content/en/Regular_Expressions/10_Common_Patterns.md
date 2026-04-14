# Common Patterns

## Learning Objectives

After completing this lesson, you will be able to:

1. Build and understand email validation patterns
2. Parse and validate URLs with regex
3. Match IPv4 and IPv6 address formats
4. Validate various date and time formats
5. Match phone numbers in multiple formats
6. Create patterns for passwords, usernames, and other common validations
7. Understand the trade-offs between strictness and practicality in validation
8. Know when regex is NOT the right tool for validation

---

## 1. Email Address Validation

Email validation with regex ranges from simple to impossibly complex. Here's a practical approach:

### Simple (Good Enough for Most Cases)

```python
import re

email_pattern = re.compile(r"""
    ^[\w.+-]+           # Local part
    @                   # @ symbol
    [\w-]+              # Domain
    (?:\.[\w-]+)*       # Subdomains
    \.[a-zA-Z]{2,}      # TLD
    $
""", re.VERBOSE)

test_emails = [
    ("user@example.com", True),
    ("first.last@domain.org", True),
    ("user+tag@sub.domain.co.uk", True),
    ("@missing-local.com", False),
    ("missing-domain@", False),
    ("no-tld@domain", False),
    ("user@domain.c", False),
]

for email, expected in test_emails:
    result = bool(email_pattern.match(email))
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] {email:35s} -> {result}")
```

```
Email Pattern Breakdown:

    user.name+tag@sub.domain.com
    ─────────────┬──────────────
    [\w.+-]+     @  [\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}
    local part      domain with TLD

    ^[\w.+-]+     : one or more word chars, dots, plus, hyphen
    @             : literal @
    [\w-]+        : domain label
    (?:\.[\w-]+)* : optional subdomains (.sub.domain)
    \.[a-zA-Z]{2,}: TLD (.com, .org, .uk)
    $             : end of string
```

> **Note**: The RFC 5322 compliant email regex is thousands of characters long. The pattern above handles 99%+ of real-world emails. For production use, consider a dedicated email validation library.

---

## 2. URL Validation

### Basic URL Pattern

```python
import re

url_pattern = re.compile(r"""
    ^(?:(?P<scheme>https?|ftp)://)   # Scheme
    (?P<host>                         # Host
        (?:[\w-]+\.)+                 # Domain labels
        [a-zA-Z]{2,}                  # TLD
        |                             # OR
        \d{1,3}(?:\.\d{1,3}){3}      # IPv4 address
    )
    (?::(?P<port>\d{1,5}))?          # Optional port
    (?P<path>/[^\s?#]*)?             # Optional path
    (?:\?(?P<query>[^\s#]*))?        # Optional query
    (?:\#(?P<fragment>\S*))?         # Optional fragment
    $
""", re.VERBOSE)

urls = [
    "https://www.example.com",
    "http://example.com:8080/path?q=1#section",
    "ftp://files.example.com/pub/data.zip",
    "https://192.168.1.1:3000/api",
    "not-a-url",
]

for url in urls:
    match = url_pattern.match(url)
    if match:
        d = match.groupdict()
        print(f"Valid: {url}")
        print(f"  scheme={d['scheme']}, host={d['host']}, "
              f"port={d['port']}, path={d['path']}")
    else:
        print(f"Invalid: {url}")
```

### Extract URLs from Text

```python
import re

text = """
Visit https://www.google.com for search.
Check http://example.com:8080/api/v1 for the API.
Download from ftp://files.example.com/data.zip.
Not a URL: www.no-scheme.com
"""

# Simple URL extraction
urls = re.findall(r'https?://\S+', text)
# Clean trailing punctuation
urls = [re.sub(r'[.,;:!?)]+$', '', u) for u in urls]
print(urls)
# ['https://www.google.com', 'http://example.com:8080/api/v1']
```

---

## 3. IPv4 Address Validation

### Basic Format Check

```python
import re

# Simple: matches format but allows invalid octets (e.g., 999.999.999.999)
simple_ip = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'

# Strict: validates each octet is 0-255
strict_ip = re.compile(r"""
    ^
    (?:
        (?:25[0-5])           # 250-255
        |(?:2[0-4]\d)         # 200-249
        |(?:1\d{2})           # 100-199
        |(?:[1-9]\d)          # 10-99
        |(?:\d)               # 0-9
    )
    (?:\.                     # Followed by dot and another octet
        (?:
            (?:25[0-5])
            |(?:2[0-4]\d)
            |(?:1\d{2})
            |(?:[1-9]\d)
            |(?:\d)
        )
    ){3}                      # Exactly 3 more octets
    $
""", re.VERBOSE)

test_ips = [
    ("192.168.1.1", True),
    ("10.0.0.1", True),
    ("255.255.255.255", True),
    ("0.0.0.0", True),
    ("256.1.1.1", False),
    ("192.168.1", False),
    ("192.168.1.1.1", False),
]

for ip, expected in test_ips:
    result = bool(strict_ip.match(ip))
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] {ip:20s} -> {result}")
```

```
IPv4 Octet Breakdown:

    Value Range    Pattern
    ───────────    ───────
    0-9            \d
    10-99          [1-9]\d
    100-199        1\d{2}
    200-249        2[0-4]\d
    250-255        25[0-5]

    Combined: (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)
```

---

## 4. Date and Time Validation

### Date Formats

```python
import re

# YYYY-MM-DD (ISO 8601)
iso_date = re.compile(r"""
    ^(?P<year>\d{4})          # Year
    -(?P<month>0[1-9]|1[0-2]) # Month (01-12)
    -(?P<day>0[1-9]|[12]\d|3[01])  # Day (01-31)
    $
""", re.VERBOSE)

# MM/DD/YYYY (US format)
us_date = re.compile(r"""
    ^(?P<month>0[1-9]|1[0-2])
    /(?P<day>0[1-9]|[12]\d|3[01])
    /(?P<year>\d{4})
    $
""", re.VERBOSE)

# DD.MM.YYYY (European format)
eu_date = re.compile(r"""
    ^(?P<day>0[1-9]|[12]\d|3[01])
    \.(?P<month>0[1-9]|1[0-2])
    \.(?P<year>\d{4})
    $
""", re.VERBOSE)

dates = ["2024-01-15", "01/15/2024", "15.01.2024", "2024-13-01", "2024-01-32"]
for d in dates:
    for name, pattern in [("ISO", iso_date), ("US", us_date), ("EU", eu_date)]:
        if pattern.match(d):
            print(f"{d} matches {name} format")
```

### Time Format

```python
import re

# HH:MM:SS (24-hour)
time_24h = re.compile(r"""
    ^(?P<hour>[01]\d|2[0-3])      # Hour (00-23)
    :(?P<minute>[0-5]\d)           # Minute (00-59)
    (?::(?P<second>[0-5]\d))?      # Optional seconds
    $
""", re.VERBOSE)

# HH:MM AM/PM (12-hour)
time_12h = re.compile(r"""
    ^(?P<hour>0?[1-9]|1[0-2])     # Hour (1-12)
    :(?P<minute>[0-5]\d)           # Minute (00-59)
    \s*(?P<period>[AaPp][Mm])      # AM/PM
    $
""", re.VERBOSE)

times = ["08:30:45", "23:59", "12:00 PM", "3:30 am", "25:00"]
for t in times:
    if time_24h.match(t):
        print(f"{t} -> 24-hour format")
    elif time_12h.match(t):
        print(f"{t} -> 12-hour format")
    else:
        print(f"{t} -> Invalid")
```

---

## 5. Phone Number Patterns

```python
import re

# Flexible US phone number
us_phone = re.compile(r"""
    ^
    (?:\+?1[\s.-]?)?          # Optional country code
    (?:\(?(\d{3})\)?[\s.-]?)  # Area code (with optional parens)
    (\d{3})                    # Exchange
    [\s.-]?                    # Separator
    (\d{4})                    # Subscriber
    $
""", re.VERBOSE)

phones = [
    "555-867-5309",
    "(555) 867-5309",
    "555.867.5309",
    "+1 555 867 5309",
    "1-555-867-5309",
    "5558675309",
]

for phone in phones:
    match = us_phone.match(phone)
    if match:
        area, exchange, subscriber = match.groups()
        print(f"{phone:25s} -> ({area}) {exchange}-{subscriber}")
    else:
        print(f"{phone:25s} -> No match")
```

---

## 6. Password Validation

```python
import re

def validate_password(password):
    """Validate password strength with detailed feedback."""
    rules = [
        (r'.{8,}', "At least 8 characters"),
        (r'[A-Z]', "At least one uppercase letter"),
        (r'[a-z]', "At least one lowercase letter"),
        (r'\d', "At least one digit"),
        (r'[!@#$%^&*(),.?":{}|<>]', "At least one special character"),
        (r'^[^\s]+$', "No whitespace"),
    ]

    passed = True
    for pattern, description in rules:
        if not re.search(pattern, password):
            print(f"  FAIL: {description}")
            passed = False

    return passed

passwords = ["P@ssw0rd!", "password", "SHORT1!", "NoSpecial1", "Has Space1!"]
for pwd in passwords:
    print(f"\n'{pwd}':")
    result = validate_password(pwd)
    print(f"  Result: {'STRONG' if result else 'WEAK'}")
```

---

## 7. Username and Identifier Patterns

```python
import re

# Username: 3-20 chars, alphanumeric + underscore, must start with letter
username_pattern = re.compile(r'^[a-zA-Z]\w{2,19}$')

# Slug: lowercase, hyphens, no consecutive hyphens
slug_pattern = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')

# Semantic version: MAJOR.MINOR.PATCH
semver_pattern = re.compile(r"""
    ^(?P<major>0|[1-9]\d*)
    \.(?P<minor>0|[1-9]\d*)
    \.(?P<patch>0|[1-9]\d*)
    (?:-(?P<pre>[a-zA-Z0-9.]+))?    # Optional pre-release
    (?:\+(?P<build>[a-zA-Z0-9.]+))? # Optional build metadata
    $
""", re.VERBOSE)

# Test usernames
for name in ["alice", "Bob_99", "ab", "1alice", "a" * 21]:
    print(f"Username '{name}': {bool(username_pattern.match(name))}")

# Test slugs
for slug in ["my-blog-post", "hello", "Bad Slug", "double--dash"]:
    print(f"Slug '{slug}': {bool(slug_pattern.match(slug))}")

# Test semver
for ver in ["1.0.0", "2.1.3-beta.1", "0.0.1+build.123", "1.0"]:
    print(f"Version '{ver}': {bool(semver_pattern.match(ver))}")
```

---

## 8. Data Extraction Patterns

### Credit Card Numbers (Format Only)

```python
import re

# Common credit card formats (NOT validation -- just format detection)
cc_pattern = re.compile(r"""
    (?:
        (?P<visa>4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})          # Visa
        |(?P<mastercard>5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})  # Mastercard
        |(?P<amex>3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5})                   # Amex
    )
""", re.VERBOSE)

cards = [
    "4111-1111-1111-1111",   # Visa
    "5500 0000 0000 0004",   # Mastercard
    "3782 822463 10005",     # Amex
]

for card in cards:
    match = cc_pattern.search(card)
    if match:
        card_type = next(k for k, v in match.groupdict().items() if v)
        print(f"{card} -> {card_type}")
```

### Extracting Markdown Links

```python
import re

markdown = """
Check out [Google](https://www.google.com) and
[Python docs](https://docs.python.org/3/).
Also see [local page](./about.md).
"""

# Extract [text](url) patterns
links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown)
for text, url in links:
    print(f"Text: {text:15s} URL: {url}")
```

---

## 9. When NOT to Use Regex

Regex is powerful but not always the best tool:

```python
# BAD: Parsing HTML with regex
# HTML is not a regular language -- use a proper parser
# from html.parser import HTMLParser
# or: from bs4 import BeautifulSoup

# BAD: Validating JSON with regex
# Use: import json; json.loads(text)

# BAD: Parsing complex date formats
# Use: from datetime import datetime; datetime.strptime(text, fmt)

# BAD: Validating email (production use)
# Use: email-validator library or send a confirmation email

# BAD: Complex arithmetic expressions
# Use: ast.literal_eval() or a proper parser

# GOOD use cases for regex:
# - Quick text search and extraction
# - Log file parsing
# - Data cleaning (whitespace, formatting)
# - Simple validation (format checking)
# - Find and replace in text
# - Tokenization
```

---

## 10. Pattern Reference Card

```
Pattern                 Purpose                  Example Match
───────                 ───────                  ─────────────
[\w.+-]+@[\w-]+\.\w+   Email (basic)            user@example.com
https?://\S+            URL (basic)              https://example.com/path
\d{1,3}(\.\d{1,3}){3}  IPv4 (format)            192.168.1.1
\d{4}-\d{2}-\d{2}      Date ISO                 2024-01-15
\d{2}:\d{2}(:\d{2})?   Time                     08:30:45
\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}  US Phone   (555) 867-5309
#[0-9a-fA-F]{3,6}      Hex color                #FF0000
\d+\.\d+\.\d+          Semantic version          1.2.3
\b[A-Z][a-z]+\b        Capitalized word          Hello
```

---

## Summary

| Category | Recommended Approach |
|----------|---------------------|
| Email | Simple regex for format check; library for production validation |
| URL | Regex for extraction; `urllib.parse` for parsing |
| IP Address | Regex for format; `ipaddress` module for validation |
| Date/Time | Regex for extraction; `datetime` for parsing and validation |
| Phone | Regex with flexible separators; normalize after matching |
| Password | Multiple regex checks (one per rule) |

Key principles:
- Start simple, add complexity only as needed
- Use `re.VERBOSE` for any pattern over 30 characters
- Test with both valid and invalid inputs
- Consider edge cases (empty strings, boundary values)
- Know when to use a proper parser instead of regex

---

## Next Lesson

In [11_Performance_and_Pitfalls](./11_Performance_and_Pitfalls.md), we'll learn about regex performance issues and how to avoid them.
