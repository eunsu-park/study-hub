"""
10 Common Patterns
===================
Demonstrates email, URL, IP, date, phone, and password patterns.
"""

import re


def email_validation():
    """Validate email addresses."""
    pattern = re.compile(r'^[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}$')
    tests = ["user@example.com", "a+b@sub.domain.org", "@bad", "no-tld@x"]
    for email in tests:
        print(f"  {email:30s} -> {'Valid' if pattern.match(email) else 'Invalid'}")


def url_extraction():
    """Extract URLs from text."""
    text = "Visit https://example.com/path?q=1 and http://test.com."
    urls = re.findall(r'https?://\S+', text)
    urls = [re.sub(r'[.,]+$', '', u) for u in urls]
    print(f"URLs: {urls}")


def ipv4_validation():
    """Validate IPv4 addresses with octet range checking."""
    octet = r'(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)'
    pattern = re.compile(f'^{octet}(?:\\.{octet}){{3}}$')
    for ip in ["192.168.1.1", "255.255.255.255", "256.0.0.1", "1.2.3"]:
        print(f"  {ip:20s} -> {'Valid' if pattern.match(ip) else 'Invalid'}")


def date_validation():
    """Validate various date formats."""
    iso = re.compile(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$')
    for d in ["2024-01-15", "2024-13-01", "2024-01-32", "2024-12-31"]:
        print(f"  {d}: {'Valid' if iso.match(d) else 'Invalid'}")


def phone_normalization():
    """Normalize various phone number formats."""
    def normalize(phone):
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 11 and digits[0] == '1':
            digits = digits[1:]
        if len(digits) != 10:
            return None
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

    phones = ["555-867-5309", "(555) 867.5309", "+1 555 867 5309", "5558675309"]
    for p in phones:
        print(f"  {p:25s} -> {normalize(p)}")


def password_validation():
    """Validate password strength."""
    rules = [
        (r'.{8,}', "8+ chars"),
        (r'[A-Z]', "uppercase"),
        (r'[a-z]', "lowercase"),
        (r'\d', "digit"),
        (r'[!@#$%^&*]', "special"),
    ]
    for pwd in ["P@ssw0rd", "weak", "NoSpecial1"]:
        failed = [d for p, d in rules if not re.search(p, pwd)]
        status = "Strong" if not failed else f"Weak (missing: {', '.join(failed)})"
        print(f"  '{pwd}': {status}")


if __name__ == "__main__":
    sections = [
        ("Email Validation", email_validation),
        ("URL Extraction", url_extraction),
        ("IPv4 Validation", ipv4_validation),
        ("Date Validation", date_validation),
        ("Phone Normalization", phone_normalization),
        ("Password Validation", password_validation),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
