"""
URL Shortener

Demonstrates:
- Base62 encoding for short URLs
- Hash-based ID generation
- Collision handling
- Analytics tracking (click counts, referrers)

Theory:
- URL shorteners map long URLs to short codes.
- Base62 (a-z, A-Z, 0-9) encodes numeric IDs into compact strings.
  6 chars = 62^6 = ~56 billion unique URLs.
- Approaches:
  1. Auto-increment ID → Base62 encode (predictable but simple)
  2. Hash-based (MD5/SHA256 truncated) — may collide
  3. Random generation with collision check
- Considerations: custom aliases, expiration, analytics.

Adapted from System Design Lesson 17.
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field


# ── Base62 Encoding ────────────────────────────────────────────────────

# Why: Base62 (digits + lowercase + uppercase) is the standard for URL shorteners
# because all characters are URL-safe without percent-encoding. Base64 includes
# '+' and '/' which require escaping in URLs, making them user-unfriendly.
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = len(CHARSET)  # 62


def base62_encode(num: int) -> str:
    """Encode a non-negative integer to base62 string."""
    if num == 0:
        return CHARSET[0]
    result = []
    while num > 0:
        result.append(CHARSET[num % BASE])
        num //= BASE
    return "".join(reversed(result))


def base62_decode(s: str) -> int:
    """Decode a base62 string to integer."""
    num = 0
    for ch in s:
        num = num * BASE + CHARSET.index(ch)
    return num


# ── URL Shortener ──────────────────────────────────────────────────────

@dataclass
class URLEntry:
    short_code: str
    long_url: str
    created_at: float
    expires_at: float | None = None
    click_count: int = 0
    referrers: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class URLShortener:
    """URL shortener service."""

    def __init__(self, domain: str = "short.url", code_length: int = 6):
        self.domain = domain
        self.code_length = code_length
        self.url_map: dict[str, URLEntry] = {}  # short_code → entry
        # Why: The reverse map enables deduplication — if the same long URL is
        # shortened twice, we return the existing short code instead of creating
        # a duplicate. This saves storage and keeps analytics consolidated.
        self.reverse_map: dict[str, str] = {}   # long_url → short_code
        self.next_id = 1
        self.total_redirects = 0

    def shorten_sequential(self, long_url: str,
                           ttl_seconds: float | None = None) -> str:
        """Shorten using auto-increment ID + Base62."""
        # Why: Sequential IDs guarantee zero collisions and produce the shortest
        # possible codes. The trade-off: codes are predictable (enumerable), which
        # may be a security concern if short URLs should not be guessable.
        if long_url in self.reverse_map:
            return self._format_url(self.reverse_map[long_url])

        code = base62_encode(self.next_id)
        self.next_id += 1
        # Pad to minimum length
        code = code.zfill(self.code_length)[-self.code_length:]

        now = time.monotonic()
        expires = now + ttl_seconds if ttl_seconds else None
        entry = URLEntry(code, long_url, created_at=now, expires_at=expires)
        self.url_map[code] = entry
        self.reverse_map[long_url] = code
        return self._format_url(code)

    def shorten_hash(self, long_url: str) -> str:
        """Shorten using MD5 hash truncation."""
        if long_url in self.reverse_map:
            return self._format_url(self.reverse_map[long_url])

        h = hashlib.md5(long_url.encode()).hexdigest()
        # Why: Using only 48 bits (12 hex chars) of the MD5 hash keeps codes
        # short while still providing ~281 trillion possible values — far more
        # than most URL shorteners will ever need.
        num = int(h[:12], 16)  # Use 48 bits
        code = base62_encode(num)[:self.code_length]

        # Why: Hash collisions are handled by appending a counter and rehashing.
        # This is simpler than open addressing and guarantees we eventually find
        # a free slot without changing the hash function itself.
        attempts = 0
        original_code = code
        while code in self.url_map:
            attempts += 1
            h = hashlib.md5(f"{long_url}#{attempts}".encode()).hexdigest()
            num = int(h[:12], 16)
            code = base62_encode(num)[:self.code_length]

        entry = URLEntry(code, long_url, created_at=time.monotonic())
        self.url_map[code] = entry
        self.reverse_map[long_url] = code

        if attempts > 0:
            return self._format_url(code) + f" (resolved {attempts} collision(s))"
        return self._format_url(code)

    def shorten_custom(self, long_url: str, custom_alias: str) -> str | None:
        """Shorten with a custom alias."""
        if custom_alias in self.url_map:
            return None  # Alias taken
        entry = URLEntry(custom_alias, long_url, created_at=time.monotonic())
        self.url_map[custom_alias] = entry
        self.reverse_map[long_url] = custom_alias
        return self._format_url(custom_alias)

    def resolve(self, short_code: str, referrer: str = "direct") -> str | None:
        """Resolve short URL to long URL."""
        entry = self.url_map.get(short_code)
        if not entry:
            return None

        # Check expiration
        if entry.expires_at and time.monotonic() > entry.expires_at:
            return None

        entry.click_count += 1
        entry.referrers[referrer] += 1
        self.total_redirects += 1
        return entry.long_url

    def get_analytics(self, short_code: str) -> dict | None:
        entry = self.url_map.get(short_code)
        if not entry:
            return None
        return {
            "short_code": entry.short_code,
            "long_url": entry.long_url,
            "clicks": entry.click_count,
            "referrers": dict(entry.referrers),
        }

    def _format_url(self, code: str) -> str:
        return f"https://{self.domain}/{code}"


# ── Demos ──────────────────────────────────────────────────────────────

def demo_base62():
    print("=" * 60)
    print("BASE62 ENCODING")
    print("=" * 60)

    print(f"\n  Base62 charset: {CHARSET}")
    print(f"  62 chars → 62^N combinations per N-length code\n")

    test_values = [0, 1, 61, 62, 1000, 56800235584, 999999999999]
    print(f"  {'Number':>15}  {'Base62':>10}  {'Decoded':>15}  {'Match':>6}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*15}  {'-'*6}")
    for num in test_values:
        encoded = base62_encode(num)
        decoded = base62_decode(encoded)
        print(f"  {num:>15}  {encoded:>10}  {decoded:>15}  "
              f"{'✓' if decoded == num else '✗':>6}")

    # Capacity analysis
    print(f"\n  Capacity by code length:")
    for length in range(4, 9):
        capacity = BASE ** length
        print(f"    {length} chars: {capacity:>15,} URLs "
              f"({capacity / 1e9:.1f} billion)")


def demo_sequential():
    print("\n" + "=" * 60)
    print("SEQUENTIAL ID SHORTENER")
    print("=" * 60)

    svc = URLShortener(domain="sho.rt")

    urls = [
        "https://example.com/very/long/path/to/resource?param=value",
        "https://docs.python.org/3/library/hashlib.html",
        "https://en.wikipedia.org/wiki/URL_shortening",
        "https://example.com/another/page",
    ]

    print(f"\n  Shortening URLs (sequential ID):")
    for url in urls:
        short = svc.shorten_sequential(url)
        print(f"    {url[:50]:<50} → {short}")

    # Duplicate returns same short URL
    short = svc.shorten_sequential(urls[0])
    print(f"\n  Duplicate: same short URL → {short}")


def demo_hash_based():
    print("\n" + "=" * 60)
    print("HASH-BASED SHORTENER")
    print("=" * 60)

    svc = URLShortener(domain="sho.rt")

    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]

    print(f"\n  Shortening URLs (hash-based):")
    for url in urls:
        short = svc.shorten_hash(url)
        print(f"    {url:<35} → {short}")


def demo_custom_alias():
    print("\n" + "=" * 60)
    print("CUSTOM ALIAS")
    print("=" * 60)

    svc = URLShortener(domain="sho.rt")

    print(f"\n  Custom aliases:")
    result = svc.shorten_custom("https://example.com/my-portfolio", "portfolio")
    print(f"    'portfolio' → {result}")

    result = svc.shorten_custom("https://example.com/blog", "blog")
    print(f"    'blog' → {result}")

    # Collision
    result = svc.shorten_custom("https://other.com", "blog")
    print(f"    'blog' (taken) → {result}")


def demo_analytics():
    print("\n" + "=" * 60)
    print("URL ANALYTICS")
    print("=" * 60)

    svc = URLShortener(domain="sho.rt")
    short = svc.shorten_sequential("https://example.com/landing-page")
    code = short.split("/")[-1]

    # Simulate clicks
    clicks = [
        ("google.com", 15),
        ("twitter.com", 8),
        ("direct", 5),
        ("facebook.com", 3),
    ]

    print(f"\n  URL: {short}")
    print(f"  Simulating clicks:")
    for referrer, count in clicks:
        for _ in range(count):
            svc.resolve(code, referrer)

    analytics = svc.get_analytics(code)
    print(f"\n  Analytics:")
    print(f"    Total clicks: {analytics['clicks']}")
    print(f"    Referrers:")
    for ref, count in sorted(analytics['referrers'].items(),
                              key=lambda x: -x[1]):
        bar = "█" * count
        print(f"      {ref:<15} {count:>3} {bar}")


def demo_comparison():
    print("\n" + "=" * 60)
    print("APPROACH COMPARISON")
    print("=" * 60)

    print(f"""
  {'Approach':<20} {'Predictable':>12} {'Collisions':>11} {'Custom':>7} {'Speed':>6}
  {'-'*20} {'-'*12} {'-'*11} {'-'*7} {'-'*6}
  {'Sequential ID':<20} {'Yes':>12} {'None':>11} {'No':>7} {'Fast':>6}
  {'Hash-based':<20} {'No':>12} {'Possible':>11} {'No':>7} {'Fast':>6}
  {'Random':<20} {'No':>12} {'Possible':>11} {'No':>7} {'Med':>6}
  {'Custom alias':<20} {'N/A':>12} {'Check req':>11} {'Yes':>7} {'Fast':>6}

  Notes:
  - Sequential: simple, but IDs are guessable (security concern)
  - Hash-based: deterministic, same URL always gets same code
  - Random: unpredictable, but needs collision check
  - Custom: best UX, but limited availability""")


if __name__ == "__main__":
    demo_base62()
    demo_sequential()
    demo_hash_based()
    demo_custom_alias()
    demo_analytics()
    demo_comparison()
