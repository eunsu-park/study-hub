"""
Exercises for Lesson 17: Practical Design Examples 1
Topic: System_Design

Solutions to practice problems from the lesson.
Covers URL shortener extensions, Pastebin security, and dynamic rate limiting.
"""

import hashlib
import time
import random
import string
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# === Exercise 1: URL Shortener Extension ===
# Problem: Design extensions with geo-redirect, A/B testing, analytics dashboard.

class URLShortenerExtended:
    """URL shortener with geo-redirect, A/B testing, and analytics."""

    def __init__(self):
        self.urls = {}  # short_code -> URLEntry
        self.clicks = defaultdict(list)  # short_code -> [ClickEvent]

    def shorten(self, original_url, config=None):
        """Create shortened URL with optional config."""
        short_code = self._generate_code()
        self.urls[short_code] = URLEntry(
            short_code=short_code,
            original_url=original_url,
            config=config or {},
        )
        return short_code

    def resolve(self, short_code, country="US", user_segment=None):
        """Resolve short URL with geo-redirect and A/B support."""
        entry = self.urls.get(short_code)
        if not entry:
            return None

        # Record click
        click = ClickEvent(
            timestamp=time.time(),
            country=country,
            user_segment=user_segment,
        )
        self.clicks[short_code].append(click)

        # Geo-redirect
        geo_urls = entry.config.get("geo_urls", {})
        if country in geo_urls:
            click.resolved_url = geo_urls[country]
            return geo_urls[country]

        # A/B testing
        ab_config = entry.config.get("ab_test", None)
        if ab_config:
            if random.random() < ab_config.get("split", 0.5):
                click.resolved_url = ab_config["url_a"]
                click.variant = "A"
                return ab_config["url_a"]
            else:
                click.resolved_url = ab_config["url_b"]
                click.variant = "B"
                return ab_config["url_b"]

        click.resolved_url = entry.original_url
        return entry.original_url

    def get_analytics(self, short_code):
        """Get analytics for a short URL."""
        clicks = self.clicks.get(short_code, [])
        if not clicks:
            return {"total_clicks": 0}

        country_dist = defaultdict(int)
        variant_dist = defaultdict(int)
        hourly = defaultdict(int)

        for click in clicks:
            country_dist[click.country] += 1
            if click.variant:
                variant_dist[click.variant] += 1
            hour = int(click.timestamp) // 3600
            hourly[hour] += 1

        return {
            "total_clicks": len(clicks),
            "by_country": dict(country_dist),
            "by_variant": dict(variant_dist),
            "hourly_distribution": len(hourly),
        }

    def _generate_code(self):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=7))


@dataclass
class URLEntry:
    short_code: str
    original_url: str
    config: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ClickEvent:
    timestamp: float
    country: str
    user_segment: Optional[str] = None
    resolved_url: str = ""
    variant: Optional[str] = None


def exercise_1():
    """URL shortener with extensions."""
    print("URL Shortener Extensions:")
    print("=" * 60)

    shortener = URLShortenerExtended()

    # 1. Geo-redirect
    print("\n--- Geo-Redirect ---")
    code = shortener.shorten("https://example.com", config={
        "geo_urls": {
            "KR": "https://example.com/ko",
            "JP": "https://example.com/ja",
            "US": "https://example.com/en",
        }
    })

    for country in ["US", "KR", "JP", "DE"]:
        url = shortener.resolve(code, country=country)
        print(f"  {country} -> {url}")

    # 2. A/B Testing
    print("\n--- A/B Testing ---")
    ab_code = shortener.shorten("https://landing.com", config={
        "ab_test": {
            "url_a": "https://landing.com/v1",
            "url_b": "https://landing.com/v2",
            "split": 0.5,
        }
    })

    random.seed(42)
    for _ in range(100):
        shortener.resolve(ab_code, country="US")

    analytics = shortener.get_analytics(ab_code)
    print(f"  Total clicks: {analytics['total_clicks']}")
    print(f"  Variant distribution: {analytics['by_variant']}")

    # 3. Analytics Dashboard
    print("\n--- Analytics Dashboard ---")
    dash_code = shortener.shorten("https://product.com")

    countries = ["US", "KR", "JP", "DE", "FR"]
    random.seed(42)
    for _ in range(1000):
        country = random.choices(countries, weights=[40, 25, 15, 10, 10])[0]
        shortener.resolve(dash_code, country=country)

    analytics = shortener.get_analytics(dash_code)
    print(f"  Total clicks: {analytics['total_clicks']}")
    print(f"  By country: {analytics['by_country']}")


# === Exercise 2: Pastebin Security ===
# Problem: Password-protected pastes, burn after read, client-side encryption.

class SecurePastebin:
    """Pastebin with security features."""

    def __init__(self):
        self.pastes = {}

    def create_paste(self, content, password=None, burn_after_read=False,
                     encrypted=False):
        """Create a paste with optional security features."""
        paste_id = hashlib.sha256(
            f"{content}{time.time()}{random.random()}".encode()
        ).hexdigest()[:12]

        paste = {
            "content": content,
            "password_hash": hashlib.sha256(password.encode()).hexdigest()
                if password else None,
            "burn_after_read": burn_after_read,
            "encrypted": encrypted,
            "created_at": time.time(),
            "view_count": 0,
            "deleted": False,
        }

        self.pastes[paste_id] = paste
        return paste_id

    def view_paste(self, paste_id, password=None):
        """View a paste with security checks."""
        paste = self.pastes.get(paste_id)
        if not paste or paste["deleted"]:
            return None, "Paste not found or deleted"

        # Password check
        if paste["password_hash"]:
            if not password:
                return None, "Password required"
            if hashlib.sha256(password.encode()).hexdigest() != paste["password_hash"]:
                return None, "Invalid password"

        content = paste["content"]
        paste["view_count"] += 1

        # Burn after read
        if paste["burn_after_read"]:
            paste["deleted"] = True
            paste["content"] = "[DELETED]"

        return content, "OK"

    @staticmethod
    def client_side_encrypt(content, key):
        """Simulate client-side encryption (XOR for simplicity)."""
        key_bytes = key.encode()
        encrypted = bytearray()
        for i, byte in enumerate(content.encode()):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        return encrypted.hex()

    @staticmethod
    def client_side_decrypt(encrypted_hex, key):
        """Simulate client-side decryption."""
        encrypted = bytes.fromhex(encrypted_hex)
        key_bytes = key.encode()
        decrypted = bytearray()
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        return decrypted.decode()


def exercise_2():
    """Pastebin security features."""
    print("Pastebin Security Features:")
    print("=" * 60)

    pastebin = SecurePastebin()

    # 1. Password-protected paste
    print("\n--- Password-Protected Paste ---")
    paste_id = pastebin.create_paste(
        "Secret configuration data",
        password="my_secret_123"
    )

    content, status = pastebin.view_paste(paste_id)
    print(f"  Without password: {status}")

    content, status = pastebin.view_paste(paste_id, password="wrong")
    print(f"  Wrong password: {status}")

    content, status = pastebin.view_paste(paste_id, password="my_secret_123")
    print(f"  Correct password: '{content}' ({status})")

    # 2. Burn after read
    print("\n--- Burn After Read ---")
    burn_id = pastebin.create_paste(
        "This message will self-destruct",
        burn_after_read=True
    )

    content, status = pastebin.view_paste(burn_id)
    print(f"  First read: '{content}' ({status})")

    content, status = pastebin.view_paste(burn_id)
    print(f"  Second read: {status}")

    # 3. Client-side encryption
    print("\n--- Client-Side Encryption ---")
    secret_key = "user_encryption_key_256"
    original = "Sensitive API keys: sk-12345"

    encrypted = SecurePastebin.client_side_encrypt(original, secret_key)
    print(f"  Original: '{original}'")
    print(f"  Encrypted: '{encrypted[:40]}...'")

    # Server stores encrypted content (server cannot read it)
    enc_paste_id = pastebin.create_paste(encrypted, encrypted=True)
    stored, status = pastebin.view_paste(enc_paste_id)

    # Client decrypts
    decrypted = SecurePastebin.client_side_decrypt(stored, secret_key)
    print(f"  Decrypted: '{decrypted}'")
    print(f"  Server never sees plaintext: {original == decrypted}")


# === Exercise 3: Dynamic Rate Limiting ===
# Problem: Different limits per tier, automatic peak adjustment, per-endpoint limits.

class DynamicRateLimiter:
    """Rate limiter with tier-based limits, peak detection, and per-endpoint granularity."""

    def __init__(self):
        self.tier_limits = {
            "free": {"default": 100, "peak_factor": 0.5},
            "paid": {"default": 1000, "peak_factor": 1.5},
            "enterprise": {"default": 10000, "peak_factor": 2.0},
        }
        self.endpoint_limits = {
            "/api/search": 0.5,       # 50% of tier limit
            "/api/upload": 0.2,       # 20% of tier limit
            "/api/read": 2.0,         # 200% of tier limit
        }
        self.user_counters = defaultdict(lambda: defaultdict(int))
        self.is_peak = False
        self.window_start = 0
        self.window_duration = 60  # 1-minute windows

    def set_peak_hours(self, is_peak):
        """Toggle peak hour mode."""
        self.is_peak = is_peak

    def get_limit(self, user_tier, endpoint="/api/default"):
        """Calculate the effective rate limit."""
        tier_config = self.tier_limits.get(user_tier, self.tier_limits["free"])
        base_limit = tier_config["default"]

        # Apply peak adjustment
        if self.is_peak:
            base_limit = int(base_limit * tier_config["peak_factor"])

        # Apply endpoint-specific multiplier
        endpoint_factor = self.endpoint_limits.get(endpoint, 1.0)
        return int(base_limit * endpoint_factor)

    def allow_request(self, user_id, user_tier, endpoint="/api/default"):
        """Check if request is allowed."""
        now = time.time()

        # Reset window
        window_key = int(now / self.window_duration)
        counter_key = f"{user_id}:{endpoint}:{window_key}"

        self.user_counters[user_id][counter_key] += 1
        current_count = self.user_counters[user_id][counter_key]

        limit = self.get_limit(user_tier, endpoint)
        return current_count <= limit, current_count, limit


def exercise_3():
    """Dynamic rate limiting design."""
    print("Dynamic Rate Limiting:")
    print("=" * 60)

    limiter = DynamicRateLimiter()

    # Show limits by tier and endpoint
    print("\n  Rate Limits (per minute):")
    print(f"  {'Tier':<12} {'Default':>10} {'Search':>10} {'Upload':>10} {'Read':>10}")
    print("  " + "-" * 55)
    for tier in ["free", "paid", "enterprise"]:
        default = limiter.get_limit(tier)
        search = limiter.get_limit(tier, "/api/search")
        upload = limiter.get_limit(tier, "/api/upload")
        read = limiter.get_limit(tier, "/api/read")
        print(f"  {tier:<12} {default:>10} {search:>10} {upload:>10} {read:>10}")

    # Peak hours
    print("\n  During Peak Hours:")
    limiter.set_peak_hours(True)
    print(f"  {'Tier':<12} {'Default':>10} {'Search':>10} {'Upload':>10} {'Read':>10}")
    print("  " + "-" * 55)
    for tier in ["free", "paid", "enterprise"]:
        default = limiter.get_limit(tier)
        search = limiter.get_limit(tier, "/api/search")
        upload = limiter.get_limit(tier, "/api/upload")
        read = limiter.get_limit(tier, "/api/read")
        print(f"  {tier:<12} {default:>10} {search:>10} {upload:>10} {read:>10}")
    limiter.set_peak_hours(False)

    # Simulate traffic
    print("\n  --- Simulation: Free tier user hitting search endpoint ---")
    for i in range(55):
        allowed, count, limit = limiter.allow_request(
            "user_free_1", "free", "/api/search"
        )
        if not allowed and count == limit + 1:
            print(f"  Request {count}: RATE LIMITED (limit={limit})")
            break
    else:
        print(f"  All 55 requests allowed")

    print(f"\n  --- Simulation: Enterprise tier user ---")
    for i in range(200):
        allowed, count, limit = limiter.allow_request(
            "user_ent_1", "enterprise", "/api/search"
        )
        if not allowed and count == limit + 1:
            print(f"  Request {count}: RATE LIMITED (limit={limit})")
            break
    else:
        print(f"  All 200 requests allowed (enterprise limit is high)")

    print("\n  Design Notes:")
    print("    - Redis INCR with TTL for distributed counter")
    print("    - Lua script for atomic check-and-increment")
    print("    - Response headers: X-RateLimit-Limit, X-RateLimit-Remaining")
    print("    - 429 Too Many Requests with Retry-After header")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: URL Shortener Extensions ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Pastebin Security ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Dynamic Rate Limiting ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
