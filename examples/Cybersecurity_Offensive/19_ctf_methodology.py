"""
Example: CTF Methodology
==========================
Multi-layer decoder, cipher identifier, steganography basics,
and CTF challenge optimization (knapsack solver).

IMPORTANT: For authorized security testing and CTF only.
"""

import base64
import hashlib
import string
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Encoding / Decoding Utilities
# ---------------------------------------------------------------------------

def decode_base64(data: str) -> bytes:
    """Decode base64 with padding fix."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.b64decode(data)


def decode_hex(data: str) -> bytes:
    """Decode hex string to bytes."""
    return bytes.fromhex(data.replace(" ", ""))


def decode_rot13(text: str) -> str:
    """Apply ROT13 cipher."""
    result = []
    for c in text:
        if "a" <= c <= "z":
            result.append(chr((ord(c) - ord("a") + 13) % 26 + ord("a")))
        elif "A" <= c <= "Z":
            result.append(chr((ord(c) - ord("A") + 13) % 26 + ord("A")))
        else:
            result.append(c)
    return "".join(result)


def decode_binary(binary_str: str) -> str:
    """Decode space-separated binary to ASCII."""
    return "".join(chr(int(b, 2)) for b in binary_str.split())


def xor_single_byte(data: bytes, key: int) -> bytes:
    """XOR data with a single byte key."""
    return bytes(b ^ key for b in data)


def multi_layer_decode(encoded: str, layers: list[str]) -> str:
    """Apply decoding layers in sequence."""
    result = encoded
    for layer in layers:
        if layer == "base64":
            result = decode_base64(result).decode() if isinstance(result, str) \
                else decode_base64(result.decode()).decode()
        elif layer == "hex":
            result = decode_hex(result).decode() if isinstance(result, str) \
                else result.decode()
        elif layer == "rot13":
            result = decode_rot13(result if isinstance(result, str) else result.decode())
    return result


# ---------------------------------------------------------------------------
# Caesar / Shift Cipher Brute Force
# ---------------------------------------------------------------------------

def caesar_brute(ciphertext: str) -> list[tuple[int, str]]:
    """Try all 26 shifts for Caesar cipher."""
    results = []
    for shift in range(26):
        plain = []
        for c in ciphertext:
            if "a" <= c <= "z":
                plain.append(chr((ord(c) - ord("a") - shift) % 26 + ord("a")))
            elif "A" <= c <= "Z":
                plain.append(chr((ord(c) - ord("A") - shift) % 26 + ord("A")))
            else:
                plain.append(c)
        results.append((shift, "".join(plain)))
    return results


# ---------------------------------------------------------------------------
# Frequency Analysis
# ---------------------------------------------------------------------------

ENGLISH_FREQ = {
    "e": 12.7, "t": 9.1, "a": 8.2, "o": 7.5, "i": 7.0,
    "n": 6.7, "s": 6.3, "h": 6.1, "r": 6.0, "d": 4.3,
}


def frequency_analysis(text: str) -> dict[str, float]:
    """Calculate letter frequency distribution."""
    letters = [c.lower() for c in text if c.isalpha()]
    total = len(letters) or 1
    freq = {}
    for c in string.ascii_lowercase:
        count = letters.count(c)
        freq[c] = round(count / total * 100, 1)
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def chi_squared_score(observed: dict[str, float]) -> float:
    """Chi-squared test against English letter frequency."""
    score = 0.0
    for letter, expected in ENGLISH_FREQ.items():
        obs = observed.get(letter, 0.0)
        score += ((obs - expected) ** 2) / expected
    return round(score, 2)


# ---------------------------------------------------------------------------
# CTF Challenge Optimizer (Knapsack)
# ---------------------------------------------------------------------------

@dataclass
class CTFChallenge:
    name: str
    category: str
    points: int
    time_minutes: int
    difficulty: str


def optimize_challenges(challenges: list[CTFChallenge],
                        total_time: int) -> dict:
    """Select challenges to maximize points within time budget."""
    n = len(challenges)
    # 0-1 Knapsack DP
    dp = [[0] * (total_time + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        ch = challenges[i - 1]
        for t in range(total_time + 1):
            dp[i][t] = dp[i - 1][t]
            if ch.time_minutes <= t:
                dp[i][t] = max(dp[i][t],
                               dp[i - 1][t - ch.time_minutes] + ch.points)

    # Backtrack to find selected challenges
    selected = []
    t = total_time
    for i in range(n, 0, -1):
        if dp[i][t] != dp[i - 1][t]:
            selected.append(challenges[i - 1])
            t -= challenges[i - 1].time_minutes

    selected.reverse()
    return {
        "solve_order": [c.name for c in selected],
        "total_points": dp[n][total_time],
        "time_used": total_time - t,
        "skipped": [c.name for c in challenges if c not in selected],
    }


# ---------------------------------------------------------------------------
# File Magic Bytes
# ---------------------------------------------------------------------------

FILE_SIGNATURES = {
    b"\x89PNG": "PNG image",
    b"\xFF\xD8\xFF": "JPEG image",
    b"PK\x03\x04": "ZIP archive",
    b"\x7fELF": "ELF binary",
    b"MZ": "PE executable",
    b"%PDF": "PDF document",
    b"\x1f\x8b": "GZIP compressed",
    b"Rar!": "RAR archive",
}


def identify_file(header: bytes) -> str:
    """Identify file type from magic bytes."""
    for magic, name in FILE_SIGNATURES.items():
        if header.startswith(magic):
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("CTF Methodology Examples")
    print("=" * 50)

    # Decoding
    print("\nDecoding Examples:")
    print(f"  ROT13('synt{{ebg13}}') = {decode_rot13('synt{ebg13}')}")
    print(f"  Binary('01001000 01101001') = {decode_binary('01001000 01101001')}")
    print(f"  Base64('ZmxhZ3tiYXNlNjR9') = {decode_base64('ZmxhZ3tiYXNlNjR9').decode()}")

    # Caesar brute force
    print("\nCaesar Brute Force ('Khoor Zruog'):")
    for shift, plain in caesar_brute("Khoor Zruog"):
        if shift <= 5:
            marker = " <--" if shift == 3 else ""
            print(f"  shift={shift:2d}: {plain}{marker}")

    # Frequency analysis
    print("\nFrequency Analysis:")
    text = "the quick brown fox jumps over the lazy dog"
    freq = frequency_analysis(text)
    top5 = list(freq.items())[:5]
    print(f"  Top 5: {', '.join(f'{k}={v}%' for k, v in top5)}")
    print(f"  Chi-squared vs English: {chi_squared_score(freq)}")

    # Challenge optimizer
    print("\nCTF Challenge Optimization (6h budget):")
    challenges = [
        CTFChallenge("Baby Web", "web", 100, 15, "easy"),
        CTFChallenge("SQL Master", "web", 300, 60, "medium"),
        CTFChallenge("Baby Pwn", "pwn", 100, 30, "easy"),
        CTFChallenge("Heap Hero", "pwn", 500, 120, "hard"),
        CTFChallenge("RSA Baby", "crypto", 150, 20, "easy"),
        CTFChallenge("Hidden Flag", "forensics", 200, 45, "medium"),
        CTFChallenge("Rev Me", "reversing", 250, 60, "medium"),
        CTFChallenge("Misc Fun", "misc", 50, 10, "easy"),
    ]
    result = optimize_challenges(challenges, 360)
    print(f"  Solve: {result['solve_order']}")
    print(f"  Points: {result['total_points']}, Time: {result['time_used']}min")
    print(f"  Skip: {result['skipped']}")

    # File identification
    print("\nFile Magic Identification:")
    test_headers = [b"\x89PNG\r\n", b"PK\x03\x04", b"\x7fELF", b"MZ\x90"]
    for h in test_headers:
        print(f"  {h[:4]!r} -> {identify_file(h)}")


if __name__ == "__main__":
    demo()
