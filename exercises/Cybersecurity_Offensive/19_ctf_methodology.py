"""
Exercises for Lesson 19: CTF Methodology
Topic: Cybersecurity_Offensive

Practice problems covering CTF challenge categories, common encoding
schemes, forensics basics, and crypto challenge patterns.
"""


# === Exercise 1: Multi-Layer Decoder ===
# Problem: Decode a flag that has been encoded with multiple layers.
# Common CTF encodings: Base64, hex, ROT13, URL encoding, binary.

def exercise_1():
    """
    encoded_flag = "NTI2MTczNGY3YjY0MzM2MzMwNjQzMzVm"

    Decoding steps (figure out the order):
      - One layer is Base64
      - One layer is hex-encoded ASCII
    Decode to reveal the flag in format: flag{...}

    Also decode these individual challenges:
      A. ROT13: "synt{ebg_guvegrra}"
      B. Binary: "01100110 01101100 01100001 01100111"
      C. URL: "flag%7Burl%5Fencode%7D"

    Return dict with all decoded strings.
    """
    # TODO: Implement multi-layer decoding
    pass


# === Exercise 2: Steganography Detector ===
# Problem: Implement basic steganography detection techniques.

def exercise_2():
    """
    Implement these stego checks:

    A. LSB detection: Given a list of pixel values (0-255), extract
       the least significant bit of each to form a hidden message.
       pixels = [104, 101, 108, 108, 111]  # ASCII values with LSB set
       Extract bits and check if they form readable ASCII.

    B. File magic detection: Given file header bytes, determine if
       a file has been appended after the end marker.
       file_data_hex = "FFD8FFE0...FFD9...504B0304"
       (JPEG ending at FFD9, ZIP appended after)

    C. Whitespace stego: Given text with trailing spaces/tabs,
       decode binary (space=0, tab=1).
       lines = ["Hello   \\t", "World\\t  ", "Test \\t\\t"]

    Return decoded messages for each technique.
    """
    # TODO: Implement steganography detection
    pass


# === Exercise 3: Crypto Challenge Identifier ===
# Problem: Given ciphertext samples, identify the encryption method
# and suggest the attack approach.

def exercise_3():
    """
    samples = [
        {"ciphertext": "Khoor Zruog", "hint": "shift cipher"},
        {"ciphertext": "Rijvs Tvnlf", "hint": "keyword cipher, key=SECRET"},
        {"ciphertext": "01001000 01101001",
         "hint": "XOR with single byte key"},
        {"ciphertext": "U2FsdGVkX1...", "hint": "OpenSSL enc format"},
    ]
    For each:
      - Identify the cipher type
      - Suggest attack approach (brute force, frequency analysis, etc.)
      - If simple enough, decrypt it
    Return list of {"cipher_type": str, "approach": str,
                    "decrypted": str|None}
    """
    # TODO: Identify and attack crypto challenges
    pass


# === Exercise 4: CTF Challenge Triage ===
# Problem: Given a list of CTF challenges with point values and
# categories, build an optimal solving strategy for a 6-hour CTF.

def exercise_4():
    """
    challenges = [
        {"name": "Baby Web", "category": "web", "points": 100,
         "estimated_time_min": 15, "difficulty": "easy"},
        {"name": "SQL Master", "category": "web", "points": 300,
         "estimated_time_min": 60, "difficulty": "medium"},
        {"name": "Baby Pwn", "category": "pwn", "points": 100,
         "estimated_time_min": 30, "difficulty": "easy"},
        {"name": "Heap Hero", "category": "pwn", "points": 500,
         "estimated_time_min": 120, "difficulty": "hard"},
        {"name": "RSA Baby", "category": "crypto", "points": 150,
         "estimated_time_min": 20, "difficulty": "easy"},
        {"name": "Hidden Flag", "category": "forensics", "points": 200,
         "estimated_time_min": 45, "difficulty": "medium"},
        {"name": "Rev Me", "category": "reversing", "points": 250,
         "estimated_time_min": 60, "difficulty": "medium"},
        {"name": "Misc Fun", "category": "misc", "points": 50,
         "estimated_time_min": 10, "difficulty": "easy"},
    ]
    total_time = 360  # minutes (6 hours)
    Optimize for maximum points within time budget.
    Return {"solve_order": list, "total_points": int,
            "time_used_min": int, "skipped": list}
    """
    # TODO: Optimize CTF solving strategy (knapsack-like problem)
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Multi-Layer Decoder ===")
    print(exercise_1())
    print("\n=== Exercise 2: Steganography Detector ===")
    print(exercise_2())
    print("\n=== Exercise 3: Crypto Challenge Identifier ===")
    print(exercise_3())
    print("\n=== Exercise 4: CTF Challenge Triage ===")
    print(exercise_4())
