"""
Exercises for Lesson 06: Authentication Attacks
Topic: Cybersecurity_Offensive

Practice problems covering password cracking, brute-force simulation,
credential stuffing, and hash analysis.
"""


# === Exercise 1: Password Hash Identifier ===
# Problem: Given hash strings, identify the hash algorithm used.
# Common patterns: MD5=32 hex, SHA1=40 hex, SHA256=64 hex,
# bcrypt starts with $2b$, NTLM=32 hex uppercase.

def exercise_1():
    """
    hashes = [
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "$2b$12$LJ3m4ys3Lg7E5S0Rq5Y8eOmZr7kGHs9kXaLTqVKB1HV2vGndm6a",
        "5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8",
        "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
        "A4F49C406510BDCAB6824EE7C30FD852",
    ]
    Return list of identified algorithm names.
    """
    # TODO: Identify each hash algorithm
    pass


# === Exercise 2: Password Strength Estimator ===
# Problem: Calculate password entropy and estimate crack time.
# Entropy = log2(charset_size ^ length)
# charset: lowercase=26, uppercase=26, digits=10, special=32

def exercise_2():
    """
    passwords = ["password", "P@ssw0rd!", "correcthorsebatterystaple",
                 "xK#9mQ$2pL"]
    For each, calculate: charset_size, entropy_bits, and estimated
    crack time at 10 billion guesses/sec.
    Return list of dicts with keys: password, entropy, crack_seconds.
    """
    # TODO: Calculate entropy and crack time for each password
    pass


# === Exercise 3: Credential Stuffing Simulator ===
# Problem: Given a breached credential list and a target login function,
# simulate credential stuffing. Track successes, failures, and lockouts.

def exercise_3():
    """
    breached_creds = [
        ("alice@test.com", "password123"),
        ("bob@test.com", "qwerty"),
        ("charlie@test.com", "letmein"),
        ("diana@test.com", "12345678"),
    ]
    target_valid_creds = {("alice@test.com", "password123"),
                          ("diana@test.com", "12345678")}
    lockout_after = 3  # Lock account after 3 failures

    Simulate the attack and return:
    {"successful": [...], "failed": [...], "locked_out": [...]}
    """
    # TODO: Simulate credential stuffing attack
    pass


# === Exercise 4: Hash Cracking with Rainbow Table ===
# Problem: Implement a simple rainbow table lookup.
# Pre-compute MD5 hashes for a wordlist, then crack given hashes.

def exercise_4():
    """
    wordlist = ["password", "123456", "admin", "letmein", "welcome",
                "monkey", "dragon", "master", "qwerty", "login"]
    target_hashes = [
        "5f4dcc3b5aa765d61d8327deb882cf99",  # password
        "e10adc3949ba59abbe56e057f20f883e",  # 123456
        "d8578edf8458ce06fbc5bb76a58c5ca4",  # qwerty (this is wrong, test it)
    ]
    Build rainbow table from wordlist and attempt to crack each hash.
    Return list of {"hash": str, "cracked": bool, "plaintext": str|None}
    """
    # TODO: Build rainbow table and crack hashes
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Password Hash Identifier ===")
    print(exercise_1())
    print("\n=== Exercise 2: Password Strength Estimator ===")
    print(exercise_2())
    print("\n=== Exercise 3: Credential Stuffing Simulator ===")
    print(exercise_3())
    print("\n=== Exercise 4: Hash Cracking with Rainbow Table ===")
    print(exercise_4())
