"""
Exercises for Lesson 08: Client-Side Attacks
Topic: Cybersecurity_Offensive

Practice problems covering DOM-based XSS, clickjacking,
open redirect, and WebSocket hijacking.
"""


# === Exercise 1: DOM XSS Sink Analysis ===
# Problem: Identify dangerous JavaScript sinks in code snippets.
# Classify each as "sink" (dangerous) or "safe".

def exercise_1():
    """
    code_snippets = [
        'document.getElementById("out").textContent = userInput;',
        'document.getElementById("out").innerHTML = userInput;',
        'eval(userInput);',
        'window.location = userInput;',
        'document.getElementById("out").setAttribute("data-value", userInput);',
        'document.write(userInput);',
        'element.style.color = userInput;',
    ]
    For each, return {"code": str, "dangerous": bool, "sink_type": str}
    sink_type: "html_injection", "code_execution", "redirect", "safe", "style_injection"
    """
    # TODO: Classify JavaScript sinks
    pass


# === Exercise 2: Clickjacking Defense Check ===
# Problem: Given HTTP response headers, determine if a page is
# vulnerable to clickjacking (iframe embedding).

def exercise_2():
    """
    responses = [
        {"url": "https://bank.com/transfer", "headers": {}},
        {"url": "https://shop.com/checkout",
         "headers": {"X-Frame-Options": "DENY"}},
        {"url": "https://social.com/settings",
         "headers": {"X-Frame-Options": "SAMEORIGIN"}},
        {"url": "https://app.com/dashboard",
         "headers": {"Content-Security-Policy": "frame-ancestors 'none'"}},
        {"url": "https://legacy.com/admin",
         "headers": {"X-Frame-Options": "ALLOW-FROM https://trusted.com"}},
    ]
    Return list of {"url": str, "vulnerable": bool, "protection": str}
    """
    # TODO: Check clickjacking protections
    pass


# === Exercise 3: Open Redirect Detector ===
# Problem: Given URLs with redirect parameters, determine which are
# vulnerable to open redirect attacks.

def exercise_3():
    """
    urls = [
        "https://app.com/login?next=/dashboard",
        "https://app.com/login?next=https://evil.com",
        "https://app.com/login?next=//evil.com",
        "https://app.com/login?next=/\\evil.com",
        "https://app.com/login?next=https://app.com.evil.com",
    ]
    For each, return {"url": str, "vulnerable": bool, "technique": str}
    Techniques: "safe_relative", "absolute_url", "protocol_relative",
                "backslash_bypass", "subdomain_confusion"
    """
    # TODO: Detect open redirect vulnerabilities
    pass


# === Exercise 4: CSP Bypass Analysis ===
# Problem: Analyze Content-Security-Policy headers for weaknesses.

def exercise_4():
    """
    csp_policies = [
        "default-src 'self'; script-src 'unsafe-inline'",
        "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net",
        "default-src 'self'; script-src 'nonce-abc123'",
        "default-src *; script-src *",
    ]
    For each CSP, return:
    {"policy": str, "secure": bool, "weaknesses": list[str], "bypass_hint": str}
    """
    # TODO: Analyze CSP policies for bypass opportunities
    pass


if __name__ == "__main__":
    print("=== Exercise 1: DOM XSS Sink Analysis ===")
    print(exercise_1())
    print("\n=== Exercise 2: Clickjacking Defense Check ===")
    print(exercise_2())
    print("\n=== Exercise 3: Open Redirect Detector ===")
    print(exercise_3())
    print("\n=== Exercise 4: CSP Bypass Analysis ===")
    print(exercise_4())
