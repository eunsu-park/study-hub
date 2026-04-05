"""
Exercises for Lesson 05: Web Application Hacking
Topic: Cybersecurity_Offensive

Practice problems covering SQL injection, XSS, CSRF, and IDOR vulnerabilities.
"""


# === Exercise 1: SQL Injection Detection ===
# Problem: Given a list of URL parameters, identify which ones are
# potentially vulnerable to SQL injection based on their behavior.
# Simulate by checking if certain test payloads change the response.

def exercise_1():
    """
    Implement a function that takes a parameter value and simulated responses:
    responses = {
        "1": "Welcome, Alice",
        "1'": "SQL error: syntax error",
        "1 OR 1=1": "Welcome, Alice\nWelcome, Bob\nWelcome, Charlie",
        "1 AND 1=2": "",
    }
    Analyze the responses and return a dict:
    {"error_based": bool, "boolean_blind": bool, "union_possible": bool}
    """
    # TODO: Detect SQLi vulnerability type from response patterns
    pass


# === Exercise 2: XSS Context Analyzer ===
# Problem: Given HTML source with a reflected input marker "INJECT_HERE",
# determine the injection context and suggest appropriate payloads.

def exercise_2():
    """
    html_samples = [
        '<div>Welcome INJECT_HERE</div>',
        '<input type="text" value="INJECT_HERE">',
        '<script>var name = "INJECT_HERE";</script>',
        '<!-- INJECT_HERE -->',
    ]
    For each, return: {"context": str, "payload": str, "encoding_needed": bool}
    Contexts: "html_body", "html_attribute", "javascript", "html_comment"
    """
    # TODO: Analyze context and suggest XSS payload
    pass


# === Exercise 3: CSRF Token Validator ===
# Problem: Implement a simple CSRF token validation check.
# Given a request dict with headers and body, determine if CSRF
# protection is properly implemented.

def exercise_3():
    """
    requests_to_check = [
        {"method": "POST", "headers": {"Cookie": "session=abc"},
         "body": {"action": "transfer", "amount": "1000"},
         "csrf_token_in_body": None, "csrf_token_in_header": None},
        {"method": "POST", "headers": {"Cookie": "session=abc", "X-CSRF-Token": "xyz"},
         "body": {"action": "transfer"}, "csrf_token_in_body": None,
         "csrf_token_in_header": "xyz"},
        {"method": "GET", "headers": {"Cookie": "session=abc"},
         "body": {}, "csrf_token_in_body": None, "csrf_token_in_header": None},
    ]
    Return list of {"vulnerable": bool, "reason": str} for each request.
    """
    # TODO: Check CSRF protection for each request
    pass


# === Exercise 4: IDOR Detection ===
# Problem: Given an API endpoint pattern and a list of response status codes
# for different ID values, determine if IDOR is present.

def exercise_4():
    """
    A user with ID=42 makes requests to /api/users/{id}/profile.
    responses = {
        42: {"status": 200, "body": "Own profile"},
        41: {"status": 200, "body": "Other user profile"},
        43: {"status": 200, "body": "Other user profile"},
        0: {"status": 404, "body": "Not found"},
        99999: {"status": 404, "body": "Not found"},
    }
    Analyze: Is this endpoint vulnerable to IDOR? Which IDs leak data?
    Return {"vulnerable": bool, "leaked_ids": list[int], "recommendation": str}
    """
    # TODO: Analyze for IDOR vulnerability
    pass


if __name__ == "__main__":
    print("=== Exercise 1: SQL Injection Detection ===")
    print(exercise_1())
    print("\n=== Exercise 2: XSS Context Analyzer ===")
    print(exercise_2())
    print("\n=== Exercise 3: CSRF Token Validator ===")
    print(exercise_3())
    print("\n=== Exercise 4: IDOR Detection ===")
    print(exercise_4())
