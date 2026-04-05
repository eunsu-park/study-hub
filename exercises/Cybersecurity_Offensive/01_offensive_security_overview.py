"""
Exercises for Lesson 01: Offensive Security Overview
Topic: Cybersecurity_Offensive

Practice problems covering engagement types, attack surface modeling,
and rules of engagement.
"""


# === Exercise 1: Classify Engagement Types ===
# Problem: Given the following scenarios, classify each as
# Vulnerability Assessment, Penetration Test, or Red Team engagement.
# Return a dict mapping scenario letter to engagement type string.

def exercise_1():
    """
    Scenarios:
      A. A company wants automated scanning of 500 hosts for known CVEs.
      B. A bank hires a team to simulate APT activity over 3 months
         including social engineering, with only the CISO aware.
      C. A startup asks for manual exploitation of their web app
         with a 2-week time-boxed scope.
    """
    # TODO: Return {"A": "...", "B": "...", "C": "..."}
    pass


# === Exercise 2: Risk Score Calculation ===
# Problem: Given an attack surface with base severity (1-4) and a list
# of mitigations, calculate a simplified risk score.
# Formula: score = min(10, severity * 2.5 * max(0.3, 1.0 - len(mitigations) * 0.15))

def exercise_2():
    """
    Calculate risk scores for:
      1. Severity=4 (CRITICAL), mitigations=["WAF"]
      2. Severity=3 (HIGH), mitigations=["MFA", "Rate limiting", "CAPTCHA"]
      3. Severity=2 (MEDIUM), mitigations=[]
    Return a list of three rounded scores.
    """
    # TODO: Implement risk score calculation
    pass


# === Exercise 3: Rules of Engagement Validator ===
# Problem: Write a function that validates a rules-of-engagement dict
# and returns a list of issues found.
# Required fields: client_name, tester_name, start_date, end_date,
#                  in_scope_targets, emergency_contact
# Rules: end_date > start_date, at least 1 in-scope target,
#        emergency_contact must not be empty.

def exercise_3():
    """
    Validate this RoE and return the list of issues:
    {
        "client_name": "Acme Corp",
        "tester_name": "",
        "start_date": "2025-06-15",
        "end_date": "2025-06-01",
        "in_scope_targets": [],
        "emergency_contact": ""
    }
    """
    # TODO: Implement validation logic, return list of issue strings
    pass


# === Exercise 4: MITRE ATT&CK Phase Mapping ===
# Problem: Map each pentest activity to the correct MITRE ATT&CK tactic.
# Activities: ["Port scanning", "Phishing email", "Pass-the-hash",
#              "Data exfiltration", "Kernel exploit for root"]
# Tactics: Reconnaissance, Initial Access, Lateral Movement,
#          Exfiltration, Privilege Escalation

def exercise_4():
    """
    Return a dict mapping each activity string to its tactic string.
    """
    # TODO: Create the mapping
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Classify Engagement Types ===")
    print(exercise_1())
    print("\n=== Exercise 2: Risk Score Calculation ===")
    print(exercise_2())
    print("\n=== Exercise 3: Rules of Engagement Validator ===")
    print(exercise_3())
    print("\n=== Exercise 4: MITRE ATT&CK Phase Mapping ===")
    print(exercise_4())
