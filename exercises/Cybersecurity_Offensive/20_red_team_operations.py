"""
Exercises for Lesson 20: Red Team Operations
Topic: Cybersecurity_Offensive

Practice problems covering red team planning, C2 infrastructure,
OPSEC, threat emulation, and reporting.
"""


# === Exercise 1: C2 Infrastructure Planner ===
# Problem: Design a resilient command-and-control infrastructure
# with redirectors, fallback channels, and domain fronting.

def exercise_1():
    """
    Design C2 infrastructure for a 30-day engagement:
    Requirements:
      - Primary and fallback C2 channels
      - At least 2 redirector layers
      - Domain categorization (avoid "uncategorized" domains)
      - SSL/TLS for all communications

    infrastructure_components = {
        "c2_server": {"ip": "?", "hosting": "?"},
        "redirectors": [{"type": "?", "domain": "?", "purpose": "?"}],
        "fallback_channels": ["?"],
        "domain_fronting": {"cdn": "?", "fronted_domain": "?"},
    }
    Return a complete infrastructure plan with justifications.
    """
    # TODO: Design C2 infrastructure
    pass


# === Exercise 2: OPSEC Checklist Evaluator ===
# Problem: Evaluate operational security for a red team engagement.
# Identify OPSEC failures in a given scenario.

def exercise_2():
    """
    scenario = {
        "actions": [
            "Used personal laptop for C2 management",
            "Registered C2 domain with team member's real name",
            "Ran nmap scan at 3 AM from office IP",
            "Used Cobalt Strike with default malleable C2 profile",
            "Exfiltrated data over DNS to a newly registered domain",
            "Stored loot on encrypted external drive",
            "Communicated findings via encrypted Signal messages",
            "Reused C2 infrastructure from previous engagement",
        ],
    }
    For each action, evaluate:
      - opsec_rating: good/bad/critical
      - issue: str (what's wrong, or "N/A" if good)
      - remediation: str
    Return list of evaluation dicts.
    """
    # TODO: Evaluate OPSEC for each action
    pass


# === Exercise 3: Threat Emulation Plan ===
# Problem: Given a threat actor profile (MITRE ATT&CK), create
# a red team emulation plan mapping techniques to tools.

def exercise_3():
    """
    threat_actor = "APT29 (Cozy Bear)"
    known_techniques = [
        {"tactic": "Initial Access", "technique": "T1566.001",
         "name": "Spearphishing Attachment"},
        {"tactic": "Execution", "technique": "T1059.001",
         "name": "PowerShell"},
        {"tactic": "Persistence", "technique": "T1053.005",
         "name": "Scheduled Task"},
        {"tactic": "Defense Evasion", "technique": "T1027",
         "name": "Obfuscated Files"},
        {"tactic": "Credential Access", "technique": "T1003.001",
         "name": "LSASS Memory"},
        {"tactic": "Lateral Movement", "technique": "T1021.006",
         "name": "Windows Remote Management"},
    ]
    For each technique, provide:
      - emulation_tool: str (open source tool to emulate)
      - procedure: str (step-by-step)
      - detection_opportunity: str (what blue team should look for)
    Return list of emulation plan entries.
    """
    # TODO: Create threat emulation plan
    pass


# === Exercise 4: Red Team Report Findings Generator ===
# Problem: Given raw exploitation data, generate structured
# findings for a red team report.

def exercise_4():
    """
    raw_findings = [
        {"action": "Phished 3 users, obtained credentials for finance team",
         "impact": "Initial foothold in corporate network",
         "evidence": ["screenshot_phish.png", "cred_capture.log"],
         "timestamp": "Day 3"},
        {"action": "Escalated to Domain Admin via Kerberoasting",
         "impact": "Full Active Directory compromise",
         "evidence": ["krb_ticket.kirbi", "da_session.png"],
         "timestamp": "Day 7"},
        {"action": "Accessed PII database with 50k customer records",
         "impact": "Data breach of sensitive customer information",
         "evidence": ["db_access.log", "sample_data.csv"],
         "timestamp": "Day 12"},
    ]
    For each finding, generate:
      - severity: critical/high/medium/low
      - mitre_mapping: list of ATT&CK technique IDs
      - business_risk: str
      - recommendation: str
      - executive_summary: str (non-technical, 1-2 sentences)
    Return list of structured finding dicts.
    """
    # TODO: Generate structured report findings
    pass


if __name__ == "__main__":
    print("=== Exercise 1: C2 Infrastructure Planner ===")
    print(exercise_1())
    print("\n=== Exercise 2: OPSEC Checklist Evaluator ===")
    print(exercise_2())
    print("\n=== Exercise 3: Threat Emulation Plan ===")
    print(exercise_3())
    print("\n=== Exercise 4: Red Team Report Findings ===")
    print(exercise_4())
