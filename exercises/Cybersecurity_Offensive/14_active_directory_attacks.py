"""
Exercises for Lesson 14: Active Directory Attacks
Topic: Cybersecurity_Offensive

Practice problems covering AD enumeration, Kerberoasting,
AS-REP roasting, and lateral movement in AD environments.
"""


# === Exercise 1: AD Object Enumeration ===
# Problem: Given simulated LDAP query results, extract useful
# information for attack planning.

def exercise_1():
    """
    ldap_results = [
        {"dn": "CN=svc-sql,OU=Service Accounts,DC=corp,DC=local",
         "objectClass": "user", "servicePrincipalName": ["MSSQLSvc/db01:1433"],
         "userAccountControl": 66048, "memberOf": ["CN=Domain Users,DC=corp,DC=local"]},
        {"dn": "CN=admin01,OU=Admins,DC=corp,DC=local",
         "objectClass": "user", "servicePrincipalName": [],
         "userAccountControl": 66048,
         "memberOf": ["CN=Domain Admins,DC=corp,DC=local"]},
        {"dn": "CN=svc-backup,OU=Service Accounts,DC=corp,DC=local",
         "objectClass": "user", "servicePrincipalName": [],
         "userAccountControl": 4260352,  # DONT_REQ_PREAUTH flag set
         "memberOf": ["CN=Backup Operators,DC=corp,DC=local"]},
        {"dn": "CN=webuser,OU=Users,DC=corp,DC=local",
         "objectClass": "user", "servicePrincipalName": [],
         "userAccountControl": 512,
         "memberOf": ["CN=Domain Users,DC=corp,DC=local"]},
    ]
    Identify:
      - Kerberoastable accounts (have SPN)
      - AS-REP roastable accounts (DONT_REQ_PREAUTH = flag 0x400000)
      - High-privilege accounts
    Return {"kerberoastable": [...], "asrep_roastable": [...], "high_priv": [...]}
    """
    # TODO: Enumerate AD objects for attack opportunities
    pass


# === Exercise 2: Kerberos Ticket Analyzer ===
# Problem: Parse simulated Kerberos ticket data and identify
# attack opportunities.

def exercise_2():
    """
    tickets = [
        {"type": "TGT", "client": "webuser@CORP.LOCAL",
         "service": "krbtgt/CORP.LOCAL", "encryption": "aes256-cts",
         "expiry": "2025-06-15T20:00:00", "flags": ["forwardable", "renewable"]},
        {"type": "TGS", "client": "webuser@CORP.LOCAL",
         "service": "MSSQLSvc/db01:1433@CORP.LOCAL", "encryption": "rc4-hmac",
         "expiry": "2025-06-15T20:00:00", "flags": []},
        {"type": "TGS", "client": "svc-sql@CORP.LOCAL",
         "service": "cifs/fileserver@CORP.LOCAL", "encryption": "aes256-cts",
         "expiry": "2025-06-15T20:00:00", "flags": ["forwardable"]},
    ]
    Analyze each ticket:
      - Is RC4 encryption used? (weaker, easier to crack)
      - Is the ticket forwardable? (delegation attack potential)
      - What attack does each enable?
    Return list of analysis dicts.
    """
    # TODO: Analyze Kerberos tickets
    pass


# === Exercise 3: Attack Path Finder ===
# Problem: Given AD relationships, find the shortest attack path
# from a compromised user to Domain Admin.

def exercise_3():
    """
    relationships = [
        ("webuser", "CanRDP", "WEB01"),
        ("WEB01", "HasSession", "svc-sql"),
        ("svc-sql", "MemberOf", "DBA_Group"),
        ("DBA_Group", "AdminTo", "DB01"),
        ("DB01", "HasSession", "admin01"),
        ("admin01", "MemberOf", "Domain Admins"),
        ("webuser", "CanRDP", "DEV01"),
        ("DEV01", "HasSession", "devuser"),
        ("devuser", "CanRDP", "WEB02"),
    ]
    Find the shortest path from "webuser" to "Domain Admins".
    Return {"path": list[str], "hops": int, "techniques": list[str]}
    """
    # TODO: Find shortest attack path using BFS
    pass


# === Exercise 4: BloodHound Data Parser ===
# Problem: Simulate BloodHound-style analysis by processing
# group membership chains and identifying indirect privileges.

def exercise_4():
    """
    group_memberships = {
        "Domain Admins": {"members": ["admin01"], "member_of": ["Administrators"]},
        "Administrators": {"members": ["Domain Admins"], "member_of": []},
        "DBA_Group": {"members": ["svc-sql", "svc-backup"], "member_of": []},
        "IT_Support": {"members": ["helpdesk01", "helpdesk02"],
                       "member_of": ["Account Operators"]},
        "Account Operators": {"members": ["IT_Support"], "member_of": []},
    }
    Resolve transitive memberships:
      - Who has effective Domain Admin privileges?
      - Which groups grant Account Operator access?
    Return {"effective_admins": list, "account_operators": list}
    """
    # TODO: Resolve transitive group memberships
    pass


if __name__ == "__main__":
    print("=== Exercise 1: AD Object Enumeration ===")
    print(exercise_1())
    print("\n=== Exercise 2: Kerberos Ticket Analyzer ===")
    print(exercise_2())
    print("\n=== Exercise 3: Attack Path Finder ===")
    print(exercise_3())
    print("\n=== Exercise 4: BloodHound Data Parser ===")
    print(exercise_4())
