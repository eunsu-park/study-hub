"""
Exercises for Lesson 13: Privilege Escalation — Windows
Topic: Cybersecurity_Offensive

Practice problems covering service misconfigurations, token manipulation,
UAC bypass, and Windows enumeration.
"""


# === Exercise 1: Unquoted Service Path Finder ===
# Problem: Identify Windows services with unquoted paths that contain
# spaces, making them vulnerable to path interception.

def exercise_1():
    """
    services = [
        {"name": "VulnService1",
         "path": 'C:\\Program Files\\Vulnerable App\\service.exe',
         "start_mode": "Auto", "run_as": "LocalSystem"},
        {"name": "SafeService",
         "path": '"C:\\Program Files\\Safe App\\service.exe"',
         "start_mode": "Auto", "run_as": "LocalSystem"},
        {"name": "VulnService2",
         "path": 'C:\\Program Files (x86)\\Another App\\bin\\svc.exe',
         "start_mode": "Manual", "run_as": "LocalSystem"},
        {"name": "NoSpaceService",
         "path": 'C:\\tools\\myservice.exe',
         "start_mode": "Auto", "run_as": "NetworkService"},
    ]
    For each unquoted-path service, list the hijack paths to try:
      e.g., C:\\Program.exe, C:\\Program Files\\Vulnerable.exe
    Return list of {"service": str, "vulnerable": bool, "hijack_paths": list[str]}
    """
    # TODO: Find unquoted service paths and generate hijack paths
    pass


# === Exercise 2: Access Token Analysis ===
# Problem: Given a process token's privileges, identify which
# privileges enable escalation and how.

def exercise_2():
    """
    token_privileges = [
        {"name": "SeImpersonatePrivilege", "enabled": True},
        {"name": "SeDebugPrivilege", "enabled": False},
        {"name": "SeBackupPrivilege", "enabled": True},
        {"name": "SeRestorePrivilege", "enabled": True},
        {"name": "SeAssignPrimaryTokenPrivilege", "enabled": True},
        {"name": "SeChangeNotifyPrivilege", "enabled": True},
        {"name": "SeShutdownPrivilege", "enabled": True},
    ]
    For each dangerous privilege, explain:
      - escalation_technique: str
      - tools: list[str] (e.g., JuicyPotato, PrintSpoofer)
      - risk: str
    Return list of analysis dicts (only dangerous privileges).
    """
    # TODO: Analyze token privileges for escalation potential
    pass


# === Exercise 3: Registry ACL Checker ===
# Problem: Check if service registry keys have weak permissions
# that allow a low-privilege user to modify the service binary path.

def exercise_3():
    """
    registry_acls = [
        {"key": "HKLM\\SYSTEM\\CurrentControlSet\\Services\\VulnSvc",
         "acl": [
             {"principal": "NT AUTHORITY\\SYSTEM", "access": "FullControl"},
             {"principal": "BUILTIN\\Users", "access": "FullControl"},
             {"principal": "BUILTIN\\Administrators", "access": "FullControl"},
         ]},
        {"key": "HKLM\\SYSTEM\\CurrentControlSet\\Services\\SafeSvc",
         "acl": [
             {"principal": "NT AUTHORITY\\SYSTEM", "access": "FullControl"},
             {"principal": "BUILTIN\\Users", "access": "ReadKey"},
             {"principal": "BUILTIN\\Administrators", "access": "FullControl"},
         ]},
    ]
    Identify keys where non-admin users have write access.
    Return list of {"key": str, "vulnerable": bool, "weak_principal": str}
    """
    # TODO: Check registry permissions
    pass


# === Exercise 4: Windows Privesc Checklist Generator ===
# Problem: Given system enumeration data, generate a prioritized
# checklist of escalation techniques to try.

def exercise_4():
    """
    system_info = {
        "os_version": "Windows 10 Build 17763",
        "current_user": "CORP\\webuser",
        "groups": ["Users", "IIS_IUSRS"],
        "privileges": ["SeImpersonatePrivilege"],
        "installed_software": ["FileZilla Server 0.9.60", "XAMPP 7.4.3"],
        "scheduled_tasks_writable": True,
        "always_install_elevated": False,
        "unquoted_services": 2,
        "modifiable_services": 1,
    }
    Return ordered list of techniques to try:
    [{"technique": str, "priority": int, "reason": str, "tool": str}, ...]
    """
    # TODO: Generate prioritized privesc checklist
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Unquoted Service Path Finder ===")
    print(exercise_1())
    print("\n=== Exercise 2: Access Token Analysis ===")
    print(exercise_2())
    print("\n=== Exercise 3: Registry ACL Checker ===")
    print(exercise_3())
    print("\n=== Exercise 4: Windows Privesc Checklist ===")
    print(exercise_4())
