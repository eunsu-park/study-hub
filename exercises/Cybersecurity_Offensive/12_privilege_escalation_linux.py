"""
Exercises for Lesson 12: Privilege Escalation — Linux
Topic: Cybersecurity_Offensive

Practice problems covering SUID binaries, cron jobs, kernel exploits,
sudo misconfigurations, and path hijacking.
"""


# === Exercise 1: SUID Binary Analyzer ===
# Problem: Given a list of SUID binaries found on a system,
# identify which ones can be exploited for privilege escalation.
# Reference: GTFOBins

def exercise_1():
    """
    suid_binaries = [
        "/usr/bin/passwd",
        "/usr/bin/find",
        "/usr/bin/vim",
        "/usr/bin/ping",
        "/usr/bin/python3",
        "/usr/bin/env",
        "/usr/bin/nmap",
        "/usr/bin/su",
    ]
    For each binary, determine:
      - exploitable: bool (can it spawn a shell or read/write files?)
      - technique: str (how to exploit it, or "N/A")
      - gtfobins_category: str (shell/file_read/file_write/suid/N/A)

    Return list of analysis dicts.
    """
    # TODO: Analyze SUID binaries for privesc potential
    pass


# === Exercise 2: Cron Job Exploit Finder ===
# Problem: Analyze crontab entries and file permissions to find
# privilege escalation opportunities.

def exercise_2():
    """
    cron_entries = [
        {"schedule": "* * * * *", "user": "root",
         "command": "/opt/scripts/backup.sh",
         "file_perms": "-rwxrwxrwx", "file_owner": "root"},
        {"schedule": "0 * * * *", "user": "root",
         "command": "/usr/local/bin/cleanup",
         "file_perms": "-rwxr-xr-x", "file_owner": "root"},
        {"schedule": "*/5 * * * *", "user": "root",
         "command": "cd /tmp && tar czf /backup/tmp.tar.gz *",
         "file_perms": None, "file_owner": None},
        {"schedule": "0 0 * * *", "user": "www-data",
         "command": "/var/www/scripts/rotate_logs.sh",
         "file_perms": "-rwxr-xr--", "file_owner": "www-data"},
    ]
    Identify exploitable entries and the technique:
      - world-writable script
      - wildcard injection
      - writable directory in PATH
    Return list of {"entry": int, "exploitable": bool, "technique": str}
    """
    # TODO: Find exploitable cron jobs
    pass


# === Exercise 3: Sudo Misconfiguration Checker ===
# Problem: Parse sudo -l output and identify privilege escalation paths.

def exercise_3():
    """
    sudo_entries = [
        "(ALL) NOPASSWD: /usr/bin/vim",
        "(ALL) NOPASSWD: /usr/bin/less /var/log/*",
        "(root) /usr/bin/apt-get update",
        "(ALL) NOPASSWD: /usr/bin/env",
        "(ALL) NOPASSWD: /home/user/scripts/*.sh",
    ]
    For each entry, determine:
      - escalation_possible: bool
      - technique: str (description of how to escalate)
      - risk: str (high/medium/low)
    Return list of analysis dicts.
    """
    # TODO: Analyze sudo permissions for privesc
    pass


# === Exercise 4: Linux Enumeration Checklist ===
# Problem: Given system information, identify all potential
# privilege escalation vectors and rank them by likelihood.

def exercise_4():
    """
    system_info = {
        "kernel": "4.4.0-21-generic",
        "distro": "Ubuntu 16.04",
        "current_user": "www-data",
        "groups": ["www-data"],
        "writable_dirs": ["/tmp", "/var/www/html", "/dev/shm"],
        "internal_ports": [3306, 6379, 8080],
        "processes_as_root": ["mysql", "redis-server", "apache2"],
        "capabilities": {"/usr/bin/python3": "cap_setuid+ep"},
        "docker_socket": False,
    }
    Return prioritized list of escalation vectors:
    [{"vector": str, "confidence": str, "steps": list[str]}, ...]
    """
    # TODO: Enumerate and prioritize escalation vectors
    pass


if __name__ == "__main__":
    print("=== Exercise 1: SUID Binary Analyzer ===")
    print(exercise_1())
    print("\n=== Exercise 2: Cron Job Exploit Finder ===")
    print(exercise_2())
    print("\n=== Exercise 3: Sudo Misconfiguration Checker ===")
    print(exercise_3())
    print("\n=== Exercise 4: Linux Enumeration Checklist ===")
    print(exercise_4())
