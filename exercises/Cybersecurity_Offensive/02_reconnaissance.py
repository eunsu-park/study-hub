"""
Exercises for Lesson 02: Reconnaissance
Topic: Cybersecurity_Offensive

Practice problems covering passive and active recon techniques,
OSINT, and DNS enumeration.
"""


# === Exercise 1: Classify Recon Techniques ===
# Problem: Classify each technique as "passive" or "active".
# Techniques:
#   1. Checking WHOIS records
#   2. Running nmap SYN scan against target
#   3. Searching LinkedIn for employee names
#   4. Sending DNS zone transfer request

def exercise_1():
    """
    Return a dict mapping technique description to "passive" or "active".
    Passive = no direct interaction with target infrastructure.
    Active = direct interaction that the target could detect.
    """
    # TODO: Classify each technique
    pass


# === Exercise 2: Subdomain Enumeration Simulator ===
# Problem: Given a domain and a wordlist, simulate subdomain brute-forcing.
# For each word in the wordlist, construct "{word}.{domain}" and check
# if it exists in the known_subdomains set. Return discovered subdomains.

def exercise_2():
    """
    domain = "example.com"
    wordlist = ["www", "mail", "ftp", "admin", "dev", "staging", "api", "vpn"]
    known_subdomains = {"www.example.com", "mail.example.com", "api.example.com",
                        "dev.example.com"}
    Return the sorted list of discovered subdomains.
    """
    # TODO: Implement subdomain brute-force simulation
    pass


# === Exercise 3: Google Dork Builder ===
# Problem: Build Google dork queries for the following objectives.
# Each dork should be a valid Google search string.
#   A. Find PDF files on target.com
#   B. Find login pages on target.com
#   C. Find directory listings on target.com

def exercise_3():
    """
    Return a dict {"A": "...", "B": "...", "C": "..."} with valid dork strings.
    Example format: 'site:target.com filetype:pdf'
    """
    # TODO: Build Google dork queries
    pass


# === Exercise 4: DNS Record Parser ===
# Problem: Parse simulated DNS records and extract useful recon info.
# Given a list of DNS record dicts, return:
#   - All IP addresses from A records
#   - All mail servers from MX records (sorted by priority)
#   - The SPF record text (from TXT records containing "v=spf1")

def exercise_4():
    """
    dns_records = [
        {"type": "A", "name": "example.com", "value": "93.184.216.34"},
        {"type": "A", "name": "www.example.com", "value": "93.184.216.34"},
        {"type": "MX", "name": "example.com", "priority": 10, "value": "mail1.example.com"},
        {"type": "MX", "name": "example.com", "priority": 20, "value": "mail2.example.com"},
        {"type": "TXT", "name": "example.com", "value": "v=spf1 include:_spf.google.com ~all"},
        {"type": "TXT", "name": "example.com", "value": "google-site-verification=abc123"},
        {"type": "NS", "name": "example.com", "value": "ns1.example.com"},
    ]
    Return {"ips": [...], "mail_servers": [...], "spf": "..."}
    """
    # TODO: Parse DNS records and extract information
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Classify Recon Techniques ===")
    print(exercise_1())
    print("\n=== Exercise 2: Subdomain Enumeration Simulator ===")
    print(exercise_2())
    print("\n=== Exercise 3: Google Dork Builder ===")
    print(exercise_3())
    print("\n=== Exercise 4: DNS Record Parser ===")
    print(exercise_4())
