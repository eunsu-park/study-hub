# Cybersecurity Offensive

## Overview

This topic covers offensive security techniques used in authorized penetration testing, red team engagements, and Capture-The-Flag (CTF) competitions. From reconnaissance and vulnerability assessment through binary exploitation and Active Directory attacks, these lessons provide a comprehensive foundation in ethical offensive security.

> **ETHICAL DISCLAIMER**
>
> All techniques, tools, and methodologies described in this topic are intended **exclusively** for:
> - Authorized penetration testing with written permission
> - Capture-The-Flag (CTF) competitions and cybersecurity training labs
> - Defensive security research and understanding attacker methodologies
> - Academic study and professional certification preparation
>
> **Unauthorized access to computer systems is illegal.** Always obtain explicit written authorization before testing any system you do not own. Violating computer fraud and abuse laws (CFAA in the US, Computer Misuse Act in the UK, and equivalent laws worldwide) carries severe criminal penalties including imprisonment.

## Prerequisites

- Strong Python programming skills (networking, file I/O, subprocess)
- Solid understanding of TCP/IP networking and HTTP protocols
- Linux command-line proficiency (shell scripting, file permissions)
- Basic understanding of web application architecture
- Familiarity with the Security topic (especially lessons 01-14)
- Comfortable reading technical documentation and RFCs

## Environment Setup

For safe practice, use only **authorized lab environments**:
- [Hack The Box](https://www.hackthebox.com/) — Online penetration testing labs
- [TryHackMe](https://tryhackme.com/) — Guided cybersecurity training
- [OWASP WebGoat](https://owasp.org/www-project-webgoat/) — Deliberately vulnerable web app
- [VulnHub](https://www.vulnhub.com/) — Downloadable vulnerable VMs
- [picoCTF](https://picoctf.org/) — Beginner-friendly CTF platform
- [OverTheWire](https://overthewire.org/) — Wargames for learning security concepts

**Key Tools** (install in isolated VM/container):
- Kali Linux or Parrot OS (security-focused distributions)
- Burp Suite Community Edition (web proxy)
- Nmap (network scanner)
- Metasploit Framework (exploitation framework)
- Ghidra (reverse engineering)
- Wireshark (packet analysis)
- pwntools (CTF exploitation library for Python)

## Lesson Plan

### Foundations and Methodology

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [01_Offensive_Security_Overview.md](./01_Offensive_Security_Overview.md) | ⭐⭐⭐ | Ethical hacking mindset, legal frameworks, PTES methodology, OWASP methodology, rules of engagement |
| [02_Reconnaissance.md](./02_Reconnaissance.md) | ⭐⭐⭐ | OSINT techniques, DNS enumeration, subdomain discovery, Google dorking, Shodan, theHarvester |
| [03_Network_Scanning.md](./03_Network_Scanning.md) | ⭐⭐⭐ | Nmap scanning, port scanning techniques, service detection, OS fingerprinting, firewall evasion |
| [04_Vulnerability_Assessment.md](./04_Vulnerability_Assessment.md) | ⭐⭐⭐ | CVE databases, CVSS scoring, vulnerability scanners, Nessus, OpenVAS, risk prioritization |

### Web and Application Attacks

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [05_Web_Application_Hacking.md](./05_Web_Application_Hacking.md) | ⭐⭐⭐ | OWASP Top 10 deep dive, SQL injection, XSS, CSRF, Burp Suite, automated scanning |
| [06_Authentication_Attacks.md](./06_Authentication_Attacks.md) | ⭐⭐⭐ | Password cracking, hashcat, John the Ripper, credential stuffing, session hijacking, MFA bypass |
| [07_Server_Side_Attacks.md](./07_Server_Side_Attacks.md) | ⭐⭐⭐ | SSRF, command injection, file inclusion (LFI/RFI), deserialization attacks, SSTI |
| [08_Client_Side_Attacks.md](./08_Client_Side_Attacks.md) | ⭐⭐⭐ | DOM-based XSS, clickjacking, browser exploitation, postMessage attacks, prototype pollution |

### Binary Exploitation

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [09_Binary_Fundamentals.md](./09_Binary_Fundamentals.md) | ⭐⭐⭐⭐ | x86/x64 assembly basics, calling conventions, stack layout, ELF format, memory layout |
| [10_Buffer_Overflow.md](./10_Buffer_Overflow.md) | ⭐⭐⭐⭐ | Stack overflow, ROP chains, NX bypass, ASLR, canaries, format string attacks |
| [11_Reverse_Engineering.md](./11_Reverse_Engineering.md) | ⭐⭐⭐⭐ | Static analysis with Ghidra, dynamic analysis with GDB, decompilation, anti-reversing techniques |

### System and Infrastructure

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [12_Privilege_Escalation_Linux.md](./12_Privilege_Escalation_Linux.md) | ⭐⭐⭐⭐ | SUID/SGID, Linux capabilities, kernel exploits, cron misconfigs, path hijacking |
| [13_Privilege_Escalation_Windows.md](./13_Privilege_Escalation_Windows.md) | ⭐⭐⭐⭐ | Token impersonation, service exploits, UAC bypass, DLL hijacking, unquoted paths |
| [14_Active_Directory.md](./14_Active_Directory.md) | ⭐⭐⭐⭐ | AD enumeration, Kerberoasting, Pass-the-Hash, BloodHound, LDAP injection, Golden Ticket |
| [15_Post_Exploitation.md](./15_Post_Exploitation.md) | ⭐⭐⭐⭐ | Persistence mechanisms, lateral movement, data exfiltration, C2 basics, pivoting |

### Specialized Domains

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [16_Wireless_Security.md](./16_Wireless_Security.md) | ⭐⭐⭐⭐ | WiFi attacks, WPA2/WPA3 cracking, evil twin, Bluetooth attacks, deauthentication |
| [17_Cloud_Security_Testing.md](./17_Cloud_Security_Testing.md) | ⭐⭐⭐⭐ | AWS/GCP misconfigurations, IAM exploitation, metadata attacks, S3 bucket enumeration |
| [18_Malware_Analysis.md](./18_Malware_Analysis.md) | ⭐⭐⭐⭐ | Static/dynamic analysis, sandboxing, YARA rules, PE structure, behavioral analysis |

### Operations and Competitions

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [19_CTF_Methodology.md](./19_CTF_Methodology.md) | ⭐⭐⭐ | CTF categories (pwn, web, crypto, forensics, misc), tools, writeup methodology, pwntools |
| [20_Red_Team_Operations.md](./20_Red_Team_Operations.md) | ⭐⭐⭐⭐ | Red team planning, threat emulation, reporting, purple teaming, MITRE ATT&CK, remediation |

## References

- OWASP Testing Guide v4.2: https://owasp.org/www-project-web-security-testing-guide/
- PTES (Penetration Testing Execution Standard): http://www.pentest-standard.org/
- MITRE ATT&CK Framework: https://attack.mitre.org/
- NIST SP 800-115 (Technical Guide to Information Security Testing): https://csrc.nist.gov/publications/detail/sp/800-115/final
- The Web Application Hacker's Handbook (Stuttard & Pinto)
- Hacking: The Art of Exploitation (Erickson)
- Red Team Field Manual (RTFM)
