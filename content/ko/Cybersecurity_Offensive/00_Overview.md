# 사이버보안 공격 기법

## 개요

이 토픽은 공인된 침투 테스트, 레드팀 활동 및 Capture-The-Flag(CTF) 대회에서 사용되는 공격 보안 기법을 다룹니다. 정찰 및 취약점 평가부터 바이너리 익스플로잇 및 Active Directory 공격까지, 이 레슨들은 윤리적 공격 보안에 대한 포괄적인 기초를 제공합니다.

> **윤리적 면책 조항**
>
> 이 토픽에서 설명하는 모든 기법, 도구 및 방법론은 **오직** 다음 목적으로만 사용됩니다:
> - 서면 허가를 받은 공인된 침투 테스트
> - Capture-The-Flag(CTF) 대회 및 사이버보안 교육 랩
> - 방어 보안 연구 및 공격자 방법론 이해
> - 학술 연구 및 전문 자격증 준비
>
> **컴퓨터 시스템에 대한 무단 접근은 불법입니다.** 소유하지 않은 시스템을 테스트하기 전에 항상 명시적인 서면 허가를 받으세요. 컴퓨터 사기 및 남용법(미국 CFAA, 영국 Computer Misuse Act 및 전 세계 동등한 법률)을 위반하면 징역을 포함한 심각한 형사 처벌을 받을 수 있습니다.

## 선행 조건

- 강력한 Python 프로그래밍 스킬 (네트워킹, 파일 I/O, subprocess)
- TCP/IP 네트워킹 및 HTTP 프로토콜에 대한 확실한 이해
- Linux 커맨드라인 능숙도 (셸 스크립팅, 파일 권한)
- 웹 애플리케이션 아키텍처에 대한 기본 이해
- Security 토픽 (특히 01-14 레슨) 숙지
- 기술 문서 및 RFC 읽기에 익숙함

## 환경 설정

안전한 연습을 위해 **공인된 랩 환경**만 사용하세요:
- [Hack The Box](https://www.hackthebox.com/) — 온라인 침투 테스트 랩
- [TryHackMe](https://tryhackme.com/) — 안내된 사이버보안 교육
- [OWASP WebGoat](https://owasp.org/www-project-webgoat/) — 의도적으로 취약한 웹 앱
- [VulnHub](https://www.vulnhub.com/) — 다운로드 가능한 취약 VM
- [picoCTF](https://picoctf.org/) — 초보자 친화적 CTF 플랫폼
- [OverTheWire](https://overthewire.org/) — 보안 개념 학습을 위한 워게임

**주요 도구** (격리된 VM/컨테이너에 설치):
- Kali Linux 또는 Parrot OS (보안 특화 배포판)
- Burp Suite Community Edition (웹 프록시)
- Nmap (네트워크 스캐너)
- Metasploit Framework (익스플로잇 프레임워크)
- Ghidra (리버스 엔지니어링)
- Wireshark (패킷 분석)
- pwntools (Python용 CTF 익스플로잇 라이브러리)

## 레슨 계획

### 기초 및 방법론

| 파일명 | 난이도 | 주요 주제 |
|--------|--------|----------|
| [01_Offensive_Security_Overview.md](./01_Offensive_Security_Overview.md) | ⭐⭐⭐ | 윤리적 해킹 마인드셋, 법적 프레임워크, PTES 방법론, OWASP 방법론, 교전 규칙 |
| [02_Reconnaissance.md](./02_Reconnaissance.md) | ⭐⭐⭐ | OSINT 기법, DNS 열거, 서브도메인 발견, 구글 도킹, Shodan, theHarvester |
| [03_Network_Scanning.md](./03_Network_Scanning.md) | ⭐⭐⭐ | Nmap 스캐닝, 포트 스캐닝 기법, 서비스 탐지, OS 핑거프린팅, 방화벽 우회 |
| [04_Vulnerability_Assessment.md](./04_Vulnerability_Assessment.md) | ⭐⭐⭐ | CVE 데이터베이스, CVSS 점수, 취약점 스캐너, Nessus, OpenVAS, 위험 우선순위 |

### 웹 및 애플리케이션 공격

| 파일명 | 난이도 | 주요 주제 |
|--------|--------|----------|
| [05_Web_Application_Hacking.md](./05_Web_Application_Hacking.md) | ⭐⭐⭐ | OWASP Top 10 심층 분석, SQL 인젝션, XSS, CSRF, Burp Suite, 자동화 스캐닝 |
| [06_Authentication_Attacks.md](./06_Authentication_Attacks.md) | ⭐⭐⭐ | 비밀번호 크래킹, hashcat, John the Ripper, 크리덴셜 스터핑, 세션 하이재킹, MFA 우회 |
| [07_Server_Side_Attacks.md](./07_Server_Side_Attacks.md) | ⭐⭐⭐ | SSRF, 명령어 인젝션, 파일 포함(LFI/RFI), 역직렬화 공격, SSTI |
| [08_Client_Side_Attacks.md](./08_Client_Side_Attacks.md) | ⭐⭐⭐ | DOM 기반 XSS, 클릭재킹, 브라우저 익스플로잇, postMessage 공격, 프로토타입 오염 |

### 바이너리 익스플로잇

| 파일명 | 난이도 | 주요 주제 |
|--------|--------|----------|
| [09_Binary_Fundamentals.md](./09_Binary_Fundamentals.md) | ⭐⭐⭐⭐ | x86/x64 어셈블리 기초, 호출 규약, 스택 레이아웃, ELF 형식, 메모리 레이아웃 |
| [10_Buffer_Overflow.md](./10_Buffer_Overflow.md) | ⭐⭐⭐⭐ | 스택 오버플로우, ROP 체인, NX 우회, ASLR, 카나리, 포맷 스트링 공격 |
| [11_Reverse_Engineering.md](./11_Reverse_Engineering.md) | ⭐⭐⭐⭐ | Ghidra를 이용한 정적 분석, GDB를 이용한 동적 분석, 디컴파일, 안티리버싱 기법 |

### 시스템 및 인프라

| 파일명 | 난이도 | 주요 주제 |
|--------|--------|----------|
| [12_Privilege_Escalation_Linux.md](./12_Privilege_Escalation_Linux.md) | ⭐⭐⭐⭐ | SUID/SGID, 리눅스 capabilities, 커널 익스플로잇, cron 설정 오류, 경로 하이재킹 |
| [13_Privilege_Escalation_Windows.md](./13_Privilege_Escalation_Windows.md) | ⭐⭐⭐⭐ | 토큰 사칭, 서비스 익스플로잇, UAC 우회, DLL 하이재킹, 인용되지 않은 경로 |
| [14_Active_Directory.md](./14_Active_Directory.md) | ⭐⭐⭐⭐ | AD 열거, Kerberoasting, Pass-the-Hash, BloodHound, LDAP 인젝션, 골든 티켓 |
| [15_Post_Exploitation.md](./15_Post_Exploitation.md) | ⭐⭐⭐⭐ | 지속성 메커니즘, 횡적 이동, 데이터 유출, C2 기초, 피벗팅 |

### 전문 분야

| 파일명 | 난이도 | 주요 주제 |
|--------|--------|----------|
| [16_Wireless_Security.md](./16_Wireless_Security.md) | ⭐⭐⭐⭐ | WiFi 공격, WPA2/WPA3 크래킹, 이블 트윈, 블루투스 공격, 인증 해제 공격 |
| [17_Cloud_Security_Testing.md](./17_Cloud_Security_Testing.md) | ⭐⭐⭐⭐ | AWS/GCP 설정 오류, IAM 악용, 메타데이터 공격, S3 버킷 열거 |
| [18_Malware_Analysis.md](./18_Malware_Analysis.md) | ⭐⭐⭐⭐ | 정적/동적 분석, 샌드박싱, YARA 규칙, PE 구조, 행위 분석 |

### 작전 및 대회

| 파일명 | 난이도 | 주요 주제 |
|--------|--------|----------|
| [19_CTF_Methodology.md](./19_CTF_Methodology.md) | ⭐⭐⭐ | CTF 카테고리(pwn, web, crypto, forensics, misc), 도구, 풀이법, pwntools |
| [20_Red_Team_Operations.md](./20_Red_Team_Operations.md) | ⭐⭐⭐⭐ | 레드팀 계획, 위협 에뮬레이션, 보고, 퍼플 팀, MITRE ATT&CK, 교정 조치 |

## 참고 자료

- OWASP Testing Guide v4.2: https://owasp.org/www-project-web-security-testing-guide/
- PTES (Penetration Testing Execution Standard): http://www.pentest-standard.org/
- MITRE ATT&CK Framework: https://attack.mitre.org/
- NIST SP 800-115 (정보보안 테스트 기술 가이드): https://csrc.nist.gov/publications/detail/sp/800-115/final
- The Web Application Hacker's Handbook (Stuttard & Pinto)
- Hacking: The Art of Exploitation (Erickson)
- Red Team Field Manual (RTFM)
