# 정찰

**이전**: [01. 공격 보안 개요](./01_Offensive_Security_Overview.md) | **다음**: [03. 네트워크 스캐닝](./03_Network_Scanning.md)

---

정찰은 모든 침투 테스트의 첫 번째이자 가장 중요한 단계입니다. 정찰 중 수집된 정보의 품질이 후속 익스플로잇 단계의 효과를 직접적으로 결정합니다. 전문 테스터는 활동 시간의 50-75%를 정찰과 열거에 할애합니다. 잘 매핑된 공격 표면은 최소 저항 경로를 드러내기 때문입니다.

> **중요**: 명시적인 서면 허가를 받은 대상에 대해서만 정찰을 수행하세요. 일부 관할권에서는 수동 OSINT조차 법적 영향을 미칠 수 있습니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수동 정찰과 능동 정찰 구별
2. 여러 소스를 사용한 OSINT 수집 수행
3. DNS 레코드 열거 및 서브도메인 발견
4. 구글 도킹을 사용하여 노출된 민감한 정보 검색
5. Shodan, Censys 및 기타 인터넷 전체 스캐너 활용
6. 공개 문서에서 메타데이터 추출
7. 조직 구조 및 직원 정보 매핑
8. Python을 사용한 정찰 워크플로우 자동화

---

## 목차

1. [수동 vs 능동 정찰](#1-수동-vs-능동-정찰)
2. [OSINT 기초](#2-osint-기초)
3. [DNS 열거](#3-dns-열거)
4. [서브도메인 발견](#4-서브도메인-발견)
5. [구글 도킹](#5-구글-도킹)
6. [Shodan과 인터넷 전체 스캐닝](#6-shodan과-인터넷-전체-스캐닝)
7. [WHOIS와 도메인 인텔리전스](#7-whois와-도메인-인텔리전스)
8. [메타데이터 추출](#8-메타데이터-추출)
9. [소셜 미디어 및 직원 OSINT](#9-소셜-미디어-및-직원-osint)
10. [자동화된 정찰 프레임워크](#10-자동화된-정찰-프레임워크)
11. [대응 방안 및 탐지](#11-대응-방안-및-탐지)
12. [연습문제](#12-연습문제)
13. [요약](#13-요약)
14. [참고 자료](#14-참고-자료)

---

## 1. 수동 vs 능동 정찰

정찰은 대상과의 상호작용 수준에 따라 두 가지 큰 카테고리로 나뉩니다:

### 1.1 수동 정찰

수동 정찰은 대상 시스템과 직접 상호작용하지 않고 정보를 수집합니다. 대상은 조사되고 있다는 것을 감지할 수 없습니다.

**수동 인텔리전스 소스:**
- 공개 DNS 레코드 및 WHOIS 데이터베이스
- 검색 엔진 캐시 페이지 및 인덱싱된 콘텐츠
- 소셜 미디어 프로필 및 채용 공고
- Certificate Transparency 로그
- 인터넷 아카이브 (Wayback Machine)
- 코드 저장소 (GitHub, GitLab)
- Shodan, Censys 및 기타 스캔 데이터베이스
- SEC 제출, 보도 자료, 특허 데이터베이스

**장점**: 탐지 불가, 대부분의 관할권에서 합법, 공식 활동 시작 전에 수행 가능.

**한계**: 정보가 오래되었거나 불완전하거나 부정확할 수 있습니다.

### 1.2 능동 정찰

능동 정찰은 대상과 직접 상호작용 — 패킷 전송, 요청 생성, 서비스 프로빙. 대상이 이 활동을 감지할 수 있습니다.

**능동 정찰 기법:**
- 포트 스캐닝 및 서비스 열거
- 웹 스파이더링 및 콘텐츠 발견
- 배너 그래빙 및 버전 탐지
- DNS 존 전송 시도
- 디렉토리 브루트포싱
- 가상 호스트 열거

**장점**: 현재적이고 정확하며 포괄적인 데이터.

**위험**: IDS/IPS에 의해 탐지 가능, 경보 발생 가능, 허가 필요.

```python
"""
정찰 분류 프레임워크.

활동 계획을 위해 다양한 정찰 활동과
그 위험 수준을 분류하고 추적합니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReconType(Enum):
    PASSIVE = "수동"
    ACTIVE = "능동"
    SEMI_PASSIVE = "반수동"


class DetectionRisk(Enum):
    NONE = 0        # 탐지 불가
    LOW = 1         # 경보 발생 가능성 낮음
    MEDIUM = 2      # 로그에 나타날 수 있음
    HIGH = 3        # IDS/IPS 경보 발생 가능성 높음
    CERTAIN = 4     # 반드시 로그에 기록됨


@dataclass
class ReconTechnique:
    """메타데이터가 포함된 정찰 기법."""
    name: str
    recon_type: ReconType
    detection_risk: DetectionRisk
    description: str
    tools: list[str] = field(default_factory=list)
    data_gathered: list[str] = field(default_factory=list)
    authorization_required: bool = True
    notes: str = ""


RECON_TECHNIQUES = [
    ReconTechnique(
        name="WHOIS 조회",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="도메인 등록 정보를 위한 공개 WHOIS DB 조회",
        tools=["whois", "python-whois"],
        data_gathered=["등록 기관", "네임서버", "생성일",
                       "관리자 연락처 (비공개 미처리 시)"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="DNS 레코드 열거",
        recon_type=ReconType.SEMI_PASSIVE,
        detection_risk=DetectionRisk.LOW,
        description="다양한 레코드 유형에 대한 공개 DNS 서버 쿼리",
        tools=["dig", "nslookup", "dnspython"],
        data_gathered=["A 레코드", "MX 레코드", "TXT 레코드",
                       "NS 레코드", "CNAME 레코드"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="서브도메인 브루트포스",
        recon_type=ReconType.ACTIVE,
        detection_risk=DetectionRisk.MEDIUM,
        description="워드리스트로 대상 DNS에서 서브도메인 해결",
        tools=["subfinder", "amass", "ffuf", "gobuster"],
        data_gathered=["숨겨진 서브도메인", "내부 호스트명",
                       "개발/스테이징 서버"],
        authorization_required=True,
    ),
    ReconTechnique(
        name="포트 스캐닝",
        recon_type=ReconType.ACTIVE,
        detection_risk=DetectionRisk.HIGH,
        description="열린 포트 및 서비스를 위한 대상 IP 대역 프로빙",
        tools=["nmap", "masscan", "rustscan"],
        data_gathered=["열린 포트", "서비스 버전", "OS 핑거프린트"],
        authorization_required=True,
    ),
    ReconTechnique(
        name="Certificate Transparency",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="대상 도메인에 발급된 인증서를 위한 CT 로그 검색",
        tools=["crt.sh", "certspotter", "ctfr"],
        data_gathered=["서브도메인", "인증서 세부 사항",
                       "발급 CA", "이력 인증서"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="구글 도킹",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="노출된 데이터를 찾기 위한 고급 검색 엔진 쿼리",
        tools=["Google", "DorkSearch", "GHDB"],
        data_gathered=["노출된 파일", "로그인 페이지",
                       "오류 메시지", "민감한 디렉토리"],
        authorization_required=False,
        notes="책임감 있게 사용 — 허가 없이 발견된 자료에 접근하지 마세요",
    ),
    ReconTechnique(
        name="Shodan 검색",
        recon_type=ReconType.PASSIVE,
        detection_risk=DetectionRisk.NONE,
        description="인터넷 연결 장치 데이터베이스인 Shodan 쿼리",
        tools=["Shodan CLI", "Shodan API", "shodan-python"],
        data_gathered=["열린 포트", "배너", "취약점",
                       "SSL 인증서", "스크린샷"],
        authorization_required=False,
    ),
    ReconTechnique(
        name="웹 콘텐츠 발견",
        recon_type=ReconType.ACTIVE,
        detection_risk=DetectionRisk.HIGH,
        description="웹 서버의 디렉토리 및 파일 브루트포스",
        tools=["gobuster", "ffuf", "dirsearch", "feroxbuster"],
        data_gathered=["숨겨진 디렉토리", "백업 파일",
                       "관리자 패널", "API 엔드포인트"],
        authorization_required=True,
    ),
]


def plan_recon(
    authorized: bool = True,
    stealth_required: bool = False,
) -> list[ReconTechnique]:
    """
    제약 조건에 따라 적절한 정찰 기법을 선택합니다.

    Args:
        authorized: 공식 허가 획득 여부
        stealth_required: 탐지 회피 필요 여부
    """
    techniques = []
    for tech in RECON_TECHNIQUES:
        if not authorized and tech.authorization_required:
            continue
        if stealth_required and tech.detection_risk.value > DetectionRisk.LOW.value:
            continue
        techniques.append(tech)
    return techniques


if __name__ == "__main__":
    print("=== 허가 전 정찰 (수동만) ===\n")
    passive = plan_recon(authorized=False)
    for t in passive:
        print(f"  [{t.recon_type.value:4s}] {t.name}")
        print(f"    탐지 위험: {t.detection_risk.name}")
        print(f"    도구: {', '.join(t.tools)}")
        print()

    print("\n=== 전체 허가된 정찰 ===\n")
    full = plan_recon(authorized=True)
    for t in full:
        print(f"  [{t.recon_type.value:4s}] {t.name}")
        print(f"    탐지 위험: {t.detection_risk.name}")
        print()
```

---

## 2. OSINT 기초

OSINT(Open Source Intelligence)는 공개적으로 사용 가능한 소스에서 정보를 수집하고 분석하는 것입니다. OSINT는 수동 정찰의 핵심을 이룹니다.

### 2.1 OSINT 프로세스

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   계획 &       │ ──▶ │    수집        │ ──▶ │    처리        │
│   방향 설정    │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
                                                     │
┌───────────────┐     ┌───────────────┐              ▼
│  배포 &        │ ◀── │    분석        │ ◀── ┌───────────────┐
│  피드백        │     │               │     │    검증        │
└───────────────┘     └───────────────┘     └───────────────┘
```

### 2.2 OSINT 소스 카테고리

**기술 소스:**
- DNS 레코드, IP 할당 (ARIN, RIPE, APNIC)
- BGP 라우팅 테이블 및 AS 번호
- Certificate Transparency 로그 (crt.sh)
- 인터넷 스캐닝 데이터베이스 (Shodan, Censys, ZoomEye)
- 코드 저장소 (GitHub, GitLab, Bitbucket)
- 페이스트 사이트 (Pastebin, GitHub Gists)

**비즈니스 소스:**
- 회사 웹사이트 및 보도 자료
- SEC 제출 및 재무 보고서
- 특허 데이터베이스
- 채용 공고 (LinkedIn, Indeed)
- 벤더 및 파트너 공개 자료

**인적 소스:**
- 소셜 미디어 프로필 (LinkedIn, Twitter, GitHub)
- 컨퍼런스 발표 및 논문
- 이메일 주소 및 사용자명
- 공개 기록 및 데이터 침해 덤프 (Have I Been Pwned)

### 2.3 OSINT 자동화

```python
"""
OSINT 데이터 수집기 — 허가된 정찰을 위해
여러 공개 소스의 정보를 집계합니다.

참고: 이 모듈은 공개적으로 사용 가능한 데이터만 쿼리합니다.
실제 대상에 사용하기 전에 항상 허가를 확인하세요.
"""

import json
import socket
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OSINTResult:
    """OSINT에서 수집된 단일 인텔리전스."""
    source: str
    category: str
    data: dict
    confidence: float  # 0.0 ~ 1.0
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class TargetProfile:
    """대상 조직의 집계된 OSINT 프로필."""
    domain: str
    results: list[OSINTResult] = field(default_factory=list)
    subdomains: set[str] = field(default_factory=set)
    ip_addresses: set[str] = field(default_factory=set)
    email_addresses: set[str] = field(default_factory=set)
    technologies: set[str] = field(default_factory=set)
    employees: list[dict] = field(default_factory=list)

    def add_result(self, result: OSINTResult) -> None:
        """OSINT 결과를 추가하고 관련 필드를 업데이트합니다."""
        self.results.append(result)

    def summary(self) -> str:
        """수집된 인텔리전스 요약을 생성합니다."""
        lines = [
            f"OSINT 프로필: {self.domain}",
            f"{'=' * 50}",
            f"발견된 서브도메인:  {len(self.subdomains)}",
            f"발견된 IP 주소:     {len(self.ip_addresses)}",
            f"발견된 이메일 주소: {len(self.email_addresses)}",
            f"탐지된 기술:        {len(self.technologies)}",
            f"식별된 직원:        {len(self.employees)}",
            f"총 데이터 포인트:   {len(self.results)}",
        ]
        if self.subdomains:
            lines.append(f"\n서브도메인:")
            for sub in sorted(self.subdomains)[:20]:
                lines.append(f"  - {sub}")
        if self.technologies:
            lines.append(f"\n기술:")
            for tech in sorted(self.technologies):
                lines.append(f"  - {tech}")
        return "\n".join(lines)


def resolve_dns_records(domain: str) -> list[OSINTResult]:
    """
    도메인의 일반적인 DNS 레코드 유형을 해결합니다.

    실제로는 포괄적인 쿼리를 위해 dnspython을 사용합니다.
    이 단순화된 버전은 개념을 시연합니다.
    """
    results = []

    try:
        # A 레코드 조회
        ips = socket.getaddrinfo(domain, None, socket.AF_INET)
        unique_ips = set(ip[4][0] for ip in ips)
        for ip in unique_ips:
            results.append(OSINTResult(
                source="DNS",
                category="infrastructure",
                data={"record_type": "A", "domain": domain, "ip": ip},
                confidence=1.0,
            ))
    except socket.gaierror:
        pass

    return results


def check_common_subdomains(domain: str) -> list[str]:
    """
    도메인에 대해 일반적인 서브도메인 접두사를 확인합니다.

    이는 단순화된 버전입니다 — 실제 도구는 훨씬 더 큰
    워드리스트와 동시 해결을 사용합니다.
    """
    common_prefixes = [
        "www", "mail", "ftp", "smtp", "pop", "imap",
        "admin", "portal", "vpn", "remote", "api",
        "dev", "staging", "test", "beta", "demo",
        "git", "gitlab", "github", "jenkins", "ci",
        "blog", "shop", "store", "app", "mobile",
        "cdn", "static", "media", "assets", "images",
        "ns1", "ns2", "dns", "mx", "relay",
        "db", "database", "mongo", "redis", "elastic",
        "grafana", "prometheus", "kibana", "sentry",
        "jira", "confluence", "wiki", "docs",
        "backup", "old", "legacy", "archive",
    ]

    found = []
    for prefix in common_prefixes:
        subdomain = f"{prefix}.{domain}"
        try:
            socket.getaddrinfo(subdomain, None, socket.AF_INET)
            found.append(subdomain)
        except socket.gaierror:
            pass

    return found


def generate_email_patterns(
    domain: str,
    first_name: str,
    last_name: str,
) -> list[str]:
    """
    직원의 일반적인 이메일 주소 패턴을 생성합니다.

    허가된 활동에서 피싱 시뮬레이션 또는 크리덴셜 테스트에 유용합니다.
    """
    fn = first_name.lower()
    ln = last_name.lower()
    fi = fn[0] if fn else ""
    li = ln[0] if ln else ""

    patterns = [
        f"{fn}.{ln}@{domain}",          # john.doe@company.com
        f"{fn}{ln}@{domain}",            # johndoe@company.com
        f"{fi}{ln}@{domain}",            # jdoe@company.com
        f"{fn}{li}@{domain}",            # johnd@company.com
        f"{fn}_{ln}@{domain}",           # john_doe@company.com
        f"{ln}.{fn}@{domain}",           # doe.john@company.com
        f"{fn}@{domain}",               # john@company.com
        f"{fi}.{ln}@{domain}",           # j.doe@company.com
        f"{fn}{ln[0:3]}@{domain}",       # johndoe@company.com
    ]
    return list(dict.fromkeys(patterns))  # 순서 유지하며 중복 제거


# 시연
if __name__ == "__main__":
    # 안전한 시연을 위해 example.com 사용
    profile = TargetProfile(domain="example.com")

    # DNS 열거
    dns_results = resolve_dns_records("example.com")
    for result in dns_results:
        profile.add_result(result)
        profile.ip_addresses.add(result.data.get("ip", ""))

    # 이메일 패턴 생성
    patterns = generate_email_patterns("example.com", "Jane", "Smith")
    print("생성된 이메일 패턴:")
    for pattern in patterns:
        print(f"  {pattern}")

    print("\n" + profile.summary())
```

---

## 3. DNS 열거

DNS는 정찰 중 가장 풍부한 정보 소스 중 하나입니다. 모든 조직의 DNS 레코드는 인프라 세부 사항을 드러냅니다.

### 3.1 정찰을 위한 DNS 레코드 유형

| 레코드 유형 | 드러나는 정보 | 정찰 가치 |
|------------|--------------|----------|
| A / AAAA | IPv4/IPv6 주소 | 인프라 매핑 |
| MX | 메일 서버 | 이메일 제공자 식별 |
| NS | 네임서버 | DNS 인프라 |
| TXT | SPF, DKIM, DMARC, 검증 | 이메일 보안 태세 |
| CNAME | 별칭 및 CDN 사용 | 서비스 제공자 |
| SOA | 기본 NS, 관리자 이메일 | 관리 정보 |
| SRV | 서비스 위치 | 내부 서비스 |
| PTR | 역방향 DNS | 호스트명 발견 |

### 3.2 DNS 열거 명령어

```bash
# dig를 사용한 기본 DNS 쿼리
dig example.com A +short
dig example.com MX +short
dig example.com TXT +short
dig example.com NS +short
dig example.com ANY +noall +answer

# DNS 존 전송 시도 (보통 차단됨)
dig axfr example.com @ns1.example.com

# 역방향 DNS 조회
dig -x 93.184.216.34 +short

# 모든 레코드 유형 DNS 열거
for type in A AAAA MX NS TXT CNAME SOA SRV; do
    echo "--- $type ---"
    dig example.com $type +short
done
```

### 3.3 DNS 존 전송 (Zone Transfer)

DNS 존 전송(AXFR)은 네임서버에서 전체 DNS 존을 복사합니다. 잘못 설정된 경우 존의 모든 호스트명을 노출합니다:

```python
"""
DNS 존 전송 확인기.

도메인의 네임서버가 존 전송을 허용하는지 테스트합니다.
잘못 설정된 DNS 서버는 내부 호스트명과 IP 주소를 노출하는
전체 존 파일을 노출할 수 있습니다.

허가된 대상에 대해서만 테스트하세요.
"""

import socket
import struct
from typing import Optional


def build_axfr_query(domain: str) -> bytes:
    """
    원시 DNS AXFR (존 전송) 쿼리 패킷을 구성합니다.

    이는 바이너리 수준에서 DNS 프로토콜을 시연합니다.
    실제로는 dnspython: dns.zone.from_xfr()를 사용하세요.
    """
    # 트랜잭션 ID (임의)
    import random
    txn_id = random.randint(0, 65535)

    # DNS 헤더
    flags = 0x0000  # 표준 쿼리
    qdcount = 1     # 질문 하나
    ancount = 0
    nscount = 0
    arcount = 0

    header = struct.pack(
        "!HHHHHH",
        txn_id, flags, qdcount, ancount, nscount, arcount
    )

    # 질문 섹션
    question = b""
    for label in domain.split("."):
        question += struct.pack("!B", len(label)) + label.encode()
    question += b"\x00"  # 루트 레이블

    # QTYPE=252 (AXFR), QCLASS=1 (IN)
    question += struct.pack("!HH", 252, 1)

    return header + question


def attempt_zone_transfer(
    domain: str,
    nameserver: str,
    timeout: float = 10.0,
) -> Optional[str]:
    """
    네임서버에 대한 DNS 존 전송을 시도합니다.

    성공하면 존 데이터를 반환하고, 차단되면 None을 반환합니다.
    대부분의 올바르게 설정된 서버는 이를 거부합니다.
    """
    try:
        # 쿼리 구성
        query = build_axfr_query(domain)

        # DNS 존 전송은 TCP 사용
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((nameserver, 53))

        # TCP DNS는 2바이트 길이 접두사 필요
        tcp_query = struct.pack("!H", len(query)) + query
        sock.send(tcp_query)

        # 응답 길이 읽기
        length_data = sock.recv(2)
        if len(length_data) < 2:
            return None

        response_length = struct.unpack("!H", length_data)[0]
        response = b""
        while len(response) < response_length:
            chunk = sock.recv(response_length - len(response))
            if not chunk:
                break
            response += chunk

        sock.close()

        # RCODE 확인 (플래그의 비트 12-15)
        if len(response) < 4:
            return None

        flags = struct.unpack("!H", response[2:4])[0]
        rcode = flags & 0x000F

        if rcode == 0:
            return f"존 전송 성공: {domain} <- {nameserver}"
        elif rcode == 5:
            return None  # REFUSED — 올바르게 설정됨
        else:
            return None

    except (socket.timeout, ConnectionRefusedError, OSError):
        return None


if __name__ == "__main__":
    # 알려진 테스트 도메인에 대한 안전한 시연
    print("DNS 존 전송 확인기")
    print("=" * 40)
    print("\n허가된 테스트에서의 사용법:")
    print("  1. NS 레코드 열거: dig example.com NS")
    print("  2. AXFR 시도: dig axfr example.com @ns1.example.com")
    print("  3. 또는 적절한 대상으로 이 스크립트 사용")
    print("\n참고: 허가된 도메인만 테스트하세요.")

    # 원시 쿼리 구조 표시
    query = build_axfr_query("example.com")
    print(f"\n원시 AXFR 쿼리 ({len(query)} 바이트):")
    print(f"  헤더:   {query[:12].hex()}")
    print(f"  질문: {query[12:].hex()}")
```

---

## 4. 서브도메인 발견

서브도메인을 발견하면 조직의 웹 존재 전체 범위를 드러내며, 종종 잊혀진 개발 서버, 스테이징 환경, 관리 패널이 발견됩니다.

### 4.1 발견 방법

**수동 방법** (직접 상호작용 없음):
- Certificate Transparency 로그 (crt.sh)
- 검색 엔진 인덱싱
- DNS 집계 데이터베이스 (VirusTotal, SecurityTrails)
- 웹 아카이브 (archive.org)

**능동 방법** (직접 상호작용):
- 워드리스트를 이용한 DNS 브루트포싱
- 가상 호스트 열거
- DNS 존 전송 (잘못 설정된 경우)
- 웹 크롤링 및 링크 추출

### 4.2 Certificate Transparency

CT(Certificate Transparency) 로그는 서브도메인 발견의 보고입니다. 신뢰할 수 있는 CA가 발급한 모든 SSL 인증서는 로그에 기록되어야 하며, 이 로그는 공개 검색이 가능합니다.

```python
"""
여러 기법을 통한 서브도메인 발견.

대상 도메인의 포괄적인 서브도메인 목록을 구축하기 위해
수동 및 능동 방법을 결합합니다.
"""

import json
import socket
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


@dataclass
class SubdomainResult:
    """메타데이터가 포함된 발견된 서브도메인."""
    subdomain: str
    source: str
    ip_address: Optional[str] = None
    is_alive: bool = False
    http_status: Optional[int] = None
    http_title: Optional[str] = None


class SubdomainEnumerator:
    """
    다중 소스 서브도메인 열거기.

    포괄적인 서브도메인 발견을 위해 수동 소스(CT 로그)와
    능동 해결을 결합합니다.
    """

    # 일반적인 서브도메인 워드리스트 (축약 — 실제 목록은 10,000+ 항목)
    COMMON_SUBDOMAINS = [
        "www", "mail", "ftp", "localhost", "webmail", "smtp", "pop",
        "ns1", "ns2", "dns", "dns1", "dns2", "mx", "mx1",
        "admin", "administrator", "portal", "webadmin",
        "api", "api2", "api3", "rest", "graphql",
        "dev", "develop", "development", "staging", "stage",
        "test", "testing", "qa", "uat", "sandbox",
        "beta", "alpha", "demo", "preview",
        "git", "gitlab", "github", "bitbucket", "svn",
        "jenkins", "ci", "cd", "build", "deploy",
        "app", "application", "mobile", "m",
        "blog", "news", "forum", "community", "support",
        "shop", "store", "ecommerce", "cart", "pay",
        "vpn", "remote", "gateway", "proxy",
        "db", "database", "sql", "mysql", "postgres", "mongo",
        "redis", "memcached", "elasticsearch", "elastic",
        "grafana", "prometheus", "kibana", "sentry", "monitoring",
        "cdn", "static", "assets", "media", "images", "img",
        "backup", "bak", "old", "legacy", "archive",
        "internal", "intranet", "extranet", "corp", "corporate",
        "sso", "auth", "login", "oauth", "idp",
        "docs", "documentation", "wiki", "confluence", "jira",
        "status", "health", "healthcheck", "ping",
    ]

    def __init__(self, domain: str, max_workers: int = 10):
        self.domain = domain
        self.max_workers = max_workers
        self.results: dict[str, SubdomainResult] = {}

    def query_crt_sh(self) -> list[str]:
        """
        crt.sh Certificate Transparency 로그를 쿼리합니다.

        crt.sh는 CT 로그 데이터에 대한 무료 접근을 제공합니다.
        """
        url = f"https://crt.sh/?q=%.{self.domain}&output=json"
        subdomains = set()

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urlopen(req, timeout=15)
            data = json.loads(response.read())

            for entry in data:
                name = entry.get("name_value", "")
                # CT 항목에는 와일드카드 및 여러 이름이 포함될 수 있음
                for sub in name.split("\n"):
                    sub = sub.strip().lower()
                    if sub.startswith("*."):
                        sub = sub[2:]
                    if sub.endswith(f".{self.domain}") or sub == self.domain:
                        subdomains.add(sub)

        except (URLError, json.JSONDecodeError, Exception) as e:
            print(f"  [!] crt.sh 쿼리 실패: {e}")

        return list(subdomains)

    def brute_force(self, wordlist: Optional[list[str]] = None) -> list[str]:
        """
        워드리스트를 이용한 DNS 브루트포스.

        각 잠재적 서브도메인을 해결하여 존재 여부를 확인합니다.
        """
        if wordlist is None:
            wordlist = self.COMMON_SUBDOMAINS

        found = []

        def check_subdomain(prefix: str) -> Optional[str]:
            subdomain = f"{prefix}.{self.domain}"
            try:
                socket.getaddrinfo(subdomain, None, socket.AF_INET)
                return subdomain
            except socket.gaierror:
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(check_subdomain, prefix): prefix
                for prefix in wordlist
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    found.append(result)

        return found

    def resolve_subdomain(self, subdomain: str) -> SubdomainResult:
        """서브도메인을 해결하고 추가 정보를 수집합니다."""
        result = SubdomainResult(
            subdomain=subdomain,
            source="resolution",
        )

        try:
            ips = socket.getaddrinfo(subdomain, None, socket.AF_INET)
            if ips:
                result.ip_address = ips[0][4][0]
                result.is_alive = True
        except socket.gaierror:
            pass

        return result

    def enumerate(self) -> list[SubdomainResult]:
        """
        전체 서브도메인 열거 파이프라인을 실행합니다.

        수동 및 능동 기법을 결합합니다.
        """
        all_subdomains = set()

        # 수동: Certificate Transparency
        print(f"[*] {self.domain}에 대한 Certificate Transparency 쿼리 중...")
        ct_subs = self.query_crt_sh()
        print(f"  [+] CT 로그를 통해 서브도메인 {len(ct_subs)}개 발견")
        all_subdomains.update(ct_subs)

        # 능동: DNS 브루트포스
        print(f"[*] 일반적인 서브도메인 브루트포싱 중...")
        bf_subs = self.brute_force()
        print(f"  [+] 브루트포스로 서브도메인 {len(bf_subs)}개 발견")
        all_subdomains.update(bf_subs)

        # 발견된 모든 서브도메인 해결
        print(f"[*] 고유한 서브도메인 {len(all_subdomains)}개 해결 중...")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.resolve_subdomain, sub): sub
                for sub in all_subdomains
            }
            for future in as_completed(futures):
                result = future.result()
                if result.is_alive:
                    results.append(result)

        self.results = {r.subdomain: r for r in results}
        return results

    def report(self) -> str:
        """서브도메인 열거 보고서를 생성합니다."""
        lines = [
            f"서브도메인 열거 보고서: {self.domain}",
            "=" * 60,
            f"해결된 고유 서브도메인 총계: {len(self.results)}",
            "",
        ]
        for sub, result in sorted(self.results.items()):
            ip = result.ip_address or "N/A"
            lines.append(f"  {sub:40s} -> {ip}")
        return "\n".join(lines)


if __name__ == "__main__":
    # 안전한 시연
    print("서브도메인 열거기")
    print("=" * 40)
    print("사용법: enumerator = SubdomainEnumerator('target.com')")
    print("       results = enumerator.enumerate()")
    print("       print(enumerator.report())")
    print("\n허가된 대상에서만 사용하세요.")

    # 워드리스트 크기 표시
    print(f"\n기본 내장 워드리스트: {len(SubdomainEnumerator.COMMON_SUBDOMAINS)}개 항목")
    print("처음 10개 항목:", SubdomainEnumerator.COMMON_SUBDOMAINS[:10])
```

---

## 5. 구글 도킹

구글 도킹(Google hacking)은 고급 검색 연산자를 사용하여 조직이 검색 엔진 크롤러에 실수로 노출한 정보를 찾습니다.

### 5.1 필수 검색 연산자

| 연산자 | 목적 | 예시 |
|--------|------|------|
| `site:` | 특정 도메인 제한 | `site:example.com` |
| `inurl:` | URL 경로 검색 | `inurl:admin` |
| `intitle:` | 페이지 제목 검색 | `intitle:"index of"` |
| `filetype:` | 특정 파일 유형 찾기 | `filetype:pdf` |
| `intext:` | 페이지 내용 검색 | `intext:"password"` |
| `ext:` | 파일 확장자 | `ext:sql` |
| `cache:` | 페이지의 캐시된 버전 | `cache:example.com` |
| `-` | 용어 제외 | `site:example.com -www` |
| `""` | 정확히 일치 | `"error in your SQL syntax"` |
| `*` | 와일드카드 | `site:*.example.com` |

### 5.2 일반적인 구글 도크

```python
"""
허가된 정찰을 위한 구글 도크 생성기.

잠재적으로 노출된 정보를 찾기 위한 검색 쿼리를 생성합니다.
발견된 자원에 접근하기 전에 항상 허가를 확인하세요.
"""

from dataclasses import dataclass
from enum import Enum


class DorkCategory(Enum):
    SENSITIVE_FILES = "민감한 파일"
    LOGIN_PAGES = "로그인 페이지"
    ERROR_MESSAGES = "오류 메시지"
    DIRECTORY_LISTINGS = "디렉토리 목록"
    DATABASE_DUMPS = "데이터베이스 덤프"
    CONFIG_FILES = "설정 파일"
    BACKUP_FILES = "백업 파일"
    SOURCE_CODE = "소스 코드"
    CREDENTIALS = "크리덴셜"
    INFRASTRUCTURE = "인프라"


@dataclass
class GoogleDork:
    """메타데이터가 포함된 구글 도크 쿼리."""
    query_template: str
    category: DorkCategory
    description: str
    risk_level: str  # info, low, medium, high, critical

    def for_domain(self, domain: str) -> str:
        """특정 도메인에 대한 도크 쿼리를 생성합니다."""
        return self.query_template.replace("{domain}", domain)


# 표준 구글 도크 데이터베이스
DORK_DATABASE = [
    # 민감한 파일
    GoogleDork(
        'site:{domain} filetype:pdf "confidential"',
        DorkCategory.SENSITIVE_FILES,
        "기밀로 표시된 PDF 파일 찾기",
        "medium",
    ),
    GoogleDork(
        'site:{domain} filetype:xlsx OR filetype:csv "password" OR "username"',
        DorkCategory.SENSITIVE_FILES,
        "크리덴셜 데이터가 포함된 스프레드시트 찾기",
        "high",
    ),
    GoogleDork(
        'site:{domain} filetype:doc OR filetype:docx "internal" OR "draft"',
        DorkCategory.SENSITIVE_FILES,
        "내부 또는 초안 문서 찾기",
        "low",
    ),

    # 로그인 페이지
    GoogleDork(
        'site:{domain} inurl:login OR inurl:signin OR inurl:admin',
        DorkCategory.LOGIN_PAGES,
        "인증 페이지 찾기",
        "info",
    ),
    GoogleDork(
        'site:{domain} intitle:"admin panel" OR intitle:"dashboard"',
        DorkCategory.LOGIN_PAGES,
        "관리자 패널 및 대시보드 찾기",
        "medium",
    ),

    # 오류 메시지
    GoogleDork(
        'site:{domain} "error in your SQL syntax" OR "mysql_fetch"',
        DorkCategory.ERROR_MESSAGES,
        "SQL 오류 메시지 찾기 (잠재적 SQLi)",
        "high",
    ),
    GoogleDork(
        'site:{domain} "Fatal error" OR "Warning:" filetype:php',
        DorkCategory.ERROR_MESSAGES,
        "스택 트레이스가 있는 PHP 오류 페이지 찾기",
        "medium",
    ),
    GoogleDork(
        'site:{domain} "Traceback (most recent call last)"',
        DorkCategory.ERROR_MESSAGES,
        "Python 스택 트레이스 찾기",
        "medium",
    ),

    # 디렉토리 목록
    GoogleDork(
        'site:{domain} intitle:"index of" "parent directory"',
        DorkCategory.DIRECTORY_LISTINGS,
        "열린 디렉토리 목록 찾기",
        "medium",
    ),
    GoogleDork(
        'site:{domain} intitle:"index of" ".git"',
        DorkCategory.DIRECTORY_LISTINGS,
        "노출된 .git 디렉토리 찾기",
        "critical",
    ),

    # 설정 파일
    GoogleDork(
        'site:{domain} filetype:env OR filetype:ini OR filetype:cfg',
        DorkCategory.CONFIG_FILES,
        "설정 파일 찾기",
        "high",
    ),
    GoogleDork(
        'site:{domain} filetype:xml "password" OR "secret"',
        DorkCategory.CONFIG_FILES,
        "크리덴셜이 있는 XML 설정 찾기",
        "high",
    ),
    GoogleDork(
        'site:{domain} filetype:yaml OR filetype:yml "apikey" OR "api_key"',
        DorkCategory.CONFIG_FILES,
        "API 키가 있는 YAML 파일 찾기",
        "critical",
    ),

    # 백업 파일
    GoogleDork(
        'site:{domain} filetype:bak OR filetype:old OR filetype:backup',
        DorkCategory.BACKUP_FILES,
        "백업 파일 찾기",
        "medium",
    ),
    GoogleDork(
        'site:{domain} filetype:sql "INSERT INTO" OR "CREATE TABLE"',
        DorkCategory.DATABASE_DUMPS,
        "SQL 데이터베이스 덤프 찾기",
        "critical",
    ),

    # 소스 코드
    GoogleDork(
        'site:github.com "{domain}" password OR secret OR api_key',
        DorkCategory.SOURCE_CODE,
        "GitHub 저장소에서 유출된 크리덴셜 찾기",
        "critical",
    ),
    GoogleDork(
        'site:pastebin.com "{domain}"',
        DorkCategory.SOURCE_CODE,
        "페이스트 사이트에서 대상 언급 찾기",
        "medium",
    ),
]


def generate_dork_report(domain: str) -> str:
    """도메인에 대한 카테고리별 도크 보고서를 생성합니다."""
    lines = [
        f"구글 도크 보고서: {domain}",
        "=" * 60,
        "",
        "참고: 허가가 있는 경우에만 발견된 자원에 접근하세요.",
        "",
    ]

    categories = sorted(set(d.category for d in DORK_DATABASE),
                       key=lambda c: c.value)
    for cat in categories:
        lines.append(f"\n--- {cat.value} ---")
        dorks = [d for d in DORK_DATABASE if d.category == cat]
        for dork in dorks:
            query = dork.for_domain(domain)
            lines.append(f"  [{dork.risk_level.upper():8s}] {dork.description}")
            lines.append(f"           쿼리: {query}")
    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_dork_report("example.com")
    print(report)
```

---

## 6. Shodan과 인터넷 전체 스캐닝

Shodan, Censys, ZoomEye는 인터넷을 지속적으로 스캔하고 연결된 장치에 대한 정보를 인덱싱합니다. 이러한 데이터베이스는 인터넷에 노출된 자산의 수동 정찰을 가능하게 합니다.

### 6.1 Shodan 검색 필터

| 필터 | 목적 | 예시 |
|------|------|------|
| `hostname:` | 호스트명으로 검색 | `hostname:example.com` |
| `ip:` | IP 주소로 검색 | `ip:93.184.216.34` |
| `port:` | 열린 포트로 검색 | `port:3389` |
| `org:` | 조직으로 검색 | `org:"Example Corp"` |
| `product:` | 소프트웨어로 검색 | `product:Apache` |
| `version:` | 버전으로 검색 | `version:2.4.49` |
| `ssl:` | SSL 인증서 필드 검색 | `ssl:"example.com"` |
| `vuln:` | CVE로 검색 | `vuln:CVE-2021-44228` |
| `country:` | 국가로 필터링 | `country:KR` |
| `city:` | 도시로 필터링 | `city:"Seoul"` |

### 6.2 유용한 Shodan 쿼리

```
# 노출된 데이터베이스 찾기
product:MongoDB port:27017 -authentication
product:Elasticsearch port:9200

# 취약한 웹 서버 찾기
http.title:"Index of /" port:80
Apache/2.4.49 country:KR

# IoT 장치 찾기
"Server: Hikvision" port:80
"Server: DVRDVS" port:80

# 산업 제어 시스템 찾기
port:502 Modbus
port:47808 "BACnet"

# 조직 자산 찾기
org:"대상 기업" port:443
ssl:"targetcorp.com"
```

---

## 7. WHOIS와 도메인 인텔리전스

WHOIS 레코드는 도메인 등록 정보를 제공하지만, 현대의 개인 정보 보호 조치(GDPR 준수 편집)로 가용 데이터가 제한됩니다.

```python
"""
WHOIS 및 도메인 인텔리전스 수집.

도메인 레코드에서 등록 정보, 네임서버,
이력 데이터를 추출합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class WHOISRecord:
    """파싱된 WHOIS 레코드 데이터."""
    domain: str
    registrar: str = ""
    creation_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    name_servers: list[str] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    registrant_org: str = ""
    registrant_country: str = ""
    dnssec: str = ""
    privacy_protected: bool = False

    @property
    def domain_age_days(self) -> Optional[int]:
        if self.creation_date:
            return (datetime.now() - self.creation_date).days
        return None

    @property
    def days_until_expiry(self) -> Optional[int]:
        if self.expiration_date:
            return (self.expiration_date - datetime.now()).days
        return None

    def security_observations(self) -> list[str]:
        """WHOIS 데이터에서 보안 관련 관찰을 생성합니다."""
        observations = []

        if self.privacy_protected:
            observations.append(
                "WHOIS 개인 정보 보호 활성화 — 등록자 세부 정보가 편집됨"
            )

        if self.days_until_expiry and self.days_until_expiry < 30:
            observations.append(
                f"도메인이 {self.days_until_expiry}일 후 만료 — "
                "갱신하지 않으면 도메인 탈취 가능성"
            )

        if self.dnssec.lower() in ("unsigned", ""):
            observations.append(
                "DNSSEC 미활성화 — DNS 스푸핑에 취약"
            )

        if len(self.name_servers) < 2:
            observations.append(
                "단일 네임서버 — DNS 이중화 없음"
            )

        return observations


def analyze_domain(domain: str) -> str:
    """
    WHOIS 및 DNS 정보를 결합한 포괄적인 도메인 분석.
    """
    # 실제로는 python-whois 라이브러리 사용
    # 이는 분석 구조를 시연합니다
    record = WHOISRecord(
        domain=domain,
        registrar="예시 등록 기관",
        creation_date=datetime(2010, 3, 15),
        expiration_date=datetime(2026, 3, 15),
        name_servers=["ns1.example.com", "ns2.example.com"],
        status=["clientTransferProhibited"],
        registrant_org="예시 조직",
        registrant_country="KR",
        dnssec="unsigned",
        privacy_protected=True,
    )

    lines = [
        f"도메인 분석: {domain}",
        "=" * 50,
        f"등록 기관:      {record.registrar}",
        f"생성일:         {record.creation_date}",
        f"만료일:         {record.expiration_date}",
        f"도메인 나이:    {record.domain_age_days}일",
        f"만료까지:       {record.days_until_expiry}일",
        f"네임서버:       {', '.join(record.name_servers)}",
        f"DNSSEC:         {record.dnssec}",
        f"개인 정보 보호: {'예' if record.privacy_protected else '아니요'}",
        "",
        "보안 관찰:",
    ]
    for obs in record.security_observations():
        lines.append(f"  [!] {obs}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze_domain("example.com"))
```

---

## 8. 메타데이터 추출

웹사이트에 게시된 문서에는 내부 정보를 드러내는 메타데이터가 포함되는 경우가 많습니다: 작성자 이름, 소프트웨어 버전, 파일 경로, 프린터 이름, 심지어 이미지의 GPS 좌표까지.

### 8.1 일반적인 메타데이터 소스

| 파일 유형 | 사용 가능한 메타데이터 |
|-----------|----------------------|
| PDF | 작성자, 생성 도구, 수정 날짜, 내장 폰트 |
| DOCX/XLSX | 작성자, 회사, 수정 횟수, 템플릿 경로 |
| JPEG/PNG | EXIF 데이터: 카메라 모델, GPS 좌표, 타임스탬프 |
| EXE/DLL | 컴파일러 버전, 디버그 경로, 디지털 서명 |

### 8.2 메타데이터 추출 도구

```python
"""
허가된 정찰을 위한 문서 메타데이터 추출기.

대상 조직에 대한 내부 정보를 드러낼 수 있는
문서에서 숨겨진 메타데이터를 추출합니다.
"""

import struct
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional


@dataclass
class DocumentMetadata:
    """문서에서 추출된 메타데이터."""
    filename: str
    file_type: str
    author: str = ""
    creator_tool: str = ""
    creation_date: str = ""
    modification_date: str = ""
    company: str = ""
    title: str = ""
    subject: str = ""
    keywords: list[str] = field(default_factory=list)
    custom_properties: dict = field(default_factory=dict)
    internal_paths: list[str] = field(default_factory=list)

    def security_findings(self) -> list[str]:
        """보안 관련 메타데이터를 식별합니다."""
        findings = []
        if self.author:
            findings.append(f"작성자 이름 공개: {self.author}")
        if self.company:
            findings.append(f"메타데이터의 회사 이름: {self.company}")
        if self.creator_tool:
            findings.append(f"사용된 소프트웨어: {self.creator_tool}")
        if self.internal_paths:
            findings.append(
                f"내부 파일 경로 노출: {', '.join(self.internal_paths[:5])}"
            )
        return findings


def extract_docx_metadata(filepath: str) -> Optional[DocumentMetadata]:
    """
    DOCX 파일에서 메타데이터를 추출합니다.

    DOCX 파일은 XML 파일을 포함하는 ZIP 아카이브입니다.
    핵심 속성은 docProps/core.xml에 있습니다.
    """
    metadata = DocumentMetadata(
        filename=Path(filepath).name,
        file_type="DOCX",
    )

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            # 핵심 속성
            if "docProps/core.xml" in zf.namelist():
                core = ET.fromstring(zf.read("docProps/core.xml"))
                ns = {
                    "dc": "http://purl.org/dc/elements/1.1/",
                    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
                    "dcterms": "http://purl.org/dc/terms/",
                }
                creator = core.find("dc:creator", ns)
                if creator is not None and creator.text:
                    metadata.author = creator.text

                title = core.find("dc:title", ns)
                if title is not None and title.text:
                    metadata.title = title.text

                modified = core.find("dcterms:modified", ns)
                if modified is not None and modified.text:
                    metadata.modification_date = modified.text

            # 앱 속성 (소프트웨어 정보)
            if "docProps/app.xml" in zf.namelist():
                app = ET.fromstring(zf.read("docProps/app.xml"))
                ns_app = {
                    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
                }
                company = app.find("ep:Company", ns_app)
                if company is not None and company.text:
                    metadata.company = company.text

                app_name = app.find("ep:Application", ns_app)
                if app_name is not None and app_name.text:
                    metadata.creator_tool = app_name.text

    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return None

    return metadata


def extract_pdf_metadata(filepath: str) -> Optional[DocumentMetadata]:
    """
    PDF 파일에서 메타데이터를 추출합니다 (기본 추출).

    포괄적인 PDF 메타데이터를 위해서는 PyPDF2 또는 pikepdf를 사용하세요.
    이는 PDF 트레일러에서의 수동 추출을 시연합니다.
    """
    metadata = DocumentMetadata(
        filename=Path(filepath).name,
        file_type="PDF",
    )

    try:
        with open(filepath, "rb") as f:
            content = f.read()

        # PDF에서 /Info 딕셔너리 찾기
        # 이는 단순화된 파서입니다 — 실제 PDF는 복잡한 구조
        text = content.decode("latin-1", errors="ignore")

        # 일반적인 메타데이터 필드 추출
        import re
        patterns = {
            "author": r"/Author\s*\(([^)]*)\)",
            "creator_tool": r"/Creator\s*\(([^)]*)\)",
            "title": r"/Title\s*\(([^)]*)\)",
            "subject": r"/Subject\s*\(([^)]*)\)",
            "creation_date": r"/CreationDate\s*\(([^)]*)\)",
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                setattr(metadata, field_name, match.group(1))

    except (OSError, UnicodeDecodeError):
        return None

    return metadata


if __name__ == "__main__":
    print("문서 메타데이터 추출기")
    print("=" * 40)
    print("\n지원 형식: DOCX, PDF")
    print("사용법:")
    print("  metadata = extract_docx_metadata('report.docx')")
    print("  findings = metadata.security_findings()")
    print("\n보안 관련 메타데이터 예시:")
    print("  - 작성자 이름 (직원 식별)")
    print("  - 회사 이름 (조직 확인)")
    print("  - 소프트웨어 버전 (공격 표면 매핑)")
    print("  - 내부 파일 경로 (네트워크 구조)")
    print("  - 이미지의 GPS 좌표 (물리적 위치)")
```

---

## 9. 소셜 미디어 및 직원 OSINT

소셜 미디어 및 전문 네트워킹 사이트는 조직의 직원, 기술, 내부 프로세스에 대한 풍부한 인텔리전스를 제공합니다.

### 9.1 LinkedIn 인텔리전스

LinkedIn은 다음 목적에 특히 유용합니다:
- **직원 열거**: 직원 목록과 역할 구축
- **기술 스택**: 채용 공고가 사용되는 기술을 드러냄
- **조직 구조**: 경영진 계층 및 팀 규모
- **이메일 패턴**: 도메인 정보와 결합하여 이메일 주소 추측
- **보안 태세 단서**: 보안 역할에 대한 채용 공고가 우선순위를 나타냄

### 9.2 GitHub 및 코드 저장소 OSINT

개발자는 공개 저장소에 민감한 정보를 자주 유출합니다:

- 커밋 이력의 API 키 및 토큰
- 설정 파일의 내부 URL 및 IP 주소
- 환경 파일의 데이터베이스 크리덴셜
- CI/CD 설정의 인프라 세부 사항
- 커밋 로그의 이메일 주소 및 내부 사용자명

---

## 10. 자동화된 정찰 프레임워크

```python
"""
자동화된 정찰 프레임워크.

여러 정찰 모듈을 단일 파이프라인으로 조율하고
구조화된 출력을 생성합니다.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional


@dataclass
class ReconModule:
    """플러그 가능한 정찰 모듈."""
    name: str
    description: str
    recon_type: str  # passive, active
    function: Optional[Callable] = None
    enabled: bool = True


@dataclass
class ReconFinding:
    """정찰에서 나온 단일 발견사항."""
    module: str
    finding_type: str
    value: str
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ReconReport:
    """완전한 정찰 보고서."""
    target: str
    start_time: str = ""
    end_time: str = ""
    findings: list[ReconFinding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_finding(self, finding: ReconFinding) -> None:
        self.findings.append(finding)

    def generate_summary(self) -> dict:
        """모든 발견사항의 요약을 생성합니다."""
        self.summary = {
            "total_findings": len(self.findings),
            "by_module": {},
            "by_type": {},
        }
        for f in self.findings:
            self.summary["by_module"][f.module] = (
                self.summary["by_module"].get(f.module, 0) + 1
            )
            self.summary["by_type"][f.finding_type] = (
                self.summary["by_type"].get(f.finding_type, 0) + 1
            )
        return self.summary

    def to_json(self) -> str:
        """보고서를 JSON으로 내보냅니다."""
        return json.dumps(asdict(self), indent=2)

    def to_text(self) -> str:
        """보고서를 사람이 읽을 수 있는 텍스트로 내보냅니다."""
        lines = [
            f"정찰 보고서: {self.target}",
            f"기간: {self.start_time} ~ {self.end_time}",
            "=" * 60,
            "",
        ]
        self.generate_summary()
        lines.append(f"총 발견사항: {self.summary['total_findings']}")
        lines.append("\n모듈별 발견사항:")
        for mod, count in self.summary["by_module"].items():
            lines.append(f"  {mod}: {count}")
        lines.append("\n유형별 발견사항:")
        for typ, count in self.summary["by_type"].items():
            lines.append(f"  {typ}: {count}")

        lines.append("\n\n상세 발견사항:")
        lines.append("-" * 60)
        for i, f in enumerate(self.findings, 1):
            lines.append(
                f"\n[{i}] [{f.module}] {f.finding_type}: {f.value}"
            )
            if f.metadata:
                for k, v in f.metadata.items():
                    lines.append(f"     {k}: {v}")
        return "\n".join(lines)


class ReconFramework:
    """
    여러 정찰 모듈을 조율합니다.

    사용법:
        framework = ReconFramework("target.com")
        framework.register_module(...)
        report = framework.run()
    """

    def __init__(self, target: str):
        self.target = target
        self.modules: list[ReconModule] = []
        self.report = ReconReport(target=target)

    def register_module(self, module: ReconModule) -> None:
        self.modules.append(module)

    def run(self, passive_only: bool = False) -> ReconReport:
        self.report.start_time = datetime.utcnow().isoformat()

        for module in self.modules:
            if not module.enabled:
                continue
            if passive_only and module.recon_type == "active":
                print(f"  [건너뜀] {module.name} (능동 모듈, 수동 전용 모드)")
                continue

            print(f"  [실행]  {module.name}...")
            if module.function:
                try:
                    findings = module.function(self.target)
                    for f in findings:
                        self.report.add_finding(f)
                    print(f"  [완료]   {len(findings)}개 발견")
                except Exception as e:
                    print(f"  [오류]  {module.name}: {e}")

        self.report.end_time = datetime.utcnow().isoformat()
        self.report.generate_summary()
        return self.report


# 예시 모듈 구현
def dns_recon_module(target: str) -> list[ReconFinding]:
    """간단한 DNS 정찰 모듈."""
    import socket
    findings = []
    try:
        ips = socket.getaddrinfo(target, None, socket.AF_INET)
        for ip_info in ips:
            findings.append(ReconFinding(
                module="dns_recon",
                finding_type="ip_address",
                value=ip_info[4][0],
                confidence=1.0,
                metadata={"record_type": "A"},
            ))
    except socket.gaierror:
        pass
    return findings


if __name__ == "__main__":
    framework = ReconFramework("example.com")
    framework.register_module(ReconModule(
        name="DNS 정찰",
        description="대상의 DNS 레코드 해결",
        recon_type="passive",
        function=dns_recon_module,
    ))

    print("정찰 시작...\n")
    report = framework.run()
    print("\n" + report.to_text())
```

---

## 11. 대응 방안 및 탐지

정찰을 탐지하고 방지하는 방법을 이해하면 공격자(더 은밀하게)와 방어자(정찰을 조기에 포착) 모두에게 도움이 됩니다.

### 11.1 방어 조치

| 정찰 기법 | 대응 방안 |
|-----------|----------|
| DNS 열거 | 공개 DNS 레코드 최소화, split-horizon DNS |
| 서브도메인 발견 | 와일드카드 DNS로 서브도메인 숨기기, CT 모니터링 |
| 구글 도킹 | robots.txt, 인덱싱된 민감 콘텐츠 제거 |
| Shodan/Censys | 노출 서비스 최소화, 방화벽 사용 |
| 메타데이터 유출 | 공개 문서에서 메타데이터 제거 |
| 소셜 미디어 OSINT | 직원 보안 인식 교육 |
| GitHub 유출 | 커밋 전 훅, 시크릿 스캐닝 |

### 11.2 탐지 지표

- 단일 소스에서 대량의 DNS 쿼리
- 방화벽 로그의 순차적 포트 스캐닝 패턴
- 접근 로그의 비정상적인 웹 크롤링 패턴
- 여러 번의 인증 실패 (크리덴셜 테스트)
- 디렉토리 열거 패턴 (순차적 404 오류)

---

## 12. 연습문제

1. **수동 정찰**: 수동 기법만 사용하여 CTF 연습 플랫폼의 대상 도메인에서 최대한 많은 정보를 수집하세요.

2. **DNS 심층 분석**: dnspython을 사용하여 도메인의 모든 레코드 유형을 쿼리하고 형식화된 보고서를 생성하는 Python 스크립트를 작성하세요.

3. **서브도메인 경주**: 연습 대상에 대해 세 가지 서브도메인 열거 도구(subfinder, amass, crt.sh)를 사용하세요. 커버리지와 속도를 비교하세요.

4. **구글 도크 감사**: 소유한 도메인에 대해 구글 도크 생성기를 실행하세요. 발견사항을 문서화하고 노출된 정보를 수정하세요.

5. **메타데이터 감사**: 정부 웹사이트에서 공개 PDF 10개를 다운로드하세요. 메타데이터를 추출하고 드러난 내부 정보를 분석하세요.

6. **정찰 자동화**: ReconFramework를 다양한 정찰 기법을 다루는 5개 이상의 모듈로 확장하세요. 포괄적인 보고서를 생성하세요.

---

## 13. 요약

정찰은 효과적인 침투 테스트의 기초입니다:

- **수동 정찰**은 대상을 건드리지 않고 정보 수집 — OSINT, CT 로그, 검색 엔진
- **능동 정찰**은 대상 시스템과 직접 상호작용 — DNS 쿼리, 포트 스캔, 웹 크롤링
- **DNS 열거**는 인프라 공개: 서브도메인, 메일 서버, 클라우드 제공자
- **구글 도킹**은 실수로 노출된 파일, 크리덴셜, 오류 메시지 발견
- **Shodan/Censys**는 인터넷에 노출된 자산에 대한 사전 스캔된 데이터 제공
- **메타데이터**는 문서에서 조직에 대한 내부 정보 유출
- **자동화**는 활동 전반에 걸쳐 일관되고 포괄적인 정찰 가능

정찰의 품질이 후속 익스플로잇 단계의 성공을 직접적으로 결정합니다.

---

## 14. 참고 자료

- OSINT Framework: https://osintframework.com/
- Shodan: https://www.shodan.io/
- crt.sh (Certificate Transparency): https://crt.sh/
- Google Hacking Database (GHDB): https://www.exploit-db.com/google-hacking-database
- theHarvester: https://github.com/laramies/theHarvester
- Amass: https://github.com/owasp-amass/amass
- Subfinder: https://github.com/projectdiscovery/subfinder
- Recon-ng: https://github.com/lanmaster53/recon-ng
