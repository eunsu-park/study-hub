# 네트워크 스캐닝

**이전**: [02. 정찰](./02_Reconnaissance.md) | **다음**: [04. 취약점 평가](./04_Vulnerability_Assessment.md)

---

네트워크 스캐닝은 대상 네트워크에서 호스트, 열린 포트, 실행 중인 서비스 및 운영 체제를 체계적으로 발견하는 프로세스입니다. 정찰이 공개 정보를 수집하는 반면, 네트워크 스캐닝은 대상 인프라를 능동적으로 프로브하여 공격 표면의 상세한 맵을 구축합니다.

> **중요**: 허가 없는 스캐닝은 범죄입니다. 서면 허가를 받은 대상만 스캐닝하세요.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ARP, ICMP, TCP 기법을 사용한 호스트 발견
2. TCP 3-way 핸드셰이크 및 SYN 스캔 원리 이해
3. 포괄적 네트워크 스캐닝을 위한 Nmap 사용
4. 열린 포트의 서비스 및 버전 탐지
5. 원격 OS 핑거프린팅
6. 방화벽 및 IDS 우회
7. Python을 이용한 스캐닝 자동화
8. 침투 테스트 계획을 위한 스캔 결과 해석

---

## 목차

1. [호스트 발견 기법](#1-호스트-발견-기법)
2. [TCP 및 UDP 포트 스캐닝](#2-tcp-및-udp-포트-스캐닝)
3. [Nmap 심층 분석](#3-nmap-심층-분석)
4. [서비스 및 버전 탐지](#4-서비스-및-버전-탐지)
5. [OS 핑거프린팅](#5-os-핑거프린팅)
6. [방화벽 및 IDS 우회](#6-방화벽-및-ids-우회)
7. [Masscan을 이용한 고속 스캔](#7-masscan을-이용한-고속-스캔)
8. [네트워크 매핑 및 시각화](#8-네트워크-매핑-및-시각화)
9. [Python을 이용한 스캐닝 자동화](#9-python을-이용한-스캐닝-자동화)
10. [대응 방안 및 탐지](#10-대응-방안-및-탐지)
11. [연습문제](#11-연습문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. 호스트 발견 기법

호스트 발견은 대상 범위에서 어떤 IP 주소에 활성 호스트가 있는지 결정합니다. 이것이 첫 번째 단계입니다 — 포트를 스캔하기 전에 어떤 호스트가 존재하는지 알아야 합니다.

### 1.1 ARP 발견 (레이어 2)

ARP 발견은 로컬 네트워크에서 가장 빠르고 신뢰할 수 있는 방법입니다. ARP는 IP 레이어 아래에서 작동하기 때문에 호스트 방화벽으로 차단할 수 없습니다.

```bash
# ARP 스캔 — 로컬 네트워크에서 가장 신뢰할 수 있음
nmap -sn -PR 192.168.1.0/24

# arp-scan 사용
arp-scan --localnet

# Nmap ARP 핑 (로컬 네트워크 기본값)
nmap -sn 192.168.1.0/24
```

### 1.2 ICMP 발견 (레이어 3)

```bash
# ICMP 에코 (전통적인 핑 스윕)
nmap -sn -PE 10.0.0.0/24

# ICMP 타임스탬프 (에코 차단 방화벽 우회)
nmap -sn -PP 10.0.0.0/24

# ICMP 주소 마스크
nmap -sn -PM 10.0.0.0/24
```

### 1.3 TCP/UDP 발견 (레이어 4)

```bash
# 일반적인 포트에 TCP SYN 핑
nmap -sn -PS22,80,443 10.0.0.0/24

# TCP ACK 핑 (스테이트리스 방화벽 우회)
nmap -sn -PA80,443 10.0.0.0/24

# UDP 핑
nmap -sn -PU53,161 10.0.0.0/24
```

```python
"""
허가된 네트워크 평가를 위한 호스트 발견 모듈.

소켓 수준에서 TCP 프로브를 사용하여
네트워크 스캐너가 활성 호스트를 발견하는 방법을 시연합니다.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import ipaddress


@dataclass
class HostResult:
    """호스트 발견 프로브의 결과."""
    ip: str
    is_alive: bool
    method: str
    response_time_ms: Optional[float] = None


def tcp_probe(target: str, port: int, timeout: float = 1.5) -> bool:
    """TCP 연결 시도를 사용하여 호스트가 활성 상태인지 확인합니다."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((target, port))
        sock.close()
        return result == 0 or result == 111  # 열림 또는 RST = 활성
    except (socket.timeout, OSError):
        return False


def discover_hosts(
    network: str,
    ports: list[int] = None,
    max_workers: int = 50,
    timeout: float = 1.5,
) -> list[HostResult]:
    """
    TCP 프로브를 사용하여 CIDR 범위에서 활성 호스트를 발견합니다.

    Args:
        network: CIDR 표기법 (예: "192.168.1.0/24")
        ports: 프로브할 포트 (기본값: [80, 443, 22])
        max_workers: 동시 스레드 수
        timeout: 소켓 타임아웃 (초)
    """
    if ports is None:
        ports = [80, 443, 22]

    net = ipaddress.ip_network(network, strict=False)
    results = []

    def check_host(ip_str: str) -> Optional[HostResult]:
        for port in ports:
            if tcp_probe(ip_str, port, timeout):
                return HostResult(
                    ip=ip_str, is_alive=True,
                    method=f"TCP/{port}",
                )
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_host, str(ip)): str(ip)
            for ip in net.hosts()
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return sorted(results, key=lambda r: ipaddress.ip_address(r.ip))


if __name__ == "__main__":
    print("호스트 발견 모듈")
    print("=" * 40)
    print("허가된 대상에서만 사용하세요.")
    print("\n예시:")
    print("  hosts = discover_hosts('192.168.1.0/24')")
    print("  for h in hosts:")
    print("      print(f'{h.ip} is alive ({h.method})')")
```

---

## 2. TCP 및 UDP 포트 스캐닝

### 2.1 TCP 3-Way 핸드셰이크

스캔 유형을 이해하는 데 TCP 핸드셰이크 이해가 기본입니다:

```
클라이언트          서버
  │──── SYN ──────▶│   1단계: 클라이언트 시작
  │◀── SYN/ACK ───│   2단계: 서버 확인
  │──── ACK ──────▶│   3단계: 연결 수립
```

### 2.2 스캔 유형

**TCP SYN 스캔** (Half-open 스캔): SYN 전송, SYN/ACK(열림) 또는 RST(닫힘) 수신. 핸드셰이크 미완료 — 더 은밀합니다.

**TCP Connect 스캔**: 전체 3-way 핸드셰이크 완료. 더 탐지되기 쉽지만 루트 권한 없이 작동합니다.

**UDP 스캔**: UDP 패킷 전송. 열린 포트는 응답하지 않을 수 있습니다; 닫힌 포트는 ICMP Port Unreachable을 보냅니다.

```python
"""
교육 목적의 TCP 포트 스캐너 구현.

소켓 수준에서 포트 스캐너가 작동하는 방법을 시연합니다.
실제 평가에서는 Nmap 또는 Masscan을 사용하세요.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class PortResult:
    """단일 포트 스캔 결과."""
    port: int
    state: str  # open, closed, filtered
    service: str = ""
    banner: str = ""
    response_time_ms: float = 0.0


SERVICES = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 80: "HTTP", 110: "POP3", 135: "MSRPC",
    139: "NetBIOS", 143: "IMAP", 443: "HTTPS", 445: "SMB",
    993: "IMAPS", 995: "POP3S", 1433: "MSSQL", 3306: "MySQL",
    3389: "RDP", 5432: "PostgreSQL", 5900: "VNC", 6379: "Redis",
    8080: "HTTP-Proxy", 8443: "HTTPS-Alt", 27017: "MongoDB",
}


def scan_port(host: str, port: int, timeout: float = 2.0) -> PortResult:
    """Connect 스캔을 사용하여 단일 TCP 포트를 스캔합니다."""
    start = time.monotonic()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        elapsed = (time.monotonic() - start) * 1000

        if result == 0:
            banner = ""
            try:
                sock.send(b"\r\n")
                banner = sock.recv(1024).decode("utf-8", errors="ignore").strip()
            except (socket.timeout, OSError):
                pass
            sock.close()
            return PortResult(
                port=port, state="open",
                service=SERVICES.get(port, "unknown"),
                banner=banner[:200],
                response_time_ms=round(elapsed, 2),
            )
        sock.close()
        return PortResult(port=port, state="closed")
    except socket.timeout:
        return PortResult(port=port, state="filtered")
    except OSError:
        return PortResult(port=port, state="error")


def scan_host(
    host: str,
    ports: list[int],
    max_workers: int = 100,
) -> list[PortResult]:
    """호스트의 여러 포트를 동시에 스캔합니다."""
    open_ports = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_port, host, port): port
            for port in ports
        }
        for future in as_completed(futures):
            result = future.result()
            if result.state == "open":
                open_ports.append(result)
    return sorted(open_ports, key=lambda r: r.port)


def report(host: str, results: list[PortResult]) -> str:
    """형식화된 스캔 보고서를 생성합니다."""
    lines = [
        f"스캔 보고서: {host}",
        f"열린 포트: {len(results)}",
        "=" * 60,
        f"{'포트':>8}  {'상태':8}  {'서비스':15}  배너",
        "-" * 60,
    ]
    for r in results:
        lines.append(
            f"{r.port:>8}  {r.state:8}  {r.service:15}  {r.banner[:40]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("TCP 포트 스캐너 (교육용)")
    print("=" * 40)
    print(f"알려진 서비스: {len(SERVICES)}개")
    print("사용법: results = scan_host('target', [22, 80, 443])")
```

---

## 3. Nmap 심층 분석

Nmap은 업계 표준 네트워크 스캐너입니다. Nmap을 마스터하는 것은 모든 침투 테스터에게 필수적입니다.

### 3.1 필수 Nmap 명령어

```bash
# 기본 SYN 스캔 + 서비스 탐지
nmap -sS -sV -O 10.0.0.1

# 종합 스캔 (SYN + 버전 + 스크립트 + OS)
nmap -sS -sV -sC -O -p- 10.0.0.1

# 빠른 스캔 (상위 100 포트)
nmap -F 10.0.0.1

# 공격적 스캔 (버전, 스크립트, OS, 트레이스루트)
nmap -A 10.0.0.1

# 전체 서브넷 스캔, 모든 형식으로 출력
nmap -sS -sV -O -oA scan_results 10.0.0.0/24

# 특정 포트 스캔
nmap -p 80,443,8080,8443 10.0.0.1

# 전체 65535 포트 스캔
nmap -p- 10.0.0.1

# UDP 스캔 (느리지만 중요)
nmap -sU --top-ports 100 10.0.0.1
```

### 3.2 Nmap Scripting Engine (NSE)

NSE는 강력한 스크립팅 기능으로 Nmap을 확장합니다:

```bash
# 기본 스크립트 실행
nmap -sC 10.0.0.1

# 특정 스크립트 카테고리 실행
nmap --script vuln 10.0.0.1
nmap --script "http-*" 10.0.0.1

# 특정 스크립트 실행
nmap --script http-title 10.0.0.1
nmap --script ssl-heartbleed 10.0.0.1

# 스크립트 카테고리: auth, broadcast, brute, default,
# discovery, dos, exploit, external, fuzzer, intrusive,
# malware, safe, version, vuln
```

### 3.3 출력 형식

```bash
# 일반 출력
nmap -oN output.txt 10.0.0.1

# XML 출력 (파싱용)
nmap -oX output.xml 10.0.0.1

# Grepable 출력
nmap -oG output.gnmap 10.0.0.1

# 모든 형식 동시 출력
nmap -oA output_base 10.0.0.1
```

---

## 4. 서비스 및 버전 탐지

서비스 탐지는 각 열린 포트에서 실행 중인 특정 소프트웨어와 버전을 식별합니다.

```bash
# 버전 탐지 (강도 0-9, 기본값 7)
nmap -sV 10.0.0.1

# 공격적 버전 탐지
nmap -sV --version-intensity 9 10.0.0.1

# 가벼운 버전 탐지 (더 빠름)
nmap -sV --version-light 10.0.0.1
```

### 4.1 배너 그래빙 (Banner Grabbing)

```python
"""
서비스 배너 그래빙 모듈.

버전 식별을 위해 열린 포트에 연결하여
서비스 배너를 캡처합니다.
"""

import socket
from dataclasses import dataclass


@dataclass
class BannerResult:
    """캡처된 서비스 배너."""
    host: str
    port: int
    banner: str
    service_guess: str = ""


# 프로토콜별 프로브
PROBES = {
    "http": b"HEAD / HTTP/1.1\r\nHost: {host}\r\n\r\n",
    "smtp": b"EHLO test\r\n",
    "ftp": b"",  # FTP는 연결 시 배너 전송
    "ssh": b"",  # SSH는 연결 시 배너 전송
    "generic": b"\r\n\r\n",
}


def grab_banner(
    host: str,
    port: int,
    timeout: float = 5.0,
    probe: str = "generic",
) -> BannerResult:
    """열린 포트에서 서비스 배너를 가져옵니다."""
    result = BannerResult(host=host, port=port, banner="")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # 일부 서비스는 즉시 배너를 전송
        try:
            initial = sock.recv(1024)
            if initial:
                result.banner = initial.decode("utf-8", errors="replace").strip()
        except socket.timeout:
            pass

        # 초기 배너가 없으면 프로브 전송
        if not result.banner and probe in PROBES:
            probe_data = PROBES[probe]
            if b"{host}" in probe_data:
                probe_data = probe_data.replace(b"{host}", host.encode())
            if probe_data:
                sock.send(probe_data)
                try:
                    response = sock.recv(4096)
                    result.banner = response.decode("utf-8", errors="replace").strip()
                except socket.timeout:
                    pass

        sock.close()

        # 배너에서 서비스 추측
        banner_lower = result.banner.lower()
        if "ssh" in banner_lower:
            result.service_guess = "SSH"
        elif "http" in banner_lower:
            result.service_guess = "HTTP"
        elif "smtp" in banner_lower:
            result.service_guess = "SMTP"
        elif "ftp" in banner_lower:
            result.service_guess = "FTP"
        elif "mysql" in banner_lower:
            result.service_guess = "MySQL"

    except (socket.timeout, ConnectionRefusedError, OSError):
        pass

    return result


if __name__ == "__main__":
    print("배너 그래빙 모듈")
    print("=" * 40)
    print("사용법: result = grab_banner('target', 22)")
    print("허가된 대상에서만 사용하세요.")
```

---

## 5. OS 핑거프린팅

OS 핑거프린팅은 네트워크 프로토콜 동작을 분석하여 대상 호스트의 운영 체제를 결정합니다.

### 5.1 능동 OS 핑거프린팅

```bash
# Nmap OS 탐지 (루트 권한 필요)
nmap -O 10.0.0.1

# 공격적 OS 탐지
nmap -O --osscan-guess 10.0.0.1

# 버전 탐지와 결합
nmap -O -sV 10.0.0.1
```

### 5.2 수동 OS 핑거프린팅

수동 핑거프린팅은 특수 프로브를 전송하지 않고 일반 트래픽을 분석합니다:

- **TTL 값**: Linux 기본값 64, Windows 기본값 128, Cisco 기본값 255
- **TCP 윈도우 크기**: OS 및 버전에 따라 다름
- **TCP 옵션**: 순서와 값이 구현에 따라 다름
- **MSS 값**: MTU 및 OS 특성 드러냄

---

## 6. 방화벽 및 IDS 우회

### 6.1 Nmap 우회 기법

```bash
# 패킷 분할
nmap -f 10.0.0.1

# 특정 MTU 설정
nmap --mtu 24 10.0.0.1

# 디코이 주소 사용
nmap -D RND:10 10.0.0.1

# 신뢰할 수 있는 포트에서 소스 포트 위장
nmap --source-port 53 10.0.0.1

# 대상 순서 무작위화
nmap --randomize-hosts 10.0.0.0/24

# 느린 스캔 타이밍
nmap -T0 10.0.0.1  # 편집증적 (프로브 사이 5분)
nmap -T1 10.0.0.1  # 은밀 (프로브 사이 15초)

# 임의 데이터 추가
nmap --data-length 25 10.0.0.1
```

### 6.2 타이밍 템플릿

| 템플릿 | 이름 | 사용 사례 |
|--------|------|----------|
| `-T0` | Paranoid (편집증적) | IDS 우회, 매우 느림 |
| `-T1` | Sneaky (은밀) | IDS 우회, 느림 |
| `-T2` | Polite (공손) | 대역폭 감소 |
| `-T3` | Normal (기본) | 기본값 |
| `-T4` | Aggressive (공격적) | 빠름, 안정적 네트워크 |
| `-T5` | Insane (미친) | 매우 빠름, 결과 누락 가능 |

---

## 7. Masscan을 이용한 고속 스캔

Masscan은 6분 이내에 인터넷 전체를 스캔할 수 있습니다. 비동기 SYN 스캐닝을 사용하는 가장 빠른 포트 스캐너입니다.

```bash
# 일반적인 포트의 빠른 스캔
masscan 10.0.0.0/24 -p 80,443,8080 --rate 1000

# 고속으로 전체 포트 스캔
masscan 10.0.0.0/24 -p 1-65535 --rate 10000

# Nmap 호환 XML 출력
masscan 10.0.0.0/24 -p 1-65535 --rate 5000 -oX masscan_output.xml

# 그런 다음 발견된 포트에 대해 Nmap으로 상세 서비스 탐지
nmap -sV -sC -p 80,443,8080 -iL masscan_hosts.txt
```

---

## 8. 네트워크 매핑 및 시각화

```python
"""
네트워크 스캔 결과 파서 및 분석기.

Nmap XML 출력을 파싱하여 침투 테스트 계획을 위한
구조화된 보고서를 생성합니다.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Service:
    """포트에서 발견된 네트워크 서비스."""
    port: int
    protocol: str
    state: str
    service_name: str = ""
    product: str = ""
    version: str = ""
    extra_info: str = ""

    @property
    def display(self) -> str:
        parts = [f"{self.port}/{self.protocol}"]
        if self.service_name:
            parts.append(self.service_name)
        if self.product:
            parts.append(self.product)
        if self.version:
            parts.append(self.version)
        return " | ".join(parts)


@dataclass
class Host:
    """발견된 네트워크 호스트."""
    ip: str
    hostname: str = ""
    os_guess: str = ""
    state: str = "up"
    services: list[Service] = field(default_factory=list)

    @property
    def open_ports(self) -> list[int]:
        return [s.port for s in self.services if s.state == "open"]


def parse_nmap_xml(xml_path: str) -> list[Host]:
    """Nmap XML 출력을 구조화된 Host 객체로 파싱합니다."""
    hosts = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for host_elem in root.findall("host"):
            # IP 주소 가져오기
            addr_elem = host_elem.find("address[@addrtype='ipv4']")
            if addr_elem is None:
                continue

            host = Host(ip=addr_elem.get("addr", ""))

            # 호스트명
            hostname_elem = host_elem.find(".//hostname")
            if hostname_elem is not None:
                host.hostname = hostname_elem.get("name", "")

            # OS 탐지
            osmatch = host_elem.find(".//osmatch")
            if osmatch is not None:
                host.os_guess = osmatch.get("name", "")

            # 포트/서비스
            for port_elem in host_elem.findall(".//port"):
                state_elem = port_elem.find("state")
                service_elem = port_elem.find("service")

                service = Service(
                    port=int(port_elem.get("portid", 0)),
                    protocol=port_elem.get("protocol", "tcp"),
                    state=state_elem.get("state", "") if state_elem is not None else "",
                )

                if service_elem is not None:
                    service.service_name = service_elem.get("name", "")
                    service.product = service_elem.get("product", "")
                    service.version = service_elem.get("version", "")

                if service.state == "open":
                    host.services.append(service)

            if host.services:
                hosts.append(host)

    except (ET.ParseError, FileNotFoundError) as e:
        print(f"XML 파싱 오류: {e}")

    return hosts


def generate_target_report(hosts: list[Host]) -> str:
    """스캔 결과에서 침투 테스트 대상 보고서를 생성합니다."""
    lines = [
        "네트워크 스캔 분석 보고서",
        "=" * 60,
        f"발견된 호스트 총계: {len(hosts)}",
        f"열린 포트 총계: {sum(len(h.services) for h in hosts)}",
        "",
    ]

    # 흥미로운 서비스별 그룹화
    web_servers = []
    databases = []
    remote_access = []

    for host in hosts:
        for svc in host.services:
            if svc.port in (80, 443, 8080, 8443) or "http" in svc.service_name:
                web_servers.append((host, svc))
            elif svc.port in (3306, 5432, 1433, 27017, 6379):
                databases.append((host, svc))
            elif svc.port in (22, 3389, 5900, 23):
                remote_access.append((host, svc))

    lines.append(f"웹 서버: {len(web_servers)}")
    lines.append(f"데이터베이스: {len(databases)}")
    lines.append(f"원격 접근: {len(remote_access)}")
    lines.append("")

    for host in hosts:
        lines.append(f"\n--- {host.ip} ({host.hostname or '호스트명 없음'}) ---")
        if host.os_guess:
            lines.append(f"  OS: {host.os_guess}")
        for svc in host.services:
            lines.append(f"  {svc.display}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Nmap XML 파서")
    print("=" * 40)
    print("사용법:")
    print("  1. 실행: nmap -sV -oX scan.xml target")
    print("  2. 파싱: hosts = parse_nmap_xml('scan.xml')")
    print("  3. 보고: print(generate_target_report(hosts))")
```

---

## 9. Python을 이용한 스캐닝 자동화

```python
"""
여러 도구를 결합하는 자동화된 스캐닝 파이프라인.

호스트 발견, 포트 스캐닝, 서비스 탐지를
단일 워크플로우로 조율합니다.
"""

import json
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ScanConfig:
    """자동화된 스캔 설정."""
    target: str
    output_dir: str = "./scan_results"
    scan_type: str = "standard"  # quick, standard, comprehensive
    max_rate: int = 1000
    timing: int = 4  # Nmap 타이밍 템플릿 (0-5)

    @property
    def nmap_args(self) -> list[str]:
        """스캔 유형에 따른 Nmap 인수를 생성합니다."""
        base = ["-sS", "-sV", f"-T{self.timing}"]
        if self.scan_type == "quick":
            base.extend(["-F", "--top-ports", "100"])
        elif self.scan_type == "standard":
            base.extend(["--top-ports", "1000", "-sC"])
        elif self.scan_type == "comprehensive":
            base.extend(["-p-", "-sC", "-O", "--script", "vuln"])
        return base


@dataclass
class ScanResult:
    """완전한 스캔 결과."""
    config: ScanConfig
    start_time: str = ""
    end_time: str = ""
    hosts_discovered: int = 0
    total_open_ports: int = 0
    findings: list[dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class ScanPipeline:
    """자동화된 스캐닝 파이프라인."""

    def __init__(self, config: ScanConfig):
        self.config = config
        self.result = ScanResult(config=config)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def check_tools(self) -> dict[str, bool]:
        """필수 도구가 설치되어 있는지 확인합니다."""
        tools = ["nmap", "masscan"]
        return {t: shutil.which(t) is not None for t in tools}

    def run_nmap(self) -> Optional[str]:
        """Nmap 스캔을 실행합니다."""
        output_base = f"{self.config.output_dir}/nmap_scan"
        cmd = [
            "nmap",
            *self.config.nmap_args,
            "-oA", output_base,
            self.config.target,
        ]
        print(f"실행 중: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            return f"{output_base}.xml"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Nmap 오류: {e}")
            return None

    def run(self) -> ScanResult:
        """전체 스캐닝 파이프라인을 실행합니다."""
        self.result.start_time = datetime.utcnow().isoformat()

        # 도구 확인
        tools = self.check_tools()
        missing = [t for t, available in tools.items() if not available]
        if "nmap" in missing:
            print("[오류] Nmap이 필요하지만 설치되어 있지 않습니다")
            return self.result

        # Nmap 실행
        print(f"\n[*] {self.config.target}에 대한 {self.config.scan_type} 스캔 시작")
        xml_path = self.run_nmap()

        self.result.end_time = datetime.utcnow().isoformat()

        # 결과 저장
        result_path = f"{self.config.output_dir}/scan_summary.json"
        self.result.save(result_path)
        print(f"\n[*] 결과가 {result_path}에 저장되었습니다")

        return self.result


if __name__ == "__main__":
    print("자동화된 스캐닝 파이프라인")
    print("=" * 40)
    print("사용법:")
    print("  config = ScanConfig(target='10.0.0.0/24', scan_type='standard')")
    print("  pipeline = ScanPipeline(config)")
    print("  result = pipeline.run()")
    print("\n허가된 대상에서만 사용하세요.")
```

---

## 10. 대응 방안 및 탐지

### 10.1 포트 스캔 탐지

| 스캔 유형 | 탐지 지표 |
|-----------|----------|
| SYN 스캔 | 핸드셰이크 미완료 SYN 패킷 다수 |
| Connect 스캔 | 다수의 단기 연결 |
| UDP 스캔 | ICMP 도달 불가 메시지 |
| Masscan | 단일 소스에서 고속 SYN 패킷 |
| OS 핑거프린팅 | 비정상적인 TCP 플래그 조합 |

### 10.2 방어 전략

- **방화벽 규칙**: 불필요한 인바운드 포트 차단
- **IDS/IPS**: 스캔 탐지 규칙 설정 (Snort, Suricata)
- **속도 제한**: 소스 IP당 연결 속도 제한
- **포트 노킹**: 포트 열기 전에 특정 시퀀스 요구
- **허니팟**: 스캐닝 탐지를 위한 미끼 서비스 배포

---

## 11. 연습문제

1. **호스트 발견**: 5개 이상의 VM이 있는 랩 네트워크를 설정하고 세 가지 다른 호스트 발견 방법을 사용하세요. 결과를 비교하세요.

2. **포트 스캐닝**: Metasploitable VM에 SYN, Connect, UDP 스캔을 수행하세요. 결과와 탐지의 차이를 문서화하세요.

3. **Nmap 마스터리**: 랩 대상에 대해 포괄적인 Nmap 스캔을 수행하세요. NSE 스크립트를 사용하여 추가 정보를 수집하세요.

4. **우회 테스트**: 다양한 우회 기법을 사용하여 방화벽을 통해 대상을 스캔하세요. 어떤 방법이 방화벽을 성공적으로 우회하는지 테스트하세요.

5. **자동화**: 초기 포트 발견을 위한 Masscan과 서비스 탐지를 위한 Nmap을 포함하도록 ScanPipeline 클래스를 확장하세요.

6. **분석**: Nmap XML 출력 파일을 파싱하고 침투 테스트를 위한 우선순위가 지정된 대상 목록을 만드세요.

---

## 12. 요약

네트워크 스캐닝은 정찰을 기반으로 대상의 공격 표면에 대한 상세한 맵을 만듭니다:

- **호스트 발견**은 ARP, ICMP, TCP 기법으로 활성 시스템 식별
- **포트 스캐닝**은 열린 서비스 공개 — SYN 스캔이 은밀성의 표준
- **Nmap**은 필수 도구 — 옵션과 NSE 스크립트 마스터가 핵심
- **서비스 탐지**는 취약점 매칭을 위한 소프트웨어 버전 식별
- **OS 핑거프린팅**은 표적 익스플로잇을 위한 대상 운영 체제 식별
- **우회 기법**은 허가된 평가 중 방화벽 및 IDS 우회에 도움
- **자동화**는 대규모 네트워크의 일관된 포괄적 스캐닝 가능

---

## 13. 참고 자료

- Nmap 공식 문서: https://nmap.org/book/
- Nmap NSE 스크립트 라이브러리: https://nmap.org/nsedoc/
- Masscan: https://github.com/robertdavidgraham/masscan
- RustScan: https://github.com/RustScan/RustScan
- TCP/IP Illustrated, Volume 1 (Stevens)
- Nmap Network Scanning (Fyodor): https://nmap.org/book/
