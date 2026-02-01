# 네트워크 학습 가이드

## 소개

이 폴더는 컴퓨터 네트워크를 체계적으로 학습하기 위한 자료를 담고 있습니다. OSI 7계층부터 TCP/IP, 라우팅, 보안까지 네트워크 통신의 원리를 이해할 수 있습니다.

**대상 독자**: 개발자, 시스템 관리자, 네트워크 기초를 배우려는 사람

---

## 학습 로드맵

```
[기초]                    [중급]                    [고급]
  │                         │                         │
  ▼                         ▼                         ▼
네트워크 개요 ─────▶ IP 주소 ───────────▶ 라우팅 프로토콜
  │                         │                         │
  ▼                         ▼                         ▼
OSI/TCP-IP ────────▶ TCP/UDP ─────────▶ 네트워크 보안
  │                         │                         │
  ▼                         ▼                         ▼
물리/데이터링크 ───▶ 애플리케이션 계층 ──▶ 실무 도구
```

---

## 선수 지식

- 컴퓨터 기초 (운영체제 개념)
- 이진수와 16진수 이해
- 기본적인 명령줄 사용

---

## 파일 목록

### 네트워크 기초 (01-04)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Network_Fundamentals.md](./01_Network_Fundamentals.md) | ⭐ | 네트워크 정의, LAN/WAN, 토폴로지 |
| [02_OSI_7_Layer_Model.md](./02_OSI_7_Layer_Model.md) | ⭐⭐ | 각 계층 역할, 프로토콜, PDU |
| [03_TCP_IP_Model.md](./03_TCP_IP_Model.md) | ⭐⭐ | TCP/IP 4계층, OSI와 비교 |
| [04_Physical_Layer.md](./04_Physical_Layer.md) | ⭐ | 전송 매체, 신호, 이더넷 케이블 |

### 데이터링크 및 네트워크 계층 (05-09)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [05_Data_Link_Layer.md](./05_Data_Link_Layer.md) | ⭐⭐ | MAC 주소, 프레임, 스위치, ARP |
| [06_IP_Address_Subnetting.md](./06_IP_Address_Subnetting.md) | ⭐⭐ | IPv4, 서브넷 마스크, CIDR |
| [07_Subnetting_Practice.md](./07_Subnetting_Practice.md) | ⭐⭐⭐ | 서브넷 계산, VLSM |
| [08_Routing_Basics.md](./08_Routing_Basics.md) | ⭐⭐⭐ | 라우팅 테이블, 정적/동적 라우팅 |
| [09_Routing_Protocols.md](./09_Routing_Protocols.md) | ⭐⭐⭐ | RIP, OSPF, BGP |

### 전송 계층 (10-11)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [10_TCP_Protocol.md](./10_TCP_Protocol.md) | ⭐⭐⭐ | 3-way handshake, 흐름/혼잡 제어 |
| [11_UDP_and_Ports.md](./11_UDP_and_Ports.md) | ⭐⭐ | UDP 특징, 포트 번호, TCP vs UDP |

### 애플리케이션 계층 (12-14)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [12_DNS.md](./12_DNS.md) | ⭐⭐ | 도메인 구조, DNS 조회, 레코드 |
| [13_HTTP_and_HTTPS.md](./13_HTTP_and_HTTPS.md) | ⭐⭐⭐ | HTTP 메서드, 상태 코드, TLS |
| [14_Other_Application_Protocols.md](./14_Other_Application_Protocols.md) | ⭐⭐ | DHCP, FTP, SMTP, SSH |

### 네트워크 보안 및 실무 (15-17)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [15_Network_Security_Basics.md](./15_Network_Security_Basics.md) | ⭐⭐⭐ | 방화벽, NAT, VPN |
| [16_Security_Threats_Response.md](./16_Security_Threats_Response.md) | ⭐⭐⭐⭐ | 스니핑, 스푸핑, DDoS |
| [17_Practical_Network_Tools.md](./17_Practical_Network_Tools.md) | ⭐⭐⭐ | ping, netstat, tcpdump, Wireshark |

---

## 추천 학습 순서

### 1단계: 네트워크 기초 (1주)
```
01_Network_Fundamentals → 02_OSI_7_Layer_Model → 03_TCP_IP_Model
```

### 2단계: 하위 계층 (1주)
```
04_Physical_Layer → 05_Data_Link_Layer
```

### 3단계: IP와 라우팅 (1~2주)
```
06_IP_Address_Subnetting → 07_Subnetting_Practice → 08_Routing_Basics → 09_Routing_Protocols
```

### 4단계: 전송 계층 (1주)
```
10_TCP_Protocol → 11_UDP_and_Ports
```

### 5단계: 애플리케이션 계층 (1주)
```
12_DNS → 13_HTTP_and_HTTPS → 14_Other_Application_Protocols
```

### 6단계: 보안 및 실무 (1~2주)
```
15_Network_Security_Basics → 16_Security_Threats_Response → 17_Practical_Network_Tools
```

---

## 실습 환경

### 명령줄 도구

```bash
# 네트워크 연결 테스트
ping google.com
traceroute google.com

# 네트워크 정보 확인
ip addr                    # Linux
ifconfig                   # macOS
ipconfig                   # Windows

# 연결 상태 확인
netstat -an
ss -tuln                   # Linux

# DNS 조회
nslookup google.com
dig google.com
```

### 패킷 캡처

```bash
# tcpdump (Linux/macOS)
sudo tcpdump -i eth0 -n

# Wireshark (GUI)
# https://www.wireshark.org/

# tshark (CLI)
tshark -i eth0
```

### 시뮬레이터

- **Cisco Packet Tracer**: 네트워크 시뮬레이션
- **GNS3**: 고급 네트워크 에뮬레이션
- **EVE-NG**: 가상 네트워크 랩

---

## 주요 포트 번호

| 포트 | 프로토콜 | 설명 |
|------|----------|------|
| 20, 21 | FTP | 파일 전송 |
| 22 | SSH | 보안 쉘 |
| 23 | Telnet | 원격 접속 (비암호화) |
| 25 | SMTP | 이메일 전송 |
| 53 | DNS | 도메인 이름 서비스 |
| 67, 68 | DHCP | IP 자동 할당 |
| 80 | HTTP | 웹 |
| 443 | HTTPS | 보안 웹 |
| 3306 | MySQL | 데이터베이스 |
| 5432 | PostgreSQL | 데이터베이스 |

---

## 관련 자료

### 다른 폴더와의 연계

| 폴더 | 관련 내용 |
|------|----------|
| [Linux/](../Linux/00_Overview.md) | 네트워크 설정, 방화벽 |
| [Docker/](../Docker/00_Overview.md) | 컨테이너 네트워킹 |
| [Web_Development/](../Web_Development/00_Overview.md) | HTTP, REST API |

### 외부 자료

- [Computer Networking: A Top-Down Approach](https://gaia.cs.umass.edu/kurose_ross/)
- [RFC 문서](https://www.rfc-editor.org/)
- [Cloudflare Learning Center](https://www.cloudflare.com/learning/)
- [Network+ Certification](https://www.comptia.org/certifications/network)

---

## 학습 팁

1. **계층별 이해**: OSI/TCP-IP 계층을 확실히 이해
2. **실습 중심**: ping, traceroute, Wireshark로 직접 확인
3. **패킷 분석**: Wireshark로 실제 패킷 구조 학습
4. **서브네팅 연습**: 서브넷 계산 문제 많이 풀기
5. **프로토콜 헤더**: 각 프로토콜의 헤더 구조 암기

