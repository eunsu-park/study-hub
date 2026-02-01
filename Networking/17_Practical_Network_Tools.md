# 실무 네트워크 도구

## 개요

네트워크 관리와 트러블슈팅에는 다양한 도구가 필요합니다. 이 장에서는 ping, traceroute, netstat, tcpdump, Wireshark 등 실무에서 자주 사용되는 네트워크 도구의 사용법을 학습합니다.

**난이도**: ⭐⭐⭐

**학습 목표**:
- 기본 네트워크 진단 도구 활용
- 패킷 캡처 및 분석 방법 습득
- DNS 조회 도구 사용법 이해
- 체계적인 네트워크 트러블슈팅 방법론 학습

---

## 목차

1. [ping](#1-ping)
2. [traceroute / tracert](#2-traceroute--tracert)
3. [netstat / ss](#3-netstat--ss)
4. [nslookup / dig](#4-nslookup--dig)
5. [tcpdump](#5-tcpdump)
6. [Wireshark 기초](#6-wireshark-기초)
7. [curl](#7-curl)
8. [네트워크 트러블슈팅 방법론](#8-네트워크-트러블슈팅-방법론)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. ping

### ping 개요

ping은 ICMP(Internet Control Message Protocol)를 사용하여 네트워크 연결을 테스트하는 기본 도구입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ping 동작 원리                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [로컬 호스트]                              [대상 호스트]        │
│       │                                        │                │
│       │──── ICMP Echo Request ───────────────▶│                │
│       │     Type: 8, Code: 0                   │                │
│       │     Sequence: 1                        │                │
│       │                                        │                │
│       │◀─── ICMP Echo Reply ──────────────────│                │
│       │     Type: 0, Code: 0                   │                │
│       │     Sequence: 1                        │                │
│       │                                        │                │
│       └─── RTT (Round-Trip Time) 계산 ─────────┘                │
│                                                                 │
│  측정 항목:                                                     │
│  - 연결 가능 여부                                               │
│  - 왕복 시간 (RTT)                                              │
│  - 패킷 손실률                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ping 기본 사용법

```bash
# 기본 ping (Linux/macOS: 계속, Windows: 4회)
ping google.com

# 횟수 지정
ping -c 4 google.com          # Linux/macOS
ping -n 4 google.com          # Windows

# 간격 지정 (초)
ping -i 0.5 google.com        # 0.5초 간격

# 패킷 크기 지정
ping -s 1000 google.com       # 1000바이트 패킷

# TTL 지정
ping -t 64 google.com         # Linux/macOS
ping -i 64 google.com         # Windows (TTL)

# 타임아웃 지정
ping -W 2 google.com          # Linux: 2초 타임아웃
ping -w 2000 google.com       # Windows: 2000ms

# 응답 없이 전송 (flood ping - root 필요)
sudo ping -f google.com       # 주의: 네트워크 부하
```

### ping 출력 분석

```bash
$ ping -c 4 google.com

PING google.com (142.250.196.110): 56 data bytes
64 bytes from 142.250.196.110: icmp_seq=0 ttl=116 time=31.2 ms
64 bytes from 142.250.196.110: icmp_seq=1 ttl=116 time=29.8 ms
64 bytes from 142.250.196.110: icmp_seq=2 ttl=116 time=30.5 ms
64 bytes from 142.250.196.110: icmp_seq=3 ttl=116 time=32.1 ms

--- google.com ping statistics ---
4 packets transmitted, 4 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 29.8/30.9/32.1/0.8 ms
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    ping 출력 해석                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  64 bytes from 142.250.196.110: icmp_seq=0 ttl=116 time=31.2 ms │
│  │            │                  │         │       │            │
│  │            │                  │         │       └─ RTT       │
│  │            │                  │         └─ TTL (경유 라우터   │
│  │            │                  │            128-116=12개)     │
│  │            │                  └─ 시퀀스 번호                  │
│  │            └─ 응답 IP 주소                                   │
│  └─ 응답 패킷 크기                                               │
│                                                                 │
│  통계:                                                          │
│  - packets transmitted: 전송된 패킷 수                          │
│  - packets received: 수신된 패킷 수                             │
│  - packet loss: 손실률 (0% 정상)                                │
│  - min/avg/max/stddev: RTT 최소/평균/최대/표준편차              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ping 실패 원인

| 메시지 | 원인 |
|--------|------|
| `Destination Host Unreachable` | 대상까지 경로 없음 |
| `Request timed out` | 응답 없음 (방화벽 차단 가능) |
| `Unknown host` | DNS 해석 실패 |
| `TTL expired in transit` | TTL 초과 (루프 가능) |
| `Network is unreachable` | 네트워크 연결 안됨 |

---

## 2. traceroute / tracert

### traceroute 개요

traceroute는 패킷이 목적지까지 이동하는 경로를 추적하는 도구입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    traceroute 동작 원리                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TTL(Time To Live)을 증가시키며 경로 추적                        │
│                                                                 │
│  [출발지]      [라우터1]     [라우터2]     [목적지]              │
│      │            │            │            │                   │
│      │            │            │            │                   │
│      │ TTL=1 ────▶│            │            │                   │
│      │◀─ ICMP ────│            │            │                   │
│      │  Time      │ TTL 0     │            │                   │
│      │  Exceeded  │ (만료)    │            │                   │
│      │            │            │            │                   │
│      │ TTL=2 ────────────────▶│            │                   │
│      │◀─ ICMP ─────────────────│            │                   │
│      │  Time Exceeded         │ TTL 0      │                   │
│      │                        │            │                   │
│      │ TTL=3 ────────────────────────────▶│                   │
│      │◀─ ICMP Echo Reply ─────────────────│                   │
│      │            │            │  목적지 도달                    │
│                                                                 │
│  각 홉에서 라우터 IP와 RTT 측정                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### traceroute 기본 사용법

```bash
# 기본 사용 (Linux/macOS)
traceroute google.com

# Windows
tracert google.com

# 이름 해석 없이 (빠름)
traceroute -n google.com
tracert -d google.com        # Windows

# 최대 홉 수 지정
traceroute -m 20 google.com
tracert -h 20 google.com     # Windows

# ICMP 사용 (일부 라우터는 UDP 차단)
traceroute -I google.com     # Linux

# TCP 사용 (방화벽 우회)
traceroute -T -p 443 google.com

# 프로브 수 지정
traceroute -q 1 google.com   # 각 홉당 1개 프로브
```

### traceroute 출력 분석

```bash
$ traceroute google.com

traceroute to google.com (142.250.196.110), 30 hops max, 60 byte packets
 1  192.168.1.1 (192.168.1.1)  1.234 ms  1.123 ms  1.098 ms
 2  10.0.0.1 (10.0.0.1)  5.432 ms  5.321 ms  5.234 ms
 3  isp-router.net (203.0.113.1)  12.345 ms  12.234 ms  12.123 ms
 4  * * *
 5  peer-link.google.com (72.14.232.84)  25.432 ms  25.321 ms  25.234 ms
 6  142.250.196.110  30.123 ms  29.987 ms  30.234 ms
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    traceroute 출력 해석                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1  192.168.1.1 (192.168.1.1)  1.234 ms  1.123 ms  1.098 ms     │
│  │        │            │         │          │         │         │
│  │        │            │         └──────────┴─────────┘         │
│  │        │            │               3회 프로브 RTT           │
│  │        │            └─ 역방향 DNS (호스트명)                  │
│  │        └─ 라우터 IP 주소                                     │
│  └─ 홉 번호                                                     │
│                                                                 │
│  특수 표시:                                                     │
│  * * *  - 응답 없음 (타임아웃, ICMP 차단)                        │
│  !H     - Host unreachable                                     │
│  !N     - Network unreachable                                  │
│  !P     - Protocol unreachable                                 │
│  !F     - Fragmentation needed                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### mtr (traceroute + ping 결합)

```bash
# 설치
sudo apt install mtr    # Ubuntu/Debian
brew install mtr        # macOS

# 기본 사용 (인터랙티브 모드)
mtr google.com

# 보고서 모드
mtr -r -c 10 google.com

# 출력 예시
HOST: myhost           Loss%   Snt   Last   Avg  Best  Wrst StDev
  1. 192.168.1.1        0.0%    10    1.2   1.3   1.1   1.5   0.1
  2. 10.0.0.1           0.0%    10    5.4   5.5   5.2   6.1   0.3
  3. isp-router.net     0.0%    10   12.3  12.4  12.1  13.0   0.3
  4. ???                       100.0    10    0.0   0.0   0.0   0.0   0.0
  5. google-peer        0.0%    10   25.4  25.6  25.1  26.2   0.4
  6. google.com         0.0%    10   30.1  30.2  29.8  30.5   0.2
```

---

## 3. netstat / ss

### netstat 개요

netstat은 네트워크 연결, 라우팅 테이블, 인터페이스 통계 등을 표시합니다. 현대 Linux에서는 ss가 권장됩니다.

```bash
# 설치 (필요한 경우)
sudo apt install net-tools  # Ubuntu/Debian
```

### netstat 주요 옵션

```bash
# 모든 연결 표시
netstat -a

# TCP 연결만
netstat -t

# UDP 연결만
netstat -u

# 리스닝 포트만
netstat -l

# 숫자로 표시 (이름 해석 안함)
netstat -n

# 프로세스 정보 표시
netstat -p

# 자주 사용하는 조합
netstat -tuln      # TCP/UDP 리스닝 포트
netstat -tulnp     # + 프로세스 정보 (root 필요)
netstat -an        # 모든 연결 (숫자)

# 라우팅 테이블
netstat -r
netstat -rn        # 숫자로

# 인터페이스 통계
netstat -i
```

### ss (Socket Statistics)

ss는 netstat보다 빠르고 더 많은 정보를 제공합니다.

```bash
# 기본 사용 (netstat과 유사한 옵션)
ss -tuln           # TCP/UDP 리스닝 포트
ss -tulnp          # + 프로세스 정보

# TCP 상태별 필터링
ss -t state established
ss -t state time-wait
ss -t state listening

# 포트 필터링
ss -tuln 'sport = :80'
ss -tuln 'dport = :443'

# 프로세스 이름으로 필터링
ss -tulnp | grep nginx

# 상세 정보
ss -tulnpe

# 타이머 정보
ss -to
```

### 연결 상태 확인 예시

```bash
$ ss -tuln

Netid  State   Recv-Q  Send-Q   Local Address:Port   Peer Address:Port
tcp    LISTEN  0       128      0.0.0.0:22           0.0.0.0:*
tcp    LISTEN  0       511      0.0.0.0:80           0.0.0.0:*
tcp    LISTEN  0       511      0.0.0.0:443          0.0.0.0:*
udp    UNCONN  0       0        0.0.0.0:68           0.0.0.0:*
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    ss 출력 해석                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Netid  : 프로토콜 (tcp, udp, unix 등)                          │
│  State  : 연결 상태                                             │
│           - LISTEN: 연결 대기 중                                │
│           - ESTAB: 연결됨                                       │
│           - TIME-WAIT: 연결 종료 대기                           │
│           - CLOSE-WAIT: 상대방 종료 대기                        │
│  Recv-Q : 수신 큐                                               │
│  Send-Q : 송신 큐                                               │
│  Local Address:Port : 로컬 주소:포트                            │
│  Peer Address:Port  : 원격 주소:포트                            │
│                                                                 │
│  0.0.0.0:* = 모든 인터페이스에서 리스닝                          │
│  127.0.0.1:* = 로컬호스트에서만 리스닝                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 특정 포트 사용 확인

```bash
# 포트 80을 사용하는 프로세스 찾기
ss -tulnp | grep :80
lsof -i :80
fuser 80/tcp

# 특정 프로세스의 네트워크 연결
ss -tulnp | grep nginx
lsof -i -a -p $(pgrep nginx)
```

---

## 4. nslookup / dig

### nslookup

nslookup은 DNS 조회를 수행하는 기본 도구입니다.

```bash
# 기본 조회 (A 레코드)
nslookup google.com

# 특정 DNS 서버 사용
nslookup google.com 8.8.8.8

# 레코드 타입 지정
nslookup -type=MX google.com      # 메일 서버
nslookup -type=NS google.com      # 네임 서버
nslookup -type=TXT google.com     # TXT 레코드
nslookup -type=CNAME www.google.com
nslookup -type=AAAA google.com    # IPv6

# 역방향 조회 (IP → 도메인)
nslookup 8.8.8.8
```

### nslookup 출력 예시

```bash
$ nslookup google.com

Server:         192.168.1.1
Address:        192.168.1.1#53

Non-authoritative answer:
Name:   google.com
Address: 142.250.196.110
```

### dig (권장)

dig는 더 상세한 DNS 정보를 제공하는 강력한 도구입니다.

```bash
# 기본 조회
dig google.com

# 간단한 출력
dig +short google.com

# 특정 레코드
dig google.com MX
dig google.com NS
dig google.com TXT
dig google.com AAAA

# 특정 DNS 서버
dig @8.8.8.8 google.com

# 역방향 조회
dig -x 8.8.8.8

# 추적 (DNS 경로 추적)
dig +trace google.com

# 모든 레코드
dig google.com ANY

# 응답만 표시
dig +noall +answer google.com

# TTL 확인
dig +ttlunits google.com
```

### dig 출력 분석

```bash
$ dig google.com

; <<>> DiG 9.16.1 <<>> google.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 12345
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; QUESTION SECTION:
;google.com.                    IN      A

;; ANSWER SECTION:
google.com.             300     IN      A       142.250.196.110

;; Query time: 25 msec
;; SERVER: 192.168.1.1#53(192.168.1.1)
;; WHEN: Mon Jan 27 10:30:00 KST 2026
;; MSG SIZE  rcvd: 55
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    dig 출력 해석                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HEADER:                                                        │
│  - status: NOERROR = 성공                                       │
│  - flags: qr (응답), rd (재귀 요청), ra (재귀 가능)              │
│                                                                 │
│  QUESTION SECTION:                                              │
│  - 요청한 질의                                                  │
│                                                                 │
│  ANSWER SECTION:                                                │
│  google.com.   300   IN   A   142.250.196.110                   │
│      │          │    │   │          │                           │
│      │          │    │   │          └─ IP 주소                  │
│      │          │    │   └─ 레코드 타입 (A)                     │
│      │          │    └─ 클래스 (IN = Internet)                  │
│      │          └─ TTL (초)                                     │
│      └─ 도메인                                                  │
│                                                                 │
│  Query time: DNS 서버 응답 시간                                 │
│  SERVER: 사용된 DNS 서버                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### host 명령어

간단한 DNS 조회에 유용합니다.

```bash
# 기본 조회
host google.com

# 상세 정보
host -a google.com

# 특정 타입
host -t MX google.com

# 역방향
host 8.8.8.8
```

---

## 5. tcpdump

### tcpdump 개요

tcpdump는 네트워크 패킷을 캡처하고 분석하는 명령줄 도구입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    tcpdump 기본 구조                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  sudo tcpdump [옵션] [필터 표현식]                               │
│                                                                 │
│  주요 옵션:                                                     │
│  -i <인터페이스>  : 캡처 인터페이스 지정                          │
│  -n              : 이름 해석 안함                               │
│  -nn             : 포트도 숫자로                                │
│  -v/-vv/-vvv     : 상세 출력 레벨                               │
│  -c <수>         : 캡처할 패킷 수                               │
│  -w <파일>       : 파일로 저장                                  │
│  -r <파일>       : 파일에서 읽기                                │
│  -A              : ASCII로 출력                                 │
│  -X              : HEX + ASCII로 출력                           │
│  -s <크기>       : 캡처 크기 (0 = 전체)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### tcpdump 기본 사용법

```bash
# 기본 캡처 (root 필요)
sudo tcpdump

# 인터페이스 지정
sudo tcpdump -i eth0
sudo tcpdump -i any        # 모든 인터페이스

# 숫자로 표시 (빠름)
sudo tcpdump -n
sudo tcpdump -nn           # 포트도 숫자로

# 패킷 수 제한
sudo tcpdump -c 10         # 10개만 캡처

# 상세 출력
sudo tcpdump -v
sudo tcpdump -vv
sudo tcpdump -vvv

# 패킷 내용 표시
sudo tcpdump -A            # ASCII
sudo tcpdump -X            # HEX + ASCII
```

### tcpdump 필터

```bash
# 호스트 필터
sudo tcpdump host 192.168.1.100
sudo tcpdump src host 192.168.1.100
sudo tcpdump dst host 192.168.1.100

# 네트워크 필터
sudo tcpdump net 192.168.1.0/24

# 포트 필터
sudo tcpdump port 80
sudo tcpdump port 80 or port 443
sudo tcpdump src port 80
sudo tcpdump dst port 80
sudo tcpdump portrange 8000-9000

# 프로토콜 필터
sudo tcpdump tcp
sudo tcpdump udp
sudo tcpdump icmp
sudo tcpdump arp

# 조합
sudo tcpdump 'tcp port 80 and host 192.168.1.100'
sudo tcpdump 'tcp port 80 and not host 192.168.1.1'
sudo tcpdump 'icmp or arp'

# TCP 플래그
sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0'
sudo tcpdump 'tcp[tcpflags] & tcp-rst != 0'
```

### tcpdump 파일 저장/읽기

```bash
# 파일로 저장
sudo tcpdump -w capture.pcap
sudo tcpdump -w capture.pcap -c 1000

# 파일에서 읽기
tcpdump -r capture.pcap
tcpdump -r capture.pcap 'port 80'

# 로테이션 저장
sudo tcpdump -w log-%H%M%S.pcap -G 3600  # 시간별
sudo tcpdump -w log.pcap -C 100          # 100MB별
```

### tcpdump 출력 분석

```bash
$ sudo tcpdump -i eth0 -nn port 80 -c 3

tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on eth0, link-type EN10MB (Ethernet), capture size 262144 bytes
10:30:00.123456 IP 192.168.1.10.45678 > 93.184.216.34.80: Flags [S], seq 123456789, win 65535, options [mss 1460,nop,wscale 6], length 0
10:30:00.234567 IP 93.184.216.34.80 > 192.168.1.10.45678: Flags [S.], seq 987654321, ack 123456790, win 65535, options [mss 1460,nop,wscale 6], length 0
10:30:00.234789 IP 192.168.1.10.45678 > 93.184.216.34.80: Flags [.], ack 1, win 1024, length 0
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    tcpdump 출력 해석                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  10:30:00.123456 IP 192.168.1.10.45678 > 93.184.216.34.80:      │
│       │              │          │           │         │         │
│       │              │          │           │         └─ 목적지 포트│
│       │              │          │           └─ 목적지 IP        │
│       │              │          └─ 출발지 포트                   │
│       │              └─ 출발지 IP                               │
│       └─ 타임스탬프                                              │
│                                                                 │
│  Flags [S], seq 123456789                                       │
│       │         │                                               │
│       │         └─ TCP 시퀀스 번호                               │
│       └─ TCP 플래그                                             │
│          [S]  = SYN                                             │
│          [S.] = SYN-ACK                                         │
│          [.]  = ACK                                             │
│          [P.] = PSH-ACK (데이터)                                │
│          [F.] = FIN-ACK                                         │
│          [R]  = RST                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 유용한 tcpdump 예시

```bash
# HTTP GET 요청 캡처
sudo tcpdump -i eth0 -A 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | grep -i 'GET\|Host'

# DNS 쿼리 캡처
sudo tcpdump -i eth0 -nn port 53

# SYN 패킷만 (연결 시도)
sudo tcpdump 'tcp[tcpflags] == tcp-syn'

# HTTPS 연결 캡처 (내용은 암호화)
sudo tcpdump -i eth0 port 443

# ARP 트래픽
sudo tcpdump -i eth0 arp

# 특정 호스트 간 통신
sudo tcpdump -i eth0 host 192.168.1.10 and host 192.168.1.20
```

---

## 6. Wireshark 기초

### Wireshark 개요

Wireshark는 그래픽 기반의 강력한 패킷 분석 도구입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wireshark 인터페이스                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     필터 바                              │   │
│  │  [ http.request.method == "GET"                     ]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   패킷 목록 패널                         │   │
│  │  No.   Time    Source       Destination   Protocol Info │   │
│  │  1     0.000   192.168.1.10 93.184.216.34 TCP     SYN   │   │
│  │  2     0.030   93.184.216.34 192.168.1.10 TCP     SYN-ACK│  │
│  │  3     0.031   192.168.1.10 93.184.216.34 TCP     ACK   │   │
│  │  4     0.032   192.168.1.10 93.184.216.34 HTTP    GET / │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   패킷 상세 패널                         │   │
│  │  ▶ Frame                                               │   │
│  │  ▶ Ethernet II                                         │   │
│  │  ▶ Internet Protocol Version 4                         │   │
│  │  ▼ Transmission Control Protocol                       │   │
│  │      Source Port: 45678                                │   │
│  │      Destination Port: 80                              │   │
│  │  ▶ Hypertext Transfer Protocol                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   바이트 패널                            │   │
│  │  0000  00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Wireshark 설치

```bash
# Ubuntu/Debian
sudo apt install wireshark

# macOS
brew install --cask wireshark

# Windows
# https://www.wireshark.org/download.html 에서 다운로드
```

### Wireshark 캡처 필터

캡처 시 적용 (tcpdump와 동일한 BPF 문법):

```
# 특정 호스트
host 192.168.1.100

# 특정 포트
port 80
port 80 or port 443

# 특정 네트워크
net 192.168.1.0/24

# 프로토콜
tcp
udp
icmp

# 조합
tcp port 80 and host 192.168.1.100
```

### Wireshark 디스플레이 필터

캡처 후 표시할 패킷 필터링:

```
# IP 필터
ip.addr == 192.168.1.100
ip.src == 192.168.1.100
ip.dst == 192.168.1.100

# 포트 필터
tcp.port == 80
tcp.srcport == 80
tcp.dstport == 443

# HTTP 필터
http
http.request
http.response
http.request.method == "GET"
http.request.method == "POST"
http.response.code == 200
http.response.code >= 400
http.host contains "google"

# DNS 필터
dns
dns.qry.name contains "google"

# TCP 필터
tcp.flags.syn == 1
tcp.flags.reset == 1
tcp.analysis.retransmission

# TLS/SSL 필터
tls
tls.handshake
ssl.handshake.type == 1  # Client Hello

# 조합
http and ip.src == 192.168.1.100
tcp.port == 443 and ip.addr == 192.168.1.100

# 부정
not arp
not broadcast
!(ip.addr == 192.168.1.1)
```

### Wireshark 주요 기능

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wireshark 주요 기능                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. TCP 스트림 추적 (Follow TCP Stream)                         │
│     - 우클릭 → Follow → TCP Stream                             │
│     - HTTP 대화 전체 확인                                       │
│                                                                 │
│  2. 통계 (Statistics)                                          │
│     - Protocol Hierarchy: 프로토콜별 통계                       │
│     - Conversations: 호스트 간 통신                             │
│     - Endpoints: 통신 엔드포인트                                │
│     - I/O Graphs: 트래픽 그래프                                 │
│                                                                 │
│  3. 전문가 정보 (Analyze → Expert Information)                  │
│     - 오류, 경고, 주의 사항 표시                                │
│     - 재전송, 중복 ACK 등 문제 식별                             │
│                                                                 │
│  4. 파일 추출 (File → Export Objects)                           │
│     - HTTP 객체 추출 (이미지, 파일 등)                          │
│                                                                 │
│  5. 시간 분석                                                   │
│     - View → Time Display Format                                │
│     - RTT, 지연 시간 분석                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### tshark (CLI Wireshark)

```bash
# 기본 캡처
sudo tshark -i eth0

# 필터 적용
sudo tshark -i eth0 -f "port 80"
sudo tshark -i eth0 -Y "http.request"

# 파일로 저장
sudo tshark -i eth0 -w capture.pcap

# 파일 읽기
tshark -r capture.pcap

# 특정 필드 추출
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -e http.host

# JSON 출력
tshark -r capture.pcap -T json
```

---

## 7. curl

### curl 개요

curl은 다양한 프로토콜로 데이터를 전송하는 명령줄 도구입니다.

```bash
# 기본 요청
curl http://example.com

# 출력 저장
curl -o file.html http://example.com
curl -O http://example.com/file.zip  # 원래 파일명 사용

# 리다이렉트 따라가기
curl -L http://example.com

# 헤더 포함 출력
curl -i http://example.com

# 헤더만 출력
curl -I http://example.com

# 상세 출력 (디버그)
curl -v http://example.com
curl -vvv http://example.com  # 더 상세

# 조용한 모드
curl -s http://example.com
curl -sS http://example.com   # 에러만 표시
```

### HTTP 메서드

```bash
# GET (기본)
curl http://api.example.com/users

# POST
curl -X POST http://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "email": "john@example.com"}'

# POST (폼 데이터)
curl -X POST http://example.com/form \
  -d "name=John&email=john@example.com"

# PUT
curl -X PUT http://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "John Updated"}'

# PATCH
curl -X PATCH http://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'

# DELETE
curl -X DELETE http://api.example.com/users/1
```

### 헤더 설정

```bash
# 헤더 추가
curl -H "Authorization: Bearer token123" http://api.example.com
curl -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     http://api.example.com

# User-Agent 설정
curl -A "MyApp/1.0" http://example.com

# Cookie 전송
curl -b "session=abc123" http://example.com
curl -b cookies.txt http://example.com

# Cookie 저장
curl -c cookies.txt http://example.com
```

### 인증

```bash
# Basic 인증
curl -u username:password http://example.com
curl -u username http://example.com  # 비밀번호 프롬프트

# Bearer 토큰
curl -H "Authorization: Bearer token123" http://api.example.com

# API 키
curl -H "X-API-Key: myapikey" http://api.example.com
```

### HTTPS 및 인증서

```bash
# HTTPS 요청
curl https://example.com

# 인증서 검증 무시 (테스트용만)
curl -k https://self-signed.example.com

# 인증서 지정
curl --cacert ca.crt https://example.com

# 클라이언트 인증서
curl --cert client.crt --key client.key https://example.com
```

### 유용한 옵션

```bash
# 타임아웃
curl --connect-timeout 5 http://example.com  # 연결 타임아웃
curl --max-time 30 http://example.com        # 전체 타임아웃

# 재시도
curl --retry 3 http://example.com

# 프록시
curl -x http://proxy:8080 http://example.com

# 압축 지원
curl --compressed http://example.com

# 진행 상황 표시
curl -# -O http://example.com/large-file.zip

# 파일 업로드
curl -F "file=@/path/to/file.pdf" http://example.com/upload

# JSON 파일 전송
curl -X POST http://api.example.com \
  -H "Content-Type: application/json" \
  -d @data.json
```

### curl 응답 분석

```bash
# 응답 코드만 확인
curl -s -o /dev/null -w "%{http_code}" http://example.com

# 상세 시간 정보
curl -s -o /dev/null -w "\
DNS lookup: %{time_namelookup}s\n\
Connect: %{time_connect}s\n\
TLS handshake: %{time_appconnect}s\n\
Start transfer: %{time_starttransfer}s\n\
Total: %{time_total}s\n\
" http://example.com

# 여러 정보 출력
curl -s -o /dev/null -w "\
Response code: %{http_code}\n\
Size: %{size_download} bytes\n\
Time: %{time_total}s\n\
" http://example.com
```

---

## 8. 네트워크 트러블슈팅 방법론

### 체계적인 접근법

```
┌─────────────────────────────────────────────────────────────────┐
│               네트워크 트러블슈팅 흐름                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 문제 정의                                                   │
│     │                                                           │
│     ▼                                                           │
│  2. 정보 수집                                                   │
│     │  - 증상은?                                                │
│     │  - 언제부터?                                              │
│     │  - 어떤 시스템이?                                         │
│     │  - 변경 사항은?                                           │
│     ▼                                                           │
│  3. 가설 수립                                                   │
│     │  - 물리적 문제?                                           │
│     │  - 네트워크 설정?                                         │
│     │  - 서비스 문제?                                           │
│     │  - 방화벽?                                                │
│     ▼                                                           │
│  4. 검증 및 테스트                                              │
│     │  - 단계별 테스트                                          │
│     │  - 변수 격리                                              │
│     ▼                                                           │
│  5. 해결 및 문서화                                              │
│        - 조치 사항 기록                                          │
│        - 재발 방지                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 계층별 진단

```
┌─────────────────────────────────────────────────────────────────┐
│               OSI 계층별 진단                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L1 물리 계층                                                   │
│  ─────────────                                                  │
│  점검: 케이블 연결, LED 상태, 링크 상태                          │
│  도구: ip link, ethtool                                        │
│                                                                 │
│  L2 데이터링크 계층                                              │
│  ───────────────                                                │
│  점검: MAC 주소, ARP 테이블, 스위치 설정                          │
│  도구: arp, ip neigh, arping                                   │
│                                                                 │
│  L3 네트워크 계층                                                │
│  ──────────────                                                 │
│  점검: IP 설정, 라우팅, 방화벽                                   │
│  도구: ip addr, ip route, ping, traceroute                     │
│                                                                 │
│  L4 전송 계층                                                   │
│  ───────────                                                    │
│  점검: 포트 열림, 연결 상태                                      │
│  도구: ss, netstat, nc, telnet                                 │
│                                                                 │
│  L7 애플리케이션 계층                                            │
│  ─────────────────                                              │
│  점검: 서비스 상태, 로그                                         │
│  도구: curl, dig, 서비스 로그                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 단계별 진단 예시

```bash
# 1. 물리/링크 확인
ip link show
ethtool eth0

# 2. IP 설정 확인
ip addr show
ip route show

# 3. 로컬 게이트웨이 ping
ping -c 4 192.168.1.1

# 4. 외부 IP ping
ping -c 4 8.8.8.8

# 5. DNS 확인
nslookup google.com
dig google.com

# 6. 외부 도메인 ping
ping -c 4 google.com

# 7. 포트 연결 테스트
nc -zv google.com 443
curl -v https://google.com

# 8. 경로 추적
traceroute google.com
```

### 일반적인 문제와 해결

| 증상 | 진단 | 해결 |
|------|------|------|
| 모든 연결 안됨 | ip link, 케이블 확인 | 케이블/인터페이스 수리 |
| 로컬만 연결됨 | ip route, 게이트웨이 ping | 라우팅/게이트웨이 설정 |
| IP 연결 OK, 도메인 안됨 | nslookup | DNS 설정 수정 |
| 특정 포트만 안됨 | ss, iptables | 방화벽/서비스 설정 |
| 간헐적 연결 문제 | mtr, 로그 분석 | 네트워크 품질 확인 |
| 느린 연결 | traceroute, iperf | 병목 구간 식별 |

### 트러블슈팅 스크립트

```bash
#!/bin/bash
# 기본 네트워크 진단 스크립트

echo "=== 네트워크 진단 시작 ==="
echo ""

echo "1. 인터페이스 상태:"
ip link show | grep -E "^[0-9]|state"
echo ""

echo "2. IP 주소:"
ip addr show | grep -E "inet |inet6 "
echo ""

echo "3. 라우팅 테이블:"
ip route show
echo ""

echo "4. DNS 설정:"
cat /etc/resolv.conf | grep nameserver
echo ""

echo "5. 게이트웨이 연결 테스트:"
GATEWAY=$(ip route | grep default | awk '{print $3}')
ping -c 2 $GATEWAY
echo ""

echo "6. 외부 연결 테스트 (8.8.8.8):"
ping -c 2 8.8.8.8
echo ""

echo "7. DNS 해석 테스트:"
nslookup google.com
echo ""

echo "8. 리스닝 포트:"
ss -tuln
echo ""

echo "=== 진단 완료 ==="
```

---

## 9. 연습 문제

### 기초 문제

1. **ping**
   - ping 명령으로 알 수 있는 정보 3가지는?
   - TTL 값 116에서 원래 TTL이 128이라면 몇 개의 라우터를 거쳤나요?

2. **traceroute**
   - traceroute가 TTL을 사용하는 원리를 설명하세요.
   - `* * *` 출력의 의미는?

3. **netstat/ss**
   - LISTEN 상태와 ESTABLISHED 상태의 차이는?
   - 포트 80을 사용하는 프로세스를 찾는 명령어는?

### 중급 문제

4. **DNS 도구**
   - nslookup과 dig의 차이점은?
   - MX 레코드와 A 레코드의 용도 차이는?

5. **tcpdump**
   - 다음 필터의 의미를 설명하세요:
     ```bash
     sudo tcpdump -i eth0 'tcp port 80 and host 192.168.1.100'
     ```

6. **실습 문제**
   다음 상황에서 사용할 도구와 명령어를 작성하세요:

   a) 웹 서버(192.168.1.100)가 응답하지 않음
   b) 도메인 접속은 되는데 특정 사이트만 안됨
   c) 간헐적으로 패킷 손실이 발생함

### 고급 문제

7. **Wireshark**
   - TCP 3-way handshake를 필터링하는 디스플레이 필터는?
   - HTTP 응답 코드 500 이상을 필터링하는 방법은?

8. **종합 트러블슈팅**
   - 웹 서비스가 느린 경우, 문제 원인을 파악하기 위한 단계별 접근법을 설명하세요.

---

## 10. 참고 자료

### 도구 공식 문서

- [Wireshark User Guide](https://www.wireshark.org/docs/wsug_html/)
- [tcpdump Manual](https://www.tcpdump.org/manpages/tcpdump.1.html)
- [curl Manual](https://curl.se/docs/manual.html)

### 치트 시트

```
┌─────────────────────────────────────────────────────────────────┐
│                    자주 사용하는 명령어 요약                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  연결 테스트:                                                   │
│    ping -c 4 google.com                                        │
│    traceroute google.com                                       │
│    mtr -r google.com                                           │
│                                                                 │
│  DNS 조회:                                                      │
│    dig +short google.com                                       │
│    nslookup google.com                                         │
│    host google.com                                             │
│                                                                 │
│  포트 확인:                                                     │
│    ss -tuln                                                    │
│    ss -tulnp | grep :80                                        │
│    lsof -i :80                                                 │
│                                                                 │
│  포트 테스트:                                                   │
│    nc -zv google.com 443                                       │
│    telnet google.com 80                                        │
│                                                                 │
│  패킷 캡처:                                                     │
│    sudo tcpdump -i eth0 -nn port 80                            │
│    sudo tcpdump -i eth0 -w capture.pcap                        │
│                                                                 │
│  HTTP 테스트:                                                   │
│    curl -I http://example.com                                  │
│    curl -v https://example.com                                 │
│    curl -o /dev/null -w "%{http_code}" http://example.com      │
│                                                                 │
│  네트워크 정보:                                                 │
│    ip addr show                                                │
│    ip route show                                               │
│    ip neigh show                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 관련 폴더

| 폴더 | 관련 내용 |
|------|----------|
| [Linux/](../Linux/00_Overview.md) | Linux 네트워크 명령어 |
| [Docker/](../Docker/00_Overview.md) | 컨테이너 네트워킹 |
| [Web_Development/](../Web_Development/00_Overview.md) | HTTP/HTTPS, API |

---

## 축하합니다!

네트워크 학습 자료를 모두 완료했습니다.

### 다음 학습 경로

1. **심화 학습**
   - 네트워크 자격증: CCNA, CompTIA Network+
   - 보안 자격증: CompTIA Security+, CEH

2. **실습 환경**
   - GNS3, Packet Tracer로 네트워크 시뮬레이션
   - 홈 랩 구축

3. **관련 분야**
   - 클라우드 네트워킹 (AWS, GCP, Azure)
   - 컨테이너 네트워킹 (Kubernetes, Docker)
   - SDN (Software Defined Networking)
