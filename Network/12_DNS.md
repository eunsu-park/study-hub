# DNS

## 개요

이 문서에서는 DNS(Domain Name System)의 구조와 동작 원리를 다룹니다. 사람이 읽을 수 있는 도메인 이름을 IP 주소로 변환하는 DNS의 계층 구조, 조회 방식, 레코드 유형을 학습합니다.

**난이도**: ⭐⭐
**예상 학습 시간**: 2시간
**선수 지식**: [11_UDP와_포트.md](./11_UDP와_포트.md)

---

## 목차

1. [DNS란?](#1-dns란)
2. [도메인 이름 구조](#2-도메인-이름-구조)
3. [DNS 동작 원리](#3-dns-동작-원리)
4. [DNS 레코드 유형](#4-dns-레코드-유형)
5. [DNS 캐싱](#5-dns-캐싱)
6. [DNS 도구](#6-dns-도구)
7. [연습 문제](#7-연습-문제)
8. [다음 단계](#8-다음-단계)
9. [참고 자료](#9-참고-자료)

---

## 1. DNS란?

### 1.1 DNS의 정의

DNS(Domain Name System)는 도메인 이름을 IP 주소로 변환하는 분산 데이터베이스 시스템입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                       DNS의 역할                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  사람: www.google.com 접속하고 싶어!                            │
│                                                                  │
│       www.google.com                                            │
│            │                                                     │
│            ▼                                                     │
│       ┌─────────┐                                               │
│       │   DNS   │   "www.google.com = 142.250.196.68"          │
│       │  Server │                                               │
│       └─────────┘                                               │
│            │                                                     │
│            ▼                                                     │
│       142.250.196.68                                            │
│            │                                                     │
│            ▼                                                     │
│       ┌─────────┐                                               │
│       │ Google  │                                               │
│       │ Server  │                                               │
│       └─────────┘                                               │
│                                                                  │
│  비유: 인터넷의 전화번호부                                      │
│        이름 → 전화번호 (도메인 → IP 주소)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 DNS가 필요한 이유

```
┌─────────────────────────────────────────────────────────────────┐
│                      DNS 없이 살 수 있을까?                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  IP 주소를 직접 사용하면?                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • http://142.250.196.68 → Google                        │  │
│  │  • http://31.13.82.36 → Facebook                         │  │
│  │  • http://52.94.236.248 → Amazon                         │  │
│  │                                                           │  │
│  │  문제점:                                                  │  │
│  │  1. 외우기 어려움                                         │  │
│  │  2. IP 주소 변경 시 사용자 혼란                           │  │
│  │  3. 하나의 IP에 여러 서비스 호스팅 어려움                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  DNS의 장점:                                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • 기억하기 쉬운 이름 사용                                │  │
│  │  • 서버 IP 변경이 투명함                                  │  │
│  │  • 부하 분산 가능 (여러 IP 매핑)                          │  │
│  │  • 지역별 최적 서버 연결 가능                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 DNS 특성

| 특성 | 설명 |
|------|------|
| 분산 시스템 | 전 세계에 분산된 서버들이 협력 |
| 계층 구조 | 루트 → TLD → 권한 서버 계층 |
| 캐싱 | 성능 향상을 위한 캐시 사용 |
| 중복성 | 여러 서버로 가용성 보장 |
| 프로토콜 | 주로 UDP 53 (대용량은 TCP 53) |

---

## 2. 도메인 이름 구조

### 2.1 도메인 계층 구조

```
                     도메인 이름 계층

                         . (Root)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
        com             org             kr
        (TLD)          (TLD)           (ccTLD)
         │               │               │
    ┌────┴────┐         ...         ┌───┴───┐
    │         │                     │       │
  google   amazon                  co      go
  (SLD)    (SLD)                 (2LD)   (2LD)
    │         │                     │       │
   www      aws                   naver   korea
  (sub)    (sub)                 (3LD)   (3LD)
                                    │
                                   www
                                  (sub)

FQDN (Fully Qualified Domain Name):
www.google.com.   ← 끝의 점(.)은 루트를 나타냄
```

### 2.2 도메인 구성 요소

```
                    www.example.co.kr
                     │      │   │  │
    ┌────────────────┘      │   │  └─── TLD (Top-Level Domain)
    │         ┌─────────────┘   │        최상위 도메인 (kr)
    │         │          ┌──────┘
    │         │          │
Subdomain   SLD    Second-level
 (3차 도메인) (2차 도메인)  (TLD 아래)

분석:
┌──────────────────────────────────────────────────────────────────┐
│ www.example.co.kr                                                │
├──────────────────────────────────────────────────────────────────┤
│ kr      : TLD (국가 코드 최상위 도메인 - ccTLD)                 │
│ co      : Second-level domain (한국에서 회사용)                  │
│ example : 등록된 도메인 이름                                     │
│ www     : 서브도메인 (호스트 이름)                               │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 TLD 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                        TLD 분류                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  gTLD (Generic TLD) - 일반 최상위 도메인                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  .com    - 상업용                                         │  │
│  │  .org    - 비영리 조직                                    │  │
│  │  .net    - 네트워크 관련                                  │  │
│  │  .edu    - 교육 기관 (미국)                               │  │
│  │  .gov    - 미국 정부                                      │  │
│  │  .mil    - 미국 군                                        │  │
│  │  .info   - 정보 제공                                      │  │
│  │  .biz    - 비즈니스                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ccTLD (Country Code TLD) - 국가 코드 최상위 도메인             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  .kr     - 한국                                           │  │
│  │  .jp     - 일본                                           │  │
│  │  .uk     - 영국                                           │  │
│  │  .de     - 독일                                           │  │
│  │  .cn     - 중국                                           │  │
│  │  .us     - 미국                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  New gTLD - 새로운 일반 최상위 도메인 (2012년 이후)             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  .app, .dev, .blog, .shop, .xyz, .io, .ai 등              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 한국 도메인 구조

```
.kr 도메인 체계

kr (ccTLD)
 │
 ├── co.kr    : 영리 기업
 ├── or.kr    : 비영리 조직
 ├── go.kr    : 정부 기관
 ├── ac.kr    : 교육 기관 (대학)
 ├── re.kr    : 연구 기관
 ├── ne.kr    : 네트워크 서비스
 ├── pe.kr    : 개인
 └── 지역.kr  : 서울.kr, 부산.kr 등

예시:
  www.naver.com        - gTLD 사용
  www.samsung.co.kr    - 한국 기업
  www.korea.go.kr      - 한국 정부
  www.snu.ac.kr        - 서울대학교
```

---

## 3. DNS 동작 원리

### 3.1 DNS 서버 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                      DNS 서버 유형                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Recursive Resolver (재귀 확인자)                            │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • 클라이언트의 요청을 받아 다른 서버에 질의            │  │
│     │ • ISP 또는 공개 DNS (8.8.8.8, 1.1.1.1)                 │  │
│     │ • 결과를 캐싱하여 재사용                                │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. Root Name Server (루트 네임 서버)                           │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • DNS 계층의 최상위                                    │  │
│     │ • 전 세계 13개 루트 서버 (A-M)                         │  │
│     │ • TLD 서버 위치 제공                                   │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. TLD Name Server (TLD 네임 서버)                             │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • .com, .org, .kr 등 TLD 담당                          │  │
│     │ • 권한 서버 위치 제공                                  │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. Authoritative Name Server (권한 네임 서버)                  │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • 특정 도메인의 실제 DNS 레코드 보유                   │  │
│     │ • 최종 IP 주소 응답                                    │  │
│     │ • 도메인 소유자가 관리                                 │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 재귀적 조회 (Recursive Query)

클라이언트가 Recursive Resolver에게 질의하면, 해당 리졸버가 모든 과정을 처리합니다.

```
재귀적 조회 과정 (www.example.com 조회)

┌──────────┐                          ┌──────────────────┐
│  Client  │───(1) www.example.com?──►│ Recursive        │
│          │                          │ Resolver         │
│          │◄──(10) 93.184.216.34 ────│ (예: 8.8.8.8)    │
└──────────┘                          └────────┬─────────┘
                                               │
        ┌──────────────────────────────────────┼──────────────┐
        │                                      │              │
        │  ┌───(2) .com 서버 어디?────────────►│              │
        │  │                                   ▼              │
        │  │                          ┌──────────────┐        │
        │  │                          │ Root Server  │        │
        │  │◄──(3) TLD 서버 주소 ─────│  (13개)      │        │
        │  │                          └──────────────┘        │
        │  │                                                  │
        │  │  ┌───(4) example.com 서버 어디?──►               │
        │  │  │                                ▼              │
        │  │  │                       ┌──────────────┐        │
        │  │  │                       │ .com TLD     │        │
        │  │  │◄──(5) 권한 서버 주소─│ Server       │        │
        │  │  │                       └──────────────┘        │
        │  │  │                                               │
        │  │  │  ┌──(6) www.example.com IP?────►              │
        │  │  │  │                             ▼              │
        │  │  │  │                    ┌────────────────┐      │
        │  │  │  │                    │ Authoritative  │      │
        │  │  │  │◄─(7) 93.184.216.34│ Server         │      │
        │  │  │  │                    │(example.com용) │      │
        │  │  │  │                    └────────────────┘      │
        │  │  │  │                                            │
        └──┴──┴──┴────────────────────────────────────────────┘
```

### 3.3 반복적 조회 (Iterative Query)

Recursive Resolver가 각 DNS 서버에 차례로 질의하고, 다음 서버 정보를 받아 직접 질의합니다.

```
반복적 조회 과정

                    Recursive Resolver
                          │
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    │ (2) "com은 어디?"   │                     │
    │ ─────────────────►  │                     │
    │                     ▼                     │
    │              ┌────────────┐               │
    │              │   Root     │               │
    │ ◄─────────── │   Server   │               │
    │ (3) "a.gtld-servers.net"                  │
    │              └────────────┘               │
    │                                           │
    │ (4) "example.com은 어디?"                 │
    │ ─────────────────────────────►            │
    │                              ▼            │
    │                       ┌────────────┐      │
    │                       │ .com TLD   │      │
    │ ◄───────────────────  │ Server     │      │
    │ (5) "ns1.example.com"                     │
    │                       └────────────┘      │
    │                                           │
    │ (6) "www.example.com의 IP는?"             │
    │ ─────────────────────────────────────►    │
    │                                      ▼    │
    │                              ┌────────────┐
    │                              │Authoritative│
    │ ◄─────────────────────────── │   Server   │
    │ (7) "93.184.216.34"          └────────────┘
    │                                           │
    └───────────────────────────────────────────┘
```

### 3.4 DNS 질의/응답 메시지

```
DNS 메시지 구조

┌────────────────────────────────────────────────────────────────┐
│                         Header                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ID (16 bits) - 질의/응답 매칭                           │   │
│  │ Flags: QR, Opcode, AA, TC, RD, RA, Z, RCODE             │   │
│  │ QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT                      │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                        Question                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ QNAME: www.example.com (질의 도메인)                    │   │
│  │ QTYPE: A (질의 유형)                                    │   │
│  │ QCLASS: IN (인터넷)                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                         Answer                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ NAME: www.example.com                                   │   │
│  │ TYPE: A                                                 │   │
│  │ CLASS: IN                                               │   │
│  │ TTL: 300 (초)                                           │   │
│  │ RDLENGTH: 4                                             │   │
│  │ RDATA: 93.184.216.34                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                       Authority                                 │
│  (권한 서버 정보)                                               │
├────────────────────────────────────────────────────────────────┤
│                       Additional                                │
│  (추가 정보 - 예: 권한 서버의 IP)                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. DNS 레코드 유형

### 4.1 주요 DNS 레코드

| 레코드 | 의미 | 설명 |
|--------|------|------|
| A | Address | IPv4 주소 매핑 |
| AAAA | IPv6 Address | IPv6 주소 매핑 |
| CNAME | Canonical Name | 도메인 별칭 |
| MX | Mail Exchanger | 메일 서버 지정 |
| NS | Name Server | 네임 서버 지정 |
| TXT | Text | 텍스트 정보 (SPF, DKIM 등) |
| PTR | Pointer | 역방향 조회 (IP → 도메인) |
| SOA | Start of Authority | 영역 권한 정보 |
| SRV | Service | 서비스 위치 정보 |
| CAA | Certification Authority | 인증서 발급 권한 |

### 4.2 A 레코드

도메인 이름을 IPv4 주소에 매핑합니다.

```
A 레코드 예시

example.com.     IN  A     93.184.216.34
www.example.com. IN  A     93.184.216.34
api.example.com. IN  A     93.184.216.35

부하 분산 (Round Robin):
www.example.com. IN  A     93.184.216.34
www.example.com. IN  A     93.184.216.35
www.example.com. IN  A     93.184.216.36

조회 결과:
$ dig www.example.com A

;; ANSWER SECTION:
www.example.com.    300    IN    A    93.184.216.34
```

### 4.3 AAAA 레코드

도메인 이름을 IPv6 주소에 매핑합니다.

```
AAAA 레코드 예시

example.com.     IN  AAAA  2606:2800:220:1:248:1893:25c8:1946
www.example.com. IN  AAAA  2606:2800:220:1:248:1893:25c8:1946

조회 결과:
$ dig www.example.com AAAA

;; ANSWER SECTION:
www.example.com.    300    IN    AAAA    2606:2800:220:1:248:1893:25c8:1946
```

### 4.4 CNAME 레코드

하나의 도메인을 다른 도메인으로 매핑합니다 (별칭).

```
CNAME 레코드 예시

www.example.com.    IN  CNAME  example.com.
blog.example.com.   IN  CNAME  blogger.l.google.com.
shop.example.com.   IN  CNAME  shops.myshopify.com.

CNAME 체이닝:
alias.example.com.  IN  CNAME  www.example.com.
www.example.com.    IN  CNAME  example.com.
example.com.        IN  A      93.184.216.34

조회 과정:
alias.example.com
    → www.example.com (CNAME)
    → example.com (CNAME)
    → 93.184.216.34 (A)

주의사항:
- 루트 도메인 (example.com)에는 CNAME 사용 불가
- MX, NS 레코드와 함께 사용 불가
- ALIAS/ANAME 레코드로 대체 (일부 DNS 제공자)
```

### 4.5 MX 레코드

도메인의 메일 서버를 지정합니다.

```
MX 레코드 예시

example.com.  IN  MX  10  mail1.example.com.
example.com.  IN  MX  20  mail2.example.com.
example.com.  IN  MX  30  mail3.backup.com.

우선순위 (Priority):
- 숫자가 낮을수록 우선순위 높음
- 10 → 20 → 30 순서로 시도

Google Workspace 예시:
example.com.  IN  MX  1   aspmx.l.google.com.
example.com.  IN  MX  5   alt1.aspmx.l.google.com.
example.com.  IN  MX  5   alt2.aspmx.l.google.com.
example.com.  IN  MX  10  alt3.aspmx.l.google.com.
example.com.  IN  MX  10  alt4.aspmx.l.google.com.

조회:
$ dig example.com MX

;; ANSWER SECTION:
example.com.    300    IN    MX    10 mail1.example.com.
example.com.    300    IN    MX    20 mail2.example.com.
```

### 4.6 NS 레코드

도메인을 관리하는 네임 서버를 지정합니다.

```
NS 레코드 예시

example.com.    IN  NS    ns1.example.com.
example.com.    IN  NS    ns2.example.com.

위임 (Delegation):
sub.example.com.  IN  NS    ns1.subdomain.com.
sub.example.com.  IN  NS    ns2.subdomain.com.

Glue 레코드 (네임 서버가 같은 도메인에 있을 때):
example.com.      IN  NS    ns1.example.com.
ns1.example.com.  IN  A     192.0.2.1
ns2.example.com.  IN  A     192.0.2.2
```

### 4.7 TXT 레코드

텍스트 정보를 저장합니다. 주로 인증 및 검증에 사용됩니다.

```
TXT 레코드 용도

1. SPF (Sender Policy Framework) - 이메일 발신 인증
example.com.  IN  TXT  "v=spf1 include:_spf.google.com ~all"

2. DKIM (DomainKeys Identified Mail) - 이메일 서명
google._domainkey.example.com.  IN  TXT  "v=DKIM1; k=rsa; p=MIGf..."

3. DMARC (Domain-based Message Authentication)
_dmarc.example.com.  IN  TXT  "v=DMARC1; p=reject; rua=mailto:dmarc@example.com"

4. 도메인 소유 확인 (Google, MS 등)
example.com.  IN  TXT  "google-site-verification=..."
example.com.  IN  TXT  "MS=ms12345678"

5. 기타 서비스 설정
example.com.  IN  TXT  "facebook-domain-verification=..."
```

### 4.8 PTR 레코드

IP 주소를 도메인 이름으로 매핑합니다 (역방향 조회).

```
PTR 레코드 예시

역방향 조회 영역:
IP: 93.184.216.34
역방향 도메인: 34.216.184.93.in-addr.arpa

PTR 레코드:
34.216.184.93.in-addr.arpa.  IN  PTR  www.example.com.

IPv6 역방향:
IP: 2001:db8::1
역방향 도메인: 1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa

용도:
- 이메일 서버 검증 (스팸 필터)
- 로그에서 IP를 도메인으로 표시
- 보안 점검

조회:
$ dig -x 93.184.216.34

;; ANSWER SECTION:
34.216.184.93.in-addr.arpa. 3600 IN PTR www.example.com.
```

### 4.9 SOA 레코드

DNS 영역의 권한 정보를 정의합니다.

```
SOA 레코드 예시

example.com. IN SOA ns1.example.com. admin.example.com. (
    2024010101   ; Serial Number (YYYYMMDDNN)
    3600         ; Refresh (1시간)
    600          ; Retry (10분)
    604800       ; Expire (1주)
    86400        ; Minimum TTL (1일)
)

필드 설명:
- Primary NS: ns1.example.com (기본 네임 서버)
- Admin Email: admin@example.com (admin.example.com으로 표기)
- Serial: 변경 시마다 증가 (보조 서버 동기화용)
- Refresh: 보조 서버가 기본 서버 확인 주기
- Retry: Refresh 실패 시 재시도 간격
- Expire: 기본 서버 연결 불가 시 데이터 유효 기간
- Minimum TTL: 부정 응답(NXDOMAIN) 캐시 시간
```

---

## 5. DNS 캐싱

### 5.1 캐싱 계층

```
DNS 캐싱 계층

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Level 1: 브라우저 캐시                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • 브라우저 자체 DNS 캐시                                   │ │
│  │ • Chrome: chrome://net-internals/#dns                      │ │
│  │ • 짧은 TTL (보통 분 단위)                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        │                                         │
│                        ▼                                         │
│  Level 2: 운영체제 캐시                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • OS의 DNS 리졸버 캐시                                     │ │
│  │ • Windows: ipconfig /displaydns                            │ │
│  │ • macOS: dscacheutil -cachedump -entries                   │ │
│  │ • Linux: systemd-resolved 등                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        │                                         │
│                        ▼                                         │
│  Level 3: Recursive Resolver 캐시                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • ISP 또는 공개 DNS 서버의 캐시                            │ │
│  │ • 많은 사용자가 공유                                       │ │
│  │ • TTL 기반 캐시 유효 시간                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        │                                         │
│                        ▼                                         │
│  Level 4: Authoritative Server                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • 캐시 미스 시 실제 조회                                   │ │
│  │ • 권한 있는 응답 제공                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 TTL (Time To Live)

```
TTL의 역할

TTL = DNS 레코드를 캐시할 수 있는 시간 (초)

example.com.  300  IN  A  93.184.216.34
              ↑
             TTL (300초 = 5분)

TTL 설정 전략:
┌──────────────────────────────────────────────────────────────────┐
│ 상황                      │ 권장 TTL    │ 이유                   │
├──────────────────────────────────────────────────────────────────┤
│ 일반 운영                 │ 3600-86400  │ 캐시 효율 극대화       │
│ (1시간 - 1일)             │             │                        │
├──────────────────────────────────────────────────────────────────┤
│ 마이그레이션 예정         │ 300-600     │ 빠른 전파              │
│ (5분 - 10분)              │             │                        │
├──────────────────────────────────────────────────────────────────┤
│ 페일오버/HA               │ 60-300      │ 신속한 장애 대응       │
│ (1분 - 5분)               │             │                        │
├──────────────────────────────────────────────────────────────────┤
│ 변경 직전                 │ 60          │ 기존 캐시 빠른 만료    │
│ (1분)                     │             │                        │
└──────────────────────────────────────────────────────────────────┘

TTL 트레이드오프:
낮은 TTL:
  + 변경 사항 빠른 전파
  - DNS 쿼리 증가, 서버 부하

높은 TTL:
  + 캐시 효율, 빠른 응답
  - 변경 전파 느림
```

### 5.3 캐시 삭제

```bash
# Windows
ipconfig /flushdns

# macOS
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder

# Linux (systemd-resolved)
sudo systemd-resolve --flush-caches

# Chrome 브라우저
chrome://net-internals/#dns → Clear host cache

# Firefox
about:networking#dns → Clear DNS Cache
```

---

## 6. DNS 도구

### 6.1 nslookup

```bash
# 기본 조회
nslookup google.com

# 특정 레코드 조회
nslookup -type=MX google.com
nslookup -type=A google.com
nslookup -type=AAAA google.com

# 특정 DNS 서버 사용
nslookup google.com 8.8.8.8

# 인터랙티브 모드
nslookup
> set type=MX
> google.com
> exit

출력 예시:
Server:    8.8.8.8
Address:   8.8.8.8#53

Non-authoritative answer:
Name:      google.com
Address:   142.250.196.78
```

### 6.2 dig

```bash
# 기본 조회
dig google.com

# 특정 레코드 조회
dig google.com A
dig google.com AAAA
dig google.com MX
dig google.com NS
dig google.com TXT

# 간단한 출력
dig +short google.com

# 상세 출력 (추적)
dig +trace google.com

# 특정 DNS 서버 사용
dig @8.8.8.8 google.com

# 역방향 조회
dig -x 142.250.196.78

# TTL 표시
dig +ttlid google.com

# 모든 레코드 조회
dig google.com ANY

출력 예시:
;; ANSWER SECTION:
google.com.        137    IN    A    142.250.196.78

;; Query time: 15 msec
;; SERVER: 8.8.8.8#53(8.8.8.8)
```

### 6.3 host

```bash
# 기본 조회
host google.com

# 레코드 유형 지정
host -t MX google.com
host -t NS google.com
host -t TXT google.com

# 상세 출력
host -v google.com

# 역방향 조회
host 142.250.196.78

출력 예시:
google.com has address 142.250.196.78
google.com has IPv6 address 2404:6800:4004:821::200e
google.com mail is handled by 10 smtp.google.com.
```

### 6.4 dig 출력 해석

```
$ dig www.example.com

; <<>> DiG 9.18.1 <<>> www.example.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 12345
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 512

;; QUESTION SECTION:
;www.example.com.            IN    A

;; ANSWER SECTION:
www.example.com.    86400    IN    A    93.184.216.34

;; Query time: 25 msec
;; SERVER: 8.8.8.8#53(8.8.8.8)
;; WHEN: Mon Jan 15 10:30:45 KST 2024
;; MSG SIZE  rcvd: 59

해석:
- status: NOERROR → 성공
- flags: qr(응답), rd(재귀 요청), ra(재귀 가능)
- QUESTION: 질의한 내용
- ANSWER: 응답 (A 레코드, TTL 86400초, IP 93.184.216.34)
- Query time: 응답 시간 (25ms)
- SERVER: 응답한 DNS 서버
```

### 6.5 공개 DNS 서버

| 제공자 | IPv4 | IPv6 | 특징 |
|--------|------|------|------|
| Google | 8.8.8.8, 8.8.4.4 | 2001:4860:4860::8888 | 전 세계, 빠름 |
| Cloudflare | 1.1.1.1, 1.0.0.1 | 2606:4700:4700::1111 | 프라이버시 중시 |
| Quad9 | 9.9.9.9, 149.112.112.112 | 2620:fe::fe | 보안 중심 |
| OpenDNS | 208.67.222.222 | 2620:119:35::35 | 필터링 옵션 |

---

## 7. 연습 문제

### 문제 1: 도메인 구조 분석

다음 도메인의 각 부분을 식별하세요.

```
a) www.shop.amazon.co.uk
b) mail.google.com
c) api.v2.example.org
```

### 문제 2: DNS 레코드 매칭

다음 상황에 적합한 DNS 레코드 유형을 선택하세요.

a) 웹 서버의 IPv4 주소 지정
b) 메일 서버 지정
c) www를 기본 도메인으로 리다이렉트
d) 도메인 소유권 인증
e) 네임 서버 지정
f) IPv6 주소 지정

### 문제 3: dig 출력 분석

다음 dig 출력을 분석하세요.

```
;; ANSWER SECTION:
example.com.        600    IN    MX    10 mail1.example.com.
example.com.        600    IN    MX    20 mail2.example.com.
example.com.        600    IN    MX    30 backup.mail.com.
```

a) TTL은 얼마인가요?
b) 어느 메일 서버가 우선 사용되나요?
c) 모든 메일 서버가 다운되면 어떻게 되나요?

### 문제 4: DNS 조회 실습

다음 명령어를 실행하고 결과를 분석하세요.

```bash
dig google.com A
dig google.com MX
dig +trace google.com
```

---

## 정답

### 문제 1 정답

a) www.shop.amazon.co.uk
```
uk    : TLD (ccTLD - 영국)
co    : Second-level domain (회사용)
amazon: 등록된 도메인
shop  : 서브도메인
www   : 서브도메인 (호스트)
```

b) mail.google.com
```
com   : TLD (gTLD)
google: SLD (등록된 도메인)
mail  : 서브도메인
```

c) api.v2.example.org
```
org    : TLD (gTLD)
example: SLD
v2     : 서브도메인
api    : 서브도메인
```

### 문제 2 정답

- a) IPv4 주소 → **A 레코드**
- b) 메일 서버 → **MX 레코드**
- c) 리다이렉트 → **CNAME 레코드**
- d) 소유권 인증 → **TXT 레코드**
- e) 네임 서버 → **NS 레코드**
- f) IPv6 주소 → **AAAA 레코드**

### 문제 3 정답

a) TTL = **600초 (10분)**
b) 우선 메일 서버: **mail1.example.com** (우선순위 10)
c) backup.mail.com까지 시도 후 실패 → **메일 전송 실패 (반송)**

### 문제 4 정답

실습 결과는 환경에 따라 다르지만, 확인 포인트:
- A 레코드: Google의 IP 주소 (여러 개일 수 있음)
- MX 레코드: Google의 메일 서버 (aspmx.l.google.com 등)
- +trace: 루트 → .com TLD → google.com 권한 서버 순서로 조회

---

## 8. 다음 단계

DNS를 이해했다면, HTTP와 HTTPS에 대해 학습하세요.

### 다음 레슨
- [13_HTTP와_HTTPS.md](./13_HTTP와_HTTPS.md) - HTTP 프로토콜, TLS/SSL

### 관련 레슨
- [11_UDP와_포트.md](./11_UDP와_포트.md) - DNS가 사용하는 UDP
- [15_네트워크_보안_기초.md](./15_네트워크_보안_기초.md) - DNS 보안

### 추천 실습
1. `dig +trace`로 DNS 조회 과정 추적
2. 자신의 DNS 서버 설정 확인
3. 다양한 도메인의 DNS 레코드 조회

---

## 9. 참고 자료

### RFC 문서

- RFC 1034 - Domain Names: Concepts and Facilities
- RFC 1035 - Domain Names: Implementation and Specification
- RFC 8484 - DNS Queries over HTTPS (DoH)
- RFC 7858 - DNS over TLS (DoT)

### 온라인 도구

- [DNS Checker](https://dnschecker.org/) - 전 세계 DNS 전파 확인
- [MX Toolbox](https://mxtoolbox.com/) - DNS/메일 진단
- [whatsmydns.net](https://www.whatsmydns.net/) - DNS 조회
- [IntoDNS](https://intodns.com/) - DNS 설정 검사

### 학습 자료

- [How DNS Works (Comic)](https://howdns.works/)
- [Cloudflare Learning: DNS](https://www.cloudflare.com/learning/dns/)
- [Google Public DNS](https://developers.google.com/speed/public-dns)

---

**문서 정보**
- 최종 수정: 2024년
- 난이도: ⭐⭐
- 예상 학습 시간: 2시간
