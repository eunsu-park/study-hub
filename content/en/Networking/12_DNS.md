# DNS

**Previous**: [UDP and Ports](./11_UDP_and_Ports.md) | **Next**: [HTTP and HTTPS](./13_HTTP_and_HTTPS.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of DNS and why it is essential for internet communication
2. Decompose a fully qualified domain name (FQDN) into its hierarchical components
3. Distinguish between recursive and iterative DNS queries
4. Identify the roles of root, TLD, and authoritative name servers
5. Describe common DNS record types (A, AAAA, CNAME, MX, NS, TXT, PTR, SOA) and their use cases
6. Explain DNS caching layers and the role of TTL in cache management
7. Use DNS diagnostic tools (`dig`, `nslookup`, `host`) to query and troubleshoot DNS records

---

**Difficulty**: ⭐⭐
**Estimated Learning Time**: 2 hours
**Prerequisites**: [11_UDP_and_Ports.md](./11_UDP_and_Ports.md)

> **Analogy -- Your Phone's Contacts App**: DNS is the internet's contacts app. You don't memorize your friend's phone number (192.168.1.100) -- you just tap their name (google.com) and your phone looks up the number. DNS does the same thing: it translates human-friendly domain names into machine-friendly IP addresses. Just like your contacts sync across devices and update when someone changes their number, DNS propagates changes across a global network of servers.

Every time you type a URL, send an email, or stream a video, DNS works behind the scenes to translate names into numbers. Without DNS, the internet would be a sea of forgettable IP addresses. This lesson reveals how that global name-resolution system is organized, how queries flow through its hierarchy, and how to diagnose problems when name lookups fail.

## Table of Contents

1. [What is DNS?](#1-what-is-dns)
2. [Domain Name Structure](#2-domain-name-structure)
3. [How DNS Works](#3-how-dns-works)
4. [DNS Record Types](#4-dns-record-types)
5. [DNS Caching](#5-dns-caching)
6. [DNS Tools](#6-dns-tools)
7. [Practice Problems](#7-practice-problems)
8. [References](#8-references)

---

## 1. What is DNS?

### 1.1 DNS Definition

DNS (Domain Name System) is a distributed database system that translates domain names into IP addresses.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Role of DNS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Human: I want to access www.google.com!                        │
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
│  Analogy: The Internet's Phone Book                             │
│           Name → Phone Number (Domain → IP Address)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why DNS is Needed

```
┌─────────────────────────────────────────────────────────────────┐
│                  Can We Live Without DNS?                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  What if we use IP addresses directly?                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • http://142.250.196.68 → Google                        │  │
│  │  • http://31.13.82.36 → Facebook                         │  │
│  │  • http://52.94.236.248 → Amazon                         │  │
│  │                                                           │  │
│  │  Problems:                                                │  │
│  │  1. Hard to memorize                                      │  │
│  │  2. User confusion when IP addresses change               │  │
│  │  3. Difficult to host multiple services on one IP         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Advantages of DNS:                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • Easy-to-remember names                                 │  │
│  │  • Server IP changes are transparent                      │  │
│  │  • Load balancing possible (multiple IP mappings)         │  │
│  │  • Regional optimal server connections                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 DNS Characteristics

| Characteristic | Description |
|----------------|-------------|
| Distributed System | Servers distributed worldwide cooperate |
| Hierarchical Structure | Root → TLD → Authoritative server hierarchy |
| Caching | Uses cache for performance improvement |
| Redundancy | Multiple servers ensure availability |
| Protocol | Primarily UDP 53 (TCP 53 for large volumes) |

---

## 2. Domain Name Structure

### 2.1 Domain Hierarchy

```
                     Domain Name Hierarchy

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
www.google.com.   ← The trailing dot (.) represents root
```

### 2.2 Domain Components

```
                    www.example.co.kr
                     │      │   │  │
    ┌────────────────┘      │   │  └─── TLD (Top-Level Domain)
    │         ┌─────────────┘   │        Top-level domain (kr)
    │         │          ┌──────┘
    │         │          │
Subdomain   SLD    Second-level
 (3rd level) (2nd level)  (under TLD)

Analysis:
┌──────────────────────────────────────────────────────────────────┐
│ www.example.co.kr                                                │
├──────────────────────────────────────────────────────────────────┤
│ kr      : TLD (Country Code Top-Level Domain - ccTLD)           │
│ co      : Second-level domain (for companies in Korea)          │
│ example : Registered domain name                                │
│ www     : Subdomain (hostname)                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 TLD Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        TLD Categories                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  gTLD (Generic TLD) - Generic Top-Level Domains                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  .com    - Commercial                                     │  │
│  │  .org    - Non-profit organizations                       │  │
│  │  .net    - Network-related                                │  │
│  │  .edu    - Educational institutions (US)                  │  │
│  │  .gov    - US Government                                  │  │
│  │  .mil    - US Military                                    │  │
│  │  .info   - Information                                    │  │
│  │  .biz    - Business                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ccTLD (Country Code TLD) - Country Code Top-Level Domains      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  .kr     - Korea                                          │  │
│  │  .jp     - Japan                                          │  │
│  │  .uk     - United Kingdom                                 │  │
│  │  .de     - Germany                                        │  │
│  │  .cn     - China                                          │  │
│  │  .us     - United States                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  New gTLD - New Generic Top-Level Domains (since 2012)          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  .app, .dev, .blog, .shop, .xyz, .io, .ai, etc.          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Korean Domain Structure

```
.kr Domain System

kr (ccTLD)
 │
 ├── co.kr    : Commercial companies
 ├── or.kr    : Non-profit organizations
 ├── go.kr    : Government agencies
 ├── ac.kr    : Educational institutions (universities)
 ├── re.kr    : Research institutions
 ├── ne.kr    : Network services
 ├── pe.kr    : Personal
 └── region.kr: seoul.kr, busan.kr, etc.

Examples:
  www.naver.com        - Uses gTLD
  www.samsung.co.kr    - Korean company
  www.korea.go.kr      - Korean government
  www.snu.ac.kr        - Seoul National University
```

---

## 3. How DNS Works

### 3.1 DNS Server Types

```
┌─────────────────────────────────────────────────────────────────┐
│                      DNS Server Types                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Recursive Resolver                                          │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Receives client requests and queries other servers   │  │
│     │ • ISP or public DNS (8.8.8.8, 1.1.1.1)                 │  │
│     │ • Caches results for reuse                              │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. Root Name Server                                            │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Top of DNS hierarchy                                 │  │
│     │ • 13 root servers worldwide (A-M)                      │  │
│     │ • Provides TLD server locations                        │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. TLD Name Server                                             │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Responsible for .com, .org, .kr, etc.                │  │
│     │ • Provides authoritative server locations              │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. Authoritative Name Server                                   │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Holds actual DNS records for specific domains        │  │
│     │ • Provides final IP address responses                  │  │
│     │ • Managed by domain owner                              │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Recursive Query

When a client queries a Recursive Resolver, the resolver handles the entire process.

```
Recursive Query Process (looking up www.example.com)

┌──────────┐                          ┌──────────────────┐
│  Client  │───(1) www.example.com?──►│ Recursive        │
│          │                          │ Resolver         │
│          │◄──(10) 93.184.216.34 ────│ (e.g., 8.8.8.8)  │
└──────────┘                          └────────┬─────────┘
                                               │
        ┌──────────────────────────────────────┼──────────────┐
        │                                      │              │
        │  ┌───(2) Where is .com server?──────►│              │
        │  │                                   ▼              │
        │  │                          ┌──────────────┐        │
        │  │                          │ Root Server  │        │
        │  │◄──(3) TLD server address │  (13 total)  │        │
        │  │                          └──────────────┘        │
        │  │                                                  │
        │  │  ┌───(4) Where is example.com server?──►         │
        │  │  │                                ▼              │
        │  │  │                       ┌──────────────┐        │
        │  │  │                       │ .com TLD     │        │
        │  │  │◄──(5) Auth server addr│ Server       │        │
        │  │  │                       └──────────────┘        │
        │  │  │                                               │
        │  │  │  ┌──(6) www.example.com IP?────►              │
        │  │  │  │                             ▼              │
        │  │  │  │                    ┌────────────────┐      │
        │  │  │  │                    │ Authoritative  │      │
        │  │  │  │◄─(7) 93.184.216.34 │ Server         │      │
        │  │  │  │                    │(example.com)   │      │
        │  │  │  │                    └────────────────┘      │
        │  │  │  │                                            │
        └──┴──┴──┴────────────────────────────────────────────┘
```

### 3.3 Iterative Query

The Recursive Resolver queries each DNS server in sequence and directly queries the next server with the information received.

```
Iterative Query Process

                    Recursive Resolver
                          │
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    │ (2) "Where is com?" │                     │
    │ ─────────────────►  │                     │
    │                     ▼                     │
    │              ┌────────────┐               │
    │              │   Root     │               │
    │ ◄─────────── │   Server   │               │
    │ (3) "a.gtld-servers.net"                  │
    │              └────────────┘               │
    │                                           │
    │ (4) "Where is example.com?"               │
    │ ─────────────────────────────►            │
    │                              ▼            │
    │                       ┌────────────┐      │
    │                       │ .com TLD   │      │
    │ ◄───────────────────  │ Server     │      │
    │ (5) "ns1.example.com"                     │
    │                       └────────────┘      │
    │                                           │
    │ (6) "What's www.example.com's IP?"        │
    │ ─────────────────────────────────────►    │
    │                                      ▼    │
    │                              ┌────────────┐
    │                              │Authoritative│
    │ ◄─────────────────────────── │   Server   │
    │ (7) "93.184.216.34"          └────────────┘
    │                                           │
    └───────────────────────────────────────────┘
```

### 3.4 DNS Query/Response Messages

```
DNS Message Structure

┌────────────────────────────────────────────────────────────────┐
│                         Header                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ID (16 bits) - Query/response matching                  │   │
│  │ Flags: QR, Opcode, AA, TC, RD, RA, Z, RCODE             │   │
│  │ QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT                      │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                        Question                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ QNAME: www.example.com (query domain)                   │   │
│  │ QTYPE: A (query type)                                   │   │
│  │ QCLASS: IN (Internet)                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                         Answer                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ NAME: www.example.com                                   │   │
│  │ TYPE: A                                                 │   │
│  │ CLASS: IN                                               │   │
│  │ TTL: 300 (seconds)                                      │   │
│  │ RDLENGTH: 4                                             │   │
│  │ RDATA: 93.184.216.34                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                       Authority                                 │
│  (Authoritative server information)                             │
├────────────────────────────────────────────────────────────────┤
│                       Additional                                │
│  (Additional information - e.g., authoritative server IP)       │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. DNS Record Types

### 4.1 Common DNS Records

| Record | Meaning | Description |
|--------|---------|-------------|
| A | Address | IPv4 address mapping |
| AAAA | IPv6 Address | IPv6 address mapping |
| CNAME | Canonical Name | Domain alias |
| MX | Mail Exchanger | Mail server specification |
| NS | Name Server | Name server specification |
| TXT | Text | Text information (SPF, DKIM, etc.) |
| PTR | Pointer | Reverse lookup (IP → domain) |
| SOA | Start of Authority | Zone authority information |
| SRV | Service | Service location information |
| CAA | Certification Authority | Certificate issuance authority |

### 4.2 A Record

Maps a domain name to an IPv4 address.

```
A Record Examples

example.com.     IN  A     93.184.216.34
www.example.com. IN  A     93.184.216.34
api.example.com. IN  A     93.184.216.35

Load Balancing (Round Robin):
www.example.com. IN  A     93.184.216.34
www.example.com. IN  A     93.184.216.35
www.example.com. IN  A     93.184.216.36

Query Result:
$ dig www.example.com A

;; ANSWER SECTION:
www.example.com.    300    IN    A    93.184.216.34
```

### 4.3 AAAA Record

Maps a domain name to an IPv6 address.

```
AAAA Record Examples

example.com.     IN  AAAA  2606:2800:220:1:248:1893:25c8:1946
www.example.com. IN  AAAA  2606:2800:220:1:248:1893:25c8:1946

Query Result:
$ dig www.example.com AAAA

;; ANSWER SECTION:
www.example.com.    300    IN    AAAA    2606:2800:220:1:248:1893:25c8:1946
```

### 4.4 CNAME Record

Maps one domain to another domain (alias).

```
CNAME Record Examples

www.example.com.    IN  CNAME  example.com.
blog.example.com.   IN  CNAME  blogger.l.google.com.
shop.example.com.   IN  CNAME  shops.myshopify.com.

CNAME Chaining:
alias.example.com.  IN  CNAME  www.example.com.
www.example.com.    IN  CNAME  example.com.
example.com.        IN  A      93.184.216.34

Query Process:
alias.example.com
    → www.example.com (CNAME)
    → example.com (CNAME)
    → 93.184.216.34 (A)

Notes:
- Cannot use CNAME on root domain (example.com)
- Cannot be used with MX, NS records
- Use ALIAS/ANAME record instead (some DNS providers)
```

### 4.5 MX Record

Specifies mail servers for the domain.

```
MX Record Examples

example.com.  IN  MX  10  mail1.example.com.
example.com.  IN  MX  20  mail2.example.com.
example.com.  IN  MX  30  mail3.backup.com.

Priority:
- Lower number = higher priority
- Try in order: 10 → 20 → 30

Google Workspace Example:
example.com.  IN  MX  1   aspmx.l.google.com.
example.com.  IN  MX  5   alt1.aspmx.l.google.com.
example.com.  IN  MX  5   alt2.aspmx.l.google.com.
example.com.  IN  MX  10  alt3.aspmx.l.google.com.
example.com.  IN  MX  10  alt4.aspmx.l.google.com.

Query:
$ dig example.com MX

;; ANSWER SECTION:
example.com.    300    IN    MX    10 mail1.example.com.
example.com.    300    IN    MX    20 mail2.example.com.
```

### 4.6 NS Record

Specifies name servers managing the domain.

```
NS Record Examples

example.com.    IN  NS    ns1.example.com.
example.com.    IN  NS    ns2.example.com.

Delegation:
sub.example.com.  IN  NS    ns1.subdomain.com.
sub.example.com.  IN  NS    ns2.subdomain.com.

Glue Records (when name server is in the same domain):
example.com.      IN  NS    ns1.example.com.
ns1.example.com.  IN  A     192.0.2.1
ns2.example.com.  IN  A     192.0.2.2
```

### 4.7 TXT Record

Stores text information. Primarily used for authentication and verification.

```
TXT Record Uses

1. SPF (Sender Policy Framework) - Email sender authentication
example.com.  IN  TXT  "v=spf1 include:_spf.google.com ~all"

2. DKIM (DomainKeys Identified Mail) - Email signature
google._domainkey.example.com.  IN  TXT  "v=DKIM1; k=rsa; p=MIGf..."

3. DMARC (Domain-based Message Authentication)
_dmarc.example.com.  IN  TXT  "v=DMARC1; p=reject; rua=mailto:dmarc@example.com"

4. Domain ownership verification (Google, MS, etc.)
example.com.  IN  TXT  "google-site-verification=..."
example.com.  IN  TXT  "MS=ms12345678"

5. Other service settings
example.com.  IN  TXT  "facebook-domain-verification=..."
```

### 4.8 PTR Record

Maps IP addresses to domain names (reverse lookup).

```
PTR Record Examples

Reverse lookup zone:
IP: 93.184.216.34
Reverse domain: 34.216.184.93.in-addr.arpa

PTR Record:
34.216.184.93.in-addr.arpa.  IN  PTR  www.example.com.

IPv6 Reverse:
IP: 2001:db8::1
Reverse domain: 1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa

Uses:
- Mail server verification (spam filters)
- Display IP as domain in logs
- Security checks

Query:
$ dig -x 93.184.216.34

;; ANSWER SECTION:
34.216.184.93.in-addr.arpa. 3600 IN PTR www.example.com.
```

### 4.9 SOA Record

Defines authority information for a DNS zone.

```
SOA Record Example

example.com. IN SOA ns1.example.com. admin.example.com. (
    2024010101   ; Serial Number (YYYYMMDDNN)
    3600         ; Refresh (1 hour)
    600          ; Retry (10 minutes)
    604800       ; Expire (1 week)
    86400        ; Minimum TTL (1 day)
)

Field Descriptions:
- Primary NS: ns1.example.com (primary name server)
- Admin Email: admin@example.com (written as admin.example.com)
- Serial: Increment on each change (for secondary server sync)
- Refresh: How often secondary checks primary
- Retry: Retry interval if Refresh fails
- Expire: How long data is valid if primary is unreachable
- Minimum TTL: Cache time for negative responses (NXDOMAIN)
```

---

## 5. DNS Caching

### 5.1 Caching Layers

```
DNS Caching Layers

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Level 1: Browser Cache                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Browser's own DNS cache                                  │ │
│  │ • Chrome: chrome://net-internals/#dns                      │ │
│  │ • Short TTL (usually minutes)                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        │                                         │
│                        ▼                                         │
│  Level 2: Operating System Cache                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • OS DNS resolver cache                                    │ │
│  │ • Windows: ipconfig /displaydns                            │ │
│  │ • macOS: dscacheutil -cachedump -entries                   │ │
│  │ • Linux: systemd-resolved, etc.                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        │                                         │
│                        ▼                                         │
│  Level 3: Recursive Resolver Cache                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • ISP or public DNS server cache                           │ │
│  │ • Shared by many users                                     │ │
│  │ • TTL-based cache validity period                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                        │                                         │
│                        ▼                                         │
│  Level 4: Authoritative Server                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Actual query on cache miss                               │ │
│  │ • Provides authoritative response                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 TTL (Time To Live)

```
Role of TTL

TTL = Time a DNS record can be cached (in seconds)

example.com.  300  IN  A  93.184.216.34
              ↑
             TTL (300 seconds = 5 minutes)

TTL Strategy:
┌──────────────────────────────────────────────────────────────────┐
│ Situation                 │ Recommended │ Reason                │
├──────────────────────────────────────────────────────────────────┤
│ Normal operation          │ 3600-86400  │ Maximize cache        │
│ (1 hour - 1 day)          │             │ efficiency            │
├──────────────────────────────────────────────────────────────────┤
│ Migration planned         │ 300-600     │ Fast propagation      │
│ (5 min - 10 min)          │             │                       │
├──────────────────────────────────────────────────────────────────┤
│ Failover/HA               │ 60-300      │ Quick failure         │
│ (1 min - 5 min)           │             │ response              │
├──────────────────────────────────────────────────────────────────┤
│ Just before change        │ 60          │ Fast old cache        │
│ (1 minute)                │             │ expiration            │
└──────────────────────────────────────────────────────────────────┘

TTL Trade-offs:
Low TTL:
  + Fast propagation of changes
  - Increased DNS queries, server load

High TTL:
  + Cache efficiency, fast response
  - Slow change propagation
```

### 5.3 Cache Flushing

```bash
# Windows
ipconfig /flushdns

# macOS
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder

# Linux (systemd-resolved)
sudo systemd-resolve --flush-caches

# Chrome browser
chrome://net-internals/#dns → Clear host cache

# Firefox
about:networking#dns → Clear DNS Cache
```

---

## 6. DNS Tools

### 6.1 nslookup

```bash
# Basic lookup
nslookup google.com

# Specific record lookup
nslookup -type=MX google.com
nslookup -type=A google.com
nslookup -type=AAAA google.com

# Use specific DNS server
nslookup google.com 8.8.8.8

# Interactive mode
nslookup
> set type=MX
> google.com
> exit

Output Example:
Server:    8.8.8.8
Address:   8.8.8.8#53

Non-authoritative answer:
Name:      google.com
Address:   142.250.196.78
```

### 6.2 dig

```bash
# Basic lookup
dig google.com

# Specific record lookup
dig google.com A
dig google.com AAAA
dig google.com MX
dig google.com NS
dig google.com TXT

# Short output
dig +short google.com

# Detailed output (trace)
dig +trace google.com

# Use specific DNS server
dig @8.8.8.8 google.com

# Reverse lookup
dig -x 142.250.196.78

# Show TTL
dig +ttlid google.com

# Query all records
dig google.com ANY

Output Example:
;; ANSWER SECTION:
google.com.        137    IN    A    142.250.196.78

;; Query time: 15 msec
;; SERVER: 8.8.8.8#53(8.8.8.8)
```

### 6.3 host

```bash
# Basic lookup
host google.com

# Specify record type
host -t MX google.com
host -t NS google.com
host -t TXT google.com

# Verbose output
host -v google.com

# Reverse lookup
host 142.250.196.78

Output Example:
google.com has address 142.250.196.78
google.com has IPv6 address 2404:6800:4004:821::200e
google.com mail is handled by 10 smtp.google.com.
```

### 6.4 Interpreting dig Output

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

Interpretation:
- status: NOERROR → Success
- flags: qr(response), rd(recursion desired), ra(recursion available)
- QUESTION: Query content
- ANSWER: Response (A record, TTL 86400 sec, IP 93.184.216.34)
- Query time: Response time (25ms)
- SERVER: Responding DNS server
```

### 6.5 Public DNS Servers

| Provider | IPv4 | IPv6 | Features |
|----------|------|------|----------|
| Google | 8.8.8.8, 8.8.4.4 | 2001:4860:4860::8888 | Global, fast |
| Cloudflare | 1.1.1.1, 1.0.0.1 | 2606:4700:4700::1111 | Privacy-focused |
| Quad9 | 9.9.9.9, 149.112.112.112 | 2620:fe::fe | Security-focused |
| OpenDNS | 208.67.222.222 | 2620:119:35::35 | Filtering options |

---

## 7. Practice Problems

### Problem 1: Domain Structure Analysis

Identify each part of the following domains.

```
a) www.shop.amazon.co.uk
b) mail.google.com
c) api.v2.example.org
```

### Problem 2: DNS Record Matching

Select the appropriate DNS record type for each situation.

a) Specify web server's IPv4 address
b) Specify mail server
c) Redirect www to base domain
d) Domain ownership authentication
e) Specify name server
f) Specify IPv6 address

### Problem 3: dig Output Analysis

Analyze the following dig output.

```
;; ANSWER SECTION:
example.com.        600    IN    MX    10 mail1.example.com.
example.com.        600    IN    MX    20 mail2.example.com.
example.com.        600    IN    MX    30 backup.mail.com.
```

a) What is the TTL?
b) Which mail server is used first?
c) What happens if all mail servers are down?

### Problem 4: DNS Query Practice

Execute the following commands and analyze the results.

```bash
dig google.com A
dig google.com MX
dig +trace google.com
```

---

## Answers

### Problem 1 Answers

a) www.shop.amazon.co.uk
```
uk    : TLD (ccTLD - United Kingdom)
co    : Second-level domain (for companies)
amazon: Registered domain
shop  : Subdomain
www   : Subdomain (hostname)
```

b) mail.google.com
```
com   : TLD (gTLD)
google: SLD (registered domain)
mail  : Subdomain
```

c) api.v2.example.org
```
org    : TLD (gTLD)
example: SLD
v2     : Subdomain
api    : Subdomain
```

### Problem 2 Answers

- a) IPv4 address → **A Record**
- b) Mail server → **MX Record**
- c) Redirect → **CNAME Record**
- d) Ownership authentication → **TXT Record**
- e) Name server → **NS Record**
- f) IPv6 address → **AAAA Record**

### Problem 3 Answers

a) TTL = **600 seconds (10 minutes)**
b) Primary mail server: **mail1.example.com** (priority 10)
c) Attempt backup.mail.com, then fail → **Mail delivery fails (bounce)**

### Problem 4 Answers

Results vary by environment, but check:
- A record: Google's IP addresses (may be multiple)
- MX record: Google's mail servers (aspmx.l.google.com, etc.)
- +trace: Query sequence from root → .com TLD → google.com authoritative server

---

## 8. References

### RFC Documents

- RFC 1034 - Domain Names: Concepts and Facilities
- RFC 1035 - Domain Names: Implementation and Specification
- RFC 8484 - DNS Queries over HTTPS (DoH)
- RFC 7858 - DNS over TLS (DoT)

### Online Tools

- [DNS Checker](https://dnschecker.org/) - Check DNS propagation worldwide
- [MX Toolbox](https://mxtoolbox.com/) - DNS/mail diagnostics
- [whatsmydns.net](https://www.whatsmydns.net/) - DNS lookup
- [IntoDNS](https://intodns.com/) - DNS configuration check

### Learning Resources

- [How DNS Works (Comic)](https://howdns.works/)
- [Cloudflare Learning: DNS](https://www.cloudflare.com/learning/dns/)
- [Google Public DNS](https://developers.google.com/speed/public-dns)

---

---

**Previous**: [UDP and Ports](./11_UDP_and_Ports.md) | **Next**: [HTTP and HTTPS](./13_HTTP_and_HTTPS.md)
