# 고급 네트워킹

## 학습 목표

이 문서를 통해 다음을 학습합니다:

- VLAN 설정과 802.1Q 태깅
- NIC Bonding/Teaming
- 브릿지 네트워킹
- iptables와 nftables 심화

**난이도**: ⭐⭐⭐⭐ (고급)

---

## 목차

1. [VLAN 설정](#1-vlan-설정)
2. [NIC Bonding](#2-nic-bonding)
3. [브릿지 네트워킹](#3-브릿지-네트워킹)
4. [iptables 심화](#4-iptables-심화)
5. [nftables](#5-nftables)
6. [고급 라우팅](#6-고급-라우팅)
7. [Traffic Control (tc)](#7-traffic-control-tc)

---

## 1. VLAN 설정

### VLAN 개요

```
┌─────────────────────────────────────────────────────────────┐
│                        물리 스위치                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Port 1    Port 2    Port 3    Port 4    Port 5     │   │
│  │  VLAN 10   VLAN 10   VLAN 20   VLAN 20   Trunk     │   │
│  └─────────────────────────────────────────────────────┘   │
│        │         │         │         │         │           │
└────────┼─────────┼─────────┼─────────┼─────────┼───────────┘
         │         │         │         │         │
     ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
     │Host A │ │Host B │ │Host C │ │Host D │ │Server │
     │192.168│ │192.168│ │192.168│ │192.168│ │eth0.10│
     │.10.1  │ │.10.2  │ │.20.1  │ │.20.2  │ │eth0.20│
     └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

### VLAN 인터페이스 생성

```bash
# 모듈 확인
lsmod | grep 8021q

# 모듈 로드
sudo modprobe 8021q
echo "8021q" | sudo tee /etc/modules-load.d/8021q.conf

# VLAN 인터페이스 생성 (ip 명령)
sudo ip link add link eth0 name eth0.10 type vlan id 10
sudo ip link add link eth0 name eth0.20 type vlan id 20

# IP 주소 할당
sudo ip addr add 192.168.10.1/24 dev eth0.10
sudo ip addr add 192.168.20.1/24 dev eth0.20

# 인터페이스 활성화
sudo ip link set eth0.10 up
sudo ip link set eth0.20 up

# VLAN 확인
cat /proc/net/vlan/config
ip -d link show eth0.10
```

### Ubuntu Netplan 설정

```yaml
# /etc/netplan/01-vlans.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no

  vlans:
    eth0.10:
      id: 10
      link: eth0
      addresses:
        - 192.168.10.1/24
      routes:
        - to: 10.0.0.0/8
          via: 192.168.10.254

    eth0.20:
      id: 20
      link: eth0
      addresses:
        - 192.168.20.1/24
```

```bash
sudo netplan apply
```

### RHEL/CentOS NetworkManager 설정

```bash
# VLAN 연결 생성
sudo nmcli connection add type vlan con-name eth0.10 dev eth0 id 10
sudo nmcli connection modify eth0.10 ipv4.addresses 192.168.10.1/24
sudo nmcli connection modify eth0.10 ipv4.method manual
sudo nmcli connection up eth0.10

# 또는 ifcfg 파일 사용
# /etc/sysconfig/network-scripts/ifcfg-eth0.10
```

```ini
# /etc/sysconfig/network-scripts/ifcfg-eth0.10
DEVICE=eth0.10
BOOTPROTO=none
ONBOOT=yes
VLAN=yes
IPADDR=192.168.10.1
NETMASK=255.255.255.0
```

---

## 2. NIC Bonding

### Bonding 모드

| 모드 | 이름 | 설명 | 스위치 설정 |
|------|------|------|-------------|
| 0 | balance-rr | 라운드 로빈 | 불필요 |
| 1 | active-backup | 액티브-백업 | 불필요 |
| 2 | balance-xor | XOR 해시 | 불필요 |
| 3 | broadcast | 브로드캐스트 | 불필요 |
| 4 | 802.3ad | LACP | LACP 필요 |
| 5 | balance-tlb | 전송 부하 분산 | 불필요 |
| 6 | balance-alb | 적응형 부하 분산 | 불필요 |

### Ubuntu Netplan Bonding

```yaml
# /etc/netplan/01-bonding.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
    enp4s0:
      dhcp4: no

  bonds:
    bond0:
      interfaces:
        - enp3s0
        - enp4s0
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      parameters:
        mode: 802.3ad
        lacp-rate: fast
        mii-monitor-interval: 100
        transmit-hash-policy: layer3+4
```

### RHEL/CentOS NetworkManager Bonding

```bash
# Bonding 연결 생성
sudo nmcli connection add type bond con-name bond0 ifname bond0 \
    bond.options "mode=802.3ad,miimon=100,lacp_rate=fast"

# 슬레이브 추가
sudo nmcli connection add type ethernet con-name bond0-slave1 \
    ifname enp3s0 master bond0
sudo nmcli connection add type ethernet con-name bond0-slave2 \
    ifname enp4s0 master bond0

# IP 설정
sudo nmcli connection modify bond0 ipv4.addresses 192.168.1.100/24
sudo nmcli connection modify bond0 ipv4.gateway 192.168.1.1
sudo nmcli connection modify bond0 ipv4.method manual

# 활성화
sudo nmcli connection up bond0
```

### Bonding 상태 확인

```bash
# Bonding 상태
cat /proc/net/bonding/bond0

# 인터페이스 정보
ip -d link show bond0
ip link show master bond0

# 슬레이브 상태
cat /sys/class/net/bond0/bonding/slaves
cat /sys/class/net/bond0/bonding/mode
```

---

## 3. 브릿지 네트워킹

### 브릿지 생성

```bash
# 브릿지 생성
sudo ip link add name br0 type bridge
sudo ip link set br0 up

# 인터페이스 추가
sudo ip link set eth0 master br0
sudo ip link set eth1 master br0

# 브릿지에 IP 할당
sudo ip addr add 192.168.1.100/24 dev br0

# 상태 확인
bridge link show
brctl show  # bridge-utils 패키지
```

### Ubuntu Netplan 브릿지

```yaml
# /etc/netplan/01-bridge.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
    enp4s0:
      dhcp4: no

  bridges:
    br0:
      interfaces:
        - enp3s0
        - enp4s0
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      parameters:
        stp: true
        forward-delay: 4
```

### 브릿지 + VLAN 조합

```yaml
# /etc/netplan/01-bridge-vlan.yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no

  vlans:
    eth0.10:
      id: 10
      link: eth0

  bridges:
    br-vlan10:
      interfaces:
        - eth0.10
      addresses:
        - 192.168.10.1/24
```

---

## 4. iptables 심화

### iptables 체인과 테이블

```
┌─────────────────────────────────────────────────────────────┐
│                    Packet Flow                              │
│                                                             │
│  PREROUTING ──▶ INPUT ──────▶ Local Process                │
│       │                              │                      │
│       │                              │                      │
│       ▼                              ▼                      │
│   FORWARD ◀───────────────────── OUTPUT                    │
│       │                              │                      │
│       └───────▶ POSTROUTING ◀───────┘                      │
│                      │                                      │
│                      ▼                                      │
│                   Network                                   │
└─────────────────────────────────────────────────────────────┘

테이블:
- filter: 패킷 필터링 (INPUT, FORWARD, OUTPUT)
- nat: 주소 변환 (PREROUTING, OUTPUT, POSTROUTING)
- mangle: 패킷 수정 (모든 체인)
- raw: 연결 추적 제외 (PREROUTING, OUTPUT)
```

### 고급 규칙 예시

```bash
# 상태 기반 필터링
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -m state --state NEW -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -m state --state INVALID -j DROP

# 연결 제한 (초당 연결 수)
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 50 -j REJECT

# Rate limiting (분당 요청 수)
iptables -A INPUT -p tcp --dport 22 -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP

# 포트 범위
iptables -A INPUT -p tcp -m multiport --dports 80,443,8080:8090 -j ACCEPT

# IP 범위
iptables -A INPUT -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -m iprange --src-range 10.0.0.1-10.0.0.100 -j ACCEPT

# 시간 기반 규칙
iptables -A INPUT -p tcp --dport 22 -m time --timestart 09:00 --timestop 18:00 --weekdays Mon,Tue,Wed,Thu,Fri -j ACCEPT
```

### NAT 설정

```bash
# IP 포워딩 활성화
echo 1 > /proc/sys/net/ipv4/ip_forward
echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf

# SNAT (Source NAT) - 내부 → 외부
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j MASQUERADE
# 또는 고정 IP 사용 시
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j SNAT --to-source 203.0.113.1

# DNAT (Destination NAT) - 포트 포워딩
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j DNAT --to-destination 192.168.1.10:8080
iptables -A FORWARD -p tcp -d 192.168.1.10 --dport 8080 -j ACCEPT
```

### iptables 저장/복원

```bash
# 현재 규칙 저장
iptables-save > /etc/iptables/rules.v4
ip6tables-save > /etc/iptables/rules.v6

# 규칙 복원
iptables-restore < /etc/iptables/rules.v4

# Ubuntu: iptables-persistent 패키지
sudo apt install iptables-persistent
sudo netfilter-persistent save

# RHEL/CentOS
sudo service iptables save
```

---

## 5. nftables

### nftables 기본

```bash
# nftables 상태 확인
sudo nft list ruleset

# 테이블 생성
sudo nft add table inet filter

# 체인 생성
sudo nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
sudo nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
sudo nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }
```

### nftables 규칙

```bash
# 기본 규칙
nft add rule inet filter input ct state established,related accept
nft add rule inet filter input ct state invalid drop
nft add rule inet filter input iif lo accept

# SSH 허용
nft add rule inet filter input tcp dport 22 accept

# 여러 포트
nft add rule inet filter input tcp dport { 80, 443, 8080 } accept

# IP 범위
nft add rule inet filter input ip saddr 192.168.1.0/24 accept

# Rate limiting
nft add rule inet filter input tcp dport 22 meter ssh-meter { ip saddr limit rate 3/minute } accept

# 로깅
nft add rule inet filter input log prefix "INPUT DROP: " counter drop
```

### nftables 설정 파일

```bash
# /etc/nftables.conf
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # 기존 연결 허용
        ct state established,related accept
        ct state invalid drop

        # 로컬 인터페이스
        iif lo accept

        # ICMP
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept

        # SSH
        tcp dport 22 accept

        # HTTP/HTTPS
        tcp dport { 80, 443 } accept

        # 로깅 후 드롭
        log prefix "INPUT DROP: " counter drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
        ct state established,related accept
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table ip nat {
    chain prerouting {
        type nat hook prerouting priority -100;

        # 포트 포워딩
        tcp dport 80 dnat to 192.168.1.10:8080
    }

    chain postrouting {
        type nat hook postrouting priority 100;

        # 마스커레이드
        oifname "eth0" masquerade
    }
}
```

```bash
# 설정 적용
sudo nft -f /etc/nftables.conf

# 서비스 활성화
sudo systemctl enable nftables
```

### iptables vs nftables 비교

| iptables | nftables |
|----------|----------|
| `iptables -A INPUT` | `nft add rule inet filter input` |
| `iptables -t nat` | `nft add table ip nat` |
| `--dport 80` | `tcp dport 80` |
| `-m multiport --dports 80,443` | `tcp dport { 80, 443 }` |
| `-m state --state ESTABLISHED` | `ct state established` |
| `-j ACCEPT` | `accept` |
| `-j MASQUERADE` | `masquerade` |

---

## 6. 고급 라우팅

### Policy Routing

```bash
# 라우팅 테이블 추가
echo "100 isp1" >> /etc/iproute2/rt_tables
echo "200 isp2" >> /etc/iproute2/rt_tables

# ISP1 라우팅 테이블 설정
ip route add default via 203.0.113.1 table isp1
ip route add 192.168.1.0/24 dev eth0 table isp1

# ISP2 라우팅 테이블 설정
ip route add default via 198.51.100.1 table isp2
ip route add 192.168.1.0/24 dev eth0 table isp2

# 규칙 추가 (소스 기반)
ip rule add from 192.168.1.0/24 lookup isp1
ip rule add from 10.0.0.0/8 lookup isp2

# 규칙 확인
ip rule show
ip route show table isp1
```

### 멀티패스 라우팅

```bash
# ECMP (Equal Cost Multi-Path)
ip route add default \
    nexthop via 203.0.113.1 weight 1 \
    nexthop via 198.51.100.1 weight 1

# 가중치 기반 부하 분산
ip route add default \
    nexthop via 203.0.113.1 weight 3 \
    nexthop via 198.51.100.1 weight 1
```

### Network Namespaces

```bash
# 네임스페이스 생성
ip netns add ns1
ip netns add ns2

# 네임스페이스 목록
ip netns list

# veth 쌍 생성
ip link add veth0 type veth peer name veth1

# 네임스페이스에 할당
ip link set veth0 netns ns1
ip link set veth1 netns ns2

# 네임스페이스 내에서 설정
ip netns exec ns1 ip addr add 10.0.0.1/24 dev veth0
ip netns exec ns1 ip link set veth0 up
ip netns exec ns1 ip link set lo up

ip netns exec ns2 ip addr add 10.0.0.2/24 dev veth1
ip netns exec ns2 ip link set veth1 up
ip netns exec ns2 ip link set lo up

# 연결 테스트
ip netns exec ns1 ping 10.0.0.2
```

---

## 7. Traffic Control (tc)

### 대역폭 제한

```bash
# 출력 대역폭 제한 (1Mbit/s)
tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms

# 설정 확인
tc qdisc show dev eth0

# 삭제
tc qdisc del dev eth0 root
```

### HTB (Hierarchical Token Bucket)

```bash
# HTB qdisc 생성
tc qdisc add dev eth0 root handle 1: htb default 30

# 전체 대역폭 클래스
tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit ceil 100mbit

# 하위 클래스 (서비스별)
tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 100mbit  # 웹
tc class add dev eth0 parent 1:1 classid 1:20 htb rate 30mbit ceil 100mbit  # 메일
tc class add dev eth0 parent 1:1 classid 1:30 htb rate 20mbit ceil 100mbit  # 기본

# 필터로 트래픽 분류
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 \
    match ip dport 80 0xffff flowid 1:10
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 \
    match ip dport 443 0xffff flowid 1:10
tc filter add dev eth0 protocol ip parent 1:0 prio 2 u32 \
    match ip dport 25 0xffff flowid 1:20
```

### 지연 및 패킷 손실 시뮬레이션

```bash
# 지연 추가 (100ms)
tc qdisc add dev eth0 root netem delay 100ms

# 지연 + 변동
tc qdisc add dev eth0 root netem delay 100ms 20ms distribution normal

# 패킷 손실 (1%)
tc qdisc add dev eth0 root netem loss 1%

# 복합 설정
tc qdisc add dev eth0 root netem delay 100ms 20ms loss 1% duplicate 0.1%
```

---

## 연습 문제

### 문제 1: VLAN 설정

eth0 인터페이스에 VLAN 100을 추가하고 192.168.100.1/24 주소를 할당하세요.

### 문제 2: nftables 방화벽

다음 조건의 nftables 규칙을 작성하세요:
- SSH(22), HTTP(80), HTTPS(443) 허용
- 192.168.1.0/24에서 오는 모든 트래픽 허용
- 나머지 입력 트래픽 차단 및 로깅

### 문제 3: Traffic Control

eth0 인터페이스의 출력 대역폭을 10Mbit/s로 제한하세요.

---

## 정답

### 문제 1 정답

```bash
# VLAN 인터페이스 생성
sudo ip link add link eth0 name eth0.100 type vlan id 100

# IP 할당
sudo ip addr add 192.168.100.1/24 dev eth0.100

# 활성화
sudo ip link set eth0.100 up

# 확인
ip addr show eth0.100
```

### 문제 2 정답

```bash
# /etc/nftables.conf
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        ct state established,related accept
        ct state invalid drop
        iif lo accept

        ip saddr 192.168.1.0/24 accept

        tcp dport { 22, 80, 443 } accept

        log prefix "INPUT DROP: " counter drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

```bash
sudo nft -f /etc/nftables.conf
```

### 문제 3 정답

```bash
# TBF (Token Bucket Filter) 사용
sudo tc qdisc add dev eth0 root tbf rate 10mbit burst 32kbit latency 400ms

# 확인
tc qdisc show dev eth0
```

---

## 다음 단계

- [24_Cloud_Integration.md](./24_Cloud_Integration.md) - cloud-init, AWS CLI

---

## 참고 자료

- [Linux Advanced Routing & Traffic Control](https://lartc.org/)
- [nftables Wiki](https://wiki.nftables.org/)
- [Netplan Documentation](https://netplan.io/)
- `man ip`, `man nft`, `man tc`
