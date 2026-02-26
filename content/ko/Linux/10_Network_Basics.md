# 네트워크 기초

**이전**: [셸 스크립팅](./09_Shell_Scripting.md) | **다음**: [시스템 모니터링](./11_System_Monitoring.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. IP 주소 체계, 서브넷팅, 주요 포트 할당 설명
2. ip addr, ip link, ip route를 사용하여 네트워크 인터페이스 확인 및 설정
3. ping, traceroute, mtr로 네트워크 연결 테스트
4. dig, nslookup, host를 사용하여 DNS 조회 수행
5. ss, netstat, lsof로 열린 포트와 소켓 연결 검사
6. 키 기반 인증을 사용하여 SSH로 원격 시스템에 안전하게 연결
7. scp, rsync, sftp를 사용하여 시스템 간 파일 전송
8. Netplan(Ubuntu)과 NetworkManager(CentOS)로 영구적인 네트워크 설정 구성

---

네트워킹은 리눅스 서버를 유용하게 만드는 핵심 요소입니다. 웹 애플리케이션이 왜 접근 불가능한지 진단하거나, 원격 관리를 위한 SSH 접근을 설정하거나, 프로덕션 서버에 정적 IP를 설정하는 등 네트워킹 명령어는 시스템 관리자가 가장 자주 사용하는 도구 중 하나입니다.

## 1. 네트워크 기본 개념

### IP 주소

```
IPv4: 192.168.1.100
      └─┬─┘ └─┬─┘
     네트워크  호스트

IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
```

### 사설 IP 대역

| 클래스 | 대역 | 용도 |
|--------|------|------|
| A | 10.0.0.0/8 | 대규모 네트워크 |
| B | 172.16.0.0/12 | 중규모 네트워크 |
| C | 192.168.0.0/16 | 가정/소규모 |

### 서브넷 마스크

```
IP:      192.168.1.100
서브넷:   255.255.255.0   (/24)
         └──네트워크──┘ └호스트┘

CIDR 표기: 192.168.1.0/24
→ 256개 주소 (192.168.1.0 ~ 192.168.1.255)
```

### 주요 포트

| 포트 | 서비스 |
|------|--------|
| 22 | SSH |
| 80 | HTTP |
| 443 | HTTPS |
| 3306 | MySQL |
| 5432 | PostgreSQL |
| 6379 | Redis |

---

## 2. 네트워크 설정 확인

### ip addr - IP 주소 확인

```bash
# 모든 인터페이스
ip addr
ip a

# 특정 인터페이스
ip addr show eth0
```

출력:
```
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP
    link/ether 00:11:22:33:44:55 brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::1/64 scope link
       valid_lft forever preferred_lft forever
```

### ip link - 인터페이스 상태

```bash
# 인터페이스 목록
ip link

# 인터페이스 활성화
sudo ip link set eth0 up

# 인터페이스 비활성화
sudo ip link set eth0 down
```

### ip route - 라우팅 테이블

```bash
# 라우팅 확인
ip route
ip r
```

출력:
```
default via 192.168.1.1 dev eth0 proto dhcp metric 100
192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.100
```

### ifconfig (레거시)

```bash
# 설치 필요할 수 있음
sudo apt install net-tools    # Ubuntu
sudo dnf install net-tools    # CentOS

# 사용
ifconfig
ifconfig eth0
```

### hostname - 호스트명

```bash
# 호스트명 확인
hostname
hostname -I    # IP 주소만

# 호스트명 변경 (임시)
sudo hostname new-hostname

# 호스트명 변경 (영구)
sudo hostnamectl set-hostname new-hostname
```

---

## 3. 연결 테스트

### ping - 연결 확인

```bash
# 기본 ping
ping google.com

# 횟수 지정
ping -c 4 google.com

# 간격 지정
ping -i 0.5 192.168.1.1
```

### traceroute - 경로 추적

```bash
# 설치
sudo apt install traceroute    # Ubuntu
sudo dnf install traceroute    # CentOS

# 사용
traceroute google.com
traceroute -n 8.8.8.8    # 이름 해석 안 함
```

### mtr - 통합 도구

```bash
# 설치
sudo apt install mtr    # Ubuntu
sudo dnf install mtr    # CentOS

# 사용 (ping + traceroute)
mtr google.com
mtr -r -c 10 google.com    # 보고서 모드
```

---

## 4. DNS 조회

### nslookup

```bash
# 기본 조회
nslookup google.com

# 특정 DNS 서버 사용
nslookup google.com 8.8.8.8

# 레코드 타입 지정
nslookup -type=MX google.com
```

### dig

```bash
# 기본 조회
dig google.com

# 간단한 출력
dig +short google.com

# 특정 레코드
dig google.com MX
dig google.com TXT

# 특정 DNS 서버
dig @8.8.8.8 google.com

# 역방향 조회
dig -x 8.8.8.8
```

### host

```bash
# 간단한 조회
host google.com
host -t MX google.com
```

### /etc/hosts

로컬 DNS 설정입니다.

```bash
cat /etc/hosts
```

```
127.0.0.1   localhost
192.168.1.50   myserver.local myserver
```

### /etc/resolv.conf

DNS 서버 설정입니다.

```bash
cat /etc/resolv.conf
```

```
nameserver 8.8.8.8
nameserver 8.8.4.4
search example.com
```

---

## 5. 포트 확인

### ss - 소켓 통계

```bash
# 열린 포트 확인
ss -tuln

# TCP 연결
ss -t

# 리스닝 포트
ss -l

# 프로세스 정보 포함
ss -tulnp

# 특정 포트
ss -tuln | grep :80
```

옵션:
| 옵션 | 설명 |
|------|------|
| `-t` | TCP |
| `-u` | UDP |
| `-l` | LISTEN 상태만 |
| `-n` | 숫자로 표시 |
| `-p` | 프로세스 정보 |

### netstat (레거시)

```bash
# 열린 포트
netstat -tuln

# 모든 연결
netstat -an

# 프로세스 포함
sudo netstat -tulnp
```

### lsof - 열린 파일/포트

```bash
# 특정 포트를 사용하는 프로세스
sudo lsof -i :80

# 특정 프로세스의 네트워크 연결
sudo lsof -i -a -p 1234

# 모든 네트워크 연결
sudo lsof -i
```

---

## 6. SSH - 원격 접속

### 기본 접속

```bash
# 기본 접속
ssh user@hostname
ssh user@192.168.1.100

# 포트 지정
ssh -p 2222 user@hostname

# 상세 출력
ssh -v user@hostname
```

### SSH 키 인증

```bash
# 1. 키 생성
ssh-keygen -t rsa -b 4096

# 또는 ed25519 (권장)
ssh-keygen -t ed25519

# 2. 공개키 복사
ssh-copy-id user@hostname

# 3. 키로 접속
ssh user@hostname
```

### SSH 키 파일

```
~/.ssh/
├── id_rsa           # 개인키 (절대 공유 금지)
├── id_rsa.pub       # 공개키 (서버에 등록)
├── authorized_keys  # 허용된 공개키 목록
├── known_hosts      # 접속했던 서버 목록
└── config           # SSH 설정
```

### SSH config 설정

```bash
# ~/.ssh/config
Host myserver
    HostName 192.168.1.100
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host production
    HostName prod.example.com
    User deploy
    Port 2222
```

사용:
```bash
ssh myserver
ssh production
```

---

## 7. 파일 전송

### scp - 파일 복사

```bash
# 로컬 → 원격
scp file.txt user@host:/path/to/destination/

# 원격 → 로컬
scp user@host:/path/to/file.txt ./

# 디렉토리 복사
scp -r directory/ user@host:/path/to/

# 포트 지정
scp -P 2222 file.txt user@host:/path/
```

### rsync - 동기화

```bash
# 기본 동기화
rsync -av source/ destination/

# 원격 동기화
rsync -av local_dir/ user@host:/remote_dir/
rsync -av user@host:/remote_dir/ local_dir/

# 삭제된 파일도 동기화
rsync -av --delete source/ destination/

# 진행 상황 표시
rsync -av --progress source/ destination/

# 압축 전송
rsync -avz source/ user@host:/destination/

# SSH 포트 지정
rsync -av -e "ssh -p 2222" source/ user@host:/dest/
```

### sftp - 대화형 전송

```bash
sftp user@hostname

# sftp 명령어
sftp> ls              # 원격 목록
sftp> lls             # 로컬 목록
sftp> cd /path        # 원격 이동
sftp> lcd /path       # 로컬 이동
sftp> get file.txt    # 다운로드
sftp> put file.txt    # 업로드
sftp> quit            # 종료
```

---

## 8. 네트워크 설정 (영구)

### Ubuntu/Debian (Netplan)

```bash
# 설정 파일
sudo vi /etc/netplan/01-netcfg.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

```bash
# 적용
sudo netplan apply
```

### CentOS/RHEL (NetworkManager)

```bash
# 설정 파일
sudo vi /etc/sysconfig/network-scripts/ifcfg-eth0
```

```ini
TYPE=Ethernet
BOOTPROTO=static
NAME=eth0
DEVICE=eth0
ONBOOT=yes
IPADDR=192.168.1.100
NETMASK=255.255.255.0
GATEWAY=192.168.1.1
DNS1=8.8.8.8
DNS2=8.8.4.4
```

```bash
# 적용
sudo systemctl restart NetworkManager
# 또는
sudo nmcli connection reload
sudo nmcli connection up eth0
```

### nmcli (NetworkManager CLI)

```bash
# 연결 목록
nmcli connection show

# 장치 상태
nmcli device status

# 고정 IP 설정
sudo nmcli connection modify eth0 ipv4.addresses 192.168.1.100/24
sudo nmcli connection modify eth0 ipv4.gateway 192.168.1.1
sudo nmcli connection modify eth0 ipv4.dns "8.8.8.8 8.8.4.4"
sudo nmcli connection modify eth0 ipv4.method manual
sudo nmcli connection up eth0

# DHCP로 변경
sudo nmcli connection modify eth0 ipv4.method auto
sudo nmcli connection up eth0
```

---

## 9. 실습 예제

### 실습 1: 네트워크 정보 확인

```bash
# IP 주소 확인
ip addr

# 라우팅 테이블
ip route

# DNS 설정
cat /etc/resolv.conf

# 호스트명
hostname -I
```

### 실습 2: 연결 테스트

```bash
# ping 테스트
ping -c 4 google.com

# DNS 조회
dig +short google.com
nslookup google.com

# 경로 추적
traceroute -n google.com
```

### 실습 3: 포트 확인

```bash
# 열린 포트 확인
ss -tuln

# 특정 포트 사용 프로세스
sudo lsof -i :22

# 외부에서 포트 테스트
nc -zv localhost 22
```

### 실습 4: SSH 키 설정

```bash
# 키 생성
ssh-keygen -t ed25519 -C "your_email@example.com"

# 키 확인
ls -la ~/.ssh/

# 공개키 내용 확인
cat ~/.ssh/id_ed25519.pub
```

### 실습 5: 파일 전송

```bash
# 로컬 파일 복사 테스트
mkdir -p ~/test_sync/source ~/test_sync/dest
echo "test content" > ~/test_sync/source/test.txt

# rsync 동기화
rsync -av ~/test_sync/source/ ~/test_sync/dest/

# 결과 확인
ls -la ~/test_sync/dest/
```

---

## 다음 단계

[시스템 모니터링](./11_System_Monitoring.md)에서 시스템 모니터링을 배워봅시다!
