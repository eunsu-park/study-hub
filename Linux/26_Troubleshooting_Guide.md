# 트러블슈팅 가이드

## 학습 목표

이 문서를 통해 다음을 학습합니다:

- 체계적인 문제 진단 방법론
- 부팅 문제 해결
- 네트워크, 디스크, 메모리 문제 진단
- 성능 병목 분석

**난이도**: ⭐⭐⭐ (중급-고급)

---

## 목차

1. [문제 해결 방법론](#1-문제-해결-방법론)
2. [부팅 문제](#2-부팅-문제)
3. [네트워크 문제](#3-네트워크-문제)
4. [디스크 문제](#4-디스크-문제)
5. [메모리 문제](#5-메모리-문제)
6. [프로세스 문제](#6-프로세스-문제)
7. [성능 분석](#7-성능-분석)

---

## 1. 문제 해결 방법론

### 체계적 접근법

```
┌─────────────────────────────────────────────────────────────┐
│                    문제 해결 프로세스                        │
│                                                             │
│  1. 문제 정의                                               │
│     └── 증상이 무엇인가?                                    │
│     └── 언제부터 발생했는가?                                │
│     └── 어떤 변경이 있었는가?                               │
│                                                             │
│  2. 정보 수집                                               │
│     └── 로그 확인                                           │
│     └── 시스템 상태 확인                                    │
│     └── 설정 검토                                           │
│                                                             │
│  3. 가설 수립                                               │
│     └── 가능한 원인 나열                                    │
│     └── 우선순위 결정                                       │
│                                                             │
│  4. 테스트 및 검증                                          │
│     └── 가설 테스트                                         │
│     └── 결과 확인                                           │
│                                                             │
│  5. 해결 및 문서화                                          │
│     └── 수정 적용                                           │
│     └── 재발 방지책 수립                                    │
│     └── 문서화                                              │
└─────────────────────────────────────────────────────────────┘
```

### 기본 진단 명령어

```bash
# 시스템 개요
uptime                    # 가동 시간, 로드 평균
uname -a                  # 커널 버전
hostnamectl               # 호스트 정보
dmidecode -t system       # 하드웨어 정보

# 리소스 개요
free -h                   # 메모리
df -h                     # 디스크
top -bn1 | head -20       # CPU/메모리 상위 프로세스

# 최근 로그
journalctl -p err -since "1 hour ago"
dmesg | tail -50

# 서비스 상태
systemctl --failed
systemctl status <service>
```

### 로그 확인 우선순위

```bash
# 1. 시스템 로그
journalctl -xe                    # 최근 에러
journalctl -b                     # 현재 부팅
journalctl -p err --since today   # 오늘 에러

# 2. 커널 메시지
dmesg --level=err,warn
dmesg -T | tail -100

# 3. 서비스별 로그
journalctl -u nginx -f
journalctl -u postgresql --since "1 hour ago"

# 4. 애플리케이션 로그
tail -f /var/log/nginx/error.log
tail -f /var/log/syslog
```

---

## 2. 부팅 문제

### 부팅 프로세스

```
BIOS/UEFI → GRUB → Kernel → systemd → Services → Login
    │          │       │        │          │
    └──────────┴───────┴────────┴──────────┴── 각 단계에서 실패 가능
```

### GRUB 복구

```bash
# GRUB 메뉴에서 'e'를 눌러 편집
# 커널 라인에 추가:
linux /vmlinuz... root=... single    # 싱글 유저 모드
linux /vmlinuz... root=... init=/bin/bash  # 직접 쉘

# GRUB 재설치 (복구 모드에서)
mount /dev/sda2 /mnt
mount /dev/sda1 /mnt/boot
mount --bind /dev /mnt/dev
mount --bind /proc /mnt/proc
mount --bind /sys /mnt/sys
chroot /mnt
grub-install /dev/sda
update-grub
exit
reboot
```

### 파일시스템 복구

```bash
# fsck 실행 (언마운트 상태에서)
fsck -y /dev/sda1

# 라이브 환경에서 루트 파일시스템 체크
# 1. 복구 모드로 부팅 또는 Live USB
# 2. 파일시스템 언마운트 후 체크
umount /dev/sda2
fsck -y /dev/sda2

# XFS 복구
xfs_repair /dev/sda2

# ext4 슈퍼블록 복구
mke2fs -n /dev/sda2  # 백업 슈퍼블록 위치 확인
e2fsck -b 32768 /dev/sda2  # 백업 슈퍼블록으로 복구
```

### systemd 부팅 문제

```bash
# 부팅 분석
systemd-analyze
systemd-analyze blame
systemd-analyze critical-chain

# 실패한 유닛 확인
systemctl --failed
systemctl reset-failed

# 특정 서비스 문제 디버깅
systemctl status nginx.service
journalctl -u nginx.service -b

# 응급 모드로 부팅
# GRUB에서 커널 라인에 추가:
systemd.unit=emergency.target
# 또는
systemd.unit=rescue.target
```

### 비밀번호 초기화

```bash
# 1. GRUB에서 'e' 눌러 편집
# 2. linux 라인 끝에 추가: init=/bin/bash
# 3. Ctrl+X로 부팅

# 루트 파일시스템 읽기-쓰기로 재마운트
mount -o remount,rw /

# 비밀번호 변경
passwd root
passwd username

# SELinux 재레이블링 (RHEL/CentOS)
touch /.autorelabel

# 재부팅
exec /sbin/init
# 또는
reboot -f
```

---

## 3. 네트워크 문제

### 단계별 진단

```bash
# 1. 인터페이스 상태
ip link show
ip addr show
ethtool eth0

# 2. 라우팅 테이블
ip route show
ip route get 8.8.8.8

# 3. DNS 확인
cat /etc/resolv.conf
nslookup google.com
dig google.com

# 4. 연결성 테스트
ping -c 3 8.8.8.8           # IP 연결
ping -c 3 google.com        # DNS 해석 + 연결
traceroute google.com       # 경로 추적

# 5. 포트 확인
ss -tlnp                    # 리스닝 포트
ss -tnp                     # 연결된 소켓
netstat -anp                # 전체 상태
```

### 연결 문제 진단

```bash
# 특정 포트 연결 테스트
nc -zv 192.168.1.100 80
telnet 192.168.1.100 80

# TCP 연결 상태
ss -tn state established
ss -tn state time-wait | wc -l

# 패킷 캡처
tcpdump -i eth0 port 80
tcpdump -i eth0 host 192.168.1.100
tcpdump -i eth0 -w capture.pcap

# MTU 문제 확인
ping -M do -s 1472 192.168.1.1  # 1500 - 28 = 1472
```

### 방화벽 문제

```bash
# iptables 확인
iptables -L -n -v
iptables -t nat -L -n -v

# nftables 확인
nft list ruleset

# firewalld 확인 (RHEL/CentOS)
firewall-cmd --list-all
firewall-cmd --get-active-zones

# UFW 확인 (Ubuntu)
ufw status verbose

# 임시로 방화벽 비활성화 (테스트용)
systemctl stop firewalld
iptables -F
```

### DNS 문제

```bash
# DNS 해석 테스트
nslookup example.com
dig example.com
host example.com

# 특정 DNS 서버 사용
nslookup example.com 8.8.8.8
dig @8.8.8.8 example.com

# DNS 캐시 확인/플러시
systemd-resolve --statistics
systemd-resolve --flush-caches
# 또는
resolvectl flush-caches

# /etc/hosts 확인
cat /etc/hosts

# nsswitch 설정 확인
cat /etc/nsswitch.conf | grep hosts
```

---

## 4. 디스크 문제

### 디스크 상태 확인

```bash
# 디스크 사용량
df -h
df -i                       # inode 사용량

# 파티션 확인
lsblk
fdisk -l
parted -l

# 디스크 건강 상태 (SMART)
smartctl -H /dev/sda
smartctl -a /dev/sda

# 디스크 I/O 통계
iostat -xz 1
iotop -o
```

### 디스크 공간 문제

```bash
# 대용량 파일 찾기
find / -xdev -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# 디렉토리별 사용량
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
du -sh /var/log/*

# 삭제된 파일이 여전히 공간 차지 (열린 파일 핸들)
lsof | grep deleted
lsof +L1

# 로그 파일 정리
journalctl --vacuum-size=100M
find /var/log -name "*.gz" -mtime +30 -delete

# inode 문제 (파일 수 초과)
find / -xdev -type d -exec sh -c 'echo "$(find "{}" -maxdepth 1 | wc -l) {}"' \; | sort -rn | head
```

### 파일시스템 복구

```bash
# 읽기 전용 모드 확인
mount | grep ' / '

# 읽기-쓰기로 재마운트
mount -o remount,rw /

# 파일시스템 오류
dmesg | grep -i "error\|fail\|corrupt"

# 강제 fsck
touch /forcefsck
reboot

# 또는 복구 모드에서
fsck -y /dev/sda1
```

### LVM 문제

```bash
# LVM 상태 확인
pvs; vgs; lvs
pvdisplay; vgdisplay; lvdisplay

# LVM 메타데이터 복구
vgcfgrestore -l vg_name              # 백업 목록
vgcfgrestore -f /etc/lvm/archive/... vg_name

# VG 활성화
vgchange -ay vg_name

# LV 활성화
lvchange -ay /dev/vg_name/lv_name
```

---

## 5. 메모리 문제

### 메모리 상태 확인

```bash
# 메모리 개요
free -h
cat /proc/meminfo

# 프로세스별 메모리
ps aux --sort=-%mem | head -20
top -o %MEM

# 스왑 사용
swapon -s
cat /proc/swaps

# 메모리 사용 상세
smem -t -k
pmap -x <PID>
```

### OOM Killer 진단

```bash
# OOM 발생 확인
dmesg | grep -i "out of memory"
journalctl -k | grep -i "oom"

# OOM 점수 확인
cat /proc/<PID>/oom_score
cat /proc/<PID>/oom_score_adj

# OOM 점수 조정 (보호)
echo -1000 > /proc/<PID>/oom_score_adj

# 또는 systemd 서비스에서
# [Service]
# OOMScoreAdjust=-500
```

### 메모리 누수 진단

```bash
# 프로세스 메모리 추적
while true; do
    ps -o pid,vsz,rss,comm -p <PID>
    sleep 60
done

# Valgrind (개발 환경)
valgrind --leak-check=full ./myapp

# smem으로 USS/PSS 확인
smem -P nginx
```

### 캐시/버퍼 정리

```bash
# 캐시 상태
cat /proc/meminfo | grep -E "Cached|Buffers|SReclaimable"

# 캐시 클리어 (운영 중 주의)
sync
echo 1 > /proc/sys/vm/drop_caches  # 페이지 캐시
echo 2 > /proc/sys/vm/drop_caches  # dentries, inodes
echo 3 > /proc/sys/vm/drop_caches  # 전부

# 스왑 정리 (메모리 여유 있을 때)
swapoff -a && swapon -a
```

---

## 6. 프로세스 문제

### 프로세스 상태 확인

```bash
# 프로세스 목록
ps aux
ps -ef
ps auxf  # 트리 형태

# 특정 프로세스 찾기
pgrep -a nginx
pidof nginx

# 프로세스 상태
cat /proc/<PID>/status
cat /proc/<PID>/limits

# 프로세스 환경 변수
cat /proc/<PID>/environ | tr '\0' '\n'
```

### 좀비/고아 프로세스

```bash
# 좀비 프로세스 찾기
ps aux | awk '$8=="Z"'

# 좀비의 부모 프로세스 찾기
ps -ef | grep <ZOMBIE_PID>

# 부모 프로세스 확인
cat /proc/<ZOMBIE_PID>/status | grep PPid

# 좀비 제거 (부모 종료)
kill -SIGCHLD <PARENT_PID>
# 또는 부모 프로세스 재시작
```

### strace/lsof 디버깅

```bash
# 시스템 호출 추적
strace -p <PID>
strace -p <PID> -e open,read,write
strace -f -p <PID>  # 자식 프로세스 포함

# 열린 파일 확인
lsof -p <PID>
lsof -c nginx
lsof -i :80
lsof +D /var/log

# 파일 디스크립터 확인
ls -la /proc/<PID>/fd
cat /proc/<PID>/limits | grep "open files"
```

### 서비스 문제

```bash
# 서비스 상태
systemctl status nginx

# 서비스 로그
journalctl -u nginx -f
journalctl -u nginx --since "10 minutes ago"

# 서비스 재시작 (상세)
systemctl restart nginx
systemctl daemon-reload && systemctl restart nginx

# 서비스 설정 확인
systemctl cat nginx
systemctl show nginx
```

---

## 7. 성능 분석

### 시스템 개요

```bash
# 종합 상태
vmstat 1 10
mpstat -P ALL 1 5
iostat -xz 1 5

# 로드 평균 해석
# load average: 1.00, 0.75, 0.50
# 1분, 5분, 15분 평균
# CPU 코어 수와 비교 (1.0 = 100% 활용)
nproc  # CPU 코어 수
```

### CPU 분석

```bash
# CPU 사용률
top -bn1 | head -20
htop

# 프로세스별 CPU
pidstat 1 5
ps aux --sort=-%cpu | head -10

# CPU 상세 정보
mpstat -P ALL 1

# 핫스팟 확인 (perf)
perf top
perf record -g -p <PID> -- sleep 30
perf report
```

### I/O 분석

```bash
# 디스크 I/O
iostat -xz 1
iotop -o

# 프로세스별 I/O
pidstat -d 1

# 대기 시간 확인
await, svctm in iostat output
# await > 10ms: 느린 디스크
# %util > 80%: 병목 가능성

# I/O 프로파일링
blktrace -d /dev/sda -o - | blkparse -i -
```

### 네트워크 분석

```bash
# 네트워크 통계
netstat -s
ss -s

# 대역폭 모니터링
iftop -i eth0
nethogs eth0

# 연결 상태
ss -tn state established | wc -l
ss -tn state time-wait | wc -l

# 패킷 손실 확인
netstat -s | grep -i "packet loss\|retrans"
```

### 병목 분석 체크리스트

```bash
#!/bin/bash
# bottleneck-check.sh

echo "=== System Overview ==="
uptime
echo

echo "=== CPU ==="
mpstat 1 3 | tail -4
echo

echo "=== Memory ==="
free -h
echo

echo "=== Disk I/O ==="
iostat -xz 1 3 | tail -10
echo

echo "=== Network ==="
ss -s
echo

echo "=== Top Processes (CPU) ==="
ps aux --sort=-%cpu | head -6
echo

echo "=== Top Processes (Memory) ==="
ps aux --sort=-%mem | head -6
echo

echo "=== Failed Services ==="
systemctl --failed
echo

echo "=== Recent Errors ==="
journalctl -p err --since "1 hour ago" | tail -20
```

### 성능 기준선 (Baseline)

```bash
# 정상 상태 기록 (정기적으로)
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M)
OUTPUT_DIR=/var/log/baseline

mkdir -p $OUTPUT_DIR

# 시스템 정보
vmstat 1 60 > $OUTPUT_DIR/vmstat-$DATE.log &
iostat -xz 1 60 > $OUTPUT_DIR/iostat-$DATE.log &
mpstat -P ALL 1 60 > $OUTPUT_DIR/mpstat-$DATE.log &
sar -n DEV 1 60 > $OUTPUT_DIR/sar-net-$DATE.log &

wait

# 스냅샷
ps aux > $OUTPUT_DIR/ps-$DATE.log
free -h > $OUTPUT_DIR/memory-$DATE.log
df -h > $OUTPUT_DIR/disk-$DATE.log
ss -s > $OUTPUT_DIR/network-$DATE.log
```

---

## 연습 문제

### 문제 1: 부팅 문제

시스템이 emergency mode로 부팅됩니다. 원인을 찾고 해결하는 단계를 설명하세요.

### 문제 2: 디스크 공간

/var 파티션이 100% 찼습니다. 원인을 찾고 해결하는 명령을 작성하세요.

### 문제 3: 네트워크 연결

외부 웹 사이트에 접속이 안 됩니다. 단계별 진단 절차를 작성하세요.

---

## 정답

### 문제 1 정답

```bash
# 1. 에러 메시지 확인
journalctl -xb
dmesg | grep -i error

# 2. 일반적인 원인
# - /etc/fstab 오류
# - 파일시스템 손상
# - SELinux 문제

# 3. /etc/fstab 확인
cat /etc/fstab
# UUID나 장치가 맞는지 확인

# 4. 파일시스템 체크
fsck -y /dev/sda1

# 5. fstab 수정 (문제가 있다면)
# nofail 옵션 추가 또는 문제 항목 주석 처리

# 6. 재부팅
reboot
```

### 문제 2 정답

```bash
# 1. 전체 사용량 확인
df -h /var

# 2. 대용량 디렉토리 찾기
du -h --max-depth=1 /var | sort -hr | head -10

# 3. 대용량 파일 찾기
find /var -type f -size +100M -exec ls -lh {} \;

# 4. 일반적인 원인 확인
du -sh /var/log
du -sh /var/cache
du -sh /var/lib/docker  # Docker 사용 시

# 5. 로그 정리
journalctl --vacuum-size=100M
find /var/log -name "*.gz" -mtime +7 -delete
truncate -s 0 /var/log/large-file.log

# 6. 삭제된 파일 핸들 확인
lsof +L1 | grep /var
# 서비스 재시작으로 핸들 해제
```

### 문제 3 정답

```bash
# 1. 인터페이스 상태
ip addr show
ip link show

# 2. 기본 게이트웨이 확인
ip route show
ping -c 3 <gateway-ip>

# 3. 외부 IP 연결
ping -c 3 8.8.8.8

# 4. DNS 확인 (IP는 되는데 도메인이 안 되면)
nslookup google.com
cat /etc/resolv.conf

# 5. 방화벽 확인
iptables -L -n
firewall-cmd --list-all

# 6. 특정 포트 테스트
nc -zv google.com 443

# 7. 라우팅 경로 확인
traceroute google.com

# 진단 결과에 따른 조치:
# - IP 없음: DHCP 또는 수동 IP 설정
# - 게이트웨이 안 됨: 네트워크 케이블/스위치 확인
# - 외부 IP 안 됨: 라우터/방화벽 확인
# - DNS만 안 됨: resolv.conf 수정
```

---

## 참고 자료

- [Linux Performance](http://www.brendangregg.com/linuxperf.html)
- [Red Hat System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/system_administrators_guide/index)
- [Ubuntu Server Guide](https://ubuntu.com/server/docs)
- `man strace`, `man lsof`, `man perf`

---

## 마무리

이 문서로 Linux 학습 시리즈가 완료됩니다.

전체 학습 내용:
- 01-03: Linux 기초
- 04-08: 중급 관리
- 09-12: 고급 서버 관리
- 13-16: 심화 (systemd, 성능, 컨테이너, 스토리지)
- 17-26: 전문가 수준 (보안, 가상화, 자동화, HA, 트러블슈팅)

[00_Overview.md](./00_Overview.md)로 돌아가서 전체 학습 로드맵을 확인하세요.
