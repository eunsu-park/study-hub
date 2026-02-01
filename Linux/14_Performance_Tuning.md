# 14. Linux 성능 튜닝

## 학습 목표
- 시스템 성능 모니터링 및 분석
- sysctl을 통한 커널 파라미터 최적화
- CPU, 메모리, I/O 성능 튜닝
- perf와 flamegraph를 활용한 프로파일링

## 목차
1. [성능 분석 기초](#1-성능-분석-기초)
2. [CPU 튜닝](#2-cpu-튜닝)
3. [메모리 튜닝](#3-메모리-튜닝)
4. [I/O 튜닝](#4-io-튜닝)
5. [네트워크 튜닝](#5-네트워크-튜닝)
6. [프로파일링 도구](#6-프로파일링-도구)
7. [연습 문제](#7-연습-문제)

---

## 1. 성능 분석 기초

### 1.1 USE 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                USE 방법론 (Brendan Gregg)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  각 리소스에 대해 확인:                                     │
│                                                             │
│  U - Utilization (사용률)                                   │
│      리소스가 얼마나 사용되고 있는가?                       │
│      예: CPU 80% 사용 중                                    │
│                                                             │
│  S - Saturation (포화도)                                    │
│      작업이 대기 중인가?                                    │
│      예: 실행 대기열에 10개 프로세스                        │
│                                                             │
│  E - Errors (에러)                                          │
│      에러가 발생하는가?                                     │
│      예: 네트워크 패킷 드롭                                 │
│                                                             │
│  주요 리소스:                                               │
│  • CPU: mpstat, vmstat, top                                │
│  • Memory: free, vmstat, /proc/meminfo                     │
│  • Disk I/O: iostat, iotop                                 │
│  • Network: netstat, ss, sar                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 기본 모니터링 도구

```bash
# top - 실시간 프로세스 모니터링
top
# 단축키: 1=CPU별, M=메모리순, P=CPU순, k=kill

# htop - 향상된 top
htop

# vmstat - 가상 메모리 통계
vmstat 1 5  # 1초 간격, 5회
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
#  2  0      0 1234567 12345 234567    0    0     1     2  100  200  5  2 93  0  0
# r: 실행 대기 프로세스
# b: I/O 대기 프로세스
# si/so: swap in/out
# bi/bo: block in/out
# us/sy/id/wa: user/system/idle/wait

# mpstat - CPU 통계
mpstat -P ALL 1  # 모든 CPU, 1초 간격

# iostat - I/O 통계
iostat -x 1      # 확장 정보, 1초 간격

# sar - 시스템 활동 리포트
sar -u 1 5       # CPU
sar -r 1 5       # 메모리
sar -d 1 5       # 디스크
sar -n DEV 1 5   # 네트워크

# free - 메모리 사용량
free -h

# uptime - 부하 평균
uptime
# load average: 1.50, 1.20, 0.80  (1분, 5분, 15분)
```

### 1.3 sysctl 기본

```bash
# 현재 설정 확인
sysctl -a                    # 모든 설정
sysctl vm.swappiness         # 특정 설정
cat /proc/sys/vm/swappiness  # 직접 읽기

# 임시 변경
sysctl -w vm.swappiness=10
# 또는
echo 10 > /proc/sys/vm/swappiness

# 영구 설정
# /etc/sysctl.conf 또는 /etc/sysctl.d/*.conf
echo "vm.swappiness = 10" >> /etc/sysctl.d/99-custom.conf
sysctl -p /etc/sysctl.d/99-custom.conf  # 적용
sysctl --system  # 모든 설정 파일 로드
```

---

## 2. CPU 튜닝

### 2.1 CPU 정보 확인

```bash
# CPU 정보
lscpu
cat /proc/cpuinfo

# CPU 주파수
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
cpupower frequency-info

# NUMA 정보
numactl --hardware
lscpu | grep NUMA
```

### 2.2 CPU Governor

```bash
# 현재 governor 확인
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# 사용 가능한 governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# performance, powersave, userspace, ondemand, conservative, schedutil

# Governor 변경
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# cpupower 사용
cpupower frequency-set -g performance

# 영구 설정 (Ubuntu)
# /etc/default/cpufrequtils
GOVERNOR="performance"
```

### 2.3 프로세스 우선순위

```bash
# nice 값 (-20 ~ 19, 낮을수록 높은 우선순위)
nice -n -10 ./high-priority-task
renice -n -10 -p <PID>

# 실시간 스케줄링
chrt -f 50 ./realtime-task  # FIFO, 우선순위 50
chrt -r 50 ./realtime-task  # Round Robin

# CPU 친화성 (affinity)
taskset -c 0,1 ./my-program  # CPU 0, 1에서만 실행
taskset -cp 0-3 <PID>        # 실행 중 프로세스 변경

# cgroups로 CPU 제한
# /sys/fs/cgroup/cpu/mygroup/
mkdir /sys/fs/cgroup/cpu/mygroup
echo 50000 > /sys/fs/cgroup/cpu/mygroup/cpu.cfs_quota_us  # 50% 제한
echo <PID> > /sys/fs/cgroup/cpu/mygroup/cgroup.procs
```

### 2.4 CPU 관련 sysctl

```bash
# /etc/sysctl.d/99-cpu.conf

# 스케줄러 튜닝
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
kernel.sched_migration_cost_ns = 5000000

# 워크로드별 최적화
# 서버 워크로드 (처리량 중심)
kernel.sched_autogroup_enabled = 0

# 데스크톱 워크로드 (응답성 중심)
kernel.sched_autogroup_enabled = 1
```

---

## 3. 메모리 튜닝

### 3.1 메모리 정보 확인

```bash
# 메모리 사용량
free -h
cat /proc/meminfo

# 프로세스별 메모리
ps aux --sort=-%mem | head
pmap -x <PID>

# 페이지 캐시 상태
cat /proc/meminfo | grep -E "Cached|Buffers|Dirty"

# NUMA 메모리
numastat
```

### 3.2 Swap 튜닝

```bash
# swappiness (0-100, 낮을수록 swap 덜 사용)
sysctl -w vm.swappiness=10  # 서버: 10, 데스크톱: 60

# Swap 파일 생성
dd if=/dev/zero of=/swapfile bs=1G count=4
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# /etc/fstab에 추가
# /swapfile none swap sw 0 0

# Swap 상태
swapon --show
cat /proc/swaps
```

### 3.3 메모리 관련 sysctl

```bash
# /etc/sysctl.d/99-memory.conf

# Swap 사용 줄이기
vm.swappiness = 10

# 더티 페이지 비율 (쓰기 지연)
vm.dirty_ratio = 20              # 전체 메모리의 20%까지 더티 허용
vm.dirty_background_ratio = 5    # 5%에서 백그라운드 플러시 시작

# 또는 절대값으로
vm.dirty_bytes = 1073741824      # 1GB
vm.dirty_background_bytes = 268435456  # 256MB

# 캐시 압박
vm.vfs_cache_pressure = 50       # 기본 100, 낮으면 캐시 유지

# OOM Killer 튜닝
vm.overcommit_memory = 0         # 0=휴리스틱, 1=항상 허용, 2=제한
vm.overcommit_ratio = 50         # overcommit_memory=2일 때 사용

# 메모리 압축
vm.compaction_proactiveness = 20

# Transparent Huge Pages
# /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never
```

### 3.4 캐시 관리

```bash
# 페이지 캐시 비우기 (프로덕션에서 주의!)
sync
echo 1 > /proc/sys/vm/drop_caches  # 페이지 캐시
echo 2 > /proc/sys/vm/drop_caches  # dentries, inodes
echo 3 > /proc/sys/vm/drop_caches  # 모두

# 특정 파일 캐시 확인
vmtouch -v /path/to/file
fincore /path/to/file

# 프로세스별 캐시 사용
cat /proc/<PID>/smaps | grep -E "^(Rss|Shared|Private)"
```

---

## 4. I/O 튜닝

### 4.1 I/O 스케줄러

```bash
# 현재 스케줄러 확인
cat /sys/block/sda/queue/scheduler
# [mq-deadline] kyber bfq none

# 스케줄러 종류
# - none: NVMe SSD용 (NOOP)
# - mq-deadline: 데드라인 기반, 서버 기본값
# - bfq: Budget Fair Queueing, 데스크톱용
# - kyber: 빠른 장치용

# 스케줄러 변경
echo mq-deadline > /sys/block/sda/queue/scheduler

# 영구 설정 (GRUB)
# /etc/default/grub
# GRUB_CMDLINE_LINUX="elevator=mq-deadline"
# update-grub

# udev 규칙으로 설정
# /etc/udev/rules.d/60-scheduler.rules
# ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="mq-deadline"
# ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"
```

### 4.2 디스크 I/O 튜닝

```bash
# 읽기 미리 가져오기 (readahead)
cat /sys/block/sda/queue/read_ahead_kb  # 기본 128
echo 256 > /sys/block/sda/queue/read_ahead_kb

# 큐 깊이
cat /sys/block/sda/queue/nr_requests
echo 256 > /sys/block/sda/queue/nr_requests

# 최대 섹터
cat /sys/block/sda/queue/max_sectors_kb

# SSD TRIM 활성화
fstrim -v /
# 또는 자동 TRIM (마운트 옵션: discard)
# /dev/sda1 / ext4 defaults,discard 0 1

# 정기 TRIM (권장)
systemctl enable fstrim.timer
```

### 4.3 파일시스템 튜닝

```bash
# ext4 마운트 옵션
# /etc/fstab
# noatime    - 접근 시간 기록 안 함 (성능 향상)
# nodiratime - 디렉토리 접근 시간 기록 안 함
# data=writeback - 저널링 모드 (위험하지만 빠름)
# barrier=0  - 쓰기 장벽 비활성화 (위험)
# commit=60  - 커밋 간격 (초)

# XFS 튜닝
# logbufs=8 - 로그 버퍼 수
# logbsize=256k - 로그 버퍼 크기

# 파일시스템 정보
tune2fs -l /dev/sda1  # ext4
xfs_info /dev/sda1    # XFS
```

### 4.4 I/O 우선순위

```bash
# ionice - I/O 우선순위
ionice -c 3 command        # Idle
ionice -c 2 -n 0 command   # Best-effort, 높은 우선순위
ionice -c 1 command        # Realtime (root만)

# 실행 중 프로세스 변경
ionice -c 2 -n 7 -p <PID>  # 낮은 우선순위로

# 현재 I/O 우선순위 확인
ionice -p <PID>
```

---

## 5. 네트워크 튜닝

### 5.1 네트워크 정보 확인

```bash
# 인터페이스 정보
ip link show
ethtool eth0

# 네트워크 통계
ss -s
netstat -s
cat /proc/net/netstat

# 연결 상태
ss -tuln   # 리스닝 포트
ss -tupn   # 모든 연결
conntrack -L  # 연결 추적 테이블
```

### 5.2 TCP 튜닝

```bash
# /etc/sysctl.d/99-network.conf

# TCP 버퍼 크기
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576

# TCP 소켓 버퍼 (min, default, max)
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216

# TCP 백로그
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# TIME_WAIT 최적화
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1

# TCP Keepalive
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 3

# TCP 혼잡 제어
net.ipv4.tcp_congestion_control = bbr  # 또는 cubic
net.core.default_qdisc = fq

# 포트 범위
net.ipv4.ip_local_port_range = 1024 65535

# SYN 쿠키 (SYN flood 방어)
net.ipv4.tcp_syncookies = 1
```

### 5.3 고성능 웹 서버 설정

```bash
# /etc/sysctl.d/99-webserver.conf

# 파일 핸들 제한
fs.file-max = 2097152
fs.nr_open = 2097152

# 네트워크 스택
net.core.somaxconn = 65535
net.ipv4.tcp_max_tw_buckets = 2000000
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# 버퍼
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 12582912 16777216
net.ipv4.tcp_wmem = 4096 12582912 16777216

# TCP 최적화
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_mtu_probing = 1

# BBR
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
```

### 5.4 연결 제한

```bash
# 시스템 제한
ulimit -n        # 현재 제한
ulimit -n 65535  # 변경

# /etc/security/limits.conf
# * soft nofile 65535
# * hard nofile 65535

# systemd 서비스 제한
# [Service]
# LimitNOFILE=65535
```

---

## 6. 프로파일링 도구

### 6.1 perf 기본

```bash
# perf 설치
apt install linux-tools-common linux-tools-$(uname -r)

# CPU 프로파일링
perf stat ./my-program
perf stat -d ./my-program  # 상세

# 샘플링
perf record -g ./my-program
perf record -g -p <PID> -- sleep 30

# 결과 분석
perf report
perf report --stdio

# 실시간 모니터링
perf top
perf top -p <PID>

# 시스템 전체
perf record -a -g -- sleep 10
```

### 6.2 Flamegraph

```bash
# FlameGraph 도구 설치
git clone https://github.com/brendangregg/FlameGraph

# perf로 데이터 수집
perf record -g -p <PID> -- sleep 60

# Flamegraph 생성
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg

# 또는 한 번에
perf record -F 99 -a -g -- sleep 60
perf script | \
  ./FlameGraph/stackcollapse-perf.pl | \
  ./FlameGraph/flamegraph.pl > flame.svg
```

### 6.3 strace/ltrace

```bash
# 시스템 콜 추적
strace ./my-program
strace -p <PID>

# 특정 시스템 콜만
strace -e open,read,write ./my-program

# 시간 측정
strace -T ./my-program    # 각 syscall 시간
strace -c ./my-program    # 요약 통계

# 라이브러리 콜 추적
ltrace ./my-program
```

### 6.4 기타 도구

```bash
# bpftrace - eBPF 기반 추적
bpftrace -e 'tracepoint:syscalls:sys_enter_open { printf("%s %s\n", comm, str(args->filename)); }'

# 메모리 프로파일링 (Valgrind)
valgrind --tool=massif ./my-program
ms_print massif.out.*

# CPU 프로파일링 (Valgrind)
valgrind --tool=callgrind ./my-program
kcachegrind callgrind.out.*

# 벤치마킹
stress-ng --cpu 4 --timeout 60s
fio --name=random-write --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --direct=1 --size=1G --numjobs=4 --runtime=60
```

### 6.5 성능 체크리스트

```bash
#!/bin/bash
# performance-check.sh

echo "=== 시스템 정보 ==="
uname -a
uptime

echo -e "\n=== CPU ==="
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Model name)"
mpstat 1 1

echo -e "\n=== 메모리 ==="
free -h
cat /proc/meminfo | grep -E "^(MemTotal|MemFree|Buffers|Cached|SwapTotal|SwapFree)"

echo -e "\n=== 디스크 I/O ==="
iostat -x 1 1

echo -e "\n=== 네트워크 ==="
ss -s
cat /proc/net/netstat | grep -E "^(Tcp|Udp)"

echo -e "\n=== 로드 평균 ==="
cat /proc/loadavg

echo -e "\n=== Top 프로세스 (CPU) ==="
ps aux --sort=-%cpu | head -5

echo -e "\n=== Top 프로세스 (Memory) ==="
ps aux --sort=-%mem | head -5

echo -e "\n=== 열린 파일 수 ==="
cat /proc/sys/fs/file-nr

echo -e "\n=== 네트워크 연결 ==="
ss -s
```

---

## 7. 연습 문제

### 연습 1: 웹 서버 튜닝
```bash
# 요구사항:
# 1. 동시 연결 10만 지원
# 2. TCP 최적화 (BBR, keepalive)
# 3. 파일 핸들 제한 증가
# 4. 적절한 I/O 스케줄러 선택

# sysctl 설정 작성:
```

### 연습 2: 데이터베이스 서버 튜닝
```bash
# 요구사항:
# 1. 메모리 최적화 (낮은 swappiness)
# 2. 디스크 I/O 최적화
# 3. 더티 페이지 관리
# 4. CPU 친화성 설정

# 설정 및 명령어 작성:
```

### 연습 3: 성능 문제 진단
```bash
# 시나리오:
# 서버가 느려졌을 때 순차적으로 확인할 항목 작성

# 진단 명령어 목록:
```

### 연습 4: Flamegraph 분석
```bash
# 요구사항:
# 1. CPU 집약적인 프로그램 작성 또는 선택
# 2. perf로 프로파일링
# 3. Flamegraph 생성
# 4. 병목 지점 분석

# 명령어 및 분석 방법:
```

---

## 다음 단계

- [15_컨테이너_내부_구조](15_컨테이너_내부_구조.md) - cgroups, namespaces
- [16_저장소_관리](16_저장소_관리.md) - LVM, RAID
- [Brendan Gregg's Blog](https://www.brendangregg.com/)

## 참고 자료

- [Linux Performance](https://www.brendangregg.com/linuxperf.html)
- [Red Hat Performance Tuning Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/index)
- [kernel.org sysctl Documentation](https://www.kernel.org/doc/Documentation/sysctl/)
- [perf Examples](https://www.brendangregg.com/perf.html)

---

[← 이전: systemd 심화](13_systemd_심화.md) | [다음: 컨테이너 내부 구조 →](15_컨테이너_내부_구조.md) | [목차](00_Overview.md)
