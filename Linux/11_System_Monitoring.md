# 시스템 모니터링

## 1. 시스템 정보

### uname - 커널 정보

```bash
# 전체 정보
uname -a

# 커널 버전
uname -r

# 운영체제
uname -s

# 하드웨어
uname -m
```

출력:
```
Linux server01 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux
```

### hostnamectl

```bash
hostnamectl
```

출력:
```
 Static hostname: server01
       Icon name: computer-vm
         Chassis: vm
      Machine ID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
         Boot ID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  Virtualization: kvm
Operating System: Ubuntu 22.04.3 LTS
          Kernel: Linux 5.15.0-91-generic
    Architecture: x86-64
```

### lsb_release - 배포판 정보

```bash
# Ubuntu/Debian
lsb_release -a

# 또는
cat /etc/os-release
```

---

## 2. CPU 정보

### /proc/cpuinfo

```bash
# CPU 정보
cat /proc/cpuinfo

# CPU 모델만
grep "model name" /proc/cpuinfo | head -1

# CPU 코어 수
grep -c "processor" /proc/cpuinfo
# 또는
nproc
```

### lscpu

```bash
lscpu
```

출력:
```
Architecture:          x86_64
CPU(s):                4
Thread(s) per core:    2
Core(s) per socket:    2
Socket(s):             1
Model name:            Intel(R) Core(TM) i5-8250U
CPU MHz:               1600.000
```

### CPU 사용률

```bash
# top에서 확인
top -bn1 | head -5

# vmstat
vmstat 1 5

# mpstat (sysstat 패키지)
mpstat 1 5
```

---

## 3. 메모리 정보

### free - 메모리 사용량

```bash
# 기본 출력
free

# 읽기 쉽게
free -h

# 상세 정보
free -h --wide
```

출력:
```
              total        used        free      shared  buff/cache   available
Mem:          7.8Gi       3.2Gi       1.5Gi       256Mi       3.1Gi       4.0Gi
Swap:         2.0Gi          0B       2.0Gi
```

| 필드 | 설명 |
|------|------|
| total | 전체 메모리 |
| used | 사용 중 |
| free | 미사용 |
| shared | 공유 메모리 |
| buff/cache | 버퍼/캐시 |
| available | 사용 가능 (free + 해제 가능한 캐시) |

### /proc/meminfo

```bash
# 상세 메모리 정보
cat /proc/meminfo

# 특정 항목
grep -E "MemTotal|MemFree|MemAvailable" /proc/meminfo
```

---

## 4. 디스크 정보

### df - 파일시스템 사용량

```bash
# 기본 출력
df

# 읽기 쉽게
df -h

# 파일시스템 타입
df -Th

# 특정 경로
df -h /home
```

출력:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        50G   15G   33G  32% /
/dev/sda2       100G   45G   50G  48% /home
tmpfs           3.9G     0  3.9G   0% /dev/shm
```

### du - 디렉토리 사용량

```bash
# 디렉토리 크기
du -sh /var/log

# 하위 폴더별
du -h --max-depth=1 /home

# 가장 큰 디렉토리
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10
```

### lsblk - 블록 장치

```bash
lsblk
```

출력:
```
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sda      8:0    0  100G  0 disk
├─sda1   8:1    0   50G  0 part /
├─sda2   8:2    0   45G  0 part /home
└─sda3   8:3    0    5G  0 part [SWAP]
```

### fdisk - 파티션 정보

```bash
sudo fdisk -l
```

---

## 5. 로그 관리

### 주요 로그 파일

| 로그 파일 | 내용 |
|-----------|------|
| `/var/log/syslog` | 시스템 로그 (Ubuntu) |
| `/var/log/messages` | 시스템 로그 (CentOS) |
| `/var/log/auth.log` | 인증 로그 (Ubuntu) |
| `/var/log/secure` | 인증 로그 (CentOS) |
| `/var/log/kern.log` | 커널 로그 |
| `/var/log/dmesg` | 부팅 메시지 |
| `/var/log/nginx/` | Nginx 로그 |
| `/var/log/apache2/` | Apache 로그 |

### 로그 확인

```bash
# 시스템 로그 (최근)
tail -100 /var/log/syslog

# 실시간 모니터링
tail -f /var/log/syslog

# 에러 검색
grep -i error /var/log/syslog | tail -20

# 여러 로그 동시 모니터링
tail -f /var/log/syslog /var/log/auth.log
```

### journalctl - systemd 로그

```bash
# 전체 로그
journalctl

# 최근 로그
journalctl -n 100

# 실시간
journalctl -f

# 특정 서비스
journalctl -u nginx

# 오늘 로그
journalctl --since today

# 시간 범위
journalctl --since "2024-01-23 00:00" --until "2024-01-23 12:00"

# 부팅 이후
journalctl -b

# 에러만
journalctl -p err

# 커널 로그
journalctl -k
```

### dmesg - 커널 메시지

```bash
# 커널 메시지
dmesg

# 최근 메시지
dmesg | tail -50

# 실시간
dmesg -w

# 읽기 쉽게
dmesg -H
```

---

## 6. 크론잡 (cron)

### crontab 기본

```bash
# 현재 사용자 crontab 보기
crontab -l

# crontab 편집
crontab -e

# 다른 사용자 crontab (root)
sudo crontab -u username -l
```

### cron 형식

```
* * * * * command
│ │ │ │ │
│ │ │ │ └── 요일 (0-7, 0과 7은 일요일)
│ │ │ └──── 월 (1-12)
│ │ └────── 일 (1-31)
│ └──────── 시 (0-23)
└────────── 분 (0-59)
```

### cron 예시

```bash
# 매분 실행
* * * * * /path/to/script.sh

# 매시간 정각
0 * * * * /path/to/script.sh

# 매일 오전 2시
0 2 * * * /path/to/script.sh

# 매주 월요일 오전 3시
0 3 * * 1 /path/to/script.sh

# 매월 1일 자정
0 0 1 * * /path/to/script.sh

# 5분마다
*/5 * * * * /path/to/script.sh

# 평일 오전 9시
0 9 * * 1-5 /path/to/script.sh

# 여러 시간
0 9,12,18 * * * /path/to/script.sh
```

### 실무 cron 예시

```bash
# 백업 (매일 새벽 3시)
0 3 * * * /home/user/scripts/backup.sh >> /var/log/backup.log 2>&1

# 로그 정리 (매주 일요일 새벽 4시)
0 4 * * 0 find /var/log -name "*.log" -mtime +30 -delete

# 시스템 업데이트 (매주 토요일 새벽 2시)
0 2 * * 6 apt update && apt upgrade -y

# 상태 체크 (10분마다)
*/10 * * * * /home/user/scripts/health_check.sh
```

### 시스템 cron 디렉토리

```
/etc/cron.d/        # cron 설정 파일
/etc/cron.daily/    # 매일 실행
/etc/cron.hourly/   # 매시간 실행
/etc/cron.weekly/   # 매주 실행
/etc/cron.monthly/  # 매월 실행
```

---

## 7. 시스템 부하

### uptime - 부하 평균

```bash
uptime
```

출력:
```
 10:30:00 up 15 days,  3:45,  2 users,  load average: 0.15, 0.10, 0.08
                                                         │     │     │
                                                         │     │     └── 15분 평균
                                                         │     └── 5분 평균
                                                         └── 1분 평균
```

부하 평균 해석:
- CPU 코어 수보다 낮으면 여유 있음
- CPU 코어 수와 같으면 완전 사용
- CPU 코어 수보다 높으면 과부하

### vmstat - 가상 메모리 통계

```bash
# 1초 간격 5회
vmstat 1 5
```

출력:
```
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 1  0      0 1500000 200000 3000000   0    0     5    10  100  200  2  1 97  0  0
```

| 필드 | 설명 |
|------|------|
| r | 실행 대기 프로세스 |
| b | 블록된 프로세스 |
| swpd | 사용 중인 스왑 |
| si/so | 스왑 in/out |
| bi/bo | 블록 in/out |
| us | 사용자 CPU |
| sy | 시스템 CPU |
| id | 유휴 CPU |
| wa | I/O 대기 |

### iostat - I/O 통계

```bash
# 설치
sudo apt install sysstat    # Ubuntu
sudo dnf install sysstat    # CentOS

# 사용
iostat -x 1 5
```

---

## 8. 모니터링 스크립트

### 시스템 상태 리포트

```bash
#!/bin/bash
# system_report.sh

echo "=== 시스템 상태 리포트 ==="
echo "날짜: $(date)"
echo

echo "=== 시스템 정보 ==="
uname -a
echo

echo "=== CPU 사용률 ==="
top -bn1 | grep "Cpu(s)" | awk '{print "사용: " 100-$8 "%"}'
echo

echo "=== 메모리 ==="
free -h | grep Mem
echo

echo "=== 디스크 사용량 ==="
df -h | grep -E "^/dev"
echo

echo "=== 부하 평균 ==="
uptime
echo

echo "=== 네트워크 연결 ==="
ss -tuln | grep LISTEN | wc -l
echo "리스닝 포트 수"
```

### 디스크 용량 경고

```bash
#!/bin/bash
# disk_alert.sh

THRESHOLD=80

df -h | grep -E "^/dev" | while read line; do
    usage=$(echo "$line" | awk '{print $5}' | tr -d '%')
    mount=$(echo "$line" | awk '{print $6}')

    if [ "$usage" -gt "$THRESHOLD" ]; then
        echo "경고: $mount 사용량 ${usage}%"
        # 메일 발송 등 알림 추가 가능
    fi
done
```

---

## 9. 실습 예제

### 실습 1: 시스템 정보 확인

```bash
# 시스템 정보
uname -a
hostnamectl

# CPU 정보
lscpu | head -15

# 메모리
free -h

# 디스크
df -h
```

### 실습 2: 로그 분석

```bash
# 시스템 로그 확인
sudo tail -50 /var/log/syslog

# 에러 검색
sudo grep -i "error\|fail" /var/log/syslog | tail -20

# 인증 로그 확인
sudo grep "Failed" /var/log/auth.log | tail -10
```

### 실습 3: journalctl 사용

```bash
# 부팅 이후 로그
journalctl -b --no-pager | tail -50

# 오늘 에러
journalctl --since today -p err

# SSH 서비스 로그
journalctl -u sshd -n 20
```

### 실습 4: cron 설정

```bash
# crontab 편집
crontab -e

# 테스트 작업 추가 (매분 현재 시간 기록)
# * * * * * date >> ~/cron_test.log

# 확인
crontab -l

# 1분 후 결과 확인
cat ~/cron_test.log
```

### 실습 5: 리소스 모니터링

```bash
# CPU 부하
uptime

# vmstat 5초 간격
vmstat 5 3

# top에서 CPU/메모리 높은 프로세스
ps aux --sort=-%cpu | head -6
ps aux --sort=-%mem | head -6
```

---

## 다음 단계

[12_Security_and_Firewall.md](./12_Security_and_Firewall.md)에서 시스템 보안을 배워봅시다!
