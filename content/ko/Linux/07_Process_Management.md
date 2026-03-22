# 프로세스 관리

**이전**: [사용자/그룹 관리](./06_User_Group_Management.md) | **다음**: [패키지 관리](./08_Package_Management.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 프로세스(process)가 무엇인지 설명하고 리눅스 프로세스 생명주기(lifecycle)와 상태(state)를 기술한다
2. ps, pstree, top/htop으로 실행 중인 프로세스를 나열하고 검사한다
3. 적절한 시그널(signal)과 함께 kill, killall, pkill로 프로세스를 종료한다
4. 작업 제어(job control, fg, bg, jobs, &)를 사용해 포그라운드(foreground)와 백그라운드(background) 작업을 관리한다
5. nohup과 disown으로 로그아웃 후에도 프로세스를 유지한다
6. systemctl로 시스템 서비스를 관리한다(start, stop, enable, status)
7. nice와 renice로 프로세스 우선순위(priority)를 조정한다

---

리눅스 시스템에서 일어나는 모든 것, 즉 init 프로세스(PID 1)부터 사용자의 인터랙티브 셸 세션까지 모두 프로세스입니다. 프로세스를 모니터링, 제어, 종료하는 방법을 이해하는 것은 느린 서버 트러블슈팅, 폭주하는 스크립트 정지, 로그아웃 후에도 중요한 서비스가 계속 실행되도록 보장하는 데 핵심적입니다.

> **비유: 레스토랑 주방** -- 프로세스는 주방에서 조리 중인 주문과 같습니다. 커널(kernel)은 다음에 어떤 요리가 불을 쓸지 결정하는(스케줄링) 수석 셰프이며, 요리사들을 병렬로 일하게 하고(스레드), 타버린 요리가 화재 경보를 울리기 전에 멈춥니다(프로세스 종료). `ps`는 주문 현황판이고, `top`은 주방의 실시간 상태 표시판입니다.

## 1. 프로세스 개념

프로세스는 실행 중인 프로그램의 인스턴스입니다.

```
┌─────────────────────────────────────────────────────────┐
│                      프로세스                            │
├─────────────────────────────────────────────────────────┤
│  PID: 1234                    (프로세스 ID)              │
│  PPID: 1                      (부모 프로세스 ID)         │
│  UID: 1000                    (실행 사용자)              │
│  상태: Running                                           │
│  메모리: 50MB                                            │
│  CPU: 2%                                                 │
└─────────────────────────────────────────────────────────┘
```

### 프로세스 상태

| 상태 | 코드 | 설명 |
|------|------|------|
| Running | R | 실행 중 또는 실행 대기 |
| Sleeping | S | 대기 중 (인터럽트 가능) |
| Disk Sleep | D | 대기 중 (인터럽트 불가) |
| Stopped | T | 정지됨 (Ctrl+Z) |
| Zombie | Z | 종료됐지만 부모가 수거 안 함 |

### 프로세스 계층

```
init/systemd (PID 1)
├── sshd
│   └── bash
│       └── vim
├── nginx
│   ├── nginx worker
│   └── nginx worker
└── cron
```

---

## 2. ps - 프로세스 목록

### 기본 사용법

```bash
# 현재 터미널의 프로세스
ps

# 모든 프로세스
ps aux

# 전체 형식
ps -ef
```

### ps aux 출력 해석

```
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.1 168936 11784 ?        Ss   Jan20   0:08 /sbin/init
ubuntu    1234  0.5  1.2 723456 98765 pts/0    Sl   10:00   0:15 /usr/bin/node app.js
```

| 필드 | 설명 |
|------|------|
| USER | 실행 사용자 |
| PID | 프로세스 ID |
| %CPU | CPU 사용률 |
| %MEM | 메모리 사용률 |
| VSZ | 가상 메모리 크기 |
| RSS | 실제 메모리 사용량 |
| TTY | 터미널 (? = 없음) |
| STAT | 상태 |
| START | 시작 시간 |
| TIME | CPU 사용 시간 |
| COMMAND | 명령어 |

### 주요 옵션

```bash
# 모든 프로세스 (BSD 스타일)
ps aux

# 모든 프로세스 (UNIX 스타일)
ps -ef

# 특정 사용자 프로세스
ps -u ubuntu

# 특정 프로세스
ps -p 1234

# 트리 형태
ps auxf
ps -ef --forest

# 특정 명령 검색
ps aux | grep nginx
```

### pstree - 프로세스 트리

```bash
# 전체 트리
pstree

# PID 표시
pstree -p

# 특정 사용자
pstree ubuntu

# 특정 PID 기준
pstree -p 1234
```

---

## 3. top - 실시간 모니터링

### 기본 사용법

```bash
top
```

출력:
```
top - 10:30:00 up 5 days,  3:45,  2 users,  load average: 0.15, 0.10, 0.05
Tasks: 120 total,   1 running, 119 sleeping,   0 stopped,   0 zombie
%Cpu(s):  2.0 us,  1.0 sy,  0.0 ni, 96.5 id,  0.5 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   7976.0 total,   2048.0 free,   3500.0 used,   2428.0 buff/cache
MiB Swap:   2048.0 total,   2048.0 free,      0.0 used.   4000.0 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
 1234 ubuntu    20   0  723456  98765  12345 S   5.0   1.2   0:15.23 node
 5678 mysql     20   0 1234567 234567  23456 S   2.0   2.9   1:23.45 mysqld
```

### top 헤더 설명

| 항목 | 설명 |
|------|------|
| load average | 1분, 5분, 15분 평균 부하 |
| us | 사용자 프로세스 CPU |
| sy | 시스템(커널) CPU |
| ni | nice된 프로세스 CPU |
| id | 유휴 CPU |
| wa | I/O 대기 |

### top 단축키

| 키 | 동작 |
|----|------|
| `q` | 종료 |
| `k` | 프로세스 kill |
| `r` | nice 값 변경 |
| `M` | 메모리 순 정렬 |
| `P` | CPU 순 정렬 |
| `1` | CPU 개별 표시 |
| `c` | 전체 명령어 표시 |
| `f` | 필드 선택 |
| `h` | 도움말 |

### htop - 향상된 top

```bash
# 설치
# Ubuntu/Debian
sudo apt install htop

# CentOS/RHEL
sudo dnf install htop

# 실행
htop
```

htop 특징:
- 컬러 인터페이스
- 마우스 지원
- 스크롤 가능
- 프로세스 트리 보기
- 검색 기능

---

## 4. 프로세스 제어

### kill - 프로세스 종료

```bash
# 기본 종료 (SIGTERM)
kill 1234

# 강제 종료 (SIGKILL)
kill -9 1234
kill -KILL 1234

# 시그널 목록
kill -l
```

### 주요 시그널

| 시그널 | 번호 | 설명 |
|--------|------|------|
| SIGHUP | 1 | 재시작/설정 다시 읽기 |
| SIGINT | 2 | 인터럽트 (Ctrl+C) |
| SIGQUIT | 3 | 종료 + 코어 덤프 |
| SIGKILL | 9 | 강제 종료 (무시 불가) |
| SIGTERM | 15 | 정상 종료 (기본값) |
| SIGSTOP | 19 | 일시 정지 |
| SIGCONT | 18 | 재개 |

```bash
# 종료 요청 (graceful)
kill -TERM 1234

# 강제 종료 (마지막 수단)
kill -9 1234

# 설정 다시 읽기
kill -HUP 1234

# 프로세스 일시 정지
kill -STOP 1234

# 프로세스 재개
kill -CONT 1234
```

### killall - 이름으로 종료

```bash
# 이름으로 종료
killall nginx

# 강제 종료
killall -9 node

# 대화형 확인
killall -i process_name
```

### pkill - 패턴으로 종료

```bash
# 패턴 매칭
pkill -f "python app.py"

# 사용자의 프로세스
pkill -u username

# 시그널 지정
pkill -9 -f "node server.js"
```

### pgrep - 프로세스 ID 찾기

```bash
# PID 찾기
pgrep nginx

# 상세 정보
pgrep -a nginx

# 특정 사용자
pgrep -u root sshd
```

---

## 5. 포그라운드와 백그라운드

### 포그라운드 (Foreground)

터미널을 점유하고 실행됩니다.

```bash
# 일반 실행 (포그라운드)
./long_running_script.sh
```

### 백그라운드 (Background)

터미널을 점유하지 않고 실행됩니다.

```bash
# 백그라운드 실행
./long_running_script.sh &

# 출력 리다이렉트
./script.sh > output.log 2>&1 &
```

### 작업 제어

```bash
# 포그라운드 작업 일시 정지
# Ctrl + Z

# 백그라운드 작업 목록
jobs

# 백그라운드로 보내기
bg %1

# 포그라운드로 가져오기
fg %1

# 작업 종료
kill %1
```

### nohup - 로그아웃 후에도 실행

```bash
# 로그아웃해도 계속 실행
nohup ./script.sh &

# 출력 지정
nohup ./script.sh > output.log 2>&1 &

# PID 확인
echo $!
```

### disown - 터미널과 분리

```bash
# 백그라운드 실행 후 분리
./script.sh &
disown

# 또는 바로 분리
./script.sh &
disown %1
```

---

## 6. systemctl - 서비스 관리

### 서비스 상태 확인

```bash
# 상태 확인
systemctl status nginx

# 실행 중인 서비스 목록
systemctl list-units --type=service

# 활성화된 서비스
systemctl list-units --type=service --state=running

# 실패한 서비스
systemctl --failed
```

### 서비스 제어

```bash
# 시작
sudo systemctl start nginx

# 중지
sudo systemctl stop nginx

# 재시작
sudo systemctl restart nginx

# 설정 다시 읽기 (중단 없이)
sudo systemctl reload nginx

# 재시작 또는 reload
sudo systemctl reload-or-restart nginx
```

### 부팅 시 자동 시작

```bash
# 자동 시작 활성화
sudo systemctl enable nginx

# 자동 시작 비활성화
sudo systemctl disable nginx

# 활성화 상태 확인
systemctl is-enabled nginx

# 활성화하면서 바로 시작
sudo systemctl enable --now nginx
```

### 서비스 로그 확인

```bash
# 서비스 로그
journalctl -u nginx

# 실시간 로그
journalctl -u nginx -f

# 최근 100줄
journalctl -u nginx -n 100

# 오늘 로그
journalctl -u nginx --since today
```

---

## 7. 프로세스 우선순위

### nice - 우선순위 설정

nice 값: -20(높은 우선순위) ~ 19(낮은 우선순위), 기본값 0

```bash
# 낮은 우선순위로 실행
nice -n 10 ./heavy_task.sh

# 높은 우선순위 (root 필요)
sudo nice -n -10 ./important_task.sh
```

### renice - 실행 중인 프로세스 우선순위 변경

```bash
# 우선순위 변경
renice -n 10 -p 1234

# 사용자의 모든 프로세스
sudo renice -n 5 -u username
```

---

## 8. 실습 예제

### 실습 1: 프로세스 모니터링

```bash
# 현재 프로세스 확인
ps aux | head -20

# 특정 프로세스 찾기
ps aux | grep sshd

# 프로세스 트리
pstree -p | head -30

# 실시간 모니터링
top
# (q로 종료)
```

### 실습 2: 백그라운드 작업

```bash
# 테스트 스크립트 생성
cat > test_bg.sh << 'EOF'
#!/bin/bash
for i in {1..10}; do
    echo "Count: $i"
    sleep 2
done
EOF
chmod +x test_bg.sh

# 포그라운드 실행 후 Ctrl+Z로 정지
./test_bg.sh
# Ctrl+Z

# 작업 목록 확인
jobs

# 백그라운드로 보내기
bg %1

# 다시 포그라운드로
fg %1
```

### 실습 3: 프로세스 종료

```bash
# sleep 프로세스 실행
sleep 300 &
echo "PID: $!"

# 프로세스 확인
ps aux | grep sleep

# 종료
kill $!

# 확인
ps aux | grep sleep
```

### 실습 4: 서비스 관리

```bash
# SSH 서비스 상태
systemctl status sshd

# 로그 확인
journalctl -u sshd -n 20

# 실행 중인 서비스 목록
systemctl list-units --type=service --state=running
```

### 실습 5: 리소스 모니터링

```bash
# CPU 많이 사용하는 프로세스 (top 5)
ps aux --sort=-%cpu | head -6

# 메모리 많이 사용하는 프로세스 (top 5)
ps aux --sort=-%mem | head -6

# 프로세스 수
ps aux | wc -l

# 좀비 프로세스 확인
ps aux | grep Z
```

---

## 다음 단계

[패키지 관리](./08_Package_Management.md)에서 패키지 관리를 배워봅시다!
