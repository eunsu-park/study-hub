# 13. systemd 심화

**이전**: [보안과 방화벽](./12_Security_and_Firewall.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. systemd 아키텍처, 유닛 타입, 유닛 파일 위치 설명
2. 적절한 Type, Exec, Restart 지시자를 사용하여 커스텀 서비스 유닛 파일 작성
3. 서비스 유닛에 보안 강화 옵션(ProtectSystem, PrivateTmp, NoNewPrivileges) 적용
4. cron 작업의 현대적 대안으로 타이머 유닛 생성
5. 소켓 활성화를 구현하여 필요 시 서비스를 시작하고 부팅 성능 향상
6. Requires, Wants, After, Before를 사용하여 유닛 의존성 및 순서 설정
7. journalctl과 journald 설정을 사용하여 시스템 로그 조회, 필터링, 관리
8. systemd-analyze로 부팅 성능 및 의존성 체인 분석

---

systemd는 사실상 모든 현대 리눅스 배포판에서 PID 1로 실행되는 init 시스템이자 서비스 관리자입니다. 기존의 SysVinit과 Upstart를 대체하면서 병렬 서비스 시작, 온디맨드 활성화, 통합 로깅을 제공합니다. 웹 애플리케이션 배포, 백업 예약, 서비스 시작 실패 디버깅 등 어떤 작업이든 systemd와 함께하게 됩니다. 이 레슨은 기본적인 `systemctl start/stop` 명령을 넘어 직접 유닛 파일을 작성하고 systemd의 모든 기능을 활용하는 방법을 보여줍니다.

## 목차
1. [systemd 아키텍처](#1-systemd-아키텍처)
2. [서비스 유닛 작성](#2-서비스-유닛-작성)
3. [타이머 유닛](#3-타이머-유닛)
4. [소켓 활성화](#4-소켓-활성화)
5. [의존성과 순서](#5-의존성과-순서)
6. [journald 로깅](#6-journald-로깅)
7. [연습 문제](#7-연습-문제)

---

## 1. systemd 아키텍처

### 1.1 systemd 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    systemd 아키텍처                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  systemd (PID 1)                     │   │
│  │  • 시스템 초기화 및 서비스 관리                      │   │
│  │  • 병렬 서비스 시작                                  │   │
│  │  • 소켓/D-Bus 활성화                                 │   │
│  │  • cgroups 기반 리소스 관리                          │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│    ┌──────┼──────────────────────────────┐                 │
│    │      │                              │                  │
│    ▼      ▼                              ▼                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐             │
│  │udevd │ │logind│ │journald│ │networkd│ │resolved│         │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘             │
│                                                             │
│  유닛 타입:                                                 │
│  • .service  - 서비스/데몬                                  │
│  • .socket   - 소켓                                         │
│  • .timer    - 타이머 (cron 대체)                          │
│  • .target   - 그룹 (런레벨 대체)                          │
│  • .mount    - 마운트 포인트                                │
│  • .device   - 장치                                         │
│  • .path     - 경로 모니터링                                │
│  • .slice    - 리소스 그룹                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 유닛 파일 위치

```bash
# 유닛 파일 위치 (우선순위 순)
/etc/systemd/system/        # 시스템 관리자 설정 (최우선)
/run/systemd/system/        # 런타임 생성 유닛
/usr/lib/systemd/system/    # 패키지 설치 유닛

# 사용자 유닛
~/.config/systemd/user/     # 사용자 설정
/etc/systemd/user/          # 전역 사용자 설정
/usr/lib/systemd/user/      # 패키지 설치 사용자 유닛

# 유닛 파일 위치 확인
systemctl show -p FragmentPath nginx.service
systemctl cat nginx.service
```

### 1.3 기본 systemctl 명령

```bash
# 서비스 관리
systemctl start nginx
systemctl stop nginx
systemctl restart nginx
systemctl reload nginx       # 설정만 리로드
systemctl status nginx

# 부팅 시 자동 시작
systemctl enable nginx
systemctl disable nginx
systemctl is-enabled nginx

# 서비스 마스킹 (시작 완전 방지)
systemctl mask nginx
systemctl unmask nginx

# 유닛 목록
systemctl list-units
systemctl list-units --type=service
systemctl list-units --state=failed

# 유닛 파일 목록
systemctl list-unit-files

# 의존성 확인
systemctl list-dependencies nginx.service
systemctl list-dependencies --reverse nginx.service

# 데몬 리로드 (유닛 파일 수정 후)
systemctl daemon-reload
```

---

## 2. 서비스 유닛 작성

### 2.1 기본 서비스 유닛

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application Service
Documentation=https://myapp.example.com/docs
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=myapp
Group=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/myapp --config /etc/myapp/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5

# 환경 변수
Environment=NODE_ENV=production
EnvironmentFile=-/etc/myapp/env  # - 는 파일 없어도 에러 안 남

# 로깅
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myapp

[Install]
WantedBy=multi-user.target
```

### 2.2 서비스 Type

```ini
# Type=simple (기본값)
# 프로세스가 바로 메인 프로세스
[Service]
Type=simple
ExecStart=/usr/bin/myapp

# Type=forking
# 프로세스가 fork하고 부모가 종료
[Service]
Type=forking
PIDFile=/var/run/myapp.pid
ExecStart=/usr/bin/myapp --daemon

# Type=oneshot
# 시작 후 종료되는 일회성 작업
[Service]
Type=oneshot
ExecStart=/usr/local/bin/setup-script.sh
RemainAfterExit=yes  # 종료 후에도 active 상태 유지

# Type=notify
# 프로세스가 sd_notify()로 준비 완료 알림
[Service]
Type=notify
ExecStart=/usr/bin/myapp-with-notify

# Type=dbus
# D-Bus 이름 획득 시 준비 완료
[Service]
Type=dbus
BusName=org.example.MyApp
ExecStart=/usr/bin/myapp

# Type=idle
# 다른 작업 완료 후 실행 (부팅 메시지 출력 등)
[Service]
Type=idle
ExecStart=/usr/bin/welcome-message
```

### 2.3 Exec 옵션

```ini
[Service]
# 시작/종료 명령
ExecStartPre=/usr/bin/myapp-check    # 시작 전 실행
ExecStart=/usr/bin/myapp             # 메인 명령
ExecStartPost=/usr/bin/myapp-notify  # 시작 후 실행
ExecReload=/bin/kill -HUP $MAINPID   # reload 시 실행
ExecStop=/usr/bin/myapp stop         # 종료 명령
ExecStopPost=/usr/bin/cleanup        # 종료 후 실행

# 실패해도 계속 (-접두사)
ExecStartPre=-/usr/bin/optional-check

# 쉘 사용 (;는 여러 명령)
ExecStart=/bin/sh -c 'echo start && /usr/bin/myapp'

# 타임아웃
TimeoutStartSec=30
TimeoutStopSec=30
TimeoutSec=30  # 둘 다 설정

# 재시작 조건
Restart=no              # 재시작 안 함
Restart=on-success      # 정상 종료 시만
Restart=on-failure      # 비정상 종료 시만
Restart=on-abnormal     # 시그널/타임아웃 시
Restart=on-watchdog     # watchdog 타임아웃 시
Restart=on-abort        # 캐치 안 된 시그널 시
Restart=always          # 항상 재시작

RestartSec=5            # 재시작 전 대기 시간
RestartPreventExitStatus=1 23  # 이 종료코드면 재시작 안 함
```

### 2.4 보안 옵션

```ini
[Service]
# 사용자/그룹
User=myapp
Group=myapp
DynamicUser=yes  # 동적 사용자 생성 (임시)

# 파일시스템 접근 제한
ProtectSystem=strict     # /usr, /boot 읽기 전용
ProtectHome=yes          # /home 접근 불가
PrivateTmp=yes           # 격리된 /tmp
ReadWritePaths=/var/lib/myapp
ReadOnlyPaths=/etc/myapp

# 네트워크 제한
PrivateNetwork=yes       # 격리된 네트워크
RestrictAddressFamilies=AF_INET AF_INET6

# 시스템 콜 필터링
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources

# 기타 보안
NoNewPrivileges=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

# 리소스 제한
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=1G
CPUQuota=50%
```

### 2.5 완전한 예제: Node.js 앱

```ini
# /etc/systemd/system/nodeapp.service
[Unit]
Description=Node.js Application
Documentation=https://example.com/docs
After=network.target mongodb.service
Wants=mongodb.service

[Service]
Type=simple
User=nodeapp
Group=nodeapp
WorkingDirectory=/opt/nodeapp

# Node.js 경로
Environment=PATH=/opt/nodeapp/node/bin:/usr/bin
Environment=NODE_ENV=production
EnvironmentFile=/etc/nodeapp/env

# 실행
ExecStart=/opt/nodeapp/node/bin/node /opt/nodeapp/app.js
ExecReload=/bin/kill -HUP $MAINPID

# 재시작 정책
Restart=always
RestartSec=10
WatchdogSec=30

# 로깅
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nodeapp

# 보안
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=/var/lib/nodeapp /var/log/nodeapp

# 리소스 제한
LimitNOFILE=65536
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

---

## 3. 타이머 유닛

### 3.1 타이머 기본

```ini
# /etc/systemd/system/backup.timer
[Unit]
Description=Daily Backup Timer

[Timer]
# 실시간 (wallclock) 타이머
OnCalendar=*-*-* 02:00:00  # 매일 새벽 2시

# 또는 모노토닉 타이머
# OnBootSec=15min           # 부팅 15분 후
# OnUnitActiveSec=1h        # 마지막 활성화 후 1시간

# 정확도 (배터리 절약)
AccuracySec=1min

# 놓친 실행 처리
Persistent=yes  # 시스템 꺼져있던 동안 놓친 실행 보정

[Install]
WantedBy=timers.target

---
# /etc/systemd/system/backup.service
[Unit]
Description=Backup Service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
```

### 3.2 OnCalendar 문법

```bash
# OnCalendar 형식: DayOfWeek Year-Month-Day Hour:Minute:Second

# 매일 자정
OnCalendar=daily
OnCalendar=*-*-* 00:00:00

# 매시간
OnCalendar=hourly
OnCalendar=*-*-* *:00:00

# 매주 월요일 오전 6시
OnCalendar=Mon *-*-* 06:00:00
OnCalendar=weekly

# 매월 1일
OnCalendar=monthly
OnCalendar=*-*-01 00:00:00

# 매년 1월 1일
OnCalendar=yearly
OnCalendar=*-01-01 00:00:00

# 5분마다
OnCalendar=*:0/5
OnCalendar=*-*-* *:00/5:00

# 평일 오전 9시
OnCalendar=Mon..Fri *-*-* 09:00:00

# 특정 날짜
OnCalendar=2024-12-25 00:00:00

# 범위
OnCalendar=*-*-* 08..18:00:00  # 매시간 (8시-18시)

# 타이머 테스트
systemd-analyze calendar "Mon *-*-* 09:00:00"
systemd-analyze calendar --iterations=5 "daily"
```

### 3.3 타이머 관리

```bash
# 타이머 시작/활성화
systemctl start backup.timer
systemctl enable backup.timer

# 타이머 목록
systemctl list-timers
systemctl list-timers --all

# 타이머 상태
systemctl status backup.timer

# 즉시 실행 (타이머 무시)
systemctl start backup.service

# cron에서 마이그레이션
# crontab: 0 2 * * * /usr/local/bin/backup.sh
# → systemd timer: OnCalendar=*-*-* 02:00:00
```

### 3.4 여러 스케줄

```ini
# /etc/systemd/system/multi-schedule.timer
[Unit]
Description=Multiple Schedule Timer

[Timer]
# 여러 시간 지정 가능
OnCalendar=Mon *-*-* 06:00:00
OnCalendar=Wed *-*-* 06:00:00
OnCalendar=Fri *-*-* 06:00:00

# 또는
OnCalendar=Mon,Wed,Fri *-*-* 06:00:00

[Install]
WantedBy=timers.target
```

---

## 4. 소켓 활성화

### 4.1 소켓 활성화 개념

```
┌─────────────────────────────────────────────────────────────┐
│                    소켓 활성화 흐름                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 부팅 시                                                 │
│     ┌─────────┐                                            │
│     │ systemd │──▶ 소켓 열기 (서비스 시작 안 함)           │
│     └─────────┘                                            │
│          │                                                  │
│          ▼                                                  │
│     ┌─────────┐                                            │
│     │ socket  │ (대기 중)                                  │
│     └─────────┘                                            │
│                                                             │
│  2. 연결 요청 시                                            │
│     ┌─────────┐      ┌─────────┐      ┌─────────┐         │
│     │ Client  │─────▶│ socket  │─────▶│ systemd │         │
│     └─────────┘      └─────────┘      └─────────┘         │
│                                              │              │
│                                              ▼              │
│                                        서비스 시작          │
│                                              │              │
│                                              ▼              │
│                                        ┌─────────┐         │
│                                        │ service │         │
│                                        └─────────┘         │
│                                                             │
│  장점:                                                      │
│  • 부팅 시간 단축                                           │
│  • 요청 시에만 서비스 시작                                  │
│  • 서비스 재시작 중에도 연결 유지                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 소켓 유닛

```ini
# /etc/systemd/system/myapp.socket
[Unit]
Description=My App Socket

[Socket]
# TCP 소켓
ListenStream=8080
# 또는 IP 지정
# ListenStream=127.0.0.1:8080
# 또는 IPv6
# ListenStream=[::1]:8080

# Unix 소켓
# ListenStream=/run/myapp/myapp.sock
# SocketUser=myapp
# SocketGroup=myapp
# SocketMode=0660

# UDP 소켓
# ListenDatagram=8081

# 연결 대기열
Backlog=128

# 연결당 하나의 서비스 인스턴스
Accept=no  # 기본값, 하나의 서비스가 모든 연결 처리
# Accept=yes  # 연결마다 새 인스턴스 (inetd 스타일)

# 서비스와 연결 (기본: 같은 이름.service)
# Service=myapp.service

[Install]
WantedBy=sockets.target

---
# /etc/systemd/system/myapp.service
[Unit]
Description=My App Service
Requires=myapp.socket

[Service]
Type=simple
ExecStart=/opt/myapp/bin/myapp
# 소켓은 fd 3으로 전달됨 (또는 $LISTEN_FDS)

[Install]
WantedBy=multi-user.target
```

### 4.3 소켓 활성화 서비스 예제

```python
#!/usr/bin/env python3
# /opt/myapp/bin/myapp.py
# 소켓 활성화를 지원하는 Python 서버

import socket
import os
import sys

def get_systemd_socket():
    """systemd에서 전달된 소켓 가져오기"""
    # LISTEN_FDS: 전달된 fd 개수
    # LISTEN_PID: 대상 프로세스 PID
    listen_fds = int(os.environ.get('LISTEN_FDS', 0))
    listen_pid = int(os.environ.get('LISTEN_PID', 0))

    if listen_pid != os.getpid():
        return None

    if listen_fds >= 1:
        # fd 3부터 시작 (0=stdin, 1=stdout, 2=stderr)
        return socket.fromfd(3, socket.AF_INET, socket.SOCK_STREAM)

    return None

def main():
    # systemd 소켓 또는 새 소켓 생성
    sock = get_systemd_socket()
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', 8080))
        sock.listen(128)
        print("Listening on port 8080")
    else:
        print("Using systemd socket")

    while True:
        conn, addr = sock.accept()
        with conn:
            print(f"Connection from {addr}")
            conn.sendall(b"Hello from socket-activated service!\n")

if __name__ == '__main__':
    main()
```

---

## 5. 의존성과 순서

### 5.1 의존성 지시자

```ini
[Unit]
# Requires: 필수 의존성 (의존 유닛 실패 시 이 유닛도 실패)
Requires=postgresql.service

# Wants: 선택적 의존성 (의존 유닛 실패해도 계속)
Wants=redis.service

# Requisite: 이미 활성화된 유닛에만 의존
Requisite=network.target

# BindsTo: 강한 의존성 (의존 유닛 종료 시 이 유닛도 종료)
BindsTo=libvirtd.service

# PartOf: 의존 유닛 재시작/종료 시 같이 재시작/종료
PartOf=docker.service

# Conflicts: 동시 실행 불가
Conflicts=shutdown.target
```

### 5.2 순서 지시자

```ini
[Unit]
# After: 지정 유닛 시작 후에 시작
After=network.target postgresql.service

# Before: 지정 유닛 시작 전에 시작
Before=httpd.service

# 의존성과 순서는 별개!
# Wants=postgresql.service → postgresql과 함께 시작 시도
# After=postgresql.service → postgresql 시작 완료 후 시작

# 올바른 조합
Wants=postgresql.service
After=postgresql.service
```

### 5.3 Target 유닛

```ini
# /etc/systemd/system/myapp.target
[Unit]
Description=My Application Stack
Requires=myapp-web.service myapp-worker.service
After=myapp-web.service myapp-worker.service

[Install]
WantedBy=multi-user.target

# 사용
systemctl start myapp.target    # 모든 관련 서비스 시작
systemctl stop myapp.target     # 모든 관련 서비스 종료
systemctl restart myapp.target
```

### 5.4 의존성 시각화

```bash
# 의존성 트리
systemctl list-dependencies nginx.service
systemctl list-dependencies --reverse nginx.service

# 부팅 순서 분석
systemd-analyze
systemd-analyze blame
systemd-analyze critical-chain
systemd-analyze critical-chain nginx.service

# 그래프 생성 (SVG)
systemd-analyze dot | dot -Tsvg > systemd.svg
systemd-analyze dot "nginx.service" | dot -Tsvg > nginx-deps.svg
```

---

## 6. journald 로깅

### 6.1 journalctl 기본

```bash
# 전체 로그
journalctl

# 특정 유닛 로그
journalctl -u nginx.service

# 실시간 로그 (tail -f)
journalctl -f
journalctl -fu nginx.service

# 부팅 로그
journalctl -b          # 현재 부팅
journalctl -b -1       # 이전 부팅
journalctl --list-boots

# 시간 범위
journalctl --since "2024-01-01"
journalctl --since "1 hour ago"
journalctl --since "2024-01-01" --until "2024-01-02"
journalctl --since yesterday

# 우선순위 필터
journalctl -p err      # error 이상
journalctl -p warning  # warning 이상
# 0=emerg, 1=alert, 2=crit, 3=err, 4=warning, 5=notice, 6=info, 7=debug

# 커널 메시지
journalctl -k
journalctl --dmesg

# JSON 출력
journalctl -o json
journalctl -o json-pretty

# 디스크 사용량
journalctl --disk-usage

# 로그 정리
journalctl --vacuum-size=500M
journalctl --vacuum-time=7d
```

### 6.2 journald 설정

```ini
# /etc/systemd/journald.conf
[Journal]
# 저장 방식
Storage=persistent     # 영구 저장 (/var/log/journal)
# Storage=volatile     # 메모리만 (/run/log/journal)
# Storage=auto         # /var/log/journal 있으면 persistent

# 크기 제한
SystemMaxUse=500M      # 최대 디스크 사용량
SystemMaxFileSize=50M  # 개별 파일 최대 크기
RuntimeMaxUse=100M     # 런타임(메모리) 최대

# 보존 기간
MaxRetentionSec=1month

# 압축
Compress=yes

# syslog 전달
ForwardToSyslog=no

# 콘솔 출력
ForwardToConsole=no

# 속도 제한
RateLimitIntervalSec=30s
RateLimitBurst=10000
```

```bash
# 설정 적용
systemctl restart systemd-journald

# 영구 저장 디렉토리 생성
mkdir -p /var/log/journal
systemd-tmpfiles --create --prefix /var/log/journal
```

### 6.3 구조화된 로깅

```bash
# systemd-cat으로 로그 전송
echo "Hello" | systemd-cat -t myapp -p info

# 스크립트에서
#!/bin/bash
exec 1> >(systemd-cat -t myscript -p info)
exec 2> >(systemd-cat -t myscript -p err)
echo "This goes to journal"
```

```python
# Python에서 (systemd.journal)
from systemd import journal

journal.send('Hello from Python',
             PRIORITY=journal.LOG_INFO,
             SYSLOG_IDENTIFIER='myapp',
             MYFIELD='custom_value')
```

### 6.4 로그 필터링 고급

```bash
# 필드 기반 필터링
journalctl _SYSTEMD_UNIT=nginx.service
journalctl _UID=1000
journalctl _PID=1234
journalctl _COMM=nginx

# 여러 조건 (AND)
journalctl _SYSTEMD_UNIT=nginx.service _PID=1234

# OR 조건
journalctl _SYSTEMD_UNIT=nginx.service + _SYSTEMD_UNIT=php-fpm.service

# 특정 필드 출력
journalctl -o verbose
journalctl -u nginx --output-fields=MESSAGE,_PID

# grep과 함께
journalctl -u nginx | grep -i error
journalctl -u nginx -g "error|warning"  # -g는 grep 패턴
```

---

## 7. 연습 문제

### 연습 1: 웹 애플리케이션 서비스
```ini
# 요구사항:
# 1. Python/Node.js 웹 앱을 서비스로 등록
# 2. 재시작 정책 설정 (on-failure)
# 3. 환경 변수 파일 사용
# 4. 보안 옵션 적용

# 서비스 유닛 작성:
```

### 연습 2: 백업 타이머
```ini
# 요구사항:
# 1. 매일 새벽 3시 백업 실행
# 2. 매주 일요일 전체 백업
# 3. 놓친 실행 보정
# 4. 로그를 journal에 기록

# 타이머 및 서비스 유닛 작성:
```

### 연습 3: 소켓 활성화 서비스
```ini
# 요구사항:
# 1. 포트 9000에서 대기
# 2. 연결 시에만 서비스 시작
# 3. 유휴 시 자동 종료 (IdleTimeout)

# 소켓 및 서비스 유닛 작성:
```

### 연습 4: 마이크로서비스 스택
```ini
# 요구사항:
# 1. API 서비스 (api.service)
# 2. 워커 서비스 (worker.service)
# 3. 데이터베이스 의존성
# 4. 전체 스택 target

# 모든 유닛 파일 작성:
```

---

## 다음 단계

- [14_성능_튜닝](14_성능_튜닝.md) - 시스템 성능 최적화
- [15_컨테이너_내부_구조](15_컨테이너_내부_구조.md) - cgroups, namespaces
- [systemd 공식 문서](https://systemd.io/)

## 참고 자료

- [systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [Arch Wiki - systemd](https://wiki.archlinux.org/title/systemd)
- [RHEL systemd Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_basic_system_settings/managing-services-with-systemd_configuring-basic-system-settings)

---

[← 이전: 보안과 방화벽](12_보안과_방화벽.md) | [다음: 성능 튜닝 →](14_성능_튜닝.md) | [목차](00_Overview.md)
