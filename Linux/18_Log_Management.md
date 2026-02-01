# 로그 관리

## 학습 목표

이 문서를 통해 다음을 학습합니다:

- systemd-journald의 설정과 활용
- rsyslog 설정 및 필터링
- logrotate를 통한 로그 순환
- 원격 로그 수집 구성

**난이도**: ⭐⭐⭐ (중급-고급)

---

## 목차

1. [Linux 로그 시스템 개요](#1-linux-로그-시스템-개요)
2. [systemd-journald](#2-systemd-journald)
3. [journalctl 고급 사용법](#3-journalctl-고급-사용법)
4. [rsyslog 설정](#4-rsyslog-설정)
5. [logrotate](#5-logrotate)
6. [원격 로그 수집](#6-원격-로그-수집)
7. [로그 분석 도구](#7-로그-분석-도구)

---

## 1. Linux 로그 시스템 개요

### 로그 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    애플리케이션 / 서비스                      │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
                ▼                         ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│    systemd-journald       │   │    rsyslog / syslog-ng      │
│    (바이너리 저널)         │──▶│    (텍스트 로그 파일)        │
└───────────────────────────┘   └─────────────────────────────┘
                │                         │
                ▼                         ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│  /run/log/journal/        │   │  /var/log/*.log             │
│  /var/log/journal/        │   │  원격 서버                   │
└───────────────────────────┘   └─────────────────────────────┘
```

### 주요 로그 파일

| 파일 | 내용 |
|------|------|
| `/var/log/messages` | 일반 시스템 메시지 (RHEL/CentOS) |
| `/var/log/syslog` | 일반 시스템 메시지 (Ubuntu/Debian) |
| `/var/log/auth.log` | 인증 관련 로그 (Ubuntu) |
| `/var/log/secure` | 인증 관련 로그 (RHEL) |
| `/var/log/kern.log` | 커널 메시지 |
| `/var/log/dmesg` | 부팅 시 커널 메시지 |
| `/var/log/cron` | 크론 작업 로그 |
| `/var/log/maillog` | 메일 서버 로그 |

### 로그 우선순위 (Severity)

| 레벨 | 이름 | 설명 |
|------|------|------|
| 0 | emerg | 시스템 사용 불가 |
| 1 | alert | 즉시 조치 필요 |
| 2 | crit | 심각한 오류 |
| 3 | err | 에러 |
| 4 | warning | 경고 |
| 5 | notice | 정상이지만 주목할 만한 상황 |
| 6 | info | 정보성 메시지 |
| 7 | debug | 디버그 메시지 |

---

## 2. systemd-journald

### journald 설정

```bash
# 설정 파일
sudo vi /etc/systemd/journald.conf
```

```ini
# /etc/systemd/journald.conf
[Journal]
# 저장 방식: volatile(메모리), persistent(디스크), auto, none
Storage=persistent

# 최대 크기 (디스크 저장 시)
SystemMaxUse=500M
SystemKeepFree=1G
SystemMaxFileSize=50M
SystemMaxFiles=100

# 런타임 저장소 (메모리)
RuntimeMaxUse=50M

# 로그 압축
Compress=yes

# 봉인 (tamper-evident)
Seal=yes

# rsyslog로 전달
ForwardToSyslog=yes

# 콘솔 출력
ForwardToConsole=no

# 최대 보존 기간
MaxRetentionSec=1month

# 속도 제한
RateLimitIntervalSec=30s
RateLimitBurst=10000
```

```bash
# 설정 적용
sudo systemctl restart systemd-journald
```

### 영구 저장 활성화

```bash
# 저널 디렉토리 생성 (persistent storage)
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal

# 권한 설정
sudo chown root:systemd-journal /var/log/journal
sudo chmod 2755 /var/log/journal

# journald 재시작
sudo systemctl restart systemd-journald
```

### 저널 상태 확인

```bash
# 디스크 사용량
journalctl --disk-usage

# 저널 파일 확인
journalctl --header

# 저널 무결성 검증
journalctl --verify
```

---

## 3. journalctl 고급 사용법

### 기본 조회

```bash
# 모든 로그
journalctl

# 역순 (최신 먼저)
journalctl -r

# 실시간 팔로우
journalctl -f

# 마지막 N줄
journalctl -n 50

# 페이저 없이 출력
journalctl --no-pager
```

### 시간 기반 필터링

```bash
# 오늘 로그
journalctl --since today

# 어제 로그
journalctl --since yesterday --until today

# 특정 시간 범위
journalctl --since "2024-01-15 10:00:00" --until "2024-01-15 12:00:00"

# 상대적 시간
journalctl --since "1 hour ago"
journalctl --since "30 minutes ago"

# 부팅 관련
journalctl -b          # 현재 부팅
journalctl -b -1       # 이전 부팅
journalctl --list-boots # 부팅 목록
```

### 서비스/유닛 필터링

```bash
# 특정 서비스
journalctl -u nginx.service
journalctl -u nginx -u php-fpm

# 커널 메시지
journalctl -k

# 특정 PID
journalctl _PID=1234

# 특정 실행 파일
journalctl /usr/bin/bash

# 특정 사용자
journalctl _UID=1000
```

### 우선순위 필터링

```bash
# 에러 이상
journalctl -p err

# 경고 이상
journalctl -p warning

# 범위 지정
journalctl -p err..crit

# 숫자로 지정
journalctl -p 3
```

### 출력 형식

```bash
# JSON 형식
journalctl -o json
journalctl -o json-pretty

# 상세 출력
journalctl -o verbose

# 간단한 출력
journalctl -o short
journalctl -o short-precise  # 마이크로초 포함

# cat 스타일 (메시지만)
journalctl -o cat

# 내보내기 형식
journalctl -o export
```

### 복합 쿼리

```bash
# 조합 (AND)
journalctl -u nginx -p err --since today

# 커스텀 필드
journalctl _SYSTEMD_UNIT=sshd.service _PID=1234

# 메시지 검색
journalctl -g "error|fail|critical"

# 필드 목록 보기
journalctl -F _SYSTEMD_UNIT
journalctl -F PRIORITY
```

### 저널 유지보수

```bash
# 오래된 로그 삭제 (시간 기준)
sudo journalctl --vacuum-time=30d

# 오래된 로그 삭제 (크기 기준)
sudo journalctl --vacuum-size=500M

# 파일 수 기준 삭제
sudo journalctl --vacuum-files=10

# 모든 저널 삭제
sudo journalctl --rotate
sudo journalctl --vacuum-time=1s
```

---

## 4. rsyslog 설정

### rsyslog 기본 설정

```bash
# 메인 설정 파일
sudo vi /etc/rsyslog.conf
```

```bash
# /etc/rsyslog.conf (주요 섹션)

# 모듈 로드
module(load="imuxsock")    # 로컬 시스템 로그
module(load="imjournal")   # journald 통합
module(load="imklog")      # 커널 로그

# 전역 설정
global(
    workDirectory="/var/lib/rsyslog"
    maxMessageSize="64k"
)

# 기본 규칙
*.info;mail.none;authpriv.none;cron.none    /var/log/messages
authpriv.*                                   /var/log/secure
mail.*                                       -/var/log/maillog
cron.*                                       /var/log/cron
*.emerg                                      :omusrmsg:*
```

### 필터 문법

```bash
# 기본 문법: facility.priority  action

# facility: auth, authpriv, cron, daemon, kern, mail, user, local0-7, *
# priority: emerg, alert, crit, err, warning, notice, info, debug, none, *

# 예시
kern.*                      /var/log/kern.log        # 모든 커널 메시지
*.crit                      /var/log/critical.log    # 모든 심각한 에러
mail.err                    /var/log/mail-err.log    # 메일 에러
*.info;mail.none            /var/log/messages        # info 이상, 메일 제외
```

### 고급 필터링

```bash
# /etc/rsyslog.d/custom.conf

# 속성 기반 필터
:programname, isequal, "nginx" /var/log/nginx/access.log
:programname, startswith, "postfix" /var/log/mail/postfix.log

# 메시지 내용 기반
:msg, contains, "error" /var/log/errors.log
:msg, regex, "failed.*authentication" /var/log/auth-failures.log

# 복합 조건
if $programname == 'sshd' and $msg contains 'Failed' then {
    action(type="omfile" file="/var/log/ssh-failures.log")
    stop
}
```

### 템플릿 사용

```bash
# 커스텀 로그 형식
template(name="CustomFormat" type="string"
    string="%timegenerated% %HOSTNAME% %syslogtag%%msg%\n")

# JSON 형식
template(name="JsonFormat" type="list") {
    constant(value="{")
    constant(value="\"timestamp\":\"")     property(name="timereported" dateFormat="rfc3339")
    constant(value="\",\"host\":\"")       property(name="hostname")
    constant(value="\",\"program\":\"")    property(name="programname")
    constant(value="\",\"severity\":\"")   property(name="syslogseverity-text")
    constant(value="\",\"message\":\"")    property(name="msg" format="json")
    constant(value="\"}\n")
}

# 템플릿 적용
*.* action(type="omfile" file="/var/log/json.log" template="JsonFormat")
```

### 조건부 처리

```bash
# RainerScript 문법
if $programname == 'nginx' then {
    if $syslogseverity <= 3 then {
        # 에러 이상은 별도 파일
        action(type="omfile" file="/var/log/nginx/error.log")
    } else {
        # 나머지는 일반 로그
        action(type="omfile" file="/var/log/nginx/access.log")
    }
    stop
}
```

---

## 5. logrotate

### 기본 설정

```bash
# 전역 설정
sudo vi /etc/logrotate.conf
```

```bash
# /etc/logrotate.conf

# 순환 주기: daily, weekly, monthly
weekly

# 보관할 로그 수
rotate 4

# 새 로그 파일 생성
create

# 날짜 확장자 사용
dateext

# 압축
compress
delaycompress

# 빈 로그 파일 무시
notifempty

# 개별 설정 포함
include /etc/logrotate.d
```

### 애플리케이션별 설정

```bash
# /etc/logrotate.d/nginx
/var/log/nginx/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

```bash
# /etc/logrotate.d/mysql
/var/log/mysql/*.log {
    daily
    rotate 7
    missingok
    create 640 mysql adm
    compress
    sharedscripts
    postrotate
        test -x /usr/bin/mysqladmin || exit 0
        if [ -f /root/.my.cnf ]; then
            /usr/bin/mysqladmin flush-logs
        fi
    endscript
}
```

### 고급 옵션

```bash
# /etc/logrotate.d/custom-app
/var/log/myapp/*.log {
    # 순환 주기
    daily

    # 보관 개수
    rotate 30

    # 크기 기반 순환
    size 100M

    # 최소 크기 (이보다 작으면 순환 안 함)
    minsize 10M

    # 최대 보관 기간
    maxage 365

    # 압축 설정
    compress
    compresscmd /usr/bin/xz
    compressoptions -9
    compressext .xz
    delaycompress

    # 파일 없어도 에러 아님
    missingok

    # 빈 파일 순환 안 함
    notifempty

    # 새 파일 생성
    create 0644 root root

    # 또는 기존 파일 유지
    # copytruncate

    # 스크립트
    prerotate
        echo "About to rotate logs"
    endscript

    postrotate
        systemctl reload myapp
    endscript

    firstaction
        echo "Starting log rotation batch"
    endscript

    lastaction
        echo "Finished log rotation batch"
    endscript
}
```

### logrotate 테스트

```bash
# 드라이런 (실제 실행 안 함)
sudo logrotate -d /etc/logrotate.d/nginx

# 강제 실행
sudo logrotate -f /etc/logrotate.d/nginx

# 상세 출력
sudo logrotate -v /etc/logrotate.conf

# 상태 파일 확인
cat /var/lib/logrotate/status
```

---

## 6. 원격 로그 수집

### rsyslog 서버 설정

```bash
# /etc/rsyslog.conf (서버)

# UDP 수신 활성화
module(load="imudp")
input(type="imudp" port="514")

# TCP 수신 활성화
module(load="imtcp")
input(type="imtcp" port="514")

# 호스트별 로그 분리
$template RemoteLogs,"/var/log/remote/%HOSTNAME%/%PROGRAMNAME%.log"
*.* ?RemoteLogs

# 또는 RainerScript 사용
template(name="RemoteLogsByHost" type="string"
    string="/var/log/remote/%HOSTNAME%/%$YEAR%-%$MONTH%-%$DAY%.log")

if $fromhost-ip != '127.0.0.1' then {
    action(type="omfile" dynaFile="RemoteLogsByHost")
    stop
}
```

### rsyslog 클라이언트 설정

```bash
# /etc/rsyslog.d/remote.conf (클라이언트)

# UDP로 전송 (@)
*.* @logserver.example.com:514

# TCP로 전송 (@@)
*.* @@logserver.example.com:514

# 특정 로그만 전송
auth.* @@logserver.example.com:514
*.err @@logserver.example.com:514

# 큐 설정 (안정적 전송)
action(
    type="omfwd"
    target="logserver.example.com"
    port="514"
    protocol="tcp"
    queue.type="LinkedList"
    queue.filename="remote_queue"
    queue.saveOnShutdown="on"
    queue.maxDiskSpace="1g"
    action.resumeRetryCount="-1"
)
```

### TLS 암호화 설정

```bash
# 서버 설정
module(load="imtcp"
    StreamDriver.Name="gtls"
    StreamDriver.Mode="1"
    StreamDriver.AuthMode="x509/name"
)

global(
    DefaultNetstreamDriver="gtls"
    DefaultNetstreamDriverCAFile="/etc/rsyslog.d/ca.pem"
    DefaultNetstreamDriverCertFile="/etc/rsyslog.d/server-cert.pem"
    DefaultNetstreamDriverKeyFile="/etc/rsyslog.d/server-key.pem"
)

input(type="imtcp" port="6514")
```

```bash
# 클라이언트 설정
global(
    DefaultNetstreamDriver="gtls"
    DefaultNetstreamDriverCAFile="/etc/rsyslog.d/ca.pem"
    DefaultNetstreamDriverCertFile="/etc/rsyslog.d/client-cert.pem"
    DefaultNetstreamDriverKeyFile="/etc/rsyslog.d/client-key.pem"
)

action(
    type="omfwd"
    target="logserver.example.com"
    port="6514"
    protocol="tcp"
    StreamDriver="gtls"
    StreamDriverMode="1"
    StreamDriverAuthMode="x509/name"
)
```

### 방화벽 설정

```bash
# RHEL/CentOS (firewalld)
sudo firewall-cmd --permanent --add-port=514/tcp
sudo firewall-cmd --permanent --add-port=514/udp
sudo firewall-cmd --reload

# Ubuntu (ufw)
sudo ufw allow 514/tcp
sudo ufw allow 514/udp
```

---

## 7. 로그 분석 도구

### lnav (Log Navigator)

```bash
# 설치
# Ubuntu/Debian
sudo apt install lnav

# RHEL/CentOS
sudo yum install epel-release
sudo yum install lnav

# 사용
lnav /var/log/syslog
lnav /var/log/nginx/*.log

# 원격 로그 (SSH)
lnav ssh://user@server/var/log/syslog

# 필터링 (내부 명령)
:filter-in error
:filter-out debug
```

### multitail

```bash
# 설치
sudo apt install multitail  # Ubuntu
sudo yum install multitail  # RHEL

# 여러 파일 동시 모니터링
multitail /var/log/syslog /var/log/auth.log

# 색상 구분
multitail -ci green /var/log/access.log -ci red /var/log/error.log
```

### GoAccess (웹 로그 분석)

```bash
# 설치
sudo apt install goaccess  # Ubuntu
sudo yum install goaccess  # RHEL

# 터미널에서 실시간 분석
goaccess /var/log/nginx/access.log -c

# HTML 보고서 생성
goaccess /var/log/nginx/access.log -o report.html --log-format=COMBINED

# 실시간 HTML 대시보드
goaccess /var/log/nginx/access.log -o /var/www/html/report.html \
    --log-format=COMBINED --real-time-html
```

### 간단한 분석 명령어

```bash
# 가장 많은 요청 IP
awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head

# HTTP 상태 코드 분포
awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c | sort -rn

# 시간대별 요청 수
awk '{print $4}' /var/log/nginx/access.log | cut -d: -f2 | sort | uniq -c

# 에러 메시지 빈도
grep -i error /var/log/syslog | awk '{print $5}' | sort | uniq -c | sort -rn | head

# 실패한 SSH 로그인
grep "Failed password" /var/log/auth.log | awk '{print $11}' | sort | uniq -c | sort -rn
```

---

## 연습 문제

### 문제 1: journalctl 쿼리

다음 조건의 로그를 조회하는 명령을 작성하세요:
1. nginx 서비스의 에러 로그만 (오늘)
2. 특정 PID(1234)의 로그를 JSON으로 출력
3. 지난 1시간 동안의 커널 경고 이상 메시지

### 문제 2: rsyslog 필터

다음 요구사항을 만족하는 rsyslog 규칙을 작성하세요:
- 모든 auth 메시지를 `/var/log/auth-all.log`에 저장
- "Failed" 문자열이 포함된 메시지는 `/var/log/failures.log`에도 저장
- 원격 서버 `192.168.1.100`으로 에러 이상 로그 전송

### 문제 3: logrotate 설정

`/var/log/myapp/` 디렉토리의 로그에 대해:
- 매일 순환
- 30일 보관
- 100MB 초과 시 순환
- xz 압축
- 순환 후 애플리케이션에 SIGHUP 전송

---

## 정답

### 문제 1 정답

```bash
# 1. nginx 에러 로그 (오늘)
journalctl -u nginx -p err --since today

# 2. PID 1234 JSON 출력
journalctl _PID=1234 -o json-pretty

# 3. 커널 경고 이상 (1시간)
journalctl -k -p warning --since "1 hour ago"
```

### 문제 2 정답

```bash
# /etc/rsyslog.d/custom.conf

# auth 로그
auth.*  /var/log/auth-all.log

# Failed 포함 메시지
:msg, contains, "Failed" /var/log/failures.log

# 원격 전송 (에러 이상)
*.err @@192.168.1.100:514
```

### 문제 3 정답

```bash
# /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    daily
    rotate 30
    size 100M
    compress
    compresscmd /usr/bin/xz
    compressext .xz
    delaycompress
    missingok
    notifempty
    create 0644 root root
    postrotate
        [ -f /var/run/myapp.pid ] && kill -HUP $(cat /var/run/myapp.pid)
    endscript
}
```

---

## 다음 단계

- [19_Backup_Recovery.md](./19_Backup_Recovery.md) - rsync, Borg Backup, 재해복구 전략

---

## 참고 자료

- [systemd Journal](https://www.freedesktop.org/software/systemd/man/systemd-journald.service.html)
- [rsyslog Documentation](https://www.rsyslog.com/doc/)
- [logrotate Manual](https://linux.die.net/man/8/logrotate)
- `man journalctl`, `man rsyslog.conf`, `man logrotate`
