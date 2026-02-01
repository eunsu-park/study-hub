# SELinux와 AppArmor

## 학습 목표

이 문서를 통해 다음을 학습합니다:

- 필수 접근 제어(MAC)의 개념과 필요성
- SELinux 모드와 정책 관리
- AppArmor 프로파일 작성 및 관리
- 보안 모듈 트러블슈팅

**난이도**: ⭐⭐⭐⭐ (고급)

---

## 목차

1. [필수 접근 제어 개요](#1-필수-접근-제어-개요)
2. [SELinux 기초](#2-selinux-기초)
3. [SELinux 정책 관리](#3-selinux-정책-관리)
4. [SELinux 트러블슈팅](#4-selinux-트러블슈팅)
5. [AppArmor 기초](#5-apparmor-기초)
6. [AppArmor 프로파일](#6-apparmor-프로파일)
7. [실무 시나리오](#7-실무-시나리오)

---

## 1. 필수 접근 제어 개요

### DAC vs MAC

```
┌─────────────────────────────────────────────────────────────┐
│                    접근 제어 비교                            │
├─────────────────────────────────────────────────────────────┤
│  DAC (Discretionary Access Control)                         │
│  - 전통적인 Unix 권한 모델                                  │
│  - 파일 소유자가 권한 결정                                  │
│  - chmod, chown으로 관리                                    │
│  - root는 모든 제한 우회 가능                               │
├─────────────────────────────────────────────────────────────┤
│  MAC (Mandatory Access Control)                             │
│  - 시스템 정책이 접근 결정                                  │
│  - 사용자가 정책 변경 불가                                  │
│  - SELinux, AppArmor로 구현                                 │
│  - root도 정책에 의해 제한됨                                │
└─────────────────────────────────────────────────────────────┘
```

### 보안 모듈 비교

| 특성 | SELinux | AppArmor |
|------|---------|----------|
| 기반 배포판 | RHEL/CentOS/Fedora | Ubuntu/Debian/SUSE |
| 접근 방식 | 레이블 기반 | 경로 기반 |
| 복잡도 | 높음 | 낮음 |
| 세밀함 | 매우 세밀 | 중간 |
| 학습 곡선 | 가파름 | 완만 |
| 기본 정책 | 포괄적 | 제한적 |

---

## 2. SELinux 기초

### SELinux 모드

```bash
# 현재 모드 확인
getenforce
# Enforcing, Permissive, 또는 Disabled

# 상세 상태 확인
sestatus

# 임시 모드 변경 (재부팅 시 복원)
sudo setenforce 0  # Permissive
sudo setenforce 1  # Enforcing
```

### 영구적 모드 변경

```bash
# /etc/selinux/config 편집
# RHEL/CentOS
sudo vi /etc/selinux/config
```

```ini
# /etc/selinux/config
SELINUX=enforcing     # enforcing, permissive, disabled
SELINUXTYPE=targeted  # targeted, minimum, mls
```

### SELinux 컨텍스트

모든 파일, 프로세스, 포트에 보안 컨텍스트가 할당됩니다:

```
사용자:역할:타입:레벨
user_u:role_r:type_t:s0
```

```bash
# 파일 컨텍스트 확인
ls -Z /var/www/html/
# -rw-r--r--. root root unconfined_u:object_r:httpd_sys_content_t:s0 index.html

# 프로세스 컨텍스트 확인
ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0    12345 ?  00:00:01 httpd

# 사용자 컨텍스트 확인
id -Z
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023
```

### 주요 타입

| 타입 | 설명 |
|------|------|
| `httpd_t` | Apache 웹 서버 프로세스 |
| `httpd_sys_content_t` | 웹 콘텐츠 파일 |
| `mysqld_t` | MySQL 프로세스 |
| `sshd_t` | SSH 데몬 |
| `user_home_t` | 사용자 홈 디렉토리 |
| `tmp_t` | 임시 파일 |

---

## 3. SELinux 정책 관리

### 파일 컨텍스트 변경

```bash
# 임시 변경 (재레이블링 시 복원)
chcon -t httpd_sys_content_t /var/www/custom/index.html

# 디렉토리 재귀적 변경
chcon -R -t httpd_sys_content_t /var/www/custom/

# 다른 파일의 컨텍스트 복사
chcon --reference=/var/www/html/index.html /var/www/custom/index.html
```

### 영구적 컨텍스트 설정

```bash
# 정책에 컨텍스트 규칙 추가
sudo semanage fcontext -a -t httpd_sys_content_t "/srv/www(/.*)?"

# 정책 적용
sudo restorecon -Rv /srv/www

# 컨텍스트 규칙 목록 확인
sudo semanage fcontext -l | grep httpd

# 규칙 삭제
sudo semanage fcontext -d "/srv/www(/.*)?"
```

### SELinux 불리언

불리언은 SELinux 정책의 특정 기능을 켜고 끄는 스위치입니다:

```bash
# 모든 불리언 목록
getsebool -a

# 특정 불리언 확인
getsebool httpd_can_network_connect

# 임시 변경
sudo setsebool httpd_can_network_connect on

# 영구 변경 (-P 옵션)
sudo setsebool -P httpd_can_network_connect on

# 불리언 검색
getsebool -a | grep httpd
```

### 주요 불리언 예시

```bash
# 웹 서버 관련
httpd_can_network_connect      # 외부 네트워크 연결 허용
httpd_can_network_connect_db   # DB 연결 허용
httpd_can_sendmail            # 메일 전송 허용
httpd_enable_homedirs         # 사용자 홈 디렉토리 접근

# FTP 관련
ftpd_anon_write              # 익명 쓰기 허용
ftpd_full_access             # 전체 파일시스템 접근

# 기타
samba_enable_home_dirs       # Samba 홈 디렉토리 공유
```

### 포트 컨텍스트

```bash
# 포트 레이블 확인
sudo semanage port -l | grep http
# http_port_t                    tcp      80, 81, 443, 488, 8008, 8009, 8443, 9000

# 새 포트 추가
sudo semanage port -a -t http_port_t -p tcp 8080

# 포트 삭제
sudo semanage port -d -t http_port_t -p tcp 8080

# 포트 수정
sudo semanage port -m -t http_port_t -p tcp 8888
```

---

## 4. SELinux 트러블슈팅

### 감사 로그 확인

```bash
# SELinux 거부 로그 확인
sudo ausearch -m avc -ts recent

# 특정 서비스 관련 로그
sudo ausearch -m avc -c httpd

# 읽기 쉬운 형식으로 변환
sudo ausearch -m avc -ts recent | audit2why
```

### audit2why 사용

```bash
# 거부 이유 분석
sudo cat /var/log/audit/audit.log | audit2why

# 예시 출력:
# type=AVC msg=audit(...): avc:  denied  { read } for  pid=1234
# comm="httpd" name="index.html" dev="sda1" ino=12345
# scontext=system_u:system_r:httpd_t:s0
# tcontext=unconfined_u:object_r:user_home_t:s0 tclass=file
#
# Was caused by:
#   Missing type enforcement (TE) allow rule.
```

### audit2allow로 정책 생성

```bash
# 허용 규칙 생성 (확인만)
sudo ausearch -m avc -ts recent | audit2allow

# 로컬 모듈로 컴파일
sudo ausearch -m avc -ts recent | audit2allow -M mypolicy

# 모듈 설치
sudo semodule -i mypolicy.pp

# 설치된 모듈 확인
sudo semodule -l | grep mypolicy

# 모듈 제거
sudo semodule -r mypolicy
```

### sealert 사용 (GUI/상세 분석)

```bash
# setroubleshoot 패키지 필요
sudo yum install setroubleshoot-server

# 분석 실행
sudo sealert -a /var/log/audit/audit.log

# 실시간 알림 확인
sudo sealert -l "*"
```

### 일반적인 문제 해결

```bash
# 문제: 웹 서버가 파일을 읽지 못함
# 1. 컨텍스트 확인
ls -Z /var/www/html/problem_file

# 2. 컨텍스트 수정
sudo restorecon -v /var/www/html/problem_file

# 문제: 커스텀 포트 사용 불가
# 1. 현재 포트 레이블 확인
sudo semanage port -l | grep 8080

# 2. 포트 추가
sudo semanage port -a -t http_port_t -p tcp 8080

# 문제: 네트워크 연결 거부
# 1. 관련 불리언 확인
getsebool -a | grep httpd_can_network

# 2. 불리언 활성화
sudo setsebool -P httpd_can_network_connect on
```

---

## 5. AppArmor 기초

### AppArmor 상태 확인

```bash
# Ubuntu/Debian
sudo aa-status

# 또는
sudo apparmor_status
```

예시 출력:
```
apparmor module is loaded.
38 profiles are loaded.
36 profiles are in enforce mode.
   /snap/snapd/19457/usr/lib/snapd/snap-confine
   /usr/bin/evince
   ...
2 profiles are in complain mode.
   /usr/sbin/cups-browsed
   /usr/sbin/cupsd
```

### AppArmor 모드

```bash
# Enforce 모드: 정책 위반 차단
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx

# Complain 모드: 위반 로깅만 (차단 안 함)
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx

# 프로파일 비활성화
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx

# 프로파일 다시 로드
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.nginx
```

### 프로파일 위치

```bash
# 시스템 프로파일
ls /etc/apparmor.d/

# 주요 파일들
/etc/apparmor.d/usr.sbin.nginx    # Nginx 프로파일
/etc/apparmor.d/usr.sbin.mysqld   # MySQL 프로파일
/etc/apparmor.d/abstractions/     # 공유 규칙
/etc/apparmor.d/tunables/         # 변수 정의
```

---

## 6. AppArmor 프로파일

### 프로파일 구조

```
#include <tunables/global>

/path/to/program {
  #include <abstractions/base>

  # 파일 접근 규칙
  /etc/myapp.conf r,
  /var/log/myapp.log w,
  /usr/lib/myapp/** r,

  # 네트워크 규칙
  network inet stream,

  # 실행 규칙
  /usr/bin/helper ix,
}
```

### 권한 플래그

| 플래그 | 의미 |
|--------|------|
| `r` | 읽기 |
| `w` | 쓰기 |
| `a` | 추가 |
| `k` | 파일 잠금 |
| `l` | 링크 |
| `m` | 메모리 맵 실행 |
| `x` | 실행 |
| `ix` | 동일 프로파일로 실행 |
| `px` | 다른 프로파일로 실행 |
| `ux` | 제한 없이 실행 |
| `Px` | px + 환경 정리 |
| `Ux` | ux + 환경 정리 |

### 프로파일 작성 예시

```bash
# /etc/apparmor.d/usr.local.bin.myapp
#include <tunables/global>

/usr/local/bin/myapp {
  #include <abstractions/base>
  #include <abstractions/nameservice>

  # 설정 파일 읽기
  /etc/myapp/** r,

  # 데이터 디렉토리
  /var/lib/myapp/ r,
  /var/lib/myapp/** rw,

  # 로그 파일
  /var/log/myapp/ r,
  /var/log/myapp/** rw,
  owner /var/log/myapp/*.log w,

  # 런타임 파일
  /run/myapp.pid rw,
  /run/myapp.sock rw,

  # 라이브러리
  /usr/lib/myapp/** rm,

  # 네트워크 접근
  network inet tcp,
  network inet udp,

  # 시스템 호출 제한
  deny @{PROC}/** w,
  deny /sys/** w,

  # 자식 프로세스
  /usr/bin/logger Px,
}
```

### 자동 프로파일 생성

```bash
# aa-genprof로 프로파일 생성
sudo aa-genprof /usr/local/bin/myapp

# 프로그램을 실행하고 일반적인 작업 수행
# aa-genprof가 접근을 모니터링하고 프로파일 생성

# aa-logprof로 기존 프로파일 업데이트
sudo aa-logprof
```

### Abstractions 사용

```bash
# /etc/apparmor.d/abstractions/ 내 공통 규칙
# base          - 기본 시스템 접근
# nameservice   - DNS, NSS 등
# authentication - PAM, shadow 등
# apache2-common - Apache 공통 규칙
# mysql         - MySQL 클라이언트 접근
# php           - PHP 관련 접근
```

프로파일에서 사용:
```
#include <abstractions/base>
#include <abstractions/nameservice>
```

---

## 7. 실무 시나리오

### 시나리오 1: 웹 서버 커스텀 디렉토리 (SELinux)

```bash
# 문제: /data/www에서 웹 콘텐츠 제공 시 403 에러

# 1. 현재 컨텍스트 확인
ls -Zd /data/www
# drwxr-xr-x. root root unconfined_u:object_r:default_t:s0 /data/www

# 2. 올바른 컨텍스트 설정
sudo semanage fcontext -a -t httpd_sys_content_t "/data/www(/.*)?"
sudo restorecon -Rv /data/www

# 3. 확인
ls -Zd /data/www
# drwxr-xr-x. root root unconfined_u:object_r:httpd_sys_content_t:s0 /data/www
```

### 시나리오 2: PHP 애플리케이션 DB 연결 (SELinux)

```bash
# 문제: PHP에서 원격 MySQL 연결 실패

# 1. 로그 확인
sudo ausearch -m avc -c httpd | audit2why

# 2. 불리언 확인
getsebool httpd_can_network_connect_db
# httpd_can_network_connect_db --> off

# 3. 불리언 활성화
sudo setsebool -P httpd_can_network_connect_db on
```

### 시나리오 3: Nginx 커스텀 포트 (AppArmor)

```bash
# /etc/apparmor.d/local/nginx
# 로컬 커스터마이징용 파일

# 추가 포트 허용
network inet stream,

# 추가 경로 허용
/data/nginx/** r,
/var/log/nginx-custom/ rw,
/var/log/nginx-custom/** rw,
```

```bash
# 프로파일 리로드
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.nginx
```

### 시나리오 4: Docker와 SELinux

```bash
# Docker 컨테이너에서 호스트 볼륨 마운트

# 방법 1: z 옵션 (공유 레이블)
docker run -v /data:/data:z myimage

# 방법 2: Z 옵션 (전용 레이블)
docker run -v /data:/data:Z myimage

# 방법 3: 수동 레이블 지정
sudo chcon -Rt svirt_sandbox_file_t /data
docker run -v /data:/data myimage
```

### 시나리오 5: 새 서비스 프로파일 생성 (AppArmor)

```bash
# 1. complain 모드로 시작
sudo aa-complain /usr/local/bin/newservice

# 2. 서비스 실행 및 모든 기능 테스트

# 3. 로그에서 프로파일 업데이트
sudo aa-logprof

# 4. enforce 모드로 전환
sudo aa-enforce /usr/local/bin/newservice

# 5. 테스트
```

---

## 연습 문제

### 문제 1: SELinux 컨텍스트

다음 상황에서 어떤 명령을 사용해야 할까요?
- `/opt/webapp` 디렉토리를 웹 서버 콘텐츠로 영구 설정
- Apache가 8443 포트를 사용하도록 허용
- httpd가 사용자 홈 디렉토리에 접근하도록 허용

### 문제 2: AppArmor 프로파일

`/usr/local/bin/backup.sh` 스크립트가 다음 작업을 수행합니다:
- `/etc/` 읽기
- `/var/backup/`에 쓰기
- `rsync` 실행
- TCP 22번 포트 네트워크 접근

이 스크립트의 AppArmor 프로파일을 작성하세요.

### 문제 3: 트러블슈팅

SELinux Enforcing 모드에서 웹 애플리케이션이 작동하지 않습니다:
1. 문제를 진단하는 단계를 나열하세요
2. 어떤 도구를 사용해야 할까요?

---

## 정답

### 문제 1 정답

```bash
# 웹 콘텐츠 설정
sudo semanage fcontext -a -t httpd_sys_content_t "/opt/webapp(/.*)?"
sudo restorecon -Rv /opt/webapp

# 포트 추가
sudo semanage port -a -t http_port_t -p tcp 8443

# 홈 디렉토리 접근 허용
sudo setsebool -P httpd_enable_homedirs on
```

### 문제 2 정답

```
#include <tunables/global>

/usr/local/bin/backup.sh {
  #include <abstractions/base>
  #include <abstractions/bash>

  # 설정 읽기
  /etc/** r,

  # 백업 디렉토리
  /var/backup/ r,
  /var/backup/** rw,

  # rsync 실행
  /usr/bin/rsync Px,

  # SSH 네트워크
  network inet stream,
  network inet6 stream,
}
```

### 문제 3 정답

```bash
# 1. SELinux 로그 확인
sudo ausearch -m avc -ts recent

# 2. 원인 분석
sudo ausearch -m avc -ts recent | audit2why

# 3. 상세 분석 (setroubleshoot 설치 시)
sudo sealert -a /var/log/audit/audit.log

# 4. 해결책 적용
# - 컨텍스트 문제: restorecon, semanage fcontext
# - 불리언 문제: setsebool
# - 포트 문제: semanage port
# - 정책 필요: audit2allow로 커스텀 모듈 생성
```

---

## 다음 단계

- [18_Log_Management.md](./18_Log_Management.md) - journald, rsyslog, logrotate 학습

---

## 참고 자료

- [SELinux User Guide (Red Hat)](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/using_selinux/index)
- [AppArmor Wiki](https://gitlab.com/apparmor/apparmor/-/wikis/home)
- [SELinux Project Wiki](https://selinuxproject.org/page/Main_Page)
- `man semanage`, `man restorecon`, `man audit2why`
- `man apparmor`, `man aa-status`, `man apparmor.d`
