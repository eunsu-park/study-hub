# 사용자와 그룹 관리

**이전**: [권한과 소유권](./05_Permissions_Ownership.md) | **다음**: [프로세스 관리](./07_Process_Management.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. /etc/passwd, /etc/shadow, /etc/group의 구조와 목적을 설명한다
2. useradd, usermod, userdel로 사용자 계정을 생성, 수정, 삭제한다
3. groupadd, groupmod, groupdel, gpasswd를 사용해 그룹을 관리한다
4. su로 사용자를 전환하고 sudo로 권한을 상승한다
5. visudo로 /etc/sudoers를 안전하게 편집해 sudo 권한을 설정한다
6. id, groups, who, w, last를 사용해 사용자 및 그룹 정보를 조회한다
7. 비밀번호 정책(password policy)과 계정 관리에 보안 모범 사례를 적용한다

---

리눅스는 설계상 다중 사용자(multi-user) 운영체제입니다. 단일 사용자 워크스테이션에서도 웹 서버, 데이터베이스, 메일 등 서비스를 위한 수십 개의 시스템 계정이 실행됩니다. 사용자와 그룹을 올바르게 관리하는 것은 접근 제어(access control)의 기초로, 누가 로그인할 수 있는지, 어떤 파일에 접근할 수 있는지, 어떤 명령어를 상승된 권한으로 실행할 수 있는지를 결정합니다.

## 1. 사용자 관련 파일

리눅스는 사용자 정보를 특정 파일들에 저장합니다.

### /etc/passwd

사용자 계정 정보를 저장합니다.

```bash
cat /etc/passwd | head -5
```

출력:
```
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
```

```
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
  │    │  │    │     │        │           │
  │    │  │    │     │        │           └── 로그인 쉘
  │    │  │    │     │        └── 홈 디렉토리
  │    │  │    │     └── 설명 (GECOS)
  │    │  │    └── 기본 그룹 ID (GID)
  │    │  └── 사용자 ID (UID)
  │    └── 비밀번호 (x = shadow 파일에 저장)
  └── 사용자명
```

### /etc/shadow

암호화된 비밀번호를 저장합니다 (root만 읽기 가능).

```bash
sudo cat /etc/shadow | head -3
```

출력:
```
root:$6$xxxx...:19000:0:99999:7:::
ubuntu:$6$yyyy...:19000:0:99999:7:::
```

```
ubuntu:$6$...:19000:0:99999:7:::
  │      │     │   │   │   │
  │      │     │   │   │   └── 비밀번호 만료 경고 일수
  │      │     │   │   └── 비밀번호 최대 사용 일수
  │      │     │   └── 비밀번호 최소 사용 일수
  │      │     └── 마지막 변경일 (1970년 1월 1일 기준 일수)
  │      └── 암호화된 비밀번호
  └── 사용자명
```

### /etc/group

그룹 정보를 저장합니다.

```bash
cat /etc/group | head -5
```

출력:
```
root:x:0:
daemon:x:1:
ubuntu:x:1000:
sudo:x:27:ubuntu
developers:x:1001:alice,bob
```

```
developers:x:1001:alice,bob
    │      │  │      │
    │      │  │      └── 그룹 멤버 (추가 멤버)
    │      │  └── 그룹 ID (GID)
    │      └── 비밀번호 (보통 사용 안 함)
    └── 그룹명
```

---

## 2. 사용자 관리 명령어

### useradd - 사용자 생성

```bash
# 기본 생성
sudo useradd username

# 옵션과 함께 생성 (권장)
sudo useradd -m -s /bin/bash -c "John Doe" john

# 주요 옵션
# -m : 홈 디렉토리 생성
# -s : 로그인 쉘 지정
# -c : 설명(코멘트)
# -d : 홈 디렉토리 경로 지정
# -g : 기본 그룹
# -G : 추가 그룹
# -u : UID 지정
```

```bash
# 여러 그룹에 추가하며 생성
sudo useradd -m -s /bin/bash -G sudo,developers newuser

# 비밀번호 설정
sudo passwd newuser
```

### adduser - 대화형 사용자 생성 (Ubuntu/Debian)

```bash
# 대화형으로 사용자 생성 (더 편리)
sudo adduser newuser
```

출력:
```
Adding user `newuser' ...
Adding new group `newuser' (1002) ...
Adding new user `newuser' (1002) with group `newuser' ...
Creating home directory `/home/newuser' ...
Copying files from `/etc/skel' ...
New password:
Retype new password:
passwd: password updated successfully
Full Name []: New User
Room Number []:
Work Phone []:
Home Phone []:
Other []:
Is the information correct? [Y/n] y
```

### usermod - 사용자 수정

```bash
# 쉘 변경
sudo usermod -s /bin/zsh username

# 홈 디렉토리 변경
sudo usermod -d /home/newhome -m username

# 그룹에 추가 (기존 그룹 유지)
sudo usermod -aG sudo username
sudo usermod -aG docker,developers username

# 사용자명 변경
sudo usermod -l newname oldname

# 계정 잠금
sudo usermod -L username

# 계정 잠금 해제
sudo usermod -U username
```

### userdel - 사용자 삭제

```bash
# 사용자만 삭제
sudo userdel username

# 홈 디렉토리와 메일도 삭제
sudo userdel -r username
```

### passwd - 비밀번호 관리

```bash
# 자신의 비밀번호 변경
passwd

# 다른 사용자 비밀번호 변경 (root)
sudo passwd username

# 비밀번호 만료시키기 (다음 로그인 시 변경 강제)
sudo passwd -e username

# 비밀번호 잠금
sudo passwd -l username

# 비밀번호 잠금 해제
sudo passwd -u username

# 비밀번호 상태 확인
sudo passwd -S username
```

---

## 3. 그룹 관리 명령어

### groupadd - 그룹 생성

```bash
# 그룹 생성
sudo groupadd developers

# GID 지정
sudo groupadd -g 2000 mygroup
```

### groupmod - 그룹 수정

```bash
# 그룹명 변경
sudo groupmod -n newname oldname

# GID 변경
sudo groupmod -g 2001 groupname
```

### groupdel - 그룹 삭제

```bash
sudo groupdel groupname
```

### gpasswd - 그룹 멤버 관리

```bash
# 사용자를 그룹에 추가
sudo gpasswd -a username groupname

# 사용자를 그룹에서 제거
sudo gpasswd -d username groupname

# 그룹 관리자 지정
sudo gpasswd -A adminuser groupname
```

---

## 4. 사용자 전환

### su - 사용자 전환

```bash
# 다른 사용자로 전환
su username

# root로 전환
su -
su - root

# 환경변수 포함 전환 (권장)
su - username

# 명령 하나만 실행
su -c 'command' username
```

### sudo - 권한 상승

```bash
# 관리자 권한으로 명령 실행
sudo command

# 다른 사용자로 명령 실행
sudo -u username command

# root 쉘 열기
sudo -i

# 환경변수 유지
sudo -E command

# sudo 권한 캐시 초기화
sudo -k
```

---

## 5. sudo 설정

### /etc/sudoers

sudo 권한을 설정하는 파일입니다. **항상 visudo로 편집해야 합니다.**

```bash
sudo visudo
```

### 기본 형식

```
# 사용자별 설정
사용자   호스트=(실행사용자) 명령어

# 그룹별 설정 (% 접두사)
%그룹   호스트=(실행사용자) 명령어
```

### 설정 예시

```bash
# root는 모든 권한
root    ALL=(ALL:ALL) ALL

# sudo 그룹 멤버는 모든 권한
%sudo   ALL=(ALL:ALL) ALL

# 특정 사용자에게 모든 권한
john    ALL=(ALL:ALL) ALL

# 비밀번호 없이 sudo 허용
john    ALL=(ALL) NOPASSWD: ALL

# 특정 명령만 허용
backup  ALL=(ALL) /usr/bin/rsync, /usr/bin/tar

# 특정 명령 비밀번호 없이
deploy  ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nginx
```

### /etc/sudoers.d/

별도 파일로 설정을 관리할 수 있습니다.

```bash
# 파일 생성
sudo visudo -f /etc/sudoers.d/developers

# 내용
%developers ALL=(ALL) NOPASSWD: /usr/bin/docker
```

---

## 6. 사용자 정보 확인

### id - 사용자 ID 정보

```bash
# 현재 사용자
id

# 특정 사용자
id username
```

출력:
```
uid=1000(ubuntu) gid=1000(ubuntu) groups=1000(ubuntu),27(sudo),999(docker)
```

### groups - 그룹 멤버십

```bash
# 현재 사용자 그룹
groups

# 특정 사용자 그룹
groups username
```

### who - 로그인 사용자

```bash
# 현재 로그인 사용자
who
```

출력:
```
ubuntu   pts/0        2024-01-23 10:00 (192.168.1.100)
```

### w - 상세 로그인 정보

```bash
w
```

출력:
```
 10:30:00 up 5 days,  3:45,  2 users,  load average: 0.00, 0.01, 0.05
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
ubuntu   pts/0    192.168.1.100    10:00    0.00s  0.03s  0.00s w
john     pts/1    192.168.1.101    10:15    5:00   0.01s  0.01s bash
```

### last - 로그인 기록

```bash
# 최근 로그인 기록
last

# 특정 사용자
last username

# 최근 10개
last -n 10

# 재부팅 기록
last reboot
```

### lastlog - 마지막 로그인

```bash
lastlog
```

---

## 7. 시스템 사용자

| UID 범위 | 용도 |
|----------|------|
| 0 | root |
| 1-999 | 시스템 사용자 |
| 1000+ | 일반 사용자 |

### 시스템 사용자 생성

```bash
# 시스템 사용자 (로그인 불가)
sudo useradd -r -s /usr/sbin/nologin serviceuser
```

### 주요 시스템 사용자

| 사용자 | 용도 |
|--------|------|
| root | 시스템 관리자 |
| www-data | 웹 서버 |
| mysql | MySQL 데이터베이스 |
| postgres | PostgreSQL |
| nobody | 최소 권한 프로세스 |

---

## 8. 실무 예제

### 개발팀 환경 구성

```bash
# 1. 개발팀 그룹 생성
sudo groupadd developers

# 2. 개발자 계정 생성
sudo useradd -m -s /bin/bash -G developers alice
sudo useradd -m -s /bin/bash -G developers bob
sudo passwd alice
sudo passwd bob

# 3. 공유 디렉토리 설정
sudo mkdir -p /projects/shared
sudo chgrp developers /projects/shared
sudo chmod 2775 /projects/shared

# 4. sudo 권한 부여 (Docker 명령만)
sudo visudo -f /etc/sudoers.d/developers
# %developers ALL=(ALL) NOPASSWD: /usr/bin/docker
```

### 웹 개발자 환경

```bash
# 웹 개발자 계정
sudo useradd -m -s /bin/bash -G www-data,developers webdev
sudo passwd webdev

# 웹 디렉토리 권한
sudo chown -R webdev:www-data /var/www/mysite
sudo chmod -R 775 /var/www/mysite
```

### 배포 전용 계정

```bash
# 배포 계정 (로그인은 키로만)
sudo useradd -m -s /bin/bash deploy
sudo mkdir -p /home/deploy/.ssh
sudo chmod 700 /home/deploy/.ssh

# SSH 키 설정
sudo touch /home/deploy/.ssh/authorized_keys
sudo chmod 600 /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh

# 제한된 sudo 권한
sudo visudo -f /etc/sudoers.d/deploy
# deploy ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart myapp
```

---

## 9. 보안 모범 사례

### 비밀번호 정책

```bash
# /etc/login.defs 수정
sudo vi /etc/login.defs
```

```
PASS_MAX_DAYS   90     # 최대 사용 기간
PASS_MIN_DAYS   7      # 최소 사용 기간
PASS_WARN_AGE   14     # 만료 경고 일수
PASS_MIN_LEN    12     # 최소 길이
```

### root 직접 로그인 비활성화

```bash
# SSH에서 root 로그인 비활성화
sudo vi /etc/ssh/sshd_config
# PermitRootLogin no

sudo systemctl restart sshd
```

### 불필요한 계정 잠금

```bash
# 사용하지 않는 계정 잠금
sudo passwd -l unuseduser

# 쉘을 nologin으로
sudo usermod -s /usr/sbin/nologin unuseduser
```

---

## 10. 실습 예제

### 실습 1: 사용자 정보 확인

```bash
# 현재 사용자 정보
id
groups
whoami

# /etc/passwd 확인
grep $USER /etc/passwd

# 로그인 기록
last -n 5
```

### 실습 2: 사용자 생성과 삭제

```bash
# 테스트 사용자 생성
sudo useradd -m -s /bin/bash -c "Test User" testuser
sudo passwd testuser

# 확인
id testuser
grep testuser /etc/passwd
ls -la /home/testuser

# 삭제
sudo userdel -r testuser
```

### 실습 3: 그룹 관리

```bash
# 그룹 생성
sudo groupadd testgroup

# 사용자 추가
sudo usermod -aG testgroup $USER

# 확인 (재로그인 필요)
groups

# 그룹 삭제
sudo groupdel testgroup
```

### 실습 4: sudo 테스트

```bash
# sudo 권한 확인
sudo -l

# root 권한으로 명령 실행
sudo whoami

# 다른 사용자로 명령 실행
sudo -u www-data whoami
```

---

## 다음 단계

[프로세스 관리](./07_Process_Management.md)에서 프로세스 관리를 배워봅시다!
