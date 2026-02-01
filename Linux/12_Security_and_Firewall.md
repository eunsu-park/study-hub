# 보안과 방화벽

## 1. 보안 기본 원칙

### 최소 권한 원칙

```
┌─────────────────────────────────────────────────────────┐
│                    보안 계층                             │
├─────────────────────────────────────────────────────────┤
│  1. 물리적 보안 - 서버실 접근 통제                        │
│  2. 네트워크 보안 - 방화벽, VPN                          │
│  3. 호스트 보안 - OS 설정, 패치                          │
│  4. 애플리케이션 보안 - 취약점 관리                       │
│  5. 데이터 보안 - 암호화, 백업                           │
└─────────────────────────────────────────────────────────┘
```

### 기본 보안 점검 사항

- [ ] 불필요한 서비스 비활성화
- [ ] 기본 포트 변경 (SSH 등)
- [ ] 강력한 비밀번호 정책
- [ ] 정기적인 보안 업데이트
- [ ] 로그 모니터링
- [ ] 방화벽 설정
- [ ] SSH 키 인증 사용

---

## 2. SSH 보안 설정

### sshd_config 설정

```bash
sudo vi /etc/ssh/sshd_config
```

### 권장 설정

```bash
# 포트 변경 (기본 22 → 다른 포트)
Port 2222

# root 로그인 비활성화
PermitRootLogin no

# 비밀번호 인증 비활성화 (키 인증만)
PasswordAuthentication no

# 빈 비밀번호 비허용
PermitEmptyPasswords no

# 특정 사용자만 허용
AllowUsers ubuntu deploy

# 특정 그룹만 허용
AllowGroups sshusers

# 로그인 시도 제한
MaxAuthTries 3

# 유휴 타임아웃
ClientAliveInterval 300
ClientAliveCountMax 2

# X11 포워딩 비활성화
X11Forwarding no

# 프로토콜 2만 사용 (대부분 기본값)
Protocol 2
```

### 설정 적용

```bash
# 설정 검증
sudo sshd -t

# 서비스 재시작
sudo systemctl restart sshd
```

### SSH 키 관리

```bash
# 키 생성 (ed25519 권장)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 키 권한 설정 (필수)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/authorized_keys
```

---

## 3. 방화벽 - UFW (Ubuntu)

UFW(Uncomplicated Firewall)는 Ubuntu의 기본 방화벽입니다.

### 기본 명령어

```bash
# 상태 확인
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered

# 활성화/비활성화
sudo ufw enable
sudo ufw disable

# 기본 정책 설정
sudo ufw default deny incoming    # 들어오는 연결 거부 (기본)
sudo ufw default allow outgoing   # 나가는 연결 허용 (기본)
```

### 규칙 추가

```bash
# 포트 허용
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443

# 포트 범위 허용
sudo ufw allow 6000:6010/tcp

# 서비스명으로 허용
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https

# 특정 IP에서만 허용
sudo ufw allow from 192.168.1.100
sudo ufw allow from 192.168.1.100 to any port 22

# 서브넷 허용
sudo ufw allow from 192.168.1.0/24

# TCP/UDP 지정
sudo ufw allow 53/tcp
sudo ufw allow 53/udp
```

### 규칙 삭제

```bash
# 규칙 번호로 삭제
sudo ufw status numbered
sudo ufw delete 2

# 규칙 직접 삭제
sudo ufw delete allow 80
```

### 고급 설정

```bash
# 속도 제한 (DoS 방지)
sudo ufw limit ssh    # SSH 연결 제한 (30초에 6회)

# 로깅
sudo ufw logging on
sudo ufw logging high

# 특정 인터페이스
sudo ufw allow in on eth0 to any port 80
```

### 일반적인 서버 설정

```bash
# 기본 정책
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH (포트 변경 시 해당 포트)
sudo ufw allow 2222/tcp

# 웹 서버
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 활성화
sudo ufw enable
```

---

## 4. 방화벽 - firewalld (CentOS/RHEL)

firewalld는 CentOS/RHEL의 기본 방화벽입니다.

### 기본 명령어

```bash
# 상태 확인
sudo firewall-cmd --state
sudo systemctl status firewalld

# 활성화/비활성화
sudo systemctl start firewalld
sudo systemctl stop firewalld
sudo systemctl enable firewalld

# 설정 다시 로드
sudo firewall-cmd --reload
```

### Zone 개념

| Zone | 설명 |
|------|------|
| drop | 모든 연결 거부 |
| block | 연결 거부 + ICMP 응답 |
| public | 공개 (기본) |
| external | 외부 (NAT) |
| dmz | DMZ |
| work | 업무 |
| home | 가정 |
| internal | 내부 |
| trusted | 모든 연결 허용 |

```bash
# 현재 zone 확인
sudo firewall-cmd --get-default-zone

# zone 목록
sudo firewall-cmd --get-zones

# zone 변경
sudo firewall-cmd --set-default-zone=public
```

### 규칙 추가

```bash
# 서비스 허용
sudo firewall-cmd --add-service=ssh --permanent
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-service=https --permanent

# 포트 허용
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --add-port=3000-3010/tcp --permanent

# 특정 IP 허용
sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.100" accept' --permanent

# 특정 IP에서 특정 포트
sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="22" protocol="tcp" accept' --permanent

# 설정 적용
sudo firewall-cmd --reload
```

### 규칙 삭제

```bash
# 서비스 제거
sudo firewall-cmd --remove-service=http --permanent

# 포트 제거
sudo firewall-cmd --remove-port=8080/tcp --permanent

# 적용
sudo firewall-cmd --reload
```

### 설정 확인

```bash
# 현재 설정
sudo firewall-cmd --list-all

# 서비스 목록
sudo firewall-cmd --list-services

# 포트 목록
sudo firewall-cmd --list-ports
```

---

## 5. SELinux (CentOS/RHEL)

### 상태 확인

```bash
# 현재 상태
getenforce
sestatus
```

### 모드

| 모드 | 설명 |
|------|------|
| Enforcing | 정책 적용 (기본) |
| Permissive | 로그만 기록 |
| Disabled | 비활성화 |

### 모드 변경

```bash
# 임시 변경 (재부팅 시 복원)
sudo setenforce 0    # Permissive
sudo setenforce 1    # Enforcing

# 영구 변경
sudo vi /etc/selinux/config
# SELINUX=enforcing → SELINUX=permissive
```

### SELinux 문제 해결

```bash
# 거부 로그 확인
sudo ausearch -m avc -ts recent

# 문제 분석 (audit2why 필요)
sudo ausearch -m avc | audit2why

# 포트 허용 예시
sudo semanage port -a -t http_port_t -p tcp 8080
```

---

## 6. AppArmor (Ubuntu)

### 상태 확인

```bash
# 상태
sudo aa-status

# 프로파일 목록
ls /etc/apparmor.d/
```

### 모드

```bash
# 프로파일 enforce 모드
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx

# 프로파일 complain 모드 (로그만)
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx

# 프로파일 비활성화
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx
```

---

## 7. Fail2ban

fail2ban은 로그를 모니터링하여 악의적인 시도를 차단합니다.

### 설치

```bash
# Ubuntu
sudo apt install fail2ban

# CentOS
sudo dnf install fail2ban
```

### 기본 설정

```bash
# 설정 파일 복사
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo vi /etc/fail2ban/jail.local
```

### 설정 예시

```ini
[DEFAULT]
# 차단 시간 (초)
bantime = 3600

# 감시 시간 (초)
findtime = 600

# 최대 실패 횟수
maxretry = 5

# 이메일 알림
destemail = admin@example.com
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh,2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400
```

### 관리 명령어

```bash
# 시작/중지
sudo systemctl start fail2ban
sudo systemctl enable fail2ban

# 상태 확인
sudo fail2ban-client status
sudo fail2ban-client status sshd

# 차단된 IP 확인
sudo fail2ban-client status sshd | grep "Banned IP"

# IP 차단 해제
sudo fail2ban-client set sshd unbanip 192.168.1.100

# IP 수동 차단
sudo fail2ban-client set sshd banip 192.168.1.100
```

---

## 8. 보안 업데이트

### Ubuntu/Debian

```bash
# 업데이트 확인
sudo apt update
apt list --upgradable

# 보안 업데이트만
sudo apt upgrade -y

# 자동 업데이트 설정
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

### CentOS/RHEL

```bash
# 업데이트 확인
sudo dnf check-update

# 보안 업데이트만
sudo dnf upgrade --security

# 자동 업데이트 설정
sudo dnf install dnf-automatic
sudo systemctl enable --now dnf-automatic.timer
```

---

## 9. 보안 점검 체크리스트

### 사용자 점검

```bash
# 비밀번호 없는 계정
sudo awk -F: '($2 == "") {print $1}' /etc/shadow

# UID 0 계정 (root 외)
awk -F: '($3 == 0) {print $1}' /etc/passwd

# 최근 로그인 실패
sudo lastb | head -20

# sudo 권한 사용자
grep -Po '^sudo.+:\K.*$' /etc/group
```

### 서비스 점검

```bash
# 실행 중인 서비스
systemctl list-units --type=service --state=running

# 열린 포트
ss -tuln

# 불필요한 서비스 확인
systemctl list-unit-files --type=service | grep enabled
```

### 파일 권한 점검

```bash
# world-writable 파일
find / -type f -perm -002 2>/dev/null

# SUID 파일
find / -perm -4000 2>/dev/null

# SGID 파일
find / -perm -2000 2>/dev/null

# 소유자 없는 파일
find / -nouser -o -nogroup 2>/dev/null
```

---

## 10. 실습 예제

### 실습 1: SSH 보안 강화

```bash
# 현재 설정 백업
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# 설정 변경 (예: root 로그인 비활성화)
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# 설정 검증
sudo sshd -t

# 재시작
sudo systemctl restart sshd
```

### 실습 2: 방화벽 설정 (Ubuntu)

```bash
# 현재 상태
sudo ufw status

# 기본 정책
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH 허용
sudo ufw allow ssh

# 활성화
sudo ufw enable

# 확인
sudo ufw status verbose
```

### 실습 3: 방화벽 설정 (CentOS)

```bash
# 현재 상태
sudo firewall-cmd --list-all

# SSH 허용
sudo firewall-cmd --add-service=ssh --permanent

# 적용
sudo firewall-cmd --reload

# 확인
sudo firewall-cmd --list-services
```

### 실습 4: fail2ban 설정

```bash
# 설치
sudo apt install fail2ban    # Ubuntu
# sudo dnf install fail2ban  # CentOS

# 기본 설정
sudo systemctl start fail2ban
sudo systemctl enable fail2ban

# 상태 확인
sudo fail2ban-client status sshd
```

### 실습 5: 보안 점검

```bash
# 열린 포트 확인
ss -tuln

# 로그인 실패 기록
sudo lastb | head -10

# 보안 업데이트 확인
apt list --upgradable 2>/dev/null | grep -i security

# SUID 파일 확인
find /usr/bin -perm -4000 2>/dev/null
```

---

## 축하합니다!

Linux 학습 자료를 모두 완료했습니다. 이제 다음 단계로:

- 실제 서버에서 실습
- Docker 컨테이너 활용: [Docker/](../Docker/00_Overview.md)
- 데이터베이스 운영: [PostgreSQL/](../PostgreSQL/00_Overview.md)
- 자동화 스크립트 작성 연습
