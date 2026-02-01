# Linux 학습 가이드

## 소개

이 폴더는 Linux 운영체제의 기초부터 서버 관리까지 체계적으로 학습할 수 있는 자료를 담고 있습니다.

- **대상 독자**: 리눅스 입문자 ~ 서버 관리자
- **배포판**: Ubuntu/Debian 및 CentOS/RHEL 모두 안내
- **목표**: 명령어 사용부터 서버 운영까지

---

## 학습 로드맵

```
[초급]                [중급]                 [고급]
  │                     │                      │
  ▼                     ▼                      ▼
리눅스 기초 ──────▶ 텍스트 처리 ──────▶ 쉘 스크립팅
  │                     │                      │
  ▼                     ▼                      ▼
파일시스템 ──────▶ 권한/소유권 ─────▶ 네트워크 기초
  │                     │                      │
  ▼                     ▼                      ▼
파일 관리 ───────▶ 사용자 관리 ─────▶ 시스템 모니터링
                        │                      │
                        ▼                      ▼
                  프로세스 관리 ────▶ 보안과 방화벽
                        │
                        ▼
                  패키지 관리
```

---

## 선수 지식

- 기본적인 컴퓨터 사용 능력
- 터미널/명령 프롬프트 개념 이해
- 영어 명령어 읽기 (필수 아님)

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Linux_Basics.md](./01_Linux_Basics.md) | ⭐ | 리눅스 개념, 배포판, 터미널, 기본 명령어 |
| [02_Filesystem_Navigation.md](./02_Filesystem_Navigation.md) | ⭐ | 디렉토리 구조, 경로, ls, cd, find |
| [03_File_Directory_Management.md](./03_File_Directory_Management.md) | ⭐ | touch, mkdir, cp, mv, rm, tar |
| [04_Text_Processing.md](./04_Text_Processing.md) | ⭐⭐ | grep, sed, awk, 파이프, 리다이렉션 |
| [05_Permissions_Ownership.md](./05_Permissions_Ownership.md) | ⭐⭐ | chmod, chown, 특수 권한, umask |
| [06_User_Group_Management.md](./06_User_Group_Management.md) | ⭐⭐ | useradd, sudo, 사용자/그룹 관리 |
| [07_Process_Management.md](./07_Process_Management.md) | ⭐⭐ | ps, top, kill, systemctl |
| [08_Package_Management.md](./08_Package_Management.md) | ⭐⭐ | apt, yum/dnf, 저장소 관리 |
| [09_Shell_Scripting.md](./09_Shell_Scripting.md) | ⭐⭐⭐ | 변수, 조건문, 반복문, 실무 스크립트 |
| [10_Network_Basics.md](./10_Network_Basics.md) | ⭐⭐⭐ | ip, ssh, 포트 확인, 원격 접속 |
| [11_System_Monitoring.md](./11_System_Monitoring.md) | ⭐⭐⭐ | df, free, 로그, cron |
| [12_Security_and_Firewall.md](./12_Security_and_Firewall.md) | ⭐⭐⭐⭐ | SSH 보안, ufw, firewalld, fail2ban |
| [13_Systemd_Advanced.md](./13_Systemd_Advanced.md) | ⭐⭐⭐⭐ | 서비스 유닛, 타이머, 소켓, journald |
| [14_Performance_Tuning.md](./14_Performance_Tuning.md) | ⭐⭐⭐⭐ | sysctl, 커널 파라미터, perf, flamegraph |
| [15_Container_Internals.md](./15_Container_Internals.md) | ⭐⭐⭐⭐ | cgroups, namespaces, 컨테이너 런타임 |
| [16_Storage_Management.md](./16_Storage_Management.md) | ⭐⭐⭐⭐ | LVM, RAID, 파일시스템, LUKS 암호화 |
| [17_SELinux_AppArmor.md](./17_SELinux_AppArmor.md) | ⭐⭐⭐⭐ | SELinux 정책, AppArmor 프로파일, 트러블슈팅 |
| [18_Log_Management.md](./18_Log_Management.md) | ⭐⭐⭐ | journald, rsyslog, logrotate, 원격 로그 |
| [19_Backup_Recovery.md](./19_Backup_Recovery.md) | ⭐⭐⭐⭐ | rsync, Borg Backup, 재해복구 전략 |
| [20_Kernel_Management.md](./20_Kernel_Management.md) | ⭐⭐⭐⭐ | 커널 컴파일, 모듈, DKMS, GRUB |
| [21_Virtualization_KVM.md](./21_Virtualization_KVM.md) | ⭐⭐⭐⭐ | libvirt, virsh, VM 관리, 스냅샷 |
| [22_Ansible_Basics.md](./22_Ansible_Basics.md) | ⭐⭐⭐ | 인벤토리, playbook, roles, Vault |
| [23_Advanced_Networking.md](./23_Advanced_Networking.md) | ⭐⭐⭐⭐ | VLAN, bonding, iptables/nftables |
| [24_Cloud_Integration.md](./24_Cloud_Integration.md) | ⭐⭐⭐ | cloud-init, AWS CLI, 메타데이터 |
| [25_High_Availability_Cluster.md](./25_High_Availability_Cluster.md) | ⭐⭐⭐⭐⭐ | Pacemaker, Corosync, DRBD |
| [26_Troubleshooting_Guide.md](./26_Troubleshooting_Guide.md) | ⭐⭐⭐ | 부팅, 네트워크, 디스크, 성능 문제 해결 |

---

## 추천 학습 순서

### 1단계: 리눅스 입문 (초급)

```
01_Linux_Basics → 02_Filesystem_Navigation → 03_File_Directory_Management
```

터미널 사용법과 기본 명령어를 익힙니다.

### 2단계: 실무 활용 (중급)

```
04_Text_Processing → 05_Permissions_Ownership → 06_User_Group_Management
    → 07_Process_Management → 08_Package_Management
```

파일 처리, 권한 관리, 시스템 운영 기초를 배웁니다.

### 3단계: 서버 관리자 (고급)

```
09_Shell_Scripting → 10_Network_Basics → 11_System_Monitoring → 12_Security_and_Firewall
```

자동화, 네트워크, 모니터링, 보안까지 서버 관리 전반을 다룹니다.

### 4단계: 시스템 심화 (전문가)

```
13_Systemd_Advanced → 14_Performance_Tuning → 15_Container_Internals → 16_Storage_Management
```

systemd, 성능 최적화, 컨테이너 원리, 스토리지 관리를 학습합니다.

### 5단계: 엔터프라이즈 운영 (전문가)

```
17_SELinux_AppArmor → 18_Log_Management → 19_Backup_Recovery → 20_Kernel_Management
```

보안 모듈, 로그 관리, 백업 전략, 커널 관리를 다룹니다.

### 6단계: 인프라 엔지니어링 (전문가)

```
21_Virtualization_KVM → 22_Ansible_Basics → 23_Advanced_Networking → 24_Cloud_Integration
    → 25_High_Availability_Cluster → 26_Troubleshooting_Guide
```

가상화, 자동화, 고급 네트워킹, 클라우드, HA, 트러블슈팅을 마스터합니다.

---

## 실습 환경

### Ubuntu (권장)

```bash
# Docker로 빠르게 시작
docker run -it ubuntu:22.04 bash

# 또는 VM/WSL 사용
# - VirtualBox + Ubuntu ISO
# - Windows WSL2
```

### CentOS/RHEL

```bash
# Docker로 시작
docker run -it rockylinux:9 bash

# 또는 VM 사용
# - VirtualBox + Rocky Linux ISO
```

### 클라우드 (실습용)

- AWS EC2 Free Tier
- Google Cloud Free Tier
- DigitalOcean (유료)

---

## 배포판 비교

| 항목 | Ubuntu/Debian | CentOS/RHEL |
|------|---------------|-------------|
| 패키지 관리 | APT (`apt`) | YUM/DNF (`dnf`) |
| 패키지 형식 | .deb | .rpm |
| 방화벽 | UFW | firewalld |
| 보안 모듈 | AppArmor | SELinux |
| 서비스 관리 | systemctl | systemctl |
| 주요 용도 | 데스크톱, 서버 | 엔터프라이즈 서버 |

---

## 관련 자료

- [Docker/](../Docker/00_Overview.md) - 컨테이너 환경에서 Linux 활용
- [Git/](../Git/00_Overview.md) - Linux에서 버전 관리
- [PostgreSQL/](../PostgreSQL/00_Overview.md) - Linux 서버에서 DB 운영
