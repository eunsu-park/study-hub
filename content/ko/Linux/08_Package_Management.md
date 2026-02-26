# 패키지 관리

**이전**: [프로세스 관리](./07_Process_Management.md) | **다음**: [셸 스크립팅](./09_Shell_Scripting.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 패키지 관리자가 의존성을 해결하고 무결성을 검증하는 방법 설명
2. APT(Ubuntu/Debian)를 사용하여 패키지 설치, 업데이트, 검색, 제거
3. DNF/YUM(CentOS/RHEL)을 사용하여 동일한 작업 수행
4. 서드파티 저장소 추가 및 GPG 키 관리
5. dpkg와 rpm을 사용하여 패키지 정보 조회 및 파일 소유 패키지 탐색
6. 패키지를 사용할 수 없을 때 소스에서 소프트웨어 컴파일
7. 의도치 않은 업그레이드를 방지하기 위한 패키지 버전 고정
8. CentOS EOL 이후 마이그레이션 옵션(Rocky Linux, AlmaLinux) 평가

---

리눅스에서 소프트웨어를 설치하는 방식은 Windows나 macOS에서 설치 프로그램을 내려받는 방식과 근본적으로 다릅니다. 패키지 관리자(Package Manager)는 소프트웨어 다운로드, 의존성 해결, 체크섬 검증, 올바른 위치에 파일 배치까지 모든 것을 자동으로 처리합니다. 패키지 관리자를 통해 시스템을 최신 상태로 유지하는 것은 보안과 안정성을 위해 할 수 있는 가장 중요한 일 중 하나입니다.

## 1. 패키지 관리 개념

패키지 관리자는 소프트웨어 설치, 업데이트, 제거를 자동화합니다.

```
┌─────────────────────────────────────────────────────────┐
│                    패키지 저장소                          │
│              (Repository / Mirror)                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    패키지 관리자                          │
│         APT (Debian/Ubuntu) / DNF (RHEL/CentOS)         │
│                                                          │
│  • 의존성 자동 해결                                       │
│  • 버전 관리                                              │
│  • 무결성 검증                                            │
└─────────────────────────────────────────────────────────┘
```

### 배포판별 패키지 관리자

| 배포판 | 패키지 형식 | 저수준 도구 | 고수준 도구 |
|--------|------------|------------|------------|
| Ubuntu/Debian | .deb | dpkg | apt |
| CentOS/RHEL 8+ | .rpm | rpm | dnf |
| CentOS/RHEL 7 | .rpm | rpm | yum |
| Fedora | .rpm | rpm | dnf |

---

## 2. APT (Ubuntu/Debian)

### 저장소 업데이트

```bash
# 패키지 목록 업데이트 (설치 전 필수)
sudo apt update
```

### 패키지 설치

```bash
# 패키지 설치
sudo apt install nginx

# 여러 패키지
sudo apt install nginx php mysql-server

# 확인 없이 설치
sudo apt install -y vim

# 특정 버전 설치
sudo apt install nginx=1.18.0-0ubuntu1
```

### 패키지 제거

```bash
# 패키지만 제거
sudo apt remove nginx

# 패키지와 설정 파일 제거
sudo apt purge nginx

# 사용하지 않는 의존성 제거
sudo apt autoremove

# 제거 + 불필요한 의존성 정리
sudo apt remove --autoremove nginx
```

### 패키지 업데이트

```bash
# 설치된 모든 패키지 업그레이드
sudo apt upgrade

# 배포판 업그레이드 (의존성 변경 포함)
sudo apt full-upgrade

# 배포판 버전 업그레이드
sudo do-release-upgrade
```

### 패키지 검색

```bash
# 패키지 검색
apt search nginx

# 패키지 정보
apt show nginx

# 설치된 패키지 목록
apt list --installed

# 업그레이드 가능한 패키지
apt list --upgradable
```

### 캐시 정리

```bash
# 다운로드한 패키지 파일 정리
sudo apt clean

# 오래된 패키지 파일만 정리
sudo apt autoclean
```

---

## 3. DNF/YUM (CentOS/RHEL)

### DNF (RHEL 8+, CentOS 8+, Fedora)

```bash
# 저장소 업데이트
sudo dnf check-update

# 패키지 설치
sudo dnf install nginx

# 여러 패키지
sudo dnf install nginx php mysql-server

# 확인 없이 설치
sudo dnf install -y vim

# 패키지 제거
sudo dnf remove nginx

# 패키지와 의존성 제거
sudo dnf autoremove nginx

# 모든 패키지 업데이트
sudo dnf upgrade

# 패키지 검색
dnf search nginx

# 패키지 정보
dnf info nginx

# 설치된 패키지 목록
dnf list installed

# 캐시 정리
sudo dnf clean all
```

### YUM (RHEL 7, CentOS 7)

```bash
# 패키지 설치
sudo yum install nginx

# 패키지 제거
sudo yum remove nginx

# 패키지 업데이트
sudo yum update

# 패키지 검색
yum search nginx

# 패키지 정보
yum info nginx
```

---

## 4. 저장소 관리

### Ubuntu/Debian 저장소

#### /etc/apt/sources.list

```bash
# 저장소 목록 확인
cat /etc/apt/sources.list

# 저장소 추가 디렉토리
ls /etc/apt/sources.list.d/
```

#### PPA (Personal Package Archive) 추가

```bash
# PPA 추가
sudo add-apt-repository ppa:ondrej/php

# PPA 제거
sudo add-apt-repository --remove ppa:ondrej/php

# 저장소 추가 후 업데이트
sudo apt update
```

#### 외부 저장소 추가

```bash
# Docker 저장소 추가 예시
# 1. GPG 키 추가
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 2. 저장소 추가
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 3. 업데이트 및 설치
sudo apt update
sudo apt install docker-ce
```

### CentOS/RHEL 저장소

#### 저장소 목록

```bash
# 저장소 목록
dnf repolist

# 상세 정보
dnf repolist -v

# 저장소 설정 위치
ls /etc/yum.repos.d/
```

#### 저장소 추가

```bash
# EPEL 저장소 (Extra Packages for Enterprise Linux)
sudo dnf install epel-release

# 저장소 파일 직접 추가
sudo vi /etc/yum.repos.d/custom.repo
```

저장소 파일 형식:
```ini
[custom-repo]
name=Custom Repository
baseurl=https://example.com/repo/
enabled=1
gpgcheck=1
gpgkey=https://example.com/RPM-GPG-KEY
```

---

## 5. 패키지 정보 확인

### Ubuntu/Debian (dpkg)

```bash
# 설치된 패키지 목록
dpkg -l

# 특정 패키지 검색
dpkg -l | grep nginx

# 패키지 상태
dpkg -s nginx

# 패키지가 설치한 파일 목록
dpkg -L nginx

# 파일이 속한 패키지 찾기
dpkg -S /usr/sbin/nginx
```

### CentOS/RHEL (rpm)

```bash
# 설치된 패키지 목록
rpm -qa

# 특정 패키지 검색
rpm -qa | grep nginx

# 패키지 정보
rpm -qi nginx

# 패키지가 설치한 파일 목록
rpm -ql nginx

# 파일이 속한 패키지 찾기
rpm -qf /usr/sbin/nginx
```

---

## 6. 소스 컴파일 설치

저장소에 없는 소프트웨어를 직접 컴파일합니다.

### 빌드 도구 설치

#### Ubuntu/Debian

```bash
sudo apt install build-essential
```

#### CentOS/RHEL

```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install gcc gcc-c++ make
```

### 컴파일 설치 과정

```bash
# 1. 소스 다운로드 및 압축 해제
wget https://example.com/software-1.0.tar.gz
tar -xzvf software-1.0.tar.gz
cd software-1.0

# 2. 의존성 확인 및 설정
./configure --prefix=/usr/local

# 3. 컴파일
make

# 4. 설치
sudo make install

# 5. 정리 (선택)
make clean
```

### checkinstall (패키지로 관리)

```bash
# 설치
sudo apt install checkinstall

# make install 대신 사용
sudo checkinstall

# 나중에 패키지 관리자로 제거 가능
```

---

## 7. 실무 패턴

### 시스템 업데이트

#### Ubuntu/Debian

```bash
# 업데이트 스크립트
#!/bin/bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt clean
```

#### CentOS/RHEL

```bash
# 업데이트 스크립트
#!/bin/bash
sudo dnf check-update
sudo dnf upgrade -y
sudo dnf autoremove -y
sudo dnf clean all
```

### 필수 패키지 설치

#### Ubuntu/Debian

```bash
# 서버 기본 패키지
sudo apt install -y \
    vim \
    curl \
    wget \
    git \
    htop \
    net-tools \
    unzip \
    tree
```

#### CentOS/RHEL

```bash
# 서버 기본 패키지
sudo dnf install -y \
    vim \
    curl \
    wget \
    git \
    htop \
    net-tools \
    unzip \
    tree
```

### 패키지 고정 (버전 업그레이드 방지)

#### Ubuntu/Debian

```bash
# 패키지 고정
sudo apt-mark hold nginx

# 고정 해제
sudo apt-mark unhold nginx

# 고정된 패키지 목록
apt-mark showhold
```

#### CentOS/RHEL

```bash
# versionlock 플러그인 설치
sudo dnf install dnf-plugin-versionlock

# 패키지 고정
sudo dnf versionlock add nginx

# 고정 해제
sudo dnf versionlock delete nginx

# 고정 목록
dnf versionlock list
```

---

## 8. 문제 해결

### 의존성 문제

#### Ubuntu/Debian

```bash
# 깨진 패키지 수정
sudo apt --fix-broken install

# 강제 설치 (주의)
sudo apt install -f

# dpkg 설정 복구
sudo dpkg --configure -a
```

#### CentOS/RHEL

```bash
# 의존성 정리
sudo dnf clean all
sudo dnf makecache

# 문제 있는 패키지 제거 후 재설치
sudo dnf remove package_name
sudo dnf install package_name
```

### 잠금 문제

#### Ubuntu/Debian

```bash
# apt 잠금 해제 (다른 apt가 실행 중일 때)
sudo rm /var/lib/dpkg/lock-frontend
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo dpkg --configure -a
```

---

## 9. CentOS 수명 종료 및 마이그레이션

### CentOS EOL 현황

CentOS Linux는 모든 버전이 수명 종료(End-of-Life, EOL)되었습니다.

- **CentOS 8**: 2021년 12월 31일 EOL
- **CentOS 7**: 2024년 6월 30일 EOL

**CentOS Stream**은 현재 유일하게 사용 가능한 CentOS 변종이지만, 다른 목적으로 사용됩니다.
- CentOS Stream은 **롤링 릴리스(rolling-release)** 개발 플랫폼입니다
- RHEL의 **업스트림(upstream)** 위치 (CentOS Linux처럼 다운스트림이 아님)
- **RHEL과 1:1 바이너리 호환 대체판이 아님**
- RHEL보다 먼저 업데이트 수신 (최첨단, 덜 안정적)

### 마이그레이션 옵션

CentOS를 사용하는 조직은 대안 배포판으로 마이그레이션해야 합니다.

| 배포판 | 유지보수 주체 | RHEL 호환성 | 비용 |
|-------------|-----------|-------------------|------|
| **Rocky Linux** | Rocky Enterprise Software Foundation | 1:1 바이너리 호환 | 무료 |
| **AlmaLinux** | AlmaLinux OS Foundation (CloudLinux) | 1:1 바이너리 호환 | 무료 |
| **Oracle Linux** | Oracle | 바이너리 호환 | 무료 |
| **RHEL** | Red Hat | 원본 | 유료 (무료 개발자 구독 제공) |

#### Rocky Linux

- Gregory Kurtzer(CentOS 공동 창립자) 설립
- 커뮤니티 주도, 비영리
- RHEL과 1:1 바이너리 호환
- 활발한 커뮤니티 및 기업 지원

```bash
# 현재 CentOS 버전 확인
cat /etc/redhat-release

# Rocky Linux로 마이그레이션 (CentOS 8)
sudo curl -O https://raw.githubusercontent.com/rocky-linux/rocky-tools/main/migrate2rocky/migrate2rocky.sh
sudo bash migrate2rocky.sh -r
```

#### AlmaLinux

- CloudLinux Inc. 지원
- RHEL과 1:1 바이너리 호환
- 강력한 상용 지원
- 잘 구축된 인프라

```bash
# AlmaLinux로 마이그레이션 (CentOS 8)
sudo curl -O https://raw.githubusercontent.com/AlmaLinux/almalinux-deploy/master/almalinux-deploy.sh
sudo bash almalinux-deploy.sh
```

#### Oracle Linux

- Oracle 유지보수
- RHEL과 바이너리 호환
- Unbreakable Enterprise Kernel (UEK) 사용 옵션
- 무료 사용 및 배포

```bash
# Oracle Linux로 마이그레이션 (CentOS 7/8)
sudo curl -O https://raw.githubusercontent.com/oracle/centos2ol/main/centos2ol.sh
sudo bash centos2ol.sh
```

### 주요 차이점: Rocky vs AlmaLinux

| 측면 | Rocky Linux | AlmaLinux |
|--------|------------|-----------|
| **거버넌스(Governance)** | 커뮤니티 주도 재단 | CloudLinux 지원 재단 |
| **자금 조달** | 기부, 스폰서 | CloudLinux Inc. + 스폰서 |
| **릴리스 주기** | 일반적으로 RHEL 밀접 추적 | 일반적으로 RHEL 밀접 추적 |
| **라이브 패칭(Live Patching)** | 제한적 | 사용 가능 (유료 KernelCare) |
| **상용 지원** | 서드파티 벤더 | CloudLinux + 파트너 |

두 배포판 모두 훌륭한 선택이며 매우 유사한 기능을 제공합니다. 선택은 주로 다음에 달려 있습니다.
- **Rocky Linux**: 커뮤니티 주도 거버넌스와 CentOS 레거시를 선호하는 경우
- **AlmaLinux**: 기업 지원과 선택적 상용 지원을 원하는 경우

### 마이그레이션 모범 사례

1. **먼저 테스트**: 프로덕션이 아닌 시스템에서 먼저 마이그레이션
2. **백업**: 마이그레이션 전 전체 시스템 백업
3. **호환성 확인**: 서드파티 소프트웨어 호환성 검토
4. **마이그레이션 전 업데이트**: CentOS를 완전히 업데이트
5. **마이그레이션 후 검증**: 마이그레이션 후 서비스 및 애플리케이션 확인

```bash
# 마이그레이션 전 체크리스트
# 1. 설치된 패키지 목록
rpm -qa > /root/packages-before.txt

# 2. 중요 설정 백업
sudo tar -czf /root/etc-backup.tar.gz /etc

# 3. 시스템 완전 업데이트
sudo yum update -y

# 4. 최신 커널로 재부팅
sudo reboot

# 마이그레이션 후 검증
# 1. OS 버전 확인
cat /etc/redhat-release

# 2. 패키지 수 확인
rpm -qa | wc -l

# 3. 깨진 의존성 확인
sudo dnf check

# 4. 서비스 확인
sudo systemctl list-units --state=failed
```

### 권장사항

대부분의 CentOS 마이그레이션 사용자를 위해:
- 프로덕션 워크로드에는 **Rocky Linux 또는 AlmaLinux 선택**
- 두 배포판 모두 안정적이고 엔터프라이즈급 RHEL 대체판 제공
- 최첨단 기능이 필요하지 않다면 프로덕션에서 CentOS Stream 피하기
- 공식 Red Hat 지원이 필요하면 RHEL 고려

---

## 10. 실습 예제

### 실습 1: 패키지 검색 및 설치

#### Ubuntu/Debian

```bash
# htop 검색
apt search htop

# 정보 확인
apt show htop

# 설치
sudo apt update
sudo apt install -y htop

# 확인
htop --version
```

#### CentOS/RHEL

```bash
# htop 검색
dnf search htop

# 정보 확인
dnf info htop

# 설치
sudo dnf install -y htop

# 확인
htop --version
```

### 실습 2: 패키지 정보 확인

#### Ubuntu/Debian

```bash
# 설치된 패키지 수
dpkg -l | grep "^ii" | wc -l

# 특정 패키지가 설치한 파일
dpkg -L bash | head -20

# 파일이 속한 패키지
dpkg -S /bin/bash
```

#### CentOS/RHEL

```bash
# 설치된 패키지 수
rpm -qa | wc -l

# 특정 패키지가 설치한 파일
rpm -ql bash | head -20

# 파일이 속한 패키지
rpm -qf /bin/bash
```

### 실습 3: 시스템 업데이트

#### Ubuntu/Debian

```bash
# 저장소 목록 업데이트
sudo apt update

# 업그레이드 가능 패키지 확인
apt list --upgradable

# 업그레이드
sudo apt upgrade -y

# 정리
sudo apt autoremove -y
sudo apt clean
```

#### CentOS/RHEL

```bash
# 업데이트 확인
sudo dnf check-update

# 업그레이드
sudo dnf upgrade -y

# 정리
sudo dnf autoremove -y
sudo dnf clean all
```

### 실습 4: 패키지 제거

```bash
# Ubuntu/Debian
sudo apt remove htop
sudo apt purge htop    # 설정 파일도 제거
sudo apt autoremove

# CentOS/RHEL
sudo dnf remove htop
sudo dnf autoremove
```

---

## 다음 단계

[쉘 스크립팅](./09_Shell_Scripting.md)에서 쉘 스크립팅을 배워봅시다!
