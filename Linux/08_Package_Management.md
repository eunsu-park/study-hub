# 패키지 관리

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

## 9. 실습 예제

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

[09_Shell_Scripting.md](./09_Shell_Scripting.md)에서 쉘 스크립팅을 배워봅시다!
