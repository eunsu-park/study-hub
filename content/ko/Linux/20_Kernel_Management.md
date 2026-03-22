# 커널 관리

**이전**: [백업과 복구](./19_Backup_Recovery.md) | **다음**: [가상화(KVM)](./21_Virtualization_KVM.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Linux 커널 버전 번호를 해석하고 설치된 커널 패키지를 관리한다
2. modprobe 및 영구 구성 파일을 사용하여 커널 모듈을 로드, 언로드, 구성한다
3. 문제가 있는 모듈을 블랙리스트(blacklist)에 등록하고 DKMS로 자동 모듈 로딩을 관리한다
4. 커널 소스를 다운로드하고, 빌드 옵션을 구성하고, 커스텀 커널을 컴파일한다
5. GRUB 부트로더의 기본 항목, 타임아웃, 커널 파라미터를 구성한다
6. initramfs 이미지를 관리하고 sysctl로 런타임 커널 파라미터를 튜닝한다
7. io_uring의 아키텍처와 기존 비동기 I/O 인터페이스 대비 장점을 설명할 수 있다

---

## 목차

1. [커널 개요](#1-커널-개요)
2. [커널 버전 관리](#2-커널-버전-관리)
3. [커널 모듈](#3-커널-모듈)
4. [DKMS](#4-dkms)
5. [커널 컴파일](#5-커널-컴파일)
6. [GRUB 부트로더](#6-grub-부트로더)
7. [커널 파라미터](#7-커널-파라미터)
8. [io_uring: 고성능 비동기 I/O](#8-io_uring-고성능-비동기-io)

---

커널은 모든 Linux 시스템의 심장부입니다 -- 하드웨어를 관리하고, 프로세스를 스케줄링하며, 보안 경계를 강제합니다. 대부분의 관리자는 배포판이 제공하는 커널을 사용하지만, 커널 버전을 관리하고, 모듈을 로드하고, 부팅 파라미터를 구성하는 방법을 이해하는 것은 하드웨어 지원 활성화, 보안 패치 적용, 서버 워크로드 최적화, 부팅 장애 복구 같은 작업에 필수적입니다.

## 1. 커널 개요

### 커널의 역할

```
┌─────────────────────────────────────────────────────────────┐
│                     사용자 공간                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 애플리케이션│  │  쉘     │  │  서비스  │  │  GUI    │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
├───────┴────────────┴────────────┴────────────┴──────────────┤
│                   시스템 호출 인터페이스                      │
├─────────────────────────────────────────────────────────────┤
│                      커널 공간                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  프로세스 관리  │  메모리 관리  │  파일시스템  │  네트워크 │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              디바이스 드라이버 (모듈)                 │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                        하드웨어                              │
│  CPU  │  메모리  │  디스크  │  네트워크 카드  │  기타 장치    │
└─────────────────────────────────────────────────────────────┘
```

### 커널 구성 요소

| 구성 요소 | 역할 |
|-----------|------|
| **프로세스 관리** | 프로세스 생성, 스케줄링, 종료 |
| **메모리 관리** | 가상 메모리, 페이징, 캐싱 |
| **파일시스템** | VFS, ext4, XFS, NFS 등 |
| **네트워킹** | TCP/IP 스택, 소켓, 라우팅 |
| **디바이스 드라이버** | 하드웨어 추상화, 모듈 |
| **시스템 호출** | 사용자 공간과 커널 인터페이스 |

---

## 2. 커널 버전 관리

### 현재 커널 정보

```bash
# 커널 버전 확인
uname -r
# 예: 6.8.0-51-generic

# 상세 정보
uname -a
# Linux hostname 6.8.0-51-generic #52-Ubuntu SMP x86_64 GNU/Linux

# 커널 설정 정보
cat /proc/version

# 커널 빌드 설정 (설정으로 컴파일된 경우)
cat /boot/config-$(uname -r) | head -20
```

### 커널 버전 체계

```
6.8.0-51-generic
│ │  │ │  └──────── 배포판 특정 이름
│ │  │ └────────── ABI 버전 (배포판 패치)
│ │  └──────────── 패치 레벨
│ └─────────────── 마이너 버전
└───────────────── 메이저 버전
```

### 설치된 커널 목록

```bash
# Ubuntu/Debian
dpkg --list | grep linux-image

# RHEL/CentOS
rpm -qa | grep kernel

# 또는 /boot 확인
ls -la /boot/vmlinuz-*
```

### 커널 업데이트

```bash
# Ubuntu/Debian
sudo apt update
sudo apt upgrade linux-image-generic

# 특정 버전 설치
sudo apt install linux-image-6.8.0-52-generic

# RHEL/CentOS
sudo yum update kernel

# 특정 버전 설치
sudo yum install kernel-5.14.0-362.el9
```

### 이전 커널 제거

```bash
# Ubuntu - 자동 제거
sudo apt autoremove

# Ubuntu - 특정 버전 제거
sudo apt remove linux-image-6.8.0-49-generic

# 현재 커널 제외하고 이전 커널 제거
sudo apt purge $(dpkg --list | grep -E 'linux-(image|headers|modules)' | \
    grep -v $(uname -r | sed 's/-generic//') | awk '{print $2}')

# RHEL/CentOS - 오래된 커널 제거 (2개 유지)
sudo dnf remove $(dnf repoquery --installonly --latest-limit=-2 -q)
```

---

## 3. 커널 모듈

### 모듈 정보 확인

```bash
# 로드된 모듈 목록
lsmod

# 특정 모듈 정보
modinfo ext4
modinfo nvidia

# 모듈 의존성 확인
modprobe --show-depends ext4

# 모듈 파일 위치
ls /lib/modules/$(uname -r)/kernel/
```

### 모듈 로드/언로드

```bash
# 모듈 로드
sudo modprobe nouveau
sudo modprobe snd-hda-intel

# 모듈 언로드
sudo modprobe -r nouveau
sudo rmmod nouveau

# 강제 언로드 (위험)
sudo rmmod -f nouveau

# 의존성 포함 로드
sudo modprobe -v nvidia
```

### 모듈 파라미터

```bash
# 모듈 파라미터 확인
modinfo -p e1000e

# 현재 적용된 파라미터
cat /sys/module/e1000e/parameters/IntMode

# 파라미터와 함께 로드
sudo modprobe e1000e IntMode=2

# /etc/modprobe.d/ 설정 파일
echo "options e1000e IntMode=2" | sudo tee /etc/modprobe.d/e1000e.conf
```

### 모듈 자동 로드 설정

```bash
# 부팅 시 자동 로드
echo "vhost_net" | sudo tee /etc/modules-load.d/vhost_net.conf

# 모듈 블랙리스트 (로드 방지)
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# initramfs 재생성 필요
sudo update-initramfs -u  # Ubuntu/Debian
sudo dracut -f            # RHEL/CentOS
```

### 모듈 별칭

```bash
# 별칭 확인
modprobe --show-depends -a pci:v00008086d00001502sv*sd*bc*sc*i*

# 하드웨어 ID로 모듈 찾기
lspci -nn  # PCI 장치와 ID 확인
lspci -k   # 사용 중인 모듈 확인

# USB 장치
lsusb
lsusb -t  # 트리 형태
```

---

## 4. DKMS

### DKMS 개요

DKMS (Dynamic Kernel Module Support)는 커널 업데이트 시 외부 모듈을 자동으로 재빌드합니다.

```bash
# DKMS 설치
sudo apt install dkms  # Ubuntu/Debian
sudo yum install dkms  # RHEL/CentOS
```

### DKMS 상태 확인

```bash
# 등록된 모듈 목록
dkms status

# 예시 출력:
# nvidia/535.154.05, 6.8.0-51-generic, x86_64: installed
# nvidia/535.154.05, 6.8.0-52-generic, x86_64: installed
# virtualbox/7.0.12_Ubuntu, 6.8.0-51-generic, x86_64: installed
```

### DKMS 모듈 관리

```bash
# 모듈 추가
sudo dkms add -m module-name -v version

# 모듈 빌드
sudo dkms build -m module-name -v version

# 모듈 설치
sudo dkms install -m module-name -v version

# 모듈 제거
sudo dkms remove -m module-name -v version --all

# 모든 커널에 대해 재빌드
sudo dkms autoinstall
```

### DKMS 모듈 생성 예시

```bash
# 모듈 디렉토리 구조
/usr/src/mymodule-1.0/
├── dkms.conf
├── Makefile
└── mymodule.c
```

```bash
# dkms.conf 예시
PACKAGE_NAME="mymodule"
PACKAGE_VERSION="1.0"
BUILT_MODULE_NAME[0]="mymodule"
DEST_MODULE_LOCATION[0]="/kernel/drivers/misc"
AUTOINSTALL="yes"
```

```bash
# Makefile 예시
obj-m := mymodule.o

KVERSION := $(shell uname -r)
KDIR := /lib/modules/$(KVERSION)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```

```bash
# DKMS에 등록
sudo dkms add -m mymodule -v 1.0
sudo dkms build -m mymodule -v 1.0
sudo dkms install -m mymodule -v 1.0
```

---

## 5. 커널 컴파일

### 소스 다운로드

```bash
# kernel.org에서 다운로드
cd /usr/src
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.7.tar.xz
tar -xvf linux-6.7.tar.xz
cd linux-6.7

# 또는 git으로
git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux
git checkout v6.7
```

### 빌드 의존성 설치

```bash
# Ubuntu/Debian
sudo apt install build-essential libncurses-dev bison flex \
    libssl-dev libelf-dev bc dwarves

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install ncurses-devel bison flex elfutils-libelf-devel \
    openssl-devel bc dwarves
```

### 커널 설정

```bash
# 현재 커널 설정 복사
cp /boot/config-$(uname -r) .config

# 메뉴 기반 설정
make menuconfig

# 또는 그래픽 설정 (X11 필요)
make xconfig

# 또는 텍스트 질의 방식
make config

# 기존 설정 기반으로 새 옵션만 질의
make oldconfig

# 기본값으로 새 옵션 설정
make olddefconfig
```

### 주요 설정 옵션

```
General setup --->
    Local version - append to kernel release: -custom
    [*] Automatically append version information

Processor type and features --->
    Processor family (Core 2/newer Xeon) --->
    [*] Symmetric multi-processing support

Device Drivers --->
    [M] Network device support --->
    [M] SCSI device support --->

File systems --->
    <*> Ext4 POSIX Access Control Lists
    <M> XFS filesystem support
    <M> Btrfs filesystem support
```

### 컴파일 및 설치

```bash
# 컴파일 (병렬 빌드)
make -j$(nproc)

# 모듈 컴파일
make modules

# 모듈 설치
sudo make modules_install

# 커널 설치
sudo make install

# initramfs 생성 (자동으로 되지 않은 경우)
sudo update-initramfs -c -k 6.7.0-custom

# GRUB 업데이트
sudo update-grub
```

### 빠른 테스트 빌드

```bash
# 최소 설정으로 시작
make tinyconfig

# 또는 현재 하드웨어에 맞는 최소 설정
make localmodconfig

# 빌드
make -j$(nproc)
```

---

## 6. GRUB 부트로더

### GRUB 설정 파일

```bash
# 메인 설정 (수정하지 않음)
/boot/grub/grub.cfg

# 사용자 설정 (이것을 수정)
/etc/default/grub

# 커스텀 스크립트
/etc/grub.d/
```

### /etc/default/grub 설정

```bash
# /etc/default/grub

# 기본 부팅 항목 (0부터 시작, 또는 "saved")
GRUB_DEFAULT=0

# 메뉴 표시 시간 (초)
GRUB_TIMEOUT=5

# 메뉴 숨김 (부팅 빠르게)
GRUB_TIMEOUT_STYLE=menu  # menu, countdown, hidden

# 커널 파라미터
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX=""

# 해상도
GRUB_GFXMODE=1920x1080

# 복구 모드 항목 비활성화
GRUB_DISABLE_RECOVERY="false"

# OS 프로버 (다른 OS 감지)
GRUB_DISABLE_OS_PROBER=false
```

### GRUB 업데이트

```bash
# 설정 변경 후 적용
sudo update-grub  # Ubuntu/Debian
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/CentOS

# BIOS 시스템
sudo grub2-mkconfig -o /boot/grub2/grub.cfg

# UEFI 시스템
sudo grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg
```

### GRUB 재설치

```bash
# BIOS 시스템
sudo grub-install /dev/sda
sudo update-grub

# UEFI 시스템
sudo grub-install --target=x86_64-efi --efi-directory=/boot/efi
sudo update-grub
```

### 부팅 항목 수동 추가

```bash
# /etc/grub.d/40_custom
#!/bin/sh
exec tail -n +3 $0

menuentry "Custom Kernel" {
    set root='hd0,msdos1'
    linux /boot/vmlinuz-6.7.0-custom root=/dev/sda2 ro quiet
    initrd /boot/initrd.img-6.7.0-custom
}

menuentry "Recovery Mode" {
    set root='hd0,msdos1'
    linux /boot/vmlinuz-6.7.0-custom root=/dev/sda2 ro single
    initrd /boot/initrd.img-6.7.0-custom
}
```

### 기본 부팅 커널 변경

```bash
# 사용 가능한 메뉴 항목 확인
grep -E "^menuentry|^submenu" /boot/grub/grub.cfg

# 기본값 설정 (인덱스)
sudo grub-set-default 2

# 기본값 설정 (이름)
sudo grub-set-default "Ubuntu, with Linux 6.8.0-52-generic"

# 일회성 부팅 선택
sudo grub-reboot "Ubuntu, with Linux 6.8.0-51-generic"

# /etc/default/grub에서 GRUB_DEFAULT 설정
GRUB_DEFAULT="1>2"  # 서브메뉴 1번의 3번째 항목
```

---

## 7. 커널 파라미터

### 부팅 시 파라미터

```bash
# /etc/default/grub에서 설정
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX="crashkernel=auto rd.lvm.lv=centos/root"

# 또는 GRUB 메뉴에서 'e'를 눌러 일시적 편집
linux /boot/vmlinuz-... root=... quiet splash nouveau.modeset=0
```

### 일반적인 커널 파라미터

| 파라미터 | 설명 |
|----------|------|
| `quiet` | 부팅 메시지 최소화 |
| `splash` | 그래픽 스플래시 화면 |
| `single` | 단일 사용자 모드 |
| `init=/bin/bash` | 직접 쉘로 부팅 |
| `nomodeset` | 비디오 모드 설정 비활성화 |
| `acpi=off` | ACPI 비활성화 |
| `noapic` | APIC 비활성화 |
| `mem=4G` | 사용 메모리 제한 |
| `maxcpus=2` | CPU 수 제한 |

### 런타임 파라미터 (sysctl)

```bash
# 현재 값 확인
sysctl -a
sysctl net.ipv4.ip_forward

# 임시 변경
sudo sysctl -w net.ipv4.ip_forward=1

# 영구 설정
echo "net.ipv4.ip_forward = 1" | sudo tee /etc/sysctl.d/99-custom.conf
sudo sysctl -p /etc/sysctl.d/99-custom.conf

# 또는 /etc/sysctl.conf 편집 후
sudo sysctl -p
```

### 주요 sysctl 파라미터

```bash
# /etc/sysctl.d/99-custom.conf

# 네트워크
net.ipv4.ip_forward = 1
net.ipv4.tcp_syncookies = 1
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# 메모리
vm.swappiness = 10
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10

# 파일 시스템
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# 보안
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
```

### initramfs 관리

```bash
# 현재 커널의 initramfs 재생성
sudo update-initramfs -u

# 특정 커널 버전
sudo update-initramfs -u -k 6.8.0-52-generic

# 새로 생성
sudo update-initramfs -c -k 6.8.0-52-generic

# RHEL/CentOS (dracut)
sudo dracut -f
sudo dracut -f /boot/initramfs-$(uname -r).img $(uname -r)

# 내용 확인
lsinitramfs /boot/initrd.img-$(uname -r) | head -50
```

---

## 8. io_uring: 고성능 비동기 I/O

io_uring(Linux 5.1에서 도입)은 사용자 공간과 커널 사이의 공유 메모리 링 버퍼를 사용하는 고성능 비동기 I/O 인터페이스입니다. `epoll`, `select`, POSIX AIO와 같은 기존 인터페이스에 비해 시스템 콜 오버헤드를 극적으로 줄여줍니다.

### io_uring이 필요한 이유

```
기존 I/O:                           io_uring:
  사용자 공간         커널             사용자 공간 ←──공유 메모리──→ 커널
  ┌────────┐         ┌────────┐        ┌────────┐                  ┌────────┐
  │ 제출    │──syscall→│ 처리   │        │ 제출   │──링에 기록──→   │ 폴링   │
  │ 대기    │←─syscall─│ 완료   │        │ 수확   │←─링에서 읽기──  │ 완료   │
  └────────┘         └────────┘        └────────┘                  └────────┘

  I/O 작업당 2번의 syscall              SQPOLL 모드에서 0번의 syscall
```

### 핵심 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    io_uring 아키텍처                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  사용자 공간                                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  애플리케이션                                             │ │
│  │  1. SQE를 제출 큐(SQ)에 기록                              │ │
│  │  2. SQ tail 포인터 전진                                   │ │
│  │  3. 완료 큐(CQ)에서 CQE 읽기                             │ │
│  └──────────┬──────────────────────────────────┬────────────┘ │
│             │                                  │              │
│  ┌──────────▼──────────┐      ┌────────────────▼────────────┐│
│  │  제출 큐             │      │  완료 큐                     ││
│  │  (SQ Ring Buffer)    │      │  (CQ Ring Buffer)           ││
│  │  ┌─────┬─────┬─────┐│      │  ┌─────┬─────┬─────┐       ││
│  │  │SQE 0│SQE 1│SQE 2││      │  │CQE 0│CQE 1│CQE 2│       ││
│  │  └─────┴─────┴─────┘│      │  └─────┴─────┴─────┘       ││
│  └──────────┬───────────┘      └────────────────▲────────────┘│
│             │  공유 메모리 (mmap)                │             │
├─────────────┼───────────────────────────────────┼─────────────┤
│  커널       │                                   │             │
│  ┌──────────▼───────────────────────────────────┴───────────┐│
│  │  io_uring Worker                                          ││
│  │  - SQ에서 SQE 읽기                                        ││
│  │  - I/O 작업 디스패치                                      ││
│  │  - 완료 시 CQ에 CQE 기록                                  ││
│  │  - SQPOLL 모드: 커널 스레드가 SQ를 폴링 (시스템 콜 제로)    ││
│  └───────────────────────────────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 주요 기능

| 기능 | 설명 | 커널 버전 |
|------|------|----------|
| **기본 SQ/CQ** | 제출 및 완료 링 버퍼 | 5.1 |
| **고정 버퍼(Fixed buffers)** | 사전 등록 I/O 버퍼로 반복적인 매핑 방지 | 5.1 |
| **SQPOLL** | 커널 스레드가 SQ를 폴링 — 제출 시 시스템 콜 불필요 | 5.1 |
| **연결된 SQE** | 종속 작업 연쇄 (예: 읽기 후 쓰기) | 5.3 |
| **멀티샷 accept** | 단일 SQE로 여러 accept() 완료 처리 | 5.19 |
| **io_uring 패스스루** | 직접 NVMe 명령 제출 | 6.0 |
| **io_uring cmd (NVMe)** | 커스텀 NVMe 관리/IO 명령 | 6.3 |
| **제로 카피 전송** | 커널 버퍼 복사 없는 네트워크 전송 | 6.6 |

### liburing을 사용한 최소 예제

```c
/* io_uring_read.c — liburing 래퍼 라이브러리를 통해 io_uring으로 파일을 읽는 예제.
 * 컴파일: gcc -o io_uring_read io_uring_read.c -luring
 * 필요 패키지: liburing-dev (apt install liburing-dev)
 */
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <liburing.h>

#define BUF_SIZE 4096

int main(int argc, char *argv[]) {
    struct io_uring ring;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    char buf[BUF_SIZE];
    int fd, ret;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        return 1;
    }

    /* 8개 SQ 엔트리로 io_uring 초기화 */
    ret = io_uring_queue_init(8, &ring, 0);
    if (ret < 0) {
        perror("io_uring_queue_init");
        return 1;
    }

    fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    /* 읽기 SQE 준비 — 수행할 I/O 작업을 기술 */
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, fd, buf, BUF_SIZE, 0);

    /* SQE를 커널에 제출 */
    io_uring_submit(&ring);

    /* 완료 대기 — CQE가 사용 가능해질 때까지 블록 */
    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
        perror("io_uring_wait_cqe");
        return 1;
    }

    if (cqe->res < 0) {
        fprintf(stderr, "Read failed: %s\n", strerror(-cqe->res));
    } else {
        printf("Read %d bytes:\n%.*s\n", cqe->res, cqe->res, buf);
    }

    /* CQE를 소비 완료로 표시 — CQ head 포인터를 전진 */
    io_uring_cqe_seen(&ring, cqe);

    close(fd);
    io_uring_queue_exit(&ring);
    return 0;
}
```

### io_uring 사용이 적합한 경우

- **고처리량 스토리지 I/O**: 데이터베이스 엔진, 로그 수집기 — 작업당 syscall 오버헤드가 지배적인 경우
- **네트워크 서버**: 멀티샷 accept와 제로 카피 전송은 대규모 연결 수 서버에 유리
- **NVMe 패스스루**: 블록 레이어를 우회하는 직접 디바이스 명령 제출로 최대 스토리지 성능 달성

> **보안 참고**: io_uring은 큰 공격 표면으로 인해 일부 컨테이너 런타임과 강화된 커널(예: Google의 프로덕션 커널)에서 비활성화되어 있습니다. 프로덕션에서 사용하기 전에 환경 정책을 확인하세요.

---

## 연습 문제

### 문제 1: 모듈 관리

1. `nouveau` 드라이버를 블랙리스트에 추가하고 확인하세요.
2. NVIDIA 독점 드라이버 사용 시나리오를 가정합니다.

### 문제 2: GRUB 설정

다음 요구사항을 만족하는 GRUB 설정을 작성하세요:
- 부팅 타임아웃 10초
- 기본 커널: 두 번째 항목
- 메모리 4GB 제한
- 조용한 부팅

### 문제 3: sysctl 설정

웹 서버에 최적화된 sysctl 설정을 작성하세요:
- 연결 백로그 증가
- 파일 디스크립터 제한 증가
- TCP 튜닝

---

## 정답

### 문제 1 정답

```bash
# nouveau 블랙리스트
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# initramfs 업데이트
sudo update-initramfs -u  # Ubuntu
sudo dracut -f            # RHEL

# 확인
cat /etc/modprobe.d/blacklist-nouveau.conf

# 재부팅 후 확인
lsmod | grep nouveau  # 출력 없어야 함
```

### 문제 2 정답

```bash
# /etc/default/grub
GRUB_DEFAULT=1
GRUB_TIMEOUT=10
GRUB_TIMEOUT_STYLE=menu
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash mem=4G"
GRUB_CMDLINE_LINUX=""
```

```bash
# 적용
sudo update-grub
```

### 문제 3 정답

```bash
# /etc/sysctl.d/99-webserver.conf

# 연결 백로그
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# 파일 디스크립터
fs.file-max = 2097152

# TCP 튜닝
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# 메모리 버퍼
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 12582912 16777216
net.ipv4.tcp_wmem = 4096 12582912 16777216
```

```bash
# 적용
sudo sysctl -p /etc/sysctl.d/99-webserver.conf
```

---

## 다음 단계

- [가상화 (KVM)](./21_Virtualization_KVM.md) - libvirt, virsh, VM 관리

---

## 참고 자료

- [The Linux Kernel Archives](https://www.kernel.org/)
- [Kernel Documentation](https://www.kernel.org/doc/html/latest/)
- [GRUB Manual](https://www.gnu.org/software/grub/manual/)
- `man modprobe`, `man dkms`, `man sysctl`
