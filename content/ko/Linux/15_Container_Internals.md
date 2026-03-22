# 15. 컨테이너 내부 구조

**이전**: [성능 튜닝](./14_Performance_Tuning.md) | **다음**: [스토리지 관리](./16_Storage_Management.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 컨테이너와 가상머신(Virtual Machine)의 근본적인 차이를 설명한다
2. Linux 네임스페이스(namespace)를 생성하고 관리하여 프로세스, 네트워크, 파일시스템을 격리한다
3. cgroups v2를 구성하여 프로세스 그룹의 CPU, 메모리, I/O 자원을 제한한다
4. OverlayFS가 컨테이너 이미지에 계층적 쓰기 시 복사(copy-on-write) 스토리지를 제공하는 방식을 설명한다
5. 네임스페이스, cgroups, chroot를 사용하여 최소 컨테이너를 수동으로 구축한다
6. 캐퍼빌리티(capabilities), seccomp, AppArmor 등 보안 메커니즘을 적용하여 컨테이너를 강화한다

## 목차
1. [컨테이너 기초](#1-컨테이너-기초)
2. [Linux Namespaces](#2-linux-namespaces)
3. [Control Groups (cgroups)](#3-control-groups-cgroups)
4. [Union Filesystem](#4-union-filesystem)
5. [컨테이너 런타임](#5-컨테이너-런타임)
6. [보안](#6-보안)
7. [eBPF 기초](#7-ebpf-기초)
8. [컨테이너 네트워킹 심화](#8-컨테이너-네트워킹-심화)
9. [연습 문제](#9-연습-문제)

---

컨테이너는 소프트웨어를 빌드하고, 배포하고, 실행하는 방식을 혁신했습니다. 그러나 Docker와 Podman 같은 도구 뒤에는 수년간 존재해온 Linux 커널 기능들 -- 네임스페이스(namespaces), cgroups, 유니온 파일시스템(union filesystems) -- 이 자리 잡고 있습니다. 이 내부 구조를 이해하는 것은 컨테이너 문제를 디버깅하고, 안전한 컨테이너 설정을 작성하고, 컨테이너가 실제로 무엇을 보장하는지(그리고 보장하지 않는지)를 파악하는 데 필수적입니다. 이 레슨은 추상화 계층을 걷어내어 커널 수준에서 컨테이너를 다룰 수 있도록 합니다.

## 1. 컨테이너 기초

### 1.1 컨테이너 vs 가상머신

```
┌─────────────────────────────────────────────────────────────┐
│               가상머신 vs 컨테이너                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  가상머신 (Virtual Machine)                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  App A  │ │  App B  │ │  App C  │                       │
│  ├─────────┤ ├─────────┤ ├─────────┤                       │
│  │ Guest OS│ │ Guest OS│ │ Guest OS│                       │
│  └─────────┴─────────┴─────────┘                           │
│  ┌─────────────────────────────────────┐                   │
│  │           Hypervisor                │                   │
│  └─────────────────────────────────────┘                   │
│  ┌─────────────────────────────────────┐                   │
│  │            Host OS                  │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  컨테이너 (Container)                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  App A  │ │  App B  │ │  App C  │                       │
│  ├─────────┤ ├─────────┤ ├─────────┤                       │
│  │ Bins/   │ │ Bins/   │ │ Bins/   │                       │
│  │ Libs    │ │ Libs    │ │ Libs    │                       │
│  └─────────┴─────────┴─────────┘                           │
│  ┌─────────────────────────────────────┐                   │
│  │        Container Runtime            │                   │
│  └─────────────────────────────────────┘                   │
│  ┌─────────────────────────────────────┐                   │
│  │          Host OS (커널 공유)         │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 컨테이너 핵심 기술

```
┌─────────────────────────────────────────────────────────────┐
│                  컨테이너 핵심 기술                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Namespaces - 격리 (Isolation)                          │
│     • PID namespace    - 프로세스 ID 격리                  │
│     • Network namespace - 네트워크 스택 격리               │
│     • Mount namespace  - 파일시스템 격리                   │
│     • UTS namespace    - 호스트명 격리                     │
│     • IPC namespace    - 프로세스 간 통신 격리             │
│     • User namespace   - 사용자/그룹 ID 격리               │
│     • Cgroup namespace - cgroup 루트 격리                  │
│                                                             │
│  2. Cgroups - 리소스 제한 (Resource Limiting)              │
│     • CPU, 메모리, I/O, 네트워크 대역폭 제한               │
│     • 프로세스 그룹 관리                                    │
│                                                             │
│  3. Union Filesystem - 레이어 이미지                       │
│     • OverlayFS, AUFS                                      │
│     • Copy-on-Write                                        │
│                                                             │
│  4. Capabilities - 권한 분리                               │
│     • root 권한 세분화                                      │
│                                                             │
│  5. Seccomp - 시스템 콜 필터링                             │
│     • 허용된 syscall만 실행                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

> **비유: 아파트 건물.** Linux 컨테이너는 건물 속 아파트와 같습니다. 네임스페이스(namespace)는 각 입주자에게 고유한 시야를 제공합니다 -- 자신만의 우편함(PID 네임스페이스), 자신만의 호수(네트워크 네임스페이스), 자신만의 계량기(마운트 네임스페이스) -- 동일한 건물 구조(커널)를 공유하면서도 말입니다. Cgroups는 단 한 명의 입주자도 물과 전기를 과도하게 사용하지 못하도록 관리하는 건물 관리인입니다.

## 2. Linux Namespaces

### 2.1 Namespace 종류

```bash
# 현재 프로세스의 namespace 확인
ls -la /proc/$$/ns/
# cgroup -> 'cgroup:[4026531835]'
# ipc -> 'ipc:[4026531839]'
# mnt -> 'mnt:[4026531840]'
# net -> 'net:[4026531992]'
# pid -> 'pid:[4026531836]'
# user -> 'user:[4026531837]'
# uts -> 'uts:[4026531838]'

# 시스템의 모든 namespace
lsns

# 특정 프로세스의 namespace
lsns -p <PID>
```

### 2.2 unshare로 namespace 생성

```bash
# UTS namespace (호스트명 격리)
unshare --uts /bin/bash
hostname container-test
hostname  # container-test
exit
hostname  # 원래 호스트명

# PID namespace (프로세스 격리)
unshare --pid --fork --mount-proc /bin/bash
ps aux  # 격리된 프로세스만 보임
echo $$  # PID 1
exit

# Mount namespace (파일시스템 격리)
unshare --mount /bin/bash
mount --bind /tmp /mnt
ls /mnt  # 호스트의 /mnt에는 영향 없음
exit

# Network namespace (네트워크 격리)
unshare --net /bin/bash
ip a  # lo만 존재
exit

# User namespace (사용자 격리)
unshare --user --map-root-user /bin/bash
id  # uid=0(root) gid=0(root)
# 실제로는 일반 사용자
exit

# 모두 조합 (컨테이너와 유사)
unshare --mount --uts --ipc --net --pid --fork --user --map-root-user /bin/bash
```

### 2.3 nsenter로 namespace 진입

```bash
# 다른 프로세스의 namespace로 진입
nsenter -t <PID> --all /bin/bash

# 특정 namespace만
nsenter -t <PID> --net /bin/bash
nsenter -t <PID> --pid --mount /bin/bash

# Docker 컨테이너의 namespace 진입
docker inspect --format '{{.State.Pid}}' <container_id>
nsenter -t <PID> --all /bin/bash
```

### 2.4 Namespace C 예제

```c
// simple_container.c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define STACK_SIZE (1024 * 1024)

static char child_stack[STACK_SIZE];

int child_fn(void *arg) {
    // 호스트명 변경
    sethostname("container", 9);

    // 새 root 파일시스템으로 chroot (준비된 경우)
    // chroot("/path/to/rootfs");
    // chdir("/");

    // 쉘 실행
    char *argv[] = {"/bin/bash", NULL};
    execv(argv[0], argv);
    return 0;
}

int main() {
    // 새 namespace와 함께 자식 프로세스 생성
    int flags = CLONE_NEWUTS |     // UTS namespace
                CLONE_NEWPID |     // PID namespace
                CLONE_NEWNS |      // Mount namespace
                CLONE_NEWNET |     // Network namespace
                SIGCHLD;

    pid_t pid = clone(child_fn, child_stack + STACK_SIZE, flags, NULL);

    if (pid == -1) {
        perror("clone");
        exit(1);
    }

    waitpid(pid, NULL, 0);
    return 0;
}
```

```bash
# 컴파일 및 실행
gcc -o simple_container simple_container.c
sudo ./simple_container
```

---

## 3. Control Groups (cgroups)

### 3.1 cgroups v2 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    cgroups v2 계층 구조                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /sys/fs/cgroup/ (cgroup2 루트)                            │
│  ├── cgroup.controllers      # 사용 가능한 컨트롤러        │
│  ├── cgroup.subtree_control  # 자식에게 위임할 컨트롤러    │
│  ├── cgroup.procs            # 이 cgroup의 프로세스        │
│  │                                                          │
│  ├── system.slice/           # systemd 시스템 서비스       │
│  │   ├── cgroup.procs                                      │
│  │   ├── cpu.max                                           │
│  │   └── memory.max                                        │
│  │                                                          │
│  ├── user.slice/             # 사용자 세션                 │
│  │   └── user-1000.slice/                                  │
│  │                                                          │
│  └── mygroup/                # 사용자 정의 그룹            │
│      ├── cgroup.procs                                      │
│      ├── cpu.max             # CPU 제한                    │
│      ├── cpu.stat            # CPU 통계                    │
│      ├── memory.max          # 메모리 제한                 │
│      ├── memory.current      # 현재 메모리 사용량          │
│      └── io.max              # I/O 제한                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 cgroups 기본 명령

```bash
# cgroups v2 확인
mount | grep cgroup2
cat /sys/fs/cgroup/cgroup.controllers
# cpuset cpu io memory hugetlb pids rdma misc

# 새 cgroup 생성
mkdir /sys/fs/cgroup/mygroup

# 컨트롤러 활성화
echo "+cpu +memory +io" > /sys/fs/cgroup/cgroup.subtree_control

# 프로세스 추가
echo $$ > /sys/fs/cgroup/mygroup/cgroup.procs

# 프로세스 확인
cat /sys/fs/cgroup/mygroup/cgroup.procs

# cgroup 삭제 (프로세스 없어야 함)
rmdir /sys/fs/cgroup/mygroup
```

### 3.3 CPU 제한

```bash
# CPU 제한 (quota / period)
# cpu.max: "quota period" (마이크로초)
# 50% CPU 제한
echo "50000 100000" > /sys/fs/cgroup/mygroup/cpu.max

# 특정 CPU만 사용
# cpuset.cpus: 사용할 CPU
echo "0-1" > /sys/fs/cgroup/mygroup/cpuset.cpus

# CPU 가중치 (1-10000, 기본 100)
echo "50" > /sys/fs/cgroup/mygroup/cpu.weight

# 통계 확인
cat /sys/fs/cgroup/mygroup/cpu.stat
# usage_usec 12345
# user_usec 10000
# system_usec 2345
```

### 3.4 메모리 제한

```bash
# 메모리 제한
echo "512M" > /sys/fs/cgroup/mygroup/memory.max
# 또는 바이트로
echo "536870912" > /sys/fs/cgroup/mygroup/memory.max

# 메모리 + swap 제한
echo "1G" > /sys/fs/cgroup/mygroup/memory.swap.max

# OOM 설정
# memory.oom.group: 1이면 그룹 전체 kill
echo 1 > /sys/fs/cgroup/mygroup/memory.oom.group

# 현재 사용량
cat /sys/fs/cgroup/mygroup/memory.current
cat /sys/fs/cgroup/mygroup/memory.stat
```

### 3.5 I/O 제한

```bash
# 장치 확인
lsblk
# 예: sda -> 8:0

# I/O 대역폭 제한 (바이트/초)
echo "8:0 rbps=10485760 wbps=10485760" > /sys/fs/cgroup/mygroup/io.max
# 10MB/s 읽기/쓰기 제한

# IOPS 제한
echo "8:0 riops=1000 wiops=1000" > /sys/fs/cgroup/mygroup/io.max

# I/O 가중치
echo "8:0 100" > /sys/fs/cgroup/mygroup/io.weight

# 통계
cat /sys/fs/cgroup/mygroup/io.stat
```

### 3.6 systemd와 cgroups

```bash
# systemd-cgls - cgroup 트리 보기
systemd-cgls

# 특정 슬라이스
systemd-cgls /system.slice

# systemd-cgtop - 실시간 모니터링
systemd-cgtop

# 서비스 리소스 제한 (유닛 파일)
# [Service]
# CPUQuota=50%
# MemoryMax=512M
# IOWriteBandwidthMax=/dev/sda 10M

# 런타임에 제한 변경
systemctl set-property nginx.service CPUQuota=50%
systemctl set-property nginx.service MemoryMax=512M
```

---

## 4. Union Filesystem

### 4.1 OverlayFS 개념

```
┌─────────────────────────────────────────────────────────────┐
│                    OverlayFS 구조                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Merged (통합 뷰) - 컨테이너가 보는 파일시스템              │
│  /merged                                                    │
│     │                                                       │
│     ├── [Upper에서]     │                                   │
│     ├── [Lower에서]     │                                   │
│     └── [Lower에서]     │                                   │
│                                                             │
│  ┌─────────────────────┐                                   │
│  │    Upper Layer      │  ← 쓰기 가능 (컨테이너 변경)      │
│  │    /upper           │                                    │
│  └─────────────────────┘                                   │
│           ↑                                                 │
│  ┌─────────────────────┐                                   │
│  │   Lower Layer(s)    │  ← 읽기 전용 (이미지 레이어)      │
│  │   /lower1           │                                    │
│  │   /lower2           │                                    │
│  │   /lower3           │                                    │
│  └─────────────────────┘                                   │
│                                                             │
│  Work Directory                                             │
│  /work - OverlayFS 내부 작업용                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 OverlayFS 사용

```bash
# OverlayFS 마운트
mkdir -p /lower /upper /work /merged

# lower에 기본 파일 생성
echo "from lower" > /lower/file1.txt
echo "will be overwritten" > /lower/file2.txt

# upper에 파일 생성
echo "from upper" > /upper/file2.txt
echo "only in upper" > /upper/file3.txt

# OverlayFS 마운트
mount -t overlay overlay \
  -o lowerdir=/lower,upperdir=/upper,workdir=/work \
  /merged

# 결과 확인
ls /merged/
# file1.txt  file2.txt  file3.txt

cat /merged/file1.txt  # from lower
cat /merged/file2.txt  # from upper (덮어씀)
cat /merged/file3.txt  # only in upper

# 새 파일 쓰기
echo "new file" > /merged/file4.txt
ls /upper/  # file4.txt가 upper에 생성됨

# 파일 삭제 (whiteout)
rm /merged/file1.txt
ls -la /upper/
# c--------- ... file1.txt  (whiteout 파일)

# 언마운트
umount /merged
```

### 4.3 Docker 이미지 레이어

```bash
# Docker 이미지 레이어 확인
docker image inspect ubuntu:22.04 --format '{{.RootFS.Layers}}'

# 레이어 저장 위치
ls /var/lib/docker/overlay2/

# 특정 컨테이너의 마운트 정보
docker inspect <container_id> --format '{{.GraphDriver.Data}}'
# LowerDir, UpperDir, MergedDir, WorkDir 확인
```

---

## 5. 컨테이너 런타임

### 5.1 런타임 계층

```
┌─────────────────────────────────────────────────────────────┐
│                    컨테이너 런타임 계층                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  High-Level Runtime (컨테이너 엔진)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Docker Engine / Podman / containerd                │   │
│  │  • 이미지 관리 (pull, push, build)                  │   │
│  │  • 네트워킹                                          │   │
│  │  • 볼륨 관리                                         │   │
│  │  • API 제공                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  Low-Level Runtime (OCI Runtime)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  runc / crun / kata-containers                      │   │
│  │  • 실제 컨테이너 생성                                │   │
│  │  • namespace, cgroups 설정                          │   │
│  │  • OCI 스펙 준수                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  Linux Kernel                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  namespaces, cgroups, seccomp, capabilities        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 수동으로 컨테이너 만들기

```bash
#!/bin/bash
# manual_container.sh

# 1. rootfs 준비
mkdir -p /tmp/mycontainer/{rootfs,upper,work,merged}

# 기본 rootfs 다운로드 (Alpine)
curl -o /tmp/alpine.tar.gz https://dl-cdn.alpinelinux.org/alpine/v3.18/releases/x86_64/alpine-minirootfs-3.18.0-x86_64.tar.gz
tar -xzf /tmp/alpine.tar.gz -C /tmp/mycontainer/rootfs

# 2. OverlayFS 마운트
mount -t overlay overlay \
  -o lowerdir=/tmp/mycontainer/rootfs,upperdir=/tmp/mycontainer/upper,workdir=/tmp/mycontainer/work \
  /tmp/mycontainer/merged

# 3. 필수 마운트
mount -t proc proc /tmp/mycontainer/merged/proc
mount -t sysfs sysfs /tmp/mycontainer/merged/sys
mount -o bind /dev /tmp/mycontainer/merged/dev

# 4. 새 namespace로 chroot
unshare --mount --uts --ipc --net --pid --fork \
  chroot /tmp/mycontainer/merged /bin/sh -c '
    hostname mycontainer
    mount -t proc proc /proc
    exec /bin/sh
  '

# 정리
umount /tmp/mycontainer/merged/{proc,sys,dev}
umount /tmp/mycontainer/merged
```

### 5.3 runc 사용

```bash
# runc 설치
apt install runc

# OCI 번들 구조
mkdir -p bundle/rootfs
cd bundle

# rootfs 준비 (Docker에서 추출)
docker export $(docker create alpine) | tar -C rootfs -xf -

# config.json 생성
runc spec

# config.json 수정 (terminal: true로 변경)

# 컨테이너 실행
runc run mycontainer

# 다른 터미널에서
runc list
runc state mycontainer
runc kill mycontainer
runc delete mycontainer
```

### 5.4 Rootless 컨테이너

```bash
# Podman rootless
podman run --rm -it alpine sh

# 사용자 namespace 매핑 확인
cat /proc/self/uid_map
cat /proc/self/gid_map

# subuid/subgid 설정
# /etc/subuid
# username:100000:65536
# /etc/subgid
# username:100000:65536

# Rootless Docker
dockerd-rootless-setuptool.sh install
export PATH=/home/$USER/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker run --rm hello-world
```

---

## 6. 보안

### 6.1 Capabilities

```bash
# 프로세스 capabilities 확인
cat /proc/$$/status | grep Cap
getpcaps $$

# Capabilities 목록
capsh --print

# 특정 capability만 부여
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE nginx

# 주요 capabilities:
# CAP_NET_ADMIN - 네트워크 설정
# CAP_NET_BIND_SERVICE - 1024 미만 포트 바인딩
# CAP_SYS_ADMIN - 시스템 관리 (위험)
# CAP_SYS_PTRACE - 프로세스 추적
# CAP_MKNOD - 특수 파일 생성
```

### 6.2 Seccomp

```bash
# 기본 seccomp 프로파일 확인
docker info --format '{{.SecurityOptions}}'

# 커스텀 seccomp 프로파일
cat > seccomp.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "exit", "exit_group"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF

# 프로파일 적용
docker run --security-opt seccomp=seccomp.json alpine sh

# seccomp 없이 실행 (비권장)
docker run --security-opt seccomp=unconfined alpine sh
```

### 6.3 AppArmor/SELinux

```bash
# AppArmor 상태
aa-status

# Docker AppArmor 프로파일
cat /etc/apparmor.d/docker

# 커스텀 프로파일 적용
docker run --security-opt apparmor=my-profile alpine

# SELinux (RHEL/CentOS)
getenforce
# docker run --security-opt label=type:my_container_t alpine
```

### 6.4 읽기 전용 루트

```bash
# 읽기 전용 루트 파일시스템
docker run --read-only alpine sh

# 임시 디렉토리 허용
docker run --read-only --tmpfs /tmp alpine sh

# 볼륨으로 쓰기 허용
docker run --read-only -v /data alpine sh
```

## 7. eBPF 기초

eBPF(extended Berkeley Packet Filter)는 커널 소스 코드를 변경하거나 커널 모듈을 로드하지 않고도 Linux 커널 내부에서 샌드박스 프로그램을 실행할 수 있는 혁신적인 기술입니다. 원래 패킷 필터링용으로 설계되었지만, eBPF는 현대적인 관측성(observability), 네트워킹, 보안 도구를 구동하는 범용 커널 내 가상 머신으로 발전했습니다.

### 7.1 eBPF란?

```
┌─────────────────────────────────────────────────────────────┐
│                       eBPF 아키텍처                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  사용자 공간 (User Space)                                   │
│  ┌───────────────────────────────────────────────────┐     │
│  │  bcc / bpftrace / libbpf                          │     │
│  │  (eBPF 프로그램 로더 & 프론트엔드)                │     │
│  └───────────────────┬───────────────────────────────┘     │
│                      │ 프로그램 로드                         │
│                      ▼                                      │
│  ─────────────────────────────────────────────────────      │
│  커널 공간 (Kernel Space)                                   │
│  ┌───────────────┐   ┌───────────────┐                     │
│  │   Verifier    │──>│  JIT Compiler │                     │
│  │ (안전성 검사) │   │ (네이티브 코드)│                     │
│  └───────────────┘   └───────┬───────┘                     │
│                              │ 부착(attach)                 │
│                              ▼                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ kprobes  │ │tracepoint│ │   XDP    │ │  cgroup  │      │
│  │          │ │          │ │(네트워크)│ │(리소스) │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│  eBPF Maps (커널-사용자 공간 공유 데이터)                   │
│  ┌─────────────────────────────────────────────────┐       │
│  │  Hash maps, Arrays, Ring buffers, Per-CPU maps  │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> **클래식 BPF vs eBPF.** 클래식 BPF(cBPF)는 단순한 32비트 명령어 집합과 2개의 레지스터로 패킷 필터링에 한정되었습니다. eBPF는 이를 64비트 명령어 집합, 11개 레지스터로 확장하고, 상태 저장을 위한 맵(map)을 지원하며, 함수 호출이 가능하고, 네트워크 패킷뿐 아니라 거의 모든 커널 이벤트에 부착할 수 있습니다. cBPF가 공학용 계산기라면 eBPF는 완전한 프로그래밍 환경입니다.

### 7.2 eBPF 프로그램 타입

| 프로그램 타입 | 부착 지점(Attach Point) | 용도 |
|-------------|-------------|----------|
| `kprobe` / `kretprobe` | 커널 함수 진입/반환 | 커널 내부 추적 |
| `tracepoint` | 정적 커널 트레이스포인트(tracepoint) | 안정적 성능 모니터링 |
| `XDP` (eXpress Data Path) | 네트워크 드라이버 (sk_buff 이전) | 초고속 패킷 처리 |
| `tc` (traffic control) | 네트워크 트래픽 제어 계층 | 패킷 변조, 폴리싱 |
| `cgroup` | cgroup 이벤트 | 컨테이너별 리소스 제어 |
| `socket_filter` | 소켓 계층 | 소켓별 패킷 필터링 |
| `perf_event` | CPU 성능 카운터 | 하드웨어 수준 프로파일링 |
| `LSM` (Linux Security Module) | 보안 훅(hook) | 세분화된 보안 정책 |

### 7.3 eBPF 검증기(Verifier)와 JIT 컴파일

eBPF 검증기(verifier)는 eBPF 프로그램이 커널을 크래시시키거나 비인가 메모리에 접근하지 못하도록 보장하는 안전 메커니즘입니다:

```bash
# 검증기가 확인하는 항목:
# 1. 무한 루프 없음 (프로그램이 반드시 종료해야 함)
# 2. 모든 메모리 접근에 경계 검사
# 3. 널 포인터 역참조 없음
# 4. 스택 크기 512바이트 제한
# 5. 프로그램 크기 제한 (커널 5.2+ 기준 100만 명령어)

# JIT 컴파일은 검증된 eBPF 바이트코드를 네이티브 머신 코드로 변환
# JIT 상태 확인
cat /proc/sys/net/core/bpf_jit_enable
# 0 = 비활성, 1 = 활성, 2 = 디버그 활성

# JIT 컴파일 활성화
echo 1 | sudo tee /proc/sys/net/core/bpf_jit_enable
```

### 7.4 bpftool과 BCC 도구

```bash
# bpftool: eBPF 프로그램/맵 검사 및 관리
# 로드된 eBPF 프로그램 목록
bpftool prog list

# 특정 프로그램 상세 정보
bpftool prog show id 42

# eBPF 맵 목록
bpftool map list

# 맵 내용 덤프
bpftool map dump id 10

# --- BCC (BPF Compiler Collection) 도구 ---
# BCC 도구 설치
apt install bpfcc-tools  # Debian/Ubuntu
# 또는
yum install bcc-tools    # RHEL/CentOS

# tcptop: 프로세스별 TCP 트래픽 모니터링 (TCP용 top)
tcptop

# execsnoop: 새 프로세스 실행 추적
execsnoop
# 출력: PCOMM PID PPID RET ARGS
# bash   1234 1000 0   /usr/bin/ls -la

# biolatency: 블록 I/O 지연 히스토그램
biolatency

# opensnoop: 시스템 전체 파일 열기 추적
opensnoop

# tcpconnect: 능동적 TCP 연결 추적
tcpconnect

# funccount: 커널 함수 호출 횟수 카운트
funccount 'tcp_send*'
```

### 7.5 eBPF 활용 사례

```
┌─────────────────────────────────────────────────────────────┐
│                    eBPF 활용 사례                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 관측성 (Observability)                                  │
│     • 시스템 콜 추적 (strace 대체)                          │
│     • 애플리케이션 지연 프로파일링                           │
│     • 지속적 프로덕션 프로파일링 (Parca, Pyroscope)         │
│     • 코드 변경 없이 커스텀 메트릭 수집                     │
│                                                             │
│  2. 네트워킹 (Networking)                                   │
│     • XDP 기반 DDoS 완화 (Cloudflare, Facebook)            │
│     • 로드 밸런싱 (Cilium, Katran)                          │
│     • K8s 네트워크 정책 적용                                │
│     • DNS 요청 모니터링                                     │
│                                                             │
│  3. 보안 (Security)                                         │
│     • 런타임 위협 탐지 (Falco, Tetragon)                   │
│     • 파일 무결성 모니터링                                   │
│     • 네트워크 정책 적용                                    │
│     • syscall 감사 (auditd 대체)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 컨테이너 네트워킹 심화

컨테이너의 통신 원리를 이해하려면 Linux 네트워킹 기본 요소에 대한 지식이 필요합니다. Docker 브릿지, Kubernetes 파드 네트워킹, 서비스 메시 등 모든 컨테이너 네트워크는 이러한 커널 수준의 빌딩 블록 위에 구축됩니다.

### 8.1 네트워크 네임스페이스(Network Namespace)

```bash
# 두 개의 네트워크 네임스페이스 생성 (컨테이너 2개 시뮬레이션)
ip netns add container1
ip netns add container2

# 확인
ip netns list
# container1
# container2

# 각 네임스페이스는 자체 격리된 네트워크 스택을 가짐
ip netns exec container1 ip a
# lo (루프백) 인터페이스만 존재하며 DOWN 상태

# 각 네임스페이스에서 루프백 활성화
ip netns exec container1 ip link set lo up
ip netns exec container2 ip link set lo up
```

### 8.2 Veth 페어와 브릿지 네트워킹

```
┌─────────────────────────────────────────────────────────────┐
│              컨테이너 브릿지 네트워킹                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Container 1                Container 2                     │
│  (netns: container1)        (netns: container2)             │
│  ┌───────────────┐          ┌───────────────┐              │
│  │ eth0          │          │ eth0          │              │
│  │ 172.17.0.2/24 │          │ 172.17.0.3/24 │              │
│  └───────┬───────┘          └───────┬───────┘              │
│          │ veth pair                │ veth pair              │
│          │                          │                        │
│  ┌───────┴──────────────────────────┴───────┐              │
│  │              docker0 bridge               │              │
│  │              172.17.0.1/24                │              │
│  └──────────────────┬───────────────────────┘              │
│                     │                                       │
│                     │ NAT (iptables MASQUERADE)             │
│                     ▼                                       │
│  ┌─────────────────────────────────────────┐               │
│  │          Host eth0 / ens33              │               │
│  │          192.168.1.100                  │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```bash
# 단계별: 컨테이너 네트워킹 수동 구축

# 1. 브릿지 생성 (docker0에 해당)
ip link add br0 type bridge
ip addr add 172.18.0.1/24 dev br0
ip link set br0 up

# 2. container1용 veth 페어 생성
ip link add veth1 type veth peer name veth1-br

# 3. 한쪽 끝을 컨테이너 네임스페이스로 이동
ip link set veth1 netns container1

# 4. 다른 끝을 브릿지에 연결
ip link set veth1-br master br0
ip link set veth1-br up

# 5. 컨테이너 내부에서 IP 설정
ip netns exec container1 ip addr add 172.18.0.2/24 dev veth1
ip netns exec container1 ip link set veth1 up
ip netns exec container1 ip route add default via 172.18.0.1

# 6. container2도 동일하게 반복
ip link add veth2 type veth peer name veth2-br
ip link set veth2 netns container2
ip link set veth2-br master br0
ip link set veth2-br up
ip netns exec container2 ip addr add 172.18.0.3/24 dev veth2
ip netns exec container2 ip link set veth2 up
ip netns exec container2 ip route add default via 172.18.0.1

# 7. 컨테이너 간 통신 테스트
ip netns exec container1 ping -c 3 172.18.0.3
# PING 172.18.0.3: 3 packets transmitted, 3 received, 0% packet loss

# 8. 외부 접속을 위한 NAT 활성화
sysctl -w net.ipv4.ip_forward=1
iptables -t nat -A POSTROUTING -s 172.18.0.0/24 -j MASQUERADE
```

### 8.3 Docker 네트워크 모드

| 모드 | 명령어 | 격리 수준 | 성능 | 용도 |
|------|---------|-----------|------|------|
| **bridge** (기본) | `--network bridge` | 컨테이너 수준 | 중간 (NAT 오버헤드) | 범용 |
| **host** | `--network host` | 없음 (호스트 스택 공유) | 네이티브 | 성능 중시 앱 |
| **none** | `--network none` | 완전 격리 | 해당 없음 | 보안 민감 워크로드 |
| **overlay** | `--network my-overlay` | 크로스 호스트 | 낮음 (VXLAN 캡슐화) | Swarm / 다중 호스트 |
| **macvlan** | `--network my-macvlan` | VLAN 수준 | 네이티브에 가까움 | 직접 LAN 통합 |

```bash
# Docker 브릿지 네트워크 검사
docker network inspect bridge
# 출력: subnet, gateway, 연결된 컨테이너, IPAM 설정

# 특정 서브넷으로 커스텀 브릿지 생성
docker network create --driver bridge --subnet 10.10.0.0/16 mynet

# 커스텀 네트워크에서 컨테이너 실행 (컨테이너 이름으로 자동 DNS)
docker run -d --name web --network mynet nginx
docker run -it --network mynet alpine ping web
# 컨테이너 이름 "web"이 자동으로 DNS 해석됨
```

### 8.4 Kubernetes CNI (Container Network Interface)

```
┌─────────────────────────────────────────────────────────────┐
│                Kubernetes CNI 아키텍처                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  kubelet                                                    │
│    │                                                        │
│    │ "파드 네트워크 생성"                                    │
│    ▼                                                        │
│  CRI (containerd/CRI-O)                                    │
│    │                                                        │
│    │ CNI ADD / DEL                                          │
│    ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    CNI 플러그인                        │  │
│  │                                                      │  │
│  │  Calico     │  Cilium      │  Flannel               │  │
│  │  • BGP      │  • eBPF      │  • VXLAN overlay       │  │
│  │  • IP-in-IP │  • kube-     │  • 간단한 설정         │  │
│  │  • 정책     │    proxy     │  • 제한적 정책         │  │
│  │    엔진     │    불필요    │                         │  │
│  │  • IPAM     │  • L7 정책   │                         │  │
│  │             │  • Hubble    │                         │  │
│  │             │   (관측성)   │                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  파드 간 통신: 모든 파드가 라우팅 가능한 IP 주소를 부여받음 │
│  파드 간 NAT 없음 (플랫 네트워크 모델)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 CNI 플러그인 비교: Calico vs Cilium vs Flannel

| 특징 | Calico | Cilium | Flannel |
|------|--------|--------|---------|
| **데이터 플레인** | iptables 또는 eBPF | eBPF | VXLAN / host-gw |
| **네트워크 정책** | 완전 (L3/L4) | 완전 (L3/L4/L7) | 없음 (애드온 필요) |
| **암호화** | WireGuard | WireGuard / IPsec | 없음 |
| **관측성** | 제한적 | Hubble (심층 가시성) | 없음 |
| **성능** | 높음 (BGP 모드) | 최고 (eBPF 바이패스) | 중간 |
| **복잡도** | 중간 | 중상 | 낮음 |
| **적합 환경** | 대규모 클러스터, BGP | 보안 + 관측성 | 간단한/소규모 클러스터 |

```bash
# Calico: 매니페스트로 설치
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/calico.yaml

# Cilium: Helm으로 설치
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium --namespace kube-system

# Flannel: 매니페스트로 설치
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

# CNI 동작 확인
kubectl get pods -n kube-system | grep -E 'calico|cilium|flannel'
```

> **Cilium이 주목받는 이유.** Cilium은 kube-proxy를 eBPF 프로그램으로 완전히 대체하여, 수천 개의 서비스에서 확장이 어려운 iptables 규칙을 제거합니다. L7(HTTP, gRPC, Kafka) 네트워크 정책, Hubble을 통한 내장 관측성, 투명한 암호화를 제공하며 -- 이 모든 것을 사이드카 프록시 없이 달성합니다. 성능과 보안 가시성이 모두 필요한 클러스터에서 Cilium은 선도적인 선택지가 되었습니다.

---

## 9. 연습 문제

### 연습 1: namespace 실습
```bash
# 요구사항:
# 1. 모든 namespace 유형을 사용하여 격리된 환경 생성
# 2. 호스트와 격리 확인 (hostname, PID, network)
# 3. nsenter로 namespace 진입

# 명령어 작성:
```

### 연습 2: cgroups 리소스 제한
```bash
# 요구사항:
# 1. CPU 25% 제한
# 2. 메모리 256MB 제한
# 3. I/O 1MB/s 제한
# 4. stress 도구로 테스트

# 설정 및 명령어 작성:
```

### 연습 3: 수동 컨테이너 생성
```bash
# 요구사항:
# 1. rootfs 준비 (Alpine)
# 2. OverlayFS 설정
# 3. namespace 격리
# 4. cgroups 제한
# 5. 쉘 실행

# 스크립트 작성:
```

### 연습 4: 보안 강화 컨테이너
```bash
# 요구사항:
# 1. 최소 capabilities만 부여
# 2. seccomp 프로파일 적용
# 3. 읽기 전용 루트
# 4. non-root 사용자

# docker run 명령어 작성:
```

---

## 다음 단계

- [16_저장소_관리](16_저장소_관리.md) - LVM, RAID
- [Docker 문서](https://docs.docker.com/)
- [OCI Runtime Spec](https://github.com/opencontainers/runtime-spec)

## 참고 자료

- [Linux Namespaces](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [cgroups v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [OverlayFS](https://docs.kernel.org/filesystems/overlayfs.html)
- [runc](https://github.com/opencontainers/runc)
- [Container Security](https://docs.docker.com/engine/security/)
- [eBPF Documentation](https://ebpf.io/what-is-ebpf/)
- [BCC Tools](https://github.com/iovisor/bcc)
- [Cilium Documentation](https://docs.cilium.io/)
- [Kubernetes CNI Specification](https://github.com/containernetworking/cni)

---

[← 이전: 성능 튜닝](14_성능_튜닝.md) | [다음: 저장소 관리 →](16_저장소_관리.md) | [목차](00_Overview.md)
