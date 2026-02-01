# 15. 컨테이너 내부 구조

## 학습 목표
- Linux 컨테이너 격리 기술 이해
- namespaces로 리소스 격리
- cgroups로 리소스 제한
- 컨테이너 런타임 동작 원리

## 목차
1. [컨테이너 기초](#1-컨테이너-기초)
2. [Linux Namespaces](#2-linux-namespaces)
3. [Control Groups (cgroups)](#3-control-groups-cgroups)
4. [Union Filesystem](#4-union-filesystem)
5. [컨테이너 런타임](#5-컨테이너-런타임)
6. [보안](#6-보안)
7. [연습 문제](#7-연습-문제)

---

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

---

## 7. 연습 문제

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

---

[← 이전: 성능 튜닝](14_성능_튜닝.md) | [다음: 저장소 관리 →](16_저장소_관리.md) | [목차](00_Overview.md)
