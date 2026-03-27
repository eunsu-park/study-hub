[이전: 실시간 OS](./22_Real_Time_OS.md)

---

# 23. 컨테이너 내부 구조

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Linux 네임스페이스와 이것이 프로세스 격리를 제공하는 방법을 설명할 수 있다
2. 리소스 제한 및 계정을 위한 cgroups v2를 구현할 수 있다
3. 오버레이 파일시스템과 컨테이너 이미지 레이어링을 설명할 수 있다
4. Linux 프리미티브를 사용하여 최소 컨테이너 런타임을 처음부터 구축할 수 있다
5. OCI 런타임 사양과 컨테이너 생명주기를 분석할 수 있다

---

## 목차

1. [컨테이너 vs VM](#1-컨테이너-vs-vm)
2. [Linux 네임스페이스](#2-linux-네임스페이스)
3. [Control Groups (cgroups v2)](#3-control-groups-cgroups-v2)
4. [오버레이 파일시스템](#4-오버레이-파일시스템)
5. [최소 컨테이너 구축](#5-최소-컨테이너-구축)
6. [OCI 런타임 사양](#6-oci-런타임-사양)
7. [컨테이너 네트워킹](#7-컨테이너-네트워킹)
8. [연습문제](#8-연습문제)

---

## 1. 컨테이너 vs VM

### 1.1 아키텍처 비교

```
가상 머신:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  App A   │ │  App B   │ │  App C   │
  ├──────────┤ ├──────────┤ ├──────────┤
  │ 바이너리/│ │ 바이너리/│ │ 바이너리/│
  │ 라이브러리│ │ 라이브러리│ │ 라이브러리│
  ├──────────┤ ├──────────┤ ├──────────┤
  │ 게스트 OS │ │ 게스트 OS │ │ 게스트 OS │  ← VM당 전체 OS
  └────┬─────┘ └────┬─────┘ └────┬─────┘
  ┌────┴──────────────┴──────────────┴────┐
  │            하이퍼바이저                 │
  ├───────────────────────────────────────┤
  │            호스트 OS                   │
  └───────────────────────────────────────┘

컨테이너:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  App A   │ │  App B   │ │  App C   │
  ├──────────┤ ├──────────┤ ├──────────┤
  │ 바이너리/│ │ 바이너리/│ │ 바이너리/│
  │ 라이브러리│ │ 라이브러리│ │ 라이브러리│
  └────┬─────┘ └────┬─────┘ └────┬─────┘
  ┌────┴──────────────┴──────────────┴────┐
  │     컨테이너 런타임 (runc)             │
  ├───────────────────────────────────────┤
  │            호스트 OS (공유 커널)        │  ← 하나의 커널
  └───────────────────────────────────────┘
```

---

## 2. Linux 네임스페이스

### 2.1 네임스페이스 유형

```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>

/*
 * Linux 네임스페이스는 서로 다른 시스템 리소스를 격리:
 *
 * CLONE_NEWPID:   프로세스 ID (컨테이너가 PID 1을 봄)
 * CLONE_NEWNS:    마운트 포인트 (자체 파일시스템 뷰)
 * CLONE_NEWNET:   네트워크 스택 (자체 인터페이스, IP)
 * CLONE_NEWUTS:   호스트네임 (자체 호스트네임)
 * CLONE_NEWIPC:   IPC (자체 세마포어, 공유 메모리)
 * CLONE_NEWUSER:  사용자 ID (내부에서 root, 외부에서 비특권)
 * CLONE_NEWCGROUP: Cgroup 루트
 */

#define STACK_SIZE (1024 * 1024)

int child_fn(void *arg) {
    /* 컨테이너 내부! */

    /* 호스트네임 설정 */
    sethostname("container", 9);

    char hostname[64];
    gethostname(hostname, sizeof(hostname));
    printf("[Container] Hostname: %s\n", hostname);
    printf("[Container] PID: %d\n", getpid());
    printf("[Container] UID: %d\n", getuid());

    /* 새 PID 네임스페이스를 위한 proc 마운트 */
    mount("proc", "/proc", "proc", 0, NULL);

    /* 컨테이너 내부에서 셸 실행 */
    char *args[] = {"/bin/sh", NULL};
    execvp(args[0], args);
    perror("execvp");
    return 1;
}

int main(void) {
    char *stack = malloc(STACK_SIZE);
    if (!stack) { perror("malloc"); return 1; }

    printf("[Host] PID: %d\n", getpid());

    /* 새 네임스페이스에서 자식 프로세스 생성 */
    int flags = CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWUTS |
                CLONE_NEWNET | CLONE_NEWIPC | SIGCHLD;

    pid_t child = clone(child_fn, stack + STACK_SIZE, flags, NULL);
    if (child == -1) {
        perror("clone");
        free(stack);
        return 1;
    }

    printf("[Host] Container PID (from host view): %d\n", child);
    waitpid(child, NULL, 0);

    free(stack);
    return 0;
}
```

---

## 3. Control Groups (cgroups v2)

### 3.1 cgroups v2 아키텍처

```
cgroups v2: 리소스 관리를 위한 통합 계층 구조.

제어 가능한 리소스:
  cpu:     CPU 시간 할당
  memory:  메모리 제한, 계정
  io:      블록 I/O 대역폭
  pids:    프로세스 수 제한
  cpuset:  CPU/메모리 노드 할당
  rdma:    RDMA 리소스

cgroup 계층 구조:
  /sys/fs/cgroup/
  ├── cgroup.controllers    # 사용 가능한 컨트롤러
  ├── cgroup.subtree_control # 자식에 활성화된 것
  ├── system.slice/         # 시스템 서비스
  ├── user.slice/           # 사용자 세션
  └── my_container/         # 우리의 컨테이너 cgroup
      ├── cgroup.procs      # 이 cgroup의 PID
      ├── cpu.max           # CPU 제한
      ├── memory.max        # 메모리 제한
      ├── memory.current    # 현재 메모리 사용량
      ├── io.max            # I/O 대역폭 제한
      └── pids.max          # 프로세스 수 제한
```

### 3.2 cgroups 구현

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define CGROUP_ROOT "/sys/fs/cgroup"

void write_file(const char *path, const char *content) {
    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); return; }
    fputs(content, fp);
    fclose(fp);
}

void setup_cgroup(const char *name, int cpu_percent,
                  long memory_bytes, int max_pids) {
    char path[512];

    /* cgroup 디렉토리 생성 */
    snprintf(path, sizeof(path), "%s/%s", CGROUP_ROOT, name);
    mkdir(path, 0755);

    /* 컨트롤러 활성화 */
    snprintf(path, sizeof(path), "%s/cgroup.subtree_control", CGROUP_ROOT);
    write_file(path, "+cpu +memory +pids +io");

    /* CPU 제한: cpu.max = "quota period" (마이크로초) */
    /* 50% CPU = "50000 100000" (100ms 중 50ms) */
    snprintf(path, sizeof(path), "%s/%s/cpu.max", CGROUP_ROOT, name);
    char cpu_max[64];
    snprintf(cpu_max, sizeof(cpu_max), "%d 100000",
             cpu_percent * 1000);
    write_file(path, cpu_max);
    printf("CPU limit: %d%%\n", cpu_percent);

    /* 메모리 제한 */
    snprintf(path, sizeof(path), "%s/%s/memory.max", CGROUP_ROOT, name);
    char mem_max[64];
    snprintf(mem_max, sizeof(mem_max), "%ld", memory_bytes);
    write_file(path, mem_max);
    printf("Memory limit: %ld bytes\n", memory_bytes);

    /* PID 제한 */
    snprintf(path, sizeof(path), "%s/%s/pids.max", CGROUP_ROOT, name);
    char pids_max[32];
    snprintf(pids_max, sizeof(pids_max), "%d", max_pids);
    write_file(path, pids_max);
    printf("PID limit: %d\n", max_pids);

    /* 현재 프로세스를 cgroup으로 이동 */
    snprintf(path, sizeof(path), "%s/%s/cgroup.procs", CGROUP_ROOT, name);
    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d", getpid());
    write_file(path, pid_str);
    printf("Process %s moved to cgroup %s\n", pid_str, name);
}
```

---

## 4. 오버레이 파일시스템

### 4.1 컨테이너 이미지의 작동 원리

```
OverlayFS: 컨테이너 이미지를 위한 유니온 마운트 파일시스템.

Docker 이미지 레이어:
  레이어 3 (상단): 애플리케이션 코드      (읽기-쓰기)
  레이어 2:       pip install packages   (읽기 전용)
  레이어 1:       apt-get update         (읽기 전용)
  레이어 0:       Ubuntu 베이스 이미지    (읽기 전용)

OverlayFS 병합 뷰:
  ┌─────────────────────────────┐
  │     병합 뷰 (유니온)         │  ← 컨테이너가 보는 것
  ├─────────────────────────────┤
  │  upperdir (읽기-쓰기)       │  ← 컨테이너가 여기에 쓰기
  ├─────────────────────────────┤
  │  lowerdir (읽기 전용 레이어) │  ← 이미지 레이어
  └─────────────────────────────┘

  읽기: 먼저 upper 확인, 그 다음 lower 레이어
  쓰기: 항상 upper 레이어로 (필요시 copy-up)
  삭제: upper 레이어에 whiteout 파일
```

### 4.2 OverlayFS 설정

```c
#include <stdio.h>
#include <sys/mount.h>
#include <sys/stat.h>

void setup_overlay(const char *lower, const char *upper,
                   const char *work, const char *merged) {
    /* 디렉토리 생성 */
    mkdir(upper, 0755);
    mkdir(work, 0755);
    mkdir(merged, 0755);

    /* overlayfs 마운트 */
    char options[1024];
    snprintf(options, sizeof(options),
             "lowerdir=%s,upperdir=%s,workdir=%s",
             lower, upper, work);

    int ret = mount("overlay", merged, "overlay", 0, options);
    if (ret != 0) {
        perror("mount overlay");
        return;
    }

    printf("OverlayFS mounted at %s\n", merged);
    printf("  Lower (read-only): %s\n", lower);
    printf("  Upper (read-write): %s\n", upper);
}
```

---

## 5. 최소 컨테이너 구축

### 5.1 미니 컨테이너 런타임

```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <string.h>

#define STACK_SIZE (1024 * 1024)

typedef struct {
    char *rootfs;
    char *hostname;
    char **command;
    int cpu_percent;
    long memory_limit;
} container_config_t;

int container_main(void *arg) {
    container_config_t *config = (container_config_t *)arg;

    /* 1. 호스트네임 설정 */
    sethostname(config->hostname, strlen(config->hostname));

    /* 2. 마운트 네임스페이스 설정 */
    /* 새 rootfs로 루트 전환 */
    if (chroot(config->rootfs) != 0) {
        perror("chroot");
        return 1;
    }
    chdir("/");

    /* 3. 필수 파일시스템 마운트 */
    mount("proc", "/proc", "proc", 0, NULL);
    mount("sysfs", "/sys", "sysfs", 0, NULL);
    mount("tmpfs", "/tmp", "tmpfs", 0, NULL);

    /* 4. 컨테이너 정보 출력 */
    printf("=== Container Started ===\n");
    printf("Hostname: %s\n", config->hostname);
    printf("PID: %d (appears as 1 inside container)\n", getpid());
    printf("Root: %s\n", config->rootfs);

    /* 5. 명령어 실행 */
    execvp(config->command[0], config->command);
    perror("execvp");
    return 1;
}

void run_container(container_config_t *config) {
    char *stack = malloc(STACK_SIZE);

    int flags = CLONE_NEWPID    /* 새 PID 네임스페이스 */
              | CLONE_NEWNS     /* 새 마운트 네임스페이스 */
              | CLONE_NEWUTS    /* 새 UTS 네임스페이스 (호스트네임) */
              | CLONE_NEWNET    /* 새 네트워크 네임스페이스 */
              | CLONE_NEWIPC    /* 새 IPC 네임스페이스 */
              | SIGCHLD;

    pid_t child = clone(container_main, stack + STACK_SIZE,
                        flags, config);
    if (child == -1) {
        perror("clone");
        free(stack);
        return;
    }

    /* 컨테이너 프로세스에 대한 cgroups 설정 */
    printf("[Host] Container process: %d\n", child);

    int status;
    waitpid(child, &status, 0);
    printf("[Host] Container exited with status %d\n",
           WEXITSTATUS(status));

    free(stack);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rootfs> <command> [args...]\n", argv[0]);
        return 1;
    }

    char *cmd[] = {argv[2], NULL};  /* 단순화 */

    container_config_t config = {
        .rootfs = argv[1],
        .hostname = "mini-container",
        .command = cmd,
        .cpu_percent = 50,
        .memory_limit = 256 * 1024 * 1024,
    };

    run_container(&config);
    return 0;
}
```

---

## 6. OCI 런타임 사양

### 6.1 OCI 생명주기

```
OCI (Open Container Initiative)가 정의하는 컨테이너 생명주기:

1. create:  네임스페이스, cgroups, rootfs 설정
2. start:   컨테이너의 엔트리포인트 실행
3. running: 컨테이너 실행 중
4. stop:    SIGTERM 전송, 타임아웃 후 SIGKILL
5. delete:  모든 리소스 정리

컨테이너 상태:
  {
    "ociVersion": "1.0.0",
    "id": "container-abc123",
    "status": "running",
    "pid": 12345,
    "bundle": "/containers/myapp",
    "annotations": {}
  }
```

---

## 7. 컨테이너 네트워킹

### 7.1 네트워크 네임스페이스와 veth 쌍

```
컨테이너 네트워킹은 가상 이더넷(veth) 쌍을 사용:

  호스트 네임스페이스            컨테이너 네임스페이스
  ┌──────────────┐           ┌──────────────┐
  │   eth0       │           │   eth0       │
  │   (실제 NIC) │           │   (veth 피어)│
  │   bridge0    │◄─ veth ──▶│   172.17.0.2 │
  │   172.17.0.1 │           │              │
  └──────────────┘           └──────────────┘

  NAT/iptables 규칙이 처리:
  - 컨테이너 -> 인터넷: Masquerade (SNAT)
  - 인터넷 -> 컨테이너: 포트 매핑 (DNAT)
```

---

## 8. 연습문제

### 연습문제 1: 네임스페이스 탐구

Linux 네임스페이스를 실습으로 탐구하세요:
1. PID 네임스페이스 생성: 내부에서 PID 1, 외부에서 실제 PID 확인
2. UTS 네임스페이스 생성: 호스트에 영향 없이 호스트네임 변경
3. 마운트 네임스페이스 생성: 호스트에 보이지 않는 tmpfs 마운트
4. 네트워크 네임스페이스 생성: 처음에 인터페이스가 없음을 확인
5. 모든 네임스페이스를 하나의 프로그램에 결합하여 미니 컨테이너 구현

### 연습문제 2: cgroups 리소스 제한

cgroups 기반 리소스 제한을 구현하세요:
1. 50% CPU 제한으로 cgroup 생성하고 스트레스 테스트로 검증
2. 메모리 제한을 100 MB로 설정하고 메모리 소비 프로그램으로 테스트
3. PID 제한을 10으로 설정하고 초과 fork 시도
4. cgroup 통계 파일을 통해 리소스 사용량 모니터링
5. 종료 시 정리하는 간단한 cgroup 관리자 구현

### 연습문제 3: 나만의 컨테이너 구축

C로 최소 컨테이너 런타임 생성:
1. 격리를 위해 네임스페이스 플래그와 함께 clone() 사용
2. 최소 rootfs(busybox)로 chroot 설정
3. 컨테이너 내부에 /proc과 /sys 마운트
4. CPU와 메모리에 대한 cgroup 제한 적용
5. /bin/sh를 실행하고 격리 검증

### 연습문제 4: OverlayFS 이미지 레이어

오버레이 파일시스템으로 작업하세요:
1. 3계층 오버레이 파일시스템 생성 (base, packages, app)
2. copy-up 시연: lower 레이어의 파일 수정
3. whiteout 시연: lower 레이어의 파일 삭제
4. 측정: lower vs upper 레이어의 읽기 성능
5. 간단한 "docker commit" 구현: upper 레이어를 새 lower로 스냅샷

### 연습문제 5: 컨테이너 네트워킹

컨테이너 네트워킹을 수동으로 설정하세요:
1. 네트워크 네임스페이스 생성
2. 호스트와 컨테이너 네임스페이스를 연결하는 veth 쌍 생성
3. IP 주소 할당 및 라우팅 설정
4. NAT(iptables masquerade)로 인터넷 접근 활성화
5. 컨테이너 서비스를 노출하는 포트 포워딩 설정

---

*레슨 23 끝*
