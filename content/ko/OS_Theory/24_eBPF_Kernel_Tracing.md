[이전: 컨테이너 내부 구조](./23_Container_Internals.md)

---

# 24. eBPF와 커널 트레이싱

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. eBPF 아키텍처와 현대 커널 관측성에서의 역할을 설명할 수 있다
2. BCC (BPF Compiler Collection)와 bpftrace를 사용하여 BPF 프로그램을 작성할 수 있다
3. 고성능 패킷 처리를 위한 XDP 프로그램을 구현할 수 있다
4. 시스템 성능 분석과 보안 모니터링에 eBPF를 사용할 수 있다
5. eBPF 검증기와 안전성 보장을 설명할 수 있다

---

## 목차

1. [eBPF란?](#1-ebpf란)
2. [eBPF 아키텍처](#2-ebpf-아키텍처)
3. [BCC: BPF Compiler Collection](#3-bcc-bpf-compiler-collection)
4. [bpftrace: 고수준 트레이싱](#4-bpftrace-고수준-트레이싱)
5. [XDP: eXpress Data Path](#5-xdp-express-data-path)
6. [보안을 위한 eBPF](#6-보안을-위한-ebpf)
7. [eBPF를 활용한 성능 분석](#7-ebpf를-활용한-성능-분석)
8. [연습문제](#8-연습문제)

---

## 1. eBPF란?

### 1.1 eBPF 개요

```
eBPF (extended Berkeley Packet Filter):
  커널 코드를 수정하지 않고 Linux 커널에서 샌드박스된 프로그램 실행.

전통적인 커널 확장 방식:
  커널 모듈 작성 -> 컴파일 -> 로드 -> 커널 크래시 위험

eBPF 방식:
  BPF 프로그램 작성 -> 검증기가 안전성 확인 -> JIT 컴파일 -> 안전하게 실행

사용 사례:
  - 네트워킹: 패킷 필터링, 로드 밸런싱 (Cilium)
  - 관측성: 트레이싱, 프로파일링 (bpftrace, BCC)
  - 보안: 시스콜 필터링, 런타임 시행 (Falco)

  "eBPF는 커널에게 JavaScript가 웹 브라우저에게인 것과 같다"
  - 제어된 환경에서 커스텀 코드를 안전하게 실행
```

### 1.2 eBPF 프로그램 유형

```
BPF_PROG_TYPE_KPROBE:        커널 함수 트레이싱
BPF_PROG_TYPE_TRACEPOINT:    사전 정의된 커널 이벤트 트레이싱
BPF_PROG_TYPE_PERF_EVENT:    성능 모니터링
BPF_PROG_TYPE_XDP:           네트워크 패킷 처리
BPF_PROG_TYPE_SOCKET_FILTER: 소켓 수준 패킷 필터링
BPF_PROG_TYPE_SCHED_CLS:     트래픽 제어
BPF_PROG_TYPE_CGROUP_SKB:    cgroup별 네트워크 필터링
BPF_PROG_TYPE_LSM:           Linux Security Module 훅
```

---

## 2. eBPF 아키텍처

### 2.1 실행 흐름

```
사용자 공간                   커널 공간
┌──────────┐                 ┌──────────────────┐
│ BPF      │   bpf()        │  eBPF 검증기      │
│ 프로그램  │──시스콜────────▶│  (안전성 검사)     │
│ (C 코드) │                │        │           │
└──────────┘                │        ▼           │
                            │  JIT 컴파일러      │
                            │  (네이티브 코드로)  │
                            │        │           │
                            │        ▼           │
                            │  훅에 연결:        │
                            │  - kprobe          │
                            │  - tracepoint      │
                            │  - XDP             │
                            │        │           │
                            │        ▼           │
                            │  eBPF Maps         │
                            │  (공유 데이터)      │
                            └────────┬───────────┘
                                     │
┌──────────┐                         │
│ 사용자 앱│◀──맵 읽기───────────────┘
│ (Python/ │
│  Go/C)   │
└──────────┘
```

### 2.2 eBPF Maps

```
eBPF Maps: 커널 BPF 프로그램과 사용자 공간 애플리케이션 사이에
공유되는 키-값 데이터 구조.

맵 유형:
  BPF_MAP_TYPE_HASH:          해시 테이블
  BPF_MAP_TYPE_ARRAY:         배열 (정수 키)
  BPF_MAP_TYPE_PERF_EVENT_ARRAY: CPU별 이벤트 버퍼
  BPF_MAP_TYPE_RINGBUF:       링 버퍼 (효율적)
  BPF_MAP_TYPE_LRU_HASH:      LRU 해시 테이블
  BPF_MAP_TYPE_STACK_TRACE:   스택 트레이스
```

---

## 3. BCC: BPF Compiler Collection

### 3.1 BCC Python 예제

```python
#!/usr/bin/env python3
"""
모든 open() 시스콜을 파일 경로와 함께 트레이싱.
필요: pip install bcc
root로 실행.
"""

from bcc import BPF

# BPF 프로그램 (eBPF 바이트코드로 컴파일되는 C 코드)
bpf_program = """
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>

struct event_t {
    u32 pid;
    char comm[16];
    char filename[256];
};

BPF_PERF_OUTPUT(events);

int trace_open(struct pt_regs *ctx, const char __user *filename, int flags) {
    struct event_t event = {};

    event.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    bpf_probe_read_user_str(&event.filename, sizeof(event.filename), filename);

    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}
"""

# 로드 및 연결
b = BPF(text=bpf_program)
b.attach_kprobe(event=b.get_syscall_fnname("open"), fn_name="trace_open")

# 이벤트 처리
def print_event(cpu, data, size):
    event = b["events"].event(data)
    print(f"PID {event.pid:6d} ({event.comm.decode():16s}): {event.filename.decode()}")

b["events"].open_perf_buffer(print_event)

print("Tracing open() calls... Ctrl+C to stop")
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        break
```

### 3.2 시스템 호출 카운팅

```python
#!/usr/bin/env python3
"""프로세스별 시스템 호출 카운트."""

from bcc import BPF
from time import sleep

bpf_program = """
BPF_HASH(syscall_count, u32, u64);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *count = syscall_count.lookup(&pid);
    if (count) {
        (*count)++;
    } else {
        u64 one = 1;
        syscall_count.update(&pid, &one);
    }
    return 0;
}
"""

b = BPF(text=bpf_program)

print("Counting syscalls for 5 seconds...")
sleep(5)

print(f"{'PID':>8s} {'COUNT':>12s}")
for k, v in sorted(b["syscall_count"].items(), key=lambda x: -x[1].value):
    print(f"{k.value:>8d} {v.value:>12d}")
```

---

## 4. bpftrace: 고수준 트레이싱

### 4.1 bpftrace 원라이너

```bash
# 프로세스별 시스콜 카운트
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# 파일 열기 트레이싱
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { printf("%s %s\n", comm, str(args->filename)); }'

# read() 크기의 히스토그램
bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret > 0/ { @bytes = hist(args->ret); }'

# 프로세스 실행 트레이싱
bpftrace -e 'tracepoint:sched:sched_process_exec { printf("%d %s\n", pid, comm); }'

# 초당 컨텍스트 스위치 카운트
bpftrace -e 'tracepoint:sched:sched_switch { @[comm] = count(); } interval:s:1 { print(@); clear(@); }'

# 블록 I/O 지연 시간 히스토그램
bpftrace -e 'tracepoint:block:block_rq_issue { @start[args->dev, args->sector] = nsecs; }
             tracepoint:block:block_rq_complete /@start[args->dev, args->sector]/ {
               @usecs = hist((nsecs - @start[args->dev, args->sector]) / 1000);
               delete(@start[args->dev, args->sector]); }'

# TCP 연결 지연 시간
bpftrace -e 'kprobe:tcp_v4_connect { @start[tid] = nsecs; }
             kretprobe:tcp_v4_connect /@start[tid]/ {
               @us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]); }'
```

---

## 5. XDP: eXpress Data Path

### 5.1 XDP 개요

```
XDP: 네트워크 스택에서 가능한 가장 이른 지점에서 패킷 처리.

전통적인 패킷 경로:
  NIC → 드라이버 → sk_buff 할당 → Netfilter → TCP/IP → 애플리케이션
  (많은 할당과 복사)

XDP 패킷 경로:
  NIC → 드라이버 → XDP 프로그램 → 액션
  (sk_buff 할당 전! 최소 오버헤드)

XDP 액션:
  XDP_PASS:     일반 네트워크 스택으로 전달
  XDP_DROP:     패킷 드롭 (DDoS 완화!)
  XDP_TX:       같은 NIC으로 되돌려 보냄
  XDP_REDIRECT: 다른 NIC이나 CPU로 전송
  XDP_ABORTED:  오류, 트레이스와 함께 드롭

성능: 코어당 2,400만 패킷/초 (iptables의 ~100만 대비)
```

### 5.2 XDP 방화벽 예제

```c
/*
 * 간단한 XDP 방화벽: 특정 IP의 패킷 드롭.
 * 이것은 커널에 로드되는 BPF C 코드.
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <bpf/bpf_helpers.h>

/* 차단된 IP를 저장하는 맵 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32);      /* IPv4 주소 */
    __type(value, __u64);    /* 패킷 카운트 */
    __uint(max_entries, 1024);
} blocked_ips SEC(".maps");

SEC("xdp")
int xdp_firewall(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    /* 이더넷 헤더 파싱 */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    /* IP 헤더 파싱 */
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    /* 소스 IP가 차단 목록에 있는지 확인 */
    __u32 src_ip = ip->saddr;
    __u64 *count = bpf_map_lookup_elem(&blocked_ips, &src_ip);

    if (count) {
        /* IP가 차단됨 - 카운터 증가 후 드롭 */
        __sync_fetch_and_add(count, 1);
        return XDP_DROP;
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
```

---

## 6. 보안을 위한 eBPF

### 6.1 시스콜 모니터링

```python
#!/usr/bin/env python3
"""보안을 위한 의심스러운 시스템 호출 모니터링."""

from bcc import BPF

bpf_program = """
#include <uapi/linux/ptrace.h>

struct security_event_t {
    u32 pid;
    u32 uid;
    char comm[16];
    int syscall_nr;
};

BPF_PERF_OUTPUT(security_events);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    struct security_event_t event = {};

    /* 민감한 시스콜 모니터링 */
    int nr = args->id;

    /* ptrace (디버깅/주입), execve (실행),
     * init_module (커널 모듈 로딩) */
    if (nr == 101 || nr == 59 || nr == 175) {
        event.pid = bpf_get_current_pid_tgid() >> 32;
        event.uid = bpf_get_current_uid_gid();
        event.syscall_nr = nr;
        bpf_get_current_comm(&event.comm, sizeof(event.comm));

        security_events.perf_submit(args, &event, sizeof(event));
    }

    return 0;
}
"""

b = BPF(text=bpf_program)

syscall_names = {101: "ptrace", 59: "execve", 175: "init_module"}

def print_event(cpu, data, size):
    event = b["security_events"].event(data)
    name = syscall_names.get(event.syscall_nr, str(event.syscall_nr))
    print(f"[SECURITY] PID={event.pid} UID={event.uid} "
          f"comm={event.comm.decode()} syscall={name}")

b["security_events"].open_perf_buffer(print_event)
print("Monitoring sensitive syscalls... Ctrl+C to stop")
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        break
```

---

## 7. eBPF를 활용한 성능 분석

### 7.1 CPU 분석

```bash
# 스택 트레이스별 CPU 사용량 프로파일링
# CPU 시간이 어디에 소비되는지 확인
profile -F 99 -p $(pgrep myapp) 10

# Off-CPU 분석: 프로세스가 어디서 대기하는가?
offcputime -p $(pgrep myapp) 5

# 스케줄러 지연 시간: 태스크가 런 큐에서 얼마나 대기하는가
runqlat 1

# 프로세스별 스케줄러 지연 시간
runqslower 10000  # 10ms 이상 대기하는 태스크 표시
```

### 7.2 메모리 분석

```bash
# 스택 트레이스별 메모리 할당 추적
memleak -p $(pgrep myapp)

# 페이지 폴트 트레이싱
bpftrace -e 'software:page-faults:1 { @[comm, ustack] = count(); }'

# OOM killer 모니터링
bpftrace -e 'kprobe:oom_kill_process { printf("OOM: %s pid=%d\n", comm, pid); }'
```

---

## 8. 연습문제

### 연습문제 1: 시스템 호출 트레이서

BCC로 시스콜 트레이서를 구축하세요:
1. 특정 PID의 모든 시스콜 트레이싱
2. 유형별 시스콜 카운트 (open, read, write 등)
3. 각 시스콜 유형의 지연 시간 측정
4. 리포트 생성: 카운트와 총 시간 기준 상위 10개 시스콜
5. 정확성 검증을 위해 strace 출력과 비교

### 연습문제 2: bpftrace 스크립트

일반적인 분석을 위한 bpftrace 스크립트 작성:
1. 프로세스별 파일 열기 추적 (전체 경로 포함)
2. TCP 연결 지속 시간의 히스토그램
3. 프로세스별 DNS 쿼리 카운트
4. 장치별 디스크 I/O 지연 시간 측정
5. 과도한 컨텍스트 스위치를 하는 프로세스 감지

### 연습문제 3: XDP 패킷 카운터

XDP 기반 패킷 카운터 구축:
1. 프로토콜별(TCP/UDP/ICMP) 패킷을 카운트하는 XDP 프로그램 작성
2. BPF 맵에 카운트 저장
3. 통계를 읽고 표시하는 사용자 공간 프로그램 작성
4. 측정: 카운터가 처리할 수 있는 초당 패킷 수
5. 오버헤드 비교: XDP 카운터 vs iptables 카운터

### 연습문제 4: eBPF 보안 모니터

보안 모니터링 도구 생성:
1. 전체 명령줄과 함께 프로세스 실행(execve 호출) 모니터링
2. 민감한 디렉토리(/etc, /root)의 파일 수정 추적
3. 비정상적인 포트로의 네트워크 연결 감지
4. 권한 상승(setuid 호출) 시 경고
5. 포렌식 분석을 위해 타임스탬프와 함께 모든 이벤트 기록

### 연습문제 5: 성능 프로파일러

포괄적인 프로파일러 구축:
1. CPU 프로파일링: eBPF 스택 트레이스로부터 플레임 그래프
2. 메모리 프로파일링: 할당 추적 및 누수 감지
3. I/O 프로파일링: 파일별 지연 시간 히스토그램
4. 네트워크 프로파일링: 연결 지연 시간 및 처리량
5. 모든 메트릭이 포함된 단일 페이지 HTML 리포트 생성

---

*레슨 24 끝*
