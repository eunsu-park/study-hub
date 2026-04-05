# 24. eBPF 관측 가능성(Observability)

**이전**: [OpenTelemetry 파이프라인](./23_OpenTelemetry_Pipelines.md) | **다음**: [지속적 프로파일링](./25_Continuous_Profiling.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. eBPF의 작동 원리와 애플리케이션 코드 변경 없이 관측 가능성을 구현하는 방법을 설명할 수 있습니다
2. bpftrace를 사용하여 프로덕션 디버깅을 위한 원라이너(one-liner) 및 스크립트 기반 관측 가능성 프로브를 작성할 수 있습니다
3. Kubernetes 네트워크 관측 가능성을 위한 Cilium Hubble의 아키텍처를 설명할 수 있습니다
4. eBPF 기반 도구를 적용하여 시스템 콜(system call), 네트워크 트래픽, 애플리케이션 동작을 커널 수준에서 관찰할 수 있습니다
5. eBPF 기반 관측 가능성과 전통적인 계측(instrumentation) 접근법을 비교할 수 있습니다
6. eBPF 관측 가능성이 적합한 경우와 OpenTelemetry 기반 계측이 적합한 경우를 평가할 수 있습니다

---

eBPF(extended Berkeley Packet Filter)는 커널 소스 코드를 수정하거나 커널 모듈을 로드하지 않고 Linux 커널 내에서 샌드박스된 프로그램을 실행할 수 있는 기술입니다. 관측 가능성 측면에서, 이는 언어, 프레임워크, 소스 코드 접근 여부에 관계없이 커널 수준에서 프로브를 연결하여 모든 애플리케이션을 계측할 수 있다는 것을 의미합니다.

> **비유 -- 공항 보안 카메라**: 전통적인 계측(OpenTelemetry)은 모든 승객에게 바디캠을 착용하도록 요청하는 것과 같습니다. 협조가 필요하며, 각 승객이 동의해야 합니다. eBPF는 공항의 보안 카메라와 같습니다 -- 모든 사람이 통과하는 것을 관찰하고, 승객의 협조가 필요 없으며, 아무도 행동을 변경하지 않으면서 이동 패턴을 추적할 수 있습니다. 카메라(eBPF 프로그램)는 고정 지점(커널 훅)에 설치되어 모든 트래픽을 관찰합니다.

## 1. eBPF 기초(Fundamentals)

### 1.1 eBPF 작동 방식

```
사용자 공간(User Space)                    커널 공간(Kernel Space)
┌──────────────────┐                   ┌──────────────────────────┐
│                  │                   │                          │
│  eBPF 프로그램    │  로드 + 검증     │  eBPF 가상 머신            │
│  (C 또는 Rust)   │ ──────────────→  │  ┌────────────────────┐  │
│                  │                   │  │ JIT 컴파일된 코드   │  │
│  bpftrace 스크립트│                   │  │ (커널 속도로 실행)   │  │
│  또는 컴파일된    │                   │  │                    │  │
│  바이너리        │                   │  └────────┬───────────┘  │
└──────────────────┘                   │           │              │
                                       │  훅 포인트에 연결:        │
       ┌───────────────────────────────┤  ┌──────────────────┐   │
       │  결과 전달:                    │  │ - kprobes (커널   │   │
       │  - BPF maps (공유 메모리)      │  │   함수 진입)       │   │
       │  - Perf events (링 버퍼)       │  │ - tracepoints     │   │
       │  - 트레이스 파이프 출력         │  │ - uprobes (사용자  │   │
       │                               │  │   함수 진입)       │   │
       ▼                               │  │ - XDP (네트워크    │   │
┌──────────────────┐                   │  │   패킷 훅)        │   │
│  사용자 공간 도구  │                   │  │ - perf events     │   │
│  (대시보드,       │                   │  └──────────────────┘   │
│   CLI 출력,      │                   │                          │
│   Prometheus     │                   └──────────────────────────┘
│   exporter)      │
└──────────────────┘
```

### 1.2 eBPF 안전 모델(Safety Model)

커널 검증기(verifier)는 eBPF 프로그램이 안전한지 확인합니다:

| 검사 항목 | 목적 |
|----------|------|
| **무한 루프 금지** | 커널 컨텍스트에서 무한 루프 방지 |
| **메모리 경계 검사** | 버퍼 오버플로우 방지 |
| **스택 크기 제한** (512 바이트) | 스택 오버플로우 방지 |
| **명령어 수 제한** (~100만 검증) | 프로그램 복잡도 제한 |
| **임의 커널 메모리 접근 금지** | 헬퍼 함수를 통해서만 허용 |
| **슬리핑 또는 블로킹 금지** | eBPF 프로그램은 빠르게 완료되어야 함 |

### 1.3 관측 가능성용 eBPF 훅 포인트(Hook Points)

| 훅 유형 | 관찰 대상 | 예시 |
|--------|---------|------|
| **kprobes** | 커널 함수 진입/종료 | `tcp_connect`, `do_sys_open`, `vfs_write` |
| **tracepoints** | 안정적인 커널 이벤트 포인트 | `sched:sched_process_exec`, `net:net_dev_xmit` |
| **uprobes** | 사용자 공간 함수 진입/종료 | OpenSSL의 `SSL_write`, libc의 `malloc` |
| **USDT** | 사용자 정의 정적 트레이스포인트 | Python GC 이벤트, MySQL 쿼리 이벤트 |
| **XDP** | 네트워크 패킷 도착 (프리스택) | 패킷 필터링, DDoS 완화 |
| **tc** | 트래픽 컨트롤 (포스트스택) | 네트워크 정책 시행 |
| **perf events** | 하드웨어 및 소프트웨어 카운터 | CPU 캐시 미스, 페이지 폴트 |
| **LSM** | Linux 보안 모듈 훅 | 보안 정책 시행 |

---

## 2. bpftrace

### 2.1 원라이너 예시(One-Liner Examples)

bpftrace는 eBPF를 위한 고수준 트레이싱 언어로, 임시(ad-hoc) 프로덕션 디버깅에 이상적입니다:

```bash
# 시스템 콜 이름별 카운트 (시스템이 무엇을 하고 있는가?)
bpftrace -e 'tracepoint:syscalls:sys_enter_* { @[probe] = count(); }'

# 새 프로세스 생성 추적 (어떤 프로세스가 생성되고 있는가?)
bpftrace -e 'tracepoint:sched:sched_process_exec { printf("%s executed %s\n", comm, str(args->filename)); }'

# read() 크기 히스토그램 (프로세스가 얼마나 많은 데이터를 읽고 있는가?)
bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret > 0/ { @bytes = hist(args->ret); }'

# TCP 연결 추적 (누가 어디로 연결하는가?)
bpftrace -e 'kprobe:tcp_connect { printf("%s → %s\n", comm, ntop(((struct sock *)arg0)->__sk_common.skc_daddr)); }'

# 프로세스별 파일 열기 카운트
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'

# DNS 조회 지연 히스토그램
bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:getaddrinfo { @start[tid] = nsecs; }
             uretprobe:/lib/x86_64-linux-gnu/libc.so.6:getaddrinfo /@start[tid]/ {
               @dns_latency_us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]);
             }'
```

### 2.2 프로덕션 준비된 bpftrace 스크립트

```
#!/usr/bin/env bpftrace
/*
 * http_latency.bt -- Go 서비스의 HTTP 요청 지연 추적
 * 사용법: bpftrace http_latency.bt -p $(pidof my-service)
 */

// HTTP 핸들러 시작 추적
uprobe:/usr/local/bin/my-service:net/http.(*ServeMux).ServeHTTP
{
    @start[tid] = nsecs;
    @count++;
}

// HTTP 핸들러 반환 추적
uretprobe:/usr/local/bin/my-service:net/http.(*ServeMux).ServeHTTP
/@start[tid]/
{
    $duration_us = (nsecs - @start[tid]) / 1000;
    @latency_us = hist($duration_us);

    // 느린 요청에 대한 알림
    if ($duration_us > 1000000) {
        printf("SLOW REQUEST: %d us (tid=%d)\n", $duration_us, tid);
    }

    delete(@start[tid]);
}

// Ctrl-C 시 요약 출력
END
{
    printf("\n--- HTTP 요청 지연 요약 ---\n");
    printf("총 요청 수: %d\n", @count);
    print(@latency_us);
}
```

### 2.3 bpftrace vs 전통적 도구

| 기능 | 전통적 도구 | bpftrace |
|------|-----------|----------|
| CPU 프로파일링 | `perf top` | `profile:hz:99 { @[ustack] = count(); }` |
| 디스크 I/O | `iostat` | `tracepoint:block:block_rq_complete { @us = hist(args->nr_sector * 512); }` |
| 네트워크 연결 | `ss`, `netstat` | `kprobe:tcp_connect { @[comm] = count(); }` |
| 파일 시스템 지연 | N/A (내장 도구 없음) | `kprobe:vfs_read { @start[tid] = nsecs; }` |
| 함수 지연 | N/A (APM 필요) | `uprobe:function { @start[tid] = nsecs; }` |

---

## 3. BCC 도구(Tools)

### 3.1 관측 가능성용 필수 BCC 도구

BCC(BPF Compiler Collection)는 프로덕션 준비된 도구를 제공합니다:

```bash
# --- 네트워크 ---
# 지연 시간을 포함한 TCP 연결 추적
tcpconnect -t           # 타임스탬프, PID, 대상, 지연 표시

# TCP 재전송 추적 (네트워크 신뢰성 지표)
tcpretrans              # 소스, 대상, 상태와 함께 재전송 표시

# 원격 호스트별 TCP 왕복 시간 요약
tcprtt -i 1 -d 10      # 10초 동안 1초 간격

# --- 스토리지 ---
# 블록 I/O 지연 추적
biolatency -m           # 밀리초 단위 블록 I/O 지연 히스토그램

# 느린 파일 시스템 작업 추적
ext4slower 1            # 1ms보다 느린 ext4 작업 표시

# --- CPU ---
# CPU 프로파일링 (플레임 그래프 입력)
profile -af 60 > profile.out     # 60초 스택 샘플링
flamegraph.pl profile.out > profile.svg

# --- 애플리케이션 ---
# 특정 프로세스의 함수 지연 추적
funclatency -p $(pidof my-service) 'SSL_read' -m   # SSL 읽기 지연

# 메모리 할당 추적
memleak -p $(pidof my-service) --top=10             # 상위 10 할당 사이트

# --- DNS ---
# DNS 쿼리 추적
gethostlatency          # 쿼리별 DNS 해석 지연 표시
```

### 3.2 BCC 도구와 관측 가능성 신호 매핑

| 관측 가능성 요구 | BCC 도구 | 출력 |
|----------------|---------|------|
| 서비스 간 네트워크 지연 | `tcpconnect`, `tcprtt` | 연결별 지연 |
| 디스크 I/O 병목 | `biolatency`, `biosnoop` | I/O 지연 히스토그램 |
| DNS 해석 문제 | `gethostlatency` | 쿼리별 해석 시간 |
| CPU 핫스팟 | `profile` | 스택 트레이스 빈도 |
| 메모리 누수 | `memleak` | 할당 사이트 및 크기 |
| 파일 시스템 느림 | `ext4slower`, `xfsslower` | 느린 FS 작업 |
| TCP 재전송 | `tcpretrans` | 네트워크 신뢰성 문제 |

---

## 4. Cilium Hubble

### 4.1 아키텍처

Cilium은 Kubernetes 네트워킹에 eBPF를 사용하고, Hubble은 네트워크 관측 가능성을 위해 이를 확장합니다:

```
┌─────────────────────────────────────────────┐
│              Hubble UI                       │
│         (서비스 맵 + 플로우 로그)              │
└──────────────────┬──────────────────────────┘
                   │ gRPC
┌──────────────────▼──────────────────────────┐
│           Hubble Relay                       │
│    (모든 노드에서 집계)                        │
└──────┬──────────────────────────┬───────────┘
       │                          │
┌──────▼──────────┐     ┌────────▼────────────┐
│ Hubble Agent    │     │ Hubble Agent        │
│ (Node 1)        │     │ (Node 2)            │
│                 │     │                     │
│ Cilium Agent    │     │ Cilium Agent        │
│ ┌─────────────┐ │     │ ┌─────────────┐    │
│ │ eBPF Programs│ │     │ │ eBPF Programs│   │
│ │ (datapath)  │ │     │ │ (datapath)   │   │
│ └─────────────┘ │     │ └─────────────┘    │
└─────────────────┘     └────────────────────┘
```

### 4.2 Hubble CLI 예시

```bash
# 네임스페이스의 모든 네트워크 플로우 관찰
hubble observe --namespace production

# 특정 파드로/에서의 트래픽 관찰
hubble observe --to-pod production/payment-service-abc123

# HTTP 상태 코드로 필터링 (L7 가시성)
hubble observe --http-status 500 --namespace production

# DNS 쿼리로 필터링
hubble observe --protocol DNS --namespace production

# 드롭된 패킷 표시 (네트워크 정책 위반)
hubble observe --verdict DROPPED --namespace production

# 두 서비스 간 트래픽 추적
hubble observe \
  --from-label app=order-service \
  --to-label app=inventory-service \
  --protocol HTTP

# 분석을 위해 플로우를 JSON으로 내보내기
hubble observe --output json --namespace production > flows.json
```

### 4.3 Hubble 메트릭

Hubble은 eBPF로 관찰된 네트워크 플로우에서 Prometheus 메트릭을 내보냅니다:

```yaml
# Hubble 메트릭을 위한 Cilium Helm values
hubble:
  enabled: true
  metrics:
    enabled:
      - dns
      - drop
      - tcp
      - flow
      - icmp
      - httpV2:exemplars=true;labelsContext=source_ip,source_namespace,source_workload,destination_ip,destination_namespace,destination_workload
    serviceMonitor:
      enabled: true
```

```promql
# 서비스 간 HTTP 요청 비율 (eBPF에서, 계측 없이)
sum(rate(hubble_http_requests_total{
  source_workload="order-service",
  destination_workload="inventory-service"
}[5m]))

# DNS 해석 실패
sum(rate(hubble_dns_responses_total{rcode!="No Error"}[5m])) by (rcode)

# 사유별 드롭된 패킷 (네트워크 정책 디버깅)
sum(rate(hubble_drop_total[5m])) by (reason)

# 서비스 간 TCP 연결 지연
histogram_quantile(0.99,
  sum by (le, source_workload, destination_workload) (
    rate(hubble_tcp_connect_duration_seconds_bucket[5m])
  )
)
```

---

## 5. eBPF 기반 자동 계측(Automatic Instrumentation)

### 5.1 Beyla (Grafana)

Beyla는 eBPF를 사용하여 코드 변경 없이 HTTP 및 gRPC 서비스를 자동으로 계측합니다:

```yaml
# Beyla 구성
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: beyla
spec:
  template:
    spec:
      containers:
        - name: beyla
          image: grafana/beyla:latest
          securityContext:
            privileged: true    # eBPF에 필요
          env:
            - name: BEYLA_OPEN_PORT
              value: "8080,8443,3000"  # 계측할 포트
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://otel-collector:4318"
            - name: BEYLA_SERVICE_NAMESPACE
              value: "production"
          volumeMounts:
            - name: cgroup
              mountPath: /sys/fs/cgroup
            - name: debug
              mountPath: /sys/kernel/debug
      volumes:
        - name: cgroup
          hostPath:
            path: /sys/fs/cgroup
        - name: debug
          hostPath:
            path: /sys/kernel/debug
```

**Beyla가 자동으로 캡처하는 것 (코드 변경 없음):**
- HTTP 요청 지속 시간, 상태 코드, 메서드, 경로
- gRPC 요청 지속 시간, 상태 코드, 메서드
- SQL 쿼리 지속 시간 (데이터베이스 라이브러리의 uprobe를 통해)
- 분산 트레이스 컨텍스트 전파 (헤더 검사를 통해)

### 5.2 비교: eBPF 자동 계측 vs OTel 자동 계측

| 측면 | eBPF (Beyla, Pixie) | OTel 자동 계측 |
|------|-------------------|--------------------------|
| **코드 변경** | 없음 (커널에서 관찰) | 최소 (에이전트/라이브러리 추가) |
| **언어 지원** | 모든 언어 (커널 수준) | 언어별 (Python, Java, Go 등) |
| **배포** | DaemonSet 또는 사이드카 (특권 필요) | 애플리케이션당 라이브러리 또는 에이전트 |
| **세분성** | HTTP/gRPC/SQL 경계 | 라이브러리 경계 + 커스텀 스팬 |
| **비즈니스 컨텍스트** | 없음 (애플리케이션 의미론 불가) | 커스텀 속성 주입 가능 |
| **오버헤드** | 매우 낮음 (~1-2% CPU) | 낮음-중간 (3-5% CPU) |
| **보안** | 특권 모드 필요 | 특별 권한 불필요 |
| **최적 용도** | 빠른 성과, 레거시 앱, 폴리글랏 | 심층 계측, 커스텀 텔레메트리 |

---

## 6. 보안 관측 가능성을 위한 eBPF

### 6.1 Tetragon (런타임 보안)

Tetragon은 보안 관련 관측 가능성을 위해 eBPF를 사용합니다:

```yaml
# Tetragon 트레이싱 정책: 의심스러운 파일 접근 감지
apiVersion: cilium.io/v1alpha1
kind: TracingPolicy
metadata:
  name: sensitive-file-access
spec:
  kprobes:
    - call: "security_file_open"
      syscall: false
      args:
        - index: 0
          type: "file"
      selectors:
        - matchArgs:
            - index: 0
              operator: "Prefix"
              values:
                - "/etc/shadow"
                - "/etc/passwd"
                - "/root/.ssh"
                - "/var/run/secrets/kubernetes.io"
      return: true
      returnArg:
        index: 0
        type: "int"
```

```bash
# 보안 이벤트 관찰
tetra getevents --namespace production

# 출력:
# process: /usr/bin/cat
# args: /etc/shadow
# pod: production/compromised-pod
# action: ALERT
```

### 6.2 네트워크 정책 관측 가능성

```bash
# 어떤 네트워크 정책이 시행되고 있는지 확인
hubble observe --verdict DROPPED -o json | jq '.flow.drop_reason'

# 일반적인 드롭 사유:
# - POLICY_DENIED: Cilium NetworkPolicy가 트래픽을 차단
# - UNSUPPORTED_L3_PROTOCOL: 알 수 없는 프로토콜
# - CT_MAP_INSERTION_FAILED: 연결 추적 테이블 가득 참 (스케일 문제)
```

---

## 7. eBPF vs OpenTelemetry 사용 시점

### 7.1 결정 프레임워크(Decision Framework)

```
관찰이 필요한 대상?
    │
    ├── 애플리케이션 비즈니스 로직 (주문 생성, 결제 처리)
    │   └── OpenTelemetry 사용 (커스텀 스팬, 메트릭, 로그)
    │
    ├── HTTP/gRPC 요청 패턴 (코드 접근 불가)
    │   └── eBPF 사용 (Beyla, Hubble)
    │
    ├── 서비스 간 네트워크 통신
    │   └── Cilium Hubble 사용 (eBPF)
    │
    ├── 커널 수준 성능 (시스템 콜, I/O, TCP)
    │   └── eBPF 사용 (bpftrace, BCC 도구)
    │
    ├── 보안 이벤트 (파일 접근, 프로세스 실행)
    │   └── eBPF 사용 (Tetragon, Falco)
    │
    └── 비즈니스 컨텍스트와 네트워크/커널 세부 사항 모두
        └── 둘 다 사용: OTel은 앱 수준, eBPF는 인프라 수준
```

### 7.2 상호 보완적 사용

최고의 관측 가능성 스택은 둘 다 사용합니다:

```
┌─────────────────────────────────────────────────┐
│ 애플리케이션 계층 (OTel)                          │
│ - 비즈니스 메트릭 (주문, 매출)                    │
│ - 도메인 속성을 가진 커스텀 스팬                   │
│ - 비즈니스 컨텍스트를 가진 구조화된 로그            │
├─────────────────────────────────────────────────┤
│ 서비스 통신 계층 (eBPF / Hubble)                  │
│ - HTTP/gRPC 골든 시그널 (제로 계측)               │
│ - 서비스 의존성 맵                                │
│ - 네트워크 정책 시행                              │
├─────────────────────────────────────────────────┤
│ 커널 / 인프라 계층 (eBPF / BCC)                   │
│ - 시스템 콜 지연                                  │
│ - TCP 연결 품질                                   │
│ - 디스크 I/O 패턴                                │
│ - CPU 스케줄링                                   │
└─────────────────────────────────────────────────┘
```

---

## 8. 실전 eBPF 디버깅 레시피

### 8.1 "이 서비스가 왜 느린가?"

```bash
# 1단계: CPU 바운드인가?
bpftrace -e 'profile:hz:99 /comm == "payment-svc"/ { @[ustack(5)] = count(); }' -c 'sleep 10'

# 2단계: I/O 바운드인가?
bpftrace -e 'tracepoint:syscalls:sys_exit_read /comm == "payment-svc" && args->ret > 0/ {
  @read_latency = hist(nsecs - @start[tid]);
}'

# 3단계: 네트워크 바운드인가?
tcpconnect -p $(pidof payment-svc) -t     # TCP 연결 지연 표시
tcpretrans -p $(pidof payment-svc)         # 재전송 표시

# 4단계: DNS인가?
gethostlatency -p $(pidof payment-svc)     # DNS 해석 시간 표시

# 5단계: 락 경합인가?
bpftrace -e 'uprobe:/lib/libpthread.so:pthread_mutex_lock /comm == "payment-svc"/ {
  @start[tid] = nsecs;
}
uretprobe:/lib/libpthread.so:pthread_mutex_lock /comm == "payment-svc" && @start[tid]/ {
  @lock_hold_us = hist((nsecs - @start[tid]) / 1000);
  delete(@start[tid]);
}'
```

### 8.2 "왜 연결이 실패하는가?"

```bash
# 모든 TCP 연결 시도와 결과 추적
bpftrace -e '
kprobe:tcp_connect {
    @conn[tid] = nsecs;
    @dest[tid] = ntop(((struct sock *)arg0)->__sk_common.skc_daddr);
}
kretprobe:tcp_connect /@conn[tid]/ {
    $ret = retval;
    $latency_ms = (nsecs - @conn[tid]) / 1000000;
    if ($ret != 0) {
        printf("FAILED: %s → %s (err=%d, %dms)\n", comm, @dest[tid], $ret, $latency_ms);
    }
    delete(@conn[tid]);
    delete(@dest[tid]);
}'

# TCP 리셋 폭풍 확인
tcpretrans -l     # 손실 유형과 함께 표시 (재전송 vs 타임아웃)
```

---

## 9. 제한 사항 및 고려 사항

### 9.1 eBPF 제한 사항

| 제한 | 영향 | 우회 방법 |
|------|------|----------|
| **Linux 커널 4.14+ 필요** | macOS, Windows, 이전 커널 불가 | 비 Linux 플랫폼에서 OTel 사용 |
| **특권 또는 CAP_BPF 필요** | 공유 환경에서 보안 우려 | 전용 관측 가능성 노드 사용 |
| **애플리케이션 컨텍스트 없음** | 비즈니스 로직, 사용자 ID 볼 수 없음 | 앱 수준 데이터를 위해 OTel과 결합 |
| **스택 워킹(stack walking) 제한** | Go, Rust, JIT 언어의 복잡한 스택 | 언어별 프레임 포인터 설정 사용 |
| **높은 프로브 비율에서 오버헤드** | 모든 시스템 콜 트레이싱은 오버헤드 추가 | 프로브 샘플링 또는 필터링; 핫 패스 프로브 회피 |

---

## 10. 다음 단계

- [25_Continuous_Profiling.md](./25_Continuous_Profiling.md) -- 프로덕션 CPU 및 메모리 프로파일링
- [26_Incident_Response.md](./26_Incident_Response.md) -- 온콜 실무 및 인시던트 관리

---

## 연습 문제

### 연습 1: bpftrace 프로브 설계

다음 프로덕션 디버깅 시나리오 각각에 대해 bpftrace 원라이너를 작성하세요:

1. 서비스가 너무 많은 DNS 조회를 하고 있다는 의심이 있습니다. 프로세스 이름별로 그룹화된 초당 DNS 쿼리 수를 표시하세요.
2. 데이터베이스 집약적인 서비스에 지연 급증이 있습니다. 특정 PID의 모든 `write()` 시스콜을 추적하고 지연 히스토그램을 표시하세요.
3. 컨테이너가 디스크에 큰 파일을 기록하고 있다는 의심이 있습니다. 모든 `vfs_write` 호출을 추적하고 프로세스별 총 기록 바이트를 표시하세요.

<details>
<summary>정답 보기</summary>

**1. 프로세스별 DNS 쿼리 비율:**
```bash
bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:getaddrinfo {
  @dns_queries[comm] = count();
}
interval:s:1 {
  print(@dns_queries);
  clear(@dns_queries);
}'
```

**2. 특정 PID의 write 시스콜 지연 히스토그램:**
```bash
bpftrace -e '
tracepoint:syscalls:sys_enter_write /pid == 12345/ {
  @start[tid] = nsecs;
}
tracepoint:syscalls:sys_exit_write /pid == 12345 && @start[tid]/ {
  @write_latency_us = hist((nsecs - @start[tid]) / 1000);
  delete(@start[tid]);
}'
```

**3. vfs_write를 통한 프로세스별 기록 바이트:**
```bash
bpftrace -e 'kretprobe:vfs_write /retval > 0/ {
  @bytes_written[comm] = sum(retval);
}
interval:s:5 {
  print(@bytes_written);
  clear(@bytes_written);
}'
```

</details>

### 연습 2: eBPF vs OTel 결정

각 시나리오에서 eBPF 기반 관측 가능성, OpenTelemetry, 또는 둘 다 사용할지 결정하세요. 근거를 설명하세요.

1. 소스 코드 접근이 불가능한 레거시 Java 애플리케이션이 Kubernetes에서 실행 중이며 기본 HTTP 메트릭이 필요합니다.
2. 새로운 Python 마이크로서비스에 커스텀 속성을 포함한 상세한 비즈니스 트랜잭션 트레이싱이 필요합니다.
3. 과도한 DNS 조회를 유발하는 Kubernetes 파드를 식별해야 합니다.
4. Go 서비스에 가비지 컬렉션과 상관되는 간헐적 지연 급증이 있습니다.
5. 서비스 간 네트워크 정책을 시행하고 모니터링해야 합니다.

<details>
<summary>정답 보기</summary>

**1. 소스 코드 없는 레거시 Java 앱 → eBPF (Beyla 또는 Pixie)**
- 코드 변경이 불가능하므로, 배포를 수정(Java 에이전트 추가)할 수 없다면 OTel 자동 계측은 옵션이 아닙니다.
- eBPF(Beyla)는 애플리케이션을 건드리지 않고 커널에서 HTTP 요청 패턴을 관찰할 수 있습니다.
- 배포 커맨드 라인을 수정할 수 있다면(`-javaagent:opentelemetry-javaagent.jar` 추가), OTel 자동 계측이 더 좋습니다(더 풍부한 데이터).

**2. 커스텀 속성이 필요한 새 Python 마이크로서비스 → OpenTelemetry**
- 커스텀 비즈니스 속성(order_id, customer_tier, payment_method)은 애플리케이션 수준 계측이 필요합니다.
- eBPF는 이러한 애플리케이션 수준 개념을 볼 수 없습니다.
- OTel SDK로 비즈니스 핵심 경로에 수동 스팬을 사용하고 HTTP/DB에 자동 계측을 추가합니다.

**3. 파드별 과도한 DNS 조회 → eBPF (Cilium Hubble 또는 bpftrace)**
- DNS는 커널 수준 활동이므로 eBPF가 애플리케이션 변경 없이 관찰할 수 있습니다.
- Hubble은 파드별 DNS 쿼리 메트릭을 기본 제공합니다.
- OTel은 DNS 조회를 관찰할 수 없습니다(C 라이브러리에서 발생하며 애플리케이션 코드 아래).

**4. Go GC 관련 지연 급증 → 둘 다 사용**
- eBPF로 커널 수준 관찰: CPU 스케줄링, 메모리 할당 패턴.
- OTel로 애플리케이션 수준: GC 일시 정지에 영향받는 특정 요청 추적.
- Go 런타임은 OTel이 수집할 수 있는 GC 메트릭(`runtime/metrics`)을 노출합니다.
- bpftrace는 uprobe를 통해 Go GC 함수를 직접 추적하여 정확한 타이밍을 제공합니다.

**5. 네트워크 정책 시행 모니터링 → eBPF (Cilium Hubble / Tetragon)**
- 네트워크 정책은 Cilium의 eBPF 프로그램에 의해 커널 수준에서 시행됩니다.
- Hubble은 소스와 대상 컨텍스트와 함께 결과(ALLOWED/DROPPED)를 제공합니다.
- OTel은 네트워크 정책 시행에 대한 가시성이 없습니다.

</details>

---

## 참고 자료

- [eBPF.io -- 공식 eBPF 문서](https://ebpf.io/)
- [BPF Performance Tools (Brendan Gregg)](https://www.brendangregg.com/bpf-performance-tools-book.html)
- [bpftrace Reference Guide](https://github.com/bpftrace/bpftrace/blob/master/docs/reference_guide.md)
- [Cilium Hubble Documentation](https://docs.cilium.io/en/stable/observability/)
- [Grafana Beyla](https://grafana.com/docs/beyla/latest/)
- [Tetragon -- eBPF Security Observability](https://tetragon.io/)
