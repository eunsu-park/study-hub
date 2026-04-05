# 25. 지속적 프로파일링(Continuous Profiling)

**이전**: [eBPF 관측 가능성](./24_eBPF_Observability.md) | **다음**: [인시던트 대응](./26_Incident_Response.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 지속적 프로파일링을 설명하고 임시(ad-hoc) 프로파일링과의 차이를 구별할 수 있습니다
2. CPU, 메모리, Off-CPU 프로파일링을 위한 플레임 그래프(flame graph)를 읽고 해석할 수 있습니다
3. pprof를 사용하여 Go, Python, JVM 애플리케이션을 프로덕션에서 프로파일링할 수 있습니다
4. 대규모 지속적 프로파일링을 위해 Pyroscope 또는 Parca를 배포할 수 있습니다
5. 근본 원인 분석을 위해 프로파일링 데이터를 트레이스(trace) 및 메트릭(metric)과 상관시킬 수 있습니다
6. 프로파일링 인사이트를 적용하여 CPU 및 메모리 비용을 절감할 수 있습니다

---

메트릭은 *무엇이* 일어나고 있는지(CPU 사용률 80%) 알려줍니다. 트레이스는 *어디서* 일어나는지(order-service가 느림) 알려줍니다. 프로파일링은 코드 수준에서 *왜* 그런지 알려줍니다(`serializeJSON` 함수가 리플렉션(reflection)으로 인해 CPU의 40%를 소비). 지속적 프로파일링은 낮은 오버헤드로 프로덕션에서 항상 프로파일링을 실행하여, 필요하다는 것을 알기 전에 데이터를 캡처합니다.

> **비유 -- 의료 모니터링 vs 진단 검사**: 메트릭은 활력 징후(심박수, 혈압)처럼 항상 모니터링됩니다. 트레이스는 병원을 통한 환자의 여정(응급실 → 검사실 → 수술실)을 추적하는 것입니다. 프로파일링은 혈액 검사나 MRI처럼 세포 수준에서 신체 내부에서 무슨 일이 일어나고 있는지 보여줍니다. 지속적 프로파일링은 당뇨병 환자의 연속 혈당 모니터링 장치(CGM)를 착용하는 것처럼 24/7 상세 데이터를 캡처하여 이상이 발생했을 때 이미 히스토리를 가지고 있습니다.

## 1. 프로파일링 기초(Fundamentals)

### 1.1 프로파일 유형(Profile Types)

| 프로파일 유형 | 측정 대상 | 사용 시기 |
|-------------|---------|---------|
| **CPU** | CPU 시간을 소비하는 함수 | 높은 CPU 사용, 느린 응답 |
| **힙 (Alloc)** | 함수별 할당된 메모리 | 높은 메모리 사용, GC 압력 |
| **힙 (InUse)** | 함수별 현재 사용 중인 메모리 | 메모리 누수 |
| **고루틴(Goroutine)** (Go) | 스택별 고루틴 수 | 고루틴 누수, 데드락 |
| **뮤텍스(Mutex)** | 락 대기 시간 | 락 경합(contention) |
| **블록(Block)** | 동기화 프리미티브에서 블로킹 시간 | 채널 작업, I/O 대기 |
| **Off-CPU** | CPU에 있지 않은 시간 (I/O, 슬립, 락) | I/O 바운드 지연 |
| **벽시계(Wall clock)** | 총 경과 시간 (CPU + off-CPU) | 전체 함수 지속 시간 |

### 1.2 샘플링(Sampling) vs 인스트루먼테이션(Instrumentation)

| 접근법 | 작동 방식 | 오버헤드 | 정확도 |
|--------|---------|---------|--------|
| **샘플링** | 주기적으로 인터럽트하여 스택 트레이스 기록 | 매우 낮음 (~1-3%) | 통계적 (더 많은 샘플 = 더 높은 정확도) |
| **인스트루먼테이션** | 함수 진입/종료에 타이밍 코드 삽입 | 높음 (10-50%) | 정확 (모든 호출 측정) |

지속적 프로파일링은 프로덕션에서 오버헤드가 무시할 수 있어야 하므로 **샘플링**을 사용합니다.

### 1.3 CPU 샘플링 작동 방식

```
시간 →  |-----|-----|-----|-----|-----|-----|-----|-----|
         t1    t2    t3    t4    t5    t6    t7    t8

100 Hz (10ms마다)로 샘플링:

t1: main → handleRequest → serializeJSON → json.Marshal
t2: main → handleRequest → queryDB → postgres.Query
t3: main → handleRequest → serializeJSON → json.Marshal
t4: main → handleRequest → serializeJSON → reflect.Value.String
t5: main → handleRequest → queryDB → postgres.Query
t6: main → GC → runtime.mallocgc
t7: main → handleRequest → serializeJSON → json.Marshal
t8: main → handleRequest → serializeJSON → json.Marshal

결과:
  serializeJSON: 5/8 샘플 = 62.5% CPU
  queryDB:       2/8 샘플 = 25.0% CPU
  GC:            1/8 샘플 = 12.5% CPU
```

---

## 2. 플레임 그래프(Flame Graphs)

### 2.1 플레임 그래프 읽기

```
┌──────────────────────────────────────────────────────────┐
│ root                                                      │  100%
├──────────────────────────────────────┬───────────────────┤
│ handleRequest                        │ processQueue      │  70% / 30%
├────────────────────┬─────────────────┤                   │
│ serializeJSON      │ queryDB         │                   │  45% / 25%
├──────────┬─────────┤                 │                   │
│json.Marshal│reflect │ postgres.Query  │                   │  30% / 15%
└──────────┴─────────┴─────────────────┴───────────────────┘
```

**읽기 규칙:**
- **X축**: 너비 = 샘플의 비율 (시간이 아님). 넓을수록 더 많은 CPU.
- **Y축**: 스택 깊이. 하단 = 진입점, 상단 = 리프(leaf) 함수.
- **색상**: 보통 랜덤(패키지별 그룹핑) 또는 핫/콜드 표시.
- **상단의 넓은 바에 집중**: CPU가 실제로 소비되는 함수들.
- **하단의 넓은 바**는 단지 많은 것을 호출한다는 의미 (자체가 느린 것은 아님).

### 2.2 플레임 그래프 유형

| 유형 | X축 의미 | 하단 | 상단 |
|------|---------|------|------|
| **CPU 플레임 그래프** | CPU 시간 비율 | 진입점 (main) | 리프 함수 (핫 코드) |
| **Off-CPU 플레임 그래프** | 대기 시간 비율 | 진입점 | 블로킹 함수 (I/O, 락) |
| **메모리 플레임 그래프** | 할당 바이트 | 진입점 | 할당 사이트 |
| **차분 플레임 그래프** | 두 프로파일 간 변화 | 빨강 = 회귀, 파랑 = 개선 |

### 2.3 플레임 그래프 생성

```bash
# perf 데이터에서 (Linux)
perf record -F 99 -g -p $(pidof my-service) -- sleep 30
perf script | stackcollapse-perf.pl | flamegraph.pl > cpu-flame.svg

# Go pprof에서
go tool pprof -http=:6060 http://localhost:8080/debug/pprof/profile?seconds=30
# 브라우저에서 인터랙티브 플레임 그래프 열림

# Python py-spy에서
py-spy record -o profile.svg --pid $(pidof python3) --duration 30

# Java async-profiler에서
asprof -d 30 -f profile.html $(pidof java)
```

---

## 3. Go 프로파일링(pprof)

### 3.1 프로덕션에서 pprof 활성화

```go
package main

import (
    "net/http"
    _ "net/http/pprof"  // pprof 핸들러 등록
)

func main() {
    // pprof 엔드포인트가 /debug/pprof/에서 사용 가능
    // 프로덕션에서는 별도 포트에서 서빙 (공개 포트 아님)
    go func() {
        http.ListenAndServe("localhost:6060", nil)
    }()

    // ... 애플리케이션 코드 ...
}
```

### 3.2 pprof 엔드포인트

| 엔드포인트 | 프로파일 유형 | 사용법 |
|----------|-------------|-------|
| `/debug/pprof/profile?seconds=30` | CPU (30초) | `go tool pprof http://host:6060/debug/pprof/profile?seconds=30` |
| `/debug/pprof/heap` | 힙 (현재) | `go tool pprof http://host:6060/debug/pprof/heap` |
| `/debug/pprof/allocs` | 힙 (누적) | 시작 이후 모든 할당 표시 |
| `/debug/pprof/goroutine` | 고루틴 | `go tool pprof http://host:6060/debug/pprof/goroutine` |
| `/debug/pprof/mutex` | 뮤텍스 경합 | `runtime.SetMutexProfileFraction(5)` 필요 |
| `/debug/pprof/block` | 블록 (동기화) | `runtime.SetBlockProfileRate(1)` 필요 |

### 3.3 pprof 분석 워크플로우

```bash
# 1. CPU 프로파일 캡처
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=60

# 2. pprof 인터랙티브 모드에서:
(pprof) top20              # 상위 20 CPU 소비자
(pprof) top20 -cum         # 누적 시간 기준 상위 20
(pprof) list serializeJSON # 소스 수준 주석
(pprof) web                # 브라우저에서 플레임 그래프 열기
(pprof) peek queryDB       # 호출자와 피호출자 표시

# 3. 두 프로파일 비교 (최적화 전/후)
go tool pprof -base before.prof after.prof
(pprof) top20              # 차이 표시
```

### 3.4 메모리 프로파일링

```bash
# 힙 프로파일 캡처
go tool pprof http://localhost:6060/debug/pprof/heap

(pprof) top20 -inuse_space    # 현재 할당된 메모리
(pprof) top20 -alloc_space    # 시작 이후 총 할당 (GC 압력)
(pprof) top20 -alloc_objects  # 할당 횟수 (GC 트리거 비율)

# 메모리 누수 찾기: 수 분 간격의 두 힙 프로파일 비교
go tool pprof -base heap1.prof heap2.prof
(pprof) top20 -inuse_space    # 스냅샷 간 증가한 것 표시
```

---

## 4. Python 프로파일링

### 4.1 py-spy (샘플링 프로파일러)

```bash
# 실행 중인 Python 프로세스에 연결 (코드 변경 없음)
py-spy top --pid $(pidof python3)          # 실시간 top과 유사한 뷰
py-spy record -o profile.svg --pid $(pidof python3) --duration 30  # 플레임 그래프

# 특정 명령어 프로파일링
py-spy record -o profile.svg -- python3 app.py

# 서브프로세스 (포크 추적)
py-spy record -o profile.svg --subprocesses -- gunicorn app:app
```

### 4.2 cProfile과 Scalene

```python
# cProfile: 내장 결정적 프로파일러 (높은 오버헤드, 프로덕션 부적합)
import cProfile
cProfile.run('process_requests()', 'output.prof')

# snakeviz로 분석
# pip install snakeviz
# snakeviz output.prof  → 브라우저에서 플레임 그래프 열림
```

```bash
# Scalene: 낮은 오버헤드 CPU + 메모리 + GPU 프로파일러
pip install scalene
scalene --cpu --memory --reduced-profile app.py
```

### 4.3 memray를 사용한 메모리 프로파일링

```bash
# memray: Python을 위한 프로덕션급 메모리 프로파일러
pip install memray

# 실행 중인 프로세스에 연결
memray attach $(pidof python3)

# 시작부터 프로파일링
memray run app.py
memray flamegraph memray-output.bin -o memory.html

# 누수 추적 (해제되지 않은 할당 표시)
memray flamegraph memray-output.bin --leaks -o leaks.html
```

---

## 5. 지속적 프로파일링 플랫폼

### 5.1 Pyroscope

Pyroscope는 오픈소스 지속적 프로파일링 플랫폼입니다:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  App + Agent │  │  App + Agent │  │  App + Agent │
│  (SDK 또는   │  │  (SDK 또는   │  │  (SDK 또는   │
│   eBPF)      │  │   eBPF)      │  │   eBPF)      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌──────────▼───────────┐
              │   Pyroscope Server   │
              │  - 수집(Ingestion)   │
              │  - 저장(블록)        │
              │  - 쿼리 엔진         │
              │  - 플레임 그래프 UI   │
              └──────────────────────┘
```

### 5.2 Pyroscope 통합

```python
# Python: Pyroscope SDK
import pyroscope

pyroscope.configure(
    application_name="payment-service",
    server_address="http://pyroscope:4040",
    tags={
        "environment": "production",
        "region": "us-east",
    },
    # 특정 프로파일러 활성화
    enabled_profilers=[
        pyroscope.CpuProfiler,
        pyroscope.AllocProfiler,
        pyroscope.LockProfiler,
    ],
    # 샘플링 비율
    sample_rate=100,  # 100 Hz
)

# 필터링을 위한 특정 코드 경로에 태그 지정
with pyroscope.tag_wrapper({"endpoint": "/api/orders", "method": "POST"}):
    process_order(order)
```

```go
// Go: Pyroscope SDK
import "github.com/grafana/pyroscope-go"

func main() {
    pyroscope.Start(pyroscope.Config{
        ApplicationName: "payment-service",
        ServerAddress:   "http://pyroscope:4040",
        Tags:            map[string]string{"env": "production"},
        ProfileTypes: []pyroscope.ProfileType{
            pyroscope.ProfileCPU,
            pyroscope.ProfileAllocObjects,
            pyroscope.ProfileAllocSpace,
            pyroscope.ProfileInuseObjects,
            pyroscope.ProfileInuseSpace,
            pyroscope.ProfileGoroutines,
            pyroscope.ProfileMutexCount,
            pyroscope.ProfileMutexDuration,
            pyroscope.ProfileBlockCount,
            pyroscope.ProfileBlockDuration,
        },
    })
    defer pyroscope.Stop()
}
```

### 5.3 eBPF 기반 지속적 프로파일링

SDK 변경 없이 언어에 관계없는 프로파일링:

```yaml
# Pyroscope eBPF 에이전트 (DaemonSet)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pyroscope-ebpf
spec:
  template:
    spec:
      containers:
        - name: agent
          image: grafana/pyroscope:latest
          args:
            - "ebpf"
            - "--server-address=http://pyroscope:4040"
            - "--node=$(NODE_NAME)"
          securityContext:
            privileged: true
          env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
          volumeMounts:
            - name: modules
              mountPath: /lib/modules
            - name: debugfs
              mountPath: /sys/kernel/debug
      volumes:
        - name: modules
          hostPath:
            path: /lib/modules
        - name: debugfs
          hostPath:
            path: /sys/kernel/debug
```

---

## 6. 프로파일링-트레이스 상관 관계(Profiling-Trace Correlation)

### 6.1 프로파일을 트레이스에 연결

가장 강력한 디버깅 워크플로우: 느린 트레이스 스팬에서 해당 정확한 시간대의 CPU 프로파일로 이동합니다.

```
트레이스 (Tempo/Jaeger):
  order-service: POST /orders (2.5s)
    → createOrder (2.3s)
      → validateInventory (100ms)
      → calculateTotals (2.1s)  ← 왜 느린가?
          span.start: 14:00:00.100
          span.end: 14:00:02.200

프로파일 (Pyroscope):
  쿼리: service=order-service, from=14:00:00, to=14:00:02
  플레임 그래프 표시:
    calculateTotals → applyDiscountRules → regexp.Compile (85% CPU)
  → 근본 원인: 캐싱 대신 매 요청마다 정규식을 컴파일하고 있음
```

### 6.2 Grafana 통합

```
Grafana Tempo → Pyroscope 연결:
  1. Tempo 데이터 소스 설정에서:
     - "Traces to Profiles" 활성화
     - Pyroscope 데이터 소스에 연결
     - service.name 레이블로 매칭

  2. 트레이스 조회 시:
     - 스팬 클릭
     - "View Profile" 버튼 클릭
     - 해당 시간 범위와 서비스의 Pyroscope 플레임 그래프 열림

  3. 프로파일 비교:
     - 기준선 시간 범위 선택 (인시던트 전)
     - 인시던트 시간 범위 선택
     - 차분 플레임 그래프 보기 (빨강 = 회귀)
```

---

## 7. 프로파일링을 통한 비용 최적화

### 7.1 CPU 낭비 식별

```
프로파일링 최적화 전:
  payment-service: 8 파드 × 2 CPU = 16 CPU 코어

프로파일링 분석 후:
  - json.Marshal 리플렉션 사용: 35% CPU → jsoniter로 전환: 12% CPU
  - 요청당 regexp.Compile: 20% CPU → 컴파일된 정규식 캐시: 0.1% CPU
  - 요청당 TLS 핸드셰이크: 15% CPU → 커넥션 풀링: 2% CPU
  - 총 CPU 감소: 70% → 30% = 57% 감소

최적화 후:
  payment-service: 4 파드 × 2 CPU = 8 CPU 코어
  절약: 8 CPU 코어 × $0.05/hr × 720 hr/mo = $288/mo (서비스당)
```

### 7.2 메모리 최적화

```
프로파일링 결과:
  - 루프 내 문자열 연결: 500MB/분 할당 (GC 압력)
    수정: strings.Builder 사용 → 5MB/분
  - 함수 인수에서 큰 구조체 복사: 200MB 사용 중
    수정: 포인터 전달 → 50MB 사용 중
  - 무제한 캐시: 24시간에 걸쳐 2GB까지 증가
    수정: 최대 크기가 있는 LRU 캐시 → 256MB에서 안정

결과: 파드당 메모리 요청이 4Gi에서 1Gi로 감소
  절약: 3Gi × 8 파드 × $0.004/GiB/hr × 720 hr/mo = $69/mo
```

---

## 8. 모범 사례(Best Practices)

### 8.1 프로덕션 프로파일링 체크리스트

| 실천 사항 | 이유 |
|----------|------|
| 샘플링 프로파일러 사용 (인스트루먼팅 아님) | 오버헤드를 2% 미만으로 유지 |
| 인시던트 때만이 아니라 지속적으로 프로파일링 | 비교를 위한 기준선 데이터 확보 |
| CPU 샘플 비율을 100 Hz로 설정 | 최소 오버헤드로 좋은 정확도 |
| 사용 중 메모리뿐 아니라 메모리 할당 비율(allocs) 프로파일링 | GC 압력 파악 |
| 서비스 및 환경 태그 추가 | 멀티 서비스 배포에서 프로파일 필터링 |
| 컨텍스트를 위해 트레이스와 통합 | 느린 스팬에서 코드 수준 프로파일로 이동 |
| 인시던트 때뿐 아니라 주간 프로파일 검토 | 점진적 회귀 발견 |
| 전/후 비교에 차분 플레임 그래프 사용 | 최적화를 객관적으로 검증 |

---

## 9. 다음 단계

- [26_Incident_Response.md](./26_Incident_Response.md) -- 온콜 실무 및 인시던트 관리
- [27_AIOps_Anomaly_Detection.md](./27_AIOps_Anomaly_Detection.md) -- ML 기반 이상 탐지

---

## 연습 문제

### 연습 1: 플레임 그래프 분석

다음 플레임 그래프 데이터(CPU 샘플 수)가 주어졌습니다:

```
main → handleRequest → serializeJSON → json.Marshal → reflect.Value.String: 350
main → handleRequest → serializeJSON → json.Marshal → reflect.Value.Int: 150
main → handleRequest → queryDB → sql.Query → pgx.conn.exec: 200
main → handleRequest → queryDB → sql.Rows.Scan: 50
main → handleRequest → authenticate → bcrypt.CompareHashAndPassword: 180
main → handleRequest → compress → gzip.Writer.Write: 70
총 샘플: 1000
```

답하세요: (a) `serializeJSON`이 소비하는 CPU 비율은? (b) 가장 영향력 있는 단일 최적화는? (c) `json.Marshal`을 5배 빠른 코드 생성 직렬화기로 교체하면 새로운 총 CPU 소비는 어떻게 되는가?

<details>
<summary>정답 보기</summary>

**(a) serializeJSON CPU 비율:**
```
serializeJSON 샘플 = 350 + 150 = 500
비율 = 500 / 1000 = 50%
```

**(b) 가장 영향력 있는 최적화:**
`json.Marshal`을 통한 `serializeJSON` (CPU의 50%). 구체적으로, `reflect.Value.String` (35%)이 지배적인 리프 함수입니다. 표준 라이브러리 `json.Marshal`은 직렬화에 리플렉션을 사용하며, 이는 CPU 집약적입니다.

**최적화**: `encoding/json`을 리플렉션을 피하는 `easyjson`, `jsoniter`, `sonic` 같은 코드 생성 직렬화기로 교체합니다. 예상 속도 향상: JSON 직렬화 3-10배.

**(c) 5배 빠른 JSON 직렬화 후:**
```
전: serializeJSON = 500 샘플 (50%)
후: serializeJSON = 500 / 5 = 100 샘플

새 총 = 1000 - 500 + 100 = 600 샘플
새 분포:
  serializeJSON:  100/600 = 16.7% (이전 50%)
  queryDB:        250/600 = 41.7% (이전 25%)
  authenticate:   180/600 = 30.0% (이전 18%)
  compress:        70/600 = 11.7% (이전 7%)

전체: 600/1000 = 총 CPU 40% 감소.
동일한 트래픽을 ~40% 적은 CPU 코어로 처리할 수 있습니다.
```

</details>

### 연습 2: 메모리 누수 탐지

Go 서비스의 메모리 사용량이 24시간에 걸쳐 500MB에서 4GB로 증가한 후 OOM 킬됩니다. pprof를 사용하여 누수를 식별하는 단계별 프로세스를 설명하세요. 구체적인 명령어, pprof 쿼리, 출력에서 찾아볼 패턴을 포함하세요.

<details>
<summary>정답 보기</summary>

**1단계: 기준선 힙 프로파일 캡처**
```bash
# 서비스 재시작 직후 (500MB)
curl -o heap_baseline.prof http://service:6060/debug/pprof/heap
```

**2단계: 대기 후 두 번째 프로파일 캡처**
```bash
# 2-4시간 후 (선형 증가라면 ~1-2GB)
curl -o heap_after4h.prof http://service:6060/debug/pprof/heap
```

**3단계: 프로파일 비교 (차분 분석)**
```bash
go tool pprof -base heap_baseline.prof heap_after4h.prof

(pprof) top20 -inuse_space
# 스냅샷 간 사용 중 메모리가 증가한 함수 표시
# 상위 항목이 누수 원인일 가능성이 높음

# 예상 출력 (예시):
# 1.2GB  leakyCache.Store      (제거하지 않는 캐시)
# 200MB  bufPool.Get           (풀에 반환되지 않는 버퍼)
```

**4단계: 누수 원인 상세 조사**
```bash
(pprof) list leakyCache.Store
# 줄별 메모리 주석이 있는 소스 코드 표시
# 45줄:   cache[key] = largeStruct   ← 여기서 1.2GB 할당

(pprof) peek leakyCache.Store
# 호출자 표시: 누가 이 함수를 호출하는가?
# handleRequest → processData → leakyCache.Store
```

**5단계: 할당 비율 확인 (GC 압력)**
```bash
go tool pprof http://service:6060/debug/pprof/allocs

(pprof) top20 -alloc_space
# 시작 이후 누적 할당 표시
# 높은 할당 비율 + 증가하는 inuse = 누수
# 높은 할당 비율 + 안정적 inuse = 단순 GC 압력 (누수 아님)
```

**6단계: 고루틴 수 확인 (고루틴 누수)**
```bash
go tool pprof http://service:6060/debug/pprof/goroutine

(pprof) top20
# 고루틴 수가 시간이 지남에 따라 증가하면 고루틴 누수
# 각 고루틴은 ~2-8KB 스택 + 참조하는 데이터를 보유
```

**누수를 나타내는 패턴:**
- 함수의 `inuse_space`가 시간에 따라 선형으로 증가
- 제한 없이 커지는 맵(map) 또는 슬라이스(slice) (제거/정리 없음)
- 시작하지만 완료되지 않는 고루틴 (채널 또는 I/O에서 블로킹)
- `sync.Pool`을 통해 할당되지만 반환되지 않는 버퍼 (잘못된 풀 사용)
- 큰 객체에 대한 참조를 보유하는 전역 변수

</details>

---

## 참고 자료

- [Pyroscope Documentation](https://pyroscope.io/docs/)
- [Go pprof Documentation](https://pkg.go.dev/net/http/pprof)
- [Brendan Gregg -- Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [py-spy -- Sampling Profiler for Python](https://github.com/benfred/py-spy)
- [Grafana Tempo -- Traces to Profiles](https://grafana.com/docs/tempo/latest/)
- [Parca -- Continuous Profiling](https://www.parca.dev/)
