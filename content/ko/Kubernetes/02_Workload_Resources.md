# 02. 워크로드 리소스

**이전**: [아키텍처 심층 분석](./01_Architecture_Deep_Dive.md) | **다음**: [네트워킹 기초](./03_Networking_Fundamentals.md)

## 학습 목표
- 파드(Pod) 라이프사이클, 초기화 컨테이너(Init Containers), 멀티컨테이너 패턴을 숙달한다
- 레플리카셋(ReplicaSets), 디플로이먼트(Deployments), 스테이트풀셋(StatefulSets)을 배포하고 관리한다
- 롤링 업데이트(Rolling Updates), 롤백(Rollbacks), 배포 전략을 구성한다
- 배치 워크로드를 위해 잡(Jobs)과 크론잡(CronJobs)을 사용한다
- 요청(Requests), 제한(Limits), QoS 클래스를 포함한 리소스 관리를 이해한다

---

워크로드 리소스는 쿠버네티스에서 애플리케이션 코드를 실행하는 구성 요소입니다.
원자적인 파드(Pod)에서 정교한 스테이트풀셋(StatefulSets)까지, 각 리소스 타입은
특정 운영 패턴을 위해 설계되었습니다. 이 레슨에서는 모든 워크로드 리소스를
프로덕션에 바로 사용할 수 있는 예제와 함께 상세히 다룹니다.

워크로드 투어에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 각 워크로드 종류가 동일한 Pod 프리미티브 위에서 서로 다른 reconciler로 존재하는 이유, Deployment의 롤링 업데이트 수식, StatefulSet의 순서·안정 네트워크 식별 보장, requests/limits/QoS가 리눅스 커널 cgroup 노브로 어떻게 매핑되는지를 다룹니다.

## 목차
0. [이론과 원리](#이론과-원리)
1. [파드](#1-파드)
2. [레플리카셋](#2-레플리카셋)
3. [디플로이먼트](#3-디플로이먼트)
4. [스테이트풀셋](#4-스테이트풀셋)
5. [데몬셋](#5-데몬셋)
6. [잡과 크론잡](#6-잡과-크론잡)
7. [파드 중단 예산](#7-파드-중단-예산)
8. [리소스 요청과 제한](#8-리소스-요청과-제한)
9. [QoS 클래스](#9-qos-클래스)
10. [연습문제](#연습문제)

---

## 이론과 원리

워크로드 리소스는 긴 메뉴처럼 보입니다 — Pod, ReplicaSet, Deployment, StatefulSet, DaemonSet, Job, CronJob — 그러나 모두 하나의 프리미티브(Pod)와 하나의 설계 패턴(원하는 상태 객체를 watch하고 자식 Pod 집합을 조정하는 컨트롤러) 위에 세워져 있습니다. 차이는 *조정 정책(reconciliation policy)*에 있습니다: 레플리카 수, 순서, 식별 보장, 실패 시 대응. 이 섹션은 모든 워크로드 종류를 설명하는 네 가지 아이디어와, 메모리 압박 상황에서 파드가 죽을지를 결정하는 리소스 관리 시맨틱을 다룹니다.

### A. 원자적 스케줄링 단위로서의 Pod

Pod는 **운명을 공유(shared fate)**하는 하나 이상의 컨테이너입니다 — 같은 노드, 같은 네트워크 네임스페이스(같은 IP, 같은 `localhost`, 같은 포트 공간), 선택적 공유 볼륨, 함께 스케줄, 함께 종료. Pod(컨테이너가 아닌)를 원자 단위로 정한 결정은 단일 컨테이너로는 깔끔히 모델링할 수 없는 세 가지 패턴을 가능하게 합니다:

- **사이드카(Sidecar)**: 메인 앱과 볼륨·localhost를 공유해야 하지만, 독립적으로 업그레이드되어야 하는 로깅·프록시 컨테이너.
- **초기화 컨테이너(Init container)**: 메인 컨테이너가 시작되기 전에 완료되어야 하는 순차적 준비 단계(DB 마이그레이션, config fetch). 실패하면 시작이 차단됨.
- **어댑터(Adapter)**: localhost를 통해 메인 앱의 메트릭이나 API를 다른 프로토콜로 노출하는 변환 컨테이너.

흔한 단일 컨테이너 케이스에서는 Pod의 오버헤드가 사실상 0입니다. 핵심적으로, **Pod는 일시적이며 IP도 일시적입니다.** 반려동물(pet)이 아닙니다. 재시작 시 IP가 바뀌고, liveness probe에 실패한 Pod는 새 IP를 가진 새 Pod로 대체됩니다. 안정적 네트워크 식별이 필요한 모든 것은 Service(또는 Pod별 식별이 필요하면 Headless Service + StatefulSet)를 거쳐야 합니다.

이 일시성이야말로 모든 상위 워크로드를 가능하게 합니다 — Pod가 소중한 존재였다면 안전하게 롤링·스케일링·축출할 수 없을 것입니다. Pod가 일회용이기에, 위 컨트롤러는 언제든 더 만들 수 있습니다.

### B. ReplicaSet → Deployment — 롤링 업데이트 알고리즘

**ReplicaSet**은 Pod 위의 가장 단순한 reconciler입니다: "이 셀렉터에 매치되는 N개의 Pod를 원한다. 지금 M개가 보인다. M < N이면 생성, M > N이면 삭제." 이 전체가 알고리즘이며, 레벨 트리거 루프로 표현됩니다. ReplicaSet은 스케일링과 자가 치유를 처리하지만, 템플릿 변경은 처리하지 *않습니다* — 파드 템플릿을 수정해도 기존 Pod는 갱신되지 않습니다.

**Deployment**는 ReplicaSet을 감싸고(롤아웃 중에는 실제로 두 개를) 롤링 업데이트 알고리즘을 추가합니다. 새 파드 템플릿이 주어지면:

1. 새 ReplicaSet RS-new를 `replicas=0`과 새 템플릿으로 생성.
2. RS-new가 목표 수에 도달하고 RS-old가 0이 될 때까지 반복:
   - RS-new를 `maxSurge`(기본 25%)만큼 스케일 업.
   - 새 파드가 Ready가 될 때까지 대기.
   - RS-old를 `maxUnavailable`(기본 25%)만큼 스케일 다운.
3. 원커맨드 롤백을 위해 RS-old를 `replicas=0`으로 보존.

수식: `replicas=10`, `maxSurge=25%`, `maxUnavailable=25%`이면 어느 순간 최대 13개의 파드가 존재할 수 있고, 최소 7개는 Ready여야 합니다. 이는 롤아웃 중 추가 비용(서지)과 용량 손실(불가용)을 모두 제한합니다. `recreate` 전략은 이를 건너뛰고 모두 죽인 뒤 재생성합니다 — 두 버전이 공존할 수 없을 때 사용(예: 스키마 마이그레이션).

롤백은 단순히 "디플로이먼트 템플릿을 이전 ReplicaSet의 pod-template-hash로 되돌리기"입니다. 디플로이먼트 컨트롤러는 동일한 알고리즘을 역방향으로 실행하여 RS-old를 다시 스케일 업하고 RS-new를 스케일 다운합니다. 이 때문에 보존된 이력(`revisionHistoryLimit`)이 중요합니다.

### C. StatefulSet — 순서, 식별, 스토리지

StatefulSet은 일부 워크로드(데이터베이스, 메시지 브로커, 분산 합의 시스템)가 **레플리카별 안정 식별(stable identity per replica)**을 필요로 하기 때문에 존재합니다. Deployment와 다른 세 가지 보장:

- **순서 있는, 안정적 네트워크 식별.** Pod는 `<set>-0`, `<set>-1`, ..., `<set>-(N-1)`로 명명됩니다. Headless Service와 함께 각 파드는 `mysql-0.mysql.default.svc.cluster.local` 같은 DNS 이름을 받고, 재시작·재스케줄을 거쳐도 유지됩니다.
- **순서 있는 배포와 종료.** 파드는 한 번에 하나씩 순서대로(`-0`, 그다음 `-1`, 그다음 `-2`) 올라오며, 각각 이전 파드가 Ready가 될 때까지 대기합니다. 종료는 역순. 클러스터 부트스트랩에 중요합니다(예: 첫 레플리카가 시드, 1번 레플리카는 0번의 안정 DNS 이름을 참조해 합류).
- **VolumeClaimTemplate을 통한 안정적·레플리카별 스토리지.** 각 파드는 자신의 PVC를 가지며, 이름은 파드 ordinal에서 파생됩니다. `mysql-1`이 재스케줄되면 동일한 PVC에 다시 attach됩니다 — 동일한 데이터. Deployment는 "어느 파드가 누구인지" 개념이 없으므로 이를 안전히 할 수 없습니다.

이 보장의 비용은 민첩성 감소입니다 — StatefulSet을 3에서 6으로 스케일하는 것은 파드가 순차적으로 올라오기 때문에 Deployment보다 오래 걸립니다. 업데이트는 `RollingUpdate`(한 번에 한 파드, 역 ordinal) 또는 `OnDelete`(자동화하기에 너무 위험할 때 운영자 주도) 중 하나를 사용합니다.

### D. DaemonSet, Job, CronJob — 특화된 reconciler들

각각은 동일한 컨트롤러 패턴을 "원하는 상태가 무엇을 의미하는가?"만 달리한 것입니다:

- **DaemonSet**: "매치되는 노드마다 Pod 하나." Reconciler는 DaemonSet과 Node 목록을 모두 watch하며, 노드 셀렉터에 매치되는 새 노드마다 Pod를 생성하고 드레인되거나 삭제된 노드에서 Pod를 제거합니다. 로그 수집기, node exporter, CNI 에이전트 등 클러스터 전역에서 실행되어야 하는 모든 것에 사용됩니다.
- **Job**: "최대 P개를 병렬로 실행하면서 N번의 성공적 완료를 달성하라." Reconciler는 Pod를 생성하고 완료를 watch하며, 성공 횟수에 도달하면 멈춥니다. `backoffLimit`이 실패 시 재시도를 제한하고, `activeDeadlineSeconds`가 총 벽시계 시간을 제한합니다. 배치 워크로드에 핵심적입니다.
- **CronJob**: Job 팩토리. Reconciler는 crontab 스타일 스케줄을 읽고, 발화 시점마다 템플릿에서 새 Job을 생성합니다. `concurrencyPolicy`가 이전 Job이 아직 실행 중일 때의 동작을 결정합니다(`Allow`, `Forbid`, `Replace`).

당신은 Pod 템플릿을 작성하고, 컨트롤러는 몇 개를·어떤 순서로·어느 노드에·어떤 성공 기준으로 실행할지 결정합니다.

### E. Requests, Limits, QoS — YAML에서 cgroup으로

리소스 관리에는 컨테이너당 리소스(CPU, 메모리)별로 두 숫자가 있습니다:

- **`requests`**: 스케줄러가 파드의 예약(reservation)으로 취급하는 값. 노드 위 모든 파드 requests의 합은 노드 allocatable을 초과할 수 없습니다. 파드가 노드에 *적합한지* 여부를 결정합니다.
- **`limits`**: 커널이 강제하는 상한. 한도를 넘는 CPU는 스로틀되고(cgroup `cpu.cfs_quota_us`), 한도를 넘는 메모리는 OOM kill을 트리거합니다(cgroup `memory.limit_in_bytes`).

requests/limits 조합으로부터 세 가지 QoS 클래스가 자동으로 결정됩니다:

| 클래스 | 조건 | 축출 우선순위 |
|-------|------|--------------|
| `Guaranteed` | 모든 컨테이너가 CPU·메모리에 대해 `requests == limits` | 가장 마지막에 축출 |
| `Burstable` | 적어도 하나의 request 설정, 그러나 모두 `requests == limits`은 아님 | 중간 |
| `BestEffort` | 어디에도 requests나 limits 미설정 | 가장 먼저 축출 |

노드 메모리 압박 시, kubelet은 BestEffort를 먼저 축출하고, 그다음 Burstable을 자신의 request 초과 정도 순으로 축출하며, Guaranteed는 정말 필요한 경우에만 축출합니다. 따라서 **`Guaranteed`는 단순한 리소스 예산이 아니라 축출 보험입니다.**

CPU와 메모리는 결정적으로 다릅니다 — CPU는 **압축 가능(compressible)**(느린 테넌트를 스로틀하면 모두가 계속 동작), 메모리는 **압축 불가능(incompressible)**(다 쓰면 누군가는 죽어야 함). 이 때문에 메모리 OOM은 갑작스럽고 치명적인 반면, CPU 스로틀은 점진적이고 회복 가능합니다.

### 이론에서 아래의 YAML으로

이제 레슨은 이러한 추상을 구체적인 매니페스트로 안내합니다:

- **섹션 1 (파드)**는 §A의 원자 단위를 시연합니다 — 멀티컨테이너 패턴, 초기화 컨테이너, 사이드카, 라이프사이클 훅.
- **섹션 2 (ReplicaSet)**은 Deployment가 인계받기 전에 §B의 가장 단순한 reconciler를 단독으로 보여줍니다.
- **섹션 3 (Deployment)**는 실제 서비스에 적용된 §B의 롤링 업데이트 알고리즘이며, `maxSurge`/`maxUnavailable` 노브를 조정합니다.
- **섹션 4 (StatefulSet)**은 §C이며, 파드별 식별을 부여하는 VolumeClaimTemplate과 Headless Service를 다룹니다.
- **섹션 5–6 (DaemonSet, Job, CronJob)**은 §D의 특화 reconciler들입니다.
- **섹션 7 (PDB)**는 롤링 업데이트 알고리즘에 대한 가드레일입니다 — "자발적 중단(voluntary disruption) 중에도 동시에 X개 이상 다운시키지 마라."
- **섹션 8–9 (Requests/Limits, QoS)**는 §E를 kubelet이 cgroup에 전달하는 YAML 필드로 옮깁니다.

Pod-as-atom + reconciler-as-pattern을 보고 나면, DaemonSet과 Job의 차이는 의사코드 두 줄에 불과합니다.

---

## 1. 파드

### 1.1 파드 기본 사항

파드(Pod)는 쿠버네티스에서 가장 작은 배포 가능한 단위입니다. 네트워크와 스토리지를
공유하는 하나 이상의 컨테이너와 이들을 실행하는 방법에 대한 사양을 캡슐화합니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: simple-pod
  labels:
    app: demo
    version: v1
spec:
  containers:
    - name: web
      image: nginx:1.25
      ports:
        - containerPort: 80
          name: http
          protocol: TCP
```

주요 특성:
- 파드 내 모든 컨테이너는 동일한 네트워크 네임스페이스를 공유합니다 (localhost)
- 모든 컨테이너는 동일한 IPC 네임스페이스를 공유합니다
- 볼륨은 컨테이너 간에 공유할 수 있습니다
- 파드는 일시적(Ephemeral)입니다 — "수리"되지 않고 교체만 됩니다

### 1.2 파드 라이프사이클

```
Pending → Running → Succeeded / Failed
              │
              └──→ Unknown (노드와의 연결 끊김)
```

| 단계 | 설명 |
|-------|-------------|
| Pending | 클러스터에서 수락되었으나 컨테이너가 아직 실행되지 않음 |
| Running | 적어도 하나의 컨테이너가 실행 중 |
| Succeeded | 모든 컨테이너가 종료 코드 0으로 종료됨 |
| Failed | 적어도 하나의 컨테이너가 0이 아닌 종료 코드로 종료됨 |
| Unknown | 노드 통신 실패 |

```bash
# 파드 단계 전환 감시
kubectl get pod simple-pod -w

# 상세 단계 및 조건 정보
kubectl get pod simple-pod -o jsonpath='{.status.phase}'
kubectl get pod simple-pod -o jsonpath='{.status.conditions}' | python3 -m json.tool
```

### 1.3 컨테이너 상태

파드 내 각 컨테이너는 자체 상태를 가집니다:

| 상태 | 설명 |
|-------|-------------|
| Waiting | 컨테이너가 아직 실행되지 않음 (이미지 풀링 등) |
| Running | 컨테이너가 실행 중 |
| Terminated | 컨테이너가 실행을 완료함 |

### 1.4 초기화 컨테이너(Init Containers)

초기화 컨테이너는 앱 컨테이너가 시작되기 전에 순차적으로 실행됩니다. 데이터베이스
마이그레이션, 설정 생성, 종속 서비스 대기 등의 설정 작업에 유용합니다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: init-demo
spec:
  initContainers:
    # 초기화 컨테이너 1: 서비스 가용성 대기
    - name: wait-for-db
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Waiting for database..."
          until nslookup postgres-service.default.svc.cluster.local; do
            echo "Database not ready, retrying in 2s..."
            sleep 2
          done
          echo "Database is available"

    # 초기화 컨테이너 2: 데이터베이스 마이그레이션 실행
    - name: run-migration
      image: my-app:v1.0
      command: ["./migrate", "--target=latest"]
      env:
        - name: DB_HOST
          value: "postgres-service"

  containers:
    - name: app
      image: my-app:v1.0
      ports:
        - containerPort: 8080
```

초기화 컨테이너의 속성:
- 한 번에 하나씩 순서대로 실행
- 각각이 성공적으로 완료되어야 다음이 시작됨
- 초기화 컨테이너가 실패하면 kubelet이 재시작함 (`restartPolicy`에 따라)
- `livenessProbe`, `readinessProbe`, `startupProbe`를 지원하지 않음
- 앱 컨테이너와 다른 리소스 제한을 가질 수 있음

### 1.4.1 네이티브 사이드카 컨테이너(Native Sidecar Containers) (Kubernetes 1.29+)

Kubernetes 1.29에서 `initContainers`에 `restartPolicy: Always`를 지정하는 네이티브 사이드카 지원이 도입되었습니다. 일반 초기화 컨테이너와 달리, 네이티브 사이드카는 앱 컨테이너보다 먼저 시작되어 파드 수명 내내 계속 실행됩니다 — 메인 앱 컨테이너보다 오래 살아야 하는 로그 전송기나 서비스 메시 프록시의 문제를 해결합니다.

```yaml
spec:
  initContainers:
    - name: log-collector          # 네이티브 사이드카: 앱 컨테이너와 함께 실행
      image: fluent/fluent-bit:2.2
      restartPolicy: Always        # 핵심 필드 — 사이드카로 만드는 것, 일회성 init이 아님
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app
          readOnly: true
    - name: run-migration          # 일반 초기화 컨테이너 (restartPolicy 없음)
      image: my-app:v1.0
      command: ["./migrate", "--target=latest"]
  containers:
    - name: app
      image: my-app:v1.0
```

순서: 네이티브 사이드카는 그 앞에 있는 일반 초기화 컨테이너가 완료된 후 시작되며, 이후의 초기화 컨테이너나 앱 컨테이너가 시작되기 전에 Running 상태가 보장됩니다.

### 1.5 멀티컨테이너 패턴

#### 사이드카 패턴(Sidecar Pattern)

헬퍼 컨테이너가 메인 애플리케이션을 보강합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sidecar-logging
spec:
  volumes:
    - name: shared-logs
      emptyDir: {}
  containers:
    # 메인 애플리케이션
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app

    # 사이드카: 로그 전송기
    - name: log-shipper
      image: fluent/fluent-bit:2.2
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app
          readOnly: true
      env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc"
```

#### 앰배서더 패턴(Ambassador Pattern)

프록시 컨테이너가 아웃바운드 연결을 처리합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ambassador-pattern
spec:
  containers:
    - name: app
      image: my-app:v1.0
      env:
        # 앱이 localhost에 연결, 앰배서더가 실제 서비스로 프록시
        - name: DB_HOST
          value: "localhost"
        - name: DB_PORT
          value: "5432"

    - name: ambassador
      image: haproxy:2.9
      ports:
        - containerPort: 5432
      volumeMounts:
        - name: haproxy-config
          mountPath: /usr/local/etc/haproxy
  volumes:
    - name: haproxy-config
      configMap:
        name: ambassador-haproxy-config
```

#### 어댑터 패턴(Adapter Pattern)

컨테이너가 메인 컨테이너의 출력을 변환합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: adapter-pattern
spec:
  volumes:
    - name: shared-data
      emptyDir: {}
  containers:
    # 메인 앱이 커스텀 형식으로 메트릭 작성
    - name: app
      image: legacy-app:v2.0
      volumeMounts:
        - name: shared-data
          mountPath: /metrics

    # 어댑터가 메트릭을 Prometheus 형식으로 변환
    - name: prometheus-adapter
      image: prom/statsd-exporter:v0.26.0
      ports:
        - containerPort: 9102
          name: metrics
      volumeMounts:
        - name: shared-data
          mountPath: /metrics
          readOnly: true
```

### 1.6 프로브(Probes)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: probes-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      ports:
        - containerPort: 8080

      # 시작 프로브(Startup Probe): 느리게 시작하는 앱
      startupProbe:
        httpGet:
          path: /healthz
          port: 8080
        failureThreshold: 30
        periodSeconds: 10
        # 시작에 최대 300초 (30 * 10) 허용

      # 활성 프로브(Liveness Probe): 비정상이면 재시작
      livenessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 0
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3

      # 준비 프로브(Readiness Probe): 준비되지 않으면 서비스에서 제거
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 5
        failureThreshold: 1
```

| 프로브 유형 | 실패 시 동작 | 사용 사례 |
|-----------|-------------------|----------|
| 시작(Startup) | 재시도 계속 | 느리게 시작하는 앱 (DB 마이그레이션) |
| 활성(Liveness) | 컨테이너 재시작 | 데드락 감지 |
| 준비(Readiness) | 서비스 엔드포인트에서 제거 | 일시적 사용 불가 |

### 1.7 임시 컨테이너(Ephemeral Containers, kubectl debug)

임시 컨테이너(Ephemeral Container)는 디버깅을 위해 실행 중인 파드에 주입되는 임시 컨테이너입니다. 메인 컨테이너 이미지가 최소화(minimal)되어 있거나 디스트로리스(distroless)여서 셸이나 디버깅 도구가 없을 때 유용합니다.

```bash
# 실행 중인 파드에 디버그 컨테이너 주입 (파드의 네임스페이스를 공유)
kubectl debug -it my-pod --image=busybox:1.36 --target=app

# 더 많은 도구를 위한 풍부한 이미지 사용
kubectl debug -it my-pod --image=nicolaka/netshoot --target=app

# 파드 복사 (더 깊은 디버깅을 위해 수정된 스펙으로 새 파드 생성)
kubectl debug my-pod --copy-to=debug-pod --set-image=app=busybox:1.36 -it
```

임시 컨테이너의 주요 특성:
- 재시작할 수 없으며, 파드가 삭제되거나 재시작되면 제거됨
- 프로브(Probe)나 리소스 요청/제한을 가질 수 없음
- `--target`이 지정되면 대상 컨테이너의 프로세스 네임스페이스를 공유함
- 셸이 없는 디스트로리스 이미지(예: `gcr.io/distroless/static`) 디버깅에 이상적

---

## 2. 레플리카셋

레플리카셋(ReplicaSet)은 지정된 수의 파드 레플리카가 항상 실행되도록 보장합니다.

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: web-rs
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
      tier: frontend
  template:
    metadata:
      labels:
        app: web
        tier: frontend
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "100m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
```

```bash
# 레플리카셋 생성
kubectl apply -f web-rs.yaml

# 상태 확인
kubectl get rs web-rs

# 수동 스케일링
kubectl scale rs web-rs --replicas=5

# 이 RS에 속하는 파드 확인
kubectl get pods -l app=web,tier=frontend --show-labels
```

> **참고**: 레플리카셋을 직접 생성하는 경우는 드뭅니다. 대신 디플로이먼트(Deployments)를
> 사용하세요 — 레플리카셋을 관리해주며 롤아웃/롤백 기능을 추가합니다.

---

## 3. 디플로이먼트

디플로이먼트는 레플리카셋을 관리하고, 레플리카셋은 파드를 관리합니다. 롤링 업데이트를 수행하면 디플로이먼트가 새 레플리카셋을 생성하고 이전 RS에서 새 RS로 파드를 점진적으로 이동시킵니다. 0으로 스케일된 이전 레플리카셋은 `revisionHistoryLimit`에 의해 제어되는 롤백을 위해 보관됩니다.

```
Deployment
├── ReplicaSet (v2 — 현재, replicas=3)
│   ├── Pod web-deploy-6d7f9b-abc
│   ├── Pod web-deploy-6d7f9b-def
│   └── Pod web-deploy-6d7f9b-ghi
└── ReplicaSet (v1 — 이전, replicas=0)  ← 롤백을 위해 보관
    └── (파드 없음)
```

### 3.1 기본 디플로이먼트

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deploy
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          readinessProbe:
            httpGet:
              path: /
              port: 80
            periodSeconds: 5
          resources:
            requests:
              cpu: "100m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
```

### 3.2 롤링 업데이트 전략(Rolling Update Strategy)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deploy
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 2     # 업데이트 중 최대 2개 파드가 사용 불가능할 수 있음
      maxSurge: 3           # 업데이트 중 최대 3개의 추가 파드가 존재할 수 있음
  minReadySeconds: 10       # 파드가 준비된 후 10초 대기 후 진행
  revisionHistoryLimit: 5   # 롤백을 위해 이전 레플리카셋 5개 유지
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: nginx
          image: nginx:1.26
          ports:
            - containerPort: 80
```

롤링 업데이트 순서 (10개 레플리카, maxUnavailable=2, maxSurge=3):
1. 새 RS를 생성하고 3개로 스케일업 (서지)
2. 이전 RS를 8개로 스케일다운 (2개 사용 불가)
3. 새 파드가 준비되면 새 RS 스케일업, 이전 RS 스케일다운 계속
4. 최종: 새 RS 10개, 이전 RS 0개

```bash
# 이미지를 변경하여 롤링 업데이트 트리거
kubectl set image deployment/web-deploy nginx=nginx:1.26

# 롤아웃 감시
kubectl rollout status deployment/web-deploy

# 롤아웃 이력 확인
kubectl rollout history deployment/web-deploy

# 특정 리비전 확인
kubectl rollout history deployment/web-deploy --revision=2
```

### 3.3 재생성 전략(Recreate Strategy)

기존 파드를 모두 종료한 후 새 파드를 생성합니다:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legacy-app
spec:
  replicas: 3
  strategy:
    type: Recreate    # 모두 종료 후 모두 생성 (다운타임 발생!)
  selector:
    matchLabels:
      app: legacy
  template:
    metadata:
      labels:
        app: legacy
    spec:
      containers:
        - name: app
          image: legacy-app:v2.0
```

재생성 전략을 사용하는 경우:
- 앱이 두 버전의 동시 실행을 허용하지 않는 경우
- 앱이 `ReadWriteOnce` 접근 모드의 볼륨을 사용하는 경우
- 데이터베이스 스키마 변경이 이전 버전과 호환되지 않는 경우

### 3.4 롤백

```bash
# 이전 리비전으로 롤백
kubectl rollout undo deployment/web-deploy

# 특정 리비전으로 롤백
kubectl rollout undo deployment/web-deploy --to-revision=2

# 롤아웃 일시 중지 (카나리 스타일 테스트용)
kubectl rollout pause deployment/web-deploy

# 일시 중지 후 재개
kubectl rollout resume deployment/web-deploy
```

### 3.5 블루-그린 배포 패턴(Blue-Green Deployment Pattern)

쿠버네티스에는 내장 블루-그린 리소스가 없지만, 두 개의 디플로이먼트와
하나의 서비스로 구현할 수 있습니다:

```yaml
# 블루 디플로이먼트 (현재 프로덕션)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
      version: blue
  template:
    metadata:
      labels:
        app: web
        version: blue
    spec:
      containers:
        - name: app
          image: my-app:v1.0
---
# 그린 디플로이먼트 (새 버전)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
      version: green
  template:
    metadata:
      labels:
        app: web
        version: green
    spec:
      containers:
        - name: app
          image: my-app:v2.0
---
# 서비스: 셀렉터를 변경하여 전환
apiVersion: v1
kind: Service
metadata:
  name: web-svc
spec:
  selector:
    app: web
    version: blue    # "green"으로 변경하여 트래픽 전환
  ports:
    - port: 80
      targetPort: 8080
```

```bash
# 블루에서 그린으로 트래픽 전환
kubectl patch service web-svc -p '{"spec":{"selector":{"version":"green"}}}'

# 롤백: 블루로 다시 전환
kubectl patch service web-svc -p '{"spec":{"selector":{"version":"blue"}}}'
```

---

## 4. 스테이트풀셋

스테이트풀셋(StatefulSets)은 순서, 안정적인 네트워크 정체성, 영속 스토리지에 대한
보장과 함께 상태 저장 애플리케이션을 관리합니다.

### 4.1 주요 속성

| 기능 | 디플로이먼트(Deployment) | 스테이트풀셋(StatefulSet) |
|---------|-----------|-------------|
| 파드 이름 | 랜덤 해시 접미사 | 순서 인덱스 (0, 1, 2...) |
| 스케일링 | 병렬 | 순차 (기본) |
| 스토리지 | 공유 또는 없음 | 파드별 PVC |
| 네트워크 정체성 | 랜덤 | 안정적 DNS (`pod-0.svc`) |
| 업데이트 순서 | 임의 | 역순 (N-1 → 0) |

### 4.2 스테이트풀셋 매니페스트

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres-headless   # 필수: 헤드리스 서비스 이름
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16
          ports:
            - containerPort: 5432
              name: postgres
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
          readinessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - postgres
            periodSeconds: 10

  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: standard
        resources:
          requests:
            storage: 10Gi

  # 파드 관리 정책
  podManagementPolicy: OrderedReady   # 기본: 한 번에 하나씩 스케일링
  # podManagementPolicy: Parallel     # 모두 동시에 스케일링 (순서가 필요 없는 워크로드)
---
# 안정적 DNS를 위한 헤드리스 서비스
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
spec:
  clusterIP: None            # 헤드리스 서비스
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
```

### 4.3 안정적 네트워크 정체성

스테이트풀셋은 디플로이먼트가 제공하지 않는 세 가지 안정적 보장을 제공합니다:

```
StatefulSet: web (replicas=3)
│
├── 안정적인 파드 이름 (예측 가능, 무작위 아님)
│   ├── web-0
│   ├── web-1
│   └── web-2
│
├── 헤드리스 서비스를 통한 안정적인 DNS (clusterIP: None)
│   ├── web-0.web-headless.default.svc.cluster.local
│   ├── web-1.web-headless.default.svc.cluster.local
│   └── web-2.web-headless.default.svc.cluster.local
│
└── 파드별 PersistentVolumeClaim (파드 재시작 시에도 유지)
    ├── data-web-0  (PV에 바인딩됨)
    ├── data-web-1  (PV에 바인딩됨)
    └── data-web-2  (PV에 바인딩됨)
```

위 스테이트풀셋으로 파드는 예측 가능한 DNS 이름을 받습니다:

```
postgres-0.postgres-headless.default.svc.cluster.local
postgres-1.postgres-headless.default.svc.cluster.local
postgres-2.postgres-headless.default.svc.cluster.local
```

```bash
# 클러스터 내에서 DNS 해석 확인
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup postgres-0.postgres-headless.default.svc.cluster.local
```

### 4.4 업데이트 전략

```yaml
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2    # 순서 >= 2인 파드만 업데이트
      # 카나리 테스트에 유용: pod-2를 먼저 업데이트한 후 파티션을 낮춤
```

```bash
# 이미지 업데이트
kubectl set image statefulset/postgres postgres=postgres:17

# partition=2로 postgres-2만 업데이트됨
# 파티션을 낮춰 더 많은 파드로 롤아웃
kubectl patch statefulset postgres -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":1}}}}'
# 이제 postgres-1과 postgres-2가 업데이트됨

kubectl patch statefulset postgres -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":0}}}}'
# 전체 롤아웃 완료
```

### 4.5 순서 보장 배포와 스케일링

- **스케일업**: 파드가 순서대로 생성: 0, 1, 2, ...
- **스케일다운**: 파드가 역순으로 종료: 2, 1, 0
- 각 파드가 Running이고 Ready여야 다음 파드가 생성됨
- 이는 프라이머리 우선 초기화가 필요한 데이터베이스에 중요함

---

## 5. 데몬셋

데몬셋(DaemonSet)은 모든 (또는 선택된) 노드에서 파드의 복사본이 실행되도록 보장합니다.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      # 컨트롤 플레인을 포함한 모든 노드에서 실행하기 위해 모든 테인트 허용
      tolerations:
        - operator: Exists
      hostNetwork: true     # 노드의 네트워크 네임스페이스 사용
      hostPID: true         # 메트릭을 위해 호스트 프로세스 접근
      containers:
        - name: node-exporter
          image: prom/node-exporter:v1.7.0
          ports:
            - containerPort: 9100
              hostPort: 9100
              name: metrics
          args:
            - --path.procfs=/host/proc
            - --path.sysfs=/host/sys
            - --path.rootfs=/host/root
          volumeMounts:
            - name: proc
              mountPath: /host/proc
              readOnly: true
            - name: sys
              mountPath: /host/sys
              readOnly: true
            - name: root
              mountPath: /host/root
              readOnly: true
              mountPropagation: HostToContainer
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
      volumes:
        - name: proc
          hostPath:
            path: /proc
        - name: sys
          hostPath:
            path: /sys
        - name: root
          hostPath:
            path: /
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1    # 한 번에 하나의 노드만 업데이트
```

일반적인 데몬셋 사용 사례:
- 노드 모니터링 (node-exporter, Datadog 에이전트)
- 로그 수집 (Fluentd, Fluent Bit)
- 네트워크 플러그인 (Calico, Cilium)
- 스토리지 플러그인 (CSI 노드 드라이버)

```bash
# 데몬셋 상태 확인
kubectl -n monitoring get ds node-exporter

# 파드가 어느 노드에 있는지 확인
kubectl -n monitoring get pods -l app=node-exporter -o wide
```

### 5.1 특정 노드 타겟팅

```yaml
spec:
  template:
    spec:
      nodeSelector:
        gpu: "true"      # GPU 노드에서만 실행
      # 또는 더 복잡한 규칙을 위해 어피니티 사용
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: node.kubernetes.io/instance-type
                    operator: In
                    values: ["p4d.24xlarge", "g5.xlarge"]
```

---

## 6. 잡과 크론잡

### 6.1 잡(Jobs)

잡(Job)은 하나 이상의 파드를 생성하고 지정된 수의 파드가 성공적으로
종료되도록 보장합니다.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  completions: 1           # 필요한 성공적 완료 수
  parallelism: 1           # 병렬로 실행되는 파드 수
  backoffLimit: 3           # 실패로 표시하기 전 재시도 횟수
  activeDeadlineSeconds: 600  # 타임아웃: 10분 후 종료
  ttlSecondsAfterFinished: 3600  # 1시간 후 자동 삭제

  template:
    spec:
      restartPolicy: Never   # 필수: Never 또는 OnFailure
      containers:
        - name: migrate
          image: my-app:v1.0
          command: ["./migrate", "--target=latest"]
          env:
            - name: DB_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
          resources:
            requests:
              cpu: "500m"
              memory: "256Mi"
```

### 6.2 병렬 잡(Parallel Jobs)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: batch-processor
spec:
  completions: 10          # 총 10개 항목 처리
  parallelism: 3           # 한 번에 3개의 파드 실행
  completionMode: Indexed  # 각 파드가 고유 인덱스 수신 (0-9)
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: processor
          image: batch-processor:v1.0
          command:
            - sh
            - -c
            - |
              echo "Processing item $JOB_COMPLETION_INDEX"
              # 인덱스를 사용하여 작업 파티셔닝
              ./process --partition=$JOB_COMPLETION_INDEX --total=10
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
```

```bash
# 잡 진행 상황 모니터링
kubectl get job batch-processor -w

# 완료/실패된 파드 확인
kubectl get pods -l job-name=batch-processor

# 특정 인덱스 파드의 로그 확인
kubectl logs batch-processor-2-xxxxx
```

### 6.3 크론잡(CronJobs)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-backup
spec:
  schedule: "0 2 * * *"              # 매일 오전 2시
  timeZone: "America/New_York"       # 타임존 인식 (v1.27+)
  concurrencyPolicy: Forbid          # 이전 잡이 아직 실행 중이면 건너뜀
  successfulJobsHistoryLimit: 3      # 성공한 잡 3개 유지
  failedJobsHistoryLimit: 5          # 실패한 잡 5개 유지
  startingDeadlineSeconds: 300       # 5분 이상 지연되면 건너뜀
  suspend: false                     # true로 설정하면 스케줄링 일시 중지

  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 3600    # 1시간 타임아웃
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: backup
              image: postgres:16
              command:
                - sh
                - -c
                - |
                  pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | \
                    gzip > /backup/db-$(date +%Y%m%d).sql.gz
              env:
                - name: DB_HOST
                  value: "postgres-headless"
                - name: DB_USER
                  valueFrom:
                    secretKeyRef:
                      name: db-credentials
                      key: username
                - name: DB_NAME
                  value: "production"
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: db-credentials
                      key: password
              volumeMounts:
                - name: backup-storage
                  mountPath: /backup
          volumes:
            - name: backup-storage
              persistentVolumeClaim:
                claimName: backup-pvc
```

크론잡 동시성 정책:
| 정책 | 동작 |
|--------|----------|
| Allow | 여러 잡이 동시에 실행 가능 (기본) |
| Forbid | 이전 잡이 아직 활성이면 새 실행 건너뜀 |
| Replace | 실행 중인 잡을 종료하고 새 잡 시작 |

```bash
# 크론잡 나열
kubectl get cronjobs

# 크론잡 수동 트리거
kubectl create job --from=cronjob/nightly-backup manual-backup

# 크론잡 일시 중지
kubectl patch cronjob nightly-backup -p '{"spec":{"suspend":true}}'
```

---

## 7. 파드 중단 예산

파드 중단 예산(Pod Disruption Budgets, PDB)은 자발적 중단(노드 드레인, 클러스터
업그레이드) 중 최소한의 파드가 가용 상태를 유지하도록 애플리케이션을 보호합니다.

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-pdb
spec:
  # 다음 중 하나를 사용:
  minAvailable: 2          # 최소 2개의 파드가 유지되어야 함
  # maxUnavailable: 1      # 최대 1개의 파드가 다운될 수 있음
  # minAvailable: "50%"    # 백분율도 가능

  selector:
    matchLabels:
      app: web

  # 비정상 파드 퇴거 정책 (v1.31+)
  unhealthyPodEvictionPolicy: AlwaysAllow
  # IfHealthy (기본): 다른 모든 파드가 정상일 때만 비정상 파드 퇴거
  # AlwaysAllow: 비정상 파드 퇴거를 항상 허용
```

```bash
# PDB 상태 확인
kubectl get pdb web-pdb

# 출력:
# NAME      MIN AVAILABLE   MAX UNAVAILABLE   ALLOWED DISRUPTIONS   AGE
# web-pdb   2               N/A               1                     5m

# 노드 드레인 (PDB를 준수함)
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data
```

노드 드레인 중 PDB 상호작용:
1. kubectl drain이 퇴거 요청을 전송
2. API 서버가 퇴거를 허용하기 전에 PDB 확인
3. 퇴거가 PDB를 위반하면 요청이 거부됨 (429)
4. kubectl drain이 PDB가 퇴거를 허용할 때까지 재시도

---

## 8. 리소스 요청과 제한

### 8.1 CPU와 메모리

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: resource-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      resources:
        requests:
          cpu: "250m"       # 0.25 CPU 코어 (보장된 최소치)
          memory: "256Mi"   # 256 MiB (보장된 최소치)
        limits:
          cpu: "500m"       # 0.5 CPU 코어 (허용된 최대치)
          memory: "512Mi"   # 512 MiB (초과 시 OOMKilled)
```

리소스 동작:
| 리소스 | 요청(Request) | 제한(Limit) | 제한 초과 시 |
|----------|---------|-------|-----------------|
| CPU | 보장된 스케줄링 | CFS로 스로틀링 | 스로틀링됨 (종료되지 않음) |
| 메모리 | 보장된 스케줄링 | 하드 캡 | OOMKilled |

### 8.2 CPU 단위 이해

```
1 CPU = 1000m (밀리코어)
1 CPU = 1 vCPU (AWS) = 1 Core (GCP) = 1 vCore (Azure)

일반적인 값:
  100m = 0.1 CPU (한 코어의 10%)
  250m = 0.25 CPU
  500m = 0.5 CPU
  1000m = 1.0 CPU = 1
```

### 8.3 메모리 단위 이해

```
이진 단위 (2의 거듭제곱):
  Ki = 1024 바이트
  Mi = 1024 Ki = 1,048,576 바이트
  Gi = 1024 Mi = 1,073,741,824 바이트

십진 단위 (10의 거듭제곱):
  K = 1000 바이트
  M = 1000 K = 1,000,000 바이트
  G = 1000 M = 1,000,000,000 바이트

일관성을 위해 항상 이진 단위 (Mi, Gi)를 사용하세요.
```

### 8.4 리밋레인지(LimitRange)

네임스페이스 수준에서 기본값과 제약 조건을 적용합니다:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: production
spec:
  limits:
    - type: Container
      default:           # 기본 제한 (지정되지 않은 경우)
        cpu: "500m"
        memory: "256Mi"
      defaultRequest:    # 기본 요청 (지정되지 않은 경우)
        cpu: "100m"
        memory: "128Mi"
      min:               # 허용된 최소값
        cpu: "50m"
        memory: "64Mi"
      max:               # 허용된 최대값
        cpu: "2"
        memory: "2Gi"
    - type: Pod
      max:
        cpu: "4"
        memory: "4Gi"
```

### 8.5 리소스쿼터(ResourceQuota)

네임스페이스당 총 리소스 소비를 제한합니다:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
spec:
  hard:
    requests.cpu: "20"
    requests.memory: "40Gi"
    limits.cpu: "40"
    limits.memory: "80Gi"
    pods: "50"
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
    count/deployments.apps: "20"
```

```bash
# 쿼터 사용량 확인
kubectl describe resourcequota compute-quota -n production
```

---

## 9. QoS 클래스

쿠버네티스는 리소스 설정에 따라 모든 파드에 세 가지 QoS 클래스 중 하나를 할당합니다.
QoS 클래스는 노드 메모리 압박 시 파드가 종료되는 순서를 결정합니다.

### 9.1 Guaranteed

모든 컨테이너의 CPU와 메모리에 대해 요청 == 제한입니다.

```yaml
# QoS: Guaranteed
spec:
  containers:
    - name: app
      resources:
        requests:
          cpu: "500m"
          memory: "256Mi"
        limits:
          cpu: "500m"       # 요청과 동일
          memory: "256Mi"   # 요청과 동일
```

- 메모리 압박 시 가장 마지막으로 퇴거됨
- 전용 CPU 시간을 받음 (CPU 매니저 정책이 `static`인 경우)

### 9.2 Burstable

적어도 하나의 컨테이너에 요청 또는 제한이 설정되어 있지만, 같지 않습니다.

```yaml
# QoS: Burstable
spec:
  containers:
    - name: app
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "500m"       # 요청과 다름
          memory: "512Mi"   # 요청과 다름
```

- BestEffort 파드 이후에 퇴거됨
- 리소스가 가용할 때 요청 이상으로 버스트 가능

### 9.3 BestEffort

리소스 요청이나 제한이 전혀 설정되지 않습니다.

```yaml
# QoS: BestEffort
spec:
  containers:
    - name: app
      image: my-app:v1.0
      # 리소스 지정 없음
```

- 메모리 압박 시 가장 먼저 퇴거됨
- 사용 가능한 리소스를 받음
- 프로덕션 워크로드에는 권장하지 않음

### 9.4 QoS 클래스 확인

```bash
# 파드에 할당된 QoS 클래스 확인
kubectl get pod resource-demo -o jsonpath='{.status.qosClass}'

# 모든 파드의 QoS 클래스 나열
kubectl get pods -o custom-columns='NAME:.metadata.name,QOS:.status.qosClass'
```

### 9.5 퇴거 순서

노드 메모리 압박 시, kubelet은 다음 순서로 파드를 퇴거합니다:

1. **BestEffort** 파드 (가장 먼저 퇴거)
2. **Burstable** 파드 중 메모리 요청을 초과한 것
3. **Guaranteed** 파드 (제한을 초과한 경우에만, 이는 OOM과 같음)

동일한 QoS 클래스 내에서는 요청 대비 메모리를 더 많이 사용하는
파드가 먼저 퇴거됩니다.

```bash
# 노드 메모리 압박 조건 확인
kubectl describe node minikube | grep -A 5 Conditions

# 퇴거 임계값 확인
kubectl get --raw /api/v1/nodes/minikube/proxy/configz | python3 -m json.tool | grep eviction
```

---

## 연습문제

### 연습문제 1: 멀티컨테이너 파드

초기화 컨테이너가 파일을 작성하고, 메인 컨테이너가 이를 서빙하는 파드를 생성합니다.
초기화 컨테이너는 "Hello from init!"을 공유 볼륨에 쓰고, 메인 컨테이너는
포트 80에서 이를 서빙해야 합니다.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/multi-container.yaml로 저장
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-demo
spec:
  volumes:
    - name: shared-data
      emptyDir: {}
  initContainers:
    - name: init-writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "<html><body><h1>Hello from init!</h1></body></html>" \
            > /data/index.html
          echo "Init container completed"
      volumeMounts:
        - name: shared-data
          mountPath: /data
  containers:
    - name: web-server
      image: nginx:1.25
      ports:
        - containerPort: 80
      volumeMounts:
        - name: shared-data
          mountPath: /usr/share/nginx/html
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
```

```bash
kubectl apply -f /tmp/multi-container.yaml
kubectl wait --for=condition=Ready pod/multi-container-demo --timeout=60s

# 콘텐츠 확인
kubectl exec multi-container-demo -c web-server -- curl -s localhost

# 초기화 컨테이너 로그 확인
kubectl logs multi-container-demo -c init-writer

# 정리
kubectl delete pod multi-container-demo
```

</details>

### 연습문제 2: 디플로이먼트 롤링 업데이트

nginx:1.24를 실행하는 5개의 레플리카로 디플로이먼트를 생성합니다. maxUnavailable=1,
maxSurge=2로 nginx:1.25로 롤링 업데이트를 수행합니다. 그런 다음 이전 버전으로
롤백합니다.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/rolling-update.yaml로 저장
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-demo
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: rolling-demo
  template:
    metadata:
      labels:
        app: rolling-demo
    spec:
      containers:
        - name: nginx
          image: nginx:1.24
          ports:
            - containerPort: 80
          readinessProbe:
            httpGet:
              path: /
              port: 80
            periodSeconds: 3
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
```

```bash
# 초기 버전 배포
kubectl apply -f /tmp/rolling-update.yaml
kubectl rollout status deployment/rolling-demo

# 초기 상태 기록
kubectl rollout history deployment/rolling-demo

# 롤링 업데이트 트리거
kubectl set image deployment/rolling-demo nginx=nginx:1.25

# 실시간 롤아웃 감시
kubectl rollout status deployment/rolling-demo

# 새 이미지 확인
kubectl get pods -l app=rolling-demo -o jsonpath='{.items[0].spec.containers[0].image}'
# nginx:1.25

# 이전 버전으로 롤백
kubectl rollout undo deployment/rolling-demo

# 롤백 확인
kubectl get pods -l app=rolling-demo -o jsonpath='{.items[0].spec.containers[0].image}'
# nginx:1.24

# 정리
kubectl delete deployment rolling-demo
```

</details>

### 연습문제 3: 영속 스토리지를 가진 스테이트풀셋

3개의 레플리카를 가진 Redis 클러스터용 스테이트풀셋을 생성합니다. 각각 자체
1Gi PersistentVolumeClaim을 가집니다. 안정적인 네트워크 정체성과 스토리지 영속성을
확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/statefulset-exercise.yaml로 저장
apiVersion: v1
kind: Service
metadata:
  name: redis-headless
spec:
  clusterIP: None
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis-headless
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7.2
          ports:
            - containerPort: 6379
          command:
            - redis-server
            - --appendonly
            - "yes"
            - --dir
            - /data
          volumeMounts:
            - name: redis-data
              mountPath: /data
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "200m"
              memory: "256Mi"
          readinessProbe:
            exec:
              command: ["redis-cli", "ping"]
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: redis-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

```bash
kubectl apply -f /tmp/statefulset-exercise.yaml

# 모든 파드 대기
kubectl rollout status statefulset/redis

# 순서가 있는 파드 이름 확인
kubectl get pods -l app=redis
# redis-0   1/1   Running
# redis-1   1/1   Running
# redis-2   1/1   Running

# 안정적 DNS 확인
kubectl run dns-check --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup redis-0.redis-headless.default.svc.cluster.local

# redis-0에 데이터 쓰기
kubectl exec redis-0 -- redis-cli set test-key "persisted-value"

# 파드 삭제 (스테이트풀셋이 같은 이름과 PVC로 재생성)
kubectl delete pod redis-0
kubectl wait --for=condition=Ready pod/redis-0 --timeout=60s

# 데이터 영속성 확인
kubectl exec redis-0 -- redis-cli get test-key
# "persisted-value"

# PVC 확인
kubectl get pvc -l app=redis
# redis-data-redis-0   Bound   1Gi
# redis-data-redis-1   Bound   1Gi
# redis-data-redis-2   Bound   1Gi

# 정리
kubectl delete statefulset redis
kubectl delete svc redis-headless
kubectl delete pvc -l app=redis
```

</details>

### 연습문제 4: 인덱스 완료를 가진 잡

인덱스 완료 모드를 사용하여 5개의 항목을 병렬로(한 번에 2개) 처리하는 잡을
생성합니다. 각 파드는 자신의 완료 인덱스를 출력해야 합니다.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/indexed-job.yaml로 저장
apiVersion: batch/v1
kind: Job
metadata:
  name: indexed-processor
spec:
  completions: 5
  parallelism: 2
  completionMode: Indexed
  backoffLimit: 3
  ttlSecondsAfterFinished: 300
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              INDEX=${JOB_COMPLETION_INDEX}
              echo "Worker processing item ${INDEX} of 5"
              echo "Start time: $(date)"
              # 인덱스에 비례하는 작업 시뮬레이션
              sleep $((INDEX + 1))
              echo "Item ${INDEX} completed at $(date)"
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
```

```bash
kubectl apply -f /tmp/indexed-job.yaml

# 잡 진행 상황 감시
kubectl get job indexed-processor -w

# 인덱스가 있는 파드 확인
kubectl get pods -l job-name=indexed-processor \
  -o custom-columns='NAME:.metadata.name,INDEX:.metadata.annotations.batch\.kubernetes\.io/job-completion-index,STATUS:.status.phase'

# 각 워커의 로그 확인
for i in $(seq 0 4); do
  echo "=== Worker $i ==="
  kubectl logs -l batch.kubernetes.io/job-completion-index=$i -l job-name=indexed-processor
done

# 잡 상태 확인
kubectl describe job indexed-processor

# 정리
kubectl delete job indexed-processor
```

</details>

### 연습문제 5: 리소스 제한과 QoS

세 가지 QoS 클래스(Guaranteed, Burstable, BestEffort)를 보여주는 세 개의 파드를
생성합니다. 각 파드의 QoS 클래스 할당을 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/qos-exercise.yaml로 저장
# 파드 1: Guaranteed (requests == limits)
apiVersion: v1
kind: Pod
metadata:
  name: qos-guaranteed
spec:
  containers:
    - name: app
      image: nginx:1.25
      resources:
        requests:
          cpu: "200m"
          memory: "128Mi"
        limits:
          cpu: "200m"
          memory: "128Mi"
---
# 파드 2: Burstable (requests != limits)
apiVersion: v1
kind: Pod
metadata:
  name: qos-burstable
spec:
  containers:
    - name: app
      image: nginx:1.25
      resources:
        requests:
          cpu: "100m"
          memory: "64Mi"
        limits:
          cpu: "500m"
          memory: "256Mi"
---
# 파드 3: BestEffort (리소스 지정 없음)
apiVersion: v1
kind: Pod
metadata:
  name: qos-besteffort
spec:
  containers:
    - name: app
      image: nginx:1.25
```

```bash
kubectl apply -f /tmp/qos-exercise.yaml

# 모든 파드가 준비될 때까지 대기
kubectl wait --for=condition=Ready pod/qos-guaranteed pod/qos-burstable pod/qos-besteffort --timeout=60s

# QoS 클래스 확인
echo "=== QoS Classes ==="
kubectl get pod qos-guaranteed -o jsonpath='qos-guaranteed:  {.status.qosClass}{"\n"}'
kubectl get pod qos-burstable -o jsonpath='qos-burstable:   {.status.qosClass}{"\n"}'
kubectl get pod qos-besteffort -o jsonpath='qos-besteffort:  {.status.qosClass}{"\n"}'

# 예상 출력:
# qos-guaranteed:  Guaranteed
# qos-burstable:   Burstable
# qos-besteffort:  BestEffort

# QoS와 함께 모든 파드 확인
kubectl get pods -o custom-columns='NAME:.metadata.name,QOS:.status.qosClass,CPU_REQ:.spec.containers[0].resources.requests.cpu,CPU_LIM:.spec.containers[0].resources.limits.cpu,MEM_REQ:.spec.containers[0].resources.requests.memory,MEM_LIM:.spec.containers[0].resources.limits.memory'

# 정리
kubectl delete pod qos-guaranteed qos-burstable qos-besteffort
```

</details>

---

**이전**: [아키텍처 심층 분석](./01_Architecture_Deep_Dive.md) | **다음**: [네트워킹 기초](./03_Networking_Fundamentals.md)
