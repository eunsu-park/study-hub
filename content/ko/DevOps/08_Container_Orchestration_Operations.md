# 레슨 8: 컨테이너 오케스트레이션 운영

**이전**: [구성 관리](./07_Configuration_Management.md) | **다음**: [서비스 메시와 네트워킹](./09_Service_Mesh_and_Networking.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 무중단 릴리스를 위한 롤링 업데이트 전략이 포함된 Kubernetes Deployment 매니페스트를 작성하고 적용할 수 있습니다
2. 클러스터 내부 및 외부에서 애플리케이션을 노출하기 위한 Service(ClusterIP, NodePort, LoadBalancer)를 구성할 수 있습니다
3. ConfigMap으로 애플리케이션 구성을, Secret으로 민감한 데이터를 관리할 수 있습니다
4. 공정한 스케줄링을 보장하고 리소스 기아(starvation)를 방지하기 위해 리소스 요청(request)과 제한(limit)을 설정할 수 있습니다
5. CPU, 메모리 또는 사용자 정의 메트릭을 기반으로 워크로드를 자동 확장하기 위해 Horizontal Pod Autoscaler(HPA)를 구성할 수 있습니다
6. 노드 어피니티(affinity), 테인트(taint), 톨러레이션(toleration)을 적용하여 노드 간 파드 배치를 제어할 수 있습니다

---

Kubernetes는 프로덕션 환경에서 컨테이너화된 애플리케이션을 실행하기 위한 표준 플랫폼이 되었습니다. Docker 레슨에서 Kubernetes 기본 사항(Pod, Deployment, Service)을 다루었지만, 이 레슨에서는 DevOps 엔지니어가 매일 사용하는 운영 패턴인 롤링 업데이트, 리소스 관리, 오토스케일링, 고급 스케줄링에 초점을 맞춥니다. 이러한 기술은 프로덕션 Kubernetes 클러스터를 원활하고, 효율적이며, 안정적으로 운영하는 핵심입니다.

> **비유 -- 도시 인프라 관리**: 프로덕션 환경에서 Kubernetes를 운영하는 것은 도시의 인프라를 관리하는 것과 같습니다. Deployment는 건설 프로젝트(도시를 멈추지 않고 새 건물을 배치)입니다. Service는 도로 네트워크(올바른 건물로 교통을 안내)입니다. 리소스 제한은 용도지역 법규(한 건물이 모든 전력을 사용하는 것을 방지)입니다. HPA는 도시의 성장 계획(인구가 증가하면 더 많은 주택을 건설)입니다.

## 1. Deployment 심화

### Deployment 매니페스트

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: v2.1.0
  annotations:
    deployment.kubernetes.io/revision: "3"
    description: "Main API server for the application"
spec:
  replicas: 3
  revisionHistoryLimit: 10                  # Keep 10 old ReplicaSets for rollback

  selector:
    matchLabels:
      app: api-server                       # Must match template labels

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1                           # At most 1 extra pod during update
      maxUnavailable: 0                     # Never reduce below desired replicas

  template:
    metadata:
      labels:
        app: api-server
        version: v2.1.0
    spec:
      terminationGracePeriodSeconds: 30     # Time for graceful shutdown

      containers:
        - name: api
          image: registry.example.com/api-server:v2.1.0
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics

          # Health checks
          readinessProbe:                    # Is the pod ready to receive traffic?
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 3

          livenessProbe:                     # Is the pod alive and functioning?
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 15
            periodSeconds: 20
            failureThreshold: 3

          startupProbe:                      # Has the pod finished starting up?
            httpGet:
              path: /health/startup
              port: http
            initialDelaySeconds: 0
            periodSeconds: 5
            failureThreshold: 30             # 30 * 5 = 150s max startup time

          # Resource management
          resources:
            requests:
              cpu: 250m                      # 0.25 CPU cores
              memory: 256Mi                  # 256 MiB RAM
            limits:
              cpu: 500m                      # 0.5 CPU cores
              memory: 512Mi                  # 512 MiB RAM

          # Environment variables
          env:
            - name: PORT
              value: "8080"
            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: api-config
                  key: log_level
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: db_password

          # Volume mounts
          volumeMounts:
            - name: config-volume
              mountPath: /etc/app/config
              readOnly: true

      volumes:
        - name: config-volume
          configMap:
            name: api-config
```

### 롤링 업데이트 전략

```
전략: RollingUpdate (기본값)
──────────────────────────────────
기존 파드를 새 파드로 점진적으로 교체합니다.

maxSurge: 1, maxUnavailable: 0 (가장 안전 -- 원하는 수 이하로 떨어지지 않음)
  Replicas: 3
  Step 1: [v1] [v1] [v1] [v2]      4 total (1 surge)
  Step 2: [v1] [v1] [v2] [v2]      4 total
  Step 3: [v1] [v2] [v2] [v2]      4 total
  Step 4: [v2] [v2] [v2]           3 total (완료)

maxSurge: 0, maxUnavailable: 1 (추가 리소스 불필요)
  Replicas: 3
  Step 1: [v1] [v1] [--]           2 available (1 terminating)
  Step 2: [v1] [v1] [v2]           3 total (new pod starting)
  Step 3: [v1] [--] [v2]           2 available
  Step 4: [v1] [v2] [v2]           3 total
  Step 5: [v2] [v2] [v2]           3 total (완료)

maxSurge: 25%, maxUnavailable: 25% (균형 잡힌 방식 -- 기본값)
  Replicas: 4
  최대 5개 파드 (4 + 25% surge)
  최소 3개 파드 사용 가능 (4 - 25% unavailable)

전략: Recreate
──────────────────
모든 기존 파드를 종료한 후 새 파드를 생성합니다.
  Step 1: [v1] [v1] [v1]           실행 중
  Step 2: [--] [--] [--]           모두 종료 (다운타임 발생)
  Step 3: [v2] [v2] [v2]           모든 새 파드 시작
  사용 시기: 새 버전이 이전 버전과 호환되지 않는 경우
```

### Deployment 운영 명령어

```bash
# Deployment 적용
kubectl apply -f deployment.yaml

# 롤아웃 상태 확인
kubectl rollout status deployment/api-server

# 롤아웃 히스토리 조회
kubectl rollout history deployment/api-server

# 이전 버전으로 롤백
kubectl rollout undo deployment/api-server

# 특정 리비전으로 롤백
kubectl rollout undo deployment/api-server --to-revision=2

# 롤아웃 일시 중지 (카나리 방식의 수동 검증용)
kubectl rollout pause deployment/api-server

# 일시 중지된 롤아웃 재개
kubectl rollout resume deployment/api-server

# Deployment 스케일링
kubectl scale deployment/api-server --replicas=5

# 이미지 업데이트 (롤링 업데이트 트리거)
kubectl set image deployment/api-server api=registry.example.com/api-server:v2.2.0

# 모든 파드 재시작 (롤링 재시작, 구성 변경 없음)
kubectl rollout restart deployment/api-server
```

---

## 2. Service

Service는 파드에 접근하기 위한 안정적인 네트워크 엔드포인트를 제공합니다.

### Service 유형

```
┌────────────────────────────────────────────────────────────────┐
│                    Kubernetes Service Types                     │
│                                                                 │
│  ClusterIP (기본값)                                             │
│  ┌──────────┐                                                  │
│  │ Service   │──▶ [Pod 1] [Pod 2] [Pod 3]                     │
│  │ 10.0.0.5  │    내부 접근만 가능                              │
│  └──────────┘                                                  │
│                                                                 │
│  NodePort                                                       │
│  ┌──────────┐     ┌──────────┐                                 │
│  │ External  │────▶│ Service   │──▶ [Pod 1] [Pod 2]           │
│  │ :30080    │     │ :80       │    모든 노드에서 노출           │
│  └──────────┘     └──────────┘                                 │
│                                                                 │
│  LoadBalancer                                                   │
│  ┌──────────┐     ┌──────────┐                                 │
│  │ Cloud LB  │────▶│ Service   │──▶ [Pod 1] [Pod 2] [Pod 3]  │
│  │ (AWS ELB) │     │ :80       │    클라우드 제공자가 외부      │
│  └──────────┘     └──────────┘    로드밸런서를 프로비저닝       │
│                                                                 │
│  ExternalName                                                   │
│  ┌──────────┐                                                  │
│  │ Service   │──▶ CNAME to external.database.com               │
│  │ (DNS)     │    외부 서비스에 매핑                             │
│  └──────────┘                                                  │
└────────────────────────────────────────────────────────────────┘
```

### Service 매니페스트

```yaml
# ClusterIP Service (내부 전용)
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: production
spec:
  type: ClusterIP                           # Default type
  selector:
    app: api-server                         # Routes to pods with this label
  ports:
    - name: http
      port: 80                              # Service port
      targetPort: 8080                      # Container port
      protocol: TCP
    - name: metrics
      port: 9090
      targetPort: 9090
```

```yaml
# NodePort Service (노드 IP를 통한 외부 접근)
apiVersion: v1
kind: Service
metadata:
  name: api-nodeport
spec:
  type: NodePort
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080                       # External port (30000-32767)
```

```yaml
# LoadBalancer Service (클라우드 제공자 로드밸런서)
apiVersion: v1
kind: Service
metadata:
  name: api-lb
  annotations:
    # AWS-specific annotations
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-scheme: internet-facing
spec:
  type: LoadBalancer
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 8080
```

### DNS 해석

```
같은 네임스페이스 내:
  curl http://api-service           → ClusterIP로 해석됩니다

네임스페이스 간:
  curl http://api-service.production.svc.cluster.local

전체 DNS 형식:
  <service-name>.<namespace>.svc.cluster.local
```

---

## 3. ConfigMap과 Secret

### ConfigMap

```yaml
# YAML에서 ConfigMap 생성
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
  namespace: production
data:
  # Simple key-value pairs
  log_level: "info"
  max_connections: "100"
  feature_flags: "new_dashboard=true,dark_mode=false"

  # Entire configuration file
  app.properties: |
    server.port=8080
    server.host=0.0.0.0
    database.pool.size=10
    cache.ttl=300
```

```bash
# 명령줄에서 ConfigMap 생성
kubectl create configmap api-config \
  --from-literal=log_level=info \
  --from-literal=max_connections=100

# 파일에서 ConfigMap 생성
kubectl create configmap nginx-config \
  --from-file=nginx.conf

# 디렉토리에서 ConfigMap 생성 (각 파일이 키가 됨)
kubectl create configmap app-config \
  --from-file=config/
```

### 파드에서 ConfigMap 사용

```yaml
spec:
  containers:
    - name: api
      image: api-server:v1

      # 옵션 1: 환경 변수로 사용
      env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: api-config
              key: log_level

      # 옵션 2: 모든 키를 환경 변수로 사용
      envFrom:
        - configMapRef:
            name: api-config

      # 옵션 3: 마운트된 볼륨(파일)으로 사용
      volumeMounts:
        - name: config-volume
          mountPath: /etc/app/config
          readOnly: true

  volumes:
    - name: config-volume
      configMap:
        name: api-config
        items:                              # Mount specific keys as files
          - key: app.properties
            path: application.properties    # File name in the mount
```

### Secret

```yaml
# Secret 매니페스트 (값은 base64로 인코딩해야 합니다)
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
  namespace: production
type: Opaque
data:
  db_password: cGFzc3dvcmQxMjM=           # echo -n 'password123' | base64
  api_key: c2stYWJjMTIzZGVm               # echo -n 'sk-abc123def' | base64

# 또는 stringData 사용 (평문, 자동 인코딩)
stringData:
  db_password: password123
  api_key: sk-abc123def
```

```bash
# 명령줄에서 Secret 생성
kubectl create secret generic api-secrets \
  --from-literal=db_password=password123 \
  --from-literal=api_key=sk-abc123def

# TLS Secret 생성
kubectl create secret tls tls-cert \
  --cert=server.crt \
  --key=server.key

# Docker 레지스트리 Secret 생성
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=admin \
  --docker-password=secretpass
```

---

## 4. 리소스 요청(Requests)과 제한(Limits)

### 리소스 이해

```
Requests: 컨테이너에 보장되는 최소 리소스
  - 스케줄러가 파드를 어느 노드에 배치할지 결정하는 데 사용됩니다
  - 컨테이너에 이 양이 보장됩니다

Limits: 컨테이너가 사용할 수 있는 최대 리소스
  - CPU: 제한을 초과하면 컨테이너가 스로틀링됩니다
  - Memory: 제한을 초과하면 컨테이너가 OOMKilled됩니다

┌────────────────────────────────────────────────────┐
│  Container Resource Usage                           │
│                                                     │
│  0 ─────── Request ─────── Limit ─────── Node Max  │
│  │         (guaranteed)    (maximum)                │
│  │                                                  │
│  │  [====]                 CPU: throttled            │
│  │  [===========]          Memory: OOMKilled         │
│                                                     │
│  모범 사례: Request ≈ 일반적인 사용량                │
│             Limit ≈ 최대 사용량                      │
└────────────────────────────────────────────────────┘
```

### 리소스 사양

```yaml
spec:
  containers:
    - name: api
      image: api-server:v1
      resources:
        requests:
          cpu: 250m              # 250 millicores = 0.25 CPU
          memory: 256Mi          # 256 Mebibytes
          ephemeral-storage: 1Gi # 1 GiB local disk
        limits:
          cpu: 500m              # 0.5 CPU (throttled above this)
          memory: 512Mi          # OOMKilled above this
          ephemeral-storage: 2Gi

# CPU 단위:
#   1 CPU = 1000m (millicores)
#   250m = 0.25 CPU = 코어의 1/4
#   1.5 = 1500m = 1.5 코어

# 메모리 단위:
#   Ki = Kibibytes (1024 bytes)
#   Mi = Mebibytes (1024 Ki)
#   Gi = Gibibytes (1024 Mi)
```

### 서비스 품질(QoS) 클래스

```
Kubernetes는 리소스 사양에 따라 QoS 클래스를 할당합니다:

Guaranteed (최고 우선순위, 가장 마지막에 축출):
  - 모든 컨테이너에서 CPU와 메모리 모두 requests == limits
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 500m
      memory: 512Mi

Burstable (중간 우선순위):
  - 하나 이상의 컨테이너에 request가 설정됨
  - Requests != limits
  resources:
    requests:
      cpu: 250m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi

BestEffort (최저 우선순위, 가장 먼저 축출):
  - requests 또는 limits가 설정되지 않음
  resources: {}   # Empty or omitted
```

### LimitRange (네임스페이스 기본값)

```yaml
# 네임스페이스에 대한 기본 리소스 제약 설정
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: production
spec:
  limits:
    - type: Container
      default:                          # Default limits (if not specified)
        cpu: 500m
        memory: 512Mi
      defaultRequest:                   # Default requests (if not specified)
        cpu: 100m
        memory: 128Mi
      max:                              # Maximum allowed limits
        cpu: 2
        memory: 4Gi
      min:                              # Minimum allowed requests
        cpu: 50m
        memory: 64Mi
```

### ResourceQuota (네임스페이스 예산)

```yaml
# 네임스페이스의 총 리소스 제한
apiVersion: v1
kind: ResourceQuota
metadata:
  name: namespace-quota
  namespace: development
spec:
  hard:
    requests.cpu: "10"                 # Total CPU requests across all pods
    requests.memory: 20Gi             # Total memory requests
    limits.cpu: "20"                  # Total CPU limits
    limits.memory: 40Gi               # Total memory limits
    pods: "50"                        # Maximum number of pods
    services: "10"                    # Maximum number of services
    persistentvolumeclaims: "20"      # Maximum PVCs
```

---

## 5. Horizontal Pod Autoscaler (HPA)

HPA는 관찰된 메트릭을 기반으로 파드 레플리카 수를 자동으로 스케일링합니다.

### 기본 HPA (CPU 기반)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server

  minReplicas: 3
  maxReplicas: 20

  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70          # Scale up when avg CPU > 70%

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60      # Wait 60s before scaling up again
      policies:
        - type: Percent
          value: 100                      # Can double the replicas
          periodSeconds: 60
        - type: Pods
          value: 4                        # Or add up to 4 pods
          periodSeconds: 60
      selectPolicy: Max                   # Use whichever policy allows more scaling

    scaleDown:
      stabilizationWindowSeconds: 300     # Wait 5 min before scaling down
      policies:
        - type: Percent
          value: 10                       # Remove at most 10% of pods
          periodSeconds: 60
```

### 다중 메트릭 HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server

  minReplicas: 3
  maxReplicas: 50

  metrics:
    # CPU 기반 스케일링
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    # 메모리 기반 스케일링
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

    # 사용자 정의 메트릭 (예: Prometheus의 초당 요청 수)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: 1000              # Scale when avg RPS > 1000
```

```bash
# 명령적 방식으로 HPA 생성
kubectl autoscale deployment api-server \
  --min=3 --max=20 --cpu-percent=70

# HPA 상태 확인
kubectl get hpa api-server-hpa

# HPA 상세 정보
kubectl describe hpa api-server-hpa

# HPA 동작 실시간 모니터링
kubectl get hpa -w
```

### HPA 사전 요구 사항

```bash
# HPA를 사용하려면 Metrics Server가 설치되어 있어야 합니다
# metrics-server 실행 여부 확인
kubectl get deployment metrics-server -n kube-system

# metrics-server 설치 (없는 경우)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 메트릭 사용 가능 여부 확인
kubectl top nodes
kubectl top pods
```

---

## 6. 노드 어피니티(Node Affinity)와 안티 어피니티(Anti-Affinity)

### 노드 레이블

```bash
# 노드에 레이블 지정
kubectl label nodes node-1 disk=ssd
kubectl label nodes node-2 disk=hdd
kubectl label nodes node-1 zone=us-east-1a
kubectl label nodes node-2 zone=us-east-1b

# 레이블 확인
kubectl get nodes --show-labels
```

### 노드 어피니티

```yaml
spec:
  affinity:
    nodeAffinity:
      # 필수(HARD) 요구사항 -- 이 조건 없이는 스케줄링 불가
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: disk
                operator: In
                values:
                  - ssd                     # Must schedule on SSD nodes

      # 선호(SOFT) 조건 -- 스케줄러가 시도하지만 필수는 아님
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 80                        # Priority weight (1-100)
          preference:
            matchExpressions:
              - key: zone
                operator: In
                values:
                  - us-east-1a              # Prefer zone 1a
```

### 파드 안티 어피니티 (파드 분산)

```yaml
spec:
  affinity:
    podAntiAffinity:
      # 필수(HARD): 같은 노드에 두 개의 api-server 파드를 배치하지 않음
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - api-server
          topologyKey: kubernetes.io/hostname

      # 선호(SOFT): 가용 영역 간에 분산 시도
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchExpressions:
                - key: app
                  operator: In
                  values:
                    - api-server
            topologyKey: topology.kubernetes.io/zone
```

### 테인트(Taint)와 톨러레이션(Toleration)

```bash
# 노드에 테인트 적용 (톨러레이션이 없으면 스케줄링 방지)
kubectl taint nodes node-gpu gpu=true:NoSchedule

# 매칭되는 톨러레이션이 있는 파드만 node-gpu에 스케줄링됩니다
```

```yaml
# GPU 테인트를 허용하는 파드
spec:
  tolerations:
    - key: "gpu"
      operator: "Equal"
      value: "true"
      effect: "NoSchedule"

  containers:
    - name: ml-training
      image: ml-training:v1
      resources:
        limits:
          nvidia.com/gpu: 1               # Request 1 GPU
```

```
테인트 효과:
  NoSchedule:       새 파드를 스케줄링하지 않음 (기존 파드는 유지)
  PreferNoSchedule: 스케줄링을 피하려고 시도 (소프트 버전)
  NoExecute:        기존 파드를 축출하고 새 파드도 스케줄링하지 않음
```

---

## 7. 토폴로지 분산 제약 조건(Topology Spread Constraints)

```yaml
# 영역과 노드 간에 파드를 균등하게 분산
spec:
  topologySpreadConstraints:
    # 가용 영역 간 분산
    - maxSkew: 1                            # Max difference in pod count between zones
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: DoNotSchedule      # Hard constraint
      labelSelector:
        matchLabels:
          app: api-server

    # 각 영역 내의 노드 간 분산
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: ScheduleAnyway     # Soft constraint
      labelSelector:
        matchLabels:
          app: api-server
```

---

## 8. 운영 명령어 참조

```bash
# === 디버깅 ===
# 파드 로그 조회
kubectl logs api-server-abc123
kubectl logs api-server-abc123 -c sidecar     # Specific container
kubectl logs api-server-abc123 --previous      # Previous crash logs
kubectl logs -f api-server-abc123              # Follow logs
kubectl logs -l app=api-server --all-containers # All pods with label

# 파드 내부에서 실행
kubectl exec -it api-server-abc123 -- /bin/sh
kubectl exec -it api-server-abc123 -c sidecar -- bash

# 상세 정보 (이벤트, 조건)
kubectl describe pod api-server-abc123
kubectl describe deployment api-server
kubectl describe node node-1

# 이벤트 조회 (시간순 정렬)
kubectl get events --sort-by='.lastTimestamp'
kubectl get events --field-selector type=Warning

# === 리소스 검사 ===
kubectl get all -n production
kubectl get pods -o wide                       # Show node and IP
kubectl get pods -o yaml                       # Full YAML
kubectl get pods -l app=api-server             # Filter by label

# === 포트 포워딩 ===
kubectl port-forward pod/api-server-abc123 8080:8080
kubectl port-forward svc/api-service 8080:80

# === 리소스 사용량 ===
kubectl top pods
kubectl top nodes
kubectl top pods --sort-by=memory
```

---

## 연습 문제

### 연습 문제 1: 무중단 배포

롤링 업데이트 전략으로 Deployment를 생성하십시오:
1. v1을 3개의 레플리카로 배포하고, maxSurge=1, maxUnavailable=0으로 설정합니다
2. readiness 프로브와 liveness 프로브를 추가합니다
3. v2로 업데이트하고 `kubectl get pods -w`로 롤링 업데이트를 관찰합니다
4. 업데이트 중 어느 시점에서도 준비된 파드가 3개 미만이 아니었음을 확인합니다
5. v1으로 롤백하고 롤백을 확인합니다

### 연습 문제 2: 리소스 관리

네임스페이스에 대한 리소스 관리를 설정하십시오:
1. `team-alpha`라는 네임스페이스를 생성합니다
2. 기본 요청(100m CPU, 128Mi 메모리)과 제한(500m CPU, 512Mi 메모리)을 가진 LimitRange를 적용합니다
3. 총 요청을 4 CPU와 8Gi 메모리로 제한하는 ResourceQuota를 적용합니다
4. 명시적 리소스 사양 없이 애플리케이션을 배포하고 기본값이 적용되었는지 확인합니다
5. ResourceQuota를 초과하도록 충분한 레플리카를 배포해보고 오류를 관찰합니다

### 연습 문제 3: 오토스케일링

애플리케이션에 대한 오토스케일링을 구성하십시오:
1. CPU 집약적인 애플리케이션을 배포합니다 (예: 해시를 계산하는 컨테이너 사용)
2. 리소스 요청을 100m CPU로 설정합니다
3. 50% CPU 사용률을 목표로 하고, min=1, max=10인 HPA를 생성합니다
4. 부하를 생성합니다 (예: `kubectl run -i --tty load-generator --image=busybox -- /bin/sh -c "while true; do wget -q -O- http://api-service; done"`)
5. HPA가 파드를 스케일 업하는 것을 관찰합니다
6. 부하를 중지하고 스케일 다운되는 것을 관찰합니다

### 연습 문제 4: 파드 스케줄링

고급 파드 스케줄링을 연습하십시오:
1. 두 노드에 레이블을 지정합니다: 하나는 `tier=frontend`, 다른 하나는 `tier=backend`
2. `tier=frontend`을 요구하는 노드 어피니티가 있는 Deployment를 생성합니다
3. 노드 간에 파드를 분산하는 파드 안티 어피니티가 있는 다른 Deployment를 생성합니다
4. 한 노드에 `maintenance=true:NoSchedule` 테인트를 적용합니다
5. 새 파드가 테인트된 노드에 스케줄링되지 않는 것을 확인합니다
6. 파드에 톨러레이션을 추가하고 테인트된 노드에 스케줄링될 수 있는지 확인합니다

### 연습 문제 5: ConfigMap과 Secret 교체

구성 관리를 연습하십시오:
1. 애플리케이션 설정으로 ConfigMap을 생성하고 볼륨으로 마운트합니다
2. 데이터베이스 자격 증명으로 Secret을 생성하고 환경 변수로 주입합니다
3. ConfigMap을 업데이트하고 볼륨 마운트된 구성이 자동으로 업데이트되는 것을 관찰합니다 (최대 60초 소요될 수 있음)
4. Secret을 업데이트하고 환경 변수 변경은 파드 재시작이 필요한 것을 관찰합니다
5. 새 Secret 값을 적용하기 위해 롤링 재시작을 수행합니다

---

**이전**: [구성 관리](./07_Configuration_Management.md) | [개요](00_Overview.md) | **다음**: [서비스 메시와 네트워킹](./09_Service_Mesh_and_Networking.md)

**License**: CC BY-NC 4.0
