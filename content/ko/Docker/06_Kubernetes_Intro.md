# Kubernetes 입문

**이전**: [실전 예제](./05_Practical_Examples.md) | **다음**: [Kubernetes 보안](./07_Kubernetes_Security.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Kubernetes가 무엇인지, 그리고 대규모 환경에서 컨테이너 오케스트레이션(container orchestration)이 왜 필요한지 설명할 수 있다
2. 컨트롤 플레인(Control Plane)과 노드(Node) 구성 요소를 포함한 Kubernetes 클러스터 아키텍처를 설명할 수 있다
3. YAML 매니페스트(manifest)를 사용하여 Pod, Deployment, Service를 정의하고 생성할 수 있다
4. minikube와 kubectl을 사용해 로컬 Kubernetes 환경을 구성할 수 있다
5. 기본 kubectl 명령어를 활용하여 리소스를 생성, 조회, 스케일링, 삭제할 수 있다
6. 무중단 배포를 위한 롤링 업데이트(rolling update)와 롤백(rollback)을 구현할 수 있다
7. ConfigMap과 Secret을 사용하여 애플리케이션 설정과 민감한 데이터를 관리할 수 있다
8. 논리적 격리를 위해 네임스페이스(Namespace)로 리소스를 구성할 수 있다

---

Docker는 단일 머신에서 컨테이너를 실행하는 데 탁월하지만, 프로덕션 시스템은 일반적으로 여러 서버에 걸쳐 있으며 자동 스케줄링, 자가 치유(self-healing), 로드 밸런싱, 롤링 업데이트가 필요합니다. Kubernetes는 이 모든 과제를 해결하는 업계 표준 플랫폼입니다. Docker를 마스터한 다음 단계로 Kubernetes 기초를 배우는 것은 자연스러운 흐름이며, 현대 클라우드 네이티브 애플리케이션의 대부분을 구동하는 확장 가능하고 안정적인 인프라로의 문을 열어줍니다.

> **비유 — 공항 관제탑:** Kubernetes를 공항 관제탑으로 생각해보세요. 개별 Docker 컨테이너는 항공기와 같습니다 — 각각은 자체 화물을 싣고 독립적으로 운항할 수 있습니다. 하지만 수백 편의 항공편이 있다면, 각 비행기가 어느 활주로(노드)에 착륙할지 결정하고, 활주로가 폐쇄되었을 때 트래픽을 우회시키며(자가 치유), 피크 시간대에 더 많은 게이트를 추가하고(자동 스케일링), 교대 근무 간의 원활한 전환(롤링 업데이트)을 보장하는 관제탑(Kubernetes)이 필요합니다.

## 1. Kubernetes란?

Kubernetes(K8s)는 **컨테이너 오케스트레이션 플랫폼**입니다. 여러 컨테이너의 배포, 확장, 관리를 자동화합니다.

### Docker vs Kubernetes

| Docker | Kubernetes |
|--------|------------|
| 컨테이너 실행 | 컨테이너 관리/오케스트레이션 |
| 단일 호스트 | 클러스터 (여러 서버) |
| 수동 스케일링 | 자동 스케일링 |
| 단순 배포 | 롤링 업데이트, 롤백 |

### 왜 Kubernetes가 필요한가?

**문제 상황:**
```
When you have 100 containers...
- Which server should they be deployed to?
- Who restarts containers when they die?
- How to scale when traffic increases?
- Downtime during new version deployment?
```

**Kubernetes 해결책:**
```
- Auto-scheduling: Deploy to optimal nodes
- Self-healing: Automatic recovery on failure
- Auto-scaling: Scale up/down based on load
- Rolling updates: Zero-downtime deployment
```

---

## 2. Kubernetes 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   Control Plane                         │ │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ │ │
│  │  │ API     │ │ Scheduler│ │ Controller│ │   etcd    │ │ │
│  │  │ Server  │ │          │ │  Manager  │ │           │ │ │
│  │  └─────────┘ └──────────┘ └───────────┘ └───────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│           ┌────────────────┼────────────────┐               │
│           │                │                │               │
│           ▼                ▼                ▼               │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐          │
│  │   Node 1   │   │   Node 2   │   │   Node 3   │          │
│  │ ┌────────┐ │   │ ┌────────┐ │   │ ┌────────┐ │          │
│  │ │ kubelet│ │   │ │ kubelet│ │   │ │ kubelet│ │          │
│  │ ├────────┤ │   │ ├────────┤ │   │ ├────────┤ │          │
│  │ │  Pod   │ │   │ │  Pod   │ │   │ │  Pod   │ │          │
│  │ │  Pod   │ │   │ │  Pod   │ │   │ │  Pod   │ │          │
│  │ └────────┘ │   │ └────────┘ │   │ └────────┘ │          │
│  └────────────┘   └────────────┘   └────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 주요 구성 요소

| 구성 요소 | 역할 |
|-----------|------|
| **API Server** | 모든 요청을 처리하는 중앙 게이트웨이 |
| **Scheduler** | Pod를 어느 Node에 배치할지 결정 |
| **Controller Manager** | 원하는 상태 유지 (복제, 배포 등) |
| **etcd** | 클러스터 상태 저장소 |
| **kubelet** | 각 Node에서 컨테이너 실행 관리 |
| **kube-proxy** | 네트워크 프록시, 서비스 로드밸런싱 |

---

## 3. 핵심 개념

### Pod

- Kubernetes의 **최소 배포 단위**
- 하나 이상의 컨테이너 포함
- 같은 Pod의 컨테이너는 네트워크/스토리지 공유

```yaml
# pod.yaml — rarely created directly; use Deployments instead for self-healing and scaling
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: nginx
      image: nginx:alpine       # Alpine: ~5 MB base — smaller attack surface
      ports:
        - containerPort: 80     # Informational; actual exposure requires a Service
```

### Deployment

- Pod의 **선언적 배포 관리**
- 복제본 수 관리 (ReplicaSet)
- 롤링 업데이트, 롤백 지원

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3                    # Maintain 3 Pods — K8s auto-replaces any that crash or get evicted
  selector:
    matchLabels:
      app: my-app               # Must match template labels — this is how the Deployment finds its Pods
  template:                      # Pod template
    metadata:
      labels:
        app: my-app             # Labels connect Deployments → Pods → Services (the glue of K8s)
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
```

### Service

- Pod에 대한 **네트워크 접근점**
- 로드밸런싱
- Pod가 바뀌어도 일관된 접근 제공

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app                  # Route traffic to Pods with this label — decouples routing from Pod IPs
  ports:
    - port: 80                   # Port other services use to reach this Service
      targetPort: 80             # Port the container actually listens on
  type: ClusterIP                # Internal only (default) — use NodePort or LoadBalancer for external access
```

### Service 타입

| 타입 | 설명 |
|------|------|
| `ClusterIP` | 클러스터 내부에서만 접근 (기본값) |
| `NodePort` | 각 Node의 포트로 외부 접근 |
| `LoadBalancer` | 클라우드 로드밸런서 연결 |

---

## 4. 로컬 환경 설정

### minikube 설치

로컬에서 Kubernetes를 실행하는 도구입니다.

**macOS:**
```bash
brew install minikube
```

**Windows (Chocolatey):**
```bash
choco install minikube
```

**Linux:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### minikube 시작

```bash
# Start cluster
minikube start

# Check status
minikube status

# Open dashboard
minikube dashboard

# Stop cluster
minikube stop

# Delete cluster
minikube delete
```

### kubectl 설치

Kubernetes 클러스터와 통신하는 CLI 도구입니다.

**macOS:**
```bash
brew install kubectl
```

**Windows:**
```bash
choco install kubernetes-cli
```

**확인:**
```bash
kubectl version --client
```

---

## 5. kubectl 기본 명령어

### 리소스 조회

```bash
# View all Pods
kubectl get pods

# View all resources
kubectl get all

# Detailed information
kubectl get pods -o wide

# Output in YAML format
kubectl get pod my-pod -o yaml

# Specify namespace
kubectl get pods -n kube-system
```

### 리소스 생성/삭제

```bash
# Create from YAML file
kubectl apply -f deployment.yaml

# Delete
kubectl delete -f deployment.yaml

# Delete by name
kubectl delete pod my-pod
kubectl delete deployment my-deployment
```

### 상세 정보

```bash
# Resource details
kubectl describe pod my-pod
kubectl describe deployment my-deployment

# View logs
kubectl logs my-pod
kubectl logs -f my-pod              # Real-time

# Access container
kubectl exec -it my-pod -- /bin/sh
```

### 스케일링

```bash
# Change replica count
kubectl scale deployment my-deployment --replicas=5
```

---

## 6. 실습 예제

### 예제 1: 첫 번째 Pod 실행

```bash
# 1. Run Pod directly
kubectl run nginx-pod --image=nginx:alpine

# 2. Verify
kubectl get pods

# 3. Detailed information
kubectl describe pod nginx-pod

# 4. Check logs
kubectl logs nginx-pod

# 5. Delete
kubectl delete pod nginx-pod
```

### 예제 2: Deployment로 앱 배포

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-app
spec:
  replicas: 3                    # 3 replicas — K8s distributes them across nodes for high availability
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello               # Labels tie Deployment → ReplicaSet → Pods → Service together
    spec:
      containers:
        - name: hello
          image: nginxdemos/hello
          ports:
            - containerPort: 80
```

```bash
# 1. Create Deployment
kubectl apply -f deployment.yaml

# 2. Verify
kubectl get deployments
kubectl get pods

# 3. Delete one Pod (verify auto-recovery)
kubectl delete pod <pod-name>
kubectl get pods  # New Pod created

# 4. Scale up
kubectl scale deployment hello-app --replicas=5
kubectl get pods
```

### 예제 3: Service로 노출

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hello-service
spec:
  selector:
    app: hello                   # Matches Deployment's Pod labels — Service auto-discovers matching Pods
  ports:
    - port: 80
      targetPort: 80
  type: NodePort                 # Allocates a high port (30000-32767) on every node for external access
```

```bash
# 1. Create Service
kubectl apply -f service.yaml

# 2. Verify
kubectl get services

# 3. Access on minikube
minikube service hello-service

# Or port forwarding
kubectl port-forward service/hello-service 8080:80
# Access at http://localhost:8080
```

### 예제 4: 전체 애플리케이션 (Node.js + MongoDB)

**app-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: node-app
spec:
  replicas: 2                     # 2 replicas for basic high availability
  selector:
    matchLabels:
      app: node-app
  template:
    metadata:
      labels:
        app: node-app
    spec:
      containers:
        - name: node
          image: node:18-alpine
          command: ["node", "-e", "require('http').createServer((req,res)=>{res.end('Hello K8s!')}).listen(3000)"]
          ports:
            - containerPort: 3000
          env:
            - name: MONGO_URL
              # K8s DNS resolves 'mongo-service' to the MongoDB Service's ClusterIP
              value: "mongodb://mongo-service:27017/mydb"
---
apiVersion: v1
kind: Service
metadata:
  name: node-service
spec:
  selector:
    app: node-app
  ports:
    - port: 80                    # External-facing port
      targetPort: 3000            # Container's actual listening port — Service bridges the difference
  type: NodePort
```

**mongo-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongo
spec:
  replicas: 1                     # Single replica — databases typically use StatefulSets for production
  selector:
    matchLabels:
      app: mongo
  template:
    metadata:
      labels:
        app: mongo
    spec:
      containers:
        - name: mongo
          image: mongo:6
          ports:
            - containerPort: 27017
          volumeMounts:
            - name: mongo-storage
              mountPath: /data/db        # MongoDB's default data directory
      volumes:
        - name: mongo-storage
          emptyDir: {}                   # emptyDir: data lost when Pod is deleted — use PersistentVolume for production
---
apiVersion: v1
kind: Service
metadata:
  name: mongo-service              # Other Pods reach MongoDB via this DNS name
spec:
  selector:
    app: mongo
  ports:
    - port: 27017
      targetPort: 27017
  # type defaults to ClusterIP — MongoDB should not be exposed outside the cluster
```

```bash
# 1. Deploy MongoDB
kubectl apply -f mongo-deployment.yaml

# 2. Deploy Node.js app
kubectl apply -f app-deployment.yaml

# 3. Verify
kubectl get all

# 4. Access
minikube service node-service
```

---

## 7. 롤링 업데이트

### 업데이트 적용

```bash
# Update image
kubectl set image deployment/hello-app hello=nginxdemos/hello:latest

# Or modify YAML then apply
kubectl apply -f deployment.yaml
```

### 업데이트 상태 확인

```bash
# Rollout status
kubectl rollout status deployment/hello-app

# History
kubectl rollout history deployment/hello-app
```

### 롤백

```bash
# Rollback to previous version
kubectl rollout undo deployment/hello-app

# Rollback to specific version
kubectl rollout undo deployment/hello-app --to-revision=2
```

---

## 8. ConfigMap과 Secret

### ConfigMap - 설정 데이터

```yaml
# configmap.yaml — externalizes config from the image so the same image works in dev/staging/prod
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "db-service"
  LOG_LEVEL: "info"
```

**Deployment에서 사용:**
```yaml
spec:
  containers:
    - name: app
      envFrom:
        - configMapRef:
            name: app-config   # Injects all keys as env vars — avoids listing each one individually
```

### Secret - 민감한 데이터

```bash
# Create Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123
```

```yaml
# Create with YAML (requires base64 encoding — not encryption; use RBAC to restrict access)
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque                # Opaque = generic key-value; K8s also supports TLS and docker-registry types
data:
  username: YWRtaW4=      # echo -n 'admin' | base64
  password: c2VjcmV0MTIz  # echo -n 'secret123' | base64
```

**Deployment에서 사용:**
```yaml
spec:
  containers:
    - name: app
      env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password    # Inject a single key — safer than envFrom which exposes all keys
```

---

## 9. 네임스페이스

리소스를 논리적으로 분리합니다.

```bash
# Namespaces provide logical isolation — same resource names can exist in different namespaces
kubectl create namespace dev
kubectl create namespace prod

# Deploy to specific namespace — keeps dev and prod resources separate in one cluster
kubectl apply -f deployment.yaml -n dev

# Change default namespace — avoids typing -n dev on every subsequent command
kubectl config set-context --current --namespace=dev
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `kubectl get pods` | Pod 목록 |
| `kubectl get all` | 모든 리소스 |
| `kubectl apply -f file.yaml` | 리소스 생성/업데이트 |
| `kubectl delete -f file.yaml` | 리소스 삭제 |
| `kubectl describe pod name` | 상세 정보 |
| `kubectl logs pod-name` | 로그 확인 |
| `kubectl exec -it pod -- sh` | 컨테이너 접속 |
| `kubectl scale deployment name --replicas=N` | 스케일링 |
| `kubectl rollout status` | 배포 상태 |
| `kubectl rollout undo` | 롤백 |

---

## 다음 학습 추천

1. **Ingress**: HTTP 라우팅, SSL 처리
2. **Persistent Volume**: 영구 저장소
3. **Helm**: 패키지 관리자
4. **모니터링**: Prometheus, Grafana
5. **서비스 메시**: Istio, Linkerd

### 추가 학습 자료

- [Kubernetes 공식 문서](https://kubernetes.io/docs/)
- [Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/)
- [Play with Kubernetes](https://labs.play-with-k8s.com/)

---

## 연습 문제

### 연습 1: 첫 번째 Pod와 Deployment(배포) 생성

Kubernetes의 가장 기본적인 리소스를 직접 다뤄봅니다.

1. 로컬 클러스터를 시작합니다: `minikube start`
2. Pod를 명령형으로 실행합니다: `kubectl run nginx-test --image=nginx:alpine`
3. 실행 중인지 확인합니다: `kubectl get pods -w` (Ctrl+C로 감시 중지)
4. Pod를 describe하여 스케줄된 노드(Node)를 확인합니다: `kubectl describe pod nginx-test`
5. 로그를 확인합니다: `kubectl logs nginx-test`
6. Pod를 삭제하고, 재생성되지 않음을 확인합니다 (Deployment와 비교)
7. 복제본 2개를 가진 Deployment를 생성합니다: `kubectl create deployment web --image=nginx:alpine --replicas=2`
8. Pod 중 하나를 삭제하고, Kubernetes가 자동으로 대체 Pod를 생성하는지 확인합니다

### 연습 2: Service(서비스)로 Deployment 노출

워크로드(workload)를 외부에 노출하는 주요 Service 유형을 실습합니다.

1. Deployment를 생성합니다: `kubectl create deployment hello --image=nginxdemos/hello --replicas=3`
2. ClusterIP Service로 노출합니다: `kubectl expose deployment hello --port=80 --type=ClusterIP`
3. Service가 생성되었는지 확인합니다: `kubectl get svc hello`
4. 포트 포워딩(port-forwarding)으로 로컬에서 접근합니다: `kubectl port-forward svc/hello 8080:80`
5. 브라우저에서 `http://localhost:8080`을 열고, 표시되는 호스트명이 Pod마다 달라지는 것을 확인합니다
6. Service 유형을 NodePort로 변경합니다: `kubectl patch svc hello -p '{"spec":{"type":"NodePort"}}'`
7. minikube를 통해 접근합니다: `minikube service hello --url`로 URL을 확인하고 테스트합니다

### 연습 3: 롤링 업데이트(Rolling Update)와 롤백(Rollback)

무중단 배포와 롤백을 실습합니다.

1. 구버전 이미지로 Deployment를 생성합니다: `kubectl create deployment app --image=nginxdemos/hello:plain-text`
2. 롤아웃(rollout) 상태를 확인합니다: `kubectl rollout status deployment/app`
3. 새 이미지로 업데이트합니다: `kubectl set image deployment/app hello=nginx:1.25`
4. 롤링 업데이트 과정을 실시간으로 관찰합니다: `kubectl get pods -w`
5. 롤아웃 기록을 확인합니다: `kubectl rollout history deployment/app`
6. 이전 버전으로 롤백합니다: `kubectl rollout undo deployment/app`
7. `kubectl rollout status deployment/app`으로 롤백 성공 여부를 확인합니다

### 연습 4: ConfigMap과 Secret(시크릿)

Kubernetes 기본 메커니즘을 사용하여 설정과 민감한 데이터를 저장합니다.

1. 두 개의 키를 가진 ConfigMap을 생성합니다:
   ```bash
   kubectl create configmap app-config \
     --from-literal=LOG_LEVEL=info \
     --from-literal=APP_PORT=8080
   ```
2. 생성 여부를 확인합니다: `kubectl get configmap app-config -o yaml`
3. Secret을 생성합니다:
   ```bash
   kubectl create secret generic db-secret \
     --from-literal=username=admin \
     --from-literal=password=supersecret
   ```
4. ConfigMap을 환경 변수로, Secret을 `/secrets` 경로의 볼륨 마운트로 사용하는 Pod 매니페스트(manifest)를 작성합니다
5. 매니페스트를 적용하고, Pod에 exec로 접속하여 값을 확인합니다: `env | grep -E "LOG_LEVEL|APP_PORT"` 및 `cat /secrets/password`
6. ConfigMap을 수정합니다: `kubectl edit configmap app-config` (`LOG_LEVEL`을 `debug`로 변경)
7. 볼륨 마운트된 ConfigMap이 자동으로 업데이트되는 것을 관찰합니다 (약 1분 소요될 수 있음)

### 연습 5: 네임스페이스(Namespace)와 멀티 환경 구성

단일 클러스터에서 격리된 환경을 시뮬레이션합니다.

1. 두 개의 네임스페이스를 생성합니다: `kubectl create namespace dev` 및 `kubectl create namespace prod`
2. 동일한 애플리케이션을 두 네임스페이스에 배포합니다:
   ```bash
   kubectl create deployment web --image=nginx:alpine -n dev
   kubectl create deployment web --image=nginx:alpine -n prod
   ```
3. 각각 다른 복제본 수로 스케일링합니다: `dev`는 1개, `prod`는 3개
4. 모든 네임스페이스의 Pod를 조회합니다: `kubectl get pods --all-namespaces`
5. 기본 컨텍스트(context)를 `dev`로 전환합니다: `kubectl config set-context --current --namespace=dev`
6. `kubectl get pods`를 실행하여 `dev` 네임스페이스의 Pod만 표시되는지 확인합니다
7. 두 네임스페이스를 정리합니다: `kubectl delete namespace dev prod`

---

**이전**: [실전 예제](./05_Practical_Examples.md) | **다음**: [Kubernetes 보안](./07_Kubernetes_Security.md)
