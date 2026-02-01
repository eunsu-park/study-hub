# Kubernetes 입문

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
컨테이너가 100개일 때...
- 어떤 서버에 배포해야 하나?
- 컨테이너가 죽으면 누가 다시 시작하나?
- 트래픽이 늘면 어떻게 확장하나?
- 새 버전 배포 중 다운타임은?
```

**Kubernetes 해결책:**
```
- 자동 스케줄링: 최적의 노드에 배치
- 자가 치유: 장애 시 자동 복구
- 자동 스케일링: 부하에 따라 확장/축소
- 롤링 업데이트: 무중단 배포
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
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: nginx
      image: nginx:alpine
      ports:
        - containerPort: 80
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
  replicas: 3                    # Pod 3개 유지
  selector:
    matchLabels:
      app: my-app
  template:                      # Pod 템플릿
    metadata:
      labels:
        app: my-app
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
    app: my-app                  # 이 라벨의 Pod로 트래픽 전달
  ports:
    - port: 80                   # Service 포트
      targetPort: 80             # Pod 포트
  type: ClusterIP                # 서비스 타입
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
# 클러스터 시작
minikube start

# 상태 확인
minikube status

# 대시보드 열기
minikube dashboard

# 클러스터 중지
minikube stop

# 클러스터 삭제
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
# 모든 Pod 조회
kubectl get pods

# 모든 리소스 조회
kubectl get all

# 상세 정보
kubectl get pods -o wide

# YAML 형식으로 출력
kubectl get pod my-pod -o yaml

# 네임스페이스 지정
kubectl get pods -n kube-system
```

### 리소스 생성/삭제

```bash
# YAML 파일로 생성
kubectl apply -f deployment.yaml

# 삭제
kubectl delete -f deployment.yaml

# 이름으로 삭제
kubectl delete pod my-pod
kubectl delete deployment my-deployment
```

### 상세 정보

```bash
# 리소스 상세 정보
kubectl describe pod my-pod
kubectl describe deployment my-deployment

# 로그 확인
kubectl logs my-pod
kubectl logs -f my-pod              # 실시간

# 컨테이너 접속
kubectl exec -it my-pod -- /bin/sh
```

### 스케일링

```bash
# 복제본 수 변경
kubectl scale deployment my-deployment --replicas=5
```

---

## 6. 실습 예제

### 예제 1: 첫 번째 Pod 실행

```bash
# 1. Pod 직접 실행
kubectl run nginx-pod --image=nginx:alpine

# 2. 확인
kubectl get pods

# 3. 상세 정보
kubectl describe pod nginx-pod

# 4. 로그 확인
kubectl logs nginx-pod

# 5. 삭제
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
  replicas: 3
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello
    spec:
      containers:
        - name: hello
          image: nginxdemos/hello
          ports:
            - containerPort: 80
```

```bash
# 1. Deployment 생성
kubectl apply -f deployment.yaml

# 2. 확인
kubectl get deployments
kubectl get pods

# 3. Pod 하나 삭제해보기 (자동 복구 확인)
kubectl delete pod <pod-name>
kubectl get pods  # 새 Pod가 생성됨

# 4. 스케일 업
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
    app: hello
  ports:
    - port: 80
      targetPort: 80
  type: NodePort
```

```bash
# 1. Service 생성
kubectl apply -f service.yaml

# 2. 확인
kubectl get services

# 3. minikube에서 접근
minikube service hello-service

# 또는 포트 포워딩
kubectl port-forward service/hello-service 8080:80
# http://localhost:8080 에서 확인
```

### 예제 4: 전체 애플리케이션 (Node.js + MongoDB)

**app-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: node-app
spec:
  replicas: 2
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
    - port: 80
      targetPort: 3000
  type: NodePort
```

**mongo-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongo
spec:
  replicas: 1
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
              mountPath: /data/db
      volumes:
        - name: mongo-storage
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: mongo-service
spec:
  selector:
    app: mongo
  ports:
    - port: 27017
      targetPort: 27017
```

```bash
# 1. MongoDB 배포
kubectl apply -f mongo-deployment.yaml

# 2. Node.js 앱 배포
kubectl apply -f app-deployment.yaml

# 3. 확인
kubectl get all

# 4. 접속
minikube service node-service
```

---

## 7. 롤링 업데이트

### 업데이트 적용

```bash
# 이미지 업데이트
kubectl set image deployment/hello-app hello=nginxdemos/hello:latest

# 또는 YAML 수정 후
kubectl apply -f deployment.yaml
```

### 업데이트 상태 확인

```bash
# 롤아웃 상태
kubectl rollout status deployment/hello-app

# 히스토리
kubectl rollout history deployment/hello-app
```

### 롤백

```bash
# 이전 버전으로 롤백
kubectl rollout undo deployment/hello-app

# 특정 버전으로 롤백
kubectl rollout undo deployment/hello-app --to-revision=2
```

---

## 8. ConfigMap과 Secret

### ConfigMap - 설정 데이터

```yaml
# configmap.yaml
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
            name: app-config
```

### Secret - 민감한 데이터

```bash
# Secret 생성
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123
```

```yaml
# YAML로 생성 (base64 인코딩 필요)
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
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
              key: password
```

---

## 9. 네임스페이스

리소스를 논리적으로 분리합니다.

```bash
# 네임스페이스 생성
kubectl create namespace dev
kubectl create namespace prod

# 특정 네임스페이스에 배포
kubectl apply -f deployment.yaml -n dev

# 기본 네임스페이스 변경
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
