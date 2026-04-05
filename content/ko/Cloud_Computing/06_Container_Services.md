# 컨테이너 서비스 (ECS/EKS/Fargate vs GKE/Cloud Run)

**이전**: [서버리스 함수](./05_Serverless_Functions.md) | **다음**: [객체 스토리지](./07_Object_Storage.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컨테이너가 가상 머신과 리소스 격리 및 효율성 측면에서 어떻게 다른지 설명할 수 있다
2. AWS 컨테이너 서비스(ECS, EKS, Fargate)와 GCP 동등 서비스(GKE, Cloud Run)를 비교할 수 있다
3. 관리형 오케스트레이션 서비스를 사용해 컨테이너화된 애플리케이션을 배포할 수 있다
4. 서버리스 컨테이너(Fargate, Cloud Run)와 자체 관리 클러스터의 차이를 구분할 수 있다
5. 이미지 저장을 위해 컨테이너 레지스트리(ECR, Artifact Registry)를 설정할 수 있다
6. 확장, 네트워킹, 서비스 디스커버리를 고려한 컨테이너 배포 전략을 설계할 수 있다

---

컨테이너(Container)는 현대 애플리케이션 배포의 표준 단위가 되었습니다. 코드와 의존성을 이식 가능한 이미지로 패키징하여 개발, 스테이징, 프로덕션 환경에서 동일하게 실행됩니다. 클라우드 제공자는 클러스터 관리를 추상화하는 관리형 컨테이너 서비스를 제공하여, 팀이 인프라 운영 대신 소프트웨어 개발과 배포에 집중할 수 있게 합니다.

## 1. 컨테이너 개요

### 1.1 컨테이너 vs VM

```
┌─────────────────────────────────────────────────────────────┐
│                     가상 머신 (VM)                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │Guest OS │ │Guest OS │ │Guest OS │  ← 각 VM마다 OS        │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Hypervisor                           ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       컨테이너                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │  Libs   │ │  Libs   │ │  Libs   │  ← 라이브러리만        │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Container Runtime                      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 서비스 비교

| 항목 | AWS | GCP |
|------|-----|-----|
| **컨테이너 레지스트리** | ECR | Artifact Registry |
| **컨테이너 오케스트레이션** | ECS | - |
| **Kubernetes 관리형** | EKS | GKE |
| **서버리스 컨테이너** | Fargate | Cloud Run |
| **App Platform** | App Runner | Cloud Run |

---

## 2. 컨테이너 레지스트리

### 2.1 AWS ECR (Elastic Container Registry)

```bash
# 1. ECR 레포지토리 생성
aws ecr create-repository \
    --repository-name my-app \
    --region ap-northeast-2

# 2. Docker 로그인
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com

# 3. 이미지 빌드 및 태그
docker build -t my-app .
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 4. 이미지 푸시
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 5. 이미지 목록 확인
aws ecr list-images --repository-name my-app
```

### 2.2 GCP Artifact Registry

```bash
# 1. Artifact Registry API 활성화
gcloud services enable artifactregistry.googleapis.com

# 2. 레포지토리 생성
gcloud artifacts repositories create my-repo \
    --repository-format=docker \
    --location=asia-northeast3 \
    --description="My Docker repository"

# 3. Docker 인증 설정
gcloud auth configure-docker asia-northeast3-docker.pkg.dev

# 4. 이미지 빌드 및 태그
docker build -t my-app .
docker tag my-app:latest \
    asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest

# 5. 이미지 푸시
docker push asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest

# 6. 이미지 목록 확인
gcloud artifacts docker images list \
    asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo
```

---

## 3. AWS ECS (Elastic Container Service)

### 3.1 ECS 개념

```
┌─────────────────────────────────────────────────────────────┐
│                        ECS Cluster                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                      Service                            ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    Task       │  │    Task       │  ← 컨테이너 그룹   ││
│  │  │ ┌───────────┐ │  │ ┌───────────┐ │                   ││
│  │  │ │ Container │ │  │ │ Container │ │                   ││
│  │  │ └───────────┘ │  │ └───────────┘ │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │ EC2 Instance      │  │ Fargate           │               │
│  │ (자체 관리)       │  │ (서버리스)        │               │
│  └───────────────────┘  └───────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 ECS 클러스터 생성

```bash
# 1. 클러스터 생성 (Fargate)
aws ecs create-cluster \
    --cluster-name my-cluster \
    --capacity-providers FARGATE FARGATE_SPOT

# 2. Task Definition 생성
# task-definition.json
{
    "family": "my-task",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "my-container",
            "image": "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 80,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/my-task",
                    "awslogs-region": "ap-northeast-2",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}

aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. 서비스 생성
aws ecs create-service \
    --cluster my-cluster \
    --service-name my-service \
    --task-definition my-task:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 3.3 ECS Service Connect

ECS Service Connect는 별도의 프록시나 서비스 메시 설정 없이 서비스 간 통신을 위한 내장 서비스 메시 기능을 제공합니다.

```json
// Service Connect가 포함된 서비스 정의
{
    "cluster": "my-cluster",
    "serviceName": "backend-service",
    "taskDefinition": "backend-task:1",
    "serviceConnectConfiguration": {
        "enabled": true,
        "namespace": "my-app-namespace",
        "services": [
            {
                "portName": "http",
                "discoveryName": "backend",
                "clientAliases": [
                    {
                        "port": 80,
                        "dnsName": "backend.local"
                    }
                ]
            }
        ]
    },
    "desiredCount": 2,
    "launchType": "FARGATE",
    "networkConfiguration": {
        "awsvpcConfiguration": {
            "subnets": ["subnet-xxx"],
            "securityGroups": ["sg-xxx"]
        }
    }
}
```

**주요 장점:**
- 내장 서비스 디스커버리 (AWS Cloud Map 통합)
- 서비스 간 자동 로드 밸런싱
- 추가 에이전트 없이 트래픽 메트릭 및 관측성(Observability) 제공
- 외부 서비스 메시(Istio, Consul) 불필요

### 3.4 ECS Exec (컨테이너 디버깅)

ECS Exec을 사용하면 실행 중인 컨테이너에 대화형 셸로 접근하여 디버깅할 수 있습니다.

```bash
# 서비스에서 ECS Exec 활성화
aws ecs update-service \
    --cluster my-cluster \
    --service my-service \
    --enable-execute-command

# 대화형 셸 세션 시작
aws ecs execute-command \
    --cluster my-cluster \
    --task TASK_ID \
    --container my-container \
    --interactive \
    --command "/bin/sh"

# 일회성 명령 실행
aws ecs execute-command \
    --cluster my-cluster \
    --task TASK_ID \
    --container my-container \
    --command "cat /app/config.json"
```

> **참고:** ECS Exec을 사용하려면 태스크 역할에 `ssmmessages` 권한이 필요하며, 태스크 정의에 `initProcessEnabled: true`가 포함되어야 합니다.

---

## 4. AWS EKS (Elastic Kubernetes Service)

### 4.1 EKS 클러스터 생성

```bash
# 1. eksctl 설치 (macOS)
brew install eksctl

# 2. 클러스터 생성
eksctl create cluster \
    --name my-cluster \
    --region ap-northeast-2 \
    --nodegroup-name my-nodes \
    --node-type t3.medium \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 4

# 3. kubeconfig 업데이트
aws eks update-kubeconfig --name my-cluster --region ap-northeast-2

# 4. 클러스터 확인
kubectl get nodes
```

### 4.2 EKS Auto Mode

EKS Auto Mode(2024년 말 출시)는 GKE Autopilot과 유사하게 노드 관리를 자동화하여 EKS 운영을 단순화합니다.

```bash
# Auto Mode로 EKS 클러스터 생성
eksctl create cluster \
    --name my-auto-cluster \
    --region ap-northeast-2 \
    --auto-mode

# 또는 기존 클러스터에 Auto Mode 활성화
aws eks update-cluster-config \
    --name my-cluster \
    --compute-config enabled=true \
    --kubernetes-network-config '{"elasticLoadBalancing":{"enabled":true}}' \
    --storage-config '{"blockStorage":{"enabled":true}}'
```

| 기능 | EKS Standard | EKS Auto Mode |
|------|-------------|---------------|
| **노드 프로비저닝** | 수동 (관리형 노드 그룹 또는 Karpenter) | 자동 |
| **노드 OS 업데이트** | 사용자 관리 | AWS 관리 |
| **로드 밸런서** | AWS LB Controller 설치 필요 | 내장 |
| **스토리지(EBS CSI)** | EBS CSI 드라이버 설치 필요 | 내장 |
| **과금** | EC2 인스턴스 기반 | Pod 리소스 기반 (오버헤드 포함) |
| **적합 대상** | 세밀한 제어 필요 시 | 간소화된 운영 |

### 4.3 애플리케이션 배포

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 80
```

```bash
# 배포
kubectl apply -f deployment.yaml

# 상태 확인
kubectl get pods
kubectl get services
```

---

## 5. GCP GKE (Google Kubernetes Engine)

### 5.1 GKE 클러스터 생성

```bash
# 1. GKE API 활성화
gcloud services enable container.googleapis.com

# 2. 클러스터 생성 (Autopilot - 권장)
gcloud container clusters create-auto my-cluster \
    --region=asia-northeast3

# 또는 Standard 클러스터
gcloud container clusters create my-cluster \
    --region=asia-northeast3 \
    --num-nodes=2 \
    --machine-type=e2-medium

# 3. 클러스터 인증 정보 가져오기
gcloud container clusters get-credentials my-cluster \
    --region=asia-northeast3

# 4. 클러스터 확인
kubectl get nodes
```

### 5.2 GKE Autopilot 심화

GKE Autopilot은 Google이 노드, 스케일링, 보안을 포함한 전체 클러스터 인프라를 관리하는 완전 관리형 Kubernetes 모드입니다.

**Autopilot vs Standard:**

| 항목 | Autopilot | Standard |
|------|-----------|----------|
| **노드 관리** | Google 자동 관리 | 사용자 관리 |
| **과금** | Pod 리소스 기반 | 노드 기반 |
| **보안** | 강화된 기본값 (경량화 OS, Workload Identity, Shielded GKE 노드) | 수동 구성 |
| **확장성** | 자동 HPA/VPA | 수동/자동 구성 |
| **GPU 지원** | 지원 (L4, A100, H100, TPU) | 지원 |
| **Spot Pod** | 지원 | 지원 (선점형 노드) |
| **DaemonSet** | 허용 (과금 포함) | 허용 |
| **특권 Pod** | 불허 | 허용 |
| **적합 대상** | 대부분의 워크로드, 비용 최적화 | 세밀한 제어, 특수 커널 요구 시 |

**Autopilot 보안 기능 (기본 활성화):**
- `containerd` 기반 Container-Optimized OS
- Workload Identity (노드 서비스 계정 키 불필요)
- Shielded GKE 노드 (보안 부팅, 무결성 모니터링)
- 네트워크 정책 적용
- Pod 보안 표준 (기본 Baseline)
- Binary Authorization 지원

```bash
# Autopilot에서 Spot Pod 배포 (최대 60-91% 비용 절감)
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
spec:
  replicas: 5
  selector:
    matchLabels:
      app: batch-processor
  template:
    metadata:
      labels:
        app: batch-processor
    spec:
      nodeSelector:
        cloud.google.com/gke-spot: "true"
      terminationGracePeriodSeconds: 25
      containers:
      - name: worker
        image: asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/worker:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
            # Autopilot에서 GPU 요청
            # nvidia.com/gpu: "1"
          limits:
            cpu: "500m"
            memory: "1Gi"
      tolerations:
      - key: cloud.google.com/gke-spot
        operator: Equal
        value: "true"
        effect: NoSchedule
EOF
```

### 5.3 애플리케이션 배포

```yaml
# deployment.yaml (GKE)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 80
```

```bash
kubectl apply -f deployment.yaml
kubectl get services
```

---

## 6. 서버리스 컨테이너

### 6.1 AWS Fargate

Fargate는 서버 프로비저닝 없이 컨테이너를 실행합니다.

**특징:**
- EC2 인스턴스 관리 불필요
- 태스크 수준에서 리소스 정의
- ECS 또는 EKS와 함께 사용

```bash
# ECS + Fargate로 서비스 생성
aws ecs create-service \
    --cluster my-cluster \
    --service-name my-fargate-service \
    --task-definition my-task:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 6.2 GCP Cloud Run

Cloud Run은 컨테이너를 서버리스로 실행합니다.

**특징:**
- 완전 관리형
- 요청 기반 자동 확장 (0까지)
- 사용한 만큼만 과금
- HTTP 트래픽 또는 이벤트 기반

```bash
# 1. 이미지 배포
gcloud run deploy my-service \
    --image=asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest \
    --region=asia-northeast3 \
    --platform=managed \
    --allow-unauthenticated

# 2. 서비스 URL 확인
gcloud run services describe my-service \
    --region=asia-northeast3 \
    --format='value(status.url)'

# 3. 트래픽 분할 (Blue/Green)
gcloud run services update-traffic my-service \
    --region=asia-northeast3 \
    --to-revisions=my-service-00002-abc=50,my-service-00001-xyz=50
```

### 6.3 Cloud Run vs App Runner 비교

| 항목 | GCP Cloud Run | AWS App Runner |
|------|--------------|----------------|
| **소스** | 컨테이너 이미지, 소스 코드 | 컨테이너 이미지, 소스 코드 |
| **최대 메모리** | 32GB | 12GB |
| **최대 타임아웃** | 60분 | 30분 |
| **0으로 스케일** | 지원 | 지원 (옵션) |
| **VPC 연결** | 지원 | 지원 |
| **GPU** | 지원 | 미지원 |

---

## 7. 서비스 선택 가이드

### 7.1 결정 트리

```
서버리스 컨테이너가 필요한가?
├── Yes → Cloud Run / Fargate / App Runner
│         └── Kubernetes 기능 필요?
│             ├── Yes → Fargate on EKS
│             └── No → Cloud Run (GCP) / Fargate on ECS (AWS)
└── No → Kubernetes가 필요한가?
          ├── Yes → GKE (Autopilot/Standard) / EKS
          └── No → ECS on EC2 / Compute Engine + Docker
```

### 7.2 사용 사례별 권장

| 사용 사례 | AWS 권장 | GCP 권장 |
|----------|---------|---------|
| **단순 웹앱** | App Runner | Cloud Run |
| **마이크로서비스** | ECS Fargate + Service Connect | Cloud Run |
| **K8s (간소화)** | EKS Auto Mode | GKE Autopilot |
| **K8s (전체 제어)** | EKS Standard | GKE Standard |
| **ML/GPU 워크로드** | EKS + GPU | GKE Autopilot + GPU |
| **배치 작업** | ECS Task | Cloud Run Jobs |
| **이벤트 처리** | Fargate + EventBridge | Cloud Run + Eventarc |
| **비용 민감 배치** | Fargate Spot | Autopilot Spot Pod |

---

## 8. 과금 비교

### 8.1 ECS/EKS vs GKE

**AWS ECS (Fargate):**
```
vCPU: $0.04048/시간 (서울)
메모리: $0.004445/GB/시간 (서울)

예: 0.5 vCPU, 1GB, 24시간
= (0.5 × $0.04048 × 24) + (1 × $0.004445 × 24)
= $0.49 + $0.11 = $0.60/일
```

**AWS EKS:**
```
클러스터: $0.10/시간 ($72/월)
+ 노드 비용 (EC2) 또는 Fargate 비용
```

**GCP GKE:**
```
Autopilot: vCPU $0.0445/시간, 메모리 $0.0049/GB/시간
Standard: 관리 수수료 $0.10/시간/클러스터 + 노드 비용

예: Autopilot 0.5 vCPU, 1GB, 24시간
= (0.5 × $0.0445 × 24) + (1 × $0.0049 × 24)
= $0.53 + $0.12 = $0.65/일
```

### 8.2 Cloud Run 과금

```
CPU: $0.00002400/vCPU-초 (요청 처리 중)
메모리: $0.00000250/GB-초
요청: $0.40/100만 요청

무료 티어:
- 200만 요청/월
- 360,000 GB-초
- 180,000 vCPU-초
```

---

## 9. 실습: 간단한 웹앱 배포

### 9.1 Dockerfile 준비

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
```

```python
# app.py
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return f"Hello from {os.environ.get('CLOUD_PROVIDER', 'Container')}!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

```
# requirements.txt
flask==3.0.0
gunicorn==21.2.0
```

### 9.2 GCP Cloud Run 배포

```bash
# 빌드 및 배포 (소스에서 직접)
gcloud run deploy my-app \
    --source=. \
    --region=asia-northeast3 \
    --allow-unauthenticated \
    --set-env-vars=CLOUD_PROVIDER=GCP
```

### 9.3 AWS App Runner 배포

```bash
# 1. ECR에 이미지 푸시 (앞서 설명한 방법)

# 2. App Runner 서비스 생성
aws apprunner create-service \
    --service-name my-app \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8080",
                "RuntimeEnvironmentVariables": {
                    "CLOUD_PROVIDER": "AWS"
                }
            }
        },
        "AuthenticationConfiguration": {
            "AccessRoleArn": "arn:aws:iam::123456789012:role/AppRunnerECRAccessRole"
        }
    }'
```

---

## 10. 다음 단계

- [07_Object_Storage.md](./07_Object_Storage.md) - 객체 스토리지
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC 네트워킹

---

## 연습 문제

### 연습 문제 1: 컨테이너 vs VM 트레이드오프

팀이 모놀리식(monolithic) 웹 애플리케이션을 클라우드로 마이그레이션하고 있습니다. EC2 VM에서 실행하는 것과 Docker로 컨테이너화하여 ECS Fargate에 배포하는 것을 놓고 논의 중입니다.

컨테이너 방식의 장점 두 가지와 VM을 선택하는 것이 더 나은 두 가지 시나리오를 찾아보세요.

<details>
<summary>정답 보기</summary>

**컨테이너(Fargate)의 장점**:
1. **이식성(portability)과 일관성** — Docker 이미지는 애플리케이션과 모든 의존성을 캡슐화합니다. 동일한 이미지가 개발, CI/CD, 프로덕션 환경에서 동일하게 실행되어 "내 컴퓨터에서는 되는데" 문제를 없앱니다.
2. **밀도(density)와 비용 효율성** — 여러 컨테이너가 호스트 OS 커널을 공유하므로, 각각 자체 OS가 있는 VM보다 훨씬 적은 메모리를 사용합니다. Fargate는 태스크(task)가 예약한 CPU/메모리에 대해서만 요금을 청구하며, 유휴 EC2 오버헤드가 없습니다.

**VM(EC2)이 더 나은 시나리오**:
1. **전체 OS 접근 또는 커널 수준 커스터마이징이 필요한 애플리케이션** — 일부 워크로드(예: 커스텀 커널 모듈, 특수 하드웨어 드라이버, `/proc`와 `/sys`를 수정하는 애플리케이션)는 컨테이너가 제공할 수 없는 루트 수준 OS 접근이 필요합니다.
2. **컨테이너화할 수 없는 레거시 애플리케이션** — 특정 OS 버전, 레지스트리 항목(Windows), COM 객체에 대한 강한 의존성이 있다면, 컨테이너화에 상당한 리팩토링이 필요할 수 있습니다. EC2 VM에서 직접 실행하는 것이 실용적인 마이그레이션 경로일 수 있습니다.

</details>

### 연습 문제 2: 서비스 선택

각 시나리오에 가장 적합한 AWS 컨테이너 서비스(ECS on EC2, ECS Fargate, EKS, App Runner)를 선택하고 이유를 설명하세요:

1. 소규모 스타트업이 인프라 관리 없이 자동 HTTPS로 간단한 컨테이너화된 REST API를 배포하려 합니다.
2. 플랫폼 팀이 50개 이상의 마이크로서비스를 관리하며 고급 네트워킹, 커스텀 어드미션 컨트롤러(admission controller), 멀티 클라우드 이식성이 필요합니다.
3. 머신러닝 팀이 데이터셋을 처리하고 종료하는 GPU 가속 학습 작업을 컨테이너로 실행합니다.
4. 회사가 워크로드에 커스텀 EC2 인스턴스 유형(예: 스토리지 최적화 i3 인스턴스)이 필요합니다.

<details>
<summary>정답 보기</summary>

1. **AWS App Runner** — 인프라 관리 없음: 클러스터, 태스크 정의(task definition), 로드 밸런서 설정이 필요하지 않습니다. App Runner가 HTTPS, 스케일링, 로드 밸런싱, 배포를 자동으로 처리합니다. 단순한 무상태(stateless) 서비스에 이상적입니다.

2. **Amazon EKS** — 쿠버네티스(Kubernetes)는 복잡한 마이크로서비스 플랫폼의 업계 표준입니다. 커스텀 리소스 정의(CRD), 어드미션 웹훅(admission webhook), 파드 보안 정책(pod security policy), 서비스 메시(Istio) 통합은 쿠버네티스 네이티브 기능입니다. 멀티 클라우드 이식성은 쿠버네티스의 강점입니다(동일한 매니페스트가 GKE, AKS에서 실행 가능).

3. **GPU가 있는 ECS Fargate** 또는 **GPU 노드 그룹이 있는 EKS** — 시작, 처리, 종료하는 배치 스타일 작업은 ECS 태스크 또는 쿠버네티스 Job에 적합합니다. GPU 워크로드의 경우 두 플랫폼 모두 GPU 지원 인스턴스를 지원합니다.

4. **ECS on EC2** — Fargate가 지원하지 않는 특정 EC2 인스턴스 유형(i3, c5n 등)이 필요하거나 특정 드라이버가 있는 물리적 하드웨어를 연결해야 할 때, 기본 EC2 노드를 직접 관리해야 합니다. ECS on EC2는 컨테이너 오케스트레이션을 제공하면서 인스턴스 유형에 대한 완전한 제어권을 유지합니다.

</details>

### 연습 문제 3: 컨테이너 레지스트리(Container Registry) 워크플로우

다음 CLI 명령어 순서를 설명하세요:
1. 계정 `123456789012`의 `ap-northeast-2` 리전에서 AWS ECR에 Docker를 인증합니다.
2. 로컬 `Dockerfile`에서 Docker 이미지를 빌드하고 저장소 이름 `my-app`과 태그 `v1.0`으로 ECR에 태그합니다.
3. ECR에 이미지를 푸시합니다.

<details>
<summary>정답 보기</summary>

```bash
# 1단계: Docker를 ECR에 인증
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com

# 2단계: 이미지 빌드 및 태그
docker build -t my-app .
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:v1.0

# (선택 사항) latest 태그도 추가
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 3단계: ECR에 푸시
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:v1.0
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest
```

**참고**: 푸시 전에 ECR 저장소를 생성해야 합니다. 아직 없다면:
```bash
aws ecr create-repository --repository-name my-app --region ap-northeast-2
```

</details>

### 연습 문제 4: Cloud Run 트래픽 분할(Traffic Splitting)

`asia-northeast3` 리전의 `payment-service`라는 Cloud Run 서비스의 새 버전을 배포했습니다. 새 리비전(revision)은 `payment-service-00003-xyz`이고 이전 안정 버전은 `payment-service-00002-abc`입니다.

신규 버전으로 트래픽의 10%, 이전 버전으로 90%를 라우팅하는 `gcloud` 명령어를 작성하고(카나리 배포, canary deployment), 이 패턴이 유용한 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

```bash
gcloud run services update-traffic payment-service \
    --region=asia-northeast3 \
    --to-revisions=payment-service-00003-xyz=10,payment-service-00002-abc=90
```

**이 패턴(카나리 배포)이 유용한 이유**:
- **위험 감소**: 실제 사용자의 10%만 새 버전에 접근합니다. 심각한 버그(충돌, 데이터 손상)가 있더라도 100%가 아닌 10%의 사용자만 영향을 받습니다.
- **실제 트래픽 테스트**: 합성 테스트와 스테이징 환경이 모든 프로덕션 조건을 재현하지 못할 수 있습니다. 카나리 배포는 실제 프로덕션 트래픽 패턴에 새 버전을 노출합니다.
- **점진적 롤아웃**: 10% 카나리가 좋은 지표(오류율, 지연 시간)를 보이면 50%, 100%로 늘릴 수 있습니다. 문제가 발생하면 안정 버전으로 100% 즉시 롤백합니다.

새 버전을 100%로 승격:
```bash
gcloud run services update-traffic payment-service \
    --region=asia-northeast3 \
    --to-latest
```

</details>

### 연습 문제 5: Fargate vs Cloud Run 비교

AWS Fargate와 GCP Cloud Run의 아키텍처 차이를 분석하세요. 초당 ~100개 요청을 처리하며 가끔 10배 급증이 있는 무상태(stateless) HTTP 마이크로서비스의 경우, 어느 것을 선택하겠으며 핵심 운영 차이점은 무엇입니까?

<details>
<summary>정답 보기</summary>

**핵심 아키텍처 차이**:

| 측면 | AWS Fargate | GCP Cloud Run |
|------|-------------|---------------|
| **스케일링 모델** | 최소/최대 태스크 수 설정; 기본적으로 최솟값(0이 아님)으로 스케일 다운 | 0으로 스케일 다운; 요청 기반 자동 스케일링 |
| **시작 시간** | 새 태스크의 컨테이너 콜드 스타트; 보통 30~60초 | 빠른 콜드 스타트; 보통 수 초 |
| **요금 기준** | 태스크 실행 중 vCPU-시간과 GB-시간당 (유휴 상태 포함) | 실제 요청 처리 중 vCPU-초와 GB-초당만 |
| **클러스터 요건** | ECS 클러스터 필요(서버리스이지만 클러스터 설정 필요) | 완전 관리형, 클러스터 개념 없음 |
| **트래픽 분할** | ALB 가중 대상 그룹 필요 | 리비전(revision) 기반 트래픽 분할 내장 |

**이 시나리오에 대한 권장**: **GCP Cloud Run**이 이 워크로드에 더 적합합니다:
1. **비용**: 초당 100건의 요청은 기본 부하가 있지만, Cloud Run의 요청별 과금 방식이 버스티 워크로드에 더 저렴합니다. Fargate는 급증 사이의 유휴 시간에도 과금됩니다.
2. **스케일링**: Cloud Run의 빠른 스케일 아웃이 10배 급증을 잘 처리합니다. 내장된 트래픽 분할로 안전한 배포가 가능합니다.
3. **단순성**: 클러스터 관리, 태스크 정의 버전 관리, ALB 설정이 필요 없습니다.

AWS에서는 **App Runner**가 단순성 측면에서 Cloud Run에 더 가까운 동등물입니다. Fargate는 세밀한 네트워킹(VPC, 보안 그룹, 프라이빗 서브넷), 영구 연결, 또는 다른 ECS 기반 서비스와의 통합이 필요할 때 더 적합합니다.

</details>

---

## 참고 자료

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [GCP GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker/](../Docker/) - Docker 기초
