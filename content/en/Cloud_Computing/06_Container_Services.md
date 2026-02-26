# Container Services (ECS/EKS/Fargate vs GKE/Cloud Run)

**Previous**: [Serverless Functions](./05_Serverless_Functions.md) | **Next**: [Object Storage](./07_Object_Storage.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how containers differ from virtual machines in resource isolation and efficiency
2. Compare AWS container services (ECS, EKS, Fargate) with GCP equivalents (GKE, Cloud Run)
3. Deploy a containerized application using a managed orchestration service
4. Distinguish between serverless containers (Fargate, Cloud Run) and self-managed clusters
5. Configure container registries (ECR, Artifact Registry) for image storage
6. Design a container deployment strategy considering scaling, networking, and service discovery

---

Containers have become the standard unit of deployment for modern applications. They package code and dependencies into a portable image that runs identically across development, staging, and production. Cloud providers offer managed container services that abstract away cluster management, letting teams focus on building and shipping software rather than operating infrastructure.

## 1. Container Overview

### 1.1 Container vs VM

```
┌─────────────────────────────────────────────────────────────┐
│                   Virtual Machine (VM)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │Guest OS │ │Guest OS │ │Guest OS │  ← OS per VM          │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Hypervisor                           ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Container                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │  Libs   │ │  Libs   │ │  Libs   │  ← Libraries only     │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Container Runtime                      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Service Comparison

| Category | AWS | GCP |
|------|-----|-----|
| **Container Registry** | ECR | Artifact Registry |
| **Container Orchestration** | ECS | - |
| **Managed Kubernetes** | EKS | GKE |
| **Serverless Containers** | Fargate | Cloud Run |
| **App Platform** | App Runner | Cloud Run |

---

## 2. Container Registry

### 2.1 AWS ECR (Elastic Container Registry)

```bash
# 1. Create ECR repository
aws ecr create-repository \
    --repository-name my-app \
    --region ap-northeast-2

# 2. Docker login
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com

# 3. Build and tag image
docker build -t my-app .
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 4. Push image
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 5. List images
aws ecr list-images --repository-name my-app
```

### 2.2 GCP Artifact Registry

```bash
# 1. Enable Artifact Registry API
gcloud services enable artifactregistry.googleapis.com

# 2. Create repository
gcloud artifacts repositories create my-repo \
    --repository-format=docker \
    --location=asia-northeast3 \
    --description="My Docker repository"

# 3. Configure Docker authentication
gcloud auth configure-docker asia-northeast3-docker.pkg.dev

# 4. Build and tag image
docker build -t my-app .
docker tag my-app:latest \
    asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest

# 5. Push image
docker push asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest

# 6. List images
gcloud artifacts docker images list \
    asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo
```

---

## 3. AWS ECS (Elastic Container Service)

### 3.1 ECS Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                        ECS Cluster                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                      Service                            ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    Task       │  │    Task       │  ← Container group││
│  │  │ ┌───────────┐ │  │ ┌───────────┐ │                   ││
│  │  │ │ Container │ │  │ │ Container │ │                   ││
│  │  │ └───────────┘ │  │ └───────────┘ │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │ EC2 Instance      │  │ Fargate           │               │
│  │ (self-managed)    │  │ (serverless)      │               │
│  └───────────────────┘  └───────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Creating an ECS Cluster

```bash
# 1. Create cluster (Fargate)
aws ecs create-cluster \
    --cluster-name my-cluster \
    --capacity-providers FARGATE FARGATE_SPOT

# 2. Create Task Definition
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

# 3. Create service
aws ecs create-service \
    --cluster my-cluster \
    --service-name my-service \
    --task-definition my-task:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 3.3 ECS Service Connect

ECS Service Connect provides built-in service mesh capabilities for service-to-service communication without requiring a separate proxy or service mesh setup.

```json
// Service definition with Service Connect
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

**Key Benefits:**
- Built-in service discovery (AWS Cloud Map integration)
- Automatic load balancing between services
- Traffic metrics and observability without additional agents
- No need for external service mesh (Istio, Consul)

### 3.4 ECS Exec (Container Debugging)

ECS Exec allows interactive shell access to running containers for debugging.

```bash
# Enable ECS Exec on a service
aws ecs update-service \
    --cluster my-cluster \
    --service my-service \
    --enable-execute-command

# Start an interactive shell session
aws ecs execute-command \
    --cluster my-cluster \
    --task TASK_ID \
    --container my-container \
    --interactive \
    --command "/bin/sh"

# Run a one-off command
aws ecs execute-command \
    --cluster my-cluster \
    --task TASK_ID \
    --container my-container \
    --command "cat /app/config.json"
```

> **Note:** ECS Exec requires the task role to have `ssmmessages` permissions and the task definition must include `initProcessEnabled: true`.

---

## 4. AWS EKS (Elastic Kubernetes Service)

### 4.1 Creating an EKS Cluster

```bash
# 1. Install eksctl (macOS)
brew install eksctl

# 2. Create cluster
eksctl create cluster \
    --name my-cluster \
    --region ap-northeast-2 \
    --nodegroup-name my-nodes \
    --node-type t3.medium \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 4

# 3. Update kubeconfig
aws eks update-kubeconfig --name my-cluster --region ap-northeast-2

# 4. Verify cluster
kubectl get nodes
```

### 4.2 EKS Auto Mode

EKS Auto Mode (launched late 2024) simplifies EKS by automating node management, similar to GKE Autopilot.

```bash
# Create an EKS cluster with Auto Mode
eksctl create cluster \
    --name my-auto-cluster \
    --region ap-northeast-2 \
    --auto-mode

# Or enable Auto Mode on an existing cluster
aws eks update-cluster-config \
    --name my-cluster \
    --compute-config enabled=true \
    --kubernetes-network-config '{"elasticLoadBalancing":{"enabled":true}}' \
    --storage-config '{"blockStorage":{"enabled":true}}'
```

| Feature | EKS Standard | EKS Auto Mode |
|---------|-------------|---------------|
| **Node Provisioning** | Manual (managed node groups or Karpenter) | Automatic |
| **Node OS Updates** | User-managed | AWS-managed |
| **Load Balancer** | Install AWS LB Controller | Built-in |
| **Storage (EBS CSI)** | Install EBS CSI driver | Built-in |
| **Billing** | EC2 instance-based | Pod resource-based (with overhead) |
| **Best For** | Fine-grained control | Simplified operations |

### 4.3 Deploying Applications

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
# Deploy
kubectl apply -f deployment.yaml

# Check status
kubectl get pods
kubectl get services
```

---

## 5. GCP GKE (Google Kubernetes Engine)

### 5.1 Creating a GKE Cluster

```bash
# 1. Enable GKE API
gcloud services enable container.googleapis.com

# 2. Create cluster (Autopilot - recommended)
gcloud container clusters create-auto my-cluster \
    --region=asia-northeast3

# Or Standard cluster
gcloud container clusters create my-cluster \
    --region=asia-northeast3 \
    --num-nodes=2 \
    --machine-type=e2-medium

# 3. Get cluster credentials
gcloud container clusters get-credentials my-cluster \
    --region=asia-northeast3

# 4. Verify cluster
kubectl get nodes
```

### 5.2 GKE Autopilot Deep Dive

GKE Autopilot is a fully managed Kubernetes mode where Google manages the entire cluster infrastructure, including nodes, scaling, and security.

**Autopilot vs Standard:**

| Category | Autopilot | Standard |
|------|-----------|----------|
| **Node Management** | Google auto-managed | User-managed |
| **Billing** | Pod resource-based | Node-based |
| **Security** | Enhanced defaults (hardened OS, Workload Identity, Shielded GKE Nodes) | Manual configuration |
| **Scalability** | Automatic HPA/VPA | Manual/auto configuration |
| **GPU Support** | Supported (L4, A100, H100, TPU) | Supported |
| **Spot Pods** | Supported | Supported (preemptible nodes) |
| **DaemonSets** | Allowed (billing included) | Allowed |
| **Privileged Pods** | Not allowed | Allowed |
| **Best For** | Most workloads, cost optimization | Fine-grained control, special kernel needs |

**Autopilot Security Features (enabled by default):**
- Container-Optimized OS with `containerd`
- Workload Identity (no node service account keys)
- Shielded GKE Nodes (secure boot, integrity monitoring)
- Network policy enforcement
- Pod security standards (Baseline by default)
- Binary Authorization ready

```bash
# Deploy with Spot Pods on Autopilot (cost savings up to 60-91%)
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
            # GPU request on Autopilot
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

### 5.3 Deploying Applications

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

## 6. Serverless Containers

### 6.1 AWS Fargate

Fargate runs containers without server provisioning.

**Features:**
- No EC2 instance management needed
- Define resources at task level
- Use with ECS or EKS

```bash
# Create service with ECS + Fargate
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

Cloud Run runs containers in a serverless manner.

**Features:**
- Fully managed
- Request-based auto-scaling (to zero)
- Pay only for what you use
- HTTP traffic or event-driven

```bash
# 1. Deploy image
gcloud run deploy my-service \
    --image=asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest \
    --region=asia-northeast3 \
    --platform=managed \
    --allow-unauthenticated

# 2. Get service URL
gcloud run services describe my-service \
    --region=asia-northeast3 \
    --format='value(status.url)'

# 3. Traffic splitting (Blue/Green)
gcloud run services update-traffic my-service \
    --region=asia-northeast3 \
    --to-revisions=my-service-00002-abc=50,my-service-00001-xyz=50
```

### 6.3 Cloud Run vs App Runner Comparison

| Category | GCP Cloud Run | AWS App Runner |
|------|--------------|----------------|
| **Source** | Container image, source code | Container image, source code |
| **Max Memory** | 32GB | 12GB |
| **Max Timeout** | 60 minutes | 30 minutes |
| **Scale to Zero** | Supported | Supported (optional) |
| **VPC Connection** | Supported | Supported |
| **GPU** | Supported | Not supported |

---

## 7. Service Selection Guide

### 7.1 Decision Tree

```
Do you need serverless containers?
├── Yes → Cloud Run / Fargate / App Runner
│         └── Need Kubernetes features?
│             ├── Yes → Fargate on EKS
│             └── No → Cloud Run (GCP) / Fargate on ECS (AWS)
└── No → Do you need Kubernetes?
          ├── Yes → GKE (Autopilot/Standard) / EKS
          └── No → ECS on EC2 / Compute Engine + Docker
```

### 7.2 Recommendations by Use Case

| Use Case | AWS Recommended | GCP Recommended |
|----------|---------|---------|
| **Simple Web App** | App Runner | Cloud Run |
| **Microservices** | ECS Fargate + Service Connect | Cloud Run |
| **K8s (Simplified)** | EKS Auto Mode | GKE Autopilot |
| **K8s (Full Control)** | EKS Standard | GKE Standard |
| **ML/GPU Workloads** | EKS + GPU | GKE Autopilot + GPU |
| **Batch Jobs** | ECS Task | Cloud Run Jobs |
| **Event Processing** | Fargate + EventBridge | Cloud Run + Eventarc |
| **Cost-Sensitive Batch** | Fargate Spot | Autopilot Spot Pods |

---

## 8. Pricing Comparison

### 8.1 ECS/EKS vs GKE

**AWS ECS (Fargate):**
```
vCPU: $0.04048/hour (Seoul)
Memory: $0.004445/GB/hour (Seoul)

Example: 0.5 vCPU, 1GB, 24 hours
= (0.5 × $0.04048 × 24) + (1 × $0.004445 × 24)
= $0.49 + $0.11 = $0.60/day
```

**AWS EKS:**
```
Cluster: $0.10/hour ($72/month)
+ Node costs (EC2) or Fargate costs
```

**GCP GKE:**
```
Autopilot: vCPU $0.0445/hour, Memory $0.0049/GB/hour
Standard: Management fee $0.10/hour/cluster + node costs

Example: Autopilot 0.5 vCPU, 1GB, 24 hours
= (0.5 × $0.0445 × 24) + (1 × $0.0049 × 24)
= $0.53 + $0.12 = $0.65/day
```

### 8.2 Cloud Run Pricing

```
CPU: $0.00002400/vCPU-second (during request processing)
Memory: $0.00000250/GB-second
Requests: $0.40/million requests

Free Tier:
- 2 million requests/month
- 360,000 GB-seconds
- 180,000 vCPU-seconds
```

---

## 9. Hands-on: Deploy a Simple Web App

### 9.1 Prepare Dockerfile

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

### 9.2 Deploy to GCP Cloud Run

```bash
# Build and deploy (directly from source)
gcloud run deploy my-app \
    --source=. \
    --region=asia-northeast3 \
    --allow-unauthenticated \
    --set-env-vars=CLOUD_PROVIDER=GCP
```

### 9.3 Deploy to AWS App Runner

```bash
# 1. Push image to ECR (as described above)

# 2. Create App Runner service
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

## 10. Next Steps

- [07_Object_Storage.md](./07_Object_Storage.md) - Object Storage
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Networking

---

## Exercises

### Exercise 1: Container vs VM Trade-off

A team is migrating a monolithic web application to the cloud. They are debating between running it on an EC2 VM versus containerizing it with Docker and deploying to ECS Fargate.

Identify two advantages of the container approach and two scenarios where staying with a VM would be the better choice.

<details>
<summary>Show Answer</summary>

**Advantages of containers (Fargate)**:
1. **Portability and consistency** — A Docker image encapsulates the application and all its dependencies. The same image runs identically in development, CI/CD, and production, eliminating "it works on my machine" problems.
2. **Density and cost efficiency** — Multiple containers share the host OS kernel, using far less memory than VMs where each has its own OS. Fargate charges only for the CPU/memory reserved by your task, with no idle EC2 overhead.

**Scenarios where VM (EC2) is better**:
1. **Applications requiring full OS access or kernel-level customization** — Some workloads (e.g., custom kernel modules, specialized hardware drivers, or applications that modify `/proc` and `/sys`) require root-level OS access that containers cannot provide.
2. **Legacy applications that cannot be containerized** — If the application has hard dependencies on specific OS versions, registry entries (Windows), or com objects, containerizing may require significant refactoring. Running directly on an EC2 VM may be the pragmatic migration path.

</details>

### Exercise 2: Service Selection

For each scenario, choose the most appropriate AWS container service (ECS on EC2, ECS Fargate, EKS, App Runner) and explain why:

1. A small startup wants to deploy a simple containerized REST API with zero infrastructure management and automatic HTTPS.
2. A platform team manages 50+ microservices and needs advanced networking, custom admission controllers, and multi-cloud portability.
3. A machine learning team runs GPU-accelerated training jobs as containers that process datasets and exit.
4. A company wants to run containers but requires custom EC2 instance types (e.g., storage-optimized i3 instances) for their workload.

<details>
<summary>Show Answer</summary>

1. **AWS App Runner** — Zero infrastructure management: no clusters, no task definitions, no load balancer configuration. App Runner handles HTTPS, scaling, load balancing, and deployments automatically. Ideal for simple stateless services.

2. **Amazon EKS** — Kubernetes is the industry standard for complex microservice platforms. Advanced features like custom resource definitions (CRDs), admission webhooks, pod security policies, and service mesh integration (Istio) are Kubernetes-native. Multi-cloud portability is a Kubernetes strength (same manifests can run on GKE, AKS).

3. **ECS Fargate with GPU** or **EKS with GPU node groups** — Batch-style jobs that start, process, and exit are well-suited for ECS tasks or Kubernetes Jobs. For GPU workloads, both platforms support GPU-enabled instances. ECS Tasks are simpler for batch; EKS Jobs offer more scheduling control.

4. **ECS on EC2** — When you need specific EC2 instance types (i3, c5n, etc.) that Fargate doesn't support, or when you need to attach physical hardware (GPUs with specific drivers), you must manage the underlying EC2 nodes. ECS on EC2 gives you full control over the instance type while still providing container orchestration.

</details>

### Exercise 3: Container Registry Workflow

Describe the complete sequence of CLI commands to:
1. Authenticate Docker to AWS ECR in the `ap-northeast-2` region for account `123456789012`.
2. Build a Docker image from a local `Dockerfile` and tag it for ECR with the repository name `my-app` and tag `v1.0`.
3. Push the image to ECR.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Authenticate Docker to ECR
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com

# Step 2: Build and tag the image
docker build -t my-app .
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:v1.0

# (Optional) Also tag as latest
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# Step 3: Push to ECR
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:v1.0
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest
```

**Note**: You must create the ECR repository before pushing. If it doesn't exist yet:
```bash
aws ecr create-repository --repository-name my-app --region ap-northeast-2
```

</details>

### Exercise 4: Cloud Run Traffic Splitting

You have deployed a new version of a Cloud Run service named `payment-service` in region `asia-northeast3`. The new revision is `payment-service-00003-xyz` and the previous stable version is `payment-service-00002-abc`.

Write the `gcloud` command to route 10% of traffic to the new version and 90% to the old version (canary deployment), and explain why this pattern is useful.

<details>
<summary>Show Answer</summary>

```bash
gcloud run services update-traffic payment-service \
    --region=asia-northeast3 \
    --to-revisions=payment-service-00003-xyz=10,payment-service-00002-abc=90
```

**Why this pattern (canary deployment) is useful**:
- **Risk reduction**: Only 10% of real users hit the new version. If it has a critical bug (crashes, data corruption), only 10% of users are affected instead of 100%.
- **Real traffic testing**: Synthetic tests and staging environments may not reproduce all production conditions. Canary deployments expose the new version to actual production load patterns.
- **Gradual rollout**: If the 10% canary shows good metrics (error rate, latency), you can increase it to 50%, then 100%. If problems arise, you roll back instantly by routing 100% back to the stable version.

To promote the new version to 100%:
```bash
gcloud run services update-traffic payment-service \
    --region=asia-northeast3 \
    --to-latest
```

</details>

### Exercise 5: Fargate vs Cloud Run Comparison

Analyze the architectural differences between AWS Fargate and GCP Cloud Run. For a stateless HTTP microservice that handles ~100 requests/second with occasional 10x spikes, which would you choose, and what are the key operational differences?

<details>
<summary>Show Answer</summary>

**Key architectural differences**:

| Aspect | AWS Fargate | GCP Cloud Run |
|--------|-------------|---------------|
| **Scaling model** | Pre-configured min/max task count; scales to minimum (not zero by default) | Scales to zero; request-based autoscaling |
| **Startup time** | Container cold start on new task; typically 30–60 seconds | Fast cold starts; typically a few seconds |
| **Pricing basis** | Per vCPU-hour and GB-hour while task is running (even idle) | Per vCPU-second and GB-second only during active request handling |
| **Cluster requirement** | Requires ECS cluster (though serverless, still needs cluster config) | Fully managed, no cluster concept |
| **Traffic splitting** | Requires ALB weighted target groups | Built-in revision-based traffic splitting |

**Recommendation for this scenario**: **GCP Cloud Run** is a better fit for this workload because:
1. **Cost**: 100 requests/second means the service has some baseline load, but Cloud Run's per-request billing is cheaper for bursty workloads. Fargate charges for idle time between spikes.
2. **Scaling**: Cloud Run's fast scale-out handles 10x spikes well. Built-in traffic splitting enables safe deployments.
3. **Simplicity**: No cluster management, no task definition versioning, no ALB configuration needed.

For AWS, **App Runner** would be the closer equivalent to Cloud Run for simplicity. Fargate is better when you need fine-grained networking (VPC, security groups, private subnets), persistent connections, or integration with other ECS-based services.

</details>

---

## References

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [GCP GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker/](../Docker/) - Docker Basics
