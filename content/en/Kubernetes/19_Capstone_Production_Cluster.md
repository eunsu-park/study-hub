# 19. Capstone: Production Cluster

**Previous**: [18. Kubernetes for ML](./18_Kubernetes_for_ML.md) | **Next**: [00. Overview](./00_Overview.md)

## Learning Objectives

- Gather requirements and translate them into Kubernetes architecture decisions
- Design a highly available production cluster with proper node pools, networking, and storage
- Implement security hardening, observability, and CI/CD integration
- Set up disaster recovery with automated backups and tested restore procedures
- Optimize costs while maintaining reliability and performance SLOs

---

This capstone project brings together everything from the previous 18 lessons into a single, cohesive exercise: designing, deploying, and validating a production-grade Kubernetes cluster. You will work through requirements gathering, architecture design, security hardening, observability, CI/CD integration, disaster recovery, and cost optimization. Each section builds on the last, culminating in a complete production platform.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Requirements Gathering](#1-requirements-gathering)
- [2. Architecture Design](#2-architecture-design)
- [3. HA Control Plane](#3-ha-control-plane)
- [4. Node Pool Design](#4-node-pool-design)
- [5. Networking Architecture](#5-networking-architecture)
- [6. Security Hardening](#6-security-hardening)
- [7. Observability Stack](#7-observability-stack)
- [8. CI/CD Pipeline Integration](#8-cicd-pipeline-integration)
- [9. Disaster Recovery Setup](#9-disaster-recovery-setup)
- [10. Cost Optimization](#10-cost-optimization)
- [Exercises](#exercises)

---

## 1. Requirements Gathering

### Theory: Reciprocal Compromise: Every Choice Has a Hidden Cost

Each technical decision creates a reciprocal demand somewhere else:

- **Choose Istio service mesh** → gain mTLS, traffic shifting, observability → pay with extra latency (~1ms per hop), sidecar memory (~100MB per pod), and a new control plane to operate.
- **Choose KubeVirt for VM workloads** → gain unified compute platform → pay with non-trivial KubeVirt operator complexity and a different runtime model than vanilla Kubernetes.
- **Choose hub-spoke multi-cluster** (lesson 15) → gain centralized platform services → pay with a critical hub cluster that needs its own HA story.
- **Choose strict Pod Security restricted profile** → gain compliance + reduced attack surface → pay with workloads that need exceptions and the operational overhead of approving them.
- **Choose immutable infrastructure (rebuild nodes for upgrades)** → gain clean upgrades and reproducibility → pay with longer upgrade windows and more cloud spend during transitions.

The "no free lunch" property means every design review should ask: *what does this choice cost us elsewhere?* If the answer is "nothing," you haven't looked hard enough. Production-quality designs are honest about the costs and explicit about why the chosen trade-off is correct *for this organization*.

### Theory: The Diagnostic Lens: Stress-Test Before Committing

A good design survives questioning. Before signing off on an architecture, walk through these scenarios:

- **What if etcd is corrupted?** Do you have a tested restore? RPO/RTO? (Lesson 17 §B.)
- **What if a region goes offline?** Is traffic routed away automatically? Do RPO/RTO meet business needs?
- **What if a security policy needs to change cluster-wide?** Is there a single applied source (GitOps, lesson 15) or do you log into 50 clusters?
- **What if a developer accidentally deploys a privileged pod?** Is admission (lesson 12) enforcing that, or is it only audited?
- **What if your traffic 10×?** Does HPA scale workloads? Does Cluster Autoscaler scale nodes? Does the metrics pipeline (lesson 13 §B) keep up?
- **What if a key engineer is on vacation?** Are runbooks documented? Can on-call execute without that person?
- **What if AWS deprecates the instance type?** Are nodes immutable enough to swap? Are there hard couplings to specific hardware?
- **What if a pod gets popped (RCE)?** Is the blast radius bounded by NetworkPolicy, ServiceAccount permissions, and Pod Security? (Lessons 06, 08.)

Each "what if" stress-tests one assumption. A design that has answers — even imperfect ones — to all of them is production-grade. A design that says "we'll figure that out later" has hidden risk that will surface at the worst possible moment.

The lens for the capstone exercise: every section of the design (HA control plane, node pools, networking, security, observability, CI/CD, DR, cost) should be defensible against this kind of questioning.

### 1.1 Stakeholder Questions

Before designing a cluster, you must understand the workloads, constraints, and expectations:

```
Requirements Matrix:
┌───────────────────────────────────────────────────────────┐
│  Category            │ Questions to Answer                │
├──────────────────────┼────────────────────────────────────┤
│  Workloads           │ How many applications?             │
│                      │ Stateless vs stateful ratio?       │
│                      │ CPU-bound vs memory-bound vs GPU?  │
│                      │ Expected pod count?                │
│                      │ Batch jobs vs long-running?        │
├──────────────────────┼────────────────────────────────────┤
│  Scale               │ Expected request rate (RPS)?       │
│                      │ Peak vs average traffic ratio?     │
│                      │ Growth rate (6/12/24 months)?      │
│                      │ Number of tenant teams?            │
├──────────────────────┼────────────────────────────────────┤
│  Availability        │ Target uptime SLO?                 │
│                      │ Acceptable RTO and RPO?            │
│                      │ Multi-region requirement?          │
│                      │ Maintenance window policy?         │
├──────────────────────┼────────────────────────────────────┤
│  Compliance          │ Data residency requirements?       │
│                      │ Encryption at rest and in transit? │
│                      │ Audit logging requirements?        │
│                      │ Network segmentation needs?        │
├──────────────────────┼────────────────────────────────────┤
│  Budget              │ Monthly infrastructure budget?     │
│                      │ Cost per environment limit?        │
│                      │ Spot/preemptible tolerance?        │
│                      │ Reserved instance commitment?      │
└──────────────────────┴────────────────────────────────────┘
```

### 1.2 Reference Scenario

For this capstone, we use the following scenario:

```yaml
# capstone-requirements.yaml
company: "TechCorp"
platform: "e-commerce + ML recommendation engine"

workloads:
  applications: 25
  microservices: 18
  stateful_services: 4          # PostgreSQL, Redis, Elasticsearch, Kafka
  ml_workloads: 3               # training, serving, pipelines
  total_pods_peak: 500
  gpu_requirement: true         # 8x A100 for ML

scale:
  peak_rps: 10000
  average_rps: 3000
  peak_to_average_ratio: 3.3
  growth_rate_annual: 40%
  teams: 5                      # platform, backend, frontend, data, ML

availability:
  slo: "99.95%"
  rto: "30 minutes"
  rpo: "1 hour"
  multi_region: false           # Single region, multi-AZ
  maintenance_window: "Sunday 02:00-06:00 UTC"

compliance:
  data_residency: "US"
  encryption_at_rest: true
  encryption_in_transit: true
  audit_logging: true
  network_segmentation: true    # Per-team namespace isolation

budget:
  monthly_limit: "$25,000"
  spot_tolerance: "training jobs only"
  reserved_instances: "control plane + system nodes"
```

---

## 2. Architecture Design

### Theory: The Layered Design Model

Production clusters decompose into four layers, each built on the one below:

**Layer 1: Foundations.** The cluster itself — control plane HA (lesson 17), node pools, networking model (lesson 08), storage classes (lesson 04), DNS, ingress controllers (lesson 07). This is your "OS" — you should rarely change it after rollout, and changes here have the broadest blast radius.

**Layer 2: Platform services.** What runs *for* every workload — observability (lesson 14), GitOps controller (lesson 15), secret management (lesson 05), policy enforcement (lesson 12), backup/restore tooling (lesson 17). These are the inner-platform that workload teams consume but don't operate.

**Layer 3: Workloads.** Application Deployments, StatefulSets, Jobs (lesson 02), exposed via Services and Ingresses (lessons 03, 07). The user-visible value of the cluster lives here.

**Layer 4: Day-2 operations.** SLO definitions, runbooks, on-call rotations, capacity planning, change management (lesson 17). Not in YAML but every bit as critical to whether the cluster is "production."

The reason this layering matters: **changes at lower layers have larger blast radius**, and you should design accordingly. A workload deployment going wrong takes down one app; a CNI upgrade going wrong takes down everything. Plan layer-1 changes with multi-cluster strategies (lesson 15), test in lower environments first, accept slower iteration in exchange for safety.

### Theory: The Trade-Off Triangle: Cost, Availability, Change Velocity

Every design decision sits in a three-way trade-off:

- **Cost.** Cluster spend per month — node hours, storage, observability ingest, managed-service fees.
- **Availability.** Effective uptime — multi-AZ, multi-region, redundancy at every layer.
- **Change velocity.** Rate at which you can ship safely — CI/CD throughput, time-from-commit-to-prod, test coverage.

You can optimize for any two, but not all three:

- **Cost + Availability** without velocity: a small, hyper-stable platform team that approves every change manually. Banks. Slow but cheap and reliable.
- **Cost + Velocity** without availability: minimal redundancy, ship fast, accept incidents. Early-stage startups.
- **Availability + Velocity** without low cost: full multi-region active-active, automated canary everything, robust observability. Modern SaaS.

Recognizing this triangle prevents arguments. "Why don't we just deploy multi-region?" → "Because that doubles cost and we've prioritized velocity." "Why is the deploy pipeline so slow?" → "Because we prioritized availability and added approvals." Make the trade-off explicit; let leadership pick the corner.

The triangle plays out concretely:

| Decision | Cost ↑ | Availability ↑ | Velocity ↑ |
|----------|--------|----------------|------------|
| Multi-region cluster | + + + | + + + | – |
| Spot-only nodes | – – | – | 0 |
| Service mesh | + | + + | – |
| Strict admission policies | 0 | + | – – |
| Autoscaling everywhere | – | + + | + |

There is no universally right cell; there is the cell that fits *your* constraints.

### 2.1 High-Level Architecture

```
Production Cluster Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                         VPC (10.0.0.0/16)                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Public Subnets (3 AZs)                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │    │
│  │  │ NAT GW   │  │ NAT GW   │  │ NAT GW   │               │    │
│  │  │ AZ-a     │  │ AZ-b     │  │ AZ-c     │               │    │
│  │  └──────────┘  └──────────┘  └──────────┘               │    │
│  │  ┌──────────────────────────────────────┐                │    │
│  │  │        NLB (API Server endpoint)     │                │    │
│  │  └──────────────────────────────────────┘                │    │
│  │  ┌──────────────────────────────────────┐                │    │
│  │  │        ALB (Ingress traffic)         │                │    │
│  │  └──────────────────────────────────────┘                │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Private Subnets (3 AZs)                      │    │
│  │                                                           │    │
│  │  Control Plane (HA):                                      │    │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                      │    │
│  │  │ CP-1   │  │ CP-2   │  │ CP-3   │                      │    │
│  │  │ AZ-a   │  │ AZ-b   │  │ AZ-c   │                      │    │
│  │  │ etcd   │  │ etcd   │  │ etcd   │                      │    │
│  │  └────────┘  └────────┘  └────────┘                      │    │
│  │                                                           │    │
│  │  Worker Nodes:                                            │    │
│  │  ┌──────────────────────────────────────────────────┐     │    │
│  │  │ System Pool (3x m5.xlarge, reserved)             │     │    │
│  │  │ General Pool (5-20x m5.2xlarge, on-demand)       │     │    │
│  │  │ Stateful Pool (3x r5.2xlarge, on-demand)         │     │    │
│  │  │ GPU Pool (2-8x p4d.24xlarge, spot + on-demand)   │     │    │
│  │  └──────────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Supporting Services                                      │    │
│  │  ├── ECR (container registry)                             │    │
│  │  ├── S3 (backups, ML models, logs)                        │    │
│  │  ├── Route 53 (DNS)                                       │    │
│  │  ├── ACM (TLS certificates)                               │    │
│  │  └── CloudWatch (audit logs)                              │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Infrastructure as Code

```bash
# Terraform project structure
terraform/
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── eks/
│   │   ├── main.tf
│   │   ├── node_groups.tf
│   │   ├── addons.tf
│   │   └── variables.tf
│   ├── observability/
│   │   ├── prometheus.tf
│   │   ├── grafana.tf
│   │   └── loki.tf
│   └── security/
│       ├── iam.tf
│       ├── kms.tf
│       └── security_groups.tf
├── environments/
│   ├── production/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   └── staging/
│       ├── main.tf
│       ├── terraform.tfvars
│       └── backend.tf
└── README.md
```

```bash
# Core Terraform configuration
# terraform/environments/production/main.tf

module "vpc" {
  source = "../../modules/vpc"

  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets    = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false    # One per AZ for HA
}

module "eks" {
  source = "../../modules/eks"

  cluster_name       = "prod-cluster"
  cluster_version    = "1.29"
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids

  # Control plane logging
  enabled_cluster_log_types = [
    "api", "audit", "authenticator",
    "controllerManager", "scheduler"
  ]

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = module.security.kms_key_arn
    resources        = ["secrets"]
  }
}
```

---

## 3. HA Control Plane

### 3.1 Control Plane Configuration

```yaml
# kubeadm HA control plane configuration
# (For self-managed clusters; EKS/GKE handle this automatically)
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
kubernetesVersion: v1.29.2
controlPlaneEndpoint: "api.prod.example.com:6443"    # Load balancer
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
  dnsDomain: "cluster.local"
apiServer:
  extraArgs:
    audit-policy-file: /etc/kubernetes/audit-policy.yaml
    audit-log-path: /var/log/kubernetes/audit.log
    audit-log-maxage: "30"
    audit-log-maxbackup: "10"
    encryption-provider-config: /etc/kubernetes/encryption-config.yaml
    enable-admission-plugins: >-
      NodeRestriction,
      PodSecurity,
      ResourceQuota,
      LimitRanger,
      ServiceAccount
    max-requests-inflight: "800"
    max-mutating-requests-inflight: "400"
  extraVolumes:
    - name: audit-config
      hostPath: /etc/kubernetes/audit-policy.yaml
      mountPath: /etc/kubernetes/audit-policy.yaml
      readOnly: true
    - name: audit-log
      hostPath: /var/log/kubernetes
      mountPath: /var/log/kubernetes
    - name: encryption-config
      hostPath: /etc/kubernetes/encryption-config.yaml
      mountPath: /etc/kubernetes/encryption-config.yaml
      readOnly: true
etcd:
  local:
    extraArgs:
      quota-backend-bytes: "8589934592"       # 8GB
      auto-compaction-mode: periodic
      auto-compaction-retention: "8h"
      snapshot-count: "10000"
    serverCertSANs:
      - "etcd-0.etcd.kube-system.svc.cluster.local"
      - "etcd-1.etcd.kube-system.svc.cluster.local"
      - "etcd-2.etcd.kube-system.svc.cluster.local"
```

### 3.2 etcd Encryption at Rest

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
      - configmaps
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}    # Fallback for reading unencrypted data
```

### 3.3 API Server Audit Policy

```yaml
# /etc/kubernetes/audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  # Do not log requests to certain non-sensitive endpoints
  - level: None
    nonResourceURLs:
      - "/healthz*"
      - "/readyz*"
      - "/livez*"
      - "/metrics"

  # Do not log watch requests (too verbose)
  - level: None
    verbs: ["watch"]

  # Log secret access at Metadata level (do not log secret contents)
  - level: Metadata
    resources:
      - group: ""
        resources: ["secrets"]
    omitStages:
      - RequestReceived

  # Log all other requests at RequestResponse level
  - level: RequestResponse
    resources:
      - group: ""
        resources: ["pods", "services", "configmaps"]
      - group: "apps"
        resources: ["deployments", "statefulsets"]
      - group: "rbac.authorization.k8s.io"
        resources: ["roles", "rolebindings", "clusterroles", "clusterrolebindings"]
    omitStages:
      - RequestReceived

  # Catch-all at Metadata level
  - level: Metadata
    omitStages:
      - RequestReceived
```

---

## 4. Node Pool Design

### 4.1 Node Pool Specifications

```yaml
# EKS Managed Node Groups
node_pools:
  # System components (monitoring, ingress, DNS)
  system:
    instance_types: ["m5.xlarge"]          # 4 vCPU, 16 GB
    capacity_type: ON_DEMAND
    desired: 3
    min: 3
    max: 3
    labels:
      node-role: system
    taints:
      - key: node-role
        value: system
        effect: NoSchedule
    ami_type: AL2_x86_64
    disk_size: 100

  # General purpose workloads
  general:
    instance_types: ["m5.2xlarge"]         # 8 vCPU, 32 GB
    capacity_type: ON_DEMAND
    desired: 5
    min: 3
    max: 20
    labels:
      node-role: general
    taints: []
    ami_type: AL2_x86_64
    disk_size: 200

  # Stateful workloads (databases, caches)
  stateful:
    instance_types: ["r5.2xlarge"]         # 8 vCPU, 64 GB
    capacity_type: ON_DEMAND
    desired: 3
    min: 3
    max: 6
    labels:
      node-role: stateful
    taints:
      - key: workload-type
        value: stateful
        effect: NoSchedule
    ami_type: AL2_x86_64
    disk_size: 500
    volume_type: io2
    iops: 10000

  # GPU nodes for ML
  gpu:
    instance_types: ["p4d.24xlarge"]       # 8x A100, 96 vCPU, 1.1 TB
    capacity_type: ON_DEMAND
    desired: 1
    min: 0
    max: 4
    labels:
      node-role: gpu
      nvidia.com/gpu.present: "true"
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    ami_type: AL2_x86_64_GPU
    disk_size: 500

  # GPU spot nodes for training
  gpu_spot:
    instance_types: ["p3.8xlarge", "p3.16xlarge"]
    capacity_type: SPOT
    desired: 0
    min: 0
    max: 8
    labels:
      node-role: gpu-spot
      instance-lifecycle: spot
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
      - key: instance-lifecycle
        value: spot
        effect: NoSchedule
    ami_type: AL2_x86_64_GPU
    disk_size: 300
```

### 4.2 Cluster Autoscaler Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      nodeSelector:
        node-role: system
      tolerations:
        - key: node-role
          value: system
          effect: NoSchedule
      containers:
        - name: cluster-autoscaler
          image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.29.0
          command:
            - ./cluster-autoscaler
            - --v=4
            - --cloud-provider=aws
            - --skip-nodes-with-local-storage=false
            - --expander=priority          # Use priority-based expander
            - --scale-down-delay-after-add=10m
            - --scale-down-unneeded-time=10m
            - --scale-down-utilization-threshold=0.5
            - --max-graceful-termination-sec=600
            - --balance-similar-node-groups=true
            - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/prod-cluster
          resources:
            requests:
              cpu: 100m
              memory: 300Mi
            limits:
              cpu: 500m
              memory: 600Mi
```

---

## 5. Networking Architecture

### 5.1 CNI and Network Policy

```yaml
# Cilium CNI configuration (Helm values)
cilium:
  ipam:
    mode: eni                    # AWS ENI mode
  eni:
    enabled: true
    awsEnablePrefixDelegation: true   # More IPs per ENI
  tunnel: disabled               # Native routing (no overlay)
  enableIPv4Masquerade: true

  # Network policy engine
  policyEnforcementMode: default
  hostFirewall:
    enabled: true

  # Hubble (observability)
  hubble:
    enabled: true
    relay:
      enabled: true
    ui:
      enabled: true

  # Bandwidth manager
  bandwidthManager:
    enabled: true

  # Encryption in transit
  encryption:
    enabled: true
    type: wireguard
```

### 5.2 Ingress Architecture

```yaml
# AWS Load Balancer Controller + Ingress
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  name: alb
  annotations:
    ingressclass.kubernetes.io/is-default-class: "true"
spec:
  controller: ingress.k8s.io/alb
---
# Production ingress with WAF, SSL, and rate limiting
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: production-ingress
  namespace: production
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456:certificate/abc-123
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS13-1-2-2021-06
    alb.ingress.kubernetes.io/wafv2-acl-arn: arn:aws:wafv2:us-east-1:123456:regional/webacl/prod-waf
    alb.ingress.kubernetes.io/shield-advanced-protection: "true"
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "15"
spec:
  ingressClassName: alb
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-gateway
                port:
                  number: 80
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-frontend
                port:
                  number: 80
```

### 5.3 Network Policies for Namespace Isolation

```yaml
# Default deny all ingress in production namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
---
# Allow ingress only from ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      exposure: external
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              app.kubernetes.io/name: aws-load-balancer-controller
---
# Allow inter-service communication within namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector: {}
---
# Allow monitoring namespace to scrape metrics
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-monitoring-scrape
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - port: metrics
          protocol: TCP
```

---

## 6. Security Hardening

### 6.1 Security Checklist

```
Production Security Hardening Checklist:
┌──────────────────────────────────────────────────────────────┐
│  ✓  Control Plane                                            │
│  ├── [ ] API server accessible only via private endpoint     │
│  ├── [ ] Audit logging enabled and shipped to SIEM           │
│  ├── [ ] Encryption at rest for etcd (secrets, configmaps)   │
│  ├── [ ] RBAC with least-privilege ServiceAccounts           │
│  ├── [ ] Admission controllers: PodSecurity, OPA/Kyverno     │
│  └── [ ] Certificate auto-rotation enabled                   │
│                                                              │
│  ✓  Workloads                                                │
│  ├── [ ] Pod Security Standards enforced (restricted)        │
│  ├── [ ] No privileged containers                            │
│  ├── [ ] Read-only root filesystem                           │
│  ├── [ ] Non-root user in all containers                     │
│  ├── [ ] Resource limits on all pods                         │
│  ├── [ ] Image pull from private registry only               │
│  └── [ ] Image vulnerability scanning in CI/CD               │
│                                                              │
│  ✓  Network                                                  │
│  ├── [ ] Default deny NetworkPolicies per namespace          │
│  ├── [ ] Encryption in transit (mTLS via service mesh)       │
│  ├── [ ] Ingress with WAF and DDoS protection                │
│  ├── [ ] Egress controls for sensitive namespaces            │
│  └── [ ] DNS policies (no external resolution for DB pods)   │
│                                                              │
│  ✓  Supply Chain                                             │
│  ├── [ ] Signed container images (Sigstore/cosign)           │
│  ├── [ ] SBOM generation for all images                      │
│  ├── [ ] Admission webhook to verify image signatures        │
│  └── [ ] Base image pinned to digest, not tag                │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Pod Security Standards

```yaml
# Enforce restricted Pod Security Standard on production namespace
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.29
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

A compliant pod under the `restricted` profile:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: production
spec:
  securityContext:
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: registry.example.com/app:v1.0@sha256:abc123...
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        runAsUser: 1000
        runAsGroup: 1000
        capabilities:
          drop:
            - ALL
      resources:
        requests:
          cpu: 100m
          memory: 128Mi
        limits:
          cpu: 500m
          memory: 512Mi
      volumeMounts:
        - name: tmp
          mountPath: /tmp
  volumes:
    - name: tmp
      emptyDir:
        sizeLimit: 100Mi
```

### 6.3 RBAC Configuration

```yaml
# Team-level RBAC: backend team can manage their namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: backend-developer
  namespace: backend
rules:
  - apiGroups: ["", "apps", "batch"]
    resources: ["pods", "deployments", "services", "configmaps", "jobs", "cronjobs"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods/log", "pods/exec"]
    verbs: ["get", "create"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]         # Can read but not create/modify secrets
  - apiGroups: ["networking.k8s.io"]
    resources: ["ingresses"]
    verbs: ["get", "list", "watch", "create", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backend-developer-binding
  namespace: backend
subjects:
  - kind: Group
    name: "backend-developers"
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: backend-developer
  apiGroup: rbac.authorization.k8s.io
---
# Read-only cluster role for on-call engineers
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: oncall-readonly
rules:
  - apiGroups: ["", "apps", "batch", "networking.k8s.io"]
    resources: ["*"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
```

### 6.4 Image Security with Kyverno

```yaml
# Kyverno policy: require images from private registry
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-private-registry
spec:
  validationFailureAction: Enforce
  background: true
  rules:
    - name: validate-image-registry
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
                - staging
      validate:
        message: "Images must come from the private registry"
        pattern:
          spec:
            containers:
              - image: "registry.example.com/*"
            initContainers:
              - image: "registry.example.com/*"
---
# Require image digest (not just tag)
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-digest
spec:
  validationFailureAction: Enforce
  rules:
    - name: check-digest
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
      validate:
        message: "Production images must reference a digest (@sha256:...)"
        pattern:
          spec:
            containers:
              - image: "*@sha256:*"
```

---

## 7. Observability Stack

### 7.1 Monitoring Architecture

```
Observability Stack:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Metrics:                                                   │
│  ┌────────────┐   ┌────────────────┐   ┌───────────────┐   │
│  │ Prometheus │──►│ Thanos/Cortex  │──►│ Grafana       │   │
│  │ (per-      │   │ (long-term     │   │ (dashboards)  │   │
│  │  cluster)  │   │  storage, S3)  │   │               │   │
│  └────────────┘   └────────────────┘   └───────────────┘   │
│                                                             │
│  Logging:                                                   │
│  ┌────────────┐   ┌────────────────┐   ┌───────────────┐   │
│  │ Fluent Bit │──►│ Loki           │──►│ Grafana       │   │
│  │ (DaemonSet)│   │ (log store)    │   │ (log search)  │   │
│  └────────────┘   └────────────────┘   └───────────────┘   │
│                                                             │
│  Tracing:                                                   │
│  ┌────────────┐   ┌────────────────┐   ┌───────────────┐   │
│  │ OTel       │──►│ Tempo          │──►│ Grafana       │   │
│  │ Collector  │   │ (trace store)  │   │ (trace view)  │   │
│  └────────────┘   └────────────────┘   └───────────────┘   │
│                                                             │
│  Alerting:                                                  │
│  ┌────────────┐   ┌────────────────┐                        │
│  │ Alertmanag.│──►│ PagerDuty /    │                        │
│  │            │   │ Slack / Email  │                        │
│  └────────────┘   └────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Prometheus Stack Deployment

```yaml
# kube-prometheus-stack Helm values
prometheus:
  prometheusSpec:
    retention: 15d
    retentionSize: 50GB
    resources:
      requests:
        cpu: "1"
        memory: 4Gi
      limits:
        cpu: "2"
        memory: 8Gi
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          resources:
            requests:
              storage: 100Gi
    nodeSelector:
      node-role: system
    tolerations:
      - key: node-role
        value: system
        effect: NoSchedule
    # Thanos sidecar for long-term storage
    thanos:
      objectStorageConfig:
        existingSecret:
          name: thanos-s3-config
          key: objstore.yml

alertmanager:
  alertmanagerSpec:
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
  config:
    route:
      receiver: "null"
      group_by: ["alertname", "namespace"]
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
        - receiver: pagerduty-critical
          match:
            severity: critical
          continue: true
        - receiver: slack-warnings
          match:
            severity: warning
    receivers:
      - name: "null"
      - name: pagerduty-critical
        pagerduty_configs:
          - service_key_file: /etc/alertmanager/secrets/pagerduty-key
      - name: slack-warnings
        slack_configs:
          - api_url_file: /etc/alertmanager/secrets/slack-webhook
            channel: "#k8s-alerts"
            title: "{{ .GroupLabels.alertname }}"
            text: "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}"

grafana:
  adminPassword: <from-secret>
  persistence:
    enabled: true
    size: 10Gi
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: default
          folder: Kubernetes
          type: file
          options:
            path: /var/lib/grafana/dashboards
```

### 7.3 Logging with Fluent Bit and Loki

```yaml
# Fluent Bit DaemonSet configuration
fluent-bit:
  config:
    inputs: |
      [INPUT]
          Name              tail
          Tag               kube.*
          Path              /var/log/containers/*.log
          Parser            cri
          DB                /var/log/flb_kube.db
          Mem_Buf_Limit     50MB
          Skip_Long_Lines   On
          Refresh_Interval  10

    filters: |
      [FILTER]
          Name                kubernetes
          Match               kube.*
          Merge_Log           On
          Keep_Log            Off
          K8S-Logging.Parser  On
          K8S-Logging.Exclude On

    outputs: |
      [OUTPUT]
          Name          loki
          Match         kube.*
          Host          loki-gateway.monitoring.svc
          Port          80
          Labels        job=fluent-bit
          Auto_Kubernetes_Labels On

  tolerations:
    - operator: Exists    # Run on all nodes including GPU
```

---

## 8. CI/CD Pipeline Integration

### 8.1 GitOps with ArgoCD

```yaml
# ArgoCD Application for production deployment
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: production-apps
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: production
  source:
    repoURL: https://github.com/techcorp/k8s-manifests.git
    targetRevision: main
    path: production/
    directory:
      recurse: true
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
---
# ArgoCD Project with RBAC
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: production
  namespace: argocd
spec:
  description: "Production applications"
  sourceRepos:
    - "https://github.com/techcorp/k8s-manifests.git"
  destinations:
    - namespace: production
      server: https://kubernetes.default.svc
    - namespace: production-jobs
      server: https://kubernetes.default.svc
  clusterResourceWhitelist:
    - group: ""
      kind: Namespace
  namespaceResourceWhitelist:
    - group: "*"
      kind: "*"
  roles:
    - name: deployer
      policies:
        - p, proj:production:deployer, applications, sync, production/*, allow
        - p, proj:production:deployer, applications, get, production/*, allow
      groups:
        - platform-team
```

### 8.2 CI Pipeline with Image Building

```yaml
# GitHub Actions CI pipeline
# .github/workflows/ci.yaml
name: CI Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write          # For OIDC with AWS
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456:role/github-actions
          aws-region: us-east-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and scan image
        run: |
          IMAGE_TAG="${{ github.sha }}"
          docker build -t $ECR_REGISTRY/app:$IMAGE_TAG .

          # Vulnerability scan with Trivy
          trivy image --exit-code 1 --severity HIGH,CRITICAL \
            $ECR_REGISTRY/app:$IMAGE_TAG

      - name: Push image
        run: |
          IMAGE_TAG="${{ github.sha }}"
          docker push $ECR_REGISTRY/app:$IMAGE_TAG

          # Sign the image with cosign
          cosign sign --yes $ECR_REGISTRY/app:$IMAGE_TAG

      - name: Update Kubernetes manifest
        run: |
          IMAGE_TAG="${{ github.sha }}"
          cd k8s-manifests
          kustomize edit set image app=$ECR_REGISTRY/app:$IMAGE_TAG
          git add .
          git commit -m "Deploy app:$IMAGE_TAG"
          git push
```

### 8.3 Progressive Delivery with Argo Rollouts

```yaml
# Canary deployment with Argo Rollouts
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: api-server
  namespace: production
spec:
  replicas: 10
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
        - name: api
          image: registry.example.com/api:v2.0@sha256:abc123
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
  strategy:
    canary:
      steps:
        - setWeight: 5
        - pause: {duration: 5m}
        - setWeight: 20
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
        - setWeight: 80
        - pause: {duration: 5m}
      analysis:
        templates:
          - templateName: success-rate
        startingStep: 1
        args:
          - name: service-name
            value: api-server
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: production
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 2m
      successCondition: result[0] >= 0.99
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{
              service="{{ args.service-name }}",
              status=~"2.."
            }[5m]))
            /
            sum(rate(http_requests_total{
              service="{{ args.service-name }}"
            }[5m]))
```

---

## 9. Disaster Recovery Setup

### 9.1 Automated Backup System

```yaml
# Velero backup schedule for production
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: production-backup
  namespace: velero
spec:
  schedule: "0 */4 * * *"       # Every 4 hours
  template:
    includedNamespaces:
      - production
      - backend
      - data-platform
    includedResources:
      - deployments
      - services
      - configmaps
      - secrets
      - persistentvolumeclaims
      - ingresses
      - networkpolicies
    snapshotVolumes: true
    ttl: 720h                    # Retain for 30 days
    storageLocation: aws-s3-primary
    volumeSnapshotLocations:
      - aws-ebs-snapshots
---
# etcd backup CronJob (in addition to Velero)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 */6 * * *"       # Every 6 hours
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          hostNetwork: true
          nodeSelector:
            node-role.kubernetes.io/control-plane: ""
          tolerations:
            - key: node-role.kubernetes.io/control-plane
              effect: NoSchedule
          containers:
            - name: etcd-backup
              image: bitnami/etcd:3.5
              command: ["/bin/sh", "-c"]
              args:
                - |
                  ETCDCTL_API=3 etcdctl \
                    --endpoints=https://127.0.0.1:2379 \
                    --cacert=/certs/ca.crt \
                    --cert=/certs/server.crt \
                    --key=/certs/server.key \
                    snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db
                  aws s3 sync /backup/ s3://prod-backups/etcd/ --exclude "*" --include "*.db"
                  find /backup -name "*.db" -mtime +3 -delete
              volumeMounts:
                - name: etcd-certs
                  mountPath: /certs
                  readOnly: true
                - name: backup
                  mountPath: /backup
          volumes:
            - name: etcd-certs
              hostPath:
                path: /etc/kubernetes/pki/etcd
            - name: backup
              hostPath:
                path: /var/backup/etcd
          restartPolicy: OnFailure
```

### 9.2 DR Validation Testing

```bash
#!/usr/bin/env bash
# dr-drill.sh - Quarterly DR validation drill

set -euo pipefail

DRILL_NS="dr-drill-$(date +%s)"
REPORT_FILE="/tmp/dr-drill-report-$(date +%Y%m%d).md"

echo "# DR Drill Report - $(date)" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "## 1. Backup Verification" >> "$REPORT_FILE"

# Verify Velero backup exists and is recent
LATEST_BACKUP=$(velero backup get --output json | \
  jq -r '.items | sort_by(.status.completionTimestamp) | last | .metadata.name')
BACKUP_AGE=$(velero backup get "$LATEST_BACKUP" -o json | \
  jq -r '.status.completionTimestamp')
echo "- Latest backup: $LATEST_BACKUP ($BACKUP_AGE)" >> "$REPORT_FILE"

# Verify etcd backup in S3
ETCD_BACKUP=$(aws s3 ls s3://prod-backups/etcd/ --recursive | tail -1)
echo "- Latest etcd backup: $ETCD_BACKUP" >> "$REPORT_FILE"

echo "## 2. Restore Test" >> "$REPORT_FILE"

# Create isolated namespace and restore into it
kubectl create namespace "$DRILL_NS"
velero restore create "dr-drill-$(date +%s)" \
  --from-backup "$LATEST_BACKUP" \
  --namespace-mappings "production:$DRILL_NS" \
  --wait

# Count restored resources
PODS=$(kubectl get pods -n "$DRILL_NS" --no-headers | wc -l)
SVCS=$(kubectl get services -n "$DRILL_NS" --no-headers | wc -l)
echo "- Restored pods: $PODS" >> "$REPORT_FILE"
echo "- Restored services: $SVCS" >> "$REPORT_FILE"

echo "## 3. Health Verification" >> "$REPORT_FILE"

# Wait and check pod health
sleep 60
HEALTHY=$(kubectl get pods -n "$DRILL_NS" --field-selector=status.phase=Running --no-headers | wc -l)
TOTAL=$(kubectl get pods -n "$DRILL_NS" --no-headers | wc -l)
echo "- Healthy pods: $HEALTHY / $TOTAL" >> "$REPORT_FILE"

echo "## 4. Cleanup" >> "$REPORT_FILE"
kubectl delete namespace "$DRILL_NS" --wait=false
echo "- Drill namespace $DRILL_NS scheduled for deletion" >> "$REPORT_FILE"

echo "DR drill complete. Report: $REPORT_FILE"
cat "$REPORT_FILE"
```

---

## 10. Cost Optimization

### 10.1 Cost Analysis Framework

```
Cost Breakdown (Monthly Estimate):
┌────────────────────────────────────────────────────────────┐
│  Component                    │ Monthly Cost │ % of Total  │
├───────────────────────────────┼──────────────┼─────────────┤
│  Control Plane (EKS)          │    $219      │    0.9%     │
│  System Nodes (3x m5.xl, RI)  │    $750      │    3.0%     │
│  General Nodes (avg 10x m5.2xl│  $5,550      │   22.2%     │
│  Stateful Nodes (3x r5.2xl)   │  $2,280      │    9.1%     │
│  GPU On-Demand (1x p4d.24xl)  │ $10,080      │   40.3%     │
│  GPU Spot (avg 2x p3.8xl)     │  $2,640      │   10.6%     │
│  Storage (EBS + S3)           │  $1,200      │    4.8%     │
│  Networking (NAT GW, ALB)     │  $1,500      │    6.0%     │
│  Monitoring (Prometheus data)  │    $300      │    1.2%     │
│  Backups (S3 + snapshots)     │    $200      │    0.8%     │
│  Misc (DNS, ECR, CloudWatch)  │    $280      │    1.1%     │
├───────────────────────────────┼──────────────┼─────────────┤
│  TOTAL                        │ $24,999      │  100.0%     │
└───────────────────────────────┴──────────────┴─────────────┘
```

### 10.2 Cost Optimization Strategies

```yaml
# Strategy 1: Right-size with VPA recommendations
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-server-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  updatePolicy:
    updateMode: "Off"    # Recommendations only
---
# Strategy 2: Scale to zero for dev/staging
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: staging-api
  namespace: staging
spec:
  scaleTargetRef:
    name: api-server
  minReplicaCount: 0             # Scale to zero when no traffic
  maxReplicaCount: 5
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: http_requests_per_second
        query: sum(rate(http_requests_total{namespace="staging"}[2m]))
        threshold: "1"
---
# Strategy 3: Spot instances for non-critical workloads
# (See node pool design - gpu_spot pool)

# Strategy 4: Storage lifecycle policies
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: retain-7-days
driver: ebs.csi.aws.com
deletionPolicy: Delete
parameters:
  tagSpecification_1: "RetentionDays=7"
```

### 10.3 Cost Monitoring with Kubecost

```bash
# Install Kubecost for cost visibility
helm install kubecost cost-analyzer \
  --repo https://kubecost.github.io/cost-analyzer/ \
  --namespace kubecost \
  --create-namespace \
  --set kubecostToken="<token>" \
  --set prometheus.nodeExporter.enabled=false \
  --set prometheus.kube-state-metrics.disabled=true \
  --set global.prometheus.enabled=false \
  --set global.prometheus.fqdn=http://prometheus.monitoring:9090

# Query cost per namespace
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090
# Visit http://localhost:9090 for the dashboard

# API query for cost data
curl -s "http://localhost:9090/model/allocation?window=30d&aggregate=namespace" | \
  jq '.data[0] | to_entries[] | {namespace: .key, cost: .value.totalCost}'
```

### 10.4 Cluster Consolidation Report

```go
// Go tool to analyze cluster utilization and suggest consolidation
package main

import (
    "context"
    "fmt"
    "os"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/clientcmd"
    metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

func main() {
    config, _ := clientcmd.BuildConfigFromFlags("",
        os.Getenv("HOME")+"/.kube/config")

    clientset, _ := kubernetes.NewForConfig(config)
    metricsClient, _ := metricsv.NewForConfig(config)

    ctx := context.TODO()

    // Get node allocatable resources
    nodes, _ := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
    // Get actual node metrics
    nodeMetrics, _ := metricsClient.MetricsV1beta1().NodeMetricses().List(
        ctx, metav1.ListOptions{})

    fmt.Println("=== Cluster Utilization Report ===")
    fmt.Println()

    for _, node := range nodes.Items {
        allocCPU := node.Status.Allocatable.Cpu().MilliValue()
        allocMem := node.Status.Allocatable.Memory().Value() / (1024 * 1024 * 1024)

        // Find matching metrics
        for _, nm := range nodeMetrics.Items {
            if nm.Name == node.Name {
                usedCPU := nm.Usage.Cpu().MilliValue()
                usedMem := nm.Usage.Memory().Value() / (1024 * 1024 * 1024)

                cpuPct := float64(usedCPU) / float64(allocCPU) * 100
                memPct := float64(usedMem) / float64(allocMem) * 100

                status := "OK"
                if cpuPct < 20 && memPct < 20 {
                    status = "UNDERUTILIZED - consider consolidation"
                }

                fmt.Printf("Node: %s\n", node.Name)
                fmt.Printf("  CPU: %dm / %dm (%.1f%%)\n", usedCPU, allocCPU, cpuPct)
                fmt.Printf("  Memory: %dGi / %dGi (%.1f%%)\n", usedMem, allocMem, memPct)
                fmt.Printf("  Status: %s\n\n", status)
            }
        }
    }
}
```

---

## Exercises

### Exercise 1: Architecture Design Document

Given the TechCorp requirements in Section 1.2, produce an architecture decision record (ADR) that covers: (a) the choice between self-managed vs. managed Kubernetes, (b) the selected CNI plugin with justification, (c) the storage strategy for stateful workloads, and (d) the multi-tenancy model for 5 teams. Present your answer as a structured document.

<details><summary>Show Answer</summary>

```
# ADR-001: Kubernetes Platform Architecture

## Status: Accepted

## Date: 2025-01-15

## Context
TechCorp requires a Kubernetes platform for 25 applications including
microservices, stateful services, and ML workloads. The platform must
support 5 teams, achieve 99.95% SLO, and stay within $25K/month.

## (a) Decision: Managed Kubernetes (AWS EKS)

Rationale:
- Control plane management handled by AWS (HA, upgrades, patching)
- Reduces operational burden vs self-managed (saves ~1 FTE)
- Integrates natively with AWS services (ALB, EBS, S3, IAM)
- Cost: $219/month for EKS control plane vs ~$2,000/month
  for 3 self-managed control plane nodes
- SLA: AWS provides 99.95% uptime SLA for EKS control plane,
  which aligns with our SLO target

Trade-offs:
- Less control over API server configuration
- Upgrade cadence dictated by AWS support windows
- Vendor lock-in for some integrations (IAM for pods, ALB)

## (b) Decision: Cilium CNI

Rationale:
- eBPF-based networking provides better performance than
  iptables-based alternatives (kube-proxy replacement)
- Native AWS ENI integration (no overlay, reduced latency)
- Advanced network policies (L7 HTTP-aware policies)
- Built-in observability with Hubble
- WireGuard encryption for pod-to-pod encryption in transit
- Bandwidth management for QoS

Alternatives considered:
- AWS VPC CNI: simpler but lacks L7 policies and Hubble
- Calico: mature but iptables-based, less performant at scale

## (c) Decision: Storage Strategy

Stateful workloads use dedicated node pools with local NVMe:
- PostgreSQL: EBS io2 volumes (10,000 IOPS), 3 AZ StatefulSet
  with Patroni for HA. StorageClass: gp3 for replicas, io2 for primary.
- Redis: EBS gp3 with persistence. Redis Sentinel for HA.
- Elasticsearch: Local NVMe (i3en instances) for performance.
  Data replicated across 3 nodes for durability.
- Kafka: EBS gp3 with replication factor 3.

Backup strategy:
- EBS snapshots via Velero (every 4 hours)
- PostgreSQL: pg_dump to S3 (every 1 hour, aligns with RPO)
- Kafka: topic data replicated, no separate backup needed

## (d) Decision: Multi-Tenancy Model

Namespace-per-team with RBAC and NetworkPolicy isolation:
- platform-team:    kube-system, monitoring, argocd
- backend-team:     backend, backend-jobs
- frontend-team:    frontend
- data-team:        data-platform, kafka
- ml-team:          ml-training, ml-serving, ml-notebooks

Isolation mechanisms:
- RBAC: Team-specific Roles bound to OIDC groups
- NetworkPolicy: Default deny per namespace, explicit allow rules
- ResourceQuota: CPU, memory, and GPU limits per namespace
- LimitRange: Default resource requests/limits
- PodSecurity: Restricted profile on all team namespaces

Shared resources:
- Ingress controller (system namespace)
- Monitoring stack (monitoring namespace)
- Secret management (external-secrets-operator)
```

</details>

### Exercise 2: Security Hardening Implementation

Implement the following security controls for the production namespace: (a) a Pod Security Standard that prevents privileged containers, (b) a Kyverno policy that requires all pods to have resource limits, (c) network policies that allow only ingress from the ingress controller and monitoring namespaces, and (d) an RBAC configuration that gives the backend team read-write access to deployments but read-only access to secrets.

<details><summary>Show Answer</summary>

```yaml
# (a) Pod Security Standard - Restricted
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.29
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
# (b) Kyverno policy requiring resource limits
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
spec:
  validationFailureAction: Enforce
  background: true
  rules:
    - name: check-limits
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
                - backend
                - frontend
      validate:
        message: "All containers must specify resource requests and limits"
        pattern:
          spec:
            containers:
              - resources:
                  requests:
                    cpu: "?*"
                    memory: "?*"
                  limits:
                    cpu: "?*"
                    memory: "?*"
            =(initContainers):
              - resources:
                  requests:
                    cpu: "?*"
                    memory: "?*"
                  limits:
                    cpu: "?*"
                    memory: "?*"
---
# (c) Network Policies
# Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  egress:
    # Allow DNS
    - to: []
      ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP
    # Allow within namespace
    - to:
        - podSelector: {}
---
# Allow ingress from ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-controller
  namespace: production
spec:
  podSelector:
    matchLabels:
      exposure: external
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-system
---
# Allow monitoring to scrape metrics
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-prometheus-scrape
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
          podSelector:
            matchLabels:
              app.kubernetes.io/name: prometheus
      ports:
        - protocol: TCP
          port: metrics
---
# Allow same-namespace communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector: {}
---
# (d) RBAC for backend team
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: backend-developer
  namespace: production
rules:
  # Read-write on deployments, services, configmaps
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["services", "configmaps", "pods"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
  - apiGroups: ["batch"]
    resources: ["jobs", "cronjobs"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  # Read-only on secrets
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list", "watch"]
  # Read-only on events
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backend-team-binding
  namespace: production
subjects:
  - kind: Group
    name: backend-developers
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: backend-developer
  apiGroup: rbac.authorization.k8s.io
```

</details>

### Exercise 3: Observability Stack Deployment

Write the Helm values and configuration to deploy a complete observability stack: (a) Prometheus with 15-day retention and Thanos sidecar, (b) Grafana with SSO (OIDC), (c) Loki for log aggregation, and (d) alerting rules for the 3 most critical SLOs (API server availability, scheduling latency, node health).

<details><summary>Show Answer</summary>

```yaml
# (a) Prometheus with Thanos sidecar
# helm install prometheus prometheus-community/kube-prometheus-stack -f values.yaml
prometheus:
  prometheusSpec:
    retention: 15d
    retentionSize: 50GB
    replicas: 2
    resources:
      requests:
        cpu: "1"
        memory: 4Gi
      limits:
        cpu: "2"
        memory: 8Gi
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          resources:
            requests:
              storage: 100Gi
    nodeSelector:
      node-role: system
    tolerations:
      - key: node-role
        value: system
        effect: NoSchedule
    thanos:
      image: quay.io/thanos/thanos:v0.34.0
      objectStorageConfig:
        existingSecret:
          name: thanos-objstore-config
          key: objstore.yml
    additionalScrapeConfigs:
      - job_name: gpu-dcgm
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: nvidia-dcgm-exporter
            action: keep

# (b) Grafana with OIDC SSO
grafana:
  replicas: 2
  persistence:
    enabled: true
    size: 10Gi
  grafana.ini:
    server:
      root_url: https://grafana.example.com
    auth.generic_oauth:
      enabled: true
      name: SSO
      allow_sign_up: true
      client_id: grafana-client-id
      client_secret: ${GF_AUTH_GENERIC_OAUTH_CLIENT_SECRET}
      scopes: openid profile email groups
      auth_url: https://sso.example.com/authorize
      token_url: https://sso.example.com/token
      api_url: https://sso.example.com/userinfo
      role_attribute_path: "contains(groups[*], 'platform-team') && 'Admin' || 'Viewer'"
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: default
          folder: Platform
          type: file
          options:
            path: /var/lib/grafana/dashboards/default
        - name: slo
          folder: SLO
          type: file
          options:
            path: /var/lib/grafana/dashboards/slo
  nodeSelector:
    node-role: system
  tolerations:
    - key: node-role
      value: system
      effect: NoSchedule

# (c) Loki
# helm install loki grafana/loki-stack -f loki-values.yaml
---
# loki-values.yaml
loki:
  persistence:
    enabled: true
    size: 50Gi
    storageClassName: gp3
  config:
    limits_config:
      retention_period: 30d
    schema_config:
      configs:
        - from: "2024-01-01"
          store: tsdb
          object_store: s3
          schema: v13
          index:
            prefix: loki_index_
            period: 24h
    storage_config:
      aws:
        s3: s3://us-east-1/prod-loki-logs
        bucketnames: prod-loki-logs
        region: us-east-1

promtail:
  config:
    clients:
      - url: http://loki:3100/loki/api/v1/push
    snippets:
      extraRelabelConfigs:
        - action: drop
          regex: kube-system
          source_labels: [__meta_kubernetes_namespace]

---
# (d) Critical SLO alerting rules
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: critical-slo-alerts
  namespace: monitoring
spec:
  groups:
    # SLO 1: API Server Availability (99.95%)
    - name: api-server-slo
      rules:
        - record: slo:apiserver:availability:5m
          expr: |
            1 - (
              sum(rate(apiserver_request_total{code=~"5.."}[5m]))
              /
              sum(rate(apiserver_request_total[5m]))
            )
        - alert: APIServerAvailabilitySLOBreach
          expr: slo:apiserver:availability:5m < 0.999
          for: 5m
          labels:
            severity: critical
            slo: api-availability
          annotations:
            summary: "API server availability dropped below 99.9% (5m window)"
            description: "Current: {{ $value | humanizePercentage }}. SLO: 99.95%"

    # SLO 2: Scheduling Latency (p99 < 5s)
    - name: scheduling-slo
      rules:
        - record: slo:scheduling:latency_p99:5m
          expr: |
            histogram_quantile(0.99,
              sum(rate(scheduler_scheduling_attempt_duration_seconds_bucket[5m])) by (le)
            )
        - alert: SchedulingLatencySLOBreach
          expr: slo:scheduling:latency_p99:5m > 5
          for: 10m
          labels:
            severity: critical
            slo: scheduling-latency
          annotations:
            summary: "Scheduling p99 latency above 5s SLO"
            description: "Current p99: {{ $value }}s"

    # SLO 3: Node Health (99.9% Ready)
    - name: node-health-slo
      rules:
        - record: slo:node:ready_ratio
          expr: |
            sum(kube_node_status_condition{condition="Ready",status="true"})
            /
            sum(kube_node_status_condition{condition="Ready"})
        - alert: NodeHealthSLOBreach
          expr: slo:node:ready_ratio < 0.999
          for: 5m
          labels:
            severity: critical
            slo: node-health
          annotations:
            summary: "Node readiness below 99.9%"
            description: "{{ $value | humanizePercentage }} nodes Ready"
```

</details>

### Exercise 4: Disaster Recovery Drill

Write a complete DR drill script that validates: (a) etcd backup exists and is less than 6 hours old, (b) Velero can restore a namespace to an isolated test namespace, (c) restored pods reach Running state within 5 minutes, and (d) a report is generated with pass/fail status for each check. The script should exit with code 0 only if all checks pass.

<details><summary>Show Answer</summary>

```bash
#!/usr/bin/env bash
# dr-drill.sh - Automated DR validation
set -euo pipefail

DRILL_ID="drill-$(date +%s)"
TEST_NS="dr-test-${DRILL_ID}"
SOURCE_NS="production"
REPORT="/tmp/dr-report-$(date +%Y%m%d).json"
PASS=0
FAIL=0
RESULTS=()

check() {
    local name="$1"
    local result="$2"
    local detail="$3"

    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    RESULTS+=("{\"check\": \"$name\", \"result\": \"$result\", \"detail\": \"$detail\"}")
    echo "[$result] $name: $detail"
}

echo "=== DR Drill $DRILL_ID Starting ==="
echo ""

# (a) Check etcd backup age
echo "--- Check 1: etcd Backup Freshness ---"
LATEST_ETCD=$(aws s3 ls s3://prod-backups/etcd/ --recursive \
    | sort | tail -1 | awk '{print $1, $2}')
if [ -z "$LATEST_ETCD" ]; then
    check "etcd-backup-exists" "FAIL" "No etcd backup found in S3"
else
    BACKUP_DATE=$(echo "$LATEST_ETCD" | awk '{print $1}')
    BACKUP_TIME=$(echo "$LATEST_ETCD" | awk '{print $2}')
    BACKUP_EPOCH=$(date -d "${BACKUP_DATE} ${BACKUP_TIME}" +%s 2>/dev/null || \
                   date -j -f "%Y-%m-%d %H:%M:%S" "${BACKUP_DATE} ${BACKUP_TIME}" +%s)
    NOW_EPOCH=$(date +%s)
    AGE_HOURS=$(( (NOW_EPOCH - BACKUP_EPOCH) / 3600 ))

    if [ "$AGE_HOURS" -lt 6 ]; then
        check "etcd-backup-freshness" "PASS" "Latest backup is ${AGE_HOURS}h old (< 6h)"
    else
        check "etcd-backup-freshness" "FAIL" "Latest backup is ${AGE_HOURS}h old (> 6h SLA)"
    fi
fi

# Check Velero backup
echo ""
echo "--- Check 2: Velero Backup Availability ---"
LATEST_VELERO=$(velero backup get -o json 2>/dev/null | \
    jq -r '[.items[] | select(.status.phase=="Completed")] |
    sort_by(.status.completionTimestamp) | last | .metadata.name // empty')

if [ -z "$LATEST_VELERO" ]; then
    check "velero-backup-exists" "FAIL" "No completed Velero backup found"
    echo "Cannot proceed with restore test. Exiting."
    FAIL=$((FAIL + 2))
else
    VELERO_AGE=$(velero backup get "$LATEST_VELERO" -o json | \
        jq -r '.status.completionTimestamp')
    check "velero-backup-exists" "PASS" "Latest: $LATEST_VELERO ($VELERO_AGE)"

    # (b) Restore to isolated namespace
    echo ""
    echo "--- Check 3: Velero Restore ---"
    kubectl create namespace "$TEST_NS" 2>/dev/null || true

    RESTORE_NAME="dr-restore-${DRILL_ID}"
    velero restore create "$RESTORE_NAME" \
        --from-backup "$LATEST_VELERO" \
        --include-namespaces "$SOURCE_NS" \
        --namespace-mappings "${SOURCE_NS}:${TEST_NS}" \
        --wait 2>/dev/null

    RESTORE_STATUS=$(velero restore get "$RESTORE_NAME" -o json | \
        jq -r '.status.phase')
    RESTORE_WARNINGS=$(velero restore get "$RESTORE_NAME" -o json | \
        jq -r '.status.warnings // 0')

    if [ "$RESTORE_STATUS" = "Completed" ]; then
        check "velero-restore" "PASS" \
            "Restore completed (warnings: $RESTORE_WARNINGS)"
    else
        check "velero-restore" "FAIL" "Restore status: $RESTORE_STATUS"
    fi

    # (c) Check pod health within 5 minutes
    echo ""
    echo "--- Check 4: Restored Pod Health ---"
    TIMEOUT=300
    ELAPSED=0
    INTERVAL=15

    while [ $ELAPSED -lt $TIMEOUT ]; do
        TOTAL=$(kubectl get pods -n "$TEST_NS" --no-headers 2>/dev/null | wc -l | tr -d ' ')
        RUNNING=$(kubectl get pods -n "$TEST_NS" --no-headers \
            --field-selector=status.phase=Running 2>/dev/null | wc -l | tr -d ' ')

        if [ "$TOTAL" -gt 0 ] && [ "$RUNNING" -eq "$TOTAL" ]; then
            break
        fi
        echo "  Waiting... ${RUNNING}/${TOTAL} pods running (${ELAPSED}s / ${TIMEOUT}s)"
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    done

    TOTAL=$(kubectl get pods -n "$TEST_NS" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    RUNNING=$(kubectl get pods -n "$TEST_NS" --no-headers \
        --field-selector=status.phase=Running 2>/dev/null | wc -l | tr -d ' ')

    if [ "$TOTAL" -gt 0 ] && [ "$RUNNING" -eq "$TOTAL" ]; then
        check "pod-health" "PASS" \
            "${RUNNING}/${TOTAL} pods Running within ${ELAPSED}s"
    else
        check "pod-health" "FAIL" \
            "${RUNNING}/${TOTAL} pods Running after ${TIMEOUT}s timeout"
        # List failed pods
        echo "  Failed pods:"
        kubectl get pods -n "$TEST_NS" --no-headers \
            --field-selector=status.phase!=Running 2>/dev/null | \
            awk '{print "    " $1 " (" $3 ")"}'
    fi

    # Cleanup
    echo ""
    echo "--- Cleanup ---"
    kubectl delete namespace "$TEST_NS" --wait=false 2>/dev/null || true
    velero restore delete "$RESTORE_NAME" --confirm 2>/dev/null || true
fi

# (d) Generate report
echo ""
echo "=== DR Drill Report ==="

RESULTS_JSON=$(printf '%s,' "${RESULTS[@]}")
RESULTS_JSON="[${RESULTS_JSON%,}]"

cat > "$REPORT" <<EOF
{
    "drill_id": "$DRILL_ID",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_checks": $((PASS + FAIL)),
    "passed": $PASS,
    "failed": $FAIL,
    "overall": "$([ $FAIL -eq 0 ] && echo "PASS" || echo "FAIL")",
    "checks": $RESULTS_JSON
}
EOF

echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Overall: $([ $FAIL -eq 0 ] && echo "PASS" || echo "FAIL")"
echo "Report: $REPORT"

# Exit with appropriate code
[ $FAIL -eq 0 ] && exit 0 || exit 1
```

</details>

### Exercise 5: Cost Optimization Plan

Given the cost breakdown in Section 10.1, identify 5 specific cost optimization actions. For each action, provide: (a) the current cost, (b) the projected savings, (c) the implementation steps, and (d) any risks or trade-offs. Target a 20% overall cost reduction.

<details><summary>Show Answer</summary>

```
Cost Optimization Plan - Target: 20% reduction ($5,000/month savings)

Current total: $24,999/month
Target total:  $19,999/month

=== Action 1: Spot Instances for General Pool (nights + weekends) ===

(a) Current:  10x m5.2xlarge on-demand = $5,550/month
(b) Savings:  Mix 50% spot during off-peak = ~$1,665/month saved
(c) Steps:
    1. Create spot node group with same instance type
    2. Configure Karpenter/Cluster Autoscaler to prefer spot
       during 8PM-8AM and weekends
    3. Ensure PDBs protect all workloads
    4. Monitor spot interruption rate
(d) Risks:
    - Spot interruptions during off-peak could affect batch jobs
    - Mitigation: only use for stateless workloads, keep 3 on-demand minimum

=== Action 2: GPU Reserved Instances for Serving ===

(a) Current:  1x p4d.24xlarge on-demand = $10,080/month
(b) Savings:  1-year RI (no upfront) saves 36% = ~$3,629/month saved
              Alternatively, use 2x p3.8xlarge for serving ($2,400)
              and keep p4d for training only (spot)
              Net savings: ~$4,000/month
(c) Steps:
    1. Analyze GPU utilization per workload type (serving vs training)
    2. Purchase 1-year RI for dedicated serving GPU node
    3. Move training to spot p3 instances
    4. Use GPU time-slicing for inference (4 virtual GPUs per physical)
(d) Risks:
    - RI commitment (1 year) - need stable workload forecast
    - Time-slicing reduces per-model GPU memory
    - Mitigation: start with convertible RI, monitor for 1 month first

=== Action 3: Right-Size General Pool Instances ===

(a) Current:  m5.2xlarge (8 vCPU, 32GB) = $555/month each
(b) Savings:  Switch to m5.xlarge (4 vCPU, 16GB) where VPA shows
              requests < 2 CPU / 8GB. Estimated 40% of pods qualify.
              Save ~$1,100/month (4 fewer m5.2xlarge needed)
(c) Steps:
    1. Deploy VPA in recommendation mode for all workloads (2 weeks)
    2. Analyze VPA recommendations vs actual usage
    3. Right-size resource requests based on p95 usage
    4. Reduce node count as requests shrink
(d) Risks:
    - Pods may be too tightly packed, causing contention
    - Mitigation: keep 20% headroom, monitor latency after changes

=== Action 4: Optimize Storage Costs ===

(a) Current:  $1,200/month (EBS io2 + gp3 + S3)
(b) Savings:  ~$350/month
    - Switch Elasticsearch from io2 to gp3 (EBS) or local NVMe
    - Implement S3 lifecycle policies (move old logs to Glacier after 30d)
    - Delete unused PVCs (audit shows 8 orphaned volumes)
(c) Steps:
    1. Audit all PVCs: kubectl get pvc --all-namespaces
    2. Delete orphaned PVCs not bound to any pod
    3. Add S3 lifecycle policy for Loki/Velero buckets
    4. Evaluate gp3 for Elasticsearch (test IOPS first)
(d) Risks:
    - Elasticsearch performance may degrade on gp3
    - Mitigation: benchmark before migrating, keep io2 for primary shard

=== Action 5: Scale Staging/Dev to Zero ===

(a) Current:  Staging runs 24/7 with ~50% of production capacity = ~$3,000/month
(b) Savings:  Scale to zero during non-business hours = ~$1,500/month
(c) Steps:
    1. Install KEDA in staging cluster
    2. Configure ScaledObjects with minReplicas=0 for all services
    3. Scale trigger: HTTP traffic or cron (8AM-8PM weekdays)
    4. Cluster autoscaler scales nodes to 0 when no pods scheduled
(d) Risks:
    - Cold start latency when staging scales from zero (~2-3 min)
    - Nightly integration tests need special handling
    - Mitigation: schedule tests to run during business hours,
      or add a cron trigger to scale up before test window

=== Summary ===

| Action                    | Monthly Savings | Implementation |
|---------------------------|-----------------|----------------|
| 1. Spot for general pool  | $1,665          | 1 week         |
| 2. GPU RI + spot training | $4,000          | 2 weeks        |
| 3. Right-size instances   | $1,100          | 3 weeks        |
| 4. Optimize storage       | $350            | 1 week         |
| 5. Scale staging to zero  | $1,500          | 2 weeks        |
| TOTAL SAVINGS             | $8,615 (34.5%)  |                |

This exceeds the 20% target, providing buffer for growth.
Priority order: 2 → 1 → 5 → 3 → 4 (highest impact first)
```

</details>

---

**Previous**: [18. Kubernetes for ML](./18_Kubernetes_for_ML.md) | **Next**: [00. Overview](./00_Overview.md)
