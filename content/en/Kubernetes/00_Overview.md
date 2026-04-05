# Kubernetes

## Topic Overview

Kubernetes (K8s) is the industry-standard container orchestration platform for automating deployment, scaling, and management of containerized applications. Originally designed by Google and now maintained by the Cloud Native Computing Foundation (CNCF), Kubernetes has become the foundation of modern cloud-native infrastructure — powering everything from small startups to the largest enterprises.

This topic goes far beyond basic `kubectl` commands. It covers Kubernetes architecture internals, advanced networking with CNI plugins, storage systems, security hardening, Custom Resource Definitions, Operators, admission controllers, multi-cluster management, API programming with client-go, production operations, and ML workloads on Kubernetes. The goal is to build deep, transferable knowledge that enables you to design, operate, and extend Kubernetes clusters in production.

This topic assumes familiarity with containers (Docker), Linux fundamentals, and basic networking. Prior completion of the Docker, Linux, and Networking topics is recommended.

## Learning Path

```
Architecture & Core              Networking & Storage             Security
─────────────────                ─────────────────                ─────────────────
01 Architecture        ★★       03 Networking          ★★★      06 RBAC & Security   ★★★
02 Workloads           ★★       04 Storage             ★★       07 Ingress/Gateway   ★★★
                                 05 Config & Secrets    ★★       08 CNI Advanced      ★★★★

Package & Extend                 Operations                       Project
─────────────────                ─────────────────                ─────────────────
09 Helm & Kustomize    ★★★      13 Autoscaling         ★★★      19 Capstone          ★★★★
10 CRDs                ★★★★     14 Observability       ★★★
11 Operators           ★★★★     15 Multi-Cluster       ★★★★     ML & API
12 Admission Ctrl      ★★★★     16 API Programming     ★★★★     ─────────────────
                                 17 Production Ops      ★★★      18 K8s for ML        ★★★
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [Architecture Deep Dive](./01_Architecture_Deep_Dive.md) | ⭐⭐ | Control plane, etcd, API server, scheduler, kubelet |
| 02 | [Workload Resources](./02_Workload_Resources.md) | ⭐⭐ | Deployments, StatefulSets, DaemonSets, Jobs, CronJobs |
| 03 | [Networking Fundamentals](./03_Networking_Fundamentals.md) | ⭐⭐⭐ | Services, DNS, kube-proxy, iptables vs IPVS |
| 04 | [Storage and Persistence](./04_Storage_and_Persistence.md) | ⭐⭐ | PV, PVC, StorageClasses, CSI drivers, dynamic provisioning |
| 05 | [Configuration and Secrets](./05_Configuration_and_Secrets.md) | ⭐⭐ | ConfigMaps, Secrets, external-secrets-operator, sealed-secrets |
| 06 | [RBAC and Security](./06_RBAC_and_Security.md) | ⭐⭐⭐ | RBAC, Pod Security Standards, OPA/Gatekeeper |
| 07 | [Ingress and Gateway API](./07_Ingress_and_Gateway_API.md) | ⭐⭐⭐ | Ingress controllers, Gateway API, TLS termination |
| 08 | [CNI and Advanced Networking](./08_CNI_and_Advanced_Networking.md) | ⭐⭐⭐⭐ | Calico, Cilium, eBPF networking, NetworkPolicy advanced |
| 09 | [Helm and Kustomize](./09_Helm_and_Kustomize.md) | ⭐⭐⭐ | Package management, overlays, Helm chart development |
| 10 | [Custom Resource Definitions](./10_Custom_Resource_Definitions.md) | ⭐⭐⭐⭐ | CRD design, validation, versioning, conversion webhooks |
| 11 | [Operators](./11_Operators.md) | ⭐⭐⭐⭐ | Operator pattern, operator-sdk, kubebuilder, lifecycle management |
| 12 | [Admission Controllers](./12_Admission_Controllers.md) | ⭐⭐⭐⭐ | Validating/mutating webhooks, OPA Gatekeeper policies |
| 13 | [Autoscaling](./13_Autoscaling.md) | ⭐⭐⭐ | HPA, VPA, Cluster Autoscaler, KEDA event-driven scaling |
| 14 | [Observability](./14_Observability.md) | ⭐⭐⭐ | Prometheus stack, EFK/Loki logging, distributed tracing |
| 15 | [Multi-Cluster](./15_Multi_Cluster.md) | ⭐⭐⭐⭐ | Federation, multi-cluster service mesh, Liqo, Submariner |
| 16 | [Kubernetes API Programming](./16_Kubernetes_API_Programming.md) | ⭐⭐⭐⭐ | client-go, informers, dynamic client, controller-runtime |
| 17 | [Production Operations](./17_Production_Operations.md) | ⭐⭐⭐ | Upgrades, etcd backup/restore, disaster recovery, capacity planning |
| 18 | [Kubernetes for ML](./18_Kubernetes_for_ML.md) | ⭐⭐⭐ | GPU scheduling, Kubeflow, model serving on K8s |
| 19 | [Capstone: Production Cluster](./19_Capstone_Production_Cluster.md) | ⭐⭐⭐⭐ | Design and deploy a production-grade cluster with all patterns |

## Prerequisites

- Container fundamentals (Docker images, containers, Dockerfile)
- Linux command line proficiency
- Basic networking concepts (TCP/IP, DNS, HTTP)
- Recommended: [Docker](../Docker/00_Overview.md), [Linux](../Linux/00_Overview.md), [Networking](../Networking/00_Overview.md)

## Environment Setup

```bash
# Install kubectl
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install minikube (local development cluster)
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install Helm
brew install helm  # macOS
# or
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Start a local cluster
minikube start --cpus=4 --memory=8192

# Verify
kubectl cluster-info
kubectl get nodes
helm version
```

## Recommended Resources

- [Kubernetes Official Documentation](https://kubernetes.io/docs/) — Authoritative reference
- [Kubernetes in Action, 2nd Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša
- [Programming Kubernetes](https://www.oreilly.com/library/view/programming-kubernetes/9781492047094/) — Michael Hausenblas & Stefan Schimanski
- [Kubernetes Patterns, 2nd Edition](https://www.oreilly.com/library/view/kubernetes-patterns-2nd/9781098131678/) — Ibryam & Huß
- [CNCF Landscape](https://landscape.cncf.io/) — Cloud-native ecosystem overview
