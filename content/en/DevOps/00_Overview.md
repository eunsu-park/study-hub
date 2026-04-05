# DevOps

## Introduction

This folder contains learning materials for DevOps engineering. Learn step-by-step from DevOps culture and fundamentals through CI/CD pipelines, Infrastructure as Code, configuration management, container orchestration, and service mesh networking.

**Target Audience**: Software engineers, system administrators, platform engineers, and anyone building or operating production systems.

---

## Learning Roadmap

```
[Foundations]              [CI/CD]                  [Infrastructure]
     |                        |                           |
     v                        v                           v
DevOps Fundamentals ---> CI Fundamentals --------> Infrastructure as Code
     |                        |                           |
     v                        v                           v
Version Control -------> GitHub Actions            Terraform Advanced
  Workflows                   |                           |
                              v                           v
                    Container Orchestration     Configuration Management
                        Operations                        |
                              |                           v
                              +-------> Service Mesh & Networking
```

---

## Prerequisites

- [Linux](../Linux/00_Overview.md) -- Command line and system administration fundamentals
- [Git](../Git/00_Overview.md) -- Version control basics
- [Docker](../Docker/00_Overview.md) -- Containerization concepts and usage
- [Cloud Computing](../Cloud_Computing/00_Overview.md) -- Cloud platform familiarity

---

## File List

| File | Difficulty | Main Topics |
|------|------------|-------------|
| [01_DevOps_Fundamentals.md](./01_DevOps_Fundamentals.md) | ⭐ | DevOps culture, CALMS framework, DORA metrics, lifecycle |
| [02_Version_Control_Workflows.md](./02_Version_Control_Workflows.md) | ⭐⭐ | GitFlow, trunk-based development, branching strategies, monorepo vs polyrepo |
| [03_CI_Fundamentals.md](./03_CI_Fundamentals.md) | ⭐⭐ | CI pipeline concepts, build/test/deploy stages, artifact management |
| [04_GitHub_Actions_Deep_Dive.md](./04_GitHub_Actions_Deep_Dive.md) | ⭐⭐⭐ | Workflow syntax, runners, matrix strategy, reusable workflows |
| [05_Infrastructure_as_Code.md](./05_Infrastructure_as_Code.md) | ⭐⭐⭐ | IaC principles, Terraform basics, HCL, state management |
| [06_Terraform_Advanced.md](./06_Terraform_Advanced.md) | ⭐⭐⭐⭐ | Modules, workspaces, data sources, Terragrunt, testing |
| [07_Configuration_Management.md](./07_Configuration_Management.md) | ⭐⭐⭐ | Ansible playbooks, roles, inventory, Jinja2, vault |
| [08_Container_Orchestration_Operations.md](./08_Container_Orchestration_Operations.md) | ⭐⭐⭐⭐ | K8s Deployments, Services, HPA, resource management, rolling updates |
| [09_Service_Mesh_and_Networking.md](./09_Service_Mesh_and_Networking.md) | ⭐⭐⭐⭐ | Istio, Envoy sidecar, traffic management, mTLS, observability |
| [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) | ⭐⭐⭐ | Prometheus, Grafana, PromQL, Alertmanager, USE/RED methods |
| [11_Logging_Infrastructure.md](./11_Logging_Infrastructure.md) | ⭐⭐⭐ | ELK stack, Loki/Grafana, structured logging, log aggregation |
| [12_Distributed_Tracing.md](./12_Distributed_Tracing.md) | ⭐⭐⭐ | OpenTelemetry, Jaeger, traces/spans, context propagation, sampling |
| [13_Deployment_Strategies.md](./13_Deployment_Strategies.md) | ⭐⭐⭐⭐ | Blue-green, canary, rolling updates, A/B testing, feature flags |
| [14_GitOps.md](./14_GitOps.md) | ⭐⭐⭐⭐ | ArgoCD, Flux, pull-based deployment, drift detection, multi-environment |
| [15_Secrets_Management.md](./15_Secrets_Management.md) | ⭐⭐⭐ | HashiCorp Vault, SOPS, Sealed Secrets, dynamic secrets, rotation |
| [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) | ⭐⭐⭐⭐ | Chaos Monkey, Litmus, fault injection, game days, blast radius |
| [17_Platform_Engineering.md](./17_Platform_Engineering.md) | ⭐⭐⭐⭐ | Internal developer platforms, Backstage, golden paths, self-service |
| [18_SRE_Practices.md](./18_SRE_Practices.md) | ⭐⭐⭐⭐ | SRE principles, SLOs/SLIs/SLAs, error budgets, toil elimination, incident management |
| [19_Observability_Engineering.md](./19_Observability_Engineering.md) | ⭐⭐⭐⭐ | Observability vs monitoring, telemetry types, OpenTelemetry, instrumentation |
| [20_SLO_Engineering.md](./20_SLO_Engineering.md) | ⭐⭐⭐⭐ | SLIs/SLOs/SLAs, error budgets, burn rate alerts, SLO-based alerting |
| [21_Signal_Correlation.md](./21_Signal_Correlation.md) | ⭐⭐⭐⭐ | Correlating metrics/logs/traces, exemplars, trace-to-log linking |
| [22_Advanced_Metrics_Architecture.md](./22_Advanced_Metrics_Architecture.md) | ⭐⭐⭐⭐⭐ | Prometheus federation, Thanos/Mimir, cardinality management, recording rules |
| [23_OpenTelemetry_Pipelines.md](./23_OpenTelemetry_Pipelines.md) | ⭐⭐⭐⭐⭐ | OTel Collector architecture, processors, exporters, tail sampling |
| [24_eBPF_Observability.md](./24_eBPF_Observability.md) | ⭐⭐⭐⭐⭐ | eBPF fundamentals, bpftrace, Cilium Hubble, kernel-level observability |
| [25_Continuous_Profiling.md](./25_Continuous_Profiling.md) | ⭐⭐⭐⭐ | CPU/memory profiling, Pyroscope, pprof, flame graphs |
| [26_Incident_Response.md](./26_Incident_Response.md) | ⭐⭐⭐⭐ | On-call practices, incident management, postmortems, runbooks |
| [27_AIOps_Anomaly_Detection.md](./27_AIOps_Anomaly_Detection.md) | ⭐⭐⭐⭐⭐ | ML-based anomaly detection, intelligent alerting, automated remediation |
| [28_Capstone_Full_Stack_Observability.md](./28_Capstone_Full_Stack_Observability.md) | ⭐⭐⭐⭐⭐ | End-to-end observability platform design, tool selection, cost management |

---

## Recommended Learning Order

### Step 1: DevOps Foundations
1. DevOps Fundamentals -- Culture, principles, and metrics
2. Version Control Workflows -- Branching strategies for team collaboration

### Step 2: CI/CD Pipelines
3. CI Fundamentals -- Pipeline concepts and best practices
4. GitHub Actions Deep Dive -- Hands-on CI/CD implementation

### Step 3: Infrastructure as Code
5. Infrastructure as Code -- Terraform fundamentals
6. Terraform Advanced -- Production-grade IaC patterns

### Step 4: Configuration and Orchestration
7. Configuration Management -- Ansible for automation
8. Container Orchestration Operations -- Kubernetes in production

### Step 5: Advanced Networking and Operations
9. Service Mesh and Networking -- Istio and traffic management
10. Monitoring and Alerting -- Prometheus, Grafana, alerting

### Step 6: Operations and Reliability
11. Logging Infrastructure -- ELK stack, Loki, structured logging
12. Distributed Tracing -- OpenTelemetry, Jaeger, context propagation
13. Deployment Strategies -- Blue-green, canary, rolling updates, feature flags
14. GitOps -- ArgoCD, Flux, pull-based deployment
15. Secrets Management -- Vault, SOPS, dynamic secrets
16. Chaos Engineering -- Fault injection, game days, blast radius
17. Platform Engineering -- Internal developer platforms, Backstage
18. SRE Practices -- SLOs, error budgets, toil elimination, incident management

### Step 7: Advanced Observability
19. Observability Engineering -- Observability vs monitoring, OpenTelemetry
20. SLO Engineering -- SLIs/SLOs/SLAs, error budgets, burn rate alerts
21. Signal Correlation -- Correlating metrics, logs, and traces
22. Advanced Metrics Architecture -- Prometheus federation, Thanos, Mimir
23. OpenTelemetry Pipelines -- OTel Collector, tail sampling
24. eBPF Observability -- Kernel-level observability, bpftrace, Hubble
25. Continuous Profiling -- Flame graphs, pprof, Pyroscope
26. Incident Response -- On-call, postmortems, runbooks
27. AIOps and Anomaly Detection -- ML-based alerting, automated remediation
28. Capstone: Full-Stack Observability -- End-to-end platform design

---

## Practice Environment

### Essential Tools

```bash
# Install Terraform
brew install terraform    # macOS
# or download from https://www.terraform.io/downloads

# Install Ansible
pip install ansible

# Install kubectl
brew install kubectl      # macOS

# Install GitHub CLI
brew install gh

# Verify installations
terraform --version
ansible --version
kubectl version --client
gh --version
```

### Local Kubernetes

```bash
# minikube for local K8s
brew install minikube
minikube start

# Verify
kubectl cluster-info
```

---

## Related Materials

- [Docker](../Docker/00_Overview.md) -- Containerization fundamentals
- [Cloud Computing](../Cloud_Computing/00_Overview.md) -- Cloud platforms and services
- [Git](../Git/00_Overview.md) -- Version control system
- [Security](../Security/00_Overview.md) -- Security principles and practices
- [Software Engineering](../Software_Engineering/00_Overview.md) -- Development methodologies

---

**License**: CC BY-NC 4.0
