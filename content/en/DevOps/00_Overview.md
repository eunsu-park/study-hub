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
| [10_Monitoring_and_Observability.md](./10_Monitoring_and_Observability.md) | ⭐⭐⭐ | Prometheus, Grafana, ELK/EFK stack, distributed tracing |
| [11_Log_Management.md](./11_Log_Management.md) | ⭐⭐⭐ | Structured logging, log aggregation, Fluentd, Loki |
| [12_GitOps.md](./12_GitOps.md) | ⭐⭐⭐⭐ | ArgoCD, Flux, GitOps principles, declarative delivery |
| [13_Secrets_Management.md](./13_Secrets_Management.md) | ⭐⭐⭐ | HashiCorp Vault, sealed secrets, SOPS, rotation strategies |
| [14_Cloud_Native_CI_CD.md](./14_Cloud_Native_CI_CD.md) | ⭐⭐⭐⭐ | Tekton, cloud-native pipelines, Argo Workflows |
| [15_Reliability_Engineering.md](./15_Reliability_Engineering.md) | ⭐⭐⭐⭐ | SRE principles, SLOs/SLIs/SLAs, error budgets, incident response |
| [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) | ⭐⭐⭐⭐ | Chaos Monkey, Litmus, fault injection, game days |
| [17_Platform_Engineering.md](./17_Platform_Engineering.md) | ⭐⭐⭐⭐ | Internal developer platforms, Backstage, golden paths |
| [18_DevSecOps.md](./18_DevSecOps.md) | ⭐⭐⭐⭐ | Shift-left security, SAST/DAST, supply chain security, SBOM |

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
10. Monitoring and Observability (planned)

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
