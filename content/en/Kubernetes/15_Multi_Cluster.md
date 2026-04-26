# 15. Multi-Cluster

**Previous**: [Observability](./14_Observability.md) | **Next**: [Kubernetes API Programming](./16_Kubernetes_API_Programming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare multi-cluster architectures (replicated, federated, hub-spoke) and choose the right model
2. Deploy and manage workloads across clusters using Kubernetes Federation v2
3. Configure cross-cluster networking with Submariner and multi-cluster service discovery
4. Set up multi-cluster service mesh with Istio multi-cluster configurations
5. Implement GitOps for multi-cluster environments using ArgoCD ApplicationSets

---

A single Kubernetes cluster has hard limits -- etcd performance degrades beyond ~5,000 nodes, blast radius for control plane failures affects all workloads, and regulatory requirements may mandate geographic data residency. Multi-cluster architectures address these constraints by distributing workloads across independent clusters. But multi-cluster introduces its own complexity: service discovery, cross-cluster networking, consistent configuration, and unified observability. This lesson covers the architectures, tools, and patterns for running Kubernetes at multi-cluster scale.

Before the configurations, read [**Theory & Principles**](#theory--principles) — the four reasons clusters multiply (scale, blast radius, geo, regulation), the topology choices (replicated, federated, hub-spoke) with their consistency trade-offs, why cross-cluster service discovery is fundamentally a DNS + identity problem, and the GitOps model that lets one config drive many clusters without a control plane that owns them.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Multi-Cluster Architectures](#1-multi-cluster-architectures)
- [2. Kubernetes Federation v2](#2-kubernetes-federation-v2)
- [3. Multi-Cluster Service Discovery](#3-multi-cluster-service-discovery)
- [4. Submariner for Cross-Cluster Networking](#4-submariner-for-cross-cluster-networking)
- [5. Liqo for Cluster Sharing](#5-liqo-for-cluster-sharing)
- [6. Multi-Cluster Service Mesh (Istio)](#6-multi-cluster-service-mesh-istio)
- [7. GitOps for Multi-Cluster (ArgoCD ApplicationSets)](#7-gitops-for-multi-cluster-argocd-applicationsets)
- [8. Multi-Cluster Security](#8-multi-cluster-security)
- [Exercises](#exercises)

---

## Theory & Principles

A single Kubernetes cluster is the simplest deployment target — one API, one set of credentials, one observability stack. So the first question is: *why ever run more than one?* This section answers that, then explains the architectural choices for connecting them. The trade-offs are subtle: federation gives you a unified API at the cost of consistency complexity; service mesh-based connectivity gives you cross-cluster traffic but expands the trust boundary; GitOps gives you a unified deployment model without a runtime control plane. The right pick depends on which constraint dominates — scale, blast radius, geography, regulation.

### A. Why Clusters Multiply

Four pressures push organizations beyond one cluster:

- **Scale ceilings.** etcd performance degrades roughly above 5,000 nodes and ~150,000 pods. The official "supported" maximum is well-documented but in practice many teams shard at 1,000–2,000 nodes for predictable upgrade and disaster-recovery characteristics. Past that ceiling, you split workloads across multiple clusters.
- **Blast radius.** A misconfigured admission webhook, a corrupted etcd, a bad cluster upgrade can take down every workload in the cluster. Splitting "prod" into 3 regional clusters means an incident affects 1/3 of users, not all of them. This is the single most important non-scale reason.
- **Geographic latency.** Users in Tokyo should not be served from us-east-1. Multi-region applications need a cluster (or set of clusters) per region, with traffic routed by latency or origin.
- **Regulation and data residency.** EU customer data must stay in the EU; some jurisdictions require local control planes. A single cluster spanning continents is operationally awkward and may be illegal.

A common path: start with one cluster, hit blast radius pain when an upgrade goes sideways, split into "prod" + "staging" + "dev." Then split prod by region for latency. Then add a hub for shared services (logging, monitoring, internal tools). Five clusters happens before you planned.

### B. Three Topologies: Replicated, Federated, Hub-Spoke

The connection model between clusters has three archetypes:

**Replicated (independent clusters).** Each cluster is fully autonomous; you deploy the same workload to multiple clusters and route traffic at the edge (DNS, global load balancer). No cross-cluster state, no cross-cluster control plane. This is the **simplest** model and works for stateless services where each region serves its own users. Operational pain: you have N copies of every config, every secret, every observability dashboard. Mitigated by GitOps (§D).

**Federated (Federation v2 / KubeFed).** A central host cluster holds *federated* versions of resources (`FederatedDeployment`, `FederatedService`); a controller projects them out to member clusters. You write one manifest, it lands in all selected clusters. Pros: centralized API, dynamic placement policies (e.g., "70% to us-east, 30% to us-west"). Cons: the host cluster becomes a single point of failure; the projection lag is observable; debugging "why didn't this propagate?" is its own discipline. Federation v2 is in maintenance mode in 2026 — most production teams use GitOps + service mesh instead.

**Hub-Spoke.** One "hub" cluster runs shared platform services (CI/CD orchestration, observability aggregation, central policy enforcement); workload "spoke" clusters run only application workloads. The hub is small but critical; spokes are large but stateless from the platform's view. This model dominates in enterprises adopting OpenShift / Rancher / Anthos / EKS Anywhere where the vendor provides the hub.

The right choice depends on what you're sharing: nothing across clusters → replicated; resources synchronized across clusters → federated; platform services across clusters → hub-spoke.

### C. Cross-Cluster Service Discovery: A DNS + Identity Problem

Within one cluster, a Pod calls `redis.cache.svc.cluster.local` and CoreDNS resolves it (lesson 03 §D). Across clusters this breaks — `cluster.local` is per-cluster. Three patterns make it work:

**1. Multi-Cluster Services API (KEP-1645).** A standard CRD `ServiceExport` marks a Service as exportable; a controller in each cluster mirrors it under a global DNS name like `redis.cache.svc.clusterset.local`. Pods in any cluster of the clusterset can resolve and reach it. Implementations: AWS Cloud Map, GKE Multi-cluster Services, Submariner.

**2. Service Mesh Multi-Cluster (Istio, Linkerd, Cilium Cluster Mesh).** Sidecars or eBPF programs in each cluster know about Services in *all* peer clusters. Calls to `redis.cache.svc.cluster.local` may transparently land in a remote cluster's Pod. Strong identity (mTLS between sidecars) is the gate; the data plane handles connectivity. Operationally heavy but the most powerful — you get traffic splitting, failover, and locality-aware routing across clusters.

**3. Cluster-aware DNS + flat L3 (Submariner).** Submariner builds an encrypted IPsec tunnel between cluster nodes, making Pods in cluster A directly reachable from Pod IPs in cluster B (no NAT). Combined with a multi-cluster DNS view (Lighthouse), you get the in-cluster experience across clusters. Lighter than service mesh; doesn't give you mTLS or L7 features.

The fundamental insight: cross-cluster connectivity is **not just a networking problem.** Identity (who is the caller, who is the callee), DNS (how do they find each other), and trust (do they validate each other) are all required. Service mesh bundles all three; the others compose them from pieces.

### D. GitOps for Multi-Cluster: The Pull Model Saves You From the Hub

In a GitOps model (Argo CD, Flux), each cluster runs an agent that pulls its desired state from a Git repository. The Git repo is the source of truth; the agent reconciles the cluster to match.

For multi-cluster, this scales beautifully:

- **One repo, many clusters.** A `clusters/` directory has one subdirectory per cluster; each agent pulls only its directory. Adding a cluster = adding a directory + bootstrapping the agent.
- **No central control plane to fail.** If the hub cluster is down, the spoke agents keep reconciling against Git — which is independently HA. This is a fundamental advantage over Federation v2's push model.
- **Argo CD ApplicationSets** generate Argo `Application` resources for many clusters from a template + a generator (cluster list, Git directory, pull request). One template, N clusters, automatic membership tracking.

The mental model: GitOps replaces the "central controller pushes config to spokes" pattern (Federation v2) with "spokes pull config from a shared source" (Argo). Same end state, very different failure modes — the pull model has no central single-point-of-failure beyond the Git server.

For workloads that need cross-cluster *connectivity* (not just deployment), GitOps composes with service mesh: GitOps deploys the mesh + the workloads to each cluster, the mesh handles cross-cluster traffic.

### From Theory to the Configuration Below

The lesson now applies these abstractions:

- **Section 1 (Multi-Cluster Architectures)** is §A and §B — the why and the three topologies with concrete trade-offs.
- **Section 2 (Federation v2)** is §B's federated topology in detail.
- **Section 3 (Multi-Cluster Service Discovery)** is §C — the standardized CRD-based approach.
- **Section 4 (Submariner)** is §C's flat-L3 implementation.
- **Section 5 (Liqo)** is a newer "cluster sharing" model that lets a Pod from cluster A run on cluster B as if it were a virtual node.
- **Section 6 (Multi-Cluster Service Mesh with Istio)** is §C's heaviest, most-featured option.
- **Section 7 (GitOps for Multi-Cluster with ArgoCD ApplicationSets)** is §D in detail.
- **Section 8 (Multi-Cluster Security)** is the cross-cutting concern: identity federation, secret distribution, RBAC consistency.

Once you see clusters as units of failure isolation and §D's pull-based GitOps as the deployment glue, the multi-cluster story decomposes into "what do I need to share?" — config (GitOps), traffic (service mesh), identity (federation), or nothing (replicated).

---

## 1. Multi-Cluster Architectures

### 1.1 Why Multi-Cluster?

| Driver | Single Cluster Problem | Multi-Cluster Solution |
|---|---|---|
| Blast radius | Control plane failure = all workloads affected | Independent failure domains |
| Scale limits | etcd degrades at ~5,000 nodes | Distribute load across clusters |
| Compliance | Data must stay in specific regions | Region-specific clusters |
| Team isolation | Noisy neighbors, RBAC complexity | Dedicated clusters per team/tenant |
| Availability | Single region outage = full downtime | Active-active across regions |
| Upgrade safety | Cluster upgrade affects all workloads | Rolling upgrade across clusters |

### 1.2 Architecture Patterns

```
Pattern 1: Replicated (Standalone)
==================================
Each cluster runs independently with identical deployments.
Load balancer distributes traffic across clusters.

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Cluster A    │    │  Cluster B    │    │  Cluster C    │
  │  (us-east)    │    │  (us-west)    │    │  (eu-west)    │
  │               │    │               │    │               │
  │  ┌──────────┐ │    │  ┌──────────┐ │    │  ┌──────────┐ │
  │  │ App v1.2 │ │    │  │ App v1.2 │ │    │  │ App v1.2 │ │
  │  └──────────┘ │    │  └──────────┘ │    │  └──────────┘ │
  └──────────────┘    └──────────────┘    └──────────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                    ┌───────▼───────┐
                    │  Global LB    │
                    │  (DNS/Anycast) │
                    └───────────────┘


Pattern 2: Federated
====================
A control plane distributes resources across member clusters.

                    ┌──────────────┐
                    │  Federation   │
                    │  Control Plane│
                    └───────┬──────┘
                            │ distribute
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ Cluster A │  │ Cluster B │  │ Cluster C │
       └──────────┘  └──────────┘  └──────────┘


Pattern 3: Hub-Spoke
====================
A management cluster (hub) controls workload clusters (spokes).

              ┌─────────────────────┐
              │     Hub Cluster      │
              │  (management plane)  │
              │  - ArgoCD            │
              │  - Policy engine     │
              │  - Fleet management  │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Spoke 1  │  │  Spoke 2  │  │  Spoke 3  │
    │ (prod-us) │  │(prod-eu)  │  │ (staging) │
    └──────────┘  └──────────┘  └──────────┘
```

### 1.3 Choosing an Architecture

| Pattern | Complexity | Use Case | Cross-Cluster Communication |
|---|---|---|---|
| Replicated | Low | Independent regional deployments | Via external LB only |
| Federated | High | Unified API across clusters | Federation API |
| Hub-Spoke | Medium | Central management, GitOps | Management plane only |

---

## 2. Kubernetes Federation v2

> **Deprecation Notice**: KubeFed (kubernetes-sigs/kubefed) was **archived in
> 2023** and is no longer actively maintained. Do not use it for new projects.
> For multi-cluster management, prefer the active alternatives listed in
> [section 2.3](#23-active-alternatives-to-kubefed) below:
> **Cluster API (CAPI)**, **Open Cluster Management (OCM)**, or **Rancher Fleet**.

### 2.1 KubeFed Architecture

KubeFed (Kubernetes Federation v2) provides a control plane for distributing Kubernetes resources across multiple clusters (documented here for reference only):

```
┌─────────────────────────────────────────────┐
│              Host Cluster                    │
│                                             │
│  ┌────────────────────────────────────────┐  │
│  │         KubeFed Controller             │  │
│  │                                        │  │
│  │  ┌──────────────┐  ┌───────────────┐   │  │
│  │  │ FederatedType │  │   Placement   │   │  │
│  │  │ Controller    │  │   Controller  │   │  │
│  │  └──────────────┘  └───────────────┘   │  │
│  │  ┌──────────────┐  ┌───────────────┐   │  │
│  │  │   Override    │  │   Scheduling  │   │  │
│  │  │  Controller   │  │  Controller   │   │  │
│  │  └──────────────┘  └───────────────┘   │  │
│  └────────────────────────────────────────┘  │
│                                             │
│  ┌────────────────────────────────────────┐  │
│  │  Federated Resources                   │  │
│  │  - FederatedDeployment                 │  │
│  │  - FederatedService                    │  │
│  │  - FederatedConfigMap                  │  │
│  │  - FederatedNamespace                  │  │
│  └────────────────────────────────────────┘  │
└──────────────┬──────────────┬───────────────┘
               │              │
         ┌─────▼─────┐  ┌────▼──────┐
         │ Member     │  │ Member    │
         │ Cluster A  │  │ Cluster B │
         └───────────┘  └───────────┘
```

### 2.2 Installation

```bash
# Install KubeFed in the host cluster
helm repo add kubefed-charts https://raw.githubusercontent.com/kubernetes-sigs/kubefed/master/charts
helm install kubefed kubefed-charts/kubefed \
  --namespace kube-federation-system \
  --create-namespace

# Join member clusters
kubefedctl join cluster-a \
  --cluster-context=cluster-a-context \
  --host-cluster-context=host-cluster-context \
  --v=2

kubefedctl join cluster-b \
  --cluster-context=cluster-b-context \
  --host-cluster-context=host-cluster-context \
  --v=2

# Verify
kubectl get kubefedclusters -n kube-federation-system
```

### 2.3 Federated Deployment

```yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedDeployment
metadata:
  name: web-app
  namespace: production
spec:
  template:
    metadata:
      labels:
        app: web-app
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: web-app
      template:
        metadata:
          labels:
            app: web-app
        spec:
          containers:
          - name: web-app
            image: example.com/web-app:v1.2
            ports:
            - containerPort: 8080
            resources:
              requests:
                cpu: 200m
                memory: 256Mi

  placement:
    clusters:
    - name: cluster-a
    - name: cluster-b
    # Or use clusterSelector:
    # clusterSelector:
    #   matchLabels:
    #     region: us

  overrides:
  - clusterName: cluster-a
    clusterOverrides:
    - path: "/spec/replicas"
      value: 5                    # More replicas in primary cluster
  - clusterName: cluster-b
    clusterOverrides:
    - path: "/spec/replicas"
      value: 2                    # Fewer replicas in secondary
    - path: "/spec/template/spec/containers/0/image"
      value: "example.com/web-app:v1.1"  # Canary: older version in cluster-b
```

### 2.4 Federated Service

```yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedService
metadata:
  name: web-app
  namespace: production
spec:
  template:
    spec:
      selector:
        app: web-app
      ports:
      - port: 80
        targetPort: 8080
      type: ClusterIP
  placement:
    clusters:
    - name: cluster-a
    - name: cluster-b
```

### 2.5 FederatedNamespace

```yaml
# Namespaces must be federated first
apiVersion: types.kubefed.io/v1beta1
kind: FederatedNamespace
metadata:
  name: production
  namespace: production
spec:
  placement:
    clusters:
    - name: cluster-a
    - name: cluster-b
    - name: cluster-c
```

### 2.3 Active Alternatives to KubeFed

Since KubeFed was archived in 2023, the community has converged on three active
projects for multi-cluster management:

#### Cluster API (CAPI)

Cluster API standardizes cluster lifecycle management (provisioning, upgrading,
deleting) using Kubernetes-style declarative APIs. It does not federate
workloads, but provides the infrastructure foundation on which federation tools
are built.

```bash
# Install clusterctl (CAPI CLI)
curl -L https://github.com/kubernetes-sigs/cluster-api/releases/latest/download/clusterctl-linux-amd64 -o clusterctl
chmod +x clusterctl && mv clusterctl /usr/local/bin/

# Initialize the management cluster with the AWS provider
clusterctl init --infrastructure aws

# Generate and apply a cluster definition
clusterctl generate cluster my-cluster \
  --kubernetes-version v1.29.0 \
  --control-plane-machine-count=3 \
  --worker-machine-count=3 | kubectl apply -f -
```

Key concepts: `Cluster`, `Machine`, `MachineDeployment`, `ClusterClass`
(topology-based templating). Providers: AWS, Azure, GCP, vSphere, OpenStack, and more.

#### Open Cluster Management (OCM)

OCM (open-cluster-management.io) provides a hub-spoke model for multi-cluster
governance, placement, and add-on management. It is the foundation for Red Hat
Advanced Cluster Management (RHACM).

```bash
# Install the OCM CLI
brew install open-cluster-management/tap/clusteradm    # macOS
# or: curl -L https://raw.githubusercontent.com/open-cluster-management-io/clusteradm/main/install.sh | bash

# Initialize the hub cluster
clusteradm init --wait

# Join a managed cluster (run on the spoke cluster)
clusteradm join --hub-token <token> --hub-apiserver <hub-url> --cluster-name cluster1

# Accept the join request on the hub
clusteradm accept --clusters cluster1
```

Key concepts: `ManagedCluster`, `ManagedClusterSet`, `Placement`,
`ManifestWork` (push workloads to spokes), `AddOn` (lifecycle managed plugins).

#### Rancher Fleet

Fleet (fleet.rancher.io) is a GitOps-native continuous delivery tool for
Kubernetes that scales to thousands of clusters. It is bundled with Rancher but
can be used standalone.

```bash
# Install Fleet standalone (without Rancher)
helm repo add fleet https://rancher.github.io/fleet-helm-charts/
helm install -n cattle-fleet-system --create-namespace fleet-crd fleet/fleet-crd
helm install -n cattle-fleet-system fleet fleet/fleet

# Create a GitRepo resource to deploy from Git
kubectl apply -f - <<EOF
apiVersion: fleet.cattle.io/v1alpha1
kind: GitRepo
metadata:
  name: my-app
  namespace: fleet-local
spec:
  repo: https://github.com/myorg/my-app
  branch: main
  targets:
    - clusterSelector:
        matchLabels:
          env: production
EOF
```

Key concepts: `GitRepo`, `Bundle`, `ClusterGroup`, `ClusterRegistrationToken`.
Fleet uses a hub-spoke model: the Fleet Manager cluster pushes Bundles to
registered downstream clusters based on label selectors.

| Tool | Primary Use | Workload Federation | Cluster Lifecycle |
|------|-------------|--------------------|--------------------|
| Cluster API | Provisioning clusters | No | Yes |
| OCM | Governance + placement | Via ManifestWork | Via CAPI add-on |
| Rancher Fleet | GitOps delivery | Yes | Via Rancher/CAPI |
| (KubeFed — archived) | Federated resources | Yes (deprecated) | No |

---

## 3. Multi-Cluster Service Discovery

### 3.1 The Discovery Problem

In a single cluster, DNS resolution (`service-name.namespace.svc.cluster.local`) handles service discovery. Across clusters, this breaks because each cluster has its own DNS and network space.

### 3.2 Kubernetes Multi-Cluster Services API (MCS)

The MCS API (KEP-1645) provides a standardized way to export and import services across clusters:

```yaml
# In Cluster A: export a service
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: ServiceExport
metadata:
  name: web-app
  namespace: production
---
# In Cluster B: the service becomes available as:
# web-app.production.svc.clusterset.local
```

### 3.3 ClusterSet Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     ClusterSet                            │
│                                                          │
│  ┌──────────────┐         ┌──────────────┐              │
│  │  Cluster A    │         │  Cluster B    │              │
│  │               │         │               │              │
│  │  Service:     │         │  Service:     │              │
│  │  web-app      │         │  web-app      │              │
│  │  (exported)   │         │  (exported)   │              │
│  └──────┬───────┘         └──────┬───────┘              │
│         │                        │                       │
│         └────────────┬───────────┘                       │
│                      │                                   │
│              ┌───────▼───────┐                           │
│              │ ServiceImport │                           │
│              │               │                           │
│              │ DNS:          │                           │
│              │ web-app.      │                           │
│              │ production.   │                           │
│              │ svc.clusterset│                           │
│              │ .local        │                           │
│              └───────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

### 3.4 DNS-Based Service Discovery

For simpler setups, external DNS can be used for cross-cluster service discovery:

```yaml
# ExternalDNS to register services in Route53/CloudDNS
apiVersion: v1
kind: Service
metadata:
  name: web-app
  namespace: production
  annotations:
    external-dns.alpha.kubernetes.io/hostname: web-app.us-east.example.com
    external-dns.alpha.kubernetes.io/ttl: "60"
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8080
```

```bash
# Global DNS configuration with weighted routing
# Route53 example:
# web-app.example.com -> CNAME
#   - web-app.us-east.example.com (weight: 70)
#   - web-app.eu-west.example.com (weight: 30)
```

---

## 4. Submariner for Cross-Cluster Networking

### 4.1 What is Submariner?

Submariner creates secure network tunnels between Kubernetes clusters, enabling direct pod-to-pod and pod-to-service communication across clusters. It handles overlapping CIDR ranges through its Globalnet component.

### 4.2 Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│    Cluster A          │         │    Cluster B          │
│    10.244.0.0/16     │         │    10.245.0.0/16     │
│                      │         │                      │
│  ┌────────────────┐  │  IPsec  │  ┌────────────────┐  │
│  │   Gateway Node  │◀─────────▶│  │   Gateway Node  │  │
│  │   (submariner-  │  │ tunnel │  │   (submariner-  │  │
│  │    gateway)     │  │         │  │    gateway)     │  │
│  └────────────────┘  │         │  └────────────────┘  │
│                      │         │                      │
│  ┌────────────────┐  │         │  ┌────────────────┐  │
│  │ Route Agent    │  │         │  │ Route Agent    │  │
│  │ (all nodes)    │  │         │  │ (all nodes)    │  │
│  └────────────────┘  │         │  └────────────────┘  │
│                      │         │                      │
│  ┌────────────────┐  │         │  ┌────────────────┐  │
│  │ Lighthouse     │  │         │  │ Lighthouse     │  │
│  │ (DNS discovery)│  │         │  │ (DNS discovery)│  │
│  └────────────────┘  │         │  └────────────────┘  │
└──────────────────────┘         └──────────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
                     ┌───────▼───────┐
                     │  Broker        │
                     │  (metadata     │
                     │   exchange)    │
                     └───────────────┘
```

### 4.3 Installation

```bash
# Install subctl CLI
curl -Ls https://get.submariner.io | bash
export PATH=$PATH:~/.local/bin

# Deploy the broker (on any cluster or a dedicated broker cluster)
subctl deploy-broker --kubeconfig broker-cluster.kubeconfig

# Join clusters to the broker
subctl join --kubeconfig cluster-a.kubeconfig broker-info.subm \
  --clusterid cluster-a \
  --natt=false

subctl join --kubeconfig cluster-b.kubeconfig broker-info.subm \
  --clusterid cluster-b \
  --natt=false

# Verify connectivity
subctl show all
subctl diagnose all
subctl verify --kubeconfig cluster-a.kubeconfig \
  --toconfig cluster-b.kubeconfig \
  --only connectivity
```

### 4.4 Exporting Services

```bash
# Export a service from Cluster A
subctl export service web-app -n production

# Or declaratively
kubectl apply -f - <<EOF
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: ServiceExport
metadata:
  name: web-app
  namespace: production
EOF

# The service becomes accessible from Cluster B as:
# web-app.production.svc.clusterset.local
```

### 4.5 Testing Cross-Cluster Connectivity

```bash
# From a pod in Cluster B, access a service in Cluster A
kubectl exec -it test-pod -- curl http://web-app.production.svc.clusterset.local

# Verify DNS resolution
kubectl exec -it test-pod -- nslookup web-app.production.svc.clusterset.local

# Check Submariner connection status
kubectl get clusters.submariner.io -n submariner-operator
kubectl get endpoints.submariner.io -n submariner-operator
kubectl get gateways.submariner.io -n submariner-operator
```

### 4.6 Globalnet (Overlapping CIDRs)

When clusters have overlapping pod or service CIDRs, Submariner's Globalnet component assigns global virtual IPs:

```bash
# Enable Globalnet during join
subctl join broker-info.subm \
  --clusterid cluster-a \
  --globalnet \
  --globalnet-cidr 242.0.0.0/16

# Each cluster gets a unique global CIDR
# Global IPs are assigned to exported services and pods
kubectl get globalingressips -n production
```

---

## 5. Liqo for Cluster Sharing

### 5.1 What is Liqo?

Liqo enables seamless multi-cluster resource sharing by creating virtual nodes that represent remote clusters. Pods scheduled to a virtual node are transparently offloaded to the remote cluster.

### 5.2 Architecture

```
┌──────────────────────────────────┐
│         Home Cluster              │
│                                   │
│  ┌─────────┐  ┌───────────────┐  │
│  │  Node 1  │  │  Virtual Node │  │
│  │  (real)  │  │  (cluster-b)  │──────▶ Pods run in Cluster B
│  └─────────┘  └───────────────┘  │
│  ┌─────────┐  ┌───────────────┐  │
│  │  Node 2  │  │  Virtual Node │  │
│  │  (real)  │  │  (cluster-c)  │──────▶ Pods run in Cluster C
│  └─────────┘  └───────────────┘  │
│                                   │
│  Scheduler sees virtual nodes     │
│  as regular nodes with capacity   │
└──────────────────────────────────┘
```

### 5.3 Installation and Peering

```bash
# Install Liqo on both clusters
curl -sL https://get.liqo.io | bash

# Or via Helm
helm repo add liqo https://helm.liqo.io
helm install liqo liqo/liqo \
  --namespace liqo-system \
  --create-namespace \
  --set controllerManager.config.enableResourceEnforcement=true

# Generate peering command on Cluster B
liqoctl generate peer-command

# Run the output command on Cluster A to establish peering
liqoctl peer --remote-kubeconfig cluster-b.kubeconfig

# Verify peering
kubectl get foreignclusters
liqoctl status
```

### 5.4 Offloading Workloads

```yaml
# Enable namespace offloading
apiVersion: offloading.liqo.io/v1beta1
kind: NamespaceOffloading
metadata:
  name: offloading
  namespace: production
spec:
  namespaceMappingStrategy: EnforceSameName
  podOffloadingStrategy: LocalAndRemote  # or Remote
  clusterSelector:
    nodeSelectorTerms:
    - matchExpressions:
      - key: liqo.io/remote-cluster-id
        operator: In
        values:
        - cluster-b
        - cluster-c
```

```yaml
# Pods are offloaded using standard node affinity or the Liqo scheduler
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-app
  namespace: production
spec:
  replicas: 6
  selector:
    matchLabels:
      app: distributed-app
  template:
    metadata:
      labels:
        app: distributed-app
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: liqo.io/type
                operator: In
                values:
                - virtual-node  # Schedule only on remote clusters
      containers:
      - name: app
        image: example.com/app:v1
```

### 5.5 Resource Sharing Configuration

```bash
# Configure resource sharing quotas
kubectl annotate foreigncluster cluster-b \
  liqo.io/cpu-sharing-percentage="50" \
  liqo.io/memory-sharing-percentage="50"

# Check available resources from remote clusters
kubectl describe node liqo-cluster-b
# Capacity:
#   cpu:     8     (50% of remote cluster's 16 CPUs)
#   memory:  16Gi  (50% of remote cluster's 32Gi)
```

---

## 6. Multi-Cluster Service Mesh (Istio)

### 6.1 Istio Multi-Cluster Models

Istio supports several multi-cluster deployment models:

```
Model 1: Multi-Primary (each cluster has its own control plane)
===============================================================

  ┌──────────────────┐     ┌──────────────────┐
  │  Cluster A        │     │  Cluster B        │
  │  ┌──────────────┐ │     │  ┌──────────────┐ │
  │  │  istiod      │◀──────▶│  istiod      │ │
  │  │  (primary)   │ │ sync│  │  (primary)   │ │
  │  └──────────────┘ │     │  └──────────────┘ │
  │  ┌──────────────┐ │     │  ┌──────────────┐ │
  │  │  east-west   │◀──────▶│  east-west   │ │
  │  │  gateway     │ │ data│  │  gateway     │ │
  │  └──────────────┘ │     │  └──────────────┘ │
  └──────────────────┘     └──────────────────┘

Model 2: Primary-Remote (one control plane for multiple clusters)
=================================================================

  ┌──────────────────┐     ┌──────────────────┐
  │  Cluster A        │     │  Cluster B        │
  │  ┌──────────────┐ │     │  (no istiod)      │
  │  │  istiod      │──────▶│                  │
  │  │  (primary)   │ │ push│  ┌──────────────┐ │
  │  └──────────────┘ │ config│  east-west   │ │
  │  ┌──────────────┐ │     │  │  gateway     │ │
  │  │  east-west   │◀──────▶│              │ │
  │  │  gateway     │ │ data│  └──────────────┘ │
  │  └──────────────┘ │     └──────────────────┘
  └──────────────────┘
```

### 6.2 Multi-Primary Setup

```bash
# Prerequisites: shared root CA between clusters
# Create a shared root certificate
mkdir -p certs
cd certs

# Generate root CA
make -f istio-1.20.0/tools/certs/Makefile.selfsigned.mk root-ca

# Generate intermediate CA for each cluster
make -f istio-1.20.0/tools/certs/Makefile.selfsigned.mk cluster-a-cacerts
make -f istio-1.20.0/tools/certs/Makefile.selfsigned.mk cluster-b-cacerts

# Install certs as secrets in each cluster
kubectl --context=cluster-a create namespace istio-system
kubectl --context=cluster-a create secret generic cacerts -n istio-system \
  --from-file=cluster-a/ca-cert.pem \
  --from-file=cluster-a/ca-key.pem \
  --from-file=cluster-a/root-cert.pem \
  --from-file=cluster-a/cert-chain.pem

kubectl --context=cluster-b create namespace istio-system
kubectl --context=cluster-b create secret generic cacerts -n istio-system \
  --from-file=cluster-b/ca-cert.pem \
  --from-file=cluster-b/ca-key.pem \
  --from-file=cluster-b/root-cert.pem \
  --from-file=cluster-b/cert-chain.pem
```

### 6.3 Install Istio on Each Cluster

```yaml
# Cluster A IstioOperator
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-cluster-a
spec:
  profile: default
  values:
    global:
      meshID: mesh1
      multiCluster:
        clusterName: cluster-a
      network: network-a
  meshConfig:
    defaultConfig:
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
        ISTIO_META_DNS_AUTO_ALLOCATE: "true"
  components:
    ingressGateways:
    - name: istio-eastwestgateway
      label:
        istio: eastwestgateway
        app: istio-eastwestgateway
        topology.istio.io/network: network-a
      enabled: true
      k8s:
        env:
        - name: ISTIO_META_REQUESTED_NETWORK_VIEW
          value: network-a
        service:
          ports:
          - name: status-port
            port: 15021
            targetPort: 15021
          - name: tls
            port: 15443
            targetPort: 15443
          - name: tls-istiod
            port: 15012
            targetPort: 15012
          - name: tls-webhook
            port: 15017
            targetPort: 15017
```

```bash
# Install on both clusters
istioctl install --context=cluster-a -f cluster-a-operator.yaml
istioctl install --context=cluster-b -f cluster-b-operator.yaml

# Expose services via east-west gateway
kubectl --context=cluster-a apply -f samples/multicluster/expose-services.yaml
kubectl --context=cluster-b apply -f samples/multicluster/expose-services.yaml

# Exchange remote secrets (each cluster needs to know about the other)
istioctl create-remote-secret --context=cluster-a --name=cluster-a | \
  kubectl apply -f - --context=cluster-b

istioctl create-remote-secret --context=cluster-b --name=cluster-b | \
  kubectl apply -f - --context=cluster-a

# Verify multi-cluster
istioctl remote-clusters --context=cluster-a
```

### 6.4 Cross-Cluster Traffic Management

```yaml
# DestinationRule for locality-aware load balancing
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: web-app-dr
  namespace: production
spec:
  host: web-app.production.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
    loadBalancer:
      localityLbSetting:
        enabled: true
        distribute:
        - from: "us-east/*"
          to:
            "us-east/*": 80
            "us-west/*": 20
        - from: "us-west/*"
          to:
            "us-west/*": 80
            "us-east/*": 20
        failover:
        - from: us-east
          to: us-west
        - from: us-west
          to: us-east
```

### 6.5 Multi-Cluster VirtualService

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: web-app-vs
  namespace: production
spec:
  hosts:
  - web-app.production.svc.cluster.local
  http:
  - match:
    - headers:
        x-region:
          exact: us-east
    route:
    - destination:
        host: web-app.production.svc.cluster.local
        subset: cluster-a
      weight: 100
  - route:
    - destination:
        host: web-app.production.svc.cluster.local
      weight: 100  # Default: locality-aware routing
```

---

## 7. GitOps for Multi-Cluster (ArgoCD ApplicationSets)

### 7.1 ArgoCD Multi-Cluster Architecture

```
┌─────────────────────────────────────────┐
│          Management Cluster              │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │         ArgoCD                   │    │
│  │                                  │    │
│  │  ┌──────────────────────────┐   │    │
│  │  │   ApplicationSet         │   │    │
│  │  │   Controller             │   │    │
│  │  │                          │   │    │
│  │  │   Generators:            │   │    │
│  │  │   - List                 │   │    │
│  │  │   - Cluster              │   │    │
│  │  │   - Git                  │   │    │
│  │  │   - Matrix               │   │    │
│  │  └──────────┬───────────────┘   │    │
│  │             │ generates          │    │
│  │             ▼                    │    │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │    │
│  │  │App │ │App │ │App │ │App │   │    │
│  │  │ 1  │ │ 2  │ │ 3  │ │ 4  │   │    │
│  │  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘   │    │
│  └─────┼──────┼──────┼──────┼─────┘    │
│        │      │      │      │           │
└────────┼──────┼──────┼──────┼───────────┘
         │      │      │      │
    ┌────▼──┐ ┌─▼────┐ │  ┌───▼───┐
    │Prod-US│ │Prod-EU│ │  │Staging│
    └───────┘ └──────┘  │  └───────┘
                   ┌────▼───┐
                   │Prod-AP │
                   └────────┘
```

### 7.2 Registering Clusters with ArgoCD

```bash
# Register target clusters in ArgoCD
argocd cluster add cluster-a-context --name prod-us
argocd cluster add cluster-b-context --name prod-eu
argocd cluster add cluster-c-context --name staging

# Verify registered clusters
argocd cluster list

# Or declaratively via Secret
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: prod-us-cluster
  namespace: argocd
  labels:
    argocd.argoproj.io/secret-type: cluster
    environment: production
    region: us-east
type: Opaque
stringData:
  name: prod-us
  server: https://prod-us.example.com:6443
  config: |
    {
      "bearerToken": "<token>",
      "tlsClientConfig": {
        "insecure": false,
        "caData": "<base64-ca>"
      }
    }
EOF
```

### 7.3 ApplicationSet with Cluster Generator

```yaml
# Deploy the same app to all production clusters
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: web-app
  namespace: argocd
spec:
  generators:
  - clusters:
      selector:
        matchLabels:
          environment: production
      values:
        revision: main
  template:
    metadata:
      name: 'web-app-{{name}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/web-app.git
        targetRevision: '{{values.revision}}'
        path: k8s/overlays/{{metadata.labels.region}}
      destination:
        server: '{{server}}'
        namespace: production
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
        - CreateNamespace=true
        retry:
          limit: 5
          backoff:
            duration: 5s
            factor: 2
            maxDuration: 3m
```

### 7.4 ApplicationSet with Git Generator

```yaml
# Generate applications from directory structure in Git
# Repository structure:
# clusters/
#   prod-us/
#     config.json    {"cluster": "prod-us", "region": "us-east", "env": "prod"}
#   prod-eu/
#     config.json    {"cluster": "prod-eu", "region": "eu-west", "env": "prod"}
#   staging/
#     config.json    {"cluster": "staging", "region": "us-east", "env": "staging"}

apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: cluster-configs
  namespace: argocd
spec:
  generators:
  - git:
      repoURL: https://github.com/example/cluster-configs.git
      revision: main
      directories:
      - path: clusters/*
      - path: clusters/experimental-*
        exclude: true
  template:
    metadata:
      name: '{{path.basename}}-config'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/cluster-configs.git
        targetRevision: main
        path: '{{path}}'
      destination:
        server: 'https://{{path.basename}}.example.com:6443'
        namespace: kube-system
```

### 7.5 ApplicationSet with Matrix Generator

```yaml
# Matrix: clusters x applications (Cartesian product)
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: platform-services
  namespace: argocd
spec:
  generators:
  - matrix:
      generators:
      # First dimension: clusters
      - clusters:
          selector:
            matchLabels:
              environment: production
      # Second dimension: applications
      - list:
          elements:
          - app: monitoring
            chart: kube-prometheus-stack
            repoURL: https://prometheus-community.github.io/helm-charts
            version: "55.0.0"
          - app: logging
            chart: loki-stack
            repoURL: https://grafana.github.io/helm-charts
            version: "2.10.0"
          - app: ingress
            chart: ingress-nginx
            repoURL: https://kubernetes.github.io/ingress-nginx
            version: "4.9.0"
  template:
    metadata:
      name: '{{app}}-{{name}}'
    spec:
      project: platform
      source:
        repoURL: '{{repoURL}}'
        chart: '{{chart}}'
        targetRevision: '{{version}}'
        helm:
          valueFiles:
          - values/{{metadata.labels.region}}.yaml
      destination:
        server: '{{server}}'
        namespace: '{{app}}'
      syncPolicy:
        automated:
          selfHeal: true
        syncOptions:
        - CreateNamespace=true
```

### 7.6 Progressive Rollout Across Clusters

```yaml
# Use ApplicationSet with rolling sync strategy
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: web-app-progressive
  namespace: argocd
spec:
  generators:
  - list:
      elements:
      - cluster: staging
        server: https://staging.example.com:6443
        order: "1"
      - cluster: prod-us
        server: https://prod-us.example.com:6443
        order: "2"
      - cluster: prod-eu
        server: https://prod-eu.example.com:6443
        order: "3"
  strategy:
    type: RollingSync
    rollingSync:
      steps:
      - matchExpressions:
        - key: order
          operator: In
          values: ["1"]    # Deploy to staging first
      - matchExpressions:
        - key: order
          operator: In
          values: ["2"]    # Then prod-us
        maxUpdate: 1
      - matchExpressions:
        - key: order
          operator: In
          values: ["3"]    # Finally prod-eu
        maxUpdate: 1
  template:
    metadata:
      name: 'web-app-{{cluster}}'
      labels:
        order: '{{order}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/web-app.git
        path: k8s/base
        targetRevision: main
      destination:
        server: '{{server}}'
        namespace: production
```

---

## 8. Multi-Cluster Security

### 8.1 Identity and Trust

Cross-cluster communication requires a shared trust domain. Options:

```
Option 1: Shared Root CA
========================
    ┌────────────┐
    │  Root CA    │
    └──────┬─────┘
     ┌─────┼─────┐
     ▼     ▼     ▼
  ┌────┐ ┌────┐ ┌────┐
  │Int │ │Int │ │Int │
  │CA-A│ │CA-B│ │CA-C│
  └────┘ └────┘ └────┘
    │      │      │
  Cluster Cluster Cluster
    A      B      C

Option 2: SPIFFE/SPIRE
=======================
Each cluster runs a SPIRE agent.
SPIFFE IDs: spiffe://trust-domain/ns/production/sa/web-app
Cross-cluster trust via federation.
```

### 8.2 Network Policies for Multi-Cluster

```yaml
# Allow traffic only from known remote clusters
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-cross-cluster
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web-app
  policyTypes:
  - Ingress
  ingress:
  # Allow from local namespace
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: production
  # Allow from Submariner gateway CIDR
  - from:
    - ipBlock:
        cidr: 242.0.0.0/16  # Globalnet CIDR
  # Allow from Istio east-west gateway
  - from:
    - namespaceSelector:
        matchLabels:
          istio: system
      podSelector:
        matchLabels:
          istio: eastwestgateway
```

### 8.3 RBAC for Multi-Cluster Management

```yaml
# ArgoCD project with cluster-scoped permissions
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: platform
  namespace: argocd
spec:
  description: Platform services across all clusters
  sourceRepos:
  - 'https://github.com/example/*'
  destinations:
  - namespace: monitoring
    server: '*'  # All registered clusters
  - namespace: logging
    server: '*'
  - namespace: ingress
    server: '*'
  clusterResourceWhitelist:
  - group: ''
    kind: Namespace
  - group: rbac.authorization.k8s.io
    kind: ClusterRole
  - group: rbac.authorization.k8s.io
    kind: ClusterRoleBinding
  namespaceResourceWhitelist:
  - group: '*'
    kind: '*'
  roles:
  - name: platform-admin
    description: Platform team admin
    policies:
    - p, proj:platform:platform-admin, applications, *, platform/*, allow
    groups:
    - platform-team
```

### 8.4 Secret Management Across Clusters

```yaml
# Use External Secrets Operator with a central secret store
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets
---
# Same ExternalSecret deployed to all clusters via ArgoCD
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: db-credentials
  data:
  - secretKey: username
    remoteRef:
      key: production/database
      property: username
  - secretKey: password
    remoteRef:
      key: production/database
      property: password
```

---

## Exercises

### Exercise 1: Multi-Cluster Architecture Design

Your company runs an e-commerce platform that serves customers in North America, Europe, and Asia-Pacific. Requirements: (a) data residency -- EU customer data must stay in EU, (b) latency -- <100ms response time, (c) availability -- survive a full region outage, (d) cost -- minimize cross-region traffic. Design the multi-cluster architecture: choose the pattern (replicated/federated/hub-spoke), describe the cluster topology, explain how traffic is routed, and specify what tools you would use for each concern (service discovery, networking, deployment).

<details>
<summary>Show Answer</summary>

**Architecture: Replicated with Hub-Spoke Management**

```
                     ┌──────────────────────┐
                     │  Management Cluster   │
                     │  (us-east)            │
                     │  - ArgoCD             │
                     │  - Monitoring (Thanos) │
                     │  - Policy (Kyverno)    │
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  prod-us      │  │  prod-eu      │  │  prod-ap      │
    │  (us-east)    │  │  (eu-west)    │  │  (ap-south)   │
    │               │  │               │  │               │
    │  Full app     │  │  Full app     │  │  Full app     │
    │  stack        │  │  stack        │  │  stack        │
    │  US database  │  │  EU database  │  │  AP database  │
    └──────────────┘  └──────────────┘  └──────────────┘
```

**Cluster topology:** 4 clusters total. 1 management hub + 3 regional workload clusters. Each workload cluster runs the complete application stack with its own database for data residency compliance.

**Traffic routing:** Global DNS (Route53 latency-based routing or Cloudflare) directs users to their nearest regional cluster. Within each region, Istio service mesh handles traffic management with locality-aware load balancing. For cross-region failover, DNS health checks remove unhealthy regions and redirect to the next-closest.

**Tool selection:**
- Deployment: ArgoCD with ApplicationSets (cluster generator) on the hub
- Service discovery: Regional -- Kubernetes DNS. Cross-region -- Route53 latency-based DNS
- Networking: Each cluster is independent (no cross-cluster pod networking needed). Cross-region communication uses API calls over public TLS endpoints
- Data sync: Database replication (e.g., CockroachDB multi-region or PostgreSQL logical replication) for non-EU data. EU data stays in eu-west only
- Observability: Thanos for global Prometheus federation. Loki with multi-tenant ingestion
- Secret management: External Secrets Operator with regional AWS Secrets Manager instances

</details>

### Exercise 2: Submariner Cross-Cluster Setup

Write the complete procedure to connect two clusters using Submariner: (a) Cluster A with pod CIDR 10.244.0.0/16 and service CIDR 10.96.0.0/12, (b) Cluster B with pod CIDR 10.244.0.0/16 (overlapping) and service CIDR 10.96.0.0/12. Include: the commands to deploy the broker, join both clusters with Globalnet enabled, export a service from Cluster A, and access it from Cluster B. Provide verification commands.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Install subctl
curl -Ls https://get.submariner.io | bash
export PATH=$PATH:~/.local/bin

# Step 2: Deploy the broker on Cluster A (or a dedicated broker cluster)
subctl deploy-broker --kubeconfig ~/.kube/cluster-a.kubeconfig

# This creates broker-info.subm file containing connection details

# Step 3: Join Cluster A with Globalnet (CIDRs overlap, so Globalnet is required)
subctl join --kubeconfig ~/.kube/cluster-a.kubeconfig \
  broker-info.subm \
  --clusterid cluster-a \
  --globalnet \
  --globalnet-cidr 242.1.0.0/16 \
  --cable-driver libreswan \
  --natt=false

# Step 4: Join Cluster B with Globalnet
subctl join --kubeconfig ~/.kube/cluster-b.kubeconfig \
  broker-info.subm \
  --clusterid cluster-b \
  --globalnet \
  --globalnet-cidr 242.2.0.0/16 \
  --cable-driver libreswan \
  --natt=false

# Step 5: Verify connectivity
subctl show all --kubeconfig ~/.kube/cluster-a.kubeconfig
subctl diagnose all --kubeconfig ~/.kube/cluster-a.kubeconfig
subctl verify --kubeconfig ~/.kube/cluster-a.kubeconfig \
  --toconfig ~/.kube/cluster-b.kubeconfig \
  --only connectivity,service-discovery

# Step 6: Export a service from Cluster A
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  apply -f - <<EOF
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: ServiceExport
metadata:
  name: database
  namespace: production
EOF

# Step 7: Access from Cluster B
kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig run test \
  --image=busybox --rm -it --restart=Never -- \
  nslookup database.production.svc.clusterset.local

kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig run test \
  --image=busybox --rm -it --restart=Never -- \
  wget -qO- http://database.production.svc.clusterset.local:5432

# Step 8: Verify ServiceImport was created in Cluster B
kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig \
  get serviceimports -n production

# Step 9: Check Globalnet IP assignments
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  get globalingressips -n production

kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig \
  get globalingressips -n production

# Step 10: Check gateway status
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  get gateways.submariner.io -n submariner-operator -o wide
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  get clusters.submariner.io -n submariner-operator
```

</details>

### Exercise 3: ArgoCD ApplicationSet

Create an ArgoCD ApplicationSet that: (a) deploys a `payment-service` to all clusters labeled `environment: production`, (b) uses a different Helm values file per region (values/us-east.yaml, values/eu-west.yaml), (c) deploys to staging first, waits for sync, then rolls to production clusters one at a time, (d) includes an automated sync policy with self-heal and prune. Write the full ApplicationSet YAML and the Git repository structure.

<details>
<summary>Show Answer</summary>

Git repository structure:

```
payment-service/
├── Chart.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── values.yaml              # Base values
└── values/
    ├── staging.yaml          # Staging overrides
    ├── us-east.yaml          # US production overrides
    ├── eu-west.yaml          # EU production overrides
    └── ap-south.yaml         # AP production overrides
```

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: payment-service
  namespace: argocd
spec:
  generators:
  - list:
      elements:
      - cluster: staging
        server: https://staging.example.com:6443
        region: staging
        order: "1"
        environment: staging
      - cluster: prod-us
        server: https://prod-us.example.com:6443
        region: us-east
        order: "2"
        environment: production
      - cluster: prod-eu
        server: https://prod-eu.example.com:6443
        region: eu-west
        order: "3"
        environment: production
      - cluster: prod-ap
        server: https://prod-ap.example.com:6443
        region: ap-south
        order: "4"
        environment: production
  strategy:
    type: RollingSync
    rollingSync:
      steps:
      # Step 1: Deploy to staging
      - matchExpressions:
        - key: order
          operator: In
          values: ["1"]
      # Step 2: First production cluster (US)
      - matchExpressions:
        - key: order
          operator: In
          values: ["2"]
        maxUpdate: 1
      # Step 3: Remaining production clusters one at a time
      - matchExpressions:
        - key: order
          operator: In
          values: ["3", "4"]
        maxUpdate: 1
  template:
    metadata:
      name: 'payment-service-{{cluster}}'
      labels:
        order: '{{order}}'
        environment: '{{environment}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/payment-service.git
        targetRevision: main
        path: .
        helm:
          valueFiles:
          - values.yaml
          - values/{{region}}.yaml
      destination:
        server: '{{server}}'
        namespace: payment
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
        - CreateNamespace=true
        - PrunePropagationPolicy=foreground
        retry:
          limit: 5
          backoff:
            duration: 10s
            factor: 2
            maxDuration: 5m
```

</details>

### Exercise 4: Multi-Cluster Istio Service Mesh

Describe the steps to set up Istio multi-primary on two clusters with different networks. Write: (a) the IstioOperator configuration for both clusters, (b) the commands to create and exchange remote secrets, (c) a DestinationRule that implements locality-aware load balancing with failover (80% local, 20% remote, full failover if local is unhealthy), (d) a VirtualService that routes traffic based on headers (x-canary: true routes to cluster-b only).

<details>
<summary>Show Answer</summary>

**(a) IstioOperator for Cluster A:**

```yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-cluster-a
spec:
  profile: default
  values:
    global:
      meshID: shared-mesh
      multiCluster:
        clusterName: cluster-a
      network: network-a
  meshConfig:
    defaultConfig:
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
  components:
    ingressGateways:
    - name: istio-eastwestgateway
      label:
        istio: eastwestgateway
        topology.istio.io/network: network-a
      enabled: true
      k8s:
        env:
        - name: ISTIO_META_REQUESTED_NETWORK_VIEW
          value: network-a
        service:
          ports:
          - name: status-port
            port: 15021
          - name: tls
            port: 15443
          - name: tls-istiod
            port: 15012
          - name: tls-webhook
            port: 15017
```

IstioOperator for Cluster B is identical but with `clusterName: cluster-b` and `network: network-b`.

**(b) Remote secret exchange:**

```bash
# Install Istio on both clusters
istioctl install --context=cluster-a -f cluster-a-operator.yaml -y
istioctl install --context=cluster-b -f cluster-b-operator.yaml -y

# Expose services on east-west gateways
kubectl --context=cluster-a apply -n istio-system -f \
  samples/multicluster/expose-services.yaml
kubectl --context=cluster-b apply -n istio-system -f \
  samples/multicluster/expose-services.yaml

# Create and exchange remote secrets
istioctl create-remote-secret --context=cluster-a --name=cluster-a | \
  kubectl apply -f - --context=cluster-b

istioctl create-remote-secret --context=cluster-b --name=cluster-b | \
  kubectl apply -f - --context=cluster-a

# Verify
istioctl remote-clusters --context=cluster-a
istioctl remote-clusters --context=cluster-b
```

**(c) DestinationRule with locality failover:**

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: web-app-locality
  namespace: production
spec:
  host: web-app.production.svc.cluster.local
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 100
    loadBalancer:
      localityLbSetting:
        enabled: true
        distribute:
        - from: "us-east/us-east-1/*"
          to:
            "us-east/us-east-1/*": 80
            "eu-west/eu-west-1/*": 20
        - from: "eu-west/eu-west-1/*"
          to:
            "eu-west/eu-west-1/*": 80
            "us-east/us-east-1/*": 20
        failover:
        - from: us-east
          to: eu-west
        - from: eu-west
          to: us-east
```

**(d) VirtualService with canary header routing:**

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: web-app-canary
  namespace: production
spec:
  hosts:
  - web-app.production.svc.cluster.local
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: web-app.production.svc.cluster.local
      headers:
        request:
          set:
            x-preferred-locality: "eu-west/eu-west-1"  # Route to cluster-b's region
  - route:
    - destination:
        host: web-app.production.svc.cluster.local
```

Note: Istio's locality-aware routing combined with the `x-preferred-locality` header (or subset-based routing) directs canary traffic to cluster-b. For strict cluster targeting, use subsets with locality labels.

</details>

### Exercise 5: Multi-Cluster Observability

Design a multi-cluster observability stack. Write the configuration for: (a) Thanos to federate Prometheus metrics across 3 clusters (include the Thanos sidecar, store gateway, and query components), (b) Loki in multi-tenant mode receiving logs from all clusters with cluster labels, (c) a Grafana dashboard that shows comparative metrics across clusters (error rate, latency, resource usage per cluster). Include the PromQL queries with the external label for cluster identification.

<details>
<summary>Show Answer</summary>

**(a) Thanos federation:**

```yaml
# Thanos Sidecar (deployed alongside Prometheus in each cluster)
# Add to Prometheus Helm values:
prometheus:
  prometheusSpec:
    externalLabels:
      cluster: prod-us    # Different per cluster
      region: us-east
    thanos:
      image: quay.io/thanos/thanos:v0.34.0
      objectStorageConfig:
        existingSecret:
          name: thanos-objstore-config
          key: objstore.yml
    retention: 24h  # Short retention, Thanos stores long-term

---
# Thanos Store Gateway (central cluster)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-store-gateway
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: thanos-store
  template:
    spec:
      containers:
      - name: thanos-store
        image: quay.io/thanos/thanos:v0.34.0
        args:
        - store
        - --data-dir=/var/thanos/store
        - --objstore.config-file=/etc/thanos/objstore.yml
        - --index-cache-size=500MB
        ports:
        - containerPort: 10901
          name: grpc
        - containerPort: 10902
          name: http
        volumeMounts:
        - name: objstore-config
          mountPath: /etc/thanos
      volumes:
      - name: objstore-config
        secret:
          secretName: thanos-objstore-config
---
# Thanos Query (central cluster)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-query
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: thanos-query
  template:
    spec:
      containers:
      - name: thanos-query
        image: quay.io/thanos/thanos:v0.34.0
        args:
        - query
        - --http-address=0.0.0.0:9090
        - --store=dnssrv+_grpc._tcp.thanos-store-gateway.monitoring.svc
        - --store=thanos-sidecar-prod-us.monitoring.svc:10901
        - --store=thanos-sidecar-prod-eu.monitoring.svc:10901
        - --store=thanos-sidecar-prod-ap.monitoring.svc:10901
        - --query.replica-label=prometheus_replica
        ports:
        - containerPort: 9090
          name: http
```

Object store configuration:

```yaml
# thanos-objstore-config Secret
type: S3
config:
  bucket: thanos-metrics
  endpoint: s3.us-east-1.amazonaws.com
  region: us-east-1
```

**(b) Multi-tenant Loki:**

```yaml
# Loki config with multi-tenant mode
auth_enabled: true  # Enables multi-tenancy

# Promtail in each cluster adds cluster label
# promtail.yaml (per cluster)
clients:
- url: http://loki-central.monitoring.svc:3100/loki/api/v1/push
  tenant_id: prod-us      # Different per cluster
  external_labels:
    cluster: prod-us
    region: us-east
```

**(c) Grafana dashboard queries:**

```promql
# Error rate per cluster (via Thanos query)
sum by (cluster) (
  rate(http_requests_total{status=~"5.."}[5m])
) /
sum by (cluster) (
  rate(http_requests_total[5m])
) * 100

# P99 latency per cluster
histogram_quantile(0.99,
  sum by (cluster, le) (
    rate(http_request_duration_seconds_bucket[5m])
  )
)

# CPU usage per cluster
sum by (cluster) (
  rate(container_cpu_usage_seconds_total{container!=""}[5m])
)

# Memory usage per cluster (GiB)
sum by (cluster) (
  container_memory_working_set_bytes{container!=""}
) / 1024 / 1024 / 1024

# Pod count per cluster
count by (cluster) (kube_pod_info)

# Node count per cluster
count by (cluster) (kube_node_info)

# Cross-cluster comparison: deployment replica drift
sum by (cluster, deployment) (kube_deployment_spec_replicas)
-
sum by (cluster, deployment) (kube_deployment_status_ready_replicas)
```

LogQL for cross-cluster log analysis:

```logql
# Error rate per cluster (Loki)
sum by (cluster) (rate({level="error"}[5m]))

# Errors from a specific cluster
{cluster="prod-us"} | json | level="error"
```

</details>

---

**Previous**: [Observability](./14_Observability.md) | **Next**: [Kubernetes API Programming](./16_Kubernetes_API_Programming.md)
