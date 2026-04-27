# 17. Production Operations

**Previous**: [16. Kubernetes API Programming](./16_Kubernetes_API_Programming.md) | **Next**: [18. Kubernetes for ML](./18_Kubernetes_for_ML.md)

## Learning Objectives

- Plan and execute Kubernetes cluster upgrades using in-place and blue-green strategies
- Perform etcd operations including backup, restore, and defragmentation
- Design disaster recovery plans for Kubernetes clusters
- Conduct capacity planning, node maintenance, and certificate rotation
- Troubleshoot production issues and tune cluster performance

---

Running Kubernetes in production requires mastery of operational tasks that go far beyond deploying workloads. Cluster upgrades must happen without downtime. etcd — the single source of truth — demands careful backup and maintenance. Nodes fail, certificates expire, and capacity must grow with demand. This lesson covers the operational disciplines that keep production Kubernetes clusters healthy, reliable, and performant.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Cluster Upgrade Strategies](#1-cluster-upgrade-strategies)
- [2. etcd Operations](#2-etcd-operations)
- [3. Disaster Recovery Planning](#3-disaster-recovery-planning)
- [4. Capacity Planning and Right-Sizing](#4-capacity-planning-and-right-sizing)
- [5. Node Maintenance](#5-node-maintenance)
- [6. Certificate Management and Rotation](#6-certificate-management-and-rotation)
- [7. Troubleshooting Production Issues](#7-troubleshooting-production-issues)
- [8. Performance Tuning](#8-performance-tuning)
- [9. SLA and SLO for Kubernetes](#9-sla-and-slo-for-kubernetes)
- [Exercises](#exercises)

---

## 1. Cluster Upgrade Strategies

### Theory: Upgrades and the Version-Skew Policy

You cannot just upgrade etcd, then the API server, then nodes in any order. Kubernetes defines a strict **version skew policy** that bounds the version differences between components:

- **Newest API server** (HA: all API server replicas should be within 1 minor version of each other during the rolling upgrade).
- **kube-controller-manager, kube-scheduler, cloud-controller-manager**: at most 1 minor version *older* than the API server.
- **kubelet**: at most 3 minor versions older than the API server (so an N-3 kubelet works against an N API server).
- **kube-proxy**: at most 3 minor versions older.
- **kubectl**: at most 1 minor version different from the API server (older or newer).

The implication: you upgrade in the order **etcd → API server → other control plane → kubelet → kube-proxy → kubectl/clients**. Skipping versions (1.27 → 1.30 directly) is unsupported because in-tree migration code is only written for adjacent versions; you must walk the version ladder one minor at a time.

Two upgrade strategies dominate:

**In-place** (kubeadm upgrade, managed cloud upgrades): drain a node, upgrade kubelet+container runtime in place, uncordon. Repeat per node. Pros: no extra capacity needed; in-cluster IPs and node names remain stable. Cons: the upgrade-in-progress state is observable and can be partially failed. Requires careful PDBs (lesson 02 §7) so drains don't violate availability.

**Blue-green** (provision a new cluster at the new version, migrate workloads, decommission old): zero-downtime by design; lets you test the new cluster before any workload moves. Cons: needs full second-cluster capacity; cross-cluster networking and stateful workload migration are nontrivial. Common in cloud setups using GitOps + multi-cluster service mesh.

The decision is mostly about whether your stateful workloads can tolerate in-cluster restarts (favor in-place) or you need true zero-downtime even for those (favor blue-green).

### 1.1 Kubernetes Version Policy

Kubernetes follows semantic versioning (`MAJOR.MINOR.PATCH`). The project maintains release branches for the three most recent minor versions. The version skew policy governs which component versions are compatible:

```
Version Skew Policy:
┌─────────────────────────────────────────────────────────────┐
│  kube-apiserver     v1.29    (reference version)            │
│  kube-controller-manager     v1.29 or v1.28                 │
│  kube-scheduler              v1.29 or v1.28                 │
│  kubelet                     v1.29, v1.28, or v1.27         │
│  kube-proxy                  same minor as kubelet           │
│  kubectl                     v1.30, v1.29, or v1.28 (+/- 1) │
└─────────────────────────────────────────────────────────────┘

Upgrade Order:
  1. etcd (if separate)
  2. kube-apiserver (all instances)
  3. kube-controller-manager
  4. kube-scheduler
  5. cloud-controller-manager
  6. kubelet + kube-proxy (node by node)
```

### 1.2 In-Place Upgrade

The in-place strategy upgrades the existing cluster components one by one. This is the default approach for kubeadm-managed clusters.

```bash
# Step 1: Check current versions
kubectl get nodes -o wide
kubeadm version

# Step 2: Upgrade kubeadm on the first control plane node
sudo apt-mark unhold kubeadm
sudo apt-get update && sudo apt-get install -y kubeadm=1.29.0-1.1
sudo apt-mark hold kubeadm

# Step 3: Plan the upgrade (dry run)
sudo kubeadm upgrade plan

# Step 4: Apply the upgrade to the control plane
sudo kubeadm upgrade apply v1.29.0

# Step 5: Upgrade kubelet and kubectl on the control plane node
sudo apt-mark unhold kubelet kubectl
sudo apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
sudo apt-mark hold kubelet kubectl
sudo systemctl daemon-reload
sudo systemctl restart kubelet

# Step 6: For additional control plane nodes, use 'node' instead of 'apply'
sudo kubeadm upgrade node

# Step 7: Upgrade worker nodes one by one
# (See Node Maintenance section for drain procedure)
```

### 1.3 Blue-Green Upgrade

The blue-green strategy provisions a complete new cluster (green) alongside the existing one (blue), then switches traffic:

```
Blue-Green Cluster Upgrade:
┌────────────────────────────────────────────────┐
│                Load Balancer / DNS              │
│                                                 │
│    ┌─────────────────┐  ┌─────────────────┐    │
│    │  Blue Cluster    │  │  Green Cluster   │    │
│    │  v1.28 (current) │  │  v1.29 (new)     │    │
│    │                  │  │                  │    │
│    │  ┌──────────┐   │  │  ┌──────────┐   │    │
│    │  │ Workloads │   │  │  │ Workloads │   │    │
│    │  └──────────┘   │  │  │ (migrated) │   │    │
│    │                  │  │  └──────────┘   │    │
│    └─────────────────┘  └─────────────────┘    │
│                                                 │
│  Phase 1: Deploy green, migrate workloads       │
│  Phase 2: Shift traffic to green                │
│  Phase 3: Validate, then decommission blue      │
└────────────────────────────────────────────────┘
```

Blue-green upgrade workflow with infrastructure-as-code:

```bash
# Step 1: Provision new cluster with updated version
terraform apply -var="cluster_version=1.29" -var="cluster_name=prod-green"

# Step 2: Deploy shared infrastructure (Ingress, monitoring, etc.)
kubectl --context=prod-green apply -k infrastructure/

# Step 3: Migrate workloads via GitOps (ArgoCD targets new cluster)
argocd cluster add prod-green
argocd app set my-app --dest-server https://prod-green-api:6443

# Step 4: Run validation tests
kubectl --context=prod-green run smoke-test --image=curlimages/curl \
  --rm -it -- curl -s http://my-app.default.svc/health

# Step 5: Switch DNS / load balancer to green
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456 \
  --change-batch file://switch-to-green.json

# Step 6: Monitor error rates, then decommission blue
terraform destroy -var="cluster_name=prod-blue"
```

### 1.4 Canary Upgrade (Node Pool)

For managed Kubernetes (EKS, GKE, AKS), upgrade node pools incrementally:

```bash
# GKE: Create a new node pool with the updated version
gcloud container node-pools create pool-v129 \
  --cluster=prod-cluster \
  --node-version=1.29.0 \
  --num-nodes=3 \
  --machine-type=e2-standard-4

# Cordon old nodes to prevent new scheduling
kubectl cordon -l cloud.google.com/gke-nodepool=pool-v128

# Drain old nodes one by one
kubectl drain node-old-1 --ignore-daemonsets --delete-emptydir-data

# Verify workloads are healthy on new pool
kubectl get pods -o wide | grep pool-v129

# Delete old node pool
gcloud container node-pools delete pool-v128 --cluster=prod-cluster
```

### 1.5 API Deprecation Management

Each Kubernetes minor release may remove or deprecate API versions. Failing to
migrate before an upgrade breaks existing manifests and CI pipelines. Address
API deprecations **before** upgrading the control plane.

#### Step 1: Discover Deprecated API Usage

```bash
# Query the API server metrics for deprecated API usage (Kubernetes 1.22+)
kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis

# Example output (decode the labels to find the culprit resources):
# apiserver_requested_deprecated_apis{group="extensions",removed_release="1.25",
#   resource="ingresses",subresource="",version="v1beta1"} 1

# Use pluto (static analysis tool) to scan live cluster and local manifests
# https://github.com/FairwindsOps/pluto
pluto detect-all-in-cluster
pluto detect -f ./manifests/

# Use kubectl-convert to list the current API version of a resource
kubectl convert -f ingress.yaml --output-version networking.k8s.io/v1
```

#### Step 2: Convert Manifests

`kubectl convert` migrates manifests from deprecated to current API versions:

```bash
# Install the kubectl-convert plugin
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl-convert"
chmod +x kubectl-convert && mv kubectl-convert /usr/local/bin/

# Convert a single file
kubectl convert -f old-ingress.yaml --output-version networking.k8s.io/v1 > new-ingress.yaml

# Convert all files in a directory
for f in ./manifests/*.yaml; do
  kubectl convert -f "$f" --output-version networking.k8s.io/v1 > "new-${f##*/}" 2>/dev/null || cp "$f" "new-${f##*/}"
done
```

#### Step 3: Track Planned Removals

Key API removals by version (for upgrade planning):

| Removed In | API | Replacement |
|-----------|-----|-------------|
| 1.16 | `extensions/v1beta1` Deployments, DaemonSets, etc. | `apps/v1` |
| 1.22 | `networking.k8s.io/v1beta1` Ingress | `networking.k8s.io/v1` |
| 1.25 | `policy/v1beta1` PodSecurityPolicy | Removed (use PSA or OPA) |
| 1.25 | `batch/v1beta1` CronJob | `batch/v1` |
| 1.26 | `autoscaling/v2beta2` HPA | `autoscaling/v2` |
| 1.27 | `storage.k8s.io/v1beta1` CSIStorageCapacity | `storage.k8s.io/v1` |
| 1.29 | `flowcontrol.apiserver.k8s.io/v1beta2` | `v1beta3` / `v1` |

Always check the [Kubernetes API deprecation guide](https://kubernetes.io/docs/reference/using-api/deprecation-guide/)
before each minor version upgrade.

---

## 2. etcd Operations

### Theory: etcd: The Only True Backup You Have

Lose etcd, lose the cluster — every Kubernetes object lives there (lesson 01 §B). Backups are not optional, and **only restore-tested backups count.** A backup that has never been restored in a drill is a hope, not a backup.

The mechanics are simple. etcd has a built-in snapshot command:

```bash
ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db
```

Run this on a regular schedule (CronJob, systemd timer, cloud backup service); ship the snapshots somewhere etcd cannot reach (off-cluster S3, off-region storage). Test restore quarterly into a sandbox cluster — actually restore the snapshot, not just `etcdctl snapshot status`.

Restore replaces the etcd data directory and restarts etcd as a new cluster (the old member IDs in the snapshot are different from the live members, so a fresh cluster is mandatory). For HA: restore on one member, then add the others fresh — they will sync from the restored member.

Two operational knobs that affect backup quality:

- **Defragmentation.** etcd accumulates fragmentation over time as keys are written and deleted; periodic `etcdctl defrag` reclaims space. Without it, etcd can OOM unexpectedly. Schedule monthly.
- **Auto-compaction.** etcd retains revision history. The `--auto-compaction-retention=8h` flag prunes revisions older than 8 hours, bounding storage growth. Without it, etcd grows unboundedly until full.

Your real **RPO** (max data loss) is the backup interval; your **RTO** (max time to restore) is the snapshot copy + restore + cluster rejoin time. Most well-run clusters target RPO ≤ 1 hour and RTO ≤ 30 minutes, both achievable with hourly snapshots and automated restore tooling.

### 2.1 etcd Architecture in Kubernetes

etcd stores all cluster state: every object, every watch revision, and the entire RBAC configuration. It uses the Raft consensus protocol and requires a quorum of `(n/2)+1` members to function.

```
etcd Cluster (3-member):
┌──────────┐     ┌──────────┐     ┌──────────┐
│  etcd-0   │────│  etcd-1   │────│  etcd-2   │
│  (Leader) │    │ (Follower)│    │ (Follower)│
│           │    │           │    │           │
│  Raft Log │    │  Raft Log │    │  Raft Log │
│  WAL      │    │  WAL      │    │  WAL      │
│  Snapshot │    │  Snapshot │    │  Snapshot │
└──────────┘     └──────────┘     └──────────┘

Quorum: 2 of 3 members must agree
Failure tolerance: 1 member
```

### 2.2 Backup

etcd snapshots capture the complete database state:

```bash
# Check etcd cluster health
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint health

# Create a snapshot backup
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db

# Verify the snapshot
ETCDCTL_API=3 etcdctl snapshot status /backup/etcd-20250115-100000.db --write-table
```

Automated backup with a CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 */6 * * *"    # Every 6 hours
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
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
            - name: backup
              image: bitnami/etcd:3.5
              command:
                - /bin/sh
                - -c
                - |
                  ETCDCTL_API=3 etcdctl \
                    --endpoints=https://127.0.0.1:2379 \
                    --cacert=/etc/kubernetes/pki/etcd/ca.crt \
                    --cert=/etc/kubernetes/pki/etcd/server.crt \
                    --key=/etc/kubernetes/pki/etcd/server.key \
                    snapshot save /backup/etcd-$(date +%Y%m%d-%H%M%S).db

                  # Upload to S3
                  aws s3 cp /backup/etcd-*.db s3://my-etcd-backups/

                  # Clean local backups older than 7 days
                  find /backup -name "etcd-*.db" -mtime +7 -delete
              volumeMounts:
                - name: etcd-certs
                  mountPath: /etc/kubernetes/pki/etcd
                  readOnly: true
                - name: backup-volume
                  mountPath: /backup
          volumes:
            - name: etcd-certs
              hostPath:
                path: /etc/kubernetes/pki/etcd
            - name: backup-volume
              hostPath:
                path: /var/backup/etcd
          restartPolicy: OnFailure
```

### 2.3 Restore

```bash
# Stop all API server instances (or they will conflict with restored data)
sudo mv /etc/kubernetes/manifests/kube-apiserver.yaml /tmp/

# Restore from snapshot to a new data directory
ETCDCTL_API=3 etcdctl snapshot restore /backup/etcd-20250115-100000.db \
  --data-dir=/var/lib/etcd-restored \
  --name=etcd-0 \
  --initial-cluster=etcd-0=https://10.0.1.10:2380 \
  --initial-advertise-peer-urls=https://10.0.1.10:2380

# Replace the old data directory
sudo mv /var/lib/etcd /var/lib/etcd-old
sudo mv /var/lib/etcd-restored /var/lib/etcd
sudo chown -R etcd:etcd /var/lib/etcd

# Restart etcd
sudo systemctl restart etcd

# Restore the API server manifest
sudo mv /tmp/kube-apiserver.yaml /etc/kubernetes/manifests/

# Verify cluster health
kubectl get nodes
kubectl get pods --all-namespaces
```

### 2.4 Defragmentation

Over time etcd accumulates fragmented free space from deleted and compacted revisions:

```bash
# Check database size vs in-use size
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint status --write-table

# Compact old revisions (keep only recent)
# Get current revision
rev=$(ETCDCTL_API=3 etcdctl endpoint status --write-table \
  | awk -F'|' 'NR==2{print $4}' | tr -d ' ')
ETCDCTL_API=3 etcdctl compact $rev

# Defragment each member (one at a time to maintain quorum)
ETCDCTL_API=3 etcdctl \
  --endpoints=https://etcd-0:2379 \
  defrag

ETCDCTL_API=3 etcdctl \
  --endpoints=https://etcd-1:2379 \
  defrag

ETCDCTL_API=3 etcdctl \
  --endpoints=https://etcd-2:2379 \
  defrag
```

### 2.5 etcd Performance Monitoring

```bash
# Check latency metrics
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  check perf --load="s" --prefix="/registry"

# Key Prometheus metrics for etcd
# etcd_disk_wal_fsync_duration_seconds    - WAL fsync latency (should be < 10ms)
# etcd_disk_backend_commit_duration_seconds - DB commit latency
# etcd_server_leader_changes_seen_total   - Leader elections (should be rare)
# etcd_mvcc_db_total_size_in_bytes        - Total DB size
# etcd_mvcc_db_total_size_in_use_in_bytes - In-use size (compare with total)
```

---

## 3. Disaster Recovery Planning

### 3.1 Recovery Objectives

```
Disaster Recovery Metrics:
┌─────────────────────────────────────────────────────┐
│  RPO (Recovery Point Objective)                     │
│  = Maximum acceptable data loss                     │
│  = Time since last backup                           │
│  Example: RPO = 1 hour (backup every hour)          │
│                                                     │
│  RTO (Recovery Time Objective)                      │
│  = Maximum acceptable downtime                      │
│  = Time to restore service                          │
│  Example: RTO = 30 minutes                          │
│                                                     │
│  Recovery Tiers:                                    │
│  Tier 1: Full cluster loss → restore from backup    │
│  Tier 2: Control plane failure → HA failover        │
│  Tier 3: Node failure → rescheduling                │
│  Tier 4: Pod failure → restart/replace              │
└─────────────────────────────────────────────────────┘
```

### 3.2 DR Runbook Components

A complete disaster recovery plan must include:

```yaml
# dr-runbook.yaml
metadata:
  name: kubernetes-dr-plan
  last_tested: "2025-01-15"
  owner: platform-team

recovery_scenarios:
  - name: complete-cluster-loss
    rto: 2h
    rpo: 6h
    steps:
      - provision new infrastructure (terraform)
      - restore etcd from S3 backup
      - bootstrap control plane with kubeadm
      - verify node registration
      - restore persistent volumes from snapshots
      - validate application health
    prerequisites:
      - etcd snapshots in S3 (every 6 hours)
      - infrastructure-as-code in Git
      - PV snapshots enabled (CSI driver)
      - DNS TTL set to 300s

  - name: control-plane-failure
    rto: 5m
    rpo: 0
    steps:
      - HA control plane auto-failover (3 replicas)
      - load balancer health check removes failed node
      - remaining nodes form etcd quorum
    prerequisites:
      - 3+ control plane nodes across AZs
      - external load balancer for API server
      - etcd quorum maintained (2 of 3)

  - name: single-node-failure
    rto: 2m
    rpo: 0
    steps:
      - node marked NotReady after 40s
      - pods evicted after 5m (default)
      - scheduler places pods on healthy nodes
      - cluster autoscaler provisions replacement
    prerequisites:
      - pod disruption budgets defined
      - anti-affinity rules for HA workloads
      - cluster autoscaler configured

backup_locations:
  etcd: s3://prod-backups/etcd/
  velero: s3://prod-backups/velero/
  terraform_state: s3://prod-tfstate/
```

### 3.3 Velero for Application-Level Backup

```bash
# Install Velero with AWS provider
velero install \
  --provider aws \
  --bucket prod-velero-backups \
  --secret-file ./credentials-velero \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1

# Create a backup of a namespace
velero backup create app-backup \
  --include-namespaces production \
  --include-resources deployments,services,configmaps,secrets,pvc

# Schedule regular backups
velero schedule create daily-backup \
  --schedule="0 2 * * *" \
  --include-namespaces production,staging \
  --ttl 720h

# Restore to a different namespace
velero restore create --from-backup app-backup \
  --namespace-mappings production:production-restored
```

---

## 4. Capacity Planning and Right-Sizing

### 4.1 Resource Analysis

```bash
# View actual resource usage vs requests across all nodes
kubectl top nodes

# View pod resource usage
kubectl top pods --all-namespaces --sort-by=memory

# Compare requests vs actual usage
kubectl get pods -o custom-columns=\
"NAME:.metadata.name,\
REQ_CPU:.spec.containers[*].resources.requests.cpu,\
REQ_MEM:.spec.containers[*].resources.requests.memory,\
LIM_CPU:.spec.containers[*].resources.limits.cpu,\
LIM_MEM:.spec.containers[*].resources.limits.memory"
```

### 4.2 Right-Sizing with VPA Recommendations

```yaml
# Install VPA in recommendation-only mode
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: web-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  updatePolicy:
    updateMode: "Off"    # Recommendation only, no auto-update
  resourcePolicy:
    containerPolicies:
      - containerName: web
        minAllowed:
          cpu: 50m
          memory: 64Mi
        maxAllowed:
          cpu: "2"
          memory: 4Gi
```

```bash
# Check VPA recommendations
kubectl get vpa web-app-vpa -o jsonpath='{.status.recommendation.containerRecommendations}' | jq .

# Example output:
# [
#   {
#     "containerName": "web",
#     "lowerBound": {"cpu": "100m", "memory": "128Mi"},
#     "target": {"cpu": "250m", "memory": "256Mi"},
#     "upperBound": {"cpu": "500m", "memory": "512Mi"}
#   }
# ]
```

### 4.3 Cluster Capacity Dashboard

Key metrics for capacity planning:

```
Capacity Planning Metrics:
┌──────────────────────────────────────────────────────────┐
│  Metric                      │ Warning   │ Critical      │
├──────────────────────────────┼───────────┼───────────────┤
│  Node CPU allocation         │ > 70%     │ > 85%         │
│  Node memory allocation      │ > 75%     │ > 90%         │
│  Node count vs autoscaler max│ > 80%     │ > 95%         │
│  PVC utilization             │ > 75%     │ > 90%         │
│  Pod count per node          │ > 80/110  │ > 100/110     │
│  etcd database size          │ > 4GB     │ > 6GB         │
│  API server latency (p99)    │ > 1s      │ > 3s          │
└──────────────────────────────┴───────────┴───────────────┘
```

```yaml
# Prometheus alert rules for capacity
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: capacity-alerts
  namespace: monitoring
spec:
  groups:
    - name: capacity
      rules:
        - alert: NodeCPUAllocationHigh
          expr: |
            sum(kube_pod_container_resource_requests{resource="cpu"}) by (node)
            /
            sum(kube_node_status_allocatable{resource="cpu"}) by (node)
            > 0.85
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: "Node {{ $labels.node }} CPU allocation above 85%"

        - alert: ClusterMemoryPressure
          expr: |
            sum(kube_pod_container_resource_requests{resource="memory"})
            /
            sum(kube_node_status_allocatable{resource="memory"})
            > 0.80
          for: 30m
          labels:
            severity: warning
          annotations:
            summary: "Cluster memory allocation above 80%"
```

### 4.4 Node Pool Strategy

```
Node Pool Design:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  System Pool (dedicated)                                 │
│  ├── 3x m5.xlarge (4 vCPU, 16 GB)                       │
│  ├── Taints: node-role=system:NoSchedule                 │
│  └── Runs: monitoring, ingress, DNS, cert-manager        │
│                                                          │
│  General Pool (autoscaling)                              │
│  ├── 3-20x m5.2xlarge (8 vCPU, 32 GB)                   │
│  ├── No taints                                           │
│  └── Runs: most application workloads                    │
│                                                          │
│  High-Memory Pool (autoscaling)                          │
│  ├── 1-10x r5.2xlarge (8 vCPU, 64 GB)                   │
│  ├── Taints: workload=highmem:NoSchedule                 │
│  └── Runs: databases, caches, JVM applications           │
│                                                          │
│  Spot Pool (cost-optimized)                              │
│  ├── 0-30x m5.2xlarge (spot instances)                   │
│  ├── Taints: workload=spot:NoSchedule                    │
│  └── Runs: batch jobs, dev/test workloads                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Node Maintenance

### 5.1 Cordon, Drain, Uncordon

```bash
# Step 1: Mark node as unschedulable (new pods will not be placed here)
kubectl cordon worker-node-3
# node/worker-node-3 cordoned

# Step 2: Verify node is cordoned
kubectl get node worker-node-3
# NAME            STATUS                     ROLES    AGE   VERSION
# worker-node-3   Ready,SchedulingDisabled   <none>   90d   v1.29.0

# Step 3: Drain all pods (except DaemonSets)
kubectl drain worker-node-3 \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=120 \
  --timeout=300s

# Perform maintenance (kernel update, hardware, etc.)
ssh worker-node-3 "sudo apt-get update && sudo apt-get upgrade -y && sudo reboot"

# Step 4: Uncordon after maintenance
kubectl uncordon worker-node-3
# node/worker-node-3 uncordoned
```

### 5.2 Pod Disruption Budgets

PDBs protect workload availability during voluntary disruptions (drains, upgrades):

```yaml
# Ensure at least 2 replicas are always available during drain
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-app-pdb
  namespace: production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: web-app
---
# Alternatively, allow at most 1 pod to be unavailable
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
  namespace: production
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: api-server
---
# For StatefulSets: percentage-based
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: database-pdb
  namespace: production
spec:
  maxUnavailable: "33%"
  selector:
    matchLabels:
      app: database
```

### 5.3 Automated Node Remediation

```yaml
# Node Problem Detector + remediation
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-problem-detector
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: node-problem-detector
  template:
    metadata:
      labels:
        app: node-problem-detector
    spec:
      tolerations:
        - operator: Exists
      containers:
        - name: node-problem-detector
          image: registry.k8s.io/node-problem-detector/node-problem-detector:v0.8.14
          command:
            - /node-problem-detector
            - --logtostderr
            - --config.system-log-monitor=/config/kernel-monitor.json
          volumeMounts:
            - name: log
              mountPath: /var/log
              readOnly: true
            - name: kmsg
              mountPath: /dev/kmsg
              readOnly: true
          resources:
            requests:
              cpu: 20m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 128Mi
      volumes:
        - name: log
          hostPath:
            path: /var/log/
        - name: kmsg
          hostPath:
            path: /dev/kmsg
```

---

## 6. Certificate Management and Rotation

### Theory: Certificates: The Quiet Expiration

A Kubernetes cluster runs ~20 certificates by default — API server cert, etcd peer/client certs, kubelet client cert, kubelet server cert, controller-manager cert, scheduler cert, front-proxy cert, ServiceAccount signing key, plus any certs for ingress and webhooks. Most are issued at cluster creation with a 1-year validity (kubeadm default). On the 366th day, they expire, and parts of the cluster stop talking to each other.

The recovery is well-known but disruptive:

```bash
kubeadm certs renew all
systemctl restart kubelet
```

But a much better discipline is to *not let it happen*:

- **Monitor expiration.** `kubeadm certs check-expiration` lists every cert with its remaining validity. Run this in a CronJob; alert when any cert has < 30 days. Most production clusters integrate this into their observability stack.
- **Rotate kubelet certs automatically.** `--rotate-certificates` and `--rotate-server-certificates` flags on the kubelet plus the `RotateKubeletClientCertificate` and `RotateKubeletServerCertificate` feature gates make kubelet certs rotate without manual action.
- **Increase validity at creation time.** kubeadm allows custom validity (`--cert-validity-period`); some teams use 5-year certs to reduce rotation frequency, accepting the security trade-off.
- **Use cert-manager for application TLS.** Automate everything user-facing (Ingress certs, webhook server certs); only the cluster-internal certs need the kubeadm/manual path.

Cert expiry is the most preventable production incident in Kubernetes. The reason it happens is that the certs are invisible until they break — alerting on remaining lifetime makes them visible.

### 6.1 Kubernetes PKI Overview

```
Kubernetes Certificate Architecture:
┌─────────────────────────────────────────────────────────────┐
│  /etc/kubernetes/pki/                                       │
│  ├── ca.crt / ca.key                 (Cluster CA, 10 years) │
│  ├── apiserver.crt / apiserver.key   (API server, 1 year)   │
│  ├── apiserver-kubelet-client.crt    (API→kubelet, 1 year)  │
│  ├── front-proxy-ca.crt / key       (Front proxy CA)        │
│  ├── front-proxy-client.crt / key   (Aggregation layer)     │
│  ├── sa.pub / sa.key                (ServiceAccount signing) │
│  └── etcd/                                                   │
│      ├── ca.crt / ca.key            (etcd CA, 10 years)     │
│      ├── server.crt / server.key    (etcd server, 1 year)   │
│      ├── peer.crt / peer.key        (etcd peer, 1 year)     │
│      └── healthcheck-client.crt/key (etcd health, 1 year)   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Checking Certificate Expiration

```bash
# Check all kubeadm-managed certificates
sudo kubeadm certs check-expiration

# Example output:
# CERTIFICATE                EXPIRES                  RESIDUAL TIME
# admin.conf                 Jan 15, 2026 10:00 UTC   364d
# apiserver                  Jan 15, 2026 10:00 UTC   364d
# apiserver-kubelet-client   Jan 15, 2026 10:00 UTC   364d
# controller-manager.conf    Jan 15, 2026 10:00 UTC   364d
# etcd-healthcheck-client    Jan 15, 2026 10:00 UTC   364d
# etcd-peer                  Jan 15, 2026 10:00 UTC   364d
# etcd-server                Jan 15, 2026 10:00 UTC   364d
# front-proxy-client         Jan 15, 2026 10:00 UTC   364d
# scheduler.conf             Jan 15, 2026 10:00 UTC   364d

# Check a specific certificate with OpenSSL
openssl x509 -in /etc/kubernetes/pki/apiserver.crt -noout -dates -subject -issuer
```

### 6.3 Certificate Rotation

```bash
# Renew all kubeadm-managed certificates
sudo kubeadm certs renew all

# Renew a specific certificate
sudo kubeadm certs renew apiserver

# After renewal, restart control plane components
# (Static pods auto-restart when manifests change)
sudo crictl pods --name kube-apiserver -q | xargs sudo crictl stopp
sudo crictl pods --name kube-controller-manager -q | xargs sudo crictl stopp
sudo crictl pods --name kube-scheduler -q | xargs sudo crictl stopp

# Update kubeconfig files
sudo kubeadm kubeconfig user --client-name=admin --org=system:masters \
  > /etc/kubernetes/admin.conf

# Enable automatic kubelet certificate rotation
# In /var/lib/kubelet/config.yaml:
# rotateCertificates: true
# serverTLSBootstrap: true
```

### 6.4 Monitoring Certificate Expiration

```yaml
# Prometheus alert for certificate expiration
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cert-expiration-alerts
  namespace: monitoring
spec:
  groups:
    - name: certificates
      rules:
        - alert: KubernetesCertExpiringSoon
          expr: |
            apiserver_client_certificate_expiration_seconds_count > 0
            and
            histogram_quantile(0.01,
              rate(apiserver_client_certificate_expiration_seconds_bucket[5m])
            ) < 604800
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "Kubernetes client certificate expires within 7 days"

        - alert: KubeletCertExpiring
          expr: |
            kubelet_certificate_manager_server_expiration_renew_errors > 0
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: "Kubelet certificate renewal failing on {{ $labels.node }}"
```

---

## 7. Troubleshooting Production Issues

### 7.1 Systematic Troubleshooting Flow

```
Troubleshooting Decision Tree:
                    ┌─────────────┐
                    │  Pod Issue?  │
                    └──────┬──────┘
              ┌────────────┼────────────┐
         Pending      CrashLoop     Evicted
              │            │            │
      ┌───────┴──┐   ┌────┴────┐  ┌───┴────┐
      │Scheduling│   │ OOM?    │  │Resource │
      │ Failure  │   │ Config? │  │Pressure?│
      └──────────┘   │ Deps?   │  └────────┘
                      └─────────┘
```

### 7.2 Common Troubleshooting Commands

```bash
# Pod stuck in Pending: check events and node resources
kubectl describe pod <pod-name> -n <namespace>
kubectl get events --sort-by=.lastTimestamp -n <namespace>
kubectl describe nodes | grep -A 5 "Allocated resources"

# Pod in CrashLoopBackOff: check logs
kubectl logs <pod-name> -n <namespace> --previous
kubectl logs <pod-name> -n <namespace> -c <container-name>

# Network connectivity issues
kubectl run debug --image=nicolaka/netshoot --rm -it -- \
  bash -c "nslookup kubernetes.default && curl -v http://my-service:8080/health"

# Debug a node
kubectl debug node/worker-node-1 -it --image=ubuntu -- bash

# Check API server responsiveness
kubectl get --raw /healthz
kubectl get --raw /readyz?verbose

# etcd health check
kubectl -n kube-system exec etcd-master-0 -- \
  etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint health
```

### 7.3 OOMKilled Investigation

```bash
# Find OOMKilled pods
kubectl get pods --all-namespaces -o json | \
  jq -r '.items[] |
    select(.status.containerStatuses[]?.lastState.terminated.reason == "OOMKilled") |
    "\(.metadata.namespace)/\(.metadata.name)"'

# Check memory limits vs actual usage
kubectl top pod <pod-name> -n <namespace>
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[0].resources.limits.memory}'

# Check node-level OOM events
kubectl get events --all-namespaces --field-selector reason=OOMKilling
```

### 7.4 DNS Troubleshooting

```bash
# Verify CoreDNS is running
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Test DNS resolution from a pod
kubectl run dns-test --image=busybox:1.36 --rm -it -- \
  nslookup kubernetes.default.svc.cluster.local

# Check CoreDNS logs
kubectl logs -n kube-system -l k8s-app=kube-dns --tail=100

# Verify DNS config inside a pod
kubectl exec <pod-name> -- cat /etc/resolv.conf

# Test external DNS resolution
kubectl run dns-test --image=busybox:1.36 --rm -it -- \
  nslookup google.com
```

---

## 8. Performance Tuning

### 8.1 API Server Tuning

```yaml
# /etc/kubernetes/manifests/kube-apiserver.yaml (key flags)
spec:
  containers:
    - command:
        - kube-apiserver
        # Increase concurrent request limits
        - --max-requests-inflight=800          # default: 400
        - --max-mutating-requests-inflight=400 # default: 200
        # Watch cache size (per resource type)
        - --watch-cache-sizes=pods#1000,nodes#100,services#100
        # Enable API priority and fairness
        - --enable-priority-and-fairness=true
        # Audit logging (performance impact — be selective)
        - --audit-policy-file=/etc/kubernetes/audit-policy.yaml
        - --audit-log-path=/var/log/kubernetes/audit.log
        - --audit-log-maxage=7
        - --audit-log-maxbackup=3
        - --audit-log-maxsize=100
```

### 8.2 Scheduler Tuning

```yaml
# Scheduler configuration
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
  - schedulerName: default-scheduler
    plugins:
      score:
        enabled:
          - name: NodeResourcesBalancedAllocation
            weight: 1
          - name: NodeResourcesFit
            weight: 2
    pluginConfig:
      - name: NodeResourcesFit
        args:
          scoringStrategy:
            type: MostAllocated    # Pack nodes tightly (cost savings)
            # type: LeastAllocated # Spread across nodes (performance)
            resources:
              - name: cpu
                weight: 1
              - name: memory
                weight: 1
# Percentage of nodes to score (large clusters)
percentageOfNodesToScore: 50    # default: 0 (automatic)
```

### 8.3 Kubelet Tuning

```yaml
# /var/lib/kubelet/config.yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
# Pod eviction thresholds
evictionHard:
  memory.available: "200Mi"
  nodefs.available: "10%"
  imagefs.available: "15%"
evictionSoft:
  memory.available: "500Mi"
  nodefs.available: "15%"
evictionSoftGracePeriod:
  memory.available: "1m30s"
  nodefs.available: "1m"
# Max pods per node (default 110)
maxPods: 110
# Image garbage collection
imageGCHighThresholdPercent: 85
imageGCLowThresholdPercent: 80
# Registry pull concurrency
serializeImagePulls: false
maxParallelImagePulls: 5
# Pod startup optimization
podsPerCore: 0    # 0 = no limit
```

### 8.4 etcd Performance Tuning

```bash
# Key etcd tuning parameters
# --heartbeat-interval=100      (default, ms between heartbeats)
# --election-timeout=1000       (default, ms before new election)
# --snapshot-count=10000        (default, ops between snapshots)
# --quota-backend-bytes=8589934592  (8GB max DB size)
# --auto-compaction-mode=periodic
# --auto-compaction-retention=8h

# Verify disk I/O performance (etcd needs low-latency storage)
# WAL fsync should be under 10ms
fio --name=etcd-test --ioengine=sync --rw=write \
  --bs=2300 --numjobs=1 --size=22m --runtime=60 \
  --directory=/var/lib/etcd \
  --fdatasync=1
```

---

## 9. SLA and SLO for Kubernetes

### Theory: SLOs, SLIs, and the Discipline of Measuring Health

"Is the cluster healthy?" without an operational definition is unanswerable. The SRE practice — **SLI** (Service Level Indicator), **SLO** (Service Level Objective), **error budget** — turns it into something you can manage.

A typical Kubernetes-platform SLO set:

| SLI | SLO | Why it matters |
|-----|-----|----------------|
| API server p99 GET latency | < 1s | Indicates etcd/apiserver health |
| API server availability | > 99.95% | The plane controllers depend on |
| Pod scheduling latency p95 (Pending → Bound) | < 10s | Indicates scheduler + admission health |
| Container restart rate | < 0.1% per hour | Indicates workload health |
| Successful upgrade rate | 100% within rollout window | Indicates change risk |
| etcd read p99 latency | < 100ms | Indicates etcd disk + network |

The **error budget** = `1 - SLO`. If your API availability SLO is 99.95%, your error budget is 0.05% = ~22 minutes per month. When the budget is exhausted (you've already lost 22 minutes this month), you stop deploying changes until the budget recovers — pushing more change while burning budget makes the next outage worse.

The discipline this enforces: every change is risky, and you balance change velocity against reliability with quantitative limits. Platform teams that adopt SLOs find their incident count drops because the error budget makes "stop deploying" non-political.

### 9.1 Defining Platform SLOs

```
Kubernetes Platform SLOs:
┌──────────────────────────────────────────────────────────────┐
│  SLI (Indicator)          │ SLO (Objective)    │ Measurement │
├───────────────────────────┼────────────────────┼─────────────┤
│  API server availability  │ 99.95% / month     │ /readyz     │
│  API server latency (p99) │ < 1s for non-list  │ Prometheus  │
│  Scheduling latency (p99) │ < 5s pod-to-bound  │ Prometheus  │
│  Node readiness           │ 99.9% of nodes     │ node cond.  │
│  etcd leader stability    │ < 2 elections/day   │ etcd metrics│
│  DNS resolution success   │ 99.99%             │ CoreDNS     │
│  Ingress error rate       │ < 0.1% 5xx         │ Ingress logs│
│  Deployment rollout time  │ < 5 min for 100    │ events      │
│                           │   replica deploy   │             │
└───────────────────────────┴────────────────────┴─────────────┘
```

### 9.2 Error Budget

```
Error Budget Calculation:
  Monthly SLO: 99.95% availability
  Total minutes in 30 days: 43,200
  Allowed downtime: 43,200 * 0.05% = 21.6 minutes/month

  If 15 minutes consumed by incident:
  Remaining budget: 6.6 minutes
  Budget consumed: 69.4%
  Action: Freeze non-critical changes, focus on reliability
```

### 9.3 SLO-Based Alerting

```yaml
# Multi-window, multi-burn-rate SLO alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: api-server-slo
  namespace: monitoring
spec:
  groups:
    - name: api-server-slo
      rules:
        # Error ratio recording rule
        - record: apiserver:error_ratio:rate5m
          expr: |
            sum(rate(apiserver_request_total{code=~"5.."}[5m]))
            /
            sum(rate(apiserver_request_total[5m]))

        # Fast burn (2% of monthly budget in 1 hour)
        - alert: APIServerHighErrorRate
          expr: |
            apiserver:error_ratio:rate5m > (14.4 * 0.0005)
          for: 2m
          labels:
            severity: critical
            slo: api-server-availability
          annotations:
            summary: "API server burning error budget 14.4x faster than allowed"

        # Slow burn (10% of monthly budget in 3 days)
        - alert: APIServerElevatedErrorRate
          expr: |
            apiserver:error_ratio:rate5m > (1.0 * 0.0005)
          for: 1h
          labels:
            severity: warning
            slo: api-server-availability
          annotations:
            summary: "API server error rate elevated above SLO threshold"
```

### 9.4 Platform Observability Dashboard

Essential panels for a Kubernetes platform SLO dashboard:

```
Dashboard Layout:
┌────────────────────────────┬────────────────────────────┐
│  API Server Availability   │  API Server Latency (p99)  │
│  [99.97% current]          │  [243ms current]           │
│  SLO: 99.95%               │  SLO: < 1000ms             │
├────────────────────────────┼────────────────────────────┤
│  Error Budget Remaining    │  Scheduling Latency (p99)  │
│  [72% of monthly budget]   │  [1.2s current]            │
│  ████████░░ 72%            │  SLO: < 5000ms             │
├────────────────────────────┼────────────────────────────┤
│  etcd Leader Changes       │  Node Readiness             │
│  [0 in last 24h]           │  [100% nodes Ready]        │
│  SLO: < 2/day             │  SLO: 99.9%                │
├────────────────────────────┼────────────────────────────┤
│  DNS Resolution Success    │  Ingress 5xx Rate          │
│  [99.998%]                 │  [0.02%]                   │
│  SLO: 99.99%              │  SLO: < 0.1%              │
└────────────────────────────┴────────────────────────────┘
```

---

## Exercises

### Exercise 1: Cluster Upgrade Planning

You have a production cluster running Kubernetes v1.27.8 with 3 control plane nodes and 15 worker nodes. The target version is v1.29.2. Write a detailed upgrade plan that includes: (a) the intermediate version required, (b) the order of operations, (c) how to handle PDBs during worker node upgrades, and (d) a rollback strategy.

<details><summary>Show Answer</summary>

**Upgrade Plan: v1.27.8 to v1.29.2**

**(a) Intermediate version required:**
Kubernetes only supports upgrading one minor version at a time. The path is:
- v1.27.8 → v1.28.latest (e.g., v1.28.6)
- v1.28.6 → v1.29.2

**(b) Order of operations for each minor version jump:**

```bash
# Phase 1: v1.27.8 → v1.28.6
# 1. Backup etcd
ETCDCTL_API=3 etcdctl snapshot save /backup/pre-upgrade-v1.27.8.db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# 2. Check deprecated APIs
kubectl get --raw /metrics | grep apiserver_requested_deprecated_apis

# 3. Upgrade first control plane node
sudo apt-get install -y kubeadm=1.28.6-1.1
sudo kubeadm upgrade plan
sudo kubeadm upgrade apply v1.28.6
sudo apt-get install -y kubelet=1.28.6-1.1 kubectl=1.28.6-1.1
sudo systemctl daemon-reload && sudo systemctl restart kubelet

# 4. Upgrade remaining control plane nodes
for node in cp-2 cp-3; do
  ssh $node "sudo apt-get install -y kubeadm=1.28.6-1.1"
  ssh $node "sudo kubeadm upgrade node"
  ssh $node "sudo apt-get install -y kubelet=1.28.6-1.1 kubectl=1.28.6-1.1"
  ssh $node "sudo systemctl daemon-reload && sudo systemctl restart kubelet"
done

# 5. Upgrade worker nodes (2 at a time respecting PDBs)
for node in worker-{1..15}; do
  kubectl drain $node --ignore-daemonsets --delete-emptydir-data --timeout=300s
  ssh $node "sudo apt-get install -y kubeadm=1.28.6-1.1 kubelet=1.28.6-1.1"
  ssh $node "sudo kubeadm upgrade node"
  ssh $node "sudo systemctl daemon-reload && sudo systemctl restart kubelet"
  kubectl uncordon $node
  # Wait for node to be Ready and pods to stabilize
  kubectl wait --for=condition=Ready node/$node --timeout=120s
done

# Phase 2: Repeat for v1.28.6 → v1.29.2
```

**(c) Handling PDBs:**
- Drain respects PDBs automatically. If a PDB blocks drain, it will wait up to `--timeout`.
- Upgrade workers in batches of 2 (out of 15 = 13% at a time).
- For critical workloads with minAvailable=2 and replicas=3, only one worker can be drained at a time.
- Use `kubectl get pdb --all-namespaces` to identify constraints before starting.

**(d) Rollback strategy:**
- If control plane upgrade fails: restore etcd from pre-upgrade snapshot and reinstall the previous kubeadm/kubelet versions.
- If worker upgrade fails: uncordon the node with the old version; version skew policy allows kubelet v1.27 with API server v1.28.
- Keep the etcd snapshot for at least 48 hours after successful upgrade.

</details>

### Exercise 2: etcd Backup and Restore Drill

Write a complete drill script that: (a) creates a test namespace with a ConfigMap, (b) takes an etcd snapshot, (c) deletes the namespace, (d) restores the snapshot, and (e) verifies the namespace and ConfigMap exist again.

<details><summary>Show Answer</summary>

```bash
#!/usr/bin/env bash
set -euo pipefail

ETCD_OPTS="--endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key"

BACKUP_FILE="/tmp/etcd-drill-$(date +%s).db"

echo "=== Step 1: Create test resources ==="
kubectl create namespace dr-drill-test
kubectl create configmap drill-config \
  -n dr-drill-test \
  --from-literal=key1=value1 \
  --from-literal=key2=value2
kubectl get configmap drill-config -n dr-drill-test -o yaml
echo "Test resources created."

echo "=== Step 2: Take etcd snapshot ==="
ETCDCTL_API=3 etcdctl $ETCD_OPTS snapshot save "$BACKUP_FILE"
ETCDCTL_API=3 etcdctl snapshot status "$BACKUP_FILE" --write-table
echo "Snapshot saved to $BACKUP_FILE"

echo "=== Step 3: Delete test resources ==="
kubectl delete namespace dr-drill-test --wait=true
echo "Namespace deleted. Verifying..."
kubectl get namespace dr-drill-test 2>&1 && echo "ERROR: namespace still exists" || echo "Confirmed: namespace gone."

echo "=== Step 4: Restore etcd snapshot ==="
# Stop API server
sudo mv /etc/kubernetes/manifests/kube-apiserver.yaml /tmp/kube-apiserver.yaml.bak
sleep 10

# Restore to new data dir
sudo rm -rf /var/lib/etcd-restored
ETCDCTL_API=3 etcdctl snapshot restore "$BACKUP_FILE" \
  --data-dir=/var/lib/etcd-restored \
  --name=$(hostname) \
  --initial-cluster="$(hostname)=https://127.0.0.1:2380" \
  --initial-advertise-peer-urls=https://127.0.0.1:2380

# Swap data directories
sudo mv /var/lib/etcd /var/lib/etcd-old
sudo mv /var/lib/etcd-restored /var/lib/etcd
sudo chown -R etcd:etcd /var/lib/etcd 2>/dev/null || true

# Restart etcd
sudo systemctl restart etcd 2>/dev/null || true

# Restore API server
sudo mv /tmp/kube-apiserver.yaml.bak /etc/kubernetes/manifests/kube-apiserver.yaml
echo "Waiting for API server to come back..."
until kubectl get nodes &>/dev/null; do sleep 2; done
echo "API server is back."

echo "=== Step 5: Verify restored resources ==="
kubectl get namespace dr-drill-test
kubectl get configmap drill-config -n dr-drill-test -o yaml

VALUE=$(kubectl get configmap drill-config -n dr-drill-test \
  -o jsonpath='{.data.key1}')
if [ "$VALUE" = "value1" ]; then
  echo "SUCCESS: ConfigMap data verified after restore."
else
  echo "FAILURE: ConfigMap data mismatch. Got: $VALUE"
  exit 1
fi

echo "=== Cleanup ==="
kubectl delete namespace dr-drill-test
rm -f "$BACKUP_FILE"
sudo rm -rf /var/lib/etcd-old
echo "DR drill complete."
```

</details>

### Exercise 3: Node Drain with PDB Constraints

You have 3 worker nodes, each running 2 replicas of `web-app` (6 total, with a PDB of `minAvailable: 4`). Write the procedure to drain all 3 nodes one at a time and explain what happens at each step. What happens if you try to drain 2 nodes simultaneously?

<details><summary>Show Answer</summary>

**Setup:**
- 3 worker nodes: node-1, node-2, node-3
- 6 replicas of web-app spread across nodes (2 per node)
- PDB: minAvailable=4 (at most 2 pods can be unavailable)

**Step-by-step drain procedure:**

```bash
# Step 1: Drain node-1
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data

# What happens:
# - node-1 is cordoned (no new pods scheduled)
# - 2 web-app pods on node-1 are evicted
# - PDB check: 6 - 2 = 4 available >= minAvailable(4) -> ALLOWED
# - Scheduler places the 2 evicted pods on node-2 and node-3
# - Distribution: node-1=0, node-2=3, node-3=3
# Wait for pods to be Running
kubectl wait --for=condition=Ready pod -l app=web-app --timeout=120s

# Step 2: Drain node-2
kubectl drain node-2 --ignore-daemonsets --delete-emptydir-data

# What happens:
# - 3 web-app pods on node-2 need eviction
# - PDB check for first 2: 6 - 2 = 4 >= 4 -> ALLOWED
# - After 2 evicted: 4 remaining, try to evict 3rd
# - PDB check: 4 - 1 = 3 < 4 -> BLOCKED
# - Drain WAITS for the evicted pods to become Ready on node-3
# - Once they are Ready: 6 running total, evict the 3rd
# - Distribution: node-1=0, node-2=0, node-3=6

# Step 3: Drain node-3
kubectl drain node-3 --ignore-daemonsets --delete-emptydir-data

# What happens:
# - 6 web-app pods on node-3 need eviction
# - PDB blocks: 6 - 1 = 5, 6 - 2 = 4 (OK for 2)
# - After 2 evicted, NO nodes available for scheduling (all cordoned)
# - Pods stay Pending
# - PDB blocks further eviction: 4 - 1 = 3 < 4
# - Drain HANGS indefinitely until timeout

# To resolve: uncordon a node first
kubectl uncordon node-1
# Now evicted pods schedule on node-1, drain can proceed
```

**What happens if you drain 2 nodes simultaneously?**

```bash
# Simultaneous drain of node-1 and node-2
kubectl drain node-1 --ignore-daemonsets &
kubectl drain node-2 --ignore-daemonsets &

# Race condition:
# - Both try to evict 2 pods each simultaneously
# - Total eviction attempt: 4 pods at once
# - PDB: 6 - 4 = 2 < minAvailable(4) -> SOME EVICTIONS BLOCKED
# - The API server's eviction API checks PDB atomically per pod
# - Result: Some evictions succeed, others are retried
# - Eventually both drains complete, but it takes longer due to
#   serial PDB-respecting evictions and rescheduling waits
# - If scheduling is tight, one drain may time out
```

</details>

### Exercise 4: Certificate Expiry Response

Write a monitoring and remediation procedure: (a) a Prometheus alerting rule that fires when any Kubernetes component certificate will expire within 30 days, (b) a shell script that renews all certificates and restarts the affected components, and (c) a post-renewal validation checklist.

<details><summary>Show Answer</summary>

**(a) Prometheus alerting rule:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cert-expiry-alerts
  namespace: monitoring
spec:
  groups:
    - name: certificate-expiry
      rules:
        - alert: KubeCertExpiry30Days
          expr: |
            apiserver_client_certificate_expiration_seconds_count > 0
            and on()
            (
              apiserver_client_certificate_expiration_seconds_bucket{le="2592000"}
              /
              ignoring(le) apiserver_client_certificate_expiration_seconds_count
            ) > 0
          for: 1h
          labels:
            severity: warning
          annotations:
            summary: "A Kubernetes client certificate expires within 30 days"
            runbook: "https://wiki.internal/runbooks/cert-renewal"

        - alert: KubeCertExpiry7Days
          expr: |
            apiserver_client_certificate_expiration_seconds_count > 0
            and on()
            (
              apiserver_client_certificate_expiration_seconds_bucket{le="604800"}
              /
              ignoring(le) apiserver_client_certificate_expiration_seconds_count
            ) > 0
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "A Kubernetes client certificate expires within 7 days"
```

**(b) Renewal script:**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-renewal: Check current expiration ==="
sudo kubeadm certs check-expiration

echo "=== Backing up current certificates ==="
BACKUP_DIR="/etc/kubernetes/pki-backup-$(date +%Y%m%d)"
sudo cp -r /etc/kubernetes/pki "$BACKUP_DIR"
echo "Backup saved to $BACKUP_DIR"

echo "=== Renewing all certificates ==="
sudo kubeadm certs renew all

echo "=== Regenerating kubeconfig files ==="
sudo kubeadm kubeconfig user --client-name=admin \
  --org=system:masters > /tmp/admin.conf
sudo cp /tmp/admin.conf /etc/kubernetes/admin.conf
sudo cp /etc/kubernetes/admin.conf ~/.kube/config

echo "=== Restarting control plane components ==="
sudo crictl pods --name kube-apiserver -q | \
  xargs -r sudo crictl stopp
sudo crictl pods --name kube-controller-manager -q | \
  xargs -r sudo crictl stopp
sudo crictl pods --name kube-scheduler -q | \
  xargs -r sudo crictl stopp
sudo crictl pods --name etcd -q | \
  xargs -r sudo crictl stopp

echo "=== Waiting for components to restart ==="
sleep 15
until kubectl get nodes &>/dev/null; do
  echo "Waiting for API server..."
  sleep 5
done

echo "=== Post-renewal: Verify new expiration ==="
sudo kubeadm certs check-expiration
echo "Certificate renewal complete."
```

**(c) Post-renewal validation checklist:**

```bash
# 1. Verify all certificates renewed
sudo kubeadm certs check-expiration | grep -c "364d"

# 2. Verify API server is responding
kubectl get --raw /healthz && echo "API server healthy"

# 3. Verify all nodes are Ready
kubectl get nodes -o wide
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'

# 4. Verify controller-manager and scheduler are running
kubectl get pods -n kube-system -l tier=control-plane

# 5. Verify etcd cluster health
kubectl -n kube-system exec etcd-$(hostname) -- \
  etcdctl endpoint health $ETCD_OPTS

# 6. Test workload operations
kubectl run cert-test --image=busybox:1.36 --rm -it -- echo "Workloads OK"

# 7. Verify service account token creation
kubectl create serviceaccount cert-test-sa --dry-run=server -o yaml

# 8. Check for any error events
kubectl get events --all-namespaces --field-selector type=Warning \
  --sort-by=.lastTimestamp | head -20
```

</details>

### Exercise 5: SLO Definition and Error Budget

Define a complete SLO framework for a Kubernetes platform team. Include: (a) 3 SLIs with their SLO targets for a 30-day window, (b) calculate the error budget for each, (c) write the Prometheus recording rules for one SLI, and (d) describe the error budget policy (what actions to take at 50%, 75%, and 100% budget consumption).

<details><summary>Show Answer</summary>

**(a) SLIs and SLOs:**

| SLI | Measurement | SLO Target (30-day) |
|-----|-------------|---------------------|
| API Server Availability | Ratio of successful (non-5xx) /readyz responses | 99.95% |
| Pod Scheduling Latency | p99 time from pod creation to pod bound to node | < 5 seconds |
| Node Health | % of nodes in Ready condition at any point in time | 99.9% |

**(b) Error budgets (30 days = 43,200 minutes):**

```
API Server Availability:
  Budget = 100% - 99.95% = 0.05%
  Allowed downtime = 43,200 * 0.0005 = 21.6 minutes/month

Pod Scheduling Latency:
  Budget = requests where p99 > 5s
  If 10,000 pod creations/month, allow 50 to exceed 5s (0.5% budget)

Node Health:
  Budget = 100% - 99.9% = 0.1%
  Per node (10 nodes, 43,200 min): 43.2 minutes of NotReady per node
```

**(c) Prometheus recording rules for API Server Availability:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: platform-slo-recording
  namespace: monitoring
spec:
  groups:
    - name: platform-slo
      interval: 30s
      rules:
        # 5-minute error rate
        - record: platform:apiserver:error_rate:5m
          expr: |
            1 - (
              sum(rate(apiserver_request_total{code!~"5.."}[5m]))
              /
              sum(rate(apiserver_request_total[5m]))
            )

        # 1-hour error rate (for burn rate calculations)
        - record: platform:apiserver:error_rate:1h
          expr: |
            1 - (
              sum(rate(apiserver_request_total{code!~"5.."}[1h]))
              /
              sum(rate(apiserver_request_total[1h]))
            )

        # 30-day availability (rolling)
        - record: platform:apiserver:availability:30d
          expr: |
            1 - (
              sum(increase(apiserver_request_total{code=~"5.."}[30d]))
              /
              sum(increase(apiserver_request_total[30d]))
            )

        # Error budget remaining (percentage)
        - record: platform:apiserver:error_budget_remaining:30d
          expr: |
            1 - (
              (1 - platform:apiserver:availability:30d)
              /
              (1 - 0.9995)
            )
```

**(d) Error budget policy:**

```
Budget Consumption | Status  | Actions
-------------------+---------+------------------------------------------
< 50%              | Green   | Normal operations. Ship features freely.
                   |         | Weekly SLO review in team standup.
                   |         |
50-75%             | Yellow  | Increase review rigor for changes.
                   |         | Require rollback plan for every deploy.
                   |         | Investigate top error contributors.
                   |         | Daily SLO check in standup.
                   |         |
75-100%            | Orange  | Freeze non-critical changes.
                   |         | Dedicate 50% of sprint to reliability.
                   |         | Post-incident review for all new errors.
                   |         | Escalate to engineering leadership.
                   |         |
> 100%             | Red     | Full change freeze (emergencies only).
                   |         | 100% of engineering on reliability.
                   |         | Executive incident review.
                   |         | No new features until budget recovers.
                   |         | Consider architecture changes.
```

</details>

---

**Previous**: [16. Kubernetes API Programming](./16_Kubernetes_API_Programming.md) | **Next**: [18. Kubernetes for ML](./18_Kubernetes_for_ML.md)
