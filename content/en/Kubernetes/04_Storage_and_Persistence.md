# 04. Storage and Persistence

**Previous**: [Networking Fundamentals](./03_Networking_Fundamentals.md) | **Next**: [Configuration and Secrets](./05_Configuration_and_Secrets.md)

## Learning Objectives
- Understand volume types and their appropriate use cases
- Configure PersistentVolumes, PersistentVolumeClaims, and StorageClasses
- Implement dynamic provisioning and manage access modes and reclaim policies
- Work with CSI drivers and volume snapshots
- Design storage patterns for StatefulSets and ephemeral workloads

---

Containers are ephemeral by design—when a container restarts, its filesystem
is reset to the image state. Kubernetes volumes solve this by providing storage
that outlives individual containers and even pods. This lesson covers the full
storage stack from basic volumes to production-grade dynamic provisioning with
CSI drivers.

Before the volume tour, read [**Theory & Principles**](#theory--principles) — why Kubernetes splits storage into PV (the supply side) and PVC (the demand side), how the binding algorithm matches them, what dynamic provisioning via StorageClass actually triggers under the hood, and why the CSI interface let storage vendors stop modifying Kubernetes core.

## Table of Contents
0. [Theory & Principles](#theory--principles)
1. [Volumes and Volume Types](#1-volumes-and-volume-types)
2. [PersistentVolumes (PV)](#2-persistentvolumes-pv)
3. [PersistentVolumeClaims (PVC)](#3-persistentvolumeclaims-pvc)
4. [StorageClasses and Dynamic Provisioning](#4-storageclasses-and-dynamic-provisioning)
5. [Access Modes](#5-access-modes)
6. [Reclaim Policies](#6-reclaim-policies)
7. [CSI (Container Storage Interface)](#7-csi-container-storage-interface)
8. [Volume Snapshots](#8-volume-snapshots)
9. [Ephemeral Volumes](#9-ephemeral-volumes)
10. [StatefulSet Storage Patterns](#10-statefulset-storage-patterns)
11. [Exercises](#exercises)

---

## Theory & Principles

Storage is the part of Kubernetes where two opposing concerns meet: containers want to be ephemeral and replaceable, while databases, queues, and caches insist on data outliving the process that wrote it. Kubernetes resolves this not by patching ephemerality away but by **separating storage from workloads**: a workload declares "I need this much storage with these properties," and a separate subsystem provides it, attaches it, mounts it, and reclaims it independently. This section explains the supply-and-demand model, the binding algorithm, dynamic provisioning, and the CSI plugin contract that makes the whole thing extensible.

### A. PV / PVC: A Supply-and-Demand Decoupling

Kubernetes models storage as a marketplace:

- **PersistentVolume (PV)** is the *supply side* — a piece of storage that exists in the cluster, with attributes (capacity, access modes, reclaim policy, storage class, backing driver). PVs are cluster-scoped (not namespaced); they describe storage available somewhere in the infrastructure.
- **PersistentVolumeClaim (PVC)** is the *demand side* — a namespaced request for storage with attributes (requested capacity, required access mode, optional storage class). PVCs are written by workload owners.
- The **binding controller** matches PVCs to PVs.

The decoupling matters because storage admins and app developers think on different timescales. The cluster operator pre-provisions a pool of PVs (or sets up dynamic provisioning, §C), and developers consume PVCs without knowing whether the underlying disk is EBS, Ceph RBD, NFS, or local SSD. The same PVC YAML works on every cloud and on-prem.

A Pod references the **PVC by name**, never a PV directly. That indirection is what allows the same workload manifest to deploy across environments.

### B. The Binding Algorithm

When a PVC is created, the controller looks for a PV that satisfies all of:

1. **Capacity ≥ requested.** A 5Gi PVC binds to a 10Gi PV (the difference is wasted; PVs are not subdivided).
2. **AccessMode is in the PV's supported set.** RWO (ReadWriteOnce: one node), ROX (ReadOnlyMany), RWX (ReadWriteMany), RWOP (ReadWriteOncePod, single pod). Block storage is RWO; networked filesystems can be RWX.
3. **StorageClass matches** (including the "" / nil case for static provisioning).
4. **Selector / volumeName** match if specified by the PVC.

If multiple PVs match, the controller picks the smallest one that fits to minimize waste. If none match and a StorageClass is set with a provisioner, dynamic provisioning kicks in (§C). Otherwise the PVC stays Pending.

Once bound, PV and PVC are exclusive — the binding is 1:1 and stored in both objects' `spec.claimRef` / `spec.volumeName`. Even if the PVC is deleted and recreated with the same name, it gets a new PV (or stays Pending if the old one is still bound, depending on reclaim policy).

The Pod scheduler is involved too: with `volumeBindingMode: WaitForFirstConsumer`, binding is deferred until a Pod actually uses the PVC, so the PV can be created in the same zone as the chosen node. Without it, you can end up with a PV in zone A and a Pod that the scheduler then must place in zone A — over-constrained.

### C. Dynamic Provisioning via StorageClass

Pre-provisioning PVs is operationally painful for any cluster that does not know its workload mix in advance. The **StorageClass** abstraction lets the cluster *create PVs on demand*:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

When a PVC arrives with `storageClassName: gp3`, the controller asks the named provisioner ("the EBS CSI driver") to create a real EBS volume and a corresponding PV. The PVC then binds to that auto-generated PV. This converts storage from a static inventory problem to an on-demand utility.

Multiple StorageClasses let you offer tiers (`gp3` for general workloads, `io2` for databases, `cold` for archive). One can be the default (`storageclass.kubernetes.io/is-default-class: "true"`) so PVCs without an explicit class still work.

The reclaim policy decides what happens when the PVC is deleted: `Delete` (also delete the underlying volume — destructive!), `Retain` (keep the PV in `Released` state for manual cleanup, used for production data), `Recycle` (deprecated; basic scrub-and-reuse).

### D. CSI: The Plugin Contract That Made Storage Extensible

Pre-CSI, every storage driver (NFS, RBD, EBS, GCE PD, ...) was compiled into Kubernetes core. Adding a vendor required a Kubernetes release. **CSI (Container Storage Interface)** broke this by defining a standard gRPC interface that any vendor can implement out-of-tree.

A CSI driver is two pieces:

- **Controller plugin** (cluster-wide): handles `CreateVolume` / `DeleteVolume`, snapshot operations, and (on cloud providers) attach/detach to nodes. Runs as a Deployment in the kube-system namespace.
- **Node plugin** (per-node DaemonSet): handles `NodeStageVolume` / `NodePublishVolume` — formats and mounts the device into the Pod's filesystem namespace.

The kubelet does not know about EBS or RBD; it just calls CSI gRPC methods. This isolation is why modern storage vendors ship a single Helm chart and you get full Kubernetes integration without recompiling anything.

The lifecycle of a Pod using a dynamically provisioned PVC:

1. PVC created → external-provisioner sidecar calls CSI `CreateVolume` → cloud creates disk, provisioner creates PV, binding controller binds PVC↔PV.
2. Pod scheduled to a node → external-attacher calls CSI `ControllerPublishVolume` → cloud attaches the disk to that node.
3. kubelet on the node calls CSI `NodeStageVolume` (format if needed, mount to staging dir) and `NodePublishVolume` (bind-mount into Pod's filesystem).
4. Pod runs.
5. Pod deleted → reverse: `NodeUnpublish`, `NodeUnstage`, `ControllerUnpublish`. PVC deletion (with `Delete` reclaim) triggers `DeleteVolume`.

Volume snapshots and clones are CSI optional capabilities (`VolumeSnapshot`, `VolumeSnapshotClass`) following the same pattern: a request object, a CSI call, and a controller that bridges the two.

### From Theory to the YAML Below

The lesson now walks through these abstractions:

- **Section 1 (Volumes and Volume Types)** covers the lower-level Pod-scoped volumes that aren't PVs (emptyDir, configMap, projected, etc.) — useful before introducing the persistence model.
- **Sections 2–3 (PV, PVC)** are §A: the supply and demand objects. Read both YAMLs side-by-side to see how `accessModes` and `storage` connect them.
- **Section 4 (StorageClasses, Dynamic Provisioning)** is §C — see how `provisioner` and `parameters` map to a real cloud driver.
- **Section 5 (Access Modes)** unpacks the binding-algorithm constraint from §B with concrete RWO/ROX/RWX examples.
- **Section 6 (Reclaim Policies)** is the destructive vs. preserving choice from §C — pick wrong and you can lose data on PVC delete.
- **Section 7 (CSI)** is §D — see the architecture diagram of controller + node plugins.
- **Section 8 (Volume Snapshots)** uses the CSI snapshot capability for backup/restore patterns.
- **Section 9 (Ephemeral Volumes)** is a related design — generic ephemeral volumes use the PVC machinery for short-lived storage.
- **Section 10 (StatefulSet Storage Patterns)** ties storage back to the workloads that need per-replica identity (lesson 02 §C).

Once you see PV/PVC as supply/demand and StorageClass as on-demand provisioning, every storage YAML is just specializing the four parts of §B's binding algorithm.

---

## 1. Volumes and Volume Types

A Kubernetes volume is a directory accessible to containers in a pod. The volume's
lifecycle depends on its type—some are tied to the pod, others persist independently.

### 1.1 emptyDir

Created when a pod is assigned to a node, deleted when the pod is removed.
All containers in the pod can read and write to it.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: emptydir-demo
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          while true; do
            echo "$(date): Log entry" >> /data/app.log
            sleep 5
          done
      volumeMounts:
        - name: shared-data
          mountPath: /data

    - name: reader
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          tail -f /data/app.log
      volumeMounts:
        - name: shared-data
          mountPath: /data
          readOnly: true

  volumes:
    - name: shared-data
      emptyDir: {}
      # emptyDir:
      #   medium: Memory      # Use tmpfs (RAM-backed)
      #   sizeLimit: 256Mi    # Enforce size limit
```

Use cases:
- Scratch space for computation
- Sharing data between containers in a pod
- Cache directory

### 1.2 hostPath

Mounts a file or directory from the host node's filesystem into the pod.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-demo
spec:
  containers:
    - name: app
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: host-logs
          mountPath: /host-logs
          readOnly: true
  volumes:
    - name: host-logs
      hostPath:
        path: /var/log
        type: Directory    # Must exist; fail if not a directory
```

hostPath types:

| Type | Behavior |
|------|----------|
| `""` (empty) | No check; create if needed |
| `DirectoryOrCreate` | Create directory if missing |
| `Directory` | Must exist as a directory |
| `FileOrCreate` | Create file if missing |
| `File` | Must exist as a file |
| `Socket` | Must exist as a Unix socket |

> **Warning**: hostPath volumes are a security risk. They bypass container isolation
> and can access any file on the node. Avoid in production; use PersistentVolumes
> instead.

### 1.3 configMap and secret

Mount ConfigMap or Secret data as files:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: config-volume-demo
spec:
  containers:
    - name: app
      image: nginx:1.25
      volumeMounts:
        - name: config-vol
          mountPath: /etc/nginx/conf.d
        - name: tls-vol
          mountPath: /etc/nginx/ssl
          readOnly: true
  volumes:
    - name: config-vol
      configMap:
        name: nginx-config
        items:
          - key: default.conf
            path: default.conf     # Mount specific keys
    - name: tls-vol
      secret:
        secretName: nginx-tls
        defaultMode: 0400          # File permissions
```

### 1.4 projected

Combines multiple volume sources into a single mount:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: projected-demo
spec:
  containers:
    - name: app
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: all-in-one
          mountPath: /projected
  volumes:
    - name: all-in-one
      projected:
        sources:
          - configMap:
              name: app-config
              items:
                - key: config.yaml
                  path: config.yaml
          - secret:
              name: app-secret
              items:
                - key: api-key
                  path: credentials/api-key
          - downwardAPI:
              items:
                - path: labels
                  fieldRef:
                    fieldPath: metadata.labels
                - path: cpu-request
                  resourceFieldRef:
                    containerName: app
                    resource: requests.cpu
          - serviceAccountToken:
              path: token
              expirationSeconds: 3600
              audience: vault
```

### 1.5 downwardAPI

Exposes pod metadata as files:

```yaml
volumes:
  - name: podinfo
    downwardAPI:
      items:
        - path: "name"
          fieldRef:
            fieldPath: metadata.name
        - path: "namespace"
          fieldRef:
            fieldPath: metadata.namespace
        - path: "labels"
          fieldRef:
            fieldPath: metadata.labels
        - path: "annotations"
          fieldRef:
            fieldPath: metadata.annotations
        - path: "cpu-request"
          resourceFieldRef:
            containerName: app
            resource: requests.cpu
            divisor: "1m"    # Express in millicores
```

---

## 2. PersistentVolumes (PV)

A PersistentVolume is a cluster-wide storage resource provisioned by an
administrator or dynamically by a StorageClass. PVs have a lifecycle independent
of any pod.

### 2.1 Static Provisioning

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs-data
  labels:
    type: nfs
    environment: production
spec:
  capacity:
    storage: 100Gi
  volumeMode: Filesystem          # Or Block
  accessModes:
    - ReadWriteMany               # Multiple nodes can mount read-write
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs-slow      # Links to StorageClass (or "" for no class)
  mountOptions:
    - hard
    - nfsvers=4.1
  nfs:
    server: 192.168.1.100
    path: /exports/data
```

### 2.2 PV with Local Storage

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-local-ssd
spec:
  capacity:
    storage: 500Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Delete
  storageClassName: local-ssd
  local:
    path: /mnt/ssd/data
  nodeAffinity:                   # Required for local volumes
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values:
                - worker-node-1
```

### 2.3 PV Lifecycle Phases

```
Available → Bound → Released → (Reclaimed/Deleted)
```

| Phase | Description |
|-------|-------------|
| Available | PV is free and not yet bound to a PVC |
| Bound | PV is bound to a PVC |
| Released | PVC was deleted; PV is not yet reclaimed |
| Failed | Automatic reclamation failed |

```bash
# List PersistentVolumes
kubectl get pv

# Output:
# NAME           CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      STORAGECLASS
# pv-nfs-data    100Gi      RWX            Retain           Available   nfs-slow
# pv-local-ssd   500Gi      RWO            Delete           Bound       local-ssd

# Detailed PV information
kubectl describe pv pv-nfs-data
```

---

## 3. PersistentVolumeClaims (PVC)

A PVC is a request for storage by a user. It binds to a PV that satisfies the
request based on capacity, access mode, and storage class.

### 3.1 Basic PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-claim
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard       # Match a StorageClass
  # selector:                      # Optional: select specific PVs
  #   matchLabels:
  #     type: nfs
```

### 3.2 Using a PVC in a Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-storage
spec:
  containers:
    - name: app
      image: postgres:16
      volumeMounts:
        - name: pg-data
          mountPath: /var/lib/postgresql/data
      env:
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
  volumes:
    - name: pg-data
      persistentVolumeClaim:
        claimName: data-claim
        # readOnly: false          # Default
```

### 3.3 PV-PVC Binding

**Dynamic provisioning flow** (most common in cloud environments):

```
PVC created
    │
    ▼
StorageClass selected (by storageClassName field)
    │
    ▼
Dynamic provisioner notified (e.g., ebs.csi.aws.com)
    │
    ▼
PV created automatically (cloud API call: CreateVolume)
    │
    ▼
PVC ←── Bound ──→ PV
    │
    ▼
Pod mounts the volume
```

**Static provisioning flow** (pre-created PVs):

```
Admin creates PV  ──→  PVC created  ──→  Kubernetes binds (matching
                                          StorageClass + capacity + access modes)
```

The binding algorithm matches PVCs to PVs based on:

1. **StorageClass**: Must match exactly
2. **Access modes**: PV must support all requested modes
3. **Capacity**: PV capacity >= PVC request
4. **Selector**: If specified, PV labels must match
5. **Volume mode**: Must match (Filesystem or Block)

```bash
# Check binding status
kubectl get pvc data-claim

# Output:
# NAME         STATUS   VOLUME         CAPACITY   ACCESS MODES   STORAGECLASS
# data-claim   Bound    pv-nfs-data    100Gi      RWX            nfs-slow

# If Pending, check events for why
kubectl describe pvc data-claim | grep -A 10 Events
```

### 3.4 PVC Expansion

Expand a bound PVC (requires StorageClass with `allowVolumeExpansion: true`):

```bash
# Edit the PVC to increase size
kubectl patch pvc data-claim -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# Check expansion status
kubectl get pvc data-claim
kubectl describe pvc data-claim | grep -A 5 Conditions

# For filesystem expansion, the pod may need to be restarted
# (depends on the CSI driver's online expansion support)
```

---

## 4. StorageClasses and Dynamic Provisioning

StorageClasses enable **dynamic provisioning**—PVs are created automatically when
a PVC requests storage from a class.

### 4.1 StorageClass Definition

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: ebs.csi.aws.com     # CSI driver name
parameters:
  type: gp3
  iops: "5000"
  throughput: "250"
  encrypted: "true"
  fsType: ext4
reclaimPolicy: Delete            # Delete PV when PVC is deleted
allowVolumeExpansion: true       # Allow PVC resizing
volumeBindingMode: WaitForFirstConsumer  # Delay binding until pod is scheduled
mountOptions:
  - discard
  - noatime
```

### 4.2 Volume Binding Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| Immediate | PV is provisioned as soon as PVC is created | Network-attached storage |
| WaitForFirstConsumer | PV is provisioned when a pod using the PVC is scheduled | Local or zone-specific storage |

`WaitForFirstConsumer` is critical for topology-aware storage:

```yaml
# Without WaitForFirstConsumer, the PV might be provisioned in zone A,
# but the pod gets scheduled to zone B → stuck in Pending

apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: zone-aware
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer   # PV created in same zone as pod
allowedTopologies:
  - matchLabelExpressions:
      - key: topology.kubernetes.io/zone
        values:
          - us-east-1a
          - us-east-1b
```

### 4.3 Default StorageClass

One StorageClass can be marked as default. PVCs without a `storageClassName` use it.

```bash
# Check the default StorageClass
kubectl get storageclass

# Output:
# NAME                 PROVISIONER                RECLAIMPOLICY   VOLUMEBINDINGMODE
# standard (default)   k8s.io/minikube-hostpath   Delete          Immediate
# fast-ssd             ebs.csi.aws.com            Delete          WaitForFirstConsumer

# Set a StorageClass as default
kubectl patch storageclass fast-ssd -p \
  '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

# Remove default from old class
kubectl patch storageclass standard -p \
  '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'
```

### 4.4 Cloud Provider StorageClasses

**AWS EBS:**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: aws-gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
  kmsKeyId: "arn:aws:kms:us-east-1:123456:key/abcd-1234"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

**GCP Persistent Disk:**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gcp-ssd
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
  replication-type: regional-pd    # Regional replication
reclaimPolicy: Retain
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

**Azure Disk:**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-premium
provisioner: disk.csi.azure.com
parameters:
  skuName: Premium_LRS
  cachingMode: ReadOnly
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

---

## 5. Access Modes

Access modes define how a volume can be mounted by nodes.

| Mode | Abbreviation | Description |
|------|-------------|-------------|
| ReadWriteOnce | RWO | Mounted read-write by a single node |
| ReadOnlyMany | ROX | Mounted read-only by many nodes |
| ReadWriteMany | RWX | Mounted read-write by many nodes |
| ReadWriteOncePod | RWOP | Mounted read-write by a single pod (v1.29+) |

### 5.1 Access Mode Support by Storage Type

| Storage Type | RWO | ROX | RWX | RWOP |
|-------------|-----|-----|-----|------|
| AWS EBS | Yes | No | No | Yes |
| GCP PD | Yes | Yes | No | Yes |
| Azure Disk | Yes | No | No | Yes |
| NFS | Yes | Yes | Yes | No |
| CephFS | Yes | Yes | Yes | No |
| Local volume | Yes | No | No | Yes |

### 5.2 ReadWriteOncePod (RWOP)

Ensures exclusive access at the pod level (stricter than RWO):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exclusive-claim
spec:
  accessModes:
    - ReadWriteOncePod       # Only one pod can mount this, ever
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

RWO vs RWOP:
- **RWO**: Multiple pods on the *same node* can mount the volume
- **RWOP**: Only *one pod* in the entire cluster can mount the volume

---

## 6. Reclaim Policies

When a PVC is deleted, the reclaim policy determines what happens to the PV.

| Policy | Behavior | Use Case |
|--------|----------|----------|
| Retain | PV is kept; data is preserved; manual cleanup required | Production data |
| Delete | PV and underlying storage are deleted | Temporary/reproducible data |
| Recycle | Deprecated; performs `rm -rf /vol/*` | Legacy only |

### 6.1 Changing Reclaim Policy

```bash
# Change an existing PV's reclaim policy
kubectl patch pv pv-nfs-data -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
```

### 6.2 Recovering a Released PV

When a PVC is deleted and the PV has `Retain` policy, the PV enters `Released`
state. To rebind it:

```bash
# 1. Check the released PV
kubectl get pv pv-nfs-data
# STATUS: Released

# 2. Remove the claimRef to make it Available again
kubectl patch pv pv-nfs-data -p '{"spec":{"claimRef":null}}'

# 3. PV is now Available and can be bound to a new PVC
kubectl get pv pv-nfs-data
# STATUS: Available
```

> **Warning**: Be careful with data when rebinding. The existing data remains on
> the volume. Ensure the new consumer expects or can handle the existing data.

---

## 7. CSI (Container Storage Interface)

CSI is the standard interface between Kubernetes and external storage systems.
It replaced in-tree volume plugins with out-of-tree drivers.

CSI splits storage responsibilities between two plugin types:

```
Container Orchestrator (CO)           Node
┌──────────────────────────┐          ┌────────────────────────────┐
│  API Server / Controller  │          │  kubelet                   │
│                           │          │    │                        │
│  external-provisioner ────┼──gRPC───▶│  Node Plugin (DaemonSet)  │
│  (watches PVCs)           │          │    │                        │
│          │                │          │    ▼                        │
│          ▼                │          │  NodeStageVolume()          │
│  Controller Plugin        │          │  NodePublishVolume()        │
│  (Deployment)             │          └────────────────────────────┘
│    CreateVolume()         │
│    DeleteVolume()         │
│    ControllerPublish()    │
└──────────────────────────┘
```

### 7.1 CSI Architecture

```
┌─────────────────────────────────────────────────┐
│                Kubernetes                        │
│  ┌──────────────┐         ┌──────────────────┐  │
│  │  API Server   │         │  kubelet         │  │
│  └──────┬───────┘         └────────┬─────────┘  │
│         │                          │             │
│  ┌──────┴───────┐         ┌────────┴─────────┐  │
│  │  External     │         │  CSI Node        │  │
│  │  Provisioner  │         │  Driver Plugin   │  │
│  │  Sidecar      │         │  (per node)      │  │
│  └──────┬───────┘         └────────┬─────────┘  │
│         │                          │             │
└─────────┼──────────────────────────┼─────────────┘
          │     CSI gRPC API         │
    ┌─────┴──────────────────────────┴─────┐
    │         CSI Driver Controller         │
    │  (CreateVolume, DeleteVolume,          │
    │   ControllerPublish, Snapshot)         │
    └──────────────────┬───────────────────┘
                       │
              ┌────────┴────────┐
              │  Storage Backend │
              │  (AWS EBS, etc.) │
              └─────────────────┘
```

### 7.2 CSI Driver Components

| Component | Runs On | Purpose |
|-----------|---------|---------|
| Controller Plugin | Deployment (1-3 replicas) | Volume create/delete, snapshot |
| Node Plugin | DaemonSet (every node) | Volume mount/unmount, format |
| External Provisioner | Sidecar with controller | Watches PVCs, triggers CreateVolume |
| External Attacher | Sidecar with controller | Attaches volumes to nodes |
| External Snapshotter | Sidecar with controller | Creates/deletes snapshots |
| Node Driver Registrar | Sidecar with node plugin | Registers driver with kubelet |

### 7.3 Installing a CSI Driver (AWS EBS Example)

```bash
# Install the AWS EBS CSI driver using Helm
helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver
helm repo update

helm install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver \
  --namespace kube-system \
  --set controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="arn:aws:iam::123456:role/ebs-csi-role"

# Verify the driver is running
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-ebs-csi-driver

# Check CSIDriver object
kubectl get csidriver ebs.csi.aws.com -o yaml
```

### 7.4 CSIDriver Object

```yaml
apiVersion: storage.k8s.io/v1
kind: CSIDriver
metadata:
  name: ebs.csi.aws.com
spec:
  attachRequired: true           # Requires ControllerPublishVolume
  podInfoOnMount: false          # Don't pass pod info to NodePublishVolume
  fsGroupPolicy: File            # Apply fsGroup ownership
  volumeLifecycleModes:
    - Persistent                 # Supports PV/PVC workflow
  storageCapacity: true          # Reports storage capacity
  tokenRequests: []
  requiresRepublish: false
  seLinuxMount: false
```

### 7.5 Storage Capacity Tracking

CSI drivers can report available storage capacity per topology:

```bash
# View storage capacity (if driver supports it)
kubectl get csistoragecapacities -A

# Output:
# NAMESPACE     NAME             STORAGE CLASS   CAPACITY     NODE TOPOLOGY
# kube-system   csi-cap-abc123   fast-ssd        450Gi        node=worker-1
# kube-system   csi-cap-def456   fast-ssd        320Gi        node=worker-2
```

The scheduler uses this information to avoid scheduling pods on nodes without
sufficient storage.

---

## 8. Volume Snapshots

Volume snapshots create point-in-time copies of volumes. They require a CSI
driver that supports snapshots and the snapshot controller.

### 8.1 VolumeSnapshotClass

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: ebs-snapshot-class
driver: ebs.csi.aws.com
deletionPolicy: Delete           # Delete snapshot when VolumeSnapshot is deleted
# deletionPolicy: Retain         # Keep snapshot even after VolumeSnapshot is deleted
parameters:
  # Driver-specific parameters
  tagSpecification_1: "backup=true"
```

### 8.2 Creating a Snapshot

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot-20240115
spec:
  volumeSnapshotClassName: ebs-snapshot-class
  source:
    persistentVolumeClaimName: data-claim    # Source PVC
```

```bash
# Create the snapshot
kubectl apply -f snapshot.yaml

# Check snapshot status
kubectl get volumesnapshot data-snapshot-20240115

# Output:
# NAME                      READYTOUSE   RESTORESIZE   SNAPSHOTCLASS
# data-snapshot-20240115     true         50Gi          ebs-snapshot-class

# Detailed status
kubectl describe volumesnapshot data-snapshot-20240115
```

### 8.3 Restoring from a Snapshot

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-restored
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
  dataSource:
    name: data-snapshot-20240115
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
```

### 8.4 Cloning a PVC

Create a new PVC from an existing one (without going through a snapshot):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-clone
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
  dataSource:
    name: data-claim             # Source PVC
    kind: PersistentVolumeClaim
    # No apiGroup needed for PVC cloning
```

### 8.5 Scheduled Snapshots

Kubernetes does not have built-in snapshot scheduling. Use a CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: snapshot-scheduler
spec:
  schedule: "0 */6 * * *"        # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: snapshot-creator
          restartPolicy: OnFailure
          containers:
            - name: snapshot
              image: bitnami/kubectl:1.29
              command:
                - sh
                - -c
                - |
                  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
                  cat <<SNAP | kubectl apply -f -
                  apiVersion: snapshot.storage.k8s.io/v1
                  kind: VolumeSnapshot
                  metadata:
                    name: data-snap-${TIMESTAMP}
                    labels:
                      app: scheduled-snapshot
                  spec:
                    volumeSnapshotClassName: ebs-snapshot-class
                    source:
                      persistentVolumeClaimName: data-claim
                  SNAP
                  echo "Snapshot data-snap-${TIMESTAMP} created"
```

---

## 9. Ephemeral Volumes

Ephemeral volumes are created and deleted alongside the pod. They are useful
for temporary data that does not need to persist.

### 9.1 Generic Ephemeral Volumes

Use any StorageClass for pod-scoped storage:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ephemeral-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: scratch
          mountPath: /scratch
  volumes:
    - name: scratch
      ephemeral:
        volumeClaimTemplate:
          spec:
            accessModes: ["ReadWriteOnce"]
            storageClassName: fast-ssd
            resources:
              requests:
                storage: 10Gi
```

The system creates a PVC named `<pod-name>-scratch` (pod name + volume name).
When the pod is deleted, the PVC is automatically deleted.

### 9.2 CSI Ephemeral Volumes

Some CSI drivers support inline ephemeral volumes:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: csi-ephemeral-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: secret-store
          mountPath: /mnt/secrets
          readOnly: true
  volumes:
    - name: secret-store
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: aws-secrets
```

### 9.3 emptyDir vs Generic Ephemeral

| Feature | emptyDir | Generic Ephemeral |
|---------|----------|-------------------|
| Storage backend | Node disk or tmpfs | Any StorageClass |
| Size enforcement | sizeLimit (soft) | PVC quota (hard) |
| Performance | Node-local | Depends on backend |
| Snapshot support | No | Yes (if driver supports) |
| Metrics | Limited | Full CSI metrics |

---

## 10. StatefulSet Storage Patterns

### 10.1 VolumeClaimTemplates

StatefulSets use `volumeClaimTemplates` to create a unique PVC for each pod:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
spec:
  serviceName: es-headless
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      initContainers:
        - name: fix-permissions
          image: busybox:1.36
          command: ["sh", "-c", "chown -R 1000:1000 /usr/share/elasticsearch/data"]
          volumeMounts:
            - name: data
              mountPath: /usr/share/elasticsearch/data
      containers:
        - name: elasticsearch
          image: elasticsearch:8.12.0
          ports:
            - containerPort: 9200
              name: http
            - containerPort: 9300
              name: transport
          env:
            - name: cluster.name
              value: "k8s-cluster"
            - name: node.name
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: discovery.seed_hosts
              value: "es-headless"
            - name: cluster.initial_master_nodes
              value: "elasticsearch-0,elasticsearch-1,elasticsearch-2"
          volumeMounts:
            - name: data
              mountPath: /usr/share/elasticsearch/data
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"

  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi
```

This creates:
```
PVC: data-elasticsearch-0 → PV (100Gi, fast-ssd)
PVC: data-elasticsearch-1 → PV (100Gi, fast-ssd)
PVC: data-elasticsearch-2 → PV (100Gi, fast-ssd)
```

### 10.2 PVC Retention Policy (v1.27+)

Control what happens to PVCs when the StatefulSet is scaled down or deleted:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
spec:
  persistentVolumeClaimRetentionPolicy:
    whenDeleted: Retain      # Keep PVCs when StatefulSet is deleted
    whenScaled: Delete       # Delete PVCs when scaling down
    # Options: Retain (default) or Delete
```

### 10.3 Multiple Volumes per Pod

```yaml
volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: wal               # Write-ahead log on separate volume
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: ultra-ssd   # Even faster storage for WAL
      resources:
        requests:
          storage: 20Gi
```

### 10.4 Data Migration Pattern

When upgrading storage, you cannot modify `volumeClaimTemplates`. Use this pattern:

```bash
# 1. Scale down the StatefulSet
kubectl scale statefulset elasticsearch --replicas=0

# 2. Create snapshots of all PVCs
for i in 0 1 2; do
  cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-migration-$i
spec:
  source:
    persistentVolumeClaimName: data-elasticsearch-$i
EOF
done

# 3. Create new PVCs from snapshots with updated storage class
for i in 0 1 2; do
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-elasticsearch-new-$i
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: new-fast-ssd
  resources:
    requests:
      storage: 200Gi
  dataSource:
    name: data-migration-$i
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF
done

# 4. Delete old PVCs and rename new ones
for i in 0 1 2; do
  kubectl delete pvc data-elasticsearch-$i
  # Note: kubectl cannot rename PVCs; you may need to recreate the StatefulSet
done

# 5. Scale the StatefulSet back up
kubectl scale statefulset elasticsearch --replicas=3
```

---

## Exercises

### Exercise 1: PV and PVC Binding

Create a PersistentVolume with 5Gi capacity and a PersistentVolumeClaim that
requests 3Gi. Verify they bind correctly, then mount the PVC in a pod and
write data to it.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/pv-pvc-exercise.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: exercise-pv
spec:
  capacity:
    storage: 5Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Delete
  storageClassName: manual
  hostPath:
    path: /tmp/exercise-pv-data
    type: DirectoryOrCreate
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exercise-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 3Gi
  storageClassName: manual
---
apiVersion: v1
kind: Pod
metadata:
  name: storage-writer
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Hello from persistent storage!" > /data/greeting.txt
          echo "Written at $(date)" >> /data/greeting.txt
          cat /data/greeting.txt
          sleep 3600
      volumeMounts:
        - name: persistent-data
          mountPath: /data
  volumes:
    - name: persistent-data
      persistentVolumeClaim:
        claimName: exercise-pvc
```

```bash
kubectl apply -f /tmp/pv-pvc-exercise.yaml

# Verify PV and PVC are bound
kubectl get pv exercise-pv
# STATUS: Bound

kubectl get pvc exercise-pvc
# STATUS: Bound, VOLUME: exercise-pv

# Verify data was written
kubectl wait --for=condition=Ready pod/storage-writer --timeout=60s
kubectl exec storage-writer -- cat /data/greeting.txt

# Delete the pod, create a new one with the same PVC — data persists
kubectl delete pod storage-writer
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: storage-reader
spec:
  containers:
    - name: reader
      image: busybox:1.36
      command: ["sh", "-c", "cat /data/greeting.txt && sleep 3600"]
      volumeMounts:
        - name: persistent-data
          mountPath: /data
  volumes:
    - name: persistent-data
      persistentVolumeClaim:
        claimName: exercise-pvc
EOF
kubectl wait --for=condition=Ready pod/storage-reader --timeout=60s
kubectl exec storage-reader -- cat /data/greeting.txt
# Should show the data written by the first pod

# Clean up
kubectl delete pod storage-reader
kubectl delete pvc exercise-pvc
kubectl delete pv exercise-pv
```

</details>

### Exercise 2: Dynamic Provisioning

Create a StorageClass for minikube's hostpath provisioner. Then create a PVC
that uses dynamic provisioning. Verify a PV is automatically created.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/dynamic-provision.yaml
# On minikube, the default StorageClass already supports dynamic provisioning
# Let's create a custom one
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: custom-hostpath
provisioner: k8s.io/minikube-hostpath
reclaimPolicy: Delete
volumeBindingMode: Immediate
allowVolumeExpansion: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamic-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  storageClassName: custom-hostpath
```

```bash
kubectl apply -f /tmp/dynamic-provision.yaml

# Verify dynamic provisioning
kubectl get pvc dynamic-pvc
# STATUS: Bound (should be almost immediate)

# Check the auto-created PV
kubectl get pv
# A new PV with "pvc-<uuid>" name should appear, bound to dynamic-pvc

PV_NAME=$(kubectl get pvc dynamic-pvc -o jsonpath='{.spec.volumeName}')
kubectl describe pv $PV_NAME
# Source.Type should be HostPath
# StorageClass should be custom-hostpath

# Test with a pod
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: dynamic-test
spec:
  containers:
    - name: test
      image: busybox:1.36
      command: ["sh", "-c", "echo 'Dynamic provisioning works!' > /data/test.txt && cat /data/test.txt && sleep 3600"]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: dynamic-pvc
EOF
kubectl wait --for=condition=Ready pod/dynamic-test --timeout=60s
kubectl exec dynamic-test -- cat /data/test.txt

# Clean up
kubectl delete pod dynamic-test
kubectl delete pvc dynamic-pvc
kubectl delete storageclass custom-hostpath
```

</details>

### Exercise 3: Volume Snapshots

Create a PVC, write data to it, take a volume snapshot, then restore the snapshot
to a new PVC and verify the data.

<details>
<summary>Show Answer</summary>

```yaml
# Note: This exercise requires a CSI driver with snapshot support.
# On minikube, enable the volumesnapshots addon:
# minikube addons enable volumesnapshots
# minikube addons enable csi-hostpath-driver

# Save as /tmp/snapshot-exercise.yaml
# Step 1: Create source PVC and write data
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: source-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: csi-hostpath-sc    # minikube CSI hostpath
---
apiVersion: v1
kind: Pod
metadata:
  name: data-writer
spec:
  containers:
    - name: writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Original data written at $(date)" > /data/important.txt
          echo "This data will survive a snapshot restore" >> /data/important.txt
          cat /data/important.txt
          sleep 3600
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: source-pvc
```

```bash
# Enable required addons on minikube
minikube addons enable volumesnapshots
minikube addons enable csi-hostpath-driver

# Create source PVC and write data
kubectl apply -f /tmp/snapshot-exercise.yaml
kubectl wait --for=condition=Ready pod/data-writer --timeout=120s
kubectl exec data-writer -- cat /data/important.txt

# Create the snapshot
cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot
spec:
  volumeSnapshotClassName: csi-hostpath-snapclass
  source:
    persistentVolumeClaimName: source-pvc
EOF

# Wait for snapshot to be ready
kubectl get volumesnapshot data-snapshot -w
# READYTOUSE should become true

# Restore from snapshot
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: csi-hostpath-sc
  dataSource:
    name: data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
---
apiVersion: v1
kind: Pod
metadata:
  name: data-reader
spec:
  containers:
    - name: reader
      image: busybox:1.36
      command: ["sh", "-c", "cat /data/important.txt && sleep 3600"]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: restored-pvc
EOF

kubectl wait --for=condition=Ready pod/data-reader --timeout=120s
kubectl exec data-reader -- cat /data/important.txt
# Should show the original data

# Clean up
kubectl delete pod data-writer data-reader
kubectl delete pvc source-pvc restored-pvc
kubectl delete volumesnapshot data-snapshot
```

</details>

### Exercise 4: StatefulSet with Storage

Create a StatefulSet with 3 replicas, each with its own 1Gi PVC. Write unique
data to each pod's volume. Scale down to 1 replica, then back to 3, and verify
the data persists.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/sts-storage.yaml
apiVersion: v1
kind: Service
metadata:
  name: sts-headless
spec:
  clusterIP: None
  selector:
    app: sts-storage-demo
  ports:
    - port: 80
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: sts-storage-demo
spec:
  serviceName: sts-headless
  replicas: 3
  selector:
    matchLabels:
      app: sts-storage-demo
  template:
    metadata:
      labels:
        app: sts-storage-demo
    spec:
      containers:
        - name: app
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              # Write pod identity to storage if file doesn't exist
              if [ ! -f /data/identity.txt ]; then
                echo "Pod $(hostname) created at $(date)" > /data/identity.txt
              fi
              echo "=== Stored Identity ==="
              cat /data/identity.txt
              sleep 3600
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

```bash
kubectl apply -f /tmp/sts-storage.yaml
kubectl rollout status statefulset/sts-storage-demo

# Verify each pod has its own PVC
kubectl get pvc -l app=sts-storage-demo
# data-sts-storage-demo-0   Bound   1Gi
# data-sts-storage-demo-1   Bound   1Gi
# data-sts-storage-demo-2   Bound   1Gi

# Read each pod's unique data
for i in 0 1 2; do
  echo "=== Pod $i ==="
  kubectl exec sts-storage-demo-$i -- cat /data/identity.txt
done

# Scale down to 1
kubectl scale statefulset sts-storage-demo --replicas=1

# PVCs are NOT deleted (retained by default)
kubectl get pvc -l app=sts-storage-demo
# All 3 PVCs still exist

# Scale back to 3
kubectl scale statefulset sts-storage-demo --replicas=3
kubectl rollout status statefulset/sts-storage-demo

# Verify data persists
for i in 0 1 2; do
  echo "=== Pod $i ==="
  kubectl exec sts-storage-demo-$i -- cat /data/identity.txt
done
# Each pod should show its original creation time

# Clean up
kubectl delete statefulset sts-storage-demo
kubectl delete svc sts-headless
kubectl delete pvc -l app=sts-storage-demo
```

</details>

### Exercise 5: Ephemeral Volumes

Create a pod that uses a generic ephemeral volume for scratch space. Write data
to it, delete the pod, and verify the volume is cleaned up.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/ephemeral-exercise.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ephemeral-worker
spec:
  containers:
    - name: processor
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Processing data..."
          # Write temporary data
          for i in $(seq 1 100); do
            echo "Record $i: $(date)" >> /scratch/output.csv
          done
          echo "Wrote $(wc -l < /scratch/output.csv) records to scratch volume"
          ls -la /scratch/
          sleep 3600
      volumeMounts:
        - name: scratch-space
          mountPath: /scratch
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
  volumes:
    - name: scratch-space
      ephemeral:
        volumeClaimTemplate:
          spec:
            accessModes: ["ReadWriteOnce"]
            resources:
              requests:
                storage: 1Gi
```

```bash
kubectl apply -f /tmp/ephemeral-exercise.yaml
kubectl wait --for=condition=Ready pod/ephemeral-worker --timeout=120s

# Check the auto-created PVC
kubectl get pvc
# NAME                              STATUS   VOLUME          CAPACITY
# ephemeral-worker-scratch-space    Bound    pvc-xxxxxxxx    1Gi

# Verify data was written
kubectl exec ephemeral-worker -- cat /scratch/output.csv | head -5

# Note the PVC name
PVC_NAME=$(kubectl get pvc -o jsonpath='{.items[?(@.metadata.name=="ephemeral-worker-scratch-space")].metadata.name}')
echo "Ephemeral PVC: $PVC_NAME"

# Delete the pod
kubectl delete pod ephemeral-worker

# Verify the PVC is automatically deleted
kubectl get pvc
# ephemeral-worker-scratch-space should be gone (owned by the pod)

# If it still exists briefly, wait a moment
sleep 5
kubectl get pvc
# Should be gone

echo "Ephemeral volume was automatically cleaned up"
```

</details>

---

**Previous**: [Networking Fundamentals](./03_Networking_Fundamentals.md) | **Next**: [Configuration and Secrets](./05_Configuration_and_Secrets.md)
