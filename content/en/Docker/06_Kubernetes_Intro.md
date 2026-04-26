# Kubernetes Introduction

**Previous**: [Docker Practical Examples](./05_Practical_Examples.md) | **Next**: [Kubernetes Security](./07_Kubernetes_Security.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what Kubernetes is and why container orchestration is needed at scale
2. Describe the Kubernetes cluster architecture including Control Plane and Node components
3. Define and create Pods, Deployments, and Services using YAML manifests
4. Set up a local Kubernetes environment with minikube and kubectl
5. Apply basic kubectl commands to create, inspect, scale, and delete resources
6. Implement rolling updates and rollbacks for zero-downtime deployments
7. Use ConfigMaps and Secrets to manage application configuration and sensitive data
8. Organize resources with Namespaces for logical isolation

---

Before the kubectl walkthrough, read [**Theory & Principles**](#theory--principles) — why the Pod (not the container) is the scheduling unit, the control plane vs data plane split, and the declarative reconciliation loop that makes the whole system self-healing.

Docker excels at running containers on a single machine, but production systems typically span many servers and require automated scheduling, self-healing, load balancing, and rolling updates. Kubernetes is the industry-standard platform that solves all of these challenges. Learning Kubernetes fundamentals is the natural next step after mastering Docker, opening the door to scalable, resilient infrastructure that powers the majority of modern cloud-native applications.

> **Analogy -- Airport Control Tower:** Think of Kubernetes as an airport control tower. Individual Docker containers are like aircraft -- each one carries its own payload and can operate independently. But when you have hundreds of flights, you need a control tower (Kubernetes) to decide which runway (node) each plane lands on, reroute traffic when a runway is closed (self-healing), add more gates during peak hours (auto-scaling), and ensure smooth transitions between shifts (rolling updates).

---

## Theory & Principles

Kubernetes is not "Docker but distributed." It is a **declarative state-reconciliation system** that happens to use containers as its execution primitive. Three ideas explain almost everything in the YAML you will write: the **Pod** as the scheduling unit, the **control plane / data plane** split, and the **reconciliation loop** that turns desired state into actual state. Once those click, every resource type (Deployment, Service, Job, StatefulSet, DaemonSet, ...) is just a different desired-state schema riding on the same machinery.

### A. Why the Pod, Not the Container, Is the Atom

A Pod is one or more containers that **share fate**: same node, same network namespace (same IP, same localhost), same IPC, optional shared volumes. They are scheduled together, started together, killed together.

The decision to make Pods (not containers) the atom is deliberate. A bare container is too small a unit for some real patterns:

- **Sidecar.** Main app container plus a log shipper, both reading/writing the same volume. Restarting the log shipper without restarting the app: useful. Scheduling them on different nodes: would defeat the purpose.
- **Init container.** Runs before the main container to set up state (run DB migrations, wait for a dependency, copy config). Sequenced startup, shared volume.
- **Adapter.** A second container that translates the main app's metrics format to whatever the cluster's monitoring expects, hiding the protocol behind localhost.

For the common case — one container per Pod — the Pod adds essentially zero overhead and is just "a container with metadata." For the multi-container case, it is exactly the abstraction you would otherwise have to invent. So K8s standardizes on Pod and accepts the redundancy in the simple case.

A consequence: **Pods are ephemeral and have ephemeral IPs.** They are not pets. The IP changes on restart. Anything that needs a stable network identity goes through a Service (which we will see in §C).

### B. Control Plane vs Data Plane

Every K8s cluster has two kinds of machines:

**Control plane nodes** run the brains:

- `kube-apiserver` — the only component anyone else talks to. RESTful HTTP API over the resource graph (Pods, Services, ConfigMaps, ...). Stateless; persists everything in etcd.
- `etcd` — the consistent key-value store. The source of truth. Lose etcd, lose the cluster (this is why production clusters back it up).
- `kube-scheduler` — decides which node each unscheduled Pod runs on (see §D).
- `kube-controller-manager` — runs the built-in controllers (Deployment controller, ReplicaSet controller, Node controller, ...). Each is a reconciliation loop (§E).
- `cloud-controller-manager` — same idea but for cloud-provider integrations (provisioning load balancers, persistent disks).

**Data plane nodes** (also called "worker nodes") run the workloads:

- `kubelet` — the node agent. Pulls assigned Pods from the API server and tells the local container runtime to run them. Reports node and Pod status back upstream.
- `kube-proxy` — programs iptables/IPVS rules for Service load balancing on this node.
- `container runtime` — containerd, CRI-O, or another CRI-compliant runtime. The thing that actually calls runc to spawn containers.

The split matters because **control plane outages do not stop running workloads.** If the API server crashes, your Pods keep serving traffic. Only the ability to *change* the cluster (scale, deploy, recover from failure) is impaired. This is by design — control plane HA is desirable but not on the request hot path.

Almost every interaction (kubectl, controllers, kubelets) ultimately translates to "talk HTTP to the API server, which talks to etcd." Master that mental model and the architecture stops being intimidating.

### C. The Resource Model: Everything Is YAML in etcd

Every Kubernetes object — Pod, Deployment, Service, Secret, custom resource — has the same envelope:

```yaml
apiVersion: v1               # which API group/version
kind: Pod                    # which resource type
metadata:                    # name, namespace, labels, annotations
  name: my-pod
  namespace: default
spec:                        # desired state (what you want)
  containers:
    - name: app
      image: nginx:1.27
status:                      # actual state (what the controller observes)
  phase: Running
  podIP: 10.0.1.5
```

You write `spec`. Controllers write `status`. The split lets you reason about intent (spec) separately from reality (status), and it makes diffs trivial: the controller's job is to make `status` match `spec`.

`metadata.labels` and `metadata.annotations` are how everything else finds objects. A Deployment selects its Pods by `spec.selector.matchLabels: {app: web}`; the matched Pods carry `metadata.labels: {app: web}`. A Service routes to Pods the same way. Labels are the implicit foreign-key system tying the entire object graph together.

`apiVersion: v1` (core), `apps/v1` (workloads), `networking.k8s.io/v1` (networking), and so on — Kubernetes versions API groups independently so newer features can land without breaking older clients. Custom Resource Definitions (CRDs, covered in the Advanced lesson) let you add your own `apiVersion`s and have controllers reconcile them with the same machinery.

### D. The Reconciliation Loop: Declarative State Made Real

This is the heart of Kubernetes. Every controller runs the same algorithm:

```
loop forever:
    desired = read current spec from API server
    actual  = read current status from API server (or world)
    diff    = desired - actual
    if diff is non-empty:
        take an action that reduces diff
        update status
```

Concretely, the **Deployment controller** does this for `replicas: 3`:

1. Watch the Deployment objects in etcd.
2. For each Deployment, ensure a matching ReplicaSet exists.
3. The ReplicaSet controller watches ReplicaSets and ensures the number of Pods matches `replicas`.
4. If a Pod dies, the ReplicaSet sees `actual = 2 < 3 = desired`, so it creates a new Pod object.
5. The Pod is unscheduled (no `nodeName`), so the **scheduler** picks a node, sets `spec.nodeName`, and writes back.
6. The kubelet on that node sees a Pod assigned to it, asks the container runtime to start it, reports status back.

Every step is idempotent. Every step writes through the API server (which writes to etcd). If any component crashes mid-step, the next iteration of its loop picks up where things were left. There is no global plan or transactional commit; the system converges by repeated local nudges.

This is also why Kubernetes is **self-healing**: node dies → kubelet stops reporting → node controller marks node `NotReady` → Pods on it are evicted → ReplicaSet sees `actual < desired` → new Pods created on healthy nodes → scheduler binds them → kubelet starts them. No human intervention; the same loop that handled the original deployment handles the failure.

### E. The Scheduler: A Two-Phase Filter and Score

When a Pod is created without `spec.nodeName`, the scheduler picks one. The algorithm is two phases:

1. **Filter (predicates).** From all nodes, eliminate any that cannot run this Pod: insufficient CPU/memory, missing required GPUs, doesn't match `nodeSelector`, taints not tolerated, host port conflict, PVC zone mismatch, ...
2. **Score (priorities).** For each surviving node, compute a score (typically 0–100). Default scoring includes resource balance (don't pile every Pod on the same node), image locality bonus (the node already has the image cached), inter-Pod affinity/anti-affinity, ...

The highest-scoring node wins (ties are broken randomly). The scheduler then writes `spec.nodeName` back to the API server and the kubelet picks up from there. This entire decision is *advisory* in the sense that nothing prevents you from setting `spec.nodeName` yourself; the scheduler just runs when you don't.

For the Intro lesson, the takeaway is that scheduling is a configurable, pluggable decision based on declared resource requests and constraints. Get the resource requests on your Pods right and most "the scheduler picked a bad node" problems go away on their own.

### F. Services: Stable Names for Ephemeral Pods

Pods come and go and have ephemeral IPs. A **Service** is a stable virtual IP (cluster IP) that load-balances to Pods matching a label selector. Three common types:

- **ClusterIP** (default) — virtual IP reachable only inside the cluster. The standard way for service A to call service B.
- **NodePort** — exposes the Service on a fixed port on every node's IP. Useful for development, rarely the right choice for production.
- **LoadBalancer** — asks the cloud provider for an external load balancer pointing at NodePorts. Standard for "internet-facing single service" on cloud K8s.

Under the hood, kube-proxy on each node programs iptables (or IPVS) rules: `if dest_ip == ClusterIP and dest_port == servicePort, DNAT to one of these podIPs:targetPort, chosen randomly per connection`. There is no separate L4 load balancer process; the kernel does it.

DNS is layered on top: the cluster runs a DNS service (CoreDNS) that returns the ClusterIP for service names like `my-svc.my-namespace.svc.cluster.local`. So application code uses hostnames, kube-proxy translates the resulting ClusterIP, and the kernel forwards to one of the matching Pods.

### From Theory to the kubectl Below

- `kubectl apply -f deployment.yaml` — sends YAML to the API server, which writes it to etcd. The Deployment controller's loop sees a new spec, creates a ReplicaSet, which creates Pods, which the scheduler binds to nodes, which kubelets start.
- `kubectl get pods` — reads Pod objects from the API server. Shows `status.phase`, which the kubelet wrote.
- `kubectl scale deployment web --replicas=5` — patches `spec.replicas`. The reconciliation loop notices and creates or destroys Pods to match.
- `kubectl rollout restart deployment web` — bumps a label on the Pod template, which the controller sees as a "new spec." It rolls out new Pods and tears down old ones gracefully.
- `kubectl expose deployment web --port=80` — creates a Service object. kube-proxy on every node programs iptables rules within seconds.
- `kubectl exec` and `kubectl port-forward` — open a tunnel through the API server to the kubelet, which forwards into the container.
- `kubectl describe pod` — pulls events and current status; events are the audit log of "what happened to this Pod" written by various controllers.

The remaining sections walk these primitives in YAML form. Whenever a `kubectl` command surprises you, ask: which controller is reconciling, what spec did it see, what status did it write?

---

## 1. What is Kubernetes?

Kubernetes (K8s) is a **container orchestration platform**. It automates deployment, scaling, and management of containerized applications.

### Docker vs Kubernetes

| Docker | Kubernetes |
|--------|------------|
| Runs containers | Manages/orchestrates containers |
| Single host | Cluster (multiple servers) |
| Manual scaling | Auto-scaling |
| Simple deployment | Rolling updates, rollbacks |

### Why is Kubernetes needed?

**Problem scenario:**
```
When you have 100 containers...
- Which server should they be deployed to?
- Who restarts containers when they die?
- How to scale when traffic increases?
- Downtime during new version deployment?
```

**Kubernetes solution:**
```
- Auto-scheduling: Deploy to optimal nodes
- Self-healing: Automatic recovery on failure
- Auto-scaling: Scale up/down based on load
- Rolling updates: Zero-downtime deployment
```

---

## 2. Kubernetes Architecture

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

### Key Components

| Component | Role |
|-----------|------|
| **API Server** | Central gateway handling all requests |
| **Scheduler** | Decides which Node to place Pods on |
| **Controller Manager** | Maintains desired state (replication, deployment) |
| **etcd** | Cluster state storage |
| **kubelet** | Manages container execution on each Node |
| **kube-proxy** | Network proxy, service load balancing |

---

## 3. Core Concepts

### Pod

- **Smallest deployment unit** in Kubernetes
- Contains one or more containers
- Containers in same Pod share network/storage

```yaml
# pod.yaml — rarely created directly; use Deployments instead for self-healing and scaling
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: nginx
      image: nginx:alpine       # Alpine: ~5 MB base — smaller attack surface
      ports:
        - containerPort: 80     # Informational; actual exposure requires a Service
```

### Deployment

- **Declarative deployment management** of Pods
- Manages replica count (ReplicaSet)
- Supports rolling updates and rollbacks

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3                    # Maintain 3 Pods — K8s auto-replaces any that crash or get evicted
  selector:
    matchLabels:
      app: my-app               # Must match template labels — this is how the Deployment finds its Pods
  template:                      # Pod template
    metadata:
      labels:
        app: my-app             # Labels connect Deployments → Pods → Services (the glue of K8s)
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
```

### Service

- **Network access point** for Pods
- Load balancing
- Provides consistent access even when Pods change

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app                  # Route traffic to Pods with this label — decouples routing from Pod IPs
  ports:
    - port: 80                   # Port other services use to reach this Service
      targetPort: 80             # Port the container actually listens on
  type: ClusterIP                # Internal only (default) — use NodePort or LoadBalancer for external access
```

### Service Types

| Type | Description |
|------|-------------|
| `ClusterIP` | Accessible only within cluster (default) |
| `NodePort` | External access via Node ports |
| `LoadBalancer` | Connect to cloud load balancer |

---

## 4. Local Environment Setup

### minikube Installation

Tool for running Kubernetes locally.

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

### minikube Start

```bash
# Start cluster
minikube start

# Check status
minikube status

# Open dashboard
minikube dashboard

# Stop cluster
minikube stop

# Delete cluster
minikube delete
```

### kubectl Installation

CLI tool for communicating with Kubernetes cluster.

**macOS:**
```bash
brew install kubectl
```

**Windows:**
```bash
choco install kubernetes-cli
```

**Verify:**
```bash
kubectl version --client
```

---

## 5. kubectl Basic Commands

### View Resources

```bash
# View all Pods
kubectl get pods

# View all resources
kubectl get all

# Detailed information
kubectl get pods -o wide

# Output in YAML format
kubectl get pod my-pod -o yaml

# Specify namespace
kubectl get pods -n kube-system
```

### Create/Delete Resources

```bash
# Create from YAML file
kubectl apply -f deployment.yaml

# Delete
kubectl delete -f deployment.yaml

# Delete by name
kubectl delete pod my-pod
kubectl delete deployment my-deployment
```

### Detailed Information

```bash
# Resource details
kubectl describe pod my-pod
kubectl describe deployment my-deployment

# View logs
kubectl logs my-pod
kubectl logs -f my-pod              # Real-time

# Access container
kubectl exec -it my-pod -- /bin/sh
```

### Scaling

```bash
# Change replica count
kubectl scale deployment my-deployment --replicas=5
```

---

## 6. Practice Examples

### Example 1: First Pod Execution

```bash
# 1. Run Pod directly
kubectl run nginx-pod --image=nginx:alpine

# 2. Verify
kubectl get pods

# 3. Detailed information
kubectl describe pod nginx-pod

# 4. Check logs
kubectl logs nginx-pod

# 5. Delete
kubectl delete pod nginx-pod
```

### Example 2: Deploy App with Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-app
spec:
  replicas: 3                    # 3 replicas — K8s distributes them across nodes for high availability
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello               # Labels tie Deployment → ReplicaSet → Pods → Service together
    spec:
      containers:
        - name: hello
          image: nginxdemos/hello
          ports:
            - containerPort: 80
```

```bash
# 1. Create Deployment
kubectl apply -f deployment.yaml

# 2. Verify
kubectl get deployments
kubectl get pods

# 3. Delete one Pod (verify auto-recovery)
kubectl delete pod <pod-name>
kubectl get pods  # New Pod created

# 4. Scale up
kubectl scale deployment hello-app --replicas=5
kubectl get pods
```

### Example 3: Expose with Service

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hello-service
spec:
  selector:
    app: hello                   # Matches Deployment's Pod labels — Service auto-discovers matching Pods
  ports:
    - port: 80
      targetPort: 80
  type: NodePort                 # Allocates a high port (30000-32767) on every node for external access
```

```bash
# 1. Create Service
kubectl apply -f service.yaml

# 2. Verify
kubectl get services

# 3. Access on minikube
minikube service hello-service

# Or port forwarding
kubectl port-forward service/hello-service 8080:80
# Access at http://localhost:8080
```

### Example 4: Full Application (Node.js + MongoDB)

**app-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: node-app
spec:
  replicas: 2                     # 2 replicas for basic high availability
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
              # K8s DNS resolves 'mongo-service' to the MongoDB Service's ClusterIP
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
    - port: 80                    # External-facing port
      targetPort: 3000            # Container's actual listening port — Service bridges the difference
  type: NodePort
```

**mongo-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongo
spec:
  replicas: 1                     # Single replica — databases typically use StatefulSets for production
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
              mountPath: /data/db        # MongoDB's default data directory
      volumes:
        - name: mongo-storage
          emptyDir: {}                   # emptyDir: data lost when Pod is deleted — use PersistentVolume for production
---
apiVersion: v1
kind: Service
metadata:
  name: mongo-service              # Other Pods reach MongoDB via this DNS name
spec:
  selector:
    app: mongo
  ports:
    - port: 27017
      targetPort: 27017
  # type defaults to ClusterIP — MongoDB should not be exposed outside the cluster
```

```bash
# 1. Deploy MongoDB
kubectl apply -f mongo-deployment.yaml

# 2. Deploy Node.js app
kubectl apply -f app-deployment.yaml

# 3. Verify
kubectl get all

# 4. Access
minikube service node-service
```

---

## 7. Rolling Updates

### Apply Update

```bash
# Update image
kubectl set image deployment/hello-app hello=nginxdemos/hello:latest

# Or modify YAML then apply
kubectl apply -f deployment.yaml
```

### Check Update Status

```bash
# Rollout status
kubectl rollout status deployment/hello-app

# History
kubectl rollout history deployment/hello-app
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/hello-app

# Rollback to specific version
kubectl rollout undo deployment/hello-app --to-revision=2
```

---

## 8. ConfigMap and Secret

### ConfigMap - Configuration Data

```yaml
# configmap.yaml — externalizes config from the image so the same image works in dev/staging/prod
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "db-service"
  LOG_LEVEL: "info"
```

**Use in Deployment:**
```yaml
spec:
  containers:
    - name: app
      envFrom:
        - configMapRef:
            name: app-config   # Injects all keys as env vars — avoids listing each one individually
```

### Secret - Sensitive Data

```bash
# Create Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123
```

```yaml
# Create with YAML (requires base64 encoding — not encryption; use RBAC to restrict access)
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque                # Opaque = generic key-value; K8s also supports TLS and docker-registry types
data:
  username: YWRtaW4=      # echo -n 'admin' | base64
  password: c2VjcmV0MTIz  # echo -n 'secret123' | base64
```

**Use in Deployment:**
```yaml
spec:
  containers:
    - name: app
      env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password    # Inject a single key — safer than envFrom which exposes all keys
```

---

## 9. Namespaces

Logically separate resources.

```bash
# Namespaces provide logical isolation — same resource names can exist in different namespaces
kubectl create namespace dev
kubectl create namespace prod

# Deploy to specific namespace — keeps dev and prod resources separate in one cluster
kubectl apply -f deployment.yaml -n dev

# Change default namespace — avoids typing -n dev on every subsequent command
kubectl config set-context --current --namespace=dev
```

---

## Command Summary

| Command | Description |
|---------|-------------|
| `kubectl get pods` | List Pods |
| `kubectl get all` | List all resources |
| `kubectl apply -f file.yaml` | Create/update resource |
| `kubectl delete -f file.yaml` | Delete resource |
| `kubectl describe pod name` | Detailed information |
| `kubectl logs pod-name` | View logs |
| `kubectl exec -it pod -- sh` | Access container |
| `kubectl scale deployment name --replicas=N` | Scale |
| `kubectl rollout status` | Deployment status |
| `kubectl rollout undo` | Rollback |

---

## Exercises

### Exercise 1: Deploy Your First Pod and Deployment

Get hands-on experience with the most fundamental Kubernetes resources.

1. Start a local cluster: `minikube start`
2. Run a Pod imperatively: `kubectl run nginx-test --image=nginx:alpine`
3. Wait for it to be running: `kubectl get pods -w` (Ctrl+C to stop watching)
4. Describe the Pod and find the Node it was scheduled on: `kubectl describe pod nginx-test`
5. View logs: `kubectl logs nginx-test`
6. Delete the Pod and observe it is not recreated (compare with a Deployment)
7. Create a Deployment with 2 replicas using `kubectl create deployment web --image=nginx:alpine --replicas=2`
8. Delete one of its Pods and confirm Kubernetes creates a replacement automatically

### Exercise 2: Expose a Deployment with a Service

Practice the most common Service types to expose workloads.

1. Create a Deployment: `kubectl create deployment hello --image=nginxdemos/hello --replicas=3`
2. Expose it as a ClusterIP Service: `kubectl expose deployment hello --port=80 --type=ClusterIP`
3. Verify the Service was created: `kubectl get svc hello`
4. Use port-forwarding to access it locally: `kubectl port-forward svc/hello 8080:80`
5. Open `http://localhost:8080` in your browser and note the hostname shown (it rotates between Pods)
6. Change the Service type to NodePort: `kubectl patch svc hello -p '{"spec":{"type":"NodePort"}}'`
7. Access it via minikube: `minikube service hello --url` and test the URL

### Exercise 3: Rolling Update and Rollback

Practice zero-downtime deployments and rollbacks.

1. Create a Deployment with the older image: `kubectl create deployment app --image=nginxdemos/hello:plain-text`
2. Watch the rollout: `kubectl rollout status deployment/app`
3. Update to a new image: `kubectl set image deployment/app hello=nginx:1.25`
4. Watch the rolling update in real time: `kubectl get pods -w`
5. Check rollout history: `kubectl rollout history deployment/app`
6. Roll back to the previous version: `kubectl rollout undo deployment/app`
7. Verify the rollback was successful with `kubectl rollout status deployment/app`

### Exercise 4: ConfigMap and Secret

Store configuration and sensitive data using Kubernetes-native mechanisms.

1. Create a ConfigMap with two keys:
   ```bash
   kubectl create configmap app-config \
     --from-literal=LOG_LEVEL=info \
     --from-literal=APP_PORT=8080
   ```
2. Verify it was created: `kubectl get configmap app-config -o yaml`
3. Create a Secret:
   ```bash
   kubectl create secret generic db-secret \
     --from-literal=username=admin \
     --from-literal=password=supersecret
   ```
4. Write a Pod manifest that consumes the ConfigMap as environment variables and the Secret as a volume mount at `/secrets`
5. Apply the manifest, exec into the Pod, and verify the values: `env | grep -E "LOG_LEVEL|APP_PORT"` and `cat /secrets/password`
6. Update the ConfigMap: `kubectl edit configmap app-config` (change `LOG_LEVEL` to `debug`)
7. Observe how a volume-mounted ConfigMap updates automatically (may take ~1 minute)

### Exercise 5: Namespaces and Multi-Environment Setup

Use Namespaces to simulate isolated environments in a single cluster.

1. Create two namespaces: `kubectl create namespace dev` and `kubectl create namespace prod`
2. Deploy the same application to both namespaces:
   ```bash
   kubectl create deployment web --image=nginx:alpine -n dev
   kubectl create deployment web --image=nginx:alpine -n prod
   ```
3. Scale them differently: 1 replica in `dev`, 3 replicas in `prod`
4. List all Pods across all namespaces: `kubectl get pods --all-namespaces`
5. Switch the default context to `dev`: `kubectl config set-context --current --namespace=dev`
6. Run `kubectl get pods` and confirm it only shows the `dev` namespace Pods
7. Clean up both namespaces: `kubectl delete namespace dev prod`

---

**Previous**: [Docker Practical Examples](./05_Practical_Examples.md) | **Next**: [Kubernetes Security](./07_Kubernetes_Security.md)
