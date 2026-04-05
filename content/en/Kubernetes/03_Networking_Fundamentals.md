# 03. Networking Fundamentals

**Previous**: [Workload Resources](./02_Workload_Resources.md) | **Next**: [Storage and Persistence](./04_Storage_and_Persistence.md)

## Learning Objectives
- Understand the Kubernetes networking model and its fundamental requirements
- Configure and differentiate between Service types (ClusterIP, NodePort, LoadBalancer, ExternalName)
- Explain how CoreDNS provides service discovery within a cluster
- Compare kube-proxy modes (iptables, IPVS, nftables) and their trade-offs
- Debug networking issues using standard tools and techniques

---

Networking is the connective tissue of Kubernetes. Every pod gets its own IP
address, every service gets a stable virtual IP, and DNS ties it all together.
Unlike traditional infrastructure where networking is often an afterthought,
Kubernetes places networking at the core of its design. This lesson covers
the networking model, service abstractions, DNS, proxy modes, and debugging
techniques.

## Table of Contents
1. [The Kubernetes Networking Model](#1-the-kubernetes-networking-model)
2. [Service Types](#2-service-types)
3. [DNS in Kubernetes (CoreDNS)](#3-dns-in-kubernetes-coredns)
4. [kube-proxy Modes](#4-kube-proxy-modes)
5. [Endpoints and EndpointSlices](#5-endpoints-and-endpointslices)
6. [Service Topology and Traffic Policies](#6-service-topology-and-traffic-policies)
7. [Headless Services](#7-headless-services)
8. [Network Debugging](#8-network-debugging)
9. [Exercises](#exercises)

---

## 1. The Kubernetes Networking Model

Kubernetes imposes three fundamental networking requirements:

1. **Pod-to-Pod**: Every pod can communicate with every other pod without NAT
2. **Pod-to-Service**: Pods access services through stable virtual IPs
3. **External-to-Service**: External traffic reaches services through NodePort,
   LoadBalancer, or Ingress

### 1.1 Pod Networking

Each pod receives a unique IP address from the cluster's pod CIDR range. Containers
within the same pod share the network namespace and communicate via `localhost`.

```
┌──────────────────────────────────────┐
│  Node 1 (10.0.1.0/24)               │
│  ┌──────────┐  ┌──────────┐         │
│  │ Pod A     │  │ Pod B     │        │
│  │ 10.244.1.5│  │ 10.244.1.6│        │
│  │ ┌──┐ ┌──┐│  │ ┌──┐      │        │
│  │ │C1│ │C2││  │ │C1│      │        │
│  │ └──┘ └──┘│  │ └──┘      │        │
│  └──────────┘  └──────────┘         │
│        veth          veth            │
│         │             │              │
│     ┌───┴─────────────┴───┐         │
│     │       cbr0 / cni0   │         │
│     └──────────┬──────────┘         │
│                │                     │
└────────────────┼─────────────────────┘
                 │
          ┌──────┴──────┐
          │   Network   │
          │   Fabric    │
          └──────┬──────┘
                 │
┌────────────────┼─────────────────────┐
│  Node 2 (10.0.2.0/24)               │
│     ┌──────────┴──────────┐         │
│     │       cbr0 / cni0   │         │
│     └───┬─────────────┬───┘         │
│        veth          veth            │
│  ┌──────────┐  ┌──────────┐         │
│  │ Pod C     │  │ Pod D     │        │
│  │ 10.244.2.3│  │ 10.244.2.4│        │
│  └──────────┘  └──────────┘         │
└──────────────────────────────────────┘
```

### 1.2 CNI (Container Network Interface)

Kubernetes delegates pod networking to CNI plugins. The kubelet calls the CNI
plugin to set up networking when a pod is created.

Popular CNI plugins:

| Plugin | Approach | Key Feature |
|--------|----------|-------------|
| Calico | L3 routing (BGP) | NetworkPolicy enforcement, high performance |
| Cilium | eBPF-based | L7 visibility, no iptables dependency |
| Flannel | Overlay (VXLAN) | Simple setup, limited features |
| Weave | Overlay (mesh) | Encryption built-in |
| AWS VPC CNI | Native VPC IPs | Pods get real VPC IPs |

```bash
# Check which CNI is installed
ls /etc/cni/net.d/

# On minikube, check the CNI config
minikube ssh -- cat /etc/cni/net.d/*.conf

# View pod CIDR allocation
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}: {.spec.podCIDR}{"\n"}{end}'
```

### 1.3 Network Namespaces

Each pod runs in its own Linux network namespace. The pause container (infrastructure
container) holds the namespace, and all other containers join it.

```bash
# From a node, list network namespaces
minikube ssh -- sudo ip netns list

# Inspect a pod's network namespace
POD_ID=$(minikube ssh -- sudo crictl pods --name simple-pod -q)
minikube ssh -- sudo crictl inspectp $POD_ID | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Network namespace:', data['info']['runtimeSpec']['linux']['namespaces'])
"
```

---

## 2. Service Types

Services provide stable networking for a set of pods. They abstract away pod IP
volatility by assigning a fixed virtual IP (ClusterIP).

### 2.1 ClusterIP (Default)

Exposes the service on a cluster-internal IP. Only reachable from within the cluster.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
spec:
  type: ClusterIP          # Default; can be omitted
  selector:
    app: backend
  ports:
    - name: http
      port: 80             # Service port (what clients connect to)
      targetPort: 8080     # Container port (where traffic is forwarded)
      protocol: TCP
    - name: grpc
      port: 9090
      targetPort: 9090
      protocol: TCP
```

```bash
# Create and verify
kubectl apply -f backend-svc.yaml
kubectl get svc backend-svc

# Output:
# NAME          TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
# backend-svc   ClusterIP   10.96.142.15   <none>        80/TCP,9090/TCP  5s

# Test from within the cluster
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s http://backend-svc.default.svc.cluster.local/health
```

### 2.2 NodePort

Exposes the service on each node's IP at a static port (30000-32767).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080       # Optional; auto-assigned if omitted (30000-32767)
      protocol: TCP
```

```bash
# Access via node IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
curl http://$NODE_IP:30080

# On minikube
minikube service web-nodeport --url
```

Traffic flow: `Client → NodeIP:NodePort → ClusterIP:Port → PodIP:TargetPort`

### 2.3 LoadBalancer

Provisions an external load balancer (cloud provider required).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-lb
  annotations:
    # Cloud-specific annotations
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
    - port: 443
      targetPort: 8443
      protocol: TCP
  # Optional: restrict source IPs
  loadBalancerSourceRanges:
    - "203.0.113.0/24"
    - "198.51.100.0/24"
```

```bash
# Wait for external IP assignment
kubectl get svc web-lb -w

# Output (cloud):
# NAME     TYPE           CLUSTER-IP     EXTERNAL-IP       PORT(S)         AGE
# web-lb   LoadBalancer   10.96.45.123   a1b2c3.elb.aws    443:31234/TCP   60s

# On minikube, use tunnel
minikube tunnel
# Then access via the assigned external IP
```

### 2.4 ExternalName

Maps a service to an external DNS name (CNAME record). No proxying involved.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: db.example.com
  # No selector, no ports needed
```

```bash
# From a pod, resolving external-db returns a CNAME to db.example.com
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup external-db.default.svc.cluster.local

# Output:
# external-db.default.svc.cluster.local  canonical name = db.example.com
```

Use cases:
- Referencing external databases during migration
- Abstracting external service endpoints behind a Kubernetes name
- Gradual migration from external to in-cluster services

### 2.5 Service Comparison

| Type | Access Scope | External IP | Use Case |
|------|-------------|-------------|----------|
| ClusterIP | In-cluster only | No | Internal microservices |
| NodePort | External via node IP | Node IPs | Development, on-prem |
| LoadBalancer | External via LB | Cloud LB IP | Production cloud |
| ExternalName | DNS alias | N/A | External service reference |

---

## 3. DNS in Kubernetes (CoreDNS)

### 3.1 CoreDNS Architecture

CoreDNS is the default DNS server in Kubernetes. It runs as a Deployment in the
`kube-system` namespace and serves DNS records for services and pods.

```bash
# Check CoreDNS deployment
kubectl -n kube-system get deployment coredns

# View CoreDNS configuration
kubectl -n kube-system get configmap coredns -o yaml
```

Default CoreDNS Corefile:

```
.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
```

### 3.2 DNS Record Format

Services get DNS records in this format:

```
<service-name>.<namespace>.svc.<cluster-domain>
```

Examples:
```
# Service in default namespace
backend-svc.default.svc.cluster.local

# Service in production namespace
api-gateway.production.svc.cluster.local

# Short forms (within same namespace)
backend-svc                              # Same namespace
backend-svc.production                   # Cross-namespace
backend-svc.production.svc               # Explicit svc
backend-svc.production.svc.cluster.local # Fully qualified
```

### 3.3 DNS Records for Different Service Types

**ClusterIP service**:
```
# A record
backend-svc.default.svc.cluster.local → 10.96.142.15

# SRV record (for port discovery)
_http._tcp.backend-svc.default.svc.cluster.local → 0 100 80 backend-svc.default.svc.cluster.local
```

**Headless service** (ClusterIP: None):
```
# A record returns pod IPs directly
postgres-headless.default.svc.cluster.local → 10.244.1.5, 10.244.1.6, 10.244.2.3

# Individual pod DNS (StatefulSet only)
postgres-0.postgres-headless.default.svc.cluster.local → 10.244.1.5
postgres-1.postgres-headless.default.svc.cluster.local → 10.244.1.6
```

### 3.4 Pod DNS Configuration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: custom-dns
spec:
  dnsPolicy: None          # Use custom DNS config
  dnsConfig:
    nameservers:
      - 8.8.8.8
      - 8.8.4.4
    searches:
      - default.svc.cluster.local
      - svc.cluster.local
      - cluster.local
    options:
      - name: ndots
        value: "5"
      - name: timeout
        value: "2"
  containers:
    - name: app
      image: nginx:1.25
```

DNS policies:

| Policy | Behavior |
|--------|----------|
| ClusterFirst | Use cluster DNS; fall through to upstream (default) |
| Default | Inherit DNS config from the node |
| None | Use only `dnsConfig` settings |
| ClusterFirstWithHostNet | ClusterFirst for pods using hostNetwork |

### 3.5 DNS Performance Tuning

```bash
# Check ndots setting inside a pod
kubectl exec my-pod -- cat /etc/resolv.conf

# Output:
# nameserver 10.96.0.10
# search default.svc.cluster.local svc.cluster.local cluster.local
# options ndots:5

# With ndots:5, any name with fewer than 5 dots triggers search list expansion
# "api.example.com" (2 dots < 5) tries:
#   1. api.example.com.default.svc.cluster.local
#   2. api.example.com.svc.cluster.local
#   3. api.example.com.cluster.local
#   4. api.example.com.                       (absolute)
```

For applications making many external DNS queries, reducing `ndots` or using
FQDNs (trailing dot) improves performance:

```yaml
dnsConfig:
  options:
    - name: ndots
      value: "2"    # Reduces unnecessary DNS lookups for external names
```

---

## 4. kube-proxy Modes

kube-proxy implements the Service abstraction by programming data-plane rules
on each node. It watches Service and EndpointSlice objects and configures the
kernel networking accordingly.

### 4.1 iptables Mode (Default)

kube-proxy creates iptables rules for each Service/EndpointSlice.

```
Client Pod → iptables DNAT → Backend Pod
                 │
    ┌────────────┴────────────┐
    │  KUBE-SERVICES chain    │
    │  Match: dest=ClusterIP  │
    │  Jump: KUBE-SVC-xxx     │
    └────────────┬────────────┘
                 │
    ┌────────────┴────────────┐
    │  KUBE-SVC-xxx chain     │
    │  Random: 33% → EP1     │
    │  Random: 50% → EP2     │
    │  Default:   → EP3      │
    └─────────────────────────┘
```

```bash
# View iptables rules for a service
minikube ssh -- sudo iptables -t nat -L KUBE-SERVICES -n | grep backend-svc

# View the specific service chain
minikube ssh -- sudo iptables -t nat -L KUBE-SVC-XXXXXXXX -n

# Count iptables rules (grows with number of services)
minikube ssh -- sudo iptables -t nat -L | wc -l
```

Characteristics:
- **Pros**: Stable, well-tested, works everywhere
- **Cons**: O(n) rule evaluation; slow with thousands of services
- **Load balancing**: Random with equal probability
- **Connection tracking**: Uses conntrack for established connections

### 4.2 IPVS Mode

Uses the Linux IPVS (IP Virtual Server) kernel module for L4 load balancing.

```bash
# Enable IPVS mode in kube-proxy
kubectl -n kube-system edit configmap kube-proxy
# Set mode: "ipvs"
# Then restart kube-proxy pods

# Or on minikube
minikube start --extra-config=kube-proxy.mode=ipvs
```

```bash
# View IPVS rules
minikube ssh -- sudo ipvsadm -Ln

# Output example:
# TCP  10.96.142.15:80 rr
#   -> 10.244.1.5:8080    Masq    1      0       0
#   -> 10.244.1.6:8080    Masq    1      0       0
#   -> 10.244.2.3:8080    Masq    1      0       0
```

IPVS scheduling algorithms:

| Algorithm | Flag | Description |
|-----------|------|-------------|
| Round Robin | rr | Equal distribution |
| Least Connections | lc | Prefer less-loaded backends |
| Destination Hashing | dh | Consistent hashing by destination |
| Source Hashing | sh | Same source → same backend |
| Shortest Expected Delay | sed | Weighted least connections |

Characteristics:
- **Pros**: O(1) lookup, multiple LB algorithms, better scalability
- **Cons**: Requires IPVS kernel modules, slightly more complex debugging
- **Scale**: Handles 10,000+ services efficiently

### 4.3 nftables Mode (v1.29+)

Uses nftables, the successor to iptables:

```bash
# Enable nftables mode
# Set mode: "nftables" in kube-proxy configmap

# View nftables rules
minikube ssh -- sudo nft list ruleset | grep kube
```

Characteristics:
- **Pros**: Better performance than iptables, atomic rule updates
- **Cons**: Newer, less battle-tested
- **Compatibility**: Requires Linux kernel 5.13+

### 4.4 Mode Comparison

| Feature | iptables | IPVS | nftables |
|---------|----------|------|----------|
| Lookup complexity | O(n) | O(1) | O(1) |
| LB algorithms | Random | Multiple | Random |
| Max services | ~5,000 | 10,000+ | 10,000+ |
| Rule updates | Full replace | Incremental | Atomic |
| Session affinity | Yes | Yes | Yes |
| Kernel requirement | Any | IPVS modules | 5.13+ |

---

## 5. Endpoints and EndpointSlices

### 5.1 Endpoints (Legacy)

An Endpoints object contains the IP addresses of pods matching a Service selector.

```bash
# View endpoints for a service
kubectl get endpoints backend-svc

# Output:
# NAME          ENDPOINTS                                      AGE
# backend-svc   10.244.1.5:8080,10.244.1.6:8080,10.244.2.3:8080   5m

# Detailed view
kubectl describe endpoints backend-svc
```

Limitations of Endpoints:
- One Endpoints object per Service, containing all pod IPs
- Does not scale well beyond ~1,000 endpoints
- Full object must be transferred on any change

### 5.2 EndpointSlices (Modern)

EndpointSlices solve the scalability problem by splitting endpoints into smaller
chunks (default: 100 endpoints per slice).

```bash
# View EndpointSlices
kubectl get endpointslices -l kubernetes.io/service-name=backend-svc

# Output:
# NAME                  ADDRESSTYPE   PORTS   ENDPOINTS                  AGE
# backend-svc-abc12     IPv4          8080    10.244.1.5,10.244.1.6,...  5m

# Detailed view
kubectl describe endpointslice backend-svc-abc12
```

```yaml
# EndpointSlice structure (auto-managed by the endpoint controller)
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: backend-svc-abc12
  labels:
    kubernetes.io/service-name: backend-svc
addressType: IPv4
ports:
  - name: http
    protocol: TCP
    port: 8080
endpoints:
  - addresses:
      - "10.244.1.5"
    conditions:
      ready: true
      serving: true
      terminating: false
    nodeName: node-1
    zone: us-east-1a
  - addresses:
      - "10.244.1.6"
    conditions:
      ready: true
      serving: true
      terminating: false
    nodeName: node-1
    zone: us-east-1a
```

### 5.3 Manual Endpoints

For services without selectors (pointing to external resources):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-database
spec:
  # No selector
  ports:
    - port: 5432
      targetPort: 5432
---
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: external-database-1
  labels:
    kubernetes.io/service-name: external-database
addressType: IPv4
ports:
  - port: 5432
    protocol: TCP
endpoints:
  - addresses:
      - "192.168.1.100"   # External database IP
  - addresses:
      - "192.168.1.101"   # Replica
```

---

## 6. Service Topology and Traffic Policies

### 6.1 Internal Traffic Policy

Controls how traffic from pods within the cluster reaches service endpoints.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
spec:
  selector:
    app: backend
  ports:
    - port: 80
      targetPort: 8080
  internalTrafficPolicy: Local    # Only route to pods on the same node
  # internalTrafficPolicy: Cluster  # Route to any pod (default)
```

`Local` policy:
- Routes only to endpoints on the same node
- Reduces latency and cross-node traffic
- Risk: if no local endpoints, traffic is dropped (503)

### 6.2 External Traffic Policy

Controls how traffic from external sources reaches service endpoints.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-lb
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
    - port: 80
      targetPort: 8080
  externalTrafficPolicy: Local    # Preserve client source IP
  # externalTrafficPolicy: Cluster  # May SNAT, losing source IP (default)
```

| Policy | Source IP Preserved | Load Distribution | Failure Mode |
|--------|--------------------|-------------------|--------------|
| Cluster | No (SNAT) | Even across all pods | Always works |
| Local | Yes | Only local endpoints | 503 if no local pods |

### 6.3 Topology Aware Routing (v1.27+)

Prefers routing to endpoints in the same zone:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
  annotations:
    service.kubernetes.io/topology-mode: Auto
spec:
  selector:
    app: backend
  ports:
    - port: 80
      targetPort: 8080
```

When enabled, kube-proxy programs hints from EndpointSlices to prefer same-zone
endpoints, reducing cross-zone traffic and latency.

```bash
# Verify topology hints are set
kubectl get endpointslice -l kubernetes.io/service-name=backend-svc -o yaml | grep -A 5 hints
```

---

## 7. Headless Services

A headless service has `clusterIP: None`. Instead of a virtual IP, DNS returns
the pod IPs directly.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: app-headless
spec:
  clusterIP: None
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

### 7.1 DNS Behavior

```bash
# Regular service: returns single ClusterIP
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup backend-svc
# Address: 10.96.142.15

# Headless service: returns all pod IPs
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup app-headless
# Address: 10.244.1.5
# Address: 10.244.1.6
# Address: 10.244.2.3
```

### 7.2 Use Cases

| Use Case | Why Headless? |
|----------|--------------|
| StatefulSets | Stable per-pod DNS (pod-0.svc, pod-1.svc) |
| Client-side load balancing | Application chooses which pod to connect to |
| Service mesh | Sidecar proxies handle routing, not kube-proxy |
| Database clusters | Need to address specific replicas |
| gRPC | Client-side load balancing across connections |

### 7.3 Headless with StatefulSet

```yaml
apiVersion: v1
kind: Service
metadata:
  name: cassandra-headless
spec:
  clusterIP: None
  selector:
    app: cassandra
  ports:
    - port: 9042
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cassandra
spec:
  serviceName: cassandra-headless
  replicas: 3
  selector:
    matchLabels:
      app: cassandra
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      containers:
        - name: cassandra
          image: cassandra:4.1
          ports:
            - containerPort: 9042
          env:
            - name: CASSANDRA_SEEDS
              # Use headless DNS to discover seed nodes
              value: "cassandra-0.cassandra-headless.default.svc.cluster.local"
```

Each pod gets a DNS entry:
```
cassandra-0.cassandra-headless.default.svc.cluster.local
cassandra-1.cassandra-headless.default.svc.cluster.local
cassandra-2.cassandra-headless.default.svc.cluster.local
```

---

## 8. Network Debugging

### 8.1 Debugging Toolkit

```bash
# Deploy a debug pod with networking tools
kubectl run netshoot --rm -it --image=nicolaka/netshoot --restart=Never -- /bin/bash

# Inside the debug pod:
# DNS resolution
nslookup backend-svc.default.svc.cluster.local
dig backend-svc.default.svc.cluster.local +short

# HTTP connectivity
curl -v http://backend-svc/health

# TCP connectivity
nc -zv backend-svc 80

# Trace route
traceroute backend-svc

# DNS response time
dig @10.96.0.10 backend-svc.default.svc.cluster.local +stats

# Check /etc/resolv.conf
cat /etc/resolv.conf
```

### 8.2 Common Networking Issues

#### Service not reachable

```bash
# 1. Check the service exists and has endpoints
kubectl get svc backend-svc
kubectl get endpoints backend-svc

# 2. If endpoints are empty, check selector matches
kubectl get svc backend-svc -o jsonpath='{.spec.selector}'
kubectl get pods -l app=backend

# 3. Check pod readiness (unready pods are removed from endpoints)
kubectl get pods -l app=backend -o wide

# 4. Check kube-proxy is running
kubectl -n kube-system get pods -l k8s-app=kube-proxy
```

#### DNS resolution fails

```bash
# 1. Check CoreDNS pods are running
kubectl -n kube-system get pods -l k8s-app=kube-dns

# 2. Check CoreDNS logs
kubectl -n kube-system logs -l k8s-app=kube-dns --tail=20

# 3. Test DNS directly
kubectl run dns-debug --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup kubernetes.default.svc.cluster.local

# 4. Check CoreDNS configmap
kubectl -n kube-system get configmap coredns -o yaml
```

#### Cross-namespace communication

```bash
# Pods can always reach services in other namespaces using FQDN
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s http://backend-svc.production.svc.cluster.local/health

# NetworkPolicies may block cross-namespace traffic
kubectl get networkpolicies --all-namespaces
```

### 8.3 NetworkPolicy

NetworkPolicies control pod-to-pod traffic at the network level (requires a
CNI that supports NetworkPolicy, such as Calico or Cilium).

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
    - Egress

  ingress:
    # Allow traffic from frontend pods in the same namespace
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080

    # Allow traffic from monitoring namespace
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090

  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53

    # Allow connection to database
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

```bash
# List NetworkPolicies
kubectl get networkpolicies -n production

# Test connectivity after applying policy
kubectl -n production exec frontend-pod -- curl -s http://backend-svc:8080/health
# Should work (allowed by ingress rule)

kubectl -n production exec other-pod -- curl -s http://backend-svc:8080/health
# Should fail (not in allowed ingress)
```

### 8.4 Packet Capture

```bash
# Capture packets on a specific pod's network interface
# First, find the pod's node and container ID
NODE=$(kubectl get pod my-pod -o jsonpath='{.spec.nodeName}')
CONTAINER_ID=$(kubectl get pod my-pod -o jsonpath='{.status.containerStatuses[0].containerID}' | cut -d/ -f3)

# Use nsenter to enter the pod's network namespace (from the node)
# Then use tcpdump
kubectl debug node/$NODE -it --image=nicolaka/netshoot -- \
  nsenter -t $(crictl inspect $CONTAINER_ID | jq .info.pid) -n \
  tcpdump -i eth0 -c 20 port 8080
```

### 8.5 Service Connectivity Matrix

```bash
# Quick connectivity test script
cat <<'SCRIPT' > /tmp/test-connectivity.sh
#!/bin/bash
SERVICES=("frontend" "backend" "database")
for src in "${SERVICES[@]}"; do
  for dst in "${SERVICES[@]}"; do
    if [ "$src" != "$dst" ]; then
      result=$(kubectl exec deploy/$src -- curl -s -o /dev/null -w "%{http_code}" http://$dst/ 2>/dev/null)
      echo "$src -> $dst: $result"
    fi
  done
done
SCRIPT
bash /tmp/test-connectivity.sh
```

---

## Exercises

### Exercise 1: Service Discovery

Create a Deployment with 3 replicas and a ClusterIP Service. From a debug pod,
verify DNS resolution returns the ClusterIP, and that HTTP requests are load-balanced
across all pods.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/svc-discovery.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whoami
spec:
  replicas: 3
  selector:
    matchLabels:
      app: whoami
  template:
    metadata:
      labels:
        app: whoami
    spec:
      containers:
        - name: whoami
          image: traefik/whoami:v1.10
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: whoami-svc
spec:
  selector:
    app: whoami
  ports:
    - port: 80
      targetPort: 80
```

```bash
kubectl apply -f /tmp/svc-discovery.yaml
kubectl wait --for=condition=Available deployment/whoami --timeout=60s

# Verify DNS resolution
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup whoami-svc.default.svc.cluster.local
# Should return the ClusterIP (e.g., 10.96.xxx.xxx)

# Verify load balancing (make 10 requests, check different pod hostnames)
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  sh -c 'for i in $(seq 1 10); do curl -s http://whoami-svc/ | grep Hostname; done'
# Should show different hostnames (pod names), demonstrating load balancing

# Verify endpoints
kubectl get endpoints whoami-svc
# Should show 3 pod IPs

# Clean up
kubectl delete -f /tmp/svc-discovery.yaml
```

</details>

### Exercise 2: NodePort Service

Expose a web application using a NodePort service on port 30080. Access it from
outside the cluster using the node's IP address.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/nodeport-exercise.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-nodeport
spec:
  type: NodePort
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30080
```

```bash
kubectl apply -f /tmp/nodeport-exercise.yaml
kubectl wait --for=condition=Available deployment/web-app --timeout=60s

# Get node IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
echo "Node IP: $NODE_IP"

# Access via NodePort (on minikube)
minikube service web-app-nodeport --url
# Or directly:
curl http://$NODE_IP:30080

# Verify the service
kubectl get svc web-app-nodeport
# NAME               TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
# web-app-nodeport   NodePort   10.96.x.x      <none>        80:30080/TCP   30s

# Verify endpoints
kubectl get endpoints web-app-nodeport
# Should show 2 pod IPs

# Clean up
kubectl delete -f /tmp/nodeport-exercise.yaml
```

</details>

### Exercise 3: NetworkPolicy

Create two namespaces (`frontend` and `backend`). Deploy an app in each. Create a
NetworkPolicy that only allows the frontend namespace to access the backend service
on port 8080.

<details>
<summary>Show Answer</summary>

```bash
# Create namespaces with labels
kubectl create namespace frontend
kubectl label namespace frontend name=frontend
kubectl create namespace backend
kubectl label namespace backend name=backend
```

```yaml
# Save as /tmp/netpol-exercise.yaml
# Backend deployment and service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
        - name: whoami
          image: traefik/whoami:v1.10
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: api-server
  namespace: backend
spec:
  selector:
    app: api-server
  ports:
    - port: 8080
      targetPort: 80
---
# NetworkPolicy: only allow from frontend namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-only
  namespace: backend
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: frontend
      ports:
        - protocol: TCP
          port: 80
```

```bash
kubectl apply -f /tmp/netpol-exercise.yaml

# Test from frontend namespace (should succeed)
kubectl -n frontend run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s --max-time 5 http://api-server.backend.svc.cluster.local:8080/
# Should return whoami response

# Test from default namespace (should fail/timeout)
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s --max-time 5 http://api-server.backend.svc.cluster.local:8080/
# Should timeout (blocked by NetworkPolicy)

# Verify NetworkPolicy
kubectl -n backend get networkpolicy allow-frontend-only

# Clean up
kubectl delete namespace frontend backend
```

</details>

### Exercise 4: Headless Service with DNS

Create a StatefulSet with a headless service. Verify that DNS returns individual
pod IPs and that each pod has a stable DNS name.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/headless-exercise.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-headless
spec:
  clusterIP: None
  selector:
    app: web-sts
  ports:
    - port: 80
      targetPort: 80
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web-sts
spec:
  serviceName: web-headless
  replicas: 3
  selector:
    matchLabels:
      app: web-sts
  template:
    metadata:
      labels:
        app: web-sts
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
```

```bash
kubectl apply -f /tmp/headless-exercise.yaml
kubectl rollout status statefulset/web-sts

# Verify headless DNS returns all pod IPs
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup web-headless.default.svc.cluster.local
# Should return 3 pod IP addresses (no ClusterIP)

# Verify individual pod DNS names
for i in 0 1 2; do
  echo "=== web-sts-$i ==="
  kubectl run dns-test-$i --rm -it --image=busybox:1.36 --restart=Never -- \
    nslookup web-sts-$i.web-headless.default.svc.cluster.local
done
# Each should resolve to the specific pod's IP

# Compare with pod IPs
kubectl get pods -l app=web-sts -o wide
# The DNS IPs should match the pod IPs

# Verify no ClusterIP is assigned
kubectl get svc web-headless
# CLUSTER-IP should show "None"

# Clean up
kubectl delete -f /tmp/headless-exercise.yaml
```

</details>

### Exercise 5: DNS Debugging

A pod cannot reach a service called `payment-api` in the `billing` namespace.
Walk through a systematic debugging process to identify and fix the issue.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Verify the service exists
kubectl get svc payment-api -n billing
# If not found, the service does not exist

# Step 2: Check if the namespace exists
kubectl get namespace billing
# If not found, create it

# Step 3: Create the test environment
kubectl create namespace billing

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-api
  namespace: billing
spec:
  replicas: 2
  selector:
    matchLabels:
      app: payment-api
  template:
    metadata:
      labels:
        app: payment-api
    spec:
      containers:
        - name: api
          image: traefik/whoami:v1.10
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: payment-api
  namespace: billing
spec:
  selector:
    app: payment-api
  ports:
    - port: 80
      targetPort: 80
EOF

# Step 4: Systematic debugging
# 4a. Check CoreDNS is running
kubectl -n kube-system get pods -l k8s-app=kube-dns

# 4b. Test DNS resolution from debug pod
kubectl run debug --rm -it --image=nicolaka/netshoot --restart=Never -- \
  bash -c '
    echo "=== DNS Test ==="
    nslookup payment-api.billing.svc.cluster.local

    echo "=== Connectivity Test ==="
    curl -s --max-time 5 http://payment-api.billing.svc.cluster.local/

    echo "=== resolv.conf ==="
    cat /etc/resolv.conf

    echo "=== Route to DNS ==="
    traceroute -m 3 10.96.0.10
  '

# 4c. Check endpoints
kubectl get endpoints payment-api -n billing
# If empty: selector mismatch or pods not ready

# 4d. Check pod readiness
kubectl get pods -n billing -l app=payment-api

# 4e. Check for NetworkPolicies blocking traffic
kubectl get networkpolicies -n billing

# Clean up
kubectl delete namespace billing
```

</details>

---

**Previous**: [Workload Resources](./02_Workload_Resources.md) | **Next**: [Storage and Persistence](./04_Storage_and_Persistence.md)
