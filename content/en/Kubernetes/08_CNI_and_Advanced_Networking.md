# 08. CNI and Advanced Networking

**Previous**: [Ingress and Gateway API](./07_Ingress_and_Gateway_API.md) | **Next**: [Helm and Kustomize](./09_Helm_and_Kustomize.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Container Network Interface (CNI) specification and plugin lifecycle
2. Compare Calico, Cilium, and other CNI plugins across data plane technologies (iptables, eBPF, IPVS)
3. Write advanced NetworkPolicies covering egress rules, CIDR blocks, and DNS-aware filtering
4. Describe eBPF fundamentals and how Cilium leverages them for networking and observability
5. Troubleshoot Kubernetes networking issues using standard diagnostic tools

---

Kubernetes networking is deceptively simple at the surface -- every pod gets an IP, every service gets a DNS name -- but the underlying implementation involves complex interactions between CNI plugins, iptables/eBPF rules, routing tables, and overlay networks. This lesson dives into the CNI specification, explores the two dominant plugins (Calico and Cilium), introduces eBPF as a networking primitive, and covers advanced NetworkPolicy patterns, service mesh fundamentals, and network troubleshooting.

> **The Kubernetes Network Model:** Kubernetes imposes three fundamental requirements: (1) every pod gets a unique IP, (2) pods can communicate with any other pod without NAT, and (3) agents on a node can communicate with all pods on that node. How this is achieved is entirely up to the CNI plugin.

## Table of Contents

- [1. CNI Specification](#1-cni-specification)
  - [1.1 How CNI Works](#11-how-cni-works)
  - [1.2 CNI Plugin Lifecycle](#12-cni-plugin-lifecycle)
  - [1.3 CNI Configuration](#13-cni-configuration)
- [2. Calico](#2-calico)
  - [2.1 Architecture](#21-architecture)
  - [2.2 Data Plane Modes](#22-data-plane-modes)
  - [2.3 Calico NetworkPolicy Extensions](#23-calico-networkpolicy-extensions)
- [3. Cilium](#3-cilium)
  - [3.1 eBPF-Based Architecture](#31-ebpf-based-architecture)
  - [3.2 Installation and Configuration](#32-installation-and-configuration)
  - [3.3 Cilium Network Policies](#33-cilium-network-policies)
  - [3.4 Hubble Observability](#34-hubble-observability)
- [4. eBPF Fundamentals](#4-ebpf-fundamentals)
  - [4.1 What is eBPF?](#41-what-is-ebpf)
  - [4.2 eBPF for Networking](#42-ebpf-for-networking)
  - [4.3 eBPF vs iptables](#43-ebpf-vs-iptables)
- [5. Advanced NetworkPolicy](#5-advanced-networkpolicy)
  - [5.1 Egress Policies](#51-egress-policies)
  - [5.2 CIDR-Based Rules](#52-cidr-based-rules)
  - [5.3 Port Range Policies](#53-port-range-policies)
  - [5.4 DNS-Aware Policies (Cilium)](#54-dns-aware-policies-cilium)
- [6. Service Mesh Overview](#6-service-mesh-overview)
  - [6.1 What is a Service Mesh?](#61-what-is-a-service-mesh)
  - [6.2 Istio](#62-istio)
  - [6.3 Linkerd](#63-linkerd)
  - [6.4 Cilium Service Mesh](#64-cilium-service-mesh)
- [7. Bandwidth and QoS](#7-bandwidth-and-qos)
- [8. IPv4/IPv6 Dual-Stack](#8-ipv4ipv6-dual-stack)
- [9. Network Troubleshooting](#9-network-troubleshooting)
  - [9.1 Diagnostic Tools](#91-diagnostic-tools)
  - [9.2 Common Issues](#92-common-issues)
- [Exercises](#exercises)

---

## 1. CNI Specification

The Container Network Interface (CNI) is a specification that defines how container runtimes configure networking for containers. Kubernetes uses CNI plugins to assign IP addresses, configure routes, and set up network namespaces for pods.

### 1.1 How CNI Works

```
┌──────────────────────────────────────────────────────────────┐
│  Node                                                        │
│                                                              │
│  ┌──────────┐    1. CreatePod    ┌──────────┐                │
│  │ kubelet  │───────────────────▶│ CRI      │                │
│  │          │                    │(containerd)                │
│  └──────────┘                    └────┬─────┘                │
│                                       │                      │
│                              2. CNI ADD                      │
│                                       ▼                      │
│                                ┌──────────────┐              │
│                                │  CNI Plugin   │              │
│                                │ (calico/cilium)│             │
│                                └──────┬───────┘              │
│                                       │                      │
│                      3. Create veth pair, assign IP          │
│                         configure routes                     │
│                                       │                      │
│  ┌──────────────────┐                 │                      │
│  │  Pod Network NS  │◀───────────────┘                       │
│  │  eth0: 10.0.1.5  │                                       │
│  └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 CNI Plugin Lifecycle

CNI plugins implement these operations:

| Operation | Description | When Called |
|-----------|-------------|------------|
| `ADD` | Set up networking for a container | Pod creation |
| `DEL` | Clean up networking for a container | Pod deletion |
| `CHECK` | Verify networking is still correct | Health checks |
| `VERSION` | Report supported CNI versions | Discovery |

A simple CNI plugin interaction (for understanding -- not something you write daily):

```go
// CNI plugin skeleton in Go
package main

import (
	"encoding/json"
	"fmt"
	"net"
	"runtime"

	"github.com/containernetworking/cni/pkg/skel"
	"github.com/containernetworking/cni/pkg/types"
	current "github.com/containernetworking/cni/pkg/types/100"
	"github.com/containernetworking/cni/pkg/version"
)

// NetConf represents the CNI network configuration
type NetConf struct {
	types.NetConf
	Subnet string `json:"subnet"`
}

func cmdAdd(args *skel.CmdArgs) error {
	conf := &NetConf{}
	if err := json.Unmarshal(args.StdinData, conf); err != nil {
		return fmt.Errorf("failed to parse config: %v", err)
	}

	// 1. Allocate an IP address from the subnet
	_, subnet, _ := net.ParseCIDR(conf.Subnet)
	ip := allocateIP(subnet) // implementation omitted

	// 2. Create a veth pair (one end in pod, one on host)
	// 3. Move one end into the container network namespace
	// 4. Assign the IP to the container interface
	// 5. Set up routes

	result := &current.Result{
		CNIVersion: conf.CNIVersion,
		IPs: []*current.IPConfig{
			{
				Address: net.IPNet{
					IP:   ip,
					Mask: subnet.Mask,
				},
				Gateway: gatewayIP(subnet),
			},
		},
	}
	return types.PrintResult(result, conf.CNIVersion)
}

func cmdDel(args *skel.CmdArgs) error {
	// Clean up: remove veth pair, release IP
	return nil
}

func cmdCheck(args *skel.CmdArgs) error {
	// Verify networking is still configured correctly
	return nil
}

func main() {
	runtime.LockOSThread()
	skel.PluginMainFuncs(skel.CNIFuncs{
		Add:   cmdAdd,
		Del:   cmdDel,
		Check: cmdCheck,
	}, version.All, "example-cni-plugin")
}
```

### 1.3 CNI Configuration

CNI configuration is stored on each node, typically in `/etc/cni/net.d/`.

```json
{
  "cniVersion": "1.0.0",
  "name": "k8s-pod-network",
  "type": "calico",
  "datastore_type": "kubernetes",
  "mtu": 1440,
  "nodename_file_optional": false,
  "log_level": "Info",
  "log_file_path": "/var/log/calico/cni/cni.log",
  "ipam": {
    "type": "calico-ipam",
    "assign_ipv4": "true",
    "assign_ipv6": "false"
  },
  "container_settings": {
    "allow_ip_forwarding": false
  },
  "policy": {
    "type": "k8s"
  },
  "kubernetes": {
    "kubeconfig": "/etc/cni/net.d/calico-kubeconfig"
  }
}
```

```bash
# Inspect CNI configuration on a node
ssh node01 "ls /etc/cni/net.d/"
# 10-calico.conflist

ssh node01 "cat /etc/cni/net.d/10-calico.conflist"

# CNI plugin binaries location
ssh node01 "ls /opt/cni/bin/"
# bandwidth  bridge  calico  calico-ipam  dhcp  flannel  host-local  ...
```

---

## 2. Calico

Calico is one of the most widely deployed CNI plugins. It provides networking and network policy enforcement using BGP routing, IP-in-IP tunneling, or VXLAN overlays.

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Calico Architecture                                        │
│                                                              │
│  ┌──────────────────┐     ┌──────────────────┐              │
│  │    calico-node   │     │    calico-node   │              │
│  │   (DaemonSet)    │     │   (DaemonSet)    │              │
│  │                  │     │                  │              │
│  │ ┌──────────────┐ │     │ ┌──────────────┐ │              │
│  │ │ Felix         │ │     │ │ Felix         │ │              │
│  │ │ (policy      │ │     │ │ (policy      │ │              │
│  │ │  enforcement) │ │     │ │  enforcement) │ │              │
│  │ └──────────────┘ │     │ └──────────────┘ │              │
│  │ ┌──────────────┐ │     │ ┌──────────────┐ │              │
│  │ │ BIRD         │ │◀───▶│ │ BIRD         │ │              │
│  │ │ (BGP daemon) │ │ BGP │ │ (BGP daemon) │ │              │
│  │ └──────────────┘ │     │ └──────────────┘ │              │
│  └──────────────────┘     └──────────────────┘              │
│                                                              │
│  ┌───────────────────────────────────────────┐              │
│  │ calico-kube-controllers (Deployment)      │              │
│  │ Syncs Calico datastore with K8s API       │              │
│  └───────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

```bash
# Install Calico on a cluster
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calico.yaml

# Or via the Tigera operator (recommended for production)
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/custom-resources.yaml

# Verify Calico is running
kubectl get pods -n calico-system
# NAME                                      READY   STATUS
# calico-node-xxxxx                         1/1     Running
# calico-kube-controllers-xxxxx             1/1     Running

# Install calicoctl CLI
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calicoctl.yaml
```

### 2.2 Data Plane Modes

Calico supports three encapsulation modes:

| Mode | How it works | When to use |
|------|-------------|-------------|
| **BGP (no overlay)** | Distributes routes via BGP | On-premises with BGP-capable routers |
| **IP-in-IP** | Encapsulates packets in IP headers | Cross-subnet communication |
| **VXLAN** | UDP-based overlay | Cloud environments, no BGP support |

```yaml
# Calico IPPool configuration
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: default-ipv4-ippool
spec:
  cidr: 192.168.0.0/16
  ipipMode: CrossSubnet      # IP-in-IP only across subnets
  vxlanMode: Never
  natOutgoing: true
  nodeSelector: all()
```

```yaml
# VXLAN mode (for cloud environments)
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: vxlan-pool
spec:
  cidr: 10.244.0.0/16
  ipipMode: Never
  vxlanMode: Always
  natOutgoing: true
  blockSize: 26              # /26 = 64 IPs per node block
```

```bash
# Check current Calico configuration
kubectl exec -n calico-system calico-node-xxxxx -- calico-node -show-status

# View IP pools
kubectl get ippools -o yaml

# View node-to-node BGP peering
kubectl exec -n calico-system calico-node-xxxxx -- birdcl show protocols
```

### 2.3 Calico NetworkPolicy Extensions

Calico extends the standard Kubernetes NetworkPolicy with additional features.

```yaml
# Calico GlobalNetworkPolicy: applies across all namespaces
apiVersion: crd.projectcalico.org/v1
kind: GlobalNetworkPolicy
metadata:
  name: deny-external-egress
spec:
  selector: role == "internal"
  types:
    - Egress
  egress:
    # Allow traffic within the cluster
    - action: Allow
      destination:
        nets:
          - 10.0.0.0/8
          - 172.16.0.0/12
          - 192.168.0.0/16
    # Block everything else
    - action: Deny
```

```yaml
# Calico NetworkPolicy with application layer policy (HTTP methods)
apiVersion: crd.projectcalico.org/v1
kind: NetworkPolicy
metadata:
  name: allow-get-only
  namespace: production
spec:
  selector: app == "api"
  types:
    - Ingress
  ingress:
    - action: Allow
      protocol: TCP
      source:
        selector: role == "frontend"
      destination:
        ports: [8080]
      http:
        methods: ["GET", "HEAD"]    # L7 policy (requires Envoy proxy)
```

---

## 3. Cilium

Cilium is a CNI plugin that uses eBPF to provide networking, security, and observability. It operates at the kernel level without iptables, delivering better performance and richer features.

### 3.1 eBPF-Based Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Cilium Architecture                                        │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │  cilium-agent (DaemonSet per node)       │               │
│  │                                          │               │
│  │  ┌────────────────┐  ┌────────────────┐  │               │
│  │  │ eBPF programs  │  │ eBPF programs  │  │               │
│  │  │ (TC ingress)   │  │ (TC egress)    │  │               │
│  │  └───────┬────────┘  └───────┬────────┘  │               │
│  │          │                    │           │               │
│  │  ┌───────▼────────────────────▼────────┐  │               │
│  │  │     eBPF Maps (kernel data plane)   │  │               │
│  │  │  - Connection tracking              │  │               │
│  │  │  - Policy maps                      │  │               │
│  │  │  - Service maps (replaces kube-proxy)│ │               │
│  │  │  - NAT maps                         │  │               │
│  │  └────────────────────────────────────┘  │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  ┌───────────────────────────────┐                          │
│  │  Hubble (observability layer) │                          │
│  │  - Flow logs                  │                          │
│  │  - Service map                │                          │
│  │  - Metrics (Prometheus)       │                          │
│  └───────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Installation and Configuration

```bash
# Install Cilium CLI
curl -L --remote-name https://github.com/cilium/cilium-cli/releases/latest/download/cilium-linux-amd64.tar.gz
tar xzvf cilium-linux-amd64.tar.gz
sudo mv cilium /usr/local/bin/

# Install Cilium on a cluster (replaces kube-proxy)
cilium install --version 1.16.0 \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=API_SERVER_IP \
  --set k8sServicePort=6443

# Or via Helm
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium \
  --namespace kube-system \
  --set kubeProxyReplacement=true \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true

# Verify installation
cilium status
# Output:
#     /\
#  /\  Warning: Unable to detect...
# /  \
# \  /  Cilium:       OK
#  \/   Operator:     OK
#       Hubble:       OK

# Run connectivity test
cilium connectivity test
```

```bash
# Start minikube with Cilium (no kube-proxy)
minikube start --network-plugin=cni --cni=false
cilium install
cilium status --wait
```

### 3.3 Cilium Network Policies

Cilium extends Kubernetes NetworkPolicy with L7 (application layer) awareness and DNS-based rules.

```yaml
# CiliumNetworkPolicy: L7 HTTP filtering
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: api-l7-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: api-server
  ingress:
    - fromEndpoints:
        - matchLabels:
            app: frontend
      toPorts:
        - ports:
            - port: "8080"
              protocol: TCP
          rules:
            http:
              - method: "GET"
                path: "/api/v1/.*"
              - method: "POST"
                path: "/api/v1/orders"
                headers:
                  - 'Content-Type: application/json'
```

```yaml
# CiliumClusterWideNetworkPolicy: cluster-wide default deny
apiVersion: cilium.io/v2
kind: CiliumClusterwideNetworkPolicy
metadata:
  name: default-deny
spec:
  endpointSelector: {}
  ingress:
    - fromEndpoints:
        - {}                    # Allow within-cluster traffic only
  egress:
    - toEndpoints:
        - {}
    - toEntities:
        - kube-dns              # Always allow DNS
```

### 3.4 Hubble Observability

Hubble provides deep network observability using eBPF flow data.

```bash
# Enable Hubble
cilium hubble enable --ui

# Port-forward Hubble UI
cilium hubble ui
# Opens browser at http://localhost:12000

# Use Hubble CLI for flow observation
hubble observe --namespace production
# Timestamp  Source                Destination           Type    Verdict
# 10:30:01   production/frontend   production/api        L7/HTTP  FORWARDED
# 10:30:01   production/api        production/database   L4/TCP   FORWARDED

# Filter specific flows
hubble observe --namespace production \
  --from-pod production/frontend \
  --to-pod production/api \
  --protocol http

# Get flow summary
hubble observe --namespace production -o json | \
  jq '.flow | {src: .source.labels, dst: .destination.labels, verdict: .verdict}'
```

---

## 4. eBPF Fundamentals

### 4.1 What is eBPF?

eBPF (extended Berkeley Packet Filter) is a technology that allows programs to run in the Linux kernel without modifying kernel source code or loading kernel modules. It is a programmable data plane inside the kernel.

```
┌─────────────────────────────────────────────────────────────┐
│  User Space                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ cilium-agent │  │ Hubble       │  │ bpftool      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
├─────────┼──────────────────┼──────────────────┼──────────────┤
│  Kernel │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌──────────────────────────────────────────────────┐       │
│  │              eBPF Virtual Machine                 │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │       │
│  │  │  TC     │  │ XDP     │  │ Socket  │          │       │
│  │  │ ingress │  │(express │  │ filter  │  ...     │       │
│  │  │/egress  │  │ data    │  │         │          │       │
│  │  │         │  │ path)   │  │         │          │       │
│  │  └─────────┘  └─────────┘  └─────────┘          │       │
│  │                                                   │       │
│  │  eBPF Maps (shared kernel/user state):            │       │
│  │  - Hash maps, arrays, LRU, ring buffers           │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 eBPF for Networking

eBPF hook points relevant to Kubernetes networking:

| Hook Point | Location | Use Case |
|-----------|----------|----------|
| **XDP** (eXpress Data Path) | NIC driver, before kernel stack | DDoS mitigation, load balancing |
| **TC** (Traffic Control) | After network stack | Pod networking, policy enforcement |
| **Socket** | Socket operations | Connection tracking, load balancing |
| **cgroup** | cgroup-level | Per-pod bandwidth limits |

### 4.3 eBPF vs iptables

| Aspect | iptables | eBPF (Cilium) |
|--------|----------|---------------|
| Rule evaluation | Linear chain traversal | Hash map lookup (O(1)) |
| Service performance | Degrades with service count | Constant regardless of count |
| Update latency | Full chain rebuild | Incremental map update |
| L7 visibility | None | HTTP, gRPC, Kafka, DNS |
| Connection tracking | conntrack module | BPF maps |
| Resource usage | Higher with many rules | Lower, scales better |

```bash
# Compare service routing with iptables vs eBPF
# iptables: O(n) rules for n services
sudo iptables -t nat -L KUBE-SERVICES | wc -l
# 500+ rules for a modest cluster

# eBPF: O(1) map lookup regardless of service count
sudo bpftool map show
# Shows eBPF maps used by Cilium for service routing
```

---

## 5. Advanced NetworkPolicy

### 5.1 Egress Policies

Control what external services pods can reach.

```yaml
# Allow pods to reach only specific external services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restricted-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: payment-service
  policyTypes:
    - Egress
  egress:
    # Allow DNS
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    # Allow HTTPS to payment processor
    - to:
        - ipBlock:
            cidr: 203.0.113.0/24     # Payment processor IP range
      ports:
        - protocol: TCP
          port: 443
    # Allow internal database
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

### 5.2 CIDR-Based Rules

```yaml
# Block access to cloud metadata services (security best practice)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-metadata
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    # Allow everything except cloud metadata endpoints
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 169.254.169.254/32   # AWS/GCP metadata
              - 100.100.100.200/32   # Azure metadata
```

### 5.3 Port Range Policies

```yaml
# Allow a range of ports (Kubernetes 1.25+)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ephemeral-ports
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: data-processor
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: data-store
      ports:
        - protocol: TCP
          port: 9000
          endPort: 9100           # Port range 9000-9100
```

### 5.4 DNS-Aware Policies (Cilium)

```yaml
# CiliumNetworkPolicy: DNS-based egress filtering
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: dns-egress-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: web-scraper
  egress:
    # Allow DNS resolution
    - toEndpoints:
        - matchLabels:
            io.kubernetes.pod.namespace: kube-system
            k8s-app: kube-dns
      toPorts:
        - ports:
            - port: "53"
              protocol: UDP
          rules:
            dns:
              - matchPattern: "*.example.com"
              - matchPattern: "api.github.com"
    # Allow HTTPS to resolved FQDNs only
    - toFQDNs:
        - matchPattern: "*.example.com"
        - matchName: "api.github.com"
      toPorts:
        - ports:
            - port: "443"
              protocol: TCP
```

---

## 6. Service Mesh Overview

### 6.1 What is a Service Mesh?

A service mesh adds a transparent infrastructure layer for service-to-service communication, providing mTLS encryption, traffic management, and observability without application code changes.

```
┌────────────────────────────────────────────────────────────┐
│  Without Service Mesh          With Service Mesh           │
│                                                            │
│  ┌─────┐    ┌─────┐          ┌─────┐    ┌─────┐          │
│  │App A│───▶│App B│          │App A│    │App B│          │
│  └─────┘    └─────┘          │     │    │     │          │
│                               │proxy│───▶│proxy│          │
│  Direct connection            │(sidecar) │(sidecar)       │
│  No encryption                └─────┘    └─────┘          │
│  No observability             mTLS, retries, tracing      │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Istio

Istio is the most feature-rich service mesh, using Envoy sidecars.

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.23.0
export PATH=$PWD/bin:$PATH

istioctl install --set profile=demo
kubectl label namespace production istio-injection=enabled
```

```yaml
# Istio VirtualService for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: app-routing
  namespace: production
spec:
  hosts:
    - app-service
  http:
    - match:
        - headers:
            x-user-type:
              exact: beta
      route:
        - destination:
            host: app-service
            subset: v2
    - route:
        - destination:
            host: app-service
            subset: v1
          weight: 90
        - destination:
            host: app-service
            subset: v2
          weight: 10
```

### 6.3 Linkerd

Linkerd is a lightweight service mesh focused on simplicity and performance.

```bash
# Install Linkerd
curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install | sh
linkerd install --crds | kubectl apply -f -
linkerd install | kubectl apply -f -
linkerd check

# Inject sidecar into a namespace
kubectl annotate namespace production linkerd.io/inject=enabled
```

### 6.4 Cilium Service Mesh

Cilium provides service mesh features without sidecars, using eBPF in the kernel.

```bash
# Enable Cilium service mesh features
helm upgrade cilium cilium/cilium \
  --namespace kube-system \
  --set envoy.enabled=true \
  --set loadBalancer.l7.backend=envoy
```

**Service mesh comparison:**

| Feature | Istio | Linkerd | Cilium |
|---------|-------|---------|--------|
| Proxy | Envoy sidecar | Rust micro-proxy sidecar | eBPF (no sidecar for L4) |
| mTLS | Yes | Yes | Yes (WireGuard or IPsec) |
| L7 policies | Full (HTTP, gRPC) | HTTP headers | HTTP, gRPC, Kafka, DNS |
| Resource overhead | High (Envoy per pod) | Low | Lowest |
| Complexity | High | Low | Medium |

---

## 7. Bandwidth and QoS

Kubernetes supports bandwidth limits via annotations (using the bandwidth CNI plugin).

```yaml
# Pod with bandwidth limits
apiVersion: v1
kind: Pod
metadata:
  name: bandwidth-limited
  annotations:
    kubernetes.io/ingress-bandwidth: "10M"    # 10 Mbps ingress
    kubernetes.io/egress-bandwidth: "5M"      # 5 Mbps egress
spec:
  containers:
    - name: app
      image: my-app:v1
```

```yaml
# Cilium bandwidth manager (more efficient than CNI bandwidth plugin)
# Enable in Helm values:
# bandwidthManager:
#   enabled: true

# Then use annotations on pods
apiVersion: v1
kind: Pod
metadata:
  name: rate-limited-pod
  annotations:
    kubernetes.io/egress-bandwidth: "50M"
spec:
  containers:
    - name: app
      image: my-app:v1
```

---

## 8. IPv4/IPv6 Dual-Stack

Kubernetes supports dual-stack networking where pods and services get both IPv4 and IPv6 addresses.

```yaml
# Service with dual-stack configuration
apiVersion: v1
kind: Service
metadata:
  name: dual-stack-service
spec:
  type: ClusterIP
  ipFamilyPolicy: PreferDualStack    # Or RequireDualStack
  ipFamilies:
    - IPv4
    - IPv6
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

```bash
# Check dual-stack service
kubectl get svc dual-stack-service -o yaml | grep -A5 clusterIPs
# clusterIPs:
# - 10.96.0.100
# - fd00::1:100

# Verify pod has both IPv4 and IPv6
kubectl exec my-pod -- ip addr show eth0
# inet 10.244.1.5/24
# inet6 fd00::1:5/128
```

---

## 9. Network Troubleshooting

### 9.1 Diagnostic Tools

```bash
# Deploy a network debug pod
kubectl run netdebug --image=nicolaka/netshoot --rm -it --restart=Never -- bash

# Inside the debug pod:
# DNS resolution
nslookup kubernetes.default.svc.cluster.local
dig +short my-service.production.svc.cluster.local

# Connectivity test
curl -v http://my-service.production.svc.cluster.local:8080/health
wget -qO- --timeout=2 http://10.96.0.1:443

# TCP connection test
nc -zv my-service.production 8080

# Trace route to a pod
traceroute 10.244.1.5

# Packet capture
tcpdump -i eth0 -n port 8080 -c 100

# Check MTU
ip link show eth0
ping -M do -s 1472 10.244.2.5
```

```bash
# Node-level diagnostics
# Check CNI plugin logs
kubectl logs -n kube-system -l k8s-app=calico-node --tail=50

# Check kube-proxy logs (if not using Cilium kube-proxy replacement)
kubectl logs -n kube-system -l k8s-app=kube-proxy --tail=50

# Inspect iptables rules on a node
ssh node01 "sudo iptables -t nat -L KUBE-SERVICES -n | head -20"

# Check conntrack table
ssh node01 "sudo conntrack -L | wc -l"
```

### 9.2 Common Issues

| Symptom | Likely Cause | Diagnostic Command |
|---------|-------------|-------------------|
| Pod cannot reach Service | kube-proxy/eBPF misconfiguration | `kubectl get endpoints <svc>` |
| DNS not resolving | CoreDNS pod down or misconfigured | `kubectl get pods -n kube-system -l k8s-app=kube-dns` |
| Cross-node pod communication fails | CNI overlay/routing misconfigured | `kubectl exec pod -- ping <cross-node-pod-ip>` |
| Pod gets no IP | IPAM exhaustion | `kubectl describe pod <pod>` (events) |
| NetworkPolicy not working | CNI does not support NetworkPolicy | Check CNI docs (e.g., flannel does NOT support NetworkPolicy) |
| Intermittent timeouts | MTU mismatch in overlay | `ping -M do -s 1400 <pod-ip>` from node |

---

## Exercises

### Exercise 1: Install and Verify Cilium on Minikube

Start a minikube cluster with Cilium as the CNI plugin (replacing kube-proxy). Run the Cilium connectivity test and enable Hubble to observe flows between pods.

<details><summary>Show Answer</summary>

```bash
# Start minikube without default CNI
minikube start --network-plugin=cni --cni=false --cpus=4 --memory=4096

# Install Cilium with kube-proxy replacement
cilium install --set kubeProxyReplacement=true

# Wait for Cilium to be ready
cilium status --wait

# Run the connectivity test
cilium connectivity test
# All tests should pass (this takes a few minutes)

# Enable Hubble
cilium hubble enable --ui

# Observe flows
hubble observe --all
# You will see flows from the connectivity test pods

# Port-forward Hubble UI
cilium hubble ui &

# Verify kube-proxy replacement
cilium status | grep KubeProxyReplacement
# KubeProxyReplacement: True
```

</details>

### Exercise 2: Calico vs Cilium NetworkPolicy

Create a namespace with two pods (`client` and `server`). Write a NetworkPolicy that allows the client to make only HTTP GET requests to the server on port 8080. Implement this using both standard Kubernetes NetworkPolicy and Cilium CiliumNetworkPolicy (showing the difference in L7 capability).

<details><summary>Show Answer</summary>

```yaml
# setup.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: netpol-test
---
apiVersion: v1
kind: Pod
metadata:
  name: server
  namespace: netpol-test
  labels:
    role: server
spec:
  containers:
    - name: server
      image: hashicorp/http-echo
      args: ["-text=hello", "-listen=:8080"]
      ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Pod
metadata:
  name: client
  namespace: netpol-test
  labels:
    role: client
spec:
  containers:
    - name: client
      image: nicolaka/netshoot
      command: ["sleep", "3600"]
```

```yaml
# Standard Kubernetes NetworkPolicy (L3/L4 only, cannot filter HTTP methods)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-client-to-server
  namespace: netpol-test
spec:
  podSelector:
    matchLabels:
      role: server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: client
      ports:
        - protocol: TCP
          port: 8080
```

```yaml
# Cilium CiliumNetworkPolicy (L7 filtering -- can restrict HTTP methods)
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: allow-client-get-only
  namespace: netpol-test
spec:
  endpointSelector:
    matchLabels:
      role: server
  ingress:
    - fromEndpoints:
        - matchLabels:
            role: client
      toPorts:
        - ports:
            - port: "8080"
              protocol: TCP
          rules:
            http:
              - method: "GET"    # Only GET is allowed; POST, PUT, DELETE are blocked
```

```bash
kubectl apply -f setup.yaml

# Test with standard NetworkPolicy (allows all HTTP methods on port 8080)
kubectl apply -f k8s-netpol.yaml
kubectl exec -n netpol-test client -- curl -s http://server:8080
# hello (works)
kubectl exec -n netpol-test client -- curl -s -X POST http://server:8080
# hello (also works -- L4 cannot distinguish HTTP methods)

# Remove standard policy, apply Cilium policy
kubectl delete networkpolicy allow-client-to-server -n netpol-test
kubectl apply -f cilium-netpol.yaml
kubectl exec -n netpol-test client -- curl -s http://server:8080
# hello (GET works)
kubectl exec -n netpol-test client -- curl -s -X POST http://server:8080
# Access denied (POST is blocked by L7 policy)
```

</details>

### Exercise 3: DNS-Aware Egress Policy

Using CiliumNetworkPolicy, create a policy that allows a pod labeled `app=scraper` to:
- Resolve DNS for `api.github.com` and `*.amazonaws.com` only
- Make HTTPS connections only to those resolved FQDNs
- Block all other egress traffic

<details><summary>Show Answer</summary>

```yaml
# dns-egress.yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: scraper-dns-egress
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: scraper
  egress:
    # Allow DNS resolution (only for allowed domains)
    - toEndpoints:
        - matchLabels:
            io.kubernetes.pod.namespace: kube-system
            k8s-app: kube-dns
      toPorts:
        - ports:
            - port: "53"
              protocol: UDP
          rules:
            dns:
              - matchName: "api.github.com"
              - matchPattern: "*.amazonaws.com"
    # Allow HTTPS to resolved FQDNs
    - toFQDNs:
        - matchName: "api.github.com"
        - matchPattern: "*.amazonaws.com"
      toPorts:
        - ports:
            - port: "443"
              protocol: TCP
```

```bash
kubectl apply -f dns-egress.yaml

# Deploy a test pod
kubectl run scraper --image=nicolaka/netshoot --rm -it \
  --restart=Never --labels="app=scraper" -n production -- bash

# Inside the pod:
# This should work
curl -s https://api.github.com/rate_limit
# {"resources": ...}

# This should work
curl -s https://s3.amazonaws.com
# Response from S3

# This should be blocked
curl -s --connect-timeout 3 https://google.com
# Connection timeout (blocked by policy)

# DNS for unauthorized domains should also fail
nslookup google.com
# NXDOMAIN or timeout
```

</details>

### Exercise 4: Network Troubleshooting

A pod in the `debug` namespace cannot reach a service called `backend` in the `app` namespace. The service exists and the backend pods are running. Describe a systematic troubleshooting process and write the diagnostic commands.

<details><summary>Show Answer</summary>

```bash
# Step 1: Verify the service and endpoints exist
kubectl get svc backend -n app
kubectl get endpoints backend -n app
# If endpoints are empty, check pod labels match service selector
kubectl get pods -n app --show-labels | grep backend

# Step 2: Check DNS resolution from the source pod
kubectl exec -n debug debug-pod -- nslookup backend.app.svc.cluster.local
# If DNS fails, check CoreDNS
kubectl get pods -n kube-system -l k8s-app=kube-dns
kubectl logs -n kube-system -l k8s-app=kube-dns --tail=20

# Step 3: Check direct pod-to-pod connectivity (bypass service)
BACKEND_POD_IP=$(kubectl get pod -n app -l app=backend -o jsonpath='{.items[0].status.podIP}')
kubectl exec -n debug debug-pod -- curl -v --connect-timeout 3 http://${BACKEND_POD_IP}:8080

# Step 4: Check NetworkPolicies
kubectl get networkpolicies -n app
kubectl get networkpolicies -n debug
# Look for default-deny policies or missing allow rules

kubectl describe networkpolicy -n app
# Check if the policy allows ingress from the debug namespace

# Step 5: If using Cilium, check policy verdicts
hubble observe --namespace app --to-pod app/backend --verdict DROPPED

# Step 6: Check if it is a cross-node issue
kubectl get pod debug-pod -n debug -o wide
kubectl get pod -n app -l app=backend -o wide
# If pods are on different nodes, check overlay networking

# Step 7: Check node-level networking
# SSH to the source node and test connectivity
ssh node01 "curl --connect-timeout 3 http://${BACKEND_POD_IP}:8080"

# Step 8: Fix -- if NetworkPolicy is the issue, add an allow rule
cat <<'EOF' | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-debug-to-backend
  namespace: app
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: debug
      ports:
        - protocol: TCP
          port: 8080
EOF
```

</details>

### Exercise 5: Dual-Stack Service Configuration

Configure a Kubernetes service with IPv4/IPv6 dual-stack. Create the service with `RequireDualStack` policy and verify that both addresses are assigned and reachable.

<details><summary>Show Answer</summary>

```yaml
# dual-stack-app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dual-stack-app
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dual-stack
  template:
    metadata:
      labels:
        app: dual-stack
    spec:
      containers:
        - name: app
          image: hashicorp/http-echo
          args: ["-text=dual-stack-works", "-listen=:8080"]
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: dual-stack-svc
  namespace: default
spec:
  type: ClusterIP
  ipFamilyPolicy: RequireDualStack
  ipFamilies:
    - IPv4
    - IPv6
  selector:
    app: dual-stack
  ports:
    - port: 80
      targetPort: 8080
```

```bash
kubectl apply -f dual-stack-app.yaml

# Verify dual-stack IPs are assigned
kubectl get svc dual-stack-svc -o jsonpath='{.spec.clusterIPs}'
# ["10.96.0.150","fd00::96:150"]

# Check both IP families
kubectl get svc dual-stack-svc -o jsonpath='{.spec.ipFamilies}'
# ["IPv4","IPv6"]

# Verify from a test pod
kubectl run test --image=nicolaka/netshoot --rm -it --restart=Never -- bash

# Inside the pod:
# IPv4 access
curl -4 http://dual-stack-svc/
# dual-stack-works

# IPv6 access
curl -6 http://dual-stack-svc/
# dual-stack-works

# DNS returns both A and AAAA records
dig dual-stack-svc.default.svc.cluster.local A +short
# 10.96.0.150
dig dual-stack-svc.default.svc.cluster.local AAAA +short
# fd00::96:150

# Verify pod has both addresses
kubectl exec dual-stack-app-xxxxx -- ip addr show eth0
# inet 10.244.1.50/24
# inet6 fd00:244:1::50/128
```

</details>

---

**Previous**: [Ingress and Gateway API](./07_Ingress_and_Gateway_API.md) | **Next**: [Helm and Kustomize](./09_Helm_and_Kustomize.md)
