#!/bin/bash
# Exercise: Lesson 19 — Capstone: Production Cluster Design
# Complete the TODO items below.

# === Exercise 1: Cluster Architecture Design ===
# Design a production-ready cluster for an e-commerce platform.
# Hint: Consider HA, multi-AZ, node pools, and separation of concerns.

exercise_1() {
    echo "=== Exercise 1: Cluster Architecture ==="

    # TODO: Document your cluster design decisions as comments:
    #
    # Control Plane:
    #   - Number of control plane nodes: ___
    #   - Why this number? (hint: etcd quorum requires odd numbers)
    #   - Spread across availability zones: yes/no
    #
    # Node Pools:
    #   - Pool 1: "general" — for stateless workloads
    #     Instance type: ___  Nodes: ___  Autoscaling: min ___ max ___
    #   - Pool 2: "memory" — for caching and databases
    #     Instance type: ___  Nodes: ___  Autoscaling: min ___ max ___
    #   - Pool 3: "gpu" — for ML inference
    #     Instance type: ___  Nodes: ___  Autoscaling: min ___ max ___
    #
    # Networking:
    #   - CNI plugin choice: ___ (Calico/Cilium/Flannel) — why?
    #   - Pod CIDR: ___
    #   - Service CIDR: ___
    #   - Ingress controller: ___

    # TODO: Write the cluster creation command (EKS/GKE/AKS example)
    # Hint for EKS:
    #   eksctl create cluster \
    #     --name production \
    #     --region us-east-1 \
    #     --zones us-east-1a,us-east-1b,us-east-1c \
    #     --nodegroup-name general \
    #     --node-type m5.xlarge \
    #     --nodes-min 3 --nodes-max 10 \
    #     --managed --asg-access

}

# === Exercise 2: Namespace Strategy and RBAC ===
# Set up a multi-tenant namespace structure with proper access control.
# Hint: Combine namespaces, RBAC, network policies, and resource quotas.

exercise_2() {
    echo "=== Exercise 2: Namespace Strategy ==="

    # TODO: Create the namespace hierarchy:
    #   kubectl create namespace platform-system    # Platform team infrastructure
    #   kubectl create namespace monitoring          # Observability stack
    #   kubectl create namespace app-frontend        # Frontend team
    #   kubectl create namespace app-backend         # Backend team
    #   kubectl create namespace app-data            # Data team

    # TODO: Apply labels to each namespace for policy targeting
    # Hint: kubectl label namespace app-frontend team=frontend env=production

    # TODO: Create a ClusterRole "namespace-admin" with full access
    #       within a namespace (but not cluster-scoped resources)
    # Write the YAML as a heredoc

    # TODO: Create RoleBindings in each app namespace binding the
    #       respective team group to "namespace-admin"

    # TODO: Create a ResourceQuota for each app namespace:
    #   - app-frontend: 8 CPU, 16Gi memory, 20 pods
    #   - app-backend:  16 CPU, 32Gi memory, 40 pods
    #   - app-data:     32 CPU, 64Gi memory, 30 pods

}

# === Exercise 3: GitOps Deployment Pipeline ===
# Set up Argo CD for declarative, GitOps-based deployments.
# Hint: GitOps = Git is the single source of truth for cluster state.

exercise_3() {
    echo "=== Exercise 3: GitOps Pipeline ==="

    # TODO: Install Argo CD
    # kubectl create namespace argocd
    # kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

    # TODO: Create an Argo CD Application for the frontend service
    # Write the Application YAML as a heredoc with:
    #   - source: git repo URL, path to manifests, target revision
    #   - destination: cluster URL, namespace
    #   - syncPolicy: automated with prune and selfHeal enabled
    #   - health checks for Deployment and Ingress

    # TODO: Create an ApplicationSet for multi-environment deployment
    # Write the ApplicationSet YAML with:
    #   - Git generator reading from directories (envs/dev, envs/staging, envs/prod)
    #   - Each directory generates a separate Application
    #   - Progressive sync: dev -> staging -> prod

    # TODO: Configure Argo CD notifications for Slack
    # Hint: Use argocd-notifications ConfigMap

}

# === Exercise 4: Security Hardening Checklist ===
# Apply defense-in-depth security measures to the cluster.
# Hint: Security is layered — no single measure is sufficient.

exercise_4() {
    echo "=== Exercise 4: Security Hardening ==="

    # TODO: Implement each security layer and write the commands:

    # Layer 1: Pod Security Standards
    # - Apply "restricted" PSS to all app namespaces
    # kubectl label namespace app-frontend \
    #   pod-security.kubernetes.io/enforce=restricted

    # Layer 2: Network Policies
    # TODO: Write a default-deny-all policy for each app namespace
    # TODO: Write allow rules for known communication paths:
    #   frontend -> backend (port 8080)
    #   backend -> data (port 5432)
    #   monitoring -> all (port 9090, metrics scraping)

    # Layer 3: Secret Management
    # TODO: Install and configure External Secrets Operator
    # TODO: Write a SecretStore connecting to AWS Secrets Manager or Vault
    # TODO: Write an ExternalSecret that syncs a database password

    # Layer 4: Image Security
    # TODO: Create an admission policy that rejects images from untrusted registries
    # Allowed registries: gcr.io/my-project, my-registry.example.com

    # Layer 5: Audit Logging
    # TODO: Write an audit policy that logs all write operations to Secrets
    # and all exec operations into Pods

}

# === Exercise 5: Observability and SLO Dashboard ===
# Define SLIs, SLOs, and alerting for a production service.
# Hint: SLIs measure service quality; SLOs define acceptable thresholds.

exercise_5() {
    echo "=== Exercise 5: SLOs and Observability ==="

    # TODO: Define SLIs for the e-commerce platform:
    # SLI 1: Availability — proportion of successful HTTP requests
    #   PromQL: ___
    # SLI 2: Latency — proportion of requests faster than 200ms
    #   PromQL: ___
    # SLI 3: Error rate — proportion of non-5xx responses
    #   PromQL: ___

    # TODO: Define SLOs (write as comments):
    # SLO 1: 99.9% availability (monthly error budget: ~43 minutes)
    # SLO 2: 95% of requests under 200ms (p95 latency)
    # SLO 3: Error rate below 0.1%

    # TODO: Write PrometheusRule alerts for SLO burn rate
    # Hint: Multi-window burn rate alerting (Google SRE approach)
    # Fast burn (2% budget in 1 hour): fire immediately
    # Slow burn (5% budget in 6 hours): fire as warning

    # TODO: Write a Grafana dashboard JSON (as heredoc) showing:
    #   - Request rate (total, by status code)
    #   - Latency percentiles (p50, p90, p99)
    #   - Error budget remaining (%)
    #   - Pod resource usage (CPU, memory)
    #   - SLO compliance over 30 days

    # TODO: Set up PagerDuty/Slack integration for critical alerts
    # Write the Alertmanager receiver configuration as a heredoc

}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
