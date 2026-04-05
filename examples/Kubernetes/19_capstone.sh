#!/usr/bin/env bash
# 19_capstone — Production cluster bootstrap script
# Bootstraps a complete minikube cluster with essential components
# Usage: bash 19_capstone.sh

set -euo pipefail

CLUSTER_NAME="capstone-cluster"
K8S_VERSION="v1.29.0"

echo "============================================"
echo " Kubernetes Production Cluster Bootstrap"
echo "============================================"

# ============================================================
# Step 1: Create Cluster
# ============================================================
echo ""
echo "=== Step 1: Creating minikube cluster ==="
minikube start \
  --profile="$CLUSTER_NAME" \
  --kubernetes-version="$K8S_VERSION" \
  --cpus=4 \
  --memory=8192 \
  --disk-size=40g \
  --driver=docker \
  --addons=metrics-server,ingress,dashboard

kubectl config use-context "$CLUSTER_NAME"
echo "Cluster created. Waiting for readiness..."
kubectl wait --for=condition=Ready nodes --all --timeout=120s

# ============================================================
# Step 2: Namespaces and RBAC
# ============================================================
echo ""
echo "=== Step 2: Creating namespaces ==="
for NS in staging production monitoring logging; do
  kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
done

# Apply Pod Security Standards
kubectl label namespace production \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/warn=restricted --overwrite

kubectl label namespace staging \
  pod-security.kubernetes.io/enforce=baseline \
  pod-security.kubernetes.io/warn=restricted --overwrite

# ============================================================
# Step 3: Install Core Infrastructure (Helm)
# ============================================================
echo ""
echo "=== Step 3: Installing infrastructure via Helm ==="

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 2>/dev/null || true
helm repo update

# Prometheus + Grafana
helm upgrade --install kube-prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.retention=7d \
  --wait --timeout=300s 2>/dev/null || echo "Prometheus install skipped (may need more resources)"

# ============================================================
# Step 4: Deploy Sample Application
# ============================================================
echo ""
echo "=== Step 4: Deploying sample application ==="

kubectl apply -n staging -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: demo
  template:
    metadata:
      labels:
        app: demo
    spec:
      containers:
        - name: web
          image: hashicorp/http-echo:0.2.3
          args: ["-text=Hello from capstone cluster"]
          ports:
            - containerPort: 5678
          resources:
            requests:
              cpu: 50m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 64Mi
---
apiVersion: v1
kind: Service
metadata:
  name: demo-app
spec:
  selector:
    app: demo
  ports:
    - port: 80
      targetPort: 5678
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: demo-ingress
spec:
  ingressClassName: nginx
  rules:
    - host: demo.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: demo-app
                port:
                  number: 80
EOF

echo "Waiting for deployment..."
kubectl wait -n staging --for=condition=Available deployment/demo-app --timeout=120s

# ============================================================
# Step 5: Verify Everything
# ============================================================
echo ""
echo "=== Step 5: Cluster Verification ==="

echo ""
echo "--- Nodes ---"
kubectl get nodes -o wide

echo ""
echo "--- Namespaces ---"
kubectl get namespaces

echo ""
echo "--- Pods (all namespaces) ---"
kubectl get pods --all-namespaces --sort-by='.metadata.namespace'

echo ""
echo "--- Services ---"
kubectl get svc --all-namespaces | grep -v kube-system

echo ""
echo "--- Ingress ---"
kubectl get ingress --all-namespaces

echo ""
echo "--- Resource Usage ---"
kubectl top nodes 2>/dev/null || echo "(metrics not ready yet)"

echo ""
echo "============================================"
echo " Bootstrap Complete!"
echo " Cluster: $CLUSTER_NAME"
echo " Dashboard: minikube dashboard -p $CLUSTER_NAME"
echo " Demo app: curl -H 'Host: demo.local' \$(minikube ip -p $CLUSTER_NAME)"
echo "============================================"
