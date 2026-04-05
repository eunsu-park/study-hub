#!/usr/bin/env bash
# 09_helm_kustomize — Helm chart creation and Kustomize overlay
# Requires: helm, kustomize (or kubectl with kustomize support)

set -euo pipefail

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
echo "Working in $WORKDIR"

# ============================================================
# Part 1: Helm Chart Creation
# ============================================================
echo "=== Creating Helm Chart ==="
cd "$WORKDIR"
helm create myapp

echo ""
echo "=== Chart Structure ==="
find myapp -type f | sort

# Customize values.yaml
cat > myapp/values.yaml << 'VALS'
replicaCount: 2
image:
  repository: nginx
  tag: "1.25-alpine"
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 80
resources:
  limits:
    cpu: 200m
    memory: 128Mi
  requests:
    cpu: 50m
    memory: 64Mi
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
VALS

echo ""
echo "=== Linting Chart ==="
helm lint myapp

echo ""
echo "=== Dry-Run Install ==="
helm install myapp-release myapp --dry-run --debug 2>&1 | head -40

echo ""
echo "=== Packaging Chart ==="
helm package myapp
ls -la myapp-*.tgz

# ============================================================
# Part 2: Kustomize Overlays
# ============================================================
echo ""
echo "=== Setting Up Kustomize ==="

mkdir -p kustom/base kustom/overlays/dev kustom/overlays/prod

# Base deployment
cat > kustom/base/deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
        - name: web
          image: nginx:1.25-alpine
          ports:
            - containerPort: 80
EOF

cat > kustom/base/service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-app
spec:
  selector:
    app: web-app
  ports:
    - port: 80
EOF

cat > kustom/base/kustomization.yaml << 'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
commonLabels:
  managed-by: kustomize
EOF

# Dev overlay: fewer replicas, dev namespace
cat > kustom/overlays/dev/kustomization.yaml << 'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: dev
namePrefix: dev-
resources:
  - ../../base
patches:
  - target:
      kind: Deployment
      name: web-app
    patch: |
      - op: replace
        path: /spec/replicas
        value: 1
EOF

# Prod overlay: more replicas, resource limits
cat > kustom/overlays/prod/kustomization.yaml << 'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
namePrefix: prod-
resources:
  - ../../base
patches:
  - target:
      kind: Deployment
      name: web-app
    patch: |
      - op: replace
        path: /spec/replicas
        value: 5
EOF

echo ""
echo "=== Kustomize Build: Dev ==="
kubectl kustomize kustom/overlays/dev

echo ""
echo "=== Kustomize Build: Prod ==="
kubectl kustomize kustom/overlays/prod

echo ""
echo "=== Done ==="
