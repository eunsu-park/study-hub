#!/bin/bash
# Exercises for Lesson 09: Helm Package Management
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Create and Install Your First Helm Chart ===
# Problem: Scaffold a chart, customize it, and install it.
exercise_1() {
    echo "=== Exercise 1: Create and Install Your First Helm Chart ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a new chart
helm create myapp
# This generates a full chart scaffold:
# myapp/
# ├── Chart.yaml           # Name, version, description
# ├── values.yaml          # Default configuration
# ├── templates/           # K8s manifest templates
# │   ├── deployment.yaml
# │   ├── service.yaml
# │   ├── ingress.yaml
# │   ├── hpa.yaml
# │   ├── _helpers.tpl     # Template helper functions
# │   ├── NOTES.txt        # Post-install message
# │   └── tests/
# └── charts/              # Dependency charts

# Step 2: Explore the structure
tree myapp/
# Familiarize yourself with:
# - Chart.yaml: metadata (name: myapp, version: 0.1.0, appVersion: 1.16.0)
# - values.yaml: all configurable parameters
# - templates/: Go-template YAML files that reference .Values

# Step 3: Customize values.yaml
# Change replicaCount from 1 to 2 and image.tag to "alpine"
# In values.yaml:
#   replicaCount: 2
#   image:
#     repository: nginx
#     tag: "alpine"

# Step 4: Lint the chart for errors
helm lint myapp
# ==> Linting myapp
# [INFO] Chart.yaml: icon is recommended
# 1 chart(s) linted, 0 chart(s) failed
# Lint catches: missing required fields, invalid YAML, template errors

# Step 5: Render templates without installing (dry-run)
helm template myapp ./myapp
# Outputs the fully-rendered K8s manifests to stdout.
# Useful for reviewing exactly what will be applied to the cluster.
# Check: replicas should be 2, image should be nginx:alpine

# Step 6: Install the chart
helm install myapp-release ./myapp
# Creates a Helm "release" named myapp-release
# Renders templates with values and applies them to the cluster

# Step 7: Verify
helm list
# NAME            NAMESPACE  REVISION  STATUS    CHART        APP VERSION
# myapp-release   default    1         deployed  myapp-0.1.0  1.16.0

kubectl get pods
# myapp-release-myapp-abc123   1/1   Running
# myapp-release-myapp-def456   1/1   Running
# 2 replicas as configured

# Step 8: Uninstall
helm uninstall myapp-release
# Removes all K8s resources created by this release
SOLUTION
}

# === Exercise 2: Customize Deployments with Values ===
# Problem: Deploy the same chart to dev and prod with different configs.
exercise_2() {
    echo "=== Exercise 2: Customize Deployments with Values ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- values-dev.yaml ---"
    cat << 'SOLUTION'
replicaCount: 1
service:
  type: NodePort
  # NodePort: accessible externally via node IP + random high port
  # Good for development: easy access without Ingress setup
SOLUTION
    echo ""
    echo "--- values-prod.yaml ---"
    cat << 'SOLUTION'
replicaCount: 3
service:
  type: LoadBalancer
  # LoadBalancer: provisions a cloud LB (AWS ELB, GCP LB, etc.)
  # In production: single stable IP/DNS for external traffic
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Install to dev namespace with dev values
helm install myapp-dev ./myapp \
  -f values-dev.yaml \
  -n dev --create-namespace
# -f values-dev.yaml: overrides values.yaml defaults
# -n dev: install in the 'dev' namespace
# --create-namespace: create the namespace if it doesn't exist

# Install to prod namespace with prod values
helm install myapp-prod ./myapp \
  -f values-prod.yaml \
  -n prod --create-namespace

# Compare the two releases
helm list -A
# NAME        NAMESPACE  REVISION  STATUS    CHART
# myapp-dev   dev        1         deployed  myapp-0.1.0
# myapp-prod  prod       1         deployed  myapp-0.1.0
# Same chart, different configurations!

# Compare Services
kubectl get svc -A | grep myapp
# dev    myapp-dev-myapp    NodePort      10.96.x.x   <none>  80:31234/TCP
# prod   myapp-prod-myapp   LoadBalancer  10.96.x.x   <pending>  80:80/TCP
# dev: NodePort (1 replica, developer access)
# prod: LoadBalancer (3 replicas, production traffic)

# This pattern is the standard way to manage per-environment config:
# - One chart defines the application structure
# - values-<env>.yaml files customize each environment
# - CI/CD pipelines select the right values file

# Cleanup
helm uninstall myapp-dev -n dev
helm uninstall myapp-prod -n prod
kubectl delete namespace dev prod
SOLUTION
}

# === Exercise 3: Upgrade and Rollback a Release ===
# Problem: Practice the Helm release lifecycle.
exercise_3() {
    echo "=== Exercise 3: Upgrade and Rollback a Release ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Install with 1 replica
helm install myapp ./myapp --set replicaCount=1

# Step 2: Check status
helm status myapp
# NAME: myapp
# LAST DEPLOYED: ...
# NAMESPACE: default
# STATUS: deployed
# REVISION: 1

# Step 3: Upgrade to 3 replicas
helm upgrade myapp ./myapp --set replicaCount=3
# Helm performs a diff between the current and desired state,
# then applies only the changes (similar to kubectl apply).

# Step 4: Confirm
kubectl get pods
# 3 Pods running — upgrade was successful

# Step 5: View release history
helm history myapp
# REVISION  UPDATED                   STATUS      DESCRIPTION
# 1         2024-01-01 10:00:00       superseded  Install complete
# 2         2024-01-01 10:05:00       deployed    Upgrade complete
# Each revision is a snapshot — Helm stores the full rendered manifest
# so rollbacks are exact and instantaneous.

# Step 6: Roll back to revision 1 (1 replica)
helm rollback myapp 1
# Helm creates a NEW revision (3) that matches the state of revision 1.
# It does NOT delete revision 2 — history is preserved.

# Step 7: Verify
kubectl get pods
# 1 Pod running — back to the original state

helm history myapp
# REVISION  STATUS      DESCRIPTION
# 1         superseded  Install complete
# 2         superseded  Upgrade complete
# 3         deployed    Rollback to 1
# Revision 3 exists as the "rollback" — you can even undo the rollback.

# Key insights:
# - 'helm upgrade' is idempotent: running it with the same values is a no-op
# - 'helm rollback' creates a new revision (not destructive)
# - Use '--atomic' with upgrade to auto-rollback if the upgrade fails
#   helm upgrade myapp ./myapp --set replicaCount=3 --atomic

# Cleanup
helm uninstall myapp
SOLUTION
}

# === Exercise 4: Use a Helm Repository ===
# Problem: Install a chart from a public repository.
exercise_4() {
    echo "=== Exercise 4: Use a Helm Repository ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Add the Bitnami repository
helm repo add bitnami https://charts.bitnami.com/bitnami
# Bitnami maintains production-ready charts for 100+ applications
# (nginx, PostgreSQL, Redis, WordPress, Kafka, etc.)

# Step 2: Update repository index
helm repo update
# Fetches the latest chart versions from all added repositories
# Always run this before installing to get the latest versions

# Step 3: Search for the nginx chart
helm search repo bitnami/nginx
# NAME            CHART VERSION  APP VERSION  DESCRIPTION
# bitnami/nginx   15.x.x         1.25.x       NGINX Open Source for Kubernetes

# Step 4: Inspect default values
helm show values bitnami/nginx | head -50
# Shows all configurable parameters:
# replicaCount, image, service.type, resources, etc.
# Use this to understand what can be customized

# Step 5: Install with custom values
helm install my-nginx bitnami/nginx \
  --set replicaCount=2 \
  --set service.type=NodePort
# --set overrides specific values without creating a values file
# Useful for quick one-off customizations

# Step 6: Access the service
minikube service my-nginx --url
# http://192.168.49.2:31234
# Open in browser to see the nginx welcome page

# Verify the installation
helm list
kubectl get pods
kubectl get svc my-nginx

# Step 7: Uninstall
helm uninstall my-nginx
SOLUTION
}

# === Exercise 5: Add a Chart Dependency ===
# Problem: Compose a multi-component app using chart dependencies.
exercise_5() {
    echo "=== Exercise 5: Add a Chart Dependency ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Chart.yaml (with Redis dependency) ---"
    cat << 'SOLUTION'
apiVersion: v2
name: webapp
description: A web application with Redis cache
version: 0.1.0
appVersion: "1.0.0"
type: application

dependencies:
  - name: redis
    version: "19.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    # This tells Helm to download and include the Bitnami Redis chart
    # as a sub-chart within our webapp chart.
    # The Redis Pods, Service, etc. are managed automatically.
SOLUTION
    echo ""
    echo "--- values.yaml (Redis connection config) ---"
    cat << 'SOLUTION'
# ... (existing webapp values) ...

# Redis dependency configuration
# These values are passed to the Redis sub-chart
redis:
  architecture: standalone
  # standalone: single Redis instance (no replicas)
  # replication: master-replica setup
  auth:
    enabled: false
    # Disable auth for development simplicity
    # Enable in production: auth.enabled=true, auth.password=...

# Configure the webapp to connect to Redis
env:
  REDIS_HOST: "webapp-release-redis-master"
  # The Redis Service name follows the pattern:
  # <release-name>-redis-master
  # Helm templates use .Release.Name to construct this
  REDIS_PORT: "6379"
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Step 3: Download the dependency
helm dependency update webapp
# This downloads redis-19.x.x.tgz into webapp/charts/
# The Chart.lock file is also created to pin the exact version

# Step 4: Verify the dependency was downloaded
ls webapp/charts/
# redis-19.x.x.tgz

# Step 5: Install the combined chart
helm install webapp-release ./webapp

# Step 6: Verify both webapp and Redis pods are running
kubectl get pods
# NAME                                      READY   STATUS    RESTARTS   AGE
# webapp-release-webapp-abc123-xyz12        1/1     Running   0          30s
# webapp-release-redis-master-0             1/1     Running   0          30s
# The Redis Pod is managed by the sub-chart — no manual configuration needed

# Test Redis connectivity from the webapp Pod
kubectl exec deployment/webapp-release-webapp -- \
  sh -c "apk add --no-cache redis && redis-cli -h webapp-release-redis-master ping"
# PONG — the webapp can reach Redis using the Service DNS name

# Key benefits of chart dependencies:
# 1. Reuse: Don't reinvent Redis deployment — use Bitnami's battle-tested chart
# 2. Version pinning: Chart.lock ensures reproducible deployments
# 3. Single install: 'helm install' deploys the entire stack
# 4. Unified config: All settings in one values.yaml file
# 5. Lifecycle: 'helm uninstall' removes everything cleanly

# Cleanup
helm uninstall webapp-release
SOLUTION
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
