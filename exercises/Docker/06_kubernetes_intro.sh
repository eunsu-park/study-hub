#!/bin/bash
# Exercises for Lesson 06: Kubernetes Introduction
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Deploy Your First Pod and Deployment ===
# Problem: Get hands-on with the most fundamental K8s resources.
exercise_1() {
    echo "=== Exercise 1: Deploy Your First Pod and Deployment ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Start a local Kubernetes cluster
minikube start
# minikube provisions a single-node K8s cluster using Docker or a VM.
# This takes 1-2 minutes on first run (downloads K8s images).

# Step 2: Run a Pod imperatively (without YAML)
kubectl run nginx-test --image=nginx:alpine
# Creates a bare Pod — NOT managed by a Deployment.
# If this Pod crashes or is deleted, K8s will NOT recreate it.

# Step 3: Watch until the Pod is Running
kubectl get pods -w
# NAME         READY   STATUS              RESTARTS   AGE
# nginx-test   0/1     ContainerCreating   0          2s
# nginx-test   1/1     Running             0          5s
# Press Ctrl+C to stop watching

# Step 4: Find which node the Pod was scheduled on
kubectl describe pod nginx-test | grep Node:
# Node: minikube/192.168.49.2
# The scheduler placed it on the only available node (minikube)

# Step 5: View logs
kubectl logs nginx-test
# Shows Nginx startup messages

# Step 6: Delete the Pod — observe it is NOT recreated
kubectl delete pod nginx-test
kubectl get pods
# No resources found — bare Pods are not self-healing

# Step 7: Create a Deployment with 2 replicas
kubectl create deployment web --image=nginx:alpine --replicas=2
# A Deployment manages a ReplicaSet, which maintains the desired Pod count.
# If any Pod dies, the ReplicaSet creates a replacement.

# Step 8: Delete one Pod and observe automatic replacement
kubectl get pods
# NAME                   READY   STATUS    RESTARTS   AGE
# web-abc123-xyz12       1/1     Running   0          10s
# web-abc123-abc34       1/1     Running   0          10s

kubectl delete pod web-abc123-xyz12
kubectl get pods
# A NEW Pod appears immediately to maintain replicas=2
# This is the key difference: Deployments self-heal, bare Pods do not.

# Cleanup
kubectl delete deployment web
SOLUTION
}

# === Exercise 2: Expose a Deployment with a Service ===
# Problem: Practice ClusterIP and NodePort Service types.
exercise_2() {
    echo "=== Exercise 2: Expose a Deployment with a Service ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a Deployment with 3 replicas
kubectl create deployment hello --image=nginxdemos/hello --replicas=3
# nginxdemos/hello serves a page showing the Pod hostname,
# making it easy to see load balancing in action.

# Step 2: Expose as a ClusterIP Service (internal only)
kubectl expose deployment hello --port=80 --type=ClusterIP
# ClusterIP: accessible only from within the cluster
# Pods and other Services can reach it, but external clients cannot

# Step 3: Verify the Service
kubectl get svc hello
# NAME    TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
# hello   ClusterIP   10.96.45.123   <none>        80/TCP    5s

# Step 4: Access via port-forwarding (bridges local machine to cluster)
kubectl port-forward svc/hello 8080:80
# Open http://localhost:8080 in your browser
# Each refresh may show a different hostname — the Service load-balances
# across all 3 Pods behind it.
# Press Ctrl+C to stop

# Step 5: Refresh multiple times — note the rotating hostname
# The hostname changes because the Service distributes requests
# across Pods using round-robin (iptables-based load balancing).

# Step 6: Change to NodePort (external access via node IP + high port)
kubectl patch svc hello -p '{"spec":{"type":"NodePort"}}'
# NodePort allocates a port in the 30000-32767 range on every node

kubectl get svc hello
# NAME    TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
# hello   NodePort   10.96.45.123   <none>        80:31234/TCP   1m

# Step 7: Access via minikube
minikube service hello --url
# http://192.168.49.2:31234
# This URL is accessible from your host machine

# Cleanup
kubectl delete deployment hello
kubectl delete svc hello
SOLUTION
}

# === Exercise 3: Rolling Update and Rollback ===
# Problem: Practice zero-downtime deployments and rollbacks.
exercise_3() {
    echo "=== Exercise 3: Rolling Update and Rollback ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a Deployment with an older image
kubectl create deployment app --image=nginxdemos/hello:plain-text
# plain-text version serves a simple text response with hostname

# Step 2: Watch the initial rollout
kubectl rollout status deployment/app
# deployment "app" successfully rolled out

# Step 3: Update to a new image
kubectl set image deployment/app hello=nginx:1.25
# This triggers a rolling update:
# - K8s creates new Pods with nginx:1.25
# - Once each new Pod is Ready, an old Pod is terminated
# - At no point are all Pods down (zero-downtime)

# Step 4: Watch the rolling update in real time
kubectl get pods -w
# NAME                  READY   STATUS              RESTARTS   AGE
# app-old-abc123        1/1     Running             0          2m
# app-new-def456        0/1     ContainerCreating   0          1s
# app-new-def456        1/1     Running             0          3s
# app-old-abc123        1/1     Terminating         0          2m
# The new Pod starts BEFORE the old one is terminated — no downtime

# Step 5: Check rollout history
kubectl rollout history deployment/app
# REVISION  CHANGE-CAUSE
# 1         <none>
# 2         <none>
# Each revision represents a deployment version you can roll back to

# Step 6: Roll back to the previous version
kubectl rollout undo deployment/app
# This creates revision 3, which is identical to revision 1
# The rollback is itself a rolling update — zero downtime

# Step 7: Verify the rollback
kubectl rollout status deployment/app
# deployment "app" successfully rolled out

kubectl get pods -o jsonpath='{.items[*].spec.containers[*].image}'
# nginxdemos/hello:plain-text — back to the original image

# Key insights:
# - Rolling updates replace Pods gradually (controlled by maxSurge/maxUnavailable)
# - Rollbacks are instant because K8s keeps old ReplicaSets around
# - Use 'kubectl rollout history' to see all available revisions
# - '--to-revision=N' lets you roll back to any specific version

# Cleanup
kubectl delete deployment app
SOLUTION
}

# === Exercise 4: ConfigMap and Secret ===
# Problem: Store config and sensitive data using K8s-native mechanisms.
exercise_4() {
    echo "=== Exercise 4: ConfigMap and Secret ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- pod-with-config.yaml ---"
    cat << 'SOLUTION'
# Step 1: Create a ConfigMap with two keys
kubectl create configmap app-config \
  --from-literal=LOG_LEVEL=info \
  --from-literal=APP_PORT=8080

# Step 2: Verify
kubectl get configmap app-config -o yaml
# data:
#   APP_PORT: "8080"
#   LOG_LEVEL: "info"

# Step 3: Create a Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=supersecret
# Secrets are base64-encoded (NOT encrypted) by default.
# Use RBAC to restrict who can read Secrets.

# Step 4: Create a Pod that uses both ConfigMap and Secret
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: config-demo
spec:
  containers:
  - name: app
    image: alpine:latest
    command: ["sleep", "3600"]
    # ConfigMap as environment variables
    envFrom:
    - configMapRef:
        name: app-config
        # All keys (LOG_LEVEL, APP_PORT) become env vars
    # Secret as volume mount
    volumeMounts:
    - name: secret-volume
      mountPath: /secrets
      readOnly: true
      # Secret keys become files: /secrets/username, /secrets/password
  volumes:
  - name: secret-volume
    secret:
      secretName: db-secret
EOF

# Step 5: Verify the values inside the Pod
kubectl exec config-demo -- env | grep -E "LOG_LEVEL|APP_PORT"
# LOG_LEVEL=info
# APP_PORT=8080

kubectl exec config-demo -- cat /secrets/password
# supersecret

# Step 6: Update the ConfigMap
kubectl patch configmap app-config --type merge -p '{"data":{"LOG_LEVEL":"debug"}}'

# Step 7: Observe auto-update behavior
# Volume-mounted ConfigMaps update automatically (~1-2 minutes delay)
# Environment variables from ConfigMaps do NOT auto-update (requires Pod restart)
# This is a key distinction when choosing how to consume ConfigMaps.

# To force a Pod restart after ConfigMap change:
kubectl delete pod config-demo
# Then recreate it — env vars will reflect the new values

# Cleanup
kubectl delete pod config-demo
kubectl delete configmap app-config
kubectl delete secret db-secret
SOLUTION
}

# === Exercise 5: Namespaces and Multi-Environment Setup ===
# Problem: Use Namespaces to simulate isolated environments.
exercise_5() {
    echo "=== Exercise 5: Namespaces and Multi-Environment Setup ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create two namespaces
kubectl create namespace dev
kubectl create namespace prod
# Namespaces provide logical isolation within a single cluster.
# Same resource names can exist in different namespaces without conflict.

# Step 2: Deploy the same application to both namespaces
kubectl create deployment web --image=nginx:alpine -n dev
kubectl create deployment web --image=nginx:alpine -n prod
# Both are named 'web' but exist in separate namespaces — no conflict

# Step 3: Scale them differently
kubectl scale deployment web --replicas=1 -n dev
kubectl scale deployment web --replicas=3 -n prod
# Dev: 1 replica (save resources for development)
# Prod: 3 replicas (high availability for production)

# Step 4: List Pods across ALL namespaces
kubectl get pods --all-namespaces
# NAMESPACE     NAME                   READY   STATUS    RESTARTS   AGE
# dev           web-abc123-xyz12       1/1     Running   0          30s
# prod          web-def456-abc34       1/1     Running   0          25s
# prod          web-def456-def56       1/1     Running   0          25s
# prod          web-def456-ghi78       1/1     Running   0          25s
# kube-system   coredns-...            1/1     Running   0          5m
# ...

# Step 5: Switch default namespace to dev
kubectl config set-context --current --namespace=dev
# Now all kubectl commands without -n default to 'dev'

# Step 6: Verify namespace isolation
kubectl get pods
# Only shows the 1 Pod in 'dev' namespace
# Pods in 'prod' and 'kube-system' are hidden from this view

kubectl get pods -n prod
# Shows the 3 Pods in 'prod' (explicit -n override still works)

# Step 7: Clean up both namespaces
kubectl delete namespace dev prod
# Deleting a namespace removes ALL resources within it:
# Pods, Deployments, Services, ConfigMaps, Secrets, etc.
# This is a convenient way to clean up entire environments.

# Reset default namespace
kubectl config set-context --current --namespace=default
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
