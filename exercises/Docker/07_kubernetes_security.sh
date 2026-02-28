#!/bin/bash
# Exercises for Lesson 07: Kubernetes Security
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Create RBAC for a Development Team ===
# Problem: Apply least privilege using Roles and RoleBindings.
exercise_1() {
    echo "=== Exercise 1: Create RBAC for a Development Team ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Setup ---"
    cat << 'SOLUTION'
# Create the dev namespace
kubectl create namespace dev

# Create a ServiceAccount for the developer
kubectl create serviceaccount developer -n dev
SOLUTION
    echo ""
    echo "--- role.yaml ---"
    cat << 'SOLUTION'
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev
  name: developer-role
rules:
# Allow managing Pods — developers need to create, debug, and restart Pods
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch", "create", "delete"]

# Allow managing Deployments — needed for deploying application updates
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "delete"]

# Note: Secrets are intentionally NOT included.
# Developers should not have direct access to production credentials.
# Use a CI/CD pipeline or external secrets manager instead.
SOLUTION
    echo ""
    echo "--- rolebinding.yaml ---"
    cat << 'SOLUTION'
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: dev
subjects:
- kind: ServiceAccount
  name: developer
  namespace: dev
  # Bind to the ServiceAccount, not a user — ServiceAccounts are managed
  # by K8s and can be used by CI/CD pipelines and automation tools
roleRef:
  kind: Role
  name: developer-role
  apiGroup: rbac.authorization.k8s.io
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Apply both manifests
kubectl apply -f role.yaml -f rolebinding.yaml

# Test: Can the developer list Pods? (should be YES)
kubectl auth can-i list pods \
  --as=system:serviceaccount:dev:developer -n dev
# yes

# Test: Can the developer create Deployments? (should be YES)
kubectl auth can-i create deployments \
  --as=system:serviceaccount:dev:developer -n dev
# yes

# Test: Can the developer access Secrets? (should be NO)
kubectl auth can-i get secrets \
  --as=system:serviceaccount:dev:developer -n dev
# no
# This confirms the principle of least privilege is enforced.

# Test: Can the developer access resources in other namespaces? (should be NO)
kubectl auth can-i list pods \
  --as=system:serviceaccount:dev:developer -n default
# no
# Roles are namespace-scoped — the developer has NO access outside 'dev'

# Cleanup
kubectl delete namespace dev
SOLUTION
}

# === Exercise 2: Enforce Network Isolation with NetworkPolicy ===
# Problem: Allow only intended traffic flows between services.
exercise_2() {
    echo "=== Exercise 2: Enforce Network Isolation with NetworkPolicy ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Setup: Create namespace and deployments ---"
    cat << 'SOLUTION'
kubectl create namespace netpol-test
kubectl create deployment frontend --image=nginx:alpine -n netpol-test
kubectl create deployment backend --image=nginx:alpine -n netpol-test
kubectl create deployment database --image=nginx:alpine -n netpol-test

kubectl expose deployment frontend --port=80 -n netpol-test
kubectl expose deployment backend --port=8080 --target-port=80 -n netpol-test
kubectl expose deployment database --port=5432 --target-port=80 -n netpol-test

# Label the pods for NetworkPolicy selectors
kubectl label deployment frontend app=frontend -n netpol-test --overwrite
kubectl label deployment backend app=backend -n netpol-test --overwrite
kubectl label deployment database app=database -n netpol-test --overwrite
SOLUTION
    echo ""
    echo "--- default-deny.yaml ---"
    cat << 'SOLUTION'
# Start with default-deny: block ALL ingress traffic in the namespace.
# This forces every service to explicitly declare its allowed sources.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: netpol-test
spec:
  podSelector: {}      # Empty selector = applies to ALL pods in namespace
  policyTypes:
    - Ingress          # Block all inbound traffic by default
SOLUTION
    echo ""
    echo "--- allow-frontend-to-backend.yaml ---"
    cat << 'SOLUTION'
# Allow frontend pods to reach backend on port 8080
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: netpol-test
spec:
  podSelector:
    matchLabels:
      app: backend            # This policy applies to backend pods
  policyTypes:
    - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend       # Only frontend can send traffic to backend
    ports:
    - protocol: TCP
      port: 8080              # Restrict to the exact port
SOLUTION
    echo ""
    echo "--- allow-backend-to-database.yaml ---"
    cat << 'SOLUTION'
# Allow backend pods to reach database on port 5432
# Frontend is NOT listed — it cannot reach the database directly.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-database
  namespace: netpol-test
spec:
  podSelector:
    matchLabels:
      app: database           # This policy applies to database pods
  policyTypes:
    - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend        # Only backend can reach the database
    ports:
    - protocol: TCP
      port: 5432
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Apply all policies
kubectl apply -f default-deny.yaml
kubectl apply -f allow-frontend-to-backend.yaml
kubectl apply -f allow-backend-to-database.yaml

# Test: frontend -> backend (ALLOWED)
kubectl exec -n netpol-test deployment/frontend -- \
  wget -qO- --timeout=2 http://backend:8080
# Should succeed (HTML response from nginx)

# Test: backend -> database (ALLOWED)
kubectl exec -n netpol-test deployment/backend -- \
  wget -qO- --timeout=2 http://database:5432
# Should succeed (some response, even if not a real DB)

# Test: frontend -> database (BLOCKED)
kubectl exec -n netpol-test deployment/frontend -- \
  wget -qO- --timeout=2 http://database:5432
# Should timeout/fail — frontend is not allowed to reach database
# This confirms network isolation is working correctly

# Note: NetworkPolicy requires a CNI plugin that supports it
# (Calico, Cilium, Weave Net). The default kubenet does NOT enforce policies.

# Cleanup
kubectl delete namespace netpol-test
SOLUTION
}

# === Exercise 3: Harden a Pod with SecurityContext ===
# Problem: Apply Pod-level hardening with least privilege.
exercise_3() {
    echo "=== Exercise 3: Harden a Pod with SecurityContext ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- secure-pod.yaml ---"
    cat << 'SOLUTION'
apiVersion: v1
kind: Pod
metadata:
  name: secure-nginx
spec:
  securityContext:
    runAsNonRoot: true         # Reject any container that tries to run as UID 0
    runAsUser: 1000            # Run all containers as UID 1000
    fsGroup: 2000              # Volumes are owned by GID 2000

  containers:
  - name: app
    image: nginx:alpine
    securityContext:
      allowPrivilegeEscalation: false   # Block setuid/setgid binaries
      readOnlyRootFilesystem: true      # Immutable filesystem
      capabilities:
        drop: ["ALL"]                   # Drop all Linux capabilities

    # nginx needs to write to several directories.
    # With readOnlyRootFilesystem: true, we must provide writable emptyDirs.
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: nginx-cache
      mountPath: /var/cache/nginx
    - name: nginx-run
      mountPath: /var/run
      # /var/run: nginx writes its PID file here
      # /var/cache/nginx: nginx caches proxied responses here
      # /tmp: general temp directory

  volumes:
  - name: tmp
    emptyDir: {}
  - name: nginx-cache
    emptyDir: {}
  - name: nginx-run
    emptyDir: {}
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Apply the manifest
kubectl apply -f secure-pod.yaml

# Check if the Pod starts successfully
kubectl get pod secure-nginx
# STATUS should be Running (after adding the volume mounts)

# Verify: running as non-root
kubectl exec secure-nginx -- whoami
# Output depends on the user mapping in the image
# id command shows UID 1000

kubectl exec secure-nginx -- id
# uid=1000 gid=0(root) groups=2000

# Verify: read-only root filesystem
kubectl exec secure-nginx -- sh -c "echo test > /test.txt" 2>&1
# sh: can't create /test.txt: Read-only file system
# Writing to / fails — the filesystem is immutable

# Verify: writable emptyDir volumes
kubectl exec secure-nginx -- sh -c "echo test > /tmp/test.txt && cat /tmp/test.txt"
# test
# /tmp is writable because it's backed by an emptyDir volume

# Cleanup
kubectl delete pod secure-nginx
SOLUTION
}

# === Exercise 4: Manage Secrets Securely ===
# Problem: Practice secure Secret creation and consumption patterns.
exercise_4() {
    echo "=== Exercise 4: Manage Secrets Securely ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a Secret from literal values
kubectl create secret generic app-credentials \
  --from-literal=DB_USER=admin \
  --from-literal=DB_PASS=s3cr3t

# Step 2: Inspect the Secret
kubectl get secret app-credentials -o yaml
# data:
#   DB_PASS: czNjcjN0     (base64 of "s3cr3t")
#   DB_USER: YWRtaW4=     (base64 of "admin")
# Note: base64 is encoding, NOT encryption. Anyone with kubectl access
# can decode these values.

# Step 3: Decode a value
kubectl get secret app-credentials -o jsonpath='{.data.DB_PASS}' | base64 -d
# Output: s3cr3t

# Step 4: Create a Pod that mounts the Secret as a VOLUME
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: secret-vol-demo
spec:
  containers:
  - name: app
    image: alpine:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: secrets
      mountPath: /run/secrets/app
      readOnly: true
      # Each Secret key becomes a file:
      # /run/secrets/app/DB_USER contains "admin"
      # /run/secrets/app/DB_PASS contains "s3cr3t"
  volumes:
  - name: secrets
    secret:
      secretName: app-credentials
EOF

# Step 5: Read the mounted files
kubectl exec secret-vol-demo -- cat /run/secrets/app/DB_PASS
# Output: s3cr3t

kubectl exec secret-vol-demo -- ls -la /run/secrets/app/
# lrwxrwxrwx  DB_PASS -> ..data/DB_PASS
# lrwxrwxrwx  DB_USER -> ..data/DB_USER
# Files are symlinked — K8s can atomically update them

# Step 6: Why volume mounts are more secure than environment variables
#
# Environment variables:
# - Visible in 'docker inspect' and 'kubectl describe pod'
# - Inherited by child processes (potential leak via fork/exec)
# - Logged by many frameworks (e.g., Node.js process.env dumps)
# - Cannot be updated without restarting the Pod
#
# Volume mounts:
# - Files are not visible in 'kubectl describe pod'
# - Not inherited by child processes automatically
# - Auto-update when the Secret changes (~1-2 min delay)
# - Can set file permissions (mode: 0400 for owner-read-only)
#
# Best practice: Always prefer volume mounts for sensitive data.

# Cleanup
kubectl delete pod secret-vol-demo
kubectl delete secret app-credentials
SOLUTION
}

# === Exercise 5: Apply Pod Security Standards ===
# Problem: Enforce workload security policies at the namespace level.
exercise_5() {
    echo "=== Exercise 5: Apply Pod Security Standards ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Label a namespace to enforce the 'restricted' standard
kubectl create namespace secure-ns
kubectl label namespace secure-ns \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/enforce-version=latest
# 'restricted' is the strictest standard — it requires:
# - Non-root user
# - No privilege escalation
# - All capabilities dropped
# - Read-only root filesystem (recommended)
# - Seccomp profile set

# Step 2: Try to create a Pod that runs as root
cat << 'EOF' | kubectl apply -n secure-ns -f - 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: insecure-pod
spec:
  containers:
  - name: app
    image: nginx:latest
    # No securityContext — defaults to root
EOF
# Error: pods "insecure-pod" is forbidden:
#   violates PodSecurity "restricted:latest":
#   allowPrivilegeEscalation != false,
#   unrestricted capabilities,
#   runAsNonRoot != true,
#   seccompProfile not set
# The Pod is REJECTED before it even starts!

# Step 3: Fix the Pod to comply with 'restricted' standard
cat << 'EOF' | kubectl apply -n secure-ns -f -
apiVersion: v1
kind: Pod
metadata:
  name: compliant-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault       # Required by 'restricted' standard
  containers:
  - name: app
    image: nginx:alpine
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /var/cache/nginx
    - name: run
      mountPath: /var/run
  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
  - name: run
    emptyDir: {}
EOF
# pod/compliant-pod created — the Pod passes all restrictions

# Step 4: Verify it's running
kubectl get pod compliant-pod -n secure-ns
# STATUS: Running

# Step 5: Change to 'baseline' and test
kubectl label namespace secure-ns \
  pod-security.kubernetes.io/enforce=baseline --overwrite
# 'baseline' is less strict — it forbids:
# - hostNetwork, hostPID, hostIPC
# - Privileged containers
# - Certain dangerous capabilities
# But it ALLOWS:
# - Running as root
# - Writable root filesystem

# Now the insecure Pod would be accepted under 'baseline':
cat << 'EOF' | kubectl apply -n secure-ns -f - 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: baseline-pod
spec:
  containers:
  - name: app
    image: nginx:latest
    # No securityContext — defaults to root (allowed under baseline)
EOF
# pod/baseline-pod created — baseline is more permissive

# Cleanup
kubectl delete namespace secure-ns
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
