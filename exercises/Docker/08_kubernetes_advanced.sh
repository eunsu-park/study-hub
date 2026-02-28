#!/bin/bash
# Exercises for Lesson 08: Kubernetes Advanced
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Deploy a Stateful Application with StatefulSet ===
# Problem: Experience the difference between Deployment and StatefulSet.
exercise_1() {
    echo "=== Exercise 1: Deploy a Stateful Application with StatefulSet ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- redis-headless-svc.yaml ---"
    cat << 'SOLUTION'
apiVersion: v1
kind: Service
metadata:
  name: redis-headless
spec:
  clusterIP: None
  # Headless Service (clusterIP: None):
  # Instead of a single ClusterIP, DNS returns the IP of each individual Pod.
  # This gives each Pod a stable DNS name: redis-0.redis-headless, redis-1.redis-headless, etc.
  # Required for StatefulSets — enables peer discovery for replication/clustering.
  selector:
    app: redis
  ports:
  - port: 6379
    name: redis
SOLUTION
    echo ""
    echo "--- redis-statefulset.yaml ---"
    cat << 'SOLUTION'
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: "redis-headless"    # Links to the headless Service for DNS
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: redis
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Apply both manifests
kubectl apply -f redis-headless-svc.yaml -f redis-statefulset.yaml

# Step 3: Observe ordered, stable Pod names
kubectl get pods -l app=redis -w
# NAME      READY   STATUS    RESTARTS   AGE
# redis-0   1/1     Running   0          10s
# redis-1   1/1     Running   0          8s
# redis-2   1/1     Running   0          5s
# Note: Pods are created in order (0, 1, 2) and each waits for the previous
# to be Running before starting. This is OrderedReady policy.

# Step 4: Set a key in redis-0
kubectl exec redis-0 -- redis-cli set mykey "hello"
# OK

kubectl exec redis-0 -- redis-cli get mykey
# "hello"

# Step 5: Delete redis-0 and observe recreation
kubectl delete pod redis-0
kubectl get pods -w
# redis-0   0/1     Terminating   0          2m
# redis-0   0/1     Pending       0          0s
# redis-0   1/1     Running       0          3s
# The Pod is recreated with the SAME name and ordinal (redis-0)
# This is the key StatefulSet guarantee: stable identity.

# Step 6: Check if the key persists
kubectl exec redis-0 -- redis-cli get mykey
# (nil) — the key is GONE!
# Without a PersistentVolumeClaim, data is stored in the container's
# ephemeral filesystem. When the Pod is recreated, the filesystem is fresh.
# This is why Exercise 2 adds volumeClaimTemplates.

# Cleanup
kubectl delete statefulset redis
kubectl delete svc redis-headless
SOLUTION
}

# === Exercise 2: Provision Persistent Storage with PVC ===
# Problem: Add PersistentVolumeClaims so StatefulSet data survives Pod recreation.
exercise_2() {
    echo "=== Exercise 2: Provision Persistent Storage with PVC ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- redis-statefulset-with-pvc.yaml ---"
    cat << 'SOLUTION'
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: "redis-headless"
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
          # Redis stores its dump.rdb and appendonly.aof files in /data
        command: ["redis-server", "--appendonly", "yes", "--dir", "/data"]
        # --appendonly yes: persist every write to disk
        # --dir /data: store persistence files in the mounted volume

  # volumeClaimTemplates: each replica gets its own PVC
  # redis-0 gets redis-data-redis-0, redis-1 gets redis-data-redis-1, etc.
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      # ReadWriteOnce: only one node can mount this volume at a time
      # This is fine because each Pod runs on one node
      resources:
        requests:
          storage: 1Gi
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Apply the headless service and updated StatefulSet
kubectl apply -f redis-headless-svc.yaml
kubectl apply -f redis-statefulset-with-pvc.yaml

# Step 2: Verify each Pod has its own PVC
kubectl get pvc
# NAME                   STATUS   VOLUME            CAPACITY   ACCESS MODES
# redis-data-redis-0     Bound    pvc-abc123...     1Gi        RWO
# redis-data-redis-1     Bound    pvc-def456...     1Gi        RWO
# redis-data-redis-2     Bound    pvc-ghi789...     1Gi        RWO
# Each Pod gets a dedicated PVC — no shared storage, no data corruption

# Step 3: Set a key in redis-0
kubectl exec redis-0 -- redis-cli set mykey "persistent-hello"
# OK

# Step 4: Delete the Pod
kubectl delete pod redis-0

# Wait for recreation
kubectl get pods -w
# redis-0 recreated with the same name and the same PVC

# Step 5: Verify the key persists!
kubectl exec redis-0 -- redis-cli get mykey
# "persistent-hello"
# Data survived because the PVC (and its underlying PV) was retained.

# Step 6: See the dynamically provisioned PersistentVolumes
kubectl get pv
# NAME             CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS
# pvc-abc123...    1Gi        RWO            Delete           Bound

# Step 7: Delete the StatefulSet — observe PVC retention
kubectl delete statefulset redis
kubectl get pvc
# PVCs are RETAINED even after the StatefulSet is deleted.
# This is a safety feature: you must explicitly delete PVCs.
# This prevents accidental data loss.

# To fully clean up (destroys data):
kubectl delete pvc -l app=redis
kubectl delete svc redis-headless
SOLUTION
}

# === Exercise 3: Expose a Service with Ingress ===
# Problem: Route external HTTP traffic to multiple Services via Ingress.
exercise_3() {
    echo "=== Exercise 3: Expose a Service with Ingress ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Setup ---"
    cat << 'SOLUTION'
# Step 1: Enable Nginx Ingress Controller on minikube
minikube addons enable ingress
# This deploys the nginx-ingress-controller in the ingress-nginx namespace
# Wait for it to be ready:
kubectl get pods -n ingress-nginx -w
SOLUTION
    echo ""
    echo "--- Deployments and Services ---"
    cat << 'SOLUTION'
# Step 2: Create two Deployments and ClusterIP Services
kubectl create deployment app-v1 --image=hashicorp/http-echo -- -text="Version 1"
kubectl expose deployment app-v1 --port=8080 --target-port=5678

kubectl create deployment app-v2 --image=hashicorp/http-echo -- -text="Version 2"
kubectl expose deployment app-v2 --port=8081 --target-port=5678
SOLUTION
    echo ""
    echo "--- ingress.yaml ---"
    cat << 'SOLUTION'
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    # Rewrite /v1 and /v2 prefixes to / before forwarding to the backend
    # Without this, the backend would receive /v1 as the path
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: myapp.local
    http:
      paths:
      # /v1 -> app-v1 Service on port 8080
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: app-v1
            port:
              number: 8080
      # /v2 -> app-v2 Service on port 8081
      - path: /v2
        pathType: Prefix
        backend:
          service:
            name: app-v2
            port:
              number: 8081
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Apply the Ingress
kubectl apply -f ingress.yaml

# Step 4: Get the Ingress IP
kubectl get ingress
# NAME          CLASS   HOSTS         ADDRESS         PORTS   AGE
# app-ingress   nginx   myapp.local   192.168.49.2    80      10s

# Step 5: Add to /etc/hosts
echo "192.168.49.2 myapp.local" | sudo tee -a /etc/hosts

# Step 6: Test routing
curl http://myapp.local/v1
# Version 1

curl http://myapp.local/v2
# Version 2

# The Ingress Controller routes based on the URL path:
# /v1 -> app-v1 -> Pod running http-echo "Version 1"
# /v2 -> app-v2 -> Pod running http-echo "Version 2"

# Cleanup
kubectl delete ingress app-ingress
kubectl delete deployment app-v1 app-v2
kubectl delete svc app-v1 app-v2
# Remove the /etc/hosts entry manually
SOLUTION
}

# === Exercise 4: Run a Batch Workload with Job and CronJob ===
# Problem: Use Job and CronJob for one-time and scheduled tasks.
exercise_4() {
    echo "=== Exercise 4: Run a Batch Workload with Job and CronJob ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- job.yaml ---"
    cat << 'SOLUTION'
apiVersion: batch/v1
kind: Job
metadata:
  name: my-job
spec:
  template:
    spec:
      containers:
      - name: batch
        image: alpine:latest
        command: ["sh", "-c", "echo 'Batch job complete' && sleep 2"]
        # The job runs, prints a message, waits 2 seconds, then exits 0
      restartPolicy: Never
      # Never: if the container fails, create a new Pod (backoffLimit controls retries)
      # OnFailure: restart the same Pod (use for transient errors)
  backoffLimit: 3
  # Retry up to 3 times on failure before marking the Job as Failed
SOLUTION
    echo ""
    echo "--- cronjob.yaml ---"
    cat << 'SOLUTION'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: my-cronjob
spec:
  schedule: "*/1 * * * *"     # Run every minute
  successfulJobsHistoryLimit: 3   # Keep last 3 successful Jobs
  failedJobsHistoryLimit: 1       # Keep last 1 failed Job
  # History limits prevent etcd from accumulating old Job objects
  # which would waste cluster resources over time
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: batch
            image: alpine:latest
            command: ["sh", "-c", "echo 'CronJob ran at $(date)' && sleep 1"]
          restartPolicy: OnFailure
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Step 1-2: Apply the Job and watch it complete
kubectl apply -f job.yaml
kubectl get jobs -w
# NAME     COMPLETIONS   DURATION   AGE
# my-job   0/1           2s         2s
# my-job   1/1           5s         5s

kubectl get pods
# NAME           READY   STATUS      RESTARTS   AGE
# my-job-abc12   0/1     Completed   0          10s

# Step 3: Read the Job logs
kubectl logs job/my-job
# Batch job complete

# Step 4: Apply the CronJob
kubectl apply -f cronjob.yaml

# Step 5: Wait 2 minutes and check
sleep 120
kubectl get jobs
# NAME                      COMPLETIONS   DURATION   AGE
# my-cronjob-28460123       1/1           3s         90s
# my-cronjob-28460124       1/1           3s         30s
# Two Jobs were created — one per minute

# Step 6: successfulJobsHistoryLimit is already set to 3
# After 4+ minutes, only the 3 most recent successful Jobs are retained
# Older Jobs are automatically cleaned up

kubectl get cronjob my-cronjob
# NAME          SCHEDULE      SUSPEND   ACTIVE   LAST SCHEDULE
# my-cronjob    */1 * * * *   False     0        30s

# Cleanup
kubectl delete job my-job
kubectl delete cronjob my-cronjob
SOLUTION
}

# === Exercise 5: Control Pod Placement with Node Affinity ===
# Problem: Use scheduling rules to influence which nodes workloads run on.
exercise_5() {
    echo "=== Exercise 5: Control Pod Placement with Node Affinity ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Label the minikube node
kubectl label node minikube tier=frontend
# Labels are key-value pairs attached to nodes (or any K8s object)
# The scheduler uses labels to match affinity rules

# Step 2: Create a Deployment with required node affinity
cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend-app
  template:
    metadata:
      labels:
        app: frontend-app
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: tier
                operator: In
                values:
                - frontend
              # REQUIRED: Pods MUST be placed on nodes with tier=frontend
              # If no matching node exists, the Pod stays Pending
      containers:
      - name: app
        image: nginx:alpine
EOF

# Step 3: Verify Pods are on the labeled node
kubectl get pods -o wide
# NAME                           READY   NODE      AGE
# frontend-app-abc123-xyz12      1/1     minikube  10s
# frontend-app-abc123-abc34      1/1     minikube  10s
# Both Pods are on 'minikube' because it has tier=frontend

# Step 4: Change the label (simulating a different node)
kubectl label node minikube tier=backend --overwrite

# Step 5: Create a Deployment with preferred (soft) affinity
cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend-app
  template:
    metadata:
      labels:
        app: backend-app
    spec:
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: tier
                operator: In
                values:
                - backend
              # PREFERRED: scheduler tries to place on tier=backend nodes
              # but will schedule elsewhere if no matching node is available
      containers:
      - name: app
        image: nginx:alpine
EOF

# Step 6: Observe scheduling behavior
kubectl get pods -o wide
# backend-app Pods are scheduled on minikube (which has tier=backend)
# If the label didn't match, Pods would STILL be scheduled (it's preferred, not required)

# Compare: the frontend-app Pods may now be Pending or still Running
# because they REQUIRE tier=frontend, which was changed to tier=backend
# Existing Pods are not evicted (IgnoredDuringExecution), but new Pods won't schedule

# Cleanup
kubectl delete deployment frontend-app backend-app
kubectl label node minikube tier-
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
