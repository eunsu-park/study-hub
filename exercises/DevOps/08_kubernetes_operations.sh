#!/bin/bash
# Exercises for Lesson 08: Kubernetes Operations
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Deployment Manifest ===
# Problem: Write a production-ready Kubernetes Deployment with health
# probes, resource limits, anti-affinity, and rolling update strategy.
exercise_1() {
    echo "=== Exercise 1: Deployment Manifest ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# deployment.yaml — Production-ready Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-api
  namespace: production
  labels:
    app: order-api
    version: v1.2.0
spec:
  replicas: 3
  revisionHistoryLimit: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1           # Create 1 extra pod during update
      maxUnavailable: 0     # Never reduce below desired count
  selector:
    matchLabels:
      app: order-api
  template:
    metadata:
      labels:
        app: order-api
        version: v1.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: order-api
      terminationGracePeriodSeconds: 30
      # Spread pods across nodes for HA
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values: [order-api]
                topologyKey: kubernetes.io/hostname
      containers:
        - name: order-api
          image: ghcr.io/myorg/order-api:v1.2.0
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: order-api-secrets
                  key: database-url
            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: order-api-config
                  key: log-level
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          # Startup probe: allow slow starts (e.g., DB migrations)
          startupProbe:
            httpGet:
              path: /health
              port: http
            failureThreshold: 30
            periodSeconds: 2
          # Liveness: restart if stuck
          livenessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 15
            timeoutSeconds: 3
            failureThreshold: 3
          # Readiness: remove from Service if unhealthy
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 2
          securityContext:
            runAsNonRoot: true
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities:
              drop: [ALL]

# Key decisions:
# - maxUnavailable: 0 ensures zero-downtime during rollouts
# - startupProbe prevents liveness from killing slow-starting pods
# - readinessProbe vs livenessProbe: readiness = traffic routing, liveness = restart
# - podAntiAffinity spreads pods across nodes (survives node failure)
# - securityContext follows least-privilege principle
SOLUTION
}

# === Exercise 2: Service and Ingress ===
# Problem: Create Service and Ingress resources for the deployment,
# including TLS termination and path-based routing.
exercise_2() {
    echo "=== Exercise 2: Service and Ingress ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: order-api
  namespace: production
  labels:
    app: order-api
spec:
  type: ClusterIP          # Internal only; Ingress handles external traffic
  ports:
    - port: 80
      targetPort: http     # Named port from Deployment
      protocol: TCP
  selector:
    app: order-api

---
# ingress.yaml — Path-based routing with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit-rps: "100"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
      secretName: api-tls-cert       # cert-manager auto-provisions this
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /api/orders
            pathType: Prefix
            backend:
              service:
                name: order-api
                port:
                  number: 80
          - path: /api/users
            pathType: Prefix
            backend:
              service:
                name: user-api
                port:
                  number: 80
          - path: /api/payments
            pathType: Prefix
            backend:
              service:
                name: payment-api
                port:
                  number: 80

# Service types explained:
# ClusterIP   — Internal only (default). Use with Ingress for external access.
# NodePort    — Exposes on each node's IP at a static port (30000-32767).
# LoadBalancer — Provisions cloud LB. Expensive (one LB per service).
# Headless    — No cluster IP. DNS returns pod IPs directly (for StatefulSets).
SOLUTION
}

# === Exercise 3: Resource Management ===
# Problem: Calculate resource requests and limits for a service based on
# observed metrics, and configure HPA accordingly.
exercise_3() {
    echo "=== Exercise 3: Resource Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
def calculate_resource_allocation(
    observed_cpu_p50_millicores: int,
    observed_cpu_p99_millicores: int,
    observed_mem_p50_mb: int,
    observed_mem_p99_mb: int,
) -> dict:
    """Calculate Kubernetes resource requests and limits from observed metrics.

    Request = p50 + 20% buffer (scheduler guarantee)
    Limit   = p99 + 30% buffer (OOM/throttle protection)
    """
    cpu_request = int(observed_cpu_p50_millicores * 1.2)
    cpu_limit = int(observed_cpu_p99_millicores * 1.3)
    mem_request = int(observed_mem_p50_mb * 1.2)
    mem_limit = int(observed_mem_p99_mb * 1.3)

    return {
        "resources": {
            "requests": {"cpu": f"{cpu_request}m", "memory": f"{mem_request}Mi"},
            "limits": {"cpu": f"{cpu_limit}m", "memory": f"{mem_limit}Mi"},
        },
        "rationale": {
            "cpu_request": f"p50({observed_cpu_p50_millicores}m) * 1.2 = {cpu_request}m",
            "cpu_limit": f"p99({observed_cpu_p99_millicores}m) * 1.3 = {cpu_limit}m",
            "mem_request": f"p50({observed_mem_p50_mb}Mi) * 1.2 = {mem_request}Mi",
            "mem_limit": f"p99({observed_mem_p99_mb}Mi) * 1.3 = {mem_limit}Mi",
        },
    }

# Example: order-api observed metrics from Prometheus
allocation = calculate_resource_allocation(
    observed_cpu_p50_millicores=200,
    observed_cpu_p99_millicores=450,
    observed_mem_p50_mb=200,
    observed_mem_p99_mb=380,
)

print("Recommended resource allocation:")
for resource_type, values in allocation["resources"].items():
    print(f"  {resource_type}: cpu={values['cpu']}, memory={values['memory']}")
print("\nRationale:")
for key, val in allocation["rationale"].items():
    print(f"  {key}: {val}")

# HPA configuration:
print("\nHorizontalPodAutoscaler:")
print("  minReplicas: 3")
print("  maxReplicas: 20")
print("  metrics:")
print("    - cpu averageUtilization: 70%")
print("    - memory averageUtilization: 80%")
SOLUTION
}

# === Exercise 4: Troubleshooting Kubernetes ===
# Problem: Given a pod in CrashLoopBackOff, walk through the
# systematic debugging process.
exercise_4() {
    echo "=== Exercise 4: Troubleshooting Kubernetes ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Systematic Kubernetes Troubleshooting Flowchart

# Step 1: Check pod status
# $ kubectl get pods -l app=order-api
# NAME                        READY   STATUS             RESTARTS   AGE
# order-api-7b8f9c-abc12      0/1     CrashLoopBackOff   5          10m

# Step 2: Describe the pod for events
# $ kubectl describe pod order-api-7b8f9c-abc12
# Look for:
#   - Events section (scheduling failures, image pull errors)
#   - Last State (exit code, reason)
#   - Conditions (Ready, ContainersReady)

# Step 3: Check container logs
# $ kubectl logs order-api-7b8f9c-abc12                # Current attempt
# $ kubectl logs order-api-7b8f9c-abc12 --previous     # Previous crash

# Step 4: Common causes and fixes

debug_flowchart = {
    "ImagePullBackOff": {
        "cause": "Cannot pull container image",
        "checks": [
            "Is the image tag correct? (typo in version)",
            "Does the registry require authentication? (imagePullSecrets)",
            "Is the image pushed to the registry?",
        ],
        "fix": "kubectl create secret docker-registry ...",
    },
    "CrashLoopBackOff": {
        "cause": "Container starts and immediately exits",
        "checks": [
            "Check logs: kubectl logs <pod> --previous",
            "Is the entrypoint/CMD correct?",
            "Are required env vars set? (missing DB_URL)",
            "Are config files mounted correctly?",
            "Is the port already in use?",
        ],
        "fix": "Fix the application error or configuration",
    },
    "OOMKilled": {
        "cause": "Container exceeded memory limit",
        "checks": [
            "kubectl describe pod -> Last State: OOMKilled",
            "Is the memory limit too low?",
            "Does the app have a memory leak?",
        ],
        "fix": "Increase memory limit or fix the leak",
    },
    "Pending": {
        "cause": "Pod cannot be scheduled",
        "checks": [
            "kubectl describe pod -> Events (FailedScheduling)",
            "Insufficient CPU/memory on nodes?",
            "Node selector/affinity too restrictive?",
            "PersistentVolumeClaim not bound?",
        ],
        "fix": "Add nodes, reduce requests, or fix PVC",
    },
}

for status, info in debug_flowchart.items():
    print(f"\n{status}:")
    print(f"  Cause: {info['cause']}")
    print(f"  Checks:")
    for check in info["checks"]:
        print(f"    - {check}")
    print(f"  Fix: {info['fix']}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 08: Kubernetes Operations"
echo "========================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
