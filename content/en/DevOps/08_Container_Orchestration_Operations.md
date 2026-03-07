# Lesson 8: Container Orchestration Operations

**Previous**: [Configuration Management](./07_Configuration_Management.md) | **Next**: [Service Mesh and Networking](./09_Service_Mesh_and_Networking.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write and apply Kubernetes Deployment manifests with rolling update strategies for zero-downtime releases
2. Configure Services (ClusterIP, NodePort, LoadBalancer) to expose applications within and outside the cluster
3. Manage application configuration with ConfigMaps and sensitive data with Secrets
4. Set resource requests and limits to ensure fair scheduling and prevent resource starvation
5. Configure Horizontal Pod Autoscaler (HPA) to automatically scale workloads based on CPU, memory, or custom metrics
6. Apply node affinity, taints, and tolerations to control pod placement across nodes

---

Kubernetes has become the standard platform for running containerized applications in production. While the Docker lesson covered Kubernetes fundamentals (Pods, Deployments, Services), this lesson focuses on the operational patterns that DevOps engineers use daily: rolling updates, resource management, autoscaling, and advanced scheduling. These are the skills that keep production Kubernetes clusters running smoothly, efficiently, and reliably.

> **Analogy -- City Infrastructure Management:** Running Kubernetes in production is like managing a city's infrastructure. Deployments are construction projects (rolling out new buildings without shutting down the city). Services are the road network (routing traffic to the right buildings). Resource limits are zoning laws (preventing one building from using all the electricity). HPA is the city's growth plan (building more housing when population increases).

## 1. Deployments in Depth

### Deployment Manifest

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: v2.1.0
  annotations:
    deployment.kubernetes.io/revision: "3"
    description: "Main API server for the application"
spec:
  replicas: 3
  revisionHistoryLimit: 10                  # Keep 10 old ReplicaSets for rollback

  selector:
    matchLabels:
      app: api-server                       # Must match template labels

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1                           # At most 1 extra pod during update
      maxUnavailable: 0                     # Never reduce below desired replicas

  template:
    metadata:
      labels:
        app: api-server
        version: v2.1.0
    spec:
      terminationGracePeriodSeconds: 30     # Time for graceful shutdown

      containers:
        - name: api
          image: registry.example.com/api-server:v2.1.0
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics

          # Health checks
          readinessProbe:                    # Is the pod ready to receive traffic?
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 3

          livenessProbe:                     # Is the pod alive and functioning?
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 15
            periodSeconds: 20
            failureThreshold: 3

          startupProbe:                      # Has the pod finished starting up?
            httpGet:
              path: /health/startup
              port: http
            initialDelaySeconds: 0
            periodSeconds: 5
            failureThreshold: 30             # 30 * 5 = 150s max startup time

          # Resource management
          resources:
            requests:
              cpu: 250m                      # 0.25 CPU cores
              memory: 256Mi                  # 256 MiB RAM
            limits:
              cpu: 500m                      # 0.5 CPU cores
              memory: 512Mi                  # 512 MiB RAM

          # Environment variables
          env:
            - name: PORT
              value: "8080"
            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: api-config
                  key: log_level
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: db_password

          # Volume mounts
          volumeMounts:
            - name: config-volume
              mountPath: /etc/app/config
              readOnly: true

      volumes:
        - name: config-volume
          configMap:
            name: api-config
```

### Rolling Update Strategies

```
Strategy: RollingUpdate (default)
──────────────────────────────────
Gradually replaces old pods with new pods.

maxSurge: 1, maxUnavailable: 0 (safest -- never below desired count)
  Replicas: 3
  Step 1: [v1] [v1] [v1] [v2]      4 total (1 surge)
  Step 2: [v1] [v1] [v2] [v2]      4 total
  Step 3: [v1] [v2] [v2] [v2]      4 total
  Step 4: [v2] [v2] [v2]           3 total (done)

maxSurge: 0, maxUnavailable: 1 (no extra resources needed)
  Replicas: 3
  Step 1: [v1] [v1] [--]           2 available (1 terminating)
  Step 2: [v1] [v1] [v2]           3 total (new pod starting)
  Step 3: [v1] [--] [v2]           2 available
  Step 4: [v1] [v2] [v2]           3 total
  Step 5: [v2] [v2] [v2]           3 total (done)

maxSurge: 25%, maxUnavailable: 25% (balanced -- default)
  Replicas: 4
  Up to 5 pods total (4 + 25% surge)
  Down to 3 pods available (4 - 25% unavailable)

Strategy: Recreate
──────────────────
Kill all old pods, then create new pods.
  Step 1: [v1] [v1] [v1]           Running
  Step 2: [--] [--] [--]           All terminated (DOWNTIME)
  Step 3: [v2] [v2] [v2]           All new pods starting
  Use only when: new version is incompatible with old version
```

### Deployment Operations

```bash
# Apply a deployment
kubectl apply -f deployment.yaml

# Watch the rollout
kubectl rollout status deployment/api-server

# View rollout history
kubectl rollout history deployment/api-server

# Rollback to previous version
kubectl rollout undo deployment/api-server

# Rollback to specific revision
kubectl rollout undo deployment/api-server --to-revision=2

# Pause a rollout (for canary-style manual verification)
kubectl rollout pause deployment/api-server

# Resume a paused rollout
kubectl rollout resume deployment/api-server

# Scale deployment
kubectl scale deployment/api-server --replicas=5

# Update image (triggers rolling update)
kubectl set image deployment/api-server api=registry.example.com/api-server:v2.2.0

# Restart all pods (rolling restart, no config change)
kubectl rollout restart deployment/api-server
```

---

## 2. Services

Services provide stable network endpoints for accessing pods.

### Service Types

```
┌────────────────────────────────────────────────────────────────┐
│                    Kubernetes Service Types                     │
│                                                                 │
│  ClusterIP (default)                                           │
│  ┌──────────┐                                                  │
│  │ Service   │──▶ [Pod 1] [Pod 2] [Pod 3]                     │
│  │ 10.0.0.5  │    Internal access only                        │
│  └──────────┘                                                  │
│                                                                 │
│  NodePort                                                       │
│  ┌──────────┐     ┌──────────┐                                 │
│  │ External  │────▶│ Service   │──▶ [Pod 1] [Pod 2]           │
│  │ :30080    │     │ :80       │    Exposed on every node       │
│  └──────────┘     └──────────┘                                 │
│                                                                 │
│  LoadBalancer                                                   │
│  ┌──────────┐     ┌──────────┐                                 │
│  │ Cloud LB  │────▶│ Service   │──▶ [Pod 1] [Pod 2] [Pod 3]  │
│  │ (AWS ELB) │     │ :80       │    Cloud provider provisions  │
│  └──────────┘     └──────────┘    external load balancer       │
│                                                                 │
│  ExternalName                                                   │
│  ┌──────────┐                                                  │
│  │ Service   │──▶ CNAME to external.database.com               │
│  │ (DNS)     │    Maps to external service                     │
│  └──────────┘                                                  │
└────────────────────────────────────────────────────────────────┘
```

### Service Manifests

```yaml
# ClusterIP Service (internal only)
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: production
spec:
  type: ClusterIP                           # Default type
  selector:
    app: api-server                         # Routes to pods with this label
  ports:
    - name: http
      port: 80                              # Service port
      targetPort: 8080                      # Container port
      protocol: TCP
    - name: metrics
      port: 9090
      targetPort: 9090
```

```yaml
# NodePort Service (external access via node IP)
apiVersion: v1
kind: Service
metadata:
  name: api-nodeport
spec:
  type: NodePort
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080                       # External port (30000-32767)
```

```yaml
# LoadBalancer Service (cloud provider load balancer)
apiVersion: v1
kind: Service
metadata:
  name: api-lb
  annotations:
    # AWS-specific annotations
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-scheme: internet-facing
spec:
  type: LoadBalancer
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 8080
```

### DNS Resolution

```
Within the same namespace:
  curl http://api-service           → resolves to ClusterIP

Across namespaces:
  curl http://api-service.production.svc.cluster.local

Full DNS format:
  <service-name>.<namespace>.svc.cluster.local
```

---

## 3. ConfigMaps and Secrets

### ConfigMaps

```yaml
# ConfigMap from YAML
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
  namespace: production
data:
  # Simple key-value pairs
  log_level: "info"
  max_connections: "100"
  feature_flags: "new_dashboard=true,dark_mode=false"

  # Entire configuration file
  app.properties: |
    server.port=8080
    server.host=0.0.0.0
    database.pool.size=10
    cache.ttl=300
```

```bash
# Create ConfigMap from command line
kubectl create configmap api-config \
  --from-literal=log_level=info \
  --from-literal=max_connections=100

# Create ConfigMap from file
kubectl create configmap nginx-config \
  --from-file=nginx.conf

# Create ConfigMap from directory (each file becomes a key)
kubectl create configmap app-config \
  --from-file=config/
```

### Using ConfigMaps in Pods

```yaml
spec:
  containers:
    - name: api
      image: api-server:v1

      # Option 1: As environment variables
      env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: api-config
              key: log_level

      # Option 2: All keys as environment variables
      envFrom:
        - configMapRef:
            name: api-config

      # Option 3: As a mounted volume (file)
      volumeMounts:
        - name: config-volume
          mountPath: /etc/app/config
          readOnly: true

  volumes:
    - name: config-volume
      configMap:
        name: api-config
        items:                              # Mount specific keys as files
          - key: app.properties
            path: application.properties    # File name in the mount
```

### Secrets

```yaml
# Secret manifest (values must be base64 encoded)
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
  namespace: production
type: Opaque
data:
  db_password: cGFzc3dvcmQxMjM=           # echo -n 'password123' | base64
  api_key: c2stYWJjMTIzZGVm               # echo -n 'sk-abc123def' | base64

# Or use stringData (plain text, auto-encoded)
stringData:
  db_password: password123
  api_key: sk-abc123def
```

```bash
# Create Secret from command line
kubectl create secret generic api-secrets \
  --from-literal=db_password=password123 \
  --from-literal=api_key=sk-abc123def

# Create TLS Secret
kubectl create secret tls tls-cert \
  --cert=server.crt \
  --key=server.key

# Create Docker registry Secret
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=admin \
  --docker-password=secretpass
```

---

## 4. Resource Requests and Limits

### Understanding Resources

```
Requests: Minimum resources guaranteed to the container
  - Used by the scheduler to decide which node to place the pod on
  - The container is guaranteed to get this amount

Limits: Maximum resources the container can use
  - CPU: Container is throttled if it exceeds the limit
  - Memory: Container is OOMKilled if it exceeds the limit

┌────────────────────────────────────────────────────┐
│  Container Resource Usage                           │
│                                                     │
│  0 ─────── Request ─────── Limit ─────── Node Max  │
│  │         (guaranteed)    (maximum)                │
│  │                                                  │
│  │  [====]                 CPU: throttled            │
│  │  [===========]          Memory: OOMKilled         │
│                                                     │
│  Best practice: Request ≈ typical usage              │
│                 Limit ≈ peak usage                   │
└────────────────────────────────────────────────────┘
```

### Resource Specifications

```yaml
spec:
  containers:
    - name: api
      image: api-server:v1
      resources:
        requests:
          cpu: 250m              # 250 millicores = 0.25 CPU
          memory: 256Mi          # 256 Mebibytes
          ephemeral-storage: 1Gi # 1 GiB local disk
        limits:
          cpu: 500m              # 0.5 CPU (throttled above this)
          memory: 512Mi          # OOMKilled above this
          ephemeral-storage: 2Gi

# CPU units:
#   1 CPU = 1000m (millicores)
#   250m = 0.25 CPU = quarter of a core
#   1.5 = 1500m = 1.5 cores

# Memory units:
#   Ki = Kibibytes (1024 bytes)
#   Mi = Mebibytes (1024 Ki)
#   Gi = Gibibytes (1024 Mi)
```

### Quality of Service (QoS) Classes

```
Kubernetes assigns QoS classes based on resource specifications:

Guaranteed (highest priority, evicted last):
  - Every container has requests == limits for both CPU and memory
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 500m
      memory: 512Mi

Burstable (medium priority):
  - At least one container has a request set
  - Requests != limits
  resources:
    requests:
      cpu: 250m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi

BestEffort (lowest priority, evicted first):
  - No requests or limits set
  resources: {}   # Empty or omitted
```

### LimitRange (Namespace Defaults)

```yaml
# Set default resource constraints for a namespace
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: production
spec:
  limits:
    - type: Container
      default:                          # Default limits (if not specified)
        cpu: 500m
        memory: 512Mi
      defaultRequest:                   # Default requests (if not specified)
        cpu: 100m
        memory: 128Mi
      max:                              # Maximum allowed limits
        cpu: 2
        memory: 4Gi
      min:                              # Minimum allowed requests
        cpu: 50m
        memory: 64Mi
```

### ResourceQuota (Namespace Budget)

```yaml
# Limit total resources in a namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: namespace-quota
  namespace: development
spec:
  hard:
    requests.cpu: "10"                 # Total CPU requests across all pods
    requests.memory: 20Gi             # Total memory requests
    limits.cpu: "20"                  # Total CPU limits
    limits.memory: 40Gi               # Total memory limits
    pods: "50"                        # Maximum number of pods
    services: "10"                    # Maximum number of services
    persistentvolumeclaims: "20"      # Maximum PVCs
```

---

## 5. Horizontal Pod Autoscaler (HPA)

HPA automatically scales the number of pod replicas based on observed metrics.

### Basic HPA (CPU-based)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server

  minReplicas: 3
  maxReplicas: 20

  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70          # Scale up when avg CPU > 70%

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60      # Wait 60s before scaling up again
      policies:
        - type: Percent
          value: 100                      # Can double the replicas
          periodSeconds: 60
        - type: Pods
          value: 4                        # Or add up to 4 pods
          periodSeconds: 60
      selectPolicy: Max                   # Use whichever policy allows more scaling

    scaleDown:
      stabilizationWindowSeconds: 300     # Wait 5 min before scaling down
      policies:
        - type: Percent
          value: 10                       # Remove at most 10% of pods
          periodSeconds: 60
```

### Multi-Metric HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server

  minReplicas: 3
  maxReplicas: 50

  metrics:
    # CPU-based scaling
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    # Memory-based scaling
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

    # Custom metric (e.g., requests per second from Prometheus)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: 1000              # Scale when avg RPS > 1000
```

```bash
# Create HPA imperatively
kubectl autoscale deployment api-server \
  --min=3 --max=20 --cpu-percent=70

# Check HPA status
kubectl get hpa api-server-hpa

# Detailed HPA info
kubectl describe hpa api-server-hpa

# Watch HPA in action
kubectl get hpa -w
```

### Prerequisites for HPA

```bash
# HPA requires the Metrics Server to be installed
# Check if metrics-server is running
kubectl get deployment metrics-server -n kube-system

# Install metrics-server (if not present)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics are available
kubectl top nodes
kubectl top pods
```

---

## 6. Node Affinity and Anti-Affinity

### Node Labels

```bash
# Label nodes
kubectl label nodes node-1 disk=ssd
kubectl label nodes node-2 disk=hdd
kubectl label nodes node-1 zone=us-east-1a
kubectl label nodes node-2 zone=us-east-1b

# View labels
kubectl get nodes --show-labels
```

### Node Affinity

```yaml
spec:
  affinity:
    nodeAffinity:
      # HARD requirement -- pod won't schedule without this
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: disk
                operator: In
                values:
                  - ssd                     # Must schedule on SSD nodes

      # SOFT preference -- scheduler tries but not required
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 80                        # Priority weight (1-100)
          preference:
            matchExpressions:
              - key: zone
                operator: In
                values:
                  - us-east-1a              # Prefer zone 1a
```

### Pod Anti-Affinity (Spread Pods)

```yaml
spec:
  affinity:
    podAntiAffinity:
      # HARD: Never put two api-server pods on the same node
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - api-server
          topologyKey: kubernetes.io/hostname

      # SOFT: Try to spread across availability zones
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchExpressions:
                - key: app
                  operator: In
                  values:
                    - api-server
            topologyKey: topology.kubernetes.io/zone
```

### Taints and Tolerations

```bash
# Taint a node (prevent scheduling unless tolerated)
kubectl taint nodes node-gpu gpu=true:NoSchedule

# Only pods with matching toleration will be scheduled on node-gpu
```

```yaml
# Pod that tolerates the GPU taint
spec:
  tolerations:
    - key: "gpu"
      operator: "Equal"
      value: "true"
      effect: "NoSchedule"

  containers:
    - name: ml-training
      image: ml-training:v1
      resources:
        limits:
          nvidia.com/gpu: 1               # Request 1 GPU
```

```
Taint Effects:
  NoSchedule:       Don't schedule new pods (existing pods stay)
  PreferNoSchedule: Try to avoid scheduling (soft version)
  NoExecute:        Evict existing pods AND don't schedule new ones
```

---

## 7. Topology Spread Constraints

```yaml
# Evenly distribute pods across zones and nodes
spec:
  topologySpreadConstraints:
    # Spread across availability zones
    - maxSkew: 1                            # Max difference in pod count between zones
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: DoNotSchedule      # Hard constraint
      labelSelector:
        matchLabels:
          app: api-server

    # Spread across nodes within each zone
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: ScheduleAnyway     # Soft constraint
      labelSelector:
        matchLabels:
          app: api-server
```

---

## 8. Operational Commands Reference

```bash
# === Debugging ===
# Get pod logs
kubectl logs api-server-abc123
kubectl logs api-server-abc123 -c sidecar     # Specific container
kubectl logs api-server-abc123 --previous      # Previous crash logs
kubectl logs -f api-server-abc123              # Follow logs
kubectl logs -l app=api-server --all-containers # All pods with label

# Exec into a pod
kubectl exec -it api-server-abc123 -- /bin/sh
kubectl exec -it api-server-abc123 -c sidecar -- bash

# Describe (events, conditions)
kubectl describe pod api-server-abc123
kubectl describe deployment api-server
kubectl describe node node-1

# Get events (sorted by time)
kubectl get events --sort-by='.lastTimestamp'
kubectl get events --field-selector type=Warning

# === Resource inspection ===
kubectl get all -n production
kubectl get pods -o wide                       # Show node and IP
kubectl get pods -o yaml                       # Full YAML
kubectl get pods -l app=api-server             # Filter by label

# === Port forwarding ===
kubectl port-forward pod/api-server-abc123 8080:8080
kubectl port-forward svc/api-service 8080:80

# === Resource usage ===
kubectl top pods
kubectl top nodes
kubectl top pods --sort-by=memory
```

---

## Exercises

### Exercise 1: Zero-Downtime Deployment

Create a deployment with a rolling update strategy:
1. Deploy version v1 with 3 replicas, maxSurge=1, maxUnavailable=0
2. Add readiness and liveness probes
3. Update to version v2 and observe the rolling update with `kubectl get pods -w`
4. Verify that at no point during the update were fewer than 3 pods ready
5. Rollback to v1 and verify the rollback

### Exercise 2: Resource Management

Set up resource management for a namespace:
1. Create a namespace called `team-alpha`
2. Apply a LimitRange with default requests (100m CPU, 128Mi memory) and limits (500m CPU, 512Mi memory)
3. Apply a ResourceQuota limiting total requests to 4 CPU and 8Gi memory
4. Deploy an application without explicit resource specs and verify defaults are applied
5. Try to deploy enough replicas to exceed the ResourceQuota and observe the error

### Exercise 3: Autoscaling

Configure autoscaling for an application:
1. Deploy a CPU-intensive application (e.g., use a container that computes hashes)
2. Set resource requests to 100m CPU
3. Create an HPA targeting 50% CPU utilization, min=1, max=10
4. Generate load (e.g., `kubectl run -i --tty load-generator --image=busybox -- /bin/sh -c "while true; do wget -q -O- http://api-service; done"`)
5. Watch the HPA scale up pods
6. Stop the load and watch it scale down

### Exercise 4: Pod Scheduling

Practice advanced pod scheduling:
1. Label two nodes: one with `tier=frontend`, one with `tier=backend`
2. Create a deployment with node affinity that requires `tier=frontend`
3. Create another deployment with pod anti-affinity that spreads pods across nodes
4. Taint a node with `maintenance=true:NoSchedule`
5. Verify that new pods are not scheduled on the tainted node
6. Add a toleration to a pod and verify it can schedule on the tainted node

### Exercise 5: ConfigMap and Secret Rotation

Practice configuration management:
1. Create a ConfigMap with application settings and mount it as a volume
2. Create a Secret with database credentials and inject as environment variables
3. Update the ConfigMap and observe how volume-mounted configs update automatically (may take ~60s)
4. Update the Secret and observe that environment variable changes require a pod restart
5. Perform a rolling restart to pick up the new Secret values

---

**Previous**: [Configuration Management](./07_Configuration_Management.md) | [Overview](00_Overview.md) | **Next**: [Service Mesh and Networking](./09_Service_Mesh_and_Networking.md)

**License**: CC BY-NC 4.0
