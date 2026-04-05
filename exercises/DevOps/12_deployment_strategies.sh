#!/bin/bash
# Exercises for Lesson 12: Deployment Strategies
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Rolling Deployment Configuration ===
# Problem: Configure a Kubernetes rolling deployment with proper surge,
# unavailable settings, and rollback criteria.
exercise_1() {
    echo "=== Exercise 1: Rolling Deployment Configuration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Kubernetes Rolling Update Configuration

# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-api
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # Create up to 2 extra pods during update (6+2=8 max)
      maxUnavailable: 1    # Allow 1 pod down during update (5 minimum serving)
  # ...

# How rolling update works with these settings:
# Starting state: 6 pods running v1
# Step 1: Create 2 new v2 pods (maxSurge=2) -> 6 v1 + 2 v2 = 8 pods
# Step 2: Once v2 pods pass readiness, terminate 2 v1 pods -> 4 v1 + 2 v2
# Step 3: Create 2 more v2 pods -> 4 v1 + 4 v2 = 8 pods
# Step 4: Terminate 2 v1 -> 2 v1 + 4 v2
# Step 5: Create 2 more v2 -> 2 v1 + 6 v2 = 8 pods
# Step 6: Terminate 2 v1 -> 0 v1 + 6 v2 (done)

# Rollback commands:
# kubectl rollout status deployment/order-api    # Watch progress
# kubectl rollout undo deployment/order-api      # Instant rollback to previous
# kubectl rollout undo deployment/order-api --to-revision=3  # Specific revision
# kubectl rollout history deployment/order-api   # View revision history

# Strategy selection matrix:
strategies = {
    "maxSurge=1, maxUnavailable=0": {
        "behavior": "Zero-downtime, slow (one pod at a time)",
        "use_case": "Payment-critical services where every request matters",
    },
    "maxSurge=25%, maxUnavailable=25%": {
        "behavior": "Balanced speed and availability (K8s default)",
        "use_case": "Most stateless web services",
    },
    "maxSurge=100%, maxUnavailable=0": {
        "behavior": "Blue-green style: spin up all new, then drain old",
        "use_case": "When you need instant rollback by scaling old back up",
    },
}

for config, details in strategies.items():
    print(f"\n  {config}")
    print(f"    Behavior: {details['behavior']}")
    print(f"    Use case: {details['use_case']}")
SOLUTION
}

# === Exercise 2: Blue-Green with Load Balancer ===
# Problem: Implement blue-green deployment using Kubernetes Services
# and traffic switching.
exercise_2() {
    echo "=== Exercise 2: Blue-Green with Load Balancer ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Blue-Green in Kubernetes: Two Deployments, one Service selector switch

# 1. Deploy BLUE (current production)
# deployment-blue.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-api-blue
  labels:
    app: order-api
    slot: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-api
      slot: blue
  template:
    metadata:
      labels:
        app: order-api
        slot: blue
        version: v1.0.0
    spec:
      containers:
        - name: order-api
          image: ghcr.io/myorg/order-api:v1.0.0

# 2. Service points to BLUE
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: order-api
spec:
  selector:
    app: order-api
    slot: blue           # <-- Points to blue
  ports:
    - port: 80
      targetPort: 8080

# 3. Deploy GREEN (new version) alongside blue
# deployment-green.yaml (same but slot: green, image: v1.1.0)

# 4. Verify green is healthy
# kubectl get pods -l slot=green  # All Running and Ready

# 5. SWITCH traffic: update Service selector to green
# kubectl patch service order-api -p '{"spec":{"selector":{"slot":"green"}}}'
# Instant cutover — all traffic now goes to green

# 6. ROLLBACK if needed: switch back to blue
# kubectl patch service order-api -p '{"spec":{"selector":{"slot":"blue"}}}'

# 7. Cleanup: delete blue deployment after confidence period
# kubectl delete deployment order-api-blue

# Advantages:
#   + Instant rollback (just flip the selector back)
#   + Full testing of green before switching
#   + Zero-downtime cutover
# Disadvantages:
#   - 2x resources during deployment (both blue and green running)
#   - No gradual traffic shifting (all-or-nothing)
SOLUTION
}

# === Exercise 3: Canary with Traffic Splitting ===
# Problem: Implement canary deployment with Istio virtual service
# for weighted traffic routing.
exercise_3() {
    echo "=== Exercise 3: Canary with Traffic Splitting ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Canary with Istio VirtualService for precise traffic splitting

# Step 1: Deploy canary alongside stable
# Two Deployments: order-api-stable (v1.0) and order-api-canary (v1.1)
# Both behind the same Service (selected by app: order-api)

# Step 2: Istio DestinationRule — define subsets
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-api
spec:
  host: order-api
  subsets:
    - name: stable
      labels:
        version: v1.0.0
    - name: canary
      labels:
        version: v1.1.0

# Step 3: VirtualService — route 5% to canary
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-api
spec:
  hosts:
    - order-api
  http:
    - route:
        - destination:
            host: order-api
            subset: stable
          weight: 95
        - destination:
            host: order-api
            subset: canary
          weight: 5

# Step 4: Monitor canary metrics (compare vs stable)
# If canary error rate < stable error rate + threshold:
#   Increase: 5% -> 25% -> 50% -> 100%
# If canary error rate > threshold:
#   Rollback: set canary weight to 0, scale down canary

# Progressive delivery tools:
tools = {
    "Argo Rollouts": "Kubernetes-native, integrates with Istio/NGINX/ALB",
    "Flagger":       "Progressive delivery for Istio/Linkerd/App Mesh",
    "Istio":         "Service mesh with VirtualService traffic splitting",
    "AWS ALB":       "Weighted target groups (no service mesh required)",
}

print("Canary Deployment Tools:")
for tool, desc in tools.items():
    print(f"  {tool:16s}: {desc}")
SOLUTION
}

# === Exercise 4: Feature Flags ===
# Problem: Implement feature flags for progressive rollout without
# redeployment.
exercise_4() {
    echo "=== Exercise 4: Feature Flags ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass
import hashlib

@dataclass
class FeatureFlag:
    name: str
    enabled: bool
    rollout_percentage: int = 100    # 0-100
    allowed_users: list[str] | None = None

    def is_enabled_for(self, user_id: str) -> bool:
        """Check if the flag is enabled for a specific user."""
        if not self.enabled:
            return False

        # Allow-listed users always get the feature
        if self.allowed_users and user_id in self.allowed_users:
            return True

        # Percentage rollout: hash user_id for deterministic bucketing
        # Same user always gets the same result (no flickering)
        hash_val = int(hashlib.md5(
            f"{self.name}:{user_id}".encode()
        ).hexdigest(), 16)
        bucket = hash_val % 100

        return bucket < self.rollout_percentage

# Feature flag configuration
flags = {
    "new_checkout_flow": FeatureFlag(
        name="new_checkout_flow",
        enabled=True,
        rollout_percentage=10,        # 10% of users
        allowed_users=["internal-tester-1"],
    ),
    "dark_mode": FeatureFlag(
        name="dark_mode",
        enabled=True,
        rollout_percentage=100,       # Everyone
    ),
    "experimental_search": FeatureFlag(
        name="experimental_search",
        enabled=False,                # Killed globally
    ),
}

# Usage in application code:
# if flags["new_checkout_flow"].is_enabled_for(current_user.id):
#     return render_new_checkout()
# else:
#     return render_old_checkout()

# Test
for user_id in ["user-1", "user-2", "user-50", "internal-tester-1"]:
    result = flags["new_checkout_flow"].is_enabled_for(user_id)
    print(f"  new_checkout_flow for {user_id}: {result}")

# Feature flag lifecycle:
# 1. Create flag (disabled) -> deploy code with flag check
# 2. Enable for internal testers (allowed_users)
# 3. Gradual rollout: 5% -> 25% -> 50% -> 100%
# 4. Measure metrics at each stage
# 5. If successful: remove flag from code (tech debt cleanup)
# 6. If failed: disable flag (instant kill switch, no redeploy)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 12: Deployment Strategies"
echo "========================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
