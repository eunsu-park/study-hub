#!/bin/bash
# Exercises for Lesson 14: GitOps
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: GitOps Repository Structure ===
# Problem: Design a Git repository structure for a GitOps workflow
# managing multiple environments and applications.
exercise_1() {
    echo "=== Exercise 1: GitOps Repository Structure ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Two common patterns for GitOps repo structure:

# Pattern A: Mono-repo (one repo for all environments)
# infra-manifests/
#   base/                      # Shared base manifests (Kustomize)
#     order-api/
#       deployment.yaml
#       service.yaml
#       kustomization.yaml
#     user-api/
#       deployment.yaml
#       service.yaml
#       kustomization.yaml
#   overlays/                  # Environment-specific patches
#     dev/
#       kustomization.yaml     # Uses base + dev patches
#       patches/
#         replicas.yaml        # replicas: 1
#     staging/
#       kustomization.yaml
#       patches/
#         replicas.yaml        # replicas: 2
#     production/
#       kustomization.yaml
#       patches/
#         replicas.yaml        # replicas: 5
#         resources.yaml       # Higher CPU/memory limits

# Pattern B: Multi-repo (separate repos per environment)
# app-repo/          # Application source code + Dockerfile
# infra-dev/         # Dev environment manifests
# infra-staging/     # Staging environment manifests
# infra-production/  # Production manifests (stricter access control)

# Kustomization example:
# base/order-api/kustomization.yaml
# apiVersion: kustomize.config.k8s.io/v1beta1
# kind: Kustomization
# resources:
#   - deployment.yaml
#   - service.yaml

# overlays/production/kustomization.yaml
# apiVersion: kustomize.config.k8s.io/v1beta1
# kind: Kustomization
# bases:
#   - ../../base/order-api
# patchesStrategicMerge:
#   - patches/replicas.yaml
#   - patches/resources.yaml
# namespace: production
# commonLabels:
#   environment: production

print("GitOps Repository Patterns:")
print("  Mono-repo:  Simpler management, single source of truth")
print("              Risk: one bad merge affects all environments")
print("  Multi-repo: Better access control per environment")
print("              Risk: config drift between repos")
print()
print("Recommendation: Start with mono-repo + branch protection")
print("  - main branch = production (auto-sync)")
print("  - PR required for changes (human approval gate)")
print("  - Environment promotion via Kustomize overlays")
SOLUTION
}

# === Exercise 2: ArgoCD Application Setup ===
# Problem: Configure an ArgoCD Application resource for automated
# sync with health checks and sync policies.
exercise_2() {
    echo "=== Exercise 2: ArgoCD Application Setup ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# ArgoCD Application manifest
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: order-api-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default

  # Source: Git repository containing manifests
  source:
    repoURL: https://github.com/myorg/infra-manifests.git
    targetRevision: main            # Track main branch
    path: overlays/production       # Kustomize overlay path

  # Destination: target Kubernetes cluster and namespace
  destination:
    server: https://kubernetes.default.svc
    namespace: production

  # Sync policy: auto-sync + auto-prune
  syncPolicy:
    automated:
      prune: true              # Delete resources removed from Git
      selfHeal: true           # Revert manual changes (drift correction)
      allowEmpty: false        # Don't sync if manifests resolve to nothing
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - ApplyOutOfSyncOnly=true   # Only apply changed resources
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # Health checks
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas       # Ignore HPA-managed replica count

# ArgoCD sync states:
# Synced     — cluster matches Git (green)
# OutOfSync  — cluster differs from Git (yellow)
# Unknown    — cannot determine status (gray)
# Degraded   — application health check failing (red)

# Key settings explained:
print("ArgoCD Sync Policy:")
print("  automated.prune:    Delete K8s resources removed from Git")
print("  automated.selfHeal: Auto-revert manual kubectl changes")
print("  retry:              Retry failed syncs with exponential backoff")
print("  ignoreDifferences:  Skip fields managed by other controllers (HPA)")
SOLUTION
}

# === Exercise 3: GitOps Promotion Workflow ===
# Problem: Design a promotion workflow: dev -> staging -> production
# using Git branches or directory-based environments.
exercise_3() {
    echo "=== Exercise 3: GitOps Promotion Workflow ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Promotion workflow: image update flows through environments

# Step 1: CI builds new image from app repo
# CI pipeline: build -> test -> push image (ghcr.io/myorg/order-api:v1.2.0)

# Step 2: Automated PR to update dev overlay
# A bot (Renovate, Kustomize image updater, ArgoCD Image Updater) opens a PR:
#
# overlays/dev/kustomization.yaml:
# images:
#   - name: order-api
#     newTag: v1.2.0    # Updated from v1.1.0

# Step 3: ArgoCD auto-syncs dev (automated sync policy)
# Dev cluster now runs v1.2.0

# Step 4: Manual PR to promote to staging
# Copy the image tag change to staging overlay:
# overlays/staging/kustomization.yaml:
# images:
#   - name: order-api
#     newTag: v1.2.0

# Step 5: Staging PR requires approval (branch protection)
# Reviewer checks: dev metrics look good, no errors, tests passed

# Step 6: Manual PR to promote to production
# Same process, but with stricter approval (2 reviewers + SRE sign-off)

# Automation options:
promotion_tools = {
    "ArgoCD Image Updater": "Auto-update image tags in Git based on registry",
    "Flux Image Automation": "Monitor container registry, commit tag updates",
    "Renovate Bot":          "General dependency update bot (Docker tags too)",
    "Custom CI step":        "Script that opens PR with new image tag",
}

print("Promotion Flow: dev -> staging -> production")
print()
print("  dev:        Auto-sync on every commit to main")
print("  staging:    PR with 1 reviewer approval")
print("  production: PR with 2 reviewers + SRE sign-off")
print()
print("Promotion Tools:")
for tool, desc in promotion_tools.items():
    print(f"  {tool:28s}: {desc}")
SOLUTION
}

# === Exercise 4: Rollback in GitOps ===
# Problem: Explain how rollback works in GitOps vs traditional deployment.
exercise_4() {
    echo "=== Exercise 4: Rollback in GitOps ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# GitOps rollback = Git revert (not kubectl rollout undo)

# Traditional rollback:
# kubectl rollout undo deployment/order-api
# Problem: cluster state diverges from Git (Git still says v1.2.0)

# GitOps rollback (correct way):
# git revert HEAD       # Reverts the commit that updated the image tag
# git push              # ArgoCD auto-syncs the reverted state
# Cluster and Git are always in sync!

rollback_scenarios = {
    "Bad deployment (latest commit)": {
        "command": "git revert HEAD && git push",
        "effect": "Reverts image tag to previous version in Git",
        "argocd": "Auto-syncs, rolls back pods to previous image",
    },
    "Rollback to specific version": {
        "command": "Edit kustomization.yaml to set newTag: v1.0.0, commit, push",
        "effect": "Explicitly set the desired version",
        "argocd": "Auto-syncs to specified version",
    },
    "Emergency rollback (ArgoCD UI)": {
        "command": "ArgoCD UI -> Sync -> select previous revision",
        "effect": "Temporarily syncs to old Git commit",
        "argocd": "WARN: Git and cluster may diverge until Git is updated",
    },
    "Disable auto-sync first": {
        "command": "argocd app set order-api --sync-policy none",
        "effect": "Pauses auto-sync for manual investigation",
        "argocd": "Prevents selfHeal from undoing your manual fixes",
    },
}

print("GitOps Rollback Strategies:")
for scenario, details in rollback_scenarios.items():
    print(f"\n  {scenario}:")
    print(f"    Command: {details['command']}")
    print(f"    Effect:  {details['effect']}")

# Golden rule: In GitOps, Git is ALWAYS the source of truth.
# kubectl apply, kubectl edit, kubectl patch are anti-patterns.
# All changes must go through Git (PR -> merge -> auto-sync).
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 14: GitOps"
echo "=========================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
