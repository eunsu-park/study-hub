#!/bin/bash
# Exercises for Lesson 10: CI/CD Pipelines
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Build a Basic GitHub Actions CI Workflow ===
# Problem: Create a workflow that runs tests and builds a Docker image on every push.
exercise_1() {
    echo "=== Exercise 1: Build a Basic GitHub Actions CI Workflow ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- .github/workflows/ci.yml ---"
    cat << 'SOLUTION'
name: CI

# Trigger on push to main and on pull requests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    # PRs trigger CI for code review — the build must pass before merging

jobs:
  build:
    runs-on: ubuntu-latest
    # ubuntu-latest includes Docker pre-installed

    steps:
    # Step 1: Check out the repository code
    - uses: actions/checkout@v4
      # This clones the repo into the runner's workspace
      # Without this, no source code is available for the build

    # Step 2: Set up Docker Buildx (enables advanced build features)
    - uses: docker/setup-buildx-action@v3
      # Buildx provides: multi-platform builds, build caching,
      # and the ability to push directly from the build step

    # Step 3: Build the Docker image (do NOT push)
    - name: Build Docker image
      run: docker build -t myapp:test .
      # -t myapp:test: tag the image locally for smoke testing
      # No --push: this is CI, not CD — we just verify the build works

    # Step 4: Run a smoke test to verify the image works
    - name: Smoke test
      run: docker run --rm myapp:test echo "Image works"
      # --rm: auto-remove the container after it exits
      # If the image is broken (bad CMD, missing deps), this step fails
      # and the workflow is marked as failed
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Push a commit to trigger the workflow:
git add .github/workflows/ci.yml
git commit -m "Add CI workflow"
git push origin main

# Check the Actions tab in GitHub — the workflow should run and pass.

# Introduce a deliberate error to test failure detection:
# e.g., change the base image to a non-existent image:
#   FROM nonexistent-image:99.99
# Push and observe the workflow fail at the "Build Docker image" step.

# Key insight: CI catches build failures BEFORE they reach production.
# Every PR is validated automatically — no manual testing needed.
SOLUTION
}

# === Exercise 2: Push an Image to a Registry ===
# Problem: Build and push a versioned image to a container registry.
exercise_2() {
    echo "=== Exercise 2: Push an Image to a Registry ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- .github/workflows/ci-push.yml ---"
    cat << 'SOLUTION'
name: CI with Registry Push

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - uses: docker/setup-buildx-action@v3

    # Login to Docker Hub (or GHCR)
    - uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        # Secrets are stored in GitHub Settings > Secrets and variables
        # NEVER hardcode credentials in workflow files

    # Generate image tags from Git metadata
    - uses: docker/metadata-action@v4
      id: meta
      with:
        images: yourusername/myapp
        tags: |
          type=ref,event=branch
          type=sha,prefix=sha-
        # Produces tags like:
        #   yourusername/myapp:main       (branch name)
        #   yourusername/myapp:sha-abc123 (Git commit SHA)
        # Branch tags are mutable (overwritten each push)
        # SHA tags are immutable (unique per commit) — ideal for rollbacks

    # Build and push in a single step
    - uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        # labels: OCI metadata (maintainer, source URL, etc.)
        # Useful for image provenance tracking
SOLUTION
    echo ""
    echo "--- Setup Steps ---"
    cat << 'SOLUTION'
# 1. Go to Docker Hub > Account Settings > Security > New Access Token
# 2. Copy the token
# 3. In GitHub repo > Settings > Secrets and variables > Actions
#    Add: DOCKERHUB_USERNAME = your-username
#    Add: DOCKERHUB_TOKEN = the-access-token

# For GitHub Container Registry (GHCR) instead:
# - Use: registry: ghcr.io
# - Use: username: ${{ github.actor }}
# - Use: password: ${{ secrets.GITHUB_TOKEN }}
# GITHUB_TOKEN is auto-provided — no manual setup needed

# After pushing a commit, verify:
# - Docker Hub: hub.docker.com/r/yourusername/myapp
# - Tags: main, sha-abc1234
SOLUTION
}

# === Exercise 3: Multi-Stage Pipeline with Environment Promotion ===
# Problem: Create separate test and deploy jobs.
exercise_3() {
    echo "=== Exercise 3: Multi-Stage Pipeline with Environment Promotion ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- .github/workflows/pipeline.yml ---"
    cat << 'SOLUTION'
name: Multi-Stage Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Job 1: Test (runs on every push and PR)
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [18, 20]
        # Matrix strategy runs the job N times in parallel,
        # once per combination. Tests on Node 18 AND Node 20
        # to ensure compatibility across versions.
    steps:
    - uses: actions/checkout@v4

    - name: Build test image
      run: docker build --target test -t myapp:test-node${{ matrix.node }} .
      # --target test: build only up to the 'test' stage in a multi-stage Dockerfile
      # This avoids building the production image just to run tests

    - name: Run unit tests
      run: |
        docker run --rm myapp:test-node${{ matrix.node }} npm test
        # Run tests inside the container for reproducible results
        # The container has the exact same deps as production

  # Job 2: Deploy (only runs after test passes, only on main)
  deploy:
    runs-on: ubuntu-latest
    needs: test
    # needs: test — this job only starts AFTER the test job succeeds.
    # If tests fail, deploy is skipped entirely.
    if: github.ref == 'refs/heads/main'
    # Only deploy from main branch.
    # PRs run tests but never deploy — this prevents accidental deployments.

    steps:
    - uses: actions/checkout@v4

    - uses: docker/setup-buildx-action@v3

    - uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: yourusername/myapp:${{ github.sha }}
        # Tag with the full Git SHA for exact traceability:
        # "Which commit is running in production?" → check the image tag
SOLUTION
    echo ""
    echo "--- Behavior ---"
    cat << 'SOLUTION'
# Push to a feature branch:
#   - test job runs (both Node 18 and 20)
#   - deploy job is SKIPPED (not on main)
#   Result: fast feedback on code quality

# Merge PR to main:
#   - test job runs (both Node 18 and 20)
#   - deploy job runs (pushes tagged image to registry)
#   Result: only tested code reaches production

# Key pattern: "test gates deploy"
# The 'needs' keyword enforces this ordering.
# The 'if' condition enforces branch restrictions.
SOLUTION
}

# === Exercise 4: Cache Docker Build Layers ===
# Problem: Use GitHub Actions cache to speed up Docker builds.
exercise_4() {
    echo "=== Exercise 4: Cache Docker Build Layers ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Build step with GHA cache ---"
    cat << 'SOLUTION'
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: yourusername/myapp:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
        # type=gha: use GitHub Actions cache backend
        # mode=max: cache ALL layers (not just the final image layers)
        #
        # How it works:
        # 1. First build: no cache, all layers built from scratch (~60s)
        # 2. Layers are uploaded to GHA cache after build
        # 3. Second build: unchanged layers are pulled from cache (~5s)
        #
        # This is especially impactful for:
        # - npm install / pip install layers (dependencies change rarely)
        # - Base image layers
        # - Multi-stage build intermediate stages
SOLUTION
    echo ""
    echo "--- Experiment ---"
    cat << 'SOLUTION'
# Push 1: Initial build (no cache)
# Workflow duration: ~60 seconds
# All layers built from scratch

# Push 2: Small code change (not dependency change)
# Workflow duration: ~15 seconds
# Dependency layers (npm install) pulled from cache
# Only the COPY . . layer and subsequent layers are rebuilt
# Savings: ~75% faster

# Push 3: --no-cache (bypass cache for comparison)
# Add to build step: no-cache: true
# Workflow duration: ~60 seconds (back to original)
# Proves the cache was providing the speedup

# Cache storage:
# - GHA cache has a 10 GB limit per repository
# - Oldest caches are evicted when the limit is reached
# - Cache is scoped to the branch (PRs use the base branch cache as fallback)

# Alternative caching strategies:
# cache-from: type=registry,ref=yourusername/myapp:buildcache
# cache-to: type=registry,ref=yourusername/myapp:buildcache,mode=max
# Registry-based cache: survives across CI runners, no size limit
SOLUTION
}

# === Exercise 5: GitOps Deployment with ArgoCD ===
# Problem: Connect a Git repository to ArgoCD for declarative CD.
exercise_5() {
    echo "=== Exercise 5: GitOps Deployment with ArgoCD ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Install ArgoCD on minikube
kubectl create namespace argocd
kubectl apply -n argocd -f \
  https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
# ArgoCD runs as a set of Pods in the argocd namespace:
# - argocd-server: UI + API
# - argocd-repo-server: clones Git repos
# - argocd-application-controller: syncs cluster state to Git

# Wait for ArgoCD to be ready
kubectl wait --for=condition=available deployment/argocd-server \
  -n argocd --timeout=120s

# Step 2: Access the ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Open https://localhost:8080 in your browser
# Get the initial admin password:
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d
# Login: username=admin, password=<output above>

# Step 3: Create an ArgoCD Application manifest
cat << 'EOF' > argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/yourusername/k8s-manifests.git
    targetRevision: HEAD
    path: manifests/
    # ArgoCD monitors this path in the Git repo
    # Any change to YAML files here triggers a sync
  destination:
    server: https://kubernetes.default.svc
    namespace: default
    # Apply manifests to the local cluster's default namespace
  syncPolicy:
    automated:
      prune: true       # Delete resources removed from Git
      selfHeal: true    # Revert manual kubectl changes to match Git
      # selfHeal: if someone runs 'kubectl scale' manually,
      # ArgoCD reverts it to the Git-defined state within ~3 minutes
    syncOptions:
    - CreateNamespace=true
EOF

# Step 4: Apply the Application
kubectl apply -f argocd-app.yaml
# ArgoCD immediately clones the repo and applies the manifests

# Step 5: Watch the sync in the ArgoCD UI
# The application should show as "Synced" and "Healthy"
# Each resource (Deployment, Service, etc.) shows its status

# Step 6: Make a change in Git
# Edit manifests/deployment.yaml: change replicas from 1 to 2
# git commit -am "Scale to 2 replicas" && git push

# ArgoCD detects the change within ~3 minutes (default poll interval)
# and automatically syncs: the cluster now has 2 replicas

# GitOps benefits:
# 1. Git is the single source of truth — cluster state = Git state
# 2. Audit trail: every change is a Git commit (who, when, what)
# 3. Self-healing: manual changes are reverted automatically
# 4. Easy rollback: git revert + push = instant cluster rollback
# 5. Multi-cluster: same Git repo can drive multiple clusters
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
