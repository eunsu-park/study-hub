#!/bin/bash
# Exercises for Lesson 12: Security Best Practices
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Scan an Image for Vulnerabilities ===
# Problem: Use Trivy to identify and remediate known CVEs in a container image.
exercise_1() {
    echo "=== Exercise 1: Scan an Image for Vulnerabilities ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Install Trivy
# macOS:
brew install trivy
# Linux (Debian/Ubuntu):
# sudo apt-get install wget apt-transport-https gnupg lsb-release
# wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
# echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | \
#   sudo tee -a /etc/apt/sources.list.d/trivy.list
# sudo apt-get update && sudo apt-get install trivy

# Step 2: Scan an older image with known vulnerabilities
trivy image python:3.8
# OUTPUT (typical):
# python:3.8 (debian 11.6)
# Total: 1234 (UNKNOWN: 2, LOW: 450, MEDIUM: 520, HIGH: 200, CRITICAL: 62)
#
# Key insight: Older base images accumulate CVEs because their OS packages
# are no longer receiving security patches. python:3.8 is end-of-life.

# Step 3: Note the CRITICAL and HIGH counts
# CRITICAL: ~62, HIGH: ~200
# These represent real, exploitable vulnerabilities that attackers target.
# CRITICAL means: remotely exploitable, no authentication required, or
# complete system compromise possible.

# Step 4: Scan a newer Alpine-based variant
trivy image python:3.12-alpine
# OUTPUT (typical):
# python:3.12-alpine (alpine 3.19.0)
# Total: 3 (UNKNOWN: 0, LOW: 1, MEDIUM: 2, HIGH: 0, CRITICAL: 0)
#
# Why the dramatic reduction?
# 1. Alpine uses musl libc (smaller attack surface than glibc)
# 2. Alpine includes ~5MB of packages vs ~120MB for Debian
# 3. python:3.12 has current security patches
# 4. Fewer packages = fewer potential vulnerabilities

# Step 5: Compare results
# python:3.8        → 1234 vulnerabilities (62 CRITICAL, 200 HIGH)
# python:3.12-alpine → 3 vulnerabilities (0 CRITICAL, 0 HIGH)
# Reduction: >99% — this is why base image choice matters enormously.

# Step 6: Write a secure Dockerfile
cat << 'EOF' > Dockerfile
FROM python:3.12-alpine

# Create a non-root user BEFORE copying application code
# This ensures the app never runs with UID 0 (root)
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Change ownership to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user for all subsequent commands
USER appuser

CMD ["python", "app.py"]
EOF

# Step 7: Build and scan the custom image
docker build -t myapp:latest .
trivy image myapp:latest
# Expected: 0 CRITICAL, 0 HIGH — a secure baseline
# Trivy also checks:
# - Application dependencies (pip packages in this case)
# - OS package vulnerabilities
# - Secrets accidentally embedded in the image
SOLUTION
}

# === Exercise 2: Write a Secure Dockerfile ===
# Problem: Apply Dockerfile security best practices to harden an application image.
exercise_2() {
    echo "=== Exercise 2: Write a Secure Dockerfile ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Insecure Dockerfile (given) ---"
    cat << 'SOLUTION'
FROM ubuntu:latest
RUN apt-get update && apt-get install -y curl wget vim python3
COPY . /app
RUN chmod 777 /app
CMD ["python3", "/app/main.py"]
SOLUTION
    echo ""
    echo "--- Security Issues Identified ---"
    cat << 'SOLUTION'
# Issue 1: FROM ubuntu:latest
#   - 'latest' is a mutable tag — it changes with every release
#   - Builds are not reproducible; today's 'latest' differs from yesterday's
#   - ubuntu is a large base (~78MB) with many packages you don't need
#   FIX: Pin the version (ubuntu:22.04) or use a minimal base (python:3.12-alpine)

# Issue 2: apt-get install curl wget vim
#   - Installs unnecessary tools (vim, wget) that expand the attack surface
#   - An attacker who gains shell access can use curl/wget to download malware
#   FIX: Install only what's strictly needed; remove curl/wget after build

# Issue 3: COPY . /app
#   - Copies EVERYTHING including .git/, .env, credentials, test files
#   - Secrets may be embedded in the image and visible via 'docker history'
#   FIX: Use .dockerignore to exclude sensitive/unnecessary files

# Issue 4: chmod 777 /app
#   - World-readable, writable, and executable — maximum exposure
#   - Any process (or attacker) can modify or execute any file
#   FIX: Use restrictive permissions (755 for dirs, 644 for files, 500 for scripts)

# Issue 5: Running as root (implicit)
#   - No USER directive → the container runs as root (UID 0)
#   - A container escape exploit as root = host compromise
#   FIX: Create and switch to a non-root user

# Issue 6: No multi-stage build
#   - Build tools and caches remain in the final image
#   - Larger image, more CVEs, slower pulls
#   FIX: Use multi-stage build to separate build and runtime stages

# Issue 7: apt-get update without cleanup
#   - Package lists cached in the image waste space (~30MB)
#   FIX: Chain update + install + cleanup in a single RUN layer
SOLUTION
    echo ""
    echo "--- Secure Dockerfile (rewritten) ---"
    cat << 'SOLUTION'
# Stage 1: Build stage — install dependencies
FROM python:3.12-alpine AS builder
# python:3.12-alpine: minimal base (~50MB), current patches, pinned version

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# --no-cache-dir: don't store pip cache (saves space)
# --prefix=/install: install to a separate directory for clean COPY later

# Stage 2: Runtime stage — minimal final image
FROM python:3.12-alpine

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

# Copy only the installed dependencies from the build stage
COPY --from=builder /install /usr/local

# Copy application code with correct ownership
COPY --chown=appuser:appgroup main.py .
# --chown: set ownership at COPY time (no separate chmod needed)

# Set restrictive permissions
RUN chmod 500 main.py
# 500 = owner read+execute only — no write, no group/other access

# Switch to non-root user
USER appuser

CMD ["python", "main.py"]
SOLUTION
    echo ""
    echo "--- Comparison ---"
    cat << 'SOLUTION'
# Build both images
docker build -f Dockerfile.insecure -t myapp:insecure .
docker build -f Dockerfile.secure -t myapp:secure .

# Compare image sizes
docker images | grep myapp
# REPOSITORY  TAG        SIZE
# myapp       insecure   ~350MB  (ubuntu + curl + wget + vim + python3)
# myapp       secure     ~55MB   (alpine + python + app only)
# Size reduction: ~85%

# Scan both with Trivy
trivy image myapp:insecure
# Total: ~150 vulnerabilities (CRITICAL: ~5, HIGH: ~20)

trivy image myapp:secure
# Total: ~2 vulnerabilities (CRITICAL: 0, HIGH: 0)
# Vulnerability reduction: >98%

# Key takeaway:
# The secure Dockerfile is not just safer — it's also smaller, faster to pull,
# and has a dramatically reduced attack surface. Security and efficiency align.
SOLUTION
}

# === Exercise 3: Run Containers with Least Privilege ===
# Problem: Apply runtime security controls when starting containers.
exercise_3() {
    echo "=== Exercise 3: Run Containers with Least Privilege ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Run as a non-root user
docker run --rm --user 1000:1000 alpine whoami
# uid=1000 gid=1000
# --user 1000:1000 overrides the container's default user (root)
# Even if the Dockerfile doesn't have a USER directive,
# the container process runs as UID 1000 — not root.
# This is a defense-in-depth measure: if the app is compromised,
# the attacker has limited privileges.

docker run --rm --user 1000:1000 alpine id
# uid=1000 gid=1000 groups=1000
# No supplementary groups — minimal group membership

# Step 2: Read-only root filesystem
docker run --rm --read-only alpine sh -c "echo test > /test.txt"
# sh: can't create /test.txt: Read-only file system
# --read-only mounts the container's root filesystem as read-only.
# An attacker cannot:
# - Drop malware to disk
# - Modify application binaries
# - Create persistence mechanisms (cron jobs, startup scripts)

# Step 3: Read-only with writable tmpfs for /tmp
docker run --rm --read-only --tmpfs /tmp alpine sh -c "echo test > /tmp/test.txt && cat /tmp/test.txt"
# test
# --tmpfs /tmp: mounts a temporary filesystem at /tmp (in-memory, not on disk)
# The app can write temp files, but:
# - /tmp is wiped when the container stops
# - Everything else remains read-only
# - The rest of the filesystem is still protected

# Step 4: Drop all Linux capabilities
docker run --rm --cap-drop ALL alpine ping -c 1 8.8.8.8
# ping: permission denied (are you root?)
# Linux capabilities are fine-grained root privileges.
# --cap-drop ALL removes ALL capabilities, including:
# - CAP_NET_RAW (raw sockets, needed for ping/tcpdump)
# - CAP_CHOWN (change file ownership)
# - CAP_DAC_OVERRIDE (bypass file permission checks)
# - CAP_SYS_ADMIN (mount, namespace operations — very dangerous)
# Without capabilities, even root inside the container is severely limited.

# Step 5: Add back only the required capability
docker run --rm --cap-drop ALL --cap-add NET_RAW alpine ping -c 1 8.8.8.8
# PING 8.8.8.8 (8.8.8.8): 56 data bytes
# 64 bytes from 8.8.8.8: seq=0 ttl=118 time=12.345 ms
# --cap-add NET_RAW: re-enable only the specific capability needed
# Principle: drop everything, then add back the minimum required.
# This is the capability equivalent of "default-deny" network policies.

# Step 6: Combine all three constraints
docker run --rm \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --cap-drop ALL \
  alpine sh -c "id && echo 'Secure container running' > /tmp/msg.txt && cat /tmp/msg.txt"
# uid=1000 gid=1000 groups=1000
# Secure container running
#
# This container runs with:
# 1. Non-root user (UID 1000) — no root privileges
# 2. Read-only filesystem — no persistent modifications
# 3. Writable /tmp only — minimal write surface
# 4. No Linux capabilities — even UID 0 would be powerless
#
# Defense layers:
# Container escape → non-root, so limited host damage
# Malware persistence → read-only FS, nothing survives restart
# Privilege escalation → all capabilities dropped, no setuid
# Lateral movement → no network tools available (no NET_RAW for recon)

# Bonus: Add --security-opt=no-new-privileges to prevent setuid escalation
docker run --rm \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --cap-drop ALL \
  --security-opt=no-new-privileges \
  alpine id
# --security-opt=no-new-privileges: prevents any process from gaining
# privileges through setuid/setgid binaries or file capabilities.
# Even if an attacker finds a setuid binary, it won't escalate.
SOLUTION
}

# === Exercise 4: Manage Secrets Without Embedding Them ===
# Problem: Practice secret injection patterns that avoid storing secrets in images or environment variables.
exercise_4() {
    echo "=== Exercise 4: Manage Secrets Without Embedding Them ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a secret file on the host
echo "supersecret_db_password" > /tmp/db_password.txt
# In production, this file would come from:
# - A secrets manager (HashiCorp Vault, AWS Secrets Manager)
# - A CI/CD pipeline's secret store
# - An encrypted file decrypted at deploy time

# Step 2: Mount the secret at runtime using a bind mount
docker run --rm \
  -v /tmp/db_password.txt:/run/secrets/db_password:ro \
  alpine cat /run/secrets/db_password
# supersecret_db_password
#
# Key points:
# -v host:container:ro — the :ro flag makes the mount read-only
# /run/secrets/ is the conventional path for secrets in containers
# The secret is injected at runtime, NOT baked into the image.
# It exists only in the container's filesystem view — never on a layer.

# Step 3: Confirm the secret is NOT in any image layer
docker history myapp:latest
# The output shows each layer's command. You should see:
# COPY app.py, RUN pip install, etc. — but NO secret content.
# If secrets were in a COPY or RUN command, they'd be in the history.
# 'docker history' is a quick check for accidentally embedded secrets.

# More thorough check:
docker save myapp:latest | tar -tv | head -20
# Lists all layers. You can extract individual layers to inspect their contents.
# If a secret was ADDed or COPYed, it would appear in a layer tar.

# Step 4: Docker Compose secrets pattern
SOLUTION
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
secrets:
  db_password:
    file: ./db_password.txt
    # Compose reads the file from the host at deploy time
    # and mounts it into the container at /run/secrets/<name>

services:
  app:
    image: alpine
    secrets:
      - db_password
      # The secret is mounted at /run/secrets/db_password
      # with permissions 0444 (world-readable inside the container)
      # In production, tighten permissions with the 'mode' option
    command: cat /run/secrets/db_password

  db:
    image: postgres:15-alpine
    secrets:
      - db_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      # PostgreSQL supports _FILE suffix convention:
      # Instead of POSTGRES_PASSWORD=literal_value,
      # POSTGRES_PASSWORD_FILE points to a file containing the password.
      # This avoids exposing the secret via 'docker inspect' env vars.
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Step 5: Run and verify the secret is accessible
docker compose up
# app_1 | supersecret_db_password
# The secret was read from /run/secrets/db_password inside the container.

# Step 6: Verify the secret does NOT appear in environment variables
docker compose up -d
docker inspect $(docker compose ps -q app) --format '{{json .Config.Env}}'
# ["PATH=/usr/local/sbin:/usr/local/bin:..."]
# Notice: db_password is NOT in the Env array.
# Compare with --env approach:
# docker run -e DB_PASSWORD=secret alpine env
# DB_PASSWORD=secret  ← visible in env, inspect, logs, and child processes

# Why volume-mounted secrets are safer than environment variables:
#
# Environment variables:
# ✗ Visible in 'docker inspect'
# ✗ Inherited by ALL child processes (fork/exec)
# ✗ Often dumped in crash reports and debug logs
# ✗ Cannot be updated without container restart
#
# Volume-mounted secrets:
# ✓ Not visible in 'docker inspect' env section
# ✓ NOT inherited by child processes automatically
# ✓ The application must explicitly read the file
# ✓ Can be updated by updating the source file (with Compose restart)
# ✓ Can enforce file permissions (mode: 0400)

# Cleanup
docker compose down
rm /tmp/db_password.txt
SOLUTION
}

# === Exercise 5: Enable Docker Content Trust and Sign an Image ===
# Problem: Use Docker Content Trust to sign and verify container images.
exercise_5() {
    echo "=== Exercise 5: Enable Docker Content Trust ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Enable Docker Content Trust
export DOCKER_CONTENT_TRUST=1
# When DCT is enabled, Docker will:
# - Verify signatures on EVERY pull (reject unsigned images)
# - Sign images on EVERY push (requires signing keys)
# - Refuse to run unsigned images
# This prevents supply-chain attacks where an attacker pushes a
# malicious image to a registry with the same name as a trusted one.

# Step 2: Pull a trusted image (official images are signed by Docker, Inc.)
docker pull nginx:alpine
# Pull complete
# Tagging nginx:alpine
# sha256:abc123...
# The pull succeeds because official images have Notary signatures.
# Docker verified the signature against Docker's root of trust.

# Step 3: Tag a local image for your Docker Hub account
docker tag nginx:alpine yourusername/signed-nginx:latest
# Creates a new tag pointing to the same image layers.
# No network operation — just a local metadata change.

# Step 4: Push the signed image (DCT is still enabled)
docker push yourusername/signed-nginx:latest
# First push with DCT enabled triggers key generation:
#
# You are about to create a new root signing key passphrase.
# Enter passphrase for new root key with ID abc123:
# Enter passphrase for new repository key with ID def456:
#
# Key hierarchy (Notary/TUF framework):
# Root key     → master key, protects all other keys (KEEP OFFLINE)
# Targets key  → per-repository, signs specific image tags
# Snapshot key → tracks the current set of signed tags
# Timestamp key → managed by the registry (proves freshness)
#
# After entering passphrases, the image is pushed and signed.
# The signature is stored in a Notary server (Docker Hub hosts one).

# Step 5: Pull the signed image (verify the signature)
docker pull yourusername/signed-nginx:latest
# Pull (1 of 1): yourusername/signed-nginx:latest@sha256:abc123...
# The pull succeeds because:
# 1. The image has a valid Notary signature
# 2. The signature was created by a trusted key
# 3. The content digest matches the signed digest

# Step 6: Attempt to pull an unsigned image with DCT disabled
DOCKER_CONTENT_TRUST=0 docker pull someuser/unsigned-image:latest
# Pull succeeds — DCT is disabled, so no signature check.
# Without DCT, Docker trusts whatever the registry serves.

# Re-enable and try again:
DOCKER_CONTENT_TRUST=1
docker pull someuser/unsigned-image:latest 2>&1
# Error: remote trust data does not exist for someuser/unsigned-image
# REJECTED — the image has no Notary signature.
# DCT prevents pulling unsigned or tampered images.

# Step 7: What Docker Content Trust protects against

# PROTECTS AGAINST:
# 1. Image tampering: registry compromise can't serve modified images
#    (the content hash is signed; any modification invalidates the signature)
# 2. Replay attacks: timestamp key ensures you get the latest signed version
#    (an attacker can't serve an older, vulnerable version)
# 3. Key compromise (partial): if a targets key is compromised,
#    the root key can revoke it and issue a new one
# 4. Man-in-the-middle: signatures are verified client-side

# LIMITATIONS:
# 1. Only protects pull/push — does not scan for vulnerabilities
#    (a signed image can still contain CVEs)
# 2. Root key loss is catastrophic — if lost, you cannot sign new tags
#    for existing repositories (must create new repos)
# 3. Key management burden — passphrases must be securely stored
#    (use a hardware security module or secrets manager in production)
# 4. Not all registries support Notary — only Docker Hub and some
#    enterprise registries (Harbor, GitLab) have Notary integration
# 5. Tag-based, not digest-based — signing is per tag, so the same
#    digest can be unsigned under a different tag

# Best practice:
# 1. Always enable DCT in CI/CD pipelines: export DOCKER_CONTENT_TRUST=1
# 2. Store root keys OFFLINE (USB drive in a safe)
# 3. Use delegation keys for CI/CD (not the root key)
# 4. Combine DCT with image scanning (Trivy) for defense in depth:
#    - DCT ensures the image hasn't been tampered with (integrity)
#    - Trivy ensures the image doesn't have known CVEs (vulnerability)
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
