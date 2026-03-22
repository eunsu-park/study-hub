# Podman and OCI

**Previous**: [Multi-Stage Build Patterns](./14_Multi_Stage_Build_Patterns.md) | **Next**: [Container Debugging](./16_Container_Debugging.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain OCI (Open Container Initiative) standards and their role in the container ecosystem
2. Describe how Podman's daemonless, rootless architecture differs from Docker
3. Use Podman CLI commands as drop-in replacements for Docker commands
4. Build container images with Buildah and manage them with Skopeo
5. Create and manage Podman pods as a lightweight Kubernetes-like abstraction
6. Integrate Podman containers with systemd for production service management
7. Plan and execute a migration from Docker to Podman

## Table of Contents
1. [OCI Standards](#1-oci-standards)
2. [Podman Architecture](#2-podman-architecture)
3. [Podman CLI Compatibility](#3-podman-cli-compatibility)
4. [Buildah for Image Building](#4-buildah-for-image-building)
5. [Skopeo for Image Management](#5-skopeo-for-image-management)
6. [Podman Pods](#6-podman-pods)
7. [Systemd Integration](#7-systemd-integration)
8. [Migration from Docker to Podman](#8-migration-from-docker-to-podman)
9. [Podman Compose and Kubernetes](#9-podman-compose-and-kubernetes)
10. [Practice Exercises](#10-practice-exercises)

**Difficulty**: ⭐⭐⭐

---

Docker popularized containers, but the ecosystem has evolved beyond any single tool. The Open Container Initiative (OCI) established open standards for container formats and runtimes, enabling alternatives like Podman, Buildah, and Skopeo. Podman's daemonless, rootless design addresses fundamental security concerns with Docker's privileged daemon model, making it particularly attractive for enterprise and security-sensitive environments.

---

## 1. OCI Standards

### What Is OCI?

The Open Container Initiative, founded in 2015 under the Linux Foundation, defines three core specifications:

```
┌──────────────────────────────────────────────────────────────┐
│                    OCI Specifications                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Image Spec (image-spec)                                   │
│     ┌─────────────────────────────────────────────┐          │
│     │ Defines how container images are structured  │          │
│     │ • Image manifest                             │          │
│     │ • Image index (multi-arch)                   │          │
│     │ • Filesystem layers (tar+gzip)               │          │
│     │ • Image configuration (env, cmd, etc.)       │          │
│     └─────────────────────────────────────────────┘          │
│                                                               │
│  2. Runtime Spec (runtime-spec)                               │
│     ┌─────────────────────────────────────────────┐          │
│     │ Defines how to run a container               │          │
│     │ • Container lifecycle (create/start/stop)    │          │
│     │ • Configuration format (config.json)         │          │
│     │ • Linux-specific: namespaces, cgroups, caps  │          │
│     └─────────────────────────────────────────────┘          │
│                                                               │
│  3. Distribution Spec (distribution-spec)                     │
│     ┌─────────────────────────────────────────────┐          │
│     │ Defines how to distribute container images   │          │
│     │ • Push/pull operations                       │          │
│     │ • Registry API (HTTP-based)                  │          │
│     │ • Content discovery and resolution           │          │
│     └─────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### OCI-Compliant Tools

| Tool | Purpose | OCI Compliant |
|---|---|---|
| Docker | Build, run, distribute | Yes |
| Podman | Build, run | Yes |
| Buildah | Build images | Yes |
| Skopeo | Copy/inspect images | Yes |
| containerd | Container runtime | Yes |
| CRI-O | Kubernetes runtime | Yes |
| runc | Low-level runtime | Reference implementation |

### Why OCI Matters

```bash
# An image built with Docker works with Podman (and vice versa)
docker build -t myapp .
docker save myapp -o myapp.tar

# Load into Podman
podman load -i myapp.tar
podman run myapp

# Push to any OCI-compliant registry
podman push myapp docker.io/myuser/myapp:latest
```

---

## 2. Podman Architecture

### Docker vs Podman Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Docker Architecture                                          │
│                                                               │
│  User ──► docker CLI ──► Docker Daemon (dockerd) ──► containerd
│                              │ (runs as root)         │       │
│                              │                     ┌──┴──┐   │
│                              │                     │runc │   │
│                              │                     └──┬──┘   │
│                              │                        │      │
│                              ▼                        ▼      │
│                         Container A              Container B  │
│                                                               │
│  ⚠ Single point of failure: daemon crash kills all containers │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Podman Architecture (daemonless)                             │
│                                                               │
│  User ──► podman CLI ──► conmon ──► runc ──► Container        │
│              │              │                                  │
│              │    (each container gets its own conmon process) │
│              │                                                 │
│  User ──► podman CLI ──► conmon ──► runc ──► Container        │
│                                                               │
│  ✓ No daemon: containers run as direct child processes        │
│  ✓ No single point of failure                                 │
│  ✓ Rootless by default                                        │
└──────────────────────────────────────────────────────────────┘
```

### Key Differences

| Feature | Docker | Podman |
|---|---|---|
| Daemon | Required (dockerd) | No daemon |
| Root required | Yes (daemon runs as root) | No (rootless default) |
| Socket | `/var/run/docker.sock` | Per-user socket |
| Container process parent | dockerd | conmon (per container) |
| Pods (Kubernetes-like) | No native support | First-class pods |
| Systemd integration | Separate unit files | `podman generate systemd` |
| Docker Compose | Native | Via podman-compose |
| Container restart on reboot | Via daemon auto-start | Via systemd units |

### Rootless Containers

Podman runs containers without any root privileges by leveraging user namespaces:

```bash
# Check rootless setup
podman info --format '{{.Host.Security.Rootless}}'
# true

# User namespace mapping
podman unshare cat /proc/self/uid_map
#     0    1000       1
#     1  100000   65536

# Rootless containers cannot bind to ports < 1024 by default
podman run -p 8080:80 nginx   # Works
podman run -p 80:80 nginx     # Fails (unless configured)

# Allow low ports for rootless
sudo sysctl net.ipv4.ip_unprivileged_port_start=80
```

---

## 3. Podman CLI Compatibility

### Drop-In Replacement

Podman was designed as a CLI-compatible replacement for Docker:

```bash
# These commands work identically in both Docker and Podman
podman pull nginx:alpine
podman run -d --name web -p 8080:80 nginx:alpine
podman ps
podman logs web
podman exec -it web sh
podman stop web
podman rm web

# Common alias for migration
alias docker=podman
```

### Container Management

```bash
# Run a container
podman run -d --name myapp \
  -p 8080:8080 \
  -v mydata:/data \
  -e DB_HOST=localhost \
  myapp:latest

# List containers (running and stopped)
podman ps -a

# Container resource stats
podman stats --no-stream

# Container top (process listing)
podman top myapp

# Copy files
podman cp myapp:/app/config.json ./config.json
podman cp ./newconfig.json myapp:/app/config.json
```

### Image Management

```bash
# Build an image
podman build -t myapp:latest .

# List images
podman images

# Tag and push
podman tag myapp:latest docker.io/myuser/myapp:latest
podman push docker.io/myuser/myapp:latest

# Image history
podman history myapp:latest

# Remove unused images
podman image prune -a
```

### Differences to Watch For

```bash
# Docker-specific features NOT in Podman:
# 1. Docker Swarm (use Kubernetes instead)
# 2. docker-compose (use podman-compose or podman play kube)

# Podman-specific features NOT in Docker:
# 1. Pods (podman pod create)
# 2. Generate systemd units (podman generate systemd)
# 3. Generate Kubernetes YAML (podman generate kube)
# 4. Rootless by default

# Registry handling difference
# Docker defaults to docker.io; Podman uses unqualified-search-registries
# Configure in /etc/containers/registries.conf
```

```ini
# /etc/containers/registries.conf
unqualified-search-registries = ["docker.io", "quay.io", "ghcr.io"]
```

---

## 4. Buildah for Image Building

### Why Buildah?

Buildah is a specialized tool for building OCI images. It does not require a daemon and can build images without a Dockerfile.

```
┌──────────────────────────────────────────────────────────────┐
│               Buildah vs Docker Build                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Docker Build:                                                │
│  • Requires daemon                                            │
│  • Uses Dockerfile only                                       │
│  • Builds complete image                                      │
│                                                               │
│  Buildah:                                                     │
│  • Daemonless                                                 │
│  • Dockerfile OR scripted builds                              │
│  • Fine-grained layer control                                 │
│  • Can mount host filesystem into build                       │
│  • Can commit any container to an image                       │
└──────────────────────────────────────────────────────────────┘
```

### Building with Dockerfile

```bash
# Buildah supports standard Dockerfiles
buildah bud -t myapp:latest .

# Same as:
buildah build -t myapp:latest .

# With build arguments
buildah build --build-arg VERSION=1.0 -t myapp:1.0 .
```

### Scripted Builds (No Dockerfile)

```bash
#!/bin/bash
# build.sh -- Build an image without a Dockerfile

# Create a new container from base image
container=$(buildah from python:3.12-slim)

# Run commands inside the container
buildah run $container pip install flask gunicorn

# Copy files into the container
buildah copy $container ./app /app

# Set configuration
buildah config --workingdir /app $container
buildah config --port 8000 $container
buildah config --cmd '["gunicorn", "app:app", "-b", "0.0.0.0:8000"]' $container
buildah config --label maintainer="dev@example.com" $container

# Commit the container to an image
buildah commit $container myapp:latest

# Clean up
buildah rm $container
```

### Buildah Mount (Host Integration)

```bash
# Mount a container's filesystem to the host
container=$(buildah from fedora)
mountpoint=$(buildah mount $container)

# Now you can use host tools on the container's filesystem
dnf install --installroot $mountpoint --releasever 39 python3 -y

# Unmount and commit
buildah unmount $container
buildah commit $container my-fedora-python
```

---

## 5. Skopeo for Image Management

### Image Inspection

```bash
# Inspect a remote image without pulling it
skopeo inspect docker://docker.io/library/nginx:alpine

# Get image digest
skopeo inspect --format '{{.Digest}}' docker://nginx:alpine

# List tags for a repository
skopeo list-tags docker://docker.io/library/python

# Inspect a local image
skopeo inspect containers-storage:localhost/myapp:latest
```

### Image Copying

```bash
# Copy between registries (no local storage needed)
skopeo copy \
  docker://docker.io/library/nginx:alpine \
  docker://myregistry.example.com/nginx:alpine

# Copy to a local directory (OCI layout)
skopeo copy \
  docker://nginx:alpine \
  oci:/tmp/nginx-oci:alpine

# Copy to a Docker archive (tar file)
skopeo copy \
  docker://nginx:alpine \
  docker-archive:/tmp/nginx.tar:nginx:alpine

# Copy from a local Podman image to a registry
skopeo copy \
  containers-storage:localhost/myapp:latest \
  docker://myregistry.example.com/myapp:latest
```

### Image Synchronization

```bash
# Sync all tags of an image to a local directory
skopeo sync --src docker --dest dir \
  docker.io/library/python /tmp/python-mirror

# Sync from a directory to a private registry
skopeo sync --src dir --dest docker \
  /tmp/python-mirror myregistry.example.com/mirror

# Useful for air-gapped environments
```

### Image Deletion

```bash
# Delete an image from a registry
skopeo delete docker://myregistry.example.com/myapp:old-tag
```

---

## 6. Podman Pods

### What Are Pods?

A pod is a group of containers that share network, PID, and IPC namespaces -- the same concept as Kubernetes pods.

```
┌──────────────────────────────────────────────────────────────┐
│                        Podman Pod                             │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Shared Namespaces: network, IPC, (optional) PID      │   │
│  │  Shared localhost (127.0.0.1)                          │   │
│  │                                                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ infra    │  │ app      │  │ sidecar  │            │   │
│  │  │ (pause)  │  │ (nginx)  │  │ (logging)│            │   │
│  │  │          │  │ :80      │  │ :9090    │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  Port mapping is on the pod level (via infra container)       │
│  Containers communicate via localhost                         │
└──────────────────────────────────────────────────────────────┘
```

### Creating and Managing Pods

```bash
# Create a pod with published ports
podman pod create --name webapp \
  -p 8080:80 \
  -p 5432:5432

# Add containers to the pod
podman run -d --pod webapp \
  --name web \
  nginx:alpine

podman run -d --pod webapp \
  --name db \
  -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

# The web container can reach postgres at localhost:5432
# External access via host:8080 (nginx) and host:5432 (postgres)

# List pods
podman pod ls

# Pod details
podman pod inspect webapp

# Stop/start/restart the entire pod
podman pod stop webapp
podman pod start webapp
podman pod restart webapp

# Remove a pod and all its containers
podman pod rm -f webapp
```

### Pods vs Docker Compose

```
┌──────────────────────────────────────────────────────────┐
│          Podman Pods vs Docker Compose                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Docker Compose:                                          │
│  • Containers share a bridge network                      │
│  • Service discovery via DNS names                        │
│  • Each container has its own IP                          │
│  • Port mapping per container                             │
│                                                           │
│  Podman Pods:                                             │
│  • Containers share localhost (127.0.0.1)                 │
│  • Communication via localhost:port                       │
│  • Single IP for the pod                                  │
│  • Port mapping on the pod (via infra container)          │
│  • Closer to Kubernetes pod model                         │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Systemd Integration

### Generating Systemd Units

Podman can generate systemd service files for containers and pods:

```bash
# Generate a systemd unit for a container
podman generate systemd --new --name webapp > webapp.service

# Generate with additional options
podman generate systemd --new --name webapp \
  --restart-policy=always \
  --time 30 \
  > webapp.service

# Generate for an entire pod
podman generate systemd --new --name mypod --files
# Creates: pod-mypod.service, container-web.service, container-db.service
```

### Installing User-Level Services (Rootless)

```bash
# Create systemd user directory
mkdir -p ~/.config/systemd/user

# Generate and install the service
podman generate systemd --new --name webapp \
  > ~/.config/systemd/user/webapp.service

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now webapp.service

# Check status
systemctl --user status webapp.service

# Enable lingering (keep running after logout)
loginctl enable-linger $USER
```

### Installing System-Level Services (Root)

```bash
# Generate as root
sudo podman generate systemd --new --name webapp \
  > /etc/systemd/system/webapp.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now webapp.service
```

### Quadlet (Podman 4.4+)

Quadlet provides a declarative way to define Podman containers as systemd units:

```ini
# ~/.config/containers/systemd/webapp.container
[Container]
Image=docker.io/library/nginx:alpine
PublishPort=8080:80
Volume=webdata.volume:/usr/share/nginx/html:ro

[Service]
Restart=always

[Install]
WantedBy=default.target
```

```ini
# ~/.config/containers/systemd/webdata.volume
[Volume]
Label=app=webapp
```

```bash
# Reload and start
systemctl --user daemon-reload
systemctl --user start webapp.service
```

---

## 8. Migration from Docker to Podman

### Step-by-Step Migration

```
┌──────────────────────────────────────────────────────────────┐
│              Docker to Podman Migration Path                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Phase 1: Evaluation                                          │
│  ├─ Inventory Docker usage (images, volumes, networks)        │
│  ├─ Identify Docker-specific features (Swarm, etc.)           │
│  └─ Test Podman with existing Dockerfiles                     │
│                                                               │
│  Phase 2: Coexistence                                         │
│  ├─ Install Podman alongside Docker                           │
│  ├─ alias docker=podman for testing                           │
│  └─ Run dev/test workloads on Podman                          │
│                                                               │
│  Phase 3: Migration                                           │
│  ├─ Export Docker images → Import to Podman                   │
│  ├─ Convert docker-compose.yml → pod YAML                     │
│  ├─ Replace Docker systemd units → Podman systemd units       │
│  └─ Update CI/CD pipelines                                    │
│                                                               │
│  Phase 4: Cleanup                                             │
│  ├─ Remove Docker daemon                                      │
│  ├─ Remove docker.sock dependencies                           │
│  └─ Document Podman-specific workflows                        │
└──────────────────────────────────────────────────────────────┘
```

### Migrating Images

```bash
# Export from Docker
docker save myapp:latest -o myapp.tar

# Import to Podman
podman load -i myapp.tar

# Or copy directly using Skopeo
skopeo copy \
  docker-daemon:myapp:latest \
  containers-storage:myapp:latest
```

### Migrating Volumes

```bash
# Export Docker volume data
docker run --rm -v mydata:/source:ro -v $(pwd):/backup \
  alpine tar czf /backup/mydata.tar.gz -C /source .

# Create Podman volume and restore
podman volume create mydata
podman run --rm -v mydata:/target -v $(pwd):/backup:ro \
  alpine sh -c "cd /target && tar xzf /backup/mydata.tar.gz"
```

### Migrating Docker Compose

```bash
# Option 1: podman-compose (Python, drop-in replacement)
pip install podman-compose
podman-compose up -d

# Option 2: Use Podman's built-in compose support (Podman 3.0+)
podman compose up -d

# Option 3: Convert to Kubernetes YAML
podman generate kube mypod > pod.yaml
podman play kube pod.yaml
```

---

## 9. Podman Compose and Kubernetes

### Podman Generate Kube

Convert running pods/containers to Kubernetes-compatible YAML:

```bash
# Generate Kubernetes YAML from a pod
podman generate kube webapp > webapp-pod.yaml

# Generate with service definition
podman generate kube webapp -s > webapp-with-service.yaml
```

```yaml
# Generated webapp-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: webapp
  name: webapp
spec:
  containers:
    - name: web
      image: docker.io/library/nginx:alpine
      ports:
        - containerPort: 80
          hostPort: 8080
    - name: db
      image: docker.io/library/postgres:16-alpine
      env:
        - name: POSTGRES_PASSWORD
          value: secret
```

### Podman Play Kube

Deploy Kubernetes YAML files directly with Podman:

```bash
# Deploy a Kubernetes YAML
podman play kube webapp-pod.yaml

# With volume creation
podman play kube --build webapp-pod.yaml

# Tear down
podman play kube --down webapp-pod.yaml

# Update (delete and recreate)
podman play kube --replace webapp-pod.yaml
```

This enables a workflow where you develop locally with Podman and deploy to Kubernetes with the same YAML definitions.

---

## 10. Practice Exercises

### Exercise 1: Podman Basics (Beginner)

Run an nginx container with Podman, verify it works, and clean up.

```bash
# 1. Pull the nginx:alpine image with Podman
# 2. Run it with port 8080 mapped to 80
# 3. Verify with curl
# 4. Stop and remove the container
```

<details>
<summary>Solution</summary>

```bash
podman pull nginx:alpine
podman run -d --name web -p 8080:80 nginx:alpine
curl http://localhost:8080
podman stop web
podman rm web
```

</details>

### Exercise 2: Pod Creation (Intermediate)

Create a Podman pod with a Python Flask app and Redis, where the Flask app connects to Redis via localhost.

<details>
<summary>Solution</summary>

```bash
# Create a pod
podman pod create --name flask-redis -p 5000:5000

# Add Redis
podman run -d --pod flask-redis --name redis redis:alpine

# Create a simple Flask app
mkdir /tmp/flask-app
cat > /tmp/flask-app/app.py << 'PYEOF'
from flask import Flask
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379)

@app.route('/')
def hello():
    count = r.incr('hits')
    return f'Hello! This page has been visited {count} times.\n'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
PYEOF

cat > /tmp/flask-app/requirements.txt << 'EOF'
flask
redis
EOF

cat > /tmp/flask-app/Dockerfile << 'EOF'
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
EOF

# Build and run
podman build -t flask-app /tmp/flask-app
podman run -d --pod flask-redis --name app flask-app

# Test
curl http://localhost:5000

# Cleanup
podman pod rm -f flask-redis
```

</details>

### Exercise 3: Buildah Scripted Build (Intermediate)

Use Buildah (without a Dockerfile) to create an image that serves a static HTML page with nginx.

<details>
<summary>Solution</summary>

```bash
#!/bin/bash
# Create HTML content
mkdir -p /tmp/mysite
echo "<h1>Built with Buildah!</h1>" > /tmp/mysite/index.html

# Buildah scripted build
ctr=$(buildah from nginx:alpine)
buildah copy $ctr /tmp/mysite/index.html /usr/share/nginx/html/index.html
buildah config --port 80 $ctr
buildah config --label maintainer="student@example.com" $ctr
buildah commit $ctr mysite:latest
buildah rm $ctr

# Run with Podman
podman run -d --name mysite -p 8080:80 mysite:latest
curl http://localhost:8080
podman rm -f mysite
```

</details>

### Exercise 4: Skopeo and Migration (Advanced)

Use Skopeo to inspect a remote image, copy it to a local OCI directory, and then load it into Podman.

<details>
<summary>Solution</summary>

```bash
# Inspect remote image
skopeo inspect docker://docker.io/library/alpine:3.19

# Copy to local OCI directory
skopeo copy docker://alpine:3.19 oci:/tmp/alpine-oci:3.19

# Inspect the OCI layout
ls -la /tmp/alpine-oci/

# Copy from OCI directory to Podman storage
skopeo copy oci:/tmp/alpine-oci:3.19 containers-storage:alpine-local:3.19

# Verify
podman images alpine-local
podman run --rm alpine-local:3.19 cat /etc/os-release

# Cleanup
podman rmi alpine-local:3.19
rm -rf /tmp/alpine-oci
```

</details>

### Exercise 5: Systemd Service (Advanced)

Create a Podman container for a web application and configure it as a rootless systemd service that starts on boot and restarts on failure.

<details>
<summary>Solution</summary>

```bash
# Run a container
podman run -d --name webapp -p 8080:80 nginx:alpine

# Generate systemd unit
mkdir -p ~/.config/systemd/user
podman generate systemd --new --name webapp \
  --restart-policy=always \
  > ~/.config/systemd/user/webapp.service

# Stop the manually created container
podman stop webapp
podman rm webapp

# Enable the systemd service
systemctl --user daemon-reload
systemctl --user enable --now webapp.service

# Verify
systemctl --user status webapp.service
curl http://localhost:8080

# Enable lingering for boot start
loginctl enable-linger $USER

# Cleanup
systemctl --user disable --now webapp.service
rm ~/.config/systemd/user/webapp.service
systemctl --user daemon-reload
```

</details>

---

**Previous**: [Multi-Stage Build Patterns](./14_Multi_Stage_Build_Patterns.md) | **Next**: [Container Debugging](./16_Container_Debugging.md)
