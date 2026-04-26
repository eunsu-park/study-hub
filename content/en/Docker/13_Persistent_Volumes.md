# Persistent Volumes

**Previous**: [Security Best Practices](./12_Security_Best_Practices.md) | **Next**: [Multi-Stage Build Patterns](./14_Multi_Stage_Build_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between Docker volumes, bind mounts, and tmpfs mounts and choose the right option
2. Create and manage named volumes and anonymous volumes
3. Use volume drivers and plugins for remote and cloud-backed storage
4. Implement data backup and restore strategies for containerized applications
5. Share volumes between multiple containers safely
6. Apply storage best practices for databases such as PostgreSQL, MySQL, and MongoDB
7. Use volume inspection, pruning, and lifecycle management commands

## Table of Contents

Before the volume reference, read [**Theory & Principles**](#theory--principles) — how the kernel mounts each storage type (bind mount, named volume, tmpfs), how volume drivers extend this to network/cloud backends, and how Kubernetes's PV/PVC/StorageClass binding maps to the same underlying primitives.

1. [Docker Storage Overview](#1-docker-storage-overview)
2. [Volumes vs Bind Mounts vs tmpfs](#2-volumes-vs-bind-mounts-vs-tmpfs)
3. [Named Volumes and Anonymous Volumes](#3-named-volumes-and-anonymous-volumes)
4. [Volume Drivers and Plugins](#4-volume-drivers-and-plugins)
5. [Volume Lifecycle Management](#5-volume-lifecycle-management)
6. [Data Backup and Restore](#6-data-backup-and-restore)
7. [Volume Sharing Between Containers](#7-volume-sharing-between-containers)
8. [Database Storage Best Practices](#8-database-storage-best-practices)
9. [Volume Commands Reference](#9-volume-commands-reference)
10. [Practice Exercises](#10-practice-exercises)

**Difficulty**: ⭐⭐⭐

---

Containers are ephemeral by design -- when a container is removed, all data written inside its writable layer disappears. Persistent volumes solve this fundamental problem by decoupling data from the container lifecycle. Understanding Docker's storage subsystem is critical for running stateful workloads like databases, message queues, and file-based applications in production.

---

## Theory & Principles

Volumes look conceptually simple — "data that survives the container" — but the implementation has real depth. Each storage type uses the kernel's mount machinery differently, the driver layer abstracts over local and remote backends, and Kubernetes adds another tier of indirection (PV / PVC / StorageClass) on top. The recurring issue with stateful containers is choosing the storage type whose semantics match what your application expects from a "filesystem."

### A. The Three Mount Types and Their Linux Semantics

Docker exposes three primitives for getting non-ephemeral storage into a container:

| Type | What it actually is | Where data lives | Survives `docker rm`? |
|------|---------------------|------------------|----------------------|
| **bind mount** | A `mount --bind` from a host path into the container's mount namespace | Wherever you specified on the host | Yes (it was never owned by Docker) |
| **named volume** | A directory under `/var/lib/docker/volumes/<name>/_data`, bind-mounted in | Inside Docker's data root | Yes (Docker owns it; explicit `docker volume rm` to delete) |
| **tmpfs mount** | A `mount -t tmpfs` of an in-memory filesystem | RAM, never on disk | No (gone the moment the container exits) |

All three end up as entries in the container's mount namespace, visible inside as a normal directory at the requested mount point. The kernel does not distinguish them at runtime — it is just a mount. The difference is *who manages the underlying storage*: you (bind), Docker (volume), the kernel page cache (tmpfs).

The semantic differences that matter:

- **Bind mounts inherit host filesystem semantics exactly.** If your host filesystem is ext4 you get ext4 semantics; if it is the macOS bind-mount-over-VirtioFS in Docker Desktop you get a thin shim with all of that shim's quirks (slow stats, broken file locking, occasional inotify weirdness). This is why Postgres on a macOS bind mount is a bug factory.
- **Named volumes always live on the Docker host's native filesystem.** Even on Docker Desktop, a named volume sits inside the Linux VM, not on the macOS host. Performance and POSIX-correctness match a real Linux filesystem. This is why named volumes are the default recommendation for databases.
- **tmpfs is fast and ephemeral.** Reads/writes hit RAM; latency is microseconds; cap is the smaller of `--tmpfs size=` and host RAM. Useful for `/tmp`, scratch space, secret material that should never touch disk.

Kubernetes mirrors this trichotomy as `hostPath` (= bind), `emptyDir` (= tmpfs or per-Pod ephemeral disk), and `PersistentVolume` (= named volume but pluggable to network/cloud backends).

### B. Volume Drivers: Local, Network, Cloud

A Docker volume is created with a *driver*. The default driver is `local` — the directory under `/var/lib/docker/volumes/...` described above. The driver interface is pluggable: `docker volume create --driver <name>` invokes a registered plugin to provision storage.

Common drivers and their substrates:

| Driver | Backed by | Typical use |
|--------|-----------|-------------|
| `local` | Local filesystem under Docker root | Default; single-host stateful apps |
| `nfs` (built into `local` with options, or as a plugin) | NFS server | Multi-host shared filesystem; classic enterprise NAS |
| `cifs` / `smb` | SMB share | Windows file shares |
| `rexray`, `convoy`, `flocker` (older) | Cloud block storage (EBS, GCE PD), or storage backends like Ceph | Multi-host orchestration with detach/reattach |
| Cloud-native CSI drivers | EBS, EFS, Azure Disk, GCE PD, Cinder, ... | Kubernetes-managed cloud storage |

The plugin contract is small — `Create`, `Remove`, `Mount`, `Unmount`, `Path`, `Get`, `List`, `Capabilities` — implemented as a Unix socket the daemon talks to. This is why Docker can talk to dozens of storage backends through one CLI.

The local driver's `nfs` mode is worth knowing about: `docker volume create --driver local --opt type=nfs --opt o=addr=10.0.0.5,rw --opt device=:/exports/data myvol` creates a "volume" that is actually an NFS mount. The container sees a normal directory; the kernel routes its reads and writes to the NFS server.

### C. Filesystem Semantics and Why They Bite

Most applications assume their filesystem behaves like a real ext4 or xfs. When mounted storage doesn't match, things break in subtle ways:

- **Locking.** SQLite, Postgres, MySQL all rely on `flock` / `fcntl` advisory locking. NFS v3 does not implement them correctly without `lockd`; SMB has its own locking model; macOS bind mounts in Docker Desktop forward locks through the Linux VM's VirtioFS layer with edge cases. A "database mysteriously corrupted" symptom often traces back to broken locking on the underlying mount.
- **`fsync` durability.** Databases call `fsync` to ensure writes hit stable storage. tmpfs returns instantly with no actual durability (data is in RAM). Some network filesystems lie about fsync to look fast. Putting a database on tmpfs makes it fast and useless after a crash.
- **Atomic rename.** Many applications write to `file.tmp` then `rename(file.tmp, file)` for atomic replacement. POSIX guarantees this within a filesystem; it does *not* guarantee it across mount points (and the rename will EXDEV-fail). Watch for this when bind-mounting deep into a container's tree.
- **inotify.** File-watching tools (development hot reload, log tailers) use `inotify` to be notified of changes. NFS, FUSE, and some bind-mount layers do not propagate inotify events correctly. Symptom — your dev container doesn't notice your save.
- **Permissions.** A bind-mounted host directory has host UIDs; if the container's process runs as a different UID, it cannot write. Solutions — make the host directory world-writable (bad), `chown` it to the container's UID (better), or use a volume (best — Docker manages permissions).

### D. Kubernetes PV / PVC / StorageClass: The Same Idea, Decoupled

Kubernetes splits storage into three resources to separate concerns between cluster admin and application developer:

- **PersistentVolume (PV)** — an actual chunk of storage that exists. A specific EBS volume, an NFS export, a Ceph image. Cluster-scoped.
- **PersistentVolumeClaim (PVC)** — a request for storage with required attributes (size, access mode, storage class). Namespaced; written by app developers.
- **StorageClass** — a template for *dynamic provisioning*. When a PVC asks for storage class `fast-ssd`, Kubernetes calls the storage provisioner registered for that class to create a fitting PV on demand.

The matchmaking algorithm:

1. PVC is created with `requests: storage: 10Gi` and `storageClassName: fast-ssd`.
2. The PV controller looks for an existing unclaimed PV that matches.
3. If found, bind PVC to PV.
4. If not found, look up StorageClass `fast-ssd`, find its provisioner (e.g. `ebs.csi.aws.com`), invoke it to create a 10Gi EBS volume, register a corresponding PV, then bind.
5. The Pod referencing the PVC gets the volume mounted via the kubelet calling the CSI driver's `NodeStageVolume` and `NodePublishVolume` hooks.

`accessModes` constrain how a PVC can be used:

- `ReadWriteOnce` (RWO) — single node read-write. Most cloud block storage. Forces single-node Pod placement.
- `ReadOnlyMany` (ROX) — multiple nodes read-only.
- `ReadWriteMany` (RWX) — multiple nodes read-write. Requires NFS, EFS, CephFS, or similar.
- `ReadWriteOncePod` (RWOP, newer) — single Pod (not just single node).

**Reclaim policy** decides what happens when a PVC is deleted: `Retain` (PV keeps data, admin must clean), `Delete` (provisioner deletes the PV and its underlying storage), `Recycle` (deprecated).

CSI (Container Storage Interface) is the standard plugin API. Every cloud and storage vendor ships a CSI driver, K8s talks to all of them through the same interface, and snapshots/clones/online resize are CSI features your driver may or may not support.

### E. Volume Sharing and Concurrency

A volume mounted into multiple containers is a *shared filesystem*. The same locking and concurrency rules apply as on a real shared FS:

- **Two writers without coordination → data corruption.** This is not Docker's problem to solve; it's POSIX. If you mount `/data` into both `app-1` and `app-2` and both write to the same files without locking or partitioning, expect corruption.
- **Reader/writer pattern.** One container writes, others read. Common for log aggregation, generated assets, configuration distribution. Works fine.
- **Producer/consumer with a queue.** Use a real queue (Redis, RabbitMQ) instead of a shared filesystem. Filesystems are bad message queues.

In Kubernetes, RWX volumes naturally support multi-Pod sharing; RWO does not (the scheduler refuses to place a second Pod on a different node). This is the most common reason a Deployment with `replicas: 3` and an RWO PVC ends up with all three Pods stuck Pending — only one can mount.

### From Theory to the Volume CLI Below

- **`docker volume create`, `docker volume ls`, `docker volume rm`, `docker volume inspect`** — the management interface for §A's named volumes (driver = local by default).
- **`-v /host:/container`, `--mount type=bind,source=/host,target=/container`** — the bind-mount syntax of §A; `:ro` for read-only, `:Z` / `:z` for SELinux relabeling.
- **`-v vol-name:/container`, `--mount type=volume,source=vol-name,target=/container`** — named volume syntax.
- **`--mount type=tmpfs,destination=/tmp,tmpfs-size=64m`** — tmpfs mount.
- **`docker volume create --driver nfs --opt ...`** — §B's driver-mediated provisioning.
- **`docker volume prune`** — garbage-collect unused (un-referenced) volumes; useful when CI churn leaves dozens of dead volumes around.
- **Compose `volumes:` top-level + service `volumes:`** — the named-volume model in declarative form.
- **Kubernetes PV / PVC / StorageClass + `volumeMounts` in Pod spec** — §D in YAML form.

The remaining sections walk these CLI primitives. Whenever a database "loses data" or a stateful Pod refuses to schedule, work back through the §C (semantics) and §D (RWO vs RWX) checklists before blaming the application.

---

## 1. Docker Storage Overview

### The Container Filesystem

Every Docker container has a layered filesystem built from the image's read-only layers plus a thin writable layer on top.

```
┌──────────────────────────────────────────────┐
│           Container Writable Layer           │  ← Lost on container removal
├──────────────────────────────────────────────┤
│           Image Layer N (read-only)          │
├──────────────────────────────────────────────┤
│           Image Layer N-1 (read-only)        │
├──────────────────────────────────────────────┤
│           ...                                │
├──────────────────────────────────────────────┤
│           Base Image Layer (read-only)       │
└──────────────────────────────────────────────┘
```

The writable layer uses a **copy-on-write (CoW)** strategy. When a container modifies a file from a lower layer, the file is first copied into the writable layer. This is efficient for reads but adds overhead for write-heavy workloads.

### Why Persistent Storage Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                     Without Persistent Storage                   │
│                                                                  │
│  Container A (running)          Container A (removed)            │
│  ┌────────────────────┐         ┌────────────────────┐          │
│  │  /var/lib/mysql     │   ──►  │     DATA LOST!     │          │
│  │  (writable layer)   │         │                    │          │
│  └────────────────────┘         └────────────────────┘          │
│                                                                  │
│                     With Persistent Storage                      │
│                                                                  │
│  Container A          Container B (replacement)                  │
│  ┌──────────┐         ┌──────────┐                              │
│  │  mount ──┼────┐    │  mount ──┼────┐                         │
│  └──────────┘    │    └──────────┘    │                         │
│                  ▼                    ▼                          │
│           ┌──────────────────────────────┐                      │
│           │    Volume: db_data           │  ← Data persists     │
│           └──────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Volumes vs Bind Mounts vs tmpfs

Docker provides three mechanisms for persisting data:

### Comparison Table

| Feature | Volumes | Bind Mounts | tmpfs |
|---|---|---|---|
| Managed by Docker | Yes | No | No |
| Location on host | `/var/lib/docker/volumes/` | Anywhere on host | Memory only |
| Survives container removal | Yes | Yes (host files remain) | No |
| Supports volume drivers | Yes | No | No |
| Pre-populated with image data | Yes | No | No |
| Performance | Native | Native | Fastest (RAM) |
| Use case | Production data | Development, config | Sensitive temp data |

### Volumes

Volumes are the preferred mechanism for persisting data. Docker manages the storage location on the host filesystem.

```bash
# Create and use a named volume
docker volume create mydata
docker run -d --name app -v mydata:/app/data nginx

# Using the --mount syntax (more explicit, recommended)
docker run -d --name app \
  --mount type=volume,source=mydata,target=/app/data \
  nginx
```

### Bind Mounts

Bind mounts map a specific host directory into the container. They are ideal for development workflows where you want live code reloading.

```bash
# Bind mount the current directory
docker run -d --name dev \
  -v $(pwd)/src:/app/src \
  node:20

# Using --mount syntax
docker run -d --name dev \
  --mount type=bind,source=$(pwd)/src,target=/app/src \
  node:20

# Read-only bind mount
docker run -d --name app \
  --mount type=bind,source=$(pwd)/config,target=/app/config,readonly \
  myapp
```

> **Warning**: Bind mounts can overwrite files in the container. If you mount an empty host directory to a container path that has files, those files become invisible.

### tmpfs Mounts

tmpfs mounts store data in the host's memory only. Data is never written to disk and is lost when the container stops.

```bash
# tmpfs mount for sensitive temporary data
docker run -d --name secure \
  --mount type=tmpfs,target=/app/secrets,tmpfs-size=100m \
  myapp

# Short syntax
docker run -d --name secure \
  --tmpfs /app/secrets:size=100m \
  myapp
```

Use tmpfs for:
- Temporary session data
- Secrets that should never touch disk
- Scratch space for computations

---

## 3. Named Volumes and Anonymous Volumes

### Named Volumes

Named volumes have explicit names and are easy to reference and manage.

```bash
# Create a named volume
docker volume create app_data

# List volumes
docker volume ls

# Use in docker run
docker run -d -v app_data:/data myapp

# Use in docker-compose.yml
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    image: myapp
    volumes:
      - app_data:/data

volumes:
  app_data:
    driver: local
```

### Anonymous Volumes

Anonymous volumes are created when you specify a mount point without a name. Docker assigns a random hash as the name.

```bash
# Anonymous volume -- Docker generates a random name
docker run -d -v /data myapp

# VOLUME instruction in Dockerfile also creates anonymous volumes
```

```dockerfile
# Dockerfile
FROM postgres:16
VOLUME /var/lib/postgresql/data
```

```bash
# List volumes -- anonymous volumes have hash names
docker volume ls
# DRIVER    VOLUME NAME
# local     app_data
# local     a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6

# Anonymous volumes are harder to track and manage
# Named volumes are always preferred
```

### Volume Labels

You can attach metadata to volumes using labels:

```bash
# Create a volume with labels
docker volume create \
  --label project=myapp \
  --label environment=production \
  myapp_data

# Filter volumes by label
docker volume ls --filter label=project=myapp
```

---

## 4. Volume Drivers and Plugins

### Local Driver Options

The default `local` driver supports options for creating volumes with specific filesystem types:

```bash
# Create a volume with specific mount options
docker volume create --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw \
  --opt device=:/exports/data \
  nfs_data

# Create a tmpfs-backed volume
docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=500m \
  tmpfs_vol

# Create a volume with ext4 on a specific block device
docker volume create --driver local \
  --opt type=ext4 \
  --opt device=/dev/sdb1 \
  fast_storage
```

### Third-Party Volume Drivers

```
┌────────────────────────────────────────────────────────────┐
│                   Volume Driver Ecosystem                   │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Cloud Storage           │  Network Storage                │
│  ┌───────────────────┐   │  ┌───────────────────┐         │
│  │ REX-Ray (AWS EBS,  │   │  │ NFS               │         │
│  │   Azure Disk, GCE) │   │  │ CIFS/Samba        │         │
│  │ NetApp Trident     │   │  │ GlusterFS         │         │
│  │ DigitalOcean Block │   │  │ CephFS            │         │
│  └───────────────────┘   │  └───────────────────┘         │
│                          │                                  │
│  Specialized             │  Distributed                    │
│  ┌───────────────────┐   │  ┌───────────────────┐         │
│  │ Convoy (snapshots)  │   │  │ Portworx          │         │
│  │ Flocker (migration) │   │  │ StorageOS         │         │
│  │ Local-persist       │   │  │ Longhorn          │         │
│  └───────────────────┘   │  └───────────────────┘         │
└────────────────────────────────────────────────────────────┘
```

```bash
# Install a volume plugin
docker plugin install rexray/ebs

# Create a volume with the plugin
docker volume create -d rexray/ebs \
  --opt size=100 \
  --opt volumetype=gp3 \
  ebs_data

# Use in docker-compose.yml
```

```yaml
# docker-compose.yml with external volume driver
version: "3.9"
services:
  db:
    image: postgres:16
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
    driver: rexray/ebs
    driver_opts:
      size: "100"
      volumetype: "gp3"
```

---

## 5. Volume Lifecycle Management

### Volume Creation and Inspection

```bash
# Create a volume
docker volume create mydata

# Inspect volume details
docker volume inspect mydata
```

```json
[
    {
        "CreatedAt": "2025-01-15T10:30:00Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/mydata/_data",
        "Name": "mydata",
        "Options": {},
        "Scope": "local"
    }
]
```

### Finding Unused Volumes

```bash
# List all volumes
docker volume ls

# List dangling (unused) volumes
docker volume ls -f dangling=true

# Show volume disk usage
docker system df -v | grep "VOLUME NAME" -A 100
```

### Pruning Volumes

```bash
# Remove all unused volumes (interactive confirmation)
docker volume prune

# Remove all unused volumes without confirmation
docker volume prune -f

# Remove all unused volumes including those with labels
docker volume prune --all

# WARNING: Pruning is irreversible! Always verify before running.
```

### Volume Lifecycle Diagram

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  create   │────►│  attach   │────►│  detach   │────►│  remove   │
│           │     │ (docker   │     │ (docker   │     │ (docker   │
│ docker    │     │  run -v)  │     │  stop/rm) │     │  volume   │
│ volume    │     │           │     │           │     │  rm)      │
│ create    │     │           │     │           │     │           │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                       │                │
                       │     ┌──────────┘
                       ▼     ▼
                  ┌──────────────┐
                  │   re-attach   │
                  │  (new container│
                  │   mounts same  │
                  │   volume)      │
                  └──────────────┘
```

---

## 6. Data Backup and Restore

### Backup Strategy Using a Helper Container

```bash
# Backup a volume to a tar archive
docker run --rm \
  -v mydata:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine \
  tar czf /backup/mydata-$(date +%Y%m%d_%H%M%S).tar.gz -C /source .

# This creates a compressed archive of the volume contents
```

### Restore from Backup

```bash
# Create a new volume
docker volume create mydata_restored

# Restore from backup
docker run --rm \
  -v mydata_restored:/target \
  -v $(pwd)/backups:/backup:ro \
  alpine \
  sh -c "cd /target && tar xzf /backup/mydata-20250115_103000.tar.gz"
```

### Automated Backup Script

```bash
#!/bin/bash
# backup-volumes.sh -- Automated Docker volume backup

BACKUP_DIR="/opt/backups/docker-volumes"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Get all named volumes
volumes=$(docker volume ls -q --filter dangling=false)

for vol in $volumes; do
    echo "Backing up volume: $vol"
    docker run --rm \
        -v "$vol":/source:ro \
        -v "$BACKUP_DIR":/backup \
        alpine \
        tar czf "/backup/${vol}_${DATE}.tar.gz" -C /source .

    if [ $? -eq 0 ]; then
        echo "  ✓ Backup successful: ${vol}_${DATE}.tar.gz"
    else
        echo "  ✗ Backup failed for: $vol"
    fi
done

# Clean up old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
echo "Cleaned up backups older than $RETENTION_DAYS days"
```

### Database-Specific Backup

For databases, prefer logical backups (SQL dumps) over filesystem-level backups:

```bash
# PostgreSQL backup
docker exec my_postgres \
  pg_dump -U myuser -d mydb > backup.sql

# MySQL backup
docker exec my_mysql \
  mysqldump -u root -p"$MYSQL_ROOT_PASSWORD" mydb > backup.sql

# MongoDB backup
docker exec my_mongo \
  mongodump --archive=/tmp/backup.archive --gzip
docker cp my_mongo:/tmp/backup.archive ./backup.archive
```

---

## 7. Volume Sharing Between Containers

### Shared Volume Pattern

Multiple containers can mount the same volume for data exchange:

```yaml
# docker-compose.yml
version: "3.9"
services:
  # Writer container generates log files
  writer:
    image: alpine
    command: sh -c "while true; do echo $$(date) >> /shared/log.txt; sleep 5; done"
    volumes:
      - shared_data:/shared

  # Reader container processes log files
  reader:
    image: alpine
    command: tail -f /shared/log.txt
    volumes:
      - shared_data:/shared:ro
    depends_on:
      - writer

volumes:
  shared_data:
```

### Web Application with Shared Static Assets

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    volumes:
      - static_files:/app/static

  nginx:
    image: nginx:alpine
    volumes:
      - static_files:/usr/share/nginx/html/static:ro
    ports:
      - "80:80"
    depends_on:
      - app

volumes:
  static_files:
```

### Concurrency Considerations

```
┌──────────────────────────────────────────────────────────────┐
│              Volume Sharing: Concurrency Risks                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Safe Patterns:                                               │
│  ┌───────────────────────────────────────────────────┐       │
│  │ • One writer, many readers (read-only mounts)     │       │
│  │ • Each container writes to different files         │       │
│  │ • Application-level locking (e.g., flock)          │       │
│  └───────────────────────────────────────────────────┘       │
│                                                               │
│  Dangerous Patterns:                                          │
│  ┌───────────────────────────────────────────────────┐       │
│  │ • Multiple writers to the same file               │       │
│  │ • Database files shared without coordination      │       │
│  │ • No locking mechanism in place                   │       │
│  └───────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Database Storage Best Practices

### PostgreSQL

```yaml
# docker-compose.yml
version: "3.9"
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    volumes:
      # Named volume for data directory
      - pgdata:/var/lib/postgresql/data
      # Bind mount for custom configuration
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro
      # Bind mount for initialization scripts
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    secrets:
      - db_password
    deploy:
      resources:
        limits:
          memory: 2G

volumes:
  pgdata:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### MySQL

```yaml
# docker-compose.yml
version: "3.9"
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD_FILE: /run/secrets/mysql_root_pw
      MYSQL_DATABASE: myapp
    volumes:
      - mysqldata:/var/lib/mysql
      - ./my.cnf:/etc/mysql/conf.d/custom.cnf:ro
    secrets:
      - mysql_root_pw
    # Ensure data consistency
    command: >
      --innodb-flush-log-at-trx-commit=1
      --sync-binlog=1

volumes:
  mysqldata:
    driver: local

secrets:
  mysql_root_pw:
    file: ./secrets/mysql_root_pw.txt
```

### MongoDB

```yaml
# docker-compose.yml
version: "3.9"
services:
  mongo:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD_FILE: /run/secrets/mongo_pw
    volumes:
      - mongodata:/data/db
      - mongoconfigdb:/data/configdb
    secrets:
      - mongo_pw

volumes:
  mongodata:
    driver: local
  mongoconfigdb:
    driver: local

secrets:
  mongo_pw:
    file: ./secrets/mongo_pw.txt
```

### General Database Storage Guidelines

```
┌──────────────────────────────────────────────────────────────┐
│              Database Volume Best Practices                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Always use named volumes (never anonymous)                │
│  2. Never use bind mounts for database data in production     │
│  3. Set appropriate filesystem permissions                    │
│  4. Use volume labels for organization                        │
│  5. Implement regular backup schedules                        │
│  6. Test restore procedures periodically                      │
│  7. Monitor volume disk usage                                │
│  8. Use dedicated storage for write-heavy workloads           │
│  9. Consider I/O scheduler tuning for database volumes        │
│  10. Never share database volumes between instances           │
│      (unless using database-native replication)               │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Volume Commands Reference

### Essential Commands

```bash
# Create a volume
docker volume create [OPTIONS] [VOLUME]

# List volumes
docker volume ls [OPTIONS]

# Inspect a volume
docker volume inspect [VOLUME]

# Remove a volume
docker volume rm [VOLUME...]

# Remove unused volumes
docker volume prune [OPTIONS]
```

### Practical Examples

```bash
# Create volume with specific labels
docker volume create --label env=prod --label app=web webdata

# List volumes with filters
docker volume ls --filter driver=local
docker volume ls --filter label=env=prod
docker volume ls --filter dangling=true

# Format volume listing
docker volume ls --format "{{.Name}}\t{{.Driver}}\t{{.Mountpoint}}"

# Get volume mount point
docker volume inspect --format '{{.Mountpoint}}' mydata

# Check which containers use a volume
docker ps -a --filter volume=mydata \
  --format "{{.ID}}\t{{.Names}}\t{{.Status}}"

# Copy data between volumes
docker run --rm \
  -v source_vol:/source:ro \
  -v target_vol:/target \
  alpine sh -c "cp -a /source/. /target/"

# Get total volume disk usage
docker system df -v
```

---

## 10. Practice Exercises

### Exercise 1: Volume Basics (Beginner)

Create a named volume, run a container that writes data to it, remove the container, and verify the data persists in a new container.

```bash
# 1. Create a named volume called "exercise_data"
# 2. Run an alpine container that writes "Hello Volumes!" to /data/hello.txt
# 3. Remove the container
# 4. Run a new alpine container mounting the same volume
# 5. Verify the file contents
```

<details>
<summary>Solution</summary>

```bash
docker volume create exercise_data
docker run --rm -v exercise_data:/data alpine sh -c "echo 'Hello Volumes!' > /data/hello.txt"
docker run --rm -v exercise_data:/data alpine cat /data/hello.txt
# Output: Hello Volumes!
docker volume rm exercise_data
```

</details>

### Exercise 2: Backup and Restore (Intermediate)

Set up a PostgreSQL container with a named volume, insert data, back up the volume, restore it to a new volume, and verify the data.

```bash
# 1. Start a PostgreSQL container with a named volume "pg_exercise"
# 2. Create a table and insert sample data
# 3. Backup the volume using the tar method
# 4. Create a new volume "pg_exercise_restored"
# 5. Restore the backup to the new volume
# 6. Start a new PostgreSQL container with the restored volume
# 7. Verify the data
```

<details>
<summary>Solution</summary>

```bash
# Start PostgreSQL
docker run -d --name pg_test \
  -e POSTGRES_PASSWORD=testpass \
  -v pg_exercise:/var/lib/postgresql/data \
  postgres:16-alpine

# Wait for initialization
sleep 5

# Insert data
docker exec pg_test psql -U postgres -c "
  CREATE TABLE users (id SERIAL, name TEXT);
  INSERT INTO users (name) VALUES ('Alice'), ('Bob');
"

# Stop container for consistent backup
docker stop pg_test

# Backup
docker run --rm \
  -v pg_exercise:/source:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/pg_backup.tar.gz -C /source .

# Create restored volume and restore
docker volume create pg_exercise_restored
docker run --rm \
  -v pg_exercise_restored:/target \
  -v $(pwd):/backup:ro \
  alpine sh -c "cd /target && tar xzf /backup/pg_backup.tar.gz"

# Verify with new container
docker run -d --name pg_restored \
  -e POSTGRES_PASSWORD=testpass \
  -v pg_exercise_restored:/var/lib/postgresql/data \
  postgres:16-alpine

sleep 5
docker exec pg_restored psql -U postgres -c "SELECT * FROM users;"

# Cleanup
docker rm -f pg_test pg_restored
docker volume rm pg_exercise pg_exercise_restored
rm pg_backup.tar.gz
```

</details>

### Exercise 3: Multi-Container Volume Sharing (Intermediate)

Create a docker-compose setup where a "generator" container writes timestamped entries to a shared volume and a "web" container serves those entries via nginx.

<details>
<summary>Solution</summary>

```yaml
# docker-compose.yml
version: "3.9"
services:
  generator:
    image: alpine
    command: >
      sh -c "mkdir -p /shared/html &&
             while true; do
               echo \"<p>Generated at: $$(date)</p>\" >> /shared/html/index.html;
               sleep 10;
             done"
    volumes:
      - shared:/shared

  web:
    image: nginx:alpine
    volumes:
      - shared:/usr/share/nginx:ro
    ports:
      - "8080:80"
    depends_on:
      - generator

volumes:
  shared:
```

```bash
docker compose up -d
# Visit http://localhost:8080 to see timestamped entries
# Wait and refresh to see new entries
docker compose down -v
```

</details>

### Exercise 4: Volume Driver Exploration (Advanced)

Create an NFS-backed volume (or simulate with a local driver using specific mount options) and demonstrate that it can be used across multiple containers simultaneously.

<details>
<summary>Solution</summary>

```bash
# Create a volume with specific local driver options (simulating NFS)
docker volume create \
  --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=50m \
  shared_tmpfs

# Run two containers using the same volume
docker run -d --name writer \
  -v shared_tmpfs:/data \
  alpine sh -c "while true; do date >> /data/log.txt; sleep 2; done"

docker run -d --name reader \
  -v shared_tmpfs:/data:ro \
  alpine sh -c "while true; do echo '--- Latest ---'; tail -3 /data/log.txt 2>/dev/null; sleep 5; done"

# Check reader logs
sleep 10
docker logs reader

# Inspect the volume
docker volume inspect shared_tmpfs

# Cleanup
docker rm -f writer reader
docker volume rm shared_tmpfs
```

</details>

---

**Previous**: [Security Best Practices](./12_Security_Best_Practices.md) | **Next**: [Multi-Stage Build Patterns](./14_Multi_Stage_Build_Patterns.md)
