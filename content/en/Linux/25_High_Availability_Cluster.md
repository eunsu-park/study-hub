# High Availability Cluster

**Previous**: [Cloud Integration](./24_Cloud_Integration.md) | **Next**: [Troubleshooting Guide](./26_Troubleshooting_Guide.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the components of an HA cluster architecture including Corosync, Pacemaker, and DRBD
2. Set up a two-node cluster using pcs with proper quorum and property configuration
3. Create and manage Pacemaker resources including virtual IPs, filesystems, and services
4. Define resource groups, ordering constraints, and colocation constraints for correct failover
5. Configure DRBD for real-time block-level replication between cluster nodes
6. Implement STONITH fencing to prevent split-brain scenarios in production
7. Test failover procedures and monitor cluster health with crm_mon and pcs

---

## Table of Contents

1. [High Availability Overview](#1-high-availability-overview)
2. [Corosync Configuration](#2-corosync-configuration)
3. [Pacemaker Configuration](#3-pacemaker-configuration)
4. [Resource Management](#4-resource-management)
5. [DRBD Configuration](#5-drbd-configuration)
6. [Fencing (STONITH)](#6-fencing-stonith)
7. [Production Cluster Setup](#7-production-cluster-setup)

---

For critical services like databases, web applications, and file servers, even minutes of downtime can mean lost revenue and broken SLAs. High Availability clustering eliminates single points of failure by running services across multiple nodes that automatically take over when one fails. This lesson covers the complete HA stack -- Corosync for cluster communication, Pacemaker for resource management, DRBD for data replication, and fencing for split-brain prevention -- giving you the skills to build systems that survive hardware failures without manual intervention.

## 1. High Availability Overview

### HA Cluster Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HA Cluster                               │
│                                                             │
│  ┌───────────────────┐       ┌───────────────────┐         │
│  │      Node 1       │       │      Node 2       │         │
│  │   (Active)        │       │   (Standby)       │         │
│  │                   │       │                   │         │
│  │  ┌─────────────┐  │       │  ┌─────────────┐  │         │
│  │  │ Pacemaker   │◄─┼───────┼─►│ Pacemaker   │  │         │
│  │  └─────────────┘  │       │  └─────────────┘  │         │
│  │         │         │       │         │         │         │
│  │  ┌─────────────┐  │       │  ┌─────────────┐  │         │
│  │  │  Corosync   │◄─┼───────┼─►│  Corosync   │  │         │
│  │  └─────────────┘  │       │  └─────────────┘  │         │
│  │         │         │       │         │         │         │
│  │  ┌─────────────┐  │       │  ┌─────────────┐  │         │
│  │  │   DRBD      │◄─┼───────┼─►│   DRBD      │  │         │
│  │  │  (Primary)  │  │       │  │ (Secondary) │  │         │
│  │  └─────────────┘  │       │  └─────────────┘  │         │
│  │         │         │       │                   │         │
│  │  ┌─────────────┐  │       │                   │         │
│  │  │ Application │  │       │                   │         │
│  │  │  (Running)  │  │       │                   │         │
│  │  └─────────────┘  │       │                   │         │
│  └───────────────────┘       └───────────────────┘         │
│             │                                               │
│             ▼                                               │
│      ┌─────────────┐                                       │
│      │  Virtual IP │  ← Client access point                │
│      └─────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### HA Components

| Component | Role |
|-----------|------|
| **Corosync** | Cluster communication and membership management |
| **Pacemaker** | Resource management and failover |
| **DRBD** | Block-level storage replication |
| **STONITH/Fencing** | Split-brain prevention |

### Package Installation

```bash
# Ubuntu/Debian
sudo apt install pacemaker corosync pcs resource-agents fence-agents

# RHEL/CentOS
sudo yum install pacemaker corosync pcs resource-agents fence-agents-all
```

---

## 2. Corosync Configuration

### Basic Configuration

```bash
# /etc/corosync/corosync.conf
totem {
    version: 2
    cluster_name: mycluster
    transport: knet

    crypto_cipher: aes256
    crypto_hash: sha256

    interface {
        ringnumber: 0
        bindnetaddr: 192.168.1.0
        mcastport: 5405
    }
}

logging {
    to_logfile: yes
    logfile: /var/log/corosync/corosync.log
    to_syslog: yes
    timestamp: on
}

quorum {
    provider: corosync_votequorum
    two_node: 1
    wait_for_all: 1
}

nodelist {
    node {
        ring0_addr: node1.example.com
        nodeid: 1
    }
    node {
        ring0_addr: node2.example.com
        nodeid: 2
    }
}
```

### Cluster Configuration Using pcs

```bash
# Start pcsd service
sudo systemctl enable pcsd
sudo systemctl start pcsd

# Set hacluster user password (on all nodes)
sudo passwd hacluster

# Authenticate nodes (run on one node)
sudo pcs host auth node1 node2

# Create cluster
sudo pcs cluster setup mycluster node1 node2

# Start cluster
sudo pcs cluster start --all

# Enable cluster (auto-start at boot)
sudo pcs cluster enable --all

# Check status
sudo pcs cluster status
sudo pcs status
```

### Checking Corosync Status

```bash
# Membership check
sudo corosync-cmapctl | grep members

# Quorum status
sudo corosync-quorumtool

# Ring status
sudo corosync-cfgtool -s
```

---

## 3. Pacemaker Configuration

### Cluster Property Configuration

```bash
# Disable STONITH (for testing, required in production)
sudo pcs property set stonith-enabled=false

# Quorum policy
sudo pcs property set no-quorum-policy=ignore  # 2-node cluster

# Default stickiness (prevent resource movement)
sudo pcs resource defaults update resource-stickiness=100

# Check properties
sudo pcs property list
```

### Checking Cluster Status

```bash
# Overall status
sudo pcs status

# Resource status
sudo pcs resource status

# Node status
sudo pcs node status

# Constraints
sudo pcs constraint list --full

# Cluster configuration
sudo pcs config
```

---

## 4. Resource Management

### Creating Basic Resources

```bash
# Virtual IP resource
sudo pcs resource create VirtualIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.100 \
    cidr_netmask=24 \
    op monitor interval=30s

# Web server resource
sudo pcs resource create WebServer ocf:heartbeat:nginx \
    configfile=/etc/nginx/nginx.conf \
    op start timeout=40s \
    op stop timeout=60s \
    op monitor interval=10s

# Filesystem resource
sudo pcs resource create WebFS ocf:heartbeat:Filesystem \
    device=/dev/drbd0 \
    directory=/var/www \
    fstype=ext4 \
    op start timeout=60s \
    op stop timeout=60s
```

### Resource Groups

```bash
# Create group (starts in order, stops in reverse order)
sudo pcs resource group add WebGroup \
    WebFS \
    VirtualIP \
    WebServer

# Check group status
sudo pcs resource show WebGroup
```

### Resource Constraints

```bash
# Location constraint (prefer specific node)
sudo pcs constraint location WebServer prefers node1=100
sudo pcs constraint location WebServer avoids node2

# Order constraint (start order)
sudo pcs constraint order WebFS then VirtualIP then WebServer

# Colocation constraint (run on same node)
sudo pcs constraint colocation add WebServer with VirtualIP INFINITY
sudo pcs constraint colocation add VirtualIP with WebFS INFINITY

# Check constraints
sudo pcs constraint list --full
```

### Resource Management Commands

```bash
# Start/stop resource
sudo pcs resource enable WebServer
sudo pcs resource disable WebServer

# Move resource (manual failover)
sudo pcs resource move WebServer node2

# Remove move constraint (allow return to original location)
sudo pcs resource clear WebServer

# Restart resource
sudo pcs resource restart WebServer

# Delete resource
sudo pcs resource delete WebServer
```

---

## 5. DRBD Configuration

### Installing DRBD

```bash
# Ubuntu/Debian
sudo apt install drbd-utils

# RHEL/CentOS (using ELRepo)
sudo rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
sudo yum install https://www.elrepo.org/elrepo-release-9.el9.elrepo.noarch.rpm
sudo yum install drbd90-utils kmod-drbd90
```

### DRBD Resource Configuration

```bash
# /etc/drbd.d/r0.res (same on both nodes)
resource r0 {
    protocol C;

    startup {
        wfc-timeout  15;
        degr-wfc-timeout 60;
    }

    net {
        cram-hmac-alg sha1;
        shared-secret "mysecret123";
    }

    disk {
        on-io-error detach;
    }

    on node1 {
        device    /dev/drbd0;
        disk      /dev/sdb1;
        address   192.168.1.11:7788;
        meta-disk internal;
    }

    on node2 {
        device    /dev/drbd0;
        disk      /dev/sdb1;
        address   192.168.1.12:7788;
        meta-disk internal;
    }
}
```

### Initializing DRBD

```bash
# Create metadata (on both nodes)
sudo drbdadm create-md r0

# Start DRBD (on both nodes)
sudo drbdadm up r0

# Set Primary (on one node only)
sudo drbdadm primary --force r0

# Check status
cat /proc/drbd
sudo drbdadm status

# Create filesystem (on Primary)
sudo mkfs.ext4 /dev/drbd0
```

### Integrating Pacemaker with DRBD

```bash
# Create DRBD resource
sudo pcs resource create DRBD ocf:linbit:drbd \
    drbd_resource=r0 \
    op monitor interval=60s

# Master/slave configuration
sudo pcs resource promotable DRBD \
    promoted-max=1 \
    clone-max=2 \
    notify=true

# Filesystem resource
sudo pcs resource create WebFS ocf:heartbeat:Filesystem \
    device=/dev/drbd0 \
    directory=/var/www \
    fstype=ext4

# Constraint: WebFS only on DRBD Primary
sudo pcs constraint colocation add WebFS with DRBD-clone INFINITY with-rsc-role=Master
sudo pcs constraint order promote DRBD-clone then start WebFS
```

### Checking DRBD Status

```bash
# Check status
sudo drbdadm status
sudo drbdadm dstate r0
sudo drbdadm cstate r0
sudo drbdadm role r0

# Sync status
cat /proc/drbd

# Example output:
# version: 8.4.11
#  0: cs:Connected ro:Primary/Secondary ds:UpToDate/UpToDate C r-----
```

---

## 6. Fencing (STONITH)

### Fencing Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Split-Brain Situation                     │
│                                                             │
│  ┌───────────┐     Network Partition    ┌───────────┐      │
│  │   Node1   │ ────────X──────────────  │   Node2   │      │
│  │ (Primary) │                          │ (Primary) │ ← Danger! │
│  └───────────┘                          └───────────┘      │
│       │                                        │            │
│       └─────────────┬──────────────────────────┘            │
│                     │                                       │
│              ┌──────┴──────┐                               │
│              │ Shared Data │ ← Possible data corruption!   │
│              └─────────────┘                               │
│                                                             │
│  Solution: STONITH (Shoot The Other Node In The Head)      │
│  - Force reset/power off the problematic node               │
└─────────────────────────────────────────────────────────────┘
```

### Virtual Environment Fencing (fence_virsh)

```bash
# Test fence_virsh agent
sudo fence_virsh -a qemu+ssh://hypervisor -l root \
    -p password -n vm-node1 -o status

# Pacemaker resource configuration
sudo pcs stonith create fence_node1 fence_virsh \
    pcmk_host_list="node1" \
    ipaddr="qemu+ssh://hypervisor" \
    login="root" \
    passwd="password" \
    port="vm-node1" \
    ssl_insecure=1

sudo pcs stonith create fence_node2 fence_virsh \
    pcmk_host_list="node2" \
    ipaddr="qemu+ssh://hypervisor" \
    login="root" \
    passwd="password" \
    port="vm-node2" \
    ssl_insecure=1

# Location constraint (prevent running on its own node)
sudo pcs constraint location fence_node1 avoids node1
sudo pcs constraint location fence_node2 avoids node2
```

### IPMI Fencing

```bash
# fence_ipmilan agent
sudo pcs stonith create fence_node1_ipmi fence_ipmilan \
    pcmk_host_list="node1" \
    ipaddr="192.168.1.101" \
    login="admin" \
    passwd="password" \
    lanplus=1 \
    power_timeout=20

sudo pcs stonith create fence_node2_ipmi fence_ipmilan \
    pcmk_host_list="node2" \
    ipaddr="192.168.1.102" \
    login="admin" \
    passwd="password" \
    lanplus=1 \
    power_timeout=20
```

### Cloud Fencing

```bash
# AWS (fence_aws)
sudo pcs stonith create fence_aws fence_aws \
    pcmk_host_map="node1:i-0123456789abcdef0;node2:i-0fedcba9876543210" \
    region="ap-northeast-2" \
    power_timeout=60

# GCP (fence_gce)
sudo pcs stonith create fence_gcp fence_gce \
    pcmk_host_map="node1:instance-1;node2:instance-2" \
    project="my-project" \
    zone="asia-northeast3-a"
```

### Testing Fencing

```bash
# Enable STONITH
sudo pcs property set stonith-enabled=true

# Test fencing
sudo stonith_admin --reboot node2 --verbose

# Manually fence a node
sudo pcs stonith fence node2

# Fencing history
sudo stonith_admin --history node2
```

---

## 7. Production Cluster Setup

### 2-Node Web Server Cluster

```bash
#!/bin/bash
# setup-ha-cluster.sh

# Cluster configuration
pcs cluster setup mycluster node1 node2

# Start cluster
pcs cluster start --all
pcs cluster enable --all

# Property configuration
pcs property set stonith-enabled=true
pcs property set no-quorum-policy=ignore

# STONITH resources (e.g., IPMI)
pcs stonith create fence_node1 fence_ipmilan \
    pcmk_host_list="node1" \
    ipaddr="10.0.0.101" login="admin" passwd="password" lanplus=1

pcs stonith create fence_node2 fence_ipmilan \
    pcmk_host_list="node2" \
    ipaddr="10.0.0.102" login="admin" passwd="password" lanplus=1

pcs constraint location fence_node1 avoids node1
pcs constraint location fence_node2 avoids node2

# DRBD resource
pcs resource create DRBD ocf:linbit:drbd drbd_resource=r0 \
    op monitor interval=60s
pcs resource promotable DRBD promoted-max=1 clone-max=2 notify=true

# Filesystem resource
pcs resource create WebFS ocf:heartbeat:Filesystem \
    device=/dev/drbd0 directory=/var/www fstype=ext4

# VIP resource
pcs resource create VIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.100 cidr_netmask=24

# Web server resource
pcs resource create WebServer ocf:heartbeat:nginx \
    configfile=/etc/nginx/nginx.conf

# Resource group
pcs resource group add WebGroup WebFS VIP WebServer

# Constraints
pcs constraint colocation add WebGroup with DRBD-clone INFINITY with-rsc-role=Master
pcs constraint order promote DRBD-clone then start WebGroup

# Check status
pcs status
```

### Failover Testing

```bash
# Check current status
pcs status

# Manual failover test
pcs resource move WebGroup node2

# Put node in standby mode
pcs node standby node1

# Release standby
pcs node unstandby node1

# Simulation (force stop resource)
pcs resource debug-stop WebServer

# Failback (return to original node)
pcs resource clear WebGroup
```

### Monitoring

```bash
# Real-time status monitoring
watch -n 1 'pcs status'

# Cluster event log
sudo journalctl -u pacemaker -f

# crm_mon (detailed monitoring)
sudo crm_mon -1
sudo crm_mon -Afr  # Full info, fail count, resources

# Resource history
pcs resource history show WebServer
```

---

## Practice Problems

### Problem 1: Cluster Configuration

Write the command sequence to set up a 2-node cluster using pcs.

### Problem 2: Resource Group

Create a resource group containing VIP (192.168.1.200), filesystem (/dev/sdb1 → /data), and PostgreSQL service.

### Problem 3: DRBD Replication

Write commands to check the current status and sync state of DRBD resource r0.

---

## Answers

### Problem 1 Answer

```bash
# 1. Start pcsd and authenticate
sudo systemctl enable pcsd
sudo systemctl start pcsd
sudo passwd hacluster
sudo pcs host auth node1 node2

# 2. Create and start cluster
sudo pcs cluster setup mycluster node1 node2
sudo pcs cluster start --all
sudo pcs cluster enable --all

# 3. Set basic properties
sudo pcs property set stonith-enabled=false  # For testing
sudo pcs property set no-quorum-policy=ignore
```

### Problem 2 Answer

```bash
# VIP resource
sudo pcs resource create VIP ocf:heartbeat:IPaddr2 \
    ip=192.168.1.200 cidr_netmask=24

# Filesystem resource
sudo pcs resource create DataFS ocf:heartbeat:Filesystem \
    device=/dev/sdb1 directory=/data fstype=ext4

# PostgreSQL resource
sudo pcs resource create PostgreSQL ocf:heartbeat:pgsql \
    pgctl=/usr/lib/postgresql/14/bin/pg_ctl \
    pgdata=/var/lib/postgresql/14/main \
    op start timeout=60s \
    op stop timeout=60s \
    op monitor interval=10s

# Create group
sudo pcs resource group add DBGroup DataFS VIP PostgreSQL

# Order constraint (explicit)
sudo pcs constraint order DataFS then VIP then PostgreSQL
```

### Problem 3 Answer

```bash
# Check DRBD status
sudo drbdadm status r0

# Check /proc/drbd
cat /proc/drbd

# Connection state
sudo drbdadm cstate r0

# Disk state
sudo drbdadm dstate r0

# Role check
sudo drbdadm role r0

# Detailed status (all resources)
sudo drbdadm status all
```

---

---

## References

- [Pacemaker Documentation](https://clusterlabs.org/pacemaker/doc/)
- [DRBD User's Guide](https://linbit.com/drbd-user-guide/)
- [Red Hat HA Cluster](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/configuring_and_managing_high_availability_clusters/index)
- `man pcs`, `man corosync`, `man drbdadm`

---

**Previous**: [Cloud Integration](./24_Cloud_Integration.md) | **Next**: [Troubleshooting Guide](./26_Troubleshooting_Guide.md)
