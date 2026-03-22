# Linux Learning Guide

## Introduction

This folder contains materials for systematic learning of the Linux operating system, from basics to server administration.

- **Target Audience**: Linux beginners ~ server administrators
- **Distributions**: Ubuntu/Debian and CentOS/RHEL both covered
- **Goal**: From command usage to server operations

---

## Learning Roadmap

```
[Beginner]            [Intermediate]         [Advanced]
  │                     │                      │
  ▼                     ▼                      ▼
Linux Basics ──────▶ Text Processing ──────▶ Shell Scripting
  │                     │                      │
  ▼                     ▼                      ▼
Filesystem ────────▶ Permissions ───────────▶ Network Basics
  │                     │                      │
  ▼                     ▼                      ▼
File Mgmt ─────────▶ User Mgmt ─────────────▶ System Monitoring
                        │                      │
                        ▼                      ▼
                  Process Mgmt ──────────▶ Security & Firewall
                        │
                        ▼
                  Package Mgmt
```

---

## Prerequisites

- Basic computer skills
- Understanding of terminal/command prompt concepts
- English command reading (not required)

---

## File List

| File | Difficulty | Key Topics |
|------|------------|-----------|
| [01_Linux_Basics.md](./01_Linux_Basics.md) | ⭐ | Linux concepts, distributions, terminal, basic commands |
| [02_Filesystem_Navigation.md](./02_Filesystem_Navigation.md) | ⭐ | Directory structure, paths, ls, cd, find |
| [03_File_Directory_Management.md](./03_File_Directory_Management.md) | ⭐ | touch, mkdir, cp, mv, rm, tar |
| [04_Text_Processing.md](./04_Text_Processing.md) | ⭐⭐ | grep, sed, awk, pipes, redirection |
| [05_Permissions_Ownership.md](./05_Permissions_Ownership.md) | ⭐⭐ | chmod, chown, special permissions, umask |
| [06_User_Group_Management.md](./06_User_Group_Management.md) | ⭐⭐ | useradd, sudo, user/group management |
| [07_Process_Management.md](./07_Process_Management.md) | ⭐⭐ | ps, top, kill, systemctl |
| [08_Package_Management.md](./08_Package_Management.md) | ⭐⭐ | apt, yum/dnf, repository management |
| [09_Shell_Scripting.md](./09_Shell_Scripting.md) | ⭐⭐⭐ | variables, conditionals, loops, practical scripts |
| [10_Network_Basics.md](./10_Network_Basics.md) | ⭐⭐⭐ | ip, ssh, port checking, remote access |
| [11_System_Monitoring.md](./11_System_Monitoring.md) | ⭐⭐⭐ | df, free, logs, cron |
| [12_Security_and_Firewall.md](./12_Security_and_Firewall.md) | ⭐⭐⭐⭐ | SSH security, ufw, firewalld, fail2ban |
| [13_Systemd_Advanced.md](./13_Systemd_Advanced.md) | ⭐⭐⭐⭐ | service units, timers, sockets, journald |
| [14_Performance_Tuning.md](./14_Performance_Tuning.md) | ⭐⭐⭐⭐ | sysctl, kernel parameters, perf, flamegraph |
| [15_Container_Internals.md](./15_Container_Internals.md) | ⭐⭐⭐⭐ | cgroups, namespaces, container runtime |
| [16_Storage_Management.md](./16_Storage_Management.md) | ⭐⭐⭐⭐ | LVM, RAID, filesystems, LUKS encryption |
| [17_SELinux_AppArmor.md](./17_SELinux_AppArmor.md) | ⭐⭐⭐⭐ | SELinux policies, AppArmor profiles, troubleshooting |
| [18_Log_Management.md](./18_Log_Management.md) | ⭐⭐⭐ | journald, rsyslog, logrotate, remote logging |
| [19_Backup_Recovery.md](./19_Backup_Recovery.md) | ⭐⭐⭐⭐ | rsync, Borg Backup, disaster recovery strategies |
| [20_Kernel_Management.md](./20_Kernel_Management.md) | ⭐⭐⭐⭐ | kernel compilation, modules, DKMS, GRUB |
| [21_Virtualization_KVM.md](./21_Virtualization_KVM.md) | ⭐⭐⭐⭐ | libvirt, virsh, VM management, snapshots |
| [22_Ansible_Basics.md](./22_Ansible_Basics.md) | ⭐⭐⭐ | inventory, playbooks, roles, Vault |
| [23_Advanced_Networking.md](./23_Advanced_Networking.md) | ⭐⭐⭐⭐ | VLAN, bonding, iptables/nftables |
| [24_Cloud_Integration.md](./24_Cloud_Integration.md) | ⭐⭐⭐ | cloud-init, AWS CLI, metadata |
| [25_High_Availability_Cluster.md](./25_High_Availability_Cluster.md) | ⭐⭐⭐⭐⭐ | Pacemaker, Corosync, DRBD |
| [26_Troubleshooting_Guide.md](./26_Troubleshooting_Guide.md) | ⭐⭐⭐ | boot, network, disk, performance troubleshooting |

---

## Recommended Learning Path

### Stage 1: Linux Introduction (Beginner)

```
01_Linux_Basics → 02_Filesystem_Navigation → 03_File_Directory_Management
```

Learn terminal usage and basic commands.

### Stage 2: Practical Usage (Intermediate)

```
04_Text_Processing → 05_Permissions_Ownership → 06_User_Group_Management
    → 07_Process_Management → 08_Package_Management
```

Learn file processing, permission management, and system operations basics.

### Stage 3: Server Administration (Advanced)

```
09_Shell_Scripting → 10_Network_Basics → 11_System_Monitoring → 12_Security_and_Firewall
```

Cover all aspects of server management including automation, networking, monitoring, and security.

### Stage 4: System Deep Dive (Expert)

```
13_Systemd_Advanced → 14_Performance_Tuning → 15_Container_Internals → 16_Storage_Management
```

Study systemd, performance optimization, container internals, and storage management.

### Stage 5: Enterprise Operations (Expert)

```
17_SELinux_AppArmor → 18_Log_Management → 19_Backup_Recovery → 20_Kernel_Management
```

Cover security modules, log management, backup strategies, and kernel management.

### Stage 6: Infrastructure Engineering (Expert)

```
21_Virtualization_KVM → 22_Ansible_Basics → 23_Advanced_Networking → 24_Cloud_Integration
    → 25_High_Availability_Cluster → 26_Troubleshooting_Guide
```

Master virtualization, automation, advanced networking, cloud, HA, and troubleshooting.

---

## Practice Environment

### Ubuntu (Recommended)

```bash
# Quick start with Docker
docker run -it ubuntu:22.04 bash

# Or use VM/WSL
# - VirtualBox + Ubuntu ISO
# - Windows WSL2
```

### CentOS/RHEL

```bash
# Start with Docker
docker run -it rockylinux:9 bash

# Or use VM
# - VirtualBox + Rocky Linux ISO
```

### Cloud (for practice)

- AWS EC2 Free Tier
- Google Cloud Free Tier
- DigitalOcean (paid)

---

## Distribution Comparison

| Item | Ubuntu/Debian | CentOS/RHEL |
|------|---------------|-------------|
| Package Management | APT (`apt`) | YUM/DNF (`dnf`) |
| Package Format | .deb | .rpm |
| Firewall | UFW | firewalld |
| Security Module | AppArmor | SELinux |
| Service Management | systemctl | systemctl |
| Main Use | Desktop, server | Enterprise server |

---

## Related Resources

- [Docker/](../Docker/00_Overview.md) - Using Linux in container environments
- [Git/](../Git/00_Overview.md) - Version control on Linux
- [PostgreSQL/](../PostgreSQL/00_Overview.md) - Database operations on Linux servers
