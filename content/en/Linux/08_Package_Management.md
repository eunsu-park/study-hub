# Package Management

**Previous**: [Process Management](./07_Process_Management.md) | **Next**: [Shell Scripting](./09_Shell_Scripting.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how package managers resolve dependencies and verify integrity
2. Install, update, search, and remove packages using APT (Ubuntu/Debian)
3. Perform the same operations using DNF/YUM (CentOS/RHEL)
4. Add third-party repositories and manage GPG keys
5. Query package information and find which package owns a file using dpkg and rpm
6. Compile software from source when packages are not available
7. Pin package versions to prevent unwanted upgrades
8. Evaluate CentOS migration options (Rocky Linux, AlmaLinux) after CentOS EOL

---

Installing software on Linux is fundamentally different from downloading installers on Windows or macOS. Package managers handle everything automatically -- fetching the software, resolving dependencies, verifying checksums, and placing files in the correct locations. Keeping your system up to date through the package manager is one of the single most important things you can do for security and stability.

## 1. Package Management Concepts

Package managers automate software installation, updates, and removal.

```
┌─────────────────────────────────────────────────────────┐
│                    Package Repository                    │
│              (Repository / Mirror)                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Package Manager                       │
│         APT (Debian/Ubuntu) / DNF (RHEL/CentOS)         │
│                                                          │
│  • Automatic dependency resolution                       │
│  • Version management                                    │
│  • Integrity verification                                │
└─────────────────────────────────────────────────────────┘
```

### Package Managers by Distribution

| Distribution | Package Format | Low-level Tool | High-level Tool |
|--------------|----------------|----------------|-----------------|
| Ubuntu/Debian | .deb | dpkg | apt |
| CentOS/RHEL 8+ | .rpm | rpm | dnf |
| CentOS/RHEL 7 | .rpm | rpm | yum |
| Fedora | .rpm | rpm | dnf |

---

## 2. APT (Ubuntu/Debian)

### Update Repository

```bash
# Update package list (required before installation)
sudo apt update
```

### Install Packages

```bash
# Install package
sudo apt install nginx

# Multiple packages
sudo apt install nginx php mysql-server

# Install without confirmation
sudo apt install -y vim

# Install specific version
sudo apt install nginx=1.18.0-0ubuntu1
```

### Remove Packages

```bash
# Remove package only
sudo apt remove nginx

# Remove package with configuration files
sudo apt purge nginx

# Remove unused dependencies
sudo apt autoremove

# Remove with automatic cleanup
sudo apt remove --autoremove nginx
```

### Update Packages

```bash
# Upgrade all installed packages
sudo apt upgrade

# Distribution upgrade (including dependency changes)
sudo apt full-upgrade

# Upgrade distribution version
sudo do-release-upgrade
```

### Search Packages

```bash
# Search for package
apt search nginx

# Package information
apt show nginx

# List installed packages
apt list --installed

# List upgradable packages
apt list --upgradable
```

### Clean Cache

```bash
# Clean downloaded package files
sudo apt clean

# Clean only old package files
sudo apt autoclean
```

---

## 3. DNF/YUM (CentOS/RHEL)

### DNF (RHEL 8+, CentOS 8+, Fedora)

```bash
# Check for updates
sudo dnf check-update

# Install package
sudo dnf install nginx

# Multiple packages
sudo dnf install nginx php mysql-server

# Install without confirmation
sudo dnf install -y vim

# Remove package
sudo dnf remove nginx

# Remove package with dependencies
sudo dnf autoremove nginx

# Update all packages
sudo dnf upgrade

# Search for package
dnf search nginx

# Package information
dnf info nginx

# List installed packages
dnf list installed

# Clean cache
sudo dnf clean all
```

### YUM (RHEL 7, CentOS 7)

```bash
# Install package
sudo yum install nginx

# Remove package
sudo yum remove nginx

# Update packages
sudo yum update

# Search for package
yum search nginx

# Package information
yum info nginx
```

---

## 4. Repository Management

### Ubuntu/Debian Repositories

#### /etc/apt/sources.list

```bash
# View repository list
cat /etc/apt/sources.list

# Additional repository directory
ls /etc/apt/sources.list.d/
```

#### Add PPA (Personal Package Archive)

```bash
# Add PPA
sudo add-apt-repository ppa:ondrej/php

# Remove PPA
sudo add-apt-repository --remove ppa:ondrej/php

# Update after adding repository
sudo apt update
```

#### Add External Repository

```bash
# Docker repository example
# 1. Add GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 2. Add repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 3. Update and install
sudo apt update
sudo apt install docker-ce
```

### CentOS/RHEL Repositories

#### Repository List

```bash
# List repositories
dnf repolist

# Detailed information
dnf repolist -v

# Repository configuration location
ls /etc/yum.repos.d/
```

#### Add Repository

```bash
# EPEL repository (Extra Packages for Enterprise Linux)
sudo dnf install epel-release

# Add repository file directly
sudo vi /etc/yum.repos.d/custom.repo
```

Repository file format:
```ini
[custom-repo]
name=Custom Repository
baseurl=https://example.com/repo/
enabled=1
gpgcheck=1
gpgkey=https://example.com/RPM-GPG-KEY
```

---

## 5. Package Information

### Ubuntu/Debian (dpkg)

```bash
# List installed packages
dpkg -l

# Search for specific package
dpkg -l | grep nginx

# Package status
dpkg -s nginx

# List files installed by package
dpkg -L nginx

# Find package that owns file
dpkg -S /usr/sbin/nginx
```

### CentOS/RHEL (rpm)

```bash
# List installed packages
rpm -qa

# Search for specific package
rpm -qa | grep nginx

# Package information
rpm -qi nginx

# List files installed by package
rpm -ql nginx

# Find package that owns file
rpm -qf /usr/sbin/nginx
```

---

## 6. Source Compilation

Compile software not available in repositories.

### Install Build Tools

#### Ubuntu/Debian

```bash
sudo apt install build-essential
```

#### CentOS/RHEL

```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install gcc gcc-c++ make
```

### Compilation Process

```bash
# 1. Download and extract source
wget https://example.com/software-1.0.tar.gz
tar -xzvf software-1.0.tar.gz
cd software-1.0

# 2. Check dependencies and configure
./configure --prefix=/usr/local

# 3. Compile
make

# 4. Install
sudo make install

# 5. Clean (optional)
make clean
```

### checkinstall (Manage as Package)

```bash
# Install
sudo apt install checkinstall

# Use instead of make install
sudo checkinstall

# Can be removed later with package manager
```

---

## 7. Practical Patterns

### System Update

#### Ubuntu/Debian

```bash
# Update script
#!/bin/bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt clean
```

#### CentOS/RHEL

```bash
# Update script
#!/bin/bash
sudo dnf check-update
sudo dnf upgrade -y
sudo dnf autoremove -y
sudo dnf clean all
```

### Install Essential Packages

#### Ubuntu/Debian

```bash
# Server basic packages
sudo apt install -y \
    vim \
    curl \
    wget \
    git \
    htop \
    net-tools \
    unzip \
    tree
```

#### CentOS/RHEL

```bash
# Server basic packages
sudo dnf install -y \
    vim \
    curl \
    wget \
    git \
    htop \
    net-tools \
    unzip \
    tree
```

### Package Pinning (Prevent Version Upgrade)

#### Ubuntu/Debian

```bash
# Pin package
sudo apt-mark hold nginx

# Unpin package
sudo apt-mark unhold nginx

# List pinned packages
apt-mark showhold
```

#### CentOS/RHEL

```bash
# Install versionlock plugin
sudo dnf install dnf-plugin-versionlock

# Pin package
sudo dnf versionlock add nginx

# Unpin package
sudo dnf versionlock delete nginx

# List pinned packages
dnf versionlock list
```

---

## 8. Troubleshooting

### Dependency Issues

#### Ubuntu/Debian

```bash
# Fix broken packages
sudo apt --fix-broken install

# Force install (caution)
sudo apt install -f

# Recover dpkg configuration
sudo dpkg --configure -a
```

#### CentOS/RHEL

```bash
# Clean dependencies
sudo dnf clean all
sudo dnf makecache

# Remove and reinstall problematic package
sudo dnf remove package_name
sudo dnf install package_name
```

### Lock Issues

#### Ubuntu/Debian

```bash
# Release apt lock (when another apt is running)
sudo rm /var/lib/dpkg/lock-frontend
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo dpkg --configure -a
```

---

## 9. CentOS End-of-Life and Migration

### CentOS EOL Status

CentOS Linux has reached end-of-life (EOL) for all versions:

- **CentOS 8**: EOL on December 31, 2021
- **CentOS 7**: EOL on June 30, 2024

**CentOS Stream** is now the only CentOS variant available, but it serves a different purpose:
- CentOS Stream is a **rolling-release** development platform
- Positioned **upstream** of RHEL (not downstream like CentOS Linux)
- **Not a 1:1 binary-compatible RHEL replacement**
- Receives updates before RHEL (cutting-edge, less stable)

### Migration Options

Organizations using CentOS need to migrate to alternative distributions:

| Distribution | Maintainer | RHEL Compatibility | Cost |
|-------------|-----------|-------------------|------|
| **Rocky Linux** | Rocky Enterprise Software Foundation | 1:1 binary compatible | Free |
| **AlmaLinux** | AlmaLinux OS Foundation (CloudLinux) | 1:1 binary compatible | Free |
| **Oracle Linux** | Oracle | Binary compatible | Free |
| **RHEL** | Red Hat | Original | Paid (free dev subscriptions available) |

#### Rocky Linux

- Founded by Gregory Kurtzer (original CentOS co-founder)
- Community-driven, non-profit
- 1:1 binary compatible with RHEL
- Active community and corporate support

```bash
# Check current CentOS version
cat /etc/redhat-release

# Migrate to Rocky Linux (CentOS 8)
sudo curl -O https://raw.githubusercontent.com/rocky-linux/rocky-tools/main/migrate2rocky/migrate2rocky.sh
sudo bash migrate2rocky.sh -r
```

#### AlmaLinux

- Backed by CloudLinux Inc.
- 1:1 binary compatible with RHEL
- Strong commercial support
- Well-established infrastructure

```bash
# Migrate to AlmaLinux (CentOS 8)
sudo curl -O https://raw.githubusercontent.com/AlmaLinux/almalinux-deploy/master/almalinux-deploy.sh
sudo bash almalinux-deploy.sh
```

#### Oracle Linux

- Maintained by Oracle
- Binary compatible with RHEL
- Option to use Unbreakable Enterprise Kernel (UEK)
- Free to use and distribute

```bash
# Migrate to Oracle Linux (CentOS 7/8)
sudo curl -O https://raw.githubusercontent.com/oracle/centos2ol/main/centos2ol.sh
sudo bash centos2ol.sh
```

### Key Differences: Rocky vs AlmaLinux

| Aspect | Rocky Linux | AlmaLinux |
|--------|------------|-----------|
| **Governance** | Community-driven foundation | CloudLinux-backed foundation |
| **Funding** | Donations, sponsors | CloudLinux Inc. + sponsors |
| **Release Cycle** | Typically tracks RHEL closely | Typically tracks RHEL closely |
| **Live Patching** | Limited | Available (paid KernelCare) |
| **Commercial Support** | Third-party vendors | CloudLinux + partners |

Both distributions are excellent choices and have very similar features. The choice often comes down to:
- **Rocky Linux**: If you prefer community-driven governance and CentOS legacy
- **AlmaLinux**: If you want corporate backing and optional commercial support

### Migration Best Practices

1. **Test first**: Migrate a non-production system first
2. **Backup**: Full system backup before migration
3. **Check compatibility**: Review third-party software compatibility
4. **Update before migration**: Ensure CentOS is fully updated
5. **Verify after migration**: Check services and applications post-migration

```bash
# Pre-migration checklist
# 1. List installed packages
rpm -qa > /root/packages-before.txt

# 2. Backup important configs
sudo tar -czf /root/etc-backup.tar.gz /etc

# 3. Update system fully
sudo yum update -y

# 4. Reboot to latest kernel
sudo reboot

# Post-migration verification
# 1. Check OS version
cat /etc/redhat-release

# 2. Verify package count
rpm -qa | wc -l

# 3. Check for broken dependencies
sudo dnf check

# 4. Verify services
sudo systemctl list-units --state=failed
```

### Recommendation

For most users migrating from CentOS:
- **Choose Rocky Linux or AlmaLinux** for production workloads
- Both provide stable, enterprise-grade RHEL replacements
- Avoid CentOS Stream for production unless you need cutting-edge features
- Consider RHEL if you need official Red Hat support

---

## 10. Practice Exercises

### Exercise 1: Search and Install Package

#### Ubuntu/Debian

```bash
# Search for htop
apt search htop

# Check information
apt show htop

# Install
sudo apt update
sudo apt install -y htop

# Verify
htop --version
```

#### CentOS/RHEL

```bash
# Search for htop
dnf search htop

# Check information
dnf info htop

# Install
sudo dnf install -y htop

# Verify
htop --version
```

### Exercise 2: Check Package Information

#### Ubuntu/Debian

```bash
# Count installed packages
dpkg -l | grep "^ii" | wc -l

# Files installed by specific package
dpkg -L bash | head -20

# Find package that owns file
dpkg -S /bin/bash
```

#### CentOS/RHEL

```bash
# Count installed packages
rpm -qa | wc -l

# Files installed by specific package
rpm -ql bash | head -20

# Find package that owns file
rpm -qf /bin/bash
```

### Exercise 3: System Update

#### Ubuntu/Debian

```bash
# Update repository list
sudo apt update

# Check upgradable packages
apt list --upgradable

# Upgrade
sudo apt upgrade -y

# Clean up
sudo apt autoremove -y
sudo apt clean
```

#### CentOS/RHEL

```bash
# Check for updates
sudo dnf check-update

# Upgrade
sudo dnf upgrade -y

# Clean up
sudo dnf autoremove -y
sudo dnf clean all
```

### Exercise 4: Remove Package

```bash
# Ubuntu/Debian
sudo apt remove htop
sudo apt purge htop    # Also remove configuration files
sudo apt autoremove

# CentOS/RHEL
sudo dnf remove htop
sudo dnf autoremove
```

---

**Previous**: [Process Management](./07_Process_Management.md) | **Next**: [Shell Scripting](./09_Shell_Scripting.md)
