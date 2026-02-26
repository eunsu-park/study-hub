# Kernel Management

**Previous**: [Backup and Recovery](./19_Backup_Recovery.md) | **Next**: [Virtualization (KVM)](./21_Virtualization_KVM.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Interpret Linux kernel version numbers and manage installed kernel packages
2. Load, unload, and configure kernel modules using modprobe and persistent configuration files
3. Blacklist problematic modules and manage automatic module loading with DKMS
4. Download kernel source, configure build options, and compile a custom kernel
5. Configure the GRUB bootloader including default entries, timeout, and kernel parameters
6. Manage initramfs images and tune runtime kernel parameters with sysctl

---

## Table of Contents

1. [Kernel Overview](#1-kernel-overview)
2. [Kernel Version Management](#2-kernel-version-management)
3. [Kernel Modules](#3-kernel-modules)
4. [DKMS](#4-dkms)
5. [Kernel Compilation](#5-kernel-compilation)
6. [GRUB Bootloader](#6-grub-bootloader)
7. [Kernel Parameters](#7-kernel-parameters)

---

The kernel is the heart of every Linux system -- it manages hardware, schedules processes, and enforces security boundaries. While most administrators use distribution-provided kernels, understanding how to manage kernel versions, load modules, and configure boot parameters is essential for tasks like enabling hardware support, applying security patches, optimizing server workloads, and recovering from boot failures.

## 1. Kernel Overview

### Role of the Kernel

```
┌─────────────────────────────────────────────────────────────┐
│                      User Space                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │Application│ │  Shell  │  │ Services│  │   GUI   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
├───────┴────────────┴────────────┴────────────┴──────────────┤
│                   System Call Interface                      │
├─────────────────────────────────────────────────────────────┤
│                      Kernel Space                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Process Mgmt  │  Memory Mgmt  │  Filesystem  │ Network │ │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Device Drivers (Modules)                │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                        Hardware                              │
│  CPU  │  Memory  │  Disk  │  Network Card  │  Other Devices │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Components

| Component | Role |
|-----------|------|
| **Process Management** | Process creation, scheduling, termination |
| **Memory Management** | Virtual memory, paging, caching |
| **Filesystem** | VFS, ext4, XFS, NFS, etc. |
| **Networking** | TCP/IP stack, sockets, routing |
| **Device Drivers** | Hardware abstraction, modules |
| **System Calls** | Interface between user space and kernel |

---

## 2. Kernel Version Management

### Current Kernel Information

```bash
# Check kernel version
uname -r
# Example: 5.15.0-91-generic

# Detailed information
uname -a
# Linux hostname 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux

# Kernel configuration info
cat /proc/version

# Kernel build configuration (if compiled with config)
cat /boot/config-$(uname -r) | head -20
```

### Kernel Version Scheme

```
5.15.0-91-generic
│ │  │ │  └──────── Distribution-specific name
│ │  │ └────────── ABI version (distribution patches)
│ │  └──────────── Patch level
│ └─────────────── Minor version
└───────────────── Major version
```

### Installed Kernel List

```bash
# Ubuntu/Debian
dpkg --list | grep linux-image

# RHEL/CentOS
rpm -qa | grep kernel

# Or check /boot
ls -la /boot/vmlinuz-*
```

### Kernel Update

```bash
# Ubuntu/Debian
sudo apt update
sudo apt upgrade linux-image-generic

# Install specific version
sudo apt install linux-image-5.15.0-92-generic

# RHEL/CentOS
sudo yum update kernel

# Install specific version
sudo yum install kernel-5.14.0-362.el9
```

### Removing Old Kernels

```bash
# Ubuntu - automatic removal
sudo apt autoremove

# Ubuntu - remove specific version
sudo apt remove linux-image-5.15.0-88-generic

# Remove old kernels except current
sudo apt purge $(dpkg --list | grep -E 'linux-(image|headers|modules)' | \
    grep -v $(uname -r | sed 's/-generic//') | awk '{print $2}')

# RHEL/CentOS - remove old kernels (keep 2)
sudo dnf remove $(dnf repoquery --installonly --latest-limit=-2 -q)
```

---

## 3. Kernel Modules

### Module Information

```bash
# List loaded modules
lsmod

# Specific module info
modinfo ext4
modinfo nvidia

# Check module dependencies
modprobe --show-depends ext4

# Module file location
ls /lib/modules/$(uname -r)/kernel/
```

### Loading/Unloading Modules

```bash
# Load module
sudo modprobe nouveau
sudo modprobe snd-hda-intel

# Unload module
sudo modprobe -r nouveau
sudo rmmod nouveau

# Force unload (dangerous)
sudo rmmod -f nouveau

# Load with dependencies
sudo modprobe -v nvidia
```

### Module Parameters

```bash
# Check module parameters
modinfo -p e1000e

# Check currently applied parameters
cat /sys/module/e1000e/parameters/IntMode

# Load with parameters
sudo modprobe e1000e IntMode=2

# /etc/modprobe.d/ configuration file
echo "options e1000e IntMode=2" | sudo tee /etc/modprobe.d/e1000e.conf
```

### Auto-loading Module Configuration

```bash
# Auto-load at boot
echo "vhost_net" | sudo tee /etc/modules-load.d/vhost_net.conf

# Module blacklist (prevent loading)
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# Requires initramfs regeneration
sudo update-initramfs -u  # Ubuntu/Debian
sudo dracut -f            # RHEL/CentOS
```

### Module Aliases

```bash
# Check alias
modprobe --show-depends -a pci:v00008086d00001502sv*sd*bc*sc*i*

# Find module by hardware ID
lspci -nn  # Check PCI devices and IDs
lspci -k   # Check modules in use

# USB devices
lsusb
lsusb -t  # Tree format
```

---

## 4. DKMS

### DKMS Overview

DKMS (Dynamic Kernel Module Support) automatically rebuilds external modules on kernel updates.

```bash
# Install DKMS
sudo apt install dkms  # Ubuntu/Debian
sudo yum install dkms  # RHEL/CentOS
```

### DKMS Status Check

```bash
# List registered modules
dkms status

# Example output:
# nvidia/535.154.05, 5.15.0-91-generic, x86_64: installed
# nvidia/535.154.05, 5.15.0-92-generic, x86_64: installed
# virtualbox/7.0.12_Ubuntu, 5.15.0-91-generic, x86_64: installed
```

### DKMS Module Management

```bash
# Add module
sudo dkms add -m module-name -v version

# Build module
sudo dkms build -m module-name -v version

# Install module
sudo dkms install -m module-name -v version

# Remove module
sudo dkms remove -m module-name -v version --all

# Rebuild for all kernels
sudo dkms autoinstall
```

### DKMS Module Creation Example

```bash
# Module directory structure
/usr/src/mymodule-1.0/
├── dkms.conf
├── Makefile
└── mymodule.c
```

```bash
# dkms.conf example
PACKAGE_NAME="mymodule"
PACKAGE_VERSION="1.0"
BUILT_MODULE_NAME[0]="mymodule"
DEST_MODULE_LOCATION[0]="/kernel/drivers/misc"
AUTOINSTALL="yes"
```

```bash
# Makefile example
obj-m := mymodule.o

KVERSION := $(shell uname -r)
KDIR := /lib/modules/$(KVERSION)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```

```bash
# Register with DKMS
sudo dkms add -m mymodule -v 1.0
sudo dkms build -m mymodule -v 1.0
sudo dkms install -m mymodule -v 1.0
```

---

## 5. Kernel Compilation

### Source Download

```bash
# Download from kernel.org
cd /usr/src
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.7.tar.xz
tar -xvf linux-6.7.tar.xz
cd linux-6.7

# Or via git
git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux
git checkout v6.7
```

### Install Build Dependencies

```bash
# Ubuntu/Debian
sudo apt install build-essential libncurses-dev bison flex \
    libssl-dev libelf-dev bc dwarves

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install ncurses-devel bison flex elfutils-libelf-devel \
    openssl-devel bc dwarves
```

### Kernel Configuration

```bash
# Copy current kernel configuration
cp /boot/config-$(uname -r) .config

# Menu-based configuration
make menuconfig

# Or graphical configuration (requires X11)
make xconfig

# Or text query mode
make config

# Query only new options based on existing config
make oldconfig

# Set new options to defaults
make olddefconfig
```

### Key Configuration Options

```
General setup --->
    Local version - append to kernel release: -custom
    [*] Automatically append version information

Processor type and features --->
    Processor family (Core 2/newer Xeon) --->
    [*] Symmetric multi-processing support

Device Drivers --->
    [M] Network device support --->
    [M] SCSI device support --->

File systems --->
    <*> Ext4 POSIX Access Control Lists
    <M> XFS filesystem support
    <M> Btrfs filesystem support
```

### Compilation and Installation

```bash
# Compile (parallel build)
make -j$(nproc)

# Compile modules
make modules

# Install modules
sudo make modules_install

# Install kernel
sudo make install

# Generate initramfs (if not done automatically)
sudo update-initramfs -c -k 6.7.0-custom

# Update GRUB
sudo update-grub
```

### Quick Test Build

```bash
# Start with minimal configuration
make tinyconfig

# Or minimal config for current hardware
make localmodconfig

# Build
make -j$(nproc)
```

---

## 6. GRUB Bootloader

### GRUB Configuration Files

```bash
# Main configuration (do not modify)
/boot/grub/grub.cfg

# User configuration (modify this)
/etc/default/grub

# Custom scripts
/etc/grub.d/
```

### /etc/default/grub Configuration

```bash
# /etc/default/grub

# Default boot entry (starts from 0, or "saved")
GRUB_DEFAULT=0

# Menu display time (seconds)
GRUB_TIMEOUT=5

# Menu hidden (faster boot)
GRUB_TIMEOUT_STYLE=menu  # menu, countdown, hidden

# Kernel parameters
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX=""

# Resolution
GRUB_GFXMODE=1920x1080

# Disable recovery mode entry
GRUB_DISABLE_RECOVERY="false"

# OS prober (detect other OS)
GRUB_DISABLE_OS_PROBER=false
```

### GRUB Update

```bash
# Apply after configuration changes
sudo update-grub  # Ubuntu/Debian
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/CentOS

# BIOS system
sudo grub2-mkconfig -o /boot/grub2/grub.cfg

# UEFI system
sudo grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg
```

### GRUB Reinstallation

```bash
# BIOS system
sudo grub-install /dev/sda
sudo update-grub

# UEFI system
sudo grub-install --target=x86_64-efi --efi-directory=/boot/efi
sudo update-grub
```

### Manually Adding Boot Entries

```bash
# /etc/grub.d/40_custom
#!/bin/sh
exec tail -n +3 $0

menuentry "Custom Kernel" {
    set root='hd0,msdos1'
    linux /boot/vmlinuz-6.7.0-custom root=/dev/sda2 ro quiet
    initrd /boot/initrd.img-6.7.0-custom
}

menuentry "Recovery Mode" {
    set root='hd0,msdos1'
    linux /boot/vmlinuz-6.7.0-custom root=/dev/sda2 ro single
    initrd /boot/initrd.img-6.7.0-custom
}
```

### Changing Default Boot Kernel

```bash
# Check available menu entries
grep -E "^menuentry|^submenu" /boot/grub/grub.cfg

# Set default (by index)
sudo grub-set-default 2

# Set default (by name)
sudo grub-set-default "Ubuntu, with Linux 5.15.0-92-generic"

# One-time boot selection
sudo grub-reboot "Ubuntu, with Linux 5.15.0-91-generic"

# Set GRUB_DEFAULT in /etc/default/grub
GRUB_DEFAULT="1>2"  # 3rd item of submenu 1
```

---

## 7. Kernel Parameters

### Boot-time Parameters

```bash
# Set in /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX="crashkernel=auto rd.lvm.lv=centos/root"

# Or press 'e' in GRUB menu to edit temporarily
linux /boot/vmlinuz-... root=... quiet splash nouveau.modeset=0
```

### Common Kernel Parameters

| Parameter | Description |
|-----------|-------------|
| `quiet` | Minimize boot messages |
| `splash` | Graphical splash screen |
| `single` | Single user mode |
| `init=/bin/bash` | Boot directly to shell |
| `nomodeset` | Disable video mode setting |
| `acpi=off` | Disable ACPI |
| `noapic` | Disable APIC |
| `mem=4G` | Limit usable memory |
| `maxcpus=2` | Limit CPU count |

### Runtime Parameters (sysctl)

```bash
# Check current value
sysctl -a
sysctl net.ipv4.ip_forward

# Temporary change
sudo sysctl -w net.ipv4.ip_forward=1

# Permanent setting
echo "net.ipv4.ip_forward = 1" | sudo tee /etc/sysctl.d/99-custom.conf
sudo sysctl -p /etc/sysctl.d/99-custom.conf

# Or edit /etc/sysctl.conf then
sudo sysctl -p
```

### Key sysctl Parameters

```bash
# /etc/sysctl.d/99-custom.conf

# Network
net.ipv4.ip_forward = 1
net.ipv4.tcp_syncookies = 1
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# Memory
vm.swappiness = 10
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10

# Filesystem
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Security
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
```

### initramfs Management

```bash
# Regenerate initramfs for current kernel
sudo update-initramfs -u

# Specific kernel version
sudo update-initramfs -u -k 5.15.0-92-generic

# Create new
sudo update-initramfs -c -k 5.15.0-92-generic

# RHEL/CentOS (dracut)
sudo dracut -f
sudo dracut -f /boot/initramfs-$(uname -r).img $(uname -r)

# Check contents
lsinitramfs /boot/initrd.img-$(uname -r) | head -50
```

---

## Practice Problems

### Problem 1: Module Management

1. Add the `nouveau` driver to the blacklist and verify.
2. Assume a scenario using NVIDIA proprietary drivers.

### Problem 2: GRUB Configuration

Write a GRUB configuration that meets the following requirements:
- Boot timeout 10 seconds
- Default kernel: second entry
- Memory limit 4GB
- Quiet boot

### Problem 3: sysctl Configuration

Write sysctl settings optimized for a web server:
- Increase connection backlog
- Increase file descriptor limit
- TCP tuning

---

## Answers

### Problem 1 Answer

```bash
# Blacklist nouveau
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# Update initramfs
sudo update-initramfs -u  # Ubuntu
sudo dracut -f            # RHEL

# Verify
cat /etc/modprobe.d/blacklist-nouveau.conf

# Check after reboot
lsmod | grep nouveau  # Should show no output
```

### Problem 2 Answer

```bash
# /etc/default/grub
GRUB_DEFAULT=1
GRUB_TIMEOUT=10
GRUB_TIMEOUT_STYLE=menu
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash mem=4G"
GRUB_CMDLINE_LINUX=""
```

```bash
# Apply
sudo update-grub
```

### Problem 3 Answer

```bash
# /etc/sysctl.d/99-webserver.conf

# Connection backlog
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# File descriptors
fs.file-max = 2097152

# TCP tuning
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# Memory buffer
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 12582912 16777216
net.ipv4.tcp_wmem = 4096 12582912 16777216
```

```bash
# Apply
sudo sysctl -p /etc/sysctl.d/99-webserver.conf
```

---

---

## References

- [The Linux Kernel Archives](https://www.kernel.org/)
- [Kernel Documentation](https://www.kernel.org/doc/html/latest/)
- [GRUB Manual](https://www.gnu.org/software/grub/manual/)
- `man modprobe`, `man dkms`, `man sysctl`

---

**Previous**: [Backup and Recovery](./19_Backup_Recovery.md) | **Next**: [Virtualization (KVM)](./21_Virtualization_KVM.md)
