# Virtualization (KVM)

**Previous**: [Kernel Management](./20_Kernel_Management.md) | **Next**: [Ansible Basics](./22_Ansible_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the KVM/QEMU architecture and how libvirt provides a unified management layer
2. Verify hardware virtualization support and install the KVM virtualization stack
3. Create virtual machines using virt-install with custom CPU, memory, disk, and network options
4. Manage VM lifecycle operations (start, stop, snapshot, migrate) with virsh commands
5. Configure virtual networking including NAT, bridged, and isolated networks
6. Perform live and offline VM migration between hosts
7. Manage storage pools, volumes, and disk images with virsh and qemu-img

---

## Table of Contents

1. [KVM/QEMU Overview](#1-kvmqemu-overview)
2. [Installation and Setup](#2-installation-and-setup)
3. [VM Creation](#3-vm-creation)
4. [virsh Commands](#4-virsh-commands)
5. [Network Configuration](#5-network-configuration)
6. [Storage Management](#6-storage-management)
7. [Snapshots and Migration](#7-snapshots-and-migration)

---

KVM turns the Linux kernel itself into a hypervisor, providing near-native performance for virtual machines without the cost of proprietary solutions. Whether you are running development environments, consolidating servers, or building private cloud infrastructure, KVM is the foundation of virtualization on Linux. This lesson teaches you to create, manage, and migrate VMs using the same tools that power OpenStack and major hosting providers.

## 1. KVM/QEMU Overview

### Virtualization Types

```
┌─────────────────────────────────────────────────────────────┐
│  Type 1 (Bare-metal)          Type 2 (Hosted)               │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ VM │ VM │ VM   │          │ VM │ VM │ VM   │          │
│  ├─────────────────┤          ├─────────────────┤          │
│  │   Hypervisor    │          │   Hypervisor    │          │
│  ├─────────────────┤          ├─────────────────┤          │
│  │    Hardware     │          │   Host OS       │          │
│  └─────────────────┘          ├─────────────────┤          │
│  ESXi, Xen, Hyper-V           │    Hardware     │          │
│                               └─────────────────┘          │
│                               VirtualBox, VMware Workstation│
│                                                             │
│  KVM - Linux kernel acts as Hypervisor (close to Type 1)   │
└─────────────────────────────────────────────────────────────┘
```

### KVM/QEMU Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Virtual Machine                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Guest OS (Linux, Windows, etc.)                     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  QEMU: Device emulation (I/O, network, disk)               │
├─────────────────────────────────────────────────────────────┤
│  KVM kernel module: CPU/memory virtualization (HW support)  │
├─────────────────────────────────────────────────────────────┤
│  Linux Kernel                                                │
├─────────────────────────────────────────────────────────────┤
│  Hardware (VT-x/AMD-V, VT-d)                                │
└─────────────────────────────────────────────────────────────┘
```

### libvirt Management Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Management Tools                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ virsh    │  │virt-manager│ │Cockpit   │  │ API      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │   libvirt   │                         │
│                    │ (libvirtd)  │                         │
│                    └──────┬──────┘                         │
│                           │                                 │
│  ┌────────────────────────┼────────────────────────┐       │
│  │            │           │           │            │       │
│  │  ┌─────┐   │   ┌─────┐│   ┌─────┐│   ┌─────┐  │       │
│  │  │ KVM │   │   │ QEMU││   │ LXC ││   │ Xen │  │       │
│  │  └─────┘   │   └─────┘│   └─────┘│   └─────┘  │       │
│  └────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Installation and Setup

### Check Hardware Virtualization Support

```bash
# Check CPU virtualization support
grep -E '(vmx|svm)' /proc/cpuinfo

# vmx: Intel VT-x
# svm: AMD-V

# or
lscpu | grep Virtualization

# Check KVM module
lsmod | grep kvm
```

### Package Installation

```bash
# Ubuntu/Debian
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients \
    bridge-utils virt-manager virtinst

# RHEL/CentOS
sudo yum install qemu-kvm libvirt libvirt-python \
    libguestfs-tools virt-install virt-manager

# Fedora
sudo dnf install @virtualization
```

### Start Services

```bash
# Start libvirtd
sudo systemctl enable --now libvirtd

# Check status
sudo systemctl status libvirtd

# Add user to libvirt group (requires re-login)
sudo usermod -aG libvirt $USER
sudo usermod -aG kvm $USER
```

### Verify Connection

```bash
# Test local connection
virsh -c qemu:///system list --all

# Or without sudo
virsh list --all

# System information
virsh nodeinfo
```

---

## 3. VM Creation

### VM Creation with virt-install

```bash
# Basic VM creation
virt-install \
    --name ubuntu-vm \
    --ram 2048 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/ubuntu-vm.qcow2,size=20 \
    --os-variant ubuntu22.04 \
    --network bridge=virbr0 \
    --graphics vnc,listen=0.0.0.0 \
    --cdrom /path/to/ubuntu-22.04.iso \
    --boot cdrom,hd
```

### Detailed Options

```bash
virt-install \
    --name centos-vm \
    --memory 4096 \
    --vcpus 4,maxvcpus=8 \
    --cpu host-passthrough \
    --disk path=/var/lib/libvirt/images/centos-vm.qcow2,size=40,format=qcow2,bus=virtio \
    --disk path=/var/lib/libvirt/images/centos-data.qcow2,size=100,format=qcow2 \
    --os-variant centos-stream9 \
    --network network=default,model=virtio \
    --graphics spice,listen=0.0.0.0 \
    --video qxl \
    --channel spicevmc \
    --location /path/to/CentOS-Stream-9.iso \
    --extra-args "console=ttyS0,115200n8 serial" \
    --initrd-inject /path/to/kickstart.cfg \
    --extra-args "ks=file:/kickstart.cfg" \
    --noautoconsole
```

### Check OS Variant List

```bash
# List supported OS
osinfo-query os

# Search specific OS
osinfo-query os | grep -i ubuntu
osinfo-query os | grep -i centos
```

### VM Creation with XML Definition

```bash
# VM definition XML example (/tmp/vm-definition.xml)
virsh define /tmp/vm-definition.xml

# Dump XML of existing VM
virsh dumpxml ubuntu-vm > ubuntu-vm.xml
```

```xml
<!-- vm-definition.xml example -->
<domain type='kvm'>
  <name>test-vm</name>
  <memory unit='GiB'>2</memory>
  <vcpu placement='static'>2</vcpu>
  <os>
    <type arch='x86_64' machine='pc-q35-6.2'>hvm</type>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-passthrough'/>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file='/var/lib/libvirt/images/test-vm.qcow2'/>
      <target dev='vda' bus='virtio'/>
    </disk>
    <interface type='network'>
      <source network='default'/>
      <model type='virtio'/>
    </interface>
    <graphics type='vnc' port='-1' autoport='yes'/>
    <console type='pty'/>
  </devices>
</domain>
```

---

## 4. virsh Commands

### VM State Management

```bash
# VM list
virsh list             # Running VMs
virsh list --all       # All VMs
virsh list --inactive  # Stopped VMs

# Start/stop VM
virsh start vm-name
virsh shutdown vm-name     # Graceful shutdown
virsh destroy vm-name      # Force shutdown (power off)
virsh reboot vm-name

# Suspend/resume VM
virsh suspend vm-name
virsh resume vm-name

# Auto-start configuration
virsh autostart vm-name
virsh autostart --disable vm-name
```

### VM Information Query

```bash
# Basic info
virsh dominfo vm-name

# Detailed configuration (XML)
virsh dumpxml vm-name

# CPU info
virsh vcpuinfo vm-name

# Memory info
virsh dommemstat vm-name

# Block device info
virsh domblklist vm-name
virsh domblkinfo vm-name vda

# Network interfaces
virsh domiflist vm-name
virsh domifstat vm-name vnet0
```

### VM Console Access

```bash
# Serial console (requires guest configuration)
virsh console vm-name

# VNC/SPICE port check
virsh domdisplay vm-name
virsh vncdisplay vm-name

# Connect with virt-viewer
virt-viewer vm-name
```

### VM Resource Changes

```bash
# Change CPU count (online)
virsh setvcpus vm-name 4 --live

# Change memory (online, requires pre-configuration)
virsh setmem vm-name 4G --live

# Set maximum (offline)
virsh setmaxmem vm-name 8G --config
virsh setvcpus vm-name 8 --maximum --config

# Edit XML directly
virsh edit vm-name
```

### VM Deletion

```bash
# Delete VM definition (keep disk)
virsh undefine vm-name

# Delete VM and related storage
virsh undefine vm-name --remove-all-storage

# Delete including NVRAM (UEFI)
virsh undefine vm-name --nvram
```

---

## 5. Network Configuration

### Default NAT Network

```bash
# Check default network
virsh net-list --all

# Start default network
virsh net-start default
virsh net-autostart default

# Network information
virsh net-info default
virsh net-dumpxml default
```

### Bridge Network Setup

```bash
# Ubuntu: netplan configuration
# /etc/netplan/01-bridge.yaml
```

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
  bridges:
    br0:
      interfaces:
        - enp3s0
      dhcp4: yes
      parameters:
        stp: false
        forward-delay: 0
```

```bash
# Apply
sudo netplan apply
```

```bash
# RHEL/CentOS: NetworkManager
sudo nmcli connection add type bridge con-name br0 ifname br0
sudo nmcli connection add type ethernet con-name br0-slave-enp3s0 ifname enp3s0 master br0
sudo nmcli connection modify br0 ipv4.method auto
sudo nmcli connection up br0
```

### libvirt Bridge Network Definition

```xml
<!-- /tmp/bridge-network.xml -->
<network>
  <name>host-bridge</name>
  <forward mode='bridge'/>
  <bridge name='br0'/>
</network>
```

```bash
# Create network
virsh net-define /tmp/bridge-network.xml
virsh net-start host-bridge
virsh net-autostart host-bridge
```

### Isolated Network

```xml
<!-- /tmp/isolated-network.xml -->
<network>
  <name>isolated</name>
  <bridge name='virbr1'/>
  <ip address='192.168.100.1' netmask='255.255.255.0'>
    <dhcp>
      <range start='192.168.100.10' end='192.168.100.254'/>
    </dhcp>
  </ip>
</network>
```

### Adding Network Interface to VM

```bash
# Add interface (online)
virsh attach-interface vm-name network default --model virtio --live --config

# Remove interface
virsh detach-interface vm-name network --mac 52:54:00:xx:xx:xx --live --config

# List interfaces
virsh domiflist vm-name
```

---

## 6. Storage Management

### Storage Pools

```bash
# Default pool (directory-based)
virsh pool-list --all

# Create pool (directory)
virsh pool-define-as mypool dir --target /var/lib/libvirt/images/mypool
virsh pool-build mypool
virsh pool-start mypool
virsh pool-autostart mypool

# LVM pool
virsh pool-define-as lvm-pool logical --source-name vg_vms --target /dev/vg_vms

# Pool information
virsh pool-info default
```

### Volume Management

```bash
# List volumes
virsh vol-list default

# Create volume
virsh vol-create-as default disk1.qcow2 20G --format qcow2

# Volume information
virsh vol-info disk1.qcow2 --pool default
virsh vol-dumpxml disk1.qcow2 --pool default

# Delete volume
virsh vol-delete disk1.qcow2 --pool default

# Clone volume
virsh vol-clone disk1.qcow2 disk1-clone.qcow2 --pool default

# Resize volume
virsh vol-resize disk1.qcow2 30G --pool default
```

### Disk Image Management (qemu-img)

```bash
# Create image
qemu-img create -f qcow2 disk.qcow2 20G

# Image information
qemu-img info disk.qcow2

# Convert image
qemu-img convert -f raw -O qcow2 disk.raw disk.qcow2
qemu-img convert -f vmdk -O qcow2 disk.vmdk disk.qcow2

# Resize image
qemu-img resize disk.qcow2 +10G

# Compress sparse image
qemu-img convert -O qcow2 -c disk.qcow2 disk-compressed.qcow2
```

### Adding Disk to VM

```bash
# Add disk (online)
virsh attach-disk vm-name /var/lib/libvirt/images/extra.qcow2 vdb \
    --driver qemu --subdriver qcow2 --live --config

# Remove disk
virsh detach-disk vm-name vdb --live --config

# List block devices
virsh domblklist vm-name
```

---

## 7. Snapshots and Migration

### Snapshot Management

```bash
# Create snapshot
virsh snapshot-create-as vm-name snap1 "First snapshot" --atomic

# Disk-only snapshot (no memory)
virsh snapshot-create-as vm-name snap-disk --disk-only --atomic

# List snapshots
virsh snapshot-list vm-name

# Snapshot information
virsh snapshot-info vm-name snap1

# Revert to snapshot
virsh snapshot-revert vm-name snap1

# Delete snapshot
virsh snapshot-delete vm-name snap1

# Check current snapshot
virsh snapshot-current vm-name
```

### External Snapshots

```bash
# Create external snapshot (recommended for production)
virsh snapshot-create-as vm-name snap-external \
    --diskspec vda,snapshot=external \
    --disk-only --atomic

# Commit external snapshot (merge)
virsh blockcommit vm-name vda --active --pivot

# Block job information
virsh blockjob vm-name vda --info
```

### Live Migration

```bash
# Target host preparation
# - Same libvirt version
# - Shared storage (NFS, GlusterFS, Ceph, etc.)
# - Network connectivity

# Execute migration
virsh migrate --live vm-name qemu+ssh://target-host/system

# With options
virsh migrate --live --persistent --undefinesource \
    --copy-storage-all \
    vm-name qemu+ssh://target-host/system

# Tunnelled migration (NAT environment)
virsh migrate --live --p2p --tunnelled \
    vm-name qemu+ssh://target-host/system

# Check migration status
virsh domjobinfo vm-name
```

### Offline Migration

```bash
# Dump XML from source
virsh dumpxml vm-name > vm.xml

# Copy disk image
rsync -av /var/lib/libvirt/images/vm-disk.qcow2 target-host:/var/lib/libvirt/images/

# Define on target
virsh define vm.xml

# Delete from source
virsh undefine vm-name
```

---

## Practice Problems

### Problem 1: VM Creation

Create a VM with the following specifications using virt-install:
- Name: test-server
- Memory: 2GB
- CPU: 2
- Disk: 20GB (qcow2)
- Network: default (NAT)
- Graphics: VNC

### Problem 2: Network Configuration

Create an isolated internal network:
- Name: internal
- Subnet: 10.10.10.0/24
- DHCP: 10.10.10.100-200
- No NAT

### Problem 3: Snapshot Management

1. Create a snapshot of a VM
2. Make changes to the VM (create a file, etc.)
3. Revert to the snapshot
4. Verify that changes were undone

---

## Answers

### Problem 1 Answer

```bash
virt-install \
    --name test-server \
    --memory 2048 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/test-server.qcow2,size=20,format=qcow2 \
    --os-variant generic \
    --network network=default \
    --graphics vnc,listen=0.0.0.0 \
    --cdrom /path/to/installer.iso \
    --boot cdrom,hd
```

### Problem 2 Answer

```xml
<!-- /tmp/internal-net.xml -->
<network>
  <name>internal</name>
  <bridge name='virbr-int'/>
  <ip address='10.10.10.1' netmask='255.255.255.0'>
    <dhcp>
      <range start='10.10.10.100' end='10.10.10.200'/>
    </dhcp>
  </ip>
</network>
```

```bash
virsh net-define /tmp/internal-net.xml
virsh net-start internal
virsh net-autostart internal
```

### Problem 3 Answer

```bash
# 1. Create snapshot
virsh snapshot-create-as vm-name before-change "Before changes"

# 2. Make changes inside VM
virsh console vm-name
# (inside guest) touch /tmp/test-file

# 3. Revert to snapshot
virsh snapshot-revert vm-name before-change

# 4. Verify (file should not exist)
virsh console vm-name
# (inside guest) ls /tmp/test-file  # No such file
```

---

---

## References

- [libvirt Documentation](https://libvirt.org/docs.html)
- [KVM Documentation](https://www.linux-kvm.org/page/Documents)
- [Red Hat Virtualization Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/configuring_and_managing_virtualization/index)
- `man virsh`, `man virt-install`, `man qemu-img`

---

**Previous**: [Kernel Management](./20_Kernel_Management.md) | **Next**: [Ansible Basics](./22_Ansible_Basics.md)
