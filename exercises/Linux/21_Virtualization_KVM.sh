#!/bin/bash
# Exercises for Lesson 21: Virtualization (KVM)
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: VM Creation ===
# Problem: Create a VM with virt-install: test-server, 2GB RAM, 2 CPUs,
#          20GB qcow2 disk, NAT network, VNC graphics.
exercise_1() {
    echo "=== Exercise 1: VM Creation with virt-install ==="
    echo ""
    echo "Scenario: Create a KVM virtual machine with specific hardware specs."
    echo ""

    echo "Solution:"
    echo ""
    cat << 'CMD'
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
CMD
    echo ""

    echo "Flag explanations:"
    echo "  --name test-server"
    echo "      VM name used by virsh commands (must be unique)"
    echo ""
    echo "  --memory 2048"
    echo "      RAM in MiB (2048 MiB = 2 GB)"
    echo ""
    echo "  --vcpus 2"
    echo "      Virtual CPUs assigned to the VM"
    echo ""
    echo "  --disk path=...,size=20,format=qcow2"
    echo "      path  = location of virtual disk image"
    echo "      size  = disk size in GB"
    echo "      qcow2 = QEMU Copy-On-Write v2 format"
    echo "              Advantages: thin provisioning, snapshots, compression"
    echo "              Alternative: 'raw' for better I/O performance"
    echo ""
    echo "  --os-variant generic"
    echo "      OS type hint for optimization. Use 'osinfo-query os' for full list."
    echo "      Examples: ubuntu22.04, rhel9.0, win10"
    echo ""
    echo "  --network network=default"
    echo "      Connect to libvirt's default NAT network (virbr0, 192.168.122.0/24)"
    echo "      Other options: bridge=br0 (bridged), none (no network)"
    echo ""
    echo "  --graphics vnc,listen=0.0.0.0"
    echo "      VNC console accessible from any IP (use 127.0.0.1 for local only)"
    echo "      Connect with: virt-viewer test-server  or  VNC client to port 5900+"
    echo ""
    echo "  --cdrom /path/to/installer.iso"
    echo "      ISO image for OS installation"
    echo ""
    echo "  --boot cdrom,hd"
    echo "      Boot order: try CD-ROM first, then hard disk"
    echo ""

    echo "Post-creation management commands:"
    echo "  virsh list --all          # List all VMs (running and stopped)"
    echo "  virsh start test-server   # Start the VM"
    echo "  virsh console test-server # Attach to serial console"
    echo "  virsh shutdown test-server # Graceful shutdown (ACPI)"
    echo "  virsh destroy test-server  # Force power off (like pulling the plug)"
    echo "  virsh undefine test-server --remove-all-storage  # Delete VM and disk"
}

# === Exercise 2: Network Configuration ===
# Problem: Create an isolated internal network with specific subnet and DHCP range.
exercise_2() {
    echo "=== Exercise 2: Isolated Network Configuration ==="
    echo ""
    echo "Scenario: Create a private network (no NAT) for inter-VM communication."
    echo ""

    echo "--- Step 1: Create the network XML definition ---"
    echo ""
    cat << 'XML'
<!-- /tmp/internal-net.xml -->
<network>
  <name>internal</name>
  <!-- Note: No <forward> element = isolated network (no NAT, no routing) -->
  <!-- VMs on this network can talk to each other but NOT to the host or internet -->

  <bridge name='virbr-int'/>
  <!-- Creates a virtual bridge interface on the host -->

  <ip address='10.10.10.1' netmask='255.255.255.0'>
    <!-- Host gets 10.10.10.1; this is also the DHCP server address -->
    <dhcp>
      <range start='10.10.10.100' end='10.10.10.200'/>
      <!-- VMs get IPs from .100 to .200 via DHCP -->
      <!-- .2-.99 available for static assignments -->
    </dhcp>
  </ip>
</network>
XML
    echo ""

    echo "--- Step 2: Define, start, and auto-start the network ---"
    echo "  virsh net-define /tmp/internal-net.xml   # Register the network definition"
    echo "  virsh net-start internal                 # Activate the network"
    echo "  virsh net-autostart internal             # Start automatically on host boot"
    echo ""

    echo "--- Step 3: Verify ---"
    echo "  virsh net-list --all            # List all networks"
    echo "  virsh net-info internal         # Show network details"
    echo "  virsh net-dumpxml internal      # Show full XML config"
    echo "  ip addr show virbr-int          # Verify bridge interface on host"
    echo ""

    echo "--- Step 4: Attach a VM to this network ---"
    echo "  # During VM creation:"
    echo "  virt-install ... --network network=internal ..."
    echo ""
    echo "  # Add to existing VM (hot-add):"
    echo "  virsh attach-interface test-server network internal --model virtio --live"
    echo ""

    echo "Network types comparison:"
    echo "  NAT (default):      VMs -> Internet via host NAT (most common)"
    echo "  Isolated:           VMs <-> VMs only, no external access"
    echo "  Bridged:            VMs appear as physical hosts on LAN"
    echo "  Routed:             VMs accessible from LAN without NAT (requires routing)"
    echo "  Open vSwitch:       SDN-capable virtual switching"
}

# === Exercise 3: Snapshot Management ===
# Problem: Create, test, and revert a VM snapshot.
exercise_3() {
    echo "=== Exercise 3: Snapshot Management ==="
    echo ""
    echo "Scenario: Use snapshots to create a safe rollback point before changes."
    echo ""

    echo "--- Step 1: Create a snapshot ---"
    echo "  virsh snapshot-create-as vm-name before-change \"Snapshot before changes\""
    echo ""
    echo "  Flags:"
    echo "    vm-name         = name of the virtual machine"
    echo "    before-change   = snapshot name (used for revert)"
    echo "    \"...\"           = human-readable description"
    echo ""
    echo "  What gets captured:"
    echo "    - Disk state (all data on virtual disks)"
    echo "    - Memory state (if VM is running, RAM contents are saved too)"
    echo "    - Device configuration"
    echo ""

    echo "--- Step 2: Make changes inside the VM ---"
    echo "  virsh console vm-name"
    echo "  # (inside guest) touch /tmp/test-file"
    echo "  # (inside guest) echo 'test data' > /tmp/test-file"
    echo "  # (inside guest) apt install some-package  # Or any change"
    echo ""

    echo "--- Step 3: Revert to the snapshot ---"
    echo "  virsh snapshot-revert vm-name before-change"
    echo ""
    echo "  What happens:"
    echo "    - Disk is restored to exact state when snapshot was taken"
    echo "    - If running snapshot: VM resumes from saved memory state"
    echo "    - If offline snapshot: VM is powered off after revert"
    echo "    - ALL changes since snapshot are discarded"
    echo ""

    echo "--- Step 4: Verify that changes were undone ---"
    echo "  virsh console vm-name"
    echo "  # (inside guest) ls /tmp/test-file"
    echo "  # Expected: 'No such file or directory' (change was reverted)"
    echo ""

    echo "Snapshot management commands:"
    echo "  virsh snapshot-list vm-name                     # List all snapshots"
    echo "  virsh snapshot-info vm-name before-change       # Snapshot details"
    echo "  virsh snapshot-delete vm-name before-change     # Delete a snapshot"
    echo "  virsh snapshot-current vm-name                  # Show current snapshot"
    echo "  virsh snapshot-parent vm-name --current         # Show parent snapshot"
    echo ""

    echo "Snapshot best practices:"
    echo "  - Take snapshots before risky operations (upgrades, config changes)"
    echo "  - Don't keep too many snapshots (performance degrades with long chains)"
    echo "  - Delete snapshots when no longer needed (reclaims disk space)"
    echo "  - For long-term backups, use proper backup tools (Borg, rsync) instead"
    echo "  - Internal snapshots (qcow2) vs external: internal is simpler,"
    echo "    external allows live backup of the base image"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
