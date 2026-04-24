#!/usr/bin/env bash
# =============================================================================
# 24_cloud_integration.sh — Cloud-Init, Metadata, and SSH Bootstrapping
#
# PURPOSE: Tour of the Linux facilities that make a virtual machine come up
#          "configured" on a cloud. Inspects cloud-init state, shows example
#          user-data / meta-data structure, and explains the bootstrapping
#          flow. Read-only: no cloud resources are touched.
#
# USAGE:
#   ./24_cloud_integration.sh [--concepts|--cloudinit|--userdata|--metadata|--all]
# =============================================================================

set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. The bootstrapping flow
# ---------------------------------------------------------------------------
demo_concepts() {
    section "1. How a Cloud VM Bootstraps"

    cat <<'EOF'
  Step 1  — The cloud provider creates the VM from a base AMI / image.
  Step 2  — The VM boots; cloud-init starts as a systemd service.
  Step 3  — cloud-init queries the provider's metadata service:
            * AWS:    http://169.254.169.254/latest/meta-data/
            * GCP:    http://metadata.google.internal/computeMetadata/v1/
            * Azure:  http://169.254.169.254/metadata/instance
  Step 4  — It fetches the user-data (a script or YAML you provided on launch).
  Step 5  — It applies the user-data: installs packages, writes files, adds
            SSH keys, runs commands.
  Step 6  — Subsequent boots are "per-boot" mode: user-data is NOT re-applied,
            but some modules (e.g., timezone) may run again.

  The result: an image that is generic at rest but specialized at first boot
  according to whatever the infrastructure-as-code layer requested.
EOF
}

# ---------------------------------------------------------------------------
# 2. cloud-init inspection
# ---------------------------------------------------------------------------
demo_cloudinit() {
    section "2. cloud-init Status (if present)"

    if ! have cloud-init; then
        explain "cloud-init not installed — this machine is not a cloud-init-managed VM."
        explain "On such a VM you would see it here after 'apt install cloud-init'."
        return 0
    fi

    show "cloud-init status --long"
    cloud-init status --long 2>/dev/null || echo "  (may need sudo)"

    explain ""
    explain "cloud-init logs live in /var/log/cloud-init.log and cloud-init-output.log."
    explain "The analyze subcommand reports module runtimes (useful when boot is slow):"
    show "cloud-init analyze show 2>/dev/null | head -10"
    cloud-init analyze show 2>/dev/null | head -10 || true
}

# ---------------------------------------------------------------------------
# 3. Example user-data
# ---------------------------------------------------------------------------
demo_userdata() {
    section "3. Example user-data"

    cat > "$WORKDIR/user-data.yaml" <<'EOF'
#cloud-config
# cloud-init expects this literal header line. The rest is YAML.

hostname: web01
timezone: UTC

# Add a sudo-capable user with pubkey auth (no password)
users:
  - name: deploy
    groups: sudo
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ssh-ed25519 AAAA... deploy@workstation

# Install packages at first boot
package_update: true
package_upgrade: true
packages:
  - nginx
  - ufw
  - fail2ban

# Write configuration files
write_files:
  - path: /etc/nginx/sites-available/default
    permissions: '0644'
    content: |
      server { listen 80 default_server; root /var/www/html; }

# Commands to run once, at the end of first boot
runcmd:
  - ufw allow 80/tcp
  - ufw --force enable
  - systemctl restart nginx
EOF

    explain "A typical user-data YAML (stored in $WORKDIR/user-data.yaml):"
    show "cat user-data.yaml"
    cat "$WORKDIR/user-data.yaml"

    explain ""
    explain "Key sections:"
    explain "  users       — accounts + SSH keys (the #1 use of user-data)"
    explain "  packages    — installed at first boot"
    explain "  write_files — creates config files with explicit permissions"
    explain "  runcmd      — one-shot commands, run after packages are installed"
}

# ---------------------------------------------------------------------------
# 4. Metadata queries (safe — only attempted on an actual cloud VM)
# ---------------------------------------------------------------------------
demo_metadata() {
    section "4. Metadata Service Queries"

    explain "On an AWS VM:"
    show "  IMDSv2 (recommended):"
    show "    TOKEN=\$(curl -sS -X PUT 'http://169.254.169.254/latest/api/token' \\"
    show "              -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')"
    show "    curl -sS -H \"X-aws-ec2-metadata-token: \$TOKEN\" \\"
    show "         http://169.254.169.254/latest/meta-data/instance-id"

    explain ""
    explain "On a GCP VM:"
    show "  curl -sS -H 'Metadata-Flavor: Google' \\"
    show "       http://metadata.google.internal/computeMetadata/v1/instance/name"

    explain ""
    explain "On an Azure VM:"
    show "  curl -sS -H 'Metadata:true' \\"
    show "       'http://169.254.169.254/metadata/instance?api-version=2023-07-01'"

    explain ""
    # Detect, but do not attempt to connect, since we may be on any machine.
    # A short connect timeout would still generate spurious syslog noise.
    if [[ -r /sys/class/dmi/id/sys_vendor ]]; then
        show "cat /sys/class/dmi/id/sys_vendor"
        cat /sys/class/dmi/id/sys_vendor 2>/dev/null || true
        explain "(Look for 'Amazon EC2', 'Google', 'Microsoft Corporation' to confirm VM origin.)"
    fi
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"

    explain "Read-only. No metadata endpoints are contacted; cloud resources unchanged."

    case "$mode" in
        --concepts)  demo_concepts ;;
        --cloudinit) demo_cloudinit ;;
        --userdata)  demo_userdata ;;
        --metadata)  demo_metadata ;;
        --all|*)
            demo_concepts
            demo_cloudinit
            demo_userdata
            demo_metadata
            ;;
    esac
}

main "$@"
