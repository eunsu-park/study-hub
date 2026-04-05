#!/usr/bin/env bash
# 17_production_ops — etcd backup, cluster upgrade, drain commands
# WARNING: Run these commands carefully in production environments

set -euo pipefail

# ============================================================
# Part 1: etcd Backup and Restore
# ============================================================
echo "=== etcd Backup ==="

BACKUP_DIR="/tmp/etcd-backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Snapshot the etcd database
ETCDCTL_API=3 etcdctl snapshot save "$BACKUP_DIR/snapshot_${TIMESTAMP}.db" \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

echo "Backup saved to $BACKUP_DIR/snapshot_${TIMESTAMP}.db"

# Verify the snapshot
ETCDCTL_API=3 etcdctl snapshot status "$BACKUP_DIR/snapshot_${TIMESTAMP}.db" \
  --write-out=table

# Restore (example, do NOT run unless needed)
# ETCDCTL_API=3 etcdctl snapshot restore "$BACKUP_DIR/snapshot_${TIMESTAMP}.db" \
#   --data-dir=/var/lib/etcd-restored \
#   --initial-cluster=master=https://127.0.0.1:2380

# ============================================================
# Part 2: Node Drain and Maintenance
# ============================================================
echo ""
echo "=== Node Maintenance Workflow ==="

NODE_NAME="${1:-minikube}"

echo "Step 1: Cordon node (prevent new scheduling)"
kubectl cordon "$NODE_NAME"

echo "Step 2: Check pods on the node"
kubectl get pods --all-namespaces --field-selector "spec.nodeName=$NODE_NAME" \
  -o wide --no-headers | head -20

echo "Step 3: Drain node (evict pods gracefully)"
kubectl drain "$NODE_NAME" \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=60 \
  --timeout=120s \
  --force 2>&1 || echo "Drain completed with warnings"

echo "Step 4: Verify node is drained"
kubectl get pods --all-namespaces --field-selector "spec.nodeName=$NODE_NAME" --no-headers

echo "Step 5: Uncordon node (allow scheduling again)"
kubectl uncordon "$NODE_NAME"
kubectl get node "$NODE_NAME"

# ============================================================
# Part 3: Cluster Upgrade (kubeadm)
# ============================================================
echo ""
echo "=== Cluster Upgrade Plan ==="

# Check current version
echo "Current cluster version:"
kubectl version --short 2>/dev/null || kubectl version

# Check available upgrade (control plane)
echo ""
echo "Available upgrades:"
sudo kubeadm upgrade plan 2>/dev/null || echo "(Run on control plane node)"

# Upgrade steps (commented out for safety):
# 1. Upgrade kubeadm: apt-get install -y kubeadm=1.29.0-00
# 2. Verify: kubeadm version
# 3. Plan: sudo kubeadm upgrade plan
# 4. Apply: sudo kubeadm upgrade apply v1.29.0
# 5. Upgrade kubelet: apt-get install -y kubelet=1.29.0-00 kubectl=1.29.0-00
# 6. Restart: sudo systemctl daemon-reload && sudo systemctl restart kubelet

# ============================================================
# Part 4: Resource Auditing
# ============================================================
echo ""
echo "=== Resource Audit ==="

echo "Pods without resource limits:"
kubectl get pods --all-namespaces -o json | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for pod in data['items']:
    ns = pod['metadata']['namespace']
    name = pod['metadata']['name']
    for c in pod['spec'].get('containers', []):
        limits = c.get('resources', {}).get('limits', {})
        if not limits.get('cpu') or not limits.get('memory'):
            print(f'  {ns}/{name} container={c[\"name\"]}')
" 2>/dev/null || echo "  (python3 required for audit)"

echo ""
echo "Top resource-consuming pods:"
kubectl top pods --all-namespaces --sort-by=cpu 2>/dev/null | head -10 || \
  echo "  (metrics-server required)"

echo ""
echo "Persistent volumes status:"
kubectl get pv --no-headers 2>/dev/null | \
  awk '{printf "  %-30s %-10s %-10s %s\n", $1, $2, $5, $6}'

echo ""
echo "=== Production Ops Complete ==="
