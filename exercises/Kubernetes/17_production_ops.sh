#!/bin/bash
# Exercise: Lesson 17 — Production Operations
# Complete the TODO items below.

# === Exercise 1: etcd Backup and Restore ===
# Protect cluster state with etcd snapshots.
# Hint: etcd stores all cluster data — losing it means losing the cluster.

exercise_1() {
    echo "=== Exercise 1: etcd Backup and Restore ==="

    # TODO: Take an etcd snapshot
    # Hint: ETCDCTL_API=3 etcdctl snapshot save /backup/etcd-snapshot.db \
    #       --endpoints=https://127.0.0.1:2379 \
    #       --cacert=/etc/kubernetes/pki/etcd/ca.crt \
    #       --cert=/etc/kubernetes/pki/etcd/server.crt \
    #       --key=/etc/kubernetes/pki/etcd/server.key

    # TODO: Verify the snapshot integrity
    # Hint: etcdctl snapshot status /backup/etcd-snapshot.db --write-table

    # TODO: Restore from the snapshot to a new data directory
    # Hint: etcdctl snapshot restore /backup/etcd-snapshot.db \
    #       --data-dir=/var/lib/etcd-restored

    # TODO: Create a CronJob manifest to automate daily backups
    # Write a YAML snippet as a comment or heredoc:
    # - Schedule: daily at 1:00 AM
    # - Mount etcd certs from the host
    # - Store snapshots on a PersistentVolume

}

# === Exercise 2: Node Maintenance with Drain ===
# Safely evacuate a node for maintenance.
# Hint: drain = cordon + evict all pods, respecting PDBs.

exercise_2() {
    echo "=== Exercise 2: Node Maintenance ==="

    # TODO: Check node status and identify the target node
    # Command: kubectl get nodes

    # TODO: Cordon the node (mark unschedulable, no pod eviction)
    # Command: kubectl cordon <node-name>

    # TODO: Drain the node (evict all pods gracefully)
    # Command: kubectl drain <node-name> \
    #          --ignore-daemonsets \
    #          --delete-emptydir-data \
    #          --grace-period=60 \
    #          --timeout=120s

    # TODO: Verify no non-DaemonSet pods remain on the node
    # Command: kubectl get pods --all-namespaces --field-selector spec.nodeName=<node-name>

    # TODO: After maintenance, uncordon the node
    # Command: kubectl uncordon <node-name>

}

# === Exercise 3: Resource Quota and LimitRange ===
# Enforce resource consumption boundaries per namespace.
# Hint: Quotas prevent a single team from consuming all cluster resources.

exercise_3() {
    echo "=== Exercise 3: Resource Quotas ==="

    # TODO: Create a ResourceQuota for the "dev" namespace with:
    #   - Max 10 Pods
    #   - Max 4 CPU requests, 8 CPU limits
    #   - Max 8Gi memory requests, 16Gi memory limits
    #   - Max 5 PVCs
    # Hint: kubectl create quota dev-quota -n dev \
    #       --hard=pods=10,requests.cpu=4,limits.cpu=8,...

    # TODO: Create a LimitRange to set default resource limits:
    #   - Default container limits: 500m CPU, 512Mi memory
    #   - Default container requests: 100m CPU, 128Mi memory
    #   - Max container limits: 2 CPU, 2Gi memory
    # Write the LimitRange YAML as a heredoc and apply it

    # TODO: Verify the quota is enforced
    # Command: kubectl describe quota dev-quota -n dev

    # TODO: Try to exceed the quota and observe the error
    # Command: kubectl create deployment test --image=nginx --replicas=20 -n dev

}

# === Exercise 4: Cluster Upgrade (kubeadm) ===
# Upgrade a kubeadm cluster from one version to the next.
# Hint: Always upgrade one minor version at a time (e.g., 1.28 -> 1.29).

exercise_4() {
    echo "=== Exercise 4: Cluster Upgrade ==="

    # TODO: Check the current cluster version
    # Command: kubectl version --short
    # Command: kubeadm version

    # TODO: Plan the upgrade (dry run)
    # Command: sudo kubeadm upgrade plan

    # TODO: Upgrade the control plane (on the control plane node)
    # Step 1: Update kubeadm
    #   sudo apt-get update && sudo apt-get install -y kubeadm=1.29.0-00
    # Step 2: Apply the upgrade
    #   sudo kubeadm upgrade apply v1.29.0
    # Step 3: Upgrade kubelet and kubectl
    #   sudo apt-get install -y kubelet=1.29.0-00 kubectl=1.29.0-00
    #   sudo systemctl daemon-reload && sudo systemctl restart kubelet

    # TODO: Upgrade worker nodes (on each worker)
    # Step 1: Drain the node (from control plane)
    # Step 2: Update kubeadm, run 'kubeadm upgrade node'
    # Step 3: Update kubelet, restart
    # Step 4: Uncordon the node

    # TODO: Verify all nodes show the new version
    # Command: kubectl get nodes

}

# === Exercise 5: Disaster Recovery Runbook ===
# Document and test a disaster recovery procedure.
# Hint: A tested runbook is critical — untested backups are not backups.

exercise_5() {
    echo "=== Exercise 5: Disaster Recovery ==="

    # TODO: Create a pre-flight checklist (write as comments):
    # - [ ] Verify etcd backup exists and is recent (< 24 hours)
    # - [ ] Confirm backup storage is accessible
    # - [ ] Document current cluster version and node count
    # - [ ] List all critical namespaces and their resources

    # TODO: Write a recovery script that:
    # 1. Stops the API server
    # 2. Restores etcd from the latest snapshot
    # 3. Restarts the control plane components
    # 4. Verifies cluster health

    # TODO: Test recovery by:
    # 1. Creating test resources (deployment, service, configmap)
    # 2. Taking an etcd snapshot
    # 3. Deleting the test resources
    # 4. Restoring from snapshot
    # 5. Verifying the resources are back

    # TODO: Document RTO (Recovery Time Objective) and RPO (Recovery Point Objective)
    # RTO target: _____ minutes
    # RPO target: _____ hours (determined by backup frequency)

}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
