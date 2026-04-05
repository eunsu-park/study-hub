#!/bin/bash
# Exercise: Lesson 01 — Kubernetes Architecture
# Complete the TODO items below.

# === Exercise 1: Explore the Control Plane ===
# Verify that core control-plane components are running.
# Hint: Control-plane Pods live in the kube-system namespace.

exercise_1() {
    echo "=== Exercise 1: Explore the Control Plane ==="

    # TODO: List all Pods in the kube-system namespace
    # Expected components: kube-apiserver, etcd, kube-scheduler, kube-controller-manager

    # TODO: Describe the kube-apiserver Pod and note:
    #   - Which port it listens on
    #   - What flags it was started with

    # TODO: Check the health of the API server using kubectl
    # Hint: Use 'kubectl get --raw /healthz'

}

# === Exercise 2: Inspect etcd ===
# Understand etcd as the cluster's backing store.
# Hint: etcdctl requires certs; check the etcd Pod spec for paths.

exercise_2() {
    echo "=== Exercise 2: Inspect etcd ==="

    # TODO: Find the etcd Pod and display its command-line arguments
    # Hint: kubectl describe pod etcd-<node> -n kube-system

    # TODO: Exec into the etcd Pod and list keys under /registry/
    # Hint: etcdctl --endpoints=https://127.0.0.1:2379 \
    #       --cacert=<ca> --cert=<cert> --key=<key> get /registry --prefix --keys-only | head -20

    # TODO: Count how many keys exist for deployments
    # Hint: filter keys containing '/deployments/'

}

# === Exercise 3: Scheduler Behavior ===
# Observe how the scheduler assigns Pods to nodes.
# Hint: Create a Pod and watch the Events section in 'describe'.

exercise_3() {
    echo "=== Exercise 3: Scheduler Behavior ==="

    # TODO: Create a simple nginx Pod
    # Command: kubectl run scheduler-test --image=nginx:alpine

    # TODO: Describe the Pod and find the "Scheduled" event
    # Note which node the scheduler picked and why

    # TODO: Create a Pod with a nodeSelector that does NOT match any node
    # Hint: Use --overrides='{"spec":{"nodeSelector":{"disktype":"ssd"}}}'
    # Observe the Pod stays in Pending state — explain why

    # TODO: Clean up all test Pods

}

# === Exercise 4: kubelet and Node Status ===
# Explore node-level components.
# Hint: kubelet runs as a systemd service on each node.

exercise_4() {
    echo "=== Exercise 4: kubelet and Node Status ==="

    # TODO: List all nodes and their status
    # Command hint: kubectl get nodes -o wide

    # TODO: Describe a node and identify:
    #   - Allocatable CPU and memory
    #   - Number of running Pods
    #   - Conditions (Ready, MemoryPressure, DiskPressure)

    # TODO: Check kubelet logs on a node (minikube example)
    # Hint: minikube ssh -- journalctl -u kubelet --no-pager | tail -30

}

# === Exercise 5: kube-proxy and Networking ===
# Understand how kube-proxy manages Service routing rules.
# Hint: kube-proxy runs as a DaemonSet in kube-system.

exercise_5() {
    echo "=== Exercise 5: kube-proxy and Networking ==="

    # TODO: Find the kube-proxy DaemonSet and check its mode (iptables vs IPVS)
    # Hint: kubectl describe ds kube-proxy -n kube-system

    # TODO: Create a Deployment and a ClusterIP Service, then inspect
    #       the iptables rules kube-proxy creates
    # Hint: minikube ssh -- sudo iptables -t nat -L | grep <service-name>

    # TODO: Explain the difference between iptables and IPVS proxy modes
    # Write your answer as a comment below:
    #
    # iptables mode:
    # IPVS mode:

    # TODO: Clean up the test Deployment and Service

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
