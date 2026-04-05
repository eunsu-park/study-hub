#!/usr/bin/env bash
# 01_architecture — kubectl commands to explore cluster architecture
# Requires: minikube running, kubectl configured

set -euo pipefail

echo "=== Cluster Info ==="
kubectl cluster-info

echo ""
echo "=== Node Details ==="
kubectl get nodes -o wide

echo ""
echo "=== Node Resource Capacity ==="
kubectl describe nodes | grep -A 6 "Capacity:"

echo ""
echo "=== Control Plane Components ==="
kubectl get pods -n kube-system -l tier=control-plane -o wide

echo ""
echo "=== All kube-system Pods ==="
kubectl get pods -n kube-system --sort-by=.metadata.name

echo ""
echo "=== API Resources Available ==="
kubectl api-resources --sort-by=name | head -30

echo ""
echo "=== API Versions ==="
kubectl api-versions | sort

echo ""
echo "=== Component Statuses (legacy) ==="
kubectl get componentstatuses 2>/dev/null || echo "ComponentStatus deprecated in newer versions"

echo ""
echo "=== Cluster Events (last 10) ==="
kubectl get events -n kube-system --sort-by='.lastTimestamp' | tail -10

echo ""
echo "=== Kubelet Version on Each Node ==="
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.nodeInfo.kubeletVersion}{"\n"}{end}'

echo ""
echo "=== Inspect kube-apiserver Pod Spec ==="
kubectl get pod -n kube-system -l component=kube-apiserver \
  -o jsonpath='{.items[0].spec.containers[0].command}' 2>/dev/null | tr ',' '\n' || \
  echo "apiserver pod not directly visible (managed provider)"

echo ""
echo "=== Etcd Endpoint (if accessible) ==="
kubectl get pod -n kube-system -l component=etcd \
  -o jsonpath='{.items[0].spec.containers[0].command}' 2>/dev/null | \
  grep -oP 'listen-client-urls=\K[^ ]+' || \
  echo "etcd not directly accessible"

echo ""
echo "=== Namespaces ==="
kubectl get namespaces

echo ""
echo "=== Storage Classes ==="
kubectl get storageclasses

echo ""
echo "=== Cluster Role Count ==="
echo "ClusterRoles: $(kubectl get clusterroles --no-headers | wc -l)"
echo "ClusterRoleBindings: $(kubectl get clusterrolebindings --no-headers | wc -l)"

echo ""
echo "=== Done: Cluster architecture exploration complete ==="
