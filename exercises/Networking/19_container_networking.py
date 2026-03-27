"""
Exercises for Lesson 19: Container Networking
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: Docker Custom Network
    Create custom Docker network, run 3 containers (web, api, db),
    and test connectivity.

    Reasoning: Custom Docker bridge networks provide DNS-based service
    discovery, unlike the default bridge network where containers
    must use IP addresses to communicate.
    """
    print("Docker Custom Network Setup:")
    print()

    # Simulate Docker network configuration
    network_config = {
        "name": "mynetwork",
        "driver": "bridge",
        "subnet": "172.20.0.0/16",
        "gateway": "172.20.0.1",
    }

    containers = [
        {"name": "db", "image": "postgres", "ip": "172.20.0.2"},
        {"name": "api", "image": "my-api", "ip": "172.20.0.3"},
        {"name": "web", "image": "nginx", "ip": "172.20.0.4"},
    ]

    print("  Commands:")
    print(f"  # Create custom network")
    print(f"  docker network create \\")
    print(f"    --driver {network_config['driver']} \\")
    print(f"    --subnet {network_config['subnet']} \\")
    print(f"    --gateway {network_config['gateway']} \\")
    print(f"    {network_config['name']}")
    print()

    print("  # Run containers on the network")
    for c in containers:
        print(f"  docker run -d --name {c['name']} --network {network_config['name']} {c['image']}")

    print("\n  # Test connectivity (DNS-based)")
    print("  docker exec web ping api        # Uses Docker DNS")
    print("  docker exec api ping db          # Container names as hostnames")
    print("  docker exec web curl http://api:8080/health")

    print("\n  Key points:")
    print("    - Custom bridge networks provide automatic DNS resolution")
    print("    - Containers can reach each other by name (not just IP)")
    print("    - Default bridge network does NOT provide DNS (use --link or custom)")
    print("    - Network isolation: containers on different networks cannot communicate")


def exercise_2():
    """
    Problem 2: Kubernetes Service
    Deploy 3 nginx pods with ClusterIP service, verify load balancing.

    Reasoning: Kubernetes Services provide stable endpoints (ClusterIP)
    that load-balance across pods, abstracting away pod IP changes.
    """
    print("Kubernetes Deployment + ClusterIP Service:")
    print()

    deployment_yaml = """  # deployment.yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: web
  spec:
    replicas: 3
    selector:
      matchLabels:
        app: web
    template:
      metadata:
        labels:
          app: web
      spec:
        containers:
          - name: nginx
            image: nginx
            ports:
              - containerPort: 80"""

    service_yaml = """  # service.yaml
  apiVersion: v1
  kind: Service
  metadata:
    name: web-service
  spec:
    selector:
      app: web
    ports:
      - protocol: TCP
        port: 80
        targetPort: 80"""

    print(deployment_yaml)
    print()
    print(service_yaml)

    print("\n  Verification commands:")
    print("    kubectl apply -f deployment.yaml")
    print("    kubectl get pods -l app=web          # Verify 3 pods running")
    print("    kubectl get endpoints web-service     # See pod IPs as endpoints")

    print("\n  Load balancing test:")
    print("    kubectl run test --rm -it --image=busybox -- sh")
    print("    # In test pod:")
    print("    for i in $(seq 1 10); do wget -qO- http://web-service; done")
    print("    # Should see responses from different pods")

    print("\n  Service types:")
    svc_types = {
        "ClusterIP": "Internal only (default), accessible within cluster",
        "NodePort": "Exposes on each node's IP at a static port (30000-32767)",
        "LoadBalancer": "External load balancer (cloud provider)",
        "ExternalName": "Maps to DNS name (CNAME record)",
    }
    for stype, desc in svc_types.items():
        print(f"    {stype:15s}: {desc}")


def exercise_3():
    """
    Problem 3: NetworkPolicy
    Allow: frontend -> backend:8080, backend -> database:5432
    Deny: all other traffic.

    Reasoning: NetworkPolicies implement microsegmentation in Kubernetes,
    following the principle of least privilege for network access.
    """
    print("Kubernetes NetworkPolicy Implementation:")
    print()

    print("  Architecture:")
    print("    [frontend] --port 8080--> [backend] --port 5432--> [database]")
    print("    All other traffic is denied.")

    print("\n  backend-policy.yaml:")
    print("    - Ingress: Allow from frontend on port 8080")
    print("    - Egress: Allow to database on port 5432 + DNS (port 53)")

    print("\n  database-policy.yaml:")
    print("    - Ingress: Allow from backend on port 5432 only")
    print("    - No egress rules needed (database doesn't initiate)")

    print("\n  Key NetworkPolicy concepts:")
    concepts = [
        ("Default deny", "If a pod is selected by any policy, all non-matching traffic is denied"),
        ("Additive", "Multiple policies are unioned (OR logic)"),
        ("Namespace scoped", "Policies apply within their namespace"),
        ("DNS egress", "Always allow UDP port 53 for DNS resolution"),
        ("CNI requirement", "NetworkPolicy requires a CNI that supports it (Calico, Cilium)"),
    ]
    for concept, detail in concepts:
        print(f"    {concept:20s}: {detail}")

    print("\n  Common mistake: Forgetting DNS egress rule.")
    print("  Without it, pods cannot resolve service names, breaking all communication.")


def exercise_4():
    """
    Problem 4: Debugging Network Issue
    Scenario: Pod can't reach external service (example.com)

    Reasoning: Container networking issues follow the same bottom-up
    troubleshooting approach as traditional networking, with additional
    Kubernetes-specific checks (CNI, NetworkPolicy, kube-proxy).
    """
    print("Troubleshooting: Pod Cannot Reach External Service")
    print()

    steps = [
        ("1. Check pod network config", [
            "kubectl exec <pod> -- ip addr show    # Verify pod has IP",
            "kubectl exec <pod> -- ip route show    # Check default route",
        ]),
        ("2. Test DNS resolution", [
            "kubectl exec <pod> -- nslookup example.com",
            "If fails: kubectl get pods -n kube-system -l k8s-app=kube-dns",
            "Check CoreDNS logs: kubectl logs -n kube-system -l k8s-app=kube-dns",
        ]),
        ("3. Test external connectivity", [
            "kubectl exec <pod> -- ping 8.8.8.8     # Bypass DNS",
            "If ping works but DNS fails -> DNS issue",
            "If ping fails -> network/policy issue",
        ]),
        ("4. Check NetworkPolicy", [
            "kubectl get networkpolicy -n <namespace>",
            "kubectl describe networkpolicy <name>",
            "Ensure egress to 0.0.0.0/0 is allowed (or no restrictive policy)",
        ]),
        ("5. Check NAT/masquerade on node", [
            "sudo iptables -t nat -L POSTROUTING -n -v",
            "Should see MASQUERADE rule for pod CIDR",
            "Pod traffic must be NAT'd to node IP for external access",
        ]),
        ("6. Verify CNI configuration", [
            "cat /etc/cni/net.d/*.conf",
            "Check CNI plugin logs for errors",
            "Verify CNI supports external egress",
        ]),
        ("7. Check node routing", [
            "ip route show                           # On the node",
            "Verify routes exist for pod CIDR",
            "Check if node can reach external networks",
        ]),
    ]

    for phase, commands in steps:
        print(f"  {phase}:")
        for cmd in commands:
            print(f"    {cmd}")
        print()


def exercise_5():
    """
    Problem 5: Service Mesh Traffic Splitting
    Implement canary deployment: 90% v1, 10% v2 using Istio.

    Reasoning: Service meshes like Istio provide L7 traffic management
    without application code changes, enabling canary deployments,
    A/B testing, and gradual rollouts.
    """
    print("Istio Canary Deployment (90/10 Traffic Split):")
    print()

    print("  VirtualService configuration:")
    print("    Routes traffic 90% to v1, 10% to v2")
    vs_config = {
        "v1": {"subset": "v1", "weight": 90},
        "v2": {"subset": "v2", "weight": 10},
    }
    for version, config in vs_config.items():
        print(f"    {version}: subset={config['subset']}, weight={config['weight']}%")

    print("\n  DestinationRule configuration:")
    print("    Defines subsets based on pod labels")
    print("    v1 subset: label version=v1")
    print("    v2 subset: label version=v2")

    print("\n  Verification:")
    print("    # Generate traffic and observe distribution")
    print("    for i in $(seq 1 100); do")
    print("      curl -s http://my-app/version")
    print("    done | sort | uniq -c")
    print("    # Expected: ~90 v1, ~10 v2")

    print("\n  Canary deployment workflow:")
    workflow = [
        "1. Deploy v2 alongside v1 (separate deployment)",
        "2. Start with 0% traffic to v2 (test internally)",
        "3. Shift 10% traffic to v2 (monitor error rates)",
        "4. If healthy: gradually increase (25%, 50%, 75%, 100%)",
        "5. If errors: immediately shift 100% back to v1",
        "6. Remove v1 deployment once v2 is stable at 100%",
    ]
    for step in workflow:
        print(f"    {step}")

    print("\n  Service mesh benefits for deployments:")
    benefits = [
        "Traffic splitting without application changes",
        "Automatic retries and timeouts",
        "mTLS between services (zero-trust)",
        "Distributed tracing (Jaeger/Zipkin)",
        "Traffic mirroring for shadow testing",
    ]
    for b in benefits:
        print(f"    - {b}")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
