"""
Infrastructure as Code (IaC) Simulation

Demonstrates IaC concepts used in Terraform, CloudFormation, and Pulumi:
- Resource dependency graph with topological sort for creation order
- Plan/Apply lifecycle (preview changes before executing)
- State management (tracking what exists vs what is desired)
- Drift detection (when reality diverges from declared state)
- Resource lifecycle: create, update, replace, delete

No cloud account required -- all behavior is simulated locally.
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class ChangeAction(Enum):
    CREATE = "create"     # Resource does not exist, will be created
    UPDATE = "update"     # Resource exists but properties differ
    REPLACE = "replace"   # Resource must be destroyed and recreated (e.g., changing AMI)
    DELETE = "delete"     # Resource exists in state but not in config (removed)
    NO_CHANGE = "no-op"   # Resource matches desired state


@dataclass
class ResourceConfig:
    """Declared desired state for a single resource.
    This is what you write in your .tf or CloudFormation template."""
    resource_type: str   # e.g., "aws_instance", "aws_vpc", "aws_subnet"
    name: str            # Logical name in the IaC config
    properties: Dict     # Desired properties (e.g., {"instance_type": "t3.medium"})
    depends_on: List[str] = field(default_factory=list)  # Explicit dependencies

    @property
    def resource_id(self) -> str:
        return f"{self.resource_type}.{self.name}"


@dataclass
class ResourceState:
    """Actual state of a provisioned resource, stored in the state file.
    The state file is the single source of truth for what IaC has created.
    Losing the state file means IaC loses track of all managed resources."""
    resource_id: str
    resource_type: str
    name: str
    physical_id: str     # The real cloud ID (e.g., "i-0abc123", "vpc-xyz789")
    properties: Dict
    status: str = "active"


class DependencyGraph:
    """Builds and resolves resource dependencies using topological sort.
    Resources must be created in dependency order: VPC before subnet,
    subnet before instance. Deletion happens in reverse order."""

    def __init__(self):
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.nodes: Set[str] = set()

    def add_resource(self, resource_id: str, depends_on: List[str]) -> None:
        self.nodes.add(resource_id)
        for dep in depends_on:
            self.adjacency[dep].add(resource_id)
            self.nodes.add(dep)

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological ordering.
        This determines the safe creation order: a resource is only created
        after ALL its dependencies are ready."""
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for neighbor in self.adjacency[node]:
                in_degree[neighbor] += 1

        queue = deque(n for n in self.nodes if in_degree[n] == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Circular dependency detected!")
        return order


class IaCEngine:
    """Simulates an IaC engine (like Terraform) with plan/apply/destroy lifecycle."""

    def __init__(self):
        # State file: maps resource_id -> ResourceState
        self.state: Dict[str, ResourceState] = {}
        self._next_physical_id = 1

    def _generate_physical_id(self, resource_type: str) -> str:
        """Generate a fake cloud resource ID."""
        prefixes = {
            "aws_vpc": "vpc",
            "aws_subnet": "subnet",
            "aws_instance": "i",
            "aws_security_group": "sg",
            "aws_s3_bucket": "bucket",
            "aws_rds_instance": "db",
            "aws_lambda_function": "lambda",
        }
        prefix = prefixes.get(resource_type, "res")
        pid = f"{prefix}-{self._next_physical_id:08x}"
        self._next_physical_id += 1
        return pid

    def plan(self, desired_config: List[ResourceConfig]) -> List[dict]:
        """Generate an execution plan by diffing desired state vs actual state.
        This is the 'terraform plan' equivalent -- it shows what WILL happen
        without actually making any changes. Always review plans before applying!"""
        changes = []
        desired_ids = {r.resource_id for r in desired_config}

        # Check each desired resource against current state
        for resource in desired_config:
            existing = self.state.get(resource.resource_id)

            if existing is None:
                changes.append({
                    "action": ChangeAction.CREATE,
                    "resource_id": resource.resource_id,
                    "reason": "Resource not in state (new)",
                })
            elif existing.properties != resource.properties:
                # Some property changes require replacement (e.g., changing VPC CIDR)
                # while others can be updated in-place (e.g., changing instance tags)
                immutable_props = {"cidr_block", "ami", "availability_zone"}
                changed_keys = {
                    k for k in resource.properties
                    if resource.properties.get(k) != existing.properties.get(k)
                }
                needs_replace = bool(changed_keys & immutable_props)

                changes.append({
                    "action": ChangeAction.REPLACE if needs_replace else ChangeAction.UPDATE,
                    "resource_id": resource.resource_id,
                    "changed_properties": list(changed_keys),
                    "reason": "Immutable property changed" if needs_replace else "Properties differ",
                })
            else:
                changes.append({
                    "action": ChangeAction.NO_CHANGE,
                    "resource_id": resource.resource_id,
                })

        # Check for resources in state but not in desired config (to be deleted)
        for resource_id in self.state:
            if resource_id not in desired_ids:
                changes.append({
                    "action": ChangeAction.DELETE,
                    "resource_id": resource_id,
                    "reason": "Removed from configuration",
                })

        return changes

    def apply(self, desired_config: List[ResourceConfig]) -> List[str]:
        """Apply the desired configuration, creating/updating/deleting resources.
        This is the 'terraform apply' equivalent."""
        plan = self.plan(desired_config)
        log = []

        # Build dependency graph for creation order
        graph = DependencyGraph()
        for resource in desired_config:
            graph.add_resource(resource.resource_id, resource.depends_on)
        creation_order = graph.topological_sort()

        # Process in dependency order
        action_map = {c["resource_id"]: c for c in plan}

        for resource_id in creation_order:
            change = action_map.get(resource_id)
            if not change:
                continue

            resource = next((r for r in desired_config if r.resource_id == resource_id), None)
            if not resource:
                continue

            if change["action"] == ChangeAction.CREATE:
                physical_id = self._generate_physical_id(resource.resource_type)
                self.state[resource_id] = ResourceState(
                    resource_id=resource_id,
                    resource_type=resource.resource_type,
                    name=resource.name,
                    physical_id=physical_id,
                    properties=dict(resource.properties),
                )
                log.append(f"  + {resource_id} (created: {physical_id})")

            elif change["action"] == ChangeAction.UPDATE:
                self.state[resource_id].properties = dict(resource.properties)
                log.append(f"  ~ {resource_id} (updated in-place)")

            elif change["action"] == ChangeAction.REPLACE:
                old_id = self.state[resource_id].physical_id
                new_physical = self._generate_physical_id(resource.resource_type)
                self.state[resource_id] = ResourceState(
                    resource_id=resource_id,
                    resource_type=resource.resource_type,
                    name=resource.name,
                    physical_id=new_physical,
                    properties=dict(resource.properties),
                )
                log.append(f"  -/+ {resource_id} (replaced: {old_id} -> {new_physical})")

        # Handle deletions (reverse dependency order)
        for change in plan:
            if change["action"] == ChangeAction.DELETE:
                old = self.state.pop(change["resource_id"])
                log.append(f"  - {change['resource_id']} (destroyed: {old.physical_id})")

        return log

    def detect_drift(self) -> List[dict]:
        """Simulate drift detection: check if actual cloud state matches IaC state.
        Drift happens when someone makes manual changes via the console/CLI,
        bypassing IaC. This is a common source of production incidents."""
        drifts = []
        for resource_id, state in self.state.items():
            # Simulate: 20% chance each resource has drifted
            if random.random() < 0.2:
                drifted_key = random.choice(list(state.properties.keys()))
                original = state.properties[drifted_key]
                drifts.append({
                    "resource_id": resource_id,
                    "physical_id": state.physical_id,
                    "property": drifted_key,
                    "expected": original,
                    "actual": f"{original}_MODIFIED",
                    "severity": "HIGH" if drifted_key in ("cidr_block", "ami") else "MEDIUM",
                })
        return drifts


def build_sample_config() -> List[ResourceConfig]:
    """Build a typical 3-tier infrastructure configuration."""
    return [
        ResourceConfig("aws_vpc", "main", {
            "cidr_block": "10.0.0.0/16",
            "enable_dns": True,
        }),
        ResourceConfig("aws_subnet", "public", {
            "cidr_block": "10.0.1.0/24",
            "map_public_ip": True,
        }, depends_on=["aws_vpc.main"]),
        ResourceConfig("aws_subnet", "private", {
            "cidr_block": "10.0.2.0/24",
            "map_public_ip": False,
        }, depends_on=["aws_vpc.main"]),
        ResourceConfig("aws_security_group", "web_sg", {
            "ingress_ports": [80, 443],
            "egress": "all",
        }, depends_on=["aws_vpc.main"]),
        ResourceConfig("aws_instance", "web_server", {
            "ami": "ami-12345678",
            "instance_type": "t3.medium",
            "tags": {"Name": "web-server"},
        }, depends_on=["aws_subnet.public", "aws_security_group.web_sg"]),
        ResourceConfig("aws_rds_instance", "database", {
            "engine": "postgres",
            "instance_class": "db.t3.medium",
            "allocated_storage": 100,
        }, depends_on=["aws_subnet.private"]),
    ]


def demo_plan_and_apply():
    """Demonstrate the full plan/apply lifecycle."""
    print("=" * 70)
    print("IaC Plan/Apply Lifecycle")
    print("=" * 70)

    engine = IaCEngine()
    config = build_sample_config()

    # Initial plan
    print("\n  --- terraform plan (initial) ---")
    plan = engine.plan(config)
    for change in plan:
        symbol = {
            ChangeAction.CREATE: "+", ChangeAction.UPDATE: "~",
            ChangeAction.REPLACE: "-/+", ChangeAction.DELETE: "-",
            ChangeAction.NO_CHANGE: " ",
        }[change["action"]]
        print(f"  {symbol} {change['resource_id']}: {change['action'].value}")

    # Apply
    print("\n  --- terraform apply ---")
    log = engine.apply(config)
    for entry in log:
        print(entry)

    print(f"\n  State: {len(engine.state)} resources managed")

    # Modify config and re-apply (update scenario)
    print("\n  --- terraform plan (modify instance type) ---")
    config[4].properties["instance_type"] = "t3.large"  # In-place update
    plan = engine.plan(config)
    for change in plan:
        if change["action"] != ChangeAction.NO_CHANGE:
            print(f"  ~ {change['resource_id']}: {change['action'].value} "
                  f"({change.get('reason', '')})")

    print("\n  --- terraform apply (update) ---")
    log = engine.apply(config)
    for entry in log:
        print(entry)
    print()


def demo_dependency_resolution():
    """Show how dependency graph determines creation order."""
    print("=" * 70)
    print("Dependency Graph Resolution")
    print("=" * 70)

    config = build_sample_config()
    graph = DependencyGraph()
    for r in config:
        graph.add_resource(r.resource_id, r.depends_on)

    order = graph.topological_sort()
    print("\n  Creation order (topological sort):")
    for i, resource_id in enumerate(order, 1):
        deps = next((r.depends_on for r in config if r.resource_id == resource_id), [])
        dep_str = f" (after: {', '.join(deps)})" if deps else " (no dependencies)"
        print(f"    {i}. {resource_id}{dep_str}")

    print(f"\n  Deletion order (reverse):")
    for i, resource_id in enumerate(reversed(order), 1):
        print(f"    {i}. {resource_id}")
    print()


def demo_drift_detection():
    """Detect configuration drift."""
    print("=" * 70)
    print("Drift Detection")
    print("=" * 70)

    engine = IaCEngine()
    config = build_sample_config()
    engine.apply(config)

    drifts = engine.detect_drift()
    if drifts:
        print(f"\n  Detected {len(drifts)} drifted resource(s):")
        for d in drifts:
            print(f"    [{d['severity']}] {d['resource_id']} ({d['physical_id']})")
            print(f"           {d['property']}: expected={d['expected']}, "
                  f"actual={d['actual']}")
    else:
        print("\n  No drift detected -- infrastructure matches declared state.")

    print(f"\n  Recommendation: Run 'terraform plan' regularly to catch drift early.")
    print(f"  Consider enabling AWS Config Rules for continuous drift monitoring.")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_plan_and_apply()
    demo_dependency_resolution()
    demo_drift_detection()
