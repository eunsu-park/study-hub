"""
IAM (Identity and Access Management) Policy Simulation

Demonstrates IAM concepts central to cloud security:
- Policy document structure (Effect, Action, Resource, Condition)
- Policy evaluation logic (explicit Deny beats Allow)
- Role assumption and temporary credentials
- Least-privilege analysis (detecting overly permissive policies)
- Permission boundaries

No cloud account required -- all behavior is simulated locally.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from fnmatch import fnmatch


@dataclass
class PolicyStatement:
    """A single statement in an IAM policy document.
    Every statement has: Effect (Allow/Deny), Action(s), Resource(s),
    and optional Condition(s). This mirrors the real AWS/GCP IAM structure."""
    sid: str                          # Statement ID for identification
    effect: str                       # "Allow" or "Deny"
    actions: List[str]                # e.g., ["s3:GetObject", "s3:PutObject"]
    resources: List[str]             # e.g., ["arn:aws:s3:::my-bucket/*"]
    conditions: Dict[str, dict] = field(default_factory=dict)

    def matches_action(self, requested_action: str) -> bool:
        """Check if the requested action matches any action pattern.
        Actions support wildcards: 's3:*' matches all S3 actions.
        This is how IAM evaluates action-level permissions."""
        for pattern in self.actions:
            if fnmatch(requested_action.lower(), pattern.lower()):
                return True
        return False

    def matches_resource(self, requested_resource: str) -> bool:
        """Check if the requested resource matches any resource pattern.
        Resources use ARN patterns with wildcards for flexible matching."""
        for pattern in self.resources:
            if fnmatch(requested_resource, pattern):
                return True
        return False

    def evaluate_conditions(self, context: Dict[str, str]) -> bool:
        """Evaluate optional conditions (e.g., IP restrictions, MFA required).
        Conditions add fine-grained control beyond just action+resource."""
        if not self.conditions:
            return True
        for condition_type, condition_pairs in self.conditions.items():
            for key, expected in condition_pairs.items():
                actual = context.get(key, "")
                if condition_type == "StringEquals" and actual != expected:
                    return False
                if condition_type == "Bool" and str(actual).lower() != str(expected).lower():
                    return False
                if condition_type == "IpAddress":
                    # Simplified: just check prefix match for demo
                    if not actual.startswith(expected.split("/")[0][:6]):
                        return False
        return True


@dataclass
class IAMPolicy:
    """An IAM policy containing multiple statements."""
    name: str
    description: str
    statements: List[PolicyStatement] = field(default_factory=list)

    def to_json(self) -> str:
        """Generate the policy document in standard AWS JSON format."""
        doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": s.sid,
                    "Effect": s.effect,
                    "Action": s.actions,
                    "Resource": s.resources,
                    **({"Condition": s.conditions} if s.conditions else {}),
                }
                for s in self.statements
            ],
        }
        return json.dumps(doc, indent=2)


@dataclass
class IAMRole:
    """An IAM role that can be assumed by users or services.
    Roles provide temporary credentials -- this is more secure than
    long-lived access keys because credentials auto-expire."""
    name: str
    policies: List[IAMPolicy] = field(default_factory=list)
    trust_policy: Optional[str] = None  # Who can assume this role
    permission_boundary: Optional[IAMPolicy] = None  # Maximum permission ceiling


class IAMEvaluator:
    """Implements the IAM policy evaluation algorithm.

    AWS evaluation logic (simplified):
    1. Start with implicit DENY (everything denied by default)
    2. Evaluate all applicable policies
    3. If ANY statement has explicit Deny -> DENY (Deny always wins)
    4. If ANY statement has Allow -> ALLOW
    5. Otherwise -> implicit DENY (default)

    This 'default deny' model is a security fundamental -- you must
    explicitly grant every permission needed.
    """

    def __init__(self):
        self.evaluation_log: List[dict] = []

    def evaluate(self, policies: List[IAMPolicy], action: str,
                 resource: str, context: Optional[Dict[str, str]] = None,
                 permission_boundary: Optional[IAMPolicy] = None) -> dict:
        """Evaluate whether an action is allowed given the policies."""
        context = context or {}
        explicit_deny = False
        explicit_allow = False
        matched_statements = []

        # Step 1: Check all policy statements
        for policy in policies:
            for stmt in policy.statements:
                if not stmt.matches_action(action):
                    continue
                if not stmt.matches_resource(resource):
                    continue
                if not stmt.evaluate_conditions(context):
                    matched_statements.append({
                        "policy": policy.name, "sid": stmt.sid,
                        "effect": "SKIPPED (condition failed)",
                    })
                    continue

                matched_statements.append({
                    "policy": policy.name, "sid": stmt.sid,
                    "effect": stmt.effect,
                })

                if stmt.effect == "Deny":
                    explicit_deny = True
                elif stmt.effect == "Allow":
                    explicit_allow = True

        # Step 2: Check permission boundary (if set)
        boundary_allows = True
        if permission_boundary and explicit_allow:
            boundary_allows = False
            for stmt in permission_boundary.statements:
                if (stmt.effect == "Allow"
                        and stmt.matches_action(action)
                        and stmt.matches_resource(resource)):
                    boundary_allows = True
                    break

        # Step 3: Apply evaluation logic
        if explicit_deny:
            decision = "DENY (explicit)"
        elif explicit_allow and boundary_allows:
            decision = "ALLOW"
        elif explicit_allow and not boundary_allows:
            decision = "DENY (permission boundary)"
        else:
            decision = "DENY (implicit - no matching Allow)"

        result = {
            "action": action,
            "resource": resource,
            "decision": decision,
            "matched_statements": matched_statements,
        }
        self.evaluation_log.append(result)
        return result


def create_sample_policies() -> Dict[str, IAMPolicy]:
    """Create realistic IAM policies for demonstration."""
    policies = {}

    # Admin policy: full access (dangerous in production!)
    policies["AdminAccess"] = IAMPolicy(
        "AdminAccess", "Full access to all resources",
        [PolicyStatement("FullAccess", "Allow", ["*"], ["*"])],
    )

    # Read-only S3 policy
    policies["S3ReadOnly"] = IAMPolicy(
        "S3ReadOnly", "Read-only access to S3",
        [PolicyStatement("S3Read", "Allow",
                         ["s3:GetObject", "s3:ListBucket"],
                         ["arn:aws:s3:::*"])],
    )

    # Scoped S3 policy with explicit deny
    policies["S3ScopedAccess"] = IAMPolicy(
        "S3ScopedAccess", "Access to specific bucket, deny sensitive paths",
        [
            PolicyStatement("AllowBucket", "Allow",
                            ["s3:*"], ["arn:aws:s3:::my-app-data/*"]),
            # Explicit Deny overrides the Allow above for sensitive paths
            PolicyStatement("DenySensitive", "Deny",
                            ["s3:*"], ["arn:aws:s3:::my-app-data/secrets/*"]),
        ],
    )

    # MFA-required policy
    policies["MFARequired"] = IAMPolicy(
        "MFARequired", "Require MFA for destructive actions",
        [
            PolicyStatement("AllowReadAlways", "Allow",
                            ["ec2:Describe*", "s3:Get*", "s3:List*"], ["*"]),
            PolicyStatement("AllowWriteWithMFA", "Allow",
                            ["ec2:*", "s3:*"], ["*"],
                            conditions={"Bool": {"aws:MultiFactorAuthPresent": "true"}}),
        ],
    )

    return policies


def demo_policy_evaluation():
    """Demonstrate the IAM evaluation algorithm with various scenarios."""
    print("=" * 75)
    print("IAM Policy Evaluation Engine")
    print("=" * 75)

    policies = create_sample_policies()
    evaluator = IAMEvaluator()

    # Test scenarios showing different evaluation outcomes
    scenarios = [
        {
            "name": "Admin: everything allowed",
            "policies": [policies["AdminAccess"]],
            "action": "ec2:TerminateInstances",
            "resource": "arn:aws:ec2:us-east-1:123456:instance/i-abc",
        },
        {
            "name": "S3 Read: allowed action",
            "policies": [policies["S3ReadOnly"]],
            "action": "s3:GetObject",
            "resource": "arn:aws:s3:::my-bucket/file.txt",
        },
        {
            "name": "S3 Read: denied action (write attempt)",
            "policies": [policies["S3ReadOnly"]],
            "action": "s3:PutObject",
            "resource": "arn:aws:s3:::my-bucket/file.txt",
        },
        {
            "name": "Scoped: allowed path",
            "policies": [policies["S3ScopedAccess"]],
            "action": "s3:GetObject",
            "resource": "arn:aws:s3:::my-app-data/reports/q1.pdf",
        },
        {
            "name": "Scoped: denied path (explicit Deny wins over Allow)",
            "policies": [policies["S3ScopedAccess"]],
            "action": "s3:GetObject",
            "resource": "arn:aws:s3:::my-app-data/secrets/api-key.txt",
        },
        {
            "name": "MFA: write without MFA (denied)",
            "policies": [policies["MFARequired"]],
            "action": "s3:PutObject",
            "resource": "arn:aws:s3:::my-bucket/file.txt",
            "context": {"aws:MultiFactorAuthPresent": "false"},
        },
        {
            "name": "MFA: write with MFA (allowed)",
            "policies": [policies["MFARequired"]],
            "action": "s3:PutObject",
            "resource": "arn:aws:s3:::my-bucket/file.txt",
            "context": {"aws:MultiFactorAuthPresent": "true"},
        },
    ]

    for scenario in scenarios:
        result = evaluator.evaluate(
            scenario["policies"],
            scenario["action"],
            scenario["resource"],
            scenario.get("context"),
        )
        print(f"\n  Scenario: {scenario['name']}")
        print(f"    Action:   {result['action']}")
        print(f"    Resource: {result['resource']}")
        print(f"    Decision: {result['decision']}")
        if result["matched_statements"]:
            for ms in result["matched_statements"]:
                print(f"    Matched:  [{ms['policy']}] {ms['sid']} -> {ms['effect']}")
    print()


def demo_least_privilege_analysis():
    """Analyze policies for overly permissive patterns."""
    print("=" * 75)
    print("Least Privilege Analysis")
    print("=" * 75)

    policies = create_sample_policies()

    # Check for common over-permission patterns
    warnings = {
        "AdminAccess": [],
        "S3ReadOnly": [],
        "S3ScopedAccess": [],
        "MFARequired": [],
    }

    for name, policy in policies.items():
        for stmt in policy.statements:
            if stmt.effect != "Allow":
                continue
            # Check for wildcard actions
            if "*" in stmt.actions or any("*" == a for a in stmt.actions):
                warnings[name].append(
                    f"CRITICAL: Wildcard action '*' in {stmt.sid} -- grants ALL permissions")
            elif any(a.endswith(":*") for a in stmt.actions):
                services = [a.split(":")[0] for a in stmt.actions if a.endswith(":*")]
                warnings[name].append(
                    f"WARNING: Service-wide wildcard for {services} in {stmt.sid}")
            # Check for wildcard resources
            if "*" in stmt.resources or any(r == "*" for r in stmt.resources):
                warnings[name].append(
                    f"WARNING: Wildcard resource '*' in {stmt.sid} -- applies to ALL resources")
            # Check for missing conditions on write actions
            write_patterns = ["Put", "Create", "Delete", "Update", "Terminate"]
            has_write = any(
                any(wp.lower() in a.lower() for wp in write_patterns)
                for a in stmt.actions
            )
            if has_write and not stmt.conditions:
                warnings[name].append(
                    f"INFO: Write actions in {stmt.sid} without conditions "
                    f"(consider adding MFA or IP restrictions)")

    for name, warns in warnings.items():
        status = "PASS" if not warns else f"{len(warns)} issue(s)"
        print(f"\n  Policy: {name} [{status}]")
        if warns:
            for w in warns:
                print(f"    - {w}")
        else:
            print(f"    - No issues detected (well-scoped)")
    print()


def demo_policy_document():
    """Display a policy in standard JSON format."""
    print("=" * 75)
    print("Policy Document (AWS JSON format)")
    print("=" * 75)

    policy = create_sample_policies()["S3ScopedAccess"]
    print(f"\n  Policy: {policy.name}")
    print(f"  Description: {policy.description}")
    print()
    # Indent the JSON output for display
    for line in policy.to_json().split("\n"):
        print(f"  {line}")
    print()
    print("  Key insight: The explicit Deny statement for 'secrets/*' overrides")
    print("  the broad Allow on 'my-app-data/*'. In IAM, Deny ALWAYS wins.")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_policy_evaluation()
    demo_least_privilege_analysis()
    demo_policy_document()
