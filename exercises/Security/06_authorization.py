"""
Exercise Solutions: Authorization and Access Control
====================================================
Lesson 06 from Security topic.

Covers RBAC, ABAC, authorization vulnerability fixes,
multi-tenant authorization, and OPA-style policy engines.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Exercise 1: RBAC System for a Blogging Platform
# ---------------------------------------------------------------------------

class Permission(Enum):
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    POST_CREATE = "post:create"
    POST_READ = "post:read"
    POST_UPDATE = "post:update"
    POST_DELETE = "post:delete"
    COMMENT_CREATE = "comment:create"
    COMMENT_READ = "comment:read"
    COMMENT_DELETE = "comment:delete"
    COMMENT_MODERATE = "comment:moderate"
    SETTINGS_READ = "settings:read"
    SETTINGS_UPDATE = "settings:update"


@dataclass
class Role:
    name: str
    permissions: set[str]
    parent: Optional[str] = None


@dataclass
class BlogUser:
    id: int
    username: str
    roles: set[str] = field(default_factory=set)


class BlogRBAC:
    """Complete RBAC system with role hierarchy for a blogging platform."""

    # Role hierarchy: super_admin > admin > moderator > author > reader
    HIERARCHY_ORDER = ["reader", "author", "moderator", "admin", "super_admin"]

    def __init__(self):
        self.roles: dict[str, Role] = {}
        self.users: dict[int, BlogUser] = {}
        self._setup_default_roles()

    def _setup_default_roles(self):
        """Set up the default role hierarchy."""
        self.create_role("reader", {
            "post:read", "comment:read", "comment:create",
        })
        self.create_role("author", {
            "post:create", "post:update", "post:delete",
        }, parent="reader")
        self.create_role("moderator", {
            "comment:moderate", "comment:delete",
        }, parent="author")
        self.create_role("admin", {
            "user:create", "user:read", "user:update", "user:delete",
            "settings:read", "settings:update",
        }, parent="moderator")
        self.create_role("super_admin", set(), parent="admin")

    def create_role(self, name: str, permissions: set[str],
                    parent: str = None) -> bool:
        """Create a new role with optional parent."""
        if parent and parent not in self.roles:
            print(f"  Error: Parent role '{parent}' does not exist")
            return False
        self.roles[name] = Role(name=name, permissions=permissions, parent=parent)
        return True

    def _get_all_permissions(self, role_name: str) -> set[str]:
        """Get all permissions for a role including inherited ones."""
        if role_name not in self.roles:
            return set()
        role = self.roles[role_name]
        perms = set(role.permissions)
        if role.parent:
            perms |= self._get_all_permissions(role.parent)
        return perms

    def _get_hierarchy_level(self, role_name: str) -> int:
        """Get the hierarchy level of a role (higher = more powerful)."""
        try:
            return self.HIERARCHY_ORDER.index(role_name)
        except ValueError:
            return -1

    def assign_role(self, admin_id: int, target_user_id: int,
                    role: str) -> bool:
        """Assign role (only admins can do this)."""
        admin = self.users.get(admin_id)
        if not admin:
            print(f"  Error: Admin user {admin_id} not found")
            return False

        # Check if assigner has admin+ privileges
        if not self.check_permission(admin_id, "user:update"):
            print(f"  Error: User {admin_id} lacks permission to assign roles")
            return False

        # Prevent privilege escalation: cannot assign roles above own level
        admin_max_level = max(
            (self._get_hierarchy_level(r) for r in admin.roles),
            default=-1,
        )
        target_level = self._get_hierarchy_level(role)
        if target_level >= admin_max_level:
            print(f"  Error: Cannot assign role '{role}' (at or above own level)")
            return False

        target = self.users.get(target_user_id)
        if not target:
            print(f"  Error: Target user {target_user_id} not found")
            return False

        target.roles.add(role)
        print(f"  Role '{role}' assigned to user {target_user_id}")
        return True

    def check_permission(self, user_id: int, permission: str) -> bool:
        """Check if user has permission (including inherited)."""
        user = self.users.get(user_id)
        if not user:
            return False
        for role_name in user.roles:
            if permission in self._get_all_permissions(role_name):
                return True
        return False

    def get_accessible_resources(self, user_id: int,
                                 resource_type: str) -> list[str]:
        """Get all permissions a user has for a resource type."""
        user = self.users.get(user_id)
        if not user:
            return []
        all_perms = set()
        for role_name in user.roles:
            all_perms |= self._get_all_permissions(role_name)
        return [p for p in all_perms if p.startswith(f"{resource_type}:")]


def exercise_1_rbac():
    """Demonstrate the RBAC system."""
    rbac = BlogRBAC()

    # Create users
    rbac.users[1] = BlogUser(1, "super_admin", {"super_admin"})
    rbac.users[2] = BlogUser(2, "admin_user", {"admin"})
    rbac.users[3] = BlogUser(3, "moderator_user", {"moderator"})
    rbac.users[4] = BlogUser(4, "author_user", {"author"})
    rbac.users[5] = BlogUser(5, "reader_user", {"reader"})

    # Test permissions
    print("Permission checks:")
    tests = [
        (5, "post:read", True),
        (5, "post:create", False),
        (4, "post:create", True),
        (4, "comment:moderate", False),
        (3, "comment:moderate", True),
        (3, "post:create", True),  # Inherited from author
        (2, "settings:update", True),
    ]
    for user_id, perm, expected in tests:
        result = rbac.check_permission(user_id, perm)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] User {user_id} has {perm}: {result}")

    # Role assignment
    print("\nRole assignments:")
    rbac.assign_role(2, 5, "author")       # Admin assigns author -> OK
    rbac.assign_role(4, 5, "moderator")    # Author assigns moderator -> DENIED
    rbac.assign_role(2, 5, "admin")        # Admin assigns admin -> DENIED (same level)

    # Accessible resources
    print(f"\nReader's post permissions: "
          f"{rbac.get_accessible_resources(5, 'post')}")


# ---------------------------------------------------------------------------
# Exercise 2: ABAC Policy Engine (Healthcare)
# ---------------------------------------------------------------------------

@dataclass
class ABACPolicy:
    name: str
    condition: Callable
    effect: str  # "allow" or "deny"
    priority: int = 0  # Higher = evaluated first


class HospitalABAC:
    """Attribute-Based Access Control engine for a hospital system."""

    def __init__(self):
        self.policies: list[ABACPolicy] = []
        self._setup_default_policies()

    def _setup_default_policies(self):
        """Set up hospital access policies."""

        # Policy 1: Doctors can read patient records in their department
        self.add_policy(
            "doctor_department_access",
            lambda s, r, a, c: (
                s.get("role") == "doctor"
                and a.get("action") == "read"
                and r.get("type") == "patient_record"
                and s.get("department") == r.get("department")
            ),
            "allow",
            priority=10,
        )

        # Policy 2: Nurses can read records during their shift only
        self.add_policy(
            "nurse_shift_access",
            lambda s, r, a, c: (
                s.get("role") == "nurse"
                and a.get("action") == "read"
                and r.get("type") == "patient_record"
                and s.get("department") == r.get("department")
                and c.get("is_shift_active", False)
            ),
            "allow",
            priority=10,
        )

        # Policy 3: Emergency doctors can override department restrictions
        self.add_policy(
            "emergency_override",
            lambda s, r, a, c: (
                s.get("role") == "doctor"
                and s.get("is_emergency", False)
                and a.get("action") == "read"
                and r.get("type") == "patient_record"
            ),
            "allow",
            priority=20,  # Higher priority
        )

        # Policy 4: No access from non-hospital IP addresses
        self.add_policy(
            "hospital_network_only",
            lambda s, r, a, c: not c.get("ip", "").startswith("10."),
            "deny",
            priority=100,  # Highest priority deny
        )

        # Policy 5: Psychiatry records require additional clearance
        self.add_policy(
            "psychiatry_clearance",
            lambda s, r, a, c: (
                r.get("department") == "psychiatry"
                and not s.get("has_psychiatry_clearance", False)
            ),
            "deny",
            priority=50,
        )

        # Policy 6: Research access requires IRB approval
        self.add_policy(
            "research_irb_required",
            lambda s, r, a, c: (
                a.get("purpose") == "research"
                and not s.get("has_irb_approval", False)
            ),
            "deny",
            priority=50,
        )

    def add_policy(self, name: str, condition: Callable,
                   effect: str, priority: int = 0) -> None:
        self.policies.append(ABACPolicy(name, condition, effect, priority))
        # Sort by priority (highest first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)

    def evaluate(self, subject: dict, resource: dict,
                 action: dict, context: dict) -> tuple[str, str]:
        """
        Evaluate all policies. Default deny.
        Returns (decision, matching_policy_name).
        """
        for policy in self.policies:
            try:
                if policy.condition(subject, resource, action, context):
                    return policy.effect, policy.name
            except Exception:
                continue  # Skip policies that error

        return "deny", "default_deny"


def exercise_2_abac():
    """Demonstrate the hospital ABAC system."""
    abac = HospitalABAC()

    test_cases = [
        {
            "desc": "Doctor reads own department record",
            "subject": {"role": "doctor", "department": "cardiology"},
            "resource": {"type": "patient_record", "department": "cardiology"},
            "action": {"action": "read"},
            "context": {"ip": "10.0.1.50"},
            "expected": "allow",
        },
        {
            "desc": "Doctor reads other department record",
            "subject": {"role": "doctor", "department": "cardiology"},
            "resource": {"type": "patient_record", "department": "neurology"},
            "action": {"action": "read"},
            "context": {"ip": "10.0.1.50"},
            "expected": "deny",
        },
        {
            "desc": "Emergency doctor overrides department",
            "subject": {"role": "doctor", "department": "cardiology",
                        "is_emergency": True},
            "resource": {"type": "patient_record", "department": "neurology"},
            "action": {"action": "read"},
            "context": {"ip": "10.0.1.50"},
            "expected": "allow",
        },
        {
            "desc": "Access from external IP",
            "subject": {"role": "doctor", "department": "cardiology"},
            "resource": {"type": "patient_record", "department": "cardiology"},
            "action": {"action": "read"},
            "context": {"ip": "203.0.113.5"},
            "expected": "deny",
        },
        {
            "desc": "Psychiatry access without clearance",
            "subject": {"role": "doctor", "department": "psychiatry"},
            "resource": {"type": "patient_record", "department": "psychiatry"},
            "action": {"action": "read"},
            "context": {"ip": "10.0.1.50"},
            "expected": "deny",
        },
    ]

    print("Hospital ABAC Policy Evaluation:")
    for tc in test_cases:
        decision, policy = abac.evaluate(
            tc["subject"], tc["resource"], tc["action"], tc["context"]
        )
        status = "PASS" if decision == tc["expected"] else "FAIL"
        print(f"  [{status}] {tc['desc']}")
        print(f"         Decision: {decision} (via: {policy})")


# ---------------------------------------------------------------------------
# Exercise 3: Fix Authorization Vulnerabilities
# ---------------------------------------------------------------------------

def exercise_3_fix_authz_vulnerabilities():
    """Identify and fix all 7 authorization vulnerabilities."""
    issues = [
        {
            "code": "@app.route('/api/users', methods=['GET'])\ndef list_all_users():",
            "issue": "No authentication required — anyone can list all users",
            "fix": "Add @require_auth and @require_role('admin') decorators",
        },
        {
            "code": "@app.route('/api/users/<int:id>/password', methods=['PUT'])\ndef change_password(id):",
            "issue": "No ownership check — any authenticated user can change "
                     "any other user's password (IDOR)",
            "fix": "Verify current_user.id == id or current_user.role == 'admin'",
        },
        {
            "code": "@app.route('/api/posts/<int:id>', methods=['DELETE'])\ndef delete_post(id):",
            "issue": "No ownership check — any authenticated user can delete "
                     "any post (IDOR)",
            "fix": "Verify post.author_id == current_user.id or is_moderator()",
        },
        {
            "code": "@app.route('/api/admin/promote', methods=['POST'])\ndef promote_user():",
            "issue": "No admin-level authorization check — any authenticated "
                     "user can promote users",
            "fix": "Add @require_role('admin') decorator",
        },
        {
            "code": "role = request.json['role']",
            "issue": "No validation on role value — attacker can set "
                     "role='super_admin' to escalate privileges",
            "fix": "Whitelist allowed roles: if role not in ALLOWED_ROLES: abort(400)",
        },
        {
            "code": "@app.route('/api/files/<path:filepath>')\ndef download_file(filepath):",
            "issue": "Path traversal — attacker can use '../' to access "
                     "files outside /uploads (e.g., /etc/passwd)",
            "fix": "Sanitize path: use os.path.realpath() and verify it starts "
                   "with the allowed base directory",
        },
        {
            "code": "@app.route('/api/settings', methods=['PUT'])\ndef update_settings():",
            "issue": "Mass assignment — user can update ANY setting including "
                     "admin-only ones (e.g., billing, permissions)",
            "fix": "Whitelist updateable fields: only allow specific keys",
        },
    ]

    print("Authorization Vulnerability Fixes")
    print("=" * 60)
    for i, item in enumerate(issues, 1):
        print(f"\nIssue {i}:")
        print(f"  Code:    {item['code']}")
        print(f"  Problem: {item['issue']}")
        print(f"  Fix:     {item['fix']}")


# ---------------------------------------------------------------------------
# Exercise 4: Multi-Tenant Authorization
# ---------------------------------------------------------------------------

@dataclass
class Organization:
    id: str
    name: str
    owner_id: str
    members: dict[str, str] = field(default_factory=dict)  # user_id -> role


class MultiTenantAuth:
    """Multi-tenant authorization for SaaS applications."""

    def __init__(self):
        self.orgs: dict[str, Organization] = {}
        self.user_org: dict[str, str] = {}  # user_id -> org_id
        self.super_admins: set[str] = set()

    def create_org(self, org_id: str, owner_id: str) -> dict:
        """Create a new organization."""
        org = Organization(
            id=org_id, name=org_id, owner_id=owner_id,
            members={owner_id: "admin"},
        )
        self.orgs[org_id] = org
        self.user_org[owner_id] = org_id
        return {"org_id": org_id, "owner": owner_id}

    def add_member(self, org_id: str, user_id: str, role: str,
                   requester_id: str) -> bool:
        """Add a member to an org (requires admin role in that org)."""
        org = self.orgs.get(org_id)
        if not org:
            return False
        # Check requester is admin of this org
        if org.members.get(requester_id) != "admin":
            return False
        org.members[user_id] = role
        self.user_org[user_id] = org_id
        return True

    def check_tenant_access(self, user_id: str, org_id: str,
                            resource_id: str, action: str) -> bool:
        """Check if user can access a resource in the given org."""
        # Super admins can access any tenant
        if user_id in self.super_admins:
            return True

        # User must belong to the org
        if self.user_org.get(user_id) != org_id:
            return False  # Cross-tenant access denied

        org = self.orgs.get(org_id)
        if not org or user_id not in org.members:
            return False

        # Role-based check within tenant
        role = org.members[user_id]
        if action in ("read",) and role in ("admin", "member", "viewer"):
            return True
        if action in ("write", "update") and role in ("admin", "member"):
            return True
        if action in ("delete", "manage") and role == "admin":
            return True

        return False

    def ensure_tenant_isolation(self, query_params: dict,
                                user_org_id: str) -> dict:
        """Add tenant filter to query parameters."""
        # Always inject the org_id filter to prevent cross-tenant data access
        query_params["org_id"] = user_org_id
        return query_params


def exercise_4_multi_tenant():
    """Demonstrate multi-tenant authorization."""
    auth = MultiTenantAuth()

    # Create two organizations
    auth.create_org("acme-corp", "alice")
    auth.create_org("beta-inc", "bob")
    auth.add_member("acme-corp", "charlie", "member", "alice")

    # Super admin
    auth.super_admins.add("platform-admin")

    tests = [
        ("alice", "acme-corp", "doc-1", "read", True),
        ("alice", "acme-corp", "doc-1", "delete", True),
        ("charlie", "acme-corp", "doc-1", "read", True),
        ("charlie", "acme-corp", "doc-1", "delete", False),
        ("alice", "beta-inc", "doc-2", "read", False),  # Cross-tenant
        ("bob", "beta-inc", "doc-2", "read", True),
        ("platform-admin", "acme-corp", "doc-1", "read", True),  # Super admin
    ]

    print("Multi-Tenant Authorization:")
    for user, org, resource, action, expected in tests:
        result = auth.check_tenant_access(user, org, resource, action)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {user} -> {org}/{resource} ({action}): {result}")

    # Tenant isolation in queries
    query = {"status": "active"}
    filtered = auth.ensure_tenant_isolation(query.copy(), "acme-corp")
    print(f"\nQuery with tenant filter: {filtered}")


# ---------------------------------------------------------------------------
# Exercise 5: OPA-Style Policy Engine (Python)
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Simple OPA-style policy engine in Python."""

    def __init__(self):
        self.policies: list[dict] = []

    def add_policy(self, name: str, condition: Callable, effect: str = "allow"):
        self.policies.append({"name": name, "condition": condition, "effect": effect})

    def evaluate(self, input_data: dict) -> tuple[bool, list[str]]:
        """Evaluate all policies. Default deny."""
        allow_reasons = []
        deny_reasons = []

        for policy in self.policies:
            if policy["condition"](input_data):
                if policy["effect"] == "allow":
                    allow_reasons.append(policy["name"])
                else:
                    deny_reasons.append(policy["name"])

        # Deny takes precedence
        if deny_reasons:
            return False, deny_reasons
        if allow_reasons:
            return True, allow_reasons
        return False, ["default_deny"]


def exercise_5_opa_policies():
    """Demonstrate OPA-style policies in Python."""
    engine = PolicyEngine()

    # Policy 1: Department access
    engine.add_policy(
        "department_access",
        lambda d: d["user"]["department"] == d["resource"]["department"],
        "allow",
    )

    # Policy 2: Manager cross-department access
    engine.add_policy(
        "manager_cross_dept",
        lambda d: (
            d["user"]["role"] == "manager"
            and d["resource"]["department"] in d["user"].get("managed_depts", [])
        ),
        "allow",
    )

    # Policy 3: Financial reports require finance role AND senior level
    engine.add_policy(
        "finance_reports_denied",
        lambda d: (
            d["resource"]["type"] == "financial_report"
            and not (d["user"]["role"] == "finance" and d["user"].get("level") == "senior")
        ),
        "deny",
    )

    # Policy 4: Rate limiting (basic tier)
    engine.add_policy(
        "rate_limit_basic",
        lambda d: (
            d["user"].get("tier") == "basic"
            and d.get("request_count_hour", 0) > 100
        ),
        "deny",
    )

    # Policy 5: MFA for top_secret resources
    engine.add_policy(
        "mfa_required_top_secret",
        lambda d: (
            d["resource"].get("classification") == "top_secret"
            and (time.time() - d["user"].get("last_mfa_time", 0)) > 1800
        ),
        "deny",
    )

    # Test cases
    tests = [
        {
            "desc": "Same department access",
            "input": {
                "user": {"department": "engineering", "role": "engineer"},
                "resource": {"department": "engineering", "type": "code"},
            },
            "expected": True,
        },
        {
            "desc": "Manager accessing managed department",
            "input": {
                "user": {"department": "engineering", "role": "manager",
                         "managed_depts": ["qa", "design"]},
                "resource": {"department": "qa", "type": "report"},
            },
            "expected": True,
        },
        {
            "desc": "Finance report without finance role",
            "input": {
                "user": {"department": "engineering", "role": "engineer"},
                "resource": {"department": "engineering", "type": "financial_report"},
            },
            "expected": False,
        },
    ]

    print("OPA-Style Policy Engine:")
    for tc in tests:
        allowed, reasons = engine.evaluate(tc["input"])
        status = "PASS" if allowed == tc["expected"] else "FAIL"
        print(f"  [{status}] {tc['desc']}: {allowed} ({', '.join(reasons)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: RBAC System")
    print("=" * 70)
    exercise_1_rbac()

    print("\n" + "=" * 70)
    print("Exercise 2: ABAC Policy Engine")
    print("=" * 70)
    exercise_2_abac()

    print("\n" + "=" * 70)
    print("Exercise 3: Fix Authorization Vulnerabilities")
    print("=" * 70)
    exercise_3_fix_authz_vulnerabilities()

    print("\n" + "=" * 70)
    print("Exercise 4: Multi-Tenant Authorization")
    print("=" * 70)
    exercise_4_multi_tenant()

    print("\n" + "=" * 70)
    print("Exercise 5: OPA-Style Policy Engine")
    print("=" * 70)
    exercise_5_opa_policies()
