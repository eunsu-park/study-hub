# 06. Authorization and Access Control

**Previous**: [05. Authentication Systems](05_Authentication.md) | **Next**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md)

---

Authorization determines what an authenticated user is allowed to do. While authentication answers "Who are you?", authorization answers "What can you do?" A robust authorization system enforces the principle of least privilege, ensuring that users and services have only the minimum permissions necessary. This lesson covers the major access control models (RBAC, ABAC, ACL), policy engines, token-based authorization with JWT and OAuth scopes, practical implementation patterns in Python/Flask, and common authorization vulnerabilities.

## Learning Objectives

- Distinguish between authentication and authorization
- Implement Role-Based Access Control (RBAC) systems
- Understand Attribute-Based Access Control (ABAC) and when to use it
- Work with Access Control Lists (ACLs) for resource-level permissions
- Use policy engines like OPA (Open Policy Agent) for externalized authorization
- Leverage JWT claims and OAuth 2.0 scopes for API authorization
- Build authorization middleware and decorators in Flask
- Identify and prevent common authorization vulnerabilities (IDOR, privilege escalation)

---

## 1. Authentication vs Authorization

```
┌─────────────────────────────────────────────────────────────────┐
│          Authentication vs Authorization                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Authentication (AuthN)         Authorization (AuthZ)            │
│  ─────────────────────         ──────────────────────            │
│  "Who are you?"                "What can you do?"                │
│                                                                  │
│  Verifies identity             Verifies permissions              │
│  Happens FIRST                 Happens AFTER authentication      │
│  401 Unauthorized              403 Forbidden                     │
│  (not authenticated)           (authenticated but not allowed)   │
│                                                                  │
│                                                                  │
│  Request Flow:                                                   │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐           │
│  │ Request  │───▶│ Authentication│───▶│Authorization│           │
│  │          │    │ "Who is this?"│    │ "Can they   │           │
│  └──────────┘    │              │    │  do this?"  │            │
│                  └──────┬───────┘    └──────┬──────┘           │
│                         │                   │                    │
│                    ┌────┴───┐          ┌────┴───┐               │
│                    │        │          │        │                │
│                  Valid   Invalid     Allowed  Denied             │
│                    │        │          │        │                │
│                    │    401 Error      │    403 Error            │
│                    │                   │                          │
│                    └───────┬───────────┘                         │
│                            │                                     │
│                            ▼                                     │
│                     ┌──────────┐                                │
│                     │ Resource │                                 │
│                     │  Access  │                                 │
│                     └──────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Role-Based Access Control (RBAC)

### 2.1 RBAC Concepts

RBAC assigns permissions to **roles**, and users are assigned to roles. This simplifies management because you manage role-permission mappings rather than individual user permissions.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RBAC Model                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Users              Roles              Permissions               │
│  ┌────────┐        ┌──────────┐       ┌──────────────────┐     │
│  │ Alice  │───────▶│  Admin   │──────▶│ create_user      │     │
│  └────────┘        │          │──────▶│ delete_user      │     │
│  ┌────────┐        │          │──────▶│ view_reports     │     │
│  │  Bob   │───┐    └──────────┘──────▶│ manage_settings  │     │
│  └────────┘   │    ┌──────────┐       └──────────────────┘     │
│               └───▶│  Editor  │       ┌──────────────────┐     │
│  ┌────────┐        │          │──────▶│ create_post      │     │
│  │ Carol  │───────▶│          │──────▶│ edit_post        │     │
│  └────────┘        └──────────┘──────▶│ delete_own_post  │     │
│  ┌────────┐        ┌──────────┐       └──────────────────┘     │
│  │  Dan   │───────▶│  Viewer  │       ┌──────────────────┐     │
│  └────────┘        │          │──────▶│ view_post        │     │
│                    └──────────┘──────▶│ view_reports     │     │
│                                       └──────────────────┘     │
│                                                                  │
│  Hierarchy (optional):                                           │
│  Admin ──▶ Editor ──▶ Viewer                                    │
│  (Admin inherits all Editor and Viewer permissions)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 RBAC Implementation in Python

```python
"""
rbac.py - Role-Based Access Control implementation
"""
from dataclasses import dataclass, field
from typing import Set, Dict, Optional
from enum import Enum
from functools import wraps


# ==============================================================
# Core RBAC Model
# ==============================================================

class Permission(str, Enum):
    """Define all permissions in the system."""
    # User management
    CREATE_USER = "user:create"
    READ_USER = "user:read"
    UPDATE_USER = "user:update"
    DELETE_USER = "user:delete"

    # Post management
    CREATE_POST = "post:create"
    READ_POST = "post:read"
    UPDATE_POST = "post:update"
    DELETE_POST = "post:delete"
    PUBLISH_POST = "post:publish"

    # Report management
    VIEW_REPORTS = "report:view"
    EXPORT_REPORTS = "report:export"

    # System
    MANAGE_SETTINGS = "system:settings"
    VIEW_AUDIT_LOG = "system:audit"


@dataclass
class Role:
    """A role is a named collection of permissions."""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    parent: Optional['Role'] = None  # For role hierarchy

    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions including inherited ones."""
        perms = set(self.permissions)
        if self.parent:
            perms |= self.parent.get_all_permissions()
        return perms

    def has_permission(self, permission: Permission) -> bool:
        """Check if this role has a specific permission."""
        return permission in self.get_all_permissions()


@dataclass
class User:
    """A user with one or more roles."""
    id: int
    username: str
    roles: Set[Role] = field(default_factory=set)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission through any role."""
        return any(role.has_permission(permission) for role in self.roles)

    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return any(role.name == role_name for role in self.roles)

    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions from all roles."""
        perms = set()
        for role in self.roles:
            perms |= role.get_all_permissions()
        return perms


# ==============================================================
# Role Definitions
# ==============================================================

def create_default_roles() -> Dict[str, Role]:
    """Create the default role hierarchy."""

    # Base role - everyone gets these
    viewer = Role(
        name="viewer",
        permissions={
            Permission.READ_POST,
            Permission.READ_USER,
            Permission.VIEW_REPORTS,
        }
    )

    # Editor inherits from Viewer
    editor = Role(
        name="editor",
        permissions={
            Permission.CREATE_POST,
            Permission.UPDATE_POST,
            Permission.DELETE_POST,
            Permission.PUBLISH_POST,
            Permission.EXPORT_REPORTS,
        },
        parent=viewer
    )

    # Admin inherits from Editor
    admin = Role(
        name="admin",
        permissions={
            Permission.CREATE_USER,
            Permission.UPDATE_USER,
            Permission.DELETE_USER,
            Permission.MANAGE_SETTINGS,
            Permission.VIEW_AUDIT_LOG,
        },
        parent=editor
    )

    return {
        "viewer": viewer,
        "editor": editor,
        "admin": admin,
    }


# ==============================================================
# RBAC Manager
# ==============================================================

class RBACManager:
    """Centralized RBAC management."""

    def __init__(self):
        self.roles = create_default_roles()
        self.users: Dict[int, User] = {}

    def create_user(self, user_id: int, username: str,
                    role_names: list = None) -> User:
        """Create a user with specified roles."""
        roles = set()
        for name in (role_names or ["viewer"]):
            if name in self.roles:
                roles.add(self.roles[name])
            else:
                raise ValueError(f"Unknown role: {name}")

        user = User(id=user_id, username=username, roles=roles)
        self.users[user_id] = user
        return user

    def assign_role(self, user_id: int, role_name: str):
        """Assign a role to a user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} not found")
        self.users[user_id].roles.add(self.roles[role_name])

    def revoke_role(self, user_id: int, role_name: str):
        """Remove a role from a user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        self.users[user_id].roles = {
            r for r in self.users[user_id].roles
            if r.name != role_name
        }

    def check_access(self, user_id: int, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        user = self.users.get(user_id)
        if not user:
            return False
        return user.has_permission(permission)


# ==============================================================
# Demo
# ==============================================================

if __name__ == "__main__":
    rbac = RBACManager()

    # Create users
    alice = rbac.create_user(1, "alice", ["admin"])
    bob = rbac.create_user(2, "bob", ["editor"])
    carol = rbac.create_user(3, "carol", ["viewer"])

    # Check permissions
    print("=== Permission Checks ===")
    checks = [
        (alice, Permission.DELETE_USER, "Alice can delete users"),
        (alice, Permission.READ_POST, "Alice can read posts (inherited)"),
        (bob, Permission.CREATE_POST, "Bob can create posts"),
        (bob, Permission.DELETE_USER, "Bob can delete users"),
        (carol, Permission.READ_POST, "Carol can read posts"),
        (carol, Permission.CREATE_POST, "Carol can create posts"),
    ]

    for user, perm, description in checks:
        result = user.has_permission(perm)
        status = "ALLOWED" if result else "DENIED"
        print(f"  [{status}] {description}")

    # List all permissions
    print(f"\nAlice's permissions: {len(alice.get_all_permissions())}")
    for p in sorted(alice.get_all_permissions(), key=lambda x: x.value):
        print(f"  - {p.value}")
```

### 2.3 Flask RBAC Middleware

```python
"""
flask_rbac.py - RBAC authorization middleware for Flask
"""
from flask import Flask, request, jsonify, g
from functools import wraps
import jwt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Use env var in production


# ==============================================================
# Authorization Decorators
# ==============================================================

def require_auth(f):
    """Decorator: Require authenticated user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Authentication required"}), 401

        try:
            payload = jwt.decode(
                token,
                app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            g.current_user = {
                'id': payload['sub'],
                'roles': payload.get('roles', []),
                'permissions': payload.get('permissions', []),
            }
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated


def require_role(*allowed_roles):
    """Decorator: Require user to have one of the specified roles."""
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            user_roles = set(g.current_user.get('roles', []))
            if not user_roles.intersection(set(allowed_roles)):
                return jsonify({
                    "error": "Forbidden",
                    "detail": f"Required roles: {allowed_roles}"
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


def require_permission(*required_permissions):
    """Decorator: Require user to have all specified permissions."""
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            user_perms = set(g.current_user.get('permissions', []))
            missing = set(required_permissions) - user_perms
            if missing:
                return jsonify({
                    "error": "Forbidden",
                    "detail": f"Missing permissions: {list(missing)}"
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ==============================================================
# Protected Routes
# ==============================================================

@app.route('/api/posts', methods=['GET'])
@require_auth
def list_posts():
    """Any authenticated user can list posts."""
    return jsonify({"posts": []})


@app.route('/api/posts', methods=['POST'])
@require_permission('post:create')
def create_post():
    """Only users with post:create permission."""
    data = request.json
    # Create post logic...
    return jsonify({"message": "Post created"}), 201


@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
@require_permission('post:delete')
def delete_post(post_id):
    """Only users with post:delete permission."""
    # Additional check: can only delete own posts unless admin
    post = get_post(post_id)  # Your DB lookup

    if not post:
        return jsonify({"error": "Not found"}), 404

    user = g.current_user
    is_admin = 'admin' in user.get('roles', [])
    is_owner = post['author_id'] == user['id']

    if not is_admin and not is_owner:
        return jsonify({"error": "Can only delete your own posts"}), 403

    # Delete post logic...
    return jsonify({"message": "Post deleted"})


@app.route('/api/admin/users', methods=['GET'])
@require_role('admin')
def list_users():
    """Admin-only endpoint."""
    return jsonify({"users": []})


@app.route('/api/admin/settings', methods=['PUT'])
@require_role('admin')
def update_settings():
    """Admin-only: modify system settings."""
    data = request.json
    # Update settings logic...
    return jsonify({"message": "Settings updated"})


# ==============================================================
# Dynamic Permission Check (for resource-level auth)
# ==============================================================

def check_resource_access(user_id: int, resource_type: str,
                         resource_id: int, action: str) -> bool:
    """
    Check if a user can perform an action on a specific resource.
    This goes beyond role-based checks to resource-level authorization.
    """
    # Example: Check if user owns the resource or has admin role
    user = get_user(user_id)
    resource = get_resource(resource_type, resource_id)

    if not user or not resource:
        return False

    # Admin can do anything
    if 'admin' in user.get('roles', []):
        return True

    # Owner can read/update their own resources
    if resource.get('owner_id') == user_id:
        if action in ('read', 'update', 'delete'):
            return True

    # Check shared access
    shared_with = resource.get('shared_with', [])
    for share in shared_with:
        if share['user_id'] == user_id:
            if action in share.get('allowed_actions', []):
                return True

    return False
```

---

## 3. Attribute-Based Access Control (ABAC)

### 3.1 ABAC Concepts

ABAC makes access decisions based on **attributes** of the subject (user), resource, action, and environment. It is more flexible than RBAC but also more complex.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABAC Model                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Access Decision =                                               │
│    f(Subject Attributes, Resource Attributes,                    │
│      Action Attributes, Environment Attributes)                  │
│                                                                  │
│  Subject Attributes:     Resource Attributes:                    │
│  ├── role: "doctor"      ├── type: "medical_record"              │
│  ├── department: "ER"    ├── department: "ER"                    │
│  ├── clearance: "L3"     ├── sensitivity: "L2"                   │
│  └── certification: true └── owner: "patient_123"                │
│                                                                  │
│  Action Attributes:      Environment Attributes:                 │
│  ├── type: "read"        ├── time: "14:30 UTC"                   │
│  └── purpose: "treatment"├── ip_address: "10.0.1.50"            │
│                          ├── location: "hospital_network"        │
│                          └── device_trust: "managed"             │
│                                                                  │
│  Policy Example:                                                 │
│  "A doctor in the ER department can read medical records          │
│   in the ER department during work hours from hospital            │
│   network on a managed device."                                  │
│                                                                  │
│  Subject.role == "doctor" AND                                    │
│  Subject.department == Resource.department AND                   │
│  Action.type == "read" AND                                       │
│  Environment.time BETWEEN "07:00" AND "19:00" AND               │
│  Environment.location == "hospital_network"                      │
│  ──▶ ALLOW                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 When to Use RBAC vs ABAC

| Criteria | RBAC | ABAC |
|----------|------|------|
| Number of roles | Small, well-defined | Many or dynamic |
| Access decisions | Role membership | Multiple attributes |
| Complexity | Simple to implement | Complex but flexible |
| "Role explosion" | Risk (too many roles) | Not an issue |
| Context-aware | No (static roles) | Yes (time, location, etc.) |
| Compliance | Basic needs | Regulatory requirements |
| Best for | Most web apps, APIs | Healthcare, finance, government |
| Performance | Fast (role lookup) | Slower (policy evaluation) |

### 3.3 ABAC Implementation

```python
"""
abac.py - Attribute-Based Access Control implementation
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Callable
from enum import Enum


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class AccessRequest:
    """Represents a request for access to a resource."""
    # Subject attributes (who)
    subject: Dict[str, Any]

    # Resource attributes (what)
    resource: Dict[str, Any]

    # Action attributes (how)
    action: Dict[str, Any]

    # Environment attributes (context)
    environment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Policy:
    """A single ABAC policy."""
    name: str
    description: str
    priority: int  # Lower = higher priority
    effect: Decision  # ALLOW or DENY
    condition: Callable[[AccessRequest], bool]

    def evaluate(self, request: AccessRequest) -> Decision:
        """Evaluate this policy against a request."""
        try:
            if self.condition(request):
                return self.effect
            return Decision.NOT_APPLICABLE
        except (KeyError, TypeError):
            return Decision.NOT_APPLICABLE


class PolicyEngine:
    """Evaluates access requests against a set of policies."""

    def __init__(self, default_decision: Decision = Decision.DENY):
        self.policies: List[Policy] = []
        self.default_decision = default_decision

    def add_policy(self, policy: Policy):
        """Add a policy to the engine."""
        self.policies.append(policy)
        # Keep sorted by priority
        self.policies.sort(key=lambda p: p.priority)

    def evaluate(self, request: AccessRequest) -> Decision:
        """
        Evaluate all policies. Uses deny-overrides combining algorithm:
        - If any policy says DENY, result is DENY
        - If at least one says ALLOW and none say DENY, result is ALLOW
        - Otherwise, use default decision
        """
        has_allow = False

        for policy in self.policies:
            decision = policy.evaluate(request)

            if decision == Decision.DENY:
                return Decision.DENY  # Deny overrides

            if decision == Decision.ALLOW:
                has_allow = True

        return Decision.ALLOW if has_allow else self.default_decision


# ==============================================================
# Example Policies
# ==============================================================

def create_medical_policies() -> PolicyEngine:
    """Create policies for a healthcare system."""
    engine = PolicyEngine(default_decision=Decision.DENY)

    # Policy 1: Doctors can read patient records in their department
    engine.add_policy(Policy(
        name="doctor_read_department_records",
        description="Doctors can read records in their department",
        priority=10,
        effect=Decision.ALLOW,
        condition=lambda req: (
            req.subject.get("role") == "doctor" and
            req.action.get("type") == "read" and
            req.resource.get("type") == "medical_record" and
            req.subject.get("department") == req.resource.get("department")
        )
    ))

    # Policy 2: Doctors can write records for their patients
    engine.add_policy(Policy(
        name="doctor_write_own_patients",
        description="Doctors can update records of patients assigned to them",
        priority=10,
        effect=Decision.ALLOW,
        condition=lambda req: (
            req.subject.get("role") == "doctor" and
            req.action.get("type") in ("write", "update") and
            req.resource.get("type") == "medical_record" and
            req.subject.get("id") in req.resource.get("assigned_doctors", [])
        )
    ))

    # Policy 3: Nurses can read records in their department during shifts
    engine.add_policy(Policy(
        name="nurse_read_during_shift",
        description="Nurses can read records during their shift hours",
        priority=20,
        effect=Decision.ALLOW,
        condition=lambda req: (
            req.subject.get("role") == "nurse" and
            req.action.get("type") == "read" and
            req.resource.get("type") == "medical_record" and
            req.subject.get("department") == req.resource.get("department") and
            _is_during_shift(req.environment.get("time"),
                            req.subject.get("shift_start"),
                            req.subject.get("shift_end"))
        )
    ))

    # Policy 4: No access from untrusted devices
    engine.add_policy(Policy(
        name="deny_untrusted_devices",
        description="Deny access from non-managed devices",
        priority=1,  # High priority - evaluated first
        effect=Decision.DENY,
        condition=lambda req: (
            req.resource.get("sensitivity", "low") in ("high", "critical") and
            req.environment.get("device_trust") != "managed"
        )
    ))

    # Policy 5: Emergency override - on-duty doctors in ER
    engine.add_policy(Policy(
        name="emergency_override",
        description="ER doctors can access any record during emergency",
        priority=5,
        effect=Decision.ALLOW,
        condition=lambda req: (
            req.subject.get("role") == "doctor" and
            req.subject.get("department") == "emergency" and
            req.action.get("type") == "read" and
            req.environment.get("emergency_mode") is True
        )
    ))

    return engine


def _is_during_shift(current_time, shift_start, shift_end):
    """Check if current time is within shift hours."""
    if not all([current_time, shift_start, shift_end]):
        return False
    if isinstance(current_time, str):
        current_time = datetime.fromisoformat(current_time).time()
    shift_start = time.fromisoformat(shift_start)
    shift_end = time.fromisoformat(shift_end)

    if shift_start <= shift_end:
        return shift_start <= current_time <= shift_end
    else:  # Overnight shift
        return current_time >= shift_start or current_time <= shift_end


# ==============================================================
# Demo
# ==============================================================

if __name__ == "__main__":
    engine = create_medical_policies()

    # Test Case 1: Doctor reading records in their department
    request1 = AccessRequest(
        subject={"id": "dr_smith", "role": "doctor", "department": "cardiology"},
        resource={"type": "medical_record", "department": "cardiology",
                  "sensitivity": "high"},
        action={"type": "read"},
        environment={"device_trust": "managed", "time": "2024-01-15T10:30:00"}
    )
    print(f"Doctor reads own dept: {engine.evaluate(request1).value}")

    # Test Case 2: Doctor reading records in different department
    request2 = AccessRequest(
        subject={"id": "dr_smith", "role": "doctor", "department": "cardiology"},
        resource={"type": "medical_record", "department": "neurology",
                  "sensitivity": "high"},
        action={"type": "read"},
        environment={"device_trust": "managed"}
    )
    print(f"Doctor reads other dept: {engine.evaluate(request2).value}")

    # Test Case 3: Access from untrusted device (denied regardless)
    request3 = AccessRequest(
        subject={"id": "dr_smith", "role": "doctor", "department": "cardiology"},
        resource={"type": "medical_record", "department": "cardiology",
                  "sensitivity": "high"},
        action={"type": "read"},
        environment={"device_trust": "personal"}  # Not managed!
    )
    print(f"Untrusted device: {engine.evaluate(request3).value}")

    # Test Case 4: Emergency override
    request4 = AccessRequest(
        subject={"id": "dr_jones", "role": "doctor", "department": "emergency"},
        resource={"type": "medical_record", "department": "cardiology",
                  "sensitivity": "critical"},
        action={"type": "read"},
        environment={"device_trust": "managed", "emergency_mode": True}
    )
    print(f"Emergency override: {engine.evaluate(request4).value}")
```

---

## 4. Access Control Lists (ACL)

### 4.1 ACL Concepts

ACLs define permissions at the **individual resource level**. Each resource has a list of entries specifying which subjects (users, groups) can perform which actions.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACL Model                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Resource: "Project Plan.docx"                                   │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ ACL Entry          │ Permissions                      │      │
│  ├─────────────────────┼─────────────────────────────────┤      │
│  │ alice (owner)       │ read, write, delete, share      │      │
│  │ bob                 │ read, write                     │      │
│  │ carol               │ read                            │      │
│  │ engineering (group) │ read, comment                   │      │
│  │ * (everyone)        │ (no access)                     │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  Resource: "Company Financials.xlsx"                             │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ ACL Entry          │ Permissions                      │      │
│  ├─────────────────────┼─────────────────────────────────┤      │
│  │ alice (owner)       │ read, write, delete, share      │      │
│  │ finance (group)     │ read, write                     │      │
│  │ ceo                 │ read                            │      │
│  │ * (everyone)        │ (no access)                     │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  Unix File Permissions (simplified ACL):                         │
│  -rwxr-xr--  owner  group  file.txt                             │
│   │││ │││ │││                                                    │
│   │││ │││ └┴┴── Others: read only                               │
│   │││ └┴┴────── Group: read + execute                           │
│   └┴┴────────── Owner: read + write + execute                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 ACL Implementation

```python
"""
acl.py - Access Control List implementation for resource-level permissions
"""
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List
from enum import Flag, auto


class ACLPermission(Flag):
    """Permission flags (combinable with bitwise OR)."""
    NONE = 0
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    SHARE = auto()
    ADMIN = READ | WRITE | DELETE | SHARE  # All permissions combined


@dataclass
class ACLEntry:
    """A single ACL entry mapping a principal to permissions."""
    principal_type: str  # "user" or "group"
    principal_id: str
    permissions: ACLPermission


@dataclass
class Resource:
    """A resource with an access control list."""
    id: str
    name: str
    owner_id: str
    acl: List[ACLEntry] = field(default_factory=list)

    def grant(self, principal_type: str, principal_id: str,
              permissions: ACLPermission):
        """Grant permissions to a principal."""
        # Check if entry already exists
        for entry in self.acl:
            if (entry.principal_type == principal_type and
                entry.principal_id == principal_id):
                entry.permissions |= permissions  # Add permissions
                return

        # Create new entry
        self.acl.append(ACLEntry(
            principal_type=principal_type,
            principal_id=principal_id,
            permissions=permissions,
        ))

    def revoke(self, principal_type: str, principal_id: str,
               permissions: ACLPermission = None):
        """Revoke permissions (or all access) from a principal."""
        if permissions is None:
            # Remove entire entry
            self.acl = [
                e for e in self.acl
                if not (e.principal_type == principal_type and
                       e.principal_id == principal_id)
            ]
        else:
            for entry in self.acl:
                if (entry.principal_type == principal_type and
                    entry.principal_id == principal_id):
                    entry.permissions &= ~permissions  # Remove specific perms

    def check_access(self, user_id: str, user_groups: Set[str],
                     required: ACLPermission) -> bool:
        """Check if a user has the required permissions."""
        # Owner always has full access
        if user_id == self.owner_id:
            return True

        effective_perms = ACLPermission.NONE

        for entry in self.acl:
            if entry.principal_type == "user" and entry.principal_id == user_id:
                effective_perms |= entry.permissions
            elif (entry.principal_type == "group" and
                  entry.principal_id in user_groups):
                effective_perms |= entry.permissions

        return bool(effective_perms & required)

    def get_effective_permissions(self, user_id: str,
                                  user_groups: Set[str]) -> ACLPermission:
        """Get all effective permissions for a user."""
        if user_id == self.owner_id:
            return ACLPermission.ADMIN

        effective = ACLPermission.NONE
        for entry in self.acl:
            if entry.principal_type == "user" and entry.principal_id == user_id:
                effective |= entry.permissions
            elif (entry.principal_type == "group" and
                  entry.principal_id in user_groups):
                effective |= entry.permissions

        return effective


# ==============================================================
# ACL Manager with Database Backend (simplified)
# ==============================================================

class ACLManager:
    """Manage ACLs across multiple resources."""

    def __init__(self):
        self.resources: Dict[str, Resource] = {}

    def create_resource(self, resource_id: str, name: str,
                        owner_id: str) -> Resource:
        """Create a new resource."""
        resource = Resource(id=resource_id, name=name, owner_id=owner_id)
        self.resources[resource_id] = resource
        return resource

    def check(self, user_id: str, user_groups: Set[str],
              resource_id: str, permission: ACLPermission) -> bool:
        """Check access to a resource."""
        resource = self.resources.get(resource_id)
        if not resource:
            return False
        return resource.check_access(user_id, user_groups, permission)

    def share(self, resource_id: str, requesting_user_id: str,
              target_type: str, target_id: str,
              permissions: ACLPermission) -> bool:
        """Share a resource (only owner or users with SHARE can do this)."""
        resource = self.resources.get(resource_id)
        if not resource:
            return False

        # Check if requesting user can share
        if requesting_user_id != resource.owner_id:
            if not resource.check_access(
                requesting_user_id, set(), ACLPermission.SHARE
            ):
                return False

        resource.grant(target_type, target_id, permissions)
        return True


# ==============================================================
# Demo
# ==============================================================

if __name__ == "__main__":
    manager = ACLManager()

    # Create a document
    doc = manager.create_resource("doc1", "Project Plan.docx", owner_id="alice")

    # Share with bob (read + write)
    manager.share("doc1", "alice", "user", "bob",
                  ACLPermission.READ | ACLPermission.WRITE)

    # Share with engineering group (read only)
    manager.share("doc1", "alice", "group", "engineering", ACLPermission.READ)

    # Check access
    print("=== ACL Checks ===")
    print(f"Alice (owner) read:  {manager.check('alice', set(), 'doc1', ACLPermission.READ)}")
    print(f"Alice (owner) delete: {manager.check('alice', set(), 'doc1', ACLPermission.DELETE)}")
    print(f"Bob read:   {manager.check('bob', set(), 'doc1', ACLPermission.READ)}")
    print(f"Bob write:  {manager.check('bob', set(), 'doc1', ACLPermission.WRITE)}")
    print(f"Bob delete: {manager.check('bob', set(), 'doc1', ACLPermission.DELETE)}")

    # Carol is in engineering group
    print(f"Carol (eng) read:  {manager.check('carol', {'engineering'}, 'doc1', ACLPermission.READ)}")
    print(f"Carol (eng) write: {manager.check('carol', {'engineering'}, 'doc1', ACLPermission.WRITE)}")

    # Dan has no access
    print(f"Dan read: {manager.check('dan', set(), 'doc1', ACLPermission.READ)}")
```

---

## 5. Policy Engines: OPA (Open Policy Agent)

### 5.1 What is OPA?

Open Policy Agent (OPA) is a general-purpose policy engine that decouples policy decision-making from application logic. Policies are written in **Rego**, a declarative query language.

```
┌─────────────────────────────────────────────────────────────────┐
│              OPA Architecture                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐       ┌──────────────┐                       │
│  │ Application  │──────▶│     OPA      │                       │
│  │   (Python,   │ Query │   ┌──────┐   │                       │
│  │    Java,     │───────│──▶│ Rego │   │                       │
│  │    Go...)    │       │   │Policy│   │                       │
│  │              │◀──────│   └──────┘   │                       │
│  │              │Decision│   ┌──────┐   │                       │
│  └──────────────┘       │   │ Data │   │                       │
│                         │   │(JSON)│   │                       │
│                         │   └──────┘   │                       │
│                         └──────────────┘                       │
│                                                                  │
│  Deployment Options:                                             │
│  1. Library (embedded in app)                                    │
│  2. Sidecar (daemon alongside app)                              │
│  3. Standalone service (centralized)                            │
│                                                                  │
│  Key Benefits:                                                   │
│  - Policy as Code (version controlled, tested, audited)         │
│  - Language-agnostic (any app can query OPA)                    │
│  - Separation of concerns (dev writes logic, security writes    │
│    policy)                                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Rego Policy Language

```rego
# authz.rego - Example OPA policy for API authorization

package authz

import rego.v1

# Default deny all requests
default allow := false

# Admin users can do anything
allow if {
    input.user.roles[_] == "admin"
}

# Users can read their own profile
allow if {
    input.action == "read"
    input.resource.type == "profile"
    input.resource.owner == input.user.id
}

# Editors can create and update posts
allow if {
    input.user.roles[_] == "editor"
    input.action in {"create", "update"}
    input.resource.type == "post"
}

# Users can delete their own posts
allow if {
    input.action == "delete"
    input.resource.type == "post"
    input.resource.owner == input.user.id
}

# Time-based access: deny outside business hours for sensitive data
deny if {
    input.resource.sensitivity == "high"
    not is_business_hours
}

is_business_hours if {
    hour := time.clock(time.now_ns())[0]
    hour >= 8
    hour < 18
}

# Final decision (deny overrides allow)
decision := "allow" if {
    allow
    not deny
}

decision := "deny" if {
    not allow
}

decision := "deny" if {
    deny
}
```

### 5.3 Integrating OPA with Python

```python
"""
opa_integration.py - Integrating OPA with a Python/Flask application
pip install requests
"""
import requests
import json
from flask import Flask, request, jsonify, g
from functools import wraps

app = Flask(__name__)

# OPA runs as a sidecar or standalone service
OPA_URL = "http://localhost:8181/v1/data/authz/decision"


def check_opa_policy(user: dict, action: str, resource: dict) -> bool:
    """Query OPA for an authorization decision."""
    opa_input = {
        "input": {
            "user": user,
            "action": action,
            "resource": resource,
        }
    }

    try:
        response = requests.post(
            OPA_URL,
            json=opa_input,
            timeout=1,  # Fail fast
        )
        response.raise_for_status()
        result = response.json()
        return result.get("result") == "allow"
    except requests.RequestException as e:
        app.logger.error(f"OPA query failed: {e}")
        return False  # Fail closed (deny on error)


def require_opa(action: str, resource_fn=None):
    """
    Decorator that checks OPA policy before allowing access.
    resource_fn: function that builds resource dict from request context
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = g.get('current_user')
            if not user:
                return jsonify({"error": "Not authenticated"}), 401

            # Build resource context
            if resource_fn:
                resource = resource_fn(request, **kwargs)
            else:
                resource = {"type": "unknown"}

            # Query OPA
            if not check_opa_policy(user, action, resource):
                return jsonify({"error": "Access denied by policy"}), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


# Example resource builders
def post_resource(req, post_id=None, **kwargs):
    """Build resource dict for post endpoints."""
    if post_id:
        post = get_post(post_id)  # Your DB lookup
        return {
            "type": "post",
            "id": post_id,
            "owner": post.get("author_id") if post else None,
            "sensitivity": "normal",
        }
    return {"type": "post"}


# Protected routes using OPA
@app.route('/api/posts', methods=['POST'])
@require_opa("create", lambda req, **kw: {"type": "post"})
def create_post():
    return jsonify({"message": "Post created"}), 201


@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
@require_opa("delete", post_resource)
def delete_post(post_id):
    return jsonify({"message": "Post deleted"})
```

---

## 6. JWT Claims for Authorization

### 6.1 Standard and Custom Claims

```
┌─────────────────────────────────────────────────────────────────┐
│              JWT Claims for Authorization                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Standard (Registered) Claims:                                   │
│  ┌────────────┬──────────────────────────────────────────┐     │
│  │ Claim      │ Purpose                                   │     │
│  ├────────────┼──────────────────────────────────────────┤     │
│  │ sub        │ Subject (user ID)                         │     │
│  │ iss        │ Issuer (who created the token)            │     │
│  │ aud        │ Audience (who should accept the token)    │     │
│  │ exp        │ Expiration time                           │     │
│  │ iat        │ Issued at time                            │     │
│  │ nbf        │ Not before time                           │     │
│  │ jti        │ JWT ID (unique identifier)                │     │
│  └────────────┴──────────────────────────────────────────┘     │
│                                                                  │
│  Custom Authorization Claims:                                    │
│  {                                                               │
│    "sub": "user_123",                                            │
│    "roles": ["editor", "reviewer"],                              │
│    "permissions": ["post:create", "post:update", "post:read"],   │
│    "org_id": "org_456",                                          │
│    "department": "engineering",                                  │
│    "tier": "premium",                                            │
│    "scope": "read write"                                         │
│  }                                                               │
│                                                                  │
│  WARNING: Keep JWT payload small!                                │
│  - JWT is sent with every request (in Authorization header)      │
│  - Large payloads increase bandwidth and latency                 │
│  - Put minimal auth info in JWT, fetch details from DB/cache     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Claim-Based Authorization Middleware

```python
"""
jwt_authz.py - JWT claim-based authorization
"""
from flask import Flask, request, jsonify, g
from functools import wraps
import jwt

app = Flask(__name__)
JWT_SECRET = "your-256-bit-secret"  # Use env var in production


def decode_jwt(token: str) -> dict:
    """Decode and validate JWT."""
    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=["HS256"],
        options={"require": ["exp", "sub", "iss"]}
    )


def require_claims(**required_claims):
    """
    Decorator that checks JWT claims match requirements.

    Usage:
        @require_claims(roles=["admin"], tier="premium")
        @require_claims(permissions=["post:create"])
        @require_claims(org_id=lambda v: v == g.resource_org_id)
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Extract token
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({"error": "Missing token"}), 401

            try:
                token = auth_header.split(' ', 1)[1]
                claims = decode_jwt(token)
                g.claims = claims
            except jwt.InvalidTokenError as e:
                return jsonify({"error": str(e)}), 401

            # Check each required claim
            for claim_name, expected in required_claims.items():
                actual = claims.get(claim_name)

                # Callable check (custom validation function)
                if callable(expected):
                    if not expected(actual):
                        return jsonify({
                            "error": f"Claim '{claim_name}' validation failed"
                        }), 403

                # List check (user must have at least one matching value)
                elif isinstance(expected, list):
                    user_values = actual if isinstance(actual, list) else [actual]
                    if not set(expected).intersection(set(user_values)):
                        return jsonify({
                            "error": f"Required claim '{claim_name}': {expected}"
                        }), 403

                # Direct value check
                else:
                    if actual != expected:
                        return jsonify({
                            "error": f"Claim '{claim_name}' mismatch"
                        }), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


# Route examples
@app.route('/api/admin/dashboard')
@require_claims(roles=["admin"])
def admin_dashboard():
    return jsonify({"message": "Welcome, admin"})


@app.route('/api/premium/features')
@require_claims(tier="premium")
def premium_features():
    return jsonify({"features": ["advanced_analytics", "api_access"]})


@app.route('/api/posts', methods=['POST'])
@require_claims(permissions=["post:create"])
def create_post():
    return jsonify({"message": "Post created"})


@app.route('/api/org/<org_id>/data')
@require_claims(org_id=lambda v: v == request.view_args.get('org_id'))
def org_data(org_id):
    """Only users belonging to this org can access."""
    return jsonify({"org_id": org_id, "data": []})
```

---

## 7. OAuth 2.0 Scopes

### 7.1 Understanding Scopes

OAuth 2.0 scopes limit what an access token can do. They represent the **permissions granted by the user** to the client application.

```
┌─────────────────────────────────────────────────────────────────┐
│                  OAuth 2.0 Scopes                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Consent Screen:                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  "MyApp" wants to access your account:          │            │
│  │                                                  │            │
│  │  [x] Read your profile (scope: profile:read)    │            │
│  │  [x] Read your emails (scope: email:read)       │            │
│  │  [ ] Send emails on your behalf (email:send)    │            │
│  │  [ ] Delete your data (data:delete)             │            │
│  │                                                  │            │
│  │         [Allow]    [Deny]                        │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  Resulting token:                                                │
│  {                                                               │
│    "scope": "profile:read email:read",                           │
│    "client_id": "myapp",                                         │
│    "sub": "user_123"                                             │
│  }                                                               │
│                                                                  │
│  Common Scope Patterns:                                          │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Pattern          │ Example                       │           │
│  ├──────────────────┼───────────────────────────────┤           │
│  │ resource:action   │ posts:read, posts:write       │           │
│  │ resource.action   │ user.email, user.profile      │           │
│  │ hierarchical      │ admin (implies all)            │           │
│  │ OIDC standard     │ openid, profile, email         │           │
│  │ GitHub style      │ repo, user, gist              │           │
│  │ Google style      │ drive.readonly, calendar       │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Scope Enforcement

```python
"""
scope_enforcement.py - OAuth scope checking for APIs
"""
from flask import Flask, request, jsonify, g
from functools import wraps


app = Flask(__name__)


def require_scope(*required_scopes):
    """
    Decorator to enforce OAuth 2.0 scopes.
    Token must contain ALL required scopes.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Token scopes are typically space-separated
            token_scopes = set(g.claims.get('scope', '').split())
            required = set(required_scopes)

            missing = required - token_scopes
            if missing:
                return jsonify({
                    "error": "insufficient_scope",
                    "error_description": f"Missing scopes: {', '.join(missing)}",
                    "required_scopes": list(required),
                    "granted_scopes": list(token_scopes),
                }), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


@app.route('/api/user/profile')
@require_scope('profile:read')
def get_profile():
    return jsonify({"name": "Alice", "email": "alice@example.com"})


@app.route('/api/user/profile', methods=['PUT'])
@require_scope('profile:read', 'profile:write')
def update_profile():
    return jsonify({"message": "Profile updated"})


@app.route('/api/user/data/export')
@require_scope('data:export')
def export_data():
    return jsonify({"download_url": "https://..."})
```

---

## 8. Resource-Level Permissions

### 8.1 Beyond Role-Based: Resource Ownership

Many applications need to check not just "does the user have the right role" but "does the user have access to this specific resource?"

```
┌─────────────────────────────────────────────────────────────────┐
│          Resource-Level Permission Checks                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: Role Check                                             │
│  "Is the user an editor?" → Yes/No                              │
│                                                                  │
│  Level 2: Resource Ownership                                     │
│  "Is the user the owner of this post?" → Yes/No                │
│                                                                  │
│  Level 3: Shared Access                                          │
│  "Has the post been shared with this user?" → Yes/No            │
│                                                                  │
│  Level 4: Organizational Scope                                   │
│  "Does the user belong to the same org?" → Yes/No              │
│                                                                  │
│  Combined:                                                       │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐          │
│  │ Role check │───▶│ Resource   │───▶│ Additional   │          │
│  │ (editor?)  │    │ ownership  │    │ constraints  │          │
│  └────────────┘    │ (owner?)   │    │ (same org?)  │          │
│                    └────────────┘    └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Implementation: Multi-Layer Authorization

```python
"""
resource_auth.py - Multi-layer authorization for resources
"""
from flask import Flask, request, jsonify, g, abort
from functools import wraps
from typing import Optional, Callable

app = Flask(__name__)


# ==============================================================
# Resource Authorization Framework
# ==============================================================

class ResourceAuthorizer:
    """
    Multi-layer authorization for resources.
    Checks are performed in order: role → ownership → sharing → custom.
    """

    def __init__(self):
        self.custom_checks = {}

    def register_check(self, resource_type: str,
                       check_fn: Callable) -> None:
        """Register a custom authorization check for a resource type."""
        if resource_type not in self.custom_checks:
            self.custom_checks[resource_type] = []
        self.custom_checks[resource_type].append(check_fn)

    def authorize(self, user: dict, resource: dict,
                  action: str) -> bool:
        """
        Check if user can perform action on resource.
        Returns True if authorized.
        """
        resource_type = resource.get('type')

        # Layer 1: Super admin bypass
        if 'super_admin' in user.get('roles', []):
            return True

        # Layer 2: Resource ownership
        if resource.get('owner_id') == user.get('id'):
            return True

        # Layer 3: Role-based for the resource type
        role_permissions = self._get_role_permissions(
            user.get('roles', []), resource_type
        )
        if action in role_permissions:
            return True

        # Layer 4: Shared access
        shares = resource.get('shares', [])
        for share in shares:
            if share.get('user_id') == user.get('id'):
                if action in share.get('permissions', []):
                    return True
            if share.get('group_id') in user.get('groups', []):
                if action in share.get('permissions', []):
                    return True

        # Layer 5: Custom checks
        for check_fn in self.custom_checks.get(resource_type, []):
            if check_fn(user, resource, action):
                return True

        return False

    def _get_role_permissions(self, roles: list,
                               resource_type: str) -> set:
        """Get permissions for roles on a resource type."""
        # In production, this comes from a database
        role_perms = {
            'admin': {
                'post': {'read', 'create', 'update', 'delete', 'publish'},
                'comment': {'read', 'create', 'update', 'delete'},
                'user': {'read', 'create', 'update', 'delete'},
            },
            'editor': {
                'post': {'read', 'create', 'update'},
                'comment': {'read', 'create', 'update', 'delete'},
            },
            'viewer': {
                'post': {'read'},
                'comment': {'read', 'create'},
            },
        }

        perms = set()
        for role in roles:
            if role in role_perms and resource_type in role_perms[role]:
                perms |= role_perms[role][resource_type]
        return perms


# Global authorizer instance
authorizer = ResourceAuthorizer()


# ==============================================================
# Authorization Decorator
# ==============================================================

def authorize_resource(action: str, resource_loader: Callable):
    """
    Decorator for resource-level authorization.
    resource_loader: function that returns the resource dict.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = g.get('current_user')
            if not user:
                return jsonify({"error": "Not authenticated"}), 401

            resource = resource_loader(**kwargs)
            if not resource:
                return jsonify({"error": "Resource not found"}), 404

            if not authorizer.authorize(user, resource, action):
                return jsonify({
                    "error": "Forbidden",
                    "detail": f"You don't have '{action}' access to this resource"
                }), 403

            g.resource = resource
            return f(*args, **kwargs)
        return decorated
    return decorator


# Resource loaders
def load_post(post_id: int = None, **kwargs):
    """Load a post from the database."""
    # Simulated DB lookup
    posts = {
        1: {"type": "post", "id": 1, "owner_id": "alice", "org_id": "org1",
            "shares": [{"user_id": "bob", "permissions": ["read"]}]},
        2: {"type": "post", "id": 2, "owner_id": "bob", "org_id": "org1",
            "shares": []},
    }
    return posts.get(post_id)


# Routes
@app.route('/api/posts/<int:post_id>', methods=['GET'])
@authorize_resource('read', load_post)
def get_post(post_id):
    return jsonify(g.resource)


@app.route('/api/posts/<int:post_id>', methods=['PUT'])
@authorize_resource('update', load_post)
def update_post(post_id):
    return jsonify({"message": "Updated"})


@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
@authorize_resource('delete', load_post)
def delete_post(post_id):
    return jsonify({"message": "Deleted"})
```

---

## 9. Common Authorization Vulnerabilities

### 9.1 IDOR (Insecure Direct Object Reference)

IDOR occurs when an application exposes internal object references (like database IDs) without verifying that the requesting user is authorized to access them.

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDOR Vulnerability                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Vulnerable:                                                     │
│  GET /api/invoices/12345                                         │
│  → Returns invoice 12345 (no ownership check!)                   │
│                                                                  │
│  Attacker changes ID:                                            │
│  GET /api/invoices/12346                                         │
│  → Returns someone else's invoice! 🚨                            │
│                                                                  │
│  GET /api/users/100/profile → Attacker's profile                │
│  GET /api/users/101/profile → Another user's profile!           │
│  GET /api/users/102/profile → Yet another!                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
idor_example.py - IDOR vulnerability and fix
"""

# ============================================================
# VULNERABLE: No authorization check on resource access
# ============================================================

@app.route('/api/invoices/<int:invoice_id>')
@require_auth  # Only checks authentication, NOT authorization!
def get_invoice_vulnerable(invoice_id):
    # Any authenticated user can access ANY invoice by ID
    invoice = db.get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Not found"}), 404
    return jsonify(invoice)  # IDOR: No ownership check!


# ============================================================
# FIXED: Verify resource ownership
# ============================================================

@app.route('/api/invoices/<int:invoice_id>')
@require_auth
def get_invoice_secure(invoice_id):
    invoice = db.get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Not found"}), 404

    # Check: Does this invoice belong to the current user?
    if invoice['user_id'] != g.current_user['id']:
        # Return 404 (not 403) to avoid information disclosure
        # A 403 tells attacker "the resource exists but you can't access it"
        return jsonify({"error": "Not found"}), 404

    return jsonify(invoice)


# ============================================================
# BETTER: Use indirect references
# ============================================================

@app.route('/api/my/invoices')
@require_auth
def list_my_invoices():
    """Only return the current user's invoices."""
    invoices = db.get_invoices_for_user(g.current_user['id'])
    return jsonify({"invoices": invoices})
```

### 9.2 Privilege Escalation

```
┌─────────────────────────────────────────────────────────────────┐
│              Privilege Escalation Types                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Vertical Escalation (gaining higher privileges):                │
│  ┌──────────┐         ┌──────────┐                              │
│  │ Regular  │ ──────▶ │  Admin   │                              │
│  │  User    │ attacks │   User   │                              │
│  └──────────┘         └──────────┘                              │
│                                                                  │
│  Example: Modifying JWT role claim                               │
│  Original:  {"sub": "user1", "role": "user"}                    │
│  Tampered:  {"sub": "user1", "role": "admin"}                   │
│                                                                  │
│  Horizontal Escalation (accessing other users' data):            │
│  ┌──────────┐         ┌──────────┐                              │
│  │  User A  │ ──────▶ │  User B  │                              │
│  │  (self)  │ accesses│  (other) │                              │
│  └──────────┘         └──────────┘                              │
│                                                                  │
│  Example: Changing user_id parameter                             │
│  Own data:     GET /api/users/100/settings                      │
│  Other's data: GET /api/users/101/settings                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
privilege_escalation.py - Privilege escalation vulnerabilities and fixes
"""

# ============================================================
# VULNERABLE: Role in client-controlled input
# ============================================================

@app.route('/api/register', methods=['POST'])
def register_vulnerable():
    data = request.json
    user = {
        'username': data['username'],
        'email': data['email'],
        'role': data.get('role', 'user'),  # Attacker sends role=admin!
    }
    db.create_user(user)
    return jsonify(user), 201

# Attack:
# POST /api/register
# {"username": "attacker", "email": "a@b.com", "role": "admin"}


# ============================================================
# FIXED: Never trust client-supplied role
# ============================================================

@app.route('/api/register', methods=['POST'])
def register_secure():
    data = request.json
    user = {
        'username': data['username'],
        'email': data['email'],
        'role': 'user',  # Always set server-side!
    }
    db.create_user(user)
    return jsonify(user), 201


# ============================================================
# VULNERABLE: Mass assignment (updating fields not intended)
# ============================================================

@app.route('/api/users/<int:user_id>', methods=['PATCH'])
@require_auth
def update_user_vulnerable(user_id):
    data = request.json
    # Blindly update all provided fields
    db.update_user(user_id, **data)  # Attacker sends {"role": "admin"}!
    return jsonify({"message": "Updated"})


# ============================================================
# FIXED: Whitelist allowed fields
# ============================================================

ALLOWED_UPDATE_FIELDS = {'username', 'email', 'bio', 'avatar_url'}

@app.route('/api/users/<int:user_id>', methods=['PATCH'])
@require_auth
def update_user_secure(user_id):
    # Ensure user can only update their own profile
    if user_id != g.current_user['id']:
        return jsonify({"error": "Forbidden"}), 403

    data = request.json

    # Only allow whitelisted fields
    safe_data = {
        k: v for k, v in data.items()
        if k in ALLOWED_UPDATE_FIELDS
    }

    if not safe_data:
        return jsonify({"error": "No valid fields to update"}), 400

    db.update_user(user_id, **safe_data)
    return jsonify({"message": "Updated"})


# ============================================================
# VULNERABLE: Broken function-level access control
# ============================================================

# Admin endpoint with no authorization check
@app.route('/api/admin/delete-user/<int:user_id>', methods=['DELETE'])
@require_auth  # Only checks if user is logged in, not if they're admin!
def delete_user_vulnerable(user_id):
    db.delete_user(user_id)
    return jsonify({"message": "User deleted"})


# ============================================================
# FIXED: Proper role check on admin endpoints
# ============================================================

@app.route('/api/admin/delete-user/<int:user_id>', methods=['DELETE'])
@require_role('admin')  # Checks both auth AND admin role
def delete_user_secure(user_id):
    # Additional safety: prevent deleting self or other admins
    target = db.get_user(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    if target['id'] == g.current_user['id']:
        return jsonify({"error": "Cannot delete yourself"}), 400

    if 'admin' in target.get('roles', []):
        return jsonify({"error": "Cannot delete other admins via API"}), 403

    db.delete_user(user_id)
    return jsonify({"message": "User deleted"})
```

### 9.3 Authorization Vulnerability Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│          Authorization Security Checklist                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [ ] Every endpoint has explicit authorization checks            │
│  [ ] Authorization is enforced server-side (never client-only)   │
│  [ ] Direct object references are validated for ownership        │
│  [ ] Admin functions require admin role verification             │
│  [ ] Role/permission changes require admin authorization         │
│  [ ] User input never controls role or permission assignments    │
│  [ ] API responses don't expose other users' data                │
│  [ ] Failed authorization returns 403 (or 404 for IDOR)         │
│  [ ] Authorization logic is centralized (not duplicated)         │
│  [ ] Mass assignment is prevented (field whitelisting)           │
│  [ ] Horizontal access is checked (user A can't access B's data)│
│  [ ] Authorization decisions are logged for audit                │
│  [ ] Default-deny policy (deny unless explicitly allowed)        │
│  [ ] Token-based auth checks scope/claims                       │
│  [ ] Multi-tenant isolation is enforced at data layer           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Exercises

### Exercise 1: Implement RBAC System

Build a complete RBAC system with role hierarchy:

```python
"""
Exercise: Implement a complete RBAC system for a blogging platform.

Requirements:
- Roles: super_admin, admin, moderator, author, reader
- Role hierarchy: super_admin > admin > moderator > author > reader
- Permissions: user:*, post:*, comment:*, settings:*
- Users can have multiple roles
- Implement role assignment with authorization (only admin+ can assign)
"""

class BlogRBAC:
    def create_role(self, name: str, permissions: set,
                    parent: str = None) -> bool:
        """Create a new role with optional parent."""
        pass

    def assign_role(self, admin_id: int, target_user_id: int,
                    role: str) -> bool:
        """Assign role (only admins can do this)."""
        pass

    def check_permission(self, user_id: int,
                         permission: str) -> bool:
        """Check if user has permission (including inherited)."""
        pass

    def get_accessible_resources(self, user_id: int,
                                  resource_type: str) -> list:
        """Get all resources a user can access."""
        pass
```

### Exercise 2: Build ABAC Policy Engine

Create a healthcare ABAC system:

```python
"""
Exercise: Build an ABAC engine for a hospital system.

Policies to implement:
1. Doctors can read patient records in their department
2. Nurses can read records during their shift only
3. Emergency doctors can override department restrictions
4. No one can access records from non-hospital IP addresses
5. Psychiatry records require additional clearance level
6. Research access requires IRB approval attribute
"""

class HospitalABAC:
    def add_policy(self, name: str, condition: callable,
                   effect: str) -> None:
        pass

    def evaluate(self, subject: dict, resource: dict,
                 action: dict, context: dict) -> str:
        pass
```

### Exercise 3: Fix Authorization Vulnerabilities

Find and fix all authorization issues in this code:

```python
"""
Exercise: This API has at least 7 authorization vulnerabilities.
Find and fix ALL of them.
"""

@app.route('/api/users', methods=['GET'])
def list_all_users():
    # Issue 1: ???
    return jsonify(db.get_all_users())

@app.route('/api/users/<int:id>/password', methods=['PUT'])
@require_auth
def change_password(id):
    # Issue 2: ???
    new_password = request.json['password']
    db.update_password(id, new_password)
    return jsonify({"status": "updated"})

@app.route('/api/posts/<int:id>', methods=['DELETE'])
@require_auth
def delete_post(id):
    post = db.get_post(id)
    # Issue 3: ???
    db.delete_post(id)
    return jsonify({"status": "deleted"})

@app.route('/api/admin/promote', methods=['POST'])
@require_auth
def promote_user():
    # Issue 4: ???
    user_id = request.json['user_id']
    role = request.json['role']  # Issue 5: ???
    db.set_role(user_id, role)
    return jsonify({"status": "promoted"})

@app.route('/api/files/<path:filepath>')
@require_auth
def download_file(filepath):
    # Issue 6: ???
    return send_file(f'/uploads/{filepath}')

@app.route('/api/settings', methods=['PUT'])
@require_auth
def update_settings():
    # Issue 7: ???
    settings = request.json
    db.update_all_settings(settings)
    return jsonify({"status": "updated"})
```

### Exercise 4: Multi-Tenant Authorization

Design and implement a multi-tenant authorization system:

```python
"""
Exercise: Build authorization for a multi-tenant SaaS application.

Requirements:
- Each tenant (organization) has isolated data
- Users belong to exactly one organization
- Roles are per-organization (admin in org A != admin in org B)
- Cross-tenant access is never allowed
- Super-admins (platform level) can access any tenant
"""

class MultiTenantAuth:
    def create_org(self, org_id: str, owner_id: str) -> dict:
        pass

    def check_tenant_access(self, user_id: str, org_id: str,
                            resource_id: str, action: str) -> bool:
        pass

    def ensure_tenant_isolation(self, query: str,
                                 user_org_id: str) -> str:
        """Add tenant filter to database queries."""
        pass
```

### Exercise 5: OPA Policy Writing

Write Rego policies for these scenarios:

```rego
# Exercise: Write Rego policies for:
#
# 1. Users can only access resources in their department
# 2. Managers can access resources in their department and
#    departments they manage
# 3. Financial reports require "finance" role AND "senior" level
# 4. API rate limiting: users with "basic" tier limited to 100 req/hour
# 5. Data classification: "top_secret" resources require MFA
#    authentication within the last 30 minutes

package exercise

import rego.v1

# Write your policies here:

default allow := false

# Policy 1: Department access
# ...

# Policy 2: Manager cross-department access
# ...

# Policy 3: Financial reports
# ...

# Policy 4: Rate limiting
# ...

# Policy 5: MFA requirement for classified data
# ...
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│          Authorization and Access Control Summary                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Models:                                                         │
│  - RBAC: Roles → Permissions. Simple, widely used.              │
│  - ABAC: Attributes → Policy → Decision. Flexible, context-aware│
│  - ACL: Per-resource permission lists. Fine-grained.            │
│  - Combine models as needed (RBAC + resource ownership is common)│
│                                                                  │
│  Key Principles:                                                 │
│  - Least Privilege: Grant minimum necessary permissions          │
│  - Default Deny: Block everything not explicitly allowed         │
│  - Separation of Duties: No single role can do everything        │
│  - Defense in Depth: Check at multiple layers                    │
│  - Centralize: Authorization logic in one place                  │
│                                                                  │
│  Implementation:                                                 │
│  - Server-side enforcement (never trust the client)              │
│  - JWT claims for stateless API authorization                    │
│  - OAuth scopes for third-party access delegation               │
│  - Policy engines (OPA) for complex or externalized policies    │
│  - Decorators/middleware for DRY authorization in Flask          │
│                                                                  │
│  Common Vulnerabilities:                                         │
│  - IDOR: Always validate resource ownership                     │
│  - Privilege escalation: Never trust client-supplied roles       │
│  - Mass assignment: Whitelist updateable fields                  │
│  - Missing function-level checks: Every endpoint needs auth     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [05. Authentication Systems](05_Authentication.md) | **Next**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md)
