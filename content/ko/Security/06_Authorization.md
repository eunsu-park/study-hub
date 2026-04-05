# 06. 인가 및 접근 제어

**이전**: [05. 인증 시스템](05_Authentication.md) | **다음**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md)

---

인가(Authorization)는 인증된 사용자가 무엇을 할 수 있는지를 결정합니다. 인증이 "당신은 누구입니까?"라는 질문에 답한다면, 인가는 "당신은 무엇을 할 수 있습니까?"라는 질문에 답합니다. 강력한 인가 시스템은 최소 권한의 원칙을 강제하여, 사용자와 서비스가 필요한 최소한의 권한만 가지도록 보장합니다. 이 레슨에서는 주요 접근 제어 모델(RBAC, ABAC, ACL), 정책 엔진, JWT 및 OAuth 스코프를 사용한 토큰 기반 인가, Python/Flask에서의 실용적인 구현 패턴, 그리고 일반적인 인가 취약점을 다룹니다.

## 학습 목표

- 인증과 인가의 차이점 구분
- 역할 기반 접근 제어(RBAC) 시스템 구현
- 속성 기반 접근 제어(ABAC)와 사용 시점 이해
- 리소스 수준 권한을 위한 접근 제어 목록(ACL) 작업
- 외부화된 인가를 위한 OPA(Open Policy Agent)와 같은 정책 엔진 사용
- API 인가를 위한 JWT 클레임 및 OAuth 2.0 스코프 활용
- Flask에서 인가 미들웨어 및 데코레이터 구축
- 일반적인 인가 취약점(IDOR, 권한 상승) 식별 및 방지

---

## 1. 인증 vs 인가

```
┌─────────────────────────────────────────────────────────────────┐
│          인증 vs 인가                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  인증 (AuthN)                   인가 (AuthZ)                    │
│  ─────────────────────         ──────────────────────            │
│  "당신은 누구입니까?"          "무엇을 할 수 있습니까?"          │
│                                                                  │
│  신원 확인                      권한 확인                        │
│  먼저 발생                      인증 후 발생                     │
│  401 Unauthorized              403 Forbidden                     │
│  (인증되지 않음)               (인증됐지만 허용되지 않음)        │
│                                                                  │
│                                                                  │
│  요청 흐름:                                                      │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐           │
│  │  요청    │───▶│     인증     │───▶│    인가     │           │
│  │          │    │ "이 사람은   │    │ "이 작업을  │           │
│  └──────────┘    │ 누구인가?"   │    │ 할 수 있나?"│            │
│                  └──────┬───────┘    └──────┬──────┘           │
│                         │                   │                    │
│                    ┌────┴───┐          ┌────┴───┐               │
│                    │        │          │        │                │
│                  유효   무효         허용     거부                │
│                    │        │          │        │                │
│                    │    401 에러       │    403 에러             │
│                    │                   │                          │
│                    └───────┬───────────┘                         │
│                            │                                     │
│                            ▼                                     │
│                     ┌──────────┐                                │
│                     │ 리소스   │                                 │
│                     │  접근    │                                 │
│                     └──────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 역할 기반 접근 제어 (RBAC)

### 2.1 RBAC 개념

RBAC는 권한을 **역할(roles)**에 할당하고, 사용자는 역할에 할당됩니다. 개별 사용자 권한이 아닌 역할-권한 매핑을 관리하기 때문에 관리가 간단합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RBAC 모델                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  사용자              역할              권한                      │
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
│  계층 구조 (선택 사항):                                         │
│  Admin ──▶ Editor ──▶ Viewer                                    │
│  (Admin은 모든 Editor 및 Viewer 권한을 상속)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Python에서 RBAC 구현

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

### 2.3 Flask RBAC 미들웨어

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

## 3. 속성 기반 접근 제어 (ABAC)

### 3.1 ABAC 개념

ABAC는 주체(사용자), 리소스, 작업 및 환경의 **속성(attributes)**을 기반으로 접근 결정을 내립니다. RBAC보다 더 유연하지만 더 복잡합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABAC 모델                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  접근 결정 =                                                     │
│    f(주체 속성, 리소스 속성,                                     │
│      작업 속성, 환경 속성)                                       │
│                                                                  │
│  주체 속성:              리소스 속성:                            │
│  ├── role: "doctor"      ├── type: "medical_record"              │
│  ├── department: "ER"    ├── department: "ER"                    │
│  ├── clearance: "L3"     ├── sensitivity: "L2"                   │
│  └── certification: true └── owner: "patient_123"                │
│                                                                  │
│  작업 속성:              환경 속성:                              │
│  ├── type: "read"        ├── time: "14:30 UTC"                   │
│  └── purpose: "treatment"├── ip_address: "10.0.1.50"            │
│                          ├── location: "hospital_network"        │
│                          └── device_trust: "managed"             │
│                                                                  │
│  정책 예시:                                                      │
│  "응급실 부서의 의사는 병원 네트워크에서 관리되는 기기를 통해    │
│   근무 시간 중에 응급실 부서의 의료 기록을 읽을 수 있다."        │
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

### 3.2 RBAC vs ABAC 사용 시점

| 기준 | RBAC | ABAC |
|----------|------|------|
| 역할 수 | 소수, 명확히 정의됨 | 많거나 동적 |
| 접근 결정 | 역할 멤버십 | 다중 속성 |
| 복잡성 | 구현이 간단 | 복잡하지만 유연함 |
| "역할 폭발" | 위험 (역할이 너무 많음) | 문제 없음 |
| 컨텍스트 인식 | 없음 (정적 역할) | 있음 (시간, 위치 등) |
| 규정 준수 | 기본 요구사항 | 규제 요구사항 |
| 적합한 경우 | 대부분의 웹 앱, API | 의료, 금융, 정부 |
| 성능 | 빠름 (역할 조회) | 느림 (정책 평가) |

### 3.3 ABAC 구현

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

## 4. 접근 제어 목록 (ACL)

### 4.1 ACL 개념

ACL은 **개별 리소스 수준**에서 권한을 정의합니다. 각 리소스에는 어떤 주체(사용자, 그룹)가 어떤 작업을 수행할 수 있는지 지정하는 항목 목록이 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACL 모델                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  리소스: "Project Plan.docx"                                     │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ ACL 항목           │ 권한                             │      │
│  ├─────────────────────┼─────────────────────────────────┤      │
│  │ alice (소유자)      │ read, write, delete, share      │      │
│  │ bob                 │ read, write                     │      │
│  │ carol               │ read                            │      │
│  │ engineering (그룹)  │ read, comment                   │      │
│  │ * (모든 사람)       │ (접근 권한 없음)                │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  리소스: "Company Financials.xlsx"                               │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ ACL 항목           │ 권한                             │      │
│  ├─────────────────────┼─────────────────────────────────┤      │
│  │ alice (소유자)      │ read, write, delete, share      │      │
│  │ finance (그룹)      │ read, write                     │      │
│  │ ceo                 │ read                            │      │
│  │ * (모든 사람)       │ (접근 권한 없음)                │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  Unix 파일 권한 (단순화된 ACL):                                 │
│  -rwxr-xr--  owner  group  file.txt                             │
│   │││ │││ │││                                                    │
│   │││ │││ └┴┴── Others: 읽기만                                 │
│   │││ └┴┴────── Group: 읽기 + 실행                             │
│   └┴┴────────── Owner: 읽기 + 쓰기 + 실행                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 ACL 구현

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

## 5. 정책 엔진: OPA (Open Policy Agent)

### 5.1 OPA란?

Open Policy Agent (OPA)는 애플리케이션 로직에서 정책 의사 결정을 분리하는 범용 정책 엔진입니다. 정책은 선언적 쿼리 언어인 **Rego**로 작성됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              OPA 아키텍처                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐       ┌──────────────┐                       │
│  │ 애플리케이션  │──────▶│     OPA      │                       │
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
│  배포 옵션:                                                      │
│  1. 라이브러리 (앱에 내장)                                       │
│  2. 사이드카 (앱과 함께 실행되는 데몬)                          │
│  3. 독립 실행형 서비스 (중앙 집중식)                            │
│                                                                  │
│  주요 이점:                                                      │
│  - 코드로서의 정책 (버전 관리, 테스트, 감사)                   │
│  - 언어 독립적 (모든 앱이 OPA를 쿼리 가능)                     │
│  - 관심사 분리 (개발자는 로직 작성, 보안팀은 정책 작성)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Rego 정책 언어

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

### 5.3 Python과 OPA 통합

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

## 6. 인가를 위한 JWT 클레임

### 6.1 표준 및 사용자 정의 클레임

```
┌─────────────────────────────────────────────────────────────────┐
│              인가를 위한 JWT 클레임                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  표준 (등록된) 클레임:                                          │
│  ┌────────────┬──────────────────────────────────────────┐     │
│  │ 클레임     │ 목적                                      │     │
│  ├────────────┼──────────────────────────────────────────┤     │
│  │ sub        │ 주체 (사용자 ID)                          │     │
│  │ iss        │ 발급자 (토큰을 생성한 주체)              │     │
│  │ aud        │ 대상자 (토큰을 수락해야 하는 주체)        │     │
│  │ exp        │ 만료 시간                                 │     │
│  │ iat        │ 발급 시간                                 │     │
│  │ nbf        │ 이전 시간                                 │     │
│  │ jti        │ JWT ID (고유 식별자)                      │     │
│  └────────────┴──────────────────────────────────────────┘     │
│                                                                  │
│  사용자 정의 인가 클레임:                                       │
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
│  경고: JWT 페이로드를 작게 유지하세요!                          │
│  - JWT는 모든 요청과 함께 전송됩니다 (Authorization 헤더)       │
│  - 큰 페이로드는 대역폭과 지연 시간을 증가시킵니다              │
│  - JWT에는 최소한의 인증 정보만 넣고, 세부 정보는 DB/캐시에서   │
│    가져오세요                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 클레임 기반 인가 미들웨어

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

## 7. OAuth 2.0 스코프

### 7.1 스코프 이해하기

OAuth 2.0 스코프는 액세스 토큰이 할 수 있는 작업을 제한합니다. 클라이언트 애플리케이션에 **사용자가 부여한 권한**을 나타냅니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                  OAuth 2.0 스코프                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  동의 화면:                                                      │
│  ┌─────────────────────────────────────────────┐            │
│  │  "MyApp"이 귀하의 계정에 접근하려고 합니다:  │            │
│  │                                                  │            │
│  │  [x] 프로필 읽기 (scope: profile:read)          │            │
│  │  [x] 이메일 읽기 (scope: email:read)            │            │
│  │  [ ] 대신 이메일 보내기 (email:send)            │            │
│  │  [ ] 데이터 삭제 (data:delete)                  │            │
│  │                                                  │            │
│  │         [허용]    [거부]                         │            │
│  └─────────────────────────────────────────────┘            │
│                                                                  │
│  결과 토큰:                                                      │
│  {                                                               │
│    "scope": "profile:read email:read",                           │
│    "client_id": "myapp",                                         │
│    "sub": "user_123"                                             │
│  }                                                               │
│                                                                  │
│  일반적인 스코프 패턴:                                           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 패턴             │ 예시                          │           │
│  ├──────────────────┼───────────────────────────────┤           │
│  │ resource:action   │ posts:read, posts:write       │           │
│  │ resource.action   │ user.email, user.profile      │           │
│  │ hierarchical      │ admin (모든 것을 의미)         │           │
│  │ OIDC standard     │ openid, profile, email         │           │
│  │ GitHub style      │ repo, user, gist              │           │
│  │ Google style      │ drive.readonly, calendar       │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 스코프 강제

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

## 8. 리소스 수준 권한

### 8.1 역할 기반을 넘어서: 리소스 소유권

많은 애플리케이션은 "사용자가 적절한 역할을 가지고 있는가"뿐만 아니라 "사용자가 이 특정 리소스에 접근할 수 있는가"를 확인해야 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          리소스 수준 권한 확인                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  레벨 1: 역할 확인                                               │
│  "사용자가 편집자인가?" → 예/아니오                             │
│                                                                  │
│  레벨 2: 리소스 소유권                                           │
│  "사용자가 이 게시물의 소유자인가?" → 예/아니오                 │
│                                                                  │
│  레벨 3: 공유 접근                                               │
│  "게시물이 이 사용자와 공유되었는가?" → 예/아니오               │
│                                                                  │
│  레벨 4: 조직 범위                                               │
│  "사용자가 같은 조직에 속하는가?" → 예/아니오                   │
│                                                                  │
│  결합:                                                           │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐          │
│  │ 역할 확인  │───▶│ 리소스     │───▶│ 추가         │          │
│  │ (편집자?)  │    │ 소유권     │    │ 제약 조건    │          │
│  └────────────┘    │ (소유자?)  │    │ (같은 조직?) │          │
│                    └────────────┘    └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 구현: 다층 인가

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

## 9. 일반적인 인가 취약점

### 9.1 IDOR (안전하지 않은 직접 객체 참조)

IDOR은 애플리케이션이 내부 객체 참조(예: 데이터베이스 ID)를 노출하면서 요청 사용자가 접근할 권한이 있는지 확인하지 않을 때 발생합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDOR 취약점                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  취약한 경우:                                                    │
│  GET /api/invoices/12345                                         │
│  → invoice 12345 반환 (소유권 확인 없음!)                        │
│                                                                  │
│  공격자가 ID 변경:                                               │
│  GET /api/invoices/12346                                         │
│  → 다른 사람의 invoice 반환! 🚨                                  │
│                                                                  │
│  GET /api/users/100/profile → 공격자의 프로필                    │
│  GET /api/users/101/profile → 다른 사용자의 프로필!              │
│  GET /api/users/102/profile → 또 다른 사용자!                    │
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

### 9.2 권한 상승

```
┌─────────────────────────────────────────────────────────────────┐
│              권한 상승 유형                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  수직 상승 (더 높은 권한 획득):                                 │
│  ┌──────────┐         ┌──────────┐                              │
│  │  일반    │ ──────▶ │  관리자  │                              │
│  │ 사용자   │  공격   │  사용자  │                              │
│  └──────────┘         └──────────┘                              │
│                                                                  │
│  예시: JWT 역할 클레임 수정                                     │
│  원본:     {"sub": "user1", "role": "user"}                     │
│  변조됨:   {"sub": "user1", "role": "admin"}                    │
│                                                                  │
│  수평 상승 (다른 사용자의 데이터 접근):                         │
│  ┌──────────┐         ┌──────────┐                              │
│  │ 사용자 A │ ──────▶ │ 사용자 B │                              │
│  │  (본인)  │  접근   │  (타인)  │                              │
│  └──────────┘         └──────────┘                              │
│                                                                  │
│  예시: user_id 파라미터 변경                                    │
│  본인 데이터:    GET /api/users/100/settings                    │
│  타인 데이터:    GET /api/users/101/settings                    │
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

### 9.3 인가 취약점 체크리스트

```
┌─────────────────────────────────────────────────────────────────┐
│          인가 보안 체크리스트                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [ ] 모든 엔드포인트에 명시적인 인가 확인이 있음                │
│  [ ] 인가는 서버 측에서 강제됨 (클라이언트 전용 금지)           │
│  [ ] 직접 객체 참조는 소유권에 대해 검증됨                      │
│  [ ] 관리자 기능은 관리자 역할 확인이 필요함                    │
│  [ ] 역할/권한 변경은 관리자 인가가 필요함                      │
│  [ ] 사용자 입력은 역할 또는 권한 할당을 제어하지 않음          │
│  [ ] API 응답은 다른 사용자의 데이터를 노출하지 않음            │
│  [ ] 실패한 인가는 403 (또는 IDOR의 경우 404)을 반환함          │
│  [ ] 인가 로직은 중앙 집중화됨 (중복되지 않음)                  │
│  [ ] 대량 할당이 방지됨 (필드 화이트리스팅)                     │
│  [ ] 수평 접근이 확인됨 (사용자 A는 B의 데이터에 접근 불가)     │
│  [ ] 인가 결정이 감사를 위해 로깅됨                             │
│  [ ] 기본 거부 정책 (명시적으로 허용되지 않는 한 거부)          │
│  [ ] 토큰 기반 인증은 스코프/클레임을 확인함                    │
│  [ ] 다중 테넌트 격리가 데이터 계층에서 강제됨                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 연습 문제

### 연습 문제 1: RBAC 시스템 구현

역할 계층을 가진 완전한 RBAC 시스템 구축:

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

### 연습 문제 2: ABAC 정책 엔진 구축

의료 ABAC 시스템 생성:

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

### 연습 문제 3: 인가 취약점 수정

이 코드의 모든 인가 문제를 찾아 수정:

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

### 연습 문제 4: 다중 테넌트 인가

다중 테넌트 인가 시스템 설계 및 구현:

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

### 연습 문제 5: OPA 정책 작성

다음 시나리오에 대한 Rego 정책 작성:

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

## 11. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│          인가 및 접근 제어 요약                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  모델:                                                           │
│  - RBAC: 역할 → 권한. 간단하고 널리 사용됨.                     │
│  - ABAC: 속성 → 정책 → 결정. 유연하고 컨텍스트 인식.           │
│  - ACL: 리소스별 권한 목록. 세밀한 제어.                        │
│  - 모델 결합 (RBAC + 리소스 소유권이 일반적)                    │
│                                                                  │
│  주요 원칙:                                                      │
│  - 최소 권한: 필요한 최소 권한만 부여                           │
│  - 기본 거부: 명시적으로 허용되지 않은 모든 것을 차단           │
│  - 직무 분리: 단일 역할이 모든 것을 할 수 없음                  │
│  - 심층 방어: 여러 계층에서 확인                                │
│  - 중앙 집중화: 인가 로직을 한 곳에                             │
│                                                                  │
│  구현:                                                           │
│  - 서버 측 강제 (클라이언트를 절대 신뢰하지 않음)               │
│  - 상태 비저장 API 인가를 위한 JWT 클레임                       │
│  - 제3자 접근 위임을 위한 OAuth 스코프                          │
│  - 복잡하거나 외부화된 정책을 위한 정책 엔진 (OPA)              │
│  - Flask에서 DRY 인가를 위한 데코레이터/미들웨어                │
│                                                                  │
│  일반적인 취약점:                                                │
│  - IDOR: 항상 리소스 소유권을 검증                              │
│  - 권한 상승: 클라이언트가 제공한 역할을 절대 신뢰하지 않음     │
│  - 대량 할당: 업데이트 가능한 필드를 화이트리스트로 관리        │
│  - 기능 수준 확인 누락: 모든 엔드포인트에 인증이 필요           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**이전**: [05. 인증 시스템](05_Authentication.md) | **다음**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md)
