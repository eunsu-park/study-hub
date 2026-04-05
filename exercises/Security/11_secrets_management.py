"""
Exercise Solutions: Secrets Management
======================================
Lesson 11 from Security topic.

Covers environment configuration, secret rotation, git secret scanning,
vault integration patterns, encrypted config workflows, and CI/CD auditing.
"""

import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Exercise 1: Environment Configuration System
# ---------------------------------------------------------------------------

class SecretStr:
    """Prevents accidental logging of secret values (like Pydantic SecretStr)."""
    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    def __str__(self) -> str:
        return "**********"


class ConfigManager:
    """
    Configuration management with layered priority:
    env vars > .env.local > .env > defaults
    """

    def __init__(self):
        self._config: dict[str, Any] = {}
        self._secrets: set[str] = set()
        self._required: set[str] = set()

    def define(self, key: str, default: Any = None, required: bool = False,
               is_secret: bool = False, type_hint: type = str):
        """Define a configuration key."""
        self._config[key] = {
            "value": default,
            "type": type_hint,
            "is_secret": is_secret,
        }
        if is_secret:
            self._secrets.add(key)
        if required:
            self._required.add(key)

    def _coerce_type(self, value: str, type_hint: type) -> Any:
        """Coerce string values to the specified type."""
        if type_hint == bool:
            return value.lower() in ("true", "1", "yes")
        if type_hint == int:
            return int(value)
        if type_hint == float:
            return float(value)
        if type_hint == list:
            return [v.strip() for v in value.split(",")]
        return value

    def _load_env_file(self, filepath: str) -> dict[str, str]:
        """Parse a .env file."""
        result = {}
        path = Path(filepath)
        if not path.exists():
            return result
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    def load(self, env_file: str = ".env", local_file: str = ".env.local"):
        """Load configuration from all sources (lowest to highest priority)."""
        # Layer 1: Defaults (already set via define())

        # Layer 2: .env file
        env_values = self._load_env_file(env_file)
        for key, val in env_values.items():
            if key in self._config:
                self._config[key]["value"] = self._coerce_type(
                    val, self._config[key]["type"]
                )

        # Layer 3: .env.local file
        local_values = self._load_env_file(local_file)
        for key, val in local_values.items():
            if key in self._config:
                self._config[key]["value"] = self._coerce_type(
                    val, self._config[key]["type"]
                )

        # Layer 4: Environment variables (highest priority)
        for key in self._config:
            env_val = os.environ.get(key)
            if env_val is not None:
                self._config[key]["value"] = self._coerce_type(
                    env_val, self._config[key]["type"]
                )

    def validate(self):
        """Validate all required settings are present (fail fast)."""
        missing = []
        for key in self._required:
            if self._config[key]["value"] is None:
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    def get(self, key: str) -> Any:
        """Get a configuration value."""
        entry = self._config.get(key)
        if not entry:
            raise KeyError(f"Unknown config key: {key}")
        value = entry["value"]
        if entry["is_secret"]:
            return SecretStr(value) if value else None
        return value

    def dump_safe(self) -> dict[str, str]:
        """Dump all settings with secrets masked."""
        result = {}
        for key, entry in self._config.items():
            if entry["is_secret"]:
                result[key] = "**********" if entry["value"] else "<not set>"
            else:
                result[key] = str(entry["value"])
        return result


def exercise_1_config_system():
    """Demonstrate the configuration management system."""
    config = ConfigManager()

    # Define settings
    config.define("APP_NAME", default="MyApp")
    config.define("DEBUG", default="false", type_hint=bool)
    config.define("PORT", default="8080", type_hint=int)
    config.define("DATABASE_URL", required=True, is_secret=True)
    config.define("API_KEY", required=True, is_secret=True)
    config.define("ALLOWED_HOSTS", default="localhost", type_hint=list)

    # Simulate environment variables for demo
    os.environ["DATABASE_URL"] = "postgresql://user:pass@db:5432/app"
    os.environ["API_KEY"] = "sk-test-abc123def456"

    config.load()
    config.validate()

    print("Configuration dump (safe):")
    for key, value in config.dump_safe().items():
        print(f"  {key}: {value}")

    # Accessing secrets
    db_url = config.get("DATABASE_URL")
    print(f"\nSecret as string: {db_url}")
    print(f"Secret actual value: {db_url.get_secret_value()[:20]}...")

    # Cleanup
    del os.environ["DATABASE_URL"]
    del os.environ["API_KEY"]


# ---------------------------------------------------------------------------
# Exercise 2: Secret Rotation Service
# ---------------------------------------------------------------------------

@dataclass
class SecretEntry:
    name: str
    current_value: str
    previous_value: Optional[str] = None
    rotation_interval: int = 86400  # seconds
    last_rotated: float = 0.0
    history: list[dict] = field(default_factory=list)


class SecretRotationService:
    """Automated secret rotation with zero-downtime support."""

    def __init__(self):
        self._secrets: dict[str, SecretEntry] = {}
        self._consumers: dict[str, list[callable]] = {}

    def register_secret(self, name: str, initial_value: str,
                        rotation_interval: int = 86400):
        """Register a secret for rotation management."""
        self._secrets[name] = SecretEntry(
            name=name,
            current_value=initial_value,
            rotation_interval=rotation_interval,
            last_rotated=time.time(),
        )

    def register_consumer(self, secret_name: str, callback: callable):
        """Register a consumer to be notified on rotation."""
        self._consumers.setdefault(secret_name, []).append(callback)

    def rotate(self, name: str, new_value: str = None) -> bool:
        """
        Rotate a secret with dual-secret window.
        Old value remains valid during transition.
        """
        entry = self._secrets.get(name)
        if not entry:
            return False

        if new_value is None:
            new_value = secrets.token_urlsafe(32)

        # Keep previous for dual-secret window
        entry.previous_value = entry.current_value
        entry.current_value = new_value
        entry.last_rotated = time.time()

        # Record in history
        entry.history.append({
            "rotated_at": time.time(),
            "rotated_by": "rotation-service",
        })

        # Notify consumers
        for callback in self._consumers.get(name, []):
            try:
                callback(name, new_value)
            except Exception as e:
                print(f"  Consumer notification failed: {e}")
                # Rollback
                entry.current_value = entry.previous_value
                entry.previous_value = None
                return False

        print(f"  Rotated '{name}' successfully")
        return True

    def verify(self, name: str, value: str) -> bool:
        """Verify a value against current OR previous (dual-secret window)."""
        entry = self._secrets.get(name)
        if not entry:
            return False
        return (
            secrets.compare_digest(value, entry.current_value)
            or (entry.previous_value is not None
                and secrets.compare_digest(value, entry.previous_value))
        )

    def check_rotation_needed(self) -> list[str]:
        """Check which secrets need rotation."""
        now = time.time()
        needs_rotation = []
        for name, entry in self._secrets.items():
            age = now - entry.last_rotated
            if age >= entry.rotation_interval:
                needs_rotation.append(name)
        return needs_rotation


def exercise_2_secret_rotation():
    """Demonstrate secret rotation with zero-downtime."""
    svc = SecretRotationService()

    # Register secrets
    old_key = "initial-api-key-123"
    svc.register_secret("api_key", old_key, rotation_interval=3600)
    svc.register_secret("db_password", "old-db-pass", rotation_interval=86400)

    # Register a consumer
    notified = []
    svc.register_consumer("api_key", lambda name, val: notified.append(name))

    # Old key still valid
    print(f"Old key valid: {svc.verify('api_key', old_key)}")

    # Rotate
    svc.rotate("api_key", "new-api-key-456")

    # Both keys valid (dual-secret window)
    print(f"Old key still valid: {svc.verify('api_key', old_key)}")
    print(f"New key valid: {svc.verify('api_key', 'new-api-key-456')}")
    print(f"Consumer notified: {notified}")


# ---------------------------------------------------------------------------
# Exercise 3: Git Secret Scanner
# ---------------------------------------------------------------------------

class GitSecretScanner:
    """Scan files for accidentally committed secrets."""

    DEFAULT_PATTERNS = [
        (r"(?i)(?:api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})",
         "API Key"),
        (r"(?i)(?:secret|password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{8,})",
         "Password/Secret"),
        (r"(?:AKIA|ASIA)[A-Z0-9]{16}",
         "AWS Access Key"),
        (r"(?i)(?:sk[_-]?live|sk[_-]?test)[_-]?[a-zA-Z0-9]{20,}",
         "Stripe Secret Key"),
        (r"ghp_[a-zA-Z0-9]{36}",
         "GitHub Personal Access Token"),
        (r"(?i)(?:jdbc|postgresql|mysql|mongodb)://[^\s]+:[^\s]+@",
         "Database Connection String"),
        (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
         "Private Key"),
    ]

    def __init__(self, custom_patterns: list[tuple[str, str]] = None,
                 allowlist: set[str] = None):
        self.patterns = self.DEFAULT_PATTERNS + (custom_patterns or [])
        self.allowlist = allowlist or set()

    def scan_content(self, content: str, filename: str = "") -> list[dict]:
        """Scan content for secrets."""
        findings = []
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern, secret_type in self.patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    finding_key = f"{filename}:{line_num}:{secret_type}"
                    if finding_key in self.allowlist:
                        continue
                    # Mask the matched value
                    value = match.group(0)
                    masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
                    findings.append({
                        "file": filename,
                        "line": line_num,
                        "type": secret_type,
                        "match": masked,
                        "severity": "HIGH",
                    })
        return findings

    def scan_directory(self, directory: str,
                       extensions: set[str] = None) -> list[dict]:
        """Scan a directory recursively for secrets."""
        if extensions is None:
            extensions = {".py", ".js", ".ts", ".yaml", ".yml", ".json",
                          ".env", ".cfg", ".conf", ".ini", ".toml"}

        all_findings = []
        root = Path(directory)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                try:
                    content = path.read_text(errors="ignore")
                    findings = self.scan_content(content, str(path))
                    all_findings.extend(findings)
                except Exception:
                    pass
        return all_findings

    def generate_report(self, findings: list[dict],
                        format: str = "text") -> str:
        """Generate a report from scan findings."""
        if format == "json":
            return json.dumps(findings, indent=2)

        lines = [f"Secret Scan Report: {len(findings)} finding(s)\n"]
        for f in findings:
            lines.append(
                f"  [{f['severity']}] {f['file']}:{f['line']} "
                f"— {f['type']}: {f['match']}"
            )
        return "\n".join(lines)


def exercise_3_git_secret_scanner():
    """Demonstrate the secret scanner."""
    scanner = GitSecretScanner()

    # Scan sample content
    test_code = """
import os
API_KEY = "sk_test_EXAMPLE_DO_NOT_USE_1234567890ab"
DB_URL = "postgresql://admin:p4ssw0rd@db.example.com:5432/mydb"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
github_token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
password = "changeme123"
safe_variable = "this is not a secret"
"""

    findings = scanner.scan_content(test_code, "app/config.py")
    print(scanner.generate_report(findings))


# ---------------------------------------------------------------------------
# Exercise 4: Vault Integration Library (Simplified)
# ---------------------------------------------------------------------------

class VaultClient:
    """
    Simplified Vault-like client with caching and fallback.
    Demonstrates the integration pattern without requiring a real Vault server.
    """

    def __init__(self, vault_url: str = "http://vault:8200",
                 token: str = None, cache_ttl: int = 300):
        self._vault_url = vault_url
        self._token = token
        self._cache: dict[str, dict] = {}
        self._cache_ttl = cache_ttl
        # Simulated vault storage
        self._vault_store: dict[str, dict] = {}

    def _simulate_vault_read(self, path: str) -> Optional[dict]:
        """Simulate reading from Vault."""
        return self._vault_store.get(path)

    def _simulate_vault_write(self, path: str, data: dict):
        """Simulate writing to Vault."""
        self._vault_store[path] = data

    def read_secret(self, path: str) -> Optional[dict]:
        """Read a secret with caching and env fallback."""
        # Check cache first
        cached = self._cache.get(path)
        if cached and time.time() < cached["expires"]:
            return cached["data"]

        # Try Vault
        try:
            data = self._simulate_vault_read(path)
            if data:
                self._cache[path] = {
                    "data": data,
                    "expires": time.time() + self._cache_ttl,
                }
                return data
        except Exception:
            pass

        # Fallback to environment variables
        env_key = path.replace("/", "_").upper()
        env_val = os.environ.get(env_key)
        if env_val:
            return {"value": env_val, "source": "environment"}

        return None

    def write_secret(self, path: str, data: dict):
        """Write a secret to Vault."""
        self._simulate_vault_write(path, data)
        # Invalidate cache
        self._cache.pop(path, None)


def exercise_4_vault_integration():
    """Demonstrate Vault integration patterns."""
    vault = VaultClient(cache_ttl=60)

    # Store secrets
    vault.write_secret("secret/database", {
        "username": "app_user",
        "password": "vault-managed-password",
    })

    # Read (from vault)
    secret = vault.read_secret("secret/database")
    print(f"From Vault: {secret}")

    # Read again (from cache)
    secret = vault.read_secret("secret/database")
    print(f"From cache: {secret}")

    # Fallback to env
    os.environ["SECRET_API_KEY"] = "env-fallback-key"
    fallback = vault.read_secret("secret/api_key")
    print(f"Env fallback: {fallback}")
    del os.environ["SECRET_API_KEY"]


# ---------------------------------------------------------------------------
# Exercise 5: SOPS Workflow (Simplified)
# ---------------------------------------------------------------------------

def exercise_5_sops_workflow():
    """
    Demonstrate SOPS-like encrypted config workflow concepts.
    Uses simple symmetric encryption as a stand-in for real SOPS.
    """
    print("SOPS Workflow Automation Concepts")
    print("=" * 60)
    print("""
1. Create encrypted configs for multiple environments:
   sops --encrypt --age <public-key> config/dev.yaml > config/dev.enc.yaml
   sops --encrypt --age <public-key> config/staging.yaml > config/staging.enc.yaml
   sops --encrypt --age <public-key> config/prod.yaml > config/prod.enc.yaml

2. Validate encrypted files contain required keys:
   sops -d config/prod.enc.yaml | yq '.database_url, .api_key' > /dev/null

3. Diff two encrypted files (structure only):
   diff <(sops -d env1.yaml | yq 'keys') <(sops -d env2.yaml | yq 'keys')

4. Key rotation (re-encrypt all files):
   for f in config/*.enc.yaml; do
     sops rotate -i --add-age <new-key> --rm-age <old-key> "$f"
   done

5. Git hook to prevent unencrypted commits:
   # .git/hooks/pre-commit
   if grep -rn "password:" config/*.yaml 2>/dev/null; then
     echo "ERROR: Unencrypted secrets detected"
     exit 1
   fi

6. Generate .env from encrypted config:
   sops -d config/dev.enc.yaml | yq -r 'to_entries | .[] | .key + "=" + .value' > .env
""")


# ---------------------------------------------------------------------------
# Exercise 6: CI/CD Secrets Audit
# ---------------------------------------------------------------------------

class CICDSecretsAuditor:
    """Audit GitHub Actions workflows for secret handling issues."""

    def audit_workflow(self, workflow_content: str,
                       filename: str = "workflow.yml") -> list[dict]:
        """Audit a GitHub Actions workflow file."""
        findings = []

        # Check 1: Secrets in command arguments (visible in logs)
        if re.search(r"run:.*\$\{\{ secrets\.\w+ \}\}", workflow_content):
            findings.append({
                "check": "Secret in shell command",
                "severity": "HIGH",
                "detail": "Secrets passed directly in 'run' commands may be visible in logs",
                "fix": "Use environment variables: env: MY_SECRET: ${{ secrets.MY_SECRET }}",
            })

        # Check 2: Actions not pinned to SHA
        tag_refs = re.findall(r"uses:\s+(\S+@(?:v\d|main|master|latest))", workflow_content)
        for ref in tag_refs:
            findings.append({
                "check": "Action not pinned to SHA",
                "severity": "MEDIUM",
                "detail": f"'{ref}' uses mutable tag, vulnerable to supply chain attack",
                "fix": "Pin to full SHA: uses: actions/checkout@abc123...",
            })

        # Check 3: Overly broad GITHUB_TOKEN permissions
        if "permissions:" not in workflow_content:
            findings.append({
                "check": "No explicit permissions",
                "severity": "MEDIUM",
                "detail": "GITHUB_TOKEN gets default permissions (may be overly broad)",
                "fix": "Add explicit permissions block with minimum required scopes",
            })

        # Check 4: Secrets not scoped to environment
        if "environment:" not in workflow_content and "secrets." in workflow_content:
            findings.append({
                "check": "Secrets not environment-scoped",
                "severity": "LOW",
                "detail": "Secrets available to all environments/branches",
                "fix": "Use GitHub environment protection rules to scope secrets",
            })

        return findings


def exercise_6_cicd_audit():
    """Demonstrate CI/CD secrets auditing."""
    sample_workflow = """
name: Deploy
on: push
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          password: ${{ secrets.DOCKER_PASSWORD }}
      - run: echo ${{ secrets.API_KEY }} | deploy.sh
      - run: curl -H "Authorization: Bearer ${{ secrets.TOKEN }}" https://api.example.com
"""

    auditor = CICDSecretsAuditor()
    findings = auditor.audit_workflow(sample_workflow)

    print("CI/CD Secrets Audit Report")
    print("=" * 60)
    for f in findings:
        print(f"\n  [{f['severity']}] {f['check']}")
        print(f"  Detail: {f['detail']}")
        print(f"  Fix: {f['fix']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Environment Configuration System")
    print("=" * 70)
    exercise_1_config_system()

    print("\n" + "=" * 70)
    print("Exercise 2: Secret Rotation Service")
    print("=" * 70)
    exercise_2_secret_rotation()

    print("\n" + "=" * 70)
    print("Exercise 3: Git Secret Scanner")
    print("=" * 70)
    exercise_3_git_secret_scanner()

    print("\n" + "=" * 70)
    print("Exercise 4: Vault Integration Library")
    print("=" * 70)
    exercise_4_vault_integration()

    print("\n" + "=" * 70)
    print("Exercise 5: SOPS Workflow")
    print("=" * 70)
    exercise_5_sops_workflow()

    print("\n" + "=" * 70)
    print("Exercise 6: CI/CD Secrets Audit")
    print("=" * 70)
    exercise_6_cicd_audit()
