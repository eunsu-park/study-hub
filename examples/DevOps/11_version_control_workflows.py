#!/usr/bin/env python3
"""Example: Version Control Workflows — Branching Models & Release Strategies

Demonstrates Git branching model simulation, semantic versioning automation,
merge conflict detection, and branch policy enforcement for DevOps teams.
Related lesson: 02_Version_Control_Workflows.md
"""

# =============================================================================
# WHY VERSION CONTROL WORKFLOWS MATTER
# Consistent branching strategies (Git Flow, GitHub Flow, Trunk-Based) reduce
# integration friction, enable parallel development, and create auditable
# release histories. Automating version bumps and branch policies prevents
# human error in the release process.
# =============================================================================

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# =============================================================================
# 1. SEMANTIC VERSIONING
# =============================================================================

class BumpType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class SemanticVersion:
    """Semantic version (semver) with bump and comparison logic."""
    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: str = ""

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string like '1.2.3' or '1.2.3-rc.1'."""
        match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", version_str)
        if not match:
            raise ValueError(f"Invalid semver: {version_str}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4) or "",
        )

    def bump(self, bump_type: BumpType) -> "SemanticVersion":
        """Return a new version bumped by the given type."""
        if bump_type == BumpType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif bump_type == BumpType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        else:
            return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base


# =============================================================================
# 2. BRANCHING MODEL SIMULATOR
# =============================================================================

@dataclass
class Branch:
    """Represents a Git branch with metadata."""
    name: str
    base: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commits: list[str] = field(default_factory=list)
    merged: bool = False


@dataclass
class GitFlowModel:
    """Simulates Git Flow branching: main, develop, feature/*, release/*, hotfix/*."""
    branches: dict[str, Branch] = field(default_factory=dict)
    current_version: SemanticVersion = field(default_factory=SemanticVersion)
    release_log: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Initialize core branches
        self.branches["main"] = Branch(name="main", base="")
        self.branches["develop"] = Branch(name="develop", base="main")

    def create_feature(self, feature_name: str) -> Branch:
        """Create a feature branch from develop."""
        branch_name = f"feature/{feature_name}"
        if branch_name in self.branches:
            raise ValueError(f"Branch {branch_name} already exists")
        branch = Branch(name=branch_name, base="develop")
        self.branches[branch_name] = branch
        return branch

    def finish_feature(self, feature_name: str) -> str:
        """Merge a feature branch back into develop."""
        branch_name = f"feature/{feature_name}"
        branch = self.branches.get(branch_name)
        if not branch:
            raise ValueError(f"Branch {branch_name} not found")
        if branch.merged:
            raise ValueError(f"Branch {branch_name} already merged")
        branch.merged = True
        # Transfer commits to develop
        self.branches["develop"].commits.extend(branch.commits)
        return f"Merged {branch_name} into develop ({len(branch.commits)} commits)"

    def start_release(self, bump_type: BumpType = BumpType.MINOR) -> Branch:
        """Create a release branch from develop with a version bump."""
        new_version = self.current_version.bump(bump_type)
        branch_name = f"release/{new_version}"
        branch = Branch(name=branch_name, base="develop")
        self.branches[branch_name] = branch
        return branch

    def finish_release(self, version_str: str) -> str:
        """Merge release into main and develop, tag the version."""
        branch_name = f"release/{version_str}"
        branch = self.branches.get(branch_name)
        if not branch:
            raise ValueError(f"Release branch {branch_name} not found")
        branch.merged = True
        self.current_version = SemanticVersion.parse(version_str)
        tag = f"v{self.current_version}"
        self.release_log.append(tag)
        return f"Released {tag} — merged into main and develop"

    def create_hotfix(self, fix_name: str) -> Branch:
        """Create a hotfix branch from main for urgent patches."""
        branch_name = f"hotfix/{fix_name}"
        branch = Branch(name=branch_name, base="main")
        self.branches[branch_name] = branch
        return branch

    def get_active_branches(self) -> list[str]:
        """Return names of unmerged branches."""
        return [b.name for b in self.branches.values() if not b.merged]


# =============================================================================
# 3. BRANCH POLICY ENFORCEMENT
# =============================================================================

BRANCH_RULES = {
    "main": {"require_pr": True, "min_approvals": 2, "require_ci_pass": True},
    "develop": {"require_pr": True, "min_approvals": 1, "require_ci_pass": True},
    "feature/*": {"require_pr": False, "min_approvals": 0, "require_ci_pass": False},
}


def check_merge_policy(target_branch: str, approvals: int, ci_passed: bool) -> list[str]:
    """Validate that a merge meets the branch protection policy."""
    violations: list[str] = []
    # Match wildcard rules
    rule = BRANCH_RULES.get(target_branch)
    if not rule:
        for pattern, r in BRANCH_RULES.items():
            if "*" in pattern and target_branch.startswith(pattern.replace("*", "")):
                rule = r
                break
    if not rule:
        return [f"No policy defined for branch '{target_branch}'"]
    if rule["require_pr"]:
        if approvals < rule["min_approvals"]:
            violations.append(
                f"Need {rule['min_approvals']} approvals, got {approvals}"
            )
    if rule["require_ci_pass"] and not ci_passed:
        violations.append("CI checks must pass before merging")
    return violations


# =============================================================================
# 4. COMMIT MESSAGE CONVENTION VALIDATOR
# =============================================================================

CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\(.+\))?!?: .{1,72}$"
)


def validate_commit_message(message: str) -> tuple[bool, str]:
    """Validate a commit message against Conventional Commits spec."""
    first_line = message.strip().split("\n")[0]
    if CONVENTIONAL_COMMIT_RE.match(first_line):
        return True, "Valid conventional commit"
    return False, f"Invalid format: '{first_line}' — expected 'type(scope): description'"


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Semantic Versioning ---
    print("=" * 60)
    print("Semantic Versioning")
    print("=" * 60)
    v = SemanticVersion.parse("1.4.2")
    print(f"Current:     {v}")
    print(f"Patch bump:  {v.bump(BumpType.PATCH)}")
    print(f"Minor bump:  {v.bump(BumpType.MINOR)}")
    print(f"Major bump:  {v.bump(BumpType.MAJOR)}")

    # --- Git Flow Simulation ---
    print(f"\n{'=' * 60}")
    print("Git Flow Branching Model")
    print("=" * 60)
    gf = GitFlowModel(current_version=SemanticVersion(1, 0, 0))
    feat = gf.create_feature("user-auth")
    feat.commits.extend(["Add login form", "Add JWT validation"])
    print(gf.finish_feature("user-auth"))
    rel = gf.start_release()
    print(gf.finish_release(str(SemanticVersion(1, 1, 0))))
    print(f"Active branches: {gf.get_active_branches()}")
    print(f"Release log: {gf.release_log}")

    # --- Branch Policy ---
    print(f"\n{'=' * 60}")
    print("Branch Policy Enforcement")
    print("=" * 60)
    violations = check_merge_policy("main", approvals=1, ci_passed=False)
    for v in violations:
        print(f"  VIOLATION: {v}")
    ok = check_merge_policy("main", approvals=2, ci_passed=True)
    print(f"  main with 2 approvals + CI pass: {'PASS' if not ok else 'FAIL'}")

    # --- Commit Validation ---
    print(f"\n{'=' * 60}")
    print("Conventional Commit Validation")
    print("=" * 60)
    for msg in ["feat(auth): add OAuth2 support", "fixed the bug", "ci: update GHA runners"]:
        valid, detail = validate_commit_message(msg)
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {msg!r} — {detail}")
