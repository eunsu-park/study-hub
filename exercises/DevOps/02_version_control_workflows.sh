#!/bin/bash
# Exercises for Lesson 02: Version Control Workflows
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Git Branching Strategy Comparison ===
# Problem: Compare GitFlow, GitHub Flow, and Trunk-Based Development.
# Determine which strategy fits a given team scenario.
exercise_1() {
    echo "=== Exercise 1: Git Branching Strategy Comparison ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
def recommend_branching_strategy(
    team_size: int,
    release_cadence: str,       # "continuous", "weekly", "monthly", "quarterly"
    ci_maturity: str,           # "high", "medium", "low"
    compliance_required: bool,
) -> dict:
    """Recommend a branching strategy based on team characteristics."""
    strategies = {
        "trunk-based": {
            "description": "All developers commit to main; short-lived feature branches (<1 day)",
            "pros": ["Fastest feedback loop", "No merge conflicts", "Simplest model"],
            "cons": ["Requires strong CI", "Feature flags needed for WIP"],
            "best_for": "High CI maturity, continuous delivery, small to large teams",
        },
        "github-flow": {
            "description": "Feature branches off main; PR review; merge to main; deploy from main",
            "pros": ["Simple mental model", "Good PR review flow", "Works with GitHub"],
            "cons": ["Long-lived branches cause drift", "No release branches"],
            "best_for": "SaaS products, teams of 5-20, weekly releases",
        },
        "gitflow": {
            "description": "develop + main + feature + release + hotfix branches",
            "pros": ["Clear release process", "Supports parallel releases", "Audit trail"],
            "cons": ["Complex branching", "Slow feedback", "Merge conflicts"],
            "best_for": "Versioned software, compliance needs, quarterly releases",
        },
    }

    # Decision logic
    if ci_maturity == "high" and release_cadence == "continuous":
        recommended = "trunk-based"
    elif compliance_required and release_cadence in ("monthly", "quarterly"):
        recommended = "gitflow"
    elif team_size <= 20 and release_cadence in ("continuous", "weekly"):
        recommended = "github-flow"
    else:
        recommended = "github-flow"  # Safe default

    return {
        "recommended": recommended,
        "details": strategies[recommended],
        "all_strategies": strategies,
    }

# Example: 8-person SaaS team, weekly releases, good CI
result = recommend_branching_strategy(
    team_size=8, release_cadence="weekly",
    ci_maturity="high", compliance_required=False,
)
print(f"Recommended: {result['recommended']}")
print(f"  {result['details']['description']}")
SOLUTION
}

# === Exercise 2: Conventional Commits ===
# Problem: Write a commit message validator that enforces the
# Conventional Commits specification.
exercise_2() {
    echo "=== Exercise 2: Conventional Commits ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import re

VALID_TYPES = {"feat", "fix", "docs", "style", "refactor", "perf",
               "test", "build", "ci", "chore", "revert"}

# Pattern: type(scope)?: description
CONVENTIONAL_PATTERN = re.compile(
    r'^(?P<type>\w+)'
    r'(?:\((?P<scope>[^)]+)\))?'
    r'(?P<breaking>!)?'
    r': '
    r'(?P<description>.+)$'
)

def validate_commit_message(message: str) -> dict:
    """Validate a commit message against Conventional Commits spec."""
    lines = message.strip().split('\n')
    subject = lines[0]
    errors = []

    match = CONVENTIONAL_PATTERN.match(subject)
    if not match:
        return {"valid": False, "errors": ["Does not match conventional format: type(scope): description"]}

    commit_type = match.group("type")
    scope = match.group("scope")
    breaking = match.group("breaking") == "!"
    description = match.group("description")

    if commit_type not in VALID_TYPES:
        errors.append(f"Invalid type '{commit_type}'. Must be one of: {', '.join(sorted(VALID_TYPES))}")

    if len(subject) > 72:
        errors.append(f"Subject line too long ({len(subject)} > 72 chars)")

    if description and description[0].isupper():
        errors.append("Description should start with lowercase")

    if description and description.endswith('.'):
        errors.append("Description should not end with a period")

    # Check for body separator
    if len(lines) > 1 and lines[1].strip() != "":
        errors.append("Second line must be blank (separates subject from body)")

    # Check for BREAKING CHANGE footer
    has_breaking_footer = any(
        line.startswith("BREAKING CHANGE:") or line.startswith("BREAKING-CHANGE:")
        for line in lines
    )

    return {
        "valid": len(errors) == 0,
        "type": commit_type,
        "scope": scope,
        "breaking": breaking or has_breaking_footer,
        "description": description,
        "errors": errors,
    }

# Test cases
test_messages = [
    "feat(auth): add JWT token refresh endpoint",
    "fix: resolve null pointer in user lookup",
    "feat!: drop support for Python 3.8",
    "INVALID commit message without type",
    "feat: Add uppercase description.",
]

for msg in test_messages:
    result = validate_commit_message(msg)
    status = "VALID" if result["valid"] else "INVALID"
    print(f"[{status}] {msg}")
    if result.get("errors"):
        for e in result["errors"]:
            print(f"         {e}")
SOLUTION
}

# === Exercise 3: Merge vs Rebase Workflow ===
# Problem: Demonstrate when to use merge vs rebase with concrete
# Git command sequences.
exercise_3() {
    echo "=== Exercise 3: Merge vs Rebase Workflow ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Scenario: You have a feature branch that is 3 commits behind main.

# === OPTION A: Merge (preserves history, creates merge commit) ===
# Use when: shared branches, you want a complete history record
git checkout feature/add-search
git merge main
# Creates a merge commit that ties both histories together
# History: feature commits + merge commit (non-linear)

# === OPTION B: Rebase (rewrites history, linear timeline) ===
# Use when: local/unshared branches, you want a clean linear history
git checkout feature/add-search
git rebase main
# Replays your commits ON TOP of the latest main
# History: linear sequence (as if you started from current main)

# === OPTION C: Interactive Rebase (clean up before merge) ===
# Squash multiple WIP commits into one meaningful commit
git checkout feature/add-search
git rebase -i main
# In the editor, mark commits as:
#   pick   abc1234 feat: add search API endpoint
#   squash def5678 WIP: search filtering
#   squash ghi9012 fix: search edge case
# Result: one clean commit with all changes

# === Decision Matrix ===
# +-------------------+---------------------------+---------------------------+
# | Situation         | Use Merge                 | Use Rebase                |
# +-------------------+---------------------------+---------------------------+
# | Shared branch     | YES (safe for others)     | NO (rewrites shared SHA)  |
# | Local feature     | OK (but noisy)            | YES (clean history)       |
# | Main <- feature   | YES (PR merge)            | Rebase first, then merge  |
# | Long-lived branch | YES (periodic sync)       | Risky (conflict replay)   |
# | Compliance audit  | YES (complete record)     | NO (history altered)      |
# +-------------------+---------------------------+---------------------------+

# GOLDEN RULE: Never rebase commits that have been pushed and shared
# with others. Rebase rewrites commit SHAs, which breaks others' history.
SOLUTION
}

# === Exercise 4: Semantic Versioning ===
# Problem: Implement a version bumping tool that follows SemVer
# based on conventional commit types.
exercise_4() {
    echo "=== Exercise 4: Semantic Versioning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass

@dataclass
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: str = ""

    def __str__(self) -> str:
        v = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            v += f"-{self.prerelease}"
        return v

    def bump_major(self) -> "SemVer":
        return SemVer(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemVer":
        return SemVer(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemVer":
        return SemVer(self.major, self.minor, self.patch + 1)

def determine_bump(commits: list[dict]) -> str:
    """Determine version bump from conventional commits.

    BREAKING CHANGE -> major
    feat            -> minor
    fix             -> patch
    """
    has_breaking = any(c.get("breaking") for c in commits)
    has_feat = any(c["type"] == "feat" for c in commits)

    if has_breaking:
        return "major"
    if has_feat:
        return "minor"
    return "patch"

# Example
current = SemVer(1, 3, 7)
commits = [
    {"type": "fix", "description": "handle null user", "breaking": False},
    {"type": "feat", "description": "add search endpoint", "breaking": False},
    {"type": "docs", "description": "update API docs", "breaking": False},
]

bump_type = determine_bump(commits)
if bump_type == "major":
    new_version = current.bump_major()
elif bump_type == "minor":
    new_version = current.bump_minor()
else:
    new_version = current.bump_patch()

print(f"Current: v{current}")
print(f"Bump:    {bump_type} (from commits: {[c['type'] for c in commits]})")
print(f"New:     v{new_version}")
# Output: Current: v1.3.7 -> Bump: minor -> New: v1.4.0
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 02: Version Control Workflows"
echo "============================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
