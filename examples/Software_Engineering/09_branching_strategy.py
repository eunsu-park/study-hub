"""
Git Branching Strategy Simulator

Demonstrates:
- Git Flow branching model
- GitHub Flow (trunk-based)
- Branch lifecycle simulation
- Merge conflict scenarios
- Release management

Theory:
- Git Flow: develop → feature → develop → release → main.
  Good for scheduled releases.
- GitHub Flow: main → feature → PR → main.
  Good for continuous deployment.
- Trunk-based: very short-lived branches (< 1 day), feature flags.
  Good for high-velocity teams.

Adapted from Software Engineering Lesson 09.
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class BranchType(Enum):
    MAIN = "main"
    DEVELOP = "develop"
    FEATURE = "feature"
    RELEASE = "release"
    HOTFIX = "hotfix"


@dataclass
class Commit:
    hash: str
    message: str
    author: str
    branch: str
    parents: list[str] = field(default_factory=list)

    def short_hash(self) -> str:
        return self.hash[:7]


@dataclass
class Branch:
    name: str
    branch_type: BranchType
    base: str  # parent branch
    commits: list[Commit] = field(default_factory=list)
    merged: bool = False
    deleted: bool = False


class GitSimulator:
    """Simplified git repository simulator."""

    def __init__(self):
        self.branches: dict[str, Branch] = {}
        self.all_commits: list[Commit] = []
        self.commit_counter = 0
        self.log: list[str] = []

    def _next_hash(self) -> str:
        self.commit_counter += 1
        import hashlib
        return hashlib.md5(str(self.commit_counter).encode()).hexdigest()[:10]

    def init(self) -> None:
        """Initialize repository with main branch."""
        main = Branch("main", BranchType.MAIN, "")
        self.branches["main"] = main
        self.commit("main", "Initial commit", "system")

    def create_branch(self, name: str, base: str,
                      branch_type: BranchType) -> None:
        if base not in self.branches:
            raise ValueError(f"Base branch '{base}' not found")
        branch = Branch(name, branch_type, base)
        # Copy commits from base
        branch.commits = list(self.branches[base].commits)
        self.branches[name] = branch
        self.log.append(f"  branch '{name}' created from '{base}'")

    def commit(self, branch: str, message: str, author: str) -> Commit:
        if branch not in self.branches:
            raise ValueError(f"Branch '{branch}' not found")
        h = self._next_hash()
        parents = ([self.branches[branch].commits[-1].hash]
                   if self.branches[branch].commits else [])
        c = Commit(h, message, author, branch, parents)
        self.branches[branch].commits.append(c)
        self.all_commits.append(c)
        self.log.append(f"  [{branch}] {c.short_hash()} {message}")
        return c

    def merge(self, source: str, target: str,
              strategy: str = "merge") -> Commit:
        if source not in self.branches or target not in self.branches:
            raise ValueError("Branch not found")

        src = self.branches[source]
        tgt = self.branches[target]

        # Create merge commit
        msg = f"Merge '{source}' into '{target}'"
        h = self._next_hash()
        parents = [tgt.commits[-1].hash, src.commits[-1].hash]
        merge_commit = Commit(h, msg, "system", target, parents)

        # Add source-only commits to target
        target_hashes = {c.hash for c in tgt.commits}
        for c in src.commits:
            if c.hash not in target_hashes:
                tgt.commits.append(c)
        tgt.commits.append(merge_commit)

        self.all_commits.append(merge_commit)
        src.merged = True
        self.log.append(f"  merge '{source}' → '{target}' ({merge_commit.short_hash()})")
        return merge_commit

    def delete_branch(self, name: str) -> None:
        if name in self.branches:
            self.branches[name].deleted = True
            self.log.append(f"  branch '{name}' deleted")

    def print_branches(self) -> None:
        print(f"    {'Branch':<25} {'Type':<10} {'Commits':>8} {'Status'}")
        print(f"    {'-'*25} {'-'*10} {'-'*8} {'-'*10}")
        for name, branch in sorted(self.branches.items()):
            status = "deleted" if branch.deleted else \
                     "merged" if branch.merged else "active"
            if not branch.deleted:
                print(f"    {name:<25} {branch.branch_type.value:<10} "
                      f"{len(branch.commits):>8} {status}")


# ── Demos ──────────────────────────────────────────────────────────────

def demo_git_flow():
    print("=" * 60)
    print("GIT FLOW BRANCHING MODEL")
    print("=" * 60)

    git = GitSimulator()
    git.init()

    # Create develop branch
    git.create_branch("develop", "main", BranchType.DEVELOP)
    git.commit("develop", "Set up project structure", "alice")

    # Feature branches
    print(f"\n  --- Feature Development ---")
    git.create_branch("feature/login", "develop", BranchType.FEATURE)
    git.commit("feature/login", "Add login form", "bob")
    git.commit("feature/login", "Add OAuth integration", "bob")

    git.create_branch("feature/search", "develop", BranchType.FEATURE)
    git.commit("feature/search", "Add search endpoint", "charlie")
    git.commit("feature/search", "Add search UI", "charlie")

    # Merge features to develop
    print(f"\n  --- Merge Features ---")
    git.merge("feature/login", "develop")
    git.delete_branch("feature/login")
    git.merge("feature/search", "develop")
    git.delete_branch("feature/search")

    # Release branch
    print(f"\n  --- Release ---")
    git.create_branch("release/1.0", "develop", BranchType.RELEASE)
    git.commit("release/1.0", "Bump version to 1.0", "alice")
    git.commit("release/1.0", "Fix release bug", "alice")

    # Merge to main and develop
    git.merge("release/1.0", "main")
    git.merge("release/1.0", "develop")
    git.commit("main", "Tag v1.0", "system")
    git.delete_branch("release/1.0")

    # Hotfix
    print(f"\n  --- Hotfix ---")
    git.create_branch("hotfix/1.0.1", "main", BranchType.HOTFIX)
    git.commit("hotfix/1.0.1", "Fix critical security bug", "bob")
    git.merge("hotfix/1.0.1", "main")
    git.merge("hotfix/1.0.1", "develop")
    git.commit("main", "Tag v1.0.1", "system")
    git.delete_branch("hotfix/1.0.1")

    print(f"\n  Final branch state:")
    git.print_branches()

    print(f"""
  Git Flow Summary:
    main     ─── stable releases only
    develop  ─── integration branch
    feature/ ─── new features (from develop)
    release/ ─── release prep (from develop → main + develop)
    hotfix/  ─── urgent fixes (from main → main + develop)""")


def demo_github_flow():
    print("\n" + "=" * 60)
    print("GITHUB FLOW (TRUNK-BASED)")
    print("=" * 60)

    git = GitSimulator()
    git.init()

    # Feature 1: short-lived branch
    print(f"\n  --- Feature 1 (short-lived) ---")
    git.create_branch("feature/add-search", "main", BranchType.FEATURE)
    git.commit("feature/add-search", "Implement search API", "alice")
    git.commit("feature/add-search", "Add tests for search", "alice")
    # PR review + merge
    git.merge("feature/add-search", "main")
    git.delete_branch("feature/add-search")

    # Feature 2
    print(f"\n  --- Feature 2 (short-lived) ---")
    git.create_branch("feature/dark-mode", "main", BranchType.FEATURE)
    git.commit("feature/dark-mode", "Add dark mode toggle", "bob")
    git.merge("feature/dark-mode", "main")
    git.delete_branch("feature/dark-mode")

    # Deploy from main
    git.commit("main", "Deploy to production", "ci-bot")

    print(f"\n  Final branch state:")
    git.print_branches()

    print(f"""
  GitHub Flow Summary:
    1. Create branch from main
    2. Make commits
    3. Open Pull Request
    4. Review + CI checks
    5. Merge to main
    6. Deploy immediately""")


def demo_comparison():
    print("\n" + "=" * 60)
    print("BRANCHING STRATEGY COMPARISON")
    print("=" * 60)

    print(f"""
  {'Feature':<30} {'Git Flow':>12} {'GitHub Flow':>12} {'Trunk-Based':>12}
  {'-'*30} {'-'*12} {'-'*12} {'-'*12}
  {'Complexity':<30} {'High':>12} {'Low':>12} {'Low':>12}
  {'Release cadence':<30} {'Scheduled':>12} {'Continuous':>12} {'Continuous':>12}
  {'Long-lived branches':<30} {'Yes':>12} {'No':>12} {'No':>12}
  {'Parallel releases':<30} {'Yes':>12} {'No':>12} {'No':>12}
  {'Feature flags needed':<30} {'Rarely':>12} {'Sometimes':>12} {'Always':>12}
  {'Branch lifetime':<30} {'Days-weeks':>12} {'Hours-days':>12} {'Hours':>12}
  {'Best for':<30} {'Enterprise':>12} {'Web apps':>12} {'High-vel':>12}
  {'CI/CD integration':<30} {'Moderate':>12} {'High':>12} {'Essential':>12}

  Recommendation:
  - Small team, web app, frequent deploys → GitHub Flow
  - Large team, mobile app, versioned releases → Git Flow
  - Expert team, high trust, very fast iteration → Trunk-Based""")


if __name__ == "__main__":
    demo_git_flow()
    demo_github_flow()
    demo_comparison()
