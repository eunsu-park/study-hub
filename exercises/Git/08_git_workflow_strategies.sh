#!/bin/bash
# Exercises for Lesson 08: Git Workflow Strategies
# Topic: Git
# Solutions to practice problems from the lesson.

# === Exercise 1: Git Flow Full Cycle ===
# Problem: Simulate a complete Git Flow cycle: init, feature, release, hotfix.
exercise_1() {
    echo "=== Exercise 1: Git Flow Full Cycle ==="
    echo ""
    echo "Solution commands (using manual Git commands):"
    echo ""
    echo "  # Step 1: Initialize Git Flow on a new repository"
    echo "  mkdir gitflow-exercise && cd gitflow-exercise"
    echo "  git init"
    echo "  echo '# Git Flow Exercise' > README.md"
    echo "  git add . && git commit -m 'initial commit'"
    echo "  git branch develop    # Create the develop branch"
    echo ""
    echo "  # Step 2: Start feature/user-profile from develop"
    echo "  git switch develop"
    echo "  git switch -c feature/user-profile"
    echo ""
    echo "  # Make two feature commits"
    echo "  echo 'User model' > user.py"
    echo "  git add . && git commit -m 'feat: add user model'"
    echo "  echo 'User API' > user_api.py"
    echo "  git add . && git commit -m 'feat: add user profile API'"
    echo ""
    echo "  # Finish feature: merge into develop with --no-ff"
    echo "  # Why: --no-ff creates a merge commit, preserving the feature branch history"
    echo "  git switch develop"
    echo "  git merge --no-ff feature/user-profile -m 'merge: feature/user-profile into develop'"
    echo "  git branch -d feature/user-profile"
    echo ""
    echo "  # Step 3: Start release/1.0.0 from develop"
    echo "  git switch -c release/1.0.0"
    echo "  echo 'Version 1.0.0' > VERSION"
    echo "  git add . && git commit -m 'chore: bump version to 1.0.0'"
    echo ""
    echo "  # Finish release: merge into BOTH main and develop"
    echo "  # Merge into main (production)"
    echo "  git switch main"
    echo "  git merge --no-ff release/1.0.0 -m 'merge: release/1.0.0 into main'"
    echo "  git tag -a v1.0.0 -m 'Release version 1.0.0'"
    echo ""
    echo "  # Merge into develop (back-merge release fixes)"
    echo "  git switch develop"
    echo "  git merge --no-ff release/1.0.0 -m 'merge: release/1.0.0 back into develop'"
    echo "  git branch -d release/1.0.0"
    echo ""
    echo "  # Step 4: Simulate a hotfix"
    echo "  # Why: Hotfixes branch from main (production), not develop"
    echo "  git switch main"
    echo "  git switch -c hotfix/1.0.1"
    echo "  echo 'critical fix' >> user.py"
    echo "  git add . && git commit -m 'fix: critical security patch'"
    echo ""
    echo "  # Finish hotfix: merge into BOTH main and develop"
    echo "  git switch main"
    echo "  git merge --no-ff hotfix/1.0.1 -m 'merge: hotfix/1.0.1 into main'"
    echo "  git tag -a v1.0.1 -m 'Hotfix 1.0.1'"
    echo ""
    echo "  git switch develop"
    echo "  git merge --no-ff hotfix/1.0.1 -m 'merge: hotfix/1.0.1 back into develop'"
    echo "  git branch -d hotfix/1.0.1"
    echo ""
    echo "  # Step 5: View the resulting branch history"
    echo "  git log --oneline --graph --all"
    echo ""
    echo "  # Expected graph structure:"
    echo "  #   main:    init -- [release merge v1.0.0] -- [hotfix merge v1.0.1]"
    echo "  #   develop: init -- [feature merge] -- [release back-merge] -- [hotfix back-merge]"
    echo "  #   Tags:    v1.0.0 on release merge, v1.0.1 on hotfix merge"
}

# === Exercise 2: GitHub Flow with Branch Protection ===
# Problem: Set up branch protection, create feature branch, open PR,
# go through review cycle, and clean up.
exercise_2() {
    echo "=== Exercise 2: GitHub Flow with Branch Protection ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Configure branch protection on GitHub"
    echo "  # Repository -> Settings -> Branches -> Add branch protection rule"
    echo "  # Branch name pattern: 'main'"
    echo "  # Enable:"
    echo "  #   - Require a pull request before merging"
    echo "  #     - Require approvals: 1"
    echo "  #   - Require status checks to pass"
    echo "  #   - Automatically delete head branches (optional but recommended)"
    echo ""
    echo "  # Step 2: Create a feature branch and open a PR"
    echo "  git switch main && git pull origin main"
    echo "  git switch -c feature/dark-mode"
    echo "  echo 'Dark mode CSS' > dark-mode.css"
    echo "  git add . && git commit -m 'feat: add dark mode styles'"
    echo "  git push -u origin feature/dark-mode"
    echo ""
    echo "  # Open PR via GitHub CLI or web UI"
    echo "  # gh pr create --title 'feat: add dark mode' --body '## Changes\n- Dark mode CSS'"
    echo ""
    echo "  # Step 3: Review cycle"
    echo "  # Reviewer approves the PR on GitHub"
    echo "  # CI status checks pass"
    echo "  # Merge the PR (squash and merge recommended for clean history)"
    echo ""
    echo "  # Step 4: Clean up after merge"
    echo "  # Why: Delete remote branch to keep the branch list clean"
    echo "  git push origin --delete feature/dark-mode"
    echo "  git switch main && git pull"
    echo "  git branch -d feature/dark-mode    # Delete local branch"
    echo ""
    echo "  # Step 5: Create PR template"
    echo "  mkdir -p .github"
    echo "  cat > .github/pull_request_template.md << 'EOF'"
    echo "## Changes"
    echo "<!-- Describe what changed in this PR -->"
    echo ""
    echo "## Reason for Changes"
    echo "<!-- Explain why this change is needed -->"
    echo ""
    echo "## Testing"
    echo "- [ ] Unit tests added/updated"
    echo "- [ ] Integration tests added/updated"
    echo "- [ ] Manual testing complete"
    echo ""
    echo "## Checklist"
    echo "- [ ] Follows code style guidelines"
    echo "- [ ] Documentation updated (if needed)"
    echo "- [ ] No breaking changes (or explained)"
    echo ""
    echo "## Related Issues"
    echo "Closes #"
    echo "EOF"
    echo "  git add .github/pull_request_template.md"
    echo "  git commit -m 'chore: add PR template'"
    echo "  git push"
}

# === Exercise 3: Feature Flag Implementation ===
# Problem: Implement a feature flag system with consistent hash-based rollout.
exercise_3() {
    echo "=== Exercise 3: Feature Flag Implementation ==="
    echo ""
    echo "Solution (Python implementation):"
    echo ""
    cat << 'PYTHON'
import hashlib

def is_feature_enabled(feature_name: str, user_id: str, rollout_percentage: int) -> bool:
    """Determine if a feature is enabled for a given user.

    Uses a consistent hash of user_id + feature_name so the same user
    always gets the same result (no random flipping between requests).

    Args:
        feature_name: Name of the feature flag
        user_id: Unique identifier for the user
        rollout_percentage: 0-100, percentage of users who should see the feature

    Returns:
        True if the feature is enabled for this user
    """
    # Why: Combining user_id and feature_name ensures different features
    # have independent rollout groups (a user might get feature A but not B)
    key = f"{user_id}:{feature_name}"
    hash_value = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    bucket = hash_value % 100  # Map to 0-99
    return bucket < rollout_percentage


# Usage with feature flag
def render_checkout(user_id: str):
    if is_feature_enabled("new_checkout", user_id, 25):
        return "Rendering NEW checkout (feature flag ON)"
    return "Rendering OLD checkout (feature flag OFF)"


# Test: Verify approximately 25% of 1000 users get the feature
if __name__ == "__main__":
    enabled_count = sum(
        is_feature_enabled("new_checkout", f"user-{i}", 25)
        for i in range(1000)
    )
    percentage = enabled_count / 10  # Out of 1000
    print(f"Feature enabled for {enabled_count}/1000 users ({percentage:.1f}%)")
    # Expected: approximately 250 users (25%), typically within 220-280
    assert 200 < enabled_count < 300, f"Expected ~250, got {enabled_count}"
    print("Test passed!")
PYTHON
    echo ""
    echo "  # Save the above as feature_flags.py and run:"
    echo "  # python feature_flags.py"
}

# === Exercise 4: Workflow Selection Justification ===
# Problem: Choose the right workflow for three different scenarios.
exercise_4() {
    echo "=== Exercise 4: Workflow Selection Justification ==="
    echo ""
    echo "Scenario 1: A 3-person startup building a web SaaS product with daily deployments"
    echo "  Recommended: GitHub Flow"
    echo "  Justification: With only 3 developers, the simplicity of GitHub Flow is ideal."
    echo "  Daily deployments mean there is no need for release branches or version"
    echo "  management. The main branch should always be deployable. Short-lived feature"
    echo "  branches with PRs provide enough code review without the overhead of Git Flow's"
    echo "  multi-branch structure. The team is small enough that coordination is easy."
    echo ""
    echo "Scenario 2: A 50-person enterprise team releasing a mobile app every 6 weeks"
    echo "  Recommended: Git Flow"
    echo "  Justification: Mobile apps require formal release cycles because app store"
    echo "  submissions take days and hotfixes must target specific released versions."
    echo "  Git Flow's release branches allow the team to stabilize a version while"
    echo "  continuing development on the next release. Hotfix branches enable emergency"
    echo "  patches to production without disrupting the release pipeline. The larger team"
    echo "  benefits from the structured branch roles (develop, release, hotfix)."
    echo ""
    echo "Scenario 3: A large tech company with 200 engineers, microservices, and feature flags"
    echo "  Recommended: Trunk-based Development"
    echo "  Justification: At this scale, long-lived branches create integration nightmares."
    echo "  Trunk-based development with feature flags allows 200 engineers to commit to"
    echo "  main frequently (via short-lived branches and PRs), preventing divergence."
    echo "  Feature flags decouple deployment from feature release, enabling continuous"
    echo "  deployment of the main branch while hiding incomplete features. This requires"
    echo "  mature CI/CD infrastructure, which a large tech company can invest in."
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
echo "All exercises completed!"
