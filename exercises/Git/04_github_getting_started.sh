#!/bin/bash
# Exercises for Lesson 04: Getting Started with GitHub
# Topic: Git
# Solutions to practice problems from the lesson.

# === Exercise 1: SSH Key Setup ===
# Problem: Generate an SSH key pair and register the public key with GitHub.
# Verify the connection using ssh -T git@github.com.
exercise_1() {
    echo "=== Exercise 1: SSH Key Setup ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Generate an Ed25519 SSH key pair"
    echo "  # Why: Ed25519 is the modern, recommended algorithm (shorter, faster, more secure than RSA)"
    echo "  ssh-keygen -t ed25519 -C \"your_email@example.com\""
    echo "  # Press Enter 3 times to accept defaults (file location + no passphrase)"
    echo "  # Or set a passphrase for extra security"
    echo ""
    echo "  # Step 2: Display the public key so you can copy it"
    echo "  cat ~/.ssh/id_ed25519.pub"
    echo ""
    echo "  # Step 3: Register on GitHub"
    echo "  # Go to: GitHub -> Settings -> SSH and GPG keys -> New SSH key"
    echo "  # Paste the entire output from Step 2 (starts with 'ssh-ed25519 ...')"
    echo ""
    echo "  # Step 4: Verify the connection"
    echo "  ssh -T git@github.com"
    echo "  # Expected: 'Hi <username>! You've successfully authenticated...'"
    echo ""
    echo "  # Optional: Start ssh-agent and add key (useful if you set a passphrase)"
    echo "  eval \"\$(ssh-agent -s)\""
    echo "  ssh-add ~/.ssh/id_ed25519"
    echo ""

    # Safe read-only verification: check if an SSH key already exists
    if [ -f ~/.ssh/id_ed25519.pub ]; then
        echo "[INFO] An Ed25519 SSH key already exists on this machine."
    elif [ -f ~/.ssh/id_rsa.pub ]; then
        echo "[INFO] An RSA SSH key exists. Consider upgrading to Ed25519."
    else
        echo "[INFO] No SSH key found. Follow the steps above to generate one."
    fi
}

# === Exercise 2: Create and Push a Repository ===
# Problem: Initialize a local repo, create README.md, connect to GitHub, and push.
exercise_2() {
    echo "=== Exercise 2: Create and Push a Repository ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Initialize a new local repository"
    echo "  mkdir my-first-remote"
    echo "  cd my-first-remote"
    echo "  git init"
    echo ""
    echo "  # Step 2: Create README.md with a project description"
    echo "  echo '# My First Remote Repository' > README.md"
    echo "  echo '' >> README.md"
    echo "  echo 'A practice project for learning GitHub push workflows.' >> README.md"
    echo ""
    echo "  # Step 3: Stage and commit with an appropriate message"
    echo "  # Why: A descriptive first commit helps collaborators understand the project"
    echo "  git add README.md"
    echo "  git commit -m 'docs: initialize project with README'"
    echo ""
    echo "  # Step 4: Create an EMPTY repository on GitHub (no README, no .gitignore, no license)"
    echo "  # Why: If GitHub adds files, it creates commits that conflict with local history"
    echo ""
    echo "  # Step 5: Connect local repo to remote and push"
    echo "  # Why: -u sets the upstream tracking so future 'git push' just works"
    echo "  git remote add origin git@github.com:<username>/my-first-remote.git"
    echo "  git push -u origin main"
    echo ""
    echo "  # Step 6: Confirm on GitHub by visiting the repository page"
    echo "  # The README.md should render automatically on the repo homepage"
}

# === Exercise 3: Fetch vs Pull Exploration ===
# Problem: Use git fetch + inspect + merge instead of plain git pull.
# Explain why you might prefer this workflow.
exercise_3() {
    echo "=== Exercise 3: Fetch vs Pull Exploration ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Simulate a remote change"
    echo "  # On GitHub web editor, edit a file and commit directly to main"
    echo ""
    echo "  # Step 2: Fetch remote changes WITHOUT merging"
    echo "  # Why: fetch downloads new commits but does NOT modify your working directory"
    echo "  git fetch origin"
    echo ""
    echo "  # Step 3: Inspect what changed on the remote"
    echo "  # Why: This lets you review changes before integrating them"
    echo "  git log origin/main --oneline -5"
    echo "  git log HEAD..origin/main --oneline     # Show only new remote commits"
    echo "  git diff HEAD origin/main                # See the actual changes"
    echo ""
    echo "  # Step 4: After reviewing, merge when ready"
    echo "  git merge origin/main"
    echo ""
    echo "Why prefer fetch + inspect + merge over plain pull?"
    echo ""
    echo "  1. SAFETY: You can review what changed before integrating, avoiding surprises"
    echo "  2. CONTROL: If the remote changes conflict with your local work, you can"
    echo "     prepare (stash, commit, or rebase) before merging"
    echo "  3. UNDERSTANDING: 'git pull' is just fetch+merge in one step -- if a conflict"
    echo "     occurs during pull, you may not know what caused it. Separating the steps"
    echo "     gives you time to understand the incoming changes"
    echo "  4. REBASE OPTION: After fetching, you can choose to rebase instead of merge"
    echo "     (git rebase origin/main) for a cleaner linear history"
}

# === Exercise 4: Remote Branch Workflow ===
# Problem: Create a branch on GitHub, fetch it locally, work on it,
# push changes, then delete the remote branch.
exercise_4() {
    echo "=== Exercise 4: Remote Branch Workflow ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Create branch 'feature/experiment' on GitHub web UI"
    echo "  # Go to the repo -> Click branch dropdown -> Type 'feature/experiment' -> Create"
    echo ""
    echo "  # Step 2: Fetch the new branch from remote"
    echo "  # Why: Your local repo does not yet know about the new remote branch"
    echo "  git fetch origin"
    echo ""
    echo "  # Step 3: Check out the remote branch locally"
    echo "  # Why: -c creates a local tracking branch from the remote one"
    echo "  git switch -c feature/experiment origin/feature/experiment"
    echo ""
    echo "  # Step 4: Make a small change, commit, and push"
    echo "  echo 'Experiment note' >> notes.txt"
    echo "  git add notes.txt"
    echo "  git commit -m 'feat: add experiment notes'"
    echo "  git push origin feature/experiment"
    echo ""
    echo "  # Step 5: Delete the remote branch"
    echo "  # Why: --delete removes the branch from the remote server"
    echo "  git push origin --delete feature/experiment"
    echo ""
    echo "  # Step 6: Confirm the remote branch is gone"
    echo "  git fetch --prune   # Clean up stale remote-tracking references"
    echo "  git branch -r       # Should no longer show origin/feature/experiment"
    echo ""
    echo "  # Step 7: Clean up the local branch too"
    echo "  git switch main"
    echo "  git branch -d feature/experiment"
}

# === Exercise 5: Resolving a Push Rejection ===
# Problem: Simulate a push rejection by cloning into two directories,
# pushing from both, and resolving the conflict.
exercise_5() {
    echo "=== Exercise 5: Resolving a Push Rejection ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Clone the same repository into two directories"
    echo "  git clone git@github.com:<username>/my-repo.git clone-a"
    echo "  git clone git@github.com:<username>/my-repo.git clone-b"
    echo ""
    echo "  # Step 2: In clone-a, make a commit and push"
    echo "  cd clone-a"
    echo "  echo 'Change from A' >> shared.txt"
    echo "  git add shared.txt"
    echo "  git commit -m 'feat: add change from clone A'"
    echo "  git push origin main"
    echo "  cd .."
    echo ""
    echo "  # Step 3: In clone-b, make a different commit and try to push"
    echo "  cd clone-b"
    echo "  echo 'Change from B' >> shared.txt"
    echo "  git add shared.txt"
    echo "  git commit -m 'feat: add change from clone B'"
    echo "  git push origin main"
    echo "  # ERROR: rejected! The remote has commits that clone-b does not have."
    echo "  # Why: Git rejects because pushing would lose clone-a's commit"
    echo ""
    echo "  # Step 4: Resolve by pulling first, handling conflicts, then pushing"
    echo "  git pull origin main"
    echo "  # If there is a conflict in shared.txt:"
    echo "  #   1. Open the file and resolve the conflict markers"
    echo "  #   2. git add shared.txt"
    echo "  #   3. git commit -m 'merge: resolve conflict between A and B changes'"
    echo "  # If no conflict (changes on different lines), the merge auto-completes"
    echo ""
    echo "  # Step 5: Push the resolved state"
    echo "  git push origin main"
    echo "  cd .."
    echo ""
    echo "Key takeaway:"
    echo "  Always pull (or fetch+merge) before pushing when working with a shared branch."
    echo "  The rejection is Git protecting you from accidentally overwriting teammates' work."
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
exercise_5
echo ""
echo "All exercises completed!"
