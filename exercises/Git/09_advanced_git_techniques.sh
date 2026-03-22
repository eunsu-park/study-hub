#!/bin/bash
# Exercises for Lesson 09: Advanced Git Techniques
# Topic: Git
# Solutions to practice problems from the lesson.

# === Exercise 1: Write and Test a commit-msg Hook ===
# Problem: Create a commit-msg hook that enforces Conventional Commits format.
# Test with invalid and valid messages. Share via core.hooksPath.
exercise_1() {
    echo "=== Exercise 1: Write and Test a commit-msg Hook ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "  # Step 1: Create the hook file"
    echo "  touch .git/hooks/commit-msg"
    echo "  chmod +x .git/hooks/commit-msg"
    echo ""
    echo "  # Step 2: Write the hook script"
    cat << 'HOOK'
  # Contents of .git/hooks/commit-msg:

  #!/bin/bash
  COMMIT_MSG_FILE=$1
  COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

  # Regex: must start with one of the allowed types, optionally followed by (scope)
  # Why: Conventional Commits provide a standardized, machine-readable commit history
  PATTERN="^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .{1,100}$"

  FIRST_LINE=$(echo "$COMMIT_MSG" | head -1)

  if ! echo "$FIRST_LINE" | grep -qE "$PATTERN"; then
      echo "ERROR: Invalid commit message format!"
      echo ""
      echo "Must match: <type>(<optional-scope>): <description>"
      echo "Allowed types: feat, fix, docs, style, refactor, test, chore"
      echo ""
      echo "Examples:"
      echo "  feat(auth): add login endpoint"
      echo "  fix: resolve null pointer in parser"
      echo "  docs: update API documentation"
      echo ""
      echo "Your message was: '$FIRST_LINE'"
      exit 1
  fi

  # Check line length
  if [ ${#FIRST_LINE} -gt 72 ]; then
      echo "ERROR: First line must be 72 characters or fewer (got ${#FIRST_LINE})"
      exit 1
  fi

  exit 0
HOOK
    echo ""
    echo "  # Step 3: Test with an INVALID message"
    echo "  echo 'test file' > test.txt"
    echo "  git add test.txt"
    echo "  git commit -m 'updated stuff'"
    echo "  # Expected: Hook rejects the commit with an error message"
    echo ""
    echo "  # Step 4: Test with a VALID message"
    echo "  git commit -m 'feat: add login endpoint'"
    echo "  # Expected: Commit succeeds"
    echo ""
    echo "  # Step 5: Share the hook with the team via core.hooksPath"
    echo "  # Why: .git/hooks/ is not tracked by Git, so teammates do not get your hooks."
    echo "  # By putting hooks in a tracked directory and pointing Git to it, everyone shares them."
    echo "  mkdir -p .githooks"
    echo "  cp .git/hooks/commit-msg .githooks/commit-msg"
    echo "  chmod +x .githooks/commit-msg"
    echo "  git add .githooks/"
    echo "  git commit -m 'chore: add shared commit-msg hook'"
    echo ""
    echo "  # Configure Git to use the tracked hooks directory"
    echo "  git config core.hooksPath .githooks"
    echo ""
    echo "  # Teammates run this after cloning:"
    echo "  git config core.hooksPath .githooks"
    echo "  # Or add it to a setup script / Makefile"
}

# === Exercise 2: Submodule Lifecycle ===
# Problem: Create a library repo, add it as a submodule to a main project,
# clone recursively, update the submodule pointer.
exercise_2() {
    echo "=== Exercise 2: Submodule Lifecycle ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Create the 'library' repository"
    echo "  mkdir my-library && cd my-library"
    echo "  git init"
    echo "  cat > hello.py << 'EOF'"
    echo "def hello():"
    echo "    return 'Hello from the library!'"
    echo "EOF"
    echo "  git add hello.py"
    echo "  git commit -m 'feat: add hello function'"
    echo "  # Push to a remote (e.g., GitHub) so it can be referenced"
    echo "  git remote add origin git@github.com:<user>/my-library.git"
    echo "  git push -u origin main"
    echo "  cd .."
    echo ""
    echo "  # Step 2: Create 'main project' and add library as submodule"
    echo "  mkdir my-project && cd my-project"
    echo "  git init"
    echo "  echo '# Main Project' > README.md"
    echo "  git add . && git commit -m 'initial commit'"
    echo ""
    echo "  # Why: submodule add clones the library into a subdirectory and records"
    echo "  # the exact commit SHA, so the main project is pinned to a specific version"
    echo "  git submodule add git@github.com:<user>/my-library.git libs/my-library"
    echo "  git commit -m 'chore: add my-library as submodule'"
    echo "  git push -u origin main"
    echo "  cd .."
    echo ""
    echo "  # Step 3: Clone the main project into a fresh directory (with --recursive)"
    echo "  # Why: --recursive initializes and clones all submodules in one step"
    echo "  git clone --recursive git@github.com:<user>/my-project.git my-project-fresh"
    echo "  ls my-project-fresh/libs/my-library/hello.py    # Should exist"
    echo ""
    echo "  # Step 4: Update the library and update the submodule pointer"
    echo "  cd my-library"
    echo "  cat >> hello.py << 'EOF'"
    echo ""
    echo "def goodbye():"
    echo "    return 'Goodbye from the library!'"
    echo "EOF"
    echo "  git add hello.py"
    echo "  git commit -m 'feat: add goodbye function'"
    echo "  git push"
    echo "  cd .."
    echo ""
    echo "  # In the main project, update the submodule to the latest commit"
    echo "  cd my-project"
    echo "  # Why: --remote fetches the latest from the submodule's remote branch"
    echo "  git submodule update --remote libs/my-library"
    echo "  # The submodule pointer now references the new commit"
    echo "  git add libs/my-library"
    echo "  git commit -m 'chore: update my-library submodule to latest'"
    echo "  git push"
}

# === Exercise 3: Worktree for Parallel Work ===
# Problem: While working on a feature, create a worktree for a hotfix,
# fix the bug, and clean up -- without stashing anything.
exercise_3() {
    echo "=== Exercise 3: Worktree for Parallel Work ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # You are currently on feature/big-refactor with uncommitted changes"
    echo "  # An urgent bug arrives. Normally you would stash, but worktrees are cleaner."
    echo ""
    echo "  # Step 1: Create a new worktree for main at ../hotfix-wt"
    echo "  # Why: This creates a SEPARATE working directory pointing to main"
    echo "  # Your current directory (with uncommitted changes) is untouched"
    echo "  git worktree add ../hotfix-wt main"
    echo ""
    echo "  # Step 2: In the new worktree, create a hotfix branch and fix the bug"
    echo "  cd ../hotfix-wt"
    echo "  git switch -c hotfix/urgent-fix"
    echo "  echo 'bug fixed' > bugfix.txt"
    echo "  git add . && git commit -m 'fix: urgent production bug'"
    echo "  git push -u origin hotfix/urgent-fix"
    echo "  # Open PR, get it reviewed, merge on GitHub"
    echo ""
    echo "  # Step 3: Return to the main worktree and continue feature work"
    echo "  cd ../my-project    # Back to original worktree"
    echo "  # Your uncommitted changes on feature/big-refactor are still here!"
    echo "  git status           # Shows your WIP changes, untouched"
    echo ""
    echo "  # Step 4: Remove the hotfix worktree"
    echo "  # Why: Worktrees consume disk space; remove them after use"
    echo "  git worktree remove ../hotfix-wt"
    echo "  git worktree list    # Confirm only the main worktree remains"
    echo ""
    echo "Key advantages over stashing:"
    echo "  1. No risk of stash conflicts when restoring"
    echo "  2. Both working directories exist simultaneously (you can diff between them)"
    echo "  3. No need to remember what was stashed or in what order"
    echo "  4. Each worktree has its own index (staging area)"
}

# === Exercise 4: Automate Bug Finding with git bisect ===
# Problem: Create 10 commits, introduce a bug in commit #6,
# write a test script, and use git bisect run to find it automatically.
exercise_4() {
    echo "=== Exercise 4: Automate Bug Finding with git bisect ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Create a repository with 10 commits"
    echo "  mkdir bisect-exercise && cd bisect-exercise"
    echo "  git init"
    echo ""
    echo "  # Commits 1-5: function works correctly"
    echo "  for i in 1 2 3 4 5; do"
    echo "    echo 'def is_valid(): return True' > checker.py"
    echo "    git add . && git commit -m \"commit \$i: working code\""
    echo "  done"
    echo ""
    echo "  # Commit 6: INTRODUCE THE BUG"
    echo "  echo 'def is_valid(): return False' > checker.py"
    echo "  git add . && git commit -m 'commit 6: refactor checker (introduces bug)'"
    echo ""
    echo "  # Commits 7-10: bug persists (unrelated changes)"
    echo "  for i in 7 8 9 10; do"
    echo "    echo \"# Comment \$i\" >> checker.py"
    echo "    git add . && git commit -m \"commit \$i: add comment\""
    echo "  done"
    echo ""
    echo "  # Step 2: Write a test script"
    echo "  cat > test.sh << 'EOF'"
    echo "#!/bin/bash"
    echo "# Exit 0 = good (no bug), Exit 1 = bad (bug present)"
    echo "python3 -c 'import checker; exit(0 if checker.is_valid() else 1)'"
    echo "EOF"
    echo "  chmod +x test.sh"
    echo ""
    echo "  # Step 3: Run automated bisect"
    echo "  # Why: bisect performs binary search across commits, cutting search space in half each time"
    echo "  git bisect start"
    echo "  git bisect bad HEAD          # Current (commit 10) has the bug"
    echo "  git bisect good HEAD~9       # Commit 1 was known good"
    echo ""
    echo "  # Why: 'run' automates the good/bad decision using the test script"
    echo "  git bisect run ./test.sh"
    echo ""
    echo "  # Expected output:"
    echo "  # '<hash> is the first bad commit'"
    echo "  # commit message: 'commit 6: refactor checker (introduces bug)'"
    echo ""
    echo "  # Step 4: Confirm and exit"
    echo "  # The identified commit should be commit #6"
    echo "  git bisect reset    # Return to the original HEAD"
    echo ""
    echo "  # Binary search efficiency: instead of checking all 10 commits,"
    echo "  # bisect checks only ~4 (log2(10)) commits to find the culprit."
}

# === Exercise 5: Inspect Git Internals ===
# Problem: Run plumbing commands and describe what each output means.
exercise_5() {
    echo "=== Exercise 5: Inspect Git Internals ==="
    echo ""
    echo "Solution commands and explanations:"
    echo ""
    echo "  # Command 1: git cat-file -t HEAD"
    echo "  # What it does: Shows the TYPE of the Git object that HEAD points to"
    echo "  # Output: 'commit'"
    echo "  # Explanation: HEAD is a symbolic reference to the current branch tip,"
    echo "  # which is always a commit object. Git has four object types:"
    echo "  # blob (file content), tree (directory), commit, and tag."
    echo ""
    echo "  # Command 2: git cat-file -p HEAD"
    echo "  # What it does: Pretty-prints the CONTENT of the commit object"
    echo "  # Output example:"
    echo "  #   tree abc123def456..."
    echo "  #   parent 789abc012def..."
    echo "  #   author John <john@example.com> 1706000000 +0900"
    echo "  #   committer John <john@example.com> 1706000000 +0900"
    echo "  #"
    echo "  #   feat: add login endpoint"
    echo "  # Explanation: A commit object contains:"
    echo "  #   - tree: SHA of the root tree (snapshot of all files)"
    echo "  #   - parent: SHA of the previous commit(s)"
    echo "  #   - author: who wrote the change + timestamp"
    echo "  #   - committer: who committed it + timestamp"
    echo "  #   - message: the commit message body"
    echo ""
    echo "  # Command 3: git cat-file -p HEAD^{tree}"
    echo "  # What it does: Pretty-prints the root TREE object of the current commit"
    echo "  # Output example:"
    echo "  #   100644 blob abc123... README.md"
    echo "  #   100644 blob def456... main.py"
    echo "  #   040000 tree ghi789... src"
    echo "  # Explanation: A tree object lists the entries in a directory:"
    echo "  #   - 100644 = regular file permissions"
    echo "  #   - 040000 = subdirectory"
    echo "  #   - blob = file content object"
    echo "  #   - tree = subdirectory (another tree object)"
    echo "  #   This is how Git stores the directory structure at each commit."
    echo ""
    echo "  # Command 4: git rev-parse HEAD"
    echo "  # What it does: Resolves HEAD to its full 40-character SHA-1 hash"
    echo "  # Output: e.g., 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0'"
    echo "  # Explanation: rev-parse converts human-readable references (HEAD, main,"
    echo "  #   v1.0.0, HEAD~3) into the exact SHA that Git uses internally."
    echo "  # When to use in scripts:"
    echo "  #   CURRENT_SHA=\$(git rev-parse HEAD)"
    echo "  #   SHORT_SHA=\$(git rev-parse --short HEAD)"
    echo "  #   # Useful for build tags, deployment labels, cache keys"
    echo "  #   # Example: docker build -t myapp:\$SHORT_SHA ."
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
