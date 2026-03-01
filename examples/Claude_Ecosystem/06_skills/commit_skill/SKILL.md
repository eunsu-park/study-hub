# /commit — Smart Commit Skill

Create a well-structured git commit with an AI-generated message.

## Instructions

1. Run `git status` and `git diff --staged` to see what's being committed
2. If nothing is staged, stage all modified files (but warn about untracked files)
3. Analyze the changes and generate a commit message following Conventional Commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding/updating tests
   - `chore:` for maintenance tasks
4. The commit message should:
   - Have a concise subject line (max 72 characters)
   - Include a body explaining WHY the change was made (not WHAT)
   - Reference issue numbers if found in branch name
5. Show the proposed commit message and ask for confirmation
6. Create the commit

## Example

```
feat: add user avatar upload endpoint

Enables users to upload profile avatars with automatic resizing.
Supports JPEG, PNG, and WebP formats up to 5MB.

Closes #142
```
