# /review — Code Review Skill

Perform a thorough code review on the current changes or a specific file.

## Arguments

- `<file>` (optional): Specific file to review. If omitted, reviews all staged/modified files.

## Instructions

1. Identify the files to review:
   - If a file argument is provided, review that file
   - Otherwise, run `git diff` to find all modified files
2. For each file, analyze:
   - **Correctness**: Logic errors, edge cases, off-by-one errors
   - **Security**: SQL injection, XSS, command injection, hardcoded secrets
   - **Performance**: N+1 queries, unnecessary loops, missing indexes
   - **Style**: Naming conventions, code organization, DRY violations
   - **Testing**: Missing test coverage for new code paths
3. Categorize findings by severity:
   - 🔴 **Critical**: Must fix before merging (bugs, security issues)
   - 🟡 **Warning**: Should fix (performance, maintainability)
   - 🔵 **Suggestion**: Nice to have (style, minor improvements)
4. Present findings in a structured format
5. Optionally offer to fix critical and warning issues

## Output Format

```
## Code Review: <filename>

### 🔴 Critical
- Line 42: SQL injection vulnerability in user query
  → Use parameterized queries instead of string concatenation

### 🟡 Warning
- Line 87: N+1 query pattern in the loop
  → Consider using eager loading or batch fetching

### 🔵 Suggestion
- Line 15: Variable name `x` is not descriptive
  → Consider renaming to `user_count`

### Summary
- Critical: 1 | Warning: 1 | Suggestion: 1
```
