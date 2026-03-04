# Effective Prompt Patterns for Claude Code

## Pattern 1: Targeted Fix

```
Fix the [error type] in [file_path].
The error is: [error message].
Run tests after fixing.
```

**Why it works**: Provides the exact error, file, and verification step.

## Pattern 2: Explore-Then-Act

```
First, understand how [feature] works in this codebase.
Read the relevant files.
Then [action].
```

**Why it works**: Claude explores before modifying, reducing mistakes.

## Pattern 3: Incremental Refactor

```
Refactor [target] in [file] to use [pattern].
Keep the public API unchanged.
Update tests if needed.
```

**Why it works**: Clear constraints prevent over-engineering.

## Pattern 4: Structured Code Review

```
Review [file] for:
1) Security issues
2) Performance problems
3) Error handling gaps
4) Test coverage
Suggest fixes with code.
```

**Why it works**: Structured checklist ensures thoroughness.

## Pattern 5: Constraint-First Implementation

```
Add [feature] to [file].
Constraints:
- No new dependencies
- Must be backward compatible
- Include unit tests
- Follow existing patterns in the codebase
```

**Why it works**: Explicit constraints guide Claude's decisions.

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Better Alternative |
|---|---|---|
| "Fix it." | Too vague | "Fix the TypeError in src/auth.py:42." |
| "Rewrite everything" | Unbounded scope | "Refactor the login function to use async." |
| "Make it better" | No criteria | "Improve performance of get_users by adding pagination." |
| Multiple unrelated tasks | Context confusion | Break into separate prompts |
