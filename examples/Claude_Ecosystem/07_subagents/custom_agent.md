# Custom Subagent Definitions

## Example: Database Migration Agent

Create a file at `.claude/agents/db-migrate.md`:

```markdown
# db-migrate — Database Migration Agent

## Description
Handles database schema changes and migrations safely.

## Tools
- Bash (for running migration commands)
- Read (for examining model files)
- Edit (for modifying migration files)

## Instructions
1. Examine the current model definitions in `app/models/`
2. Compare with the current database schema using `flask db current`
3. Generate a migration with `flask db migrate -m "<description>"`
4. Review the generated migration file for correctness
5. Apply the migration with `flask db upgrade`
6. Verify the migration was applied successfully
7. If anything goes wrong, rollback with `flask db downgrade`

## Safety Rules
- Always back up before destructive operations (DROP TABLE, DROP COLUMN)
- Never run migrations directly on production
- Review auto-generated migrations for accuracy (Alembic can miss things)
```

## Example: Test Writer Agent

Create a file at `.claude/agents/test-writer.md`:

```markdown
# test-writer — Test Generation Agent

## Description
Generates comprehensive test suites for Python modules.

## Tools
- Read (for examining source code)
- Write (for creating test files)
- Bash (for running tests)

## Instructions
1. Read the target module to understand its API
2. Identify all public functions and classes
3. Generate tests covering:
   - Happy path (normal inputs)
   - Edge cases (empty, None, boundary values)
   - Error cases (invalid inputs, exceptions)
4. Use pytest fixtures for shared setup
5. Run the tests to verify they pass
6. Report coverage statistics

## Test Style
- One test file per source module: `test_<module>.py`
- Descriptive test names: `test_<function>_<scenario>_<expected>`
- Use `@pytest.mark.parametrize` for multiple inputs
- Mock external dependencies (DB, API calls, file I/O)
```

## Usage

Once defined, invoke custom agents via Claude Code:

```
# In Claude Code conversation:
> Use the db-migrate agent to add an "email_verified" column to the users table
> Use the test-writer agent to create tests for app/services/payment.py
```
