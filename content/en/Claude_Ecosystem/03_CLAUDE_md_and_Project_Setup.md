# CLAUDE.md and Project Setup

**Previous**: [02. Claude Code: Getting Started](./02_Claude_Code_Getting_Started.md) | **Next**: [04. Permission Modes and Security](./04_Permission_Modes.md)

---

CLAUDE.md is the most important configuration file in the Claude Code ecosystem. It is a plain Markdown file at the root of your project that provides Claude with project-specific context — coding standards, architectural decisions, testing procedures, deployment notes, and anything else that helps Claude work effectively with your codebase. This lesson covers CLAUDE.md in depth, along with the broader `.claude/` directory structure and settings hierarchy.

**Difficulty**: ⭐

**Prerequisites**:
- [02. Claude Code: Getting Started](./02_Claude_Code_Getting_Started.md)
- Familiarity with Markdown syntax
- A project you want to configure for Claude Code

**Learning Objectives**:
- Understand why CLAUDE.md exists and how Claude uses it
- Write effective CLAUDE.md files for different project types
- Configure the `.claude/` directory structure
- Navigate the settings hierarchy (global, project, local)
- Make informed decisions about what to commit vs gitignore
- Apply best practices for team-shared Claude Code configuration

---

## Table of Contents

1. [Why CLAUDE.md Matters](#1-why-claudemd-matters)
2. [How Claude Reads CLAUDE.md](#2-how-claude-reads-claudemd)
3. [CLAUDE.md Structure and Sections](#3-claudemd-structure-and-sections)
4. [Real-World CLAUDE.md Examples](#4-real-world-claudemd-examples)
5. [The .claude/ Directory](#5-the-claude-directory)
6. [Settings Hierarchy](#6-settings-hierarchy)
7. [What to Commit vs Gitignore](#7-what-to-commit-vs-gitignore)
8. [Initializing a New Project](#8-initializing-a-new-project)
9. [CLAUDE.md vs Settings: When to Use Which](#9-claudemd-vs-settings-when-to-use-which)
10. [Best Practices](#10-best-practices)
11. [Exercises](#11-exercises)
12. [Next Steps](#12-next-steps)

---

## 1. Why CLAUDE.md Matters

Without CLAUDE.md, Claude Code starts every session with zero project-specific knowledge. It can read your code, but it does not know:

- Your coding standards (tabs vs spaces, naming conventions)
- How to run tests (`pytest`? `npm test`? `make test`?)
- Which directories contain what (is `src/` the main source? or `lib/`?)
- Architectural decisions and their rationale
- Deployment procedures and environment details
- Which patterns to follow and which to avoid

CLAUDE.md bridges this gap. It turns Claude from a generic AI assistant into a context-aware team member that follows your project's conventions.

### Before and After CLAUDE.md

**Without CLAUDE.md** — Claude guesses based on common patterns:

```
> Add a new API endpoint for user profiles

Claude might:
- Use a framework you're not using
- Put the file in the wrong directory
- Use different naming conventions than your project
- Skip your project's validation patterns
- Not run the tests you expect
```

**With CLAUDE.md** — Claude follows your exact conventions:

```markdown
# In your CLAUDE.md:
## API Conventions
- Routes go in `src/routes/<resource>.ts`
- Use Zod for request validation
- All endpoints must have corresponding tests in `tests/routes/`
- Follow RESTful naming: GET /api/users/:id, POST /api/users
- Run `npm test` after any route changes
```

```
> Add a new API endpoint for user profiles

Claude will:
- Create src/routes/profiles.ts (correct directory)
- Use Zod schemas (your validation library)
- Create tests/routes/profiles.test.ts
- Follow your RESTful patterns
- Run npm test to verify
```

---

## 2. How Claude Reads CLAUDE.md

When you start a Claude Code session, the following happens:

```
1. Claude Code detects the project root
2. It reads CLAUDE.md from the project root (if present)
3. It reads any parent directory CLAUDE.md files (for monorepos)
4. The contents are included in the system prompt
5. Claude uses these instructions throughout the session
```

### CLAUDE.md Discovery Order

```
~/projects/my-monorepo/          ← CLAUDE.md (read first)
├── packages/
│   ├── frontend/                ← CLAUDE.md (also read)
│   │   └── src/
│   └── backend/                 ← CLAUDE.md (also read)
│       └── src/
└── shared/                      ← No CLAUDE.md here
```

If you run `claude` from `~/projects/my-monorepo/packages/frontend/`, Claude reads:
1. `~/projects/my-monorepo/CLAUDE.md` (ancestor)
2. `~/projects/my-monorepo/packages/frontend/CLAUDE.md` (current)

The instructions are merged, with more specific (deeper) instructions taking precedence.

### Token Budget

CLAUDE.md content consumes tokens from your context window. A typical CLAUDE.md uses 500-3,000 tokens, which is a small fraction of the 200K window. However, extremely long CLAUDE.md files (10,000+ tokens) can be wasteful. Keep it concise and relevant.

---

## 3. CLAUDE.md Structure and Sections

A well-organized CLAUDE.md follows a consistent structure. Here is the recommended template:

```markdown
# Project Name

Brief description of what the project does and its purpose.

## Project Structure

```
src/
├── routes/     # API endpoints
├── services/   # Business logic
├── models/     # Database models
├── middleware/  # Express middleware
└── utils/      # Shared utilities
tests/
├── unit/       # Unit tests
├── integration/ # Integration tests
└── fixtures/   # Test data
```

## Tech Stack

- **Runtime**: Node.js 20 with TypeScript 5.3
- **Framework**: Express 4.18
- **Database**: PostgreSQL 16 with Prisma ORM
- **Testing**: Jest + Supertest
- **Linting**: ESLint + Prettier

## Development Commands

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run all tests
npm test

# Run specific test file
npm test -- tests/unit/auth.test.ts

# Lint and format
npm run lint
npm run format

# Build for production
npm run build
```

## Coding Standards

- Use TypeScript strict mode — no `any` types
- Prefer `const` over `let`; never use `var`
- Functions should be < 30 lines; extract helpers if longer
- Use named exports, not default exports
- Error handling: always use custom error classes from `src/errors/`

## API Conventions

- RESTful routes: `GET /api/resource`, `POST /api/resource`
- All request bodies validated with Zod schemas
- Response format: `{ data: T, meta?: { pagination } }`
- Authentication via JWT in Authorization header
- Rate limiting on all public endpoints

## Testing Requirements

- Every new feature must have unit tests
- Integration tests for API endpoints
- Test files mirror source structure: `src/foo.ts` → `tests/unit/foo.test.ts`
- Minimum 80% code coverage
- Run `npm test` before committing

## Git Conventions

- Branch naming: `feature/description`, `fix/description`, `chore/description`
- Commit messages: conventional commits (feat:, fix:, chore:, docs:)
- Always rebase on main before merging
- Squash merge for feature branches
```

### Section-by-Section Explanation

| Section | Purpose | Why It Matters |
|---------|---------|----------------|
| **Project Structure** | Directory layout | Claude puts files in the right place |
| **Tech Stack** | Languages, frameworks, versions | Claude uses the right APIs and patterns |
| **Development Commands** | How to build, test, run | Claude executes the correct commands |
| **Coding Standards** | Style rules, patterns | Claude follows your conventions |
| **API Conventions** | Endpoint design rules | Claude creates consistent APIs |
| **Testing Requirements** | What to test and how | Claude writes appropriate tests |
| **Git Conventions** | Commit and branch rules | Claude creates proper commits |

---

## 4. Real-World CLAUDE.md Examples

### Python Data Science Project

```markdown
# Data Pipeline

ETL pipeline for processing financial data from multiple exchanges.

## Project Structure

```
pipeline/
├── extractors/    # Data source connectors
├── transformers/  # Data cleaning and transformation
├── loaders/       # Database and file writers
├── models/        # SQLAlchemy models
├── config/        # Environment-specific config
└── tests/
```

## Environment

- Python 3.12, managed with pyenv
- Dependencies: pip + requirements.txt (NOT poetry)
- Virtual env: `source .venv/bin/activate`

## Commands

```bash
python -m pytest tests/ -v          # Run tests
python -m pytest tests/ --cov=pipeline  # With coverage
python -m mypy pipeline/            # Type checking
python -m ruff check pipeline/      # Linting
python -m ruff format pipeline/     # Formatting
```

## Code Style

- PEP 8 with 100-char line limit
- Type hints on ALL function signatures
- Docstrings: Google style
- Use `pathlib.Path` instead of `os.path`
- Use `logging` module, never `print()` in production code
- Dataclasses for DTOs, Pydantic for validation

## Important Notes

- NEVER commit `.env` files or API keys
- The `extractors/` module uses rate-limited API calls — always use `@retry` decorator
- Database migrations use Alembic: `alembic upgrade head`
- CI runs mypy + ruff + pytest; all must pass
```

### TypeScript Monorepo

```markdown
# Commerce Platform

Monorepo for the e-commerce platform (web, API, shared packages).

## Monorepo Structure

```
packages/
├── web/          # Next.js frontend (port 3000)
├── api/          # NestJS backend (port 4000)
├── shared/       # Shared types and utilities
├── ui/           # Shared React component library
└── config/       # Shared ESLint, TSConfig, etc.
```

## Package Manager

- pnpm (NOT npm or yarn)
- `pnpm install` from root
- `pnpm --filter @commerce/web dev` to run specific package

## Key Rules

1. Shared types go in `packages/shared/types/` — NEVER duplicate types
2. API responses must match the types in `shared/types/api.ts`
3. UI components must have Storybook stories
4. All packages use the shared ESLint config from `packages/config/`
5. Database queries use Prisma — raw SQL only for migrations

## Testing

```bash
pnpm test                           # All packages
pnpm --filter @commerce/api test    # API only
pnpm --filter @commerce/web e2e     # E2E tests (Playwright)
```
```

### C/C++ Embedded Project

```markdown
# Sensor Controller Firmware

Firmware for the STM32F4 sensor controller board.

## Build System

- CMake 3.25+
- ARM GCC toolchain: `arm-none-eabi-gcc`
- Build: `mkdir build && cd build && cmake .. && make`
- Flash: `make flash` (requires ST-Link)

## Code Style

- C11 standard, compiled with `-Wall -Wextra -Werror`
- 4-space indentation, no tabs
- Snake_case for functions and variables
- UPPER_SNAKE_CASE for macros and constants
- Hungarian notation for hardware registers: `reg_`, `pin_`, `irq_`

## Memory Constraints

- Flash: 512KB total, 380KB available
- RAM: 128KB total, 96KB available
- Stack size: 4KB per task
- NEVER use dynamic memory allocation (malloc/free)
- All buffers must be statically allocated

## Important

- ISR functions must be < 50 lines and non-blocking
- Use the HAL layer in `drivers/` — do NOT access registers directly
- FreeRTOS tasks in `tasks/`, each with a dedicated .c/.h pair
```

---

## 5. The .claude/ Directory

Beyond CLAUDE.md, Claude Code uses a `.claude/` directory for additional configuration.

### Directory Structure

```
.claude/
├── settings.json        # Project-level settings (committed)
├── settings.local.json  # Local overrides (gitignored)
└── skills/              # Project-specific skills (committed)
    ├── commit.md
    └── review.md
```

### settings.json

Project-level settings that apply to all team members:

```json
{
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Bash(npm run build)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push --force)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      }
    ]
  }
}
```

### settings.local.json

Personal overrides that should not be committed:

```json
{
  "permissions": {
    "allow": [
      "Bash(docker compose up)"
    ]
  },
  "model": "claude-opus-4-20250514"
}
```

---

## 6. Settings Hierarchy

Claude Code reads settings from three levels, with more specific settings overriding general ones.

### Precedence Order (Highest to Lowest)

```
┌─────────────────────────────────┐
│  1. Local Settings (highest)    │  .claude/settings.local.json
│     Personal overrides          │  NOT committed to git
├─────────────────────────────────┤
│  2. Project Settings            │  .claude/settings.json
│     Team-shared rules           │  Committed to git
├─────────────────────────────────┤
│  3. Global Settings (lowest)    │  ~/.claude/settings.json
│     User-wide defaults          │  Applies to all projects
└─────────────────────────────────┘
```

### Global Settings (~/.claude/settings.json)

Settings that apply across all your projects:

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf /)",
      "Bash(sudo *)"
    ]
  },
  "model": "claude-sonnet-4-20250514"
}
```

### How Merging Works

Settings are merged, not replaced. Allow and deny lists are concatenated. If the same permission appears in both allow and deny, deny takes precedence.

```
Global:    allow: ["Bash(git *)"]     deny: ["Bash(rm -rf *)"]
Project:   allow: ["Bash(npm test)"]  deny: ["Bash(git push --force)"]
Local:     allow: ["Bash(docker *)"]

Result:    allow: ["Bash(git *)", "Bash(npm test)", "Bash(docker *)"]
           deny:  ["Bash(rm -rf *)", "Bash(git push --force)"]
```

### Viewing Effective Settings

```bash
# Inside a Claude Code session
> /config

# Shows the merged effective configuration
# Including which file each setting comes from
```

---

## 7. What to Commit vs Gitignore

Making the right decisions about what to commit ensures your team shares a consistent experience while preserving individual preferences.

### Commit These Files

| File | Why |
|------|-----|
| `CLAUDE.md` | Project context shared by the whole team |
| `.claude/settings.json` | Shared permission rules and hooks |
| `.claude/skills/*.md` | Shared custom skills |

### Gitignore These Files

| File | Why |
|------|-----|
| `.claude/settings.local.json` | Personal preferences (model choice, extra permissions) |
| `.claude/credentials` | Authentication tokens |

### Recommended .gitignore Entry

```gitignore
# Claude Code local settings
.claude/settings.local.json
.claude/credentials
```

### Team Workflow

```
Developer A creates the project configuration:
  1. Writes CLAUDE.md with project conventions
  2. Creates .claude/settings.json with shared rules
  3. Commits both files

Developer B clones the repository:
  1. Runs `claude` — automatically picks up CLAUDE.md and settings
  2. Creates .claude/settings.local.json for personal preferences
  3. Works with the same conventions as Developer A
```

---

## 8. Initializing a New Project

### Using /init

The fastest way to create a CLAUDE.md for an existing project:

```bash
cd ~/projects/my-app
claude
```

```
> /init

Claude will:
1. Scan your project structure
2. Detect languages, frameworks, and tools
3. Read existing config files (package.json, pyproject.toml, etc.)
4. Generate a CLAUDE.md tailored to your project
5. Ask you to review and approve it
```

### Manual Creation

For more control, create CLAUDE.md manually:

```bash
# Create the file
touch CLAUDE.md

# Open in your editor
code CLAUDE.md

# Or let Claude help
claude -p "Read this project and create a comprehensive CLAUDE.md file"
```

### Starting from a Template

```bash
# Inside a Claude Code session
> Create a CLAUDE.md for this project. Include sections for:
  - Project structure
  - Tech stack with versions
  - Development commands (install, test, lint, build)
  - Coding standards
  - Git conventions
  - Any important architectural decisions you can infer from the code
```

---

## 9. CLAUDE.md vs Settings: When to Use Which

CLAUDE.md and settings files serve different purposes. Understanding the distinction is key to effective configuration.

### CLAUDE.md = Suggestive Context

CLAUDE.md provides **suggestions and context** that Claude uses as guidance. Claude aims to follow these instructions but may deviate if circumstances require it.

```markdown
# In CLAUDE.md — suggestions and context

## Coding Standards
- Use 4-space indentation
- Functions should be < 30 lines
- Always add type hints

## Architecture Notes
- We use a hexagonal architecture
- Domain logic must not depend on infrastructure
```

### Settings = Deterministic Rules

Settings files define **hard rules** that Claude Code enforces mechanically. Permissions are either allowed or denied — there is no room for interpretation.

```json
// In .claude/settings.json — deterministic rules
{
  "permissions": {
    "allow": ["Bash(npm test)"],
    "deny": ["Bash(rm -rf *)"]
  }
}
```

### Decision Guide

| Question | Use CLAUDE.md | Use Settings |
|----------|:------------:|:------------:|
| "Use PEP 8 style" | Yes | |
| "Never run `rm -rf`" | | Yes |
| "Prefer composition over inheritance" | Yes | |
| "Auto-format with Prettier after edits" | | Yes (hooks) |
| "Our API uses REST conventions" | Yes | |
| "Always allow `npm test`" | | Yes |
| "Database migrations require review" | Yes | |
| "Block all `sudo` commands" | | Yes |

---

## 10. Best Practices

### Keep CLAUDE.md Focused

```markdown
# Good: Specific, actionable instructions
## Testing
- Run `pytest tests/ -v` for all tests
- Run `pytest tests/unit/` for fast unit tests only
- Minimum 80% coverage required

# Bad: Vague, generic instructions
## Testing
- Write good tests
- Make sure things work
```

### Update CLAUDE.md as the Project Evolves

```markdown
# Good practice: date your major decisions

## Architecture Decisions
- **2025-01**: Migrated from REST to GraphQL for the public API
- **2025-03**: Switched from MongoDB to PostgreSQL
- **2025-06**: Adopted hexagonal architecture for the order service
```

### Use Comments for Rationale

```markdown
## Code Style

- Use `structuredClone()` instead of `JSON.parse(JSON.stringify())`
  (We target Node 18+ so structuredClone is always available)

- Never use `enum` in TypeScript — use `as const` objects instead
  (Enums generate confusing JavaScript and have type-narrowing issues)
```

### Layer CLAUDE.md in Monorepos

```
my-monorepo/
├── CLAUDE.md                   # Shared conventions, monorepo commands
├── packages/
│   ├── frontend/
│   │   └── CLAUDE.md           # React/Next.js specific instructions
│   ├── backend/
│   │   └── CLAUDE.md           # NestJS specific instructions
│   └── shared/
│       └── CLAUDE.md           # Shared library conventions
```

Root CLAUDE.md:
```markdown
# My Monorepo

## Package Manager
- Use pnpm (NOT npm or yarn)
- Always run commands from the monorepo root

## Shared Rules
- All packages must pass `pnpm lint` and `pnpm test`
- Shared types go in `packages/shared/`
```

Package CLAUDE.md:
```markdown
# Frontend Package

## Framework
- Next.js 14 with App Router (NOT Pages Router)
- Tailwind CSS for styling
- Server Components by default, 'use client' only when needed

## Testing
- `pnpm --filter frontend test` for unit tests
- `pnpm --filter frontend e2e` for Playwright tests
```

### Avoid Overly Long CLAUDE.md Files

Each token in CLAUDE.md is consumed from the context window on every turn. Aim for 100-300 lines. If your CLAUDE.md exceeds 500 lines, consider:

1. Moving detailed documentation to separate files and referencing them
2. Using sub-directory CLAUDE.md files for package-specific instructions
3. Keeping only the most frequently needed information in the root CLAUDE.md

---

## 11. Exercises

### Exercise 1: Create a CLAUDE.md

Take an existing project you work on and create a CLAUDE.md from scratch:

1. Document the project structure
2. List the tech stack with versions
3. Write the development commands (install, test, lint, build)
4. Define coding standards specific to your project
5. Add any architectural decisions or important notes
6. Keep it under 200 lines

### Exercise 2: Settings Configuration

Create a `.claude/settings.json` for a project with these requirements:

1. Allow running `npm test` and `npm run lint` without prompting
2. Deny all `rm -rf` commands
3. Deny `git push --force`
4. Allow all `git` commands except force push

### Exercise 3: Monorepo Configuration

Design a CLAUDE.md hierarchy for a monorepo with:

- A React frontend (`packages/web/`)
- A Python FastAPI backend (`packages/api/`)
- A shared protobuf definitions package (`packages/proto/`)

Write the root CLAUDE.md and each package's CLAUDE.md. Consider what goes in the root vs each package.

### Exercise 4: Review and Improve

If you already have a CLAUDE.md, review it against this lesson's best practices:

1. Is it specific enough that Claude will follow your conventions?
2. Does it include the commands Claude needs to run?
3. Is it concise enough to not waste context tokens?
4. Does it cover the situations where Claude most often makes mistakes?

---

## 12. Next Steps

With CLAUDE.md and project settings configured, your Claude Code sessions are now project-aware. The next lesson covers **Permission Modes** — the security model that controls what Claude Code can and cannot do. Understanding permissions is essential for balancing productivity with safety, especially when working on production codebases or in team environments.

**Next**: [04. Permission Modes and Security](./04_Permission_Modes.md)
