# Claude Cowork — Use Cases

## What is Cowork?

Cowork is Claude as a digital colleague — a persistent agent that can:
- Execute multi-step tasks across tools and services
- Connect to external services via MCP connectors
- Work across domains (code, docs, data, communication)

## Example Use Cases

### 1. Documentation Sprint

```
Prompt: "Review all public API endpoints in src/api/ and update the
         API documentation in docs/api.md. Add any missing endpoints
         and remove deprecated ones."

Cowork:
  1. Reads all route files in src/api/
  2. Extracts endpoint signatures, methods, and docstrings
  3. Reads current docs/api.md
  4. Diffs the two, identifies gaps
  5. Updates docs/api.md with new endpoints
  6. Removes deprecated entries
  7. Commits with descriptive message
```

### 2. Research and Summarize

```
Prompt: "Research the latest changes in React 19 and create a
         migration checklist for our project."

Cowork:
  1. Searches for React 19 release notes and migration guides
  2. Analyzes our project's React usage patterns
  3. Creates a prioritized migration checklist
  4. Saves to docs/react-19-migration.md
```

### 3. Project Management

```
Prompt: "Look at the open GitHub issues labeled 'bug' and create
         a prioritized list based on impact and effort."

Cowork:
  1. Fetches open issues via GitHub MCP connector
  2. Reads each issue's description and comments
  3. Estimates impact (affected users, severity)
  4. Estimates effort (code complexity, files involved)
  5. Creates prioritized table in docs/bug-triage.md
```

## MCP Connectors

Cowork extends its capabilities through MCP:
- **GitHub**: Issues, PRs, repo management
- **Slack**: Send/read messages, create threads
- **Google Drive**: Read/write documents and spreadsheets
- **Linear**: Create and manage project tickets
- **Notion**: Read and update wiki pages
