# Claude Projects and Artifacts — Examples

## Project Setup

### Creating a Project

1. Go to claude.ai → Projects → New Project
2. Set project name and description
3. Add knowledge documents (up to 200K tokens)

### Project Knowledge Structure

```
My API Project/
├── Custom Instructions:
│   "You are an API developer working on a FastAPI backend.
│    Follow REST conventions. Use Pydantic for validation.
│    Always include error handling and type hints."
│
├── Knowledge Documents:
│   ├── api_spec.yaml          — OpenAPI specification
│   ├── database_schema.sql    — Current DB schema
│   ├── coding_standards.md    — Team coding standards
│   └── architecture.md        — System architecture doc
│
└── Conversations:
    ├── "Design user auth endpoints"
    ├── "Add pagination to list endpoints"
    └── "Write integration tests"
```

## Artifact Examples

### Code Artifact

When Claude generates a substantial code block, it creates an Artifact:
- React components
- Python scripts
- HTML pages
- Configuration files

Artifacts can be:
- **Edited** inline with follow-up prompts
- **Copied** to clipboard
- **Downloaded** as files
- **Versioned** (see change history within the conversation)

### Document Artifact

Claude can create structured documents:
- Technical specifications
- API documentation
- Meeting notes
- Project proposals

### Comparison: Projects vs CLAUDE.md

| Feature | Projects (claude.ai) | CLAUDE.md (Claude Code) |
|---------|---------------------|------------------------|
| Platform | Web (claude.ai) | CLI / IDE |
| Persistence | Cloud-stored | File in repo |
| Sharing | Team members | Anyone with repo access |
| Knowledge | Upload files (200K) | Auto-reads codebase |
| Best for | Research, writing, planning | Coding, repo tasks |
