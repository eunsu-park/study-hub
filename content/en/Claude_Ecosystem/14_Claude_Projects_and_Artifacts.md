# Claude Projects and Artifacts

**Previous**: [13. Building Custom MCP Servers](./13_Building_MCP_Servers.md) | **Next**: [15. Claude API Fundamentals](./15_Claude_API_Fundamentals.md)

---

Claude Projects and Artifacts are features of the Claude web and desktop interfaces that transform Claude from a stateless chatbot into a persistent, context-aware workspace. Projects let you ground Claude's responses in your own documents and code, while Artifacts let Claude produce standalone content pieces that you can edit, iterate on, and export. This lesson covers both features in depth, including when to use Projects versus CLAUDE.md for engineering workflows.

**Difficulty**: ⭐

**Prerequisites**:
- Basic familiarity with Claude (web interface or desktop app)
- Understanding of how LLMs use context (helpful but not required)

**Learning Objectives**:
- Create and manage Claude Projects for organized workspaces
- Add documents, code, and custom instructions as project knowledge
- Understand how Claude uses project knowledge to ground its responses
- Create, edit, and iterate on Artifacts (code, documents, HTML, diagrams)
- Set up team collaboration with shared projects
- Choose between Projects and CLAUDE.md for different use cases
- Apply best practices for organizing knowledge in projects

---

## Table of Contents

1. [What Are Claude Projects?](#1-what-are-claude-projects)
2. [Creating and Managing Projects](#2-creating-and-managing-projects)
3. [Knowledge Grounding](#3-knowledge-grounding)
4. [Custom Instructions](#4-custom-instructions)
5. [Artifacts](#5-artifacts)
6. [Team Collaboration](#6-team-collaboration)
7. [Projects vs CLAUDE.md](#7-projects-vs-claudemd)
8. [Best Practices](#8-best-practices)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. What Are Claude Projects?

A **Project** in Claude is an organized workspace that persists across conversations. Think of it as a folder that contains:

- **Project knowledge**: Documents, code files, and data that Claude can reference
- **Custom instructions**: System-level guidance for how Claude should behave
- **Conversations**: Chat threads that share the project's context
- **Team access**: Collaboration settings for shared projects

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Project                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────────────────────────┐        │
│  │   Project     │   │       Project Knowledge          │        │
│  │   Settings    │   │  ┌────────────────────────────┐  │        │
│  │              │   │  │  design-spec.md             │  │        │
│  │  Name        │   │  │  api-schema.json            │  │        │
│  │  Description │   │  │  style-guide.pdf            │  │        │
│  │  Custom      │   │  │  database-schema.sql        │  │        │
│  │  Instructions│   │  │  component-library.tsx      │  │        │
│  └──────────────┘   │  └────────────────────────────┘  │        │
│                      └──────────────────────────────────┘        │
│                                                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │                 Conversations                          │      │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │      │
│  │  │ Chat #1:    │ │ Chat #2:    │ │ Chat #3:    │     │      │
│  │  │ API Design  │ │ Bug Fix     │ │ Code Review │     │      │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  All conversations share the project knowledge and instructions  │
└─────────────────────────────────────────────────────────────────┘
```

**Why use Projects?**
- **Persistent context**: Upload documents once, reference them across many conversations
- **Consistency**: Custom instructions ensure Claude behaves the same way every time
- **Organization**: Group related conversations together
- **Collaboration**: Share projects with team members (on Team and Enterprise plans)

---

## 2. Creating and Managing Projects

### 2.1 Creating a Project

On claude.ai or Claude Desktop:

1. Click the **Projects** icon in the sidebar (or press `P` on Desktop).
2. Click **Create Project**.
3. Give it a name and optional description.
4. Add project knowledge (documents, code files).
5. Write custom instructions (optional).
6. Start a conversation within the project.

### 2.2 Project Organization Strategies

Organize projects by scope and purpose:

```
Recommended Project Structure:
├── Product Projects
│   ├── "Mobile App v3"      ← One project per product/feature
│   ├── "Admin Dashboard"
│   └── "Payment System"
│
├── Role-Based Projects
│   ├── "Technical Writer"   ← Project with writing style guides
│   ├── "Code Reviewer"      ← Project with coding standards
│   └── "Data Analyst"       ← Project with schema docs + SQL patterns
│
├── Learning Projects
│   ├── "Rust Learning"      ← Project with Rust book excerpts
│   └── "ML Fundamentals"    ← Project with course notes
│
└── Reference Projects
    ├── "Company Standards"  ← Project with style guides, policies
    └── "API Reference"      ← Project with API documentation
```

### 2.3 Managing Projects

- **Star** frequently used projects for quick access.
- **Archive** completed projects to reduce clutter (they remain searchable).
- **Delete** projects you no longer need (this is permanent).
- **Duplicate** a project to create a template for similar workloads.

---

## 3. Knowledge Grounding

Knowledge grounding is the core feature that makes Projects powerful. By uploading documents to a project, you give Claude direct access to your specific information -- not just its training data.

### 3.1 Adding Documents

You can add knowledge through:
- **File upload**: Drag and drop files into the project knowledge area
- **Text paste**: Paste content directly as a knowledge entry
- **URL** (limited): Some integrations allow importing from URLs

### 3.2 Supported Formats and Limits

```
┌──────────────────────────────────────────────────────────────────┐
│                Supported File Formats                             │
├──────────────────┬───────────────────────────────────────────────┤
│ Format           │ Notes                                         │
├──────────────────┼───────────────────────────────────────────────┤
│ PDF (.pdf)       │ Text extracted; scanned PDFs may have limits  │
│ Text (.txt)      │ Plain text, fully supported                   │
│ Markdown (.md)   │ Rendered and searchable                       │
│ Code files       │ .py, .js, .ts, .java, .c, .cpp, .rs, etc.   │
│ JSON (.json)     │ Parsed as structured data                     │
│ CSV (.csv)       │ Tabular data, best under 10k rows            │
│ HTML (.html)     │ Text content extracted                        │
│ DOCX (.docx)     │ Word documents, text extracted                │
│ XLSX (.xlsx)     │ Spreadsheets, converted to tabular format     │
│ Images           │ .png, .jpg, .gif, .webp (visual analysis)    │
└──────────────────┴───────────────────────────────────────────────┘

Limits (as of early 2026):
- Individual file size:  Up to ~30 MB
- Total project knowledge: Up to ~200k tokens (~500 pages of text)
- Number of files:  Practical limit around 50-100 files
- File count limit varies by plan (Pro, Team, Enterprise)
```

### 3.3 How Claude Uses Project Knowledge

When you start a conversation within a project, Claude receives all project knowledge as part of its context window. This means:

1. **Full context access**: Claude can reference any part of any uploaded document.
2. **Cross-document reasoning**: Claude can connect information across multiple files.
3. **Grounded responses**: Claude will cite and reference your documents rather than relying solely on training data.
4. **Priority over training data**: When project knowledge conflicts with Claude's training, the project knowledge takes precedence.

```
┌──────────────────────────────────────────────────────────┐
│            How Project Knowledge Flows                    │
│                                                          │
│  Project Knowledge                                       │
│  ┌────────────────┐                                      │
│  │ design-spec.md │──┐                                   │
│  │ api-schema.json│──┤   ┌──────────────────┐           │
│  │ style-guide.md │──┼──▶│  System Context   │           │
│  └────────────────┘  │   │  (sent with every │           │
│                      │   │   message)        │           │
│  Custom Instructions │   └────────┬─────────┘           │
│  ┌────────────────┐  │            │                      │
│  │ "You are a..." │──┘            ▼                      │
│  └────────────────┘      ┌──────────────────┐           │
│                          │  User's Message   │           │
│                          └────────┬─────────┘           │
│                                   │                      │
│                                   ▼                      │
│                          ┌──────────────────┐           │
│                          │ Claude's Response │           │
│                          │ (grounded in      │           │
│                          │  project docs)    │           │
│                          └──────────────────┘           │
└──────────────────────────────────────────────────────────┘
```

### 3.4 Effective Knowledge Organization

Tips for maximizing the value of project knowledge:

```markdown
# Good: Structured document with clear sections
## API Endpoints

### POST /api/users
Creates a new user account.

**Request Body:**
| Field    | Type   | Required | Description           |
|----------|--------|----------|-----------------------|
| name     | string | Yes      | User's full name      |
| email    | string | Yes      | Valid email address    |
| role     | string | No       | Default: "viewer"     |

**Response (201):**
```json
{
  "id": "usr_abc123",
  "name": "Alice Smith",
  "email": "alice@example.com",
  "role": "viewer",
  "createdAt": "2026-01-15T10:30:00Z"
}
```

**Error Responses:**
- 400: Validation error (missing required fields)
- 409: Email already in use
- 500: Internal server error
```

Contrast this with a poorly organized document that dumps unstructured text without headers, tables, or clear delineations. Claude can work with both, but structured documents yield significantly better results.

---

## 4. Custom Instructions

Custom instructions act as a system prompt for every conversation within the project. They shape Claude's behavior, tone, focus, and output format.

### 4.1 Setting Custom Instructions

Navigate to Project Settings and enter your instructions in the "Custom Instructions" field. These apply to all new conversations in the project.

### 4.2 Instruction Categories

**Role-based instructions:**

```
You are a senior backend engineer at Acme Corp. You specialize in Python
(FastAPI) and PostgreSQL. When reviewing code, you focus on:
- Type safety and Pydantic model usage
- SQL injection prevention
- Proper error handling with structured logging
- Performance implications (N+1 queries, missing indexes)

When writing new code, always include:
- Type hints on all function signatures
- Docstrings with Args/Returns/Raises
- Unit test suggestions
```

**Tone and style instructions:**

```
Write in a clear, direct style. Avoid jargon unless the user uses it first.
When explaining technical concepts, use analogies from everyday life.
Always provide code examples in Python unless the user specifies otherwise.
Keep responses concise — aim for clarity over completeness.
If you are unsure about something, say so explicitly.
```

**Industry-specific instructions:**

```
You are a healthcare data analyst. When working with data:
- Never display or output raw PHI (Protected Health Information)
- Always use de-identified sample data in examples
- Reference HIPAA compliance requirements where relevant
- Use ICD-10 codes when discussing diagnoses
- Prefer R for statistical analysis, Python for data engineering
```

**Output format instructions:**

```
When I ask you to review a document, use this format:

## Summary
[1-2 sentence summary of the document]

## Strengths
- [Bulleted list]

## Issues Found
| # | Severity | Location | Issue | Suggestion |
|---|----------|----------|-------|------------|
| 1 | High     | p.3      | ...   | ...        |

## Recommended Actions
1. [Numbered, prioritized list]
```

### 4.3 Instruction Precedence

When multiple instructions are present (project instructions + conversation context), Claude follows this precedence:

1. **Project custom instructions** (highest priority -- system-level)
2. **User messages in conversation** (can override or refine)
3. **Claude's default behavior** (fallback)

This means project instructions are reliable for enforcing consistent behavior across all conversations.

---

## 5. Artifacts

### 5.1 What Are Artifacts?

Artifacts are standalone content pieces that Claude creates during a conversation. Unlike inline text responses, artifacts appear in a separate panel and can be:

- Viewed independently of the conversation
- Edited and iterated on
- Downloaded or copied
- Rendered in real-time (for HTML, React, SVG)

Think of artifacts as Claude's "canvas" -- a dedicated space for producing structured output.

### 5.2 Artifact Types

```
┌─────────────────────────────────────────────────────────────────┐
│                     Artifact Types                               │
├──────────────────┬──────────────────────────────────────────────┤
│ Type             │ Description and Use Cases                     │
├──────────────────┼──────────────────────────────────────────────┤
│ Code             │ Python, JavaScript, TypeScript, SQL, etc.    │
│                  │ Syntax highlighted, copyable                  │
│                  │ Use: algorithms, scripts, configurations     │
├──────────────────┼──────────────────────────────────────────────┤
│ Documents        │ Markdown documents                            │
│                  │ Rendered with headings, lists, tables        │
│                  │ Use: reports, specifications, guides          │
├──────────────────┼──────────────────────────────────────────────┤
│ HTML/CSS/JS      │ Complete web pages rendered in real time      │
│                  │ Interactive with JavaScript                   │
│                  │ Use: prototypes, dashboards, landing pages   │
├──────────────────┼──────────────────────────────────────────────┤
│ React Components │ Full React components with live preview      │
│                  │ Supports hooks, state, props                  │
│                  │ Use: UI prototyping, component design        │
├──────────────────┼──────────────────────────────────────────────┤
│ SVG              │ Vector graphics rendered inline               │
│                  │ Scalable, editable                            │
│                  │ Use: icons, diagrams, illustrations          │
├──────────────────┼──────────────────────────────────────────────┤
│ Mermaid Diagrams │ Flowcharts, sequence diagrams, ER diagrams   │
│                  │ Rendered as visual diagrams                   │
│                  │ Use: architecture diagrams, process flows     │
└──────────────────┴──────────────────────────────────────────────┘
```

### 5.3 Creating Artifacts

Artifacts are created naturally in conversation. Claude decides to create an artifact when the content is:
- **Substantial** (more than a few lines)
- **Self-contained** (makes sense on its own)
- **Likely to be reused** (code, documents, designs)

You can also explicitly request artifacts:

```
User: Create an HTML artifact with a responsive pricing table
      for three tiers: Basic ($9/mo), Pro ($29/mo), Enterprise (custom).

Claude: [Creates an HTML artifact with a fully styled pricing table]
```

### 5.4 Editing and Iterating

One of the most powerful features of artifacts is iterative refinement:

```
Conversation Flow:
─────────────────
User: "Create a Python class for managing a todo list"
Claude: [Creates artifact: TodoList class with add/remove/complete methods]

User: "Add priority levels (low, medium, high) and sorting"
Claude: [Updates the same artifact with priority enum and sort method]

User: "Add JSON serialization and file persistence"
Claude: [Updates artifact again with to_json/from_json and save/load methods]

User: "Now write unit tests for this class"
Claude: [Creates a new artifact: test_todo_list.py with pytest tests]
```

Each iteration builds on the previous version. Claude tracks the artifact's state and applies changes incrementally rather than rewriting from scratch.

### 5.5 Using Artifacts for Prototyping

Artifacts excel at rapid prototyping, especially for web interfaces:

```
User: "Build a dashboard that shows real-time CPU and memory usage
       with animated charts. Use HTML/CSS/JS only (no frameworks)."

Claude: [Creates an HTML artifact with:
  - CSS Grid layout for dashboard panels
  - Canvas-based animated line charts
  - Simulated real-time data updates
  - Responsive design
  - Color scheme with dark mode support]
```

The artifact renders immediately in the preview panel, allowing you to see and interact with the prototype without leaving Claude.

**React component prototyping:**

```
User: "Create a React component for a file upload area with
       drag-and-drop, progress bars, and file type validation."

Claude: [Creates a React artifact with:
  - Drag-and-drop zone with visual feedback
  - File type validation (images, PDFs, max 10MB)
  - Upload progress simulation
  - File list with remove buttons
  - Error states for invalid files]
```

### 5.6 Artifact Limitations

- **No persistent state**: Artifacts don't save between sessions (copy/download them).
- **Limited dependencies**: React artifacts use a sandboxed environment with limited libraries.
- **No backend**: HTML/React artifacts run client-side only -- no API calls to external servers.
- **Size limits**: Very large artifacts (thousands of lines) may be truncated.
- **No file system access**: Artifacts cannot read or write files on your computer.

---

## 6. Team Collaboration

### 6.1 Sharing Projects (Team and Enterprise Plans)

On Claude Team and Enterprise plans, projects can be shared with team members:

```
┌────────────────────────────────────────────────────────────────┐
│                Team Project Sharing                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Project Owner                                                  │
│  ├── Full control: edit knowledge, instructions, settings      │
│  ├── Can add/remove team members                                │
│  └── Can delete the project                                     │
│                                                                 │
│  Team Members                                                   │
│  ├── Can start conversations within the project                │
│  ├── All conversations share the same knowledge base           │
│  ├── Each member's conversations are private by default        │
│  └── Can share individual conversations with the team          │
│                                                                 │
│  Shared Knowledge (visible to all members):                     │
│  ├── Uploaded documents                                         │
│  ├── Custom instructions                                        │
│  └── Project settings                                           │
│                                                                 │
│  Private (per member):                                          │
│  ├── Individual conversations                                   │
│  └── Conversation-specific artifacts                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Activity and History

Team projects provide:
- **Activity feed**: See when knowledge was added/updated and by whom
- **Conversation sharing**: Share a conversation URL with teammates
- **Conversation snapshots**: Save a conversation state for reference

### 6.3 Team Project Use Cases

| Use Case | Project Setup |
|----------|--------------|
| Onboarding | Upload codebase architecture docs, coding standards, team processes |
| Code Review | Upload style guide, linting rules, past review decisions |
| Customer Support | Upload product docs, FAQs, escalation procedures |
| Content Creation | Upload brand guidelines, voice/tone docs, example content |
| Research | Upload papers, datasets, methodology notes |

---

## 7. Projects vs CLAUDE.md

Both Projects and CLAUDE.md provide Claude with persistent context, but they serve different purposes and work in different environments.

```
┌─────────────────┬────────────────────────┬────────────────────────┐
│                 │ Claude Projects        │ CLAUDE.md              │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Environment     │ claude.ai, Desktop     │ Claude Code (CLI)      │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Context source  │ Uploaded files         │ Files in repository    │
│                 │ (manually managed)     │ (auto-discovered)      │
├─────────────────┼────────────────────────┼────────────────────────┤
│ File access     │ Only uploaded files    │ Entire codebase via    │
│                 │                        │ tools (Read, Glob, Grep│
│                 │                        │ Bash, etc.)            │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Code execution  │ No (artifacts only)    │ Yes (full terminal)    │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Version control │ Manual uploads         │ Git-tracked with code  │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Team sharing    │ Built-in (Team plan)   │ Shared via repository  │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Best for        │ Discussions, planning, │ Active development,    │
│                 │ analysis, writing,     │ code changes, testing, │
│                 │ prototyping            │ debugging, automation  │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Update          │ Re-upload files        │ Edit CLAUDE.md in repo │
│ workflow        │                        │ (changes apply         │
│                 │                        │  immediately)          │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Instructions    │ Project settings UI    │ CLAUDE.md files at     │
│ location        │                        │ repo/folder level      │
└─────────────────┴────────────────────────┴────────────────────────┘
```

### When to Use Which

**Use Claude Projects when:**
- You are discussing, planning, or analyzing (not actively coding)
- You need to share context with non-technical team members
- You want to prototype UIs with artifacts
- You are working with documents (PDFs, reports, specs) that are not in a Git repo
- You want persistent knowledge that does not change frequently

**Use CLAUDE.md when:**
- You are actively developing code in a repository
- You need Claude to read, write, and execute code
- You want instructions to be version-controlled with your code
- You want folder-level instructions that apply to specific parts of a codebase
- Your team shares context through the repository (not through a Claude plan)

**Use both together:**
- Upload your architecture docs to a Project for high-level discussions
- Maintain CLAUDE.md in the repo for day-to-day coding with Claude Code
- Use the Project for planning, then switch to Claude Code for implementation

---

## 8. Best Practices

### 8.1 Knowledge Organization

1. **Use descriptive file names**: `api-endpoints-v3.md` is better than `doc1.md`.
2. **Structure documents with headers**: Claude navigates structured documents more effectively.
3. **Keep files focused**: One topic per file. A 10-page API spec is better than a 100-page everything-doc.
4. **Include examples**: Documents with concrete examples (sample requests, code snippets) yield better Claude responses.
5. **Update regularly**: Remove outdated documents. Stale context produces stale answers.

### 8.2 Custom Instructions

1. **Be specific**: "Write Python code" is weak. "Write Python 3.12+ code with type hints, using FastAPI for web endpoints and SQLAlchemy 2.0 for database access" is strong.
2. **Define what NOT to do**: "Do not suggest JavaScript solutions" prevents unhelpful detours.
3. **Set output format expectations**: Tell Claude exactly how you want responses structured.
4. **Test and iterate**: Start conversations, see how Claude behaves, and refine instructions.
5. **Keep instructions under 1000 words**: Overly long instructions dilute their effectiveness.

### 8.3 Artifacts

1. **Start simple, iterate**: Request a basic version first, then add features incrementally.
2. **Be specific about requirements**: "Create a responsive pricing table with three tiers" beats "make a pricing page."
3. **Save important artifacts**: Copy or download artifacts you want to keep -- they do not persist across sessions.
4. **Use artifacts for exploration**: Prototype multiple approaches as separate artifacts, then pick the best one.
5. **Combine with project knowledge**: Upload a design system document, then ask Claude to create artifacts that follow it.

### 8.4 Team Collaboration

1. **Designate a project owner**: One person should maintain the knowledge base and instructions.
2. **Document project conventions**: Add a "How to use this project" knowledge entry.
3. **Review shared conversations**: Look at how team members interact with the project to improve instructions.
4. **Separate concerns**: Create different projects for different workstreams rather than one mega-project.

---

## 9. Exercises

### Exercise 1: Create a Project (Beginner)

Create a Claude Project for a personal hobby or interest (cooking, photography, fitness, etc.):
1. Upload 3-5 relevant documents (recipes, technique guides, workout plans).
2. Write custom instructions that define Claude's role (personal chef, coach, etc.).
3. Start a conversation and verify that Claude references your uploaded documents.
4. Ask Claude something that requires cross-referencing multiple documents.

### Exercise 2: Custom Instructions Workshop (Beginner)

Write custom instructions for three different roles:
1. A senior code reviewer for a Python Django project
2. A technical writer creating API documentation
3. A data analyst working with sales data

For each, include: role definition, output format, things to avoid, and domain-specific requirements. Test each in a conversation and refine.

### Exercise 3: Artifact Prototyping (Intermediate)

Use artifacts to prototype a complete landing page:
1. Ask Claude to create an HTML artifact for a SaaS product landing page.
2. Iterate to add: responsive navigation, hero section, feature grid, pricing table, footer.
3. Ask Claude to create a React component version of just the pricing table.
4. Request an SVG artifact for the product logo.
5. Export all artifacts and assemble them into a working project.

### Exercise 4: Team Project Design (Intermediate)

Design a project structure for a 5-person engineering team building a REST API:
1. List all documents you would upload as project knowledge.
2. Write the custom instructions.
3. Describe how different team members (frontend, backend, QA, PM, designer) would use the project.
4. Explain how you would keep the project knowledge synchronized with the actual codebase.

### Exercise 5: Projects + CLAUDE.md Integration (Advanced)

Design a workflow that uses both Claude Projects and CLAUDE.md together for a full-stack application:
1. Define what goes into the Project (design docs, architecture decisions, meeting notes).
2. Define what goes into CLAUDE.md (coding standards, build commands, test patterns).
3. Write a process document explaining when team members should use which tool.
4. Create a sample CLAUDE.md and matching Project custom instructions that complement each other.

---

## 10. References

- Claude Projects Documentation - https://docs.anthropic.com/en/docs/claude-ai/projects
- Anthropic Cookbook: Projects - https://github.com/anthropics/anthropic-cookbook
- Claude Artifacts Guide - https://support.anthropic.com/en/articles/artifacts
- CLAUDE.md Documentation - https://docs.anthropic.com/en/docs/claude-code/memory
- Claude Team Plan - https://www.anthropic.com/claude/team

---

## Next Lesson

[15. Claude API Fundamentals](./15_Claude_API_Fundamentals.md) covers programmatic access to Claude through the Messages API, including authentication, request/response structure, streaming, and error handling with working code examples in Python and TypeScript.
