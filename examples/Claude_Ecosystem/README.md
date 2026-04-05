# Claude Ecosystem Examples

Example files demonstrating the Claude ecosystem features.

## Directory Structure

```
Claude_Ecosystem/
├── 01_introduction/        # Model selection guide
│   └── model_selection.md      # Model comparison and decision framework
│
├── 02_getting_started/     # Claude Code getting started
│   └── workflow_example.sh     # CLI workflow and slash commands reference
│
├── 03_claude_md/           # CLAUDE.md and settings examples
│   ├── basic_claude_md.md      # Basic project CLAUDE.md template
│   ├── team_claude_md.md       # Team project CLAUDE.md template
│   └── settings_example.json   # .claude/settings.json example
│
├── 05_hooks/               # Hook configuration examples
│   ├── format_on_edit.json     # Auto-format on file edit
│   ├── test_on_save.json       # Run tests on file save
│   └── session_start.json      # Session start status summary
│
├── 06_skills/              # Custom skill examples
│   ├── commit_skill/
│   │   └── SKILL.md            # Smart commit skill
│   └── review_skill/
│       └── SKILL.md            # Code review skill
│
├── 07_subagents/           # Subagent definitions
│   └── custom_agent.md         # Custom subagent examples
│
├── 08_agent_teams/         # Multi-agent collaboration
│   └── team_orchestration.py   # Team orchestration pattern
│
├── 09_ide_integration/     # IDE extension settings
│   └── vscode_settings.json    # VS Code extension configuration
│
├── 10_desktop/             # Claude Desktop workflows
│   └── desktop_workflows.md    # Worktrees, App Preview, GitHub integration
│
├── 11_cowork/              # Claude Cowork use cases
│   └── cowork_use_cases.md     # Documentation, research, project management
���
├── 12_mcp_basics/          # MCP client configuration
│   └── mcp_client_config.json  # mcpServers settings example
│
├── 13_mcp_server/          # MCP server implementations
│   ├── simple_mcp_server.py    # Python MCP server (weather)
│   └── simple_mcp_server.ts    # TypeScript MCP server (notes)
│
├── 15_api/                 # Claude API usage
│   ├── api_messages.py         # Messages API (basic, multi-turn, streaming)
│   └── api_multimodal.py       # Image and PDF input
│
├── 16_tool_use/            # API tool use examples
│   ├── basic_tool_use.py       # Basic tool calling pattern
│   └── parallel_tools.py       # Parallel tool execution
│
├── 17_agent_sdk/           # Agent SDK examples
│   ├── basic_agent.py          # Basic agent with streaming
│   └── custom_tools_agent.py   # Agent with custom tools
│
├── 14_projects_artifacts/  # Claude Projects and Artifacts
│   └── project_setup.md        # Project knowledge structure and artifact types
│
├── 18_custom_agents/       # Production agent patterns
│   └─��� guardrails_agent.py     # Agent with input/output guardrails
│
├── 19_optimization/        # Cost optimization
│   └── cost_calculator.py      # Model pricing comparison
│
├── 20_workflows/           # Development workflow guides
│   └── tdd_workflow.md         # TDD with Claude Code
│
├── 21_best_practices/      # Best practices reference
│   └── prompt_patterns.md      # Effective prompt patterns
│
├── 22_troubleshooting/     # Debugging reference
│   └── debug_checklist.md      # Troubleshooting checklist
│
├── 23_vision/              # Vision agent patterns
│   └── vision_api_example.py   # Image analysis and structured extraction
│
├── 24_caching_batch/       # Cost optimization
│   └── prompt_caching.py       # Prompt caching and Batch API patterns
│
└── 25_rag_patterns/        # RAG implementation
    └── basic_rag.py            # Chunking, retrieval, hybrid search, citations
```

## Requirements

- **Python examples**: `pip install anthropic mcp claude-agent-sdk`
- **TypeScript examples**: `npm install @modelcontextprotocol/sdk`
- **API key**: Set `ANTHROPIC_API_KEY` environment variable for API examples

## Related Lessons

Each subdirectory corresponds to a lesson in the Claude_Ecosystem topic:
- `03_claude_md/` → Lesson 03: CLAUDE.md and Project Setup
- `05_hooks/` → Lesson 05: Hooks and Event-Driven Automation
- `06_skills/` → Lesson 06: Skills and Slash Commands
- `07_subagents/` → Lesson 07: Subagents and Task Delegation
- `08_agent_teams/` → Lesson 08: Agent Teams
- `12_mcp_basics/` → Lesson 12: Model Context Protocol (MCP)
- `13_mcp_server/` → Lesson 13: Building Custom MCP Servers
- `15_api/` → Lesson 15: Claude API Fundamentals
- `16_tool_use/` → Lesson 16: Tool Use and Function Calling
- `17_agent_sdk/` → Lesson 17: Claude Agent SDK
- `18_custom_agents/` → Lesson 18: Building Custom Agents
- `19_optimization/` → Lesson 19: Models, Pricing, and Optimization
- `20_workflows/` → Lesson 20: Advanced Development Workflows
- `21_best_practices/` → Lesson 21: Best Practices and Patterns
- `22_troubleshooting/` → Lesson 22: Troubleshooting and Debugging
