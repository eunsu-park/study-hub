# Claude Ecosystem Examples

Example files demonstrating the Claude ecosystem features.

## Directory Structure

```
Claude_Ecosystem/
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
├── 13_mcp_server/          # MCP server implementations
│   ├── simple_mcp_server.py    # Python MCP server (weather)
│   └── simple_mcp_server.ts    # TypeScript MCP server (notes)
│
├── 16_tool_use/            # API tool use examples
│   ├── basic_tool_use.py       # Basic tool calling pattern
│   └── parallel_tools.py       # Parallel tool execution
│
└── 17_agent_sdk/           # Agent SDK examples
    ├── basic_agent.py          # Basic agent with streaming
    └── custom_tools_agent.py   # Agent with custom tools
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
- `13_mcp_server/` → Lesson 13: Building Custom MCP Servers
- `16_tool_use/` → Lesson 16: Tool Use and Function Calling
- `17_agent_sdk/` → Lesson 17: Claude Agent SDK
