# Claude Ecosystem

A comprehensive guide to the Claude AI ecosystem — covering Claude Code (the CLI coding tool), Claude Desktop, Cowork, Model Context Protocol (MCP), the Agent SDK, and the Claude API. This topic provides both conceptual understanding and practical skills for leveraging Claude across the full spectrum of AI-assisted development, from interactive coding sessions to building production-grade AI agents.

## What You'll Learn

- **Foundations**: What Claude is, the model family (Opus, Sonnet, Haiku), and the product ecosystem
- **Claude Code**: Installation, project configuration (CLAUDE.md), permission modes, and daily workflows
- **Automation**: Hooks for event-driven automation, custom Skills and slash commands
- **Agents**: Subagents for task delegation, Agent Teams for multi-agent collaboration
- **IDE & Desktop**: VS Code / JetBrains integration, Claude Desktop features, Cowork
- **MCP**: Model Context Protocol — connecting Claude to external tools and data sources
- **API & SDK**: Messages API, tool use / function calling, and the Agent SDK for building custom agents
- **Advanced**: Cost optimization, development workflows, best practices, and troubleshooting

> **Note**: This topic focuses on **using and building with** the Claude ecosystem. For the theory behind large language models, see **LLM_and_NLP** and **Foundation_Models**. For general prompt engineering concepts, see **LLM_and_NLP/08_Prompt_Engineering.md**.

## Lessons

| # | Title | Difficulty | Description |
|---|-------|------------|-------------|
| 01 | [Introduction to Claude](01_Introduction_to_Claude.md) | ⭐ | Model family, core capabilities, product ecosystem overview |
| 02 | [Claude Code: Getting Started](02_Claude_Code_Getting_Started.md) | ⭐ | Installation, first session, basic workflow (read → edit → test → commit) |
| 03 | [CLAUDE.md and Project Setup](03_CLAUDE_md_and_Project_Setup.md) | ⭐ | Project instructions, .claude/ directory, settings hierarchy |
| 04 | [Permission Modes and Security](04_Permission_Modes.md) | ⭐⭐ | Five permission modes, allow/deny rules, sandboxing |
| 05 | [Hooks and Event-Driven Automation](05_Hooks.md) | ⭐⭐ | Hook lifecycle, configuration, practical automation examples |
| 06 | [Skills and Slash Commands](06_Skills_and_Slash_Commands.md) | ⭐⭐ | SKILL.md, custom skill creation, built-in commands |
| 07 | [Subagents and Task Delegation](07_Subagents.md) | ⭐⭐ | Explore/Plan/General-Purpose subagents, custom definitions |
| 08 | [Agent Teams](08_Agent_Teams.md) | ⭐⭐⭐ | Multi-agent collaboration, shared task lists, parallel workstreams |
| 09 | [IDE Integration](09_IDE_Integration.md) | ⭐ | VS Code extension, JetBrains plugin, keyboard shortcuts |
| 10 | [Claude Desktop Application](10_Claude_Desktop.md) | ⭐ | Desktop features, App Preview, GitHub integration |
| 11 | [Cowork: AI Digital Colleague](11_Cowork.md) | ⭐⭐ | Multi-step task execution, plugins, MCP connectors |
| 12 | [Model Context Protocol (MCP)](12_Model_Context_Protocol.md) | ⭐⭐ | MCP architecture, pre-built servers, connecting tools |
| 13 | [Building Custom MCP Servers](13_Building_MCP_Servers.md) | ⭐⭐⭐ | Resource/Tool/Prompt definitions, TypeScript/Python implementation |
| 14 | [Claude Projects and Artifacts](14_Claude_Projects_and_Artifacts.md) | ⭐ | Project organization, knowledge grounding, artifact types |
| 15 | [Claude API Fundamentals](15_Claude_API_Fundamentals.md) | ⭐⭐ | Authentication, Messages API, streaming, client SDKs |
| 16 | [Tool Use and Function Calling](16_Tool_Use_and_Function_Calling.md) | ⭐⭐ | Tool definitions, calling patterns, parallel execution |
| 17 | [Claude Agent SDK](17_Claude_Agent_SDK.md) | ⭐⭐⭐ | SDK overview, agent loop, built-in tools, context management |
| 18 | [Building Custom Agents](18_Building_Custom_Agents.md) | ⭐⭐⭐ | Custom tools, system prompts, production agent patterns |
| 19 | [Models, Pricing, and Optimization](19_Models_and_Pricing.md) | ⭐⭐ | Model comparison, pricing, prompt caching, batch API |
| 20 | [Advanced Development Workflows](20_Advanced_Workflows.md) | ⭐⭐⭐ | Multi-file refactoring, TDD, CI/CD integration, codebase exploration |
| 21 | [Best Practices and Patterns](21_Best_Practices.md) | ⭐⭐ | Effective prompts, context management, security, team patterns |
| 22 | [Troubleshooting and Debugging](22_Troubleshooting.md) | ⭐⭐ | Permission errors, hook failures, context limits, MCP issues |

## Prerequisites

- **Basic programming experience**: Comfort with at least one programming language (Python or TypeScript recommended)
- **Command-line familiarity**: Basic terminal usage (see **Shell_Script** topic if needed)
- **Git basics**: Understanding of commits, branches, and pull requests (see **Git** topic)

No prior experience with AI tools or APIs is required — the topic starts from the fundamentals.

## Learning Path

**Tier 1 — Foundations (Lessons 1–3)**
Get started with Claude: understand the model family and product ecosystem, install Claude Code, and learn how to configure projects with CLAUDE.md. After these lessons, you can start using Claude Code for everyday development.

**Tier 2 — Core Features (Lessons 4–9)**
Master the power features of Claude Code: permission modes for security, hooks for automation, skills for custom commands, subagents for complex tasks, agent teams for collaboration, and IDE integration for seamless development.

**Tier 3 — Platform & Integration (Lessons 10–14)**
Explore the broader Claude platform: the Desktop application, Cowork for autonomous task execution, MCP for connecting to external tools and data sources, and Projects for organizing knowledge and generating artifacts.

**Tier 4 — API & Advanced (Lessons 15–22)**
Build production-grade AI applications: use the Claude API and Agent SDK, create custom tools and agents, optimize costs, master advanced development workflows, and learn best practices for teams and troubleshooting.

## Related Topics

- **LLM_and_NLP**: Theory behind large language models, BERT, GPT architecture, prompt engineering
- **Foundation_Models**: Scaling laws, model training, PEFT, RAG architecture
- **Programming**: Clean code, design patterns, testing — foundational skills for AI-assisted development
- **Git**: Version control workflows that Claude Code integrates with deeply
- **Docker**: Container environments for secure Claude Code execution (Bypass mode)
- **Web_Development**: Building applications that Claude can help create and iterate on

**License**: Content licensed under CC BY-NC 4.0
