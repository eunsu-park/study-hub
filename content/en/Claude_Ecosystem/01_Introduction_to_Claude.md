# Introduction to Claude

**Next**: [02. Claude Code: Getting Started](./02_Claude_Code_Getting_Started.md)

---

Claude is Anthropic's family of large language models designed for safety, helpfulness, and honesty. Unlike models that optimize purely for capability, Claude is built with Constitutional AI — a training methodology that aligns the model with human values while maintaining frontier-level performance across reasoning, coding, analysis, and creative tasks. This lesson provides a comprehensive overview of the Claude model family, the product ecosystem, and the foundational concepts you need to work effectively with Claude.

**Difficulty**: ⭐

**Prerequisites**:
- No prior AI experience required
- Basic familiarity with software development concepts is helpful

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the Claude model family and when to use each model
2. Identify Claude's core capabilities and differentiators
3. Navigate the product ecosystem: Claude.ai, Claude Code, Claude Desktop, API
4. Understand context windows, tokens, and how Claude processes text
5. Make informed decisions about model selection based on task requirements
6. Understand the pricing model and cost considerations

---

## Table of Contents

1. [The Claude Model Family](#1-the-claude-model-family)
2. [Core Capabilities](#2-core-capabilities)
3. [How Claude Differs: Constitutional AI](#3-how-claude-differs-constitutional-ai)
4. [Context Windows and Tokens](#4-context-windows-and-tokens)
5. [The Product Ecosystem](#5-the-product-ecosystem)
6. [Pricing and Cost Considerations](#6-pricing-and-cost-considerations)
7. [When to Use Which Product](#7-when-to-use-which-product)
8. [Exercises](#8-exercises)
9. [Next Steps](#9-next-steps)

---

## 1. The Claude Model Family

Anthropic offers three model tiers, each optimized for different trade-offs between capability, speed, and cost. All models share the same safety training and core architecture but differ in scale and performance characteristics.

### Model Comparison Table

| Property | Claude Opus 4 | Claude Sonnet 4 | Claude Haiku |
|----------|--------------|-----------------|--------------|
| **Intelligence** | Highest | High | Good |
| **Speed** | Slower | Fast | Fastest |
| **Cost** | Highest | Moderate | Lowest |
| **Context Window** | 200K tokens | 200K tokens | 200K tokens |
| **Max Output** | 32K tokens | 16K tokens | 8K tokens |
| **Best For** | Complex reasoning, research, architecture | Daily coding, analysis, balanced tasks | Quick queries, classification, high-volume |
| **Extended Thinking** | Yes | Yes | No |

### Claude Opus 4

Opus is Anthropic's most capable model. It excels at tasks requiring deep reasoning, nuanced understanding, and multi-step problem solving. Use Opus when accuracy and depth matter more than speed.

**Strengths**:
- Complex code architecture and system design
- Long-form analysis and research synthesis
- Subtle reasoning about edge cases
- Extended thinking for multi-step problems
- Handling ambiguous or under-specified requirements

**Typical use cases**:
- Designing system architectures
- Reviewing complex pull requests
- Writing production-critical code
- Analyzing research papers
- Multi-file refactoring across large codebases

### Claude Sonnet 4

Sonnet is the balanced workhorse — fast enough for interactive use, capable enough for most professional tasks. It is the default model for Claude Code and the most commonly used model in production.

**Strengths**:
- Rapid code generation and editing
- Interactive development sessions
- Good balance of quality and throughput
- Reliable instruction following
- Cost-effective for sustained use

**Typical use cases**:
- Day-to-day coding assistance
- Writing tests and documentation
- Translation and content generation
- Data analysis and visualization
- API integration work

### Claude Haiku

Haiku is optimized for speed and cost. It handles straightforward tasks reliably and is ideal for high-volume applications where latency and cost are primary concerns.

**Strengths**:
- Sub-second response times
- Lowest per-token cost
- Reliable classification and extraction
- Good for structured output generation

**Typical use cases**:
- Text classification and labeling
- Data extraction from documents
- Simple code completion
- Chat applications with cost constraints
- Pre-processing and routing in agent pipelines

### Model Selection Decision Tree

```
Is the task complex, ambiguous, or safety-critical?
├── Yes → Claude Opus 4
│         (deep reasoning, architecture, complex analysis)
│
├── Moderate → Claude Sonnet 4
│              (daily coding, analysis, content generation)
│
└── Simple/High-volume → Claude Haiku
                          (classification, extraction, routing)
```

---

## 2. Core Capabilities

Claude is a general-purpose AI that handles a wide range of tasks. Understanding its capabilities helps you leverage the right features for each situation.

### Reasoning and Analysis

Claude can engage in multi-step logical reasoning, break down complex problems, and provide structured analysis. With extended thinking enabled, Opus and Sonnet can show their chain of thought before responding.

```python
# Example: Asking Claude to analyze a design decision
prompt = """
Our API currently returns all user data in a single endpoint.
We're considering splitting it into separate endpoints per resource.

Analyze the trade-offs considering:
1. Client complexity
2. Network overhead
3. Caching strategies
4. API versioning
5. Backend complexity
"""

# Claude will provide a structured analysis considering each factor,
# potential migration strategies, and a recommendation.
```

### Code Generation and Understanding

Claude reads and writes code across dozens of programming languages. It understands project structure, follows coding conventions, and can work with existing codebases.

```python
# Claude can generate code from natural language descriptions
# Example: "Create a retry decorator with exponential backoff"

import time
import functools
from typing import Type


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        backoff_factor: Multiplier for each subsequent delay.
        exceptions: Tuple of exception types to catch.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (backoff_factor ** attempt)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


@retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, TimeoutError))
def fetch_data(url: str) -> dict:
    """Fetch data from an API with automatic retry."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()
```

### Multilingual Support

Claude communicates fluently in many languages and can translate between them while preserving technical nuance. It handles code comments, documentation, and technical writing in languages including English, Korean, Japanese, Chinese, French, German, Spanish, and many others.

### Vision (Multimodal)

Claude can analyze images, screenshots, diagrams, and documents. This is valuable for understanding UI mockups, reading whiteboard sketches, analyzing charts, and processing scanned documents.

```python
# Using the API to send an image to Claude
import anthropic
import base64

client = anthropic.Anthropic()

# Read and encode an image
with open("architecture_diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this architecture diagram. Identify potential bottlenecks."
                }
            ],
        }
    ],
)

print(message.content[0].text)
```

### Long Context Processing

With a 200K token context window, Claude can process entire codebases, long documents, and extensive conversation histories. This is roughly equivalent to 500 pages of text or a medium-sized codebase.

---

## 3. How Claude Differs: Constitutional AI

Claude is built using a distinctive approach called **Constitutional AI (CAI)**, which sets it apart from models trained primarily with reinforcement learning from human feedback (RLHF).

### What Is Constitutional AI?

Constitutional AI is a training methodology where the model is guided by a set of principles (a "constitution") rather than relying solely on human labelers to determine what is helpful and harmless. The process has two key phases:

1. **Supervised Learning Phase**: The model generates responses, then critiques and revises its own outputs based on constitutional principles
2. **Reinforcement Learning Phase**: Instead of human preferences, the model's own evaluations (guided by the constitution) provide the training signal

```
Traditional RLHF:
  Model → Response → Human Labeler → Reward Signal → Model Update

Constitutional AI:
  Model → Response → Self-Critique (via principles) → Revised Response
  Model → Response pairs → AI Feedback (via principles) → Reward Signal → Model Update
```

### Why This Matters in Practice

| Aspect | Impact on Users |
|--------|----------------|
| **Transparency** | Claude can explain its reasoning and limitations |
| **Consistency** | More predictable behavior across edge cases |
| **Honesty** | Willing to say "I don't know" rather than fabricate |
| **Nuance** | Handles sensitive topics with care rather than blanket refusal |
| **Helpfulness** | Aims to be maximally helpful within safety boundaries |

### Extended Thinking

Claude Opus and Sonnet support **extended thinking** — an explicit reasoning phase where the model works through a problem step-by-step before producing its final answer. This is visible in the API and Claude.ai as a "thinking" block.

```json
{
  "model": "claude-opus-4-20250514",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "messages": [
    {
      "role": "user",
      "content": "Prove that the square root of 2 is irrational."
    }
  ]
}
```

Extended thinking is particularly valuable for:
- Mathematical proofs and derivations
- Complex debugging scenarios
- Architecture decisions with many trade-offs
- Problems requiring exploration of multiple solution paths

---

## 4. Context Windows and Tokens

Understanding how Claude processes text is essential for effective use, especially when working with large codebases or documents.

### What Is a Token?

A **token** is the fundamental unit Claude uses to process text. Tokens are not words — they are subword units determined by the model's tokenizer. On average:

- **1 token** is approximately 3-4 characters of English text
- **1 word** is approximately 1.3 tokens
- **1 line of code** is approximately 10-15 tokens
- **1 page of text** (~500 words) is approximately 650 tokens

```
Example tokenization:
"Hello, world!" → ["Hello", ",", " world", "!"]  (4 tokens)
"def calculate_total(items):" → ["def", " calculate", "_total", "(", "items", "):"]  (6 tokens)
"안녕하세요" → ["안녕", "하세요"]  (2 tokens, varies by language)
```

### Context Window

The **context window** is the total number of tokens Claude can consider at once, including both the input (your prompt, system instructions, file contents) and the output (Claude's response).

```
┌─────────────────────────────────────────────────────────┐
│                 200K Token Context Window                │
│                                                         │
│  ┌─────────────────────────┐  ┌──────────────────────┐  │
│  │     Input Tokens        │  │   Output Tokens      │  │
│  │                         │  │                      │  │
│  │  System prompt          │  │  Claude's response   │  │
│  │  CLAUDE.md contents     │  │  (up to max_tokens)  │  │
│  │  Conversation history   │  │                      │  │
│  │  File contents          │  │                      │  │
│  │  Tool results           │  │                      │  │
│  │                         │  │                      │  │
│  └─────────────────────────┘  └──────────────────────┘  │
│                                                         │
│  ← Input fills from left     Output fills from right →  │
└─────────────────────────────────────────────────────────┘
```

### Practical Implications

| Scenario | Approximate Token Usage |
|----------|------------------------|
| Short question | 50-100 tokens |
| CLAUDE.md file | 500-3,000 tokens |
| Single source file (~200 lines) | 2,000-3,000 tokens |
| 10 source files | 20,000-30,000 tokens |
| Entire medium codebase | 100,000-150,000 tokens |
| Full conversation (1 hour session) | 50,000-200,000 tokens |

When the context window fills up, Claude Code uses a strategy called **context compaction** — it summarizes the conversation so far to free up space while preserving the most important information.

---

## 5. The Product Ecosystem

Claude is available through multiple products, each designed for different use cases and workflows.

### Product Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Ecosystem                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Claude.ai   │  │Claude Desktop│  │ Claude Code  │       │
│  │  (Web App)   │  │  (macOS/Win) │  │   (CLI)      │       │
│  │              │  │              │  │              │       │
│  │ Chat, Files, │  │ MCP, App     │  │ Code editing,│       │
│  │ Projects,    │  │ Preview,     │  │ Terminal,    │       │
│  │ Artifacts    │  │ System       │  │ Git, Tests   │       │
│  │              │  │ Integration  │  │              │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│                    ┌──────┴───────┐                           │
│                    │  Claude API  │                           │
│                    │              │                           │
│                    │ Messages API │                           │
│                    │ Tool Use     │                           │
│                    │ Streaming    │                           │
│                    │ Batch API    │                           │
│                    └──────┬───────┘                           │
│                           │                                  │
│                    ┌──────┴───────┐                           │
│                    │  Agent SDK   │                           │
│                    │              │                           │
│                    │ Build custom │                           │
│                    │ AI agents    │                           │
│                    └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Claude.ai (Web Application)

The web interface at [claude.ai](https://claude.ai) is the most accessible way to interact with Claude.

**Key features**:
- Conversational chat interface with model selection
- File upload (PDF, images, code files, CSV)
- **Projects**: Organize conversations with shared context and instructions
- **Artifacts**: Claude generates interactive content (code, documents, diagrams)
- Conversation history and search
- Team plans with shared billing and admin controls

**Best for**: General questions, document analysis, brainstorming, writing, quick prototyping.

### Claude Desktop

A native desktop application for macOS and Windows that provides deeper system integration than the web app.

**Key features**:
- **App Preview**: Generate and preview web applications directly in the Desktop app
- **Model Context Protocol (MCP)**: Connect Claude to external tools and data sources
- **GitHub integration**: Pull requests, issues, and repository analysis
- Keyboard shortcut for quick access
- Offline conversation history

**Best for**: Users who want Claude integrated into their desktop workflow with MCP capabilities.

### Claude Code (CLI)

Claude Code is a command-line AI coding assistant that runs directly in your terminal. It is the primary tool for AI-assisted software development and the focus of this topic.

**Key features**:
- Reads and understands your codebase
- Edits files with your approval (or automatically)
- Runs terminal commands (tests, builds, git)
- Integrates with your existing development tools
- Configurable via CLAUDE.md and settings files
- Extensible with hooks and skills

```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Start an interactive session
claude

# Run a one-shot command
claude -p "Explain the authentication flow in this project"

# Use a specific model
claude --model claude-opus-4-20250514
```

**Best for**: Software development, debugging, code review, refactoring, writing tests.

### Claude API

The API provides programmatic access to Claude for building applications, automating workflows, and integrating AI into existing systems.

```python
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a haiku about programming."}
    ]
)

print(message.content[0].text)
```

**Best for**: Building AI-powered applications, automated pipelines, custom integrations.

### Claude for Enterprise

Enterprise offerings include:
- **Claude for Work** (Team plan): Shared billing, admin console, higher rate limits
- **Claude for Enterprise**: SSO, SCIM provisioning, custom data retention, dedicated support
- **Amazon Bedrock / Google Vertex AI**: Access Claude through cloud provider APIs with existing billing

---

## 6. Pricing and Cost Considerations

Claude uses a **pay-per-token** pricing model. You pay for the tokens you send (input) and the tokens Claude generates (output). Output tokens are more expensive because they require more computation.

### Pricing Table (as of early 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku | $0.25 | $1.25 |

> **Note**: Pricing is subject to change. Check [anthropic.com/pricing](https://anthropic.com/pricing) for current rates.

### Cost Estimation Examples

```python
# Cost estimation helper
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost of an API call in USD."""
    pricing = {
        "opus":   {"input": 15.00, "output": 75.00},
        "sonnet": {"input":  3.00, "output": 15.00},
        "haiku":  {"input":  0.25, "output":  1.25},
    }

    rates = pricing[model]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


# Example scenarios
print(f"Quick question (Haiku):    ${estimate_cost('haiku', 500, 200):.4f}")
print(f"Code review (Sonnet):      ${estimate_cost('sonnet', 10000, 2000):.4f}")
print(f"Architecture (Opus):       ${estimate_cost('opus', 50000, 5000):.4f}")
print(f"Full codebase scan (Opus): ${estimate_cost('opus', 150000, 10000):.4f}")

# Output:
# Quick question (Haiku):    $0.0004
# Code review (Sonnet):      $0.0600
# Architecture (Opus):       $1.1250
# Full codebase scan (Opus): $2.9500
```

### Cost Optimization Strategies

1. **Use the right model**: Most tasks do not need Opus. Start with Sonnet and upgrade only when needed.
2. **Prompt caching**: The API supports caching for repeated system prompts and large contexts, reducing input costs by up to 90%.
3. **Batch API**: For non-time-sensitive workloads, the Batch API offers a 50% discount.
4. **Minimize context**: Send only the relevant code and information, not entire files when a snippet suffices.
5. **Claude Code's `/compact` command**: Summarizes conversation to reduce context size mid-session.

---

## 7. When to Use Which Product

### Decision Matrix

| Scenario | Recommended Product | Why |
|----------|-------------------|-----|
| "Help me understand this error" | Claude Code | Direct access to your codebase and terminal |
| "Analyze this PDF report" | Claude.ai | File upload and artifact generation |
| "Build a web app prototype" | Claude Desktop | App Preview for live rendering |
| "Refactor this module" | Claude Code | File editing, test running, git integration |
| "Add AI to my product" | Claude API | Programmatic access, custom integration |
| "Quick translation check" | Claude.ai / Haiku | Fast, low-cost, no setup needed |
| "Review my architecture" | Claude Code (Opus) | Deep reasoning with codebase context |
| "Classify 10K documents" | Claude API (Batch) | High volume, cost-effective batch processing |
| "Connect to my database" | Claude Desktop (MCP) | MCP server provides database access |

### Workflow Integration

For software developers, the typical daily workflow combines multiple products:

```
Morning:
  └─ Claude Code: Review overnight CI failures, fix bugs

Development:
  └─ Claude Code: Write features, tests, documentation

Code Review:
  └─ Claude Code: Review PRs, suggest improvements

Research:
  └─ Claude.ai: Analyze papers, explore design options

Communication:
  └─ Claude.ai: Draft technical documents, emails

Production:
  └─ Claude API: Power user-facing AI features
```

---

## 8. Exercises

### Exercise 1: Model Selection

For each scenario, identify which Claude model (Opus, Sonnet, or Haiku) you would choose and explain why:

1. Classifying 50,000 support tickets into categories
2. Designing a microservices architecture for an e-commerce platform
3. Writing unit tests for a Python module with 15 functions
4. Translating a technical document from English to Korean
5. Building a chatbot for answering FAQ questions

### Exercise 2: Token Estimation

Estimate the token count and cost for these scenarios using Sonnet:

1. Sending a 200-line Python file and asking for a code review
2. Providing 3 files (100 lines each) and asking Claude to refactor them
3. A 30-minute interactive coding session with 20 back-and-forth exchanges

### Exercise 3: Product Selection

A startup is building a code review tool. They need:
- Automated PR reviews triggered by GitHub webhooks
- A dashboard where developers can chat with Claude about their code
- Weekly architecture reports sent to the team lead

Which Claude products would you recommend for each component? Justify your choices considering cost, latency, and integration complexity.

---

## 9. Next Steps

This lesson established the foundational understanding of Claude's model family, capabilities, and product ecosystem. In the next lesson, we will get hands-on with **Claude Code** — installing it, running your first session, and understanding the core workflow that makes it a powerful development tool.

**Next**: [02. Claude Code: Getting Started](./02_Claude_Code_Getting_Started.md)
