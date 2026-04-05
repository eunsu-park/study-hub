# Prompt Engineering

## Topic Overview

Prompt engineering is the discipline of designing, optimizing, and evaluating inputs to large language models (LLMs) to reliably produce desired outputs. As LLMs have become central to software development, data analysis, content creation, and decision support, the ability to communicate effectively with these models has become a core professional skill — not just for AI researchers, but for anyone who uses LLMs in their work.

This topic goes far beyond basic "how to write a good prompt" advice. It covers the theoretical foundations of why certain prompting techniques work, systematic methodologies for prompt design and optimization, automated prompt engineering tools, evaluation frameworks, adversarial robustness, and production deployment patterns. The goal is to build a principled understanding that transfers across models and use cases.

This topic assumes familiarity with LLM concepts (tokenization, temperature, context windows) and basic API usage. Prior completion of the NLP_and_LLM topic is recommended.

## Learning Path

```
Foundations                     Reasoning & Structure            Applied Domains
─────────────────              ─────────────────                ─────────────────
01 Prompt Fundamentals  ★      05 Structured Output   ★★       08 Multimodal         ★★★
02 Zero/Few-Shot        ★★     06 System Prompts      ★★       09 Code Generation    ★★★
03 Chain-of-Thought     ★★     07 Multi-Turn          ★★★      10 RAG Patterns       ★★★
04 Advanced Reasoning   ★★★                                     14 Domain-Specific    ★★

Optimization & Safety          Production                       Project
─────────────────              ─────────────────                ─────────────────
11 Optimization         ★★★    15 Production Mgmt     ★★★      17 Capstone           ★★★★
12 Evaluation           ★★★    16 Agent Prompting     ★★★★
13 Adversarial          ★★★
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [Prompt Fundamentals](./01_Prompt_Fundamentals.md) | ⭐ | Prompt anatomy, role/task/format/context, mental models |
| 02 | [Zero-Shot and Few-Shot](./02_Zero_Shot_and_Few_Shot.md) | ⭐⭐ | Example selection, ordering effects, dynamic few-shot |
| 03 | [Chain of Thought](./03_Chain_of_Thought.md) | ⭐⭐ | CoT, zero-shot CoT, auto-CoT, when CoT helps vs hurts |
| 04 | [Advanced Reasoning Prompts](./04_Advanced_Reasoning_Prompts.md) | ⭐⭐⭐ | Tree-of-Thought, Self-Consistency, Graph-of-Thought, meta-prompting |
| 05 | [Structured Output Prompting](./05_Structured_Output_Prompting.md) | ⭐⭐ | JSON/XML output, schema-constrained generation, grammar-based decoding |
| 06 | [System Prompt Design](./06_System_Prompt_Design.md) | ⭐⭐ | Persona design, instruction hierarchy, behavioral guardrails |
| 07 | [Multi-Turn Conversation](./07_Multi_Turn_Conversation.md) | ⭐⭐⭐ | Context management, memory injection, conversation steering |
| 08 | [Multimodal Prompting](./08_Multimodal_Prompting.md) | ⭐⭐⭐ | Vision prompts, image+text reasoning, document understanding |
| 09 | [Code Generation Prompting](./09_Code_Generation_Prompting.md) | ⭐⭐⭐ | Coding prompts, test-driven prompting, debugging prompts |
| 10 | [RAG Prompt Patterns](./10_RAG_Prompt_Patterns.md) | ⭐⭐⭐ | Retrieval-augmented prompting, citation, grounding, faithfulness |
| 11 | [Prompt Optimization](./11_Prompt_Optimization.md) | ⭐⭐⭐ | DSPy, OPRO, automated prompt tuning, gradient-free optimization |
| 12 | [Evaluation and Metrics](./12_Evaluation_and_Metrics.md) | ⭐⭐⭐ | Prompt quality metrics, A/B testing, regression testing, benchmarks |
| 13 | [Adversarial Prompting](./13_Adversarial_Prompting.md) | ⭐⭐⭐ | Jailbreaks, prompt injection, defensive design, input sanitization |
| 14 | [Domain-Specific Prompting](./14_Domain_Specific_Prompting.md) | ⭐⭐ | Data extraction, analysis, summarization, translation, education |
| 15 | [Prompt Management in Production](./15_Prompt_Management_in_Production.md) | ⭐⭐⭐ | Versioning, templating, prompt registries, CI/CD for prompts |
| 16 | [Agent Prompting Patterns](./16_Agent_Prompting_Patterns.md) | ⭐⭐⭐⭐ | Tool-use prompts, planning prompts, reflection, orchestration |
| 17 | [Capstone: Prompt Library](./17_Capstone_Prompt_Library.md) | ⭐⭐⭐⭐ | Build a reusable prompt library with evaluation suite |

## Prerequisites

- Familiarity with LLM concepts (tokenization, temperature, context windows)
- Basic Python programming and API usage
- Recommended: [NLP and LLM](../NLP_and_LLM/00_Overview.md) topic

## Environment Setup

```bash
# Python environment
python -m venv prompt-eng
source prompt-eng/bin/activate

# Core libraries
pip install anthropic openai tiktoken

# For optimization lessons (Lesson 11)
pip install dspy-ai

# For evaluation lessons (Lesson 12)
pip install promptfoo  # or install via npm: npm install -g promptfoo

# Verify
python -c "import anthropic; print('Ready')"
```

## Recommended Resources

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — Official Claude prompting guide
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — GPT-focused strategies
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Community-maintained comprehensive guide
- [DSPy Documentation](https://dspy-docs.vercel.app/) — Programmatic prompt optimization
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
