# AI Safety and Alignment

## Topic Overview

AI Safety and Alignment is the interdisciplinary field concerned with ensuring that advanced AI systems behave in ways that are beneficial, predictable, and aligned with human values and intentions. As AI capabilities grow rapidly — particularly in large language models and autonomous agents — the gap between what AI systems *can* do and what we can *guarantee* they will do has become one of the most pressing challenges in technology.

This topic covers the full spectrum of AI safety: from near-term practical concerns (prompt injection, bias, hallucination) to long-term alignment challenges (deceptive alignment, mesa-optimization, scalable oversight). It spans technical methods (RLHF, Constitutional AI, DPO, red-teaming, representation engineering), evaluation frameworks (safety benchmarks, automated auditing), governance structures (organizational safety, frontier model policies), and societal impacts.

This topic assumes familiarity with deep learning, NLP/LLM concepts, and reinforcement learning fundamentals. It builds on the theoretical foundations from those topics to address the question: *How do we make powerful AI systems safe and trustworthy?*

## Learning Path

```
Foundations                     Alignment Methods                Evaluation & Defense
─────────────────              ─────────────────                ─────────────────
01 Safety Landscape    ★★      03 RLHF for Alignment  ★★★      07 Red Teaming        ★★★
02 Alignment Problem   ★★★     04 Constitutional AI   ★★★      08 Safety Evaluation  ★★★
                                05 DPO Methods         ★★★      09 Robustness         ★★★
                                06 Scalable Oversight  ★★★★     10 Repr. Engineering  ★★★★

Deployment & Defense            Governance & Society             Project
─────────────────              ─────────────────                ─────────────────
11 Guardrails          ★★★     13 Governance          ★★★      17 Capstone           ★★★★
12 Deceptive Align.    ★★★★    14 Responsible Deploy  ★★★
                                15 Societal Impact     ★★
                                16 Open Problems       ★★★★
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [AI Safety Landscape](./01_AI_Safety_Landscape.md) | ⭐⭐ | Risk taxonomy, near-term vs long-term, key organizations |
| 02 | [The Alignment Problem](./02_Alignment_Problem.md) | ⭐⭐⭐ | Inner/outer alignment, Goodhart's law, reward hacking |
| 03 | [RLHF for Alignment](./03_RLHF_for_Alignment.md) | ⭐⭐⭐ | RLHF pipeline, reward modeling, PPO for alignment |
| 04 | [Constitutional AI](./04_Constitutional_AI.md) | ⭐⭐⭐ | RLAIF, principle hierarchies, self-critique |
| 05 | [Direct Preference Optimization](./05_Direct_Preference_Optimization.md) | ⭐⭐⭐ | DPO, KTO, IPO — reward-free alignment methods |
| 06 | [Scalable Oversight](./06_Scalable_Oversight.md) | ⭐⭐⭐⭐ | Debate, recursive reward modeling, weak-to-strong generalization |
| 07 | [Red Teaming](./07_Red_Teaming.md) | ⭐⭐⭐ | Systematic red-teaming, automated red-teaming, evaluation |
| 08 | [Safety Evaluation](./08_Safety_Evaluation.md) | ⭐⭐⭐ | TruthfulQA, BBQ, toxicity benchmarks, custom eval harnesses |
| 09 | [Robustness and Adversarial](./09_Robustness_and_Adversarial.md) | ⭐⭐⭐ | Adversarial attacks on LLMs, robustness training, input filtering |
| 10 | [Representation Engineering](./10_Representation_Engineering.md) | ⭐⭐⭐⭐ | Activation steering, representation reading, safety probes |
| 11 | [Guardrails and Filters](./11_Guardrails_and_Filters.md) | ⭐⭐⭐ | NeMo Guardrails, Guardrails AI, input/output filtering |
| 12 | [Deceptive Alignment](./12_Deceptive_Alignment.md) | ⭐⭐⭐⭐ | Mesa-optimization, deception, goal misgeneralization |
| 13 | [Governance Frameworks](./13_Governance_Frameworks.md) | ⭐⭐⭐ | Organizational safety, AI policy, frontier model governance |
| 14 | [Responsible Deployment](./14_Responsible_Deployment.md) | ⭐⭐⭐ | Staged release, monitoring, incident response for AI |
| 15 | [Societal Impact](./15_Societal_Impact.md) | ⭐⭐ | Labor displacement, dual-use, concentration of power |
| 16 | [Open Problems](./16_Open_Problems.md) | ⭐⭐⭐⭐ | Current research frontiers, unsolved alignment problems |
| 17 | [Capstone: Safety Audit](./17_Capstone_Safety_Audit.md) | ⭐⭐⭐⭐ | Conduct a full safety audit on a model/application |

## Prerequisites

- Deep learning fundamentals (neural networks, training, optimization)
- LLM concepts (transformers, tokenization, fine-tuning, RLHF basics)
- Reinforcement learning basics (reward, policy, value functions)
- Recommended: [NLP and LLM](../NLP_and_LLM/00_Overview.md), [Reinforcement Learning](../Reinforcement_Learning/00_Overview.md), [Interpretable AI](../Interpretable_AI/00_Overview.md)

## Environment Setup

```bash
# Python environment
python -m venv ai-safety
source ai-safety/bin/activate

# Core libraries
pip install torch transformers datasets accelerate
pip install anthropic openai

# Safety-specific tools
pip install nemoguardrails guardrails-ai
pip install evaluate  # HuggingFace evaluation library

# For representation engineering (Lesson 10)
pip install baukit  # activation analysis

# Verify
python -c "import torch; import transformers; print('Ready')"
```

## Recommended Resources

- [Anthropic Research](https://www.anthropic.com/research) — Constitutional AI, interpretability, alignment research
- [Alignment Forum](https://www.alignmentforum.org/) — Technical AI alignment discussion
- Ngo et al., "The Alignment Problem from a Deep Learning Perspective" (2023)
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)
- Rafailov et al., "Direct Preference Optimization" (2023)
- Burns et al., "Weak-to-Strong Generalization" (2023)
- Zou et al., "Representation Engineering" (2023)
