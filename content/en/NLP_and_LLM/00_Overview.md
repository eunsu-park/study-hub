# NLP & LLM Learning Guide

## Introduction

This folder covers Natural Language Processing (NLP) fundamentals and practical Large Language Model (LLM) applications. The curriculum progresses from classical NLP through modern LLM techniques to production-grade application patterns.

**Target Audience**: Learners who have completed the Deep_Learning folder (Transformer and Attention understanding required)

---

## Learning Roadmap

```
[NLP Basics]              [Pre-trained Models]         [LLM Applications]
    │                          │                          │
    ▼                          ▼                          ▼
Tokenization/Embedding ─▶ BERT Understanding ─▶ Prompt Engineering
    │                          │                          │
    ▼                          ▼                          ▼
Word2Vec/GloVe ─────────▶ GPT Understanding ──▶ RAG Systems
    │                          │                          │
    ▼                          ▼                          ▼
Transformer Review ─────▶ HuggingFace ────────▶ Agents & Tools
                               │                          │
                               ▼                          ▼
                          Fine-Tuning/PEFT ───▶ Production Patterns
```

---

## File List

### Section 1: NLP Fundamentals (01-03)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [01_NLP_Basics.md](./01_NLP_Basics.md) | ⭐⭐ | Tokenization, normalization, vocabulary building |
| [02_Word2Vec_GloVe.md](./02_Word2Vec_GloVe.md) | ⭐⭐ | Word embeddings, Skip-gram, CBOW |
| [03_Transformer_Review.md](./03_Transformer_Review.md) | ⭐⭐⭐ | Attention, Encoder-Decoder |

### Section 2: Pre-trained Models (04-08)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [04_BERT_Understanding.md](./04_BERT_Understanding.md) | ⭐⭐⭐ | MLM, NSP, bidirectional encoder |
| [05_GPT_Understanding.md](./05_GPT_Understanding.md) | ⭐⭐⭐ | Autoregressive model, text generation |
| [06_HuggingFace_Basics.md](./06_HuggingFace_Basics.md) | ⭐⭐ | Transformers library, Pipeline |
| [07_Fine_Tuning.md](./07_Fine_Tuning.md) | ⭐⭐⭐ | Classification, QA, summarization fine-tuning |
| [08_PEFT_and_QLoRA.md](./08_PEFT_and_QLoRA.md) | ⭐⭐⭐ | LoRA, QLoRA, DoRA, Adapters, IA3 |

### Section 3: Retrieval & RAG (09-12)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [09_Prompt_Engineering.md](./09_Prompt_Engineering.md) | ⭐⭐ | Prompt design, Few-shot, CoT |
| [10_RAG_Fundamentals.md](./10_RAG_Fundamentals.md) | ⭐⭐⭐ | Retrieval-Augmented Generation, chunking strategies |
| [11_Vector_Search_for_RAG.md](./11_Vector_Search_for_RAG.md) | ⭐⭐⭐ | Chroma, Pinecone, FAISS, embedding models |
| [12_Advanced_RAG.md](./12_Advanced_RAG.md) | ⭐⭐⭐⭐ | Agentic RAG, HyDE, RAPTOR, ColBERT |

### Section 4: Agents & Orchestration (13-16)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [13_LangChain_Basics.md](./13_LangChain_Basics.md) | ⭐⭐⭐ | Chains, agents, memory |
| [14_LLM_Agents.md](./14_LLM_Agents.md) | ⭐⭐⭐⭐ | ReAct, Tool Use, LangChain Agent |
| [15_Multi_Agent_Systems.md](./15_Multi_Agent_Systems.md) | ⭐⭐⭐⭐ | CrewAI, AutoGen, LangGraph, agent orchestration |
| [16_Practical_Chatbot.md](./16_Practical_Chatbot.md) | ⭐⭐⭐⭐ | Building conversational AI systems |

### Section 5: Optimization & Alignment (17-20)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [17_Model_Quantization.md](./17_Model_Quantization.md) | ⭐⭐⭐ | INT8/INT4, GPTQ, AWQ, bitsandbytes |
| [18_Inference_Optimization.md](./18_Inference_Optimization.md) | ⭐⭐⭐ | vLLM, TGI, Speculative Decoding, PagedAttention |
| [19_RLHF_Alignment.md](./19_RLHF_Alignment.md) | ⭐⭐⭐⭐ | PPO, Reward Model, DPO, Constitutional AI |
| [20_Evaluation_Metrics.md](./20_Evaluation_Metrics.md) | ⭐⭐⭐ | BLEU, ROUGE, BERTScore, Human Eval, Benchmarks |

### Section 6: Production LLM Engineering (21-24)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [21_LLM_Security_Safety.md](./21_LLM_Security_Safety.md) | ⭐⭐⭐ | Prompt injection, guardrails, red teaming, PII redaction |
| [22_Structured_Output.md](./22_Structured_Output.md) | ⭐⭐⭐ | JSON mode, Pydantic parsing, instructor, schema design |
| [23_Function_Calling_Tools.md](./23_Function_Calling_Tools.md) | ⭐⭐⭐⭐ | OpenAI/Anthropic tool use, MCP, tool orchestration |
| [24_Production_LLM_Patterns.md](./24_Production_LLM_Patterns.md) | ⭐⭐⭐⭐ | Caching, cost optimization, observability, deployment |

### Section 7: Advanced Agent Engineering (25-27)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [25_Agent_Memory_and_Planning.md](./25_Agent_Memory_and_Planning.md) | ⭐⭐⭐⭐ | Memory architectures, planning frameworks, task decomposition |
| [26_Agent_Evaluation_and_Benchmarks.md](./26_Agent_Evaluation_and_Benchmarks.md) | ⭐⭐⭐⭐ | AgentBench, SWE-bench, failure analysis, custom evals |
| [27_Agent_Design_Patterns.md](./27_Agent_Design_Patterns.md) | ⭐⭐⭐⭐ | Orchestrator-worker, router, HITL, guardrailed agents |

---

## Prerequisites

- Deep_Learning folder (required)
  - Attention mechanism
  - Transformer architecture
- Advanced Python
- PyTorch basics

---

## Environment Setup

### Required Packages

```bash
# PyTorch
pip install torch torchvision torchaudio

# HuggingFace
pip install transformers datasets tokenizers accelerate

# PEFT
pip install peft bitsandbytes

# LangChain
pip install langchain langchain-community langchain-openai langgraph

# Vector Databases
pip install chromadb faiss-cpu sentence-transformers

# Agents
pip install crewai autogen

# Production
pip install instructor guardrails-ai vllm

# Others
pip install openai anthropic tiktoken numpy pandas
```

### API Key Setup

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export HUGGINGFACE_TOKEN="your-token"
```

---

## Recommended Learning Order

1. **NLP Basics (3 days)**: 01 → 02 → 03
2. **Pre-trained Models (5 days)**: 04 → 05 → 06 → 07 → 08
3. **RAG Systems (5 days)**: 09 → 10 → 11 → 12
4. **Agents (5 days)**: 13 → 14 → 15 → 16
5. **Optimization (4 days)**: 17 → 18 → 19 → 20
6. **Production (4 days)**: 21 → 22 → 23 → 24
7. **Advanced Agents (3 days)**: 25 → 26 → 27

---

## Related Materials

- [Deep_Learning/](../Deep_Learning/00_Overview.md) - Prerequisite (Transformer)
- [Foundation_Models/](../Foundation_Models/00_Overview.md) - Theoretical FM architectures and research
- [Python/](../Python/00_Overview.md) - Advanced Python

---

## Reference Links

- [HuggingFace Documentation](https://huggingface.co/docs)
- [LangChain Documentation](https://python.langchain.com/docs)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Anthropic API Reference](https://docs.anthropic.com)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
