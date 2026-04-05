# Model Selection Guide

## Claude Model Family Quick Reference

| Model | Best For | Context | Speed | Cost |
|-------|----------|---------|-------|------|
| **Opus 4** | Complex reasoning, research, multi-step coding | 200K | Slower | $$$ |
| **Sonnet 4** | Balanced daily coding, code review, refactoring | 200K | Fast | $$ |
| **Haiku 3.5** | Quick tasks, classification, simple Q&A | 200K | Fastest | $ |

## Decision Framework

```
Is the task complex (multi-file refactor, architecture design)?
  → Yes: Use Opus
  → No: Is latency critical (autocomplete, inline suggestions)?
    → Yes: Use Haiku
    → No: Use Sonnet (default)
```

## Token Estimation

```
Rule of thumb:
  1 token ≈ 4 characters (English)
  1 token ≈ 1.5 characters (Korean)
  1 line of code ≈ 10-15 tokens
  1 page of text ≈ 300-400 tokens
```

## Cost Comparison (per 1M tokens, as of 2025)

| Model | Input | Output |
|-------|-------|--------|
| Opus 4 | $15 | $75 |
| Sonnet 4 | $3 | $15 |
| Haiku 3.5 | $0.80 | $4 |
