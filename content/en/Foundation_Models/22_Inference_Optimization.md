# 22. Inference Optimization

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify the key memory and compute bottlenecks in autoregressive LLM inference, including KV cache fragmentation and low GPU utilization
2. Explain how vLLM's PagedAttention eliminates KV cache memory fragmentation and increases serving throughput
3. Implement speculative decoding to accelerate inference by using a smaller draft model to propose token candidates
4. Compare continuous batching, tensor parallelism, and quantization (INT8/INT4) as orthogonal optimization strategies
5. Deploy an optimized LLM inference server using vLLM or Text Generation Inference (TGI) and benchmark its latency and throughput

---

## Overview

LLM inference optimization is a key technology for reducing costs and latency in production environments. This lesson covers vLLM, TGI, Speculative Decoding, and more.

---

## 1. LLM Inference Bottlenecks

### 1.1 Memory Bottleneck

```
LLM Inference Characteristics:
┌─────────────────────────────────────────────────────────┐
│  KV Cache Size Calculation:                             │
│                                                         │
│  Memory = 2 × n_layers × n_heads × head_dim × seq_len  │
│                       × batch_size × dtype_size        │
│                                                         │
│  Example: LLaMA-7B, batch=1, seq=2048, FP16            │
│  = 2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes             │
│  = 1.07 GB per sequence                                │
│                                                         │
│  With batch=32: ~34 GB (KV cache only)                 │
└─────────────────────────────────────────────────────────┘

Problems:
1. GPU memory limitations
2. Variable length sequences → Memory fragmentation
3. Batch size limitations → Low throughput
```

### 1.2 Compute Bottleneck

```
Autoregressive Generation Inefficiency:
┌────────────────────────────────────────────────────────┐
│  Step 1: [prompt] → token_1                            │
│  Step 2: [prompt, token_1] → token_2                   │
│  Step 3: [prompt, token_1, token_2] → token_3          │
│  ...                                                   │
│                                                        │
│  At each step:                                         │
│  - Load entire KV cache                                │
│  - Generate only 1 token                               │
│  - Low GPU utilization (memory-bound)                  │
└────────────────────────────────────────────────────────┘
```

---

## 2. vLLM

### 2.1 PagedAttention

```
PagedAttention Core Idea:
┌────────────────────────────────────────────────────────────┐
│  Traditional: Contiguous memory allocation                  │
│                                                            │
│  Sequence A: [████████████████████░░░░░░]  (padding waste) │
│  Sequence B: [██████████░░░░░░░░░░░░░░░░]  (more waste)    │
│                                                            │
│  PagedAttention: Non-contiguous block allocation           │
│                                                            │
│  Block Pool: [B1][B2][B3][B4][B5][B6][B7][B8]...           │
│                                                            │
│  Sequence A → [B1, B3, B5, B7] (only what's needed)        │
│  Sequence B → [B2, B4] (efficient)                         │
│                                                            │
│  Advantages:                                               │
│  - Minimize memory waste                                   │
│  - Dynamic allocation/deallocation                         │
│  - Copy-on-Write support (beam search efficiency)          │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Using vLLM

```python
from vllm import LLM, SamplingParams

class VLLMInference:
    """vLLM inference engine"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )

    def generate(
        self,
        prompts: list,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Batch generation"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append({
                "prompt": output.prompt,
                "generated": generated_text,
                "tokens": len(output.outputs[0].token_ids)
            })

        return results

    def streaming_generate(self, prompt: str, **kwargs):
        """Streaming generation"""
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        # Async engine needed
        engine_args = AsyncEngineArgs(model=self.model_name)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Streaming implementation requires separate async code


# Running vLLM server
"""
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 2 \
    --port 8000
"""

# Using OpenAI-compatible API
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 3. Text Generation Inference (TGI)

### 3.1 TGI Features

```
TGI (HuggingFace):
┌────────────────────────────────────────────────────────────┐
│  Core Features:                                            │
│  - Continuous batching                                     │
│  - Flash Attention 2                                       │
│  - Tensor parallelism                                      │
│  - Token streaming                                         │
│  - Quantization (GPTQ, AWQ, EETQ)                         │
│  - Watermarking                                            │
│                                                            │
│  Supported Models:                                         │
│  - LLaMA, Mistral, Falcon                                 │
│  - GPT-2, BLOOM, StarCoder                                │
│  - T5, BART                                               │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Using TGI

```python
# Running TGI with Docker
"""
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --num-shard 2 \
    --quantize awq
"""

from huggingface_hub import InferenceClient

class TGIClient:
    """TGI client"""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        self.client = InferenceClient(endpoint)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """Generate"""
        if stream:
            return self._stream_generate(prompt, max_new_tokens, temperature)

        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            details=True
        )

        return response

    def _stream_generate(self, prompt: str, max_new_tokens: int, temperature: float):
        """Streaming generation"""
        for token in self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True
        ):
            yield token

    def get_model_info(self):
        """Model info"""
        return self.client.get_model_info()


# Usage example
def tgi_example():
    client = TGIClient()

    # Normal generation
    response = client.generate(
        "Write a short poem about AI:",
        max_new_tokens=100
    )
    print(response.generated_text)

    # Streaming
    print("\nStreaming:")
    for token in client.generate("Once upon a time,", stream=True):
        print(token, end="", flush=True)
```

---

## 4. Speculative Decoding

### 4.1 Concept

```
Speculative Decoding:
┌────────────────────────────────────────────────────────────┐
│  Idea: Generate draft with small model → Verify with large │
│                                                            │
│  Normal decoding:                                          │
│  Large Model: t1 → t2 → t3 → t4 → t5  (5 forward passes)  │
│                                                            │
│  Speculative decoding:                                     │
│  Draft Model: [t1, t2, t3, t4, t5]  (fast speculation)    │
│  Large Model: verify all at once    (1 forward pass)      │
│                                                            │
│  Result: t1 ✓, t2 ✓, t3 ✗ → regenerate                    │
│                                                            │
│  Speedup: 2-3x (depends on acceptance rate)               │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
import torch
from typing import Tuple

class SpeculativeDecoder:
    """Speculative Decoding implementation"""

    def __init__(
        self,
        target_model,  # Large model
        draft_model,   # Small model
        tokenizer,
        num_speculative_tokens: int = 5
    ):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.k = num_speculative_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate with speculative decoding"""
        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Speculate k tokens with draft model
            draft_tokens, draft_probs = self._draft_tokens(
                generated, self.k, temperature
            )

            # 2. Verify with target model
            accepted, target_probs = self._verify_tokens(
                generated, draft_tokens, temperature
            )

            # 3. Add accepted tokens
            generated = torch.cat([generated, accepted], dim=1)

            # 4. Sample from target at last rejection position
            if accepted.shape[1] < self.k:
                # Some rejected → Sample next token from target
                next_token = self._sample_from_target(
                    generated, target_probs, temperature
                )
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _draft_tokens(
        self,
        context: torch.Tensor,
        k: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate k tokens with draft model"""
        draft_tokens = []
        draft_probs = []
        current = context

        for _ in range(k):
            outputs = self.draft(current)
            logits = outputs.logits[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)

            # Sampling
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            draft_probs.append(probs)

            current = torch.cat([current, token], dim=1)

        return torch.cat(draft_tokens, dim=1), torch.stack(draft_probs, dim=1)

    def _verify_tokens(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify with target model"""
        # Forward on entire sequence at once
        full_seq = torch.cat([context, draft_tokens], dim=1)
        outputs = self.target(full_seq)

        # Target probabilities
        target_logits = outputs.logits[:, context.shape[1]-1:-1] / temperature
        target_probs = torch.softmax(target_logits, dim=-1)

        # Draft probabilities (already computed)
        draft_probs = self._get_draft_probs(context, draft_tokens, temperature)

        # Acceptance probability: min(1, p_target / p_draft)
        accepted = []
        for i in range(draft_tokens.shape[1]):
            token = draft_tokens[:, i]
            p_target = target_probs[:, i].gather(1, token.unsqueeze(1))
            p_draft = draft_probs[:, i].gather(1, token.unsqueeze(1))

            accept_prob = torch.clamp(p_target / p_draft, max=1.0)

            if torch.rand(1) < accept_prob:
                accepted.append(token)
            else:
                break

        if accepted:
            return torch.stack(accepted, dim=1), target_probs
        else:
            return torch.tensor([]).reshape(1, 0), target_probs

    def _get_draft_probs(self, context, draft_tokens, temperature):
        """Recompute draft probs"""
        full_seq = torch.cat([context, draft_tokens], dim=1)
        outputs = self.draft(full_seq)
        logits = outputs.logits[:, context.shape[1]-1:-1] / temperature
        return torch.softmax(logits, dim=-1)

    def _sample_from_target(self, context, target_probs, temperature):
        """Sample from target model"""
        # Next token at rejection position
        probs = target_probs[:, -1]
        return torch.multinomial(probs, num_samples=1)
```

---

## 5. Quantization

### 5.1 Quantization Method Comparison

| Method | Precision | Speed | Quality | Memory |
|--------|-----------|-------|---------|--------|
| FP16 | 16-bit | 1x | 100% | 1x |
| GPTQ | 4-bit | ~1.5x | 98-99% | 0.25x |
| AWQ | 4-bit | ~2x | 98-99% | 0.25x |
| GGUF | 2-8bit | ~2x | 95-99% | 0.15-0.5x |
| bitsandbytes | 4/8-bit | ~1.2x | 97-99% | 0.25-0.5x |

### 5.2 Using Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# bitsandbytes 4-bit
def load_4bit_model(model_name: str):
    """Load 4-bit quantized model"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return model


# GPTQ
def load_gptq_model(model_name: str):
    """Load GPTQ quantized model"""
    from auto_gptq import AutoGPTQForCausalLM

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        device_map="auto",
        use_safetensors=True
    )

    return model


# AWQ
def load_awq_model(model_name: str):
    """Load AWQ quantized model"""
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_quantized(
        model_name,
        fuse_layers=True,
        device_map="auto"
    )

    return model
```

---

## 6. Batch Processing Optimization

### 6.1 Continuous Batching

```python
class ContinuousBatcher:
    """Continuous Batching implementation"""

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        max_seq_len: int = 2048
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Active requests
        self.active_requests = {}
        self.request_queue = []

    def add_request(self, request_id: str, prompt: str, max_tokens: int):
        """Add new request"""
        tokens = self.tokenizer.encode(prompt)
        self.request_queue.append({
            "id": request_id,
            "tokens": tokens,
            "generated": [],
            "max_tokens": max_tokens
        })

    def step(self) -> dict:
        """Process one step"""
        # 1. Add new requests to batch
        while (len(self.active_requests) < self.max_batch_size and
               self.request_queue):
            req = self.request_queue.pop(0)
            self.active_requests[req["id"]] = req

        if not self.active_requests:
            return {}

        # 2. Compose batch
        batch_ids, batch_tokens = self._prepare_batch()

        # 3. Forward pass
        with torch.no_grad():
            outputs = self.model(batch_tokens)
            next_tokens = outputs.logits[:, -1].argmax(dim=-1)

        # 4. Update results
        results = {}
        completed = []

        for i, req_id in enumerate(batch_ids):
            req = self.active_requests[req_id]
            token = next_tokens[i].item()
            req["generated"].append(token)

            # Check completion
            if (len(req["generated"]) >= req["max_tokens"] or
                token == self.tokenizer.eos_token_id):
                results[req_id] = self.tokenizer.decode(req["generated"])
                completed.append(req_id)

        # 5. Remove completed requests
        for req_id in completed:
            del self.active_requests[req_id]

        return results

    def _prepare_batch(self):
        """Prepare batch (with padding)"""
        batch_ids = list(self.active_requests.keys())
        sequences = []

        for req_id in batch_ids:
            req = self.active_requests[req_id]
            seq = req["tokens"] + req["generated"]
            sequences.append(seq)

        # Padding
        max_len = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)))

        return batch_ids, torch.tensor(padded)
```

---

## 7. Performance Benchmarking

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    throughput: float  # tokens/second
    latency_p50: float  # ms
    latency_p99: float  # ms
    memory_gb: float

def benchmark_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    num_runs: int = 10
) -> BenchmarkResult:
    """Inference benchmark"""
    latencies = []
    total_tokens = 0

    # Warmup
    for prompt in prompts[:2]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    for _ in range(num_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms
            total_tokens += outputs.shape[1] - inputs["input_ids"].shape[1]

    # Memory
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        memory_gb = 0

    latencies.sort()

    return BenchmarkResult(
        throughput=total_tokens / (sum(latencies) / 1000),
        latency_p50=latencies[len(latencies) // 2],
        latency_p99=latencies[int(len(latencies) * 0.99)],
        memory_gb=memory_gb
    )
```

---

## Key Summary

### Inference Optimization Techniques
```
1. PagedAttention (vLLM): KV cache efficiency
2. Continuous Batching: Dynamic batch processing
3. Speculative Decoding: Draft+Verify
4. Quantization: 4-bit/8-bit compression
5. Flash Attention: Memory-efficient attention
6. Tensor Parallelism: Multi-GPU distribution
```

### Tool Selection Guide
```
- High throughput serving: vLLM
- HuggingFace integration: TGI
- Edge devices: llama.cpp + GGUF
- Development/experimentation: Transformers + bitsandbytes
```

---

## References

1. Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. Leviathan et al. (2023). "Fast Inference from Transformers via Speculative Decoding"
3. [vLLM Documentation](https://docs.vllm.ai/)

---

## Exercises

### Exercise 1: KV Cache Memory Calculation
A production team wants to serve LLaMA-7B with a batch size of 16 and maximum sequence length of 4096 tokens. Calculate the KV cache memory requirement in FP16 (BF16). Then explain what happens to batch size if GPU memory is limited to 40GB total (model + KV cache).

```
# Model specs:
# - num_layers = 32
# - num_heads = 32
# - head_dim = 128
# - dtype = FP16 (2 bytes/value)
# Formula: 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_size
```

<details>
<summary>Show Answer</summary>

**KV cache calculation for one sequence**:
```python
num_layers = 32
num_heads = 32
head_dim = 128
seq_len = 4096
dtype_size = 2  # FP16 = 2 bytes

# Factor of 2 for K and V
kv_per_seq = 2 * num_layers * num_heads * head_dim * seq_len * dtype_size
           = 2 * 32 * 32 * 128 * 4096 * 2
           = 2,147,483,648 bytes
           = 2 GB per sequence
```

**For batch size 16**:
```python
kv_batch16 = 2 GB * 16 = 32 GB (KV cache alone)
```

**Model weights (FP16)**:
```python
# LLaMA-7B ≈ 7B parameters × 2 bytes = 14 GB
model_memory = 14 GB
```

**Total with batch=16**: 14 GB + 32 GB = 46 GB — exceeds the 40 GB limit.

**Maximum batch size within 40 GB**:
```python
available_for_kv = 40 GB - 14 GB = 26 GB
max_batch = 26 GB / 2 GB per seq = 13 sequences
```

**Maximum batch size = 13** (not 16 as originally desired).

**Why this matters for production**: This is the core problem that PagedAttention solves. With static pre-allocation, the system must reserve the *maximum* possible KV cache for each sequence upfront (even if the actual output is only 100 tokens). PagedAttention allocates blocks on demand, so average utilization is much higher and you can fit more concurrent requests into the same GPU memory.

</details>

### Exercise 2: Speculative Decoding Acceptance Rate Analysis
The `SpeculativeDecoder` uses an acceptance probability of `min(1, p_target / p_draft)`. Analyze what happens to speedup when:
A) The draft and target models produce nearly identical distributions (acceptance rate ≈ 95%)
B) The draft model is poor (acceptance rate ≈ 30%)
C) The generated text contains rare technical jargon vs. common conversational phrases

<details>
<summary>Show Answer</summary>

**Speedup formula**:
With k speculative tokens and acceptance rate α, expected tokens accepted per target forward pass:
```
E[accepted] = k * α  (simplified)
Speedup ≈ k * α / 1  (vs. 1 token per target pass)
```

**A) High acceptance rate (α ≈ 95%, k=5)**:
```python
# Expected tokens per target pass ≈ 5 * 0.95 = 4.75
# Each target forward pass produces ~4.75 tokens (vs 1 without speculative decoding)
# Theoretical speedup ≈ 4.75x

# In practice: limited by draft model overhead
# Effective speedup ≈ 2-3x (draft model adds latency)
# Best scenario: draft model runs 5-10x faster than target
```
This is the ideal case. Happens when draft and target are similar architectures (e.g., LLaMA-68M as draft for LLaMA-7B).

**B) Low acceptance rate (α ≈ 30%, k=5)**:
```python
# Expected tokens per target pass ≈ 5 * 0.30 = 1.5
# But: target still runs on full k-token sequence (can't partially verify)
# Overhead: draft model ran k times, target ran once on k+1 tokens
# Net speedup ≈ 1.5x minus draft overhead → barely faster than no speculation

# Worse case: if draft consistently wrong, system overhead from rejected tokens
# May be slower than standard decoding!
```

**C) Text type analysis**:
- **Common phrases** ("the quick brown fox"): draft model is accurate → high α → good speedup. Common sequences appear frequently in training data, both models agree.
- **Rare technical jargon** ("the CYP3A4 enzyme inhibits"): draft model may guess generic words while target model predicts specific technical terms → low α → poor speedup.

**Key insight**: Speculative decoding works best when the output is "predictable" by the draft model. For code generation, math, or domain-specific text, choose a draft model that specializes in the same domain as the target.

</details>

### Exercise 3: Quantization Method Selection
Your team needs to deploy a 13B model for three different scenarios. Select the most appropriate quantization method from {FP16, GPTQ-4bit, AWQ-4bit, GGUF-Q4} for each, justifying your choice.

| Scenario | Hardware | Requirements | Method | Reason |
|----------|----------|-------------|--------|--------|
| A) Customer chatbot, 1000 req/day | A100 80GB × 1 | Max quality | ??? | ??? |
| B) Edge deployment on laptop | CPU only, 16GB RAM | Must fit in RAM | ??? | ??? |
| C) High-throughput API, 100K req/day | A100 80GB × 2 | Throughput priority | ??? | ??? |

<details>
<summary>Show Answer</summary>

| Scenario | Method | Reason |
|----------|--------|--------|
| A) Customer chatbot, 1000 req/day, A100 80GB | **FP16** | 13B × 2 bytes = 26GB — easily fits in 80GB A100. With only 1000 req/day (~0.7 req/minute), throughput is not the bottleneck. Full FP16 precision avoids any quantization quality loss, maximizing response quality for each customer interaction. No reason to sacrifice quality for hardware savings when memory is ample. |
| B) Edge deployment, CPU only, 16GB RAM | **GGUF Q4_K_M** | CPU inference requires llama.cpp + GGUF format. 13B in FP16 = 26GB — doesn't fit in 16GB RAM. GGUF Q4 = 13B × 0.5 bytes ≈ 6.5GB — fits easily. llama.cpp is optimized for CPU inference with GGUF. The Q4_K_M variant gives best quality/size trade-off for quantized CPU inference. |
| C) High-throughput API, 100K req/day, 2× A100 | **AWQ 4-bit** | 100K req/day = ~70 req/minute — throughput is critical. AWQ provides ~2x inference speedup over FP16 at 4-bit precision. 13B AWQ ≈ 6.5GB per GPU — using tensor parallelism across 2 A100s, you can run batch sizes of ~60 sequences simultaneously. AWQ preserves 98-99% of FP16 quality while doubling throughput. GPTQ is slightly slower than AWQ at the same bit width. |

**Key principle**: The right quantization depends on the bottleneck. For quality → FP16. For edge CPU → GGUF. For throughput → AWQ/GPTQ.

</details>

### Exercise 4: Continuous Batching vs. Static Batching
Explain why continuous batching dramatically increases GPU throughput compared to static batching. Use a concrete example with 3 concurrent requests of different lengths.

```
Request A: expects 500 tokens (long response)
Request B: expects 20 tokens (short response)
Request C: expects 200 tokens (medium response)
Batch size = 3
```

<details>
<summary>Show Answer</summary>

**Static batching behavior**:
```
Time step 0-20:   [A, B, C] — all 3 generating
Time step 21:     B finishes. GPU now runs [A, _, C] — 1/3 wasted
Time step 21-200: [A, _, C] — only 2/3 GPU utilization
Time step 201:    C finishes. GPU runs [A, _, _] — 2/3 wasted
Time step 201-500:[A, _, _] — only 1/3 GPU utilization

Total GPU wasted: heavy padding waste across steps 21-500
```

**Continuous batching behavior**:
```
Time step 0-20:   [A, B, C] — all 3 generating (batch size = 3)
Time step 21:     B finishes → immediately add Request D to batch
Time step 21-??:  [A, C, D] — batch stays full
Time step 201:    C finishes → immediately add Request E to batch
Time step 201-??  [A, D, E] — batch stays full

GPU utilization ≈ constant at maximum capacity
```

**Key insight**: In static batching, once a sequence finishes, that "slot" is wasted until all sequences in the batch finish. In continuous batching, the iteration loop checks for completion after every *single token step* and immediately inserts new requests into freed slots.

**Throughput improvement**:
- Static: Average batch size over 500 steps = (3×20 + 2×180 + 1×300) / 500 = 1.44 average
- Continuous: Average batch size ≈ 3 (maintained with new requests)
- Throughput improvement ≈ 3/1.44 ≈ **2.1x** in this example

Real-world gains are often 3-5x because production workloads have much higher variance in output lengths, and continuous batching keeps GPU utilization near 100% regardless of length variance.

</details>
4. [TGI Documentation](https://huggingface.co/docs/text-generation-inference/)
