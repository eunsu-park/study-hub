# 22. Inference 최적화

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. KV 캐시(KV cache) 단편화와 낮은 GPU 활용률 등 자기회귀(autoregressive) LLM 추론의 핵심 메모리 및 연산 병목을 파악할 수 있다
2. vLLM의 PagedAttention이 KV 캐시 메모리 단편화를 해소하고 서빙 처리량(throughput)을 높이는 원리를 설명할 수 있다
3. 소형 드래프트 모델(draft model)을 사용하여 토큰 후보를 제안하는 추론 가속 기법인 스페큘레이티브 디코딩(Speculative Decoding)을 구현할 수 있다
4. 연속 배칭(continuous batching), 텐서 병렬화(tensor parallelism), 양자화(quantization, INT8/INT4)를 독립적인 최적화 전략으로 비교할 수 있다
5. vLLM 또는 TGI(Text Generation Inference)를 사용하여 최적화된 LLM 추론 서버를 배포하고 지연 시간(latency)과 처리량(throughput)을 벤치마크할 수 있다

---

## 개요

LLM 추론(inference) 최적화는 프로덕션 환경에서 비용과 지연 시간을 줄이는 핵심 기술입니다. 이 레슨에서는 vLLM, TGI, Speculative Decoding 등을 다룹니다.

---

## 1. LLM 추론의 병목

### 1.1 Memory Bottleneck

```
LLM 추론 특성:
┌─────────────────────────────────────────────────────────┐
│  KV Cache 크기 계산:                                    │
│                                                         │
│  Memory = 2 × n_layers × n_heads × head_dim × seq_len  │
│                       × batch_size × dtype_size        │
│                                                         │
│  예: LLaMA-7B, batch=1, seq=2048, FP16                 │
│  = 2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes             │
│  = 1.07 GB per sequence                                │
│                                                         │
│  batch=32일 경우: ~34 GB (KV cache만)                   │
└─────────────────────────────────────────────────────────┘

문제:
1. GPU 메모리 제한
2. 가변 길이 시퀀스 → 메모리 단편화
3. 배치 크기 제한 → 낮은 처리량
```

### 1.2 Compute Bottleneck

```
Autoregressive 생성의 비효율:
┌────────────────────────────────────────────────────────┐
│  Step 1: [prompt] → token_1                            │
│  Step 2: [prompt, token_1] → token_2                   │
│  Step 3: [prompt, token_1, token_2] → token_3          │
│  ...                                                   │
│                                                        │
│  각 step에서:                                          │
│  - 전체 KV cache 로드                                   │
│  - 단 1개 토큰 생성                                     │
│  - GPU utilization 낮음 (memory-bound)                 │
└────────────────────────────────────────────────────────┘
```

---

## 2. vLLM

### 2.1 PagedAttention

```
PagedAttention 핵심 아이디어:
┌────────────────────────────────────────────────────────────┐
│  기존 방식: 연속 메모리 할당                                │
│                                                            │
│  Sequence A: [████████████████████░░░░░░]  (padding 낭비)  │
│  Sequence B: [██████████░░░░░░░░░░░░░░░░]  (더 많은 낭비)  │
│                                                            │
│  PagedAttention: 비연속 블록 할당                          │
│                                                            │
│  Block Pool: [B1][B2][B3][B4][B5][B6][B7][B8]...           │
│                                                            │
│  Sequence A → [B1, B3, B5, B7] (필요한 만큼만)             │
│  Sequence B → [B2, B4] (효율적)                            │
│                                                            │
│  장점:                                                     │
│  - 메모리 낭비 최소화                                      │
│  - 동적 할당/해제                                          │
│  - Copy-on-Write 지원 (beam search 효율화)                │
└────────────────────────────────────────────────────────────┘
```

### 2.2 vLLM 사용

```python
from vllm import LLM, SamplingParams

class VLLMInference:
    """vLLM 추론 엔진"""

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
        """배치 생성"""
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
        """스트리밍 생성"""
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        # Async engine 필요
        engine_args = AsyncEngineArgs(model=self.model_name)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 스트리밍 구현은 별도 async 코드 필요


# vLLM 서버 실행
"""
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 2 \
    --port 8000
"""

# OpenAI 호환 API 사용
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

### 3.1 TGI 특징

```
TGI (HuggingFace):
┌────────────────────────────────────────────────────────────┐
│  핵심 기능:                                                │
│  - Continuous batching                                     │
│  - Flash Attention 2                                       │
│  - Tensor parallelism                                      │
│  - Token streaming                                         │
│  - Quantization (GPTQ, AWQ, EETQ)                         │
│  - Watermarking                                            │
│                                                            │
│  지원 모델:                                                │
│  - LLaMA, Mistral, Falcon                                 │
│  - GPT-2, BLOOM, StarCoder                                │
│  - T5, BART                                               │
└────────────────────────────────────────────────────────────┘
```

### 3.2 TGI 사용

```python
# Docker로 TGI 실행
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
    """TGI 클라이언트"""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        self.client = InferenceClient(endpoint)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """생성"""
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
        """스트리밍 생성"""
        for token in self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True
        ):
            yield token

    def get_model_info(self):
        """모델 정보"""
        return self.client.get_model_info()


# 사용 예시
def tgi_example():
    client = TGIClient()

    # 일반 생성
    response = client.generate(
        "Write a short poem about AI:",
        max_new_tokens=100
    )
    print(response.generated_text)

    # 스트리밍
    print("\nStreaming:")
    for token in client.generate("Once upon a time,", stream=True):
        print(token, end="", flush=True)
```

---

## 4. Speculative Decoding

### 4.1 개념

```
Speculative Decoding:
┌────────────────────────────────────────────────────────────┐
│  아이디어: 작은 모델로 초안 생성 → 큰 모델로 검증          │
│                                                            │
│  일반 decoding:                                            │
│  Large Model: t1 → t2 → t3 → t4 → t5  (5 forward passes)  │
│                                                            │
│  Speculative decoding:                                     │
│  Draft Model: [t1, t2, t3, t4, t5]  (빠른 추측)           │
│  Large Model: verify all at once    (1 forward pass)      │
│                                                            │
│  결과: t1 ✓, t2 ✓, t3 ✗ → 재생성                          │
│                                                            │
│  속도 향상: 2-3x (acceptance rate에 따라)                 │
└────────────────────────────────────────────────────────────┘
```

### 4.2 구현

```python
import torch
from typing import Tuple

class SpeculativeDecoder:
    """Speculative Decoding 구현"""

    def __init__(
        self,
        target_model,  # 큰 모델
        draft_model,   # 작은 모델
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
        """Speculative decoding으로 생성"""
        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Draft model로 k개 토큰 추측
            draft_tokens, draft_probs = self._draft_tokens(
                generated, self.k, temperature
            )

            # 2. Target model로 검증
            accepted, target_probs = self._verify_tokens(
                generated, draft_tokens, temperature
            )

            # 3. 수락된 토큰 추가
            generated = torch.cat([generated, accepted], dim=1)

            # 4. 마지막 거절 위치에서 target으로 샘플링
            if accepted.shape[1] < self.k:
                # 일부 거절됨 → target에서 다음 토큰 샘플링
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
        """Draft model로 k개 토큰 생성"""
        draft_tokens = []
        draft_probs = []
        current = context

        for _ in range(k):
            outputs = self.draft(current)
            logits = outputs.logits[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)

            # 샘플링
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
        """Target model로 검증"""
        # 전체 시퀀스에 대해 한 번에 forward
        full_seq = torch.cat([context, draft_tokens], dim=1)
        outputs = self.target(full_seq)

        # Target probabilities
        target_logits = outputs.logits[:, context.shape[1]-1:-1] / temperature
        target_probs = torch.softmax(target_logits, dim=-1)

        # Draft probabilities (이미 계산됨)
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
        """Draft probs 재계산"""
        full_seq = torch.cat([context, draft_tokens], dim=1)
        outputs = self.draft(full_seq)
        logits = outputs.logits[:, context.shape[1]-1:-1] / temperature
        return torch.softmax(logits, dim=-1)

    def _sample_from_target(self, context, target_probs, temperature):
        """Target model에서 샘플링"""
        # Rejection 위치의 다음 토큰
        probs = target_probs[:, -1]
        return torch.multinomial(probs, num_samples=1)
```

---

## 5. 양자화 (Quantization)

### 5.1 양자화 방법 비교

| 방법 | 정밀도 | 속도 | 품질 | 메모리 |
|------|--------|------|------|--------|
| FP16 | 16-bit | 1x | 100% | 1x |
| GPTQ | 4-bit | ~1.5x | 98-99% | 0.25x |
| AWQ | 4-bit | ~2x | 98-99% | 0.25x |
| GGUF | 2-8bit | ~2x | 95-99% | 0.15-0.5x |
| bitsandbytes | 4/8-bit | ~1.2x | 97-99% | 0.25-0.5x |

### 5.2 양자화 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# bitsandbytes 4-bit
def load_4bit_model(model_name: str):
    """4-bit 양자화 모델 로드"""
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
    """GPTQ 양자화 모델 로드"""
    from auto_gptq import AutoGPTQForCausalLM

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        device_map="auto",
        use_safetensors=True
    )

    return model


# AWQ
def load_awq_model(model_name: str):
    """AWQ 양자화 모델 로드"""
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_quantized(
        model_name,
        fuse_layers=True,
        device_map="auto"
    )

    return model
```

---

## 6. 배치 처리 최적화

### 6.1 Continuous Batching

```python
class ContinuousBatcher:
    """Continuous Batching 구현"""

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

        # 활성 요청들
        self.active_requests = {}
        self.request_queue = []

    def add_request(self, request_id: str, prompt: str, max_tokens: int):
        """새 요청 추가"""
        tokens = self.tokenizer.encode(prompt)
        self.request_queue.append({
            "id": request_id,
            "tokens": tokens,
            "generated": [],
            "max_tokens": max_tokens
        })

    def step(self) -> dict:
        """한 스텝 처리"""
        # 1. 새 요청을 배치에 추가
        while (len(self.active_requests) < self.max_batch_size and
               self.request_queue):
            req = self.request_queue.pop(0)
            self.active_requests[req["id"]] = req

        if not self.active_requests:
            return {}

        # 2. 배치 구성
        batch_ids, batch_tokens = self._prepare_batch()

        # 3. Forward pass
        with torch.no_grad():
            outputs = self.model(batch_tokens)
            next_tokens = outputs.logits[:, -1].argmax(dim=-1)

        # 4. 결과 업데이트
        results = {}
        completed = []

        for i, req_id in enumerate(batch_ids):
            req = self.active_requests[req_id]
            token = next_tokens[i].item()
            req["generated"].append(token)

            # 완료 체크
            if (len(req["generated"]) >= req["max_tokens"] or
                token == self.tokenizer.eos_token_id):
                results[req_id] = self.tokenizer.decode(req["generated"])
                completed.append(req_id)

        # 5. 완료된 요청 제거
        for req_id in completed:
            del self.active_requests[req_id]

        return results

    def _prepare_batch(self):
        """배치 준비 (padding)"""
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

## 7. 성능 벤치마킹

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
    """추론 벤치마크"""
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

    # 메모리
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

## 핵심 정리

### 추론 최적화 기법
```
1. PagedAttention (vLLM): KV cache 효율화
2. Continuous Batching: 동적 배치 처리
3. Speculative Decoding: Draft+Verify
4. Quantization: 4-bit/8-bit 압축
5. Flash Attention: Memory-efficient attention
6. Tensor Parallelism: 다중 GPU 분산
```

### 도구 선택 가이드
```
- 고처리량 서빙: vLLM
- HuggingFace 통합: TGI
- 엣지 디바이스: llama.cpp + GGUF
- 개발/실험: Transformers + bitsandbytes
```

---

## 참고 자료

1. Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. Leviathan et al. (2023). "Fast Inference from Transformers via Speculative Decoding"
3. [vLLM Documentation](https://docs.vllm.ai/)

---

## 연습 문제

### 연습 문제 1: KV 캐시 메모리 계산
프로덕션 팀이 배치 크기(batch size) 16, 최대 시퀀스 길이 4096 토큰으로 LLaMA-7B를 서빙하려고 합니다. FP16(BF16) 기준 KV 캐시 메모리 요구량을 계산하세요. 그런 다음 GPU 메모리가 40GB로 제한될 때(모델 + KV 캐시 포함) 배치 크기에 어떤 영향이 생기는지 설명하세요.

```
# 모델 사양:
# - num_layers = 32
# - num_heads = 32
# - head_dim = 128
# - dtype = FP16 (2 bytes/value)
# 공식: 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_size
```

<details>
<summary>정답 보기</summary>

**시퀀스 하나에 대한 KV 캐시 계산**:
```python
num_layers = 32
num_heads = 32
head_dim = 128
seq_len = 4096
dtype_size = 2  # FP16 = 2 bytes

# K와 V에 대한 인수 2
kv_per_seq = 2 * num_layers * num_heads * head_dim * seq_len * dtype_size
           = 2 * 32 * 32 * 128 * 4096 * 2
           = 2,147,483,648 bytes
           = 시퀀스당 2 GB
```

**배치 크기 16의 경우**:
```python
kv_batch16 = 2 GB * 16 = 32 GB (KV 캐시만)
```

**모델 가중치 (FP16)**:
```python
# LLaMA-7B ≈ 70억 파라미터 × 2 바이트 = 14 GB
model_memory = 14 GB
```

**배치=16일 때 총 합계**: 14 GB + 32 GB = 46 GB — 40 GB 한계 초과.

**40 GB 내에서 최대 배치 크기**:
```python
available_for_kv = 40 GB - 14 GB = 26 GB
max_batch = 26 GB / 시퀀스당 2 GB = 13개 시퀀스
```

**최대 배치 크기 = 13** (원래 원했던 16이 아님).

**프로덕션에서의 중요성**: 이것이 페이지드어텐션(PagedAttention)이 해결하는 핵심 문제입니다. 정적 사전 할당(static pre-allocation)에서는 실제 출력이 100 토큰에 불과하더라도 각 시퀀스에 대해 *최대* 가능한 KV 캐시를 미리 예약해야 합니다. PagedAttention은 블록을 필요에 따라 할당하므로 평균 활용도가 훨씬 높아지고 동일한 GPU 메모리에 더 많은 동시 요청을 처리할 수 있습니다.

</details>

### 연습 문제 2: 추측적 디코딩(Speculative Decoding) 수용률 분석
`SpeculativeDecoder`는 `min(1, p_target / p_draft)`의 수용 확률을 사용합니다. 다음 각 경우에 속도 향상이 어떻게 달라지는지 분석하세요:
A) 드래프트 모델과 타겟 모델이 거의 동일한 분포를 생성하는 경우 (수용률 ≈ 95%)
B) 드래프트 모델이 나쁜 경우 (수용률 ≈ 30%)
C) 생성되는 텍스트가 희귀한 기술 용어 vs. 일반적인 구어체 문장인 경우

<details>
<summary>정답 보기</summary>

**속도 향상 공식**:
k개의 투기 토큰과 수용률 α에서, 타겟 순전파(forward pass)당 수용된 기대 토큰 수:
```
E[수용됨] = k * α  (단순화)
속도 향상 ≈ k * α / 1  (추측적 디코딩 없을 때 1 토큰/타겟 패스 대비)
```

**A) 높은 수용률 (α ≈ 95%, k=5)**:
```python
# 타겟 패스당 기대 토큰 ≈ 5 * 0.95 = 4.75
# 각 타겟 순전파가 ~4.75 토큰을 생성 (추측적 디코딩 없을 때의 1 대비)
# 이론적 속도 향상 ≈ 4.75x

# 실제로는: 드래프트 모델 오버헤드에 의해 제한됨
# 효과적인 속도 향상 ≈ 2-3x (드래프트 모델이 지연 추가)
# 최적 시나리오: 드래프트 모델이 타겟보다 5-10배 빠를 때
```
이것이 이상적인 경우입니다. 드래프트와 타겟이 유사한 아키텍처일 때 발생합니다 (예: LLaMA-68M이 LLaMA-7B의 드래프트로).

**B) 낮은 수용률 (α ≈ 30%, k=5)**:
```python
# 타겟 패스당 기대 토큰 ≈ 5 * 0.30 = 1.5
# 하지만: 타겟은 여전히 전체 k 토큰 시퀀스에서 실행 (부분 검증 불가)
# 오버헤드: 드래프트 모델이 k번 실행, 타겟이 k+1 토큰에서 한 번 실행
# 순 속도 향상 ≈ 1.5x minus 드래프트 오버헤드 → 추측 없는 경우보다 거의 빠르지 않음

# 최악의 경우: 드래프트가 지속적으로 틀리면, 거부된 토큰으로 시스템 오버헤드 발생
# 표준 디코딩보다 느려질 수 있습니다!
```

**C) 텍스트 유형 분석**:
- **일반적인 구문** ("the quick brown fox"): 드래프트 모델이 정확 → 높은 α → 좋은 속도 향상. 일반적인 시퀀스는 훈련 데이터에 자주 나타나 두 모델이 동의합니다.
- **희귀한 기술 용어** ("the CYP3A4 enzyme inhibits"): 드래프트 모델은 일반적인 단어를 추측하지만 타겟 모델은 특정 기술 용어를 예측 → 낮은 α → 낮은 속도 향상.

**핵심 통찰**: 추측적 디코딩은 출력이 드래프트 모델에 의해 "예측 가능"할 때 가장 잘 작동합니다. 코드 생성, 수학, 또는 도메인 특화 텍스트의 경우, 타겟과 동일한 도메인에 특화된 드래프트 모델을 선택하세요.

</details>

### 연습 문제 3: 양자화(Quantization) 방법 선택
팀이 세 가지 다른 시나리오에 13B 모델을 배포해야 합니다. 각 시나리오에 가장 적합한 양자화 방법을 {FP16, GPTQ-4비트, AWQ-4비트, GGUF-Q4} 중에서 선택하고 그 이유를 설명하세요.

| 시나리오 | 하드웨어 | 요구사항 | 방법 | 이유 |
|----------|----------|-------------|--------|--------|
| A) 고객 챗봇, 하루 1000건 | A100 80GB × 1 | 최고 품질 | ??? | ??? |
| B) 노트북에 엣지 배포 | CPU만, 16GB RAM | RAM에 맞아야 함 | ??? | ??? |
| C) 고처리량 API, 하루 10만 건 | A100 80GB × 2 | 처리량 우선 | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 시나리오 | 방법 | 이유 |
|----------|--------|--------|
| A) 고객 챗봇, 하루 1000건, A100 80GB | **FP16** | 13B × 2 바이트 = 26GB — 80GB A100에 쉽게 들어감. 하루 1000건(분당 ~0.7건)으로 처리량이 병목이 아닙니다. 전체 FP16 정밀도는 양자화 품질 손실을 방지하고, 각 고객 상호작용의 응답 품질을 극대화합니다. 메모리가 충분한데 하드웨어 절약을 위해 품질을 희생할 이유가 없습니다. |
| B) 엣지 배포, CPU만, 16GB RAM | **GGUF Q4_K_M** | CPU 추론은 llama.cpp + GGUF 형식이 필요합니다. 13B를 FP16으로 = 26GB — 16GB RAM에 맞지 않습니다. GGUF Q4 = 13B × 0.5 바이트 ≈ 6.5GB — 쉽게 맞습니다. llama.cpp는 GGUF를 사용한 CPU 추론에 최적화되어 있습니다. Q4_K_M 변형은 양자화된 CPU 추론에 대해 최적의 품질/크기 트레이드오프를 제공합니다. |
| C) 고처리량 API, 하루 10만 건, 2× A100 | **AWQ 4비트** | 하루 10만 건 = 분당 ~70건 — 처리량이 중요합니다. AWQ는 4비트 정밀도에서 FP16 대비 ~2배의 추론 속도 향상을 제공합니다. 13B AWQ ≈ GPU당 6.5GB — 2개의 A100에 텐서 병렬화(tensor parallelism)를 사용하면 동시에 ~60개 시퀀스 배치를 실행할 수 있습니다. AWQ는 처리량을 두 배로 늘리면서 FP16 품질의 98-99%를 유지합니다. GPTQ는 같은 비트 폭에서 AWQ보다 약간 느립니다. |

**핵심 원칙**: 올바른 양자화는 병목에 따라 다릅니다. 품질 우선 → FP16. 엣지 CPU → GGUF. 처리량 우선 → AWQ/GPTQ.

</details>

### 연습 문제 4: 연속 배칭(Continuous Batching) vs. 정적 배칭(Static Batching)
연속 배칭이 정적 배칭에 비해 GPU 처리량을 크게 향상시키는 이유를 설명하세요. 길이가 다른 3개의 동시 요청을 구체적인 예시로 사용하세요.

```
요청 A: 500 토큰 예상 (긴 응답)
요청 B: 20 토큰 예상 (짧은 응답)
요청 C: 200 토큰 예상 (중간 응답)
배치 크기 = 3
```

<details>
<summary>정답 보기</summary>

**정적 배칭 동작**:
```
타임스텝 0-20:   [A, B, C] — 3개 모두 생성 중
타임스텝 21:     B 완료. GPU가 [A, _, C] 실행 — 1/3 낭비
타임스텝 21-200: [A, _, C] — GPU 활용률 2/3만
타임스텝 201:    C 완료. GPU가 [A, _, _] 실행 — 2/3 낭비
타임스텝 201-500:[A, _, _] — GPU 활용률 1/3만

총 GPU 낭비: 타임스텝 21-500에 걸쳐 심각한 패딩 낭비
```

**연속 배칭 동작**:
```
타임스텝 0-20:   [A, B, C] — 3개 모두 생성 중 (배치 크기 = 3)
타임스텝 21:     B 완료 → 즉시 요청 D를 배치에 추가
타임스텝 21-??:  [A, C, D] — 배치가 꽉 찬 상태 유지
타임스텝 201:    C 완료 → 즉시 요청 E를 배치에 추가
타임스텝 201-??  [A, D, E] — 배치가 꽉 찬 상태 유지

GPU 활용률 ≈ 최대 용량으로 일정하게 유지
```

**핵심 통찰**: 정적 배칭에서는 시퀀스가 완료되면 배치의 모든 시퀀스가 완료될 때까지 해당 "슬롯"이 낭비됩니다. 연속 배칭에서는 반복 루프가 매 *단일 토큰 스텝* 후에 완료 여부를 확인하고 새로운 요청을 빈 슬롯에 즉시 삽입합니다.

**처리량 향상**:
- 정적: 500 스텝에 걸친 평균 배치 크기 = (3×20 + 2×180 + 1×300) / 500 = 평균 1.44
- 연속: 평균 배치 크기 ≈ 3 (새 요청으로 유지)
- 처리량 향상 ≈ 3/1.44 ≈ **2.1배** (이 예시 기준)

실제 향상은 종종 3-5배에 달하는데, 프로덕션 워크로드에서는 출력 길이 분산이 훨씬 크고 연속 배칭이 길이 분산에 관계없이 GPU 활용률을 거의 100%로 유지하기 때문입니다.

</details>
4. [TGI Documentation](https://huggingface.co/docs/text-generation-inference/)
