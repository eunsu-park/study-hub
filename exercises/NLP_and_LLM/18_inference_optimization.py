"""
Exercises for Lesson 18: Inference Optimization
Topic: NLP_and_LLM

Practice problems for LLM inference optimization techniques.
"""

import time
import math
from typing import List, Dict, Tuple
from collections import OrderedDict


# === Exercise 1: KV Cache Implementation ===
# Problem: Implement a simple key-value cache for transformer inference.
# The cache stores previously computed key and value tensors to avoid recomputation.

def exercise_1():
    """Implement a KV cache for autoregressive generation."""
    print("=" * 60)
    print("Exercise 1: KV Cache")
    print("=" * 60)

    class KVCache:
        """Simple KV cache for transformer inference."""

        def __init__(self, num_layers: int, max_seq_len: int):
            self.num_layers = num_layers
            self.max_seq_len = max_seq_len
            # TODO: Initialize cache storage for each layer
            self.keys: dict[int, list[list[float]]] = {i: [] for i in range(num_layers)}
            self.values: dict[int, list[list[float]]] = {i: [] for i in range(num_layers)}

        def append(self, layer: int, key: list[float], value: list[float]):
            """Append new K,V for a token at the given layer."""
            # TODO: Append and handle max_seq_len eviction
            if len(self.keys[layer]) >= self.max_seq_len:
                self.keys[layer].pop(0)
                self.values[layer].pop(0)
            self.keys[layer].append(key)
            self.values[layer].append(value)

        def get(self, layer: int) -> tuple[list, list]:
            """Get all cached K,V for a layer."""
            return self.keys[layer], self.values[layer]

        def seq_len(self, layer: int = 0) -> int:
            return len(self.keys[layer])

        def clear(self):
            for i in range(self.num_layers):
                self.keys[i] = []
                self.values[i] = []

    cache = KVCache(num_layers=4, max_seq_len=8)

    # Simulate caching 10 tokens across 4 layers
    for token_idx in range(10):
        for layer in range(4):
            key = [float(token_idx * 10 + layer)] * 4  # Simulated key vector
            value = [float(token_idx * 100 + layer)] * 4  # Simulated value vector
            cache.append(layer, key, value)

    print(f"Cache sequence length: {cache.seq_len()}")
    print(f"Max allowed: {cache.max_seq_len}")
    keys, values = cache.get(0)
    print(f"Layer 0 cached keys count: {len(keys)}")
    print(f"First cached key: {keys[0][:2]}...")
    print(f"Last cached key: {keys[-1][:2]}...")


# === Exercise 2: Speculative Decoding Simulation ===
# Problem: Simulate speculative decoding where a small "draft" model
# proposes tokens and a large "verifier" model accepts or rejects them.

def exercise_2():
    """Simulate speculative decoding."""
    print("\n" + "=" * 60)
    print("Exercise 2: Speculative Decoding")
    print("=" * 60)

    import random
    random.seed(42)

    vocab = ["the", "cat", "sat", "on", "mat", "a", "dog", "big", "small", "red"]

    def draft_model(context: list[str], n: int = 4) -> list[str]:
        """Fast but less accurate draft model."""
        return [random.choice(vocab) for _ in range(n)]

    def verify_model(context: list[str], proposed: list[str]) -> int:
        """Slower but more accurate verifier. Returns number of accepted tokens."""
        # TODO: Simulate verification - accept tokens with decreasing probability
        accepted = 0
        for i, token in enumerate(proposed):
            # Acceptance probability decreases with position
            accept_prob = 0.8 - i * 0.15
            if random.random() < accept_prob:
                accepted += 1
            else:
                break
        return accepted

    def speculative_decode(prompt: list[str], max_tokens: int = 20,
                          draft_size: int = 4) -> tuple[list[str], dict]:
        """Run speculative decoding."""
        context = list(prompt)
        stats = {"draft_calls": 0, "verify_calls": 0, "tokens_accepted": 0,
                 "tokens_rejected": 0, "total_generated": 0}

        while stats["total_generated"] < max_tokens:
            # Step 1: Draft model proposes tokens
            proposed = draft_model(context, n=draft_size)
            stats["draft_calls"] += 1

            # Step 2: Verifier accepts/rejects
            accepted = verify_model(context, proposed)
            stats["verify_calls"] += 1
            stats["tokens_accepted"] += accepted
            stats["tokens_rejected"] += draft_size - accepted

            # Accept tokens up to the accepted count
            context.extend(proposed[:accepted])
            stats["total_generated"] += accepted

            # If all rejected, generate one token with verifier
            if accepted == 0:
                fallback = random.choice(vocab)
                context.append(fallback)
                stats["total_generated"] += 1

        return context, stats

    prompt = ["the", "big"]
    result, stats = speculative_decode(prompt, max_tokens=15)

    print(f"Generated: {' '.join(result)}")
    print(f"Stats: {stats}")
    acceptance_rate = stats["tokens_accepted"] / (stats["tokens_accepted"] + stats["tokens_rejected"])
    print(f"Acceptance rate: {acceptance_rate:.2%}")


# === Exercise 3: Batching Strategies ===
# Problem: Implement static and continuous batching for inference requests.

def exercise_3():
    """Compare static vs continuous batching."""
    print("\n" + "=" * 60)
    print("Exercise 3: Batching Strategies")
    print("=" * 60)

    @dataclass
    class InferenceRequest:
        request_id: int
        input_length: int
        output_length: int
        arrival_time: float

    from dataclasses import dataclass

    @dataclass
    class InferenceRequest:
        request_id: int
        input_length: int
        output_length: int
        arrival_time: float

    def simulate_requests(n: int = 10) -> list[InferenceRequest]:
        import random
        random.seed(123)
        requests = []
        t = 0.0
        for i in range(n):
            requests.append(InferenceRequest(
                request_id=i,
                input_length=random.randint(10, 100),
                output_length=random.randint(5, 50),
                arrival_time=t,
            ))
            t += random.uniform(0.01, 0.1)
        return requests

    # TODO: Static batching - wait for batch_size requests, process together
    def static_batching(requests: list[InferenceRequest], batch_size: int = 4) -> dict:
        total_time = 0.0
        latencies = []

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            # Batch time = max output length in batch (padded)
            max_output = max(r.output_length for r in batch)
            batch_time = max_output * 0.01  # Time per token step
            total_time += batch_time

            for r in batch:
                waste = (max_output - r.output_length) * 0.01
                latencies.append(batch_time)

        avg_latency = sum(latencies) / len(latencies)
        return {"total_time": round(total_time, 3), "avg_latency": round(avg_latency, 3),
                "batches": math.ceil(len(requests) / batch_size)}

    # TODO: Continuous batching - requests can join/leave mid-batch
    def continuous_batching(requests: list[InferenceRequest], max_batch: int = 4) -> dict:
        total_time = 0.0
        latencies = []
        active: list[dict] = []  # {"request": req, "remaining": int}
        queue = list(requests)
        completed = 0

        while queue or active:
            # Fill batch
            while queue and len(active) < max_batch:
                r = queue.pop(0)
                active.append({"request": r, "remaining": r.output_length})

            if not active:
                break

            # Process one step
            step_time = 0.01
            total_time += step_time

            for item in active:
                item["remaining"] -= 1

            # Remove completed
            still_active = []
            for item in active:
                if item["remaining"] <= 0:
                    latencies.append(total_time - item["request"].arrival_time)
                    completed += 1
                else:
                    still_active.append(item)
            active = still_active

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        return {"total_time": round(total_time, 3), "avg_latency": round(avg_latency, 3),
                "completed": completed}

    requests = simulate_requests(12)

    static = static_batching(requests, batch_size=4)
    continuous = continuous_batching(requests, max_batch=4)

    print(f"Static batching:     {static}")
    print(f"Continuous batching: {continuous}")


# === Exercise 4: LRU Token Cache ===
# Problem: Implement an LRU cache for frequently repeated prompts.

def exercise_4():
    """Implement LRU cache for prompt prefix matching."""
    print("\n" + "=" * 60)
    print("Exercise 4: LRU Prompt Cache")
    print("=" * 60)

    class LRUCache:
        """Least Recently Used cache for prompt prefixes."""

        def __init__(self, max_size: int = 5):
            self.max_size = max_size
            self._cache: OrderedDict[str, str] = OrderedDict()
            self.hits = 0
            self.misses = 0

        # TODO: Implement get with LRU update
        def get(self, key: str) -> str | None:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

        # TODO: Implement put with eviction
        def put(self, key: str, value: str):
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)  # Remove oldest
            self._cache[key] = value

        def stats(self) -> dict:
            total = self.hits + self.misses
            return {"hits": self.hits, "misses": self.misses,
                    "hit_rate": round(self.hits / total, 3) if total > 0 else 0,
                    "size": len(self._cache)}

    cache = LRUCache(max_size=3)

    prompts = [
        "What is Python?",
        "Explain Docker",
        "What is Python?",  # Cache hit
        "What is Rust?",
        "Explain Docker",  # Cache hit
        "What is Go?",
        "What is Java?",   # Evicts oldest
        "What is Python?", # Cache miss (evicted)
    ]

    for prompt in prompts:
        result = cache.get(prompt)
        if result:
            print(f"  HIT:  '{prompt}' -> '{result[:30]}'")
        else:
            response = f"Response for: {prompt}"
            cache.put(prompt, response)
            print(f"  MISS: '{prompt}' -> generated and cached")

    print(f"\nCache stats: {cache.stats()}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
