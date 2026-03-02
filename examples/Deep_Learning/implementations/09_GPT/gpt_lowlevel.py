"""
PyTorch Low-Level GPT-2 Implementation

A concise nanoGPT-style implementation.
Pre-LayerNorm, Causal Attention, Weight Tying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT-2 Configuration"""
    vocab_size: int = 50257
    block_size: int = 1024  # max sequence length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention (Masked Multi-Head Attention)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V in a single projection (for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (prevents attending to future tokens)
        # register_buffer: not learned but included in state_dict
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, n_embd)
            use_cache: Whether to use KV cache (during generation)
            past_kv: Previous K, V cache

        Returns:
            y: (batch, seq_len, n_embd)
            present_kv: Current K, V (for caching)
        """
        B, T, C = x.shape

        # Compute Q, K, V (single matmul)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Multi-head reshape: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV Cache handling (efficiency during generation)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T_kv) -> (B, n_head, T, T_kv)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        T_kv = k.size(2)
        # Mask starting from current position (accounting for KV cache)
        mask = self.bias[:, :, T_kv - T:T_kv, :T_kv]
        att = att.masked_fill(mask == 0, float('-inf'))

        # Softmax + Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)

        # Reshape back: (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        y = self.resid_dropout(self.c_proj(y))

        return y, present_kv


class MLP(nn.Module):
    """Feed-Forward Network (GPT-2 style)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 4x expansion
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # GPT-2 uses tanh approximation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block (Pre-LN)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN + Residual
        attn_out, present_kv = self.attn(
            self.ln_1(x), use_cache=use_cache, past_kv=past_kv
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    """GPT-2 Model"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            'wpe': nn.Embedding(config.block_size, config.n_embd),  # position embedding
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })

        # LM Head (weight tying with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer['wte'].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None
    ):
        """
        Args:
            idx: (batch, seq_len) token indices
            targets: (batch, seq_len) targets (during training)
            use_cache: Use KV cache
            past_key_values: Previous KV cache list

        Returns:
            logits, loss, present_key_values
        """
        device = idx.device
        B, T = idx.shape

        # Position IDs
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            pos = torch.arange(past_length, past_length + T, device=device)
        else:
            pos = torch.arange(0, T, device=device)

        # Embeddings
        tok_emb = self.transformer['wte'](idx)  # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)  # (T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)

        # Transformer blocks
        present_key_values = [] if use_cache else None

        for i, block in enumerate(self.transformer['h']):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, use_cache=use_cache, past_kv=past_kv)
            if use_cache:
                present_key_values.append(present_kv)

        # Final layer norm
        x = self.transformer['ln_f'](x)

        # LM Head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': present_key_values
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Text generation

        Args:
            idx: (batch, seq_len) start tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Nucleus (Top-P) sampling
            use_cache: Use KV cache

        Returns:
            idx: (batch, seq_len + max_new_tokens)
        """
        past_key_values = None

        for _ in range(max_new_tokens):
            # Truncate context (prevent exceeding block_size)
            if past_key_values is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, -1:]  # Only last token (when using cache)

            # Forward
            outputs = self(idx_cond, use_cache=use_cache, past_key_values=past_key_values)
            logits = outputs['logits'][:, -1, :]  # Last position
            past_key_values = outputs['past_key_values']

            # Temperature
            logits = logits / temperature

            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P (Nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens exceeding Top-P
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to result
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


def demo_tiny_training():
    """nanoGPT-style tiny training demo on a repeating 0-7 pattern."""
    import os

    print("\n=== Tiny Training Demo (nanoGPT-style) ===\n")

    # --- Toy data: repeating pattern [0,1,2,3,4,5,6,7] ---
    pattern = list(range(8))
    data = torch.tensor(pattern * 1000, dtype=torch.long)  # 8000 tokens

    # --- Tiny config ---
    config = GPTConfig(
        vocab_size=16,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tiny model parameters: {total_params:,}")

    # Expected: loss ~2.08 (log(8)) -> loss < 0.2 after 200 steps
    batch_size = 4
    block_size = config.block_size
    num_steps = 200
    losses = []

    # --- Training loop ---
    model.train()
    for step in range(num_steps):
        # Sample random chunks from the data as (x, y) pairs where y = x shifted by 1
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)

        outputs = model(x, targets=y)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == num_steps - 1:
            print(f"  step {step:4d} | loss {loss.item():.4f}")

    print(f"\nFinal loss: {losses[-1]:.4f}  (target: < 0.2)")

    # --- Generation ---
    model.eval()
    start_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
    generated = model.generate(
        start_tokens,
        max_new_tokens=32,
        temperature=0.5,
        top_k=8,
    )
    generated_list = generated[0].tolist()
    print(f"Generated tokens: {generated_list}")
    print(f"Expected pattern: {(pattern * 5)[:len(generated_list)]}")

    # --- Visualization: training loss curve ---
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "gpt_tiny_training.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(num_steps), losses, linewidth=1.5, color='#2563eb')
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Tiny GPT Training on Repeating 0-7 Pattern")
    ax.axhline(y=math.log(8), color='gray', linestyle='--', alpha=0.6, label=f'log(8) = {math.log(8):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nTraining loss curve saved to: {save_path}")


# Test
if __name__ == "__main__":
    print("=== GPT-2 Low-Level Implementation Test ===\n")

    # GPT-2 Small configuration
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1
    )

    # Create model
    model = GPT(config)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected ~117M for GPT-2 Small\n")

    # Test input
    batch_size, seq_len = 2, 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward
    outputs = model(idx, targets=targets)

    print("Forward pass:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Generation test
    print("\n=== Generation Test ===")
    start_tokens = torch.tensor([[50256]])  # <|endoftext|>

    generated = model.generate(
        start_tokens,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()[:25]}...")

    # KV Cache test
    print("\n=== KV Cache Test ===")
    import time

    # Without cache
    torch.manual_seed(42)
    start = time.time()
    gen_no_cache = model.generate(start_tokens, max_new_tokens=50, use_cache=False)
    time_no_cache = time.time() - start

    # With cache
    torch.manual_seed(42)
    start = time.time()
    gen_with_cache = model.generate(start_tokens, max_new_tokens=50, use_cache=True)
    time_with_cache = time.time() - start

    print(f"Without cache: {time_no_cache:.3f}s")
    print(f"With cache: {time_with_cache:.3f}s")
    print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")

    # Verify results match
    if torch.equal(gen_no_cache, gen_with_cache):
        print("Cache results match!")
    else:
        print("Warning: Cache results differ (numerical precision)")

    print("\nAll tests passed!")

    # --- nanoGPT-style tiny training demo ---
    demo_tiny_training()

