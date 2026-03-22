"""
Exercises for Lesson 46: State Space Models
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: Linear Recurrence vs Self-Attention Complexity ===
# Problem: Empirically compare the time and memory complexity of a linear
# recurrence (SSM-style) vs scaled dot-product attention for increasing
# sequence lengths.

def exercise_1():
    """Compare linear recurrence vs attention complexity."""
    import time

    d_model = 64

    def linear_recurrence(x, A, B, C):
        """Simple linear SSM recurrence: h_k = A h_{k-1} + B x_k, y_k = C h_k."""
        B_size, L, d = x.shape
        state_dim = A.shape[0]
        h = torch.zeros(B_size, state_dim)
        outputs = []
        for k in range(L):
            h = h @ A.T + x[:, k, :] @ B.T  # h: (B, state_dim)
            y = h @ C.T                        # y: (B, d)
            outputs.append(y.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def attention(x):
        """Scaled dot-product attention: O(L^2) memory."""
        L = x.shape[1]
        scale = x.shape[-1] ** -0.5
        scores = (x @ x.transpose(-2, -1)) * scale   # (B, L, L)
        attn = F.softmax(scores, dim=-1)
        return attn @ x

    state_dim = 16
    A = torch.randn(state_dim, state_dim) * 0.1
    B_ssm = torch.randn(state_dim, d_model) * 0.1
    C_ssm = torch.randn(d_model, state_dim) * 0.1

    print("  {:>8s}  {:>14s}  {:>14s}".format("Seq Len", "Recurrence (s)", "Attention (s)"))
    for L in [64, 256, 512, 1024]:
        x = torch.randn(1, L, d_model)

        t0 = time.perf_counter()
        _ = linear_recurrence(x, A, B_ssm, C_ssm)
        t_rec = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = attention(x)
        t_attn = time.perf_counter() - t0

        print("  {:>8d}  {:>14.4f}  {:>14.4f}".format(L, t_rec, t_attn))

    print("  Note: Recurrence is O(L*N), Attention is O(L^2*D)")


# === Exercise 2: Discrete SSM Layer (Recurrent Mode) ===
# Problem: Implement a basic discretized SSM layer using the ZOH
# approximation (Ā ≈ exp(ΔA), B̄ ≈ ΔB) and run it in recurrent mode.

def exercise_2():
    """Implement a discrete SSM layer and run in recurrent mode."""

    class SimpleSSMLayer(nn.Module):
        """
        Simplified SSM layer (per-channel, independent state spaces).
        For real S4/Mamba, A uses structured HiPPO initialization.
        """

        def __init__(self, d_model, state_dim=16):
            super().__init__()
            self.d_model = d_model
            self.state_dim = state_dim

            # SSM parameters (d_model independent channels, each with state_dim)
            # A: negative real (ensures stability): use -softplus for positive eigenvalues
            self.log_A = nn.Parameter(torch.randn(d_model, state_dim))
            self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
            self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
            self.D = nn.Parameter(torch.ones(d_model))

            # Learnable discretization step
            self.log_delta = nn.Parameter(torch.zeros(d_model))

        def discretize(self):
            """ZOH discretization: Ā = exp(Δ*A), B̄ ≈ Δ*B."""
            delta = F.softplus(self.log_delta)  # (d_model,)
            # Use negative A for stability
            A = -F.softplus(self.log_A)          # (d_model, state_dim)
            A_bar = torch.exp(delta.unsqueeze(-1) * A)   # (d_model, state_dim)
            B_bar = delta.unsqueeze(-1) * self.B         # (d_model, state_dim)
            return A_bar, B_bar

        def forward_recurrent(self, x):
            """
            Args:
                x: (batch, seq_len, d_model)
            Returns:
                y: (batch, seq_len, d_model)
            """
            B_size, L, d = x.shape
            A_bar, B_bar = self.discretize()  # (d_model, state_dim) each

            # Initialize hidden state
            h = torch.zeros(B_size, d, self.state_dim, device=x.device)

            outputs = []
            for k in range(L):
                x_k = x[:, k, :]  # (B, d_model)
                # h_k = Ā * h_{k-1} + B̄ * x_k  (element-wise along state_dim)
                h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_k.unsqueeze(-1)
                # y_k = C * h_k (sum over state_dim)
                y_k = (self.C.unsqueeze(0) * h).sum(-1)  # (B, d_model)
                y_k = y_k + self.D.unsqueeze(0) * x_k    # Skip connection
                outputs.append(y_k.unsqueeze(1))

            return torch.cat(outputs, dim=1)

    torch.manual_seed(42)
    d_model, state_dim = 16, 8
    ssm = SimpleSSMLayer(d_model, state_dim)

    x = torch.randn(2, 32, d_model)
    y = ssm.forward_recurrent(x)

    print("  Input:  shape={}".format(x.shape))
    print("  Output: shape={}".format(y.shape))
    print("  Shape preserved: {}".format(x.shape == y.shape))

    n_params = sum(p.numel() for p in ssm.parameters())
    print("  SSM parameters: {:,} (d_model={}, state_dim={})".format(
        n_params, d_model, state_dim))


# === Exercise 3: Selective State Space (Mamba-style Input-Dependent Parameters) ===
# Problem: Extend the basic SSM to make B and C input-dependent (selective).
# In Mamba, B, C, and Δ are computed from the input x via small linear projections,
# giving the model content-aware filtering ability.

def exercise_3():
    """Implement selective SSM (Mamba-style input-dependent B, C, delta)."""

    class SelectiveSSMLayer(nn.Module):
        """
        Simplified selective SSM layer where B, C, delta depend on input.
        This is the key innovation of Mamba vs S4.
        """

        def __init__(self, d_model, state_dim=16, dt_rank=None):
            super().__init__()
            self.d_model = d_model
            self.state_dim = state_dim
            dt_rank = dt_rank or max(1, d_model // 16)
            self.dt_rank = dt_rank

            # Fixed A (log parameterized for stability)
            self.log_A = nn.Parameter(
                torch.arange(1, state_dim + 1, dtype=torch.float).log().unsqueeze(0).expand(d_model, -1).clone()
            )

            # Fixed D (skip connection)
            self.D = nn.Parameter(torch.ones(d_model))

            # Input-dependent projections
            self.x_proj = nn.Linear(d_model, dt_rank + 2 * state_dim, bias=False)
            self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        def forward(self, x):
            """
            Args:
                x: (batch, seq_len, d_model)
            """
            B_size, L, d = x.shape

            # Project input to get delta, B, C
            x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*state_dim)
            dt_raw, B_sel, C_sel = x_proj.split(
                [self.dt_rank, self.state_dim, self.state_dim], dim=-1
            )
            # delta: (B, L, d_model) — input-dependent step size
            delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_model)

            # Discretize A (input-dependent via delta)
            A = -torch.exp(self.log_A)  # (d_model, state_dim)
            # A_bar: (B, L, d_model, state_dim) — changes per token!
            A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

            # Recurrent scan (simplified, O(L) sequential)
            h = torch.zeros(B_size, d, self.state_dim, device=x.device)
            outputs = []
            for k in range(L):
                x_k = x[:, k, :]        # (B, d_model)
                B_k = B_sel[:, k, :]    # (B, state_dim)
                C_k = C_sel[:, k, :]    # (B, state_dim)
                A_k = A_bar[:, k, :, :] # (B, d_model, state_dim)

                B_bar_k = delta[:, k, :].unsqueeze(-1) * B_k.unsqueeze(1)  # (B, d_model, state_dim)
                h = A_k * h + B_bar_k * x_k.unsqueeze(-1)
                y_k = (C_k.unsqueeze(1) * h).sum(-1) + self.D.unsqueeze(0) * x_k
                outputs.append(y_k.unsqueeze(1))

            return torch.cat(outputs, dim=1)

    torch.manual_seed(0)
    d_model, state_dim = 16, 8
    ssm = SelectiveSSMLayer(d_model, state_dim)

    x = torch.randn(2, 16, d_model)
    y = ssm(x)

    print("  Selective SSM input:  {}".format(x.shape))
    print("  Selective SSM output: {}".format(y.shape))
    print("  Key property: B, C, delta are input-dependent -> selective filtering")

    # Show that different inputs produce differently-weighted outputs
    x_zero = torch.zeros_like(x)
    y_zero = ssm(x_zero)
    print("  Output for x=0 is near-zero: {:.6f}".format(y_zero.abs().mean().item()))
    print("  Output for random x differs: {:.6f}".format(y.abs().mean().item()))


# === Exercise 4: SSM vs Transformer — Sequence Length Scaling ===
# Problem: Build a minimal SSM block and a Transformer attention block,
# run them at various sequence lengths, and compare throughput.
# Confirm that SSM inference is O(L) while attention is O(L^2).

def exercise_4():
    """Compare SSM and Transformer attention at different sequence lengths."""
    import time

    d_model = 32

    class SimpleAttention(nn.Module):
        """Single-head dot-product attention — O(L^2)."""

        def __init__(self, d_model):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, L, D = x.shape
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = qkv
            scale = D ** -0.5
            attn = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            return self.out(attn @ v)

    class MinimalSSM(nn.Module):
        """Minimal SSM block — O(L) via sequential recurrence."""

        def __init__(self, d_model, state_dim=16):
            super().__init__()
            self.A = nn.Parameter(-torch.rand(d_model, state_dim))
            self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
            self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
            self.D = nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            B_size, L, d = x.shape
            h = torch.zeros(B_size, d, 16, device=x.device)
            A_bar = torch.exp(self.A * 0.01)
            outputs = []
            for k in range(L):
                h = A_bar.unsqueeze(0) * h + self.B.unsqueeze(0) * x[:, k, :].unsqueeze(-1)
                y = (self.C.unsqueeze(0) * h).sum(-1) + self.D * x[:, k, :]
                outputs.append(y.unsqueeze(1))
            return torch.cat(outputs, dim=1)

    attn = SimpleAttention(d_model)
    ssm = MinimalSSM(d_model)

    print("  {:>8s}  {:>12s}  {:>12s}  {:>10s}".format(
        "Seq Len", "Attention(s)", "SSM(s)", "Speedup"))
    for L in [32, 128, 256, 512]:
        x = torch.randn(1, L, d_model)
        N = 10

        t0 = time.perf_counter()
        for _ in range(N):
            with torch.no_grad():
                _ = attn(x)
        t_attn = (time.perf_counter() - t0) / N

        t0 = time.perf_counter()
        for _ in range(N):
            with torch.no_grad():
                _ = ssm(x)
        t_ssm = (time.perf_counter() - t0) / N

        speedup = t_attn / t_ssm if t_ssm > 0 else float('inf')
        print("  {:>8d}  {:>12.5f}  {:>12.5f}  {:>10.2f}x".format(
            L, t_attn, t_ssm, speedup))

    print("  Note: at very long sequences, attention O(L^2) becomes bottleneck")


if __name__ == "__main__":
    print("=== Exercise 1: Linear Recurrence vs Attention Complexity ===")
    exercise_1()
    print("\n=== Exercise 2: Discrete SSM Layer (Recurrent Mode) ===")
    exercise_2()
    print("\n=== Exercise 3: Selective SSM (Mamba-style) ===")
    exercise_3()
    print("\n=== Exercise 4: SSM vs Transformer Sequence Length Scaling ===")
    exercise_4()
    print("\nAll exercises completed!")
