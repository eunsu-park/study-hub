"""
Exercises for Lesson 45: Diffusion Models — Advanced Topics
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn


# === Exercise 1: Classifier-Free Guidance (CFG) ===
# Problem: Implement the CFG sampling formula.
# Given unconditional and conditional noise predictions, compute the guided
# prediction at guidance_scale = 7.5 and guidance_scale = 1.0, and observe
# how the scale interpolates/extrapolates between the two estimates.

def exercise_1():
    """Classifier-free guidance: blend conditional and unconditional predictions."""
    torch.manual_seed(42)
    B, C, H, W = 2, 4, 8, 8  # Small spatial size for demo

    # Simulate noise predictions from the UNet
    eps_uncond = torch.randn(B, C, H, W)          # Unconditional epsilon
    eps_cond = eps_uncond + 0.5 * torch.randn_like(eps_uncond)  # Conditional adds signal

    def cfg(eps_uncond, eps_cond, guidance_scale):
        # eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    for w in [1.0, 3.0, 7.5, 15.0]:
        eps_guided = cfg(eps_uncond, eps_cond, w)
        diff_from_uncond = (eps_guided - eps_uncond).abs().mean().item()
        print("  guidance_scale={:5.1f}: mean |eps_guided - eps_uncond| = {:.4f}".format(
            w, diff_from_uncond))

    print("  Observation: higher guidance_scale amplifies the conditional signal.")

    # Dynamic rescaling (Imagen-style) to prevent saturation
    def dynamic_cfg(eps_uncond, eps_cond, guidance_scale, rescale=0.7):
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        # Rescale std to avoid oversaturation
        std_guided = eps_guided.std(dim=list(range(1, eps_guided.ndim)), keepdim=True)
        std_cond = eps_cond.std(dim=list(range(1, eps_cond.ndim)), keepdim=True)
        factor = std_cond / (std_guided + 1e-8)
        factor = rescale * factor + (1 - rescale)
        return eps_guided * factor

    eps_dyn = dynamic_cfg(eps_uncond, eps_cond, 7.5)
    print("  Dynamic CFG std ratio: {:.4f}".format(
        eps_dyn.std().item() / cfg(eps_uncond, eps_cond, 7.5).std().item()))


# === Exercise 2: DDIM Noise Schedule and Timestep Subsequence ===
# Problem: Build the DDPM noise schedule (linear beta schedule), compute
# cumulative alphas (alpha_bar), and select a subsequence of T_inf steps
# for DDIM sampling (showing that fewer steps are possible).

def exercise_2():
    """DDPM noise schedule and DDIM timestep subsampling."""
    T = 1000
    beta_start, beta_end = 1e-4, 0.02

    # Linear beta schedule
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t

    print("  beta[0] = {:.6f}, beta[-1] = {:.6f}".format(betas[0].item(), betas[-1].item()))
    print("  ᾱ[0]    = {:.6f} (almost 1 — barely noisy)".format(alphas_cumprod[0].item()))
    print("  ᾱ[-1]   = {:.6f} (near 0 — fully noisy)".format(alphas_cumprod[-1].item()))

    # DDIM: select T_inf evenly spaced timesteps
    for T_inf in [1000, 200, 50, 20, 10]:
        step_ratio = T // T_inf
        timesteps = (np.arange(0, T_inf) * step_ratio).astype(int)
        # Reverse: T -> 0
        timesteps = timesteps[::-1]
        speedup = T / T_inf
        print("  T_inf={:4d}: timesteps[0]={:4d}, speedup={:.0f}x".format(
            T_inf, int(timesteps[0]), speedup))

    # Signal-to-noise ratio at key timesteps
    snr = alphas_cumprod / (1 - alphas_cumprod)
    print("  SNR at t=0:    {:.2f}".format(snr[0].item()))
    print("  SNR at t=500:  {:.4f}".format(snr[499].item()))
    print("  SNR at t=999:  {:.6f}".format(snr[-1].item()))


# === Exercise 3: DDIM Deterministic Sampling Step ===
# Problem: Implement a single DDIM reverse step.
# Given x_t, the predicted noise epsilon, and the noise schedule,
# compute x_{t-1} deterministically (eta=0 means no added noise).

def exercise_3():
    """Single DDIM reverse step (deterministic, eta=0)."""
    torch.manual_seed(0)

    # Noise schedule
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def ddim_step(x_t, t, t_prev, eps_theta, alphas_cumprod, eta=0.0):
        """
        Perform one DDIM reverse step.

        Args:
            x_t:            noisy sample at timestep t, shape (B, C, H, W)
            t:              current timestep index (integer)
            t_prev:         previous timestep index (integer, t_prev < t)
            eps_theta:      predicted noise from the model at (x_t, t)
            alphas_cumprod: cumulative product of alphas, shape (T,)
            eta:            stochasticity (0=deterministic DDIM, 1=DDPM)

        Returns:
            x_{t-1}: denoised sample at previous timestep
        """
        abar_t = alphas_cumprod[t]
        abar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predicted x_0 from the current noisy sample and noise prediction
        x0_pred = (x_t - (1 - abar_t).sqrt() * eps_theta) / abar_t.sqrt()

        # Direction pointing to x_t
        sigma_t = eta * ((1 - abar_t_prev) / (1 - abar_t)).sqrt() * (1 - abar_t / abar_t_prev).sqrt()
        eps_direction = (1 - abar_t_prev - sigma_t**2).clamp(min=0).sqrt() * eps_theta

        # Random noise term
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        x_prev = abar_t_prev.sqrt() * x0_pred + eps_direction + sigma_t * noise
        return x_prev

    # Simulate one step at t=500 -> t=499
    B, C, H, W = 1, 3, 8, 8
    x_500 = torch.randn(B, C, H, W)
    eps_pred = torch.randn(B, C, H, W)

    x_499 = ddim_step(x_500, 500, 499, eps_pred, alphas_cumprod, eta=0.0)
    print("  DDIM step (deterministic): x_500 -> x_499")
    print("  x_500 std = {:.4f}".format(x_500.std().item()))
    print("  x_499 std = {:.4f}".format(x_499.std().item()))
    print("  Different values (not same as x_500): {}".format(
        not torch.allclose(x_500, x_499)))

    # Stochastic version (eta=1, equivalent to DDPM)
    x_499_stochastic = ddim_step(x_500, 500, 499, eps_pred, alphas_cumprod, eta=1.0)
    diff = (x_499_stochastic - x_499).abs().mean().item()
    print("  Stochastic vs deterministic difference (eta=1 vs 0): {:.4f}".format(diff))


# === Exercise 4: LoRA Fine-Tuning Mechanics ===
# Problem: Implement a LoRA (Low-Rank Adaptation) linear layer.
# LoRA adds trainable low-rank matrices (A, B) while freezing the original weights.
# Show that LoRA drastically reduces trainable parameters for fine-tuning.

def exercise_4():
    """Implement LoRA and compare trainable parameter counts."""

    class LoRALinear(nn.Module):
        """Linear layer with LoRA adaptation."""

        def __init__(self, in_features, out_features, rank=4, alpha=16.0, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.scaling = alpha / rank

            # Original frozen weights
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.02, requires_grad=False
            )
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
            else:
                self.bias = None

            # LoRA low-rank matrices (trainable)
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        def forward(self, x):
            # Original output (frozen)
            original = x @ self.weight.T
            if self.bias is not None:
                original = original + self.bias

            # LoRA delta: x @ A^T @ B^T * scaling
            lora_delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            return original + lora_delta

    in_feat, out_feat = 768, 768  # Typical transformer hidden size

    # Standard linear layer
    linear = nn.Linear(in_feat, out_feat)
    total_params = sum(p.numel() for p in linear.parameters())
    print("  Standard Linear params: {:,}".format(total_params))

    # LoRA variants
    for rank in [4, 8, 16, 32]:
        lora = LoRALinear(in_feat, out_feat, rank=rank)
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
        reduction = total_params / trainable
        print("  LoRA rank={:2d}: trainable={:,} ({:.1f}x fewer), frozen={:,}".format(
            rank, trainable, reduction, frozen))

    # Verify forward pass
    lora = LoRALinear(in_feat, out_feat, rank=8)
    x = torch.randn(4, in_feat)
    out = lora(x)
    print("  LoRA output shape: {}".format(out.shape))


if __name__ == "__main__":
    print("=== Exercise 1: Classifier-Free Guidance ===")
    exercise_1()
    print("\n=== Exercise 2: DDIM Noise Schedule and Timestep Subsampling ===")
    exercise_2()
    print("\n=== Exercise 3: DDIM Deterministic Sampling Step ===")
    exercise_3()
    print("\n=== Exercise 4: LoRA Fine-Tuning Mechanics ===")
    exercise_4()
    print("\nAll exercises completed!")
