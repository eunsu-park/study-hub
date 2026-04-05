# Flagship Projects

## Introduction

A **Flagship project** is a self-contained, single-file implementation that proves you understand a concept by building it from scratch. Each project deliberately crosses multiple topic boundaries — combining deep learning theory with probability, calculus, or reinforcement learning — while keeping external dependencies to an absolute minimum.

The name "Flagship" reflects the idea that each implementation is a showcase piece: small enough to read in one sitting, yet complete enough to train and produce real results.

## What Makes a Flagship Project

1. **Single file**: The entire implementation lives in one `.py` file. No package structure, no imports from sibling modules.
2. **Compact**: Each file targets 200-400 lines of code (excluding comments). If it grows beyond that, the scope is too large.
3. **Minimal dependencies**: Only NumPy is allowed. No PyTorch, no TensorFlow, no JAX. The point is to implement the machinery yourself.
4. **Runnable**: Every file includes a `if __name__ == "__main__"` block that trains on synthetic data and prints results. No dataset downloads required.
5. **Cross-topic**: Each project draws on at least two study topics, reinforcing connections between subjects.

## Naming Convention

Files use a **size prefix** that hints at implementation scope:

| Prefix | Approximate Scale | Meaning |
|--------|------------------|---------|
| `pico_` | ~150-200 lines | Smallest viable implementation |
| `nano_` | ~200-250 lines | Small but with a meaningful training loop |
| `micro_` | ~250-350 lines | Moderate complexity, multiple components |
| `tiny_` | ~300-400 lines | Largest scope, adversarial or multi-network |

## Prerequisite Topics

Each project assumes familiarity with specific study topics. Complete those topics first (or at least skim their overviews) before attempting the corresponding Flagship.

| Project | Prerequisites |
|---------|--------------|
| Micro Autograd | Deep_Learning (L01-L03), Math_for_AI, Calculus_and_Differential_Equations |
| Tiny GAN | Deep_Learning (L01-L06, L28-L30), Probability_and_Statistics |
| Nano RL | Reinforcement_Learning, Probability_and_Statistics |
| Pico Diffusion | Deep_Learning (L01-L06, L31-L33), Probability_and_Statistics |
| Micro VAE | Deep_Learning (L01-L06, L28-L29), Probability_and_Statistics |

## Recommended Order

The projects are ordered by conceptual progression:

```
Micro Autograd ──> Tiny GAN ──> Micro VAE ──> Pico Diffusion ──> Nano RL
(foundations)     (adversarial) (latent vars) (denoising)       (sequential decisions)
```

Start with **Micro Autograd** — it builds the automatic differentiation engine that every other project relies on conceptually. Then proceed through the generative models (GAN, VAE, Diffusion) before tackling reinforcement learning.

## Project Catalog

| # | File | Title | Lines | Related Topics | Key Concept |
|---|------|-------|-------|---------------|-------------|
| 01 | `micro_autograd.py` | Micro Autograd | ~300 | Deep_Learning, Math_for_AI, Calculus | Reverse-mode autodiff engine + tiny neural network |
| 02 | `tiny_gan.py` | Tiny GAN | ~350 | Deep_Learning, Probability_and_Statistics | Generator vs. Discriminator adversarial training |
| 03 | `nano_rl.py` | Nano RL | ~250 | Reinforcement_Learning, Probability_and_Statistics | REINFORCE with baseline on a GridWorld |
| 04 | `pico_diffusion.py` | Pico Diffusion | ~200 | Deep_Learning, Probability_and_Statistics | Forward/reverse diffusion on 1-D data |
| 05 | `micro_vae.py` | Micro VAE | ~300 | Deep_Learning, Probability_and_Statistics | Encoder-decoder with reparameterization trick |

## How to Use These Guides

Each lesson (01-05) follows the same structure:

1. **Learning Objectives** — what you will be able to do after completing the lesson
2. **Theory** — the mathematical and conceptual background
3. **Implementation Walkthrough** — a guided tour through the source code
4. **Exercises** — extensions and experiments to deepen understanding
5. **References** — original papers and related resources

Read the lesson alongside the corresponding `.py` file. The lesson explains *why* each design choice was made; the code shows *how*.

## File List

| Lesson | Filename | Description |
|--------|----------|-------------|
| L00 | `00_Overview.md` | This overview |
| L01 | `01_Micro_Autograd.md` | Reverse-mode autodiff engine from scratch |
| L02 | `02_Tiny_GAN.md` | Generative Adversarial Network from scratch |
| L03 | `03_Nano_RL.md` | REINFORCE policy gradient agent from scratch |
| L04 | `04_Pico_Diffusion.md` | Denoising diffusion model from scratch |
| L05 | `05_Micro_VAE.md` | Variational Autoencoder from scratch |
