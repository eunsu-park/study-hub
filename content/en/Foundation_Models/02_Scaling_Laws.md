# Scaling Laws

## Learning Objectives
- Understand the concept and mathematical form of Scaling Laws
- Compare Kaplan et al. vs Chinchilla laws
- Learn compute-optimal training strategies
- Grasp how to apply Scaling Laws in practice

---

## 1. What are Scaling Laws?

### 1.1 Definition

**Scaling Laws** are empirical laws that describe the relationship between **number of parameters (N)**, **amount of data (D)**, **compute (C)**, and **performance (Loss)** of models.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core Relationships in Scaling Laws            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss ≈ A/N^α + B/D^β + E                                       │
│                                                                 │
│  N = Number of model parameters                                 │
│  D = Number of training data tokens                             │
│  C = Compute (FLOPs) ≈ 6 × N × D                                │
│  E = Irreducible minimum loss (entropy of data)                 │
│                                                                 │
│  Key findings:                                                  │
│  • Loss decreases according to Power Law with respect to N, D   │
│  • When C is fixed, there exists an optimal ratio of N and D    │
│  • Larger models utilize data more efficiently                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Important?

```python
"""
Practical value of Scaling Laws:

1. Cost Prediction
   - Estimate required resources before training
   - "How much is needed to train a 10B model?"

2. Optimal Allocation
   - Decide model size vs data amount with fixed budget
   - "What's the best setup with $100M budget?"

3. Performance Prediction
   - Estimate large model performance from small models
   - "With current 7B model, how much better will 70B be?"

4. Research Planning
   - Determine research directions with high ROI
   - "Should we increase data or scale up model?"
"""
```

---

## 2. Kaplan Scaling Laws (2020)

### 2.1 OpenAI's Initial Research

Laws discovered in Kaplan et al.'s 2020 paper "Scaling Laws for Neural Language Models":

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kaplan Scaling Laws                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Loss vs Parameters                                          │
│     L(N) = (N_c / N)^α_N, where α_N ≈ 0.076                     │
│                                                                 │
│  2. Loss vs Data                                                │
│     L(D) = (D_c / D)^α_D, where α_D ≈ 0.095                     │
│                                                                 │
│  3. Loss vs Compute                                             │
│     L(C) = (C_c / C)^α_C, where α_C ≈ 0.050                     │
│                                                                 │
│  Key claims:                                                    │
│  • Parameter count is most important (α_N < α_D)                │
│  • For same compute, larger model + less data is better         │
│  • N ∝ C^0.73, D ∝ C^0.27 (Compute-optimal allocation)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Visualization

```
   Loss (Log)
       │
   3.5 ├─●───────────────────────────── 100M params
       │   ╲
   3.0 ├─────●─────────────────────────  1B params
       │       ╲
   2.5 ├─────────●───────────────────── 10B params
       │           ╲
   2.0 ├─────────────●─────────────────100B params
       │               ╲
   1.5 ├─────────────────●────────────  1T params (predicted)
       │
       └───┬───┬───┬───┬───┬───┬───┬──▶
          10^18  19   20   21   22   23   Compute (FLOPs)

   • Straight line = Power Law (linear in log scale)
   • Slope = α_C ≈ 0.05
```

### 2.3 Model Design Following Kaplan's Law

```python
"""
Example application of Kaplan's law:

Compute budget: 10^21 FLOPs

Kaplan optimal allocation:
- N ∝ C^0.73 → N ≈ 10^15 (about 1 trillion parameters?!)
- D ∝ C^0.27 → D ≈ 10^9 (about 1 billion tokens)

Problem:
- Model becomes too large with insufficient data
- GPT-3 (175B) followed this law but...
- Chinchilla refuted this
"""
```

---

## 3. Chinchilla Scaling Laws (2022)

### 3.1 DeepMind's Rediscovery

Hoffmann et al.'s "Training Compute-Optimal Large Language Models" revised Kaplan's law:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chinchilla Scaling Laws                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Key finding: Existing models are Under-trained!                │
│                                                                 │
│  Compute-optimal scaling:                                        │
│  • N ∝ C^0.5  (number of parameters)                            │
│  • D ∝ C^0.5  (number of data tokens)                           │
│  • i.e., N and D should increase at the same rate for optimality│
│                                                                 │
│  Practical rule:                                                │
│  D ≈ 20 × N  (tokens ≈ 20 × parameters)                         │
│                                                                 │
│  Examples:                                                      │
│  • 1B model → 20B tokens needed                                 │
│  • 7B model → 140B tokens needed                                │
│  • 70B model → 1.4T tokens needed                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Chinchilla vs Gopher Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│               Chinchilla (70B) vs Gopher (280B)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model      │ Parameters│ Train Tokens│ Compute   │ Performance│
│  ───────────│──────────│─────────────│───────────│────────────│
│  Gopher     │ 280B     │ 300B        │ 5.0×10^23 │ Baseline   │
│  Chinchilla │ 70B      │ 1.4T        │ 5.0×10^23 │ +10% better│
│                                                                 │
│  Conclusion:                                                    │
│  • 4x smaller model performs better with same compute!          │
│  • Gopher is Under-trained (insufficient data)                  │
│  • Simply increasing model size is inefficient                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Status of Existing Models

```
             Tokens (D)
                │
          10T   ├                               ● LLaMA 2 (2023)
                │                           ●
           1T   ├                       ● Chinchilla (Optimal)
                │                   ╱
         100B   ├               ╱       ● GPT-3 (Under-trained)
                │           ╱
          10B   ├       ╱
                │   ╱                   ● Gopher (Very Under-trained)
           1B   ├─
                └───┬───┬───┬───┬───┬───┬───┬───▶
                   1B  10B 100B  1T  10T      Parameters (N)

             ╱ = Compute-optimal frontier (D ≈ 20N)

             Points below the line are Under-trained
```

---

## 4. Mathematical Formulation

### 4.1 Loss Function

```python
"""
Mathematical form of Scaling Law:

1. Single Variable Scaling
   L(N) = (N_c / N)^α + L_∞     # Consider parameters only
   L(D) = (D_c / D)^β + L_∞     # Consider data only

2. Combined Scaling (Chinchilla)
   L(N, D) = E + A/N^α + B/D^β

   where:
   - E ≈ 1.69 (irreducible loss, data entropy)
   - A ≈ 406.4
   - B ≈ 410.7
   - α ≈ 0.34
   - β ≈ 0.28

3. Compute Perspective
   C ≈ 6 × N × D  (FLOPs for training)

   Optimization: min L(N, D) subject to C = 6ND

   Result: N* ∝ C^0.5, D* ∝ C^0.5
"""
```

### 4.2 Scaling Law Simulation with Python

```python
import numpy as np
import matplotlib.pyplot as plt

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Calculate Loss according to Chinchilla Scaling Law

    Args:
        N: Number of parameters (billions)
        D: Number of tokens (billions)

    Returns:
        Expected Loss (log of perplexity)
    """
    return E + A / (N ** alpha) + B / (D ** beta)

def optimal_allocation(compute_budget, flops_per_token=6):
    """
    Calculate optimal N, D for given compute budget

    Args:
        compute_budget: Total FLOPs (e.g., 10^23)
        flops_per_token: FLOPs per token (approximately 6N)

    Returns:
        optimal_N, optimal_D (in billions)
    """
    # Chinchilla optimal ratio: D ≈ 20N
    # C = 6 * N * D = 6 * N * 20N = 120 * N^2
    # N = sqrt(C / 120)

    optimal_N = np.sqrt(compute_budget / 120) / 1e9  # billions
    optimal_D = 20 * optimal_N                        # billions

    return optimal_N, optimal_D

# Example: 10^23 FLOPs budget
compute = 1e23
N_opt, D_opt = optimal_allocation(compute)
print(f"Compute budget: 10^23 FLOPs")
print(f"Optimal parameters: {N_opt:.1f}B")
print(f"Optimal tokens: {D_opt:.1f}B")
print(f"Expected loss: {chinchilla_loss(N_opt, D_opt):.3f}")

# Visualization: Loss according to N vs D
N_range = np.logspace(0, 3, 50)  # 1B to 1000B
D_range = np.logspace(0, 4, 50)  # 1B to 10000B

N_grid, D_grid = np.meshgrid(N_range, D_range)
Loss_grid = chinchilla_loss(N_grid, D_grid)

plt.figure(figsize=(10, 8))
plt.contour(N_grid, D_grid, Loss_grid, levels=20)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Parameters N (Billions)')
plt.ylabel('Tokens D (Billions)')
plt.title('Chinchilla Scaling Law: Loss Contours')
plt.colorbar(label='Loss')
plt.plot(N_range, 20*N_range, 'r--', label='Optimal ratio (D=20N)')
plt.legend()
plt.show()
```

---

## 5. Application in Real Models

### 5.1 Scaling Comparison of Major Models

| Model | Parameters (N) | Tokens (D) | D/N Ratio | Status |
|------|-------------|----------|----------|------|
| GPT-3 | 175B | 300B | 1.7 | Under-trained |
| Gopher | 280B | 300B | 1.1 | Very Under-trained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| LLaMA 1 | 65B | 1.4T | 21.5 | Near-optimal |
| LLaMA 2 | 70B | 2T | 28.6 | Slight Over-trained |
| Mistral | 7B | 8T (est.) | ~1000 | Over-trained |

### 5.2 Benefits of Over-training

```
┌─────────────────────────────────────────────────────────────────┐
│                Over-training Strategy (LLaMA 2, Mistral)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chinchilla is "training" optimal but not "deployment" optimal! │
│                                                                 │
│  From deployment perspective:                                   │
│  • Inference cost ∝ N (model size)                              │
│  • Training once, inference trillions of times                  │
│                                                                 │
│  Therefore:                                                     │
│  • Smaller model + more data = inference efficient              │
│  • "Inference-optimal" ≠ "Compute-optimal"                      │
│                                                                 │
│  LLaMA 2 strategy:                                              │
│  • 70B model with 2T tokens (D/N ≈ 29)                          │
│  • Train longer than Chinchilla                                 │
│  • Result: Better performance with smaller model                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Practical Guidelines

```python
"""
Scaling strategies in practice:

1. Research/Experimentation Phase (Compute-limited)
   - Follow Chinchilla rule: D ≈ 20N
   - Iterate quickly with smaller models

2. Production Deployment (Inference-limited)
   - Consider over-training: D > 20N
   - Smaller model + more data
   - Example: Mistral 7B > LLaMA 2 13B (on some tasks)

3. Budget Planning
   - C = 6 * N * D (FLOPs)
   - GPU hours ≈ C / (GPU_FLOPS * utilization)
   - Example: A100 80GB = ~300 TFLOPS (effective)

4. Scale-up Strategy
   - Tune hyperparameters with small models
   - Predict large model performance with Scaling Law
   - Execute large-scale training after validation
"""

def estimate_training_cost(N_billions, D_billions, gpu_price_per_hour=2.0):
    """
    Estimate training cost

    Args:
        N_billions: Number of parameters (B)
        D_billions: Number of tokens (B)
        gpu_price_per_hour: GPU cost per hour (USD)

    Returns:
        dict: Expected cost information
    """
    N = N_billions * 1e9
    D = D_billions * 1e9

    # 6ND FLOPs for training
    total_flops = 6 * N * D

    # A100 80GB: ~300 TFLOPS effective
    gpu_tflops = 300
    gpu_flops = gpu_tflops * 1e12

    # Total GPU time
    total_gpu_seconds = total_flops / gpu_flops
    total_gpu_hours = total_gpu_seconds / 3600

    # Cost
    total_cost = total_gpu_hours * gpu_price_per_hour

    return {
        "total_flops": f"{total_flops:.2e}",
        "gpu_hours": f"{total_gpu_hours:,.0f}",
        "cost_usd": f"${total_cost:,.0f}",
        "cost_with_8gpus": f"${total_cost/8:,.0f} ({total_gpu_hours/8:,.0f} hours)"
    }

# Example: LLaMA 2 7B training cost
cost_7b = estimate_training_cost(7, 2000)
print("LLaMA 2 7B (2T tokens):")
for k, v in cost_7b.items():
    print(f"  {k}: {v}")
```

---

## 6. Extensions of Scaling Laws

### 6.1 Scaling in Other Domains

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scaling Laws by Domain                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Vision (ViT):                                                  │
│  • Similar power law observed                                   │
│  • α ≈ 0.05 (smaller than Language)                            │
│  • Data quality is more important                               │
│                                                                 │
│  Multimodal (CLIP):                                             │
│  • Separate optimization needed for image and text scaling      │
│  • Quality of data pairs is critical                            │
│                                                                 │
│  Code:                                                          │
│  • Steeper scaling (larger α)                                   │
│  • High-quality code data is scarce                             │
│                                                                 │
│  Reasoning:                                                     │
│  • Not smooth due to emergent behavior                          │
│  • Sudden performance improvements at specific thresholds       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Fine-tuning Scaling Laws

```python
"""
Scaling Law also applies to fine-tuning:

Research findings:
- Larger base model = less fine-tuning data needed
- Fine-tuning data also scales according to power law
- PEFT like LoRA follows similar patterns

Practical rules:
- Base model size × 10 = Fine-tuning data amount (approximately)
- 7B model: ~1K-10K examples
- 70B model: ~100-1K examples (to achieve same performance)

However, quality > quantity:
- 100 high-quality examples > 10,000 low-quality examples
"""
```

### 6.3 Inference Scaling (Test-time Compute)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Scaling (o1-style)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Traditional Scaling: Increase compute during training          │
│  Inference Scaling: Increase compute during inference           │
│                                                                 │
│  Methods:                                                       │
│  • Generate longer Chain-of-Thought                             │
│  • Generate multiple answers and vote (Self-consistency)        │
│  • Tree of Thoughts / Beam Search                               │
│  • Iterative Verification/Refinement                            │
│                                                                 │
│  Effects:                                                       │
│  • Significantly improved accuracy on difficult problems        │
│  • Performance improvement possible without training            │
│  • Paradigm shift from GPT-4 → o1 (inference-time scaling)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Limitations of Scaling

### 7.1 Physical Limits

```python
"""
Real limitations of Scaling:

1. Data Limits
   - Total internet text: ~10-50T tokens
   - High-quality data is much less
   - As of 2024, data exhaustion discussion beginning

2. Compute Limits
   - Power consumption (MW scale)
   - Semiconductor supply
   - Cost (billions of dollars)

3. Architecture Limits
   - Attention's O(n²) complexity
   - Memory bandwidth bottleneck
   - Communication overhead in distributed training

4. Diminishing Returns
   - α ≈ 0.05 means 10x compute → ~12% loss reduction
   - Increasingly larger investments needed
"""
```

### 7.2 Improvement Directions Beyond Scaling

| Direction | Description | Examples |
|------|------|------|
| **Architecture** | More efficient structures | Mamba, RWKV, Hyena |
| **Data Quality** | High-quality data curation | Phi, LIMA |
| **Synthetic Data** | Generate training data with AI | Self-Instruct |
| **Efficient Training** | Improve training efficiency | Flash Attention, ZeRO |
| **Test-time Compute** | Increase compute during inference | CoT, Self-consistency, o1 |

---

## Summary

### Key Concepts
- **Scaling Laws**: Power law relationship between parameters, data, compute, and performance
- **Kaplan**: Prioritize N (large model + less data)
- **Chinchilla**: Balance N and D (D ≈ 20N)
- **Over-training**: Train smaller models longer for inference efficiency

### Practical Formulas
```
Compute-optimal: D ≈ 20 × N (tokens)
Training FLOPs: C ≈ 6 × N × D
Inference-optimal: Smaller N, larger D
```

### Next Steps
- [03_Emergent_Abilities.md](03_Emergent_Abilities.md): Emergent abilities at scale
- [08_LLaMA_Family.md](08_LLaMA_Family.md): Scaling application case (LLaMA)

---

## References

### Key Papers
- Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"

### Additional Resources
- [Epoch AI Compute Trends](https://epochai.org/trends)
- [AI Scaling Calculator](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/ai-scaling-calculator)

---

## Exercises

### Exercise 1: Chinchilla Optimal Allocation

Using the Chinchilla rule (D ≈ 20 × N) and the compute formula C ≈ 6 × N × D, answer the following:

1. If you have a compute budget of 6 × 10^23 FLOPs, what are the Chinchilla-optimal values for N (parameters) and D (training tokens)?
2. GPT-3 trained a 175B parameter model on 300B tokens. Is GPT-3 over-trained or under-trained according to Chinchilla? By what factor?

<details>
<summary>Show Answer</summary>

**Part 1: Chinchilla-optimal allocation for C = 6 × 10^23 FLOPs**

From the Chinchilla optimal rule: D = 20N, so:

```
C = 6 × N × D = 6 × N × 20N = 120 × N²
N² = C / 120 = 6×10²³ / 120 = 5×10²¹
N = √(5×10²¹) ≈ 2.24×10¹⁰ ≈ 22B parameters
D = 20 × 22B = 440B tokens
```

**Part 2: GPT-3 assessment**

Chinchilla-optimal for 175B params: D* = 20 × 175B = 3.5T tokens
GPT-3 actual: D = 300B tokens

300B / 3500B ≈ 0.086 → GPT-3 was trained on only ~8.6% of the optimal token count.
GPT-3 is severely **under-trained** by a factor of ~11.7× (3.5T / 300B).

</details>

---

### Exercise 2: Training Cost Estimation

Use the `estimate_training_cost` function logic from the lesson to estimate the training cost of a hypothetical 13B parameter model trained on 260B tokens (Chinchilla-optimal), assuming A100 GPUs at $2/hour.

Show your calculation step by step.

<details>
<summary>Show Answer</summary>

```python
N = 13e9    # 13B parameters
D = 260e9   # 260B tokens (= 20 × 13B, Chinchilla-optimal)

# Step 1: Total FLOPs
total_flops = 6 * N * D
# = 6 × 13×10⁹ × 260×10⁹
# = 6 × 3.38×10²¹
# = 2.028×10²² FLOPs

# Step 2: A100 effective throughput
gpu_flops_per_sec = 300e12  # 300 TFLOPS

# Step 3: Total GPU-seconds
gpu_seconds = total_flops / gpu_flops_per_sec
# = 2.028×10²² / 3×10¹⁴
# = 6.76×10⁷ GPU-seconds

# Step 4: GPU-hours
gpu_hours = gpu_seconds / 3600
# ≈ 18,778 GPU-hours

# Step 5: Cost
cost = gpu_hours * 2.0
# ≈ $37,556
```

Approximate training cost: **~$37,600** on a single A100 equivalent.
With 8 GPUs running in parallel: ~$4,700, taking ~2,347 GPU-hours of wall-clock time.

</details>

---

### Exercise 3: Over-training vs Compute-optimal Trade-off

Explain in your own words why a deployment-focused organization might choose to **over-train a smaller model** rather than follow the Chinchilla compute-optimal recipe. What are the trade-offs?

<details>
<summary>Show Answer</summary>

**Why over-train a smaller model:**

- **Inference cost dominates in production.** Inference cost scales with model size N (memory bandwidth, compute per token). If a model serves millions of requests per day, even a modest reduction in N (e.g., 7B vs 13B) saves enormous ongoing costs.
- **Train once, infer trillions of times.** The training cost is a one-time expense; inference costs accumulate indefinitely. Chinchilla is "training-optimal" but not "total-cost-of-ownership optimal."
- **Over-training improves the small model's quality** until it matches or exceeds a larger Chinchilla-optimal model on many benchmarks (Mistral 7B > LLaMA 2 13B on some tasks).

**Trade-offs:**

| Factor | Chinchilla-optimal | Over-training |
|--------|-------------------|---------------|
| Training efficiency | Maximum | Diminishing returns on data |
| Inference cost | Higher (larger N) | Lower (smaller N) |
| Deployment flexibility | Less portable | More portable / edge-friendly |
| Data requirement | Moderate | Very large (may exhaust quality data) |

The key insight: the "optimal" strategy depends on whether compute is measured at training time or total deployment lifetime.

</details>

---

### Exercise 4: Power Law Interpretation

The Chinchilla scaling law states: L(N, D) = E + A/N^α + B/D^β, where α ≈ 0.34 and β ≈ 0.28.

1. If you double the number of parameters N while keeping D fixed, by what factor does the parameter-dependent component A/N^α decrease?
2. Which has a stronger marginal effect on loss reduction: doubling N or doubling D? Explain why.

<details>
<summary>Show Answer</summary>

**Part 1: Effect of doubling N**

```
A / (2N)^α = A / (2^α × N^α) = (A / N^α) × (1 / 2^α)

Reduction factor = 1 / 2^0.34 ≈ 1 / 1.265 ≈ 0.790
```

Doubling N reduces the parameter component by ~21% (it becomes 0.79× of its original value).

**Part 2: Doubling N vs doubling D**

- Doubling N: parameter component multiplied by 2^(-0.34) ≈ 0.790
- Doubling D: data component multiplied by 2^(-0.28) ≈ 0.825

**Doubling N has a stronger effect** (0.790 < 0.825), because α = 0.34 > β = 0.28 — the loss decreases more steeply with parameters than with data.

However, in practice the total effect also depends on the current ratio of A/N^α vs B/D^β. When a model is under-trained (data is the bottleneck), adding data has larger absolute impact despite the smaller exponent.

</details>

---

### Exercise 5: Inference Scaling Analysis

OpenAI's o1 model family exemplifies "inference-time scaling" rather than traditional training-time scaling. Describe three concrete techniques that can improve model performance by spending more compute at inference time, and explain what task types benefit most from each.

<details>
<summary>Show Answer</summary>

| Technique | Mechanism | Best Task Types |
|-----------|-----------|-----------------|
| **Chain-of-Thought (CoT)** | Generate explicit reasoning steps before the final answer, using token generation budget for intermediate computation | Multi-step math, logical deduction, word problems |
| **Self-Consistency** | Sample multiple independent reasoning paths (high temperature), then take the majority vote on the final answer | Arithmetic, factual QA, tasks with a single correct answer where individual paths may err |
| **Tree of Thoughts (ToT) / Beam Search** | Explore a branching tree of intermediate reasoning states, evaluate each node, and prune unpromising branches | Planning tasks, puzzles (e.g., 24-game), code generation with verification, multi-hop reasoning |

**Why inference scaling is powerful:**
- No additional training required — the base model's capabilities are better elicited
- Can be applied retroactively to already-deployed models
- Particularly effective for tasks where correct reasoning chains exist in the training distribution (math, code, logic)

**Limitation:** Inference scaling increases latency and cost per query proportionally to the number of samples or tree nodes explored.

</details>
