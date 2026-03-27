# DL_Scratch_C — Exercise Index

Hands-on exercises that accompany the *Deep Learning from Scratch in C* course (L01–L45).
Each exercise file maps to one block of lessons.

Difficulty scale: ★ easy / ★★ medium / ★★★ hard / ★★★★ very hard

---

## Block 1 — Tensors & Autograd (L01–L07)

File: `block1_tensor_autograd.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 1.1 | `matrix_add_inplace` with broadcasting `[N,1] + [N,M]` | ★ |
| 1.2 | `softmax_2d` row-wise with numerical stability | ★★ |
| 1.3 | Extend autograd engine: `relu` and `sigmoid` with backward | ★★ |
| 1.4 | Arena allocator with `arena_reset()` (no free) | ★★ |
| 1.5 | Verify `matmul_backward` via finite differences (ε=1e-4) | ★★★ |

---

## Block 2 — CNN Foundations (L08–L14)

File: `block2_cnn_foundations.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 2.1 | `conv2d_dilated` (dilation > 1) and output shape check | ★★ |
| 2.2 | `col2im` — verify `col2im(im2col(x))` ≈ x on uniform image | ★★ |
| 2.3 | `avgpool_backward` for non-overlapping windows | ★★ |
| 2.4 | BatchNorm with momentum=0.9 EMA for running stats | ★★ |
| 2.5 | Horizontal flip augmentation in DataLoader | ★ |
| 2.6 | Train 2-layer CNN on CIFAR-10; report test accuracy | ★★★ |

---

## Block 3 — Modern CNN Architectures (L15–L20)

File: `block3_modern_cnn.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 3.1 | Count VGG-16 parameters and verify formula | ★ |
| 3.2 | ResNet projection shortcut: 1×1 conv + BN | ★★ |
| 3.3 | FLOP ratio: DW+PW vs standard conv (K=3, C_in=64, C_out=128) | ★ |
| 3.4 | SE block with reduction r=16; test on dummy tensor | ★★ |
| 3.5 | EfficientNet compound scaling: compute α, β, γ from φ=1 | ★★ |

---

## Block 4 — Tokenization & Embeddings (L21–L23)

File: `block4_tokenization.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 4.1 | One round of BPE merges on "aaabdaaabac" | ★★ |
| 4.2 | `embed_forward` + `embed_backward` (scatter-add); verify ∂L/∂E | ★★ |
| 4.3 | RoPE `rope_apply` for d_head=4; verify rotation-equivariance of Q·K | ★★★ |

---

## Block 5 — Transformer (L24–L30)

File: `block5_transformer.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 5.1 | RMSNorm forward; verify matches LayerNorm when mean=0 | ★★ |
| 5.2 | Add causal mask to `mha_forward`; verify upper triangle ≈ 0 | ★★ |
| 5.3 | `kvcache_append` + `cached_attention_forward` for batch=1 decoder | ★★★ |
| 5.4 | SwiGLU backward: given dOut, compute dGate, dUp, dDown | ★★★ |
| 5.5 | Load GPT-2 weights; verify first 5 logits for tokens [15496,11,995,0] | ★★★★ |

---

## Block 6 — Vision Transformer (L31–L33)

File: `block6_vit.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 6.1 | Verify `patch_embed_forward` matches `conv2d_naive` (stride=P=16) | ★★ |
| 6.2 | Count ViT-Base parameters; verify ≈ 86.6M | ★★ |
| 6.3 | InfoNCE loss for batch of 4 image-text pairs | ★★★ |

---

## Block 7 — Training Recipes (L34–L38)

File: `block7_training.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 7.1 | Numerically stable log-softmax; verify gradient = softmax − one_hot | ★★ |
| 7.2 | AdamW vs Adam on toy quadratic; confirm weight decay differs | ★★ |
| 7.3 | `global_grad_clip`; verify resulting norm equals `max_norm` | ★★ |
| 7.4 | Attention backward (single head, T=4, d_head=8); compare finite diff | ★★★★ |
| 7.5 | mmap DataLoader on Shakespeare; 10 steps, report loss | ★★★ |

---

## Block 8 — Inference & Optimization (L39–L45)

File: `block8_inference.md`

| # | Exercise | Difficulty |
|---|----------|------------|
| 8.1 | Top-p sampling; verify no out-of-nucleus token ever sampled | ★★ |
| 8.2 | Absmax INT8 quantization; verify max |error| < 127/max_abs | ★★ |
| 8.3 | Online (single-pass) softmax; verify matches two-pass result | ★★★ |
| 8.4 | Parse GGUF header; print tensor names, types, shapes | ★★★ |
| 8.5 | OpenMP attention loop; measure tokens/sec for 4 vs 8 threads | ★★★ |

---

## How to Use

1. Read the lesson(s) for the block before attempting exercises.
2. Each exercise file contains: **Problem**, **Starter Code**, **Test Cases**, **Hints**, and a **Solution Approach** (no full solution — figure it out!).
3. Compile with: `gcc -std=c11 -Wall -Wextra -O2 -o ex your_file.c -lm`
4. For OpenMP exercises add `-fopenmp`.
