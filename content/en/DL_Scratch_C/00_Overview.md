# Deep Learning from Scratch in C/C++ — Study Guide

## Introduction

This folder implements **complete deep learning systems in pure C/C++** — no PyTorch, no TensorFlow, no Python. You will build every component from scratch: a tensor library with automatic differentiation, convolutional neural networks (LeNet → ResNet → EfficientNet → ViT), and full Transformer architectures (GPT-2 → Llama), culminating in a deployable LLM inference engine comparable to llama.cpp.

**Why C/C++?**
Modern LLM inference engines (llama.cpp, llm.c, whisper.cpp, ggml) are written in C/C++ because Python cannot control memory layout, SIMD vectorization, or allocation patterns precisely enough for production performance. Building in C forces you to understand *exactly* what a GPU or CPU does with your data — there is no abstraction to hide behind.

The curriculum follows a **5-level implementation philosophy**:

| Level | Description | Example |
|-------|-------------|---------|
| **L1: Naive C** | Correct but unoptimized; validates math | First matmul, naive convolution |
| **L2: Cache-Aware** | Loop tiling, memory layout, BLAS calls | Optimized SGEMM, im2col |
| **L3: SIMD** | AVX2/AVX-512 intrinsics for throughput | Vectorized inner loops |
| **L4: Systems** | Arena allocators, mmap I/O, threading | Memory-mapped data loader, OpenMP |
| **L5: Production** | GGUF loading, INT4 quant, speculative decoding | Final inference engine |

## Target Audience

- Engineers who have completed **Deep_Learning** (PyTorch-based) and want to understand what happens underneath the abstractions
- Systems programmers (**C_Advanced**, **CPP_Advanced**) entering the ML domain
- Researchers who want to implement and modify model internals without framework overhead
- Anyone who wants to understand llama.cpp, ggml, or llm.c at source-code level

## Prerequisites

| Topic | Required Level |
|-------|---------------|
| **C_Advanced** | Proficient — pointers, dynamic allocation, file I/O, `Makefile` |
| **CPP_Advanced** | Proficient — templates, RAII, operator overloading, C++17 |
| **Linear_Algebra** | Strong — matrix multiplication, broadcasting, SVD (conceptual) |
| **Deep_Learning** | Completed — backpropagation, attention, GPT/Llama architectures |
| **Computer_Architecture** | Familiar — cache hierarchy, SIMD concepts, memory bandwidth |
| Foundation_Models | Recommended — scaling, KV cache, quantization, GGUF |
| OS_Theory | Recommended — mmap, virtual memory, thread model |

## Learning Roadmap

```
┌─────────────────────┐
│  Block 1: Tensor    │  L01–L07
│  Engine + Autograd  │  C tensor lib, AVX2 matmul, autograd engine
└──────────┬──────────┘
           │
     ┌─────▼──────────────┐          ┌──────────────────────┐
     │  Block 2: CNN      │  L08–L14 │  Block 3: Modern CNN │  L15–L20
     │  Foundations       │          │  ResNet / ViT prep   │
     │  Conv2D, BN, LeNet │          │  EfficientNet        │
     └─────┬──────────────┘          └──────────┬───────────┘
           └────────────┬────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │  Block 4: Tokenization  │  L21–L23
           │  BPE, Embeddings, RoPE  │
           └────────────┬────────────┘
                        │
     ┌──────────────────▼──────────┐     ┌──────────────────────┐
     │  Block 5: Transformer       │     │  Block 6: ViT        │
     │  Forward Pass               │     │  + Multimodal        │
     │  GPT-2 / Llama / GQA / RoPE │     │  L31–L33             │
     │  L24–L30                    │     └──────────────────────┘
     └──────────────┬──────────────┘
                    │
     ┌──────────────▼──────────────┐
     │  Block 7: Training          │  L34–L38
     │  from Scratch               │
     │  AdamW, backprop, llm.c     │
     └──────────────┬──────────────┘
                    │
     ┌──────────────▼──────────────┐
     │  Block 8: Modern Inference  │  L39–L45
     │  Quantization, FlashAttn,   │
     │  Speculative, GGUF engine   │
     └─────────────────────────────┘
```

## File List

| Lesson | Filename | Difficulty | Description |
|--------|----------|------------|-------------|
| **Block 1: Tensor Engine + Autograd** |
| L01 | `01_Why_C_for_DL.md` | ⭐⭐ | Why implement DL in C/C++; llama.cpp, llm.c survey |
| L02 | `02_Memory_Layout_and_Strides.md` | ⭐⭐⭐ | Stride arithmetic, shape/view, cache-line alignment |
| L03 | `03_Tensor_Ops_BLAS.md` | ⭐⭐⭐ | Element-wise ops, reductions, naive matmul vs OpenBLAS |
| L04 | `04_Optimized_Matmul.md` | ⭐⭐⭐⭐ | Loop tiling, register blocking, AVX2 SGEMM |
| L05 | `05_Autograd_Engine.md` | ⭐⭐⭐⭐ | Computation graph, topological sort, `backward()` in C |
| L06 | `06_Autograd_Tensor_Ops.md` | ⭐⭐⭐⭐ | Matmul/softmax/cross-entropy backward, finite-diff tests |
| L07 | `07_Memory_Manager.md` | ⭐⭐⭐⭐ | Arena allocator, reference counting, zero-copy views |
| **Block 2: CNN — Foundations** |
| L08 | `08_Convolution_from_Scratch.md` | ⭐⭐⭐ | Naive conv2D, stride/padding/dilation, im2col trick |
| L09 | `09_Convolution_Backward.md` | ⭐⭐⭐⭐ | Input/filter/bias gradients, numerical verification |
| L10 | `10_Pooling_Layers.md` | ⭐⭐⭐ | Max/average/global pooling, forward/backward |
| L11 | `11_Batch_Normalization.md` | ⭐⭐⭐⭐ | BN train/eval modes, running stats, backward pass |
| L12 | `12_Data_Pipeline_Images.md` | ⭐⭐⭐ | STB image loading, NCHW/NHWC, data augmentation |
| L13 | `13_LeNet_and_AlexNet.md` | ⭐⭐⭐ | LeNet-5 + AlexNet in C, CIFAR-10 training pipeline |
| L14 | `14_Training_CNN_CIFAR10.md` | ⭐⭐⭐⭐ | End-to-end CNN training: loader → forward → loss → backward |
| **Block 3: CNN — Modern Architectures** |
| L15 | `15_VGG_and_Deep_Networks.md` | ⭐⭐⭐ | VGG-16/19, depth vs. vanishing gradient, parameter count |
| L16 | `16_ResNet_and_Skip_Connections.md` | ⭐⭐⭐⭐ | Residual block, identity/projection shortcuts, backward |
| L17 | `17_Depthwise_Separable_Conv.md` | ⭐⭐⭐ | Depthwise + pointwise, MobileNet style, FLOP comparison |
| L18 | `18_Squeeze_Excitation_and_Attention.md` | ⭐⭐⭐ | SE block (channel attention), CBAM, ViT bridge |
| L19 | `19_EfficientNet_Scaling.md` | ⭐⭐⭐⭐ | Compound scaling, NAS concept, EfficientNet-B0 |
| L20 | `20_Modern_CNN_Benchmark.md` | ⭐⭐⭐ | CIFAR-10/100: LeNet vs ResNet-20 vs EfficientNet |
| **Block 4: Tokenization & Embeddings** |
| L21 | `21_Tokenization_BPE.md` | ⭐⭐⭐ | BPE algorithm, byte-level BPE (GPT-2), tiktoken files |
| L22 | `22_Embedding_Table.md` | ⭐⭐⭐ | Lookup table, weight tying, binary weight loading |
| L23 | `23_Positional_Encodings.md` | ⭐⭐⭐ | Sinusoidal, learned PE, RoPE in real arithmetic |
| **Block 5: Transformer Forward Pass** |
| L24 | `24_Layer_Normalization.md` | ⭐⭐⭐ | LayerNorm vs RMSNorm, forward/backward, gamma/beta |
| L25 | `25_Attention_Mechanism.md` | ⭐⭐⭐⭐ | MHA: Q/K/V projections, scaled dot-product, causal mask |
| L26 | `26_KV_Cache.md` | ⭐⭐⭐⭐ | Pre-allocated KV buffer, append-only, memory analysis |
| L27 | `27_FFN_and_Activations.md` | ⭐⭐⭐ | GELU (GPT-2) vs SwiGLU (Llama): `silu(gate) * up` |
| L28 | `28_Transformer_Block.md` | ⭐⭐⭐⭐ | Pre-norm + residual + attn + FFN; vs PyTorch output |
| L29 | `29_GPT2_Forward_Pass.md` | ⭐⭐⭐⭐ | GPT-2 (124M) full forward, real weights, logit check |
| L30 | `30_Llama_Architecture.md` | ⭐⭐⭐⭐ | Llama 2/3: RMSNorm, SwiGLU, RoPE, GQA |
| **Block 6: Vision Transformer** |
| L31 | `31_Vision_Transformer_ViT.md` | ⭐⭐⭐⭐ | Patch embedding, [CLS] token, 2D PE, ViT-Base |
| L32 | `32_ViT_Training_and_Fine_Tuning.md` | ⭐⭐⭐⭐ | Warm-up + cosine LR, CutMix, ImageNet-style training |
| L33 | `33_Multimodal_CLIP_Style.md` | ⭐⭐⭐⭐ | InfoNCE loss, image+text encoders, cosine similarity |
| **Block 7: LLM Training from Scratch** |
| L34 | `34_Cross_Entropy_Loss.md` | ⭐⭐⭐ | Log-softmax + NLL, numerical stability, fused backward |
| L35 | `35_Optimizers.md` | ⭐⭐⭐ | SGD momentum, Adam, AdamW, gradient clipping, LR schedule |
| L36 | `36_Training_Loop.md` | ⭐⭐⭐⭐ | mmap data loader, mini-batch sampling, loss logging |
| L37 | `37_Backprop_Through_Transformer.md` | ⭐⭐⭐⭐⭐ | Attention backward, softmax-QK^T grad, full Transformer bp |
| L38 | `38_Training_GPT2_Small.md` | ⭐⭐⭐⭐⭐ | GPT-2 small end-to-end, llm.c reproduction, benchmarks |
| **Block 8: Modern Inference** |
| L39 | `39_Sampling_Strategies.md` | ⭐⭐⭐ | Greedy, temperature, top-k, top-p, min-p, repetition |
| L40 | `40_Quantization_Int8_Int4.md` | ⭐⭐⭐⭐ | Absmax INT8, per-channel, INT4 weight-only (GGUF style) |
| L41 | `41_FlashAttention_CPU.md` | ⭐⭐⭐⭐⭐ | FA1/FA2 tiling, IO-complexity, CPU implementation |
| L42 | `42_Speculative_Decoding.md` | ⭐⭐⭐⭐⭐ | Draft-verify loop, rejection sampling, speedup measurement |
| L43 | `43_GGUF_and_Loading.md` | ⭐⭐⭐⭐ | GGUF format parsing, Q4_K_M loading, real Llama-3 inference |
| L44 | `44_Parallel_Inference.md` | ⭐⭐⭐⭐ | OpenMP/pthreads tensor parallelism, bandwidth bottleneck |
| L45 | `45_Capstone_Inference_Engine.md` | ⭐⭐⭐⭐⭐ | Full CLI engine: GGUF + INT4 + KV cache + GQA + sampling |

**Total: 45 lessons**

## Difficulty Curve

```
Block 1 │▓▓▓▓░░░│  Upper-intermediate — C autograd is the first big wall
Block 2 │▓▓▓░░░░│  Intermediate — Conv backward is tricky but manageable
Block 3 │▓▓▓▓░░░│  Upper-intermediate — Skip connection backward
Block 4 │▓▓░░░░░│  Intermediate — Tokenization is relatively intuitive
Block 5 │▓▓▓▓░░░│  Upper-intermediate — Assembly is the challenge
Block 6 │▓▓▓▓▓░░│  Advanced — ViT patch embedding + contrastive loss
Block 7 │▓▓▓▓▓▓▓│  Expert — Full Transformer backward is peak difficulty
Block 8 │▓▓▓▓▓░░│  Advanced — Systems engineering + algorithm depth
```

**Peak Difficulty Lessons**: L05 (autograd in C), L09 (conv backward), L37 (full Transformer backprop), L41 (FlashAttention CPU), L42 (speculative decoding)

## Key Milestones

| After | You Can |
|-------|---------|
| L07 | Run a 2-layer MLP forward+backward in C with numerical verification |
| L14 | Train LeNet/AlexNet on CIFAR-10 entirely in C |
| L20 | Implement ResNet-20 and EfficientNet-B0; understand the full CNN lineage |
| L29 | Run GPT-2 (124M) with real weights, matching HuggingFace logits |
| L33 | Build a CLIP-style multimodal model where CNN meets Transformer |
| L38 | Train GPT-2 small from scratch in C (reproducing Karpathy's llm.c) |
| L45 | Load a real quantized Llama GGUF and generate text from CLI |

## Environment Setup

```bash
# macOS
xcode-select --install
brew install openblas

# Ubuntu/Debian
sudo apt-get install build-essential libopenblas-dev

# Build examples (each lesson has its own Makefile)
cd study-hub/examples/DL_Scratch_C/01_Why_C_for_DL/
make && ./hello_tensor

# Run all block examples
make -C study-hub/examples/DL_Scratch_C/
```

### Recommended Compiler Flags

```makefile
CFLAGS  = -std=c11   -O2 -march=native -Wall -Wextra
CXXFLAGS = -std=c++17 -O2 -march=native -Wall -Wextra
LIBS    = -lopenblas -lm -lpthread
```

### Optional Tools

- **Valgrind** — memory leak detection during Block 1–2
- **perf / Instruments** — CPU profiling for Block 4 (matmul optimization)
- **Python + PyTorch** — reference outputs for numerical correctness tests

## Related Topics

- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: The PyTorch-based companion course — same architectures, higher-level abstractions
- **[CUDA](../CUDA/00_Overview.md)**: GPU acceleration for the kernels built in this course (attention, GEMM)
- **[C_Advanced](../C_Advanced/00_Overview.md)**: Systems programming prerequisite
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: Scaling, quantization theory, GGUF ecosystem
- **[Computer_Architecture](../Computer_Architecture/00_Overview.md)**: Cache hierarchy, SIMD, roofline model

## Study Tips

1. **Verify numerically first**: Before optimizing any layer, write a Python/NumPy reference and compare outputs to 6+ decimal places.
2. **Build incrementally**: Each lesson's code should compile and produce correct output before moving on. Never accumulate debt.
3. **Read real source**: Study llama.cpp and llm.c alongside these lessons — the code choices will make more sense with context.
4. **Profile before optimizing**: Use `perf stat` or Instruments to confirm where time is actually spent.
5. **Memory is the bottleneck**: Almost every performance puzzle in DL inference reduces to "not enough memory bandwidth." Think in bytes/second, not FLOP/s.

## Learning Outcomes

After completing this course, you will be able to:

- ✅ Implement a tensor library with automatic differentiation in C from scratch
- ✅ Build and train CNNs (LeNet → ResNet → EfficientNet) in pure C
- ✅ Implement the complete Transformer architecture (GPT-2, Llama) with GQA and RoPE
- ✅ Train a language model from scratch in C (reproducing llm.c results)
- ✅ Apply INT8/INT4 quantization and measure perplexity degradation
- ✅ Implement FlashAttention-2 tiling logic on CPU
- ✅ Load real GGUF model files and run LLM inference from a CLI
- ✅ Read and extend llama.cpp / ggml source code with confidence

---

Start with `01_Why_C_for_DL.md` to understand the landscape, then `02_Memory_Layout_and_Strides.md` to build the foundational data model.
